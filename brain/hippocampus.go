package brain

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	memai "github.com/ieee0824/memAI-go"
	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/config"
	"github.com/ieee0824/lmind/logger"
	"github.com/ieee0824/lmind/ollama"
)

// Hippocampus は海馬（記憶野）。memAI-goのSTM/LTM/感情検出を統合する。
// 思考バスを流れる情報を記憶に変換し、関連記憶を想起して返す。
type Hippocampus struct {
	Name     string
	Model    string
	Interval time.Duration

	client   *ollama.Client
	bus      *bus.ThoughtBus
	history  *bus.History
	logger   *logger.Logger
	inbox    <-chan bus.Thought
	stm       *memai.STM
	ltm       *memai.LTM[int64]
	store     memai.MemoryStore[int64]
	embedding memai.EmbeddingFunc
	analyzer  memai.EmotionAnalyzer
	ltmParams config.LTMParams
	turn      int
}

type HippocampusConfig struct {
	Model       string
	Client      *ollama.Client
	Bus         *bus.ThoughtBus
	History     *bus.History
	Logger      *logger.Logger
	Store       memai.MemoryStore[int64]
	EmbeddingFn memai.EmbeddingFunc
	Interval    time.Duration
	LTMParams   config.LTMParams
}

func NewHippocampus(cfg HippocampusConfig) *Hippocampus {
	h := &Hippocampus{
		Name:     "hippocampus",
		Model:    cfg.Model,
		Interval: cfg.Interval,
		client:   cfg.Client,
		bus:      cfg.Bus,
		history:  cfg.History,
		logger:   cfg.Logger,
		stm:       memai.NewSTM(memai.DefaultSTMConfig()),
		ltm:       memai.NewLTM(cfg.Store, cfg.EmbeddingFn, memai.DefaultLTMConfig()),
		store:     cfg.Store,
		embedding: cfg.EmbeddingFn,
		analyzer:  memai.NewKeywordEmotionAnalyzer(memai.LangJapanese),
		ltmParams: cfg.LTMParams,
	}
	h.inbox = cfg.Bus.Subscribe(h.Name)
	return h
}

// Run は記憶野の処理ループを開始する
func (h *Hippocampus) Run(ctx context.Context) {
	h.logger.Info(h.Name, fmt.Sprintf("記憶野ループ開始 (model=%s)", h.Model))

	ticker := time.NewTicker(h.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			h.logger.Info(h.Name, "記憶野ループ終了")
			return

		case thought := <-h.inbox:
			h.logger.Info(h.Name, "記憶処理中...")
			h.turn++
			h.process(ctx, thought)

		case <-ticker.C:
			h.consolidate(ctx)
		}
	}
}

// process は受信した思考を記憶に取り込み、関連記憶を想起する
func (h *Hippocampus) process(ctx context.Context, incoming bus.Thought) {
	content := incoming.Content

	// 感情分析
	emotion, err := h.analyzer.Analyze(ctx, content)
	if err != nil {
		h.logger.Error(h.Name, fmt.Sprintf("感情分析エラー: %v", err))
		emotion = &memai.EmotionalState{Primary: memai.EmotionNeutral}
	}

	// STMを更新
	h.stm.Update(h.turn, content, emotion)

	// STMに新しいアイテムを追加
	keywords := extractKeywords(content)
	h.stm.Add(&memai.WorkingMemoryItem{
		Topic:        incoming.From,
		Content:      content,
		Keywords:     keywords,
		Activation:   1.0,
		TurnCreated:  h.turn,
		TurnAccessed: h.turn,
		Emotional:    emotion.Intensity > 0.3,
	})

	// LTM処理を完全非同期化（embedding計算・保存・検索・想起発行をバックグラウンドで行う）
	go h.processLTM(ctx, incoming, content, emotion)
}

// processLTM はembedding計算・LTM保存・検索・想起発行をバックグラウンドで行う
func (h *Hippocampus) processLTM(ctx context.Context, incoming bus.Thought, content string, emotion *memai.EmotionalState) {
	emb, err := h.embedding(ctx, content)
	if err != nil {
		h.logger.Error(h.Name, fmt.Sprintf("LTM embedding生成エラー: %v", err))
		return
	}

	// 保存
	if err := h.saveLTM(ctx, incoming, emotion, emb); err != nil {
		h.logger.Error(h.Name, fmt.Sprintf("LTM保存エラー: %v", err))
	}

	// 検索
	results, err := h.ltm.Search(ctx, memai.SearchQuery{
		Query:              content,
		QueryEmbedding:     emb,
		EmotionalIntensity: emotion.Intensity,
	})
	if err != nil {
		h.logger.Error(h.Name, fmt.Sprintf("LTM検索エラー: %v", err))
		return
	}
	h.logger.Info(h.Name, fmt.Sprintf("LTM検索: %d件ヒット (保存済み: %d件)", len(results), h.memoryCount(ctx)))

	// RecallWeightによる確率的フィルタ（0なら想起しない、1なら常に想起）
	if h.ltmParams.RecallWeight <= 0 || rand.Float64() > h.ltmParams.RecallWeight {
		return
	}

	// RecallMinScoreでフィルタ
	var filtered []memai.SearchResult[int64]
	for _, r := range results {
		if r.Score >= h.ltmParams.RecallMinScore {
			filtered = append(filtered, r)
		}
	}

	// RecallMaxResultsで件数制限
	maxResults := h.ltmParams.RecallMaxResults
	if maxResults <= 0 {
		maxResults = 3
	}
	if len(filtered) > maxResults {
		filtered = filtered[:maxResults]
	}

	// 関連記憶があれば想起として思考バスに流す
	if len(filtered) > 1 {
		recall := h.formatRecall(filtered, emotion)
		if recall != "" {
			thought := bus.Thought{
				From:    h.Name,
				Content: recall,
			}
			h.history.Record(thought)
			h.bus.Publish(thought)
		}
	}
}

// consolidate はSTMの内容を定期的にレビューし、重要なものをLTMに固定する
func (h *Hippocampus) consolidate(ctx context.Context) {
	items := h.stm.Items()
	if len(items) == 0 {
		return
	}

	h.logger.Info(h.Name, fmt.Sprintf("STM整理中: %d items", len(items)))

	// STMをdecayさせる（空メッセージでupdate）
	h.turn++
	h.stm.Update(h.turn, "", &memai.EmotionalState{Primary: memai.EmotionNeutral})
}

func (h *Hippocampus) memoryCount(ctx context.Context) int {
	mems, err := h.store.GetMemories(ctx)
	if err != nil {
		return -1
	}
	return len(mems)
}

func (h *Hippocampus) saveLTM(ctx context.Context, t bus.Thought, emotion *memai.EmotionalState, emb []float64) error {
	mem := &memai.Memory[int64]{
		Content:            t.Content,
		Embedding:          emb,
		ThreadKey:          t.From,
		EventDate:          time.Now().Format("2006-01-02"),
		EmotionalIntensity: emotion.Intensity,
	}
	return h.store.SaveMemory(ctx, mem)
}

func (h *Hippocampus) formatRecall(results []memai.SearchResult[int64], emotion *memai.EmotionalState) string {
	var sb strings.Builder
	sb.WriteString("【記憶想起】")
	if emotion.Intensity > 0.3 {
		sb.WriteString(fmt.Sprintf("（感情: %s, 強度: %.1f）", emotion.Primary, emotion.Intensity))
	}
	sb.WriteString("\n")

	for _, r := range results {
		fmt.Fprintf(&sb, "- [%s] %s (関連度: %.2f)\n", r.Memory.ThreadKey, r.Memory.Content, r.Score)
	}
	return sb.String()
}

// extractKeywords は簡易的にキーワードを抽出する
func extractKeywords(content string) []string {
	words := strings.FieldsFunc(content, func(r rune) bool {
		return r == ' ' || r == '。' || r == '、' || r == '\n' || r == '　'
	})

	var keywords []string
	for _, w := range words {
		w = strings.TrimSpace(w)
		if len([]rune(w)) >= 2 {
			keywords = append(keywords, w)
		}
	}
	if len(keywords) > 10 {
		keywords = keywords[:10]
	}
	return keywords
}
