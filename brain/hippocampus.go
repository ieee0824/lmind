package brain

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	memai "github.com/ieee0824/memAI-go"
	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/ollama"
)

// Hippocampus は海馬（記憶野）。memAI-goのSTM/LTM/感情検出を統合する。
// 思考バスを流れる情報を記憶に変換し、関連記憶を想起して返す。
type Hippocampus struct {
	Name     string
	Model    string
	Interval time.Duration

	client    *ollama.Client
	bus       *bus.ThoughtBus
	history   *bus.History
	inbox     <-chan bus.Thought
	stm       *memai.STM
	ltm       *memai.LTM[int64]
	store     memai.MemoryStore[int64]
	analyzer  memai.EmotionAnalyzer
	turn      int
}

type HippocampusConfig struct {
	Model       string
	Client      *ollama.Client
	Bus         *bus.ThoughtBus
	History     *bus.History
	Store       memai.MemoryStore[int64]
	EmbeddingFn memai.EmbeddingFunc
	Interval    time.Duration
}

func NewHippocampus(cfg HippocampusConfig) *Hippocampus {
	h := &Hippocampus{
		Name:     "hippocampus",
		Model:    cfg.Model,
		Interval: cfg.Interval,
		client:   cfg.Client,
		bus:      cfg.Bus,
		history:  cfg.History,
		stm:      memai.NewSTM(memai.DefaultSTMConfig()),
		ltm:      memai.NewLTM(cfg.Store, cfg.EmbeddingFn, memai.DefaultLTMConfig()),
		store:    cfg.Store,
		analyzer: memai.NewKeywordEmotionAnalyzer(memai.LangJapanese),
	}
	h.inbox = cfg.Bus.Subscribe(h.Name)
	return h
}

// Run は記憶野の処理ループを開始する
func (h *Hippocampus) Run(ctx context.Context) {
	log.Printf("[%s] 記憶野ループ開始 (model=%s)", h.Name, h.Model)

	ticker := time.NewTicker(h.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Printf("[%s] 記憶野ループ終了", h.Name)
			return

		case thought := <-h.inbox:
			h.turn++
			h.process(ctx, thought)

		case <-ticker.C:
			// 定期的にSTMを整理
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
		log.Printf("[%s] 感情分析エラー: %v", h.Name, err)
		emotion = &memai.EmotionalState{Primary: memai.EmotionNeutral}
	}

	// STMを更新
	h.stm.Update(h.turn, content, emotion)

	// STMに新しいアイテムを追加
	keywords := extractKeywords(content)
	h.stm.Add(&memai.WorkingMemoryItem{
		Topic:       incoming.From,
		Content:     content,
		Keywords:    keywords,
		Activation:  1.0,
		TurnCreated: h.turn,
		TurnAccessed: h.turn,
		Emotional:   emotion.Intensity > 0.3,
	})

	// LTMに保存
	if err := h.saveLTM(ctx, incoming, emotion); err != nil {
		log.Printf("[%s] LTM保存エラー: %v", h.Name, err)
	}

	// LTMから関連記憶を検索
	results, err := h.ltm.Search(ctx, memai.SearchQuery{
		Query:              content,
		EmotionalIntensity: emotion.Intensity,
	})
	if err != nil {
		log.Printf("[%s] LTM検索エラー: %v", h.Name, err)
		return
	}

	// 関連記憶があれば想起として思考バスに流す
	if len(results) > 1 { // 自分自身以外の記憶がある場合
		recall := h.formatRecall(results, emotion)
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

	// 現在のSTMの状態をログ
	log.Printf("[%s] STM整理中: %d items", h.Name, len(items))

	// STMをdecayさせる（空メッセージでupdate）
	h.turn++
	h.stm.Update(h.turn, "", &memai.EmotionalState{Primary: memai.EmotionNeutral})
}

func (h *Hippocampus) saveLTM(ctx context.Context, t bus.Thought, emotion *memai.EmotionalState) error {
	mem := &memai.Memory[int64]{
		Content:            t.Content,
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

	count := 0
	for _, r := range results {
		if count >= 3 {
			break
		}
		fmt.Fprintf(&sb, "- [%s] %s (関連度: %.2f)\n", r.Memory.ThreadKey, r.Memory.Content, r.Score)
		count++
	}
	return sb.String()
}

// extractKeywords は簡易的にキーワードを抽出する
func extractKeywords(content string) []string {
	// スペースと句読点で分割した簡易実装
	words := strings.FieldsFunc(content, func(r rune) bool {
		return r == ' ' || r == '。' || r == '、' || r == '\n' || r == '　'
	})

	var keywords []string
	for _, w := range words {
		w = strings.TrimSpace(w)
		if len([]rune(w)) >= 2 { // 2文字以上のみ
			keywords = append(keywords, w)
		}
	}
	if len(keywords) > 10 {
		keywords = keywords[:10]
	}
	return keywords
}
