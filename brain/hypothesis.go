package brain

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
	"github.com/ieee0824/lmind/ollama"
)

// Hypothesis は仮説生成モジュール。
// 前頭葉（構造分析）と側頭葉（状況理解）の出力を受けて、
// 現在のgoalに対する仮説を生成・更新する。
type Hypothesis struct {
	model      string
	client     *ollama.Client
	bus        *bus.ThoughtBus
	history    *bus.History
	logger     *logger.Logger
	state      *State
	modulation *Modulation
	inbox      <-chan bus.Thought
	interval   time.Duration
}

type HypothesisConfig struct {
	Model      string
	Client     *ollama.Client
	Bus        *bus.ThoughtBus
	History    *bus.History
	Logger     *logger.Logger
	State      *State
	Modulation *Modulation
	Interval   time.Duration
}

func NewHypothesis(cfg HypothesisConfig) *Hypothesis {
	h := &Hypothesis{
		model:      cfg.Model,
		client:     cfg.Client,
		bus:        cfg.Bus,
		history:    cfg.History,
		logger:     cfg.Logger,
		state:      cfg.State,
		modulation: cfg.Modulation,
		interval:   cfg.Interval,
	}
	h.inbox = cfg.Bus.Subscribe("hypothesis")
	return h
}

// Run は仮説生成ループを開始する
func (h *Hypothesis) Run(ctx context.Context) {
	h.logger.Info("hypothesis", "仮説生成モジュール開始")

	ticker := time.NewTicker(h.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-h.inbox:
			// 前頭葉・側頭葉の分析結果にのみ反応（ユーザー入力では直接発火しない）
			// 順序: input → temporal/frontal(解釈) → hypothesis(仮説)
			if t.From == "frontal" || t.From == "temporal" {
				h.generate(ctx, t)
			}
		case <-ticker.C:
			if h.modulation != nil && h.modulation.ShouldSkip("hypothesis") {
				h.logger.Info("hypothesis", fmt.Sprintf("ゲイン低下によりスキップ (gain=%.1f)", h.modulation.Gain("hypothesis")))
				continue
			}
			// 定期的にも仮説を見直す
			h.review(ctx)
		}
	}
}

// generate は新しい入力を受けて仮説を生成する
func (h *Hypothesis) generate(ctx context.Context, incoming bus.Thought) {
	goal, currentHyp := h.state.Snapshot()

	// センチメントゲーティング: ポジティブ文脈でanxiety系仮説を生成しない
	sentiment := h.state.Sentiment()
	if sentiment > 0.3 && currentHyp != "" {
		lower := strings.ToLower(currentHyp)
		for _, kw := range []string{"anxiety", "distress", "instability", "fear", "worry", "crisis", "turmoil"} {
			if strings.Contains(lower, kw) {
				h.logger.Info("hypothesis", "ポジティブ文脈のためanxiety仮説をスキップ")
				return
			}
		}
	}

	recent := h.history.Recent(8)
	contextStr := formatThoughts(recent)

	inputLabel := fmt.Sprintf("New analysis from [%s]: %s", incoming.From, incoming.Content)

	sentimentCtx := ""
	if sentiment > 0.3 {
		sentimentCtx = "\nIMPORTANT: The user's current sentiment is POSITIVE. Do not generate anxiety/distress hypotheses."
	} else if sentiment < -0.3 {
		sentimentCtx = "\nNote: The user's current sentiment is negative."
	}

	prompt := fmt.Sprintf(`Goal: %s
Current hypothesis: %s

%s

Recent thought stream:
%s
%s
Based on the goal, current hypothesis, and new input, generate or update the hypothesis.
- If the new input supports the current hypothesis, refine it
- If the new input contradicts it, propose a new hypothesis
- If there's no clear hypothesis yet, propose one
- If nothing meaningful can be hypothesized, reply "NONE"
Output ONLY the hypothesis in one short sentence (max 30 words), or "NONE".`, goal, currentHyp, inputLabel, contextStr, sentimentCtx)

	resp, err := h.client.Chat(ctx, ollama.ChatRequest{
		Model: h.model,
		Messages: []ollama.Message{
			{Role: "system", Content: "You are a hypothesis generation module. Form hypotheses about what the USER wants, means, or is interested in. Do NOT hypothesize about your own behavior or how 'the AI' works. Focus on the user's intent, topic, or situation. Always respond in English."},
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		h.logger.Error("hypothesis", fmt.Sprintf("仮説生成エラー: %v", err))
		return
	}

	result := rewriteFirstPerson(truncateToSentences(strings.TrimSpace(resp.Message.Content), 2))
	if result == "" || strings.Contains(strings.ToUpper(result), "NONE") {
		return
	}

	// 直近の仮説と類似なら捨てる（堂々巡り防止）
	_, currentHyp2 := h.state.Snapshot()
	if currentHyp2 != "" {
		sim := jaccardSimilarity(tokenize(result), tokenize(currentHyp2))
		if sim > 0.6 {
			return
		}
	}

	// stateのhypothesisを更新
	h.state.SetHypothesis(result)

	// 思考バスにも流す（他の脳部位が参照できるように）
	thought := bus.Thought{
		From:    "hypothesis",
		Content: fmt.Sprintf("Hypothesis: %s", result),
	}
	h.history.Record(thought)
	h.bus.Publish(thought)
	h.logger.Info("hypothesis", fmt.Sprintf("仮説更新: %s", result))
}

// review は定期的に現在の仮説を見直す
func (h *Hypothesis) review(ctx context.Context) {
	goal, currentHyp := h.state.Snapshot()
	if currentHyp == "" {
		return
	}

	recent := h.history.Recent(10)
	if len(recent) < 3 {
		return
	}

	contextStr := formatThoughts(recent)

	sentimentCtx := ""
	sentiment := h.state.Sentiment()
	if sentiment > 0.3 {
		sentimentCtx = "\nIMPORTANT: The user's current sentiment is POSITIVE. If the hypothesis contains anxiety/distress themes, it is likely outdated."
	}

	prompt := fmt.Sprintf(`Goal: %s
Current hypothesis: %s

Recent thought stream:
%s
%s
Review the current hypothesis against recent thoughts.
- Is it still valid? If so, reply "VALID"
- Does it need refinement? Output the refined hypothesis in one sentence
- Is it contradicted? Output a new hypothesis in one sentence
Output ONLY "VALID" or the updated hypothesis.`, goal, currentHyp, contextStr, sentimentCtx)

	resp, err := h.client.Chat(ctx, ollama.ChatRequest{
		Model: h.model,
		Messages: []ollama.Message{
			{Role: "system", Content: "You are a hypothesis review module. Evaluate hypotheses about the user's intent or topic. Do NOT hypothesize about your own behavior. Always respond in English."},
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		h.logger.Error("hypothesis", fmt.Sprintf("仮説レビューエラー: %v", err))
		return
	}

	result := rewriteFirstPerson(truncateToSentences(strings.TrimSpace(resp.Message.Content), 2))
	if result == "" || strings.Contains(strings.ToUpper(result), "VALID") {
		return
	}

	// 直近の仮説と類似なら捨てる
	_, prevHyp := h.state.Snapshot()
	if prevHyp != "" {
		sim := jaccardSimilarity(tokenize(result), tokenize(prevHyp))
		if sim > 0.6 {
			return
		}
	}

	h.state.SetHypothesis(result)

	thought := bus.Thought{
		From:    "hypothesis",
		Content: fmt.Sprintf("Revised hypothesis: %s", result),
	}
	h.history.Record(thought)
	h.bus.Publish(thought)
	h.logger.Info("hypothesis", fmt.Sprintf("仮説改訂: %s", result))
}
