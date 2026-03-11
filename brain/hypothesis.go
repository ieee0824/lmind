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
	model    string
	client   *ollama.Client
	bus      *bus.ThoughtBus
	history  *bus.History
	logger   *logger.Logger
	state    *State
	inbox    <-chan bus.Thought
	interval time.Duration
}

type HypothesisConfig struct {
	Model    string
	Client   *ollama.Client
	Bus      *bus.ThoughtBus
	History  *bus.History
	Logger   *logger.Logger
	State    *State
	Interval time.Duration
}

func NewHypothesis(cfg HypothesisConfig) *Hypothesis {
	h := &Hypothesis{
		model:    cfg.Model,
		client:   cfg.Client,
		bus:      cfg.Bus,
		history:  cfg.History,
		logger:   cfg.Logger,
		state:    cfg.State,
		interval: cfg.Interval,
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
			// 前頭葉・側頭葉・ユーザー入力に反応して仮説を更新
			if t.From == "frontal" || t.From == "temporal" || t.From == "user" {
				h.generate(ctx, t)
			}
		case <-ticker.C:
			// 定期的にも仮説を見直す
			h.review(ctx)
		}
	}
}

// generate は新しい入力を受けて仮説を生成する
func (h *Hypothesis) generate(ctx context.Context, incoming bus.Thought) {
	goal, currentHyp := h.state.Snapshot()

	recent := h.history.Recent(8)
	contextStr := formatThoughts(recent)

	var inputLabel string
	switch incoming.From {
	case "user":
		inputLabel = fmt.Sprintf("New external input: %s", incoming.Content)
	default:
		inputLabel = fmt.Sprintf("New analysis from [%s]: %s", incoming.From, incoming.Content)
	}

	prompt := fmt.Sprintf(`Goal: %s
Current hypothesis: %s

%s

Recent thought stream:
%s

Based on the goal, current hypothesis, and new input, generate or update the hypothesis.
- If the new input supports the current hypothesis, refine it
- If the new input contradicts it, propose a new hypothesis
- If there's no clear hypothesis yet, propose one
- If nothing meaningful can be hypothesized, reply "NONE"
Output ONLY the hypothesis in one sentence, or "NONE".`, goal, currentHyp, inputLabel, contextStr)

	resp, err := h.client.Chat(ctx, ollama.ChatRequest{
		Model: h.model,
		Messages: []ollama.Message{
			{Role: "system", Content: "You are a hypothesis generation module. Your role is to form and refine hypotheses based on observations and analysis. Always respond in English."},
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		h.logger.Error("hypothesis", fmt.Sprintf("仮説生成エラー: %v", err))
		return
	}

	result := strings.TrimSpace(resp.Message.Content)
	if result == "" || strings.Contains(strings.ToUpper(result), "NONE") {
		return
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

	prompt := fmt.Sprintf(`Goal: %s
Current hypothesis: %s

Recent thought stream:
%s

Review the current hypothesis against recent thoughts.
- Is it still valid? If so, reply "VALID"
- Does it need refinement? Output the refined hypothesis in one sentence
- Is it contradicted? Output a new hypothesis in one sentence
Output ONLY "VALID" or the updated hypothesis.`, goal, currentHyp, contextStr)

	resp, err := h.client.Chat(ctx, ollama.ChatRequest{
		Model: h.model,
		Messages: []ollama.Message{
			{Role: "system", Content: "You are a hypothesis review module. Evaluate and refine hypotheses. Always respond in English."},
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		h.logger.Error("hypothesis", fmt.Sprintf("仮説レビューエラー: %v", err))
		return
	}

	result := strings.TrimSpace(resp.Message.Content)
	if result == "" || strings.Contains(strings.ToUpper(result), "VALID") {
		return
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
