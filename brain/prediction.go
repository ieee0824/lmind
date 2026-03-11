package brain

import (
	"context"
	"fmt"
	"strings"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
	"github.com/ieee0824/lmind/ollama"
)

// Prediction はユーザーの次の発言を予測するモジュール。
// 会話の流れ・現在のgoal・直近の思考から、ユーザーが次に何を言うかを推測する。
// 予測はStateに格納され、他の脳部位が先回りして思考するのに使われる。
type Prediction struct {
	model   string
	client  *ollama.Client
	bus     *bus.ThoughtBus
	history *bus.History
	logger  *logger.Logger
	state   *State
	inbox   <-chan bus.Thought
}

type PredictionConfig struct {
	Model   string
	Client  *ollama.Client
	Bus     *bus.ThoughtBus
	History *bus.History
	Logger  *logger.Logger
	State   *State
}

func NewPrediction(cfg PredictionConfig) *Prediction {
	p := &Prediction{
		model:   cfg.Model,
		client:  cfg.Client,
		bus:     cfg.Bus,
		history: cfg.History,
		logger:  cfg.Logger,
		state:   cfg.State,
	}
	p.inbox = cfg.Bus.Subscribe("prediction")
	return p
}

// Run は予測ループを開始する
func (p *Prediction) Run(ctx context.Context) {
	p.logger.Info("prediction", "予測モジュール開始")

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-p.inbox:
			// Brocaの応答後（＝会話の1ターン完了後）に次の発言を予測
			if t.From == "broca" {
				p.predict(ctx)
			}
		}
	}
}

func (p *Prediction) predict(ctx context.Context) {
	goal, _ := p.state.Snapshot()

	// 会話の流れを取得（user/broca のみ抽出）
	recent := p.history.Recent(15)
	var convLines []string
	for _, t := range recent {
		switch t.From {
		case "user":
			convLines = append(convLines, fmt.Sprintf("User: %s", t.Content))
		case "broca":
			content := t.Content
			// "I said to the user: " プレフィックスを除去
			if after, ok := strings.CutPrefix(content, "I said to the user: "); ok {
				content = after
			}
			convLines = append(convLines, fmt.Sprintf("Bot: %s", content))
		}
	}

	if len(convLines) < 2 {
		return
	}

	// 直近5ターンまで
	if len(convLines) > 10 {
		convLines = convLines[len(convLines)-10:]
	}

	convStr := strings.Join(convLines, "\n")

	prompt := fmt.Sprintf(`Current goal: %s

Recent conversation:
%s

Based on the conversation flow and current goal, predict what the user will say next.
Consider:
- The topic and direction of conversation
- Unanswered questions or unresolved topics
- Natural follow-up patterns

Output ONLY the predicted next user message in one sentence. If unpredictable, reply "NONE".`, goal, convStr)

	resp, err := p.client.Chat(ctx, ollama.ChatRequest{
		Model: p.model,
		Messages: []ollama.Message{
			{Role: "system", Content: "You are a prediction module. Predict what the user will say next based on conversation context. Always respond in the same language the user has been using."},
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		p.logger.Error("prediction", fmt.Sprintf("予測エラー: %v", err))
		return
	}

	result := strings.TrimSpace(resp.Message.Content)
	if result == "" || strings.Contains(strings.ToUpper(result), "NONE") {
		p.state.SetPrediction("")
		return
	}

	p.state.SetPrediction(result)
	p.logger.Info("prediction", fmt.Sprintf("次の発言予測: %s", result))
}
