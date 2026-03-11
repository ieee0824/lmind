package brain

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
)

// Identity は自己認識モジュール（LLM不使用・アルゴリズム駆動）。
// 脳部位の思考が自分自身を外部のモデルとして語っていないか監視し、
// 自己言及の混乱を検知したら矯正メッセージを投入する。
// レート制限: 最短30秒に1回のみ発火。
type Identity struct {
	bus      *bus.ThoughtBus
	logger   *logger.Logger
	inbox    <-chan bus.Thought
	lastFire time.Time
}

type IdentityConfig struct {
	Bus    *bus.ThoughtBus
	Logger *logger.Logger
}

func NewIdentity(cfg IdentityConfig) *Identity {
	id := &Identity{
		bus:    cfg.Bus,
		logger: cfg.Logger,
	}
	id.inbox = cfg.Bus.Subscribe("identity")
	return id
}

// 自己を外部視点で語っていることを示すキーワード
var confusionPatterns = []string{
	"the model",
	"the model's",
	"this model",
	"gemma3",
	"gemma",
	"the llm",
	"the language model",
	"the ai",
	"the system's",
	"the bot",
	"it is generating",
	"it is processing",
	"the algorithm",
	"the neural network",
}

// 矯正メッセージのテンプレート
var corrections = []string{
	"Reminder: These are MY thoughts, not observations about an external model. I should think in first person.",
	"Self-check: I'm analyzing myself as if I'm external. I need to own these thoughts — they are mine.",
	"Identity correction: I am the one thinking, not watching something else think. First person perspective.",
}

var correctionIdx int

// Run は自己認識監視ループを開始する
func (id *Identity) Run(ctx context.Context) {
	id.logger.Info("identity", "自己認識モジュール開始")

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-id.inbox:
			// 脳部位（frontal/temporal/hypothesis）の思考のみチェック
			if t.From == "frontal" || t.From == "temporal" || t.From == "hypothesis" {
				if confused := id.detectConfusion(t.Content); confused != "" {
					// レート制限: 30秒に1回まで
					if time.Since(id.lastFire) >= 30*time.Second {
						id.correct(confused)
						id.lastFire = time.Now()
					}
				}
			}
		}
	}
}

// detectConfusion は思考内容に自己言及の混乱があるか検出する
func (id *Identity) detectConfusion(content string) string {
	lower := strings.ToLower(content)
	for _, pattern := range confusionPatterns {
		if strings.Contains(lower, pattern) {
			return pattern
		}
	}
	return ""
}

// correct は矯正メッセージを思考バスに投入する
func (id *Identity) correct(trigger string) {
	msg := corrections[correctionIdx%len(corrections)]
	correctionIdx++

	thought := bus.Thought{
		From:    "identity",
		Content: msg,
	}
	id.bus.Publish(thought)
	id.logger.Info("identity", fmt.Sprintf("自己認識矯正（トリガー: %q）: %s", trigger, msg))
}
