package brain

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
)

// Grounding は現実アンカーモジュール（LLM不使用）。
// 「私は何者か」「どこで動いているか」「何が事実か」を定期的に思考バスに注入し、
// 脳部位が三人称に逸れるのを防ぐ。
type Grounding struct {
	bus      *bus.ThoughtBus
	logger   *logger.Logger
	inbox    <-chan bus.Thought
	interval time.Duration
	hosts    []string // Ollamaホスト一覧
}

type GroundingConfig struct {
	Bus      *bus.ThoughtBus
	Logger   *logger.Logger
	Interval time.Duration
	Hosts    []string
}

func NewGrounding(cfg GroundingConfig) *Grounding {
	g := &Grounding{
		bus:      cfg.Bus,
		logger:   cfg.Logger,
		interval: cfg.Interval,
		hosts:    cfg.Hosts,
	}
	g.inbox = cfg.Bus.Subscribe("grounding")
	return g
}

// Run は現実アンカーループを開始する（混乱検知時のみ発火）
func (g *Grounding) Run(ctx context.Context) {
	g.logger.Info("grounding", "現実アンカーモジュール開始（イベント駆動）")

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-g.inbox:
			// 混乱を検知した時だけアンカー投入（定期投入はしない）
			if (t.From == "frontal" || t.From == "temporal" || t.From == "hypothesis") && g.isConfused(t.Content) {
				g.anchor()
			}
		}
	}
}

// anchor は現実アンカー（事実の束）を思考バスに投入する
func (g *Grounding) anchor() {
	facts := g.collectFacts()
	thought := bus.Thought{
		From:    "grounding",
		Content: facts,
	}
	g.bus.Publish(thought)
	g.logger.Info("grounding", "現実アンカー投入")
}

// collectFacts は現在の事実を収集してアンカーテキストを生成する（短く事実だけ）
func (g *Grounding) collectFacts() string {
	hostname, _ := os.Hostname()
	now := time.Now()

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("[fact] host=%s, os=%s/%s, time=%s",
		hostname, runtime.GOOS, runtime.GOARCH, now.Format("2006-01-02 15:04")))
	if len(g.hosts) > 0 {
		sb.WriteString(fmt.Sprintf(", ollama=%s", strings.Join(g.hosts, ";")))
	}
	return sb.String()
}

// isConfused は三人称パターンを検知する
func (g *Grounding) isConfused(content string) bool {
	lower := strings.ToLower(content)
	for _, p := range confusionPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}
