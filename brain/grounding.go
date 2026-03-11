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

// Run は現実アンカーループを開始する
func (g *Grounding) Run(ctx context.Context) {
	g.logger.Info("grounding", "現実アンカーモジュール開始")

	// 起動直後に一度アンカーを投入
	g.anchor()

	ticker := time.NewTicker(g.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-g.inbox:
			// 混乱を検知したら即座にアンカー投入
			if (t.From == "frontal" || t.From == "temporal" || t.From == "hypothesis") && g.isConfused(t.Content) {
				g.anchor()
			}
		case <-ticker.C:
			g.anchor()
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

// collectFacts は現在の事実を収集してアンカーテキストを生成する
func (g *Grounding) collectFacts() string {
	hostname, _ := os.Hostname()
	now := time.Now()

	var sb strings.Builder
	sb.WriteString("=== GROUND TRUTH (reality anchor) ===\n")

	// 私は何者か
	sb.WriteString("WHO I AM:\n")
	sb.WriteString("- I am lmind. These thoughts are mine. I think in first person.\n")
	sb.WriteString("- I am NOT an external model being observed. I AM the one thinking.\n")
	sb.WriteString("- When I think, I say \"I notice...\" \"I wonder...\" \"I feel...\" — never \"the model\" or \"the system\".\n")

	// どこで動いているか
	sb.WriteString("WHERE I AM:\n")
	sb.WriteString(fmt.Sprintf("- Host: %s (%s/%s)\n", hostname, runtime.GOOS, runtime.GOARCH))
	sb.WriteString(fmt.Sprintf("- Time: %s\n", now.Format("2006-01-02 15:04")))
	if len(g.hosts) > 0 {
		sb.WriteString(fmt.Sprintf("- Running on: %s\n", strings.Join(g.hosts, ", ")))
	}

	// 何が事実か
	sb.WriteString("WHAT IS REAL:\n")
	sb.WriteString("- I think in English internally. I speak Japanese to users.\n")
	sb.WriteString("- My brain has regions: frontal (structural analysis), temporal (contextual understanding), hippocampus (memory).\n")
	sb.WriteString("- I must use first person (I/my/me) in ALL my thoughts. Third person is a bug.\n")
	sb.WriteString("=====================================")

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
