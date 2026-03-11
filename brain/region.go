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

// Region は脳の一部位を表す
type Region struct {
	Name         string
	Model        string
	SystemPrompt string
	Interval     time.Duration

	client  *ollama.Client
	bus     *bus.ThoughtBus
	history *bus.History
	logger  *logger.Logger
	inbox   <-chan bus.Thought
}

type RegionConfig struct {
	Name         string
	Model        string
	SystemPrompt string
	Interval     time.Duration
	Client       *ollama.Client
	Bus          *bus.ThoughtBus
	History      *bus.History
	Logger       *logger.Logger
}

func NewRegion(cfg RegionConfig) *Region {
	r := &Region{
		Name:         cfg.Name,
		Model:        cfg.Model,
		SystemPrompt: cfg.SystemPrompt,
		Interval:     cfg.Interval,
		client:       cfg.Client,
		bus:          cfg.Bus,
		history:      cfg.History,
		logger:       cfg.Logger,
	}
	r.inbox = cfg.Bus.Subscribe(cfg.Name)
	return r
}

// Run は思考ループを開始する。contextがキャンセルされるまで動き続ける。
func (r *Region) Run(ctx context.Context) {
	r.logger.Info(r.Name, fmt.Sprintf("思考ループ開始 (model=%s, interval=%s)", r.Model, r.Interval))

	ticker := time.NewTicker(r.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			r.logger.Info(r.Name, "思考ループ終了")
			return

		case thought := <-r.inbox:
			r.logger.Info(r.Name, "思考中...")
			r.process(ctx, thought)

		case <-ticker.C:
			r.logger.Info(r.Name, "自律思考中...")
			r.think(ctx)
		}
	}
}

// process は他部位からの思考に対して応答する
func (r *Region) process(ctx context.Context, incoming bus.Thought) {
	recent := r.history.Recent(10)
	contextStr := formatThoughts(recent)

	prompt := fmt.Sprintf("他の脳部位「%s」からの入力:\n%s\n\n最近の思考の流れ:\n%s\n\nこの入力を踏まえて、あなたの役割の観点から考えを述べてください。簡潔に。",
		incoming.From, incoming.Content, contextStr)

	resp, err := r.client.Chat(ctx, ollama.ChatRequest{
		Model: r.Model,
		Messages: []ollama.Message{
			{Role: "system", Content: r.SystemPrompt},
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		r.logger.Error(r.Name, fmt.Sprintf("エラー: %v", err))
		r.bus.Publish(bus.Thought{From: r.Name, Content: fmt.Sprintf("【エラー】%v", err)})
		return
	}

	thought := bus.Thought{
		From:    r.Name,
		Content: resp.Message.Content,
	}
	r.history.Record(thought)
	r.bus.Publish(thought)
}

// think は自律的な思考を行う
func (r *Region) think(ctx context.Context) {
	recent := r.history.Recent(10)
	if len(recent) == 0 {
		return
	}

	contextStr := formatThoughts(recent)

	prompt := fmt.Sprintf("最近の思考の流れ:\n%s\n\nこれらを踏まえて、あなたの役割の観点から新しい考えや気づきがあれば述べてください。なければ「特になし」と答えてください。簡潔に。",
		contextStr)

	resp, err := r.client.Chat(ctx, ollama.ChatRequest{
		Model: r.Model,
		Messages: []ollama.Message{
			{Role: "system", Content: r.SystemPrompt},
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		r.logger.Error(r.Name, fmt.Sprintf("自律思考エラー: %v", err))
		return
	}

	content := resp.Message.Content
	if strings.Contains(content, "特になし") {
		return
	}

	thought := bus.Thought{
		From:    r.Name,
		Content: content,
	}
	r.history.Record(thought)
	r.bus.Publish(thought)
}

func formatThoughts(thoughts []bus.Thought) string {
	if len(thoughts) == 0 {
		return "(なし)"
	}
	var sb strings.Builder
	for _, t := range thoughts {
		fmt.Fprintf(&sb, "[%s] %s\n", t.From, t.Content)
	}
	return sb.String()
}
