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
	state   *State
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
	State        *State
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
		state:        cfg.State,
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

	var inputLabel string
	switch incoming.From {
	case "user":
		inputLabel = fmt.Sprintf("External input (someone spoke to us):\n%s", incoming.Content)
	case "broca":
		inputLabel = fmt.Sprintf("Our speech output (what we said in response):\n%s", incoming.Content)
	default:
		inputLabel = fmt.Sprintf("Input from brain region [%s]:\n%s", incoming.From, incoming.Content)
	}

	stateStr := ""
	if r.state != nil {
		stateStr = r.state.Format() + "\n\n"
	}

	prompt := fmt.Sprintf("%s%s\n\nRecent thought stream:\n%s\n\nBased on this input and current state, provide your analysis from your role's perspective. Be concise.",
		stateStr, inputLabel, contextStr)

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

	stateStr := ""
	if r.state != nil {
		stateStr = r.state.Format() + "\n\n"
	}

	prompt := fmt.Sprintf("%sRecent thought stream:\n%s\n\nBased on current state and these thoughts, share any new insights or observations from your role's perspective. If none, reply \"nothing\". Be concise.",
		stateStr, contextStr)

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
	if strings.Contains(strings.ToLower(content), "nothing") {
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
		content := t.Content
		if t.Summary != "" {
			content = t.Summary
		}
		switch t.From {
		case "user":
			fmt.Fprintf(&sb, "[external] %s\n", content)
		case "broca":
			fmt.Fprintf(&sb, "[speech] %s\n", content)
		default:
			fmt.Fprintf(&sb, "[%s] %s\n", t.From, content)
		}
	}
	return sb.String()
}
