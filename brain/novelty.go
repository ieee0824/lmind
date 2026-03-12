package brain

import (
	"context"
	"fmt"
	"strings"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
)

// Novelty は新規性検出モジュール。
// 新しい思考が直近の思考とどれだけ異なるかをキーワード重複率で判定し、
// 新規性が高い場合にバスに通知する。
type Novelty struct {
	bus            *bus.ThoughtBus
	history        *bus.History
	logger         *logger.Logger
	inbox          <-chan bus.Thought
	scoreThreshold float64
}

type NoveltyConfig struct {
	Bus            *bus.ThoughtBus
	History        *bus.History
	Logger         *logger.Logger
	ScoreThreshold float64
}

func NewNovelty(cfg NoveltyConfig) *Novelty {
	n := &Novelty{
		bus:            cfg.Bus,
		history:        cfg.History,
		logger:         cfg.Logger,
		scoreThreshold: orDefault(cfg.ScoreThreshold, 0.85),
	}
	n.inbox = cfg.Bus.Subscribe("novelty")
	return n
}

// Run は新規性検出ループを開始する
func (n *Novelty) Run(ctx context.Context) {
	n.logger.Info("novelty", "新規性検出モジュール開始")

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-n.inbox:
			// 外部入力（user/broca）のみ評価。内部モジュール出力は無視。
			if t.From == "user" || t.From == "broca" {
				n.evaluate(t)
			}
		}
	}
}

func (n *Novelty) evaluate(incoming bus.Thought) {
	recent := n.history.Recent(10)
	if len(recent) < 3 {
		return // 履歴が少なすぎると判定不能
	}

	incomingWords := tokenize(incoming.Content)
	if len(incomingWords) == 0 {
		return
	}

	// 直近の思考からキーワードセットを構築
	recentWords := make(map[string]bool)
	for _, t := range recent {
		if t.From == incoming.From && t.Content == incoming.Content {
			continue
		}
		for _, w := range tokenize(t.Content) {
			recentWords[w] = true
		}
	}

	if len(recentWords) == 0 {
		return
	}

	// 重複率を計算（低い = 新規性が高い）
	overlap := 0
	for _, w := range incomingWords {
		if recentWords[w] {
			overlap++
		}
	}
	overlapRate := float64(overlap) / float64(len(incomingWords))
	noveltyScore := 1.0 - overlapRate

	// 新規性が高い場合のみ通知（閾値厳しめ: user入力でも安易に1.00にしない）
	if noveltyScore > n.scoreThreshold {
		thought := bus.Thought{
			From:    "novelty",
			Content: fmt.Sprintf("Novel input detected (score: %.2f) from [%s]. New concepts worth exploring.", noveltyScore, incoming.From),
		}
		n.history.Record(thought)
		n.bus.Publish(thought)
		n.logger.Info("novelty", fmt.Sprintf("新規性検出: %.2f from %s", noveltyScore, incoming.From))
	}
}

// truncateToSentences はテキストを最大n文に切り詰める
func truncateToSentences(text string, n int) string {
	separators := []string{". ", "。", "! ", "? "}
	count := 0
	for i := 0; i < len(text); i++ {
		for _, sep := range separators {
			if i+len(sep) <= len(text) && text[i:i+len(sep)] == sep {
				count++
				if count >= n {
					return strings.TrimSpace(text[:i+len(sep)])
				}
			}
		}
	}
	return text
}

// tokenize は簡易トークナイザ。英語と日本語の両方に対応。
func tokenize(text string) []string {
	text = strings.ToLower(text)
	words := strings.FieldsFunc(text, func(r rune) bool {
		return r == ' ' || r == '。' || r == '、' || r == '\n' || r == '　' ||
			r == '.' || r == ',' || r == ':' || r == ';' || r == '!' || r == '?' ||
			r == '(' || r == ')' || r == '[' || r == ']' || r == '"' || r == '\''
	})

	var result []string
	for _, w := range words {
		w = strings.TrimSpace(w)
		if len([]rune(w)) >= 2 {
			result = append(result, w)
		}
	}
	return result
}
