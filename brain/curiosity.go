package brain

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
)

// Curiosity は探索モジュール。
// criticが停滞を検知したとき、既存の思考からキーワードを再結合して
// 新しい問いを生成し、思考の多様性を維持する。
type Curiosity struct {
	bus      *bus.ThoughtBus
	history  *bus.History
	logger   *logger.Logger
	inbox    <-chan bus.Thought
	interval time.Duration
}

type CuriosityConfig struct {
	Bus      *bus.ThoughtBus
	History  *bus.History
	Logger   *logger.Logger
	Interval time.Duration
}

func NewCuriosity(cfg CuriosityConfig) *Curiosity {
	c := &Curiosity{
		bus:      cfg.Bus,
		history:  cfg.History,
		logger:   cfg.Logger,
		interval: cfg.Interval,
	}
	c.inbox = cfg.Bus.Subscribe("curiosity")
	return c
}

// Run は探索ループを開始する
func (c *Curiosity) Run(ctx context.Context) {
	c.logger.Info("curiosity", "探索モジュール開始")

	ticker := time.NewTicker(c.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-c.inbox:
			// criticからの停滞通知に反応して即座に探索
			if t.From == "critic" {
				c.explore()
			}
		case <-ticker.C:
			// 定期的にも軽く探索（新しい視点を投入）
			c.explore()
		}
	}
}

func (c *Curiosity) explore() {
	recent := c.history.Recent(20)
	if len(recent) < 3 {
		return
	}

	// 全思考からキーワードを収集し、出現頻度を数える
	wordFreq := make(map[string]int)
	for _, t := range recent {
		if t.From == "curiosity" || t.From == "critic" || t.From == "novelty" {
			continue // メタモジュールの出力は除外
		}
		for _, w := range tokenize(t.Content) {
			wordFreq[w]++
		}
	}

	if len(wordFreq) < 4 {
		return
	}

	// 頻出キーワード（主流の思考）と稀出キーワード（周辺概念）を分離
	var frequent, rare []string
	for w, count := range wordFreq {
		if count >= 3 {
			frequent = append(frequent, w)
		} else if count == 1 {
			rare = append(rare, w)
		}
	}

	// 稀出キーワード同士、または頻出+稀出を組み合わせて問いを生成
	question := c.generateQuestion(frequent, rare)
	if question == "" {
		return
	}

	thought := bus.Thought{
		From:    "curiosity",
		Content: question,
	}
	c.history.Record(thought)
	c.bus.Publish(thought)
	c.logger.Info("curiosity", fmt.Sprintf("探索的問い: %s", question))
}

// generateQuestion はキーワードの組み合わせから探索的な問いを生成する
func (c *Curiosity) generateQuestion(frequent, rare []string) string {
	templates := []string{
		"What if we connect %s with %s?",
		"How does %s relate to %s from a different angle?",
		"Is there a hidden pattern between %s and %s?",
		"What would happen if %s were applied to %s?",
		"Consider the opposite of %s — how does that change %s?",
	}

	var word1, word2 string

	switch {
	case len(rare) >= 2:
		// 稀出キーワード同士（最も新しい組み合わせ）
		idx := rand.Intn(len(rare))
		word1 = rare[idx]
		// 別のキーワードを選ぶ
		for i := 0; i < 10; i++ {
			w := rare[rand.Intn(len(rare))]
			if w != word1 {
				word2 = w
				break
			}
		}
	case len(frequent) > 0 && len(rare) > 0:
		// 頻出 + 稀出（主流と周辺の接続）
		word1 = frequent[rand.Intn(len(frequent))]
		word2 = rare[rand.Intn(len(rare))]
	case len(frequent) >= 2:
		// 頻出同士（視点の転換）
		idx := rand.Intn(len(frequent))
		word1 = frequent[idx]
		for i := 0; i < 10; i++ {
			w := frequent[rand.Intn(len(frequent))]
			if w != word1 {
				word2 = w
				break
			}
		}
	}

	if word1 == "" || word2 == "" {
		return ""
	}

	tmpl := templates[rand.Intn(len(templates))]
	return fmt.Sprintf(tmpl, word1, word2)
}

// shuffleStrings はスライスをシャッフルする
func shuffleStrings(s []string) {
	for i := len(s) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		s[i], s[j] = s[j], s[i]
	}
}

// uniqueWords は重複を除いたユニークな単語リストを返す
func uniqueWords(words []string) []string {
	seen := make(map[string]bool)
	var result []string
	for _, w := range words {
		lower := strings.ToLower(w)
		if !seen[lower] {
			seen[lower] = true
			result = append(result, w)
		}
	}
	return result
}
