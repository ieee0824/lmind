package brain

import (
	"context"
	"fmt"
	"time"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
)

// Critic は自己評価モジュール。
// 思考の繰り返し・停滞を検出し、思考の質をモニタリングする。
type Critic struct {
	bus                 *bus.ThoughtBus
	history             *bus.History
	logger              *logger.Logger
	inbox               <-chan bus.Thought
	interval            time.Duration
	repetitionJaccard   float64
	stagnationRepRate   float64
	stagnationDominance float64
}

type CriticConfig struct {
	Bus                 *bus.ThoughtBus
	History             *bus.History
	Logger              *logger.Logger
	Interval            time.Duration
	RepetitionJaccard   float64
	StagnationRepRate   float64
	StagnationDominance float64
}

func NewCritic(cfg CriticConfig) *Critic {
	c := &Critic{
		bus:                 cfg.Bus,
		history:             cfg.History,
		logger:              cfg.Logger,
		interval:            cfg.Interval,
		repetitionJaccard:   orDefault(cfg.RepetitionJaccard, 0.6),
		stagnationRepRate:   orDefault(cfg.StagnationRepRate, 0.5),
		stagnationDominance: orDefault(cfg.StagnationDominance, 0.6),
	}
	c.inbox = cfg.Bus.Subscribe("critic")
	return c
}

// Run は自己評価ループを開始する
func (c *Critic) Run(ctx context.Context) {
	c.logger.Info("critic", "自己評価モジュール開始")

	ticker := time.NewTicker(c.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-c.inbox:
			// メッセージは消費するが、評価は定期的にまとめて行う
		case <-ticker.C:
			c.evaluate()
		}
	}
}

func (c *Critic) evaluate() {
	recent := c.history.Recent(15)
	if len(recent) < 5 {
		return
	}

	// 指標1: 発信元の多様性（特定部位に偏っていないか）
	fromCounts := make(map[string]int)
	for _, t := range recent {
		fromCounts[t.From]++
	}
	dominance := 0.0
	for _, count := range fromCounts {
		ratio := float64(count) / float64(len(recent))
		if ratio > dominance {
			dominance = ratio
		}
	}

	// 指標2: 内容の反復性（連続する思考の類似度）
	repetitionCount := 0
	for i := 1; i < len(recent); i++ {
		words1 := tokenize(recent[i-1].Content)
		words2 := tokenize(recent[i].Content)
		sim := jaccardSimilarity(words1, words2)
		if sim > c.repetitionJaccard {
			repetitionCount++
		}
	}
	repetitionRate := float64(repetitionCount) / float64(len(recent)-1)

	// 指標3: メタモジュール以外の思考があるか
	brainThoughts := 0
	for _, t := range recent {
		if t.From == "frontal" || t.From == "temporal" || t.From == "hippocampus" {
			brainThoughts++
		}
	}

	// 停滞判定
	isStagnant := repetitionRate > c.stagnationRepRate || dominance > c.stagnationDominance

	if isStagnant {
		var reason string
		if repetitionRate > c.stagnationRepRate {
			reason = fmt.Sprintf("Thought repetition detected (%.0f%% similar consecutive thoughts).", repetitionRate*100)
		} else {
			var dominant string
			for from, count := range fromCounts {
				if float64(count)/float64(len(recent)) >= dominance {
					dominant = from
				}
			}
			reason = fmt.Sprintf("Thinking dominated by [%s] (%.0f%%). Need more diverse perspectives.", dominant, dominance*100)
		}

		thought := bus.Thought{
			From:    "critic",
			Content: reason,
		}
		c.history.Record(thought)
		c.bus.Publish(thought)
		c.logger.Info("critic", fmt.Sprintf("停滞検出: repetition=%.2f, dominance=%.2f", repetitionRate, dominance))
	}
}

// jaccardSimilarity は2つの単語リストのJaccard類似度を計算する
func jaccardSimilarity(a, b []string) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}

	setA := make(map[string]bool)
	for _, w := range a {
		setA[w] = true
	}
	setB := make(map[string]bool)
	for _, w := range b {
		setB[w] = true
	}

	intersection := 0
	for w := range setA {
		if setB[w] {
			intersection++
		}
	}

	union := len(setA) + len(setB) - intersection
	if union == 0 {
		return 0
	}
	return float64(intersection) / float64(union)
}
