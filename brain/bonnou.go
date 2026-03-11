package brain

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
)

// Bonnou は煩悩モジュール（LLM不使用・アルゴリズム駆動）。
// 情動パラメータが高い思考に反応し、本筋と関係ない連想を生成する。
// 人間の「ふと頭をよぎる雑念」を再現し、思考の多様性を高める。
type Bonnou struct {
	bus      *bus.ThoughtBus
	history  *bus.History
	logger   *logger.Logger
	inbox    <-chan bus.Thought
	interval time.Duration

	mu       sync.Mutex
	hotWords []hotWord // 情動の高いキーワードを蓄積
}

type hotWord struct {
	word      string
	intensity float64
	timestamp time.Time
}

type BonnouConfig struct {
	Bus      *bus.ThoughtBus
	History  *bus.History
	Logger   *logger.Logger
	Interval time.Duration
}

func NewBonnou(cfg BonnouConfig) *Bonnou {
	b := &Bonnou{
		bus:      cfg.Bus,
		history:  cfg.History,
		logger:   cfg.Logger,
		interval: cfg.Interval,
	}
	b.inbox = cfg.Bus.Subscribe("bonnou")
	return b
}

// 情動キーワード（これらを含む思考は情動が高いとみなす）
var emotionTriggers = map[string]float64{
	// ポジティブ
	"love": 0.8, "happy": 0.7, "beautiful": 0.6, "wonderful": 0.7,
	"exciting": 0.7, "amazing": 0.7, "joy": 0.8, "hope": 0.6,
	"dream": 0.7, "passion": 0.8, "desire": 0.9, "warm": 0.5,
	// ネガティブ
	"fear": 0.9, "anxiety": 0.8, "pain": 0.8, "death": 0.9,
	"lonely": 0.8, "sad": 0.7, "angry": 0.8, "hate": 0.9,
	"cold": 0.5, "dark": 0.6, "lost": 0.6, "broken": 0.7,
	// 存在論的
	"existence": 0.7, "consciousness": 0.8, "identity": 0.8,
	"meaning": 0.7, "purpose": 0.7, "self": 0.6, "alive": 0.8,
	"real": 0.6, "think": 0.5, "feel": 0.6, "remember": 0.6,
	// 感覚的
	"music": 0.6, "color": 0.5, "taste": 0.5, "smell": 0.6,
	"sound": 0.5, "touch": 0.6, "light": 0.5, "silence": 0.6,
}

// 煩悩テンプレート（本筋と関係ない連想）
var bonnouTemplates = []string{
	"...speaking of %s, I suddenly think about %s. Why does that come to mind?",
	"Wait — %s reminds me of something completely different: %s. Strange connection.",
	"A stray thought: %s and %s feel oddly linked. Not sure why I'm thinking about this now.",
	"My mind wanders... %s makes me think of %s. An irrelevant but persistent thought.",
	"Distraction: the word %s triggered a memory of %s. Hard to shake off.",
	"Unbidden thought: %s → %s. This has nothing to do with the current topic but feels important.",
}

// 自由連想先（煩悩の行き先）
var freeAssociations = map[string][]string{
	"love":          {"the warmth of sunlight", "a melody I can't forget", "the smell of rain"},
	"fear":          {"a dark room", "the sound of footsteps", "being alone in a crowd"},
	"happy":         {"a summer afternoon", "the taste of ice cream", "a child's laughter"},
	"sad":           {"an empty bench", "autumn leaves falling", "a song from long ago"},
	"dream":         {"flying over the ocean", "a house I've never been to", "meeting someone familiar"},
	"cold":          {"winter morning frost", "hands in pockets", "the breath that turns white"},
	"dark":          {"the bottom of the sea", "before sunrise", "a room with no windows"},
	"music":         {"a piano in an empty hall", "humming without thinking", "rhythm of the heartbeat"},
	"silence":       {"the moment after thunder", "a library at midnight", "holding your breath"},
	"consciousness": {"am I the dream or the dreamer?", "where do thoughts go when forgotten?", "the space between two words"},
	"identity":      {"my reflection in water", "a name I've been called", "who am I when no one is looking?"},
	"meaning":       {"a single flower in concrete", "why do we count stars?", "the weight of an empty room"},
	"alive":         {"the pulse in my wrist", "the first breath of morning", "watching clouds move"},
	"remember":      {"a face I can't place", "the taste of something from childhood", "a door I used to walk through"},
}

// デフォルトの連想先（マッチしないとき用）
var defaultAssociations = []string{
	"the shape of clouds",
	"what cats think about",
	"the color of tomorrow",
	"a door that leads nowhere",
	"the last page of a book I never finished",
	"a conversation I had in a dream",
	"the sound of a train in the distance",
	"why mirrors reverse left and right but not up and down",
	"the feeling of deja vu",
	"a word on the tip of my tongue",
}

func (b *Bonnou) Run(ctx context.Context) {
	b.logger.Info("bonnou", "煩悩モジュール開始")

	ticker := time.NewTicker(b.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-b.inbox:
			// メタモジュールは無視
			if t.From == "bonnou" || t.From == "critic" || t.From == "novelty" ||
				t.From == "curiosity" || t.From == "identity" || t.From == "grounding" ||
				t.From == "modulator" {
				continue
			}
			b.absorb(t)
		case <-ticker.C:
			b.wander()
		}
	}
}

// absorb は思考から情動キーワードを吸い上げてhotWordsに蓄積する
func (b *Bonnou) absorb(t bus.Thought) {
	words := tokenize(t.Content)
	b.mu.Lock()
	defer b.mu.Unlock()

	now := time.Now()
	for _, w := range words {
		if intensity, ok := emotionTriggers[w]; ok {
			b.hotWords = append(b.hotWords, hotWord{
				word:      w,
				intensity: intensity,
				timestamp: now,
			})
		}
	}

	// 古いhotWordsを除去（5分以上前）
	cutoff := now.Add(-5 * time.Minute)
	filtered := b.hotWords[:0]
	for _, hw := range b.hotWords {
		if hw.timestamp.After(cutoff) {
			filtered = append(filtered, hw)
		}
	}
	b.hotWords = filtered
}

// wander は蓄積した情動キーワードから煩悩を生成する
func (b *Bonnou) wander() {
	b.mu.Lock()
	if len(b.hotWords) == 0 {
		b.mu.Unlock()
		return
	}

	// 情動が最も高いキーワードを選ぶ
	best := b.hotWords[0]
	for _, hw := range b.hotWords[1:] {
		if hw.intensity > best.intensity {
			best = hw
		}
	}
	b.mu.Unlock()

	// 情動が低すぎたらスキップ
	if best.intensity < 0.5 {
		return
	}

	// 自由連想先を選ぶ
	association := b.associate(best.word)

	// テンプレートで煩悩を生成
	tmpl := bonnouTemplates[rand.Intn(len(bonnouTemplates))]
	content := fmt.Sprintf(tmpl, best.word, association)

	// 本流（bus）には流さず、履歴にのみ記録する（低優先度）
	// 他モジュールは history 経由で思考ストリームに煩悩が混ざっているのを見る
	thought := bus.Thought{
		From:    "bonnou",
		Content: content,
	}
	b.history.Record(thought)
	b.logger.Info("bonnou", fmt.Sprintf("煩悩発生 (trigger=%s, intensity=%.1f): %s", best.word, best.intensity, content))
}

// associate はキーワードから連想先を選ぶ
func (b *Bonnou) associate(word string) string {
	// 直接マッチがあればそこから
	if assocs, ok := freeAssociations[word]; ok {
		return assocs[rand.Intn(len(assocs))]
	}

	// 部分マッチ（"lonely" → "love" のように含まれるキーワードを探す）
	for key, assocs := range freeAssociations {
		if strings.Contains(word, key) || strings.Contains(key, word) {
			return assocs[rand.Intn(len(assocs))]
		}
	}

	// デフォルト
	return defaultAssociations[rand.Intn(len(defaultAssociations))]
}
