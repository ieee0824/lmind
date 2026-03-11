package brain

import (
	"context"
	"fmt"
	"strings"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
)

// Sentiment はセンチメントゲート（LLM不使用）。
// ユーザー入力の感情極性をキーワードベースで高速判定し、
// ポジティブ入力時にtemporal抑制・frontal強化・anxiety仮説の無効化を行う。
// temporalが「pattern completion」モード（不安ナラティブへの吸収）に陥るのを防ぐ。
type Sentiment struct {
	bus        *bus.ThoughtBus
	logger     *logger.Logger
	state      *State
	modulation *Modulation
	inbox      <-chan bus.Thought
}

type SentimentConfig struct {
	Bus        *bus.ThoughtBus
	Logger     *logger.Logger
	State      *State
	Modulation *Modulation
}

func NewSentiment(cfg SentimentConfig) *Sentiment {
	s := &Sentiment{
		bus:        cfg.Bus,
		logger:     cfg.Logger,
		state:      cfg.State,
		modulation: cfg.Modulation,
	}
	s.inbox = cfg.Bus.Subscribe("sentiment")
	return s
}

// Run はセンチメントゲートループを開始する
func (s *Sentiment) Run(ctx context.Context) {
	s.logger.Info("sentiment", "センチメントゲート開始")

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-s.inbox:
			if t.From == "user" {
				s.evaluate(t)
			}
		}
	}
}

// evaluate はユーザー入力のセンチメントを判定しゲインを調整する
func (s *Sentiment) evaluate(t bus.Thought) {
	score := sentimentScore(t.Content)
	s.logger.Info("sentiment", fmt.Sprintf("スコア: %.2f (%s)", score, t.Content))

	if score > 0.3 {
		// ポジティブ入力: temporalの物語生成を抑制し、frontalの文脈解釈を強化
		s.modulation.Set("temporal", s.modulation.Gain("temporal")-0.2)
		s.modulation.Set("frontal", s.modulation.Gain("frontal")+0.1)

		// anxiety固定された仮説を無効化
		s.clearAnxietyHypothesis()

		s.logger.Info("sentiment", fmt.Sprintf("ポジティブ補正: temporal=%.1f, frontal=%.1f",
			s.modulation.Gain("temporal"), s.modulation.Gain("frontal")))
	} else if score < -0.3 {
		// ネガティブ入力: temporalを少しブーストして共感的文脈理解を促す
		s.modulation.Set("temporal", s.modulation.Gain("temporal")+0.1)

		s.logger.Info("sentiment", fmt.Sprintf("ネガティブ補正: temporal=%.1f", s.modulation.Gain("temporal")))
	}
}

// clearAnxietyHypothesis はanxiety系キーワードを含む仮説をクリアする
func (s *Sentiment) clearAnxietyHypothesis() {
	if s.state == nil {
		return
	}
	_, hyp := s.state.Snapshot()
	if hyp == "" {
		return
	}
	lower := strings.ToLower(hyp)
	for _, keyword := range anxietyKeywords {
		if strings.Contains(lower, keyword) {
			s.state.SetHypothesis("")
			s.logger.Info("sentiment", fmt.Sprintf("anxiety仮説をクリア: %s", hyp))
			return
		}
	}
}

// anxietyKeywords は仮説がanxietyループに固定されていることを示すキーワード
var anxietyKeywords = []string{
	"anxiety",
	"anxious",
	"distress",
	"instability",
	"fear",
	"worry",
	"worried",
	"panic",
	"nervous",
	"uneasy",
	"threat",
	"danger",
	"crisis",
	"turmoil",
	"suffering",
}

// sentimentScore はテキストのセンチメントスコアを返す（-1.0〜+1.0）
// キーワードベースの高速判定。LLM不使用。
func sentimentScore(text string) float64 {
	lower := strings.ToLower(text)
	words := strings.Fields(lower)

	var pos, neg float64
	for _, w := range words {
		if score, ok := positiveWords[w]; ok {
			pos += score
		}
		if score, ok := negativeWords[w]; ok {
			neg += score
		}
	}

	total := pos + neg
	if total == 0 {
		return 0
	}
	// -1.0〜+1.0にクランプ
	score := (pos - neg) / float64(len(words))
	if score > 1.0 {
		return 1.0
	}
	if score < -1.0 {
		return -1.0
	}
	return score
}

// ポジティブキーワード（英語・日本語混合）
var positiveWords = map[string]float64{
	// 日本語
	"嬉しい":   1.0,
	"嬉しく":   0.8,
	"楽しい":   1.0,
	"楽しく":   0.8,
	"ありがとう": 1.0,
	"感謝":    0.9,
	"良い":    0.7,
	"いい":    0.6,
	"素晴らしい": 1.0,
	"素敵":    0.9,
	"好き":    0.8,
	"最高":    1.0,
	"喜んで":   0.9,
	"喜ぶ":    0.9,
	"幸せ":    1.0,
	"うれしい":  1.0,
	"たのしい":  1.0,
	"よかった":  0.8,
	"いいね":   0.7,
	"すごい":   0.8,
	"わーい":   0.9,
	"やった":   0.8,
	"おめでとう": 0.9,
	"安心":    0.7,
	"面白い":   0.7,
	// 英語
	"happy":     1.0,
	"glad":      0.9,
	"pleased":   0.8,
	"great":     0.8,
	"good":      0.7,
	"love":      0.9,
	"wonderful":  1.0,
	"excellent":  1.0,
	"amazing":    1.0,
	"thank":     0.8,
	"thanks":    0.8,
	"awesome":    0.9,
	"nice":      0.7,
	"joy":       1.0,
	"excited":   0.8,
	"fantastic":  1.0,
	"beautiful":  0.8,
	"perfect":   0.9,
	"fun":       0.8,
	"enjoy":     0.8,
}

// ネガティブキーワード
var negativeWords = map[string]float64{
	// 日本語
	"悲しい":  1.0,
	"辛い":   0.9,
	"つらい":  0.9,
	"苦しい":  0.9,
	"怖い":   0.8,
	"不安":   0.8,
	"心配":   0.7,
	"嫌い":   0.8,
	"嫌":    0.7,
	"困った":  0.7,
	"ダメ":   0.7,
	"だめ":   0.7,
	"最悪":   1.0,
	"疲れた":  0.6,
	"イライラ": 0.8,
	"むかつく": 0.8,
	"怒り":   0.8,
	// 英語
	"sad":         1.0,
	"angry":       0.9,
	"frustrated":  0.8,
	"terrible":    1.0,
	"awful":       1.0,
	"bad":         0.7,
	"hate":        0.9,
	"worried":     0.7,
	"anxious":     0.8,
	"afraid":      0.8,
	"scared":      0.8,
	"upset":       0.8,
	"disappointed": 0.8,
	"miserable":    1.0,
	"annoyed":     0.7,
	"painful":     0.8,
	"horrible":    1.0,
}
