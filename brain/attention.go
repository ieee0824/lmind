package brain

import (
	"context"
	"fmt"
	"strings"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
)

// Attention はサリエンス（顕著性）フィルター（LLM不使用・アルゴリズム駆動）。
// 思考ストリームの各入力に重要度スコアを付与し、
// 低サリエンスの内部思考が思考バスを支配するのを防ぐ。
//
// サリエンスが高い入力:
//   - ユーザーの発言（常に高い）
//   - 予測誤差が高い時の分析
//   - 新規性が高い入力
//   - 現在のgoalに関連する内容
//
// サリエンスが低い入力:
//   - 直近の思考と内容が重複する内部思考
//   - goalと無関係な自己言及的メタ思考
type Attention struct {
	bus               *bus.ThoughtBus
	history           *bus.History
	logger            *logger.Logger
	state             *State
	inbox             <-chan bus.Thought
	salienceThreshold float64
	baseline          float64
	goalWeight        float64
	repetitionPenalty float64
	metaPenalty       float64
	userBonus         float64
}

type AttentionConfig struct {
	Bus               *bus.ThoughtBus
	History           *bus.History
	Logger            *logger.Logger
	State             *State
	SalienceThreshold float64
	Baseline          float64
	GoalWeight        float64
	RepetitionPenalty float64
	MetaPenalty       float64
	UserBonus         float64
}

func NewAttention(cfg AttentionConfig) *Attention {
	a := &Attention{
		bus:               cfg.Bus,
		history:           cfg.History,
		logger:            cfg.Logger,
		state:             cfg.State,
		salienceThreshold: orDefault(cfg.SalienceThreshold, 0.3),
		baseline:          orDefault(cfg.Baseline, 0.5),
		goalWeight:        orDefault(cfg.GoalWeight, 0.3),
		repetitionPenalty: orDefault(cfg.RepetitionPenalty, 0.3),
		metaPenalty:       orDefault(cfg.MetaPenalty, 0.2),
		userBonus:         orDefault(cfg.UserBonus, 0.2),
	}
	a.inbox = cfg.Bus.Subscribe("attention")
	return a
}

// Run はアテンションフィルターループを開始する
func (a *Attention) Run(ctx context.Context) {
	a.logger.Info("attention", "アテンションフィルター開始")

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-a.inbox:
			// 内部モジュールの思考をフィルタリング
			// user/broca は常に通過（外部入力は無条件で重要）
			if t.From == "user" || t.From == "broca" {
				continue
			}
			// メタモジュール（critic/novelty/identity等）も通過
			if isMetaModule(t.From) {
				continue
			}
			// frontal/temporal/hypothesis の出力をサリエンス判定
			salience := a.scoreSalience(t)
			if salience < a.salienceThreshold {
				a.logger.Info("attention", fmt.Sprintf(
					"低サリエンス(%.2f)を抑制: [%s] %s",
					salience, t.From, truncateForLog(t.Content, 60)))
				// 低サリエンスの思考は履歴に残さない
				// （既にバスに流れているが、次のサイクルで影響を減らす）
				a.bus.Publish(bus.Thought{
					From:    "attention",
					Content: fmt.Sprintf("Low salience thought from [%s] detected (%.2f). Focus on what the user actually said.", t.From, salience),
				})
			}
		}
	}
}

// scoreSalience は思考のサリエンス（重要度）スコアを算出する (0.0〜1.0)
func (a *Attention) scoreSalience(t bus.Thought) float64 {
	score := a.baseline

	words := tokenize(t.Content)
	if len(words) == 0 {
		return 0.0
	}

	// 1. goalとの関連性 (+0.3)
	goal, _ := a.state.Snapshot()
	if goal != "" && goal != "No specific goal. Freely exploring thoughts." {
		goalWords := tokenize(goal)
		if len(goalWords) > 0 {
			sim := jaccardSimilarity(words, goalWords)
			score += sim * a.goalWeight
		}
	}

	// 2. 直近の思考との重複度 (-0.3)
	recent := a.history.Recent(5)
	if len(recent) > 0 {
		var totalSim float64
		for _, r := range recent {
			if r.From == t.From {
				rWords := tokenize(r.Content)
				totalSim += jaccardSimilarity(words, rWords)
			}
		}
		avgSim := totalSim / float64(len(recent))
		// 重複が高いほどサリエンスを下げる
		score -= avgSim * a.repetitionPenalty
	}

	// 3. メタ自己言及キーワードの検出 (-0.2)
	if containsMetaKeywords(t.Content) {
		score -= a.metaPenalty
	}

	// 4. ユーザー入力への言及があれば加点
	if containsUserReference(t.Content) {
		score += a.userBonus
	}

	// クランプ
	if score > 1.0 {
		return 1.0
	}
	if score < 0.0 {
		return 0.0
	}
	return score
}

// メタ自己言及キーワード（AIが自身の振る舞いを分析している兆候）
func containsMetaKeywords(text string) bool {
	lower := strings.ToLower(text)
	metaPatterns := []string{
		"my response",
		"my approach",
		"my behavior",
		"my analysis",
		"i prioritize",
		"i demonstrate",
		"my reliance",
		"my focus on",
		"my current",
		"i am analyzing",
		"i am processing",
		"i should",
		"i need to",
		"my role",
		"feedback loop",
		"algorithmic",
	}
	for _, p := range metaPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

// ユーザーの発言内容に言及しているかのチェック
func containsUserReference(text string) bool {
	lower := strings.ToLower(text)
	patterns := []string{
		"the user",
		"user said",
		"user asked",
		"user wants",
		"user means",
		"user's question",
		"user→me",
	}
	for _, p := range patterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func isMetaModule(from string) bool {
	switch from {
	case "critic", "novelty", "curiosity", "identity", "grounding",
		"sentiment", "modulator", "prediction_error", "attention", "bonnou":
		return true
	}
	return false
}

func truncateForLog(s string, maxLen int) string {
	runes := []rune(s)
	if len(runes) <= maxLen {
		return s
	}
	return string(runes[:maxLen]) + "..."
}
