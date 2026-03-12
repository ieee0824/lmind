package config

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// Params は全モジュールの調整可能パラメータを保持する。
// JSON外出しで遺伝的アルゴリズムによる最適化に対応。
type Params struct {
	Intervals       IntervalParams       `json:"intervals"`
	Gain            GainParams           `json:"gain"`
	Stagnation      StagnationParams     `json:"stagnation"`
	Hypothesis      HypothesisParams     `json:"hypothesis"`
	PredictionError PredictionErrorParams `json:"prediction_error"`
	Attention       AttentionParams      `json:"attention"`
	Critic          CriticParams         `json:"critic"`
	Curiosity       CuriosityParams      `json:"curiosity"`
	Novelty         NoveltyParams        `json:"novelty"`
	Bonnou          BonnouParams         `json:"bonnou"`
	History         HistoryParams        `json:"history"`
}

// IntervalParams は各モジュールの思考サイクル間隔（秒）
type IntervalParams struct {
	Frontal     float64 `json:"frontal"`
	Temporal    float64 `json:"temporal"`
	Hippocampus float64 `json:"hippocampus"`
	Hypothesis  float64 `json:"hypothesis"`
	Critic      float64 `json:"critic"`
	Curiosity   float64 `json:"curiosity"`
	Grounding   float64 `json:"grounding"`
	Bonnou      float64 `json:"bonnou"`
	Modulator   float64 `json:"modulator"`
}

// GainParams はモジュールゲインの制御パラメータ
type GainParams struct {
	SkipThreshold  float64 `json:"skip_threshold"`
	MaxGain        float64 `json:"max_gain"`
	IdleDecayStart float64 `json:"idle_decay_start"` // 秒
	RecoverStep    float64 `json:"recover_step"`
	DecayStep      float64 `json:"decay_step"`
}

// StagnationParams は停滞検知時のゲイン調整量
type StagnationParams struct {
	DecayFrontal     float64 `json:"decay_frontal"`
	DecayTemporal    float64 `json:"decay_temporal"`
	DecayHypothesis  float64 `json:"decay_hypothesis"`
	BoostCuriosity   float64 `json:"boost_curiosity"`
	BoostGrounding   float64 `json:"boost_grounding"`
	NoveltyRecovery  float64 `json:"novelty_recovery"`
}

// HypothesisParams は仮説生成モジュールのパラメータ
type HypothesisParams struct {
	DedupJaccard      float64 `json:"dedup_jaccard"`
	SentimentPositive float64 `json:"sentiment_positive"`
	SentimentNegative float64 `json:"sentiment_negative"`
}

// PredictionErrorParams は予測誤差モジュールのパラメータ
type PredictionErrorParams struct {
	HighThreshold   float64 `json:"high_threshold"`
	MediumThreshold float64 `json:"medium_threshold"`
}

// AttentionParams はサリエンスフィルタのパラメータ
type AttentionParams struct {
	SalienceThreshold float64 `json:"salience_threshold"`
	Baseline          float64 `json:"baseline"`
	GoalWeight        float64 `json:"goal_weight"`
	RepetitionPenalty float64 `json:"repetition_penalty"`
	MetaPenalty       float64 `json:"meta_penalty"`
	UserBonus         float64 `json:"user_bonus"`
}

// CriticParams は自己評価モジュールのパラメータ
type CriticParams struct {
	RepetitionJaccard    float64 `json:"repetition_jaccard"`
	StagnationRepRate    float64 `json:"stagnation_rep_rate"`
	StagnationDominance  float64 `json:"stagnation_dominance"`
}

// CuriosityParams は探索モジュールのパラメータ
type CuriosityParams struct {
	FrequentThreshold int `json:"frequent_threshold"`
	RareThreshold     int `json:"rare_threshold"`
}

// NoveltyParams は新規性検出モジュールのパラメータ
type NoveltyParams struct {
	ScoreThreshold float64 `json:"score_threshold"`
}

// BonnouParams は煩悩モジュールのパラメータ
type BonnouParams struct {
	HotWordRetention  float64 `json:"hot_word_retention"` // 秒
	IntensityThreshold float64 `json:"intensity_threshold"`
}

// HistoryParams は思考履歴の圧縮パラメータ
type HistoryParams struct {
	FreshCount int `json:"fresh_count"`
}

// DefaultParams は現在のハードコード値をデフォルトとして返す
func DefaultParams() *Params {
	return &Params{
		Intervals: IntervalParams{
			Frontal:     30,
			Temporal:    30,
			Hippocampus: 25,
			Hypothesis:  35,
			Critic:      45,
			Curiosity:   60,
			Grounding:   40,
			Bonnou:      50,
			Modulator:   30,
		},
		Gain: GainParams{
			SkipThreshold:  0.3,
			MaxGain:        2.0,
			IdleDecayStart: 60,
			RecoverStep:    0.1,
			DecayStep:      0.1,
		},
		Stagnation: StagnationParams{
			DecayFrontal:    0.3,
			DecayTemporal:   0.3,
			DecayHypothesis: 0.2,
			BoostCuriosity:  0.3,
			BoostGrounding:  0.2,
			NoveltyRecovery: 0.2,
		},
		Hypothesis: HypothesisParams{
			DedupJaccard:      0.6,
			SentimentPositive: 0.3,
			SentimentNegative: -0.3,
		},
		PredictionError: PredictionErrorParams{
			HighThreshold:   0.8,
			MediumThreshold: 0.5,
		},
		Attention: AttentionParams{
			SalienceThreshold: 0.3,
			Baseline:          0.5,
			GoalWeight:        0.3,
			RepetitionPenalty: 0.3,
			MetaPenalty:       0.2,
			UserBonus:         0.2,
		},
		Critic: CriticParams{
			RepetitionJaccard:   0.6,
			StagnationRepRate:   0.5,
			StagnationDominance: 0.6,
		},
		Curiosity: CuriosityParams{
			FrequentThreshold: 3,
			RareThreshold:     1,
		},
		Novelty: NoveltyParams{
			ScoreThreshold: 0.85,
		},
		Bonnou: BonnouParams{
			HotWordRetention:   300, // 5分
			IntensityThreshold: 0.5,
		},
		History: HistoryParams{
			FreshCount: 3,
		},
	}
}

// LoadParams はJSONファイルからパラメータを読み込む。
// 未指定のフィールドはDefaultParamsで埋まる。
func LoadParams(path string) (*Params, error) {
	p := DefaultParams()
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("params読み込み失敗: %w", err)
	}
	if err := json.Unmarshal(data, p); err != nil {
		return nil, fmt.Errorf("params解析失敗: %w", err)
	}
	return p, nil
}

// IntervalDuration はIntervalParamsの各値をtime.Durationに変換するヘルパー
func (p *IntervalParams) Duration(seconds float64) time.Duration {
	return time.Duration(seconds * float64(time.Second))
}
