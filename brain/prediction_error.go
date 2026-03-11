package brain

import (
	"context"
	"fmt"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
)

// PredictionError は予測誤差モジュール（LLM不使用・アルゴリズム駆動）。
// ユーザー入力が来た時に、前回の予測（State.prediction）と実際の入力を比較し、
// 誤差が大きければ仮説をドロップして思考をリセットする。
// これにより「同じ仮説の言い換え」ループ（narrative attractor）を破壊する。
type PredictionError struct {
	bus    *bus.ThoughtBus
	logger *logger.Logger
	state  *State
	inbox  <-chan bus.Thought
}

type PredictionErrorConfig struct {
	Bus    *bus.ThoughtBus
	Logger *logger.Logger
	State  *State
}

func NewPredictionError(cfg PredictionErrorConfig) *PredictionError {
	pe := &PredictionError{
		bus:    cfg.Bus,
		logger: cfg.Logger,
		state:  cfg.State,
	}
	pe.inbox = cfg.Bus.Subscribe("prediction_error")
	return pe
}

// Run は予測誤差検出ループを開始する
func (pe *PredictionError) Run(ctx context.Context) {
	pe.logger.Info("prediction_error", "予測誤差モジュール開始")

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-pe.inbox:
			if t.From == "user" {
				pe.evaluate(t.Content)
			}
		}
	}
}

// evaluate は予測と実際の入力を比較し、誤差に応じて仮説を制御する
func (pe *PredictionError) evaluate(actualInput string) {
	prediction := pe.state.Prediction()
	if prediction == "" {
		return // 予測がなければ評価不能
	}

	// 予測と実際の入力のJaccard距離で誤差を計算
	predWords := tokenize(prediction)
	actualWords := tokenize(actualInput)

	if len(predWords) == 0 || len(actualWords) == 0 {
		return
	}

	similarity := jaccardSimilarity(predWords, actualWords)
	error_ := 1.0 - similarity // 0.0 = 完全一致, 1.0 = 完全不一致

	pe.logger.Info("prediction_error", fmt.Sprintf(
		"誤差: %.2f (予測: %s / 実際: %s)", error_, prediction, actualInput))

	switch {
	case error_ > 0.8:
		// 高誤差: 予測が大きく外れた → 仮説をドロップ
		_, hyp := pe.state.Snapshot()
		if hyp != "" {
			pe.state.SetHypothesis("")
			pe.logger.Info("prediction_error", fmt.Sprintf("高誤差→仮説ドロップ: %s", hyp))

			pe.bus.Publish(bus.Thought{
				From:    "prediction_error",
				Content: fmt.Sprintf("Prediction error HIGH (%.2f). Previous hypothesis dropped. Fresh analysis needed.", error_),
			})
		}

	case error_ > 0.5:
		// 中誤差: 部分的に外れた → 仮説の再評価を促す
		pe.bus.Publish(bus.Thought{
			From:    "prediction_error",
			Content: fmt.Sprintf("Prediction error MODERATE (%.2f). Current hypothesis may need revision.", error_),
		})
		pe.logger.Info("prediction_error", "中誤差→仮説再評価シグナル")

	default:
		// 低誤差: 予測が当たった → 仮説は妥当
		pe.logger.Info("prediction_error", fmt.Sprintf("低誤差(%.2f)→仮説維持", error_))
	}

	// 予測をクリア（次のターンで再生成される）
	pe.state.SetPrediction("")
}
