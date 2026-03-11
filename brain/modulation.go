package brain

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
)

// Modulation は各モジュールのゲイン（出力強度）を管理する。
// gain = 1.0 が通常、0.0 で完全抑制、>1.0 でブースト。
type Modulation struct {
	mu    sync.RWMutex
	gains map[string]float64
}

func NewModulation() *Modulation {
	return &Modulation{
		gains: map[string]float64{
			"frontal":    1.0,
			"temporal":   1.0,
			"hypothesis": 1.0,
			"curiosity":  1.0,
			"grounding":  1.0,
		},
	}
}

// Gain はモジュールの現在のゲインを返す（未登録なら1.0）
func (m *Modulation) Gain(module string) float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if g, ok := m.gains[module]; ok {
		return g
	}
	return 1.0
}

// Set はモジュールのゲインを設定する（0.0〜2.0にクランプ）
func (m *Modulation) Set(module string, gain float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.gains[module] = math.Max(0.0, math.Min(2.0, gain))
}

// Snapshot は全モジュールのゲインをコピーで返す
func (m *Modulation) Snapshot() map[string]float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	cp := make(map[string]float64, len(m.gains))
	for k, v := range m.gains {
		cp[k] = v
	}
	return cp
}

// ShouldSkip はゲインに基づいてこのサイクルをスキップすべきか返す。
// gain >= 1.0: スキップしない
// gain < 1.0: gain を確率としてスキップ（例: 0.3なら70%の確率でスキップ）
// gain == 0.0: 常にスキップ
func (m *Modulation) ShouldSkip(module string) bool {
	g := m.Gain(module)
	if g >= 1.0 {
		return false
	}
	if g <= 0.0 {
		return true
	}
	// 簡易的: gain未満の閾値でスキップ
	// 毎回同じ判定にするためランダムは使わず、0.3未満なら常にスキップ
	return g < 0.3
}

// Format はログ用に現在のゲインを文字列化する
func (m *Modulation) Format() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var parts []string
	for name, gain := range m.gains {
		parts = append(parts, fmt.Sprintf("%s=%.1f", name, gain))
	}
	return strings.Join(parts, ", ")
}

// Modulator はcriticの停滞検知・noveltyの新規入力・ユーザー入力に応じて
// 各モジュールのゲインを動的に調整するモジュール。
type Modulator struct {
	mod      *Modulation
	bus      *bus.ThoughtBus
	history  *bus.History
	logger   *logger.Logger
	inbox    <-chan bus.Thought
	interval time.Duration
}

type ModulatorConfig struct {
	Modulation *Modulation
	Bus        *bus.ThoughtBus
	History    *bus.History
	Logger     *logger.Logger
	Interval   time.Duration // 回復チェック間隔
}

func NewModulator(cfg ModulatorConfig) *Modulator {
	m := &Modulator{
		mod:      cfg.Modulation,
		bus:      cfg.Bus,
		history:  cfg.History,
		logger:   cfg.Logger,
		interval: cfg.Interval,
	}
	m.inbox = cfg.Bus.Subscribe("modulator")
	return m
}

// Run はゲイン調整ループを開始する
func (m *Modulator) Run(ctx context.Context) {
	m.logger.Info("modulator", "ゲイン調整モジュール開始")

	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return

		case t := <-m.inbox:
			m.react(t)

		case <-ticker.C:
			m.recover()
		}
	}
}

// react はイベントに応じてゲインを調整する
func (m *Modulator) react(t bus.Thought) {
	switch {
	case t.From == "critic":
		// 停滞検知 → ループ元を抑制、探索系をブースト
		m.onStagnation()

	case t.From == "user":
		// ユーザー入力 → 全モジュールを通常に復帰
		m.onUserInput()

	case t.From == "novelty":
		// 新規性検出 → 思考系を少し回復
		m.onNovelty()
	}
}

// onStagnation は停滞検知時のゲイン調整
func (m *Modulator) onStagnation() {
	// ループの主犯（思考生成系）を抑制
	m.mod.Set("frontal", m.mod.Gain("frontal")-0.3)
	m.mod.Set("temporal", m.mod.Gain("temporal")-0.3)
	m.mod.Set("hypothesis", m.mod.Gain("hypothesis")-0.2)

	// 打開策（探索・アンカー系）をブースト
	m.mod.Set("curiosity", math.Min(2.0, m.mod.Gain("curiosity")+0.3))
	m.mod.Set("grounding", math.Min(2.0, m.mod.Gain("grounding")+0.2))

	m.logger.Info("modulator", fmt.Sprintf("停滞→ゲイン調整: %s", m.mod.Format()))
}

// onUserInput はユーザー入力時のゲイン調整（全復帰）
func (m *Modulator) onUserInput() {
	m.mod.Set("frontal", 1.0)
	m.mod.Set("temporal", 1.0)
	m.mod.Set("hypothesis", 1.0)
	m.mod.Set("curiosity", 1.0)
	m.mod.Set("grounding", 1.0)

	m.logger.Info("modulator", "ユーザー入力→ゲイン全復帰")
}

// onNovelty は新規性検出時のゲイン調整
func (m *Modulator) onNovelty() {
	// 思考系を少し回復（新しいネタがあるなら考える価値がある）
	if m.mod.Gain("frontal") < 1.0 {
		m.mod.Set("frontal", m.mod.Gain("frontal")+0.2)
	}
	if m.mod.Gain("temporal") < 1.0 {
		m.mod.Set("temporal", m.mod.Gain("temporal")+0.2)
	}
}

// recover は定期的にゲインを通常値に向けて緩やかに回復させる
func (m *Modulator) recover() {
	snapshot := m.mod.Snapshot()
	changed := false

	for name, gain := range snapshot {
		if gain < 1.0 {
			m.mod.Set(name, gain+0.1)
			changed = true
		} else if gain > 1.0 {
			m.mod.Set(name, gain-0.1)
			changed = true
		}
	}

	if changed {
		m.logger.Info("modulator", fmt.Sprintf("ゲイン回復: %s", m.mod.Format()))
	}
}
