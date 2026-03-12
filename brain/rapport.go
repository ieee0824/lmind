package brain

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
)

// Rapport はユーザーとの親密度を管理するモジュール（LLM不使用・アルゴリズム駆動）。
// ユーザーの同意・共感・肯定を検出してスコアを上昇させ、
// 否定・拒絶で下降させる。スコアはBrocaプロンプトに注入され、
// 親密度が高いほど人格の「素」が出るようになる。
type Rapport struct {
	mu     sync.RWMutex
	score  float64 // 0.0〜1.0
	bus    *bus.ThoughtBus
	logger *logger.Logger
	store  *logger.Logger // 永続化用
	inbox  <-chan bus.Thought
}

type RapportConfig struct {
	Bus    *bus.ThoughtBus
	Logger *logger.Logger
	Store  *logger.Logger // SQLite永続化用
}

func NewRapport(cfg RapportConfig) *Rapport {
	r := &Rapport{
		score:  0.3, // 初期値: やや警戒
		bus:    cfg.Bus,
		logger: cfg.Logger,
		store:  cfg.Store,
	}
	r.inbox = cfg.Bus.Subscribe("rapport")

	// SQLiteから前回のスコアを復元
	if cfg.Store != nil {
		if s := cfg.Store.LoadState("rapport"); s != "" {
			var v float64
			if _, err := fmt.Sscanf(s, "%f", &v); err == nil && v >= 0 && v <= 1 {
				r.score = v
			}
		}
	}

	return r
}

// Run はラポート検出ループを開始する
func (r *Rapport) Run(ctx context.Context) {
	r.logger.Info("rapport", fmt.Sprintf("ラポートモジュール開始 (初期スコア: %.2f)", r.Score()))

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-r.inbox:
			if t.From == "user" {
				r.evaluate(t.Content)
			}
		}
	}
}

// Score は現在のラポートスコアを返す
func (r *Rapport) Score() float64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.score
}

// Level はスコアに基づくラポートレベルを返す
func (r *Rapport) Level() string {
	s := r.Score()
	switch {
	case s >= 0.8:
		return "intimate" // 素に近い、照れが出る
	case s >= 0.6:
		return "friendly" // 軽口が増える、リラックス
	case s >= 0.4:
		return "neutral" // 普通の距離感
	default:
		return "guarded" // 警戒、丁寧め
	}
}

// evaluate はユーザー入力からラポートの変動を判定する
func (r *Rapport) evaluate(text string) {
	lower := strings.ToLower(text)

	var delta float64

	// 同意・共感（上昇）
	for keyword, weight := range agreementWords {
		if strings.Contains(lower, keyword) {
			delta += weight
		}
	}

	// 否定・拒絶（下降）
	for keyword, weight := range rejectionWords {
		if strings.Contains(lower, keyword) {
			delta -= weight
		}
	}

	if delta == 0 {
		return
	}

	// 変動を緩やかにする（急激な変化を防ぐ）
	delta = math.Max(-0.05, math.Min(0.05, delta*0.1))

	r.mu.Lock()
	oldScore := r.score
	r.score = math.Max(0.0, math.Min(1.0, r.score+delta))
	newScore := r.score
	r.mu.Unlock()

	// 永続化
	if r.store != nil {
		r.store.SaveState("rapport", fmt.Sprintf("%.3f", newScore))
	}

	if oldScore != newScore {
		r.logger.Info("rapport", fmt.Sprintf(
			"スコア変動: %.2f → %.2f (delta=%.3f, level=%s, input=%s)",
			oldScore, newScore, delta, r.Level(), text))
	}
}

// 同意・共感キーワード（日英混合）
var agreementWords = map[string]float64{
	// 日本語 - 同意
	"そうだね":  0.5,
	"そうだよね": 0.5,
	"たしかに":  0.5,
	"確かに":   0.5,
	"わかる":   0.5,
	"なるほど":  0.4,
	"その通り":  0.6,
	"そう思う":  0.5,
	"だよね":   0.4,
	"ね":      0.2,
	// 日本語 - 共感・肯定
	"すごい":   0.4,
	"いいね":   0.4,
	"面白い":   0.4,
	"おもしろい": 0.4,
	"さすが":   0.5,
	"うん":    0.3,
	"そっか":   0.3,
	"へぇ":    0.2,
	"ふーん":   0.2,
	// 日本語 - 親しみ
	"ありがとう":  0.5,
	"助かる":    0.4,
	"頼りになる":  0.5,
	"好き":     0.6,
	"楽しい":    0.4,
	"嬉しい":    0.4,
	"また話そう":  0.6,
	"また話したい": 0.6,
	// 英語
	"i agree":     0.5,
	"you're right": 0.5,
	"exactly":     0.5,
	"true":        0.4,
	"good point":  0.5,
	"i see":       0.3,
	"makes sense":  0.4,
	"nice":        0.3,
	"cool":        0.3,
	"thanks":      0.4,
	"thank you":   0.5,
}

// 否定・拒絶キーワード
var rejectionWords = map[string]float64{
	// 日本語
	"違う":    0.5,
	"違うよ":   0.5,
	"それは違う": 0.6,
	"違くない":  0.3,
	"いや":    0.3,
	"でも":    0.2,
	"うーん":   0.1,
	"微妙":    0.3,
	"わからない": 0.2,
	"つまらない": 0.4,
	"うざい":   0.6,
	"黙って":   0.7,
	"やめて":   0.5,
	// 英語
	"no":          0.3,
	"wrong":       0.5,
	"i disagree":  0.5,
	"not really":  0.3,
	"whatever":    0.4,
	"shut up":     0.7,
	"boring":      0.5,
	"stop":        0.4,
}
