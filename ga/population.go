package ga

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/ieee0824/lmind/config"
)

// Population はGA世代の個体群を管理する
type Population struct {
	Binary          string // lmindバイナリのパス
	BasePort        int
	BaseDir         string // 一時ディレクトリのルート
	CharSettingPath string // char_setting.md のパス
	Size            int
	Individuals     []*Individual
}

// NewPopulation は初期集団を生成する（DefaultParamsにランダム摂動を加える）
func NewPopulation(binary string, basePort int, baseDir string, charSettingPath string, size int) *Population {
	pop := &Population{
		Binary:          binary,
		BasePort:        basePort,
		BaseDir:         baseDir,
		CharSettingPath: charSettingPath,
		Size:            size,
	}

	for i := range size {
		params := config.DefaultParams()
		mutateParams(params, 0.3) // 初期摂動は大きめ
		pop.Individuals = append(pop.Individuals, &Individual{
			ID:     fmt.Sprintf("gen0-ind%d", i),
			Params: *params,
			Port:   basePort + i,
		})
	}
	return pop
}

// Spawn は全個体をサブプロセスとして起動する
func (p *Population) Spawn(ctx context.Context) error {
	for _, ind := range p.Individuals {
		if err := p.spawnOne(ctx, ind); err != nil {
			return fmt.Errorf("spawn %s: %w", ind.ID, err)
		}
	}

	// ヘルスチェック: 全個体が応答するまで待機
	for _, ind := range p.Individuals {
		if err := waitReady(ctx, ind.Port, 30*time.Second); err != nil {
			return fmt.Errorf("healthcheck %s (port %d): %w", ind.ID, ind.Port, err)
		}
	}
	return nil
}

func (p *Population) spawnOne(ctx context.Context, ind *Individual) error {
	// データディレクトリ作成
	ind.DataDir = filepath.Join(p.BaseDir, ind.ID)
	if err := os.MkdirAll(ind.DataDir, 0755); err != nil {
		return err
	}

	// char_setting.md をコピー
	if p.CharSettingPath != "" {
		charData, err := os.ReadFile(p.CharSettingPath)
		if err != nil {
			return fmt.Errorf("char_setting.md読み込み失敗: %w", err)
		}
		if err := os.WriteFile(filepath.Join(ind.DataDir, "char_setting.md"), charData, 0644); err != nil {
			return err
		}
	}

	// パラメータJSONを書き出し
	paramPath := filepath.Join(ind.DataDir, "params.json")
	data, err := json.MarshalIndent(ind.Params, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(paramPath, data, 0644); err != nil {
		return err
	}

	// サブプロセス起動
	addr := fmt.Sprintf(":%d", ind.Port)
	cmd := exec.CommandContext(ctx, p.Binary,
		"-api", addr,
		"-config", paramPath,
		"-data", ind.DataDir,
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return err
	}
	ind.Process = cmd
	return nil
}

// Shutdown は全個体のプロセスを停止する
func (p *Population) Shutdown() {
	for _, ind := range p.Individuals {
		if ind.Process != nil && ind.Process.Process != nil {
			ind.Process.Process.Signal(os.Interrupt)
		}
	}
	// 猶予をもって待機
	for _, ind := range p.Individuals {
		if ind.Process != nil {
			done := make(chan error, 1)
			go func() { done <- ind.Process.Wait() }()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				if ind.Process.Process != nil {
					ind.Process.Process.Kill()
				}
			}
		}
	}
}

// CleanupData は全個体のデータディレクトリを削除する
func (p *Population) CleanupData() {
	for _, ind := range p.Individuals {
		if ind.DataDir != "" {
			os.RemoveAll(ind.DataDir)
		}
	}
}

// waitReady はポートが応答するまでポーリングする
func waitReady(ctx context.Context, port int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	addr := fmt.Sprintf("http://localhost:%d/api/thoughts", port)

	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// TCP接続チェック
		conn, err := net.DialTimeout("tcp", fmt.Sprintf("localhost:%d", port), time.Second)
		if err != nil {
			time.Sleep(500 * time.Millisecond)
			continue
		}
		conn.Close()

		// HTTP応答チェック
		resp, err := http.Get(addr)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("timeout waiting for port %d", port)
}

// mutateParams はパラメータにガウス摂動を加える
// σは各パラメータの範囲のrate倍
func mutateParams(p *config.Params, rate float64) {
	perturb := func(v, lo, hi float64) float64 {
		sigma := (hi - lo) * rate
		v += rand.NormFloat64() * sigma
		if v < lo {
			v = lo
		}
		if v > hi {
			v = hi
		}
		return v
	}
	perturbInt := func(v, lo, hi int) int {
		sigma := float64(hi-lo) * rate
		nv := float64(v) + rand.NormFloat64()*sigma
		if nv < float64(lo) {
			return lo
		}
		if nv > float64(hi) {
			return hi
		}
		return int(nv)
	}

	// Intervals (60〜180秒 — GA評価時はOllama負荷を抑えるため下限を高く)
	p.Intervals.Frontal = perturb(p.Intervals.Frontal, 60, 180)
	p.Intervals.Temporal = perturb(p.Intervals.Temporal, 60, 180)
	p.Intervals.Hippocampus = perturb(p.Intervals.Hippocampus, 60, 180)
	p.Intervals.Hypothesis = perturb(p.Intervals.Hypothesis, 60, 180)
	p.Intervals.Critic = perturb(p.Intervals.Critic, 60, 180)
	p.Intervals.Curiosity = perturb(p.Intervals.Curiosity, 60, 180)
	p.Intervals.Grounding = perturb(p.Intervals.Grounding, 60, 180)
	p.Intervals.Bonnou = perturb(p.Intervals.Bonnou, 60, 180)
	p.Intervals.Modulator = perturb(p.Intervals.Modulator, 60, 180)

	// Gain
	p.Gain.SkipThreshold = perturb(p.Gain.SkipThreshold, 0.05, 0.8)
	p.Gain.MaxGain = perturb(p.Gain.MaxGain, 1.0, 5.0)
	p.Gain.IdleDecayStart = perturb(p.Gain.IdleDecayStart, 10, 300)
	p.Gain.RecoverStep = perturb(p.Gain.RecoverStep, 0.01, 0.5)
	p.Gain.DecayStep = perturb(p.Gain.DecayStep, 0.01, 0.5)

	// Stagnation
	p.Stagnation.DecayFrontal = perturb(p.Stagnation.DecayFrontal, 0.05, 0.8)
	p.Stagnation.DecayTemporal = perturb(p.Stagnation.DecayTemporal, 0.05, 0.8)
	p.Stagnation.DecayHypothesis = perturb(p.Stagnation.DecayHypothesis, 0.05, 0.8)
	p.Stagnation.BoostCuriosity = perturb(p.Stagnation.BoostCuriosity, 0.05, 0.8)
	p.Stagnation.BoostGrounding = perturb(p.Stagnation.BoostGrounding, 0.05, 0.8)
	p.Stagnation.NoveltyRecovery = perturb(p.Stagnation.NoveltyRecovery, 0.05, 0.8)

	// Hypothesis
	p.Hypothesis.DedupJaccard = perturb(p.Hypothesis.DedupJaccard, 0.3, 0.9)
	p.Hypothesis.SentimentPositive = perturb(p.Hypothesis.SentimentPositive, 0.1, 0.8)
	p.Hypothesis.SentimentNegative = perturb(p.Hypothesis.SentimentNegative, -0.8, -0.1)

	// PredictionError
	p.PredictionError.HighThreshold = perturb(p.PredictionError.HighThreshold, 0.5, 1.0)
	p.PredictionError.MediumThreshold = perturb(p.PredictionError.MediumThreshold, 0.2, 0.8)

	// Attention
	p.Attention.SalienceThreshold = perturb(p.Attention.SalienceThreshold, 0.1, 0.8)
	p.Attention.Baseline = perturb(p.Attention.Baseline, 0.1, 0.9)
	p.Attention.GoalWeight = perturb(p.Attention.GoalWeight, 0.05, 0.8)
	p.Attention.RepetitionPenalty = perturb(p.Attention.RepetitionPenalty, 0.05, 0.8)
	p.Attention.MetaPenalty = perturb(p.Attention.MetaPenalty, 0.05, 0.8)
	p.Attention.UserBonus = perturb(p.Attention.UserBonus, 0.05, 0.8)

	// Critic
	p.Critic.RepetitionJaccard = perturb(p.Critic.RepetitionJaccard, 0.3, 0.9)
	p.Critic.StagnationRepRate = perturb(p.Critic.StagnationRepRate, 0.2, 0.9)
	p.Critic.StagnationDominance = perturb(p.Critic.StagnationDominance, 0.3, 0.9)

	// Curiosity
	p.Curiosity.FrequentThreshold = perturbInt(p.Curiosity.FrequentThreshold, 1, 10)
	p.Curiosity.RareThreshold = perturbInt(p.Curiosity.RareThreshold, 1, 5)

	// Novelty
	p.Novelty.ScoreThreshold = perturb(p.Novelty.ScoreThreshold, 0.5, 1.0)

	// Bonnou
	p.Bonnou.HotWordRetention = perturb(p.Bonnou.HotWordRetention, 30, 600)
	p.Bonnou.IntensityThreshold = perturb(p.Bonnou.IntensityThreshold, 0.1, 0.9)

	// History
	p.History.FreshCount = perturbInt(p.History.FreshCount, 1, 10)

	// Personality
	p.Personality.Warmth = perturb(p.Personality.Warmth, 0.0, 1.0)
	p.Personality.Directness = perturb(p.Personality.Directness, 0.0, 1.0)
	p.Personality.Humor = perturb(p.Personality.Humor, 0.0, 1.0)
	p.Personality.Curiosity = perturb(p.Personality.Curiosity, 0.0, 1.0)
	p.Personality.Verbosity = perturb(p.Personality.Verbosity, 0.0, 1.0)
	p.Personality.Empathy = perturb(p.Personality.Empathy, 0.0, 1.0)
}
