package ga

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"

	"github.com/ieee0824/lmind/brain"
)

// Evaluate は個体の適応度を計算する
// /api/metrics からメトリクスを取得し、適応度関数を適用する
func Evaluate(ctx context.Context, ind *Individual) error {
	snapshot, err := fetchMetrics(ctx, ind.Port)
	if err != nil {
		return err
	}

	ind.Fitness = fitness(snapshot)
	return nil
}

// fitness は適応度関数
// 高いほど良い（0〜1の範囲）
func fitness(m brain.MetricsSnapshot) float64 {
	if m.TotalThoughts == 0 {
		return 0
	}

	// 繰り返し率が低いほど良い（weight: 0.4）
	repScore := 1.0 - m.ThoughtRepetitionRate

	// メタ自己言及が少ないほど良い（weight: 0.2）
	metaRate := float64(m.MetaSelfReferenceCount) / float64(m.TotalThoughts)
	metaScore := 1.0 - metaRate

	// ユニークトピックが多いほど良い（weight: 0.2, 20で飽和）
	topicScore := math.Min(1.0, float64(m.UniqueTopicCount)/20.0)

	// 応答時間が短いほど良い（weight: 0.2, 30秒で0）
	respScore := 1.0 - math.Min(1.0, float64(m.AvgResponseTimeMs)/30000.0)

	return 0.4*repScore + 0.2*metaScore + 0.2*topicScore + 0.2*respScore
}

// fetchMetrics はlmindインスタンスからメトリクスを取得する
func fetchMetrics(ctx context.Context, port int) (brain.MetricsSnapshot, error) {
	url := fmt.Sprintf("http://localhost:%d/api/metrics", port)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return brain.MetricsSnapshot{}, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return brain.MetricsSnapshot{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return brain.MetricsSnapshot{}, fmt.Errorf("metrics: status %d", resp.StatusCode)
	}

	var snapshot brain.MetricsSnapshot
	if err := json.NewDecoder(resp.Body).Decode(&snapshot); err != nil {
		return brain.MetricsSnapshot{}, err
	}
	return snapshot, nil
}
