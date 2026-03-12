package brain

import (
	"strings"
	"sync"
	"time"

	"github.com/ieee0824/lmind/bus"
)

// MetricsSnapshot はGA適応度計算用のメトリクス
type MetricsSnapshot struct {
	ThoughtRepetitionRate  float64 `json:"thought_repetition_rate"`
	MetaSelfReferenceCount int     `json:"meta_self_reference_count"`
	UniqueTopicCount       int     `json:"unique_topic_count"`
	AvgResponseTimeMs      int64   `json:"avg_response_time_ms"`
	TotalThoughts          int     `json:"total_thoughts"`
}

// Metrics は思考バスを購読してメトリクスを蓄積する
type Metrics struct {
	bus     *bus.ThoughtBus
	history *bus.History
	inbox   <-chan bus.Thought

	mu              sync.Mutex
	totalThoughts   int
	metaSelfRef     int
	uniqueWords     map[string]bool
	pairSimilarities []float64 // 連続思考間のJaccard類似度
	lastContent     string

	// user→broca応答時間
	responseTimes []time.Duration
	userTimestamp  time.Time
}

func NewMetrics(b *bus.ThoughtBus, history *bus.History) *Metrics {
	m := &Metrics{
		bus:         b,
		history:     history,
		uniqueWords: make(map[string]bool),
	}
	m.inbox = b.Subscribe("_metrics")
	return m
}

// メタ自己言及キーワード
var metaKeywords = []string{
	"the model", "the system", "as an ai", "i am a",
	"language model", "i'm programmed", "my programming",
	"i don't have feelings", "i cannot feel",
}

// Run はメトリクス収集ループ（goroutineで起動）
func (m *Metrics) Run(stop <-chan struct{}) {
	for {
		select {
		case <-stop:
			return
		case t := <-m.inbox:
			m.record(t)
		}
	}
}

func (m *Metrics) record(t bus.Thought) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.totalThoughts++

	// ユニークキーワード収集
	words := tokenize(t.Content)
	for _, w := range words {
		m.uniqueWords[w] = true
	}

	// メタ自己言及チェック
	lower := strings.ToLower(t.Content)
	for _, kw := range metaKeywords {
		if strings.Contains(lower, kw) {
			m.metaSelfRef++
			break
		}
	}

	// 連続思考間のJaccard類似度
	if m.lastContent != "" {
		prevWords := tokenize(m.lastContent)
		sim := jaccardSimilarity(prevWords, words)
		m.pairSimilarities = append(m.pairSimilarities, sim)
	}
	m.lastContent = t.Content

	// user→broca応答時間の計測
	if t.From == "user" {
		m.userTimestamp = t.CreatedAt
	} else if t.From == "broca" && !m.userTimestamp.IsZero() {
		elapsed := t.CreatedAt.Sub(m.userTimestamp)
		if elapsed > 0 && elapsed < 5*time.Minute {
			m.responseTimes = append(m.responseTimes, elapsed)
		}
		m.userTimestamp = time.Time{}
	}
}

// Snapshot は現在のメトリクスを返す
func (m *Metrics) Snapshot() MetricsSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()

	var repRate float64
	if len(m.pairSimilarities) > 0 {
		sum := 0.0
		for _, s := range m.pairSimilarities {
			sum += s
		}
		repRate = sum / float64(len(m.pairSimilarities))
	}

	var avgResp int64
	if len(m.responseTimes) > 0 {
		var total time.Duration
		for _, d := range m.responseTimes {
			total += d
		}
		avgResp = (total / time.Duration(len(m.responseTimes))).Milliseconds()
	}

	return MetricsSnapshot{
		ThoughtRepetitionRate:  repRate,
		MetaSelfReferenceCount: m.metaSelfRef,
		UniqueTopicCount:       len(m.uniqueWords),
		AvgResponseTimeMs:      avgResp,
		TotalThoughts:          m.totalThoughts,
	}
}
