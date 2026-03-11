package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type hostEntry struct {
	url    string
	weight int // 重み（大きいほど優先）
}

type Client struct {
	hosts      []hostEntry
	httpClient *http.Client
	debug      bool

	// ホストごとの実行中リクエスト数（weighted least-connections用）
	inflight []atomic.Int64
	// ホストごとの累計リクエスト数
	totalReqs []atomic.Int64
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

type ChatResponse struct {
	Message Message `json:"message"`
}

// parseHost は "http://host:port*weight" 形式をパースする。
// 重み省略時はデフォルト1。
func parseHost(raw string) hostEntry {
	raw = strings.TrimSpace(raw)
	if idx := strings.LastIndex(raw, "*"); idx > 0 {
		url := raw[:idx]
		if w, err := strconv.Atoi(raw[idx+1:]); err == nil && w > 0 {
			return hostEntry{url: url, weight: w}
		}
	}
	return hostEntry{url: raw, weight: 1}
}

// New はOllamaクライアントを作成する。
// OLLAMA_HOSTS 環境変数にカンマ区切りで複数ホストを指定すると
// weighted least-connections方式で分散する。
// 重み指定: http://host:port*3 （省略時は1）
// 例: OLLAMA_HOSTS=http://localhost:11434*3,http://192.168.50.47:11434*1
func New(baseURL string) *Client {
	var hosts []hostEntry

	if baseURL != "" {
		hosts = []hostEntry{parseHost(baseURL)}
	}

	if len(hosts) == 0 {
		if env := os.Getenv("OLLAMA_HOSTS"); env != "" {
			for _, h := range strings.Split(env, ",") {
				h = strings.TrimSpace(h)
				if h != "" {
					hosts = append(hosts, parseHost(h))
				}
			}
		}
	}

	if len(hosts) == 0 {
		if env := os.Getenv("OLLAMA_HOST"); env != "" {
			hosts = []hostEntry{parseHost(env)}
		}
	}

	if len(hosts) == 0 {
		hosts = []hostEntry{{url: "http://localhost:11434", weight: 1}}
	}

	return &Client{
		hosts: hosts,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
		debug:     os.Getenv("LMIND_DEBUG") != "",
		inflight:  make([]atomic.Int64, len(hosts)),
		totalReqs: make([]atomic.Int64, len(hosts)),
	}
}

// Hosts は接続先のホスト一覧を返す
func (c *Client) Hosts() []string {
	result := make([]string, len(c.hosts))
	for i, h := range c.hosts {
		if h.weight != 1 {
			result[i] = fmt.Sprintf("%s*%d", h.url, h.weight)
		} else {
			result[i] = h.url
		}
	}
	return result
}

// Stats はホストごとのリクエスト統計を返す
func (c *Client) Stats() []HostStats {
	stats := make([]HostStats, len(c.hosts))
	for i, h := range c.hosts {
		stats[i] = HostStats{
			Host:     h.url,
			Weight:   h.weight,
			Inflight: c.inflight[i].Load(),
			Total:    c.totalReqs[i].Load(),
		}
	}
	return stats
}

type HostStats struct {
	Host     string
	Weight   int
	Inflight int64
	Total    int64
}

// leastLoadedHosts はweighted inflight（inflight/weight）が少ない順にホストインデックスを返す
func (c *Client) leastLoadedHosts() []int {
	type hostLoad struct {
		index        int
		weightedLoad float64 // inflight / weight（小さいほど空いている）
	}
	loads := make([]hostLoad, len(c.hosts))
	for i := range c.hosts {
		loads[i] = hostLoad{
			index:        i,
			weightedLoad: float64(c.inflight[i].Load()) / float64(c.hosts[i].weight),
		}
	}
	// 安定ソート: weighted load少ない順
	for i := 1; i < len(loads); i++ {
		for j := i; j > 0 && loads[j].weightedLoad < loads[j-1].weightedLoad; j-- {
			loads[j], loads[j-1] = loads[j-1], loads[j]
		}
	}
	indices := make([]int, len(loads))
	for i, l := range loads {
		indices[i] = l.index
	}
	return indices
}

// doPost はPOSTリクエストを送信し、失敗時に別ホストへフォールバックする。
// weighted least-connections方式: inflight/weightが最も小さいホストを優先する。
func (c *Client) doPost(ctx context.Context, path string, body []byte) ([]byte, error) {
	var lastErr error
	for _, idx := range c.leastLoadedHosts() {
		host := c.hosts[idx].url

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, host+path, bytes.NewReader(body))
		if err != nil {
			lastErr = fmt.Errorf("create request (%s): %w", host, err)
			continue
		}
		httpReq.Header.Set("Content-Type", "application/json")

		c.inflight[idx].Add(1)
		c.totalReqs[idx].Add(1)
		resp, err := c.httpClient.Do(httpReq)
		c.inflight[idx].Add(-1)

		if err != nil {
			lastErr = fmt.Errorf("do request (%s): %w", host, err)
			continue
		}

		respBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("ollama returned %d (%s): %s", resp.StatusCode, host, string(respBody))
			continue
		}
		if err != nil {
			lastErr = fmt.Errorf("read response (%s): %w", host, err)
			continue
		}

		if c.debug && len(c.hosts) > 1 {
			fmt.Fprintf(os.Stderr, "[ollama] %s → %s (w=%d, inflight: %d, total: %d)\n",
				path, host, c.hosts[idx].weight, c.inflight[idx].Load(), c.totalReqs[idx].Load())
		}

		return respBody, nil
	}
	return nil, lastErr
}

func (c *Client) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	req.Stream = false

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	respBody, err := c.doPost(ctx, "/api/chat", body)
	if err != nil {
		return nil, err
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &chatResp, nil
}

// Embedding はOllamaのembedding APIを呼び出す
func (c *Client) Embedding(ctx context.Context, model, text string) ([]float64, error) {
	body, _ := json.Marshal(map[string]string{
		"model":  model,
		"prompt": text,
	})

	respBody, err := c.doPost(ctx, "/api/embeddings", body)
	if err != nil {
		return nil, err
	}

	var result struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return result.Embedding, nil
}

// StatsReporter は定期的にホスト統計をログ出力するgoroutineを起動する
func (c *Client) StatsReporter(ctx context.Context, interval time.Duration, wg *sync.WaitGroup) {
	if len(c.hosts) <= 1 || !c.debug {
		return
	}
	if wg != nil {
		wg.Add(1)
	}
	go func() {
		if wg != nil {
			defer wg.Done()
		}
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				var parts []string
				for _, s := range c.Stats() {
					parts = append(parts, fmt.Sprintf("%s(w=%d, inflight=%d, total=%d)", s.Host, s.Weight, s.Inflight, s.Total))
				}
				fmt.Fprintf(os.Stderr, "[ollama-stats] %s\n", strings.Join(parts, " | "))
			}
		}
	}()
}
