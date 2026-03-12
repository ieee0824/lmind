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

// hostPool はホストプール（weighted least-connections用）
type hostPool struct {
	hosts     []hostEntry
	inflight  []atomic.Int64
	totalReqs []atomic.Int64
}

func newHostPool(hosts []hostEntry) *hostPool {
	return &hostPool{
		hosts:     hosts,
		inflight:  make([]atomic.Int64, len(hosts)),
		totalReqs: make([]atomic.Int64, len(hosts)),
	}
}

// leastLoadedHosts はweighted inflight（inflight/weight）が少ない順にホストインデックスを返す
func (p *hostPool) leastLoadedHosts() []int {
	type hostLoad struct {
		index        int
		weightedLoad float64
	}
	loads := make([]hostLoad, len(p.hosts))
	for i := range p.hosts {
		loads[i] = hostLoad{
			index:        i,
			weightedLoad: float64(p.inflight[i].Load()) / float64(p.hosts[i].weight),
		}
	}
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

func (p *hostPool) stats() []HostStats {
	stats := make([]HostStats, len(p.hosts))
	for i, h := range p.hosts {
		stats[i] = HostStats{
			Host:     h.url,
			Weight:   h.weight,
			Inflight: p.inflight[i].Load(),
			Total:    p.totalReqs[i].Load(),
		}
	}
	return stats
}

type Client struct {
	chat       *hostPool // chat用ホストプール
	embed      *hostPool // embedding用ホストプール（nilならchatと共用）
	httpClient *http.Client
	debug      bool
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

type HostStats struct {
	Host     string
	Weight   int
	Inflight int64
	Total    int64
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

func parseHosts(env string) []hostEntry {
	var hosts []hostEntry
	for _, h := range strings.Split(env, ",") {
		h = strings.TrimSpace(h)
		if h != "" {
			hosts = append(hosts, parseHost(h))
		}
	}
	return hosts
}

// New はOllamaクライアントを作成する。
// OLLAMA_HOSTS — chat用ホスト（カンマ区切り、重み指定可）
// OLLAMA_EMBED_HOSTS — embedding専用ホスト（指定時はembeddingをこちらに振る）
// OLLAMA_EMBED_HOSTS未指定時はOLLAMA_HOSTSをembeddingにも使う
// 例:
//
//	OLLAMA_HOSTS=http://localhost:11434*3,http://192.168.50.2:11434
//	OLLAMA_EMBED_HOSTS=http://192.168.50.47:11434,http://localhost:11434
func New(baseURL string) *Client {
	var chatHosts []hostEntry

	if baseURL != "" {
		chatHosts = []hostEntry{parseHost(baseURL)}
	}

	if len(chatHosts) == 0 {
		if env := os.Getenv("OLLAMA_HOSTS"); env != "" {
			chatHosts = parseHosts(env)
		}
	}

	if len(chatHosts) == 0 {
		if env := os.Getenv("OLLAMA_HOST"); env != "" {
			chatHosts = []hostEntry{parseHost(env)}
		}
	}

	if len(chatHosts) == 0 {
		chatHosts = []hostEntry{{url: "http://localhost:11434", weight: 1}}
	}

	c := &Client{
		chat: newHostPool(chatHosts),
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
		debug: os.Getenv("LMIND_DEBUG") != "",
	}

	// embedding専用ホスト
	if env := os.Getenv("OLLAMA_EMBED_HOSTS"); env != "" {
		embedHosts := parseHosts(env)
		if len(embedHosts) > 0 {
			c.embed = newHostPool(embedHosts)
		}
	}

	return c
}

// poolFor は指定パスに対応するホストプールを返す
func (c *Client) poolFor(path string) *hostPool {
	if c.embed != nil && strings.Contains(path, "embed") {
		return c.embed
	}
	return c.chat
}

// Hosts は接続先のホスト一覧を返す
func (c *Client) Hosts() []string {
	var result []string
	for _, h := range c.chat.hosts {
		s := h.url
		if h.weight != 1 {
			s = fmt.Sprintf("%s*%d", h.url, h.weight)
		}
		result = append(result, s)
	}
	return result
}

// Stats はホストごとのリクエスト統計を返す
func (c *Client) Stats() []HostStats {
	stats := c.chat.stats()
	if c.embed != nil {
		stats = append(stats, c.embed.stats()...)
	}
	return stats
}

// doPost はPOSTリクエストを送信し、失敗時に別ホストへフォールバックする。
func (c *Client) doPost(ctx context.Context, path string, body []byte) ([]byte, error) {
	pool := c.poolFor(path)
	var lastErr error
	for _, idx := range pool.leastLoadedHosts() {
		host := pool.hosts[idx].url

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, host+path, bytes.NewReader(body))
		if err != nil {
			lastErr = fmt.Errorf("create request (%s): %w", host, err)
			continue
		}
		httpReq.Header.Set("Content-Type", "application/json")

		pool.inflight[idx].Add(1)
		pool.totalReqs[idx].Add(1)
		resp, err := c.httpClient.Do(httpReq)
		pool.inflight[idx].Add(-1)

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

		if c.debug && len(pool.hosts) > 1 {
			fmt.Fprintf(os.Stderr, "[ollama] %s → %s (w=%d, inflight: %d, total: %d)\n",
				path, host, pool.hosts[idx].weight, pool.inflight[idx].Load(), pool.totalReqs[idx].Load())
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
	totalHosts := len(c.chat.hosts)
	if c.embed != nil {
		totalHosts += len(c.embed.hosts)
	}
	if totalHosts <= 1 || !c.debug {
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
				for _, s := range c.chat.stats() {
					parts = append(parts, fmt.Sprintf("%s(w=%d, inflight=%d, total=%d)", s.Host, s.Weight, s.Inflight, s.Total))
				}
				if c.embed != nil {
					parts = append(parts, "|embed|")
					for _, s := range c.embed.stats() {
						parts = append(parts, fmt.Sprintf("%s(w=%d, inflight=%d, total=%d)", s.Host, s.Weight, s.Inflight, s.Total))
					}
				}
				fmt.Fprintf(os.Stderr, "[ollama-stats] %s\n", strings.Join(parts, " | "))
			}
		}
	}()
}
