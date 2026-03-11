package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"
)

type Client struct {
	hosts      []string
	httpClient *http.Client
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

// New はOllamaクライアントを作成する。
// OLLAMA_HOSTS 環境変数にカンマ区切りで複数ホストを指定するとランダムに分散する。
// 例: OLLAMA_HOSTS=http://localhost:11434,http://192.168.50.47:11434
func New(baseURL string) *Client {
	var hosts []string

	if baseURL != "" {
		hosts = []string{baseURL}
	}

	if len(hosts) == 0 {
		if env := os.Getenv("OLLAMA_HOSTS"); env != "" {
			for _, h := range strings.Split(env, ",") {
				h = strings.TrimSpace(h)
				if h != "" {
					hosts = append(hosts, h)
				}
			}
		}
	}

	if len(hosts) == 0 {
		if env := os.Getenv("OLLAMA_HOST"); env != "" {
			hosts = []string{env}
		}
	}

	if len(hosts) == 0 {
		hosts = []string{"http://localhost:11434"}
	}

	return &Client{
		hosts: hosts,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

// Hosts は接続先のホスト一覧を返す
func (c *Client) Hosts() []string {
	return c.hosts
}

// shuffledHosts はホスト一覧をランダム順で返す
func (c *Client) shuffledHosts() []string {
	shuffled := make([]string, len(c.hosts))
	copy(shuffled, c.hosts)
	rand.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	return shuffled
}

// doPost はPOSTリクエストを送信し、失敗時に別ホストへフォールバックする
func (c *Client) doPost(ctx context.Context, path string, body []byte) ([]byte, error) {
	var lastErr error
	for _, host := range c.shuffledHosts() {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, host+path, bytes.NewReader(body))
		if err != nil {
			lastErr = fmt.Errorf("create request (%s): %w", host, err)
			continue
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := c.httpClient.Do(httpReq)
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
