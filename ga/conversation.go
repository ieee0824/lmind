package ga

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// chatRequest はlmind /api/chat のリクエスト
type chatRequest struct {
	Message string `json:"message"`
}

// chatResponse はlmind /api/chat のレスポンス
type chatResponse struct {
	Response string `json:"response"`
}

// Converse は2体のlmindインスタンス間でN往復の対話を行う
// seed は最初のメッセージ。A→B→A→B... の順で対話する。
func Converse(ctx context.Context, a, b *Individual, rounds int, seed string) error {
	msg := seed
	for i := range rounds {
		var sender, receiver *Individual
		if i%2 == 0 {
			sender, receiver = a, b
		} else {
			sender, receiver = b, a
		}

		resp, err := sendChat(ctx, receiver.Port, msg)
		if err != nil {
			return fmt.Errorf("round %d (%s→%s): %w", i+1, sender.ID, receiver.ID, err)
		}
		msg = resp
		if msg == "" {
			msg = "..." // 空応答時のフォールバック
		}
	}
	return nil
}

// sendChat はlmindインスタンスにチャットメッセージを送信する
func sendChat(ctx context.Context, port int, message string) (string, error) {
	url := fmt.Sprintf("http://localhost:%d/api/chat", port)

	body, err := json.Marshal(chatRequest{Message: message})
	if err != nil {
		return "", err
	}

	chatCtx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()

	req, err := http.NewRequestWithContext(chatCtx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("status %d: %s", resp.StatusCode, string(data))
	}

	var cr chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&cr); err != nil {
		return "", err
	}
	return cr.Response, nil
}
