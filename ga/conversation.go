package ga

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
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

// topicShifts はループ検知時に注入する話題転換メッセージ
var topicShifts = []string{
	"ところで、最近なにか面白いこと考えた？",
	"話変わるけど、君にとって大事なものってなに？",
	"ふと思ったんだけど、ボクたちって何のために話してるんだろう？",
	"そういえば、君って怒ることある？",
	"ねえ、もし明日世界が終わるとしたら何する？",
	"急に聞くけど、君が一番苦手なことってなに？",
	"なんか今の話、ループしてない？別の話しようよ。",
	"ちょっと深い話していい？　君は自分のこと好き？",
}

// isLooping は直近の応答がループしているか判定する
func isLooping(history []string) bool {
	n := len(history)
	if n < 3 {
		return false
	}
	// 直近3つのうち2つ以上が類似していたらループ
	similar := 0
	for i := n - 3; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if textSimilar(history[i], history[j]) {
				similar++
			}
		}
	}
	return similar >= 2
}

// textSimilar は2つのテキストが似ているか簡易判定する
func textSimilar(a, b string) bool {
	a = strings.TrimSpace(a)
	b = strings.TrimSpace(b)
	if a == b {
		return true
	}
	// 短いテキスト同士で片方がもう片方を含む
	if len(a) > 0 && len(b) > 0 {
		if strings.Contains(a, b) || strings.Contains(b, a) {
			return true
		}
	}
	// 共通語の割合で判定
	wordsA := strings.Fields(a)
	wordsB := strings.Fields(b)
	if len(wordsA) == 0 || len(wordsB) == 0 {
		return false
	}
	set := make(map[string]bool)
	for _, w := range wordsA {
		set[w] = true
	}
	common := 0
	for _, w := range wordsB {
		if set[w] {
			common++
		}
	}
	shorter := len(wordsA)
	if len(wordsB) < shorter {
		shorter = len(wordsB)
	}
	return shorter > 0 && float64(common)/float64(shorter) > 0.7
}

// Converse は2体のlmindインスタンス間でN往復の対話を行う
// seed は最初のメッセージ。A→B→A→B... の順で対話する。
// ループを検知したら話題転換メッセージを注入する。
func Converse(ctx context.Context, a, b *Individual, rounds int, seed string) error {
	msg := seed
	var history []string
	for i := range rounds {
		var sender, receiver *Individual
		if i%2 == 0 {
			sender, receiver = a, b
		} else {
			sender, receiver = b, a
		}

		// ループ検知 → 話題転換
		if isLooping(history) {
			msg = topicShifts[rand.Intn(len(topicShifts))]
			fmt.Printf("      [loop→shift] %s→%s: %s\n", sender.ID, receiver.ID, msg)
		}

		resp, err := sendChat(ctx, receiver.Port, msg)
		if err != nil {
			return fmt.Errorf("round %d (%s→%s): %w", i+1, sender.ID, receiver.ID, err)
		}
		msg = resp
		if msg == "" {
			msg = "..." // 空応答時のフォールバック
		}
		history = append(history, msg)
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
