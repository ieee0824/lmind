package chat

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
	"github.com/ieee0824/lmind/ollama"
)

// Server はCLIベースのチャットインターフェース
type Server struct {
	client           *ollama.Client
	model            string
	bus              *bus.ThoughtBus
	history          *bus.History
	logger           *logger.Logger
	systemPrompt     string
	inhibitionPrompt string
	inbox            <-chan bus.Thought

	mu         sync.Mutex
	lastActive string
}

func New(client *ollama.Client, model string, b *bus.ThoughtBus, history *bus.History, lg *logger.Logger, systemPrompt, inhibitionPrompt string) *Server {
	s := &Server{
		client:           client,
		model:            model,
		bus:              b,
		history:          history,
		logger:           lg,
		systemPrompt:     systemPrompt,
		inhibitionPrompt: inhibitionPrompt,
	}
	s.inbox = b.Subscribe("chat")
	return s
}

// Run はユーザーからの入力を待ち受ける
func (s *Server) Run(ctx context.Context) {
	// バスからの思考を消費しつつ、アクティブ部位を追跡
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case t := <-s.inbox:
				s.mu.Lock()
				s.lastActive = t.From
				s.mu.Unlock()

				if strings.HasPrefix(t.Content, "【エラー】") {
					fmt.Fprintf(os.Stderr, "\n%s: %s\n> ", t.From, t.Content)
				}
			}
		}
	}()

	lines := make(chan string)
	go func() {
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			lines <- scanner.Text()
		}
		close(lines)
	}()

	fmt.Println("lmind chat (type 'quit' to exit, 'thoughts' to see recent thinking)")
	fmt.Print("> ")

	for {
		select {
		case <-ctx.Done():
			fmt.Println("\nshutting down...")
			return
		case line, ok := <-lines:
			if !ok {
				return
			}
			input := strings.TrimSpace(line)
			if input == "" {
				fmt.Print("> ")
				continue
			}
			if input == "quit" {
				return
			}
			if input == "thoughts" {
				s.showThoughts()
				fmt.Print("> ")
				continue
			}

			// ユーザーの質問を思考バスに流す
			userThought := bus.Thought{
				From:    "user",
				Content: input,
			}
			s.history.Record(userThought)
			s.bus.Publish(userThought)

			// スピナー表示しながら応答を待つ
			done := make(chan string, 1)
			go func() {
				done <- s.respond(ctx, input)
			}()

			s.showSpinner(ctx, done)
		}
	}
}

// showSpinner はLLM応答待ちの間、アクティブな脳部位名を切り替え表示する
func (s *Server) showSpinner(ctx context.Context, done <-chan string) {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	dots := 0
	for {
		select {
		case <-ctx.Done():
			fmt.Print("\r\033[K")
			return
		case response := <-done:
			fmt.Printf("\r\033[K\n%s\n\n> ", response)
			return
		case <-ticker.C:
			s.mu.Lock()
			active := s.lastActive
			s.mu.Unlock()

			if active == "" {
				active = "chat"
			}
			dots = (dots + 1) % 4
			d := strings.Repeat(".", dots)
			fmt.Printf("\r\033[K[%s] 考え中%s", active, d)
		}
	}
}

const maxRefineRounds = 2

func (s *Server) respond(ctx context.Context, question string) string {
	recent := s.history.Recent(10)

	var thoughts strings.Builder
	for _, t := range recent {
		if t.From == "user" || t.From == "system" {
			continue
		}
		fmt.Fprintf(&thoughts, "%s\n", t.Content)
	}

	var basePrompt string
	if thoughts.Len() > 0 {
		basePrompt = fmt.Sprintf("【ユーザーの発言】\n%s\n\n【参考: 最近の内部思考】\n%s\n\n上記の内部思考は参考程度に。ユーザーの発言に直接答えることを最優先にして。", question, thoughts.String())
	} else {
		basePrompt = question
	}

	var refined string
	prompt := basePrompt

	for i := range maxRefineRounds {
		// Broca起案
		resp, err := s.client.Chat(ctx, ollama.ChatRequest{
			Model: s.model,
			Messages: []ollama.Message{
				{Role: "system", Content: s.systemPrompt},
				{Role: "user", Content: prompt},
			},
		})
		if err != nil {
			s.logger.Error("broca", fmt.Sprintf("応答エラー: %v", err))
			return fmt.Sprintf("エラー: %v", err)
		}
		draft := resp.Message.Content
		s.logger.Info("broca", fmt.Sprintf("起案(%d): %s", i+1, draft))

		// 前頭葉の抑制機能：要約＋地の文除去＋整形
		var err2 error
		refined, err2 = s.refine(ctx, question, draft)
		if err2 != nil {
			s.logger.Error("inhibition", fmt.Sprintf("整形エラー: %v", err2))
			return draft
		}
		s.logger.Info("inhibition", fmt.Sprintf("整形後(%d): %s", i+1, refined))

		// 起案と整形後の差が小さければ収束とみなす
		if s.isSimilarLength(draft, refined) {
			break
		}

		// 差が大きい → 理由付きで差し戻し
		s.logger.Info("inhibition", fmt.Sprintf("差し戻し(%d): 起案が大幅に整形された", i+1))
		prompt = fmt.Sprintf("%s\n\n【前回の起案】\n%s\n【整形後】\n%s\n\n前回の起案は地の文や冗長な表現が多く、上記のように整形された。最初からこのレベルの簡潔さで書き直して。", basePrompt, draft, refined)
	}

	return refined
}

// isSimilarLength は起案と整形後の長さが近いかを判定する
// 整形で半分以下に縮んだら「大幅に違う」とみなす
func (s *Server) isSimilarLength(draft, refined string) bool {
	dl := len([]rune(draft))
	rl := len([]rune(refined))
	if dl == 0 {
		return true
	}
	return float64(rl)/float64(dl) > 0.5
}

// refine は前頭葉の抑制機能としてドラフトを整形する（地の文除去・要約）
// question を渡すことで、ユーザーの質問から離れた整形を防ぐ
func (s *Server) refine(ctx context.Context, question, draft string) (string, error) {
	prompt := fmt.Sprintf("【ユーザーの発言】\n%s\n\n【返答案】\n%s", question, draft)
	resp, err := s.client.Chat(ctx, ollama.ChatRequest{
		Model: s.model,
		Messages: []ollama.Message{
			{Role: "system", Content: s.inhibitionPrompt},
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		return "", err
	}
	return resp.Message.Content, nil
}

func (s *Server) showThoughts() {
	recent := s.history.Recent(10)
	if len(recent) == 0 {
		fmt.Println("(まだ思考がありません)")
		return
	}
	fmt.Println("--- 最近の思考 ---")
	for _, t := range recent {
		fmt.Printf("[%s] %s: %s\n", t.CreatedAt.Format("15:04:05"), t.From, t.Content)
	}
	fmt.Println("---")
}
