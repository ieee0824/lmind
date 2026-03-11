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
	client  *ollama.Client
	model   string
	bus     *bus.ThoughtBus
	history *bus.History
	logger  *logger.Logger
	inbox   <-chan bus.Thought

	mu         sync.Mutex
	lastActive string // 最後にバスで発言した部位名
}

func New(client *ollama.Client, model string, b *bus.ThoughtBus, history *bus.History, lg *logger.Logger) *Server {
	s := &Server{
		client:  client,
		model:   model,
		bus:     b,
		history: history,
		logger:  lg,
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

const brocaSystemPrompt = `あなたはブローカ野（言語出力部位）です。
あなたの内部では複数の思考プロセスが常に動いています。それらの思考は以下の「内部思考」として与えられます。

あなたの役割：
- 内部思考を統合し、一つの意識・人格として自然に応答すること
- 部位名（frontal, temporal, hippocampus等）やシステム用語は絶対に出力しないこと
- 「私の内部では〜」「脳部位が〜」のようなメタ的な説明はしないこと
- あたかも一人の人間が考えて答えているかのように振る舞うこと
- 内部思考に有用な情報がなければ、自分の知識で普通に答えること`

func (s *Server) respond(ctx context.Context, question string) string {
	recent := s.history.Recent(20)

	var thoughts strings.Builder
	for _, t := range recent {
		if t.From == "user" || t.From == "system" {
			continue
		}
		fmt.Fprintf(&thoughts, "%s\n", t.Content)
	}

	var prompt string
	if thoughts.Len() > 0 {
		prompt = fmt.Sprintf("【内部思考】\n%s\n【ユーザーの発言】\n%s", thoughts.String(), question)
	} else {
		prompt = question
	}

	resp, err := s.client.Chat(ctx, ollama.ChatRequest{
		Model: s.model,
		Messages: []ollama.Message{
			{Role: "system", Content: brocaSystemPrompt},
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		s.logger.Error("broca", fmt.Sprintf("応答エラー: %v", err))
		return fmt.Sprintf("エラー: %v", err)
	}

	return resp.Message.Content
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
