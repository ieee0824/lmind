package chat

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/ollama"
)

// Server はCLIベースのチャットインターフェース
type Server struct {
	client  *ollama.Client
	model   string
	bus     *bus.ThoughtBus
	history *bus.History
	inbox   <-chan bus.Thought
}

func New(client *ollama.Client, model string, b *bus.ThoughtBus, history *bus.History) *Server {
	s := &Server{
		client:  client,
		model:   model,
		bus:     b,
		history: history,
	}
	s.inbox = b.Subscribe("chat")
	return s
}

// Run はユーザーからの入力を待ち受ける
func (s *Server) Run(ctx context.Context) {
	// バスからの思考を消費（溜めないように）
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case <-s.inbox:
				// チャット部位は受信した思考を無視（historyに記録済み）
			}
		}
	}()

	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("lmind chat (type 'quit' to exit, 'thoughts' to see recent thinking)")
	fmt.Print("> ")

	for scanner.Scan() {
		input := strings.TrimSpace(scanner.Text())
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

		// 現在の思考文脈を使ってLLMに応答させる
		response := s.respond(ctx, input)
		fmt.Printf("\n%s\n\n> ", response)
	}
}

func (s *Server) respond(ctx context.Context, question string) string {
	recent := s.history.Recent(20)

	var thoughtContext strings.Builder
	for _, t := range recent {
		if t.From == "user" {
			continue
		}
		fmt.Fprintf(&thoughtContext, "[%s] %s\n", t.From, t.Content)
	}

	prompt := question
	if thoughtContext.Len() > 0 {
		prompt = fmt.Sprintf("現在の内部思考の文脈:\n%s\n\nユーザーの質問: %s\n\n内部思考を踏まえた上で、ユーザーに分かりやすく回答してください。",
			thoughtContext.String(), question)
	}

	resp, err := s.client.Chat(ctx, ollama.ChatRequest{
		Model: s.model,
		Messages: []ollama.Message{
			{Role: "system", Content: "あなたはlmindの対話インターフェースです。内部で複数の脳部位が常に思考しています。その思考の文脈を活かしてユーザーに応答してください。"},
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		log.Printf("[chat] エラー: %v", err)
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
