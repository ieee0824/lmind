package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/ieee0824/lmind/brain"
	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/chat"
	"github.com/ieee0824/lmind/config"
	"github.com/ieee0824/lmind/logger"
	"github.com/ieee0824/lmind/ollama"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	// データディレクトリ
	dataDir, err := os.UserHomeDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
	dataDir = filepath.Join(dataDir, ".lmind")
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}

	// 性格設定を読み込み（char_setting.md）
	personality, err := config.LoadPersonality(filepath.Join(dataDir, "char_setting.md"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}

	// SQLiteログ初期化（expire: 1週間）
	lg, err := logger.New(filepath.Join(dataDir, "thoughts.db"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "fatal: logger初期化失敗: %v\n", err)
		os.Exit(1)
	}
	defer lg.Close()

	client := ollama.New("")
	lg.Info("system", fmt.Sprintf("Ollamaホスト: %s", strings.Join(client.Hosts(), ", ")))

	thoughtBus := bus.New()
	history := bus.NewHistory(100)

	// 古い思考の要約関数を設定（gemma3:1bで軽量に）
	history.SetSummarizer(func(content string) string {
		resp, err := client.Chat(ctx, ollama.ChatRequest{
			Model: "gemma3:1b",
			Messages: []ollama.Message{
				{Role: "system", Content: "与えられたテキストを1文に要約してください。要約だけを出力。"},
				{Role: "user", Content: content},
			},
		})
		if err != nil {
			// エラー時はフォールバック: 80文字切り詰め
			runes := []rune(content)
			if len(runes) > 80 {
				return string(runes[:80]) + "…"
			}
			return content
		}
		return resp.Message.Content
	})

	// 思考バスの全メッセージを履歴に記録 + SQLiteに永続化
	recorder := thoughtBus.Subscribe("_recorder")
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case t := <-recorder:
				history.Record(t)
				lg.Info(t.From, t.Content)
			}
		}
	}()

	// 前頭葉: 推論・判断・統合を担当（重いモデル）
	frontal := brain.NewRegion(brain.RegionConfig{
		Name:         "frontal",
		Model:        "gemma3:4b",
		SystemPrompt: personality.FrontalPrompt(),
		Interval:     30 * time.Second,
		Client:       client,
		Bus:          thoughtBus,
		History:      history,
		Logger:       lg,
	})

	// 側頭葉: 連想・記憶想起・パターン認識を担当（軽いモデル）
	temporal := brain.NewRegion(brain.RegionConfig{
		Name:         "temporal",
		Model:        "gemma3:1b",
		SystemPrompt: personality.TemporalPrompt(),
		Interval:     20 * time.Second,
		Client:       client,
		Bus:          thoughtBus,
		History:      history,
		Logger:       lg,
	})

	// 海馬: 記憶の形成・想起を担当（memAI-go使用）
	store := brain.NewInMemoryStore()
	hippocampus := brain.NewHippocampus(brain.HippocampusConfig{
		Model:   "gemma3:1b",
		Client:  client,
		Bus:     thoughtBus,
		History: history,
		Logger:  lg,
		Store:   store,
		EmbeddingFn: func(ctx context.Context, text string) ([]float64, error) {
			return client.Embedding(ctx, "gemma3:1b", text)
		},
		Interval: 25 * time.Second,
	})

	// 脳部位の思考ループを開始
	go frontal.Run(ctx)
	go temporal.Run(ctx)
	go hippocampus.Run(ctx)

	lg.Info("system", "lmind起動")

	// 初期思考を投入（思考のきっかけ）
	thoughtBus.Publish(bus.Thought{
		From:    "system",
		Content: "システム起動。自由に思考を始めてください。",
	})

	// チャットインターフェースを起動（メインgoroutine）
	chatServer := chat.New(chat.ServerConfig{
		Client:           client,
		Model:            "gemma3:4b",
		JudgeModel:       "gemma3:1b",
		Bus:              thoughtBus,
		History:          history,
		Logger:           lg,
		ChatPrompt:       personality.BrocaChatPrompt(),
		TaskPrompt:       personality.BrocaTaskPrompt(),
		ModeJudgePrompt:  personality.ModeJudgePrompt(),
		InhibitionPrompt: personality.InhibitionPrompt(),
	})
	chatServer.Run(ctx)
}
