package main

import (
	"context"
	"flag"
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
	apiAddr := flag.String("api", "", "Web APIモードで起動 (例: :8080)")
	flag.Parse()

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

	// 複数ホスト時は60秒ごとに分散統計をログ出力
	client.StatsReporter(ctx, 60*time.Second, nil)

	thoughtBus := bus.New()
	history := bus.NewHistory(100)

	// 古い思考の要約関数を設定（gemma3:1bで軽量に）
	history.SetSummarizer(func(content string) string {
		resp, err := client.Chat(ctx, ollama.ChatRequest{
			Model: "gemma3:1b",
			Messages: []ollama.Message{
				{Role: "system", Content: "Summarize the given text in one sentence. Output only the summary."},
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

	// 明示的な内部状態（goal/hypothesis）— SQLiteから前回のgoalを復元
	mindState := brain.NewState()
	mindState.RestoreFrom(lg)

	// モジュールゲイン管理（ループ時に動的調整）
	modulation := brain.NewModulation()

	// State更新モジュール（ユーザー入力→goal、前頭葉→hypothesis）
	stateUpdater := brain.NewStateUpdater(brain.StateUpdaterConfig{
		State:   mindState,
		Client:  client,
		Model:   "gemma3:1b",
		Bus:     thoughtBus,
		History: history,
		Logger:  lg,
	})

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
		State:        mindState,
		Modulation:   modulation,
	})

	// 側頭葉: 連想・記憶想起・パターン認識を担当（軽いモデル）
	temporal := brain.NewRegion(brain.RegionConfig{
		Name:         "temporal",
		Model:        "gemma3:1b",
		SystemPrompt: personality.TemporalPrompt(),
		Interval:     30 * time.Second,
		Client:       client,
		Bus:          thoughtBus,
		History:      history,
		Logger:       lg,
		State:        mindState,
		Modulation:   modulation,
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
			return client.Embedding(ctx, "nomic-embed-text", text)
		},
		Interval: 25 * time.Second,
	})

	// 仮説生成モジュール: 前頭葉・側頭葉の出力から仮説を生成・更新
	hypothesisMod := brain.NewHypothesis(brain.HypothesisConfig{
		Model:    "gemma3:4b",
		Client:   client,
		Bus:      thoughtBus,
		History:  history,
		Logger:     lg,
		State:      mindState,
		Modulation: modulation,
		Interval:   35 * time.Second,
	})

	// 予測モジュール: ユーザーの次の発言を予測
	predictionMod := brain.NewPrediction(brain.PredictionConfig{
		Model:   "gemma3:1b",
		Client:  client,
		Bus:     thoughtBus,
		History: history,
		Logger:  lg,
		State:   mindState,
	})

	// メタモジュール（LLM不使用・アルゴリズム駆動）
	identityMod := brain.NewIdentity(brain.IdentityConfig{
		Bus:    thoughtBus,
		Logger: lg,
	})
	noveltyMod := brain.NewNovelty(brain.NoveltyConfig{
		Bus:     thoughtBus,
		History: history,
		Logger:  lg,
	})
	criticMod := brain.NewCritic(brain.CriticConfig{
		Bus:      thoughtBus,
		History:  history,
		Logger:   lg,
		Interval: 45 * time.Second,
	})
	curiosityMod := brain.NewCuriosity(brain.CuriosityConfig{
		Bus:      thoughtBus,
		History:  history,
		Logger:   lg,
		Interval: 60 * time.Second,
	})
	groundingMod := brain.NewGrounding(brain.GroundingConfig{
		Bus:      thoughtBus,
		Logger:   lg,
		Interval: 40 * time.Second,
		Hosts:    client.Hosts(),
	})
	bonnouMod := brain.NewBonnou(brain.BonnouConfig{
		Bus:      thoughtBus,
		History:  history,
		Logger:   lg,
		Interval: 50 * time.Second,
	})
	sentimentMod := brain.NewSentiment(brain.SentimentConfig{
		Bus:        thoughtBus,
		Logger:     lg,
		State:      mindState,
		Modulation: modulation,
	})
	modulatorMod := brain.NewModulator(brain.ModulatorConfig{
		Modulation: modulation,
		Bus:        thoughtBus,
		History:    history,
		Logger:     lg,
		Interval:   30 * time.Second,
	})
	predictionErrorMod := brain.NewPredictionError(brain.PredictionErrorConfig{
		Bus:    thoughtBus,
		Logger: lg,
		State:  mindState,
	})
	attentionMod := brain.NewAttention(brain.AttentionConfig{
		Bus:     thoughtBus,
		History: history,
		Logger:  lg,
		State:   mindState,
	})
	rapportMod := brain.NewRapport(brain.RapportConfig{
		Bus:    thoughtBus,
		Logger: lg,
		Store:  lg,
	})

	// 脳部位の思考ループを開始
	go stateUpdater.Run(ctx)
	go frontal.Run(ctx)
	go temporal.Run(ctx)
	go hypothesisMod.Run(ctx)
	go predictionMod.Run(ctx)
	go hippocampus.Run(ctx)
	go identityMod.Run(ctx)
	go noveltyMod.Run(ctx)
	go criticMod.Run(ctx)
	go curiosityMod.Run(ctx)
	go groundingMod.Run(ctx)
	go modulatorMod.Run(ctx)
	go bonnouMod.Run(ctx)
	go sentimentMod.Run(ctx)
	go predictionErrorMod.Run(ctx)
	go attentionMod.Run(ctx)
	go rapportMod.Run(ctx)

	lg.Info("system", "lmind起動")

	// 前回の思考を復元（SQLiteから直近の思考を読み込み）
	pastThoughts, err := lg.RecentThoughts(ctx, 20)
	if err == nil && len(pastThoughts) > 0 {
		for _, entry := range pastThoughts {
			history.Restore(bus.Thought{
				From:    entry.Region,
				Content: entry.Message,
			})
		}
		lg.Info("system", fmt.Sprintf("前回の思考を%d件復元", len(pastThoughts)))
	}

	// 初期思考を投入（思考のきっかけ）
	if len(pastThoughts) == 0 {
		// 初回起動時のみ種を投入
		thoughtBus.Publish(bus.Thought{
			From:    "frontal",
			Content: "A quiet moment. What should I think about?",
		})
	} else {
		// 前回の思考を引き継いで再開
		thoughtBus.Publish(bus.Thought{
			From:    "frontal",
			Content: "Resuming from previous thoughts. What was I thinking about?",
		})
	}

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
		Rapport:          rapportMod,
	})
	if *apiAddr != "" {
		// Web APIモード
		if err := chatServer.RunAPI(ctx, *apiAddr); err != nil {
			fmt.Fprintf(os.Stderr, "API server error: %v\n", err)
			os.Exit(1)
		}
	} else {
		// CLIモード
		chatServer.Run(ctx)
	}
}
