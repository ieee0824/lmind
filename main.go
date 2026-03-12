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
	configPath := flag.String("config", "", "パラメータ設定JSONファイル")
	dataPath := flag.String("data", "", "データディレクトリ (デフォルト: ~/.lmind)")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	// パラメータ読み込み
	var params *config.Params
	if *configPath != "" {
		var err error
		params, err = config.LoadParams(*configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
			os.Exit(1)
		}
	} else {
		params = config.DefaultParams()
	}

	// データディレクトリ
	dataDir := *dataPath
	if dataDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
			os.Exit(1)
		}
		dataDir = filepath.Join(home, ".lmind")
	}
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}

	// 性格設定を読み込み（char_setting.md + params由来の性格次元）
	personality, err := config.LoadPersonality(filepath.Join(dataDir, "char_setting.md"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
	personality = personality.WithTraits(&params.Personality)

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
	history.SetFreshCount(params.History.FreshCount)

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
	modulation.Configure(
		params.Gain.SkipThreshold,
		params.Gain.MaxGain,
		time.Duration(params.Gain.IdleDecayStart*float64(time.Second)),
	)

	// ヘルパー: IntervalのDuration変換
	dur := func(seconds float64) time.Duration {
		return time.Duration(seconds * float64(time.Second))
	}

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
		Interval:     dur(params.Intervals.Frontal),
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
		Interval:     dur(params.Intervals.Temporal),
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
		Interval: dur(params.Intervals.Hippocampus),
	})

	// 仮説生成モジュール: 前頭葉・側頭葉の出力から仮説を生成・更新
	hypothesisMod := brain.NewHypothesis(brain.HypothesisConfig{
		Model:             "gemma3:4b",
		Client:            client,
		Bus:               thoughtBus,
		History:           history,
		Logger:            lg,
		State:             mindState,
		Modulation:        modulation,
		Interval:          dur(params.Intervals.Hypothesis),
		DedupJaccard:      params.Hypothesis.DedupJaccard,
		SentimentPositive: params.Hypothesis.SentimentPositive,
		SentimentNegative: params.Hypothesis.SentimentNegative,
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
		Bus:            thoughtBus,
		History:        history,
		Logger:         lg,
		ScoreThreshold: params.Novelty.ScoreThreshold,
	})
	criticMod := brain.NewCritic(brain.CriticConfig{
		Bus:                 thoughtBus,
		History:             history,
		Logger:              lg,
		Interval:            dur(params.Intervals.Critic),
		RepetitionJaccard:   params.Critic.RepetitionJaccard,
		StagnationRepRate:   params.Critic.StagnationRepRate,
		StagnationDominance: params.Critic.StagnationDominance,
	})
	curiosityMod := brain.NewCuriosity(brain.CuriosityConfig{
		Bus:               thoughtBus,
		History:           history,
		Logger:            lg,
		Interval:          dur(params.Intervals.Curiosity),
		FrequentThreshold: params.Curiosity.FrequentThreshold,
		RareThreshold:     params.Curiosity.RareThreshold,
	})
	groundingMod := brain.NewGrounding(brain.GroundingConfig{
		Bus:      thoughtBus,
		Logger:   lg,
		Interval: dur(params.Intervals.Grounding),
		Hosts:    client.Hosts(),
	})
	bonnouMod := brain.NewBonnou(brain.BonnouConfig{
		Bus:                thoughtBus,
		History:            history,
		Logger:             lg,
		Interval:           dur(params.Intervals.Bonnou),
		HotWordRetention:   time.Duration(params.Bonnou.HotWordRetention * float64(time.Second)),
		IntensityThreshold: params.Bonnou.IntensityThreshold,
	})
	sentimentMod := brain.NewSentiment(brain.SentimentConfig{
		Bus:        thoughtBus,
		Logger:     lg,
		State:      mindState,
		Modulation: modulation,
	})
	modulatorMod := brain.NewModulator(brain.ModulatorConfig{
		Modulation:      modulation,
		Bus:             thoughtBus,
		History:         history,
		Logger:          lg,
		Interval:        dur(params.Intervals.Modulator),
		DecayFrontal:    params.Stagnation.DecayFrontal,
		DecayTemporal:   params.Stagnation.DecayTemporal,
		DecayHypothesis: params.Stagnation.DecayHypothesis,
		BoostCuriosity:  params.Stagnation.BoostCuriosity,
		BoostGrounding:  params.Stagnation.BoostGrounding,
		NoveltyRecovery: params.Stagnation.NoveltyRecovery,
		RecoverStep:     params.Gain.RecoverStep,
		DecayStep:       params.Gain.DecayStep,
	})
	predictionErrorMod := brain.NewPredictionError(brain.PredictionErrorConfig{
		Bus:             thoughtBus,
		Logger:          lg,
		State:           mindState,
		HighThreshold:   params.PredictionError.HighThreshold,
		MediumThreshold: params.PredictionError.MediumThreshold,
	})
	attentionMod := brain.NewAttention(brain.AttentionConfig{
		Bus:               thoughtBus,
		History:           history,
		Logger:            lg,
		State:             mindState,
		SalienceThreshold: params.Attention.SalienceThreshold,
		Baseline:          params.Attention.Baseline,
		GoalWeight:        params.Attention.GoalWeight,
		RepetitionPenalty: params.Attention.RepetitionPenalty,
		MetaPenalty:       params.Attention.MetaPenalty,
		UserBonus:         params.Attention.UserBonus,
	})

	// メトリクス収集
	metrics := brain.NewMetrics(thoughtBus, history)
	go metrics.Run(ctx.Done())

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
		Metrics:          metrics,
		Params:           params,
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
