package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"time"

	"github.com/ieee0824/lmind/ga"
)

// Result は世代ごとの結果
type Result struct {
	Generation int          `json:"generation"`
	Best       ResultEntry  `json:"best"`
	All        []ResultEntry `json:"all"`
}

type ResultEntry struct {
	ID      string  `json:"id"`
	Fitness float64 `json:"fitness"`
}

func main() {
	population := flag.Int("population", 4, "個体数")
	generations := flag.Int("generations", 5, "世代数")
	rounds := flag.Int("rounds", 10, "対話往復数")
	seed := flag.String("seed", "こんにちは、最近どう？", "初期メッセージ")
	basePort := flag.Int("base-port", 9000, "ベースポート")
	output := flag.String("output", "results.json", "結果出力先")
	binary := flag.String("binary", "", "lmindバイナリパス (デフォルト: 自動検出)")
	charSetting := flag.String("char-setting", "", "char_setting.md のパス (デフォルト: ~/.lmind/char_setting.md)")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	// バイナリパス
	bin := *binary
	if bin == "" {
		exe, err := os.Executable()
		if err != nil {
			bin = "lmind"
		} else {
			bin = filepath.Join(filepath.Dir(exe), "lmind")
		}
	}

	// char_setting.md パス
	charPath := *charSetting
	if charPath == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
			os.Exit(1)
		}
		charPath = filepath.Join(home, ".lmind", "char_setting.md")
	}
	if _, err := os.Stat(charPath); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: char_setting.md が見つかりません: %s\n", charPath)
		os.Exit(1)
	}

	// 一時ディレクトリ
	baseDir, err := os.MkdirTemp("", "lmind-ga-*")
	if err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
	defer os.RemoveAll(baseDir)

	fmt.Printf("=== lmind GA ===\n")
	fmt.Printf("個体数: %d, 世代数: %d, 対話往復: %d\n", *population, *generations, *rounds)
	fmt.Printf("ベースポート: %d, バイナリ: %s\n", *basePort, bin)
	fmt.Printf("一時ディレクトリ: %s\n\n", baseDir)

	var results []Result

	// 初期集団生成
	pop := ga.NewPopulation(bin, *basePort, baseDir, charPath, *population)

	for gen := range *generations {
		fmt.Printf("--- 第%d世代 ---\n", gen+1)

		// インスタンス起動
		fmt.Print("起動中...")
		if err := pop.Spawn(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "\nfatal: spawn失敗: %v\n", err)
			pop.Shutdown()
			os.Exit(1)
		}
		fmt.Println(" OK")

		// 思考が安定するまで少し待つ
		fmt.Print("思考開始待ち...")
		select {
		case <-ctx.Done():
			pop.Shutdown()
			return
		case <-time.After(10 * time.Second):
		}
		fmt.Println(" OK")

		// ペアリング対話
		fmt.Printf("対話中 (%d往復)...\n", *rounds)
		inds := pop.Individuals
		for i := 0; i+1 < len(inds); i += 2 {
			fmt.Printf("  %s <-> %s\n", inds[i].ID, inds[i+1].ID)
			if err := ga.Converse(ctx, inds[i], inds[i+1], *rounds, *seed); err != nil {
				fmt.Fprintf(os.Stderr, "  対話エラー: %v\n", err)
			}
		}

		// メトリクス収集・適応度計算
		fmt.Print("評価中...")
		for _, ind := range inds {
			if err := ga.Evaluate(ctx, ind); err != nil {
				fmt.Fprintf(os.Stderr, "\n  %s 評価エラー: %v\n", ind.ID, err)
			}
		}
		fmt.Println(" OK")

		// 結果表示
		sort.Slice(inds, func(i, j int) bool {
			return inds[i].Fitness > inds[j].Fitness
		})
		var entries []ResultEntry
		for _, ind := range inds {
			fmt.Printf("  %s: fitness=%.4f\n", ind.ID, ind.Fitness)
			entries = append(entries, ResultEntry{ID: ind.ID, Fitness: ind.Fitness})
		}
		results = append(results, Result{
			Generation: gen + 1,
			Best:       entries[0],
			All:        entries,
		})

		// シャットダウン
		fmt.Print("シャットダウン中...")
		pop.Shutdown()
		pop.CleanupData()
		fmt.Println(" OK")

		// 次世代生成（最終世代以外）
		if gen+1 < *generations {
			nextInds := ga.NextGeneration(inds, gen+1, *basePort)
			pop.Individuals = nextInds
		}

		fmt.Println()
	}

	// 結果出力
	data, _ := json.MarshalIndent(results, "", "  ")
	if err := os.WriteFile(*output, data, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "結果書き出しエラー: %v\n", err)
	} else {
		fmt.Printf("結果を %s に保存しました\n", *output)
	}

	// 最良パラメータを出力
	if len(results) > 0 {
		lastGen := results[len(results)-1]
		fmt.Printf("\n=== 最良個体: %s (fitness=%.4f) ===\n", lastGen.Best.ID, lastGen.Best.Fitness)

		// 最良個体のパラメータをファイルに保存
		bestInd := pop.Individuals[0] // ソート済みなので先頭
		bestParams, _ := json.MarshalIndent(bestInd.Params, "", "  ")
		bestPath := "best_params.json"
		if err := os.WriteFile(bestPath, bestParams, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "パラメータ書き出しエラー: %v\n", err)
		} else {
			fmt.Printf("最良パラメータを %s に保存しました\n", bestPath)
		}
	}
}
