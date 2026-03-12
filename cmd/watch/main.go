package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

// ANSI colors
const (
	colorReset  = "\033[0m"
	colorUser   = "\033[1;36m" // cyan bold
	colorBroca  = "\033[1;32m" // green bold
	colorThink  = "\033[0;33m" // yellow
	colorMeta   = "\033[0;35m" // magenta
	colorDim    = "\033[2m"    // dim
	colorBold   = "\033[1m"
)

var regionColors = map[string]string{
	"user":      colorUser,
	"broca":     colorBroca,
	"frontal":   colorThink,
	"temporal":  colorThink,
	"hypothesis": colorMeta,
	"critic":    colorMeta,
	"curiosity": colorMeta,
	"bonnou":    colorMeta,
	"grounding": colorDim,
	"modulator": colorDim,
}

type logRow struct {
	id        int64
	timestamp string
	region    string
	message   string
	source    string // individual ID
}

func main() {
	dir := flag.String("dir", "", "GA一時ディレクトリ (例: /tmp/lmind-ga-xxx)")
	dbPath := flag.String("db", "", "単一のthoughts.dbパス")
	regions := flag.String("regions", "", "表示するregionをカンマ区切り (例: user,broca,frontal) 空=全部")
	chatOnly := flag.Bool("chat", false, "会話のみ表示 (user,broca)")
	interval := flag.Float64("interval", 1.0, "ポーリング間隔(秒)")
	flag.Parse()

	if *dir == "" && *dbPath == "" {
		// 自動検出: ~/.lmind/thoughts.db
		home, _ := os.UserHomeDir()
		defaultDB := filepath.Join(home, ".lmind", "thoughts.db")
		if _, err := os.Stat(defaultDB); err == nil {
			*dbPath = defaultDB
		} else {
			fmt.Fprintln(os.Stderr, "使い方: lmind-watch -dir <GA一時ディレクトリ> または -db <thoughts.db>")
			os.Exit(1)
		}
	}

	var regionFilter map[string]bool
	if *chatOnly {
		regionFilter = map[string]bool{"user": true, "broca": true}
	} else if *regions != "" {
		regionFilter = make(map[string]bool)
		for _, r := range strings.Split(*regions, ",") {
			regionFilter[strings.TrimSpace(r)] = true
		}
	}

	// DB一覧を収集
	type dbSource struct {
		path string
		name string
	}
	var sources []dbSource

	if *dbPath != "" {
		sources = append(sources, dbSource{*dbPath, "lmind"})
	} else {
		entries, err := os.ReadDir(*dir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ディレクトリ読み込み失敗: %v\n", err)
			os.Exit(1)
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			db := filepath.Join(*dir, e.Name(), "thoughts.db")
			if _, err := os.Stat(db); err == nil {
				sources = append(sources, dbSource{db, e.Name()})
			}
		}
	}

	if len(sources) == 0 {
		fmt.Fprintln(os.Stderr, "DBが見つかりません")
		os.Exit(1)
	}

	// DB接続
	type dbConn struct {
		db     *sql.DB
		name   string
		lastID int64
	}
	var conns []*dbConn

	for _, s := range sources {
		db, err := sql.Open("sqlite", s.path+"?_busy_timeout=5000&mode=ro")
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: open失敗: %v\n", s.name, err)
			continue
		}
		db.SetMaxOpenConns(1)

		// 最新IDを取得（既存ログはスキップ）
		var maxID int64
		_ = db.QueryRow(`SELECT COALESCE(MAX(id), 0) FROM logs`).Scan(&maxID)

		conns = append(conns, &dbConn{db: db, name: s.name, lastID: maxID})
	}
	defer func() {
		for _, c := range conns {
			c.db.Close()
		}
	}()

	fmt.Printf("%s=== lmind-watch ===%s\n", colorBold, colorReset)
	fmt.Printf("監視中: %d個のDB (%.1f秒間隔)\n", len(conns), *interval)
	if regionFilter != nil {
		keys := make([]string, 0, len(regionFilter))
		for k := range regionFilter {
			keys = append(keys, k)
		}
		fmt.Printf("フィルタ: %s\n", strings.Join(keys, ", "))
	}
	fmt.Println()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	ticker := time.NewTicker(time.Duration(*interval * float64(time.Second)))
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			fmt.Println("\n中断しました")
			return
		case <-ticker.C:
			var rows []logRow
			for _, c := range conns {
				newRows, newMax := poll(c.db, c.name, c.lastID, regionFilter)
				if newMax > c.lastID {
					c.lastID = newMax
				}
				rows = append(rows, newRows...)
			}
			// タイムスタンプ+IDでソート
			sort.Slice(rows, func(i, j int) bool {
				if rows[i].timestamp == rows[j].timestamp {
					return rows[i].id < rows[j].id
				}
				return rows[i].timestamp < rows[j].timestamp
			})
			for _, r := range rows {
				printRow(r, len(conns) > 1)
			}
		}
	}
}

func poll(db *sql.DB, name string, lastID int64, regionFilter map[string]bool) ([]logRow, int64) {
	query := `SELECT id, timestamp, region, message FROM logs WHERE id > ? ORDER BY id ASC`
	rows, err := db.Query(query, lastID)
	if err != nil {
		return nil, lastID
	}
	defer rows.Close()

	var result []logRow
	maxID := lastID
	for rows.Next() {
		var r logRow
		if err := rows.Scan(&r.id, &r.timestamp, &r.region, &r.message); err != nil {
			continue
		}
		r.source = name
		if r.id > maxID {
			maxID = r.id
		}
		if regionFilter != nil && !regionFilter[r.region] {
			continue
		}
		result = append(result, r)
	}
	return result, maxID
}

func printRow(r logRow, showSource bool) {
	color := regionColors[r.region]
	if color == "" {
		color = colorDim
	}

	// タイムスタンプから時刻部分だけ抽出
	ts := r.timestamp
	if len(ts) > 11 {
		ts = ts[11:]
	}

	msg := r.message
	// "I said to the user: " プレフィックスを除去して見やすくする
	msg = strings.TrimPrefix(msg, "I said to the user: ")

	if showSource {
		fmt.Printf("%s%s%s %s%-12s%s %s[%s]%s %s\n",
			colorDim, ts, colorReset,
			color, r.region, colorReset,
			colorDim, r.source, colorReset,
			msg)
	} else {
		fmt.Printf("%s%s%s %s%-12s%s %s\n",
			colorDim, ts, colorReset,
			color, r.region, colorReset,
			msg)
	}
}
