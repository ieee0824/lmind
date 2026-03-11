package logger

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	_ "modernc.org/sqlite"
)

const expireDuration = 7 * 24 * time.Hour // 1週間

type Logger struct {
	db *sql.DB
}

func New(dbPath string) (*Logger, error) {
	db, err := sql.Open("sqlite", dbPath+"?_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}

	// WALモードで並行書き込みのロック競合を軽減
	if _, err := db.Exec(`PRAGMA journal_mode=WAL`); err != nil {
		db.Close()
		return nil, fmt.Errorf("set WAL mode: %w", err)
	}

	// 同時接続を1に制限してlock errorを防ぐ
	db.SetMaxOpenConns(1)

	if _, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS logs (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp DATETIME NOT NULL DEFAULT (datetime('now')),
			region TEXT NOT NULL,
			level TEXT NOT NULL DEFAULT 'info',
			message TEXT NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
		CREATE INDEX IF NOT EXISTS idx_logs_region ON logs(region);
		CREATE TABLE IF NOT EXISTS state (
			key TEXT PRIMARY KEY,
			value TEXT NOT NULL,
			updated_at DATETIME NOT NULL DEFAULT (datetime('now'))
		);
	`); err != nil {
		db.Close()
		return nil, fmt.Errorf("create table: %w", err)
	}

	l := &Logger{db: db}

	// 起動時に期限切れログを削除
	if err := l.cleanup(); err != nil {
		// 初回cleanup失敗は無視（DB自体は使える）
	}

	// 定期的にクリーンアップ
	go l.cleanupLoop()

	return l, nil
}

func (l *Logger) Log(region, level, message string) {
	_, err := l.db.Exec(
		`INSERT INTO logs (region, level, message) VALUES (?, ?, ?)`,
		region, level, message,
	)
	if err != nil {
		_ = err // DB書き込み失敗は静かに無視
	}
}

func (l *Logger) Info(region, message string) {
	l.Log(region, "info", message)
}

func (l *Logger) Error(region, message string) {
	l.Log(region, "error", message)
}

// Query は指定条件でログを取得する
func (l *Logger) Query(ctx context.Context, region string, limit int) ([]LogEntry, error) {
	query := `SELECT id, timestamp, region, level, message FROM logs`
	var args []any

	if region != "" {
		query += ` WHERE region = ?`
		args = append(args, region)
	}
	query += ` ORDER BY timestamp DESC LIMIT ?`
	args = append(args, limit)

	rows, err := l.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var entries []LogEntry
	for rows.Next() {
		var e LogEntry
		if err := rows.Scan(&e.ID, &e.Timestamp, &e.Region, &e.Level, &e.Message); err != nil {
			return nil, err
		}
		entries = append(entries, e)
	}
	return entries, rows.Err()
}

func (l *Logger) Close() error {
	return l.db.Close()
}

func (l *Logger) cleanup() error {
	cutoff := time.Now().Add(-expireDuration).UTC().Format("2006-01-02 15:04:05")
	result, err := l.db.Exec(`DELETE FROM logs WHERE timestamp < ?`, cutoff)
	if err != nil {
		return err
	}
	_, _ = result.RowsAffected()
	return nil
}

func (l *Logger) cleanupLoop() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()
	for range ticker.C {
		_ = l.cleanup()
	}
}

type LogEntry struct {
	ID        int64
	Timestamp string
	Region    string
	Level     string
	Message   string
}

// SaveState はkey-value形式でStateを永続化する
func (l *Logger) SaveState(key, value string) {
	_, err := l.db.Exec(
		`INSERT INTO state (key, value, updated_at) VALUES (?, ?, datetime('now'))
		 ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`,
		key, value,
	)
	if err != nil {
		_ = err
	}
}

// LoadState はkey-value形式で永続化されたStateを取得する
func (l *Logger) LoadState(key string) string {
	var value string
	err := l.db.QueryRow(`SELECT value FROM state WHERE key = ?`, key).Scan(&value)
	if err != nil {
		return ""
	}
	return value
}

// RecentThoughts は脳部位の直近の思考ログを時系列順で取得する
func (l *Logger) RecentThoughts(ctx context.Context, limit int) ([]LogEntry, error) {
	rows, err := l.db.QueryContext(ctx, `
		SELECT id, timestamp, region, level, message FROM logs
		WHERE region IN ('frontal', 'temporal', 'hippocampus', 'user')
		AND level = 'info'
		ORDER BY timestamp DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var entries []LogEntry
	for rows.Next() {
		var e LogEntry
		if err := rows.Scan(&e.ID, &e.Timestamp, &e.Region, &e.Level, &e.Message); err != nil {
			return nil, err
		}
		entries = append(entries, e)
	}
	// 逆順にして時系列順にする
	for i, j := 0, len(entries)-1; i < j; i, j = i+1, j-1 {
		entries[i], entries[j] = entries[j], entries[i]
	}
	return entries, rows.Err()
}
