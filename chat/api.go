package chat

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/ieee0824/lmind/bus"
)

// APIRequest はWeb API経由のチャットリクエスト
type APIRequest struct {
	Message string `json:"message"`
}

// APIResponse はWeb API経由のチャットレスポンス
type APIResponse struct {
	Response string `json:"response"`
	Mode     string `json:"mode,omitempty"`
}

// ThoughtEntry はthoughts APIのレスポンス用
type ThoughtEntry struct {
	From      string `json:"from"`
	Content   string `json:"content"`
	CreatedAt string `json:"created_at"`
}

// RunAPI はHTTPサーバーとしてAPIを起動する（CLIの代替）
// ctxがキャンセルされるとgraceful shutdownする。
func (s *Server) RunAPI(ctx context.Context, addr string) error {
	go s.drainBus(ctx)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /api/chat", s.handleChat)
	mux.HandleFunc("GET /api/thoughts", s.handleThoughts)
	mux.HandleFunc("GET /api/metrics", s.handleMetrics)
	mux.HandleFunc("GET /api/params", s.handleParams)
	srv := &http.Server{Addr: addr, Handler: mux}

	// ctxキャンセル時にgraceful shutdown
	go func() {
		<-ctx.Done()
		s.logger.Info("api", "シャットダウン中...")
		srv.Shutdown(ctx)
	}()

	s.logger.Info("api", fmt.Sprintf("Web API起動: %s", addr))
	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		return err
	}
	return nil
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req APIRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		return
	}

	msg := strings.TrimSpace(req.Message)
	if msg == "" {
		http.Error(w, `{"error":"message is required"}`, http.StatusBadRequest)
		return
	}

	// ユーザー入力を履歴に記録
	userThought := bus.Thought{From: "user", Content: msg}
	s.history.Record(userThought)
	s.addConvHistory("user: " + msg)

	// 応答生成（既存のrespond()を再利用）
	ctx := r.Context()
	response := s.respond(ctx, msg)

	// 会話履歴に追加
	s.addConvHistory("bot: " + response)

	// バスに配信（応答完了後）
	s.bus.Publish(userThought)
	if response != "" {
		brocaThought := bus.Thought{From: "broca", Content: fmt.Sprintf("I said to the user: %s", response)}
		s.history.Record(brocaThought)
		s.bus.Publish(brocaThought)
	}

	resp := APIResponse{
		Response: response,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleThoughts(w http.ResponseWriter, r *http.Request) {
	recent := s.history.Recent(10)
	entries := make([]ThoughtEntry, 0, len(recent))
	for _, t := range recent {
		entries = append(entries, ThoughtEntry{
			From:      t.From,
			Content:   t.Content,
			CreatedAt: t.CreatedAt.Format("15:04:05"),
		})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(entries)
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if s.metrics == nil {
		http.Error(w, `{"error":"metrics not available"}`, http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s.metrics.Snapshot())
}

func (s *Server) handleParams(w http.ResponseWriter, r *http.Request) {
	if s.params == nil {
		http.Error(w, `{"error":"params not available"}`, http.StatusServiceUnavailable)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s.params)
}

