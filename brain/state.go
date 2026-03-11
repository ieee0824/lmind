package brain

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
	"github.com/ieee0824/lmind/ollama"
)

// State は思考の明示的な内部状態。
// goal（今何を目指しているか）、hypothesis（現在の仮説・理解）、
// prediction（ユーザーの次の発言予測）を保持する。
type State struct {
	mu         sync.RWMutex
	goal       string  // 現在の目標・関心事
	hypothesis string  // 現在の仮説・理解
	prediction string  // ユーザーの次の発言予測
	identity   string  // 自己認識
	sentiment  float64 // 直近のユーザー感情極性（-1.0〜+1.0）
	store      *logger.Logger // 永続化用（nilなら永続化しない）
}

const defaultIdentity = "I am lmind, a thinking system. These thoughts are my own internal process. I am not observing an external model — I AM the one thinking. The user is a separate person talking to me. I must not confuse my own words with theirs."

func NewState() *State {
	return &State{
		goal:       "No specific goal. Freely exploring thoughts.",
		hypothesis: "",
		prediction: "",
		identity:   defaultIdentity,
	}
}

// RestoreFrom はSQLiteから前回のStateを復元する。
// goalのみ復元する。hypothesisはセッション間で引き継がない
// （古い仮説が全モジュールの解釈を歪める問題の防止）。
func (s *State) RestoreFrom(lg *logger.Logger) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.store = lg

	if g := lg.LoadState("goal"); g != "" {
		s.goal = g
	}
	// hypothesisは意図的に復元しない
}

// Snapshot はstateの現在値を返す（読み取り専用コピー）
func (s *State) Snapshot() (goal, hypothesis string) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.goal, s.hypothesis
}

// Prediction は現在の予測を返す
func (s *State) Prediction() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.prediction
}

func (s *State) SetPrediction(prediction string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.prediction = prediction
}

func (s *State) SetGoal(goal string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.goal = goal
	if s.store != nil {
		s.store.SaveState("goal", goal)
	}
}

func (s *State) SetHypothesis(hypothesis string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.hypothesis = hypothesis
	if s.store != nil {
		s.store.SaveState("hypothesis", hypothesis)
	}
}

func (s *State) SetSentiment(score float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sentiment = score
}

// Sentiment は直近のユーザー感情極性を返す
func (s *State) Sentiment() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.sentiment
}

// Format はプロンプトに注入するための文字列を生成する
func (s *State) Format() string {
	s.mu.RLock()
	goal := s.goal
	hyp := s.hypothesis
	pred := s.prediction
	identity := s.identity
	sentiment := s.sentiment
	s.mu.RUnlock()

	var sb strings.Builder
	sb.WriteString("=== Current State ===\n")
	sb.WriteString(fmt.Sprintf("Identity: %s\n", identity))
	sb.WriteString(fmt.Sprintf("Goal: %s\n", goal))
	if hyp != "" {
		sb.WriteString(fmt.Sprintf("Hypothesis: %s\n", hyp))
	}
	if pred != "" {
		sb.WriteString(fmt.Sprintf("Predicted next user input: %s\n", pred))
	}
	if sentiment != 0 {
		label := "neutral"
		if sentiment > 0.3 {
			label = "positive"
		} else if sentiment < -0.3 {
			label = "negative"
		}
		sb.WriteString(fmt.Sprintf("User sentiment: %s (%.2f)\n", label, sentiment))
	}
	sb.WriteString("=====================")
	return sb.String()
}

// StateUpdater は思考バスを監視してStateを更新するモジュール
type StateUpdater struct {
	state   *State
	client  *ollama.Client
	model   string
	bus     *bus.ThoughtBus
	history *bus.History
	logger  *logger.Logger
	inbox   <-chan bus.Thought
}

type StateUpdaterConfig struct {
	State   *State
	Client  *ollama.Client
	Model   string // 軽量モデル（gemma3:1b）
	Bus     *bus.ThoughtBus
	History *bus.History
	Logger  *logger.Logger
}

func NewStateUpdater(cfg StateUpdaterConfig) *StateUpdater {
	u := &StateUpdater{
		state:   cfg.State,
		client:  cfg.Client,
		model:   cfg.Model,
		bus:     cfg.Bus,
		history: cfg.History,
		logger:  cfg.Logger,
	}
	u.inbox = cfg.Bus.Subscribe("state")
	return u
}

// Run はstate更新ループを開始する（ユーザー入力からgoalを更新）
func (u *StateUpdater) Run(ctx context.Context) {
	u.logger.Info("state", "State更新モジュール開始")

	for {
		select {
		case <-ctx.Done():
			return
		case t := <-u.inbox:
			if t.From == "user" {
				u.updateGoal(ctx, t.Content)
			}
		}
	}
}

// updateGoal はユーザー入力からgoalを更新する
func (u *StateUpdater) updateGoal(ctx context.Context, userInput string) {
	currentGoal, _ := u.state.Snapshot()

	prompt := fmt.Sprintf(`Current goal: %s

User just said: %s

Based on what the user said, what is the current goal or topic of interest?
If the user is just chatting casually, reply: "No specific goal. Freely exploring thoughts."
If the user has a clear intent or question, summarize it in one sentence.
Output ONLY the goal, nothing else.`, currentGoal, userInput)

	resp, err := u.client.Chat(ctx, ollama.ChatRequest{
		Model: u.model,
		Messages: []ollama.Message{
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		u.logger.Error("state", fmt.Sprintf("goal更新エラー: %v", err))
		return
	}

	newGoal := strings.TrimSpace(resp.Message.Content)
	if newGoal != "" {
		u.state.SetGoal(newGoal)
		u.logger.Info("state", fmt.Sprintf("goal更新: %s", newGoal))
	}
}
