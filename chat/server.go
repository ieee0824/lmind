package chat

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/ieee0824/lmind/brain"
	"github.com/ieee0824/lmind/bus"
	"github.com/ieee0824/lmind/logger"
	"github.com/ieee0824/lmind/ollama"
)

// ServerConfig はチャットサーバーの設定
type ServerConfig struct {
	Client           *ollama.Client
	Model            string
	JudgeModel       string // モード判定用の軽量モデル
	Bus              *bus.ThoughtBus
	History          *bus.History
	Logger           *logger.Logger
	ChatPrompt       string // 雑談モード用
	TaskPrompt       string // 秘書モード用
	ModeJudgePrompt  string // モード判定用
	InhibitionPrompt string
	Rapport          *brain.Rapport // ラポートモジュール（nilなら無効）
}

// Server はCLIベースのチャットインターフェース
type Server struct {
	client           *ollama.Client
	model            string
	judgeModel       string
	bus              *bus.ThoughtBus
	history          *bus.History
	logger           *logger.Logger
	chatPrompt       string
	taskPrompt       string
	modeJudgePrompt  string
	inhibitionPrompt string
	rapport          *brain.Rapport
	inbox            <-chan bus.Thought

	mu         sync.Mutex
	lastActive string

	convMu      sync.Mutex
	convHistory []string // ユーザー発言専用の会話履歴
}

func New(cfg ServerConfig) *Server {
	s := &Server{
		client:           cfg.Client,
		model:            cfg.Model,
		judgeModel:       cfg.JudgeModel,
		bus:              cfg.Bus,
		history:          cfg.History,
		logger:           cfg.Logger,
		chatPrompt:       cfg.ChatPrompt,
		taskPrompt:       cfg.TaskPrompt,
		modeJudgePrompt:  cfg.ModeJudgePrompt,
		inhibitionPrompt: cfg.InhibitionPrompt,
		rapport:          cfg.Rapport,
	}
	s.inbox = cfg.Bus.Subscribe("chat")
	return s
}

// drainBus はバスからの思考を消費し、アクティブ部位を追跡する
func (s *Server) drainBus(ctx context.Context) {
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
}

// Run はユーザーからの入力を待ち受ける（CLIモード）
func (s *Server) Run(ctx context.Context) {
	go s.drainBus(ctx)

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

			// ユーザーの質問を履歴に記録（バスへの配信は応答後）
			userThought := bus.Thought{
				From:    "user",
				Content: input,
			}
			s.history.Record(userThought)

			// 会話専用履歴にユーザー発言を追加（思考バスとは独立）
			s.addConvHistory("user: " + input)

			// スピナー表示しながら応答を待つ
			done := make(chan string, 1)
			go func() {
				done <- s.respond(ctx, input)
			}()

			response := s.waitResponse(ctx, done)

			// botの応答も会話履歴に追加
			s.addConvHistory("bot: " + response)

			// 応答完了後にユーザー入力とBroca応答をバスに流す（Ollama渋滞を防ぐ）
			s.bus.Publish(userThought)
			if response != "" {
				brocaThought := bus.Thought{
					From:    "broca",
					Content: fmt.Sprintf("I said to the user: %s", response),
				}
				s.history.Record(brocaThought)
				s.bus.Publish(brocaThought)
			}
		}
	}
}

// waitResponse はLLM応答待ちの間スピナーを表示し、応答テキストを返す
func (s *Server) waitResponse(ctx context.Context, done <-chan string) string {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	dots := 0
	for {
		select {
		case <-ctx.Done():
			fmt.Print("\r\033[K")
			return ""
		case response := <-done:
			fmt.Printf("\r\033[K\n%s\n\n> ", response)
			return response
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

// addConvHistory は会話専用履歴にエントリを追加する
func (s *Server) addConvHistory(entry string) {
	s.convMu.Lock()
	defer s.convMu.Unlock()
	s.convHistory = append(s.convHistory, entry)
	if len(s.convHistory) > 20 {
		s.convHistory = s.convHistory[1:]
	}
}

const maxRefineRounds = 2

// judgeMode は軽量モデルでユーザー入力のモード（chat/task）を判定する
func (s *Server) judgeMode(ctx context.Context, question string) string {
	resp, err := s.client.Chat(ctx, ollama.ChatRequest{
		Model: s.judgeModel,
		Messages: []ollama.Message{
			{Role: "system", Content: s.modeJudgePrompt},
			{Role: "user", Content: question},
		},
	})
	if err != nil {
		s.logger.Error("judge", fmt.Sprintf("モード判定エラー: %v", err))
		return "chat" // エラー時はchatモードにフォールバック
	}
	mode := strings.TrimSpace(strings.ToLower(resp.Message.Content))
	if strings.Contains(mode, "task") {
		return "task"
	}
	return "chat"
}

func (s *Server) respond(ctx context.Context, question string) string {
	// モード判定（軽量モデルで高速に）
	mode := s.judgeMode(ctx, question)
	s.logger.Info("judge", fmt.Sprintf("モード: %s", mode))

	systemPrompt := s.chatPrompt
	if mode == "task" {
		systemPrompt = s.taskPrompt
	}

	// ラポートレベルに応じてプロンプトを動的に調整
	if s.rapport != nil {
		systemPrompt += "\n\n" + rapportDirective(s.rapport.Level(), s.rapport.Score())
	}

	recent := s.history.Recent(10)

	var thoughtList []string
	for _, t := range recent {
		if t.From == "user" || t.From == "system" {
			continue
		}
		thoughtList = append(thoughtList, t.Content)
	}

	// 会話専用履歴から取得（思考バスに埋もれない）
	s.convMu.Lock()
	historyList := make([]string, len(s.convHistory))
	copy(historyList, s.convHistory)
	s.convMu.Unlock()

	type brocaInput struct {
		Question    string   `json:"question"`
		History     []string `json:"history,omitempty"`
		Thoughts    []string `json:"thoughts,omitempty"`
		Instruction string   `json:"instruction"`
	}

	input := brocaInput{
		Question:    question,
		History:     historyList,
		Thoughts:    thoughtList,
		Instruction: "questionに直接答えて。historyは直近の会話履歴。thoughtsは参考程度。",
	}
	baseJSON, _ := json.Marshal(input)
	basePrompt := string(baseJSON)

	var refined string
	prompt := basePrompt

	for i := range maxRefineRounds {
		// Broca起案
		resp, err := s.client.Chat(ctx, ollama.ChatRequest{
			Model: s.model,
			Messages: []ollama.Message{
				{Role: "system", Content: systemPrompt},
				{Role: "user", Content: prompt},
			},
		})
		if err != nil {
			s.logger.Error("broca", fmt.Sprintf("応答エラー: %v", err))
			return fmt.Sprintf("エラー: %v", err)
		}
		draft := resp.Message.Content
		s.logger.Info("broca", fmt.Sprintf("起案(%d): %s", i+1, draft))

		// Step 1: コードで地の文・装飾を機械的に除去
		stripped := stripNarrative(draft)
		s.logger.Info("inhibition", fmt.Sprintf("除去後(%d): %s", i+1, stripped))

		// Step 2: LLMで要約（長すぎる場合のみ）
		if len([]rune(stripped)) > 150 {
			var err2 error
			refined, err2 = s.refine(ctx, question, stripped)
			if err2 != nil {
				s.logger.Error("inhibition", fmt.Sprintf("要約エラー: %v", err2))
				refined = stripped
			} else {
				refined = stripNarrative(refined) // 要約結果にも地の文が入る可能性があるので再除去
			}
		} else {
			refined = stripped
		}
		s.logger.Info("inhibition", fmt.Sprintf("最終(%d): %s", i+1, refined))

		// 起案と整形後の差が小さければ収束とみなす
		if s.isSimilarLength(draft, refined) {
			break
		}

		// 差が大きい → 理由付きで差し戻し
		s.logger.Info("inhibition", fmt.Sprintf("差し戻し(%d): 起案が大幅に整形された", i+1))

		type remandInput struct {
			Question    string   `json:"question"`
			Thoughts    []string `json:"thoughts,omitempty"`
			PrevDraft   string   `json:"prev_draft"`
			Refined     string   `json:"refined"`
			Instruction string   `json:"instruction"`
		}
		ri := remandInput{
			Question:    question,
			Thoughts:    thoughtList,
			PrevDraft:   draft,
			Refined:     refined,
			Instruction: "prev_draftは冗長だったのでrefinedに整形された。最初からrefinedレベルの簡潔さでquestionに答えて。",
		}
		remandJSON, _ := json.Marshal(ri)
		prompt = string(remandJSON)
	}

	return refined
}

// isSimilarLength は起案と整形後の長さが近いかを判定する
// 整形で半分以下に縮んだら「大幅に違う」とみなす
func (s *Server) isSimilarLength(draft, refined string) bool {
	dl := len([]rune(draft))
	rl := len([]rune(refined))
	if dl == 0 {
		return true
	}
	return float64(rl)/float64(dl) > 0.5
}

// refine は前頭葉の抑制機能としてドラフトを整形する（地の文除去・要約）
// question を渡すことで、ユーザーの質問から離れた整形を防ぐ
func (s *Server) refine(ctx context.Context, question, draft string) (string, error) {
	prompt := fmt.Sprintf("【ユーザーの発言】\n%s\n\n【返答案】\n%s", question, draft)
	resp, err := s.client.Chat(ctx, ollama.ChatRequest{
		Model: s.model,
		Messages: []ollama.Message{
			{Role: "system", Content: s.inhibitionPrompt},
			{Role: "user", Content: prompt},
		},
	})
	if err != nil {
		return "", err
	}
	return stripNarrative(resp.Message.Content), nil
}

// 地の文除去の正規表現パターン（コンパイル済み）
var narrativePatterns = []*regexp.Regexp{
	regexp.MustCompile(`（[^）]*）`),                // 全角括弧（心情・ト書き）
	regexp.MustCompile(`\([^)]*\)`),                // 半角括弧
	regexp.MustCompile(`\*[^*]+\*`),                // *動作*
	regexp.MustCompile(`(?m)^[\s　]*[「」][\s　]*$`), // 空のかぎかっこ行
	regexp.MustCompile(`…+[\s　]*$`),               // 行末の余韻「…」だけの行
	regexp.MustCompile(`[\*＊]`),                    // 残ったアスタリスク
	regexp.MustCompile(`(?m)^[\s　]*\*\s+.*$`),     // マークダウンリスト（* item）
	regexp.MustCompile(`「|」`),                                                             // かぎかっこ除去
	regexp.MustCompile(`[\x{1F300}-\x{1F9FF}\x{2600}-\x{27BF}\x{FE00}-\x{FE0F}\x{200D}]+`), // 絵文字除去
}

// stripNarrative はLLM出力から地の文・ト書き・装飾を機械的に除去する
func stripNarrative(text string) string {
	result := text
	for _, re := range narrativePatterns {
		result = re.ReplaceAllString(result, "")
	}

	// 空行の整理と各行のトリム
	lines := strings.Split(result, "\n")
	var cleaned []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		cleaned = append(cleaned, trimmed)
	}

	return strings.Join(cleaned, "\n")
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

// rapportDirective はラポートレベルに応じた人格表現の指示を返す
//
// 重要: 親密度が上がっても人格の芯（口調・文量・振る舞い）は変わらない。
// 変わるのは「心の距離感」だけ。テンションは上げない。
func rapportDirective(level string, score float64) string {
	// 全レベル共通の制約
	const constraint = `
【重要】親密度が高くても以下は守ること：
- 返答は1〜2文。短いほどいい。
- 絵文字は使わない。
- テンションは上げない。落ち着いたトーンを維持。
- 同じ話題やフレーズを繰り返さない。
- 相手の話題に合わせる。自分の過去の話に戻らない。`

	var directive string
	switch level {
	case "intimate":
		directive = fmt.Sprintf(`【親密度: %.0f%% — intimate】
- 本音が出る。取り繕わない。
- ぼそっと照れたり、素っ気なく優しい。
- 気になったことは自然に聞き返す。
- くだけた語尾OK。ただし騒がしくならない。`, score*100)
	case "friendly":
		directive = fmt.Sprintf(`【親密度: %.0f%% — friendly】
- リラックスした口調。軽口を混ぜていい。
- 自分の意見も自然に出す。
- 興味があれば聞き返していい。`, score*100)
	case "neutral":
		directive = fmt.Sprintf(`【親密度: %.0f%% — neutral】
- 普通の距離感。丁寧すぎず、くだけすぎず。
- たまに聞き返すくらいはOK。`, score*100)
	default: // guarded
		directive = fmt.Sprintf(`【親密度: %.0f%% — guarded】
- やや警戒気味。丁寧めの口調。
- 踏み込まない。様子見。
- 自分からは聞き返さない。`, score*100)
	}
	return directive + constraint
}
