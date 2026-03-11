package bus

import (
	"sync"
	"time"
)

// Thought は脳部位間を流れる中間情報（＝思考）
type Thought struct {
	From      string    // 発信元の脳部位
	To        string    // 宛先（空なら全部位へブロードキャスト）
	Content   string    // 思考の内容
	Summary   string    // 圧縮済みの要約（空なら未圧縮）
	CreatedAt time.Time // 生成時刻
}

// ThoughtBus は脳部位間の通信バス
type ThoughtBus struct {
	mu          sync.RWMutex
	subscribers map[string]chan Thought
}

func New() *ThoughtBus {
	return &ThoughtBus{
		subscribers: make(map[string]chan Thought),
	}
}

// Subscribe は脳部位がバスに接続する
func (b *ThoughtBus) Subscribe(name string) <-chan Thought {
	b.mu.Lock()
	defer b.mu.Unlock()

	ch := make(chan Thought, 64)
	b.subscribers[name] = ch
	return ch
}

// Publish は思考をバスに流す
func (b *ThoughtBus) Publish(t Thought) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	t.CreatedAt = time.Now()

	for name, ch := range b.subscribers {
		// 自分自身には送らない
		if name == t.From {
			continue
		}
		// 宛先指定があれば、その部位のみ
		if t.To != "" && t.To != name {
			continue
		}
		select {
		case ch <- t:
		default:
			// バッファが溢れたら古い思考を捨てる
		}
	}
}

// SummarizeFunc は思考を1行に要約する関数の型
type SummarizeFunc func(content string) string

// History は直近の思考を取得する（Chat I/Fでの文脈表示用）
// freshCount件より古い思考は自動的に要約される
type History struct {
	mu          sync.Mutex
	thoughts    []Thought
	maxSize     int
	freshCount  int           // 全文保持する直近件数
	summarizeFn SummarizeFunc // 要約関数（nilなら切り詰め）
}

func NewHistory(maxSize int) *History {
	return &History{
		thoughts:   make([]Thought, 0, maxSize),
		maxSize:    maxSize,
		freshCount: 3,
	}
}

// SetSummarizer は要約関数を設定する
func (h *History) SetSummarizer(fn SummarizeFunc) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.summarizeFn = fn
}

func (h *History) Record(t Thought) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.thoughts = append(h.thoughts, t)
	if len(h.thoughts) > h.maxSize {
		h.thoughts = h.thoughts[1:]
	}

	// freshCount境界を超えた思考を圧縮
	h.compressOld()
}

// compressOld は直近freshCount件より古い未圧縮の思考を要約する
// mu.Lockを取得した状態で呼ぶこと
func (h *History) compressOld() {
	if len(h.thoughts) <= h.freshCount {
		return
	}
	boundary := len(h.thoughts) - h.freshCount
	for i := 0; i < boundary; i++ {
		if h.thoughts[i].Summary != "" {
			continue // 要約済み
		}
		if h.summarizeFn != nil {
			h.thoughts[i].Summary = h.summarizeFn(h.thoughts[i].Content)
		} else {
			// フォールバック: 先頭80文字で切り詰め
			runes := []rune(h.thoughts[i].Content)
			if len(runes) > 80 {
				h.thoughts[i].Summary = string(runes[:80]) + "…"
			} else {
				h.thoughts[i].Summary = h.thoughts[i].Content
			}
		}
	}
}

func (h *History) Recent(n int) []Thought {
	h.mu.Lock()
	defer h.mu.Unlock()

	if n > len(h.thoughts) {
		n = len(h.thoughts)
	}
	result := make([]Thought, n)
	copy(result, h.thoughts[len(h.thoughts)-n:])
	return result
}
