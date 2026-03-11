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

// History は直近の思考を取得する（Chat I/Fでの文脈表示用）
// 注: 簡易実装。必要なら別途リングバッファを追加
type History struct {
	mu       sync.Mutex
	thoughts []Thought
	maxSize  int
}

func NewHistory(maxSize int) *History {
	return &History{
		thoughts: make([]Thought, 0, maxSize),
		maxSize:  maxSize,
	}
}

func (h *History) Record(t Thought) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.thoughts = append(h.thoughts, t)
	if len(h.thoughts) > h.maxSize {
		h.thoughts = h.thoughts[1:]
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
