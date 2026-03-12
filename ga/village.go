package ga

import (
	"context"
	"fmt"
	"math/rand"
	"sync"

	"github.com/ieee0824/lmind/ollama"
)

// OddballRate は「変わったやつ」の発生確率（村間交流のうち遠い村を選ぶ割合）
const OddballRate = 0.2

// Village は個体の社会的グループ
type Village struct {
	ID      string
	Members []*Individual
}

// Topology は村の配置と対話構造を管理する
type Topology struct {
	Villages       []*Village
	IntraRounds    int     // 村内対話の往復数
	InterRounds    int     // 村間対話の往復数
	InterRate      float64 // 村間交流の発生確率 (0.0-1.0)
}

// NewTopology は個体群を村に分割する
// villageSize=0 の場合、自動で2-3人の村に分割する
func NewTopology(inds []*Individual, villageSize int, intraRounds, interRounds int, interRate float64) *Topology {
	if villageSize <= 0 {
		villageSize = 2
		if len(inds) >= 6 {
			villageSize = 3
		}
	}

	t := &Topology{
		IntraRounds: intraRounds,
		InterRounds: interRounds,
		InterRate:   interRate,
	}

	for i := 0; i < len(inds); i += villageSize {
		end := i + villageSize
		if end > len(inds) {
			end = len(inds)
		}
		v := &Village{
			ID:      fmt.Sprintf("village-%d", len(t.Villages)),
			Members: inds[i:end],
		}
		t.Villages = append(t.Villages, v)
	}

	// 端数が1人の村は前の村に合流
	if len(t.Villages) > 1 {
		last := t.Villages[len(t.Villages)-1]
		if len(last.Members) == 1 {
			prev := t.Villages[len(t.Villages)-2]
			prev.Members = append(prev.Members, last.Members...)
			t.Villages = t.Villages[:len(t.Villages)-1]
		}
	}

	return t
}

// RunConversations は村内の密な対話と村間の緩やかな交流を実行する
// ollamaClient が非nilの場合、embed距離ベースで村間交流を最適化する
func (t *Topology) RunConversations(ctx context.Context, seed string, ollamaClient ...*ollama.Client) error {
	// Phase 1: 村内の密な対話（総当たり、並列）
	fmt.Printf("  村内対話 (%d村, %d往復)...\n", len(t.Villages), t.IntraRounds)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var errs []error

	for _, v := range t.Villages {
		pairs := allPairs(v.Members)
		for _, pair := range pairs {
			fmt.Printf("    [%s] %s <-> %s\n", v.ID, pair[0].ID, pair[1].ID)
			wg.Add(1)
			go func(a, b *Individual) {
				defer wg.Done()
				if err := Converse(ctx, a, b, t.IntraRounds, seed); err != nil {
					mu.Lock()
					errs = append(errs, fmt.Errorf("[%s] %s<->%s: %w", v.ID, a.ID, b.ID, err))
					mu.Unlock()
				}
			}(pair[0], pair[1])
		}
	}
	wg.Wait()

	if len(errs) > 0 {
		for _, e := range errs {
			fmt.Printf("    対話エラー: %v\n", e)
		}
	}

	// Phase 2: 村間の交流
	if len(t.Villages) < 2 {
		return nil
	}

	var interPairs [][2]*Individual

	// Ollamaクライアントがあればセマンティック距離ベースの交流
	var client *ollama.Client
	if len(ollamaClient) > 0 && ollamaClient[0] != nil {
		client = ollamaClient[0]
	}

	if client != nil {
		fmt.Println("  村のembedding計算中...")
		vecs, err := EmbedVillages(ctx, client, t.Villages)
		if err == nil {
			interPairs = semanticInterPairs(t.Villages, vecs, t.InterRate)
		} else {
			fmt.Printf("    embed失敗、ランダムにフォールバック: %v\n", err)
		}
	}

	// セマンティック交流が得られなかった場合はランダム
	if interPairs == nil {
		interPairs = t.randomInterPairs()
	}

	if len(interPairs) == 0 {
		return nil
	}

	fmt.Printf("  村間交流 (%d組, %d往復)...\n", len(interPairs), t.InterRounds)
	for _, pair := range interPairs {
		wg.Add(1)
		go func(a, b *Individual) {
			defer wg.Done()
			if err := Converse(ctx, a, b, t.InterRounds, seed); err != nil {
				mu.Lock()
				errs = append(errs, fmt.Errorf("inter: %s<->%s: %w", a.ID, b.ID, err))
				mu.Unlock()
			}
		}(pair[0], pair[1])
	}
	wg.Wait()

	if len(errs) > 0 {
		for _, e := range errs {
			fmt.Printf("    交流エラー: %v\n", e)
		}
	}

	return nil
}

// randomInterPairs は隣接する村から代表者を1人ずつ選んでペアにする（ランダム版）
func (t *Topology) randomInterPairs() [][2]*Individual {
	var pairs [][2]*Individual
	for i := 0; i+1 < len(t.Villages); i++ {
		if rand.Float64() > t.InterRate {
			continue
		}
		a := t.Villages[i]
		b := t.Villages[i+1]
		// 各村からランダムに1人選出
		ambA := a.Members[rand.Intn(len(a.Members))]
		ambB := b.Members[rand.Intn(len(b.Members))]
		pairs = append(pairs, [2]*Individual{ambA, ambB})
	}

	// リング状: 最後の村と最初の村も交流の可能性
	if len(t.Villages) > 2 && rand.Float64() <= t.InterRate {
		first := t.Villages[0]
		last := t.Villages[len(t.Villages)-1]
		ambA := first.Members[rand.Intn(len(first.Members))]
		ambB := last.Members[rand.Intn(len(last.Members))]
		pairs = append(pairs, [2]*Individual{ambA, ambB})
	}

	return pairs
}

// allPairs は個体リストから全ペアを生成する
func allPairs(inds []*Individual) [][2]*Individual {
	var pairs [][2]*Individual
	for i := 0; i < len(inds); i++ {
		for j := i + 1; j < len(inds); j++ {
			pairs = append(pairs, [2]*Individual{inds[i], inds[j]})
		}
	}
	return pairs
}
