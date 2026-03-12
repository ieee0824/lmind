package ga

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
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

// VillageFitness は村の平均fitnessを返す
func (v *Village) AvgFitness() float64 {
	if len(v.Members) == 0 {
		return 0
	}
	var sum float64
	for _, m := range v.Members {
		sum += m.Fitness
	}
	return sum / float64(len(v.Members))
}

// ConquestResult は征服の結果
type ConquestResult struct {
	WinnerID   string
	LoserID    string
	WinnerAvg  float64
	LoserAvg   float64
	MigratedID string  // 移住したエースのID
	Replaced   int     // 子孫で埋めた人数
}

// Conquest は村の生存競争を行う。
// 1. 最下位の村を滅亡（全員消去）
// 2. 最上位の村のトップ個体が移住（パラメータそのまま）
// 3. 残りの枠は勝者の村のメンバーを親にした子孫で埋める
// → 勝者の村はエースを失い、敗者の村は勝者の遺伝子で再建される
func (t *Topology) Conquest(genIdx, basePort int) (*ConquestResult, []*Individual) {
	if len(t.Villages) < 2 {
		return nil, t.AllIndividuals()
	}

	// 村を平均fitnessでソート（高い順）
	ranked := make([]*Village, len(t.Villages))
	copy(ranked, t.Villages)
	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].AvgFitness() > ranked[j].AvgFitness()
	})

	winner := ranked[0]
	loser := ranked[len(ranked)-1]

	// 同じ村なら征服しない
	if winner.ID == loser.ID {
		return nil, t.AllIndividuals()
	}

	// 勝者の村のトップ個体を特定
	var ace *Individual
	for _, m := range winner.Members {
		if ace == nil || m.Fitness > ace.Fitness {
			ace = m
		}
	}

	result := &ConquestResult{
		WinnerID:   winner.ID,
		LoserID:    loser.ID,
		WinnerAvg:  winner.AvgFitness(),
		LoserAvg:   loser.AvgFitness(),
		MigratedID: ace.ID,
		Replaced:   len(loser.Members) - 1, // エース1人 + 子孫で残り
	}

	// ポート番号用のオフセット
	portOffset := 0
	for _, v := range t.Villages {
		portOffset += len(v.Members)
	}

	// 敗者の村を再建: エース移住 + 子孫で埋める
	newMembers := make([]*Individual, 0, len(loser.Members))

	// エースを移住（パラメータそのまま引き継ぎ）
	aceCopy := &Individual{
		ID:     ace.ID, // IDもそのまま
		Params: ace.Params,
		Port:   basePort + portOffset,
	}
	newMembers = append(newMembers, aceCopy)

	// 残りは勝者の村のメンバーから子孫を生成
	for i := 1; i < len(loser.Members); i++ {
		parentA := winner.Members[rand.Intn(len(winner.Members))]
		parentB := winner.Members[rand.Intn(len(winner.Members))]
		if len(winner.Members) > 1 {
			for parentB.ID == parentA.ID {
				parentB = winner.Members[rand.Intn(len(winner.Members))]
			}
		}
		child := Crossover(parentA, parentB, genIdx, portOffset+i)
		Mutate(child, 0.2)
		child.Port = basePort + portOffset + i
		newMembers = append(newMembers, child)
	}
	loser.Members = newMembers

	// 勝者の村からエースを除去
	remaining := make([]*Individual, 0, len(winner.Members)-1)
	for _, m := range winner.Members {
		if m.ID != ace.ID {
			remaining = append(remaining, m)
		}
	}
	winner.Members = remaining

	return result, t.AllIndividuals()
}

// allIndividuals は全村のメンバーをフラットに返す
func (t *Topology) AllIndividuals() []*Individual {
	var all []*Individual
	for _, v := range t.Villages {
		all = append(all, v.Members...)
	}
	return all
}
