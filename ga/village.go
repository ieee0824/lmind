package ga

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"github.com/ieee0824/lmind/ollama"
)

// OddballRate は「変わったやつ」の発生確率（遠い村でも交流する割合）
const OddballRate = 0.2

// 吸収の閾値
const (
	AbsorptionSimThreshold = 0.85 // セマンティック類似度がこれ以上で「同化しすぎ」
	AbsorptionFitGap       = 0.02 // fitness差がこれ以上で強者が弱者を吸収
)

// Village は個体の社会的グループ（地理座標付き）
type Village struct {
	ID      string
	Members []*Individual
	X, Y    float64 // 地理的座標
}

// Topology は村の配置と対話構造を管理する
type Topology struct {
	Villages    []*Village
	IntraRounds int     // 村内対話の往復数
	InterRounds int     // 村間対話の往復数
	InterRate   float64 // 村間交流の基本確率 (0.0-1.0)
}

// NewTopology は個体群を村に分割し、地理座標を割り当てる
// prevPositions が与えられた場合はその座標を引き継ぐ（世代間の地理的連続性）
func NewTopology(inds []*Individual, villageSize int, intraRounds, interRounds int, interRate float64, prevPositions [][2]float64) *Topology {
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

	// 地理座標の割り当て
	if len(prevPositions) >= len(t.Villages) {
		// 前世代の座標を引き継ぐ
		for i, v := range t.Villages {
			v.X = prevPositions[i][0]
			v.Y = prevPositions[i][1]
		}
	} else {
		// 単位円上に等間隔配置
		n := len(t.Villages)
		for i, v := range t.Villages {
			angle := 2 * math.Pi * float64(i) / float64(n)
			v.X = math.Cos(angle)
			v.Y = math.Sin(angle)
		}
	}

	return t
}

// Positions は村の座標を返す（次世代に引き継ぎ用）
func (t *Topology) Positions() [][2]float64 {
	pos := make([][2]float64, len(t.Villages))
	for i, v := range t.Villages {
		pos[i] = [2]float64{v.X, v.Y}
	}
	return pos
}

// GeoDist は2つの村のユークリッド距離を返す
func GeoDist(a, b *Village) float64 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	return math.Sqrt(dx*dx + dy*dy)
}

// RunConversations は村内の密な対話と村間の地理的距離ベースの交流を実行する
func (t *Topology) RunConversations(ctx context.Context, seed string, ollamaClient ...*ollama.Client) error {
	// Phase 1: 村内の密な対話（総当たり、並列）
	fmt.Printf("  村内対話 (%d村, %d往復)...\n", len(t.Villages), t.IntraRounds)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var errs []error

	for _, v := range t.Villages {
		pairs := allPairs(v.Members)
		for _, pair := range pairs {
			fmt.Printf("    [%s] %s <-> %s (%.1f, %.1f)\n", v.ID, pair[0].ID, pair[1].ID, v.X, v.Y)
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

	// Phase 2: 村間の交流（地理的距離ベース）
	if len(t.Villages) < 2 {
		return nil
	}

	interPairs := t.geoInterPairs()
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

// geoInterPairs は地理的距離に基づいて村間交流ペアを生成する
// 近い村ほど交流確率が高い: prob = interRate * exp(-dist / scale)
func (t *Topology) geoInterPairs() [][2]*Individual {
	// 平均距離をスケールとして使う
	var totalDist float64
	var count int
	for i := 0; i < len(t.Villages); i++ {
		for j := i + 1; j < len(t.Villages); j++ {
			totalDist += GeoDist(t.Villages[i], t.Villages[j])
			count++
		}
	}
	scale := 1.0
	if count > 0 {
		scale = totalDist / float64(count)
	}

	var pairs [][2]*Individual
	for i := 0; i < len(t.Villages); i++ {
		for j := i + 1; j < len(t.Villages); j++ {
			vA := t.Villages[i]
			vB := t.Villages[j]
			dist := GeoDist(vA, vB)

			// 距離ベースの交流確率
			prob := t.InterRate * math.Exp(-dist/scale)

			// Oddball: 低確率で遠い村でも交流
			if rand.Float64() < OddballRate {
				prob = t.InterRate
			}

			if rand.Float64() > prob {
				continue
			}

			ambA := vA.Members[randN(len(vA.Members))]
			ambB := vB.Members[randN(len(vB.Members))]
			pairs = append(pairs, [2]*Individual{ambA, ambB})

			label := "近隣"
			if dist > scale {
				label = "遠方"
			}
			fmt.Printf("    [%s] %s(%s) <-> %s(%s) 距離=%.2f\n",
				label, vA.ID, ambA.ID, vB.ID, ambB.ID, dist)
		}
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

// AvgFitness は村の平均fitnessを返す
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

// AbsorptionResult は吸収の結果
type AbsorptionResult struct {
	AbsorberID  string
	AbsorbedID  string
	AbsorberAvg float64
	AbsorbedAvg float64
	Similarity  float64
	GeoDist     float64
	MigratedID  string // 移住したエースのID
	Replaced    int    // 置き換えられた人数
}

// Absorb は村の吸収判定を行う。
// 条件: 地理的に近い + セマンティック類似度が高い（独自性を失った） + fitness差がある
// → 弱い村のメンバーの半数が強い村の子孫で置き換えられる
// 村自体は消滅しない（地理的なスロットは維持）
func (t *Topology) Absorb(ctx context.Context, client *ollama.Client, genIdx, basePort int) (*AbsorptionResult, error) {
	if len(t.Villages) < 2 {
		return nil, nil
	}

	// 吸収距離の閾値: 平均距離の半分以下なら「近い」
	var totalDist float64
	var count int
	for i := 0; i < len(t.Villages); i++ {
		for j := i + 1; j < len(t.Villages); j++ {
			totalDist += GeoDist(t.Villages[i], t.Villages[j])
			count++
		}
	}
	distThreshold := 1.5 // デフォルト
	if count > 0 {
		distThreshold = totalDist / float64(count) * 0.8
	}

	// セマンティックベクトルを計算
	var vecs [][]float64
	if client != nil {
		fmt.Println("  吸収判定: embedding計算中...")
		var err error
		vecs, err = EmbedVillages(ctx, client, t.Villages)
		if err != nil {
			fmt.Printf("    embed失敗: %v\n", err)
		}
	}

	// 最も吸収されやすいペアを探す
	type candidate struct {
		absorberIdx int
		absorbedIdx int
		similarity  float64
		geoDist     float64
	}
	var best *candidate

	for i := 0; i < len(t.Villages); i++ {
		for j := i + 1; j < len(t.Villages); j++ {
			vA := t.Villages[i]
			vB := t.Villages[j]

			// 地理的距離チェック
			dist := GeoDist(vA, vB)
			if dist > distThreshold {
				continue
			}

			// セマンティック類似度チェック
			if vecs == nil || vecs[i] == nil || vecs[j] == nil {
				continue
			}
			sim := cosineSimilarity(vecs[i], vecs[j])
			if sim < AbsorptionSimThreshold {
				fmt.Printf("    %s <-> %s: 距離=%.2f, 類似度=%.3f (独自性あり→共存)\n",
					vA.ID, vB.ID, dist, sim)
				continue
			}

			// fitness差チェック
			avgA := vA.AvgFitness()
			avgB := vB.AvgFitness()
			gap := math.Abs(avgA - avgB)
			if gap < AbsorptionFitGap {
				fmt.Printf("    %s <-> %s: 距離=%.2f, 類似度=%.3f, 差=%.4f (拮抗→共存)\n",
					vA.ID, vB.ID, dist, sim, gap)
				continue
			}

			// 吸収候補
			absorberIdx, absorbedIdx := i, j
			if avgB > avgA {
				absorberIdx, absorbedIdx = j, i
			}

			if best == nil || sim > best.similarity {
				best = &candidate{
					absorberIdx: absorberIdx,
					absorbedIdx: absorbedIdx,
					similarity:  sim,
					geoDist:     dist,
				}
			}
		}
	}

	if best == nil {
		fmt.Println("  吸収判定: 全村が独自性を維持 → 吸収なし")
		return nil, nil
	}

	absorber := t.Villages[best.absorberIdx]
	absorbed := t.Villages[best.absorbedIdx]

	// エース（吸収側のトップ）を特定
	var ace *Individual
	for _, m := range absorber.Members {
		if ace == nil || m.Fitness > ace.Fitness {
			ace = m
		}
	}

	// 被吸収村のメンバーをfitnessでソート（低い順）
	sort.Slice(absorbed.Members, func(i, j int) bool {
		return absorbed.Members[i].Fitness < absorbed.Members[j].Fitness
	})

	// 被吸収村の下位半分を置き換え
	replaceCount := len(absorbed.Members) / 2
	if replaceCount < 1 {
		replaceCount = 1
	}

	// ポート番号用のオフセット
	portOffset := 0
	for _, v := range t.Villages {
		portOffset += len(v.Members)
	}

	// 下位メンバーを吸収側の子孫で置き換え
	for i := 0; i < replaceCount; i++ {
		parentA := absorber.Members[rand.Intn(len(absorber.Members))]
		parentB := absorber.Members[rand.Intn(len(absorber.Members))]
		if len(absorber.Members) > 1 {
			for parentB.ID == parentA.ID {
				parentB = absorber.Members[rand.Intn(len(absorber.Members))]
			}
		}
		child := Crossover(parentA, parentB, genIdx, portOffset+i)
		Mutate(child, 0.2)
		child.Port = basePort + portOffset + i
		absorbed.Members[i] = child // 下位から置き換え
	}

	result := &AbsorptionResult{
		AbsorberID:  absorber.ID,
		AbsorbedID:  absorbed.ID,
		AbsorberAvg: absorber.AvgFitness(),
		AbsorbedAvg: absorbed.AvgFitness(),
		Similarity:  best.similarity,
		GeoDist:     best.geoDist,
		MigratedID:  ace.ID,
		Replaced:    replaceCount,
	}

	return result, nil
}

// ConquestResult は征服の結果（後方互換用）
type ConquestResult struct {
	WinnerID   string
	LoserID    string
	WinnerAvg  float64
	LoserAvg   float64
	MigratedID string
	Replaced   int
}

// Conquest は村の生存競争を行う（後方互換用、Absorbの使用を推奨）
func (t *Topology) Conquest(genIdx, basePort int) (*ConquestResult, []*Individual) {
	if len(t.Villages) < 2 {
		return nil, t.AllIndividuals()
	}

	ranked := make([]*Village, len(t.Villages))
	copy(ranked, t.Villages)
	sort.Slice(ranked, func(i, j int) bool {
		return ranked[i].AvgFitness() > ranked[j].AvgFitness()
	})

	winner := ranked[0]
	loser := ranked[len(ranked)-1]

	if winner.ID == loser.ID {
		return nil, t.AllIndividuals()
	}

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
		Replaced:   len(loser.Members) - 1,
	}

	portOffset := 0
	for _, v := range t.Villages {
		portOffset += len(v.Members)
	}

	newMembers := make([]*Individual, 0, len(loser.Members))
	aceCopy := &Individual{
		ID:     ace.ID,
		Params: ace.Params,
		Port:   basePort + portOffset,
	}
	newMembers = append(newMembers, aceCopy)

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

	remaining := make([]*Individual, 0, len(winner.Members)-1)
	for _, m := range winner.Members {
		if m.ID != ace.ID {
			remaining = append(remaining, m)
		}
	}
	winner.Members = remaining

	return result, t.AllIndividuals()
}

// AllIndividuals は全村のメンバーをフラットに返す
func (t *Topology) AllIndividuals() []*Individual {
	var all []*Individual
	for _, v := range t.Villages {
		all = append(all, v.Members...)
	}
	return all
}
