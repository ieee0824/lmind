package ga

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strings"

	"github.com/ieee0824/lmind/ollama"
)

// thoughtEntry は /api/thoughts のレスポンス要素
type thoughtEntry struct {
	From    string `json:"from"`
	Content string `json:"content"`
}

// fetchThoughts は個体の直近の思考を取得する
func fetchThoughts(ctx context.Context, port int) ([]thoughtEntry, error) {
	url := fmt.Sprintf("http://localhost:%d/api/thoughts", port)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("thoughts: status %d", resp.StatusCode)
	}

	var entries []thoughtEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, err
	}
	return entries, nil
}

// villageDigest は村のメンバーの思考を結合して1つのテキストにする
func villageDigest(ctx context.Context, v *Village) string {
	var parts []string
	for _, ind := range v.Members {
		thoughts, err := fetchThoughts(ctx, ind.Port)
		if err != nil {
			continue
		}
		for _, t := range thoughts {
			if t.Content != "" {
				parts = append(parts, t.Content)
			}
		}
	}
	return strings.Join(parts, " ")
}

// EmbedVillages は各村の思考をembedしてベクトルを返す
func EmbedVillages(ctx context.Context, client *ollama.Client, villages []*Village) ([][]float64, error) {
	vecs := make([][]float64, len(villages))
	for i, v := range villages {
		digest := villageDigest(ctx, v)
		if digest == "" {
			continue
		}
		// 長すぎるテキストは末尾を切り詰め（nomic-embed-textの上限考慮）
		if len(digest) > 8000 {
			digest = digest[:8000]
		}
		vec, err := client.Embedding(ctx, "nomic-embed-text", digest)
		if err != nil {
			fmt.Printf("    [%s] embed失敗: %v\n", v.ID, err)
			continue
		}
		vecs[i] = vec
	}
	return vecs, nil
}

// cosineSimilarity はコサイン類似度を計算する
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// villagePairByDistance は村ペアをコサイン距離（遠い順）でソートする
type villagePairByDistance struct {
	iA, iB   int     // 村のインデックス
	distance float64 // 1 - cosine_similarity
}

// semanticInterPairs は小世界ネットワーク的な村間ペアリングを行う。
// - 大半は意味的に近い村同士が交流（共鳴を深める）
// - 一部（OddballRate）は意味的に遠い村から「変わったやつ」が来る（意外性で刺激）
func semanticInterPairs(villages []*Village, vecs [][]float64, interRate float64) [][2]*Individual {
	// 全村ペアの距離を計算
	var candidates []villagePairByDistance
	for i := 0; i < len(villages); i++ {
		if vecs[i] == nil {
			continue
		}
		for j := i + 1; j < len(villages); j++ {
			if vecs[j] == nil {
				continue
			}
			dist := 1.0 - cosineSimilarity(vecs[i], vecs[j])
			candidates = append(candidates, villagePairByDistance{i, j, dist})
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	// 近い順にソート（通常交流用）
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].distance < candidates[j].distance
	})

	var pairs [][2]*Individual

	for _, c := range candidates {
		if rand.Float64() > interRate {
			continue
		}

		vA := villages[c.iA]
		vB := villages[c.iB]

		// OddballRate の確率で「変わったやつ」ルート
		// → 最も遠い村から代表者を引っ張ってくる
		isOddball := rand.Float64() < OddballRate
		if isOddball && len(candidates) > 1 {
			// 一番遠いペアを探す
			farthest := candidates[len(candidates)-1]
			vA = villages[farthest.iA]
			vB = villages[farthest.iB]
			ambA := vA.Members[randN(len(vA.Members))]
			ambB := vB.Members[randN(len(vB.Members))]
			pairs = append(pairs, [2]*Individual{ambA, ambB})
			fmt.Printf("    [oddball] %s(%s) <-> %s(%s) 距離=%.3f\n",
				vA.ID, ambA.ID, vB.ID, ambB.ID, farthest.distance)
		} else {
			ambA := vA.Members[randN(len(vA.Members))]
			ambB := vB.Members[randN(len(vB.Members))]
			pairs = append(pairs, [2]*Individual{ambA, ambB})
			fmt.Printf("    [近隣] %s(%s) <-> %s(%s) 距離=%.3f\n",
				vA.ID, ambA.ID, vB.ID, ambB.ID, c.distance)
		}
	}

	return pairs
}

// randN は [0,n) の乱数を返す
func randN(n int) int {
	if n <= 0 {
		return 0
	}
	return rand.Intn(n)
}
