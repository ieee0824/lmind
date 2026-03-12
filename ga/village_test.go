package ga

import (
	"fmt"
	"math"
	"testing"

	"github.com/ieee0824/lmind/config"
)

func TestNewTopology(t *testing.T) {
	makeInds := func(n int) []*Individual {
		inds := make([]*Individual, n)
		for i := range n {
			inds[i] = &Individual{ID: fmt.Sprintf("ind%d", i), Params: *config.DefaultParams()}
		}
		return inds
	}

	tests := []struct {
		name         string
		n            int
		villageSize  int
		wantVillages int
		wantMinSize  int // 最小の村のサイズ
	}{
		{"4inds_size2", 4, 2, 2, 2},
		{"6inds_size2", 6, 2, 3, 2},
		{"6inds_size3", 6, 3, 2, 3},
		{"5inds_size2", 5, 2, 2, 2}, // 端数1人は前の村に合流→2+3
		{"3inds_size2", 3, 2, 1, 3}, // 端数1人合流→3人1村
		{"8inds_auto", 8, 0, 3, 2},  // 自動→size3 (>=6), 3+3+2
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inds := makeInds(tt.n)
			topo := NewTopology(inds, tt.villageSize, 5, 2, 0.5)

			if len(topo.Villages) != tt.wantVillages {
				t.Errorf("villages = %d, want %d", len(topo.Villages), tt.wantVillages)
			}

			// 全個体がどこかの村に属しているか
			total := 0
			for _, v := range topo.Villages {
				total += len(v.Members)
				if len(v.Members) < tt.wantMinSize {
					t.Errorf("village %s has %d members, want >= %d", v.ID, len(v.Members), tt.wantMinSize)
				}
			}
			if total != tt.n {
				t.Errorf("total members = %d, want %d", total, tt.n)
			}
		})
	}
}

func TestAllPairs(t *testing.T) {
	inds := []*Individual{
		{ID: "a"}, {ID: "b"}, {ID: "c"},
	}
	pairs := allPairs(inds)
	// 3C2 = 3
	if len(pairs) != 3 {
		t.Errorf("allPairs(3) = %d pairs, want 3", len(pairs))
	}

	// ペアの重複チェック
	seen := map[string]bool{}
	for _, p := range pairs {
		key := p[0].ID + "-" + p[1].ID
		if seen[key] {
			t.Errorf("duplicate pair: %s", key)
		}
		seen[key] = true
	}
}

func TestInterVillagePairs(t *testing.T) {
	inds := make([]*Individual, 8)
	for i := range 8 {
		inds[i] = &Individual{ID: fmt.Sprintf("ind%d", i), Params: *config.DefaultParams()}
	}

	// interRate=1.0 で必ず交流が発生する
	topo := NewTopology(inds, 2, 5, 2, 1.0)
	pairs := topo.randomInterPairs()

	// 4村のリング → 隣接3組 + リング1組 = 4組
	if len(pairs) != 4 {
		t.Errorf("interVillagePairs with rate=1.0: got %d pairs, want 4", len(pairs))
	}

	// interRate=0.0 で交流なし
	topo2 := NewTopology(inds, 2, 5, 2, 0.0)
	pairs2 := topo2.randomInterPairs()
	if len(pairs2) != 0 {
		t.Errorf("interVillagePairs with rate=0.0: got %d pairs, want 0", len(pairs2))
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want float64
	}{
		{"identical", []float64{1, 0, 0}, []float64{1, 0, 0}, 1.0},
		{"orthogonal", []float64{1, 0, 0}, []float64{0, 1, 0}, 0.0},
		{"opposite", []float64{1, 0}, []float64{-1, 0}, -1.0},
		{"similar", []float64{1, 1, 0}, []float64{1, 0, 0}, 1.0 / math.Sqrt(2)},
		{"empty", nil, nil, 0},
		{"mismatch", []float64{1}, []float64{1, 2}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cosineSimilarity(tt.a, tt.b)
			if math.Abs(got-tt.want) > 0.001 {
				t.Errorf("cosineSimilarity = %.4f, want %.4f", got, tt.want)
			}
		})
	}
}

func TestSemanticInterPairs(t *testing.T) {
	// 3村: village-0とvillage-2が遠い、village-0とvillage-1が近い
	villages := []*Village{
		{ID: "v0", Members: []*Individual{{ID: "ind0"}}},
		{ID: "v1", Members: []*Individual{{ID: "ind1"}}},
		{ID: "v2", Members: []*Individual{{ID: "ind2"}}},
	}
	vecs := [][]float64{
		{1, 0, 0},   // v0
		{0.9, 0.1, 0}, // v1: v0に近い
		{0, 0, 1},   // v2: v0から遠い
	}

	// interRate=1.0で全ペアが交流
	pairs := semanticInterPairs(villages, vecs, 1.0)
	if len(pairs) == 0 {
		t.Error("semanticInterPairs returned no pairs with rate=1.0")
	}

	// interRate=0.0で交流なし
	pairs0 := semanticInterPairs(villages, vecs, 0.0)
	if len(pairs0) != 0 {
		t.Errorf("semanticInterPairs with rate=0.0: got %d pairs, want 0", len(pairs0))
	}
}
