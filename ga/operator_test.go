package ga

import (
	"testing"

	"github.com/ieee0824/lmind/config"
)

func TestTournamentSelect(t *testing.T) {
	pop := []*Individual{
		{ID: "a", Fitness: 0.1},
		{ID: "b", Fitness: 0.5},
		{ID: "c", Fitness: 0.9},
	}

	// k=3で全個体が候補に入るので、最良個体が選ばれやすい
	wins := map[string]int{}
	for range 100 {
		selected := TournamentSelect(pop, 3)
		wins[selected.ID]++
	}

	// cが最も多く選ばれるべき
	if wins["c"] < wins["a"] {
		t.Errorf("expected 'c' to win more than 'a', got c=%d a=%d", wins["c"], wins["a"])
	}
}

func TestCrossover(t *testing.T) {
	a := &Individual{
		ID:     "parent-a",
		Params: *config.DefaultParams(),
	}
	b := &Individual{
		ID:     "parent-b",
		Params: *config.DefaultParams(),
	}
	// 片方のパラメータを大きく変えておく
	b.Params.Intervals.Frontal = 100
	b.Params.Gain.MaxGain = 5.0

	child := Crossover(a, b, 1, 0)

	if child.ID != "gen1-ind0" {
		t.Errorf("unexpected ID: %s", child.ID)
	}

	// 子のパラメータは親AまたはBの値であるべき
	frontal := child.Params.Intervals.Frontal
	if frontal != a.Params.Intervals.Frontal && frontal != b.Params.Intervals.Frontal {
		t.Errorf("child frontal %f is neither parent's value", frontal)
	}
}

func TestMutateParams(t *testing.T) {
	p := config.DefaultParams()
	original := *p
	mutateParams(p, 0.3)

	// 少なくとも何かが変わっているはず
	changed := false
	if p.Intervals.Frontal != original.Intervals.Frontal {
		changed = true
	}
	if p.Gain.MaxGain != original.Gain.MaxGain {
		changed = true
	}
	if p.Attention.Baseline != original.Attention.Baseline {
		changed = true
	}
	if !changed {
		t.Error("mutateParams did not change any parameter (statistically unlikely)")
	}

	// 境界チェック
	if p.Intervals.Frontal < 5 || p.Intervals.Frontal > 120 {
		t.Errorf("frontal interval out of bounds: %f", p.Intervals.Frontal)
	}
	if p.Gain.MaxGain < 1.0 || p.Gain.MaxGain > 5.0 {
		t.Errorf("max gain out of bounds: %f", p.Gain.MaxGain)
	}
}

func TestNextGeneration(t *testing.T) {
	current := []*Individual{
		{ID: "gen0-ind0", Fitness: 0.8, Params: *config.DefaultParams()},
		{ID: "gen0-ind1", Fitness: 0.3, Params: *config.DefaultParams()},
		{ID: "gen0-ind2", Fitness: 0.5, Params: *config.DefaultParams()},
		{ID: "gen0-ind3", Fitness: 0.6, Params: *config.DefaultParams()},
	}

	next := NextGeneration(current, 1, 9000)

	if len(next) != len(current) {
		t.Errorf("expected %d individuals, got %d", len(current), len(next))
	}

	// ポートが正しく割り当てられているか
	for i, ind := range next {
		if ind.Port != 9000+i {
			t.Errorf("ind %d: expected port %d, got %d", i, 9000+i, ind.Port)
		}
	}
}
