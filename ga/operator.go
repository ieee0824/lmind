package ga

import (
	"fmt"
	"math/rand"

	"github.com/ieee0824/lmind/config"
)

// TournamentSelect はトーナメント選択で親を選ぶ（k=3）
func TournamentSelect(pop []*Individual, k int) *Individual {
	best := pop[rand.Intn(len(pop))]
	for range k - 1 {
		candidate := pop[rand.Intn(len(pop))]
		if candidate.Fitness > best.Fitness {
			best = candidate
		}
	}
	return best
}

// Crossover は一様交叉で2親から子を生成する
// 各パラメータを50%の確率で親Aまたは親Bから選択
func Crossover(a, b *Individual, genIdx, childIdx int) *Individual {
	child := &Individual{
		ID: fmt.Sprintf("gen%d-ind%d", genIdx, childIdx),
	}

	pa, pb := &a.Params, &b.Params

	pick := func(va, vb float64) float64 {
		if rand.Float64() < 0.5 {
			return va
		}
		return vb
	}
	pickInt := func(va, vb int) int {
		if rand.Float64() < 0.5 {
			return va
		}
		return vb
	}

	child.Params = config.Params{
		Intervals: config.IntervalParams{
			Frontal:     pick(pa.Intervals.Frontal, pb.Intervals.Frontal),
			Temporal:    pick(pa.Intervals.Temporal, pb.Intervals.Temporal),
			Hippocampus: pick(pa.Intervals.Hippocampus, pb.Intervals.Hippocampus),
			Hypothesis:  pick(pa.Intervals.Hypothesis, pb.Intervals.Hypothesis),
			Critic:      pick(pa.Intervals.Critic, pb.Intervals.Critic),
			Curiosity:   pick(pa.Intervals.Curiosity, pb.Intervals.Curiosity),
			Grounding:   pick(pa.Intervals.Grounding, pb.Intervals.Grounding),
			Bonnou:      pick(pa.Intervals.Bonnou, pb.Intervals.Bonnou),
			Modulator:   pick(pa.Intervals.Modulator, pb.Intervals.Modulator),
		},
		Gain: config.GainParams{
			SkipThreshold:  pick(pa.Gain.SkipThreshold, pb.Gain.SkipThreshold),
			MaxGain:        pick(pa.Gain.MaxGain, pb.Gain.MaxGain),
			IdleDecayStart: pick(pa.Gain.IdleDecayStart, pb.Gain.IdleDecayStart),
			RecoverStep:    pick(pa.Gain.RecoverStep, pb.Gain.RecoverStep),
			DecayStep:      pick(pa.Gain.DecayStep, pb.Gain.DecayStep),
		},
		Stagnation: config.StagnationParams{
			DecayFrontal:    pick(pa.Stagnation.DecayFrontal, pb.Stagnation.DecayFrontal),
			DecayTemporal:   pick(pa.Stagnation.DecayTemporal, pb.Stagnation.DecayTemporal),
			DecayHypothesis: pick(pa.Stagnation.DecayHypothesis, pb.Stagnation.DecayHypothesis),
			BoostCuriosity:  pick(pa.Stagnation.BoostCuriosity, pb.Stagnation.BoostCuriosity),
			BoostGrounding:  pick(pa.Stagnation.BoostGrounding, pb.Stagnation.BoostGrounding),
			NoveltyRecovery: pick(pa.Stagnation.NoveltyRecovery, pb.Stagnation.NoveltyRecovery),
		},
		Hypothesis: config.HypothesisParams{
			DedupJaccard:      pick(pa.Hypothesis.DedupJaccard, pb.Hypothesis.DedupJaccard),
			SentimentPositive: pick(pa.Hypothesis.SentimentPositive, pb.Hypothesis.SentimentPositive),
			SentimentNegative: pick(pa.Hypothesis.SentimentNegative, pb.Hypothesis.SentimentNegative),
		},
		PredictionError: config.PredictionErrorParams{
			HighThreshold:   pick(pa.PredictionError.HighThreshold, pb.PredictionError.HighThreshold),
			MediumThreshold: pick(pa.PredictionError.MediumThreshold, pb.PredictionError.MediumThreshold),
		},
		Attention: config.AttentionParams{
			SalienceThreshold: pick(pa.Attention.SalienceThreshold, pb.Attention.SalienceThreshold),
			Baseline:          pick(pa.Attention.Baseline, pb.Attention.Baseline),
			GoalWeight:        pick(pa.Attention.GoalWeight, pb.Attention.GoalWeight),
			RepetitionPenalty: pick(pa.Attention.RepetitionPenalty, pb.Attention.RepetitionPenalty),
			MetaPenalty:       pick(pa.Attention.MetaPenalty, pb.Attention.MetaPenalty),
			UserBonus:         pick(pa.Attention.UserBonus, pb.Attention.UserBonus),
		},
		Critic: config.CriticParams{
			RepetitionJaccard:   pick(pa.Critic.RepetitionJaccard, pb.Critic.RepetitionJaccard),
			StagnationRepRate:   pick(pa.Critic.StagnationRepRate, pb.Critic.StagnationRepRate),
			StagnationDominance: pick(pa.Critic.StagnationDominance, pb.Critic.StagnationDominance),
		},
		Curiosity: config.CuriosityParams{
			FrequentThreshold: pickInt(pa.Curiosity.FrequentThreshold, pb.Curiosity.FrequentThreshold),
			RareThreshold:     pickInt(pa.Curiosity.RareThreshold, pb.Curiosity.RareThreshold),
		},
		Novelty: config.NoveltyParams{
			ScoreThreshold: pick(pa.Novelty.ScoreThreshold, pb.Novelty.ScoreThreshold),
		},
		Bonnou: config.BonnouParams{
			HotWordRetention:   pick(pa.Bonnou.HotWordRetention, pb.Bonnou.HotWordRetention),
			IntensityThreshold: pick(pa.Bonnou.IntensityThreshold, pb.Bonnou.IntensityThreshold),
		},
		History: config.HistoryParams{
			FreshCount: pickInt(pa.History.FreshCount, pb.History.FreshCount),
		},
		Personality: config.PersonalityParams{
			Warmth:     pick(pa.Personality.Warmth, pb.Personality.Warmth),
			Directness: pick(pa.Personality.Directness, pb.Personality.Directness),
			Humor:      pick(pa.Personality.Humor, pb.Personality.Humor),
			Curiosity:  pick(pa.Personality.Curiosity, pb.Personality.Curiosity),
			Verbosity:  pick(pa.Personality.Verbosity, pb.Personality.Verbosity),
			Empathy:    pick(pa.Personality.Empathy, pb.Personality.Empathy),
		},
		LTM: config.LTMParams{
			RecallWeight:     pick(pa.LTM.RecallWeight, pb.LTM.RecallWeight),
			RecallMaxResults: pickInt(pa.LTM.RecallMaxResults, pb.LTM.RecallMaxResults),
			RecallMinScore:   pick(pa.LTM.RecallMinScore, pb.LTM.RecallMinScore),
		},
	}

	return child
}

// Mutate は突然変異を適用する（確率mutationRateで各パラメータにガウス摂動）
func Mutate(ind *Individual, mutationRate float64) {
	if rand.Float64() < mutationRate {
		mutateParams(&ind.Params, 0.15) // σ = 範囲の15%
	}
}

// NextGeneration は現世代から次世代を生成する
// エリート1個体 + トーナメント選択→交叉→突然変異
func NextGeneration(current []*Individual, genIdx int, basePort int) []*Individual {
	size := len(current)
	next := make([]*Individual, 0, size)

	// エリート保存（最良個体をそのまま次世代へ）
	elite := current[0]
	for _, ind := range current[1:] {
		if ind.Fitness > elite.Fitness {
			elite = ind
		}
	}
	eliteCopy := &Individual{
		ID:     fmt.Sprintf("gen%d-ind0", genIdx),
		Params: elite.Params,
		Port:   basePort,
	}
	next = append(next, eliteCopy)

	// 残りは交叉+突然変異
	for i := 1; i < size; i++ {
		parentA := TournamentSelect(current, 3)
		parentB := TournamentSelect(current, 3)
		child := Crossover(parentA, parentB, genIdx, i)
		Mutate(child, 0.1)
		child.Port = basePort + i
		next = append(next, child)
	}
	return next
}
