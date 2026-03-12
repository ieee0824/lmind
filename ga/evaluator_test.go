package ga

import (
	"math"
	"testing"

	"github.com/ieee0824/lmind/brain"
)

func TestFitness(t *testing.T) {
	tests := []struct {
		name     string
		snapshot brain.MetricsSnapshot
		wantMin  float64
		wantMax  float64
	}{
		{
			name:     "empty",
			snapshot: brain.MetricsSnapshot{TotalThoughts: 0},
			wantMin:  0,
			wantMax:  0,
		},
		{
			name: "ideal",
			snapshot: brain.MetricsSnapshot{
				ThoughtRepetitionRate:  0.0,
				MetaSelfReferenceCount: 0,
				UniqueTopicCount:       30,
				AvgResponseTimeMs:      0,
				TotalThoughts:          100,
			},
			wantMin: 0.9,
			wantMax: 1.0,
		},
		{
			name: "self-loop",
			snapshot: brain.MetricsSnapshot{
				ThoughtRepetitionRate:  0.9,
				MetaSelfReferenceCount: 50,
				UniqueTopicCount:       3,
				AvgResponseTimeMs:      25000,
				TotalThoughts:          100,
			},
			wantMin: 0.0,
			wantMax: 0.25,
		},
		{
			name: "moderate",
			snapshot: brain.MetricsSnapshot{
				ThoughtRepetitionRate:  0.3,
				MetaSelfReferenceCount: 5,
				UniqueTopicCount:       15,
				AvgResponseTimeMs:      5000,
				TotalThoughts:          50,
			},
			wantMin: 0.4,
			wantMax: 0.8,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := fitness(tt.snapshot)
			if math.IsNaN(got) {
				t.Error("fitness returned NaN")
			}
			if got < tt.wantMin || got > tt.wantMax {
				t.Errorf("fitness = %.4f, want [%.2f, %.2f]", got, tt.wantMin, tt.wantMax)
			}
		})
	}
}
