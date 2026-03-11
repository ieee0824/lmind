package brain

import (
	"context"
	"sync"

	memai "github.com/ieee0824/memAI-go"
)

// InMemoryStore はmemAI-goのMemoryStore[int64]のインメモリ実装
type InMemoryStore struct {
	mu      sync.RWMutex
	memories []memai.Memory[int64]
	nextID  int64
}

func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{}
}

func (s *InMemoryStore) GetMemories(_ context.Context) ([]memai.Memory[int64], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]memai.Memory[int64], len(s.memories))
	copy(result, s.memories)
	return result, nil
}

func (s *InMemoryStore) SaveMemory(_ context.Context, mem *memai.Memory[int64]) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.nextID++
	mem.ID = s.nextID
	s.memories = append(s.memories, *mem)
	return nil
}

func (s *InMemoryStore) DeleteMemory(_ context.Context, id int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i, m := range s.memories {
		if m.ID == id {
			s.memories = append(s.memories[:i], s.memories[i+1:]...)
			return nil
		}
	}
	return nil
}

func (s *InMemoryStore) UpdateBoost(_ context.Context, id int64, delta float64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i, m := range s.memories {
		if m.ID == id {
			s.memories[i].Boost = m.Boost + delta
			return nil
		}
	}
	return nil
}
