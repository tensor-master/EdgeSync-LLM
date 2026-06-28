package prefetch

import (
	"strings"
	"sync"
	"time"
)

// Predictor tracks historical user inputs and predicts upcoming prompts using an N-gram frequency model.
type Predictor struct {
	mu            sync.RWMutex
	history       []string          // Sliding window of the last 20 prompts
	transitions   map[string]map[string]int // Transition frequencies: [context_prompt][next_prompt] -> count
	maxHistorySize int
}

// PrefetchItem represents a prefetched prompt candidate queued for background caching.
type PrefetchItem struct {
	Prompt    string
	Predicted time.Time
	ExpiresAt time.Time // 30min TTL for prefetch, upgraded to 24h upon active use (confirmation)
	Confirmed bool
}

// NewPredictor instantiates a new N-gram prediction model.
func NewPredictor() *Predictor {
	return &Predictor{
		history:        make([]string, 0, 20),
		transitions:    make(map[string]map[string]int),
		maxHistorySize: 20,
	}
}

// RecordPrompt registers a new prompt in the history sliding window and updates transition probabilities.
func (p *Predictor) RecordPrompt(prompt string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Clean the prompt to avoid duplicate keys due to casing/whitespace
	cleanPrompt := strings.ToLower(strings.TrimSpace(prompt))
	if cleanPrompt == "" {
		return
	}

	// Update transition matrix if there is preceding context
	if len(p.history) > 0 {
		prev := p.history[len(p.history)-1]
		if _, exists := p.transitions[prev]; !exists {
			p.transitions[prev] = make(map[string]int)
		}
		p.transitions[prev][cleanPrompt]++
	}

	// Append to sliding history window (max size 20)
	if len(p.history) >= p.maxHistorySize {
		p.history = p.history[1:]
	}
	p.history = append(p.history, cleanPrompt)
}

// PredictNext returns the top-3 predicted subsequent prompts based on the current context.
func (p *Predictor) PredictNext() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if len(p.history) == 0 {
		return []string{}
	}

	currentPrompt := p.history[len(p.history)-1]
	nextFreq, exists := p.transitions[currentPrompt]
	if !exists {
		// Fallback: If no direct bigram transition exists, scan for common prefixes in history
		return p.predictByHistoryPrefix(currentPrompt)
	}

	// Sort transitions descending by frequency
	type score struct {
		prompt string
		count  int
	}
	var scores []score
	for prompt, count := range nextFreq {
		scores = append(scores, score{prompt, count})
	}

	// Sort simple bubble sorting to keep code dependency-free
	for i := 0; i < len(scores)-1; i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[i].count < scores[j].count {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	// Take top-3 predictions
	limit := 3
	if len(scores) < limit {
		limit = len(scores)
	}

	predictions := make([]string, limit)
	for i := 0; i < limit; i++ {
		predictions[i] = scores[i].prompt
	}

	return predictions
}

// predictByHistoryPrefix is a fallback predictor that matches current prefix tokens to find likely next sentences.
func (p *Predictor) predictByHistoryPrefix(current string) []string {
	tokens := strings.Fields(current)
	if len(tokens) == 0 {
		return []string{}
	}
	lastToken := tokens[len(tokens)-1]

	// Find any unique historical prompt starting with this token (excluding the exact current prompt)
	visited := make(map[string]bool)
	var matches []string

	for i := len(p.history) - 1; i >= 0; i-- {
		hPrompt := p.history[i]
		if hPrompt != current && strings.Contains(hPrompt, lastToken) && !visited[hPrompt] {
			visited[hPrompt] = true
			matches = append(matches, hPrompt)
			if len(matches) >= 3 {
				break
			}
		}
	}
	return matches
}

// CacheTTLManager handles TTL rules: 30 minutes for prefetched cache items and 24 hours once confirmed.
type CacheTTLManager struct {
	mu    sync.RWMutex
	items map[string]*PrefetchItem
}

func NewCacheTTLManager() *CacheTTLManager {
	return &CacheTTLManager{
		items: make(map[string]*PrefetchItem),
	}
}

// AddPrefetched adds a newly prefetched background candidate with a 30-minute TTL.
func (m *CacheTTLManager) AddPrefetched(prompt string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	m.items[prompt] = &PrefetchItem{
		Prompt:    prompt,
		Predicted: now,
		ExpiresAt: now.Add(30 * time.Minute), // 30min TTL for unconfirmed prefetches
		Confirmed: false,
	}
}

// ConfirmActivePrompt upgrades a prefetched item to fully active status with a 24-hour TTL upon active user hit.
func (m *CacheTTLManager) ConfirmActivePrompt(prompt string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	clean := strings.ToLower(strings.TrimSpace(prompt))
	if item, exists := m.items[clean]; exists {
		item.Confirmed = true
		item.ExpiresAt = time.Now().Add(24 * time.Hour) // Upgraded to 24h confirmed TTL!
	} else {
		// Create direct confirmed item
		m.items[clean] = &PrefetchItem{
			Prompt:    clean,
			Predicted: time.Now(),
			ExpiresAt: time.Now().Add(24 * time.Hour),
			Confirmed: true,
		}
	}
}

// GetExpiry checks if a prompt is valid and returns its expiration timestamp.
func (m *CacheTTLManager) GetExpiry(prompt string) (time.Time, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	clean := strings.ToLower(strings.TrimSpace(prompt))
	if item, exists := m.items[clean]; exists {
		if time.Now().Before(item.ExpiresAt) {
			return item.ExpiresAt, true
		}
	}
	return time.Time{}, false
}

// EvictExpired removes any prefetched or confirmed entries that have exceeded their designated TTL.
func (m *CacheTTLManager) EvictExpired() []string {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	var evicted []string

	for key, item := range m.items {
		if now.After(item.ExpiresAt) {
			evicted = append(evicted, key)
			delete(m.items, key)
		}
	}

	return evicted
}
