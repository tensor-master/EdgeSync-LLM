package cache

import (
	"database/sql"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/bossandboss/EdgeSync-LLM/core"
)

// State representing the caching branch evaluated
type CacheState string

const (
	StateExact   CacheState = "EXACT"
	StatePartial CacheState = "PARTIAL"
	StateMiss    CacheState = "MISS"
)

// DifferentialResult bundles the response and execution diagnostics
type DifferentialResult struct {
	State          CacheState
	Response       string
	ParentEntryID  int64
	DivergenceIdx  int
	Similarity     float32
	LookupTimeMs   float64
	InferenceTimeMs float64
	NpuReduction   float64 // NPU workload savings percentage (100% for EXACT, ~75% for PARTIAL, 0% for MISS)
}

// DifferentialEngine manages the routing of LLM inferences through the HNSW semantic index and WAL cache database.
type DifferentialEngine struct {
	HNSW       *core.HNSW
	DB         *sql.DB
	Embedding  func(text string) ([]float32, error)      // Fn pointer to create embeddings
	LocalLLM   func(prompt string, prefix string) (string, error) // Fn pointer to run local LLM completion
}

// NewDifferentialEngine creates a new differential cache engine instance.
func NewDifferentialEngine(h *core.HNSW, db *sql.DB, embFn func(text string) ([]float32, error), llmFn func(prompt string, prefix string) (string, error)) *DifferentialEngine {
	return &DifferentialEngine{
		HNSW:      h,
		DB:        db,
		Embedding: embFn,
		LocalLLM:  llmFn,
	}
}

// Process evaluates the prompt, queries the HNSW index, executes the differential state logic, and logs metrics.
func (de *DifferentialEngine) Process(prompt string) (*DifferentialResult, error) {
	start := time.Now()

	// 1. Generate query embedding vector
	queryVec, err := de.Embedding(prompt)
	if err != nil {
		return nil, fmt.Errorf("embedding generation failed: %w", err)
	}

	lookupStart := time.Now()
	// 2. Query HNSW Index for approximate nearest neighbor
	neighbors := de.HNSW.Search(queryVec, 1)
	lookupTime := time.Since(lookupStart).Seconds() * 1000.0

	// If index is empty or has no neighbor, trigger full MISS
	if len(neighbors) == 0 {
		return de.handleMiss(prompt, queryVec, start, lookupTime, 0.0)
	}

	bestMatch := neighbors[0]
	similarity := bestMatch.Similarity

	// Evaluate Similarity Branches
	switch {
	case similarity >= 0.92:
		// BRANCH 1: EXACT HIT (>0.92) -> Returns cached response directly, 0% NPU usage
		return de.handleExactHit(bestMatch.ID, similarity, start, lookupTime)

	case similarity >= 0.75 && similarity < 0.92:
		// BRANCH 2: PARTIAL HIT (0.75 - 0.92) -> Segment prompt, fetch cached prefix, generate delta only
		return de.handlePartialHit(prompt, queryVec, bestMatch.ID, similarity, start, lookupTime)

	default:
		// BRANCH 3: MISS (<0.75) -> Perform full local NPU inference, store index + DB
		return de.handleMiss(prompt, queryVec, start, lookupTime, similarity)
	}
}

// handleExactHit updates cache statistics and returns the fully cached response with zero NPU usage.
func (de *DifferentialEngine) handleExactHit(id int, similarity float32, start time.Time, lookupTime float64) (*DifferentialResult, error) {
	var response string
	err := de.DB.QueryRow("SELECT response FROM cache_entries WHERE id = ?", id).Scan(&response)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch cached response: %w", err)
	}

	// Increment cache hits in SQLite (non-blocking)
	go func() {
		_, _ = de.DB.Exec("UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = ?", id)
	}()

	totalLatency := time.Since(start).Seconds() * 1000.0

	return &DifferentialResult{
		State:           StateExact,
		Response:        response,
		ParentEntryID:   int64(id),
		DivergenceIdx:   0,
		Similarity:      similarity,
		LookupTimeMs:    lookupTime,
		InferenceTimeMs: totalLatency - lookupTime,
		NpuReduction:    100.0, // 100% NPU reduction!
	}, nil
}

// handlePartialHit extracts semantic overlaps, runs completion for the divergent portion, and caches the delta.
func (de *DifferentialEngine) handlePartialHit(prompt string, queryVec []float32, parentID int, similarity float32, start time.Time, lookupTime float64) (*DifferentialResult, error) {
	var cachedPrompt, cachedResponse string
	err := de.DB.QueryRow("SELECT prompt, response FROM cache_entries WHERE id = ?", parentID).Scan(&cachedPrompt, &cachedResponse)
	if err != nil {
		// Fallback to MISS if DB query fails
		return de.handleMiss(prompt, queryVec, start, lookupTime, similarity)
	}

	// Calculate divergence point in prompts using Levenshtein or prefix match helper
	divergenceIdx := findDivergenceIndex(cachedPrompt, prompt)
	
	// Slice the cached response to act as our prefix to pre-populate LLM KV cache (representing cached prefix recovery)
	// For local LLMs, pre-populating KV-cache with a prefix speeds up TTFT and saves GPU/NPU cycles.
	words := strings.Fields(cachedResponse)
	prefixLimit := len(words) / 2
	if prefixLimit < 1 {
		prefixLimit = 1
	}
	cachedPrefix := strings.Join(words[:prefixLimit], " ")

	// Ask local LLM to generate only the missing delta based on prompt delta
	inferenceStart := time.Now()
	deltaResponse, err := de.LocalLLM(prompt, cachedPrefix)
	if err != nil {
		return nil, fmt.Errorf("partial local inference failed: %w", err)
	}
	inferenceTime := time.Since(inferenceStart).Seconds() * 1000.0

	// Merge prefix and newly generated delta
	fullResponse := cachedPrefix + " " + deltaResponse

	// Save the delta in SQLite (non-blocking write)
	go func() {
		_, _ = de.DB.Exec("INSERT INTO cache_deltas (parent_entry_id, divergence_idx, delta_response) VALUES (?, ?, ?)", parentID, divergenceIdx, deltaResponse)
		_, _ = de.DB.Exec("UPDATE cache_entries SET hit_count = hit_count + 1 WHERE id = ?", parentID)
	}()

	// Partial reduction estimate: prefix size compared to total response size
	npuReduction := (float64(len(cachedPrefix)) / float64(len(fullResponse))) * 100.0
	if npuReduction > 90.0 {
		npuReduction = 90.0 // Cap partial hit savings at 90%
	} else if npuReduction < 20.0 {
		npuReduction = 20.0
	}

	return &DifferentialResult{
		State:           StatePartial,
		Response:        fullResponse,
		ParentEntryID:   int64(parentID),
		DivergenceIdx:   divergenceIdx,
		Similarity:      similarity,
		LookupTimeMs:    lookupTime,
		InferenceTimeMs: inferenceTime,
		NpuReduction:    npuReduction,
	}, nil
}

// handleMiss performs full local inference and populates the cache databases.
func (de *DifferentialEngine) handleMiss(prompt string, queryVec []float32, start time.Time, lookupTime float64, similarity float32) (*DifferentialResult, error) {
	// Execute full local LLM inference (with empty prefix)
	inferenceStart := time.Now()
	fullResponse, err := de.LocalLLM(prompt, "")
	if err != nil {
		return nil, fmt.Errorf("local LLM inference failed: %w", err)
	}
	inferenceTime := time.Since(inferenceStart).Seconds() * 1000.0

	// Store result in WAL Database
	expiresAt := time.Now().Add(24 * time.Hour) // Standard 24h confirmation TTL
	
	var lastInsertID int64
	err = de.DB.QueryRow(
		"INSERT INTO cache_entries (prompt, embedding, response, expires_at) VALUES (?, ?, ?, ?) RETURNING id",
		prompt, float32SliceToBytes(queryVec), fullResponse, expiresAt,
	).Scan(&lastInsertID)

	if err != nil {
		// Attempt without RETURNING if sqlite driver lacks support in go version
		res, err2 := de.DB.Exec("INSERT INTO cache_entries (prompt, embedding, response, expires_at) VALUES (?, ?, ?, ?)", prompt, float32SliceToBytes(queryVec), fullResponse, expiresAt)
		if err2 == nil {
			lastInsertID, _ = res.LastInsertId()
		}
	}

	// Insert into HNSW graph index for future queries
	if lastInsertID > 0 {
		de.HNSW.Insert(int(lastInsertID), queryVec)
	}

	return &DifferentialResult{
		State:           StateMiss,
		Response:        fullResponse,
		ParentEntryID:   lastInsertID,
		DivergenceIdx:   -1,
		Similarity:      similarity,
		LookupTimeMs:    lookupTime,
		InferenceTimeMs: inferenceTime,
		NpuReduction:    0.0, // Full NPU cost incurred
	}, nil
}

// findDivergenceIndex returns the index of character alignment divergence between two prompts.
func findDivergenceIndex(a, b string) int {
	lenA, lenB := len(a), len(b)
	minLen := lenA
	if lenB < minLen {
		minLen = lenB
	}
	for i := 0; i < minLen; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return minLen
}

// Helper to convert []float32 slice to flat SQLite BLOB byte format.
func float32SliceToBytes(slice []float32) []byte {
	// Simple cast mapping, 4 bytes per float32
	buf := make([]byte, len(slice)*4)
	for i, f := range slice {
		u := math.Float32bits(f)
		buf[i*4] = byte(u)
		buf[i*4+1] = byte(u >> 8)
		buf[i*4+2] = byte(u >> 16)
		buf[i*4+3] = byte(u >> 24)
	}
	return buf
}
