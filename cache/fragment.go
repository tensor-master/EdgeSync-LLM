// Package cache defines the canonical KVFragment structure —
// the atomic unit of reusable context for any local LLM engine.
//
// DESIGN RATIONALE
// ────────────────
// A KV cache (Key-Value cache) is the matrix of attention keys and values
// computed during a transformer forward pass. Reusing it avoids recomputing
// the same prefix tokens on every request — the dominant cost on edge hardware.
//
// A KVFragment captures ONE contiguous slice of that matrix, identified by:
//   - its token range  [TokenStart, TokenEnd)
//   - its layer range  [LayerStart, LayerEnd)  (sparse caching: cache every N layers)
//   - the model it belongs to (architecture + quantization must match exactly)
//   - a content hash to detect staleness
//   - a TTL to enforce eviction without external GC
//
// This definition is intentionally engine-agnostic.
// The raw tensor bytes (Keys, Values) are opaque []byte blobs.
// Each engine adapter (llama.cpp, MLC-LLM, ONNX Runtime) is responsible
// for serializing / deserializing them in its own format.
// See adapter/interface.go for the contract.
package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"
)

// ─────────────────────────────────────────────────────────────────────────────
// Dimension constants — these are the formal grammar of a fragment.
// ─────────────────────────────────────────────────────────────────────────────

const (
	// FragmentGranularityTokens is the minimum token block size for a fragment.
	// Fragments smaller than this are not worth the index overhead.
	// Derived from llama.cpp rope-scaling: 64 tokens ≈ one RoPE period at 4096 ctx.
	FragmentGranularityTokens = 64

	// FragmentMaxTokenSpan is the largest token range a single fragment may cover.
	// Larger spans should be split into multiple fragments for partial reuse.
	// 2048 tokens ≈ one full screen of context on a typical mobile session.
	FragmentMaxTokenSpan = 2048

	// FragmentLayerStride is the layer-sampling interval for sparse caching.
	// Caching every layer costs 2× memory; every other layer (stride=2) gives
	// ~70% of the latency benefit at half the storage cost.
	// Set to 1 for full-density caching (high-RAM devices only).
	FragmentLayerStride = 2

	// DefaultTTLSession is the fragment lifetime for a live interactive session.
	DefaultTTLSession = 30 * time.Minute

	// DefaultTTLPersistent is the fragment lifetime for cross-session reuse.
	// Persistent fragments should only be stored for high-hit-count prompts.
	DefaultTTLPersistent = 7 * 24 * time.Hour

	// SimilarityExact is the cosine similarity threshold for exact fragment reuse.
	SimilarityExact = float32(0.92)

	// SimilarityPartial is the lower bound for partial delta reconstruction.
	SimilarityPartial = float32(0.75)
)

// ─────────────────────────────────────────────────────────────────────────────
// ModelID — identifies the exact model a fragment belongs to.
// A fragment is NEVER reusable across different ModelIDs.
// ─────────────────────────────────────────────────────────────────────────────

// ModelID uniquely identifies a model configuration.
// Two instances of the same weights but different quantization schemes
// produce incompatible KV tensors and MUST have different ModelIDs.
type ModelID struct {
	// Architecture is the transformer family, e.g. "llama", "gemma", "qwen", "phi".
	Architecture string

	// Name is the model variant, e.g. "Qwen2.5-0.5B", "Gemma-3-1B".
	Name string

	// Quantization describes the weight/KV quantization, e.g. "Q4_K_M", "F16", "INT8".
	// llama.cpp and MLC use different naming conventions — normalize to GGUF names here.
	Quantization string

	// ContextLength is the maximum context window this model was configured for.
	// RoPE scaling parameters change with context length, making KV tensors incompatible.
	ContextLength int

	// HeadDim is the key/value head dimension (d_k). Required to validate tensor shapes.
	HeadDim int

	// NumKVHeads is the number of KV attention heads (GQA/MQA aware).
	NumKVHeads int

	// NumLayers is the total number of transformer layers.
	NumLayers int
}

// String returns a human-readable model identifier for logging.
func (m ModelID) String() string {
	return fmt.Sprintf("%s/%s/%s/ctx%d", m.Architecture, m.Name, m.Quantization, m.ContextLength)
}

// Hash returns a stable 8-char hex fingerprint of the ModelID.
// Used as a prefix in storage keys to guarantee isolation between models.
func (m ModelID) Hash() string {
	h := sha256.Sum256([]byte(m.String()))
	return hex.EncodeToString(h[:])[:8]
}

// ─────────────────────────────────────────────────────────────────────────────
// KVFragment — the atomic unit of the cache.
// ─────────────────────────────────────────────────────────────────────────────

// KVFragment is a contiguous slice of a transformer KV cache.
// It covers a token range [TokenStart, TokenEnd) and a layer range
// [LayerStart, LayerEnd) with stride LayerStride.
//
// Storage layout of Keys and Values:
//
//	Keys[layer_idx][head][token][d_k]   — packed as flat []byte, row-major
//	Values[layer_idx][head][token][d_v] — same layout
//
// The engine adapter is responsible for packing/unpacking these tensors.
// This struct intentionally holds raw bytes to remain engine-agnostic.
type KVFragment struct {
	// ── Identity ──────────────────────────────────────────────────────────────

	// ID is a globally unique identifier for this fragment (UUID v4 or ULID).
	ID string

	// Model identifies the exact model configuration this fragment belongs to.
	Model ModelID

	// ── Token range ───────────────────────────────────────────────────────────

	// TokenStart is the index of the first token in this fragment (inclusive).
	// Token indexing is 0-based from the start of the conversation context.
	TokenStart int

	// TokenEnd is the index past the last token (exclusive).
	// Span = TokenEnd - TokenStart. Must satisfy:
	//   span >= FragmentGranularityTokens
	//   span <= FragmentMaxTokenSpan
	TokenEnd int

	// ── Layer range ───────────────────────────────────────────────────────────

	// LayerStart is the first transformer layer index covered (inclusive).
	LayerStart int

	// LayerEnd is the layer index past the last covered layer (exclusive).
	LayerEnd int

	// LayerStride is the sampling interval. 1 = every layer, 2 = every other layer.
	// Must match the stride used during fragment creation to reconstruct correctly.
	LayerStride int

	// ── Tensor data ───────────────────────────────────────────────────────────

	// Keys holds the raw attention key tensors, engine-serialized.
	// Shape: [num_layers_covered × num_kv_heads × token_span × head_dim]
	// Encoding: engine-specific (llama.cpp uses ggml tensor binary, MLC uses DLPack).
	Keys []byte

	// Values holds the raw attention value tensors, engine-serialized.
	// Same layout as Keys.
	Values []byte

	// ── Content identity ──────────────────────────────────────────────────────

	// TokenIDs holds the input token IDs that produced this fragment.
	// Used to verify the fragment is still valid after model reload.
	// Also used as the "prompt prefix" for partial delta reconstruction.
	TokenIDs []int32

	// ContentHash is SHA-256(TokenIDs). Verified on load to detect corruption.
	ContentHash string

	// EmbeddingVector is the semantic embedding of the text prefix this fragment
	// was generated from. Used by the HNSW index for approximate matching.
	// Dimension must match the embedding model (typically 384 or 768).
	EmbeddingVector []float32

	// ── Lifecycle ─────────────────────────────────────────────────────────────

	// CreatedAt is the UTC timestamp when this fragment was first computed.
	CreatedAt time.Time

	// ExpiresAt is the UTC timestamp after which this fragment must be evicted.
	// Set to CreatedAt + DefaultTTLSession for interactive fragments,
	// CreatedAt + DefaultTTLPersistent for promoted high-hit-count fragments.
	ExpiresAt time.Time

	// HitCount is the number of times this fragment has been successfully reused.
	// Used by the eviction policy: fragments with HitCount > HitThresholdPromote
	// are promoted to persistent TTL.
	HitCount int

	// LastUsedAt is updated on every cache hit. Used for LRU eviction.
	LastUsedAt time.Time

	// ── Engine provenance ─────────────────────────────────────────────────────

	// Engine identifies which backend produced this fragment: "llamacpp", "mlc", "onnx".
	// A fragment produced by one engine CAN be consumed by another engine IF that engine
	// implements KVAdapter.Deserialize() for the source engine's format.
	// This field enables cross-engine reuse detection (e.g. llama.cpp → ONNX fallback).
	Engine string

	// EngineVersion is a semver string of the engine at creation time.
	// Tensor layouts sometimes change between major versions; this prevents silent corruption.
	EngineVersion string
}

// ─────────────────────────────────────────────────────────────────────────────
// Constructors and validation
// ─────────────────────────────────────────────────────────────────────────────

// NewFragment creates a KVFragment and validates its dimensional invariants.
// Returns an error if any invariant is violated — fail loudly at construction
// rather than silently storing a malformed fragment.
func NewFragment(
	id string,
	model ModelID,
	tokenStart, tokenEnd int,
	layerStart, layerEnd, layerStride int,
	keys, values []byte,
	tokenIDs []int32,
	embedding []float32,
	engine, engineVersion string,
	ttl time.Duration,
) (*KVFragment, error) {
	// ── Invariant checks ──────────────────────────────────────────────────────

	span := tokenEnd - tokenStart
	if span < FragmentGranularityTokens {
		return nil, fmt.Errorf(
			"fragment token span %d is below minimum granularity %d",
			span, FragmentGranularityTokens,
		)
	}
	if span > FragmentMaxTokenSpan {
		return nil, fmt.Errorf(
			"fragment token span %d exceeds maximum %d; split into multiple fragments",
			span, FragmentMaxTokenSpan,
		)
	}
	if layerStride < 1 {
		return nil, fmt.Errorf("layerStride must be >= 1, got %d", layerStride)
	}
	if layerEnd > model.NumLayers {
		return nil, fmt.Errorf(
			"layerEnd %d exceeds model NumLayers %d",
			layerEnd, model.NumLayers,
		)
	}
	if len(keys) == 0 || len(values) == 0 {
		return nil, fmt.Errorf("keys and values must not be empty")
	}
	if len(tokenIDs) != span {
		return nil, fmt.Errorf(
			"tokenIDs length %d does not match token span %d",
			len(tokenIDs), span,
		)
	}

	// ── Content hash ──────────────────────────────────────────────────────────
	// Hash the token IDs (not the tensors) — tensors are floats and may differ
	// by epsilon across quantization runs; token IDs are exact.
	hashInput := make([]byte, len(tokenIDs)*4)
	for i, tok := range tokenIDs {
		hashInput[i*4] = byte(tok)
		hashInput[i*4+1] = byte(tok >> 8)
		hashInput[i*4+2] = byte(tok >> 16)
		hashInput[i*4+3] = byte(tok >> 24)
	}
	h := sha256.Sum256(hashInput)
	contentHash := hex.EncodeToString(h[:])

	now := time.Now().UTC()

	return &KVFragment{
		ID:              id,
		Model:           model,
		TokenStart:      tokenStart,
		TokenEnd:        tokenEnd,
		LayerStart:      layerStart,
		LayerEnd:        layerEnd,
		LayerStride:     layerStride,
		Keys:            keys,
		Values:          values,
		TokenIDs:        tokenIDs,
		ContentHash:     contentHash,
		EmbeddingVector: embedding,
		CreatedAt:       now,
		ExpiresAt:       now.Add(ttl),
		HitCount:        0,
		LastUsedAt:      now,
		Engine:          engine,
		EngineVersion:   engineVersion,
	}, nil
}

// IsExpired returns true if the fragment has passed its TTL.
func (f *KVFragment) IsExpired() bool {
	return time.Now().UTC().After(f.ExpiresAt)
}

// TokenSpan returns the number of tokens covered by this fragment.
func (f *KVFragment) TokenSpan() int {
	return f.TokenEnd - f.TokenStart
}

// NumLayersCovered returns the number of transformer layers actually stored.
// With stride=2, a range [0,32) stores layers 0,2,4,...,30 → 16 layers.
func (f *KVFragment) NumLayersCovered() int {
	count := 0
	for l := f.LayerStart; l < f.LayerEnd; l += f.LayerStride {
		count++
	}
	return count
}

// SizeBytes returns the total memory footprint of the tensor blobs.
func (f *KVFragment) SizeBytes() int {
	return len(f.Keys) + len(f.Values)
}

// StorageKey returns a namespaced key for use in SQLite / LevelDB / file cache.
// Format: "<model_hash>/<engine>/<token_start>-<token_end>/<fragment_id>"
// This layout enables prefix-scan eviction per model without a full table scan.
func (f *KVFragment) StorageKey() string {
	return fmt.Sprintf("%s/%s/%06d-%06d/%s",
		f.Model.Hash(),
		f.Engine,
		f.TokenStart,
		f.TokenEnd,
		f.ID,
	)
}

// ─────────────────────────────────────────────────────────────────────────────
// Eviction policy helpers
// ─────────────────────────────────────────────────────────────────────────────

const (
	// HitThresholdPromote is the HitCount at which a session fragment
	// is promoted to persistent TTL (DefaultTTLPersistent).
	HitThresholdPromote = 5

	// HitThresholdEvict is the max number of fragments to keep per model per token range.
	// When the store exceeds this, the LRU fragment is evicted.
	HitThresholdEvict = 512
)

// ShouldPromote returns true if this fragment has been hit enough times
// to warrant promoting to a persistent TTL.
func (f *KVFragment) ShouldPromote() bool {
	return f.HitCount >= HitThresholdPromote && f.ExpiresAt.Sub(f.CreatedAt) < DefaultTTLPersistent
}

// Promote extends the fragment TTL to DefaultTTLPersistent.
func (f *KVFragment) Promote() {
	f.ExpiresAt = time.Now().UTC().Add(DefaultTTLPersistent)
}

// RecordHit updates hit count and last-used timestamp.
func (f *KVFragment) RecordHit() {
	f.HitCount++
	f.LastUsedAt = time.Now().UTC()
	if f.ShouldPromote() {
		f.Promote()
	}
}
