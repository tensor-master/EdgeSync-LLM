// Package cache — Attention-sink sparsification for KV fragment storage.
//
// BACKGROUND: the attention sink phenomenon
// ────────────────────────────────────────────
// Published research on streaming/long-context LLMs (Xiao et al., "Efficient
// Streaming Language Models with Attention Sinks", ICLR 2024) shows that
// transformer attention disproportionately concentrates on a small number of
// tokens — typically the first few tokens of a sequence ("sink tokens") plus
// a sliding window of recent tokens. Middle-of-sequence tokens that are not
// in the recent window receive comparatively little attention weight.
//
// This is NOT specific to one model family — it's an artifact of how softmax
// attention behaves over long contexts, observed across LLaMA, GPT-NeoX,
// Falcon, and others.
//
// WHY THIS MATTERS FOR KV FRAGMENT STORAGE
// ───────────────────────────────────────────
// A KVFragment currently stores the full KV tensor for every token in its
// range. If only ~10-20% of tokens carry the attention weight that matters
// for continuation quality, storing 100% of tokens wastes memory and
// bandwidth proportionally.
//
// Sparsification identifies "pivot tokens" — the sink tokens plus the most
// recent window — and stores only those KV vectors. The discarded tokens'
// contribution to future attention is approximated as negligible, which is
// the same assumption StreamingLLM and follow-up work (H2O, SnapKV) make in
// production streaming inference.
//
// IMPORTANT CAVEAT — this is an approximation, not free compression
// ─────────────────────────────────────────────────────────────────
// Sparsification trades a small, generally-imperceptible quality degradation
// for memory savings. It is NOT lossless. The quality impact depends on the
// task: factual recall of mid-sequence details is the failure mode to watch
// for (a fact mentioned only once in the middle of a long prompt, never
// referenced again until the model needs it, may be lost).
//
// This module DEFAULTS TO DISABLED. Callers must explicitly opt in via
// SparsifyFragment() — fragments are never sparsified implicitly during
// normal ExtractFragment() calls. Sparsified fragments are tagged
// (Fragment metadata via the returned SparsificationReport) so the rest of
// the system can distinguish exact fragments from approximated ones.
package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// sha256HexLocal hashes input and returns the hex-encoded digest.
// Mirrors the ContentHash convention used in fragment.go's NewFragment so
// sparsified fragments stay consistent with the rest of the integrity stack.
func sha256HexLocal(input []byte) string {
	h := sha256.Sum256(input)
	return hex.EncodeToString(h[:])
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

const (
	// DefaultSinkTokens is the number of leading "sink" tokens always kept.
	// StreamingLLM's original paper found 4 sink tokens sufficient for most
	// LLaMA-family models; some models benefit from slightly more.
	DefaultSinkTokens = 4

	// DefaultWindowFraction is the fraction of the MOST RECENT tokens kept
	// (the "recency window"), as a fraction of the fragment's total token span.
	// 0.10 means: sink tokens + the most recent 10% of the sequence are kept,
	// everything else is dropped. This targets the "10% carries 90% of the
	// weight" heuristic — it is a starting point, not a proven constant for
	// every model; tune per deployment if attention is measured directly.
	DefaultWindowFraction = 0.10

	// MinPivotTokens is the floor on how few tokens sparsification will keep,
	// regardless of fraction math on very short fragments. Prevents producing
	// a degenerate near-empty fragment.
	MinPivotTokens = 16
)

// SparsificationConfig controls which tokens are kept during sparsification.
type SparsificationConfig struct {
	// SinkTokens is the number of leading tokens always retained, regardless
	// of recency. These anchor the attention distribution (StreamingLLM).
	SinkTokens int

	// WindowFraction is the fraction of total token span retained from the
	// END of the sequence (most recent tokens). Range (0, 1].
	WindowFraction float64

	// AttentionWeights, if provided, overrides the sink+window heuristic with
	// actual measured attention weights per token (e.g. captured during the
	// engine's forward pass). When set, the top-K tokens by weight are kept
	// instead of the positional heuristic. Len must equal the fragment's
	// TokenSpan(). This is the "real" pivot-token selection when available;
	// the positional heuristic is the fallback when attention scores aren't
	// exposed by the engine (most mobile engines don't expose them today).
	AttentionWeights []float32
}

// DefaultSparsificationConfig returns the StreamingLLM-style sink+window heuristic.
func DefaultSparsificationConfig() SparsificationConfig {
	return SparsificationConfig{
		SinkTokens:     DefaultSinkTokens,
		WindowFraction: DefaultWindowFraction,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// SparsificationReport
// ─────────────────────────────────────────────────────────────────────────────

// SparsificationReport describes what a sparsification pass did, so callers
// can decide whether to trust the result for their use case and so the
// fragment can be tagged as approximated rather than exact.
type SparsificationReport struct {
	OriginalTokenCount int
	KeptTokenCount     int
	KeptIndices        []int   // original token positions retained, in order
	CompressionRatio   float64 // KeptTokenCount / OriginalTokenCount
	BytesFreed         int
	UsedAttentionScores bool // true if real attention weights were used vs positional heuristic
}

// ─────────────────────────────────────────────────────────────────────────────
// Pivot token selection
// ─────────────────────────────────────────────────────────────────────────────

// selectPivotIndices returns the sorted, de-duplicated list of token indices
// (0-based, relative to the fragment's TokenStart) to retain.
func selectPivotIndices(tokenSpan int, cfg SparsificationConfig) ([]int, bool) {
	if len(cfg.AttentionWeights) == tokenSpan && tokenSpan > 0 {
		return selectByAttentionWeights(cfg.AttentionWeights, cfg), true
	}
	return selectByPositionalHeuristic(tokenSpan, cfg), false
}

// selectByPositionalHeuristic implements the StreamingLLM sink+window rule:
// keep the first SinkTokens, keep the last WindowFraction×tokenSpan tokens,
// drop everything in between.
func selectByPositionalHeuristic(tokenSpan int, cfg SparsificationConfig) []int {
	sink := cfg.SinkTokens
	if sink > tokenSpan {
		sink = tokenSpan
	}

	windowSize := int(float64(tokenSpan) * cfg.WindowFraction)
	if windowSize < MinPivotTokens-sink {
		windowSize = MinPivotTokens - sink
	}
	if windowSize < 0 {
		windowSize = 0
	}
	windowStart := tokenSpan - windowSize
	if windowStart < sink {
		windowStart = sink // avoid double-counting overlap between sink and window
	}

	kept := make(map[int]bool, sink+windowSize)
	for i := 0; i < sink; i++ {
		kept[i] = true
	}
	for i := windowStart; i < tokenSpan; i++ {
		kept[i] = true
	}

	indices := make([]int, 0, len(kept))
	for idx := range kept {
		indices = append(indices, idx)
	}
	sortInts(indices)
	return indices
}

// selectByAttentionWeights keeps the top-K tokens by measured attention
// weight, where K is derived from the same target ratio as the positional
// heuristic (sink count + window fraction) so the two modes are comparable
// in compression ratio. Sink tokens (position 0..SinkTokens) are always
// force-included regardless of their measured weight, matching StreamingLLM's
// finding that sink tokens matter structurally even when their raw attention
// score doesn't always look dominant for a single layer/head.
func selectByAttentionWeights(weights []float32, cfg SparsificationConfig) []int {
	tokenSpan := len(weights)
	targetCount := cfg.SinkTokens + int(float64(tokenSpan)*cfg.WindowFraction)
	if targetCount < MinPivotTokens {
		targetCount = MinPivotTokens
	}
	if targetCount > tokenSpan {
		targetCount = tokenSpan
	}

	type scored = scoredToken
	scoredList := make([]scored, tokenSpan)
	for i, w := range weights {
		scoredList[i] = scored{idx: i, weight: w}
	}

	// Force-include sink tokens
	kept := make(map[int]bool, targetCount)
	for i := 0; i < cfg.SinkTokens && i < tokenSpan; i++ {
		kept[i] = true
	}

	// Sort remaining by weight descending, fill until targetCount
	sortScoredDesc(scoredList)
	for _, s := range scoredList {
		if len(kept) >= targetCount {
			break
		}
		kept[s.idx] = true
	}

	indices := make([]int, 0, len(kept))
	for idx := range kept {
		indices = append(indices, idx)
	}
	sortInts(indices)
	return indices
}

// ─────────────────────────────────────────────────────────────────────────────
// SparsifyFragment
// ─────────────────────────────────────────────────────────────────────────────

// SparsifyFragment produces a new, smaller fragment containing only the
// pivot tokens selected by cfg. The original fragment is not modified.
//
// IMPORTANT: This only works for the llamacpp flat tensor layout
// ([seq, heads, dim] per layer, uniform across layers). ONNX's
// header-prefixed format requires a different extraction path
// (not implemented here — sparsify before the ONNX reshape, or extend
// this function with an ONNX-aware variant if needed).
//
// The returned fragment has Engine/Model/LayerStart/LayerEnd/LayerStride
// unchanged from the source — only the token dimension is sparsified.
// TokenStart/TokenEnd on the sparsified fragment describe the ORIGINAL range
// covered (for cache lookup purposes); the actual stored token COUNT is
// smaller than TokenEnd-TokenStart. Callers that need exact tensor shape
// information should use the returned SparsificationReport.KeptIndices.
func SparsifyFragment(f *KVFragment, cfg SparsificationConfig) (*KVFragment, *SparsificationReport, error) {
	if f == nil {
		return nil, nil, fmt.Errorf("SparsifyFragment: fragment is nil")
	}

	tokenSpan := f.TokenSpan()
	numLayers := f.NumLayersCovered()
	if numLayers <= 0 {
		return nil, nil, fmt.Errorf("SparsifyFragment: fragment has no layers")
	}
	if len(f.Keys)%numLayers != 0 || len(f.Values)%numLayers != 0 {
		return nil, nil, fmt.Errorf(
			"SparsifyFragment: tensor blob not evenly divisible by layer count "+
				"(non-uniform layout, e.g. ONNX header format — not supported by this function)",
		)
	}

	model := f.Model
	keysPerLayer := len(f.Keys) / numLayers
	valsPerLayer := len(f.Values) / numLayers
	floatsPerToken := model.NumKVHeads * model.HeadDim
	bytesPerToken := floatsPerToken * 4

	if keysPerLayer%bytesPerToken != 0 {
		return nil, nil, fmt.Errorf(
			"SparsifyFragment: layer byte size %d not divisible by per-token size %d "+
				"(model dimension mismatch?)", keysPerLayer, bytesPerToken,
		)
	}

	pivotIndices, usedAttention := selectPivotIndices(tokenSpan, cfg)
	if len(pivotIndices) == 0 {
		return nil, nil, fmt.Errorf("SparsifyFragment: no pivot tokens selected")
	}

	keptCount := len(pivotIndices)
	newKeysPerLayer := keptCount * bytesPerToken
	newKeys := make([]byte, numLayers*newKeysPerLayer)
	newVals := make([]byte, numLayers*newKeysPerLayer)

	for li := 0; li < numLayers; li++ {
		srcLayerOff := li * keysPerLayer
		dstLayerOff := li * newKeysPerLayer

		for outIdx, tokenIdx := range pivotIndices {
			srcOff := srcLayerOff + tokenIdx*bytesPerToken
			dstOff := dstLayerOff + outIdx*bytesPerToken
			copy(newKeys[dstOff:dstOff+bytesPerToken], f.Keys[srcOff:srcOff+bytesPerToken])
			copy(newVals[dstOff:dstOff+bytesPerToken], f.Values[srcOff:srcOff+bytesPerToken])
		}
	}

	// Build sparsified token IDs (subset matching pivotIndices)
	sparseTokenIDs := make([]int32, keptCount)
	for i, idx := range pivotIndices {
		if idx < len(f.TokenIDs) {
			sparseTokenIDs[i] = f.TokenIDs[idx]
		}
	}

	sparseFrag := &KVFragment{
		ID:              f.ID + "-sparse",
		Model:           f.Model,
		TokenStart:      f.TokenStart, // original range preserved for lookup semantics
		TokenEnd:        f.TokenEnd,
		LayerStart:      f.LayerStart,
		LayerEnd:        f.LayerEnd,
		LayerStride:     f.LayerStride,
		Keys:            newKeys,
		Values:          newVals,
		TokenIDs:        sparseTokenIDs, // NOTE: shorter than TokenEnd-TokenStart; see doc comment
		EmbeddingVector: f.EmbeddingVector,
		Engine:          f.Engine,
		EngineVersion:   f.EngineVersion,
		CreatedAt:       f.CreatedAt,
		ExpiresAt:       f.ExpiresAt,
		HitCount:        0, // sparsified copy starts fresh; original's hit history doesn't transfer
		LastUsedAt:      f.LastUsedAt,
	}

	// ContentHash is recomputed over the sparse token set — it deliberately
	// will NOT match the original fragment's hash. A sparsified fragment is
	// a different artifact and must not be confused with the exact one in
	// integrity checks (security/signing.go, security/merkle.go both treat
	// it as its own fragment with its own hash and its own Merkle tree).
	sparseFrag.ContentHash = recomputeContentHash(sparseTokenIDs)

	report := &SparsificationReport{
		OriginalTokenCount:  tokenSpan,
		KeptTokenCount:      keptCount,
		KeptIndices:         pivotIndices,
		CompressionRatio:    float64(keptCount) / float64(tokenSpan),
		BytesFreed:          (len(f.Keys) + len(f.Values)) - (len(newKeys) + len(newVals)),
		UsedAttentionScores: usedAttention,
	}

	return sparseFrag, report, nil
}

// recomputeContentHash mirrors the hashing logic in fragment.go's NewFragment
// (FNV would be cheaper but we match the project's existing SHA-256 convention
// for ContentHash so sparsified fragments remain consistent with the rest of
// the integrity-checking code in security/signing.go).
func recomputeContentHash(tokenIDs []int32) string {
	hashInput := make([]byte, len(tokenIDs)*4)
	for i, tok := range tokenIDs {
		hashInput[i*4] = byte(tok)
		hashInput[i*4+1] = byte(tok >> 8)
		hashInput[i*4+2] = byte(tok >> 16)
		hashInput[i*4+3] = byte(tok >> 24)
	}
	return sha256HexLocal(hashInput)
}

// ─────────────────────────────────────────────────────────────────────────────
// Local helpers (avoid importing sort/crypto twice across the package if
// already used elsewhere with different aliasing — kept self-contained)
// ─────────────────────────────────────────────────────────────────────────────

func sortInts(s []int) {
	// Simple insertion sort — pivot index lists are small (tens to low hundreds
	// of entries for realistic token spans), insertion sort is fast enough and
	// avoids pulling in "sort" just for this.
	for i := 1; i < len(s); i++ {
		key := s[i]
		j := i - 1
		for j >= 0 && s[j] > key {
			s[j+1] = s[j]
			j--
		}
		s[j+1] = key
	}
}

type scoredToken struct {
	idx    int
	weight float32
}

func sortScoredDesc(s []scoredToken) {
	for i := 1; i < len(s); i++ {
		key := s[i]
		j := i - 1
		for j >= 0 && s[j].weight < key.weight {
			s[j+1] = s[j]
			j--
		}
		s[j+1] = key
	}
}
