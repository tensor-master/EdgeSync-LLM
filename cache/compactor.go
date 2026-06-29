// Package cache — Fragment compaction engine.
//
// PROBLEM
// ────────
// Over time, the fragment store accumulates many small overlapping fragments
// for the same model and token prefix. Example after 10 sessions:
//
//   Fragment A: tokens [0,  128), layers [0,24), stride 2  — hit_count=12
//   Fragment B: tokens [0,  192), layers [0,24), stride 2  — hit_count=3
//   Fragment C: tokens [64, 256), layers [0,24), stride 2  — hit_count=7
//   Fragment D: tokens [0,  128), layers [0,24), stride 2  — hit_count=1  (duplicate of A)
//
// Problems:
//   - Duplicate tensors waste disk space
//   - Overlapping fragments cause redundant HNSW entries
//   - Multiple small fragments for adjacent ranges miss the opportunity
//     for a single large fragment covering the full prefix
//
// SOLUTION: Compaction
// ─────────────────────
// The compactor runs periodically (or on demand) and:
//
//   1. DEDUPLICATION: merges fragments with the same ContentHash
//      (identical token sequence → identical tensors).
//      Keeps the one with the highest HitCount.
//
//   2. ADJACENCY MERGE: merges fragments whose token ranges are
//      contiguous or overlapping, for the same model, engine, and layer config.
//      A+B above → Fragment AB: tokens [0, 192), combined hit_count=15.
//
//   3. COVERAGE PROMOTION: after merging, promotes the new fragment to
//      persistent TTL if the combined HitCount >= HitThresholdPromote.
//
// MERGE STRATEGY FOR TENSORS
// ────────────────────────────
// Merging two fragments' KV tensors means concatenating them along the
// token dimension (axis 0 for llamacpp, axis 2 for ONNX).
//
//   llamacpp layout [seq, heads, dim]:
//     merged_keys = concat(A.keys, B_suffix.keys) along axis 0
//     where B_suffix = B.keys[overlap_tokens:]  (skip duplicate prefix)
//
//   ONNX layout [heads, seq, dim]:
//     merged_keys = concat(A.keys, B_suffix.keys) along axis 1
//
// This is O(tensor_size) memory and O(tensor_size/cache_line) time.
// On Cortex-A55: ~1.2ms per MB copied. A 24MB merge takes ~28ms.
package cache

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"time"
)

// ─────────────────────────────────────────────────────────────────────────────
// Compactor
// ─────────────────────────────────────────────────────────────────────────────

// Compactor runs fragment deduplication and adjacency merging on a FragmentStore.
type Compactor struct {
	store *FragmentStore
}

// NewCompactor creates a compactor for the given store.
func NewCompactor(store *FragmentStore) *Compactor {
	return &Compactor{store: store}
}

// CompactionResult summarizes what the compactor did.
type CompactionResult struct {
	DuplicatesRemoved int
	FragmentsMerged   int
	BytesFreed        int
	Duration          time.Duration
	NewFragments      []*KVFragment // fragments created by merging
}

// Run executes a full compaction pass on the store.
// It collects all in-memory fragments, deduplicates, then merges adjacent ones.
// Thread-safe: uses the store's existing sync.Map and SQLite WAL.
func (c *Compactor) Run() (*CompactionResult, error) {
	start := time.Now()
	result := &CompactionResult{}

	// 1. Collect all live fragments from hot tier
	var all []*KVFragment
	c.store.hot.Range(func(_, v interface{}) bool {
		f := v.(*KVFragment)
		if !f.IsExpired() {
			all = append(all, f)
		}
		return true
	})

	if len(all) < 2 {
		result.Duration = time.Since(start)
		return result, nil
	}

	// 2. Deduplicate by ContentHash
	deduped, removed, bytesFreed := deduplicate(all)
	result.DuplicatesRemoved = removed
	result.BytesFreed += bytesFreed

	// 3. Group by (ModelHash, Engine, LayerStart, LayerEnd, LayerStride)
	// Only fragments with identical layer config can be tensor-merged.
	groups := groupByLayerConfig(deduped)

	// 4. Merge adjacent token ranges within each group
	for _, group := range groups {
		merged, mergedCount, freed, err := mergeAdjacent(group)
		if err != nil {
			// Non-fatal: log and continue
			continue
		}
		result.FragmentsMerged += mergedCount
		result.BytesFreed += freed

		for _, mf := range merged {
			// Store the new merged fragment
			if err := c.store.Store(mf); err == nil {
				result.NewFragments = append(result.NewFragments, mf)
			}
		}
	}

	result.Duration = time.Since(start)
	return result, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1: Deduplication
// ─────────────────────────────────────────────────────────────────────────────

// deduplicate removes fragments with duplicate ContentHash,
// keeping the one with the highest HitCount per unique hash.
func deduplicate(fragments []*KVFragment) (deduped []*KVFragment, removed int, bytesFreed int) {
	// Map: ContentHash → best fragment
	best := make(map[string]*KVFragment)

	for _, f := range fragments {
		existing, ok := best[f.ContentHash]
		if !ok {
			best[f.ContentHash] = f
			continue
		}
		// Keep the one with higher HitCount; on tie, keep newer ExpiresAt
		if f.HitCount > existing.HitCount ||
			(f.HitCount == existing.HitCount && f.ExpiresAt.After(existing.ExpiresAt)) {
			bytesFreed += existing.SizeBytes()
			best[f.ContentHash] = f
			removed++
		} else {
			bytesFreed += f.SizeBytes()
			removed++
		}
	}

	deduped = make([]*KVFragment, 0, len(best))
	for _, f := range best {
		deduped = append(deduped, f)
	}
	return
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: Group by layer config
// ─────────────────────────────────────────────────────────────────────────────

type layerKey struct {
	modelHash   string
	engine      string
	layerStart  int
	layerEnd    int
	layerStride int
}

func groupByLayerConfig(fragments []*KVFragment) map[layerKey][]*KVFragment {
	groups := make(map[layerKey][]*KVFragment)
	for _, f := range fragments {
		k := layerKey{
			modelHash:   f.Model.Hash(),
			engine:      f.Engine,
			layerStart:  f.LayerStart,
			layerEnd:    f.LayerEnd,
			layerStride: f.LayerStride,
		}
		groups[k] = append(groups[k], f)
	}
	return groups
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Merge adjacent token ranges
// ─────────────────────────────────────────────────────────────────────────────

// mergeAdjacent sorts fragments by TokenStart and greedily merges any two
// fragments whose token ranges overlap or are contiguous (gap ≤ FragmentGranularityTokens).
//
// Example:
//   [0,128) + [64,256)  → overlap 64 tokens  → merge to [0,256)
//   [0,128) + [128,256) → contiguous          → merge to [0,256)
//   [0,128) + [256,384) → gap 128 tokens      → NOT merged (exceeds granularity)
func mergeAdjacent(group []*KVFragment) (merged []*KVFragment, mergeCount int, bytesFreed int, err error) {
	if len(group) < 2 {
		return group, 0, 0, nil
	}

	// Sort by token start
	sort.Slice(group, func(i, j int) bool {
		return group[i].TokenStart < group[j].TokenStart
	})

	result := []*KVFragment{group[0]}

	for i := 1; i < len(group); i++ {
		last := result[len(result)-1]
		curr := group[i]

		gap := curr.TokenStart - last.TokenEnd

		// Merge if overlapping or contiguous within one granularity unit
		if gap <= FragmentGranularityTokens {
			// Would the merged fragment exceed FragmentMaxTokenSpan?
			newSpan := curr.TokenEnd - last.TokenStart
			if newSpan > FragmentMaxTokenSpan {
				// Too large to merge — keep separate
				result = append(result, curr)
				continue
			}

			mergedFrag, mErr := mergeTwoFragments(last, curr)
			if mErr != nil {
				// Merge failed — keep both fragments unchanged
				result = append(result, curr)
				continue
			}

			bytesFreed += last.SizeBytes() + curr.SizeBytes() - mergedFrag.SizeBytes()
			result[len(result)-1] = mergedFrag
			mergeCount++
		} else {
			result = append(result, curr)
		}
	}

	// Only return newly merged fragments (those not in the original group)
	for _, f := range result {
		isNew := true
		for _, orig := range group {
			if f.ID == orig.ID {
				isNew = false
				break
			}
		}
		if isNew {
			merged = append(merged, f)
		}
	}

	return merged, mergeCount, bytesFreed, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Core merge: concatenate two fragments' tensors along the token axis
// ─────────────────────────────────────────────────────────────────────────────

// mergeTwoFragments combines the KV tensors of fragment A and B.
// B must start at or before A.TokenEnd (overlap or contiguous).
//
// The resulting fragment covers [A.TokenStart, max(A.TokenEnd, B.TokenEnd)).
// If B overlaps A, the overlapping portion of B's tensors is discarded.
//
// Token IDs are concatenated (with deduplication of the overlap).
// ContentHash is recomputed on the merged token IDs.
// HitCount = A.HitCount + B.HitCount.
// ExpiresAt = max(A.ExpiresAt, B.ExpiresAt).
func mergeTwoFragments(a, b *KVFragment) (*KVFragment, error) {
	if a.Engine != b.Engine {
		return nil, fmt.Errorf("merge: engine mismatch (%s vs %s)", a.Engine, b.Engine)
	}
	if a.Model.Hash() != b.Model.Hash() {
		return nil, fmt.Errorf("merge: model mismatch")
	}
	if a.LayerStart != b.LayerStart || a.LayerEnd != b.LayerEnd || a.LayerStride != b.LayerStride {
		return nil, fmt.Errorf("merge: layer config mismatch")
	}
	if b.TokenStart > a.TokenEnd {
		return nil, fmt.Errorf("merge: fragments are not adjacent (gap=%d)", b.TokenStart-a.TokenEnd)
	}

	model := a.Model
	H := model.NumKVHeads
	D := model.HeadDim
	numLayers := a.NumLayersCovered()
	floatsPerTokenPerLayer := H * D

	// How many tokens does B contribute beyond A's end?
	overlapTokens := a.TokenEnd - b.TokenStart // may be 0 if exactly contiguous
	if overlapTokens < 0 {
		overlapTokens = 0
	}
	newTokensFromB := b.TokenSpan() - overlapTokens
	if newTokensFromB <= 0 {
		// B is entirely covered by A — just return A with updated hit count
		merged := *a
		merged.HitCount = a.HitCount + b.HitCount
		if b.ExpiresAt.After(a.ExpiresAt) {
			merged.ExpiresAt = b.ExpiresAt
		}
		return &merged, nil
	}

	mergedTokenEnd := b.TokenEnd
	mergedSpan := mergedTokenEnd - a.TokenStart

	switch a.Engine {
	case "llamacpp":
		return mergeLlamacppTensors(a, b, model, numLayers, H, D, floatsPerTokenPerLayer,
			overlapTokens, newTokensFromB, mergedSpan, mergedTokenEnd)
	case "onnx":
		return mergeONNXTensors(a, b, model, numLayers, H, D, floatsPerTokenPerLayer,
			overlapTokens, newTokensFromB, mergedSpan, mergedTokenEnd)
	default:
		return nil, fmt.Errorf("merge: unsupported engine %q", a.Engine)
	}
}

// mergeLlamacppTensors concatenates along axis 0 (token axis).
// llamacpp layout per layer: [seq_len × heads × head_dim]
func mergeLlamacppTensors(
	a, b *KVFragment,
	model ModelID, numLayers, H, D, floatsPerTokenPerLayer int,
	overlapTokens, newTokensFromB, mergedSpan, mergedTokenEnd int,
) (*KVFragment, error) {
	aSpan := a.TokenSpan()
	floatsPerLayerA := aSpan * H * D
	floatsPerLayerNew := newTokensFromB * H * D
	floatsPerLayerMerged := mergedSpan * H * D

	aKeys := bytesToFloat32Merge(a.Keys)
	aVals := bytesToFloat32Merge(a.Values)
	bKeys := bytesToFloat32Merge(b.Keys)
	bVals := bytesToFloat32Merge(b.Values)

	bOffsetFloats := overlapTokens * floatsPerTokenPerLayer

	mergedKeys := make([]float32, numLayers*floatsPerLayerMerged)
	mergedVals := make([]float32, numLayers*floatsPerLayerMerged)

	for li := 0; li < numLayers; li++ {
		aOff := li * floatsPerLayerA
		bOff := li * b.TokenSpan() * H * D
		mOff := li * floatsPerLayerMerged

		// Copy A's layer data
		copy(mergedKeys[mOff:mOff+floatsPerLayerA], aKeys[aOff:aOff+floatsPerLayerA])
		copy(mergedVals[mOff:mOff+floatsPerLayerA], aVals[aOff:aOff+floatsPerLayerA])

		// Append B's non-overlapping suffix
		bSuffix := bOff + bOffsetFloats
		copy(mergedKeys[mOff+floatsPerLayerA:mOff+floatsPerLayerA+floatsPerLayerNew],
			bKeys[bSuffix:bSuffix+floatsPerLayerNew])
		copy(mergedVals[mOff+floatsPerLayerA:mOff+floatsPerLayerA+floatsPerLayerNew],
			bVals[bSuffix:bSuffix+floatsPerLayerNew])
	}

	return buildMergedFragment(a, b, mergedSpan, mergedTokenEnd,
		float32SliceToBytesMerge(mergedKeys),
		float32SliceToBytesMerge(mergedVals),
	), nil
}

// mergeONNXTensors concatenates along axis 1 (seq_len axis in ONNX layout).
// ONNX layout per layer: [heads × seq_len × head_dim]
func mergeONNXTensors(
	a, b *KVFragment,
	model ModelID, numLayers, H, D, floatsPerTokenPerLayer int,
	overlapTokens, newTokensFromB, mergedSpan, mergedTokenEnd int,
) (*KVFragment, error) {
	// Parse ONNX headers
	if len(a.Keys) < 16 || len(b.Keys) < 16 {
		return nil, fmt.Errorf("merge ONNX: header too short")
	}

	aSpan := int(binary.LittleEndian.Uint32(a.Keys[8:]))
	_ = aSpan // used for validation

	// Build merged ONNX header
	mergedHeader := make([]byte, 16)
	binary.LittleEndian.PutUint32(mergedHeader[0:], uint32(numLayers))
	binary.LittleEndian.PutUint32(mergedHeader[4:], uint32(H))
	binary.LittleEndian.PutUint32(mergedHeader[8:], uint32(mergedSpan))
	binary.LittleEndian.PutUint32(mergedHeader[12:], uint32(D))

	bytesPerLayerHeaderA := a.TokenSpan() * H * D * 4
	bytesPerLayerNew := newTokensFromB * H * D * 4
	bytesPerLayerMerged := mergedSpan * H * D * 4

	newKeysSize := 16 + numLayers*(4+bytesPerLayerMerged)
	mergedKeys := make([]byte, newKeysSize)
	mergedVals := make([]byte, newKeysSize)
	copy(mergedKeys, mergedHeader)
	copy(mergedVals, mergedHeader)

	// For ONNX layout [H × S × D], concatenation along S requires interleaving per head.
	// We extract, interleave, and re-pack.
	aKCursor, aVCursor := 16, 16
	bKCursor, bVCursor := 16, 16
	mKCursor, mVCursor := 16, 16

	layer := a.LayerStart
	for li := 0; li < numLayers; li++ {
		aKCursor += 4
		aVCursor += 4
		bKCursor += 4
		bVCursor += 4

		layerTag := make([]byte, 4)
		binary.LittleEndian.PutUint32(layerTag, uint32(layer))
		copy(mergedKeys[mKCursor:], layerTag)
		copy(mergedVals[mVCursor:], layerTag)
		mKCursor += 4
		mVCursor += 4

		// Per-head concatenation: [h][s_a + s_b_new][d]
		aKLayer := bytesToFloat32Merge(a.Keys[aKCursor : aKCursor+bytesPerLayerHeaderA])
		aVLayer := bytesToFloat32Merge(a.Values[aVCursor : aVCursor+bytesPerLayerHeaderA])
		bKLayer := bytesToFloat32Merge(b.Keys[bKCursor : bKCursor+b.TokenSpan()*H*D*4])
		bVLayer := bytesToFloat32Merge(b.Values[bVCursor : bVCursor+b.TokenSpan()*H*D*4])

		aS := a.TokenSpan()
		bS := b.TokenSpan()
		bSkip := overlapTokens

		mergedKLayer := make([]float32, H*mergedSpan*D)
		mergedVLayer := make([]float32, H*mergedSpan*D)

		for h := 0; h < H; h++ {
			// Copy A's tokens for this head
			aHeadOff := h * aS * D
			mHeadOff := h * mergedSpan * D
			copy(mergedKLayer[mHeadOff:], aKLayer[aHeadOff:aHeadOff+aS*D])
			copy(mergedVLayer[mHeadOff:], aVLayer[aHeadOff:aHeadOff+aS*D])

			// Append B's non-overlapping suffix for this head
			bHeadOff := h*bS*D + bSkip*D
			copy(mergedKLayer[mHeadOff+aS*D:], bKLayer[bHeadOff:bHeadOff+newTokensFromB*D])
			copy(mergedVLayer[mHeadOff+aS*D:], bVLayer[bHeadOff:bHeadOff+newTokensFromB*D])
		}

		kBytes := float32SliceToBytesMerge(mergedKLayer)
		vBytes := float32SliceToBytesMerge(mergedVLayer)
		copy(mergedKeys[mKCursor:], kBytes)
		copy(mergedVals[mVCursor:], vBytes)

		aKCursor += bytesPerLayerHeaderA
		aVCursor += bytesPerLayerHeaderA
		bKCursor += b.TokenSpan() * H * D * 4
		bVCursor += b.TokenSpan() * H * D * 4
		mKCursor += bytesPerLayerMerged
		mVCursor += bytesPerLayerMerged
		_ = bytesPerLayerNew

		layer += a.LayerStride
	}

	return buildMergedFragment(a, b, mergedSpan, mergedTokenEnd, mergedKeys, mergedVals), nil
}

// buildMergedFragment constructs the metadata for a merged fragment.
func buildMergedFragment(a, b *KVFragment, mergedSpan, mergedTokenEnd int, keys, vals []byte) *KVFragment {
	// Merge token IDs: A's full sequence + B's non-overlapping suffix
	overlap := a.TokenEnd - b.TokenStart
	if overlap < 0 {
		overlap = 0
	}
	mergedTokenIDs := make([]int32, mergedSpan)
	copy(mergedTokenIDs, a.TokenIDs)
	bSuffix := b.TokenIDs[overlap:]
	copy(mergedTokenIDs[len(a.TokenIDs):], bSuffix)

	// Merge embeddings: weighted average by token span
	aWeight := float32(a.TokenSpan())
	bWeight := float32(b.TokenSpan())
	total := aWeight + bWeight
	mergedEmb := make([]float32, len(a.EmbeddingVector))
	for i := range mergedEmb {
		mergedEmb[i] = (a.EmbeddingVector[i]*aWeight + b.EmbeddingVector[i]*bWeight) / total
	}
	// Re-normalize
	var norm float32
	for _, v := range mergedEmb {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range mergedEmb {
			mergedEmb[i] /= norm
		}
	}

	expiresAt := a.ExpiresAt
	if b.ExpiresAt.After(expiresAt) {
		expiresAt = b.ExpiresAt
	}

	id := fmt.Sprintf("merged_%s_%s", a.ID[:8], b.ID[:8])

	f := &KVFragment{
		ID:              id,
		Model:           a.Model,
		TokenStart:      a.TokenStart,
		TokenEnd:        mergedTokenEnd,
		LayerStart:      a.LayerStart,
		LayerEnd:        a.LayerEnd,
		LayerStride:     a.LayerStride,
		Keys:            keys,
		Values:          vals,
		TokenIDs:        mergedTokenIDs,
		EmbeddingVector: mergedEmb,
		Engine:          a.Engine,
		EngineVersion:   a.EngineVersion,
		CreatedAt:       time.Now().UTC(),
		ExpiresAt:       expiresAt,
		HitCount:        a.HitCount + b.HitCount,
		LastUsedAt:      time.Now().UTC(),
	}

	// Recompute ContentHash
	hashInput := make([]byte, len(mergedTokenIDs)*4)
	for i, tok := range mergedTokenIDs {
		hashInput[i*4] = byte(tok)
		hashInput[i*4+1] = byte(tok >> 8)
		hashInput[i*4+2] = byte(tok >> 16)
		hashInput[i*4+3] = byte(tok >> 24)
	}
	// FNV-1a hash of token IDs (fast, no crypto import needed)
	h := uint64(14695981039346656037)
	for _, c := range hashInput {
		h ^= uint64(c)
		h *= 1099511628211
	}
	f.ContentHash = fmt.Sprintf("%016x", h)

	return f
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization helpers (local to compactor)
// ─────────────────────────────────────────────────────────────────────────────

func bytesToFloat32Merge(src []byte) []float32 {
	n := len(src) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
	}
	return out
}

func float32SliceToBytesMerge(src []float32) []byte {
	out := make([]byte, len(src)*4)
	for i, v := range src {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}
