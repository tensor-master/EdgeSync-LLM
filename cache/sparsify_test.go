package cache

import (
	"testing"
)

func makeSparsifyTestFragment(t *testing.T, tokens int) *KVFragment {
	t.Helper()
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(tokens)
	numLayers := 12
	keys := make([]byte, numLayers*tokens*model.NumKVHeads*model.HeadDim*4)
	vals := make([]byte, numLayers*tokens*model.NumKVHeads*model.HeadDim*4)
	for i := range keys {
		keys[i] = byte((i * 31) % 256)
	}
	for i := range vals {
		vals[i] = byte((i*17 + 3) % 256)
	}
	emb := make([]float32, 384)

	f, err := NewFragment("sparsify-test", model, 0, tokens, 0, 24, 2,
		keys, vals, tokenIDs, emb, "llamacpp", "b3117", DefaultTTLSession)
	if err != nil {
		t.Fatalf("setup: %v", err)
	}
	return f
}

// ─────────────────────────────────────────────────────────────────────────────
// selectByPositionalHeuristic tests
// ─────────────────────────────────────────────────────────────────────────────

func TestSelectByPositionalHeuristic_KeepsSinkTokens(t *testing.T) {
	cfg := SparsificationConfig{SinkTokens: 4, WindowFraction: 0.10}
	indices := selectByPositionalHeuristic(256, cfg)

	for i := 0; i < 4; i++ {
		found := false
		for _, idx := range indices {
			if idx == i {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("sink token %d should always be kept", i)
		}
	}
}

func TestSelectByPositionalHeuristic_KeepsRecentWindow(t *testing.T) {
	cfg := SparsificationConfig{SinkTokens: 4, WindowFraction: 0.10}
	tokenSpan := 256
	indices := selectByPositionalHeuristic(tokenSpan, cfg)

	// Last token should always be kept (part of the recency window)
	lastKept := false
	for _, idx := range indices {
		if idx == tokenSpan-1 {
			lastKept = true
		}
	}
	if !lastKept {
		t.Error("the most recent token should be kept by the recency window")
	}
}

func TestSelectByPositionalHeuristic_DropsMiddleTokens(t *testing.T) {
	cfg := SparsificationConfig{SinkTokens: 4, WindowFraction: 0.10}
	tokenSpan := 256
	indices := selectByPositionalHeuristic(tokenSpan, cfg)

	kept := make(map[int]bool)
	for _, idx := range indices {
		kept[idx] = true
	}

	// A token in the middle (well past sink, well before recency window) should be dropped
	middleIdx := tokenSpan / 2
	if kept[middleIdx] {
		t.Errorf("middle token %d should be dropped by sparsification, but was kept", middleIdx)
	}
}

func TestSelectByPositionalHeuristic_RespectsMinPivotTokens(t *testing.T) {
	cfg := SparsificationConfig{SinkTokens: 2, WindowFraction: 0.01} // tiny window
	tokenSpan := 64
	indices := selectByPositionalHeuristic(tokenSpan, cfg)

	if len(indices) < MinPivotTokens {
		// Note: when sink+window < MinPivotTokens, the heuristic widens the
		// window to satisfy the floor — this should never be violated.
		if tokenSpan >= MinPivotTokens {
			t.Errorf("expected at least %d pivot tokens, got %d", MinPivotTokens, len(indices))
		}
	}
}

func TestSelectByPositionalHeuristic_SmallFragment(t *testing.T) {
	cfg := DefaultSparsificationConfig()
	indices := selectByPositionalHeuristic(64, cfg) // exactly MinPivotTokens-ish span

	if len(indices) == 0 {
		t.Error("expected at least some pivot tokens for a 64-token fragment")
	}
	if len(indices) > 64 {
		t.Error("pivot indices should never exceed the token span")
	}
}

func TestSelectByPositionalHeuristic_NoDuplicates(t *testing.T) {
	cfg := DefaultSparsificationConfig()
	indices := selectByPositionalHeuristic(256, cfg)

	seen := make(map[int]bool)
	for _, idx := range indices {
		if seen[idx] {
			t.Errorf("duplicate index %d in pivot selection", idx)
		}
		seen[idx] = true
	}
}

func TestSelectByPositionalHeuristic_SortedAscending(t *testing.T) {
	cfg := DefaultSparsificationConfig()
	indices := selectByPositionalHeuristic(256, cfg)

	for i := 1; i < len(indices); i++ {
		if indices[i] <= indices[i-1] {
			t.Errorf("indices not sorted ascending at position %d: %d <= %d", i, indices[i], indices[i-1])
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// selectByAttentionWeights tests
// ─────────────────────────────────────────────────────────────────────────────

func TestSelectByAttentionWeights_KeepsHighestWeights(t *testing.T) {
	tokenSpan := 100
	weights := make([]float32, tokenSpan)
	// Put high weight on a specific subset, low elsewhere
	highWeightIndices := map[int]bool{50: true, 60: true, 70: true, 80: true}
	for i := range weights {
		if highWeightIndices[i] {
			weights[i] = 10.0
		} else {
			weights[i] = 0.01
		}
	}

	cfg := SparsificationConfig{SinkTokens: 0, WindowFraction: 0.10, AttentionWeights: weights}
	indices := selectByAttentionWeights(weights, cfg)

	kept := make(map[int]bool)
	for _, idx := range indices {
		kept[idx] = true
	}
	for idx := range highWeightIndices {
		if !kept[idx] {
			t.Errorf("high-weight token %d should be kept", idx)
		}
	}
}

func TestSelectByAttentionWeights_ForceIncludesSinkTokens(t *testing.T) {
	tokenSpan := 100
	weights := make([]float32, tokenSpan)
	// Sink tokens have artificially LOW weight — should still be force-included
	for i := range weights {
		weights[i] = 5.0
	}
	weights[0] = 0.0001
	weights[1] = 0.0001

	cfg := SparsificationConfig{SinkTokens: 2, WindowFraction: 0.10, AttentionWeights: weights}
	indices := selectByAttentionWeights(weights, cfg)

	kept := make(map[int]bool)
	for _, idx := range indices {
		kept[idx] = true
	}
	if !kept[0] || !kept[1] {
		t.Error("sink tokens should be force-included regardless of measured weight")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// SparsifyFragment integration tests
// ─────────────────────────────────────────────────────────────────────────────

func TestSparsifyFragment_ReducesSize(t *testing.T) {
	f := makeSparsifyTestFragment(t, 256)
	cfg := DefaultSparsificationConfig()

	sparse, report, err := SparsifyFragment(f, cfg)
	if err != nil {
		t.Fatalf("SparsifyFragment: %v", err)
	}

	if sparse.SizeBytes() >= f.SizeBytes() {
		t.Errorf("sparsified fragment should be smaller: original=%d, sparse=%d",
			f.SizeBytes(), sparse.SizeBytes())
	}
	if report.BytesFreed <= 0 {
		t.Error("report should show positive bytes freed")
	}
	if report.CompressionRatio >= 1.0 {
		t.Errorf("compression ratio should be < 1.0, got %.3f", report.CompressionRatio)
	}
}

func TestSparsifyFragment_PreservesLayerCount(t *testing.T) {
	f := makeSparsifyTestFragment(t, 256)
	sparse, _, err := SparsifyFragment(f, DefaultSparsificationConfig())
	if err != nil {
		t.Fatalf("SparsifyFragment: %v", err)
	}

	if sparse.NumLayersCovered() != f.NumLayersCovered() {
		t.Errorf("layer count should be unchanged: original=%d, sparse=%d",
			f.NumLayersCovered(), sparse.NumLayersCovered())
	}
}

func TestSparsifyFragment_DifferentContentHash(t *testing.T) {
	f := makeSparsifyTestFragment(t, 256)
	sparse, _, err := SparsifyFragment(f, DefaultSparsificationConfig())
	if err != nil {
		t.Fatalf("SparsifyFragment: %v", err)
	}

	if sparse.ContentHash == f.ContentHash {
		t.Error("sparsified fragment must have a different ContentHash from the original " +
			"(it represents a different, approximated artifact)")
	}
}

func TestSparsifyFragment_NilFragment(t *testing.T) {
	_, _, err := SparsifyFragment(nil, DefaultSparsificationConfig())
	if err == nil {
		t.Error("expected error for nil fragment")
	}
}

func TestSparsifyFragment_ReportAccuracy(t *testing.T) {
	f := makeSparsifyTestFragment(t, 256)
	sparse, report, err := SparsifyFragment(f, DefaultSparsificationConfig())
	if err != nil {
		t.Fatalf("SparsifyFragment: %v", err)
	}

	if report.OriginalTokenCount != 256 {
		t.Errorf("OriginalTokenCount: want 256, got %d", report.OriginalTokenCount)
	}
	if report.KeptTokenCount != len(report.KeptIndices) {
		t.Errorf("KeptTokenCount (%d) should match len(KeptIndices) (%d)",
			report.KeptTokenCount, len(report.KeptIndices))
	}
	if len(sparse.TokenIDs) != report.KeptTokenCount {
		t.Errorf("sparse.TokenIDs length (%d) should match KeptTokenCount (%d)",
			len(sparse.TokenIDs), report.KeptTokenCount)
	}
}

func TestSparsifyFragment_NonUniformLayoutRejected(t *testing.T) {
	f := makeSparsifyTestFragment(t, 256)
	f.Keys = f.Keys[:len(f.Keys)-3] // break uniform layer division

	_, _, err := SparsifyFragment(f, DefaultSparsificationConfig())
	if err == nil {
		t.Error("expected error for non-uniform tensor layout")
	}
}

func TestSparsifyFragment_WithAttentionWeights(t *testing.T) {
	f := makeSparsifyTestFragment(t, 256)
	weights := make([]float32, 256)
	for i := range weights {
		weights[i] = float32(i % 10) // arbitrary varying weights
	}
	cfg := SparsificationConfig{SinkTokens: 4, WindowFraction: 0.10, AttentionWeights: weights}

	sparse, report, err := SparsifyFragment(f, cfg)
	if err != nil {
		t.Fatalf("SparsifyFragment with attention weights: %v", err)
	}
	if !report.UsedAttentionScores {
		t.Error("report should indicate attention scores were used")
	}
	if sparse.SizeBytes() >= f.SizeBytes() {
		t.Error("sparsified fragment with attention weights should still be smaller")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper function tests
// ─────────────────────────────────────────────────────────────────────────────

func TestSortInts(t *testing.T) {
	s := []int{5, 3, 8, 1, 9, 2}
	sortInts(s)
	for i := 1; i < len(s); i++ {
		if s[i] < s[i-1] {
			t.Errorf("sortInts failed: %v not sorted at index %d", s, i)
		}
	}
}

func TestSortScoredDesc(t *testing.T) {
	s := []scoredToken{
		{idx: 0, weight: 1.0},
		{idx: 1, weight: 5.0},
		{idx: 2, weight: 3.0},
	}
	sortScoredDesc(s)
	for i := 1; i < len(s); i++ {
		if s[i].weight > s[i-1].weight {
			t.Errorf("sortScoredDesc failed: not descending at index %d", i)
		}
	}
}
