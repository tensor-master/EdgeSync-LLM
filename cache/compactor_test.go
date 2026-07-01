package cache

import (
	"testing"
	"time"
)

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func makeFragment(id string, tokenStart, tokenEnd, hitCount int) *KVFragment {
	model := makeTestModel()
	span := tokenEnd - tokenStart
	tokenIDs := makeTestTokenIDs(tokenStart, span)
	numLayers := 12
	keys := makeTestTensor(numLayers, span, 8, 64)
	vals := makeTestTensor(numLayers, span, 8, 64)
	emb := make([]float32, 384)
	for i := range emb {
		emb[i] = float32(i) / 384.0
	}

	f, err := NewFragment(id, model, tokenStart, tokenEnd, 0, 24, 2,
		keys, vals, tokenIDs, emb, "llamacpp", "b3117", DefaultTTLSession)
	if err != nil {
		panic("makeFragment: " + err.Error())
	}
	f.HitCount = hitCount
	return f
}

// ─────────────────────────────────────────────────────────────────────────────
// Deduplication tests
// ─────────────────────────────────────────────────────────────────────────────

func TestDeduplicate_NoDuplicates(t *testing.T) {
	frags := []*KVFragment{
		makeFragment("a", 0, 128, 3),
		makeFragment("b", 128, 256, 5),
		makeFragment("c", 256, 384, 1),
	}
	deduped, removed, _ := deduplicate(frags)
	if removed != 0 {
		t.Errorf("no duplicates: want removed=0, got %d", removed)
	}
	if len(deduped) != 3 {
		t.Errorf("no duplicates: want 3 fragments, got %d", len(deduped))
	}
}

func TestDeduplicate_KeepsHighestHitCount(t *testing.T) {
	// Two fragments with same token range → same ContentHash
	f1 := makeFragment("a", 0, 128, 2)
	f2 := makeFragment("b", 0, 128, 7) // higher hit count
	// Make them share the same ContentHash by copying tokenIDs
	f2.TokenIDs = f1.TokenIDs
	f2.ContentHash = f1.ContentHash

	deduped, removed, _ := deduplicate([]*KVFragment{f1, f2})
	if removed != 1 {
		t.Errorf("want 1 removed, got %d", removed)
	}
	if len(deduped) != 1 {
		t.Fatalf("want 1 fragment, got %d", len(deduped))
	}
	if deduped[0].HitCount != 7 {
		t.Errorf("want HitCount=7 (higher), got %d", deduped[0].HitCount)
	}
}

func TestDeduplicate_TieBreaksByExpiry(t *testing.T) {
	f1 := makeFragment("a", 0, 128, 5)
	f2 := makeFragment("b", 0, 128, 5) // same hit count
	f2.TokenIDs = f1.TokenIDs
	f2.ContentHash = f1.ContentHash
	// f2 expires later
	f2.ExpiresAt = time.Now().Add(DefaultTTLPersistent)

	deduped, _, _ := deduplicate([]*KVFragment{f1, f2})
	if len(deduped) != 1 {
		t.Fatalf("want 1 fragment, got %d", len(deduped))
	}
	if !deduped[0].ExpiresAt.After(f1.ExpiresAt) {
		t.Error("tie-break should keep the fragment with the later ExpiresAt")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Group by layer config tests
// ─────────────────────────────────────────────────────────────────────────────

func TestGroupByLayerConfig_SameConfig(t *testing.T) {
	frags := []*KVFragment{
		makeFragment("a", 0, 128, 1),
		makeFragment("b", 128, 256, 1),
		makeFragment("c", 256, 384, 1),
	}
	groups := groupByLayerConfig(frags)
	if len(groups) != 1 {
		t.Errorf("same layer config: want 1 group, got %d", len(groups))
	}
}

func TestGroupByLayerConfig_DifferentStride(t *testing.T) {
	f1 := makeFragment("a", 0, 128, 1)
	f2 := makeFragment("b", 128, 256, 1)
	f2.LayerStride = 4 // different stride

	groups := groupByLayerConfig([]*KVFragment{f1, f2})
	if len(groups) != 2 {
		t.Errorf("different stride: want 2 groups, got %d", len(groups))
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Merge adjacent tests
// ─────────────────────────────────────────────────────────────────────────────

func TestMergeAdjacent_ContiguousFragments(t *testing.T) {
	f1 := makeFragment("a", 0, 128, 3)
	f2 := makeFragment("b", 128, 256, 5)

	merged, mergeCount, _, err := mergeAdjacent([]*KVFragment{f1, f2})
	if err != nil {
		t.Fatalf("mergeAdjacent: %v", err)
	}
	if mergeCount != 1 {
		t.Errorf("want mergeCount=1, got %d", mergeCount)
	}
	if len(merged) != 1 {
		t.Fatalf("want 1 merged fragment, got %d", len(merged))
	}
	if merged[0].TokenStart != 0 {
		t.Errorf("merged TokenStart: want 0, got %d", merged[0].TokenStart)
	}
	if merged[0].TokenEnd != 256 {
		t.Errorf("merged TokenEnd: want 256, got %d", merged[0].TokenEnd)
	}
	if merged[0].HitCount != 8 {
		t.Errorf("merged HitCount: want 8, got %d", merged[0].HitCount)
	}
}

func TestMergeAdjacent_NonAdjacentFragments(t *testing.T) {
	f1 := makeFragment("a", 0, 128, 1)
	// Gap of 512 tokens between f1 and f2 — exceeds FragmentGranularityTokens
	f2 := makeFragment("b", 640, 768, 1)

	merged, mergeCount, _, err := mergeAdjacent([]*KVFragment{f1, f2})
	if err != nil {
		t.Fatalf("mergeAdjacent: %v", err)
	}
	if mergeCount != 0 {
		t.Errorf("non-adjacent: want mergeCount=0, got %d", mergeCount)
	}
	if len(merged) != 0 {
		t.Errorf("non-adjacent: want 0 merged fragments, got %d", len(merged))
	}
}

func TestMergeAdjacent_MergeWouldExceedMaxSpan(t *testing.T) {
	// Two fragments that would produce a merged span > FragmentMaxTokenSpan
	f1 := makeFragment("a", 0, 1024, 1)
	f2 := makeFragment("b", 1024, 2048+100, 1) // merged span = 2148 > 2048

	_, mergeCount, _, err := mergeAdjacent([]*KVFragment{f1, f2})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mergeCount != 0 {
		t.Errorf("merge exceeding MaxSpan should be skipped, got mergeCount=%d", mergeCount)
	}
}

func TestMergeAdjacent_SingleFragment(t *testing.T) {
	f1 := makeFragment("a", 0, 128, 5)
	merged, mergeCount, _, err := mergeAdjacent([]*KVFragment{f1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mergeCount != 0 {
		t.Errorf("single fragment: want mergeCount=0, got %d", mergeCount)
	}
	_ = merged
}

func TestMergeAdjacent_ThreeContiguous(t *testing.T) {
	f1 := makeFragment("a", 0, 128, 2)
	f2 := makeFragment("b", 128, 256, 3)
	f3 := makeFragment("c", 256, 384, 4)

	merged, mergeCount, _, err := mergeAdjacent([]*KVFragment{f1, f2, f3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mergeCount < 1 {
		t.Errorf("three contiguous: want at least 1 merge, got %d", mergeCount)
	}
	_ = merged
}

// ─────────────────────────────────────────────────────────────────────────────
// mergeTwoFragments tests
// ─────────────────────────────────────────────────────────────────────────────

func TestMergeTwoFragments_EngineMismatch(t *testing.T) {
	f1 := makeFragment("a", 0, 128, 1)
	f2 := makeFragment("b", 128, 256, 1)
	f2.Engine = "onnx"

	_, err := mergeTwoFragments(f1, f2)
	if err == nil {
		t.Error("expected error for engine mismatch")
	}
}

func TestMergeTwoFragments_ModelMismatch(t *testing.T) {
	f1 := makeFragment("a", 0, 128, 1)
	f2 := makeFragment("b", 128, 256, 1)
	f2.Model.Quantization = "F16"

	_, err := mergeTwoFragments(f1, f2)
	if err == nil {
		t.Error("expected error for model mismatch")
	}
}

func TestMergeTwoFragments_NonAdjacent(t *testing.T) {
	f1 := makeFragment("a", 0, 128, 1)
	f2 := makeFragment("b", 512, 640, 1) // gap=384, too large

	_, err := mergeTwoFragments(f1, f2)
	if err == nil {
		t.Error("expected error for non-adjacent fragments")
	}
}

func TestMergeTwoFragments_MergedEmbeddingNormalized(t *testing.T) {
	f1 := makeFragment("a", 0, 128, 3)
	f2 := makeFragment("b", 128, 256, 5)

	merged, err := mergeTwoFragments(f1, f2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check L2 norm of merged embedding ≈ 1.0
	var norm float32
	for _, v := range merged.EmbeddingVector {
		norm += v * v
	}
	if norm < 0.99 || norm > 1.01 {
		t.Errorf("merged embedding should be L2-normalized, norm=%.4f", norm)
	}
}

func TestMergeTwoFragments_ContentHashRecomputed(t *testing.T) {
	f1 := makeFragment("a", 0, 128, 1)
	f2 := makeFragment("b", 128, 256, 1)

	merged, err := mergeTwoFragments(f1, f2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if merged.ContentHash == f1.ContentHash {
		t.Error("merged ContentHash should differ from f1")
	}
	if merged.ContentHash == f2.ContentHash {
		t.Error("merged ContentHash should differ from f2")
	}
	if merged.ContentHash == "" {
		t.Error("merged ContentHash should not be empty")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Compactor.Run integration test
// ─────────────────────────────────────────────────────────────────────────────

func TestCompactor_Run_EmptyStore(t *testing.T) {
	store := &FragmentStore{}
	c := NewCompactor(store)
	result, err := c.Run()
	if err != nil {
		t.Fatalf("unexpected error on empty store: %v", err)
	}
	if result.DuplicatesRemoved != 0 || result.FragmentsMerged != 0 {
		t.Error("empty store should produce zero compaction activity")
	}
}
