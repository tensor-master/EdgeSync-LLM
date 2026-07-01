package cache

import (
	"testing"
	"time"
)

func makeTestModel() ModelID {
	return ModelID{
		Architecture:  "qwen",
		Name:          "Qwen2.5-0.5B",
		Quantization:  "Q4_K_M",
		ContextLength: 4096,
		HeadDim:       64,
		NumKVHeads:    8,
		NumLayers:     24,
	}
}

func makeTestTokenIDs(offset, n int) []int32 {
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(offset + i + 100)
	}
	return ids
}

func makeTestTensor(layers, tokens, heads, dim int) []byte {
	size := layers * tokens * heads * dim * 4
	b := make([]byte, size)
	for i := range b {
		b[i] = byte(i % 256)
	}
	return b
}

// ─────────────────────────────────────────────────────────────────────────────
// NewFragment validation
// ─────────────────────────────────────────────────────────────────────────────

func TestNewFragment_ValidInputs(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 128)
	keys := makeTestTensor(12, 128, 8, 64)
	vals := makeTestTensor(12, 128, 8, 64)
	emb := make([]float32, 384)

	f, err := NewFragment("test-id", model, 0, 128, 0, 24, 2,
		keys, vals, tokenIDs, emb, "llamacpp", "b3117", DefaultTTLSession)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if f.ID != "test-id" {
		t.Errorf("ID: want test-id, got %s", f.ID)
	}
	if f.TokenSpan() != 128 {
		t.Errorf("TokenSpan: want 128, got %d", f.TokenSpan())
	}
	if f.ContentHash == "" {
		t.Error("ContentHash should not be empty")
	}
	if f.CreatedAt.IsZero() {
		t.Error("CreatedAt should not be zero")
	}
	if f.ExpiresAt.Before(f.CreatedAt) {
		t.Error("ExpiresAt should be after CreatedAt")
	}
}

func TestNewFragment_TokenSpanTooSmall(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 32) // below FragmentGranularityTokens=64
	keys := makeTestTensor(12, 32, 8, 64)
	vals := makeTestTensor(12, 32, 8, 64)

	_, err := NewFragment("id", model, 0, 32, 0, 24, 2,
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)
	if err == nil {
		t.Error("expected error for token span below minimum granularity")
	}
}

func TestNewFragment_TokenSpanTooLarge(t *testing.T) {
	model := makeTestModel()
	span := FragmentMaxTokenSpan + 100
	tokenIDs := makeTestTokenIDs(0, span)
	keys := makeTestTensor(12, span, 8, 64)
	vals := makeTestTensor(12, span, 8, 64)

	_, err := NewFragment("id", model, 0, span, 0, 24, 2,
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)
	if err == nil {
		t.Error("expected error for token span above maximum")
	}
}

func TestNewFragment_LayerStrideTooSmall(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 128)
	keys := makeTestTensor(12, 128, 8, 64)
	vals := makeTestTensor(12, 128, 8, 64)

	_, err := NewFragment("id", model, 0, 128, 0, 24, 0, // stride=0
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)
	if err == nil {
		t.Error("expected error for layerStride < 1")
	}
}

func TestNewFragment_LayerEndExceedsModel(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 128)
	keys := makeTestTensor(12, 128, 8, 64)
	vals := makeTestTensor(12, 128, 8, 64)

	_, err := NewFragment("id", model, 0, 128, 0, 999, 2, // layerEnd > model.NumLayers
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)
	if err == nil {
		t.Error("expected error for layerEnd exceeding model.NumLayers")
	}
}

func TestNewFragment_EmptyKeys(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 128)

	_, err := NewFragment("id", model, 0, 128, 0, 24, 2,
		[]byte{}, []byte{1, 2, 3}, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)
	if err == nil {
		t.Error("expected error for empty keys")
	}
}

func TestNewFragment_TokenIDsMismatch(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 64) // only 64, but span=128
	keys := makeTestTensor(12, 128, 8, 64)
	vals := makeTestTensor(12, 128, 8, 64)

	_, err := NewFragment("id", model, 0, 128, 0, 24, 2,
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)
	if err == nil {
		t.Error("expected error for tokenIDs length mismatch")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelID
// ─────────────────────────────────────────────────────────────────────────────

func TestModelID_Hash_Stable(t *testing.T) {
	m := makeTestModel()
	h1 := m.Hash()
	h2 := m.Hash()
	if h1 != h2 {
		t.Error("ModelID.Hash() should be deterministic")
	}
	if len(h1) != 8 {
		t.Errorf("hash should be 8 chars, got %d", len(h1))
	}
}

func TestModelID_Hash_DifferentModels(t *testing.T) {
	m1 := makeTestModel()
	m2 := makeTestModel()
	m2.Quantization = "F16"
	if m1.Hash() == m2.Hash() {
		t.Error("different models should have different hashes")
	}
}

func TestModelID_String(t *testing.T) {
	m := makeTestModel()
	s := m.String()
	if s == "" {
		t.Error("String() should not be empty")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Fragment lifecycle
// ─────────────────────────────────────────────────────────────────────────────

func TestFragment_IsExpired_Fresh(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 128)
	keys := makeTestTensor(12, 128, 8, 64)
	vals := makeTestTensor(12, 128, 8, 64)

	f, _ := NewFragment("id", model, 0, 128, 0, 24, 2,
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)

	if f.IsExpired() {
		t.Error("fresh fragment should not be expired")
	}
}

func TestFragment_IsExpired_Past(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 128)
	keys := makeTestTensor(12, 128, 8, 64)
	vals := makeTestTensor(12, 128, 8, 64)

	f, _ := NewFragment("id", model, 0, 128, 0, 24, 2,
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", -1*time.Hour) // already expired

	if !f.IsExpired() {
		t.Error("fragment with past TTL should be expired")
	}
}

func TestFragment_RecordHit_IncrementsCount(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 128)
	keys := makeTestTensor(12, 128, 8, 64)
	vals := makeTestTensor(12, 128, 8, 64)

	f, _ := NewFragment("id", model, 0, 128, 0, 24, 2,
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)

	for i := 0; i < HitThresholdPromote; i++ {
		f.RecordHit()
	}
	if f.HitCount != HitThresholdPromote {
		t.Errorf("want HitCount=%d, got %d", HitThresholdPromote, f.HitCount)
	}
}

func TestFragment_Promote_ExtendsExpiry(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 128)
	keys := makeTestTensor(12, 128, 8, 64)
	vals := makeTestTensor(12, 128, 8, 64)

	f, _ := NewFragment("id", model, 0, 128, 0, 24, 2,
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)

	originalExpiry := f.ExpiresAt
	f.Promote()

	if !f.ExpiresAt.After(originalExpiry) {
		t.Error("Promote should extend ExpiresAt beyond session TTL")
	}
	expectedMin := time.Now().Add(DefaultTTLPersistent - time.Minute)
	if f.ExpiresAt.Before(expectedMin) {
		t.Errorf("ExpiresAt after Promote should be near %v, got %v",
			expectedMin, f.ExpiresAt)
	}
}

func TestFragment_RecordHit_AutoPromotes(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 128)
	keys := makeTestTensor(12, 128, 8, 64)
	vals := makeTestTensor(12, 128, 8, 64)

	f, _ := NewFragment("id", model, 0, 128, 0, 24, 2,
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)

	sessionExpiry := f.ExpiresAt

	// Hit enough times to trigger auto-promotion
	for i := 0; i < HitThresholdPromote; i++ {
		f.RecordHit()
	}

	if !f.ExpiresAt.After(sessionExpiry) {
		t.Error("auto-promotion should extend ExpiresAt")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// NumLayersCovered
// ─────────────────────────────────────────────────────────────────────────────

func TestFragment_NumLayersCovered(t *testing.T) {
	tests := []struct {
		start, end, stride, want int
	}{
		{0, 24, 1, 24},
		{0, 24, 2, 12},
		{0, 24, 4, 6},
		{0, 1, 1, 1},
		{4, 8, 2, 2}, // layers 4, 6
	}

	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 128)

	for _, tt := range tests {
		// Build enough fake tensor data
		numLayers := 0
		for l := tt.start; l < tt.end; l += tt.stride {
			numLayers++
		}
		if numLayers == 0 {
			continue
		}
		keys := makeTestTensor(numLayers, 128, 8, 64)
		vals := makeTestTensor(numLayers, 128, 8, 64)

		f, err := NewFragment("id", model, 0, 128, tt.start, tt.end, tt.stride,
			keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)
		if err != nil {
			t.Errorf("stride=%d: unexpected error: %v", tt.stride, err)
			continue
		}
		got := f.NumLayersCovered()
		if got != tt.want {
			t.Errorf("start=%d end=%d stride=%d: want %d layers, got %d",
				tt.start, tt.end, tt.stride, tt.want, got)
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// StorageKey
// ─────────────────────────────────────────────────────────────────────────────

func TestFragment_StorageKey_Format(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 128)
	keys := makeTestTensor(12, 128, 8, 64)
	vals := makeTestTensor(12, 128, 8, 64)

	f, _ := NewFragment("abc123", model, 0, 128, 0, 24, 2,
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)

	key := f.StorageKey()
	if key == "" {
		t.Error("StorageKey should not be empty")
	}
	// Should contain model hash, engine, and fragment ID
	if len(key) < 10 {
		t.Errorf("StorageKey too short: %q", key)
	}
}

func TestFragment_StorageKey_UniquePerFragment(t *testing.T) {
	model := makeTestModel()
	tokenIDs := makeTestTokenIDs(0, 128)
	keys := makeTestTensor(12, 128, 8, 64)
	vals := makeTestTensor(12, 128, 8, 64)

	f1, _ := NewFragment("id1", model, 0, 128, 0, 24, 2,
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)
	f2, _ := NewFragment("id2", model, 128, 256, 0, 24, 2,
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", DefaultTTLSession)

	if f1.StorageKey() == f2.StorageKey() {
		t.Error("different fragments should have different storage keys")
	}
}
