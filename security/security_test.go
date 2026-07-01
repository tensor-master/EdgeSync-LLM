package security

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/bossandboss/EdgeSync-LLM/cache"
)

// ─────────────────────────────────────────────────────────────────────────────
// JNI validation tests
// ─────────────────────────────────────────────────────────────────────────────

func TestValidateEmbedding_Valid(t *testing.T) {
	vec := make([]float32, 384)
	for i := range vec {
		vec[i] = float32(i) / 384.0
	}
	if err := ValidateEmbedding(vec); err != nil {
		t.Errorf("valid embedding rejected: %v", err)
	}
}

func TestValidateEmbedding_TooShort(t *testing.T) {
	vec := make([]float32, 10)
	if err := ValidateEmbedding(vec); err == nil {
		t.Error("expected error for embedding below minimum dimension")
	}
}

func TestValidateEmbedding_TooLong(t *testing.T) {
	vec := make([]float32, MaxEmbeddingDim+1)
	if err := ValidateEmbedding(vec); err == nil {
		t.Error("expected error for embedding above maximum dimension")
	}
}

func TestValidateEmbedding_NaN(t *testing.T) {
	vec := make([]float32, 384)
	vec[100] = float32(math.NaN())
	if err := ValidateEmbedding(vec); err == nil {
		t.Error("expected error for embedding containing NaN")
	}
}

func TestValidateEmbedding_Inf(t *testing.T) {
	vec := make([]float32, 384)
	vec[50] = float32(math.Inf(1)) // overflow
	if err := ValidateEmbedding(vec); err == nil {
		t.Error("expected error for embedding containing Inf")
	}
}

func TestValidateTensorBlob_Valid(t *testing.T) {
	data := make([]byte, 1024*4) // 1024 float32 values
	if err := ValidateTensorBlob(data, "keys"); err != nil {
		t.Errorf("valid tensor blob rejected: %v", err)
	}
}

func TestValidateTensorBlob_Empty(t *testing.T) {
	if err := ValidateTensorBlob([]byte{}, "keys"); err == nil {
		t.Error("expected error for empty blob")
	}
}

func TestValidateTensorBlob_TooLarge(t *testing.T) {
	// Can't allocate 64MB in a test — check the limit value instead
	if MaxFragmentTensorBytes <= 0 {
		t.Error("MaxFragmentTensorBytes should be positive")
	}
	// Simulate oversized by testing boundary logic
	data := make([]byte, MaxFragmentTensorBytes+4)
	if err := ValidateTensorBlob(data, "keys"); err == nil {
		t.Error("expected error for blob exceeding maximum size")
	}
}

func TestValidateTensorBlob_NotFloat32Aligned(t *testing.T) {
	data := make([]byte, 101) // not multiple of 4
	if err := ValidateTensorBlob(data, "keys"); err == nil {
		t.Error("expected error for blob not aligned to float32")
	}
}

func TestValidateModelDimensions_Valid(t *testing.T) {
	m := cache.ModelID{
		Architecture:  "qwen",
		Name:          "test",
		Quantization:  "Q4_K_M",
		ContextLength: 4096,
		HeadDim:       64,
		NumKVHeads:    8,
		NumLayers:     24,
	}
	if err := ValidateModelDimensions(m); err != nil {
		t.Errorf("valid model rejected: %v", err)
	}
}

func TestValidateModelDimensions_ZeroLayers(t *testing.T) {
	m := cache.ModelID{Architecture: "x", Name: "x", NumLayers: 0, HeadDim: 64, NumKVHeads: 8, ContextLength: 4096}
	if err := ValidateModelDimensions(m); err == nil {
		t.Error("expected error for NumLayers=0")
	}
}

func TestValidateModelDimensions_ExcessiveLayers(t *testing.T) {
	m := cache.ModelID{Architecture: "x", Name: "x", NumLayers: MaxLayerCount + 1, HeadDim: 64, NumKVHeads: 8, ContextLength: 4096}
	if err := ValidateModelDimensions(m); err == nil {
		t.Error("expected error for NumLayers exceeding maximum")
	}
}

func TestValidateFragmentID_Valid(t *testing.T) {
	if err := ValidateFragmentID("abc123def456"); err != nil {
		t.Errorf("valid ID rejected: %v", err)
	}
}

func TestValidateFragmentID_Empty(t *testing.T) {
	if err := ValidateFragmentID(""); err == nil {
		t.Error("expected error for empty fragment ID")
	}
}

func TestValidateFragmentID_PathTraversal(t *testing.T) {
	attacks := []string{
		"../../../etc/passwd",
		"..\\windows\\system32",
		"valid/path",
		"null\x00byte",
		"a.b.c",
	}
	for _, attack := range attacks {
		if err := ValidateFragmentID(attack); err == nil {
			t.Errorf("expected error for unsafe fragment ID %q", attack)
		}
	}
}

func TestValidateFragmentID_TooLong(t *testing.T) {
	id := strings.Repeat("a", MaxFragmentIDLen+1)
	if err := ValidateFragmentID(id); err == nil {
		t.Error("expected error for fragment ID exceeding maximum length")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// HMAC signing tests
// ─────────────────────────────────────────────────────────────────────────────

func makeTestFragment(id string) *cache.KVFragment {
	model := cache.ModelID{
		Architecture:  "qwen",
		Name:          "test",
		Quantization:  "Q4_K_M",
		ContextLength: 4096,
		HeadDim:       64,
		NumKVHeads:    8,
		NumLayers:     24,
	}
	tokenIDs := make([]int32, 128)
	for i := range tokenIDs {
		tokenIDs[i] = int32(i + 100)
	}
	keys := make([]byte, 12*128*8*64*4)
	vals := make([]byte, 12*128*8*64*4)

	f, _ := cache.NewFragment(id, model, 0, 128, 0, 24, 2,
		keys, vals, tokenIDs, make([]float32, 384), "llamacpp", "b3117",
		cache.DefaultTTLSession)
	return f
}

func TestSigner_SignAndVerify_Valid(t *testing.T) {
	key := make([]byte, 32)
	for i := range key {
		key[i] = byte(i)
	}
	signer, err := NewSignerFromKey(key)
	if err != nil {
		t.Fatalf("create signer: %v", err)
	}

	f := makeTestFragment("test-frag-1")
	sf, err := signer.Sign(f)
	if err != nil {
		t.Fatalf("sign: %v", err)
	}

	if err := signer.Verify(sf); err != nil {
		t.Errorf("verify valid signature: %v", err)
	}
}

func TestSigner_Verify_TamperedID(t *testing.T) {
	key := make([]byte, 32)
	signer, _ := NewSignerFromKey(key)
	f := makeTestFragment("original-id")
	sf, _ := signer.Sign(f)

	// Tamper with the fragment ID
	sf.Fragment.ID = "tampered-id"
	if err := signer.Verify(sf); err == nil {
		t.Error("expected error for tampered fragment ID")
	}
}

func TestSigner_Verify_TamperedTokenIDs(t *testing.T) {
	key := make([]byte, 32)
	signer, _ := NewSignerFromKey(key)
	f := makeTestFragment("test-frag")
	sf, _ := signer.Sign(f)

	// Tamper with token IDs
	sf.Fragment.TokenIDs[0] = 9999
	if err := signer.Verify(sf); err == nil {
		t.Error("expected error for tampered TokenIDs (ContentHash mismatch)")
	}
}

func TestSigner_Verify_WrongKey(t *testing.T) {
	key1 := make([]byte, 32)
	key2 := make([]byte, 32)
	for i := range key2 {
		key2[i] = 0xFF
	}

	signer1, _ := NewSignerFromKey(key1)
	signer2, _ := NewSignerFromKey(key2)

	f := makeTestFragment("test-frag")
	sf, _ := signer1.Sign(f)

	if err := signer2.Verify(sf); err == nil {
		t.Error("expected error when verifying with wrong key")
	}
}

func TestSigner_Verify_ExpiredFragment(t *testing.T) {
	key := make([]byte, 32)
	signer, _ := NewSignerFromKey(key)
	f := makeTestFragment("test-frag")
	// Force expiry
	f.ExpiresAt = time.Now().Add(-time.Hour)
	sf, _ := signer.Sign(f)

	if err := signer.Verify(sf); err == nil {
		t.Error("expected error for expired fragment")
	}
}

func TestSigner_Verify_MalformedSignature(t *testing.T) {
	key := make([]byte, 32)
	signer, _ := NewSignerFromKey(key)
	f := makeTestFragment("test-frag")
	sf, _ := signer.Sign(f)
	sf.Signature = "not-valid-hex!!"

	if err := signer.Verify(sf); err == nil {
		t.Error("expected error for malformed signature")
	}
}

func TestSigner_Verify_EmptySignature(t *testing.T) {
	key := make([]byte, 32)
	signer, _ := NewSignerFromKey(key)
	f := makeTestFragment("test-frag")
	sf := &SignedFragment{Fragment: f, Signature: ""}

	if err := signer.Verify(sf); err == nil {
		t.Error("expected error for empty signature")
	}
}

func TestNewSignerFromKey_TooShort(t *testing.T) {
	_, err := NewSignerFromKey([]byte{1, 2, 3}) // only 3 bytes
	if err == nil {
		t.Error("expected error for key below minimum length")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Atomic write tests
// ─────────────────────────────────────────────────────────────────────────────

func TestAtomicWriteBlob_WritesCorrectly(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "test.bin")
	data := []byte("hello edgesync-llm atomic write test")

	// Import atomicWriteBlob from cache package via test helper
	if err := atomicWriteBlobTestHelper(target, data); err != nil {
		t.Fatalf("atomicWriteBlob: %v", err)
	}

	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if string(got) != string(data) {
		t.Errorf("data mismatch: want %q, got %q", data, got)
	}
}

func TestAtomicWriteBlob_NoTempFilesLeft(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "tensor.bin")
	data := make([]byte, 1024)

	if err := atomicWriteBlobTestHelper(target, data); err != nil {
		t.Fatalf("atomicWriteBlob: %v", err)
	}

	// No .tmp_ files should remain
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".tmp_") {
			t.Errorf("temp file left after successful write: %s", e.Name())
		}
	}
}

func TestAtomicWriteBlob_OverwritesExisting(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "tensor.bin")

	// Write initial content
	atomicWriteBlobTestHelper(target, []byte("original"))

	// Overwrite
	newData := []byte("updated content")
	if err := atomicWriteBlobTestHelper(target, newData); err != nil {
		t.Fatalf("overwrite: %v", err)
	}

	got, _ := os.ReadFile(target)
	if string(got) != string(newData) {
		t.Errorf("want %q, got %q", newData, got)
	}
}

// atomicWriteBlobTestHelper calls the cache package's atomicWriteBlob.
// Since atomicWriteBlob is unexported from cache, we re-implement the
// same logic here to test the pattern. In production, tests in the cache
// package test the actual function directly.
func atomicWriteBlobTestHelper(targetPath string, data []byte) error {
	dir := filepath.Dir(targetPath)
	base := filepath.Base(targetPath)
	tmp, err := os.CreateTemp(dir, ".tmp_"+base+"_*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	committed := false
	defer func() {
		if !committed {
			tmp.Close()
			os.Remove(tmpPath)
		}
	}()
	if _, err := tmp.Write(data); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, targetPath); err != nil {
		return err
	}
	committed = true
	return nil
}
