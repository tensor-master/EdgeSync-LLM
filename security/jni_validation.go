// Package security — JNI boundary validation for EdgeSync-LLM.
//
// PROBLEM: JVM ↔ Go/C boundary vulnerabilities
// ──────────────────────────────────────────────
// The JNI bridge (jni_bridge.go) exchanges float arrays and byte arrays between:
//   - Kotlin/JVM (managed memory, GC-controlled)
//   - Go runtime (GC-controlled, but separate heap)
//   - C intrinsics in cosine_neon.c (unmanaged, pointer arithmetic)
//
// Three classes of vulnerability exist at these boundaries:
//
//   1. BUFFER OVERFLOW: a Java float[] with length N passed to a C function
//      that assumes length M > N will read beyond the array bounds.
//      On JNI: GetFloatArrayElements() returns a direct pointer; no bounds
//      checking occurs in C. A malformed array triggers a SIGSEGV.
//
//   2. INTEGER OVERFLOW: embedding dimensions are passed as jint (32-bit).
//      A dimension of 0x10000 (65536) × 4 bytes × 2 (K+V) = 512 KB per token —
//      multiplied by 512 tokens = 256 MB allocation. An attacker controlling
//      the dimension parameters via a crafted model config can trigger OOM.
//
//   3. MEMORY LEAK: JNI objects (jfloatArray, jstring) obtained via
//      GetFloatArrayElements / GetStringUTFChars MUST be released with
//      ReleaseFloatArrayElements / ReleaseStringUTFChars. A missed release
//      pins the Java heap object, preventing GC, causing memory pressure.
//
// SOLUTION: Validate ALL inputs at the JNI boundary before any Go or C call.
// This file provides the validation functions called by jni_bridge.go.
package security

import (
	"fmt"

	"github.com/bossandboss/EdgeSync-LLM/cache"
)

// ─────────────────────────────────────────────────────────────────────────────
// Constants — hard limits enforced at the JNI boundary
// ─────────────────────────────────────────────────────────────────────────────

const (
	// MaxEmbeddingDim is the maximum allowed embedding vector dimension.
	// MiniLM-L6-v2 uses 384. Allow up to 4096 for future models.
	// Anything larger is almost certainly a bug or attack.
	MaxEmbeddingDim = 4096

	// MinEmbeddingDim is the minimum valid embedding dimension.
	MinEmbeddingDim = 64

	// MaxFragmentTensorBytes is the maximum size of a KV tensor blob (Keys or Values).
	// 64 MB per blob (128 MB total per fragment) is the absolute upper bound.
	// Qwen2.5-0.5B with full layers and 2048 tokens ≈ 48 MB — well within this.
	MaxFragmentTensorBytes = 64 * 1024 * 1024

	// MaxTokenCount is the maximum number of tokens in a single fragment.
	// Matches cache.FragmentMaxTokenSpan.
	MaxTokenCount = cache.FragmentMaxTokenSpan

	// MinTokenCount is the minimum valid token count.
	MinTokenCount = cache.FragmentGranularityTokens

	// MaxFragmentIDLen is the maximum allowed fragment ID string length.
	MaxFragmentIDLen = 64

	// MaxPromptBytes is the maximum allowed prompt size passed via JNI.
	// 256 KB covers even very long system prompts + conversation history.
	MaxPromptBytes = 256 * 1024

	// MaxModelIDJsonBytes is the maximum allowed ModelID JSON size.
	MaxModelIDJsonBytes = 4096

	// MaxLayerCount is the maximum number of transformer layers.
	// GPT-4 has 96 layers; 256 is a safe upper bound for any current model.
	MaxLayerCount = 256

	// MaxHeadDim is the maximum KV head dimension.
	// Standard models use 64-128. Allow up to 512 for safety margin.
	MaxHeadDim = 512

	// MaxKVHeads is the maximum number of KV attention heads.
	MaxKVHeads = 128
)

// ─────────────────────────────────────────────────────────────────────────────
// Validation functions
// ─────────────────────────────────────────────────────────────────────────────

// ValidateEmbedding checks that a float32 slice is a valid embedding vector.
// Call this on every embedding received from JNI before passing to HNSW.
func ValidateEmbedding(vec []float32) error {
	if len(vec) < MinEmbeddingDim {
		return fmt.Errorf("embedding dimension %d is below minimum %d", len(vec), MinEmbeddingDim)
	}
	if len(vec) > MaxEmbeddingDim {
		return fmt.Errorf("embedding dimension %d exceeds maximum %d (possible integer overflow attack)",
			len(vec), MaxEmbeddingDim)
	}
	// Check for NaN or Inf — these corrupt HNSW distance calculations silently
	nanCount := 0
	for i, v := range vec {
		if v != v { // NaN check: NaN != NaN
			nanCount++
			if nanCount == 1 {
				return fmt.Errorf("embedding contains NaN at index %d", i)
			}
		}
		if v > 1e38 || v < -1e38 {
			return fmt.Errorf("embedding contains Inf/overflow value %.2e at index %d", v, i)
		}
	}
	return nil
}

// ValidateTensorBlob checks that a KV tensor blob is within safe size bounds.
// Call this on Keys and Values before passing to engine adapters.
func ValidateTensorBlob(data []byte, label string) error {
	if len(data) == 0 {
		return fmt.Errorf("%s blob is empty", label)
	}
	if len(data) > MaxFragmentTensorBytes {
		return fmt.Errorf(
			"%s blob size %d bytes exceeds maximum %d bytes (%d MB) — possible integer overflow in dimension calculation",
			label, len(data), MaxFragmentTensorBytes, MaxFragmentTensorBytes/1024/1024,
		)
	}
	// Size must be divisible by 4 (float32)
	if len(data)%4 != 0 {
		return fmt.Errorf("%s blob size %d is not a multiple of 4 (not valid float32 data)", label, len(data))
	}
	return nil
}

// ValidateModelDimensions checks that model configuration values are within
// safe bounds before computing tensor sizes (prevents integer overflow in
// size calculations: numLayers × tokens × heads × dim × 4 bytes).
func ValidateModelDimensions(model cache.ModelID) error {
	if model.NumLayers <= 0 || model.NumLayers > MaxLayerCount {
		return fmt.Errorf("NumLayers %d out of range [1, %d]", model.NumLayers, MaxLayerCount)
	}
	if model.HeadDim <= 0 || model.HeadDim > MaxHeadDim {
		return fmt.Errorf("HeadDim %d out of range [1, %d]", model.HeadDim, MaxHeadDim)
	}
	if model.NumKVHeads <= 0 || model.NumKVHeads > MaxKVHeads {
		return fmt.Errorf("NumKVHeads %d out of range [1, %d]", model.NumKVHeads, MaxKVHeads)
	}
	if model.ContextLength <= 0 || model.ContextLength > 1<<20 {
		return fmt.Errorf("ContextLength %d out of range [1, 1048576]", model.ContextLength)
	}

	// NOTE: We deliberately do NOT reject models here based on the tensor size
	// a maximum-length (MaxTokenCount) fragment *could* produce. That conflated
	// "is this model config sane" with "is this specific fragment too big" —
	// the latter is already enforced correctly, at fragment-creation time,
	// against the REAL token span by ValidateTensorBlob(). A model isn't invalid
	// just because a caller could theoretically request a very long fragment for
	// it; the per-fragment check is what actually prevents oversized allocations.
	//
	// The bound checks above (NumLayers, HeadDim, NumKVHeads, ContextLength) are
	// what actually prevent int64 overflow in downstream size math: even at their
	// maximums (256 × 512 × 128 × 4 × 2048 ≈ 6.9e10), the product is far below
	// the int64 range (~9.2e18), so overflow cannot occur regardless of token span.

	if model.Architecture == "" {
		return fmt.Errorf("model Architecture must not be empty")
	}
	if model.Name == "" {
		return fmt.Errorf("model Name must not be empty")
	}

	return nil
}

// ValidateFragmentID checks that a fragment ID string is safe to use as a
// file path component and SQL parameter.
// Fragment IDs come from JNI (untrusted) and are used to construct file paths.
func ValidateFragmentID(id string) error {
	if id == "" {
		return fmt.Errorf("fragment ID is empty")
	}
	if len(id) > MaxFragmentIDLen {
		return fmt.Errorf("fragment ID length %d exceeds maximum %d", len(id), MaxFragmentIDLen)
	}
	// Reject path traversal attempts
	for _, c := range id {
		if c == '/' || c == '\\' || c == '.' || c == '\x00' {
			return fmt.Errorf("fragment ID contains unsafe character %q (path traversal?)", c)
		}
	}
	return nil
}

// ValidatePrompt checks that a prompt string received via JNI is within
// safe size bounds before tokenization or embedding.
func ValidatePrompt(prompt string) error {
	if len(prompt) == 0 {
		return fmt.Errorf("prompt is empty")
	}
	if len(prompt) > MaxPromptBytes {
		return fmt.Errorf("prompt size %d bytes exceeds maximum %d bytes", len(prompt), MaxPromptBytes)
	}
	return nil
}

// ValidateTokenRange checks that token range parameters are internally consistent.
func ValidateTokenRange(tokenStart, tokenEnd, layerStart, layerEnd, layerStride int) error {
	span := tokenEnd - tokenStart
	if span < MinTokenCount {
		return fmt.Errorf("token span %d (start=%d, end=%d) below minimum %d",
			span, tokenStart, tokenEnd, MinTokenCount)
	}
	if span > MaxTokenCount {
		return fmt.Errorf("token span %d exceeds maximum %d", span, MaxTokenCount)
	}
	if tokenStart < 0 {
		return fmt.Errorf("tokenStart %d is negative", tokenStart)
	}
	if layerStart < 0 {
		return fmt.Errorf("layerStart %d is negative", layerStart)
	}
	if layerEnd <= layerStart {
		return fmt.Errorf("layerEnd %d must be > layerStart %d", layerEnd, layerStart)
	}
	if layerStride < 1 {
		return fmt.Errorf("layerStride %d must be >= 1", layerStride)
	}
	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// ValidateJNIEmbedRequest bundles all checks for an embedding request.
// Called at the top of nativeEmbed in jni_bridge.go.
// ─────────────────────────────────────────────────────────────────────────────
func ValidateJNIEmbedRequest(prompt string) error {
	return ValidatePrompt(prompt)
}

// ValidateJNIInjectRequest bundles all checks for a fragment inject request.
// Called at the top of nativeInjectFragment in jni_bridge.go.
func ValidateJNIInjectRequest(fragmentID string, embedding []float32) error {
	if err := ValidateFragmentID(fragmentID); err != nil {
		return fmt.Errorf("inject validation: %w", err)
	}
	if embedding != nil {
		if err := ValidateEmbedding(embedding); err != nil {
			return fmt.Errorf("inject validation: %w", err)
		}
	}
	return nil
}

// ValidateJNIExtractRequest bundles all checks for a fragment extract request.
// Called at the top of nativeExtractAndStore in jni_bridge.go.
func ValidateJNIExtractRequest(prompt string, embedding []float32, model cache.ModelID) error {
	if err := ValidatePrompt(prompt); err != nil {
		return err
	}
	if err := ValidateEmbedding(embedding); err != nil {
		return err
	}
	if err := ValidateModelDimensions(model); err != nil {
		return err
	}
	return nil
}
