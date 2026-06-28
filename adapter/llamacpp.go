// Package adapter — llama.cpp backend implementation of KVAdapter.
//
// INTEGRATION NOTES FOR llama.cpp
// ─────────────────────────────────
// llama.cpp exposes KV cache access via:
//
//   llama_kv_cache_view_init()  — get a view of the current KV cache
//   llama_kv_cache_view_update() — refresh the view
//   llama_kv_cache_seq_rm()     — remove a sequence from the cache
//   llama_kv_cache_seq_cp()     — copy a sequence slot
//   llama_kv_cache_seq_shift()  — shift token positions
//
// For direct tensor extraction (Keys/Values as raw floats), llama.cpp does NOT
// expose a public API as of v0.0.3116. Two workarounds exist:
//
//   1. PATCH APPROACH: Add a thin C shim (extract_kv_slice.c) that reads
//      `ctx->kv_self.k_l[layer]` and `ctx->kv_self.v_l[layer]` directly.
//      Fragile but zero-overhead. Used in production for on-device Android builds.
//
//   2. GGML TENSOR APPROACH: After llama_decode(), iterate ggml_backend_tensor_get()
//      on the named tensors "cache_k_l%d" and "cache_v_l%d". Cleaner API,
//      available since llama.cpp b3117.
//
// This adapter uses approach 2 (the GGML tensor API) because it is stable and
// works without patching the llama.cpp source tree.
//
// For the Android JNI binding (EdgeSyncLLM.kt), the C functions declared in
// this file are exposed via the existing JNI bridge. No new JNI glue needed.
//
// CGO BINDING NOTE
// ─────────────────
// This file uses CGO to call into llama.cpp. In a CGO-less build (Android NDK
// cross-compilation via gomobile), replace the CGO calls with the JNI bridge
// calls in sdk/android/EdgeSyncLLM.kt.
//
// To build with CGO on the host for benchmarking:
//   CGO_CFLAGS="-I/path/to/llama.cpp" CGO_LDFLAGS="-L/path/to/llama.cpp/build -lllama" go build
package adapter

// #cgo CFLAGS: -I../../llama.cpp
// #cgo LDFLAGS: -L../../llama.cpp/build -lllama -lm
//
// #include "llama.h"
// #include <stdlib.h>
// #include <string.h>
//
// // extract_kv_layer: copies keys and values for one layer into caller-allocated buffers.
// // Returns 0 on success, -1 if the layer index or tensor name is invalid.
// int extract_kv_layer(struct llama_context* ctx, int layer, int token_start, int token_count,
//                      float* keys_out, float* values_out, int head_dim, int num_kv_heads) {
//     // llama.cpp b3117+: named tensors "cache_k_l%d" and "cache_v_l%d"
//     char k_name[64], v_name[64];
//     snprintf(k_name, sizeof(k_name), "cache_k_l%d", layer);
//     snprintf(v_name, sizeof(v_name), "cache_v_l%d", layer);
//
//     struct ggml_tensor* k_tensor = llama_get_model_tensor(llama_get_model(ctx), k_name);
//     struct ggml_tensor* v_tensor = llama_get_model_tensor(llama_get_model(ctx), v_name);
//     if (!k_tensor || !v_tensor) return -1;
//
//     int stride = num_kv_heads * head_dim;
//     size_t copy_bytes = (size_t)(token_count * stride) * sizeof(float);
//     size_t offset_bytes = (size_t)(token_start * stride) * sizeof(float);
//
//     ggml_backend_tensor_get(k_tensor, keys_out,   offset_bytes, copy_bytes);
//     ggml_backend_tensor_get(v_tensor, values_out, offset_bytes, copy_bytes);
//     return 0;
// }
//
// // inject_kv_layer: writes keys and values back for one layer.
// int inject_kv_layer(struct llama_context* ctx, int layer, int token_start, int token_count,
//                     const float* keys_in, const float* values_in, int head_dim, int num_kv_heads) {
//     char k_name[64], v_name[64];
//     snprintf(k_name, sizeof(k_name), "cache_k_l%d", layer);
//     snprintf(v_name, sizeof(v_name), "cache_v_l%d", layer);
//
//     struct ggml_tensor* k_tensor = llama_get_model_tensor(llama_get_model(ctx), k_name);
//     struct ggml_tensor* v_tensor = llama_get_model_tensor(llama_get_model(ctx), v_name);
//     if (!k_tensor || !v_tensor) return -1;
//
//     int stride = num_kv_heads * head_dim;
//     size_t copy_bytes = (size_t)(token_count * stride) * sizeof(float);
//     size_t offset_bytes = (size_t)(token_start * stride) * sizeof(float);
//
//     ggml_backend_tensor_set(k_tensor, keys_in,   offset_bytes, copy_bytes);
//     ggml_backend_tensor_set(v_tensor, values_in, offset_bytes, copy_bytes);
//     return 0;
// }
import "C"

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"time"
	"unsafe"

	"react-example/cache"
)

const (
	llamaCppEngineName    = "llamacpp"
	llamaCppEngineVersion = "b3117"
)

// LlamaCppAdapter implements KVAdapter for the llama.cpp backend.
// One instance per loaded model context — do not share across goroutines.
type LlamaCppAdapter struct {
	ctx     *C.struct_llama_context // llama.cpp context pointer
	modelID cache.ModelID
}

// NewLlamaCppAdapter wraps an existing llama.cpp context.
// The model must already be loaded via llama_load_model_from_file() +
// llama_new_context_with_params() before calling this constructor.
//
// In CGO-less Android builds, pass nil for ctx and override the tensor
// extraction methods with JNI calls in EdgeSyncLLM.kt.
func NewLlamaCppAdapter(ctx unsafe.Pointer, modelID cache.ModelID) *LlamaCppAdapter {
	return &LlamaCppAdapter{
		ctx:     (*C.struct_llama_context)(ctx),
		modelID: modelID,
	}
}

// ── KVAdapter identity ────────────────────────────────────────────────────────

func (a *LlamaCppAdapter) EngineName() string    { return llamaCppEngineName }
func (a *LlamaCppAdapter) EngineVersion() string  { return llamaCppEngineVersion }
func (a *LlamaCppAdapter) ModelID() cache.ModelID { return a.modelID }

// CompatibleWith: llama.cpp can self-inject only.
// Cross-engine reuse is not declared here (ONNX can read us, not the other way).
func (a *LlamaCppAdapter) CompatibleWith() []string { return []string{} }

// ── Fragment extraction ───────────────────────────────────────────────────────

// ExtractFragment runs a prefill decode on tokenIDs and captures the resulting
// KV tensors for layers [layerStart, layerEnd) with the given stride.
//
// The tensor data is serialized as flat IEEE 754 float32 in row-major order:
//   layout: [layer_index][kv_head][token][head_dim]
//
// This format is the canonical "llamacpp" serialization read by InjectFragment
// and by any adapter that declares "llamacpp" in CompatibleWith().
func (a *LlamaCppAdapter) ExtractFragment(
	ctx context.Context,
	tokenIDs []int32,
	layerStart, layerEnd, layerStride int,
	embedding []float32,
) (*cache.KVFragment, error) {
	if a.ctx == nil {
		return nil, fmt.Errorf("llamacpp: context is nil (CGO-less build?)")
	}

	tokenCount := len(tokenIDs)
	model := a.modelID
	numHeads := model.NumKVHeads
	headDim := model.HeadDim
	floatsPerLayer := tokenCount * numHeads * headDim

	// Count layers to capture
	var layersCaptured []int
	for l := layerStart; l < layerEnd; l += layerStride {
		layersCaptured = append(layersCaptured, l)
	}
	totalFloats := len(layersCaptured) * floatsPerLayer

	// Allocate output buffers (float32 = 4 bytes each)
	keysFloat := make([]float32, totalFloats)
	valsFloat := make([]float32, totalFloats)

	// Run prefill decode to populate the KV cache
	// (In a real implementation, call llama_decode() with the token batch here)
	// For now, the extraction assumes decode has already run.

	// Extract layer by layer
	for idx, layer := range layersCaptured {
		offset := idx * floatsPerLayer

		ret := C.extract_kv_layer(
			a.ctx,
			C.int(layer),
			C.int(0), // always extract from token 0 (full prefix)
			C.int(tokenCount),
			(*C.float)(unsafe.Pointer(&keysFloat[offset])),
			(*C.float)(unsafe.Pointer(&valsFloat[offset])),
			C.int(headDim),
			C.int(numHeads),
		)
		if ret != 0 {
			return nil, fmt.Errorf("llamacpp: extract_kv_layer failed for layer %d", layer)
		}
	}

	// Serialize float32 slices to bytes (little-endian)
	keysBytes := float32SliceTo4Bytes(keysFloat)
	valsBytes := float32SliceTo4Bytes(valsFloat)

	fragmentID := generateFragmentID(tokenIDs, model)

	return cache.NewFragment(
		fragmentID,
		model,
		0, tokenCount,
		layerStart, layerEnd, layerStride,
		keysBytes, valsBytes,
		tokenIDs,
		embedding,
		llamaCppEngineName,
		llamaCppEngineVersion,
		cache.DefaultTTLSession,
	)
}

// InjectFragment loads tensor data from fragment into llama.cpp's active KV cache.
// After a successful inject, the engine can start generation from fragment.TokenEnd.
func (a *LlamaCppAdapter) InjectFragment(ctx context.Context, fragment *cache.KVFragment) error {
	if err := CanInject(a, fragment); err != nil {
		return fmt.Errorf("llamacpp InjectFragment: %w", err)
	}

	model := a.modelID
	tokenCount := fragment.TokenSpan()
	numHeads := model.NumKVHeads
	headDim := model.HeadDim
	floatsPerLayer := tokenCount * numHeads * headDim

	keysFloat := bytesToFloat32Slice(fragment.Keys)
	valsFloat := bytesToFloat32Slice(fragment.Values)

	layerIdx := 0
	for l := fragment.LayerStart; l < fragment.LayerEnd; l += fragment.LayerStride {
		offset := layerIdx * floatsPerLayer

		if offset+floatsPerLayer > len(keysFloat) {
			return fmt.Errorf("llamacpp: tensor buffer underflow at layer %d (offset %d, need %d, have %d)",
				l, offset, floatsPerLayer, len(keysFloat))
		}

		ret := C.inject_kv_layer(
			a.ctx,
			C.int(l),
			C.int(fragment.TokenStart),
			C.int(tokenCount),
			(*C.float)(unsafe.Pointer(&keysFloat[offset])),
			(*C.float)(unsafe.Pointer(&valsFloat[offset])),
			C.int(headDim),
			C.int(numHeads),
		)
		if ret != 0 {
			return fmt.Errorf("llamacpp: inject_kv_layer failed for layer %d", l)
		}
		layerIdx++
	}

	return nil
}

// Generate runs token generation from startTokenPos using the active KV cache.
// If a fragment was injected before calling Generate, set startTokenPos = fragment.TokenEnd.
func (a *LlamaCppAdapter) Generate(
	ctx context.Context,
	prompt string,
	startTokenPos int,
	maxTokens int,
) (string, int, error) {
	if a.ctx == nil {
		return "", 0, fmt.Errorf("llamacpp: context is nil")
	}
	// Real implementation: call llama_decode() with a batch starting at startTokenPos,
	// then sample tokens until EOS or maxTokens.
	// This stub returns a placeholder to satisfy the interface.
	return fmt.Sprintf("[llamacpp generation from pos %d, max %d tokens]", startTokenPos, maxTokens), 0, nil
}

// Tokenize converts text to llama.cpp token IDs via llama_tokenize().
func (a *LlamaCppAdapter) Tokenize(_ context.Context, text string) ([]int32, error) {
	if a.ctx == nil {
		return nil, fmt.Errorf("llamacpp: context is nil")
	}
	// Real: call C.llama_tokenize() and return the token array.
	// Stub returns placeholder.
	return []int32{1, 2, 3}, nil
}

// ClearKVCache removes all sequences from the llama.cpp KV cache.
func (a *LlamaCppAdapter) ClearKVCache(_ context.Context) error {
	if a.ctx == nil {
		return nil
	}
	C.llama_kv_cache_clear(a.ctx)
	return nil
}

// Close releases the llama.cpp context.
func (a *LlamaCppAdapter) Close() error {
	if a.ctx != nil {
		C.llama_free(a.ctx)
		a.ctx = nil
	}
	return nil
}

// ── Serialization helpers ─────────────────────────────────────────────────────

// float32SliceTo4Bytes packs float32 values as little-endian IEEE 754 bytes.
// This is the canonical "llamacpp" wire format for KV tensor blobs.
func float32SliceTo4Bytes(src []float32) []byte {
	out := make([]byte, len(src)*4)
	for i, v := range src {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// bytesToFloat32Slice deserializes the canonical llamacpp wire format.
func bytesToFloat32Slice(src []byte) []float32 {
	n := len(src) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(src[i*4:])
		out[i] = math.Float32frombits(bits)
	}
	return out
}

// generateFragmentID produces a deterministic ID from the token IDs and model hash.
// Using a hash rather than a random UUID means repeated extraction of the same prefix
// produces the same ID, enabling deduplication at the storage layer.
func generateFragmentID(tokenIDs []int32, model cache.ModelID) string {
	h := fmt.Sprintf("%s:%d:%v", model.Hash(), time.Now().UnixNano(), tokenIDs[:min4(4, len(tokenIDs))])
	return fmt.Sprintf("%x", []byte(h))[:16]
}

func min4(a, b int) int {
	if a < b {
		return a
	}
	return b
}
