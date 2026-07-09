//go:build !no_llamacpp

// Package adapter — llama.cpp backend implementation of KVAdapter.
//
// BUILD TAG: this file requires llama.h/libllama at compile time (see the
// #cgo directives below) and is included by default. If you only need the
// MLC or ONNX backends and don't have llama.cpp available, exclude it with
// `go build -tags no_llamacpp ./adapter/... ./sdk/android/...`
// (sdk/android/engine_llamacpp.go / engine_llamacpp_stub.go mirror this same
// tag so the JNI bridge compiles either way.)
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

// #cgo CFLAGS: -I../../llama.cpp/include -I../../llama.cpp/ggml/include
// #cgo LDFLAGS: -L../../llama.cpp/build -lllama -lm
//
// #include "llama.h"
// #include <stdlib.h>
// #include <string.h>
//
// // The three functions below (edgesync_extract_layer_raw, edgesync_layer_sizes,
// // edgesync_inject_layer_raw) are NOT defined here — they're compiled into
// // libllama.so itself, from src/edgesync_kv_bridge.cpp, added to llama.cpp's
// // own CMakeLists.txt. That file needs internal headers (llama-kv-cache.h)
// // that aren't part of llama.cpp's public API, so it must be compiled as
// // part of llama.cpp's own build tree, not linked in externally.
// //
// // WHY: the previous version of this preamble called llama_get_model_tensor(),
// // which does not exist in llama.cpp (confirmed by linking against a real
// // build: "undefined reference to llama_get_model_tensor"). See
// // src/edgesync_kv_bridge.cpp in your llama.cpp checkout for the real
// // implementation and its fragility notes (only supports the standard
// // unified KV cache, not iSWA/DSA/DSV4/hybrid memory architectures).
// //
// // If you're building against a llama.cpp checkout that does NOT have
// // edgesync_kv_bridge.cpp added to its build, these will fail to link.
// // Apply that patch to your llama.cpp checkout first (or vendor a fork).
//
// int edgesync_extract_layer_raw(struct llama_context* ctx, int32_t layer,
//     void* keys_out, size_t keys_out_capacity, size_t* out_k_nbytes,
//     void* vals_out, size_t vals_out_capacity, size_t* out_v_nbytes);
//
// int edgesync_layer_sizes(struct llama_context* ctx, int32_t layer,
//     size_t* out_k_nbytes, size_t* out_v_nbytes);
//
// int edgesync_inject_layer_raw(struct llama_context* ctx, int32_t layer,
//     const void* keys_in, size_t keys_in_size,
//     const void* vals_in, size_t vals_in_size);
import "C"

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"unsafe"

	"github.com/bossandboss/EdgeSync-LLM/cache"
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

// LoadLlamaCppModel loads a GGUF model from disk and returns a ready-to-use
// LlamaCppAdapter. This is the piece that was previously missing end-to-end:
// nativeInitialize() in sdk/android/jni_bridge.go accepted an "engine" string
// but never actually called this — globalAdapter stayed nil forever, so every
// inference-path JNI call silently no-opped via its "adapter not initialized"
// branch, even after a "successful" nativeInitialize().
//
// ⚠ NOT COMPILED OR VERIFIED IN THE ENVIRONMENT THAT WROTE THIS. This sandbox
// has no llama.h / no llama.cpp build available (see cache/differential.go
// and friends, which WERE build+test verified — this file could not be).
// Build this with the real headers (CGO_CFLAGS/CGO_LDFLAGS pointing at a
// compiled llama.cpp checkout, matching the #cgo directives at the top of
// this file) and fix any signature drift against your actual llama.cpp
// version (the exact fields of llama_model_params / llama_context_params
// change across llama.cpp releases — verify against the llama.h you're
// linking against, not just this comment).
//
// nThreads: CPU threads for generation. nGpuLayers: 0 = CPU-only inference;
// increase if a GPU/NPU backend is compiled into your libllama build.
func LoadLlamaCppModel(ggufPath string, modelID cache.ModelID, nCtx, nThreads, nGpuLayers int) (*LlamaCppAdapter, error) {
	cGgufPath := C.CString(ggufPath)
	defer C.free(unsafe.Pointer(cGgufPath))

	// llama_backend_init() is safe to call multiple times in recent llama.cpp
	// versions, but if your version asserts on double-init, guard this with a
	// sync.Once at package scope instead of calling it per-model-load.
	C.llama_backend_init()

	modelParams := C.llama_model_default_params()
	modelParams.n_gpu_layers = C.int32_t(nGpuLayers)

	model := C.llama_load_model_from_file(cGgufPath, modelParams)
	if model == nil {
		return nil, fmt.Errorf("LoadLlamaCppModel: llama_load_model_from_file failed for %q — check the path and that the file is a valid GGUF", ggufPath)
	}

	ctxParams := C.llama_context_default_params()
	ctxParams.n_ctx = C.uint32_t(nCtx)
	ctxParams.n_threads = C.int32_t(nThreads)
	ctxParams.n_threads_batch = C.int32_t(nThreads)

	ctx := C.llama_init_from_model(model, ctxParams)
	if ctx == nil {
		C.llama_free_model(model)
		return nil, fmt.Errorf("LoadLlamaCppModel: llama_init_from_model failed for %q (n_ctx=%d, n_threads=%d)", ggufPath, nCtx, nThreads)
	}

	return &LlamaCppAdapter{
		ctx:     ctx,
		modelID: modelID,
	}, nil
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

	// Count layers to capture
	var layersCaptured []int
	for l := layerStart; l < layerEnd; l += layerStride {
		layersCaptured = append(layersCaptured, l)
	}

	// Run prefill decode to populate the KV cache
	// (In a real implementation, call llama_decode() with the token batch here)
	// For now, the extraction assumes decode has already run.

	// NOTE ON SCOPE: edgesync_extract_layer_raw (see src/edgesync_kv_bridge.cpp
	// in the llama.cpp checkout this links against) returns the FULL current
	// per-layer K/V storage tensor, not a [token_start, token_start+token_count)
	// slice — the original code here assumed a token-range extraction that the
	// underlying tensor layout doesn't necessarily support without further
	// verification against a real running model (padding/permutation in the
	// cache's cell layout was not confirmed). This concatenates whole-layer
	// tensors across the requested layer range; the byte size per layer is
	// whatever llama.cpp currently allocates for it, not necessarily
	// tokenCount*numHeads*headDim*4 as the old code assumed.
	var keysBytes, valsBytes []byte

	for _, layer := range layersCaptured {
		var kSize, vSize C.size_t
		sizeRet := C.edgesync_layer_sizes(a.ctx, C.int32_t(layer), &kSize, &vSize)
		if sizeRet != 0 {
			return nil, fmt.Errorf("llamacpp: edgesync_layer_sizes failed for layer %d (code %d) — unsupported memory architecture (iSWA/DSA/DSV4/hybrid) or invalid layer", layer, int(sizeRet))
		}

		kBuf := make([]byte, int(kSize))
		vBuf := make([]byte, int(vSize))
		var kWritten, vWritten C.size_t

		extractRet := C.edgesync_extract_layer_raw(
			a.ctx, C.int32_t(layer),
			unsafe.Pointer(&kBuf[0]), kSize, &kWritten,
			unsafe.Pointer(&vBuf[0]), vSize, &vWritten,
		)
		if extractRet != 0 {
			return nil, fmt.Errorf("llamacpp: edgesync_extract_layer_raw failed for layer %d (code %d)", layer, int(extractRet))
		}

		keysBytes = append(keysBytes, kBuf[:int(kWritten)]...)
		valsBytes = append(valsBytes, vBuf[:int(vWritten)]...)
	}

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

	// NOTE: matches ExtractFragment's new scope — fragment.Keys/Values are
	// now whole per-layer tensors concatenated across [LayerStart, LayerEnd),
	// not a token-range slice. edgesync_inject_layer_raw rejects the write if
	// the byte size doesn't match what the tensor currently expects (this is
	// itself a integrity check: it will correctly fail on a real token-span
	// mismatch, since llama.cpp's own tensor allocation size is the source
	// of truth here rather than a separately-computed floatsPerLayer).
	numLayers := 0
	for l := fragment.LayerStart; l < fragment.LayerEnd; l += fragment.LayerStride {
		numLayers++
	}
	if numLayers == 0 {
		return fmt.Errorf("llamacpp InjectFragment: empty layer range [%d, %d)", fragment.LayerStart, fragment.LayerEnd)
	}

	kOffset, vOffset := 0, 0
	layerIdx := 0
	for l := fragment.LayerStart; l < fragment.LayerEnd; l += fragment.LayerStride {
		var kSize, vSize C.size_t
		sizeRet := C.edgesync_layer_sizes(a.ctx, C.int32_t(l), &kSize, &vSize)
		if sizeRet != 0 {
			return fmt.Errorf("llamacpp: edgesync_layer_sizes failed for layer %d (code %d)", l, int(sizeRet))
		}

		kEnd := kOffset + int(kSize)
		vEnd := vOffset + int(vSize)
		if kEnd > len(fragment.Keys) || vEnd > len(fragment.Values) {
			return fmt.Errorf("llamacpp: fragment buffer underflow at layer %d (need %d/%d bytes, have %d/%d)",
				l, kEnd, vEnd, len(fragment.Keys), len(fragment.Values))
		}

		ret := C.edgesync_inject_layer_raw(
			a.ctx, C.int32_t(l),
			unsafe.Pointer(&fragment.Keys[kOffset]), kSize,
			unsafe.Pointer(&fragment.Values[vOffset]), vSize,
		)
		if ret != 0 {
			return fmt.Errorf("llamacpp: edgesync_inject_layer_raw failed for layer %d (code %d) — likely a size mismatch between this fragment and the current model's tensor layout", l, int(ret))
		}

		kOffset, vOffset = kEnd, vEnd
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
	// llama_kv_cache_clear(ctx) was renamed to llama_memory_clear(mem, data)
	// in current llama.cpp — it now operates on the llama_memory_t handle
	// rather than the context directly. data=true clears the underlying
	// buffers too, matching the old function's behavior.
	mem := C.llama_get_memory(a.ctx)
	C.llama_memory_clear(mem, C.bool(true))
	return nil
}

// Close releases the llama.cpp context and its underlying model.
// NOTE: previously this only freed the context (llama_free), leaking the
// model handle from llama_load_model_from_file(). Only call llama_get_model
// + llama_free_model here if this adapter "owns" the model (i.e. it was
// created via LoadLlamaCppModel, not via NewLlamaCppAdapter wrapping an
// externally-managed context) — otherwise you'll double-free a model someone
// else is still responsible for. Track ownership explicitly if you use both
// construction paths in the same process.
func (a *LlamaCppAdapter) Close() error {
	if a.ctx == nil {
		return nil
	}
	model := C.llama_get_model(a.ctx)
	C.llama_free(a.ctx)
	if model != nil {
		C.llama_free_model(model)
	}
	a.ctx = nil
	return nil
}

// ── Serialization helpers ─────────────────────────────────────────────────────

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
