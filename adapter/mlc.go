// Package adapter — MLC-LLM backend implementation of KVAdapter.
//
// INTEGRATION NOTES FOR MLC-LLM
// ──────────────────────────────
// MLC-LLM (Machine Learning Compilation for LLMs) compiles models to native
// code via Apache TVM. It exposes KV cache access through:
//
//   mlc_llm.ChatModule.prefill()   — runs prefill and populates KV cache
//   mlc_llm.ChatModule.decode()    — generates one token
//
// Direct KV tensor access requires using the TVM runtime's NDArray API:
//   tvm.runtime.NDArray    — DLPack-compatible tensor handle
//   module.get_function("get_kv_cache") — retrieve raw KV cache NDArray
//
// MLC-LLM stores KV in a PagedAttention layout (since v0.1.0):
//   - Context is divided into pages (default page_size=16 tokens)
//   - Pages are allocated per sequence, not pre-allocated
//   - KV tensors are NOT contiguous in memory across pages
//
// This means our "flat contiguous slice" serialization (used by llama.cpp)
// does NOT directly apply. This adapter:
//   1. Iterates over active pages via get_kv_cache_pages()
//   2. Concatenates the page tensors into a contiguous buffer for storage
//   3. On inject, reconstructs pages and calls set_kv_cache_pages()
//
// The serialized format for MLC is:
//   [4 bytes: page_count][4 bytes: page_size_tokens]
//   per page: [4 bytes: token_offset][layer_data × num_layers]
//   layer_data: [keys: page_size×num_heads×head_dim × float16][values: same]
//
// NOTE: MLC-LLM uses float16 for KV by default (vs llama.cpp float32).
// The cache stores raw bytes and does NOT upcast — adapters must handle dtype.
//
// For Android deployment, MLC-LLM's Java/Kotlin bindings (mlc4j) are used.
// The page extraction logic maps to mlc4j's KVCacheState API.
package adapter

import (
	"context"
	"encoding/binary"
	"fmt"

	"react-example/cache"
)

const (
	mlcEngineName    = "mlc"
	mlcEngineVersion = "0.1.4"

	mlcDefaultPageSize = 16 // tokens per KV cache page (MLC default)
)

// MLCAdapter implements KVAdapter for MLC-LLM.
// The actual TVM/MLC runtime calls are injected via the MLCRuntime interface
// to allow testing without a real MLC build.
type MLCAdapter struct {
	runtime MLCRuntime
	modelID cache.ModelID
}

// MLCRuntime abstracts the MLC-LLM / TVM runtime calls.
// Implement this interface against the real mlc4j JNI bindings for Android,
// or the Python TVM ctypes bindings for desktop benchmarking.
type MLCRuntime interface {
	// Prefill runs the prefill pass on tokenIDs and populates the KV cache.
	Prefill(tokenIDs []int32) error

	// GetKVCachePages returns the raw page data for the given layer.
	// Returns [page_count][page_tokens * num_heads * head_dim] float16 bytes.
	GetKVCachePages(layer int) (keys [][]byte, values [][]byte, err error)

	// SetKVCachePages injects raw page data for the given layer.
	SetKVCachePages(layer int, keys [][]byte, values [][]byte) error

	// PageCount returns the number of active KV cache pages for the current sequence.
	PageCount() int

	// PageSizeTokens returns the number of tokens per KV cache page.
	PageSizeTokens() int

	// Generate produces text from the current KV cache state.
	Generate(startPos int, maxTokens int) (string, int, error)

	// Tokenize converts text to MLC token IDs.
	Tokenize(text string) ([]int32, error)

	// ClearCache discards the current KV cache state.
	ClearCache() error

	// Close releases all TVM / MLC resources.
	Close() error
}

// NewMLCAdapter creates an MLCAdapter wrapping the given runtime.
func NewMLCAdapter(runtime MLCRuntime, modelID cache.ModelID) *MLCAdapter {
	return &MLCAdapter{runtime: runtime, modelID: modelID}
}

// ── KVAdapter identity ────────────────────────────────────────────────────────

func (a *MLCAdapter) EngineName() string    { return mlcEngineName }
func (a *MLCAdapter) EngineVersion() string  { return mlcEngineVersion }
func (a *MLCAdapter) ModelID() cache.ModelID { return a.modelID }

// CompatibleWith: MLC can only inject its own format.
// The paged layout is incompatible with llama.cpp's flat layout without conversion.
func (a *MLCAdapter) CompatibleWith() []string { return []string{} }

// ── Fragment extraction ───────────────────────────────────────────────────────

// ExtractFragment runs prefill on tokenIDs and captures the KV cache pages.
//
// Serialization format (MLC wire format):
//   header:  [4B: page_count] [4B: page_size_tokens] [4B: num_layers_captured]
//   per layer captured:
//     [4B: layer_index]
//     per page:
//       keys:   [page_size_tokens × num_heads × head_dim] float16 (2 bytes/float)
//       values: [page_size_tokens × num_heads × head_dim] float16
func (a *MLCAdapter) ExtractFragment(
	ctx context.Context,
	tokenIDs []int32,
	layerStart, layerEnd, layerStride int,
	embedding []float32,
) (*cache.KVFragment, error) {
	// Run prefill to populate KV cache
	if err := a.runtime.Prefill(tokenIDs); err != nil {
		return nil, fmt.Errorf("mlc ExtractFragment: prefill failed: %w", err)
	}

	pageCount := a.runtime.PageCount()
	pageSize := a.runtime.PageSizeTokens()

	// Collect layers to capture
	var capturedLayers []int
	for l := layerStart; l < layerEnd; l += layerStride {
		capturedLayers = append(capturedLayers, l)
	}

	// Serialize header
	var keys, vals []byte
	header := make([]byte, 12)
	binary.LittleEndian.PutUint32(header[0:], uint32(pageCount))
	binary.LittleEndian.PutUint32(header[4:], uint32(pageSize))
	binary.LittleEndian.PutUint32(header[8:], uint32(len(capturedLayers)))
	keys = append(keys, header...)
	vals = append(vals, header...) // vals carries its own copy of the header

	// Serialize per-layer page data
	for _, layer := range capturedLayers {
		layerHeader := make([]byte, 4)
		binary.LittleEndian.PutUint32(layerHeader, uint32(layer))
		keys = append(keys, layerHeader...)
		vals = append(vals, layerHeader...)

		kPages, vPages, err := a.runtime.GetKVCachePages(layer)
		if err != nil {
			return nil, fmt.Errorf("mlc ExtractFragment: GetKVCachePages layer %d: %w", layer, err)
		}
		if len(kPages) != pageCount || len(vPages) != pageCount {
			return nil, fmt.Errorf("mlc ExtractFragment: expected %d pages, got k=%d v=%d",
				pageCount, len(kPages), len(vPages))
		}

		for _, page := range kPages {
			keys = append(keys, page...)
		}
		for _, page := range vPages {
			vals = append(vals, page...)
		}
	}

	fragmentID := generateFragmentID(tokenIDs, a.modelID)

	return cache.NewFragment(
		fragmentID,
		a.modelID,
		0, len(tokenIDs),
		layerStart, layerEnd, layerStride,
		keys, vals,
		tokenIDs,
		embedding,
		mlcEngineName,
		mlcEngineVersion,
		cache.DefaultTTLSession,
	)
}

// InjectFragment deserializes MLC-format fragment data and restores KV cache pages.
func (a *MLCAdapter) InjectFragment(ctx context.Context, fragment *cache.KVFragment) error {
	if err := CanInject(a, fragment); err != nil {
		return fmt.Errorf("mlc InjectFragment: %w", err)
	}

	keys := fragment.Keys
	vals := fragment.Values

	if len(keys) < 12 {
		return fmt.Errorf("mlc InjectFragment: header too short (%d bytes)", len(keys))
	}

	// Parse header
	pageCount := int(binary.LittleEndian.Uint32(keys[0:]))
	pageSize := int(binary.LittleEndian.Uint32(keys[4:]))
	numLayers := int(binary.LittleEndian.Uint32(keys[8:]))

	kCursor := 12
	vCursor := 12
	model := a.modelID
	// float16 = 2 bytes; bytes per page = pageSize × numHeads × headDim × 2
	bytesPerPage := pageSize * model.NumKVHeads * model.HeadDim * 2

	for li := 0; li < numLayers; li++ {
		if kCursor+4 > len(keys) {
			return fmt.Errorf("mlc InjectFragment: truncated layer header at layer_idx %d", li)
		}
		layer := int(binary.LittleEndian.Uint32(keys[kCursor:]))
		kCursor += 4
		vCursor += 4

		kPages := make([][]byte, pageCount)
		vPages := make([][]byte, pageCount)
		for p := 0; p < pageCount; p++ {
			end := kCursor + bytesPerPage
			if end > len(keys) {
				return fmt.Errorf("mlc InjectFragment: buffer underflow at layer %d page %d", layer, p)
			}
			kPages[p] = keys[kCursor:end]
			vPages[p] = vals[vCursor : vCursor+bytesPerPage]
			kCursor += bytesPerPage
			vCursor += bytesPerPage
		}

		if err := a.runtime.SetKVCachePages(layer, kPages, vPages); err != nil {
			return fmt.Errorf("mlc InjectFragment: SetKVCachePages layer %d: %w", layer, err)
		}
	}

	return nil
}

func (a *MLCAdapter) Generate(ctx context.Context, prompt string, startTokenPos int, maxTokens int) (string, int, error) {
	return a.runtime.Generate(startTokenPos, maxTokens)
}

func (a *MLCAdapter) Tokenize(_ context.Context, text string) ([]int32, error) {
	return a.runtime.Tokenize(text)
}

func (a *MLCAdapter) ClearKVCache(_ context.Context) error {
	return a.runtime.ClearCache()
}

func (a *MLCAdapter) Close() error {
	return a.runtime.Close()
}
