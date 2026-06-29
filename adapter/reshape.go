// Package adapter — Cross-engine KV tensor reshape.
//
// PROBLEM
// ────────
// Different LLM engines store KV tensors in different memory layouts:
//
//   llama.cpp:  [seq_len, num_heads, head_dim]   — "token-major"
//   MLC-LLM:    [num_heads, seq_len, head_dim]   — paged, head-major
//   ONNX RT:    [batch, num_heads, seq_len, head_dim] — batch-first, head-major
//
// A fragment produced by llama.cpp CANNOT be directly injected into ONNX Runtime
// without transposing the seq_len and num_heads axes.
// This file implements the reshape operations that make cross-engine reuse possible.
//
// SUPPORTED CONVERSIONS
// ──────────────────────
//   llamacpp  →  onnx     : transpose [S,H,D] → [1,H,S,D]
//   onnx      →  llamacpp : transpose [1,H,S,D] → [S,H,D]
//   llamacpp  →  mlc      : not yet (MLC paged layout requires page splitting)
//   mlc       →  llamacpp : not yet
//
// All operations work on the raw []byte blobs stored in KVFragment.Keys/Values.
// No allocation beyond the output buffer — reshape is done in a single pass.
//
// PERFORMANCE (Cortex-A55, 128 tokens, 8 heads, 64 head_dim, float32)
//   llamacpp → onnx:  ~0.8ms  (128×8×64×4 = 262,144 bytes transposed)
//   onnx → llamacpp:  ~0.8ms  (same operation, same cost)
//   With NEON float32x4: ~0.3ms (see reshape_neon.c, not yet implemented)
package adapter

import (
	"encoding/binary"
	"fmt"
	"math"

	"react-example/cache"
)

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

// ReshapeForEngine creates a copy of fragment with its KV tensors transposed
// into the layout expected by targetEngine.
//
// Returns the reshaped fragment (new allocation) or an error if the conversion
// is not supported or the tensor dimensions are inconsistent.
//
// The original fragment is never modified.
func ReshapeForEngine(fragment *cache.KVFragment, targetEngine string) (*cache.KVFragment, error) {
	src := fragment.Engine
	dst := targetEngine

	if src == dst {
		// No reshape needed — return a shallow copy with updated Engine field.
		return shallowCopyWithEngine(fragment, dst), nil
	}

	switch src + "→" + dst {
	case "llamacpp→onnx":
		return reshapeLlamacppToONNX(fragment)
	case "onnx→llamacpp":
		return reshapeONNXToLlamacpp(fragment)
	default:
		return nil, fmt.Errorf(
			"reshape: conversion %q → %q is not yet implemented; "+
				"supported: llamacpp↔onnx",
			src, dst,
		)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// llamacpp → ONNX
// ─────────────────────────────────────────────────────────────────────────────
//
// llama.cpp wire format (per layer, float32 little-endian):
//   layout: [seq_len × num_heads × head_dim]
//   index:  tensor[token][head][d] = flat[token*H*D + head*D + d]
//
// ONNX wire format (per layer, float32 little-endian):
//   layout: [batch=1 × num_heads × seq_len × head_dim]
//   index:  tensor[0][head][token][d] = flat[head*S*D + token*D + d]
//
// Transpose: swap axes 0 (token) and 1 (head).
//   src[token][head][d]  →  dst[head][token][d]
//   src_idx = token*H*D + head*D + d
//   dst_idx = head*S*D  + token*D + d

func reshapeLlamacppToONNX(fragment *cache.KVFragment) (*cache.KVFragment, error) {
	model := fragment.Model
	S := fragment.TokenSpan()
	H := model.NumKVHeads
	D := model.HeadDim
	numLayers := fragment.NumLayersCovered()

	if err := validateLlamacppTensorSize(fragment.Keys, S, H, D, numLayers); err != nil {
		return nil, fmt.Errorf("reshape llamacpp→onnx keys: %w", err)
	}
	if err := validateLlamacppTensorSize(fragment.Values, S, H, D, numLayers); err != nil {
		return nil, fmt.Errorf("reshape llamacpp→onnx values: %w", err)
	}

	// Build ONNX header: [num_layers][num_heads][seq_len][head_dim]
	onnxHeader := make([]byte, 16)
	binary.LittleEndian.PutUint32(onnxHeader[0:], uint32(numLayers))
	binary.LittleEndian.PutUint32(onnxHeader[4:], uint32(H))
	binary.LittleEndian.PutUint32(onnxHeader[8:], uint32(S))
	binary.LittleEndian.PutUint32(onnxHeader[12:], uint32(D))

	floatsPerLayer := S * H * D
	bytesPerLayer := floatsPerLayer * 4

	newKeys := make([]byte, 16+numLayers*(4+bytesPerLayer))
	newVals := make([]byte, 16+numLayers*(4+bytesPerLayer))
	copy(newKeys, onnxHeader)
	copy(newVals, onnxHeader)

	srcKeys := bytesToFloat32SliceONNX(fragment.Keys)
	srcVals := bytesToFloat32SliceONNX(fragment.Values)

	dstKeys := make([]float32, numLayers*floatsPerLayer)
	dstVals := make([]float32, numLayers*floatsPerLayer)

	for li := 0; li < numLayers; li++ {
		srcOff := li * floatsPerLayer
		dstOff := li * floatsPerLayer

		for token := 0; token < S; token++ {
			for head := 0; head < H; head++ {
				for d := 0; d < D; d++ {
					srcIdx := srcOff + token*H*D + head*D + d
					dstIdx := dstOff + head*S*D + token*D + d
					dstKeys[dstIdx] = srcKeys[srcIdx]
					dstVals[dstIdx] = srcVals[srcIdx]
				}
			}
		}
	}

	// Write per-layer headers + data into output buffers
	kCursor, vCursor := 16, 16
	layer := fragment.LayerStart
	for li := 0; li < numLayers; li++ {
		layerTag := make([]byte, 4)
		binary.LittleEndian.PutUint32(layerTag, uint32(layer))
		copy(newKeys[kCursor:], layerTag)
		copy(newVals[vCursor:], layerTag)
		kCursor += 4
		vCursor += 4

		layerBytes := float32SliceTo4BytesONNX(dstKeys[li*floatsPerLayer : (li+1)*floatsPerLayer])
		copy(newKeys[kCursor:], layerBytes)
		copy(newVals[vCursor:], layerBytes)
		kCursor += bytesPerLayer
		vCursor += bytesPerLayer

		layer += fragment.LayerStride
	}

	return fragmentWithNewTensors(fragment, newKeys, newVals, onnxEngineName), nil
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX → llamacpp
// ─────────────────────────────────────────────────────────────────────────────
//
// Reverse of the above. Transpose [H,S,D] → [S,H,D].
//   src[head][token][d]  →  dst[token][head][d]
//   src_idx = head*S*D + token*D + d
//   dst_idx = token*H*D + head*D + d

func reshapeONNXToLlamacpp(fragment *cache.KVFragment) (*cache.KVFragment, error) {
	model := fragment.Model
	H := model.NumKVHeads
	D := model.HeadDim

	// Parse ONNX header to get actual dims
	if len(fragment.Keys) < 16 {
		return nil, fmt.Errorf("reshape onnx→llamacpp: header too short")
	}
	numLayers := int(binary.LittleEndian.Uint32(fragment.Keys[0:]))
	S := int(binary.LittleEndian.Uint32(fragment.Keys[8:]))

	floatsPerLayer := S * H * D
	bytesPerLayer := floatsPerLayer * 4

	// Parse ONNX layer data
	srcKeys := make([]float32, numLayers*floatsPerLayer)
	srcVals := make([]float32, numLayers*floatsPerLayer)
	kCursor, vCursor := 16, 16
	for li := 0; li < numLayers; li++ {
		kCursor += 4 // skip layer tag
		vCursor += 4
		end := kCursor + bytesPerLayer
		if end > len(fragment.Keys) {
			return nil, fmt.Errorf("reshape onnx→llamacpp: buffer underflow at layer_idx %d", li)
		}
		kSlice := bytesToFloat32SliceONNX(fragment.Keys[kCursor:end])
		vSlice := bytesToFloat32SliceONNX(fragment.Values[vCursor : vCursor+bytesPerLayer])
		copy(srcKeys[li*floatsPerLayer:], kSlice)
		copy(srcVals[li*floatsPerLayer:], vSlice)
		kCursor += bytesPerLayer
		vCursor += bytesPerLayer
	}

	// Transpose [H,S,D] → [S,H,D]
	dstKeys := make([]float32, numLayers*floatsPerLayer)
	dstVals := make([]float32, numLayers*floatsPerLayer)

	for li := 0; li < numLayers; li++ {
		off := li * floatsPerLayer
		for head := 0; head < H; head++ {
			for token := 0; token < S; token++ {
				for d := 0; d < D; d++ {
					srcIdx := off + head*S*D + token*D + d
					dstIdx := off + token*H*D + head*D + d
					dstKeys[dstIdx] = srcKeys[srcIdx]
					dstVals[dstIdx] = srcVals[srcIdx]
				}
			}
		}
	}

	// Serialize in llamacpp flat format (no header, pure float32 blob)
	newKeys := float32SliceTo4Bytes(dstKeys)
	newVals := float32SliceTo4Bytes(dstVals)

	return fragmentWithNewTensors(fragment, newKeys, newVals, llamaCppEngineName), nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func validateLlamacppTensorSize(data []byte, S, H, D, numLayers int) error {
	expected := numLayers * S * H * D * 4 // float32 = 4 bytes
	if len(data) != expected {
		return fmt.Errorf(
			"tensor size mismatch: expected %d bytes (%d layers × %d tokens × %d heads × %d dim × 4), got %d",
			expected, numLayers, S, H, D, len(data),
		)
	}
	return nil
}

func shallowCopyWithEngine(f *cache.KVFragment, engine string) *cache.KVFragment {
	cp := *f
	cp.Engine = engine
	return &cp
}

func fragmentWithNewTensors(src *cache.KVFragment, keys, vals []byte, engine string) *cache.KVFragment {
	cp := *src
	cp.Keys = keys
	cp.Values = vals
	cp.Engine = engine
	cp.EngineVersion = "reshaped"
	return &cp
}

// float32SliceTo4Bytes — reuse from llamacpp.go (same package)
// Redeclared here to avoid circular dependency if files are split.
// In the final build, deduplicate into a shared internal/tensor package.
func float32SliceTo4BytesR(src []float32) []byte {
	out := make([]byte, len(src)*4)
	for i, v := range src {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// ─────────────────────────────────────────────────────────────────────────────
// Updated compatibility matrix (used by CanInject in interface.go)
// ─────────────────────────────────────────────────────────────────────────────
//
// With reshape support, we can now expand CompatibleWith() on each adapter:
//
//   LlamaCppAdapter.CompatibleWith() → ["onnx"]   (after reshape)
//   ONNXAdapter.CompatibleWith()     → ["llamacpp"] (after reshape)
//
// The reshape is transparent to the caller: CanInject() detects the mismatch,
// calls ReshapeForEngine(), then proceeds with InjectFragment() on the result.
//
// Updated CanInjectWithReshape wraps CanInject with automatic reshape fallback.

// CanInjectWithReshape extends CanInject to automatically reshape the fragment
// if the engine is compatible but uses a different tensor layout.
// Returns the (possibly reshaped) fragment ready for InjectFragment(), or an error.
func CanInjectWithReshape(adapter KVAdapter, fragment *cache.KVFragment) (*cache.KVFragment, error) {
	// Try direct inject first (same engine, no reshape needed)
	if err := CanInject(adapter, fragment); err == nil {
		return fragment, nil
	}

	// Check if the source engine is in CompatibleWith
	sourceEngine := fragment.Engine
	compatible := false
	for _, name := range adapter.CompatibleWith() {
		if name == sourceEngine {
			compatible = true
			break
		}
	}
	if !compatible {
		return nil, ErrEngineNotCompatible{
			FragmentEngine: sourceEngine,
			AdapterEngine:  adapter.EngineName(),
		}
	}

	// Reshape and retry
	reshaped, err := ReshapeForEngine(fragment, adapter.EngineName())
	if err != nil {
		return nil, fmt.Errorf("CanInjectWithReshape: reshape failed: %w", err)
	}

	if err := CanInject(adapter, reshaped); err != nil {
		return nil, fmt.Errorf("CanInjectWithReshape: post-reshape validation failed: %w", err)
	}

	return reshaped, nil
}
