// Package adapter — ONNX Runtime backend implementation of KVAdapter.
//
// INTEGRATION NOTES FOR ONNX RUNTIME
// ─────────────────────────────────────
// ONNX Runtime (ORT) exposes past KV states via "past_key_values" inputs.
// Models exported with `use_cache=True` (HuggingFace style) accept:
//
//   Input:  input_ids, attention_mask, past_key_values.N.key, past_key_values.N.value
//   Output: logits, present.N.key, present.N.value
//
// The "present" outputs ARE the new KV cache state after one forward pass.
// To reuse a cached prefix:
//   1. Run first N tokens → collect present.N.key / present.N.value for all layers
//   2. On subsequent requests with the same prefix → pass as past_key_values
//   3. Only pass new tokens in input_ids
//
// This is the standard HuggingFace KV cache pattern, directly supported by ORT.
// No patching required.
//
// TENSOR FORMAT
// ──────────────
// ONNX KV tensors have shape: [batch, num_heads, seq_len, head_dim]
// This differs from llama.cpp's [seq_len, num_heads, head_dim] layout.
// The serialization here stores ONNX-native layout (no transposition).
// Cross-engine reuse from llamacpp → ONNX requires a transpose (not implemented here).
// CompatibleWith() returns [] to prevent silent cross-engine injection.
//
// Wire format (ONNX):
//   header:  [4B: num_layers] [4B: num_heads] [4B: seq_len] [4B: head_dim]
//   per layer: [keys: batch=1 × num_heads × seq_len × head_dim × float32]
//              [vals: same]
//
// GO BINDING
// ───────────
// Uses github.com/yalue/onnxruntime_go for CGO-free ORT bindings.
// Install: go get github.com/yalue/onnxruntime_go
//
// For Android: use ORT's Java/Kotlin SDK (com.microsoft.onnxruntime).
// The ONNXRuntime interface below maps to OrtSession.run() calls.
package adapter

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/bossandboss/EdgeSync-LLM/cache"
)

const (
	onnxEngineName    = "onnx"
	onnxEngineVersion = "1.18.0"
)

// ONNXAdapter implements KVAdapter for ONNX Runtime.
// Uses the HuggingFace past_key_values convention.
type ONNXAdapter struct {
	session ONNXSession
	modelID cache.ModelID

	// pendingPastKeys/pendingPastVals hold the last injected KV state for use
	// in the next Generate() call. Necessary because ONNX Runtime is
	// stateless — past_key_values must be passed as explicit inputs on every
	// call, unlike llama.cpp which keeps state inside its own context.
	//
	// Previously these three fields were missing entirely from this struct —
	// InjectFragment() and Generate() both referenced them, so the package
	// failed to compile as soon as anything else stopped masking the error
	// (see the removed placeholder line further down this file).
	pendingPastKeys [][]float32
	pendingPastVals [][]float32
	pendingTokenEnd int
}

// ONNXSession abstracts ONNX Runtime session calls.
// Implement against github.com/yalue/onnxruntime_go for desktop,
// or com.microsoft.onnxruntime for Android.
type ONNXSession interface {
	// RunPrefill executes a prefill pass and returns the present KV states.
	// Returns [num_layers]keys and [num_layers]values as flat float32 slices.
	// Shape of each: [1 × num_heads × seq_len × head_dim]
	RunPrefill(tokenIDs []int32) (keys [][]float32, values [][]float32, err error)

	// RunWithPast executes generation using past_key_values.
	// pastKeys, pastValues: [num_layers][]float32, each [1 × heads × past_len × dim]
	// newTokenIDs: only the NEW tokens (not the cached prefix)
	// Returns generated text, token count, and error.
	RunWithPast(
		pastKeys, pastValues [][]float32,
		newTokenIDs []int32,
		maxNewTokens int,
	) (string, int, error)

	// Tokenize converts text to ORT-compatible token IDs.
	Tokenize(text string) ([]int32, error)

	// Close releases ORT session resources.
	Close() error
}

// NewONNXAdapter creates an ONNXAdapter wrapping the given session.
func NewONNXAdapter(session ONNXSession, modelID cache.ModelID) *ONNXAdapter {
	return &ONNXAdapter{session: session, modelID: modelID}
}

// ── KVAdapter identity ────────────────────────────────────────────────────────

func (a *ONNXAdapter) EngineName() string    { return onnxEngineName }
func (a *ONNXAdapter) EngineVersion() string  { return onnxEngineVersion }
func (a *ONNXAdapter) ModelID() cache.ModelID { return a.modelID }

// CompatibleWith: ONNX cannot inject llamacpp or mlc fragments without transposition.
// Declare empty to be safe. Cross-engine support can be added with a reshape converter.
func (a *ONNXAdapter) CompatibleWith() []string { return []string{} }

// ── Fragment extraction ───────────────────────────────────────────────────────

// ExtractFragment runs a prefill pass and captures the present_key_values output.
//
// Wire format header (ONNX):
//   [4B: num_layers_captured] [4B: num_heads] [4B: seq_len] [4B: head_dim]
// Per captured layer:
//   [4B: layer_index]
//   keys:   num_heads × seq_len × head_dim × float32 (little-endian)
//   values: same
func (a *ONNXAdapter) ExtractFragment(
	ctx context.Context,
	tokenIDs []int32,
	layerStart, layerEnd, layerStride int,
	embedding []float32,
) (*cache.KVFragment, error) {
	allKeys, allVals, err := a.session.RunPrefill(tokenIDs)
	if err != nil {
		return nil, fmt.Errorf("onnx ExtractFragment: prefill failed: %w", err)
	}

	model := a.modelID
	seqLen := len(tokenIDs)

	var capturedLayers []int
	for l := layerStart; l < layerEnd; l += layerStride {
		capturedLayers = append(capturedLayers, l)
	}

	// Header: [num_layers_captured][num_heads][seq_len][head_dim]
	header := make([]byte, 16)
	binary.LittleEndian.PutUint32(header[0:], uint32(len(capturedLayers)))
	binary.LittleEndian.PutUint32(header[4:], uint32(model.NumKVHeads))
	binary.LittleEndian.PutUint32(header[8:], uint32(seqLen))
	binary.LittleEndian.PutUint32(header[12:], uint32(model.HeadDim))

	var keysOut, valsOut []byte
	keysOut = append(keysOut, header...)
	valsOut = append(valsOut, header...)

	floatsPerLayer := model.NumKVHeads * seqLen * model.HeadDim

	for _, layer := range capturedLayers {
		if layer >= len(allKeys) {
			return nil, fmt.Errorf("onnx ExtractFragment: layer %d out of range (%d layers returned)", layer, len(allKeys))
		}

		layerTag := make([]byte, 4)
		binary.LittleEndian.PutUint32(layerTag, uint32(layer))
		keysOut = append(keysOut, layerTag...)
		valsOut = append(valsOut, layerTag...)

		kSlice := allKeys[layer]
		vSlice := allVals[layer]
		if len(kSlice) < floatsPerLayer || len(vSlice) < floatsPerLayer {
			return nil, fmt.Errorf("onnx ExtractFragment: layer %d key/val size mismatch (got %d/%d, need %d)",
				layer, len(kSlice), len(vSlice), floatsPerLayer)
		}

		keysOut = append(keysOut, float32SliceTo4BytesONNX(kSlice[:floatsPerLayer])...)
		valsOut = append(valsOut, float32SliceTo4BytesONNX(vSlice[:floatsPerLayer])...)
	}

	fragmentID := generateFragmentID(tokenIDs, a.modelID)

	return cache.NewFragment(
		fragmentID,
		a.modelID,
		0, seqLen,
		layerStart, layerEnd, layerStride,
		keysOut, valsOut,
		tokenIDs,
		embedding,
		onnxEngineName,
		onnxEngineVersion,
		cache.DefaultTTLSession,
	)
}

// InjectFragment restores ONNX past_key_values from fragment and runs generation.
// Unlike llama.cpp (where inject modifies mutable KV cache in-place),
// ONNX fragments are stateless: we decode the tensors and pass them as inputs
// to RunWithPast() during the next Generate() call.
// The decoded tensors are stored in a.pendingPastKeys/Vals for immediate use.
func (a *ONNXAdapter) InjectFragment(ctx context.Context, fragment *cache.KVFragment) error {
	if err := CanInject(a, fragment); err != nil {
		return fmt.Errorf("onnx InjectFragment: %w", err)
	}

	keys := fragment.Keys
	vals := fragment.Values

	if len(keys) < 16 {
		return fmt.Errorf("onnx InjectFragment: header too short")
	}

	numLayers := int(binary.LittleEndian.Uint32(keys[0:]))
	numHeads := int(binary.LittleEndian.Uint32(keys[4:]))
	seqLen := int(binary.LittleEndian.Uint32(keys[8:]))
	headDim := int(binary.LittleEndian.Uint32(keys[12:]))

	floatsPerLayer := numHeads * seqLen * headDim
	bytesPerLayer := floatsPerLayer * 4

	a.pendingPastKeys = make([][]float32, numLayers)
	a.pendingPastVals = make([][]float32, numLayers)
	a.pendingTokenEnd = fragment.TokenEnd

	kCursor, vCursor := 16, 16
	for li := 0; li < numLayers; li++ {
		if kCursor+4 > len(keys) {
			return fmt.Errorf("onnx InjectFragment: truncated at layer_idx %d", li)
		}
		kCursor += 4 // skip layer_index tag
		vCursor += 4

		end := kCursor + bytesPerLayer
		if end > len(keys) {
			return fmt.Errorf("onnx InjectFragment: buffer underflow at layer_idx %d", li)
		}
		a.pendingPastKeys[li] = bytesToFloat32SliceONNX(keys[kCursor:end])
		a.pendingPastVals[li] = bytesToFloat32SliceONNX(vals[vCursor : vCursor+bytesPerLayer])
		kCursor += bytesPerLayer
		vCursor += bytesPerLayer
	}

	return nil
}

func (a *ONNXAdapter) Generate(
	ctx context.Context,
	prompt string,
	startTokenPos int,
	maxTokens int,
) (string, int, error) {
	if a.pendingPastKeys != nil {
		// Tokenize only the new suffix (tokens after the cached prefix)
		allTokenIDs, err := a.session.Tokenize(prompt)
		if err != nil {
			return "", 0, fmt.Errorf("onnx Generate: tokenize failed: %w", err)
		}
		newTokens := allTokenIDs[startTokenPos:]
		text, count, err := a.session.RunWithPast(a.pendingPastKeys, a.pendingPastVals, newTokens, maxTokens)
		a.pendingPastKeys = nil
		a.pendingPastVals = nil
		return text, count, err
	}

	// Cold generation: no cached prefix
	tokenIDs, err := a.session.Tokenize(prompt)
	if err != nil {
		return "", 0, fmt.Errorf("onnx Generate: tokenize failed: %w", err)
	}
	return a.session.RunWithPast(nil, nil, tokenIDs, maxTokens)
}

func (a *ONNXAdapter) Tokenize(_ context.Context, text string) ([]int32, error) {
	return a.session.Tokenize(text)
}

func (a *ONNXAdapter) ClearKVCache(_ context.Context) error {
	a.pendingPastKeys = nil
	a.pendingPastVals = nil
	a.pendingTokenEnd = 0
	return nil
}

func (a *ONNXAdapter) Close() error {
	return a.session.Close()
}

// ── ONNX serialization helpers ────────────────────────────────────────────────

// Separate functions to avoid naming collision with llamacpp.go helpers.
func float32SliceTo4BytesONNX(src []float32) []byte {
	out := make([]byte, len(src)*4)
	for i, v := range src {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func bytesToFloat32SliceONNX(src []byte) []float32 {
	n := len(src) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
	}
	return out
}
