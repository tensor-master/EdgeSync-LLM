// Package adapter defines the engine-agnostic interface that any local LLM
// backend must implement to participate in the EdgeCache KV fragment system.
//
// CONTRACT
// ────────
// The cache engine (differential.go, HNSW index) operates exclusively on
// KVFragment structs. It never calls engine-specific APIs directly.
// Concrete backends (llama.cpp, MLC-LLM, ONNX Runtime) register themselves
// by implementing KVAdapter and calling Register().
//
// REUSE ACROSS ENGINES
// ────────────────────
// A fragment produced by engine A can be injected into engine B if and only if:
//   1. Both engines use the same model (ModelID must match exactly).
//   2. Engine B implements Deserialize() for engine A's serialization format.
//      This is declared via CompatibleWith() — engine B lists the engine names
//      it can deserialize from.
//
// If cross-engine reuse is not possible, the cache falls back to MISS for that
// engine, which is always safe.
package adapter

import (
	"context"
	"fmt"
	"sync"

	"react-example/cache"
)

// ─────────────────────────────────────────────────────────────────────────────
// Core interface
// ─────────────────────────────────────────────────────────────────────────────

// KVAdapter is the contract every LLM engine must satisfy to participate
// in the EdgeCache fragment system.
//
// Implementations: LlamaCppAdapter, MLCAdapter, ONNXAdapter.
// See adapter/llamacpp.go, adapter/mlc.go, adapter/onnx.go.
type KVAdapter interface {
	// ── Identity ──────────────────────────────────────────────────────────────

	// EngineName returns a stable lowercase identifier for this engine.
	// Examples: "llamacpp", "mlc", "onnx".
	// This string is stored in KVFragment.Engine and must be stable across versions.
	EngineName() string

	// EngineVersion returns the semver of this engine build.
	// Used to detect tensor format changes between major versions.
	EngineVersion() string

	// ModelID returns the exact model configuration this adapter is loaded with.
	// The cache uses this to reject fragments from incompatible models.
	ModelID() cache.ModelID

	// CompatibleWith returns a list of engine names whose serialization format
	// this adapter can deserialize. An empty list means no cross-engine reuse.
	// Example: an ONNXRuntime adapter might return ["llamacpp"] if it can import
	// GGML-serialized KV tensors via a custom converter.
	CompatibleWith() []string

	// ── Fragment operations ───────────────────────────────────────────────────

	// ExtractFragment runs a prefill pass on tokenIDs and extracts the resulting
	// KV tensors as a KVFragment.
	//
	// tokenIDs: the full prefix token sequence to prefill.
	// layerStart, layerEnd, layerStride: which layers to capture.
	//   Pass layerStart=0, layerEnd=model.NumLayers, layerStride=FragmentLayerStride
	//   for a standard sparse capture.
	//
	// The adapter is responsible for serializing the raw tensors into
	// fragment.Keys and fragment.Values in its own binary format.
	// The cache engine treats these as opaque blobs.
	//
	// Returns a fully populated KVFragment ready to be stored.
	ExtractFragment(
		ctx context.Context,
		tokenIDs []int32,
		layerStart, layerEnd, layerStride int,
		embedding []float32,
	) (*cache.KVFragment, error)

	// InjectFragment loads the KV tensors from fragment into the engine's
	// active KV cache, starting at the token position fragment.TokenStart.
	//
	// After a successful inject, the engine can continue generation from
	// token position fragment.TokenEnd WITHOUT reprocessing the prefix —
	// that is the entire point of the cache.
	//
	// The adapter must validate:
	//   - fragment.Model matches its own ModelID()
	//   - fragment.Engine is either its own EngineName() or listed in CompatibleWith()
	//   - tensor shapes are consistent with fragment.NumLayersCovered() and fragment.TokenSpan()
	//
	// Returns ErrIncompatibleModel or ErrTensorShapeMismatch on validation failure.
	InjectFragment(ctx context.Context, fragment *cache.KVFragment) error

	// Generate runs token generation starting from startTokenPos, optionally
	// using a previously injected fragment as the KV cache prefix.
	//
	// prompt: the full prompt text (used for tokenization).
	// startTokenPos: position in the context from which generation begins.
	//   If a fragment was injected up to position N, pass startTokenPos=N.
	//   Pass 0 for a full cold generation (no cache reuse).
	// maxTokens: maximum number of tokens to generate.
	//
	// Returns the generated text and the total number of tokens generated.
	Generate(
		ctx context.Context,
		prompt string,
		startTokenPos int,
		maxTokens int,
	) (text string, tokensGenerated int, err error)

	// Tokenize converts a text prompt into engine-specific token IDs.
	// Must be called before ExtractFragment to produce the tokenIDs argument.
	Tokenize(ctx context.Context, text string) ([]int32, error)

	// ── Lifecycle ─────────────────────────────────────────────────────────────

	// ClearKVCache discards the engine's active KV cache state.
	// Must be called between unrelated requests to prevent context bleed.
	ClearKVCache(ctx context.Context) error

	// Close releases all engine resources (model weights, GPU buffers, etc).
	Close() error
}

// ─────────────────────────────────────────────────────────────────────────────
// Sentinel errors
// ─────────────────────────────────────────────────────────────────────────────

// ErrIncompatibleModel is returned by InjectFragment when the fragment's
// ModelID does not match the adapter's loaded model.
type ErrIncompatibleModel struct {
	FragmentModel cache.ModelID
	AdapterModel  cache.ModelID
}

func (e ErrIncompatibleModel) Error() string {
	return fmt.Sprintf(
		"fragment model %q is incompatible with adapter model %q",
		e.FragmentModel.String(),
		e.AdapterModel.String(),
	)
}

// ErrTensorShapeMismatch is returned when the fragment's tensor dimensions
// are inconsistent with the model configuration.
type ErrTensorShapeMismatch struct {
	Expected string
	Got      string
}

func (e ErrTensorShapeMismatch) Error() string {
	return fmt.Sprintf("tensor shape mismatch: expected %s, got %s", e.Expected, e.Got)
}

// ErrEngineNotCompatible is returned when InjectFragment is called with a
// fragment from an engine that this adapter cannot deserialize.
type ErrEngineNotCompatible struct {
	FragmentEngine string
	AdapterEngine  string
}

func (e ErrEngineNotCompatible) Error() string {
	return fmt.Sprintf(
		"adapter %q cannot deserialize fragments from engine %q",
		e.AdapterEngine, e.FragmentEngine,
	)
}

// ─────────────────────────────────────────────────────────────────────────────
// Registry — runtime lookup of available adapters
// ─────────────────────────────────────────────────────────────────────────────

var (
	registryMu sync.RWMutex
	registry   = make(map[string]KVAdapter)
)

// Register makes an adapter available under its EngineName().
// Call from each adapter's init() function or at application startup.
// Panics on duplicate registration to catch misconfiguration early.
func Register(a KVAdapter) {
	registryMu.Lock()
	defer registryMu.Unlock()
	name := a.EngineName()
	if _, exists := registry[name]; exists {
		panic(fmt.Sprintf("adapter: duplicate registration for engine %q", name))
	}
	registry[name] = a
}

// Get returns the registered adapter for engineName, or an error if not found.
func Get(engineName string) (KVAdapter, error) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	a, ok := registry[engineName]
	if !ok {
		return nil, fmt.Errorf("adapter: no adapter registered for engine %q", engineName)
	}
	return a, nil
}

// List returns the names of all registered adapters.
func List() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	return names
}

// ─────────────────────────────────────────────────────────────────────────────
// Compatibility check helper (used by the cache engine)
// ─────────────────────────────────────────────────────────────────────────────

// CanInject returns nil if adapter can safely inject fragment, or an
// appropriate sentinel error explaining why not.
// The cache engine calls this before InjectFragment to short-circuit mismatches.
func CanInject(adapter KVAdapter, fragment *cache.KVFragment) error {
	// Model must match exactly.
	adapterModel := adapter.ModelID()
	if fragment.Model.Hash() != adapterModel.Hash() {
		return ErrIncompatibleModel{
			FragmentModel: fragment.Model,
			AdapterModel:  adapterModel,
		}
	}

	// Engine must be self or declared compatible.
	if fragment.Engine != adapter.EngineName() {
		compatible := false
		for _, name := range adapter.CompatibleWith() {
			if name == fragment.Engine {
				compatible = true
				break
			}
		}
		if !compatible {
			return ErrEngineNotCompatible{
				FragmentEngine: fragment.Engine,
				AdapterEngine:  adapter.EngineName(),
			}
		}
	}

	// Fragment must not be expired.
	if fragment.IsExpired() {
		return fmt.Errorf("fragment %q is expired (TTL exceeded at %s)", fragment.ID, fragment.ExpiresAt)
	}

	return nil
}
