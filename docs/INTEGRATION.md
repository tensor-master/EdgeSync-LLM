# EdgeCache — Integration Guide

## Table of Contents

1. [Concepts in 5 minutes](#concepts)
2. [Quickstart — llama.cpp](#quickstart-llamacpp)
3. [Quickstart — ONNX Runtime](#quickstart-onnx)
4. [Quickstart — MLC-LLM (Android)](#quickstart-mlc-android)
5. [Cross-engine fragment reuse](#cross-engine)
6. [Persistent store setup](#persistent-store)
7. [Tuning parameters](#tuning)
8. [Troubleshooting](#troubleshooting)

---

## Concepts in 5 minutes {#concepts}

### What a KV cache is

Every transformer forward pass computes attention Keys and Values for each
input token. On the second request with the same prefix, these tensors are
identical — recomputing them wastes time and battery.

EdgeCache stores these tensors as **fragments**, retrieves them with a fast
approximate nearest-neighbor search (HNSW), and injects them directly into
the engine's KV cache — skipping the prefill for tokens already computed.

### Three outcomes

```
Request arrives
      │
      ▼
HNSW lookup (~3ms)
      │
      ├── similarity ≥ 0.92 ──► EXACT HIT
      │                          inject fragment (~0.2ms)
      │                          generate suffix only (~8ms total TTFT)
      │
      ├── 0.75 ≤ sim < 0.92 ──► PARTIAL HIT
      │                          inject prefix fragment
      │                          generate delta only (~280ms total TTFT)
      │
      └── similarity < 0.75 ──► MISS
                                 full prefill + generate (~1800ms)
                                 extract + store fragment for next time
```

### What EdgeCache does NOT do

- It does not store response strings (that is a naive semantic cache).
- It does not call any LLM API. It is a layer between your app and the engine.
- It does not require a GPU. Designed for CPU-only ARM64 (Cortex-A55/A78).

---

## Quickstart — llama.cpp {#quickstart-llamacpp}

### Prerequisites

- llama.cpp built from source (b3117 or later)
- Go 1.21+, CGO enabled
- A GGUF model file (e.g. `qwen2.5-0.5b-q4_k_m.gguf`)

### Step 1 — Define your ModelID

```go
model := cache.ModelID{
    Architecture:  "qwen",
    Name:          "Qwen2.5-0.5B",
    Quantization:  "Q4_K_M",
    ContextLength: 4096,
    HeadDim:       64,
    NumKVHeads:    8,
    NumLayers:     24,
}
```

**Why this matters:** a fragment is invalid if any of these fields differ.
If you load the same weights with a different context length, the RoPE
parameters change and the KV tensors are incompatible.

### Step 2 — Create the adapter

```go
import "react-example/adapter"

// llama_ctx is your *llama_context from llama_new_context_with_params()
llamaAdapter := adapter.NewLlamaCppAdapter(unsafe.Pointer(llama_ctx), model)
adapter.Register(llamaAdapter)
```

### Step 3 — Extract a fragment

```go
ctx := context.Background()

// Tokenize your prompt prefix
tokenIDs, err := llamaAdapter.Tokenize(ctx, "explain write-ahead logging in sqlite")
if err != nil { ... }

// Generate a semantic embedding (use any 384-dim embedding model)
embedding := myEmbeddingModel.Encode("explain write-ahead logging in sqlite")

// Extract KV tensors for layers 0-24, stride 2 (every other layer)
fragment, err := llamaAdapter.ExtractFragment(
    ctx, tokenIDs,
    0, model.NumLayers, cache.FragmentLayerStride,
    embedding,
)
if err != nil { ... }

fmt.Printf("Fragment ID: %s, size: %d MB, TTL: %s\n",
    fragment.ID,
    fragment.SizeBytes()/1024/1024,
    fragment.ExpiresAt.Sub(time.Now()).Round(time.Minute),
)
```

### Step 4 — Inject and generate

```go
// On a subsequent request with a similar prompt:
err = llamaAdapter.InjectFragment(ctx, fragment)
if err != nil { ... }

// Generate from token position fragment.TokenEnd (skip recomputing the prefix)
text, count, err := llamaAdapter.Generate(ctx, fullPrompt, fragment.TokenEnd, 200)
fmt.Printf("Generated %d tokens: %s\n", count, text)
```

### Step 5 — Use the full cache pipeline

```go
// Instead of managing fragments manually, use the differential engine:
import "react-example/cache"
import "react-example/core"

hnsw := core.NewHNSW(16, 50)
store, err := cache.NewFragmentStore(
    "/data/edgecache/fragments.db",
    "/data/edgecache/blobs",
)
if err != nil { ... }
defer store.Close()

// The DifferentialEngine wraps the HNSW lookup + inject + generate pipeline.
// See cache/differential.go for the full API.
```

---

## Quickstart — ONNX Runtime {#quickstart-onnx}

ONNX Runtime uses the HuggingFace `past_key_values` convention. Models must
be exported with `use_cache=True`.

### Export your model

```python
# Python — export Qwen2.5-0.5B to ONNX with KV cache enabled
from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="Qwen/Qwen2.5-0.5B",
    output="./qwen2.5-0.5b-onnx",
    task="text-generation-with-past",  # enables past_key_values
    opset=17,
)
```

### Create the adapter

```go
// Implement the ONNXSession interface against github.com/yalue/onnxruntime_go
// See docs/examples/onnx_session.go for a complete implementation.

session := NewORTSession("./qwen2.5-0.5b-onnx/model.onnx", model)
onnxAdapter := adapter.NewONNXAdapter(session, model)
adapter.Register(onnxAdapter)
```

### Key difference from llama.cpp

ONNX Runtime is **stateless**: there is no mutable in-process KV cache buffer.
Instead, `InjectFragment()` stores the tensors in memory and `Generate()`
passes them as `past_key_values` inputs on the next ORT session run.

This means `InjectFragment()` and `Generate()` must be called in the same
request lifecycle on the same `ONNXAdapter` instance.

```go
// CORRECT — same adapter instance, same request
err = onnxAdapter.InjectFragment(ctx, fragment)
text, _, err = onnxAdapter.Generate(ctx, prompt, fragment.TokenEnd, 200)

// WRONG — different goroutine, different adapter instance
go func() {
    err = onnxAdapter2.Generate(...)  // fragment state is lost
}()
```

---

## Quickstart — MLC-LLM (Android) {#quickstart-mlc-android}

MLC-LLM on Android uses a PagedAttention KV cache. Fragments are stored as
page arrays, not contiguous tensors.

### Kotlin setup (mlc4j)

```kotlin
import react_example.adapter.MLCAdapter
import react_example.cache.ModelID

val model = ModelID(
    architecture = "qwen",
    name = "Qwen2.5-0.5B",
    quantization = "Q4_K_M",
    contextLength = 4096,
    headDim = 64,
    numKVHeads = 8,
    numLayers = 24
)

// EdgeSyncLLM is the JNI bridge in sdk/android/EdgeSyncLLM.kt
val edgeSync = EdgeSyncLLM(context, model)

// Register the adapter (calls into Go via JNI)
edgeSync.registerMLCAdapter()
```

### Extract and inject from Kotlin

```kotlin
// In a coroutine (EdgeSyncLLM uses suspend functions):
val tokenIds = edgeSync.tokenize(prompt)
val embedding = embeddingModel.encode(prompt)

val fragment = edgeSync.extractFragment(
    tokenIds = tokenIds,
    layerStart = 0,
    layerEnd = model.numLayers,
    layerStride = 2,
    embedding = embedding
)

// On next request:
edgeSync.injectFragment(fragment)
val response = edgeSync.generate(fullPrompt, startPos = fragment.tokenEnd, maxTokens = 200)
```

---

## Cross-engine fragment reuse {#cross-engine}

A fragment produced by llama.cpp can be injected into ONNX Runtime (and vice
versa) using the reshape adapter. This enables hybrid pipelines where, for
example, a server runs llama.cpp and an Android device runs ONNX Runtime —
and they share a fragment store.

```go
import "react-example/adapter"

// Fragment was produced by llama.cpp
llamaFragment, _ := llamaAdapter.ExtractFragment(...)

// Reshape for ONNX Runtime injection
onnxFragment, err := adapter.ReshapeForEngine(llamaFragment, "onnx")
if err != nil {
    // Unsupported conversion — falls back to full prefill
    log.Printf("reshape not available: %v", err)
}

// Inject into ONNX adapter
err = onnxAdapter.InjectFragment(ctx, onnxFragment)
```

### Current compatibility matrix

| From \ To | llamacpp | mlc | onnx |
|:---:|:---:|:---:|:---:|
| **llamacpp** | ✓ direct | — | ✓ reshape |
| **mlc** | — | ✓ direct | — |
| **onnx** | ✓ reshape | — | ✓ direct |

`—` = not yet implemented (MLC paged layout requires page splitting logic).

### Performance cost of reshape

The transpose operation (swap seq_len ↔ num_heads axes) costs ~0.8ms for a
128-token fragment on Cortex-A55. This is paid once per cross-engine injection
and is negligible compared to the prefill cost it replaces (~870ms).

---

## Persistent store setup {#persistent-store}

By default, fragments live only in memory and are lost on process restart.
For cross-session reuse (system prompt fragments, FAQ prefixes), enable the
persistent store.

```go
store, err := cache.NewFragmentStore(
    dbPath:  "/data/edgecache/fragments.db",
    blobDir: "/data/edgecache/blobs/",
)
if err != nil {
    log.Fatalf("cannot open fragment store: %v", err)
}
defer store.Close()

// Store a fragment (persisted only when HitCount >= 5)
err = store.Store(fragment)

// Retrieve by ID
f, err := store.Get(fragmentID)

// Find fragments covering a token range
fragments, err := store.QueryByTokenRange(model.Hash(), 0, 256)

// Check store health
hot, persistent, err := store.Stats()
fmt.Printf("hot: %d fragments, persistent: %d fragments\n", hot, persistent)
```

### Storage layout on disk

```
/data/edgecache/
├── fragments.db          ← SQLite WAL database (metadata only, ~few KB per fragment)
├── fragments.db-wal      ← WAL journal (auto-managed by SQLite)
└── blobs/
    ├── a1b2c3d4.keys.bin ← raw float32 tensor (Keys), ~3-12 MB per fragment
    ├── a1b2c3d4.vals.bin ← raw float32 tensor (Values), same size
    ├── e5f6a7b8.keys.bin
    └── ...
```

### Android permissions

```xml
<!-- AndroidManifest.xml -->
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"
    android:maxSdkVersion="28" />
```

For Android 10+, use app-specific internal storage (`context.filesDir`) —
no permission required:

```kotlin
val dbPath = File(context.filesDir, "edgecache/fragments.db").absolutePath
val blobDir = File(context.filesDir, "edgecache/blobs").absolutePath
```

---

## Tuning parameters {#tuning}

All tunable constants are in `cache/fragment.go`:

| Constant | Default | Effect |
|---|---|---|
| `FragmentGranularityTokens` | 64 | Minimum fragment size. Smaller = more granular reuse, more index overhead. |
| `FragmentMaxTokenSpan` | 2048 | Largest fragment. Larger = fewer fragments, less partial reuse. |
| `FragmentLayerStride` | 2 | Layer sampling. `1` = full quality, 2× storage. `2` = ~70% quality, 1× storage. |
| `DefaultTTLSession` | 30 min | Fragment lifetime for interactive sessions. |
| `DefaultTTLPersistent` | 7 days | Lifetime after promotion. |
| `SimilarityExact` | 0.92 | Threshold for exact hit. Lower = more hits, more incorrect reuse risk. |
| `SimilarityPartial` | 0.75 | Threshold for partial hit. |
| `HitThresholdPromote` | 5 | Hit count before a fragment is persisted to SQLite. |
| `HitThresholdEvict` | 512 | Max in-memory fragments before LRU eviction. |

### Recommended profiles

**Low-RAM device (≤3 GB, e.g. entry-level Android)**
```go
cache.FragmentLayerStride = 4      // cache every 4th layer
cache.HitThresholdEvict   = 128    // smaller hot pool
cache.FragmentMaxTokenSpan = 512   // smaller fragments
```

**High-RAM device (≥8 GB, e.g. flagship Android)**
```go
cache.FragmentLayerStride = 1      // full layer coverage
cache.HitThresholdEvict   = 1024
cache.FragmentMaxTokenSpan = 2048
```

---

## Troubleshooting {#troubleshooting}

**`ErrIncompatibleModel`: fragment model hash does not match adapter**

The fragment was produced with a different `ModelID`. Check that
`Architecture`, `Quantization`, `ContextLength`, `HeadDim`, `NumKVHeads`,
and `NumLayers` all match exactly between the producer and consumer.

**`ErrTensorShapeMismatch`: tensor size mismatch**

The fragment's `Keys`/`Values` byte length does not match what the adapter
expects given the model dimensions. This usually means `LayerStride` or
`NumLayersCovered()` differs between extraction and injection. Verify
`LayerStart`, `LayerEnd`, and `LayerStride` are identical.

**Fragment hit rate is unexpectedly low**

The embedding model is not capturing semantic similarity correctly.
Run `EDGE_VERBOSE=1 go run ./benchmark/` to see per-query similarity scores.
If most scores are below 0.60, your embedding model may not be well-suited
to your prompt domain. Consider fine-tuning a MiniLM model on your data.

**High memory usage on Android**

Each in-memory fragment uses ~6 MB (128 tokens, 12 layers, Q4_K_M).
512 fragments = ~3 GB RAM — too much for most devices. Reduce
`HitThresholdEvict` to 64-128 for entry-level devices.

**SQLite `database is locked` error**

The `busy_timeout=5000` pragma (5 seconds) should handle transient locks.
If you see this error persistently, ensure only one `FragmentStore` instance
is open per process. For multi-process scenarios, use WAL mode with
`_locking_mode=NORMAL` (the default).
