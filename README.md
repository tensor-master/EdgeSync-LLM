# EdgeCache — KV Fragment Engine for Local LLMs

A **engine-agnostic KV cache fragment system** for on-device LLM inference.
Designed for ARM64 Android (Cortex-A55/A78), portable to any platform running
llama.cpp, MLC-LLM, or ONNX Runtime.

---

## What this is

A **reusable KV cache layer** that sits between the application and the LLM
engine. Instead of re-running the full prefill on every request, it stores
slices of the attention KV tensors (Keys and Values), retrieves them via
approximate nearest-neighbor search (HNSW), and injects them directly into
the engine's KV cache — skipping the most expensive part of inference.

This is not a "semantic cache" that stores response strings. It stores the
**actual transformer KV tensors**, identified by token range and layer range,
and reconstructs them at request time.

---

## Architecture

```
              [ PROMPT ]
                  │
                  ▼
         [ Embedding Model ]        MiniLM-L6-v2 (384-dim, ~8ms CPU)
                  │
                  ▼
           [ HNSW Index ]           Pure Go, M=16, efSearch=50
                  │
          ┌───────┴───────────────────────────┐
          │                                   │
    sim ≥ 0.92                          0.75 ≤ sim < 0.92       sim < 0.75
          │                                   │                      │
   ┌──────▼──────┐                  ┌─────────▼──────┐      ┌───────▼────────┐
   │ EXACT HIT   │                  │  PARTIAL HIT   │      │     MISS       │
   │             │                  │                │      │                │
   │ Inject KV   │                  │ Inject prefix  │      │ Full prefill   │
   │ fragment    │                  │ Generate delta │      │ Extract frag.  │
   │ ~8ms TTFT   │                  │ ~280ms TTFT    │      │ Store in HNSW  │
   └─────────────┘                  └────────────────┘      └────────────────┘
          │                                   │                      │
          └───────────────────────────────────┴──────────────────────┘
                                              │
                                     [ KVAdapter Layer ]
                                              │
                         ┌────────────────────┼─────────────────────┐
                         ▼                    ▼                     ▼
                  [ llamacpp ]           [ mlc-llm ]         [ onnx runtime ]
                 (GGML tensor API)    (TVM paged KV)      (past_key_values)
```

---

## File Structure

```
├── cache/
│   ├── fragment.go          ← KVFragment: formal definition of a cache unit
│   │                          (dimensions, TTL, eviction policy, storage key)
│   ├── differential.go      ← DifferentialEngine: EXACT / PARTIAL / MISS router
│   └── schema.go            ← SQLite WAL schema for fragment metadata
│
├── adapter/
│   ├── interface.go         ← KVAdapter: engine-agnostic contract
│   │                          (ExtractFragment / InjectFragment / Generate)
│   ├── llamacpp.go          ← llama.cpp adapter (GGML tensor API, CGO)
│   ├── mlc.go               ← MLC-LLM adapter (TVM paged KV, mlc4j)
│   └── onnx.go              ← ONNX Runtime adapter (past_key_values)
│
├── core/
│   ├── hnsw.go              ← Pure Go HNSW index (M=16, efSearch=50)
│   └── cosine_neon.c        ← ARM NEON fp16 cosine similarity
│
├── sdk/android/
│   └── EdgeSyncLLM.kt       ← Kotlin JNI bridge (suspend coroutines)
│
├── monitor/
│   └── energy_android.go    ← Android /sys/class/power_supply/ profiler
│
├── prefetch/
│   └── predictor.go         ← N-gram prefetch predictor (top-3 candidates)
│
└── benchmark/
    └── runner.go            ← Falsifiable benchmark: 3 modes × 1000 requests
```

---

## The KVFragment

The **atomic unit** of the cache. Formally defined in `cache/fragment.go`.

| Field | Type | Meaning |
|---|---|---|
| `TokenStart / TokenEnd` | int | Token range covered `[start, end)` |
| `LayerStart / LayerEnd` | int | Transformer layers captured |
| `LayerStride` | int | Sampling interval (2 = every other layer) |
| `Keys / Values` | `[]byte` | Raw attention tensors (engine-serialized) |
| `TokenIDs` | `[]int32` | Input tokens → used to verify prefix |
| `ContentHash` | string | SHA-256 of TokenIDs (not tensors) |
| `EmbeddingVector` | `[]float32` | 384-dim semantic vector for HNSW lookup |
| `ExpiresAt` | time.Time | TTL: 30 min (session) → 7 days (promoted) |
| `HitCount` | int | Auto-promotes at hit ≥ 5 |
| `Engine` | string | "llamacpp" / "mlc" / "onnx" |

**Invariants enforced at construction:**

- `TokenSpan ∈ [64, 2048]` tokens
- `LayerEnd ≤ model.NumLayers`
- `len(TokenIDs) == TokenSpan`
- `len(Keys) > 0 && len(Values) > 0`
- `LayerStride ≥ 1`

---

## The KVAdapter Interface

Defined in `adapter/interface.go`. Any engine implements 6 methods:

```go
ExtractFragment(ctx, tokenIDs, layerStart, layerEnd, layerStride, embedding)
    → *KVFragment, error

InjectFragment(ctx, fragment)
    → error

Generate(ctx, prompt, startTokenPos, maxTokens)
    → text string, tokensGenerated int, error

Tokenize(ctx, text)
    → []int32, error

ClearKVCache(ctx)
    → error

Close()
    → error
```

Cross-engine reuse: engine B can inject a fragment from engine A if and only if
B lists A in `CompatibleWith()`. Current compatibility matrix:

| Producer → Consumer | llamacpp | mlc | onnx |
|---|:---:|:---:|:---:|
| **llamacpp** | ✓ | — | — |
| **mlc** | — | ✓ | — |
| **onnx** | — | — | ✓ |

Cross-engine reuse (e.g. llamacpp → onnx) requires a KV tensor reshape adapter
(transpose `[seq, heads, dim]` → `[heads, seq, dim]`). Not implemented yet.

---

## Benchmark

The benchmark in `benchmark/runner.go` compares 3 modes over 1000 requests
drawn from 8 semantic prompt clusters (64 unique prompts + 4 variants each).

**Timing model** (not ad-hoc random ranges — derived from Cortex-A55 measurements):

| Constant | Value | Source |
|---|---|---|
| Prefill | 6.8 ms/token | llama.cpp bench, Snapdragon 685 |
| Generate | 18.4 ms/token | same |
| HNSW search | 3.2 ms | N=1000, efSearch=50 |
| Fragment inject | 0.029 ms/MB | LPDDR4X bandwidth |
| Fragment size | ~6 MB | 128 tokens, 12 layers, Q4_K_M |

**Expected results:**

| Mode | Avg TTFT | Hit rate | Mem BW | Energy |
|---|---|---|---|---|
| Baseline (no cache) | ~1800 ms | 0% | 100% | 253 mAh |
| Naive string cache | ~1600 ms | ~12% | ~88% | 222 mAh |
| **Fragment cache** | **~350 ms** | **~70%** | **~35%** | **88 mAh** |

Run:
```bash
go run ./benchmark/

# Verbose per-query output:
EDGE_VERBOSE=1 go run ./benchmark/
```

---

## Building

```bash
# Host build (benchmark only, no CGO):
go run ./benchmark/

# Android ARM64 (with llama.cpp CGO):
export CGO_CFLAGS="-I/path/to/llama.cpp"
export CGO_LDFLAGS="-L/path/to/llama.cpp/build -lllama -lm"
CGO_ENABLED=1 CC=aarch64-linux-gnu-gcc GOOS=linux GOARCH=arm64 \
    go build -o edgecache ./...

# NEON cosine module (ARM64 only):
aarch64-linux-gnu-gcc -O3 -march=armv8.2-a+fp16 \
    -c core/cosine_neon.c -o core/cosine_neon.o
```

---

## What is NOT implemented yet

- [ ] Cross-engine KV tensor reshape (llamacpp ↔ onnx)
- [ ] Fragment compaction (merge adjacent fragments for the same prefix)
- [ ] Persistent fragment store (LevelDB or SQLite blob storage for `Keys/Values`)
- [ ] Real embedding model integration (currently: simulated in benchmark)
- [ ] Android JNI bridge update for `adapter/` package (pending Kotlin bindings)

---

## License

## ⚠️ Commercial & Licensing Notice

**EdgeSync-LLM** is published under the **Business Source License 1.1 (BUSL-1.1)**.
* **Non-Commercial & Evaluation:** 100% Free to use, modify, and test.
* **Commercial Production Use:** Strictly prohibited for production deployment (mobile apps, SaaS, embedded hardware) without a commercial license.

On **July 1, 2029**, this version of the software will automatically transition to the **AGPL-3.0** license.

*To obtain a commercial production license, enterprise support, or custom hardware tuning (ARM NEON/NPU), contact:* **[kechaouwajdi@gmail.com]**
