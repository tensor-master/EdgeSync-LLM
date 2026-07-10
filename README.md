# EdgeSync-LLM — KV Cache Reuse for Local LLMs

Skip the prefill. When two prompts share a long prefix — a system preamble, a
reused RAG chunk, a document, a few-shot block — the second one should not pay
to recompute what the first already computed.

EdgeSync captures the engine's KV cache state after the prefix is prefilled,
stores it, and restores it on the next request that shares that prefix. The
model then decodes only the new tokens.

**Measured: 7.5× lower time-to-first-token on cache hits, with output identical
to the uncached path.** Details and caveats below — including what is *not* yet
proven.

Targets ARM64 Android; currently benchmarked on x86-64.

---

## Status

This project publishes what it can prove, and marks the rest.

| | |
|---|---|
| ✅ **Validated** | KV state reuse via llama.cpp's public state API. TTFT speedup measured, correctness verified (see below). Links against **vanilla llama.cpp** — no fork, no patch. |
| ⚠️ **Unvalidated** | Semantic (approximate) prefix matching. Fragment compaction. Persistent store. Android JNI bridge. |
| ⛔ **Not sound as designed** | `PARTIAL` hits (reusing a fragment across *similar* rather than *identical* prefixes). Cross-engine fragment reuse. Per-layer striding. See [Known limitations](#known-limitations). |
| ❌ **Not measured** | Anything on-device. All numbers here are x86-64. |

---

## How it works

```
                        [ PROMPT = PREFIX + SUFFIX ]
                                    │
                     ┌──────────────┴──────────────┐
                     │  lookup on the PREFIX only  │
                     └──────────────┬──────────────┘
                                    │
                  ┌─────────────────┴──────────────────┐
                  │                                    │
           prefix seen before                   prefix is new
                  │                                    │
        ┌─────────▼──────────┐              ┌──────────▼──────────┐
        │  RESTORE state     │              │  Full prefill       │
        │  llama_state_seq_  │              │  Capture state via  │
        │    set_data        │              │  llama_state_seq_   │
        │                    │              │    get_data         │
        │  Decode SUFFIX     │              │  Store fragment     │
        │  from pos = |prefix|              │                     │
        └─────────┬──────────┘              └──────────┬──────────┘
                  │                                    │
                  └────────────────┬───────────────────┘
                                   ▼
                            [ FIRST TOKEN ]
```

The fragment is scoped to the **shared prefix**, never to the whole prompt.
A fragment covering the full prompt would bake in the varying user turn; injecting
it for a different request would generate from a KV cache that does not
correspond to that request's tokens.

### Why the public state API, and not raw tensor surgery

The obvious implementation copies the K/V tensors out of the cache and writes
them back later. It does not work, and it fails *silently*.

llama.cpp tracks, per cell, a position and a sequence id, and builds its attention
mask from those. A raw tensor write moves the numbers but not the bookkeeping: the
cache still believes the sequence is empty, attention never sees the injected
cells, and the next decode reallocates over them.

The fragment becomes **inert** — fast, because the prefix is skipped, and wrong,
because the prefix is gone. Our first implementation did exactly this and reported
an 8.8× speedup. The correctness harness showed that 14 of 24 cache hits
reproduced, token for token, the output of a generation with *no prefix at all*.
That speedup was dropped context, not cache reuse.

`llama_state_seq_get_data` / `llama_state_seq_set_data` serialise cells **and**
metadata, for every layer. They are public, upstream, and require no fork. See
[`attic/`](attic/) for the abandoned bridge and its post-mortem.

---

## Measured results

x86-64 (Windows, MinGW64, 4 threads), Qwen2.5-0.5B-Instruct Q4_K_M, vanilla
llama.cpp, 123-token shared prefix, 133-token prompts, 40 requests, 3 timed
repeats each.

| TTFT | mean | p50 | p95 |
|---|---|---|---|
| Cold (no cache) | 1395 ms | 1363 ms | 1571 ms |
| Fragment reuse | 185 ms | 185 ms | 223 ms |

**7.5× on cache hits.** Cold 95% CI [1351, 1439] ms.

Breakdown of the 185 ms warm path: ~106 ms decoding the 10-token suffix, ~79 ms
restoring the state blob. Fragment size **1.65 MB** for a 123-token prefix.

Hit rate was 60%, but that is a property of the synthetic corpus, not of any
real workload. **The citable number is the speedup on hits, with the hit rate
stated beside it.**

### Correctness

A latency number from a cache that changes the model's output is worthless. Two
checks run on every benchmark, and both must pass before any speedup is reported.

**Output equivalence.** The first token generated on the warm path must equal the
first token the cold path produces for the same prompt. Result: **24/24**.

**Inert-fragment control.** The model also generates from the *suffix alone*, with
no prefix and no injection. If the warm output matched this, the fragment would be
contributing nothing. Result: **0/24**.

Timing cannot distinguish a working fragment from an ignored one — both skip the
prefix computation, so both are fast. Only the output separates them. The
inert-fragment control is what makes the speedup falsifiable, and it is why the
raw-tensor implementation was caught rather than published.

Reproduce: [`benchmark/real/BENCHMARK.md`](benchmark/real/BENCHMARK.md).
Manifest: `results/bench-measured-20260710-055842.json`.

> **These are not on-device numbers.** They are x86-64. A Cortex-A55 is roughly an
> order of magnitude slower per core; absolute TTFT will be in seconds. The
> speedup is a ratio and should largely hold, but that is a prediction, not a
> measurement. ARM64 results pending.

---

## The KVFragment

`cache/fragment.go`.

| Field | Meaning |
|---|---|
| `TokenStart / TokenEnd` | Token range covered `[start, end)` — the shared prefix |
| `LayerStart / LayerEnd / LayerStride` | Always `0 / NumLayers / 1`. The state blob covers every layer; striding is not expressible and was never sound |
| `Keys` | The opaque `llama_state_seq_get_data` blob — cells **and** metadata |
| `Values` | Sentinel `EDGESYNC-SEQSTATE-v1`, so a fragment from an incompatible extractor cannot be injected |
| `TokenIDs` | Prefix tokens, used to verify the prefix matches |
| `ContentHash` | SHA-256 of `TokenIDs` (not of tensors: tensors differ by epsilon across runs) |
| `EmbeddingVector` | 384-dim vector for HNSW lookup |
| `ExpiresAt` / `HitCount` | TTL 30 min (session) → 7 days (promoted at hit ≥ 5) |
| `Engine` / `EngineVersion` | The blob format is engine- **and version-** specific |

Invariants enforced at construction: `TokenSpan ∈ [64, 2048]`,
`LayerEnd ≤ model.NumLayers`, `len(TokenIDs) == TokenSpan`, `LayerStride ≥ 1`,
`Keys` and `Values` non-empty.

---

## The KVAdapter interface

`adapter/interface.go`. Six methods:

```go
Tokenize(ctx, text)                                    → []int32
ExtractFragment(ctx, tokenIDs, lStart, lEnd, lStride, embedding) → *KVFragment
InjectFragment(ctx, fragment)                          → error
Generate(ctx, prompt, startTokenPos, maxTokens)        → text, nTokens, error
ClearKVCache(ctx)                                      → error
Close()                                                → error
```

`Generate`'s `startTokenPos` is the number of prompt tokens whose KV state is
already present. `0` means cold (prefill everything); `fragment.TokenEnd` means
warm (prefill only the suffix). That skipped prefill is the entire product.

Only the **llama.cpp adapter is real**. `adapter/mlc.go` and `adapter/onnx.go`
compile against their engines' APIs but have never been run against a loaded
model.

---

## Building

Requires CGO and a llama.cpp build. **No patch, no fork.**

```bash
# 1. Build vanilla llama.cpp (static or shared)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --target llama -j8

# 2. Host benchmark, no model, exercises the harness only (prints SIMULATED)
go run ./benchmark/real/

# 3. Real benchmark against a real model
export CGO_CFLAGS="-I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include"
export CGO_LDFLAGS="-L/path/to/llama.cpp/build/src -L/path/to/llama.cpp/build/ggml/src \
  -Wl,--start-group -lllama -l:ggml.a -l:ggml-base.a -l:ggml-cpu.a -Wl,--end-group \
  -lgomp -lstdc++ -lm"

CGO_ENABLED=1 go build -tags realdevice -o edgebench ./benchmark/real/

./edgebench -model-path model.gguf \
  -arch qwen -layers 24 -kv-heads 2 -head-dim 64 -nctx 4096 -threads 4 \
  -n 40 -repeats 3 -warmup 1 -prefix-share 0.8
```

Set `-layers/-kv-heads/-head-dim` to match the GGUF you load; fragments are
rejected on `ModelID` mismatch. Set `OMP_NUM_THREADS` explicitly — a latency
number without a fixed thread count is not reproducible.

The `-tags realdevice` build links the real engine. Without it you get a mock and
every output is stamped `SIMULATED`.

---

## Known limitations

**No on-device measurement.** Everything above is x86-64.

**`PARTIAL` hits are not sound.** The router (`cache/differential.go`) classifies
a lookup as `PARTIAL` when cosine similarity falls in `[0.75, 0.92)`. But a
fragment holds the KV state of prefix *A*. Injecting it to serve a merely
*similar* prefix *B* means generating from a cache that does not correspond to
B's tokens — the output is wrong, and the correctness check would reject it. In
every run to date `partial = 0`, so this path has never actually executed. It
should either be removed or redefined as "longest common prefix", which is a
different mechanism.

**The semantic layer is unproven, and may be unnecessary.** Correct KV reuse
requires the prefix to match **byte for byte**. For that, a hash or a radix tree
is exact and O(1) — which is what vLLM (prefix caching) and SGLang
(RadixAttention) do. The HNSW index and MiniLM embeddings only ever produced
`EXACT` hits on byte-identical prefixes. Their value over a hash is not
demonstrated.

**Cross-engine reuse is dead on this path.** A `llama_state_seq` blob is specific
to llama.cpp *and to its serialization version*. `adapter/reshape.go` transposes
raw tensors between layouts; it does not apply to opaque state blobs. The
compatibility matrix in earlier revisions of this README claimed both "not
implemented" and "implemented" — neither was true in the sense that mattered.

**Per-layer striding cannot preserve output.** `LayerStride = 2` stored 12 of 24
layers. The skipped layers hold no KV for the prefix, so their attention reads
empty cells. Fixed to `1`; the constant is retained only for schema compatibility.

**Fragment memory.** 1.65 MB per 123-token prefix, and it scales with prefix
length and layer count. On a phone this is the binding constraint, not TTFT.
Eviction and TTL exist in `cache/` but their policies are untuned and unmeasured.

**Correctness compares one token.** The `-maxgen` flag is declared but not wired;
`Generate` is always called with `maxTokens = 1`. A single token is nonetheless
decisive *here*, because the inert-fragment control provides the contrast: the
same token cleanly separated 14/24 inert (raw tensors) from 0/24 (state API).
Multi-token comparison would still be a stronger check.

**Unvalidated components.** `cache/compactor.go`, `cache/store.go`,
`prefetch/predictor.go`, `monitor/energy_android.go`, and
`sdk/android/jni_bridge.go` compile and have unit tests, but none has been
exercised end-to-end against a running model on a device.

---

## Roadmap

1. ARM64 cross-compile and an `adb`-driven run on a real phone. This is the only
   step that converts "latency accelerator" into "on-device latency accelerator".
2. Wire `-maxgen`; compare full generated strings, not one token.
3. Replace or justify the semantic lookup. Benchmark HNSW against a radix tree on
   the same corpus. If the radix tree wins, delete the index.
4. Measure `llama_state_seq_set_data` cost as a function of prefix length. It is
   ~79 ms at 123 tokens; if it grows linearly it will eventually erase the gain.
5. Resolve `PARTIAL`: remove it, or reimplement as longest-common-prefix reuse.

---

## License

Business Source License 1.1 (BUSL-1.1) — see [LICENSE](LICENSE).

| Parameter | Value |
|---|---|
| Licensor | bossandboss (Wajdi Kechaou) |
| Licensed Work | EdgeSync-LLM (source, submodules, adapters, documentation) |
| Additional Use Grant | Non-commercial use, research, evaluation, development, internal testing |
| Change Date | July 1, 2029 |
| Change License | AGPL-3.0 |

Free for research, evaluation, development, and internal testing. Production use
in commercial apps, SaaS platforms, or shipped mobile apps requires a commercial
license. On July 1, 2029 the project becomes AGPL-3.0 automatically.

**Commercial licensing:** open an issue at
[github.com/bossandboss/EdgeSync-LLM](https://github.com/bossandboss/EdgeSync-LLM/issues)
with the label `commercial-license`.
