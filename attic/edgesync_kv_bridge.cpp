// edgesync_kv_bridge.cpp — EdgeSync-LLM raw KV tensor access shim.
//
// WHY THIS FILE EXISTS
// ─────────────────────
// EdgeSync-LLM's fragment cache needs raw, per-layer float K/V tensor bytes
// (to fingerprint, sparsify, and reuse across requests). llama.cpp does NOT
// expose this through its public API (llama.h) — the only public state
// access is the opaque llama_state_seq_get_data()/set_data() blob, which
// can't be sliced or interpreted per-layer.
//
// The original EdgeSync-LLM repo's adapter/llamacpp.go assumed a function
// `llama_get_model_tensor(ctx, name)` with named tensors "cache_k_l%d" /
// "cache_v_l%d" — this does not exist in llama.cpp (verified by attempting
// to link against a real build: "undefined reference to llama_get_model_tensor").
// The actual current (as of this file's writing) internal storage is
// llama_kv_cache::layers[i].k / .v, reachable only via internal headers.
//
// This file is compiled as part of llama.cpp's own build (added to
// src/CMakeLists.txt) specifically so it can #include the internal headers
// (llama-kv-cache.h, llama-memory.h) that are NOT installed/exported as
// part of llama.cpp's public API. It exposes a small extern "C" surface
// that adapter/llamacpp.go's CGO code calls instead.
//
// FRAGILITY WARNING (same one the original repo's own comments already
// flagged): this depends on llama_kv_cache's internal layout. It works for
// the standard unified KV cache. Architectures using llama_kv_cache_iswa /
// llama_kv_cache_dsa / llama_kv_cache_dsv4 / hybrid memory (Mamba, SWA-only
// models) will fail the dynamic_cast below and return an error rather than
// silently reading wrong data — check the return code.
//
// UNVERIFIED AGAINST A REAL MODEL: this compiles and links (confirmed), but
// the token-range byte-offset math has NOT been validated against a real
// loaded GGUF and a real llama_decode() call — no model file was available
// in the environment that wrote this. Before trusting fragment contents,
// verify get_k_storage()/get_v_storage()'s tensor shape (ne[]/nb[]) matches
// what this file assumes, with a real model, and adjust the offset
// calculation if not.

#include "llama-kv-cache.h"
#include "llama-context.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>

extern "C" {

// Returns 0 on success, negative on failure. On failure, *keys_out and
// *vals_out are left untouched — caller must check the return code, not
// infer success from non-zero output.
//
// keys_out/vals_out: caller-allocated buffers, each must hold at least
// (ggml_nbytes of the full per-layer storage tensor) bytes. This function
// copies the ENTIRE current per-layer K/V storage tensor — NOT a
// token_start/token_count slice — because slicing requires knowing the
// cache's current cell layout (padding, permutation), which was not
// verifiable without a running model. Callers must slice the returned
// buffer themselves once they've confirmed the real layout, or extend this
// function once that's known.
//
// out_k_nbytes/out_v_nbytes: actual byte sizes written, so Go doesn't need
// to duplicate ggml's size math.
int edgesync_extract_layer_raw(
    struct llama_context * ctx,
    int32_t layer,
    void * keys_out, size_t keys_out_capacity, size_t * out_k_nbytes,
    void * vals_out, size_t vals_out_capacity, size_t * out_v_nbytes
) {
    if (ctx == nullptr || keys_out == nullptr || vals_out == nullptr ||
        out_k_nbytes == nullptr || out_v_nbytes == nullptr) {
        return -1;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return -2;
    }

    // Only the standard unified KV cache is supported by this shim. Hybrid /
    // iSWA / DSA / DSV4 memory types use different internal classes; this
    // safely returns an error rather than misreading memory.
    llama_kv_cache * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (kv == nullptr) {
        return -3; // unsupported memory type for this model architecture
    }

    ggml_tensor * k_tensor = kv->get_k_storage(layer);
    ggml_tensor * v_tensor = kv->get_v_storage(layer);
    if (k_tensor == nullptr || v_tensor == nullptr) {
        return -4; // invalid layer index
    }

    size_t k_nbytes = ggml_nbytes(k_tensor);
    size_t v_nbytes = ggml_nbytes(v_tensor);

    if (k_nbytes > keys_out_capacity || v_nbytes > vals_out_capacity) {
        return -5; // caller-provided buffer too small
    }

    ggml_backend_tensor_get(k_tensor, keys_out, 0, k_nbytes);
    ggml_backend_tensor_get(v_tensor, vals_out, 0, v_nbytes);

    *out_k_nbytes = k_nbytes;
    *out_v_nbytes = v_nbytes;

    return 0;
}

// Returns the byte size ggml_nbytes() would report for layer `layer`'s K and
// V storage tensors, so Go can allocate correctly-sized buffers before
// calling edgesync_extract_layer_raw. Returns 0/0 via out params on failure;
// check the return code, not just the sizes.
int edgesync_layer_sizes(
    struct llama_context * ctx,
    int32_t layer,
    size_t * out_k_nbytes,
    size_t * out_v_nbytes
) {
    if (ctx == nullptr || out_k_nbytes == nullptr || out_v_nbytes == nullptr) {
        return -1;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return -2;
    }

    llama_kv_cache * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (kv == nullptr) {
        return -3;
    }

    ggml_tensor * k_tensor = kv->get_k_storage(layer);
    ggml_tensor * v_tensor = kv->get_v_storage(layer);
    if (k_tensor == nullptr || v_tensor == nullptr) {
        return -4;
    }

    *out_k_nbytes = ggml_nbytes(k_tensor);
    *out_v_nbytes = ggml_nbytes(v_tensor);

    return 0;
}

// Injects raw K/V bytes back into a layer's storage tensor. Same caveats as
// extraction: writes the FULL tensor, not a token-range slice, and only
// supports the standard unified KV cache.
int edgesync_inject_layer_raw(
    struct llama_context * ctx,
    int32_t layer,
    const void * keys_in, size_t keys_in_size,
    const void * vals_in, size_t vals_in_size
) {
    if (ctx == nullptr || keys_in == nullptr || vals_in == nullptr) {
        return -1;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return -2;
    }

    llama_kv_cache * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (kv == nullptr) {
        return -3;
    }

    ggml_tensor * k_tensor = kv->get_k_storage(layer);
    ggml_tensor * v_tensor = kv->get_v_storage(layer);
    if (k_tensor == nullptr || v_tensor == nullptr) {
        return -4;
    }

    size_t k_nbytes = ggml_nbytes(k_tensor);
    size_t v_nbytes = ggml_nbytes(v_tensor);

    if (keys_in_size > k_nbytes || vals_in_size > v_nbytes) {
        return -5; // size mismatch — caller's buffer doesn't match current tensor size
    }

    ggml_backend_tensor_set(k_tensor, keys_in, 0, keys_in_size);
    ggml_backend_tensor_set(v_tensor, vals_in, 0, vals_in_size);

    return 0;
}

} // extern "C"
