# Attic — the abandoned raw-tensor KV bridge

`edgesync_kv_bridge.cpp` and its patch expose llama.cpp's per-layer K/V tensors
so fragments could be read and written directly. Kept for reference; **not used**.

## Why it was abandoned

Copying K/V tensors moves the numbers but not the bookkeeping. llama.cpp stores,
per cell, a position and a sequence id, and builds its attention mask from those.
A raw tensor write leaves the cell table empty: the cache still believes the
sequence is empty, attention never sees the injected cells, and the next decode
reallocates over them.

The fragment is therefore **inert** — fast (the prefix is skipped) and wrong (the
prefix is gone). Measured: 14 of 24 cache hits reproduced, token for token, the
output of a generation with no prefix at all. The 8.76x "speedup" was dropped
context, not cache reuse.

Two further defects, both structural:

- `FragmentLayerStride = 2` stored only 12 of 24 layers. The skipped layers hold
  no KV for the prefix, so their attention reads empty cells. Output can never
  match the uncached path, whatever the metadata says.
- The bridge required a forked, recompiled llama.cpp. No downstream user would
  accept that.

## What replaced it

`llama_state_seq_get_data` / `llama_state_seq_set_data` — public API, present in
upstream llama.cpp. They serialise cells **and** metadata, for every layer.
EdgeSync now links against a vanilla libllama.

Result after the switch: correctness 24/24, inert-fragment control 0/24,
7.5x TTFT speedup on cache hits.
