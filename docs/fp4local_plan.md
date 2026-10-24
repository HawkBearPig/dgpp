# Qwen3.8-Flash-Next FP4-Local

`nvidia/Qwen3.8-Flash-Next-FP4-Local` uses the Qwen3.8 engine and its
48-layer GDN/sparse-attention MoE. The deployment template is
`deploy/cluster_qwen-3.8-flash-next-fp4local_w1.example.json`.
The checkpoint used for validation is revision
`b9ce18a6887e18c089fdea8509a33e63429a3300`.

## Checkpoint contract

| Component | Checkpoint form | Execution |
|---|---|---|
| Backbone routed experts | NVFP4 g16 with CSF scale sidecars | Lossless scale expansion at load; shared NVFP4 kernels |
| Attention and shared experts | MXFP8 g32, E4M3 payload and E8M0 scales | Resident quantized weights; shared FP8 prefill and decode kernels |
| N-gram table | NVFP4 payload and group scales | Shared resident or mmap table path |
| MTP routed experts | Weight-only NVFP4 with plain scales, no input scale | Shared draft and NVFP4 paths |
| GR, PLE projections, head, embeddings and router | BF16 | Existing checkpoint or at-load FP8 policy |
| HC, mixer and PLE norms | F32 | Existing F32 norm path |
| Vision linears | MXFP8 or weight-only NVFP4 | Validated against expected shapes, then widened to BF16 at load |

ModelOpt's `weight_scale_encoding: "csf"` selects CSF independently for the
backbone and draft. Mixed declarations within either scope are rejected.
Every compressed matrix requires its own fixed and exception sidecars with
the expected dtype and geometry. Unrelated sidecars are rejected.

CSF stores 16-row slabs. Each slab has 16 row-base bytes, followed by packed
4-bit deltas in row order. An exception's high byte replaces the scale at
its low-24-bit flat index. Expansion preserves the original E4M3 bytes;
row and column slicing happens afterward. The resident weight layout and
kernel inputs are the same as the plain NVFP4 export.

Shard discovery and deployment preflight both follow
`model.safetensors.index.json`, including its `tensors/` paths. Preflight
rejects absolute paths and traversal while accepting nested relative paths. The loader retains source scale pages until their copy
finishes, then releases them through the shared resident-loader machinery.

## Shared engine paths

The model does not introduce a separate forward implementation. The PR is
integrated with the engine changes on master through `a0d4e055`.

| Engine optimization | FP4-Local path |
|---|---|
| FP4 single-pass down projection and tile stealing | `QwenMoeLayer` → `GlmMoeLayer`; the 640-column expert down projection qualifies |
| Optional FP8 weight-tile prefill | MXFP8 attention supports resident E8M0 byte scales; shared experts support the same tile with F32 scales |
| Staged attention-combine metadata | Shared full-attention/DSA kernels, including the FP8 KV pool |
| Long-context indexer cache reuse | Shared QSA scoring visits query rows together for each key stripe once the index cache exceeds L2 |
| Programmatic dependent launches | Shared graph and kernel launch paths |
| Draft hidden-state copies | Shared Qwen MTP kernels |
| Neutral sampling penalties | Shared sampling fast path |
| BF12 companions and L2 prefetch | Existing Qwen dispatch where the resident weight form permits them |

`engine.prefill_dense: "auto"` preserves the faster existing MXFP8 bridge
policy for Qwen. The shared tile remains available for explicit selection;
DeepSeek-V4.1 continues to choose its tile under `auto`.

`engine.prefill_dense: "tile"` selects the shared weight tile for MXFP8
prefill at every supported row count. It reads compressed weights directly,
without materializing an entire BF16 matrix. Decode retains its existing
streaming accumulation chain. `streaming` selects the previous MXFP8
streaming/large-row BF16 bridge behavior for comparison. Different
accumulation orders can differ within FP32 rounding; the tile is not a
bitwise replacement for the bridge.

The template encodes the remaining BF16 dense weights to block FP8 at load,
uses the MMA vocabulary head, FP8 KV and prefill attention, MTP depth 3,
the screened draft head, and lookup drafting. The n-gram table is mmap'ed.
It reserves 1,048,576 KV tokens and uses YaRN factor 4 over the checkpoint's
262,144-token window. The pool is shared across four request slots; a
single request at the full positional ceiling consumes the pool.

Sparse attention bounds the final attention walk to 2,048 selected tokens
plus an incomplete compression group. Its indexer still scores every
visible group of four tokens for each query, so total cold-prefill indexing
work grows quadratically with prompt length. Long-context prefill reorders
score blocks to reuse each key stripe across queries rather than repeatedly
streaming the entire compressed key cache. This preserves every score bit,
selection, and causal bound; short contexts and decode retain their previous
ordering. It removes an avoidable cache penalty without changing that
underlying scaling.

## Regression coverage

The nested-shard fixture has the same numerical weights as the plain QAD
fixture. It exercises CSF backbone experts, plain draft scales, MXFP8
projections, F32 norms, and the NVFP4 n-gram table. Loader checks compare
payloads and scales byte-for-byte at worlds 1, 2, and 4. Forward checks use
the independent Python reference from the plain fixture. Decode and
engine tests exercise graph replay, speculative decoding, concurrent
requests, and mixed prefill.

The shared FP8 tile tests cover byte and float scales, ragged output tiles,
row counts around 128, chunk cuts, CUDA graph replay, output padding, and
large valid E8M0 exponents. Host tests cover malformed CSF sidecars and
quantized vision shapes, including same-byte-count transpositions.

Real-checkpoint serving results are recorded in [benchmarks](benchmarks.md)
with source and binary provenance. The final campaign covers context through
512K, quality, API behavior, and plain/default greedy transcript agreement.
Live image API checks passed. A separate capacity probe completed three
requests at 1,048,094 prompt tokens with zero swap, including prefix reuse;
it does not establish retrieval quality at that context. Solo/batched greedy
text differs for the recorded probe, as with the other Qwen exports, while
repeated C1 and plain/default transcripts match across all five classes.

Run the deployment's row set with:

```sh
python3 scripts/serve_campaign.py run \
  --config deploy/cluster_qwen-3.8-flash-next-fp4local_w1.example.json \
  --out benchmarks/results/qwen-fp4local-w1
```

Keep generated records untracked. Record the tested source revision,
binary hash, zero-swap checks, cold-cache checks, and operation-stream
identity with the validation summary in the PR.
