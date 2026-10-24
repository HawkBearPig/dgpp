# Qwen3.8-Flash-Next FP8 cache and native attention

This implementation extends PR #101 with native E4M3 attention and keeps
storage precision separate from attention arithmetic. It applies to Flash-Next
QSA layers, including the draft layer. Other model families retain their
existing implementation.

## Configuration

```json
{
  "engine": {
    "kv_dtype": "fp8",
    "prefill_attention": "fp8",
    "decode_attention": "fp8"
  }
}
```

Both attention settings default to `bf16` and can be selected independently.
Both require an FP8 pool; native attention currently requires head dimension
256. The settings travel in the world handshake and configuration digest.
Older journal records without `decode_attention` retain BF16 compute.

The available choices are:

| Cache | Arithmetic | Tradeoff |
| --- | --- | --- |
| BF16 | BF16 | Existing precision and performance baseline |
| FP8 | BF16 | Smaller cache; only cache quantization changes the operands |
| FP8 | FP8 | Smaller cache and native QK/PV products; additionally quantizes queries and weighted probabilities |

Native FP8 benefits the long selected-token walks measured below. Very short,
single-row decode is faster with BF16 arithmetic. Keep decode and prefill
independently configurable, and evaluate the intended workload before changing
a deployment. Defaults and deployment templates are unchanged.

## Storage and cache lifecycle

Each physical token/head has 256 E4M3 K codes, 256 E4M3 V codes, and separate
FP32 K/V scales. Each scale is `absmax / 448`; zero rows encode to zero. The
compressed index keys and pending-key rings remain BF16. At head dimension 256,
K/V plus scales need 520 bytes per token/head instead of 1,024 bytes. Whole-pool
savings depend on the index/ring geometry.

K normalization, RoPE, quantization and scatter are fused. This preserves the
BF16 rounding points of the original norm/RoPE chain before cache quantization.
The FP8 path bypasses the `kn_` staging read/write, although that scratch buffer
remains allocated for the layer. V is quantized during append. Append iterates
over all heads, including geometries with more than eight KV heads.

Block copying includes K, V, both scale planes, and the compressed index keys
for every layer. Sharing, prefix snapshots, reset, and MTP use the same paged
pool. The memory plan counts the format's payload and scales.

The BF16 arithmetic path converts cached codes with CUDA's hardware FP8
conversion, applies their FP32 scales, and rounds to BF16. Every finite E4M3
value is exactly representable in FP16 and FP32. Tests compare this read path
bitwise with the same attention kernels over explicitly decoded BF16 caches.

## Native arithmetic

`src/kernels/qsa_fp8.cu` gathers K/V codes into shared memory without widening
them to BF16. Both QK and PV use `mma.sync` E4M3 products with FP32 accumulators.
One warp handles up to 16 query heads sharing a KV head, in 32-token tiles.
Additional head groups and KV heads are independent blocks.

1. Quantize the normalized/rotated query per head using `absmax / 448`.
2. Compute QK on the stored codes. Apply the query scale, the selected token's
   K scale, and the attention multiplier to each score.
3. Maintain the running maximum and denominator in FP32 using base-2 softmax.
4. Multiply each probability by that token's V scale **before** quantization.
   A dynamic scale per query head and tile normalizes the largest weighted
   probability to 256. Rescale the running numerator when this scale changes.
5. Compute PV using E4M3 weighted probabilities and the original V codes.
   Normalize the FP32 result using the unrounded softmax denominator.

A single scale on the final output would be incorrect: V scales differ across
tokens inside the reduction. Tests deliberately vary these scales by 4,096×.
Prefill uses one split; decode uses the layer's existing split count and FP32
partial combine. Empty splits write zero numerators. Device-driven counts,
request IDs and selection lists remain graph-replay inputs.

Norm/RoPE, gating, and projection boundaries retain their established types.
The dense weight path is separately controlled by `dense_weights` and the
existing `prefill_fp8_gemm` option. For eligible block-FP8 projections that
option already uses native FP8 GEMM; smaller projections and other scale grids
retain their existing dispatch. Selecting FP8 attention does not change expert
or projection arithmetic. This work does not claim that the entire model
executes in FP8.

## Validation (2026-10-17)

### Kernel performance and numerical error

Single GB10, CUDA 13.0, 12 query heads / 1 KV head, head dimension 256, two
requests with shuffled physical pages. Timings are median CUDA-graph execution
from seven paired trials, rotating mode order, 50 replays per trial. Each query
selects an independent sorted sparse list. These measure attention and its
combine only, not append, projections, model loading, or end-to-end serving.

| Query rows / selected tokens | BF16 cache + compute (µs) | FP8 cache, BF16 compute (µs) | Native FP8 (µs) |
| --- | ---: | ---: | ---: |
| Decode 1 / 32 | 9.30 | 10.24 | 14.74 |
| Decode 1 / 128 | 10.24 | 12.28 | 14.78 |
| Decode 1 / 2,051 | 54.29 | 74.47 | 26.62 |
| Decode 16 / 2,051 | 199.37 | 254.34 | 30.84 |
| Prefill 256 / 2,051 | 625.66 | 2,240.69 | 343.86 |

The long-list cases use a 32,768-token context. Decode uses eight splits;
prefill uses one native split and the existing BF16 warp kernel. Against BF16
cache/compute, relative L2 error on these seeded inputs is approximately
3.7–3.8% for FP8 storage and 4.9–5.0% for native compute. These are empirical
measurements, not universal attention-error bounds. Elementwise FP8 rounding
does not bound the error of softmax or a cancellation-prone output.

Build with `DGPP_BUILD_BENCHMARKS=ON`, then run, for example:

```sh
build-release/qsa_fp8_bench --rows 1 --context 32768 --selected 2051 --splits 8
build-release/qsa_fp8_bench --rows 256 --context 32768 --selected 2051 --splits 1 --prefill
```

### Full-checkpoint smoke comparison

The full 48-layer `nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint, revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47`, was run at world size one with
`ngram_table=mmap`, `dense_weights=fp8`, and streaming residency. All three modes
used the same 147-token plain-text input covering CUDA graphs, arithmetic, and
Python/distributed inference. Every position's full vocabulary logits were
compared; next-token loss uses the following input token at 146 positions.

| Mode | Mean next-token negative log likelihood | Perplexity | Top-1 agreement with BF16 | Mean KL from BF16 |
| --- | ---: | ---: | ---: | ---: |
| BF16 cache/compute | 2.05127 | 7.77779 | 100% | 0 |
| FP8 cache, BF16 compute | 2.05214 | 7.78456 | 93.88% | 0.01278 |
| Native FP8 | 2.05242 | 7.78674 | 94.56% | 0.00892 |

This is a smoke comparison, not a task-quality benchmark, a long-context
accuracy gate, or an end-to-end throughput result. Argmax tokens can differ.
The lower measured KL for native compute on this one input does not establish
that it is generally more accurate than BF16 compute over the FP8 cache.

`qwen_forward_check` supports `--kv-dtype`, `--prefill-attention`, and
`--dump-logits` for repeating such comparisons. The dump is row-major FP32 with
this rank's vocabulary; the program prints its dimensions and vocabulary start.

### Regression coverage

- Independent host FP8 recipe, split counts 1/3/8, empty and partial tiles,
  query-head groups above 16, irregular paging, zero and sharp queries, zero
  cache scales, large V-scale variation, and graph replay with changed inputs.
- Append covers 1/2/9/16 KV heads, strided inputs, padding rows, and untouched
  cache sentinels. Fused norm/RoPE/append reproduces the unfused codec chain.
- BF16 compute over the FP8 pool matches explicit BF16 dequantization, including
  both partial kernels and warp prefill.
- Every layer's block copy preserves both code and scale planes and neighboring
  blocks. Prefix snapshot tests exercise all three storage/arithmetic modes.
- MTP graph tests exercise each independent attention switch through four live
  requests, mixed prefill, verification, retirement and slot reuse.
- Configuration parsing and journal round trips cover the new switch, invalid
  values and legacy records. CUDA memcheck and racecheck report no errors on
  the native kernel tests.

### Pre-merge regression and two-node serving checks

The implementation at `771dc83d` passed a full release build, all 34 host tests
excluding checkpoint-labelled cases, and 23 targeted GPU/fixture/RDMA tests.
The latter include the complete Qwen engine and TP suites, both FP8 attention
test binaries, prefix snapshots, and Qwen3.8-27B BF16/FP8 compatibility checks.

A two-GB10 RoCE deployment of the same NVIDIA NVFP4 checkpoint then passed all
checks in `scripts/serve_api_check.py`. It used `kv_dtype=fp8`,
`prefill_attention=fp8`, `decode_attention=fp8`, graph execution, MTP depth one,
two request slots, an 8,192-token pool, a 0.25-GiB prefix cache,
`dense_weights=fp8`, `bf16_weights=bf12+bf16`, and the mapped n-gram table.
The checks covered streamed and non-streamed stops, two simultaneous choices,
forced/banned tokens, invalid logit bias, and usage accounting. A repeated
2,186-token prompt reused 2,180 cached tokens.

Both ranks logged the same settings digest with native FP8 prefill and decode
enabled. After orderly shutdown, their operation streams had identical MD5
`02e1324bf170d9e5f75fbab466c24e30`. This establishes two-node serving correctness
for the tested configuration. Matched end-to-end performance measurements,
broader task-quality checks, and long-context accuracy remain separate gates.
