---
license: other
license_name: glm-5.3
base_model: zai-org/GLM-5.3-BF16
base_model_relation: quantized
pipeline_tag: text-generation
tags: [glm, moe, quantization, gptq, mixed-precision, int8, custom-format, dgpp]
---

# GLM-5.3 Mixed346 GPTQ H32 A8, group 128

A quantization of the full **GLM-5.3** (754B-parameter MoE, model type `glm_moe_dsa`) from the
original BF16 release, for the DGPP engine on four DGX Sparks (TP=4). It is the
[Int4/Int8 RTN checkpoint](GLM-5.3-Int4-Int8Mix-RTN-g64.md) with the routed experts of layers
3–77 re-encoded expert by expert at 3, 4 or 6 bits, calibrated by GPTQ in a block-32
Hadamard-rotated input basis, and multiplied by INT8 expert activation codes; 116 experts keep
the Int4/Int8 form.

| | |
|---|---|
| Quantized components | gate, up and down projections of the 256 routed experts, main MoE layers 3–77 (19,084 of 19,200 experts) |
| Expert forms | gate/up at one width and down at its own: `444` 15,672 experts, `334` 2,171, `446` 441, `333` 352, `443` 289, `336` 85, `666` 74; `existing` (Int4 g64, BF16 activations, no rotation) 116 |
| Weight representation | 3-bit indices into `[-127, -79, -45, -14, 14, 45, 79, 127]` (Gaussian Lloyd-Max), 4-bit indices into the NF4I8 codebook `[-127, -88, -67, -50, -36, -23, -12, 0, 10, 20, 31, 43, 56, 71, 92, 127]`, 6-bit signed integers (index − 32); one dense bit stream per row, codes crossing word boundaries |
| Scales | one BF16 scale per 128 input channels per output row (4.05 bits/weight with scales over the main experts, the Int4/Int8 checkpoint's 4.25) |
| Activations | H32 rotation, one BF16 rounding, then per 128 rotated values one FP32 scale `max(amax / 127, 1e-30)` and INT8 codes `clamp(rint(x / s), -128, 127)`; the expert input before gate/up, the SwiGLU output before down; integer dots per group, FP32 scaling |
| Rotation | normalized 32-wide Sylvester Hadamard on the routed experts' inputs; never on the residual, router, shared expert or attention |
| Calibration | GPTQ, 128-column blocks, seven documents (40,960 tokens), up to 2,048 routed rows per expert, original BF16 activations |
| Everything else | byte for byte the Int4/Int8 checkpoint: int8 g64 attention and shared expert, BF16 dense layers 0–2, indexers, routers, norms, embedding, head, draft layer |
| Tensors | `weight_indices` I32 `[N, K*bits/32]`, `weight_scale` BF16 `[N, K/128]`, `weight_shape`; `config.json` `quant_method: dgpp_mixed346`, format `dgpp_mixed346_h32_a8_g128_v1`, with the sidecars `quantization-recipe.json` (every expert's form, the codebooks, the activation policy) and `baseline-quantization-config.json` (the kept tensors' compressed-tensors block) |
| Size | 409.3 GB in 79 shards (per-layer files and one passthrough shard); tensor payload 381.19 GiB (the Int4/Int8 checkpoint: 398.06), 4.22 GiB less per rank at TP=4 |

## The publisher's accuracy table

Teacher-forced through all 78 main layers on four held-out 4,096-token documents (16,380
scored tokens; the allocation frozen before the test), against the original BF16 model:

| Model | Mean NLL | Mean KL to BF16 | Top-token agreement |
|---|---:|---:|---:|
| Original BF16 | 0.565816 | 0 | 100 % |
| Int4/Int8 RTN g64 | 0.591183 | 0.050547 | 95.220 % |
| **Mixed346 H32 A8 g128** | 0.587319 | 0.040918 | 95.788 % |

19 % less divergence from BF16 at an NLL change whose 95 % interval spans zero. The card's
kernel benchmarks (expert blocks alone, not whole-model throughput) report a median 16 %
decode time reduction against the Int4/Int8 expert execution; it makes no serving claim.

## DGPP serving notes (2026-10-10)

**The format in the engine:** packed scale format 3 (`kPackedScaleBf16G128Mixed346`,
`models/quant_matrix.hpp`), parsed from the checkpoint's `quant_method: dgpp_mixed346`
block and its two sidecars (or both inlined under `quantization_config.baseline` /
`.recipe`), refused on any drift of the format, group, rotation, activation policy, codebooks
or the recipe's coverage; the binding names every expert's triple by its own form, the loader
slices the dense bit streams at the 128-group. The activation quantizer
(`kernels/hadamard32.cu`) rotates and encodes each row once per step: the token rows inside
the post-attention norm's own pass (`glm_rmsnorm_bf16_quant_int8`, one kernel for the norm
and the codes), the SwiGLU rows before down (the shared slot's row and an existing expert's
input stay BF16). A Mixed346 GEMV core (`kernels/packq_a8_gemv.cuh`) multiplies the codes:
at 4 bits a lane owns a 16-byte piece of a row (four lanes a group, the group's exact INT32
dot summed by two xor shuffles); at 3 and 6 bits a lane owns a 48-byte piece — a whole group
or half of one — that the row's lanes load cooperatively as contiguous 16-byte vectors and
swap through a per-warp shared tile; every group dot is scaled once in FP32 by the weight and
activation scales. The slot and grouped kernels dispatch per expert view between this core,
the Int4 g64 core for an existing expert and the Int8 core for the shared expert. Prefill
runs a converted expert's tiles on int8 tensor cores (`kernels/packq_gemm.cu`,
`wide_tile_i8`): the rows' INT8 codes against the weight rows decoded to INT8 levels through
mma m16n8k32, a 128-code group's dot the exact INT32 sum of its four MMAs, scaled once; an
existing expert's tiles run the BF16 group-64 chain in the same launch. Every path computes
the same group dots; they differ by the FP32 accumulation order only.

**Memory plan, world 4 (rank 0):** model weights resident 92.10 GiB per rank (the Int4/Int8
checkpoint: 96.32, NF4I8: 93.68), 52.6 KiB per context token with the fp8 latent cache. The
template `deploy/cluster_glm-5.3_mixed346_w4.example.json` carries a 272K-token fp8 K/V pool
at eight slots, MTP depth 1 (plan 108.97 GiB + 4 GiB headroom). The first boot writes the
resident image (381 GiB read per node); a rank that finishes its image first waits for the
others at the first collective, whose 60-second deadline the skew exceeded once on
2026-10-10 — boot again, the images then restore in ~30 s.

**Sensitivity to upstream noise:** the INT8 codes make a layer about twice as responsive to
a one-ulp input change as the BF16-activation formats (`glm_moe_test`'s probe: row l2 0.0044
per input ulp against 0.0025 for NF4I8 and Int4), because a code flips. The kernels match
their oracles exactly; the fixture gates that compare chains (the Python reference, a TP
world, the decode path against a tensor-core re-forward) carry budgets stated for this
contract (per-layer l2 0.015, a 4 % near-tie margin — or the row's own divergence, where a
chunked prefill's decode rows sit 10–15 % off the one-shot walk — the audit's re-forward under
the tensor-core threshold).

**Gates (2026-10-09/10):** the Mixed346 GEMV core at every width and routed K (0 mismatches
against the oracle), the quantizer bitwise its host reference and the fused norm bitwise the
norm then the quantizer, the int8 tensor-core tile against the oracle (relative error 5e-8)
and the GEMV core at every form, warp layout and tile list, the MoE layer (0 ulps against the
double oracle, slot path bitwise the host chain, graph table, sliced folds within budget),
the loader byte-exact at worlds 1/2/4 with the image round trip, the config and binding of
the landed checkpoint, the forward/decode/TP/engine fixture gates.

**Measured on the four nodes (2026-10-10, the fp8-KV template, greedy, MTP depth 1):** decode
62.7 ms per pass at one request (Int4/Int8 63.6, NF4I8 64.0; 1.76–1.98 tokens per pass by
class, the same acceptance as both) and 236 ms per pass at eight (Int4/Int8 252, NF4I8 253);
engine 28.1–31.6 tok/s at one request, 58.3–62.9 at eight. Cold prefill 3.906 / 17.943 / 76.714 s
at 2K / 8K / 32K (Int4/Int8 3.896 / 17.866 / 76.439). Decode at 0 / 32K / 64K / 128K / 256K
context: 62.9 / 74.7 / 79.5 / 88.9 / 108.2 ms per pass (NF4I8 64.3 / 76.4 / 81.0 / 90.2 / 109.2);
the cold 256K prefill 893 s. Teacher-forced on the hard text: 2.3284 nat/token (Int4/Int8 2.3295,
NF4I8 2.3276; every mean delta inside its standard error). Quality sets: HumanEval 157/164,
GSM8K 291/300, schema extraction 100/100 (NF4I8 157 / 294 / 100, Int4/Int8 160 / 293 / 100).
At two and four requests 87 and 141 ms per pass (37.3–39.1 and 47.3–52.0 wall tok/s). In situ on rank 0 (nsys, a 2K prefill): the int8
tensor-core gate/up tile 2.30 ms per launch against the Int4 tile's 2.55, the down tile 3.84
against 3.54; the decode slot kernels 224 / 109 µs against 231 / 110.

### The window's full form (2026-10-20)

The L2 weight prefetcher's full-rate windows took a narrower kernel shape on 2026-10-20 (16 blocks × 8 loads, 512 KB
in flight, the DRAM rate at the light form's SM footprint; DeepSeek-V4.1, Qwen3.8 and GLM-5.3-Flash measured ahead on
it). This family's windows run beside its streaming kernels, where the bytes in flight set a window's share, and the
narrow form cost it 0.3–0.8 % a pass at C1 (code 62.29 → 62.47/62.78 ms per pass on two boots; the 48 × 4 form rebuilt
for attribution read 62.29 again). The three GLM-5.3 templates therefore ship `engine.l2_prefetch_full_form: "48x4"`;
the key's default is `16x8` (`docs/configuration.md`).
