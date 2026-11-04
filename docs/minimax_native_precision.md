# MiniMax M3.1 Flash: native precision contract

The [publisher's model card](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
recommends SGLang, vLLM and TokenSpeed. It describes a model trained with
quantization awareness. The Blackwell serving paths quantize activations and
attention, in addition to reading the quantized checkpoint weights. The
current DGPP BF16 activation/cache path is a separate comparison recipe.
Failure of that recipe's parity limits does not establish failure of native
inference fidelity or task quality.

## Pinned reference

The initial native reference audit uses:

- SGLang `58484246d26523eaf1a7f16ab524deee288d2120`.
- FlashInfer 0.7.2, `d84335d8c54e0c8f9f7e22faa15c7e34a58abf94`, as required by
  that SGLang revision's
  [package manifest](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/pyproject.toml).
- MSA `c8e1013a16a708d2a7656805d5fc26f2b87715d1`, the tested revision named in
  the [SGLang deployment recipe](https://docs.sglang.io/cookbook/autoregressive/MiniMax/MiniMax-M3.1).

This pins source behavior; it is not a claim that the complete upstream
Blackwell stack has been executed on the four GB10 nodes. The upstream MSA
kernels target SM100/SM103 and use facilities absent from SM121. DGPP needs
compatible kernels and independent numerical validation on GB10.

## Projection boundaries

Dense quantized projections use MXFP8 weights and dynamic MXFP8 activations:
32 values share one E8M0 power-of-two scale; values use E4M3. The scale rounds
up to cover the group's maximum. Payloads and scales must be validated,
including zero groups and exponent boundaries, before comparing products.

The pinned SGLang dense/shared MLP path materializes BF16 gate/up outputs.
Its fused SwiGLU-OAI/MXFP8 producer also rounds the activation result to BF16
before quantizing it, preserving the unfused two-kernel recipe. This differs
from a separate no-BF16-rounding fused implementation in the same source
tree. The actual call path, not the availability of a similarly named
kernel, defines this reference.
[MLP dispatch](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/models/minimax_m3.py),
[fused producer](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/kernels/ops/activation/swiglu_mxfp8_swizzled.py).

Routed experts use NVFP4 W4A4. Both GEMM inputs are quantized dynamically per
token. The MiniMax adapter explicitly neutralizes the checkpoint's static
input globals; those globals must not be reused as static activation scales.
The SwiGLU-OAI function uses alpha 1.702, an upper gate clamp of 7, an up
clamp of [-7, 7] and an up offset of 1. Expert fusion boundaries and the
second GEMM's scale/epilogue remain part of the reference audit; the earlier
activation prescreen does not certify them.
[MiniMax quantization adapter](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/layers/quantization/minimax_m3_nvfp4.py).

## Q8KV4 attention

After normalization and rotary embedding, attention queries and index
queries are converted from BF16 to E4M3 with unit outer scale. Main K/V and
index K are stored as NVFP4: a 128-value head has 64 payload bytes and eight
E4M3 scales. Each 16-value group uses `E4M3(amax / 6)`; the cache's outer
scale is fixed at one, unlike the expert activation row scale.
[Attention dispatch](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/layers/attention/minimax_sparse_nvfp4_backend.py),
[cache storage](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/mem_cache/minimax_m3_nvfp4_pool.py).

When consumed, E2M1 payloads are multiplied by E4M3 scales and **rounded to
E4M3 again**, including saturation. Index scoring uses these reconstructed
FP8 keys. Attention also quantizes unnormalized softmax weights to E4M3,
after multiplying them by 448. The softmax denominator retains the FP32
weights. Omitting either rounding implements a different formula.
[MSA conversion reference](https://github.com/MiniMax-AI/MSA/blob/c8e1013a16a708d2a7656805d5fc26f2b87715d1/tests/inference/dequant/nvfp4_to_fp8/reference.py).

The upstream phase formulas differ:

| Phase | Softmax/PV accumulation |
| --- | --- |
| Prefill | Normalize each selected 128-token page independently; quantize its weights; round each page output to BF16; combine pages using FP32 log-sum-exp weights and round the result to BF16. |
| Decode | Walk selected pages with an online maximum, an FP32 denominator and an FP32 output accumulator; quantize each page's weights relative to the running maximum; round the final output to BF16. |

These are not bitwise interchangeable: the probability quantization and
intermediate BF16 boundaries can change values. DGPP's batch/chunk
invariance requirement therefore needs explicit validation of the selected
execution formula. Matching the format names alone does not establish it.
[Prefill oracle](https://github.com/MiniMax-AI/MSA/blob/c8e1013a16a708d2a7656805d5fc26f2b87715d1/tests/inference/msa_v1/attention/prefill/q8kv4/reference.py),
[decode oracle](https://github.com/MiniMax-AI/MSA/blob/c8e1013a16a708d2a7656805d5fc26f2b87715d1/tests/inference/msa_v1/attention/decode/reference.py).

`tools/minimax_native_attention_reference.py` represents both formulas with
contiguous tensors and PyTorch FP32 arithmetic, independently of DGPP's
paged buffers and CUDA kernels. It is a mathematical reference, not an
upstream GPU execution or a performance measurement. Full-model validation
must record the chosen phase formula, projection reductions, TP boundaries
and quantizer implementation. Existing numerical limits remain unchanged;
BF16-vs-native task quality is a separate comparison.

Run the CPU cross-check against an MSA source checkout with:

```sh
python tools/minimax_native_attention_check.py --msa-source /path/to/MSA
```

The checker verifies the upstream file hashes before loading the reference
functions. It covers all 16 E2M1 codes crossed with all 127 nonnegative
finite E4M3 scales, signed-zero packing, zero groups, causal page boundaries,
more than sixteen candidate pages, permuted physical pages and query chunk
invariance within each formula. At 1, 129, 270 and 2,057 keys, both formulas
match their upstream CPU references exactly. They deliberately produce
different outputs from each other. GPU quantizer/kernel and full-model
validation are still pending.
