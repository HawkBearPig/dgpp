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
clamp of [-7, 7] and an up offset of 1. The per-token FlashInfer path fuses
the FP32 first GEMM accumulations with the activation and emits BF16 before
the explicit second input quantizer. The second GEMM emits BF16 before the
weighted expert reduction. In particular, it does not insert separate BF16
gate/up roundings. The earlier activation prescreen did not certify these
boundaries.
[MiniMax quantization adapter](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/layers/quantization/minimax_m3_nvfp4.py).
[FlashInfer per-token workspace and output types](https://github.com/flashinfer-ai/flashinfer/blob/d84335d8c54e0c8f9f7e22faa15c7e34a58abf94/csrc/trtllm_fused_moe_kernel_launcher.cu).

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
different outputs from each other. The isolated GPU checks below cover the
prefill formula; full-model validation is still pending.

## Initial GB10 projection kernel validation

The new `mxfp8_gemm` kernel uses SM121's block-scaled E4M3 tensor-core
instruction with 32-value E8M0 groups. Its quantizer matches the actual
pinned SGLang Triton producer byte for byte on all 65,280 finite BF16 bit
patterns and another 18,528 inputs around scale-exponent boundaries, with
padded row strides and a partial final quantizer tile. Both payloads and
scales are compared; no tolerance is used for this check.

Four matrix cases cover 1 through 65 rows, incomplete output tiles and
reduction lengths through 6,144. They pass independent FP64 forward-error
bounds; the largest relative RMS error is `3.09e-6`. BF16 outputs are exactly
the rounding of FP32 outputs. Splitting the same queries into 1-, 8- and
17-row calls preserves every FP32 output bit. The warnings-as-errors build,
CUDA memcheck and racecheck pass, with zero errors or hazards.

These are isolated kernel checks. The kernels are not yet connected to the
model's serving path, and full-model native fidelity, throughput and final
deployment selection remain pending. Reproduce the upstream quantizer check
with `tools/minimax_mxfp8_quantization_check.py`, providing the built
`mxfp8_gemm_test`, the pinned SGLang source and an untracked output directory.

## Initial GB10 Q8KV4 kernel validation

The new native attention kernels implement NVFP4 cache append, FP8 query
conversion, sparse index scoring and the upstream **prefill** attention
formula. They use that formula for every query geometry, including single
rows. This preserves DGPP's batch/chunk invariance; it does not establish
equivalence to the upstream online decode formula.

Five cases cover 1 through 65 query rows, mixed requests, permuted physical
pages, 1 through 4 KV heads, causal boundaries, more than sixteen candidate
pages, exact index ties and padded rows with invalid request IDs. Query and
cache bytes match the independent precision reference exactly. Selected
pages match, index scores satisfy FP64 forward-error bounds, and attention
relative RMS error is at most `1.74e-5`, within the existing `0.004` gate.
Splits into 1-, 8- and 17-row calls and two PDL CUDA-graph replays preserve
the output bits. CUDA memcheck and racecheck report zero errors or hazards.

Run `ctest --test-dir build-release -R '^minimax_native_attention_test$'
--output-on-failure` after building that target. The test generates temporary
fixtures and checks CUDA output against the independent CPU formula. These
tests certify the isolated path, not full-model fidelity or throughput.

## Initial native expert stage validation

The native expert quantizer matches the actual hash-pinned FlashInfer 0.7.2
CuTe GPU producer byte for byte: E2M1 payloads, E4M3 block scales and FP32
per-token globals. Cases include all 65,280 finite BF16 patterns, signed
zero rows, padded input strides, outliers and row widths through 32,768.
The exhaustive tiny inputs also reproduce upstream's approximate reciprocal
with subnormal flushing; they are not evaluated with a normal-range
nearest-format rounding bound.

Four expert chains cover 1 through 65 rows and a 6,144-wide hidden input.
Their first matrix products satisfy independent FP64 forward-error bounds;
SwiGLU has the required FP32-to-BF16 boundary; and the second product meets
its FP64 accumulation plus BF16-rounding bound. Splitting into 1-, 8- and
17-row calls preserves the stage outputs exactly. The new producers pass
PDL graph replay, CUDA memcheck and racecheck, with zero errors or hazards.

Run `minimax_native_moe_test` through CTest for the independent stage checks.
For the additional upstream GPU comparison, run
`tools/minimax_native_moe_regression.py --app <test-binary> --upstream` in an
environment containing the pinned FlashInfer source and its CuTe dependency.
The checker verifies the quantizer source hashes before execution. This is
not a full upstream MoE or full-model comparison.
