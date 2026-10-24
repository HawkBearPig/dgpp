# Flint (FP8 / NVFP4 mixed export) on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[HawkBearPig/flint-FP8-NVFP4](https://huggingface.co/HawkBearPig/flint-FP8-NVFP4).
It is **not** a republished checkpoint: the engine loads the export as
shipped, with no re-quantization pass.

## The checkpoint, as loaded

34 GiB across 49 per-layer shards and `model-non-experts.safetensors`. The
routed experts are in the export's per-projection formats
(`quantization_config.routed_experts.precision_by_layer`): FP8 e4m3 with
fp32 scales on 128×128 blocks on layers 2–11, 13–30 and 32; NVFP4 (e2m1
pairs, an e4m3 scale per 16 along K, one fp32 reciprocal global per
matrix) on layers 31, 33–36 and 38–48; and two mixed layers — layer 12
with an NVFP4 `up_proj` beside FP8 `gate_proj` / `down_proj`, layer 37
with NVFP4 `gate_proj` / `up_proj` over an FP8 `down_proj`. Every other
tensor is BF16: the q / k / v / o projections, the per-head `q_norm` /
`k_norm`, the per-head sink (`softmax_offset`), the layer norms, the
routers, the shared experts, the two dense MLPs, the embedding and the
LM head (the matrices the GEMM interface serves stream as their lossless
12-bit companions under `engine.bf16_weights = "bf12"`). The engine
decodes every code exactly and runs activations in BF16; the K/V cache is
BF16 or the fp8 row form (`engine.kv_dtype`).

## Architecture

49 layers of GQA attention — 32 query heads over 4 kv heads of width 128,
a per-head RMSNorm on q and k, a learned per-head attention sink on every
layer — 35 sliding-window layers (a 513-token window, RoPE over the whole
head) and 14 full-attention layers without RoPE (0, 1, 5, 9, 13, …, 45,
48). Layers 0–1 carry a dense SwiGLU MLP (6144); every other layer a
256-expert sigmoid-routed MoE (top 6, the picked scores renormalized,
no selection bias) plus an ungated BF16 shared expert (1024) added with
weight 1. Every norm rounds once (`y = bf16(x · rstd · w)`). No draft
head: the family has no MTP block, and `engine.mtp` is refused.

## Serving

One DGX Spark holds the weights at about 34 GiB; two and four nodes
(tensor parallel) split the attention heads (16 / 8 query heads over 2 /
1 kv heads per rank), the fused `[Q | K | V]` projection re-stacked per
rank, every expert's intermediate dimension (512 → 256 / 128), the dense
and shared MLPs and the vocabulary of the head. Decode runs a CUDA-graph
step: per layer the fused bf16 projection, one attention launch (the
finish with the q / k norms and the RoPE, the split partials, the
combine), the o_proj, and the MoE in the layer's formats — one decode
kernel for an FP8 pair, one for an NVFP4 pair and one for the mixed
pairs — with the ungated shared expert folded into the down projection's
epilogue and the residual add fused into each norm. Prefill attention is
query-tiled on the tensor cores; the routed experts run the grouped
tensor-core kernels per projection format. The tokenizer is the
three-stage byte-level BPE (number runs, CJK runs, the o200k case-class
pattern); the prompt renderer is the checkpoint's `chat_template.jinja`;
tool calls are emitted and constrained in the
`<tool_call>NAME<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`
format the template writes. The generation prompt ends at `<|assistant|>`
and the model opens its own `<think>` block: the family declares this to
the service (a prompt's tokens never decide it), so the parser splits the
block, the usage counts it and constrained decoding admits it before the
answer.

## Measured

See `docs/benchmarks.md` for the current numbers.

## License and attribution

The model, its weights, tokenizer and chat template are the upstream
release's; see the upstream model card for its terms.
