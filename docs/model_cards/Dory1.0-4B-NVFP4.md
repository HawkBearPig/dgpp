# Dory1.0-4B on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4).
It is **not** a republished checkpoint: the engine loads the upstream
release as shipped, with no re-quantization pass.

## The checkpoint, as loaded

One 2.982 GiB image, 1083 tensors: the packed NVFP4 (modelopt W4A16,
group 16) keeps its e2m1 codes, its E4M3 block-scale grid and its F32
per-tensor globals exactly as shipped; the embedding, the lm head, the
per-vocabulary `logit_scale` and the learned nGPT scales are BF16. There
are no `input_scale` tensors — weight-only, so activations stay BF16 and
no activation-quantization path exists. See the
[architecture plan](../dory_plan.md) for the tensor census.

## Architecture

A **looped normalized GPT** (`DoryForCausalLM`, `model_type dory`): 72
physical layers of hidden 2560, each a single sublayer — attention (even)
or MLP (odd), never both. 16 input layers, a 40-layer recurrent block
applied `n_recurrent_loops` = 2 times with the input re-injected
(`h = justnorm(state + input_h)`) and the state re-normalized at every
loop, then 16 output layers. There is no LayerNorm anywhere: the only
normalization is nGPT's `justnorm` (row-wise L2, eps 1e-6), and every
residual is `slerp_alpha` toward the sublayer output — no `h += f(x)`.
Attention is GQA (16 query / 4 kv heads of 256 dims) with the nGPT qk-norm
applied **after** RoPE (sliding layers θ 1e4, full ones θ 1e7; 27 sliding
layers with a 1024-token window, 9 full), a learned `sqk` scale squared
into q, a `s_gate` output gate before a justnorm that spans all 16 heads,
and the head scaled per vocabulary entry (`logit_scale · logit_factor`).
The plan documents the reference this reads from (vLLM PR #60149); the
checkpoint ships no remote code.

What the engine serves: one rank (the reference refuses
`tensor_parallel_size != 1` and the release is a single image), resident
weights, a paged BF16 K/V pool with one plane per attention layer per
loop, chunk-invariant prefill, prefix caching, the decode graph, and the
sampled/packed-batch engine machinery. **No MTP/draft layer exists** (the
plan D4): speculation keys are refused by name, the served world is plain.
No MoE, no vision, no linear attention, no `rope_scaling`.

## Engine keys

- `engine.dory_loops` — override the loop count (default: the
  checkpoint's 2). Changes what the engine computes.
- `engine.dory_kv_cache` — `per_loop` (the default, exact) or
  `last_loop` (one shared plane per layer, the earlier loops' rows gone:
  about half the recurrent pool, lossy for decode and prefix reuse, a
  cold prefill is unchanged).

Both are documented in the [configuration reference](../configuration.md)
and the [operations notes](../operations.md#dory10-4b-loop-options); the
[one-node template](../../deploy/cluster_dory1.0-4b_nvfp4_w1.example.json)
pins both explicitly: the checkpoint's loop count of 2 and the exact
`per_loop` cache.

## Validation

The forward gate runs the model over a tiny release-layout fixture
generated from the binding table (`dory_forward_test --write-fixture`)
and checks every logits row — prefill and a session decode step — against
a CPU fp64 host-reference walk transcribed from the plan (worst
divergence 0.9 % of the logits' peak, the bf16 sites' own noise), plus
determinism, causality, window, policy and loop rungs. The kernels are
each pinned against the host primitives of `models/dory/reference.hpp`.
The 2026-10-18 boot smoke on the release: a 1.7 s boot to listening, a
48-token greedy generation bitwise reproducible across runs. The
transcript battery at the release's shapes is the landing gate
([plan](../dory_plan.md) §5 stage 9).

The chat template is the checkpoint's own (the transformers-generic one,
with the `nemotron_v3` reasoning and `qwen3_coder` tool conventions the
model card names); the engine's jinja subset renders it directly — the
generic template's `for x in m if cond` inline filter, mapping-key
iteration and `sequence` test are engine features, gated in
`chat_template_test`.
