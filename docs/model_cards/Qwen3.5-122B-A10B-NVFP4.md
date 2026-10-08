# Qwen3.5-122B-A10B-NVFP4

Serving note for [nvidia/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-122B-A10B-NVFP4)
on DGPP. Model architecture and weights belong to their authors; this page
records only how DGPP serves this checkpoint and what was measured here.

The MoE sibling of [Qwen3.8-27B-NVFP4](Qwen3.8-27B-NVFP4.md): the same
gated-delta-net + full-attention hybrid (3:1, interval 4, swish GDN gate,
partial-rotary MROPE + per-head output gate), 3072 hidden, 48 layers
(36 GDN + 12 Full), 256K vocab, 262144 context. Each layer carries a
256-expert top-8 MoE (softmax + renorm, `moe_inter` 1024) plus a gated
shared expert (1024, sigmoid gate). The MTP draft layer (one Full + MoE
head) ships per-expert BF16: `engine.mtp_expert_format=bf16` encodes each
draft expert to block FP8 at load (proposals lossy, verify exact); without
the key the draft is refused by name and v1 serves `mtp: false`.

The release is ModelOpt NVFP4 for the routed experts only
(`weight` U8 packed e2m1 + `weight_scale` F8-E4M3 per 16 +
`weight_scale_2` F32 global stored reciprocal, kernels divide once;
`input_scale` consumed, unused under exact BF16 activations).
Everything else — attention/GDN, shared experts, router, norms,
embeddings, `lm_head`, the MTP experts (BF16) — is BF16. The declared
FP8 `kv_cache_scheme` scales are absent from the checkpoint; the
engine's KV stays BF16.

## Formats in DGPP

DGPP reads the release directly (no requantization). `config_groups`
claims `Linear` for NVFP4; the ModelOpt glob `ignore` list keeps
attention, shared experts, `lm_head`, vision and `mtp.*` BF16, so only
`mlp.experts.*` bind NVFP4. `prefill_fp8_per_tensor` and
`dense_weights: fp8` are refused under this quantization (the FP8
release's levers).

TP: router and norms replicate; GDN heads (16/64), Full heads (32/2),
routed (1024) and shared (1024) slices divide the world. Checkpoints are
~83.5G; point templates at a resident cache outside the HF snapshot dir.

## Gates

`qwen35_bind_check` pins the table (148976 tensors, 333 vision skipped).
Forward parity vs the transformers reference and the greedy transcript
battery (plain vs spec, graph on/off) run before any number is quoted;
the dated campaign lives under `benchmarks/results/`. No serving numbers
are claimed yet.
