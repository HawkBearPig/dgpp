# Qwen3.5-122B-A10B-NVFP4 on dgpp — architecture study and implementation plan (2026-10-07)

Status: integrated with the current Qwen35 family, including its FP8 KV,
vision configuration, shared prefill and recorded decode interfaces. The
model card documents serving options; full-checkpoint quality and template
performance validation remain separate from the synthetic regression gates.

References: `nvidia/Qwen3.5-122B-A10B-NVFP4` (cached `98915d83`, 83.5G `total_size`, 9 shards, 149309 tensors), transformers 5.17 `modular_qwen3_5_moe.py` (`Qwen3_5Moe*` over `Qwen3Next*` over `Qwen2Moe*`), `docs/adding_a_model.md`, dense sibling card `docs/model_cards/Qwen3.8-27B-NVFP4.md`.

## 0. Summary

122B-A10B is the MoE sibling of the dense 27B family (`Qwen3_5ForConditionalGeneration` → `Qwen3_5MoeForConditionalGeneration`, `qwen3_5` → `qwen3_5_moe`, text `qwen3_5_text` → `qwen3_5_moe_text`). Same hybrid attention (GDN `linear_attention` + Full GQA `full_attention`, 3:1, interval 4, swish GDN output gate, Full partial-rotary MROPE + per-head output gate `[q|gate]`), same vocab 248320, same nesting (`model.language_model.layers.L`, `mtp.layers.0`, globals `embed_tokens`/`lm_head`/`language_model.norm`).

Differences from dense 27B (hidden 5120, 64 layers, dense SwiGLU 17408):

| field | 122B MoE | 27B dense |
|---|---|---|
| hidden / layers | 3072 / 48 (36 GDN + 12 Full) | 5120 / 64 |
| GDN | 16 key ×128, **64** value ×128 (dense 48), conv 4, fp32 state | 16×128, 48×128 |
| Full | **32** q, **2** kv, head 256 (dense 24/4) | 24/4×256 |
| MoE | 256 experts, top-8, softmax renorm (`norm_topk_prob` default true), `moe_inter` 1024, shared 1024 + sigmoid gate `[1,H]` | dense SwiGLU, no MoE |
| MTP | 1 Full layer, MoE BF16 experts (256×3 BF16) + shared BF16 | 1 Full layer, dense BF16 |
| quant | ModelOpt NVFP4, **routed experts only** (`weight` U8 + `weight_scale` F8_E4M3 group16 + `weight_scale_2` F32 + `input_scale` F32); rest BF16 | compressed-tensors mixed (dense MLP NVFP4 + attn/head FP8-channel) |

Checkpoint bytes: BF16 18.2G (attn/GDN/shared/router/norms/embed/head/MTP-BF16 incl 4.6G MTP experts), F8 7.2G (scales), U8 58.0G (packed fp4). The original text census excluded 333 vision tensors. The current family loads a configured vision tower separately. The release declares an FP8 KV scheme but lacks K/V scale tensors; the deployment templates select BF16 KV.

Worlds: weights 83G → TP=1 needs ~90G+KV (~6.4G @262k: 12 Full ×2×2×256×2B×262k) ≈ 100G, marginal on 128G Spark; public targets TP=2/TP=4 (41/21G weights per rank). TP geometry: GDN heads 16/64, Full 32/2, MoE inter 1024 and shared 1024 must divide W (1024%4=0 ok); kv 2 < W handled by existing `kv_head_begin=rank/(W/kv)` rule. Router/shared-gate replicated; experts/shared sliced.

## 1. Tensor census (from safetensors headers + `model.safetensors.index.json`)

Per main layer L (0..47): `input_layernorm`/`post_attention_layernorm` BF16 `[H]`; GDN (36 layers): `A_log`/`dt_bias` `[Vh]`, `conv1d` `[2K+V,1,4]`, `in_proj_a/b` `[Vh,H]`, `in_proj_qkv` `[2K+V,H]`, `in_proj_z` `[LV,H]`, `norm` `[128]`, `out_proj` `[H,LV]` all BF16; Full (12 layers): `q_proj` `[2QH,H]` (attn_output_gate), `k/v_proj` `[KVH,H]`, `o_proj` `[H,QH]`, `q/k_norm` `[256]` all BF16; MoE: `mlp.gate` `[256,H]` BF16, `mlp.shared_expert_gate` `[1,H]` BF16, `shared_expert.{gate,up}_proj` `[1024,H]` + `down_proj` `[H,1024]` BF16, `experts.e.{gate,up}_proj` U8 `[1024,1536]` + F8 `[1024,192]` + 2×F32, `down_proj` U8 `[3072,512]` + F8 `[3072,64]` + 2×F32 (e=0..255). Globals: `embed` `[248320,H]`, `lm_head` `[V,H]`, `language_model.norm` `[H]` BF16. MTP `mtp.layers.0.*`: same Full + MoE shapes but experts BF16 `[1024,3072]`/`[3072,1024]` (785 tensors incl `mtp.fc`/`norm`/`pre_*`). `QwenMoeLayer` contract matches: softmax top-k, renorm, swiglu experts, sigmoid shared tail, ascending-fp32 chain.

ModelOpt uses `.weight/.weight_scale/.weight_scale_2/.input_scale`, unlike
the dense compressed-tensors release. The loader now handles both naming
schemes. ModelOpt `config_groups` with `targets: ["Linear"]` and glob
`ignore` entries identify the routed experts; unsupported formats are
rejected during configuration parsing.

## 2. Placement decisions

* Family: extend `Qwen3_5` (`Qwen35TextConfig` + `is_moe`, `moe_*`, `modelopt_nvfp4` quant kind), not a new `ModelArchitecture` — arch prefix already routes here; serve wiring `Qwen35Family` unchanged. Reuse GDN/Full kernels + `QwenMoeLayer` (routed NVFP4 + BF16 shared, `routed_config(H,1024,256,8,true)`).
* Binding: `expect_moe35` per layer (Router/Shared/Routed classes, ModelOpt quad names); MTP expects BF16 MoE. `is_replicated_35` gains Router→true (match `loader.cpp`), Shared/Routed→false.
* Loader: `load_fp4_*_modelopt` beside `load_fp4_*` (payload `.weight`, scales `.weight_scale`, global `.weight_scale_2`, consume `.input_scale`); TP slices `I/W`, `S/W`; globals F32 divisor (kernels divide once, dense § loader35 `load_fp4_global` precedent).
* Model: `Qwen35DenseMlpResident` → union with `QwenMoeResident`-like (router/shared/experts_fp4 + globals); `dense_mlp` → `moe_enqueue` (`enqueue_decode`/`enqueue_prefill`, same chunk-invariance/graph rules as `QwenMoeLayer`). MTP: encode the per-expert BF16 draft to block FP8 only with `mtp_expert_format: bf16`; target verification retains the checkpoint weights.
* Target routed experts use BF16 activations with NVFP4 weights. The dense-family conversion options remain restricted to their supported checkpoint formats.

## 4. MTP draft (2026-10-07)

The checkpoint's draft is Full + MoE with per-expert BF16 (router
`[256,3072]`, shared + 256×3 experts ≈ 4.6 GB BF16) — no native quantized
form, and `QwenMoeLayer` serves routed FP8/NVFP4/packed only. Decision:
`engine.mtp_expert_format=bf16` (qwen3_5-MoE-only) encodes each draft
expert's slice to block FP8 at load (`load_bf16_rows/cols_fp8`, the
`bf16_fused` precedent); proposals lossy, verify exact. Refused:
MoE+mtp without the key, dense+`bf16` (dense draft is dense BF16,
serves as shipped). Loader format bit 8 carries the key. Model:
`mtp_run_rows` rides `moe_mlp` on the draft resident (route-table slot
`num_hidden_layers`), `graph_prepare` harvests it; ctor + plan refuse
the wrong combinations by key name. Templates select MTP depth 2. Shared-prefill tests exercise MTP on and off, including graph token parity.

## 5. Forward parity (2026-10-08)

Tiny fixture: 4 layers, H=256, 8 experts top-8-of-8, ModelOpt NVFP4
routed experts, BF16 rest (incl. MTP draft experts). E2e relaxed +
teacher strict both green (top-1 exact, top-8 exact under teacher).

Two fixture-design findings, both verified by bisection (engine stage
dumps vs reference submodules) and both closed in the fixture, not the
engine (every isolated comparison — GDN, Full, NVFP4 dequant, routing,
shared tail, accumulation — matches; layer 0 and Full layers go exact):

* Routing knife-edges: random routers leave rows where the kernel's
  bf16 logit rounding flips 2nd-vs-3rd against an fp64 reference (both
  correct, outputs totally different; observed as whole-row 0.2-0.7
  errors with no matching expert pair). Top-8-of-8 removes the
  selection boundary entirely (full softmax + renorm still divide for
  real, every expert runs every row); top-k selection itself is covered by the shared MoE kernel tests.
* Softmax weight-noise amplification (dw/w ~ ||gate||·dx): wide routers
  turn the inherited ~5-ulp bf16 floor into percent-level weight noise;
  rms 1.0 at H=256 (logits spread ~5) threads it.
* Reference replicates the kernel's bf16-rounded router logits
  (`launch_moe_router` leaves bf16 LOGITs in scores).
* Group invariance holds under a 1e-5 budget for Nvfp4Moe (grouped
  expert kernels reduce in batch-shape order; one 2e-6 observation, no
  flips, current fixture passes bitwise).
