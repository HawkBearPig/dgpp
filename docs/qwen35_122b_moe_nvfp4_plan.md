# Qwen3.5-122B-A10B-NVFP4 on dgpp — architecture study and implementation plan (2026-10-07)

Status: port in progress on `qwen35-122b-moe-nvfp4` (from PR #95 `e28d8a2` + `origin/master` `07a32fb` = `70419f9`). Text path only; vision ignored like the dense 27B family.

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

Checkpoint bytes: BF16 18.2G (attn/GDN/shared/router/norms/embed/head/MTP-BF16 incl 4.6G MTP experts), F8 7.2G (scales), U8 58.0G (packed fp4). Vision 333 tensors ignored. KV FP8 scheme declared but engine KV stays BF16 (dense precedent: count `k_scale`/`v_scale`, never resident).

Worlds: weights 83G → TP=1 needs ~90G+KV (~6.4G @262k: 12 Full ×2×2×256×2B×262k) ≈ 100G, marginal on 128G Spark; public targets TP=2/TP=4 (41/21G weights per rank). TP geometry: GDN heads 16/64, Full 32/2, MoE inter 1024 and shared 1024 must divide W (1024%4=0 ok); kv 2 < W handled by existing `kv_head_begin=rank/(W/kv)` rule. Router/shared-gate replicated; experts/shared sliced.

## 1. Tensor census (from safetensors headers + `model.safetensors.index.json`)

Per main layer L (0..47): `input_layernorm`/`post_attention_layernorm` BF16 `[H]`; GDN (36 layers): `A_log`/`dt_bias` `[Vh]`, `conv1d` `[2K+V,1,4]`, `in_proj_a/b` `[Vh,H]`, `in_proj_qkv` `[2K+V,H]`, `in_proj_z` `[LV,H]`, `norm` `[128]`, `out_proj` `[H,LV]` all BF16; Full (12 layers): `q_proj` `[2QH,H]` (attn_output_gate), `k/v_proj` `[KVH,H]`, `o_proj` `[H,QH]`, `q/k_norm` `[256]` all BF16; MoE: `mlp.gate` `[256,H]` BF16, `mlp.shared_expert_gate` `[1,H]` BF16, `shared_expert.{gate,up}_proj` `[1024,H]` + `down_proj` `[H,1024]` BF16, `experts.e.{gate,up}_proj` U8 `[1024,1536]` + F8 `[1024,192]` + 2×F32, `down_proj` U8 `[3072,512]` + F8 `[3072,64]` + 2×F32 (e=0..255). Globals: `embed` `[248320,H]`, `lm_head` `[V,H]`, `language_model.norm` `[H]` BF16. MTP `mtp.layers.0.*`: same Full + MoE shapes but experts BF16 `[1024,3072]`/`[3072,1024]` (785 tensors incl `mtp.fc`/`norm`/`pre_*`). `QwenMoeLayer` contract matches: softmax top-k, renorm, swiglu experts, sigmoid shared tail, ascending-fp32 chain.

Naming gap vs dense-NVFP4: ModelOpt (`.weight/.weight_scale/.weight_scale_2/.input_scale`) not compressed-tensors (`.weight_packed/.weight_scale/.weight_global_scale/.input_global_scale`). `loader.cpp:621` (`load_fp4_*_mo`) already speaks ModelOpt; `loader35.cpp` only speaks compressed-tensors. Config gap: `quant_method=modelopt` + `quant_algo=NVFP4`, `config_groups.group_0.{weights{4b,group16},targets:[Linear]}` with no `format` field; `hf_quant_config.json` mirrors (`NVFP4`/`FP8` kv, `exclude_modules`). `config35.cpp` demands `compressed-tensors` + `format`, `intermediate_size`, `model_type=qwen3_5_text` — all three reject 122B today.

## 2. Placement decisions

* Family: extend `Qwen3_5` (`Qwen35TextConfig` + `is_moe`, `moe_*`, `modelopt_nvfp4` quant kind), not a new `ModelArchitecture` — arch prefix already routes here; serve wiring `Qwen35Family` unchanged. Reuse GDN/Full kernels + `QwenMoeLayer` (routed NVFP4 + BF16 shared, `routed_config(H,1024,256,8,true)`).
* Binding: `expect_moe35` per layer (Router/Shared/Routed classes, ModelOpt quad names); MTP expects BF16 MoE. `is_replicated_35` gains Router→true (match `loader.cpp`), Shared/Routed→false.
* Loader: `load_fp4_*_modelopt` beside `load_fp4_*` (payload `.weight`, scales `.weight_scale`, global `.weight_scale_2`, consume `.input_scale`); TP slices `I/W`, `S/W`; globals F32 divisor (kernels divide once, dense § loader35 `load_fp4_global` precedent).
* Model: `Qwen35DenseMlpResident` → union with `QwenMoeResident`-like (router/shared/experts_fp4 + globals); `dense_mlp` → `moe_enqueue` (`enqueue_decode`/`enqueue_prefill`, same chunk-invariance/graph rules as `QwenMoeLayer`). MTP: load BF16 MoE but serve `mtp:false` + DFlash first (BF16 routed has no kernel; fused-FP8 encode would be lossy — keyed decision, not default).
* Keys: exact BF16-act NVFP4 by default; no new `engine.*` lever in v1 (dense levers `prefill_fp8_per_tensor`/`dense_weights:fp8` refused under NVFP4, same rule).

## 3. Port record

* 2026-10-07: survey (this doc), branch `qwen35-122b-moe-nvfp4` from PR95+master.
* Next: config+binding unit gates on real snapshot → loader smoke (layer 0 + MTP skipped) → fixture/reference-dump → forward parity vs transformers → templates/card → `ci` suite + transcript battery + fabric md5 + campaign.

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
the wrong combinations by key name. Live: `mtp:true` boot, prompts
identical to plain (391/Paris/1-10), templates ship MTP depth 2.

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
  real, every expert runs every row); top-k selection itself rides the
  shared MoE kernels plus live-checkpoint routed answers.
* Softmax weight-noise amplification (dw/w ~ ||gate||·dx): wide routers
  turn the inherited ~5-ulp bf16 floor into percent-level weight noise;
  rms 1.0 at H=256 (logits spread ~5) threads it.
* Reference replicates the kernel's bf16-rounded router logits
  (`launch_moe_router` leaves bf16 LOGITs in scores).
* Group invariance holds under a 1e-5 budget for Nvfp4Moe (grouped
  expert kernels reduce in batch-shape order; one 2e-6 observation, no
  flips, current fixture passes bitwise).
