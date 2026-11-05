# Dory1.0-4B (nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4) on dgpp — architecture study and implementation plan (2026-10-18)

Status: **stages 1–8 landed, stage 9 in progress.** This document is the survey
and the design that the port is judged against (`docs/adding_a_model.md` §1).
Section 5 carries the staged plan and the progress markers; nothing below is
claimed as working beyond the landings recorded there.

The survey reads the checkpoint's `config.json` and safetensors headers
(1083 tensors, censused in §1.8) and the reference implementation:
the vLLM port in [vllm-project/vllm#60149](https://github.com/vllm-project/vllm/pull/60149)
(`vllm/model_executor/models/dory.py`, `vllm/transformers_utils/configs/dory.py`,
copies kept during the survey), which is the only executable description of
the architecture — `transformers` has no `dory` model directory yet, and the
checkpoint ships no remote code.

## 0. Summary

Dory1.0-4B (`DoryForCausalLM`, `model_type dory`) is a **normalized GPT**
(nGPT) whose 72 physical layers are **looped**: 16 input layers, a 40-layer
recurrent block applied `n_recurrent_loops` times (2 in the checkpoint) with
the input re-injected and the state re-normalized at every loop, then 16
output layers. Every physical layer is a **single sublayer** — attention or
MLP, never both (`layer_types` alternates `sliding_attention` / `mlp`,
with a `full_attention` block every 8 layers). There is no LayerNorm or
RMSNorm anywhere: normalization is nGPT's `justnorm` (row-wise L2 with
`eps = 1e-6`), and every residual is a **slerp** toward the sublayer output
(`slerp_norm(residual_out, sublayer_out, |alpha| * ngpt_alpha_init / initializer_range)`).

It is a small, dense, single-GPU model: 4 B parameters, hidden 2560, vocab
131 072, 16 query heads / 4 KV heads of `head_dim` 256, SwiGLU MLP of 7168,
256 K positions, all projections weight-only NVFP4 (W4A16) and the rest
BF16. The checkpoint is 2.982 GiB on disk. There is **no MoE, no MTP/draft,
no vision, no linear attention and no TP** (the reference refuses
`tensor_parallel_size != 1`).

The engine reuse is substantial — the NVFP4 W4A16 weight path, paged GQA
with a sliding window, the two-theta RoPE precedent (MiMo-V2.6), the session
core, the memory plan and the family interface all exist. New work: the
loop, the nGPT primitives, the per-sublayer walk, two recurrent K/V cache
policies, and the LM head's `logit_scale`.

## 1. The model

### 1.1 Configuration facts

| field | value |
|---|---|
| `architectures[0]` / `model_type` | `DoryForCausalLM` / `dory` |
| `hidden_size` / `vocab_size` / `num_hidden_layers` | 2560 / 131 072 / 72 |
| `n_input_layers` / `n_recurrent_layers` / `n_output_layers` | 16 / 40 / 16 |
| `n_recurrent_loops` | **2** (the checkpoint's own value) |
| `recurrent_kv_cache_mode` | absent from the checkpoint; `DoryConfig` default `"per_loop"` |
| `intermediate_size` / `hidden_act` | 7168 / silu |
| heads | 16 query / 4 KV, `head_dim` 256 → q 4096, kv 1024 |
| `layer_types` | `mlp` on odd `i`; else `full_attention` when `i % 8 == 6`, else `sliding_attention` — 36 mlp, 27 sliding, 9 full |
| `sliding_window` | 1024 (sliding layers only) |
| `rope_parameters` | per type: `sliding_attention` θ = 1e4, `full_attention` θ = 1e7, `rope_type` default, `partial_rotary_factor` 1.0 (neoX pairs) |
| `max_position_embeddings` | 262 144 |
| `tie_word_embeddings` | false |
| `initializer_range` | **0.0177** (normalizes every learned nGPT scale) |
| `ngpt_alpha_init` | 0.05 → residual scale `0.05 / 0.0177` = 2.824 858 8 |
| `attention_output_gate_logit_scale_init` | null → gate scale `sqrt(hidden_size)` |
| `attention_bias` / `mlp_bias` | false / false |
| `quantization_config` | modelopt `W4A16_NVFP4`, group_size 16, ignore `backbone.embeddings` + `lm_head`, `kv_cache_quant_algo` null |

Derived constants the model owns (fp32, computed from the config as the
reference computes them):

| constant | definition | value |
|---|---|---|
| `alpha_scale` | `ngpt_alpha_init / initializer_range` | 2.824 858 8 |
| `sqk_factor` | `head_dim**0.5 / initializer_range` | 903.954 80 |
| `s_gate_factor` | `sqrt(g0) / initializer_range`, `g0 = attention_output_gate_logit_scale_init or sqrt(hidden_size)` | 401.871 1 |
| `suv_factor` | `hidden_size**0.5` | 50.596 44 |
| `logit_factor` | `1 / initializer_range` | 56.497 18 |

### 1.2 The primitives

```
justnorm(x)      = bf16( fp32(x) * rsqrt(sum(fp32(x)^2, -1) + 1e-6) )
slerp_norm(a,b,l)= justnorm( justnorm_f32(a) + l * (justnorm_f32(b) - justnorm_f32(a)) )
slerp_alpha(a,b,α, s) = slerp_norm(a, b, |fp32(α) * s|)
```

Both operands of the slerp are normalized **before** the interpolation, and
the interpolation itself is fp32; the result is then normalized again. The
residual `a` is the sublayer input, `b` the sublayer output. This is the
only residual form in the model, so the `h += f(x)` accumulation the other
families use does not appear.

`justnorm` is applied at eight distinct sites, and the reduction width
differs by site (see §1.3–§1.5): the per-head `q`/`k` (256), the attention
output (4096 = all query heads), the MLP intermediate (7168), the slerp
operands and results (2560), and the loop's state combination (2560).

### 1.3 One attention block (`layer_types[i] != "mlp"`)

```
x    = justnorm-able input h (no pre-norm)
qkv  = qkv_proj(h)                       NVFP4 W4A16, bias-free: q 4096 | k 1024 | v 1024
gate = g_proj(h)                         NVFP4, 4096 wide
q, k = RoPE(head_dim 256, neoX, θ per layer type)
q    = ( (sqk ⊙ sqk_factor)^2 ⊙ justnorm_per_head(q) )        per head, fp32 sqk
k    = justnorm_per_kv_head(k)
attn = softmax(q k^T / sqrt(256)) v      paged GQA, 16 q heads / 4 kv heads
       causal; sliding layers mask to the last 1024 positions
out  = fp32(attn) ⊙ sigmoid( fp32(gate) ⊙ (s_gate ⊙ s_gate_factor)^2 )
out  = justnorm(out)                     width 4096, i.e. all query heads at once
o    = o_proj(out)                       NVFP4, 2560 wide
h'   = slerp_alpha(h, o, attn_alpha)     width 2560
```

Note the order the engine must reproduce: the `sqk` scaling is applied
**after** RoPE (unusual), the output gate multiplies **before** `justnorm`
and `o_proj`, and the `justnorm` over the attention output spans all 16
heads rather than one head.

### 1.4 One MLP block (`layer_types[i] == "mlp"`)

```
m   = gate_up_proj(h)                    NVFP4, 14336 wide: gate 7168 | up 7168
g   = fp32(g) ⊙ (suv_gate ⊙ suv_factor)
u   = fp32(u) ⊙ (suv_up   ⊙ suv_factor)
m   = silu(fp32(g)) ⊙ fp32(u)
m   = justnorm(m)                        width 7168
o   = down_proj(m)                       NVFP4, 2560 wide
h'  = slerp_alpha(h, o, mlp_alpha)       width 2560
```

`attn_alpha` and `mlp_alpha` are distinct parameter names on the two kinds
of layer, not a shared vector.

### 1.5 The looped backbone

```
h       = embeddings(input_ids)
for layer in layers[:16]:            h = layer(h)          # input block, 1 pass
input_h = h
state   = input_h
for i in range(n_recurrent_loops):
    h = justnorm(state + input_h)                          # re-inject, re-normalize
    for layer in layers[16:56]:      h = layer(h, i)       # shared 40-layer block
    state = justnorm(h)
h = state
for layer in layers[56:]:            h = layer(h)          # output block, 1 pass
```

The recurrent block's weights are shared across loops; only the attention
K/V caches differ (§2). `justnorm(state + input_h)` is an addition of two
normalized vectors followed by a re-normalization, not a slerp — it is
`state` and `input_h` both normalized (the loop's state already is) added
at equal weight.

### 1.6 Embedding, head, KV

* Embedding: plain BF16 table lookup, no scale, no norm.
* Head: `logits = fp32(h) @ fp32(lm_head.weight)^T ⊙ (logit_scale / initializer_range)`,
  `lm_head.weight` BF16 `[131072, 2560]` untied, `logit_scale` BF16 `[131072]`.
  The scale is per-vocabulary-entry, applied in fp32 after the projection.
* K/V: one paged plane per attention layer — one per loop under `per_loop`,
  one shared and overwritten by each loop under `last_loop` (`kv_repeats`) —
  holding the **full** history, with the sliding layers' 1024-token window
  applied at read time. §2 has the cost this implies, and why the engine
  holds to it rather than to the reference's per-request rings.

### 1.7 Tensor census

1083 tensors, 2.982 GiB: packed U8 NVFP4 1575 MiB, BF16 1282 MiB, E4M3
block scales 197 MiB, F32 per-tensor scales 288 × 4 B. Every key is
`backbone.*` or `lm_head.*`:

| group | tensors |
|---|---|
| embedding / head | `backbone.embeddings.weight` BF16 [131072, 2560]; `lm_head.weight` BF16 [131072, 2560]; `lm_head.logit_scale` BF16 [131072] |
| attention block (36, even `i`) | `backbone.layers.i.mixer.{q,k,v,o,g}_proj.weight` U8 with `.weight_scale` E4M3 (in/16 columns) and `.weight_scale_2` F32 (); BF16 `s_gate` [4096], `sqk` [4096]; `backbone.layers.i.attn_alpha` BF16 [2560] |
| mlp block (36, odd `i`) | `backbone.layers.i.mixer.{gate,up,down}_proj.weight` U8 + `.weight_scale` + `.weight_scale_2`; BF16 `suv_gate` [7168], `suv_up` [7168]; `backbone.layers.i.mlp_alpha` BF16 [2560] |

The NVFP4 layout is the modelopt one the engine already loads for the GLM
and Qwen NVIDIA releases: U8 with `in/2` columns (two e2m1 codes per byte),
an E4M3 scale grid of `in/16` columns, and one F32 per-tensor `weight_scale_2`.
There are **no `input_scale` tensors**: the release is W4A16, so activations
stay BF16 and no activation-quantization path is needed.

**Absent** (and therefore not implemented): MoE routers and experts, MTP /
draft layers, vision and audio towers, linear attention, TP slices, and any
`rope_scaling`.

## 2. Placement and cost (W = 1)

The reference refuses TP > 1 and the checkpoint is 3 GiB, so the family is
**one node, one rank** (`w1`); the loader takes no TP slices and the plan
assumes the whole checkpoint per rank.

Resident weights ≈ 2.98 GiB, plus the dequantization the loader chooses:
the engine may keep NVFP4 packed (the GLM/Qwen precedent) or expand to BF16
at load (≈ 8 GiB). The plan keeps them packed.

K/V is the only large variable, and the loop is what makes it interesting.
Bf16 K and V cost `kv_size × 2 × 2 B = 4096 B/token` per attention layer. The
pool is the engine's paged one (the GLM-4.7 / MiMo shape): **one plane per
attention layer**, one per loop under `per_loop` and one under `last_loop`
(`DoryConfig::kv_repeats`), each `token_slots` rows wide, so the pool costs
`planes × token_slots × 4096 B`:

| planes | `per_loop` | `last_loop` |
|---|---|---|
| 56 = 16 non-recurrent + 2 × 20 recurrent | 56 GiB | — |
| 36 = every attention layer once | — | 36 GiB |

at `token_slots` = 262 144 (256 K pooled tokens), and 14 GiB / 9 GiB at
65 536.

The sliding layers (27 of the 36) keep the **full history** and apply their
1024-token window at read time — the MiMo-V2.6 precedent, whose comment gives
the reason the engine holds to it: one block list restores a prefix on every
layer, so snapshots and prefix sharing stay exact. That is not what the
reference pays: its sliding layers wrap at 1024, which is where the vendor's
≈ 13 GiB (`per_loop`) / ≈ 8.7 GiB (`last_loop`) at 256 K comes from. The
engine's number is that plus the 42 sliding planes' 43 GiB — the price of the
shared paged pool, and the reason `recurrent_kv_cache_mode` and the pool's
size are engine keys rather than hard-coded choices (§3, §6).

## 3. Design decisions

**D1 — the loop count is the checkpoint's, overridable by an engine key.**
`n_recurrent_loops` is in the checkpoint's `config.json` (2), so the exact
path is the config's value and `engine.dory_loops` overrides it for the
vendor's compute/accuracy trade (1, 3, 4). It changes what the engine
computes, so it is an `engine.*` key, plumbed the seven ways of
`adding_a_model.md` §4.

**D2 — the recurrent K/V mode is an engine key.**
`recurrent_kv_cache_mode` is absent from the checkpoint; the architecture's
default is `per_loop` (one cache per loop, joined). `engine.dory_kv_cache`
takes `per_loop` / `last_loop`, defaulting to `per_loop` — the exact,
lossless path — with `last_loop` documented as the memory lever (§2: the 20
recurrent layers' second copy, 20 GiB at a 256 K pool) and its measured
transcript cost recorded before it is recommended anywhere.

**D3 — weight-only, packed NVFP4, exact.** The release is W4A16 with no
`input_scale`; the loader keeps the packed e2m1 weights, the E4M3 block
scales and the F32 per-tensor scale, exactly as the GLM-4.7 / Qwen NVFP4
path does, and computes in BF16 activations. No load-time requantization.

**D4 — no MTP.** The checkpoint has no draft layer, so the family has no
speculative path; the deploy template is a plain world and the family
refuses the drafter keys by name rather than silently ignoring them.

**D5 — one physical layer, one sublayer.** The engine's layer walk must not
assume `attention + MLP` per index; the family's walk splits attention and
MLP blocks, which also makes the recurrent block a clean 40-entry list.

**D6 — sliding window per layer.** 27 of 36 attention layers carry the
1024 window; the family declares it per layer (the MiMo precedent) rather
than globally.

**D7 — two RoPE thetas.** θ = 1e4 on the sliding layers, 1e7 on the full
layers, neoX pairs over the full 256-dim head (`partial_rotary_factor` 1.0).
This is the MiMo-V2.6 precedent (1e7/1e4) with a different assignment rule.

**D8 — `justnorm` at the raggeder widths.** The eight sites in §1.2 need a
row-wise L2-with-eps kernel that accepts the width as a parameter (256,
2560, 4096, 7168) and an fp32 accumulation; the slerp needs the composed
form (normalize both, lerp, normalize). Both are new kernels with host
references.

**D9 — the head's per-token `logit_scale`.** The vocabulary-wide BF16 vector
multiplies the fp32 logits before sampling. New for this family.

## 4. Reuse, adapt, new

Reused as-is: the modelopt NVFP4 decode/GEMM kernels (`fp4_gemv`,
`fp4w_gemm`, `dense_fp4_decode`) and the E4M3/F32 scale handling; the paged
GQA attention with a per-layer sliding window (the MiMo-V2.6 kernels); the
two-theta RoPE; the GQA prefill and decode paths; the session core (graph
capture, snapshot, prefix arena, memory plan, `ServeFamily`); the BF16 GEMV
for the head; the tokenizer/Jinja renderer.

Adapted: the family config/binding/loader/model; the K/V pool (a per-layer,
per-repeat pool with a `per_loop` / `last_loop` policy); the layer walk (one
sublayer per index); the memory plan (the loop's cache multiplicity).

New: `justnorm` and `slerp_norm` kernels plus host references; the
`sqk`/`s_gate`/`suv` scaling sites; the loop driver and its state
combination; the `logit_scale` head; the recurrent K/V policy.

## 5. Implementation stages

1. **Detection, config, census** — `ModelArchitecture::Dory`, the parser
   (`models/dory/config.{hpp,cpp}`) rejecting every field outside §1.1 by
   name, the tensor census test against a tiny synthetic checkpoint.
   **Landed** (commits 1b8a77d0, 8e253364).
2. **Binding and loader** — the binding table, the packed NVFP4 slices, the
   BF16 vectors, the resident image, and the census test in the release's
   own layout (the `adding_a_model.md` §2 rules, including the exhaustive
   `is_replicated` switch). **Landed**: the binding table and its test
   (`models/dory/binding.{hpp,cpp}`, `tests/unit/dory_binding_test.cpp`) and
   the resident loader (`models/dory/loader.{hpp,cpp}`, the
   `dgpp_models_dory_loader` library). The census test in the release's own
   layout through the model's own `dory_forward_test --write-fixture` is
   stage 8. Validated 2026-10-18 against the landed release header:
   `dory_binding_matches_the_landed_checkpoint` reads the real
   `model.safetensors` and matches all 1083 tensors by name/dtype/shape — 288
   of them NVFP4 matrices (36 attention blocks × 5, 36 MLP blocks × 3) — with
   none unexpected.
3. **Primitives** — `justnorm`/`slerp_norm` kernels + host references +
   kernel tests, with the fp32 accumulation and the `eps` pinned.
   **Landed**: the host references and their test
   (`models/dory/reference.{hpp,cpp}`, `tests/unit/dory_reference_test.cpp`,
   commit 007d40b7) and the CUDA kernels, including the fused residual
   (`kernels/dory_norm.{hpp,cu}`, `tests/cuda/dory_norm_test.cu`). The kernels
   sum in fp64 — every bf16 square is exact there, so the sum is exact and
   `float(sum) + eps` rounds once, which is the reference's one-rounding
   contract — and build with `-fmad=false` so the residual combine rounds as
   the host's `-ffp-contract=off` does. The test compares them bitwise against
   an fp64-exact oracle and, against the transcription's own fp32 chain,
   within 2% of the case scale with 0.03% of the values differing (its
   reduction error, ~sqrt(dim)·2^-24). 6.3 us for a 4096-wide justnorm and
   13.5 us for the fused residual on one Spark.
4. **The layer walk** — one sublayer per index, the attention and MLP
   numerics of §1.3–§1.4, the two RoPE thetas, the sliding window.
   **Landed.** First the attention kernels (`kernels/dory_attn.{hpp,cu}` —
   the per-(row, head) finish with the neoX RoPE, the `sqk` square and
   `justnorm_per_head`, the K/V cache writes, the split-KV partial and the
   combine) and the two fused sites (`dory_gate_justnorm_bf16`,
   `dory_silu_justnorm_bf16` in `kernels/dory_norm.{hpp,cu}`), with their
   tests (`tests/cuda/dory_attn_test.cu`, the extended
   `tests/cuda/dory_norm_test.cu`): q and k are bitwise against the host
   oracle and the attention output is within 0.4% of its values at 2 ulp.
   One decode row at the model's geometry over 2048 tokens costs 8.2 us
   (finish) + 46.3 us (partial, `n_split` 16 — the split saturates there) +
   4.1 us (combine) on one Spark. Then the paged pool
   (`models/dory/kv_pool.{hpp,cpp}`: one plane per attention layer per
   `kv_repeat`, zero on MLP layers) and the layer objects
   (`models/dory/layers.{hpp,cpp}`: one `DoryAttentionLayer` /
   `DoryMlpLayer` per kind, rebound per physical layer with its own window
   and RoPE theta, an fp4 GEMV per Linear on decode rows and the dense MMA
   on prefill rows), composed by the walk of §1.5 and the head of §1.6 in
   `models/dory/forward.{hpp,cpp}`. `tests/cuda/dory_forward_test.cpp`
   drives a tiny 8-layer fixture (`tests/cuda/dory_fixture.hpp`): a repeated
   forward is bitwise application by application, a T-1 prefix is bitwise
   against the T-token walk, the capture yields one state per sublayer
   application, the window changes a third of the logits, the two K/V
   policies give 6/4 planes, the loop override changes every logit, and a
   decode row matches a fresh prefill's last row to 2.3e-7 relative L2.
   One real bug fell out of that first run and is fixed: `rebind` uploaded
   the inverse-frequency table with a host-side *blocking* `cudaMemcpy`
   while the walk's stream is non-blocking, so a kernel could read another
   layer's thetas (the K cache came out rotated differently while V — and
   token 0, whose angle is 0 — stayed bitwise identical). The upload is now
   stream-ordered, and the layer-state capture stages its rows on the device
   and flushes once, so a capture walk keeps a plain walk's arithmetic.
5. **The loop** — the input/recurrent/output partition, the re-injection
   and state normalization, and the two K/V policies. **Landed** as the
   §1.5 loop in `models/dory/forward.cpp`: `h = justnorm(state + input_h)`
   once per loop, the recurrent block walked with one K/V plane per
   (layer, loop) under `per_loop` and one shared under `last_loop`,
   `state = justnorm(h)`.
6. **The head** — `logit_scale`, and the lm head on the shared path.
   **Landed**: the BF16 lm-head GEMM with fp32 accumulators, then
   `kernels/dory_head.{hpp,cu}` applies `logit_scale · logit_factor` to the
   fp32 logits in place.
7. **Serve wiring and template** — the `ServeFamily` (name, model build,
   resident-image dir, refused keys), `deploy/cluster_dory1.0-4b_nvfp4_w1.example.json`,
   the `portability_test.py` model map and the `deploy/README.md` row.
   **Landed** (2026-10-18): the `dory` family in `apps/dgpp_serve.cpp`
   (resident, one rank — `world > 1` refuses before the journal, MTP and
   the drafter keys refuse by name, plan D4), the two keys
   (`engine.dory_loops`, `engine.dory_kv_cache`) plumbed the seven ways
   (§4 of `adding_a_model.md`; the family never forms a peer world, so
   the settings-record fields ride along for the uniform rule alone), the
   w1 template, the portability map entry (with the draftless-family
   exception the template needs) and the deploy row. The checkpoint's
   generic transformers chat template needed three jinja-subset additions
   (the `for x in ys if cond` inline filter, mapping-key iteration, the
   `sequence` test) — gated in `chat_template_test`. Its tokenizer's
   Nemotron CamelCase pattern joined the pinned scanner set
   (`tokenizer.cpp`, `unicode_tables.hpp` rebuilt with the Lu/Lt/Ll split).
8. **Gates** — the `dory_forward_test` fixture ladder (§6 of
   `adding_a_model.md`): `--write-fixture` from the binding table itself,
   host-reference `--smoke` / `generate` / `decode` rows.
   **Landed** (2026-10-18): `--smoke DIR` and `--reference DIR` (the
   generate and decode rungs in one binary), the oracle being the CPU
   fp64 host walk of `tests/cuda/dory_host_reference.hpp` — raw
   safetensors, the binding's format contract, the §1.3–§1.6 formulas,
   sharing no code with the CUDA path beyond the norm primitives the
   kernel tests already pin. The fixture's logits match within 0.008
   relative L2 / 0.9 % of peak, prefill and decode alike. Two real bugs
   fell out of the first real-checkpoint run: the MLP down projection's
   K = 7168 was not in the packed GEMV's compiled set (added), and the
   walk's per-rebind theta upload plus its three device-to-device staging
   copies made 59 memcpy nodes in the decode-graph capture — both are
   kernels now (the two theta tables stage at build;
   `dory_copy_rows_bf16`), and the release warm-captures kernels-only.
9. **Validation** — `cmake --preset ci` clean with `-Werror`, the full
   serial `ctest` on an idle Spark with master merged, and the greedy
   transcript battery at T=1 and with the decode graph.

Stages 1–3 are pure host/kernel work; stages 4–6 are the model; 7–9 are the
landing.

## 6. Risks and open questions

* **The attention-output `justnorm` spans all 16 heads** (width 4096), not a
  single head. A per-head implementation is a plausible silent error; the
  host reference must pin the width.
* **`justnorm` after RoPE but before the `sqk` square** — the reference's
  literal order. Any reordering changes the numerics.
* **`per_loop` K/V multiplicity** is the difference between 56 GiB and
  36 GiB of pool at a 256 K `token_slots` (§2). The default is `per_loop`
  (exact), so a long-context deployment is a deliberate keyed decision, not
  a default.
* **The sliding layers' full history** (§2) is 42 of those 56 planes: the
  reference wraps the 27 sliding layers at 1024 and the engine does not. A
  per-request ring for them — `Csa2StatePool`'s pattern (`ring_slots` rows,
  `slot = pos % ring_slots`) with a second read path in the attention
  kernel — would cut the 256 K pool to ≈ 13 GiB, at the price the dsv41
  comment names: the rings are positional and would not be snapshot state.
  A separate decision, not a tweak, and out of scope for this port.
* **No MTP** means no speculative battery; the plain-world transcript is
  the reference for the decode graph.
* **Dense NVFP4 W4A16 linears** are exercised by the GLM-4.7 / Qwen NVIDIA
  releases; confirm the packed decode path covers a 4096-wide `g_proj` and
  a 14336-wide fused `gate_up_proj` before stage 2 lands.
* **The chat template and parsers** (the card names `nemotron_v3` reasoning
  and `qwen3_coder` tools) are not part of the architecture; the engine's
  renderer/parser choices for this family are a separate decision to settle
  before the template ships.
* **Looping and prefix caching / snapshots**: the recurrent block's state
  is a function of the whole prefix, so a snapshot must be taken at a loop
  boundary the scheduler's cut list contains, and the `per_loop` caches
  must snapshot together. This is the subtlest interaction in the port.
