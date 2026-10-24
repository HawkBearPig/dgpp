# Flint (HawkBearPig/flint-FP8-NVFP4) on dgpp — architecture facts and implementation plan (2026-10-22)

Status: the port record is section 6. The checkpoint
`HawkBearPig/flint-FP8-NVFP4` (revision `95e74655`) is the mixed-precision
export of `RecursiveSI/flint` (revision `2a247be0`): 85,886 tensors in 48
shards (one per MoE layer plus `model-non-experts.safetensors`), 35.74 GB
of tensor payload. The architecture study uses its `config.json`, the
safetensors headers, the export's model card and precision policy, the
release's `flint_sglang` 0.1.4 `model.py` and the torch reference the
quantization study pinned (one bf16 rounding after every Linear, the
fused SwiGLU, the sink softmax).

## 0. Summary

Flint (`FlintForCausalLM`, `model_type flint`) is a 49-layer hybrid-attention
MoE of 39.8 B parameters: layers 0 and 1 carry a dense SwiGLU MLP (6144
wide); layers 2–48 a 256-expert MoE (top 6 by sigmoid score, the picked
scores renormalized, no router bias in this release) with an ungated BF16
shared expert (1024 wide) added to the routed sum. Attention is GQA with 32
query heads of width 128 over 4 kv heads, QK RMSNorm per head, a learned
per-head sink logit on every layer (`softmax_type: learnable`), and two
layer kinds on `layer_types`: 35 sliding-window layers (window 513: the row
and the 512 before it, RoPE θ = 1e4 over the full head) and 14 full-attention
layers (every one of them in `no_rope_layer_indices`: no positional encoding
at all). Hidden 2048, vocab 150 144, 262 144 positions, no MTP head.

The export keeps every non-expert tensor in the source's BF16 bytes
(attention projections, norms, sinks, routers, shared experts, the two dense
MLPs, the embedding, the head) and quantizes the routed experts per layer and
per projection: FP8 e4m3 on 128 × 128 blocks with an fp32 multiplicative
`weight_scale_inv` (layers 2–11, 13–30, 32 whole; layer 12's gate and down;
layer 37's down), NVFP4 (`weight_packed` e2m1 pairs, `weight_scale` e4m3 per
16 along K, `weight_global_scale` an fp32 reciprocal: w = e2m1 × s / g) on
layers 31, 33–36, 38–48 whole, layer 12's up and layer 37's gate and up. The
allocation is `quantization_config.routed_experts.precision_by_layer`.

Relative to what the engine serves the reuse is large: the session core
(the row walk, chunked and grouped prefill, the decode graph, the prefix
arena, snapshots at any position), the resident loader (`ResidentLayerStream`
with the fp8 / NVFP4 / bf16 slice builders), the routed-expert executor
(`GlmMoeLayer` behind `QwenMoeLayer`'s BF16 shared tail: the sigmoid router,
the fp8 and NVFP4 slot kernels, the device-segmented grouped prefill), the
MiMo attention kernels (split-KV with a window and a sink, the fused decode
launch, the query-tiled tensor-core prefill, the fp8 K/V row form) and the
paged per-layer K/V pool, the bf16 GEMM seam with its 12-bit companions,
the fused add + norm, the GLM tool-call markers and `<think>` parsing. New:
the attention kernels' 128-wide instantiation with the QK norm and the
no-RoPE form, the one-rounding RMSNorm, the per-projection expert format in
the MoE executor (one mixed gate/up slot kernel), the o200k-style
three-stage pre-tokenizer, block `set` in the template renderer, and the
family's config / binding / loader / model.

## 1. The model

### 1.1 Configuration facts

| field | value |
|---|---|
| `hidden_size` / `vocab_size` / `num_hidden_layers` | 2048 / 150 144 / 49 |
| `layer_types` | 35 × `sliding_attention`, 14 × `full_attention` (0, 1, 5, 9, …, 45, 48) |
| `no_rope_layer_indices` | exactly the full-attention layers |
| `mlp_only_layers` | [0, 1] (dense, `intermediate_size` 6144); `decoder_sparse_step` 1 |
| heads | 32 query heads, 4 kv heads, `head_dim` 128 (q, k and v) |
| `rope_theta` / `max_position_embeddings` | 1e4 / 262 144 (no scaling) |
| `sliding_window` / `use_sliding_window` | 513 / true |
| `softmax_type` | `learnable` — `self_attn.softmax_offset` BF16 [32] per layer |
| MoE | 256 experts of 512, top 6, `router_score_function` sigmoid, `norm_topk_prob` true, no bias tensor |
| shared expert | `mlp.shared_expert.{gate,up,down}_proj` BF16, 1024 wide, ungated |
| norms | `rms_norm_eps` 1e-6; `q_norm` / `k_norm` BF16 [128] per layer |
| tokens | bos 150000, eos {150001, 150003, 150005, 150018}, pad 150001 |
| `quantization_config` | `quant_method: dgpp_mixed`, `routed_experts.precision_by_layer`, the fp8 and nvfp4 format descriptions |

The parser (`models/flint/config.hpp`) rejects by field name anything
outside this contract: a layer type other than the two, a rope scaling, a
`softmax_type` other than learnable, a head_dim other than 128, a window
that is not positive, a routed precision other than fp8 / nvfp4, a block
size other than 128 × 128 or a group other than 16, a tied embedding.

### 1.2 One layer (the release's `FlintDecoderLayer`)

```
x   = rmsnorm(h) * input_layernorm                (fp32 interior, ONE rounding)
q, k, v = x @ [Wq | Wk | Wv]^T                     BF16, fp32 out, bf16(dot)
q, k: per-head RMSNorm (one rounding) with q_norm / k_norm
      sliding-window layers: RoPE over all 128 dims (rotate_half pairs (i, i + 64),
      fp32 cos / sin, one rounding); full-attention layers: none
K, V appended to the paged cache at the row's slot
attn: softmax(q k^T / sqrt(128) [, sink_h]) v over [max(0, pos - 512), pos] (sliding)
      or [0, pos] (full); the sink one more softmax column with no value
h  += attn @ Wo^T                                  BF16 [2048, 32 x 128]
x   = rmsnorm(h) * post_attention_layernorm
h  += dense(x)                                     layers 0, 1: BF16 SwiGLU, 6144 wide
h  += routed(x) + shared(x)                        layers 2-48
```

The MoE: logits = fp32(x) @ fp32(gate) (the bf16 gate), scores = sigmoid,
top 6 over the scores (ties to the lower id), weights = the picked scores /
(sum + 1e-20); experts SwiGLU without clamps; the per-token accumulation in
ascending expert id in fp32, the shared expert's fp32 down projection added
with weight 1, one rounding — `QwenMoeLayer`'s chain with a unit shared
weight (`shared_gate` null) over a `SigmoidBias` routed config with a zero
bias vector (the kernels' selection key is then the score itself).

### 1.3 Attention (kernels/mimo_attn.hpp, the 128-wide instantiation)

The MiMo kernels are templates over the head geometry: `<192, 128, 64>`
(MiMo: qk 192 / v 128 / rotary 64, pairs (i, i + 32), nope dims after) and
`<128, 128, 128>` (Flint: a lane owns dims {l, l + 32, l + 64, l + 96}, the
rotary pairs (l, l + 64) and (l + 32, l + 96)). The finish takes the per-head
`q_norm` / `k_norm` weights (null: none) and a null `inv_freq` for the
no-RoPE layers; the norm is the one-rounding form over the bf16-rounded
dots. Everything else — the split-KV partials with the window and the sink,
the combine, the fused decode launch (finish + partials + combine, the
batch's own rows overlaid), the query-tiled tensor-core prefill, the fp8 K/V
form — is the same code. The window 513 is not a tile multiple: the kernels
mask per key, so a sliding row's visible range is [pos − 512, pos] exactly
and the decode split count is the window's tiles (17).

### 1.4 The K/V pool

`MimoKvPool` as is (per-layer kv head counts, k_dim = v_dim = 128): 49
layers × 4 kv heads × 128 × 2 × 2 B = 100 KiB per token at world 1 (25 at
world 4) in bf16; the fp8 row form (`engine.kv_dtype: "fp8"`) halves it. The
sliding-window layers keep their full history in the paged cache (the MiMo
decision D4: prefix-cache uniformity); a 513-token ring is the recorded
memory lever (71 % of the pool at long contexts).

### 1.5 Tensor census

Per layer: `self_attn.{q,k,v,o}_proj.weight` BF16 [4096, 2048] / [512, 2048]
/ [512, 2048] / [2048, 4096]; `q_norm.weight`, `k_norm.weight` BF16 [128];
`softmax_offset` BF16 [32]; `input_layernorm`, `post_attention_layernorm`
[2048]. MoE layers: `mlp.gate.weight` BF16 [256, 2048]; `mlp.shared_expert.
{gate,up}_proj` BF16 [1024, 2048], `down_proj` [2048, 1024]; 256 × gate / up
/ down each FP8 (`weight` F8_E4M3 [512, 2048] + `weight_scale_inv` F32 [4,
16]; down [2048, 512] + [16, 4]) or NVFP4 (`weight_packed` U8 [512, 1024] +
`weight_scale` F8_E4M3 [512, 128] + `weight_global_scale` F32 [1]; down
[2048, 256] + [2048, 32] + [1]). Dense layers: `mlp.{gate,up}_proj` BF16
[6144, 2048], `down_proj` [2048, 6144]. Globals: `model.embed_tokens.weight`,
`lm_head.weight` BF16 [150144, 2048], `model.norm.weight`.

## 2. Placement by world size W ∈ {1, 2, 4}

The kv heads (4) must divide by W. Rank r holds query heads [32r/W, +32/W)
and kv heads [4r/W, +4/W): the q / k / v rows of those heads stacked as ONE
bf16 matrix [(32 + 8) × 128 / W, 2048] in [Q | K | V] order (the finish
kernel's chunk layout with one chunk), o_proj's packed columns, the sink's
and the norms' slices; the dense MLP at 6144/W, the shared expert at 1024/W,
every routed expert at 512/W (a 128-row fp8 block at world 4, so every slice
re-anchors its scale grid exactly; NVFP4 slices on 16-blocks); the router,
the layer norms and the embedding replicated; the head vocab-sharded. The
bf16 matrices the decode GEMV streams (qkv, o_proj, the dense and shared
MLPs, the head) pack into their 12-bit companions under
`engine.bf16_weights`.

| per rank | W = 1 | W = 2 | W = 4 |
|---|---|---|---|
| routed experts (fp8 + nvfp4) | 29.7 GiB | 14.8 GiB | 7.4 GiB |
| bf16 attention + shared + dense | 2.9 GiB | 1.5 GiB | 0.7 GiB |
| embedding + head | 1.15 GiB | 0.86 GiB | 0.72 GiB |
| weights | ~33.7 GiB | ~17.1 GiB | ~8.9 GiB |
| K/V per token (bf16 / fp8) | 100 / 52 KiB | 50 / 26 KiB | 25 / 13 KiB |

Decode traffic at T = 1, W = 1 per step: 47 × (6 experts × (2 × 1 MiB +
1 MiB fp8, or half in NVFP4) + 15 MiB of bf16 attention and shared-expert
companions) + 2 × 75 MiB dense + 440 MiB head ≈ 2.9 GB — a 12 ms floor at
the line rate; at W = 4 about 0.8 GB.

## 3. Design decisions

* **D1 — the release's formats as shipped.** FP8 block-128 experts through
  `load_quant_rows/cols`, NVFP4 experts through `load_fp4_rows/cols` with
  the checkpoint's reciprocal global (the kernels' epilogue divides), every
  BF16 tensor verbatim. No requantization; `engine.dense_weights`-style
  levers are not wired for this family.
* **D2 — per-projection expert formats in the shared executor.**
  `GlmMoeWeights` carries both expert tables and a three-bit mask naming
  which projection is NVFP4; the decode slot path picks the gate/up kernel
  by the pair (both fp8, both fp4, or the new mixed kernel: the fp8 row dots
  for one matrix into shared memory, the fp4 core for the other, the
  SwiGLU on the pair) and the down kernel by its own bit; the grouped
  prefill chain picks per launch. Layer 12 (gate fp8, up nvfp4, down fp8)
  and layer 37 (gate / up nvfp4, down fp8) are the only mixed layers; every
  uniform layer runs the kernels it ran before.
* **D3 — the attention kernels generalized, not forked.** One kernel
  source over `<DK, DV, R>`; MiMo's instantiation is unchanged and its
  bitwise gates pin it.
* **D4 — the one-rounding norm.** Flint's norms (the layer norms, the final
  norm, the per-head q/k norms) round once: y = bf16(x × rstd × w) — the
  released reference's and SGLang's `rmsnorm` form; `add_rmsnorm` gains the
  variant beside its two-rounding GLM form. The residual stays bf16 (the
  torch reference the quantization study measured against).
* **D5 — BF16 projections on the GEMM seam.** The fused qkv (fp32 out: the
  finish rounds once), o_proj, the dense MLPs, the shared expert and the
  head run through `IGemm::matmul` — the 12-bit companions at decode rows,
  cuBLASLt on prefill chunks — exactly as the BF16 sites of MiMo and
  GLM-4.7.
* **D6 — no speculative block.** The checkpoint has no MTP head and no
  drafter; the templates serve plain decode (`mtp: false`), the plain boot
  of the campaign is the shipped mode.
* **D7 — the router's bias is a zero vector.** `MoeRouterMode::SigmoidBias`
  with a synthesized F32 [256] zero vector per layer keeps the kernels'
  selection key the score itself (score + 0 = score) without a fourth
  router mode.

## 4. Layout

```
src/models/flint/config.{hpp,cpp}        the parsed contract (precision_by_layer included)
src/models/flint/binding.{hpp,cpp}       the expected-tensor table, the TP geometry check
src/models/flint/loader.{hpp,cpp}        FlintLayerStream on ResidentLayerStream
src/models/flint/layers.{hpp,cpp}        FlintAttentionLayer (bf16 fused qkv), FlintDenseMlp (bf16)
src/models/flint/forward.{hpp,cpp}       FlintModel on SessionModel: the row walk, the memory plan
src/models/flint/attn_reference.{hpp,cpp} the host oracle of the 128-wide finish (norm, full RoPE)
src/kernels/mimo_attn.{hpp,cu}, mimo_attn_prefill.cu   the kernels as templates over <DK, DV, R>
src/kernels/add_rmsnorm.{hpp,cu}         the one-rounding variant
src/kernels/glm_moe.cu, models/glm/moe_layer.cpp       the per-projection format mask, the mixed slot kernel
src/text/tokenizer.cpp, unicode_tables.hpp              the o200k-style three-stage scanner
src/text/chat_template.cpp               block set, \x and \u escapes
tests/cuda/flint_fixture.hpp             the tiny release (8 heads, 4 kv heads, window 32, 8 experts of 64, the mixed layers)
tests/cuda/flint_{attn,loader,forward,decode,tp,engine}_test.*   the ladder
tests/unit/flint_{config,binding}_test.cpp
tools/flint_reference_dump.py            the numpy reference over the fixture
apps/flint_{load,forward}_check.cpp      the real-checkpoint checks
deploy/cluster_flint_fp8-nvfp4_w{1,2,4}.example.json
```

## 5. Implementation stages and validation

G0 config + binding (unit tests over the real config.json) · G1 the
kernels against the host oracle (`flint_attn_test`, the MoE mixed-format
gates in `glm_moe_test`, `add_rmsnorm_test`) · G2 the loader over the
fixture at worlds 1/2/4 (`flint_loader_test`) · G3 the forward against the
numpy reference (`flint_forward_test`) · G4 the session surface
(`flint_decode_test`) · G5 loopback TP worlds 2 and 4 (`flint_tp_test`) ·
G6 the graph engine on loopback worlds (`flint_engine_test`) · G7 the real
checkpoint: `flint_load_check`, `flint_forward_check` transcripts, then the
fabric worlds 1, 2 and 4 through `scripts/dgpp-cluster` · G8 the benchmark
campaign into docs/benchmarks.md and the README.

## 6. Status

Recorded as the stages land (see the dated entries below).

**2026-10-22 — G0–G3 landed.** G0: `FlintTextConfig` / the binding table
over the real config.json (8 unit tests). G1: the sink-attention family's
qk-128 instantiation (`flint_attn_test`, 6 tests: the finish within two
ulps of the host oracle, the 513 window, fused == chain, tiled prefill ==
chain, the fp8 cache's codes bitwise the host codec); the MoE
per-projection form (D2) in `GlmMoeLayer` — both tables bound with the
mask, the grouped chain picking the core per launch, the decode slot path
on a new `moe_slot_gate_up_swiglu_mixed_kernel` (the fp8 core and the fp4
core on the pair, each dot bitwise its uniform kernel's) — pinned in
`glm_moe_test` (`moe_decode_slot_path_is_bitwise_host_path_mixed`: seven
mixes, the host path 0 ulps against the double oracle, decode bitwise the
host path, prefill bitwise the host path on the tensor-core cores);
`QwenMoeLayer` takes a sigmoid router with a bias and a null shared gate
(weight 1); `add_rmsnorm_bf16_one` (D4) pinned against its host oracle. G2:
`flint_loader_test` — every class byte-exact at worlds 1, 2 and 4, the
byte formulas, resident mode. G3: `flint_forward_test` over the six-layer
fixture (dense / fp8 / mixed / NVFP4 layers, the 33 window): end to end
(relaxed) passes; teacher-forced strict: every layer within 7 bf16 ulps
(l2 ≤ 0.0034), the final hidden BITWISE the numpy reference, top-8 exact,
routing exact. The fp8 K/V pool: the dense layers within 2.2 % of the
bf16 pool's residual (the e4m3 step); the random fixture's MoE routing
near ties flip under it (reported, not judged).

**2026-10-22 — G4–G7 landed.** G4 `flint_decode_test`: the prefill's last
row bitwise the cold forward; 60 greedy steps across the pool's block
boundary against the re-forward (the rows outside the routing flips
within 2.7 % relative l2, top-1 exact beyond near ties; the flip bound 8
of 240 decisions — every bf16 site reassociates between the decode GEMV
and cuBLASLt's prefill rows, so the random router flips more than on the
fp8-projection families); the rolling per-step snapshots bitwise; the
verify rows bitwise the scalar steps through the GEMV rows (four),
tolerance-equal above with the post-rollback step tolerance-equal there
too (the accepted row's K/V came from the m-row launch); interleaved slots
bitwise; the chunked prefill, close/reopen, the pool accounting, the prefix
snapshots hot == cold bitwise; the 33-window pinned at the session level.
G5 `flint_tp_test`: loopback worlds 2 and 4 bitwise across ranks; against
world 1 the dense layers within 0.5 % and the first MoE layers within
1.7 % on the rows before the first routing flip (the budget judges a
layer on 16+ clean rows; the later layers of the tie-prone fixture are
reported). G6 `flint_engine_test`: the graph engine's scalar and
row-batched replays bitwise the eager engine at world 2. G7 the real
checkpoint on this node: `flint_load_check` binds all 85 886 tensors
(23 040 fp8, 13 056 NVFP4 matrices) at 33.29 GiB resident; the tokenizer
is the DeepSeek three-stage shape with the o200k main pattern — a new
scanner stage (`pattern_ 4`, Ll / Lu+Lt tables) pinned byte-exact on 83
goldens; the chat template needed block `set … endset` and the `\x` /
`\u` string escapes, 15 goldens render byte-exact, the tool round trip
and grammar pass with the parser starting outside reasoning (the model
opens its own `<think>`); `flint_forward_check` on "The capital of France
is" decodes " Paris. The capital of Italy is Rome. The capital of Canada
is Ottawa. The capital of Australia is Canberra." with 24 of 25 decode
rows agreeing with the re-forward's argmax (the one divergence a near tie
between "." and " is").

**2026-10-22 — G8 in progress (the campaign).** Two defects the first
one-node leg exposed, fixed before its rows: the prefill probe's
`--no-think` asked an always-on template to switch thinking off (HTTP 400:
the probe now retries without the switch and records `thinking:
always_on`); the family declared no positional ceiling, so the server
advertised the 512K pool as the request limit (now
`max_position_embeddings`, 262144 — the pool seats concurrency). The
one-node image capture failed on a full disk (the partial image removed;
the campaign runs with `DGPP_RESIDENT_CACHE=off`, which touches boot time
only). First readings (release `4709d8d6`, one node, eight slots, 512K
BF16 pool): decode 55.3–56.4 tok/s at C1 (18 ms/step), 96.6–98.1 at C2,
140.6–142.4 at C4, 191.9–197.2 at C8; cold prefill 0.225 / 0.51 / 2.07 /
13.9 s at ~512 / 2K / 8K / 32K tokens (0.44 / 0.25 / 0.26 / 0.43 ms per
token — the 14 full-attention layers' quadratic term past 8K). The T=1
step streams about 3.4 GB (the 12-bit projections and head, the fp8 and
NVFP4 experts, the BF16 shared experts): 14.6 ms at the 233.6 GB/s read
ceiling, so the step sits at ~81 % of the DRAM floor; the levers left are
the shared expert's BF16 tail (12.6 MB per layer, read raw) and the
launch count per layer.

The quality leg of that run then exposed a third defect, fixed before the
rows are re-measured: schema extraction scored 93/100 (every other family
scores 100) because the service treated the model's own `<think>` as
content — the model-opened block was a MiMo-dialect coupling, so under
the JSON grammar Flint's opener was refused at the first position and the
outputs came out garbled, and the usage counted no reasoning tokens.
Whether the model opens its own `<think>` is now a family declaration
(`ServeFamily::model_opens_thinking`, carried by the frontend's markers
and the grammar vocabulary — MiMo-V2.6 and Flint state it; a prompt's
tokens never select it), and the prompt-tail rule only checks that the
template settled nothing itself (no `<think>` opened, no block closed).
Pinned by a unit test over every family's generation-prompt shape with
and without the declaration, a sweep of the cached checkpoints' real
templates, and the GLM JSON grammar gate driving Flint's documents behind
the model's own block (and refusing the opener without the declaration).
The campaign's rows are re-measured on the fixed binary.

Prefill attention tiling (the lever the 32K row pointed at): the
query-tiled kernel stages each 32-token K/V tile for 128 query vectors
per block instead of 64 (`kSinkAttnPrefillQueryVecs`, both head
geometries), doubling the FLOP per cache byte. On the 32K prompt
(`flint_forward_check --ids-file`, nsys kernel sums, one launch per
layer) the kernel's 49 launches take 5.16 s against 6.44 s before — 24.8
vs 19.9 TF on the 1.28e14 attention FLOP — and the forward's kernel total
10.04 s against 11.32 s; the attention tests and the forward chains are
unchanged (bitwise). MiMo-V2.6 shares the kernel: on an 8K prompt
(`mimo_forward_check`, 48 launches) its attention takes 0.621 s against
0.820 s. Kept.

**2026-10-22 — G8: the campaign on `b3f9b538` (binary `15ac3372bb2e3230`).**
One node, the template as shipped (eight slots, 512K BF16 pool, the
262,144 positional ceiling), every row from one `serve_campaign.py run`
with the resident image cache off: decode 56.2–58.0 tok/s at C1 (17.4
ms/step), 99.9–101.2 at C2, 146.2–148.0 at C4, 197.8–204.0 at C8; cold
prefill 0.484 / 1.951 / 12.077 s at 2K / 8K / 32K (the 32K probe 13.9 s
before the attention tiling); context decode 57.6 / 47.5 / 39.6 / 29.4 /
19.4 tok/s at 0 / 32K / 64K / 128K / 256K with cold prefill 11.7 / 36.4 /
133.2 / 503.1 s at 32K / 64K / 128K / 256K (45.5 / 175.5 / 663.8 s before
the tiling); quality 137/164 HumanEval, 255/300 GSM8K, 94/100 extraction
under the standard caps with 73 responses at the cap, of which 65
completed at the 8,192 retry and 60 passed (the eval's retry column,
added this day — docs/benchmarks.md "Quality and correctness" has the
reading of each miss); API checks 10/10 with reasoning tokens reported;
operation streams matched, zero swap.
Two nodes, the template as shipped (eight slots, 512K BF16 pool, the
same ceiling): decode 104.3–105.2 tok/s at C1 (9.5 ms/step), 175.1–177.5
at C2, 253.7–258.0 at C4, 342.5–352.5 at C8; cold prefill 0.413 / 1.570
/ 8.661 s at 2K / 8K / 32K; context decode 104.4 / 83.8 / 70.6 / 54.4 /
37.3 tok/s at 0 / 32K / 64K / 128K / 256K with cold prefill 8.5 / 24.4 /
78.2 / 276.2 s at 32K / 64K / 128K / 256K; quality 133/164, 257/300,
96/100 with 77 responses at the cap, 63 completed at the retry and 55
passing; API checks all OK; operation streams matched, zero swap.
Four nodes, the template as shipped: decode 154.8–158.9 tok/s at C1 (6.4
ms/step), 256.2–260.8 at C2, 393.9–399.7 at C4, 524.8–542.9 at C8; cold
prefill 0.358 / 1.297 / 6.468 s at 2K / 8K / 32K; context decode 154.4 /
127.1 / 110.3 / 89.5 / 64.9 tok/s at 0 / 32K / 64K / 128K / 256K with
cold prefill 6.3 / 16.5 / 47.7 / 153.8 s at 32K / 64K / 128K / 256K;
quality 137/164, 255/300, 100/100 with 72 responses at the cap, 59
completed at the retry and 53 passing; API checks all OK; operation
streams matched, zero swap. The per-task scores move by a few items
between the legs in both directions (HumanEval 137 / 133 / 137, GSM8K
255 / 257 / 255, extraction 94 / 96 / 100 at one / two / four nodes —
the two "Oona" records pass at four nodes): the tensor-parallel split
changes the reduction order of every projection, and the long-thinking
items near a greedy tie resolve differently — the same mechanism as the
solo-versus-batched transcript difference, which every leg shows. The
one-to-two-node step is 1.8× at C1 and 1.7× at C8, the two-to-four 1.5×
at both: the per-rank expert bytes halve each time while the collective
and the replicated attention do not. The benchmarks doc carries the rows.
