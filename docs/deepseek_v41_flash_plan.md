# DeepSeek-V4.1-Flash on dgpp — quantization decision, architecture facts and implementation plan (2026-09-13)

Status: plan. Nothing is built and nothing is downloaded. The architecture
study uses the repository's `config.json`, the safetensors headers of all
48 shards (read over HTTP range requests, 96,085 tensors, 475.24 GiB), the
reference implementation shipped with the checkpoint (`inference/model.py`,
`engram.py`, `kernel.py`, `convert.py`, 2,300 lines — the ground truth for
every formula below), the technical report (`DeepSeek_V41_Tech_Report.pdf`
§2.2–2.4, §3.2, §4.2.1), the prompt-format reference (`encoding/`), the
DSpark paper (arXiv 2607.05147) and the three community NVFP4 re-packs
listed in §0.1. Nothing here is measured on the fabric; every number is
derived and says so.

This is the fifth served family. Per PLAN.md it goes on the shared cores
(`SessionModel<Derived>`, `ResidentLayerStream<Family>`, `WeightBuilder`,
`GlmMoeLayer`, the select networks in `kernels/dsa.cu`, the Qwen n-gram
table's mmap and host-node machinery, the graph/prefix/MTP engine), never on
copies of another family's model class.

References: <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash>,
<https://arxiv.org/abs/2607.05147> (DSpark),
<https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4>,
<https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-NVFP4-nvidia>,
<https://huggingface.co/s-zaizen/DeepSeek-V4.1-Flash-NVFP4>.

## 0. Summary

DeepSeek-V4.1-Flash (`DeepseekV41ForCausalLM`, `model_type deepseek_v41`)
is a 40-layer multimodal MoE: 552 B backbone parameters plus 196 B of
Engram n-gram memory, 8 B active per token in prefill and 16 B in decode,
1,048,576 positions. Every layer is one CSA2 attention block (64 query heads
of 512 over ONE shared 512-wide latent that is both key and value, a
128-token sliding window on every layer, compressed main KV shared across
layer groups, a 32-head fp4 indexer with top-512 selection) and one MoE
(384 routed experts top-6 with `sqrtsoftplus` scoring and a `noaux_tc` bias,
one shared expert, intermediate 2304, SwiGLU clamped at 10). The residual
is four mHC streams in the "single-pass" form (each sublayer collapses its
input with the coefficients the PREVIOUS sublayer predicted). The Engram
module at layers 1 and 14 gathers 24 rows of 256 from a 384 M-row hashed
table per token and gates them into the streams. DSpark, the speculative
head, is three extra layers that draft FIVE tokens per pass in parallel with
a low-rank Markov head and a confidence head. Vision (a 32-layer ViT) is
out of scope.

**The quantization decision (§0.1): download the upstream checkpoint and
serve its native formats; none of the three community NVFP4 re-packs is
used.** The model ships already quantized by DeepSeek: routed and draft
experts in MXFP4 (e2m1 with one e8m0 scale per 32 — 268.9 + 6.7 GiB),
attention/dense projections in FP8 e4m3 with e8m0 scales on 32×32 blocks
(5 GiB), the Engram tables in FP8 e4m3 with e8m0 scales per 32 (188.8 GiB),
embeddings/head/vision in BF16 (4 GiB). The two "NVFP4" repos from the
modelopt cast carry bit-identical expert values in a 6 % larger encoding
whose only addition (`input_scale`) serves W4A4 tensor-core paths this
engine does not run; LibertAI's re-pack adds a lossy FP4 Engram (cosine
0.9934, unevaluated) that saves disk bytes but not NVMe reads. The engine's
own fp4 core decodes `e2m1 × 2^k` exactly in bf16 and keeps activations in
bf16, so it is MORE precise than the reference's W4A8/W8A8 kernels; the
KV caches are stored in the formats the model was trained for (fp4 e4m3/16
main KV, fp8 e8m0/32 window KV, fp4 e8m0/32 indexer keys).

**The Engram tables stay on the NVMe, mmap'ed, head-sharded** (the Qwen
n-gram design, D5): 12 random row reads per token per rank at world 4,
gathered by a host node that overlaps the first layer. Everything else is
resident: **≈ 73 GiB per rank at world 4** (§2.1), which leaves ~45 GiB of
headroom that the page cache spends on hot Engram rows. **The global KV
cache is 890 bytes per token per rank** — one million tokens of context
costs 0.87 GiB — so the context budget is the model's position limit, not
memory. Two nodes do not fit (144 GiB of weights per rank); three do not
divide the head, group and vocabulary counts; the family is a world-4
deployment.

The decode traffic model (§2.2): **3.83 GB per rank per token at T = 1, a
16.7 ms floor at 230 GB/s**, ~81 collectives per step (~4 ms), so an honest
expectation of 22–26 ms per T = 1 step. A DSpark pass verifies six rows
for ~0.4 GB more; at the paper's block-5 acceptance the expectation is
**7–9 ms per token**, against GLM-5.3-Flash's 19.8 and the full GLM-5.3's
36–42. Prefill runs the 20 encoder layers over the prompt and the 20 decoder
layers over its last 128 tokens (the model's own CED/bounded-replay
semantics, D8), about half the work of a 40-layer walk.

### 0.1 The quantization decision, in full

What the four repositories contain (from their headers and model cards):

| repo | routed experts (57 %) | Engram (40 %) | dense / attention | size | calibration | eval |
|---|---|---|---|---|---|---|
| `deepseek-ai/DeepSeek-V4.1-Flash` (upstream) | MXFP4: e2m1 I8 `[N, K/2]` + e8m0 `[N, K/32]` | FP8 e4m3 `[384 M, 256]` + e8m0 `[384 M, 8]` | FP8 e4m3 + e8m0 on 32×32 blocks; `wo_a` the same | 475.2 GiB | — (trained this way, QAT for the fp4 KV/indexer) | the model card's |
| `AtomicChat/…-NVFP4-nvidia` | the same nibbles re-scaled: e4m3 per 16 + fp32 global (+5.75 %), plus a calibrated `input_scale` per projection | unchanged | unchanged | 491.1 GiB | 64 × 512 tokens (cnn_dailymail + Nemotron) | on vLLM's W4A4 path: −1.8 pt top-1, +0.85 % ppl on general text vs the original's own path; the calibration itself "no measurable benefit" |
| `s-zaizen/…-NVFP4` | the same cast, `input_scale` = 1.0 | unchanged | unchanged | 491.1 GiB | none | none |
| `LibertAIDAI/…-NVFP4` | the same cast (verified bit-exact against s-zaizen) | **lossy e2m1 + e8m0/32** (mean cosine 0.9934 over 1.5 M sampled rows) | unchanged | 399.9 GiB | none | none ("no engine can run this architecture yet") |

The decision and its reasons:

1. **The upstream checkpoint is the most accurate one available**, because
   it is the model: the experts were trained/QAT'd at fp4, so there is no
   "unquantized" original to be closer to, and every other repo is derived
   from these bytes. Every NVFP4 cast is provably lossless (the 32-element
   e8m0 block is a refinement of NVFP4's 16-element e4m3 block; all
   34.8 G block scales fall in e4m3's exact window) but costs 6 % more
   bytes on the class that dominates decode traffic and resident memory.
2. **The engine gains nothing from NVFP4's encoding.** Its fp4 core
   (`kernels/fp4_gemv.cuh`) decodes codes in registers and multiplies by
   the block scale exactly in bf16; with an e8m0 scale the multiply is an
   exponent add and the epilogue's division by the global scale disappears
   — the MXFP4 variant of the core is simpler and reads fewer bytes than the
   NVFP4 one (D2). The `input_scale` that the calibrated repo adds places
   the e4m3 window for FP4 *activations*; this engine keeps activations in
   bf16 (W4A16 decode, the fp4×bf16 tile kernel in prefill; W4A4 was
   measured and parked in `docs/nvfp4_plan.md`), so the scale is unused,
   and AtomicChat's measurement says the calibration did not help even
   where it applies.
3. **The lossy Engram re-pack is not adopted, and is not needed for
   memory.** A row is 264 B in FP8 and 136 B in FP4; either is one 4 KiB
   page fault, so the FP4 table halves disk bytes and page-cache footprint
   per row but not the NVMe IOPS the gather costs. Under D5 the tables are
   not resident, so the 91 GiB saving buys nothing at world 4 (73 GiB of
   resident weights leave ~45 GiB free). The quality cost is unmeasured by
   its authors; if the page-cache hit rate ever matters (§2.4), the same
   re-pack can be produced locally from the upstream table in an
   afternoon and A/B'd with `fabric_logprob.py` — a lever, not a decision.
4. **Activation and cache precision.** Weights are consumed exactly
   (fp4 × 2^k, e4m3 × 2^k — both exact in bf16); activations stay bf16
   where the reference quantizes them to MXFP8 per 32 (`act_quant` with
   `ue8m0` scales before every fp8/fp4 GEMM), so every projection is at
   least as precise as the reference's; the mHC coefficient matrices stay
   fp32 as stored (`hc_*_fn` F32 `[24, 20480]`, a bf16 form is a measured
   lever in §5, D10); the three KV caches use the model's trained formats
   and no other (`kv_dtype` is refused for this family, D7) — a bf16 cache
   would hold the same fp4-representable values in more bytes.
5. **Disk.** 510 GB per node; each of the four nodes has 821 GB free with
   the current checkpoints in place (2026-09-13), enough for one copy each
   with 310 GB left. The Engram tables are read straight from the two
   94.56 GiB shards in the Hub snapshot (`model-00047/48-of-00048`,
   the Qwen precedent); nothing is converted or re-packed.

## 1. The model

### 1.1 Configuration facts

`config.json` nests the text model under `text_config`; the flat
`inference/config.json` names the same values in the reference's spelling
(second column).

| field | value | reference name |
|---|---|---|
| `architectures` / `model_type` | `DeepseekV41ForCausalLM` / `deepseek_v41` (text: `deepseek_v41_text`) | |
| layers | 40 = 20 causal-encoder + 20 decoder; `num_nextn_predict_layers` 3 (DSpark, `mtp.0..2`) | `n_layers`, `n_mtp_layers` |
| hidden / vocab / eps | 5120 / 129,280 / `rms_norm_eps` **1e-20**, silu, untied head | `dim`, `norm_eps` |
| attention | 64 heads × `head_dim` 512 (`qk_rope_head_dim` 64 inside it), `num_key_value_heads` 1, `q_lora_rank` 1280, `o_lora_rank` 1024, `o_groups` 8, no bias, `sliding_window` 128 | `n_heads`, `rope_head_dim`, `window_size` |
| compression | `compress_ratios` (43 entries = 40 + 3 draft layers): layers 0–1 → 0 (window only), 2–19 → 2, 20–39 → 1, draft → 0; `kv_source_layer_ids` [2, 8, 14, 20]; `index_source_layer_ids` [2, 8, 14, 20, 24, 28, 32, 36]; `candidate_source_layer_id` 20, `candidate_topk_blocks` 2048, `candidate_block_size` 8 | the same |
| indexer | `index_n_heads` 32, `index_head_dim` 128, `index_topk` 512 | |
| RoPE | `rope_theta` 10,000 (window-only layers, no scaling); `compress_rope_theta` 160,000 with YaRN `factor` 16, `original_max_position_embeddings` 65,536, `beta_fast` 32, `beta_slow` 1 (every layer with `compress_ratio` > 0); `max_position_embeddings` 1,048,576 | `original_seq_len`, `rope_factor` |
| mHC | `hc_mult` 4, `hc_sinkhorn_iters` 20, `hc_eps` 1e-6 | |
| MoE | 384 routed + 1 shared, top-6, `scoring_func` **`sqrtsoftplus`**, `topk_method` `noaux_tc`, `norm_topk_prob`, `routed_scaling_factor` 1.5, `moe_intermediate_size` 2304, `swiglu_limit` 10 | `score_func`, `route_scale` |
| Engram | `engram_layer_ids` [1, 14]; `engram_num_embeddings` [384,006,168, 384,016,682]; `engram_max_ngram_size` 4; `engram_vocab_size` 16,000,000; `engram_n_heads` 8; `engram_head_dim` 256; `engram_pad_token_id` 2; `engram_compressed_vocab_size` 99,092 | |
| DSpark | `dspark_block_size` 5, `dspark_noise_token_id` 128,799, `dspark_target_layer_ids` [37, 38, 39], `dspark_markov_rank` 256, `dspark_n_routed_experts` 128, `dspark_num_experts_per_tok` 3 | |
| tokens | bos 0, eos 1, pad 2, `image_token_id` 129,264; recommended sampling temperature 1.0, top_p 0.95 (`generation_config.json` is absent) | |
| quantization | `quant_method` `fp8`, `activation_scheme` `dynamic`, `weight_block_size` [32, 32], `scale_fmt` `ue8m0`, `expert_dtype` `fp4` | |
| vision | 32 layers × 1024, patch 14, 3× pixel unshuffle — not loaded (D9) | |

`loaders/architecture.cpp` refuses the class name today; the family is a
new `ModelArchitecture::DeepseekV41` and a new parser (§4).

### 1.2 One layer (the reference `Block`, single-pass mHC)

The residual is `x ∈ [T, 4, 5120]` (four streams, all equal to the
embedding at layer 0). Every sublayer runs between an `hc_pre` collapse and
an `hc_post` expansion; the collapse coefficient comes from the previous
sublayer (`pre_mix`), the post/comb coefficients from this one:

```
(pre_a, post_a, comb_a) = hc_mixes(x, hc_attn_fn, hc_attn_scale, hc_attn_base)   # from x, fp32
u  = hc_pre(x, pre_mix)                # Σ_i pre_mix[i]·x[i] in fp32, one bf16 rounding
u  = RMSNorm_5120(u; eps 1e-20)·attn_norm
y  = Attention(u)                                                            # §1.3
x  = hc_post(y, x, post_a, comb_a)     # x'[i] = post_a[i]·y + Σ_j comb_a[j,i]·x[j], fp32, ONE bf16 rounding
(pre_f, post_f, comb_f) = hc_mixes(x, hc_ffn_fn, …)
u  = hc_pre(x, pre_a)                  # the ATTENTION's pre, not the FFN's
u  = RMSNorm_5120(u)·ffn_norm
y  = MoE(u)                                                                  # §1.5
x  = hc_post(y, x, post_f, comb_f);  pre_mix ← pre_f                         # carried to the next layer
```

`hc_mixes`: `flat = x.flatten(2).float()` (20480 wide), `rsqrt =
(mean(flat²) + 1e-20)^-1/2`, `mixes = (flat @ hc_fn^T) · rsqrt` (fp32
weights `[24, 20480]`), then `pre = sigmoid(mixes[0:4]·scale[0] + base[0:4])
+ 1e-6`, `post = 2·sigmoid(mixes[4:8]·scale[1] + base[4:8])`, `comb =
softmax(mixes[8:24].view(4,4)·scale[2] + base[8:24]) + 1e-6`, one column
normalisation, then 19 row+column passes, every division adding 1e-6 — the
GLM mHC's Sinkhorn exactly (`models/glm/mhc.hpp`), with four differences
the kernel takes as parameters (D4): the collapse uses the previous site's
`pre`; `hc_fn` is fp32 in the file; `norm_eps` 1e-20; `hc_post` rounds
once (GLM's reference rounds `post·h` and the comb sum separately).
Layer 0's attention collapses with the one-hot `[1, 0, 0, 0]`; after
layer 39 the head input is `hc_pre(x, pre_f_39)` (a weighted collapse, not
GLM's mean) → `RMSNorm·norm` → `head`. Layers 1 and 14 apply Engram to `x`
(all four streams) before their `hc_mixes` (§1.6). The DSpark target hidden
at layers 37, 38, 39 is `x.mean(dim=streams)` of the attention INPUT
(§1.7).

### 1.3 CSA2 attention (the reference `Attention`, `Compressor`, `Indexer`)

Per layer `l` with `ratio = compress_ratios[l]`, `rd = 64`:

```
qr   = RMSNorm_1280(W_qa u)·q_norm                               # wq_a fp8 [1280, 5120]
q    = W_qb qr → [64 heads, 512];  rope(q[..., -64:], pos)        # wq_b fp8 [32768, 1280], head-sharded
kv   = RMSNorm_512(W_kv u)·kv_norm;  rope(kv[..., -64:], pos)     # wkv fp8 [512, 5120]; ONE latent, K == V
kv  ← fp8 e4m3 per-32 e8m0 (quantize-dequantize)                 # the window KV's stored form
ring[l][pos % 128] = kv                                            # per layer, per request
window rows = the ring's ≤ 128 filled slots with position ≤ pos
if ratio > 0:
   if l ∈ kv_source: latent = Compressor(u)                        # §1.3.1; every `ratio` tokens
                     index_k = fp4(rope(RMSNorm_128(W_k latent)·k_norm))   # wk bf16 [128, 512]; e8m0/32
                     rope(latent[..., -64:], j·ratio); main = fp4 e4m3/16 (latent)   # the shared main KV
   if l ∈ index_source: sel = Indexer(u, qr, pos)                  # §1.3.2, top-512 entries, ascending
   else:                sel = the last index source's selection
   rows += main[sel]                                               # the shared cache of the last kv source
o    = Σ_j softmax_j(q·row_j / √512, +sink_h)·row_j               # attn_sink fp32 [64]: exp(sink−m) in the denominator
rope⁻¹(o[..., -64:], pos)                                          # the output's rope tail un-rotated
o    = per group g (8 heads = 4096): W_oa[g] o_g  → [8 × 1024]     # wo_a fp8 [8192, 4096], block-diagonal
y    = W_ob o                                                      # wo_b fp8 [5120, 8192], row-parallel + fold
```

The rotation is the complex form over ADJACENT pairs `(2i, 2i+1)` computed
in fp32 and written back to bf16 once (`apply_rotary_emb`) — the
interleaved pairing of the engine's `dsa_rope_interleave` with a
one-rounding policy (a kernel parameter, D6). One table per layer:
`(theta 10000, no YaRN)` for layers 0, 1 and the draft; `(theta 160000,
YaRN 16× over 65536, betas 32/1)` for layers 2–39 — the window branch of a
compressed layer rotates with the compressed table too. A compressed entry
`j` rotates at position `j·ratio` (its first token). The softmax scale is
`512^-0.5` with no YaRN mscale.

Which layers do what (from the config; the tech report §4.2.1 confirms):

| layers | ratio | main KV / index K from | selection | notes |
|---|---|---|---|---|
| 0, 1 | 0 | — | — | window only; theta 10000 |
| 2, 8, 14 | 2 | own (Full) | own | encoder Full layers; the Compressor pools pairs |
| 3–7, 9–13, 15–19 | 2 | 2 / 8 / 14 | reused | encoder Reuse layers |
| 20 | 1 | own (Full): a plain projection of the ENCODER OUTPUT | own; builds the candidate pool | the CED's global KV |
| 21–23, 25–27, 29–31, 33–35, 37–39 | 1 | 20 | reused | decoder Reuse |
| 24, 28, 32, 36 | 1 | 20 | own, restricted to layer 20's candidate pool | decoder Reindex |
| draft 40–42 | 0 | — | — | DSpark, §1.7 |

So there are four main-KV caches (three at ratio 2, one at ratio 1), four
index-key caches beside them, eight selections per token, and 43 window
rings. The per-token global KV is exactly the card's **890 bytes**:
`3 × (288 + 68)/2 + (288 + 68)`, with 288 = 512 e2m1/2 + 32 e4m3 block
scales and 68 = 128/2 + 4 e8m0 scales.

#### 1.3.1 The compressor (kv-source layers)

`ratio 1` (layer 20): `latent = RMSNorm_512(W_kv u)` with a bf16 `wkv`
`[512, 5120]` — one entry per token, no gate. `ratio 2` (layers 2, 8, 14):
`kv = W_kv u`, `score = W_gate u` (bf16 weights `[512, 5120]` each,
promoted to fp32 — a bf16 weight cast to fp32 is exact, so the engine keeps
bf16 bytes and fp32 arithmetic), tokens grouped in consecutive pairs,
`latent = RMSNorm_512(Σ_pair softmax_pair(score)·kv)` with the softmax over
the pair PER CHANNEL, emitted when the pair completes; an odd tail waits in
per-request state (`kv_state`/`score_state`, 2 × 512 fp32 per layer). The
entry for pair `(2j, 2j+1)` becomes visible to the query at position
`2j + 1` and later (`compress_len = (pos + 1) // ratio`). The engine's
`index_kpool` machinery (a gated pool of `kpool` tokens with a per-request
tail ring and per-row tail snapshots for rollback, `dsa_kpool_*`) is the
same shape applied to a 128-wide key; here it applies to the 512-wide
latent plus its 128-wide key and is the pattern for the new kernel (D6).

#### 1.3.2 The indexer and the candidate pool (index-source layers)

```
q   = W_iqb qr → [32, 128];  rope(q[..., -64:], pos);  q ← fp4 e8m0/32 (quantize-dequantize)   # wq_b fp8 [4096, 1280]
w   = (W_p u) · 128^-0.5 · 32^-0.5                                                                # weights_proj bf16 [32, 5120]
s_j = Σ_h w_h · relu(q_h · index_k_j)      over visible entries j < compress_len
layer 20 only: block score = max over 8 consecutive entries; the query's newest block pinned to +inf;
               keep the top-2048 blocks → a boolean candidate mask over entries (16,384 positions)
layers 24/28/32/36: s_j ← −inf outside the mask
sel = the 512 highest s_j (all, when fewer are visible), returned ASCENDING, offset past the window rows
```

The reference computes the dots in bf16 (`einsum` on the fp4-dequantized
bf16 q and k) and selects with torch's tie order; the engine scores in fp32
with the pinned composite-key rule (`dsa.hpp`, "exact ties to the lower
index"), the same stance as for GLM: selection flips are certified as
near-ties, never eliminated. The `relu` and `select_k` 512 are the full
GLM-5.3's select parameters (`index_relu`, `kDsaSelectMaxK` ≥ 512); the
index cache stores the fp4 values in the engine's planar e4m3 + row-scale
layout exactly (an e2m1 × 2^k value has ≤ 2 significant bits, so it is
exact in e4m3 whenever the row's four block exponents span ≤ 17 binades —
the append kernel checks and counts violations; D6), which keeps the fused
decode select's streaming layout and Hadamard-free dots.

**Decoder implementation update (2026-09-21, issue #28).** Layer 20's
candidate selector and the restricted selectors at layers 20/24/28/32/36
now score in parallel stripes, then select by exact radix refinement of
the score/original-entry-ID keys. Each stripe prefetches four index rows
before evaluating the existing per-head FP32 arithmetic. The selectors
retain the newest complete block's pin, append every visible entry of an
incomplete block, and return ascending IDs with the same padding. When
every candidate fits, IDs are written directly without scoring or sorting.
The encoder's three DSA selectors and the prefill path retain their existing
implementations.

Both decoder stages use the key region of the existing DSA workspace,
with a fixed `max_entries` row stride across graph replays and batch sizes.
The old eight-part candidate buffer is gone: six slots with depth-four MTP
save 3.75 MiB per rank. No per-step allocation or additional global histogram
is needed. The September 13 records below describe the original implementation;
the September 21 record
documents the replacement and its exactness and performance checks.

### 1.4 The window ring and rollback

Every layer keeps `[max_requests, 128, 512 fp8 + 16 scale bytes]`. The
engine allocates **160 slots** per ring (window + `kSpecRows` + padding):
slot `pos % 160`. A rejected draft at `pos + k` overwrote the slot of
position `pos + k − 160`, which no later query's window (starting at
`pos − 126` or later) ever reads, so rejected rows need no restoration and
the ring is not snapshot state (D7). The draft layers' rings (`main_x`
projected to fp8) are the same object. Prefix-cache snapshots carry the
43 + 3 rings (3.7 MiB per snapshot), the compressor tails, the Engram
context and the block list; the tech report's "SWA bounded replay"
(regenerate the rings from the last 128 tokens) is the later lever that
would shrink a snapshot to the block list.

### 1.5 MoE (the reference `Gate`, `Expert`, `MoE`)

```
logits  = fp32(u) · fp32(W_gate)                       # gate.weight bf16 [384, 5120]
scores  = sqrt(softplus(logits))                        # NEW scoring rule (GLM: sigmoid; Qwen: softmax)
ids     = top-6 of (scores + gate.bias)                 # bias fp32 [384] (bias_vl for image tokens: unused, D9)
w       = scores[ids] / (Σ + 1e-20) · 1.5
gate_e  = clamp_max(W1 u, 10); up_e = clamp(W3 u, −10, 10); act = silu(gate)·up           # fp32 in the reference
y       = Σ_e w_e · W2_e act_e   (fp32)  + shared(u)                                    # shared: the same FFN, fp8, weight 1
```

Experts: `w1`/`w3` I8 `[2304, 2560]` + e8m0 `[2304, 160]`, `w2` I8
`[5120, 1152]` + e8m0 `[5120, 72]` — the MXFP4 triple (`convert.py`
`FP4_TABLE`: nibble values `0, .5, 1, 1.5, 2, 3, 4, 6` with the sign in
bit 3, LOW nibble = even element, scale `[n, k/32]`). The shared expert is
fp8 `[2304, 5120]`/`[5120, 2304]` with e8m0 `[72, 160]`/`[160, 72]`
(32×32 blocks). The engine's `GlmMoeLayer` runs this chain today for two
scoring rules; `sqrtsoftplus` is a third `MoeRouterMode` (D3), the clamps
are GLM-5.3-Flash's (`swiglu_limit`), and the MXFP4 triple is a fourth
expert format beside FP8 / NVFP4 / packed-int (D2).

### 1.6 Engram (the reference `Engram`, `ParallelEngramEmbedding`, `NgramHashState`)

Hashing (token ids only, so every row a step needs is known before the
step starts):

```
c_t   = token_map[x_t]                                    # 129,280 → 99,092 classes (§1.6.1)
y_s   = c_{t−s} if t−s ≥ 0 else pad (= token_map[2]),  s = 0..3       # no EOS reset; image spans are DEAD (unused)
p_s   = y_s · mult[layer][s]                              # int64, odd multipliers (§1.6.1)
r_1   = p_0 ^ p_1;  r_2 = r_1 ^ p_2;  r_3 = r_2 ^ p_3     # the 2-, 3-, 4-gram hashes
id[layer][n][h] = (r_n mod prime[layer][n][h]) + offset[layer][n][h]      # 3 n-gram sizes × 8 heads = 24 rows
```

The module, on the four streams `x ∈ [T, 4, 5120]`:

```
e     = concat over the 24 rows of bf16(e4m3 row × 2^scale per 32)         # [6144]; table fp8 + e8m0 [·, 8]
kv    = W_kv e                                                             # wkv fp8 [25600, 6144]
key   = kv[:20480].view(4, 5120) (fp32); value = kv[20480:] (fp32)
rstd_i = rsqrt(mean(x_i²) + 1e-20) · rsqrt(mean(key_i²) + 1e-20)         # per stream
dot_i  = Σ_d x_i[d]·(q_weight[i,d]·k_weight[i,d])·key_i[d] · rstd_i · 5120^-0.5
gate_i = sigmoid(sign(dot_i)·√max(|dot_i|, 1e-6))
x_i   ← x_i + gate_i · value                                                # one bf16 rounding
```

The same shape as the Qwen PLE's gate (`docs/qwen38_flash_next_plan.md`
§1.7: a signed-sqrt sigmoid gate per branch on a normalised dot) without
the conv and with a joint key/value projection.

#### 1.6.1 The derived constants (G0 produces them, the loader checks them)

- `token_map`: the reference normalises every token's decoded text
  (NFKC → NFD → strip accents → lowercase → collapse `[ \t\r\n]+` to one
  space → a lone-space sentinel → strip; a token decoding to U+FFFD keys by
  its raw form) and numbers the distinct results in id order; the count
  must be exactly `engram_compressed_vocab_size` 99,092 or every hash is
  wrong. Produced once with the `tokenizers` library
  (`tools/dsv41_engram_tables.py`), stored as `engram_token_map.i32` beside
  the checkpoint's snapshot with the primes and multipliers, checked at
  load against the config (count, `pad` class). The engine's own
  tokenizer must decode the same strings; the tool cross-checks 129,280
  decodes against the C++ tokenizer.
- primes: 48 consecutive primes above 15,999,999 handed out in
  (layer, n-gram, head) order, never reused: layer 0's rows total
  384,006,168, layer 1's 384,016,682 — the table sizes in the config,
  which pins the sequence.
- multipliers: `numpy.random.default_rng(10007 × layer_id).integers(0,
  ⌊(2^63−1) / 99092⌋ / 2, size 4) · 2 + 1` per Engram layer (PCG64; the
  tool records the eight values, the loader never recomputes them).

### 1.7 DSpark (the reference `DSparkBlock`, the DSpark paper)

Three draft layers `mtp.0..2` of the backbone's shape (ratio 0: window
only; MoE of 128 experts top-3, fp4; their own mHC) plus:

```
main_x   = RMSNorm(W_main [h_37 | h_38 | h_39])          # main_proj fp8 [5120, 15360]; h_l = the stream mean of layer l's attention input
draft rings: every accepted real token appends fp8(rope(kv_norm(W_kv main_x))) to each draft layer's ring   # the draft's "window KV"
draft input: tokens [next, noise, noise, noise, noise] at positions p+1..p+5, embedded (the shared table), ×4 streams
draft attention: the 5 query rows attend to the ring's ≤ 128 rows AND all 5 draft rows (bidirectional inside the block)
logits_k = head(RMSNorm(hc_pre(x, pre)))_k,  k = 1..5                                 # the shared head, 5 rows
for k = 1..5:  logits_k += embed_M(tok_{k−1}) · head_M^T ;  tok_k = sample(logits_k)   # Markov: [129280, 256] × 2, sequential
conf_k   = σ(w_c · [x_k | embed_M(tok_{k−1})])                                        # confidence_head [1, 5376] fp32
```

`tok_0` is `next` (the token the main model just produced). The drafts
`d_1..d_5` = `tok_1..tok_5` are verified by ONE main forward of the six
rows `[next, d_1, …, d_5]`; greedy acceptance is the engine's
`judge_verify` (row `k` stands while row `k − 1`'s argmax equals `d_k`),
sampled acceptance the paper's `min(1, p_t/p_d)` with the draft's
Markov-biased softmax as `p_d` and residual sampling on rejection (the
Qwen precedent). The paper's scheduler picks a verification length per
request from the survival products `a_k = Π_{i≤k} conf_i` and the engine's
step-time curve; on this engine the six-row verify costs what a one-row
step costs (bytes, not rows, D8), so a single stream always verifies the
whole block and the scheduler matters only when live requests compete for
the 32-row batch (§5, the optimization gate). `kSpecRows` becomes 6 for
this family (D8).

### 1.8 Prefill under the CED

The decoder's global KV (layer 20's cache and index keys, read by layers
20–39) is a projection of the encoder output, and the decoder's own
per-layer state is only its window rings. The model's serving stack
therefore prefills a prompt of N tokens through the 20 encoder layers,
computes layer 20's compressor and index keys for all N, and runs the 20
decoder layers over the LAST 128 tokens only ("Decoder SWA Bounded
Replay", tech report §2.2 and §3.2.2: a query at position i attends window
keys in `[max(s, i − 127), i]` with s the replay start; the report's
evidence says the quality cost is negligible, and it is also how their
prefix cache resumes). The reference `model.py` runs all 40 layers over
the prompt (exact rings). The engine builds both (D8): `exact` (40 layers,
the parity mode, every gate) and `bounded` (the production default, half
the prefill FLOPs: 8 B active per prompt token). The Engram layers (1 and
14) and the draft rings (from layers 37–39's inputs, last 128 tokens) fall
on the right side of the split in both modes.

### 1.9 Tokenizer, prompt format, tools

`tokenizer.json`: byte-level BPE, 128,000 merges-vocab + 1,283 added tokens
(ids to 129,279; `vocab_size` 129,280), no normalizer, a pre-tokenizer
SEQUENCE of three `Split` steps (`\p{N}{1,3}`; CJK/kana runs
`[一-鿥぀-ゟ゠-ヿ]+`; then a main pattern with a punctuation-letter clause,
`[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`, ` ?[\p{P}\p{S}]+[\r\n]*`,
`\s*[\r\n]+`, `\s+(?!\S)`, `\s+`) each with `Isolated` behaviour, then
`ByteLevel` without its regex; `post_processor` `ByteLevel` (no BOS
added: `add_bos_token` false — the prompt string carries the BOS token
literally). The engine's tokenizer takes a third pre-tokenization pattern,
this time a three-stage split (D11).

There is no Jinja template. `encoding/encoding.py` (979 lines, five golden
cases) is the prompt format: `<｜begin▁of▁sentence｜>{system}<｜User｜>{user}
<｜Assistant｜></think>{answer}<｜end▁of▁sentence｜>` in chat mode,
`<｜Assistant｜><think>{reasoning}</think>{answer}` in thinking mode with
the effort prefix `<｜System｜>Reasoning Effort: {1..100} (range 1-100, the
higher the value, the more thorough the reasoning)\n\n` at index 0 only
(`low` 50, `high` 75 = default, `max` 100), `<｜System｜>` mid-conversation
messages, prior turns' reasoning dropped unless tools are present, tool
schemas injected into the system prompt, tool calls as
`<｜DSML｜ calls><｜DSML｜ invoke name="f"><｜DSML｜ parameter name="p"
string="true|false">v</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>`
(note the leading space in the tag names — the V4.1 change), tool results
as `<tool_result>` blocks merged into the following user turn, namespaces
as `ns::name`. The engine gets a native (non-Jinja) template backend for
this format, a DSML tool parser (`ToolFormat::kDsml`) and grammar, and a
golden set generated by the reference (D11).

### 1.10 Tensor census (the headers, 2026-09-13)

48 shards. Shards 3–42 hold one backbone layer each (6.88 GiB); 1–2 the
globals; 43 the draft globals; 44–46 the three draft layers; 47–48 the two
Engram tables (94.56 GiB each: `layers.{1,14}.engram.embed.weight` F8_E4M3
`[384,0xx,xxx, 256]` and `.scale` F8_E8M0 `[·, 8]`).

| class | tensors | dtype / shape | GiB |
|---|---:|---|---:|
| Engram tables | 4 | as above | 188.83 |
| routed experts, 40 × 384 | 92,160 | `w1`/`w3` I8 `[2304, 2560]` + E8M0 `[2304, 160]`; `w2` I8 `[5120, 1152]` + E8M0 `[5120, 72]` | 268.95 |
| draft experts, 3 × 128 | 2,304 | the same shapes | 6.72 |
| `attn.wq_b` (+ draft) | 80 | F8_E4M3 `[32768, 1280]` + E8M0 `[1024, 40]` | 1.68 |
| `attn.wo_b` | 80 | `[5120, 8192]` + `[160, 256]` | 1.68 |
| `attn.wo_a` | 80 | `[8192, 4096]` + `[256, 128]` | 1.34 |
| `attn.wq_a` | 80 | `[1280, 5120]` + `[40, 160]` | 0.26 |
| `attn.wkv` | 80 | `[512, 5120]` + `[16, 160]` | 0.11 |
| shared experts (+ draft) | 258 | F8 `[2304, 5120]`/`[5120, 2304]` + E8M0 `[72, 160]`/`[160, 72]` | 1.42 |
| `embed.weight`, `head.weight` | 2 | BF16 `[129280, 5120]` | 2.47 |
| Engram `wkv` | 2 | F8 `[25600, 6144]` + E8M0 `[800, 192]`; `q_weight`/`k_weight` BF16 `[4, 5120]` | 0.29 |
| `hc_{attn,ffn}_fn` (+ draft) | 86 | F32 `[24, 20480]`; `_base` F32 `[24]`, `_scale` F32 `[3]` | 0.16 |
| routers | 43 | `gate.weight` BF16 `[384, 5120]` (draft `[128, 5120]`), `gate.bias`/`bias_vl` F32 | 0.15 |
| indexers (8 layers) | 32 | `wq_b` F8 `[4096, 1280]` + scale; `weights_proj` BF16 `[32, 5120]`; kv sources also `wk` BF16 `[128, 512]`, `k_norm` | 0.04 |
| compressors (4 layers) | 11 | `wkv` BF16 `[512, 5120]`; ratio 2 also `wgate` BF16 `[512, 5120]`; `norm` | 0.03 |
| DSpark extras | 5 | `main_proj` F8 `[5120, 15360]`; `markov_head.{embed,head}` BF16 `[129280, 256]`; `confidence_head.proj` BF16 `[1, 5376]`; `main_norm`, `norm` | 0.20 |
| norms, sinks | 200 | `attn_norm`/`ffn_norm`/`q_norm`/`kv_norm` BF16; `attn_sink` F32 `[64]` | 0.00 |
| vision + aligner + image tokens | 266 | BF16 | 0.90 |
| **total** | **96,085** | | **475.24** |

The MTP layers tie `embed`/`head` to the backbone's (`convert.py` skips
`mtp.*.embed.weight` / `head.weight`; the index has none).

## 2. Placement and cost by world size

TP only, world 4 (D1). Placement per class: `wq_b` and `wo_a` by head
group (16 heads = 2 output groups per rank), `wo_b` column-packed (one
fold), experts and the shared expert sliced on the intermediate dim
(576 per rank = 18 MXFP4 blocks, 18 fp8 32-blocks — every slice
block-aligned), `wq_a`/`wkv`/compressors/indexers/routers/mHC/norms
replicated (the latent, the selection and the coefficients must be
rank-identical; the indexer sharded as the reference does would cost one
`[T, entries]`-wide reduction per index layer), Engram tables by hash head
(heads `{2r, 2r+1}` of each n-gram size → rows `[off, off')` contiguous per
(n-gram, head), the `wkv` K-slice of those 6 × 256 columns, the `[25600]`
partial riding the preceding boundary fold), `head` and the Markov head
vocab-sharded, `embed` and the Markov embedding replicated, the draft's
`main_proj` replicated (a fold-free 79 MB; sharding it is a measured
item), every KV cache and ring replicated.

### 2.1 Resident bytes per rank (W = 4)

| class | GiB |
|---|---:|
| routed experts, MXFP4 (0.53125 B/weight), sliced | 67.24 |
| draft experts, sliced | 1.68 |
| shared experts, fp8, sliced | 0.35 |
| `wq_b` / `wo_a` / `wo_b` slices (43 layers) | 1.18 |
| replicated per layer: `wq_a` + `wkv` (fp8), router (bf16), `hc_fn` (fp32), norms | 0.68 |
| indexers, compressors, Engram `wkv` slice + gates | 0.14 |
| `embed` (replicated) + `head` slice | 1.54 |
| DSpark: `main_proj`, Markov (embed replicated, head slice), confidence | 0.15 |
| **weights** | **72.97** (W = 2: 143.8 — does not fit) |

The caches: 890 B per context token (§1.3), 3.5 MiB of rings per request
slot, ~250 KiB of draft rings, tails and Engram context per slot; a
1.5 GiB prefix arena holds ~400 snapshots. Beside the ~14 GiB CUDA context
and 2 GiB of arena/scratch/staging, about **30 GiB per node stays free for
the page cache** under the 5 GiB headroom — that is where hot Engram rows
live (§2.4). The memory plan is the authority (G3).

### 2.2 Decode traffic and the floor (T = 1, per rank, W = 4)

| term | MB |
|---|---:|
| routed experts (6 of 384 × 3 slices × 40 layers, MXFP4) | 1,128 |
| `wq_b` / `wo_a` / `wo_b` slices | 1,176 |
| shared experts (fp8 slices) | 354 |
| `head` slice (bf16) | 331 |
| `wq_a` + `wkv` replicated (fp8) | 367 |
| routers replicated (bf16) | 157 |
| `hc_fn` replicated (fp32) | 157 |
| Engram `wkv` slice + 12 rows | 79 |
| indexers (8) + compressors (4) replicated | 82 |
| **total** | **3,831 → 16.7 ms at 230 GB/s (14.0 at 273)** |

Replicated share 20 %. Collectives: two folds per layer (`wo_b`, MoE) +
the head exchange ≈ 81 recorded nodes per T = 1 step, ~4 ms at the
measured 47–60 µs each (`docs/qwen38_flash_next_plan.md` D3). Kernel
count per layer is high (mHC × 2, attention with two sources, compressor
on 4 layers, indexer on 8, MoE), so the honest T = 1 expectation is
**22–26 ms per step** before optimisation. A DSpark pass adds the draft
(three layers at top-3 of 128, `main_proj`, five Markov GEMVs, the shared
head over five rows: ≈ 0.7 GB, 3 ms, 7 collectives) to a six-row verify
whose weight bytes are the T = 1 step's: **≈ 26–30 ms per pass**; at the
paper's production block of five (60–85 % faster than MTP-1 at matched
throughput) the expectation is 3–4 accepted tokens per pass, **7–9 ms per
token**. All of it is replaced by measurements in §6.

Levers after the port, in order of bytes: `hc_fn` in bf16 (−79 MB, a
quality decision), the router in fp8 (−79), `main_proj` sharded (−59 per
pass, +1 fold), and the collectives (the bus, not the model).

### 2.3 Context budget

890 B per token per rank: 1 M tokens = 0.87 GiB. At four request slots the
per-slot rings add 15 MiB. The KV pool is sized by the position limit and
the prefix arena, not by memory; `kv_capacity` 1,048,576 costs under 1 GiB.
The practical ceilings are elsewhere: the prefill select at layer 20 scans
every visible entry per query (§7), and the scheduler's prompt budgets.

### 2.4 The Engram gather

Per token: 24 rows per layer, 48 in all, 264 B each (12.4 KB). Per rank at
world 4: **12 rows per token** (2 heads × 3 n-gram sizes × 2 layers), one
4 KiB page fault each when cold. Decode: a six-row verify needs 72 rows,
issued in parallel from a host node forked before layer 0 and joined at
layer 1 (the Qwen `QwenPleLayer::stage`/`embed` split); layer 14's rows are
gathered in the same pass and joined 13 layers later. At the measured
~90 µs per cold page and ~110 K IOPS deep (Qwen, 2026-09-10), 72 parallel
faults cost ~0.2 ms, inside layer 0's ~0.5 ms. Prefill: 12 rows per prompt
token per rank = 25 K faults per 2,048-token chunk = ~0.25 s at 110 K IOPS
against ~1–2 s of chunk compute; the ids of every prompt token are known
before the first chunk, so the gather runs ahead of the walk by one chunk
(D5) and only the first chunk's rows are exposed. The page cache (≈ 30 GiB
free per node, 16 % of the two tables by bytes, more by hit rate under a
Zipfian n-gram distribution) is measured, not assumed: the gather reports
faults per step in the stats line.

### 2.5 Prefill

Bounded mode (D8): 20 encoder layers + layer 20's projection over N
tokens, 20 decoder layers over 128 — 8 B active per prompt token,
≈ 16 GFLOP per token, 8,192 tokens ≈ 131 TFLOP. GLM-5.3-Flash (6 B active,
NVFP4 experts through the fp4 tile kernel) prefills 8,192 tokens in 5.7 s
at world 4; the expectation here is **7–10 s at 8 K in bounded mode** once
the MXFP4 tile kernel exists (G2) and about twice that in exact mode or
through the grouped GEMV chain. The one-time expert stream (every expert of
every layer touched by a 2,048-token chunk) is 67 GiB per rank per layer
sweep — the resident weights, read once per chunk.

## 3. Reuse, adapt, new

| module | status | note |
|---|---|---|
| CollectiveBus, roster, graph engine, eager engine, boundary reducer, prefix arena, resident image, scheduler, HTTP/SSE, sampler, JSON grammar, memory plan | reuse | generic since Q1/D7 |
| `ResidentLayerStream<F>`, `WeightBuilder`, safetensors mmap, `hf_cache` | reuse | a new family struct |
| `fp4_gemv.cuh` core, slot/grouped/tile launchers (`glm_moe.cu`) | adapt | the MXFP4 scale variant (e8m0 per 32, no global), D2 |
| fp8 GEMV/GEMM on a 32-grid (`fp8_gemv.cuh`, `scale_gemm`, `quant_matrix`) | reuse | the Qwen D2 re-blocking already takes 32×32; e8m0 scales converted to fp32 at load |
| `GlmMoeLayer`, router kernel, swiglu clamps, chain accumulation | adapt | `sqrtsoftplus` mode, the MXFP4 view, top-6 / top-3 |
| `glm_mhc.cu`, `models/glm/mhc.hpp` | adapt | single-pass (`pre_in`/`pre_out`), fp32 `fn`, eps, one-rounding update, weighted final collapse, D4 |
| Qwen n-gram table mmap, host node, staged gather/convert, hash kernel, context rows | adapt | two tensors per row (payload + scales), 3 n-gram sizes × 8 heads, the token map, pad semantics, D5 |
| Qwen PLE gate | pattern | the Engram gate/value kernel is new but the same shape |
| DSA select kernels (`dsa_select_decode/prefill`, relu, select_k 512, kpool 1), the index cache layout, the split/listed attention with (m, l, c) partials and `dsa_attn_combine` | adapt | visibility `(pos+1)/ratio`, the candidate pool and the restricted select, the two-source attention (ring + cache) with the sink and K == V (kv_lora 512, rope inside), D6 |
| `dsa_kpool_*` (gated pooling, tail ring, per-row tail snapshots) | pattern | the compressor at ratio 2 on the 512-wide latent, D6 |
| `dsa_rope_interleave`, `dsa_rope_table_host` | adapt | YaRN table, fp32 one-rounding policy, inverse rotation of the output |
| `latent_format.hpp` fp4 codec | adapt | e4m3 per 16 WITHOUT the row scale (the model's KV format); e8m0 per 32 for the index keys and window rows |
| MTP transaction, `judge_verify`, graph feed, pick kernels | adapt | `kSpecRows` 6, the block draft in one pass, the Markov chain, sampled acceptance with `p_d`, D8 |
| tokenizer | adapt | a three-stage split pre-tokenizer, the added-token set, goldens, D11 |
| chat template | new backend | the DeepSeek V4.1 encoder (no Jinja), D11 |
| tool parser / grammar | new | DSML, D11 |
| config, binding, loader, layer walk, session hooks, serve family, apps, scripts, fixtures, reference dumps | new tree | `src/models/dsv41/`, §4 |

## 4. Design decisions

**D1 — A new family `deepseek_v41` on the shared cores, world 4, TP only,
text only.** `src/models/dsv41/` with `Dsv41Model : SessionModel<Dsv41Model>`
and `Dsv41LayerStream : ResidentLayerStream<Dsv41LoaderFamily>`;
`ModelArchitecture::DeepseekV41` for `DeepseekV41ForCausalLM` (and the
`deepseek_v41` type as the fallback). No expert parallelism (the Qwen
argument: equal bytes per rank at every routing beats a busiest-rank
tail); no world 2 (memory) or 3 (64 heads, 8 groups, 32 index heads and
129,280 vocabulary rows do not divide); the geometry check refuses them by
name. Image inputs are refused at the API; `bias_vl`, the vision tower,
the aligner and the image delimiters are not loaded.

**D2 — MXFP4 is a scale variant of the fp4 core, not a new core.**
`fp4_gemv.cuh` gains a compile-time scale mode: `kNvfp4` (e4m3 per 16,
÷ global in the epilogue — untouched, bitwise) and `kMxfp4` (e8m0 per
32: the 16-byte chunk's 32 codes share ONE scale byte, the scale is
`2^(byte − 127)` applied as an exponent add on the fp16-placed code, no
epilogue division, NaN scale 0xFF propagated). The dequantized weight
`e2m1 × 2^k` is exact in bf16 and fp32, the FMA chain, lane geometry and
xor tree are the NVFP4 core's, so the slot, grouped and sliced-fold
launches stay bitwise twins per row. `GlmFp4Matrix` gains `scale_group`
∈ {16, 32} with `global_scale` null for MXFP4; `MoeExpertView` a mode
byte; the slot, grouped-GEMV and ldmatrix tile kernels (`glm_moe.cu`:
fragment-time decode) take the mode as a template parameter; the shared
expert stays fp8 at view-table entry `n_experts` (the composed-hybrid
shape, `shared_view_base = −1`). Column slices start on a 32-element block
(true of every production slice: `w2` K = 576 per rank); K = 5120 and 576
join the compiled set (576 = 18 chunks → 2 lanes × 9 chunks in passes of
4, the GLM-4.7 non-power-of-two rule). The host codec (`loaders/mxfp4.hpp`:
decode for the oracles and the checkpoint pin; no encoder is needed since
nothing is requantized) reproduces `convert.py`'s nibble table.

**D3 — `sqrtsoftplus` is a third `MoeRouterMode`, the clamps are Flash's,
top-k is 6/3.** Scores `sqrt(softplus(logit))` in fp32 (softplus with the
usual threshold), the bias on the selection key only, the picked scores
normalised with `+ 1e-20` and × 1.5 — one branch in `launch_moe_router`'s
scoring functor, the same tie rule (lower id), the same ascending output.
The experts' `swiglu_limit` 10 clamps are `GlmMoeConfig::swiglu_limit`.
The draft MoE is the same layer at 128 experts / top-3.

**D4 — Single-pass mHC is a parameter set of the GLM mHC kernel.**
`launch_mhc_compute_normed` gains `pre_in` (the collapse coefficients to
USE) and `pre_out` (the ones to EMIT), an `fn` dtype (bf16 or fp32), the
one-rounding stream update (`launch_mhc_stream_update` with a policy flag:
GLM's two roundings or the reference's fp32 sum rounded once), `norm_eps`
1e-20, and a weighted final collapse beside `launch_mhc_final_mean`. The
GLM path passes `pre_in = pre_out` and keeps every bit (the gate:
`glm_mhc_test` unchanged). The deferred-comb side stream and the fused
finish stay as they are: comb is consumed at the stream update on both
families. The `hc_mixes` normalisation-after-projection form
(`(flat @ fn^T) · rsqrt`) is what the tensor-core prefill form already
computes; the per-coefficient decode form multiplies the finished dots by
`rsqrt`.

**D5 — The Engram tables stay on the NVMe, head-sharded, gathered by the
host node, the partials folded through the preceding boundary.**
`QwenNgramTableMmap` generalises to a table of two parts per row set
(payload `[rows, 256]` e4m3 and scales `[rows, 8]` e8m0, both mapped from
the same shard with `MADV_RANDOM`, rows gathered into pinned staging
`[M, 12, 256 + 8]`), the staged convert kernel dequantizes `bf16(e4m3 ×
2^k)` per 32 (exact), and the hash kernel takes the token map, four
lookbacks with the pad rule, per-layer multipliers, and 24 (prime, offset)
pairs per layer. Per rank: heads `{2r, 2r+1}` of every n-gram size at
world 4 (rows are contiguous per (n-gram, head): six ranges), `wkv`'s K
columns for those heads (six 512-wide column blocks of the fp8 matrix,
re-blocked on the 32-grid), the `[T, 25600]` bf16 partial written into the
reducer's staged buffer and summed on layer 0's / layer 13's MoE fold (the
Qwen D4 mechanism; the reducer's staged width grows to 25,600 + 5,120 at
those two boundaries), then the gate and the stream update on the reduced
key/value at the top of layers 1 and 14. Decode stages both layers' rows
before layer 0; prefill stages chunk `i + 1`'s rows while chunk `i` runs
(the ids of the whole prompt are known; the draft rows' ids too, once the
draft has sampled). A resident mode (`engine.engram_table = "resident"`)
is refused at world 4 with the reason (47 GiB per rank does not fit beside
73 GiB of weights); the knob exists for a wider world. The gather reports
its page-fault count per step; the LibertAI-style FP4 re-pack is the
recorded lever if the hit rate ever matters.

**D6 — CSA2 is a new layer object built from the DSA kernels' pieces.**
`Csa2Layer` (`src/models/dsv41/csa2_layer.{hpp,cu}`) owns: the fused
`[wq_a | wkv]` projection (fp8, 32-grid), the two RMSNorms, `wq_b` per
local head, the window append (fp8 e8m0/32 quantize, rope at `pos`, ring
slot `pos % 160`), the compressor at ratio 1 (a projection + norm) and 2
(the pair pooling with the per-channel gate softmax on the
`dsa_kpool_decode_update` pattern: a per-request tail row and per-row tail
snapshots for rollback; prefill pools whole pairs and stashes the odd
tail), the index-key path (`wk`, `k_norm`, rope at `j·ratio`, fp4 e8m0/32
quantize-dequantize, stored in the planar e4m3 + row-scale index cache with
an exactness counter), the main-KV append (rope at `j·ratio`, fp4 e4m3/16
codes with block scales — a fourth `LatentFormat`, `kFp4Block` without the
row scale), the indexer query path (`wq_b` fp8 → rope → fp4 e8m0/32
quantize-dequantize → the fp8 planar form the select streams), the select
(`dsa_select_decode/prefill` with `relu`, `select_k` 512, `kpool` 1,
visibility `(pos + 1) / ratio`, and a candidate list: layer 20's select
also emits the top-2048 block mask with the newest block pinned; the
Reindex layers' select streams only the 16,384 candidate entries), the
attention as TWO partial sources merged by `dsa_attn_combine` — the ring
(a dense 128-row source with the sink folded in as a partial of `m = sink,
l = 1, c = 0`) and the selected cache rows (the listed/split kernels at
`kv_lora` 512, `rope` 0: the rotated tail is inside the 512 and the value
IS the row) — the inverse rotation of the output's last 64 dims per head,
the grouped `wo_a` (two `[1024, 4096]` fp8 GEMVs per rank at decode, a
grouped GEMM at prefill) and `wo_b` with the fold. Two rope tables per
model (window and YaRN), built on the host in double, rounded to fp32 then
bf16 (the `dsa_rope_table_host` rule) with the reference's one-rounding
application as a policy flag of `dsa_rope_interleave`. The walk enqueues
Reuse layers with a selection-reusing view and Reindex layers with the
candidate mask, the full GLM-5.3's "select once, attend N times" contract
(`DsaLayer` D4) restated for eight owners and one candidate source.

**D7 — Every cache is the model's format; `kv_dtype` is refused; rings
need no rollback.** Main KV fp4 e4m3/16, index keys fp4 e8m0/32 (stored as
exact e4m3 + row scale), window rows fp8 e8m0/32, in the paged pool
(main + index caches, blocks of 128 entries, four cache ordinals) and the
per-request rings (160 slots, §1.4). The compressor tails (ratio-2 layers:
2 × 512 fp32 × 3 layers per request) and the Engram context (3 ids) are
the state families with per-row snapshots; the rings and the paged caches
are positional (a rejected row's entries are overwritten by the next
accepted ones, and the pool's block table is grown by admission as for
DSA). `session_snapshot_align` = 128 entries of the ratio-1 cache = 128
tokens (256 for the ratio-2 caches, which the pool's block size handles
by holding 64 entries per block on those ordinals).

**D8 — DSpark is the MTP contract at `kSpecRows` = 6 with a block draft;
prefill has two modes.** `kSpecRows` becomes a per-family constant (6
here; `kSpecMaxDrafts` 5, the pick tables and spec tables sized by it),
`mtp_depth` for this family means the verified block length 1–5 (default
5). `mtp_run_rows` runs the three draft layers ONCE over the block's five
rows (bidirectional attention inside the block over the ring's 128 rows),
the shared head over the five rows, then the Markov chain on device (five
sequential `[256] × [vocab/W, 256]` GEMVs against the rank's head slice,
each biasing the row's logits before that row's pick — the engine's
candidate-table pick kernels run per row, so the chain is five pick
launches with a Markov GEMV between them) and the confidence dots. The
draft rings receive the accepted rows' `main_x` (from the verify walk's
per-row `[h_37 | h_38 | h_39]` scratch) before the draft runs. Greedy
first (bitwise the plain transcript, `mtp_depth_check.py`), sampled
acceptance second with `p_d` = the Markov-biased softmax of the draft row
(five floats kept per request per pass) and residual sampling (the Qwen
precedent). The confidence-scheduled row allocation across live requests
(the paper's Algorithm 1 over the 32-row batch) is the optimization gate's
first item; until then every slot verifies its whole block and the
batch-family selection is the engine's. Prefill mode: `engine.prefill =
"bounded"` (default; the encoder over the prompt, layer 20's projection
over the prompt, the decoder over the last 128 tokens with the window
truncated to the replay segment) or `"exact"` (40 layers, the parity
mode); chunked prefill in both (2,048-token chunks; bounded mode runs the
decoder on the last chunk's tail only).

**D8a — Confidence-scheduled verify depth (the optimization gate's first
item; 2026-09-14).** The draft's `confidence_head` emits a raw acceptance
LOGIT per block position (`dsv41_dspark.cu`; no sigmoid), so
`sigmoid(c_i) = P(draft i accepts | its prefix survived)` and
`S_i = Π_{j<i} sigmoid(c_j)` is the prefix-survival probability, monotone
non-increasing. Verifying `k` of the block's drafts runs `1+k` rows and
commits the greedy-matching prefix — EXACTLY the tokens a full-block
verify commits for that prefix — so `k` changes throughput only, never the
output (a draft is committed iff it equals the target's argmax there, and a
draft not verified is decoded plainly next step). The policy
(`models/dsv41/dspark_schedule.hpp`, `dsv41_verify_depth`) maximizes
aggregate decode throughput `Σcommitted/Σtime`, NOT the per-step ratio
`committed/time` (maximizing a sum of ratios is the wrong surrogate for a
ratio of sums, and the two disagree exactly when confidence varies across
steps — chat vs counting). Its Dinkelbach optimum is the per-step linear
rule at one global multiplier `λ` (the achieved throughput, tokens/ms):
verify draft `i` while `S_i > λ·row_ms`, stop at the first below (S
monotone ⇒ the kept drafts are a prefix). The fixed base cost drops out of
the depth decision; it only sets `λ`. **Cross-rank determinism:** `λ` must
NOT be measured wall-clock (each rank's jitter differs ⇒ ranks would pick
different `k` ⇒ the TP collectives desync); it is a configured constant
(default the reservation rate `1/(base+row)` of the profiled curve, or a
future EWMA over the identical committed counts and a modeled time). With a
constant `λ` and the replicated confidence (folded `collapsed_` + the
replicated draft head), every rank derives the same `k`. Six unit tests
(`dsv41_dspark_schedule_test`) pin the economics (hot⇒full, cold⇒0,
monotone in `λ`, deeper as the base grows). **Exactness proven on the model
code, not just argued:** `GreedySpeculator::set_depth_policy` feeds only a
policy-chosen prefix of the block to each verify; `dsv41_decode_test` §7b
runs full depth, adversarial `k=1`, and the confidence policy, and all
three reproduce the plain greedy transcript bitwise (the k=1 path proves it
through real draft conditioning + rollback). **Graph-engine port (the
remaining work):** `rows_per_request_` is baked at capture, so variable
depth needs extra captured verify variants selected per step from the
replicated confidence — the model already accepts a per-capture
`rows_per_request` (`session_graph_capture_batch`). The step graph is
verify(N)→verdict→draft(N+1), so N+1's block confidence is produced at the
tail of N's graph and is available before N+1's verify launches (no stale
content); the depth read piggybacks on the verdict readback, and the
pipelined path (`pipeline_`) is the subtle case (the draft follows the
verdict node, so the conf copy must be stamped by the graph or read after
the settle). Land it behind a default-off flag so the 99/99 suite and the
production full-depth path are untouched, then A/B on the fabric — the A/B
simultaneously measures the win, validates calibration (a win ⇒ the
confidence is calibrated), and re-checks exactness (transcript equality).

*Built (2026-09-14, later the same day).* The policy moved to the engine
layer (`engine/verify_schedule.hpp`: `scheduled_verify_depth`,
`verify_reservation_lambda`; `verify_schedule_test`) — it is
model-agnostic, keyed by `Model::kVerifyConfidence` (true on `Dsv41Model`,
which exposes `confidence_rows()` and `device_confidence()`; false on the
session-core base and on GLM-5.3-Flash's own model). The graph engine
(`configure_verify_schedule`, before the warm capture) captures, per slot,
one reduced-row variant per depth option after every scalar and batch
variant (`sched_variant`; the bus's 32 variants bound the options — an
even spread that keeps the full block when short, a policy depth rounding
up), all built by one `record_scalar_mtp(req, rows)`: the verify over the
first `rows` of the slot's persistent feed (the session core's
`session_graph_capture_step(..., feed_rows)` keeps the feed at the full
block's rows so every depth variant shares it), the draft over those rows
(its input rows are the accepted prefix plus padding, so it needs no
change; the block runs its full width), the chain rows, the full-width
feed write (the next-tokens check reads the feed rows, not the verify's),
and — scheduling — a `glm_publish_f32` kernel node that copies the slot's
block confidence to a pinned mirror and releases a per-slot sequence; the
batch variant publishes every slot's. `step_scalar` waits for the slot's
last publication, applies the policy (a test hook can replace it), picks
the option, stages the feed prefix and replays that variant; the verdict
readback takes the replay's rows (`Replay::rows`); a fresh, re-drafted or
sampled slot verifies the whole block. The wait is the one cost: the
pipelined replay's launch-ahead is lost on the scheduled path (the graph
to launch is unknown until the confidence lands), ~0.5 ms a step on a
stream the policy never shortens. Gates: `dsv41_engine_test`'s scheduled
world (29964) — a forced depth sequence 1,4,2,5,3,… over every reduced
variant, the batch's publication, a scalar step over it, both ranks
identical, every transcript the plain eager engine's. Server knobs:
`engine.mtp_schedule{,_row_ms,_base_ms,_lambda,_min_depth}` /
`--mtp-schedule…`, journaled to every rank (`mss`, `msrow`, `msbase`,
`mslam`, `msmin`). *Extended the same day:* (1) **the batched replay** —
per family one reduced-row variant per depth option (`sched_batch_variant`,
after the reduced scalar ones; the budget is `32 / (2 · (slots +
families))` options), built by `record_batch_mtp(family, rows)`: the
session core's `session_graph_capture_batch(rows, k, feed_rows)` compacts
the first `rows` of every slot's feed into the front token scratch
(`glm_spec_gather_feed`) so the walk's rows stay contiguous per request,
the draft runs over those rows, the next-tokens write lands in the feeds
at the full width; the pick's rows, positions and mask rows are the
compact layout (`stage_masks_compact`: every header written, since the
per-slot and compact layouts overlap — and `stage_masks` re-zeroes an
unconstrained slot's headers when scheduling is on); the batch takes one
depth, the deepest any live slot asks for (exact either way), the full
block when any is fresh or sampled. (2) **Every MTP family**: a family
without a confidence head (`kVerifyConfidence` false — GLM-5.3-Flash,
Qwen3.8-Flash-Next, GLM-4.7, the full GLM-5.3) takes the draft head's own
probability of its pick: the draft picks read a spec table whose rows
report logprobs (`d_draft_specs_`; the argmax under the raw normalizer is
the same token, so the transcript is unchanged), the sampled verdict's
outcome carries `log p` in pinned memory, and a kernel node
(`device_sample_draft_confidence`) turns the `depth` outcomes into
`logit(p)` per position before the publication — identical on every rank
because it comes out of the pick's fold; needs the device sampler. Gates:
`dsv41_engine_test`'s scheduled world now forces per-slot depths on the
batch (the reduced batch variants replay; transcripts exact) and
`glm4_engine_test` (29953) runs a depth-2 world with real sampler scratch
under the draft-probability confidence (scalar and batched, exact). The
fabric A/B (the win, the calibration, the transcript compare) is §9's next
entry.

**D9 — Text only; the vision-language router bias is dead code.** Image
tokens never reach the model (the API refuses `image_url` content by
name), so `gate.bias_vl`, `image_start/end/newline`, `vision.*` and
`aligner.*` are listed in the binding as present-and-skipped (the Qwen
D8 rule), and the Engram token mask is always all-true.

**D10 — Numerics: exact weights, bf16 activations, the reference mirrored
at its rounding points, quality by evaluation.** The reference is the
checkpoint's own `inference/model.py`. Its rounding points (fp32 mHC,
fp32 gate softmax in the compressor, one bf16 rounding after rope, fp32
Engram gate, fp32 router, bf16 index dots) are mirrored where the engine's
kernels can (the pure-python reference dump follows the engine's chains);
its MXFP8 activation quantization is mirrored on prompt rows under
`engine.prefill_activations: native`, which the release template sets, and
not on decode rows or under `bf16`, where the engine is more precise and
the cross-check tolerance absorbs the difference, as the GLM-4.7/5.3
cross-checks absorb the same class of difference. `hc_fn` in
bf16 and the router in fp8 are levers behind `engine.hc_weights` /
`engine.router_weights` knobs, off by default, decided by
`fabric_logprob.py` and `serve_eval.py`. The end-to-end quality bar is the
task eval (HumanEval / GSM8K / extraction, DeepSeek's card reports 79.4 %
HumanEval and 93.0 % GSM8K for the base model) and the teacher-forced NLL
of the three shipped texts, recorded absolute (no reference build of this
model runs on the fabric).

**D11 — Tokenizer, prompt format, tools.** The tokenizer gets a
three-stage `Split` sequence (each stage `Isolated`: the match and the
gaps both continue to the next stage) ahead of the byte-level encoder,
the 1,283 added tokens (special ones matched before the split), no
normalizer, and goldens generated against the `tokenizers` library over
the existing golden corpus plus CJK/kana, digit-run and punctuation-letter
cases. The prompt format is a native `ChatTemplate` backend
(`text/dsv41_encoding.{hpp,cpp}`) implementing `encoding.py`'s
`encode_messages` for the roles system/user/assistant/tool/latest_reminder,
thinking on/off (`enable_thinking` → `thinking_mode`), `reasoning_effort`
as an integer 1–100 or `low`/`high`/`max` (any other string refused by
name), `drop_thinking` semantics, tool schemas in the system prompt,
`<tool_result>` merging in call order, namespaces, mid-conversation system
messages; the DSML parser (`ToolFormat::kDsml`: the spaced tag names,
`string="true|false"` typed parameters, `ns::name`) and its grammar for
constrained tool calls; goldens = the reference's five test cases plus a
generated set (the reference `encoding.py` run under the tools venv),
byte-exact renders and ids. `<think>`/`</think>` are ordinary tokens in
the vocabulary; `<｜DSML｜>`-prefixed tags are text plus the single
`｜DSML｜` token — the parser works on decoded text.

## 5. Layout

| path | contents |
|---|---|
| `src/models/dsv41/config.{hpp,cpp}` | `Dsv41TextConfig` (the nested `text_config`, `compress_ratios` → per-layer mode table cross-checked against the source lists, the Engram/DSpark blocks, `quantization_config` `fp8`+`fp4`+`ue8m0`+`[32, 32]`), `ModelArchitecture::DeepseekV41` |
| `src/models/dsv41/binding.{hpp,cpp}` | the expected-tensor table (MXFP4 triples, fp8 pairs with e8m0 scales on the 32-grid, the compressor/indexer sets on their layers, Engram, the draft, globals; vision present-and-skipped), roles, the TP geometry check (heads, groups, index heads, vocabulary, Engram heads, expert slice) |
| `src/models/dsv41/loader.{hpp,cpp}` | `Dsv41LayerStream`: e8m0 → fp32 scale conversion on the 32-grid, the MXFP4 views, the Engram table mmap (payload + scales) and sidecar check, the derived constants, byte formulas, `loader_format()`, resident images |
| `src/kernels/fp4_gemv.cuh`, `glm_moe.cu`, `glm_moe_launch.hpp`, `models/quant_matrix.hpp`, `models/glm/moe.hpp`, `loaders/mxfp4.hpp` | D2 |
| `src/kernels/glm_moe_launch.hpp` / `glm_moe.cu` router | D3 |
| `src/kernels/glm_mhc.cu`, `glm_mhc_launch.hpp`, `models/glm/mhc.hpp` | D4 |
| `src/models/dsv41/engram.{hpp,cu}`, `src/kernels/dsv41_engram.{hpp,cu}`, `models/qwen/loader.{hpp,cpp}` (the table mmap generalised) | D5; `tools/dsv41_engram_tables.py` (the sidecar) |
| `src/models/dsv41/csa2_layer.{hpp,cu}`, `csa2_state.{hpp,cpp}`, `csa2_reference.{hpp,cpp}`, `src/kernels/csa2.{hpp,cu}` (compressor, ring append, sink partial, inverse rope, candidate mask, restricted select), `src/kernels/dsa.{hpp,cu}` (visibility ratio, candidate list), `kernels/latent_format.hpp` (`kFp4Block`) | D6, D7 |
| `src/models/dsv41/{layers,model}.{hpp,cpp}`, `dspark.{hpp,cu}`, `src/kernels/dsv41_dspark.{hpp,cu}` | the walk (encoder/decoder modes), `Dsv41Model` hooks, the block draft, the Markov chain, the confidence head; D8 |
| `src/engine/decode_outputs.hpp`, `kernels/glm_spec.hpp`, `kernels/pick.hpp`, `src/serve/cluster_config.*` | `kSpecRows` per family (6), `mtp_depth` 1–5, `engine.prefill`, `engine.engram_table`, `engine.hc_weights`, `engine.router_weights` |
| `src/text/tokenizer.*`, `text/dsv41_encoding.{hpp,cpp}`, `text/tool_parser.*`, `text/tool_grammar.*` | D11 |
| `tools/dsv41_reference_dump.py`, `tools/dsv41_torch_reference.py`, `tools/checkpoint_audit.py` (the `deepseek_v41` branch → `docs/checkpoint_budget_dsv41.md`) | the oracles and the audit |
| `tests/unit/dsv41_{config,binding}_test.cpp`, `tests/unit/mxfp4_test.cpp`, `tests/cuda/fp4_gemv_test.cu` (+ MXFP4 cases), `fp4_gemv_checkpoint.cpp`, `glm_moe_test.cu`, `glm_mhc_test.cu`, `tests/cuda/csa2_test.cu`, `dsv41_engram_test.cu`, `dsv41_dspark_test.cu`, `dsv41_{loader,forward,decode,tp,engine}_test.cpp`, `dsv41_fixture.hpp` | the gates (§6) |
| `apps/dsv41_load_check.cpp`, `dsv41_forward_check.cpp`, `dsv41_gen_check.cpp`, `apps/dgpp_serve.cpp` (`Dsv41Family`) | the apps |
| `deploy/cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json` (since 2026-09-14 one template: six slots at depth 4 with the scheduled verify depth; the plain and depth-5 shapes are knobs) | model name `deepseek-v4.1-flash`, quant `mxfp4-fp8`, mode `mtp5` (the block draft: the deployment rule names the depth) |
| `scripts/fabric_dsv41_{load,forward,serve}.sh` | the fabric procedures |
| `tests/data/dsv41_{tokenizer,chat_template}_goldens.jsonl` | D11 |
| `docs/checkpoint_budget_dsv41.md`, `docs/model_cards/DeepSeek-V4.1-Flash.md` | the audit's budget, the card |

## 6. Implementation stages and acceptance gates

Order: G0 → G1 → G2 → G3 → G4 → G5 → G6 → G7; nothing merges without its
gate, every stage ends with the full build (`cmake --build build-ci -j --
-k`, never overlapped) and an unfiltered `ctest` from `build-ci` — the
existing 78 tests gate every stage because the port touches shared code
(the fp4 core, the MoE layer, the mHC kernel, the select kernels, the
n-gram mmap, the spec-row constants).

**G0 — the checkpoint on the fabric and the facts.** Rank 0 downloads
`deepseek-ai/DeepSeek-V4.1-Flash` (510 GB; ≥ 600 GB free on every node
first, `--verify-only` after the peer sync on all four); the
`checkpoint_audit.py` branch reproduces §1.10 and §2.1–2.2 from the local
headers and writes `docs/checkpoint_budget_dsv41.md` (the §2 model must
agree to 1 %); `tools/dsv41_engram_tables.py` builds the token map (exactly
99,092 classes, the pad class recorded), the 48 primes (row totals
384,006,168 / 384,016,682) and the eight multipliers into the sidecar and
cross-checks 129,280 decodes against the engine's tokenizer; the tools
venv gets `tokenizers`, `numpy`, `sympy` (this box has none of them today
— the sidecar and the goldens are generated where they are available and
committed under `tests/data/`). The prompt-format goldens are generated
from the reference `encoding.py`. Gate: the audit, the sidecar and the
goldens exist and are pinned by hash.

**G1 — config + binding.** `dsv41_config_test`: the real `config.json`
parses; every rejection names its field (a `compress_ratios` list not of
length 43, a source layer whose ratio is 0, a candidate source outside
the decoder, `engram_layer_ids` off the encoder, `hc_mult` ≠ 4,
`num_key_value_heads` ≠ 1, `weight_block_size` ≠ [32, 32], `scale_fmt` ≠
`ue8m0`, `expert_dtype` ≠ `fp4`, `scoring_func` unknown); the per-layer
mode table reproduces §1.3's. `dsv41_binding_test` + `dsv41_load_check
--bind`: the table reproduces the 96,085 names, dtypes and shapes
(`unexpected` 0; the four compressor and eight indexer layers and no
others; the two Engram rows counts; vision skipped), headers only. The
detector selects the family; the four existing families still select
themselves.

**G2 — the cores.** `fp4_gemv_test` gains the MXFP4 mode at every
compiled K (576, 5120 and the existing set): the host oracle
(`e2m1 × 2^k`, fp64 accumulation, the dot rounded to bf16; zero
mismatches at 2 bf16 ulps), rows bitwise across row counts, NaN scales
propagated, NVFP4 cases unchanged bitwise. `fp4_gemv_checkpoint.cpp`
(`checkpoint` label): one routed expert triple mmapped from a real shard,
decoded against `convert.py`'s table (the nibble-order pin on real data),
single rows bitwise batched rows. `glm_moe_test` gains the `_mxfp4` twin
of every fp8/nvfp4 gate (slot == grouped == sliced fold bitwise, the fp8
shared expert on every path, the tile kernel bitwise its dense form per
segment and within the mma-order budget of the GEMV chain) and the
`sqrtsoftplus` router against the oracle with near-tie certification.
`glm_mhc_test` gains the single-pass cases (`pre_in ≠ pre_out`, fp32 `fn`,
eps 1e-20, one-rounding update, weighted collapse) with the GLM cases
bitwise unchanged. `dsv41_engram_test`: the hash kernel against a host
oracle on random contexts including the sequence start (pad) and the
multiplier/prime pins from the sidecar; the staged gather + convert
against `bf16(e4m3 × 2^k)`; the gate/value kernel against the fp32 oracle.
`compute-sanitizer` memcheck/racecheck on the new shapes.

**G3 — loader.** `dsv41_loader_test` on a fixture (a synthetic
mini-checkpoint written from the binding table: 8 layers with the
`0,0,2,2,1,1,1,1` ratio pattern, sources at 2 and 4, index sources 2/4/6,
candidate source 4, Engram at 1 and 3 with 20 K-row tables, 8 experts, 64
heads at the real head dims — the listed attention needs `local_heads %
16` at world 4 —, DSpark with 4 experts, release-like scales): fixture
build == resident-image restore byte for byte at worlds 1 and 4; byte
formula == bump usage == source plan per class; the e8m0 → fp32 scale
conversion equals the oracle; the MXFP4 views' column slices refused off
a 32-boundary; the Engram mmap resolves both parts, refuses a mismatched
sidecar, and gathers rows bitwise a host read. `dsv41_load_check --world 4
--rank 0 --streaming --mtp` on the real checkpoint (foreground, the
`drop_caches` loop beside it): every layer placed, the resident formula
reconciled against the bump, the per-rank total recorded (§2.1 replaced),
the first image built and restored, the two 94.56 GiB tables mapped and a
sample of rows read.

*Record (2026-09-13).* `src/models/dsv41/loader.{hpp,cpp}`:
`Dsv41LoaderFamily` on `ResidentLayerStream`, the builders `load_fp8`,
`load_fp8_rows` (32-row starts), `load_fp8_col_ranges` (whole 32-column
blocks, packed in order; the e8m0 bytes become the fp32 grid the fp8
core reads), `load_mxfp4_rows` / `load_mxfp4_cols` (payload + e8m0
scales verbatim, `scale_group` 32, no global), `load_f32_as_bf16` (the
mHC `fn`, plan D10). `Dsv41EngramTableMmap`: the table's two tensors
mapped read-only from their shard (`MADV_RANDOM`), rows gathered into
the `[n, rows_local, 264]` staging with the page faults issued together
(`MADV_WILLNEED`) and the copies spread over threads past 256 rows; the
mappings outlive `release_sources()`. `tests/cuda/dsv41_fixture.hpp` +
`dsv41_loader_test` (6 gates, 762 tensors, 31 MB): every class byte-exact
at worlds 1/2/4 against the fixture slices — the fp8 grids, the MXFP4
row/column slices, the Engram `wkv` column ranges of the rank's hash
heads, the sink per head, `fn` rounded once, the draft's head tensors —
the byte formula == bump usage == source plan on every layer, the
globals (replicated and vocab-sharded embedding), the tables mapped and
a 64-token gather bitwise the shard (world 1's multi-threaded path and
world 4's one head per rank), the resident image round trip with the
digest rank-invariant across ranks, world 3 and a non-binding config
refused by name, the sidecar loaded / refused by field. Two departures
from the paragraph above: the fixture keeps 8 attention heads (the
loader has no head-count floor; a G4 fixture grows to 64 if the listed
attention's `local_heads % 16` holds), and the DSpark Markov head is
replicated per rank rather than vocab-sharded — a layer's bytes cannot
depend on the head sharding (the stream's counting pass has none), so
the `[vocab, 256]` bf16 head is held whole (66 MB per rank) and the
draft addresses its lm-head rows at use. `dsv41_load_check` (`--engram`,
`--sidecar`) runs on the fixture at every world; the real-checkpoint
run waits for the download.

**G4 — attention, mHC and Engram at the layer level.** Every existing DSA
and mHC gate stays bitwise on the GLM families after the kernel changes
(`dsa_test` 43/43, `glm_mhc_test`, `glm_forward_test`, `glm_dsa_*`,
`fabric_glm_regression.sh`). Then `csa2_test`: the compressor at ratio 2
against a double oracle (pair pooling, tails across a chunk boundary and
across decode rows, per-row tail snapshots and rollback), ratio 1, the
window ring (fill, wrap at 160, the 128-row visibility per position, the
spec-row overwrite argument checked by construction: after a rejection the
next accepted rows read exactly the oracle's window), the two rope tables
against a double oracle with the one-rounding policy (positions across
2^20, the YaRN ramp), the index-key exactness counter (zero on real
rows), the select at ratio 2 visibility, the candidate mask (block max,
newest block pinned, 2,048 blocks) and the restricted select against the
oracle with the tie rule, the two-source attention with the sink against
the host oracle at TP 1/4 on fp4/fp8 rows (kept-row drift within the fp4
budget, selection flips certified as near ties), the inverse rotation,
the grouped `wo_a`. `dsv41_engram_test` at the layer level: the fold of
the `[25600]` partial through the boundary reducer at world 2 bitwise the
world-1 projection. Then the per-layer torch cross-check on the real
weights (`tools/dsv41_torch_reference.py`: `model.py`'s own layer code
with the tilelang kernels replaced by dequantized bf16 `F.linear` and the
`act_quant` steps optional, layers 0–3 plus layer 20's compressor, on the
5090 box one layer at a time — a layer's experts are 18.8 GB): relative l2
per layer at the GLM-4.7 level (≤ 0.003) with `act_quant` off, the
difference with it on recorded as the reference's own quantization noise.

*Record (2026-09-13, the kernel level).* Two latent formats joined
`kernels/latent_format.hpp` and the DSA kernels (`LatentTile`, the append
kernels, every format switch): `fp8_block` (e4m3 + e8m0 per 32 inside the
row, 528 B at 512 — the window rows) and `fp4_block` (e2m1 + an absolute
e4m3 per 16, 288 B — the compressed main KV), both bitwise the release's
`act_quant` / `fp4_act_quant` (the host codecs mirror kernel.py: the
1e-4 and 6·2^-9 floors, the fp32 `amax * (1/448)` product, the true
division before the e2m1 encode); `dsa_test`'s append and attention gates
cover them beside fp8/fp4. `kernels/csa2.{hpp,cu}` + `csa2_test` (8
gates): the one-rounding RMSNorm; the rotary frequencies (window and
YaRN) and the one-rounding complex rotation with fp32 angles and no
table (the reference keeps complex64 — a bf16 table would not do);
`csa2_index_q_quant` / `csa2_index_k_append` (fp4 e8m0/32 stored as e4m3
+ a power-of-two row scale, exact within 14 binades, violations
counted); the compressor at ratio 2 (pairs, the odd tail, the decode
update in spans with per-row snapshots; a rollback replay is bitwise);
the window slot lists, the prefill scratch prologue / ring writeback,
the ring as a one-block cache the DSA attention kernels read directly;
the candidate select (a streaming block-max top-k over parts + a merge,
the newest block pinned, `expand_from_best` at kpool = block size
appends the partial newest block), the restricted / plain listed select,
the prefill logits and row selects — every selection equal to the
oracles' sets (the logits bitwise `dsa_ref::pool_logits`); the
two-source attention finish with the sink and the inverse rotation; a
window attention end to end (fp8_block ring, `dsa_attn_partial`, the
finish) within 8 bf16 ulps of the reference's `sparse_attn` arithmetic
on short and wrapped rings. One correction to §1.3.2: the row-scale
exactness bound is 14 binades, not 17 (e4m3's subnormal floor 2^-9 under
the 6·2^6 = 384 ceiling).

*Record (2026-09-13, the layer level).* `models/dsv41/csa2_state.{hpp,cpp}`:
`Csa2StatePool` on the shared `PagedBlockTable` (128-token blocks; a
cache at ratio r holds 128 / r entries per block, so every plane shares
one table and the prefix cache's protocol), the fp4_block main planes,
the planar index planes, the per-layer window rings (`ring_slots` rows
of fp8_block per request, presented to the DSA attention kernels as a
one-block cache through an identity table), the ratio-2 compressor
tails. `models/dsv41/csa2_layer.{hpp,cu}`: `Csa2Layer`, one object
rebound per layer over a shared scratch, `enqueue_prefill` (one request,
block-aligned chunks; the window from a scratch of the ring's last 127
rows + the chunk; the compressor pairs; the entries published before the
chunk's select; the gathered index keys through the fp8 dot GEMM into
per-row logits and the three prefill selects; attention in 128-row
tiles; the ring written back) and `enqueue_decode` (spans; the ring
append then the window list; the decode compressor update with tail
snapshots; the plain select on `dsa_select_decode` at kpool 1 over
`pos_sel`, the candidate stage over 8 parts + merge, the restricted
select; the listed flash attention at 16-head multiples, the split
kernel otherwise; the finish; the grouped wo_a and wo_b). The
selection and the candidate pool flow between layers through the
scratch under a call-shape check. `csa2_layer_test`: a six-layer
schedule (window; ratio-2 kv + index source and its reuse; ratio-1 kv +
index + candidate source, a Reindex layer, its reuse) on one pool
against a host oracle of the reference forward — two requests prefilled
in block chunks (the odd tail across a chunk boundary, the ring wrapping
at 48 slots for a 32 window), verify batches with spans, two rollbacks
(the tail snapshot restored, the positional rings and caches
overwritten), at TP 1 (16 local heads: the listed flash path) and as
rank 0 of TP 4 (4 heads, one output group): every layer output within
the bf16 budget, every selection set equal to the oracle's (zero flips),
zero index-key exactness violations. Left for G5: the torch cross-check
on the real weights (the download; the 5090 box), the Engram fold at
world 2 (model level), the dense-prefix attention shortcut below 512
visible entries (an optimization: the listed path is exact there).

**G5 — model and engine.** `tools/dsv41_reference_dump.py` (pure python
over the fixture, the engine's rounding points) and `dsv41_forward_test`:
per-layer streams (l2 ≤ 0.004), the final hidden, top-1, route ids and
weights, selections with margins, the Engram gates, the draft's five rows
and the Markov/confidence outputs; `dsv41_decode_test`: prefill == forward
bitwise in exact mode, bounded mode's decoder rows against the oracle's
truncated window, interleaved slots bitwise, chunked prefill across the
512-entry horizon and a ratio-2 pair boundary, snapshots hot == cold at
any position (rings, tails, context), 60 steps vs the re-forward, the
block speculator's transcript through the draft (greedy == plain,
`mtp_depth` 1..5), sampled acceptance against the oracle's `p_d`;
`dsv41_tp_test`: loopback world 4 rank-identical and within budget of
world 1 on fresh ports (29958/29959 taken by it, 29961–29963 by the engine
test; 29900–29957, 29960 and 29970 were taken — `glm_dsa_engine_test`
holds 29954–29957 — check `docs/testing.md` before choosing); the Engram
partial fold at world 4 == world 1; `dsv41_engine_test`: the graph
engine's scalar, batched and DSpark replays == the eager transcripts over
60 steps (`session_graph_host_nodes()` declares the two Engram host
nodes), world 2 == world 1. The gen app is the graph-feed contract's
second client (the 0 %-acceptance lesson). `dsv41_forward_check --layers
4` on the real checkpoint dumps the states G4's cross-check compares at 32
and ~2,100 tokens (the second crosses the candidate/top-k horizons).

*Record (2026-09-13, the model on the fixture).* `models/dsv41/model.{hpp,cpp}`:
`Dsv41Model` on the session core — the four residual streams, the
single-pass mHC per site (`launch_mhc_compute_normed` with `pre_in` /
`pre_out` / the fp32 exports, the collapse then the one-rounding norm,
`launch_mhc_stream_update_f32`; the head's weighted collapse), the CSA2
layer rebound per layer over the shared pool and scratch, the MoE
(`GlmMoeLayer` with the MXFP4 experts, the fp8 shared expert on the
32 x 32 grid, the `sqrtsoftplus` router), the Engram layers
(`models/dsv41/engram_layer.{hpp,cpp}`: the host node forked at the
walk's start gathers both tables' rows out of the mmap; the context
advances once per walk with per-row snapshots), the two folds per layer
and the Engram kv fold, the prefix snapshot (rings, tails, context) and
the rollback table (tails, context). `plan_memory` mirrors the
constructor. `dsv41_engram_layer_test` (the staging bitwise a host hash
and lookup at world 1 and every rank of world 4, the TP-4 partials
summing to the world-1 projection) and `dsv41_model_test` at world 1
(the 16-head, 32-index-head fixture): the cold forward, prefill == the
forward bitwise on the last row, the decode row against the prefill row
per layer (layers 0 and 1 bitwise, the compressed layers within 0.5 %:
the GEMV chain vs the tile kernels), decode steps and a verify with a
rollback against re-forwards. The decode-vs-prefill localizer found a
real bug the MoE gates could not see: the decode slot kernels hard-coded
the fp8 shared expert's scale grid at 128 x 128 (`resolve_slot_matrix`),
so a 32 x 32 shared expert read wrong scales at decode — every slot
launcher now takes the shared expert's grid (`sh_rs` / `sh_cs`, default
7), `glm_moe_test` 29/29 unchanged. Not yet at that record: the python
reference dump and `dsv41_forward_test`, the TP loopback test, the engine
graph test, DSpark (`mtp` is refused), the bounded prefill mode.

*Record (2026-09-13, later: the reference, TP and the engines on the
fixture).* `tools/dsv41_reference_dump.py` (numpy, the venv's python):
the engine's rounding points in double — the fp8/MXFP4 decodes, the
one-rounding RMSNorm, the single-pass mHC with fp32 exports, the fp4
block main KV and the fp8 block window rows, the e8m0/32 index keys and
queries, the folded indexer weights, the block-max candidate pool and
the restricted top-k, the attention's bf16 probabilities before PV, the
`sqrtsoftplus` router with the asymmetric swiglu clamp — dumps the
layer streams, the final hidden, the top-k logits, the route ids with
margins, every index source's selections with margins, its full logits
`[T, T]`, the candidate pools and the block-level margins. Two modes:
pure (the reference walks alone) and teacher-forced (`--teacher`: layer
l > 0 starts from the engine's streams after l - 1, the layer-local
comparison). `dsv41_forward_test` on the 40-token fixture: teacher-forced
strict — every layer's streams l2 ≤ 0.0046 (budget 0.006), no hard
element, top-1 all rows; end to end relaxed — the streams reach l2
0.013 by layer 7 (the random-weight network amplifies any
rounding-level difference some 1.5x per layer; budget 0.02), 15 of 120
selection rows flip, every one certified (the reference's boundary gap
within twice the row's engine-vs-reference logit deviation, or a block
swap at the candidate source within its block-level margin — a user
layer inherits the source's), 2 routing flips certified at margins
< 5e-4, top-1 hard mismatches 0. Two reference bugs found by the
comparison: the attention's probabilities must be rounded to bf16
before PV (the engine's kernels do), and a row with one visible entry
has no logit range (the deviation is now relative to the range or the
magnitude). `dsv41_tp_test` (ports 29958/29959, worlds 2 and 4): every
rank's streams, final read, routing and selections bitwise; against
world 1, the **layer-local twin** — the world-1 model walks the same
tokens with every layer started from the TP world's streams and the
collapse coefficients its next site read (`forward(tokens, capture,
teacher, teacher_pre)`, `debug_layer_pre()`), so each layer differs by
its own folds alone — l2 ≤ 0.0054 at every layer of both worlds, max 4
ulps, no hard element, every selection bitwise, the final read bitwise,
the merged top-1 equal on all 70 rows, 1 routing slot of 1,120 flipped
(a near tie at the MoE input after the attention fold). The end-to-end
comparison is reported on the clean prefix and gated loosely (layer 0,
every fold site once, holds l2 0.0034 / 0 hard — the slice-and-fold
gate); without the coefficients the twin's collapse read rounding-level
different inputs and the fp4-coded index queries flipped codes — the
teacher must carry both. `dsv41_engine_test` (port 29961, world 2): the
graph engine's scalar and row-batched replays (the 2-slot and full
families, a closed slot padding) equal the eager transcripts on both
ranks; the decode graph is 318 kernels + the one Engram host node, no
copy node — the walk's one copy (the one-hot collapse seed) became a
pointer rotation through `one_hot_` / `pre_a_` / `pre_b_`. Left at that record:
DSpark (D8), the bounded prefill mode, `dsv41_decode_test`'s snapshot
and chunked-prefill cases, the real-checkpoint load check and the
torch cross-check (the device is held by another deployment).

*Record (2026-09-14, the bounded prefill on the fixture — D8/§1.8).*
`Dsv41Model::set_prefill_bounded` (the server's `engine.prefill`,
`--prefill bounded|exact`, default bounded; the constructor's default is
exact, every parity gate's mode; `Dsv41Model::set_default_prefill_bounded`
is the server's switch). The walk: the encoder layers over the chunk's
rows; at the decoder's first layer (`decoder_first_layer()` = the last kv
source, 20 — the config check requires every decoder layer to read that
source at ratio 1) the attention site's input over every row goes to
`Csa2Layer::publish_prefill` (the compressor and the index keys into the
pool for all rows: the decoder's global KV); a chunk that is not the
span's last saves its last `window` rows of streams and collapse
coefficients (the tail, `tail_streams_` / `tail_pre_`, carried across the
call's chunks) and stops — no decoder, no head; the span's last chunk
assembles the segment (the tail's last rows before the chunk's last rows,
`window` at most) into the free streams buffer with `seg_pre_` as its
coefficients and walks the decoder over it with `enqueue_prefill(...,
floor, publish=false)` — the window list floored at the segment's start
(`csa2_window_slots_prefill` / `csa2_window_scratch_prologue` take the
floor) unless the segment holds every row of the span (then the floor is
0: the rows before it stand in the ring from the previous span or decode,
the resumed-prefill case), the selections over every published entry, no
alignment requirement on the segment's start; the head runs on the
segment at the chunk's last rows (a diagnostic forward's rows before it
read NaN) and the draft's prefill rows are the segment's (`mtp_run_rows`
clamps the session's chunk rows to `[seg_pos0_, seg_pos0_ + seg_rows_)`,
the tail rows included). A snapshot position closes a span (the session
core's `session_prefill_chunks`: `first_chunk` / `last_chunk` on the
`RowRun`; the saved state must be complete) and the next chunk opens one.
The reference (`tools/dsv41_reference_dump.py --prefill bounded`) walks
the same split (`csa2_forward(seg, pos0)`: publish from every row, attend
over the segment at its positions, the candidate pool at absolute
positions) and pads the decoder rows to T (NaN / -1). Gates:
`dsv41_forward_smoke` (the bounded forward's rows finite and
deterministic, NaN before the segment); the parity chain a second time
(`dsv41_forward_generate_bounded` → `dsv41_forward_test_bounded_e2e`
(relaxed, `--bounded`, the engine states padded for the teacher) →
`dsv41_forward_generate_teacher_bounded` → `dsv41_forward_test_bounded`
(strict)): the teacher-forced segment rows hold the strict budgets (l2 ≤
0.0046, 0 hard at every layer, every flip certified); end to end the
segment's kept rows sit at l2 0.011–0.016 with 0.4–0.8 % hard elements
against the 40-row budget's 0.5 % — the same six rows read 0.59 % in the
EXACT chain (the statistic is a threshold count over the row population,
and the segment is the prompt's deepest rows with no early rows to dilute
it; a segment row's floored window also leans more on the fp4 entries,
+10–25 % drift on the first segment rows), so the bounded e2e gate
measures that yardstick itself from the exact chain's states and dump
(`--exact-dump --exact-states`) and holds the segment within twice it
(never below the 40-row budget); `dsv41_decode_test` §8: a prompt within
the window and its decode bitwise the exact mode's (the same rows, floor
0); the chunked bounded walk (a 4-row tail) vs the one-shot — 36 flips
each certified by the chunks' own logit deviation, the last row l2 0.034,
and 24 teacher-forced decode rows after both: the 8 rows with equal
selections within l2 0.032 (the 16 others inherit the certified
difference); bounded prefix snapshots hot == cold bitwise at the cut and
through two steps; the eager speculator after a 300-token bounded
prefill reproduces the bounded greedy transcript with the forced draft
accepted. Bounded vs exact on the 140-token fixture prompt: last-row l2
0.139, top-1 equal (informational — the approximation's own distance on
random weights; the eval on the real model decides, §9.5). The tail
buffers cost `window × 4H` bf16 + two `window × 4` fp32 rows (in the
memory plan).

*Record (2026-09-14, the cross-check on the real weights — G4/G5's
last item).* `apps/dsv41_forward_check.cpp` walks the real checkpoint at
world 1 through the streaming loader (`--layers N` stops the diagnostic
forward after N layers, `Dsv41Model::set_debug_layer_limit`) and dumps
every walked layer's output streams, its collapse coefficients, the
routed expert ids and the index sources' selections ("DSV41ST3").
`tools/dsv41_torch_reference.py` runs the release's OWN layer code
(`inference/model.py` and `engram.py` from the snapshot, imported with
their tilelang `kernel` module replaced by exact torch math: every
Linear's weight dequantized once to bf16 — e4m3 × e8m0 blocks, e2m1
nibbles × e8m0 — which loses nothing; `act_quant` / `fp4_act_quant` /
`sparse_attn` / `hc_split_sinkhorn` ported from kernel.py's arithmetic;
the Engram table rows read from the 101 GB shard by row slice through
the release's own `NgramHashState` over the raw tokenizer; the experts
dequantized per layer on demand) on the head's CPU (a CPU torch wheel in
the tools venv: the Sparks have no CUDA torch), CHAINED and ISOLATED (a
block fed the engine's previous streams and coefficients). Layers 0–3
and 20 (the compressor / indexer / candidate source), a 35-token prompt
and a 2,100-token one (past the 512-entry top-k horizon at ratio 2 and
at ratio 1). Rows are scored alike only where the routed experts agree:
a row routed to a different expert is a different computation, and every
such flip sat on a near tie (the reference's own sixth-over-seventh
margin ≤ 3.4e-3 of the score range, most ≤ 3e-4). Measured, isolated,
relative l2 on the rows routed alike, the engine's two load-time
roundings mirrored (`--hc-bf16`: hc_fn to bf16; `--fp32-down`: the
expert down projections summed in fp32 — the release's bf16 GEMM output
rounds each expert's contribution once):

| layer | 35 tokens | 2,100 tokens | the fp32 oracle: engine / the release's bf16 pipeline (35 tokens) |
|---|---|---|---|
| 0 | 0.0037 | 0.0026 | 0.0111 / 0.0138 |
| 1 (Engram) | 0.0037 | 0.0024 | 0.0037 / 0.0029 |
| 2 (kv + index source, ratio 2) | 0.0017 | 0.0016 | 0.0033 / 0.0036 |
| 3 (its reuse) | 0.0011 | 0.0016 | 0.0025 / 0.0025 |
| 20 (ratio 1, the candidate source) | 0.0011 | 0.0011 | 0.0022 / 0.0023 |

The last column is the verdict's basis (the GLM cross-checks' `--dtype
float32` method): the same reference code run in fp32 — a near-exact
evaluation — sits as far from the engine as from its own bf16 pipeline
at every layer (ratio 0.8–1.3; both 1.1–1.4 % at layer 0, where the
sublayer outputs are the largest share of a still-small residual stream,
0.2–0.4 % after), so the engine is no further from the exact math than
the release's own kernels are. The engine-vs-bf16 numbers are two bf16
pipelines rounding at other points (0.1–0.4 %); the study that
localized them: the down-projection rounding is a tenth of layer 0's,
the hc_fn rounding ±0.04 %, the attention probabilities' bf16 rounding
nothing (`--attn-fp32-p`), the release's own fp8 activation rounding
(`--act-quant`) moves the reference AWAY from the engine (1–2 % per
layer: the engine keeps activations in bf16 by the quantization
decision). No growth down the chain (chained 0.37 → 0.51 → 0.51 →
0.51 % over layers 0–3). Selections: equal on every row while every
entry is visible; past the horizon (row 1,024 at ratio 2, row 512 at
ratio 1) 86 % of the rows differ by one or two entries of 512 — the
reference scores its index logits in bf16 and lets torch order the
ties, the engine scores in fp32 (§1.3.2) — and those rows sit at
0.32 % / 0.23 % (layers 2 / 20), no further than the rest. The script's
gate: the fp32 oracle rule (`--dtype float32 --baseline`: within 1.5× the
bf16 pipeline's distance per layer), the routed-alike budget 0.005
against the bf16 reference otherwise; both pass. The load check at the
production geometry (`dsv41_load_check --world 4 --rank 0 --mtp --engram
--sidecar`): 43 layers, 72.94 GiB resident, 137 s from the NVMe, the
two Engram tables mapped (189.13 GiB), the sidecar accepted.

*Record (2026-09-14, sampled acceptance over the whole block).*
`kSampleVerdictRows` 6 (`kernels/sample_pick.hpp`): the sampled verdict
decides the fed row plus five drafts — the DSpark block. The verdict
kernel's per-row tables (the rows' fold masses, merged logits, ids and
selector exps, the proposals' masses) and the one row's split keys moved
from static to dynamic shared memory (`kVerdictDynamicSmemBytes`, 50 KiB
at six rows against the 48 KiB static bound; the small per-row scalars
stay static); `device_sample_verdict_prepare()` sets the kernel's
max-dynamic-shared attribute once, called by the launcher and by
`DevicePicker`'s constructor so a launch recorded inside a graph capture
finds it set. `kSampleProposalSlots` follows (5 draft proposals per
request; the graph engine's proposal tables size by it). Gate:
`glm_pick_test`'s `sample_pick_full_block_matches_spec_oracle_over_simulated_world`
— six rows per request over a simulated world of 4, every request's
verdict bitwise the host's speculative chain (the greedy judge, the
accept test at every draft row in turn, the plain sample at the last row,
every reject row, the fallbacks and the accept-all path exercised over
40 trials; the normalizers, the row-0 logprob, the count table and the
digest chain equal); the T = 2 / T = 3 / proposal oracles unchanged.

*Record (2026-09-14, the DSML tool grammar).* `text/tool_grammar` gains
the DeepSeek-V4.1 format (`GrammarVocab::usable()` now true for
`ToolFormat::kDsml`): the ｜DSML｜ tag token — a special token, empty
text — rides the text automaton as a sentinel byte inside the targets
(`GrammarState::kDsmlSentinel`; `match_ids` offers the marker id alone
where a target reaches it and lets no token's text run across it). The
states: the top is free text where the tag is offered right after a "<"
(the format's "\n\n<｜DSML｜ calls>"), then `d-calls` (" calls>\n"),
`d-invoke` ("<" TAG " invoke name=\"" NAME "\">\n" over the tools' names,
the named one under tool_choice named), `d-param-or-close` ("<" TAG
" parameter name=\"" KEY "\" string=\"" for an unused closed key, or the
free-key prefix, or "</" TAG " invoke>\n" when closable), `d-free-key`,
`d-flag` ("true\">" for a text/enum value, "false\">" for a JSON-typed
one, either for a free value), `d-value` (the typed value — a JSON
machine or the enum texts — then "</" TAG " parameter>\n"; a free value
is text until its "</" makes the tag the closer's start),
`d-invoke-or-close` (another invoke while calls remain, or "</" TAG
" calls>" then the turn ends). The service qualifies the grammar's tool
names as the schema lists them ("namespace::name",
`Dsv41Prompt::qualified_tool_name`) and resolves a named choice to that
spelling; the parser strips the namespace again. Gates:
`tool_grammar_test` (`unit_tests`): the required walk position by
position (no bare tag in text, the tag after "<", no token across the
tag, closed keys offered once, the flag forced by the value's kind, a
free value's "</" + tag closer), typed values / named single / open keys
/ auto with a dead grammar on a disallowed id. With it tool_choice
required and named and parallel_tool_calls false are served on this
family.

*Record (2026-09-13, DSpark on the fixture).* The block draft rides the
session core's chain protocol unchanged (D8 restated): `kSpecRows` 6,
`kSpecMaxDrafts` 5, `DevicePicker::kSlots` 6, `mtp_depth` 1..5 (the
constants are still global — every family's spec tables widen, the
regression suite is the gate). The verify walk stores each row's
target hidden `[h_t1 | h_t2 | h_t3]` (the stream mean of the attention
input at the target layers, `kernels/dsv41_dspark.cu`) in the core's
draft window (`draft_width` 3H). The step's FIRST draft call
(`mtp_run_rows` over the accepted rows: eager after the verify, or the
in-graph rows off the verdict) gathers them, projects `main_x`
(`main_proj`, `main_norm`), appends every draft stage's ring with the
rows' window latents (`Csa2Layer::append_window_rows`), builds the
block rows `[next, noise x 4]` at the next five positions off the
staged rows (`dsv41_dspark_block_rows`: the block's first position is
1 + the max accepted position, `next` that row's winner — the same
kernel serves the eager, the scalar-graph and the batched forms), runs
the three stages once over the block (`Csa2Layer::enqueue_draft_block`:
the rows' own latents into the ring at the block's slots — past every
later window, the 160-slot ring's reserve — and the attention over the
ring's real rows plus the whole block, `csa2_dspark_window_slots`; the
draft MoE is a second `GlmMoeLayer` on the draft config), the shared
head with the draft's norm into `base_logits_`, and emits block row 0
biased by the Markov head of `next` (`dsv41_dspark_markov_bias`, the
rank's vocab slice, broadcast into the verify-shaped head rows the
pick's `row_select` reads); every CHAIN call emits the next block row
biased by the previous pick and its confidence logit
(`dsv41_dspark_confidence`, stored per slot) — no state moves, so the
chain brackets and the draft snapshot are no-ops and
`draft_state_bytes` is 0 (the rings are positional). The first/chain
distinction: a chain call heads one row per request after a block
stands (the capture sequence is host-ordered; the eager chain names its
row past the session position). A streamed stack holds one layer, so
each stage's tensors are used right after its load. Gates:
`dsv41_model_test` (the eager draft rows and chain, a six-row verify —
every row within 0.65 % of a re-forward — the rollback, the next
draft); `tools/dsv41_reference_dump.py` grew `forward_spec` (the rings
from every prompt row's `main_x`, the block through the stages, the
greedy Markov chain, the confidence logits, per-stage site tensors);
`dsv41_forward_test`: teacher-forced the five rows within l2 0.005, all
picks equal, confidence within 0.0015; end to end within 0.027 (budget
0.05: the main path's drift feeds the rings), all picks equal — the
per-stage localizer (`DGPP_DSV41_CAPTURE_DRAFT`) found the one bug on
the way, the block list's stride (row 4 read row 5's entries);
`dsv41_engine_test` (ports 29962/29963): the DSpark graph engine at the
full block (depth 5) and at depth 2, scalar A and the batched B/C —
transcripts bitwise the plain eager engine's on both ranks (a
random-weight draft is rejected, as it should be). Not yet: sampled
acceptance with `p_d` (the sampled verdict kernel's shared memory sizes
`kSampleVerdictRows` rows of 256 candidates — 6 rows exceed the 48 KiB
static bound, so `kSampleVerdictRows` stays 4 and sampled decoding with
more than three drafts is refused at record time until the kernel is
restructured), the confidence-scheduled row allocation (the
optimization gate), the bounded prefill mode.

*Record (2026-09-13, the session surface; the regression verdict).*
`dsv41_decode_test` on the fixture: the session prefill's last row
bitwise the forward's; a 40-step greedy decode from a 40-token prompt
against the cold re-forward, and the chunked prefill (128-token chunks,
a 140-token prompt: the cut is a ratio-2 pair boundary and a block
boundary) against the one-shot — both with the selection flips
CERTIFIED rather than assumed (the user's ask of 2026-09-13: understand
the failures before adjusting a gate). The mechanism, localized with
the layer-state and site captures: a prefill's GEMMs accumulate in a
row-count dependent order, so the 140-row and the 128-row walks differ
by one bf16 ulp in a few elements (4 of 128 rows at layer 0, 0.4 % of a
row), and the decode row's GEMV chain differs from the tile kernels the
same way; the fp8 window rows and fp4 entries the other path appended
re-quantize such a difference into code flips (single elements moving
by a quarter or more — the first flipped row's streams: bitwise at
layer 0, 0.4 % at layer 1, 5–7 % from the first compressed layer on),
and the fp4-coded index queries turn that into percent-level index
logit moves; on this random-weight fixture the top-16 boundary gaps
are that close for most rows, and a flipped row's attention (O(1)
different) perturbs its cached entries and, through the 16-row window,
every later row — flips cascade. The gates: rows before the first flip
hold the tight budget (0.7 %); the first flip is certified (its
reference boundary gap within 5 % of the range — 2.1 % / 4.0 % seen —
and the row's streams within 1 % before the first kv source and 10 %
after); a chunked prefill's 76 flips are each certified against the
chunks' own index logits (the gap within twice the measured deviation,
the forward test's rule; 0 uncertified); after a certified flip the
flipped rows are exempt and the kept rows hold the long-audit budget
(2.3 % seen), no hard top-1 mismatch. The exact gates around this
(prefill == forward, the interleaved slots, the snapshots, the graph
engine, the TP twin) are the decode path's bitwise evidence; the real
checkpoint (trained indexers sit on near ties far less often) decides
the rest. Two interleaved slots bitwise their solo runs;
a closed slot reopens bitwise with the pool empty; the prefix snapshots
hot == cold bitwise at the aligned cut 256 of a 300-token prompt (the
suffix resumed, two steps after the attach) and after a mid-decode
snapshot at position 128 (the rings, the compressor tails and the Engram
context travel with the entry); the eager greedy speculator at the full
block reproduces the plain greedy transcript (the forced correct first
draft accepted). The full build-ci suite after the spec-row constants
(`kSpecRows` 6): 92 / 92 with every family's gate unchanged.

**G6 — tokenizer, prompt format, tools.** `dsv41_tokenizer_test` (the
golden corpus + the new cases, byte-exact ids and round trips),
`dsv41_chat_template_test` (the five reference cases and the generated set,
byte-exact renders and ids, `reasoning_effort` forms, thinking on/off,
`drop_thinking` with and without tools, mid-conversation system, tool
results merged in call order, namespaces), the DSML parser and grammar
unit tests (typed parameters, streaming partials, malformed input refused
by name), `serve_tools_check.sh` on the fabric.

**G7 — serving, benchmarks, records.** `dgpp-serve` boots the family from
the two templates at world 4; the memory plan's number is read from a boot
refusal at an oversized shape; `fabric_dsv41_serve.sh` (greedy
transcripts, client timing, `serve_api_check.py`, eval with thinking off
and on) and the four op-stream md5s match at `dgpp-cluster down`; the
numbers follow `docs/benchmarks.md` §9 exactly as rows for "DeepSeek-V4.1-
Flash MXFP4/FP8, world 4": §9.1 `serve_bench.py` (T = 1 and DSpark ms/step,
ms/token, tokens per pass, the stats lines including the Engram fault
counts), §9.2 the five-class corpus with `mtp_depth_check.py` + `diff -r`
byte-identical between plain and DSpark, §9.3 `serve_load.py --concurrency
1,2,4` and `--isolation 4`, §9.4 `serve_prefill_probe.py 512 2048 8192
32768 --repeat 3` in both prefill modes, §9.5 `serve_eval.py --tasks
gsm8k,extract` on the fabric and HumanEval on the isolated machine
(bounded vs exact prefill scored both ways; the card's 93.0 / 79.4 are the
base model's, the served instruct numbers are recorded as their own
baseline), §9.6 the three determinism checks, the teacher-forced NLL of
the three shipped texts (absolute, rerun delta exactly 0). Operations:
`serve_soak_run.sh 60`, `serve_failure_drill.sh` on one rank,
`serve_stop_check.sh`, `serve_prefix_curve_sweep.sh` at one point, a
1 M-token context boot (the KV pool at `kv_capacity` 1,048,576, a 256 K
prompt through bounded prefill). Regression: `fabric_glm_regression.sh`
bitwise, Qwen / GLM-4.7 / full GLM-5.3 transcripts identical to their
records, `ctest` green. Records: `docs/benchmarks.md` §3–§7 rows, a
"DeepSeek-V4.1-Flash on four nodes" section in `docs/measurements.md`, a
dated local run artifacts entry, the CHANGELOG line, PLAN.md's serving
row and link, `deploy/README.md`'s table, the model card, §7 of this
document filled in.

**Optimization gate, after G7 measures.** The recorded T = 1 step is
compared with the 16.7 ms floor + ~4 ms of collectives; the port is
accepted at or under GLM-4.7's ratio to its floor (1.25×) and the nsys
breakdown names the remainder. Then, in order: the confidence-scheduled
row allocation under concurrency (D8), the MXFP4 tile kernel for prefill
if G2 shipped the GEMV chain first, `hc_fn`/router narrowing (D10) with
the NLL A/B, `main_proj` sharding, the Engram page-cache hit rate and the
FP4 re-pack if it matters, SWA bounded replay for prefix snapshots. Levers
one at a time, A/B at `--kv-capacity 16384`, never an older binary at the
production shape.

*Record (2026-09-14, the optimization gate's first kernel win — the
decode window attention).* An nsys decode profile (GPU busy 97 % of the
step, so the step is compute/bandwidth-bound, not idle on collectives or
the host) put the fp8/fp4 weight GEMVs at 70–88 % of peak cold — near the
hardware limit, no lever there. It put the biggest non-GEMV cost in the
WINDOW attention: `attn_partial<fp8Block>` at ~140 us a layer against
~12 us for the compressed source's `attn_flash`, despite the window's 128
dense keys against the compressed 512. The cause: the window attention
launched with `n_split = 1`, so at decode (one row, one head group at TP4)
it was a SINGLE thread block — four warps, latency-bound, the other ~47
SMs idle — while the compressed path splits `decode_n_split` ways and fills
the GPU. Fix (`csa2_layer.cu`): the decode caller splits the window a fixed
`kWinDecodeSplit = 8` ways across the key dimension, its partials combined
by the existing `attn_finish`; the split count is FIXED (not row-derived)
so a token's window is computed identically whether it is a one-row step
or row 0 of a multi-row MTP verify (the speculative transcript must equal
the plain greedy one), and the window workspace grew to hold the splits.
Prefill's 128-row tile already fills the GPU, so it stays at one split and
is byte-identical — the reference-parity gates are untouched. Measured on
the fabric (world 4): chat 38.3 → 34.3, code 25.3 → 21.8 ms/token
(~10–14 % on the attention-heavy classes), gsm8k 60/60 unchanged,
cross-rank op streams still identical. The cost, accepted deliberately
(2026-09-14): softmax attention cannot be split bitwise-invariantly the
way the GEMVs are, because the online-softmax combine's per-split rescale
rounds differently. So decode's window is no longer bit-identical to the
forward's single block — it moves by ~1 bf16 ULP, which on the
random-weight fixture tips near-tie selections and changes the greedy
transcript (accuracy held: gsm8k 60/60). `dsv41_model_test`'s decode-vs-
forward checks were re-calibrated from bitwise / a 0.02 logit budget to a
tight rounding budget plus an argmax-match assertion, documented in place;
`dsv41_decode_test`'s certified-flip audit (the behavioural gate) and the
eval are unchanged and pass. `kWinDecodeSplit` is a tunable; a larger
split fills more SMs but widens the rounding, and 8 stays inside the
decode audit's certified-flip budget. Still open on this gate: the
confidence-scheduled rows, the per-layer collective fusion (~5 ms, 14 %),
and — the only bitwise-preserving path to more window/attention speed —
fusing the window and compressed sources into one split flash used by both
prefill and decode.

*Record (2026-09-14, the streaming tensor-core decode GEMM — the batched
step's biggest cost).* The six-slot profile (30 verify rows, 252 ms/step,
GPU busy 98.7 %) put 56 ms in the dense fp8 GEMV chunks (4 rows per
launch, the weights re-read per chunk: 1,800 launches/step), 17 ms in the
bf16 head's chunks and 9 ms in the draft's `main_proj` at one row per
launch (K = 15,360 exceeds the chunk's activation smem budget) — 82 ms of
weight re-reads, against the MoE's ~107 ms which is at its traffic floor.
`kernels/mma_gemv.{hpp,cu}`: one warp per 8 weight rows, a quad of lanes
per row, each lane streaming 16-byte chunks (8 in flight) into
`mma.sync.m16n8k16 bf16→fp32`; activations staged per 512-k window in
double-buffered shared memory (33 KB, one barrier per window); the weights
read ONCE for every row of the launch (m ≤ 32). Numerics: the dequantized
values are bf16(e4m3 × 2^e), EXACT for the e8m0 scales, so the terms are
the GEMV chain's own and only the fp32 accumulation order differs
(tolerance-equal, not bitwise with the chunks — the accepted "different
order, same accuracy" rule); a weight row's chain depends on k only, never
on m or on which rows share the launch, so a batched row is bitwise the
row alone (the isolation property the batch families rely on). The
design point that made it stream: the mma's B fragment hands lane t the k
slots {2t, 2t+1, 2t+8, 2t+9} and a dot product is invariant to which
physical k sits in which slot as long as A and B agree, so a lane's OWN
contiguous chunk fills its slots of four slices with the same four
activations per row as the A words — no cross-lane shuffles (the
owner-lane form, 32 shuffles per chunk, was issue-bound at ~4 GB/s per
SM). Measured cold (weights cycled through 6 copies), fp8 [5120 × 5120]:
m = 1/6/16/30 → 132/133/136/142 µs against the chunks' 118/205/419/764
(−12 % at one row, +35 % at six, 5.4× at thirty); the bf16 head
[32,320 × 5120] at 6 rows 1,477 vs 2,841 µs. Dead ends measured on the
way: 64-byte lane runs halved the stream (112 GB/s); activations read from
global memory per chunk matched the staged form at one row and lost at
six. Wired for this family only — `launch_scale_gemm_grid_*(...,
decode_mma)`, `CublasLtGemm::set_decode_mma`, `Csa2Config::dense_mma`,
the engram layer's setter, the model's `dense_mma_` (default on;
`DGPP_DSV41_DENSE_GEMV=1` forces the chunks, the A/B switch) — every dense
decode site alike, since a family must switch all or none. Gates: the
kernel's oracle/m-invariance/timing test (`mma_gemv_test`); every DeepSeek
gate under both forms; `dsv41_tp_test` gained a 24-row world — the 70-row
one prefills through the tile kernels and never touched the decode form
(nsys: 493 `mma_gemv_kernel` launches in the new one; layer 0 within 2
ulps at worlds 2 and 4); `dsv41_engine_test`'s world-2-vs-world-1 rule
("the first 3 tokens agree") turned out to be luck on the random fixture
(the folds reassociate in bf16 on the wire, world 2's logits sit ~1e-2
from world 1's, and the flips had world-1 margins 7.5e-3..5e-2) — it now
requires the first three decisions to agree or to flip on a world-1 top-2
margin under 0.1, the margins recorded by the world-1 pick closure on the
decode's own logits (a prefill's logits are not the decode's here: a DSA
selection flip moved a late margin by 0.38). Fabric numbers: docs/
measurements.md (the like-for-like vLLM bench before/after).
*The prefill's rows, later the same day.* The prefill-only profile (the
bench's own prompts, 25–300 tokens, and a 2,500-token filler) put 31 % of
the kernel time in the dense projections: the 4-row GEMV chunks for
prompts of 33–128 rows (10–32 launches per site, the weights re-read per
chunk) and the 16-row tile kernel above (4.6 TFLOP/s at 2,048 rows, the
weights re-read per 16 rows); the MoE's grouped kernels 33 %, the eager
folds' collective kernel 11 %, host gaps under 5 % (the prefill is GPU-
bound, not host-bound). The tensor-core GEMM gained 64- and 128-row forms
(4 and 8 tiles at 128- and 64-k windows, the same chain) and 128-row
groups above, and every opted-in site takes it at every row count: 192 /
288 / 567 / 4,558 µs at 64 / 128 / 256 / 2,048 rows on [5120 × 5120] fp8
against 1,507 / 2,978 (the chunks) and 3,533 / 23,360 (the tile). The
fixture gates re-rolled their near-tie flips under the new prefill order:
the decode audit's first flip sat on a 7.6 % reference gap, the model
test's decode-vs-prefill row flipped at layer 6, the relaxed end-to-end
parity's kept rows read 0.021–0.023 with 0.7–1.7 % hard elements. Rather
than widen the gap rule, the gates now read the evidence the mechanism
predicts: both walks capture the row's coded index query (e4m3 codes and
row scales, `IndexLogits::q_codes`), and a flip whose codes differ between
the paths is certified as the coding's discontinuity (the decode audit's
flip: 81 codes differ at the flipped source; the model test's layer-6 row:
81 of 4,128) — a flip with identical codes must still be a near tie. The
end-to-end relaxed budgets moved to l2 0.03 / 2 % hard with the cascade
named (a flipped row's cached entries move later rows' codes by whole
steps; even the first row, with no flip anywhere, drifts to 0.047 by layer
7 through its own cache codes), while the strict layer-local run — the
kernels' accuracy gate — is unchanged at 0 hard and passes. On the fabric
the wide forms took the like-for-like bench to C1 46.3 / C2 62.3 / C6
77.9 tok/s aggregate (TTFT 0.38 / 0.59 / 1.53 s) and cold prefill to 708 /
711 tok/s (docs/measurements.md).
*The prefill's experts, later still.* The prefill profile's other third was
the MoE's grouped GEMV core (`moe_grouped_gemv_fp4_kernel`: rows staged
four at a time, every expert's weights re-read per 4 rows — 12 re-reads at
2,048 tokens), taken because the fp4 tensor-core tile kernel (the NVFP4
ldmatrix kernel of docs/nvfp4_plan.md) knew the NVFP4 triple only and the
layer's `mma_takes_grid()` refused MXFP4 tables. The kernel gained a
`kGroup` template (16: e4m3 per 16 + the global; 32: e8m0 per 32, no
global): two scale bytes per row per 64-k stage instead of four (plain
loads: cp.async has no 2-byte form), the pair decoded through fp32 (the
e2m1 value from the hardware e2m1x2 → f16x2 conversion times the scale as
an fp32 power of two — exact, since e8m0 can leave f16's range where the
NVFP4 path's f16 product could not), no epilogue division. Every opted-in
launcher takes `fp4_group`; the layer routes MXFP4 tables to the tile
kernel. Gate: `glm_moe_test`'s MX oracle test (gate bf16 within 3.9e-3 =
its rounding, down fp32 within 4e-5, the z split bitwise, ragged segments
and a ragged last k-stage at I = 224). `glm_moe_test`'s two MXFP4 contract tests now expect the tile
kernel (the decode slot path stays bitwise the GEMV host path; the prefill
path is bitwise the tile host path), every DeepSeek gate unchanged,
100/100. On the fabric: cold prefill 708 → 1,286 / 1,315 tok/s (the
recipe's 902 at 2K and 1,539 at 46K), C1 48.6 / C2 66.2 / C6 82.7, TTFT
0.31 / 0.48 / 1.26 s.
*The batch's depth (D8a, the same evening).* The batched replay took the
deepest live slot's depth, so at six slots one confident slot held every
slot at the full block (the six-slot histogram: 260 of 442 batches at
depth 4). The batch's own Dinkelbach rule (`scheduled_verify_depth_batch`):
extending the batch by one position costs one row per slot and yields the
sum of the slots' survivals, so the batch verifies position i while the
mean survival over its slots beats λ·row — the scalar rule with the mean
in place of one slot's S_i (each mean non-increasing: the kept positions
are a prefix). Exact at any depth; fresh or sampled slots still force the
full block; the test hook forces per slot with the batch at the deepest.
The engine gates on 29964/29965 pass unchanged (they force through the
hook). Measured next at λ 0.045 and at the six-stream world's own achieved
rate (λ ≈ 0.08, the Dinkelbach fixed point at that concurrency).
Measured: λ 0.045 → C6 84.5, λ 0.08 → 86.9 (C1 47.0); then the adaptive
λ — an EWMA of each MTP step's committed tokens over its modeled time
base + rows·row, floored at the configured λ, replicated inputs and one
fixed arithmetic so every rank holds the same value (`verify_lambda_update`,
`engine.mtp_schedule_adapt`, default on) — C1 48.2 / C2 66.8 / C6 87.0
with one configuration, λ ending at 0.098 after the six-stream phase.
*The group prefill (the same evening).* The six-stream gap's other half was
the scheduler's strict alternation: one admission per tick, a decode step
between, so six arrivals were six read-ins (TTFT 1.2–2.1 s at C6, 11–29 %
of the wall not decoding). Now queued cold prompts that each fit one
window (no prefix-cache attach or snapshot — a prompt under the 128-token
snapshot alignment never had one) admit together (`Scheduler::
admissible_group`, `admit_group`; the engine's `prefill_group` and
`prefill_group_span_limit`/`prefill_group_total_limit`) and prefill as the
spans of ONE walk (`session_prefill_group` → `run_rows` with span tables:
the dense sites, the MoE, the norms, the Engram (already span-aware) and
the head over every row; the CSA2 attention, the layer-20 kv publication
and the draft segment per span — `Csa2Layer::enqueue_prefill(...,
row_base)` keeps each span's selection state and shape record at its own
rows). Bitwise: a prompt prefilled in a group equals the prompt alone at
the model (world 1), per layer at world 2 on both fold paths, and through
the engine's three-slot batch — after two forms were pinned to row-
invariant chains: the mHC dots' tiled form at every prefill row count
(`mhc_set_tile_min_tokens(1)`; the vector form under 16 rows gave a
different fp32 coefficient and, at world 2, one bf16 element) and every
bf16 row count of the opted-in cuBLASLt instance through the tensor-core
form (an Lt algorithm's split changes with m). The MoE's tile kernel and
the fp8 mma were row-invariant already. Other families keep one prefill
per tick (their span limit is 0). Fabric: docs/measurements.md. The
first fabric run (C6 87.0 → 103.6, TTFT 0.90 s; a six-prompt burst one
~540 ms forward) still admitted the 292-token prompts one by one — the
span limit was one window; spans of any width now: under the bounded
prefill the decoder walks each span's last window rows packed span after
span (its coefficients in the free rotating buffer, its window floored at
the segment's start), the head's per-span last rows copied back to the
rows the session core reads, the draft's rows and segment per span from
the packed space (`dsv41_model_test`, 23 + 9 + 40 rows at window 16,
bitwise the bounded prefills alone).
Fabric: C1 47.9 / C2 71.6 / C6 106.4, TTFT 0.31 / 0.45 / 0.87 s (the
recipe 37.95 / 64.30 / 131.86 and 0.44 / — / 0.50).

Kernel record, 2026-09-14 (late): the decode forms' weight loads are
asynchronous — cp.async into a per-warp ring of two to eight window
slices (24 KB per block: two stages at eight warps so two blocks share an
SM, eight at one warp), no register staging — and the block width follows
n (eight warps from 4096 rows, four from 1024, two under; one warp per
block is issue-bound). Cold: the bf16 head [32320 × 5120] at one row 1441
µs against the GEMV's 1438 (was 1470 / 1437); fp8 [5120 × 5120] 122 / 119
at one row, 123 / 212 at six, 145 / 780 at thirty; the narrow DSA shapes
of the full GLM-5.3 at sixteen rows 2–4× ahead of the chunks
(docs/measurements.md). Tried and not taken: 512-k bf16 windows at one
block per SM (1453), 128-k windows at three blocks (1470, and fp8 133), a
48 KB ring (no better at any width, one block per SM), the register
double-buffer (registers). The one-row bf16 gap is closed; the narrow-n
one-row gap (the 576-row site: 29 against the chunks' 11 µs) is the eight
rows per warp — a split-k form would need a workspace and a reduce; the
session-core families route one to four rows to the chunks instead, this
family keeps every row on the streaming form for its bitwise group
prefill.

**D9 — The GPU-driven eager fold (BUILT 2026-09-14, late; the design
below stands).** `CollectiveBus::allreduce_stream(stream, src, dst, elems)`
and `allreduce_settle()`: the graph kernel form launched on the model's
stream, its generation from the shared counter at submit (one forward
thread, one stream: execution order equals generation order), its cell
from a 64-deep stream ring, the engine posting from a FIFO through the
replay walk's own per-generation flight (`gen_flight_pass`, shared with
the windows), one settle per pass. The gates keep the eras apart: a
host-driven collective, a handout or a recording session is rejected
while stream generations are outstanding; a stream issue is rejected while
a window is armed; a failed generation fails the era and the drain-fail
poisons the FIFO so the stream drains. `BusStreamReducer`
(engine/tp_bus.hpp) is the reducer: `stage()` hands out nothing (the fold
runs in place on the model's buffer), `reduce()` issues, boundaries above
one latency slot settle, drain and take the bulk path as before; the
model binds its stream at construction (`BoundaryReducer::bind_stream`),
skips the drain before a stream-ordered fold and settles at the end of
every eager pass. Serving: DeepSeek-V4.1-Flash by default,
`DGPP_DSV41_EAGER_FOLD=1` the host-driven reducer (the A/B switch); the
other families keep the host-driven one. Gates: `bus_test`'s
`scenario_allreduce_stream` (100 generations per pass, three passes, at
worlds 2 and 4: every destination bitwise the oracle, the gates, a
host-driven one-shot between passes) and `dsv41_tp_test`'s group-prefill
gate run through both reducers (every layer's rows and the logits
bitwise between them, at 12 and 51 rows — the 51-row group's wide folds
through the bulk path). Fabric (the six-slot config, the recipe's bench,
same evening): C1 TTFT 0.303 → 0.269 s, C6 TTFT 0.836 → 0.727 s, the
aggregates 49.3 / 108.0 → 49.6 / 108.5 tok/s (parity), prefill 2K 1,370 →
1,383 tok/s; the fold kernel at 47 rows 542 → 386 µs (the copy 158 → 24),
at 128 rows 1,397 → 933, and the host's 0.3–1.3 ms notice per fold off
the model's path. Tables: docs/measurements.md.

**D9 (the original design note) — The GPU-driven eager fold.** The prefill's
boundary reductions (93 per pass) run the host-driven eager collective:
the model stream drains, the host submits, the engine thread launches the
one-block consumer on the bus's own stream, which copies the payload into
each peer's pinned row (three copies), waits for the peers' doorbells,
folds their rows out of system memory, and the host notices the finish
0.2–0.9 ms later before the model may continue (`DGPP_BUS_TIMELINE=1`
now prints every 32nd prefill-class fold's decomposition; the 2026-09-14
table in the results file: a 60-row fold ~1.5 ms of wall). The captured
decode graphs' collective node has none of the host terms and one shared
staging row. The design (the bus analysis of 2026-09-14): a new
`allreduce_stream(stream, dsrc, ddst, elems)` that takes a generation from
the shared `ctl_seq_counter` (the same order on every rank: execution
order equals generation order), resets a cell from a reserved ring and
stores its `gen_seq` with release, pushes `(gen, cell, elems)` to an
engine FIFO, and launches `bus_allreduce_graph_kernel` on the caller's
stream; `graph_pass`'s flight machine posts from the FIFO as it does from
a window; a device-buffer reducer beside `BusBoundaryReducer` (a small
ring of stable buffers); `Dsv41Model::fold` drops its stream sync; one
`allreduce_settle` per pass. Risks to design for: the lost per-fold
synchronous verdict (a wedged peer becomes a stream that never finishes:
the deadline, `graph_fail` and the poison paths must cover the FIFO),
generation divergence when a rank aborts a pass, the stage ring's depth
under host run-ahead, lane-0-only posting at these payloads, the eager
gate and world-1 branches. Also worth a one-line experiment first: the
30–60-row folds sit just under the bulk (reduce-scatter + allgather)
threshold and pay the one-shot's (W−1)× wire bytes.

**D10 — Prefill rows inside the decode batch (designed, not built;
2026-09-14 late).** The remaining admission cost after D9 is the prefill
walk's serial execution: a group of six ~30-token prompts is one ~400 ms
walk during which no decode step runs, and the six-stream TTFT is that
walk plus the queueing behind the step in progress. The recipe hides it
by chunking prefill into the running batch. What that needs here, and
why it is not a short project:

- *The captured segment.* The decode step is a replayed graph, so prefill
  rows must ride a captured variant with a FIXED segment: family × depth
  × (segment present or not) doubles the variant count (60 → 120 at six
  slots; `kBusMaxGraphVariants` 64 → 128, +6 MB of pinned cells), and the
  segment's rows must be device-fed (tokens, positions, request id, the
  span's length) with the rows past the chunk's length masked inside
  every kernel of the prefill path — the CSA2 prefill attention, its
  selection, the kv publication, the bounded prefill's decoder-segment
  packing all take host row counts today. The alternative, treating the
  chunk as VERIFY rows of the prefilling slot (the decode attention and
  publication already handle several consecutive positions of one
  request), needs a verdict mode that accepts every row (the prompt is
  its own "draft"), a heterogeneous batch shape (one slot with P rows,
  the others with 1 + depth), and is bounded by the slot kernels' row cap
  (48: six slots at depth 4 leave 18 rows — a 30-token prompt takes two
  steps, a 500-token one 28, worse than the group walk).
- *The eager mixed step* (decode rows plus the group's spans in one
  eager walk) has no eager batched engine to ride: the batched step is a
  graph end to end (verify, verdict, the DSpark draft chain), and the
  eager path is scalar.
- *The value.* At six streams under the load probe an admission lands
  every ~2.7 s; a 1–2-prompt group costs ~150 ms of stalled decode against
  a marginal ~60 ms as rows of a step — ~3 % of the six-stream aggregate
  and ~0.2 s of the six-stream TTFT. The six-stream gap to the recipe
  (0.81×) is the step's own composition (the experts 110 ms and the
  collectives 33 ms of 215).

The order that makes sense: the collectives at wide payloads and the
per-slot verify depth first (the step), then the captured segment as the
verify-row form with the row cap raised, behind a flag.

The project's standing rules apply (restated because each has cost a day):
rebuild everything before any `ctest` verdict; one fabric ritual at a time,
long ones in the background with a Monitor; never `pkill -f` a pattern the
command itself contains; fixture scales release-like; the python reference
mirrors the kernels' chains where a tolerance would hide a gap; a passing
suite with skipped checkpoint cases validates nothing; a graph/feed
contract change makes the gen app a second client.

## 7. Status

2026-09-13: the plan. The four nodes hold 821 GB free each with the
current checkpoints; nothing is downloaded, built or measured.

2026-09-13, later: G1 landed (config + binding, 96,085 tensors bound on
a header-only mirror of the release's 48 shards before the download);
G2 landed (MXFP4 fp4 core, MoE slot/grouped launchers and the
`sqrtsoftplus` router, the single-pass mHC kernels, the Engram hash /
gather / gate kernels, the fp8 GEMV's 32 x 32 grid; the full `build-ci`
suite 79/79 with every prior gate bitwise); G3 landed on the fixture
(the record under §6 G3); G4 landed at the kernel and layer levels (the
two records under §6 G4: the block latent formats, `kernels/csa2`, the
pool and the layer, `csa2_test` 8/8 and `csa2_layer_test` at TP 1 / 4).
G0 closed on the real snapshot (48 shards verified, the sidecar written
beside it, the audit reproducing §2, the bind-check 96,085 / 96,085); the
G4 regression verdict is green (82 / 82 after the DSA kernel changes);
G5 is green on the fixture (the records under §6 G5): the pure-python
reference parity (strict teacher-forced and relaxed end to end), the TP
loopback worlds 2 and 4 (the layer-local twin), the graph engine at
world 2, DSpark (greedy: the block draft through the eager and graph
engines, the reference parity of its five rows) and the session surface
(`dsv41_decode_test`: decode, interleaving, chunked prefill, prefix
snapshots, the eager speculator); the suite stands at 92 / 92. Deferred
inside G5 at that point: the bounded prefill mode, sampled acceptance
past three drafts (the verdict kernel's shared memory), the
confidence-scheduled rows (the optimization gate).
2026-09-14: the bounded prefill mode landed (the record under §6 G5:
the split walk, the tail across chunks, the segment with the floored
window, the reference's bounded mode, the second parity chain with its
measured yardstick, `dsv41_decode_test` §8, `engine.prefill` /
`--prefill` defaulting to bounded in the server; the unfiltered suite
99 / 99), then sampled acceptance over the whole block
(`kSampleVerdictRows` 6, the verdict kernel on dynamic shared memory, the
six-row oracle in `glm_pick_test`) and the DSML tool grammar (the two
records under §6 G5 / G6). Also: the two deployment templates
(`deploy/cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_{mtp5,plain}.example.json`,
the memory plans 76.55 / 74.27 GiB), and a `dgpp-serve` fix — `--model`
with `--mtp` and no depth read the checkpoint before the model id was
resolved (the snapshot is resolved before the family's default depth
now; a peer resolves again after the handshake). Left: the
confidence-scheduled row allocation (the optimization gate, after the
fabric measures), and every real-checkpoint check (the device is still
held). G6 is green on the fixture and the snapshot's own encoders (the
three-stage tokenizer 118/118, the prompt renderer 20/20, the DSML
parser; the DSML tool grammar deferred). G7 has the family registered
in `dgpp-serve` (the app builds; the memory plan on the real config is
the next check); the fabric boot, the benchmarks and every
real-checkpoint check wait for the device. The real-checkpoint `dsv41_load_check`
and the torch cross-check wait for the device (another deployment holds
it).

*Record (2026-09-13, G6 — the tokenizer).* The checkpoint's
`tokenizer.json` is a third shape for `text/tokenizer`: an empty
normalizer Sequence, THREE Split stages before ByteLevel — `\p{N}{1,3}`
(number runs cut in threes, every script), the three literal CJK ranges
(U+4E00–9FA5, hiragana, katakana; Korean and halfwidth kana are letters),
and a main pattern over the `\p{P}` / `\p{S}` classes (one ASCII
punctuation character + ASCII letters, `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`,
` ?[\p{P}\p{S}]+[\r\n]*`, the three whitespace alternatives) — and BPE
without `ignore_merges`. The file carries CR/LF as the control
characters themselves and the CJK ranges as raw codepoints (the pinned
literals are generated from the decoded JSON). HF's Sequence runs each
Split on the previous one's pieces, so a piece's end is the lookahead's
end, and text no alternative matches (format and control characters
outside a letter run's prefix) stays a piece of its own — the scanner
(`Dsv41Scanner`) mirrors all three stages over codepoints; the Unicode
tables gained the P* and S* classes (`tools/gen_unicode_tables.py`,
Unicode 15.0.0 as before). `tests/data/dsv41_tokenizer_goldens.jsonl`:
118 cases from HF tokenizers 0.23.2 on the snapshot (the reference's
decision surface probed first: `x1234y` → x/123/4/y, `I don't` → I/Ġdon/'t,
`半角ｶﾅ ①②③`, format characters as gaps), `dsv41_tokenizer_test` byte-exact
118/118; the four other families' gates unchanged.

*Record (2026-09-13, G6 — the prompt renderer and the DSML parser).*
`text/dsv41_prompt.{hpp,cpp}` is `encoding.py`'s `encode_messages`
(text only) over the service's template globals: the tool messages
merged into the preceding user turn as `<tool_result>` blocks (a
`tool` after a user turn joins it; consecutive users join, as the
reference does), the results reordered by the previous assistant's
call ids, the tools block on the first system message (an empty one
inserted when the conversation has none), the reasoning of earlier
turns dropped in thinking mode unless the conversation carries tools
(`clear_thinking`), `enable_thinking` false as the chat mode, the
numeric effort budget at the conversation's start (low 50 / high 75 as
the reference maps them; minimal 25 and medium 62 interpolate; absent:
high), the DSML block of an assistant's tool calls (a string argument
raw with `string="true"`, any other value as JSON; a JSON string
parsed, twice when double-encoded; a non-object under "arguments"),
namespaced tools (`ns::name`, the namespace description prefixed), and
the generation header after the last user or mid-conversation system
message (`<think>` in thinking mode, `</think>` in chat mode).
`tools/gen_dsv41_prompt_goldens.py` imports the snapshot's own
`encoding.py` and renders 20 cases (the mapping from the globals to the
reference's call written there once); `dsv41_prompt_test`: 20/20
byte-exact with the tokenizer ids matching. The one deviation the
goldens caught: a tool result's list content reduces to its text parts
before the merge (the reference's image path runs first), not to
"[Unsupported X]" markers. Image content is refused (D9).
`text/tool_parser`: `ToolFormat::kDsml` — the `｜DSML｜` tag token is the
only marker (a SPECIAL token the service's decode skips; the parser
restores it between the decoded runs), the block opens at the tag after
a "<" (the content run holds back a possible "\n\n<" prefix until the
next id decides — at most three characters of latency), buffers to the
text "</｜DSML｜ calls>", then parses the reference's grammar exactly
(">\n" after the calls tag, ` name="NAME">\n` per invoke, ` name="K"
string="true|false">V<` per parameter): every invoke a tool-call event
(a namespaced name reports its bare function name), a malformed block
its literal text, text after a closed block content; 4 unit cases
(175/175 in the unit binary). Not built: the DSML tool GRAMMAR for
constrained decoding — `GrammarVocab::usable()` is false for the
format, so the service serves `tool_choice` auto with parallel calls
and refuses `required`, a named function and `parallel_tool_calls:
false` with its usual "no constrained decoding" error (the grammar is
a text-match machine like Qwen's; the optimization gate or a later
item).

*Record (2026-09-13, G7 — the serve registration, on the fixture only).*
`apps/dgpp_serve.cpp`: `Dsv41Family` ("deepseek_v41"; 128-token blocks;
the 2^21 entry-id bound; `decode_rows_cap` 16; the widest fold the
Engram kv partial `[rows, 5H]`; `default_mtp_depth` = the block, 5 —
`--mtp` without `--mtp-depth` resolves to it on every rank before the
settings record leaves rank 0, `engine.mtp_depth` in the file counts as
explicit), `make_family` on `DeepseekV41`, `embed_sharding = vocab`
honoured (`Dsv41LayerStream::set_embed_vocab_sharded`), and the
frontend by family: `serve/frontend.hpp`'s `Dsv41Frontend` (the DSML
renderer, the tokenizer's markers: <think>, </think>, ｜DSML｜, the
<｜User｜> / <｜Assistant｜> / <｜System｜> turn markers as the prefix
cache's boundaries) where the others load `chat_template.jinja`; the
prefix key carries the renderer's revision. `dgpp-serve --memory-plan`
on the real checkpoint's config: the fabric shape (world 4, rank 0,
`--mtp --decode-graph`, 131,072-token context, 2 slots) resolves the
DSpark depth to 5 and 12 decode rows (2 x 6), and plans 72.94 GiB of
resident weights per rank (§2's estimate: ≈ 73), a 0.13 GiB CSA2 pool
(≈ 1 KiB per context token: the fp4 main rows, the index keys, the
rings, the tails), 0.25 GiB CSA2 scratch, 0.67 GiB activations,
0.51 + 0.30 GiB MoE scratch (the backbone and the draft stages), 0.11 GiB
Engram staging, the 1.5 GiB prefix arena — 76.54 GiB + 4 GiB headroom,
inside a 121.6 GiB node once the device is free (refused today: another
deployment holds it); the world-1 streaming shape plans 13.76 GiB and
fits now.

*Record (2026-09-14, G7 — served on four nodes, the real checkpoint).*
The GLM-5.3-Flash soak that held the device came down; the DeepSeek world
booted from `deploy/cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_mtp5.json`
(the DSpark block draft, decode graph, 2 slots, 128K context, bounded
prefill). Two lessons at the first boot: (1) rank 0's first collective
timed out after 60 s while the peers' cold loads ran 52–71 s behind it
(175–246 s per rank from the NVMe) — the captured resident images make
the second boot 22 s and even; (2) the preflight refused the checkpoint
for a missing `chat_template.jinja` — DeepSeek ships `encoding/encoding.py`
instead, so `cluster_doctor.checkpoint_size` now accepts either. `--model`
with `--mtp` and no depth resolved the family's default (5) before the
snapshot was resolved; fixed (`apps/dgpp_serve.cpp`, `resolve_model`
before the depth default). Served numbers (`build-ci/fabric-runs/
dsv41_serve_2026-09-13/`, the 2026-09-14-dsv41-flash measurements):
single stream 76 ms/pass at 2.33 tok/pass (32.5 ms/token), the class
sweep 20.7–38.2 ms/token wall with DSpark acceptance p1 55–92 % (json
92, math 82, code 72, prose 67, chat 55); bounded prefill 2.35 / 1.85 /
1.61 ms/token at 512 / 2,048 / 8,192; gsm8k 60/60, HumanEval 40/40,
extract 30/30 (thinking off); the DSML tool grammar served every shape
(auto / required / named / none / parallel / single / a follow-up with
the tool result / json_schema); the plain T=1 world's greedy transcripts
byte-identical to the DSpark world's (MTP == plain greedy), `serve_load
--isolation` identical, the four ranks' op streams identical at every
take-down. The prefix cache caches a prompt past one block (a 331-token
prompt: prefill 1,040 → 375 ms, 256 tokens cached); a prompt shorter than
a block is not cached (§8's deferred item). The suite stands at 95 / 95.
Left: the optimization gate (§5; the confidence-scheduled rows first, now
that acceptance is measured).

**2026-09-14 (later): the optimization gate's confidence-scheduled verify
depth is built and measured (D8a).** Policy `engine/verify_schedule.hpp`
(6 unit tests), the eager exactness gate (`dsv41_decode_test` §7b), the
graph-engine port (reduced-row variants per slot, the confidence
publication, the per-step choice; `dsv41_engine_test` on 29964: forced
depths 1,4,2,5,3,… reproduce the plain eager transcript, both ranks
identical), the server knobs journaled to every rank, full suite 99/99.
Fabric (three worlds from one binary; the 2026-09-14-dsv41-flash measurements): baseline reproduces the window-split
transcripts (the default path unchanged); scheduled at λ 0.045 — chat 33.4
→ 24.6, prose 27.4 → 22.7, code 21.9 → 18.9, json 21.9 → 19.1, math 21.8 →
19.8 ms/token, transcripts identical, op streams identical across the
ranks. The deploy example enables it. Left in this line of work: the
batched families at one depth per batch (the max over its slots), an
adaptive λ (a deterministic EWMA over the committed counts and a modeled
time — never wall-clock, which differs per rank), and recovering the
launch-ahead (publishing the confidence incrementally per chain row so a
shortened step can launch before the block's last pick).

*Later the same day (the batched replay and every MTP family; docs/
measurements.md):* the batch at scheduled depth is exact and wins at two
live requests (prose 34.1 → 51.1, chat 36.6 → 54.0 tok/s aggregate); its
first fabric run caught a stale settle bound after a reduced batch (fixed,
gated). The draft-probability confidence gives every other MTP family the
schedule (GLM-5.3-Flash's own model class needed the feed seam too — the
first boot hung on the guard I had left there); measured at depth 2 on
GLM-4.7 and GLM-5.3-Flash it is exact and throughput-neutral, since one
row is at stake. The DeepSeek deploy example enables the schedule; the
GLM examples do not.

## 8. Risks and open questions

- The API check's `usage cached_tokens` test fails on this family by
  construction (every recorded run since 2026-09-13): the prefix cache
  snapshots at the CSA2 block alignment (128 tokens), so a repeated
  42-token prompt finds the retired entry at position 128 and no cut at or
  below 42 — the log says so ("the nearest entry shares the first 42 of the
  prompt's 42 tokens"). A hit needs a shared prefix of at least one block.
  The other families (align 1, the DSA pool's alignment) pass the test.

**Sub-block prefix-cache cuts (deferred, 2026-09-14).** The prefix cache
snapshots at a whole 128-token block, so a prompt shorter than a block is
never cached (the `usage cached_tokens` API check fails on a 30-token
prompt; a longer prompt caches). Lowering `snapshot_align` to 2 (the
compressor pair) to cache short prompts crashed the four-node plain world
under load: a sub-block snapshot's partial-block copy
(`SessionModel::pin_blocks_at` → `acquire_pinned_block`) lands on the
`Csa2StatePool`, a second paged pool separate from the KV pool the
scheduler's `new_blocks` reservation sizes, so under real concurrency the
acquire takes a block the pool still needs and a later kernel writes out
of range (the fault surfaced at `reset_request`'s ring memset). The fix
is to give the CSA2 pool its own snapshot headroom in the reservation
(the way the KV pool reserves one block when `block_tokens > align`); the
fixture's snapshots pass at the block alignment. Until then
`snapshot_align` stays at the block.

- **The Engram token map** must reproduce exactly 99,092 classes from the
  `tokenizers` library's normalizer chain over the engine's own decodes;
  a single divergent decode rehashes a row's bucket. G0 pins it, and the
  loader refuses a sidecar whose count or pad class disagrees.
- **NVMe-bound prefill.** 12 rows per prompt token per rank; at 110 K IOPS
  a 2,048-token chunk's rows take ~0.25 s, hidden by the chunk-ahead
  gather except for the first chunk. If the page cache runs cold under
  long prompts the lever is the FP4 re-pack (halves the cache footprint
  per row, not the IOPS) or a device-resident hot-row cache — measured
  first.
- **The index-key exactness** (fp4 values stored as e4m3 + a row scale) is
  exact only while a row's block exponents span ≤ 17 binades; the append
  counts violations. If real rows violate it, the index cache takes the
  e8m0/32 layout natively (a select-kernel variant, more work).
- **Selection ties.** The reference scores in bf16 and lets torch order
  ties; the engine scores in fp32 with a pinned rule. Certified as
  near-ties as for the GLM families; the candidate pool's block-max top-k
  adds a second tie site (block scores), same treatment.
- **The window rings in prefix snapshots** cost 3.7 MiB each; a 1.5 GiB
  arena holds ~400. SWA bounded replay (recompute the rings from the last
  128 tokens on a hit) removes them; it is the model's production
  behaviour and a later lever.
- **`kSpecRows` 6** touches the spec tables, the pick kernels and the
  batch families of every family (a build-time bound today). The plan
  makes it per-family; the other families' shapes must stay bitwise
  (their tests are the gate).
- **Bounded prefill is an approximation** the report calls negligible; the
  eval scores both modes, and `exact` stays one knob away.
- **Prefill select at long context.** Layer 20's select scans every
  visible entry per query; at 256 K tokens a 2,048-token chunk scores
  2,048 × 256 K dots per index layer — bounded by the existing dot
  budget with smaller query tiles, a measured item; 1 M-token prompts are
  a stretch goal, not a gate.
- **DSpark sampled acceptance** needs the draft's distribution at the
  sampled token under temperature 1.0 / top_p 0.95; the top-p truncation
  on both sides follows the Qwen sampled-draft rule and is measured
  against the oracle in G5.
- **The reference kernels need tilelang** (0.1.8, Hopper/Blackwell
  datacenter); the torch cross-check replaces them with dequantized bf16
  linears, which is the same substitution the GLM cross-checks made.
- **Disk.** 510 GB per node leaves 310 GB; the next family will need a
  checkpoint retired. The NVFP4 re-packs are not downloaded.

*Record (2026-09-16, communication width, scheduled depths and prefill chunks).*
The four-node inference study
profiles the current six-slot service and tests alternatives without changing
weight or KV precision. The retained graph consumer uses 256/512/1,024 threads
according to payload size, and bounded prefill uses 4,096-token chunks.
The broader workload exposed regressions in the four-depth scheduling
experiment, so the existing three captured depths remain. The study contains
the final matched service rates, correctness gates,
memory cost and rejected receive-cache and scheduling-policy experiments.
