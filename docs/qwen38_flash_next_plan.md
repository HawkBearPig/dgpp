# Qwen3.8-Flash-Next-FP8 on dgpp — architecture study and implementation plan (2026-09-09)

> NVFP4 deployments: the [single-Spark guide](qwen38_single_spark.md) covers
> world 1, and the [two-Spark benchmark](../benchmarks/results/2026-09-16-qwen-nvfp4-w2.md)
> covers world 2. Both map the n-gram table from NVMe and use the graph engine.

Status: the text path is implemented, including tensor-parallel loading,
GDN/QSA/GR/PLE operators, decode sessions, prefix caching, graph serving
and MTP. FP8 serving is measured at TP=2 and TP=4. NVFP4 serving is measured
at TP=1 and TP=2.

This document combines the architecture study with the dated port and
optimization record. Its cost model describes the FP8 checkpoint with
BF16 dense projections unless stated otherwise. Proposed work and
measurements in the record retain their original dates.

The tensor census comes from the checkpoint's safetensors headers.
Architecture references include its configuration, transformers 5.8
(`modular_qwen4_exp.py`), the SGLang port and the technical report.

References: <https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8>,
<https://arxiv.org/abs/2608.30320>,
<https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/>,
<https://github.com/QwenLM/Qwen3.8-Flash-Next>.

## 0. Summary

Qwen3.8-Flash-Next is a 48-layer hybrid: 36 Gated DeltaNet (GDN) layers and
12 Qwen Sparse Attention (QSA) layers in a 3:1 pattern, every layer followed
by a 512-expert top-10 MoE with one gated shared expert, the residual widened
to four branches read through an elementwise gate (Gated Residual, GR), a
51 B-parameter hashed n-gram embedding table injected at the second layer
(PLE), and a one-layer MTP head. 125 B parameters in the backbone, 6 B active,
2560 hidden, 248 320 vocabulary, 262 144 native context. The FP8 release
quantizes ONLY the routed experts (block 128×128, the same format as the GLM
FP8 release) and the n-gram table (per-tensor scale); everything else is BF16.

Architecturally it is close to what the engine already serves: GDN is a
simpler cousin of KDA (scalar per-head decay, q/k heads shared across three v
heads), QSA is a DSA-style indexer-plus-top-k over 4-token key blocks feeding
a GQA attention with 256-wide heads, GR is a 4-stream hyper-connection like
mHC with different read/write operators and no stream-mixing matrix, and the
MoE, MTP, embeddings and head follow the GLM shapes. the additional components
are the n-gram embedding layer and the QSA block compression.

The deployment worlds are TP=2 and TP=4 (TP=1 is 172 GiB and does not fit
a Spark). The TP geometry derives from the world size W for every class
(§2.1). The decode cost model (§2) says the model streams **6.20 GB per
token per rank at TP=2 (25.8 ms floor at 240 GB/s)** and **3.83 GB at TP=4
(16.0 ms)**, against 5.86 GB and 24.5 ms for GLM-5.3-Flash at TP=4. Two
facts dominate: BF16 dense classes are 85 % of that traffic, with the GR
gate matrices alone 1.27 GB per token on every rank when replicated — the
one placement that does not derive from W and is therefore a config
variable (D3); and the 640-wide expert intermediate does not split into
128-aligned slices at either world, so the FP8 expert slice runs on a finer
scale grid (64 rows at TP=2, 32 at TP=4; D2).

The implementation uses the shared engine and loader interfaces, with
family-specific operators and state. Sections 1–4 describe the model and
placement decisions; section 5 records the port and subsequent experiments.

## 1. The model

### 1.1 Configuration facts

| field | value |
|---|---|
| `architectures` / `model_type` | `Qwen4ExpForConditionalGeneration` / `qwen4_exp` (text: `qwen4_exp_text`) |
| layers | 48; `layer_types` = `LLLF` × 12 (`full_attention` at 3, 7, …, 47 = QSA; the rest `linear_attention` = GDN) |
| hidden / vocab / eps | 2560 / 248 320 / 1e-6; `hidden_act` silu; `tie_word_embeddings` false |
| residual | `hc_count` 4, `hc_lowrank` 320 → the hyper state is 10 240 wide |
| GDN | 16 key heads × 128, 48 value heads × 128, conv width 4, output gate sigmoid, state dtype fp32 |
| QSA attention | 24 q heads, 2 kv heads, head_dim 256, no bias, per-head q/k RMSNorm, attention output gate, `partial_rotary_factor` 0.25 (64 rotary dims), θ = 1e7, `mrope_interleaved` section [11, 11, 10] |
| QSA indexer | 4 q heads × 128, 1 k head × 128, `indexer_budget` 2048 tokens, `indexer_compress_ratio` 4 → top-512 blocks + ≤ 3 tail tokens (≤ 2051 keys) |
| MoE | 512 experts, top-10, softmax + renormalize (`norm_topk_prob` true), intermediate 640, shared expert 640 with a sigmoid gate |
| PLE | `ple_layer_ids` [2] (one-indexed → layer index 1), `ngram_size` 3, `heads_per_ngram` 8 (16 hash heads × 160 = 2560), `ngram_vocab_size_base` 20 000 000, `make_ngram_vocab_size_divisible_by` 128, `split_ngram_parts` 128 in this checkpoint, conv width 4 with dilation 3 |
| MTP | 1 layer, `hybrid` true, `layer_types` [`full_attention`], shares `embed_tokens` and `lm_head` |
| tokens | bos = eos = 248044 `<\|endoftext\|>`; generation eos [248046 `<\|im_end\|>`, 248044]; `<think>` 248068, `</think>` 248069, `<tool_call>` 248058, `</tool_call>` 248059 |
| generation defaults | temperature 1.0, top_k 20, top_p 0.95 |
| quantization | `fp8`, `activation_scheme` dynamic, `weight_block_size` [128, 128]; `modules_to_not_convert` = everything except `mlp.experts.*`; `modules_to_convert` = the n-gram table |
| vision | 27-block ViT, 1.1 GiB, not served (same stance as GLM) |

### 1.2 One layer

The hyper state is `R ∈ bf16[4, 2560]` per token. Layer 0 starts from the
token embedding copied into all four branches. Every layer runs:

```
if layer == 1:  R += PLE(R, ids)                       # §1.7
x, Rn = GR_mix(R)                                      # x ∈ [2560], Rn = per-branch norm of R
y      = GDN(x) | QSA(x)                               # after the TP all-reduce
R      = R + s(Rn) ⊗ y                                 # GR_combine: one scalar per branch
x, Rn = GR_mix(R)
y      = MoE(x)
R      = R + s(Rn) ⊗ y
```

After layer 47, `h = GR_mix(R)` from the model-level `hyper_connection_mixer`
(mix only, no combine) feeds `lm_head` directly — there is no separate final
norm; the mixer's per-branch RMSNorm plays that role. The MTP head reads the
4-branch `R` after layer 47, not `h`.

Every RMSNorm in the backbone except the GDN output norm is the zero-centered
form `x · rsqrt(mean(x²) + eps) · (1 + w)` computed in fp32 and cast to bf16.
Group norms (`group_size = 2560`) normalize each branch separately with one
weight vector of 10 240.

### 1.3 Gated DeltaNet (GDN)

Per token, per layer (shapes are the full-model shapes; §2.1 gives the rank
slices):

```
qkv = W_qkv x                        # [10240] = q[16×128] | k[16×128] | v[48×128], head-major
qkv = silu(conv1d_causal_w4(qkv))    # depthwise, per-channel weights [10240, 4], 3-step state per request
z   = W_z x                          # [48×128]
β   = σ(W_b x)                       # [48]
g   = −exp(A_log) · softplus(W_a x + dt_bias)     # [48], fp32
q,k ← l2norm(q,k) (eps 1e-6 inside the sqrt);  q ← q · 128^-1/2
v-head j reads k-head ⌊j/3⌋ and q-head ⌊j/3⌋            # repeat_interleave(3)
S_j ∈ fp32[128 k, 128 v]:  S_j ← exp(g_j)·S_j;  u = v_j − S_jᵀ k;  S_j += k ⊗ (β_j u);  o_j = S_jᵀ q
o   = RMSNorm_128(o_j) · w_norm · σ(z_j)                  # w_norm plain (init 1), fp32 norm, bf16 cast
y   = W_out o                                             # [6144] → [2560]
```

Differences from KDA as the engine implements it (`src/kernels/kda.hpp`):
the decay is one scalar per head instead of a per-key-dimension vector with
the lower-bound sigmoid form; q/k have 16 heads shared by 48 v heads instead
of one head set; the conv runs over 10 240 channels laid out `q|k|v` with
unequal segment widths. Same: l2norm with eps inside the sqrt, `K^-1/2`
scaling of q, `σ(beta)`, fp32 state `[K, V]` per head, the delta
update, the post-update read, the sigmoid-gated RMSNorm output, causal conv
width 4 with silu. State per request per rank: 12 heads × 128 × 128 × 4 B =
786 KB per layer, 28 MB for 36 layers; conv state 2560 channels × 3 × 2 B.

### 1.4 Qwen Sparse Attention (QSA)

```
[q | gate] = W_q x            # [24 × 512]: per head, dims [0,256) are q, [256,512) the gate
k = W_k x,  v = W_v x         # [2 × 256] each
q ← RMSNorm_256(q)·(1+w_q);  k ← RMSNorm_256(k)·(1+w_k)
RoPE on dims [0, 64) of q and k (neox halves, 32 frequencies, θ = 1e7, position t)
S_t = index(x, t)             # the selected key positions, ≤ 2051 of them (below)
o = softmax(q · K[S_t]ᵀ / 16) · V[S_t]     (GQA: 12 q heads per kv head)
y = W_o (o · σ(gate))         # [6144] → [2560]
```

Indexer (`self_attn.indexer`, replicated on every rank):

```
[qI | kI] = W_idx x           # [4 × 128 | 128]
qI ← RMSNorm_128(qI)·(1+w_qI); RoPE on dims [0,64) at position t
kI stored raw per token; block b = tokens 4b..4b+3 is complete once 4b+3 ≤ t
c_b = RoPE( RMSNorm_128( mean_fp32(kI[4b..4b+3]) )·(1+w_kI), position 4b )     # the compressed key, bf16
s_b = Σ_h relu(⟨qI_h, c_b⟩) / √128    over complete blocks, fp32
B_t = top-512 blocks by s_b
S_t = ⋃_{b∈B_t} {4b..4b+3}  ∪  {4⌊(t+1)/4⌋ .. t}      # the tail: 0–3 tokens of the incomplete block
```

For a prompt of at most 2051 tokens the selection is every token, so short
contexts are dense causal attention. The MTP layer is also QSA with its own
caches. Text position ids are identical on the three mRoPE axes, so the
interleaved layout collapses to plain RoPE over the first 64 dims; the loader
asserts one-dimensional positions.

Caches per rank per attention layer: K and V bf16 `[tokens, 256]` for the
rank's kv head; compressed keys bf16 `[tokens/4, 128]`; a pending ring of
the last ≤ 3 raw indexer keys per request. Torch's `topk` does not define its
tie order; the engine pins "higher score first, ties to the lower block
index" as it does for DSA, and the host oracle uses the same rule.

### 1.5 Gated Residual (GR)

Per site (`attn_hyper_connection`, `mlp_hyper_connection`, and the final
`hyper_connection_mixer` without the combine):

```
Rn_i = RMSNorm_2560(R_i) · (1 + γ_i)                     i = 0..3, one γ vector of 10240 (hc_norm)
t    = silu( (W_down · vec(Rn)) / 4 )                    W_down [320, 10240]
G    = σ( W_up · t )                                     W_up   [10240, 320]  → G ∈ [4, 2560]
x    = (1/4) Σ_i G_i ⊙ Rn_i                              the block input
s    = 2 · σ( (W_inj · vec(Rn)) / 4 )                    W_inj [4, 10240]
R_i ← R_i + s_i · y                                      the block output y written to every branch
```

`Rn` is reused by the combine, so the mix keeps it (20 KB per token). There
is no `H_res`: branches never mix. Per site the two gate matrices are
2 × 6.55 MB of BF16 — 13.1 MB per site, 97 sites, **1.27 GB per token**.

### 1.6 MoE

```
p = softmax_fp32(W_gate x)                  # [512]
E = top-10 of p; w_e = p_e / Σ_E p          # renormalized, cast to bf16
y = Σ_e w_e · W_down_e (silu(W_gate_e x) ⊙ W_up_e x)  +  σ(w_s · x) · S(x)     # S = the shared expert
```

No bias, no group-limited routing, no scaling factor, no swiglu clamps.
Experts are FP8 e4m3 with `weight_scale_inv` BF16 `[5, 20]` for gate/up
(`[640, 2560]`) and `[20, 5]` for down (`[2560, 640]`) — block 128×128.

### 1.7 N-gram embedding (PLE, layer index 1)

The n-gram ids depend only on token ids. With `EOS = 248044`
(`<|endoftext|>`; `<|im_end|>` does NOT reset the context):

```
y_0 = x_t
y_s = x_{t−s}  if t−s ≥ 0 and no EOS occurs at positions t−s .. t−1, else EOS   (s = 1, 2)
m   = [23703573157769, 20109073645365, 8052911324071]         # from splitmix64(seed 1234), §1.7 note
mix2 = (y_0·m_0) XOR (y_1·m_1);   mix3 = mix2 XOR (y_2·m_2)   # int64; products < 2^63, no overflow
id_h = (mix2 mod P_h) + off_h   for h < 8,   (mix3 mod P_h) + off_h   for 8 ≤ h < 16
```

`P_h` are the sixteen primes after 19 999 999 — 20000003, 20000023, 20000033,
20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093,
20000107, 20000147, 20000153, 20000159, 20000161, 20000171 — with `off_h`
their prefix sums (0, 20000003, 40000026, …, 300001275); total 320 001 446
rows, padded to 320 001 536. The checkpoint stores the table as 128 shards
`ngram_embedding.shard_S.weight` F8_E4M3 `[2 500 012, 160]` (row range
S·2 500 012 …) with one BF16 `weight_scale`; the multipliers, prime sizes and
offsets are also stored (`layer_multipliers`, `ngram_heads_vocab_sizes`,
`ngram_heads_offsets`) and must be checked against the derivation at load.

The layer itself:

```
e   = concat_h bf16(table[id_h]) · weight_scale                 # [16 × 160] = [2560]
k   = GroupNorm_2560(W_key e) · (1+w_key)                        # W_key [10240, 2560] → [4, 2560]
v   = W_val e                                                    # W_val [2560, 2560]
qn  = GroupNorm_2560(R) · (1+w_query)                            # [4, 2560]
g_i = ⟨k_i, qn_i⟩ / √2560;  g_i ← sign(g_i)·√max(|g_i|, 1e-6);  u_i = σ(g_i) · v
un  = GroupNorm_2560(u) · (1+w_conv)
c   = silu( conv1d_depthwise(un; width 4, dilation 3) )         # reads steps t, t−3, t−6, t−9; 9-step state
R  ← R + (u + c)                                                 # [10240], before the layer's attention mix
```

The conv weight is `[10240, 1, 4]`, initialized to zero at training start
(the trained values are what the checkpoint carries). The hash multipliers
come from `_build_layer_multipliers(vocab 248320, ngram 3, ple_layer_index 0,
seed 1234)`; the values above were recomputed from that code.

### 1.8 MTP

One QSA layer with its own GR sites and MoE, plus:

```
R_i^mtp = W_fh · ( RMSNorm_10240(R_47)·(1+w_h) )_i  +  W_fe · ( RMSNorm_2560(embed(x_{t+1}))·(1+w_e) )
```

`pre_fc_norm_hidden` is a full-vector RMSNorm over all 10 240 elements (per
SGLang's `GemmaRMSNorm(hc_count·hidden)`), not the grouped norm — a detail to
pin with the first acceptance-rate measurement. The MTP model ends in its own
`hyper_connection_mixer` and shares `lm_head`. The tech report's "IndexShare"
(reusing one indexer selection across draft steps) matters only at depth ≥ 2.

### 1.9 Embedding, head, RoPE

`embed_tokens` `[248320, 2560]` BF16 and `lm_head` `[248320, 2560]` BF16,
untied. `R_0 = embed(x)` on every branch. RoPE: `inv_freq_i = 1e7^(−2i/64)`,
i < 32, applied to the first 64 of 256 dims (attention) and the first 64 of
128 dims (indexer) in neox halves.

### 1.9.1 YaRN at 512 K (`engine.rope_scaling`, 2026-09-18)

The engine's opt-in YaRN mode, verified against the vLLM stack the user's
recipe runs (`vllm/vllm-openai:qwen38-flash-next`, start.sh with
`YARN_ENABLE=true YARN_FACTOR=2.0 MAX_MODEL_LEN=524288`). Off by default:
with no `engine.rope_scaling` the table is `qsa_rope_inv_freq`'s, every
kernel argument is 1.0f and the ceiling is the checkpoint's 262144 — the
bits the 2026-09-17 build produced (pinned by
`tests/unit/qwen_rope_scaling_test.cpp` and `tests/cuda/qsa_test.cu`).

```json
"engine": {"rope_scaling": {"factor": 2.0,
                          "original_max_position_embeddings": 262144}}
```

| field | default | meaning |
| --- | --- | --- |
| `factor` | required, >= 1 | YaRN's scale; `original x factor` is the new ceiling (524 288) |
| `original_max_position_embeddings` | required | the native context the ramp is measured from |
| `beta_fast` / `beta_slow` | 32 / 1 | the rotation band's edges (vLLM's names) |
| `attn_factor` | 1 | vLLM's `attn_factor`; the effective scale is `yarn_get_mscale(factor) * attn_factor` |
| `mrope_cache_factor` | 4 | vLLM's MRotaryEmbedding enlarges its cache 4x and YaRN's correction band is computed from **that** value: 4 reproduces the recipe, 1 uses the native context |

Three findings the implementation turns on, each read off the image's own
source and checked bit for bit in a container:

1. **The recipe merges, it does not replace.** `ModelConfig._update_nested`
   recurses into `text_config.rope_parameters`, so the checkpoint's
   `mrope_section: [11, 11, 10]` survives `--hf-overrides` and `get_rope`
   builds an `MRotaryEmbedding`, whose `cache_max_position_num =
   max_position_embeddings * 4` feeds `yarn_find_correction_range`. The
   band is therefore low/high = **16/24**, not 14/22; the two tables differ
   from lane 15 on. `mrope_cache_factor` is exactly this knob.
2. **The attention factor rides the rotation, not the softmax scale.**
   vLLM keeps `self.scaling = head_dim**-0.5` (nvidia/qsa.py) and bakes
   `mscale` into the bf16 cos/sin cache (`freqs.cos() * mscale`, then the
   query dtype), which scales only the 64 rotated lanes of q·k (by
   mscale²) and leaves the other 192 untouched. The kernels here do the
   same: `round_bf16(cosf(ang) * mscale)`, in `norm_rope_thread` and the
   two indexer rotations. `mscale` = 1.0693147182 (f32 `0x3f88df4e`).
3. **Text positions make the MRoPE layout the identity.** The three
   position rows are equal for text, so `apply_interleaved_rope`'s
   per-lane section pick selects the text row everywhere and the YaRN
   table's lane i is the lane the neox pairs use — no per-lane remapping
   (`mrope_interleaved_is_the_identity_for_text_positions`).

What the mode does not touch: the QSA softmax scale, the pool's size (a
pool is sized by `engine.kv_capacity`, so the knob lifts what a request may
reach without moving a byte — `qwen_plan_check` asserts that), and every
other family (the launcher warns that the knob applies to `qwen4_exp` only).
The knob rides the settings record (`"rs"`) so a peer can never rope at
different frequencies from rank 0, and the canonical config digest.

### 1.10 Tokenizer, template, tool format

`tokenizer.json`: byte-level BPE, 248 044 vocabulary + 33 added tokens,
247 587 merges, `normalizer: NFC`, `ignore_merges: false`, pre-tokenizer
`Split(regex) → ByteLevel`. The regex differs from the engine's pinned GLM
pattern in three places: `\p{N}` alone (GLM: `\p{N}{1,3}`), `[\p{L}\p{M}]+`
(GLM: `\p{L}+`), and `[^\s\p{L}\p{M}\p{N}]+` (GLM: no `\p{M}`). The engine
today refuses any normalizer and requires `ignore_merges: true`.

`chat_template.jinja` uses features the interpreter lacks today: `[::-1]`
slicing, `loop.previtem` / `loop.nextitem`, the `trim`, `default`, `string`,
`safe` and `items` filters, the `undefined`/`true`/`false` tests
(`enable_thinking is true`), `startswith`/`endswith`, `~` concatenation,
tuple literals with `in`, and a macro with default arguments. Reasoning is
`<think>\n…\n</think>\n\n`; tool calls are
`<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`
(the `<tool_call>` markers are single tokens, the inner tags are text); tool
results arrive as `<tool_response>…</tool_response>` inside a user turn.

### 1.11 Checkpoint census

131 shards, 152 089 tensors, 172.76 GiB. By class:

| class | tensors | dtype | GiB |
|---|---:|---|---:|
| routed experts, 48 × 512 × {gate, up, down} | 73 728 (+73 728 scales) | F8_E4M3 (+BF16 `[5,20]`/`[20,5]`) | 112.50 |
| n-gram table, 128 shards `[2 500 012, 160]` | 128 | F8_E4M3, one BF16 scale | 47.68 |
| GDN, 36 layers (`in_proj_qkv` 10240×2560, `in_proj_z` 6144×2560, `out_proj` 2560×6144, conv `[10240,1,4]`, `in_proj_a/b` 48×2560, `A_log`, `dt_bias`, `norm` [128]) | 324 | BF16 | 3.87 |
| MTP (experts 2.34 + one QSA layer, GR sites, `fc_embedding`, `fc_hidden`, two pre-fc norms, mixer) | 1 578 | mixed | 2.42 |
| `embed_tokens` / `lm_head` | 2 | BF16 | 1.18 / 1.18 |
| GR, 97 sites (`hc_norm` [10240], down `[320,10240]`, up `[10240,320]`, `block_inject_weight` `[4,10240]`) | 386 | BF16 | 1.18 |
| QSA, 12 layers (`q_proj` 12288×2560, `k/v_proj` 512×2560, `o_proj` 2560×6144, `q/k_norm` [256], indexer `index_qk_proj` 640×2560, `q/k_layernorm` [128]) | 108 | BF16 | 1.15 |
| vision tower | 337 | BF16 | 1.11 |
| shared experts + `shared_expert_gate`, 48 layers | 192 | BF16 | 0.44 |
| routers `mlp.gate` `[512, 2560]` | 48 | BF16 | 0.12 |
| PLE projections + norms + conv + hash buffers (layer 1) | 10 | BF16 / I64 | 0.06 |

## 2. Cost model by world size (TP=2 and TP=4)

### 2.1 Placement per class, as a function of the world size W

Every slice below is a formula in W, so one loader, one binding table and
one set of kernels serve both worlds (and would serve W=8 unchanged: 16, 48
and 24 heads all divide by 8, the expert slice becomes 80 wide on a 16-row
scale grid, each kv head lives on four ranks). Nothing is placed by hand per
world.

| class | placement | rationale |
|---|---|---|
| routed experts | every expert on every rank, sliced on the intermediate dim: 640/W per rank (320 at TP=2, 160 at TP=4) on a gcd(128, 640/W)-row scale grid | the GLM rule (no busiest rank); D2 |
| shared expert | intermediate-sliced 640/W; its scalar gate replicated | as GLM |
| router | replicated | routing must be rank-identical; 2.6 MB per layer; an all-gather per layer costs more than it saves at any W |
| GDN | head-sliced: 16/W k/q heads and 48/W v heads per rank (8/24 at TP=2, 4/12 at TP=4). With h = 16/W: `in_proj_qkv` rows `[128hr, +128h)`, `[2048+128hr, +128h)`, `[4096+384hr, +384h)`; conv channels the same three segments; `in_proj_z` rows `[384hr, +384h)`; `in_proj_a/b`, `A_log`, `dt_bias` per v head; `out_proj` columns `[384hr, +384h)`; `norm` [128] replicated | every slice 128-aligned at both worlds; one all-reduce after `out_proj` |
| QSA | 24/W q heads per rank (`q_proj` rows per head `[512h, +512)` keep q and gate together); kv head ⌊r / (W/2)⌋, so at TP=2 each rank owns one kv head outright and at TP=4 each head is shared by a pair; `k/v_proj` rows of that head; `o_proj` columns of the rank's heads; indexer replicated | 2 kv heads cannot split past W=2; the indexer selection must be identical everywhere |
| GR | `gr_placement = replicated` (default until measured) or `sliced`: `input_mix_weight_down` rows `[320r/W, +320/W)`, `input_mix_weight_up` the matching columns, one extra all-reduce of the `[10240]` gate logits per site; `block_inject_weight` and `hc_norm` always replicated | the only placement that is a trade-off rather than a formula; D3 |
| PLE | table sharded by hash head, 16/W heads per rank (8 at TP=2, 4 at TP=4) = the contiguous row range `[off_{16r/W}, off_{16(r+1)/W})`; `key_proj`/`value_proj` sliced on K to the rank's 160·16/W embedding columns; norms and conv replicated | D4 |
| embeddings, `lm_head` | vocabulary-sharded 248320/W | as GLM |
| MTP | as its main-layer counterparts; `fc_*` replicated (26 MB) | |

### 2.2 Bytes per token per rank

| class | full model MB/token | TP=2 per rank | TP=4 per rank |
|---|---:|---:|---:|
| routed experts (10 of 512 × 3 matrices × 48, FP8) | 2 359 | 1 180 | 590 |
| GDN projections (36 layers, BF16) | 4 153 | 2 076 | 1 038 |
| GR (97 sites, BF16), replicated | 1 272 | **1 272** | **1 272** |
| `lm_head` | 1 271 | 636 | 318 |
| QSA (12 layers) | 1 235 | 637 | 354 |
| shared experts | 472 | 236 | 118 |
| routers, replicated | 126 | 126 | 126 |
| PLE projections + table rows | 65 | 32 | 16 |
| embedding row | 0.005 | ~0 | ~0 |
| **total** | **10 953** | **6 195** | **3 832** |
| floor at 240 GB/s | | **25.8 ms** | **16.0 ms** |

The stream is `Rep + S/W` with `Rep ≈ 1.44 GB` (GR, routers, indexer) and
`S ≈ 9.5 GB`, so the replicated share grows with the world: 23 % at TP=2,
38 % at TP=4, 54 % at a hypothetical TP=8. That is the horizontal-scaling
argument for `gr_placement = sliced`, which moves 1.1 GB from `Rep` to `S`
(floors 23.2 / 12.0 / 6.4 ms) at the price of 97 collectives per token —
measured at 36–60 µs each on this fabric (D3), which is more than the
bandwidth it buys at both worlds — and for FP8 GR, which halves it in place
(23.2 / 13.3 / 8.4 ms) at no collective cost. GLM-5.3-Flash at TP=4 is 5.86 GB and 24.5 ms; Qwen at TP=2
lands at GLM-TP=4's floor with a cheaper two-party collective.

With MTP the step is one replay over the verify rows plus the MTP layer
(≈ 135 MB per rank at TP=2, 90 at TP=4), so ≈ 26.3 / 16.4 ms per replay; at
GLM's 1.7 tokens per replay that is ≈ 15.5 / 9.7 ms per token at the floor.
Expected after the same optimization discipline as GLM: T=1 in the 27–29 ms
range at TP=2 and 17.5–19.5 at TP=4; MTP ≈ 17–19 and 12–13.5 ms per token.

The levers past the floor are all quantization decisions (§5, Q8): FP8 GR
(−0.64 GB at any W), FP8 GDN projections (−2.08/W GB), FP8 `lm_head`, NVFP4
experts. All four together put the TP=4 floor near 9.5 ms and the TP=2 floor
near 16 ms.

### 2.3 Resident memory per rank

| | TP=2 | TP=4 |
|---|---:|---:|
| backbone weights per rank (experts 112.5/W, MTP 2.4/W, GDN 3.9/W, QSA, embed/head 2.4/W, shared, router; GR 1.18 and PLE 0.06 replicated) | 62.7 GiB | 32.1 GiB |
| n-gram table shard (47.7/W) | 23.8 GiB | 11.9 GiB |
| **total resident** | **86.6 GiB** | **44.0 GiB** |

GLM-5.3-Flash serves at ~82 GiB per rank beside the ~14 GiB CUDA context,
so TP=2 sits 4.6 GiB above the proven working point and leaves roughly
25 GiB for the KV pool, the prefix cache, scratch and the bus staging.
*Measured 2026-09-09 with the loader's byte formulas on the landed
checkpoint (`qwen_load_check`): 44.98 GiB per rank at TP=4 (31.53 layers +
1.53 globals + 11.92 table) and 87.29 GiB at TP=2 (61.62 + 1.83 + 23.84),
loaded resident on the nodes with 116 GiB free before the load; the
embedding is replicated (a row gather, as GLM's) rather than sharded, hence
the 0.9 GiB above the table's estimate.* The full memory plan (KV pool,
arena, scratch) comes with Q6's engine. TP=1 (172 GiB) does not fit
resident and stays the streaming diagnostic world.

KV per token per rank: 13 attention layers × 1 KB (K+V bf16 for one kv
head — the same at both worlds, since a rank never holds more than one kv
head) + ≈ 70 B of compressed index keys ≈ 13.1 KB (GLM: ~44 KiB), so
262 144 tokens cost 3.4 GB. GDN state 56 MB per request slot at TP=2, 28 MB
at TP=4, plus the 184 KB PLE conv state and 2 n-gram context ids, and the
same per prefix snapshot — the prefix cache is an order of magnitude cheaper
than GLM's.

Replicating the whole n-gram table on every rank buys nothing under D4 (the
sharded gather adds no collective) and cannot fit at TP=2, so the table is
always sharded.

### 2.4 Collectives per token

96 boundary all-reduces (2 per layer, bf16 `[2560]`), the PLE partials riding
layer 0's post-attention reduce (D4), the vocabulary-sharded head exchange —
against GLM's 90 + head. The count does not depend on W; the cost per
collective does (a two-party handshake at TP=2 against the measured ~50 µs
of handshake, skew and fold across four ranks), and `gr_placement = sliced`
adds 97 per token. The bus's single-outstanding contract is unchanged.

### 2.5 Prefill

≈ 12 GFLOP per token active (6 B parameters) → 8 192 tokens ≈ 100 TFLOP,
≈ 1 s across two ranks and 0.5 s across four at the dense bf16 rate the GLM
prefill kernels reach, plus the one-time expert stream (all 512 experts of
every layer are touched: 56 GiB per rank at TP=2, 28 at TP=4; 0.25 / 0.12 s).
GLM-Flash prefills 8 192 tokens in 5.7 s at TP=4; targets of 2–3 s at TP=2
and 1–1.5 s at TP=4 are realistic. The GDN recurrence is token-sequential per
state row but has 12 heads × 128 rows of independent chains per rank per
layer, which is enough parallelism for the recurrent kernel to stay off the
critical path; the chunked (WY) tensor-core form is a later prefill lever.

## 3. Reuse, adapt, new

| module | status | note |
|---|---|---|
| CollectiveBus, roster, graph adapter mechanics, arena, streams, trace, hf_cache, safetensors, minijson | reuse | generic today |
| scheduler, prefix cache, HTTP/SSE service, sampler, json/tool grammars | reuse | the service interface takes `GlmTextConfig&` and must be widened (Q1) |
| FP8 block-128 GEMV/GEMM (`fp8_gemv.cuh`, `scale_gemm`, `quant_matrix`), grouped MoE prefill, MoE decode slot kernels | adapt | 32-row scale grid for 160-wide slices (D2); router variant; no clamps |
| bf16 GEMV, L2 prefetch, top-k select, pick/sample kernels, norm kernels | reuse / small variants | `(1+w)` norms, group norm 2560 |
| KDA conv + recurrence + gated norm (`kda.cu`) | fork → GDN | scalar decay, 3:1 head sharing, unequal conv segments |
| DSA indexer/select/listed attention (`dsa.cu`) | pattern, not code | QSA scorer over compressed keys, top-512 blocks, GQA-256 listed attention (new kernels) |
| mHC (`glm_mhc.cu`) | pattern, not code | GR mix/combine are new kernels; the site placement and stream handling carry over |
| MTP transaction (`glm_spec.cu`, `mtp.cpp`) | adapt | different input fusion; the same commit/snapshot contract |
| tokenizer | adapt | second regex, `\p{M}`, NFC, `ignore_merges=false`, goldens |
| chat template interpreter | extend | the features in §1.10 |
| tool parser / tool grammar | new | the `<function=…>` format |
| loader / binding / resident image | new tree, same shape | `src/models/qwen/` mirroring `src/models/glm/` |
| PLE (hash, gather, gate, dilated conv) | new | |
| QSA block compression and caches | new | |

## 4. Design decisions

**D1 — One world-uniform TP geometry for W ∈ {2, 4}, intermediate-sliced
experts, no expert parallelism.** Every placement in §2.1 is a formula in W,
validated by the geometry check at boot (head counts, expert width, kv-head
sharing, vocabulary and table ranges must divide as stated); the same
binding table, loader and kernels serve both worlds, and the world-1 build
is the degenerate rank 0 of the same code for the parity tests. Expert
parallelism (whole experts per rank) has the same mean bytes per rank as
slicing but a busiest-rank tail — 4–5 experts against a mean of 2.5 at
TP=4, 6–7 against 5 at TP=2 — and GLM retired it for that reason. Memory
does not force EP at either world (§2.3). Where a world size changes a
trade-off rather than a formula, the choice is a config variable with a
measured default per world (D3); knobs that only tune (prefetch rate, the
batch crossover, `kv_capacity`, `kv_dtype`) stay per-deployment settings as
they are for GLM.

**D2 — 640/W-wide expert slices on a gcd(128, 640/W)-row scale grid,
exact.** 640 = 5 × 128, so a half slice (320, 2.5 blocks) or a quarter slice
(160, 1.25 blocks) starts mid-block and DESIGN §5.2's 128-alignment rule
would reject it. The slice is made exact by re-blocking the scale grid at
load: the sliced dimension gets blocks of 64 rows at TP=2 or 32 at TP=4
whose scale is the parent 128-block's scale, replicated (bit-identical
dequantization; scale storage grows 2–4× on that axis, still under 1 MB per
layer). The block size is a parameter of the kernels; GLM's slices keep 128.
Kernel work: the scale index in `fp8_gemv.cuh` (`row / 128` → a template
scale-row count), the prefill tile kernel's 64-row tile reads one scale row
per sub-block, `quant_matrix` views anchor at the sub-block, the loader
validators accept sub-block-aligned starts for this format variant. The
alternative — re-quantizing slices onto their own 128-grid — is not exact
and is rejected.

**D3 — GR placement is a config variable, `gr_placement ∈ {replicated,
sliced}`, defaulted per world by measurement.** Every rank holds the hyper
state, so every rank can compute the gates; replicated GR costs 1.27 GB per
token at any W and no collective. Sliced GR (`input_mix_weight_down` rows
`[320r/W, +320/W)`, `input_mix_weight_up` the matching columns, one
all-reduce of the `[10240]` gate logits per site before the sigmoid, then
the weighted mean on the full `Rn` every rank has) costs 1.27/W GB and 97
collectives per token. On paper the two are within a factor of two of each
other at both worlds — 2.65 ms of bandwidth against ~97 × 30 µs at TP=2,
4.0 ms against ~97 × 50 µs at TP=4 — which is why it is a knob and not a
decision: the collective cost is a fabric measurement that moves with the
world size, while the bandwidth side moves with quantization (FP8 GR halves
it and tips the balance toward replicated at every W). The sliced variant
is the one that scales horizontally (§2.2), so it must exist and be gated at
both worlds; the default is set by Q0's measurement and recorded per world
in the cluster config. Neither variant needs a new bus primitive.

*Measured 2026-09-09 (Q0, `scripts/fabric_gr_probe.sh`, `scripts/fabric_bus_probe.sh`,
`build-ci/fabric-runs/gr_probe_2026-09-09/`):* one more recorded collective
node in the GLM-5.3-Flash decode graph on the four nodes costs **47 µs on
the T=1 replay** (29.98 → 31.49 ms per step for 32 probe nodes, transcripts
identical) and **60 µs on the MTP replay** (40.40 → 42.32 ms); the bus's
eager all-reduce is 39 µs p50 at world 4 and 30 µs at world 2, so the
two-node in-graph cost scales to ≈ 36 / 46 µs. Sliced GR therefore costs
4.6–5.8 ms per token at TP=4 against 4.1 ms of bandwidth saved (230 GB/s),
and 3.5–4.5 ms at TP=2 against 2.8 ms saved; with FP8 GR the saving halves
again. **`gr_placement = replicated` is the default at both worlds.** The
sliced variant stays designed (this paragraph) but is not built unless the
per-collective cost falls under ~28 µs at TP=2 or ~42 µs at TP=4 — a bus
change, not a model change — or a wider world (TP=8: 54 % replicated share)
arrives. Q6's gate measures the replicated build only.

**D4 — n-gram table in device memory, head-sharded, partials fused into
layer 0's boundary reduce.** On GB10 the table lives in the same LPDDR5x
either way; `cudaMalloc` (2 MB pages, ~248 GB/s) beats pinned host memory
(4 KB translations, ~180 GB/s at tens of GB). Sharding by hash head (16/W heads per rank) keeps
each rank's rows a contiguous global range, so the sharded loader reads only
its own checkpoint shards. Each rank gathers its 16/W rows per token (160 B random
reads each), multiplies by its K-slice of `key_proj`/`value_proj` (65/W MB),
and the `[12800]` partial rides layer 0's post-attention all-reduce as extra
elements of the same buffer — zero additional collectives; the gate, norms
and conv run at the top of layer 1 on the reduced vectors. The n-gram ids
are known before the step starts (they depend on token ids only), which is
also what makes the MTP draft rows' ids available before the replay.

**D5 — QSA caches and the selection contract.** Per attention layer: paged
K/V bf16 for the rank's kv head (the DSA block-table design), a compressed
key cache written once per completed 4-token block, and a per-request
pending ring of ≤ 3 raw indexer keys. Decode scores `tokens/4` compressed
keys (at 262 K context: 16.8 MB per layer, 200 MB per step, ~0.85 ms), then
top-512 with the pinned tie rule, then the listed GQA attention over ≤ 2051
positions (split across blocks and combined, as `dsa_attn_partial/combine`
do). Prompts of ≤ 2051 tokens take the dense kernel, which computes the same
softmax set. The indexer is replicated so every rank selects identically;
its kernels must be deterministic (fixed reduction order).

**D6 — GDN as a KDA variant, not a new operator family.** The recurrent form
(state row per warp, token-sequential, fp32) serves decode and prefill as it
does for KDA; the changes are the scalar decay, the k-head sharing, and the
conv channel layout. Snapshot/rollback (`KdaStateSnapshots`,
`KdaConvSnapshots`) and the request-row batching carry over unchanged.

**D7 — Extract the engine mechanics before building the second engine.**
`decode.cpp`, `fabric_engine.hpp`, `tp_bus.hpp`, `mtp.cpp`, `prefix_arena.hpp`
and `resident_image` are typed on `Glm*` but their behavior — graph capture
with the collective recorder, request slots and row spans, the MTP
commit/snapshot transaction, the prefix arena, the memory plan, the scheduler
adapter — is model-independent. To support further models without duplicating
the engine, Q1 extracts
those mechanics behind a model "layer walk" interface with the GLM engine as
the first client, gated by identical transcripts and unchanged step times on
the fabric. Kernels, binding tables, loaders and the walk itself stay
per-model.

**D8 — Text only.** The vision tower (1.1 GiB) is not loaded; image inputs
are refused at the API, as for GLM.

**D9 — Numerics and parity method.** The reference is the transformers
modular implementation; parity is established module by module with host
oracles in `float`/`double` (the `*_reference.{hpp,cpp}` convention), kernel
tests against them, and the Python per-layer dump harness as the independent
third implementation. The full HF model cannot run on a Spark (bf16
dequantized experts alone are 225 GB), so end-to-end quality is judged by
the task-level eval runner (`scripts/serve_eval.py`: HumanEval / GSM8K /
schema extraction; the GLM FP8 baseline is 94.5 / 97.7 / 100 %) and by
perplexity/logprob agreement across builds, per DESIGN §1's rounding-level
rule. MTP must stay identical to plain decode by construction.

## 5. Implementation and validation record

Each milestone lists its deliverables and its gate. Nothing merges without
its gate; GLM's transcripts and step times are a standing regression gate
from Q1 on.

**Q0 — Facts (this document; complete when the download lands).** Extend
`tools/checkpoint_audit.py` to the Qwen layout and produce
`docs/qwen38_checkpoint_budget.md` from the local shards; verify the stored
hash buffers against §1.7; run the memory plan at both worlds; write the
tokenizer and template diff into the test plan. Also the measurement that
sets the `gr_placement` default: inject one dummy `[10240]`-element
all-reduce per mHC site into the GLM decode step (≈ 90 extra collectives
per token) and time it on the four-node fabric, and time the two-party
collective on a node pair with the bus benchmarks for the TP=2 figure;
record both in `docs/measurements.md`. Gate: the audit reproduces §1.11 and
§2.3 from local bytes; the GR measurement is recorded per world. *Done
2026-09-09 except the memory plan, which needs Q2's loader:*
`docs/qwen38_checkpoint_budget.md`, the hash buffers verified against §1.7,
the measurement in D3.

**Q1 — Engine core extraction (GLM as the first client).** Move the graph
adapter, boundary reducers, request rows/spans, spec commit and snapshot
plumbing, prefix arena, resident-image cache and the `SchedulerEngine`
adapter behind a per-model walk interface; widen `serve_openai` past
`GlmTextConfig&`; add the `architectures` discriminator to `dgpp-serve`
(`--model` resolves either family from the hub cache). Gate: 4-way op-stream
md5 unchanged, `glm_gen_check` transcripts identical, T=1 and MTP step times
within noise on the fabric, all CTest targets green after a full rebuild.
*Stage 1 done 2026-09-09:* `src/engine/` (decode_outputs, boundary_reducer,
tp_bus, graph_engine and eager_engine as templates over the model type,
prefix_arena, speculative, graph_check, step_timing) with `models/glm/*.hpp`
binding them to GlmDiagnosticModel under the old names; the model surface
renamed `kv_blocks_*` / `session_snapshot_align`; GLM's DSA anomaly report
moved into its `session_close`. 35/35 CTest; fabric transcripts identical to
the pre-refactor runs (md5 068a6dff…), 29.99 / 40.33 ms per step against
29.98 / 40.40. `loaders/architecture` detects the family from `architectures`.
The `dgpp-serve` boot split (a per-family boot returning the engine, a
serving-config interface replacing `GlmTextConfig&` in `serve_openai`) lands with
Q2's config.

**Q2 — Qwen config, binding, loader, resident image.** *Done 2026-09-09; the
record is PLAN.md's Q2 row: the shared weight builder extracted from the
GLM loader (`loaders/weight_build.hpp`), the Qwen config/binding/loader on
it, the fixture gates at worlds 1/2/4, and the real checkpoint resident at
worlds 2 and 4 with identical digests on every rank.* `src/models/qwen/`:
`QwenTextConfig` (reject-rather-than-guess like GLM's), the expected-tensor
table derived from config (§1.11), the replicated/sharded classifier of §2.1,
the per-layer build in a fixed grant order, counting-mode byte formulas, the
32-row scale grid (D2), the n-gram shard range loader, boot digests and byte
reconciliation, the synthetic mini-checkpoint fixture (from the binding
table, with a tiny n-gram table). Gate: fixture and real checkpoint boot at
world 1, world 2 and world 4 (the real checkpoint resident at 2 and 4,
streaming at 1); per-rank byte identities hold; replicated digests
agree; the shard-parity test pins the sharded build bitwise to the
full-load-plus-views path.

**Q3 — Kernels and oracles.** In this order, each with a host oracle, a CUDA
test (kernel vs oracle; chunked vs unchunked bitwise; TP slice vs full;
graph replay vs eager) and a microbench: (a) norms — `(1+w)` RMSNorm, group
norm 2560, the GDN gated norm; (b) GDN conv/recurrence/gated norm (fork
of KDA, D6); (c) GR mix and combine (fused group-norm + down GEMV with
split-K; up GEMV + sigmoid + weighted mean; inject dots + stream update),
plus the M>1 tiled forms for prefill; (d) MoE: router variant (softmax
top-10 renorm, shared sigmoid gate, no clamps), the 32-row scale grid in the
FP8 decode and prefill kernels; (e) PLE: hash kernel with EOS segmentation,
head-sharded row gather with fp8→bf16×scale, K-sliced projections, the fused
gate/norm kernel, the dilated depthwise conv with its 9-step state and
snapshots; (f) QSA: indexer projection + q norm/RoPE, raw-key ring and block
compression, scorer, top-512 with the tie rule, list expansion with the
tail, listed GQA-256 attention (decode split/combine; prefill row-batched),
dense causal GQA-256 for short prompts, gate multiply. Gate: every test
green; microbench numbers recorded in `benchmarks/results/`.

Status (2026-09-09): (a)–(f) built with oracles and tests — norms, GDN,
GR, the MoE (the shared MoE layer's `SoftmaxTopk` router mode and
`n_shared_experts = 0`, the BF16 shared expert composed on the fp32 chain
in `models/qwen/moe_layer`), the PLE (`kernels/qwen_ple`) and QSA
(`kernels/qsa`: the compressed-key cache co-located with the token blocks,
the scorer in a fixed fp32 order the oracle reproduces bitwise, the
streaming composite-key selection, the split listed attention merged by
`dsa_attn_combine`). Deferred to the milestones that need them: the
sub-128 expert scale grid in the FP8 kernels (Q4's TP forward), the dense
causal attention kernel and a GEMM-fed prefill scorer (Q7), the
microbenches (Q8).

**Q4 — Forward parity.** World-1 streaming diagnostic forward
(`qwen_forward_check`) with per-layer dumps against the Python harness on the
real checkpoint; then the TP=2 and TP=4 resident forwards with the boundary
reducers.
Gate: per-layer hyper-state deltas within the numerics budget at every layer
(the GLM mHC-stream metric adapted to the 4-branch GR state), argmax
agreement on the dump prompts, world-2 and world-4 output bitwise equal to
world-1 where
the contract says so and rounding-level elsewhere.

Status (2026-09-09): world 1 done — `models/qwen/layers` + `forward`
(`QwenModel`), the pure-python reference `tools/qwen_reference_dump.py` and
the ctest chain (per-layer hyper states within l2 2.1e-3, top-1 exact bar a
near-tie); `qwen_forward_check` on the real checkpoint predicts " Paris" and
" fibonacci" for the two probe prompts. TP=2 and TP=4 (`qwen_tp_test`, loopback on the fixture): the rank/world
interface with the three folds per layer (attention, MoE, the PLE layer's
key/value partials), the lm head vocab-sharded, the fp8 GEMV core on the
re-blocked scale grid — bitwise across ranks, within l2 3.6e-3 of world 1,
routing and top-1 identical. The real checkpoint across the fabric
(`scripts/fabric_qwen_forward.sh`): TP=2 and TP=4 streaming, every rank's
49 hyper-state digests identical, the merged argmax equal to world 1's.
Left: a torch per-layer reference once torch sits beside the checkpoint.

Risk found (2026-09-09): the bus's bulk RS+AG path stalls at world 4 when
a buffer spans fewer bulk stripes than ranks (a 36 KB fold: one stripe;
the ranks without a shard never arrive) — the eager reducer routes any fold
above two latency slots there. The loopback test keeps its folds on the
latency path with a 256 KB slot; the fabric diagnostic needs the same or
the bulk plan fixed for C < W. Production decode folds ride the recorded
collectives and prefill folds are megabytes, so neither is on that path.

**Q5 — Tokenizer, template, tools.** Second pre-tokenizer pattern with
`\p{M}` from the Unicode tables, NFC normalization, `ignore_merges=false`,
tokenizer goldens generated from HF tokenizers keyed on the file hash; the
template interpreter features of §1.10; a Qwen tool-call parser on the
`<tool_call>`/`<function=…>` format and the matching constrained-decoding
grammar; `reasoning_content` from `<think>`. Gate: tokenizer goldens 100 %
on the corpus; the checkpoint's template renders the transformers-generated
strings byte-exactly for the message fixtures; tool round-trips through the
eval harness.

Status (2026-09-09): done. The tokenizer's second pinned shape (the Qwen
regex, NFC over generated tables, `ignore_merges=false`, string merges,
empty affixes; `qwen_tokenizer_test` 94/94 vs HF tokenizers); the template
interpreter's new constructs (slices, tuples, loop.previtem/nextitem, the
trim/default/string/safe/items filters, the true/false/undefined tests,
startswith/endswith, macro defaults, raise_exception; `qwen_chat_template_test`
20/20 byte-exact with ids); the Qwen tool format as `ToolFormat::kQwenXml`
in the parser (the closed block's text) and the grammar (the literals folded
into the text automaton, a free top, `<|im_end|>` the call-turn EOS), both
gated on the real tokenizer. `reasoning_content` from `<think>` is the
parser's existing split.

**Q6 — Decode engine, MTP, prefix cache, service.** The Qwen walk on the
extracted core: scalar and row-batched T=1 graphs, the one-graph MTP replay
(input fusion of §1.8, transactional GDN state, PLE conv state, n-gram
context, K/V and index caches), prefix snapshots including GDN/PLE state,
the memory plan, `--memory-plan` output, loopback gates on their own port
range, the two- and four-node gates (`fabric_xcript`, `fabric_logprob`,
`fabric_mtp_classes`). Gate: MTP transcripts identical to plain decode;
hot == cold prefix bitwise; the eval runner within the model card's band;
T=1 and MTP step times recorded against the 25.8 ms (TP=2) and 16.0 ms
(TP=4) floors (replicated GR, the measured default of D3).

Status (2026-09-09): stages A and B built and gated on the fixture. The
session surface lives on the one row walk (`models/qwen/forward`): request
slots own their GDN recurrent/conv states, PLE conv state and n-gram
context (device int32 [4], a 16-byte state family), their rings in every
QSA layer and their row of the shared block table; the K/V and compressed-
key caches are one refcounted paged pool (`models/qwen/kv_pool`, the DSA
pool's design). Prefill cuts at chunk multiples and pool-aligned boundary
images; eager decode runs T<=4 rows with per-row spec snapshots of every
family and `session_rollback`; the prefix snapshot/attach/resume is
implemented (the hot == cold gate is next). The graph era is the engine
core's: the device-driven scalar T=1 graph and the fixed slot-major row
batch capture kernels-only (173 kernel + 11 collective nodes on the
fixture) — the PLE hash reads the device context and the context/conv row
kernels select per-request state and skip padding rows, the MoE decode
runs the routed-only slot chain (`GlmMoeLayer::enqueue_decode_f32`, the
shared slot's blocks return at n == 0) with the BF16 shared expert behind
it, every boundary fold writes into the reducer's staged buffer (the PLE
key and value partials consumed one handout at a time, D4's fusion still
open). Gates: `qwen_decode_test` — a one-shot prefill's last row bitwise
the cold forward, 12 T=1 steps vs the re-forward at relative l2 2.5e-3
with no hard top-1 miss, two interleaved slots bitwise their solo runs,
8-row chunked prefill with a boundary cut, close/reopen bitwise, the
pool's accounting; `qwen_engine_test` — loopback world 2, the graph
engine's scalar and row-batched replays (the 2-slot and full families, a
closed slot padding at -1) equal the eager engine's transcripts exactly
and world 2 equals world 1 on three prompts.

The real checkpoint's decode-vs-re-forward question (2026-09-09, world 1
streaming, "The capital of France is" -> " Paris. The capital of"): the
greedy transcript disagreed with the re-forward of prompt + transcript on
one row (relative l2 0.18, a hard top-1 miss at a 2.3 % gap) where the
fixture agrees at 2.5e-3. Localized: the prefill is BITWISE the forward
at the same row count; two forwards of one prompt at T=5 and T=6 (both
the interface's GEMV family, m <= 8) are bitwise identical at every layer and
row, as are T=9 and T=12 (both cuBLASLt); across the families the hyper
state differs by 0.3-0.8 % on every row already after layer 0, then
amplifies through MoE routing flips (10 of 50 by layer 5, 5-20 per layer
after) to 20 % by layer 47. Both families match the fp64 oracle within 2
bf16 ulps at every Qwen decode/prefill shape and row count
(`bf16_gemv_test`: `bf16_gemv_qwen_shapes_every_row_count_matches_oracle`,
`cublaslt_qwen_shapes_at_prefill_rows_match_oracle`), so the difference is
the fp32 accumulation order under the real activations' cancellation (a
bf16 output flip on a few per cent of the elements), which this
512-expert stack amplifies. The consequence for the gates: real-model
decode parity against a re-forward (or across worlds, or across GEMM
families) is a near-tie question, never a bitwise one — the fixture's
bitwise/2.5e-3 gates stand for the machinery, the real model's quality
gate is the task eval through the server (the eval runner's band) and the
transcript's sanity. Stage C (the MTP draft block, §1.8) is built on the
same walk (the eager speculator and the one-graph draft gated on the
fixture); `dgpp-serve` dispatches on the architecture through a family
interface (the GLM and Qwen models, memory plans and engines behind one
interface; the bus's latency slot is the family's widest recorded fold —
GLM's 8 x 4096 rows, Qwen's 8 x 10240 for the PLE key partial);
`deploy/cluster_qwen-3.8-flash-next_fp8_w4_mtp1.json` (MTP + the decode graph) and
`deploy/cluster_qwen-3.8-flash-next_fp8_w4_plain.json` (plain T=1) name the fabric worlds, and
`scripts/fabric_qwen_serve.sh CONFIG OUT [--compare REF] [--eval]` runs a
world's gates (`scripts/serve_greedy_transcript.py` for the MTP-vs-plain
transcript identity at temperature 0, `serve_bench.py` for the pace,
`serve_api_check.py`, `serve_eval.py`).

First fabric reading (2026-09-09, world 4 resident, MTP depth 1, the
decode graph, 262 144-token pool, 4 slots; `build-ci/fabric-runs/
qwen_serve_mtp_2026-09-09/`): rank 0's model constructed in 12.5 s (the
resident image captured on the first boot), the 14 graph variants
warm-captured in 70 s, the world serving 84 s after launch. Single-stream
greedy requests of 256 tokens: 34 ms per pass, 1.6–1.9 tokens per pass,
draft acceptance 58–86 % (prose lower, math higher), 18–22 ms per token;
prefill 4.5–5.7 ms per token on 80–100-token prompts. The task eval
through the server (`serve_eval.py`, greedy, 4 in flight): GSM8K 59/60
(98.3 %), HumanEval 39/40 (97.5 %), schema extraction 30/30 — the model
card's band (GLM-5.3-Flash's FP8 baseline on the same items: 97.7 / 94.5
/ 100). Under 4 concurrent requests: 45–64 ms per batched step at 2.0
tokens per step per request (acceptance 98–99 % on the eval's
arithmetic and code), 8–11 ms per token; the four ranks' op streams
identical. The plain T=1 world (`deploy/cluster_qwen-3.8-flash-next_fp8_w4_plain.json`,
`build-ci/fabric-runs/qwen_serve_t1_2026-09-09/`): 28.9 ms per step
single-stream (the TP=4 floor is 16.0 ms — Q8's distance), 29 ms per
token at the client; its four greedy transcripts are IDENTICAL to the MTP
world's (the "MTP transcripts identical to plain decode" gate on the real
checkpoint, world 4). The API check's one failure — reasoning_tokens 0,
the reasoning text in `content` — was the opened-thinking test: the Qwen
turn's generation prompt ends in `<think>` followed by the bare newline
token, which the "last token is think_open" rule missed;
`ChatMarkers::prompt_opens_thinking` now takes both tails (the
tokenizer's newline id is a marker), the parser starts in the reasoning
and the usage counts it (unit test `tool_parser_promptOpensThinkingSeesTheQwenNewline`).
Rebooted with the fix (`qwen_serve_mtp2_2026-09-09/`): the API check
passes whole (stop, n, logit_bias, cached_tokens 48, reasoning_tokens),
the four greedy transcripts are token-identical to the first boot's, and
the boot from the resident image takes 30.5 s (11.2 s model, 17.9 s
captures — the first boot's 70 s captured the image beside them).

World 2 (two nodes, 87.3 GiB resident per rank, `deploy/cluster_qwen-3.8-flash-next_fp8_w2_*.json`,
`qwen_serve_w2_mtp_2026-09-09/`): first boot 220 s (23 s model with the
image capture, 196 s captures); MTP 46–54 ms per pass at 1.6–1.9 tokens
per pass (acceptance 59–86 %), 25–34 ms per token single-stream; at 4
concurrent 95–143 ms per batched step at 1.9 tokens per step per request,
12–18 ms per token; the API check passes; the eval GSM8K 59/60,
HumanEval 39/40, extract 30/30; op streams identical on both ranks. The
plain T=1 world (`qwen_serve_w2_t1_2026-09-09/`, 39 s boot from the
image): 37.7 ms per step (the TP=2 floor is 25.8 ms), its four greedy
transcripts identical to the MTP world's. TP=2 is a served world.

**Q7 — Prefill.** Chunked 2 048-token prefill on the same walk: GDN
recurrent over rows, QSA prefill selection tiled over `[T, T/4]` scores with
streaming top-k and the row-batched listed attention, GR as GEMMs, PLE
batched gather and GEMMs, grouped MoE. Gate: prefill parity with the
world-1 dumps at 512/2 048/8 192 tokens; steady-state prefill times recorded
(target ≤ 1.5 s at 8 192 on four nodes, ≤ 3 s on two).

Status (2026-09-09, the prefill session): the routed experts on the
tensor-core chain. The fp8 ldmatrix tile kernel (`moe_grouped_mma_fp8_ldm`)
and the reference `scale_gemm_kernel` read per-row scales with the grid as
log2 shifts (`rs`, `cs`), so the re-blocked slices — 64 at TP=2, 32 at
TP=4 — take the same kernel as GLM's 128 grid (bitwise for the 128 grid:
`glm_moe_test` 20/20, `scale_gemm_test` 12/12 unchanged); `GlmMoeLayer`
takes the tensor-core kernel on any 32/64/128 grid with hidden and the
slice multiples of 32 (`mma_takes_grid()`), the GEMV core otherwise (the
fixture's 16-wide slice). The prefill walk runs the device-segmented
routed-only chain `GlmMoeLayer::enqueue_prefill_f32` (router, device
segmentation, the grouped tensor-core gate/up/down, the ordered fp32
accumulate handed to the BF16 shared expert; the routing ids/weights ride
async copies into per-layer pinned staging, materialized after the walk's
one sync — no per-layer host sync) through `QwenMoeLayer::enqueue_prefill`;
`DGPP_QWEN_MOE_PREFILL=host` keeps the host-orchestrated chain for A/B.
Gates: `qwen_moe_test` 8/8 (the grouped kernel bitwise the tile kernel per
segment on the 64 and 32 grids, gate/up/down; the prefill chain's staged
routing == the oracle's, its output bitwise the host tensor-core chain and
within the expert budget on all three grids), the fixture chain
(qwen_forward_{fixture,generate,test}, qwen_decode_test, qwen_tp_test,
qwen_engine_test) green. Fabric, world 4 resident
(`scripts/fabric_qwen_prefill.sh` → `serve_prefill_probe.py`: prose
prompts behind a nonce, max_tokens 1, the engine's prefill_ms from
/v1/metrics deltas; `build-ci/fabric-runs/qwen_prefill_{host,mma}_2026-09-09/`):

| tokens | host chain (GEMV core) | tensor-core chain | speedup |
|---:|---:|---:|---:|
| 524 | 868 ms (1.66 ms/tok) | 455 ms (0.87 ms/tok) | 1.9× |
| 2 002 | 2 759 ms (1.38 ms/tok) | 1 248 ms (0.62 ms/tok) | 2.2× |
| 7 708 | 10 331 ms (1.34 ms/tok) | 4 494 ms (0.58 ms/tok) | 2.3× |

Against GLM-5.3-Flash FP8 the same day (1 405 / 6 258 ms at 2 048 / 8 192):
Qwen prefills 8 192 tokens in 4.5 s at TP=4; the Q7 target (≤ 1.5 s)
needs the rest of the walk on the profile (`scripts/fabric_qwen_profile.sh
… --prefill 2048`, the last-burst breakdown). Parity on the real model
(`build-ci/fabric-runs/qwen_serve_mma_2026-09-09/`, the MTP world): the
greedy transcripts diverge from the GEMV-chain run's at near ties
(characters 15–457 of 4 prompts, both continuations sensible — the
rounding-family sensitivity documented under Q6; the prefill's experts
now round as the tensor-core kernel, the decode's as the GEMV core, the
same split GLM runs), the eval unchanged at GSM8K 59/60, HumanEval 39/40,
extract 30/30, the API check whole, op streams identical on 4 ranks,
MTP 8.4 ms/token at 4 concurrent, prefill 2.7 ms/token on short prompts
(the fixed per-prompt cost: 60–70 tokens).

**Q8 — Optimization to the floor and past it.** The GLM playbook in order:
prefetcher coverage of every GEMV, graph interfaces, kernel fusion at the GR
sites, the QSA scorer at long context, then the quantization levers with
perplexity and eval gates: FP8 GR gates (−0.64 GB per token), FP8 GDN
projections (−0.52), FP8 `lm_head`, fp8 K/V, NVFP4 experts (a composed
checkpoint as `tools/compose_nvfp4_hybrid.py` did), depth-2 MTP with
IndexShare. Gate: the golden metric (single-stream decode) per lever,
recorded in the ledger.

### Q9 — the single-stream round (2026-09-10)

The next round of decode and prefill work, its measurements and its ranked
backlog live in `docs/qwen38_optimization_plan.md`: the line rate and the
two floors, today's per-kernel profiles at T=1 / MTP / prefill, the
collective's anatomy at world 4, what landed (the GR inject dots off the
chain), what was tried and reverted (the fused mix beyond one row, a
4 096-token prefill chunk), and what is left in order.

### Q8 status — the decode profile (2026-09-09, world 4, T=1)

`scripts/fabric_qwen_profile.sh deploy/cluster_qwen-3.8-flash-next_fp8_w4_plain.json OUT` (rank 0
under nsys through `dgpp-cluster up --head-wrap`, a 400-token request,
`scripts/nsys_step_breakdown.py` between `spec_commit_kernel` markers;
`build-ci/fabric-runs/qwen_profile_t1_2026-09-09/`): 29.66 ms per step
wall, 27.86 ms GPU busy (93.9 %), 1 707 kernels per step. Per step:

| kernel | ms/step | n/step | us each | share | what it is |
|---|---:|---:|---:|---:|---|
| `bf16_gemv_kernel<1,0>` | 13.05 | 532 | 24.5 | 47 % | the dense weight stream at 218–224 GB/s: GR gates (`down` [320, 10 240] + `up` [10 240, 320] = 13 MB per site, 2 sites per layer, replicated: 1.27 GB per step), GDN in/z/out projections (13 + 7.9 + 7.9 MB per rank), QSA q/k/v/o, the lm head slice (318 MB, 1.34 ms) |
| `bus_allreduce_graph_kernel` | 2.83 | 99 | 28.6 | 10 % | the recorded collectives (5 KB payloads: latency, not bytes) |
| `combine_kernel` (GR inject) | 2.77 | 96 | 28.7 | 10 % | ONE block per row: 4 warps each walking a 320-deep chain of dependent 2-byte loads — latency, not work |
| `moe_slot_gate_up_swiglu` + `moe_slot_down` | 3.51 | 96 | 42 + 31 | 13 % | the routed experts (12.3 MB per layer per rank: gate/up at 82 % of DRAM peak, down at 55 % — its rows are 160 e4m3 bytes at TP=4) |
| `bf16_gemv_kernel<1,1>` | 1.62 | 49 | 33 | 6 % | the shared expert's fp32-out GEMVs |
| `moe_router_dots` | 1.04 | 48 | 21.7 | 4 % | [512, 2 560] bf16 = 2.6 MB in 64 blocks: half of DRAM rate (too few warps in flight) |
| `group_rmsnorm` | 0.71 | 100 | 7.1 | 2.5 % | 4-block norms before every GR GEMV |
| `select_from_keys` | 0.59 | 12 | 49 | 2 % | the QSA selection |
| everything else | 1.74 | | | 6 % | GDN recurrence 6.8 us, attention partials 17.5 us, mix_finish, gate_act, norms, sampling, commit |
| idle (wall − busy) | 1.80 | | | | graph launch and the host's step work |

The step is weight streaming (14.7 ms at ~220 GB/s of the 238 peak; 3.5 GB
per rank per step: GR 1.27 GB, GDN 1.04 GB, experts 0.59 GB, lm head 0.32
GB, QSA 0.35 GB) plus 5.6 ms of latency-bound single-block kernels (the 99
collectives and the 96 GR combines) during which DRAM idles, plus 3.5 ms
of experts. The floor of 16.0 ms is the streaming alone.

The MTP profile (`qwen_profile_mtp_2026-09-09/`, 2 rows per step): 34.96
ms wall, 33.67 busy, 1 821 kernels; the GEMVs cost the same as at one row
(13.6 ms — bandwidth), the expert slot kernels double (64 + 58 us per
layer), the combine and the collectives the same.

Levers, in order (bit-identical unless noted):
1. LANDED: the GR combine as `combine_dots_kernel` (one block per row,
   warp i the i-th inject dot in the SAME per-lane strided chain and xor
   tree, the loads issued 32 deep so the warp is FMA-bound) plus
   `combine_apply_kernel` (one thread per element of the hc*H row), the
   gates staged through a [rows, hc] F32 scratch on the site
   (`qwen_gr_test` bitwise the reference).
2. LANDED: L2 prefetch windows before every fold with the other side's
   first weights — GLM's boundary windows on the Qwen walk
   (`WeightPrefetcher`, 12 MB, light rate; `DGPP_L2_PREFETCH=off` A/Bs,
   `DGPP_QWEN_PREFETCH_DEBUG=1` traces the adds): before the attention
   fold this layer's inject, the MLP-side GR mix, the router, the shared
   expert; before the MoE fold the next layer's attention-side GR mix and
   its first projection (or the mixer and the lm head slice); the PLE's
   two folds likewise; the side stream joined before the walk ends (a
   graph branch under capture). Every window's adds come from ONE image:
   the prefetcher bridges gaps up to 2 MB between adjacent adds, and the
   fixture's layer images sit closer than that — a bridged hole was an
   out-of-bounds read caught by compute-sanitizer, so the 82 KB inject is
   never added beside the next image.
   Fabric, world 4 (`qwen_serve_lever12_{t1,mtp}_2026-09-09/`): T=1 28.9
   → 24.7 ms per step; MTP 34 → 30.3 ms per pass (1.73–1.85 tok/pass,
   16.4–17.5 ms/token); the greedy transcripts identical to the previous
   run's on all 4 prompts in both worlds (both levers bitwise), API check
   whole, op streams identical on 4 ranks.
   GLM-5.3-Flash after the session, against the morning's baseline
   (`scripts/fabric_glm_regression.sh`, `glm_regress_2026-09-09/`): T=1
   29.84 vs 29.81 ms per step, MTP 40.26 vs 40.35 at 72.0 % acceptance
   (23.49 vs 23.54 ms/token), prefill 549 / 1 412 / 6 278 vs 546 / 1 405 /
   6 258 ms at 512 / 2 048 / 8 192, every generated-ids md5 identical to
   the baseline's — no regression from the shared kernels' generalization.
   The window sweep (`scripts/fabric_qwen_pace.sh`, `qwen_pace_sweep_2026-09-09/`,
   T=1 single-stream pace): off 28.2–28.6 ms; 12 MB light 25.1 (the GLM
   default); boundary rate full 27.3–27.8 at any size (as on GLM — the
   full rate lifts the collective's own memory latency); 20 MB 23.6, 24 MB
   23.0, 32 MB 23.1–23.8, 48 MB 23.3–23.5. The Qwen walk's windows carry a
   20 MB budget by default (the GR site's 13 MB pair plus the next
   projection inside the 24 MB L2); DGPP_L2_PREFETCH_MB overrides.
2b. LANDED (round 2, bitwise): the shared expert's decode tail as two
   kernels (`qwen_moe_shared_tail_decode`: gate/up/swiglu with the shared
   gate's block, then down with the fma-accumulate and the rounding in
   the epilogue — seven launches to two, `qwen_moe_test` pins the decode
   path bitwise the host path at 1/3/5/8 rows on three grids); the GR
   site's group norm staged into the down GEMV (block 0 writes Rn) and the
   gate activation into the up GEMV (`qwen_gr_norm_down_bf16`,
   `qwen_gr_act_up_bf16`, four launches to two, `qwen_gr_test` bitwise
   the chain; `DGPP_QWEN_GR_FUSED=off` keeps the chain). Per step at T=1:
   −7 nodes per layer, ~340 fewer graph nodes.
   Fabric after round 2 (`qwen_serve_round2_{t1,mtp}_2026-09-09/`): T=1
   24.9 → 22.5 ms per step (the serve_bench pace, transcripts identical to
   round 1's); MTP 30 → 31 ms per pass at the same acceptance. The knob A/B
   (`qwen_pace_ab_2026-09-09/`): at MTP the fused GR GEMVs cost 1.5 ms per
   token (20.1–20.7 with, 18.4–19.2 without — every down block recomputes
   the two rows' eight group norms in sequence) while at T=1 they gain 0.3
   (22.5 vs 22.8–22.9); the 20 MB window helps MTP too (12 MB: 21.9–22.4).
   So the fused GR path serves the scalar row only (`tokens == 1`), the
   chain the batched rows; the norm's loads are issued four ahead of the
   chain (`block_sum_squares`, the same fmaf sequence). GLM unchanged
   (`glm_regress2_2026-09-09/`: 29.88 / 40.44 ms, prefill 548 / 1 410 /
   6 276, identical ids).
3. LANDED (round 2, bitwise, GLM too): the router's fused select — the
   last block's eight warps stage the token's scores into shared memory
   before warp 0 selects (warp 0's own strided `__ldcg` loop was 16
   dependent round trips per lane at E = 512); the streaming top-k
   (`select_topk_stream`, shared with GLM's DSA) sorts the occupied power
   of two of a tile instead of all 2 048 padded slots — a 75-key decode
   select ran 66 barrier passes for 49 us. The remaining router lever
   (split-K dots) is parked.
4. TRIED and reverted: the down slot kernel with 16 rows per warp for
   the 160-byte rows (four times the bytes in flight of the same ten
   active lanes, bitwise) — the fabric read T=1 25.3 vs 24.7 ms and MTP
   30 vs 30 ms per pass (`qwen_serve_lever124_*`): at that k the kernel is
   instruction-bound on the per-row reduction, not on bytes in flight.
   LANDED (round 3, bitwise after all): the lane remap
   (`fp8_gemv::block_rows_narrow`, `moe_slot_down_narrow_kernel` for
   routed-only tables with k <= 256 bytes): a warp load covers 32 / c rows
   (three at k = 160, lane l: row l / c, chunk l % c), four such groups in
   flight per warp; each row's 32-lane xor butterfly over c live lanes and
   32 − c zeros collapses — every lane j >= c a subtree reads still holds
   0 at that stage — to the four-stage butterfly inside the group with a
   0.0f wherever the partner is beyond c, so the tree is the SAME tree
   and every row is bitwise block_rows' (`qwen_moe_test`'s decode gate at
   c = 8 and c = 10). GLM's tables have a shared slot and keep the old
   kernel untouched.
5. Round 4 (2026-09-09 late, "the last levers without quantization"):
   (a) the launch interface — the T=1 profile puts 1.62 ms of inter-kernel gaps
   in a step, 0.78 of it the one gap after the commit kernel (the host's
   verdict read, bookkeeping and the next cudaGraphLaunch's ~0.45 us per
   node), the rest ~1 700 sub-microsecond node gaps; the engine already
   pipelines the settle (2026-09-06), so what is left is the launch's
   enqueue itself. TRIED and removed: `cudaGraphUpload` of the other
   parity's exec on a side stream right after each launch — the fabric
   read no gain on Qwen T=1 (22.0–22.1 ms off vs 22.1–22.2 on,
   `qwen_pace_ab_2026-09-09/t1_upload_*`) and GLM slipped (T=1 29.9 →
   30.4, MTP 40.4 → 41.5 ms per step, `glm_regress4_2026-09-09/`): the
   upload contends with the running replay instead of hiding behind it.
   The interface that remains needs the speculative launch of N+1 before N's
   verdict (a device-resident feed exists; a bogus step past a stop must
   be rolled back — the KV blocks, the GDN/PLE/context state, the close
   snapshot — and the bus windows armed ahead): the protocol project the
   2026-09-06 note describes, not this session's. (b) The slot
   gate/up kernel issues both rows' chunk batches before consuming either
   (`fp8_gemv::row_dots_pair`; two latency rounds per warp became one;
   bitwise). The paired form holds 16 chunks live and sits at 96
   registers against the plain form's 56, halving the slot kernel's
   occupancy: one kernel carrying both paths as a branch put GLM at 96
   too and cost it 0.5–1 ms per step (`glm_regress4–6_2026-09-09/`), so
   the two forms are separate instantiations (GLM's back at 29.97 /
   40.40, `glm_regress7`); on Qwen the pair loses 0.15–0.2 ms at one row
   (22.3 vs 22.1) and gains ~0.5 ms per token at two (16.9–17.9 vs
   17.7–18.4, `qwen_pace_ab_2026-09-09/{t1,mtp}_pair_*`), so it serves
   only slot counts beyond one row (`DGPP_MOE_PAIR=0|1` forces either).
   (c) The GDN's four input projections in
   one multi-problem GEMV launch (`launch_bf16_gemv_multi`, up to four
   problems sharing m and k; `bf16_gemv_test` pins every output bitwise
   its single launch; the causal conv follows the merged launch). Not
   taken: the sampling tail (the sampled row's exact select is one block
   per row, 46 + 28 us per step — 0.3 %, a shared kernel whose canonical
   order a multi-block form would have to reproduce).
   Fabric after round 4 (`qwen_serve_round5_{t1,mtp}_2026-09-09/`): T=1
   22.0 ms per step (the serve_bench pace 21.99; 28.9 at the session's
   start), MTP 26–27 ms per pass at 1.54–1.88 tokens per pass, 14.1–17.3
   ms/token; transcripts identical to round 3's in both worlds. GLM at the
   baseline on the final binary (`glm_regress8_2026-09-09/`): T=1 29.96 vs
   29.81, MTP 40.50 vs 40.35 at 72.0 %, prefill 551 / 1 415 / 6 300 vs
   546 / 1 405 / 6 258 ms — the 8 192-token prefill has read 6 269–6 303
   on every rerun since the tile kernel's scale-grid generalization this
   morning, +0.2–0.7 % against the morning's one reading, at the
   resolution of single runs; every generated-ids md5 identical.

   World 2 on the same tree (2 nodes, 87 GiB resident per rank,
   `qwen_serve_w2_round4_{t1,mtp}_2026-09-09/`, `qwen_prefill_w2_2026-09-09/`):
   T=1 32.5 ms per step (37.7 this morning; floor 25.8), MTP 41 ms per
   pass at 1.56–1.98 tokens per pass, 20.8–26.3 ms/token (46–54 per pass
   and 25–34 ms/token this morning); the MTP world's transcripts identical
   to the T=1 world's on all 4 prompts; against the morning's world-2
   transcripts the T=1 run diverges at near ties (the prefill's experts
   now round as the tensor-core kernel, as at world 4); eval GSM8K 59/60,
   HumanEval 38/40 (39/40 at world 4 — one problem at a near tie between
   the worlds' fold orders), extract 30/30; op streams identical on both
   ranks in every run; prefill 612 / 1 508 / 5 817 ms at 528 / 1 981 /
   7 742 tokens (1.16 / 0.76 / 0.75 ms/token; world 4: 0.87 / 0.62 /
   0.58). The fixture's world-2 gates (`qwen_tp_test`, `qwen_engine_test`)
   ran green after every round.

   Where the T=1 step's 22 ms go now (against the 16.0 ms bf16 floor):
   the weight stream at ~220 GB/s with the prefetch windows hiding part
   of it behind the collectives, ~2.8 ms of collective kernels, ~0.8 ms
   of launch interface, the experts' slot kernels, and the small kernels. What
   is left without quantization is the speculative launch of replay N+1
   (the protocol project above) and a multi-block exact select for the
   sampled row; the rest of the distance to the floor is bytes.
   Fabric after round 3 (`qwen_serve_round3_{t1,mtp}_2026-09-09/`, the
   fused GR path at one row, the pipelined norm loads, the lane remap):
   T=1 22.3 ms per step (28.9 at the session's start, 24.7 after round 1,
   22.5 after round 2); MTP 26–27 ms per pass at 1.54–1.88 tokens per
   pass, 14.1–17.3 ms/token (34 → 30 → 31 → 26–27); transcripts identical
   in both worlds throughout; GLM unchanged once more
   (`glm_regress3_2026-09-09/`: 30.01 / 40.35 ms, prefill 548 / 1 404 /
   6 269, identical ids). Against the 16.0 ms bf16 floor the T=1 step
   now carries ~6 ms of non-streaming time: the 99 collectives (~2.8 ms
   of kernel plus their combines), the launch/host gap (~1.5 ms), the
   experts' remaining inefficiency and the small kernels. What is left
   without quantization: graph-launch pipelining of step N+1 behind step
   N's host work (the engine core, both models: −1 to −1.5 ms), the
   experts' gate/up kernel at 82 % of DRAM rate, the GDN's four small
   projections in one dual launch (−48 nodes), the sampling tail (~120
   us). The bus fold is already element-parallel; the skew study's 9 us
   "fold" is the placement-gate hash spin, not a lever.
5. The norms folded into the GEMV's activation staging and the small
   elementwise kernels into their producers: −0.7 ms and ~300 fewer
   nodes (launch cost).
6. Bytes (quality-gated, a decision): FP8 GR gates −0.63 GB (−2.9 ms),
   FP8 GDN projections −0.52 GB (−2.4 ms), FP8 lm head −0.16 GB (−0.7
   ms). With 1–5 landed the step would sit near 22 ms; with 6 near 16 —
   under the bf16 floor, since the floor counts bf16 bytes.
7. Not taken: sliced GR (−4 ms of DRAM for +192 collectives ≈ +10 ms);
   expert parallelism — whole experts distributed over the ranks instead
   of every expert sliced 4-way (the same one reduce per MoE layer, so no
   network saving; full-width rows make the down kernel efficient, but
   the routed experts land unevenly: E[max] over 4 ranks is 4.2 of 10
   draws at one row = 1.68× the balanced bytes, 1.48× at 2 rows (MTP),
   1.23× at 8 rows — a loss of ~1.5 ms per step single-stream, roughly
   even under MTP, a ~20 % MoE win only at 8 concurrent rows; lever 4
   recovers the kernel efficiency without the imbalance); replicated
   attention (+1.9 GB per step); unsharded
   experts (every rank runs all 10 routed experts at full width: +1.77 GB
   per step = +7.4 ms of DRAM against the 48 MoE collectives' −2.7 ms, and
   the full expert set is 120 GB per rank at FP8, 60 at NVFP4 — it does
   not fit beside the 45 GB the rest needs). The network is not the
   bottleneck in bytes (5 KB per collective, ~0.5 MB per step); its cost
   is the latency of 99 round trips, which the prefetch windows overlap.

### Q7 status — the prefill profile (2026-09-09, world 4, 2 048 tokens)

`scripts/fabric_qwen_profile.sh deploy/cluster_qwen-3.8-flash-next_fp8_w4_mtp1.json OUT --prefill 2048`
(`qwen_profile_prefill_2026-09-09/`, the last burst = the calibration
prompt and two 2 048-token prompts, 2.80 s of GPU time in 2.95 s of wall;
the probe read 1 235 ms per 1 978 tokens):

| kernel | share | per launch | what it is |
|---|---:|---:|---|
| `bus_bulk_collective_kernel` + `bus_allreduce_kernel` | 17.6 % | 264 / 115 us | the two folds per layer over [2 048, 2 560] bf16 (10.5 MB): the bulk path near the wire rate — the prefill's network-bound share |
| `moe_grouped_mma_fp8_ldm_kernel` gate/up + down | 24.4 % | 368 / 663 us | the routed experts (every expert touched: 630 MB per layer per rank, 30 GB per prefill = 127 ms of DRAM); the down's k = 160 bytes is five 32-deep stages, so its prologue and epilogue dominate (663 us for the same FLOPs as a 368 us gate) |
| `attn_partial_kernel` | 7.7 % | 1.6 ms | the QSA attention over 2 048 rows (13 launches per prefill) |
| `kda_recurrent_kernel` | 7.0 % | 543 us | the GDN recurrence, token-sequential (36 per prefill) |
| GR sites: `group_rmsnorm` + `mix_finish` + `combine_{dots,apply}` + the low-rank GEMMs | ~14 % | 73–90 us each | elementwise passes over [2 048, 10 240] rows, one kernel each |
| `moe_accum_ordered` + `moe_accum` + `moe_round` | 5.6 % | 239 us | the ordered fp32 accumulate and the shared expert's tail |
| dense GEMMs (nvjet / cutlass) | ~8 % | | the projections through cuBLASLt |
| router (tiled) + select | 3.3 % | 289 us | |
| QSA index score + select | 2.1 % | 205–222 us | |

Toward the Q7 target (≤ 1.5 s at 8 192, 0.18 ms/token against 0.58): the
collectives overlapped with compute (a row-chunk wavefront across layers:
the causal attention and the recurrence never need later rows, so chunk
i's fold can run beside chunk i+1's layer — −13 %), a down kernel shaped
for k = 160 (−8 %), the GR sites' elementwise passes fused into one
kernel per site (−8 %), the accumulate fused into the down epilogue
(−4 %); with all four the prefill would sit near 0.38 ms/token, 3.1 s at
8 192 — the 1.5 s target needs FP8 dense projections and GR gates on top.

## 6. Risks and open questions

- **GR traffic.** Replicated GR is 21 % of the TP=2 stream and 33 % of the
  TP=4 stream and would be over half at TP=8; the sliced variant costs 97
  collectives per token, measured at 36–60 µs each (D3), so replication is
  the default at both worlds and the headline number at wide worlds depends
  on FP8 GR, a quality decision.
- **TP=2 memory margin.** 86.6 GiB resident per rank is 4.6 GiB above
  GLM's working point; the memory plan sets the KV capacity, and the MTP
  layer (1.2 GiB per rank at TP=2) is the only optional class.
- **Expert slice alignment (D2)** touches every FP8 expert kernel; the GLM
  FP8 path must stay bitwise (its slices remain 128-aligned, so the grid
  parameter is 128 there). The grid is 64 at TP=2 and 32 at TP=4, so both
  worlds exercise the new path.
- **Top-k tie order** cannot match torch exactly; the oracle and the engine
  share one rule, and parity tests tolerate selection differences only when
  scores tie.
- **MTP hidden-state normalization** uses the implemented convention in
  `src/kernels/qwen_mtp.cu`; the port record documents its validation.
- **NFC normalization** is implemented with composition tables. Keep
  non-NFC inputs in the tokenizer goldens when updating those tables.
- **QSA determinism across ranks**: the replicated indexer must produce
  identical selections on every rank or the boundary reduces diverge; a
  per-tick digest of the selection (as the op-stream md5 does) is the check.
- **Long-context indexer cost** grows linearly (200 MB per step at 262 K);
  the compressed cache in fp8 is the lever if it shows in the step.
- **The bus's single-outstanding collective** rules out overlapping the PLE
  reduce with layer 0; D4's fusion avoids the question, but the reducer must
  accept a longer element count at that one boundary.
- **The vision-token path** (`<|vision_start|>` etc.) is refused, not
  ignored: a request containing images gets an error.
- **Checkpoint revisions** must pass the binding and loader checks before
  serving; the recorded audit applies to the revision tested.

## 7. Checkpoint validation checklist

Repeat these checks when changing checkpoint revisions. The original
port results are recorded in section 5.

1. The stored `layer_multipliers`, `ngram_heads_vocab_sizes`,
   `ngram_heads_offsets` equal §1.7's derivation; `weight_scale`'s value.
2. Every expert `weight_scale_inv` grid is `[5, 20]` / `[20, 5]` BF16 and the
   MTP experts match.
3. The GDN `in_proj_qkv` row order is q|k|v head-major (SGLang's comment
   says the checkpoint uses the Qwen3.5 head-first layout).
4. `q_proj` interleaves q and gate per head as `view(-1, 512).chunk(2)`
   implies.
5. The 128 n-gram shards cover exactly 320 001 536 rows in order.
