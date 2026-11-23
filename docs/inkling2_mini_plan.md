# Inkling 2 Mini (thinkingmachines/Inkling-2-Mini) on dgpp: architecture facts and implementation plan

Status: the port record is section 7. The checkpoint
`thinkingmachines/Inkling-2-Mini` (revision `b85c3cfa`) is one safetensors
file of 1,244 tensors and 18.18 GiB of payload plus a 606 MiB draft model
(`mtp.safetensors`, 54 tensors). The architecture study uses its
`config.json`, `hf_quant_config.json`, the safetensors headers, the
transformers 5.24.0 implementation (`models/inkling2_loop`, the plain
PyTorch statement of the math) and the vLLM implementation at `8c17e469`
(`vllm/models/inkling2`, the runtime that states the released numerics:
which values are bf16, where each rounding sits, how the experts'
activations are quantized, how the draft model runs).

## 0. Summary

Inkling 2 Mini (`Inkling2LoopForConditionalGeneration`, `model_type
inkling2_loop`) is a looped transformer: 24 decoder layers of weights, of
which the middle twelve (6 to 17) run three times, so a token passes
through 48 layer visits. It has 26.4 B parameters, 128 experts per layer
with 8 routed per token, hidden size 2304, a vocabulary of 200,058 tokens
(the tables are padded to 201,024 rows) and a positional ceiling of
1,048,576 tokens. It accepts text and images.

Five things set it apart from the families the engine already serves.

1. **No residual stream.** Layers are grouped in blocks of six. Inside a
   block the layers' outputs are summed into a partial state. The input of
   every sublayer (attention query, key and value, and the MLP) is a learned
   softmax mix over the completed blocks and the running partial state, each
   RMS-normalized. The model output is one more such mix.
2. **A loop.** After the first six layers the model saves its state. Each
   of the three passes over layers 6 to 17 restarts from that saved state.
   Passes two and three also receive four messages per layer (for query,
   key, value and MLP) read from the previous pass's states and blended in
   by a rank-64 gated controller. Every pass keeps its own K/V cache and
   convolution state.
3. **Attention without rotary embeddings.** Position enters through a
   learned relative bias over the nearest 1,024 keys and through short
   causal convolutions (four taps) on the keys, the values, the attention
   output and the MLP output. Queries and keys are L2-clipped, values are
   RMS-normalized, each query carries its own sink logit and the attention
   output passes through a sigmoid gate before the output projection.
   Twenty layers attend over a sliding window of 1,024 keys and four over
   the whole context.
4. **Quantized experts with quantized inputs.** Layers 0 to 19 store their
   experts in NVFP4 and layers 20 to 23 in MXFP8. The checkpoint's precision
   metadata requires the expert inputs to be quantized the same way (NVFP4
   per token with the four-over-six block scale choice, MXFP8 per 32
   values). Everything else is BF16 or FP32.
5. **A block drafter.** `mtp.safetensors` is a three-layer DSpark draft
   model that proposes seven tokens per step from four hidden states of the
   target model.

## 1. The model

### 1.1 Configuration facts

| field | value |
|---|---|
| `hidden_size` / `num_hidden_layers` | 2304 / 24 |
| `vocab_size` / `unpadded_vocab_size` | 201,024 rows stored / 200,058 used |
| `model_max_length` | 1,048,576 |
| attention heads | 24 query heads and 8 key/value heads of width 128, the same for both layer kinds |
| `local_layer_ids` | every layer except 5, 11, 17 and 23 (the last layer of each block is the full-attention one) |
| `sliding_window_size` / `rel_extent` | 1024 / 1024 (a query sees keys at distances 0 to 1023 in a local layer) |
| `d_rel` | 16 relative-state dimensions per query head |
| `qk_l2_clip_threshold` / `rel_l2_clip_threshold` | 22.627416997969522 (√512) / 8.0 |
| `log_scaling_n_floor` / `log_scaling_alpha` | 8192 / 0.1 (full-attention layers only) |
| `use_value_norm_no_gamma` | true (values are RMS-normalized per head without a weight) |
| `sconv_kernel_size` | 4 |
| `n_layers_per_attn_res_block` / `attn_res_normalize_values` | 6 / true |
| MoE | 128 experts of width 1152, top 8, sigmoid scores, `route_scale` 8.0, a selection bias (`use_gate_bias`), no shared expert |
| `mlp_gate_softcap` / `mlp_up_softcap` | 4.0 / 25.0 |
| `final_logit_softcapping` | 30.0 |
| `loop_config` | `core_start` 6, `core_end` 18, `num_passes` 3, `gate_rank` 64, `is_varloop` false |
| `rms_norm_eps` | 1e-6 |
| tokens | eos 200006 (`<|content_model_end_sampling|>`), pad 199999, image placeholder 200054 |
| `mtp_config` | 3 layers, layers 0 and 2 local (window 1024), dense MLP 9216 wide, RoPE θ 1e6, `block_size` 8, `markov_rank` 256, `markov_head_type` `rnn_hidden`, `target_layer_ids` [5, 23, 41, 48] |
| `vision_config` | 23 layers, width 1024, 16 heads, patch 20, temporal patch 2, a 2 × 2 merge and a projector of width 4096 |

A layer visit is one execution of a layer. Visits are numbered in execution
order: layers 0 to 5 are visits 0 to 5, pass p of layer l in the core is
visit l + 12 p, and layers 18 to 23 are visits 42 to 47. A core layer
therefore has visits l, l + 12 and l + 24.

### 1.2 Blocks and attention residuals

The model keeps a stack of completed block states and one partial state,
all BF16 rows of width 2304. The embedding (after its RMSNorm) is the first
partial state. A mix of the stack with a query vector q (FP32, 2304) is, for
each token,

```
x_n   = stack[n] for the completed blocks, then the partial state
s_n   = (q · x_n) / sqrt(2304) · rsqrt(mean(x_n²) + 1e-6)
p     = softmax(s)
mix   = Σ_n p_n · x_n · rsqrt(mean(x_n²) + 1e-6)          (FP32)
```

The mix feeds the next sublayer's RMSNorm without an intermediate rounding
and the normed result is rounded to BF16.

One layer visit of layer l:

```
attention inputs
  l = 0:        the partial state through the layer's norm (no mix: nothing to mix yet)
  local layer:  one mix with attn_queries[l-1], one norm (attn_norm), used for q, k and v
  full layer:   three mixes with the rows of attn_queries[l-1] [3, 2304],
                each through its own norm (q_input_norm, k_input_norm, v_input_norm)
a  = attention(q-input, k-input, v-input)                  BF16
a  = short_conv(a, attn_sconv)                             BF16
if l is the first layer of a block (l % 6 == 0):
    the partial state is pushed onto the stack and the new partial starts from a
    m_in = mix([stack], a) with mlp_queries[l]
else:
    m_in = mix([stack], partial + a) with mlp_queries[l]    (the sum in FP32)
m  = short_conv(moe(norm(m_in, mlp_norm)), mlp_sconv)      BF16
partial = bf16(partial + a + m)      (or bf16(a + m) on the first layer of a block)
```

The model output is `norm(mix([stack], partial) with output_query)` through
the final RMSNorm, then the unembedding in BF16 with an FP32 result, then
`30 · tanh(logit / 30)` on the 200,058 real rows.

### 1.3 The loop

Visits 0 to 5 are the prelude. At the entry of layer 6 the model saves the
number of completed blocks (one) and the partial state. Each pass over
layers 6 to 17 starts from that saved state, completes the block of layers
6 to 11 (pushing the prelude's partial as block 1) and runs layers 12 to 17
into a partial. Passes two and three read the previous pass: its bank is
the block completed at layer 12 and its final partial, two rows per token.
For each core layer the four message queries (attention, key, value, MLP)
mix the bank with normalized values into four FP32 messages. A message
blends into its sublayer input before that input's RMSNorm:

```
pair = [rms(dest), rms(message)]                           BF16, 4608 wide (eps 1e-6, no weight)
h    = pair @ down^T                                       BF16, 64 wide
out  = (1 - tanh(h @ forget^T)) · dest + tanh(h @ gain^T) · message     FP32
```

`dest` is the FP32 mix the sublayer would have used. A local layer uses its
one attention mix as the destination of all three attention messages, so in
passes two and three its key and value inputs differ from its query input
and the layer projects them separately, as a full layer always does. After
the third pass the coda (layers 18 to 23) continues from that pass's state.
The checkpoint stores the message queries as two [24, 2304] tensors:
`middle_cycle_cross_pass_queries` holds (attention, MLP) per core layer and
`middle_cycle_cross_pass_kv_queries` holds (key, value).

### 1.4 Attention

```
q, r, gate, sink  = x_q @ [Wq | Wr | Wgate | Wsink]^T      3072 + 384 + 3072 + 24 outputs, BF16
k                 = x_k @ Wk^T                             1024, BF16
v                 = x_v @ Wv^T                             1024, BF16
q  = l2_clip(q · q_scale, √512) per head                   BF16, and full layers then multiply by tau and round again
r  = l2_clip(r · r_scale, 8) per head (16 wide)            BF16, and full layers then multiply by tau and round again
tau = 1 + 0.1 · ln(max((pos + 1) / 8192, 1))               FP32, per token
bias[h, d] = r[h] · rel_proj[:, d]   for d in [0, 1024)    0 for farther keys
k  = l2_clip(short_conv(k), √512) per head                 BF16, cached
v  = rms(short_conv(v)) per head, eps 1e-6, no weight      BF16, cached
logits = q · k / sqrt(128) + bias[h, pos_q - pos_k]        keys with pos_k ≤ pos_q, and local layers keep distance < 1024
out = softmax over [logits, sink_h] · V, the sink column dropped
out = out · sigmoid(gate)                                  BF16
attention = out @ Wo^T                                     BF16
```

`l2_clip(x, t)` scales a head vector by `min(1, t / (|x| + 1e-6))`. The sink
is one logit per query and head that takes softmax mass and contributes no
value, which equals multiplying the sink-free attention output by
`sigmoid(lse - sink)` with `lse` the log-sum-exp of the key logits.

### 1.5 Short convolutions

Four streams per layer visit carry a causal depthwise convolution with four
taps over the token sequence: the key projection (1024 wide), the value
projection (1024), the attention output (2304) and the MLP output (2304).
All inputs are BF16 and positions before the start of the sequence read as
zero.

```
static (every stream except full-layer K and V), weight [C, 1, 4], oldest tap first:
    y[t] = (Σ_k w[c, k] · x[t - 3 + k] + x[t]) / sqrt(5)
dynamic (K and V of the full-attention layers), newest tap first:
    taps[t, head, lag] = static_w[lag, c] + (x_in[t] @ weight_proj^T)[head · 4 + lag]
    y[t] = (Σ_lag taps[t, head(c), lag] · x[t - lag] + x[t]) / 3
```

`x_in` is the normed input the key or value was projected from. The result
is rounded to BF16. A request therefore carries, per visit, the last three
rows of each stream's convolution input.

### 1.6 The MoE

```
logits  = fp32(x) @ fp32(gate.weight)^T                    128 scores
scores  = sigmoid(logits)
picked  = top 8 of scores + gate.bias, ties to the lower expert id
weights = scores[picked] / (Σ scores[picked] + 1e-20) · 8.0
per picked expert e:
    g, u = x @ w13[e]^T                                    rows interleaved (gate, up), 1152 each
    act  = 4·tanh(g/4) · sigmoid(g) · 25·tanh(u/25) · weight_e      FP32, rounded to BF16
    y_e  = act @ w2[e]^T
out = bf16(Σ_e y_e)                                        FP32 accumulation
```

The released recipe quantizes both GEMM inputs. For the NVFP4 layers a row
is scaled by `amax / 1536` and each 16-value block takes whichever of the
6-based and 4-based E4M3 block scales gives the smaller absolute
reconstruction error (`fouroversix_mae`). For the MXFP8 layers each 32-value
block takes a power-of-two scale that covers its maximum and E4M3 values.
The NVFP4 weights are E2M1 pairs with the even element in the low nibble,
one E4M3 scale per 16 values along the input dimension, and a per-expert
factor `amax / outer_scale_denominator`. The MXFP8 weights are E4M3 values
with one scale exponent per 32 values (`2^(s - 127)`).

### 1.7 The draft model

`mtp.safetensors` holds a DSpark drafter: three BF16 layers (attention with
24 heads over 8 key/value heads, per-head q and k RMSNorm with weights, RoPE
θ 1e6, a dense SwiGLU MLP 9216 wide), a final norm, a mask embedding and a
recurrent Markov head. It reads four hidden states of the target per
committed token: the partial states at the entries of visits 5, 23 and 41
and the final-norm output. For each draft layer a row of
`softmax(fuse_weight)` mixes the four states, `dflare_norm` normalizes the
mix and `k_proj_ctx` / `v_proj_ctx` turn it into context keys and values at
the token's position.

A draft block is eight rows at the next eight positions: the target's normed
embedding of the newest token (the one the last target row chose, not yet
run through the target), then seven copies of the learned mask row. Every
row of a block attends over the same keys, the block's own included. Layer
1 sees the whole context. Layers 0 and 2 see the last 1,024 positions
counted back from the block's last row, so 1,016 context positions and the
block. Each mask row proposes the token at its own position.

The Markov head corrects the mask rows' base logits (the target's
unembedding of each row's final state) one proposal after another. Its
rank-256 state starts at zero. A step reads the state, the previous token's
`markov_w1` row (the newest token for the first step, the step's own
proposal after that), the mask row's final state and one feature of the last
committed position, which is the four tapped states mixed by the last row of
`fuse_weight`. The step's latent goes through `hidden_out` and the
unembedding as a bias on the row's logits before the softcap. The two
projections are folded into one matrix of 256 values per token at load.
Interior arithmetic is FP32 and every value is rounded to BF16 where the
released kernels round it.

### 1.8 The vision tower

The image processor does not resize. It composites transparency onto a gray
checkerboard, pads the image on the right and bottom to a multiple of 40
pixels with the value -1 on the 0 to 255 scale, scales every pixel to
`x / 127.5 - 1` and cuts the result into 40 x 40 cells. Each cell becomes one
`<|unused_200054|>` placeholder inside a `<|content_image|>` block, in raster
order, so an image of `h x w` pixels costs `ceil(h / 40) * ceil(w / 40)`
tokens.

The tower is a ViT with 23 pre-norm layers (width 1024, 16 heads of 64, a
GELU MLP 4096 wide, biases on every projection, a learned scale per sublayer
output). A cell is four 20 x 20 patches. Each patch goes through a linear
embedding over its 1,200 values, without bias, and takes a position embedding
interpolated bilinearly from a learned 32 x 32 grid to the image's patch
grid. A class token leads the sequence. After a layer norm the layers run
with full attention inside the image and an axial rotary embedding: half of
each head rotates by the patch's column, half by its row, both counted from
one, with the class token at zero. No norm follows the last layer. The four
patch states of a cell are concatenated and projected through 4096 (GELU)
to 2304, and the result replaces the placeholder's normed embedding.

### 1.9 Tensor census

Checkpoint names (`model.llm.` prefix dropped) and shapes. The BF16 total is
3.59 GiB, the E4M3 total 4.98 GiB, the U8 total 9.61 GiB.

| tensor | dtype | shape | count |
|---|---|---|---|
| `embed.weight`, `unembed.weight` | BF16 | [201024, 2304] | 1 each |
| `embed_norm.weight`, `norm.weight` | BF16 | [2304] | 1 each |
| `attn_res.attn_queries.{0..22}` | F32 | [2304], or [3, 2304] for the full layers (indices 4, 10, 16, 22) | 23 (index i belongs to layer i + 1) |
| `attn_res.mlp_queries.{0..23}`, `attn_res.output_query` | F32 | [2304] | 24, 1 |
| `middle_cycle_cross_pass_queries`, `…_kv_queries` | F32 | [24, 2304] | 1 each |
| `layers.N.attn.wq_du`, `gate_attn` | BF16 | [3072, 2304] | 24 each |
| `layers.N.attn.wk_dv`, `wv_dv` | BF16 | [1024, 2304] | 24 each |
| `layers.N.attn.wr_du` | BF16 | [384, 2304] | 24 |
| `layers.N.attn.attn_sink_proj` | BF16 | [24, 2304] | 24 |
| `layers.N.attn.wo_ud` | BF16 | [2304, 3072] | 24 |
| `layers.N.attn.q_l2_clip.scale`, `rel_l2_clip.scale` | BF16 | [24, 128], [24, 16] | 24 each |
| `layers.N.attn.rel_logits_proj.proj` | BF16 | [16, 1024] | 24 |
| `layers.N.attn.{k,v}_sconv.weight` | BF16 | [1024, 1, 4] | 20 each (local layers) |
| `layers.N.attn.{k,v}_sconv.static_w`, `weight_proj` | BF16 | [4, 1024], [32, 2304] | 4 each (full layers) |
| `layers.N.attn_norm` or `{q,k,v}_input_norm`, `mlp_norm` | BF16 | [2304] | 20 or 4 × 3, 24 |
| `layers.N.attn_sconv.weight`, `mlp_sconv.weight` | BF16 | [2304, 1, 4] | 24 each |
| `layers.N.mlp.gate.weight`, `gate.bias` | BF16, F32 | [128, 2304], [128] | 24 each |
| `layers.N.mlp.experts.w13_weight` (NVFP4) | U8 | [128, 2304, 1152] | 20 |
| `…w13_weight_scale` | F8_E4M3 | [128, 2304, 144] | 20 |
| `…w2_weight` (NVFP4) | U8 | [128, 2304, 576] | 20 |
| `…w2_weight_scale` | F8_E4M3 | [128, 2304, 72] | 20 |
| `…{w13,w2}_weight_amax`, `…_outer_scale_denominator` | F32 | [128], [1] | 20 each |
| `…w13_weight`, `w2_weight` (MXFP8, layers 20 to 23) | F8_E4M3 | [128, 2304, 2304], [128, 2304, 1152] | 4 each |
| `…w13_weight_scale`, `w2_weight_scale` (MXFP8) | U8 | [128, 2304, 72], [128, 2304, 36] | 4 each |
| `layers.N.controllers.{attn,k,v,mlp}.down_weight` | BF16 | [64, 4608] | 12 each (layers 6 to 17) |
| `…forget_weight`, `gain_weight` | BF16 | [2304, 64] | 12 each per stream |
| `visual.*` | BF16 | the tower, 0.56 GiB | 422 |

Per layer the experts take 580 MiB in NVFP4 (w13 324 + 36, w2 162 + 18) and
1,107 MiB in MXFP8. The dense attention matrices take 59.7 MiB per layer.

The text model is 822 tensors and the vision tower 422.

## 2. Placement by world size W ∈ {1, 2, 4}

The key/value heads (8) and the query heads (24) divide by W. Rank r holds
query heads [24r/W, +24/W) with their key/value heads [8r/W, +8/W).

| piece | placement |
|---|---|
| `wq_du`, `wr_du`, `gate_attn`, `attn_sink_proj` | the rank's query heads' rows, stacked as one BF16 matrix [q \| r \| gate \| sink] that reads the query input |
| `wk_dv`, `wv_dv` (+ the dynamic taps' `weight_proj` rows on full layers) | the rank's key/value heads' rows, one matrix each |
| `wo_ud` | the rank's query heads' columns. The partial output is all-reduced |
| clip scales, K/V convolution taps | the rank's heads' slices |
| `rel_logits_proj`, the hidden-stream convolution taps, norms, mix queries, the router, the controllers | replicated |
| experts | every rank holds every expert on the same slice of the intermediate width: `w13` rows [2·1152r/W, +2·1152/W) (gate and up interleaved) and `w2` columns [1152r/W, +1152/W). The partial output is all-reduced |
| embedding | replicated, the 200,058 real rows |
| unembedding | vocabulary-sharded over the 200,058 real rows |

Slicing every expert instead of assigning whole experts to ranks keeps a
rank's bytes per token the same for every routing, so no rank waits on the
busiest one. The slices are whole 32-value scale blocks at every world
(1152, 576, 288). A layer visit costs two collectives (the attention output
and the MLP output), 96 per token. Everything computed after a collective
(the convolutions, the mixes, the controllers) is replicated and bitwise
equal across ranks.

| per rank | W = 1 | W = 2 | W = 4 |
|---|---|---|---|
| experts | 15.2 GiB | 7.6 GiB | 3.8 GiB |
| BF16 attention | 1.40 GiB | 0.70 GiB | 0.35 GiB |
| controllers, router, small tensors | 0.07 GiB | 0.07 GiB | 0.07 GiB |
| embedding + unembedding | 1.72 GiB | 1.29 GiB | 1.07 GiB |
| K/V per token, 48 visits (BF16 / FP8 rows) | 192 / 99 KiB | 96 / 50 KiB | 48 / 25 KiB |

A single decode row streams, per visit, the three attention matrices (39.6
MiB as BF16), the output projection (13.5 MiB) and eight experts (34.2 MiB
in NVFP4, 62.6 MiB in MXFP8), and on the 24 visits of passes two and three
the four controllers (4.5 MiB). Over 48 visits with the unembedding that is
about 4.7 GiB at W = 1 before the 12-bit BF16 companions and about 1.3 GiB a
rank at W = 4.

## 3. Design decisions

* **D1. The release's formats as shipped.** NVFP4 and MXFP8 experts are
  loaded verbatim, every BF16 and FP32 tensor verbatim. The per-expert
  NVFP4 factor `amax / denominator` is stored as its reciprocal, the divisor
  the shared NVFP4 kernels apply to a finished dot.
* **D2. Both activation recipes, on one key.** `engine.activations` selects
  `bf16` (BF16 inputs to both expert GEMMs, the arithmetic of the
  transformers model) or `native` (the checkpoint's required quantized
  inputs). The templates select `native`. The BF16 recipe is the path
  compared against the transformers model on the real checkpoint.
* **D3. A dedicated MoE layer over the shared grouped kernels.** The router
  is the shared sigmoid-with-bias router with a scaling factor of 8. The
  expert chain is device segmentation, one grouped GEMM over the interleaved
  gate/up matrix, the softcapped activation with the routing weight inside
  it, one grouped GEMM for the down projection and the ordered accumulation.
  One recipe holds on every path, the released one: an expert's gate/up and
  down outputs are rounded to BF16 at their row scale, and a token's rounded
  expert outputs are summed in ascending expert id and rounded once. Decode
  rows carry FP32 dots to the point of rounding. Reader rows take BF16
  outputs from the GEMM, so the intermediates cross memory at half the
  width. Under the native recipe the NVFP4 experts of reader rows run on the
  block-scaled tensor-core instruction with both operands in four bits, fed
  by this family's own quantizer. That instruction steps 64 values of k, so
  the down projection of a four-rank world (a 288-wide slice) stays on the
  BF16 tensor-core form over the exact dequantized rows.
* **D4. New attention kernels beside the sink family.** The prologue (q and
  r scaling and clipping, the bias table, the K/V convolutions with their
  clip and norm, the cache append), the split-KV attention with the additive
  bias and a per-row sink, and the combine with the output gate are this
  family's own source. The MiMo and Flint kernels are untouched. Reader rows
  take a query-tiled tensor-core form of the same attention: five rows of
  three query heads per warp tile, the bias added after the scale, the sink
  as a column without a value and the gate in the epilogue.
* **D5. Full history in the paged K/V pool.** Every visit owns a plane of
  the shared pool and a local layer's window is applied at read time, as for
  MiMo and Flint. A prefix is restored for every visit by one block list. A
  windowed store for the 40 local visits would cut the pool by 83 percent
  and is the recorded memory lever.
* **D6. Convolution inputs in a positional ring.** A request keeps, per
  visit and stream, the convolution inputs of its last 16 positions at slot
  `position mod 16`. A rejected speculative row's slot is rewritten before
  anything reads it, so rollback copies nothing. A prefix snapshot holds the
  three rows before its position (1.9 MiB at W = 1). The family does not
  attach inside a snapshot because those rows exist only at snapshot
  positions.
* **D7. The mix feeds the norm in one kernel.** A mix, its sublayer's
  RMSNorm and the rounding are one launch per row and query. Message
  blends take the FP32 mix, run the controller and normalize in a second
  kernel.
* **D8. The softcap on the device logits.** The unembedding reads the
  200,058 real rows and `30·tanh(logit / 30)` is applied in place before any
  pick, so greedy and sampled decoding both see the released distribution.
* **D9. The drafter on the shared block-draft executor.** The DSpark draft
  model runs on `DFlash2Drafter` with this family's block body (scalar tap
  mixes, context projections, a learned mask row, per-layer windows) and its
  recurrent head. A run copies the four tapped states, the drafter turns
  them into one more K/V plane per draft layer in the shared pool, and the
  feature the recurrent head reads sits in each position's ring slot, so
  rollback and prefix snapshots need nothing of their own. The draft's
  projections run on a GEMM object of their own in the tensor-core form: a
  matrix is read once for all the rows of a pass, and a row's sums do not
  depend on the slots stacked beside it.
* **D9a. The recurrent head stays exact and stops early.** Each of its
  steps biases the whole vocabulary: one tensor-core GEMV over the folded
  head (`unembed @ hidden_out`, 102 MiB) and one elementwise launch. A
  shortlist taken from the uncorrected logits (the shared head screen) cut
  the pass by 16 ms and the accepted tokens per step on code from 4.0 to
  2.3, so it is not read. The head keeps the schedule's running survival as
  it drafts and skips its remaining steps once that is a quarter below the
  schedule's threshold, where no later proposal can be verified: the engine
  publishes the threshold before each pass, and the skipped steps' launches
  read a flag and return. The step's own projection runs one block per output
  and slot, which took it from 1.26 ms to 0.05 ms a pass.
* **D9b. The acceptance estimate.** The draft has no confidence output. The
  schedule's estimate is the logit of a proposal's softmax mass among its
  step's sixteen candidates, through a linear fit (bias -0.64, slope 0.568)
  regressed on 2,443 verified proposals of five chat prompts. Unfitted, a
  proposal the draft gave 94 percent was accepted 73 percent of the time and
  the schedule verified rows that did not pay.
* **D9c. Wide verify passes read a matrix once.** The packed 12-bit launch
  takes eight rows, so a pass of several requests' verify rows would read
  every attention matrix and the head once per eight rows. With the BF16
  bytes resident (`engine.bf16_weights` `bf12+bf16`, what the templates set)
  decode passes of nine rows or more take the tensor-core form over them.
  The controllers' rank-wide product stays on cuBLASLt, where the form's
  launch costs more than the matrix. The head alone takes the tensor-core
  form over its 12-bit bytes, from nine to 32 rows: 2.9 ms against 4.0. The
  same form on the attention matrices ran about a sixth slower than their
  BF16 launches, and at 64 rows 7.6 ms against 4.4, so they keep BF16.
* **D9e. A verify row is the plain step.** One request's pass holds up to
  eight rows, the pending token and seven proposals, and each row must round
  as the plain step at its position does, or a draft would change what the
  model writes. Every BF16 product of up to eight rows therefore takes the
  row-independent GEMV chain: the packed matrices through their own
  launches, and the one matrix without a 12-bit form, the controllers' gain
  product, through the BF16 GEMV. With the family default of four rows the
  gain product fell to a batched algorithm from five rows on, and a pass of
  five to eight rows forked from plain decoding at near ties. Passes of
  several requests, nine rows and more, take the tensor-core forms and are
  tolerance-equal to a request decoding alone.
* **D9d. A step may verify nothing, and a batch has one depth.** The
  templates set `engine.mtp_schedule_min_depth` to 0, so a step whose first
  proposal is not worth its row feeds the pending token alone. A batch
  verifies one depth for every request, from their mean estimate. Giving
  each request its own depth inside the batch, with the rows past it as
  padding, was built and measured: at eight requests on one node it lost
  five percent, because the estimate does not separate requests well enough
  to pay for the wider pass.
* **D9f. Padding rows read no experts.** A row batch has a fixed slot
  count, so with five or seven requests decoding, the six- and eight-slot
  batches carry a closed slot whose rows are padding. A padding row used to
  be routed like any other, and the grouped launches read the experts it
  selected. The segment builder now takes the rows' positions and leaves a
  padding row in no expert's row list, and its expert output is zero. Plain
  decoding gained 10 to 13 percent at five and seven requests on one node,
  and decoding with the draft 3 to 7 percent.
* **D10. Prefill in chunks of 4,096.** A chunk reads every routed expert's
  weights once, so a larger chunk gives each expert more rows per read.
  4,096 takes the gain that 8,192 does on the measured prompts.
* **D11. A typed-block dialect for replies.** The model writes
  `[<|message_model|>NAME]<|content_KIND|>BODY<|end_message|>` blocks whose
  markers are single special ids. The parser maps thinking blocks to
  reasoning, text blocks to content and a tool block's JSON body to a call.
  The constrained-decoding grammar is one state machine over the same
  blocks for the tool modes and the JSON modes, with a call's body run by
  the JSON schema machine under a schema built from the tool. The template
  maps every reasoning effort name itself, so the service passes the name
  through.

* **D12. Cache cuts at turn ends.** The template writes no generation
  prompt, so a prompt has no role marker at its end, and an answer's
  snapshot stops at the end-of-sampling token, one short of the next
  speaker's marker. The front end declares the marker the model opens its
  turn with, and the service cuts one token before every change of speaker
  and at a prompt's last token. Markers that continue one speaker cut at
  themselves. The family does not attach inside an entry (D6), so these
  cuts are what a repeated prompt and the next turn attach to.
* **D13. The vision tower replicated on every rank.** The tower runs only
  inside an image prefill, on the pixels the admission journal carries, so
  every rank computes the same rows and nothing is broadcast. Its rows take
  the place of the placeholder tokens' normed embeddings before the first
  visit. The released processor's padding value is applied in the pixel
  kernel from the image's content region, so the canvas bytes outside it
  never reach the tower, and the region is part of the prefix cache's image
  identity. The attention keeps scores and probabilities in FP32 as the
  released kernel does.
* **D14. The template renders the image block.** The shared vision front
  end replaces image parts with delimiter text. This checkpoint's template
  writes `<|content_image|>`, one placeholder and `<|end_message|>` for an
  image part by itself, so its front end hands the parts to the template
  and expands each placeholder to the image's token count.

## 4. Layout

```
src/models/inkling2/config.{hpp,cpp}     the parsed contract (config.json and hf_quant_config.json)
src/models/inkling2/binding.{hpp,cpp}    the expected-tensor tables, the TP geometry check
src/models/inkling2/loader.{hpp,cpp}     Inkling2LayerStream on ResidentLayerStream
src/models/inkling2/layers.{hpp,cpp}     the attention and MoE layer objects
src/models/inkling2/forward.{hpp,cpp}    Inkling2Model on SessionModel: the visit walk, the memory plan
src/kernels/inkling2_*.{hpp,cu}          mixes, convolutions, attention, the expert activation and quantizers
tests/cuda/inkling2_fixture.hpp          the tiny release
tests/cuda/inkling2_*_test.*             the ladder
tests/unit/inkling2_{config,binding}_test.cpp
tools/inkling2_reference_dump.py         the numpy reference over the fixture
apps/inkling2_{load,forward}_check.cpp   the real-checkpoint checks
deploy/cluster_inkling-2-mini_nvfp4-mxfp8_w{1,2,4}.example.json
```

## 5. Stages and validation

G0 config and binding (unit tests over the real config) · G1 the loader
over the fixture at worlds 1, 2 and 4 · G2 the kernels against host
references · G3 the forward against the numpy reference in both activation
recipes, and the numpy reference against the transformers model · G4 the
session surface (decode, snapshots, the shared prefill contract) · G5
loopback worlds 2 and 4 and the graph engine · G6 the tokenizer, the chat
template and the typed-message parser · G7 the real checkpoint and the
fabric worlds · G8 the drafter · G9 images · G10 profiling and optimization
· G11 the benchmark campaign.

## 6. Open levers

In the order of what they would return.

1. **A pass that does not draft.** A verify row reads its own experts, so
   for prose at eight requests on one node a row costs almost what its
   accepted proposals return, and the draft's own block is about 9 ms of
   that pass. A pass that neither verifies nor drafts, taken when the
   estimated return is below the draft's cost, would hold those passes at
   the plain rate. The decision needs measured pass costs by request count:
   a linear model fitted at one request chose plain passes for code at eight
   requests, where drafting is over 20 percent ahead.
2. **The draft's weights in FP8.** The draft's three layers read 525 MiB of
   BF16 a block, about 2.6 ms. `engine.dflash_weights` `fp8` halves that
   read on the Qwen drafters. It changes the proposals, never the output,
   and needs its own acceptance measurement on this model.
3. **A tensor-core form that streams 12-bit weights at 64 rows.** The
   register form is tuned for a head at up to 32 rows. Eight requests' draft
   blocks are 64 rows and read the BF16 head, 4.4 ms.
4. **A windowed K/V store for the local visits.** Forty of the 48 visits
   read only the last 1,024 positions. Keeping full history for them is 83
   percent of the pool. A store that releases blocks behind the window would
   let every world seat the model's full context.
5. **The four-rank down projection in four bits.** Padding the 288-wide
   slice to 320 would put it on the block-scaled instruction, about a
   seventh of a four-node prefill.
6. **Folds and replicated row work at four ranks.** A four-node prefill
   spends about a quarter in the 96 folds a chunk and another quarter in the
   mixes, norms and controllers every rank repeats.
7. **Small decode kernels.** At four ranks the mixes, the controllers and
   the per-visit expert bookkeeping are over a third of a decode step. A
   controller fused into one kernel is bound by single-thread reads of its
   matrices, so the gain needs a wide read path inside the fused kernel.
8. **Attach inside an entry.** The convolution rows exist only where a
   snapshot was taken, so an edited prompt reuses a cached entry only up to
   a cut. Replaying about 290 positions ahead of any position against the
   cached keys and values rebuilds those rows exactly, six positions a
   visit, which would let such a prompt open on everything it shares.
9. **Decode at long context.** The eight full-attention visits read every
   cached key and value each step, at close to the memory line rate, so a
   step grows from 14 ms to 39 ms between an empty context and 512K on four
   nodes. An fp8 form of that cache would halve the read. It trades
   precision and needs its own quality measurement first.

## 7. Status

G0 to G10 are done: the forward, the session surface, worlds of 2 and 4, both
engines, the text interface, the draft model, images and serving on one, two
and four nodes. G11 is recorded in [benchmarks](benchmarks.md).
