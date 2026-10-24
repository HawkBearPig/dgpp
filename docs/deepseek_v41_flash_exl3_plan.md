# DeepSeek-V4.1-Flash-1017 in EXL3 3.5 bpw — port plan and record (2026-10-22)

Status: in progress. The checkpoint census, the format contract and the
numerics contract below come from the repository's headers (44 shards read
over HTTP range requests, 192,495 tensors, 239.95 GiB), its `config.json`,
`exl3_k_map.json` and model card, and from the exllamav3 sources at the
quantizer's revision (`ba2feb2`, the dev branch the card names); the
decoder was checked on real tensors of this pack against the base
checkpoint before any engine code was written (§2.4). Numbers that are
not measured on the fabric say so.

References: <https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw>
(revision `04a492fc3b4d25ddbab8f2d2f07f235ec77c9010`),
<https://github.com/turboderp-org/exllamav3> (MIT; the codebook, the
trellis packing and the Hadamard conventions are its), the base model
<https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017> (the Engram
tables, see §1.3), [deepseek_v41_flash_plan.md](deepseek_v41_flash_plan.md)
(the family's architecture and the MXFP4/FP8 deployment this one extends).

## 0. Summary

The pack is the 1017 release re-quantized by exllamav3's trellis
quantizer: **every projection, expert, head and draft matrix is an EXL3
matrix** (`trellis` int16 + `suh`/`svh` fp16 + the `mul1` codebook marker);
routers, norms, mHC coefficients, the embedding, the attention sinks and
the DSpark Markov/confidence heads keep native widths (some re-stored in
F16/F32, §1.2); the vision tower is BF16 and skipped as before. The two
Engram tables are **not in the pack**: they are the base repo's shards
47/48, already resident on every node of the fabric.

Bit widths (the card's recipe, confirmed from the trellis widths): routed
experts 4 bits on layers 0–9 and 30–39, 3 bits on 10–29; the shared
experts, every attention projection, the compressors, the indexer `wq_b`,
the Engram `wkv`, the head, `main_proj` and all three DSpark layers 6 bits;
the indexer `wk` 8 bits. 3.53 bits per weight over the trunk.

What the port is: a fifth weight format on the shared cores — the
**trellis-decoded fp16 B fragment** fed to `mma.sync` with fp16 A
fragments, where A is the activation row rotated by the format's 128-wide
Hadamard and scaled by `suh`, and the output is rotated back and scaled by
`svh` (§2.2). Decode GEMV, MoE slot kernels, grouped prefill GEMM and the
head all follow that one chain; the loader re-tiles the trellis so a warp
streams one output tile's k range contiguously (§3). Serving is a world-4
deployment: ~61 GiB of weights per rank (the card: 121.5 at world 2, which
does not fit a Spark).

What the port is NOT: a dequantize-at-load path. The EXL3 weight is not
representable in bf16 or fp8 (a codebook value is an fp16 number and the
effective weight is a 128-point Hadamard mix of 128 of them), so any
resident re-encoding would be a second quantization. The engine runs the
checkpoint's own chain, the one exllamav3 runs — the family rule of
[native-checkpoint recipes](adding_a_model.md#4-engine-keys-exact-defaults).

## 1. Checkpoint census

### 1.1 The EXL3 matrices

Every quantized matrix `X` is four tensors:

| tensor | dtype | shape | meaning |
|---|---|---|---|
| `X.trellis` | I16 | `[K/16, N/16, 16·bits]` | the packed trellis states, one 16x16 tile per `[kt, nt]` (K = the matrix's input width, N its output width — the HF weight is `[N, K]`) |
| `X.suh` | F16 | `[K]` | the input-side per-channel factor (real values in this pack: the quantizer's refitted input scales, ≈ ±0.03 on `wq_a`) |
| `X.svh` | F16 | `[N]` | the output-side per-channel factor (real values in [−1.06, 1.06]: `out_scales: always`) |
| `X.mul1` | I32 | `[]` | the codebook marker, `-2082680531` = `0x83DCD12D` |

`bits = trellis.shape[2] / 16`. Every K and N of the pack is a multiple of
128 (the Hadamard block), which the loader checks.

| group | tensors | K x N | bits | per-rank placement at world 4 (§3.1) |
|---|---|---|---|---|
| attention `wq_a` | 40 + 3 draft | 5120 x 1280 | 6 | replicated |
| attention `wkv` | 40 + 3 | 5120 x 512 | 6 | replicated |
| attention `wq_b` | 40 + 3 | 1280 x 32768 | 6 | N sliced by heads: 16 heads x 512 = 8192 |
| attention `wo_a.slice.0..7` | 320 + 24 | 4096 x 1024 each | 6 | two whole slices (the rank's groups) |
| attention `wo_b` | 40 + 3 | 8192 x 5120 | 6 | K sliced by groups: 2048 |
| indexer `wq_b` | 8 index sources | 1280 x 4096 | 6 | replicated |
| indexer `wk` | 4 kv sources | 512 x 128 | 8 | replicated |
| compressor `wkv` / `wgate` | 4 / 3 | 5120 x 512 | 6 | replicated |
| Engram `wkv` | 2 | 6144 x 25600 | 6 | K sliced by hash heads: 3 ranges of 512 |
| routed experts `w1`/`w3` | 15,360 each | 5120 x 2304 | 4 / 3 | N sliced on the intermediate: 640/640/512/512 on even experts, 512/512/640/640 on odd (§2.3) |
| routed experts `w2` | 15,360 | 2304 x 5120 | 4 / 3 | K sliced the same way |
| shared experts `w1`/`w3`/`w2` | 40 each | as the routed, 2304 wide | 6 | as the routed, by the layer's parity |
| draft experts (`mtp.*`, 128 per stage) | 384 each | as the routed | 6 | as the routed |
| `main_proj` | 1 | 15360 x 5120 | 6 | replicated |
| `head` | 1 | 5120 x 129280 | 6 | N sliced on 128-row blocks: 32384/32384/32256/32256 |

Bytes: the routed experts 221.5 GiB (3 x 73.8), the draft experts 9.5, the
dense attention stack 3.5, the shared experts 1.0, the head 0.46, the
embedding 1.23 (BF16), everything else under 1 GiB. Per rank at world 4,
before the memory plan runs: ≈ 61 GiB.

### 1.2 Native tensors whose dtype changed

The pack stores some native tensors in widths the base checkpoint did not
use. The loader takes them only when the conversion is exact, and refuses
the checkpoint otherwise (a check, not a silent cast):

| tensor | base | pack | loaded as |
|---|---|---|---|
| `ffn.gate.weight` (routers, trunk and draft) | BF16 | F16 | bf16 — exact iff every value is bf16-representable (checked at load) |
| `ffn.gate.bias` | F32 | F16 | f32 (the F16 value, exactly) |
| `indexer.weights_proj.weight` | BF16 | F16 | bf16, checked |
| `engram.q_weight` / `k_weight` | BF16 | F32 | bf16, checked (equal to the base's bf16) |
| `mtp.2.markov_head.head.weight` | BF16 | F16 | bf16, checked |
| `mtp.2.confidence_head.proj.weight` | BF16 | F16 | f32 (exact; the engine keeps it f32 already) |
| `ffn.gate.bias_vl` | F32 | F32 | never loaded (image tokens) |

The F16 copies are the pack's values, not the base's: an F16 cannot hold a
bf16 below 2^-24, so the routers' and `weights_proj`'s smallest entries
(0.01 % of them, |w| < 3e-8) were flushed or rounded when the pack was
written (relative RMS 6e-9 against the base), and the router bias is the
base's F32 rounded to F16 (relative 4e-4 at most). The checked bf16
conversion is lossless on the pack; the deployment's routing is the
pack's, as exllamav3's is. Everything else (norms BF16, `attn_sink` F32,
`hc_*` F32, `embed` BF16, `markov_head.embed` BF16, vision BF16) is as in
the base release.

### 1.3 The Engram tables

`layers.1.engram.embed.{weight,scale}` and `layers.14.engram.embed.*` are
absent from the pack; the card names the base repo's shards
`model-00047-of-00048.safetensors` (sha256 `76a05106…`) and
`model-00048-of-00048.safetensors` (`9dc51433…`) as their source, which
the fabric holds under the base snapshot on every node. The deployment
names that snapshot (§6: `engine.engram_source`), the loader maps the
tables from it, and the Engram sidecar (`dgpp_engram_tables.json`, the
tokenizer's hash constants) is the base snapshot's — the pack ships the
same `tokenizer.json`.

## 2. The format and numerics contract

### 2.1 The trellis code

Per 16x16 tile, 256 weights in the mma fragment order (`tensor_core_perm`:
position `8t + j` of lane `t` is tile element `(r, c)` with
`r = (t % 4)·2 + {0, 1, 8, 9}[j % 4]`, `c = t / 4 + 8·(j / 4)` — `r` along K,
`c` along N). The tile's `16·bits` int16 words, read as little-endian
uint32 words, form a 256·bits-bit stream MSB-first; position `p`'s 16-bit
state is the stream's bits `[(p+1)·bits − 16, (p+1)·bits)` modulo
256·bits (a tail-biting ring: position 0's high bits are the stream's
last `16 − bits` bits); the low `bits` bits of that window are position
`p`'s own. The codebook value of a state `s` (the `mul1` codebook) is

    x = (s · 0x83DCD12D) mod 2^32
    h = 1024 + (byte sum of x)              — an fp16 bit pattern in [0x6400, 0x67FF]
    value = fp16( fp16(h) · 0x1eee + 0xc931 ) — an hfma: 1/147.7 and −10.39, fp16-exact

i.e. 1,024 distinct fp16 values in about [−3.46, 3.47]; the kernel computes
it as imul + dp4a + hfma, no table (exllamav3 `codebook.cuh`,
`decode_3inst<2>`).

### 2.2 The weight and the forward

With `w_inner[k, n]` the decoded tile values (K x N, the "rotated basis"),
`H` the 128-point Sylvester Hadamard divided by √128, applied blockwise:

    W[k, n] = suh[k] · (H · w_inner)[k, n] … then · H on n … · svh[n]
    W = D_suh · H_K · w_inner · H_N · D_svh ;  the HF weight is W^T

and the forward of a row `x` (1 x K) is

    xh = (x ∘ suh) · H_K           (blockwise over 128-k blocks)
    y_inner = xh · w_inner          (the GEMM on the decoded values)
    y = (y_inner · H_N) ∘ svh      (blockwise over 128-n blocks)

exllamav3 computes exactly this (`had_r_128` with `suh` as the pre-scale
before the GEMM, with `svh` as the post-scale after; `exl3_gemm` /
`exl3_gemv` on the trellis). The arithmetic it uses, which this port
takes as the reference chain (`hadamard_inner.cuh`, `exl3_gemm_inner.cuh`):

1. input: the row in fp16, multiplied by `suh` **in fp16**
   (`__hmul2`), then the 128-point butterfly **in fp32** (the four-element
   stage in registers, the 32-lane stages by shuffles), the result times
   `1/√128` (`0.088388347648f`) rounded **to fp16** once;
2. the GEMM: `mma.sync.m16n8k16` with fp16 A (the row above) and fp16 B
   (the codebook values, exact), **fp32 accumulation** in ascending k
   (sm_86 alone accumulates in fp16; the GB10 does not);
3. output: the fp32 accumulators through the same fp32 butterfly, times
   `1/√128`, then times `svh` **in fp32** (the fp32-output form; the fp16
   form rounds to fp16 first and multiplies by `svh` in fp16).

The engine's chain, fixed here:

- the activation row arrives as bf16 (the residual stream); it is widened
  to fp32, multiplied by `suh` in fp32 (exact: an fp16 times a bf16 fits
  fp32), rotated in fp32, scaled by `1/√128` and rounded to fp16 once —
  the same operations as the reference's fp16 input path up to the first
  multiply, which the reference performs in fp16 (a rounding this chain
  does not take);
- the mma in fp16 x fp16 with fp32 accumulation, ascending k, the same
  chain whatever the row count (the row-count invariance every decode
  format here keeps);
- the output rotation in fp32, `1/√128`, `svh` in fp32 (the reference's
  fp32-output form), then the site's own rounding: bf16 where the stream is
  bf16, fp32 where the consumer takes fp32 (the MoE down partials, the
  logits).

So the engine's chain is the reference's fp32-output chain with one fewer
rounding at the input; it is not bitwise exllamav3 (a different fp32
accumulation order inside the mma tiles, as every GEMM here is to its
reference). The task evals and the perplexity gates judge it, as for the
other quantized families; the port's own gates are the kernel-vs-host-oracle
parity tests (§5).

Why fp16 and not bf16 on the tensor cores: a codebook value has 11
significant bits; rounding it to bf16 would be a second quantization of
every weight (relative 2^-9 per value, uncorrelated), which the lossless
rule forbids for a weight-format path. The fp16 mma runs at the bf16
mma's rate on the GB10.

Where fp16 activations could overflow: the rotated row is the bf16 row
scaled by √128 at most per element and typically ≈ 1x in magnitude
(the Hadamard preserves the 2-norm; `1/√128` normalizes it). The sites
that feed EXL3 matrices are post-norm rows (attention input, the q/kv
latents, the indexer and compressor inputs, the expert inputs, the SwiGLU
outputs clamped at 10, the attention output, the draft's concatenated
states). fp16's range (65,504) is not reached by any of them in the
reference, which runs the whole model in fp16; the kernels saturate on
conversion rather than produce inf, and the host oracle test covers the
large-magnitude rows.

### 2.3 TP slices and the 128 blocks

The forward's two rotations act on 128-element blocks of K and of N
independently, so a TP slice must hold whole blocks on its sliced axis
(an output block mixes 128 consecutive pre-rotation columns; an input
block mixes 128 consecutive activation elements):

- output-sliced (`wq_b` by heads, the experts' `w1`/`w3`, the head): the
  rank's N range is whole 128-blocks; `svh` is sliced with it;
- input-sliced (`wo_b` by groups, the experts' `w2`, the Engram `wkv` by
  hash heads): the rank's K range is whole blocks; `suh` is sliced with
  it; the rank's partial output goes through the output rotation and
  `svh` before the cross-rank sum (the rotation is linear, so the sum of
  the rotated partials is the rotated sum; each rank rotates its own
  partial and the FFN all-reduce stays a plain sum).

The attention sites divide (16 heads x 512, 2 groups x 1024/4096). The
experts' intermediate 2304 = 18 blocks does not divide by 4: **the
intermediate is sliced 5/5/4/4 blocks (640/640/512/512) on even-indexed
experts and 4/4/5/5 on odd ones** (`exl3_expert_block_split`, 2026-10-23;
a layer's shared expert takes the layer's parity) — a rank's `w1`/`w3`
carry 640 or 512 rows per expert and its `w2` the matching columns; the
collective shape (H-wide partials) does not depend on the slice width,
and the slot and grouped kernels take each expert's width from its view.
With one rule for every expert the busiest rank carried 11 % more expert
bytes than an even split (640 against 576) and the step waited for it at
every fold; alternating the remainder between the halves of the world
puts a decode row's six experts at 27.7 blocks on the busiest rank in
expectation (30 before, 27 even), and a 14-row step at 9 % fewer. The
resident buffers take the widest slice (640 on every rank). The head's
129,280 rows (1,010 blocks) split 253/253/252/252 blocks; the vocabulary
range per rank is journaled as before.

### 2.4 The decoder checked on the pack (2026-10-22)

A numpy decoder written to the rules of §2.1–2.2 (the one that reviewed
the Qwen3.8 EXL3 pack on 2026-09-30, memory `exl3-qwen38-review`) was
run on tensors fetched from this pack and compared with the base
checkpoint's own values — the check that the bit order, the permutation,
the codebook and the two rotations are the pack's and not a plausible
variant (a wrong one lands near relative RMS 1.4):

| tensor | bits | relative RMS vs the base matrix | max abs / max |base| |
|---|---|---|---|
| `layers.0.attn.wq_a` (base: fp8 e4m3 + e8m0/32x32) | 6 | 0.0203 | 0.044 |
| `layers.0.ffn.experts.0.w1` (base: MXFP4) | 4 | 0.0794 | 0.095 |
| `layers.10.ffn.experts.0.w2` (base: MXFP4) | 3 | 0.1371 | 0.148 |

The classes measured on the Qwen3.8 EXL3 pack against its BF16 base were
0.137–0.142 at 3 bits and 0.036–0.041 at 5 bits; the 4- and 6-bit values
here sit where the rate–distortion constant of that study
(relative RMS x 2^bits ≈ 1.2) puts them. The `mul1` marker is
`0x83DCD12D`; `suh` and `svh` are both real per-channel vectors (640 and
298 distinct values on `wq_a`), applied as §2.2 says whatever their values.

## 3. Resident layout and kernels

### 3.1 The resident matrix (`Exl3Matrix`, `models/exl3_matrix.hpp`)

    tiles  uint8 [N/16][K/16][32·bits]   the trellis re-tiled n-tile-major: an output tile's k range contiguous
    suh    f16   [K]                     this rank's K slice
    svh    f16   [N]                     this rank's N slice
    rows N, cols K, bits (3..8)

A lossless re-ordering of the checkpoint's tile bytes (the tile contents
are untouched); the loader slices n-tiles for an output slice and k-tiles
for an input slice. K and N multiples of 128; the n-tile-major stream of
one output tile is `K/16 · 32·bits` bytes (wq_b: 320 tiles x 192 B).

### 3.2 The decode core (`kernels/exl3_gemv.cu`)

One kernel family for m ≤ 64 activation rows (the decode batch) and the
prefill's 64-row groups, built on the streaming GEMV's shape
(`mma_gemv.cu`): a block of eight warps owns 128 outputs (eight n-tiles,
one per warp) and walks the k range in 128-k windows:

- A: the block stages the window's activation rows once (bf16 from the
  caller's buffer), rotates each row's 128 elements by `suh` and the
  Hadamard in fp32 and stores fp16 to shared memory; eight warps read the
  same fragments (the window is exactly one rotation block);
- B: each warp streams its n-tile's eight k-tiles of the window through a
  per-warp `cp.async` ring (768/1024/1536/2048 B per window at 3/4/6/8
  bits), decodes its eight states per tile from the staged words (the
  exllamav3 `dq8` forms by bit width) into the two `n8` B fragments and
  runs `mma.sync` per 16-row A tile — the decode is shared by every
  A tile, so m = 64 costs the same weight stream as m = 1;
- epilogue: the warp's 16 x m partials into shared memory, one warp per
  row rotates the block's 128 outputs, scales by `1/√128` and `svh`, stores
  bf16 or fp32;
- split-K for the narrow sites (`wkv` 512, `wq_a` 1280, a `wo_a` group
  1024): fp32 pre-rotation partials in the caller's workspace, a reduce
  kernel that sums the splits in order and applies the output rotation;
  the split count a function of the shape alone, so a row's chain is the
  same at every m (the invariant the batch family relies on).

Two matrices off the same input in one launch (the `wq_a` + `wkv` pair,
the two `wo_a` groups) as the fp8 form has them.

Expected (unmeasured): the decode is DRAM-bound at the 3.5-bit stream —
about 4.5 instructions per weight on the decode (shift, mask, imul, dp4a,
half an hfma2) against the ~0.2 warp-instructions per weight the stream
rate allows at 233 GB/s, inside the issue budget with the pipelining of
the fp8 form.

### 3.3 The MoE slot kernels (`kernels/glm_moe.cu`)

`moe_slot_gate_up_swiglu_exl3` / `moe_slot_down_exl3`: the decode core at
m = 1 per slot with the slot's expert's tiles, `suh` and `svh`
(`MoeExpertView` gains the EXL3 fields). Gate and up for one 128-column
block in one block (16 n-tiles, two per warp), both rotated and scaled in
the epilogue, then the SwiGLU op for op as the fp8/fp4 kernels round it;
the down kernel rotates the slot's SwiGLU row by its expert's `suh`,
multiplies the expert's `w2` tiles, rotates and scales its 128-output
block and stores the fp32 slot row the existing weighted accumulation
sums. The shared expert is view-table entry `n_experts` (as the fp4 and
packed forms). The uneven slice (§2.3) is the view's own `rows`/`cols`.

The expert-run form (2026-10-23, `kernels/exl3_gemv.cu`): the slots run in
the expert-sorted order (`launch_moe_slot_order`), which now also emits at
each position the length of the equal-expert run starting there; the
rotate kernel lays the rotated rows out by position, so a run's rows are
contiguous, and the run's first block takes them all as one A tile — up to
16 rows on the deep-ring tile, up to 64 (the decode batch's bound) on the
64-row tile, a second launch that exists only above 16 token rows — while
the duplicates' blocks return. An expert several rows share is streamed
and trellis-decoded once; the per-slot form had re-read it per slot with
L2 serving under half (eight identical streams cost the slot kernels 4.9x
one stream's time, §7). Per row the core chain is the one-row form's, so
the act rows and the down rows (stored at their logical slots for the
accumulation) are bitwise the per-slot form's at any duplication.

The run groups (2026-10-23 evening): the order kernel also writes the
runs longer than 16 rows as a launch list after the run lengths — each
run cut into groups of up to 64 rows, an entry `position | rows << 16`,
−1 past the last (`glm_moe_launch.hpp`: `kMoeRunTileRows`,
`kMoeRunGroupRows`, `moe_run_groups_max`) — and the 64-row launch's grid
is that list's bound (`slots / 17 + 1` entries), not every position: at
forty rows the launch had put 1,400 blocks of a 64 KB-smem kernel on the
GPU to return at once, ~240 us a layer (the C8 trace, §7), and a run of
80 or 96 rows (sixteen slots' shared expert) is now two groups. The
16-row tile's launch is unchanged. The decode slot path at 80 and 96 rows
stays bitwise the host path (the MoE test's cases).

### 3.4 Prefill

The grouped expert GEMM over the route's segments
(`launch_moe_grouped_mma_exl3_*`): the decode core's structure with up to
four 16-row A tiles per block and the segments' rows gathered and rotated
by the segment's expert's `suh` as they are staged — correctness first;
its rate is the second round (a 128-row tile and a deeper B ring).
The dense sites run the same kernel in 64-row groups in the first version
(the weights re-read per group from L2 at the dense matrices' sizes); a
reconstruct-to-fp16 + tensor-core GEMM form for the large dense chunks is
the measured alternative if the groups fall short.

### 3.5 Everything else

The L2 prefetch windows span the tile bytes; the head reads its EXL3
matrix with the fp32-logit epilogue (no bf12 companion: the head is 6-bit
already); `main_proj`, the Engram `wkv`, the compressors and the indexer
projections take the dense launcher.

### 3.6 The decode batch past sixteen rows (2026-10-23 evening)

The template ships at sixteen slots on the family's 96-row decode batch
(`Dsv41Model::decode_rows_cap`, the engine's `kDecodeRowsMax` and the
pick, GEMM, CSA2 select and upload bounds that mirror it; sixteen
five-row DSpark blocks are 80 rows, the six-row block's 96). The
step's cost past a few rows is not expert bytes: at forty rows with one
stream's experts (the counting prompt's eight identical streams, the C8
trace of §7) the step was 91 ms against 38 at C1 — ~1.2 ms a row in
kernels that read no expert — and the forms below take it down. Every
form is bitwise its predecessor per row or a fixed function of the
request alone, so a row's chain does not depend on the rows beside it
(the engine gates: the group prefill's first step bitwise the prefills
alone, the 40-step decode's budget, the sixteen-slot loopback world).

* **The attention split groups** (`kernels/dsa.cu` `attn_flash_kernel<…,
  kGroup>`, `kAttnSplitGroup` 8; `kernels/dsa.hpp`): the decode window's
  eight splits and the main source's thirty-two publish a partial per
  split — 16 heads × 512 fp32 each, fifty megabytes a layer at forty
  rows, written by the kernel and read back by the finish at the DRAM
  rate (160 us each way). The eight split blocks of a group now take a
  ticket, and the last to arrive folds the eight partials from L2 with
  the finish's own formula (`mhat = max m_s`, `w_s = exp(m_s − mhat)`,
  `l = Σ w_s l_s`, `c = Σ w_s c_s`, in split order, the eight vectors of
  a column loaded together) into the group's partial
  (`m_grp_* / l_grp_* / c_grp_*` in the CSA2 layer's scratch, one per
  group) and `discard.global.L2`s the eight dead rows after a block
  barrier, so they never write back. The finish reads four main and one
  window partial per row instead of forty. The grouping is per request,
  so a row's merge is the same whatever the batch — a one-time change of
  the decode attention's rounding class (the two-level merge: relative
  l2 5e-8 against the flat combine; `dsa_test`'s split-group case pins
  repeat and first-rows-alone bitwise). A thread-block-cluster form
  (distributed shared memory; the GB10 runs clusters of eight) was
  measured first and kept out: clusters beyond the twelve co-resident
  ones schedule in coarse waves, +35 us a layer at ten to twenty rows.
  Standalone (cold partials, 300 keys, 32 splits), flat against grouped
  flash + combine: 5 rows 65 / 73 us, 20 rows 171 / 167, 40 rows 454 /
  315, 80 rows 917 / 744.
* **The router's tiled dots from sixteen rows** (`glm_moe.cu`): the
  decode form streamed the gate matrix once per (token, expert) warp, 76
  us a layer at forty rows; the prefill's 16 × 16 tile — bitwise the
  warp form, its own test — now serves decode batches of sixteen rows and
  more with the select as its second launch (44 + 4.5 us).
* **The mHC dots' token tile from sixteen decode rows** (`glm_mhc.cu`
  `kMhcDecodeTileMinTokens`): the per-coefficient fused form re-reads the
  coefficient matrix per token (80 us a site at forty rows); the four-
  token tile — bitwise the fused form, its deferred comb kept
  (`mhc_finish_block`'s flag), the tensor-core prefill form excluded —
  takes decode batches of sixteen rows and more. Level at forty rows (78
  us: ten blocks of four tokens are latency-bound where the fused form
  was L2-feed-bound); the many-row mHC form is open (§7).
* **One split chain for the tensor-core GEMV to 128 rows**
  (`mma_gemv.hpp` `kMmaGemvSplitRows`, `mma_gemv.cu`): the 65–128-row
  form had launched unsplit while rows up to 64 split by shape; past 64
  rows a row's reduction chain changed with its batch — sixteen slots'
  transcripts diverged at near-ties where twelve matched, caught by the
  sixteen-slot engine gate and bisected with the dispatch trace. The
  eight-tile form now splits like the narrower ones; the split count was
  already a function of the shape alone.
* **The graph-variant budget** (`bus_kernel.hpp` `kBusMaxGraphVariants`
  256): sixteen slots with eight batch families kept two of the three
  scheduled verify depths at 128 and paid a fifth row on code and prose
  at C1 (+2 ms a step); 150 variants keep three, 250 all five.
* The batch-map upload's word bound follows `kPickMaxRows`.
* **The verify schedule's cost curve, measured on the pack**
  (`engine.mtp_schedule_row_ms` 2.95, `engine.mtp_schedule_base_ms` 19.9;
  the template had carried the release pack's 5.2 / 31.1). With four depth
  options the schedule took depth 2 on steps where depth 3 accepted more
  — the C1 code class's tokens a step fell 3.15 → 2.90 at a level step in
  the first sixteen-slot campaign — because the curve overstated a row.
  The mode boots give the curve directly: C1 steps at 1 / 2 / 3 / 4 / 6
  rows are 18.5 / 25.8 / 28.8 / 31.9 / 37.6 ms on the code prompt (prose
  18.5 / 25.4 / 28.4 / 31.3 / 37.0), ~2.95 ms a row over a 19.9 ms base;
  the threshold the schedule applies falls from a 0.234 to a 0.133 prefix
  survival at the same value of time. The curve alone did not return the
  code class's tokens a step (2.93 at the lower threshold too): with every
  depth an option the policy's depth-2 decisions stand, and the confidence
  under-reads the third row on code; the morning's three-option set had
  rounded them up to depth 3. The new key `engine.mtp_schedule_options`
  (`docs/configuration.md`) caps the options at the same even spread the
  budget used to impose, and the template pins three — (1, 3, 4) through
  the evening's campaign of record, (1, 3, 5) since.
- **The fifth draft** (2026-10-23, late evening). The counting, code,
  table and JSON classes saturate the four-draft block: the head's prefix
  survivals stay high through all five positions (counting accepts
  4.96 tokens a step at depth 4, the most five rows can give) and the
  schedule had no deeper option. With `engine.mtp_depth` 5 and the same
  three-option spread (1, 3, 5), C1 legs on the sixteen-slot binary
  (`9a179b15`): counting 4.96 → 5.95 tokens a step, code 4.37 → 5.17,
  tables 4.73 → 5.20, JSON 3.45 → 3.60, math 3.83 → 4.23, prose and
  narrative unchanged (2.25, 1.88: the schedule stays at 1–3 rows there),
  at a 6-row step of 38.0–38.6 ms (2.95 ms a row over the 4-draft step).
  Per-stream decode: code +8 %, counting +11 %, tables +2 %, math +2 %,
  prose −1 %, narrative −1 %; the eight-class C1 mean +1.5 % and the
  two-stream mean +2.7 %. On the campaign classes: JSON 120.8 → 133.1
  tok/s, code 91.1 → 92.0, prose 68.5 → 67.2. Four options (1, 2, 4, 5)
  and five (every depth) measure the same within the repeat spread; the
  template keeps three (the variant budget: 44 captures an option at
  sixteen slots). Memory: the sixteen slots' six-row blocks are the
  96-row batch exactly.
- **The listed attention's split floor — measured, withdrawn** (2026-10-23,
  late evening). The decode attention splits a row's selected keys 32
  ways whatever their count, and a decode selection is at most
  `engine.index_topk` (512) keys, so every split holds a tile or less and
  the kernel pays for its partials (6 MB a source through L2 at six rows,
  merged by the group's last block and read again by the finish). A
  32-key floor on the chunk (the live splits ceil(count / 32), the rest
  publishing the identity and no c row) cut the kernel pair 30–50 % at
  3–40 rows standalone and read, on the fabric, C2 +2.7 %, C4 +7 %, C8
  JSON 276 tok/s — and on the C1 prose and chat prompts it changed the
  greedy transcript (a rounding class moves with the merge order) to one
  that drafts 7 % fewer tokens a pass at the same pass cost. A C1 prose
  or chat regression is not acceptable whatever the aggregate buys, so
  the floor is not in the tree; the partial traffic it exposed is the
  attention item of the step program below (one kernel a layer over both
  key sources), to be built with its transcript gates in place.

## 4. Loader, binding, config

- `quantization_config.quant_method == "exl3"` with `codebook == "mul1"`
  selects the pack: `Dsv41TextConfig::weights_exl3`; the binding table
  emits the EXL3 quadruple for every matrix of §1.1 (roles `Exl3Trellis`,
  `Exl3Suh`, `Exl3Svh`, `Exl3Marker`) and the §1.2 dtypes; the Engram
  tables are expected from the `engram_source` checkpoint (§6).
- The builder gains `load_exl3_rows` (an output slice: n-tiles + `svh`
  range) and `load_exl3_cols` (an input slice: k-tiles + `suh` range) and
  checks the marker, the 128-multiples and the bit width from the trellis
  shape; the resident structs carry `Exl3Matrix` beside the fp8/fp4 ones
  with a per-layer format tag.
- Geometry: `Dsv41LocalGeometry` carries the even slice's intermediate
  begin, the widest slice's count and the 128-aligned head range; each
  expert's slice is `exl3_expert_block_split`'s (§2.3).
- The resident image key: `loader_format()` 2 since the alternating
  slices (an image of the one-rule layout is stale and rebuilt).

## 5. Tests

1. `exl3_codec_test` (host): the host decoder against the §2.1 rules on
   synthetic tiles (every bit width, the ring wrap, the permutation), the
   host reference forward's algebra (rotation identities), and — with
   `--checkpoint-dir` — real tensors of the pack against the base release.
2. `exl3_gemv_test` (CUDA): the decode core against the host oracle at
   the pack's shapes and bit widths, m = 1..64, split and unsplit, the
   row-count invariance bitwise, the pair form, the fp32 epilogue as the
   unrounded bf16.
3. `glm_moe_test`: the EXL3 slot path and grouped path against the host
   MoE oracle with EXL3 expert views, the uneven slice at world 4.
4. `dsv41_loader_test` / `dsv41_model_test` / `dsv41_engine_test` /
   `dsv41_tp_test` on an EXL3 fixture (`--write-fixture` from the binding
   table: random trellis bits are a valid trellis; `o_lora` 128 and
   intermediate 640 so every axis is 128-blocked and the world-4 slice is
   uneven), the existing gates (chunk invariance, verify/rollback parity,
   graph == eager, worlds 2/4 loopback).
5. The forward fixture's host reference takes EXL3 matrices through the
   full reconstruct (fp64), so the kernel chain is compared at tolerance,
   as the fp4 family's.

## 6. Deployment

`deploy/cluster_deepseek-v4.1-flash-1017_exl3-3.5bpw_w4.example.json`: the
1017 template's keys plus `engine.engram_source` (the base snapshot's
model id, resolved through the HF cache like `model`). World 4 only.

## 7. Port record

- 2026-10-22: census, contract and this plan; the pack's download to the
  one node with disk (node 4, 673 GB free; nodes 1 and 2 hold 24/27 GB)
  streaming through the head (the workers have no internet); the decoder
  check of §2.4 (results below as they land).
- 2026-10-22/23, built (every piece on the shared cores, no new family):
  the resident view and host codec (`models/exl3_matrix.hpp`), the device
  codec (`kernels/exl3_codec.cuh`) and the decode / grouped / MoE slot
  launchers (`kernels/exl3_gemv.cu`; a launch is a rotate kernel and the
  streaming GEMM, plus one reduce when split; a site hands the launcher a
  workspace sized by `exl3_gemm_ws_bytes_for`); the MoE layer's EXL3
  branch (decode: rotate slots → gate/up/SwiGLU → rotate → down into the
  existing accumulation; prefill: the grouped kernels with the gather
  forced); the config (`weights_exl3`, the k-map beside `config.json`,
  the 128-multiple checks), the binding table's EXL3 quadruples and
  checked F16 / F32 natives, the loader's row / column / column-range
  slices (whole 128-blocks; `exl3_block_split` for the intermediate and
  the head), the Engram tables through the stream's extra-shard-directory
  hook; the attention layer (`Csa2Config::exl3`, one workspace span of the
  layer scratch sized over its seven sites, every projection site branched,
  the windows prefetching the tiles), the Engram projection, the model
  (`head_logits` through the EXL3 head, the draft's main projection, the
  prefetch windows; the bf12 head pack and the draft-head screen stay off —
  there are no bf16 head bytes); the serve key `engine.engram_source`
  (resolved like `--model`, required for an EXL3 pack, refused for other
  families); the numpy reference's trellis decode (`load_exl3`, pinned to
  the host codec's real tile); the fixture switch `DGPP_DSV41_FIXTURE=exl3`
  and the `_exl3` ctest variants of every dsv41 suite (the forward chain's
  reference dumps included). Tests: `exl3_codec_test` 6/6,
  `exl3_gemv_test` all shapes, `glm_moe_test` (three EXL3 cases),
  `dsv41_loader_test` (five EXL3 cases); the model-level suites' results
  land below.
- The pack itself, checked offline on node 4 (2026-10-23): the binding
  census (`dsv41_bind_check`, headers only) matches 192,495 tensors with
  their dtypes and shapes, nothing unexpected, exactly the four Engram
  table tensors missing (the base release's); the serve binary's
  `--memory-plan` on the resolved template exposed one defect the fixture
  could not — the k-map's JSON text was a temporary while its parsed
  values (string views into it) were read, so every string key dangled —
  fixed in `Dsv41TextConfig::from_json_file`. With it, the census through
  `--engram-source deepseek-ai/DeepSeek-V4.1-Flash-1017` binds all 192,499
  tensors (47,900 EXL3 matrices), and the template's memory plan at world
  4 fits: rank 0 holds 68.0 GiB of resident weights (87.5 GiB in all with
  the 14 GiB prefix arena and the 1M-token pool), rank 3 55.1 GiB (74.6
  GiB in all), against 116.7 GiB free per node.
- The model-level suites on the EXL3 fixture (2026-10-23): the forward
  chain passes the release fixture's strict teacher-forced budgets once the
  numpy reference models the engine's one fp16 rounding (exact-double
  weights left layers 2–3 at the l2 budget's edge with a few hard
  elements); the decode path matches the prefill path at every layer
  within 0.3–0.6 % for every row whose selections agree; every larger
  deviation the suites saw coincided with a certified selection flip (the
  coded index query differs). The random EXL3 fixture's logits are
  near-tied — top-2 margins of 0.004–0.25 on rows of rms 0.29, against
  the release fixture's 0.01–1.4 — so a flip, which moves a row's logits
  by up to 0.4 of their rms on either fixture, takes the argmax with it
  where the release fixture's margins absorb it. The suites now state
  that regime instead of assuming the release fixture's draw: the model
  test's prefill-form, decode-step and verify-row comparisons exempt a
  certified flip (as its decode audit already did), the decode test's
  first-flipped-row bound is the flip's move (0.5, from 0.1), and the TP
  test's fold floor is 5 % of the layer's rms (four bf16 partials
  cancelling on isolated near-zero channels at world 4 reached 2.4 %).
  The engine test keeps HEAD's exact comparisons (the graph, DSpark,
  scheduled, six-slot and group-prefill transcripts equal the eager
  engine's token for token on both fixtures): the port had loosened them
  to a near-tie rule for a day, and that rule hid the graph-table defect
  recorded below. The release fixture shows the same cross-form regime
  (its prefill tile-vs-streaming gaps run 4e-4 at layer 0 to 6.5e-2 at
  layer 7 with a last-row selection flip its margins absorb). Defects found by the suites on the way:
  the release fixture's F32 draw had been rounded to bf16 with the EXL3
  natives (restored exact: it moved the fixture's near-tie selections),
  and a test helper indexed a one-row decode's selections at the forward's
  row. Pre-existing on HEAD a24e05b5, not of this port: `pdl_wait_audit`
  (the Flint sink-attention kernels) and `serve_test`'s
  `serve_edge_reasoningAliasesAndSwitchOnlyModels` (a 200 where a 400 is
  expected), both reproduced on a clean export of HEAD.
- Fabric state at 2026-10-23 01:00: the pack is complete on node 4 (44
  shards, sha256-verified) and copying to node 3 (442 GB free); nodes 1
  and 2 hold 23/27 GB free against the 240 GB pack (their 812–938 GB
  resident-image caches are mostly images last read on 2026-10-17/18) —
  the world-4 boot waits on that space.
- 2026-10-23 05:00–05:40, the fabric: on the user's instruction the
  images not read since 2026-10-18 were deleted from nodes 1 and 2 (612
  and 502 GB; production's two Qwen images kept), the pack copied node 4
  → node 1 and node 3 → node 2 (four ssh streams, ~1.4 GB/s, three
  minutes each, every file's size verified; all four nodes hold revision
  04a492fc), production taken down for the window, and the release binary
  built (`build-release`, source a24e05b5 with the port's working tree).
  The first boot builds the resident images: rank 0's 43 layers (71.5 GB)
  in 247 s, the peers' (55–71 GB) at ~90 MB/s, so the 120 s rendezvous
  wait cut the peers twice and the third boot ran with
  `--rendezvous-timeout-ms 900000` through the launcher's knobs; with the
  images present a rank restores in ~27 s and the world listens in ~140 s.
- The defect the first requests found (2026-10-23 05:22): the first
  token after a prefill was right and confident, the text then degraded
  within a few tokens (`The capital of France is` → ` Paris.<Route (
  word`); identical with and without the drafter, and right with the
  decode graph off. `GlmMoeLayer::prepare_graph_table` — the per-slot
  expert view table the decode graph reads — wrote the routed EXL3 views
  but had shared-triple branches for the NVFP4 and packed-int forms only,
  so every graph step ran each MoE layer's shared expert on a stale view;
  the eager table (`upload_expert_views`) had the EXL3 branch, hence the
  right prefill and eager decode. Fixed with the missing branch; the MoE
  test's EXL3 decode case now runs the graph table too (bitwise the host
  chain, `glm_moe_test`), the engine test's exact comparisons are back
  (above) and pass on both fixtures, and the template as shipped (graph,
  DSpark depth 4 scheduled) answers right on the pack: ` Paris.`, the
  counting and weekday continuations, a correct Rayleigh-scattering
  paragraph, an iterative Fibonacci, 180/2.5 = 72 km/h, at 62–97 tok/s
  wall for single short requests. Lesson kept: a comparison between two
  engines on the same kernels is exact or it is a defect; the near-tie
  regime belongs only where two kernel forms legitimately differ.
- The campaign of record (2026-10-23 05:41–06:41 UTC; source a24e05b5
  plus the port's working tree, binary c7efb833f88a06e3; the release's
  set — the five DSpark mode boots included): every phase ok, 7/7
  launches with matching operation streams, every rank at zero swap
  before and after, the 12 cold probes at zero cached tokens. Decode C1
  65.0–105.0 / C2 90.9–141.8 / C4 115.5–156.4 / C8 91.1–137.9 tok/s
  against the release template's 53.5–89.4 / 75.5–121.9 / 94.7–130.3 /
  83.8–139.0; the plain pass 20.0 ms against 23.9; the context matrix
  36.21 → 51.39 ms a pass from a short prompt to 512K against 43.12 →
  60.35; cold prefill 1.225 / 3.489 / 13.607 s against 1.006 / 2.638 /
  10.024 (§3.4's first prefill version; its second round is the open
  work); HumanEval 159/164, GSM8K 294/300, extraction 100/100 against
  157 / 295 / 100. The rows are in docs/benchmarks.md and the card's
  Measured section. Production (the Qwen FP8 world) restored at 06:43 UTC.
- The decode and prefill program (2026-10-23 from 07:00 UTC; the
  mandate: decode and prefill throughput on this pack, every lever judged
  inside the step's schedule rather than one kernel at a time). The
  morning's campaign: C1 67.0 / 91.5 / 105.0 / 93.1 / 65.0 by
  class (decode-only), C2 90.9-141.8, C4 115.5-156.4, C8 (six live)
  91.1-137.9; prefill 1,650 / 2,280 / 2,350 / 2,340 tok/s at 2K / 8K /
  32K / 128K — the aggregate past four streams and the short-prompt
  prefill are where the pack trails the most.
- The step's structure on the pack (nsys node traces of rank 0, the
  campaign's binary, C1 code at the schedule's ~3.5 verify rows: 39.0 ms
  a step under the profiler; the plain one-row step 21.2; C4 code 80.9):
  the EXL3 slot kernels stream at the DRAM ceiling once the grid fills the
  SMs — gate-up 290 us a layer for ~69 MB (238 GB/s), down 155 for ~34 MB
  (222) — so the marginal verify row IS its bytes (the campaign's mode
  boots: 20.0 ms at one row, then 4.3 ms a row through six; the floor for
  six disjoint experts on a 640-wide rank is 4.4). The one-row gate-up
  runs at 72 % (35 blocks for 48 SMs: 140 / 122 us a layer for 24.6 /
  19.7 MB), the down at 100 % (280 blocks). The rest of a step: the
  dense attention GEMVs 4.2 ms (273 launches of 15.5 us, L2-fed by the
  prefetch windows: 1.2 GB a pass would be 5.2 ms cold), the collectives
  4.5 ms (94 x 48 us; 91 us each at C4's ~16-row payloads: 8.7 ms), the
  small kernels ~7.5 ms of which the EXL3 chain's own rotate and
  split-reduce launches were 2.0 (289 + 279 + 86 launches of 2-8 us), the
  Engram stage wait 0.5, main-stream gaps ~2.3; the prefetch windows 15 ms
  of side-stream kernel time. The byte floor of a one-row pass is ~12 ms
  (routed 4.4, shared 1.3, dense 5.2, head 0.5, router 0.7) against 20.0
  measured.
- Closed with numbers, the prefetch windows (C1 legs, three repeats, equal
  tokens a pass): as shipped 34.48 / 38.97 / 28.49 ms a pass code / JSON /
  prose; `--no-l2-prefetch` 37.02 / 41.43 / 31.18 (the windows are worth
  2.5-2.7 ms a pass, 7-9 %); a 24 MiB budget 34.75 / 39.25 / 28.67; the
  full boundary rate 35.12 / 39.59 / 29.40. The template's 18 MiB light
  stays.
- The fusion of the EXL3 chain's small launches, built and measured the
  same morning. (1) The split form's reduce folded into the GEMV — the
  last split block of an output block to arrive (an arrival counter per
  block) sums the splits in split order, rotates, scales, stores; bitwise
  the separate reduce. Two defects of the first form, both caught by the
  dsv41 suites and never by the kernel's own parity test: a static
  `__shared__` word pushed the 99 KB instantiations past the opt-in limit
  (`cudaFuncSetAttribute` refused), and the counters sat after the
  partials, where a narrower site's partials landed on a wider site's
  counters in the shared per-layer workspace — and the model's two sites
  had passed the cuBLASLt workspace, which cuBLASLt writes freely. The
  counters now live in a fixed 1 KB region at the head of every EXL3
  workspace, zero between launches (the owner zeroes the workspace once at
  allocation, the last split block resets its own), the model's head and
  DSpark main projection have a private workspace, and `exl3_gemv_test`
  shares one workspace across its shapes and repeats every launch
  bitwise, as the layers share theirs. (2) The A-row rotation inside the
  GEMV's stage (the raw bf16 rows staged by cp.async, each 128-k window
  rotated in place by the matrix's suh — bitwise the rotate kernel), first
  with its own barrier, then pipelined (the next window's rows rotated
  during this window's compute, the suh in registers a window ahead, no
  barrier of its own, the ring deepened to eight stages): both forms
  SLOWER, 34.48 / 38.97 / 28.49 -> 41.0-41.5 / 47.1-47.3 / 33.7-34.2 ms a
  pass, the plain step 20.0 -> 21.1. The trace diff names the mechanism:
  every kernel that rotates in its stage runs 30-60 % longer (gate-up 290
  -> 390 us a layer, down 155 -> 199, the dense GEMV 15.5 -> 24.7 us a
  launch) while the removed launches were worth 2.0 ms — the rotating warp
  is also a computing warp, so its rotation sits on the block's critical
  path at every window, and these blocks are not DRAM-slack enough to
  absorb it (registers 48-64, no spills: the work itself). Reverted: the
  rotate kernels stay ahead of the GEMV and the slot kernels; the fused
  reduce stays. Lesson: a per-window cost inside a streaming block is
  paid by every block on every window; a launch saved is paid once.
- Kept from the same morning, measured together on one binary (the C1
  legs against the same session's baseline 34.48 / 38.97 / 28.49 ms a
  pass code / JSON / prose and the plain step's 20.0; the images rebuilt
  for the new layout, `loader_format()` 2): (1) the fused split reduce;
  (2) the deep cp.async ring (eight stages where the budget holds them);
  (3) the balanced expert slices (§2.3). Legs: code 33.73 at 3.11 tokens
  a pass (92.2 tok/s), JSON 38.50 at 4.25 (109.4, +2.6 %), prose 29.00 at
  1.93 (66.6), plain 18.93 (−5.4 %) — the slices move the TP partial sums,
  so the accepted rows per pass shift by a tolerance class and the
  comparison is per token. The trace diff against the morning's: rank 0's
  slot kernels −7 % a layer at ~3.5 rows (gate-up 291 → 272 us, down 155
  → 142) and the one-row gate-up −24 % (140 → 106 us at 4 bits, 122 → 89
  at 3: the deep ring lifts the 35-block grid from 72 % to ~95 % of the
  DRAM rate), the separate reduce launches gone (−0.57 ms); against
  that, rank 0's collectives +10 us each at ~3.5 rows and +17 at one row
  (it is no longer the rank every fold waits for: with the remainder
  blocks alternating, the heavy pair changes from layer to layer and the
  step pays the expected maximum of two binomials, 27.7 blocks a row
  against the old fixed 30 and an even 27), the dense GEMV +2.5 us a
  launch (the last block's split loop, bounded at run time, serialized
  its L2 round trips — unrolled next), the Engram stage wait +0.1. Net at
  C1 ~−2 % a pass, −5 % at one row; the expected-maximum tax is inherent
  to any contiguous whole-block split of 18 blocks over 4 ranks.
- Prefill, the same morning (cold probes, two repeats, the same prompts
  across legs; the campaign's prompts were shorter, so per token): the
  chunk policy is not a lever — a lone prompt already reaches the 4,096-
  token idle ceiling by its third tick (`prefill_idle_floor_tokens` 4096
  with `prefill_initial_tokens` 1024 moved nothing: 1,225 / 3,847 / 14,536
  ms against 1,225 / 3,887 / 14,513 at 2K / 8K / 32K). The 8K trace's
  dense sites ran 64-row A tiles (the weights re-read T/64 times, 0.74 s of
  4.1 s): the dense form now takes a 128-row tile above 64 rows (kMT 8, two
  stages in the smem budget; 8-bit launches keep 64-row groups), bitwise,
  measured −2.5 % at 8K and 32K (0.447 → 0.441 ms a token at 8K, 0.426 →
  0.414 at 32K; 2K 0.58 → 0.57; 512 0.83 → 0.81). The day's decode levers
  left prefill level per token (the base leg read +2–5 % at 8K against the
  campaign on 7 % longer prompts, level at 32K). Per token the pack now
  prefills at 1,760 / 2,270 / 2,420 tok/s at 2K / 8K / 32K: the
  short-prompt floor is the next prefill item (a 2K trace taken).
- The short-prompt prefill (2026-10-23, 2K cold probes): a 2,100-token
  prompt ran as two 1,024-token ticks (the first tick is
  `prefill_initial_tokens`, 256 by default, grown by the controller);
  `prefill_initial_tokens: 4096` (the busy limit still caps it, so it acts
  only with nothing decoding — prefill_policy.hpp) runs it as one tick:
  1,199 → 1,167 ms at 2K (−2.7 %), 3,791 → 3,755 at 8K (−1 %), 32K level.
  The merged chunk saved little because the routed experts a chunk reads
  are far fewer than all 384 on real text (the grouped gate launch at 4
  bits averaged 1.19 ms against a 2.7 ms floor for every expert's slice),
  so two chunks re-read less than twice the weights. The key goes into the
  template with the schedule re-fit. The 2K floor itself (0.55 ms a token
  against 0.41 at 32K: the per-walk costs — the folds' fixed part, the
  Engram gather, the rotate_segments copies, the attention over a short
  ring) is the open short-prompt item: a 3K prompt at 0.55 ms a token is
  ~1,800 tok/s against ~2,400 from 8K up.
- The decode batch's cap (2026-10-23, the aggregate past six streams):
  the family's `decode_rows_cap()` was 32 — six DSpark slots, since each
  slot's block walks five rows whatever the verify depth — and a
  sixteen-stream server is the shape to serve. Three bounds moved to the
  engine's 64
  (`kDecodeRowsMax`): `Dsv41Model::decode_rows_cap()`, the CSA2 layer's
  `max_decode_rows` check, and the DSA select workspace's layout rows
  (`kSelectLayoutRows`, the keys and histograms laid out for the cap); the
  memory plan at world 4 reads 81.33 / 81.38 / 81.48 GiB at 6 / 8 / 12
  slots (the balanced slices had taken rank 0 from 87.5 to 81.1), so twelve
  slots cost 0.15 GiB. Legs on one binary (decode tok/s, two repeats, code
  / JSON): six slots at C8 (six live, two queued) 126.7 / 134.5; eight
  slots at C8 153.6 / 179.7; twelve slots at C4 140.1 / 164.1, C8 154.0 /
  174.7, C12 167.1 / 180.9 (210 ms a pass for 38 tokens: 48 verify rows).
  The arithmetic of that regime: 48 rows pick 288 expert slots that cover
  ~203 of 384 experts, so with the weights read once per expert the step
  streams ~40 GB a rank — 171 ms at the DRAM rate — plus the dense stack
  and the fixed costs, ~190 ms; the measured 210 is within 10 % of that
  floor (a 48-row round of independently routed streams cannot run under
  ~150 ms even at the part's nominal 273 GB/s: the distinct experts'
  bytes alone). Past twelve streams the cap (64 rows) is the next
  wall: C16 at four rows a stream needs 64.
- Twelve slots against six at C1 (three repeats, the same binary): code
  34.81 / JSON 38.48 / prose 29.75 ms a pass against 33.89 / 39.04 / 28.99
  — within ±2 %, mixed in sign — so the template takes twelve slots
  (`max_concurrency` 12, `queue_limit` 16; with `prefill_initial_tokens`
  4096 for the lone prompt's first chunk). The C12 code trace on twelve
  slots (191.6 ms a step under the profiler, ~36 verify rows): the slot
  kernels 108 ms (2.86 ms a layer at 4 bits against a ~3.1 ms floor for
  the ~165 distinct experts those rows pick — the expert-sorted slot
  order lets L2 serve a row's duplicate of another row's expert, so the
  decode slot kernels sit at the deduplicated byte floor at high
  concurrency too), the collectives 22 ms (95 x 232 us: a ~400 KB row
  block per rank each, wire-bound on the host-proxy transport — a
  reduce-scatter / all-gather form would halve the per-rank wire bytes at
  this payload and lose at C1's 10 KB, a payload-keyed choice for a bus
  day), the attention 20 ms (the flash kernels and the finish scale with
  rows), the dense 128-row-tile GEMVs 7.8, mhc_dots 7.3, the prefetch
  windows 23 on the side stream.
- The verify schedule at twelve streams (C12, twelve slots, two repeats,
  decode tok/s code / JSON): the template's adaptive schedule 167.1 /
  180.9 (172 ms a pass at 28.8 tokens / 210 at 38.1), fixed depth 3 167.2 /
  194.5 (206 at 34.4 / 193 at 37.6), fixed depth 4 156.7 / 189.9. The
  schedule's constants are the release template's pass costs (base 31.1,
  5.2 ms a row); at twelve live it verified too few rows on code and too
  many on JSON (the same tokens for a 9 % longer pass) — the re-fit to the
  pack's curve (base 18, 4.3 ms a row) is measured next at C1 and C12.
- The re-fit measured (base 18 / 4.3 ms a row, twelve slots, two repeats):
  C1 code 89.7 / JSON 107.8 / prose 64.1 tok/s against the shipped
  constants' 90.4 / 110.5 / 65.9; C12 164.2 / 176.6 against 167.1 / 180.9 —
  no better anywhere, so the template keeps base 31.1 / row 5.2 (the
  schedule's lambda adapts to the measured pass, and the constants only
  shape its prior). A fixed depth 3 would take JSON at C12 (+7.5 %) and
  lose prose at C1 (59 against 67 tok/s): the adaptive schedule stays.
  Closed with numbers.
- The campaign of record on the finished template (2026-10-23 10:46–11:42
  UTC, binary `dde85abfaa953cc1`, source `a24e05b5` plus the port; every
  rank at zero swap before and after, zero cached tokens in the 12 cold
  probes, matching operation streams in 7/7 launches, the plain boot's
  transcripts identical to the default's in 5/5 classes, 9/9 API checks,
  isolation identical): decode 66.4–110.0 / 87.8–146.5 / 111.0–158.3 /
  136.8–174.8 tok/s at C1 / C2 / C4 / C8 against the morning's six-slot
  65.0–105.0 / 90.9–141.8 / 115.5–156.4 / 91.1–137.9 — C1 code 89.0, JSON
  110.0, math 101.6, prose 66.4 (35.35 / 38.64 / 35.87 / 29.54 ms a pass
  at 3.15 / 4.25 / 3.70 / 1.96 tokens); the C8 cell eight live streams
  against six live and two queued: prose 91.1 → 136.8, code 131.3 →
  158.5, JSON 137.9 → 174.8, math 119.3 → 162.2, chat 96.0 → 148.2 (the
  step 87.7–164.4 ms for 12.0–28.7 tokens); the plain one-row pass
  19.0–19.4 ms (20.0 in the morning); cold prefill 1.167 / 3.420 / 13.342
  s at 2K / 8K / 32K (1.225 / 3.489 / 13.607), the context matrix 35.34 →
  52.48 ms a pass from a short prompt to 512K (36.21 → 51.39), its prefill
  0.265 → 345.5 s (0.274 → 350.7); HumanEval 159/164, GSM8K 293/300,
  extraction 100/100. The rows replaced the morning's in docs/benchmarks.md
  ("1M compressed KV, 12 slots"); the card keeps both beside the release's.
  Open after the day: the fixed part of the step (the one-row pass 19.0 ms
  against its ~12 ms byte floor: 94 collectives at 48 us, ~7 ms of small
  kernels and gaps — the pipelined half-batch program's territory), the
  reduce-scatter / all-gather collective for the large payloads at C8–C12
  (232 us each, 11 % of a 48-row step), the Engram decode prestage
  (0.2–0.5 ms), the short-prompt prefill floor and the dense prefill GEMM
  form, the 64-row cap at sixteen streams.
- The expert-run slot form (2026-10-23 afternoon, §3.3). A cell method
  worth measuring beside the campaign's independent streams sends, at
  concurrency C, C copies of ONE prompt behind a per-stream tag (greedy,
  thinking off, 200-token answers; its aggregate is tokens over the batch
  wall with the first token's wait in — the wall table's definition),
  so its streams route identically. A scratch replica of that cell method
  on our endpoint (nine prompts, C1..C16, two repeats) read, on binary dde85abf, a step of 166 ms at C8 and 240 at
  C12 on the counting prompt — the one whose eight streams stay identical
  token for token — against 38 at C1: the same distinct experts, 4.3x the
  time. The C8 trace of that cell (rank 0, nsys): the slot kernels 90 ms a
  step against 18.6 at C1 (gate-up 1,309 us a layer against 300, down 855
  against 160), the collectives 22.5 (95 x 235 us), the attention 19, the
  dense 128-row GEMVs 10 — so a duplicate of an expert across slots was
  streamed and trellis-decoded again per slot, L2 serving under half. The
  per-slot kernels had been at the distinct-expert byte floor only where
  duplicates come in pairs (C12 independent streams, 216 picks on ~165
  experts), not where they come eight deep. The form built: the order
  kernel emits each position's equal-expert run length, the rotate kernel
  lays rows out by position, the run's first block takes its rows as one
  A tile (16-row tile below 17 rows, 64-row tile above — a second launch
  only when the batch has more than 16 token rows), the duplicates' blocks
  return, the down rows scatter to their logical slots; bitwise the
  per-slot form per row (the MoE test's decode case at 17 / 40 / 64 rows
  with 4-8 experts, every width pair; the dsv41 suites; 54/54). Measured
  on one boot (binary 6f8bccfa): the campaign-method legs C1 code 35.35 →
  32.18 ms a pass (97.8 tok/s, +10 %), JSON 38.64 → 35.35 (120.2, +9 %),
  prose 29.54 → 28.51 (68.8, +4 %) — the shared expert's five-row run and
  the draft block's own duplicates read once; C8 code 139.9 → 122.2 ms a
  step (158.5 → 191.1 tok/s), JSON 164.4 → 117.9 (174.8 → 227.9), prose
  87.7 → 73.6 (136.8 → 166.9). The cell method, mean of the eight prompts,
  engine decode / their aggregate: C1 94.8 / 82.9 (88.7 / 78.3 before), C2
  144.8 / 125.7 (125.3 / 111.5), C4 208.1 / 180.0 (157.5 / 141.0), C8 295.3
  / 250.6 (184.2 / 166.0), C12 328.5 / 277.6 (192.0 / 173.6), C16 242.0 /
  206.8 (151.3 / 136.8: twelve live and four queued, 2.2 s to their first
  token); the counting cell's step 92 ms at C8 (166 before) and 118 at C12
  (240). The C16 cell is the twelve-slot template's wall: sixteen live
  streams need an 80-row decode batch (five rows a slot), and a run longer
  than 64 rows (the shared expert's) needs the run split into 64-row
  groups — the next lever. The campaign of record re-ran on this binary
  (12:56–13:45 UTC, every gate green; the rows in docs/benchmarks.md):
  decode 70.6–119.3 / 96.7–170.3 / 126.4–183.7 / 167.7–224.0 tok/s at C1
  / C2 / C4 / C8 (C1 prose / code / JSON / math / chat 70.6 / 96.2 / 119.3
  / 111.3 / 70.9; C8 167.7 / 200.8 / 224.0 / 208.3 / 177.9) against the
  morning twelve-slot campaign's 66.4–110.0 / 87.8–146.5 / 111.0–158.3 /
  136.8–174.8; the plain pass 18.9 ms; prefill 1.166 / 3.398 / 13.326 s
  (level); the context matrix 33.18 → 50.58 ms a pass to 512K (35.34 →
  52.48); quality 159 / 293 / 100 (unchanged).
- The multi-row program (2026-10-23 evening, §3.3's run groups and §3.6).
  The forty-row trace of the counting prompt's identical streams named the
  step's cost past a few rows: the attention's split partials 20 ms a
  step (the window and main flash kernels writing fifty megabytes a
  layer, the finish reading them: 8.2 + 3.8 + 8.1), the collective 11.8
  (123 us each: 100 us of transport for 423 KB a peer at ~12.7 GB/s a
  direction, the ranks' skew under 1 us on identical work), the 64-row
  dense tile 8.5 (32.8 us a launch against 18 at one row), the mHC dots
  6.9 (80 us a site), the 64-row slot launches 9.6 doing nothing (1,400
  returning blocks), the router 3.3 (76 us). Built: the run-group launch
  list, the 96-row decode batch, the attention split groups (first as
  thread-block clusters — the kernels ran 50 / 85 us against 20 / 19 at
  five rows, C1 +2 ms a step, and the standalone bench showed the merge's
  barriers losing 35 us a layer at ten to twenty rows — then as the
  last-block merge in L2 with the dead rows discarded), the tiled router
  and mHC dots from sixteen rows. Two defects the gates caught and the
  bisections named: the discard racing the merge's own reads (a thread
  ahead discarded lines a thread behind was loading — the group prefill's
  first-step gate and the 40-step budget; the unit test and the memory
  checker had passed on timing) and, at sixteen slots on the release
  fixture only, the tensor-core GEMV's 65–128-row form launched unsplit
  (the dispatch trace: fp8 CSA2 sites at m = 80 on the eight-tile form;
  fourteen slots matched, fifteen were flaky, sixteen diverged at slot 0's
  fourth token). The incast probe (`ib_write_bw`, 212 KB writes into one
  node): one flow per PCIe function 105 Gb/s, three into one function 112
  aggregate, the bus's six-flow pattern 184 Gb/s — forty rows' 1.27 MB
  lands in ~55 us of wire and the collective's other ~45 us is fixed
  cost, so a reduce-scatter form (a second fixed round for half the bytes)
  is closed on the numbers at these payloads. Measured on binary
  `0196c9a4ca11f004` (one boot per template): the campaign-method legs C1
  code 32.09 / JSON 35.12 / prose 28.59 ms a pass (the morning's 32.18 /
  35.35 / 28.51 — level at equal rows), C8 code 122.2 → 111.5 ms a step
  (191.1 → 206.1 tok/s), JSON 117.9 → 112.3 (227.9 → 240.7), prose 73.6 →
  72.2; the forty-row counting step 91 → 82.5 ms and the sixty-row 118 →
  102 (the second trace's reconciliation: the finish 8.1 → 0.5 ms a step,
  the router 3.3 → 2.1, the down's empty launch 4.1 → 1.8; the mHC tile
  level — ten blocks of four tokens latency-bound where the fused form
  was L2-feed-bound — and the gate-up's long-run launch unchanged because
  its cost is the shared expert's forty-row run streamed by its five
  output blocks, not the empty ones: the next two multi-row levers, with
  the 64-row dense tile's rate); on the sixteen-slot template sixteen
  identical streams decode at 128 ms a step (the first token in 0.7 s)
  where the twelve-slot template queued four. The sixteen-slot
  template's C1 ran 2 ms a step slower until the variant budget kept its
  third depth option (§3.6). The C1 trace on the finished binary: 35.2 ms
  a step on the counting prompt (35.4 before the program), the two
  attention kernels +10 us a layer against the flat form (the group's
  last block reading 512 KB from L2 — 16 loads in flight a thread since)
  against the finish's −6 and the write-back the discard spares.
- **The fifth draft (2026-10-23, late evening; §3.6).** The five-draft
  block with the schedule's three options (1, 3, 5), measured on the
  sixteen-slot binary: C1 tokens a step on
  the per-class legs counting 4.96 → 5.95, code 4.37 → 5.17, tables 4.73
  → 5.20, JSON 3.45 → 3.60, math 3.83 → 4.23 (prose and narrative stay at
  2.25 and 1.88 — the schedule holds them at one to three rows), the
  eight-class C1 per-stream mean +1.5 %, two streams +2.7 %; campaign C1
  JSON 120.8 → 133.1 tok/s, code 91.1 → 92.0, prose 68.5 → 67.2 (the
  prose and chat passes unchanged at 29.3–29.5 ms, four or five drafts
  alike); the eight-class means C1 agg 85.7 → 86.6 (per-stream 98.2 →
  99.7), C2 130.6 → 134.2. The attention split floor measured the same
  night is recorded in §3.6 and withdrawn (a C1 prose transcript that
  drafts 7 % less). The campaign of record on the shipped template
  (binary `9a179b15a6de02fa`, 2026-10-24 00:06–00:54 UTC, all gates
  green; the record `dsv41-1017-exl3-3.5bpw-16slots-d5_2026-10-23`):
  C1 prose 70.7 / code 92.0 / JSON 134.2 / math 111.4 / chat 70.1 tok/s
  (the depth-4 sixteen-slot campaign on the same binary: 71.0 / 91.6 /
  121.5 / 112.1 / 71.5), C2 95.7–182.4, C4 127.8–206.6, C8 165.0–247.7,
  cold prefill 1.188 / 3.446 / 13.331 s, context 34.70 → 52.38 ms a pass
  to 512K, HumanEval 160/164, GSM8K 295/300, extraction 100/100. The rows
  are in docs/benchmarks.md and the card's Measured section.
  **Where the one-stream step's time sits now** (the C1 traces of
  2026-10-23 at one and five to six rows, the kernels on the main
  stream, which is kernel-time-bound: its busy time equals the step):
  at one row (18.9 ms) the slot experts 5.6 ms, the dense GEMVs 4.9
  (251 launches a step at 17 µs), the collectives 3.8 (84 at 45 µs), the
  mixes 1.6 (three launches a mix: dots 8.5 µs, comb 8.9, update 2.2),
  attention 1.4, the router 0.4, rotations 1.1; at five to six rows
  (38 ms) the experts 15–17, dense 6, collectives 4.4–5.5 (95 at 46–58
  µs), mixes 2.2, attention 2.5 (before the floor), the draft stages
  and picks ~1.5. The fixed part — what the step costs beyond its
  expert and dense bytes — is the collectives' per-call latency and the
  mixes, and both have physical floors well below where they sit: a
  one-hop RDMA write lands in 2.9 µs plus 0.11 µs a KiB (the fabric's
  transport floors, measured 2026-10-19; the bus's claim, post, doorbell
  and acknowledgement round trips are the 45 µs), and a mix's dots are
  2 MB of weights in L2. The levers, in the order of their size at
  one row, each with its mechanism: (1) the collective's fixed latency —
  a pre-posted receive ring with sequence numbers in place of the
  per-collective claim and acknowledgement, the fold done by the
  consumer kernel as it reads, the consumer launched as a programmatic
  dependent so its wait overlaps the producer's tail: ~2 ms a step at
  every row count; (2) the mix's stream dots issued on the side stream
  under the collective and only the gathered partial's contribution
  added after it (the dots are linear in the streams; a rounding-class
  change for every token, so a gate with a control): ~1 ms; (3) one
  attention kernel a layer over both key sources (the window ring and
  the selected keys as two segments of one list, the finish folded into
  the last block's merge): ~0.7 ms at one row, ~1 ms at six; (4) the
  per-layer dense projections as one grouped launch (wq_a, wkv and the
  compressor; wq_b and the indexer's) instead of 6.3 launches a layer:
  ~1 ms; (5) the short prompt's prefill — 150–190 ms for 47–67 tokens,
  GPU-busy throughout, its bytes the touched experts' — and the first
  token's emission at admission rather than after the first step.
