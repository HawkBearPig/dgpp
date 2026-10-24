# Qwen3.8-27B-FP8 on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) — the
dense 27B of the Qwen3.8 line (`Qwen3_5ForConditionalGeneration`, text
only here) — and, on one Spark, the
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
block drafter beside it. Neither is a republished checkpoint: the engine
loads both releases as shipped.

## The checkpoint, as loaded

28.8 GiB across 66 shards. Every projection — the Gated-DeltaNet layers'
qkv / z / out, the full-attention layers' q / k / v / o, the dense MLPs'
gate / up / down — is FP8 e4m3 with BF16 scales on 128 × 128 blocks. The
embedding, the lm head, the norms, the DeltaNet's `A_log` / `dt_bias` /
conv taps / `in_proj_a` / `in_proj_b`, and the MTP draft's `fc` are BF16.
The engine decodes every FP8 code exactly into BF16 (`bf16(code × scale)`,
one rounding) and runs activations in BF16 with fp32 accumulation. The
default chain is exact; the two lossy levers are explicit keys
(`engine.dense_weights: fp8` requantizes the lm head to block FP8 — the
templates ship it, half the head's bytes a decode row at the same
HumanEval / GSM8K; `engine.prefill_fp8_per_tensor` runs the prefill GEMMs
on cuBLASLt's per-tensor e4m3 kernels — off, it changes long-context
transcripts). The Full layers' K/V (four kv heads of 256 at world 1) are
cached as e4m3 codes at unit scales (`engine.kv_dtype: "fp8"`, 2026-10-18:
the release ships no `kv_cache_scheme`, so the codes are the e4m3 rounding
of K and V, the common scale-free fp8 cache — 1 KiB a token a kv head
against 2 in bf16, which `"bf16"` restores) with the prefill attention on
the fp8 tensor cores (`engine.prefill_attention: "fp8"`); see Measurements.

## Architecture

64 layers at hidden size 5120: 48 Gated-DeltaNet layers (16 key heads and
48 value heads of 128, a 4-tap causal conv, the swish output gate, an fp32
recurrent state) and 16 full-attention layers (24 query heads and 4 kv
heads of 256, a `[q | gate]` projection with a sigmoid output gate,
rotate-half RoPE over 64 of the 256 dims, θ = 10⁷), each followed by a
SwiGLU MLP of 17,408; a vocabulary of 248,320; one MTP draft layer (a full-
attention layer fed `fc([norm(embed(next)) | norm(hidden)])`). The context
is 262,144 tokens. The DFlash2 drafter is a five-layer Qwen3 backbone (32
heads of 128, 8 kv heads, MLP 17,408) fed the target's residual at five
taps through a split `fc`, two-tap dynamic grouped convolutions around its
attention and MLP, and a rank-256 top-16 selector walk that proposes a
block of seven drafts from one anchor.

## Serving

**One Spark** (`cluster_qwen3.8-27b_fp8_w1`): the drafter recipe. 27.5 GiB
of weights, a 786,432-token fp8 K/V pool (39.0 GiB, three full contexts),
the 3.6 GiB drafter and its five K/V planes, eight request slots and a
20 GiB prefix arena, the plan at 100 GiB of the node's 116 with the
replicated BF16 image tower (sized to the node on 2026-10-17 — a world-1
boot ends 6–8 GiB below its plan at listening, so the recipe leaves that
room; a 256K bf16 pool and a 1.5 GiB arena at 62 GiB until then, a
524,288-token bf16 pool for a day); the graph engine records the block proposal inside the decode
step (the 8-row verify and the drafter's forward in one graph, as on the
fabric — the template since 2026-10-05; the eager engine's transcripts 4/4,
chat 71 against 95 ms a token), the drafter's five block matrices served as
block FP8 (`engine.dflash_weights: fp8` — lossy for the proposals only: the
verify is exact, so transcripts and sampling distributions are unchanged;
MT-Bench 3.53 against 3.52 tokens a pass, 145.5 → 142.2 ms a pass), its
other bf16 matrices packed lossless 12-bit. The MTP depth-2 world is the template's mode (`--no-dflash --mtp
--mtp-depth 2`), plain decode `--no-dflash`, the eager engine
`engine.decode_graph: false` in the recipe.

**Two and four Sparks** (`cluster_qwen3.8-27b_fp8_w2`, `_w4`): tensor
parallel — the DeltaNet and attention heads, the MLP and the lm head sliced
across the ranks, two boundary folds a layer (the attention / DeltaNet
output and the MLP output, bf16 on the wire) — the DFlash2 drafter with
the decode graph (since 2026-10-05; MTP depth 3 was the template until
then): the block proposal recorded inside the graph step on every rank, the
drafter's heads and MLP rows sharded across the ranks, the ranks' top-16
lists merged through one fold ([#92](https://github.com/HawkBearPig/dgpp/issues/92)),
a sampled request's drafts drawn at 0.7 of its temperature under the ratio
verify, the drafter's block matrices as block FP8 and its other matrices
packed lossless 12-bit, the verify depth scheduled a step at a time from
the selector walk's confidence (`engine.mtp_schedule`: eight slots on four
Sparks 205 / 301 / 352 / 310 / 216 tok/s at 92 ms a step with the one-chain
split-K of 2026-10-05, 184 / 273 / 326 / 284 / 196 at 106 before it, 154 /
270 / 320 / 260 / 170 at 117 for whole blocks; C1 level; transcripts matched
across the tested isolated, scheduled-depth and joining-co-tenant cases); 15.0 GiB of weights
and a 52 GiB fp8 K/V pool a rank (2,097,152 tokens, one full context per
slot, full admission) with a 24 GiB prefix arena on two Sparks, eight
slots, 102 GiB planned a rank; on four Sparks the same pool at 26 GiB a
rank with a 48 GiB arena, 92 GiB planned (since 2026-10-18; 256K bf16 pools
and 1.5 GiB arenas until 2026-10-17). The MTP
depth-3 world is available as an alternative to each template's DFlash
default (`--no-dflash --mtp --mtp-depth 3`),
plain decode `--no-dflash`. Greedy C1 by class (prose / code / json / math /
chat), the 2026-10-05 campaign: the drafter 35.4 / 55.3 / 76.6 / 62.6 / 32.9
tok/s on two nodes at 82 ms a pass and 57.4 / 88.4 / 122.1 / 101.5 / 53.4 on
four at 50 ms; MTP depth 3 34.2 / 40.3 / 46.0 / 42.4 / 31.2 at 78 ms and
57.7 / 68.7 / 78.1 / 72.3 / 50.5 at 45 ms — the drafter ahead on code, JSON
and math by 29–56 % on four nodes, level on prose, 6 % ahead on chat, and
ahead on sampled traffic ([docs/mtp.md](../mtp.md#on-the-graph-worlds-2026-10-04-92)).

Both worlds decode through the family's own kernels: the DeltaNet chunked
and recurrent forms, the query-tiled prefill attention and split-KV decode
walk over the 256-wide heads, the fp8 GEMVs with the k = 17,408 down
projection on the streaming tensor-core form; the lossless 12-bit form of
the bf16 decode weights (`engine.bf16_weights: bf12`) packs the drafter's
layers, its fc taps and the MTP fc (−3 % a step at C1, bitwise).

## Numerics and gates

`qwen35_forward_test`: a tiny synthetic block-FP8 release from the binding
table against a numpy double reference of the whole stack and the draft
block (teacher-forced strict: per-layer residual l2 0.17–0.44 %, the final
hidden bitwise, the top-8 logits exact to 2.4e-7). `qwen35_tp_test`: worlds
2 and 4 in one process against the world-1 forward (cross-rank bitwise,
merged top-1 exact on every row). T=1 and MTP greedy transcripts are
identical within a world; the drafter world's equal an MTP depth-4 world's
(the same verify width) at world 1 and an MTP depth-5 world's at worlds 2
and 4 (4/4 each). Across worlds the bf16 wire fold moves near ties
(one of four 256-token prompts identical between worlds 1 and 2); HumanEval
39/40 and GSM8K 39/40 at both.

## Measurements

The serving numbers — single-request engine rates by prompt class,
concurrency sweeps, cold prefill at 2K / 8K / 32K, the decode modes and the
full HumanEval / GSM8K / schema evaluation — are the
[benchmarks page](../benchmarks.md)'s Qwen3.8-27B rows, re-measured on
2026-10-16 on the templates as shipped.

`engine.prefill_activations: "native"` (set by the FP8 templates since
2026-10-16) runs the checkpoint's own blockwise fp8 product on every
prefill row — per-token 1×128 activation groups against the 128×128 weight
grid, the recipe the release was made for — in place of dequantizing each
FP8 matrix into a bf16 bridge for the bf16 GEMM. The bridge chain was
measured 20 % slower on one Spark under the adaptive prefill defaults
(every 4,096-token chunk re-dequantized every matrix: 2.70 / 8.30 / 35.3 s
cold prefill at 2K / 8K / 32K against 2.22 / 6.96 / 29.5 under fixed
one-pass budgets); the native recipe reads 1.90 / 6.49 / 27.9 s with no
bridge at all, and the 300-item HumanEval / GSM8K / schema set reads
161/164, 294/300 and 100/100 under both chains on the same binary, needles
at 32K and 128K 6/6 with 3/3 controls.
`engine.kv_dtype: "fp8"` with `engine.prefill_attention: "fp8"` (set by
the FP8 templates since 2026-10-18) caches the Full layers' K/V as e4m3
codes at unit scales — this release ships no `kv_cache_scheme`, so the
codes are the plain e4m3 rounding of K and V — and runs the prefill
attention on the fp8 tensor cores. Gated on one Spark against the bf16
cache (one boot each, the same seed): the teacher-forced probe (128
positions of a fixed continuation on a 115K-token document, the top-5
logprobs at every position) puts the fp8 pool's next-token argmax at
127/128 of the bf16 cache's with a mean KL of 0.0054 nats (median 0.0009)
and 127/128 at 0.0055 (median 0.0017) with the fp8 attention on top, the
reference token never outside the fp8 top-5 (the mixed release's
calibrated scales read 126/127 and 119/128 at 0.015 and 0.039 on
2026-10-16, the bf16 boot-to-boot floor 124/127 and 122/128); five
needles per document at 32K and 128K over two seeds 20/20 and at 258K 5/5
in every mode, with every control found; the 300-item HumanEval / GSM8K /
schema set 161/164, 294/300, 100/100 on the bf16 cache, the same on the
fp8 pool, 159/164, 295/300, 100/100 with the fp8 attention (the campaign
legs' run-to-run band is 158–161 and 292–296). Cold prefill of the needle
documents at 32K / 128K / 258K: 28.5 / 150 / 387 s on the bf16 cache,
29.1 / 160 / 425 on the fp8 pool under bf16 arithmetic (the codes widened
in shared memory), 27.3 / 132 / 320 with the fp8 attention; the 262K
bucket 395 → 325 s; decode at 256K 216 → 181 ms a pass in both fp8 modes
(half the K/V bytes), 137 → 135 at a short context.

Physics on one GB10 (248 GB/s): a T=1 step streams 27 GB — 105 ms — and
measures 114; at world 2 a rank streams 12.2 GB (49 ms) plus the head's
reads and measures 63 T=1 / 74.5 an MTP pass, GPU-busy 98.9 % with the 135
boundary folds a pass at 4.2 ms
([#90](https://github.com/HawkBearPig/dgpp/issues/90)).
