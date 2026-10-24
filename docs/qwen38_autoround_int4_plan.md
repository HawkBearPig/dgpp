# Qwen3.8-Flash-Next AutoRound int4 hybrid on one Spark — implementation plan (2026-09-28)

Checkpoint: `Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN`
(revision `19f9710c`, 66.7 GiB) plus the n-gram table served from
`Qwen/Qwen3.8-Flash-Next-FP8`'s shards (or the byte-identical
`Saren/Qwen3.8-Flash-Next-ple-table-fp8`, 48.7 GiB).

Goal: serve it on a single Spark at world 1 at the decode throughput its
bytes allow, and pass the same correctness and accuracy gates the NVFP4
single-Spark template passes.

Status (2026-09-28, late): S1 done (the packed core at group 128 / f16
scales, both widths, the slot bench at the Qwen geometry beats the fp4
core at 1 / 2 / 4 rows); S2 done to the fixture gates (config, binding,
the GPTQ repack, the packed head, the shipped fp8 dense classes read as
is, the pure-Python reference extended; the tiny hybrid fixture through
the forward parity, the decode gates and the loopback engines); S4
plumbing written (engine.ngram_table_model, the loader's second shard
directory, the recipe, download and doctor); the regression run on the
full build passes 71 of 71 across the packed cores, the shared MoE layer,
full GLM-5.3's loader and every Qwen suite; S5 begun: the real checkpoint
binds and loads in streaming mode beside the production server, and eight
of its fp8 dense matrices are bitwise our own at-load encode of NVIDIA's
BF16 originals (the 2026-09-28-qwen-autoround-int4 measurements). The
resident load, the serve gates and S6 wait for an idle Spark. One fact
the real checkpoint corrected: its draft layer keeps q/k/v/o and the
shared expert in BF16 (only the 48 backbone layers' side layers are fp8).

## 0. Summary

**What the checkpoint is.** Intel's AutoRound W4A16 of the BF16 original
(routed experts int4, group 128, symmetric, 200 tuning iterations), re-tagged
as GPTQ, plus three CPU passes by the author: the lm_head to int8 GPTQ g128,
the GDN / QSA / shared-expert projections to block-FP8 128x128 with F32
`weight_scale_inv` (our own at-load recipe, done offline), and the MTP draft
layer's 512 experts to int4 g128 RTN in the same layout. The n-gram table is
not in the repo. Everything else (embeddings, hyper-connections, routers,
indexer, GDN a/b, PLE projections, MTP fc/attention/shared expert, norms) is
BF16 as the original ships it.

**What dgpp has.** No GPTQ reader anywhere. The nearest resident form is the
full-GLM `GlmPackedMatrix` (compressed-tensors pack-quantized: codes packed
along K per output row, bf16 scales per 64) and its `packq` GEMV / slot /
tile kernels. GPTQ's word `qweight[k/8][n]` is the packq word `[n][k/8]`
with the same nibble order and the same fixed-offset code semantics (value =
(code − 8)·scale; GPTQ v1 stores 7 in `qzeros` and adds one), so the repack
is a pure I32 transpose. The differences that need code: group 128 and F16
scales (the cores are hard-wired to 64 and bf16), the two Qwen widths (640
and 2560) missing from the compiled K set, an `experts_packed` binding on
the Qwen MoE, a pre-encoded FP8 reader for the dense classes, a packed int8
lm_head (no family serves one), and a table directory outside the checkpoint.

**The principle.** Every resident value is exactly the checkpoint's
dequantized value: int4 / int8 codes untouched (transposed), F16 scales
untouched (converted to fp32 in the kernel, exact), `qzeros` verified to be
the constant symmetric zero and dropped, FP8 codes and F32 scales untouched.
The only freedom is summation order, as for every family. The cheap
alternative (replicate each g128 scale into two g64 bf16 scales) is rejected:
it rounds every scale by up to 2^-9 and costs 3 % more expert bytes.

**The target, decoded.** The checkpoint's own five-prompt bench
(`bench_qwen35.sh`, §6.1) is greedy (`temperature: 0.0`), thinking on,
single stream, completion tokens divided by the whole request's wall time
(TTFT inside), MTP depth 3. Our NVFP4 template today, greedy with thinking
off: code 49.0 at depth 1, up to 60.7 at depth 2 (§2.3). A single stream
past 60 tok/s on these prompts needs at most 16 ms per token, which no
depth-2 pass can reach; it needs depth 3 (already implemented for every
family) at a pass within a few ms of its byte floor and a chain whose
acceptance is the model's. The int4 experts themselves are worth about
1–1.5 ms of the pass; the checkpoint is not what sets the pace.

**The standard.** We serve the same bytes, so we are faster unless our
engine leaves an operation short of its physical limit. Every such
shortfall is a defect. §6 states the floor of every per-pass operation and
the measured gap; §7 is the defect list with the fix for each. There is no
cell of the table we accept losing.

**Stages.** S1 kernels (group / scale-type parameters, the two widths, a
packq slot bench), S2 config + binding + loader (GPTQ repack, fixture,
tests), S3 pre-encoded FP8 dense + packed int8 head, S4 MTP experts, table
directory, memory plan, resident image, recipe, download, S5 real-checkpoint
gates on one Spark, S6 the throughput program to the target, S7 ship.
S1–S4 need only a GPU beside a serving process; S5–S7 need an idle Spark.

## 1. The checkpoint

### 1.1 Tensor census (227,526 tensors, 66.6 GiB)

| class | tensors | form | GiB |
|---|---|---|---|
| routed experts, 48 layers (512 x gate/up/down) | `qweight` I32 [K/8, N], `scales` F16 [K/128, N], `qzeros` I32 [K/128, N/8] | int4 g128 sym, AutoRound | 56.25 codes + 1.76 scales + 0.44 zeros |
| MTP layer experts (512 x 3) | same layout | int4 g128 sym, RTN (`tools/quantize_mtp_experts_int4.py`) | 1.17 + 0.04 + 0.01 |
| GDN in_proj_qkv / in_proj_z / out_proj (36 layers) | F8_E4M3 + `weight_scale_inv` F32 [N/128, K/128] | block FP8, amax/448 (`tools/fp8_convert.py`) | 1.93 |
| QSA q / k / v / o_proj (12 layers) | same | block FP8 | 0.56 |
| shared expert gate / up / down (48) | same | block FP8 | 0.22 |
| lm_head | `qweight` I32 [K/4, N], `scales` F16 [K/128, N], `qzeros` I32 [K/128, N/4] | int8 g128 full-range sym (`tools/quantize_lm_head_int8.py`) | 0.59 + 0.01 |
| hyper-connections (layers + mixers), routers, indexer, GDN a/b, PLE key/value + conv, embeddings, the draft layer's q/k/v/o and shared expert, MTP fc / norms | BF16 | as the original (the fp8 pass covered the 48 backbone layers only) | 3.0 |
| vision tower | BF16 | not served | 0.84 |
| n-gram table | not in the repo | see §1.3 | — |

Shapes per expert: gate/up `qweight` (320, 640) = [2560/8, 640],
`scales` (20, 640), `qzeros` (20, 80); down `qweight` (80, 2560) = [640/8,
2560], `scales` (5, 2560), `qzeros` (5, 320). The head `qweight` (640,
248320) = [2560/4, V], `scales` (20, V), `qzeros` (20, 62080). No `g_idx`
anywhere (`desc_act: false`).

### 1.2 The three recipes and what "exact" means for each

- **int4 g128 (both AutoRound and the RTN drafter):** value = (nibble − 8) ·
  scale, scale F16 per (group of 128 along K, output n). `qzeros` are the
  constant `0x77777777` (zero point 8 stored as 7). AutoRound tuned the
  rounding direction of each code and the scale; the stored codes and scales
  are the whole result. The loader verifies every `qzeros` word and refuses
  a checkpoint whose zeros are not the symmetric constant.
- **int8 g128 head:** value = (byte − 128) · scale, `qzeros` constant
  `0x7F7F7F7F`, scale = w[argmax |w|] / −128 per group (the largest element
  lands on the −128 slot).
- **block FP8 side layers:** scale = amax(block) / 448 in fp32, code =
  e4m3(w / scale) with clamping to ±448 (torch round-to-nearest-even).
  This is `loaders/fp8_quant.hpp`'s recipe to the letter, so the codes and
  scales are expected bitwise equal to what `dense_weights: "fp8"` computes
  at load from the BF16 tensors of the NVFP4 or FP8 releases (§5.5 makes
  that a gate: the dense stack of this checkpoint is then quality-identical
  to the one we already evaluated).

### 1.3 The n-gram table

`Saren/Qwen3.8-Flash-Next-ple-table-fp8` is shards 5–37 of
`Qwen/Qwen3.8-Flash-Next-FP8` verbatim (the whole layer-2 shards, so they
also carry layer 2's FP8 experts, GDN weights and norms). The table
tensors are the ones our mmap loader already reads:
`model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_S.weight`
(128 parts of [2,500,012, 160] E4M3) and `...ngram_embedding.weight_scale`
(BF16). Every fabric node already holds the FP8 snapshot, so no download is
needed here; a site without it downloads the 48.7 GiB table repo.

### 1.4 The `quantization_config`

```
quant_method gptq, bits 4, group_size 128, desc_act false, sym true, lm_head true,
dynamic: +:.*lm_head$ → bits 8; −: linear_attn, self_attn, hyper_connection,
         visual, shared_expert, \.ple\., embed, fc_hidden, \.gate$
```

The original AutoRound block is kept as `config.json.autoround`
(`packing_format auto_round:auto_gptq`, iters 200). The base hybrid repo
(BF16 MTP experts, `−:.*layers\.48\..*`) is out of scope: the engine
refuses it with a message naming this variant.

## 2. Bytes, the floor, and the target at world 1

### 2.1 Resident

| class | NVFP4 template (fp8 dense) | this checkpoint |
|---|---|---|
| routed experts, 48 layers | 63.3 | 58.0 (codes 56.25 + F16 scales 1.76; zeros dropped) |
| MTP experts | 1.2 (fp8) | 1.2 (int4 + scales) |
| GDN / QSA / shared, fp8 | 2.7 | 2.7 |
| GR sites (bf16 + bf12 companions), routers, indexer, PLE proj, embed, fc | ~4.4 | ~4.4 |
| lm_head | 0.6 (fp8) | 0.6 (int8) |
| total weights | ~72 | ~67 |

About 5 GiB less on the board. The one-node recipes keep `prefix_cache_gib`
at 3 and leave free memory to the page cache of the mmap'ed table, which
is what fresh-text prefill speed depends on (the 2026-09-28 field report).

### 2.2 Decode bytes per token and the pass

Per token at one row, fp8 dense: routed experts 1.24 → 1.15 GiB
(−7.6 %), everything else unchanged (GDN 1.95, QSA 0.63, GR 1.23, head
0.59, shared + routers 0.34). The T=1 floor moves from 25.5 to about 25.1
ms; the measured 30.3 ms pass would move by roughly 0.4 ms.

Under MTP the routed experts are the class that scales with rows: at 2 rows
they are 10.3 ms of the 40 ms pass (about 2.7 GB, at line rate on the fp4
core), so int4 saves about 0.8 ms there and about 1.5 ms at the 4 rows of
depth 3, where the expert class is roughly 18 ms.

The pass model at world 1, from the 2026-09-10 profile and the measured
depth-2 delta (all on the NVFP4 template, fp8 dense):

| depth | rows verified | ms per pass | note |
|---|---|---|---|
| plain (no MTP) | 1 | 30.3 | floor 25.5 |
| 1 | 2 | 39.5 | 2.5 ms of it is the draft head's full-vocabulary product |
| 2 | 3 | 48 | measured: +8 ms for the second chained draft |
| 3 | 4 | ~56 | extrapolated: +8 ms again; the fourth verify row still fits one 4-row GEMV chunk |

Each chained draft step is a QSA layer plus a 512-expert MoE at one row
(tens of MB) plus a full lm_head product (636 MB, ~2.5 ms) plus the pick,
serial. At depth 3 the three draft heads are about 7.5 ms of the pass.

### 2.3 Where we stand (NVFP4 W1 template, 2026-09-22 campaign, greedy, thinking off)

| class | engine tok/s depth 1 | ms per pass | tok per pass | TTFT ms |
|---|---|---|---|---|
| prose | 45.5 | 39.2 | 1.78 | 102 |
| code | 49.0 | 39.5 | 1.93 | 103 |
| json / math / chat | 50.0 / 47.5 / 42.3 | | | |
| deeper (depth 2), all classes | 44.0–60.7 | | | |
| plain | 32.7 | 30.5 | 1 | |

## 3. Design decisions

**D1. One resident form: packq with group 128 and F16 scales.** The packq
core's group and scale type become template parameters
(`kGroup ∈ {64, 128}`, `Scale ∈ {bf16, f16}`), instantiated for {64, bf16}
(full GLM, bitwise as today) and {128, f16} (this checkpoint). The
chunk-in-one-group property holds (a lane's 16-byte chunk is 32 int4 or 16
int8 codes; 128 is a multiple of both), so only `chunks_per_group`,
`scale_cols`, `k_supported` and the scale read change; the FMA chain does
not. The tile GEMM's scale index becomes `group · 64 / G` with row stride
K / G. `GlmPackedMatrix` and `MoeExpertView` gain `group` and `scale_f16`
fields; every launcher validates them against the compiled set.

**D2. Load-time transpose, no zeros, no scale rounding.** The loader reads
`qweight` [K/8, N] and writes packq words [N, K/8] (an I32 transpose;
nibble order identical), transposes `scales` [K/128, N] to [N, K/128] as
F16, checks every `qzeros` word against the symmetric constant and stores
nothing for them. The CPU transpose in the `WeightBuilder` (16 threads,
cache-blocked) is the first cut; the resident image captures the repacked
form, so only the first boot pays. A device transpose kernel is the
fallback if the cold boot grows by more than a minute over the NVFP4
template's 77 s.

**D3. Both routed-expert sets bind through the shared MoE layer.**
`QwenMoeWeights` gains `experts_packed` (a third exclusive form beside
`experts` and `experts_fp4`), `routed_view` copies it into `GlmMoeWeights`,
`GlmMoeLayer` dispatches to the existing packq slot / grouped / tile paths
with `n_shared_experts = 0` (Qwen's shared expert stays on its own fp8
chain, so the "packed routed needs packed int8 shared" rule never fires).
The MTP layer's experts take the same path.

**D4. Dense classes read as shipped.** The binding accepts the GDN / QSA /
shared-expert projections in either form (BF16, or F8 payload + F32
`weight_scale_inv`), decided per checkpoint by probing the first GDN layer's
`in_proj_qkv` before the expectation table is generated. Pre-encoded
matrices load through `load_quant_rows / cols`, which already take F32
scale rows; the GDN qkv segment merge gets a quantized twin (segments are
128-row aligned at every world). `dense_weights: "fp8"` on such a
checkpoint means "as shipped"; the at-load encoder still handles the BF16
classes it handles today (PLE key/value, indexer). GR sites stay BF16 with
bf12 companions.

**D5. A packed int8 lm_head.** `QwenGlobalsResident` gains `lm_head_packed`
(bits 8, group 128, F16 scales, rows V, cols H). `lm_head_logits` runs
`launch_packq_gemv_f32` at up to 4 rows (one 4-row chunk, the matrix read
once) and `launch_packq_gemm_f32` above 4 rows (the tile reads the matrix
once per 32-row tile; at C4 x depth 3 = 16 rows the GEMV would read it four
times). `last_row_only` passes the last row as m = 1. The draft head is the
same function. Vocab sharding at TP is a row view. `fp8_head` does not
apply to this head; the recipe omits it. The int8 head is more precise per
element than the fp8 one (about 0.7 % vs 2.6 % relative RMS), which is
irrelevant to the gates but worth knowing.

**D6. The table directory is an engine key.** `engine.ngram_table_dir`
names a directory whose `*.safetensors` are scanned for the
`...ngram_embedding.*` names only; every other tensor in those shards is
ignored, and a table tensor present in the main checkpoint wins. It works
under both `ngram_table: "mmap"` and `"resident"`. The recipe points at the
FP8 snapshot directory on the fabric; `download_model.py` gains an optional
`ngram_table_repo` for sites without it, and `cluster_doctor` counts and
validates the directory.

**D7. Config parse is strict.** A `quant_method: "gptq"` block is accepted
only with bits 4, group_size 128, sym true, desc_act false, and a `dynamic`
map whose `+` rule is exactly the lm_head at 8 bits and whose `−` rules
cover exactly the classes the binding expects unpacked. The parsed result is
a `QwenWeightForms` record (experts, MTP experts, dense, head, table)
printed at boot, carried in the fabric settings record, and mixed into the
resident image key through new `loader_format` bits (16: packed experts,
32: packed head, 64: dense as shipped).

**D8. World 1 only.** The down projection's K = 640 slices to 320 (world 2)
and 160 (world 4), neither a multiple of 128, and 160 is not even a
multiple of 64. World 2 would need an uneven K split (384 / 256) with the
scale grid re-anchored, or expert parallelism. Not in this plan; the
target is one Spark.

**D9. Depth is measured, not assumed.** `mtp_depth` already accepts 1–5 for
every family; Qwen's chain (`kDraftChain`, the aside ring copy) runs any
depth, `qwen_engine_test` covers depth 3 at 16 slots, and a shipped RadixArk
template runs depth 4. The shipped depth for this template is whichever
wins the five-class measurement in §6; the plan's arithmetic says 3.

**D10. The draft head's vocabulary subset is an opt-in lever, exact by
construction.** The checkpoint recipe's `DRAFT_VOCAB=1` scores the draft over the
65,536 most frequent ids (+3–5 % decode, acceptance unchanged on English /
code, worse on CJK). The verify stays full-vocabulary, so greedy output is
unchanged and the sampled ratio verify stays exact (the residual (P − Q)+
covers tokens outside Q's support). Journaled key `mtp_draft_vocab` (0 =
off; N = the N lowest ids by merge rank, or a file of ids). Default off,
and it stays off: the target is met without it (user decision 2026-09-28).
It is an additional lever a site may turn on, reported with its own
acceptance and tok/s numbers beside the default's.

**D11. Prefill threshold per family.** `kPackqMmaFromRows = 128` is full
GLM's policy (its expert segments at a 2048 chunk average 64 rows). Qwen's
average is 40 rows per expert at 2048 tokens, so the grouped GEMV (4 rows
per pass over the weights) would re-read every expert ten times. The
threshold becomes a `GlmMoeLayer` parameter; Qwen sets it from the
measurement in S6 (8 / 16 / 32 candidates). The prefill gate is the NVFP4
template's 1.15 ms per token at 2K plus 10 %.

**D12. No new correctness semantics.** Greedy MTP == plain transcripts,
sampled marginals exact, prefix cache unchanged, the n-gram gather bitwise
the resident gather: all the existing Qwen gates apply unchanged to the new
forms, and nothing in this plan alters the engine's arithmetic outside the
weight readers.

## 4. Component map

### 4.1 Reused unchanged
- The Qwen forward, layers, GDN / QSA / GR / PLE kernels, kv pool, session
  core, graph engine, MTP chain, sampled proposals, prefix cache, memory
  plan framework, resident image mechanism, the fp8 dense GEMV / tile paths
  (pre-encoded matrices are the same `GlmQuantMatrix`).
- `load_quant_rows / cols / load_quant` (`src/loaders/weight_build.hpp`
  550–660; F32 or BF16 scale rows, `copy_scale_row` 766).
- The packq slot / grouped / tile kernels' structure (`src/kernels/glm_moe.cu`
  1987–2185, 4193–4255; `packq_gemm.cu` 45–239), `MoeExpertView::of(GlmPackedMatrix)`.
- `QwenNgramTableMmap` (`src/models/qwen/loader.cpp` 765–938) once its
  tensor map includes the table directory.

### 4.2 Modified
- `src/kernels/packq_gemv.cuh` (group / scale-type templates: lines 42–44,
  66–79, 106, 130, 263–289, 356–375; `dispatch_k` 454–482 gains 640 and 2560),
  `packq_gemv.cu` (accepts, error text), `packq_gemm.cu` (11, 69, 96–100,
  168), `glm_moe_launch.hpp` / `glm_moe.cu` (view fields, `check_packq_slot_args`
  2125–2149, `k_compiled` 2135 / 2159), `src/models/glm/moe.hpp` (view),
  `moe_reference.cpp` (`dequant_packq_weights` 93–111 takes group and scale type),
  `src/loaders/packq_quant.hpp` (encoder / decoder parameterized for the tests),
  `src/models/quant_matrix.hpp` (`GlmPackedMatrix::group`, `scale_f16`,
  `packed_check_cols` on the group).
- `src/models/qwen/config.{hpp,cpp}` (the gptq branch, `QwenWeightForms`),
  `binding.{hpp,cpp}` (roles `GptqCodes`, `GptqScales`, `GptqZeros`, F32 fp8
  scale entries, the dense-form probe, the packed head), `loader.{hpp,cpp}`
  (`load_gptq_rows / cols`, `experts_packed` slabs, `lm_head_packed`, the
  zeros check, `globals_bytes`, `loader_format`, the table directory in the
  scan), `moe_layer.{hpp,cpp}` (`experts_packed`, `routed_view`,
  `check_weights`), `forward.cpp` (`lm_head_logits` 397–414, the plan lines
  210–330, the head prefetch window 617, `moe_view` 357–371).
- `src/loaders/resident_stream.hpp` (directory scan 243–259 takes a second
  directory with a name filter; image key 408–425 through `loader_format`).
- `src/serve/cluster_config.{hpp,cpp}` (`ngram_table_dir`, `mtp_draft_vocab`),
  `apps/dgpp_serve.cpp` (the keys → loader statics, the settings record,
  the Qwen-only guard), `apps/qwen_load_check.cpp`, `apps/qwen_forward_check.cpp`
  (flags), `scripts/site_env.py` (`paths` allow-list if the table dir lives
  there), `scripts/download_model.py`, `scripts/cluster_doctor.py`.
- Tests listed in §5.

### 4.3 New
- `src/kernels/packq_repack.{hpp,cu}` only if the CPU transpose is too slow
  (D2 fallback).
- `benchmarks/micro/moe_slot_bench.cu --format packq` (K 640 / 2560, I 640,
  H 2560, E 512, top-10) and a head bench row for the packed int8 GEMV / GEMM.
- `tools/qwen_gptq_fixture` support inside `tests/cuda/qwen_fixture.hpp`
  (GPTQ-layout tensors with I32 / F16 dtypes) and the GPTQ + F32-fp8 dequant
  in `tools/qwen_reference_dump.py`.
- `deploy/cluster_qwen-3.8-flash-next_autoround-int4_w1.example.json`.
- `docs/benchmarks.md` rows and local, untracked dated run artifacts
  record including the verbatim `bench_qwen35.sh` output.

## 5. Implementation stages and acceptance gates

### S1. Kernels: group 128, F16 scales, the Qwen widths (1.5–2 days)

Work: D1 on `packq_gemv.cuh`, `packq_gemm.cu`, the slot / grouped launchers
and views; K = 640 and 2560 in the compiled set for int4 (and 2560 for int8,
the head: 32 lanes x 5 chunks); the oracle parameterization.

Gates:
- `packq_gemv_test` and `packq_gemm_test` sweep {128, f16} at K = 640 and
  2560, bits 4 and 8, m = 1..16, against the FP64 / local oracles with the
  same tolerances as today; the {64, bf16} instantiations are bitwise the
  pre-change binaries on the existing sweeps (record the digests before
  touching the files).
- `packq_gemv_checkpoint` (real full-GLM slices) unchanged.
- `glm_moe_test`'s packq cases extended with group 128 at the Qwen shapes,
  routed-only (`n_shared_experts = 0`).
- `moe_slot_bench --format packq` at 1, 2, 4 rows against `--format fp4` on
  the same shapes: the packq int4 rate must be at least the fp4 core's (the
  fp4 slot class is at line rate at world 1; the per-code work is
  comparable and the geometry identical: 5 chunks per lane in passes of
  4 + 1). If it falls short, the fp4 core's paired loads
  (`row_dots_pair`) and narrow-row lane remap are the reference fixes,
  before S6.
- Full-GLM regression: the glm_dsa unit / fixture suites; the fabric A/B
  (`scripts/fabric_glm_regression.sh`) when a node frees up, before S7.

### S2. Config, binding, loader, fixture (1.5–2 days)

Work: D2, D3, D7. The gptq branch and `QwenWeightForms`; binding roles and
expectation tables for packed experts (backbone and MTP) and the packed
head; `load_gptq_rows / cols` (transpose, F16 scale transpose, zeros
check) into `experts_packed` slabs; `routed_view` / `check_weights`;
`globals_bytes` and the counting build (`bump.alloc` identical with
`copy == false`); `loader_format` bits; the fixture generator emitting GPTQ
tensors; the Python reference dump dequantizing them and extended to chained
draft rows (depth up to 3), so the fixture gate pins every chain position.

Gates:
- `qwen_config_test`: the gptq block parses to the expected forms; refusals
  for bits ≠ 4, group ≠ 128, sym false, desc_act true, a `dynamic` map that
  packs a class the binding expects unpacked, and the base hybrid's
  `layers.48` exclusion (message names the int4RTN variant).
- `qwen_binding_test`: the expectation table for the new forms; `g_idx`
  reported as unexpected.
- `qwen_loader_test`: on a GPTQ fixture, every resident expert dequantizes
  (`packq_decode` with group 128 / F16) bitwise to the fixture's GPTQ
  dequant; a fixture with non-constant `qzeros` is refused; the plan's
  bytes equal the counting build's; the resident image round-trips and an
  image written under the NVFP4 forms is not restored (key differs).
- `qwen_moe_test`: the Qwen MoE over packed experts == the host reference
  at 1 / 3 / 5 / 8 rows within the existing tolerance; decode == prefill
  path per the test's existing contract.
- `qwen_forward_test` on the GPTQ fixture: per-layer l2 against the Python
  reference within the FP8 fixture's thresholds (~2e-3), the plan check, the
  smoke.

### S3. Pre-encoded FP8 dense and the packed int8 head (1 day)

Work: D4, D5. The dense-form probe; F32 `weight_scale_inv` binding entries
for GDN / QSA / shared; `load_quant_rows / cols` in `build_gdn`, `build_qsa`,
`build_moe` (shared); the quantized qkv segment merge; `lm_head_packed`
through `packq` GEMV (≤ 4 rows) and GEMM (> 4 rows); the head prefetch window.

Gates:
- `qwen_loader_test`: a fixture whose dense classes ship as FP8 + F32 loads
  to the same resident bytes as the BF16 fixture encoded at load (the
  fixture generator produces both from one BF16 source with the recipe of
  `fp8_quant.hpp`).
- `qwen_head_check` / `tests/python/qwen_head_check_test.py` extended: the
  packed head's logits equal a host dequant + fp32 dot within the fp32
  summation tolerance, argmax identical, at m = 1, 4, 5, 16 (both launchers).
- `qwen_decode_test` / `qwen_engine_test` (loopback world 2 fixture, the
  `DGPP_TEST_DENSE_FP8` variant) on the GPTQ fixture: MTP greedy transcripts
  == plain at depths 1, 2, 3; scalar and batched replays; the fallbacks.

### S4. MTP experts, table directory, memory plan, recipe, download (1 day)

Work: D6 and the MTP layer's packed experts (the same path as S2); the
memory plan lines (packed experts, packed head, MTP packed experts, table
0 under mmap); `engine.ngram_table_dir` end to end (config → serve → loader
scan with the name filter → `after_restore` reading the scale from the
merged map); `download_model.py --ngram-table-repo`; `cluster_doctor`; the
example recipe:

```
model  Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN, world_size 1
engine max_concurrency 4, kv_capacity 262144, kv_dtype bf16, decode_graph,
       mtp, mtp_depth <S6 result>, ngram_table mmap, ngram_table_dir <FP8 snapshot>,
       dense_weights fp8, bf16_weights bf12+bf16, prefix_cache_gib 3,
       default_max_tokens 32768, admission full, admission_window 256,
       sampling_candidates 128, prefill_budget_tokens 4096,
       prefill_idle_budget_tokens 4096, model_alias qwen
```

Gates:
- `qwen_loader_test`: the table from a second directory with distractor
  tensors in its shards is the same mapping as from the checkpoint
  directory (rows bitwise, plan bytes 0 under mmap, resident copy under
  `resident`); a missing directory or a missing part is reported by name.
- `qwen_ple_test` unchanged (the staged gather).
- `cluster_config_test`: the new keys, Qwen-only guard, the settings record.
- `dgpp-serve --memory-plan` on the recipe: the plan lists the packed
  classes and the table at 0; total + 4 GiB headroom under the board.
- `scripts/site_env_test` / setup wizard: the new example appears for
  world 1.

### S5. Real checkpoint on one idle Spark (1 day, needs the node)

- `qwen_load_check --world 1 --mtp --ngram-table mmap --ngram-table-dir …`:
  the zeros check passes over all 73,728 + 1,536 + 1 matrices, resident
  ~67 GiB, cold boot and image boot times recorded, the gather microbench
  as in `docs/qwen38_single_spark.md`.
- `qwen_load_check --compare-dense-fp8 <nvfp4 snapshot>`: this checkpoint's
  FP8 codes and F32 scales for every GDN / QSA / shared matrix against our
  at-load encode of the NVFP4 release's BF16 tensors. Expected bitwise
  (§1.2); a difference is reported per matrix with its max relative
  deviation and becomes a finding, not a blocker (the as-shipped values are
  still what we serve).
- `qwen_forward_check --layers 8` against the Python reference on the 5090
  box (layers 0–7 downloaded there, ~10 GiB; the dump dequantizes GPTQ and
  F32-fp8): logits, per-layer digests and the residual rms within the band
  the fp8-dense check showed (argmax agreement on the probe positions,
  residual rms within 0.3 %); resident table vs mmap'ed identical; the
  chained draft rows at positions 2 and 3 against the reference dump's
  (§7 item 1) with the same criteria as the first draft row.
- The n-gram gather on a cold page cache at 65-token prompts, before and
  after §7 item 2: per-prompt gather time from the host node's timing,
  TTFT from `serve_bench.py`.
- Serve gates on the recipe (each a pass / fail line in the record):
  `serve_api_check.py` (incl. the prefix-cache probe), `serve_greedy_transcript.py`
  MTP == plain at depths 1, 2 and 3 (4 of 4 each; `mtp_depth_check.py`
  diff clean), `serve_mtp_classes.py` 5 of 5, `serve_eval.py --no-think`
  GSM8K / HumanEval / extract within the band of the six Qwen rows in
  `docs/benchmarks.md` (≥ 291/300, ≥ 158/164, 100/100; a miss by one is a
  near tie and re-run once), `serve_agentic_streams.py`, the W1 soak and
  drill (the fabric evidence rituals), `serve_bench.py` sampled pace.
- Prefill: `serve_prefill_probe.py` at 512 / 2K / 8K, gate 1.15 ms per token
  at 2K + 10 % (D11 threshold chosen here).

### S6. The throughput program (3–5 days, needs the node)

See §6. Exit: the five `bench_qwen35.sh` cells measured and recorded on
the shipped recipe with `mtp_draft_vocab` off, each at its depth-3 pass
within a few ms of the byte floor, with every S5 gate still green on that
recipe. The subset lever is measured and recorded separately.

### S7. Ship (0.5 day)

`docs/benchmarks.md` rows for the new template (C1 engine tok/s, loaded
C4, cold prefill 2K / 8K / 32K, modes, quality, long context) from the
campaign runner with a new results directory and the `bench_qwen35.sh`
output; README status line (packed int4 now covers Qwen); this document's
status section; `docs/qwen38_single_spark.md` cross-reference; `docs/mtp.md`
and `DESIGN.md`'s stale "depth 1–3" text; CHANGELOG; PLAN.md; the example
recipe; the memory note.

## 6. The throughput program

### 6.1 The measurement, verbatim

`bench_qwen35.sh` (the checkpoint recipe's five-prompt bench): five prompts, one
request each, two runs, `temperature 0.0`, no thinking flag, `model:
"qwen"`, `max_tokens` 256 / 512 / 1024 / 64 / 2048:

1. Q&A: "What are the main differences between TCP and UDP? Be concise."
2. Code: "Write a Python function that implements binary search on a sorted list. Include type hints and docstring."
3. JSON: "Generate a JSON array of 10 fictional employees with fields: name, age, department, salary, email, skills (array of 3). Output ONLY valid JSON, no explanation."
4. Math: "What is 7823 * 4519? Show only the answer."
5. LongCode: "Write a complete Python implementation of a red-black tree with insert, delete, search, and in-order traversal. Include all rotation methods."

Run it unmodified against the recipe (`model_alias: "qwen"`; `usage.completion_tokens`
must count reasoning tokens, to check in S5). Beside it, our own protocol
for the engine view: `timed_load.py --concurrency 1 --classes all --repeat 3
--think` (greedy, thinking on) and the per-request retire lines for ms per
pass and p1 / p2 / p3.

### 6.2 Levers, in order of expected value at world 1

| # | lever | expected ms per pass at depth 3 | exactness | status |
|---|---|---|---|---|
| L1 | depth 3 (exists) | pass ~56, tokens per pass the unknown | greedy transcripts identical by the MTP gate | measure first: per-class p1 / p2 / p3 on the five prompts, thinking on |
| L2 | int4 experts at the fp4 core's rate | −1.5 | exact (D1, D2) | S1 bench, S5 profile |
| L3 | draft head vocabulary subset (D10) | −5.5 (three draft heads 636 → 168 MB) | exact (verify full-vocab) | opt-in only, never default; not counted toward the target |
| L4 | GR down + inject split-k (55 % of line rate today), shared-expert tail (70 %) | −1 to −2 | bitwise per the existing fused-kernel gates | from `docs/qwen38_single_spark.md`'s list |
| L5 | draft-step overlap: prefetch the head window during the chain row's layer, the chain ring copy on the side stream | −1 | exact | profile-driven |
| L6 | prefill threshold D11 | prefill only | exact | S5 |
| L7 | TTFT at 65–90-token prompts | 100 ms today at C1 on warm rows; serialized faults on cold rows | — | §7 item 2: parallel fault issue in the gather |

Not levers here: the fp8 dense stack (already ours), the head bytes (int8 =
fp8), `SEQS` / concurrency (single stream), prefix caching (fresh prompts).

### 6.3 Floors and gaps per pass (world 1, 233.6 GB/s)

Everything a depth-3 pass does, with its physical floor and where we stand.
"Floor" is bytes at line rate for a stream, zero for a serial gap, and the
model's own value for acceptance. The gaps sum to the work of S6.

**Correction (2026-09-29).** The first version of this table stated a total
floor of ~38.5 ms; its own rows summed to 46.5 (an addition error), and the
draft layer's bytes were understated (0.05 GB per step: its q/k/v/o ship in
BF16, 100 MB, so a draft step is 0.17 GB besides its head). The floor of a
depth-3 pass on this checkpoint at world 1 is **48 ms**. The measured column
is the 2026-09-29 nsys profile of the served recipe (one 400-token request,
196 steps; the 2026-09-28-qwen-autoround-int4 / profile_depth3 measurements;
step 100 = 62.3 ms under the tracer, the unprofiled pass 56–60 ms).

| operation | bytes / pass | floor ms | measured (nsys) | gap | fix |
|---|---|---|---|---|---|
| backbone dense stack, read once for the 4 rows: GDN 2.08 GB (36 layers x 57.7 MB), QSA 0.60, GR 1.26 (BF16 as shipped), shared 0.24, routers 0.13, a/b + indexer + norms ~0.05 | 4.36 GB | 18.7 | 26.4 of kernel time: qkv/z and q/k/v multi-GEMV 8.6 (179 us, at rate), out/o GEMV 4.5 (93 us vs 67), GR up 3.3 + GR down 3.5 (32 + 36 us vs 28 each), KDA recurrence 1.7 (48 us vs 27), shared 1.6 (33 us vs 21), routers 1.0 (19 us vs 11), ~600 launches of ≤ 6 us 1.7 | 7.7 | narrow-N fp8 GEMV for out/o (N = 2560 over K = 6144 leaves SMs idle), GR down split-k, the KDA state held in registers across the 4 rows, fused shared tail, router fused into the norm, small-kernel fusion (§7 items 4, 5) |
| backbone routed experts, 4 chained rows (~30 distinct experts of 40 slots per layer) | ~3.8 GB | 16.1 | 16.1 (gate_up 225 us + down 102 us per layer) | 0 | at line rate on the packq slot core |
| idle between the verify's 1,311 graph nodes | 0 | 0 | 1.6 | 1.6 | node fusion: fewer, larger nodes (§7 item 5) |
| verify head, 4 rows, int8 g128 | 0.64 GB | 2.7 | 2.8 | 0.1 | at line rate on the packq GEMV |
| verify sample + verdict, 4 rows x 248K logits | 4 MB | ~0.05 | 0.43 (partials 193 us, local 110, verdict 64, prepare 21) | 0.4 | one fused argmax + verdict launch |
| commit: retract the 36 GDN states to the accepted row's snapshot (113 MB copied on the steps where a draft row fails) | 0 | 0 | 1.05 on those steps, 0.88 average | 0.9 | the next pass's recurrence reads the accepted snapshot through a device-side row index; no copy (§7 item 3) |
| draft step x 3: head 0.64 GB + draft layer 0.17 GB (q/k/v/o BF16 100 MB, GR 26, ten experts 25, fc 13, shared 10) | 3 x 0.81 GB | 3 x 3.47 = 10.4 | 3 x 4.0 = 12.1: head 2.75–2.79 (at rate); layer 1.27: q/k/v/o 0.49 at one row / 0.60 at four (floor 0.43), GR pair 0.20 (0.11), pick 0.12 (~0.005), experts 0.15 (0.11), fc on a cutlass tile 0.09 (0.06) | 1.7 | the pick's argmax (117 → ~5 us), the one-row GR kernels (199 → 111 us), the four-row q/k/v/o GEMV, fc on the GEMV path |
| n-gram gather (mmap) | 64 rows x 160 B | 0 | hidden behind layer 0 at this context; serialized faults on fresh rows | see §7 | parallel fault issue |
| total | ~10.1 GB | 48.0 | 62.3 under nsys; 56–60 unprofiled | 10–12 | |

The pass's floor is 48 ms and today's chain runs it at 56–60. Every
millisecond between the two is a listed defect with a fix. The same dense
defects set the plain (no-MTP) step: 32.7 ms measured against a 26.6 ms
floor (4.36 GB dense + 1.22 GB of experts at one row + the 0.64 GB head).
The pass also grows 56 → 60 ms from a short context to 2K where its bytes
grow by 50 MB of K/V (0.2 ms): that growth is a defect to profile next
(a 2K-context request under the same procedure).

**Acceptance is the model's, not the engine's.** With the same draft
weights and a correct feed, greedy acceptance at every chain position is a
property of the checkpoint and of the text being decoded. On the bench's
code prompts with thinking off our chain accepts 2.65 of 3 (p1/p2/p3
95/89/81 %), and the feed is right (the verify is exact: transcripts
identical across depths 1/2/3 and against the plain world). In reasoning
text the same chain accepts 1.78 of 3 (77/58/43 %), and our greedy
trajectories on the thinking-on prompts reason at length (the greedy path
of a prompt depends on the summation orders; task quality is at the FP8
baseline, §5 S5). That is what the thinking-on cells measure beyond the pass.

### 6.4 What the cells look like at the floor

At the 48 ms floor with the measured acceptance: code text (3.65 tokens
per pass, thinking off) 13.2 ms per token, 76 tok/s; reasoning text (2.78
per pass) 17.3 ms per token, 58 tok/s. Scaling the measured thinking-on
cells (43.2 / 47.7 / 61.9 / 51.2 / 45.2 at the 56–60 ms pass) to the
floor: Q&A 54, Code 60, JSON 77, Math 64, LongCode 56 (LongCode's
2048-token reasoning trajectory holds it lowest). With thinking off,
today's pass reads 55.4 / 59.1 / 63.1 / – / 59.6.

### 6.4a Status 2026-09-29 04:45 (S6 rounds 1–3)

Three fabric rounds on the depth-3 recipe, every one bitwise the baseline
transcripts and 58/58 on the Qwen, pick and KDA suites
(the 2026-09-28-qwen-autoround-int4 measurements, S6):

1. the recurrent state's checkpoint-and-replay form (§7 item 3a), the
   chunked greedy argmax, the gate activation folded into the up GEMV;
2. a site's combine inside the next site's norm launch (§7 item 4a), the
   replay rows row-major;
3. the n-gram gather on a thread with a device spin-wait (§7 item 2),
   `process_madvise`.

The pass: 60–62 → ~57 ms (stats ms/step; 59.7 at step 100 under nsys);
the five bench cells thinking on 44.8 / 49.5 / 63.7 / 51.4 / 46.8, off
57.0 / 60.7 / 65.0 / 21.9 / 61.3. What the rounds established about the
floor: the read line rate is
248–256 GB/s sustained and the clock holds (2.5 GHz at 60 W in decode),
but a 65 GB working set walks 32K 2 MB pages per pass and every
streaming kernel runs 8–10 % below its isolated rate in the step — an 8
GiB cold ring reproduces the step's kernel times exactly. Larger pages
(unreservable at runtime), host memory through ATS (65–80 %) and per-SM
contiguous spans (±4 %) do not recover it, so the streaming rate the
step can reach is ~228 GB/s and the depth-3 pass floor at that rate is
~50 ms. The prefetch's ~3.9 GB per pass reach L2 for 96 MB of hits; its
value is the page walks (a one-line-per-page touch form holds the pass).
Left on the ledger: the QSA attention over the selected keys at 2K+
context (~2 ms), the shared expert and router ramps (~0.6), the sampled
pick path (0.4), and the walks themselves.

### 6.4b Status 2026-09-29 09:05 (S6 rounds 4–12)

The pass at the author's protocol: 57 → 54.1 ms (greedy, short context;
`serve_round12/`), every round bitwise the depth-3 transcripts. Shipped in
rounds 4–12: the QSA attention gather (181 → 137 us), the draft layer's
bf12 companions under the hybrid's fp8 dense stack, the head in per-row
bit planes with the exact candidate pass (−2.6 ms: the four head reads at
75 % of their bytes, 2.13 ms each at line rate), the up GEMV with the mix
in its epilogue (one launch fewer per site), the routed accumulation folded
into the shared down. Measured and kept as they were: the byte prefetch
(a size cap or off loses 1.3–2 ms), the touch form, MTP depth 3 (depth 4
is slower and not bitwise; depth 2 wins only reasoning-heavy text). The
opt-in draft vocabulary slice (`engine.draft_vocab`): 49.2 ms per pass,
never a headline number.

Where the 54.1 ms goes against the ~50 ms floor: 1,337 kernels a pass with
0.35 ms of true idle — the gap is inside the kernels, mostly the ~150 us
per layer that the 23 launches of a layer lose to their ramps and tails
(each GEMV at 80–95 % of line rate). What is left on the ledger: the
small-kernel folds (~0.5 ms), the register-lean attention pipeline (~0.5 ms
at 2K+ context), and beyond those only a persistent per-layer structure
that overlaps one kernel's tail with the next's ramp. Measured and
rejected in round 13: the hyper-connection down's bf12 companion at world
1 (+2 ms a pass: 46.5 us per site against 31.6 — the packed decode on 41
blocks of 320 long rows is op-bound, as the 2026-09-26 bench found for the
fabric worlds; kept behind `DGPP_QWEN_GR_BF12=on`).

### 6.5 Order of work in S6

1. Depth 3 on the recipe as built: `bench_qwen35.sh` twice, the
   `timed_load --think` protocol, retire lines with p1 / p2 / p3 per prompt.
   Record acceptance per class; a chain that accepts less than the
   model's own draft would goes to the chain reference gate first (§7,
   item 1).
2. The world-1 profile (`scripts/fabric_qwen_profile.sh`,
   `nsys_step_breakdown.py`): every class against the table above.
3. Close the gaps in the order of the table: chain latency, dense-stack
   kernels, seam and gaps, then the expert class if the S1 bench left
   anything.
4. D10 measured last, as an approximate op, with the key on: its acceptance
   and tok/s per prompt go in the record beside the default's. It does not
   count toward the target and does not become the default.
5. Choose the shipped depth (3 unless depth 4 wins the code classes without
   losing the others); re-run every S5 gate on the final recipe; then S7.

### 6.6 The head in bit planes (2026-09-29; github issue #69)

An argmax over the head does not need every logit exactly. The int8 head's
rows are stored as three planes PER ROW — `[K/2 B high nibbles][K/4 B bits
3..2][K/4 B bits 1..0]` — so a pass over the high six bits reads 75 % of
the row as three contiguous runs. (The first cut interleaved the planes per
128-code group, `[64 B][32 B][32 B]`: the kernel read 75 % of the bytes and
took 92 % of the time, unchanged by a cheaper decode — the skipped 32-byte
sector shares its 64-byte DRAM burst with the mid plane on this memory
system, so every byte moved anyway. A plane must be a contiguous run at
least a burst long.) For a call whose rows all belong to plain-greedy
slots (no temperature, logprobs, penalties, bias or grammar — the engine
writes a per-slot flag beside the sampling specs; the decode graph is fixed
at capture, so the kernel reads the flag on the device), that pass gives
every head row `A ± B`: `A` the centered six-bit dot, `B = 1.5 · Σ_g |s_g|
Σ_{k∈g} |x_k|` (the scales are signed; `B` is inflated by 1e-3 for the
fp32 rounding of both chains). Rows whose `A + B` falls under the best
`A − B` cannot be the argmax; the rest run the packed core's exact chain
from all three planes, bit for bit, and every other row's logit is `-inf`,
so the pick's composite key (ties included) is the full read's. Sampled
rows read all three planes and are the row layout's chain bit for bit; the
prefill tile kernel reads the planes with a 128-code k-step and applies the
scale per 64 codes, so it is the 64-code kernel's chain bit for bit.

The study behind the split (the 2026-09-28-qwen-autoround-int4 / serve_round8 / head_dump.bin measurements, 5,571 head inputs dumped from the live server
with `DGPP_QWEN_HEAD_DUMP`, 300 studied): the true top-1 leads the
runner-up by 3.7 logits at the median (p10 0.32); the 6-bit bound is ±3.7
at the top, candidates 4 / 840 / 40k (median / mean / p99 of 248,320);
5 bits leaves ±9.2 and a fat tail (mean 23k); 4 bits is useless (±21).
Bytes per argmax-only head read: 75 % + about 2 MB of candidate rows.
Measured (round 12): 56.5 → 53.9 ms per pass (−2.6, the four head reads),
every greedy transcript identical; the author-protocol cells +5–6 %.

Kernels: `kernels/packq_head.{hpp,cu}` (three launches per four-row chunk:
prepare, bounds, candidates), the plane tile in `packq_gemm.cu`, the
permute at load in the loader (the head is repacked every boot; no
resident-image change). Gates: `packq_head_test` (full form bitwise the
row layout, argmax + logit bitwise, the tile bitwise, the slice's scatter),
the transcript gate. `DGPP_QWEN_HEAD_PLANES=off` keeps the row layout.

### 6.7 The draft vocabulary slice (opt-in; `engine.draft_vocab`)

The recipe's D10: the draft head scores a fixed subset of the vocabulary
(a `.npy` of ids; `tools/build_draft_vocab.py` builds one from the
tokenizer's alphabet, its added tokens and the most frequent merges by BPE
rank — 64,934 of the recipe's 65,536 ids, 99.1 %). The draft's rows are
gathered into their own plane-layout matrix at load; the draft's head call
scatters the slice's logits to their ids over a `-inf` row, so the pick and
the verify are unchanged; a token outside the set is never proposed and
the target verifies every proposal, so outputs are unchanged and only the
acceptance moves. Off by default, never counted toward a headline number.

### 6.8 The per-launch intercept, measured (2026-09-29, round 16)

The question rounds 4–15 ended on: how much of the pass's ~4 ms over the
byte floor is ramp and tail that only a fused or persistent form could
recover. `qwen_dense_bench --sweep` fits us = a + bytes / rate per kernel
family over a cold ring (intercepts 5–14 us on a stream, of which 2.06 us
is the empty-launch gap); `--only pdl` runs the hyper-connection pair as a
dependent chain the way the step launches it (a captured graph): the
production pair sits at 92–95 % of the ring floor, and the best prologue
either way — programmatic dependent launch with the first weight pass
issued before the grid wait, or an L2 prefetch of that pass before the
activation staging — is worth ~1 us a launch at one row and ~0.5 at four.
Programmatic launch with the wait at the top (the 2026-09-08 null probe)
hides the stream's launch gap and nothing else, which a graph has no gap
to hide.

The kernel-side form (every GEMV warp prefetching its first pass before
the staging) was built, bitwise, −4..−8 % on the ring, and lost +1.1–1.5 %
a step on the fabric (53.8 → 54.6, 53.7 → 54.3): in the step the side
stream's prefetcher has each dense matrix in L2 before its kernel starts,
so the round trip the bench hides does not exist there. Reverted; the
bench modes stay. What the measurement settles: the pass's gap is the
~1,000 small latency-chain launches (≈ 2 ms), the prefetch stream's own
DRAM contention and the head at line rate — not GEMV ramps; the
hyper-connection "7.5 vs 5.1 ms" of §6.3 was per-site accounting under
the prefetch stream; a per-layer megakernel is bounded at ~2 ms and off
the plan. The exact levers left are folds of the small launches
(combine_norm into the batched down_inject's staging with one 16-way
block reduction, the router's dots read once per expert for all rows,
kda_conv + a/b in one launch, the gated norm in the out-proj staging),
each ≤ 1 % and decided only by a two-binary fabric A/B.

### 6.9 The step at context; the attention gather as cp.async phases (2026-09-29)

Every decode number above is at a prompt under 80 tokens; an agent client
runs 10–100K. Measured through the endpoint (the engine's step time over
its decode steps, one greedy request per length): 52.6 ms a step at
130–190 tokens, 55.7–56.0 at 3.2K, 55.6–56.9 at 11.8K, 55.8–56.0 at 47K —
the sparse attention's selection budget is fixed, so its cost is: +3.2 ms
at 3K and flat from there. Prefill holds 0.63–0.65 ms a token to 47K. The
12K profile put the delta in the listed attention (136.6 us a layer, 2.05
ms a step, against a 72 us byte floor: 16.8 MB of K and V rows a layer),
the index scoring (+0.27 ms) and the prefix cache's state snapshots
(+0.5 ms of recurrence launches).

The listed attention's tile loop resolved, gathered and computed in
series; the register-pipelined form of round 7 held the next tile in 125
registers and lost. The form that ships: the tile's K rows and V rows as
two cp.async groups into the serial loop's own shared buffers, the next
tile's K copy issued once this tile's scores have read kt (it runs under
the PV phase), the next V copy once PV has read vt (under the next tile's
resolve and scores) — no register holds a tile, every arithmetic step is
the serial loop's in the same order (`qsa_test` gates the two forms
bitwise; `DGPP_QSA_ASYNC=0` keeps the serial gather). On the fabric,
legs S A S A: −0.4 to −0.5 ms a step at 12K and 47K, nothing at short
context, the prefill's under-128-row chunk tails −1.5 %. Left at context:
~0.5 ms a step in the listed attention (each kv head's rows read three
times across the head groups, the lane-0 `expf` loop), 0.4 in the index
score.

### 6.10 The short-context folds, measured (2026-09-29, round 16)

Two bitwise folds of the small-launch ledger went to the fabric as one
bundle against the round-15 binary. The GDN layer's a and b projections
(BF16 [lv, H]) as bf16 problems of the fp8 qkv + z multi launch
(`Fp8GemvProblem::bf16_weight`: the block runs bf16_gemv's row chain on
the same staged activations; `scale_gemm_test`): −0.1 to −0.2 ms a step,
kept. The batched rows' group norm staged into the down + inject GEMV
with one block reduction for every (row, group) (`qwen_gr_norm_down_inject`,
bitwise the norm + down_inject chain): +0.8 ms a step — the down GEMV's
41 blocks each recompute the sixteen norms, and that redundant work costs
more than the 8 us launch it removes, as the sequential form did on
2026-09-09. Kept behind `DGPP_QWEN_GR_NORM_FOLD=on`, off. The rule the
two rounds leave: a whole-row reduction does not fold into a many-block
GEMV's staging; work folds into the producer's epilogue (the combine into
the out-projection's and the shared expert's epilogues is the race-free
form left, ~0.3 ms). The step at short context: 53.5–53.6 ms.

### 6.11 The context attention forms, measured (2026-09-29, round 16 closing)

The two items left at context after the async gather — the listed
attention's second read of each kv head's tile across its head groups
and the index score's serial pool loop — went to the fabric as pinnable
geometries of the same kernels (every form bitwise the served one;
`qsa_test` holds them so). Against the shipped binary at 12K and 47K
(56.2–56.4 ms a step): twelve heads per block, so a kv head's rows are
gathered once, +0.2 to +0.3 ms (the second read already hit L2, and 384
threads at 125 registers leave one block per SM); two heads per warp
+0.1 to +0.2; the denominator's `expf` hoisted into the PV loop +0.1 to
+0.2; four pools' loads in flight per warp in the index score no faster
at decode and +3.5 % on the 47K prefill, which scores every pool for its
chunk tails. The shipped kernels stay. What the two kernels hold at
context is latency that these shapes do not reach; the campaign's exact
levers end here: 53.5–53.6 ms a step at short context, 56.3 from 12K to
47K.

### 6.12 The prefill (2026-09-29, round 17)

The hybrid prefilled at 1,340 tok/s (2K) and ~1,600 (8K–47K). The 32K profile (22.0 s of GPU time): the
packed-int expert GEMMs 34 %, cuBLAS / cutlass dense GEMMs 15 %, the
hyper-connection elementwise passes 12 %, the GDN chunked recurrence 8 %,
the expert accumulation 6 %, the n-gram stage wait 5 %, the listed
attention 5 %, the index score 3.5 %. The dense BF16 peak on this GPU is
89 TFLOPS (cuBLASLt, `micro_gemm_peak`); the expert GEMM ran 26–29 useful
TFLOPS on a segment set whose 64-row tiles carry 38 % padding at
4096-token chunks.

Shipped, bitwise (the 17K-token greedy completion's sha unchanged):
`packq_gemm_wide_kernel` — 64 x 128 x 64 tiles, eight warps, a two-stage
cp.async pipeline at two blocks per SM, each k-step's int4 codes decoded
once into a bf16 shared tile (nibble | 0x4300 is 128 + code in bf16
exactly, less 136 exactly code − 8) so the MMA loop is ldmatrix and mma,
the same four m16n8k16 steps per 64-code group into a fresh partial then
one fp32 fma with the group's scale as the narrow kernel (−8 / −12 % on
the bench's gate/up and down shapes; `engine.expert_gemm: "narrow"` (a config key since round 21; it was an environment switch) keeps the
old); the prefill's combine riding the next site's norm pass
(`defer_combine` returns the pending combine for prefill rows, so
`combine_norm` replaces `combine_apply` + `group_rmsnorm`); the prefill
chunk's page advice batched in `process_madvise` calls (the `total <=
4096` guard sent every chunk through the per-row `madvise` loop: the 56
ms stage wait). Together −7 / −9 / −7 % at 2K / 8K / 32K: 32K prefill
19.8 → 18.4 s, 1,745 tok/s. Left: the 8192-token chunk (the model's
forward rows are the family's constant, so the scheduler refuses a larger
budget), the wide kernel's register-pipelined k-loop, the tile padding on
skewed expert segments, the elementwise mix passes, the fp8 dequant per
chunk.

### 6.13 The expert GEMM's launch, taken apart (2026-09-29, round 18)

The segment-aware warp layouts (a 64-row block picking one, two or four
m16 tiles by its segment's rows; bitwise) won −16..−20 % on the Zipf
bench and nothing on the fabric (session T). The chunk profile said why:
in most layers of the 8K probe one routed expert takes 3,000–4,096 of the
chunk's 4,096 tokens, so the grid — n-tiles × the longest segment's
m-tiles × 512 segments — is 160K blocks of which 97 % exit, and the
launches run gate 4.0–4.4, up 4.0–5.3, down 5.4–7.0 ms (~14 ms a layer,
30 % of the prefill) against 2.0–2.5 ms for a 556-token prompt's same
weight stream. Two levers came out of the bench once it read gathered
rows through a row map (`--gather 1`) from cudaMalloc memory
(`--device-copy 1`; the managed buffers it had used cost ~1 ms a launch
of page mapping and masked everything):

1. **The compact tile list** (`launch_moe_tile_list`, `MoeTile`; the
   layer's `tile_list_for`; `engine.expert_tile_list: false` keeps the max_rows
   grid): (segment, first row) per real 64-row tile, a 1-D grid of
   n-tiles × capacity blocks, the capacity bounded on the host without a
   sync — so the per-layer `routed_seg_max_rows` sync is gone on that
   path. Bitwise (the same blocks do the same tiles). The empty blocks
   were not the cost: gate/up unchanged, down −6..−16 %.
2. **The L2 prefetch of the step three ahead** (`engine.expert_gemm_prefetch`,
   default 3): the two-stage pipeline holds one step of latency and the
   weight stream — 419 MB a launch, the launch's DRAM traffic — was not
   overlapping the compute (with the weights L2-resident the same launch
   takes 2.7 ms; three, four and five stages at one block per SM, and a
   tile-major read pattern, changed nothing). `prefetch.global.L2` of
   each activation row's line every step and each code row's line every
   fourth: gate 4.6–5.0 → 3.5 ms (Zipf) and 4.4–4.6 → 3.1 (uniform), down
   6.8–7.1 → 5.8, a 556-token launch 3.6 → 2.1. Bitwise (the values do
   not pass through the prefetch).

Left on the launch: the down projection's fp32 output (419 MB written a
launch, 1.8 ms at the line rate; a bf16 intermediate is not bitwise the
ordered fp32 accumulation — the user's decision), the partial tiles'
B-side work (a 16-row tile decodes and stages the same 128 × 64 codes as
a 64-row one), and the register-decode three-stage pipeline (no decoded
weight buffer: 41 KB of smem keeps two blocks per SM) if the fabric shows
latency still on the table. On the fabric (session U, four legs, bitwise
by the long-prompt sha): 2K 1.40 → 1.29 s, 8K 4.61 → 4.42, 32K 18.46 →
17.82 (−7 / −4 / −4 %); 32K at 1,840 tok/s.

### 6.14 Round 19 (2026-09-30)

Round 19 (the 2026-09-28-qwen-autoround-int4 measurements,
sessions X–Z3): SHIPPED the chunk-ahead n-gram staging — the next chunk's
rows hashed from its host ids and gathered on a second channel the gather
thread serves while this chunk's layers run, issued after the PLE layer
consumed the staging (its first form, issued at the chunk's start,
overwrote the one buffer under the chunk still reading it: the in-situ
self-check counted exactly the following chunk's rows corrupted), the
claim keyed on the request, rows, position, ids pointer and a first/last
token check. −0.65..−0.75 s at 32K with a cold table, level when warm,
bitwise. Measured and not shipped: the four-token router tile (+0.25 s at
32K in situ), the combine-norm's batched loads (neutral), the
register-decode three-stage expert GEMM (level; opt-in), the gate and up
projections as one launch (level to +1 %; opt-in). The bf16 down
partials (−3..−4 %, not bitwise) stay opt-in. Left: the expert GEMM at
~40–45 % of the tensor peak — ncu on device memory shows it issue-side
bound (barrier 24 %, wait 15 %, long scoreboard 15 % of the warp cycles;
two barriers a k-step, a 32 x 32 warp tile), so the next form is a larger
warp tile with the same chain — and the two accuracy trades.

### 6.15 The accuracy trades as deployment keys (2026-09-30, round 20)

The decision (2026-09-30): the levers that beat the bitwise chain by
changing the prefill's arithmetic exist in dgpp, off by default, on only
through a deployment's cluster config — `engine.prefill_bf16_partials`,
`engine.prefill_fold_scales`, `engine.prefill_fp8_gemm` (docs/operations.md,
the [configuration reference](configuration.md#prefill-computation-and-precision));
none is bitwise the default chain, so a deployment
that turns one on serves transcripts that can differ from the default's,
and the benchmarks list a template's default-chain and levers-on numbers
as separate rows (the AutoRound template turns on the two that pay, below).

- `prefill_bf16_partials`: the packed chain's down projection written in
  bf16 and the per-expert partials summed from bf16 (the reference's
  moe_sum; half the 419 MB a 4,096-token chunk writes and reads back).
  The former env knob (`DGPP_MOE_PACKQ_DOWN_BF16`) is gone; the key sets
  `GlmMoeLayer::set_prefill_options`.
- `prefill_fold_scales`: the wide packed expert GEMM's variant 3 — each
  group's scale folded into the decoded bf16 weight values
  (`fold_scale`: bf16(code x scale)) and one fp32 accumulator across K,
  no per-group partial or fma (Marlin's form). Gate
  (`packq_gemm_test`): within 2^-7 relative RMS of the exact chain, the
  finiteness pattern the same.
- `prefill_fp8_gemm` (requires `dense_weights = fp8`): a new kernel pair
  (`kernels/fp8_gemm`): the per-token 1 x 128 e4m3 quantizer (amax / 448
  scales, the codes bitwise the host encoder's) and a 128 x 128 x 64
  mma.sync m16n8k32 e4m3 GEMM — eight warps, a four-stage cp.async
  pipeline (80 KB), ldmatrix on the 8-bit tiles, an fp32 partial per
  128-wide k group promoted by the row's and the column's scale (the
  reference's cutlass blockwise recipe), tiles walked in groups of eight
  m-tiles so a wave shares its activation tiles from L2. Gates
  (`fp8_gemm_test`): 1e-7 relative RMS against the exact sum of its own
  quantized inputs, the bf16 output the f32 sum rounded once, 2.5–2.8 %
  relative RMS from the dequantized bf16 chain it replaces (the e4m3
  activation's own error). The dense hook takes it for m > 128 rows and
  n >= 1,024 columns (the quantizer's pass over the activation costs
  494 / n of the GEMM's time: the [320 x 10240] GR down would lose 60 %).

The fp8 kernel against the chain it replaces (`fp8_gemm_bench`, 4,096
rows, ms per launch pair; the GEMM alone in parentheses):

| shape | dequant + cuBLASLt bf16 | quant + fp8 GEMM | delta |
|---|---|---|---|
| q/k/v-like [2560 x 2560] | 0.69 (0.59, 91 TF) | 0.75 (0.62, 86 TF) | +9 % |
| in_proj-like [5120 x 2560] | 1.34 (1.15, 94 TF) | 1.31 (1.17, 92 TF) | −2 % |
| out_proj-like [2560 x 5120] | 1.41 (1.21, 89 TF) | 1.36 (1.10, 98 TF) | −4 % |
| GR down [320 x 10240] | 0.55 (0.50, 54 TF) | 0.87 (0.36, 74 TF) | +57 % (the hook skips it) |
| PLE key [10240 x 2560] | 2.64 (2.24, 96 TF) | 2.57 (2.45, 88 TF) | −3 % |

Floor → measured → gap: the fp8 GEMM runs at cuBLASLt's bf16 rate (86–98
TF), ~35 % of the fp8 tensor peak — issue-bound like the packed kernel
(the 64 x 32 warp tile's 6 ldmatrix per 16 mma, the per-group promotion's
64 fma a thread, one block per SM at 240 registers), so the lever saves
the dequant launches and little else; the reference's cutlass blockwise
GEMMs run the same products in 580 ms per 8K chunk against our 936. The
form that closes it is a 64 x 64 warp tile with fragment double-buffering
and the promotion interleaved with the mma issue — a kernel round of its
own. cuBLASLt 13.0 declares the block-scale modes
(`CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F`, `BLK128x128_32F`) but its
heuristic returns no kernel for them on this GPU under any pairing
(probed 2026-09-30) — the library is not a shortcut here.

On the fabric (record sessions R20–R20c, one binary, the levers by
config; cold prefill 2K / 8K / 32K, the 17K-prompt greedy sha per leg):

| configuration | 2K | 8K | 32K | vs the base legs | greedy transcript |
|---|---|---|---|---|---|
| base (three legs) | 1.289–1.294 | 4.363–4.373 | 17.606–17.671 | | 927ffd8a |
| bf16 partials | 1.204 | 4.198 | 17.118 | −6.6 / −3.8 / −2.8 % | unchanged |
| fold scales | 1.276 | 4.450 | 18.235 | −1.0 / +2.0 / +3.6 % | unchanged |
| fp8 GEMM | 1.249 | 4.352 | 17.579 | −3.1 / −0.3 / −0.2 % | unchanged |
| all three | 1.208 | 4.327 | 17.827 | −6.3 / −0.8 / +1.3 % | unchanged |
| bf16 partials + fp8 GEMM (two legs) | 1.209–1.210 | 4.226–4.236 | 17.096–17.116 | −6.3 / −3.2 / −3.1 % | diverges (43 tokens, 857f15c6) |

The bf16 partials are the lever: −3 to −7 %, the 419 MB a chunk writes and
reads back halved. The fp8 GEMM adds little because its kernel runs at
the bf16 rate (above). The fold loses at 32K and cancels the other two
when combined: on an issue-bound kernel the per-code multiply it adds to
the decode phase costs more than the per-group fma it removes. Each
single lever kept the long prompt's greedy transcript (an argmax that
survived, not a bitwise property); the pair did not. Evals on the pair
(the baseline's settings): HumanEval 159/164, GSM8K 291/300, extraction
100/100 against the default chain's 159 / 292 / 100 and the templates'
band (153–161, 291–296) — within noise. The AutoRound example template
therefore turns on `prefill_bf16_partials` and `prefill_fp8_gemm`; the
fold stays off everywhere. The benchmarks document lists the default and
the levers-on rows (decode cells re-measured under the pair: the levers
touch prefill-shaped launches only).

### 6.16 The expert GEMM's warp tile (2026-09-30, round 21)

The round-19 reading of the in-situ ncu profile — tensor pipe 40 %,
barrier 24 % and wait 15 % of the warp cycles on a 32 x 32 warp tile —
suggested the kernel was issue-bound and that a wider warp tile (twice
the mma per ldmatrix and per barrier) would return ~10 % of the prefill.
Built as a template parameter of the wide kernel (`kWarps`): four warps at
64 x 32 / 32 x 32 / 16 x 32 by the rows the segment fills (variant 4, the
decoded-tile form; variant 5, the register-decode three-stage form;
`engine.expert_gemm: "wide4" | "wide4r"`), the staged tiles, the decode and every
element's chain unchanged — bitwise the eight-warp kernel in
`packq_gemm_test` (listed and segment-major), 208–218 registers at 128
threads, two blocks per SM, no spills.

The bench (`packq_prefill_bench --m 4096 --sf 1 --experts 512 --top-k 10
--device-copy 1 --gather 1 --tiles 1`, ms):

| routing | shape | v1 (shipped) | v2 | v4 | v5 |
|---|---|---|---|---|---|
| zipf (segments avg 80 rows) | gate/up 640 x 2560 | 3.36–3.39 | 3.35–3.39 | 3.41–3.42 | 3.27–3.30 |
| zipf | down 2560 x 640 | 4.30–4.38 | 4.30–4.39 | 4.31–4.43 | 4.28–4.29 |
| hot (a few experts, no segment under 64 rows) | gate/up | 2.39 | | | 2.25 |
| hot | down | 3.22 | | | 3.18 |

The prefetch distance (3, 6, 10) moves nothing on either form. The
reading was wrong about the bound: under Zipf routing the launch is a
mix of weight streaming (419 MB a projection, a 1.5 ms floor at line
rate; a segment under 16 rows is pure streaming) and compute, and the
kernel sits within ~1.45x of the larger floor, so the warp tile returns
2–3 %; under hot routing — the in-situ regime, where the recorded
profile's launches take 1.9 ms (gate/up) and 2.4 ms (down) per
4,096-token chunk — the kernel runs at 56–60 TF, 63–67 % of the
measured dense bf16 rate (Marlin's class), and the four-warp
register-decode form is 6 % faster on gate/up, 1.5 % on down: about −2 %
of the prefill in situ, not the −10 % estimated. The lever that remains
in the hot regime is B-tile reuse across more rows — a 128-row block
tile (the decode and the B ldmatrix amortized over twice the rows) with
a tile list that emits 128-row tiles for the large segments — worth
perhaps another 8–10 % of the launch, i.e. −2..−3 % of the prefill.

On the fabric (record sessions R21 / R21b; one binary, the form by env;
cold prefill 2K / 8K / 32K, the 17K-prompt greedy sha every leg
927ffd8a): a checkpoint transfer on the node from 16:28 to 16:35 spoiled
one base and one four-warp leg (repeats at 6.7–7.0 s for 8K and 22.8 s
for 32K); the clean legs around it:

| leg | 2K | 8K | 32K |
|---|---|---|---|
| base1 (cold) | 1.339 | 4.449 | 17.628 |
| wide4r1 | 1.297 | 4.342 | 17.686 |
| base3 | 1.293 | 4.376 | 17.689 |
| wide4r3 | 1.301 | 4.326 | 17.639 |
| base4 | 1.294 | 4.381 | 17.698 |

Against the 0.1 % base band: +0.6 / −1.2 / −0.3 %. Real at 8K, level
elsewhere, bitwise. The eight-warp form stays the default (the headline
cells do not move); variants 4 and 5 stay in the tree as tested opt-ins
(`engine.expert_gemm`) and the `kWarps` parameterization is the
base for the 128-row tile. The estimate that opened the round (−10 to
−12 %) assumed an issue-bound kernel; the measurement says the in-situ
launches already run at 63–67 % of the dense bf16 rate, so the remaining
expert-GEMM lever is B-tile reuse, worth a few percent, and the larger
prefill terms are now the dense stack (the fp8 GEMM at the bf16 rate,
§6.15) and the per-chunk glue.

## 7. Defect list

Each item is a measured or suspected distance from a floor, with its fix.
None is a reason to lose a cell; each is work.

1. **Chained draft acceptance below the model's.** Our depth-2 p2 measured
   0.6–0.84 on code, under what the model's own draft should give.
   Fix path: extend `tools/qwen_reference_dump.py --mtp` to chained depth
   (the reference draft rows at positions 2 and 3, fed exactly as the
   SGLang semantics we implement), gate `qwen_forward_test` and the real
   `qwen_forward_check` on them, and fix whatever the diff names. If the
   reference itself reproduces our lower rate on our prompts, the gap is
   prompt mix and the bench's prompts settle it.
2. **The n-gram gather serializes page faults.** Measured 2026-09-29: the
   PLE layer's host node leaves the GPU idle 772 us per pass (the staging
   is forked at the pass's start and joined at layer 2, ~1.9 ms later, so
   the callback runs ~2.7 ms for 64 rows of a 48 GB mmap; resident is
   impossible on one Spark — 137.9 GiB). Next: time the callback itself,
   then fault the rows in parallel.
   Original note: One host thread copies the
   rows out of the mapping, so a fresh prompt pays one NVMe latency per
   row in series; the field report's 3–4x slower fresh-text prefill is this.
   Fix: issue the faults in parallel (a thread pool over the 16 rows x
   tokens, or `madvise(MADV_WILLNEED)` / `preadv2` batches ahead of the
   copy); floor = the NVMe's queue-depth rate, tens of microseconds per
   token batch. Applies to TTFT on the two short cells and to every cold
   prefill.
3a. **The speculative recurrent state's snapshots and commit copy (2026-09-29,
   from the profile).** Every depth-3 pass stored three post-row snapshots
   of every GDN layer's 3.15 MB state (340 MB) and, on the ~60 % of steps
   where a draft row failed, copied one back (227 MB, 1.05 ms serial): ~500
   MB of a 12 GB pass, and the snapshot write-back was what slowed the
   out_proj GEMV that follows the recurrence (93 us in the step against
   68 cold). Fix, built: the checkpoint-and-replay form (`KdaReplay`,
   kernels/kda.hpp) — the live buffer holds the state BEFORE the pass's
   rows, the pass saves its rows as their inputs (~21 KB a row: the
   post-conv q|k|v row, a_raw, beta_raw), and the next pass replays the
   `accepted` saved rows from the checkpoint before its own (the same
   per-token code on the same inputs: bitwise the snapshot it replaces).
   Per layer per pass: one state read and one state write, the T=1 floor;
   the commit copies 3 MB of saved rows per request and records the count;
   no snapshot rows (1.8 GB of device memory at four slots freed). Every
   other reader of the state — a prefill continuation, the prefix cache's
   snapshot — materializes the pending rows first (`materialize_gdn`).
   `DGPP_QWEN_GDN_REPLAY=off` keeps the snapshot form for the A/B.
3. **Chain-row latency, ~5 ms per row over its bytes.** Ring copy aside and
   back on the main stream, the mixer and pick as separate small launches,
   the head not prefetched during the layer. Fix: side-stream copies joined
   by events, the head prefetch window issued at the chain row's start,
   fused mixer + norm, a single pick launch; the graph is captured, so every
   launch removed is free at replay.
4. **Dense-stack kernels under line rate.** GR down + inject at 55 % (320
   rows over k = 10,240 is 41 blocks on 48 SMs), shared tail at 70 %, the
   single fp8 GEMVs at 90 % on rows under 512 bytes. Fix: split-k GR down,
   the fused tail on the fp8 chain, the narrow-row fp8 core; all bitwise by
   the existing fused-kernel gates.
4a. **The GR combine as its own launch after every site (96 per pass) and
   the gate activation before every up GEMV (96).** Built 2026-09-29: the
   activation is staged inside the batched rows' up GEMV (`gr_act_up<4>`,
   round 1) and a site's combine rides the NEXT site's group-norm launch
   (`qwen_gr_combine_norm_bf16`: the block that normalizes a (row, group)
   slice applies R += bf16(y * gate) first, then group_rmsnorm_kernel's
   reduction over the updated values — the same per-thread element order
   and fmaf chain, so R and Rn are bitwise the two-launch chain's; round
   2). Any other reader of R — the PLE layer's finish, the layer-state
   capture — applies the pending combine as its own launch first.
5. **Launch seam and inter-kernel gaps, 3 ms.** The speculative N+1 launch
   with rollback (documented, not built) removes the seam; node fusion the
   gaps.
6. **Packq slot kernels unmeasured.** S1 bench before anything else; the
   fp4 core is the template for reaching line rate at these widths.
7. **Prefill re-reads experts below the tile threshold.** D11; the
   threshold is measured, and the grouped GEMV's 4-row pass is replaced by
   the tile from the crossover down.
8. **Graph memory at depth 3, C4, and the candidate width.** Not in the
   memory plan; the startup log says what they cost; the plan gains the
   line.
9. **Cold boot transposes 56 GiB on the CPU.** Once per image; the device
   transpose kernel if it exceeds a minute.
10. **`usage.completion_tokens` accounting** must include reasoning tokens
    for the bench script's token counts; checked in S5.

Out of scope, stated: worlds 2 and 4 (D8), the base hybrid repo with BF16
draft experts, the vision tower.

## 8. Estimate

| stage | days | needs |
|---|---|---|
| S1 kernels + bench | 1.5–2 | a GPU beside the serving process |
| S2 config / binding / loader / fixture / tests | 1.5–2 | same |
| S3 dense fp8 as shipped + packed head | 1 | same |
| S4 MTP experts, table dir, plan, recipe, download | 1 | same |
| S5 real-checkpoint gates | 1 | one idle Spark, the checkpoint downloaded, the FP8 snapshot present |
| S6 throughput program (chain latency, gather, dense kernels, seam) | 3–5 | the same Spark |
| S7 ship | 0.5 | — |
| total | 9.5–12.5 | |

Builds go in a worktree with its own build directory (never overlapping
builds in `build-ci`); a full rebuild precedes every ctest verdict.
