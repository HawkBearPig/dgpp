# NVFP4 routed experts for GLM-5.3-Flash — implementation plan (2026-09-08)

The composed checkpoint `dgpp/GLM-5.3-Flash-NVFP4-FP8` uses NVFP4
routed experts from `dabsLabs/GLM-5.3-Flash-NVFP4` in layers 3–44.
All other tensors, including the MTP layer, retain the FP8 release's
bytes. `tools/compose_nvfp4_hybrid.py` creates the checkpoint and verifies
the copied tensors against their sources.

The loader, decode kernels and grouped prefill path support this format.
This document records their design, numerical rules, validation and
optimization experiments from 2026-09-08 onward. Section 6a contains the
dated implementation results; estimates in the original work plan are
planning context. Current serving measurements are in
[benchmarks](benchmarks.md).

## 1. The checkpoint (done)

| role | source | revision | bytes |
|---|---|---|---:|
| routed experts, layers 3–44 (`weight_packed` U8 [N, K/2], `weight_scale` F8_E4M3 [N, K/16], `weight_global_scale` F32 [1]) | dabsLabs/GLM-5.3-Flash-NVFP4 | `bb176861` | 171.0 GB |
| everything else, layer 45 included (`weight` + `weight_scale_inv` for the FP8 classes; BF16/F32 otherwise) | unsloth/GLM-5.3-Flash-FP8 (= zai-org/GLM-5.3-Flash) | `a160e229` | 24.1 GB |

195.1 GB in 40 shards, 112,396 tensors, snapshot revision
`833e7cc0e86d9eb125551660dbf48cab2e224df9` (a hash of the composition plan, so
every node produced the same revision; `MANIFEST.json` carries per-shard
sha256s and the census). No arithmetic was performed: every output tensor was
re-read and compared byte for byte with its source on write. `config.json`
carries `quantization_config.quant_method = "dgpp_mixed"` describing both
groups. Tool flags `--mtp-from experts` / `--dsa-from experts` exist to take
those classes from the BF16 repo's bytes instead; the first build takes
neither, deliberately (§9.2).

Dequantization of a routed-expert element, as the format defines it and as
verified against the FP8 checkpoint (0.0829 relative error, the card's 0.0828;
swapped nibbles 1.41):

    w = e2m1(code) * (float32(weight_scale) / weight_global_scale),   low nibble = even element

## 2. Component map

### 2.1 Reused unchanged

- The whole non-MoE stack: KDA (`kda_layer.cu`, `kda.cu`), DSA (`dsa_layer.cu`,
  `dsa.cu`), mHC, norms, embeddings, the heads, the bf16 GEMV, cuBLASLt.
- The MoE machinery around the experts: router dots and select, device
  segmentation, slot ordering, the fp32 accumulation chain and its single
  rounding, the expert-view table ring and per-graph-slot tables, route traces,
  the swiglu semantics (`glm_moe.cu`, `moe_layer.cpp`).
- The FP8 path, byte for byte: `quant_matrix.hpp`, `fp8_gemv.cuh`,
  `scale_gemm.cu`, `fp8_dequant.cu`, the loader's `load_quant_rows/cols`, the
  DSA dequant bridge. The hybrid's shared experts, dense MLPs, DSA projections
  and MTP-layer experts flow through these exactly as the FP8 checkpoint's do.
- Loader infrastructure: `GlmLayerBump`, the staging mirror, counting-mode byte
  formula, the resident image (its key already folds every shard header and
  `config.json`, so the new model id gets its own image without a format bump),
  the replicated digest, `hf_cache`, the safetensors reader (`U8` and
  `F8_E4M3` already parse).
- The e2m1 codec in `kernels/latent_format.hpp` (`fp4_e2m1_bits_to_float`,
  encoder, tests) for host oracles and fixtures.
- Graphs, the batch family, the prefix cache, MTP, scheduler, serving, the bus.
- Evidence tooling: `scripts/fabric_run.sh`, `fabric_xcript.py`,
  `fabric_logprob.py`, `moe_slot_bench` (extended, §2.3), `step_probe.py`.

### 2.2 Modified

| file | change |
|---|---|
| `models/glm/config.{hpp,cpp}` | parse `quantization_config` into a `GlmQuantProfile` — routed experts `Fp8Block128` (quant_method `fp8` or absent: today's behaviour, bit for bit) or `Nvfp4Group16` (`dgpp_mixed`); reject anything else loudly |
| `models/glm/binding.{hpp,cpp}` | the expected table emits the NVFP4 triple for routed experts under that profile (`add_nvfp4`); the validator checks dtypes and shapes of all three (U8 [N, K/2], F8_E4M3 [N, K/16], F32 [1]) and counts them; the FP8 profile's table is unchanged (pinned by the existing binding tests) |
| `models/quant_matrix.hpp` | add `GlmFp4Matrix {payload, scales, global_scale, rows, cols}` beside `GlmQuantMatrix` (which is not touched) and `fp4_rows_view`; the slice contract is 16-aligned columns (one scale block) and even columns (one packed byte) |
| `models/glm/moe.hpp` | `GlmMoeWeights` carries the routed experts in either format; `MoeExpertView` gains the scale-block pointer's partner (the global scale), so the device tables and their byte formulas follow automatically |
| `models/glm/loader.{hpp,cpp}` | `build_moe` under the NVFP4 profile: `load_fp4_rows` (gate/up: contiguous payload and scale rows) and `load_fp4_cols` (down: the rank's I/world columns = a strided copy of I/(2·world) payload bytes and I/(16·world) scale bytes per row) and the global scale; the read-class assertions and byte accounting extend to the triple; `GlmMoeResident` holds both formats |
| `models/glm/tp.{hpp,cpp}` | `GlmTpViews` (the parity reference that carves a full layer) gets fp4 row views and an fp4 down column pack, so the shard-parity gate keeps pinning the sharded loader against it |
| `models/glm/moe_layer.{hpp,cpp}` | dispatch on the expert format at the three chains (`enqueue`, `enqueue_prefill`, `enqueue_decode`); geometry check for fp4; table fills carry the new field; scratch is format-independent |
| `models/glm/decode.cpp` | `prefetch_quant` twin for fp4 (payload + scale bytes); the prefetch windows re-tuned once the byte mix changes (§6) |
| `kernels/glm_moe_launch.hpp`, `glm_moe.cu` | the slot and grouped kernels take the fp4 views; the shared expert stays FP8 inside the same launches (§3.3) |
| `CMakeLists.txt` | `CMAKE_CUDA_ARCHITECTURES 121a` (the FP4 PTX — `cvt.rn.f16x2.e2m1x2` and the block-scaled MMA — assembles only for `sm_121a`; plain `sm_121` refuses both); new sources, tests and benches |
| `tests/cuda/glm_fixture.hpp`, `glm_tp_fixture.cpp` | the fixture writer emits NVFP4 triples under the profile (valid nibbles, e4m3 scales without the NaN codes, positive global scales); a second TP fixture directory for the fp4 profile so every fixture-driven gate runs on both |
| `tests/cuda/glm_binding_test`, `glm_loader_test`, `glm_moe_test`, `glm_tp_test` | fp4 twins of the gates in §5 |
| `models/glm/moe_reference.{hpp,cpp}` | the double oracle dequantizes fp4 experts with the formula above; `GlmMoeHostWeights` holds the triple |
| `tools/checkpoint_audit.py` | classify the triple, regenerate `docs/checkpoint_budget.md` for the hybrid |
| `tools/glm_reference_dump.py` | the nvfp4 dequant for the transformers-parity dumps |
| `scripts/*` (18 places), `deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1.example.json` | the hardcoded `unsloth/GLM-5.3-Flash-FP8` becomes a variable with the hybrid as an alternative value |
| `DESIGN.md` §3/§4, `docs/operations.md`, `README.md`, `CHANGELOG.md` | the hybrid checkpoint, the NVFP4 contract, the model id |

### 2.3 NVFP4 components

| item | purpose |
|---|---|
| `kernels/fp4_gemv.cuh` | the decode core: warp per weight row, 16-byte payload chunks (32 codes, two scale blocks), e2m1 decoded in registers, `bf16(fp4·s)` exact, fp32 FMA against staged bf16 activations, ÷ global scale in the epilogue; multi-row-per-warp for the 256-byte down rows |
| fp4 slot kernels (`glm_moe.cu`) | `slot_gate_up_swiglu` and `slot_down` over routed fp4 views with the FP8 shared slot in the same launch |
| fp4 grouped GEMV kernel | the host-path twin the decode slot path is pinned against bitwise (the fp8 kernel's role today) |
| fp4 grouped tensor-core kernel | the prefill kernel: the fp8 tile kernel's structure with an fp4 weight-tile decode (8-byte loads of 16 codes per thread per 64-deep stage, 4 scale bytes per row per stage), bf16 `mma.sync` unchanged |
| `tests/cuda/fp4_gemv_test.cu` (+ checkpoint-slice mode) | oracle, exactness lemma, saturation and NaN policy, row-count invariance, real hybrid-checkpoint slices |
| `tests/unit/fp4_test.cpp` | codec table, the exactness lemma over every (code, scale) pair |
| `benchmarks/micro/moe_slot_bench --format fp4`, `gemv_bw` fp4 mode | the effective-bandwidth proof before integration |
| `tools/compose_nvfp4_hybrid.py` | done |
| `docs/nvfp4_plan.md` | this |

## 3. Kernel design

### 3.1 The numerics contract

Every routed-expert value the kernels use is `fp4 · s` held in bf16, and that
value is EXACT: an e2m1 code has at most 2 significant bits, an e4m3 scale at
most 4, so the product has at most 6, inside bf16's 8; the exponent range
(2^-10 … 2^12) is inside bf16's. The product with a bf16 activation (8 bits) is
exact in fp32. The accumulation is fp32 FMA in a fixed order, and the one
inexact operation of the format — division by the per-tensor global scale —
happens once per output, on the finished dot, as a correctly rounded fp32
division. Then the layer's existing rounding points apply unchanged: bf16 after
gate/up (inside the fused swiglu), fp32 out of the down projection into the
ordered chain, one bf16 rounding as the sum leaves for the all-reduce.

This has strictly fewer roundings than the FP8 path (which rounds
`bf16(e4m3 × fp32 scale)` per weight). It is not bit-identical to a
transformers-style reference, which rounds each weight to bf16 after dividing
by the global scale; as with the fp32 MoE chain (decision of 2026-09-02) the
double oracle is the truth and the per-element ulp budgets are the gate. Gate
and up carry separate global scales in this checkpoint and keep them.

NaN policy: e2m1 has no NaN; an e4m3 scale code 0x7F/0xFF is NaN and
propagates exactly, as the FP8 core's does. Saturation cannot occur (the
values are bounded by 6 × 448).

Row invariance: each output row's arithmetic must not depend on how many rows
share a launch (the batch family and the scalar↔batch transitions rely on it).
The fp4 core keeps the fp8 core's shape — per-lane chunk order, fixed shuffle
tree — so the property carries over and is gated.

### 3.2 The decode GEMV

Bytes are the whole cost at 1–8 rows (2 to 16 FLOP per byte against a ridge
near 400), so the design target is the same fraction of line rate the fp8 core
holds with half the bytes: the T=2 profile has gate_up at 295 GB/s effective
and down at 224. Design points, each decided in `moe_slot_bench`:

- e2m1 → f16 by `cvt.rn.f16x2.e2m1x2` (two codes per instruction, `sm_121a`)
  or the permute/LUT trick if the convert's throughput disappoints; f16 → f32
  is exact.
- the scale bytes: `[N, K/16]` e4m3 rows beside the payload (256 B per gate/up
  row, 32 B per down row) read as 2-byte loads per chunk, or interleaved with
  the payload at load time (the loader may lay the resident bytes out any way
  it likes) — measure both.
- the down projection's rows are 256 bytes (k = 512 nibbles per rank): a lane
  owning 16-byte chunks covers a row with 16 lanes, so the kernel needs
  ≥ 8 rows per warp to keep the fp8 kernel's bytes in flight (the fp8 down
  moved from one to four rows per warp for the same reason).
- gate and up in one pass over the staged activation, swiglu fused, exactly
  the fp8 kernel's structure.

Expected per-rank bytes per token: routed experts 2.11 → 1.19 GB.

### 3.3 The shared expert inside the fp4 launches

The shared expert is FP8 (the release's bytes) and runs in the same launch as
the routed slots today (`resolve_slot_matrix`: slot j == top_k). The fp4 slot
kernels keep that: a block whose slot is the shared expert runs
`fp8_gemv::row_dots` (uniform per block), a routed block runs the fp4 core.
Both cores' arithmetic stays the pinned arithmetic of its format, so the
bitwise twins (host grouped kernel ↔ decode slot path) hold per format.

### 3.4 The prefill kernel

At the 2,048-token chunk the expert GEMM reads ~43 GB of fp4 experts per rank
(~180 ms at 240 GB/s) against ~8.7 TFLOP of MMA work (~60 ms): memory-bound
three times over, as the fp8 kernel is today (780 ms of 1,708 on the read
floor, round-9 record). So the prefill kernel is the fp8 tile kernel with an
fp4 weight-tile decode: 64 rows × 64 k per stage, each thread decoding 16
codes from one 8-byte load with the row's 4 scale bytes for the stage into
exact bf16 in shared memory, the same bf16 `mma.sync` and epilogue. The
native block-scaled FP4 MMA is not used on the main path (§9.3).

Expected: expert GEMMs 780 → ~400 ms per 2,048-token chunk (TTFT 1.71 → ~1.3
s); the 256-token prefill, almost entirely expert reads, 578 → ~400 ms.

## 4. Loader and TP

- The profile decides the table; the table decides the build; the build's
  counting mode is the byte formula. Nothing about the FP8 classes changes.
- Routed fp4 experts at world W: gate/up keep this rank's I/W rows (payload
  and scale rows are contiguous); down keeps this rank's I/W columns — a
  strided copy of I/(2W) payload bytes and I/(16W) scale bytes per row. The
  slice start must be a multiple of 16 (one scale block; 512 at W = 4). The
  existing `glm_tp_validate_geometry` (128-multiple quotients) already
  implies it; the fp4 view checks 16.
- The MTP layer's experts are FP8 and load through today's path.
- Resident per rank: ~82 → ~51 GiB (routed experts 76 → 43 GB). The freed
  ~30 GiB is KV pool and prefix-arena headroom (`--memory-plan` reports it).

## 5. Gates

Every gate below has its fp8 original; the fp4 twin is added, the original
stays and must stay green on the FP8 fixture and checkpoint.

1. Codec: e2m1 table (exists), the exactness lemma `bf16(fp4·s) == fp4·s` over
   all 16 × 256 pairs, NaN propagation.
2. `fp4_gemv_test`: oracle at random and ragged shapes, rows independent of
   row count, real slices of the hybrid checkpoint (the
   `scale_gemm_checkpoint` pattern), determinism.
3. Grouped fp4 MMA bitwise the fp4 tile reference per segment; both within
   budget of the oracle.
4. `glm_moe_test` fp4: decode slot path bitwise the host path; prefill path
   bitwise the host path; expert path within budget of the double oracle at
   the small geometry; sliced ranks' fold within the partials' budget of the
   unsliced oracle.
5. `glm_binding_test` / `glm_loader_test` fp4: table reproduces the layer,
   validator catches a wrong triple, layer bytes exact from the fixture,
   resident-image round trip bitwise, formula == allocator.
6. `glm_bind_check` on the real hybrid: 112,396 tensors, 0 missing, 0
   unexpected, every triple validated.
7. `glm_tp_test` on the fp4 fixture: shard parity (loader vs `GlmTpViews`),
   forward parity, decode-session parity, the graph and MTP gates, hot == cold.
8. Fabric (`scripts/fabric_run.sh`, one procedure at a time): transcripts vs the
   FP8 build judged by `fabric_xcript.py` (near-tie flips only); op-stream md5
   identical on four ranks; same binary twice → delta exactly 0;
   `fabric_logprob.py` on the three teacher texts vs the FP8 build — this
   number is the experts' cost and is recorded, not gated at 0.02 nat.
9. FP8 regression guard: the FP8 checkpoint's full CI and the fabric step
   timings (41.9 ms short-context T=1, 22.45 ms/token MTP) re-measured on the
   final binary.

## 6. Performance work, to the floor

The step's floor is bytes ÷ bandwidth; the kernels' job is to sit on it.

| stage | measurement | target |
|---|---|---|
| fp4 GEMV core | `moe_slot_bench --format fp4` at 1, 2, 8 rows | effective GB/s ≥ the fp8 core's at the same rows (gate_up ≥ 290, down ≥ 220 in the T=2 shape) |
| prefill kernel | nsys on a 2,048-token chunk | expert launches at ≥ 75 % of DRAM peak (the fp8 kernel's fraction) |
| L2 prefetch | `DGPP_L2_PREFETCH_*` sweep at T=1 and T=2 | the boundary and layer windows re-sized for the smaller shared/expert bytes; the "light" rate re-confirmed |
| step | `glm_gen_check --decode-graph [--mtp]`, `step_probe.py` | T=1 31.3 → ~27.5 ms; MTP replay 42.4 → ~35 ms (22.45 → ~18.6 ms/token) |
| service | `serve_bench.py`, the batch family | scalar curve and 2/3/full batch re-measured; expect the batch gains > single-stream (expert bytes dominate at occupancy) |
| prefill | `fabric_prefill_repeat.sh` | 256 tokens ~0.40 s, 2,048 ~1.3 s |
| memory | `dgpp-serve --memory-plan` | resident ~51 GiB/rank; raise `kv_capacity` / prefix arena accordingly |

The ledger these targets come from (per rank per token at TP=4, from
DESIGN §3's 5.855 GB):

| component | today | hybrid |
|---|---:|---:|
| routed experts | 2.11 GB (FP8) | 1.19 GB (NVFP4) |
| shared experts + dense MLPs | 0.37 GB (FP8) | 0.37 GB (FP8, same bytes) |
| KDA projections | 2.34 GB (BF16) | 2.34 GB |
| DSA, head slice, router, mHC, norms | ~1.0 GB | ~1.0 GB |
| weight floor at 240 GB/s | 24.4 ms | ~20.5 ms |

### 6.1 Decode headroom after phase 2 (2026-09-08, parked)

The hybrid step is 28 ms against a weight floor of 17-20 ms (4.9 GB per
rank per token at the fp8 cores' 240-287 GB/s; 18 ms at the GB10's
theoretical 273). The remainder: 94 collectives ~5 ms (median 48 us,
floor 23), mHC and small kernels ~3.5, launch gaps and idle ~1.5.

Without a quality decision, ~3-4 ms (to ~24 ms/step, MTP ~16.5 ms/token):
the fp4 GEMV core to the fp8 core's rate (200 → 235-240 GB/s, ~1 ms); DSA
projections consumed as FP8 natively instead of the BF16 bridge (~1.4 ms,
no new rounding, the FP8 model gains too); launch/interface pipelining (0.5-0.8
ms); L2 prefetch windows retuned for the smaller expert bytes (0.3-0.5 ms);
collective skew (up to ~2 ms, uncertain). With one quality decision: KDA
projections BF16 → FP8 block scales, the largest byte term, ~4.5-4.9 ms
(−16 %), a new rounding on the recurrent path — its own gate. Tokens per
step: a verify row's expert bytes fell from ~10 to ~5.5 ms, so MTP depth
2 costs ~+7 ms instead of +12 and the FP8 break-even (code/JSON only) may
now clear prose — re-measure with `scripts/mtp_depth_check.py`. Past the
floor only fewer bytes or more tokens per step move the number.

## 6a. Status

- Phase 0 (the checkpoint): done 2026-09-08, all four nodes, identical hashes.
- Phase 1 (format plumbing): done 2026-09-08 — `GlmExpertFormat` profile,
  tensor roles and the NVFP4 triple in the binding table, `GlmFp4Matrix`,
  the loader's fp4 row/column slices and gathered global scales,
  `GlmTpViews` fp4 views and packs, the fixture writers, and the gates:
  unit (binding table, validator, config profile), loader (byte-exact at
  world 1, the sharded inter slice against host oracles at world 2, the
  resident-image round trip), shard parity on the NVFP4 fixture at worlds
  2 and 4 streaming and resident (1,582 and 3,164 surfaces bitwise), and
  `glm_bind_check` on the hybrid (112,049/112,049 matched, 36,288 triples
  and 1,050 FP8 pairs bound), and `glm_stream_check --rank 0 --world 4`
  on the hybrid: all 46 layers built through the sharded loader with the
  formula reconciled on every load (KDA/MoE layers 1,049 MiB, DSA/MoE
  1,324 MiB, the MTP layer 2,143 MiB per rank; 199 s streaming from the
  checkpoint). Resident formula at world 4: 50.74 GiB per rank; the
  resident load itself waits for a node whose memory `dgpp-serve` is not
  holding. The FP8 checkpoint's gates are unchanged; ctest 34/34.
- Phase 2 (the decode kernels): built 2026-09-08 — `fp4_gemv.cuh` (the
  in-register e2m1 core, templated on K so every index is compile-time;
  the first draft's lane-dependent accumulator indexing spilled to local
  memory and ran at 14 GB/s), the single-matrix launcher, the fp4 slot
  kernels (gate_up+swiglu, down; the shared expert runs the fp8 core in the
  same launch) and the fp4 grouped GEMV; the host oracle's fp4 dequant and
  divisor epilogue. Gates: unit 127/127 (the exactness lemma over every
  code x scale pair, the bit-placement decode against the codec table),
  `fp4_gemv_test` 5/5 (oracle across every supported geometry with l2_rel 0,
  row-count invariance, the f32 epilogue, NaN scales, the geometry contract)
  plus two real slices of the hybrid at 0 mismatches, `glm_moe_test` 18/18
  with the fp4 twins bitwise (decode slot == host grouped == sliced fold).
  `moe_slot_bench` per layer, one rank's slice: rows=1 fp4 209 us vs fp8
  273; rows=2 334 vs 469; rows=8 929 vs 1,420. nsys at rows=1: fp4 gate_up
  117.8 us / down 56.8 (about 200 GB/s of payload+scales) vs fp8 160.3 /
  78.8 (about 235-240 GB/s) — the fp4 core moves fewer bytes at a lower
  rate; the remaining gap is §6's first item. The resident load of the
  hybrid at world 4 on a free node: 50.74 GiB per rank, materialized in
  91.45 s, the residency contract held (0 storage bytes on the re-pass).
  Fabric evidence, four nodes, 2026-09-08 (`build-ci/fabric-runs/
  nvfp4_text_0626`, `nvfp4_evidence_0632`):
  - Transcript, 64 greedy steps on the chat prompt: every rank identical,
    two boots identical; parts from FP8 at token 18 ("launches" vs
    "launching", FP8's margin 0.46 logits) — the hybrid is a different
    model, so the judge's wide-margin call is expected; the perplexity
    gate is the instrument.
  - Perplexity (teacher-forced, `fabric_logprob.py`): quick prose 2.633 vs
    FP8 2.587 (+0.0175 nat/token, 556 tokens); hard text 10.964 vs 10.892
    (+0.0066 +-0.0043 nat/token over 7,331 tokens, top-1 49.6 % on both).
    The mean deltas are inside the 0.02-nat bound, but the per-token
    movement is far above the kernel-only floor (mean |delta| 0.044 nat
    on the quick text for a reassociating kernel change): 0.124 nat on
    the quick text and 0.213 on the hard text (p5/p95 -0.57/+0.55), 1.6 %
    and 2.6 % of tokens moving > 1 nat, and the greedy pick differing
    from FP8's at 6.8 % and 12.0 % of positions. The moves are unbiased
    (they cancel in the sum; top-1 accuracy is unchanged), so the average
    quality holds while the token-level behaviour is measurably different
    — a different model of equal perplexity, not the same model. Task-level
    equivalence (tool calls, JSON, code) is not established by this gate.
  - Decode, one request, greedy, the decode graph: 28 ms/step vs FP8's
    31 (short context; the plan's estimate was 27.5). MTP graph step 36.5
    vs 41.6 ms.
  - Prefill, steady state (the phase-2 GEMV chain vs FP8's tensor-core
    kernel): 512 tokens 1,152 vs 740 ms; 2,048 tokens 4,396 vs 1,725;
    8,192 tokens 18,246 vs 7,463 — the phase-3 kernel's gap.
  - MTP, greedy, 300 tokens per class, ms/token effective (drafts
    accepted): chat FP8 24.3 (70.5 %) vs hybrid 21.7 (69.5 %); code 21.2
    (98.7 %) vs 18.7 (98.0 %); prose 21.7 (88.7 %) vs 19.3 (88.7 %); json
    21.0 (98.0 %) vs 18.6 (96.7 %); math 21.7 (92.3 %) vs 19.5 (88.7 %).
    The graph step 41.4-42.1 ms vs 36.4-36.9; acceptance within a few
    points of FP8 on every class. (The first pass showed 0.0 % on BOTH
    checkpoints: `glm_gen_check`'s recorded verify had compared against
    the eager token rows since the 2026-09-06 feed move — an evidence-app
    defect, fixed the same day; the serving adapter was never affected.)
  - The first four-node boot of a new checkpoint builds every rank's
    resident image (104 s on rank 0, 153-156 s on the peers) and rank 0's
    first collective waits for the slowest — not a prefill cost; the
    second boot restores in 13.5 s.
- Phase 3 (the prefill kernel): built 2026-09-08 —
  `moe_grouped_mma_fp4_kernel` (§3.4 as designed: the fp8 tile kernel's
  stages and MMA chain, the weight tile from 8-byte code loads and one
  scale byte per thread per stage, exact bf16, ÷g in the epilogue; any k
  multiple of 16), dense and grouped launchers, `GlmMoeLayer` dispatch
  (routed NVFP4 segments to the fp4 tile kernel, the FP8 shared expert
  to the fp8 one). Gates: grouped bitwise the dense form per segment at
  I = 208/320 (ragged n-tile, ragged last k-stage, z split, permuted row
  map, gate bf16 + down fp32); whole layer within the oracle budget on
  four geometries (max 0 ulps over soft); real slices 0 mismatches, row
  bits independent of m; `glm_moe_test` 20/20. Fabric, steady-state
  prefill (`fabric_prefill_repeat.sh --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1_large-cache.json`,
  `build-ci/fabric-runs/nvfp4_phase3_0750`): 512 tokens 635 ms (phase 2:
  1,152; FP8 740), 2,048 tokens 1,563 (4,396; FP8 1,725), 8,192 tokens
  6,796 (18,246; FP8 7,463). The 64-step chat transcript is identical to
  the phase-2 hybrid's (the prefill's rounding moved the step-18 margin
  from 0.26 to 0.15 logits without flipping it); all ranks identical.
  The §3.4 estimate (2,048 tokens ~1.3 s) is not reached, and the exit
  criterion "nsys at the read floor" is NOT met — by either kernel. nsys
  on rank 0 through three steady-state 2,048-token prefills
  (`nvfp4_phase3_0750/nsys`): per prefill the fp4 tile kernel takes 572
  ms (84 gate/up launches at 4.62 ms, 42 down launches at 4.38 ms) where
  the fp8 tile kernel took 780 ms (539 + 241); a launch moves ~340 MB of
  fp4 expert bytes (288 experts x 512 x 4096 x 9/16) in 4.6 ms = ~75
  GB/s, a third of the floor, and ~15 TFLOP/s of MMA. Halving the weight
  bytes bought only 27 % because the kernel is not bound by them: with
  BN = 64 every (segment, m-tile) re-reads its 128 x 4096 bf16 activation
  tile once per n-tile — 8x for gate/up, 64x for the down — so the A
  traffic (~2.4 GB per launch, from L2) is seven times the weight
  traffic; the stages are load → sync → mma → sync with no overlap; and
  at 2,048 tokens an expert holds ~57 rows, so the 128-row m-tile wastes
  more than half of the MMA work on zero rows. The rest of the prefill:
  bus collectives ~250 ms, mHC ~160, KDA ~129, attention ~62, accum ~60.
  The FP8 build on the same binary: 1,717 ms (1,725 before phase 3) — no
  regression. The fp4 kernel is the one to restructure (the fp8 kernel is
  the production path and stays): BN = 128 (halves the A re-reads),
  cp.async double-buffered stages, and a 64-row m-tile variant for short
  segments — §7 phase 5's "prefill tiling for short segments", now with
  the measured reason. Expected: the 572 ms toward ~250.
- Phase 4 (integration): 2026-09-08 — `deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1_large-cache.json` is
  the switch (site-local like `cluster.json`; `kv_capacity` 786,432 and
  `prefix_cache_gib` 8 spend the freed 31 GiB/rank: `--memory-plan` total
  96.2 GiB + 8 headroom against 116.5 free). L2 prefetch sweep on the
  hybrid (`nvfp4_phase4_1442`, greedy, 300 steps): T=1 28.0 ms/step at
  8/12/16/24 MB windows alike, 30.0 with the prefetcher off or the
  boundary rate at full, 28.0 with the layer rate at full; MTP 36.63 ms/
  step default, 36.38 at 16 MB, 36.31 at 24 MB, 38.61 off. The defaults
  (12 MB, light/light) stay: the window is flat at T=1 and under 1 % at
  MTP, inside run-to-run drift; the prefetcher itself is worth 2.0-2.3
  ms/step on the hybrid as on FP8. Serve path (`scripts/dgpp-cluster up
  --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1_large-cache.json`): boot 17.8 s from the images, the
  API check all green (stop, n, logit_bias, cached_tokens, reasoning
  tokens), the server's stats at 36.7 ms/step, 1.63 tok/step, 22.4 ms/
  token; teardown with four identical op-stream md5s, 0 STALLED, 0
  failures. `serve_bench.py` / `serve_soak.py` now address the served
  model (GET /v1/models) instead of a hard-coded FP8 id (they 404'd);
  bench: 200 tokens in 4.97 s cold / 4.62 s with the prompt cached. The
  10-minute mixed soak (`serve_soak.py`): 523 requests, 0 failed, 15 shed
  at the queue limit, 88 cancelled by design, the engine never faulted;
  31,882 tokens out; at four live requests 87 ms batch steps, 1.88-1.93
  tok/step/request, accept 87-91 %, 64-69 tok/s aggregate (11.8 ms/token);
  short-request TTFT p50/p95 581/1,711 ms, pace p50 58 ms/token under
  load; prefix cache 387 hits / 121 misses; four identical op-stream
  md5s, 0 STALLED, 0 failures on every rank. Gate 9 on the same binary,
  `deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1.json`: boot, API check green, 42.2 ms/step, 24.0 ms/
  token, 1.69 tok/step at 64 % acceptance (unchanged), 200 tokens in
  5.69 s over HTTP (the hybrid: 4.97), four identical op-stream md5s.
  Both worlds were torn down afterwards; the cluster is left down.
- Phase 5, the prefill tile kernel (2026-09-08): `moe_tile_bench` (the
  routed experts of one layer at the production shape, uniform or ragged
  segments, every kernel variant timed per launch, the pipelined kernel
  checked bitwise against the reference) and a decomposition of the
  reference kernel with its loads stubbed out settled where the time
  goes. Gate shape at 2,048 tokens (ragged, ~57 rows/expert): 3.95 ms per
  launch; with no weight loads 3.16, no activation loads 3.02, neither
  2.30 — the padded MMA, decode and barriers alone are 58 % of the launch
  at ~74 TFLOP/s of padded work (cuBLASLt's bf16 rate here is ~95), and
  the operand loads add ~1.65 ms because nothing overlaps them. Skipping
  the MMAs of a warp whose 16 rows are all padding (bitwise: those rows
  are never stored) took the floor to 1.70 and the launch to 3.81. At
  8,192 tokens the extra cost is the weight tile re-read once per
  128-row m-tile of a long segment (~4 ms of the 9.8). Tile width (64 vs
  128), stage depth (32 vs 64) and occupancy (3 vs 4 blocks/SM) moved
  nothing on the synchronous structure — except that 2 blocks/SM is a
  cliff (6.1 ms) — so the reference's shape stays. The pipelined 64 x
  128 x 32 cp.async kernel wins the down shape (3.76 vs 4.13) and loses
  the gate shape (5.05 vs 3.81): `GlmMoeLayer` now dispatches by shape.
  Fabric: prefill 563 / 1,502 / 6,572 ms at 512 / 2,048 / 8,192 (phase 3:
  635 / 1,563 / 6,796; FP8 740 / 1,725 / 7,463); expert launches per
  2,048-token prefill 525 ms (gate/up 4.37, down 3.76 ms per launch);
  transcript identical. Left for this kernel, with the measured reason:
  a multistage cp.async pipeline (3-4 stages of a 64-deep A tile plus the
  raw fp4 codes, the decode off the ring) and a taller block tile so a
  long segment reads its weights once — the remaining ~2 ms per launch
  of un-overlapped loads and the 8K re-reads. Nsight Compute needs
  `NVreg_RestrictProfilingToAdminUsers=0` on the head to read counters
  (ERR_NVGPUCTRPERM as this user).
- Phase 5, the prefill kernel, second pass (2026-09-08 evening): the
  ldmatrix fp4 tile kernel (`moe_grouped_mma_fp4_ldm_kernel`, bench
  variant 13, now the production launcher on both shapes). Counters on the
  three earlier forms said: the reference stalls on loads and barriers at
  34 % issue, the 64 x 128 x 32 two-stage form on the MIO queue (20
  LDS.32 per eight mma), the three-stage ring on its lone block's warps.
  The new kernel: 64 x 128 x 64 tiles, eight warps as 1 (m64) x 8 (n16) so
  every B fragment is decoded once per block and feeds four MMAs; A by
  ldmatrix.x4; a three-slot cp.async ring (46.5 KB, two blocks per SM) of
  the activation tile and the RAW fp4 codes and scales — no decoded bf16
  tile: the B fragments are decoded at fragment time from one LDS.64 of a
  row's 16-code group (cvt e2m1x2 -> f16x2, x scale in f16, -> bf16x2,
  every step exact). Three findings made it fast, each measured:
    * the DRAM saw 32-byte requests (a stage reads 32 bytes of each
      weight row, rows 2 KB apart): a `prefetch.global.L2` of each row's
      next 128-byte line, four stages ahead, took the gate from 4.84 to
      3.33 ms (distance 1 line best; 2-3 within 3 %);
    * a third of the activation re-reads across the four n-tile blocks
      missed L2 while the weight stream passed through it: a
      `prefetch.global.L2::evict_last` on each activation line before its
      copy took the L2 read hit rate from 88 to 96 % (the cache-hinted
      cp.async form faulted with an illegal instruction inside the kernel
      although every form ran in a probe — not pursued);
    * the m-tile as the FASTEST grid index, one 64-row m-tile per block:
      a long segment's m-tiles run together and share its weight slice
      through L2 (as blockIdx.z they ran a grid apart and the slice came
      from DRAM per m-tile) — 8K gate 9.85 -> 5.0 ms.
  Bench (ragged segments): gate 1.90 / 2.33 / 5.0 ms at 512 / 2,048 /
  8,192 tokens against the reference tile's 2.76 / 3.70 / 9.9; down 2.03 /
  3.20 / 7.89 against the two-stage kernel's 2.61 / 3.84 / 9.58. Bitwise
  the reference on both shapes (the bench's checks, glm_moe_test's segment
  pin — which caught a real bug: at k = 208 the 16-byte code copies and
  4-byte scale copies were misaligned for odd rows; aligned-width
  fallbacks now cover every k % 16 == 0). Fabric, steady-state prefill
  (`ldm_prefill_2138`): 512 tokens 460 ms (was 563; FP8 740), 2,048
  tokens 1,335 (1,502; FP8 1,725), 8,192 tokens 5,954 (6,572; FP8 7,463);
  generated ids identical to every earlier phase-3/5 run at all three
  lengths, all ranks identical; ctest 35/35. The kernel now runs at
  ~146 GB/s of its weights on the 2K gate with L2 at half its peak and
  DRAM at three quarters; issue active 21 % at 16 warps per SM — the
  remaining structure cost. Next on the prefill, by size: the bus bulk
  collectives (243 ms of the 2K prefill, paced transport), the KDA
  recurrence (129, a sequential scan), the mHC tiled dots (105: a
  tensor-core GEMM would take them to ~20 within the oracle budget, the
  prefill/decode divergence at rounding level the expert path already
  accepts), the down kernel's eight-stage blocks (prologue-exposed).
- Phase 5, the prefill's mHC dots on tensor cores (2026-09-08 evening):
  the token-tiled kernel streamed the 786 KB coefficient matrix through
  shared memory once per four tokens with 96 fp32 accumulators a thread —
  1.17 ms per site at 2,048 tokens, 105 ms of the prefill. The dots are a
  [tokens x 4D] x [24 x 4D]^T GEMM: `mhc_dots_mma_kernel` runs it with
  bf16 mma.sync over a four-slot cp.async ring (32 tokens x 64 by 32 rows
  x 64; two blocks per SM), gathers each token's sum of squares from the
  same tiles and applies inv_rms in the epilogue; the standard finish
  kernel follows (comb in-block). Two findings on the way: (a) one MMA
  accumulator over the 16,384 terms flipped the bf16 rounding of 0.5 % of
  the mixing coefficients against the double oracle (the tensor core's
  accumulation truncates a small addend against a large sum) and failed
  the end-to-end budget; per-stage partials summed in fp32 round-to-
  nearest keep the flips at the fp32 chain's rate and pass it; (b) the
  finish read the streams twice (a sum-of-squares pass, then the
  collapse) — the sum now rides on the GEMM's tiles and the finish reads
  them once. Also: the sites never read the collapsed row (only normed),
  so the decode path passes a null collapsed and the 16 MB per site of
  dead stores go. NOT bitwise the per-coefficient form (the prefill's
  expert path already parts from decode's GEMV core the same way): within
  the oracle budgets (glm_mhc_test: the gemm form at 16 / 70 / 2052
  tokens, the end-to-end pipeline), the tiled form kept behind
  `mhc_set_prefill_gemm(false)` for its bitwise pin. `mhc_site_bench
  --gemm`: 1,196 -> 836 us per site at 2,048 tokens, 4,873 -> 3,380 at
  8,192 — both halves now at the DRAM rate (the dots 64 MB of streams +
  the matrix, the finish 64 MB read + 32 MB written); the third pass per
  site (the update) is the mHC's structure. Fabric (`mhcgemm_prefill_2155`):
  512 tokens 440 ms (460 before it), 2,048 tokens 1,290 (1,335), 8,192
  tokens 5,769 (5,954); the 64-step chat transcript identical to phase 5's
  (fabric_xcript: 64/64, three steps within a bf16 ulp); the 512-token
  prompt's 2-step continuation flips at step 1 on a 0.049-logit margin —
  a certified near-tie flip (`fabric_xcript.py`: <= 1 bf16 ulp); all ranks
  identical; ctest 35/35.
- Phase 5, the fp8 tile kernel ported to the ldmatrix form (2026-09-08
  late): `moe_grouped_mma_fp8_ldm_kernel` — the fp4 kernel's structure for
  FP8 block-scaled weights (the FP8 checkpoint's routed experts, both
  checkpoints' shared experts): 32-deep stages so the doubled code bytes
  keep two blocks per SM (four 11 KB slots), the stage's one 128 x 128
  block scale cp.async'd into the slot with the tile, B fragments decoded
  at fragment time as e4m3x2 -> f16x2 -> f32 x scale (RN) -> bf16 (RN),
  the reference's roundings op for op — bitwise the reference tile kernel
  (the bench's checks on both shapes at 512 / 2,048 / 8,192 tokens, the
  segment pin in glm_moe_test). `launch_moe_grouped_mma_{bf16,f32}` run it
  for k % 32 == 0 (the production 4096 and 512); other widths, the dense
  form and the pin keep the reference (`moe_set_fp8_ldm(false)` for the
  A/B). Bench, gate / down per launch: 512 tokens 4.23 -> 3.02 / 4.06 ->
  3.37 ms; 2,048 tokens 5.41 -> 3.63 / 5.54 -> 4.60; 8,192 tokens 13.4 ->
  7.1 / 13.5 -> 9.9. The down's remaining gap is its 268 MB of fp32
  partial output per launch (the ordered accumulation's input) on top of
  the 604 MB of weights: it runs at ~85 % of that floor. Fabric
  (`fp8ldm_prefill_cluster_2240`, the FP8 checkpoint, the mHC GEMM form
  included): 512 tokens 549 ms (740 this morning), 2,048 tokens 1,412
  (1,725), 8,192 tokens 6,280 (7,463); the hybrid
  (`fp8ldm_prefill_cluster.nvfp4_2242`, its shared experts): 425 / 1,275 /
  5,702 (440 / 1,290 / 5,769); generated ids identical to the previous
  runs on both; all ranks identical; ctest 35/35.
- Phase 5, the decode core (2026-09-08): `moe_slot_bench --format fp4`
  swept the fp4 GEMV core's two tunables. More per lane is worse in every
  direction (4 steps: +18 %; 8 chunks per lane: +29 %; both: +91 %); the
  gate width cannot take fewer than 4 chunks per lane (128 chunks over 32
  lanes). One row step per warp instead of two — twice the blocks, four
  loads in flight per lane — is 5 / 6 / 6.5 % faster at 1 / 2 / 8 rows in
  an alternated A/B of two binaries (199-204 vs 213 us per layer at one
  row, 320-325 vs 341-348 at two, 893-896 vs 951-957 at eight), the fp4
  core now at 177-181 GB/s of its bytes at one row against the fp8 core's
  203 and 221 vs 240 at two. A row's chain is unchanged, so the outputs
  are bitwise (the fp4 gates re-run). Fabric: T=1 28.0 -> 27.0 ms/step,
  MTP 36.6 -> 35.4 ms/step (21.2 -> 20.5 ms/token), acceptance unchanged.
- Phase 5, MTP depth 2 on the hybrid (2026-09-08, the serve path,
  `--knobs "--mtp-depth 2"`, 300-token answers): short prompts 17.4-17.5
  ms/token at depth 2 vs 18.5-18.6 at depth 1 (43.2-43.8 vs 35.1-35.3 ms/
  step, 2.49-2.51 vs 1.89-1.91 tok/step, p1 87-90 %, p2 59-61 %) — prose,
  code and JSON all 4-6 % faster; at a 7,368-token context (summarize the
  hard text) 21.0 vs 19.7 ms/token (45.2 vs 36.1 ms/step, 2.16 vs 1.83
  tok/step, p1 74 %, p2 41 %) — 6.6 % slower. The same shape as the FP8
  study: the second draft pays only where it stands, and it stands less
  at long context. Depth 1 stays the default in `cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1_large-cache.json`;
  the lever remains per-request adaptive depth (both scalar variants
  captured per slot, switched on the observed p2), not a global setting.
- Item 2 of 2026-09-08's decision list, the DSA projections consumed as
  FP8 directly (commit follows): the loader keeps q_a / kv_a / q_b /
  o_proj as the checkpoint's pairs at aligned worlds, the layer runs them
  through the scale-aware GEMM (with an output row stride for the fused
  [q_a | kv_a]), the views slice the pairs or dequantize them for the
  bridge form at misaligned worlds; gates: the DSA layer in both forms
  against the reference, the loader's pairs byte-exact, shard parity at
  worlds 2 and 4, the full TP suite, ctest 35/35. Fabric
  (`dsa_direct_1658`): FP8 31.0 -> 30.0 ms/step at T=1 and 41.4 -> 40.6
  under MTP; the hybrid 27.0 -> 26.0 and 35.4 -> 34.4 ms/step (20.5 ->
  20.0 ms/token). FP8's transcript parts from the morning's on a near-tie
  flip (<= 1 ulp) and its quick-text perplexity is unchanged within the
  kernel floor (mean -0.004 +- 0.005 nat, mean |delta| 0.053 vs the 0.044
  floor, 0 big moves); the hybrid's transcript is identical. The first
  boot after the image format bump rebuilds every rank's image; a serve
  boot then can fail on the collective deadline (docs/operations.md).
- Item 1, the prefill pipeline, measured and parked: a three-stage
  cp.async ring (the reference's tile, the activation tile and the raw fp4
  codes two stages ahead, decoded off the ring into a double-buffered bf16
  tile) is bitwise the reference and, on a quiet GPU, 3.69 vs 3.73 ms on
  the gate shape at 2,048 tokens, 9.04 vs 9.97 at 8,192 (9 %), and slower
  on the down shape (4.65 vs 4.12; the 64 x 128 x 32 kernel's 3.81 stays).
  Its 73 KB of shared memory allows one block per SM, and one block's own
  decode -> barrier -> multiply -> wait sequence idles the tensor pipe
  where the reference's three resident blocks interleave. Kept as the
  bench's variant 12, not dispatched. The next attempt is a 32-deep
  three-stage ring at two blocks per SM, or decoding the weights at
  fragment-load time (each warp decodes only its n-slice).
- Item 4, the task-level eval (`scripts/serve_eval.py`, greedy,
  reasoning_effort low, both configs on the identical items,
  `build-ci/fabric-runs/eval_0908_1625`): HumanEval FP8 155/164 (94.5 %) vs
  hybrid 157/164 (95.7 %) — 155 pass on both, 7 fail on both, 2 pass only
  on the hybrid; GSM8K (first 300 of the test split) 293/300 (97.7 %) on
  both — 290 pass on both, 4 fail on both, 3 flip each way; schema
  extraction 100/100 on both. Identical replies on 109/164, 94/300 and
  70/100 items: the two models write different text of equal quality,
  the same conclusion the perplexity pair reached at the token level.
- Decode budget of the hybrid, nsys on rank 0 over 300 graph replays
  (`decode_budget_1736`), kernel time per step at T=1: KDA bf16 GEMVs 10.4
  ms (2.34 GB at 257 GB/s — the DRAM ceiling), fp4 expert slot kernels 6.8
  ms (gate_up 107 us and down 55 us per layer: ~210 GB/s of their bytes),
  the in-graph collective kernel 3.2 ms (90 x 36 us against a 23 us
  floor), the FP8 projections 2.0 ms (faster than DRAM: the prefetcher
  has them in L2), mHC 1.5 ms in 180 launches, small kernels ~1.8 ms
  (router 0.4, KDA recurrence 0.3, conv 0.15, norms, attention 0.3) —
  which sums to the 26 ms step: no launch gap or idle is left. The
  prefetch kernels themselves burn 10 ms of GPU time per step on the side
  stream (180 launches), competing with the KDA GEMVs for DRAM inside the
  attention layers. Under MTP the same picture scaled by the two rows.
- A table decode for the fp4 GEMV core (the 256 bytes -> two e2m1 values
  in shared memory, the group scale folded once per 8 codes) was neutral
  at 1-2 rows (194-197 vs 199-204 us per layer) and 4 % slower at 8 rows:
  the core is not instruction-bound; reverted. The fp4-to-fp8 core gap
  (181 vs 203 GB/s at one row) stays open pending hardware counters.
- The collectives, `DGPP_BUS_TIMELINE=1` on the hybrid at T=1
  (`bus_timeline_1744`): 33 us per collective, 90 per step = copy 3.2 +
  handshake 12 (the first peer's payload after this rank staged) + skew
  14.7 (the last peer's) + fold 3.5; engine post 2.0. Every rank's own
  skew is 14 us and no rank is consistently slow (rank 2 arrives last 51
  of 90 times, rank 1 31, rank 3 8): jitter plus wire latency, as the
  2026-09-06 study found — ~1.3 ms per step that only a protocol change
  (speculative posting, fewer collectives) would recover. Parked. The
  prefetch layer windows: off, T=1 27.0 and MTP 35.1 against 26.0 / 34.4
  on — they pay; the defaults stand.
- The launch interface in the serve path (`serve_seam_1745`, one live request,
  `DGPP_PIPELINE_TRACE=1`): launch-to-launch 27.0 ms at T=1 (the app's
  step 26.0: ~1 ms of host work exposed with no draft tail to hide it)
  and 34.0 under MTP (the app's 34.4: hidden entirely by the draft block's
  tail). The adapter already launches replay N before settling N-1, so
  with MTP — the production configuration — the ahead-launch machinery
  of the decode ledger would buy nothing; parked as well.
- The mHC update folded into the next site's dots kernel (each of the
  24 blocks recomputing the updated streams into shared memory, block 0
  writing them, the finish reading them behind the fences): bitwise the
  pair on every output at every size, one graph node fewer per site — and
  slower: T=1 27.0 vs 26.0 ms/step, MTP 35.4 vs 34.4. The 24-fold
  redundant update (16K elements, five 16-byte loads per vector per block)
  outweighs the 3 us launch it removes; a slice-per-block update would need
  a grid-wide barrier inside the graph node. Reverted; the pair stays.
  Net of this round of decode levers without quantization: the DSA
  projections (1 ms) landed; the fp4 core, the collectives, the launch
  interface and the mHC fusion are measured and parked with their reasons.
- Hardware counters on the decode slot kernels (Nsight Compute as root:
  `sudo -n /usr/local/cuda/bin/ncu`, the profiling module left
  admin-only; reports under the session scratchpad `ncu/`). Both fp4
  kernels: 576 blocks, 4 per SM by registers, exactly three waves,
  schedulers idle 74 % of cycles, long_scoreboard the dominant stall
  (gate/up 10.4, down 14.8 cycles per issue); the fp4 gate/up alone
  stalls 9.5 on the shared-memory pipe (short_scoreboard 6.1 +
  mio_throttle 3.3; the fp8 kernel 7.8) — two codes per weight byte is
  twice the activation reads per byte. The pipelined fp4 tile kernel: 16 %
  warps active (one block per SM), stalls tiny per issue — starved of
  warps, not of memory. What followed (commit c0e52ff, all bitwise, the
  checkpoint's slices and glm_moe_test's twins): the activation window
  loaded once per chunk column and reused across row steps (the steps
  sweep re-run with it: one step 201, two 209, four 217 us at one row —
  one step stays); gate and up issued together (neutral: ptxas had
  hoisted them); the decode moved to GB10's `cvt.rn.f16x2.e2m1x2` (one
  F2FP) plus one exact f16x2 multiply by the e4m3 scale (every product
  <= 6 significant bits in [2^-10, 2688], all f16 normals — the f32 is the
  value the register decode produced), which needs the architecture-
  specific target: CMAKE_CUDA_ARCHITECTURES 121 -> 121a (the generic 121
  takes cuda_fp4.h's software path, still bitwise). The gate/up kernel
  111 -> 107 us, the slot path 204 -> 195 us per layer at one row; fabric
  (`fp4core_1853`): T=1 26.0 -> 26.0 (integer-ms log), MTP 34.36 -> 34.21
  ms/step, 20.04 -> 19.96 ms/token, transcript identical.
  Then the floors, which close the fp4 core item: a probe kernel with
  the gate/up kernel's exact grid and loads and no arithmetic streams at
  238 GB/s (a plain 1 GiB read: 234); the kernel reads 23.1 MB (the
  shared expert's fp8 rows included) in 107 us = 216 GB/s, 93 % of its
  floor; the down 11.5 MB in 54 us = 90 %. A persistent grid-stride form
  of both kernels (a block walks (slot, row block) tiles holding the next
  tile's loads in registers while decoding the current one, activations
  double-buffered in smem, one barrier per tile; 128 registers at two
  blocks per SM) is bitwise and NOT faster: gate/up 104.3 vs 104.8 us,
  down 54.0 vs 55.8 at one row, 186 vs 184 and 91 vs 94 at two — parked,
  patch kept in the session scratchpad. Pinning the block gate/up kernel
  to four blocks per SM (80 -> 64 registers) spilled and ran 5 % slower.
  The router's 17 us is its serial fused select, not its loads (the dot
  loop's unroll 4/8/16 and 2 or 8 warps per block: all 17 us).
  Programmatic dependent launch inside stream-captured graphs, probed on
  a chain of three 3-wave memory-bound kernels: 259 us plain, 256 with
  the wait at the top, 257 prefetching weights before the wait — the
  graph's interfaces cost nothing measurable when DRAM is the bottleneck;
  parked with the serve-path interface above. The L2 prefetch windows were
  swept earlier (8/12/16/24 MB alike). What is left on the decode path
  without a quantization decision: ~0.5 ms in the slot kernels' last 7-10
  % (no cheap form found), ~0.2 ms in the router select, ~0.3 ms if the
  mHC update ran once per site behind a grid barrier (cooperative launch
  in the graph, untested), ~0.3 ms of local copy+fold in the collectives;
  the KDA bf16 GEMVs (10.4 ms at the DRAM ceiling) remain the one large
  lever and it is a quantization decision.
- The crumbs, pursued (2026-09-08 evening, "shaving another ~1 ms seems
  meaningful"): the mHC site, the router select, the collective's local
  phases.
  mHC: `mhc_site_bench` (new) split the site — at one row the dots alone
  are 2.7 us, the fused finish tail 8.6, of which the 20-iteration
  Sinkhorn is 4.6 (39 dependent butterfly passes on one warp) and the
  collapse/norm loads the rest; the finish sat on the critical path to the
  sublayer though only the stream UPDATE at the site's end reads comb.
  Landed, bitwise (glm_mhc_test pins the deferred form against the
  in-block one at 1/2/8 rows, the tiled prefill form still pinned): the
  finish leaves comb to `launch_mhc_comb` (one warp per token, the same
  lanes and arithmetic) on a low-priority side stream forked after the
  finish and joined before the update — under the sublayer, as the
  prefetcher's windows are; and the fused finish reuses the dots block's
  register copy of the streams for the collapse (the flattened vectors are
  exactly the collapse's runs at hidden % 2048 == 0) and issues the norm
  weights at its top. The fused kernel 10.7 -> 4.7 us in the bench, 13.0
  -> 6.5 us in the MTP graph (nsys node trace), the comb 7.5 us on the
  side and never waited on (update start - comb end: median 317 us, min
  185). Per-step main-chain mHC kernel time 1.48 -> 0.89 ms; graph replay
  period under nsys 34.71 -> 34.31 ms; the app's own readings
  (`mhc_final_2041` vs `fp4core_1853`): MTP 34.21 -> 33.89 ms/step
  (19.96 -> 19.77 ms/token), T=1 26.46 -> 26.22 by the 300-step totals
  (the per-step log is integer-ms); transcripts identical.
  Router select in registers (each lane's nine biased scores, the winner
  lane retiring its entry, no smem and no __syncwarp in the loop; bitwise
  by the argmax's exactness, the oracle/tie tests green): stream-launched
  bench -1.4 us at one row and -1.1 at two, but in the MTP graph the
  launch read 10.7 vs 9.8 us (0.46 vs 0.42 ms/step) — a first form with
  the activation loads outside the batch had read +3.6 us at two rows.
  Reverted: no gain in the production configuration. The dot loop's
  batching (2/4/8/16 vectors in flight) and 2 vs 8 warps per block made
  no difference either way; the router is its serial fused select (~5 us)
  on top of an 11.9 us dot at 200 GB/s.
  The collective's local phases (copy 3.2 + fold 3.5 us of the 33): the
  snapshot copy, the claim's hash+stage pass and the fold are already
  16-byte vectors on one block; the remaining time is host-memory store
  latency, system fences and the last peer's staging pass — nothing
  cheap; the handshake is host-posted RDMA, a transport redesign. Parked.
  Programmatic dependent launch across the graph's interfaces: measured above
  (nothing). The decode budget's remaining shape at MTP (nsys, per step):
  GEMVs 14.6 ms (KDA bf16 at the ceiling + the FP8 projections), MoE slot
  kernels 10.9, collectives 4.6, mHC 0.9 (+0.7 on the side), small kernels
  ~2.3, in a 34.3 ms period — ~1 ms of gaps and host. What is left
  without quantization is inside the noise of a fabric reading.

## 7. Original work estimates (2026-09-08)

Estimates are engineering days for one person who knows the code, gates and
fabric evidence included.

| # | phase | days | exit |
|---|---|---:|---|
| 0 | the checkpoint (done) | 0.5 | four identical revisions, manifests agree |
| 1 | format plumbing: profile, table, `GlmFp4Matrix`, loader slices, TP views, fixtures, gates 1/5/6/7-shard-parity | 3–4 | the hybrid binds and loads resident on one rank; fixture gates green |
| 2 | decode kernels: fp4 core, slot kernels, host twin, `fp4_gemv_test`, `glm_moe_test` fp4, `moe_slot_bench` | 4–6 | effective bandwidth target met; twins bitwise; oracle within budget |
| 3 | prefill kernel: grouped fp4 MMA, gates 3/4 | 3–4 | bitwise the tile reference; nsys at the read floor |
| 4 | integration: `moe_layer` dispatch, prefetch, memory plan, graph tables, fabric bring-up, gates 8/9, docs | 3–4 | transcripts judged, perplexity recorded, FP8 timings unchanged |
| 5 | to the floor: prefetch retune, down-kernel shape, batch family, prefill tiling for short segments | 3–5 | the §6 targets, or the measured reason they are not reachable |
| | total | 17–23 | |

Follow-ons, each a separate decision with its own gate:

- W4A4 prefill experiment: an activation-quant prologue plus the native
  `kind::mxf4nvf4` MMA (verified to assemble and run on `sm_121a`), behind a
  knob, judged by `fabric_logprob.py`. Only pays in a compute-bound regime
  (much larger prefill chunks or high-occupancy batch). 2–3 d.
- KDA projections to FP8 with block scales: the largest remaining byte term
  (2.34 GB/token, ~4.9 ms/step). The KDA tensors are native BF16 in every
  release, so this is a genuine new rounding; Z.ai left them alone. 5+ d with
  the quality gate.
- DSA projections consumed as FP8 natively instead of through the BF16 bridge
  (−0.35 GB/token, ~1.4 ms/step); independent of NVFP4, applies to the FP8
  model too. 2–3 d.
- Expert-source comparison under identical non-expert bytes: dabsLabs vs
  coolbho3k vs RedHat experts through the composition tool, one perplexity
  run each. 0.5 d each.

## 8. Risks

- The fp4 core's effective bandwidth: more ALU per byte than fp8 and 2-byte
  scale loads; mitigated by measuring in phase 2 before any integration, and
  by the two layout options for the scales.
- The down projection's 256-byte rows: needs the multi-row shape from the
  start.
- `sm_121a`: arch-specific SASS runs on GB10 only (fine for this project) and
  every existing test must pass on the new arch flag before any fp4 code lands.
- Register pressure in the MMA tile decode (the round-9 record shows this
  kernel is occupancy-sensitive); keep the decode per thread small and measure.
- Two weight formats in one launch (routed fp4, shared fp8): branch per block,
  uniform; the twins gate it.
- The experts' quality: 4-bit weights, ~8 % relative error per tensor; the
  perplexity number against the FP8 build is the first fabric measurement,
  before any tuning. MTP acceptance may move with it; measured per prompt class
  as in the sign-off report.

## 9. Decisions and the evidence behind them

### 9.1 In-register decode, bf16 activations

The user asked why dequantize at all, then whether that leaves any gain over
FP8, then whether any kernel could keep the weights in their own type, then
about accumulated rounding. The answers, with the measurements:

- Bytes are the cost on both paths. Decode is a GEMV at 1–8 rows; the 2,048-
  token prefill's expert GEMM waits on DRAM ~3× longer than it computes. The
  FP8 core, which decodes e4m3 in registers per weight, already runs at line
  rate; halving the bytes halves the time whatever the ALU does.
- The only instruction that consumes the checkpoint's bytes as they are is the
  block-scaled MMA, and it requires e2m1 activations (W4A4): a lossy step per
  layer per token the checkpoint was never validated for; `kind::f8f6f4`
  cannot apply per-16 scales; integer dot products would need int8
  activations. The in-register path is therefore also the most accurate one
  available (§3.1: exact upcasts, one division per output).

### 9.2 The FP8 release's bytes for the non-expert classes

Measured 2026-09-08: `zai-org/GLM-5.3-Flash` is the FP8 release (62 shards,
331 GB; the unsloth repo mirrors it) and holds shared experts, dense MLPs and
DSA projections in FP8. In `zai-org/GLM-5.3-Flash-BF16` (whose bytes the
NVFP4 checkpoints carry — verified by range-read) those same classes sit ON
the release's e4m3 grid for 87–93 % of elements, with 480–740 distinct values
per 128×128 block (a pure upcast: 100 %, ≤ 256; native BF16: ~6 %, ~1,950 —
which is what kv_b, eh_proj and the MTP layer show, and KDA at 11–13 % /
~800). The off-grid tenth lies within half an e4m3 step. Re-quantizing those
BF16 values with the release's recipe reproduces the release's codes for 100 %
of tested elements. So the BF16 repo adds a half-step refinement on one
element in ten for those classes, at twice the bytes; the user chose the
release's bytes (and no re-quantization anywhere), keeping the BF16-source
option as a tool flag. The MTP layer and KDA are native BF16; KDA is untouched.

### 9.3 Why compose rather than download

A survey of 28 NVFP4/MXFP4 repos on the Hub (2026-09-08) found none with
NVFP4 experts + the FP8 release's bytes + the MTP layer.
`coolbho3k/GLM-5.3-Flash-NVFP4-Optimized` has the main-stack geometry exactly
(RedHat's experts re-scaled; Z.ai's FP8 restored — verified byte-identical)
but no layer 45; `RedHatAI/GLM-5.3-Flash-NVFP4` upcasts the non-experts to
BF16 and re-quantizes the MTP experts; the rest carry BF16 non-experts,
re-quantized attention, or MXFP4 experts. Both sources being on every node,
composing cost no network and gave full provenance and the best-measured
expert quant (dabsLabs: from BF16, per-block 4/6 scale selection, 0.0828).

### 9.4 Hardware facts established

`cvt.rn.f16x2.e2m1x2` (source operand a `.b8` register) and
`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X`
assemble and execute on the GB10 with `-gencode arch=compute_121a,code=sm_121a`
and are refused for plain `sm_121` (the project's current arch).
