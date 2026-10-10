# Changelog

The history by milestone. The dated engineering record in
`benchmarks/results/` has every measurement and fix behind these lines;
`PLAN.md` has the milestones' exit gates.

## Unreleased
- **Parallel eager layer load** (2026-10-07):
  `ResidentLayerStream::load_layers_parallel` (new): each worker owns a
  pinned staging mirror and a CUDA stream, so checkpoint slicing, host
  packs and H2D uploads overlap across layers; the disk still serializes
  at its line rate. `ResidentImage` takes a mutex (entry snapshot under
  lock, blobs move outside it; concurrent captures reserve disjoint
  ranges). Opt-in via `DGPP_PRELOAD_THREADS` (unset keeps the serial
  loop), wired in serve for Qwen/MiMo/GLM-4/GLM-DSA/DSV4/DSV41/Qwen35 and
  in the DSV41/DSV4 eager constructors; `qwen_load_check --threads N`
  benchmarks it. Measured Qwen3.8-Flash-Next/Spark: cold build ~76s ->
  ~47s at 4 workers (boot ~96s -> ~56s); restores stay ~20s either way
  (disk-bound). Serial vs parallel captures are bitwise identical
  (per-layer bytes+fold), and the grant-tail `memset` makes all captures
  deterministic. Loader suites green (qwen 9, glm 13, mimo 5, dsv41 6,
  dsv4 4, glm_dsa 4, glm4 4).

- **Full GLM-5.3: `engine.attention_weights`** (2026-10-09): `int4`
  re-encodes the checkpoint's int8 g64 q_a/kv_a, q_b and o_proj rows to
  int4 g64 at load with the RTN recipe (kv_b keeps its int8 form; the
  resident image is keyed by the form through the family's
  `loader_format()`, so the first boot under it writes its own image) —
  a measuring instrument for the attention-bytes lever, NOT lossless: on
  the four nodes it reads a 61.1 ms pass against 65.8–66.6 (o_proj's GEMV
  7.5 → 3.8 ms; q_a/kv_a and q_b barely move, they are L2-fed by the
  boundary prefetcher), −1.74 GiB resident, and costs +0.030 nat on the
  hard text (perplexity 10.25 → 10.57, top-1 50.3 → 49.9 %) — the RTN floor
  a calibrated 4- or 6-bit attention has to beat; o_proj is the exposed read. `checkpoint` (the default) serves the
  weights as shipped. The diagnostic forward check takes
  `--attention-weights` for the teacher-forced gate.
- **Full GLM-5.3: the draft head in block FP8** (2026-10-09, `engine.mtp_head`):
  `fp8` encodes the lm head's rows to block FP8 at load (e4m3, 128 × 128
  fp32 scales — the FP8 releases' recipe, `loaders/fp8_quant.hpp`) and
  the draft block reads that copy through the fp8 scale-GEMM core while
  the verifier keeps the main head in its 12-bit form — the served
  distribution is unchanged by construction (greedy transcripts identical
  across the two forms on the fabric), only the draft's proposals come from
  the narrower head. +0.22 GiB per rank at world 4 (the memory plan's
  weights line). On the four nodes (nsys per-step breakdown, one request, MTP depth 1):
  the draft head's GEMV 1.36 ms in the 12-bit form, 1.05 ms in FP8 (238 MB
  at 227 GB/s) — 0.3 ms per pass, 0.5 %, under the load gate's run-to-run
  noise; acceptance by class identical to the hundredth of a token per
  pass. Opt-in, not in the template: 0.22 GiB buys ~4K tokens of fp8
  context there. `checkpoint` (the
  default) keeps the draft on the main head's form.
- **Full GLM-5.3 in the Mixed346 GPTQ H32 A8 g128 checkpoint** (2026-10-09/10,
  `HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128`,
  [card](docs/model_cards/GLM-5.3-Mixed346-GPTQ-H32-A8-g128.md)): the
  Int4/Int8 checkpoint with its routed experts re-encoded expert by expert
  at 3, 4 or 6 bits (gate/up at one width, down at its own; a Gaussian
  eight-level codebook at 3 bits, the NF4I8 codebook at 4, signed INT6 at
  6; bf16 scales per 128, GPTQ in the H32-rotated basis) with INT8 expert
  activation codes (one fp32 scale per 128 rotated values), 116 experts
  kept in the Int4/Int8 form — 4.22 GiB less per rank than Int4/Int8 and
  19 % less divergence from BF16 by the publisher's table. A fourth packed
  scale format (`kPackedScaleBf16G128Mixed346`: dense 3/4/6-bit index
  streams, `packed_bits_allowed`) parsed from `quant_method: dgpp_mixed346`
  with its two sidecars (the baseline compressed-tensors block and the
  per-expert recipe, or both inline), every tensor named by its own form in
  the binding; the loader slices the dense streams at the 128-group; a
  new GEMV core (`kernels/packq_a8_gemv.cuh`: a 16-byte piece a lane at 4
  bits, a 48-byte piece — a group, or half of one — at 3 and 6 bits loaded
  cooperatively and swapped through a per-warp shared tile, dp4a int32
  group dots, one fp32 scale product per group, the staged codes at a
  conflict-free 132-byte stride) under mixed-form slot and grouped kernels;
  the H32-then-int8 activation quantizer (`kernels/hadamard32.cu`), fused
  into the post-attention norm for the token rows
  (`glm_rmsnorm_bf16_quant_int8`); an int8 tensor-core prefill tile
  (`packq_gemm.cu` `wide_tile_i8`: the rows' codes against int8 levels on
  mma m16n8k32 s8, each 128-code group an exact int32 dot scaled once; the
  fallback experts' tiles stay on the bf16 group-64 chain in the same
  launch) with the rows' and the widths' lines prefetched into L2, and the
  tile list for every Mixed346 layer. Host and Python oracles, the fixture
  gates (TP, engine graph, decode, parity) with budgets stated for the int8
  contract's doubled sensitivity to upstream noise, the slot microbench
  (`--m346 FORM|census`) and its `--prefill` chain mode: the slot path at
  the int4 core's bandwidth at every width, the gate tile 15 % under the
  int4 tile's time, the 2K-token chain level with it. On the four nodes
  (2026-10-10): decode 62.7 ms per pass at one request and 236 at eight
  (Int4/Int8 63.6 / 252, NF4I8 64.0 / 253, the same MTP acceptance), cold
  prefill 3.906 / 17.943 / 76.714 s at 2K / 8K / 32K (Int4/Int8 3.896 /
  17.866 / 76.439), 108 ms per pass at 256K context; teacher-forced 2.3284
  nat/token between the two references.
- **Full GLM-5.3 in the NF4I8 GPTQ H32 g128 checkpoint** (2026-10-08/09,
  `HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128`, [card](docs/model_cards/GLM-5.3-NF4I8-GPTQ-H32-g128.md)):
  the Int4/Int8 checkpoint with its routed experts re-encoded as four-bit
  indices into a sixteen-level INT8 codebook (bf16 scale per 128, GPTQ) in
  a block-32 Hadamard-rotated input basis — 2.64 GiB less per rank and 22 %
  less divergence from BF16 by the publisher's teacher-forced table. A
  third packed scale format (`kPackedScaleBf16G128Nf4i8`), parsed from the
  checkpoint's `quant_method: dgpp_nf4i8` block and refused on any drift
  of the codebook, group, rotation or layer range; the loader slices at
  the format's group; the GEMV core decodes the codebook with two
  byte-permute lookups into its exact integer × bf16-scale chain, the
  tensor-core prefill decodes the levels to their bf16 patterns the same
  way (no integer-to-float conversions on the decode path); the H32
  rotation is one kernel (`kernels/hadamard32.cu`: fp32 butterflies, one
  bf16 rounding, the op order of the host oracle and the Python
  reference) applied once per row — into a side buffer the decode slots
  read and in place on the gathered routed rows and the SwiGLU rows of
  the prefill chains. A first form rotated the staged row inside every
  slot-kernel block (64 blocks a slot repeating the same rotation) and
  converted the levels through int-to-float on the tensor-core path; on
  the fabric the sixteen-row step read +4 % against the Int4/Int8
  checkpoint, now level: on today's binary the two checkpoints measure the
  same step at one and at eight requests (C8 ratio 1.000) and the same
  cold prefill (3.87 vs 3.90 s at 2K; the format saves 1.1 % of a decode
  step's bytes, 8 of 256 experts a token, and the lookup decode changed
  no prefill cell). Oracles:
  the MoE host reference and both Python references carry the codebook
  and the rotation; tests at every level (kernel, layer, synthetic
  checkpoint forward/decode/TP/engine, the landed checkpoint's binding).
  Template `deploy/cluster_glm-5.3_nf4i8_w4.example.json`: eight slots,
  MTP depth 1, a 272K-token fp8 K/V pool (plan 110.43 GiB; the
  280K shape boots only on a settled node — the process holds ~3.5 GiB
  beyond the plan at listening). On the four
  nodes: the Int4/Int8 checkpoint's decode step at one request and at
  every context bucket to 256K, C1 27.1–30.6 engine tokens/s and C8 52.6–56.4 wall
  tokens/s over the five classes (the Int4/Int8 rows 27.4–30.5 / 52.9–55.7),
  cold prefill 3.87 / 17.9 / 76.5 s at 2K / 8K / 32K; teacher-forced on the
  hard text 2.3276 vs 2.3295 nat/token (inside the standard error);
  HumanEval 157/164, GSM8K 294/300, schema extraction 100/100
  at `reasoning_effort` low.
- **The L2 weight prefetcher's knobs are engine keys** (2026-10-05):
  `engine.l2_prefetch` (on/off), `engine.l2_prefetch_form` (`load` the
  bytes, `lines` one L2 prefetch per 128-byte line, `touch` one line per
  64 KB), `engine.l2_prefetch_window_mib`,
  `engine.l2_prefetch_boundary_window_mib`, `engine.l2_prefetch_boundary_rate`
  and `engine.l2_prefetch_layer_rate` (`off` / `light` / `full`), and
  `engine.l2_prefetch_merge` — the `DGPP_L2_PREFETCH*` environment
  variables they replace are gone from the prefetcher, the launcher's node
  keys and the config's `node_env` list; every rank takes the same settings
  through the worker record. Measured while at it on one node, plain T=1:
  the prefetcher makes no difference at one row to either the GEMV-chunk
  world (112.7 ms a step on and off) or the streaming-form world (121.1 /
  121.2), so the streaming form's 8 % at one row is the form itself, not an
  L2-residency effect, and the forms read the same at one row (off 121.4,
  load 121.1, lines 121.1, touch 121.4 ms a step), so it is not the page
  walks either. The one-node plain step's breakdown (`raw/trace16`): 91 %
  is the width-8 streaming form at one row, 368 launches at 219 GB/s while
  the single head launch runs at 231 — wave quantization of 64-row blocks
  (1.7–2.8 waves a launch) against the GEMV core's 8-row blocks. A finer
  streaming unit at one to four rows is the kernel item; the drafter
  templates (8-row steps) are unaffected.
- **Qwen3.8-27B: a prompt's prefill the same alone and in a group; the
  four-node chat cell explained; the drafter format confirmed** (2026-10-05,
  the evening round): the campaign's DIFFERENT reading for prompts started
  together was the prefill GEMM lowering changing with the walk's row count
  — the tiny-fixture test `qwen35_prefill_group_invariance` showed a group
  of two prompts (74 rows, the streaming form) bitwise the solo prefill and
  a group of four (148 rows, the dequant bridge on cuBLASLt) not, with
  cuBLASLt itself picking different kernels at 148 and 296 rows. Three
  changes: every prefill-shaped bf16 cuBLASLt product takes one pinned
  algorithm per shape (the heuristic's choice for the widest walk,
  `CublasLtGemm::set_pinned_rows`; groups of four and eight are now bitwise
  each other; 2K / 8K prefill 1.11 / 0.87 ms a token, unchanged); the GDN
  a / b GEMV runs in row groups at every row count, prefill included (0.5 MB
  matrices; no 64-row boundary left for them); and `engine.prefill_group`
  (default true) — `false` prefills one cold prompt per walk, so a prompt
  shorter than the 128-row lowering bound started together with others
  reads exactly as alone, at a burst's prefill throughput: a burst of eight short prompts on one node 3.2 s wall against 2.2 s grouped.
  The streaming form below 128 rows and the bridge above are different
  chains by construction (scattered against consecutive 16-k mma sets), so
  that boundary stays; the key is the exact mode for short-prompt bursts.
  The four-node chat cell (48.8 tok/s against the campaign's 53.4): not the
  schedule (chat first, chat alone, a fixed lambda and whole blocks all read
  2.38 tokens a step at the same step time) and not the fp8 drafter (the
  checkpoint drafter reads 2.38 too, 1.5 ms a step slower) — the chat
  prompt's four-node continuation changed at char 330 when short prompts'
  a / b prefill moved from cuBLASLt to the GEMV core (the one-node text is
  unchanged), and the new continuation drafts worse for this one prompt;
  the earlier 56.5 counted prefix-cache repeats under the stale-snapshot
  bug (2.8 a step against the cold request's 2.36). The multi-prompt
  measure, MT-Bench greedy accepted tokens a pass: one node 3.87 tokens a pass over 80 prompts at 256 tokens (3.53 in the morning's run at its length), four nodes 3.96 over 68 — level or better. The fp8
  drafter stays the recipe at every world (one node 140.7 against the
  checkpoint's 145.1 ms a step at identical acceptance; four nodes 48.5
  against 50.0). C1 after the round: one node 20.1 / 33.3 / 45.9 / 37.9 / 19.7 tok/s at 140.6 ms a pass, four nodes 59.0 / 92.3 / 125.2 / 97.7 / 48.8 at 47.5 — the math and chat cells are single prompts whose continuations moved with the prefill chain (one-node math up, four-node math and chat down), which is why the multi-prompt measure decides. Qwen3.8-Flash-Next:
  its decode step is one chain across row counts under the shipped
  FP8-dense recipe — the QSA projections through the dense lowering, the
  fp8 head and the dense sites on the streaming form from one row (the MoE
  layer's eight-row fused tails and the GR site's fused rows were already
  bitwise their unfused chains): the new `qwen_decode_rows_invariance`
  holds every solo row count and the two-by-eight and four-by-four batched
  verifies bitwise the eight-row solo verify's rows; under `engine.fp8_head:
  "mma"` the head takes the streaming form at every width (the compact
  prefill row and prompts above the decode capacity included — the GEMV
  chunks below five rows and above the capacity were two more chains), and
  the `qwen_head_check` gate counts that kernel at every captured shape and
  names every captured kernel when the count is off. Its one-node world
  reads 40.9 / 47.6 tok/s (prose / code) at 39.4 ms a pass in the recipe
  mode and 31.5 at 31.3 ms plain (44.5 / 48.7 at 38.5 and 32.3 at 30.5
  before; the plain step pays the one-row streaming form's 3 %, the prose
  cell is one prompt whose continuation moved). The one-row
  streaming form's rate, cold (`mma_gemv_test`, no split-K): within 2 % of
  the GEMV chunks at [5120 x 5120] (122.9 against 120.2 us, 213 GB/s) and
  6 % behind at [32320 x 5120]; 2.6x behind at a narrow [576 x 6144] site
  (28.8 against 10.9 us, nine blocks on 48 SMs) — split-K fills such a
  site in the model, and the four-node shard widths are where the plain
  T=1 world pays its 10 %. The next kernel item: the decode form at one to
  four rows on narrow shards.
- **Serve: completion response timings** (2026-10-05; compatibility follow-up
  2026-10-08): chat and legacy responses carry a top-level llama.cpp-style
  `timings` object. Streams report timings on the final usage or terminal
  chunk, including when usage is disabled. Prompt rates count uncached tokens
  and exclude the initial admission queue; decode rates exclude the first
  token supplied by each prefill. Existing usage, JSON metrics, Prometheus
  measurements and throughput logs keep their definitions. See
  [the API contract](docs/openai-compatibility.md#completion-timings-dgpp-extension)
  and [validation record](benchmarks/results/2026-10-08-completion-timings.md).
- **Qwen3.8-27B: a request's logits are the same alone, at every verify
  depth and in a batch** (2026-10-05): six row-count dependences in the
  decode step, found after the campaign with the solo-vs-co-tenant probe
  (a request joined by others read DIFFERENT on one and four Sparks) and
  the scheduled drafter's transcripts (a solo request's text moved with
  the depth the schedule chose). (1) The decode head took the fp8 GEMV
  chain below five rows and the streaming MMA from five, so a solo plain
  request (one row) diverged from its eight-slot batch: the head and every
  fp8 projection (gate / up / down, attention q|k|v|o, the GDN out) take
  the weights-once streaming form at every decode row count
  (`mma_from_rows` 1 — its row chain is bitwise the same 1..64,
  `scale_gemm_test`). (2) The GDN in-projections a / b went to cuBLASLt's
  tile kernel above eight rows: `launch_bf16_gemv_multi_rows` (row groups
  of four by blockIdx.y plus a tail launch) is bitwise the chunked launches
  at every width to 64. (3) The GDN in-projections qkv / z and the
  attention q|k|v took the multi-problem fp8 GEMV core up to
  `dense_gemv_rows` rows — which defaults to 4, not the 16 a comment
  claimed — and the MMA above, so a scheduled 2- or 4-row verify computed
  a different chain from the 6- / 8-row steps and the batches (one bf16
  ulp in one column of the first GDN layer's in-projection, then the
  text): both take the dense lowering (the streaming form) at every decode
  width; a / b stay on the bf16 GEMV core. Found by the new tiny-fixture
  test `qwen35_decode_rows_invariance` (a 1..7-row verify's rows are
  bitwise the eight-row verify's, with the fp8 head on a resident stack)
  after transcript legs could not localize it. (4) Split-K's form boundary
  — launches of <= 32 rows split the k range, 33–64 ran the wide form
  unsplit, so an eight-slot four-node batch's rows differed from solo: the
  split ranges are 256-k units (a window boundary of every tile form) and
  the 33–64-row form splits with the decode width's count
  (`kMmaGemvSplitRows` 64; `mma_gemv_test`), which also fills the four-node
  grid: eight slots 205.3 / 299.2 / 356.7 / 310.7 / 218.9 tok/s at 91.5 ms a step against 183.7 /
  273.3 / 326.3 / 284.4 / 195.6 at 106. (5) The multi-problem fp8 GEMV at
  <= 8 rows is one chain 1..8 (tested, its bf16 problems against the
  row-group launch too). (6) The recorded scalar commit was captured with
  the model's decode rows (8) for every variant; the commit kernel reads
  `accepted < rows` as a retraction and restores the conv state from the
  snapshot of row accepted − 1, which the conv kernel never writes for a
  step's last row, so a reduced-depth step that accepted all its rows
  restored a stale snapshot: the commit takes the step's rows.
  Verified on the fabric (`raw/verify10`): a request joined by co-tenants reads the same text as alone on one, two and four Sparks from the first request after boot; the scheduled drafter's transcripts match whole blocks 4/4 at W1 and W4 at every depth mix tried (the template, min-depth 3, a forced-shallow schedule: 3 of 4, 3 of 4 and 0 of 4 before); the tiny-fixture rows test and the ctest subset (75) pass. Costs: W1 / W2 / W4 drafter C1 level (20.1 / 32.8 / 45.5 / 35.1 / 19.6 tok/s at 142.0 ms; 32.6 / 57.3 / 80.0 / 66.3 / 34.9 at 78.9; 60.0 / 92.7 / 125.1 / 105.3 / 48.8 at 47.5), C8 level (W1 195.2 ms a step); the plain T=1 world pays the one-row streaming form on every fp8 site — W1 8.2 tok/s at 121.2 ms against the campaign's 8.9 at 112.7, W4 28.0 at 35.3 against 31.0 at 32 — the one-row form's rate (about 215 against the GEMV core's 240 GB/s, worse at the four-node shard widths) is the next kernel item. Prompts started together
  still differ (the group prefill, open). Also: `--no-dflash` drops a
  drafter template's `mtp_schedule` (plain decode from the drafter recipes
  booted refused); the schedule's min-depth is checked against the
  drafter's block, not `mtp_depth`; `engine.dflash_depth` is the eager
  engine's cap (the graph worlds verify the whole block or the scheduled
  depth).
- **Qwen3.8-27B drafter: fp8 block matrices, the chunked top-K, the
  schedule's gate** (2026-10-05): the drafter's five block matrices (q|k|v,
  o, gate, up, down) serve as block-128 e4m3 under `engine.dflash_weights:
  fp8` (the loader's own encoder at load; of the five only the k|v rows
  keep a bf16 copy, for the context features' GEMM; `checkpoint` keeps the
  bf16 packed 12-bit). Lossy for the proposals only — the verify
  is exact, so transcripts and sampling distributions are unchanged and
  only the acceptance can move: one Spark, same binary, MT-Bench
  nothink-greedy 3.53 against 3.52 tokens a pass, greedy C1 20.1 / 32.3 /
  44.3 / 35.2 / 19.4 against 19.9 / 31.6 / 44.4 / 34.5 / 18.8 tok/s at 142.2
  against 145.5 ms a pass, transcripts identical 4/4; every drafter recipe
  ships `fp8`. The drafter's top-K over the mask rows' head logits ran one
  block a row (thread 0 walking 4,096 candidates): 0.63 ms a step on one
  Spark for 7 MB; now kTopkChunkSpan-wide chunk blocks (32 a row at most)
  fold shared-memory lists and one block a row folds the chunks' partials
  (`dflash2_topk_ws_bytes`): the same set in the same order (ties to the
  lower id), 129 us for eight rows in the kernel test. The scheduled
  verify depth's configuration refused the block drafter ("needs MTP"):
  it takes `spec_enabled()` now, the block's confidence from the selector
  walk's table — and the drafter recipes ship the schedule (2.0 ms a row
  over a 130 / 66 / 34 ms pass on one / two / four Sparks, the traces'
  cost model): four Sparks at eight slots 183.7 / 273.3 / 326.3 / 284.4 /
  195.6 tok/s at 106 ms a step against the campaign's whole blocks 153.5 /
  270.1 / 319.6 / 260.2 / 169.7 at 117, C1 level, transcripts identical;
  sampled slots hold the whole block.
- **Qwen3.8-27B drafter: sampled proposals, the one-node graph template,
  the first-miss histogram** (2026-10-05): a sampled request's block is now
  DRAWN — the recorded selector walk samples each draft from the selector's
  softmax at the request's temperature and writes the set it drew from as
  the draft's proposal (`dflash2_selector_walk(..., pos, spec, proposal)`,
  `Qwen35Model::dflash2_arm_proposals`, the engine's `arm_block_proposals`),
  so the verify's ratio rule applies (min(1, P/Q), rate 1 − TV(P, Q))
  instead of the point-mass P(argmax) rule the argmax walk gets; the
  reference speculator's form. `engine.mtp_draft` rules the drafter too
  (`sampled` / `auto`, `greedy` = the argmax walk); greedy requests are
  unchanged. MT-Bench turn-1 prompts (thinking on, T 1.0 / top-p 0.95 /
  top-k 20, one Spark) read 2.80 tokens per pass with the argmax walk and
  3.27–3.34 drawn; greedy on the same prose-heavy categories 3.30
  (thinking on) / 3.12 (off). `timed_load` C1 at temperature 1 on one Spark
  (prose / code / json / math / chat): argmax 15.5 / 28.1 / 38.6 / 33.3 /
  15.2 → drawn 16.9 / 29.3 / 37.2 / 31.3 / 17.1 tok/s; four Sparks drawn
  49.0 / 91.0 / 115.2 / 91.1 / 50.9 (52.2 / 84.4 / 117.0 / 98.2 / 47.6 as
  the recipe ships, at 0.7). The draft's temperature is a key
  (`engine.mtp_draft_temperature`, the fraction of the request's; the
  `DGPP_SPEC_PROPOSAL_TEMP` diagnostic retires): at 0.7 the same one-Spark
  load reads 17.5 / 29.8 / 40.0 / 34.1 / 15.2 and MT-Bench 3.41 — a sharper
  draft keeps the argmax's rate where the target is sharp — so the drafter
  recipes ship 0.7 (the MTP templates stay at the request's). The ratio
  rule now runs in the pure temperature regime too (a request that sends
  only `temperature`: most clients, the arena harness), where the verify
  priced drawn drafts by the deterministic P(draft) rule (acceptance Σ Q·P
  against the overlap Σ min(P, Q)); the residual's total there is
  1 − Σ min(P, Q), exact when the prefix holds every proposed id, else the
  row keeps the deterministic rule (`sample::spec_accept_from_prefix`, the device's
  `spec_decide_prefix`; host and device bitwise in both regimes,
  `sampler_test` / `glm_pick_test`). Measured neutral: MT-Bench
  think-sampled in the pure regime 3.16 against the deterministic rule's
  3.20 tokens per pass, one-Spark `timed_load` at temperature 1 within
  noise — with the 0.7 draft temperature the proposals are sharp enough
  that the two rules nearly coincide; kept for the uniform semantics, at
  no runtime cost (no fallbacks over the legs). Block verification (Sun et al. 2024)
  as `engine.mtp_verify: block`: the sampled chain decided jointly along
  the drafts' p_i = min(1, p_{i−1} P/Q) with each sub-block accepted by
  h_i = S_i / (S_i + 1 − p_i) and the residual (p_τ P − Q)+ drawn after
  the longest accepted one — exact, never fewer tokens in expectation than
  the token rule; the device decides the block when every row prices its
  draft (else the token rule), the host draws a pure-regime tail residual
  (`sample::block_verify_from_prefixes`, `block_residual_complete`;
  `sampler_test` reproduces the paper's 11/9 against 10/9 example).
  Measured on the same binary: four Sparks under the arena harness 68.4
  against the token rule's 65.2 tok/s (two 20-run pairs each, +5 %, about
  two standard errors), MT-Bench think-sampled 3.45 against 3.41 tokens a
  pass, one-Spark `timed_load` at temperature 1 within noise, no fallbacks
  over 5,270 sampled steps; the drafter recipes set `mtp_verify: block`,
  the engine's default stays `token`.
  The Qwen3.8-27B decode batch's row ceiling is 64 (`Qwen35Model::
  decode_rows_cap`, 32 before): eight drafter slots at eight verify rows
  batch in one replay — at 32 the widest family held four slots and eight
  live slots ran scalar replays (one node C8 17–39 tok/s against C4's
  44–96 on the 2026-10-05 dry run; the eager engine's 32-row verify graph
  read 46–102). The verify's per-row GDN snapshots scale with the ceiling
  (3.1 MB a layer a row at world 1). With the ceiling the 27B model's
  recurrent state takes the checkpoint-and-replay form the Flash-Next
  model has had since 2026-09-29 (`kernels/kda.hpp` KdaReplay: the live
  buffer is the checkpoint, each pass saves its rows as their inputs —
  ~21 KB a layer a row — and the next pass replays the accepted ones; the
  replayed state is bitwise the snapshot it stands in for): the 9.7 GB of
  per-row snapshots at 64 rows become 1 GB of replay rows, and the snapshot
  writes leave the step. The drafter's selection runs once over
  every stacked row — the head, the top-K and the hidden projection as one
  launch each — where it streamed the lm head once per slot (eight head
  streams a step at eight slots: ~38 ms on one Spark, ~10 on four). The
  batched verify-rows budget (`engine.dflash_batch_rows`): a batch family
  whose slots times the block exceed it verifies the first floor(budget /
  slots) − 1 drafts of every slot's block through the reduced-depth
  variants the scheduled verify depth uses (the drafter still proposes the
  whole block; exact). Measured at eight slots, greedy: four Sparks C8
  170.5 / 231.9 / 251.3 / 239.0 / 178.2 tok/s at 101 ms a step under a
  32-row budget against 132.2 / 234.7 / 283.2 / 243.3 / 139.3 at 142 ms
  for whole blocks (one Spark 69.9 / 100.8 / 105.1 / 100.4 / 71.1 at 252 ms
  against 64.3 / 114.2 / 129.1 / 116.9 / 68.6 at 305) — the budget lifts
  prose and chat and costs the sharp classes their deep block; the MTP
  depth-3 template's eight-slot rows (213 / 290 / 335 / 304 / 232 at 84 ms)
  stay ahead of either, the drafter's edge being single-stream. The replay
  form and the batched selection together, greedy at eight slots: one Spark
  C1 19.8 / 31.5 / 44.3 / 34.3 / 18.7 and C8 87.0 / 150.5 / 171.1 / 158.8 /
  92.1 tok/s at 202 ms a step (the snapshot form's 64.3 / 114.2 / 129.1 /
  116.9 / 68.6 at 305); four Sparks C1 57.6 / 88.4 / 123.1 / 102.0 / 53.7 and
  C8 154.1 / 268.4 / 321.6 / 278.5 / 163.6 at 123 ms (132.2 / 234.7 / 283.2
  / 243.3 / 139.3 at 142), within 4–8 % of the MTP depth-3 world's eight-slot
  code / JSON / math and 28–30 % behind it on prose and chat; greedy transcripts
  bitwise 4/4 (the replayed state is the snapshot's). With the
  drawn proposals the drafter leads MTP on sampled traffic too (four Sparks
  under the arena harness: +14 % over the depth-3 template's 2026-10-04
  reading), so every Qwen3.8-27B template is now the drafter — the two- and
  four-node templates switch from MTP depth 3, which stays as their mode
  (`--no-dflash --mtp --mtp-depth 3`) — and the benchmarks page carries
  the 2026-10-05 campaign on the switched templates: greedy C1 (prose /
  code / json / math / chat) 19.8 / 31.5 / 44.2 / 34.3 / 18.7 tok/s at 146 ms
  a pass on one Spark (MTP depth 2's mode 18.0 / 20.4 / 22.0 / 21.0 / 16.4
  at 130), 35.4 / 55.3 / 76.6 / 62.6 / 32.9 at 82 ms on two (MTP depth 3
  34.2 / 40.3 / 46.0 / 42.4 / 31.2 at 78), 57.4 / 88.4 / 122.1 / 101.5 / 53.4
  at 50 ms on four (57.7 / 68.7 / 78.1 / 72.3 / 50.5 at 45); eight slots
  89.8 / 146.6 / 176.8 / 142.4 / 95.8, 127.1 / 218.0 / 263.6 / 219.1 / 144.5
  and 153.5 / 270.1 / 319.6 / 260.2 / 169.7; HumanEval 156–157 / 164, GSM8K
  291–292 / 300 and the schema extraction 100 / 100 at every world. The
  solo-against-batched exact-text check still reads DIFFERENT on one and
  four nodes and IDENTICAL on two, as on 2026-10-04 (under investigation).
  The one-node template runs the drafter on the graph engine
  (`decode_graph: true`; the eager engine's transcripts, chat 71 against
  95 ms a token). The GDN in-projections of a 5–8-row verify take the
  bf16 GEMV in two 4-row chunks (cuBLASLt's 16x16 tile kernel cost 3.3 ms
  a step at 8 rows; 0.7 now), greedy C1 on one Spark 18.9 / 30.3 / 42.6 /
  33.0 / 17.9 tok/s (prose / code / json / math / chat). The serve stats
  line adds the first-miss rank histogram ("first miss was the drafter's
  2nd / 3rd / 4th-8th / 9th-16th / none"; `MtpAcceptance::miss_rank`, the
  walked candidate tables published with the drafts): under greedy the
  correction is the drafter's 2nd candidate for 20–22 % of misses and
  outside its top-16 for 17–32 % — the sizing of a tree verify, not taken
  (a +7-row sibling tier would keep ~20 % of misses, +4–5 %, for rows that
  cost 2–3 ms each on one Spark). The one-tile fp8 GEMV form stages the
  next window's activation rows through registers one window ahead
  (`mma_gemv.cu` `Pre` / `pre_load` / `pre_store`; the synchronous staging
  put an L2 round trip on every window once every warp staged): bitwise,
  the sixteen-row launches 5–17 % faster in isolation (one node gate|up
  [17408 x 5120] 209.5 → 227 GB/s, two nodes 186 → 218), eight rows level
  (the step 154 → 152 ms, greedy C1 19.0 / 30.4 / 42.9 / 33.3 / 18.0).
  Nsight Compute (root, `ERR_NVGPUCTRPERM` otherwise) puts the one- and
  eight-row launches at the same duration in isolation: 109 registers a
  thread, two blocks an SM, 2.83 waves; the step's eight-row GEMV rate
  (~210 GB/s against the plain step's 237) is the surrounding kernels'
  interaction, not the kernel's row count.
- **Prometheus exposition of the service's metrics**: `GET /metrics/prometheus`
  now renders every JSON counter as typed `dgpp_*` families labeled with the
  served model, plus histograms of time to first token (split by prefix-cache
  attach), queue, prefill, decode and end-to-end time, time per output token,
  inter-token gaps, decode step time and request sizes, retires by reason and
  a `dgpp_build_info` line. The scheduler's observer gains `on_admit` (timing
  only; the op stream does not record it). The route's three original
  `spec_decode_*` samples keep their exact form.
- **Per-rank metrics**: every rank reports `dgpp_rank_*` (step and prefill
  wall time, generated tokens, pool, completed collectives, snapshot age);
  rank 0 on `/metrics/prometheus`, each peer on an optional listener
  (deployment JSON `ports.metrics` / `--metrics-port`, off by default)
  on its node address.
- **Qwen3.8-27B: the review's follow-ups** (2026-10-03, #84 / #85 / #87):
  the full-attention layers' kernels rewritten (`kernels/full_attn`: a
  query-tiled prefill form — sixteen queries by one head a warp over
  32-token K/V tiles — and a split-KV decode walk with a combine pass; 30K
  decode 187 → 146 ms a step, the 2K prefill's attention 87.5 → 29.9 ms,
  the 30K prefill 40.9 → 29.2 s, transcripts unchanged); the family's
  forward fixture gate (`qwen35_forward_test` over a tiny synthetic block-FP8
  checkpoint from the binding table, `tools/qwen35_reference_dump.py`'s
  numpy reference of the GDN / full-attention / MLP stack and the draft
  block, the smoke → end-to-end → teacher-forced strict chain, the MiMo
  gate's budgets); and the exact prefill measured against its floor — the
  dequant bridge stays (its dequant runs at line rate and cuBLASLt at the
  bf16 pipe's rate; a side-stream dequant never overlaps nvjet's persistent
  grid, and a fused fp8-weight GEMM lands at parity because the decode's
  instructions share the mma pipe's issue slots), with `kernels/fp8w_gemm`
  (63–66 TF, both scale forms, `fp8w_gemm_test`) kept for the scale GEMM's
  26 TF tile route (#89) and `fp8_gemm_bench --27b` reporting every lever.
  One recipe per world size (2026-10-04): the single-node template is the
  DFlash2 drafter's (`cluster_qwen3.8-27b_fp8_w1`), its MTP depth-2 and plain
  worlds the knobs `--no-dflash --mtp --mtp-depth 2` and `--no-dflash`
  (`--no-dflash` lands on the decode graph: the eager engine's plain world
  without a drafter streams its layers); the two- and four-node templates
  are MTP depth 3 (depth 3 leads depth 2 by 6–18 % across the classes at
  world 2), the drafter their mode (below).
  The family's rows on the [benchmarks page](docs/benchmarks.md) come from
  the 2026-10-04 campaign (`benchmarks/results/2026-10-04-qwen3.8-27b`: the
  runner's performance, modes and quality stages over both templates, the
  MTP depth-2 world as the drafter template's mode); the model card is
  `docs/model_cards/Qwen3.8-27B-FP8.md`.
  The drafter serves sampled requests (2026-10-04): a sampled slot's block
  is verified row by row with the sampled MTP rule — each draft stands with
  its exact probability under the request's temperature, top-k / top-p and
  penalties, the residual sample ends the step — where it ran plain before
  (8.4 tok/s at C1 against the greedy block's 17–41); logprobs, a logit bias
  and a grammar still run plain.
  `engine.bf16_weights: bf12` now packs this family's bf16 decode matrices
  too (#88: the drafter's five layers and fc taps, the MTP fc, a bf16 lm
  head), the bf16 bytes staying resident for the stacked redrafts' mma form.
  The family serves on two and four Sparks (#86): the loader's TP geometry
  (GDN and attention heads, the MLP and the lm head sliced across the ranks,
  the kv heads paired at world 4) through `Qwen35Model`, two boundary folds
  a layer (the attention / GDN output and the MLP output, bf16 on the wire)
  in the main and draft walks; the loopback TP
  gate (`qwen35_tp_test`, worlds 2 and 4 against the world-1 forward over the
  fixture) and the `cluster_qwen3.8-27b_fp8_w2` template. The scale GEMM's
  large-row FP8 products (every family's dense MLPs, shared experts and FP8
  heads past the GEMV rows) run on `kernels/fp8w_gemm`'s per-weight form
  (#89): bitwise the tile kernel they replace (`fp8w_gemm_test` pins the
  pair) at 63–66 TF against its 26–30.

- **Qwen3.8-27B TP: the decode GEMV's rate at shard widths** (2026-10-04,
  #93): the four-node profile's 187 GB/s was measured against the spec's
  273; the GB10's achievable device read bandwidth is 233.6 GB/s
  (`micro_mem_bw`), and the fp8 decode GEMVs reach 89–97 % of it in
  isolation at the per-rank shapes (`micro_mma_gemv_shapes`, the new
  microbench over the two- and four-node matrices at 1 / 4 / 8 rows: the
  streaming form, the forced block widths, the row-chunk cores, the fused
  swiglu down). In-server they ran 7–20 % under that, the worst the MLP gate
  / up at eight rows: 68 unsplit blocks, because the MLP launches passed no
  workspace — they do now (split 2: 184 → 208 GB/s on four Sparks; the
  chunked cores at m <= 4 are unaffected). The lever at TP is the fold
  time: 138 bus collectives x ~49 us a step on four Sparks with DRAM idle,
  so the decode walk opens the Qwen3.8-Flash-Next prefetcher's boundary
  windows (`WeightPrefetcher`, a side-stream graph branch before each fold:
  the MLP's gate / up / down before the attention fold, the next layer's
  input projections or the head before the MLP fold; 20 MB at the Light
  rate from the sweep — 8 / 12 / 20 / 32 MB and the Full rate measured,
  32 MB and Full slower). Bit-identical on or off (greedy transcripts 4/4
  at every budget). Four nodes MTP depth 3: 48 → 46 ms a pass (greedy C1
  54.7 / 65.2 / 74.7 / 69.0 / 47.6 → 56.3 / 67.1 / 76.6 / 70.5 / 49.1
  tok/s); two nodes 82 → 80; the four-node drafter 54 → 53 with the
  windows and 52 with the split gate / up (54.9 / 86.3 / 120.8 / 101.2 /
  52.6; transcripts 4/4 against a same-dispatch MTP depth-5 world).

- **The DFlash2 drafter on the graph worlds** (2026-10-04, #92): the graph
  engine hosts the block proposal at every world. The verify is the 8-row
  recorded step (the sampled device pick's row limit `kSampleVerdictRows`
  6 → 8, the drafts point masses — the P(draft) rule); after the commit one
  recorded block forward off the device verdict replaces the draft picks
  (`Qwen35Model::session_graph_capture_block_draft`, a batched form per
  family): the anchor and the block's positions staged from the verdict
  and the committed device position, the drafter's heads / kv heads / MLP
  rows sharded across the ranks with two boundary folds a layer (the
  replicated 3.7 GB forward cost 17 ms a step on four Sparks; 3.8 sharded
  and bf12), the mask rows' top-16 on the vocab-sharded head merged across
  the ranks through one fold of 6-bit digits (`dflash2_topk_stage/merge`,
  the pick's wire form) so every rank walks the same proposal, the drafts
  fed to the next replay and a pinned per-slot mirror (`kernels/dflash2`).
  The block attention takes a split-key form (32 ranges + combine; the
  serial walk was ~2 ms a layer at 2K). `engine.dflash_model` serves on the
  fabric (with `decode_graph`) and on one Spark with the decode graph (the
  eager engine's transcripts 4/4, chat 71 against 95 ms a token); past four
  slots the engine batches the slots that fit the 32-row ceiling and
  replays scalar graphs beyond. Gates: `dflash2_kernels_test` (the merge at
  worlds 2 and 4, the staging, the split attention), greedy transcripts
  identical 4/4 to the same world's MTP depth-5 world at worlds 1, 2 and 4.
  Measured greedy C1 (`timed_load`, prose / code / json / math / chat): two
  nodes MTP depth 3 at 84 ms a pass, the drafter 32.1 / 50.3 / 71.0 / 57.6 /
  30.3 at 94; four nodes MTP 54.7 / 65.1 / 74.6 / 68.6 / 47.8 at 49 ms, the
  drafter 52.4 / 82.5 / 115.9 / 97.1 / 50.2 at 56 — the drafter leads code /
  JSON / math by 27–55 % and ties prose and chat, the MTP template's 4-row
  verify keeps the shorter pass, so MTP stays the two- and four-node
  template and the drafter is its mode (`--no-mtp --dflash-model
  z-lab/Qwen3.8-27B-DFlash2 --bf16-weights bf12`); the drafter recipes pack
  the drafter bf12 (+4.5 %, lossless). The nsys node trace of rank 0 at four
  nodes puts the target's fp8 GEMV at 35–38 ms a step for both recipes
  (6.9 GB a rank at 187 GB/s at four-node shard widths): the shared lever
  (#93).

- **The DFlash2 block drafter on Qwen3.8-27B** (2026-10-02, #80):
  `engine.dflash_model` serves `z-lab/Qwen3.8-27B-DFlash2` in place of the
  MTP draft on the eager world-1 engine — five bidirectional draft layers
  fed by target taps `[5, 19, 33, 47, 61]` through the split `fc`, 2-tap
  dynamic grouped convolutions, the rank-256 top-16 selector walk, the
  shared embedding and lm head, the drafter's K/V in five extra planes of
  the main pool (kernels/dflash2, models/qwen/dflash2, the `dflash2_*`
  methods of `Qwen35Model`, `DFlash2Speculator`). Every speculating slot's
  verify rows ride one physical pass; the multi-slot verify replays as a
  captured graph (`engine.dflash_verify_graph`), the redrafts stack across
  slots (`engine.dflash_draft_batch`), and `engine.dflash_depth` caps the
  verified width (exact at any value). `kSpecRows` 6 → 8 for the 8-row
  block (the MTP families' `mtp_depth` stays 1–5). The step returns the
  tokens it decided (the accepted drafts and the verify's next token) like
  the plain step, so transcripts equal a plain world's of the same verify
  width (4/4 identical to MTP depth 4); the selector's
  unary term is the candidate's logit (vLLM `_score_edges`), which makes
  the walk the proposal rule (+2–11 % tokens per step over a per-slot
  top-1). The 27B family's own GEMM instance takes the streaming mma form
  for 17..128-row decode batches (the drafter's weights read once per
  step); other families' dispatch is unchanged. Template
  `deploy/cluster_qwen3.8-27b_fp8_w1.example.json` (the world-1 recipe since 2026-10-04: the drafter is the faster
  single-node world on every class, so the MTP world is this template's mode, not a second file); references in
  `dflash2_kernels_test`, the host contract in `dflash2_speculator_test`,
  the config gates in `unit_tests`. Measured on one GB10 (greedy, exact
  numerics): 164 ms/step at C1 for 2.5–6.4 tokens per step by class —
  the author's bench 22–36 tok/s against MTP depth 2's 13.6–16.4; C4
  41–84 tok/s wall. Thanks to AhmmedSamier for the lane.

- **Serve Qwen3.8-27B-FP8 on the native engine, with MTP** (2026-10-01,
  #79): a new family, `qwen3_5` — the 27B dense model, 64 layers of
  Gated-Delta-Net (48, swish output gate) and full attention (16, GQA with
  the partial rotary and the per-head output gate) — served from its native
  blockwise-FP8 checkpoint `Qwen/Qwen3.8-27B-FP8` on one Spark. The
  streaming loader (`loader35`) reads the GDN in_proj (qkv/z) + out and
  the full-attention q/k/v/o projections and the dense gate/up/down MLPs
  as blockwise FP8 (E4M3 + 128×128 scales, widened to F32 at load), keeps
  the norms BF16, and loads the MTP draft layer onto the last full-attention
  slot; `Qwen35Model` runs the shared session core (paged K/V over the
  full-attention layers, model-owned GDN state per slot, the speculative
  verify/rollback, the decode graphs). MTP (`engine.mtp`, depth 2 in the
  template) drafts on the MTP head and verifies on the greedy path, so
  transcripts equal the plain world's. Numerics are the checkpoint's by
  default (block-scaled FP8 GEMV / streaming MMA at decode rows, the
  dequantized bf16 GEMM at prefill, the BF16 lm head); two opt-in keys
  trade exactness for speed and change greedy output —
  `engine.dense_weights: "fp8"` (the lm head requantized to block FP8; the
  templates ship it: −19 ms per MTP pass at the same acceptance and eval) and
  `engine.prefill_fp8_per_tensor` (prefill GEMMs on cuBLASLt's per-tensor
  e4m3 kernels, ~2x the prefill rate, +23 GiB). One template,
  `deploy/cluster_qwen3.8-27b_fp8_w1.example.json`; kernel references in
  `qwen_full_attn_test` / `full_attn_test` / `qwen_norm_test`, the config
  and binding gates in `unit_tests`. Measured on one GB10 (greedy, the
  exact defaults): T=1 119 ms/step (the byte floor is 105), MTP depth 2
  151 ms/pass at 2.1–2.9 tokens per pass, T=1 and MTP transcripts
  identical; with both FP8 levers 132 ms/pass. The dense MLP's k=17408
  down projection runs the streaming mma form at every row count (the
  GEMV's 48 KiB staging holds one row of it, so a 3-row pass read it three
  times: 74 ms of a 180 ms step). Thanks to AhmmedSamier for the port.

- **Assistant thinking history through LiteLLM** (2026-10-03): accept
  Anthropic-style assistant thinking parts by folding their text into
  `reasoning_content`, preserving explicit reasoning strings and dropping
  redacted payloads. Thinking-only content becomes an empty string, including
  in tool-call history. Regression tests cover UTF-8 ownership, precedence,
  empty/redacted parts and invalid fields/roles. Fixes #74 via #76.
- **Serve DeepSeek-V4-Flash** (2026-10-01): the seventh family,
  `deepseek_v4` (`deepseek-ai/DeepSeek-V4-Flash-0731` as shipped: MXFP4
  routed experts, FP8 block-128 attention projections and shared expert,
  BF16 compressors, router and head). New: the sliding-window attention
  over one shared 512-wide latent with positional rings, the ratio-4
  (overlapping groups, a 64-head indexer selecting 512 entries) and
  ratio-128 compressed caches, token-table routing on layers 0–2
  (`GlmMoeLayer::set_hash_routing`), the two-pass hyper-connection sites,
  the DSpark block draft over three window-only stages, the family's
  loader / model / fixture ladder (config and binding units, loader, model,
  TP worlds 2 and 4, graph engine), the port of the release's
  `encoding_dsv4.py` prompt encoder with its DSML tool dialect, deployment
  templates for two and four nodes at the model's full 1M-token context,
  and `tools/dsv4_torch_reference.py`, which runs the release's own
  `inference/model.py` layer code against the engine's per-layer dump.
  `engine.mtp_draft` (`auto` | `sampled` | `greedy`) chooses how a sampled
  request's drafts are picked; this family defaults to the draft's most
  likely token, accepted with the target's probability of it. With such
  drafts a sampled request follows the confidence-scheduled verify depth
  too, at `engine.mtp_schedule_sampled_scale` (0.93) of the head's
  acceptance — the templates schedule the five-draft block for greedy and
  sampled requests alike. Measurements
  in [docs/benchmarks.md](docs/benchmarks.md) and the
  [campaign record](benchmarks/results/2026-10-01-deepseek-v4-flash/README.md);
  the [model card](docs/model_cards/DeepSeek-V4-Flash.md).
- **DeepSeek-V4-Flash decode and prefill kernels** (2026-10-01): L2 weight
  prefetch windows at the collectives and inside the attention layer;
  split-K on the decode walks' dense projections; the mHC dots, finish and
  sublayer norm in one launch with the comb on a side stream; the window
  attention on the tensor-core listed kernel; a tensor-core scoring pass
  for the 64-head decode select (`dsa.cu`); the collective's shared-memory
  staging budget raised to 96 KiB and sized by the world's peers, so a
  four-row 4096-wide fold is staged; the shared expert's slots run beside
  the first routed ones (`GlmMoeConfig::shared_slots_early`, this family
  only — the order alone moves); the attention finish computing its split
  weights once per block with batched loads (`csa2.cu`, shared with
  DeepSeek-V4.1); the K/V row's norm, rotation, quantization, ring append
  and window list in one launch (`dsv4_kv_tail`); the Markov bias's row
  loads batched. Prefill: the index-select tile sized by the chunk's own
  entry count instead of the 1M-context bound, the dense FP8 projections
  through the grouped tensor-core GEMM and the BF16 compressor projections
  through a new dense tensor-core kernel (`kernels/dense_mma_bf16w`), both
  row-count-invariant so a chunked prefill stays bitwise the one-shot.
  Every decode change is bitwise the chain it replaces (transcripts
  identical on the fabric). Fixes found on the way: a block-draft family's
  decode batch must cover slots × block (depths under 4 at six slots failed
  to boot; DeepSeek-V4.1 shared the bug), the scheduler's retire line
  subtracted the prefill twice from the decode time, and the 64-head decode
  select lacked its dynamic shared-memory opt-in.
- **DeepSeek-V4-Flash decode, second round** (2026-10-02), 39.0 → 37.0 ms
  per depth-3 pass on four nodes and the scheduled depth's whole range:
  the ratio-128 compressors pool lazily — the open group's layer inputs
  wait in a positional ring and the group is projected once, when it
  completes, on the prefill's tile chain (`launch_dense_mma_bf16w_groups_f32`;
  a pass that completes no group reads no compressor weights, and a
  decode-published entry is bitwise a prefill's of the same inputs); a
  compressing layer's window attention runs on a side stream beside its
  compressor and selection; the draft block's base logits come off a
  block-FP8 copy of the LM head (half the bytes; 132 MB per rank on four
  nodes) and every logit within reach of a pick's maximum is recomputed
  from the BF16 rows after the Markov bias (`dsv41_dspark_rescore`), so the
  drafts are the BF16 head's — pass counts identical on the fabric. The
  bus's graph-variant budget is 128 (was 64), so six slots schedule all
  five verify depths instead of 1 / 3 / 5: replayed over 3,140 traced
  passes the two missing depths were 1.5 % of a sampled stream.
  `engine.mtp_schedule_sampled_scale` defaults to 0.93 (the head is near
  calibrated for a sampled request behind its first draft; 0.8 stopped the
  deeper drafts short). Tried and reverted with their numbers in the
  source: the shared expert on tensor-core GEMVs on a side stream (a
  kernel launched behind the slot kernels' block queue is dispatched after
  it), and the split-K fold inside the GEMV launch (level).
- **Batching past the decode-row ceiling** (2026-10-02): a block-draft
  recipe whose slots times the full block exceed the decode batch (six
  DeepSeek-V4-Flash slots at depth 5: 36 rows of 32) replayed one scalar
  graph per live slot whenever every slot was live — six concurrent
  requests ran at one stream's aggregate rate (64–110 tokens/s). Under the
  scheduled verify depth the engine now builds a depth-capped family for
  all slots (five rows each: up to four drafts; a step the policy asks
  five of verifies four, which is exact), and at a fixed depth the widest
  family takes the slots it covers with the rest stepped alone: 127–167
  tokens/s at six requests on four nodes. Transcripts are plain decode's
  in both forms (`dsv4_engine_test`). `kBusMaxGraphVariants` is 128.
- **DeepSeek-V4-Flash prefill: the boundary folds under the other half's
  work** (2026-10-02): a prefill walk of 512 rows or more runs each layer
  in two row blocks, so a block's all-reduce — the GPU idle through it, a
  fifth of a 2K prefill on four nodes — runs under the other block's
  compute: the second block's attention under the first's fold, the next
  layer's site and attention of the first under the second's MoE fold,
  the first block's router and the second's expert accumulation under the
  folds between (`GlmMoeLayer::route_prefill_rows`,
  `enqueue_prefill_phased`, `accumulate_prefill_rows`; the expert chain
  stays one pass over every row — its cost is the weights it decodes).
  `BusStreamReducer` gains the asynchronous bulk fold
  (`begin_async` / `end_async`). Every row's arithmetic is the one-block
  walk's: logits and state bitwise, for one prompt and for a group's spans
  (`dsv4_tp_test`; in a group the block boundary sits on the nearest
  span's own block grid), and on the fabric prompts of 222 to 2,494
  tokens sent together return the transcripts they return alone. Cold
  prefill on four nodes 1,466 → 1,327 ms at 2K tokens and 6,056 → 5,394 ms
  at 8K; on two nodes 2,150 → 2,021 ms and 8,651 → 7,926 ms. The decode
  walks: the shared expert's gate and up projections on two side streams
  ahead of the router with its SwiGLU fused into the down GEMV's staging
  (`GlmMoeConfig::shared_mma_aside`, `launch_mma_gemv_fp8_swiglu_f32`):
  a depth-3 pass 37.0 → 36.5 ms on four nodes and 61.9 → 59.8 ms on two;
  the templates' schedule constants re-measured (3.7 / 22.0 ms and
  7.2 / 31.2 ms).
- **Prompts that arrive together are read in together**
  (2026-10-02, DeepSeek-V4-Flash): concurrent long prompts were admitted
  one whole prompt at a time — each a full read-in behind the last (a
  prefix-cache snapshot plan kept a prompt out of the cold group prefill,
  and two 2K prompts exceed one forward anyway), with every running
  stream stalled through each. The family's prefill is now resumable
  (`kResumablePrefill`: decode steps and cancellation between chunks),
  and the in-flight prompts' next chunks ride ONE forward as its spans
  (`session_prefill_advance_group`, `SchedulerEngine::advance_prefill_group`;
  bitwise the one-shot prefill — `dsv4_model_test`). The scheduler, on
  such an engine: reads every prompt past one aligned chunk in through
  its cursor, begins every queued one in the same tick, splits the tick's
  budget max-min fair (a short prompt finishes and is not held behind a
  long one), levels the remainders within two ticks of the end so the
  prompts finish — and start decoding — together, and takes the busy
  budget as a quantum per reading prompt. Rank 0 holds the first arrival
  at an idle engine `engine.admission_gather_ms` (3) for the rest of its
  burst. The automatic budget is 256 tokens per reading prompt beside
  decoding requests and the whole 4096-row forward with none. On four
  nodes, llama-benchy pp2048/tg128: 88.9 tokens/s at two concurrent
  requests (was 58.8) and 120.6 at five (was 56.5). Also: a group's
  second and later requests had their draft windows filled from the
  first request's rows (lower acceptance for the first ~128 tokens, never
  a wrong token — the verify decides); the draft's projection now covers
  only the rows its window keeps; and a slot count whose draft blocks
  exceed the decode batch is refused at startup instead of seven minutes
  into the warm-up.
- Fix the sampling fallback of a reduced-depth batched replay
  (2026-10-02): a batched replay snapshots its verify rows at the
  replay's rows per request, and the host fallback read them at the full
  block's stride, so a sampled slot past the first gathered another
  slot's row — which the engine refuses (`the gathered fallback row is
  not the row the device decided over`), ending the serve process.
  Reachable since sampled requests follow the schedule
  (`engine.mtp_schedule_sampled_scale`, DeepSeek-V4-Flash): any step with
  two or more sampled requests at a reduced depth whose pick fell back.
  `dsv4_engine_test` batches two sampled slots through every depth.
- Fix `--no-mtp` on a template that sets `engine.mtp_depth` or
  `engine.mtp_schedule` (2026-10-02): the plain world now drops the
  template's draft depth and schedule unless the command line asked for
  them; before, the launch failed with "--mtp-depth N needs --mtp".

- **W4A4 NVFP4 expert prefill is opt-in** (2026-10-01): default to
  W4A16 with BF16 activations; `DGPP_MOE_W4A4=1` explicitly enables
  activation quantization and its workspace. PR #50 reported kernel,
  latency and BFCL results, but the paired teacher-forced and task-evaluation
  gate required by [#68](https://github.com/HawkBearPig/dgpp/issues/68)
  remains outstanding. Calibration scales alone no longer enable W4A4.

- **Compact serving logits across all model families**: extend PR #67's
  bounded Qwen head storage to BF16 and packed Qwen heads, GLM-4.7,
  GLM-5.3, GLM-5.3-Flash, DeepSeek-V4.1 and MiMo. Serving stores only the
  decode capacity and projects each prefill request's final row; diagnostic
  full-row outputs remain available. BF16 head reassociation is checked
  against FP64 and teacher-forced scores, with strict cache, MTP and decode
  checks. Memory plans account for the smaller allocation.
- Fix out-of-bounds head and draft-window writes when bounded DeepSeek
  prefill ends with a chunk shorter than its retained replay window.

- **Engine behavior is set by the deployment config, not the environment**
  (2026-09-30): the switches this campaign had added as environment
  variables are `engine.*` keys — `expert_gemm` (`wide` | `wide3` | `wide4` |
  `wide4r` | `narrow`; was `DGPP_PACKQ_GEMM`), `expert_gemm_prefetch` (was
  `DGPP_PACKQ_PREFETCH`), `expert_tile_list` (was `DGPP_MOE_TILE_LIST`),
  `expert_gemm_pair` (was `DGPP_MOE_PACKQ_PAIR`) and `ngram_prestage` (was
  `DGPP_QWEN_PLE_PRESTAGE`) — parsed and validated with the config, carried
  to every rank in the worker record, applied through setters (the env reads
  are gone; `packq_prefill_bench --prefetch N` sets the distance). The
  remaining environment switches in the tree are the next sweep.
- **The packed expert GEMM parameterized on its warp count** (2026-09-30,
  round 21 of the [record](benchmarks/results/2026-09-28-qwen-autoround-int4/README.md),
  plan §6.16): four-warp forms at 64 x 32 warp tiles (`engine.expert_gemm:
  "wide4"`, the decoded tile; `"wide4r"`, register decode and three stages),
  bitwise the shipped eight-warp kernel and gated so in `packq_gemm_test`.
  Measured −2..−6 % on the launch in the bench and +0.6 / −1.2 / −0.3 % of
  cold prefill on the fabric: the default stays the eight-warp form, the new
  ones are opt-ins, and the record corrects the round's estimate (the
  in-situ launches already run at 63–67 % of the dense bf16 rate).

- **The prefill's accuracy trades as deployment keys, off by default**
  (2026-09-30, round 20 of the
  [record](benchmarks/results/2026-09-28-qwen-autoround-int4/README.md),
  plan §6.15): `engine.prefill_bf16_partials` (the packed expert chain's
  down projection in bf16, its partials summed from bf16 — the former
  `DGPP_MOE_PACKQ_DOWN_BF16` env knob is gone), `engine.prefill_fold_scales`
  (the wide packed expert GEMM with each group's scale folded into the
  bf16 weight values and one fp32 accumulator; kernel variant 3) and
  `engine.prefill_fp8_gemm` (the block-FP8 dense stack's prefill-shaped
  products on the fp8 tensor cores from per-token 1 x 128 e4m3 activations
  with the checkpoint's 128 x 128 weight scales — `kernels/fp8_gemm`, a new
  quantizer + mma.sync e4m3 GEMM with per-group fp32 promotion; requires
  `dense_weights: "fp8"`). None is bitwise the default chain; each is
  gated in its kernel test (2^-7 relative RMS of the exact chain for the
  fold; 1e-7 of the exact sum of its quantized inputs and 2.5–2.8 % of the
  dequantized chain for the fp8 GEMM), refused with the wrong type in the
  config, forwarded to every rank in the worker record, and documented in
  docs/operations.md and the README key table. Measured on the fabric
  (one Spark, the hybrid, cold prefill 2K / 8K / 32K against 1.29 / 4.37 /
  17.64 s): the bf16 partials −6.6 / −3.8 / −2.8 %, the fp8 GEMM −3.1 /
  −0.3 / −0.2 % (its kernel runs at cuBLASLt's bf16 rate: 86–98 TF, ~35 %
  of the fp8 peak, the gap recorded), the fold −1.0 / +2.0 / +3.6 % (a
  measured loss: it stays off); the bf16 partials + fp8 GEMM pair 1.21 /
  4.23 / 17.10 s (−6.3 / −3.2 / −3.1 %) with HumanEval 159/164, GSM8K
  291/300, extraction 100/100 against the default chain's 159 / 292 / 100.
  The AutoRound example template turns on `prefill_bf16_partials` and
  `prefill_fp8_gemm` (evals inside the default chain's band; the benchmarks
  list the default and the levers-on rows); the fold stays off everywhere.
  The README gains an accuracy-and-correctness section and
  docs/optimizing_performance.md the levers' guide.

- **Restore clean-checkout builds** ([#70](https://github.com/HawkBearPig/dgpp/issues/70)):
  skip optional microbenchmark targets when any of their source files are
  absent, including the locally ignored `qwen_dense_bench.cu`.
  `DGPP_BUILD_BENCHMARKS` controls these targets; production presets disable
  them. Missing files under `benchmarks/` no longer block configuration.
- **The hybrid's prefill: the n-gram rows of the next chunk gathered
  while this one runs** (2026-09-30, round 19 of the
  [record](benchmarks/results/2026-09-28-qwen-autoround-int4/README.md)):
  a scalar prefill chunk hashes the following chunk's tokens (its n-gram
  context this chunk's last two) and publishes them on a second staging
  channel the gather thread serves while this chunk's layers run; the next
  chunk claims the rows by request, position, row count and a token check
  and skips its own gather. The PLE layer's wait for the table — 1.08 s of
  GPU time on a 32K prefill, the gather of ~65K table rows a chunk from
  NVMe outrunning the two layers before it — is gone: −0.65..−0.75 s at
  32K when the table's pages are cold, level when the page cache holds
  them; bitwise (the long-prompt sha on every leg; an in-situ self-check,
  `DGPP_QWEN_PLE_PRESTAGE_CHECK=1`, gathers both channels and compares
  them; `DGPP_QWEN_PLE_PRESTAGE=0` keeps the one channel). The round also
  served the author's vLLM stack on the same box for a kernel-level
  comparison (their expert GEMM 1.8x ours per token, their fp8 dense GEMMs
  and bf16 partials; our attention, indexer, hyper-connection glue and
  n-gram path level or ahead) and left two opt-in forms: the expert GEMM
  with its weight fragments decoded in registers and three pipeline stages
  (`DGPP_PACKQ_GEMM=wide3`, bitwise, level), the gate and up projections
  as one launch (`DGPP_MOE_PACKQ_PAIR=1`, bitwise, level) and the down
  projection's bf16 partials (`DGPP_MOE_PACKQ_DOWN_BF16=1`, −3..−4 %, not
  bitwise, off).
- **The hybrid's prefill: the expert GEMM's weight stream kept ahead of
  the copies** (2026-09-29, round 18 of the
  [record](benchmarks/results/2026-09-28-qwen-autoround-int4/README.md)):
  the wide packed-int expert kernel prefetches each activation row's line
  and each code row's line three k-steps ahead into L2
  (`DGPP_PACKQ_PREFETCH`, default 3) — the two-stage pipeline's one step
  of lookahead had left the launch's 419 MB weight stream serialized with
  its compute (a Zipf-routed gate launch 4.6–5.0 → 3.5 ms, the down 6.8 →
  5.8, a 556-token launch 3.6 → 2.1 on the device-memory bench; the bench's
  managed buffers had hidden it behind ~1 ms of page mapping a launch, so
  `packq_prefill_bench` gained `--device-copy`, `--gather`, `--same-weights`
  and `--shuffle-rows`) — and runs a compact tile list over the routed
  segments (`launch_moe_tile_list`, `DGPP_MOE_TILE_LIST`): one block per
  real 64-row tile instead of n-tiles × the longest segment × 512
  segments (160K blocks, 97 % exiting, when one expert takes most of a
  chunk), and no per-layer host sync for the longest segment. Both bitwise
  (`packq_gemm_test`, the bench's byte compare, the long-prompt sha on
  every fabric leg). On the fabric at 4,096-token chunks: 2K 1.40 → 1.29 s,
  8K 4.61 → 4.42, 32K 18.46 → 17.82 (−7 / −4 / −4 %).
- **The hybrid's decode at the author's protocol: 57 → 54 ms a pass, the
  head in bit planes** (2026-09-29, S6 rounds 4–12 of the
  [record](benchmarks/results/2026-09-28-qwen-autoround-int4/README.md)):
  the int8 head is stored as three per-row planes (high nibbles, bits 3..2,
  bits 1..0) and an argmax-only head read — every MTP draft step and the
  greedy verify — takes a pass over the high six bits that gives each row a
  provable logit interval, then reads only the rows whose interval reaches
  the best lower bound (a few hundred of 248,320) in full through the packed
  core's exact chain: 75 % of the head bytes at line rate, the pick's argmax
  and its logit bit for bit (`kernels/packq_head`, `packq_head_test`; sampled
  rows read every plane and are the row layout's chain bit for bit; the
  engine writes a per-slot greedy flag beside the sampling specs because the
  decode graph is fixed at capture; github issue #69 carries the
  generalization to the other head formats and the DRAM-burst lesson that
  the planes must be contiguous per row). Also in the rounds: the QSA
  attention over the selected keys 181 → 137 us; the draft layer's bf16
  projections and fc matrices packed bf12 under the hybrid's fp8 dense stack;
  the recurrent state kept as a checkpoint plus the pass's row inputs
  (replayed next pass) instead of three snapshot stores a layer; the up GEMV
  of a hyper-connection site with the mix in its epilogue, the routed
  accumulation in the shared expert's down epilogue, the combine in the next
  site's norm launch, the chunked greedy argmax, the n-gram gather on a
  thread with a device wait — each bitwise the chain it replaces. Measured
  and left as they were: the byte prefetch (a size cap or off costs 1.3–2 ms
  a pass), depth 3 (depth 4 is slower and takes other kernel families), and
  a per-launch prologue prefetch of each GEMV warp's first weight pass
  (−4..−8 % on `qwen_dense_bench`'s cold ring, +1.1–1.5 % a step on the
  fabric: the step's matrices are already in L2; the bench's new
  `--sweep` fits the per-launch intercept per kernel family and
  `--only pdl` runs the down/up pair as a graph-launched chain). The
  decode step at context, measured through the endpoint: 52.6 ms at
  under 200 tokens, 55.7–56.9 from 3K to 47K (a fixed selection budget);
  the listed attention's tile gather became cp.async phases in the same
  shared buffers (the next tile's K rows under this tile's PV, its V rows
  under the next scores; the arithmetic unchanged, `DGPP_QSA_ASYNC=0`
  keeps the serial gather): −0.4 to −0.5 ms a step from 12K to 47K. The
  GDN layer's a and b projections ride the fp8 in-projection launch as
  bf16 problems (one launch fewer per GDN layer, bitwise); the batched
  group norm staged into the down GEMV (bitwise) measured +0.8 ms a step
  and stays behind `DGPP_QWEN_GR_NORM_FOLD=on`. The hybrid's prefill, profiled
  at 32K (the packed-int expert GEMMs 34 %, cuBLAS 15 %, the hyper-connection
  elementwise passes 12 %): a wide-tile packed-int tensor-core GEMM (64 x 128
  x 64, two blocks per SM, the codes decoded once per k-step into a bf16
  tile — bitwise the narrow kernel, `DGPP_PACKQ_GEMM=narrow` keeps it), the
  prefill's combine riding the next site's norm pass, and the chunk's page
  advice batched: prefill −7 to −9 % at 2K / 8K / 32K, bit for bit.
  `engine.draft_vocab` (opt-in, never a headline): the draft head scores a
  `.npy` set of ids (`tools/build_draft_vocab.py`), 49 ms a pass, outputs
  unchanged.

- **Qwen3.8-Flash-Next AutoRound int4 hybrid on one Spark** (2026-09-28):
  `Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN` is served in
  its packed form — int4 group-128 experts (backbone and draft) and the int8
  group-128 head through the packed-int cores, which gained a scale format
  (f16 scales per 128 beside the bf16 per 64 of full GLM-5.3) and the Qwen
  widths; the GPTQ layout is transposed into packed rows at load, its scales
  untouched, its zeros verified and dropped; the checkpoint's block-FP8 side
  layers load as shipped (bitwise our own at-load encode of the BF16
  originals), and the classes it ships in BF16 stay BF16. The n-gram table
  comes from another cached snapshot (`engine.ngram_table_model`, the FP8
  release's shards); `download_model.py` fetches and syncs it. New gates on
  the hybrid's fixture: forward parity against the pure-Python reference
  (which dequantizes GPTQ exactly), the decode and prefill-head gates and
  the loopback engines. Performance on the real checkpoint is the next
  campaign (docs/qwen38_autoround_int4_plan.md, the
  [record](benchmarks/results/2026-09-28-qwen-autoround-int4/README.md)).

- **Prefix cache: an entry floor, a head cut, and arenas sized to the node**
  (2026-09-28): `engine.prefix_min_tokens` (default 1024) takes no snapshot
  below that position, so short probes and side requests never push a long
  conversation's entries out of the arena; `engine.prefix_head_snapshots`
  (default on) keeps a cold prefill's cut at its first structural boundary,
  a long system prompt's end, where the next conversation under it or the
  turn after a client-side compaction attaches. Both ride the settings and
  warm records. A single-Spark field log replayed under a 13-slot arena lost
  every turn of a 66K conversation to three 45-token requests per turn and
  none with the floor. The checked-in recipes now give the arena the memory
  the node has left under the plan's headroom, except the one-node Qwen
  recipes, whose spare memory is the page cache behind the mmap'ed n-gram
  table. See the [record](benchmarks/results/2026-09-28-prefix-entry-floor-head-cut.md)
  and the [sizing guide](docs/prefix-cache.md).


- **Faster batched GLM decode** (2026-09-27): merge the validated row-batched
  attention projections, reusing weights across decode rows while preserving
  each row's bitwise result. The recorded four-Spark full GLM-5.3 comparison
  reduced decode-step time by 1.1% at four concurrent requests and 1.7% at
  eight; single- and two-request performance was unchanged within the
  measured spread. GLM-5.3-Flash passed a separate regression check. See the
  [benchmark and integration record](benchmarks/results/2026-09-26-kvb-decode-rows/README.md).

- **Qwen image prefill** (2026-09-27, PR #62): image embedding staging now
  splits chunks wider than the vision workspace, and image prefills can
  yield between scheduler ticks. Review follow-up scopes image borrows and
  staging buffers to each executing prefill, preserving cold image inputs
  and isolating interleaved cursors. CUDA regressions cover MTP catch-up,
  cancellation and slot reuse; see the
  [validation record](benchmarks/results/2026-09-27-qwen-image-prefill.md).

- **Lazy SSE preambles** (2026-09-26, PR #57): chat and legacy completion
  preambles wait for the first output, preserving first-event prefill timing.
  Empty completions still receive a preamble before their terminal chunk,
  and truncated UTF-8 output receives one before the replacement character.
  Each chat choice retains its own preamble. See the
  [validation record](benchmarks/results/2026-09-26-lazy-sse-preamble.md).

- **JSON Unicode and UTF-8 responses** (2026-09-26, issue #59): escaped
  UTF-16 surrogate pairs now decode to the same UTF-8 as raw supplementary
  characters; lone surrogates become U+FFFD and malformed hex escapes are
  rejected. Streaming and complete responses replace overlong encodings,
  encoded surrogates and out-of-range code points while preserving valid
  characters split across tokens. Legacy completion logprob strings now
  receive the same validation. Regression tests cover Unicode boundaries,
  chunk partitions, escaped/raw requests, reasoning and both response modes.

- **SSE keep-alive comments** (2026-09-26, issue #49): streaming chat and text
  completions send comments after 30 seconds of silence while queued,
  prefilling or between output chunks. Cluster `http.sse_ping_interval`,
  `--sse-ping-interval` and request `sse_ping_interval` configure the interval;
  `-1` disables it. Choices share one timer; comments do not change output,
  usage, finish events or engine deadlines. See the
  [validation record](benchmarks/results/2026-09-26-sse-keep-alive.md).

- **Deployment templates: `engine.default_max_tokens` 256 → 32768**
  (2026-09-25): every template's answer budget for requests that omit
  `max_tokens` was 256 tokens, and reasoning tokens count against it. Agent
  clients such as Hermes deliberately send no `max_tokens` to an
  OpenAI-compatible endpoint, and Qwen3.8-Flash-Next thinks at the template's
  default effort (xhigh), so the first call of a Hermes session spent 271–611
  reasoning tokens before its first tool call and retired at the budget with
  `finish_reason: "length"` and no visible content — the agent's
  "thinking-only / empty response" recovery loop. The YaRN 512K template's pool
  grows to 565 248 tokens so a full-length request without `max_tokens` is
  still admissible (53.53 GiB per rank against 52.97). The binary's own
  fallback when a configuration omits the key is unchanged at 256; existing
  local deployment files must be edited (or re-copied from the template).
  Evidence:
  [record](benchmarks/results/2026-09-25-hermes-default-max-tokens/README.md).
- **Serve MiMo-V2.6-Flash** (2026-09-22): the sixth family, `mimo_v2`
  (`XiaomiMiMo/MiMo-V2.6-Flash-RL` as shipped: MXFP4 routed experts, fp8
  block-128 dense projections, BF16 o_proj / head / eh_proj packable into
  their 12-bit companions). New: paged GQA attention over 192/128-wide
  heads with a 128-token sliding window and a per-head sink on 39 of 48
  layers, the fused pre-sharded qkv projection read in the checkpoint's
  chunk layout, a per-layer-width K/V pool, the family's loader / model /
  fixture ladder (attention oracle, loader, numpy forward reference, decode,
  TP worlds 2 and 4, graph engine), the Qwen2 pre-tokenizer pattern, the
  compact `<tool_call>` dialect and the model-opened `<think>` flow (confined
  to that dialect), deployment templates for two and four nodes, and the
  MXFP4 GEMV core's K = 2048 / 4096 instantiations. Measurements in
  [docs/benchmarks.md](docs/benchmarks.md) and the
  [campaign record](benchmarks/results/2026-09-22-mimo-v26-flash-opt/README.md);
  the plan in [docs/mimo_v26_flash_plan.md](docs/mimo_v26_flash_plan.md).
- **MiMo-V2.6-Flash decode and prefill kernels** (2026-09-22, plan §7):
  the residual add fused into the two-rounding norm that follows it
  (`kernels/add_rmsnorm`, every MiMo block boundary), the decode attention
  in one launch (finish + split partials + last-block combine, the batch's
  same-request rows overlaid from the projection, register-prefetched
  tiles), the MXFP4 chunk decode through the hardware f16 conversion when
  its scale allows (bitwise the exact path; DeepSeek-V4.1-Flash shares
  it), and the query-tiled tensor-core prefill attention
  (`kernels/mimo_attn_prefill.cu`: each K/V tile staged — and under the
  fp8 cache dequantized — once per 64 query vectors instead of once per
  row, `mma.sync` bf16 scores and PV; prefill rows are no longer bitwise
  the split-KV kernel's, l2-relative 1e-4 against it). The decode step at
  C1 with MTP −1.5 % on four nodes, the plain T=1 step unchanged (the
  in-situ profile, the cold microbench `mimo_step_bench --cold` and the
  rejected experiments — a register cap, an early L2-persisting prefetch
  of the next layer's projection — are in the plan and the
  [A/B record](benchmarks/results/2026-09-22-mimo-v26-flash/ab/README.md)).
  `WeightPrefetcher` gains a persisting-window mode and a line release
  (`applypriority.global.L2::evict_normal`) for such experiments.

- **Support sixteen Qwen requests at MTP depth 3** (2026-09-21): widen decode
  graphs to 64 rows and keep wide BF16 projections kernel-only. Preserve
  existing small MTP projection dispatch, including expanded hyper-state
  rows, and reject scheduled verification when graph capacity is insufficient.
  See the [validation record](benchmarks/results/2026-09-19-qwen-c16-mtp3-upstream.md).

- **Enable streaming MMA in the Qwen NVFP4 recipes** (2026-09-21): matched
  teacher-forced head checks pass on one and two Sparks, including the YaRN
  template, with exact unchanged-mode repeats and no global top-1 changes.
  Add a real-checkpoint scoring tool and strict comparison gate. Recipes
  select `fp8_head: "mma"`; `--fp8-head gemv` restores the previous path.
  See the [numerical validation record](benchmarks/results/2026-09-21-qwen-fp8-head-numerics.md).

- **Reduce DeepSeek-V4.1-Flash decode latency** (2026-09-21,
  [#28](https://github.com/HawkBearPig/dgpp/issues/28)): parallelize decoder
  candidate and restricted-entry scoring, replace streaming selection sorts
  with exact radix selection, and skip scoring when all entries fit. Preserve
  score arithmetic, original-ID ties, newest-block handling and output order.
  Reuse the existing workspace and remove the partial-list buffer, saving
  3.75 MiB per rank with six slots and depth-four MTP. See the
  [performance and parity record](benchmarks/results/2026-09-21-deepseek-selection/README.md).
- **Reuse long documents when their question changes** (2026-09-21,
  [#17](https://github.com/HawkBearPig/dgpp/issues/17)): save one earlier
  prefill snapshot alongside the final cut, preserve both through resumable
  prefill, and check MTP's lookahead token before reuse. Report snapshot
  capacity in memory plans and distinguish arena pressure from KV-pool
  pressure. See the [validation record](benchmarks/results/2026-09-21-prefix-document-reuse.md)
  and [sizing guide](docs/prefix-cache.md).
- **Keep batched MTP fallback decisions isolated** (2026-09-21,
  [#15](https://github.com/HawkBearPig/dgpp/issues/15)): preserve each request's
  verified drafts before another request's fallback publishes new drafts.
  Repair the live sampling oracle, cover grouped and staggered admission,
  slot reuse and disabled pipelining, and test batched fallback counters.
  See the [validation record](benchmarks/results/2026-09-21-mtp-fallback-isolation.md).
- **Reduce QSA selection cost at long context** (2026-09-21,
  [#19](https://github.com/HawkBearPig/dgpp/issues/19)): use exact radix
  selection above 2048 compressed pools, preserving score arithmetic,
  ties, token order and graph workspace. Add a cold/warm indexer benchmark
  and permanent large-pool, partial-tile and graph-replay correctness gates.
  See the [performance and parity record](benchmarks/results/2026-09-21-qwen-qsa-select.md).
- **Make the YaRN release check accurate and resumable** (2026-09-21,
  [#16](https://github.com/HawkBearPig/dgpp/issues/16)): allow reasoning in
  retrieval probes, correct decode timing units, repeat the latest cached
  probe, and budget concurrent streams for serialized long prefills.
  Save incremental reports with `--resume`; explain pool headroom when YaRN
  is already enabled. Reject invalid correction-band theta and compare
  final-position decode logits with the full forward. See the
  [validation record](benchmarks/results/2026-09-21-yarn-release-check.md).
- **Tighten decode metrics bookkeeping and documentation** (2026-09-21,
  [#20](https://github.com/HawkBearPig/dgpp/issues/20)): record each replay's
  end event immediately after its graph launch, before host counter updates.
  Cross-link the batch and speculative-counter references, retain the fixed
  sixteen-bucket histogram contract, and give the depth-one sampled-MTP
  gate its own port. See the
  [validation record](benchmarks/results/2026-09-21-metrics-follow-ups.md).
- **Constrain declared tool argument names by default** (2026-09-21,
  [#11](https://github.com/HawkBearPig/dgpp/issues/11),
  [#12](https://github.com/HawkBearPig/dgpp/pull/12)): top-level argument
  names close to declared `properties` unless `additionalProperties: true`
  explicitly opts out. Repeated Qwen XML parameters become literal content.
  The checkpoint grammar gates cover both the closed default and the opt-out
  without altering reference renders. See the
  [validation record](benchmarks/results/2026-09-21-tool-key-closure.md).
- **Reconcile timed benchmark counters** (2026-09-20,
  [#24](https://github.com/HawkBearPig/dgpp/issues/24)): `scripts/timed_load.py`
  waits for published counters to match completed requests under a deadline,
  including warmup and isolation, so late snapshots no longer cause false
  unexpected-traffic failures. Genuine mismatches still fail with expected
  and observed counts. The dated script path remains compatible. See the
  [validation record](benchmarks/results/2026-09-20-timed-load-counters.md).
- **Compare effective settings across ranks** (2026-09-20,
  [#18](https://github.com/HawkBearPig/dgpp/issues/18)): automatic graph
  batch thresholds no longer cause false settings-override warnings on
  peers. Genuine threshold and concurrency differences still warn; the
  engine settings remain unchanged. See the
  [validation record](benchmarks/results/2026-09-20-rank-settings-default.md).
- **Reject duplicate GLM tool arguments** (2026-09-20,
  [#22](https://github.com/HawkBearPig/dgpp/issues/22)): a repeated argument
  name makes the tool block literal content instead of a structured call
  with duplicate JSON keys. Valid calls retain their argument types and
  can reuse the same names in later calls. See the
  [validation record](benchmarks/results/2026-09-20-glm-tool-duplicate-keys.md).
- **Opt-in 512K YaRN context for Qwen3.8-Flash-Next** (2026-09-20,
  [#3](https://github.com/HawkBearPig/dgpp/pull/3)): `engine.rope_scaling`
  with YaRN factor 2 extends the ceiling to 524288 tokens; omitting it preserves the
  plain RoPE path bit for bit. Includes the two-node NVFP4 `_w2_yarn512k`
  deployment template and `scripts/qwen_yarn_release_check.py`. Two-Spark
  validation retrieved 5/5 needles at both 261K and 522K prompt tokens:
  cold prefill 0.909 / 1.397 ms per token, decode about 60.7 / 96.0 ms per
  pass. See the [validation record](benchmarks/results/2026-09-20-qwen-yarn512k.md)
  for the recipe, measured scope and release-check limitations.
- **Cross-compile for DGX Spark from x86 Linux** (2026-09-20,
  [#5](https://github.com/HawkBearPig/dgpp/pull/5)): `scripts/spark-cross`
  and the `spark-cross` CMake preset build the ARM64/GB10 server in Docker
  using the release settings. The host needs no GPU or NVIDIA driver;
  execution checks run on the Spark. See the [cross-compiling guide](docs/cross-compiling.md)
  and [build and target validation record](benchmarks/results/2026-09-19-spark-cross-build.md).
- **Decode graph batch counters on `/metrics` and `/v1/metrics`**
  (2026-09-20, [#8](https://github.com/HawkBearPig/dgpp/pull/8)):
  `scheduler.decode_batch` reports cumulative graph launches, verification
  rows, padded rows and a launch histogram by graph capacity, plus the
  last launch's capacity, active requests and verification width. See the
  [counter contract](docs/operations.md#decode-graph-batch-counters) and
  [validation record](benchmarks/results/2026-09-19-decode-batch-telemetry.md).
- **MTP acceptance counters on `/metrics` and `/v1/metrics`**
  (2026-09-20, [#9](https://github.com/HawkBearPig/dgpp/pull/9)):
  `scheduler.spec_decode` reports verification rounds and total and
  per-position attempted/accepted drafts. Acceptance is counted after
  exact host fallback resolves, including drafts a sampled fallback
  accepts after provisional device rejection; counts precede response
  stop/length trimming. See the [counter contract](docs/openai-compatibility.md#speculative-decoding-counters)
  and [validation record](benchmarks/results/2026-09-19-mtp-acceptance-metrics.md).
- **Lossless 12-bit weights on Qwen3.8-Flash-Next; format v2** (2026-09-19;
  round four of `benchmarks/results/2026-09-19-glm-flash-line-rate/`). The
  format no longer needs rows of whole 1024-column super-blocks: the columns
  left over follow the row in units cut by the 256-column steps they hold,
  twelve bits a weight without padding (`kernels/bf12_gemv.cuh`; k = 2560,
  1536, 320, any multiple of eight), and a row without a tail is laid out as
  before. The row chain moved to that shared device header, the multi-problem
  launch got its packed twin (`launch_bf12_gemv_multi`: the GDN's four input
  projections) and layers find companions through `IGemm::bf12_lookup`. Qwen
  packs its GDN and QSA projections, the draft block's and the head (81 % of
  what a world-4 rank reads per token is BF16; 1.65 → 1.24 GiB per rank):
  same-binary fabric A/B on the FP8 checkpoint, transcripts identical —
  world 4 +6.4 / +3.8 / +2.4 / +1.3 % at one to four live requests, world 2
  +9.0 / +6.7 / +6.6 / +4.6 %, prefill unchanged. The GR sites, routers and
  shared experts stay BF16: they are read warm behind the prefetcher, where
  the packed chain's extra integer work loses (microbenched: +56 to +109 %
  from L2, no gain cold at those shapes). Qwen keeps both forms resident
  under either value of `engine.bf16_weights` (every recipe has the room);
  under `dense_weights: "fp8"` nothing is packed. Qwen templates:
  `"bf12+bf16"`. Also: a companion enters an L2 prefetch window through
  `WeightPrefetcher::add_isolated` — the coalescing add bridges holes of up to
  2 MB between adds, which between two allocations can be a released
  (unmapped) BF16 range; no fault was ever seen, the hazard was latent under
  `"bf12"` on the full GLM-5.3's small indexer matrices.
- **bf12-only residency: `engine.bf16_weights: "bf12"` now SAVES memory, and
  the two ceiling-bound templates have their contexts back** (2026-09-19;
  round three of `benchmarks/results/2026-09-19-glm-flash-line-rate/`). The
  12-bit form can now be the only resident one: each packable BF16 matrix
  loads into its own releasable range (the CUDA virtual-memory API —
  `loaders/releasable_range.*`, `LayerBump::alloc_side`), is packed as its
  layer lands, and its BF16 bytes go back to the node at once while the
  address stays reserved as the key (a stray read is a clean fault). The
  staging mirror, byte formula and resident image keep their layout — an
  image written in one mode restores in another, no rebuild. A prefill GEMM
  expands the rows it reads into a small scratch (`launch_bf12_expand`, at
  the device memcpy's rate) and runs the algorithm it always ran: whole
  matrices for wide chunks (two slots: the fold overlap's row blocks reuse an
  expansion), 8 MiB weight-row blocks that stay in the L2 for calls of up to
  256 rows, weight-row blocks for the lm head — bitwise the unsplit call
  under the pinned algorithm. Decode batches past eight rows take eight-row
  packed launches, so a captured graph never touches the scratch. The key now
  has three values: `"checkpoint"`, `"bf12"` (the 12-bit form alone: 0.5–1.4
  GiB per rank UNDER the BF16 plan; about 10 ms per prefill chunk on
  four-node GLM-5.3-Flash — +1–2 % at 8K–32K, +25–40 ms to a short prompt's
  first token — within 1 % on the full GLM-5.3) and `"bf12+bf16"` (both forms
  resident: the previous behaviour, no prefill cost). Transcripts identical
  to the both-forms build on GLM-5.3-Flash (four and two nodes), GLM-4.7 and
  the full GLM-5.3; decode level at every concurrency, the full GLM-5.3's
  sixteen-row batches included. Templates: two-node GLM-5.3-Flash `"bf12"` at
  160K again (4.8 GiB of the node left at boot, where both forms left 0.05),
  the full GLM-5.3 `"bf12"` at 120K again (both forms were refused there);
  four-node GLM-5.3-Flash and GLM-4.7 `"bf12+bf16"`. `DGPP_BF12=off|on|both`
  overrides for A/B runs.
- **Decode collective: the interleaved gate, −5 us per collective**
  (2026-09-19, same record). The graph all-reduce kernel gates a round's
  co-claimed peers in one interleaved pass, reads the rest of a claimed
  doorbell ({len, ctl}, {hash}: one cache line) in one round trip behind its
  `seq` acquire instead of four or five dependent system loads, looks for the
  engine's poison every eighth round, and ends the wait on the last claim.
  Rank 0's collective on the four-node GLM-5.3-Flash step: 40.9 → 35.7 us
  (−0.5 ms of a 32.5 ms step); single-stream decode +0.9–1.5 %, two to four
  live requests level (their 32–64 KiB rows are past the staging budget, and
  the unstaged fold gives the rounds' gain back). Results bitwise (the
  canonical fold), transcripts identical, seven-minute soak clean;
  `DGPP_BUS_GATE=sequential` keeps the per-peer gates.
- **Lossless 12-bit BF16 decode weights, `engine.bf16_weights: "bf12"`**
  (2026-09-19; `benchmarks/results/2026-09-19-glm-flash-line-rate/`): a second
  resident form of the BF16 matrices the decode GEMV streams — the
  sign+mantissa byte plus a 4-bit exponent code against a per-row window,
  exact side tables for the 1.5e-4 of weights outside it, outlier rows kept
  BF16 (`src/kernels/bf12_gemv.*`, `bf12_companions.*`). Bitwise the BF16 GEMV
  at one to eight rows, 0.75 of the bytes: served transcripts are identical.
  Same-binary fabric A/B: GLM-5.3-Flash decode +6.0 / +4.3 / +6.4 / +6.1 % at
  one to four live requests (+8.5 % without MTP), GLM-4.7 +11–12 % single
  stream and +5–7 % at four, the full GLM-5.3 +5–6.5 % and +3–7 %. Off by
  default in the binary; every deployment template enables it. The companions
  sit beside the BF16 bytes (prefill and wider batches keep those): +1.9 GiB
  per rank on the four-node Flash recipe, +4.8 on GLM-4.7; the two templates
  sized to their nodes' ceiling pay from the context pool (two-node Flash
  160K → 132K, full GLM-5.3 120K → 100K; `--bf16-weights checkpoint
  --kv-capacity …` restores either). Qwen and DeepSeek accept the key and
  pack nothing yet. Peers now also apply the head's `embed_sharding`.
- **GLM-5.3-Flash prefill −10 to −12 %** (2026-09-19): the draft block's
  prefill rows stop once their DSA cache state is written (nothing read
  their output; the MoE there ran the host-segmented path: −6.0 / −7.3 /
  −7.6 % at ~2K / ~8K / ~32K, transcripts, passes and acceptance identical;
  `DGPP_MTP_PREFILL_FULL=1` restores), and chunks of 1024 rows or more run
  each KDA attention site in two row blocks so block A's bulk fold flies
  beside block B's compute and the FFN fold's second half beside the next
  layer's first (`BoundaryReducer::begin_async`, `CublasLtGemm::set_plan_rows`:
  bitwise the unsplit walk, a further −5.2 / −4.7 / −3.9 %;
  `DGPP_PREFILL_OVERLAP=off` restores). Cold service prefill now 1.30 / 5.26 /
  24.5 s.
- **Decode collectives claim every ready peer per round** (2026-09-19): the
  timeline's new `graph window claims:` line showed the gaps between a
  generation's consecutive claims pinned at 7–10 us — the graph kernel gated
  co-resident peers one scan round at a time, and two of the three rounds
  were being read as peer skew. One election per peer per round: 43.8 → 41.6
  us per collective, results bitwise. `scripts/bus_window_skew.py` reports the
  gaps; `benchmarks/micro/uar_probe.cpp` records that the GB10 maps an mlx5
  doorbell page for device access.
- **Guided source setup** (2026-09-19): add `scripts/setup.sh` for deployment
  selection, site and RoCE configuration, dependency checks on every node,
  optional system-package installation, release build, checkpoint download/sync
  and final serving preflight. Preserve local tuning and credentials on reruns;
  support unattended setup, read-only prerequisite checks and offline cache reuse.
  `setup.py` turns an interpreter older than 3.10 away itself, before its
  project imports (the wrapper holds no inline Python, per the scripts' lint);
  the SSH reachability probes never read the terminal, so type-ahead and piped
  answers reach the prompts; and the preflight that `dgpp-cluster up` runs
  requires write access to the checkpoint cache only while preparing — a
  read-only model store still starts. Validated on the four-node cluster in
  read-only, configure-only, unattended (with and without the build) and
  interactive modes, against throwaway site files.
- **Streaming images and GLM prefill scheduling** (2026-09-19): remove
  history-wide image count/token caps. Stage visual embeddings through fixed
  single-image and chunk buffers, including MTP lookahead and cached suffixes.
  Bound decoded request pixels and retained prefix identities by bytes.
  Enable GLM-5.3-Flash graph prefill continuations: active decodes run between
  bounded chunks, with cancellation and prefix snapshots preserved. The
  four-rank deployment uses 256-token busy and 2,048-token idle budgets.
  Scan incoming journal frames incrementally so large pixel payloads do not
  delay admission with repeated scans of the entire buffered prefix.
- **Image-aware prefix caching** (2026-09-18): reuse GLM image prompts and
  generated continuations across turns instead of reprocessing the entire
  conversation. Compare pixels, geometry and token positions exactly, account
  for MTP lookahead, and restore uncached image embeddings during suffix prefill.
- **GLM-5.3-Flash image inputs** (2026-09-18): accept PNG/JPEG data URIs
  through Chat Completions, journal processed pixels across ranks, and run
  the native BF16 vision encoder before prompt prefill, including MTP.
  Report model input modalities and visual prompt-token usage. Add encoder reference checks, API tests
  and [usage documentation](docs/vision.md). Match the CUDA BF16 eager reference
  bitwise across the pinned regression corpus after correcting rounding and
  reduction arithmetic; see the [numerical investigation](benchmarks/results/2026-09-18-glm-vision-numerics.md).
- **Release packaging with bundled PCRE2** (2026-09-18): exclude the dependency's
  standalone install rules from DGPP's runtime package. Server-only builds
  no longer fail packaging because the unused `libpcre2-posix.a` was never
  built. Add a host-only build/install regression test using the actual
  dependency configuration.
- **Chat extension edge-case validation** (2026-09-18): cover malformed file
  inputs, storage quota races, multipart boundaries, custom-tool streaming
  and history, native reasoning mappings, schema bounds/references/formats,
  and malformed Lark grammars. Fix escaped file detection, corrupt metadata
  handling, array-bound overflow, missing pointer array traversal, impossible
  scalar enums, Lark escapes/cycles, leap-second validation, custom error
  fields/history IDs and duplicate thinking globals. Record reproducible
  checks and remaining limits in `docs/openai-api-validation.md`.
- **Document inputs and richer Chat tools** (2026-09-18): add inline and
  uploaded file inputs, bounded asynchronous PDF/text extraction and a
  configurable persistent file store. Add custom tools with text, regex and
  Lark formats; map reasoning effort through each model's native controls.
  Enforce local/recursive JSON Schema references, string patterns and formats,
  and exact decimal multiples during constrained decoding. Document limits,
  model mappings and the rank-0 `poppler-utils` dependency in
  `docs/openai-compatibility.md`.
- **OpenCode provider-option compatibility** (2026-09-18): restore acceptance
  of unknown top-level client extensions, including `preserveThinking`.
  The API audit's strict allowlist incorrectly rejected previously working
  requests. Supported options remain validated and known unsupported API
  features still produce explicit errors. Add streaming and one-shot coverage.
- **Chat API audit and live prefill metrics** (2026-09-18): align nullable
  options, developer messages, numeric bounds, streamed usage and obfuscation
  with the OpenAI reference; validate unsupported capabilities explicitly.
  Correct visible-content logprobs, transport error envelopes and HTTP request
  counters for multiple choices. Publish per-request prefill progress while a
  scheduler pass is still running. Document the supported text/tool profile
  and the DGPP metrics extension in `docs/openai-compatibility.md`.
- **Release and testing builds** (2026-09-18): the release preset builds the
  production server with explicit `-O3`, no debug information, and link-time
  stripping. Deployment defaults to `build-release/dgpp-serve`; testing keeps
  `build-ci/` and debug symbols, selectable with `--bin`. Release packaging
  rejects testing build directories and debug/sanitizer artifacts.
- **Configurable HTTP body limit** (2026-09-18): `http.max_body_bytes` in
  deployment JSON, or `--http-max-body-bytes`, controls the serialized request
  limit. The default rises from 4 MiB to 256 MiB for large document prefills
  and agent histories. HTTP 413 responses name the configured byte cap;
  this limit is independent of KV capacity and does not preallocate buffers.
- **Metrics endpoint alias** (2026-09-18): `/metrics` exposes the same JSON
  counters as `/v1/metrics`, which remains available for backward compatibility.
  Prometheus exposition is still a separate follow-up.
- **Agent API compatibility** (2026-09-17): JSON Schema numeric bounds now
  constrain `number` values, including fractions and exponents. Tools can
  accompany `response_format`: auto chooses a tool-call turn or a structured
  answer, including after tool results; required/named/none and parallel
  call controls retain their guarantees. `reasoning_effort: xhigh` is accepted
  through both API spellings and passed to the checkpoint template. Qwen
  renders low/medium directly and high/xhigh as xhigh.
- **Qwen wide batches and opt-in prefill continuation** (2026-09-15):
  sixteen decode rows support eight native-MTP requests. Small-row fused
  kernels and deployment defaults are preserved. Fitting physical slot
  prefixes remain batchable when a deeper full batch exceeds the model's
  row limit. `engine.prefill_budget_tokens` gives Qwen graph decoding a
  turn between bounded prefill chunks; the cursor retains target/draft
  state, reservations and snapshot ownership through cancellation and
  cache attach. The policy is journaled across ranks. Wide draft hidden
  projections use a tensor-core kernel to avoid cuBLASLt copy-engine nodes
  in collective graphs. Benchmark scripts now distinguish request-wall
  and legacy output-span rates, require server token usage, and provide
  repeated C1 comparison and long-prompt interference checks. See the
  [validation and measurements](benchmarks/results/2026-09-15-qwen-batching-prefill.md)
  for configuration tradeoffs, the unresolved C1 promotion gate, and remaining work.
- **One deployment template per model, quant and world** (2026-09-14, late;
  `deploy/README.md`): twenty-five templates became eight —
  `cluster_<model>_<quant>_w<n>.example.json`, each with MTP at the depth its
  family measured best, the decode graph, and the slot count and cache
  budget that measured at or above the other shapes (the six-slot DeepSeek
  shape, the eight-slot full GLM-5.3, the large-cache GLM-5.3-Flash on four
  nodes, the FP8 dense stack on the single-Spark Qwen). The retired shapes
  are knobs (`--no-mtp` is new; `--mtp-depth`, `--max-concurrency`,
  `--kv-capacity`, `--kv-dtype`, `--prefix-cache-gib`, `--dense-weights`
  were there) and the catalogue maps every retired name to its knobs. The
  default deployment (`scripts/site_env.py`) and the resolved-config fixture
  follow the four-node GLM-5.3-Flash template; historical changelog entries
  and benchmark records keep the old names.
- **The GPU-driven eager fold (plan D9)** (2026-09-14, late;
  `CollectiveBus::allreduce_stream` / `allreduce_settle`, `BusStreamReducer`):
  the eager walk's boundary reductions launch on the model's stream in the
  graph kernel form — no host drain of the model stream before each fold,
  no engine launch on the collective stream, no host notice of the finish
  before the model continues; the generation comes from the shared
  counter at submit, the engine posts from a FIFO through the replay
  walk's own flight, the model settles once per pass. DeepSeek-V4.1-Flash
  serves on it by default (`DGPP_DSV41_EAGER_FOLD=1` restores the
  host-driven reducer; the other families keep it). The fold is bitwise
  the host-driven one (bus_test's stream scenario against the oracle at
  worlds 2 and 4; dsv41_tp_test's group prefill through both reducers,
  every layer bitwise). Fabric, the six-slot config: TTFT 0.30 → 0.27 s at
  one stream and 0.84 → 0.73 s at six, the aggregates unchanged, the fold
  kernel 542 → 386 µs at 47 rows (docs/measurements.md).
- **The dense lowering of the session-core families follows the rows of a
  launch** (2026-09-14, `kernels/gemm.hpp dense_gemv_rows`, the site setting
  `DGPP_DENSE_GEMV_ROWS`, default 4; 256 restores the old lowering): GLM-4.7,
  the full GLM-5.3, Qwen3.8-Flash-Next and GLM-5.3-Flash lower their dense
  sites by row count — the row-independent GEMV chunks (and Qwen's fused
  multi-problem launches) to four rows, cuBLASLt's algorithm for bf16 rows
  above (at the weight-stream floor from six rows: the chunks re-read the
  weights per four rows, 2–8× the bytes at 6–32 rows: `bf16_gemv_test`'s
  table), the streaming tensor-core GEMM (`kernels/mma_gemv`, now with an
  asynchronous per-warp weight ring and a block width chosen by n) for fp8
  rows from five to 256 (the full GLM-5.3 DSA projections at sixteen rows:
  o_proj 320 → 109 µs, q_a 155 → 59, q_b 104 → 38; Qwen's FP8 dense stack
  and the shared expert; the Flash dense MLP) and the 128-row dense kernel
  above. T = 1 keeps its chain; a batched step's rows and the eager
  scalar's are tolerance-equal, not bitwise (the engine gates hold the
  first decisions exactly unless the flipped decision was a near tie of the
  world-1 reference — `tests/cuda/engine_test_ties.hpp` — and report the
  rest). DeepSeek-V4.1-Flash keeps its every-row tensor-core lowering.
  Fabric A/B: docs/measurements.md.
- **Group prefill for GLM-4.7, the full GLM-5.3, Qwen3.8-Flash-Next and
  GLM-5.3-Flash** (2026-09-14): the families expose
  `prefill_group_span_limit()`, so the scheduler admits queued cold prompts
  together and prefills them as the spans of one walk — GLM-4.7's attention
  rows carry their own request ids and positions already; the full
  GLM-5.3's DSA layer runs per span with a selection-scratch base per span
  (`DsaLayer::enqueue_prefill(row_base)`, the selection-reuse contract per
  base); Qwen's GDN scan and QSA attention run per span; GLM-5.3-Flash's
  own walk (`GlmDiagnosticModel::session_prefill_group`) runs its KDA scan
  and DSA attention per span and mirrors every span's last row (a group
  of at most kDecodeRows spans). Gates: a group's rows against the
  prefills alone (GLM-4.7 and Qwen bitwise on their fixtures; the DSA
  family's rows within 6.5e-3 relative l2 with the top-1 equal; the Flash
  fixture's within 2e-7, its decode off the group's cache bitwise the solo
  decode) and a decode off the group's cache against the re-forward.

- **Streaming tensor-core decode GEMM for DeepSeek-V4.1-Flash** (2026-09-14,
  `kernels/mma_gemv.{hpp,cu}`; default on for this family, `DGPP_DSV41_DENSE_GEMV=1`
  restores the chunks): the dense fp8 projections, the engram wkv, the
  draft's `main_proj` and the bf16 lm head at 1–32 decode rows run one
  launch that streams every weight once through `mma.sync` (the 4-row GEMV
  chunks re-read the weights per chunk: 82 ms of a 252 ms six-slot step).
  The dequantized values are exact for the e8m0 scales, so only the fp32
  order differs from the chunks (transcripts reorder, accuracy held), and a
  row's chain never depends on the rows sharing its launch (a batched row
  stays bitwise the row alone). Cold, fp8 [5120 × 5120]: 132/133/136/142 µs
  at 1/6/16/30 rows against 118/205/419/764; the prefill's rows take the
  same kernel in 64- and 128-row forms and 128-row groups above (192 / 288
  / 567 / 4,558 µs at 64 / 128 / 256 / 2,048 rows against the chunks' 1,507
  / 2,978 and the tile kernel's 3,533 / 23,360). Fabric: docs/measurements.md.
  The fixture gates that certify selection flips now read the coded index
  query of both paths (`Dsv41Model::IndexLogits::q_codes`): a flip whose
  e4m3 codes differ between the paths is the coding's discontinuity,
  whatever the reference gap.
- **Group prefill** (2026-09-14, `SchedulerEngine::prefill_group`,
  `SessionModel::session_prefill_group`, `Scheduler::admissible_group`):
  queued cold prompts (no prefix-cache attach or snapshot; any width up
  to a walk's rows) admit together and prefill as the spans of ONE forward
  — the dense sites, the MoE, the norms, the Engram and the head over
  every row, the CSA2 attention and kv publication per span, and under the
  bounded prefill each span's decoder segment (its last window rows)
  packed span after span — instead of one read-in per tick with a decode
  step between.
  A prompt prefilled in a group is bitwise the prompt alone (gates at the
  model, the TP world and the engine): the mHC dots take their tiled form
  at every prefill row count for this family (`mhc_set_tile_min_tokens`),
  and an opted-in cuBLASLt instance lowers every bf16 row count to the
  tensor-core form. Other families keep one prefill per tick.
- **Adaptive λ for the scheduled verify depth** (2026-09-14,
  `engine.mtp_schedule_adapt`, on by default with the schedule): λ follows
  an EWMA of each MTP step's committed tokens over its modeled time
  (base + rows·row), floored at the configured λ — replicated inputs, one
  fixed arithmetic, every rank the same value. One configuration now
  serves every concurrency: C1 48.2 / C2 66.8 / C6 87.0 on the six-slot
  world (docs/measurements.md).
- **The batched replay's verify depth is the batch's own Dinkelbach rule**
  (2026-09-14, `scheduled_verify_depth_batch`): one more draft position
  costs a verify row per live slot and yields the sum of the slots' prefix
  survivals, so the batch verifies position i while the MEAN survival over
  its slots beats λ·row — where the earlier rule took the deepest slot's
  own depth, verifying rows for every slot that one confident slot alone
  justified. Exact at any depth (each slot commits its greedy prefix);
  the test hook still forces per slot. Measured on the six-slot world:
  docs/measurements.md.
- **MXFP4 form of the fp4 tensor-core expert kernel** (2026-09-14,
  `moe_grouped_mma_fp4_ldm_kernel<OutT, kGroup>`, `fp4_group` on the grouped
  launchers): DeepSeek-V4.1-Flash's routed experts (e2m1 + e8m0 per 32, no
  global) prefill through the ldmatrix tile kernel instead of the 4-row
  GEMV core (the weights re-read per 4 rows: a third of a prompt's kernel
  time). The pair is decoded in fp32 — the e8m0 scale can leave f16's range
  — and is exact; `glm_moe_test` pins the grouped launch to a host oracle
  (gate within a bf16 rounding, down within 4e-5) and the z split bitwise.
- **Confidence-scheduled verify depth for DSpark** (2026-09-14,
  `engine.mtp_schedule`, off by default): a greedy request verifies only
  the leading drafts whose prefix-survival probability (the product of the
  confidence head's per-position acceptance probabilities) beats the value
  of a verify row, each depth on its own captured graph variant; the
  batched replay and sampled requests keep the whole block. The rule is
  the Dinkelbach optimum of aggregate tokens per second (a per-step ratio
  would be the wrong surrogate), its constants are the world's so every
  rank derives the same depth, and the committed transcript is the plain
  greedy one at every depth — proven on the eager speculator and on the
  graph engine over a forced, varied depth sequence
  (`tests/unit/verify_schedule_test.cpp`, `dsv41_decode_test` §7b,
  `dsv41_engine_test` on 29964). On the fabric (four nodes, greedy, 300
  tokens per class): chat 33.4 → 24.6, prose 27.4 → 22.7, code 21.9 → 18.9,
  json 21.9 → 19.1, math 21.8 → 19.8 ms/token with λ at the achieved
  throughput (0.045 tok/ms; the reservation-rate default is within 2–4 %),
  transcripts identical, op streams identical across the ranks. No change
  to any configuration that does not enable it: the captures and the
  replay path are the same (the baseline run reproduced the previous
  transcripts and timings). The DeepSeek-V4.1 deploy example enables it.
  Extended the same day to the batched replay (one depth per batch, the
  deepest a live slot asks for, on reduced-row batch variants over the
  compacted feeds) and to every MTP family: without a confidence head the
  engine takes the draft head's own probability of each draft from the
  device sampler (the draft picks report logprobs; the tokens are
  unchanged) — GLM-5.3-Flash, Qwen3.8-Flash-Next, GLM-4.7 and the full
  GLM-5.3 can schedule at `mtp_depth` 2 and above (`glm4_engine_test`,
  `glm_tp_test`). Measured: the batched DeepSeek replay at two live
  requests prose 34 → 51 and chat 37 → 54 tok/s aggregate (exact); the
  GLM families at depth 2 exact and throughput-neutral (one row at stake).
  A stale settle bound after a reduced-depth batch, found by the two-request
  probe, is fixed and gated.
- **DeepSeek-V4.1-Flash decode window attention split** (2026-09-14): an
  nsys profile found the decode window attention ran as a single
  latency-bound thread block (`n_split=1`), the largest non-GEMV cost in
  the step (~140 us a layer, the GEMVs themselves at 70-88 % of peak). The
  decode path now splits the window a fixed eight ways to fill the SMs
  (prefill stays unsplit and byte-identical): chat 38.3 to 34.3, code 25.3
  to 21.8 ms/token, gsm8k 60/60 unchanged, cross-rank determinism kept.
  Softmax attention cannot be split bitwise-invariantly, so decode's window
  is no longer bit-identical to the forward's block (about one bf16 ULP,
  which shifts greedy token sequences at near-ties without changing
  accuracy); the decode-vs-forward fixture checks were re-calibrated to a
  rounding budget plus an argmax-match assertion, the certified-flip decode
  audit unchanged.
- **DeepSeek-V4.1-Flash is a served family** (2026-09-14): the CED
  encoder/decoder with CSA2 attention (a 128-token sliding window plus a
  compressed-KV path selected by a two-level indexer), single-pass
  hyper-connections, the two Engram n-gram tables mapped from NVMe, and
  the DSpark block draft (five drafts verified per pass, `kSpecRows` 6)
  ride the shared loader, paged-block and session cores. The MXFP4/FP8
  checkpoint serves as shipped (the quantization study found nothing to
  gain beyond it). A bounded prefill (the model's own SWA-replay: the
  encoder over the whole prompt, the decoder over the last window) is the
  default, `engine.prefill` selects it or the exact 40-layer parity mode.
  The tokenizer is a three-stage pre-tokenizer; the prompt renderer
  follows the checkpoint's own encoder; tool calls use a constrained DSML
  grammar. Served on four nodes 2026-09-14: 76 ms/pass at 2.33 tokens/pass
  (32.5 ms/token), bounded prefill 1.6–2.4 ms/token, gsm8k 60/60,
  HumanEval 40/40, extract 30/30; the release's own layer code in fp32
  sits as far from the engine as from its own bf16 pipeline at every
  cross-checked layer. Prefix caching is whole-block, so a prompt shorter
  than 128 tokens is not cached yet (docs/deepseek_v41_flash_plan.md §8).
- **The launcher's port preflight binds as the servers do** (2026-09-13):
  the probe now sets `SO_REUSEADDR` like `tcp.cpp` and `http_server.cpp`,
  so a TIME_WAIT left by the previous world is not a conflict and a
  restart within a minute of a stop passes preflight (the failure drill's
  reboot had failed on the fabric and journal ports). The drill fetches
  rank 0's op stream from the deployment's log dir, where the launcher
  runs rank 0, instead of the repository root.
- **A served request leaves no history** (2026-09-13; reported by a
  third-party tester as GitHub issue #1): the scheduler kept every
  request's prompt, grammar, cache cuts and generated ids — and a second
  copy of the tokens in `results()` — for the process's lifetime, and
  every tick scanned the whole history; the op stream, too, sat in memory
  until the exit-time write. Measured against the built scheduler: 8 bytes
  per prompt token per request (1,000 requests of 16,384 tokens: 131 MB of
  RSS; the 2026-09-07 agent session's 85 requests: 21 MiB) and 37 bytes
  per generated token, all inside the 4 GiB headroom. A retired request
  now releases everything but its tombstone (id, status, counts); the
  service and the peers (`Scheduler::set_keep_retired(false)`) compact
  the tombstones and results away at the end of every tick — the same
  quantum on every rank, since the journal carries every tick — so after
  any tick, under any load, the scheduler holds exactly its live
  requests; `/v1/metrics` reports the
  records and ids held (`scheduler.records`, `record_tokens`, `terminal`
  now cumulative); and each rank's `serve_rank<N>.ops` is written as the
  run records it, flushed at every retire, so a killed rank keeps its
  evidence. Gates: `scheduler_test`'s release and compaction tests (the
  compaction moves no op under overlapping load; the live heap flat at
  +5 KiB from request 200 to request 40,000), `fabric_serve_test`'s
  file-mode observer.
- **The full GLM-5.3's decode batch: sixteen rows** (2026-09-13, plan D9):
  the family's cap was the fused decode select's eight rows of shared
  memory; the select now launches its rows in groups of eight (a group
  is the same work at any grouping), the pick kernels verdict sixteen
  request slots, the DSA layer sizes its attention tiles and (row, split)
  workspace from the count, and the absorb/vout projections keep their
  warp kernels for every decode shape (`kProjMmaMinRows` 16 → 32: the
  tensor-core forms are tolerance-equal, not bitwise, and a batched row
  must be bitwise the same row alone). Eight request slots at MTP depth 1,
  five at depth 2, four at depth 3; every recipe up to eight rows runs
  the launches it always did. Gates: `dsa_test`'s row groups (12 and 16
  rows bitwise the host mirror), `glm_dsa_engine_test`'s sixteen-row
  world (eight slots at depth 1, batched transcripts bitwise the eager
  engine's). Found on the way: the GEMM interface lowers bf16 calls up to
  the model's decode rows to the GEMV chain, so a sixteen-row deployment
  prefills 9–16-token prompts through it where an eight-row one used
  cuBLASLt (last-bit differences, deterministic within a deployment).
  The graph engine records 4- and 6-slot batch families for recipes wider
  than four slots (four live requests on an eight-slot world had replayed
  the sixteen-row every-slot family at 250–270 ms a step).
- **The bus fold at two and three decode rows** (2026-09-13): the graph
  all-reduce stages a payload in shared memory up to 80 KB (was 48 KB), so
  the full GLM-5.3's two-row depth-1 fold (6144 x 2 x 3 peers = 72 KB) no
  longer reads the NIC-placed rows twice, and both passes over peer memory
  keep several system loads in flight per thread. Bitwise the old fold;
  depth 1 66.9 ms/pass (from 68.0), depth 2 85.1 (from 88.2), T=1
  unchanged. The L2 weight-prefetch knobs (`DGPP_L2_PREFETCH=off`,
  `DGPP_L2_PREFETCH_MB`, `..._BOUNDARY`, `..._LAYER`) are site settings
  the launcher forwards to every rank, for A/Bs; the A/B on the full
  GLM-5.3 kept the prefetch on (off costs 4 ms a step at T=1 and 5 ms a
  pass at depth 1).
- **A bare `dgpp-cluster down` stops what is running** (2026-09-12):
  `down` and `status` without `--config` (or with `--all`) scan every
  deployment recorded under `DGPP_LOG_DIR/deployments` instead of the
  default deployment file's namespace; `status` lists the running ones
  with their ranks and the stopped ones as one line each with how their
  log ended, `down` stops the running ones and names them by model, world,
  namespace and deployment file (the staged namespace now records it in
  `deployment.path`). Before, a bare `down` looked only at the default
  file's namespace and reported success after stopping nothing while
  another deployment kept the ports, so the next `up` failed preflight.
  `down --config FILE` on a deployment with no live rank now says so.
- **The full GLM-5.3** (2026-09-12, `docs/glm53_plan.md`): a fourth model
  family, `glm_moe_dsa` (`GlmMoeDsaForCausalLM`, 78 layers, 754B), served
  from the int4/int8 group-64 pack-quantized release
  (`HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64`) at 99.3 GiB of weights per
  rank on four nodes. New in the engine: a packed-int GEMV core with the
  exact code x scale dequant behind the MoE slot and grouped paths, the
  DSA layer's decoupled interleaved RoPE (the rope key stored bf16 beside
  the latent in every cache format, the flash kernels at a 576-wide
  score), per-token selection with the relu'd indexer at select_k 2048,
  cross-layer selection sharing (21 indexers over 78 layers), a loader on
  the shared resident stream with the draft layer's experts requantized
  at load, the session-core model, the serving family, three deployment
  templates and the tokenizer / chat-template goldens (the template
  interpreter's `range` gained the one-argument form). Gates: the DSA
  layer against its host oracle at the full geometry, the fixture forward
  against a pure-python reference, decode / prefix / speculator, TP
  worlds 2 and 4, the graph engine at world 2, and the real layers 0–3
  against transformers' own layer code (per layer 0.002–0.0035 relative
  l2, the bf16 floor; MoE routing near-ties certified by margin).
  `tools/checkpoint_audit.py` learned the pack-quantized triple and writes
  `docs/checkpoint_budget_glm53.md` (99.30 GiB per rank at world 4, the
  memory plan's number). Served on the four nodes the same day
  (`scripts/fabric_glm_dsa_serve.sh`, `serve_mtp_classes.py`): T=1 51
  ms/step, MTP depth 1 68–76 ms/pass at 1.77–1.97 tokens/pass, gsm8k 59/60
  and HumanEval 40/40 with thinking on, MTP == T=1 transcripts; the
  templates carry 112K bf16 (plain) / 96K bf16 (MTP) / 160K fp8 latent
  caches. The prefill tile kernel for the packed formats is deferred
  (the GEMV chain serves prefill at 7–9.5 ms per prompt token).
- **`engine.embed_sharding: vocab`** (2026-09-13): the full GLM-5.3 can hold
  its lm-head slice of the embedding rows on each rank instead of the
  whole table — 1.33 GiB per rank back at world 4. A token's row is gathered
  by the rank that holds it and summed across ranks by one fold (a row plus
  zeros is exact in bf16), on the main path and again for the draft's
  lookup: worlds 2 and 4, eager and recorded, are bitwise the replicated
  worlds. The key rides the rank-0 settings record and the config digest;
  the other families keep their tables whole. Default `replicated`; the
  three GLM-5.3 templates set `vocab`.
- **The memory plan's headroom is 4 GiB, from 8** (2026-09-12/13): the growth
  after the plan check was measured on the full GLM-5.3 at world 4 (all
  four ranks sampled through a boot, a 32K prefill and four live requests)
  at a flat 5.5–6.0 GiB, 2.2 GiB of it the loader's pinned staging mirror.
  The shared resident stream's `release_sources()` now frees that mirror
  with the checkpoint mappings; the GLM-5.3 model materializes its stack in
  its constructor and releases there, before its caches exist and before
  any collective is in flight (Qwen and GLM-4.7 load lazily and keep their
  mirror: a release mid-forward waits on the bus's persistent kernels for a
  watchdog period), and every resident plan names the staging as an item. The residual after the check is 2.4–2.8
  GiB; 4 GiB keeps 1.2–1.8 of margin, held by a one-hour soak at the
  full GLM-5.3's 120K MTP shape (2,046 requests, none failed, memory flat
  on all four ranks, no allocation stall or direct reclaim anywhere, rank 0
  never under 2.7 GiB available). With the vocab-sharded embedding the
  full GLM-5.3 templates carry 144K bf16 plain / 120K bf16 MTP / 208K fp8.
  `serve_soak_run.sh` now honors a preset `DGPP_SERVE_KNOBS` (a soak at a
  deployment's real shape) and counts stalled collectives only inside the
  run (the peers' logs accumulate across boots of one deployment);
  `serve_run.sh` passes the selected deployment as `--config`, without
  which the launcher refused to stop a `--log-dir` world and the soak's
  teardown had been leaving its world serving.
- **`cluster_glm-5.3_int4-int8_w4_mtp2`** (2026-09-13): MTP depth 2 for the
  full GLM-5.3 at two request slots (the eight-row decode bound). Measured:
  86–89 ms/pass at 2.15–2.69 tokens/pass, a few percent per token either
  way by class (code and JSON gain, chat loses); depth 1 stays the default.
- **The memory ledger** (2026-09-13): one INFO line per boot phase (after
  the plan check, the bus, the loader, the globals, the resident stack, the
  caches, the model, the graph engine, the warm capture, at listening)
  with the process's resident split and the device's free memory, and one
  per graph variant with its executable's device memory. What it found on
  the full GLM-5.3: the executables are 0.39 GiB for fourteen variants
  (7–15 KB per node), the process grows 0.7 GiB on the host, and the rest
  of the 2.7 GiB the plan does not name — about 2 GiB — is CUDA's own
  (cuBLASLt's kernels and the context's growth at first launches), not a
  plan item that can shrink. The headroom stays 4 GiB.
- Setup now covers CUDA discovery outside `PATH`, head/peer dependencies,
  outbound connectivity, verified SSH login, offline preparation and storage
  budgeting. Empty explicit `--config` arguments are rejected.
- Doctor distinguishes failed probes from zero RoCE lanes and points checkpoint
  recovery to rank 0. RoCE discovery retains partial results and displays the
  selected lane order. Its `--json` output now contains `inventories`, `selections`
  and `errors` objects keyed by host, replacing the bare host-to-inventory map.

- **GLM-5.3-Flash on two nodes** (2026-09-12): the NVFP4/FP8 hybrid fits a
  two-rank world at 94.71 GiB of weights per rank, against 50.74 at world 4.
  `deploy/cluster_glm-5.3-flash_nvfp4-fp8_w2_mtp1.example.json` serves it with
  four request slots, MTP depth 1 and an FP8 latent cache at a 163,840-token
  context; the `_large-cache` variant trades two of those slots for 262,144
  tokens, because the draft block's per-position hidden cache costs 8 KiB per
  token per slot and is 83 % of what a context token costs. No engine change
  was needed: the TP slicing, the loader and the bus were already world-generic
  and the geometry divides by two. Measured (benchmarks.md §3 to §7): T=1
  45.68 ms/pass, greedy MTP 59.40 ms/pass at 1.734 tokens per pass for
  34.25 ms/token, per-class acceptance matching the four-node run, aggregate
  22.7 to 47.3 tokens/s across the class and concurrency matrix, procedure
  prefill 664 / 1,829 / 8,280 ms at 512 / 2,048 / 8,192, and HumanEval 158/164,
  GSM8K 291/300, schema extraction 100/100 on the four-node denominators. Decode
  costs 1.75x the four-node pace, prefill about 1.45x, quality nothing. Both
  ranks' op streams matched at shutdown and the MTP and T=1 runs produced the
  same rank-consistency digest.
- `serve_soak_run.sh`, `serve_stop_check.sh` and `serve_failure_drill.sh` take
  their rank count from the selected deployment instead of assuming four nodes,
  and refuse a single-node deployment by name (`dgpp_require_peers`). The
  drill's victim argument now accepts any rank of the configured world.
- Deployment filenames now identify the full model, quant, world size and
  decode mode. Settings are unchanged; see `deploy/README.md` for the rename
  table. Older entries below retain the filenames used at the time.

- **Qwen3.8-Flash-Next on a single Spark, and its dense stack in block FP8**
  (2026-09-10, docs/qwen38_single_spark.md): the NVFP4 checkpoint's experts
  through the shared fp4 core, the 47.7 GiB n-gram table left on the NVMe
  behind a memory mapping (`engine.ngram_table`, default resident) with each
  walk's rows gathered by a host node forked inside the walk, and the resident
  graph engine at world 1 over a world-of-one bus — 47 ms/token plain,
  31–38 ms/token with MTP (59 ms per pass), prefill 1.0–1.5 ms/token. Then the
  dense projections encoded as they load with the FP8 releases' own recipe
  (`engine.dense_weights = "fp8"`, default `"checkpoint"`; amax / 448 per
  128 x 128 block, round-to-nearest-even e4m3), read by the fp8 GEMV core at
  decode rows and dequantized into a BF16 bridge for the GEMM interface: T=1
  47 -> 31 ms per pass, MTP 59 -> 40 ms per pass (21–26 ms/token, 38–48 t/s),
  four in flight 122 -> 71 ms per step. Then the fused decode forms and a
  multi-problem fp8 GEMV — the GR site's norm-staged down GEMV with the inject
  rows riding it, the act-staged up GEMV, the batched down + inject, the shared
  expert's two-launch tail, and the GDN qkv + z / QSA q, k, v, indexer
  projections as one launch — each the BF16 fused kernel templated on the rows'
  form and bitwise the unfused fp8 chain: T=1 30.2–30.9 ms per pass, MTP
  39–40 ms per pass. Eval on the FP8 world: GSM8K 59/60, HumanEval 38–39/40,
  schema extraction 30/30 (the fabric's numbers); MTP transcripts identical to
  T=1. The world-1 profile prices every class against its byte budget (the
  experts and the multi-problem GEMVs at line rate, the draft head's second
  lm_head 2.5 ms a pass, the GR down GEMV at 55 %, ~2 ms of gaps), and the L2
  prefetcher stays on at world 1 (off costs 2 ms a T=1 pass). MTP depth 2 in
  this world: 48 ms a pass at 1.77–2.74 tokens, code/math/JSON 10–15 % faster
  per token, prose 5 % slower — depth 1 stays the default.

- **The MTP draft is drawn from the draft head and verified with the ratio
  rule, at any depth** (2026-09-10): the draft pick is a sampling pick on its
  own stream (`seed ^ kSampleDraftSeedMix`, the request's counter — the
  request's own draw accounting untouched), its final set travels as
  `DraftProposal` to the next step's verify, whose row 0 accepts with
  min(1, P/Q) and resamples the (P - Q)+ residual, so the emitted marginal is
  exactly P as before (`sampler_test`: marginal == P, acceptance ==
  1 - TV(P, Q); `glm_pick_test`: the device decision, token and draw count
  bitwise the host oracle's). A re-drafted row, an over-wide set or an
  unmatched token falls back to the plain rule, exact for any draft;
  `DGPP_SPEC_PROPOSAL=off` restores the argmax draft. Fabric world 4 at the
  sampled default: acceptance 41–51 % -> 60–68 %, 1.41–1.51 -> 1.60–1.69
  tokens per pass, 17.0–18.3 -> 15.3–16.1 ms/token at the same 25.8 ms/pass;
  greedy transcripts and the concurrency curve unchanged. The depth >= 2 chain
  picks are proposals too (each draft index its own key, so one step's drafts
  never share a draw; verify tests row t against proposal slot t; the host's
  re-drafts clear every slot): p2 30–34 -> 36–37 % at world 4 and 24–39 ->
  31–36 % at world 2.

- **Qwen3.8-Flash-Next takes the draft chain** (2026-09-10): `kDraftChain` and
  `kBatchedDraftChain` for the family the depth-2 entry below had excluded —
  the chain rows run the draft block forward past the first draft, and its QSA
  ring is copied aside before the first chain row and restored after the last,
  into a buffer of its own so the draft snapshot the fallback's rollback
  restores keeps the pre-draft ring. `qwen_engine_test` gates depth 2 on the
  loopback fixture: the greedy transcripts are the plain engine's, scalar and
  batched, on both ranks. Measured break-even on the fabric, so depth 1 stays
  the default — world 4 30.8 ms/pass at 1.93–2.03 tokens = 15.2–15.9 ms/token
  against depth 1's 25.9 ms/pass at 1.63–1.64 = 15.8–15.9; world 2 49.4 ms/pass
  at 1.81–2.08 against 40.4–41.0 at 1.63–1.73. The engine test's fixture
  directory joins the ignore list.

- **The GR inject dots leave the decode chain, and four levers measured and
  left behind knobs** (2026-09-10, docs/qwen38_optimization_plan.md): the
  gates depend on Rn alone, so at one row the hc inject rows ride the mix's
  down GEMV as the row space's last rows over the staged Rn (the same
  lane-strided chain as `combine_dots_kernel`, bitwise; `qwen_gr_test` pins the
  folded gates against the standalone kernel), and at more rows the same kernel
  runs on a low-priority side stream forked at the mix and joined before the
  apply — what was a 21 us one-block kernel between the boundary collective and
  the apply, 96 sites per step. Fabric world 4: T=1 22.12 -> 21.4–21.5 ms/step,
  MTP 26.5 -> 25.7 ms/pass, four live requests 114.6 -> 120.0 tok/s; world 2
  after both decode changes: T=1 31.38 ms/step, MTP 40.4 ms/pass at 23.7–25.2
  ms/token. Kept and off by default, each with its reading: the draft's
  temperature as a scale of the request's (`DGPP_SPEC_PROPOSAL_TEMP` — exact at
  any value, acceptance flat from 0.7 to 1.2, stays at 1); the small-k form of
  the fp8 expert tile kernel for the down projection (`DGPP_MOE_FP8_SMALLK`);
  the grouped fp8 expert kernel's m-sweep (`DGPP_MOE_FP8_SWEEP` — 5.87 vs
  6.87 ms per gate launch at 228 rows per expert, but 3.6 % on an 8K prefill
  against the byte model's 14 %: at four m-tiles it is issue-bound at 47
  TFLOP/s, so the chunk stays 2,048); and the prefetch-instruction form of the
  L2 prefetcher (`DGPP_L2_PREFETCH_FORM=prefetch` — 22.4–23.7 ms per T=1 step
  against the load form's 21.5, whose fold is what bounds the bytes in flight
  to what the collectives tolerate).

- **GLM-4.7 (`nvidia/GLM-4.7-NVFP4`) served** (2026-09-10, docs/glm47_plan.md):
  the Glm4MoeForCausalLM family — 92 pre-norm layers of biased GQA
  attention (96/8 heads, per-head q/k norms, half-split partial RoPE),
  three dense NVFP4 MLP layers then 160-expert sigmoid-routed MoE layers
  with an NVFP4 shared expert, an MTP draft layer whose BF16 experts are
  requantized to NVFP4 at load — on the same session core, engines,
  scheduler and service as GLM-5.3 and Qwen3.8-Flash-Next. New: the
  modelopt NVFP4 binding and loader (`src/models/glm4/`), the fp4 GEMV
  core generalized to any K multiple of 32 (power-of-two shapes bitwise
  unchanged), the NVFP4 shared expert as view-table entry E of the routed
  launches, paged split-KV GQA attention kernels with the fused bias /
  norm / RoPE / K-V append (`src/kernels/glm4_attn.cu`, bitwise a
  tile-aware host reference), `Glm4Model` with the paged K/V pool (no
  recurrent state: rollback positional, snapshots at any position, the
  draft's hidden window), the GLM-4.7 tokenizer (a Sequence[ByteLevel]
  post-processor), the template's `rstrip`/`lstrip` with a character
  argument, `deploy/cluster_glm47.json`, `scripts/fabric_glm4_serve.sh`;
  The partial RoPE is transformers' half-split rotate_half (pairs i, i+32
  over the first 64 dims), not the interleaved form of the older glm/glm4
  architectures: the interleaved reading passed every fixture gate (the
  python reference shared it) and left the real model coherent for ~30
  tokens before looping; found against transformers' own layer code on
  the real weights (`tools/glm4_torch_reference.py`, layer-0 relative l2
  0.036 -> 0.0025 after the fix). The bus recorder's collective-node budget per graph 128 -> 256
  (`kBusMaxGraphGens`: GLM-4.7's decode step records 186 — its first
  fabric boot refused the second graph variant at 128). The draft block
  takes the model's POST-final-norm hidden (vLLM's `glm4_moe_mtp`
  convention; the pre-norm residual gave 11–46 % draft acceptance on the
  fabric, the post-norm hidden 79–92 %).

- **The decode rows are the recipe's shape, and the batched depth-2 chain**
  (2026-09-10): the fixed decode batch's row ceiling is derived at boot —
  `max_concurrency x (1 + mtp_depth)`, floored at 8 so every existing recipe
  keeps its exact shape — and carried at runtime by the session core
  (`SessionParams::decode_rows`: the per-row scratch, the token feeds, the
  draft windows, the memory plan), the picker's tables, the bus's latency
  slot and the graph engine's batch families; the build-time bound is
  `kDecodeRowsMax` = 32 (the pick kernels' fixed arrays). GLM-4.7 takes the
  derived shape (`serve: decode rows 12 (4 slots x 3 rows ...)` in the boot
  log); GLM-5.3-Flash keeps its fixed 8, Qwen3.8-Flash-Next stays capped at
  8 until its kernels are gated past it. On the runtime rows the row batch
  carries depth >= 2: `session_graph_capture_draft_chain_batch` +
  `glm_spec_chain_rows_batched` run every slot's chain row in one draft-
  block run (the batched next-token feed carries every draft per request);
  a family opts in with `kBatchedDraftChain` (GLM-4.7). The GEMM interface
  lowers bf16 decode calls up to the model's decode rows to the row-
  independent GEMV core (`CublasLtGemm::set_decode_rows`; the first 9-row
  batch fell to an Lt algorithm with its own reduction order and flipped a
  near tie), so a batched request's rows keep the scalar order at any
  width — the depth-2 engine gate runs A scalar and B, C batched on a 9-row
  ceiling, bitwise the eager engine's. Also fixed: one slot's sampled MTP
  fallback dropped every slot's draft picks of a batched replay (the mirror
  went stale for the others). Measured on the fabric (`fabric_glm4_load.sh`):
  batched depth 2 loses to depth 1 at 2 and 4 live requests (37 vs 44.5,
  43 vs 47.5 tok/s) — every 4-row GEMV chunk past the first re-reads the
  6.3 GB of BF16 attention projections per rank (+45–60 ms per chunk: 4
  rows 79 ms, 6 rows 123, 8 rows 140, 12 rows 200); depth 2 remains a
  single-stream gain (+4–13 %). The lever for concurrency at either depth is
  the attention projections' bytes per pass: wider GEMV chunks or a row-
  independent tensor-core kernel, docs/measurements.md.

- **MTP depth 2 on the hidden-window families** (2026-09-10): the session
  core carries the GLM-5.3-Flash chain — one draft-block row per further
  draft at counter + index, fed the previous draft's pick and the block's
  own output row as its hidden, landed in the slot's window
  (`glm_spec_chain_row_window`); `session_draft_chain`,
  `session_graph_capture_draft_chain`, the multi-draft feed; a family opts
  in with `kDraftChain` (GLM-4.7 immediately; Qwen3.8-Flash-Next followed
  the same day, once its draft ring learned the chain snapshot — above).
  `deploy/cluster_glm47_d2.json`.
  GLM-4.7 at depth 2: 72 ms/pass at 2.3–2.6 tokens/pass (p2 48–65 %), 4–13 %
  more tokens/s single-stream than depth 1, +5 % at a 6.5K context;
  transcripts identical. Gate: `glm4_engine_test`'s depth-2 world (port
  29951); the row batch past depth 1 followed the same day (above).

- **`chat_template_kwargs.enable_thinking` is a knob of the templates that
  read it** (2026-09-10): the service asks the loaded template
  (`ChatTemplate::reads`, `ModelFrontend::template_reads`) and passes the
  boolean through for Qwen3.8-Flash-Next and GLM-4.7 (false closes the
  think block in the generation prompt — GLM-4.7's non-thinking mode; it
  ignores reasoning_effort and otherwise thinks to the token cap); GLM-5.3-
  Flash's template never reads it and the refusal stands. `serve_eval.py
  --no-think`; `serve_api_check.py` builds its stop material with thinking
  off where the template allows it.

- **Scheduler: a step cut short takes no retire-time prefix snapshot**
  (2026-09-10). When a multi-token MTP pass's earlier token completes the
  answer (the cap or EOS), the pass's remaining tokens are dropped and the
  model's state sits past the committed position; the retire path asked
  the arena for a live snapshot at the committed position and the arena
  refused (an engine failure on GLM-4.7's first fabric request — every
  position is snapshot-eligible at its align of 1; the aligned families
  hit it only when the cut lands on an aligned position). `Request::
  step_tail` records the dropped tail; the rolling entry stands as the
  close entry. `scheduler_prefixCache_aStepCutShortByTheCapTakesNoRetireSnapshot`.
  Gates: the python double reference (`tools/glm4_reference_dump.py`,
  final hidden l2 0.0035, top-1 and routing exact, the draft's rows),
  transformers' own layer code on the real weights (layer-0 relative l2
  0.0025), decode/sessions/speculator (60 steps across the pool's block
  boundary, per-step rolling snapshots), TP worlds 2 and 4, the graph
  engines at world 2 (scalar, batched and MTP across the boundary),
  tokenizer and chat-template goldens (HF tokenizers 0.23.1, jinja2
  3.1.2). Fabric (four nodes): boot 18–20 s from the resident image, T=1
  49.0 ms/step, MTP 60–61 ms/pass at 1.86–1.99 tokens/pass (31–33
  ms/token), MTP == T=1 transcripts, eval with thinking off gsm8k 60/60,
  HumanEval 39/40, extraction 30/30 (docs/measurements.md).

- **Shared cores extracted for the second and third families** (2026-09-09):
  `loaders/resident_stream.hpp` (the resident/streaming layer stream —
  bumps, staging, the image, byte reconciles, the digest — under a
  family's builders; Qwen and GLM-4.7 on it), `engine/paged_blocks.hpp`
  (the paged block table with its sharing and pinning protocol under both
  K/V pools), `engine/session_model.hpp` (the session core: positions,
  feeds, chunking, snapshots, the graph era, the draft plumbing, under
  `QwenModel` and `Glm4Model`). Every existing gate unchanged.

- The fp8 tile kernel ported to the ldmatrix form (32-deep stages, the
  block scale copied with the tile, fragments decoded e4m3 -> f32 x scale
  -> bf16 as the reference rounds them): bitwise the reference; the FP8
  checkpoint's steady-state prefill 740 / 1,725 / 7,463 -> 549 / 1,412 /
  6,280 ms at 512 / 2,048 / 8,192 tokens (with the mHC GEMM form), the
  hybrid's shared experts 440 / 1,290 / 5,769 -> 425 / 1,275 / 5,702.

- The prefill's mHC dots on tensor cores: a bf16 mma.sync GEMM over a
  cp.async ring with per-stage fp32 partials, the token's sum of squares
  gathered from the same tiles, then the standard finish; the sites no
  longer store the unread collapsed row. Within the oracle budgets (not
  bitwise the per-coefficient decode form; the tiled kernel stays behind
  `mhc_set_prefill_gemm(false)`): 1,196 -> 836 us per site at 2,048
  tokens; steady-state prefill 460 / 1,335 / 5,954 -> 440 / 1,290 / 5,769
  ms at 512 / 2,048 / 8,192; the 64-step transcript identical, one
  near-tie flip (0.049 logits) in a 512-token continuation.

- The fp4 prefill kernel, second pass: the ldmatrix tile kernel (64 x 128
  x 64, a three-slot cp.async ring of the activation tile and the raw fp4
  codes, B fragments decoded at fragment time, each weight row's next L2
  line prefetched ahead, the activation lines tagged evict-last, one m-tile
  per block with the m-tile the fastest grid index) is the production
  launcher on both expert shapes — bitwise the reference; steady-state
  prefill of the hybrid 563 / 1,502 / 6,572 -> 460 / 1,335 / 5,954 ms at
  512 / 2,048 / 8,192 tokens, generated ids unchanged. glm_moe_test's odd
  geometries caught misaligned copies at k = 208 (aligned-width fallbacks
  added). The FP8 path is untouched.

- The mHC site off the critical path: the fused finish leaves comb (the
  20-iteration Sinkhorn, 4.6 of its 8.6 us) to `launch_mhc_comb` on a
  low-priority side stream forked after the finish and joined before the
  stream update — the only reader — and reuses the dots block's register
  copy of the streams for the collapse; bitwise (glm_mhc_test pins the
  deferred form at 1/2/8 rows). The fused kernel 13.0 -> 6.5 us per site
  in the MTP graph; the hybrid's decode MTP 34.21 -> 33.89 ms/step (19.96
  -> 19.77 ms/token), T=1 26.46 -> 26.22. `mhc_site_bench` (new) times a
  site with the finish, Sinkhorn and norm separable. The router's register
  select and the collective's local phases were measured and parked
  (`docs/nvfp4_plan.md` §6a). DGPP_MHC_COMB_SIDE=0 keeps comb in the
  finish.

- The fp4 GEMV core decodes with GB10's own e2m1x2 -> f16x2 conversion and
  one exact f16 multiply by the group scale (bitwise the register decode;
  the build moves to the architecture-specific target `121a`), the
  activation window read once per chunk column; the decode slot path
  204 -> 195 us per MoE layer at one row, MTP 20.04 -> 19.96 ms/token on
  the hybrid. Hardware counters (Nsight Compute as root) and a no-arithmetic
  probe of the same access pattern put both fp4 slot kernels at 90-93 % of
  their streaming floor; a persistent grid-stride form with next-tile
  lookahead, the paired gate/up issue, more row steps, four blocks per SM,
  the router's unroll and block width, and programmatic dependent launch
  across the graph's interfaces were each measured and parked
  (`docs/nvfp4_plan.md` §6a).

- NVFP4 routed experts, phase 1 of `docs/nvfp4_plan.md` — the format is
  loadable. The composed checkpoint `dgpp/GLM-5.3-Flash-NVFP4-FP8`
  (`tools/compose_nvfp4_hybrid.py`: dabsLabs' NVFP4 experts for layers
  3–44 beside the FP8 release's bytes for every other tensor, the MTP
  layer included; 195.1 GB, built and verified byte-for-byte on all four
  nodes, identical shard hashes) declares `quantization_config.quant_method
  = "dgpp_mixed"`, which `GlmTextConfig` parses into a routed-expert format
  (`GlmExpertFormat`: FP8 block-128 as before, or NVFP4 group-16). Under the
  NVFP4 profile the expected-tensor table emits the triple `weight_packed`
  (U8 [N, K/2]), `weight_scale` (e4m3 [N, K/16]) and `weight_global_scale`
  (F32 [1]) for every main-stack routed expert and the validator binds it
  as a unit (every tensor now carries a `GlmTensorRole`, so an e4m3 block
  scale is never mistaken for an FP8 payload); the loader keeps the bytes
  untouched in `GlmFp4Matrix` views — this rank's inter slice as
  contiguous gate/up rows and nibble-granular down column packs, every
  matrix's global scale gathered into one per-layer array — with the same
  counting-mode byte formula, resident image and byte reconcile;
  `GlmTpViews` carves the same views from a full layer so the shard-parity
  gate pins the two paths bitwise on an NVFP4 fixture at worlds 2 and 4,
  streaming and resident. The MTP layer's experts, the shared experts, the
  dense MLPs and the attention keep the FP8 paths untouched under both
  profiles; the FP8 checkpoint's table and bytes are unchanged (the
  existing gates pin that). `glm_bind_check` on the hybrid: 112,049
  expected, 112,049 matched, 36,288 NVFP4 triples and 1,050 FP8 pairs
  bound; the resident formula at world 4 is 50.7 GiB per rank against
  ~82 GiB for the FP8 checkpoint.

- NVFP4 routed experts, phase 2 — the decode path runs on the packed
  bytes. `src/kernels/fp4_gemv.cuh` is the in-register e2m1 core: a lane
  loads 16 bytes (32 codes, two group-16 scales), places each code's bits
  into an fp16 field (`((code & 7) << 9) | ((code & 8) << 12)`, which reads
  as value x 2^-14 for every code) and multiplies by the scale x 2^14, all
  exact — an e2m1 times an e4m3 has at most six significant bits, so the
  weight enters the fp32 dot unrounded and the only inexact operation is
  the one division by the tensor's global scale in the epilogue
  (`tests/unit/fp4_test.cpp` pins the lemma over every code x scale pair
  and the bit-placement against the codec table). The core is templated
  on K so every index is compile-time (the first draft's lane-dependent
  accumulator index spilled to local memory and ran at 14 GB/s); it
  backs the single-matrix launcher (`launch_fp4_gemv_bf16/f32`), the fp4
  decode slot kernels (gate_up+swiglu and down, the shared expert running
  the fp8 core inside the same launch as before) and the fp4 grouped
  GEMV that `GlmMoeLayer` dispatches under the NVFP4 format; the host
  oracle dequantizes the triple exactly and applies the same divisor
  epilogue. Gates: `fp4_gemv_test` (the oracle across every supported
  geometry at l2_rel 0, row-count invariance, the f32 epilogue, NaN
  scales, the geometry contract, and two real slices of the hybrid at 0
  mismatches), `glm_moe_test`'s fp4 twins bitwise — the decode slot path,
  the host grouped path and the sliced-rank fold agree bit for bit as the
  FP8 ones do. `moe_slot_bench --format fp4` per layer on one rank's
  slice: rows=1 209 us against fp8's 273, rows=2 334 against 469, rows=8
  929 against 1,420. Prefill under NVFP4 uses the grouped GEMV chain
  (the tensor-core kernel is phase 3). The FP8 paths are untouched.

- NVFP4 routed experts, phase 3 — the prefill runs on tensor cores.
  `moe_grouped_mma_fp4_kernel` is the fp8 tile kernel's structure (128 x
  64 x 64 stages, the same activation tile and ascending-k16 bf16
  `mma.sync` chain) with the weight tile decoded from the NVFP4 triple:
  a thread's 16 consecutive k come from one 8-byte load of codes and one
  scale byte, decoded as e2m1 x scale, exact in bf16, so unlike the fp8
  tile nothing is rounded before the MMA; the global scale divides the
  finished dot once in the epilogue. Any k that is a multiple of 16 (the
  GEMV core's power-of-two set does not apply). The grouped form
  (segments through the row map) is bitwise the dense form per segment;
  `GlmMoeLayer` dispatches it for NVFP4 routed segments under the
  tensor-core kernel while the FP8 shared expert keeps the fp8 tile
  kernel; the fp8 kernel's source is untouched. Gates: grouped bitwise
  the dense form at I = 208 (a ragged n-tile and a ragged last k-stage)
  and 320, under the z split and through a permuted row map, gate bf16
  and down fp32; the whole layer within budget of the double oracle on
  four geometries beside the GEMV chain; both real checkpoint slices at 0
  mismatches with a row's bits independent of m. Fabric, steady-state
  prefill on the hybrid: 512 tokens 1,152 -> 635 ms (FP8 740), 2,048
  tokens 4,396 -> 1,563 (FP8 1,725), 8,192 tokens 18,246 -> 6,796 (FP8
  7,463); the 64-step transcript identical to the GEMV-chain build's, all
  ranks identical; the FP8 build's 2,048-token prefill unchanged (1,717
  ms). nsys says the tile kernels — fp8 and fp4 alike — sit at a third of
  the read floor (a 340 MB fp4 launch takes 4.6 ms): the activation tile
  is re-read once per 64-wide n-tile, the stages do not overlap, and a
  128-row m-tile is more than half padding at ~57 rows per expert; the
  restructure is `docs/nvfp4_plan.md` §6a (phase 5).

- `scripts/serve_eval.py`, the task-level eval through the served endpoint
  (HumanEval, GSM8K, schema extraction; greedy), run on both configs: the
  NVFP4 hybrid 157/164, 293/300, 100/100 against FP8's 155/164, 293/300,
  100/100 — the same items pass on both to within two HumanEval problems
  and three GSM8K flips each way, while the replies are word-identical on
  only two thirds of the items. `docs/nvfp4_plan.md` §6a.

- The DSA attention projections consumed as FP8 directly (the "bridge"
  item of `docs/nvfp4_plan.md`): q_a, kv_a, q_b and o_proj — FP8 pairs in
  every checkpoint — were dequantized to BF16 at load (the M3 interface) and
  read at twice their bytes on every step. The loader now keeps them as
  the checkpoint's pairs whenever this rank's q_b row slice and o_proj
  column slice start on the 128-wide scale grid (every power-of-two world
  of the production geometry; the bf16 bridge remains for misaligned
  slices — the test fixtures at world 4), `DsaLayerWeights` carries either
  form, and the layer's three projections run through the scale-aware
  GEMM (which gained an output row stride so the fused [q_a | kv_a]
  output takes two pairs into one buffer). `GlmTpViews` slices the pairs
  at aligned worlds and dequantizes the full pairs itself (the loader's
  kernel and rounding) for the bridge form at misaligned ones, so the
  shard-parity gate pins both forms bitwise against the sharded loader.
  The resident image format version is 2 (bridge-layout images are not
  restored). Gates: `dsa_test`'s new prefill case runs the same weights
  in both forms against the host reference; `glm_loader_test` checks the
  pairs byte-exact; shard parity at worlds 2 (direct) and 4 (bridge).
  Applies to the FP8 model and the hybrid alike: 0.35 GB per token per
  rank fewer over the 11 DSA layers.

- MTP depth 2 re-measured on the NVFP4 hybrid through the serve path: 4-6 %
  faster per token on 300-token prose, code and JSON answers at short
  context (2.5 tokens per step, the second draft standing 59-61 %), 6.6 %
  slower at a 7,368-token context (the second draft standing 41 %). Depth
  1 stays the default; the numbers are in `docs/nvfp4_plan.md` §6a.

- NVFP4 routed experts, phase 5, the decode core: one row step per warp in
  the fp4 GEMV core instead of two (`fp4_gemv::kSteps`) — twice the blocks
  per launch, four loads in flight per lane — after a sweep of the core's
  tunables on `moe_slot_bench` in which every "more per lane" direction
  lost; 5-6.5 % faster per layer at 1, 2 and 8 rows in an alternated A/B,
  outputs bitwise (a row's chain does not depend on the step count).
  Fabric, the hybrid: T=1 28.0 -> 27.0 ms/step, MTP 36.6 -> 35.4 ms/step
  (21.2 -> 20.5 ms/token), acceptance unchanged.

- NVFP4 routed experts, phase 5, the prefill tile kernel: a warp whose 16
  rows lie entirely past a segment's end skips its MMAs (its rows are
  never stored, so the outputs are bitwise), and `GlmMoeLayer` dispatches
  the fp4 tile kernels by shape — the synchronous 128 x 64 x 64 tile for
  gate/up, a pipelined 64 x 128 x 32 kernel (cp.async double-buffered
  activation stages, the next stage's codes decoded after the MMA) for
  the down projection, each the faster one on its shape. `moe_tile_bench`
  times every variant per launch at the production expert shape with
  uniform or ragged segments and checks the pipelined kernel bitwise
  against the reference; a decomposition with the loads stubbed out shows
  the gate launch is 58 % padded MMA + decode + barriers and the rest
  un-overlapped operand loads (`docs/nvfp4_plan.md` §6a records the
  numbers and what remains). Fabric, steady-state prefill on the hybrid:
  512 tokens 563 ms, 2,048 tokens 1,502, 8,192 tokens 6,572 (phase 3:
  635 / 1,563 / 6,796); transcript identical, all ranks identical.

- NVFP4 routed experts, phase 4 — the hybrid serves. A site keeps one
  cluster config per checkpoint and names it on the command line
  (`deploy/cluster.*.json` is git-ignored beside `deploy/cluster.json`;
  `docs/operations.md`): `deploy/cluster.nvfp4.json` selects
  `dgpp/GLM-5.3-Flash-NVFP4-FP8` with `kv_capacity` 786,432 and
  `prefix_cache_gib` 8, the 31 GiB per rank the experts free spent on
  context and cache (`--memory-plan`: 96.2 GiB + 8 headroom of 116.5).
  `scripts/dgpp-cluster up --config deploy/cluster.nvfp4.json` boots from
  the resident images in 18 s; `serve_api_check.py` passes every case;
  the world tears down with four identical op-stream md5s and no stall.
  The L2 prefetch defaults stand on the hybrid: the window is flat from 8
  to 24 MB at T=1 (28.0 ms/step) and under 1 % apart under MTP; the
  prefetcher itself is worth 2.0-2.3 ms/step, the full boundary rate
  costs 2 ms, as on FP8. `scripts/serve_bench.py` and `serve_soak.py`
  address the model the service reports (`GET /v1/models`) instead of a
  hard-coded FP8 id, with an optional argument to override.

- `glm_gen_check --mtp --decode-graph` accepted no draft since 2026-09-06
  (every class 0.0 %, the transcript still exact): the decode-step commit
  moved the scalar graph's fed tokens to the slot's persistent feed rows
  (`device_feed`, rows 8 onward) and the serving adapter followed, but the
  evidence app's recorded verify kept comparing row 0's winner with the
  eager rows at `device_tokens()`, which nothing rewrites during a replay.
  The app's recorded verify now reads the feed rows; the eager speculator
  (77 % accepted) and the serving path were never affected. Found by the
  NVFP4 evidence chain; bisected with the app built at 2b7a6c6 and
  f8b2d34 (both 0 %) against the eager path (77 %).

- The row batch is a family: a 2-slot (4-row) and a 3-slot (6-row) batch
  are recorded beside the full 8-row one, and a step replays the smallest
  whose slots cover the live requests. Two live requests used to be either
  two scalar replays (83 ms/step) or the 8-row batch with four padding
  rows (96.5 ms/step, the reason the crossover sat at four); the default
  crossover is now two. A single live request replays its scalar graph as
  before. Gated bitwise against independent scalar sessions and
  speculators through every occupancy and the hole cases, plain and MTP,
  on the loopback world.

- A prefix cache miss is explained on its own INFO line: the cuts probed,
  the entries held, and where the prompt parts from the entry it shares
  the most with (`PrefixCache::nearest`), so a client that edits the
  system prompt or compacts its history between turns is named on the
  spot; when the conversation's own entry was pushed out of the arena the
  line says that instead, from a ring of the last 256 evicted entries
  (`PrefixCache::ghost_at`). `scripts/serve_agentic_streams.py` reproduces an agent client's
  pattern — several concurrent multi-turn tool loops with the reasoning
  stripped from the history, side requests between turns, an optional
  system-prompt edit — and reads the cache's answer per turn from
  `usage.prompt_tokens_details.cached_tokens`; the scheduler suite pins
  two interleaved conversations hitting on every turn under a six-slot
  arena, one returning its reasoning (attaching at the close entry) and
  one stripping it (attaching at the previous prompt's header cut).

- An integer's `minimum` / `maximum` / `exclusiveMinimum` /
  `exclusiveMaximum` are enforced by constrained decoding, for a tool
  argument and for a `response_format` schema alike: the bounds compile
  to one inclusive 64-bit range (a fractional bound rounded inward, an
  exclusive one stepped by one), the JSON machine admits a digit only
  while some completion can still land inside the range and the value's
  closer only once the digits do, and the mask re-judges the pure-numeric
  tokens by the same arithmetic, proven equal to brute force by the walk
  gate. A `timeout` with `minimum: 1` can no longer come out as `0` or
  `-5`, a `limit` with `maximum: 2000` cannot exceed it, and the tool
  behind them never sees a value its schema excluded (the Hermes agent's
  definitions carry such bounds on every integer argument; before this
  the server logged one INFO line per bound saying it was not applied).
  A number that admits a fraction keeps its bound unenforced — refused
  under `strict`, noted with the reason for a non-strict tool — as does
  the draft-4 boolean form, a bound beyond int64, an empty range, and an
  enum no member of which fits; an enum beside a bound is filtered to the
  members inside it. The INFO line for the keywords that stay unenforced
  now carries the reason. Also fixed on the way: beside a bounded (or
  plain) integer alternative, an enum of integers let the mask admit a
  `.` every cursor then refused; and a top-level number under an `anyOf`
  is complete where some alternative accepts it, not only where all do.

- Fixed: the decode index selection's histogram sat at a call-dependent
  workspace offset (the call's rows of keys), so a ONE-row call — the
  sampled fallback's eager verify or re-draft — read its histogram out of
  row 1's keys, which every two-row replay writes: a garbage histogram, a
  partial best-list fill, stale shared-memory pool ids expanded into token
  ids, and the listed attention reading an unmapped page (an engine
  failure once in roughly ten fallbacks past 2,048 tokens of context; the
  first report was a live service at 22:13 on 2026-09-06). The histograms
  now sit at a fixed offset for the maximum rows and are zeroed every
  call; the best list starts empty and the expansion drops any pool past
  the row's visible pools; the listed gather checks the token and its
  block and zero-fills instead of faulting; both record their first
  anomaly, logged at the slot's close (`dsa_select_anomalies`,
  `dsa_attn_anomalies`). `DGPP_SYNC_EAGER=1` syncs an eager row after each
  stage and validates the selection list before the attention — the knob
  that found it. Verified by an 11-turn long-decode sampled soak (about 150
  fallbacks past 2K tokens of context) with no anomaly, where the
  unfixed select logged short fills on every request, and by the DSA
  suite.

- The pipelined replay: the engine's step returns at the verify's verdict
  (a kernel node publishes the slot's replay sequence to pinned memory)
  and the next replay is launched before the previous one's draft tail
  and window are settled, so the verdict read, the scheduler's
  bookkeeping, the rolling prefix snapshot and `cudaGraphLaunch` itself
  (~0.5 ms at 1,168 nodes) hide behind the tail. The token feed is per
  slot on the device, the pick's masks ride a pinned staging behind an
  in-graph handshake, the bus holds two live replay windows (a ring of
  arms, FIFO finish, in-order adoption) and every graph shape has two
  alternating variants. Eager bus work drains the replays in flight.
  `DGPP_PIPELINE=0` restores the settle-after-launch step. Measured on
  the fabric (one greedy request, the same binary pipelined vs
  `DGPP_PIPELINE=0`): 41.2 vs 41.7 ms per pass at a short prompt, 41.9 vs
  43.0 at 8K, 42.6 vs 43.3 at 32K — 0.5 to 1.1 ms per step, 44.1 / 45.3 /
  43.7 vs 43.4 / 44.8 / 43.1 tok/s on prose / code / JSON (+1.1 to
  +1.6 %); the greedy transcripts are byte-identical to the earlier
  plain, depth-1 and depth-2 runs. Three loopback gates run their eager
  oracles through the drain; the bus test arms two windows at once.

- MTP draft depth (`engine.mtp_depth`, `--mtp-depth`, 1–3): the verify
  runs 1 + depth rows and the draft block's chained rows (one more block
  row per draft, on its own output) propose the drafts after the first.
  The sampled verdict kernel decides T rows in a chain (draw for draw
  what the eager sampled speculator decides), the host fallback continues
  the chain from the row that fell back, the tail ring is guarded around
  the chain rows, the hop snapshot takes the rows the step committed past
  the position, and the device picker carries one slot per draft. Depth 1
  is byte-for-byte the two-row step as built; past depth 1 every step is
  a scalar replay. New gates: the T=3 verdict oracle, the greedy depth-2
  and depth-3 loopback gates (plain transcript, feed == the eager chain),
  the sampled depth-2 loopback gate (transcripts == the eager depth-2
  speculator's through 17 fallbacks). On the fabric the greedy transcripts
  of three prompts are byte-identical plain / depth 1 / depth 2; depth 1
  is unchanged at 42–43 ms/step; depth 2 is 54–56 ms/step with the second
  draft standing 45 % (prose) to 65 % (code) of the time — −4 % on prose,
  +4 % on code and JSON — so depth 1 stays the default
  (`scripts/mtp_depth_check.py` runs the transcript check).
- Per-position draft acceptance: the engine counts, per slot and overall,
  the steps that verified each draft position and the steps in which it
  stood; the stats line's `mtp` group prints `accept p1 74 % p2 61 %` and
  every MTP retire line carries the request's own `accept p1 78 %`.

- The KV cache's dtype is a config key (`engine.kv_dtype`, `--kv-dtype`):
  `bf16`, `fp8` (e4m3 + a row scale) or `fp4` (e2m1 in blocks of 16 with
  e4m3 block scales); the attention kernels dequantize on the load, the
  index cache and the selection are unchanged, the format rides the
  settings record and the config digest. New unit and CUDA tests pin the
  codecs, the append and the three attention kernels per format.
- The memory plan: every rank itemizes every byte it will allocate for its
  configured shape and refuses to boot when it does not fit the node's
  free memory, before the first allocation, naming the largest items and
  the largest `kv_capacity` that would fit; `dgpp-serve --memory-plan`
  runs the check alone. The model's per-forward row bound is now the
  prefill chunk rather than the whole context (the activations no longer
  grow with `kv_capacity`: a 262k-token context needs ~98 GiB per rank
  instead of ~300 GiB and boots), `max_context()` is the context bound.
- Every line the server prints is timestamped: the step-timing report and
  the MoE chain dump go through the logger, an uncaught exception leaves a
  stamped line before the abort, and the launcher stamps its own lines in
  the same format. The stamps are UTC on every rank (a peer's log used to
  carry its node's own zone, seven hours off the head's).
- The throughput line leads with what an operator watches — `decode 30.9
  tok/s, 31.4 ms/tok, 54.6 ms/step (178 steps / 309 tok, 97 % of wall) |
  mtp 1.74 tok/step/req = 74 % drafts accepted | prefill … | live 0,
  queued 0 | pool … | prefix cache … | requests …` — and every request's
  retire line carries its own numbers: the prompt and cached tokens, the
  prefill's ms, the decode tokens and passes, tok/s, ms/tok, ms/pass and
  MTP's tok/pass.
- A non-strict tool argument keeps its type under a keyword that only
  narrows the value (`minimum`, `maxLength`, `pattern`, `format`, …): the
  bound is not enforced and one INFO line says so, once; a schema outside
  the subset in shape still leaves the value free, with one WARN line,
  once (the same three WARN lines used to repeat before every request of
  an agent client). Strict tools refuse as before.
- The settings record carries the resolved row-batch threshold, so the
  pushed and the effective config print the same `batchmin`.
- The decode select kernel (DSA, the step's only context-scaled kernel)
  rewritten: every block scores its stripe and publishes the keys plus a
  radix histogram, the last `rows` blocks each find one row's boundary key
  by radix refinement, gather, and expand with a rank sort — the same set
  under the key's total order, so the bitwise gates are unchanged.
  `dsa_select_bench` at the MTP shape: 4K / 8K / 32K / 393K tokens of
  context 572 / 608 / 832 / 3700 us before, 25 / 33 / 67 / 509 after; the
  12 calls per step were 7–10 ms of the 54 ms step at agent-sized contexts.
  The workspace is `dsa_select_workspace_bytes` (keys and histogram per
  row, sized for the pool capacity), counter_ws is two words.
- The MoE down projection runs four rows per warp (`block_rows_multi`): the
  sliced k of 512 bytes gave a warp one load in flight; the per-row chain
  is factored (`consume_chunk`) so every row is bitwise the old one.
  `moe_slot_bench` at the per-rank geometry: the two-row decode chain
  487 -> 457 us per layer (1.3 ms per step), the eight-row 1598 -> 1412.
- On the fabric (one request, sampled MTP): 41.8 / 44.4 / 44.5 / 45.2
  ms per step at 26 / 8K / 18K / 32K tokens of context, against 43.4 /
  53.6 / 54.8 / 56.2 before the two kernels — 23–25 ms per token at every
  context an agent client uses.
- The decode attention partial kernel keeps each thread's q window and
  c accumulator in registers instead of shared memory: the block drops
  from 95 KB to ~40 KB, two blocks per SM, the 64-block decode grid in one
  wave — 2.24 -> 1.16 ms per step on rank 0's trace at 8K, bitwise the
  smem form (each thread always owned exactly that window). The gather
  loop is unrolled so its token -> block-table -> row chains overlap. The
  split count stays 32 (48 and 64 measured the same step on the fabric).
- The L2 prefetcher coalesces adjacent adds into one launch
  (`DGPP_L2_PREFETCH_MERGE=off` restores one per tensor): the decode
  graph goes from 1,595 to 1,168 kernel nodes and `cudaGraphLaunch` from
  730 to 520 us per step (the host enqueues ~0.45 us per node and the GPU
  idles through it — the interface the design record called hidden). The
  merged kernels stream through more of each collective's handshake
  (+3 us per collective on the timeline), so the step itself is within
  noise of the per-tensor form; kept for the node count.
- The mHC dots kernel's two streaming loops unroll 8 (bitwise; ~0.1 ms
  per step). Tried and reverted, the reasons in the code: batching the
  bus fold's loads (the fold span stayed 9.3 us), and decoding q/k to
  floats in the select kernel (four times the smem bytes per pool).
- The launcher forwards every `DGPP_*` knob in the head's environment to
  the peers.
- The graph collective's fold reads staged copies: the placement gate's
  hash pass, which reads every claimed payload anyway, writes it into
  dynamic shared memory (48 KB for the decode's three 16 KB payloads;
  wider vectors fold from the NIC-placed rows as before), and the fold
  after the last arrival is a shared-memory pass — 9.3 -> 5.9 us per
  collective on the bus timeline, bitwise the same chain. Staging needs
  every peer slot 16-byte aligned (element counts a multiple of 8): the
  step's small all-gathers are not, and the first fabric boot found that
  as a misaligned address the loopback gates' widths never hit.
- The decode attention runs on the tensor-core listed kernel
  (`DGPP_DSA_DECODE_MMA=off` restores the register split kernel): two rows
  x 16 local heads is one M-block, each slab its row's own selection with
  its own request's block table. The DSA layer at the rank's shape:
  0.563 -> 0.516 ms per row per layer at 8K. Numerics: the mma summation
  order, tolerance-equal — the teacher-forced gate over the 556-token
  text passed at a mean NLL delta of +0.0008 nat (bound 0.02), perplexity
  2.587 vs 2.585, no move over 1 nat.
- On the fabric after both: 41.9 / 42.7 / 43.2 / 43.5 ms per step at 26 /
  8K / 18K / 32K tokens of context (400-token generations).
- `DGPP_BUS_TIMELINE=1` writes the bus's per-window collective timeline
  (copy / handshake / skew / fold per collective, the wait histogram, each
  peer's lag and how often it arrived last) at INFO without the rest of
  the debug output, which perturbs what it measures; the launcher forwards
  it, and the head's `DGPP_LOG_LEVEL`, to the peers (the peers used to
  expand the level on their own environment and always ran at info).

## 0.1.0 — 2026-09-06

Productionizing: the model-agnostic code left the `glm` namespace
(`dgpp::serve`, `dgpp::sched`, `dgpp::sample`, `dgpp::text`; GLM under
`src/models/glm/`; the binary is `dgpp-serve`); one cluster config
(`deploy/cluster.json`, a site's copy of `deploy/cluster.example.json`)
read by the launcher and every rank, the head
pushing the world's settings to the peers over the journal before anything
builds; versioned releases (`scripts/release.sh`) installed once per node
(`dgpp-cluster install`), the version stamped into the binary and refused
across a mixed world; `scripts/dgpp-cluster` as the launcher.

The request contract gained `stop`, `n`, `logit_bias`, and the usage's
`cached_tokens` and `reasoning_tokens`. Every rank's log carries one
aggregate throughput line per 10 s; the per-tick lines moved to DEBUG.

## 2026-09-05 — M9 sign-off, prefix caching and service hardening

- M9: the continuous op-stream drift check on every journal record; a
  malformed-HTTP fuzzer under AddressSanitizer (three defects found and
  fixed); the one-hour mixed soak; the operator's page; the sign-off report
  (32K TTFT, MTP acceptance per class, the prefix cache's capacity curve);
  the ranked next-steps list.
- M7: the exact snapshot prefix cache — decisions rank-identical in the
  scheduler, the snapshot arena in the engines, hot == cold bitwise.
- The v1 failure semantics built, gated and drilled with kill −9 on the
  four nodes; the strict tool grammar's key ledger; cancellation under the
  one-graph step gated.
- Prefill rounds 4–9: the bulk collective as a cooperative kernel with paced
  senders, the MoE experts on grouped tensor-core kernels, the attention
  prefill as a flash kernel; 256 tokens 0.58 s, 2,048 tokens 1.7 s.

## 2026-09-04 — sampling, constrained output and shutdown handling

Tool calls and reasoning on the wire; constrained decoding as a guarantee
for `tool_choice` and `parallel_tool_calls`; `response_format` with JSON
schemas through the same masks; typed tool arguments; drain-on-stop;
grow-on-demand admission; exact sampling on the service with the on-device
verdict and its gather fallback; the sampling-width sweep; prefill rounds
1–3.

## 2026-09-03 — M8: transactional MTP; the on-device step

Greedy speculative decode with the checkpoint's MTP layer as one graph
replay per step, transcript identical to plain decode (31.3 → 22.45 ms per
token); the pick behind the head on the device; the adaptive scalar and
row-batched graph variants that removed the low-occupancy regression; the
kernels-only decode graph that closed the batched-MTP loopback stall; the
teacher-forced log-probability gate; fp32 logits.

## 2026-09-01 to 2026-09-02 — M6: generation, tokenizer, service

The incremental decode engine; the byte-exact tokenizer and the Jinja
chat-template interpreter; the deterministic scheduler; the
OpenAI-compatible service at world 1, then on the fabric behind the
admission journal; the decode step as a recorded graph with the
collectives as graph nodes; eleven optimization rounds on the T=1 step
(393 → 31.45 ms per token); the one-pass resident load and the per-rank
image cache (boot 15–25 s).

## 2026-08-29 to 2026-08-31 — M5: four-rank tensor parallelism

The RC/RoCE CollectiveBus with its slot pools, dual-lane striping and
watchdogs; the epoch-based roster; the sharded resident load; the fabric
TP=4 parity gate at real dimensions; the resident serving mode.

## 2026-08-27 to 2026-08-28 — M0 to M4

Platform, topology, transport and checkpoint facts; the core runtime, the
safetensors loader and the synthetic graph testbed; the KDA operators and
state manager; DSA/MLA sparse attention with its index pools; the
assembled 45-layer GLM forward with per-layer parity against torch
references.
