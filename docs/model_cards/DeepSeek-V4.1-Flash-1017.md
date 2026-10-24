# DeepSeek-V4.1-Flash-1017 on DGPP

This documents how DGPP, a C++/CUDA inference engine for DGX Spark, serves
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
DeepSeek's 2026-10-17 refinement of
[DeepSeek-V4.1-Flash](DeepSeek-V4.1-Flash.md). It is **not** a republished
checkpoint: the engine loads the upstream release as shipped, with no
re-quantization pass.

## The checkpoint, as loaded

The release is a weight update, not an architecture change: its
`config.json`, `inference/config.json`, tokenizer and
`model.safetensors.index.json` are byte-identical to V4.1-Flash's (the same
96,085 tensors in the same 48 shards, 475.2 GiB, the same dtypes and
offsets; every shard's safetensors header is byte-identical too), and every
one of the 48 shards carries different weights. The `deepseek_v41` family
therefore serves it through the same loader, kernels and prompt encoder as
V4.1-Flash: MXFP4 routed and draft experts, FP8 e4m3 dense and attention
projections with e8m0 block scales, the two FP8 Engram tables mapped from
the NVMe, BF16 embedding and head, the trained fp4/fp8 cache formats. The
Engram hash sidecar (`dgpp_engram_tables.json`, derived from the tokenizer)
is identical to V4.1-Flash's and is regenerated beside the snapshot.

Because the headers match, the resident image cache's key (config plus
shard headers) is the same for both checkpoints. Since 2026-10-18 every
image keeps a weight-identity note beside it and an image whose note
disagrees with the checkpoint is discarded and rebuilt; the old V4.1-Flash
images on this cluster were deleted when the checkpoint was replaced.

## The indexer budget

Every compressing layer attends only the compressed-KV entries the indexer
selects for a query: `index_topk` of them (the release trains and ships
512), chosen on the reindex layers from a candidate pool of
`candidate_topk_blocks` blocks of `candidate_block_size` entries (2048 × 8)
that the candidate layer scores once. The three values are deployment keys
since 2026-10-18 (`engine.index_topk`, `engine.candidate_topk_blocks`,
`engine.candidate_block_size`; the select networks serve up to 2048), so a
deployment can show each query more of its context than the release's
budget does — the lever against the sparse attention's misses on long
inputs. A wider budget changes the model's answers (more context visible),
not only its speed. The template ships the checkpoint's 512 and carries
the key (`"index_topk": 512`, so the opt-in is a one-value edit); the
measurement below prices 2048 against it on the same binary.

## Serving

Four DGX Spark nodes, tensor parallel, the V4.1-Flash template's shape: six
request slots at DSpark depth 4 with the confidence-scheduled verify depth,
the bounded prefill (the model's own SWA-replay recipe), the full
1M-token context. The tokenizer, prompt renderer (`encoding/encoding.py`,
no Jinja template) and DSML tool-call format are V4.1-Flash's.

## Decode levers (2026-10-18)

**The L2 prefetch windows.** The decode step is one serial chain of
kernels, and while it waits in a bus fold or a selection the memory
system idles. Every other decode family streams its next weights into L2
through those phases (`kernels/l2_prefetch.hpp`, `engine.l2_prefetch*`);
the DeepSeek-V4.1 walk did not. It now opens the same windows: the MoE
site's coefficients, norm, router and shared expert before the attention
fold; the next layer's attention side (its coefficients, norm, wq_a, wkv
and wq_b in consumption order) before the MoE fold, or the Engram
projection when the next layer carries one, or the head behind the last
layer; an Engram layer's own attention side under its fold; and the
output projections wo_a and wo_b under the selection and the attention
inside the CSA2 layer. Nothing the model reads is written, so the walk is
bit-identical with the prefetcher on or off. Measured at one request on
this template against the same binary with `engine.l2_prefetch: false`
(three greedy 256-token answers per class, the engine's decode steps):

| Class | Pass, prefetch off | Pass, prefetch on | Decode tok/s |
|---|---|---|---|
| Prose | 45.1 ms | 40.9 ms | 43.8 → 48.4 (+9.4 %) |
| Code | 53.3 ms | 49.9 ms | 65.5 → 70.1 (+6.5 %) |
| JSON | 56.1 ms | 52.7 ms | 73.4 → 78.0 (+6.0 %) |
| Math | 52.7 ms | 49.4 ms | 71.2 → 73.7 (+3.4 %) |
| Chat | 47.2 ms | 43.5 ms | 46.9 → 49.9 (+5.9 %) |

Every transcript was identical between the two.

**The verify schedule's cost model.** The confidence-scheduled depth
verifies a draft while its survival probability exceeds lambda times the
row cost, so the row cost decides the depth and the template had carried
the engine's defaults (8 ms per row, 28 ms base). Sweeps at fixed depths
1 to 4 with the schedule off, and the plain step, put every class on one
line, pass = 31.1 ms + 5.17 ms × (1 + drafts verified), with residuals
under 0.25 ms:

| Mode | Prose | Code | JSON | Math | Chat |
|---|---|---|---|---|---|
| Plain step (ms) | 31.1 | 31.1 | 31.1 | 31.1 | 31.0 |
| Depth 1 (ms per pass / tokens per pass) | 40.9 / 1.66 | 41.3 / 1.83 | 41.6 / 1.95 | 41.3 / 1.92 | 41.2 / 1.70 |
| Depth 2 | 46.3 / 2.02 | 46.8 / 2.55 | 47.1 / 2.87 | 46.8 / 2.80 | 46.6 / 2.18 |
| Depth 3 | 51.3 / 2.16 | 51.7 / 3.27 | 52.4 / 3.54 | 51.8 / 3.40 | 51.6 / 2.45 |
| Depth 4 | 56.0 / 2.24 | 57.0 / 3.75 | 57.7 / 4.25 | 56.8 / 3.98 | 56.7 / 2.58 |
| Fitted base / row (ms) | 31.1 / 5.01 | 31.1 / 5.18 | 31.0 / 5.34 | 31.2 / 5.14 | 31.0 / 5.15 |

A verify row costs its extra distinct experts, about 5 ms, and the draft
itself 5.1 ms (the fitted base plus one row, less the plain step). The
template now carries `engine.mtp_schedule_row_ms: 5.2` and
`engine.mtp_schedule_base_ms: 31.1`, which lowers the survival threshold
the schedule demands from 0.36 to 0.23 at the configured lambda floor.
At one request the change is level (prose +1.8 %, code +0.7 %, JSON
−1.0 %, math −1.2 %, chat −0.4 % against the defaults on the same binary,
within the sweep's noise): the schedule was already near its optimum, and
the keys are kept because they are the measured truth. These sweeps were
measured before the prefetch windows; the row and base costs scale with
the pass and the fit's structure does not change.

**The step cost under load is linear.** A later sweep on the prefetch
build at fixed depth 4 across the concurrencies, read in the server's
steady windows (the phase averages mix the tail where fewer requests are
live), put the step at 27.0 ms for the plain row, 37.0 / 42.7 / 48.0 /
53.6 ms at 2 / 3 / 4 / 5 rows, about 77 at 10, about 130 at 20 and about
185 at 30 rows: 5.2 ms per verify row all the way to the batch's 30 rows,
the row's own expert bytes with no overlap benefit (six of 384 experts
per row overlap under 20 % at 30 rows). A measured step-cost curve
(concave in the rows) was implemented and measured here and adds nothing
over the linear model, so it is not shipped. The same windows settle the earlier reading that fixed depth 4
beat the schedule at eight requests: on code at six live the schedule
commits 2.97 tokens per request per 116 ms step (154 tok/s sustained)
against fixed depth 4's 3.7 per 182 ms (122 tok/s); the campaign's C8
phase, eight 256-token requests on six slots, averages over its tail.

## The decode pass at C1 (2026-10-19)

A node trace of the served C1 pass (46–47 ms wall, about three rows per
pass with the drafter's rows): the expert gate/up kernel 12.2 ms at the
DRAM read ceiling, the down kernel 7.7 ms at about 78 % of it after its
rows (K = 576 on four nodes) went to one pass per lane, the dense GEMVs
11.9 ms, the 95 collectives 7.3 ms (76 µs each: copy 4.7 + handshake
17–22 + skew 19–22 + fold 9), the attention 2.9 ms, and 1,013 kernels
under 15 µs 5 ms (the norms, the mHC coefficient pairs, the attention
finish, the router). The per-row cost of 5.2 ms is the verify row's
expert bytes — six expert slices of 4.4 MB on 40 layers, 1.06 GB per
rank — at 89 % of the read ceiling; the draft chain's own rows cost about
0.1 ms each. The lm head's 12-bit companion (`engine.bf16_weights: bf12`
in the template) is packed for this family now, bitwise the bf16 head.

Excluded for the collective's jitter, each with the bus timeline on every
rank: CPU idle states (level; the bus engine spins), GPU clock locks at
2500 and 3000 MHz (level; the step is DRAM-bound), the L2 prefetcher off
(the skew unchanged; the prefetcher pays 50 µs per generation); every
rank arrives last about a quarter of the time, so no node is slow. The
collective's remaining 40 µs per round is protocol latency, a bus
project; the small kernels are a fusion project with exactness
contracts. Beyond its bytes (4.25-bit experts and 8-bit dense, lossless)
this pass carries about 12 ms of latency-bound work.

## The collective transport (2026-10-19)

The decode step's folds and the pick's stream collectives travel over
`engine.collective_transport` (`write`, the default; `send` is the form
until 2026-10-19). The write transport lays every row out as 512-byte
self-describing chunks, RDMA-writes it into each peer's write ring across
both PCIe functions, and folds it with a grid whose warps poll their own
chunks with every peer's copy in flight at once; each block reports
through a pinned record the engine aggregates. Bitwise the send machine's
fold (the bus test's graph, row and stream scenarios under both).

Measured on four nodes with the bus timeline on every rank (`timed_load`,
ms per pass at equal accepted rows, decode tokens per second; the
boundary prefetch window at its template setting):

| Shape | send | write |
|---|---|---|
| C1 code (3.59 rows) | 47.99 / 74.8 | 47.2 / 76.1 |
| C1 prose (1.90 rows) | 40.59 / 46.9 | 40.3 / 47.2 |
| C1 JSON (4.11 rows) | 52.17 / 78.8 | 51.0 / 80.6 |
| C6 code (20.7 rows, 512 tokens) | 170.96 / 121.2 | 165.32 / 125.3 |

Per collective at C1 (rank 0, medians of window averages, µs): send
64.2 = copy 5.1 + handshake 21.5 + skew 21.6 + fold 9.2; write 38.4 =
3.6 + 16.9 + 16.4 + 0.5. At C6 (165 KiB rows): 267 → 198. The C1 pass
gains less than its collectives because the boundary prefetch window
drains under the fold and turns the wait into DRAM time (the window off:
the collective 42 µs, the pass +1.9 ms); what remains of the collective
is the ranks' arrival skew — the fabric delivers a row in 5.5 µs in situ,
the peers stage 10–15 µs apart. The one-lane form (no striping) costs
1.1 ms at C6 and nothing at C1.

## The six-slot template at C6 and C8 (2026-10-20)

The template serves six slots: the family's decode batch is bounded at 32 rows and six slots at depth 4 are 30.
The serving table's C8 cell is therefore six live requests and two queued — the eight run as a six-wide batch
until the first completes, then the two that waited run alone — and the engine's rows per step average 4.0 at C8,
as at C4. The cell's repeat spread is wide for the same reason (the tail's composition and the adaptive depth's
choices): the 2026-10-19 campaign's three json repeats read 123, 132 and 157 ms/pass, the 2026-10-20 campaign's
122, 144 and 153, so a three-repeat median moves by one draw. The two binaries A/B'd on the same template
(256-token completions; six repeats at C8, three at C6; medians):

| binary | C6 json | C6 code | C6 math | C8 json | C8 code | C8 math |
|---|---|---|---|---|---|---|
| 2026-10-19 (`0d69d565`) | 171.4 ms/pass, 144.5 tok/s | 125.2, 114.5 | 131.2, 119.5 | 152.5, 108.1 | 127.4, 89.5 | 123.5, 103.2 |
| 2026-10-20 (`388111ef`) | 162.2, 154.6 | 117.0, 125.8 | 125.2, 132.9 | 138.4, 124.2 | 119.0, 95.7 | 104.7, 123.9 |

C6 is the template's steady state and its peak, 126–155 tok/s; the day's binary is faster at every class at both
concurrencies (C8 json −9 %, code −7 %, math −15 % per pass). The campaign's C8 cells that read lower than the
2026-10-19 row's were one draw of that spread, not a slower pass.

## The folds' spread, decomposed (2026-10-20)

With every rank dumping its collectives' stamps (`DGPP_BUS_TIMELINE_DUMP`)
and the four files aligned by generation, the pairwise symmetric sums give
each path (10–14 µs) and the differences each rank's lateness per
collective with the clocks' offsets cancelled. The arrival spread (latest
minus earliest staging) is 22 µs at the median, 38 µs at p90 on the
attention folds and 106 on the MoE folds, and it is the ranks' compute
imbalance between collectives (correlation 0.98–1.00): no rank is late
systematically, the lateness has no persistence and no position
dependence, the engine's poll loop is under 2 µs, prefetch off changes
nothing, programmatic launches off makes it worse, and locked clocks take
a quarter off the tail. Two populations remain: the Engram collectives
(layers 1 and 14), whose host gather of rows from the 189 GiB mapped
tables page-faults on one rank at 12 % of them (p90 363 µs, p99 1.2 ms of
imbalance; ~0.4 ms a pass summed; with the first Engram layer's consumer
waiting on its own table's rows instead of the whole staging's join, p90
57 µs, p99 0.54 ms, 3 % above 200 µs, ~0.14 ms a pass summed), and a
15 µs-median jitter on every
other collective (p90 54 µs; ~2 ms a pass summed) that is the GPUs'
timing noise on identical work and converts into waiting at each of the
80 collectives — structural, removable only by fewer collectives.

## The folds' path (2026-10-20)

The two collectives a layer (the attention fold and the MoE fold, 35 KB
a rank at C1) cost 31–39 µs each: copy 4 + handshake 15–21 + skew 11–15
+ fold 0.5. The timeline now also keeps each window's shortest handshake
and, per peer, the shortest staging → landed: 1.6–2.0 µs and 0–4 µs, so
no single term is a floor. The path's own latency comes out of the
symmetric sums instead — a peer's landing time as each rank sees the
other is the clock offset plus the path from each side, and the offsets
cancel: rank pairs sum to 26–32 µs, a path of 13–16 µs each way. That is
the no-GDR structure of the fabric: the row crosses PCIe four times
(the GPU's copy into pinned host staging, the NIC's read of it, the
peer NIC's write into host memory, the peer GPU's polled reads) around
a ~4 µs wire time, with the engine's notice and post under 2 µs. The
remaining 11–15 µs of skew is the ranks' arrival spread on identical
work (no rank is last more often than another). The transport is at its
structural floor; the lever left on the folds is the spread.

## The pass's tail (2026-10-20)

After the 40 main layers a C1 pass spends 5.5 ms (13 %): the verify head
(1.0 ms), the sampling chain, the DSpark input projection (fp8
[5120 x 20480], 0.36 ms), the draft block's three layers, the draft head
(1.0 ms) and the pick chain (four sequential collectives with the Markov
bias between them). Three kernels there sat under their limits and were
fixed bitwise: the draft block's attention (the scalar partial kernel,
131–170 µs a draft layer) now takes the decode rows' flash listed form;
the Markov bias (16.5 MB a launch at 134 GB/s cold) loads its rows
warp-coalesced through shared memory (77 µs cold, 17 hot); the
hyper-state window's store and gather copy 16-byte units (29 + 12 µs →
a few). On four nodes the C1 pass went 43.15 → 42.81 ms (code), 36.89 →
36.49 (prose), 49.36 → 47.64 (JSON). The pick chain's four collectives
and the sample verdict kernel (one block a request, 37 µs) are the
tail's next items.

## The lm head's bytes and the window's full form (2026-10-20)

The head is read twice a pass (the draft block's base logits and the
verify's), 331 MB a rank as bf16, and the trace showed it running as the
bf16 tensor-core GEMV at 1.43 ms a launch although `bf16_weights: bf12`
had packed its 12-bit companion: the tensor-core dispatch took every
width from the bf16 bytes. The companion now feeds the decode forms
through a bf12 form of the same kernel — each lane loads its own 24
units of its row's packed super-block into registers and decodes its
fragments into the words the bf16 form loads — so the chain is the bf16
launch's bit for bit at every width (the scalar 12-bit kernel is 1.5 %
faster still, but above eight rows it would give a row a chain that
depends on the rows sharing its launch). On the head's slice, cold:
1,425 → 1,096–1,136 µs (226 GB/s on 248 MB); in the pass 1,432 → 1,023 µs
a launch, with the head's prefetch window streaming the packed bytes, and
the C1 pass 44.50 → 43.15 ms (code), 37.73 → 36.89 (prose) (JSON's three
repeats 46.7–49.5 either way).

The prefetcher's full form became 16 blocks with 8 loads in flight (the
same 232 GB/s cold as 48 × 4 at the light form's SM footprint); the
boundary windows at the full rate still lose about 1 % at C1: the
projections become L2-fed (B 107 → 92 µs a layer) but the gap kernels
that touch DRAM — the mHC stream update (×1.7), the fold's write kernel
(×1.3–1.4), the router dots (×1.2–1.5) — stretch under the window's
traffic. Light stays; the full rate is a lever for after the gap
kernels stop touching DRAM.

At the six-slot batch's row counts the tensor-core companion form stays ahead of the bf16 form it replaces
(head_bench, n 32320 × k 5120, the median of 15 launches): m 12 1,120 µs vs 1,444; m 16 1,156 vs 1,442; m 24 1,274 vs
1,808; m 32 1,355 vs 1,460 (the form's rate falls from 228 to 183 GB/s past 16 rows, the bf16 form's own 2-tile shape
dips at 24). Above 32 rows the dispatch declines the companion and the bf16 form runs as before.

## The decode chain as two resources (2026-10-20)

The C1 pass is one dependent chain that can either stream weights or
run its latency-bound kernels; with an L2 of ~16 MB a layer's window,
the attention weights (45 MB a layer per rank) cannot all hide under
the chain's ~160 µs of gaps, so the pass is judged per layer by a
two-term instrument over the node trace: the segments (mHC site,
projections, selection and attention, output projections, the two
folds, the MoE), each GEMV's achieved GB/s (cold or L2-fed) and each
window's overlap with each segment. On it, three exact levers landed:
`engine.decode_pdl` (programmatic launch edges across the decode graph,
0.3–0.4 ms a pass), a wo_b-only prefetch window opened behind the q/kv
projections at the light rate on the layers without an index source
(wo_b 162 → 378 GB/s; at the layer's start it displaced wq_b from L2, at
the full rate its 48 blocks held the attention kernels off the SMs), and
the paired tensor-core GEMV launch (wq_a with wkv, the two wo_a groups:
one ramp and drain, bitwise). Four nodes, `timed_load` C1, ms per pass /
decode tok/s at equal rows:

| Class | before | after |
|---|---|---|
| code | 46.03 / 75.0 | 44.61 / 77.2 |
| prose | 39.05 / 48.7 | 37.86 / 50.3 |
| JSON | 49.68 / 83.5 | 48.35 / 86.5 |

With the decode tail and the paired projection's norms fused (one
launch per row each, bitwise the separate kernels; 1,753 → 1,524
kernels a pass): code 44.50 / 77.4, prose 37.73 / 50.4, JSON 48.03 /
87.0. A window-only layer now: mHC site 20, projections 107, selection
and attention 34, output projections 88, attention fold 39, FFN site 37,
MoE 520, MoE fold 31 — 892 µs against a structural floor near 800 (the
MoE at the ceiling, the gaps, and the attention bytes the L2 cannot
hold). What the instrument declined: the attention-side window's budget
on wo_a instead of wq_b (wq_b cold +33 µs, wo_a's lines evicted before
use) and a double-buffered chunk GEMV core (slower at every shape).

## Sampled requests: the draft rule and the verify rule (2026-10-19)

`engine.mtp_draft auto` resolves to greedy for this family (the DSpark
block's argmax), and the templates set `engine.mtp_verify = block`.
Four nodes, C1 at temperature 1 (256 tokens × 3 per class; ms per pass /
tokens per pass / decode tok/s):

| Draft / verify | code | prose | JSON |
|---|---|---|---|
| draw / token (the earlier `auto`) | 52.69 / 3.19 / 60.1 | 53.77 / 2.11 / 39.2 | 53.35 / 4.18 / 78.4 |
| draw / block | 52.46 / 3.04 / 57.9 | 54.07 / 2.02 / 37.4 | 53.28 / 4.25 / 79.8 |
| argmax / token | 46.08 / 2.87 / 62.2 | 39.57 / 1.77 / 45.4 | 51.41 / 4.25 / 84.1 |
| argmax / block (shipped) | 47.68 / 3.31 / 69.5 | 39.72 / 1.99 / 49.9 | 51.11 / 4.18 / 84.8 |

A sampled-draft slot verifies the whole block every pass (the scheduled
verify depth is for greedy slots), so the draw's passes cost 53 ms on
every class; the argmax drafts take the schedule, and the block rule
(Sun et al. 2024, exact for the target distribution) adds 10–12 % on
code and prose over the token rule; the template as shipped, re-measured,
66.3 / 49.7 / 81.7 tok/s (sampled runs vary a few percent). Greedy
requests are untouched.

The screened draft head (`engine.draft_head = screen`, the 2-bit rotated
copy at 16 / 32 / 64 blocks of 128) was measured at temperature 0 and
declined: the pass is no cheaper (46.4 ms against 46.0 at 32 blocks; the
scan and the block scoring cost what the 1.42 ms full head did) and
acceptance falls (code 3.45 → 3.23 tokens per pass, prose 1.90 → 1.60,
JSON 4.18 → 3.98), decode tok/s 6–8 % lower. Acceptance per draft
position at temperature 0 (the retire lines): code 86 / 79 / 88 / 86 %,
JSON 94 / 89 / 82 / 78 %, prose 63 / 46 / 24 / 67 %.

## Split-K on the narrow attention projections (2026-10-19)

The decode walk's wkv, wq_a and wo_a GEMVs take the layer's GEMM
workspace so their k range splits across blocks (in isolation at four
rows 86 → 157, 161 → 168 and 136 → 180 GB/s; wq_b, the indexer's wq_b
and wo_b stay unsplit, they fill the grid and lose to the reduce).
Four nodes, `timed_load` C1, ms per pass / rows per pass / decode
tok/s:

| Class | unsplit | split |
|---|---|---|
| code | 47.18 / 3.59 / 76.1 | 45.97 / 3.45 / 75.0 |
| prose | 40.27 / 1.90 / 47.3 | 39.11 / 1.90 / 48.7 |
| JSON | 50.77 / 4.11 / 81.0 | 49.58 / 4.18 / 84.3 |

The pass is 1.2 ms cheaper at equal rows; the split chain is
tolerance-equal to the unsplit one, so the fixed prompts' greedy
transcripts shift (code lost accepted draft rows, JSON gained) — the
numeric path on near-ties, not a bias. Where the C1 code pass stands
(the node trace, four rows, 46.5 ms against a 35.6 ms floor from the
checkpoint's bytes): a DRAM-class kernel runs 93 % of the time; the
MoE at the ceiling (516 µs a layer); the attention GEMVs 244 → ~210 µs
against ~125 after the windows' L2 hits; some 300 µs a layer of latency
work (two folds, the mHC sites, the router, selection and attention,
~350 small launches) covered by the windows at about 100 GB/s; 3.4 ms
of DRAM-idle time. The levers left at C1 are each 1–3 %: the GEMV
cores' small-shape rate (~1.6 ms), the lm head's 75 % of the ceiling
(0.7 ms), the collectives' fixed handshake and skew (~33 µs each), the
small launches.

## The pipelined halves (2026-10-19, tried and discarded)

A captured batch of two or more requests walked as two request-aligned
row halves on two streams a half-layer apart, the folds chained in
record order on every rank, in a free-running form and with a token
that alternated the halves' MoE phases, the MoE grids capped at 24
blocks and uncapped. Measured on four nodes (`timed_load`, ms per pass /
decode tok/s; a single request walks as one either way):

| Shape | as one | halves, free-running | halves, alternating |
|---|---|---|---|
| C2 code (7.03 rows) | 72.16 / 97.4 | 81.37 / 86.3 | 86.28 / 81.4 |
| C2 JSON (8.38 rows) | 75.86 / 110.5 | 86.79 / 96.5 | 92.68 / 90.4 |
| C6 code (20.7 rows, 512 tokens) | 166.00 / 124.8 | 176.49 / 117.4 | 182.83 / 113.3 |

The node traces of rank 0: free-running, the halves' MoE phases ran over
each other for 20 ms of an 88 ms C2 pass (the collective order sequences
only the folds) and covered 5 ms of the other half's latency work;
alternating, each half's attention chain — every attention matrix read
once per half, its latency kernels 1.25–1.4× slower beside the other
half's streaming (the overlap probe: memory-latency inflation, which
stream priorities do not change) — outlasts the half's MoE, so the pass
is the sum of two stretched chains. Capping the MoE grids at 24 blocks
to leave SMs costs the MoE chain 30 % of its rate (`moe_slot_bench
--cap`; 6 % at 48), and strided double-buffered kernels did not recover
it; the collectives double, each with the ranks' two-stream skew. The
form cannot pay on this hardware at any concurrency; at C1 a row split
has the same arithmetic with worse ratios. Removed.

## The decode latency round (2026-10-19)

The C1 pass's non-DRAM time, from the node trace of `0d69d565` (three
rows, about 850–920 µs a layer): the two mHC sites at three launches each
(dots 8.5, finish 9.9, norm 7.3 µs), the scalar window attention at eight
splits (27–29 µs), the fp4 down projection at 73 % of the read ceiling
(840 tiles over the resident blocks ran as whole rounds), and the two
collectives a layer (35 + 41 µs) whose handshake and skew ran in the
open. The small kernels also stretch two- to four-fold under the L2
prefetcher (an RMSNorm at 1.4 or 8.2 µs), and the per-collective skew of
~20 µs is the sum of that variance over the kernels between collectives,
so fewer launches attack it as well.

| Lever | What it does | Trace, per three-row layer | Bitwise |
|---|---|---|---|
| Fused mHC decode site | DeepSeek-V4's form: the finish fused into the dots launch by tickets, the one-rounding norm in it, the comb on a side stream joined at the stream update. Decoder rows keep it in a mixed walk. | a site 26 → 12 µs on the critical path (dots 11.8, comb 8.2 beside) | rounding-level (the fused norm's sum order), the decode audit's class |
| Down kernel tile stealing | `moe_slot_down_fp4_kernel` takes (slot, row tile) units from a counter pair the last block resets; the grid is the device's occupancy. | 180 → 169 µs | yes |
| `engine.decode_window` flash | The 128-key window on `dsa_attn_listed`, split eight ways like the scalar form, on a side stream beside the main source's kernel, joined before the finish. | scalar 28.8 µs in line; the unsplit flash form 36.4 (one block, eight tiles); split, beside the main kernel | the window's fp32 sum order (the decode audit's class) |
| Decode fold split (discarded) | Each boundary's producer as two column halves, the first half's all-reduce on a side stream under the second half; the bus walked up to four generations in flight. | MoE site: first fold 70.6 µs hidden under the 82 µs second half, second fold 27.5; attention site: wo_b's half GEMV 16 µs cannot hide a 28 µs collective | yes |

Measured (four nodes, C1 `timed_load` code / prose / JSON, three repeats,
ms per pass / decode tokens per pass / tok/s):

| Build | code | prose | JSON |
|---|---|---|---|
| `0d69d565` as campaigned (`9350b38a`) | 46.80 / 3.11 / 66.5 | 40.06 / 1.81 / 45.2 | 51.95 / 4.11 / 79.2 |
| this round, as shipped (the output window dropped where it has no cover) | 47.93 / 3.59 / 74.9 | 40.71 / 1.90 / 46.7 | 51.75 / 4.11 / 79.5 |
| the same, the output window at the light rate under the attention (the form before) | 48.55 / 3.59 / 74.0 | 39.54 / 1.90 / 48.1 | 53.34 / 4.11 / 77.1 |
| the same, the output window at the full rate under the attention | 48.55 / 3.59 / 74.0 | 41.50 / 1.90 / 45.9 | 53.09 / 4.11 / 77.5 |
| the same, the output window at the full rate at the layer's start | 51.49 / 3.59 / 69.8 | 44.14 / 1.90 / 43.1 | 55.51 / 4.11 / 74.1 |
| the same with the window scalar (unsplit flash build) | 49.97 / 3.23 / 64.3 | 41.88 / 1.88 / 44.8 | 54.81 / 4.11 / 75.0 |
| the same with the fold split, MoE site | 49.05 / 3.59 / 73.2 | 40.54 / 1.90 / 46.9 | 53.65 / 4.11 / 76.7 |
| the same with the fold split, both sites (unsplit flash build) | 48.08 / 3.23 / 67.1 | 42.54 / 1.89 / 44.4 | 54.18 / 4.11 / 75.9 |

C6 code (one repeat, 512 tokens): 174.8 / 20.7 / 118.5 → 172.5 / 20.7 / 120.1. The
round's passes carry more accepted draft rows (code 3.11 → 3.59, prose
1.81 → 1.90 a pass) with the window and norm forms' rounding class. The
quality set on the shipped binary (`64bb41cd`, the campaign's phase alone):
HumanEval 156/164 (6 truncated), GSM8K 292/300 (3 truncated), extraction
100/100 — against the campaign binary's 157/164, 295/300, 100/100 the same
day and the family's FP8 baseline of 94.5 / 97.7 / 100 %: the rounding
class moved, the quality class did not; the three GSM8K misses are
truncations at the item budget, not wrong answers.

**The dependency the attention lever carried.** The first shipped form of
the window-beside-main lever lost 1.4 ms a pass at JSON against the
campaign binary while every kernel it touched got faster. The per-kernel
trace diff placed it: the 32-block output-projection GEMVs (139 a pass)
ran 28.8 → 49.8 µs each, and 1,507 of their 1,628 slow launches were the
first GEMV after the attention finish, overlapping a prefetch kernel. The
CSA2 layer opens an output window (wo_a and wo_b, ~1.5 MB at the light
rate) under the attention so those GEMVs find their weights in L2; the
window-beside-main form cut a window-only layer's attention phase to
12–20 µs, the prefetch no longer finished under it, and the GEMV queued
behind its own weights. Four placements measured (JSON ms/pass, 4.11
rows): under the attention at the light rate 53.34; at the full rate
53.09 (the GEMVs recover, the attention's own cache reads slow from 16 to
29 µs); at the layer's start at the full rate 55.51 (it slows the
DRAM-bound projections instead); no output window where no cover remains
51.75, the GEMVs reading wo from DRAM at p90 34 µs against the campaign
binary's 36. The window budget was not the cause (18 MiB best; 12, 8 and
6 MiB each 0.5–0.9 ms worse), and `engine.l2_prefetch_window_mib` now
governs this family's windows (the 18 MiB constant is gone; the template
pins 18). The lesson is the method: a lever that shortens a phase must
re-size or remove what that phase was hiding, in the same experiment.

Why the fold split lost, from the bus timeline on four ranks (rank 0
medians per collective, µs): off — 93 collectives a window, copy 4.8 +
handshake 22.4 + skew 24.0 + fold 9.0 = 64.2; the MoE-site split — 136 a
window, copy 8.4 + handshake 23.2 + skew 20.6 + fold 5.8 = 59.5. The
added collectives pay their handshake and skew in full, only the fold
shrinks with the half payload, and every collective's staging copy slows
under the second half's DRAM traffic. A split pays only once a
collective's fixed cost is small: the lever is the handshake (the
two-sided send, the CPU-posted doorbell), not more collectives.

## The prefill selection at long context (2026-10-18)

Prefill cost per token is flat in everything but one piece: on every
index-source layer, each query of a chunk is scored against every visible
compressed entry (the candidate layer selects from all of them, the other
index layers narrow the top-k to the candidate pool afterwards). That scan
ran as a cuBLASLt fp8 GEMM into a per-head fp32 dot buffer, a fold kernel
over the buffer and the selection kernels. The buffer's 64 MB budget held
four query rows at 128K entries, so a 4096-row chunk ran about a thousand
tiles, each re-reading the whole key set and writing and re-reading 64 MB
of dots: a chunk's cost grew with the square of its position and the
per-token cost accelerated with the prompt (the "before" column below).

The scan is now one fused kernel: a query row's 32 heads against 128
entries per CTA on the fp8 tensor cores, the heads folded in the mma
fragments and only the per-row logits written, with the logits tile sized
by the retired budget so a row tile reads the keys once for 128 rows at
128K entries (the keys stay in L2 across them). The selection kernels are
unchanged. The scorer reaches 31 TFLOP/s on the microbench (35 µs per
query row at 128K entries), about a third of the mma.sync rate, with its
synchronous key staging as the next lever; the restricted layers could
also score only the candidate pool, as decode does, instead of every
entry. Both are follow-ups; the quadratic term is gone.

| Context | Prefill before the fix (s) | ms per token |
|---|---|---|
| 32K | 20.3 | 0.62 |
| 64K | 44.5 | 0.68 |
| 128K | 113.6 | 0.87 |
| 256K | 387.9 | 1.48 |
| 512K | 1947 | 3.71 |

The context rows after the fix come with the next campaign; by the
scorer's measured rate the scan adds about 0.14 s per index layer to a
4096-row chunk at the end of a 512K prompt against several seconds before.

## The prefill's dense projections and the fabric's two lanes (2026-10-18)

A kernel ledger of one 32K prefill (nsys on rank 0, 96 % GPU busy) put
18 % of the time in the streaming decode GEMM: the family ran every dense
projection on it whatever the row count, and a 4096-row chunk re-read
each fp8 weight once per 128 rows, 7,562 launches at 108 µs, about 25
TFLOP/s. The prefill walk now runs its dense projections on the
fp8-weight tile GEMM, which gained the family's 32 × 32 scale grid
(`engine.prefill_dense`, default `tile`); decode rows keep the streaming
form, whose rows are bitwise the row alone. The two forms place the same
weights and accumulate in another order, within fp32 rounding (3.4e-6 on
the kernel test); the forward parity gates pass on the tile form, and the
model test gates the forms against each other.

The same day the site named the ConnectX-7 port's second PCIe function
as a second bus lane (`DGPP_ROCE_DEVICES="rocep1s0f0 roceP2p1s0f0"`,
[networking](../networking.md)): the bus stripes the prefill's bulk folds
across both while the decode class stays on the first.

| Cold prefill, s at 512 / 2K / 8K / 32K tokens | | tok/s at 32K |
|---|---|---|
| One lane, streaming projections (the fused scan) | 0.523 / 1.619 / 5.152 / 19.591 | 1,620 |
| Two lanes, streaming projections | 0.514 / 1.572 / 4.715 / 17.916 | 1,770 |
| Two lanes, tile projections | 0.480 / 1.415 / 4.045 / 15.328 | 2,080 |
| Two lanes, tile projections, the folds overlapped | 0.464 / 1.324 / 3.699 / 14.234 | 2,240 |
| … and the attention kernels at the byte floor | 0.418 / 1.097 / 2.801 / 10.623 | 3,000 |
| … and the Engram rows staged a chunk ahead, the folds begun behind events | 0.421 / 1.097 / 2.771 / 10.174 | 3,130 |
| … and the Engram folds by row block (shipped) | 0.413 / 1.066 / 2.717 / 10.033 | 3,170 |

Decode is untouched by all of these. The ledger's remaining prefill
shares on the tile form were the listed attention over the selected
entries about 19 %, the bulk folds about 14 %, the MXFP4 experts about
17 % at 58 TFLOP/s, the window attention about 9 % — the next two
sections take the folds and the attention.

The folds then went under compute (`engine.prefill_fold_overlap`, default
`true`): a walk of 512 rows or more runs each layer as two row blocks,
the boundary a row of the request's block grid nearest the middle, so
block A's attention fold lands under block B's attention, B's under A's
stream update, mHC site and router, A's MoE fold under B's expert
accumulation and B's MoE fold under the next layer's block A — the
DeepSeek-V4 walk's form. The Engram layers and the DSpark target layers
land the pending fold first (their gate and stream mean read every row),
and under the bounded prefill the encoder layers overlap while the
decoder's prologue lands the pending fold before it publishes. The walk
is bitwise the one-block walk (the attention takes the second block as it
takes any chunk; the MoE's phased form runs the same kernels per row),
gated in `dsv41_tp_test` on the exact and the bounded walk, a group cut
inside a span, and a loopback world of 2 through the bus's asynchronous
bulk folds. The table's last row is the campaign leg; a matched probe of
one prompt set five times, the key on against off on the same binary,
read −4.3 / −5.0 / −6.4 % per token at 512 / 2K / 8K and decode level
(code 48.0 against 47.6 ms per pass, prose 41.7 against 41.6).

## The attention kernels (2026-10-18)

The listed attention over the selected entries (`attn_flash_kernel` in
`dsa.cu`, shared with GLM-5.3's DSA) was the walk's largest kernel after
the dense projections moved: 5,406 launches at 646 µs, 3.5 s of a 32K
prefill, at about 3 TFLOP/s and 50 GB/s. Nsight Compute on an isolated
bench at this geometry (16 heads, latent 512, fp4_block entries, 512 per
query in 4 splits, 128 rows) showed why: 196 registers and 87 KB of
shared memory held one block per SM, each scheduler had two warps and
issued once per 4.8 cycles, and 22 of 32 lanes were active. Three
changes, each bitwise (the bench's partial-output checksums identical at
four shapes): the listed addresses resolved once per 512-token round into
shared memory instead of three dependent loads per thread per row (750 →
630 µs); the e2m1 decode as a nibble table instead of a switch that
diverged a warp eight ways per element, 64 elements per thread per tile
(→ 300 µs); two blocks per SM with the Q~ fragments in registers, the
exchange sized by the tile and the fetch in four passes (→ 206 µs,
183 GB/s of the 234 GB/s read ceiling). Two blocks hold for this family's
listed kernel, which fits the two-block register budget without
spilling; the 512 + 64 geometry other families run keeps one block
(compiled to two it spilled, and that build faulted DeepSeek-V4's mixed
walk in the shared prefill test while every spill-free build ran clean).
The remaining lever on it is the fp32 split partials (four per row), a
numerics change left alone.

The window source then moved from the scalar partial kernel (287 µs per
128-row tile, 12 % of the walk) onto the same listed kernel
(`engine.prefill_window`, default `flash`): the window scratch is a
one-block cache and the slot lists are per-row selections, the kernel's
inputs. Its scores sum in the mma's fragment order, so the model test's
decode audit walks the scalar form — the decode window's kernel — and a
separate gate holds the flash form within a flip's move of it; the
forward-parity gates pass on the flash form. In the profiled 32K walk
the listed kernel runs at 193 µs per launch and the window at 68.

The 32K prefill's ledger after this (the kernels of its 10.9 s, the bus
bulk kernel overlapped on its own stream): the MXFP4 experts 2.7 s (the
gate/up chain 1.6, the down chain 1.1), the fp8 dense tile GEMM 1.1 s,
the listed attention 0.93 s, the mHC stream updates 0.48, the expert
accumulation 0.45, the router 0.40, the window attention 0.36, the mHC
coefficients 0.34, the selection 0.26; about 2 s of the wall is gaps
between kernels (the overlap's host drains and launch latency). The next
levers in order: the experts (W4A8 against today's bf16 activations), the
dense GEMM on native fp8 activations, and the gaps.

## The walk's gaps (2026-10-18)

With the attention kernels at the floor, the 32K walk's main stream was
busy 8.7 s of its 10.9 s: 1.4 s were stalls of 110–160 ms per 4,096-token
chunk at the first Engram layer — the mmap'ed tables' gather, forked at
the walk's start and joined at layer 1, storage-bound with 16 threads and
page hints already — and 0.7 s were folds longer than the work under
them. The Engram rows are now staged a chunk ahead (`engine.ngram_prestage`,
Qwen's form: the next chunk hashed on the device behind this chunk's
last three tokens, gathered on the side stream, claimed by the next
chunk when its identity matches; the scheduler's batch walk names a lone
prompt's chunks so the staging can claim under it), and the overlap's
folds begin behind the producers' events with the other block's work
already enqueued. Cold prefill 0.421 / 1.097 / 2.771 / 10.174 s (−4 % at
32K; the shorter prompts are one or two chunks and stage nothing ahead).
The profiled walk's gaps fell to 1.4 s: one first-chunk gather of 90 ms,
the two Engram folds per chunk at 31 ms each (210 MB over the wire), the
MoE fold of block A exposed under block B's accumulation alone (about
0.45 s), the attention fold of block B under block A's site and router
(about 0.29 s). The Engram folds then went by row block too (A's under
B's projection, B's under A's gate and attention; bitwise): 0.413 /
1.066 / 2.717 / 10.033 s. What remains exposed is wire time under too
little work: a block's fold of hidden x 2 bytes per row moves at about
7 GB/s per rank on the paced bulk path, and block B's accumulation is
shorter than block A's fold.

## Measured (2026-10-20, four nodes): the campaign of record

The full row set of [benchmarks](../benchmarks.md) on source `283a52f7`, binary `f4e9e9cdb16acf86`, in one
`serve_campaign.py run` with the five DSpark mode boots (10:04–11:03 UTC; every rank at zero swap before and after,
zero cached tokens in the 12 cold probes, matching operation streams, the plain boot's transcripts identical to the
default's in 5/5 classes, 9/9 API checks, isolation identical). These are the rows the benchmark doc tables.

| | 2026-10-19 (`0d69d565`) | 2026-10-20 (`283a52f7`) |
|---|---|---|
| C1 decode tok/s (prose / code / JSON / math / chat) | 44.8 / 66.5 / 78.1 / 73.7 / 50.8 | 53.5 / 80.0 / 89.4 / 87.5 / 55.0 |
| Decode tok/s by concurrency, C1 / C2 / C4 / C8 | 44.8–78.1 / 63.9–111.5 / 85.9–119.2 / 76.4–122.8 | 53.5–89.4 / 75.5–121.9 / 94.7–130.3 / 83.8–139.0 |
| Wall tok/s, C1 / C2 / C4 / C8 | 43.6–73.9 / 62.2–105.8 / 83.8–114.3 / 74.6–117.5 | 52.0–84.6 / 73.4–115.8 / 92.2–125.2 / 81.7–132.7 |
| Cold prefill s, 2K / 8K / 32K | 1.079 / 2.713 / 10.030 | 1.006 / 2.638 / 10.024 |
| Context decode ms per pass (tok/s), short / 32K / 64K / 128K / 256K / 512K | 49.38 (67.1) / 51.05 (63.2) / 52.04 (57.0) / 55.15 (57.8) / 59.02 (48.5) / 66.36 (46.3) | 43.12 (73.0) / 44.82 (72.9) / 45.64 (69.5) / 48.49 (64.9) / 51.99 (53.3) / 60.35 (51.5) |
| Context prefill s, short / 32K / 64K / 128K / 256K / 512K | 0.296 / 9.699 / 20.011 / 43.780 / 103.943 / 303.423 | 0.263 / 9.684 / 19.999 / 43.577 / 103.139 / 301.219 |
| Decode modes, C1 tok/s: plain / DSpark 1 / 2 / 3 / 5 / adaptive ≤4 | — / — / — / — / — / 44.8–78.1 | 41.7–41.9 / 50.2–59.5 / 50.1–76.5 / 47.8–82.6 / 39.6–96.7 / 53.5–89.4 |
| HumanEval / GSM8K / extraction | 157/164 / 295/300 / 100/100 | 157/164 / 294/300 / 100/100 |

The day's decode levers (the folds' path and spread, the pass's tail, the head's 12-bit companion, the window's
form) carry every cell: C1 +12–20 %, C4 +9–12 %, the context matrix −9..13 % a pass, the cold prefill level to
−7 %. The DSpark modes row is new: the fixed depths bracket the adaptive schedule (depth 5 wins the code and JSON
classes at 96.7 tok/s, loses prose at 39.6; the schedule takes 53.5–89.4 without a per-class choice). The C8 cell
(six live, two queued) read 139.0 at the top of its spread this time; the A/B above is the comparison that holds.

## Measured (2026-10-18, four nodes)

Both budgets ran the full row set of [benchmarks](../benchmarks.md) on the
same binary (source `ea4fcdf9` plus this working tree, binary
`9d684e2d10ad3573`; every rank at zero swap, zero cached tokens in the cold
probes, matching operation streams, an idle fabric), one campaign each:
the template as shipped (`index_topk` 512) and the template with
`index_topk` 2048. The 512 rows are the ones tabled in the benchmark doc;
the 2048 set is recorded here only, since no shipped template carries it.

| | `index_topk` 512 (shipped) | `index_topk` 2048 |
|---|---|---|
| C1 decode tok/s (prose / code / JSON / math / chat) | 43.2 / 64.9 / 73.9 / 68.1 / 45.1 | 43.4 / 64.0 / 74.4 / 68.6 / 45.2 |
| Decode tok/s by concurrency, C2 / C4 / C8 | 60.7–106.2 / 81.1–119.2 / 71.1–124.2 | 61.0–106.4 / 82.0–119.9 / 70.7–126.8 |
| Wall tok/s, C1 / C2 / C4 / C8 | 42.1–70.2 / 59.2–101.1 / 79.3–114.7 / 69.7–119.3 | 42.3–70.8 / 59.5–101.3 / 80.1–115.4 / 69.3–121.7 |
| Cold prefill s, 512 / 2K / 8K / 32K tokens | 0.528 / 1.594 / 5.245 / 20.790 | 0.526 / 1.744 / 7.007 / 30.745 |
| Context prefill s, 0 / 32K / 64K / 128K / 256K / 512K | 0.309 / 20.2 / 44.5 / 112.9 / 386.9 / 1947.2 | 0.307 / 30.2 / 65.5 / 159.2 / 499.5 / 2334.2 |
| Decode ms per pass at those contexts | 49.77 / 53.78 / 53.42 / 56.11 / 61.01 / 67.07 | 51.03 / 56.79 / 58.28 / 58.74 / 63.03 / 69.88 |
| HumanEval / GSM8K / extraction | 157/164 / 294/300 / 100/100 | 156/164 / 295/300 / 100/100 |
| API, plain/default transcripts, isolation | 9/9, 5/5 classes, identical | 9/9, 5/5 classes, identical |

The widened budget leaves short-context decode level (every C1–C4 class
within ±1 %, C8 within the sweep's run-to-run spread), but every query on
a compressing layer now reads four times the compressed entries: cold
prefill runs 9 % longer at 2K, 34 % at 8K and 48 % at 32K, the context
prefills 17–49 % longer, and the decode pass at 32K–512K of context 3–8 %
slower. The quality set does not separate the two (one HumanEval and one
GSM8K item each way), so the budget's benefit on long inputs is not
measured here; the template therefore ships the checkpoint's 512 and the
key is the opt-in with this price.

Against the DeepSeek-V4.1-Flash 1M row of 2026-10-07 (source `07a32fb`),
the 1017 weights decode at the same rates (C1 43.2–73.9 against 41.8–78.2
by class, the per-class differences being the drafter's acceptance on the
re-trained weights; the decode pass at 0–512K of context 49.8–67.1 ms
against 50.9–66.6) and answer the quality set alike (157/164 against the
128K row's 160/164 with five answers truncated at the eval's token limit
against none; GSM8K 294 against 296; extraction 100). Cold prefill reads
6–8 % longer (1.594 / 5.245 / 20.790 s against 1.472 / 4.946 / 19.545).
That is master's drift between the two dates, not this port: the
committed release binary of `aa88b8f7` (built without this working tree)
prefills the same template at the checkpoint's budget in 0.531 / 1.595 /
5.216 / 20.714 s, within 0.5 % of the 1017 binary, and the CSA2 kernels the
512 path runs are byte-identical SASS to the committed ones (the 2048
tile is a second instantiation).

## License and attribution

The model, its weights, tokenizer and prompt encoder are DeepSeek's (MIT);
see the upstream model card for its terms. This file describes only the
engine's support and adds no weights.
