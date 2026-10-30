# DGPP moonshot

Written 2026-10-30 against the rows of [the benchmarks](docs/benchmarks.md)
at revision 78038ed4. This document asks what twelve months could make of
the engine on the hardware it runs on: four DGX Sparks with one 200 Gb RoCE
port each. It is a plan with numbers, not a record. Every "today" below is a
measured row. Every target is derived from a measured floor and stays an
estimate until a leg on the fabric confirms it under the method of record in
the benchmarks document.

The program is three pillars. The decode pass runs with no host and no
kernel boundary inside it, so a step costs what its bytes cost. Each read of
the weights yields more tokens, through the drafters the checkpoints already
ship and a schedule that spends verify rows where they pay. Prefill runs at
the tensor cores' compute bound and shards the sequence across nodes.
Everything proposed is lossless. Drafts never change output and the
exactness gates stay as they are. Trained draft heads, a heterogeneous rank
and a schedule compiler were considered and set aside, and the last section
records why.

## The headline

| Cell | Today | Target | Multiplier | What binds at the target |
|---|---|---|---|---|
| Qwen3.8-Flash-Next NVFP4, one Spark, C1 code | 49 | 85 to 100 | 1.7 to 2x | expert fan-out at four to six verify rows, the MTP head's acceptance |
| Qwen3.8-27B NVFP4, one Spark, C1 code / prose | 43 / 22 | 70 to 80 / about 32 | 1.6x | DFlash2 acceptance at block 16, the drafter's own bytes |
| GLM-5.3-Flash NVFP4/FP8, four Sparks, C1 | 59 to 65 | 80 to 90 | 1.4x | fan-out plus about 95 collectives a pass |
| GLM-5.3 NF4I8, four Sparks, C1 | 27 to 31 | 36 to 40 | 1.3x | about 3 GB per rank per verify row |
| DeepSeek-V4.1-Flash EXL3 3.5 bpw, four Sparks, C1 top class | 134 | 180 to 200 | 1.4x | pass at the byte floor, depth per stream |
| GLM-5.3-Flash NVFP4/FP8, four Sparks, C8 | 104 to 113 | about 160 | 1.5x | rows free in the projections, fan-out saturated |
| Cold prefill 32K, GLM-5.3-Flash, four Sparks | 20.5 s | 5 to 6 s | 3.5x | fp8 tensor-core compute bound |
| Cold prefill 32K, Qwen3.8-Flash-Next, one Spark | 18.1 s | 9 to 10 s | 1.9x | the same |

Decode figures are the engine's decode steps only, in tokens per second, as
the benchmarks define them. Prefill is cold, in seconds. The C8 row is the
least certain: it depends on the projection GEMMs reading their weights
once for every row of a batch, which they do not do today past four rows.

## Pillar 1. The persistent step

**Status 2026-10-30.** Phase 0 and phase 1 of [MOONSHOT-1.md](MOONSHOT-1.md)
ran: the floors were re-derived from the kernels' own accounting with their
row counts, today's engine measured at 65 to 85 percent of them at C1 rather
than 60, and the C1 targets became 1.09x to 1.27x by family (section 1 of
that plan). A persistent kernel with ceiling-rate consumers runs every
family's layer at 98 to 106 percent of the floor; today's GEMV consumers
inside it reach 60 to 80 percent; the first tensor-core consumer, over a
tile-major expert layout, streams at 94 to 96 percent standalone, the fp8
block-scaled form at 92 to 96, and the integrated prototype with both sits
at 104 to 109 percent of the floor under the fused chain at every shape in
the engine's own worker count: 1.21x the graph engine at C1 code, 1.29x at
C1 prose and 1.25x to 1.39x at C8, level with it under today's unfused
chain. What remains is phase 2's
work distribution and the fused chain of the design, each with a number
against it. By the evening, phase 2 is repo code behind `engine.decode_engine`: the stream plan and its DeepSeek-V4.1 generator (which corrected the floors to 36.3 and 22.9 ms, the head's two reads having been missed), the arena placement with its layout-keyed image, the persistent streamer with its lead, governor, deadlines and poison, the step protocol's descriptor and verdict rings and device token feed, and the MXFP4 tensor-core expert consumer running inside it, exact against a double-precision reference, with the first eight layers at 96 to 98 percent of their byte floor beside production. See `docs/step_engine.md`.

No host and no kernel boundary inside a pass.

**Evidence.** The two-term instrument of 2026-10-20 on DeepSeek-V4.1-Flash at
C1: a window-only layer spends 781 µs in its DRAM phases and 174 µs in the
gaps between them, and the GEMVs inside the DRAM phases run 10 to 50 percent
below their isolated rate because every small dependent launch drains the
memory pipeline and the next one refills it. About a thousand launches under
15 µs cost 5 ms a pass. Ninety-five collectives at about 44 µs of fixed cost
each cost 4 ms more. The pass sits at 60 percent of its 28.8 ms byte floor.
Full GLM-5.3 pays 162 collectives at 49 µs, 7.9 ms of a 66 ms step.

**Design.** One persistent grid per rank per step, driven by a device work
list built for each step shape. The probes already fix the rules it has to
obey:

- Streaming CTAs are capped near 24 so the latency CTAs beside them stay
  unstretched. The overlap probe measured the pair costing the maximum at
  16 to 24 blocks and the sum at 32 and above.
- Dependencies are phase counters, not tile counters. A dependent round trip
  through global memory costs 1 to 8 µs beside a prefetch window.
- The fold's receive is one sentinel warp per peer polling host memory.
  More pollers slow the NIC's writes.
- The host proxy keeps the posting role. GPU-rung doorbells are impossible
  on this SoC, and host-pinned memory is the only memory the NIC can see.
- Prefetch windows are entries in the same work list, sized from the 24 MB
  the L2 offers.

**The collective.** The write transport brought the transport's own fixed
cost to about 15 µs. What remains is arrival skew from per-rank jitter, and
the jitter sources are the launches and the window-stretched small kernels
this pillar removes. The skew study's floor is about 22 µs a collective. At
that round cost the qkv_a sharding closed on 2026-10-09 reopens and pays
about 3 ms a step on full GLM.

| Family at C1 | Pass today | Pass at the persistent step | Mechanism |
|---|---|---|---|
| DeepSeek-V4.1-Flash, four Sparks | 47 ms | 35 ms | gaps gone, GEMVs at 90 percent of isolated, collectives 44 to 22 µs |
| GLM-5.3, four Sparks | 66 ms | 50 ms | collectives 7.9 to 3.6 ms, attention GEMVs to the ceiling, qkv_a sharded |
| Qwen3.8-Flash-Next, one Spark | 37 ms | 26 ms | recurrence and norm chains inside the stream, draft chain under 1 ms |

**The counter-evidence.** Two ledger entries argue against this pillar. The
NVFP4 headroom round rejected persistent and PDL forms, and the decode
ledger records that fusing small kernels at 97 percent busy does not pay.
Both measured time between kernels. The 2026-10-20 instrument measured the
loss inside each GEMV's duration, which is a different quantity and the one
this pillar attacks.

**First experiment.** One layer's MoE region, gate-up through down through
fold, as a single persistent kernel on DeepSeek-V4.1, with the in-situ GEMV
rate compared to the isolated rate. If the rate does not move, the pillar is
dead in a week.

## Pillar 2. Tokens per byte

More tokens from each read of the weights, with the drafters the checkpoints
already ship: the MTP layers, DSpark and the published DFlash2 drafter. This
is the only multiplicative lever on a bandwidth-bound machine.

**The wall.** On MoE families each verify row routes its own experts, so
rows cost bytes. The depth study's break-even is p1 times p2 above 0.28
times one plus p1, and depth 2 pays only on code and JSON today. The wall is
softer than it looks once a draft step is cheap and the depth is chosen per
request. Qwen3.8-Flash-Next on one Spark reads about 6.2 GB a pass at four
verify rows and 9.3 GB at eight, so bytes are roughly 3.5 GB plus 0.75 GB a
row. The MTP head's acceptance on code is 0.85 for the first draft and 0.62
for the second in the depth study. Later positions are assumed at 0.55,
0.50, 0.45, 0.40 and 0.35 and must be measured. With the pass at the byte
floor plus 5 ms of latency work:

| Depth | Verify rows | Bytes a pass | Tokens a pass, code | Pass | tok/s |
|---|---|---|---|---|---|
| 1 | 2 | 5.0 GB | 1.85 | 25.8 ms | 72 |
| 2 | 3 | 5.75 GB | 2.38 | 29.0 ms | 82 |
| 3 | 4 | 6.5 GB | 2.67 | 32.1 ms | 83 |
| 4 | 5 | 7.25 GB | 2.81 | 35.2 ms | 80 |
| 5 | 6 | 8.0 GB | 2.88 | 38.3 ms | 75 |
| 7 | 8 | 9.5 GB | 2.91 | 44.6 ms | 65 |

Three things fall out of the table. The depth-1 cell alone has 1.4x of
headroom to its floor, which is pillar 1's work. A fixed depth of 2 or 3
adds another 15 percent. A depth chosen per step from the drafter's
confidence spends the deep rows only on the spans where they land, and the
one external A/B of that rule on this hardware class gained a further
15 percent over the best fixed depth. Together they put the cell at 85 to
100. Today a draft step on one Spark costs about 4 ms, the head read plus
the MTP layer, so a depth-7 chain would cost 28 ms before any verify, and
that cost is the first thing to remove.

**The levers, in build order.**

- **A draft step under a millisecond.** The 2-bit head screen of 2026-10-13
  cut the head read. The MTP projections go to fp8 at load. The DFlash2
  drafter's weights, bf12 today, go to fp8 for the draft forward, which the
  rule for draft-side work allows because drafts never change output, with
  acceptance re-measured per class before it ships.
- **Depth as a per-step decision from a measured cost table.** The
  drafter's confidence stops the chain on the device, and the engine spends
  rows where expected tokens per millisecond is highest. The key
  `engine.mtp_schedule_step_ms` already carries the curve for DSpark and
  DFlash2 in [verify_schedule.hpp](src/engine/verify_schedule.hpp). It has
  to reach every family, read the drafter's confidence, and run per slot
  rather than per batch. Qwen's graph variants cap at depth 3 today and go
  to 7 so the schedule has room to spend.
- **Speculation off per request when it does not pay.** Acceptance of the
  second draft fell to 41 percent at a 7K context on GLM-5.3-Flash. A
  request whose measured cost per committed token exceeds plain decode
  drops to depth 1 or to plain and is re-probed later. Lossless, and it
  protects prose at long context.
- **Rows free in the projections.** On GLM-4.7 a batched depth 2 lost to
  depth 1 under concurrency because every four-row GEMV chunk past the
  first re-read the attention projections. Wider GEMV chunks or a
  row-independent tensor-core form for the projections make extra rows free
  at C2 to C8, and that is what makes speculation under concurrency pay and
  lifts the C8 cells.
- **Trees where the drafter is split.** When the MTP head's top two
  candidates are near even, two rows beat one, and the same rows-to-bytes
  table prices the tree. Block verification handles the acceptance rule.
  Tree masks in the verify attention are the missing piece.
- **Rows at p equal to one.** Agent traffic is full of verbatim spans: a
  file echoed back in an edit, a repeated tool-call envelope, a JSON
  skeleton the schema dictates. Context lookup drafts exist in
  [lookup_draft.hpp](src/engine/lookup_draft.hpp). Grammar drafts cost
  nothing to propose. A 16-row verify that lands 16 tokens is about 1.07
  tokens per gigabyte, nearly twice the model drafter's best, and the
  schedule above decides when to spend those rows.
- **Dense block 16.** On the 27B the verify rows are free until compute
  binds near 64 rows. The DFlash2 block goes from 7 to 16, and the pass
  goes to its floor: the drafter's forward is 2.8 GB in bf12 and about half
  that in fp8, and the one-Spark pass today is 119 ms against a floor near
  85 to 92 ms with the bf12 drafter. Code at 70 to 80 and prose near 32.

**Exactness.** Greedy transcripts are already byte-identical across plain,
depth 1 and depth 2, and the ratio rule with block verification keeps the
sampled path exact. Nothing in this pillar touches the verify's math. A
drafter and its schedule change only which tokens are proposed.

## Pillar 3. Prefill at the compute bound

The engine leads on one Spark and trails by 1.5x to 2x on two and four,
and the reason is structural: its prefill shards weights while the leading
recipe shards the sequence and overlaps the all-reduce. Sequence parallelism
at four nodes is the one lever in this document with a measured external
result on the same hardware class, 2.0x to 2.2x.

On top of it, the fp8 channel GEMM runs at 139 TF against a compute bound
the kernel model puts near 185 TF, so a third is left on the GEMMs.
Checkpoints whose native recipe is W4A4, like the QAD export, get an e2m1
GEMM that doubles the fp8 rate and stays lossless relative to the recipe.
The attention prefill rounds on the 27B continue toward their 94 and 126 TF
ceilings. The second QSFP port is cabled and its functions come up, which
doubles the fabric's bulk bandwidth for the sequence-parallel exchanges.
Multicast stays parked, as measured on sm_121a.

| Prefill cell | Today | Target | Where it comes from |
|---|---|---|---|
| GLM-5.3-Flash, four Sparks, 32K | 20.5 s | 5 to 6 s | sequence parallel 2x, GEMMs plus attention 1.7x |
| Qwen3.8-Flash-Next, one Spark, 32K | 18.1 s | 9 to 10 s | GEMMs to the bound, attention rounds |
| Qwen3.8-27B, one Spark, 262K | 336 s | about 180 s | attention kernels dominate at this length |
| An agent's 100K cold context on GLM-5.3-Flash, four Sparks | about 65 s, estimated from the 32K row | about 18 s | the same levers at scale |

The last row is the one the coding agents that use this server feel.
Prefill that costs half as much per chunk also gives the latency-adaptive
controller twice the prefill per decode gap, so the mixed-workload latency
improves with it.

## Method: instruments before levers

Each pillar's claim is a budget read off an instrument, and the instrument
comes first on every family. The two-term instrument that split a
DeepSeek-V4.1 layer into DRAM phases and gaps runs on every family, so each
family's DRAM-idle time is known before pillar 1 touches it. The
rows-to-bytes and rows-to-milliseconds tables are measured per family and
per class before pillar 2 spends a row, because the depth schedule is only
as good as its cost table. Pillar 3's GEMM and attention rates are read
against the compute bound on the shipped shapes. A lever that cannot name
the budget it draws on does not get a fabric leg, and a leg that exposes a
slow path stops the campaign until the path is fixed, as the method of
record requires.

## What the hardware will not give

- **GPU-driven collectives.** The SoC does not route GPU accesses to PCIe
  BARs, so the GPU cannot ring a doorbell, and GPUDirect RDMA is closed. The
  host proxy stays and the per-collective floor is about 22 µs. Nothing
  above assumes otherwise.
- **A tier faster than LPDDR5x.** L2 is 24 MB and the prefetch windows
  already take the few percent it offers.
- **Fewer bytes losslessly.** The entropy study put NVFP4 at 3.95 bits of 4
  and bf12 at the bf16 floor. Formats are done. Bytes fall only through
  more tokens per read.
- **Two chains beside each other.** The memory controller is one queue.
  Latency kernels stretch 1.25x to 1.4x beside a saturating streamer, which
  is why the pipelined halves lost. Overlap must put DRAM-idle work under
  DRAM-busy work, never two chains side by side.
- **Full GLM past about 40 tok/s single-stream on four Sparks.** Each
  verify row routes about 3 GB per rank. The floor at depth 2 is about
  62 ms and the collectives at their floor add 3.6 ms. Eight Sparks would
  give 1.6x to 1.8x at C1, and only after pillar 1, since the fold and the
  shard widths both get worse.
- **Trillion-parameter models from NVMe.** Experts streamed from NVMe put
  the step in the hundreds of milliseconds. That is a demonstration, not a
  product that is fast.
- **TMA multicast on sm_121a.** Measured unusable. It stays parked.

## Set aside

**Trained draft heads.** A head trained on the engine's own traffic would
lift the MTP head's code acceptance from about 0.85 toward 0.9 on every
position and hold it there deep in the chain, which the rows-to-bytes table
turns into 100 to 120 on the one-Spark Flash-Next cell and 100 on the 27B.
It is out by decision: the program serves checkpoints as published,
drafters included. The ceiling it would have added is recorded here so the
targets above are read as the no-training ceiling.

**A heterogeneous rank.** A world whose members differ in bandwidth and
capacity, with each shard sized from a boot-time measurement rather than
evenly. The specific case was the RTX 5090 at about 1,800 GB/s holding the
hottest experts of Qwen3.8-Flash-Next or 70 percent of the 27B beside one
Spark, which the physics put at 3x to 5x on the one-Spark cells with a RoCE
link between the boxes. The generalizable form is any consumer Blackwell
card: sm_120 shares the GB10's kernel ISA, so the NVFP4, fp8 and TMA forms
run unchanged, and a Spark beside an RTX box on one switch is the natural
pair. Older architectures would need new kernel forms for every format.
Parked, not pursued. If it is ever revisited, the one-day measurement that
decides it is the expert-hit cumulative distribution per layer on the agent
op streams.

**A schedule compiler and a digital twin.** A declared family compiled into
its step schedule from a cost model, with the same model run offline to
evaluate levers without the fabric. Worth building for a large team. For
this program the method section keeps the part that matters: measure the
budget before spending it.

## Twelve months

The order is set by what each quarter must learn, not by what is easiest.

| Quarter | Ships | Decides |
|---|---|---|
| First | The two-term instrument on every family, the per-family cost tables, confidence-stopped depth per step on every family, Qwen graph variants to depth 7, sequence-parallel prefill at two nodes, the one-layer persistent MoE region | whether the persistent step moves the in-situ GEMV rate, the depth schedule's gain per class |
| Second | The persistent step on DeepSeek-V4.1 and Flash-Next, fp8 GEMMs at the compute bound, sequence-parallel prefill at four nodes, DFlash2 block 16 with fp8 drafter weights, lookup and grammar rows | the one-Spark cells at their targets, or the reason they are not |
| Third | The persistent step on GLM-5.3-Flash and full GLM with the collective at 22 µs and qkv_a sharded, the W4A4 GEMM for native recipes, tree drafts, rows free in the projections | the four-node cells and C8 |
| Fourth | The attention prefill rounds at long context, the campaign of record on every template, the 1.0 release | the headline table, measured as the method of record demands |

## The first ninety days

Mostly measurement. Four measurements decide the year.

1. The persistent one-layer MoE region's in-situ GEMV rate against the
   isolated rate decides pillar 1.
2. The acceptance curves per class at depths one to seven with the shipped
   drafters, set against the rows-to-bytes table, give the gain of the
   depth schedule per class and replace the assumed positions in pillar 2's
   table with measured ones.
3. The sequence-parallel prototype at two nodes confirms the external 1.6x
   on this fabric.
4. The two-term instrument on every family gives each family's DRAM-idle
   budget, which sizes pillar 1's claim family by family.

## If only one pillar

The persistent step. It closes a measured 20 to 40 percent on every family
at once, it compounds with both other pillars, and its first experiment
settles it in a week. The depth schedule is second, since without trained
heads its gain is the 15 to 30 percent the cost table and the confidence
stop can find in the shipped drafters. Prefill is third by the golden
metric and first by what the coding agents feel, and its largest lever is
the one with an external measurement behind it.
