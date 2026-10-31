# MOONSHOT-1: the persistent step

Written 2026-10-30 against revision 78038ed4, revised the same day against
the review in MOONSHOT-1-review.md. This is pillar 1 of
[the moonshot](MOONSHOT.md) worked out to a buildable plan: the design, the
caveats the engine's own record forces on it, the contract every model
family must meet for it to generalize, and the delivery checklist with its
exit gates and redesign triggers. The platform numbers below were read from
the device on the day of writing and from the measurements in
[DESIGN.md](DESIGN.md) and the model cards.

| GB10 fact | Value |
|---|---|
| SMs | 48 |
| Shared memory per SM | 100 KB, 99 KB per block opt-in |
| Registers per SM | 65,536 x 32 bit, 256 KB |
| L2 | 24 MB, persisting set-aside at most 18 MB |
| Threads per SM | 1,536, at most 24 resident blocks |
| Cooperative launch | supported, also inside a green context |
| Green contexts | CUDA 13.0, driver 580: 48 SMs split 24 + 24, disjoint and concurrent, measured 2026-10-30 |
| Weight stream ceiling | about 240 GB/s, 258 in the best in-server GEMV |
| Fold path in situ, each way | 13 to 16 µs including the staging copy and the polled reads |
| Fold arrival skew in situ | 11 to 15 µs median, p90 38 µs on attention folds and 106 on MoE folds |
| Host flag round trip | 1 µs |

## 1. The goal and the floor

A decode step's floor is the bytes it must read divided by what one rank's
memory delivers. The bytes are the weights the step touches, the routed
experts its rows select, the KV or state its attention reads, and the head.
Everything else in a pass is the engine standing between the memory and the
tensor cores, and the goal is to remove it.

Three quantities are kept apart, because a slower implementation that reads
more bytes can flatter a single ratio:

- **Logical bytes**: what the step must read, computed at run time from the
  stream plan's ranges and the attention consumer's page count. This is the
  floor's numerator.
- **Measured traffic**: what the memory actually moved, read from the
  device's DRAM counters under `ncu` on an idle fabric. On this SoC the
  counters include the CPU's and the NIC's traffic, so the measurement is
  taken with nothing else running. Rereads, write-backs and staging show up
  here and nowhere else.
- **Bandwidth efficiency**: logical bytes divided by pass time and the
  ceiling. This is percent-of-floor, published per step in the metrics
  beside the per-collective cost.

None of the three promotes a change. The promotion gate is the method of
record's own figure: decode tokens per second by class and by concurrency,
the engine's decode steps only, three repeats with the spread, and the rows
accepted per pass reported beside it. The record holds the reason: a
split-K change on 2026-10-19 took the C1 code pass from 47.18 to 45.97 ms
and decode throughput from 76.1 to 75.0 tokens a second, because the
rounding-level change moved greedy near ties and the fixed prompt lost
accepted rows. A cheaper pass is not a faster engine until the tokens say
so.

| Family at C1 | Pass today | Logical bytes | Floor at 240 GB/s | Target pass | Gain |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash-1017 MXFP4, four Sparks, five rows, code | 47.1 ms | 7.0 GB | 29.2 ms | 31.3 ms | 1.50x |
| DeepSeek-V4.1-Flash-1017 MXFP4, four Sparks, two rows, prose | 32.3 ms | 5.2 GB | 21.5 ms | 23.0 ms | 1.40x |
| Qwen3.8-27B NVFP4 with DFlash2, one Spark, eight rows | 108 ms | 20.1 GB | 83.8 ms | 86 ms | 1.25x |
| GLM-5.3-Flash NVFP4/FP8, four Sparks, two rows | 30.6 ms | 5.9 GB | 24.5 ms | 27.5 ms | 1.11x |
| GLM-5.3 Mixed346, four Sparks, two rows | 62.7 ms | 12.8 GB | 53.3 ms | 57.5 ms | 1.09x |
| Qwen3.8-Flash-Next NVFP4, one Spark, two rows, not yet traced | 37 ms | about 5.0 GB | about 21 ms | 23 to 24 ms | 1.55x, an estimate |

The rows carry the row count of the pass they describe, because a floor
without its row count is not a floor. The bytes are the distinct experts
at the uniform expectation of the picks, the fixed weights, the state and
the head, from the checkpoint's own headers. The DeepSeek rows carry the
phase-2 plan generator's count of 2026-10-30 (section 1.2), which reads
the head twice as the bf12 companion the production template serves and
corrected six omissions of the phase-0 count. The targets sit 3 to 13
percent above their floors, not a uniform few percent. The one-Spark
residue is the draft chain and the step boundary. The four-Spark residue
adds two folds a layer whose critical path is 11 to 14 µs before skew, 80
to 160 of them a pass, and whatever skew the lead does not hide. The
phase-1 experiment of section 1.3 reaches 98 to 102 percent of the floor
at the layer level on every family's shape at world 1 with ceiling-rate
consumers, and 104 to 109 on DeepSeek-V4.1's three shapes with the
consumers built that day, so the margin between those layers and these
targets is the four-node fold's skew and the pass boundary, which phase 4
measures, plus the work distribution of phase 2.

| Family at C8 | Today, decode tok/s | Target | Gain | Mechanism |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash-1017 EXL3 3.5 bpw, four Sparks | 165 to 248 | 230 to 340 | 1.4x | per-row latency work hidden, projections read once for all rows |
| DeepSeek-V4.1-Flash-1017 MXFP4, four Sparks | 84 to 139 | 105 to 175 | 1.25x | the same, measured at the layer level in section 1.3 |
| GLM-5.3-Flash NVFP4/FP8, four Sparks | 104 to 113 | 145 to 160 | 1.4x | the same, fewer latency phases per row |
| GLM-5.3 NF4I8, four Sparks | 53 to 60 | 68 to 78 | 1.3x | sixteen-row folds and the per-row tax |
| Qwen3.8-27B NVFP4, one Spark | 119 to 244 | 165 to 330 | 1.35x | dense verify rows free to the compute bound |

The C8 targets are estimates with a wider band than the C1 targets. At C8
the expert fan-out is near saturation so the stream is already most of the
step, and the gain is the per-row latency work that today scales at about
1.2 ms a row on DeepSeek-V4.1 and the projections that today reread their
weights every four rows. The C8 experiment of section 1.3 fixes the split:
the lead can only prefetch the fixed ranges, 55 MB of a 731 MB DeepSeek
layer, so it hides at most 80 µs of chain per layer there, and the rest of
the C8 gain is the fused chain, which is consumer and kernel-form work.

### 1.2 Phase 0 findings, 2026-10-30

Phase 0 ran on the release binary production serves, 0.1.0+g11fe1807d63d,
with one fabric window: nsys node traces of DeepSeek-V4.1-Flash-1017 MXFP4
on four Sparks at C1 and C8 and a C1-only run with the bus timeline,
Qwen3.8-27B NVFP4 with DFlash2 on one Spark, GLM-5.3 Mixed346 on four, and
GLM-5.3-Flash NVFP4/FP8 on four. The hand count of DeepSeek-V4.1's bytes
from the checkpoint's own headers, the family-generic two-term instrument
over the traces, and the first stream replay model were built on them. The
traces and tools live in the scratch area, untracked. Four findings change
the plan.

**The floors were understated, then over-counted, and are now from the
kernels' own accounting.** The plan's DeepSeek-V4.1 floor of 28.8 ms was a
three-row figure, and the C1 code pass verifies five rows when the DSpark
schedule allows. The expert slot kernels launch one slot per row and pick
plus a shared slot per row, duplicates across rows are L2 hits in expert
order, so the DRAM bytes are the distinct experts, the uniform expectation
of the picks until a device counter reads the run list. At five rows that
is 28.9 of 384 experts, 191 MB a layer, 8.4 GB a pass, 35.1 ms at
240 GB/s. Full GLM's floor at its two-row shape is 53.3 ms on the Mixed346
checkpoint, not 40. A floor without its row count is not a floor.

**Corrected 2026-10-30 by the phase-2 plan generator.** The stream plan of
section 2.4 is built (`src/step`, `step_plan_test`) and counts every byte a
rank's step reads from the loader's own layout, so the DeepSeek-V4.1 floor
is now the plan's figure. The hand count above had six omissions, each
confirmed against the loader: the head's two reads a pass were missing
(662 MB as bf16, 497 MB as the bf12 companion the production template
serves), the draft stages were given the backbone's top-6 rather than
their own top-3 (14.2 of 128 experts a stage, not 26.8), the mHC
coefficient matrices were counted as the F32 of the file rather than the
bf16 the loader rounds them to, the fp8 scale grids as the e8m0 of the
file rather than the F32 grid the fp8 core reads, `main_proj` as a TP
shard rather than the whole matrix every rank holds, and the KV rows at
512 bytes rather than the pool's 528-byte window rows and 288-byte
entries. The corrected count per rank at world 4, with the hand count
rerun under the same corrections agreeing to 0.1 percent:

| Shape | Fixed | Routed experts | KV | Step bytes | Floor at 240 GB/s |
|---|---|---|---|---|---|
| five verify rows, a five-row draft block, the bf16 head twice | 3.211 GB | 5.633 GB | 43 MB | 8.886 GB | 37.0 ms |
| the same with the bf12 head | 3.046 GB | 5.633 GB | 43 MB | 8.721 GB | 36.3 ms |
| two verify rows, the same draft block, the bf16 head twice | 3.211 GB | 2.424 GB | 18 MB | 5.653 GB | 23.6 ms |
| the same with the bf12 head | 3.046 GB | 2.424 GB | 18 MB | 5.488 GB | 22.9 ms |

The as-served rows, with the bf12 head, were the floors this document
used for DeepSeek-V4.1 until the night of 2026-10-30: 36.3 ms at five rows
and 22.9 at two, where the phase-0 figures were 35.1 and 20.7. The other
families' floors are re-counted the same way when their generators land in
phase 6.

**Corrected again 2026-10-30 night by the measured routing.** The routed
bytes above take the uniform expectation of the picks, rows choosing
independently: 28.9 distinct experts a layer at five rows, 11.8 at two.
Four texts through the real checkpoint (section 1.4, H1) show consecutive
rows sharing experts far more than that: 6.0, 10.1, 13.7, 16.9, 19.8, 22.6,
27.7, 36.4 and 43.8 distinct experts a layer for 1, 2, 3, 4, 5, 6, 8, 12
and 16 consecutive rows, about 3.3 more a row past the first, against the
uniform model's 11.8 at two and 28.9 at five. The verify rows of a C1 pass
are consecutive positions, so the measured figures are the pass's. The
plan's shape takes the measured count (`distinct_main`), and the floors it
yields with the bf12 head are:

| Shape at the measured overlap | Step bytes | Floor at 240 GB/s | Pass today | Today at |
|---|---|---|---|---|
| five verify rows, 19.8 experts a layer | 7.011 GB | 29.2 ms | 47.1 ms | 62 percent |
| two verify rows, 10.1 experts a layer | 5.162 GB | 21.5 ms | 32.3 ms | 67 percent |

These are the floors the rest of this document uses for DeepSeek-V4.1,
and the targets of section 1 are re-derived from them at the same 107
percent of the floor: 31.3 ms at five rows and 23.0 at two, 1.50x and
1.40x over today. Two consequences follow. Today's engine is further from
its floor than the 77 and 71 percent below say, 62 and 67, since it too
reads only the distinct experts, so the pillar's cap at a perfect pass is
1.61x on C1 code rather than 1.30x. And the routed share of a layer falls
from 69 to 59 percent at five rows, so the fixed phases, the latency and
the tails weigh more in the pass: the eight-layer gate of section 1.3 now
routes through a window of 32 experts a layer, which reproduces the
measured overlap (19.7 distinct at five rows, 10.1 at two), and with every
weight computed the median layer stands at 83 percent of its floor there
(production down, clocks locked, a warm-up run), against 89 at the uniform
count measured beside production. The draft stages keep the uniform
expectation (14.2 of 128 at top-3) until their routing is measured.

**Today's engine is far closer to its floor than the plan assumed.** The
two-term instrument at the real shapes, per rank, with the existing
prefetch windows streaming 35 to 50 MB a layer under the latency phases.
The instrument charges the DRAM-class kernels at the ceiling; the phase-1
real-consumer measurement of section 1.3 later put the expert kernels at
about 80 percent of it at C1 and 60 to 75 at C8, so the "pass at" column
is the latency share only and the consumers hold a further 10 to 20
points of each pass:

| Family, C1 shape | Pass | Byte floor at 240 GB/s | Pass at | Layer latency, folds, gaps |
|---|---|---|---|---|
| DeepSeek-V4.1 MXFP4 w4, five rows, code | 47.1 ms | 36.3 ms (29.2 at the measured overlap) | 77 percent (62) | 122 + 81 + 13 µs |
| DeepSeek-V4.1 MXFP4 w4, two rows, prose | 32.3 ms | 22.9 ms (21.5 at the measured overlap) | 71 percent (67) | 89 + 49 + 14 µs |
| Qwen3.8-27B NVFP4 w1, eight rows | 108.0 ms | 83.8 ms | 78 percent | 77 + 0 + 0 µs, GEMVs at 205 GB/s |
| GLM-5.3-Flash NVFP4/FP8 w4, two rows | 30.6 ms | 24.5 ms | 80 percent | 50 + 47 + 9 µs |
| GLM-5.3 Mixed346 w4, two rows | 62.7 ms | 53.3 ms | 85 percent | 100 + 53 + 11 µs |

The plan's 60 percent figure compared a five-row pass against a three-row
floor and landed near the truth by accident. The C1-only timeline puts a
collective at 24 to 35 µs on this binary, copy 3.3 plus handshake 11 to 15
plus skew 8 to 12.

**The replay model's design side gives 1.09x to 1.27x at C1, not 1.5x.**
The model is calibrated to within 2 to 12 percent of the measured layers
and passes on all five cases without fitting, then run with the design's
parameters: a 23 MB lead, the stream servicing released ranges out of order
across layers, a governed rate of 150 GB/s during latency segments,
consumers at the ceiling, the fused chain, four hand-offs at 2 µs, a 12 µs
fold path plus the measured skew distribution.

| Family, C1 shape | Pass today | Model of today | Design | Design at | Gain |
|---|---|---|---|---|---|
| DeepSeek-V4.1 MXFP4 w4, five rows | 47.1 ms | 44.8 | 39.0 ms | 107 percent of floor | 1.21x |
| DeepSeek-V4.1 MXFP4 w4, two rows | 32.3 ms | 28.3 | 25.5 ms | 111 percent | 1.27x |
| Qwen3.8-27B NVFP4 w1, eight rows | 108.0 ms | 105.6 | 86.5 ms | 103 percent | 1.25x |
| GLM-5.3-Flash NVFP4/FP8 w4, two rows | 30.6 ms | 31.6 | 27.5 ms | 112 percent | 1.11x |
| GLM-5.3 Mixed346 w4, two rows | 62.7 ms | 67.8 | 57.5 ms | 108 percent | 1.09x |

The DeepSeek rows carry the 2026-10-30 correction: the model's pass held
8.43 and 4.97 GB, the plan generator counts 8.72 and 5.49 as served, and
the design streams the difference at the ceiling, 1.2 and 2.1 ms, since
its consumers are at the ceiling by assumption; the model's own runs were
37.8 and 23.4 ms. Sensitivities on the DeepSeek five-row layer: the lead's capacity from 12
to 23 MB changes nothing, the governed rate from 100 to 240 GB/s moves the
layer 2 percent, the fold path from 8 to 30 µs moves it 4 percent, and
hand-offs from 1 to 8 µs move it 6 percent. The 27B gain is mostly its
GEMVs going from 205 GB/s to the ceiling, which is kernel work the graph
engine could also take. The phase-1 trigger that re-derives every target
when the gain falls under 25 percent fires at phase 0 for the MoE families
at C1.

**The headroom grows with concurrency.** At C8 the 27B pass sits at 50
percent of its floor: 31.5 ms of latency a pass, 16 ms of it the GDN
recurrence at 64 rows, and its GEMVs at 171 GB/s, compute-bound in the fp4
decode at 64 rows. DeepSeek-V4.1 at 24 to 30 rows carries 400 to 590 µs of
latency and 185 to 434 µs of folds per layer against 2.6 to 3.1 ms of
bytes, 19 to 30 percent idle per layer, so the structure alone is worth 1.2
to 1.4x there before any kernel form changes, and the many-row kernel forms
are worth as much again on the dense family. A device counter of the slot
order kernel's run list would replace the uniform expectation of distinct
experts with the measured count. It is the first additive instrument of
the next window.

**Revised C1 targets.** Section 1's table carries these, for the step
engine's structural gain alone. Kernel-form gains at many rows are a
separate line the C8 instrument will size.

| Family at C1 | Pass today | Target pass | Decode tok/s today | Target |
|---|---|---|---|---|
| DeepSeek-V4.1 MXFP4, four Sparks, code | 47.1 ms | 39 ms | 80 | 97 |
| DeepSeek-V4.1 MXFP4, four Sparks, prose | 32.3 ms | 25.5 ms | 55 | 70 |
| Qwen3.8-27B NVFP4, one Spark, code | 108 ms | 86 ms | 43.5 | 54 |
| GLM-5.3-Flash NVFP4/FP8, four Sparks | 30.6 ms | 27.5 ms | 58 | 65 |
| GLM-5.3 Mixed346, four Sparks | 62.7 ms | 57.5 ms | 31 | 34 |

What this means for the program: the persistent step is a 10 to 40 percent
lever at C1, largest where the chain's latency is largest relative to its
bytes, and a 20 to 40 percent lever at C4 to C8 on this evidence, and the
kernel forms at many rows are the other half of the C8 gain. Phase 1's
architecture experiment stays as the decider, with its gate reset to this
table, and the first quarter adds the C8 instrument with real expert
counts before any consumer is built.

### 1.3 Phase 1 architecture experiment, 2026-10-30

One persistent kernel ran eight layers of DeepSeek-V4.1's five-row window
layer as bytes: 19.9 MB of attention projections, 27.3 MB of output and
router weights, 8.9 MB of shared expert and 29 distinct experts of 4.72 MB
chosen by a late-binding router, 193 MB a layer and a floor of 804 µs at
240 GB/s, with real memory traffic, synthetic math and the chain's latency
segments as timed spins at world 1. Block 0 is the chain, forty blocks are
workers that consume released ranges with a warp-per-row GEMV and run the
lead when idle, and every hand-off is a device flag.

| Configuration | Layer | Of the floor | What it isolates |
|---|---|---|---|
| zero latency, no lead | 807 µs | 100 percent | the kernel's consumers stream the bytes at the ceiling, hand-offs cost about 7 µs a layer |
| design chain, 66 µs a layer, no lead | 868 µs | 108 percent | the latency is paid in full |
| design chain, lead 20 to 23 MB | 803 to 815 µs | 100 to 101 percent | the lead hides about 85 percent of the chain |
| today's chain, 173 µs a layer, no lead | 970 µs | 121 percent | the unfused chain paid in full |
| today's chain, lead 20 MB | 887 µs | 110 percent | the lead hides about half of a long chain |

The graph engine's real layer at the same bytes measures 1002 µs, so the
structure is worth about 1.2x at this shape once the real consumers reach
the ceiling and the chain fuses to the design's budget, and the two levers
are complementary: fusing shrinks what must be hidden and the lead hides
the rest. The first gate of phase 1 held at world 1, and the experiment
beat the replay model's design residue because the workers prefetch in
every idle moment rather than only during latency segments.

Four things the experiment taught that the design now states:

- **The prefetch must be a full-width load whose value is used.** A touch
  loop that loaded 16-byte words and folded two of their four lanes into a
  dead sink left nothing in L2: a fully prefetched range still consumed at
  the cold rate. The same range prefetched by the GEMV consumer's own loads
  consumed at the L2-hot rate, 932 GB/s. Every lead prefetch is verified by
  a hit-rate check before it is trusted.
- **The lead's cap is 16 to 20 MB, not 23.** At 23 MB the expert stream has
  nowhere to allocate and the lead's own lines go, and under a long chain
  the layer got slower than with no lead at all. The persisting-L2 probe
  put the same edge at 16 to 18 MB. With the tensor-core consumers in the
  kernel the cap fell again, to 8 MB at C1 code, because those consumers
  hold activation windows and scales in L2; the cap is tuned per family
  against its consumers (the integrated run below).
- **Forty streaming CTAs reach the ceiling in the GEMV form**, 32 reach 230
  GB/s and 24 reach 213. The streamer count follows the consumer form, not
  the plain-load probe's 16 to 24.
- **The lead cursor must run ahead of consumption and the accounting must
  be exact.** A cursor that fetched ranges the chain had just consumed, and
  a cap that subtracted whole ranges rather than what was prefetched, each
  cost the entire benefit while looking like a working lead in the counters.

The same kernel then ran every family's C1 layer shape, with each
family's measured chain latency from the two-term instrument as today's
chain and the fused budget as the design's. Per layer, as a percentage of
each shape's floor at 240 GB/s:

| Layer shape | Zero latency | Today's chain, no lead | Today's chain, lead 20 MB | Design chain, no lead | Design chain, lead 20 MB | Graph engine measured |
|---|---|---|---|---|---|---|
| DeepSeek-V4.1 window, five rows, 193 MB | 100 | 116 | 107 | 108 | 101 | 126 |
| DeepSeek-V4.1 window, two rows, 113 MB | 99 | 129 | 112 | 114 | 101 | 141 |
| GLM-5.3-Flash KDA, two rows, 120 MB | 99 | 121 | 109 | 112 | 101 | 125 |
| GLM-5.3 reuse, two rows, 151 MB | 100 | 125 | 110 | 110 | 102 | 114 |
| Qwen3.8-27B GDN, eight rows, 267 MB | 98 | 107 | 99 | 104 | 98 | 122 |

The lead alone recovers half to two thirds of an unfused chain on every
shape, the fused chain alone leaves 4 to 14 percent, and the two together
reach 98 to 102 percent of the floor everywhere. The structural headroom
against the graph engine's measured layers is therefore 1.12x to 1.39x at
the layer level, which is what the replay model predicted at the pass level
once the distinct-expert attribution was corrected.

With the fold's notice-and-post hop made real inside the kernel, a pinned
word the chain writes and a host thread answers while the workers stream
at 239 GB/s, the hop costs 5.8 µs at p50 and 7.3 at p90, inside the design's
12 µs fold budget, and the layer stays at 100 percent of the floor.

The same kernel at DeepSeek-V4.1's C8 shape, thirty rows and 143 distinct
experts of 384, 731 MB a layer and a floor of 3.05 ms, with the two-term
instrument's C8 latency and folds as today's chain, 854 µs a layer, and
200 µs as the fused chain's budget at thirty rows:

| Configuration at C8 | Layer | Of the floor |
|---|---|---|
| zero latency, no lead | 3140 to 3198 µs | 103 to 105 percent |
| today's chain, no lead | 3999 µs | 131 percent |
| today's chain, lead 20 MB | 3908 µs | 128 percent |
| design chain, no lead | 3401 µs | 112 percent |
| design chain, lead 20 MB | 3243 µs | 106 percent |

Two things differ from C1. The lead buys 3 to 6 points instead of 10 to 17,
because the only ranges released early enough to prefetch are the fixed
ones, 55 MB of the layer, and a 20 MB lead on them hides at most 80 µs of
an 854 µs chain. At C8 the fused chain is the lever and the lead is its
helper, the reverse of C1. And the zero-latency residue rises from 1 to
3 to 5 percent as the distinct-expert count rises, 101 percent at 29
experts, 104 at 60 to 100 and 103 to 105 at 143, the same at the C1 and C8
arena sizes, so it is the per-range cost of the probe's slot form, one
hand-off and one tail per expert, and not DRAM reach. The stream engine's
expert ranges at many rows are therefore coalesced, a run of several
experts' tiles released as one range, which section 2.4 now states. The
structural gain at C8 is 1.2 to 1.25x at the layer level on this shape,
and the many-row kernel forms are the other half of section 1's C8 table,
as phase 0 said.

Worker count at C8: 40 CTAs reach 103 percent of the floor, 32 reach 108,
and 48 reach 116, since a worker on every SM leaves the chain block no SM
of its own and the chain's phases wait behind a worker's tile. The step
engine keeps `engine.step_sms` under the SM count by at least one.

**The real consumer inside the kernel, 2026-10-30.** The owed item ran:
the MoE phase of the same persistent kernel executed by the slot kernels'
own device code from `src/kernels/fp4_gemv.cuh` and `fp8_gemv.cuh`, the
exact per-row chain and epilogue of `moe_slot_gate_up_swiglu_fp4_kernel`
and `moe_slot_down_fp4_kernel`, over a real MXFP4 expert pool at
DeepSeek-V4.1's world-4 geometry (gate and up 576 by 5120, down 5120 by
576, 4.70 MB an expert) with the fp8 shared expert, the router's sorted
slot order and the ordered accumulation. A reference path launches the
same device functions in the slot kernels' own grids at the real kernels'
register count and occupancy, two CTAs an SM, and the two agree bitwise on
every activation, partial and output at 1, 5 and 30 rows. The probe's
expert pool is 64 experts at C1 and 160 at C8, so its slots repeat an
expert more often than production's 384 do, 1.46 and 1.88 slots a distinct
expert against production's 1.21 and 1.47.

The rate of the bytes the units actually read, duplicates included, at
zero chain latency, in GB/s:

| MXFP4 expert consumer | gate and up, 5 rows | 30 rows | down, 5 rows | 30 rows |
|---|---|---|---|---|
| today's slot kernels as plain grids, cold | 242 | 265 | 207 | 283 |
| the same, L2-hot | 240 | 266 | 203 | 285 |
| the same units inside the persistent kernel, one CTA an SM | 186 | 206 | 181 | 202 |
| the same, two CTAs an SM | 257 | 336 | 173 | 264 |
| run-list units of up to two slots, two CTAs an SM | 210 | 257 | 151 | 185 |
| the synthetic bf16 streamer of the tables above, one CTA an SM | 240 | 240 | 240 | 240 |

Divided by the distinct bytes, which is what the floor counts, today's
kernels stream the expert phase at about 80 percent of the ceiling at five
rows and 60 to 75 percent at thirty, the lower figure at this probe's
duplicate rate and the higher at production's. The ceiling they do reach
is their own: hot equals cold, so an L2 hit is not faster than a DRAM
read for them, and `ncu` on the plain grids reports 0.4 instructions
issued a cycle a scheduler with 60 percent of cycles without an eligible
warp, two to three warps resident a scheduler, and 5 to 9 cycles between
a warp's instructions. The GEMV form is issue-bound and latency-bound at
once: a lane runs a dependent chain of two FMAs per weight byte, so one
warp issues an instruction every few cycles, and the register count caps
the residency at two CTAs an SM. The down projection's lane owns nine
chunks at 576 wide and is the slower of the two in every form.

Four things follow, and they change the plan's weighting more than its
shape.

- **Today's expert phase is not at the ceiling, and the two-term
  instrument could not see it.** The instrument charged every kernel of
  the DRAM class at the ceiling and attributed the rest of the layer to
  latency and folds. The direct measurement puts the expert kernels at
  about 80 percent of the distinct-byte floor at C1 and 60 to 75 at C8,
  so of the gap between today's pass and its floor, the consumers hold a
  third at C1 and half at C8, and the chain and folds hold the rest. The
  structural levers of this experiment, the lead and the fused chain,
  and the consumer forms of phase 3 are each worth about half of the
  gain, and the targets of section 1 need both.
- **Lifting the existing units into the persistent kernel loses.** At one
  CTA an SM the units run at 60 to 80 percent of their own grid's rate,
  because a unit's wait for its loads and its FMA chain have nothing
  else on the SM to overlap with; at two CTAs an SM they match or beat the
  grids on lines but still not the ceiling on distinct bytes. The 92
  percent gate of this experiment is missed by this form at every row
  count, and the trigger fires on the consumer form, not the staging
  scheme: phase 3 builds the split of sections 2.3 and 2.7, streamer
  warps that keep the memory pipe full across unit boundaries and
  consumer warps that compute from staged tiles, and does not port the
  kernels' units.
- **The run list is exact and removes the duplicate reads, and the GEMV
  form cannot cash it.** Grouping an expert's slots into one unit of up to
  four rows reads the expert once, 367 MB where the slot form reads 658 at
  thirty rows, and the results are bitwise the slot form's, because the
  fp4 core's per-row chain is the same whatever the row count. But the
  group's rows multiply the instruction count and the shared-memory
  activation reads per weight byte, so the grouped units run slower per
  line and end at the same 80 percent of the distinct floor. The consumer
  that cashes the run list decodes a weight chunk once for every row of
  the group, which is the tensor-core form: weights as the A fragments,
  the group's activations as B, one decode per chunk whatever the row
  count, independent accumulators instead of a chain. That is phase 3's
  streaming MMA form, and the first thing it is measured on is this
  table's shape at 1, 8, 16 and 32 rows.
- **The activation staging is a cost of its own.** Each slot unit stages
  its token's 10 KB row from L2 before it can compute, 23 percent of its
  weight bytes, and a four-row group stages 40 KB against 43.5 KB of
  weights. Consecutive units of one group now reuse the staged rows, and
  the stream engine's consumers keep a group's rows resident across every
  unit of the group.

The consumer's budget, stated once: at 240 GB/s over 48 SMs each SM
consumes 5 GB/s, ten million 512-byte warp-chunks a second. The GEMV form
spends about 130 warp instructions a chunk and achieves an issue rate near
1.6 a cycle an SM, which is the 80 percent above. A consumer at the
ceiling with margin spends under 40 warp instructions a chunk or issues
near 4 a cycle, and only independent accumulators and tensor-core
fragments give either.

**Phase 3 brought forward: the tensor-core consumer, 2026-10-30.** The
consumer's budget above says what the fix is, so its first candidate was
built and measured the same day: a persistent MXFP4 streaming consumer
for the gate and up pair at DeepSeek-V4.1's world-4 geometry in which a
warp owns 16 weight rows and every token of the group. The weights are
the A operand of `mma.sync` m16n8k16 in bf16, decoded once per 32-code
chunk by byte-permute lookups (e2m1 magnitude to bf16 high and low bytes,
the sign replicated into bit 7, the chunk's e8m0 power of two applied by
one exact bf16 multiply) and distributed into fragments by `ldmatrix` from
a shared-memory stage that `cp.async` fills in 64-byte row segments three
ranges deep; the activations are the B operand, streamed through a
shared-memory window ring, read as one 16-byte word per lane because the
k order inside a chunk is permuted the same way on both operands. Gate
and up share the B fragments; the products accumulate in fp32 in
independent fragments per matrix and token group; the epilogue is the slot
kernels' swiglu. Against an fp32 sequential reference the outputs agree
exactly on 99.9 percent of values and within two bf16 ulps on the rest,
the residue being accumulation order at the bf16 rounding before swiglu.

| Streaming consumer, 128-row units of 64 experts, cold | 8 rows | 16 | 24 | 32 |
|---|---|---|---|---|
| plain grid, 16-byte loads straight into fragments | 129 GB/s | 119 | 128 | 128 |
| plain grid, coalesced `cp.async` stage and `ldmatrix` | 151 | 152 | 152 | 149 |
| the same with a 512-byte bulk L2 prefetch a row | 207 | 206 | 199 | 203 |
| persistent, 48 CTAs, prefetch spread over the ranges | 214 | 214 | 202 | 200 |
| persistent, 40 CTAs | 215 | 214 | 211 | 208 |
| the same with decode and MMA removed, loads only | 217 | | | 203 |
| today's GEMV slot kernels, distinct bytes, for reference | about 190 | | | about 140 |

Three findings, each a design fact for phase 3.

- **The per-row cost is gone.** From 8 to 32 rows the consumer's rate moves
  by 3 percent, where the GEMV form's distinct-byte rate falls by a third,
  so the C8 half of section 1's gain is real and the budget holds with
  margin: with decode and MMA removed the kernel runs at the same rate as
  with them, so compute is fully hidden behind the stream at one CTA an
  SM and about 150 registers.
- **The DRAM wants 512-byte row segments and the fragments want 16-byte
  ones.** A fragment load touches eight rows at 16 bytes each, and the
  memory system delivered 125 GB/s for it; the same kernel reading the
  same bytes as 512-byte runs of one row reached the ceiling. The
  consumer therefore separates the two: a bulk L2 prefetch
  (`cp.async.bulk.prefetch.L2`) brings each row's next 512 bytes in one
  request, and the fine 64-byte stage fills hit L2. That recovered the
  rate from 151 to 215 GB/s, 90 percent of the ceiling, with the loads
  alone at 217. The last ten points are in the prefetch's L2 behaviour:
  granules of 1 and 2 KB, prefetching two or more granules ahead, and
  an evict-last policy on the prefetch all made it slower, by up to half,
  and an evict-first policy on the fine loads changed nothing. The L2
  behaves as a small effective window for normal-priority streaming, as
  the persisting-L2 probe found in the other direction, and about a third
  of the fine loads still miss it.
- **The layout is the remaining lever.** The 16-row fragment pattern is
  inherent to the MMA; the 512-byte row segments are inherent to the DRAM;
  the stage and the prefetch are the bridge between them and cost the ten
  points. A tile-major expert layout in the step engine's arena, each
  16-row tile's chunks contiguous so that one lane-contiguous 512-byte
  load is sixteen rows of two chunks and a register shuffle replaces the
  stage, would feed the fragments with the GEMV pattern's requests and no
  shared-memory stage at all. Section 2.4's rule that every tensor keeps
  its checkpoint layout would then hold for everything but the routed
  experts, whose prefill consumers would read the tile-major form through
  an address map. The measurement decided it the same afternoon.

**The tile-major form reaches the ceiling.** With the expert matrices laid
out in 16-row tiles whose 32-code chunks are contiguous, a warp's tile is
one 40 KB run, each lane-contiguous 512-byte load is two chunks of sixteen
rows, four such loads a matrix stay in flight per lane, and eight shuffles
a chunk put each lane's eight codes in its fragment. No shared-memory
stage, no L2 prefetch, 12 to 49 KB of shared memory for the activation
windows only, 151 registers at 8 rows and 243 at 32.

| Tile-major tensor-core consumer, cold, 64 experts | 8 rows | 16 | 24 | 32 |
|---|---|---|---|---|
| gate and up pair, 48 persistent CTAs | 227 GB/s | 227 | 226 | 212 |
| gate and up pair, 40 CTAs | 227 | 230 | 228 | 221 |
| down (5120 by 576), 48 CTAs | 221 | 193 | 175 | 155 |
| down, 40 CTAs | 220 | 195 | 180 | 157 |
| NVFP4 gate and up pair, e4m3 scales per 16 codes, 48 CTAs | 216 | 218 | 218 | 215 |
| fp8 e4m3 pair with F32 block scales 128 by 128, the shared expert's and the dense projections' format, 48 CTAs | 227 | 225 | 221 | 226 |
| the same, 40 CTAs | 229 | 228 | 222 | 227 |

The pair streams at 94 to 96 percent of the ceiling at 8 to 24 rows and
88 to 92 at 32, flat across rows, exact to two bf16 ulps against the fp32
reference; the format bench's gate holds for it at 8 to 24 rows and sits
at the edge at 32, where the kernel is at 243 registers. The same kernel
with NVFP4's two e4m3 scales a chunk, the format of Qwen3.8, GLM-5.3-Flash,
GLM-4.7 and the 27B's experts, streams at 90 to 91 percent at every row
count, exact to two ulps, two points under the gate with the extra scale
handling at 212 registers. The fp8 form, e4m3 weights with one fp32 scale
per 128 rows and 128 codes, the shared expert's format and the attention
and dense projections' on every family that ships fp8, streams at 92 to 96
percent at every row count: 32-byte row chunks make one 512-byte run
sixteen rows of one chunk, four shuffles a chunk, the e4m3 to bf16
conversion a shift and an add on the biased exponent, and the block scale
applied once per four chunks to a partial accumulator. Against the fp32
block-scaled reference it is exact at 8 rows and within two ulps above,
and at 162 to 199 registers it is the lightest of the three. The down
projection holds the gate at 8 rows and falls with rows, 11 SM cycles a
chunk a warp for every token group of eight: a single matrix amortizes
nothing, so per weight byte it spends twice the pair's B-fragment loads
and MMAs, and the sum of its pipes, not any one of them, is the bound.
Five changes were tried and none moved it outside noise: the hardware
e2m1-to-f16 conversion in place of the byte-permute decode with the f16
MMA, two tiles a warp sharing the B fragments, two CTAs an SM with the
registers capped at 128, the scale bytes loaded a tile ahead, and a
conflict-free activation stride. What remains for it is the per-group
cost itself, which at C8's thirty rows puts the down third of the expert
bytes at 65 percent and the expert phase as a whole at about 82, against
today's 60 to 75; at C1's one token group both projections stream at 90
to 96. The decision for phase 2: the step engine's arena lays the routed
experts out tile-major, the only tensors whose internal layout changes,
and the prefill path's expert consumers read that layout through an
address map; everything else keeps its checkpoint layout as section 2.4
says.

**The persistent kernel with the measured consumers, 2026-10-30.** The
architecture probe then ran DeepSeek-V4.1's layer with the day's two
tensor-core forms as its MoE units: the tile-major MXFP4 pair and down
consumers for the routed experts, a group being every slot of one expert,
and the fp8 block-scaled pair and down consumers for the shared expert, one
group of every token, both in 128-row units; the ordered accumulation as a
worker phase of 1024-output units; the fixed ranges as before. Outputs
agree with the fp32 reference to two bf16 ulps except where the swiglu
tail amplifies a one-ulp difference in its input, a few values in a
hundred thousand. Forty workers, three passes, the median layer as a
percentage of the distinct-byte floor:

| DeepSeek-V4.1 layer | C1 prose, 2 rows | C1 code, 5 rows | C8, 30 rows |
|---|---|---|---|
| zero latency, no lead: the kernel's own floor | 105 | 111 | 106 |
| the design's chain, lead 8 MB | 109 | 111 | 109 |
| the design's chain, lead 8 MB, 47 workers (the engine's shape: one SM to the chain) | 109 | 104 | 108 |
| the design's chain, lead 20 MB | 116 | 120 to 127 | 115 |
| today's chain, lead 8 MB | 131 | 134 | 138 |
| the same kernel with ceiling-rate synthetic consumers (above) | 101 | 101 | 106 |
| the graph engine's layer in the traces | 141 | 126 | 135 to 150 |

The MoE phases stream at 213 to 221 GB/s for gate and up at C1 and 232 to
243 at C8, and at 240 to 245 GB/s for down everywhere; the fixed ranges at
224; the accumulation phase takes 4.5 µs at C1 and 10 at C8. What remains
of the zero-latency residue is the gate and up phase's drain tail, every
worker's idle wait for the last 128-row units, 50 µs at C1 code, 17 at C1
prose and 40 at C8, that is 6, 3 and 1.5 percent, and the router with its
unit map at 4 to 28 µs; both have their fixes on the checklist, finer units
and a parallel unit map, and both belong to phase 2's streamer.

The lead's cap is not free of the consumers. At C1 code a 20 MB lead hides
the design's 66 µs chain and costs the gate and up phase 30 µs, because the
tensor-core units re-read their activation windows and scales from L2 and
the lead's lines take the L2 they used; a cap of 8 MB prefetches 13.6 MB a
layer, hides the whole chain, and leaves the MoE phases at their
zero-latency rate. At C8 the cap does not matter. Section 2.1's 16 to 20 MB
is therefore the ceiling of the mechanism, and the engine's `engine.l2_staging_mib`
is tuned per family against its consumers' own L2 use, 8 MB on this one.

What this prototype says, plainly. With both consumer forms built today,
the accumulation distributed and the lead at 8 MB, the persistent layer
under the fused chain runs at 109 to 111 percent of the floor at every
shape with 40 workers and at 104 to 109 with 47, the engine's own shape of
one SM to the chain and the rest streaming, where the C1 code tail falls
from 50 µs to 20: 1.21x the graph engine's traced layer at C1 code, 1.29x
at C1 prose and 1.25x to 1.39x at C8; under today's unfused chain it is
level with the graph engine, 0.94x, 1.08x and 0.98x to 1.09x, which is the
fused chain's whole worth stated as a measurement. Section 1's targets,
re-derived in phase 2 from the plan generator's count, want 1.21x at C1
code, 1.27x at C1 prose and 1.25x at C8 at the pass level: the prototype's
layers meet all three, with the pass's tail, the head's two reads and the
draft stages at 945 MB, still to be run in the same structure, and the
drain tail of the 128-row units and the unit map as the named phase-2
items. Nothing in the gap is a mechanism that has not been
built; it is the work distribution of phase 2 and the fused chain of
section 2.9, each with its number against it, and the structure measured
at 100 to 106 percent with ceiling-rate consumers remains the bound the
rest converges to.

Owed from this experiment, in the order the plan takes them: the finer
work distribution that removes the gate and up phase's drain tail and the
per-unit pipeline restarts, which is phase 2's streamer; the fused chain
of section 2.9, whose latencies the probe models as spins; the fixed
ranges through their real consumers, for which the fp8 block-scaled form
above is the candidate; the folds as real collectives on four nodes, which
is the port's work; the distinct-expert counter, since the release binary
carries no routing diagnostic and the count needs a development build and
a fabric window; and the MXFP4 down form at four token groups, which this
family never exercises on a routed expert but others may.

#### The streamer and the consumers as repo code, 2026-10-30 evening

Phase 2 and the first of phase 3 are in the tree (`src/step`,
`src/kernels/step`, docs in `docs/step_engine.md`), and their gates ran
beside the serving production world, which held 10 to 15 percent of the
GPU at the time, so the figures below carry its noise.

| Run on the first eight DeepSeek-V4.1 layers, world 4, five rows, 47 workers, 8 MB lead | Median layer, percent of the 240 GB/s floor |
|---|---|
| synthetic consumers, the design chain's latencies interleaved | 99 to 102 |
| the same without the lead | 96 to 100 |
| MXFP4 tensor-core consumers on the routed experts, gate and up then down as two phases | 89 to 94 |
| the same with the expert phases merged, a down unit released by its group's gate and up | 96 to 98 |
| the merged phase with the shared expert computed too, by the row-major fp8 form through the L2 bridge, its units handed out first and its bytes in the lead | 97 to 98 idle, 93 to 95 beside production |
| the same with the routed experts read in the checkpoint's row-major layout through the bridge, no second layout | 92 to 93 |
| every fp8 projection of the layer computed too, by the single-matrix fp8 form, 128-row units over inner-width slabs with a reduce, lead 16 MB | 86 to 91 |

The merged phase is the work distribution the phase-1 prototype owed: the
gate and up phase alone drained 68 µs a layer with 150 units of 110 µs
over 47 workers, and filling that tail with the down units cut it to 22,
the layer going from 93 to 97 percent of its floor. The expert bytes stream
at 250 GB/s over the merged phase. The accumulation as 50 worker units of
512 outputs costs 30 µs a layer against the 8 the design chain modeled,
and is the next item of the chain work. The consumers inside the kernel
are exact: every gate and up output of a layer equals the double-precision
reference or sits inside the envelope its own fp32 accumulation allows,
every down partial sits inside the fp32 bound, and the accumulated rows
are bitwise the ordered sum of the partials, with the shared expert's
partials among them. The protocol around them runs a thousand steps from
the device token feed with the next step's execute seen 1.8 µs after the
verdict at the median.

The shared expert keeps the checkpoint's row-major fp8 layout, as section
2.4 asks of every tensor but the routed experts, and its consumer
(`src/kernels/step/fp8_dense.cuh`) reads it through the bridge the probes
measured: before a warp starts a 512-byte slab of its sixteen rows it
prefetches the next slab of every row into L2 with one bulk prefetch a
row, so the 32-byte fragment loads hit L2. The F32 scale of each 32 by 32
block multiplies the chunk's tensor-core partial, the slot kernels'
structure. As plain grids over 60 cold pairs it streams the pair at 229
GB/s and the down at 216, and inside the merged phase the expert bytes run
at 247 GB/s. Two placement lessons came with it: the shared units are the
longest of the phase and go out first, and the shared expert's bytes stay
in the lead's sequence as lead-only segments although no worker streams
them as bytes.

The fp8 projections followed as consumers of the same form with one matrix
(`fp8_matrix_unit`), every token through each of the layer's fp8
matrices: wq_a, wkv, wq_b, wo_a, wo_b, the indexer's wq_b, the Engram
projection and the draft's main_proj, eight shapes compiled in. All
213,760 outputs of a layer are exact or inside the envelope. Their cost
is the phase structure, not the form: a 128-row unit over a full 5120-wide
row streams at 15 GB/s alone and L2-hot but at the ceiling's share, 5
GB/s, when every SM streams, so a phase of a few matrices is one unit's
time long whatever the worker count, and the fixed phases lost 20 to 50
µs a layer against the byte-streaming form's 64 KB units. Splitting the
inner width into slabs with a deterministic reduce in slab order gave the
tail back on the long matrices; 64-row units cost more than they saved,
each unit's staging and its halved loads in flight outweighing the finer
tail; and the lead's cap, re-swept with every consumer real, moved from 8
MB to 12 to 16, since the projections now read L2-hot what the lead
staged. With the lead at 16 MB the layers run at 86 to 91 percent of
their floor with every weight of the layer computed.

The work-distribution remedies were then tried in turn and measured. The
fixed byte units are handed out last so they fill a phase's tail. A split
form of the projection unit, 32 rows with the eight warps on four parts of
the inner width, the group's activations staged once and reused across a
grab of four units that shrinks to one as the phase drains, and a
deterministic reduce across the parts, is exact and cut the drain to a
third, but its units cost more per byte (the staging of the whole slice,
the cross-warp reduce, four times the prologues) and the phases came out
20 percent longer, so it is kept as an option and the slabbed 128-row form
stays the default. A bridge four slabs ahead instead of two changed
nothing. The plateau is 88 to 90 percent of the floor with every weight
computed, and its mechanism is now clear: during any stretch where fewer
than about forty SMs stream, the memory is not saturated, because one SM's
rate is bound by its bytes in flight against the loaded memory's latency,
and a phase of a few dependent matrices has such stretches at both ends.
The remedy is structural rather than a unit form: no phase of a few
matrices on its own. In the fused chain the fixed matrices stream under the
attention's and the folds' latency with the lead, and the expert phase,
which has no such stretches, already runs at the ceiling. That is where
the gap closes or does not, and it is measured when the chain is real.

The fp8 result raised the question whether the routed experts need the
tile-major layout at all, so the MXFP4 form was built over the row-major
layout with the same bridge (`src/kernels/step/mxfp4_row.cuh`). It is
exact and bitwise like the tile-major form. Under the same conditions, as
plain grids over cold experts, the tile-major pair streams at 233 GB/s and
the row-major pair at 215, the downs at 247 and 223; inside the streamer
the layers run at 95 to 98 percent of their floor with tile-major experts
and 92 to 93 with row-major ones. The eight points on the pair are the
probes' finding of section 1.3 again, so the tile-major layout stays the
arena's second version. What the row-major form buys is independence: the
plan's layout field picks the form at lowering, the streamer serves the
checkpoint's layout today at 92 percent, and the layout version lands when
the loader's transform and the prefill readers' address map are ready
rather than before any family can be ported.

### 1.4 Beyond the pass: the hypotheses of 2026-10-30

Pillar 1 as built makes the memory busy at a fixed byte count. With every
weight computed the layers run at 88 to 90 percent of the byte floor, the
fused chain is owed, and the C1 code target of 39 ms is the pass at 107
percent of the 36.3 ms floor (29.2 at the measured overlap, found later
that night, which makes the cap 1.61x; the figures of this section keep
the floor they were written against). Even a perfect pass caps the pillar
near 1.30x on C1 code (47.1 ms today over 36.3). Everything past that is fewer
bytes a pass or more tokens a pass. The hypotheses below widen the search
in that order, each with the experiment that decides it and the decision
rule, and each has its checklist item in section 5. None touches the graph
engine's paths, and every one keeps the lossless rule: a wrong guess costs
bandwidth or lines, never a bit of a result.

**H1. A speculative lead over predicted experts.** The lead can only run
ahead on fixed-class bytes, about 55 MB of a 731 MB layer at C8 and the
reason the C8 experiment found it worth 3 to 6 points, and the fixed
phases idle at their edges because no byte of the next phase may be
touched before its router. A guess at the next layer's routed experts
gives the lead unbounded material: idle workers stream predicted expert
units during any sub-40-SM stretch, and the chain's run list supersedes
the guess when it lands. The economics are better than a recall figure
suggests. The lead streams only when workers are idle, when the memory
would otherwise be idle, so a wrong guess costs L2 lines rather than time,
and the governor and the L2 cap bound it. Experiment: `dsv41_forward_check`
already dumps every walked layer's routed expert ids per token over a
prompt on the real checkpoint at world 1 through the streaming loader, so
three texts of about two thousand tokens each (prose, code, JSON) give the
routing of the model as served, and an analysis script measures, per
layer, how many of the six true experts fall inside predicted sets of 6,
12 and 18 from three predictors the streamer could hold: the same token's
experts at the previous layer (a co-occurrence table), the token's
identity (a per-layer table, scored leave-one-out), and the previous
token's experts at the same layer (temporal locality, known a whole step
early). Decision: recall of 60 percent or more at a set of 12 from any
predictor available in time builds the speculative lead as a pool of
predicted units in the lead's sequence, below 40 drops it, between the two
it goes through the replay model first.

*Measured 2026-10-30 evening*, three texts of 1,637, 2,032 and 2,430
tokens (prose, code, JSON) through `dsv41_forward_check` on three idle
nodes, 40 routed layers, six experts a token. Recall of the true six inside
a set of 12, mean over layers and texts: the layer's most popular experts
11 percent (the load is balanced, no hot set), the same token's experts at
the previous layer 25, the previous tokens' experts at the same layer 40,
the token's identity 56 (leave-one-out inside the text, 46 to 72 by
layer). Restricted to tokens the text had already seen, identity reaches 58
on prose, 64 on code and 67 on JSON at a set of 12 and 63 to 75 at 18, and
such tokens are 66, 91 and 95 percent of the three texts, so a table kept
over the deployment's life approaches those figures. Identity is also the
predictor available earliest: every row's token is known when the step
begins, so the guess for every layer is ready before the layer. The
verdict is the band between the rules, and the bound on the gain is the
lead's L2 cap rather than the recall: 16 MB holds about three experts of
the twenty a five-row layer reads, so with the measured recall a staged
set yields about 9 MB of useful bytes a layer, about 37 µs of memory time,
about 1.6 ms a pass or 4 percent of the floor. The data also gave pillar 2
its number: five consecutive rows read 19 to 22 distinct experts a layer,
so a row past the first costs about 3.3 experts a layer, about 0.6 GB a
pass, about 2.7 ms at the ceiling. A fourth text (1,533 tokens of prose)
left identity at 56 and lifted the previous layer to 31 and the previous
tokens to 42 at a set of 12. H1 waits behind H2's finding below, which
removes most of the stretches it would fill.

**H2. A streaming unit with more bytes in flight.** The plateau's
mechanism is one SM at 5 to 6 GB/s under load because its bytes in flight,
16 KB a block at four 16-byte loads a lane, sit against the loaded
memory's latency. Bulk copies into shared memory behind an mbarrier
pipeline put most of the 99 KB a block in flight from one issuing thread.
If sixteen SMs could then saturate the memory, a sub-40-SM stretch would
no longer leave it idle and the other SMs would be free for the chain. The
probes set TMA aside because deep rings interfered with the fold's host
polling, a finding from a different structure that this one has to
re-measure. Experiment: a scratch microbenchmark streaming disjoint
regions from 1 to 48 blocks in four forms, the plain four-deep ring of
today, plain rings eight and sixteen deep, and bulk copies two and three
stages of 32 KB deep, each cold, with the device's utilization recorded.
Decision: a form that reaches 200 GB/s from sixteen SMs where the plain
form does not becomes a staged variant of the projection unit (the
consumer reading its fragments from shared memory) and the fixed phases
are re-measured with it, a plain ring that reaches it merely deepens the
ring where registers allow, and neither reaching it closes the question
in favour of the fused chain alone.

*Measured 2026-10-30 evening on the idle device.* The premise was wrong
in a useful way. A plain streaming block, 256 threads with the streamer's
`__ldcs` 16-byte loads four deep a lane, reads 52 GB/s from one SM cold,
the same with rings eight and sixteen deep, and eight such blocks saturate
the memory at 236 GB/s. Multi-stage `cp.async` into shared memory reaches
75 to 81 GB/s from one SM and bulk copies of 32 KB stages 61 to 76. So
bytes in flight do not bound an SM near 5 GB/s, and sixteen SMs were never
needed: eight suffice for any form, and the plan's SM budget can give the
chain and the attention forty. The projection unit's own access pattern
without its compute, sixteen rows by 32 bytes a warp-load, streams 31 GB/s
from one SM and caps the whole device at 160 GB/s, because every request
is one sector of one line; the bridge's 512-byte L2 prefetch a row lifts
it to 54 GB/s from one SM and 236 from eight, which is why the bridge was
found necessary, and runs of 64 bytes or more a row saturate without it.
Bulk copies of 256-byte row pieces are the one bad form, 5 GB/s a block,
each small copy paying the async proxy's overhead. So the unit's 16 GB/s
an SM L2-hot and 5 cold are its instruction stream, and `ncu` on the hot
unit says where: 124 warp instructions a 512-byte chunk, so even perfect
issue caps an SM at 35 GB/s, and 7.0 cycles an issued instruction with
only two warps a scheduler, of which 1.9 are long-scoreboard stalls, 1.65
the block barrier and 1.2 fixed-latency waits. The long-scoreboard stall
is the per-block scale: with 32-row blocks every chunk loads the next
scale from L2 and the next chunk's accumulate depends on it. The barrier
is the activation window's, every eight chunks. The conversion of e4m3 to
bf16 is about half the instructions. Decision: rebuild the unit rather
than its loads, in this order, each step measured by the hot and cold
per-SM rate in `step_consumer_test` and the layer's share of its floor in
`step_stream_test`: the scales of a warp's rows staged in shared memory
once a unit, the activation slice staged once a unit when the group has
one token group so the chunk loop has no barrier, the ring deepened where
registers allow, then the conversion's instruction count. The target is
40 GB/s an SM so that six units saturate the memory, which makes a phase
of a few matrices stream at the ceiling without the fused chain, and the
chain's job becomes latency alone. Bulk and multi-stage `cp.async` forms
stay for the attention arena, where the per-SM rate from a single issuing
thread is the point.

*Built the same evening.* The staged projection form (the scale row and
the activation slice staged once, no barrier in the chunk loop, each lane
loading its own fragment's bytes so the shuffle network is gone, the
block's first mma with a zero C operand, the e4m3 decode through the
conversion unit) runs the hot unit at 25 GB/s an SM from 16, bitwise the
windowed form. The decode step exposed a defect: the integer rebias every
fp8 form used was wrong on the fourteen subnormal e4m3 codes, which real
DeepSeek-V4.1 shards hold at about one code in ten thousand and the
synthetic test data had never held; every form now decodes as production's
fp8 GEMV does, the test data holds exponents 0 to 14, and the stream's
exactness gates held. The layer did not move with the unit alone, and the
per-phase utilization the trace now reports said why: workers were busy
60 to 70 percent of the fixed phases and 91 of the expert phase, and the
hand-out itself cost 3.5 to 4.5 µs a grab under a saturated memory (1.3
in a quiet phase), its dependent loads of the phase and pool tables.
Three changes followed, each measured: the phase's hand-out table built
once a worker in shared memory (a grab is the counter's load and the
atomic, 1.0 to 1.6 µs), lead units that stream in 16 KB pieces and yield
when a phase is published (phase starts had waited behind lead units at
the memory's share), and slabs of at least 640 codes for the projections
(the attn_out drain fell from 56 to 15 µs). Workers are now busy 83, 78
and 96 percent of the qkv, attn_out and expert phases, the expert phase
streams at 92 percent of the ceiling, the fixed phases at the ceiling in
aggregate, and the median layer at 87 to 88 percent of its floor at any
lead cap from 4 to 20 MB (beside production, at the uniform expert count;
the clean figure at the measured count, 2026-10-31, is 83 with the
synthetic router, 80.6 with the real one, 78.4 with the mHC sites computed
as well). The residue is the layer's latency: attn_in 9,
attend 2, ffn_in 12, the router 18, the gap to the experts 8 and the
accumulation 21 µs, about 70 µs a layer, of which the lead covers a few
microseconds, because its cap is full when the latency begins. The bound
is structural: the L2 cap of about 16 MB that survives the expert phase
is 67 µs of bytes, the layer's idle time (latency plus drains) about 100,
and bytes parked across the expert phase hold the cap for its 650 µs. So
the remaining 12 percent comes from three places: the forms' own rate (233
to 248 GB/s against the floor's 240), the latency segments the fused chain
shortens (the accumulation's 21 µs first), and the lead's use of the cap
around them.

*The lead's accounting, later that night.* The trace gained the lead's
bytes per window and the bytes actually staged for each phase, and the
first thing they showed was that the earlier counts were wrong: a lead
unit cut short by a yield had counted as staged in full, and the yield
fired on every phase-word change, so the lead looked twice as effective
as it was. With latency phases published to the workers, staged bytes
credited as units consume them, yields only to a stream publish and the
lead held within the next layer, the lead stages about 22 MB a layer
(qkv 7, the shared expert 7, the router 4, attn_out 3) in idle windows
worth about 23 MB at the ceiling. The idle windows are therefore nearly
spent, and the lead cannot buy much more: the fixed phases' remaining
time is their cold bytes at the ceiling plus tails and hand-outs, and the
accumulation's 21 µs is the design's own accumulate and fold latency, not
its units. Two experiments closed side questions: staging with the
evict-last priority changed nothing (a probe showed evict-normal lines
surviving a 150 MB evict-first stream at 92 to 95 percent), and reducing
a slabbed projection in the pool's last arriving worker made the largest
pool's reduce the phase's tail, so reduce units stay units, handed out
after every pool's compute units. The layer stands at 89 percent of its
floor, plus or minus two between runs, and the residue is now mostly the
expert phase: 642 µs for bytes the forms read in 596 as plain grids and
the floor counts at 604, so the pair's request granularity (233 GB/s
against the down's 247) and the grabs' prologues are the next levers,
with the fixed phases' tails after them. The fused chain's value for the
C1 target is accordingly smaller than the plan assumed, since the lead
already covers the design's latency, and its value is the latency it
removes from the pass itself. A last probe cleared the pair's request
pattern: two streams a warp 4 MB apart read 238 GB/s at 48 blocks, as do
one stream of 512-byte runs and one of 1 KB runs, so a pure stream's
ceiling on this device is 237 to 238 and the pair's 233 is 98 percent of
it. The expert phase's remaining 4 percent of the layer is the stream's
structure around the units, not the bytes.

**H3. The owned lead.** The lead table of section 2.1 counts 23 MB and the
L2 part, 16 MB, is built. The other 7 MB is register and shared-memory
staging owned by the CTA that will consume it, which needs the stationary
tile assignment of the fixed matrices that phase 3 lists. It hides about
30 µs more latency a layer and lets the fixed phases run from on-chip
bytes. No experiment decides it, the fused chain's measurement does.

**H4. Fewer bytes a pass, losslessly.** Each is a few percent of the floor
and they compound.

| Cut | Bytes a pass per rank | At the ceiling | Condition |
|---|---|---|---|
| the ranges every rank reads whole (wq_a, wkv, the indexer, the compressor, the router, the mHC coefficients) sharded, their outputs gathered by the fold | about 685 MB read today, about 510 MB saved | about 2.1 ms saved, one more fold a layer at phase 4's 15 µs costs 0.65 ms | the in-kernel fold of phase 4 |
| bf12 companions for the remaining bf16 fixed ranges (the Engram projections, the hyper-connection coefficients, the indexer, the compressor, the router) | about 70 MB | about 0.3 ms | the bf12 form as a consumer, exact by construction |
| the 2-bit screen on the head's two reads | about 180 MB a read | about 0.75 ms a read | the bound that makes the screen exact has to be proven, not assumed, and the second read's inputs depend on the first's argmax, so the reads stay two |

Together they move the floor from 36.3 ms toward about 33 ms, and the same
107 percent pass becomes about 1.35x. The first brainstorm of the day
listed reading the head once for the verify and draft rows as a fourth
cut. It is not one: the draft block's rows take the tokens the verify
read's argmax picks, so the second read cannot start before the first
ends, and the screen is the lever on it.

**H5. Tokens a pass.** A C1 pass reads 8.7 GB whether it yields three
accepted tokens or five, so every accepted token above today's average is
worth more than the whole of this pillar. Rows cost experts: each verify
row adds about five distinct experts a layer, about 1 GB a pass, about 4
ms, so wider trees pay only where acceptance is high. This is pillar 2's
depth schedule, and what the step engine owes it is the number the
schedule needs, the measured distinct-byte cost of a row by class from the
floor meter.

**H6. Smaller items.** The ceiling has about 5 percent of play by access
pattern, 233 GB/s for the 32-byte runs and 248 for the tile-major down, so
a DRAM sector-efficiency measurement under `ncu` per consumer form tells
how much of each form's traffic is partial sectors, and the floor should
carry the measured ceiling of the form it uses. Locked clocks cut the fold
skew's tail by a quarter in the phase-0 decomposition and are an
operational setting. The fused chain's own latency budget is 72 µs a
layer, so every 10 µs saved there is 1 percent of the pass, the mHC site
being the largest piece.

*H6 measured 2026-10-30 evening* over the consumers' plain grids, cold:
the tile-major MXFP4 forms miss exactly their bytes at L2, 415.2 MB of L2
read misses for a 414 MB pair sweep and 207.2 for a 207 MB down sweep,
with 2.9 and 3.9 sectors a request, so nothing is over-fetched and the 233
to 248 GB/s spread between forms is request granularity, not partial
sectors. The fp8 and row-major forms issue 1.05 to 1.09 sectors a request,
one line each, 2.5 times the requests a byte of the tile-major form, and
their memory reads hide behind the bridge's prefetches where read misses
cannot count them. The GB10 exposes no DRAM-side counters to `ncu`, so the
L2 miss count is the deepest observable, and the floor keeps the ceiling
it has.

*The router landed the next day* as the fused chain's first real piece:
a consumer pool computing the production router's dot in its lane order,
the chain's select op for op, every score, pick and weight bitwise the
production kernel's over eight layers (`step_router_matches_the_production_kernel`).
It costs about two points of the layer at the measured expert count, its
units latency-bound at one block an SM, and it exposed two more facts: the
step kernel runs at the full 255 registers a thread, and the chain's
writes and fences crawl while the lead saturates the memory (holding the
lead while the chain routes halved the select's time and cost the lead as
much as it saved, so it is not kept).

The order of work was a measurement sprint of the three experiments that
decide a direction (H1's routing predictability, H2's per-SM rate by form,
H6's sector efficiency), run the same evening with production down at the
user's word and the three routing dumps on the other nodes. Its outcome
reorders the program: the projection unit's rebuild (H2) comes first,
since it attacks the plateau directly and cheaply, the fused chain keeps
its place as the latency work, and the speculative lead (H1) and the
remaining byte cuts follow.

### 1.1 A second engine behind a flag

The step engine is new code beside the engines that exist, selected per
template by `engine.decode_engine`, whose values are `graph`, the default
and today's behaviour, and `step`. Nothing that serves today is modified:
not the kernels, not the model walks, not the graph or eager engines, not
the loader, not the bus's eager, graph and bulk machines, not the
scheduler's existing paths. A template that does not set the key cannot
observe that the step engine exists.

New code lives in three places. `src/step/` holds the runtime: the plan,
the streamer and its governor, the hand-off primitives, the fold client,
the descriptor and verdict protocol, the trace ring and the phase-by-phase
oracle. `src/step/<family>/` holds each family's plan generator, consumer
bindings and exactness fixture. `src/kernels/step/` holds the consumers.
Tests are `step_*` binaries with loopback ports allocated past the range
the existing gates own.

The step engine reuses existing code only by reading it. Weight ranges and
format metadata come from the resident views the prefetch windows already
ask for through `IGemm::resident_view`. Format decode helpers are included
headers, not edited files. The bus's queue pairs, slabs, credit machinery,
the write ring's wire format and the flag protocol are used as they are.
The scheduler sees one more `SchedulerEngine` implementation.

Shared files the program does touch, every change additive, with the
places as the code stands on 2026-10-30: the configuration parser's
`engine` branch in `src/serve/cluster_config.cpp` and the `Engine` struct
in its header, the key table in `docs/configuration.md` under "Graph
execution and batching" where `engine.decode_graph` has its row, the
`WorldSettings` record and its journal encoding in `src/serve/fabric_serve.hpp`
so every rank carries the same engine choice, the branch in
`apps/dgpp_serve.cpp` that today selects the graph adapter by
`decode_graph` and the `ServeFamily` makers beside `make_graph_engine`,
the loader's `LayerBump` in `src/loaders/weight_build.hpp` for one
placement hook and the image key in `src/loaders/resident_stream.hpp` for
the engine's layout version, the `SchedulerEngine` interface in
`src/sched/scheduler.hpp` for one stats virtual with a default and the
scheduler's `Meters` for the three quantities above, the bus engine for
one new machine beside the three it has, the `dgpp_kernels` file list and
the test registrations in `CMakeLists.txt`, and the documents. There is
no engine factory and no central port table today: the selection is one
`if`, and each test owns its ports, so the step engine's ports are listed
in `docs/testing.md` as the others are. Each of those edits is reviewed
against this list.

What the rule buys: an A/B on one binary by flipping one key, every
existing campaign row staying valid, families switching one template at a
time when their gates are green, a fallback at every point in the year,
and no removal of the graph engine inside this program. What it costs:
GLM-5.3-Flash's mixed verify is not reproduced in the step engine, each
family's decode math exists twice, once in its walk and once in its
consumers, and a template that flips the key rebuilds its resident image
once, since each engine has its own weight layout. Those costs are
accepted.

## 2. The design

### 2.1 One stream, one chain, an on-chip lead

Today the byte stream and the dependency chain are the same thing. Each
kernel reads its own weights when its turn comes, so every latency segment
of the chain is a hole in the stream and every small GEMV pays a ramp from
zero and a drain to zero. The two-term instrument of 2026-10-20 on
DeepSeek-V4.1 put the loss at 174 µs of gaps per layer, GEMVs 10 to 50
percent below their isolated rate inside the DRAM phases, and the down
kernel at 72 to 78 percent of the ceiling, against 617 µs of bytes in a
967 µs layer.

The design decouples them. One stream reads the step's bytes in consumption
order at the ceiling. One chain consumes bytes that are already on chip.
The stream leads the chain by as much as the chip can hold, and the lead is
owned, not pooled:

- **L2 is the only staging shared across CTAs.** Up to 18 MB of it is made
  persisting and holds the routing-independent ranges that other CTAs will
  consume. The remaining 6 MB is the working set: KV lines, activations,
  and the lines that TMA copies pass through on their way to shared memory.
- **Shared memory and registers are owned by the CTA that will consume
  them.** Within a CTA, producer warps issue the TMA copy of tile t plus
  one while consumer warps compute tile t, double buffered at about 40 KB
  a tile, and a register slab holds the first k-step of the next GEMV the
  CTA is assigned. A CTA never stages for another CTA. Which tiles a CTA
  owns is fixed by the plan, so its producers know what to fetch during a
  latency segment.

| Lead | Capacity | Stream time at 240 GB/s |
|---|---|---|
| L2 persisting share | 18 MB | 75 µs |
| Shared memory, one tile ahead on 48 CTAs | about 2 MB | 8 µs |
| Registers, one k-step ahead on 48 CTAs | about 3 MB | 12 µs |
| Lead | about 23 MB | about 95 µs |

The fused chain of section 2.9 has about 70 µs of latency per layer. The
lead covers it with a margin of about 1.35x, not the 2x a pooled sum would
claim. Two things decide whether that margin is real, and both come before
any consumer is built. A compiled kernel with representative consumers and
live staged weights reports its registers, spills, occupancy and usable
lead. And the stream replay model of phase 0 plays real layer traces, with
the release times of routers and selections, the lead's occupancy as a
buffer that must drain before it refills, and the governed rate in each
phase class, and predicts the layer time. The fixed-gap synthetic of the
first draft of this plan is gone: a storage figure covering a latency
figure proves nothing about sustained overlap.

### 2.2 One kernel per step

The step is one cooperative kernel. No graph, no variants, no batch
families, no capture. The host writes a step descriptor into a pinned ring
of two slots and the kernel reads it, runs every layer, the pick, the
commit, the draft chain and the head, and publishes the verdict and the
drafts into a pinned ring with a system-scope release. Token feeds stay on
the device: a step whose descriptor has no admissions, retirements or masks
reads its tokens from the feed the previous step wrote.

The kernel is launched early and bound late, and the binding is a protocol
rather than a wait. Each descriptor slot carries a state the host publishes:

- **pending**: the kernel's control warp polls, every other warp waits on a
  device flag the control warp raises.
- **execute**: the kernel runs the described step.
- **skip**: every block exits at once with a skip status. The host uses it
  whenever the device is needed for work that cannot run beside a resident
  step kernel: an eager latency-class collective such as the sampled
  fallback's exact gather, a prefix-cache attach or detach, or a prefill
  chunk under the time-slicing mode. The host then does that work and
  launches a fresh kernel.

Ownership follows the states. A descriptor slot belongs to the host until
it publishes execute, to the kernel until it publishes the verdict, and to
the host again. The device feed written by step N is read by step N plus
one and untouched by a skipped kernel. Between steps the host works in a
fixed order on every rank: fallback gathers first, since they finish the
previous step's sampling, then admissions, retirements and cache hops, then
prefill launches, then the next descriptor. This is the drain rule the
graph engine enforces before any eager bus work, kept as a protocol instead
of a method.

The next kernel is enqueued on the same stream with a programmatic
dependency, the mechanism [graph_pdl.hpp](src/engine/graph_pdl.hpp) already
uses between kernels, so its blocks may schedule as this step's blocks exit
and its prologue may read the descriptor during the drain. That overlap is
opportunistic. The protocol is correct when the two kernels serialize
completely, and the 10 to 15 µs inter-step gap is a measurement target of
phase 1, not an assumption. A kernel that never exits is not the design: it
cannot yield the device and cannot be restarted legibly.

The kernel runs on a stream of a green context holding `engine.step_sms`
SMs, 48 by default. All of this is reached only when `engine.decode_engine`
is `step`.

### 2.3 Roles by phase

Warps take roles, and roles change by phase through the work list rather
than being fixed at launch, because the attention over a long context is a
stream in one phase and a latency kernel in another.

- **Streamers.** Within each consumer CTA, producer warps fetch the
  ranges the CTA owns next. The phase-1 probes of 2026-10-30 chose the
  load form: plain vectorized loads with evict-first hints reach 250 GB/s
  at 40 blocks and leave an L2-resident window intact, and they cost the
  fold's host hop a few microseconds, while TMA bulk-copy rings deep enough
  to reach the ceiling cost it tens of microseconds. So the stream is
  plain loads into registers and shared memory, with TMA kept for small
  chunks where a producer warp must not spend registers on loads in
  flight. Sixteen to forty CTAs streaming at once is where the probes saw
  the ceiling.
- **Consumers.** GEMV, expert and attention consumers whose weights or keys
  are already staged when the input arrives. During the routed-expert
  phase the CTA is both, since that phase is the stream.
- **Latency warps.** Norms, mHC and GR hyper-connection sites, routers,
  selection, rope, state updates, the KV append, the EXL3 activation
  rotation, the a8 activation quantization. Fused so a layer has about four
  hand-offs instead of a dozen. They live on SMs whose CTAs are not
  holding register slabs in that phase.
- **Fabric warps.** One sentinel lane per peer polls the incoming row, one
  warp stages the partial. The host proxy posts.
- **Control warp.** Reads the descriptor and its state, waits on
  host-published sequences for masks and gathered tables, publishes the
  verdict, drives the deadlines.


The expert streamers' form is now measured rather than designed (section
1.3): a warp owns a 16-row tile of a tile-major expert matrix and every
token of its group, keeps four lane-contiguous 512-byte loads a matrix in
flight in registers, and feeds its tensor-core fragments by shuffle; no
shared-memory stage sits between the load and the fragment, and the
activations are the only shared-memory traffic. Sixteen to forty such
warps, one CTA an SM, stream a gate and up pair at 94 to 96 percent of the
ceiling at one to three token groups and the down projection at 90 at one
group. The fixed ranges keep the plain evict-first streamer described
above until their consumers are measured the same way.

### 2.4 The stream plan, its dependency classes and the arena

The stream plan is a list of ranges with per-range metadata: format,
geometry, scale layout, destination staging class, consumer, and the
dependency class that says when the range's address is known:

- **Fixed**: known at load. Projections, norms, routers, hyper-connection
  weights, indexers, the shared expert, the head.
- **Token-bound**: known at step start. Embeddings and the host-gathered
  tables of section 2.11.
- **Router-bound**: known when the layer's router writes its run list,
  experts deduplicated across rows and sorted into the canonical
  accumulation order. The run list is released as a few ranges, not one
  per expert: the C8 experiment of section 1.3 priced a hand-off and a
  tail per expert at 3 to 5 percent of the layer at 143 distinct experts,
  so consecutive experts in the canonical order are released as one range
  whose tiles the workers grab in sequence, and the per-range cost is paid
  a handful of times a layer whatever the row count.
- **Selection-bound**: known when a layer's selection finishes. On
  DeepSeek-V4.1 the eight index-source layers and the candidate layer
  compute their selection from the current layer's query and only then
  address the KV, as [csa2_layer.cu](src/models/dsv41/csa2_layer.cu) does
  at its decode path, while the other 32 layers reuse that selection and
  their KV ranges are known a layer early. GLM-5.3 full selects on 21 of
  78 layers and reuses on 57. GLM-5.3-Flash's 11 DSA layers and
  Qwen3.8-Flash-Next's 12 QSA layers each select for themselves.

The replay model and the streamer both work from these classes. A
selection-bound range cannot be in the lead before its selection, so on a
selecting layer the attention's keys are read in the open at the ceiling,
and the lead's job in that layer is the routing-independent bytes around
it. Ranges need no tensor maps, only 1D bulk copies with mbarrier
completion, which sm_121a supports.

The arena is the step engine's weight layout and comes with the key. Each
rank's layer is one contiguous arena holding its tensors in the order the
step consumes them, so the dozens of small fixed-class tensors of a layer
are one range for the streamer rather than dozens, and the thin TP
projections of a layer sit adjacent.

The arena is a placement policy, not a second loader. The existing loader
reads, shards, validates and digests the checkpoint exactly as it does
today, and the step engine supplies the placement through one additive hook
in the bump allocator: the plan generator knows every tensor's size from the
checkpoint metadata before a byte is read, reserves the layer's arena, and
hands the loader each tensor's offset as the loader asks for it. Every
tensor keeps its internal layout, scale geometry and alignment, so the
layer objects point into the arena and prefill, the eager oracle and the
bf12 companions read the same bytes at the same addresses with no copy and
no second resident form. The routed experts are the one exception to the layout rule, on
the phase-1 measurement of section 1.3: each expert matrix is stored in
16-row tiles with its 32-code chunks contiguous, the layout the
tensor-core consumer streams at 94 to 96 percent of the ceiling where the
checkpoint's row-major form reaches 90 with an L2 prefetch bridge. The
tile-major form holds the same bytes in a different order, so the image
is the same size, and the prefill path's expert consumers read it through
an address map rather than a second copy. Every other tensor keeps its
checkpoint layout. The resident image written from the arena is keyed by engine and
layout version, so a template that switches engines rebuilds its image
once and the two layouts never alias.

### 2.5 The governor

The memory controller is one queue shared by the GPU, the CPU and the NIC,
and 3 MB of prefetch in flight added 12 µs to every access on the box. So
the streamer runs at the ceiling when the chain is also streaming and at a
governed rate when the chain is in a latency segment or a fold, with bytes
in flight as the knob. The prefetch windows found this operating point by
hand and call it the light rate, about 150 GB/s. In the design it is one
scalar per phase class, swept once per family and set in the template as
`engine.stream_governor`, and the replay model carries the governed rate so
the lead's refill during latency segments is modelled at the rate it will
actually get.

### 2.6 Hand-offs

The probe saw a device flag reach the first waiting warp in 1 µs and the
last in 8 µs beside a prefetch window. The design has about four hand-offs
per layer on the critical path. Each is one L2-resident flag polled by one
dedicated warp per consumer group with nothing else in flight, with the
streamer's traffic on TMA so it does not sit in the pollers' load queues.
Data hand-offs use the data: a consumer polls the generation stamp on the
row it needs, the trailer pattern the fabric already uses, so there is no
separate round trip. Every spin carries a cycle deadline and exits through
a common path that publishes a status, the rule the collective kernels
follow today. Budget: four hand-offs at 2 µs, 0.3 to 0.6 ms a step.

### 2.7 Consumers: one form per site

Every site has one kernel form for every row count from one to the family's
cap, and that form is the streaming tensor-core form with weights staged by
TMA. The exactness set of 2026-10-05 found six transcript divergences, all
from kernel forms that changed at a row count, and the streaming MMA form
was shown bitwise invariant from one to 64 rows because each output
element's reduction over K is the same whatever the row count. A one-row
step runs a sixteen-row tile with padding rows. The tensor core is idle
either way and the bytes are the same.

The compute budget at the ceiling is 240 GB/s over 48 SMs at about 2.4 GHz,
about 60 lane-operations per byte. That is 30 per weight for a 4-bit format
and 27 for EXL3's 3.5 bits, including the multiply-accumulates for every
row. Decode cost per weight is a few operations for fp8 and NVFP4, more for
the codebook and trellis formats, so the format bench of phase 1 measures
every format's rate from staged tiles at every row count before a consumer
is accepted. The EXL3 activation rotation stays ahead of the consumer, on
latency warps, because the fusion record shows a rotating warp that is also
a computing warp costs 30 to 60 percent of the kernel. Here an idle warp
carries it. The expert consumer accumulates in the canonical order, one fma
per expert ascending with the shared expert last, whatever order the stream
delivered them in.


What the first measured consumer fixes for the rest (section 1.3): a
consumer's cost is counted in pipe cycles a weight byte, summed over the
ALU, shuffle, shared-memory and tensor pipes, because two warps a
scheduler overlap them only partly; the budget is about 600 SM cycles for
every 2 KB an SM consumes. Weights are decoded once per chunk for every
row of the group, as tensor-core A fragments, so the row count adds only
MMAs and B-fragment loads; a pair of matrices sharing one activation
operand, gate and up, halves those per byte and is the form that reaches
the ceiling at 32 rows, while a lone matrix pays 11 SM cycles a chunk a
warp for each token group of eight and falls to 65 percent at 32 rows.
Exactness is at rounding level against the GEMV kernels, two bf16 ulps,
from the MMA's accumulation order; it is bitwise against itself. The
ordered accumulation of a token's routed and shared partials reads seven
values a hidden element; the step engine's down consumers write their
partials token-major, the seven of one element in one sector, so the
accumulation is one coalesced read per output. In the phase-1 prototype
the layout made no measurable difference, because that phase's cost there
is the probe's mechanics, not its loads.

### 2.8 The fold inside the kernel

Every family folds twice per layer, and only Qwen3.8-Flash-Next's n-gram
layer and DeepSeek-V4.1's two Engram layers add more. The two folds stay.
There is no algebra that merges them, since the norm before the router
needs the full residual and so does the norm before the next attention.
Reduce-scatter forms add messages and save nothing at these payloads, and
replicating attention to skip a fold costs four times the attention bytes.
The fold is made cheaper on its critical path and its skew is hidden
instead of removed.

The in-situ decomposition of 2026-10-20 on DeepSeek-V4.1 at C1, 35 KB a
rank, is the baseline: copy 4 plus handshake 15 to 21 plus skew 11 to 15
plus fold 0.5, for 31 to 39 µs a collective, with the path itself 13 to 16
µs each way out of the symmetric sums. The path is four PCIe crossings
around a 4 µs wire: the GPU's copy into pinned staging, the NIC's read of
it, the peer NIC's write into host memory, the peer GPU's polled reads.

| Fold component | In situ today | Design | How |
|---|---|---|---|
| Copy into staging | 4 µs | 0.5 to 1 µs | the consumer's epilogue stores the partial straight into the pinned staging row, 16 blocks |
| Notice and post | under 2 µs | under 2 µs | one host flag, prebuilt work requests per fold slot, only the doorbell varies |
| NIC read, wire, NIC write | about 6 to 7 µs at 35 KB | about 6 to 7 µs | physics |
| Polled read and prove | gate pass about 4 µs a peer, one block | 2 to 3 µs | 16 blocks read, prove and fold in one pass over the trailers |
| Fold | 0.5 µs | 0.5 µs | canonical rank order |
| Critical path before skew | 20 to 25 µs | 11 to 14 µs | |
| Arrival skew | 11 to 15 µs median, p90 38 and 106 µs | hidden where the lead has room, paid where it does not | section below |

The skew is not a launch artifact and the design does not claim to remove
it. The decomposition found it to be the ranks' compute imbalance on
identical work, correlated 0.98 to 1.00 with the compute between
collectives, with no rank systematically late, unchanged with prefetch
off, worse with programmatic launches off, and a quarter smaller in its
tail with locked clocks. It is the GPUs' timing noise. What the persistent
design changes is what a rank does while it waits: its streamer keeps
filling the lead, so the skew costs step time only when the lead is full.
That is a hypothesis about the lead's occupancy, and the replay model and
the phase-1 experiment test it, with the fold measured after representative
compute on all ranks at H and 5H payloads and its distribution recorded at
p50, p90 and p99 on two and four nodes. An isolated small-payload
collective does not validate the budget.

The step's fold walk is a fourth machine in the bus engine beside the
eager, graph and bulk machines, sharing their queue pairs, slabs, credits
and the write ring's wire format and changing none of them. The GPU
publishes the staging hash itself, as the bulk path already does. Widths
are H for the boundary folds and 5H for the Engram folds, and the staging
region is sized by the widest fold of the walk, the lesson the halves
experiment paid for. The pick is one more fold with the digit table as
payload. At 15 µs a round before skew the qkv_a sharding closed on
2026-10-09 reopens on full GLM and pays about 3 ms a step. At world 1 the
fabric role is empty and the fold is the identity.

### 2.9 The fused chain

| Phase on DeepSeek-V4.1 | Today | Target | What changes |
|---|---|---|---|
| Residual update and mHC | 20 µs | 8 µs | computed by the fold consumer in shared memory as the peers' rows land |
| Selection and attention | 30 µs | 22 µs | index scores and select in one CTA group, keys read in the open on selecting layers, from the lead on reusing layers |
| Attention fold | 48 µs | 15 µs before skew | section 2.8 |
| Update, mHC, router, slot order | 43 µs | 12 µs | one CTA group, no global round trip, router weights L2 resident |
| MoE fold | 32 µs | 15 µs before skew | section 2.8 |
| Total | 174 µs | about 72 µs before skew | |

The router finishing is the only hard dependency in a layer. The moment the
run list is written the streamer starts the first expert range while the
remaining rows' top-k complete.

### 2.10 KV, state and the speculative transaction

KV is bytes too. At a 100K context full GLM reads 4.4 GB of KV a step and
the dense 27B reads tens of gigabytes, and that is in the floor. The
attention consumer streams it the way the expert consumer streams weights:
TMA, deep pipelines, sixteen or more blocks in flight, split partials merged
in L2 by the last block rather than written back.

Recurrent state is written in replay form, not snapshot form. GLM-Flash's
KDA and Qwen's GDN snapshot their fp32 state and convolution history after
every verify row for rollback, and GLM-Flash's state is about 34 MiB per
request, so eight rows write hundreds of megabytes a step that are not in
the model's bytes. The replay form that shipped for Qwen on 2026-10-05
recomputes from a checkpoint at the accepted row and removed those writes.
Every stateful family takes it.

The MTP transaction runs inside the kernel as it runs inside the graph
today: verify rows, the pick, the commit of the accepted row's state, the
draft block over the accepted rows, the chain rows with positional and
provisional writes, and the tail ring copied aside and restored where a
chain row would stash over an open pool. A sampled verdict that the
candidate table cannot resolve ends the step at the verify point with a
fallback status and no draft chain, the host runs the exact gather and
samples, and the next step runs that slot plain.

### 2.11 Host tables at step start

Two families read host memory mid-step. DeepSeek-V4.1's Engram layers
gather hashed n-gram rows through a declared host node, and
Qwen3.8-Flash-Next's n-gram layer gathers rows of an mmap'ed table through a
host thread behind a pinned sequence. A persistent kernel has no host nodes.
Both gathers depend only on the step's tokens, which are known at step
start, so the control warp publishes the ids as its first act, a host
thread gathers into pinned memory and publishes a sequence, and the
consuming layer waits on it. Layer 1 is the earliest consumer in both
families and the gather has the whole of layer 0 to land.

The gather's pages must be resident. The decomposition found the Engram
host gather page-faulting on one rank in 12 percent of its collectives,
p90 363 µs of imbalance, and a consumer that waits on its own rows rather
than the whole staging cut it to p90 57 µs. The step engine's gather
therefore populates and locks the table's pages it can afford, reads the
rest ahead as the NVMe tier does, and the consuming layer waits on its own
rows only.

### 2.12 Prefill beside the step

Decode saturates the memory at 16 to 24 SMs. A prefill chunk is compute
bound and reads each weight once per chunk, a few gigabytes a second. The
mechanism that lets them share the device is a green context, not a block
count: CUDA assigns a grid to SMs by resources and placement, never to a
requested subset, but a green context is a real partition. Measured on this
box on 2026-10-30, two green contexts of 24 SMs each ran a spinning kernel
at the same time on disjoint SM sets, and a cooperative grid launches
inside one. The step kernel runs on the stream of a green context holding
`engine.step_sms` SMs and prefill chunks run on a second green context
holding the rest.

Prefill in that mode is background work, and the scheduler's ordinary
advance returns synchronously, as `Scheduler::advance_prefill` does today,
so the contract is explicit. The step engine adapter declares a
background-prefill capability through an additive pair on
`SchedulerEngine`: a launch that returns at once with the chunk in flight,
and a poll. The scheduler's branch for that capability, used only when an
engine declares it, accounts a chunk's tokens at the tick that observes its
completion. Completion is timing, and timing may not steer the lockstep,
so rank 0 observes it and journals the completing tick, and a peer applying
that record waits for its own chunk before it proceeds, the way admissions
already flow from rank 0 to the peers. A cancelled reader's in-flight chunk
completes and its result is dropped, as chunk cancellation does between
ticks today. Each partition owns its workspaces. The bus runs the step's
latency folds and the chunk's bulk folds at once, so the one outstanding
collective rule becomes one per class.

L2 is the shared resource the partition does not split. The persisting
window gives the stream's lead up to 18 MB that the chunk's GEMM tiles
stream past evict-first, and the release is per line through
`applypriority`, since the context-wide reset needs an idle device and
deadlocks against a spinning peer.

Step-boundary time slicing, the mechanism in use today, stays as the mode
a template selects when the coexistence gate of phase 6 does not hold for
its family. GLM-5.3-Flash is the one family whose decode walk carries
prefill reader rows inside it, the mixed verify of
[mixed_prefill.hpp](src/engine/mixed_prefill.hpp). The step kernel is
decode only and does not reproduce it. The mixed-workload gate runs before
that template, and every other, switches engines.

### 2.13 Exactness by construction

- One reduction tree per site, independent of row count, staging path and
  the warp that consumed it. Weights from registers, shared memory, L2 or
  DRAM are the same bits in the same order.
- The fold's fp32 chain in canonical rank order, rank 0 first, as today.
- The expert chain in canonical expert order, shared expert last.
- The eager oracle runs the same device primitives one phase at a time, so
  step and oracle agree bitwise, which is how the graph and eager adapters
  agree today.
- Greedy stays byte identical with speculation on or off, across depths and
  across row counts, and the first run after boot equals the second.
- Every pipeline drains completely. The run-to-run divergence of 2026-10-17
  was a `cp.async` drain that read scale words before they landed, and the
  repeat-launch probe with a changing output sentinel is a standing test.

## 3. Scrutiny: what must be accounted for

Each item is a fact from the engine's record or the review, its consequence
for this design, and the design's answer.

### Platform

1. **One memory queue.** Streaming traffic inflates every access, the
   collective's handshake included, and prefetch at the full rate beside a
   fold stalls the fold. Answer: the governor of section 2.5, phase-aware,
   and the phase-1 sweep that measures the curve for TMA traffic, which may
   sit differently from plain loads.
2. **Flag visibility under load** is 1 to 8 µs across waiters. Answer: four
   hand-offs a layer, one dedicated polling warp per consumer group, data
   stamps instead of separate flags, TMA for the stream.
3. **NIC-written memory.** Plain loads cache stale bytes forever, 64-byte
   lines arrive torn while 16-byte units do not, and many pollers slow the
   NIC's writes. Answer: system-scope loads only, trailers per 16-byte unit,
   one sentinel lane per peer, all inherited from the write transport.
4. **The GPU cannot ring the NIC** and GPUDirect RDMA is closed. Answer: the
   host proxy keeps the posting role and the fold budget of section 2.8
   assumes it.
5. **Cooperative residency.** A cooperative launch fails rather than
   deadlocks when its blocks cannot co-schedule, and a loopback test world
   runs up to four ranks on one GPU. Answer: `engine.step_sms` and a test
   configuration of 12 SMs a rank in four green contexts, with the fabric
   and control warps counted in the block budget.
6. **Persisting L2 is at most 18 MB** and its context-wide reset needs an
   idle device. Answer: the lead's L2 share is sized at 18 MB, lines are
   released per range through `applypriority`, and the remaining 6 MB is
   budgeted as the working set.
7. **Staging is owned, not pooled.** Registers belong to threads and shared
   memory to a CTA, and a tile on one SM cannot be handed to a consumer on
   another. Answer: section 2.1's ownership rule, the lead counted as the
   persisting L2 plus one tile ahead per CTA, and the compiled resource
   probe of phase 1 that reports registers, spills, occupancy and usable
   lead before the capacity argument is accepted.
8. **The down kernel's parallelism** was register limited at two blocks per
   SM. Answer: TMA staging decouples loads in flight from registers, and the
   phase-1 format bench accepts a consumer only at the ceiling.
9. **Selection-bound bytes.** Sparse attention computes its selection from
   the current layer's query and only then addresses the KV, so those
   bytes are not in the lead on selecting layers. Answer: the dependency
   classes of section 2.4, the replay model carrying their release times,
   and the selecting layers' keys read in the open at the ceiling.
10. **The fold path is 13 to 16 µs each way in situ**, not the 5.5 µs wire
    figure of a CPU-posted probe. Answer: the fold table of section 2.8 is
    built from the in-situ decomposition, and its design column attacks the
    copy and the one-block gate pass, which the decomposition shows are the
    reducible terms.
11. **The skew is timing noise on identical work**, 11 to 15 µs median with
    tails to 106 µs, structural per the decomposition. Answer: the design
    hides it under the lead rather than removing it, states that as a
    hypothesis, and tests it with real compute on all ranks at real
    payloads with distributions.

### Bus and protocol

12. **A host thread is inside every collective** today and posts from a
    bounded polling loop that must never nap once a graph is recorded.
    Answer: the proxy's loop is unchanged in kind, with prebuilt work
    requests per fold slot and one flag per fold, and it stays pinned to a
    fast core.
13. **Done does not imply posted, and quiescence at adopt.** Both ordering
    rules were bugs once. Answer: the step-long window keeps the walk's
    posting mask and the credit discipline verbatim, and `bus_test`'s
    injected-delay scenarios run against the new walk before any family
    does.
14. **The wire trailer carries a 32-bit generation.** Each 512-byte chunk
    ends in a 16-byte trailer of a 64-bit hash, a 32-bit generation and a
    32-bit chunk index, the receive check reads the generation as 32 bits,
    and the host truncates its 64-bit generation to post it. Today an era
    that would cross the counter's top fails loudly and the server
    restarts. Answer: the step machine numbers generations within a step,
    at most 256 of them, and mixes a 64-bit step epoch into the chunk hash
    it computes on both sides, so a stale chunk from an earlier step with
    the same generation and index fails the hash. The wire layout is
    unchanged and the hash function is the step machine's own, since the
    machine is new code with its own kernels. Tests initialize the epoch
    just below 2 to the 32nd and just below 2 to the 64th. A soak cannot
    reach either boundary and does not stand in for this test.
15. **At most one outstanding collective per bus.** Answer: one per class,
    latency for the step and bulk for a coexisting chunk, with separate
    pools and lanes as today.
16. **Variable fold widths.** The staging region sized for one width was
    overwritten by a 5H fold in the halves experiment. Answer: the region is
    sized by the walk's widest fold and every fold site declares its width
    in the plan.
17. **Deadlines.** A peer that dies must fail the world legibly, not hang
    the kernel. Answer: every spin has a cycle deadline, the common exit
    path publishes a status, the engine watchdog poisons the control cell,
    and the inactivity watchdog resets per message as the bus does now.

### Exactness

18. **Kernel forms change at 4, 8, 32, 64 and 128 rows today**, and bf16
    projections above four rows go to cuBLASLt inside decode graphs, as do
    the DFlash2 drafter's small GEMMs. Answer: one form per site to the cap,
    and the step engine's consumers call no library kernel at all. The
    existing walks keep theirs untouched. The audit of phase 3 proves the
    absence in the step's trace rather than editing call sites.
19. **Transcripts will move at rounding level.** New chains for the sites
    that leave cuBLASLt and for any fused reduction are not the old chains.
    Answer: the cross-build judges, the first divergent token by bf16-ulp
    margin and the teacher-forced perplexity delta at 0.02 nat, decide
    old-versus-new, while step-versus-oracle, MTP-versus-plain and
    row-invariance stay bitwise.
20. **Fused reductions with arrival counters** overwrote each other when
    counters sat after variable-size partials and when a workspace was
    shared with cuBLASLt. Answer: counters in a fixed region at the
    workspace head, zeroed once at allocation and reset by the last block,
    private workspaces per site, and a unit test that shares one workspace
    across shapes and repeats launches bitwise.
21. **Per-row state and the commit.** The in-graph commit copies the
    accepted row's state back, and chain rows stash into rings that open
    pools still hold. Answer: section 2.10, replay forms and the ring copy,
    with the forced-rejection tests at every depth and at index-pool
    boundaries kept.
22. **The pick's three capacities are distinct.** `DevicePicker::kSlots`
    is six buffered pick instances per step, the verify's pick and the
    draft picks, not a row limit. Rows are `kPickMaxRows`, 96, and request
    slots are `kPickMaxRequests`, 16. Answer: the step engine's pick ABI
    names pick instances, request slots and verify rows as three numbers,
    with instances sized by the deepest draft chain pillar 2 will run.
23. **Selection boundaries need hand-computed expectations**, because device
    and oracle have been wrong together. Answer: every family's exactness
    fixture keeps at least one hand-computed selection and the audit that
    certifies a near tie from the device's own inputs.
24. **A cheaper pass can lose tokens.** The split-K record of 2026-10-19
    cut the pass and lowered the throughput on the fixed code prompt.
    Answer: section 1's three quantities and the promotion gate on decode
    tokens a second by class and concurrency with rows per pass reported,
    never on pass time or percent-of-floor alone.

### Engine and scheduler

25. **Implicit device barriers deadlock beside a spinning kernel.**
    `cudaEventCreate`, `cudaMalloc`, `cudaFree`, synchronous memsets and
    device syncs in any thread of the process. Today the hazard covers each
    collective kernel's duration. With the step kernel it covers the whole
    step. Answer: no allocating or synchronizing CUDA call from any thread
    while a step is in flight, enforced by a runtime hook active only under
    the step engine, with the NVMe tier and the resident-image staging on
    their own streams with preallocated buffers.
26. **Copy-engine nodes form dependency cycles** in one-process multi-rank
    worlds. Answer: the step kernel has no copy-engine work by construction,
    and the descriptor and verdict rings are pinned memory read and written
    by kernels, never by `cudaMemcpyAsync`.
27. **A queued kernel must get out of the way.** Fallback gathers, cache
    hops and time-sliced prefill chunks need the device between steps, and
    the graph engine drains before any of them. Answer: the descriptor
    states and the fixed between-step order of section 2.2, with the skip
    state as the drain's equivalent and a test that interleaves every
    between-step operation with a pending kernel.
28. **Programmatic dependent launch is opportunistic.** Answer: the protocol
    is correct under full serialization and the inter-step gap is measured,
    not assumed.
29. **Host nodes and the n-gram handshake.** Section 2.11, including the
    page residency the Engram decomposition demands.
30. **Grammar masks** arrive through a host-published sequence before the
    pick. Answer: the same handshake, with the host's budget being the
    whole step since masks depend only on the previous step's commits.
31. **The sampled fallback** runs on the host between windows. Answer: the
    fallback status of section 2.10 and the exact gather between steps.
    Keyed-Gumbel sampled verification, the fix class for the sampled
    divergence issue, would remove the fallback and belongs to pillar 2.
32. **Compact batches, physical slots and cache hops.** The descriptor
    carries the slot map, state stays in physical slots, and prefix-cache
    attach runs between steps as today.
33. **The journal and the digest.** The journal is unchanged. The kernel
    is launched early and bound late, so the record of a tick and the
    peers' apply complete inside the previous step while the next kernel
    waits at its descriptor read, and the op-stream digest is computed from
    the published verdict as it is from the replay's today.
34. **Pipelined verdicts.** The verdict is published at the pick while the
    draft chain runs, as the replay does today, and the scheduler's
    bookkeeping that spilled past the draft tail gets the step's full
    remainder to hide in.
35. **A block count is not an SM partition.** Answer: green contexts,
    measured on this box, section 2.12.
36. **Prefill advancement is synchronous.** Answer: the background-prefill
    pair on `SchedulerEngine` with completion ticks journaled by rank 0,
    section 2.12, and the mixed-workload gate before every template switch
    rather than after.
37. **Mixed prefill on GLM-5.3-Flash.** Section 2.12, with the before and
    after measurement of mixed-workload latency as part of that template's
    switch criteria.
38. **World 1 host-runs-ahead races.** Any pinned per-layer table the host
    refills is a hazard when nothing synchronizes. Answer: the kernel reads
    one descriptor per step and nothing per layer from the host.

### Build, test and operations

38a. **Lazy module loading deadlocks beside a resident kernel.** Measured
    2026-10-30 in the phase-1 flag probe: the first launch of any kernel the
    process has not launched before loads its module, and that load waits
    for the device while a kernel on another stream spins, so the new
    kernel never starts and the spinner never ends. With
    `CUDA_MODULE_LOADING=EAGER` the same kernel ran at once. Under the step
    engine every first-time launch in the serve process, a fallback kernel,
    a rarely taken prefill form, a diagnostic, would deadlock behind the
    resident step. Answer: the step engine's boot warms every kernel the
    process can launch, the launcher sets eager loading for the serve
    process, and the CUDA-call audit of item 25 counts a first-time launch
    as a forbidden call during a step.
38b. **Hand-offs through device memory are immune to the stream, and the
    system path is not.** Measured 2026-10-30 with the phase-1 flag probe,
    47 waiters polling a device flag with `ld.acquire.gpu` while a plain
    stream ran at 0, 37, 72, 106, 139, 167, 192, 210, 228 and 238 GB/s and
    beside TMA bulk-copy streams: the last waiter saw the flag within 0.48
    to 0.54 µs at p99 in every case. The phase-1 gate of 3 µs holds with
    room, and no governor is needed for device-memory hand-offs. The
    system path behaves differently. A CPU to GPU ping-pong through pinned
    memory, the fold's notice-and-post proxy, costs 1.4 µs idle, 1.5 at
    139 GB/s, 2.0 at 192, 2.5 at 227 and 3.0 at 238 GB/s at p50, 4.4 at
    p99, so the stream adds under 2 µs to the host hop at the ceiling. A
    system-scope atomic to pinned memory, by contrast, took about 25 µs to
    complete under a saturating stream and stalled the release store and
    the polling loads queued behind it. Answer: flags in device memory for
    every hand-off inside the GPU, plain release stores and acquire loads
    for the host hop, never a system-scope atomic on a critical path, and
    the governor's job on the fold shrinks to the fabric's own traffic,
    which this probe did not exercise.
38c. **A kernel resident on every SM blocks every other kernel.** With a
    streaming kernel of 48 blocks, one per SM, a second kernel's blocks were
    never dispatched at all even though every SM had room for them, while
    at 40 blocks they were. Answer: `engine.step_sms` leaves SMs free or
    the step runs in a green context, which section 2.12 already requires,
    and nothing else in the process may launch a full-device grid while a
    step is resident.

39. **Attribution.** One kernel per step makes nsys per-kernel attribution
    disappear. Answer: in-kernel phase timers, `globaltimer` stamps per
    phase per CTA into a device trace ring, drained between steps, with the
    bus timeline retained. This is phase 0, before any lever.
40. **Kernel attributes** registered for never-launched instantiations broke
    a later test. Answer: attributes registered by the launcher that uses
    the kernel, once, and the step kernel's opt-in shared memory requested
    within the driver's reported ceiling.
41. **Sanitizer time budgets.** A persistent kernel under racecheck runs
    orders of magnitude slower. Answer: the loopback budgets are lifted for
    the step tests as they are for instrumented kernels today, and memcheck
    runs on the orchestration while racecheck targets new consumer shapes.
42. **Compile shape.** One kernel carrying every consumer of a family is a
    register-allocation and build-time problem, and `__noinline__` alone
    does not prove the absence of spills. Answer: consumers are
    `__noinline__` device functions dispatched by phase type, instantiated
    per family and format set, never through function pointers, and the
    compiled resource probe of phase 1 reports the result.
43. **Configuration.** Every lever is an engine key with a README row, no
    environment switches: `engine.decode_engine` graph or step, default
    graph, `engine.step_sms`, `engine.stream_governor` and
    `engine.l2_staging_mib`. The weight layout is not a key: it follows the
    engine. The graph engine is never removed by this program. A template
    selects the step engine when its family's gates are green, and can
    select the graph engine back at any time.
44. **The resident image** is per engine. The step engine's image is the
    arena, keyed by engine, layout version and the payload note, so a stale
    image is rebuilt rather than misread and the two layouts never alias.
    The image cache holds one image per checkpoint and engine, which is
    disk, and a template commits to one engine so a rebuild happens once
    per switch, which is a load.
45. **Production.** No build or bench beside a served world, measurements on
    an idle fabric, production restored once at the end, as the method of
    record requires.

## 4. Generalization: the family contract

A family meets the stream engine through one contract. The runtime owns the
stream, the governor, the hand-offs, the fold machinery, the descriptor and
verdict protocol, the trace ring and the exactness harness. The family
provides:

1. **A plan generator**: ordered byte ranges per layer with per-range
   metadata and dependency class, and the late-bound expert ranges through
   a run list its router writes.
2. **Consumers per site** against the staging ABI, each one form from one
   row to the cap, each with its reduction chain documented.
3. **Latency phases** as device functions for latency warps: norms, mHC or
   GR sites, routers, selection, rope, appends, activation quantization,
   the EXL3 rotation.
4. **An attention consumer per attention type** over its own KV or state
   layout, streaming at long context, with its selecting layers declared.
5. **State updates in replay form** for every recurrence and convolution.
6. **A fold schedule**: sites per layer with widths, plus the pick.
7. **A draft form**: MTP chain rows, a DSpark block, a DFlash2 block or
   lookup rows, and the transaction's snapshot points.
8. **Host gathers**, if any, declared with the ids they need, the layer
   that consumes them and the pages that must be resident.
9. **An exactness fixture** with the eager oracle on the same primitives
   and at least one hand-computed selection boundary.

### The inventory

| Family | Layers | Attention and state | FFN | Folds a layer | Draft | Row cap | Formats on the decode path | Contract items that are new work |
|---|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash | 45: 34 KDA + 11 DSA, 4-stream mHC | KDA recurrence and conv state, MLA latent ring, indexer, select on every DSA layer | 288 experts top-8 plus shared, 3 dense | 2 | 1 MTP layer, chain | 32, batch 8 | fp8 block, NVFP4 experts, bf12 KDA, int8 KV optional | KDA replay form, mixed prefill measured against the graph walk, the 94-fold pass |
| GLM-5.3 full | 78, plain pre-norm, MLA and DSA every layer, 21 indexers | latent and index pools, shared selection on 57 layers | 256 experts top-8 plus 2048-wide shared | 2 | 1 MTP layer | 32 | int4 g64, NF4I8 with Hadamard, Mixed346 with a8 codes, int8 attention | a8 quantization on latency warps, codebook consumers at the ceiling, qkv_a sharding |
| GLM-4.7 | 92, GQA 96 by 8 with biases and head norms | paged KV | 160 experts top-8 plus shared | 2, 186 a pass | 1 MTP layer | 32 | NVFP4 | bf16 q/k/v GEMMs leave cuBLASLt |
| Qwen3.8-Flash-Next | 48: 36 GDN + 12 QSA, 4-stream GR, n-gram layer at 1 | GDN recurrence and conv, QSA rings and index, select on every QSA layer, PLE conv | 512 experts top-10 plus gated shared | 2, plus 2 at the n-gram layer | 1 MTP layer or DFlash2 | 96 | NVFP4, MXFP8 QAD, fp8 release, bf12 | host gather at step start, GDN replay, QSA radix select on latency warps |
| Qwen3.8-27B and Qwen3.5-122B | 64 or 48 GDN plus full GQA layers | GDN state, paged KV with fp8 modes | dense SwiGLU or 256 experts top-8 plus shared | 2 | MTP, DFlash v1 block 16, DFlash2 block 8 | 64 | fp8, NVFP4 MLP, W4A4, fp8 KV | drafter GEMMs leave cuBLASLt, block 16 drafts |
| DeepSeek-V4-Flash | 43, two-pass mHC, window 128 plus sink, compressed paths | window and compressor rings, ratio-4 layers select | 256 experts top-6 plus shared, hash routing on 3 layers | 2 | DSpark block, confidence head | 32 | MXFP4, fp8 | compressor on latency warps |
| DeepSeek-V4.1-Flash | 40, single-pass mHC, CSA2, 8 index-source layers, candidate layer 20, Engram at 1 and 14 | compressed KV, selection shared across 32 layers, compressor tails | 384 experts top-6 plus shared | 2, plus 1 per Engram layer | DSpark block, confidence head | 96 | MXFP4, fp8, EXL3 3.5 bpw | Engram host gather with resident pages, 5H folds, EXL3 rotation on latency warps, the first instrumented family |
| MiMo-V2.6-Flash | 48: 39 sliding plus sink, 9 global | paged KV | 256 experts top-8, no shared, layer 0 dense | 2 | 1 MTP layer | 32 | MXFP4, fp8 | fused decode attention to 64 rows becomes a consumer |
| Flint | 49 GQA with sink, 35 sliding, 14 full no RoPE | paged KV | 256 experts top-6 plus bf16 shared, 2 dense | 2 | none | 32 | fp8, NVFP4 | per-projection MoE form in the plan |
| Dory1.0-4B | 16 in, a 40-layer block looped up to 8 times, 16 out | KV planes per loop | dense | none, TP 1 | none | 32 | NVFP4 | the loop re-reads its block's weights each pass, which the floor meter counts |

Two facts from the inventory shape the shared runtime more than any family.
Every family has the same fold structure, two per layer with a few declared
extras, so one fold machine serves all of them. And the three
hyper-connection families, GLM-Flash's mHC, Qwen's GR and DeepSeek's mHC,
all put a fused norm-and-mix site at both fold boundaries, so the fold
consumer that computes the residual update in shared memory as rows land is
one design with three parameterizations.

### Port order

1. **Qwen3.8-Flash-Next on one Spark.** No fabric, so the stream engine,
   the governor, the hand-offs and the host gather are validated alone. The
   cell with the most headroom to its floor.
2. **DeepSeek-V4.1-Flash on four Sparks.** The fold rebuild, mHC, Engram's
   5H folds, DSpark and the richest instrument. The second checkpoint, EXL3,
   is the format bench's hardest case.
3. **GLM-5.3-Flash.** Production. KDA replay, DSA, the MTP chain and the
   mixed-prefill measurement.
4. **GLM-5.3 full.** 78 layers and 162 folds, the three codebook formats,
   the a8 path and the qkv_a sharding that the cheap fold reopens.
5. **Qwen3.8-27B.** Dense, the DFlash2 block, the drafter's GEMMs off
   cuBLASLt. Then Qwen3.5-122B on the same class.
6. **GLM-4.7, MiMo, DeepSeek-V4, Flint, Dory**, each a plan generator and a
   small set of consumers over the runtime the first five proved.

## 5. Delivery checklist

Every item has an exit gate. Phase 1 also has redesign triggers: a named
number that, if missed, changes a stated part of the design before anything
downstream is built. A phase closes when its gates hold on the fabric or in
the loopback worlds, and the campaign method of record governs every number
that reaches the benchmarks document.

### Phase 0: instruments

- [ ] The three quantities of section 1: logical bytes from the plan and
      the attention's page count, measured traffic from the DRAM counters
      under `ncu` on an idle fabric, bandwidth efficiency per step in the
      metrics and in `serve_campaign.py`'s record. Gate: the logical bytes
      for the DeepSeek-V4.1 C1 pass equal the hand count within 2 percent,
      and the graph engine's measured traffic is recorded per family as the
      baseline the step engine must not exceed.
- [ ] The device trace ring: `globaltimer` per phase per CTA, drained
      between steps, with a script that renders the two-term view. Gate: it
      reproduces the 2026-10-20 instrument's layer terms on the graph engine.
- [ ] The two-term instrument run on every family's template at C1 and C8,
      recorded as each family's DRAM-idle budget. Gate: a table per family
      in the measurements document.
- [ ] The stream replay model: a discrete-event model of one rank's step
      fed by real node traces, with every range's bytes and dependency
      class, the release times of routers and selections, the lead as a
      buffer of stated capacity that drains at the consumers' rate, the
      governed rate per phase class, the fold's critical path and a skew
      drawn from the measured distribution. Gates: run with today's
      structure it predicts the graph engine's pass within 5 percent on
      DeepSeek-V4.1, Flash-Next and full GLM. Run with the design's
      structure it reports the predicted pass and its sensitivity to lead
      capacity from 12 to 23 MB and to the governed rate from 100 to 240
      GB/s, and those numbers set the phase-1 triggers.
- [ ] The bus timeline kept and extended with the fold component split of
      section 2.8.

### Phase 1: the architecture experiment

One real layer of DeepSeek-V4.1 in one persistent kernel with the staging
scheme of section 2.1 as designed: the fused chain, the attention
projections from staged tiles, gate-up and down over a real run list, the
two folds inside the kernel, the hand-offs, compared with the graph
engine's layer at identical work on one GPU and on four. The probes below
feed it and are recorded in section 1.3; their sources sit in the scratch
area, untracked.

- [ ] The compiled resource probe: the one-layer kernel with the fp8, NVFP4
      and EXL3 consumers present and live staged weights, reporting
      registers, spills, occupancy and the usable lead. Gate: no spills at
      the chosen tile sizes, usable lead at least 20 MB. Trigger: below
      20 MB the register slab is dropped and the lead is L2 plus shared
      memory only, re-run through the replay model before phase 2.
- [x] Flag visibility, measured 2026-10-30: one writer, 47 waiting warps,
      beside plain streams from 37 to 238 GB/s and TMA streams to 80 GB/s.
      Result: p99 0.48 to 0.54 µs in every case, gate held. Owed: the same
      beside a TMA stream at the ceiling, which needs a pipelined TMA
      streamer the probe does not yet have.
- [x] Governor curve, host hop measured 2026-10-30: CPU to GPU ping-pong
      through pinned memory 1.4 µs idle, 3 to 9 µs at p50 beside plain
      vectorized loads at 240 to 250 GB/s whatever their depth in flight,
      but 8 µs beside TMA bulk copies at only 52 GB/s with 256 KB in
      flight, 15 µs at 1.5 MB and 33 µs at 3.8 MB. Bulk copies are hostile
      to the system path and plain loads are not, so the stream's form is
      the governor: plain vectorized loads reach the ceiling at 40 blocks
      and cost the fold's host hop a few microseconds, and TMA rings deep
      enough to reach the ceiling cost it tens. Owed: the same beside the
      fabric's own traffic on a real fold.
- [x] Persisting L2, measured 2026-10-30: a 16 MB window read once then
      a 120 MB stream read with evict-first loads leaves the window 100
      percent resident with no persisting set-aside at all, an 18 MB
      window 97 percent, while a stream of normal loads evicts 98 to 100
      percent of it. With the persisting set-aside armed a 16 MB window
      survives even normal loads, and at 18 MB about half of it does. Warm
      window reads run at 1.3 to 1.5 TB/s against 221 to 240 cold. Gate
      held. The lead's L2 share is protected by streaming the experts with
      evict-first loads, and the set-aside is a second line of defence for
      the coexistence mode where prefill's tiles stream with normal loads.
- [x] Stationary tiles, measured 2026-10-30: a 3.44 MB bf16 slice, 48
      CTAs, the same dot order in every variant. Compute after the input's
      release: 4.2 µs from registers and 4.2 µs from shared memory,
      820 GB/s of weight bytes, 10.6 µs from L2, 325 GB/s, 29 µs cold from
      DRAM in this probe's 8-byte-load form. All four bitwise identical.
      The bits gate held. The rate gate was mis-stated: a GEMV form
      consuming from L2 is issue-bound near 325 to 380 GB/s, the same range
      the in-server L2-fed GEMVs reach, and the 600 GB/s figure belongs to
      staged tiles, which exceed it. So the lead in L2 buys 1.4x over DRAM
      on consumption and staging into shared memory or registers buys 3.5x,
      and the replay model's consumer rates are confirmed.
- [ ] Format bench: every format's streaming consumer from staged tiles
      at rows 1, 4, 8, 16, 32, 64 and 96. Gate: at least 92 percent of the
      ceiling at every row count for every format, EXL3 included, with its
      rotation ahead of it. Trigger: a format under 92 percent keeps a
      two-form dispatch with a documented row boundary and its invariance
      test, and the exactness contract for that family notes it. Baseline
      measured 2026-10-30 (section 1.3): today's MXFP4 GEMV units reach
      about 80 percent of the distinct-byte ceiling at 5 rows and 60 to 75
      at 30 in any arrangement, issue-bound at 130 warp instructions a
      512-byte chunk. First candidate measured the same day (section 1.3):
      the tensor-core streaming consumer at 89 percent of the ceiling at 8
      and 16 rows and 87 at 32, flat across rows, exact to two bf16 ulps,
      the remaining points in the DRAM-pattern bridge. The tile-major
      arena layout then took the pair to 94 to 96 percent at 8 to 24 rows
      and 88 to 92 at 32, and the down projection to 92 at 8 rows and 65
      at 32, five fixes tried without effect. MXFP4 gate and up: gate held
      at 8 to 24 rows. fp8 block-scaled pair: 92 to 96 percent at 8 to
      32 rows, gate held. NVFP4 pair: 90 to 91, two points short. MXFP4
      down past 8 rows and the 32-row MXFP4 pair: open; the trigger's
      two-form dispatch is the fallback for down at many rows if phase 3
      finds no cheaper per-group cost.
- [ ] The in-kernel fold after representative compute on all ranks at H
      and 5H payloads, two and four nodes, distributions at p50, p90 and
      p99. Gate: critical path before skew 14 µs or less at the median.
      Trigger: above 18 µs the fold keeps the write transport's kernel form
      as a device function inside the step and the fused chain absorbs the
      difference, with the targets of section 1 re-derived.
- [x] Green contexts, measured 2026-10-30 with the architecture experiment's
      step kernel in one partition and a cuBLAS bf16 GEMM of 4096 by 5120 by
      17408, a 4K-token MLP chunk, in the other. On 40 plus 8 SMs the step
      ran 23 percent slower beside the chunk and the chunk 7.4x slower than
      on the full device. On 32 plus 16 the step ran 31 percent slower and
      the chunk 3.9x slower. With the lead marked persisting in an 18 MB
      set-aside the step collapsed to 2.7x its solo time, since the expert
      stream and the chunk were left to share the remaining 6 MB of L2.
      Over a chunk's duration the two modes deliver the same decode and
      prefill work, so coexistence buys smoothness and no throughput. Gate
      missed by a wide margin, trigger fired: coexistence is dropped from
      phase 5, step-boundary time slicing is the only mode, and the
      background-prefill pair on `SchedulerEngine` leaves the touch list.
      The four-rank loopback configuration still comes from green contexts,
      which the earlier probe showed partition the SMs disjointly.
- [x] Programmatic chaining, measured 2026-10-30: consecutive 40- and
      48-block step kernels on one stream, 300 µs of work each. The gap
      from one step's last block exit to the next step's first block start
      is 3.0 µs in plain stream order and 0.42 µs with a programmatic
      dependent launch, p90 0.5 µs. Gate held. The step boundary costs
      under a microsecond.
- [x] The experiment's own gate, 2026-10-30, in three steps. With
      synthetic consumers at real bytes: the in-situ stream rate is 98 to
      100 percent of the ceiling at every C1 shape and 95 to 97 at C8, the
      layer times beat the replay model's design residue by 5 to 7 points,
      and the layer-level gain against the graph engine's measured layers
      is 1.12x to 1.40x at C1 and 1.2 to 1.25x at C8, so section 1's
      targets were re-derived from the phase-0 table and this experiment.
      With today's MXFP4 slot consumer inside the kernel, bitwise the slot
      kernels' at 1 to 30 rows: 60 to 80 percent of the ceiling, the rate
      gate fired on the consumer form, and phase 3 builds the split of
      sections 2.3 and 2.7 rather than lifting the existing kernels' units.
      With the day's tensor-core consumers inside the kernel, MXFP4 for
      the routed experts and fp8 block-scaled for the shared, and the
      accumulation as a worker phase: the layer at 109 to 111 percent of
      the floor under the design's chain and an 8 MB lead at every shape
      with 40 workers, 104 to 109 with 47, 1.21x the graph engine's traced
      layer at C1 code, 1.29x at C1 prose and 1.25x to 1.39x at C8, against
      section 1's targets of 1.21x, 1.27x and 1.25x as re-derived from the
      phase-2 count (they were 1.25x, 1.38x and 1.25x on the phase-0
      count, which the prototype met at C8 and C1 prose and missed by 4
      points at C1 code); the coarse units' drain tail and the unit map
      are the named phase-2 items. The
      lead's cap is shown to interact with the consumers' L2 use (8 MB,
      not 20, on this family). Under today's unfused chain the same kernel
      is level with the graph engine: the fused chain is the whole gain.

### Phase 2: the stream engine

- [x] Plan format and the plan generator: every tensor's range, format,
      geometry, staging class, dependency class and consumer, with the bf12
      companions and released bf16 ranges as the ranges they are, and the
      routed experts' tile-major layout as a per-range layout descriptor
      the prefill consumers' address map reads. Gate: the plan's byte total
      for a step equals the floor meter's within 1 percent on every family.
      Landed 2026-10-30 for DeepSeek-V4.1 (`src/step/plan.hpp`,
      `src/step/dsv41/plan.cpp`, `docs/step_engine.md`): the format with
      its totals, checks and summary, a layout query on the shared loader
      (`ResidentLayerStream::layout_layer`, the restore's layout pass with
      no checkpoint open) so the generator reads every grant's offset and
      geometry from the loader itself, and the family's generator over the
      decode walk's phases. `step_plan_test` holds the gate: all 43 layers
      and the globals tile their spans against the loader's byte formulas
      on every rank of world 4 and at world 1, and the step bytes match
      the hand count from the checkpoint's headers to 0.1 percent at both
      C1 shapes, which re-derived the floors above. The bf12 head is a
      shape flag on the range's read; the tile-major descriptor exists and
      is unused until the arena below writes that layout. The other
      families' generators come with their ports in phase 6; the floor
      meter in `Meters` comes with the step engine adapter that has a
      plan to meter.
- [ ] The arena as a placement policy through the bump allocator's hook,
      the resident image keyed by engine and layout version. Gates: the
      memory plan reports today's totals within alignment padding, an image
      written and restored round-trips bitwise, and prefill on the arena
      produces today's prefill transcripts bitwise on every family, since
      the bytes and their internal layouts are unchanged. Placement landed
      2026-10-30: `LayerBump::placement` (grant i at a named offset, the
      mirror in the same layout, so the build, the H2D copy, the capture
      and the byte formula are untouched), `ResidentLayerStream::set_layout`
      with a `ResidentLayout` whose nonzero version is mixed into the image
      key (version 0 keys as before), and `dsv41_arena_placement` from the
      plan's order (phase, fixed before routed, expert, matrix, payload
      before scales). Gates held on the tiny fixture at world 2 in
      `dsv41_loader_test`: every tensor under the arena equals its
      checkpoint slice and its grant-order twin, the arena's image restores
      bitwise with no source reads, two layouts key to two files each
      restoring its own bytes, and the span is the byte formula's, so the
      memory plan is unchanged. The model takes the layout through
      `Dsv41Model::set_default_resident_layout`, the hook the step engine
      adapter will set from `engine.decode_engine`, and
      `dsv41_model_test` holds the transcript gate on the fixture: a model
      on the arena produces the grant-order model's forward, session
      prefills and decode step bitwise. Owed: the tile-major expert form as
      layout version 2, which lands with the phase-3 MXFP4 consumer as one
      unit (the loader's transform, a layout field on the fp4 matrix
      descriptor, the prefill readers' address map in `glm_moe.cu` and
      `fp4w_gemm.cu`, and the consumer that streams it), and the transcript
      gate on the fabric once a family port serves on the arena.
- [x] The streamer with the governor, in the measured form of section 1.3:
      lane-contiguous plain loads into register rings for the tile-major
      ranges with shuffled fragments, `cp.async` windows for activations,
      TMA only where a fixed range measures better with it; the staging
      ABI for register, shared-memory, L2 and direct destinations; the
      late-bound ranges of the router and selection classes; work
      distributed in 16-row tiles, since 128-row units left a drain tail
      of 20 to 50 µs a phase; `engine.l2_staging_mib` tuned per family
      against its consumers' L2 use, 8 MB on DeepSeek-V4.1. Gate: a plan
      replaying the recorded release times of a real DeepSeek-V4.1 step
      streams 6.9 GB at 95 percent of the ceiling with the fused chain's
      latency interleaved, matching the replay model within 5 percent.
      Landed 2026-10-30 as the stream itself (`src/step/device_plan.cpp`,
      `src/kernels/step/stream.cu`, `src/step/streamer.cpp`,
      `docs/step_engine.md`): the plan lowered to segments and pools under
      the arena, the chain block with the modeled latencies on the plan's
      phases, workers grabbing 64 KB units with evict-first loads four in
      flight a lane, the lead over the fixed units ahead within a lookahead
      and the L2 cap, the governor as a leaders bound, the synthetic router
      writing the run list. Gate held in `step_stream_test`: the first
      eight DeepSeek-V4.1 layers at five rows stream at 99.6 percent of the
      ceiling at the median layer with the design chain interleaved and an
      8 MB lead, 97.0 without the lead, at 47 workers; every layer between
      98 and 103 percent of its byte floor. Owed to the consumers of phase
      3: the staging ABI and the 16-row tiles (the synthetic consumer takes
      64 KB runs), `cp.async` activation windows and the tile-major
      fragment path, and the replay model comparison once the replay model
      is repo code rather than the scratch script.
- [ ] The speculative lead (hypothesis H1 of section 1.4): predicted
      expert units of the next layer in the lead's sequence behind the
      fixed units, streamed by idle workers within the L2 cap, superseded
      by the chain's run list. The experiment ran 2026-10-30: token
      identity predicts 56 percent of a layer's six experts inside a set
      of 12 (58 to 67 on tokens seen before, which are 66 to 95 percent
      of a text), the previous layer 25, the previous tokens 40, so the
      predictor is the token table and the gain is bounded by the L2 cap
      at about 4 percent of the floor. Parked behind the projection
      unit's rebuild. Gate when built: the layers with every weight
      computed at or above the synthetic 100 to 103 percent of their
      floor, and the step bitwise its own with the lead off.
- [x] Hand-off primitives with cycle deadlines and the common exit path,
      the status word, the engine watchdog's poison. Gate: a withheld flag
      ends the kernel inside the deadline with a legible status on every
      rank. Landed 2026-10-30 in the streamer: every spin carries the
      deadline, the exit path writes the status word with the phase, the
      workers leave on the end state or the poison, and `Streamer::poison`
      is the host's handle. `step_stream_test`: a 50 µs deadline on 800 µs
      phases ends the kernel inside a millisecond naming phase 1, a poison
      set 200 ms into a thousand-pass run ends it at 201 ms. Every rank
      runs the same kernel, so the multi-rank form of the gate is the
      fold's in phase 4.
- [x] Descriptor and verdict rings with the pending, execute and skip
      states, the device token feed, the host-published sequences for masks
      and gathers, in-step generations with the 64-bit epoch in the chunk
      hash. Gates: a thousand-step soak with no host work between steps,
      a test that interleaves every between-step operation with a pending
      kernel, and the epoch boundary tests just below 2 to the 32nd and 2
      to the 64th. Landed 2026-10-30 in the streamer (`src/step/protocol.hpp`,
      the kernel's step loop, `Streamer::run`): two pinned slots of
      descriptors and of verdicts, the chain polling the slot's state with
      a system-scope acquire and publishing the verdict's sequence with a
      system-scope release after its fields, skip answered by every block
      leaving with a skipped verdict, the token feed as two device rows
      the step reads by its parity and writes for the next, a gather phase
      (the plan marks the phases that read host rows) waiting on the
      host's published sequence under the deadline, the 64-bit sequence
      and epoch echoed by each verdict. `step_stream_test` holds the
      gates: a thousand steps over two layers from the feed alone, every
      verdict in order and the feed's fold matching the host's replica, the
      kernel seeing the next execute 1.8 µs after its verdict at the
      median, 3.6 at p99 and 20 at worst, with a 4.5 µs bind; host copies
      and memsets on another stream beside the pending kernel before three
      steps, then a skip answered by a skipped verdict; sequences crossing
      2 to the 32nd and wrapping at 2 to the 64th with the slots' parity
      kept; and a gather sequence published 3 ms late holding the Engram
      phase until it lands. The chunk hash itself is the fold's, phase 4;
      the masks' sequences come with the adapter's admissions in phase 5.
- [x] `engine.step_sms`, `engine.decode_engine`, `engine.stream_governor`
      and `engine.l2_staging_mib` parsed, journaled, in the settings record
      and the key table of `docs/configuration.md`. `engine.decode_engine`
      landed 2026-10-30 (graph | step; the config parser, the settings
      record's `deng` field with the pre-key default, the serving app's
      canonical settings line, the CLI override, the docs row, the parse
      and refusal tests, the journal round trip); a template that sets
      `step` is refused at boot with a message naming the missing family
      port, so nothing serves differently until phase 6. The other three
      landed the same day with the streamer that reads them: `step_sms`
      0 to 256 (0: every SM but the chain's), `l2_staging_mib` 0 to 24
      (8), `stream_governor` 0 to 256 (0: unbounded), each through the
      parser with its range, the settings record's `ssm`, `l2s` and `sgov`
      fields with pre-key defaults, the canonical line, the CLI, the docs
      rows and the tests. The step engine adapter hands them to the
      streamer in phase 5; until then they are carried and validated only.
- [ ] The CUDA-call audit: no allocating or synchronizing call from any
      serve thread while a step is in flight. Gate: a test hooks the
      runtime and fails on a forbidden call during a loopback step.

### Phase 3: consumers

- [ ] Streaming MMA forms per format, one form to the cap, bitwise
      row-invariant, repeat-launch stable with a changing sentinel.
      Prototyped 2026-10-30 (section 1.3, sources in the scratch area):
      MXFP4 pair at 94 to 96 percent of the ceiling and down at 92 at one
      token group, fp8 block-scaled pair at 92 to 96, NVFP4 pair at 90 to
      91, all exact to two bf16 ulps against fp32 references; still to
      build: EXL3, Mixed346 and NF4I8, packq, bf12 and the bf16 forms, the
      MXFP4 down past one token group, and the warp-specialized split that
      is the production form of all of them. The MXFP4 pair and down are
      repo code since 2026-10-30 (`src/kernels/step/mxfp4_expert.cuh` over
      `primitives.cuh`): a unit is a group of up to 32 slots of one expert
      and a 128-row block, eight warps on 16-row tiles of the tile-major
      layout, four 512-byte loads a matrix in flight a lane, the byte-
      permute decode, cp.async activation windows shared by the pair, the
      slot kernels' SwiGLU epilogue op for op, templated on the hidden and
      intermediate widths. `step_consumer_test` holds the gate against a
      double-precision reference over the same bytes at DeepSeek-V4.1's
      world-4 shape: of 27,648 outputs 27,611 are exact, 23 within two
      bf16 ulps and 14 inside the envelope the gate's and up's own fp32
      accumulation allows, none beyond; every down partial within 4e-6 of
      its terms' magnitude (worst 0.2 of the bound); bitwise across three
      launches with a changing sentinel and between a slot computed alone
      and the same slot in a group; as plain grids over 132 cold experts
      the pair streams at 235 GB/s and the down at 248. The fp8
      block-scaled form over the checkpoint's row-major layout followed
      (`fp8_dense.cuh`, the L2 bridge of a bulk prefetch a row a slab, the
      F32 scale per 32 by 32 block on the chunk's partial, 128 on GLM by
      template): 27,638 of 27,648 exact and the rest inside the envelope,
      bitwise across launches and group sizes, 229 GB/s for the pair and
      216 for the down as plain grids over 60 cold pairs.
- [x] The expert consumer over run lists with canonical accumulation and
      the shared expert last: a group is every slot of one expert, exact
      against the slot form (measured 2026-10-30), and the shared expert
      is a tensor-core consumer of its own fp8 form, read once for every
      token rather than once per group of four. In the streamer since
      2026-10-30: the lowering marks the MXFP4 pools with
      their consumer kind, matrices and 128-row units, the chain's router
      writes the run list and each expert's slot group, workers run the
      consumer units (down units four a grab, the group's staged
      activations reused), and the accumulation is a worker phase of
      512-output chunks folding the routed partials in slot order with
      the shared slot last. `step_stream_test` holds the gate inside the
      kernel: every gate and up output of a layer exact or inside the
      reference's envelope, every down partial inside the fp32 bound, and
      the accumulated rows bitwise the ordered sum of the partials. The
      shared expert is the phase's second pool, one group of every token's
      shared slot through the fp8 form, its five gate and up units handed
      out before the routed ones and its bytes in the lead's sequence; the
      first eight layers then run at 97 to 98 percent of their byte floor
      with every expert computed, on an idle device. The routed experts
      take the tile-major form where the plan records tiles and the
      bridged row-major form (`mxfp4_row.cuh`) where it records the
      checkpoint's layout, at 92 to 93 percent; both are checked inside
      the kernel against the reference.
- [ ] Dense and attention projections from stationary tiles, the thin TP
      shards as adjacent ranges; the fp8 block-scaled form above is the
      candidate for the fp8 projections, the bf12 form is unbuilt. The fp8
      projections are consumers in the streamer since 2026-10-30
      (`fp8_matrix_unit`, eight DeepSeek-V4.1 shapes, 128-row units over
      inner-width slabs with a reduce in slab order), exact on every output
      of a layer; the layer runs at 86 to 91 percent of its floor with
      every weight computed, the fixed phases' unit quantization being the
      residue (section 1.3). Their inputs are synthetic until the fused
      chain feeds them; the bf12 head form stays unbuilt.
- [x] The streaming unit's bytes in flight (hypothesis H2 of section
      1.4), measured 2026-10-30: one SM streams 52 GB/s with the plain
      four-deep ring and eight SMs saturate the memory, the unit's access
      pattern with the bridge streams 54 from one SM, so the unit's 16
      GB/s hot and 5 cold are its instruction stream (124 instructions a
      512-byte chunk, a dependent per-chunk scale load, a barrier every
      eight chunks). The staged variant is not the remedy; the rebuild
      below is.
- [ ] The projection unit's rebuild to 40 GB/s an SM, so six units
      saturate the memory and a phase of a few matrices streams at the
      ceiling: the scales of a warp's rows staged once a unit, the
      activation slice staged once a unit at one token group with no
      barrier in the chunk loop, the ring deepened where registers allow,
      the conversion's instruction count cut. Gates: exact on every
      output of a layer as today, the hot and cold per-SM rates in
      `step_consumer_test` at or above 40 and 30 GB/s, and the layers
      with every weight computed back at the synthetic 100 to 103
      percent of their floor in `step_stream_test`. Landed 2026-10-30
      evening as `fp8_matrix_staged_unit`: the staging, the direct
      fragment loads in place of the shuffles, the zero-C first mma and
      the conversion-unit decode, 25 GB/s an SM hot from 16, bitwise the
      windowed form, with the e4m3 decode defect fixed in every fp8 form
      (section 1.4). Owed: the rate target, the decode still half the
      instructions; since the fixed phases now stream at the ceiling in
      aggregate with the lead's staged bytes, the per-SM rate matters at
      their tails only.
- [x] The hand-out table and the lead's yield (2026-10-30 evening): the
      phase's table built once a worker in shared memory, lead units in
      16 KB pieces that stop at a published phase, the lead's idle budget
      during latency phases and drains, projection slabs of at least 640
      codes, and the trace's per-phase worker utilization, grab time and
      hand-out time. Hand-out 3.5 to 4.5 µs a grab to 1.0 to 1.6, workers
      busy 83, 78 and 96 percent of the qkv, attn_out and expert phases,
      the median layer 87 to 88 percent of its floor at any lead cap
      (beside production, at the uniform expert count; 83 measured clean
      at the measured count the next day).
- [x] The lead's accounting (2026-10-30 night): latency phases published
      to the workers, staged bytes credited as units consume them, a
      yielded unit counting what it read and yielding only to a stream
      publish, the lead within the next layer, reduce units after every
      pool's compute units, and the trace's lead-in-window, staged and
      wait columns. The lead now stages about 22 MB a layer in windows
      worth about 23 MB, so the idle windows are nearly spent; the
      accumulation's 21 µs is the design's own accumulate and fold
      latency (8 and 12), not its units, which take 2 µs each. The median
      layer with every weight computed stands at 89 percent of its floor,
      plus or minus two between runs.
- [x] A fault in the streamer's setup (2026-10-31): a plan whose consumer
      pools were fp8 only, with no routed MXFP4 pool, launched with no
      dynamic shared memory and no activation buffers, since the setup
      keyed everything on the routed pool, and the fp8 units faulted (Xid
      13). The setup now sizes the shared memory and the buffers from
      every consumer class present; the attribution case exercises it.
- [x] The measured routing overlap in the plan and the gate (2026-10-30
      night): `distinct_main` at the measured 19.8 and 10.1 gives floors
      of 29.2 and 21.5 ms with the bf12 head, the synthetic router routes
      through a window of 32 experts a layer (`route_pool`), which
      reproduces the overlap, and the slab reduce's completion counter is
      per layer (it was per pool across layers, so a later layer's reduce
      units did not wait on their own layer's compute units, masked by
      their late hand-out). The median layer at the measured overlap: 83
      percent of its floor, measured clean.
- [x] The lead's hand-out and two policy experiments (2026-10-31): the
      lead's two binary searches over the phase table ran in global
      memory, about fourteen dependent loads a unit, and held the lead to
      half the ceiling in a ten-microsecond window; a shared-memory copy
      of the phases' lead boundaries fixed it (ffn_in now fully covered,
      the lead 28 to 38 MB a layer). The trace gained the lead's bytes
      ahead at each phase start, the cursor's layer and lookahead blocks.
      Two policies measured and rejected: a minimum of six leaders during
      hot phases cost two points (their traffic competes with the fixed
      phases' cold units and the staged set outgrows L2), and the lead
      walking a phase's units even first then odd, so hot and cold units
      run side by side, measured plus one point within the run-to-run
      noise of plus or minus one and a half over three clean A/B pairs
      (kept as the `lead_interleave` switch, off). Measurement hygiene
      learned the hard way: the rate case runs in about two seconds and
      production's requests on the shared GPU contaminate it, so a run
      counts only when the serve log shows no decode in its window.
- [x] The measurement method for the stream's rates (2026-10-31): the
      rate case runs in about two seconds, and three confounds moved it
      by three to five points: production's requests on the shared GPU,
      the device's clock ramping from P8 on a now-idle GPU (`nvidia-smi
      -lgc 2418`, the application clock, holds it), and a fresh rig's
      first run reading low (the case runs the streamer once as a
      warm-up and reports the second run). Clean, the tile-major layers
      read 83.2 percent of their floor plus or minus 0.3 over three runs,
      the row-major 77.3. The lead cap sweep extended to 128 MB is flat
      from 12 MB on (83.9, 83.7, 83.8, 83.3, 83.3 at 12 to 64), the lead
      reading the same 29 MB a layer at every cap above 12: the lead is
      bounded by idle worker time, not by the cap. Earlier figures of 85
      to 89 were taken beside production and are superseded.
- [ ] The residue after the lead: at the measured expert count and the
      clean method, a layer of 147 MB takes about 730 µs against 615 at
      the ceiling. The expert phase takes 444 of 417, its workers 94
      percent busy. The fixed phases take 210: qkv is 26 MB with 12.8
      staged and reads 80 µs, its model (cold bytes at the ceiling, hot
      bytes at the L2 rate, hand-outs, the reduce tail), and attn_out 29
      MB with 4.3 staged reads 115. The windows take 78, and the lead
      streams 20 MB in them, 87 percent of the ceiling. So the memory
      idles about 117 µs a layer: 27 in the expert phase's structure, 25
      in the fixed phases' hot stretches (units reading staged bytes run
      from L2 and nothing streams meanwhile), 15 in hand-outs, 8 in
      tails, 13 in the lead's own rate, and about 30 not yet attributed.
      Two remedies for the hot stretches were measured clean and lost: a
      minimum of leaders during hot phases (minus 2) and the lead's units
      interleaved (minus 1.3), both removed, as were leaders inside the
      expert phase (minus 4 to 5: that phase is memory-saturated, so their
      9 MB of staging added one to one and bought back a quarter in qkv)
      and handing each class's units out last to first (within noise).
      The attribution case (`step_consumers_attribution`, 2026-10-31,
      clean, three runs) runs the same phases with each consumer class
      alone and the rest as byte units: every pool as bytes 96.6 to 97.9
      percent, the routed experts as consumers 86.1 to 90.8, the shared
      expert and projections 88.2 to 89.7, the projections alone 83.6 to
      89.9, every consumer 82.5 to 83.9. The expert consumers cost about
      ten points, of which their own phase explains five (450 µs against
      the byte units' 413 for the same bytes: hand-outs, the drain, the
      forms at 233 against 240) and the rest is indirect: their units
      keep every worker busy, so the lead stages 9.4 MB of the next qkv
      instead of the 15.3 the byte units' idle slack allowed, and qkv runs
      63 µs instead of 45. The projections cost about seven, their units'
      prologues, hand-outs and reduce tails. These costs are the stream's
      structure around the consumers, and the one mechanical lever left is
      to hide the hand-out behind the running unit (a grab ahead), worth
      about three points; the rest is the fused chain's. A probe of the
      pair's request pattern cleared it: a warp streaming a gate and an
      up tile 4 MB apart reads 238 GB/s at 48 blocks, the same as one
      stream of 512-byte or 1 KB runs (237 to 238.5), so the pair's 233
      as a plain grid is 98 percent of what a pure stream reads and the
      layout's second version would not buy the expert phase its gap.
      What is left there is the stream's own structure: the hand-out
      (about 9 µs a layer), the drain (8 to 12), the down grabs'
      prologues and the forms' 2 percent. Levers: larger down units,
      the drain's tail, and the fixed phases' tails. Gate: the layers
      with every weight computed at or above 95 percent of their floor
      with the design chain's latencies.
- [x] The ceiling per form (hypothesis H6), measured 2026-10-30: the
      tile-major forms miss exactly their bytes at L2 (415.2 MB of
      misses for a 414 MB sweep) at 2.9 to 3.9 sectors a request, the
      fp8 and row-major forms at 1.05 to 1.09 behind the bridge, the
      GB10 exposes no DRAM-side counters, so the floor keeps its 240
      GB/s ceiling and the spread between forms is request granularity.
- [ ] Fewer bytes a pass (hypothesis H4): bf12 companions for the
      remaining bf16 fixed ranges through the bf12 consumer, the ranges
      every rank reads whole sharded with their outputs gathered by the
      fold once phase 4 lands, the 2-bit screen on the head's reads once
      its bound is proven. Gates: every cut bitwise against the unsharded,
      unscreened step, and the plan's step bytes falling by the table's
      figures.
- [ ] The vocabulary head and the 2-bit screen as consumers, the pick's
      digit table as a fold payload, the pick ABI with instances, request
      slots and verify rows as three numbers.
- [ ] Attention consumers: paged GQA with bf16 and fp8 KV, MLA and DSA
      listed and selected, CSA2 compressed, sliding window with sink, QSA,
      Dory planes, each streaming at 32K and 128K contexts. Gate: at 128K
      context the consumer reads KV at 90 percent of the ceiling.
- [ ] Recurrences in replay form: KDA, GDN, convolutions, the PLE conv.
      Gate: forced rejection at every depth equals plain decode bitwise.
- [x] The router as a consumer pool, bitwise the production router's
      (2026-10-31): units of 16 expert rows, two a warp, in the production
      kernel's lane order over staged activations, the chain's select op
      for op, real slot weights in the accumulation.
      `step_router_matches_the_production_kernel` holds
      every score, pick and weight bitwise over eight layers. Cost at the
      measured expert count: about two points of the layer (the phase 28
      µs against the byte units' 11, the select and finish 7 and 5), the
      layer at 81 to 82 percent of its floor. Owed: the units are
      latency-bound at one block an SM (21 µs for two rows by five tokens
      where the arithmetic is a few), so the phase is 28 µs where the
      fused chain budgets 12 for update, mHC, router and slot order; the
      mma form would be ten times cheaper but moves the dot's rounding,
      so it waits on a near-tie policy for the step engine.
- [ ] Fused latency phases: fold consumer with residual update and mHC or
      GR mix, norm plus router plus slot order, selection, the a8
      quantizer, the EXL3 rotation.
      Progress 2026-10-31: the two mHC sites are consumer pools (units of
      three coefficients of one token, the completing unit running the
      token's finish: pre, post, the Sinkhorn comb, the collapse and the
      one-rounding norm into x), bitwise the production kernels over a
      layer's whole data flow (`step_mhc_sites_match_the_production_kernels`),
      and both stream updates ride on existing units, the output
      projection's reduce units and the accumulate, so no phase was added
      (a separate update phase cost 13 µs and was removed). Cost at the
      measured count, clean: 2.2 points of the layer (80.6 percent of the
      floor without the sites, 78.4 with), each site phase 27 to 35 µs
      against the chain's 8 and 12. The mechanism is the same as the
      router's: a unit reads 160 KB in 9 to 14 µs whether cold or staged,
      issue-bound at one block an SM (register-resident loads changed
      nothing), the finish about 10 on one block, the phase boundary about
      5. Owed here: the attention consumer, the folds and the select, and
      the structural remedy for every latency-bound unit, which is
      dependency-chained units inside the stream phases (as the reduce
      units are) so these fill the fixed phases' drains instead of
      standing as phases. Two defects fixed on the way: the test rig's
      overlaid arena copies (one per distinct layer layout now) and the
      host's wait on a faulted kernel. A 2048-output row chunk was tried
      and reverted (unit time scaled with the chunk, three points).
- [ ] No library kernel on the step engine's path, the existing walks
      untouched. Gate: no library kernel in a step's trace on any family.

### Phase 4: the fold

- [ ] The step walk as a fourth bus machine with its generation walk,
      posting mask, credits in ring order, quiescence at adopt and flights,
      the eager, graph and bulk machines untouched.
- [ ] Fabric warps: one sentinel lane per peer, staging by the consumer's
      epilogue, GPU-published hash with the epoch mixed in, trailers checked
      in the fold's own pass.
- [ ] The host proxy with prebuilt work requests per fold slot and one
      flag per fold.
- [ ] Widths H and 5H, the pick payload, world 1 as the identity.
- [ ] `bus_test` scenarios against the new walk: done before posted,
      quiescence, deadline and poison, injected delays, the epoch
      boundaries, two and four ranks. Gate: all green, and the fold's
      in-situ distribution inside a real step matching phase 1's.

### Phase 5: the step engine adapter

- [ ] `StepEngineAdapter` in `src/step/` beside the eager and graph
      adapters, selected by `engine.decode_engine`, implementing
      `SchedulerEngine` with the scheduler's existing paths unchanged.
- [ ] Verify schedule per slot, compact batches, slot map, cache hops,
      lookup rows, logprob candidates in the descriptor.
- [ ] The sampled fallback as a step status and the exact gather between
      steps.
- [ ] The journal unchanged and the digest from the published verdict.
      Gate: four ranks agree on the digest across a soak with admissions,
      retirements, cancellations and fallbacks.
- [x] Coexistence: dropped on the phase-1 measurement above. Step-boundary
      time slicing, the mechanism in use today, is the only mode, and the
      scheduler's existing paths carry it unchanged.

### Phase 6: family ports

For each family in the port order, in sequence:

- [ ] Plan generator and consumers wired, the eager oracle on the same
      primitives.
- [ ] Gates bitwise: step equals oracle, MTP equals plain, rows invariant
      from one to the cap, first run equals second, TP parity classes at
      worlds 1, 2 and 4 where the family serves them, group prefill
      invariance, forced rejections, compute-sanitizer clean.
- [ ] Judges at rounding level against the graph engine's binary: first
      divergent token by bf16-ulp margin, teacher-forced perplexity within
      0.02 nat on the three texts, quality set unchanged.
- [ ] The promotion gate: decode tokens a second by class at C1 and C8 at
      or above the targets of section 1, three repeats with the spread,
      rows per pass reported, measured traffic at or below the graph
      engine's, with the trace ring's layer terms recorded.
- [ ] The mixed-workload gate, before the template switches: decode gap
      and time to first token with a 32K cold prompt arriving under C4
      decode, step engine against graph engine on the same binary, in the
      coexistence mode where it exists and in time slicing. Gate: neither
      figure worse than the graph engine's. For GLM-5.3-Flash the graph
      engine's figure is its mixed walk's.
- [ ] The campaign row set on the template under the method of record,
      and the template switched to `engine.decode_engine: step` for that
      family, with the graph engine one key away.

### Phase 7: close

- [ ] [DESIGN.md](DESIGN.md) gains the step engine beside the graph engine
      in sections 5 through 7, 9 and 12, the configuration document gains
      the keys, the porting guide gains the family contract.
- [ ] Every served template selects the step engine. The graph engine, its
      tests and its documentation stay, available under the key for A/B
      and fallback. Its removal is a separate decision after this program.
- [ ] The campaign of record on every template and the README headline
      table refreshed with current numbers only.
- [ ] Production restored.

## 6. The twelve months

| Quarter | Phases | Exit |
|---|---|---|
| First | 0, 1 | the three quantities and the trace ring on the graph engine, the replay model calibrated, every probe's number recorded, the one-layer experiment at or above its gate on one and four GPUs with the targets of section 1 re-derived from it |
| Second | 2, 3, 4, port 1 | Qwen3.8-Flash-Next on one Spark through its promotion and mixed-workload gates |
| Third | 5, ports 2, 3, 4 | DeepSeek-V4.1, GLM-5.3-Flash and full GLM through their gates, the fold's in-situ distribution inside real steps recorded |
| Fourth | ports 5 and 6, 7 | every template on the step engine, the campaign of record, the graph engine kept behind the key |

The first quarter is the architecture experiment and the model that
predicts it. Nothing downstream is built until one real layer in one kernel
beats the graph engine's layer at identical work by the margin the model
predicted, on one GPU and on four. After that the order of construction
follows the stream: the expert phase, the attention arena and the
stationary weights, the fused chain and the in-kernel fold, the scheduler's
coexistence.
