# The step engine

`engine.decode_engine` selects the decode engine of a deployment. `graph`
is the engine that serves today. `step` is the persistent-step engine under
development, built beside the existing engines so that nothing which serves
today changes while it is built. A template that sets `step` is refused at
boot until a family port exists. This page describes the parts that are in
the tree.

## The stream plan

The step engine streams a decode step's bytes from a plan rather than from
kernel launches. The plan (`src/step/plan.hpp`) is host data built from a
model's configuration alone. It lists every byte one rank's step reads as
ranges, and a family's generator (`src/step/<family>/plan.cpp`) produces it.

A range is one grant of the loader (a payload, a scale grid, a vector) or
one source outside the resident weights (KV rows, index keys, gathered table
rows). Each range carries:

- the format a consumer decodes (`bf16`, `f32`, `fp8` with an `f32` scale
  grid, `fp4` with `e8m0` scales, `exl3`, the KV row and index-key forms,
  `bf12`)
- the layout of the bytes inside it (`row_major` today, `tile16` for the
  tile-major expert form the step engine's consumers stream)
- the staging class the streamer lands it in (`register`, `shared`, `l2`,
  `direct`)
- the dependency class that says when its address is known (`fixed` at
  load, `token` at step start, `router` once the layer's run list exists,
  `selection` once the attention's selection has finished)
- the consumer and the phase that read it, and what a step reads of it:
  `read_bytes` of its `bytes`, `reads` times

Resident ranges are byte offsets into the layer's resident span in the
loader's grant order, which is the staging mirror's and the image's layout.
`plan_check` requires a layer's resident ranges to tile its span exactly,
in 256-byte grants with no overlap, so a generator cannot miss a tensor
silently. Router-bound ranges belong to one expert, and a layer states how
many of its experts a step reads at the plan's shape, so `step_bytes`
gives the step's expectation by dependency class and `consumer_step_bytes`
by site.

### The layout pass

A generator learns every tensor's offset and geometry from the loader
itself. `ResidentLayerStream::layout_layer` runs the family's builder over
a bump allocator based at an arbitrary address with no checkpoint open,
the same pass the resident image restore runs before it reads a layer's
bytes. The resident record it fills holds that address plus each grant's
offset, and the loader's byte formula is the span. No device memory and no
checkpoint are involved, so a plan is built in a fraction of a second from
`config.json`.

### DeepSeek-V4.1

`dsv41_step_plan` (`src/step/dsv41/plan.cpp`) walks the main layers, the
DSpark draft stages and the globals in the order `enqueue_layer` runs them.
A main layer's phases are `attn_in`, `qkv`, `select` on an index source,
`attend`, `attn_out`, `ffn_in`, `router`, `gate_up`, `down` and
`accumulate`. An Engram layer begins with `engram`, the first draft stage
with `draft_in`, the last draft stage ends with `draft_head`. The shape
names the verify rows, the draft block's rows, the lm head's reads a pass,
the head sharding, whether the head is read as its bf12 companion, and
either a measured distinct-expert count or the uniform expectation of the
picks.

`step_plan_test` is the gate. On the config of DeepSeek-V4.1-Flash at world
4 the plan tiles all 43 layers and the globals exactly against the loader's
byte formulas, on every rank and at world 1, and its step bytes match the
hand count taken from the checkpoint's own safetensors headers to within
0.1 percent at the two C1 shapes of the phase-0 record. The count is in the
resident-read form: the mHC coefficients as the bf16 the loader rounds them
to, the fp8 scale grids as the F32 the fp8 core reads, `main_proj` held
whole on every rank, the KV rows as the pool's 528-byte window rows and
288-byte entries, and the draft stages picking their own top-k.

| Shape, per rank, world 4 | Fixed | Routed experts | KV | Step bytes | Floor at 240 GB/s |
|---|---|---|---|---|---|
| five verify rows, a five-row draft block, the head twice as bf16 | 3.211 GB | 5.633 GB | 43 MB | 8.886 GB | 37.0 ms |
| the same with the bf12 head the production template serves | 3.046 GB | 5.633 GB | 43 MB | 8.721 GB | 36.3 ms |
| two verify rows, the same draft block, the bf16 head | 3.211 GB | 2.424 GB | 18 MB | 5.653 GB | 23.6 ms |
| the same with the bf12 head | 3.046 GB | 2.424 GB | 18 MB | 5.488 GB | 22.9 ms |
| five verify rows at the measured overlap, 19.8 experts a layer, the bf12 head | 3.046 GB | 3.922 GB | 43 MB | 7.011 GB | 29.2 ms |
| two verify rows at the measured overlap, 10.1 experts a layer, the bf12 head | 3.046 GB | 2.099 GB | 18 MB | 5.162 GB | 21.5 ms |

The routed rows of the first four shapes take the uniform expectation of
the picks (28.9 distinct experts a layer at five rows, 11.8 at two). Four
texts through the real checkpoint on 2026-10-30 showed consecutive rows
sharing experts, 19.8 distinct at five rows and 10.1 at two, about 3.3
more a row past the first, and the shape's `distinct_main` carries the
measured count; the last two rows are the floors the step engine is
measured against. The stream test's synthetic router routes through a
window of 32 experts a layer (`route_pool`), which reproduces the measured
overlap.

One rank's slice of one routed expert is 4,700,160 bytes: gate and up at
576 rows by 5120 columns and down at 5120 by 576, each with its e8m0
scales.

## The arena

The step engine's resident layout is the arena: each layer's tensors in
the order the step consumes them, so a phase's fixed tensors are one
contiguous range for the streamer and the routed experts of a phase form
one pool in expert order, gate and up in one pool and down in the next. A
run of consecutive experts in the router's canonical order is then one
contiguous range.

The arena is a placement, not a second loader. `LayerBump` takes an
optional `placement`, the device offset of every main grant in the
loader's grant order, and lands grant i at that offset instead of the next
free byte. The staging mirror takes the same layout, so the build writes,
the one H2D copy, the image capture and the byte formula are unchanged.
`ResidentLayerStream::set_layout` installs a `ResidentLayout`, a version
and a placement function per layer, before any layer loads. A nonzero
version is mixed into the resident image key, so an image of one layout
never restores into the other, and version 0 keys exactly as before the
arena existed. Placed layouts do not take bf12 side grants.

`dsv41_arena_placement` (`src/step/dsv41/arena.cpp`) derives the
placement from the plan: the layer's ranges sorted by phase, fixed before
routed inside a phase, routed by expert then matrix with the payload
before its scales, ties in grant order. The arena holds the same grants
and the same total, so the memory plan does not change.

`dsv41_loader_test` holds the arena's gates on the tiny fixture at world
2: every tensor under the arena matches its checkpoint slice and its
grant-order twin byte for byte, the image written from the arena restores
bitwise with no source reads, and the two layouts key to two image files,
each restoring its own bytes. The routed experts keep the checkpoint's
row-major form under this version of the arena. The tile-major expert
form of the measured consumers is the next layout version, together with
the address map the prefill path reads it through.

## The streamer

The streamer is one persistent kernel (`src/kernels/step/stream.cu`) in
the form the phase-1 probes measured. Block 0 is the chain: it walks a
layer's phases in order, publishes each stream phase through one release
store, waits for the phase's units to complete, and spins the modeled
latency of each latency phase and of the fold that follows a sub-layer.
Every other block is a worker: it grabs units of the published phase by
atomic counter and streams them with evict-first loads, four 16-byte loads
in flight a lane, folding the bytes into one word a warp so the loads
live. A worker with nothing to grab runs the lead: the next fixed unit of
the phases ahead, within a lookahead of layers and an L2 cap, read with
normal loads so the lines stay for the phase that consumes them. The
governor bounds how many workers lead at once.

Three details of the hand-out came from measurement with every consumer
real. A worker builds the published phase's hand-out table once in shared
memory (the unit count and, in hand-out order, each class's end and grab
size), so a grab costs the counter's load and the atomic; recomputing the
table's dependent loads under a saturated memory cost 3.5 to 4.5 µs a grab
and was most of the idle worker time in the expert phase. A lead unit
streams in 16 KB pieces and stops when the chain publishes a new phase, so
a worker is back at the hand-out within a piece rather than a unit at the
memory's share; before that, phase starts waited behind lead work. And the
lead's cap grows by an idle budget while the current phase is a latency
phase or drained, the modeled time until the next stream phase at the
ceiling's rate, since the memory would otherwise idle for it. The trace a
run reads back reports each phase's worker utilization, the mean time of a
grab, the mean hand-out time, the lead bytes streamed in the phase's
window, the bytes staged for the phase's own units and the time workers
spent on dependency waits.

The lead's accounting follows consumption. A latency phase is published
too, idle, so the workers see the phase the chain is in. The bytes staged
for a phase leave the lead's account as the phase's units consume them,
unit by unit, rather than at the phase's end, so the lead's room opens
while a phase runs and the staged set in L2 stays within the cap in
consumption order. A yielded lead unit counts only what it read, and it
yields only to a published stream phase, not to a latency publish. The
lead stays within the next layer (`lookahead_layers` 1), so its cursor is
at the bytes a latency window can stage. Reduce units of slabbed
projections are handed out after every pool's compute units, so the pool
they wait on has only its stragglers left. Measured together, the lead
stages about 29 MB a layer, the idle windows are spent, and what remains
of the layer is the expert phase's structure, the fixed phases' hot
stretches (units reading staged bytes run from L2 while nothing streams),
their hand-outs and tails. Staging into L2
with the evict-last priority rather than evict-normal changed nothing; a
probe showed evict-normal lines surviving a 150 MB evict-first stream at
92 to 95 percent. The lead's own hand-out searches the phase table in a
shared-memory copy of the phases' lead boundaries, since the two binary
searches a unit over global memory held the lead to half the ceiling in a
ten-microsecond window. Two policies were measured and left out: a minimum
of leaders during hot phases (their traffic competes with the fixed
phases' cold units) and the lead walking a phase's units even first then
odd (1.3 points worse, measured clean). The trace reports, per phase,
the lead's bytes ahead when the chain entered it, the layer its cursor
stood in and how many workers found the cursor past the lookahead.

The rates are measured with production down, the device's clock locked at
the application clock (`sudo nvidia-smi -lgc 2418`, released with `-rgc`
afterwards) and a warm-up run before the reported one. Each confound moved
the two-second rate case by three to five points: production's requests on
the shared GPU, the clock ramping from P8 on an otherwise idle device, and
a fresh rig's first run reading low. Measured that way, the tile-major
layers at the measured expert count read 83.2 percent of their floor plus
or minus 0.3 over three runs and the row-major 77.3, and the lead cap sweep
is flat from 12 MB to 128: the lead is bounded by the workers' idle time,
not by the cap.

`step_consumers_attribution` runs the same phases with each consumer class
alone and the rest as byte units, clean. Over three runs: every pool as
bytes 96.6 to 97.9 percent of the floor, the routed experts as consumers
86.1 to 90.8, the shared expert and projections 88.2 to 89.7, the
projections alone 83.6 to 89.9, every consumer 82.5 to 83.9. The expert
consumers' phase costs five points against the byte units' (hand-outs, the
drain, the forms at 233 GB/s against 240) and another five indirectly: its
units keep every worker busy, so the lead stages 9.4 MB of the next qkv
instead of 15.3. Workers leading inside that phase instead lost four to
five points, since the phase is memory-saturated and their staging added to
it one to one. The streamer sizes its shared memory and buffers from every
consumer class present; a plan with fp8 pools and no routed pool once
launched with neither and faulted.

The plan reaches the kernel through `lower_plan` (`src/step/device_plan.hpp`):
each phase's fixed ranges under the arena coalesce into a few contiguous
segments, its routed experts into one pool of groups at one stride, and
units are 64 KB runs over them. The chain's synthetic router picks the
step's experts for now, sorted into the layer's run list, and the pool
phases' unit counts follow it. `Streamer` (`src/step/streamer.hpp`) uploads
the tables, owns the control block, launches on `workers + 1` blocks after
checking they can be co-resident, and reads the trace back as phase and
layer times.

Every spin in the kernel has a deadline and leaves through one path that
writes a status word naming the phase, and the host's poison word ends a
running kernel from outside. The step engine adapter will read
`engine.step_sms` as the worker count, `engine.l2_staging_mib` as the lead's
cap and `engine.stream_governor` as the leaders bound. With synthetic
consumers the lead's cap measured best at 8 MB; with every consumer real,
whose units read what the lead staged, 12 to 16 MB.

`step_stream_test` is the gate, on the GPU with about 4 GB free. The first
eight main layers of DeepSeek-V4.1 at world 4, five rows and a 64 KB unit,
streamed from a device arena of their sizes with the design chain's
latencies interleaved, run at 99.6 percent of the ceiling at the median
layer with an 8 MB lead and 97.0 without it, over 47 workers. A 50 µs
deadline ends a run of 800 µs phases inside a millisecond with the phase in
its status, and a poison set 200 ms into a long run ends it then.

## The step protocol

The kernel runs steps, not launches. The host and the kernel share two
pinned rings of two slots (`src/step/protocol.hpp`). The host writes a
descriptor into the slot of the step's parity, the rows and the step's
64-bit sequence and epoch, and publishes its state: execute, or skip when
the host needs the device for work that cannot run beside a resident
kernel. The chain polls the state with a system-scope acquire, runs the
step, writes the verdict's fields and publishes its sequence with a
system-scope release; a skip makes every block leave with a skipped
verdict. A slot belongs to the host until it publishes execute and to the
kernel until the verdict is out.

Tokens stay on the device. The feed is two rows of tokens by step parity:
a step reads the row the previous step wrote and writes the row the next
step reads, so a step without admissions, retirements or masks needs no
host work before it. A phase that reads rows a host gather publishes, the
Engram layers' tables, waits on the host's published sequence under its
deadline before it starts.

`step_stream_test` holds the protocol's gates. A thousand steps over two
layers run from the feed alone with every verdict in order and the feed's
fold matching the host's replica, and the kernel sees the next execute
1.8 µs after its verdict at the median, 3.6 µs at p99, with a 4.5 µs bind.
Host copies and memsets on another stream run beside the pending kernel,
and a skip is answered by a skipped verdict. Sequences cross 2 to the 32nd
and wrap at 2 to the 64th with the slots' parity kept. A gather sequence
published 3 ms late holds the Engram phase until it lands.

## The MXFP4 expert consumer

The first consumer is the routed experts' (`src/kernels/step/mxfp4_expert.cuh`),
in the form phase 1 measured at 94 to 96 percent of the ceiling. A unit is
a group of up to 32 token slots of one expert and a 128-row block of the
expert's gate and up pair, or of its down matrix. Eight warps each own a
16-row tile of the tile-major layout and keep four lane-contiguous
512-byte loads a matrix in flight; a lane's sixteen bytes are two chunks'
codes of one row, decoded once by byte permute into bf16 tensor-core
fragments and scaled exactly by their e8m0 scale. The group's activations
are the other operand, staged by the block in shared memory with cp.async
windows that the gate and up share. Accumulation is fp32 on the tensor
core and the epilogue is the slot kernels' SwiGLU op for op, so the form
is bitwise its own across launches and group sizes and within two bf16
ulps of the slot form.

`step_consumer_test` is the gate. Against a double-precision reference
over the same tile-major bytes at DeepSeek-V4.1's world-4 shape, 27,611 of
27,648 gate and up outputs are exact, 23 within two bf16 ulps and 14 inside
the envelope the fp32 accumulation of the gate and up allows, none beyond;
every down partial is within 4e-6 of its terms' magnitude. Three launches
with a changing sentinel agree bitwise, and a slot computed in a group of
one equals the same slot computed in its group. As plain grids over 132
cold experts the pair streams at 235 GB/s and the down at 248.

The same form also exists over the checkpoint's row-major layout
(`src/kernels/step/mxfp4_row.cuh`), reading through the L2 bridge the fp8
consumer uses. It is exact and bitwise like the tile-major form and
streams the pair at 215 GB/s and the down at 223 as plain grids over cold
experts, against 233 and 247 for the tile-major form. The plan's layout
field picks the form when the plan is lowered, so the streamer serves the
layout the loader writes today and takes the tile-major form when the
arena's second layout version arrives.

Inside the streamer the plan's expert pools take the consumer kinds, the
chain's router writes each expert's slot group beside the run list, and
the workers run consumer units: a gate and up unit, or four down units of
one group at a grab sharing the group's staged activations. The gate and
up phase and the down phase run as one published phase whose units are
the pair's and then the down's, and a down unit of a group waits for that
group's gate and up units, so the pair's drain tail is filled with down
work. The accumulation is a worker phase of 512-output chunks that folds
the routed partials in slot order with the shared slot last. In
`step_stream_test` the consumers' outputs inside the kernel match the
reference as the plain grids do, the accumulated rows are bitwise the
ordered sum, and the first eight DeepSeek-V4.1 layers run at 97 to 98
percent of their byte floor at the median with the design chain
interleaved and every expert computed, on an idle device, 93 to 95 beside
a serving production world; with the routed experts in the checkpoint's
row-major layout the layers run at 92 to 93 percent.

## The fp8 consumer

The shared expert, and later the fp8 projections, keep the checkpoint's
row-major layout and are read by the fp8 block-scaled consumer
(`src/kernels/step/fp8_dense.cuh`). A unit is the same group and 128-row
block as the MXFP4 form, eight warps on 16-row tiles, but a lane's sixteen
bytes are half a 32-code chunk of one row, which alone would show the
memory 32-byte runs. Before a warp starts a 512-byte slab of its sixteen
rows it prefetches the next slab of every row into L2 with one bulk
prefetch a row, so the fragment loads hit L2. Each chunk's e4m3 codes
become bf16 exactly, the chunk's tensor-core partial is multiplied by the
F32 scale of its 32 by 32 block (128 by 128 on GLM, by template) and
accumulated, the slot kernels' structure, and the epilogue is the same
SwiGLU.

Inside the streamer the shared expert is the expert phase's second pool,
one group of every token's shared slot. Its gate and up units are the
longest of the phase and are handed out first, its bytes stay in the
lead's sequence as lead-only segments so the lead prefetches them, and its
partials take the shared slot of the accumulation. `step_consumer_test`
checks the form against the reference as plain grids, where it streams
the pair at 229 GB/s and the down at 216 over 60 cold pairs, and
`step_stream_test` checks the shared expert's outputs inside the kernel.

The same form with one matrix (`fp8_matrix_unit`) computes the layer's fp8
projections for every token, eight DeepSeek-V4.1 shapes compiled in, as
128-row units over slabs of at least 640 codes of the inner width whose
fp32 partials a reduce sums in slab order. Groups of at most eight slots,
the C1 shape, take the staged form (`fp8_matrix_staged_unit`): the warp's
scale row and the group's activation slice over the unit's slabs are
staged once, so the chunk loop has no block barrier and no dependent
load, and each lane loads the bytes of its own mma fragment (codes 8 tid
to 8 tid + 7 of rows gid and gid + 8) rather than taking them from a
shuffle, bitwise the windowed form's result. The e4m3 codes of every fp8
form decode through the conversion unit (two codes to f16x2, each to f32,
the pair to bf16x2), exact on every finite code and the decode
production's fp8 GEMV uses; the integer rebias it replaced was wrong on
the fourteen subnormal codes, which are about one code in ten thousand of
the checkpoint's fp8 tensors and which the synthetic test data had never
held. Inside the streamer every projection output of a layer is exact or
inside the envelope against the reference, and `step_consumer_test` holds
the staged form bitwise against the windowed form and against the
reference over data with subnormal codes.

The per-SM rate is the unit's instruction stream, not the memory's
latency: a plain streaming block reads 52 GB/s from one SM and eight
saturate the memory, while the hot projection unit went from 16 to 25
GB/s an SM through the staging, the direct fragment loads and the
conversion unit, with the e4m3 decode still about half its instructions.
With every weight of the layer computed the layers run at 87 to 88
percent of their byte floor at any lead cap from 4 to 20 MB; the fixed
phases stream at the ceiling in aggregate with the lead's staged bytes,
the expert phase at 92 percent with its workers busy 96 percent of the
time, and the residue is the layer's latency segments (about 70 µs a
layer with the accumulation's 20) that the lead can cover only as far as
the L2 cap lets it park bytes across the expert phase. A split form of the
projection unit (32 rows, the width in four parts across the warps, a
deterministic reduce) is kept as an option; it shortened the drain but
cost more per byte.

