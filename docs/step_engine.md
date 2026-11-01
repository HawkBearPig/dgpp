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
real. The chain builds the published group's class table into the control
block while the previous group runs (a routed pool's entry holds its units a
group, which the worker multiplies by the run list's length when it loads
the table, so no rebuild waits on the select), and a worker reads it in one
round trip when the phase word changes, so a grab costs the class counter's
load and the atomic, 1.4 to 1.9 µs; a worker building the table from the
pool tables took up to 17 µs a phase under a saturated memory, and
recomputing it a grab cost 3.5 to 4.5 µs. A lead unit
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
a fresh rig's first run reading low. Measured that way with the synthetic
router, the tile-major layers at the measured expert count read 83.2
percent of their floor plus or minus 0.3 over three runs and the row-major
77.3, and the lead cap sweep is flat from 12 MB to 128: the lead is bounded
by the workers' idle time, not by the cap. With the real router and the
hyper-connection sites computed as well, as gated classes of the stream
phases (2026-10-31 night), the same layers read 83.1 and 78.1 (the eight
tile-major layers 78 to 85).

`step_consumers_attribution` runs the same phases with each consumer class
alone and the rest as byte units, clean. Every pool as bytes 96.9 percent
of the floor, the routed experts as consumers 91.0, the shared expert and
projections 85.4, the projections alone 83.4, every consumer but the
hyper-connection sites 81.2, every consumer 83.3 (an earlier day's three
runs, before the sites: bytes 96.6 to 97.9, experts 86.1 to 90.8, shared
and projections 88.2 to 89.7, projections 83.6 to 89.9, all 82.5 to 83.9
with the synthetic router). The expert
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
running kernel from outside. The host's wait for a verdict ends on any
stream state but not-ready, so a faulted kernel surfaces as its error
rather than a spin (it spun once, 2026-10-31). The step engine adapter will read
`engine.step_sms` as the worker count, `engine.l2_staging_mib` as the lead's
cap and `engine.stream_governor` as the leaders bound. With synthetic
consumers the lead's cap measured best at 8 MB; with every consumer real,
whose units read what the lead staged, 12 to 16 MB.

`step_stream_test` is the gate, on the GPU with about 4 GB free for the
byte-unit rigs and about 10 GB for the consumer rigs: a consumer rig gives
every distinct layer layout its own arena copy, since DeepSeek-V4.1's index
sources, reusing layers and Engram layer differ and layers overlaid on one
copy read bytes filled for another layer's format (garbage the byte units
summed harmlessly and the sites turned into NaN rows and out-of-range
expert ids). The first eight main layers of DeepSeek-V4.1 at world 4, five
rows and a 64 KB unit, streamed from a device arena of their sizes with the
design chain's latencies interleaved, run at 99.6 percent of the ceiling at
the median layer with an 8 MB lead and 97.0 without it, over 47 workers. A 50 µs
deadline ends a run of 800 µs phases inside a millisecond with the phase in
its status, and a poison set 200 ms into a long run ends it then.

## The router

The router is a consumer pool (`kPoolRouter`): the bf16 matrix of expert
rows with its F32 bias, in units of 16 rows, one group of every token. A
unit stages the tokens' activations in shared memory and runs the
production router's dot (`kernels/glm_moe.cu`) op for op: one warp per
expert row, each lane its strided 16-byte vectors of the row, eight fmas a
vector in the file's order, then the xor tree. A warp carries two rows and
up to eight tokens through one read of each activation vector, and every
(row, token) chain is the production's, so the scores are bitwise its;
sqrt-softplus or sigmoid then the bias on the selection key, as the family
configures. When the router phase completes, the chain runs the production
select over the scores, one warp per token: top-k rounds of argmax with the
lower id on ties, the ids sorted ascending, the weights normalized with
per-element division and scaled. The run list, each expert's slots and the
slot weights follow, and the accumulation phase folds the routed partials
with those weights and the shared slot with one. (Holding the lead while the chain
routes, so its writes and fence do not wait on a saturated memory, halved
the select's time but cost the lead as much as it saved, and is not kept.)

`step_router_matches_the_production_kernel` runs the step and
`launch_moe_router` on the same activations and router weights over eight
layers: every score bitwise, every pick and every weight equal. The real
router costs the layer about two points against the synthetic one at the
measured expert count: its phase takes about 28 µs where the byte units
took 11 (24 units of about 21 µs, the units latency-bound at one block an
SM, while 23 idle workers lead through it), and the chain's select and
finish about 7 and 5 µs. The step kernel uses the full 255 registers a
thread.

## The hyper-connection sites

A layer's two hyper-connection sites (the attention site's `attn_in`, the
MoE site's `ffn_in`) are consumer pools (`kPoolMhc`) over their coefficient
matrix (bf16, 24 rows of four hidden widths), its base and scale, and the
sublayer's norm weight. A unit is three coefficients of one token: the
block holds the token's streams in registers, computes their norm, then the
three logits with the production kernel's fma chain and reduction order
(`kernels/step/mhc.cuh`), so the logits equal the production's bitwise. The
arrival that completes a token's 24 logits runs its finish on the same
block: pre and post from the logits, the comb seeded by softmax rows and
balanced by twenty Sinkhorn iterations on one warp, the collapse of the
streams under the previous site's pre, and the one-rounding RMSNorm into the
token's row of x, which the next phase's projections or experts read. The
residual streams live in two buffers by site parity and so do the pre
coefficients: the attention site reads streams[0] and pre[0] and writes
pre[1]; its stream update writes streams[1]; the MoE site reads those and
writes pre[0]; its update writes streams[0], the layer's output and the next
layer's input. A step seeds pre[0] with the one-hot, as the model's forward
does.

Neither update is a phase. The attention site's rides on the output
projection's reduce units, which hold the final y of their chunk and so the
update's h: each reduce unit mixes streams[0] under the site's post and comb
and adds post times h, in the production's one-rounding form, for its chunk
of (token, element) outputs. The MoE site's rides on the accumulate units
the same way, with the layer's y as h. A separate update phase cost a
protocol round trip of about 13 µs a layer and is gone. Nor, since the
evening of 2026-10-31, are the sites: see the end of this section.

`step_mhc_sites_match_the_production_kernels` runs one layer in the step and
replays it with `launch_mhc_compute_normed`, `launch_mhc_comb` and
`launch_mhc_stream_update_f32` on the same streams and weights: the
attention update, the MoE site's normed rows and the fused update are
bitwise the production's, and the output projection the attention site fed
sits within the fp8 reference's envelope of the production's row. The case
then runs eight layers for three steps, the pre chain and stream parity
across layers and steps.

As phases of their own the sites cost the layer 2.2 points (measured clean
at the measured expert count: 80.6 percent of the floor without them, 78.4
with), each site phase 27 to 35 µs against the chain's budget of 8 and 12:
a unit reads its 160 KB in 9 to 14 µs, the same whether the rows are cold
or staged in L2, because one block of eight warps an SM cannot hide the
unpack and fma chain (the standalone probe `mhc_unit_rate` gives 9 µs;
loads issued ten deep changed nothing), the completing unit's finish takes
about 10 (the Sinkhorn, the collapse and the norm on one block), and the
phase boundary about 5. (Wider accumulation and reduce chunks, 2048 outputs
for one wave of units, were tried the same day and reverted: a unit's time
grew with its chunk and the projection drains lengthened, three points.)

So the sites are no longer phases. A published phase hands out the units of
a group: its own and those of the phases merged behind it, as ordered
classes (a pool's compute units, that pool's reduce units, the fixed byte
units, an accumulation's chunks), each class with its own counter in the
control block and, where it depends on something, a gate: a reduce class
opens when its pool's compute units are complete, a site's class when the
chain has spun the group's fold, the router's class when the site has
finished every token. A worker takes the first open class with units left,
so a closed class is passed over rather than waited on, and the latency
segments run inside the stream phase's drain. The MoE site and the router
are merged into `attn_out`, whose output projection is handed out first and
whose fold the chain spins as soon as that projection's reduce units (which
carry the attention update) are complete; the chain selects as soon as the
router's units are complete, inside the group, so the expert phase's run
list is ready when the group ends. The attention site is merged into the
previous layer's `accumulate`, behind the layer's fold, which the chain
spins when the accumulate units are complete. The fold gate is a word per
group, not per layer: a layer has two groups with folds, and a gate shared
between them let the attention site read the streams before the accumulate
had written them (caught by the three-layer replay and the two-run
determinism check that `step_mhc_sites_match_the_production_kernels` now
carries). A latency phase is never merged, and a layer whose first phase
gathers host rows (an Engram layer) keeps its attention site as a phase.
The classes are in phase order with the gated ones in place: gated classes
first, so the merged units ran as soon as they opened, measured 3.6 points
worse, since under the saturated stream their latency-bound units ran
slower than in the drain and every grab paid their gate loads; merged
phases first deadlocked, the down units (waited per expert group inside the
unit) holding every worker ahead of the gate and up units. Completion
signals are release adds rather than a fence and a plain add. The trace
gives the first phase of a group its own units and the fold, the merged
phases the rest, and each class its window from first hand-out to last
completion.

Three more things the class trace taught the same night. The merged
phases' lead-only bytes (a site's coefficient rows, the router's rows) stay
on the lead's path when the group begins, where the first phase's own bytes
leave it: their classes open behind the fold and read them from L2 then.
Exempted with the first phase's they came from DRAM under the lead's full
stream, and the router's 24 units took 42 µs for 3.9 MB where the
standalone probe `router_unit_rate` gives 9 with the rows in L2 (in every
form of the unit: rows loaded one, five or ten ahead, tokens specialized or
not; five ahead is the kernel's). A lead unit yields when a gate opens as
well as when a phase is published, since a 64 KB lead unit at a 47-way
share of the memory is about 13 µs. And the chain's stamp of where the
lead's cursor stood, a binary search over the phase table at every group's
start, moved to the host: it cost the chain several microseconds a group.

Measured clean with all of that, the sites cost the layer nothing the
attribution can see (81.2 percent of the floor without them, 83.3 with,
inside the run-to-run band), the median layer 83.1 tile-major and 78.1
row-major in that run; later samples read 80.5 to 81.9 tile-major and 74.5
to 76.7 row-major (the earliest of them with the attention site racing the
accumulate, since the per-group fold gate had not landed), so the band of
one clean run is about 1.5 points on the median and 6 µs on a class window,
and no lever is judged on one sample or in isolation from the stages that
would have to adapt with it.

Two more pieces of the chain's own work came off its path the same night.
The lowering resolves each group's facts into its first phase's record (the
fold to spin, the phase that resolves the router, its router pool, the
layer's first routed pool after the group), so the chain reads one record at
a group's start instead of walking the phase and pool tables under a
saturated memory. And the select is two-level: each router unit emits its
top-k candidates a token over its sixteen rows, one warp a token with the
production rule (the larger biased score, the lower id on ties), and the
chain's select merges the units' candidates, 144 a token at top-k 6, in
registers with the same rule, so the picks and their weights are the
production select's over all 384 scores (a global top-k element is within
its unit's top-k; `step_router_matches_the_production_kernel` holds bitwise).
The chain's select fell from 13 to 15 µs under load to 4.4 to 6.0. The
router unit's class grew by about ten (32 to 45 µs) with the candidate scan
in it, and the kernel's stack from 32 to 72 bytes a thread (a pipelined
router form compiled beside the plain one behind a runtime switch had taken
it to 160, and is compiled no more); the unit's hot loop under that
pressure is the next thing to measure, standalone first. Two samples with
all of it: 81.0 and 80.8 percent of the floor tile-major, 74.4 and 75.2
row-major; the attribution 79.6 with the sites and 79.4 without.

Measured next (2026-11-01): standalone, the candidate scan costs the router
unit 1.4 µs a wave (13.4 against 12.0), and the forms (one, five or ten
vectors ahead, pipelined or not, tokens specialized or not) sit within
12.0 to 15.4, so the unit's class in the kernel, 32 to 45 µs, is three
times its standalone wave: the stream's contention and the hand-out's
stagger, not the unit's own work. Taking every unit entry out of line
(`__noinline__` on the consumers, the accumulation, the reduce, the select
and the finish), so each had its own register allocation rather than the
worker loop's at the 255 limit, lost 2.5 points on both forms over two
samples (78.6 and 77.4 tile-major, 72.2 and 72.6 row-major): the expert
units' hot loops pay the call frames and the ABI's reservation, and the
router class did not move. The units stay inlined. What the chain's tail
still needs is units that keep their standalone speed under the stream,
and the trace now says where. Each class's last hand-out is stamped
beside its first and its last completion (`last hand-out` in the class
lines): the router's 24 units are all handed out within 2 to 3 µs of the
gate opening, and the last of them completes 30 µs later, so the class is
the units, running at three times their standalone time under the stream
(13 µs a wave standalone, rows in L2); the MoE site's 40 units are handed
out within 5 µs and complete 20 later (14 for the dots, about 5 for the
finish, against 9 standalone); the attention site's units in the accumulate
group are handed out over 10 µs after the fold, the workers arriving from
lead units, and complete 17 after the last. The unit forms under the
stream are the target, not the arrival: the router's rows and the site's
coefficient rows staged through cp.async in stages as the fp8 projection
unit does, so a unit's loads are in flight together and its wait is one
latency rather than a round trip a batch; and the router's bf16 rows have a
lossless bf12 companion in the repo, a quarter fewer bytes.

The night's last measurements (2026-11-01) took the question down to the
unit's stages. A staging class per merged phase, gated on the group's fold
trigger, now brings a site's coefficient rows and the router's rows into L2
during the fold's idle window (the lead's early staging of them did not
survive the group); it is correct and cheap and moved nothing: the router's
units run 27 to 30 µs with their rows hot, and the same without any lead
at all (the lever matrix's `no lead` variant), and alone in the kernel with
every other pool bytes (`step_router_class_alone`: 22 to 26 µs, the 24
units handed out within 2 to 3 µs). The SM clock stays at 2398 MHz through
a rate run with no throttle reason set. The SASS with line info shows no
local-memory traffic in the router's hot loop (13.5k instructions); the
spills it does show, 496 and 320 local accesses in the expert and
projection units' staging loops, were the group struct passed by reference
and re-read from local memory a piece, and it goes by value now (the
kernel's stack 80 to 48 bytes a thread). The router unit stamps its stages
(`unit_trace`, printed by `router_unit_rate` and the lone-class case): the
standalone unit stages its activations in 1.5 µs, runs its four row batches
in 1.3 each and its epilogue and candidate scan in 4.2, 11.4 in all; inside
the kernel, alone with the lead, the same unit takes 2.0, then 2.5 to 3.0 a
batch, then 7: about 21. So the latency-bound unit's cost under the kernel
is its row batches' round trips at the L2's latency under the lead's fills,
plus an epilogue that doubles too; the arrival, the registers, the clock and
the rows' residency are not it. What would cut it: the lead paused for the
router's 20 µs (the matrix priced eight leaders at five points; zero for
twenty microseconds is a different trade), bf12 rows (a quarter fewer
bytes), and a lighter epilogue (the candidate scan and the per-warp stores
cost 4 to 7 µs where the dots cost 6 to 12).

`step_consumers_lever_matrix` measures the scheduling levers as a set, two
samples each, on the tile-major rig, with the median and the
attention-output group's site and router windows and tail. Its first full
run (2026-10-31 night, the kernel's gates and the lead's yield verified in
the source after two background patch scripts had failed silently): the
lead's yield on an opened gate is worth three to four points (82.7 and 83.0
with it, 78.3 and 79.7 without; the group's end 15 to 25 µs earlier), since
a 64 KB lead unit at a 47-way share of the memory is about 13 µs and the
merged classes' units waited that long for workers. Gated classes first
loses about three points even with the yield: the site and the router start
at the fold but the group ends no earlier, bounded by the stream they now
share workers and memory with, and the expert phase starts later. Bounding
the leaders to eight while a site's or the router's units run shortens the
group by about 15 µs and costs the layer about five points: the lead's
staging of the expert phase during that tail is worth more than the tail.
The 2048-output chunk and the pipelined router batches sit inside the band
alone and on top of the others. The kernel keeps phase order, the yield,
the 512-output chunk and the plain batches; the knobs stay on the streamer
for the next set. The class windows of `attn_out` (tile-major, layer 3, from the
group's publish): the output projection's compute units 2 to 64 µs, its
reduce units 62 to 74, the other projection 40 to 85 with its reduce to 89,
the fold, the MoE site 88 to 111, the router 109 to 132, the select and
finish to the group's end at 157, the expert phase published at 161. The
accumulate group: the accumulation 2 to 13, the fold to 25, the attention
site to 32. The chain's critical path after the attention-output fold is
now about 70 µs against the budget's 34: the site's class 24 (a hand-out
stagger, units of about 9 and the finish of about 10), the router's 23, the
select and finish about 25 under the saturated memory where they take 12
quiet. Those three are the next items.

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

