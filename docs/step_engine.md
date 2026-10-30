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
cap and `engine.stream_governor` as the leaders bound.

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

