# DGPP Engine Design

This document describes the implemented serving architecture. The
shared runtime supports GLM-5.3-Flash, Qwen3.8-Flash-Next and GLM-4.7 with
resident loading, tensor parallelism, decode graphs, sampling, prefix
caching and MTP. World size and memory requirements depend on the model
and configuration.

The attention, weight-placement and state examples below describe
GLM-5.3-Flash unless another family is named. Qwen's GDN/QSA/GR/PLE
operators are documented in [its architecture study](docs/qwen38_flash_next_plan.md),
and GLM-4.7's GQA and NVFP4 layout in [its implementation notes](docs/glm47_plan.md).
Shared session and engine interfaces live in `src/engine/`.

Use [PLAN.md](PLAN.md) for implementation status and
[operations](docs/operations.md) for deployment. Dated measurements here
explain design choices; current benchmark tables and reproduction commands
are in [docs/benchmarks.md](docs/benchmarks.md). Investigation chronology
is kept in [benchmarks/results](benchmarks/results/). GLM-5.3-Flash image
execution is described in §11 and [image inputs](docs/vision.md).

## 1. Goals and decision rules

1. Match the checkpoint and reference implementation before optimizing it.
2. Keep the batch-size-one decode path graph-capturable and allocation-free.
3. Use all measured network and memory paths; do not freeze a performance
   target from nominal specifications or average-rank traffic.
4. Make cache and speculative state transactional. A rejected or evicted
   token must not change committed model state.
5. Expose an OpenAI-compatible text API only after end-to-end parity passes.

Performance targets are set from the completed runtime and a reproducible
workload, not from a third-party implementation. Router traces and a
full-model step profile are mandatory inputs at M4–M6; comparisons with other
engines are optional. The weight floor of a single-stream decode step is
~24.5 ms (5.855 GB/rank/token at the ~240 GB/s the L2-prefetched GEMVs
reach — §3, §7.6); the plain step measures 31.3 ms/token and the MTP step
22.45 ms/token effective (2026-09-03). These figures describe the FP8 checkpoint at TP=4.

Kernels may reassociate fp32 reductions to reduce latency when the
resulting error remains at rounding level. Cross-build regression checks
therefore use logit margins and teacher-forced perplexity
(`scripts/fabric_xcript.py`, `scripts/fabric_logprob.py`; §12).
Greedy MTP must still match plain decode exactly (§9).

## 2. Validated platform facts

### 2.1 Compute and memory

Each DGX Spark has one GB10 Grace Blackwell SoC and 128 GB of coherent LPDDR5x
shared by CPU and GPU. Nominal bandwidth is 273 GB/s; the weight-stream proxy
measures about 230 GB/s. `cudaMalloc` and `cudaHostAlloc` consume the same
physical pool, although their mappings and access behavior differ.

The engine uses explicit residency classes and prefetching. Transparent page
migration is not part of the hot-path design.

### 2.2 ConnectX-7 topology and budget

DGX Spark contains one ConnectX-7 NIC with two physical QSFP ports and two
independent PCIe Gen5 x4 connections to the SoC. One cabled QSFP port appears
as two usable Linux/RoCE lanes:

| lane | PCIe function | Ethernet | RoCE | lab subnet |
|---|---|---|---|---|
| 0 | `0000:01:00.0` | `enp1s0f0np0` | `rocep1s0f0` | fabric 0 |
| 1 | `0002:01:00.0` | `enP2p1s0f0np0` | `roceP2p1s0f0` | fabric 1 |

The `f1` functions belong to the second physical QSFP port and are down in the
current cluster. The two active `f0` lanes are not separate failure domains;
they share one physical 200 Gb/s QSFP link, but they remove the per-PCIe-x4
bottleneck when used together.

Measured node-to-node RC results on 2026-08-27:

- lane 0 alone: 107.64 Gb/s;
- lane 1 alone: 107.64 Gb/s;
- concurrent lanes: 97.98 + 98.05 = **196.03 Gb/s**;
- 2-byte send latency: 2.72 µs typical on lane 0 and 2.46 µs typical on lane 1;
- 9000-byte IP frames pass on both lanes.

CollectiveBus therefore stripes bulk traffic across both `f0` lanes and uses
approximately 24.5 GB/s as the measured aggregate planning ceiling. The bus
defines no per-lane failover (§6.1): the lanes share one physical port, so a
single-lane configuration is a deployment choice at about half bandwidth, not
a recovery mechanism.

Host DCB inspection shows priority-flow-control disabled on all priorities and
global Ethernet pause enabled in both directions. Switch PFC/ECN policy is not
visible from the nodes, but the paired tests reached the physical-link ceiling
without a congestion symptom. Switch inspection and counter deltas are
diagnostics if a future four-node run shows drops, retries, throughput collapse,
or latency spikes; they are not a correctness or deployment prerequisite.

RoCEv2 GID selection: each port's GID table sorts a link-local `fe80::` v2
entry ahead of the routable IPv4-mapped one, and switch forwarding of
link-local is pair-dependent — the M5 four-node mesh measured one-way
silent drops on specific directed pairs while the same QP pair ran cleanly
in reverse. The bus therefore prefers the first non-link-local v2 GID
(lane 0 → `c0a8:58xx`, lane 1 → `c0a8:59xx`), and the M0 tools' numbers
were taken on link-local GIDs — valid for those pairs, but the routable
GIDs are the deployment selection.

### 2.3 Direct NIC-to-GPU consumption

Conventional GPUDirect RDMA registration of `cudaMalloc` memory is unsupported
on DGX Spark. That does not require a bounce copy: the NIC can DMA into an
ibverbs-registered `cudaHostAlloc` slab, which the GPU reads directly from the
coherent memory pool.

The project test sends a 64-byte payload followed by a 64-byte sequence
doorbell as ordered RC SEND work requests. A concurrently running GPU kernel
polls the doorbell with a system-scope acquire and hashes the payload before
publishing a system-scope release acknowledgement. Both active lanes passed
10,000/10,000 changing payloads. This driver/firmware-specific hardware
contract is rerun during deployment and after upgrades.

The rerun is `nic_regress` (M5): one command drives the project test over
every directed node pair on both lanes — the mesh mode derives a
deterministic schedule from `--nodes`, every node processes its obligations
in order (senders connect with bounded retries, receivers hold
identity-checked connections that arrive early), and nodes[0] merges both
endpoints' views into one matrix and one exit code broadcast to all ranks.
Probes use the deployment GID selection (the first routable RoCEv2 GID, as
`VerbsDevice` picks) — not the M0 bench default, which measured the
link-local fe80:: GIDs whose directed-pair drops motivated the routable
selection in the first place. The regression must exercise the path the bus
uses. `nic_regress pair`/`serve` probe a single directed pair (targeted
debugging); `nic_regress selftest` runs the full mesh machinery as a
two-thread world on one device in CI.

Two concurrent 98.04 Gb/s incoming RDMA streams reduced an unrelated pinned
GPU read from 251.5 to 218.4 GB/s (13.2%). Capacity estimates that assume
sustained dual-lane ingress provisionally derate simultaneous GPU memory
bandwidth by 15%. This is a stress-case planning value, not a fixed inference
penalty; M5 replaces it with measurements of the actual collective schedule.

GPUDirect RDMA is closed as unavailable on this platform (re-tested
2026-09-02): `cuMemGetHandleForAddressRange(DMA_BUF_FD)` returns
`CUDA_ERROR_INVALID_VALUE` for `cudaMalloc` memory and `ibv_reg_mr` on a
device pointer fails with EFAULT (no peermem) on all four RoCE functions.
The bus's host-pinned staging, read by the GPU with system-scope loads
(§6.3), is the design the GB10 allows, not a fallback.

The memory controller is shared by the GPU, the CPU cores and the NIC, and it
is a single queue: 3 MB of GPU prefetch in flight added ~12 µs to every
access on the box, including the bus handshake and the engine's posts
(§7.6). Anything that adds memory traffic on a serving node is measured
against the step, not in isolation.

## 3. Checkpoint and decode traffic

The checked revision contains 62 safetensors shards, 76,108 tensors, and
328.33 GB (305.78 GiB) of weights. `tools/checkpoint_audit.py` reads the
checkpoint configuration and headers and emits the authoritative breakdown in
`docs/checkpoint_budget.md`.

Each node resolves the checkpoint from its local Hugging Face cache.
`--model unsloth/GLM-5.3-Flash-FP8` uses `loaders/hf_cache`, with cache
root precedence `HF_HUB_CACHE`, `HF_HOME/hub`, then
`~/.cache/huggingface/hub`. A `refs/main` entry selects the snapshot;
ambiguous or missing snapshots are errors. `--checkpoint-dir` also
accepts fixture or staged directories. Each rank loads its own slices
from the node's checkpoint.

Resident serving loads the rank's weights at startup and keeps them for
the process lifetime. GLM-5.3-Flash-FP8 requires TP=4 on 128 GB Sparks:
the main stack occupies about 81.77 GiB per rank, and MTP adds about
7.3 GiB. TP=1 and TP=2 exceed the node's memory. The hybrid NVFP4
checkpoint has a smaller footprint; use the memory plan for its exact
configuration. Other families have their own residency requirements.

Streaming mode loads one layer at a time and rereads the checkpoint on
each forward pass. It supports diagnostic runs that cannot hold the full
model. Resident mode uses the same layer builders, with each layer's
allocation retained instead of reused.

The resident loader follows three rules:

- Read source tensors with prefetch/copy/discard so checkpoint pages can
  be reclaimed during loading. Release source mappings when all layers
  have been constructed.
- Construct all layers before serving. Prefill does not trigger a lazy
  weight load.
- Cache finished layer bytes in a per-rank resident image. The key
  includes checkpoint headers, configuration, world, rank, head layout
  and format version. A layer is recorded only after its bytes have
  been written and synchronized, so an interrupted write leaves that
  layer unavailable for restoration. A cached digest accompanies the image.

Images restore through O_DIRECT reads. The recorded GLM-FP8 deployment
reached readiness in 15–25 s with a warm image, versus about 4.5 minutes
from checkpoint shards. `DGPP_RESIDENT_CACHE`,
`DGPP_RESIDENT_CACHE_DIR` and `DGPP_RESIDENT_CACHE_VERIFY` control the
cache; see [operations](docs/operations.md).

Before allocation, the serving application checks a memory plan covering
weights, caches, activations, scratch and engine buffers. The loader also
checks its resident footprint plus 8 GiB of headroom. Memory locking is
optional, and serving requires no GPU clock override.

Main text configuration:

- 45 layers: 34 KDA linear-attention and 11 DSA/MLA layers;
- hidden size 4096; 64 heads; KDA head dimension 128;
- three dense MLP layers, then 42 MoE layers;
- 288 routed experts, top-8 per token, plus one shared expert;
- one MTP layer;
- DSA `index_kpool=4`, `index_topk=2048`, and
  `index_kpool_always_select_tail=true`;
- RoPE dimension is zero for the MLA query/key path in this checkpoint.

The unique base-text weight set touched at batch size one is 22.415 GB/token.
Vision, input-embedding storage, and the inactive MTP layer are excluded.
Replicated DSA indexers, routers, mHC and norms bring aggregate physical
traffic to 23.420 GB per token across TP=4.

At TP=4, each rank reads 5.855 GB per token with intermediate-dimension
expert slicing. This gives a bandwidth floor of 25.46 ms at 230 GB/s or
24.4 ms at 240 GB/s.

Each rank reads top-k × 3 expert slices per MoE layer regardless of the
selected expert IDs, so routing does not change the FFN weight balance.
`tools/route_trace_traffic.py` can evaluate traffic under alternative
placements. Historical whole-expert placement measurements are kept in
the engineering record.

## 4. Quantized-weight contract

All 37,338 E4M3 matrices have an F32 `weight_scale_inv` tensor whose shape
exactly matches 128×128 blocks. The resident representation is the compressed
E4M3 payload plus its block scales.

“Load” means map, validate, shard, and place that compressed representation.
It never means materializing a persistent BF16 copy. Kernels either consume the
block-scaled representation natively or apply the inverse scales within a
fused conversion/GEMM path. M4 parity tests cover block edges, saturation,
NaN/Inf policy, and real checkpoint slices. An optimization that changes the
resident byte count must update the generated checkpoint budget.

NVFP4 routed experts (2026-09-08, `docs/nvfp4_plan.md`): the composed
checkpoint `dgpp/GLM-5.3-Flash-NVFP4-FP8` carries the main-stack routed
experts as the compressed-tensors triple `weight_packed` (U8 [N, K/2], two
e2m1 codes per byte, low nibble = even element), `weight_scale` (e4m3
[N, K/16], one per 16 elements along K) and `weight_global_scale` (F32 [1]),
with every other tensor — shared experts, dense MLPs, attention, the MTP
layer — the FP8 release's bytes under this same contract. The resident
representation is again the checkpoint's bytes untouched (`GlmFp4Matrix`:
payload, scales, and a pointer into the layer's gathered global scales);
the dequantized value is `e2m1(code) × (float(scale) / global)`, and
`e2m1(code) × float(scale)` is exact in bf16 (2 + 4 significant bits), so
a kernel's only inexact step is the one division by the global scale,
applied to a finished dot. The TP slice contract is 16 columns (one scale
block, eight packed bytes); row slices need no alignment. The format is
selected by `quantization_config.quant_method = "dgpp_mixed"`
(`GlmExpertFormat`); an FP8 checkpoint's table, bytes and paths are
unchanged. The kernels keep the bytes packed: the decode GEMV core
(`fp4_gemv.cuh`) and the prefill's grouped tensor-core kernel
(`moe_grouped_mma_fp4_kernel`) both decode e2m1 x scale in registers —
exact — and divide by the global scale once per finished dot, so the
routed experts' formula is applied with one rounding fewer than the FP8
path's; the decode slot path, the host grouped path and the sliced fold
are bitwise twins as under FP8, and the tile kernel's grouped form is
bitwise its dense form per segment.

## 5. Process, memory, and execution layout

There is one process and one CUDA context per node. Rank 0 additionally owns
request admission, scheduling decisions, and cache-eviction epochs. Every
rank executes the same request order and treats a coordinator epoch mismatch
as fatal to the current batch. As built (M6, §11), rank 0's ownership is
realized by the admission journal: rank 0 is the sole HTTP ingress, every
scheduler tick it runs is one journal record the peers apply before
running the identical tick, and every engine op is a bus collective, so a
record can never interleave a peer's in-flight tick.

### 5.1 Block-boundary invariant

Hidden and residual tensors are **replicated at every transformer-block
boundary**. Replicated norms, mHC, routers and indexers therefore receive
the same inputs on every rank.

| operation | input | local work/output | collective |
|---|---|---|---|
| norm + mHC | replicated hidden | replicated | none |
| q/k/v or KDA projections | replicated hidden | column/head shard | none |
| attention output projection | head shard | partial hidden | all-reduce sum |
| MoE router | replicated hidden | identical top-8 IDs/weights | checksum in debug only |
| routed experts | intermediate-dim slice of every expert | partial hidden sum | FFN all-reduce sum |
| shared/dense MLP | column/row TP | partial hidden | same FFN all-reduce |
| block exit | all-reduced hidden | replicated hidden/residual | none |
| lm head | replicated hidden | vocab-sharded logits | sampling-dependent merge |

Thus a normal layer has one hidden-vector all-reduce after attention and one
after FFN/MoE. Prefill may use reduce-scatter/all-gather internally for large
chunks only if it restores the same replicated boundary and passes parity.

KDA heads and recurrent state are head-sharded. DSA MLA latent and indexer
caches are replicated because every rank begins with the same hidden state and
must make the same sparse-token selection; there is no global score gather.

### 5.2 Weight placement

- routed experts: every expert on every rank, sliced on the intermediate
  dimension (gate/up rows, down columns) exactly like the shared expert —
  per-rank expert bytes are top-k x 3 slices whatever the routing, so no
  rank reads more expert weights at an FFN boundary. The per-rank chain
  runs in fp32: unrounded partial down dots, one
  fma per expert ascending, shared last — and rounds to bf16 once for
  the all-reduce;
- attention, shared-expert, and dense matrices: standard column/row TP;
- router, mHC, norms, and DSA indexer: replicated;
- embeddings and lm head: vocabulary-sharded;
- MTP: the same rules as its corresponding main-layer modules.

The shard generator records the owning ranks and scale geometry. Boot checks
hash all replicated tensors and reconcile per-rank byte totals before graph
capture.

**Quantized slice alignment.** GLM's E4M3 block-scaled matrices use a
128×128 scale grid. A view that retains this grid must start on a
128-element boundary along the sliced axis; otherwise its local scale
indices would address the wrong source blocks. Validators reject those
views. Masked tails at world 1 remain supported.

Qwen's expert slices use a finer local grid where needed: 64 elements at
TP=2 and 32 at TP=4 on the sliced axis. Each local scale repeats its
parent block's value, preserving the checkpoint's dequantized weights.
The shared loaders and kernels carry the scale geometry explicitly.

**Sharded loading.** At TP>1, the loader builds layers directly at each
rank's geometry. The independent full-load plus `GlmTpViews::bind` path
provides a bitwise reference for shard-parity tests. World 1 uses the same
builders with unsliced dimensions.

GLM DSA projections use compressed FP8 pairs when their slices align with
the scale grid. The `q_b_proj` and `o_proj` slices can start mid-block
at some worlds; those configurations read and dequantize the full tensor,
then slice its BF16 representation. The loader's bridge selection and
the TP views use the same alignment rule.

**Boot checks:** the loader folds every replicated tensor's raw source
bytes into a per-layer digest (order-independent, so load order cannot
change it) plus the globals; ranks exchange at startup and a mismatch
pinpoints the layer. Per-rank byte totals reconcile arithmetically: the
sharded-class bytes partition across ranks exactly once and the
replicated+bridge bytes re-read per rank, so
`sum_ranks(source_bytes) == world1_total + (world−1)·verbatim` — a
double-owned or missing row breaks the identity. Load boundaries
synchronize exactly the bump's reader streams (the model's stream plus
the loader's dequant stream), never the whole device: a device-wide wait
in a one-process multi-rank world blocks on peers' spinning collective
kernels — the first-collective stall measured under ~15 ms of thread skew.

### 5.3 Control plane: roster, epochs, health (M5)

Membership and coordination run over TCP — never on the CUDA critical path.
The wire protocol and its validated semantics live in `src/net/roster.hpp`
and the M5 results file; the contract in brief:

- The coordinator (rank 0) seals the roster at epoch 1 once every
  configured rank has announced, or fails startup with exact progress.
  Its own rank is an implicit member (it needs no loopback connection to
  itself). Late joins and rejoins are refused in v1: a replacement member
  would need a state-consistency story that does not exist yet.
- Epochs are owned by the coordinator and carried by every roster frame;
  they advance on membership change (eviction). A rank that observes a
  non-monotonic epoch treats the batch as fatal — the same rule as the
  coordinator epoch mismatch above.
- A rank is evicted on process death (connection EOF) or on a heartbeat
  deadline that applies **only after the roster is sealed** — before the
  seal there is no batch to protect and slow announces are normal. The
  evicted rank receives its final roster so it can tell eviction from
  coordinator loss.
- Rejections carry a reason string, not a bare close: cluster-start config
  errors must be legible in the rank's log. `join()` returns the first
  sealed roster frozen at receipt — a later eviction racing the caller's
  wakeup must not rewrite what the rank was sealed with.
- Coordinator loss (ack deadline) is fatal for the batch; there is no
  coordinator failover in v1.

## 6. CollectiveBus protocol

Each peer pair owns RC QPs on both active lanes — one QP per slot pool per
(peer, lane), so every receive queue is type-predictable: latency (8 KiB) and
bulk (256 KiB) traffic can never consume each other's differently-sized
receive buffers. Each (peer, lane) has one pinned, remotely-writable slab
registered as a single MR, holding send slots, completion cells, receive
slots, doorbell and ack cells, credit staging, and a control cell. Bulk
chunks are striped round-robin across the lanes; small messages use the
lower-latency lane.

Receive protocol for slot `S`:

1. the receiver posts payload and doorbell receive buffers and grants a
   credit. The credit grant is an unsignaled 64-byte RDMA WRITE of
   `{hash, slot, seq}` into the sender's registered completion cells — no
   CQE, no consumed receive WR; the sender polls the cell as a pinned flag,
   the same visibility contract as the GPU ack. The record carries the
   consumer's result, so the credit doubles as the message's completion
   notification;
2. the sender posts the payload SEND followed by the doorbell SEND on the
   pool's RC QP; only the doorbell needs a send completion;
3. the GPU observes the doorbell with a system-scope acquire, consumes the
   payload, and publishes its acknowledgement with a system-scope release;
4. the CPU polls the CQ for transport errors and recycles the slot only after
   both the GPU acknowledgement and receive completions are known.

Slots are 64-byte aligned, initialized before kernel launch, and never reused
without a new credit. **Ring discipline**: plain SENDs are consumed FIFO by
the peer's receive ring, so a slot is a ring position, not a free buffer —
the sender assigns slots round-robin per (lane, pool) and may not skip a
busy position; the receiver recycles in ring order; credits therefore
return in ring order. Persistent kernels use an **inactivity** watchdog whose
deadline resets after every message; a reserved stop sequence provides orderly
shutdown. The cross-node hash test and CUDA regression test pin this behavior.

Algorithm selection is size-based, and both machines are built (§6.3):

- replicated hidden vectors (the decode step's 90 boundary folds, the
  pick): the one-shot all-to-all all-reduce over the latency pool, folded
  by a per-collective kernel — recorded as graph nodes in the decode step
  (§6.2);
- prefill chunks: the striped reduce-scatter + allgather over the bulk pool
  (`allreduce_bulk`), both lanes;
- control messages: TCP, never on the CUDA critical path (the roster, §5.3,
  and the admission journal, §11).

### 6.1 Threading and concurrency model

The data plane is a single-threaded polling engine. One dedicated thread owns
every data-plane QP, CQ, credit counter, and slot on the node. The runtime
has exactly three thread roles:

| role | owns | blocks on |
|---|---|---|
| forward/engine | CUDA stream(s), graph launches | stream/graph sync |
| bus poller | all RC QPs/CQs, slots, credits, watchdogs | nothing (bounded poll) |
| control plane | roster TCP (§5.3) | socket deadlines |

The interfaces are the already-pinned ones: GPU↔bus through pinned-memory flag
sequences (the flag protocol; payload-then-doorbell ordering), bus↔control
through roster snapshots. Nothing else crosses threads, so the slot and
credit lifecycles carry no locks.

Why this shape, from measurements rather than taste:

- A dedicated bus thread is *mandatory* under graph capture: the forward
  thread is inside `cudaGraphLaunch` while inbound traffic still needs
  receive-WR recycling and the graph's own send payloads need a reactive
  CPU posting SENDs. The only question is the thread count.
- Bus CPU work is proportional to message count, not bytes — the NIC DMAs
  from registered slabs, so the CPU only posts WRs and polls CQs (~1 µs per
  message, size-independent). A decode step is ~90 collectives; full-rate
  prefill striping at the measured 24.5 GB/s aggregate is on the order of
  10⁵ posts/s. One core stays under half busy at the ceiling, on a
  20-core GB10.
- Logical operations span QPs: an all-reduce step touches a different peer
  each hop, a striped chunk touches both lanes. Splitting threads across
  peers or lanes forces cross-thread joins on slot/credit state — the
  failure mode is silent corruption, not a wrong roster.
- The loop never syscalls on the hot path (`ibv_post_send`/`ibv_poll_cq`
  are user-space MMIO). Completion-channel fds would cost an interrupt plus
  a syscall per event against a measured 2.4–2.7 µs one-way budget, and
  epoll-style demux exists for fd counts this system does not have (12 QPs
  at TP=4: 3 peers × 2 lanes × 2 pools). GPU-ready flags are memory, not
  fds, and cannot be epoll'd anyway.

Loop discipline (the anti-jitter contract, since one thread serves latency
and bulk traffic): each iteration is bounded — at most N posts and M
completion drains, in priority order (GPU-ready latency flags first, bulk
stripes after); per-QP watchdog timestamps are checked once per iteration;
the thread spins through a fixed grace period when idle before sleeping, so
back-to-back decode steps never reach the sleep path.

The bus uses one engine thread for latency and bulk traffic. Per-lane
sender threads are not implemented; any change to that ownership model
would need new contention measurements and protocol validation.

The bus maintains separate latency and bulk submission queues. Each
iteration handles GPU-ready latency work first and bounds bulk posts, so
bulk traffic cannot consume the latency slot pool or starve its receive
buffers. Concurrent latency/bulk operation is tested at the transport
level. Interleaving prefill and decode at the scheduler level is separate
work; the bus does not require serialized traffic classes.

Per-lane failover is deliberately not designed: the two lanes share one
physical port, so its failure takes both, and remapping traffic between them
buys nothing. A lane loss is a transport failure surfaced by the watchdogs.
Configuring the bus with a single lane is a deployment choice at about half
bandwidth, not a recovery mechanism.

### 6.2 GPU-side consumption: per-collective stream kernels

The consumer of received payloads is a short-lived CUDA kernel per collective
step, launched on the engine's stream: it polls the doorbells of the slots
that step depends on (system-scope acquire, bounded by a cycle deadline),
performs the reduction into local memory, writes the result, and publishes
the per-slot acknowledgements (system-scope release) before exiting.

Rejected alternatives, for the record:

- **CPU-mediated completion** (NIC → pinned slab → cudaMemcpy → reduce →
  copy back): two full-data copies per message and the CPU on the critical
  path of every collective. On GB10's unified pool the copies are pure waste
  — the zerocopy bench has the GPU streaming the NIC's landing zone
  directly. It remains the documented emergency fallback if a driver or
  firmware upgrade ever breaks unified visibility, which is exactly why the
  NIC→GPU visibility probe reruns per deployment (M5 deliverable 5).
- **One persistent dispatcher kernel per QP** (six for TP=4): multi-peer
  steps would need inter-kernel coordination that CUDA does not offer
  cheaply — a hand-built scheduler of atomics, the silent-corruption bug
  class; each kernel holds an SM indefinitely; and, decisively, a
  never-terminating kernel does not compose with CUDA graph capture, which
  the decode path requires (goal 2). Making a replayed graph safely wait on
  a persistent kernel requires a graph-side node polling memory that kernel
  writes — at which point the per-collective kernel already exists, plus a
  permanent kernel on top.

What the chosen shape buys: stream order is the dependency scheduler (no
hand-written sync between all-reduce steps and the GEMMs that consume them);
the per-kernel cycle deadline is the §6 inactivity watchdog, one per
collective; the decode step is a fixed launch sequence, hence
graph-recordable; and the doorbell/ack sequence is the one flag protocol
already pinned by `flag_protocol_test` and validated end-to-end by the
NIC→GPU visibility probe.

Cost: one kernel launch per collective step — a few µs uncaptured, under 1%
of the ~32 ms decode-step floor at ~90 collectives, and amortized inside a
replayed graph. A persistent doorbell-watcher hybrid remains a measured
optimization option for a future launch-bound path, not a starting bet.

**Graph capture (d3): each decode shape's fixed launch sequence records once
and replays per token.** The collective kernel becomes a graph node like
any other — the engine no longer launches it and instead *reacts*. The
interfaces that make replay-stable what capture bakes:

- *Per-generation cells.* One pinned 64B ctl per recorded collective node
  per graph variant (vs. the eager machine's single one-flight cell). The
  arm step selects a variant, then resets its cells and assigns the window's monotonic
  generations, `gen_seq` published release-last; the replayed kernel
  acquires it at start and derives its staging row `(g−1)%ring` and its
  exit match from it. Monotonicity means a previous replay's stale
  `done_seq` can never match, so replays of one node are distinct without
  re-recording.
- *The engine generation walk.* `window_count` publishes `window_first` and
  `window_variant`; the engine adopts that shape, posts
  each generation's peer pairs when the kernel's `ready_bits` land (lane 0,
  the cursor ring position — deterministic under the era's exclusivity:
  every latency post in the era is a graph generation, in order), and
  advances on `done_seq == gen`. Stream order serializes the recorded
  kernels, so at most one generation is un-done at a time — a single-flight
  state machine, not a queue. Arm waits for the previous window's walk
  (bounded); finish joins the walk and returns the verdict.
- *The era.* Up to 64 graph variants share one bus era, with disjoint cell
  slabs and write-once node metadata. Harness sends close at the first
  `record_begin`. Eager collectives reject while a variant records or a
  replay window is armed, but run between windows (prefill and its first
  pick need this). Graph reservations and eager pickups share one monotonic
  generation counter, keeping ring positions execution-ordered across shape
  switches.
- *Consumer width.* Graph all-reduce uses one block of 256 threads below
  32 KiB, 512 below 128 KiB, and 1,024 for larger payloads. Wider blocks expose
  more system-memory loads while keeping the placement proof and canonical
  rank-order arithmetic unchanged. The
  [four-node study](benchmarks/results/2026-09-16-dsv41-perf/README.md)
  records the payload latency sweep and serving validation.
- *Slot ownership without requests.* Graph posts carry no `BusRequest`,
  but the SendSlot/credit machinery is request-shaped — a per-window
  carrier request (unregistered, `is_collective`) gives the posts owners so
  credits recycle slots exactly like eager flights; completion is the
  walk's business, not the carrier's.

Two ordering rules prevent incomplete posts and stale slot state:

- **Completion does not imply posting is finished**: the
  kernel's fold waits on the *peer's* doorbell, not this side's own post,
  so a fast peer can stamp `done` while this engine's pair is still
  ring-deferred. The walk requires the posting mask complete before it
  advances — advancing first strands the peer's kernel on a doorbell that
  never comes (measured: 131/132 posts with the peer's last generation
  spinning 5 s on the missing one).
- **Quiescence at adopt**: every kernel of the previous window exited
  (walk complete implies every claim acked), so every latency door cell
  reads consumed. An unconsumed doorbell there freezes the ring-ordered
  recycle; the receive queue then drains over the next wrap and the
  wrap's last sender RNR-retries forever — a once-in-~30k-generation
  stall observed once, unreproduced in ~40k generations since, and
  watched by a standing adopt-time quiescence note plus the per-flight
  stall microscope (failure-only, in-tree).

Measured over two fabric nodes (TP=2, 12-collective steps, decode-scale
GEMM stand-ins between the nodes): eager p50 33.6 µs per collective
(submit→wait); replayed step p50 357 µs = **29.8 µs per collective
including the inter-node compute stand-in** — the per-collective host
submit/launch cost amortizes to near-zero, and the step pays one arm +
one finish for all twelve collectives. The wire floor (measured 2.4–2.7
µs one-way) is what remains.

**The mixed era (M6 Stage 4d).** The era as first shipped was exclusive:
once a session opened, eager collectives were rejected for the bus's
lifetime — unlivable for serving, where prefill's bulk folds and the
eager pick must run BETWEEN decode windows. The interface is one generation
counter: `graph_replay_arm` reserves its window's G generations from the
same `ctl_seq_counter` eager pickups increment, so execution order equals
generation order across eras and every staging-ring reuse fence holds
verbatim (those proofs only ever needed monotonic numbering). Eager
collectives are rejected only while a session RECORDS or a window is
ARMED (arm .. finish); every gen→cell mapping is window-relative
(`(gen − adopted_first) % G` — the first mixed-era run stalled on the
absolute form). The counter is 32-bit; an arm that would cross its top
fails the era loudly (restart before the generation counter wraps). Session
discipline: one forward thread for arm/finish/eager submissions.

**Model graph capture.** The model records its layer walk with a
`GlmGraphRecordReducer` at each boundary. The producing GEMV writes a
stable device buffer; the recorded collective kernel copies it into the
generation's staging row, hashes it and signals readiness for peer sends.
Host inputs use pinned, device-mapped staging read by upload kernels, or
device buffers written by the preceding replay.

GLM's plain step contains 90 boundary collectives. At depth 1, the draft
block and picks bring the speculative step to 94 collectives. Counts and
capture budgets differ for the other families. The bus permits up to
`kBusMaxGraphGens` recorded nodes per variant.

Every capture passes `check_decode_graph`. Memcpy and memset nodes can
form copy-engine dependency cycles when ranks share a process; uploads
and resets therefore use kernels. Kernel and empty nodes are allowed,
with a bounded host-node allowance declared by the model for operations
such as Qwen's mapped n-gram gather. Event-record nodes are rejected.
The failure analysis and reproducer are in
[the graph-stall investigation](docs/batched_mtp_graph_stall.md).

Eager and graph collective kernels retain separate implementations.
A prior shared-body refactor produced incorrect optimized code; changes
to either path need their protocol and numerical tests. Persistent
consumers in test harnesses need separate streams, and all consumers must
quiesce before pinned memory is freed because `cudaFreeHost` can
synchronize the device. Harness teardown therefore calls `quiesce()`
on both sides before `stop()`.

### 6.3 The collective: one-shot all-to-all all-reduce

The M5 deliverable-3 primitive is a one-shot all-to-all all-reduce over the
latency pool: every rank sends its full hidden vector to every peer
simultaneously — all stripes are postable the moment the collective is
issued, the W−1 wire transfers run in parallel — and one per-collective
kernel folds the W vectors locally (bf16 loads, fp32 accumulation in
canonical global-rank order: rank 0's vector first, always, so every
rank computes the identical chain and all destinations agree bitwise —
per-rank orderings could differ in the last ulp and diverge replicated
state; the host oracle mirrors the same chain and verifies exactly).

The recorded alternative, recursive doubling, was rejected for this class:
its round-j payload is round-(j−1)'s *reduced output*, a data dependency no
side can pre-post — every round would pay the full
GPU-fold → host-flag → post → wire → doorbell chain serially, exactly the
host round trip the per-collective design removes. At TP=4 one-shot sends
3×8 KiB where doubling would send 2×8 KiB; against a 196 Gb/s fabric that
"redundancy" is noise, and it buys a single dependency-free posting wave.
Recursive doubling (more precisely reduce-scatter + allgather) remains the
choice for large buffers where link bandwidth binds — the prefill striped
bulk class, where the term sizes are MiB, not KiB.

The three interfaces, in launch order:

- **Staging (GPU → engine).** The per-collective kernel is also the stager:
  it writes the device source vector into each peer's claimed send slot
  (pinned LPDDR, zero-copy), `__threadfence_system`, then publishes a
  per-peer ready bit in a pinned control cell (system-scope release). The
  engine's collective pass acquire-loads the bits and posts payload+doorbell
  SENDs — the same handoff the forward integration will use when the
  producing GEMM writes slots directly. The wire protocol is unchanged:
  an all-reduce stripe is an ordinary latency-class message.
- **Inbound (NIC → GPU).** The kernel scans each peer's latency doorbell
  cells across all lanes; `doorbell.seq != ack.seq` marks an unconsumed
  message (the ring/credit discipline guarantees at most one per peer per
  outstanding collective). It claims by publishing the standard ack —
  hash = fold of the received payload, publish-last seq release — so the
  engine's recycle and credit passes run byte-identical to harness
  traffic. The engine's doorbell-CQE `expect_seq` validation catches any
  contract violation loudly.
- **Completion (GPU → engine → waiter).** The kernel publishes
  `{ctl_seq, status}` into the control cell (release) after the reduce;
  the engine's pass sees it, completes the request (the single completion
  authority, same cv semantics as `send()`), and the waiter runs one
  `cudaStreamSynchronize` after waking so cross-stream consumers of the
  destination are ordered. Credits still flow receiver → sender for slot
  recycling, but they are off the collective's critical path entirely —
  completion no longer waits for the credit round trip (the ~17 µs
  harness p50's second half).

v1 restrictions, all deliberate: at most one outstanding collective per
bus (decode is dependency-serialized layer to layer); `launch_consumers=
false` (persistent harness consumers would race the per-collective kernel
for doorbell claims); no other latency traffic while a collective is in
flight — enforced: once a bus enters collective mode, `send()` is closed
(a harness message would be claimed and folded into a peer's reduce: the
silent-corruption class, rejected loudly instead). Failure paths: the
kernel carries its own cycle deadline and exits through one common
post-scan barrier (the persistent-kernel deadlock lesson), the engine
watchdog fails the request and poisons the control cell so the kernel
exits promptly, and a lane failure completes the request with the legible
lane error. The slot that a failed request staged is freed by the normal
credit/owning-reference machinery — a late credit after the waiter reaped
never corrupts ring state.

Two ordering rules the TP forward integration taught (both were bugs):

- **Completion clears before it wakes.** The flight state (claims,
  posted stripes, the single-outstanding gate) must be fully reset
  BEFORE the request is completed — completing wakes the waiter, and a
  waiter that submits its next generation before the engine's cleanup
  races the single-outstanding check. Symmetrically, any path that
  reaps a collective out from under the engine (watchdog expiry, lane
  failure) must clear the held flight, or the bus rejects every later
  collective forever.
- **Allocation-phase device syncs never overlap a spinning collective.**
  `cudaMallocManaged`/`cudaFree` and device-wide syncs are whole-device
  barriers; a rank still constructing its model while a peer's first
  collective kernel spins on doorbells deadlocks the process (the
  kernel waits for the constructing rank's post; the constructing
  rank's allocation waits for the spinning kernel). The model
  preconstructs every layer object at construction (no allocation inside
  the forward), and TP runners barrier cluster-wide after construction
  before any forward begins. Post-startup the shape is safe — engines
  post from free threads, so staggered per-layer loads and in-flight
  collectives coexist by construction.

Three more ordering rules, all measured into existence by the bulk
machine's bring-up (§6.3's striped reduce-scatter + allgather):

- **Claim only your segment's ring window.** The bulk doorbell rings are
  flight- and phase-agnostic FIFOs and the shard geometry guarantees
  ranks do NOT progress in lockstep — a rank whose shard ends in segment
  0 enters allgather while its peer still folds reduce-scatter. The
  receiver's kernel derives each arrival's ring-lifetime index j from
  the doorbell itself and claims exactly [base, base + this lane's
  stripe share); everything else stays unconsumed for the kernel whose
  window contains it. Eager claiming acks a future segment's doorbell
  away from its owner and both phases deadlock on the loss.
- **Done does not imply posted — and neither does TX retirement.** A
  zero-arrival segment kernel (empty receive window) stamps done while
  its own outbound stripes are still being posted; the advance gate
  requires the posting drained AND the doorbell CEs retired before the
  next segment launches, or the launch's staged-counter reset strands
  unposted stripes with the release condition inverted forever.
- **The ack certifies consumption, not arrival.** The RS fold reads
  payload slots in the consume pass after the claims; acking at claim
  time returns the sender's credit early, and a sender a segment ahead
  wraps the 8-deep ring inside one claim-to-exit gap, DMAing new stripes
  over a fold input. Acking after the fold makes the credit gate close
  the race by construction — and the scan must skip the kernel's own
  claims meanwhile, or a re-presented claim steals the shared CAS slot
  from a fresh doorbell in the same warp, forever.

- **Done does not imply posted — the one-shot form** (2026-09-05). The
  one-shot kernel stages, releases its ready bits, claims, folds and
  stamps done; when every peer has already posted, all of that takes
  ~20 µs and fits between the engine's two reads of the control cell.
  The pass read the ready bits first and the done stamp second, so a
  kernel that finished in between had its flight completed with this
  rank's own stripe never posted — the peers' kernels waited on it
  forever and this rank's next generation parked behind their
  generation gate (glm_tp_test's two-rank loopback, once in fifteen
  runs: "launched" then "done after 18.5 µs" with no "ready" between,
  the packet counts one stripe short). The pass now reads done first
  (a pass that sees the stamp then sees the bits, by release/acquire
  and kernel program order) and refuses to finish a reduced flight
  before every peer's stripe is out; `bus_test`'s
  `scenario_allreduce_done_before_posted` injects a 300 µs sleep between
  the two reads on rank 0 (`BusOptions::debug_pass_delay_us`) and fails
  within seconds on the old ordering.

The bulk kernel is a cooperative grid (2026-09-05). Sixteen blocks,
co-resident by `cudaLaunchCooperativeKernel`'s contract (the launch fails
rather than deadlocks when it cannot co-schedule; two such grids run
concurrently, which the loopback test worlds need), with grid barriers
between staging | claims | consume rounds | acks. Everything that scales
with bytes runs over the whole grid in 16 KB tiles of 16-byte vectors —
the outbound staging, the RS fold, the AG landing — while the doorbell
claim loop (the shared-CAS discipline) and the deferred ack flush stay in
block 0; the claim records live in a device-resident `BusBulkScratch`.
The placement proof (a payload folds to its door's hash before anything
consumes it) moved from the claim into the consume pass: every tile
hashes the words it reads, the per-(peer, stripe) totals combine by
atomic XOR, block 0 proves each stripe between rounds and a stripe read
before its placement was fully visible is re-folded next round (the fold
and the copy are pure functions of their inputs); no ack and no done
stamp until every consumed stripe has proved (`BusStats::bulk_gate_redos`
counts the redos — zero in every measured run). The staging tiles also
fold each outbound stripe, and block 0 publishes the per-(peer, stripe)
hash table behind the staged counters, so the engine posts the door's
hash without folding 256 KB on the collective thread per stripe.

Senders are paced in software (`BusOptions::bulk_pace_gbps` per (peer,
lane) QP — derived at bus start from the slowest lane's port rate as
port / ((W − 1) × lanes) × 0.85, 28.3 Gb/s on the four-node 200 Gb/s
fabric, overridable with `--bulk-pace-gbps` on dgpp-serve and
glm_gen_check, 0 = unpaced; `bulk_inflight_per_lane` bounds the window
and the posting order rotates per sender). The sweep put the loss knee
between 80 and 120 Gb/s per QP with throughput flat from 28 to 120, so
the derived value sits at the low end of the flat region with ~3×
headroom. The one-block kernel had throttled
every sender to ~1 GB/s; the grid let three 200 Gb/s senders burst
whole rings at one 200 Gb/s port and the switch dropped packets — RoCE
sequence errors, adaptive retransmissions, CNPs, 16 MiB all-reduces
bimodal at 2.4 or 30–56 ms (`scripts/roce_counters.sh` reads the
counters around a run). The NICs pace raw-packet QPs only, so the engine
spaces a lane's stripe posts by len / rate: six inbound QPs at the cap
stay under one port even when the ack timing aligns every sender on the
same receiver. Measured on the four nodes (`glm_gen_check --bulk-bench`,
p50): 2 MiB 0.30 ms, 4 MiB 0.53, 8 MiB 1.07, 16 MiB 2.17, 32 MiB 4.45 —
10.5–11 GB/s of wire traffic per rank, against 2.6 / 5.2 / 11 / 22.5 / 75
before the grid.

Measured (fabric, bitwise-verified against the canonical-chain oracle):
TP=2 p50 37.6 µs, TP=4 (full mesh) p50 ~44 µs, submit→wait, credits off
the critical path. Kernel-internal spans are ~6 µs staging, ~6 µs doorbell
claim, ~10 µs fold+reduce; the kernel launch (~10-30 µs) is the fixed
cost that graph capture (§6.2) amortizes in the decode path. Two
scheduling lessons from the bring-up, both now engine policy: the idle
branch's `sched_yield` was a hot-path syscall that cost ~2.2 ms of poll
latency per idle stretch (fixed: a genuine hot spin whose 50 µs sleep
tail is a power courtesy, never a latency mechanism), and bare
condition-variable waits pay a CFS-timeslice wake against the hot
engine (fixed: spin-then-futex on a release-published done flag — spun
before the mutex, since spinning under it deadlocks the completer). The
same fixes took the harness send path from 16.5 to 12.1 µs p50. The
engine also never naps once a graph is recorded, and it pins itself to a
fast core (the GB10's big.LITTLE mix put it on an A725 at times).

**What a kernel may believe about NIC-written memory (the pick-path race,
2026-09-01 — three bugs deep, all in the record):**

- *A doorbell does not certify its payload — for a kernel.* RC's CQE
  ordering protects the ENGINE (the CQE consumer); a kernel polling cells
  saw ctl-correct doorbells beside payload buffers still holding stale
  bytes, and folded garbage. Every doorbell therefore carries a 64-bit
  FOLD of its payload (`StartSlot.hash`; the sender's engine hashes the
  staged row it posts), and every collective kernel — latency, graph and
  bulk — spins until the payload folds to it before consuming. The fold
  is positional (`(word + position + 1) · golden ratio`, XOR-combined):
  the first XOR fold cancelled on uniform payloads and let a stale
  `bf16(2.0)`-fill pass as an all-zero broadcast.
- *Plain loads of pinned memory cache.* The GPU's plain loads of pinned
  sysmem allocate in its caches and the NIC's DMA writes never snoop them:
  the first spin pass fetched the pre-arrival zeros and every later read
  hit the cache, forever (a 56 s spin beside a payload the CPU could see
  complete). The doorbell polls always worked because `flag_load_acquire`
  is a system-scope atomic; the payload reads never had the discipline.
  EVERY kernel read of NIC-written memory — door fields, gate words, the
  fold's peer vectors, the bulk copies and exit fold, the claim records —
  is a system-scope atomic load (`sys_load_u16/u32/u64`). This was the
  mechanism of the original fabric pick corruption AND, almost certainly,
  the gen-1825 "sent-but-never-received" wedge family.
- The gate fires in production: 1–2 waits per scheduler smoke, thousands
  under the hostile racer (`bus_small_repro --pick-race`, loopback and
  fabric, an exact oracle after every collective under injected skew).
  Telemetry stays in the ctl cell (waits/spins + three claim records with
  door len/seq/hash); the STALLED dump prints the kernel's actual claimed
  cells beside the engine's own CPU fold.

Consequences upstream: the pick scratch is pinned (`cudaHostAlloc`), never
managed — a `cudaMallocManaged` buffer shared across loopback rank-threads
on one 2 MB UVM page is the documented concurrent-access race and hid
behind this bug for a day; and the readback invariant (a decoded broadcast
must equal the locally computed winner; today the digest group, §9) is
what turns any surviving corruption into a loud, located failure.

## 7. Attention and state semantics

### 7.1 KDA

The reference state contract is:

- recurrent state: FP32, shape per rank/layer `[16, 128, 128]`;
- merged q/k/v convolution state: activation dtype (BF16 here), width
  `conv_kernel-1`, plus speculative width when MTP is active.

The M2 implementation pins the concrete layouts (tested in
`tests/unit/kda_geometry_test.cpp` and the CUDA suite):

- recurrent state is `[local_heads, head_v_dim, head_k_dim]` with the
  K dimension contiguous — the indexing of the reference recurrent kernel;
- conv state is `[3*local_proj, conv_state_width]` BF16 with the width
  contiguous ("dim-first"), committed history in columns
  `[0, conv_kernel-1)` and the speculative reserve after it — used since
  M8 by the multi-row verify (§9), whose rows but the last also leave
  their post-row snapshots in `spec_conv_`;
- the fused in-projection row is `[f_a | g_a | q | k | v | b]` — the two
  replicated shards lead so the strided f_b/g_b GEMM inputs land on
  16-byte-aligned column offsets (cuBLASLt returns wrong results, not
  errors, for misaligned strided activations);
- the merged q|k|v conv weight is `[q(all heads) | k | v]` along channels,
  so a TP rank's slice is assembled per-section, not as a contiguous
  prefix.

Across 34 KDA layers this is 34.0 MiB of recurrent state per rank/request.
The committed three-position convolution history is 1.20 MiB. A three-token
draft reserve adds another 1.20 MiB, producing a fixed 36.39 MiB mutable slot
per rank/request. V1 snapshots copy that whole fixed-width slot (the
speculative suffix is zeroed/ignored), so the same conservative 36.39 MiB
figure drives admission and prefix-cache capacity—not a BF16 recurrent-state
estimate.

Prefill uses a pool-aligned chunk size (initially 2048 tokens) and carries the
exact final recurrent and convolution states between chunks. Decode mutates a
request's committed state only through the transaction described in §9. One
shared implementation of the recurrence serves both: chunk boundaries
round-trip the FP32 state through memory exactly, so chunked and unchunked
prefill agree bitwise on the recurrence path; residual differences in the
full layer come only from bf16 GEMM outputs that differ by ulps across
chunk-size-dependent cuBLASLt algorithms.

### 7.2 DSA/MLA and `index_kpool`

`index_kpool=4` means one compressed index entry for each four input tokens,
not one selection every fourth prefill chunk. Each query scores pooled entries;
the implementation selects pools, expands them back to token positions, and
includes the valid incomplete tail as required by
`index_kpool_always_select_tail=true`.

Requirements:

- cache block sizes and internal chunk starts are multiples of four; a final
  chunk may end with one to three tokens in the persistent incomplete tail;
- `select_k = index_topk / index_kpool` must be a **power of two** — the
  bitonic select/expand networks have no fallback shape (a select_k=6
  experiment silently dropped every selected pool; rejected at layer
  construction since);
- `num_heads` must be a **power of two ≥ 4** — the absorbed-attention
  kernel's head-group tiling partitions 64 lanes; heads=2 produced
  1e33-scale output garbage with *correct selections* (bisected through
  the CI machinery; 4/8/64 heads pass). Rejected at construction since;
- the in-progress tail persists across prefill chunks, decode steps, MTP
  verification, prefix attachment, and cache transfer;
- causal masking and deterministic tie-breaking operate on original token
  positions after pool expansion;
- top-k parity is tested against the reference implementation, including
  lengths 0–7 around pool boundaries and a partially rejected MTP step.

At 300,000 cached tokens, the replicated per-rank DSA budget is approximately:

| component | calculation | bytes/rank |
|---|---|---:|
| MLA latent | `300k × 11 × 512 × 2` | 3.379 GB |
| pooled index | `300k × 11 × (128 FP8 + 4 scale bytes) / 4` | 0.109 GB |
| incomplete tails | `11 × 4 × (128 K + 128 gate) × 2` | 22.5 KB/request |

The base cache total is therefore about 3.49 GB/rank plus block tables,
allocator metadata, and alignment—not 4.2 GB cluster-wide.

The M3 implementation pins the concrete layouts and the selection spec
(tested in `tests/unit/dsa_geometry_test.cpp` and the CUDA suite):

- the index K cache is **planar** — FP8-E4M3 rows `[slots, 128]` plus FP32
  scales `[slots]`, one slot per pool. The reference's packed 132-byte pages
  exist for a DeepGEMM `block_kv` constraint this engine does not inherit,
  and packed rows would misalign every other 128-byte streaming read;
- the latent cache is `[token_slots, 512]` rows in the configured storage
  format (`engine.kv_dtype`, 2026-09-06): BF16 by default, or fp8 (e4m3
  codes with one FP32 scale per row) or fp4 (e2m1 codes, two per byte, in
  blocks of 16 with an e4m3 scale per block over the row scale; the row
  padded to 16 bytes). The append kernel quantizes a row as it lands and
  the attention kernels dequantize a tile into bf16 shared memory as they
  gather it, so the math past the load is the bf16 kernel's; the index
  cache, the tails and the selection are the same in every format. One
  shared block table per request serves both caches (block = 128 tokens =
  32 pools, so token block *b* and pool block *b* are the same entry —
  prefix attachment shares both by reference, §8);
- the tail is a per-request ring `[2, kpool, 128]` BF16 — raw K at half 0,
  gate at half 1, ring slot `pos % kpool`. Completion reads the ring with the
  current token overriding its own slot (which still holds one stale pool's
  stash — the reference's `is_current` rule); the stash happens *after* the
  completion read;
- visible pools for a query at position *p* are `floor((p+1)/kpool)`; the
  incomplete tail is never scored, only appended. Sequences with
  `visible <= select_k` degenerate to dense causal selection through the same
  code path;
- selection is pinned as: the `select_k = topk/kpool` pools with the highest
  fp32 logits, exact ties to the **lower pool index**, output ascending in
  pool index. The composite sort key `(~sortable_fp32 << 21) | pool_idx` is a
  total order, which makes the selection deterministic on any correct
  implementation. The reference's radix path is `atomicAdd`-ordered at exact
  ties; this pin matches its deterministic path. Finite logits are assumed —
  real quantized cache rows are always finite (the saturating encoder never
  mints NaN);
- the **decode select is fused**: one kernel streams the blocked index cache,
  computes pool logits inline (warp per pool, lane per head, contraction-proof
  `__fmul_rn`/`__fadd_rn` arithmetic so the host oracle's logit parity — and
  therefore position parity — is bitwise), keeps a running top-`select_k` in
  shared memory, and merges block partials with a last-block reduction whose
  counter self-resets. Fixed grid, grid-striped over device-visible counts,
  zero logits materialized: the whole decode path is CUDA-graph capturable.
  Prefill instead materializes per-(row, head) fp8 dots through the IGemm
  interface (FP8×FP8→F32, unit scales, K-cache reads amortized across query
  tiles) and runs the same streaming selection over the dot buffer;
- **the dense prefill regime** (2026-09-05): while visible pools
  floor((p+1)/kpool) ≤ select_k = topk/kpool, every visible pool is
  selected and the tail appended — the query attends to tokens [0, p]
  exactly — which holds for every position ≤ kpool × (select_k + 1) − 2
  (2050 at the checkpoint's geometry): the whole prefill of a prompt up
  to the 2048-token chunk, and the first chunk of a longer one. Those
  rows run `attn_dense_kernel`, a flash-style kernel over 32 (row, head)
  M-rows per block — bf16 `mma.sync` for S = Q̃·Lᵀ and for P·L with a
  32-token latent tile shared by all 32 M-rows, the FA2 online softmax on
  the C fragments (denominator unrounded, probabilities rounded to bf16
  as the split kernel pins), causal masking per M-row, the split kernel's
  (m, l, c) partial layout so `dsa_attn_combine` is shared — in 128-row
  tiles split four ways; rows past that position keep the per-row split
  kernel over their selection. Tolerance-equal to the split kernel (the
  tensor core's summation order), deterministic; gated against it and the
  host oracle at 64/16 heads × 512/256 latent. 82 KB of shared memory, one
  block per SM (this device caps a block at 99 KB and an SM at 100 KB).
  Where the split kernel spent 318 µs per eight rows per layer (a
  2048-token prefill: 0.9 s of attention), the dense kernel spends ~360 µs
  per 128 rows. Past that position the SAME kernel runs over each row's
  selected list (`dsa_attn_listed`, round 8): a 16-row slab is one query
  row at local_heads ≥ 16, so each slab walks its own list into its own
  16-token latent tile and the two slabs run to the longer one's tile
  count — no union of selections, no membership masks; gated against the
  split kernel on arbitrary lists with repeats and against the host
  oracle. Before it a chunk past position 2048 spent 1.8 s in the per-row
  kernel (every row carrying the full 2048 selected tokens); after, 170
  ms, and a 2048-token chunk costs the same whether dense or sparse. The
  absorb and vout projections around it are tensor-core kernels too at
  ≥ 16 rows (round 7): absorb bitwise its warp kernel, vout carrying its
  fp32 input as a three-way bf16 split (the full mantissa; only the
  summation order differs); decode keeps the warp kernels;
- MLA runs absorbed: `q̃ = W_uk^T q` (bf16 GEMM rounding), scores
  `q̃ · latent` in fp32 with split-KV online softmax — the running max lives
  in per-lane registers fed by butterfly group reductions (no shared running
  state to order), probs round to bf16 for the `c` accumulation while the
  denominator stays unrounded, and the output is v-absorbed
  (`out_h = W_uv_h · c_h`). `kv_b` is consumed in the checkpoint's
  interleaved per-head layout;
- the power-of-two fp8 scale is computed by exact bit manipulation (smallest
  `2^n >= absmax/448`, exact powers mapping to themselves) rather than
  `exp2f(ceilf(log2f(v)))`: the fp32 libm form can round across a
  power-of-two boundary on near-tie inputs and clamp the row, and glibc and
  libdevice also disagree by ulps there. The exact form matches the reference
  except on pathological near-power-of-two magnitudes, where the reference
  clamps and this engine does not.

Layer-phase pins (the orchestration above those kernels):

- `DsaStatePool` owns all 11 layers' caches as five arenas (six with a
  quantized latent format's row scales `[layers][token_slots]` F32) — latent
  `[layers][token_slots]` rows in the cache's format, planar index K
  `[layers][pool_slots][128]` FP8, index scales `[layers][pool_slots]` F32, tails
  `[layers][requests][2,kpool,128]` BF16, and one shared block table
  `[requests][blocks]` INT32 — every region 256-byte aligned. Block
  management is host-side (LIFO free list; growth uploads only the new table
  slice on the caller's stream; growth is transactional — admission control
  per DESIGN §9 must guarantee capacity before the decode hot path, which
  cannot check device-side positions). Released blocks are not scrubbed: a
  new owner rewrites every row it reads before any consumer touches it;
- `DsaLayer` shares one scratch buffer across all DSA layers (they run
  sequentially on one stream; per-layer scratch would multiply the footprint
  by 11 for zero benefit), sized by `scratch_bytes()` with a dot budget that
  bounds prefill's materialized fp8 dots — query tiles shrink to fit,
  trading K-cache re-reads for a bounded footprint. Prefill grows block
  tables itself and throws on pool exhaustion (a control-path operation);
  decode requires the tables to already cover every position a captured
  batch will write. The decode path is CUDA-graph capturable end to end;
- the near-tie contract for cross-implementation selection comparison:
  logits from tensor-core GEMMs differ from any sequential oracle by ulps
  that can flip the rank-select_k cut when two pools sit within that noise.
  Divergences are certified, not tolerated: the audit re-derives the spec
  selection from the device's own inputs (bitwise, including the actual
  dot buffer) and requires the swapped pools to straddle the boundary
  within the measured per-row noise. Device-vs-device comparisons (same
  GEMM outputs) remain bitwise — that is the reproducibility contract
  serving relies on;
- decode performance pins (GB10, 48 SMs): native hardware e4m3→f16 and
  bf16→f32 conversions (both exact — bit-identical to the software codecs,
  proven by the bitwise select fuzz), split-KV attention at
  rows×32×head-groups blocks with an empty-split early exit, padded smem
  strides in the attention kernel (the natural [head][512]/[group][64]
  layout bank-conflicts 32 ways on every access), and 32-bit split key
  arrays in the bitonic networks (a u64 key spans two banks and XOR
  indexing conflicts ~16 ways). Decode at 65k context: 2.24 ms/layer, 114
  GB/s effective; the projection GEMMs alone are ~1.0 ms — 238 MiB of
  weights at the memory floor (see
  benchmarks/results/2026-08-28-dsa-m3-layer.md).

### 7.3 mHC residual streams

Pinned to the transformers `Glm5NextTextHyperConnection` reference (the
semantics every prior section treats as ground truth). The residual stream
is `hc_mult = 4` parallel bf16 streams of `hidden` each; all four start as
the embedding row, and the model output is the unweighted mean over
streams followed by the final RMSNorm. "mHC" names the manifold-constrained
mixing — `scale` has 3 entries because there are three outputs, not three
heads.

Per sublayer site (attention and FFN each own one), with `n = hc_mult`:

1. **Mapping.** Unweighted RMSNorm over the *flattened* `[n·hidden]` vector
   (fp32, eps = rms_norm_eps); an fp32 projection by `fn [(2+n)·n, n·hidden]`
   plus `base [(2+n)·n]`, split `pre[n] | post[n] | comb[n·n]`, each scaled
   by its own `scale` entry:
   `pre = σ(·) + hc_eps` (stream-collapse weights), `post = 2σ(·)`
   (block-output placement, range [0,2]), `comb = softmax(·, rows) + hc_eps`
   then Sinkhorn–Knopp toward doubly stochastic — one column pass, then
   `sinkhorn_iters − 1` row+column passes, every denominator adding
   `hc_eps`.
2. **Collapse.** The sublayer input is `Σⱼ pre[j]·streams[j]` (fp32
   accumulate, one bf16 round).
3. **Update.** After the sublayer produces `h`: `streams'[i] =
   bf16(bf16(post[i]·h) + bf16(Σⱼ comb[j,i]·streams[j]))` — post and comb
   are rounded to bf16 *before* the products, with intermediate roundings
   exactly as shown. The engine reproduces this choreography bitwise
   (kernel and oracle share it); deviation from it is a semantics bug, not
   a tolerance question.

The whole mHC pipeline is replicated on every rank (§5.1's "norm + mHC"
row): sublayer outputs are all-reduced first, so the stream state never
leaves the replicated region. The engine kernel computes the norm and the
24-logit projection in fp32 (fixed reduction order), applies
sigmoid/softmax/Sinkhorn in registers, and pins `hc_mult = 4` the way the
DSA kernels pin Hadamard-128 — widen only when a real checkpoint demands
it. Parity: `glm_mhc_test` vs the double oracle
(`glm_mhc_reference`), ulp-budgeted at the bf16 rounding points, including
saturated logits, zero streams (norm-of-zero), and bitwise-deterministic
end-to-end replay.

The prefill forms of the mHC kernels (2026-09-05, round 7): at ≥ 16
tokens the dots run four tokens per block with the coefficient matrix
staged through shared memory (`mhc_dots_tiled_kernel`; the per-coefficient
form derived a token's inverse RMS in 24 blocks and re-read its streams
48 times), every thread walking the same vectors in the same order with
the same block-sum tree, so the outputs are BITWISE the per-coefficient
form's (glm_mhc_test pins collapsed, post, comb and the fused norm), and
the finish runs in-block per token; the stream update computes a
position's four streams in one thread. Decode's fused per-coefficient
form (graph-captured) is unchanged.


### 7.4 MoE routing (noaux_tc) and expert execution

Pinned to the transformers `Glm5NextTextTopkRouter` / `Glm5NextTextExperts`
reference. Router math is fp32 end to end (the config's `moe_router_dtype`
contract): logits = fp32 GEMM of the post-LN hidden (bf16 values) against
the bf16 gate rows; `scores = sigmoid(logits)`; selection ranks
`scores + e_score_correction_bias` (bias-corrected) but the ROUTED WEIGHTS
are the UNCORRECTED scores gathered at the selected ids; `norm_topk_prob`
divides by (Σ + 1e-20) per element, then multiplies by
`routed_scaling_factor` (2.5). The engine's tie rule — equal biased scores
select the LOWER expert id — is pinned where torch's CUDA topk leaves ties
unspecified. The kernel emits ids in ascending expert order, which is also
the accumulation order: the reference's `index_add` loop visits experts
ascending, so per token `out = 0; for e ascending: out = bf16(out +
bf16(w_e·y_e))`, and the shared expert (weight 1) is added last as a single
bf16 add. Deviating from that order changes bits, not just tolerances.

Experts are swiglu MLPs with ASYMMETRIC clamps: the gate clamps only its
maximum (no lower bound), `up` clamps both sides at `swiglu_limit` (10);
`act = bf16(bf16(silu(g)) · up)` — two rounding points, matching torch's
opmath. The engine executes experts through the scale-aware GEMM on the
compressed E4M3+scale weights (§4).

Two execution paths, as built:

- *Prefill* (chunk-amortized; the grouped path since 2026-09-04, without
  a host sync since the same evening): the router's ids stay on the
  device and `moe_segment_kernel` (one block, a thread per expert) builds
  the segmentation there — the (token, slot) pairs by ascending expert,
  stable in (token, slot) within one, the slot→row map, a segment table
  with every expert (empty ones included; their blocks exit) and the
  shared segment last — while the routing traces ride async copies into
  per-layer pinned staging materialized after the chunk's final sync
  (`Outputs.routes` unchanged); the host-orchestrated `enqueue` remains as
  the reference the gates pin the device path against, bitwise. Then the
  layer's rows — every routed pair in segment order, the tokens once more
  for the shared expert — are gathered at once and run as ONE launch per
  matrix over every segment (`moe_grouped_mma_kernel`, 2026-09-05: one
  block per 64-column n-tile and segment walks the segment in 128-row
  m-tiles, stages the bf16 activations and the fp8 weights — decoded,
  scaled and rounded to bf16 exactly as the dequant bridge does — in
  shared memory and runs bf16 `mma.sync` with fp32 accumulation in
  ascending k16 order, so every output element is bitwise the scale GEMM's
  tile kernel and a segment up to 128 rows reads its expert's weights
  once; the shared segment alone splits across blocks in whole m-tiles),
  swiglu over every row, the fp32 down the same way, and the per-token
  ordered accumulation in one pass (`moe_accum_ordered_kernel`: the K
  slots sorted by expert id, `__fmaf_rn` from zero, the shared row at
  weight 1, one bf16 rounding — the same chain). The device segmentation
  is three launches (a count per expert, the scan, a placement pass per
  expert with a block exclusive scan over the match flags — the
  single-block form's stable order, bitwise) and the tensor-core kernel
  reads the hidden rows through the row map instead of a gathered copy
  (2026-09-05, round 7); the prefill router's dots are tiled 16 tokens x
  16 experts per block with the per-lane order preserved (bitwise the
  warp form) while decode keeps the fused warp form. `MoeExpertKernel` picks
  the grouped kernel: the prefill and the forward run the tensor-core one;
  `enqueue()` (the reference) takes it as an argument, GEMV by default,
  because the decode slot path and the MTP module's multi-row eager rows
  are pinned bitwise against the GEMV core (`moe_grouped_gemv_kernel`:
  rows staged four at a time through the same `fp8_gemv::block_rows` the
  decode GEMV uses — which on a 2048-token chunk's ~57-row segments read
  every expert fifteen times per matrix, 3.4 s of the prefill). Before
  either the experts ran per segment through the small-M tile GEMM —
  578 µs a call on an 8-block grid — which was 3.5 of a 256-token
  prefill's 5.2 s. The expert-view table the grouped kernels read is
  uploaded per layer from a RING of pinned host tables, each guarded by
  the event its last upload recorded (`upload_expert_views`, 2026-09-05):
  one MoE layer object is rebound for every layer and the session prefill
  has no per-layer host sync, so a single pinned table was refilled with
  the next layer's pointers while the previous layer's asynchronous copy
  could still be pending — that layer then ran on the wrong experts
  whenever the host got a layer ahead (a world of one has no collective
  to wait on; the fabric's per-layer all-reduce had hidden it). The host
  waits only when it is four uploads ahead, and the copy sits right after
  the router and segmentation kernels, so the wait never drains the
  stream.
- *Decode* (`GlmMoeLayer::enqueue_decode`, M6 Stage 4c and rounds 2/7/8):
  zero host round trips. The router leaves ids ascending per row on the
  device; a slot-ranking kernel orders the (row, slot) work by expert id
  so experts shared across rows are L2 hits; the gate+up+swiglu kernel and
  the down kernel are fp8 GEMV cores (warp per weight row, 16-byte fp8
  loads, the scale grid applied in the epilogue — `fp8_gemv.cuh`; every
  row a slot, `top_k+1` slots per row, the +1 the shared expert) reading
  the route and the expert-view table from device memory (the eager path
  uploads that table from the same guarded ring as the prefill; capture
  reads its per-slot graph tables); the down dots
  stay fp32 (unrounded) and one ordered accumulation kernel runs the
  chain — one fma per expert ascending, shared last — rounding to bf16
  exactly once as the sum leaves for the all-reduce. Since the sliced
  placement (§5.2) a rank's slice cannot reproduce the whole expert's
  bf16 per-expert rounding, so this fp32 chain is the more accurate of the
  two reachable deviations (decision with the user, 2026-09-02); the
  unsliced engine sits at max 1 ulp against the double oracle, and the
  sliced form is certified against the UNSLICED oracle with a per-element
  bound built from the partials in hand (bf16-on-the-wire's cost, the
  class the attention o_proj already pays). The expert-view tables live
  in per-graph-slot DEVICE tables uploaded once before capture (a
  binding-keyed cache was a wrong-weights factory under the streaming
  loader, whose bindings are not identity-stable — the rule stands). The
  router's select runs behind its dots kernel (last-block ticket), and
  the top-k is a warp with the exact strict-> rule (bit-identical). Both
  paths are pinned bitwise against each other by `glm_moe_test` at three
  geometries including a TP partition.

Fusions tried on the decode MoE and rejected with numbers (2026-09-03):
the accumulation behind the down kernel (+22 µs/layer — a barrier + fence
+ atomic per block over 9216 blocks), the slot ranking in the gate_up/down
prologues (+10 µs on the 9216-block down launch against a 1.6 µs kernel).
At that grid a per-block prologue is 24 waves deep; the launch it saves is
~1 µs.

Route traces (deliverable 5): the router records per-layer ids/weights in
the `DGPPTC1` format, and `tools/route_trace_traffic.py` replaces §3's
uniform-expert assumption with measured occupancy — per-token busiest-rank
experts/layer (uniform expectation 3.515), per-layer batch critical rank
for correlated routes, and the corrected critical path in GB/token. The
tool's anchors are §3's own numbers: a uniform-random trace must reproduce
3.515 experts and 7.457 GB/32.42 ms; an all-one-rank trace must reproduce
12.198 GB. A uniform trace is the null model, not evidence — real traces
come from representative prompts through the assembled model.

### 7.5 The assembled forward and the parity discipline at depth

`GlmDiagnosticModel` (M4 deliverable 1; the name predates its serving role
and stayed) runs the full text stack: embedding -> 4-stream mHC init -> per
layer [attn_hc -> ln1 -> KDA or DSA -> stream update] and [ffn_hc -> ln2 ->
dense MLP or MoE -> update] -> unweighted stream mean -> final norm ->
lm head (fp32 accumulators since 2026-09-03: `GemmOut::F32`, the same
GEMV template — the bf16 rounding of the logits was the first thing the
teacher-forced gate could see). In streaming mode one layer is resident at
a time and the KDA/DSA/MoE layer objects are REBOUND to each layer's
resident views; in resident mode every layer is materialized at
construction (§3) and the bindings are lifetime-stable, which is what
graph capture requires. The two layer norms and the final norm are the
Glm5NextTextRMSNorm TWO-rounding choreography (u = bf16(x*rsqrt); y =
bf16(w*u)) — a dedicated kernel, because the generic single-round rmsnorm
drifts the GLM path systematically. The MTP draft layer (index 45) is
loaded when `--mtp` asks for it and runs as §9's draft block; the reference
model class ignores it.

**Sessions (the incremental engine, M6 Stage 2/2b).** The layer pipeline
is ELEMENTWISE IN TIME: the mHC streams are recomputed per layer from each
token's embedding, so all temporal recurrence lives in the KDA
recurrent/conv state and the DSA latent/index/tail caches, and a stateful
step is "don't reset the state, feed one token". `session_prefill(req,
prompt)` opens a request slot (fresh state) and runs the prompt in
pool-aligned 2048-token chunks keeping state across them, returning the
last row's logits; `session_step(req, token)` runs one token at the slot's
next position (the same recurrence implementation as prefill — what keeps
cross-path noise at GEMM ulps); `session_close(req)` releases the slot's
blocks immediately (a meter that lags a retire is an admission deadlock).
Slots: `max_requests` (≤ `kDecodeRows` = 8, this model's fixed batch and the
DSA select bound; the session-core families derive their decode rows at
runtime, engine/decode_outputs.hpp) with
slot-major KDA state so one open is one memset pair; the DSA pool's
per-request tail rings are zeroed at open and the latent/index caches are
deliberately not scrubbed (a new owner rewrites every row it reads — the
pool's release contract, confirmed by the real-model gates). Rules the
gates pinned: a continuation chunk must carry ≥ kpool tokens (the DSA
tail-seed read is pinned to in-chunk rows), so a 1..kpool−1 token tail
borrows one pool from its predecessor; a plain `forward()` while any
session is open throws (the pools are shared — the corruption would have
looked like noise); every host op that moves a position pushes it to the
device mirrors (`d_session_pos_`, `d_mtp_pos_`), so the device-driven
graph (§9) always starts from the host's view. Eager decode remains
time-multiplexed. M6.6a's batched graph runs one fixed slot-major row batch:
every stateful decode kernel takes the same request-indexed map (one span per
configured slot, per-row request ids selecting the actual state slot, and
positions with −1 marking padding). `KdaRequestRows` gives KDA conv and
recurrence the map DSA already used; DSA's ring update honours the per-row id
rather than the span ordinal, and both layers write zeros for padding rows so
a padded row's block output is deterministic by construction. Snapshots use global physical-row offsets so
each request's commit can independently restore its first accepted state.
The adaptive adapter also owns one slot-specific scalar graph per request
slot. The scheduler's ordered `step_batch` still hands it every live slot;
the adapter executes those scalar variants sequentially below the crossover
or advances all slots in the single row-batched replay above it.

Parity at real depth is chaos-limited, and the curated suite is designed
around the measured facts, not around an aspiration of bit-parity:

* Module noise floors (measured, M2/M3/M4): KDA engine-vs-torch
  3.6e-3 l2, DSA ~3e-3 kept-row (fp8-indexer near-tie flips are inside
  its design), MoE expert path 0-1 bf16 ulps vs the double oracle.
* Those floors COMPOUND: a free-run 45-layer forward diverges from the
  torch reference at ~1.18x per layer and decorrelates by layer ~30
  (measured; the mHC residual stream is mildly amplifying — post in
  [0,2], Sinkhorn-normalized comb columns). No bug is involved: every
  layer's ISOLATED drift sits flat at the floor (<= 7e-3 at layer 44 as
  at layer 0).
* Therefore the real-checkpoint suite compares per-layer ISOLATED: every
  layer starts from the reference trajectory, so drift is bounded by one
  layer's floor. Kept rows (tokens whose routing did not flip) must sit
  under 2e-2; a route flip legitimately moves that token's row O(1)
  (the reference's noaux bias exists to tie scores at the selection
  boundary, so cross-implementation noise flips them). Since the M4
  close-out, every flip is CERTIFIED, not counted: the router kernel
  exports its full biased-score row, the reference dump carries its own,
  and `models/glm/route_audit.hpp` requires (a) the engine's selection to
  be the spec top-k of the engine's OWN biased scores and (b) every
  swapped expert pair to straddle the boundary within 32x the token's
  MEASURED cross-implementation noise — the noise yardstick taken over
  the experts NOT involved in the swap, so corruption concentrated on the
  swapped experts cannot hide inside its own inflation (the router
  itself is separately pinned to 5.3e-7 against the double oracle). The
  head runs on the isolated final streams: top-1 must agree on every
  token, and a disagreement must likewise certify as a boundary near tie
  (reference top-2 logit margin within 32x the measured logit noise) —
  a reduced layer budget crowds the head's boundaries, and the
  truncated-stack suite exercises exactly that.
* Free-run outputs are still REPORTED (drift curve, route agreement
  rates, top-8 margins) — they are the honest description of what
  cross-implementation bf16 inference at 45 layers looks like — but no
  free-run parity is asserted.

The first real trace (technical prose, 272 tokens) lands on the uniform
null within measurement noise: busiest-rank 3.541 experts/token (uniform
3.515), corrected critical path 7.484 GB/token vs 7.457 (+0.4%). The
§3 traffic model's uniform assumption is validated for this prompt class;
sampling other prompt classes (code, multilingual) is future work and the
tool accepts their traces unchanged.

### 7.6 Decode kernels and prefetch

Graph decode records the layer walk, collectives and token-selection
kernels into reusable variants. Its main costs are weight reads, small
latency-bound operators and collective synchronization. The recorded
GLM-FP8 T=1 run on 2026-09-03 took 31.3 ms/token against an estimated
24.5 ms weight-read floor at TP=4. See
[benchmarks](docs/benchmarks.md) for subsequent measurements.

**Row-independent GEMV.** The BF16 and FP8 cores use a warp per weight
row and vectorized loads. Small row batches preserve each row's scalar
reduction order. Larger decode batches split into supported chunks, so
speculative verification and occupancy changes do not change the row's
arithmetic. Weight storage uses device allocations; collective staging
uses registered pinned memory.

**L2 prefetch.** `WeightPrefetcher` runs bounded weight windows on a
low-priority side stream while the model executes collectives or other
latency-bound work. Window placement and size are model-specific. GB10's
GPU, CPU and NIC share memory bandwidth, so aggressive prefetch can delay
the collective handshake and outweigh its benefit. Tune it against whole
decode steps, not an isolated memory probe.

**MoE.** Device slot tables identify each selected expert's weight slices.
The decode path performs routing, expert products and accumulation without
host segmentation. Prefill groups rows by expert for tensor-core work.
Both preserve the accumulation rule in §7.4.

**Packed full-GLM prefill.** Int4 routed experts and int8 attention/shared
projections use `packq_gemm` from 128 prompt rows. Each 32 × 64 output tile
stages two buffers of packed weights and BF16 activations. Tensor cores
multiply exact integer codes by activations; FP32 FMAs apply the BF16
scale to each 64-element partial dot. This preserves the exact dequantized
weight values without a BF16 weight copy or per-weight rounding. It changes
summation order relative to GEMV, so the two paths use numerical rather
than bitwise parity. Grouped gate/up reads original hidden rows through
the device row map, and grouped down stores FP32 before ordered expert
accumulation. The threshold counts rows in each invocation; grouped MoE
prefill can cross it by combining several shorter prompts. C1 short prompts
and all current decode/verification batches retain the packed GEMV core.

**Graph constraints.** Copy and reset operations in the captured path use
kernels to avoid copy-engine dependency cycles in in-process multi-rank
tests. Graph validation checks node types at capture. Qwen's mapped
n-gram path declares a host gather callback; see
[the graph-stall investigation](docs/batched_mtp_graph_stall.md) for the
constraint and its current exception.

**Measurement.** Separate replay latency from tokens per replay when
evaluating MTP. Kernel timing, launch gaps, collective skew and acceptance
explain different parts of the result. Numerical changes need the
margin/loss checks in §12; greedy MTP and batch transitions retain their
bitwise comparisons.

## 8. Prefix cache

The prefix cache reuses exact session snapshots at valid prefill cuts.
`src/sched/prefix_cache.*` tracks token prefixes, ownership and eviction;
`src/engine/prefix_arena.hpp` stores model snapshots. Each model supplies
snapshot, attach and release operations, so snapshot size and cache
contents depend on its state representation.

Image requests bypass this token-only index. Equal placeholder IDs do not
imply equal image embeddings; safe reuse requires image content and geometry
in the prefix identity, including for generated continuation snapshots.

For GLM-5.3, a snapshot contains KDA recurrent and convolution state, DSA
tail rings and the draft block's last hidden row. Complete cache blocks
are pinned by reference; a partial block is copied so the cached prefix
cannot be changed by continued decoding. Attaching copies mutable state
into the request's slot and shares the immutable blocks.

**Cuts and numerical equivalence.** The text frontend identifies
structural boundaries in prompt token IDs. Prefill cuts at their
pool-aligned positions and at the regular chunk boundaries. For GLM's
`kpool=4`, a boundary at `b` maps to
`floor(b / kpool) * kpool`. Lookup considers only the new prompt's valid
cuts. An attach followed by suffix prefill therefore uses the same chunk
sequence as a cold prefill, preserving its arithmetic. Merely matching
token IDs at an arbitrary position would not establish that property.

**Snapshot lifecycle.** A cold prefill saves its deepest reusable cut.
During decode, a request maintains a rolling snapshot at aligned committed
positions. At retirement, that snapshot can become a close-time entry for
the next conversation turn. Reuse requires the next prompt to contain
the same token prefix, including the rendered answer and relevant
reasoning/template content.

MTP can commit past an aligned position in one step. The scheduler can
arm a post-row snapshot for that position; the engine reconstructs it
from speculative state snapshots and the corresponding cache references.
This prevents two-token steps from skipping every usable boundary. The
engine checks the expected position when saving the snapshot.

**Ownership and eviction.** `engine.prefix_cache_gib` sets the snapshot
arena budget per rank; zero disables the cache. The slot count is derived
from the model's snapshot size, not a fixed bytes-per-token estimate.
Entries hold references to their cache blocks, which count against pool
usage. Admission can evict the least-recently-used eligible entry when it
needs blocks or an arena slot. Entries attached to live requests are
protected, and blocks are freed only when their references reach zero.

**Rank agreement.** Every scheduler derives lookup, snapshot and eviction
decisions from the same journaled inputs. The warm record supplies rank
0's slot count; each tick carries its previous decision digest. A peer
compares its own digest before applying the next record and exits on a
mismatch. Decisions also appear in the operation stream.

**Diagnostics.** Metrics report hits, misses, saved tokens, snapshots,
evictions, pinned blocks and copy times. On a miss, the cache can report
the nearest live prefix or a matching entry in its recent-eviction history.
These diagnostics do not affect lookup or eviction. Entries are local to
the process and are not persisted across restarts.

Tests cover cached/cold equality, short continuation chunks, snapshot
ownership, full-arena eviction, and MTP steps that cross a snapshot cut.
The capacity measurements are in [the v1 sign-off](docs/signoff_v1.md).

## 9. MTP transaction model

MTP verifies a pending token and up to three draft tokens per request.
Depth 1 is the default. Greedy verification must produce the same output
and committed state as plain decode; sampled verification must preserve
the target distribution.

1. Verification runs `1 + depth` rows using row-independent arithmetic.
   Stateful kernels save intermediate states needed for rollback.
2. Each rank evaluates the gathered verifier results and computes the same
   accepted count. A candidate-table result that cannot resolve sampling
   triggers an exact-gather fallback.
3. If all rows are accepted, the final state is already in place. After a
   rejection, the model restores state after the accepted prefix and
   rewinds the logical cache position. Positional KV writes beyond it are
   overwritten by later rows and remain invisible to committed queries.
4. The draft block processes the accepted rows and proposes the next
   drafts. Deeper drafts recursively use that block's own output; their
   cache writes remain provisional until the corresponding tokens are
   verified.
5. Cancellation is applied between scheduler ticks. Rank failure ends the
   service; the HTTP layer reports completed tokens and the applicable
   failure event. In-flight speculative work is not published as committed
   output.

### State snapshots and rollback

GLM's KDA recurrence/convolution and DSA tail-ring kernels save state after
each speculative row except the last, which remains in place.
`GlmDiagnosticModel::session_rollback` restores the snapshot after the
accepted rows. DSA latent rows and completed index pools use positional
writes; their visibility follows the rewound position. Verification uses
at most four rows per request (`kSpecRows`), within the row-independent
GEMV path.

Tests force rejection at each depth, including index-pool boundaries,
and compare subsequent state and greedy output with plain execution.
Other model families implement these operations through the same session
interface, using their own recurrent and K/V state layouts.

### GLM draft layer

The draft block (`models/glm/mtp.cpp`) is the checkpoint's layer 45: a plain
pre-norm DSA + MoE block (no mHC) over `eh_proj([enorm(embed(tok_{q+1})) |
hnorm(h_q)])`, where `h_q` is the main stack's pre-final-norm stream mean
kept in a per-position cache, headed by `shared_head.norm` and the shared
lm head. It owns one more DSA pool ordinal and MoE graph slot, runs over the
prompt's rows 0..P-2 at prefill, and afterwards over exactly the rows the
main stack accepted (only accepted tokens ever enter it, so it never rolls
back); its row for hidden position q uses position q (vLLM's convention —
a uniform shift is RoPE-invariant but not kpool-invariant). Depth is 1
because the layer is trained at depth 1 and because the verify's cost is
linear in rows through the MoE (a second row's experts are new DRAM bytes
unless shared with the first; the decode slots run in expert order so
shared experts are L2 hits): measured 39.8 ms for T=2 vs 31.3 for T=1, so
a k-th draft must be accepted well over half the time to pay. Rank 0 does
not broadcast an accepted count: every rank folds the identical
candidate table and computes the identical verdict (`judge_verify`); the
pick's readback check pins the equality.

**The draft depth (2026-09-06).** `engine.mtp_depth` (1–3) makes the
verify 1 + depth rows and the block propose the later drafts by recursion:
after its rows off the verdict (the head at the last accepted row gives
draft 1), it runs one more row per further draft at the position after
its counter — token the previous draft's pick, hidden its own previous
output row copied into the position cache there (the block's `h^1` in
place of the main stack's `h^0`; vLLM's single-module chaining), position
the counter plus the chain index, the counter unmoved. Everything a chain
row writes is positional and provisional: the next step's real rows
overwrite the latent, the pool and the hidden. The one non-positional
write is the block's tail ring — a second chain row can stash over the
slot a still-open pool's member holds — so the ring is copied before the
chain rows and restored after (`chain_ring_copy`, two 2 KB kernel copies).
The verdict, the commit and the rollback are unchanged over T rows; the
sampled verdict decides the rows as a chain (row t tests the draft fed to
row t+1, a stand moves on, a reject ends on the residual, the last row
reached samples plainly) with the same draws the eager `SampledSpeculator`
makes, and a host fallback at row t continues from there — the rows before
it committed on the device, the gathered row decided under the device's
normalizer, a draft that stands re-running the next row eagerly and
testing the next draft on it. The picker holds one slot per draft; the
feed is `[next, draft_1 .. draft_depth]`; the hop snapshot takes the rows
the step committed past the position. Depth 1 is the two-row step
byte for byte. GLM-5.3 uses scalar graphs beyond depth 1. Measured (2026-09-06, one request): 31.5 ms plain, 42–43 at
depth 1, 54–56 at depth 2 — a verify row is its own expert bytes (~10 ms),
the chain row ~2.5 — and the second draft stands 45–65 % of the time
(prose to code), so depth 2 is −4 % on prose and +4 % on code and JSON;
depth 1 stays the default, the transcripts are identical at every depth.

### On-device verification and drafting

The depth-1 GLM graph includes the following operations:

1. Derive speculative row positions from `d_session_pos_` and read
   the token feed produced by the preceding replay.
2. Run the two-row main-model verify with its recorded boundary folds.
3. Gather rank-local pick tables and derive the same verdict on every
   rank. The greedy table contains canonical candidates encoded as
   lossless digits; a digest group checks the preceding verdict's
   readback. Sampling extends the table as described in §10.
4. Commit the accepted rows and restore intermediate state after a
   rejection using `glm_spec_commit`. Advance the session position
   by the committed count.
5. Prepare draft-block rows from the verdict. Unused rows have negative
   positions and do not update cache state. Save the verifier's next
   token before the draft pick reuses the verdict slot.
6. Run the draft block and select its next proposal. Deeper MTP also
   runs chained draft rows through the block.
7. Write the next token feed, preserving state needed for a fallback
   or later host bookkeeping.

`glm_pick_test` checks device kernels against host oracles. The
loopback graph tests compare verdicts, subsequent state and greedy
transcripts with eager speculation. The model's decode-row limit and
graph-family selection determine how many requests share a replay (§11).

### Pipelined replay

The engine can return a step's verdict while the graph's draft tail is
still running. A kernel publishes a replay sequence to pinned memory
after verification, allowing the host to process the verdict and enqueue
the next replay without waiting for that tail.

Scalar and batch variants share persistent device token feeds. Grammar
masks and other host-prepared rows use a staging handshake:
`glm_stage_wait` waits for the host's published sequence, then
`glm_upload_words` reads the staging rows. The wait is bounded and
reports a timeout through pinned status. Publish the staging sequence
before draining a replay that depends on it.

The bus supports two live replay windows with FIFO completion. Each
shape alternates between two graph variants so cells are not reset while
their previous window is in flight. Prefill, graph capture and sampling
fallbacks drain live windows before issuing eager collectives. A fallback
also marks its provisional draft state for replacement before draining.

`DGPP_PIPELINE=0` settles each replay immediately after launch;
`DGPP_PIPELINE_TRACE=1` logs launches and settlements. The stats
line's `ms/step` measures the engine call interval, which is
verdict-to-verdict during steady pipelined execution. Compare tokens per
second as well as this interval when evaluating a change.

The verdict uses a kernel-published sequence rather than an external
event-record node. External event records caused replay stalls in
in-process multi-rank tests and are rejected by graph validation. See
[the investigation](docs/batched_mtp_graph_stall.md) for evidence.

### Sampling and draft depth

The graph picker applies each request's penalties, constraints and sampling
settings. GLM's greedy draft is accepted using its probability under the
verifier distribution; a rejection samples the residual distribution.
Qwen also supports sampled proposals, with acceptance based on the ratio
between target and proposal probabilities. Both paths use exact gathered
logits when the candidate table cannot resolve the decision.

A fallback restores provisional draft state, completes the decision on
the host and prepares the next token feed. Tests compare the device path
and fallback against the same sampling oracle, including count tables and
subsequent draft state. Greedy MTP is checked for transcript identity;
sampled runs are checked against the corresponding speculative algorithm.

`engine.mtp_depth` selects 1–3 draft tokens. GLM-5.3 and Qwen use scalar
graphs beyond depth 1. GLM-4.7 supports deeper batched draft chains within
its runtime row limit. Depth 1 remains the default because extra verify
rows read more expert weights, and higher acceptance does not always
offset that cost. See [the MTP guide](docs/mtp.md) and
[benchmarks](docs/benchmarks.md) for configuration and measurements.

The first draft after admission is prepared eagerly after prefill selects
the pending token. Confidence-based selection of verification depth is
not implemented.

## 10. Tokenization, templates, logits, and sampling

### Tokenizer and templates

`src/text/tokenizer.*` implements the supported GLM and Qwen
ByteLevel-BPE configurations. It validates the checkpoint's regex,
normalizer, BPE options and added-token rules at load. GLM uses no
normalizer and `ignore_merges=true`, including a whole-word vocabulary
lookup before merging. Qwen uses NFC normalization and performs the
merge walk. Unicode tables and normalization live under `src/text/`.
Unsupported tokenizer configurations are errors.

The tokenizer is loaded once; encode and decode are const and
thread-safe. Golden corpora compare token IDs and decoded bytes against
reference tokenizers and are keyed by the tokenizer file's hash. A
different checkpoint revision must use its matching corpus.

`src/text/chat_template.*` parses the checkpoint's Jinja template at
model load and renders its syntax tree per request. It implements the
supported statements and expressions with transformers-compatible
whitespace, truthiness and JSON rendering. Unsupported constructs fail
during parsing; rendering errors identify the template line. Rendering
state is per call, so the parsed template can be shared. Golden tests
cover rendered bytes and encoded IDs for text, reasoning and tool calls
in each supported family.

### Sampling semantics

The model exposes FP32 logits for its vocabulary slice. Greedy selection
merges local maxima in canonical order: descending logit, then ascending
token ID. Stochastic selection uses the same ordering in the host and
device implementations.

Sampling applies repetition, frequency and presence penalties, logit bias
and the constraint mask before selection. Temperature scaling is followed
by top-k, min-p and top-p; the token crossing the top-p threshold remains
in the set. The selector walks FP32 probabilities with an FP64 cumulative
sum. A counter-based SplitMix64 RNG derives draws from the request's
seed and counter. Greedy selection consumes no random draw.

Defaults come from `generation_config.json`, with explicit handling of
missing fields and EOS IDs. Process flags and request fields override
them. `/v1/models` reports effective defaults. The service journals
sampling parameters as float bits together with the seed, allowing peers
to reproduce the request's draws. Eager and graph engines both support
sampling; a backend without that capability rejects stochastic requests.

### Candidate tables and exact fallback

Each rank supplies its local top candidates and a slice log-sum-exp.
Merging the sorted lists produces the exact global prefix; folding slice
normalizers in rank order accounts for vocabulary entries outside it.
`sample_from_prefix` resolves only when that prefix contains enough
information for the request:

- Finite top-k needs the complete requested support.
- Min-p needs the cutoff, and top-p needs the nucleus crossing.
- Unrestricted temperature sampling resolves when the draw falls within
  the prefix's probability mass.

A fallback leaves the reserved draw unchanged. `bus_gather_logits`
then transports each rank's penalized FP32 slice to every rank, and
`sample_complete_logits` completes the same decision under the
already-folded normalizer. Exact prefixes widen through 1,024, 8,192 and
65,536 candidates before a full sort if needed. A complete-list fallback
must use the same sharded normalizer; substituting a differently rounded
denominator can change a threshold decision.

Candidate and normalizer fields use lossless digit encodings in BF16
collective buffers. Full-logit gathering uses four 8-bit digits per FP32
value, or eight wire bytes per vocabulary ID. Each slot has one nonzero
contributor, so summation preserves the digits, including the original
NaN payloads, signed zero and subnormal bits. A decision digest checks
that ranks interpreted the gathered data identically.

`engine.sampling_candidates` defaults to 128 per rank and accepts
1–256. The graph picker also accounts for the configured row count and
available latency-slot space. Narrower tables can increase fallback
frequency without changing the sampling distribution.

### Device execution

`src/kernels/sample_pick.*` prepares penalized logits, computes slice
normalizers, selects local candidates, merges rank tables and evaluates
the verdict. Each row retains scalar arithmetic. Normalization uses fixed
256-element chunks and pairwise trees; `common/det_math.hpp` supplies
host/device-matching exponential and logarithm functions. Explicit FMA
operations and compiler contraction settings keep the numerical contract
consistent across implementations.

The local selector uses a composite logit/ID key. A bound from per-thread
maxima narrows the candidate list; exact radix and tie-handling paths
cover lists too large for the small sort. The verdict merges canonical
prefixes, folds normalizers in rank order and applies the shared selector.
Tests compare candidates, normalizers, tokens, logprobs, counters and
digests against the host oracle, including flat distributions and ties.

Each request slot holds sampling parameters, a counter RNG and a token
count table. A speculative row sees the committed context plus the
appropriate fed tokens. The commit updates counts for accepted rows.
On fallback, the adapter gathers the saved verifier logits, completes
sampling on the host and replaces the provisional feed and draft state.
Verifier logits are saved before the draft head reuses their buffer.

### Logprobs and validation

At positive temperature, logprobs describe the final filtered and
renormalized sampling distribution. At temperature zero they describe
the raw distribution used by the greedy reporting path. Chat requests
can ask for `logprobs` and up to 20 `top_logprobs`; the legacy
route uses its integer logprobs field. Each returned token carries its
corresponding report through scheduler events and the HTTP response.

The common greedy path avoids the full sampling calculation. Greedy
requests needing penalties or logprobs use the reporting path while
retaining canonical argmax selection. Eager/graph and host/device tests
compare both results and reports.

`scripts/fabric_sampling_profile.py` evaluates candidate mass on
teacher-forced texts; `scripts/serve_width_sweep.sh` measures fallback
cost on served workloads. Diagnostic gathers are excluded from throughput
measurements. See [numerics](docs/numerics.md) for the procedure and
[benchmarks](docs/benchmarks.md) for dated results.

### Constrained tool calls

`src/text/tool_grammar.*` implements the tool-call formats used by the
supported templates. A per-request grammar supplies the allowed token
mask for each position. `tool_choice` controls whether calls are
optional, required, forbidden or restricted to a named function;
`parallel_tool_calls: false` limits the turn to one call. The grammar
also applies in auto mode to validate calls the model chooses to make.

Names and keys are matched through token text, allowing different BPE
segmentations of the same valid string. Schemas determine argument types
and whether undeclared keys are allowed. `grammar_tool_from_function`
builds the constraint. With `function.strict: true`, unsupported
schema keywords are rejected by path. In non-strict tools, supported
types remain enforced while unsupported value restrictions are logged
and left unenforced. Integer bounds are supported; arbitrary numeric
bounds, string patterns and formats are not all enforceable.

A masked token is absent from the sampling distribution: it has zero
mass, is excluded from top-k and cannot be selected by a rounding guard.
An entirely masked vocabulary slice contributes no normalizer mass.
Host and device samplers use the count of allowed IDs and renormalize
over that set. A masked draft is always rejected.

Grammar state belongs to each request slot and advances only with
committed tokens. Each rank reconstructs it from the journal's grammar
specification and the same tokenizer. For MTP verification, each row's
mask reflects the corresponding tentative prefix; it is used only if
the preceding draft was accepted. Draft generation itself remains
unconstrained. The prefill pick is the first constrained position.

The graph saves verifier logits after penalties and masking, before the
draft head reuses the buffer. Exact-gather fallback reads that snapshot
and checks its covered mass against the device result. Tests compare
host/device masks, sampled outcomes, constrained eager/graph execution,
and tool-call goldens over each real tokenizer.

### JSON output

`src/text/json_grammar.*` implements `response_format`:
`json_object` requires one complete JSON object; `json_schema`
requires a JSON value conforming to the supported schema. JSON output
can be combined with tools: under `tool_choice: auto`, the first value or
tool opener selects either a schema-conforming answer or a tool-call turn.
`none` forces the JSON answer; `required` and a named choice still require
tool calls. Reasoning can precede either branch, but EOS is withheld until
the answer or required call is complete. The combined mode and schema
travel through the admission journal to every rank.

The implementation has three layers:

1. `JsonLexer` tracks RFC 8259 syntax: containers, strings and escapes,
   numbers and literals. Structural whitespace is limited to 16 consecutive
   bytes so constrained generation cannot spend its whole budget on
   whitespace. Whitespace inside a string does not count toward that limit.
2. `compile_json_schema` validates and compiles supported keywords:
   types and type lists, properties, required keys, additional properties,
   item schemas and counts, scalar enums and constants, `anyOf`, and
   numeric bounds. Unsupported keywords report their schema path.
3. `JsonMachine` combines lexical state with schema cursors. It filters
   value types, constrains keys and enum values, checks required members
   and array sizes, and keeps alternatives for `anyOf` until input
   disambiguates them. A value is complete when an applicable cursor accepts.

Integer-only schemas admit neither fractions nor exponents. Bounds are
normalized to an inclusive int64 range, rounding fractional limits inward
and adjusting exclusive limits. For a magnitude prefix M, possible
completions lie in intervals `[M * 10^k, (M + 1) * 10^k - 1]`.
The cursor rejects a digit when no completion can satisfy its signed
range and checks the final value again at termination. Bounds combined
with enums filter the enum members during compilation.

General `number` bounds retain decimal significant digits and exponents.
Before an exponent opens, a mantissa prefix describes an interval at each
decimal scale; after it opens, the fixed mantissa determines a range of
allowed integer exponents. A prefix is admitted only if some continuation
can satisfy the bounds, and termination checks the complete value. Decimal
comparison avoids rounding an out-of-range generated value onto a boundary.
Bounds use the schema DOM's numeric values (int64 or double, serialized in
shortest round-trip form). Inclusive and exclusive bounds apply equally to
response schemas and tool arguments; nonnumeric alternatives are unaffected.
Tool-call parsing preserves the original JSON when reserializing through
the numeric DOM would round a value or exceed its range, so normalization
cannot move a constrained argument across its boundary.

`JsonTables` precomputes lexical token behavior and groups tokens with
the same structural prefixes and tails. Mask construction can then test
representatives, with individual simulation for stack-dependent or
schema-constrained text. Bounded-integer positions also recheck numeric
tokens against prefix arithmetic. Tests compare the optimized mask with
simulating every token, using random walks through each supported schema
shape; completed walks must parse and conform.

The service compiles schemas on rank 0 and journals the specification.
Strict unsupported schemas receive a 400 error naming
`response_format.json_schema.schema.<path>`. Non-strict unsupported
schemas fall back to JSON-object mode with a warning. Each peer builds
the same machine from the journal and its tokenizer.

## 11. Runtime and API

The server separates HTTP processing from model execution. Rank 0 runs
the text frontend and generation service; every rank runs a scheduler and
model engine. The HTTP thread accepts and validates requests, renders
templates, tokenizes prompts and writes responses. The engine thread owns
admissions, model operations and token events. A mutex protects the event
queue and request records; sockets are written only by the HTTP thread.

### Scheduler

`src/sched/scheduler.*` depends on `SchedulerEngine`, which exposes
prefill, reservation, decode, close and cache operations without exposing
logits. The engine includes token selection in each model operation;
on the fabric, those picks and model reductions are collectives.

Each tick admits one queued request or a fitting group of cold prompts, then
decodes the next canonical slice of active requests. Admission chooses
the oldest request that fits the free pool and an available slot, allowing
smaller requests to pass a blocked larger one. A sustained stream of small
requests can therefore delay a large request.

Qwen's opt-in prefill budget advances one aligned chunk per tick before
decode. One unfinished prefill holds its slot, KV reservation and private
snapshot until completion or cancellation; its device positions are hidden
between chunks so padded decode graphs cannot mutate its state. The optional
idle budget permits larger chunks when no request is actively decoding,
checked after the cancellation sweep at each quantum. Both budgets ride the
settings and warm journal records. The cursor retains mandatory model and
snapshot cuts separately from the current budget grid, allowing later chunks
to grow without skipping a snapshot. Zero budgets retain full-prompt admission.

EOS, cancellation and token limits are applied to each returned token,
including the prefill pick. A multi-token result can retire a request
before all returned tokens are emitted; the unused suffix is discarded.
The engine and cache handle the corresponding state boundary. Slots are
reused after retirement. Tests compare independent and concurrent runs,
including cancellations, batching and reuse.

Scheduler decisions depend only on journaled inputs and deterministic
resource accounting. Wall clocks, thread arrival order and unordered
iteration must not influence collective order.

### Text API

`src/serve/generation_service.*` and `http_server.*` implement:

| route | behavior |
|---|---|
| `POST /v1/chat/completions` | Chat templates, streaming or one-shot responses, sampling, tools, supported structured output, stop strings and multiple choices |
| `POST /v1/completions` | Legacy string-prompt completion; its accepted fields are a subset of the chat route |
| `GET /v1/models`, `GET /v1/models/{id}` | Served model information |
| `GET /health` | Liveness probe |
| `GET /metrics`, `GET /v1/metrics` | JSON scheduler and service counters; both paths return the same format |

Chat messages support system, user, assistant and tool roles. The text
frontend handles the checkpoint's template, reasoning markers and tool
format. Unsupported input modalities and request fields are rejected
with errors naming the parameter. Supported schema constraints and
tool-choice rules are enforced through token masks as described in §10.

Sampling defaults come from `generation_config.json`, with configured
and per-request overrides. Parameters include temperature, top-p, top-k,
min-p, repetition/presence/frequency penalties, seed and logit bias.
Logprobs use the sampler's distribution; temperature zero reports the raw
distribution. A backend without the needed sampling or mask capability
rejects those requests.

Streaming responses begin with the role, emit content/reasoning/tool
deltas, then finish metadata and `[DONE]`. Incomplete UTF-8 tails are
held until a complete sequence can be written. Stop-string matching holds
a possible match suffix back from the client and journals retirement when
a stop is found. Multiple chat choices use separate scheduler requests
and can share a prompt through the prefix cache.

### Image inputs

`ModelFrontend::prepare_chat` returns prompt tokens and optional owned RGB
images with token spans. `supports_images` on both the frontend and engine
controls admission and the model endpoint's `input_modalities` field.
GLM's frontend decodes PNG/JPEG data URIs, applies bounded resize/padding and
expands image markers before running the checkpoint's normal chat template.

The GLM encoder loads the replicated BF16 `model.visual.*` tensors, validates
shapes, and includes their digest in the startup agreement. It runs patch
projection, 24 noncausal vision blocks with axial RoPE, spatial downsampling
and the merger. Startup memory planning includes its weights and maximum
workspace; all CUDA buffers are allocated before ranks begin collectives.
The arithmetic target is CUDA BF16 eager attention: FP32 RMS reductions,
unfused FP32 rotary products, BF16 QK/scaling boundaries and FP32 softmax.
All vision projections select FP32 reductions and BF16 output, with a fused
epilogue for bias. Final LayerNorm uses the CUDA reference’s Welford reduction
order. Attention
batches heads with values stored as `[heads, tokens, head_dim]`; query tiles
reuse the algorithm selected for the untiled matrix to avoid changing its
reduction order. Tiling stays within the reserved workspace. The independent
oracle gates full-depth relative RMS at 0.5% and cosine at 0.99998; see the
[numerical record](benchmarks/results/2026-09-18-glm-vision-numerics.md).

Image embeddings replace their placeholder rows in all four mHC streams
during prefill. The MTP prompt pass applies its embedding norm to the same
features at its shifted token positions. This handles image spans across
language-model prefill chunks. Eager and graph adapters share normal slot
opening and sampling; decode graphs consume ordinary generated token IDs.
Image admissions currently run individually and bypass prefix-cache lookup,
insertion and resumable prefill.

### Admission journal

`src/serve/fabric_serve.*` sends newline-framed JSON from rank 0 to
peers. A settings record supplies the model, version and shared engine
configuration before construction. The warm record checks effective
configuration and cache capacity and coordinates startup graph capture.
Each subsequent tick record carries accepted submissions and
cancellations, plus digests of the previous tick's operation stream and
prefix decisions. Peers apply the record before executing that tick.
Image submissions also carry bounded base64 RGB bytes, dimensions and prompt
spans. Decoding and resize happen once on rank 0; every rank validates the
journaled geometry and runs the same encoder input.

Only accepted scheduler changes are journaled. HTTP validation failures
stay on rank 0; token production and retirements are derived by each
scheduler. A peer rejecting an admitted request or deriving a different
digest exits with an error. The stop record is handled between ticks.

### Eager and graph engines

`src/engine/eager_engine.hpp` wraps model calls and pick callbacks.
`src/engine/graph_engine.hpp` records scalar graphs per request slot
and a family of batched variants. The graph engine uses the smallest
available batch covering the active slot IDs once the live-request count
reaches `graph_batch_min_live`; the default threshold is
`min(2, max_concurrency)`. Sparse occupancy can require a wider batch
than the live count alone suggests. Closed slots use inactive positions.

The request capacity is sixteen slots. Qwen supports up to 64 batched
decode rows, including C16/MTP3; GLM-4.7 retains its 32-row cap. MTP uses
`1 + depth` rows per request. Intermediate graph families cover
2/3/4/6/8/12 slots where they fit below the full family. Qwen uses
1024-token prefill chunks, also defining regular prefix-cache cuts.
Each graph
variant owns its bus generation cells and parity-specific buffers so a
shape switch preserves collective ordering.

Single-node graph mode uses resident weights and identity collectives.
Single-node mode without graphs uses the eager streaming path. Multi-rank
serving uses resident weights in either engine. Startup memory planning
checks the selected model, context, slots and cache arenas before loading.

Performance depends on model, occupancy, context and MTP acceptance.
Use [the benchmark tables](docs/benchmarks.md) for measured comparisons;
the graph and session tests check numerical behavior across these shapes.

### Shutdown, admission and failure handling

**Shutdown.** A signal sets a flag that the engine loop reads at a pass
boundary. `begin_shutdown()` rejects new requests, sheds the queue and
marks live requests for cancellation. The next pass journals those
cancellations before broadcasting the stop record. Interrupted streams
receive their completed tokens, a `server_shutdown` event and
`[DONE]`; interrupted one-shot requests receive 503. The HTTP server
drains its responses before stopping. Peers follow the journal to the
stop record, so bus teardown occurs between collective operations.

**Admission.** The default `full` policy reserves the prompt and maximum
completion at admission. The optional `grow` policy reserves the prompt
plus a completion window, then extends the reservation before decoding.
Growth uses transactional block allocation outside graph replay. If the
pool is exhausted, the scheduler ends the youngest request with a length
finish reason. Preemption with later recomputation is not implemented.

**Constrained output.** Tool calls, reasoning output, `response_format`
and supported typed arguments use the text frontend and decoding grammars
described in §10. Unsupported fields and schema constraints are rejected
by name where required; validation tests cover the accepted API behavior.

**Rank failure.** Journal socket closure detects a rank process exiting.
Rank 0 fails active streams with `engine_failure` and exits with status
2; a peer interrupted inside a tick exits with status 3. Silent node loss
relies on the bus watchdog. There is no mid-run failover or automatic
request recovery. The launcher can restart the configured world from its
resident images; clients must resubmit requests.

**Drift detection.** Each tick record contains rank 0's operation-stream
fold (`od`) and, with prefix caching, the cache-decision digest (`pd`).
Peers compare these against their state from the preceding tick. A
mismatch identifies the tick and ends the peer, causing rank 0 to fail the
service. Shutdown hashes provide a final comparison of the full streams.
See [operations](docs/operations.md) for failure drills and log collection.

## 12. Correctness and performance gates

Every model milestone has three tiers:

1. geometry/state tests: shapes, dtypes, ownership, pool boundaries, and
   transactional rollback;
2. numerical tests: layer dumps, logits, top-k indices, and end-to-end greedy
   transcripts against a pinned reference revision;
3. performance tests: kernel time, effective bytes/s, collective time, and
   critical-rank routing measured separately.

Numerical tests must include at least one **hand-computed expectation** for
every spec-critical output. Differential (implementation-vs-implementation)
tests verify consistency, not direction: an inverted selection ordering
passed every bitwise parity test in both directions because the host oracle
and the kernels were wrong together, and only a test with hand-computed
expected logits exposed it.

Device-vs-oracle divergences at selection boundaries are **certified, not
tolerated**: the audit re-derives the spec selection from the device's own
inputs (bitwise, tensor-core dots included) and measures the boundary gap
against the row's actual cross-implementation noise. This discipline caught
two reference bugs that plain tolerance would have absorbed into "FP noise"
— an out-of-bounds tail-seed read for decode batches shorter than kpool and
a gate-indexing typo — while the device path was correct both times. The
reference-dump parity runner carries the same audit for its corpora: a
flipped row is certified from the device's probes AND the dump's own
recorded tensors before it can pass. That guard earned its keep on the
first real-checkpoint run: the dump tool's torch backend paired decoded
q_fp8 with raw uint8 index_k bytes, and because e4m3 is monotone in the raw
byte within each sign class, the corrupted logits preserved near-correct
rankings — the failure was a single boundary swap that the structural
budget happily absorbed while the engine was right and the reference was
wrong. A near-miss bug that survives a tolerance is still a bug; only a
certification path can reject it. Layered the other way: a tolerance kept
as a fallback for rows the audit lacks inputs for must never veto a
certification — a certified near tie moves the row output O(1) by design
(near-tie scores mean the boundary is tied, not that the content is
similar), and the M4 close-out fixed exactly this inversion in the
chunked-prefill test, where a drift budget silently overroved the audit
(one certified 3.1x-noise near-tie at select_k=8 was rejected as
"flipped-row drift 1.0066" while the CI config sat one seed away from the
same failure at 0.6798).
Instrumentation (in-kernel clock64 phase timers, removed after use) is the
fastest path to a *mechanism* for a slow kernel; profile first, then fix the
measured bottleneck (the attention kernel's 85% bank-conflict share was
invisible from wall time alone).

`compute-sanitizer` (memcheck, racecheck, initcheck) is part of the gate, not
an optional extra — two classes of defect pass functional tests and only
surface under it:

- nvcc **speculates loads past short-circuit guards**: `i < n && a[i]`
  reads `a[i]` regardless of `i < n` once the branch is predicated, and even
  a ternary index gets both arms loaded. Per-token arrays in test drivers
  must be sized to the token count, and bounds-dependent loads must clamp
  their indices arithmetically;
- shared-memory buffers reused across reduction phases (all threads read the
  mean, lane-0s overwrite with the variance) race without a barrier between
  the read and the reuse — scheduling luck passes functional tests.

A third class, found by the M4 close-out's full-suite memcheck sweep: a CUDA
API call whose failure the code deliberately swallows and clears (the M1
rmsnorm's dynamic-smem opt-in probing the device cap, rejected on GB10 for
that kernel's footprint) is still a sanitizer finding — the tool reports the
error return regardless of the application's handling. Never make a call
that can fail: query the driver-computed per-kernel ceiling
(`cudaFuncAttributes::maxDynamicSharedSizeBytes`) and request within it.

Graph dependency cycles need protocol tests and traces as well as memory
sanitizers. In-process ranks share a copy-engine queue, so a waiting copy
or reset can block work needed by another rank's collective. Capture-time
node validation excludes that class of operation; model-declared host
callbacks are subject to their own bounded allowance. The trace and
reproducer are in [the graph-stall investigation](docs/batched_mtp_graph_stall.md).

Sanitizer scope is chosen per phase, deliberately: full-suite memcheck for
orchestration code (pointer/size bugs — it caught two undersized test
buffers that made a graph test pass vacuously), targeted racecheck/initcheck
for new kernel shapes. The loopback worlds under memcheck need their time
budgets lifted (`DGPP_TEST_BUS_TIMEOUT_MS`, `DGPP_TEST_CONSUMER_DEADLINE_S`,
`DGPP_TEST_WAIT_TIMEOUT_MS` — the engine watchdog, the kernel deadline and
every host-side wait: reducers, picks, replay finishes) — instrumented
kernels outlast the release budgets by orders of magnitude, on the baseline
as on any change. Instrumenting vendor kernels (cuBLASLt's cutlass
implementations dominate long benchmarks) buys no coverage of our code and
costs orders of magnitude in wall time; the decision and reasoning are
recorded with the results.

Measured numbers include the exact command, binary revision, driver, firmware,
clock/power context, run count, and variability. `micro_gemm_peak` is a
cuBLASLt heuristic sweep, not a proof of hardware peak. Network claims use
perftest as the authoritative throughput source and the project benchmark for
protocol validation.

**Cross-build numerics gates (since the reassociation rule, §1):** the
transcript md5 is the regression signal only for changes that claim to be
bit-identical (and for MTP against plain, §9). For rounding-level changes
the judges are `scripts/fabric_xcript.py REF NEW` — the first divergent
token, judged by the global top-2 margin in bf16 ulps of the winning logit
(a flip within 1–2 ulps is a coin the hidden state's last rounding turned;
the reference transcript has 14 of 300 picks decided by ≤ 1 ulp) — and
`scripts/fabric_logprob.py NEW REF` over a teacher-forced run
(`glm_gen_check --teacher-file`): per-token log p from the joined
vocabulary slices, perplexity, the mean-NLL delta with a standard error
and a PASS/FAIL at 0.02 nat. Three texts ship (556 tokens of prose; 7,331
of unmemorized technical prose at perplexity ~10.8 — the sensitive one;
6,549 of Conan Doyle at 1.04). The same binary twice must show a delta of
exactly 0: the decode path is deterministic end to end. Fabric runs are
correlated across ranks by BUS GENERATION, never by wall clock (the boxes
disagree by hours; `scripts/fabric_xrank.py`).

## 13. Repository layout

| directory | responsibility |
|---|---|
| `apps/` | Server, model checks, platform probes and transport reproducers |
| `src/engine/` | Session interfaces, eager/graph adapters, speculative decoding, prefix arenas and memory plans |
| `src/models/glm/`, `src/models/qwen/`, `src/models/glm4/` | Model configuration, binding, loading, layers, forward passes and sessions |
| `src/models/` | Shared KDA/DSA state and reference implementations, quantized matrix types |
| `src/kernels/` | CUDA attention, GEMM/GEMV, MoE, sampling, state and prefetch kernels |
| `src/loaders/` | Checkpoint parsing, slicing, weight construction and resident images |
| `src/net/` | TCP, roster, verbs and CollectiveBus |
| `src/serve/`, `src/sched/`, `src/sample/`, `src/text/` | HTTP and journal, scheduling and cache index, sampling math, tokenizer/templates/grammars |
| `src/common/`, `src/core/` | Logging, data types, memory helpers, arenas, streams, graphs and tracing |
| `tests/` | Unit, host, CUDA and Python tests |
| `tools/`, `scripts/` | Reference generators, checkpoint tools, deployment and measurement commands |
| `deploy/` | Example cluster configurations |
| `benchmarks/`, `docs/` | Workloads, probes, dated measurements and documentation |

## 14. Validation scope

The v1 sign-off includes 32K-context prefill measurements, process-failure
drills and a one-hour mixed-workload soak. Subsequent model studies record
their own fixture, checkpoint and fabric results. These are measurements
of specific revisions and configurations; wider deployment coverage and
longer reliability runs remain separate work.

If fabric runs show retries, drops or latency spikes, collect node and
switch counter deltas and inspect flow-control settings. Comparisons with
other inference engines are optional. The current work list is in
[docs/next_steps.md](docs/next_steps.md).

## 15. Primary references

- [NVIDIA DGX Spark platform specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [NVIDIA DGX Spark ConnectX-7 topology and interface mapping](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
- [NVIDIA DGX Spark CUDA porting notes, including GPUDirect RDMA](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html)
- [GLM-5 vLLM KDA state dtype and speculative state handling](https://github.com/ZJY0516/vllm/blob/glm-release/vllm/models/glm5next/nvidia/kda.py)
- [vLLM KDA state-shape and dtype calculator](https://github.com/ZJY0516/vllm/blob/glm-release/vllm/model_executor/layers/mamba/mamba_utils.py)
- [GLM-5 pooled-index and incomplete-tail cache](https://github.com/ZJY0516/vllm/blob/glm-release/vllm/models/glm5next/nvidia/attention.py)
- [vLLM pooled-index insertion, expansion, and quantized-cache semantics](https://github.com/ZJY0516/vllm/blob/glm-release/vllm/model_executor/layers/sparse_attn_indexer_kpool.py)
- [CUDA C++ memory model and thread scopes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-c-memory-model)
