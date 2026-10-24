# Validated platform measurements

Measurements recorded from 2026-08-27 through 2026-09-15. Compact serving
summaries use the newest applicable production-path result; the later dated
sections retain the A/B history. Each section identifies its workload and
environment; this report is maintained manually.
See [benchmark procedures](benchmarking.md) for current commands and
[checkpoint budgets](checkpoint_budget.md) for model storage accounting.

## Platform baseline environment (2026-08-27)

- source base: `e2db401` plus the audit-remediation working tree.
- two DGX Spark GB10 nodes, AArch64 Linux `6.17.0-1026-nvidia`.
- CUDA 13.0 compiler, SM 12.1, 48 SMs, 24 MiB L2.
- ConnectX-7 firmware `28.45.4028`, RoCEv2, GID index 3.
- perftest tool version 6.20.
- model server not resident during the 2026-08-27 remediation measurements.

DGX Spark has one coherent 128 GB LPDDR5x pool, not discrete GPU VRAM plus
host RAM. `cudaMalloc` and `cudaHostAlloc` draw from that pool. The words
“device” and “host-pinned” below identify mappings/access paths.

## Network topology

One ConnectX-7 NIC connects to the SoC through two independent PCIe Gen5 x4
links. The one cabled physical QSFP port exposes two active RoCE lanes:

| lane | RoCE device | PCIe function | state | reported line speed |
|---|---|---|---|---:|
| 0 | `rocep1s0f0` | `0000:01:00.0` | active | 200 Gb/s |
| 1 | `roceP2p1s0f0` | `0002:01:00.0` | active | 200 Gb/s |

The `f1` functions map to the uncabled second QSFP port and are down. The two
active lanes share one physical 200 Gb/s port and are not independent failure
domains. This mapping agrees with the [NVIDIA ConnectX-7 networking guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html).

### Authoritative perftest results

`ib_write_bw`: 8 MiB messages, four QPs, depth 128, duration mode, relaxed
ordering on, RC/RoCEv2/MTU 4096.

| run | lane 0 | lane 1 | aggregate |
|---|---:|---:|---:|
| isolated | 107.64 Gb/s | 107.64 Gb/s | — |
| concurrent | 97.98 Gb/s | 98.05 Gb/s | **196.03 Gb/s (24.50 GB/s)** |

The old 105 Gb/s number was a valid one-lane result but not a node ceiling.
Bulk collectives must stripe both lanes.

`ib_send_lat`: 2-byte messages, 10,000 iterations.

| lane | min | typical | average | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|
| 0 | 2.66 µs | 2.72 µs | 2.72 µs | 2.80 µs | 2.98 µs |
| 1 | 2.40 µs | 2.46 µs | 2.57 µs | 2.97 µs | 3.04 µs |

Both lanes passed `ping -M do -s 8972`, validating 9000-byte IP frames.
All three other cluster nodes answered on both fabric subnets during the
reachability matrix.

### Flow-control state

Node-visible state on both active interfaces:

- Ethernet pause RX/TX enabled.
- DCB priority PFC disabled for priorities 0–7.
- `rx_out_of_buffer=0` at inspection.
- cumulative priority-0 discards are nonzero.

The switch PFC/ECN/global-pause policy is not available through node sudo.
The paired tests reached the physical-link ceiling without a congestion
symptom, so switch access is not a sign-off requirement. If a future four-node
run shows drops, retries, unstable throughput, or latency spikes, collect
before/after pause, discard, retry, and out-of-buffer counters and inspect the
switch configuration. No switch policy is inferred here.

## Project RC protocol measurements

The audit found unsignaled sends mislabeled as signaled and CQ polls writing
multiple completions into one `ibv_wc`. `micro_ibv_smoke` now:

- marks the required SENDs `IBV_SEND_SIGNALED`.
- supplies completion arrays matching every poll count.
- propagates link-thread, timeout, and sink failures.
- validates request sizes and socket failures.
- frees QP, CQ, MR, PD, context, and buffer resources after a connection.
- maps repeated peer/device arguments by position for two-lane tests.
- offers `--once` for bounded automation.

Cross-node results:

| test | result |
|---|---:|
| 1,000-iteration lane-0 SEND echo | 3.25 µs minimum, 4.17 µs mean one-way |
| 1 MiB signaled SEND, window 64 | 95.6 Gb/s |
| same SEND test on both lanes | **184.0 Gb/s aggregate** |

The SEND tool intentionally exercises receive WQEs/CQEs and signals every
message, so perftest RDMA write—not this result—is the link ceiling.
An earlier run before the final error-path hardening measured 3.29/3.47 µs
minimum/mean and 185.0 Gb/s dual-lane; both runs are retained in the dated
record rather than selecting only the faster result.

## NIC DMA → GPU payload visibility

The new `micro_ibv_smoke verify` mode registers a `cudaHostAlloc` receive slab
with ibverbs and launches a GPU consumer. For each iteration, the peer sends a
changing 64-byte payload followed by a 64-byte sequence doorbell as ordered RC
SENDs. The GPU observes the NIC-written doorbell with a system-scope acquire,
hashes the payload, and publishes a system-scope release acknowledgement.

| lane | iterations | mismatches | timeouts |
|---|---:|---:|---:|
| 0 | 10,000 | 0 | 0 |
| 1 | 10,000 | 0 | 0 |

The final hardened binary then passed another 1,000/1,000 payloads on each
lane. The larger runs validate the visibility contract; the bounded reruns
validate that later argument, CQ-error, timeout, and cleanup changes did not
regress it.

This is empirical validation of the exact kernel, driver, and firmware stack;
NIC DMA is outside the C++ abstract-machine synchronization model. The test is
a deployment regression after CUDA, kernel, mlx5, firmware, or topology
changes. CPU CQ polling remains in the production design for errors and slot
credits even though the GPU consumes the payload directly.

## Unified-memory and zero-copy results

Final `micro_zerocopy` regression run, 256 MiB buffers and best of its grid
sweep:

| pattern | GB/s |
|---|---:|
| GPU read, `cudaMalloc` | 240.8 |
| GPU read, `cudaHostAlloc` | **252.5** |
| GPU write, pinned | 210.8 |
| pinned → device copy | 59.6 |
| device → pinned copy | 59.6 |
| one-thread CPU memcpy → pinned | 27.4 |
| flag round trip | 1.02 µs min, 1.12 µs p50, 1.41 µs p99 |

The previous `volatile bool` CPU-thread stop flag was a data race; it is now a
`std::atomic<bool>`. Doorbell slots are explicitly initialized, and the CUDA
protocol uses system-scope atomics.

CPU-writer proxy cases touch separate buffers; the old same-buffer CPU/GPU
race was removed. It was not used for the production margin.

### Real incoming-RDMA contention

In a paired earlier run with a 251.5 GB/s idle pinned baseline, two remote
`ib_write_bw` streams each sustained 98.04 Gb/s into this node:

| pattern | GB/s | change from idle |
|---|---:|---:|
| GPU read, pinned | **218.4** | −13.2% |
| GPU read, device mapping | 208.6 | −13.1% |
| flag p50 / p99 | 1.12 / 1.42 µs | effectively stable |

Thus registered pinned receive slabs remain faster than a staging copy, but
dual-lane saturation consumes measurable shared-memory bandwidth. Models that
assume sustained dual-lane ingress provisionally derate simultaneous GPU
memory bandwidth by 15%. This is a stress-case planning value, not a fixed
inference penalty or an upper bound for other traffic patterns; M5 replaces it
with measurements of the real collective schedule. CPU-writer patterns remain
useful stress proxies but are no longer the evidence for NIC contention.

## Flag protocol regression

`src/kernels/flag_protocol.cuh` now provides:

- system-scope acquire/release operations for sequence words.
- payload/stamp writes ordered before acknowledgement publication.
- an inactivity deadline reset after every observed message.
- a reserved orderly-stop sequence.

CTest covers sequential acknowledgements, payload hashes, traffic continuing
for longer than the watchdog budget, quiet-time expiry, and subsequent device
use. The corrected flag-protocol test passes on GB10 in about 1.3 seconds.

## Checkpoint and traffic audit

The current generated report validates:

- 328.33 GB / 305.78 GiB, 62 shards, 76,108 tensors.
- 1.127 GB of vision tensors excluded from text decode.
- zero unmatched tensor names.
- 37,338/37,338 FP8 matrices paired with F32 inverse scales of exact
  128×128-block geometry.
- 22.415 GB/token of unique active base-text weights and 23.420 GB/token of
  aggregate physical traffic after replicated-module reads.
- TP=4 expected synchronized critical-path traffic 7.457 GB/token and a 32.42 ms
  weight-bandwidth floor at 230 GB/s.

The mean-rank value is 5.855 GB/token, but it is not used as a synchronized
decode prediction. Representative real router traces are an M4 exit gate.

## KDA operator state traffic (M2)

`kda_bench` (idle node, 50 timed iterations after 5 warmups; full record in
the 2026-08-27-kda-m2 measurements):

| case | mean | state bytes/step | effective |
|---|---:|---:|---:|
| recurrent kernel, decode, TP=1 (64 heads, 34 layers) | 3.301 ms | 272.0 MiB | 86.4 GB/s |
| recurrent kernel, decode, TP=4 rank (16 heads) | 1.123 ms | 68.0 MiB | 63.5 GB/s |
| full KDA layer, decode (T=1), TP=1 | 1.324 ms | — | — |
| full KDA layer, prefill chunk (T=2048), TP=1 | 23.787 ms | — | — |

The decode recurrence is latency-bound at these sizes (256/64 blocks on 48
SMs, 34 sequential per-layer launches), so the 230 GB/s planning floor does
not bind it; batching layers into single launches is the identified
optimization for M9. The full-layer decode step is weight-bandwidth
dominated (~0.9 ms of 1.32 ms streams the 204 MB fused projection), which
is the intended decode-traffic shape. These are correctness-first M2
numbers, not performance claims.

## Layer streaming and real route traffic (M4)

`glm_stream_check` over the full real checkpoint (full record in
the 2026-08-28-glm-m4-assembly measurements; idle node, cold page
cache):

- all 46 layers streamed one at a time through the resident loader:
  KDA/dense 408.3 MiB, DSA/MoE 7,275.7 MiB, KDA/MoE 7,204.2 MiB,
  MTP 7,338.3 MiB per layer; peak layer 7.17 GiB, globals 2.363 GiB
  (embed + lm head + final norm) — peak device footprint ≈ 9.6 GiB,
  so single-node correctness work fits without the TP placement.
- streaming total 316.4 s (per MoE layer ~7 s reading 7.2–7.3 GiB —
  ~1 GB/s from cold disk; loader overhead is not the bottleneck, and
  second passes drop once the page cache holds the shards). The loader's
  byte formula reconciles with the allocator on every load.

The first real route trace (272-token technical-prose prompt, deterministic
across re-runs) replaces the uniform-expert null model in the §3 traffic
numbers: busiest-rank experts/layer 3.541 (uniform 3.515), corrected
critical path 7.484 GB/token = 32.54 ms at the 230 GB/s planning floor
(uniform 7.457 GB/token, +0.4%). A 12-layer close-out re-run reproduces
the occupancy (+0.3%): the router input distribution is stable across
depths. Sampling other prompt classes is future work; the traffic tool
consumes their traces unchanged.

## Cost of one recorded collective node (2026-09-09, four nodes)

The Qwen plan's Q0 question (docs/qwen38_flash_next_plan.md D3): what does
one more all-reduce per residual site cost the decode step? Measured by
`glm_gen_check --gr-probe N` (one extra `[T x 10240]` bf16 all-reduce over a
scratch buffer after the attention fold of the first N layers, N = 32 within
the graph's 128-node budget) on `unsloth/GLM-5.3-Flash-FP8`, 300 steps,
transcripts identical with and without the probe
(`scripts/fabric_gr_probe.sh`, runs in `build-ci/fabric-runs/gr_probe_2026-09-09/`):

| step | base ms/step | +32 nodes | per node |
|---|---:|---:|---:|
| T=1 graph replay, TP=4 | 29.98 | 31.49 | **47 µs** |
| MTP graph replay (T=2 verify rows), TP=4 | 40.40 | 42.32 | **60 µs** |

The bus alone (`bus_check allreduce`, eager, 8 KiB, `scripts/fabric_bus_probe.sh`):
p50 **30 µs at world 2**, **39 µs at world 4** (min 21 / 33 µs). The graph
node's cost is the eager latency plus the replay's skew and fold; scaling
the in-graph number by the eager ratio puts the two-node cost near 36 µs
(T=1) and 46 µs (MTP). None of this is hidden behind weight streaming: the
probe nodes sit where a boundary does, and the step grew by the full amount.

## cuBLASLt best-of-heuristics results

`micro_gemm_peak` now warms all SMs before measurement, times every valid
heuristic returned by cuBLASLt (up to 16 requested), and selects the fastest
candidate by the median of three timed batches rather than timing the first
heuristic only. Two final process runs produced these ranges:

| model shape | FP8 | BF16 |
|---|---:|---:|
| attention output decode `1×4096×16384` | 231.8–238.3 µs | 578.3–581.4 µs |
| dense gate/up decode `1×24576×4096` | 406.9–419.2 µs | 832.4–835.7 µs |
| lm head `1×154880×4096` | 2,798.0–2,844.3 µs | 5,364.7–5,528.2 µs |
| attention output prefill, m=2048 | 195.7–200.9 TFLOP/s | 93.4–94.9 TFLOP/s |
| dense gate/up prefill, m=2048 | 220.8–220.9 TFLOP/s | 95.6–95.8 TFLOP/s |

Decode shapes whose weight set fits L2 can report implausible effective DRAM
bandwidth; those values are cache-warm diagnostics. This benchmark is a
cuBLASLt heuristic baseline, not a hardware peak claim.

## Direct-device MR probe

`micro_gdr_probe` reports direct `cudaMalloc` registration as unsupported on
this platform, consistent with NVIDIA's [DGX Spark CUDA porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html).
The binary now treats the expected unsupported result as a completed probe and
states the correct fallback: registered host memory consumed directly by the
GPU, not a pinned bounce copy.

## GLM-4.7 (nvidia/GLM-4.7-NVFP4) serving on four nodes (2026-09-10)

`scripts/fabric_glm4_serve.sh` with the
`deploy/cluster_glm-4.7_nvfp4_w4.example.json` template (MTP), and the same
template with `--knobs "--no-mtp"` (T=1), the resident image warm on every
rank, the 202,752-token pool (77.4 GiB per rank: weights 52.8, K/V 23.3).

| reading | value |
|---|---|
| boot from the resident image | 18–20 s (the first boot captures it: ~3 min); 14 graph variants warm-captured in 13 s |
| T=1 decode, one request | 49.0 ms/step (the bytes floor ~40 ms: 9.7 GB per rank per step, 6.3 GB of it the BF16 attention projections modelopt left unquantized; 186 collectives ~8 ms) |
| MTP decode, one request | 56–59 ms/pass at 1.84–1.97 tokens/pass, 28–32 ms/token |
| four live requests | 109.4 ms per eight-row step; aggregate 62.7–68.7 tok/s by class |
| prefill, short prompts (31–150 tokens) | 400–1000 ms per prompt, 5–13 ms/token (v1 warp-per-row attention; the tensor-core form is open) |
| MTP == T=1 transcripts | identical, 4 of 4 prompts; op streams identical across the four ranks |
| eval (thinking off, `serve_eval.py --no-think`) | gsm8k 60/60, HumanEval 39/40, schema extraction 30/30; no answer truncated |
| API check | every case (stop, n, logit_bias, usage, reasoning tokens) |

Two findings on the way: the partial RoPE is transformers' half-split
rotate_half (the interleaved reading passed every self-written gate and
looped the model after ~30 tokens; caught by `tools/glm4_torch_reference.py`
against transformers' own layer code on the real weights — layer-0 relative
l2 0.036 -> 0.0025), and the draft block takes the post-final-norm hidden
(11–46 % acceptance with the pre-norm residual, 79–98 % after).

### The full GLM-5.3 (HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64) serving on four nodes (2026-09-12)

`scripts/fabric_glm_dsa_serve.sh deploy/cluster_glm-5.3_int4-int8_w4_{mtp1,plain,mtp1_large-cache}.json`,
the 754B model at 99.30 GiB of int4/int8 g64 weights per rank (docs/glm53_plan.md G6;
the 2026-09-12-glm53-full measurements).

| reading | value |
|---|---|
| memory plan, MTP depth 1, 48K bf16 latent cache, 4 slots | 105.24 GiB per rank + 8 GiB headroom; T=1 103.58 GiB; sized to the fabric's smallest node (rank 2: 119.67 GiB visible, a firmware reservation 2 GiB larger than the other nodes') |
| boot | 30 s from the resident image (3.9 s model, 24 s of graph capture); the first boot, which captures the image on every rank, 405 s |
| T=1 decode, one request | 51.1 ms/step, 19.5 tok/s (the audit's bytes floor 44.5 ms: 10.2 GB per rank per step; 158 collectives) |
| MTP decode, one request | 61–66 ms/pass at 1.69–1.97 tokens/pass, 31–39 ms/token |
| four live requests | 159.0 ms per eight-row step; aggregate 42.3–47.0 tok/s by class |
| prefill through the service | 6.111 / 40.838 / 350.069 s at ~2K / 8K / 32K (2.963 / 4.959 / 10.613 ms per actual token) |
| MTP == T=1 transcripts | identical, 4 of 4 prompts; op streams identical across the four ranks at every shutdown |
| eval (thinking on at `reasoning_effort` low: the template has no switch) | gsm8k 59/60, HumanEval 40/40, schema extraction 30/30; no answer truncated (mean 92 / 142 / 58 completion tokens) |
| API check | every case |
| the draft's hidden convention | post-final-norm (plan D6), 77–97 % acceptance |
| the fp8 latent cache (`mtp1_large-cache`) | same pace (68–69 ms/pass, 74–98 % acceptance), transcripts diverge from the bf16 cache's after 97–464 characters |
| after node 2's SoC firmware update (same night) | the 64K bf16 template boots on the uniform fabric (106.68 GiB per rank, 34 s), transcripts identical to the 48K campaign's, the same pass times and acceptance |
| `engine.embed_sharding: vocab` (2026-09-13) | weights 97.97 GiB per rank (−1.33), transcripts identical 4/4, 67–68 ms/pass and 38.1 ms/token unchanged; worlds 2 and 4 and the recorded graph world bitwise the replicated ones |
| the peers headless (2026-09-13) | +0.2 GiB available on two nodes, ~0 on the ASUS: the desktop's pages were already reclaimable |
| headroom 4 GiB (2026-09-13) | a one-hour soak at the 120K MTP shape: 2,046 requests, 0 failed, memory flat on all four ranks, no allocation stall or direct reclaim, rank 0 ≥ 2.76 GiB available; templates 144K plain / 120K MTP / 208K fp8 |
| MTP depth 2, two slots (2026-09-13) | 86–89 ms/pass, 2.15–2.69 tok/pass (p1 73–95 %, p2 42–77 %): code/json −5–7 % per token, prose −4 %, math 0, chat +5 %; transcripts identical; depth 1 stays the default |
| MTP depth 3, two slots (2026-09-13) | 103–106 ms/pass, 2.37–3.22 tok/pass (p3 18–56 %): equal to depth 2 on code/json (32.9 / 33.0 ms/token), worse on prose/math/chat; the pass grows 17–19 ms per depth level |
| the residual ledger (2026-09-13) | per-phase `memory ledger` lines: the 2.7 GiB beyond the plan is 0.39 GiB of graph executables (14 variants, 7–15 KB per node), 0.7 GiB of host growth (capture bookkeeping, the service) and ~2 GiB of CUDA's own (cuBLASLt kernel selections, the context at first launches); stack frames ≤ 1.2 KB, the binary's device code 38 MiB — nothing the plan can shrink |

The first boot attempt at 64K templates was refused by rank 2 ("needs 106.68
GiB plus 8.00 GiB headroom but this node has 114.43 GiB free") and the other
ranks' lanes to it then failed with "transport retry counter exceeded" — a
peer exiting during the first collective looks like a fabric fault; the
memory plan of every rank is the first thing to read.

### The memory plan's headroom, measured (2026-09-12, late)

`kMemoryHeadroomBytes` (apps/dgpp_serve.cpp) covered what the plan does not
itemize; it was 8 GiB since the 262K freeze. Measured on the full GLM-5.3's
64K MTP world with 2 s samplers on all four ranks (node used = MemTotal −
MemAvailable, minus the idle baseline, minus the 106.68 GiB plan): +5.5–5.6
GiB on the peers and +6.0 on rank 0 at listening, during a 33.7K-token
prefill and under four live requests alike — a fixed cost, not a function
of the context or the load. Of it ~1 GiB is the process before the check
(outside the budget the check reads), 2.22 GiB the shared resident stream's
pinned staging mirror (the largest layer or the globals, kept for the
process's life), 2.4–2.8 GiB the rest (cuBLAS handles, the CUDA runtime's
2.3 GiB of shared mappings, the service's tables).

Changed: `release_sources()` on the shared resident stream frees the
staging mirror with the checkpoint mappings; the GLM-5.3 model materializes
its stack in its constructor and releases right there, before its caches
exist and before the bus has a collective in flight, so the plan counts the
staging only beyond what the caches replace. Qwen and GLM-4.7 still load
lazily and keep their mirror (an automatic release at the last layer was
tried and withdrawn: mid-forward, `cudaFreeHost` waits on the bus's
persistent kernels for a watchdog period — the loopback worlds deadlocked
for exactly 5 s); their plans now name the staging, which is conservative.
The headroom is 5 GiB. Verified: the same world settles 2.3–2.4
GiB lower, boots in 30.5 s, and the 160K fp8 template (plan 109.19 GiB)
boots on the four nodes with 3 GiB still free. Records:
`build-ci/fabric-runs/glm53_mem_2026-09-12/` and `glm53_mem_verify_2026-09-12/`.

### The full GLM-5.3: the bus fold and the L2 prefetch (2026-09-13)

Back-to-back rituals (the 2026-09-12-glm53-full measurements, the
last section). The graph all-reduce's shared-memory staging budget widened
48 → 80 KB so the depth-1 pass's two-row fold (72 KB from three peers)
stages like a one-row fold, plus unrolled system loads in the hash and
fold passes: depth 1 68.0 → 66.9 ms/pass, depth 2 88.2 → 85.1, T=1
51.3 unchanged, every transcript identical. The L2 weight prefetch off on
all four ranks: T=1 51.3 → 55.4 ms/step, depth 1 66.9 → 72.1 ms/pass —
it stays on.

### The full GLM-5.3: sixteen decode rows (2026-09-13)

The family's decode cap lifted from eight rows to sixteen (the select in
row groups of eight, sixteen pick request slots, the DSA layer's tiles from
the count, the decode path's projections pinned to the warp kernels). The
four-slot templates keep their launches: T=1 51.3 ms/step, depth 1 67.1
ms/pass, transcripts identical. Eight slots at depth 1 (a site config,
plan 110.42 GiB; the graph engine's new 4- and 6-slot families beside the
2-, 3- and every-slot ones): c=1 29 tok/s, c=4 41–42 (the four-slot
template's own 40–43), c=6 46–47, c=8 48–50 aggregate, at 67 / 170 / 220
/ 278 ms a step. Record: the 2026-09-12-glm53-full measurements.

### MTP depth 2 on GLM-4.7 (2026-09-10)

The depth-2 chain (the session core's `session_draft_chain` /
`session_graph_capture_draft_chain` on the hidden-window families,
`glm_spec_chain_row_window`) measured with the
`deploy/cluster_glm-4.7_nvfp4_w4.example.json` template and
`--mtp-depth 2`, against depth 1, one greedy request at a time. Transcripts
were identical to depth 1 on every short prompt (4/4):

| prompt (256 tokens) | depth 1: ms/pass, tok/pass, ms/token | depth 2: ms/pass, tok/pass, ms/token, p1 / p2 | tokens/s |
|---|---|---|---|
| chat | 61, 1.86, 32.5 | 72, 2.32, 31.2, 85 % / 48 % | +4 % |
| code | 61, 1.93, 33.0 | 72, 2.52, 28.7, 92 % / 60 % | +13 % |
| math | 62, 1.95, 31.3 | 72, 2.55, 28.4, 94 % / 62 % | +10 % |
| json | 61, 1.99, 31.1 | 73, 2.60, 28.0, 96 % / 65 % | +11 % |
| 6,525-token prompt, 300 tokens | 66, 1.86, 35.4 | 79, 2.35, 33.7, 85 % / 52 % | +5 % |

The chain row costs ~11 ms per pass (the second draft row's expert bytes
and its own draft-block run) and the second draft stands 48–65 % of the
time, so single-stream throughput gains 4–13 %. (At this reading the
engine built no row-batched graph past depth 1: two live requests ran
their scalar graphs back to back at 143–147 ms per pass each; the batched
chain followed the same day — the next section.) Prefill of the
6,525-token prompt: 11.5 s (1.77 ms/token) at both depths.
The other families after the chain landed (the session core is shared):
Qwen3.8-Flash-Next MTP 26.3–26.4 ms/step, transcripts identical to round 5;
GLM-5.3-Flash `fabric_glm_regression.sh` MTP 40.39 / T=1 29.94 ms/step,
prefill 548 / 1,412 / 6,284 ms, the same rank-consistency hashes as the
2026-09-09 baseline (`build-ci/fabric-runs/*_post_d2_2026-09-10`).

### Batched MTP depth 2 on GLM-4.7, on the runtime decode rows (2026-09-10)

The fixed decode batch's row ceiling is the recipe's shape since this
afternoon — `max_concurrency x (1 + mtp_depth)`, floored at 8 — so the
depth-2 mode (the base template with `--mtp-depth 2`, 4 slots) boots a 12-row
world (`serve: decode rows 12`; the bus's latency slot 120 KiB, 128
sampling candidates still fit) with the 2-, 3- and 4-slot batch families
at 6, 9 and 12 rows, each carrying every slot's chain row in one draft-
block run (`session_graph_capture_draft_chain_batch`). The GEMM interface
lowers every bf16 decode call up to those rows to the row-independent
GEMV core (`CublasLtGemm::set_decode_rows`), so a batched request's rows
keep the scalar reduction order — the first 9-row batch had fallen to a
cuBLASLt algorithm with its own order and flipped a near tie in the
engine gate. The reading, `scripts/fabric_glm4_load.sh` (distinct
long-answer prompts, thinking off, greedy, 320 tokens each; the client's
aggregate tokens/s over the phase's decode span, rank 0's stats lines
inside the phase):

| live requests | depth 1: ms/pass, tok/step/req, tok/s (per request) | depth 2 batched: ms/pass, tok/step/req, tok/s (per request) | tokens/s |
|---|---|---|---|
| 1 | 60, 1.89, 31.0 | 72, 2.31, 32.2 | +4 % |
| 2 | 79, 1.85–1.88, 44.5 (22.3) | 123–125, 2.3–2.6, 37.0 (18.5) | −17 % |
| 4 | 139–143, 1.74–1.93, 47.5 (11.9) | 198–204, 2.2–2.4, 42.8 (10.7) | −10 % |

The second draft stands 41–66 % of the time in the batch as it does
alone, but the pass grows faster than the tokens: by fixed rows the pass
is 72 ms at 3, 79 at 4 (one GEMV chunk), 123 at 6 and 140 at 8 (two
chunks), 200 at 12 (three) — each 4-row chunk past the first re-reads
the 6.3 GB of BF16 attention projections per rank (+45–60 ms), a row
within a chunk costs 7–8 ms (its K/V reads and its distinct experts). So
batched depth 2 is correct, isolated and slower than depth 1 under
concurrency on this family; the shipped template keeps depth 1. The lever for
concurrency at either depth is the attention projections' bytes per pass: wider GEMV
chunks (a 6-row chunk at K = 5120 is 61 KB of dynamic shared memory, an
8-row one 80 KB — one block per SM, to be measured), or a row-independent
tensor-core kernel for the bf16 projections that reads the weights once
for up to 16 rows; quantizing the projections halves the bytes but not
the passes (the fp8 / fp4 GEMV cores chunk the same way). Transcript
isolation: the engine gate pins the batched depth-2 rows bitwise to the
eager engine's, and on the fabric a prompt's greedy answer alone (the
scalar graph) is its answer beside three others (the 12-row batch) —
`serve_load.py --isolation`; the op streams are identical across the
four ranks in both worlds.

### The existing families after the GLM-4.7 work (2026-09-10)

`scripts/fabric_glm_regression.sh` on `unsloth/GLM-5.3-Flash-FP8` against
the 2026-09-09 baseline (`build-ci/fabric-runs/glm_baseline_2026-09-09`):
every reading's rank-consistency hash identical to the baseline's (the
transcripts bitwise unchanged); MTP chat 40.49 ms/step at 72.0 % acceptance
(baseline 40.35–40.50), T=1 29.97 ms/step mean, p50 30.0, p99 31.0 (29.81–
29.96 / 30.0 / 31.0), steady-state prefill 549 / 1417 / 6286 ms at 512 /
2048 / 8192 tokens (549 / 1405–1412 / 6258; a second sample 551 / 1428 /
6303 — the spread of the reading). `moe_slot_bench` at the GLM-5.3-Flash
per-rank shape: fp4 201 / 320 / 883 us at 1 / 2 / 8 rows (recorded 209 /
334 / 929), fp8 281–286 / 491 / 1489 (273 / 469 / 1420, the same decode
kernels — run-to-run spread on this box, the fabric step above is the
arbiter). `scripts/fabric_qwen_serve.sh` on `Qwen/Qwen3.8-Flash-Next-FP8`
against the round-5 transcripts of 2026-09-09: MTP world (`deploy/
cluster_qwen-3.8-flash-next_fp8_w4_mtp1.json`) 26–27 ms/pass at 1.5–2.0 tokens/pass, T=1 world
21.8–21.9 ms/step (recorded 22.0), transcripts identical 4 of 4 in both,
API checks clean, op streams identical across the ranks. The full ctest
after every change: 62 of 62.

## DeepSeek-V4.1-Flash (deepseek-ai/DeepSeek-V4.1-Flash) serving on four nodes (2026-09-14)

The MXFP4/FP8 checkpoint as shipped, world 4, the DSpark block draft
(`mtp_depth` 5, twelve decode rows at two slots), the decode graph, the
bounded prefill, 128K context. Memory plan 76.55 GiB + 4 GiB headroom
per rank (72.94 GiB resident weights, read from the NVMe in 137–246 s the
first time, 22 s from the captured images after). Numbers off the
service, `build-ci/fabric-runs/dsv41_serve_2026-09-13/` and the dated
the 2026-09-14-dsv41-flash measurements.

Single stream, greedy: 76 ms/pass, 2.33 tok/pass, 32.5 ms/token; ttft
404 ms on a 64-token prompt. DSpark acceptance per class (p1): json 92,
math 82, code 72, prose 67, chat 55 %; wall 20.7–38.2 ms/token. Bounded
prefill 2.35 / 1.85 / 1.61 ms/token at 512 / 2,048 / 8,192 tokens.
Concurrency (greedy): c=1 33.1, c=2 37.3 tok/s aggregate; sampled by
class up to 74 tok/s at c=2 on json. Quality (thinking off): gsm8k 60/60,
HumanEval 40/40, extract 30/30. Determinism: the plain T=1 world's greedy
transcripts byte-identical to the DSpark world's, `serve_load
--isolation` identical, the four ranks' op streams identical at the
take-down.

The T=1 step is 34 ms/pass (1.00 tok/pass). Against the ~16.7 ms/token
memory floor plus ~4 ms of collectives that is a ~1.6x ratio, the
optimization gate's starting point (§5 of the plan; the confidence-
scheduled rows are its first item now that acceptance is measured).

**The cell-method bench (2026-09-14, later; one prompt per class, a
cell at concurrency C sends C copies of it, temperature 0, thinking off;
the dated 2026-09-14 DeepSeek V4.1 measurements).** dgpp's depth-5
scheduled two-slot world: C1 47.5 / C2 64.3 aggregate, code 74.1, math
73.1, reasoning 58.1, prose 37.8 per stream. The six-slot world (depth 4,
scheduled) after the day's three kernel changes: C1 48.6, C2 66.2, C6
82.7 aggregate (43.2 / 53.0 / 58.9 in the morning), TTFT 0.31 / 0.48 /
1.26 s (0.55 / 0.85 / 2.15), cold prefill 1,286 / 1,315 tok/s at 2,950 /
11,592 tokens (568 / 595). The changes: the
streaming tensor-core decode GEMM (`kernels/mma_gemv`: the dense fp8
projections, the engram wkv, the draft's main_proj and the bf16 head read
their weights once per launch at 1–32 rows — the 30-row step 252 → 215
ms; then 64/128-row forms and 128-row groups for the prefill's rows: C6
71.7 → 77.9), and the MXFP4 form of the fp4 tensor-core expert kernel for
the prefill's experts (the weights once per 64-row tile instead of once
per 4 rows: C6 77.9 → 82.7, cold prefill 708 → 1,286 tok/s), and the
batched replay's own depth rule (the mean survival over the live slots
instead of the deepest slot's depth: C6 82.7 → 84.5 at λ 0.045, 86.9 at
λ 0.08 — the fixed point grows with concurrency), then the adaptive λ
(an EWMA of committed tokens over the modeled step time, floored at the
configured λ, replicated arithmetic: C1 48.2 / C2 66.8 / C6 87.0 with one
configuration, the default now), and the group prefill (queued cold
prompts within one window as the spans of one forward: C2 66.8 → 71.8, C6
87.0 → 103.6, C6 TTFT 1.23 → 0.90 s; a burst of six short prompts is one
~540 ms forward; then spans of any width — each span's decoder segment
packed — C6 106.4, TTFT 0.87 s). The six-slot world's day: C1 43.2 →
47.9, C2 53.0 → 71.6, C6 58.9 → 106.4 against the recipe's 37.95 / 64.30
/ 131.86; cold prefill 568 → 1,235 tok/s. The prefill's eager boundary
folds are the next cost: a 60-row fold is ~1.5 ms of wall (the one-block
staging copies and fold through pinned memory, the ranks' spread, and the
host noticing the finished kernel 0.2–0.9 ms late), 93 per pass — the
GPU-driven eager fold is the designed next step. The
six-stream gap to the recipe (1.7x) is the serialized per-prompt prefill
(six arrivals prefill one after another: TTFT and stalled decode; the
recipe batches them) and the 30-row step's composition (the MoE at its
union-of-experts traffic floor, 110 ms; the 30-row collectives' wire time,
33 ms; attention 17 ms).

The cross-check on the real weights (`tools/dsv41_torch_reference.py`
against `apps/dsv41_forward_check` dumps): the release's own layer code in
fp32 sits as far from the engine as from its own bf16 pipeline at every
layer (the plan's G4/G5 record; layers 0–3 and 20, 35- and 2,100-token
prompts).

**Optimization, the decode window attention (2026-09-14).** The nsys decode
profile showed the step is GPU-busy 97 % (not collective- or host-bound)
and the fp8/fp4 weight GEMVs run at 70–88 % of peak cold — near the
hardware limit. The largest non-GEMV cost was the window attention:
`attn_partial` at ~140 µs a layer versus ~12 µs for the compressed
`attn_flash`, because the window launched with `n_split=1` — a single
latency-bound thread block at decode while the compressed source split
across the SMs. Splitting the decode window a fixed eight ways (prefill
unchanged and byte-identical) cut it: chat 38.3 → 34.3, code 25.3 → 21.8
ms/token (~10–14 % on the attention-heavy classes), gsm8k 60/60 unchanged,
cross-rank op streams identical. Because softmax attention cannot be split
bitwise-invariantly (the online-softmax combine's per-split rescale rounds
differently, unlike the linear GEMVs), decode's window moved by about one
bf16 ULP and the greedy transcripts changed at near-ties with accuracy
unchanged. The remaining non-GEMV levers are the per-layer all-reduce
(~14 % of the step) and, for more attention speed without changing outputs,
fusing the window and compressed sources into one split flash.

**Optimization, the confidence-scheduled verify depth (2026-09-14, later;
plan §D8a, `engine.mtp_schedule`).** DSpark's confidence head emits a
per-position acceptance logit; a greedy step now verifies only the leading
drafts whose prefix-survival probability beats the value of one verify row
(`lambda × row_ms`, the Dinkelbach optimum of aggregate tokens per second),
each depth on its own captured graph variant, while the block still drafts
its full width — a draft not verified is decoded next step, so the
committed transcript is the plain greedy one. Three worlds from one binary
(`build-ci/fabric-runs/dsv41_sched_{base,on,lam045}_2026-09-14`, greedy,
300 tokens per class): baseline chat 33.4 / code 21.9 / prose 27.4 / json
21.9 / math 21.8 ms/token (its transcripts reproduced the window-split run's
4 of 4 — the default path is unchanged); scheduled at the reservation-rate
λ (0.028) 24.7 / 19.6 / 23.1 / 19.9 / 20.1; at λ 0.045 (the achieved
throughput, the fixed point) 24.6 / 18.9 / 22.7 / 19.1 / 19.8. Transcripts
identical 4 of 4 and the four ranks' op streams identical in every run.
The pass got cheaper (chat 71 → 46 ms) at the same tokens per pass (2.2 →
2.0): fewer rows, the same accepted prefix; the conditional per-position
acceptance rose to 74–87 %, so the head is calibrated. The scheduled path
gives up the pipelined launch-ahead (~0.5 ms/step), invisible in these
numbers. Later the same day the batched replay schedules too (one depth per
batch, the deepest a live slot asks for, on reduced-row batch variants over
the compacted feeds): at two live requests prose 34.1 → 51.1 and chat 36.6 →
54.0 tok/s aggregate, code 53.1 → 58.9, math and json flat; the batched
answer identical to the solo answer. Sampled requests still verify the
whole block.

**The scheduled verify depth on the other MTP families (2026-09-14).**
Without a confidence head the engine takes the draft head's own probability
of each draft from the device sampler (identical on every rank: it comes out
of the pick's fold). Measured at `mtp_depth` 2, exact in every run
(transcripts identical, isolation identical, op streams identical), and
throughput-neutral: GLM-4.7 on four nodes 28.4–32.7 ms/token either way
(the policy cut a third of the steps to one draft; the pass 72 → 69 ms at
2.30 → 2.18 tokens); GLM-5.3-Flash on two nodes, depth 1 the served
default at 61–79 ms/pass and 1.7–2.0 tok/pass, depth 2 28.6–35.5 ms/token
with or without the schedule (a quarter of the steps cut; 73 → 69 ms at
2.06 → 1.98 tok/pass on chat). The reason is structural: scheduling saves
rows at stake times their cost, and a depth-2 step has one row at stake
(~3 ms on GLM-4.7, ~12 ms on GLM-5.3-Flash) against the draft it may
forfeit, where DSpark's block has four. Depth 2 itself is worth it single-
stream on GLM-5.3-Flash (code 32.7 → 28.6, math 33.8 → 30.3 ms/token) and
not at two live requests, where its 4 slots × 3 rows exceed the family's
fixed 8-row batch and the steps fall back to scalar replays.

## The session-core families: the dense lowering by rows and the group prefill (2026-09-14)

The DeepSeek work's streaming tensor-core GEMM and group prefill, carried
to GLM-4.7, the full GLM-5.3, Qwen3.8-Flash-Next and GLM-5.3-Flash under
the rule that T = 1 must not regress and accuracy must hold.

### The bf16 interface: the chunks, cuBLASLt's algorithm, the streaming form

`bf16_gemv_test`'s timing table, cold weights (four copies rotated), µs per
product. `chunks` is the lowering before 2026-09-14 (every decode row count
through the 4-row GEMV chunks); `chunks≤4|Lt` the new one (`set_decode_rows`
at the dense_gemv_rows bound); `mma` the streaming tensor-core form at every
row count (DeepSeek's lowering).

| shape (GLM-4.7 attention, one rank of four) | m | chunks | chunks≤4 / Lt | mma |
|---|---|---|---|---|
| q_proj [12288 × 5120] | 1 | 535 | 535 | 544 |
| | 4 | 549 | 545 | 577 |
| | 6 | 1067 | 551 | 621 |
| | 8 | 1061 | 573 | 619 |
| | 12 | 1596 | 558 | 646 |
| | 16 | 2124 | 557 | 653 |
| | 32 | 4287 | 573 | 658 |
| | 128 | 600 (Lt) | 600 | 875 |
| | 2048 | 2813 (Lt) | 2734 | 14067 |
| k/v_proj [1024 × 5120] | 1 | 45.7 | 45.7 | 52.8 |
| | 8 | 61.0 | 50.2 | 55.0 |
| | 16 | 90.7 | 50.6 | 56.9 |
| o_proj [5120 × 12288] | 1 | 561 | 547 | 545 |
| | 4 | 1079 (two chunks: K fills the smem) | 1094 | 565 |
| | 8 | 2184 | 552 | 643 |
| | 16 | 4295 | 566 | 684 |

cuBLASLt's algorithm sits at the weight-stream floor from six rows on every
shape (551–573 µs for 126 MB: 220–228 GB/s); the streaming form trails it
by 10–20 % there and the chunks by 2–8×. So bf16 rows above four take the
algorithm, and the streaming form keeps only DeepSeek's every-row role
(its bitwise group prefill).

### The fp8 dense sites: the chunks against the streaming form

`mma_gemv_test`'s sweep on the full GLM-5.3 DSA shapes (one rank of four,
block scales 128 × 128), µs per product, the streaming form at the block
width its rule picks (eight warps from 4096 rows, four from 1024, two
under):

| shape | m = 1 chunks / mma | m = 16 chunks / mma |
|---|---|---|
| kv_a [576 × 6144] | 10.9 / 28.8 | 57.1 / 36.9 |
| q_a [2048 × 6144] | 52.4 / 53.6 | 180 / 59.8 |
| q_b [4096 × 2048] | 27.4 / 29.5 | 96.9 / 32.8 |
| [5120 × 5120] | 110 / 111 | 354 / 115 |
| o_proj [6144 × 4096] | 111 / 108 | 330 / 112 |
| [12288 × 5120] | 263 / 274 | 1110 / 284 |
| [32320 × 5120] | 687 / 731 | 2874 / 771 |

At one row the streaming form is at parity on the wide sites and behind on
the narrow ones (the 576-row site: a warp of eight rows per block leaves
72 warps for 48 SMs, issue-bound whatever the ring depth), so the chunks
keep rows one to four; from five rows the streaming form reads the weights
once (the chunks once per four rows). Above 256 rows the families' 128-row
dense kernel is ahead of the streaming form's 128-row groups (1060 against
1099 µs at 512 rows, 3479 against 4355 at 2048 on [5120 × 5120]) and keeps
the prefill chunks. The kernel's asynchronous weight ring (cp.async into a
per-warp ring of two to eight window slices, 24 KB per block) replaced the
register staging: the bf16 head [32320 × 5120] at one row 1441 µs against
the GEMV's 1438 (was 1470 against 1437), fp8 [5120 × 5120] 122 against 119.
A 48 KB ring measured no better at any width and cost the second block per
SM; 512-k bf16 windows and 128-k windows at three blocks per SM were both
slower (docs/deepseek_v41_flash_plan.md's kernel record).

### The gates

- GLM-4.7: a group prefill (23 + 17 rows) is bitwise the prefills alone on
  the fixture (cuBLASLt picked the same algorithm at 40 and at 23 / 17
  rows); the 8-step decode off the group's cache audits at 1.2e-7 relative
  l2 against the re-forward.
- The full GLM-5.3: the group's rows within 6.5e-3 / 0 relative l2 of the
  prefills alone, top-1 equal; the decode off the group's cache 5.3e-3 on
  the kept rows with one selection-flipped row (the fixture's DSA
  selections flip on any noise).
- Qwen3.8-Flash-Next: the group's rows bitwise the prefills alone on the
  fixture; the decode off the group's cache 1.1e-3 relative l2 against the
  re-forward, top-1 equal throughout.
- GLM-5.3-Flash: the group's rows within 1.8e-7 / 1.9e-7 relative l2 of the
  prefills alone (9 + 13 rows), top-1 equal; the four-step decode off the
  group's cache bitwise the solo decode.
- The engine gates' batched transcripts under the new lowering: GLM-4.7's
  MTP batched C differs from the eager engine at position 19 of 61
  (world-1 margin 0.31); the full GLM-5.3's sixteen-row batch differs on 6
  of 8 slots at positions 18–57 (margins 0.02–0.91) — the amplified
  random-weight fixtures, the same picture as world 2 against world 1 on
  them (agreeing prefixes 11–17). The scalar transcripts stay bitwise.

### Fabric A/B

Four nodes, `scripts/fabric_serve_load.sh` on each family's MTP config,
two boots per family — `DGPP_DENSE_GEMV_ROWS=256` (the old lowering: every
decode row count through the GEMV chunks) against the default 4 — the
greedy transcripts, the MTP acceptance per class (300 tokens) and the
concurrency probe (`serve_load`, greedy, 320 tokens, five classes at 1, 2
and 4 live requests). Aggregate tokens/s per class; the MTP line is the
per-class pass time and acceptance (identical between the boots wherever
the table says so: the scalar chain is untouched).

**GLM-4.7 NVFP4, MTP depth 1 (4 slots × 2 rows = 8 at c=4)** — the 8-row
step 129.3 → 109.4 ms (the stats line at the c=4 phase); the MTP classes
identical (chat 59 ms/pass 84 %, code 57 / 89 %, prose 56–57 / 89 %, json
57 / 97 %, math 57 / 94 %).

| class | c=1 old / new | c=2 old / new | c=4 old / new |
|---|---|---|---|
| prose | 31.7 / 31.4 | 45.3 / 45.2 | 51.4 / 62.7 |
| code | 31.9 / 31.8 | 47.3 / 47.2 | 53.5 / 65.6 |
| json | 33.4 / 33.3 | 48.1 / 48.0 | 55.3 / 68.7 |
| math | 32.5 / 32.6 | 46.7 / 46.6 | 52.8 / 65.3 |
| chat | 29.5 / 29.5 | 44.4 / 43.9 | 48.7 / 62.7 |

c=1 and c=2 (one and four rows: the chunks either way) at parity within
the run noise; c=4 +22–29 % (the eight-row step's attention projections
read once through cuBLASLt instead of twice through the chunks).

**The full GLM-5.3 (Int4-Int8Mix-RTN-g64), MTP depth 1 (8 rows at c=4)** —
the 8-row step 162.3 → 159.0 ms; the MTP pass times identical (chat 66,
code 62, prose 61, json 62, math 61 ms/pass); the per-class acceptance
moved with the transcripts (chat 77 → 69 %, prose 92 → 87 %, the others
within a point: the prompts' prefill rows take the streaming form on the
DSA projections from five rows, so a prompt's greedy path can turn at a
near tie — single 300-token samples, not an accuracy measure; the evals
below are).

| class | c=1 old / new | c=2 old / new | c=4 old / new |
|---|---|---|---|
| prose | 29.0 / 28.2 | 35.5 / 34.6 | 42.4 / 43.0 |
| code | 29.2 / 29.2 | 36.0 / 37.1 | 42.4 / 46.2 |
| json | 29.4 / 29.2 | 36.6 / 35.8 | 43.9 / 47.0 |
| math | 28.9 / 28.7 | 37.7 / 38.1 | 43.2 / 45.3 |
| chat | 26.2 / 25.4 | 34.9 / 35.2 | 41.2 / 42.3 |

c=1 and c=2 at parity within the run noise; c=4 +1–9 %: this family's
eight-row step is the int4 experts and the collectives, the DSA
projections (the sites that changed) a small share of it.

**Qwen3.8-Flash-Next FP8 at world 4, MTP depth 1 (8 rows at c=4)** — the
8-row step 44.6 → 34.2 ms; the MTP pass times identical (chat 25, the
others 24 ms/pass); the per-class acceptance within four points either way.

| class | c=1 old / new | c=2 old / new | c=4 old / new |
|---|---|---|---|
| prose | 66.8 / 67.2 | 103.1 / 101.4 | 125.5 / 142.1 |
| code | 74.2 / 75.3 | 113.3 / 114.0 | 140.2 / 158.8 |
| json | 78.3 / 77.7 | 119.7 / 120.7 | 149.6 / 167.3 |
| math | 74.0 / 74.2 | 116.5 / 117.4 | 142.2 / 159.9 |
| chat | 64.2 / 63.2 | 103.4 / 103.1 | 124.9 / 144.1 |

c=1 and c=2 at parity; c=4 +12–15 % (the BF16 dense stack's projections
through cuBLASLt at eight rows; the fused multi-problem launches keep the
one- to four-row steps).

**GLM-5.3-Flash NVFP4-FP8 at world 4, MTP depth 1 (8 rows at c=4)** — the
8-row step 79.5 → 69.8 ms; the MTP pass times identical (chat 34, the
others 32 ms/pass); the acceptance within three points.

| class | c=1 old / new | c=2 old / new | c=4 old / new |
|---|---|---|---|
| prose | 55.2 / 53.9 | 74.9 / 71.8 | 87.1 / 99.0 |
| code | 57.7 / 57.9 | 77.0 / 76.3 | 88.4 / 101.7 |
| json | 58.3 / 58.5 | 72.4 / 73.1 | 89.2 / 104.2 |
| math | 56.0 / 55.6 | 79.2 / 79.7 | 91.6 / 103.8 |
| chat | 50.8 / 50.3 | 72.1 / 72.9 | 83.4 / 94.5 |

c=1 and c=2 at parity within the run noise (prose −2 % at c=1, json +0.3 %,
the pass times identical); c=4 +13–15 % (the KDA and DSA bf16 projections
and the head through cuBLASLt at eight rows, the DSA fp8 projections and
the dense MLP through the streaming form).

**Accuracy under the new lowering** (GLM-4.7, the batched lowering
exercised at concurrency 4, thinking off): gsm8k 60/60, extraction 30/30 —
the 2026-09-10 baselines (60/60, 30/30); the op streams identical across
the four ranks. HumanEval was not run (it executes generated code; the
2026-09-10 baseline was 39/40 in an isolated environment).

**Across the four families:** the single-stream and two-stream numbers are
the old ones (the chain is unchanged there), the four-stream aggregate
+1–9 % (the full GLM-5.3: its step is the experts), +12–15 % (Qwen, the
Flash) and +22–29 % (GLM-4.7: its step was the BF16 attention re-reads).

## DeepSeek-V4.1-Flash: the GPU-driven eager fold (plan D9, 2026-09-14 late)

The six-slot config (`cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_mtp4_c6`),
a cell-method bench (prompt set v1: one prompt per class, C copies of it
at concurrency C; levels 1 and 6, the 2K prefill target), two boots the
same evening: the host-driven eager
fold (`DGPP_DSV41_EAGER_FOLD=1`) against the stream-ordered one (the
default since D9). Same binary, the bus timeline on for both.

| measure | host-driven | stream-ordered |
|---|---|---|
| C1 aggregate / per stream tok/s | 49.34 / 55.06 | 49.64 / 54.8 |
| C1 mean TTFT | 0.303 s | 0.269 s |
| C6 aggregate / per stream tok/s | 108.02 / 20.44 | 108.49 / 20.41 |
| C6 mean TTFT | 0.836 s | 0.727 s |
| prefill 2K (2,950 prompt tokens) | 1,370 tok/s, TTFT 2.153 s | 1,383 tok/s, TTFT 2.133 s |

The fold itself, rank 0's sampled decompositions (every 32nd prefill-class
fold at INFO), µs, by payload (elems = rows × 5,120): the host-driven
kernel's phases plus the host's notice of its finish, against the stream
kernel's phases (the engine's notice — `done → walked` — is off the model's
critical path now: the stream continues at the kernel's exit, settle waits
once per pass).

| rows | host-driven: kernel = copy + handshake + skew + fold; done→notice | stream: kernel = copy + handshake + skew + fold |
|---|---|---|
| 4–5 | 111–123 = 17–21 + 28–29 + 34–48 + 18–20; 731–1,343 | 54–70 = 4 + 20–23 + 20–29 + 11–16 |
| 30 | 431 = 100 + 105 + 125 + 93; 523 | 246 = 14 + 119 + 47 + 65 |
| 47 | 542 = 158 + 126 + 105 + 146; 807 | 386 = 24 + 152 + 108 + 103 |
| 65 | 749 = 215 + 232 + 96 + 199; 276 | 561 = 31 + 185 + 170 + 176 |
| 128 | 1,397 = 437 + 331 + 226 + 395; 1,082 | 933 = 61 + 415 + 187 + 269 |

What moved: the copy phase (one shared staging row instead of a row per
peer: 158 → 24 µs at 47 rows, 437 → 61 at 128), the fold (the in-place
kernel reads less), and the host's 0.3–1.3 ms notice per fold left the
critical path — TTFT −11 % at one stream and −13 % at six, the 2K prefill
+1 %; the aggregates unchanged (the folds' peers' arrival spread — the
handshake and skew phases, the ranks' compute imbalance — is the same on
both). The group prefill's remaining cost is its own compute: the design
note in docs/deepseek_v41_flash_plan.md (D10) puts the number on the next
step.

## Full GLM packed int4/int8 prefill (2026-09-15)

Tensor-core tiles replace the packed GEMV prefill chain from 128 rows while
preserving exact integer-code × BF16-scale weights. The matched four-node
service campaign uses eight slots, native MTP depth 1, BF16 KV and a 1 GiB
prefix cache. Each point is the median of three identical-hash cold prompts
across builds, with zero prefix-cache attachment.

| Prompt target | Baseline prefill | Packed GEMM prefill | Speedup | Baseline → new ms/token |
|---|---:|---:|---:|---:|
| 2,048 | 15.628 s | 6.111 s | 2.56× | 7.569 → 2.963 |
| 8,192 | 80.364 s | 40.838 s | 1.97× | 9.759 → 4.959 |
| 32,768 | 512.968 s | 350.069 s | 1.47× | 15.531 → 10.613 |

The short-prompt policy protects measured C1/MTP behavior. Numerical
validation retains the existing tolerances and adds an independent FP64
bulk-forward fixture plus real-model prefill likelihood scoring.
The matched concurrency-4 tasks preserve GSM8K 59/60 and extraction 30/30
with zero truncations and identical per-item correctness outcomes.

## Qwen QSA prefill (2026-09-15)

The current QSA prefill kernel shares each K/V tile across twelve query heads
at TP2 and computes probabilities cooperatively within a warp. Decode,
speculative verification and prefills below 128 rows retain their existing
dispatch.

Cold service medians on Qwen3.8-Flash-Next-FP8 at world 2, four slots and
monolithic admission:

| Prompt target | Current prefill | ms per actual token |
|---|---:|---:|
| 2,048 | 1.299 s | 0.637 |
| 8,192 | 5.108 s | 0.634 |
| 32,768 | 21.168 s | 0.656 |

All paired prompts, outputs, usage and finish reasons match the prior kernel.
The campaign preserved 59/60 GSM8K and 30/30 extraction, with identical text
and grading on all 90 paired responses.

## Build and test validation

The warning-as-error CI build consumes `DGPP_WERROR=ON`; the earlier unused
cache variable is fixed. CTest contains host, CUDA, synthetic-model, and Python
checkpoint-audit tests. ASan and UBSan are independent cache-variable presets.
The final validation results are listed in the dated run record.

Both CUDA milestones additionally hold their suites clean under
`compute-sanitizer`: the kernel phases ran full memcheck, racecheck, and
initcheck; the DSA layer phase runs full-suite memcheck (it caught two
undersized test buffers that made a graph test pass vacuously) plus targeted
racecheck/initcheck on the tests exercising new kernel shapes — the
real-geometry racecheck pass is deliberately skipped because its runtime is
dominated by vendor cutlass GEMM kernels covered by the kernel-phase tests
(reasoning recorded with the results). This is a standing gate for new
kernels, not a one-off: sanitizer rounds have surfaced a speculated
out-of-bounds load (short-circuit guards do not protect loads once nvcc
predicates the branch), shared-memory reuse races that functional tests
passed by scheduling luck, and the vacuous-test buffers above. Details are
pinned in `DESIGN.md` §12.

DSA layer timing lives in
the 2026-08-28-dsa-m3-layer measurements: decode 2.24 ms/layer at
65k context (114 GB/s effective; the projection GEMMs alone are ~1.0 ms,
i.e. at the 230 GB/s floor), prefill 6,443 tok/s at T=2048, and the full
profiled optimization log from the 13.26 ms starting point.

## Historical M1 observations

The base commit recorded synthetic `gpt_doll` graph/eager parity, stable graph
replay, and a ten-minute allocation-stability soak. Those observations apply
to the synthetic model only. They are not evidence for full GLM correctness or
performance and are not carried into a model throughput target.

## Further measurements

These results cover the workloads and revisions named in each section.
The serving tables in [benchmarks](benchmarks.md) identify gaps in model,
world-size and concurrency coverage. Collect node and switch counters
when a fabric run shows congestion symptoms.

Comparisons with other inference engines are optional and are not an
implementation or performance gate.
