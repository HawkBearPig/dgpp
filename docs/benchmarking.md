# DGPP Benchmarks

These procedures cover the optional platform, kernel and transport probes. Run
high-bandwidth tests in a maintenance window: they can consume both RoCE
lanes and more than 25 GB/s of shared memory bandwidth.

For serving throughput by model, world size, concurrency and prompt class,
see [the serving benchmarks](benchmarks.md); its "Workload and timing" section
is the method of record (what a deployment's row set contains, the validity
checks, how decode and prefill are reported), and
`scripts/serve_campaign.py` runs that whole set for one template in one command. Platform conclusions appear in
[measurements](measurements.md). Keep raw measurements, traces and run logs in
ignored local storage or external artifacts; publish concise findings and
validation limitations in the PR.

## Build

The `release` and `spark-cross` presets disable microbenchmarks. Enable them
explicitly for an optimized benchmark build; development presets enable them
by default. The release build preset selects only the server, so use the
build directory to build all targets or select a benchmark with `--target`.
Files under `benchmarks/` may be locally ignored. CMake skips a probe with
a status message when any of its sources are missing; available probes
remain buildable, and an absent benchmark directory does not block the project.

```bash
cmake --preset release -DDGPP_BUILD_BENCHMARKS=ON
cmake --build build-release -j
# Or build just one probe:
cmake --build build-release --target micro_mem_bw -j
```

Examples below use:

```bash
BUILD=build-release
PEER_F0=192.0.2.12
PEER_F1=198.51.100.12
DEV_F0=rocep1s0f0
DEV_F1=roceP2p1s0f0
GID_INDEX=3
```

Use the addresses and GID indices reported by `show_gids` on the target
cluster; do not copy these lab values blindly.

## Memory probes

`micro_mem_bw` reports streaming read, write, and copy rates. Its working-set
argument is MiB:

```bash
"$BUILD/micro_mem_bw" -s 4096
```

`micro_zerocopy` compares GPU access to `cudaMalloc` and `cudaHostAlloc`
memory, CPU-writer contention proxies, copy-engine staging, and the
CPU→GPU→CPU flag round trip:

```bash
"$BUILD/micro_zerocopy"
```

CPU writers touch separate buffers and are only memory-pressure proxies; the
benchmark intentionally avoids racing CPU writes with GPU reads of the same
object. For the production margin, repeat the pinned-read measurement while
real remote `ib_write_bw` streams target the node, using the transport commands below.

## GEMM heuristic sweep

```bash
"$BUILD/micro_gemm_peak"
```

The program first loads all SMs to avoid timing the transition from an idle
clock state. For every model shape and datatype, it then times every algorithm
returned by the current cuBLASLt heuristic query (up to 16), measures three
batches per algorithm, and reports the candidate with the fastest median.
Sample clocks/power externally when recording a result. Small weights may
remain L2-resident across iterations, so their effective GB/s is not a DRAM
result. This is a best-of-heuristics baseline, not proof of silicon peak.

## Packed full-GLM prefill benchmark

`packq_prefill_bench` compares the packed GEMV and tensor-core paths at
identical shapes and synthetic weights. Run on idle hardware:

```bash
# TP4 routed int4 gate (256 experts, top-k 8, BF16 output):
"$BUILD/packq_prefill_bench" --m 2048
# TP4 routed down (FP32 output):
"$BUILD/packq_prefill_bench" --m 2048 --n 6144 --k 512 --out f32
# TP4 fused int8 attention projection:
"$BUILD/packq_prefill_bench" --m 2048 --n 2624 --k 6144 \
  --bits 8 --experts 1 --top-k 1
```

Six CUDA-event trials alternate A/B order, with three warmups per side and
`--iters` timed launches (default five). Uniform routing is the default;
`--distribution hot` sends every token to the same selected experts.
Outputs must pass an L2 comparison; independent FP64 oracles live in
`packq_gemm_test`. The benchmark calls GEMM explicitly even for short rows;
automatic serving dispatch retains GEMV below 128 tokens. Kernel times
exclude routing, gathering and service overhead.

## Qwen QSA prefill and C1 timing

`qsa_prefill_bench` compares the original listed-attention kernel with the
prefill kernel using wider KV sharing and cooperative probability evaluation.
It alternates six A/B trials and requires bitwise equality of every FP32
partial. Run it on idle hardware:

```bash
"$BUILD/qsa_prefill_bench" --rows 256 --context 8192
"$BUILD/qsa_prefill_bench" --rows 2048 --context 32768
"$BUILD/qsa_prefill_bench" --rows 256 --context 8192 --heads 6 --graph
```

The default geometry is TP2: twelve query heads and one KV head. The
benchmark invokes both kernels explicitly; serving retains the old kernel
for decode, verification and prefills below 128 rows.

Against an otherwise idle server, `scripts/serve_c1_probe.py HOST PORT
--repeat 5 --json-out c1-engine.json` brackets each request with scheduler
counters. Its engine rate excludes the first token produced by prefill and
includes CPU/GPU/communication time inside step calls. It is separate from
client-visible output timing. Restart deployments and reverse A/B order
when investigating small differences.

## Qwen expert prefill experiments

`moe_prefill_bench` compares the production FP8 expert kernel with three
experimental variants linked only into this benchmark: `small` (16 × 64
tiles), `compact` (64-row down-projection blocks) and `persistent` (a fixed
number of down-projection blocks). None cleared the service performance gate.

```bash
"$BUILD/moe_prefill_bench" --variant small --distribution uniform
"$BUILD/moe_prefill_bench" --variant compact --distribution hot
"$BUILD/moe_prefill_bench" --variant persistent \
  --segments-file /path/to/segments.txt
```

Run on idle hardware. Defaults reproduce TP2 geometry: 256 tokens, 512
experts, top-k 10, hidden width 2,560, intermediate width 320 and scale block
64. Use `--inter 160 --scale-block 32` for TP4. `--distribution skewed`
adds another synthetic case. Recorded files contain one count per expert;
they replay routing sizes with synthetic tensors. Six trials alternate A/B
order, time repeated launches with CUDA events and require bitwise output
equality after each pair. `candidate_selected=no` identifies unchanged-path
controls, including the compact and persistent gate projections.

## KDA operator benchmark

```bash
"$BUILD/kda_bench" [--iters N] [--warmup N]
```

Measures the M2 KDA operators at real geometry on an idle node: the
recurrent-kernel-only decode state traffic across all 34 layers (TP=1 and
the TP=4 rank share), and the full layer's decode step and 2048-token
prefill chunk. Reports effective bytes/s against the 230 GB/s planning
floor. The decode recurrence is latency-bound at M2 (few blocks, 34
sequential launches); see [platform measurements](measurements.md) for the
numbers and optimization history. This is profiling
evidence for the M2 exit criterion, not a production throughput claim.

## DSA operator benchmark

```bash
"$BUILD/dsa_bench" [--iters N] [--warmup N] [--ctx N] [--decode-only]
```

Measures the M3 DSA layer at real geometry on an idle node: the full decode
step (eager and CUDA-graph replay) at `--ctx` context length, and prefill
chunks of 2,048/8,192 tokens. Reports effective bytes/s (weights + streamed
index cache + gathered latents) against the 230 GB/s planning floor. The
decode projection GEMMs stream 238 MiB of weights at the memory floor; the
remaining custom-kernel time and the profiled optimization log (native
hardware fp8/bf16 conversions, split-KV widening, bank-conflict-padded smem
strides, split-32 bitonic keys) are summarized in [platform measurements](measurements.md). Profiling evidence, not a
production throughput claim.

## KDA reference dumps (correctness, not bandwidth)

`tools/kda_reference_dump.py` generates parity inputs for the C++ layer:

```bash
# stdlib-only synthetic oracle (wired into CTest):
python3 tools/kda_reference_dump.py gen-pure --out FILE \
    --heads 2 --head-dim 32 --hidden 64 --tokens 8
"$BUILD/kda_test" --dump-file FILE

# self-contained format/determinism check:
python3 tools/kda_reference_dump.py selftest

# real checkpoint slices (requires torch; run where the checkpoint lives):
python3 tools/kda_reference_dump.py gen-torch --model-dir MODEL_DIR \
    --layer 0 --out FILE --tokens 32
"$BUILD/kda_test" --dump-file FILE
```

`gen-torch` reads safetensors shards with the stdlib (no safetensors
dependency) and runs the reference layer equations in torch; record its
invocation and the parity result when run against the pinned revision.

## DSA reference dumps (correctness, not bandwidth)

`tools/dsa_reference_dump.py` generates parity inputs for the C++ layer
(same `DGPP*AD` container, `DGPPDSAD` magic; the python fp8 codec is
cross-checked bit-exact against the C++ encoder):

```bash
# stdlib-only synthetic oracle (wired into CTest):
python3 tools/dsa_reference_dump.py gen-pure --out FILE
"$BUILD/dsa_test" --dump-file FILE

# self-contained format/determinism/codec check:
python3 tools/dsa_reference_dump.py selftest

# real checkpoint slices (requires torch; run where the checkpoint lives):
python3 tools/dsa_reference_dump.py gen-torch --model-dir MODEL_DIR \
    --layer 3 --out FILE --tokens 32
"$BUILD/dsa_test" --dump-file FILE
```

`gen-torch` dequantizes FP8 block-scaled core weights to BF16 (M3 consumes
BF16; FP8-native GEMMs are M4 scope) and runs the pinned reference layer
equations in torch; record its invocation and the parity result when run
against the pinned revision.

## GLM reference dumps and the curated suite (correctness, not bandwidth)

`tools/glm_reference_dump.py` generates full-model parity inputs for the
assembled forward (`DGPPGLMD` container), and `glm_forward_check` consumes
them (DESIGN §7.5):

```bash
# stdlib-only full-stack oracle over a synthetic mini-checkpoint
# (wired into CTest: fixture -> generate -> test):
python3 tools/glm_reference_dump.py gen-pure --out FILE --tokens 16
"$BUILD/glm_forward_test" --dump-file FILE

python3 tools/glm_reference_dump.py selftest

# real checkpoint (requires torch; ~3.3 s/layer measured — 4-10 min/case
# at 45 layers, ~60-120 s/case at --layers 12):
python3 tools/glm_reference_dump.py gen-torch --model-dir MODEL_DIR \
    --out FILE --token-ids "$(cat ids)" --tokens 512 [--layers N]

"$BUILD/glm_forward_check" --config MODEL_DIR/config.json \
    --checkpoint-dir MODEL_DIR --suite FILE [--layers N]
```

The optional GLM reference suite (three real-prompt cases and one
engine-only trace case) compares ISOLATED per layer — every layer starts
from the reference trajectory — with kept-row l2 floors, per-flip route
certification (engine vs reference biased router scores, noise measured
per token over the unswapped experts), and near-tie-certified top-1
agreement. Supply a suite file with the token IDs and oracle dumps for
each case. Engine-only route traces (`--trace-ids-file`) feed
`tools/route_trace_traffic.py`. Run these on the box that holds the
checkpoint; dumps are tens of MB and are not stored in the repo.

## GLM sampling-width profile (correctness, not throughput)

M6 sizes its exact distributed nucleus-sampling candidate table from the three
teacher texts. Run `glm_gen_check` once per text with `--sampling-profile` and
fetch every rank's log, then analyze the directories together:

```bash
scripts/fabric_run.sh --fetch-logs \
    --stage-file /path/to/teacher.txt --log-dir /tmp/mass-quick -- \
    --model unsloth/GLM-5.3-Flash-FP8 --text "Encyclopedia article." \
    --teacher-file /path/to/teacher.txt --sampling-profile \
    --decode-graph
# Repeat with fixed hard and memorized teacher passages.
python3 scripts/fabric_sampling_profile.py \
    /tmp/mass-quick /tmp/mass-hard /tmp/mass-memorized
```

The probe computes exact global top-{32,64,128,256} probability mass at T=1;
the analyzer rejects missing or cross-rank-divergent positions and chooses the
smallest width with at most 1% fallback for top_p=0.95. Its extra eager
candidate/LSE collective is measurement overhead, not a serving benchmark.

## Direct-device registration probe

```bash
"$BUILD/micro_gdr_probe"
```

The result distinguishes `SUPPORTED`, `UNSUPPORTED`, and probe errors. On DGX
Spark, `UNSUPPORTED` is expected and returns success because the experiment
completed. Registered `cudaHostAlloc` memory is the zero-copy receive path;
the copy timings are diagnostic comparisons, not a prescribed fallback.

## RC SEND protocol benchmark

Copy the same freshly built binary to the peer. The binary is architecture and
runtime dependent; do not reuse an old copy after source or CUDA changes.

### Latency

Peer:

```bash
"$BUILD/micro_ibv_smoke" serve 4791 --once --dev "$DEV_F0" \
  --gid-index "$GID_INDEX"
```

Initiator:

```bash
"$BUILD/micro_ibv_smoke" ping --peer "$PEER_F0:4791" --iters 1000 \
  --dev "$DEV_F0" --gid-index "$GID_INDEX"
```

Every SEND is signaled and sourced from its registered MR. The reported
one-way value is half the application echo round trip and includes software
overhead; use `ib_send_lat` for the authoritative transport latency.

### Single-lane bandwidth

Peer:

```bash
"$BUILD/micro_ibv_smoke" serve 4792 --once --dev "$DEV_F0" \
  --gid-index "$GID_INDEX"
```

Initiator:

```bash
"$BUILD/micro_ibv_smoke" bw --peer "$PEER_F0:4792" \
  --msg-size 1048576 --window 64 --dev "$DEV_F0" \
  --gid-index "$GID_INDEX"
```

The project tool exercises signaled RC SEND and receive-ring behavior. Use
`ib_write_bw` for the link ceiling.

### Concurrent lanes

Start one peer responder per lane on distinct ports, then run:

```bash
"$BUILD/micro_ibv_smoke" bw \
  --peer "$PEER_F0:4793" --dev "$DEV_F0" \
  --peer "$PEER_F1:4794" --dev "$DEV_F1" \
  --msg-size 1048576 --window 64 --gid-index "$GID_INDEX"
```

Repeated `--peer` and `--dev` arguments map by position. Supplying multiple
peers with only one device intentionally creates multiple QPs on that one
device; it is not lane striping.

## NIC DMA → GPU visibility test

This is the deployment regression for the pinned receive contract. The peer
server allocates an ibverbs-registered `cudaHostAlloc` slab and launches a GPU
consumer. The initiator sends a changing 64-byte payload followed by a
64-byte sequence doorbell as two ordered RC SENDs. The GPU polls the doorbell
with a system-scope acquire, hashes the payload, and publishes an acknowledgement.

The regression command is `nic_regress` (M5): it runs this exact protocol over
every directed node pair on both lanes from one command, uses the deployment
GID selection (first routable RoCEv2 GID), and aggregates both endpoints'
views into one matrix and exit code. On every node:

```bash
"$BUILD/nic_regress" mesh --nodes "$NODE0,$NODE1,$NODE2,$NODE3" --port 29961
```

Single-pair debugging (`serve` on the receiver, `pair` on the sender) and the
CI selftest (`nic_regress selftest`, the full mesh machinery on one device)
round out the modes.

The underlying single-pair recipe, kept as the M0 measurement tool (note it
defaults to the link-local GID unless `--gid-index` selects the routable one):

Peer:

```bash
"$BUILD/micro_ibv_smoke" serve 4795 --once --dev "$DEV_F0" \
  --gid-index "$GID_INDEX"
```

Initiator:

```bash
"$BUILD/micro_ibv_smoke" verify --peer "$PEER_F0:4795" --iters 10000 \
  --dev "$DEV_F0" --gid-index "$GID_INDEX"
```

Run the regression after any CUDA, kernel, mlx5, NIC firmware, or topology
change. A pass is empirical validation for that exact stack; a NIC is not a
C++ atomic participant, so this hardware test cannot be replaced by the
host-only CUDA test.

## Authoritative perftest commands

Per-lane latency:

```bash
ib_send_lat -d "$DEV_F0" -x "$GID_INDEX" -F -n 10000 -s 2 "$PEER_F0"
```

Per-lane bandwidth:

```bash
ib_write_bw -d "$DEV_F0" -x "$GID_INDEX" -F -N -D 5 \
  -s 8388608 -q 4 -t 128 --report_gbits "$PEER_F0"
```

Run a matching server command without the peer argument first. For an
aggregate result, start the `f0` and `P2...f0` server/client pairs concurrently
on distinct TCP ports; adding their separately timed averages is valid only
when their measurement intervals overlap.

## Result-record requirements

Record at least:

- source commit or base commit plus dirty-tree identifier.
- node model, kernel, CUDA, driver, mlx5 firmware, and perftest versions.
- interface, GID, MTU, negotiated speed, and PCIe link state.
- exact command and direction.
- warmups, duration/iterations, all runs or variability—not only the best.
- concurrent workloads and whether a model server was resident.
- when diagnosing a congestion symptom, pause, discard, retry, and
  out-of-buffer counter deltas over the affected interval.
