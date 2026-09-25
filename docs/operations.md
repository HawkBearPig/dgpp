# Operating DGPP

DGPP runs one `dgpp-serve` process per node. Rank 0 serves HTTP and
coordinates requests through the admission journal; peers follow the same
scheduler operations. The examples below use GLM-5.3 on four Sparks.
Other model templates use the same launcher with their own memory and
engine settings. See [README](../README.md) for the configuration schema.

## MiMo native MTP and prefill options

MiMo defaults to upstream's recursive block-0 drafting. To use all three
checkpoint heads, set `DGPP_MIMO_NATIVE_MTP=1` on every rank and set
`engine.mtp: true`, `engine.mtp_depth: 3` in the deployment JSON. The loader
requires three checkpoint heads when native drafting is enabled; a model
without MTP remains valid. For example, using the normal site configuration:

```sh
DGPP_MIMO_NATIVE_MTP=1 DGPP_RESIDENT_CACHE=off \
  python3 scripts/dgpp-cluster up --config /path/to/mimo.json
```

The launcher forwards these `DGPP_` settings to peers. Verify the resolved
configuration and both rank startup logs. Restart all ranks to change them.
Unset `DGPP_MIMO_NATIVE_MTP` and set depth 1 to return to the default control.
`DGPP_RESIDENT_CACHE=off` disables resident image caching; `0` does not.
Loading extra heads changes the resident image layout and memory requirement.

Native MTP loads blocks 0/1/2 and keeps their independent paged K/V state,
plus backbone history and rollback buffers. Prefix snapshots include that
history, so the same prefix-cache byte budget retains fewer snapshots. The
2048-row native walk limit is an internal chunk size, not the context limit.
`engine.kv_capacity` is a shared pool across active requests. Our measured
C2 configuration used 131072 tokens total, BF16 K/V and a 1.5 GiB prefix
budget; it does not provide 128K simultaneously to each request.

Two independent, default-off prefill switches are available:
`DGPP_MIMO_PREFILL_LAST_HEAD=1` projects only the requested final prefill
rows to vocabulary logits, and `DGPP_MIMO_MTP_CACHE_ONLY=1` avoids unused
attention/MLP work during one-head history updates. Native MTP already uses
cache-only updates for history and unselected heads. All flags must agree
across ranks. The final-row projection changes GEMM shape; validation uses
numerical tolerances and top-1 checks rather than a universal bitwise claim.

Native MTP3 improves acceptance on the measured code, JSON and arithmetic
probes, but slows the prose probe relative to MTP1. Use the
[reproduction procedure](../benchmarks/mimo_upstream/README.md) to compare
representative traffic. [Native-head measurements](../benchmarks/results/2026-09-23-mimo-native-mtp.md)
and [prefill measurements](../benchmarks/results/2026-09-23-mimo-upstream-ports.md)
record the configurations and limitations separately.

## Qwen sparse prefill attention

Qwen QSA uses the tensor-core warp kernel for prefills of at least 128 rows
with supported head groups. To use the previous partial kernels for all
prefills, launch with `DGPP_QSA_WARP=0`. The cluster launcher forwards this
setting to every rank; restart the deployment to change it. Decode and
shorter prefills retain their existing kernels in both modes.

## Configure and start

For a new machine, follow [Getting started](getting-started.md),
including dependency installation. HTTP defaults to localhost; deployment
`http.bind_host` and `http.port` override `.env` defaults. The service has no
TLS or authentication; see [networking](networking.md) before exposing it.

Add the settings from [`.env.example`](../.env.example) to the repository's
`.env`, preserving any credentials already there. Set `DGPP_NODES` to the
space-separated node addresses in rank order and `DGPP_SSH_USER` to the
SSH login. Ports and common log, staging and release paths also live there.
Scripts read this file automatically; exported settings take precedence.
Use `DGPP_ENV_FILE` for a different site file.

Copy a model template from `deploy/` to a local deployment JSON, or pass an
example directly with `--config`. The JSON selects the model, `world_size`
and engine settings. A world takes the first `world_size` nodes from `.env`;
switching models does not require copying addresses between JSONs. Any rank
count the node list can staff is accepted. Whether a model runs at that count
is decided where the answer is known: the engine refuses a geometry that does
not divide by the world, and each rank refuses a memory plan that does not fit
its node.

| template | deployment |
|---|---|
| `cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json` | GLM-5.3-Flash NVFP4/FP8 hybrid, four nodes, MTP depth 1, bf16 latent cache, 768K context, an 8 GiB prefix arena |
| `cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json` | the same hybrid on two nodes, FP8 latent cache, 132K context on four request slots (160K with `--bf16-weights checkpoint --kv-capacity 163840`) |
| `cluster_qwen-3.8-flash-next_fp8_w{2,4}.example.json` | Qwen FP8 with MTP depth 1, four or two nodes |
| `cluster_qwen-3.8-flash-next_nvfp4_w{1,2}.example.json` | Qwen NVFP4 on one or two Sparks, MTP depth 1, the dense projections FP8 at load, a mapped n-gram table |
| `cluster_qwen-3.8-flash-next_nvfp4-radixark_w{1,2}.example.json` | RadixArk's Qwen NVFP4 on one or two Sparks, identical engine configuration to the NVIDIA release |
| `cluster_glm-4.7_nvfp4_w4.example.json` | GLM-4.7 NVFP4, four nodes, MTP depth 1 |
| `cluster_glm-5.3_int4-int8_w4.example.json` | the full GLM-5.3 (int4/int8 RTN), four nodes, MTP depth 1, eight request slots, 100K bf16 context (120K with `--bf16-weights checkpoint --kv-capacity 122880`), the embedding vocab-sharded |
| `cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json` | DeepSeek-V4.1-Flash as shipped, four nodes, six request slots, DSpark depth 4 with the scheduled verify depth, the bounded prefill, 128K context |
| `cluster_mimo-v2.6-flash_mxfp4-fp8_w{4,2}.example.json` | MiMo-V2.6-Flash as shipped (MXFP4 experts, fp8 dense, the BF16 matrices in their 12-bit form), four nodes with a 128K BF16 K/V pool or two nodes with a 256K K/V pool in the fp8 row form (`kv_dtype`), four request slots, MTP depth 1 |

One template per model, quant and world (2026-09-14): the modes a template
does not name are knobs — `--no-mtp` for the plain T=1 world, `--mtp-depth N`,
`--max-concurrency N`, `--kv-capacity N`, `--kv-dtype fp8`,
`--prefix-cache-gib X`, `--dense-weights checkpoint --fp8-head gemv`,
`--bf16-weights checkpoint` — appended with
`dgpp-cluster up --knobs "..."` (deploy/README.md lists the retired variants
and the knobs that reproduce them).

`bf16_weights` (every template sets it) keeps a lossless 12-bit form of the
BF16 matrices that decode streams: bit-identical results from 0.75 of those
bytes. `"bf12+bf16"` keeps both forms resident — about 0.75× those matrices in
additional memory (the startup memory plan lists it as "bf16 decode packing"),
prefill untouched. `"bf12"` keeps the 12-bit form alone: each matrix's BF16
bytes return to the node as its layer loads (the plan's weights line says
"packed bf16 matrices released", and the boot log's second `bf12:` line reports
what came back), the footprint drops BELOW the BF16-only one, and a prefill
GEMM expands what it reads into a small scratch ("bf16 prefill expansion
scratch") — about 10 ms per prefill chunk on four-node GLM-5.3-Flash, 20 on
two nodes. Use `"bf12"` where the context is bounded by the node's memory (the
two-node GLM-5.3-Flash and full GLM-5.3 templates do) and `"bf12+bf16"` where
there is room. It takes effect on GLM-5.3-Flash, GLM-4.7, the full GLM-5.3 and
Qwen3.8-Flash-Next's FP8 checkpoint (where both values keep both forms resident);
`--bf16-weights checkpoint` turns it off. Resident images are shared by all
three values: switching never rebuilds them.

`kv_dtype` affects only GLM-5.3's latent cache. Qwen and GLM-4.7 K/V
caches stay BF16. Qwen's `ngram_table` and `dense_weights` settings
control table residency and optional FP8 encoding of dense projections.
The Qwen NVFP4 templates, including YaRN, set `engine.fp8_head: "mma"`.
With `dense_weights: "fp8"`, this uses streaming MMA above the dense GEMV
threshold and within the configured decode capacity, including short prefills
in that interval. Matched one- and two-Spark teacher-forced checks passed;
see the [numerical results and scope](../benchmarks/results/2026-09-21-qwen-fp8-head-numerics.md).
The interface default remains `"gemv"` for configurations that omit the key.

The service rejects unsupported `fp8_head` values. MMA requires Qwen and FP8
dense weights.
The setting is distributed by rank 0 and included in the configuration digest,
so peers use the same dispatch. It changes floating-point accumulation order.
Use `--fp8-head gemv` to restore the previous head path. When restoring BF16
dense weights, pass `--dense-weights checkpoint --fp8-head gemv` together.
Plain FP8 prefills compute only their last vocabulary row, retaining the
kernel selected for the original chunk length. For a full-head comparison,
set `DGPP_PREFILL_HEAD_ALL_ROWS=1` in each rank process before startup.
BF16 heads, grouped prefills and streaming-MMA chunks within the decode
envelope continue to compute every row.
The earlier [throughput measurement and limits](../benchmarks/results/2026-09-20-qwen-fp8-head-e2e.md)
remain historical evidence.

Qwen NVFP4 expert prefills default to W4A4 when the checkpoint supplies
calibrated activation scales. Set `DGPP_MOE_W4A4=0` in every rank process
to retain W4A16, or `DGPP_MOE_W4A4=1` to also enable dynamic activation
scales for uncalibrated NVFP4 experts. `DGPP_MOE_W4A4_MIN_ROWS` defaults to
256 routed rows. Hidden and local expert widths must be multiples of 64
and at most 16384; other shapes and decode use W4A16. Memory plans include
the activation workspace (28.2 MiB per rank for standard Qwen chunks).

The [single-node guide](qwen38_single_spark.md) covers the one-Spark memory
plan, and the [two-node benchmark](../benchmarks/results/2026-09-16-qwen-nvfp4-w2.md)
records the resident-versus-mapped placement decision.

```bash
scripts/dgpp-cluster up --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json
scripts/dgpp-cluster status --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json
scripts/dgpp-cluster down --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json
```

`doctor` performs read-only preflight checks locally and over SSH; `up` runs
it automatically. Use `doctor --local-only` to inspect just rank 0.
`--skip-preflight` is available for diagnosis, but does not bypass process
ownership checks. `up` refuses a running deployment unless `--replace` is
explicitly requested. Stop jobs from older launchers with their original
launcher before upgrading: an unrecorded process is never adopted or killed.

`down` and `status` without `--config` (or with `--all`) check every
deployment recorded under `DGPP_LOG_DIR/deployments`, whatever deployment
file each was started from. `status` lists running deployments and any whose
state cannot be checked, with their ranks and the end of rank 0's log. When
all are confirmed stopped, it prints `no deployments running`. `status --all`
also lists stopped deployments as one line each with when and how rank 0's
log ended. `down` stops the ones that are running and names each one by model,
world, namespace and deployment file. Use this to stop whatever is holding
the ports before starting another deployment. `up` always starts one deployment:
without `--config` it uses `DGPP_CLUSTER_CONFIG` or the default deployment file.

Pass the same `--config FILE` to `up` and to a `down` or `status` aimed
at one deployment. `scripts/dgpp-cluster resolve --config FILE` prints the merged
runtime configuration without launching or contacting any node. At startup,
the launcher saves it as `<log_dir>/cluster.resolved.json` and stages the same
content to peers as `<stage_dir>/cluster.json`. Neither `.env` nor its tokens
are copied to peers. Direct `dgpp-serve --config` commands need this resolved
JSON, not the deployment template.

The launcher defaults to `build-release/dgpp-serve` when no installed release
or `DGPP_BUILD_DIR` override is selected. `--bin PATH` selects another binary;
use `--bin build-ci/dgpp-serve` to deploy a testing build with debug symbols.
`--log-dir DIR` overrides the log directory,
and `--knobs "FLAGS"` appends server flags after the file settings.
The compatibility wrapper `scripts/serve_run.sh` maps
`DGPP_SERVE_KNOBS` and `DGPP_SERVE_LOG` to those options.

For builds made on x86 Linux, follow [cross-compiling](cross-compiling.md)
to stage the ARM64 server and its CUDA libraries before transferring them to
the Spark. Cross-compilation does not launch or update a deployment.

Rank 0 reads the shared engine settings, applies flag overrides and sends
the result to peers before model construction. Peers use their files for
bootstrap addresses and local paths; they log differences from the
settings received from rank 0. Every rank then checks the effective
configuration digest. The journal's settings record also rejects mixed
binary versions.

For development runs, `up` stages the binary and config to peers. For an
installed release, it stages the config and runs each node's installed
binary. It starts rank 0, waits for its rendezvous listener, starts peers
through SSH and waits for `serve: listening`. That log line indicates
HTTP readiness. Warm GLM-FP8 resident images took 15–25 s to reach it in
the recorded deployment; the first checkpoint load took about 4.5 minutes.

Use a separate config for each checkpoint. For example, a site-local
`deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json` can select the composed
`HawkBearPig/GLM-5.3-Flash-NVFP4-FP8` checkpoint. Size its context and prefix
arena with `--memory-plan`; the smaller weight footprint does not imply
one fixed cache capacity for every deployment.

**The same hybrid on two nodes.** Each rank then holds 94.71 GiB of weights
instead of 50.74 GiB, which leaves about 12 GiB for the context once the
4 GiB headroom is reserved (8 until 2026-09-12, 5 for a day; the residual after the plan check measured flat at 2.4–2.8 GiB and a one-hour soak held at 4). At four request slots with MTP the draft block's
hidden cache is what binds, so `cluster_glm-5.3-flash_nvfp4-fp8_w2.json`
takes the latent cache to FP8 and settles at 163,840 tokens: 106.57 GiB
planned against the 115.1 to 115.6 GiB these nodes report at boot. The boot's
ceiling tracks that reading — 177,024 tokens at 115.06 GiB free, 190,976 at
115.57 — so the shipped capacity keeps about half a GiB of slack under the
worst of it rather than chasing the best. A node that really had 117 GiB free
would hold 212,992 at four slots and 327,680 at two, but no node here does, so
no template carries those numbers. Re-size from a `--memory-plan` run on the
node that will be rank 0, remembering that the boot reads about 0.6 GiB less
than that check does. The draft block's hidden cache is 8 KiB per token
per slot, so slots are the context lever: the two-slot shape (`--knobs
"--max-concurrency 2 --kv-capacity 262144 --prefix-cache-gib 2"` on the
two-node template) holds 262,144 tokens with a 2 GiB prefix arena, and one
slot would reach about 534,000. Decode costs what the doubled per-rank weight read
implies, near 1.75x the four-node pace; prefill costs about 1.45x
(benchmarks.md §3 to §6).

### Large document and agent requests

`http.max_body_bytes` caps each serialized HTTP request body. The default is
268435456 bytes (256 MiB), allowing large document prefills and conversation
histories with tool calls and JSON escaping. To allow up to 1 GiB, add this
top-level section to the deployment JSON:

```json
{
  "http": {
    "max_body_bytes": 1073741824
  }
}
```

Use a positive integer byte count. The binary flag
`--http-max-body-bytes 1073741824` overrides the file; the launcher accepts it
through `--knobs`. The startup log reports the effective byte limit. Changing
it requires restarting the server with the new configuration.

The limit does not allocate that much memory upfront. Upload buffers and JSON
parsing use host memory as requests arrive; larger simultaneous uploads can
therefore consume more host memory. Tokenized prompts and requested output
must still fit `engine.kv_capacity` under the configured admission policy.
There is no fixed conversion between serialized bytes and model tokens.

HTTP 413 means the request exceeded this byte limit before reaching the model;
the response names `http.max_body_bytes` and its value. Some agent clients,
including OpenCode, respond to 413 by compacting their conversation, which can
look like a much smaller context window. Adjust the byte cap for that case;
increasing the KV pool alone does not change it.

### GLM-5.3-Flash image requests

GLM-5.3-Flash checkpoints with compatible vision tensors enable image inputs
automatically. `/v1/models` reports `input_modalities: ["text", "image"]`.
Send PNG/JPEG/WebP data URIs in user `image_url` content parts; the
[image input guide](vision.md) gives a complete request and limits.

The startup plan reserves 1.05 GiB of vision weights and 0.33 GiB of workspace
per rank, including for text traffic. Account for this when sizing KV capacity.
Image requests reuse prefixes with matching processed pixels and geometry,
including generated continuations. With a configured prefill budget, the GLM
graph engine yields between image-prefill chunks so active decodes continue.
Image requests bypass grouped prefill. Before changing a serving deployment, run
`python3 scripts/vision_api_check.py --url http://127.0.0.1:18080` on idle test
hardware, then compare rank operation streams after shutdown.
For numerical validation, stop the serving world before running the CUDA
encoder oracle; its default full-depth gate is documented in the
[image guide](vision.md#deployment-and-validation).

## Stop and inspect

`down` sends SIGINT to rank 0. New requests receive 503
`server_shutdown`; queued and active requests are cancelled at the
next scheduler boundary. Streams receive a shutdown error, then the stop
record releases peers. A signal during prefill waits for the current scheduler
pass. An independent watchdog exits with status 2 if shutdown has not completed
within 30 s, including a stuck engine, HTTP loop or teardown. A second SIGINT
or SIGTERM exits immediately with status 2 on any rank. Forced exits skip CUDA
teardown and may truncate responses. The launcher waits up to 240 s for rank 0
before handling peers and collecting logs.

`down` checks that every recorded rank has exited, including after SIGKILL.
If a rank remains alive or cannot be checked (SSH failure, unreadable process
record, or a missing helper beside an existing record), shutdown exits nonzero
and names the rank as running or `UNKNOWN`. Reachable ranks are still cleaned
up when another peer is unreachable. `down --all` continues across deployments
and reports confirmed stops separately from failures; `status --all` also exits
nonzero when a deployment cannot be checked. Neither command treats an unknown
state as proof that the cluster is down.

Starting and stopping the same deployment share a launcher lock. A concurrent
launcher is refused with a retry message, and `up --replace` does not stage or
start a replacement if shutdown cannot be confirmed. These checks remain scoped
to recorded process identities; they never kill arbitrary processes by name.

Logs and operation streams are collected under `DGPP_LOG_DIR/deployments/ID`, using
names such as `serve_r0.log` and `serve_rank0.ops`. The launcher
prints one MD5 per rank's operation stream; all ranks must match.
Old peer operation streams are removed before a new development run so
they cannot be mistaken for that run's results.
The ID hashes the absolute deployment-file path, not its model name.
Staging is similarly isolated under `DGPP_STAGE_DIR/deployments/ID`.
`dgpp-cluster paths` prints both locations. An explicit `--log-dir` is used
as-is; pass the same override to `up`, `down` and `status` (the scan that a
bare `down` or `status` runs does not look there). Process records
include owner, boot identity and start time so PID reuse cannot authorize
cleanup of another process. Keep the deployment path/site settings stable
while it is running.

Host log lines include UTC timestamps to the millisecond. Device-side bus
stall diagnostics (`BKFIN`) use GPU printf and have no timestamp.
Correlate rank logs when investigating a failure, and retain the version
and effective `config:` lines with the run artifacts.

## Install, upgrade, roll back

`scripts/release.sh` builds the release preset and packs
`dist/dgpp-<version>.tar.zst` (README's "Release and install" has the
layout); `scripts/dgpp-cluster install TARBALL --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json` copies it to every node
in the config, unpacks it under `paths.release_dir` and verifies every
file against `MANIFEST.sha256`. Which release runs is named — the
config's `release` key or `up --release <version>` — and `up` then runs
`<release_dir>/dgpp-<version>/bin/dgpp-serve` on every rank, staging only
the config. An upgrade is `down`, `install`, then `up` naming the new
version; a rollback is `up` naming the previous one, which is still
installed. `dgpp-cluster releases` shows what each node has;
`dgpp-serve --version` prints a binary's version, and rank 0 puts its
version on the journal's settings record so a peer of another version
exits before its first tick rather than form a mixed world.

## What a node needs

- The checkpoint in the Hugging Face cache (`--model ORG/NAME` resolves it
  per node) and the resident image cache (~82 GiB per rank, built on the
  first boot; the two sections below have the memory and cache details
  and knobs).
- Serving runs as a normal user and does not need locked clocks. Each rank
  needs a sufficient memlock limit for RDMA registration; increasing the
  launching session or service's hard limit may require administrator setup.
  See [memory registration failures](networking.md#memory-registration-failures)
  for the systemd configuration and startup checks.
- The default peers' staging directory (`/tmp/bus4/deployments/ID`)
  lives in `/tmp`: a reboot empties it, and
  `dgpp-cluster up` recreates it. The log dir (`DGPP_LOG_DIR`,
  `~/dgpp/log`) persists.
- The journal star forms first: rank 0 listens on the journal at once and
  the peers connect to it (retrying within the rendezvous window) before
  any rank builds its model; the bus world forms after the builds, its
  connect retrying within the same 120 s window. `up` encodes that order:
  the head, then the peers as soon as the journal listens.

### Memory on a serving node

For prefix caching, size the snapshot arena and KV pool separately. Startup
and `--memory-plan` report snapshot slots, bytes per slot and actual arena
allocation. The [prefix-cache guide](prefix-cache.md) lists the current recipe
capacities and explains when a larger `prefix_cache_gib` helps. With
`nvme_cache.enabled`, entries the arena evicts are kept in a preallocated
slab on each node's NVMe and restored for later matching requests; startup
checks the slab's directory, free space and minimum size on every rank
(`--memory-plan` too) and refuses to boot when they do not fit — see
[the cold tier](prefix-cache.md#the-nvme-cold-tier).

Reserve enough node memory for weights, model state and runtime buffers.
The GLM-5.3-FP8 main stack alone uses about 82 GiB per rank at TP=4;
other checkpoints have different footprints. The serving applications
check their allocations before loading:

- before anything is allocated, every rank computes its **memory plan** —
  the resident weights, the KV cache pool, the draft block's per-position
  hidden cache, every activation and scratch buffer, the prefix cache
  arena and the engine's own buffers, from the same formulas the
  constructors use — logs it itemized, and refuses to boot (exit 1, the
  world never forms) when the plan plus 4 GiB of headroom exceeds the
  node's free memory. The refusal names the largest items and the largest
  `kv_capacity` the node would hold as configured. `dgpp-serve --config
  /path/to/cluster.resolved.json --rank R --memory-plan` runs the check alone and
  exits 0 or 1. A standalone run reads about 0.6 GiB more free memory than a
  boot does, because a booting rank locks its pinned memory before it plans;
  leave that much slack when sizing `kv_capacity` from one, or read the
  ceiling out of a refusal;
- the loader's own check that the resident footprint (+ 4 GiB headroom)
  fits the device's free memory remains as the second line and fails
  immediately with a clear message — never three minutes into a load;
- the loader reads each source tensor exactly once (prefetch, copy, drop),
  so the page cache stays under ~10 GB during the load and the box never
  reaches its memory watermark; the checkpoint's mmaps are released the
  moment the last layer is on the device (`GlmLayerStream::release_sources`);
- the process *tries* to lock its memory (`mlockall(MCL_CURRENT)`, before
  the model is constructed) as a safety net against swap-in faults in the
  decode loop. This is optional: with the one-pass loader a rank with the
  pin off measured identically (p99 46 ms, 0 stalls, no swap traffic over
  1000 steps). A finite `RLIMIT_MEMLOCK` is logged, not warned about;
  `DGPP_MLOCK=off` skips the attempt. RDMA registration still requires a
  sufficient memlock limit with this optional pin disabled.

**GLM-5.3-FP8 context memory.** With per-forward activations sized to the
prefill chunk (2,048 rows) rather than the context, the memory that grows
with `kv_capacity` is the KV cache pool (12 layers including the draft
block; 12.4 KiB per token in bf16, 6.2 KiB in fp8, 3.5 KiB in fp4, index
cache included) and the draft block's per-position hidden cache (8 KiB per
token per request slot: 32 KiB per token at `max_concurrency` 4), about
44 KiB per token all told at the production shape in bf16. On a 121 GB
node, available context also depends on draft weights, request slots,
cache format and arena size. Use the startup plan for the configured limit.

**Budgeted prefill** (`engine.prefill_budget_tokens`, `--prefill-budget-tokens`)
is supported on Qwen and GLM-5.3-Flash graph engines, including GLM image
requests. The default, -1, selects 256 tokens rounded down to the engine
alignment and capped by its prefill limit (at least one aligned unit).
An explicit zero preserves full-prompt admission within a scheduler pass;
model and snapshot boundaries still split the work. Qwen retains its internal
2,048-token limit and reports token progress after each completed chunk.
The resolved budget is logged at startup and carried in rank 0's warm record.
The four-rank GLM-5.3-Flash deployment explicitly selects 256-token busy and
2,048-token idle budgets.
A positive budget executes one aligned prefill chunk per tick, followed by
a decode pass for active requests. Try 256 or 512 tokens; the budget must
be a multiple of the snapshot alignment and fit the prefill scratch limit.
Smaller chunks trade prefill throughput and TTFT for shorter pauses in
other streams. Request reservations are held before the first yield, and
cancellation releases the unfinished slot and its prefix references.

`engine.prefill_idle_budget_tokens` (`--prefill-idle-budget-tokens`) optionally
uses larger chunks when no request is actively decoding. It must be at least
the enabled busy budget, use the same alignment and fit the prefill scratch
limit. Zero keeps the fixed-budget behavior. The scheduler checks for active
decode after cancellations and before each chunk, so a long prompt can speed
up after its decoding peer retires. Required model and snapshot cuts still
split chunks. Short cold prompts can group up to the selected budget.
Larger idle chunks also increase the maximum wait for cancellation or a newly
arriving request; they do not preempt a chunk already running.

One long prefill progresses at a time in arrival order. Short prompts can
still prefill together when the group fits the budget. Snapshots from an
unfinished prefill remain private until completion. The settings and warm
journal records carry both budgets; every rank follows the same token cuts.
`/v1/metrics` reports `prefilling`, `prefill_ms` (execution counted once),
and `prefill_request_ms` (summed request waits, including waits between
chunks). Logical and computed prompt-token counters advance with each chunk,
including cancelled partial work; their difference counts attached cache
tokens. Other model families retain full-prompt admission.

Client disconnects are counted when the HTTP connection closes; retirement
occurs at the next scheduler boundary. With full-prompt admission enabled,
that boundary may be minutes away. Automatic chunking bounds the work between
cancellation checks in tokens, not elapsed time. It interleaves existing decode
requests, but a second queued prompt still waits behind the unfinished prefill.
The `starting prefill` log precedes model work; `admitted to slot` marks its first
token, after prefill has completed.

For a suspected hang, retain both ranks' logs and op streams, the exact request
and whether it connects directly or through a proxy, and repeated `/metrics`
snapshots. Compare `prefill.requests[].processed_tokens`,
`scheduler.snapshot_age_ms` and `service.pending_cancellations`. Pool counters
are scheduler snapshots and can stay unchanged during synchronous prefill.
Capture all-thread host backtraces on every rank before stopping the world.
Bulk stall dumps name the active pool and report posted/staged/expected stripes
per peer and retired/posted TX pairs. Their gate counter counts retries; zero
is not evidence that the gate was never reached. A 500 ms stall dump is diagnostic,
not a timeout. Serving uses a 120 s bus completion watchdog and a 60 s reducer
wait; the latter now also covers stream completion after the GPU's done stamp.
The bus watchdog shares its progress thread and does not cover model CUDA waits
outside a collective. Rank 0 therefore also runs an independent **120-second
engine progress watchdog** once serving starts. It observes scheduler passes,
advancing prefill token positions and successful multi-rank collective completions
using atomics, without calling CUDA or taking
the service, metrics or logging locks. An in-flight pass that stops making
progress exits with status 2 without engine teardown; the closed journal makes
the peers exit too. Clients receive a closed connection on this emergency path.
No termination signal is needed. Journal broadcasts and
stats publication are inside the monitored work scope. Idle serving has no time
limit. A full-prompt prefill can exceed 120 seconds while its internal chunks
continue to advance; an individual chunk can also exceed the deadline while
its collectives complete. Submissions, repeated polling and failed collectives
do not reset the deadline. Single-rank execution still relies on completed
chunks and scheduler passes. Startup/model loading is outside this watchdog's scope.

This is a fatal backstop, not recovery of a failed CUDA context: an external
supervisor must restart the service. The separate 30-second shutdown deadline
still applies after a termination signal. Neither watchdog identifies why the
underlying operation stopped progressing; retain the incident evidence above.

Use `scripts/serve_prefill_interference.py HOST PORT --json-out RUN.json`
on an otherwise idle server to compare the longest client update pause
with the budget disabled and enabled. Keep the same tag and prompt size
in matched fresh-server runs so the long prompt has no cached prefix.

**The draft depth** (`engine.mtp_depth`, `--mtp-depth`, 1–3, with `mtp`)
is the number of draft tokens verified per decode step. Depth 1 is the
two-row step: the pending token and one draft through the main stack, the
draft block proposing the next draft. A deeper step feeds 1 + depth rows,
and the block proposes the later drafts by running one more row per draft
on its own output (the single block's recursion — its hidden for the row
after the last accepted one is its own previous output, not the main
stack's; the KV and hidden it writes past the counter are provisional and
the next real rows overwrite them). The verdict, the commit and the
rollback are the same machinery over T rows; the sampled verdict tests
the drafts in order (a stand moves to the next row, a reject ends the
step on the residual token, the last row reached is sampled plainly) and
a host fallback continues the chain exactly as the device would have.

Qwen graph serving supports `engine.max_concurrency: 16` with MTP enabled and `engine.mtp_depth: 3`, using up to 64 verification rows. Reserve sufficient KV and graph memory; the shipped recipes remain unchanged. Validate on idle hardware before deploying a new build to all ranks. Keep `engine.mtp_schedule: false` for C16/MTP3. Its sixteen slots and seven batch families use 46 graph variants per verification depth; scheduling needs at least two depths (92 variants), exceeding the limit of 64. Enabling it fails startup with the conflicting settings and a remedy: disable scheduling to retain C16/MTP3, or reduce concurrency.

Qwen's 17-64-token decode walks use kernel-only BF16 lowering,
including the BF16 sites retained by `dense_weights: "fp8"`. This avoids
cuBLASLt memset nodes at real tensor-parallel shard shapes. Decode walks
of at most 16 tokens and prefill retain their existing dispatch, including
MTP hidden projections with multiple matrix rows per token.

The 8- and 12-slot graph families are shared across models. Configurations
above eight slots may capture more variants and choose smaller padded batches.
Those variants also reduce the scheduling budget: DeepSeek at 11 slots and
depth 2 now needs 34 variants per depth, so two depths no longer fit the
64-variant limit. The shipped recipes stay at or below eight slots and retain
their existing families.

**The scheduled verify depth** (`engine.mtp_schedule`, `--mtp-schedule`;
needs `decode_graph`, `mtp` and a depth of at least 2 to matter) lets a
greedy request verify fewer rows than the whole draft block on a step
whose drafts are unlikely to stand. The confidence is DeepSeek-V4.1's
DSpark confidence head where there is one; every other MTP family
(GLM-5.3-Flash, Qwen3.8-Flash-Next, GLM-4.7, the full GLM-5.3) takes the
draft head's own probability of each draft from the device sampler (the
draft picks report logprobs; the draft token itself is unchanged), so it
needs `sampling_candidates` > 0. The head emits a per-position acceptance logit; the engine
verifies the leading drafts whose prefix-survival probability beats the
value of one verify row, `lambda × row_ms`, and stops at the first that
falls below (the survival is monotone, so the verified drafts are a
prefix). A draft not verified is decoded plainly next step, so the
committed transcript is the plain greedy one at every depth — the change
is the step's cost only. Each depth replays its own captured variant (the
bus's 64 graph variants bound them: two per slot per depth plus two per
batch family; a spread of depths that always keeps the full block is used
when the budget is short, and a policy depth rounds up to the next
variant). The batched replay takes one depth for its slots, the deepest
any of them asks for; a sampled request, and a batch holding one, verify
the whole block. The constants are the world's (every rank takes rank 0's, so every
rank derives the same depth from the replicated confidence):
`mtp_schedule_row_ms` (`--mtp-schedule-row-ms`, one verify row, default
8), `mtp_schedule_base_ms` (`--mtp-schedule-base-ms`, the step's fixed
cost with the draft, default 28), `mtp_schedule_lambda`
(`--mtp-schedule-lambda`, the value of decode time in tokens/ms; 0, the
default, is the reservation rate `1 / (base + row)` — a verify row is
taken only when it beats a plain step's rate), `mtp_schedule_min_depth`
(`--mtp-schedule-min-depth`, the fewest drafts a step verifies, default
1). The scheduled path gives up the pipelined replay's launch-ahead (the
graph to launch is not known until the previous replay's confidence is
published at its tail), so a stream that stays at the full block pays the
graph launch (~0.5 ms a step) for nothing; the win is on the streams the
policy shortens. Measured on four nodes (2026-09-14, greedy): every class
faster — chat 33.4 → 24.6, prose 27.4 → 22.7, code 21.9 → 18.9, json 21.9 →
19.1, math 21.8 → 19.8 ms/token at `mtp_schedule_lambda` 0.045 (the
achieved throughput, the optimum; the reservation-rate default is 2–4 %
behind), transcripts identical; at two live requests prose 34 → 51 and chat
37 → 54 tok/s aggregate. On the families without a confidence head at
`mtp_depth` 2 (GLM-4.7, GLM-5.3-Flash) it is exact and throughput-neutral:
one row is at stake per step, so leave it off there unless a deeper draft
is served. Set `mtp_schedule_row_ms` to the family's measured extra row
(DeepSeek 8, GLM-5.3-Flash 12, GLM-4.7 3 ms) and `mtp_schedule_lambda` to
its achieved tokens per millisecond. The engine logs the replays per depth
at shutdown.
Measured on the fabric (2026-09-06, one greedy request, 300 tokens): the
step is 31.5 ms plain, 42–43 ms at depth 1, 54–56 ms at depth 2 — each
verify row is its own expert bytes (~10 ms), the chained block row
~2.5 ms — and the second draft stood 45 % of the time on prose, 56 % on
JSON, 65 % on code (the first: 78–87 %), for 2.21 / 2.41 / 2.51 tokens
per step against 1.80 / 1.81 / 1.88 at depth 1. So depth 2 is 41.7 vs
43.5 tok/s on prose (−4 %), 44.7 vs 43.1 on JSON and 46.5 vs 44.7 on
code (+4 %), and −4 % on an 8K-context summary; it pays when p1·p2
exceeds ~0.28·(1 + p1), about p2 > 0.63 at p1 0.8. Depth 1 stays the
default; the stats line's per-position acceptance says what a workload
would get. On GLM-5.3-Flash and Qwen3.8-Flash-Next every step past depth
1 is a scalar replay (their row batches carry the two-row step only), so
`max_concurrency` above one serves requests round-robin per step; GLM-4.7
batches depth 2 (the decode rows are the recipe's shape, `serve: decode
rows 12 (4 slot(s) x 3 row(s) ...)` in the boot log) but loses to depth 1
under concurrency there — each 4-row GEMV chunk re-reads its BF16 attention
projections (docs/measurements.md) — so its recipe keeps depth 1.
Changing the depth changes the config digest; the transcript does not
change (a greedy request decodes the plain transcript at any depth, a
sampled one the same distribution — the loopback gates pin both).

**The KV cache's dtype** (`engine.kv_dtype`, `--kv-dtype`) chooses the
latent cache's storage format: `bf16` keeps the rows as the projection
left them (every parity gate's format); `fp8` stores e4m3 codes with one
fp32 scale per row (~2^-4 relative error per element); `fp4` stores e2m1
codes in blocks of 16 with an e4m3 scale per block over the row scale
(~2^-2 per element). The attention kernels dequantize a tile into bf16
shared memory as they gather it, so everything past the load is the bf16
kernel; the index cache, the tail rings and the selection are unchanged in
every format, so a quantized cache changes the attention values, never
which tokens are attended. The format is part of the world's settings (the
head pushes it, the config digest carries it) and every rank runs the same
one. The quantized formats are a memory trade an operator makes
deliberately: at 262k tokens they save 1.5 GiB (fp8) or 2.2 GiB (fp4) per
rank against a 98 GiB plan, and the model's answers change with them.

**The DeepSeek-V4.1 prefill mode** (`engine.prefill`, `--prefill
bounded|exact`, default `bounded`) applies to the `deepseek_v41` family
only. `bounded` is the model's own serving recipe (its tech report's
"Decoder SWA Bounded Replay"): the twenty encoder layers run over every
prompt row, layer 20's compressor and index keys are published for every
row (the decoder's global KV), and the twenty decoder layers run over the
prompt's last 128 rows only, their window floored at that segment's start
— about half the prefill work of the exact walk. Across the chunks of one
prefill call the encoder output of the last 128 rows carries over to the
last chunk; a prefix-cache snapshot position closes such a span so the
saved state is complete, and a resumed prefill's segment sees the rows
before it through the ring. `exact` runs all forty layers over every row
(every parity gate's mode; `docs/deepseek_v41_flash_plan.md` §1.8 and the
G5 record). The mode is part of the world's settings (the head pushes it,
the config digest carries it); decode is the same in both.

Nothing else on the node needs setting. In particular a locked GPU clock
(`nvidia-smi -lgc`) is **not** required: the governor sits at 2400-2560 MHz
throughout decode on its own and the measured step distribution is the same
locked or unlocked. NTP between nodes only matters for reading logs side by
side, and `scripts/fabric_run.sh --node-probe` records each node's clock
offset per run so even that works without it.

### The resident image cache

The first start of a resident rank builds its layers from the checkpoint
(slice, stage, dequantize, pack) and writes the finished device bytes to
`~/.cache/dgpp/resident/<key>.img` on that node (~82 GiB per rank for GLM;
the key covers the checkpoint's shard headers, `config.json`, world, rank,
head sharding and the loader's format version, so a stale image can never
load by accident). Every later start streams that image instead with
O_DIRECT reads at the drive's line rate — 15-25 s to a ready model against
~4.5 minutes from the checkpoint. The boot digest is cached beside it
(`<key>.digest`). Knobs:

The first boot after a change that invalidates the images (a loader
format version bump or a new checkpoint) rebuilds every rank's image from the
checkpoint, and the ranks finish at different times (rank 0 in ~100 s,
the peers in ~155 s on the hybrid). `dgpp-serve`'s first collective has a
~57 s deadline, so that boot can fail with `boundary reduce ... collective
consumer exited on deadline` while the peers are still building — their
images are still written. Boot again (the images restore in ~15 s), or
run `glm_gen_check` once on the fabric first, whose rendezvous waits.


```
DGPP_RESIDENT_CACHE=off            disable (always build from the checkpoint)
DGPP_RESIDENT_CACHE_DIR=/path      put the images somewhere else
DGPP_RESIDENT_CACHE_VERIFY=1       re-fold every blob on read (a pass over 82 GiB)
```

Delete the file to force a rebuild; the loader's log line says how many
layers were restored versus captured on each start.

`scripts/fabric_run.sh --node-probe` samples each node's reclaim/swap/GPU
counters at 1 Hz for the run; `scripts/fabric_xrank.py LOGDIR` reads the
fetched logs and reports host gaps, stall windows, and step distributions per
rank.

## When a rank dies

There is no failover: any rank's death fails the service, quickly and
loudly, and the world is restarted (v1's failure semantics; built and
drilled 2026-09-05).

- **A peer dies.** Rank 0's journal watch sees the peer's connection close
  within ~100 ms (a peer never writes on the journal, so a readable
  connection is a close or a reset) and fails the service: every live
  stream gets the tokens the service had committed, then an
  `engine_failure` error event naming the dead rank and `[DONE]`;
  one-shots and later requests get 503 `engine_failure`; `/health` turns
  503 with the reason; rank 0 writes its op stream and exits with
  **status 2** once the answers are out (3 s grace). The other peers see
  rank 0's journal close — inside a tick, through their own watch — and
  exit with **status 3**. Measured on the fabric: rank 0 out 0.6 s after
  the kill, every rank gone within 3.2 s.
- **Rank 0 dies.** Clients see their connections close (nobody is left to
  write an event). The peers exit through the journal EOF (between ticks)
  or their in-tick watch (status 3) — every rank gone within 2.6 s in the
  drill.
- **A rank goes silent** (a node powered off: TCP does not close). The bus
  watchdog fails the in-flight collective after its deadline and throws
  into the same failure path.
- **Committed state is never touched.** The step in flight never completes
  on any rank, so nothing after the last completed step is committed
  anywhere, and every token a client received was committed on every
  rank. After a restart, the same prompt at temperature 0 reproduces the
  committed tokens as a prefix of its answer (the drill checks exactly
  this).

Stop the deployment with `scripts/dgpp-cluster down --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json`,
then restart with `scripts/dgpp-cluster up --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json`.
Cleanup only targets recorded processes belonging to that deployment; it
does not sweep arbitrary server processes. No boot-time service units are
installed, so an operator or external supervisor must start the service.

The drill: `scripts/serve_failure_drill.sh <victim rank> [clients]` boots
the world, streams from three clients, kills the victim with `kill -9` at
a random moment, and checks every claim above, then restarts and replays
the prompts. Its artifacts land under `build-ci/fabric-runs/failure_drill_*`.

## Checking that the world is healthy

- **At a glance, every 10 s:** each rank's log carries one aggregate line
  per interval while the world is busy (and one closing line of zeros
  when it goes quiet):

  ```
  stats: rank 0 | 10.0 s | decode 30.9 tok/s, 31.4 ms/tok, 54.6 ms/step (178 steps / 309 tok, 97 % of wall) | mtp 1.74 tok/step/req, accept p1 74 % | prefill 1 prompt / 4515 tok, 305 tok/s, 2.36 ms/tok, 10637 ms avg (72 % of wall), 13236 tok cached (1/1 hit) | live 1, queued 0 | pool 498/3072 blocks (16 %) | prefix cache 17/43 entries | requests +1 (shed 0, cancelled 0)
  ```

  Decode comes first. `tok/s` is what the interval delivered (idle time
  included); `ms/tok` is the pace while decoding (step time over the
  tokens generated); `ms/step` is the decode pass's time as the engine
  saw it. Under pipelined replay this is the interval between verdicts.
  The `mtp` group reports tokens per request-step (1 to 1 + depth) and
  acceptance by draft position. For example, `accept p1 74 % p2 61 %`
  reports first-draft acceptance and second-draft acceptance after the
  first was accepted. Depth 1 reports only `p1`.

  Prefill's tokens are the ones computed (an attach skips the rest,
  `cached`), and the two `% of wall` shares say where the engine thread's
  time went — together they approach 100 % when it is saturated. A peer's
  line carries the same counts with its own timings and no request
  counts. Every retire line carries the request's own numbers —
  `retired (eos): 327 tok in 35.7 s — prefill 7995 tok (0 cached) in 5662
  ms; decode 326 tok / 183 passes in 30.0 s: 10.9 tok/s, 92.1 ms/tok, 164
  ms/pass, 1.78 tok/pass` — where a pass is shared with every other live
  request, so `ms/pass` is the pace that request saw (the clock starts at
  admission; the queue wait is the service's `ttft` in `/v1/metrics`). A
  prefix cache miss is explained on its own INFO line at admission
  (2026-09-07): `prefix cache miss — 41 cut(s) probed against 42 entries;
  the nearest entry (position 62432) shares the first 812 of the prompt's
  64803 tokens (the prompt changed there)` — a divergence inside the
  system prompt (an agent client injecting a saved memory, a compacted
  history) reads differently from one at the previous answer; "the cache
  is empty", "a different prompt from its first token" and "a prefix of
  that entry: no entry at this prompt's own cuts" name the other cases,
  and when the conversation's own entry was pushed out the line says so
  instead — `an entry at this prompt's cut 2896 was evicted (5
  eviction(s) ago, last used at tick 1180, now tick 1412; the arena holds
  7 slots)` — from a ring of the last 256 evicted entries' prefix hashes,
  which is what separates an arena too small for the streams and their
  side requests from a client that changed its prompt. The decisions
  themselves (attach, snapshot, close,
  evict, rolling, hop) ride the op stream `down` fetches, one `X <op> <id>
  <position> <slot>` line each, so the arena's contents at any request can
  be replayed after the fact.
  The per-tick lines
  (a line per generated token, the bus's three per-window lines, the
  prefix cache's per-decision line, the adaptive engine's mode switches)
  sit at DEBUG since 2026-09-06 (`DGPP_BUS_TIMELINE=1` brings the bus's
  three per-window lines back at INFO on their own — the per-tick lines
  perturb the collective skew they measure; `DGPP_PIPELINE=0` turns the
  pipelined replay off — every replay settles right after its launch, the
  pre-2026-09-06 step — `DGPP_PIPELINE_TRACE=1` logs each launch and
  settle, `--mtp-schedule-fixed-lambda` (config `mtp_schedule_adapt: false`) holds
  the scheduled verify depth's λ at the configured constant instead of the
  adaptive EWMA (journaled as `msad`), `DGPP_DSV41_DENSE_GEMV=1` makes a DeepSeek-V4.1-Flash world take
  the 4-row GEMV chunks for its dense decode projections and head instead
  of the streaming tensor-core GEMM (the A/B switch; tolerance-equal
  forms, transcripts reorder — set it on every rank, the boot reads it at
  model construction), `DGPP_DSV41_EAGER_FOLD=1` makes a DeepSeek-V4.1-Flash world fold its
  eager walks through the host-driven reducer instead of the stream-ordered
  one (plan D9; the A/B switch, a site setting since 2026-09-14),
  `DGPP_DENSE_GEMV_ROWS=n` (default 4; a site setting
  since 2026-09-14, forwarded to every rank) is the other families' dense
  lowering bound — rows up to n take the GEMV chunks, bf16 rows above
  cuBLASLt's algorithm, fp8 rows above the streaming tensor-core GEMM to 256
  rows. Qwen's 17-64-token decode walks keep BF16 products kernel-only regardless
  of this bound. A value of 256 restores the pre-2026-09-14 lowering (every decode row count
  through the chunks) for an A/B, and `DGPP_SYNC_EAGER=1` makes an eager row — a prefill chunk,
  the sampled fallback's verify and re-draft — synchronize after every
  stage and validate its selection list before the attention, naming the
  stage a fault came from; the fault hunt's knob, not for serving) — the one-hour soak had written 307,000
  lines (40 MB) per rank at INFO, none of them aggregate; the same load
  now writes ~130 lines per minute, the per-request admitted / retired /
  deferred / cancelled lines and the stats line. `DGPP_LOG_LEVEL=debug`
  restores the rest; `scripts/serve_pace.py` and `scripts/fabric_xrank.py`
  read those lines and need it.
- **Continuously:** every tick record on the journal carries rank 0's
  running fold of its op stream (`od`) and, with the prefix cache on, of
  its cache decisions (`pd`). A peer whose own fold differs dies at once
  with the tick number (`journal: op-stream divergence at tick N`), and
  rank 0 then fails the service as for any dead peer. A quietly diverged
  rank therefore cannot serve for more than one tick.
- **At shutdown:** `down`'s four md5s (the procedure). They are the same
  evidence, post-mortem.
- **Per request:** `GET /metrics` (also available at `GET /v1/metrics`
  for backward compatibility) — JSON counters for requests, sheds, cancellations,
  failures, the admission policy, the prefix cache (entries, hits, tokens
  saved, hop snapshots, the TTFT split by hit and miss), the NVMe cache
  (`nvme_cache`: pages used and total, entries and block records, spills
  and restores begun / committed / failed, tokens restored, disk evictions,
  ops in flight, bytes moved and average op times), sampling
  fallbacks. `scheduler.spec_decode` reports cumulative MTP draft rounds,
  attempted and accepted draft tokens, and per-position counters; see the
  [counter definitions](openai-compatibility.md#speculative-decoding-counters)
  before calculating acceptance rates. `prefill.requests` reports each active
  prefill's prompt, processed, cached, computed and remaining tokens, updated at chunk boundaries even
  during a synchronous prefill. `scheduler.snapshot_age_ms` reports the age of
  the remaining scheduler/pool counters. See the
  [metrics contract and monitoring command](openai-compatibility.md#metrics-and-prefill-progress).
  Both metrics paths return `application/json`; direct Prometheus
  scraping requires a supported
  [exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/),
  which DGPP does not yet provide. `GET /health` is `{"status":"ok"}`
  while the engine lives.
- **Under load:** `scripts/serve_soak_run.sh MINUTES OUT_DIR` boots the
  world with the production knobs, starts `scripts/node_probe.sh` on every
  node, runs `scripts/serve_soak.py` (multi-turn chat, long generations,
  client cancellations, bursts above the queue bound), stops the world and
  prints the four op-stream md5s, the `STALLED` counts per rank log (the
  bus's stall witnesses — none in a healthy run) and each node's reclaim,
  swap and throttle sums; the soak's own summary gives TTFT and decode-pace
  percentiles per 10-minute window, the status counts and the prefix
  cache's line.
- **The other evidence procedures:** `scripts/fabric_prefill_repeat.sh OUT
  LEN...` (the steady-state prefill at each length with the four-way ids
  md5), `scripts/fabric_mtp_classes.sh OUT CLASS...` (MTP acceptance per
  prompt class), `scripts/serve_prefix_curve_sweep.sh OUT "GIB..." "C..."`
  (the prefix cache's capacity curve, one boot per point),
  `scripts/serve_failure_drill.sh VICTIM` (the kill −9 drill), and
  `scripts/serve_api_check.py HOST PORT` (the request fields — `stop`,
  `n`, `logit_bias`, the usage details — against a running world).

### Decode graph batch counters

`GET /metrics` and `/v1/metrics` expose `scheduler.decode_batch` in the
scheduler's published snapshot. `last_slots` is the capacity of the last
launched graph, `last_active` is the number of requests in that launch, and
`last_rows_per_request` is its verification width. These fields start at zero
and retain the last launch while idle or prefilling; use `scheduler.active`
and `scheduler.queued` for current occupancy.

`replays`, `rows` and `padded_rows` accumulate successful graph launches since
engine construction. A six-slot graph with five requests and two verification
rows per request adds one replay, twelve rows and two padded rows. Speculative
draft-chain work is excluded; rejected draft tokens are not padding.
`replays_by_slots` counts launches by graph capacity, with keys `"1"` through
`"16"`; bucket `"1"` includes scalar fallback. All sixteen buckets are always
present, even for engines with fewer slots, to keep the shape stable for
scrapers across deployments. Zero buckets do not establish which graph
families an engine supports. Non-graph engines report zeros.

For an interval, divide the increase in `padded_rows` by the increase in `rows`
when that denominator is positive. This measures verification-row padding,
not GPU time or utilization. Read the serving rank's counters once rather
than summing identical work across ranks. Launch counters do not assert GPU
completion; `snapshot_age_ms` describes the publication delay.

Graph capacity depends on the highest live slot as well as the number of
requests. With the default batching threshold, live slots 0 and 3 in a
four-slot engine require the four-slot graph even though only two requests
are active. The histogram records that capacity; it does not distinguish a
full graph from a sparse one. Use the row deltas to measure padding over a
representative traffic interval. The retained last-launch fields alone cannot
establish how often sparse launches occur, and an idle interval with no new
rows has no padding fraction. A server without `scheduler.decode_batch` needs
a newer binary; missing counters do not mean zero padding.

For draft attempts and acceptance, see the
[speculative decoding counters](openai-compatibility.md#speculative-decoding-counters).
Those count request verification decisions, excluding graph padding; rejected
drafts and padded rows measure different work.

## Ports and processes

| what | where |
|---|---|
| HTTP (rank 0) | 18080 |
| bus rendezvous | 29970 (rank 0 listens; peers connect) |
| admission journal | 29971 (rank 0 listens; peers connect and send `hello <rank>`) |
| peer binary, config and logs | `<stage_dir>/dgpp-serve`, `<stage_dir>/cluster.json`, `<stage_dir>/serve_r<rank>.log`, `<stage_dir>/serve_rank<rank>.ops` (fetched into the log dir by `down`) |
| rank 0 log, pid and op stream | `<log_dir>/serve_r0.log`, `<log_dir>/r0.pid`, `<log_dir>/serve_rank0.ops`, written as the run records it (flushed at every retire) |
| exit statuses | 0 orderly stop; 1 a startup or contract error (a configuration that differs from rank 0's included); 2 rank 0 after an engine failure or any rank after forced shutdown; 3 a peer released by its in-tick watch |

### Compact Qwen batch mappings

With `engine.compact_batches: true`, fixed-depth Qwen graph serving places active
requests in the smallest bucket that preserves the physical graph's numerical
dispatch range. Graphs of at most sixteen verification rows retain their width.
Wider graphs can shrink within the 17–32-row or above-32-row range, preserving
the FP8 split-K reduction boundary. For example, two live requests in slots 0
and 15 at MTP depth 3 use twelve groups (48 rows) instead of sixteen groups
(64 rows); a 32-row graph can still shrink to six groups (24 rows). Slots
outside every physical family retain scalar fallback. This conservative policy
avoids numerical changes from crossing kernel boundaries. Persistent KV,
recurrent/conv and prefix-cache state stays in the physical request slots; only row mappings and token feeds
are staged. Sampling RNG/counts/bias/proposals remain indexed by physical
request ID. Graph masks and verdicts are indexed by compact batch group.
The mapping is double-buffered per graph family for pipelined replays.

The setting defaults to false, retaining the physical-prefix policy. Rank 0 journals the setting
and peers adopt it before graph construction; it is not read from the
environment. Confidence-scheduled verify depth and other model families currently use the previous policy.
This changes neither model capacity nor the set of graph bucket sizes.
