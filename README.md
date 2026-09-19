# DGPP

DGPP is a C++/CUDA inference engine for NVIDIA DGX Spark (GB10) systems.
It serves GLM-5.3-Flash, full GLM-5.3, Qwen3.8-Flash-Next, GLM-4.7 and
DeepSeek-V4.1-Flash through an OpenAI-compatible HTTP API, with tensor
parallelism over RoCE for multi-node deployments. Supported configurations
use one, two or four nodes, depending on the model and its memory requirements.

The repository contains the CUDA kernels, RDMA collectives, tokenizer,
Jinja chat-template interpreter, scheduler, prefix cache and HTTP service.
It uses the CUDA runtime, cuBLASLt and libibverbs. Rank 0 coordinates
requests through an admission journal; peers check their operation streams
against it throughout a run.

Qwen supports up to 64 decode rows (C16/MTP3). See the
[implementation and validation record](benchmarks/results/2026-09-17-qwen-spark-decode/README.md)
for the Spark optimizations, focused tests and performance limitations.

## Supported models and configurations

These serving configurations have deployment templates and recorded
measurements. World size is the number of ranks, with one DGX Spark per
rank; world 1 runs locally, while worlds 2 and 4 use tensor parallelism
over RoCE. Each quant links to its specific Hugging Face model card.

| Model | Quant / Hugging Face model card | World sizes | Example configuration |
|---|---|---|---|
| GLM-5.3-Flash | [unsloth/GLM-5.3-Flash-FP8](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8) | 4 | Copy the [base template](deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json) to `cluster_glm-5.3-flash_fp8_w4.json` and set `model` to the linked FP8 repository |
| GLM-5.3-Flash (hybrid) | [HawkBearPig/GLM-5.3-Flash-NVFP4-FP8](https://huggingface.co/HawkBearPig/GLM-5.3-Flash-NVFP4-FP8) | 2, 4 | [Two nodes](deploy/cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json), [four nodes](deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json) |
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | 2, 4 | [Two nodes](deploy/cluster_qwen-3.8-flash-next_fp8_w2.example.json), [four nodes](deploy/cluster_qwen-3.8-flash-next_fp8_w4.example.json) |
| Qwen3.8-Flash-Next | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | 1, 2 | [One node](deploy/cluster_qwen-3.8-flash-next_nvfp4_w1.example.json), [two nodes](deploy/cluster_qwen-3.8-flash-next_nvfp4_w2.example.json) (mapped n-gram table, dense projections FP8 at load) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) | 4 | [Four nodes](deploy/cluster_glm-4.7_nvfp4_w4.example.json) |
| GLM-5.3 | [HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64](https://huggingface.co/HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64) | 4 | [Four nodes](deploy/cluster_glm-5.3_int4-int8_w4.example.json) |
| DeepSeek-V4.1-Flash | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 4 | [Four nodes](deploy/cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json) |

The Qwen NVFP4 templates use `engine.ngram_table: "mmap"` to read the
n-gram table from NVMe, `engine.decode_graph: true` for resident graph serving,
and `engine.dense_weights: "fp8"` to encode dense projections at load. On two
Sparks, mapping saves 23.84 GiB per rank while staying within 3.6% of resident
decode throughput and 2.9% of resident prefill time in the matched campaign.
Use `--dense-weights checkpoint` to retain the checkpoint's BF16 dense stack.
See the [single-node guide](docs/qwen38_single_spark.md) and
[two-node benchmark](benchmarks/results/2026-09-16-qwen-nvfp4-w2.md).

The GLM-5.3 hybrid takes the main-stack routed experts from
[dabsLabs](https://huggingface.co/dabsLabs/GLM-5.3-Flash-NVFP4) and the
remaining tensors, including MTP, from
[Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8).
The GLM-5.3-Flash templates use
`HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`. The setup command below downloads it
once on rank 0 and syncs the selected snapshot to peers. To use the FP8 release
instead, set
`model` to `unsloth/GLM-5.3-Flash-FP8`.
To reproduce the hybrid from its sources, use
[the composition tool](tools/compose_nvfp4_hybrid.py) and
[the NVFP4 notes](docs/nvfp4_plan.md).

The linked templates enable MTP at the depth measured best for that deployment.
Plain decode, deeper MTP, alternate cache formats and slot counts are launcher
knobs on these templates; [the deployment catalogue](deploy/README.md) maps the
supported shapes to their flags. [The benchmark tables](docs/benchmarks.md)
record the modes measured for each deployment.

## Highlights

- **Tensor-parallel resident serving** on four nodes: each rank holds its
  slice of the model resident on the GPU and typically boots from a per-rank
  image cache in 15–30 s, depending on the model.
- **Graph-based decode**, including collectives and MTP speculative
  decoding, with scalar or batched graphs selected for the active requests.
  Greedy MTP produces the same tokens as plain decode; sampled MTP
  preserves the target distribution.
- **Row-aware tensor-core execution and grouped prefill**: dense kernels select
  their lowering from the active row count, and queued cold prompts can share a
  forward pass while retaining request-local attention and state.
- **Model-specific prefill paths**: packed int4/int8 tensor-core prefill for
  full GLM-5.3, tiled QSA prefill for Qwen, and bounded grouped prefill for
  DeepSeek-V4.1-Flash. Qwen can optionally yield between prefill chunks so
  active decodes continue making progress.
- **Adaptive DSpark verification**: DeepSeek uses confidence-scheduled draft
  depth, including a batch-aware rule and an adaptive value of decode time.
- **Exact prefix caching**: matching token prefixes can reuse a stored
  session snapshot while preserving the cold-prefill result.
- **OpenAI-compatible text generation**: Chat Completions and legacy
  Completions, streaming, constrained tool calls, `response_format` with
  supported JSON schemas, `reasoning_content`, logprobs, `stop`, `n`,
  `logit_bias` and usage details; plus model, health and metrics endpoints.
  See the [API compatibility profile and live prefill metrics](docs/openai-compatibility.md)
  for supported options and model-dependent limitations.
- **GLM-5.3-Flash image inputs**: PNG/JPEG data URIs in Chat Completions,
  using the checkpoint's native vision encoder. Multiple images, streaming
  and MTP work together, with image-aware prefix caching; see
  [image inputs](docs/vision.md) for examples and memory requirements.
- **Deterministic across ranks**: admissions journaled from the head, every
  tick's operation-stream digest checked on every peer, and all ranks'
  complete streams compared at shutdown.
- **Process-failure detection**: a rank process exiting fails the service
  within seconds. Silent node loss is detected by the bus watchdog.
- **Deployment and monitoring**: a shared cluster config, versioned
  releases, periodic throughput logs and JSON counters at `/metrics`
  (also available at `/v1/metrics`). Startup checks the memory plan before
  allocation. Cache capacity is configurable,
  with BF16, FP8 or FP4 latent storage for GLM-5.3.

## Performance

These are the latest measurements for representative shipped configurations.
See [the benchmark tables](docs/benchmarks.md) for the complete per-model and
per-class results, measurement scopes and reproduction commands.

| configuration | single-request engine decode | loaded request-wall decode | cold service prefill at ~2K / 8K / 32K |
|---|---:|---:|---:|
| GLM-5.3-Flash-FP8, 4 Sparks | 42.2–48.7 tok/s | 62.2–67.5 tok/s at C4 | Rerun pending |
| GLM-5.3-Flash NVFP4/FP8, 4 Sparks | 50.6–57.8 tok/s | 96.4–104.0 tok/s at C4 | **1.465 / 6.005 / 27.753 s** |
| GLM-5.3-Flash NVFP4/FP8, 2 Sparks | 22.7–33.0 tok/s | 38.9–47.3 tok/s at C4 | Rerun pending |
| Qwen3.8-Flash-Next-FP8, 4 Sparks | 63.2–77.7 tok/s by class | 142.1–167.3 tok/s at C4 | — |
| Qwen3.8-Flash-Next-FP8, 2 Sparks | 41.7–49.6 tok/s by class | 69.3–83.2 tok/s at C4 | 1.299 / 5.108 / 21.168 s |
| Qwen3.8-Flash-Next-NVFP4, 2 Sparks, mapped n-gram | **62.1–74.9 tok/s by class** | **119.0–136.9 tok/s at C4** | **1.241 / 4.870 / 20.286 s** |
| Qwen3.8-Flash-Next-NVFP4, 1 Spark | 42.6–50.3 tok/s by class | 69.4–83.7 tok/s at C4 | — |
| GLM-4.7-NVFP4, 4 Sparks | 29.5–33.3 tok/s by class | 62.7–68.7 tok/s at C4 | 2.809 / 16.865 / — |
| full GLM-5.3 int4/int8, 4 Sparks | 25.4–29.2 tok/s by class | 42.3–47.0 tok/s at C4 | Rerun pending |
| DeepSeek-V4.1-Flash MXFP4/FP8, 4 Sparks | 49.64 aggregate tok/s | 108.49 aggregate tok/s at C6 | 1,383 prompt tok/s on its 2,950-token cold prompt |

Except for DeepSeek, decode ranges are the five prompt classes and prefill is
the cold HTTP service path. The single-request column uses the server's retired
decode work, while the loaded column includes full request wall time. Qwen's
newest cold-service campaigns were run at two
Sparks; the four-Spark and single-Spark service prefills have not been re-run
on the current path. DeepSeek uses the vLLM DGX Spark recipe's client and
prompt set, with different aggregate and per-stream timing scopes, so compare
its row within that workload. Dates, actual prompt lengths, quality gates and
reproduction commands are in [the benchmark tables](docs/benchmarks.md).
DSA prefill now sizes query tiles for the current context within its existing
workspace. The four-Spark Flash hybrid has been remeasured; other DSA
configurations need fresh prefill measurements.

## Status

As of 2026-09-16, the source tree has nine measured deployment templates
covering five model architectures on one, two or four Sparks. The shared
engine provides graph decode, transactional MTP, row-batched execution,
grouped prefill, prefix caching, deterministic multi-rank scheduling and the
OpenAI-compatible service. Current quantized paths cover FP8, NVFP4, MXFP4 and
full GLM-5.3's packed int4/int8 format. Qwen NVFP4 runs on one or two Sparks by
mapping its n-gram table from NVMe and encoding the dense stack to FP8 at load.

The Qwen eight-slot decode graphs and prefill continuation are implemented as
opt-in controls; the shipped Qwen templates retain four slots and monolithic
admission. DeepSeek ships at six slots with DSpark depth 4, confidence
scheduling, bounded grouped prefill and the stream-ordered eager collective.
Full GLM-5.3 ships at eight slots and uses packed tensor-core prefill from 128
rows. Version 0.1.0 remains the original GLM-5.3-Flash sign-off release; the
current source has advanced beyond that baseline.

[PLAN.md](PLAN.md) summarizes implementation status,
[CHANGELOG.md](CHANGELOG.md) records changes, and
[the remaining work](docs/next_steps.md) lists current limitations.

## Quickstart

Use an internet-connected Spark as rank 0. First get the source:

```bash
git clone https://github.com/HawkBearPig/dgpp.git
cd dgpp
```

Run the remaining commands in this same shell, after installing the
[dependencies on each node](docs/getting-started.md#1-install-the-dependencies).
This example serves GLM-5.3-Flash on four Sparks; choose a different
[deployment template](deploy/README.md) for another model or node count.
See [Getting started](docs/getting-started.md) for the full walkthrough and troubleshooting guidance.

### 1. Configure your deployment

Copy the model template and create your site file, preserving existing files.
Edit `.env` to set `DGPP_NODES` (rank 0 first) and `DGPP_SSH_USER`, then
[verify SSH-key access from rank 0 to each peer](docs/getting-started.md#3-set-your-node-addresses-and-ssh-user).

```bash
CONFIG=deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json
test -f "$CONFIG" || cp "${CONFIG%.json}.example.json" "$CONFIG"
test -f .env || cp .env.example .env
```

Set `CONFIG` again if you open a new shell; it is the deployment filename, not a persistent setting.

### 2. Select the RoCE lanes (multiple nodes only)

Discover the interfaces, then copy the appropriate device names and GID indices
into `.env`, matching lane subnet order across nodes; discovery does not test RDMA connectivity.

```bash
python3 scripts/discover_roce.py --config "$CONFIG"
```

### 3. Build

Build the server on rank 0; the launcher stages the executable on peers.
If CUDA is not found, see [compiler setup](docs/getting-started.md#5-build-the-server-on-rank-0).

```bash
cmake --preset release
cmake --build --preset release -j 4
```

The release preset produces `build-release/dgpp-serve` with `-O3` optimization
and no debug symbols. Testing uses the separate `ci` preset and `build-ci/`
directory, with debug symbols retained; see [testing](docs/testing.md).

### 4. Download the checkpoint

Download once into rank 0's standard Hugging Face cache, then sync peers sequentially.
Add `--sync-only` if rank 0 already has the checkpoint.
For this example, allow roughly **250 GiB per node** for the checkpoint and
one resident cache; check [storage and offline options](docs/getting-started.md#6-download-once-and-sync-to-peers) first.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-download.txt
python scripts/download_model.py --config "$CONFIG"
```

### 5. Check and start

Fix any failed preflight checks, then start the deployment and wait for `READY`.
The API stays on localhost; it has no authentication or TLS.

```bash
python3 scripts/dgpp-cluster doctor --config "$CONFIG"
python3 scripts/dgpp-cluster up --config "$CONFIG"
```

### 6. Send a request

Ask the running model a question. This short example requests low reasoning
effort because reasoning tokens also count toward `max_tokens`:

```bash
MODEL=$(curl --fail -s http://127.0.0.1:18080/v1/models | jq -r '.data[0].id')
jq -n --arg model "$MODEL" \
  '{model:$model,max_tokens:160,reasoning_effort:"low",messages:[{role:"user",content:"Name three primary colors."}]}' \
  | curl --fail http://127.0.0.1:18080/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @-
```

Stop it when finished, using the same config:

```bash
python3 scripts/dgpp-cluster down --config "$CONFIG"
```

## Release and install

A release is a versioned tarball installed once per node. Starting an
installed release stages the configuration and uses the installed binary.

```bash
scripts/release.sh                                 # build the release preset, stage, verify, pack
scripts/dgpp-cluster install dist/dgpp-VERSION.tar.zst --config "$CONFIG"
scripts/dgpp-cluster up --release VERSION --config "$CONFIG"
scripts/dgpp-cluster releases --config "$CONFIG"
```

Select a release with the config's `release` key or `--release`.

The version is the tree's, stamped at build time by `cmake/version.cmake`
into the binary (`dgpp-serve --version`; `0.1.0+g<sha12>`, `.dirty` when
the tree had uncommitted changes) and sent on the journal's settings
record, so a world of mixed versions refuses to form. Inside the tarball:

| path | what |
|---|---|
| `bin/dgpp-serve` | the server, rpath `$ORIGIN/../lib` |
| `lib/libcudart.so.13`, `lib/libcublasLt.so.13` | the CUDA runtime it was built against |
| `scripts/` | launcher, process/preflight/config helpers, checkpoint downloader and API check |
| `deploy/*.example.json` | all supported deployment templates |
| `README.md`, `docs/` | package-specific setup, dependency and networking guides |
| `MANIFEST`, `MANIFEST.sha256` | version, git sha, CUDA, build host and date; every other file's checksum |

Beyond the tarball a node needs the NVIDIA driver, rdma-core, libnl and
libstdc++, all part of the DGX OS image, plus the checkpoint and resident
image caches on its local NVMe. `install` unpacks under `paths.release_dir`
(`~/dgpp/releases/dgpp-<version>/`) and checks every file against the
manifest. Versions can be installed side by side; stop the running service
and select an older version to roll back. The launcher starts the service
on demand; no boot-time service units are included. When no release is
selected, `up` stages `build-release/dgpp-serve` to the peers. Use
`--bin build-ci/dgpp-serve` explicitly when deploying a testing build.

## Configuration

Site settings live in `.env` at the repository root, shared by every model
deployment. Scripts load it automatically through `scripts/site_env.py`
(shell scripts use `scripts/cluster_env.sh`). `DGPP_ENV_FILE` selects another
file; exported variables override file values. Values are literal, optionally
quoted: no shell commands or variable expansion are evaluated. The site helper
loads an allowlist of settings, so credentials such as `HF_ACCESS_TOKEN` stay out of the
scripts' environment and generated server config.

| `.env` key | purpose | default |
|---|---|---|
| `DGPP_NODES` | Space-separated hostnames/IPv4 addresses in rank order. The first is rank 0, where launch/download commands run. A deployment uses its first `world_size` nodes. These are host addresses, not RDMA device names. | required |
| `DGPP_SSH_USER` | Peer login for binary staging, process control, diagnostics and checkpoint sync. Needs SSH-key access and write access to the configured directories. | current login when empty or absent |
| `DGPP_HTTP_PORT` | Default client-facing API TCP port on rank 0; deployment `http.port` overrides it. | 18080 |
| `DGPP_HTTP_BIND` | Default IPv4 listening address on rank 0; deployment `http.bind_host` overrides it. Keep localhost unless you have arranged access protection. | `127.0.0.1` |
| `DGPP_FABRIC_PORT` | Rank-0 TCP rendezvous listener used to establish the inter-node transport. Peers must reach it; clients do not use it. Keep it private to the cluster. | 29970 |
| `DGPP_JOURNAL_PORT` | Rank-0 TCP listener that distributes ordered scheduler operations to peers. Must differ from the fabric/API ports; keep it private to the cluster. | 29971 |
| `DGPP_LOG_DIR` | Base directory on rank 0 for logs, process records and collected peer logs. The launcher adds a deployment-specific subdirectory. | `~/dgpp/log` |
| `DGPP_STAGE_DIR` | Base directory on peers for staged development binaries, runtime config and logs. The launcher adds a deployment-specific subdirectory; it is not the model cache. | `/tmp/bus4` |
| `DGPP_RELEASE_DIR` | Base directory on each node for versioned installed server releases. Use a persistent, writable location. | `~/dgpp/releases` |
| `DGPP_BUILD_DIR` | Explicit build-directory override. Relative paths use the repository root. Leave unset to keep release and testing builds separate. | `build-release` for serving/packaging; `build-ci` for testing (`build-<preset>` with `DGPP_PRESET`) |
| `DGPP_DATA_DIR` | Evaluation-data location. Relative paths use the repository root. | `data` |
| `HF_HUB_CACHE`, `HF_HOME` | Downloaded checkpoints. An explicit hub cache wins; otherwise use `HF_HOME/hub`. Leave unset for the standard cache or choose a disk with room for the full checkpoint on each node. | `~/.cache/huggingface/hub` |
| `DGPP_RESIDENT_CACHE_DIR` | Per-node disk cache of prepacked weight images for faster reloads, separate from the HF checkpoint. Takes precedence over deployment `paths.resident_cache`. | `~/.cache/dgpp/resident` |
| `DGPP_ROCE_DEVICES` | One or two ordered local verbs device names, not Linux interface names. Use `discover_roce.py`; match lane subnets in the same order across nodes. | discover active Ethernet devices |
| `DGPP_ROCE_GID_INDICES` | One RoCE-v2 address-table index per explicitly selected device, in the same order. Pin these when a device offers several networks; discover the values on each host. | automatic RoCE-v2 selection |
| `DGPP_NODE_OVERRIDES` | JSON map keyed by exact `DGPP_NODES` entries, with per-node device/GID/hub-cache/resident-cache values. Use when peers have different NIC names or disks; see `.env.example`. | none |

Model settings live in deployment JSONs under `deploy/`. Each has a
`world_size` of 1, 2 or 4 and uses that many nodes from the beginning of
`DGPP_NODES`. Too few nodes is an error, not a fallback to a smaller world.
Select the deployment explicitly with `--config FILE`.
Wrappers that accept a config argument pass it through to their children.
Stage and release directories must be absolute or start with `~/`, without
spaces or shell syntax.
Log and staging directories are namespaced by deployment-file path;
`dgpp-cluster paths` prints the effective locations. An explicit `--log-dir`
is used as-is. `up` refuses an existing deployment unless `--replace` is given;
`down` without `--config` stops every recorded deployment that is running;
cleanup uses recorded process identity, never a binary-name kill.

The launcher resolves the deployment and site settings into
`<log_dir>/cluster.resolved.json` when starting the service. Both the head
and peers read that resolved config; the original JSON and `.env` are not
sent to peers. To inspect or use the runtime config with the native binary:

```bash
scripts/dgpp-cluster resolve --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json
# Save that JSON to a file before passing it to dgpp-serve --config.
```

The native binary reads resolved JSON, not `.env` or deployment templates.
Unknown keys and invalid types are rejected.

Rank 0 reads the model, ports and engine options, with command-line flags
after `--config` overriding the file. It sends the effective settings to
peers before model construction. Peers use their own files for bootstrap
addresses and local paths, then verify the shared settings by digest.
The defaults below come from `ClusterConfig::Engine`; deployment
templates set their serving options explicitly.

### Deployment and endpoint

| Key | Required | Purpose and when to change it | Default |
|---|---|---|---|
| `model` | yes | Exact Hugging Face repository ID, such as `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`. Selects the checkpoint tensors, tokenizer and chat template. It is not a local filesystem path or a generic quant name. | — |
| `world_size` | yes | Number of participating nodes/ranks: 1, 2 or 4. Uses the first N entries of `DGPP_NODES`; each rank stores its tensor-parallel share in memory. Choose a supported model/world pair, not an arbitrary smaller number to save machines. | — |
| `http.bind_host` | no | IPv4 address on rank 0 that accepts API connections. `127.0.0.1` is local-only; a LAN address exposes that interface; `0.0.0.0` exposes all IPv4 interfaces. The server has no authentication/TLS. Does not select the RoCE interface. | `DGPP_HTTP_BIND`, otherwise `127.0.0.1` |
| `http.port` | no | TCP port clients use for the API, in 1–65535. Change it if the default is occupied, then update client URLs. Overrides the site HTTP port and must differ from fabric/journal ports in a multi-node deployment. | `DGPP_HTTP_PORT`, otherwise 18080 |
| `http.max_body_bytes` | no | Maximum serialized HTTP request body in bytes, as a positive integer. Allows large document prefills and agent histories; independent of the model's token/KV capacity. Oversized requests receive HTTP 413 based on Content-Length, before tokenization. Buffers grow with received data, so this does not preallocate the limit per connection. The binary flag `--http-max-body-bytes` overrides it. | 268435456 (256 MiB) |
| `release` | no | Installed software version to run on every node, not a model revision. Select a version under `DGPP_RELEASE_DIR/dgpp-VERSION`; use it to upgrade or roll back server binaries. `--release` overrides this field; `--bin` overrides binary selection. | Empty: use the development build |
| `paths.resident_cache` | no | Disk directory for prepacked per-rank weight images, which speed subsequent loads. This is separate from the Hugging Face download cache and consumes additional disk space. Prefer the shared `DGPP_RESIDENT_CACHE_DIR` site setting unless a deployment needs its own directory. | Empty: `~/.cache/dgpp/resident`; `DGPP_RESIDENT_CACHE_DIR` takes precedence |

### Request capacity and memory

For a first run, retain the template values. Tune `kv_capacity` for context
space and `max_concurrency` for simultaneous work; these are different limits.
Startup checks the combined memory plan before loading.

| Key | Required | Purpose and when to change it | Default |
|---|---|---|---|
| `engine.max_concurrency` | no | Maximum actively executing requests, not TCP connections or queued requests. More slots can improve aggregate throughput but use more state/scratch memory and may increase per-request latency. Allowed range is 1–8, subject to model/MTP row limits below. | 8 |
| `engine.kv_capacity` | no | Shared context-token pool on each rank, across active requests—not a separate allowance for every request. A prompt and its generated answer must fit. Increase for longer contexts or more simultaneous context; memory use increases and allocation is rounded to model block boundaries. | 8192 tokens |
| `engine.kv_dtype` | no | GLM-5.3 latent-cache precision: `bf16`, `fp8`, or `fp4`. Lower precision reduces latent storage at a numerical-accuracy cost; it does not quantize model weights. The index cache stays FP8. Qwen, GLM-4.7 and DeepSeek K/V caches remain BF16. | `bf16` |
| `engine.embed_sharding` | no | Full GLM-5.3 and DeepSeek embedding/head placement: `replicated` keeps the full table on every rank; `vocab` keeps each rank's vocabulary slice and folds token lookups. The full-GLM template uses `vocab` to save 1.33 GiB/rank; the DeepSeek template retains `replicated`. Other families ignore it. | `replicated` |
| `engine.default_max_tokens` | no | Answer-token budget for requests that omit `max_tokens`. Clients may supply their own value; this is not a global hard limit. A larger default also reserves more context space under full admission. | 256 |
| `engine.queue_limit` | no | Maximum requests waiting for an execution slot or memory budget. Additional arrivals receive HTTP 503 `overloaded`. Increase to tolerate bursts, at the cost of longer waits—not higher execution capacity. | 64 |
| `engine.max_connections` | no | Maximum simultaneously open HTTP connections on rank 0, including idle keep-alive connections and streams. Excess connections receive 503. Size this separately from active request slots. | 64 |
| `engine.prefix_cache_gib` | no | Memory budget per rank for reusable prefix-state snapshots. Repeated conversation prefixes can skip prefill work; larger budgets retain more snapshots but leave less memory for other state. Set 0 to disable. This is not the on-disk resident weight cache. | 1.5 GiB |
| `engine.admission` | no | When to reserve context space. `full` reserves prompt plus the requested answer budget before admitting a request. `grow` starts with a smaller reservation and extends it during generation; if space runs out, the youngest request is shed. Use `full` for predictable reservations, `grow` to trade that guarantee for denser occupancy. | `full` |
| `engine.admission_window` | no | Answer-token reservation increment used by `grow` admission. Larger increments reduce growth frequency but reserve more space ahead of use. Has no effect under `full`. Must be positive. | 256 tokens |
| `engine.prefill_budget_tokens` | no | Qwen and GLM-5.3-Flash graph engines: maximum prefill tokens per scheduler tick, with a decode pass between chunks. Use an aligned budget no larger than the model's prefill chunk limit. 0 keeps full-prompt admission. | 0 (disabled) |
| `engine.prefill_idle_budget_tokens` | no | Larger prefill budget when no request is actively decoding. Requires an enabled busy budget, must be at least that budget, aligned and within the same prefill limit. Rechecked after each chunk. 0 uses the busy budget for all chunks. | 0 (disabled) |

### Execution and performance

These settings change how the model runs. Use the matching deployment template
first; change one setting at a time and measure the effect on your workload.

| Key | Required | Purpose and when to change it | Default |
|---|---|---|---|
| `engine.decode_graph` | no | Use CUDA graph replay for decode to reduce CPU launch overhead. On one node, this selects resident graph serving instead of the eager streaming path. Required for MTP and the single-Spark Qwen templates; leave enabled for the documented serving configurations. | false |
| `engine.mtp` | no | Enable multi-token prediction: draft candidate tokens, then verify them with the main model. Can reduce decode time when drafts are accepted, but adds draft state and verification work. Requires `decode_graph`; not a larger request batch. | false |
| `engine.mtp_depth` | no | How many tokens to draft per speculative step, 1–5. Greater depth can accept more tokens per pass, but uses more verification rows and can waste work when drafts are rejected. Values above 1 require MTP; supported batching varies by model. DeepSeek's shipped template uses depth 4. | 1 |
| `engine.mtp_schedule` | no | Enable confidence-scheduled verify depth for a model with a confidence head. Greedy DeepSeek requests verify only the leading drafts whose survival probability justifies another row; sampled requests keep the configured depth. | false |
| `engine.mtp_schedule_row_ms`, `engine.mtp_schedule_base_ms` | no | Cost model for scheduled verification: milliseconds for another verify row and fixed work per pass. These values are deployment measurements; retain the DeepSeek template values unless re-profiling that world. | 8.0 / 28.0 |
| `engine.mtp_schedule_lambda`, `engine.mtp_schedule_min_depth`, `engine.mtp_schedule_adapt` | no | Floor for the value of decode time in tokens/ms, minimum verified draft depth, and whether the value adapts from committed tokens and modeled time. Lambda 0 derives the reservation rate. | 0 / 1 / true |
| `engine.prefill` | no | DeepSeek prefill mode: `bounded` runs every prompt row through the encoder and only the final window through the decoder; `exact` runs every layer over every row for parity work. Other families ignore it. | `bounded` |
| `engine.ngram_table` | no | Qwen n-gram embedding-table placement. `resident` keeps it in device-accessible memory; `mmap` leaves it on local NVMe and fetches needed rows through the host page cache. Both shipped Qwen NVFP4 templates use `mmap`; the two-Spark campaign measured at most 3.6% lower decode throughput for 23.84 GiB less planned model memory per rank. DeepSeek's Engram tables use their own mapped checkpoint sidecar. | `resident` |
| `engine.dense_weights` | no | Qwen dense-projection storage. `checkpoint` retains the checkpoint's BF16 form; `fp8` converts dense projections at load time to reduce their memory footprint, with quantization error. Does not select another HF repository or change the expert quant; other families ignore it. | `checkpoint` |
| `engine.graph_batch_min_live` | no | Active-request count at which decode switches from scalar to batched graphs. A lower threshold starts batching earlier; batching may improve throughput while doing extra padded-row work. 0 chooses min(2, `max_concurrency`); explicit values must be 1 through `max_concurrency`. | 0 (automatic) |
| `engine.sampling_candidates` | no | Number of candidate tokens gathered per rank on the sampled-token fast path, 1–256. Smaller values reduce routine work but may trigger more full-gather fallbacks. The fallback preserves sampling correctness; this is not the client's `top_k` parameter. | 128 |
| `engine.bulk_pace_gbps` | no | Sender pacing rate per queue pair for bulk/prefill communication, in gigabits per second. Negative derives the rate from link speed; 0 disables pacing. Override only when measuring network contention—this is not an API throughput limit. | -1 (automatic) |
| `engine.bulk_inflight` | no | Maximum in-flight bulk stripes per lane. More can keep the link busy but increase pressure on buffers and competing traffic. -1 uses the transport default (4); normally leave automatic. | -1 (automatic) |
| `engine.rendezvous_timeout_ms` | no | How long ranks may take to join the transport rendezvous. Increase for slow cold starts or delayed peers; it does not increase an HTTP request's timeout. | 120000 ms |
| `engine.stats_interval_s` | no | Interval between periodic server throughput/state log lines. Shorter intervals give finer operational visibility and more log output. Set 0 to disable periodic statistics. | 10 seconds |
| `engine.reasoning_in_content` | no | Put reasoning text in the response's `content`, separated by the model's `</think>` marker, instead of a separate `reasoning_content` field. Use only for clients that need that combined format; it does not disable reasoning. | false |
| `engine.no_eos` | no | Ignore the model's end-of-sequence token so measurement runs continue to their token budget. Leave false for normal serving, where a model should be allowed to finish its answer. | false |

The application allows up to eight request slots. A speculative request uses
`1 + mtp_depth` physical decode rows. GLM-5.3-Flash supports eight batched
rows; Qwen and full GLM-5.3 support sixteen; GLM-4.7 and DeepSeek support
thirty-two. Graph families capture the slot prefixes that fit, with scalar
fallback for an unsupported active shape. The startup log reports the selected
capacity and rejects a configuration that cannot fit its required rows.

Sampling defaults come from the checkpoint's `generation_config.json`
and per-request fields. Flags such as `--temperature` override them for
measurement runs. See [operations](docs/operations.md) for memory planning,
MTP depth tradeoffs and deployment checks.

## Source layout

Model implementations share the engine, service, scheduler and text stack:

| directory | namespace | what |
|---|---|---|
| `src/serve/` | `dgpp::serve` | the HTTP server, the OpenAI-compatible service, the admission journal, the throughput line, the text frontend interface |
| `src/sched/` | `dgpp::sched` | the scheduler (admission, the tick, cancel and stop), the prefix cache's index, the `SchedulerEngine` interface every model implements |
| `src/sample/` | `dgpp::sample` | the sampler's host math: penalties, the logit bias, masks, top-k/p, the exact prefix decision |
| `src/text/` | `dgpp::text` | the tokenizer (HF tokenizer.json), the chat template (Jinja), the tool grammar and parser, the JSON-schema grammar |
| `src/kernels/` | `dgpp` | CUDA kernels; `pick.cu` and `sample_pick.cu` are the device pick and sampler any model's logits can use, the rest carry their model's name |
| `src/net/` | `dgpp::net` | the RoCE collective bus, TCP, the roster |
| `src/engine/` | `dgpp` | shared session interfaces, eager and graph engines, speculative decoding, memory plans and prefix arenas |
| `src/models/qwen/`, `src/models/glm4/` | `dgpp` | Qwen3.8-Flash-Next and GLM-4.7 configuration, loaders, layers and sessions |
| `src/models/glm/` | `dgpp` | GLM-5.3-Flash: the forward pass, loader, MoE and mHC layers, session adapters and MTP draft |
| `src/models/glm_dsa/` | `dgpp` | full GLM-5.3 configuration, packed loader, MLA/DSA model and session implementation |
| `src/models/dsv41/` | `dgpp` | DeepSeek-V4.1-Flash configuration, MXFP4/FP8 loader, CSA2, Engram and DSpark session implementation |
| `src/models/` | `dgpp` | the KDA and DSA layer libraries and the quantized matrix, shared by any model that uses them |
| `src/loaders/`, `src/core/`, `src/common/` | `dgpp` | safetensors, the HF cache, JSON; arenas and tracing; logging and process memory |

The server binary is `dgpp-serve` (`apps/dgpp_serve.cpp`); the GLM
tools keep their names (`glm_gen_check`, `glm_forward_check`, …).

`tools/` holds the Python checkpoint and reference tooling the tests use;
`scripts/` the fabric and serving operations; `apps/` the binaries;
`benchmarks/` the probes and the engineering record; `deploy/` the cluster
config.

## Documentation

| page | what |
|---|---|
| `docs/operations.md` | booting, stopping and watching the serving world; memory and the image cache on a node; failure semantics |
| `docs/benchmarks.md` | every measured serving number: per model, world, concurrency and prompt class, with the method to reproduce each |
| `docs/signoff_v1.md` | the v1 performance and hardening sign-off, with every measurement |
| `docs/next_steps.md` | what is worth doing next, ranked by cost and benefit |
| `docs/tools.md` | serving, testing and diagnostic commands |
| `docs/testing.md` | the test suites and what each proves |
| `docs/numerics.md` | judging a numerics change; the sampling-width gate |
| `docs/mtp.md` | speculative decode with the MTP layer |
| `docs/measurements.md` | current validated platform, network, kernel and serving measurements, with historical observations separated |
| `docs/checkpoint_budget.md`, `docs/qwen38_checkpoint_budget.md`, `docs/checkpoint_budget_glm53.md`, `docs/checkpoint_budget_dsv41.md` | generated checkpoint inventories, per-rank residency and decode traffic budgets |
| `docs/batched_mtp_graph_stall.md` | a worked stall investigation, from symptom to root cause |
| `docs/qwen38_flash_next_plan.md` | Qwen3.8-Flash-Next's architecture, placement, decisions and status |
| `docs/glm47_plan.md` | GLM-4.7 NVFP4 architecture, modelopt format contract, decisions, gates and status |
| `docs/glm53_plan.md` | full GLM-5.3's packed checkpoint, DSA/MLA changes, placement and serving status |
| `docs/deepseek_v41_flash_plan.md` | DeepSeek-V4.1-Flash's CED, CSA2, Engram, DSpark and serving implementation |
| `docs/performance_improvement_plan.md` | current performance findings, completed changes and next measured targets |
| `DESIGN.md` | architecture and implementation contracts |
| `PLAN.md` | the milestones, their exit gates and status |
| `benchmarks/README.md`, `benchmarks/results/` | benchmark probes, dated results and reproduction commands |
| `CHANGELOG.md` | the history by milestone |

## Contributing

`CONTRIBUTING.md` describes the build presets, the test discipline, the
evidence a fabric change needs, the code style and where things go.

## License

Apache License 2.0. See `LICENSE`.
