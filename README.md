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

## Supported models and configurations

These serving configurations have deployment templates and recorded
measurements. World size is the number of ranks, with one DGX Spark per
rank; world 1 runs locally, while worlds 2 and 4 use tensor parallelism
over RoCE. Each quant links to its specific Hugging Face model card.

| Model | Quant / Hugging Face model card | World sizes | Example configuration |
|---|---|---|---|
| GLM-5.3-Flash | [unsloth/GLM-5.3-Flash-FP8](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8) | 4 | Copy the [base template](deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json) to `cluster_glm-5.3-flash_fp8_w4.json`, set `model` to the linked FP8 repository and lower `engine.kv_capacity` to 393216 (the FP8 experts are 31 GiB larger per rank; the startup memory plan refuses the base template's context) |
| GLM-5.3-Flash (hybrid) | [HawkBearPig/GLM-5.3-Flash-NVFP4-FP8](https://huggingface.co/HawkBearPig/GLM-5.3-Flash-NVFP4-FP8) | 2, 4 | [Two nodes](deploy/cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json), [four nodes](deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json) |
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | 2, 4 | [Two nodes](deploy/cluster_qwen-3.8-flash-next_fp8_w2.example.json), [four nodes](deploy/cluster_qwen-3.8-flash-next_fp8_w4.example.json) |
| Qwen3.8-Flash-Next | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | 1, 2 | [One node](deploy/cluster_qwen-3.8-flash-next_nvfp4_w1.example.json), [two nodes](deploy/cluster_qwen-3.8-flash-next_nvfp4_w2.example.json) (mapped n-gram table, dense projections FP8 at load) |
| Qwen3.8-Flash-Next | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | 1, 2 | [One node](deploy/cluster_qwen-3.8-flash-next_nvfp4-radixark_w1.example.json), [two nodes](deploy/cluster_qwen-3.8-flash-next_nvfp4-radixark_w2.example.json) (same NVFP4 format and engine configuration as the NVIDIA release) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) | 4 | [Four nodes](deploy/cluster_glm-4.7_nvfp4_w4.example.json) |
| GLM-5.3 | [HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64](https://huggingface.co/HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64) | 4 | [Four nodes](deploy/cluster_glm-5.3_int4-int8_w4.example.json) |
| DeepSeek-V4.1-Flash | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 4 | [Four nodes](deploy/cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json) |
| MiMo-V2.6-Flash | [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 2, 4 | [Two nodes](deploy/cluster_mimo-v2.6-flash_mxfp4-fp8_w2.example.json), [four nodes](deploy/cluster_mimo-v2.6-flash_mxfp4-fp8_w4.example.json) |

The Qwen NVFP4 templates select streaming MMA for the FP8 vocabulary head
with `engine.fp8_head: "mma"`, following matched one- and two-Spark
[real-checkpoint numerical validation](benchmarks/results/2026-09-21-qwen-fp8-head-numerics.md),
including the YaRN template. `--fp8-head gemv` restores the previous head
path. See [operation and numerical constraints](docs/operations.md).

The Qwen NVFP4 templates use `engine.ngram_table: "mmap"` to read the
n-gram table from NVMe, `engine.decode_graph: true` for resident graph serving,
and `engine.dense_weights: "fp8"` to encode dense projections at load. On two
Sparks, mapping saves 23.84 GiB per rank while staying within 3.6% of resident
decode throughput and 2.9% of resident prefill time in the matched campaign.
Use `--dense-weights checkpoint --fp8-head gemv` to retain the checkpoint's
BF16 dense stack.
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

MiMo-V2.6-Flash is served as shipped — MXFP4 routed experts, fp8 block-128
dense projections, BF16 `o_proj` / head / `eh_proj` (resident in their
lossless 12-bit form under `engine.bf16_weights: "bf12"`) — with its hybrid
sliding-window (128, sink-biased) / global attention, one MTP draft layer
and the checkpoint's own chat template and `<tool_call>` format. The vision
and audio encoders in the checkpoint are not loaded; text prompts only. The
two-node template keeps its K/V cache in the fp8 row form
(`engine.kv_dtype: "fp8"`, 58 KiB per token per rank against 109 in BF16)
for a 256K-token pool; `--kv-dtype bf16` restores the BF16 cache. Decode
runs the attention as one launch per layer with the residual add fused
into each norm; prefill attention is query-tiled on the tensor cores (each
K/V tile staged — and under the fp8 cache dequantized — once per 64 query
vectors). See the [model card](docs/model_cards/MiMo-V2.6-Flash.md) and
[plan](docs/mimo_v26_flash_plan.md). Native blocks 0/1/2 MTP3 and prefill
work reductions are opt-in; see [MiMo operations](docs/operations.md#mimo-native-mtp-and-prefill-options)
for settings, memory implications and measured workload tradeoffs.

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
- **Faster DeepSeek decoder selection**: parallel scoring and exact radix
  selection reduce the time spent choosing attention entries, especially at
  long context. Scores, tie rules and selected entries are preserved; existing
  recipes use the improvement automatically. See the
  [comparison and validation](benchmarks/results/2026-09-21-deepseek-selection/README.md).
- **Row-aware tensor-core execution and grouped prefill**: dense kernels select
  their lowering from the active row count, and queued cold prompts can share a
  forward pass while retaining request-local attention and state.
- **Lossless 12-bit BF16 weights (`engine.bf16_weights`)**: the BF16 matrices
  a decode step streams are kept in a 12-bit form — the sign and mantissa
  byte plus a 4-bit exponent code per weight, exact side tables for the rare
  outliers — and the kernels rebuild the exact BF16 bits in registers. Outputs
  are bit-identical (greedy transcripts do not change) from 25% fewer bytes:
  +6–12% single-stream decode on GLM-5.3-Flash, GLM-4.7, the full GLM-5.3 and
  Qwen3.8-Flash-Next-FP8. The 12-bit form can also be the only resident one,
  which puts the model below its checkpoint's footprint.
- **Model-specific prefill paths**: packed int4/int8 tensor-core prefill for
  full GLM-5.3, tiled QSA and calibrated NVFP4 W4A4 expert prefill for Qwen,
  and bounded grouped prefill for DeepSeek-V4.1-Flash. Qwen can optionally
  yield between prefill chunks so active decodes continue making progress.
- **Opt-in 512K context for Qwen3.8-Flash-Next** with `engine.rope_scaling`
  (YaRN): the [two-Spark NVFP4 template](deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json)
  supports a 524288-token request ceiling, with 5/5 retrieval probes passing
  at both 261K and 522K prompt tokens. See the [validation record](benchmarks/results/2026-09-20-qwen-yarn512k.md)
  and [release-check procedure](docs/qwen_yarn_release_check.md).
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
- **Image inputs**: PNG/JPEG/WebP data URIs in Chat Completions for GLM-5.3-Flash
  and Qwen3.8-Flash-Next, using each checkpoint's native vision encoder.
  Multiple images, streaming and MTP work together, with image-aware prefix
  caching; see [image inputs](docs/vision.md) for the geometry each family's
  processor uses, examples and memory requirements.
- **Deterministic across ranks**: admissions journaled from the head, every
  tick's operation-stream digest checked on every peer, and all ranks'
  complete streams compared at shutdown.
- **Process-failure detection**: a rank process exiting fails the service
  within seconds. Silent node loss is detected by the bus watchdog.
- **Deployment and monitoring**: a shared cluster config, versioned
  releases, periodic throughput logs and JSON counters at `/metrics`
  (also available at `/v1/metrics`), including
  [decode graph batch and padding counters](docs/operations.md#decode-graph-batch-counters)
  and [MTP acceptance counters](docs/openai-compatibility.md#speculative-decoding-counters).
  Startup checks the memory plan before allocation. Cache capacity is configurable,
  with BF16, FP8 or FP4 latent storage for GLM-5.3.

## Performance

See [Benchmarks](docs/benchmarks.md) for current serving throughput, cold
prefill latency, quality scores and long-context measurements. The document
identifies the measured source revision, hardware, workload and timing scope,
and links to the raw results and reproduction commands.

## Status

As of 2026-09-22, the source tree has eleven measured deployment templates
covering six model architectures on one, two or four Sparks. The shared
engine provides graph decode, transactional MTP, row-batched execution,
grouped prefill, prefix caching, deterministic multi-rank scheduling and the
OpenAI-compatible service. Current quantized paths cover FP8, NVFP4, MXFP4 and
full GLM-5.3's packed int4/int8 format. Qwen NVFP4 runs on one or two Sparks by
mapping its n-gram table from NVMe and encoding the dense stack to FP8 at load.

The Qwen sixteen-slot decode graphs and prefill continuation are implemented as
opt-in controls; the shipped Qwen templates retain four slots and monolithic
admission. DeepSeek ships at six slots with DSpark depth 4, confidence
scheduling, bounded grouped prefill and the stream-ordered eager collective.
Full GLM-5.3 ships at eight slots and uses packed tensor-core prefill from 128
rows. MiMo-V2.6-Flash ships at four slots with MTP depth 1 on two or four
Sparks. Version 0.1.0 remains the original GLM-5.3-Flash sign-off release; the
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

For an x86 Linux build workstation, use the Docker-based
[Spark cross-build](docs/cross-compiling.md); run the resulting binaries on a Spark.

Run the guided setup on rank 0:

```bash
./scripts/setup.sh
```

The wizard helps you choose a supported model and node count, saves your site
settings, checks SSH and software dependencies on every node, and walks through
RoCE lane selection for multi-node deployments. It then builds the release
server, prepares the downloader, downloads the checkpoint once and syncs peers,
and runs the serving preflight. Existing deployment tuning and unrelated `.env`
entries are preserved. Reruns reuse the build and complete cached checkpoints.

Python 3.10+ is needed to run setup. On Ubuntu/DGX OS, the wizard can install
standard system packages with `sudo`; `--install-system-deps` requests this
up front. NVIDIA drivers/CUDA and physical network setup remain site prerequisites.
Expect substantial checkpoint storage and download time on a fresh machine.
The wizard reports each node's free cache space and explains the lane choices;
it cannot verify cabling or end-to-end RDMA connectivity.

Setup prints the commands to start, inspect and stop the selected deployment.
Add `--start` to launch after all checks pass:

```bash
./scripts/setup.sh --start
```

The API defaults to localhost and has no authentication or TLS. Wait for `READY`
before sending requests. For unattended setup, read-only checks, offline cache
sync and the manual walkthrough, see [Getting started](docs/getting-started.md).
Use `./scripts/setup.sh --help` for all options.

### Send a request

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
python3 scripts/dgpp-cluster down --config /path/to/the/selected/deployment.json
```

## Release and install

A release is a versioned tarball installed once per node. Starting an
installed release stages the configuration and uses the installed binary.

```bash
CONFIG=/path/to/the/selected/deployment.json        # use the path printed by setup
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
| `DGPP_NODES` | Space-separated SSH/control hostnames or IPv4 addresses in rank order, using management IPs, fabric IPs, or a mixture. Rank 0 runs locally, must reach peers over SSH, and must be reachable by peers at its listed address. A deployment uses the first `world_size` entries. See [network layouts](docs/networking.md#ssh-and-control-addresses). | required |
| `DGPP_SSH_USER` | Peer login for binary staging, process control, diagnostics and checkpoint sync. Needs SSH-key access and write access to the configured directories. | current login when empty or absent |
| `DGPP_CLUSTER_CONFIG` | Default deployment filename, saved by guided setup. An explicit `--config` takes precedence. | legacy four-node Flash hybrid filename |
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
| `engine.max_concurrency` | no | Maximum actively executing requests, not TCP connections or queued requests. More slots can improve aggregate throughput but use more state/scratch memory and may increase per-request latency. Allowed range is 1–16, subject to model/MTP row limits below. | 8 |
| `engine.kv_capacity` | no | Shared context-token pool on each rank, across active requests—not a separate allowance for every request. A prompt and its generated answer must fit. Increase for longer contexts or more simultaneous context; memory use increases and allocation is rounded to model block boundaries. | 8192 tokens |
| `engine.kv_dtype` | no | GLM-5.3 latent-cache precision: `bf16`, `fp8`, or `fp4`; the MiMo-V2.6-Flash K/V cache takes `bf16` or `fp8` (e4m3 rows with one scale per head row, half the bytes). Lower precision reduces cache storage at a numerical-accuracy cost; it does not quantize model weights. The GLM index cache stays FP8. Qwen, GLM-4.7 and DeepSeek K/V caches remain BF16. | `bf16` |
| `engine.embed_sharding` | no | Full GLM-5.3 and DeepSeek embedding/head placement: `replicated` keeps the full table on every rank; `vocab` keeps each rank's vocabulary slice and folds token lookups. The full-GLM template uses `vocab` to save 1.33 GiB/rank; the DeepSeek template retains `replicated`. Other families ignore it. | `replicated` |
| `engine.default_max_tokens` | no | Answer-token budget for requests that omit `max_tokens`. Clients may supply their own value; this is not a global hard limit. A larger default also reserves more context space under full admission. | 256 |
| `engine.queue_limit` | no | Maximum requests waiting for an execution slot or memory budget. Additional arrivals receive HTTP 503 `overloaded`. Increase to tolerate bursts, at the cost of longer waits—not higher execution capacity. | 64 |
| `engine.max_connections` | no | Maximum simultaneously open HTTP connections on rank 0, including idle keep-alive connections and streams. Excess connections receive 503. Size this separately from active request slots. | 64 |
| `engine.prefix_cache_gib` | no | Memory budget per rank for reusable prefix-state snapshots. Long documents can reuse an earlier snapshot when their question changes. Snapshot slots and the KV token pool are separate limits; see [sizing and recipe capacities](docs/prefix-cache.md). Set 0 to disable. Entries evicted from memory are lost unless the NVMe cache below is enabled. | 1.5 GiB |
| `engine.admission` | no | When to reserve context space. `full` reserves prompt plus the requested answer budget before admitting a request. `grow` starts with a smaller reservation and extends it during generation; if space runs out, the youngest request is shed. Use `full` for predictable reservations, `grow` to trade that guarantee for denser occupancy. | `full` |
| `engine.admission_window` | no | Answer-token reservation increment used by `grow` admission. Larger increments reduce growth frequency but reserve more space ahead of use. Has no effect under `full`. Must be positive. | 256 tokens |
| `engine.prefill_budget_tokens` | no | Maximum prefill tokens per scheduler tick. -1 selects an aligned budget near 256 on supported Qwen and GLM-5.3-Flash graph engines, with cancellation and decode between chunks. Positive values override it; 0 explicitly keeps full-prompt admission. Other engines retain full-prompt admission. | -1 (automatic) |
| `engine.prefill_idle_budget_tokens` | no | Larger prefill budget when no request is actively decoding. Requires an enabled busy budget, must be at least that budget, aligned and within the same prefill limit. Rechecked after each chunk. 0 uses the busy budget for all chunks. | 0 (disabled) |

### NVMe cache

The `nvme_cache` block is a top-level deployment key, beside `engine`. It keeps
prefix-cache entries that memory evicts — the snapshot and the KV blocks it
pins — in one preallocated slab file per rank on local NVMe, and restores an
entry before a later matching request prefills its suffix. Startup checks the
directory, the free space and the minimum size on every rank and refuses to
boot when they do not fit; `dgpp-serve --memory-plan` runs the same check.
See [the cold tier](docs/prefix-cache.md#the-nvme-cold-tier) for what is
stored, how much a token takes and how retention works.

```json
"nvme_cache": {
  "enabled": true,
  "path": "~/dgpp/nvme-cache",
  "capacity_gib": 64
}
```

| Key | Required | Purpose and when to change it | Default |
|---|---|---|---|
| `nvme_cache.enabled` | no | Turns the cold tier on. Requires `engine.prefix_cache_gib` above 0 (the tier holds what the arena evicts) and a positive `capacity_gib`. Rank 0's value is applied on every rank. | false |
| `nvme_cache.path` | no | Directory for the slab file, created at startup if missing; each rank writes `rank<N>.slab` under it on its own node, so choose a directory on every node's local NVMe. The slab is recreated at every start (entries are not reused across restarts). | `~/dgpp/nvme-cache` |
| `nvme_cache.capacity_gib` | when enabled | Size of each rank's slab in GiB. Retention evicts the least recently used entries so the slab never grows past it. Must hold at least one entry at the request context limit plus its snapshot (startup names the number) and must fit the filesystem's free space beside a 1 GiB reserve. Every rank runs rank 0's value. | 0 |
| `nvme_cache.min_tokens` | no | The shortest entry worth keeping cold, in prompt tokens. Shorter entries stay memory-only. 0 selects one prefill chunk. | 0 |

### Execution and performance

These settings change how the model runs. Use the matching deployment template
first; change one setting at a time and measure the effect on your workload.

| Key | Required | Purpose and when to change it | Default |
|---|---|---|---|
| `engine.decode_graph` | no | Use CUDA graph replay for decode to reduce CPU launch overhead. On one node, this selects resident graph serving instead of the eager streaming path. Required for MTP and the single-Spark Qwen templates; leave enabled for the documented serving configurations. | false |
| `engine.mtp` | no | Enable multi-token prediction: draft candidate tokens, then verify them with the main model. Can reduce decode time when drafts are accepted, but adds draft state and verification work. Requires `decode_graph`; not a larger request batch. | false |
| `engine.mtp_depth` | no | How many tokens to draft per speculative step, 1–5. Greater depth can accept more tokens per pass, but uses more verification rows and can waste work when drafts are rejected. Values above 1 require MTP; supported batching varies by model. DeepSeek's shipped template uses depth 4. | 1 |
| `engine.mtp_schedule` | no | Enable confidence-scheduled verify depth for a model with a confidence head. Greedy DeepSeek requests verify only the leading drafts whose survival probability justifies another row; sampled requests keep the configured depth. Unsupported for Qwen C16/MTP3: leave this false at sixteen slots and depth 3 because additional verification depths exceed the graph-variant limit. | false |
| `engine.mtp_schedule_row_ms`, `engine.mtp_schedule_base_ms` | no | Cost model for scheduled verification: milliseconds for another verify row and fixed work per pass. These values are deployment measurements; retain the DeepSeek template values unless re-profiling that world. | 8.0 / 28.0 |
| `engine.mtp_schedule_lambda`, `engine.mtp_schedule_min_depth`, `engine.mtp_schedule_adapt` | no | Floor for the value of decode time in tokens/ms, minimum verified draft depth, and whether the value adapts from committed tokens and modeled time. Lambda 0 derives the reservation rate. | 0 / 1 / true |
| `engine.prefill` | no | DeepSeek prefill mode: `bounded` runs every prompt row through the encoder and only the final window through the decoder; `exact` runs every layer over every row for parity work. Other families ignore it. | `bounded` |
| `engine.ngram_table` | no | Qwen n-gram embedding-table placement. `resident` keeps it in device-accessible memory; `mmap` leaves it on local NVMe and fetches needed rows through the host page cache. The Qwen NVFP4 templates use `mmap`; the two-Spark campaign measured at most 3.6% lower decode throughput for 23.84 GiB less planned model memory per rank. DeepSeek's Engram tables use their own mapped checkpoint sidecar. | `resident` |
| `engine.dense_weights` | no | Qwen dense-projection storage. `checkpoint` retains the checkpoint's BF16 form; `fp8` converts dense projections at load time to reduce their memory footprint, with quantization error. Does not select another HF repository or change the expert quant; other families ignore it. | `checkpoint` |
| `engine.fp8_head` | no | Qwen FP8 vocabulary-head dispatch. `mma` uses streaming MMA above the dense GEMV threshold and within the configured decode capacity, including short prefills in that interval. Requires `dense_weights: "fp8"`; the NVFP4 templates select it after [matched numerical validation](benchmarks/results/2026-09-21-qwen-fp8-head-numerics.md). `gemv` restores the previous dispatch. | `gemv` (NVFP4 templates: `mma`) |
| `engine.bf16_weights` | no | Resident form of the BF16 weights that decode streams (attention/linear-attention projections, the LM head). `checkpoint` keeps the BF16 bytes alone. `bf12` and `bf12+bf16` add a **lossless** 12-bit form of each matrix (sign+mantissa byte plus a 4-bit exponent code; exact side tables for the rare outliers): decode launches of up to eight rows read 0.75 of the bytes and produce bit-identical results, so transcripts do not change. This is a storage format, not quantization. `bf12+bf16` keeps both forms resident: prefill is untouched and the 12-bit copies cost about 0.75× those matrices in additional memory (GLM-5.3-Flash: +2.0 GiB/rank at four Sparks, +3.8 at two; GLM-4.7: +4.9). `bf12` keeps the 12-bit form **alone**: each matrix's BF16 bytes are returned to the node as its layer loads, the footprint drops below `checkpoint`'s (GLM-5.3-Flash −0.5 GiB/rank at four Sparks, −1.0 at two; GLM-4.7 −1.4), and a prefill GEMM expands the rows it reads into a small scratch first — the same bits, so the same results, for about 10 ms per prefill chunk on four-node GLM-5.3-Flash (+1–2% on 8K–32K prompts, +25–40 ms to a short prompt's first token). The memory plan includes either. Measured single-stream decode: GLM-5.3-Flash +6–7%, GLM-4.7 +11–12%, full GLM-5.3 +5–6.5%, Qwen3.8-Flash-Next-FP8 +6.4% on four Sparks and +9.0% on two ([benchmarks](docs/benchmarks.md)). On Qwen it packs the GDN/QSA projections and the head of the FP8 checkpoint and always keeps both forms resident (every Qwen recipe has the room); under `dense_weights: "fp8"` those matrices are already FP8 and nothing is packed. DeepSeek accepts the key and currently packs nothing. Templates with memory to spare ship `bf12+bf16`; the two sized to their nodes' memory (two-node GLM-5.3-Flash, full GLM-5.3) ship `bf12`. | `checkpoint` |
| `engine.graph_batch_min_live` | no | Active-request count at which decode switches from scalar to batched graphs. A lower threshold starts batching earlier; batching may improve throughput while doing extra padded-row work. 0 chooses min(2, `max_concurrency`); explicit values must be 1 through `max_concurrency`. | 0 (automatic) |
| `engine.sampling_candidates` | no | Number of candidate tokens gathered per rank on the sampled-token fast path, 1–256. Smaller values reduce routine work but may trigger more full-gather fallbacks. The fallback preserves sampling correctness; this is not the client's `top_k` parameter. | 128 |
| `engine.bulk_pace_gbps` | no | Sender pacing rate per queue pair for bulk/prefill communication, in gigabits per second. Negative derives the rate from link speed; 0 disables pacing. Override only when measuring network contention—this is not an API throughput limit. | -1 (automatic) |
| `engine.bulk_inflight` | no | Maximum in-flight bulk stripes per lane. More can keep the link busy but increase pressure on buffers and competing traffic. -1 uses the transport default (4); normally leave automatic. | -1 (automatic) |
| `engine.rendezvous_timeout_ms` | no | How long ranks may take to join the transport rendezvous. Increase for slow cold starts or delayed peers; it does not increase an HTTP request's timeout. | 120000 ms |
| `engine.stats_interval_s` | no | Interval between periodic server throughput/state log lines. Shorter intervals give finer operational visibility and more log output. Set 0 to disable periodic statistics. | 10 seconds |
| `engine.reasoning_in_content` | no | Put reasoning text in the response's `content`, separated by the model's `</think>` marker, instead of a separate `reasoning_content` field. Use only for clients that need that combined format; it does not disable reasoning. | false |
| `engine.no_eos` | no | Ignore the model's end-of-sequence token so measurement runs continue to their token budget. Leave false for normal serving, where a model should be allowed to finish its answer. | false |

The application allows up to sixteen request slots. A speculative request uses
`1 + mtp_depth` physical decode rows. GLM-5.3-Flash supports eight batched
rows; full GLM-5.3 supports sixteen; GLM-4.7 and DeepSeek support
thirty-two; Qwen supports sixty-four (sixteen requests at MTP depth 3). Graph families capture the slot prefixes that fit, with scalar
fallback for an unsupported active shape. The startup log reports the selected
capacity and rejects a configuration that cannot fit its required rows.
Qwen's 17-64-token decode walks use kernel-only BF16 lowering, including
BF16 projections retained by FP8-dense storage. Smaller walks retain their
existing dispatch even when MTP expands a token into multiple matrix rows.
The shared 8/12-slot graph
families also affect other models above eight slots and can reduce the number
of scheduled verification depths that fit; see [operations](docs/operations.md).

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
