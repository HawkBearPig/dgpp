# Configuration

Start with a [deployment template](../deploy/README.md) for your model and
number of nodes. Keep its settings for the first run: templates often override
the program defaults to fit the model in memory and improve performance.

Put settings shared by your deployments, such as node addresses and cache
directories, in the repository's `.env` file. Put settings for one model,
such as request limits and execution options, in its deployment JSON.
Restart the deployment after changing either file.

## Find the setting you need

| I want to… | Start here |
|---|---|
| Choose nodes, network addresses or disk locations | [Site settings](#site-settings) |
| Change the API address, model name or streaming behavior | [Deployment and endpoint](#deployment-and-endpoint) |
| Accept larger file uploads or keep them across restarts | [File inputs](#file-inputs) |
| Serve longer prompts or more simultaneous requests | [Request capacity and memory](#request-capacity-and-memory) |
| Reuse work from earlier prompts | [Prefix caching](#prefix-caching) |
| Keep evicted prefix entries on local NVMe | [NVMe cache](#nvme-cache) |
| Keep an existing compatible NVMe cache across server upgrades | [`nvme_cache.reuse_binding`](nvme-cache.md#across-restarts) |
| Balance prompt processing against smooth streaming | [Prefill scheduling](#prefill-scheduling) |
| Change model storage, precision or generation speed | [Execution and performance](#execution-and-performance) |
| Extend a supported Qwen model's context limit | [Extended context](#extended-context) |
| Tune communication or server logs | [Transport and logging](#transport-and-logging) |
| Inspect the settings passed to each server process | [How settings are resolved](#how-settings-are-resolved) |

A few terms used below:

- A **rank** is one server process, normally on one DGX Spark. Rank 0 runs
  the API and coordinates the other ranks. **World size** is the rank count.
- **Prefill** is the model's processing of a prompt before it generates an
  answer. **Decode** is the subsequent generation of answer tokens.
- A **token** is a piece of text used by the model. The **KV cache** stores
  model state for the tokens in active requests. **Context** includes both
  the prompt and the generated answer.
- A **checkpoint** is the set of files containing the model. Its **weights**
  are the learned numerical values used to compute answers.
- **Throughput** is the total work completed per second. A setting that
  improves throughput can still increase **latency**, the time an individual
  request or generation step takes.
- **Speculative decoding** uses a smaller draft computation to propose tokens,
  then checks them with the main model. Accepted drafts can speed up generation.

Names such as `engine.kv_capacity` describe nested JSON fields, not literal
keys containing dots. For example, merge this fragment into the deployment
JSON to change its shared token pool, keeping the other fields already in
its `engine` object:

```json
{
  "engine": {
    "kv_capacity": 32768
  }
}
```

This is a settings fragment, not a complete deployment. The tables below give
defaults when a setting is omitted; check your template for explicit overrides.
Unless marked required, settings are optional. Units are stated in each table.
MiB and GiB are binary memory units.

## Site settings

Scripts load `.env` through [site_env.py](../scripts/site_env.py); shell
scripts use [cluster_env.sh](../scripts/cluster_env.sh). Start from
[.env.example](../.env.example). Set `DGPP_ENV_FILE` in your environment to
read another file. Exported settings override values in that file.
Values are literal, optionally quoted: shell commands and variable references
are not expanded. The helper reads only supported settings and does not copy
credentials such as `HF_ACCESS_TOKEN` into the generated server configuration.

### Nodes and network

Multi-node deployments use RDMA, which lets nodes transfer data directly
between memory regions. RoCE is RDMA over Ethernet. The device and address
settings below select the network used for those transfers. A lane is one
of the parallel network paths used between nodes.

| Setting | What it controls and what to expect | Default |
|---|---|---|
| `DGPP_NODES` | Space-separated node hostnames or IPv4 addresses, in rank order. A deployment uses the first `world_size` entries. Rank 0 runs locally, must reach peers over SSH, and must be reachable by peers at its listed address. Management addresses, fabric addresses or a mixture can work; see [network layouts](networking.md#ssh-and-control-addresses). | Required |
| `DGPP_SSH_USER` | SSH login used on peers to copy binaries and checkpoints, start and stop processes, and collect diagnostics. It needs SSH-key access and permission to write to the configured directories. | Current login if empty or omitted |
| `DGPP_CLUSTER_CONFIG` | Deployment file used when a command has no explicit `--config FILE`. Guided setup saves its selection here. Relative paths start at the repository root. | `deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json` |
| `DGPP_HTTP_PORT` | API port on rank 0. Changing it changes the port clients connect to. A deployment's `http.port` takes precedence. | 18080 |
| `DGPP_HTTP_BIND` | IPv4 address on which rank 0 accepts API connections. A deployment's `http.bind_host` takes precedence. Setup's `--http-bind` option sets this without prompting. | `0.0.0.0` |
| `DGPP_FABRIC_PORT` | Rank 0's TCP port for setting up communication between nodes. Peers must reach it; API clients do not use it. | 29970 |
| `DGPP_JOURNAL_PORT` | Rank 0's TCP port for sending ordered work to peers. It must differ from the fabric and API ports. Keep both internal ports private to the cluster. | 29971 |
| `DGPP_ROCE_DEVICES` | One or two local RDMA device names, in lane order. These are device names reported by `discover_roce.py`, not Linux network interface names. Match lane count and subnet order across nodes so they can communicate. | Discover active Ethernet RDMA devices |
| `DGPP_ROCE_GID_INDICES` | One RoCE-v2 network-address index per explicitly selected RDMA device, in the same order. Set these when a device has multiple networks and automatic selection chooses the wrong one; discover values separately on each host. | Automatic RoCE-v2 selection |
| `DGPP_NODE_OVERRIDES` | JSON map of per-node settings, keyed by the exact names in `DGPP_NODES`. Use it when peers need different device names or cache directories. See [.env.example](../.env.example) and [per-node settings](networking.md#per-node-differences). | None |

### Directories and caches

| Setting | What it controls and what to expect | Default |
|---|---|---|
| `DGPP_LOG_DIR` | Base directory on rank 0 for logs, process records and collected peer logs. The launcher adds a subdirectory for each deployment. | `~/dgpp/log` |
| `DGPP_STAGE_DIR` | Base directory on peers for copied development binaries, runtime settings and logs. The launcher adds a deployment subdirectory. Model downloads use a separate cache. | `/tmp/bus4` |
| `DGPP_RELEASE_DIR` | Persistent directory on each node for installed server versions, stored as `dgpp-VERSION`. The SSH login must be able to write here. | `~/dgpp/releases` |
| `DGPP_BUILD_DIR` | Build directory used by scripts. Relative paths start at the repository root. Leave unset to keep serving and testing builds separate. | `build-release` for serving and packaging; `build-ci` for tests, or `build-<preset>` when the test script uses `DGPP_PRESET` |
| `DGPP_DATA_DIR` | Directory for evaluation data. Relative paths start at the repository root. | `data` |
| `HF_HUB_CACHE` | Directory containing downloaded Hugging Face checkpoints. Choose a disk with room for the full checkpoint on each node. Overrides `HF_HOME/hub`. | `HF_HOME/hub`, otherwise `~/.cache/huggingface/hub` |
| `HF_HOME` | Base Hugging Face cache directory. DGPP uses its `hub` subdirectory when `HF_HUB_CACHE` is unset. | `~/.cache/huggingface` |
| `DGPP_RESIDENT_CACHE_DIR` | Disk cache of prepared, per-rank weight images that make later starts faster. It uses additional disk space alongside the downloaded checkpoint. Overrides deployment `paths.resident_cache`. | `~/.cache/dgpp/resident` |

Stage and release directories must be absolute or begin with `~/`, without
spaces or shell syntax. `~/` expands on each node. Log and stage subdirectories
depend on the deployment file's path; `scripts/dgpp-cluster paths --config FILE`
prints the effective locations. An explicit `--log-dir` is used as-is.

Experimental and diagnostic environment switches are described in
[operations](operations.md). Normal model tuning belongs in the deployment JSON.

## Deployment and endpoint

A deployment JSON needs `model` and `world_size`. The other fields in this
section are optional. Select it with `--config FILE`; wrapper scripts pass
that choice to their child commands. The [deployment catalogue](../deploy/README.md)
lists the supported model and node combinations.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `model` | Hugging Face repository ID, such as `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`. Selects the checkpoint, tokenizer and chat template. Use a repository ID, not a local path or a generic quantization name. | Required |
| `world_size` | Number of participating ranks. Documented deployments use 1, 2 or 4 nodes, taking that many entries from `DGPP_NODES`. More ranks divide the model's work and memory across nodes. Startup fails if too few nodes are listed, the model cannot be split across that many ranks, or memory is insufficient. | Required |
| `release` | Installed DGPP software version to run on every node. Use it to upgrade or roll back the server; it does not select a model revision. Versions live under `DGPP_RELEASE_DIR/dgpp-VERSION`. `--release` overrides this field; `--bin` overrides binary selection. | Empty: use the development build |
| `http.bind_host` | IPv4 address that accepts API connections on rank 0. `127.0.0.1` accepts local connections, a LAN address selects that interface, and `0.0.0.0` listens on all IPv4 interfaces. This does not select the RDMA network. The server provides no authentication or TLS. | `DGPP_HTTP_BIND`, otherwise `0.0.0.0` |
| `http.port` | API port, from 1 to 65535. Change client URLs when changing it. Must differ from the fabric and journal ports in a multi-node deployment. | `DGPP_HTTP_PORT`, otherwise 18080 |
| `http.max_body_bytes` | Largest HTTP request body accepted, in bytes; must be positive. Larger values allow bigger documents or histories but do not raise the model's token limit. Oversized requests receive HTTP 413 before tokenization. Memory grows with received data; the full limit is not allocated for every connection. `--http-max-body-bytes` overrides it. | 268435456 (256 MiB) |
| `http.stream_flush` | How quickly ready output reaches a streaming client. `pass` checks during the regular 25 ms poll; `wake` also wakes the HTTP loop as soon as output is ready. `wake` can reduce delivery delay at the cost of more CPU work and socket sends. `--stream-flush` overrides it. | `pass` |
| `http.sse_ping_interval` | Seconds of stream silence before a keep-alive comment is sent, including during queueing and prefill. This helps keep idle streams open through clients and proxies; it does not extend engine deadlines. Accepts 1–2147483647, or `-1` to disable. The server flag `--sse-ping-interval` overrides the file; a request's `sse_ping_interval` overrides the server setting. | 30 seconds |
| `tokenizer.word_cache_entries` | Number of short text pieces whose token IDs rank 0 caches across requests, from 0 to 10000. Repeated text may tokenize faster; mostly unique text pays extra cache overhead. `0` disables the cache. It uses bounded host memory and preserves token IDs. | 0 |
| `paths.resident_cache` | Deployment-specific directory for prepared weight images, which speed later starts and consume extra disk space. Prefer the site setting unless this deployment needs a separate cache. | `~/.cache/dgpp/resident`; `DGPP_RESIDENT_CACHE_DIR` takes precedence |
| `ports.metrics` | Metrics port on each peer's own node address, from 1 to 65535; `0` disables it. Lets monitoring collect each peer's step, prefill and memory-pool measurements. Rank 0 includes its own measurements at `/metrics/prometheus`. Must differ from fabric and journal ports. This is the only `ports` key allowed in a deployment template. See [per-rank metrics](openai-compatibility.md#per-rank-families). | 0 (off) |
| `engine.model_alias` | Model name advertised by `/v1/models`, responses and health status. Clients must send this exact name in their `model` field or receive HTTP 404 `model_not_found`. The loaded checkpoint is still selected by `model`. | Empty: use the `model` repository ID |
| `engine.reasoning_in_content` | Include reasoning in response `content`, separated by the model's `</think>` marker, rather than in `reasoning_content`. Enable for clients that need the combined format. Reasoning itself remains enabled. | false |

The word cache is configured through JSON only and starts empty after a restart.
Longer text pieces bypass it. It speeds tokenization, not model prefill or
decoding. See [tokenization](tokenization.md#optional-word-cache) for entry-size
and memory limits. Long-word tokenization is already bounded without this cache.

To reduce stream delivery delay, merge `"stream_flush": "wake"` into the
deployment's existing `http` object. This applies to all requests on rank 0;
requests cannot override it. Maintenance polling and keep-alive timing remain
in place. Use the [HTTP output delivery procedure](http-stream-flush.md) to
compare client latency, CPU use and throughput.

### Request preparation

Request preparation always uses a fixed pool on rank 0, with one thread per
configured `engine.max_concurrency` slot. There is no separate worker-count
setting. Workers sleep when idle, and ready requests proceed to admission
without waiting for earlier preparations. When every worker is occupied,
later requests wait in the bounded preparation queue. Preparation and file
completions wake HTTP in both stream-flush modes.

Queued, running and completed preparations together are limited to
`engine.max_concurrency + engine.queue_limit` requests and a 1 GiB retained
payload budget. Exhaustion returns HTTP 503 `preparation_busy`. The byte budget
includes request bodies, prepared tokens/images and allowances for per-choice
metadata. Reservations remain charged through generation and response
completion; temporary parser/frontend scratch and the existing file store
are additional. Increasing engine concurrency also increases possible preparation
CPU and memory demand.

### File inputs

`engine.file_inputs` is an optional object controlling uploads through
`POST /v1/files` and files attached to chat requests. All limits are positive
integers. These limits are separate from the HTTP body limit and the model's
token capacity; a request must satisfy all applicable limits.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.file_inputs.directory` | Upload storage directory on rank 0. Set a persistent directory to keep uploaded files across restarts. Empty uses a temporary store for the server process. | Empty |
| `engine.file_inputs.pdf_command` | PDF-to-text executable, run directly without a shell. Install `poppler-utils` on rank 0 for the default command. PDF processing fails if the executable is unavailable. | `pdftotext` |
| `engine.file_inputs.max_file_bytes` | Maximum size of one uploaded or inline file. Raise it to accept larger individual files, with more storage and processing work. | 52428800 (50 MiB) |
| `engine.file_inputs.max_request_bytes` | Maximum combined file size in one chat request. Raise it to allow more or larger attachments in the same request. | 52428800 (50 MiB) |
| `engine.file_inputs.max_text_bytes` | Maximum extracted text, applied both per file and across a request. Limits how much text preprocessing may produce; exceeding it rejects the input. | 134217728 (128 MiB) |
| `engine.file_inputs.max_storage_bytes` | Maximum total bytes in the upload store. New uploads are rejected when they would exceed it; increase it or delete unused files to make room. | 1073741824 (1 GiB) |
| `engine.file_inputs.pdf_timeout_ms` | Maximum time for one PDF extraction. Larger values allow slower documents but can occupy a preprocessing worker for longer. | 120000 ms |
| `engine.file_inputs.workers` | Number of preprocessing threads, from 1 to 256. More workers can process file requests in parallel but use more CPU and memory. This does not add model execution slots. | 4 |

See [file inputs](openai-compatibility.md#file-inputs) for supported formats and
request examples.

## Request capacity and memory

Four separate limits matter: open connections, queued requests, active request
slots, and tokens stored in the shared KV pool. Increasing one does not increase
the others. Startup checks the combined memory plan before loading the model.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.max_concurrency` | Maximum actively executing requests, from 1 to 16, subject to the model and speculation limits below. More slots can improve total throughput but need more working memory and can increase each request's latency. | 8 |
| `engine.kv_capacity` | Shared context-token pool on each rank, used by all active requests. Prompt and answer tokens both consume space. Increase for longer contexts or more simultaneous context, at the cost of memory. Allocation is rounded to model block boundaries. This does not extend the model's positional limit. | 8192 tokens |
| `engine.default_max_tokens` | Answer-token budget when a request omits `max_tokens`. Clients can set their own value. Under `full` admission, the prompt plus this budget must fit the context allowance or the request is rejected. Templates generally use much larger defaults to allow long answers and reasoning. | 256 tokens |
| `engine.queue_limit` | Maximum requests waiting for a slot or memory. Further arrivals receive HTTP 503 `overloaded`. A larger queue absorbs bursts but allows longer waits without adding processing capacity. | 64 |
| `engine.max_connections` | Maximum open HTTP connections on rank 0, including idle connections and streams. Excess connections receive HTTP 503. Size this separately from active request slots. | 64 |
| `engine.admission` | When context memory is reserved. `full` reserves the prompt plus requested answer budget before starting: an agent client that asks for 32K tokens per call (opencode) pins prompt plus 32K per live session, about 160K tokens for five sessions, whatever the answers use, and that reservation rather than the prompts is what evicts idle sessions' cached prefixes from a pool sized for the prompts. `grow` reserves less initially and expands during generation; if space runs out, the youngest request is stopped. `grow` can fit more work at once, with a risk of interrupting requests. A larger pool (`kv_dtype: "fp8"` doubles the tokens per byte on the families that support it; keep `prefill_attention: "fp8"` with it where the family supports that, or the default bf16 arithmetic widens every code first and prefill slows) or a smaller client output cap are the other ways out. | `full` |
| `engine.admission_window` | Number of answer tokens reserved at a time under `grow`; must be positive. Larger increments reserve more unused space but need fewer reservation updates. Has no effect with `full`. | 256 tokens |
| `engine.kv_dtype` | Precision of cached model state. Lower precision saves memory but can change answers; weights are unaffected. GLM-5.3 supports `bf16`, `fp8` and `fp4` for its latent cache; its index cache stays FP8. MiMo and Flint support `bf16` and `fp8`. Qwen3.8-27B supports `fp8` on both releases: the Full layers' K/V as e4m3 codes at the mixed NVFP4 release's shipped cache scales, or at unit scales on the FP8 release, which ships none (the codes are the e4m3 rounding of K and V, the common scale-free fp8 cache; 2026-10-18, measured at the bf16 cache's floor in the [FP8 model card](model_cards/Qwen3.8-27B-FP8.md)); half the Full layers' pool bytes, decode at 256K 16 % faster. Qwen3.8-Flash-Next supports `bf16` and per-token/head scaled `fp8`; GLM-4.7 keeps BF16 caches, and DeepSeek uses its architecture's own formats. | `bf16` |

For example, a shared pool of 32768 tokens can cover two requests reserving
16384 tokens each, but not two reserving 32768 each. Model limits, block
rounding and other memory use still apply.

Speculation also uses extra computation rows per request; each row represents
work for one token position. Native MTP uses
`1 + mtp_depth` rows before any lookup tail. GLM-5.3-Flash, full GLM-5.3,
GLM-4.7, MiMo, Flint and DeepSeek support up to 32 decode rows; Qwen supports up to 64.
The engine batches shapes that fit and can fall back to one request at a time
for others. Startup reports the selected capacity and rejects unsupported
configurations. See [operations](operations.md) for model-specific limits.

### Prefix caching

The prefix cache saves model state for matching beginnings of prompts. It can
avoid reprocessing a shared system prompt or an earlier part of a conversation.
Its memory budget is separate from the KV token pool.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.prefix_cache_gib` | Memory budget per rank for saved prompt state. A larger cache can retain more reusable prefixes but leaves less memory for other uses. `0` disables it. A chat turn costs about two entries, an agent call about three, and eviction is LRU without time expiry, so an idle agent session survives about slots ÷ 3 other calls. See [cache sizing](prefix-cache.md). | 1.5 GiB |
| `engine.prefix_min_tokens` | Earliest prompt position at which any snapshot may be saved. Higher values keep short probes and requests from replacing longer cached conversations. Shorter prompts can still reuse matching entries. `0` allows snapshots at every supported cut. | 1024 tokens |
| `engine.prefix_head_snapshots` | Save an extra snapshot at the first structural prompt boundary, such as the end of the system prompt. This can help a new conversation or a client that shortened its history reuse that prefix. Uses another cache entry and obeys `prefix_min_tokens`. Disabled on DeepSeek's bounded prefill path, which cannot take extra cuts. | true |
| `engine.prefix_pin_max_pct` | How much of the cache `prompt_cache_key` pins may protect, as a percentage of the arena's slots and of the NVMe slab's pages. While the pinned entries stay within it, unpinned entries are evicted first. Past it a pin protects nothing and plain least-recently-used order applies, so one client cannot pin the whole cache against everyone else. `100` lets pins always protect and `0` ignores pins. Every rank runs rank 0's value. | 50 |

See [prefix caching](prefix-cache.md#the-entry-floor-and-the-head-cut) for how
snapshot positions and reuse work.

### NVMe cache

The `nvme_cache` block is a top-level deployment key beside `engine`. It keeps
the prefix-cache entries memory evicts, the snapshot and the KV blocks it
pins, in one preallocated slab file per rank on local NVMe, and restores an
entry before a later matching request prefills its suffix. Startup checks the
directory, the free space and the minimum size on every rank and refuses to
boot when they do not fit, and `dgpp-serve --memory-plan` runs the same check.
[The cold tier](nvme-cache.md) describes what is stored, how entries move and
how retention works.

```json
"nvme_cache": {
  "enabled": true,
  "path": "~/dgpp/nvme-cache",
  "capacity_gib": 64
}
```

| Key | What it controls and what to expect | Default |
|---|---|---|
| `nvme_cache.enabled` | Turns the cold tier on. Requires `engine.prefix_cache_gib` above 0 (the tier holds what the arena evicts) and a positive `capacity_gib`. Rank 0's value is applied on every rank. | false |
| `nvme_cache.path` | Directory for the slab files, created at startup if missing. Choose one on every node's local NVMe. Each rank keeps `<path>/<binding>/rank<N>.slab` on its own node, where the automatic binding names the build version, the checkpoint, the deployment's effective settings and the slab geometry, with the index and the page checksums beside it. Another model or configuration served from the same node gets its own binding directory. Startup names other bindings' slabs left in the directory and their size. Remove them to reclaim the disk. | `~/dgpp/nvme-cache` |
| `nvme_cache.capacity_gib` | Size of each rank's slab in GiB. Retention evicts the least recently used entries so the slab never grows past it. Must hold at least one entry at the request context limit plus its snapshot (startup names the number) and must fit the filesystem's free space beside a 1 GiB reserve. Every rank runs rank 0's value. | 0 |
| `nvme_cache.min_tokens` | The shortest entry worth keeping cold, in prompt tokens. Shorter entries stay memory-only. `0` keeps every entry memory keeps, which `engine.prefix_min_tokens` already floors. | 0 |
| `nvme_cache.chunk_mib` | One staging slice in MiB, from 4 to 4096. Two slices of device work move per scheduler tick, so this paces a spill or restore against the decode step (128 MiB per tick at the default) and sizes the staging buffers in the memory plan (4 × chunk + 8 MiB). A slice must hold one cache block. Every rank runs rank 0's value. | 64 |
| `nvme_cache.max_ops` | Spills and restores started but not yet committed by the world, from 1 to 256. A started entry is busy, never evicted, until its commit, so a higher bound protects more freshly taken entries on a slow device at the cost of slots the arena cannot reclaim meanwhile. Every rank runs rank 0's value. | 4 |
| `nvme_cache.reserve_gib` | Free space the pre-flight leaves on the slab's filesystem, in GiB. Startup refuses a capacity that would cut into it. | 1 |
| `nvme_cache.persist` | Keep the slab across restarts. A start under the same binding (the same build version, checkpoint, effective settings and slab geometry) adopts the previous run's entries when every rank holds the same index, so switching models and back finds the earlier model's entries, while a build upgrade starts fresh by default. Use `reuse_binding` to select existing entries across upgrades. Under another binding, and with `false`, the slab starts fresh. Each entry is bound to the checksums of its pages when it was written: an entry whose pages a later run reused before a crash, or whose token ids are unreadable, is dropped at startup rather than restored. Entries with image inputs and spills in flight at shutdown are not kept. | true |
| `nvme_cache.reuse_binding` | Select an existing 16-character lowercase hexadecimal binding directory across build or settings changes. Requires `enabled` and `persist`. Rank 0 sends this selection to every rank; it must exist on each node. The operator asserts that model artifacts and cached-state interpretation remain compatible. Format, geometry, checksum and world-agreement checks remain active; rejected reuse refuses startup without clearing the selected cache. See [reuse across upgrades](nvme-cache.md#across-restarts). | empty (automatic binding) |
| `nvme_cache.allow_buffered` | Allow the slab on a filesystem that refuses direct I/O (tmpfs, some network filesystems). Off, startup refuses such a path, because buffered writes grow the page cache into memory the model plans on. For tests and experiments only. | false |
| `nvme_cache.op_timeout_s` | Give up on a spill or restore that makes no progress for this many seconds (a stuck device or filesystem): the op fails, a request waiting on the restore prefills cold, and the staging buffer the stuck call holds returns when the call does. `0` waits forever. The stall is logged at 30 s and every minute regardless. Every rank runs rank 0's value. | 120 |

## Prefill scheduling

Prompt processing runs in chunks so other requests can make progress between
chunks. Larger chunks can process prompts more efficiently; smaller chunks
give active streams and new arrivals a chance to run sooner.

By default, the scheduler measures chunk cost and adjusts chunk size against
two soft latency targets. These targets are not deadlines: long prompts, busy
slots and memory pressure can make requests miss them. Set both targets to
`0` to use fixed budgets; the startup line names the policy in force
(`adaptive: …` or `fixed budgets (both latency targets 0)`). While requests
decode, the controller takes their own pass out of the allowance and charges
a chunk only for what it adds (since 2026-10-22; before that a small chunk
looked dear per token and the limit could settle at 128 tokens beside
several readers). On the mixed-prefill families, every serving family, the
busy budget and the adaptive limit are one amount per pass shared by every
prompt reading in it, not a per-prompt figure. For several concurrent agent
sessions with long prompts, compare the defaults against fixed budgets
(`1024` busy, `2048` idle, both targets `0`): the field evidence for that load
is fixed. See [tuning automatic prefill](optimizing_performance.md#tuning-automatic-prefill)
for examples and measurement guidance.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.prefill_budget_tokens` | Prompt work allowed while other requests are decoding. `-1` chooses automatically, starting at `prefill_initial_tokens`. A positive value sets an upper limit; adaptation may choose less. `0` disables budgeted interleaving during admission, so a long prompt can delay other work. Chunk sizes also obey the engine's alignment and capacity limits. One amount per pass, shared by every prompt reading in it on the mixed-prefill families: four readers under `1024` get about 256 tokens each per pass. Size it for the readers together. | -1 (automatic) |
| `engine.prefill_idle_budget_tokens` | Maximum prompt work per chunk when idle. `0` chooses up to 4096 tokens with automatic busy budgeting, or the busy budget when that is explicit. A positive value must be aligned and at least the busy budget. Larger idle chunks can finish a lone prompt sooner but delay a newly arriving request. | 0 (derive from busy policy) |
| `engine.decode_gap_target_ms` | Soft target for time between a request's generation steps that produce tokens. Lower values favor smoother streaming by limiting competing prefill work, potentially slowing prompt processing. The decoders' own pass, learned from ticks that only decode, comes out of this allowance before any prompt token does; `scheduler.prefill_chunks.decode_pass_ms` shows it. `0` disables this target. | 1000 ms |
| `engine.first_token_target_ms` | Soft target from arrival to first token, including queue time. Lower values favor responsiveness to arrivals and may reduce prompt-processing throughput. `0` disables this target. | 3000 ms |
| `engine.prefill_initial_tokens` | Starting automatic chunk size before timing measurements exist; must be positive. Larger values can improve initial prompt throughput but make the first chunk take longer. Adjusted to alignment, capacity and the idle ceiling. | 256 tokens |
| `engine.prefill_idle_floor_tokens` | Minimum automatic chunk size after measuring cost, when nothing is decoding or waiting for a free slot. Keeps a lone long prompt from being split into inefficient tiny chunks. Higher values can delay new arrivals by one larger chunk. Rounded down to alignment and capped by the idle ceiling; `0` uses the minimum alignment. | 1024 tokens |
| `engine.prefill_latency_headroom_pct` | Percentage of the time allowance left unused to absorb timing variation, from 0 to 50. More headroom selects smaller chunks and reduces the chance of missing a target, at a throughput cost. | 15% |
| `engine.prefill_growth_factor` | Maximum increase in chunk size relative to measured work, from 2 to 4. Larger values try bigger chunks sooner but can overshoot the latency target. | 2 |
| `engine.prefill_cost_recovery_pct` | Weight given to a newly observed lower cost, from 1 to 100. Larger values grow work sooner after conditions improve; smaller values recover cautiously. Higher observed costs take effect immediately. | 25% |
| `engine.prefill_early_admission` | Allow queued text to join ongoing prefill and share execution with generation. This can reduce arrival waits. Omitted resolves from the engine's support for mixed work; all serving families support it. `false` admits queued requests using only the budget left after advancing ongoing prefill. A zero prefill budget processes admissions without budgeted interleaving regardless. | Automatic; enabled on supported serving engines |
| `engine.admission_gather_ms` | How long an idle engine waits after the first arrival to collect a burst, from 0 to 1000 ms. A short wait can make requests start together and improve batching, while adding that delay to a lone arrival. Applies only with no queued or running work. `0` starts immediately. | 3 ms |
| `engine.prefill_group` | Process compatible prompts that have no cached state together. This can improve throughput when requests arrive in a burst. `false` processes them separately, avoiding numerical changes that very short prompts can see when grouping selects a different matrix computation. | true |

Automatic chunk sizes also follow the model's efficient processing sizes where
possible. Their actual size may therefore differ from the requested starting
value. The idle floor applies only after a cost measurement; the first chunk
still uses `prefill_initial_tokens`.

## Execution and performance

Use these settings after choosing capacity and latency limits. Many apply only
to a particular model. Change one at a time and measure the result on your
workload; [optimizing performance](optimizing_performance.md) explains how.

Choose [graph execution](#graph-execution-and-batching),
[weight storage](#weight-storage-and-placement),
[prefill precision](#prefill-computation-and-precision),
[attention precision](#attention-precision),
[speculative decoding](#speculative-decoding), [extended context](#extended-context)
or [advanced kernel tuning](#advanced-kernel-tuning).

BF16, FP8 and FP4 are number formats with different precision and storage
costs. **Quantization** stores values at lower precision to save memory or
work, and can change answers. The lossless `bf12` format is different: it
compresses BF16 weights without changing their values. A **head** is the
model's final computation that scores possible next tokens; **experts** are
specialized parts of the model selected to process particular tokens.
**Activations** are intermediate values passed between model computations.
**Tensor cores** are GPU hardware that accelerates matrix multiplication.

Even without quantization, different chunk and batch sizes can change
floating-point rounding. Claims about identical outputs apply to the validated
execution shape. Use [numerical checks](numerics.md) when changing that shape;
see also [accuracy and correctness](../README.md#accuracy-and-correctness).
Measured results belong in [benchmarks](benchmarks.md); they are not guarantees
for every workload.

### Graph execution and batching

A CUDA graph records GPU work for reuse, reducing the CPU work needed to
launch each generation step. Batching runs several requests together.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.decode_graph` | Replay recorded GPU work during generation to reduce launch overhead. On one node this also selects the engine that keeps model weights resident rather than streaming layers. Required for native MTP. Retain the value in the matching serving template. | false |
| `engine.decode_engine` | The decode engine. `graph` is the engine that serves today. `step` is the persistent-step engine under development ([step_engine.md](step_engine.md)), which selects its own weight layout with it; a template that sets it is refused at boot until a family port exists, and every rank of a world runs the same engine. | `graph` |
| `engine.step_sms` | The SMs the step engine's streamer takes for its workers, from 0 to 256. `0` takes every SM but the chain's. Read by the step engine alone. | `0` |
| `engine.l2_staging_mib` | The step engine's lead: the L2 the stream may run ahead of the chain by, from 0 to 24 MiB. `0` disables the lead. Tuned per family against the consumers' own L2 use. Read by the step engine alone. | `8` |
| `engine.stream_governor` | The step engine's governor: the workers that stream the lead at once, from 0 to 256. `0` leaves it unbounded. Read by the step engine alone. | `0` |
| `engine.graph_batch_min_live` | Active-request count at which decode switches from individual to batched graphs. Lower values start batching sooner, which may improve throughput but can compute unused padding rows. `0` chooses `min(2, max_concurrency)`; explicit values must be 1 through `max_concurrency`. | 0 (automatic) |
| `engine.compact_batches` | Use the smallest compatible recorded batch for active requests, reducing unused work when some slots are empty. Some computation paths retain padding for numerical consistency; block drafters and confidence-based schedules keep their physical slot mappings. See [compact graph batches](operations.md#compact-graph-batch-mappings). | true |
| `engine.sampling_candidates` | Number of likely tokens gathered per rank on the fast sampling path, from 1 to 256. Smaller values do less routine work but may need more expensive full-gather fallbacks. Fallback preserves the sampling distribution; this does not limit the client's `top_k`. | 128 |

### Weight storage and placement

These settings affect model memory, startup work and the amount of data read
during generation. They do not select a different checkpoint unless stated.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.bf16_weights` | Storage for supported BF16 weights. `checkpoint` keeps the original form. `bf12` uses a lossless 12-bit form, reducing weight reads and usually memory use, but prefill must unpack it. `bf12+bf16` keeps both forms for faster decode without unpacking during prefill, using extra memory. Values and decode results are preserved. Qwen retains both forms even with `bf12`; already-FP8 weights are unaffected. DeepSeek-V4.1 and DeepSeek-V4 pack their lm heads (the tensor-core decode form reads the 12-bit companion; the bf16 bytes stay resident for prefill). See the model notes in [operations](operations.md). | `checkpoint` |
| `engine.dense_weights` | Qwen's dense-projection storage. `checkpoint` keeps the shipped formats. `fp8` converts supported BF16 projections to FP8 at load time, saving memory and weight reads but introducing quantization error that can change answers. On Qwen3.8-27B this includes the BF16 vocabulary head; its templates enable it. Expert quantization is unchanged. Other families ignore this setting. | `checkpoint` |
| `engine.attention_weights` | Full GLM-5.3 only: `checkpoint` keeps shipped attention projections; `int4` converts selected projections to 4-bit weights at load time. This saves memory and generation bandwidth but is lossy and can change answer quality. Other families ignore it. Validate quality before using it. | `checkpoint` |
| `engine.embed_sharding` | Full GLM-5.3 and DeepSeek embedding-table placement. `replicated` keeps the full table on every rank. `vocab` divides vocabulary rows among ranks, saving memory but adding communication for token lookups. Other families ignore it. | `replicated` |
| `engine.ngram_table` | Qwen3.8-Flash-Next's n-gram table, which supplies embeddings for token sequences. `resident` keeps it in device-accessible memory. `mmap` reads needed rows from local storage through the host page cache, saving resident memory but adding storage access that can slow requests. DeepSeek's Engram table uses its own mapped sidecar. | `resident` |
| `engine.ngram_table_model` | Hugging Face `ORG/NAME` repository supplying the n-gram table when the main Qwen checkpoint omits it, as with the AutoRound hybrid. Only table tensors come from this repository. It must be available in the model cache. | Empty: use the main checkpoint's table |
| `engine.engram_source` | Hugging Face `ORG/NAME` repository supplying DeepSeek-V4.1's Engram tables when the checkpoint omits them, as the EXL3 re-pack does: the base release (`deepseek-ai/DeepSeek-V4.1-Flash-1017`). Only the table tensors come from this repository, and its Engram sidecar serves when the checkpoint has none. It must be available in the model cache. Refused for other families. | Empty: use the checkpoint's own tables |
| `engine.ngram_prestage` | With Qwen's `ngram_table: "mmap"`, and with DeepSeek-V4.1's Engram tables (always mmap'ed), fetch the next prefill chunk's table rows while the current chunk runs: the next chunk's rows are hashed from its host ids behind the current chunk's last tokens and gathered on a side stream; the next chunk takes them when its identity matches. Overlapping the work reduces waiting without changing values (a 32K DeepSeek-V4.1 prefill had 110–160 ms of the gather exposed per 4,096-token chunk). `false` uses sequential staging. | true |
| `engine.fp8_head` | Qwen FP8 vocabulary-head computation: `gemv` processes rows in chunks; `mma` uses a streaming tensor-core computation. `mma` can improve performance and uses the same head reduction across row counts, but does not remove numerical differences earlier in the model. Requires `dense_weights: "fp8"`; NVFP4 templates select `mma`. | `gemv` |
| `engine.mtp_expert_format` | Qwen MTP draft-expert checkpoint layout. Use `fp8` for the NVIDIA release's separate FP8 experts, or `bf16_fused` for the RadixArk release's combined BF16 tensors, which are converted to FP8 at load time. The Qwen3.5 MoE release uses `bf16` for separate BF16 draft experts, converted to block FP8 at load time. This changes draft proposals; the target still verifies them. Match the checkpoint layout so the loader can read it. Other families ignore it. | `fp8` |

### Prefill computation and precision

These settings change how prompts are processed. Options that reduce precision
can shorten the wait for the first token, but can also change the answer.
Keep the template's choices unless you have measured both speed and quality.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.prefill` | DeepSeek-V4.1 prefill mode. `bounded` processes the whole prompt in the encoder and only the final window in the decoder, following the model's serving recipe. `exact` processes every row through every layer for reference comparisons and costs more work. Other families ignore it. | `bounded` |
| `engine.dory_loops` | Dory1.0-4B: override the recurrent block's loop count. The exact path leaves this unset and runs the checkpoint's own `n_recurrent_loops` (2); the block's weights are shared, so more loops cost time, not memory. Naming a count changes what the engine computes. Other families refuse it at startup. | the checkpoint's value |
| `engine.dory_kv_cache` | Dory1.0-4B: the recurrent K/V policy. `per_loop` gives every loop of the recurrent block its own paged K/V planes — the exact path. `last_loop` shares one plane per layer and keeps only the last loop's rows: the pool at half the planes, and the earlier loops' history gone (lossy; one cold prefill is unaffected, decode and prefix reuse of a mid-loop state are not). Other families refuse it at startup. | `per_loop` |
| `engine.prefill_dense` | `auto` selects the family policy: the weight tile for DeepSeek-V4.1, the existing streaming/large-row BF16 bridge for Qwen MXFP8. Explicit `tile` and `streaming` select either form. The DeepSeek-V4.1 prefill walk's dense projections (the attention's q/kv/o projections, the indexer's, the Engram projection, the draft's main projection): `tile` runs them on the fp8-weight tile GEMM, `streaming` on the decode rows' streaming form, which re-reads each weight once per 128 rows and ran a prompt's 4096-row chunks at a quarter of the tile kernel's rate. Both place the same bf16(e4m3 × scale) weights and accumulate in fp32 in different orders: equal within rounding, not bitwise, so a prefill row's value depends on the form, never on the chunk cut. Measured on DeepSeek-V4.1-Flash-1017 on four nodes (2026-10-18): cold prefill 0.480 / 1.415 / 4.045 / 15.328 s at 512 / 2K / 8K / 32K tokens against 0.514 / 1.572 / 4.715 / 17.916 streaming (−14 % at 8K and 32K); decode untouched. Qwen3.8 MXFP8 attention and shared-expert prefill can also select the tile path (resident E8M0 or F32 scales); `streaming` retains their existing streaming/large-row BF16 bridge path. Decode dispatch is unchanged. Other weight forms ignore the key. | `auto` |
| `engine.prefill_fold_overlap` | The DeepSeek-V4.1 prefill walk's boundary folds overlapped with compute: a walk of 512 rows or more on a multi-rank world runs each layer as two row blocks (the boundary a row of the request's block grid nearest the middle), so every all-reduce of one block lands under the other block's attention, mHC site, router or expert accumulation, and block B's MoE fold under the next layer's block A. Bitwise the one-block walk (the attention takes the second block as it takes any chunk; the MoE's phased form runs the same kernels per row). Under the bounded prefill the encoder layers overlap; the decoder's replay segments are shorter than a block pair. Measured on DeepSeek-V4.1-Flash-1017 on four nodes (2026-10-18): see the model card. Other families keep their own overlap (GLM, DeepSeek-V4). | `true` |
| `engine.decode_pdl` | The decode graph's kernel-to-kernel edges as programmatic dependencies: a kernel's blocks are scheduled as its predecessor's blocks exit and wait, before their first read, for the predecessor's completion, so each launch's ramp overlaps the previous drain. Only kernels that carry the wait take the edge (the registry in `engine/graph_pdl.hpp`); others keep plain edges. Arithmetic unchanged. `false` leaves every edge plain. | `true` |
| `engine.prefill_window` | The DeepSeek-V4.1 prefill walk's window attention (the sliding window over the ring and the chunk's own rows): `flash` runs it on the flash listed attention kernel the main source uses, which reads a tile's keys once into shared memory per slab; `scalar` runs the partial kernel the decode window runs, which reads them once per thread. The two sum each row's window scores in different fp32 orders: equal within rounding, not bitwise (the model test gates the forms against each other; the decode audit walks the scalar form). The scalar kernel held 12 % of a 32K prefill. Other families ignore the key. | `flash` |
| `engine.decode_window` | The DeepSeek-V4.1 decode step's window attention (the 128-key sliding window over the ring): `flash` runs it on the flash listed attention kernel, split eight ways like the scalar form (one 16-token tile per block), on a side stream beside the main source's kernel (the two read disjoint caches and write disjoint partials; the finish joins them); `scalar` runs the scalar partial kernel split eight ways in line, the form until 2026-10-19. The two differ in the window's fp32 summation order, the rounding class the decode audit certifies. The DSpark draft block's rows take the same form (before 2026-10-20 they kept the scalar kernel: 131–170 µs a draft layer). | `flash` |
| `engine.collective_transport` | The fabric's transport for the latency-class collectives (the decode step's recorded folds and the pick's stream collectives): `write` lays every row out as 512-byte self-describing chunks (31 lanes of bf16 and a trailer with the chunk's hash and generation), RDMA-writes it into each peer's write ring, and folds it with a grid of warps as the chunks prove themselves — no door cells, claim rounds, acks or credits; `send` is the form until 2026-10-19 (payload and doorbell SEND pairs claimed by one block, hash-gated, acked and credited). Both fold bitwise alike; the prefill's bulk collectives and the eager one-shot are unaffected. | `write` |
| `engine.prefill_bf16_partials` | Qwen3.8 AutoRound hybrid: store intermediate expert outputs in BF16 instead of FP32 during prefill. This halves their memory traffic and can speed prefill, with extra rounding that can change answers. The AutoRound template enables it. | false |
| `engine.prefill_fold_scales` | Qwen3.8 AutoRound hybrid: apply expert quantization scales to BF16 weights before multiplication. This changes rounding and can change answers. It is a performance experiment; recorded measurements found no speed gain and templates leave it off. | false |
| `engine.prefill_fp8_gemm` | Qwen with `dense_weights: "fp8"`: use FP8 tensor cores for eligible block-FP8 dense prefill projections instead of expanding their weights to BF16 first. Smaller projections and unsupported scale grids retain their existing computation paths. Quantizes activations and can change answers. Independent of KV-cache and attention precision. The AutoRound template enables it. | false |
| `engine.prefill_fp8_per_tensor` | Qwen3.8-27B: make extra FP8 weight copies with one scale per matrix and use them for prefill. Can speed matrix multiplication but requires substantial additional memory, about 23 GiB at the 27B shape. The added quantization can change answers, including greedy completions. | false |
| `engine.prefill_activations` | Qwen3.8-27B prefill activation precision. `bf16` keeps intermediate inputs in BF16 and expands quantized weights for multiplication. `native` uses the checkpoint's FP8 or mixed NVFP4/FP8 activation recipe, allowing faster tensor-core work without repeated weight expansion. `native` can change answers through rounding; the 27B templates enable it. Decode keeps BF16 activations. Other families accept only `bf16`. | `bf16` |

See the [Qwen3.8-27B NVFP4 model card](model_cards/Qwen3.8-27B-NVFP4.md)
and [AutoRound notes](qwen38_autoround_int4_plan.md) for measured tradeoffs.

### Attention precision

`engine.kv_dtype` selects cache storage; the settings below independently select
attention computation during prefill and decode. Setting the cache to `fp8`
alone keeps BF16 attention arithmetic. Selecting FP8 attention uses cached
E4M3 values directly for the query/key and probability/value products, with
FP32 softmax and accumulation. The additional query and probability
quantization can change answers.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.prefill_attention` | Qwen3.8-27B and Qwen3.8-Flash-Next prefill attention arithmetic. Accepts `bf16` or `fp8`. `fp8` requires `kv_dtype: "fp8"`; other families reject the `fp8` value. The 27B templates, NVFP4 and FP8, enable it. Decode uses its own setting. | `bf16` |
| `engine.decode_attention` | Qwen3.8-Flash-Next QSA decode arithmetic, including MTP draft and verification rows. Accepts `bf16` or `fp8`. `fp8` requires `kv_dtype: "fp8"`; other families accept only `bf16`. Independent of `prefill_attention`. Long selected-token lists can benefit, while short single-row decode can be faster with `bf16`. | `bf16` |

Flash-Next supports BF16 or FP8 caches; FP4 is rejected. Native FP8 attention
requires the model's 256-element attention heads. It works with MTP and prefix
caching. Attention settings do not select weight precision or change dense
projection and expert computation; they do not require `dense_weights: "fp8"`.
Flash-Next's `prefill_activations` remains `bf16`; `native` is a separate
Qwen3.8-27B option.

For example, this Flash-Next fragment enables FP8 cache storage and native
prefill attention while keeping BF16 decode arithmetic:

```json
{
  "engine": {
    "kv_dtype": "fp8",
    "prefill_attention": "fp8",
    "decode_attention": "bf16"
  }
}
```

Set `decode_attention` to `fp8` to evaluate native decode as well. Set both
attention options to `bf16` to use FP8 storage with BF16 computation. Flash-Next
templates retain BF16 cache storage and arithmetic by default. See
[Flash-Next measurements](qwen_fp8_kv_plan.md#validation-2026-10-17) for the
kernel timings and full-checkpoint smoke comparison; these do not establish
end-to-end serving gains or accuracy across workloads.

### DeepSeek-V4.1 index budget

On every compressed-KV layer DeepSeek-V4.1-Flash attends only the entries
its indexer selects: `index_topk` of them per query, chosen on the reindex
layers from a candidate pool of `candidate_topk_blocks` blocks of
`candidate_block_size` entries that the candidate layer scores once per
query. The checkpoint ships 512 entries out of a 2048 × 8 pool. These keys
replace those values for a deployment; every rank applies the same budget
(the config digest and the settings record carry it). A wider budget shows
each query more of its context — the lever against a sparse attention
missing what a long input holds — and changes the model's answers, not only
its speed: the released weights were trained at 512. The DeepSeek-V4.1 family
alone takes these keys; other families refuse them.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.index_topk` | Compressed-KV entries each query attends on a compressing layer: a power of two up to 2048, the select networks' bound. Above the checkpoint's value every query reads more cache rows and the selection ranks a wider list; below the visible entry count (short prompts) nothing changes. Measured on DeepSeek-V4.1-Flash-1017 at 2048 against 512 (2026-10-18, four nodes, the same binary): short-context decode level, cold prefill 9 % (2K) to 48 % (32K) longer, decode at 32K–512K of context 3–8 % slower per pass, quality unchanged; the 1017 template ships the checkpoint's 512. | 0: the checkpoint's (512) |
| `engine.candidate_topk_blocks` | Blocks the candidate layer keeps per query for the reindex layers to select among: a power of two up to 2048. Blocks × block size must hold at least `index_topk` entries. | 0: the checkpoint's (2048) |
| `engine.candidate_block_size` | Entries per candidate block, 1 to 1024. Scored as a block (its maximum), kept or dropped whole. | 0: the checkpoint's (8) |

### Speculative decoding

Native multi-token prediction (MTP) uses the model's draft layers. DFlash2
uses a separate draft checkpoint. Both propose tokens that the main model
checks before returning them. Speculation can speed generation when enough
drafts are accepted; rejected drafts add work.

**Greedy** generation always picks the most likely next token. **Sampled**
generation draws tokens using the client's temperature and other sampling
settings. A draft can use a different selection rule from the final answer.

Draft-only changes affect acceptance and speed while verification preserves
the main model's output probabilities. Sampled answers can still differ from
run to run. These controls do not change the client's sampling settings.
Greedy-output comparisons still need the same validated execution shape,
as described above.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.mtp` | Enable native MTP drafting and verification. Can reduce generation time but uses draft state and extra computation. Requires `decode_graph: true`; cannot be combined with `dflash_model`. | false |
| `engine.mtp_depth` | Native MTP tokens proposed per step, from 1 to 5. More drafts can produce more accepted tokens per pass, but use extra verification rows and can waste work. Values above 1 require MTP. DeepSeek templates set their supported block depth explicitly: V4.1 uses 4 and V4 uses 5. | 1; when omitted with DeepSeek MTP, use the checkpoint's DSpark block size |
| `engine.mtp_draft` | Draft choice for sampled requests. `sampled` draws from the draft distribution; `greedy` proposes its most likely token. Both use exact verification but can have different acceptance rates. `auto` selects greedy for DeepSeek-V4-Flash and DeepSeek-V4.1-Flash (their DSpark block's argmax out-proposes a draw from it, and only greedy slots take the scheduled verify depth) and sampled for other native families. DFlash2 uses sampled for `auto`. Greedy client requests are unaffected. | `auto` |
| `engine.mtp_draft_temperature` | Draft sampling temperature as a multiple of the client's temperature, greater than 0 and at most 4. Lower values make proposals more concentrated; higher values spread them out. This changes draft acceptance, not the target output distribution. Applies to sampled MTP and DFlash2 proposals. | 1.0 |
| `engine.mtp_verify` | Verification rule for sampled drafts. `token` checks them one at a time. `block` checks them jointly and can accept more tokens, while preserving the target distribution. It falls back to token verification where the block rule cannot be used. Greedy requests are unaffected. | `token` |
| `engine.mtp_head` | Full GLM-5.3 draft vocabulary head. `checkpoint` uses the main head's storage form; `fp8` adds an FP8 copy at load time to reduce draft-head reads. Uses extra memory and can change acceptance, while the main verifier stays unchanged. Other families ignore it. | `checkpoint` |
| `engine.draft_head` | How supported draft heads search the vocabulary. `exact` scores all tokens. `screen` first uses a small approximate copy to select candidate blocks, then scores those blocks with the served weights. This reduces draft work but can miss a useful proposal; verification still uses the full head. DeepSeek-V4 has its own screened head. | `exact` |
| `engine.draft_head_blocks` | Maximum candidate blocks examined with `draft_head: "screen"`, from 1 to 64. Each block has 128 vocabulary entries. More blocks cost more work but reduce the chance of missing a useful draft. | 32 |
| `engine.draft_vocab` | Qwen3.8 AutoRound hybrid only: path to a one-dimensional int32/int64 `.npy` file of token IDs the draft may propose. A smaller vocabulary reduces draft-head reads but can lower acceptance. Main-model verification is unchanged. [build_draft_vocab.py](../tools/build_draft_vocab.py) builds a candidate set. | Empty: whole vocabulary |

#### Adaptive verification depth

Instead of always checking every draft, the scheduler can stop at the depth
whose expected accepted tokens justify its cost. These are advanced tuning
settings; use your deployment's measured values.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.mtp_schedule` | Choose how many leading drafts to verify from their predicted acceptance and the cost settings below. Requires graph execution with MTP or DFlash2. Supports DSpark, DFlash2 and native MTP confidence estimates. Sampled requests use it with greedy proposals; drawn proposals keep their configured depth. Can avoid wasted verification work. Cannot be combined with `dflash_batch_rows` or `lookup_tail`. Qwen at 16 slots and MTP depth 3 exceeds the available graph variants; leave it off there. | false |
| `engine.mtp_schedule_row_ms` | Estimated milliseconds for one verification row; must be greater than 0. Higher row cost makes extra drafts less attractive. Set from measurements of the deployment: a sweep of fixed depths gives the row cost as the slope of the pass time and the base as its intercept (DeepSeek-V4.1-Flash-1017 on four nodes, 2026-10-18: 5.17 ms per row on a 31.1 ms base; the template carries 5.2 / 31.1). | 8.0 ms |
| `engine.mtp_schedule_base_ms` | Estimated fixed milliseconds per pass, including draft work; must be nonnegative. Higher fixed cost makes accepting several tokens in one pass more valuable. Used with row cost to estimate throughput. | 28.0 ms |
| `engine.mtp_schedule_lambda` | Value assigned to decode time, in tokens/ms. Higher values demand more expected benefit before checking another draft. `0` derives an initial value as `1 / (base_ms + row_ms)`; a positive value supplies the floor for adaptation. | 0 (derived) |
| `engine.mtp_schedule_min_depth` | Fewest drafts verified per scheduled step, from 1 to 5 and no more than the draft width. Higher values force more checking even when confidence is low. | 1 |
| `engine.mtp_schedule_options` | At most this many scheduled verify depths, from 0 to 6: the engine spreads them evenly between `engine.mtp_schedule_min_depth` and the draft width, always keeping the full block (three on a 4-draft block are 1, 3 and 4). `0` takes every depth the graph-variant budget holds. Fewer options round a policy depth up to the next one, never down: the DeepSeek EXL3 template pins three because its confidence under-reads the third row on code, where the depth-2 option cost 8 % of the tokens a step (2026-10-23). | `0` |
| `engine.mtp_schedule_adapt` | Update the value of decode time from committed tokens and modeled pass cost. This helps depth follow the workload and concurrency. `false` uses the fixed configured or derived value. | true |
| `engine.mtp_schedule_sampled_scale` | Scale predicted acceptance for sampled requests using greedy drafts, from 0 to 1. Smaller positive values are more conservative and tend to verify fewer drafts. `0` disables depth scheduling for those requests and verifies the full block. | 0.93 |

#### DFlash2 draft model

These controls apply when `engine.dflash_model` is set. The eager engine runs
without a full decode graph on one node; the graph engine is required for
multi-node DFlash2. Options marked eager-only do not tune the graph path.
On the Qwen3.8-Flash-Next family (`qwen4_exp`) the recorded block draft runs
on the graph engine, and the drafter's block confidence feeds the scheduler
settings above. `engine.dflash_model` names it in a template (the
Qwen3.8-27B releases) or `--dflash-model DIR_OR_ID` selects it at boot on a
template that names MTP, for example:
`dgpp-serve --config deploy/cluster_qwen-3.8-flash-next_nvfp4_w1.example.json --dflash-model incoai/Qwen3.8-Flash-Next-DFlash2 --dflash-weights fp8 --draft-head screen --mtp-schedule`
(the drafter replaces the MTP draft, so the template's `mtp_depth`,
`mtp_schedule`, `lookup_draft`, `lookup_tail` and `draft_head` revert to
their defaults unless the command line names them). With `mtp_schedule` true,
the scheduler chooses the verification depth from the drafter's confidence
and the configured cost model. Measure those costs on the selected deployment;
acceptance and throughput depend on its target weights and workload.
Add `--draft-head screen` explicitly to screen DFlash2's proposal head;
target verification keeps the full served head. Flash-Next also uses its
recurrent-state checkpoint/replay path for DFlash2. See the
[performance notes](optimizing_performance.md#flash-next-dflash2) for the
pass-time and acceptance tradeoffs.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.dflash_model` | Qwen draft checkpoint, as a Hugging Face ID or local directory; for example, `z-lab/Qwen3.8-27B-DFlash2`, `z-lab/Qwen3.5-122B-A10B-DFlash` or `incoai/Qwen3.8-Flash-Next-DFlash2`. The architecture selects v2 or v1. Replaces native MTP, so set `mtp: false`. Verifies up to seven proposals per step and needs extra model memory. The v1 drafter still computes its full 16-row block; use BF16 draft weights and keep `mtp_schedule: false`. Multi-node use requires `decode_graph: true`; one node can use either engine. Requests with logprobs, logit bias or a grammar use plain decode. | Empty (off) |
| `engine.dflash_weights` | Draft-matrix storage. `checkpoint` keeps BF16 weights, with lossless packing where supported by `bf16_weights`. `fp8` converts them to FP8 at load time, reducing draft weight reads; some BF16 data remains for context processing. Quantization can change proposals and acceptance, while target verification remains exact. | `checkpoint` |
| `engine.dflash_verify_graph` | Eager engine: replay the multi-request verification batch as a CUDA graph to reduce launch overhead. `false` launches the packed batch directly. A lone request stays on its scalar path. | true |
| `engine.dflash_draft_batch` | Eager engine: propose drafts for active speculative requests together, reusing one read of the draft weights. `false` drafts one request at a time. Batching can improve total throughput. | true |
| `engine.dflash_depth` | Eager engine: maximum drafts verified per step, from 1 to 7; `0` checks the whole block. Lower values shorten each pass but may accept fewer tokens. The graph engine uses the whole block or its scheduled depth instead. | 0 |
| `engine.dflash_batch_rows` | Graph engine: verification-row budget for a batched step, from 0 to 4096. A smaller budget shortens each request's verified prefix when many slots are active, trading tokens per step for shorter steps. Each request also needs its input row. `0` verifies whole blocks. Mutually exclusive with `mtp_schedule`. | 0 |

#### Drafts from request history

Lookup drafting can help copy-heavy greedy requests by proposing text already
seen in the same request. Sampled requests keep their normal proposals.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.lookup_draft` | Replace native MTP proposals with a matching continuation from the request's token history when the thresholds below pass. Can improve acceptance on repeated text; scanning history adds work even when no match is used. The main model verifies the proposals. | false |
| `engine.lookup_nmin` | Minimum suffix-match length to consider, from 1 to 64 tokens. Higher values skip short, possibly accidental matches but miss some useful repeats. | 6 tokens |
| `engine.lookup_nstrong` | Match length accepted without agreement from MTP, from 1 to 64 and at least `lookup_nmin`. Higher values demand stronger evidence before replacing drafts outright. | 8 tokens |
| `engine.lookup_agree` | Number of leading proposals that a shorter match must share with MTP, from 0 to 16. Higher values require stronger agreement. `0` disables the shorter-match path. | 2 tokens |
| `engine.lookup_tail` | Extra draft positions beyond the normal MTP block. Useful for long copied spans, but every step pays for the wider computation and fewer requests may fit in a batch. Requires native MTP and a total `mtp_depth + lookup_tail` of at most 5 with the current picker. Incompatible with DFlash2 and scheduled depth. `0` adds no tail. | 0 |

### Extended context

`engine.rope_scaling` is an optional object for Qwen3.8-Flash-Next and
Qwen3.8-27B. It applies **YaRN**, a method for scaling the positional encoding
that tells the model where tokens sit in a sequence. Other families reject
this override, and the checkpoint's own positional encoding must be unscaled.

The new positional limit is `original_max_position_embeddings × factor`.
This does not enlarge the KV pool: `engine.kv_capacity` must also cover the
desired prompt and answer. Scaling changes model computations and can affect
quality; start with the [validated YaRN templates](../deploy/README.md#the-512k-yarn-templates-enginerope_scaling).
Omitting the object leaves the checkpoint's positional encoding unchanged.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.rope_scaling.rope_type` | Scaling method. Only `yarn` is supported; `type` is accepted as an alias. | `yarn` when the object is present |
| `engine.rope_scaling.factor` | Multiplier for the original context limit, at least 1. For example, 2 doubles the positional ceiling. Larger factors allow longer positions but need quality validation. | Required in this object |
| `engine.rope_scaling.original_max_position_embeddings` | Original context length in tokens; must be positive. Use the model's native limit, not the desired extended length. Both the new ceiling and the scaling curve depend on it. | Required in this object |
| `engine.rope_scaling.beta_fast` | Upper threshold defining which positional frequencies receive scaling. Together with `beta_slow`, changes the transition between retained and scaled frequencies and can affect long-context quality. Must exceed `beta_slow`; retain the validated template value. | 32 |
| `engine.rope_scaling.beta_slow` | Lower threshold of that transition; must be greater than 0 and less than `beta_fast`. Changing it changes positional behavior, not memory capacity. Retain the validated template value. | 1 |
| `engine.rope_scaling.attn_factor` | Positive multiplier on YaRN's positional-rotation magnitude, affecting attention scores and model output. It does not change the context limit. | 1 |
| `engine.rope_scaling.mrope_cache_factor` | Context multiplier used to calculate YaRN's transition band, at least 1. `4` matches the enlarged-cache reference recipe; `1` calculates the band from the native context. This changes positional frequencies, not the allocated KV pool. | 4 |

### Advanced kernel tuning

A **kernel** is a GPU computation. These settings choose how work is carried
out without intentionally changing its values. Retain defaults unless measuring
a specific bottleneck.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.expert_gemm` | Packed int4 expert matrix multiplication during prefill, used by the Qwen AutoRound hybrid and full GLM-5.3. `wide`, `wide3`, `wide4`, `wide4r` and `narrow` choose different GPU tile and staging layouts. They preserve the same results; performance depends on shape. `wide` is the measured default. | `wide` |
| `engine.expert_gemm_prefetch` | How many internal multiplication steps ahead to fetch expert weights and inputs into the GPU cache, from 0 to 16. More lookahead may hide memory waits but can compete for cache space. `0` disables it. Results are unchanged. | 3 |
| `engine.expert_tile_list` | Process only the expert tiles that contain routed work, avoiding empty grid work. Usually reduces wasted computation without changing results; disable for diagnosis. | true |
| `engine.expert_gemm_pair` | Combine the expert gate and up projections into one GPU launch. Reduces launch count without changing results, but measured performance was similar, so it remains off by default. | false |
| `engine.l2_prefetch` | Fetch upcoming layer weights into the GPU's L2 cache before use. Can reduce memory stalls, but competes with other GPU work. `false` disables this prefetcher. All ranks use the same settings. | true |
| `engine.l2_prefetch_form` | How the cache is warmed: `load` reads the bytes, `lines` requests one prefetch per 128-byte cache line, and `touch` touches one line per 64 KiB to prepare memory mappings. These do different amounts of work; compare latency on the actual model. | `load` |
| `engine.l2_prefetch_full_form` | The full-rate window's shape: `16x8` (16 blocks × 8 loads, 512 KB in flight — the DRAM rate at the light form's SM footprint, so the small kernels beside it run unstretched) or `48x4` (768 KB in flight on every SM — a window keeps its bandwidth share beside streaming kernels). Measured per family: DeepSeek-V4.1, Qwen and GLM-5.3-Flash take `16x8`; GLM-5.3 ships `48x4` (+0.3–0.8 % a pass at `16x8`, 2026-10-20). | `16x8` |
| `engine.l2_prefetch_window_mib` | Data budget for a prefetch window, from 1 to 64 MiB. Larger windows fetch more ahead but can displace useful cached data. DeepSeek-V4.1's windows take it since 2026-10-19 (its template pins 18 MiB, the best of 18/12/8/6 at C1). | 12 MiB |
| `engine.l2_prefetch_boundary_window_mib` | Prefetch-window budget beside inter-node communication, from 0 to 64 MiB. `0` uses `l2_prefetch_window_mib`. Larger windows may hide more memory work but compete with communication. | 20 MiB |
| `engine.l2_prefetch_boundary_rate` | Prefetch intensity while overlapping inter-node communication: `off`, `light` or `full`. `light` streams from 16 blocks with 2 loads in flight each (~80 % of the DRAM rate, no measurable effect on the kernels beside it); `full` from 16 blocks with 8 loads in flight (the DRAM rate; a latency chain beside it runs ~3 % longer). | `light` |
| `engine.l2_prefetch_layer_rate` | Prefetch intensity inside attention layers: `off`, `light` or `full` (the same two forms as the boundary rate). Higher intensity can reduce later waits while slowing the small computations running beside it. | `light` |
| `engine.l2_prefetch_merge` | Combine adjacent prefetch ranges into fewer launches, reducing launch and graph overhead; a hole of up to 2 MiB between two ranges is bridged only when both lie in one allocation (since 2026-10-19: the hole between two allocations is unmapped, and the load-form kernel read it). Disable when comparing prefetch behavior. | true |

## Transport and logging

These settings concern communication between ranks and operational output.
Normal deployments should retain the transport defaults unless measurements
show a problem.

| Key | What it controls and what to expect | Default |
|---|---|---|
| `engine.bulk_pace_gbps` | Sender pacing rate per RDMA queue pair for bulk and prefill traffic, in gigabits/s. Lower rates can reduce network contention but slow transfers. Negative derives a rate from link speed; `0` disables pacing. This is not an API throughput limit. | -1 (automatic) |
| `engine.bulk_inflight` | Maximum bulk-transfer stripes in flight per lane. More can keep the link busy but use more buffers and compete with other traffic. `-1` uses the transport default of 4; normally leave automatic. | -1 (automatic) |
| `engine.rendezvous_timeout_ms` | How long ranks may take to join the transport setup. Increase for slow starts or delayed peers. Does not extend HTTP request timeouts. | 120000 ms |
| `engine.stats_interval_s` | Seconds between periodic throughput and server-state log lines. Shorter intervals give finer visibility and produce more logs. `0` disables these periodic lines. | 10 seconds |
| `engine.no_eos` | Ignore the model's end-of-sequence token, which normally finishes an answer. Intended for measurements that need to run to the token budget. Leave false for normal serving so answers can finish naturally. | false |

Sampling defaults come from the checkpoint's `generation_config.json`.
Deployment penalty defaults override those values; process flags override
the deployment, and explicit request fields override the served defaults.
Omitted or null request fields inherit the served value. See
[API compatibility](openai-compatibility.md) for request settings.

| Key | What it controls | Default |
|---|---|---|
| `engine.repetition_penalty` | Multiplicative penalty for tokens already in the context, greater than 0 and at most 100. `1` disables it. | Checkpoint value, otherwise 1 |
| `engine.frequency_penalty` | Subtract this value times a token's context count, from -2 to 2. `0` disables it. | 0 |
| `engine.presence_penalty` | Subtract this value once for any token present in the context, from -2 to 2. `0` disables it. | 0 |

The corresponding process flags are `--repetition-penalty`,
`--frequency-penalty` and `--presence-penalty`. `/v1/models` reports all
three effective defaults. A request can explicitly set `1`, `0`, `0` to
disable penalties configured by the deployment.

The full host sampler, eager greedy reports and bus sampling fold skip
context counting and penalty application when the effective repetition
penalty is 1 and both additive penalties are positive zero.
Active penalties and negative-zero values retain their
arithmetic. This needs no separate enable flag. The sampling contract uses
non-NaN logits, round-to-nearest and gradual underflow; floating-point
exception flags and traps are not preserved by neutral preparation elision.
Bias, masks, selection, probability reporting and RNG handling still run.

## How settings are resolved

The launcher combines the deployment JSON with site settings and writes
`<log_dir>/cluster.resolved.json`. Each server rank reads this resolved file;
the original deployment file and `.env` are not sent to peers. To inspect it,
run this command from the repository root:

```bash
scripts/dgpp-cluster resolve --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json
```

Save that output to a file before passing it to `dgpp-serve --config` directly.
The native binary does not load `.env`. Multi-node templates need resolving;
a single-node file can use `world_size: 1` directly, meaning `localhost`.
Unknown keys and invalid types are rejected.

The precedence rules are:

1. Exported site settings override the selected `.env` file.
2. Deployment `http.bind_host` and `http.port` override their site defaults.
   `DGPP_RESIDENT_CACHE_DIR` takes precedence over `paths.resident_cache`.
3. Rank 0's binary flags after `--config` override settings from that file.
   Rank 0 sends effective shared settings to peers before model construction.

Peers use their local settings for initial connection addresses and paths, and
compare a checksum of shared settings to catch disagreements. Defaults are
defined in
[ClusterConfig](../src/serve/cluster_config.hpp), with file-input limits in
[FileInputConfig](../src/serve/file_inputs.hpp) and scheduler defaults in
[prefill_latency.hpp](../src/sched/prefill_latency.hpp). Model-specific startup
checks may further restrict a setting or supply a model default.

### Launcher-generated fields

The following fields belong to resolved JSON. The launcher generates them
from site settings and rejects them in deployment templates. Do not put
credentials in them. They are useful when inspecting a launch or invoking
the native binary directly.

| Key | Source and effect |
|---|---|
| `nodes` | First `world_size` entries of `DGPP_NODES`, in rank order. Rank 0 serves the API and internal coordination listeners. Replaces `world_size`; the two keys cannot coexist. Required by the binary unless using the single-node shorthand. |
| `ssh_user` | `DGPP_SSH_USER`, or the launcher's current login when empty. Used by the launcher to control peers; accepted but ignored by the native binary. |
| `ports.http` | `DGPP_HTTP_PORT`, default 18080. Rank 0's API listener; `http.port` takes precedence. |
| `ports.fabric` | `DGPP_FABRIC_PORT`, default 29970. Rank 0's transport-setup listener. |
| `ports.journal` | `DGPP_JOURNAL_PORT`, default 29971. Rank 0's listener for distributing ordered scheduler work. All three ports use 1–65535; fabric and journal must differ, and all three must differ in a multi-node deployment. |
| `paths.log_dir` | `DGPP_LOG_DIR`, default `~/dgpp/log`, plus a `deployments/ID` suffix. Holds rank 0's log, process record, resolved configuration and collected peer logs. |
| `paths.stage_dir` | `DGPP_STAGE_DIR`, default `/tmp/bus4`, plus the deployment suffix. Holds each peer's staged binary, configuration and logs. |
| `paths.release_dir` | `DGPP_RELEASE_DIR`, default `~/dgpp/releases`, without a deployment suffix. Contains installed `dgpp-VERSION` directories. |
| `node_env` | Site settings plus `DGPP_NODE_OVERRIDES`, one object per rank in node order. Supplies each process's device, cache and diagnostic environment settings. |

The native binary accepts these keys in `node_env`: `DGPP_ROCE_DEVICES`,
`DGPP_ROCE_GID_INDICES`, `HF_HUB_CACHE`, `DGPP_RESIDENT_CACHE_DIR`,
`DGPP_LOG_LEVEL`, `DGPP_MLOCK`, `DGPP_BUS_TIMELINE`, `DGPP_BUS_TIMELINE_DUMP` (a directory: every rank appends its collectives' stamps to `bus_gens_rank<r>.tsv` there, the cross-rank alignment's input), `DGPP_DSV41_DENSE_GEMV`,
`DGPP_DENSE_GEMV_ROWS` and `DGPP_DSV41_EAGER_FOLD`. `DGPP_NO_SWAP` is used
only by the launcher and is not written into the resolved file.

### Applying a changed configuration

`up` refuses to replace an existing deployment unless `--replace` is supplied.
Use an explicit `--config FILE` for deployment-specific commands:
`down` without it stops every recorded running deployment. Cleanup uses recorded
process identities.

During startup, the launcher prints the rank 0 log path and reports elapsed
time and the latest log line every ten seconds until ready. If rank 0 exits,
it reports the exit status and trailing logs within its two-second polling
interval, then cleans up the deployment's peers. See
[operations](operations.md) for start, stop and monitoring procedures.
