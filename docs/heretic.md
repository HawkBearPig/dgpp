# Heretic integration

Heretic owns scoring plugins, modifier algorithms, optimization studies, and
exports. DGPP supplies execution for its supported checkpoint formats and will
apply the resulting runtime artifacts alongside the original weights.

The integration provides a GLM-5.3 projection catalog and a persistent local
research worker for baseline logits, greedy generation, and last-token tensor
capture. Distributed execution, candidate installation, and serving-time
adapter loading remain to be implemented. The worker is not yet a complete
Heretic optimization backend and cannot run the full Mixed346 checkpoint on
one GB10.

## Configuration and performance isolation

Set `engine.heretic_research` to `true` in a separate research cluster config
to use the inspection command or worker. The default is `false`; existing deployment
templates are unchanged. This flag currently authorizes only the separate
research tools. It has no effect on `dgpp-serve` execution or its configuration
digest because serving performs no research operations.

Capture is a separate model walk compiled with a research template parameter.
The normal walk has no capture conditionals, copies, synchronization, or host
capture allocations. The worker checks the cluster flag before model access
or CUDA initialization. Keep future trial controls and research listeners
behind this flag. Before adding distributed execution,
carry its effective state through the rank journal and configuration digest.
Runtime adapter execution in ordinary serving will require a separate explicit
configuration opt-in. Disabled features must not allocate their buffers,
enqueue their kernels or collectives, or alter the existing cache policy.

## Inspect the target projections

Build the tool with:

```sh
cmake --preset release
cmake --build build-release --target dgpp_heretic_inspect -j 4
```

Use a resolved cluster JSON with explicit `nodes`, as produced by the cluster
launcher. A local config may instead use `world_size: 1`. Set the research flag
in that config and run this on a node that has the checkpoint:

```sh
build-release/dgpp-heretic-inspect --config /path/to/cluster.resolved.json --rank 0
```

The command resolves the configured model using the local Hugging Face cache
and the selected rank's `HF_HUB_CACHE` override. An explicit model directory
must contain `config.json`; a cache repository root is not a snapshot directory.
Inspection reads configuration and quantization sidecars, without opening
weight shards, initializing CUDA, or connecting to peers. The rank argument
selects the partition description; it does not launch that rank.

The JSON response identifies schema `dgpp.heretic.targets`, version 1, and
capability `describe_targets`. It includes the resolved checkpoint, rank,
world size, layer count, and targets in Heretic capture order. Each target
has a stable `layer/component/module_index` ID, canonical checkpoint module,
full matrix dimensions, local input-column interval, and packed weight format.
This capability does not advertise inference or adapter support.

For the full 78-layer Mixed346 checkpoint, the catalog contains 78 attention
output projections, three dense MLP down projections, and 75 shared-expert down
projections. Fused routed experts and the draft block are excluded. The shared
expert uses Heretic's logical `mlp.down_proj` component, with its distinct
checkpoint module path retained in the descriptor.

## Local baseline worker

Build `dgpp_heretic_worker` to produce `build-release/dgpp-heretic-worker`.
Use a separate config for a supported checkpoint that fits the local device:

```json
{
  "model": "/path/to/checkpoint/snapshot",
  "world_size": 1,
  "engine": {
    "heretic_research": true,
    "max_concurrency": 1,
    "prefix_cache_gib": 0,
    "kv_capacity": 2048,
    "default_max_tokens": 256
  }
}
```

The initial worker requires world 1, MTP and decode graphs off, no prefix
cache, BF16 KV, and checkpoint weight formats. Other explicitly supplied
engine options are rejected rather than silently ignored. Prompts are bounded
by the smaller of 2048 tokens and context capacity; continuations must fit the
remaining context and `default_max_tokens`. Model construction is lazy, after
a memory-plan check with an 8 GiB reserve on both host and device. This guard
rejects the full checkpoint on one GB10 before weight allocation. `describe`
needs only checkpoint metadata and works with CUDA hidden.

The owner creates an AF_UNIX stream socket pair and launches:

```text
dgpp-heretic-worker --config research.json --control-fd <inherited-socket-fd>
```

There is no listening socket, HTTP endpoint, or connection to a serving
process. The local Heretic branch provides `heretic.backends.dgpp_worker` as
the Python owner, including deadlines, tensor validation, and child cleanup.
Tokenization and chat templates remain with the caller. This interface does
not yet select DGPP in the main Heretic optimization CLI.

### Protocol version 1

Each frame starts with two little-endian uint32 lengths: JSON header bytes
and binary payload bytes. Headers are at most 64 KiB and response payloads
at most 64 MiB. Requests have no payload. Every request includes
`schema: "dgpp.heretic.rpc"`, `version: 1`, a consecutive `seq` starting at 1,
and `op`. Responses repeat those fields, identify `state: "baseline"`, and
list tensors with `name`, `dtype`, `shape`, `offset`, and `nbytes`. Tensor
bytes are contiguous, row-major, little-endian BF16, FP32, or int64.

| Operation | Request arguments | Response |
| --- | --- | --- |
| `describe` | none | Catalog, exact capabilities, vocabulary/hidden dimensions and token limits; no tensors |
| `first_token_logits` | `token_ids` | Raw FP32 logits `[vocab]` at the last prompt position |
| `capture` | `token_ids` | Logits, BF16 residuals `[layers+1, hidden]`, then each catalog target's `.input` and `.output` |
| `generate` | `token_ids`, `max_new_tokens`, `stop_token_ids` | Greedy continuation IDs and `stop_reason` (`length` or `stop_token`) |
| `close` | none | Empty acknowledgement, then process exit |

Advertised capabilities are `describe_targets`, `first_token_logits`,
`capture_last_token`, and `generate_greedy`. Requests execute sequentially.
Every prompt starts with fresh KV state and closes its session afterwards.
Greedy ties choose the lowest token ID; a matching stop token is included in
the returned IDs. The caller must supply its tokenizer's EOS/stop IDs; an empty
stop list deliberately runs to the length bound. No sampling or logit
processors are applied. Raw logits are not probabilities or log probabilities.

Capture residual order matches Heretic's hidden-state convention: embedding,
intermediate layer boundaries, final normalized hidden state. Target inputs
are BF16 last-position vectors. Attention/dense outputs are their BF16 values
widened to FP32. Shared-expert outputs retain the actual FP32 down-projection
accumulator before combination with routed experts; they are not rounded to
imitate a Transformers module. Routed Mixed346/H32/A8 computations continue
unchanged and their projections are not capture targets.

Malformed frames, invalid sequences, unsupported operations, or execution
errors terminate the worker with an error on stderr. They never leave a
partially failed worker available for another trial. Clean owner disconnect
also exits and frees model resources. The owner must enforce request deadlines
and reap/terminate its child on cancellation; stdout/stderr carry logs only.
This baseline-only protocol has no adapter epochs or reusable baseline hashes
yet, and does not authorize persistent score caches across worker lifetimes.

## Validation

Host unit tests check the target set, weight formats, rank bounds, and complete
nonoverlapping input partitions for worlds of size 1, 2, and 4. The command test
uses a directory containing only a tiny model config, with CUDA hidden, to
check the flag, JSON escaping, rank errors, and four partition descriptions.

```sh
cmake --build build-release --target unit_tests dgpp_heretic_inspect -j 4
ctest --test-dir build-release -R '^(unit_tests|heretic_inspect_test)$' --output-on-failure
```

The worker host test covers capability discovery without CUDA, the cluster
gate, unsupported settings, frame/token/sequence bounds, and disconnects.
GPU tests generate tiny int8 and Mixed346 fixtures and check bitwise capture
logit parity, residual boundary order, repeatability, session isolation, and
generation against direct model sessions. An independent pure-Python matrix
product checks every captured projection against checkpoint weights: relative
L2 below 0.004 for BF16 output paths and 0.00002 for the shared FP32 output.
These are correctness checks, not full-checkpoint or throughput measurements.

```sh
cmake --build build-release --target dgpp_heretic_worker dgpp_heretic_inspect heretic_capture_test -j 4
ctest --test-dir build-release -R '^heretic_' --output-on-failure -j 1
```
