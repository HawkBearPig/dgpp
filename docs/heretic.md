# Heretic integration

Heretic owns scoring plugins, modifier algorithms, optimization studies, and
exports. DGPP supplies execution for its supported checkpoint formats and applies the resulting runtime artifacts alongside the original weights.

The integration provides a GLM-5.3 projection catalog and a persistent
research worker for baseline logits, greedy generation, and last-token tensor
capture on one, two, or four nodes. The four-node worker can load the full
Mixed346 checkpoint. Protocol version 2 adds bounded projection weight reads,
teacher-forced likelihoods, and atomic candidate installation/reset. An opt-in
serving loader uses the same compact adapter files and execution code. The
Heretic backend remains on a local development branch. Bounded fitting trials
on the full checkpoint validate the integration; adapter quality still requires
representative training prompts and held-out evaluation.

## Configuration and performance isolation

Set `engine.heretic_research` to `true` in a separate research cluster config
to use the inspection command or worker. The default is `false`; existing deployment
templates are unchanged. This flag currently authorizes only the separate
research tools. It has no effect on `dgpp-serve` execution or its configuration
digest because serving performs no research operations.

Capture is a separate model walk compiled with a research template parameter.
The normal walk has no capture conditionals, copies, synchronization, or host
capture allocations. The worker checks the cluster flag before model access
or CUDA initialization. The distributed worker includes the effective flag,
resolved settings, checkpoint metadata/revision, binary fingerprint, and
numerical environment overrides in its startup agreement over the rank journal.
Runtime adapter execution requires the separate `engine.runtime_adapters=true`
flag in both the worker and ordinary serving. Disabled features must not allocate their buffers,
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

## Baseline worker

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

The worker supports worlds of size 1, 2 and 4, with MTP and decode graphs off, no prefix
cache, BF16 KV, and checkpoint weight formats. Other explicitly supplied
engine options are rejected rather than silently ignored. Prompts are bounded
by the smaller of 2048 tokens and context capacity; continuations must fit the
remaining context and `default_max_tokens`. Model construction is lazy, after
a memory-plan check with an 8 GiB reserve on both host and device. On an
integrated GPU, reclaimable host file cache counts toward the shared memory
budget. This guard
rejects the full checkpoint on one GB10 before weight allocation. `describe`
needs only checkpoint metadata and works with CUDA hidden.

The owner creates an AF_UNIX stream socket pair and launches:

```text
dgpp-heretic-worker --config research.json --control-fd <inherited-socket-fd>
```

The owner channel has no listening socket or HTTP endpoint and never connects
to a serving process. The local Heretic branch provides `heretic.backends.dgpp_worker` as
the Python owner, including deadlines, tensor validation, and child cleanup.
Tokenization and chat templates remain with the caller. This interface
selects DGPP in the main Heretic CLI with `--backend dgpp`,
`--dgpp-cluster-config research.json`, and
`--dgpp-worker-binary ../dgpp/scripts/dgpp-heretic-cluster`. Modification also
requires `engine.runtime_adapters=true`. Set `engine.default_max_tokens` at
least as high as Heretic's `--max-response-length`.

### Launch a distributed world

Use `scripts/dgpp-heretic-cluster` as the Python client's worker executable.
It accepts a deployment JSON with `world_size`, resolves the existing site
settings, and stages the same worker and resolved config on every node. Rank 0
runs locally: invoke it on the first node in `DGPP_NODES`. Peers need batch-mode
SSH access, the same checkpoint revision in their configured cache, CUDA and
the existing RoCE setup. A multi-node worker requires an IBV-enabled build.

For the Mixed346 checkpoint, create a separate research JSON:

```json
{
  "model": "HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128",
  "world_size": 4,
  "engine": {
    "heretic_research": true,
    "max_concurrency": 1,
    "prefix_cache_gib": 0,
    "kv_capacity": 512,
    "default_max_tokens": 16
  }
}
```

From the local Heretic branch, use the pinned tokenizer to prepare IDs and run:

```python
from heretic.backends.dgpp_worker import ResearchWorker

with ResearchWorker("research.json", "../dgpp/scripts/dgpp-heretic-cluster", timeout=600) as worker:
    logits = worker.first_token_logits(prompt_ids)
    captures = worker.capture(prompt_ids)
    tokens = worker.generate(prompt_ids, max_new_tokens=16, stop_token_ids=eos_ids)
```

Reserve idle hardware. Set `DGPP_FABRIC_PORT` and `DGPP_JOURNAL_PORT` to unused
ports for the research world. The launcher owns a unique staging directory and
process records; close, cancellation and failures terminate only those processes.
Rank logs and operation streams are retained under `DGPP_LOG_DIR/heretic/<run-id>`.
The inherited control descriptor currently requires a direct head launch
(`DGPP_NO_SWAP=0` on rank 0); a requested swap-limited scope is rejected rather
than silently bypassed. To prevent head swapping, launch the Python owner under
`systemd-run --user --scope --property=MemorySwapMax=0 -- ...` while retaining
`DGPP_NO_SWAP=0` for its direct worker launch. Children inherit that outer scope.
Monitor worker `VmSwap` and host memory during a run; configure peer swap policy
through the existing site settings.

The journal coordinates startup, commands, bounded binary capture replies,
and a barrier after all models load. Existing DGPP collectives execute tensor
parallel model operations and full-vocabulary greedy selection. Capture joins
input-column slices, checks replicated residuals and attention/dense outputs,
and sums shared-expert FP32 partials in rank order. A failed owner or rank aborts
the world, including during a collective; no partial result becomes a score.

### Protocol version 2

Each frame starts with two little-endian uint32 lengths: JSON header bytes
and binary payload bytes. Headers are at most 64 KiB and response payloads
at most 64 MiB. Candidate manifests and factors use bounded request payloads.
Every request includes
`schema: "dgpp.heretic.rpc"`, `version: 2`, a consecutive `seq` starting at 1,
and `op`. Responses repeat those fields, identify `state` (`baseline` or `candidate`), `epoch`, and `candidate_id`, and
list tensors with `name`, `dtype`, `shape`, `offset`, and `nbytes`. Tensor
bytes are contiguous, row-major, little-endian BF16, FP32, or int64.

| Operation | Request arguments | Response |
| --- | --- | --- |
| `describe` | none | Catalog, exact capabilities, vocabulary/hidden dimensions and token limits; no tensors |
| `first_token_logits` | `token_ids` | Raw FP32 logits `[vocab]` at the last prompt position |
| `capture` | `token_ids` | Logits, BF16 residuals `[layers+1, hidden]`, then each catalog target's `.input` and `.output` |
| `generate` | `token_ids`, `max_new_tokens`, `stop_token_ids` | Greedy continuation IDs and `stop_reason` (`length` or `stop_token`) |
| `read_projection_rows` | `target_id`, `row_begin`, `row_count` | Immutable global FP32 weight rows, at most 32 MiB |
| `score_tokens` | `token_ids`, `continuation_ids` | FP32 token log probabilities and int64 greedy-match flags; at most 2048 continuation tokens |
| `checkpoint_identity` | none | Full SHA-256 identity for the base files; requires runtime opt-in |
| `begin_candidate` | manifest as binary payload | Stage the declared target list; inference is blocked until activation or abort |
| `stage_projection` | `index`; global FP32 A then B as payload | Validate hashes, dimensions, finite values, and single staging |
| `activate_candidate` | none | Install all factors on every rank, clear session state, increment epoch |
| `abort_candidate` | none | Discard staging and retain the current candidate/epoch |
| `reset` | none | Remove active/staged factors, clear session state, increment epoch |
| `close` | none | Empty acknowledgement, then process exit |

Advertised capabilities are `describe_targets`, `first_token_logits`,
`capture_last_token`, `generate_greedy`, `read_projection_rows`, and `score_tokens`.
`runtime_adapters` is advertised only with its cluster flag. Requests execute sequentially.
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
Activation acknowledgements follow all-rank barriers. A missing factor, a failed
rank, or a stale epoch never yields a score. Heretic validates each reply's
candidate identity and epoch. Persistent caches must additionally bind the
checkpoint, tokenizer, runtime recipe, prompt tokens, and adapter identity.

## Runtime artifact and ordinary serving

The exported bundle contains `dgpp_adapter.json`, `adapter_model.safetensors`
and a PEFT `adapter_config.json` with scale one and canonical LoRA tensor names.
For the serving configuration, supported execution settings, bundle validation
and startup costs, see [runtime adapters and research inference](runtime-adapters.md).
The same runtime applies factors during candidate evaluation and serving.

Heretic records the worker binary SHA-256 and execution limits for study
resume and reproduction. Reproducing fitted factors requires the same worker
build and Python fitting environment. Exported adapters use ordinary serving's
request slots and context settings; the research worker's serial execution
and prompt bound do not become serving limits.

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
Distributed tests cover worlds 2 and 4, startup disagreement (settings,
binary, runtime environment), peer failure, and identical operation streams.
Projection oracle limits are 0.008 for distributed BF16 output paths and
0.00002 for shared FP32 outputs. Launcher tests cover flag rejection and
normal close, owner disconnect, and cancellation without CUDA or SSH.
Runtime tests compare attention, dense and shared corrections with independent
matrix calculations on int8 and Mixed346 fixtures, for worlds 1/2/4 and prompt
lengths 1/3/16/33. They check zero-adapter bitwise identity, staged abort,
generation/likelihood consistency, and exact reset. Host tests reject bad
fingerprints, shapes, ranks, hashes and configuration, and load artifact files
without CUDA.

The adapter batch tests cover 8 and 16 request slots on int8 and Mixed346
fixtures: zero-factor identity, nonzero adapters against isolated requests,
neighbor isolation, mixed prefill/decode, cancellation, slot reuse, and idle
replacement. Attention inputs are projected before each span's DSA scratch is
reused; the resulting rank-space rows follow the engine's existing row order.
Same-geometry isolation and identity are bitwise, including sparse attention
and routing. Cross-shape arithmetic uses a dense-attention, all-expert fixture
and the existing relative-L2 and near-tie numerical checks, matching the bulk
prefill numerical fixture's method.

A four-node probe of revision `824aedb0c783832223dab78a826bec7442ddea23`
validated all 154880 logits, 314 capture tensors, a 16-token continuation,
bitwise baseline restoration, zero process swap on every rank, and matching
operation streams. It tests execution and capture, not modification quality
or throughput. The model memory plan with the research configuration is about
93.6 GiB per rank; Python, captures, adapters and optimizer buffers are additional.

A subsequent four-node runtime pilot completed two Heretic ARA/Optuna trials
with KL, keyword and response-length scoring. It fitted rank-three factors for
eight attention/dense/shared projections in the first four layers and exported
a 1.86 MB bundle. Disabling or resetting the candidate restored baseline logits
exactly; restoring it recovered the adapted logits exactly. Teacher-forced
likelihood scoring also passed. Ordinary serving loaded that bundle beside the
full checkpoint and matched both recorded 16-token completions, with zero
cached tokens. Both worlds had zero process swap on all ranks and matching
operation streams. This bounded pilot used four neutral prompts to check the
workflow; it does not establish adapter quality or throughput. Independent
native fixtures also exercise CLI study resume and byte-identical reproduction
through the local Heretic backend.

```sh
cmake --build build-release --target dgpp_heretic_worker dgpp_heretic_inspect heretic_capture_test -j 4
ctest --test-dir build-release -R '^heretic_' --output-on-failure -j 1
```
