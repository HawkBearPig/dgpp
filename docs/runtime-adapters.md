# Runtime adapters and research inference

DGPP supports compatible LoRA-format adapters loaded alongside an unchanged
base checkpoint. A separate research worker exposes logits, activations and
token scores so external tools can fit and evaluate adapters using DGPP's
model execution. Current support covers GLM-5.3 DSA on one, two or four nodes.

Both capabilities require explicit cluster configuration flags and default to
disabled. `engine.runtime_adapters` enables low-rank execution;
`engine.heretic_research` enables the inspection and research worker commands.
Serving an exported adapter needs only the runtime flag and bundle path.
Baseline research inference needs only the research flag. See the
[configuration reference](configuration.md#runtime-adapters-and-research-inference)
for each setting's default and dependencies.

## Load an adapter for serving

Copy the model's shipped deployment template, keeping the same base checkpoint
and world size as the adapter. Add only these two fields to its `engine` object:

```json
{
  "runtime_adapters": true,
  "runtime_adapter_path": "/path/to/adapter"
}
```

For example, create `serving.json` from the shipped four-node Mixed346 template:

```bash
python3 - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("deploy/cluster_glm-5.3_mixed346_w4.example.json").read_text())
config["engine"]["runtime_adapters"] = True
config["engine"]["runtime_adapter_path"] = "/path/to/adapter"
Path("serving.json").write_text(json.dumps(config, indent=2) + "\n")
PY
```

All other settings remain those of the shipped template, including CUDA graphs,
MTP, FP8 KV, BF12 weight storage, vocabulary-sharded embeddings, prefix caching,
concurrency, scheduling and output budgets. Use the usual site settings and launcher:

```bash
python3 scripts/dgpp-cluster up --config serving.json
```

The launcher resolves relative adapter paths beside the deployment JSON and
stages `dgpp_adapter.json` and `adapter_model.safetensors` on every node. Direct
binary launches require both files to be present on every node. Startup checks
the bundle against the checkpoint and execution settings; every rank must
agree on its identity before serving. Switching or removing a serving adapter
requires a restart. To return to ordinary serving, remove the bundle path,
disable `runtime_adapters`.

Serving supports the existing 1–16 request slots. `kv_capacity` is shared across
all active requests, and both prompt and output tokens consume it.
`default_max_tokens` applies when a client omits its output limit. Set these
values for available memory and the workload using the existing
[capacity and admission controls](configuration.md#request-capacity-and-memory).
Research worker limits do not restrict the exported adapter's serving context
or concurrency.

## Supported execution settings

Adapters support the same eager and graph execution paths as the model, with
the following compatibility bounds. The shipped Mixed346 configuration fits
these bounds without changing any of its settings. Unsupported combinations
fail validation.

| Setting | Required value with runtime adapters |
|---|---|
| Model and world size | GLM-5.3 DSA; 1, 2 or 4 ranks, matching the bundle |
| `engine.decode_graph`, `engine.mtp` | Either value |
| `engine.lookup_draft` | `false` |
| `engine.dflash_model` | Empty |
| `engine.prefix_cache_gib` | Existing serving cache budget |
| `nvme_cache.enabled` | `false` |
| `engine.kv_dtype` | `bf16` or `fp8` |
| `engine.bf16_weights` | `checkpoint` or `bf12` |
| `engine.attention_weights` | `checkpoint` |
| `engine.embed_sharding` | `replicated` or `vocab` |
| `engine.prefill_bf16_partials`, `engine.prefill_fold_scales` | `false` |
| `engine.expert_gemm`, `engine.expert_gemm_prefetch` | `wide`, `3` |
| `engine.expert_tile_list`, `engine.expert_gemm_pair` | `true`, `false` |
| `engine.collective_transport` | `write` |

Checkpoint weight formats leave the original packed and quantized tensors
intact. Adapter execution does not produce a dequantized or merged checkpoint.
It adds factor storage, scratch buffers and computation. Eager execution adds
one collective at each adapted boundary. Graph execution carries the adapter's
values in the boundary's own collective. Graph bookkeeping grows only for an
enabled adapter, and the
startup memory plan includes its factors, scratch and additional graph records.
Ordinary-serving benchmark results do not measure adapter overhead.
With `runtime_adapters=false`, serving allocates no adapter buffers and runs no
adapter kernels or collectives. The research flag does not enable capture in
ordinary serving.

## Bundle format and compatibility

The required files are `dgpp_adapter.json` and `adapter_model.safetensors`.
Factors use canonical LoRA tensor names with FP32 A `[rank,input]` and B
`[hidden,rank]`, scale one, rank 1–256 and a 1 GiB global factor budget.
An exporter can also write a PEFT `adapter_config.json`; DGPP loads its own
manifest and factor file. A generic PEFT directory without the DGPP manifest
is not sufficient. Supported targets are attention output projections and
dense/shared MLP down projections. Routed-expert weights remain unchanged.

Manifest version 1 identifies operation `additive_low_rank`, runtime
`glm_dsa_fp32_v1`, the fitted world size, target IDs/modules/dimensions,
per-factor SHA-256 hashes, the numerical environment hash and the base content
fingerprint. Serving checks the revision, file names and sizes, configuration,
index, external quantization sidecars and present tokenizer/template/generation
file hashes by default. Adapter factor hashes, target shapes, world size and
the numerical environment are also checked. This verification does not read
base weight contents and cannot detect changes that preserve their file sizes.

Set `engine.runtime_adapter_verify_weights=true` to also recompute every base
weight hash. This explicit option reads the entire checkpoint on each rank,
can add minutes to startup, and logs progress before each shard. Leave it
unset or false for normal serving startup. Model loading still reads the
weights required by the existing loader and cache policy. Both verification
modes check that checkpoint file sizes and modification times remain unchanged
before adapter installation. Existing version 1 bundles support both modes.

Research workers compute full fingerprints for export, reuse them across trials
and check that the base files have not changed. The numerical hash covers sorted
`DGPP_*` overrides,
excluding transport addresses, paths, logging and test controls. Different
numerical overrides reject loading. Bundles using the original runtime name
`glm_dsa_eager_fp32_v1` remain compatible with the same factor arithmetic in
both eager and graph serving. Serving precision and cache settings remain
controlled by the deployment configuration.

Each rank stores its A columns and replicated B. The runtime computes A times
the BF16 projection input, gathers FP32 partials losslessly, sums in rank order
and adds B times that sum once to the already-folded BF16 branch output,
rounding to BF16. Shared-expert corrections enter the combined MLP branch;
research capture retains the shared projection's separate FP32 output plus
correction. Graph execution writes each rank's FP32 partials as byte digits
into disjoint slices directly behind the boundary's BF16 partial, and the
boundary's collective folds both as one message. One kernel behind the
collective reconstructs the partials, sums them in rank order and adds the
correction. That kernel reads its columns of B while the collective is in
flight (`engine.decode_pdl`). A boundary whose rows and digits exceed one
latency slot folds the digits in a second collective, with the same values.
The MTP draft remains a proposer; the adapted main model verifies
its proposals. Prefix snapshots hold state produced by that same installed
adapter. Zero factors skip the addition, preserving identity exactly.
This execution contract does not promise bitwise equivalence to merged
weights or another runtime's LoRA kernels.

## Logits and activation access

The separate research worker supports these operations on token IDs supplied
by its caller:

| Operation | Returned data |
|---|---|
| Target inspection | Projection names, dimensions and supported capabilities |
| `first_token_logits` | Full-vocabulary FP32 logits at the last prompt position |
| `capture` | Those logits, last-position residuals and supported projection inputs/outputs |
| `generate` | Greedy continuation IDs and the stop reason |
| `score_tokens` | Teacher-forced token log probabilities and greedy-match flags |
| `read_projection_rows` | Bounded immutable FP32 projection weight rows |

With `engine.runtime_adapters=true`, the worker also supports staging,
activating, aborting and resetting candidate factors. Activation is atomic
across ranks and clears session state. Replies identify the active candidate
and epoch so callers can associate scores with the correct factors.

Build targets `dgpp_heretic_inspect` and `dgpp_heretic_worker` produce the
inspection and worker binaries. `scripts/dgpp-heretic-cluster` launches a
separate worker world on one, two or four nodes. A client owns an inherited
Unix socket and sends the bounded binary protocol. Raw logits and activation
capture are available through this worker protocol; the serving HTTP API
continues to provide its documented [logprobs](openai-compatibility.md).
Tokenization and chat templates remain with the caller.

Use a dedicated research config with `engine.heretic_research=true`,
`max_concurrency=1`, graphs and MTP disabled, no prefix cache, BF16 KV and
checkpoint weight formats. The worker rejects unrelated serving options and
processes requests sequentially with fresh KV state. Prompts are bounded by
the smaller of 2048 tokens and context capacity; generated continuations must
fit the remaining capacity and `default_max_tokens`. Reserve the nodes for
the research world and stop it before starting serving there.

The [worker protocol](heretic.md#protocol-version-2) documents framing, operations
and ownership. The [Heretic integration guide](heretic.md) describes an
external client that uses this interface for optimization and adapter export.
