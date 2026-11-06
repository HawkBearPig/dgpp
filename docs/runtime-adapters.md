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

Create a deployment JSON using the same base checkpoint and world size as the
adapter. For example, with an exported bundle in `adapters/example` beside the
deployment JSON:

```json
{
  "model": "HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128",
  "world_size": 4,
  "engine": {
    "runtime_adapters": true,
    "runtime_adapter_path": "adapters/example",
    "max_concurrency": 8,
    "kv_capacity": 131072,
    "default_max_tokens": 8192,
    "mtp": false,
    "decode_graph": false,
    "prefix_cache_gib": 0,
    "kv_dtype": "bf16",
    "bf16_weights": "checkpoint",
    "attention_weights": "checkpoint",
    "embed_sharding": "replicated"
  },
  "nvme_cache": {"enabled": false}
}
```

Use the usual site settings and launcher:

```bash
python3 scripts/dgpp-cluster up --config serving.json
```

The launcher resolves relative adapter paths beside the deployment JSON and
stages `dgpp_adapter.json` and `adapter_model.safetensors` on every node. Direct
binary launches require both files to be present on every node. Startup checks
the bundle against the checkpoint and execution settings; every rank must
agree on its identity before serving. Switching or removing a serving adapter
requires a restart. To return to ordinary serving, remove the bundle path,
disable `runtime_adapters`, and restore the desired base deployment settings.

The example's concurrency, KV pool and default output budget are sizing choices.
Serving supports the existing 1–16 request slots. `kv_capacity` is shared across
all active requests, and both prompt and output tokens consume it.
`default_max_tokens` applies when a client omits its output limit. Set these
values for available memory and the workload using the existing
[capacity and admission controls](configuration.md#request-capacity-and-memory).
Research worker limits do not restrict the exported adapter's serving context
or concurrency.

## Supported execution settings

The current adapter runtime uses eager execution with the following settings.
An optimized serving template may require several changes in addition to the
adapter flag and path. Unsupported combinations fail validation.

| Setting | Required value with runtime adapters |
|---|---|
| Model and world size | GLM-5.3 DSA; 1, 2 or 4 ranks, matching the bundle |
| `engine.decode_graph` | `false` |
| `engine.mtp`, `engine.lookup_draft` | `false` |
| `engine.dflash_model` | Empty |
| `engine.prefix_cache_gib`, `nvme_cache.enabled` | `0`, `false` |
| `engine.kv_dtype` | `bf16` |
| `engine.bf16_weights`, `engine.attention_weights` | `checkpoint` |
| `engine.embed_sharding` | `replicated` |
| `engine.prefill_bf16_partials`, `engine.prefill_fold_scales` | `false` |
| `engine.expert_gemm`, `engine.expert_gemm_prefetch` | `wide`, `3` |
| `engine.expert_tile_list`, `engine.expert_gemm_pair` | `true`, `false` |
| `engine.collective_transport` | `write` |

Checkpoint weight formats leave the original packed and quantized tensors
intact. Adapter execution does not produce a dequantized or merged checkpoint.
It adds factor storage, scratch buffers and computation, including collectives
across ranks. Disabling graphs, speculation or caching can also change serving
performance; ordinary-serving benchmark results do not measure this mode.
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
`glm_dsa_eager_fp32_v1`, the fitted world size, target IDs/modules/dimensions,
per-factor SHA-256 hashes, the numerical environment hash and the base content
fingerprint. Fingerprinting reads every weight shard, index, configuration,
external quantization sidecar and present tokenizer/template/generation files.
This can add minutes to cold startup for large checkpoints. It occurs only
with runtime opt-in; research trials reuse the identity and check that the base
files have not changed. The numerical hash covers sorted `DGPP_*` overrides,
excluding transport addresses, paths, logging and test controls. Different
numerical overrides reject loading.

Each rank stores its A columns and replicated B. The runtime computes A times
the BF16 projection input, gathers FP32 partials losslessly, sums in rank order
and adds B times that sum once to the already-folded BF16 branch output,
rounding to BF16. Shared-expert corrections enter the combined MLP branch;
research capture retains the shared projection's separate FP32 output plus
correction. Zero factors skip the addition, preserving identity exactly.
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
