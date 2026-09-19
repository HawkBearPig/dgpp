# DGPP implementation status

This page summarizes the implemented features and remaining work as of
2026-09-18. [DESIGN.md](DESIGN.md) describes the architecture;
[CHANGELOG.md](CHANGELOG.md) and the dated records in
[benchmarks/results](benchmarks/results/) contain implementation history
and measurements. Performance figures apply to the configurations and
revisions recorded with them.

## Serving support

The server uses a shared scheduler, text frontend and decode engine for
three model families. Rank 0 accepts HTTP requests and journals scheduler
operations to its peers. Every rank checks the operation-stream digest.

| family | implemented paths | deployment constraints |
|---|---|---|
| GLM-5.3-Flash | KDA and DSA attention, mHC, FP8 and hybrid NVFP4 experts, resident loading, graph decode, prefix cache and MTP | The full model needs four Sparks for resident serving. Batched decode has an eight-row limit; MTP depths 2–3 use scalar graphs |
| Qwen3.8-Flash-Next | GDN, QSA, gated residuals, PLE n-gram embeddings, FP8 and NVFP4 experts, optional FP8 dense projections, graph decode, prefix cache and MTP | FP8 deployment templates use two or four nodes. Single-node NVFP4 serving maps the n-gram table from NVMe. Batched decode supports sixteen rows, including eight slots at MTP depth 1; deeper MTP uses fitting physical slot prefixes or scalar graphs. Opt-in prefill continuation gives decode a turn between chunks |
| GLM-4.7 | Paged GQA, partial RoPE, NVFP4 dense and expert weights, draft-layer requantization, graph decode, prefix cache and MTP | Four-node serving is measured. The engine supports up to 32 batched decode rows, including deeper MTP; the supplied default recipe uses depth 1 |
| GLM-5.3 (full) | MLA with decoupled RoPE and per-token DSA selection shared across layers, int4/int8 pack-quantized experts and attention, draft-layer requantization, graph decode, prefix cache and MTP | Four nodes at 99.3 GiB of weights per rank (48K bf16 / 96K fp8 latent cache at four slots); served 2026-09-12: T=1 51 ms/step, MTP 68–76 ms/pass at 1.8–2.0 tokens/pass, gsm8k 59/60, HumanEval 40/40. Batched decode up to sixteen rows (eight request slots at MTP depth 1, five at depth 2; the select in row groups of eight); packed experts and attention use tensor-core prefill from 128 rows; shorter prompts retain GEMV to preserve measured C1/MTP behavior |
| DeepSeek-V4.1-Flash | CED encoder/decoder, CSA2 sliding-window + compressed-KV attention with a two-level indexer, single-pass hyper-connections, Engram n-gram tables mapped from NVMe, the DSpark block draft (five drafts per pass), the MXFP4/FP8 checkpoint as shipped, graph decode, prefix cache and a bounded (SWA-replay) prefill | Four nodes at 72.94 GiB of weights per rank (128K context at two slots); served 2026-09-14: 76 ms/pass at 2.33 tokens/pass (32.5 ms/token), bounded prefill 1.6–2.4 ms/token, gsm8k 60/60, HumanEval 40/40, extract 30/30. `engine.prefill` chooses bounded (the default) or the exact 40-layer parity mode. Prefix caching is whole-block, so prompts shorter than 128 tokens are not cached yet |

World size comes from the configuration's node list. A single-node graph
world uses resident weights and identity collectives. A single-node run
without graph decode uses the eager streaming path. Memory planning at
startup checks whether the selected model and request capacity fit.

The text API supports streaming, tools, supported JSON schemas, reasoning
output, sampling controls, logprobs, stop strings and multiple chat choices.
Unsupported fields are rejected by name. GLM-5.3-Flash also supports PNG/JPEG
image inputs with a native vision encoder, journaled RGB pixels, streaming
and MTP. Image requests bypass prefix caching and grouped prefill; see
[image inputs](docs/vision.md). Audio and video inputs are unsupported.
Vision arithmetic now follows the CUDA BF16 eager reference, with fixed
full-encoder and isolated-operation gates. The
[numerical investigation](benchmarks/results/2026-09-18-glm-vision-numerics.md)
records the reduction, bias, rotary, attention-layout and LayerNorm fixes,
with bitwise matching embeddings across the 30-case regression corpus.

## Original milestones

M0–M9 were completed for version 0.1.0. The
[v1 sign-off](docs/signoff_v1.md) records the measured workloads and
limitations, including the one-hour soak used for that release.

| milestone | delivered capability | validation |
|---|---|---|
| M0 — platform and transport | Memory, compute and RoCE probes; registered host memory consumed by the GPU | Dated platform measurements, two-lane bandwidth and NIC/GPU visibility checks |
| M1 — runtime | Arenas, streams, graphs, tracing and a synthetic model | Unit tests and synthetic eager/graph parity |
| M2 — KDA | Recurrent attention, convolution state, snapshots and head sharding | Host-reference, chunking, graph replay and reference-dump comparisons |
| M3 — DSA/MLA | Pooled sparse selection, latent caches, tail rings and paged state | Selection fuzzing, attention oracles, continuation and snapshot tests |
| M4 — assembled GLM | Config/binding validation, layer loading, full forward and reference tools | Synthetic and real-checkpoint layer/forward comparisons |
| M5 — tensor parallelism | Roster, CollectiveBus, sliced loading, resident images and transport regression tools | Loopback protocol tests, shard parity and four-node forward checks |
| M6 — generation and API | Decode sessions, tokenizer, templates, scheduler, journal, HTTP/SSE, sampling and constrained output | Host service tests, tokenizer/template goldens and fabric API checks |
| M7 — prefix cache | Pool-aligned snapshots, shared cache blocks, deterministic lookup and LRU eviction | Cached/cold prefill comparisons, MTP boundary tests and capacity sweeps |
| M8 — MTP | Transactional verification, rollback and on-device speculative steps | Greedy equality with plain decode, sampled-oracle comparisons and forced rejections |
| M9 — optimization and hardening | Kernel and collective optimizations, memory planning, failure handling and drift detection | Numerical regression checks, failure drills, fuzzing and the mixed-workload soak |

## Additional model work

The shared engine and loader interfaces support both additional families.
Their implementation and evaluation records are maintained separately:

- [Qwen architecture and port](docs/qwen38_flash_next_plan.md): tensor
  placement, GDN/QSA/GR/PLE operators, reference comparisons, sessions,
  tensor parallelism and serving.
- [Qwen optimization study](docs/qwen38_optimization_plan.md): measured
  decode and prefill changes, sampled drafting, and remaining experiments.
- [Qwen on one Spark](docs/qwen38_single_spark.md): NVFP4 experts, mapped
  n-gram storage, optional FP8 dense weights and measured quality.
- [GLM-4.7 architecture and port](docs/glm47_plan.md): modelopt NVFP4
  handling, GQA, draft preparation and serving validation.
- [Full GLM-5.3 architecture and port](docs/glm53_plan.md): the
  pack-quantized weight format, the DSA changes (RoPE, per-token
  selection, sharing), placement and cost, the gates and their status.
- [GLM-5.3 NVFP4 study](docs/nvfp4_plan.md): checkpoint composition,
  quantized kernels, numerical comparisons and optimization results.
- [DeepSeek-V4.1-Flash architecture and port](docs/deepseek_v41_flash_plan.md):
  the CED/CSA2/Engram/DSpark operators, the quantization decision (serve
  as shipped), the reference and torch cross-checks, the bounded prefill,
  the tokenizer and DSML tool grammar, and the serving record.

## Remaining work

The [Docker cross-build](docs/cross-compiling.md) provides x86-to-ARM64/GB10
compilation and install staging. Its [validation record](benchmarks/results/2026-09-19-spark-cross-build.md)
tracks host checks separately from target execution on idle Spark hardware.

Qwen graph serving can interleave prefill chunks with decode using
`engine.prefill_budget_tokens`; zero preserves monolithic admission.
An optional `engine.prefill_idle_budget_tokens` increases chunk size when
no request is actively decoding, including after a decoding peer retires.
Extending this to other families and grouped continuations remains work
for latency under mixed prompt lengths. Prefix entries
are process-local, and grow-on-demand admission ends the youngest request
when the pool is exhausted; it does not preempt and recompute it.

The [Qwen expert prefill investigation](benchmarks/results/2026-09-15-qwen-moe-prefill.md)
tested smaller tiles, compact grids and persistent blocks without finding a
production improvement. Those variants remain in a standalone benchmark;
serving kernels are unchanged. The
[full-GLM packed prefill implementation](benchmarks/results/2026-09-15-glm-packed-prefill.md)
adds int4/int8 tensor-core tiles, FP64 arithmetic checks and prefill likelihood
scoring. [Qwen QSA tile reuse](benchmarks/results/2026-09-15-qwen-qsa-prefill.md)
then reduces measured cold TP2 prefill time by 8–14%, preserving the existing
partial arithmetic and decode kernels. Grouped Qwen continuation and
GLM-Flash row expansion are the next targets in the
[performance plan](docs/performance_improvement_plan.md#10-next-priorities-after-the-first-delivery).

Other work includes request-level observability, additional API fields,
silent-node-loss detection, wider batching for GLM-5.3-Flash, arbitrary
slot subsets for oversized batches, and
model-specific performance experiments. See
[next steps](docs/next_steps.md) for the scope and validation needed for
each. Failover, data-parallel routing, cross-instance prefix sharing and
image-aware prefix identities, additional vision backends, audio and video
execution remain future work.

The [DeepSeek inference study](benchmarks/results/2026-09-16-dsv41-perf/README.md)
adds payload-dependent graph consumer widths and uses 4,096-token
bounded-prefill chunks. It records matched serving measurements and rejected
receive-cache, verification-depth and scheduling-policy experiments; weight and KV precision are unchanged.

## Validation for further changes

Run a full build before the relevant test suites. Kernel and state changes
need reference oracles and the applicable session/graph comparisons;
multi-rank changes also need fabric checks with matching operation streams.
Greedy MTP must match plain decode. Numerical changes across builds are
evaluated by logit margins and teacher-forced loss as described in
[numerics](docs/numerics.md).

Record benchmark conditions, commands, revisions and results with each
measurement. [Testing](docs/testing.md), [operations](docs/operations.md)
and [the benchmark procedures](docs/benchmarks.md) describe the available
checks. Proposed performance targets remain estimates until measured.
