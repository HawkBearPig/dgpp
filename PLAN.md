# DGPP implementation status

Qwen C16/MTP3 supports 64 decode rows with scheduled verification disabled. Review fixes keep wide BF16 projections kernel-only while preserving existing small MTP walks. All 131 native checks are clear after a targeted rerun, and real four-node BF16, FP8-dense/MMA-head and BF12 serving each pass through sixteen concurrent requests with clean graphs and matching rank operation streams. See the [engineering record](benchmarks/results/2026-09-19-qwen-c16-mtp3-upstream.md).

The first local MiMo migration onto upstream `c6ca191` adds a compact-tool
grammar repair and two default-off prefill work reductions. On two Sparks,
the combined ports reduced cold TTFT by 3.55–4.39% versus the repeated
same-source control; decode was essentially unchanged. One concurrent code
output differed and its cause remains unassigned. That first panel used
upstream MTP1 and a shared 128K pool, not our native MTP3/256K-per-request
capacity. See the [migration record](benchmarks/results/2026-09-23-mimo-upstream-ports.md)
for gates, limitations and measurements.

The native MiMo MTP follow-up adds opt-in blocks 0/1/2 with correctly offset
backbone inputs and paged KV history. On the two-Spark panel this improves
code, JSON and maths decode relative to MTP1, while prose remains slower.
The [native MTP record](benchmarks/results/2026-09-23-mimo-native-mtp.md)
separates native heads from recursive block-0 MTP3 and records acceptance,
validation failures/fixes, final gates and the deployed configuration. Capacity remains the
upstream shared 128K pool in this experiment.
The [integrated review](benchmarks/results/2026-09-23-mimo-integrated-review.md)
checks the grammar repair, native heads and both prefill switches together,
records broader family regressions and arrival-dependent output diagnostics,
and links the portable reproduction tools.

This page summarizes the implemented features and remaining work as of
2026-09-21. [DESIGN.md](DESIGN.md) describes the architecture;
[CHANGELOG.md](CHANGELOG.md) and the dated records in
[benchmarks/results](benchmarks/results/) contain implementation history
and measurements. Performance figures apply to the configurations and
revisions recorded with them.

## Serving support

Completion responses expose llama.cpp-style `timings`, including final stream
chunks without usage opt-in. The response calculations preserve existing usage,
JSON metrics, Prometheus and log semantics. See the
[API contract](docs/openai-compatibility.md#completion-timings-dgpp-extension)
and [validation record](benchmarks/results/2026-10-08-completion-timings.md).

Assistant history accepts Anthropic-style thinking parts forwarded through
LiteLLM, folding their text into `reasoning_content` and dropping redacted
payloads. Explicit reasoning strings take precedence. Host regressions cover
normalization, tool-call history, role validation and string ownership; see
the [API profile](docs/openai-compatibility.md#assistant-thinking-history-dgpp-extension).

Completion streams delay their preamble until output is available, with a
terminal fallback for empty responses. Host regressions cover both endpoints,
multiple choices, UTF-8 tails and keep-alive compatibility; see the
[PR #57 record](benchmarks/results/2026-09-26-lazy-sse-preamble.md).

Issue #49 adds SSE keep-alive comments during admission, prefill and gaps in
stream output. The interval defaults to 30 seconds and is configurable through
cluster JSON, CLI and each streaming request; `-1` disables it. Optional prompt
progress events remain separate work. See the
[implementation and validation record](benchmarks/results/2026-09-26-sse-keep-alive.md).

The engine watchdog observes successful collective completions inside model
chunks, preventing a false timeout while tensor-parallel prefill is still
advancing. Controlled TP2 regression and genuine-stall tests pass; a full
261,120-token Qwen request also completes with the fix. The reporter's exact
full-model failure remains unreproduced; see the
[issue #45 investigation](benchmarks/results/2026-09-24-issue45/README.md).

Serving keeps generation stop IDs separate from the model's trained EOS,
preserving Qwen PLE padding and segmentation. The correction passes host tests
and was exercised in a real TP2 replay. It does not resolve the near-limit
retrieval failure in issue #4; see the [EOS investigation record](benchmarks/results/2026-09-23-issue4-eos/README.md).

The server uses a shared scheduler, text frontend and decode engine for
three model families. Rank 0 accepts HTTP requests and journals scheduler
operations to its peers. Every rank checks the operation-stream digest.

| family | implemented paths | deployment constraints |
|---|---|---|
| GLM-5.3-Flash | KDA and DSA attention, mHC, FP8 and hybrid NVFP4 experts, resident loading, graph decode, prefix cache and MTP | The full model needs four Sparks for resident serving. Batched decode has an eight-row limit; MTP depths 2–3 use scalar graphs |
| Qwen3.8-Flash-Next | GDN, QSA, gated residuals, PLE n-gram embeddings, FP8, NVFP4 and AutoRound int4 (GPTQ g128) experts with the int8 g128 head, optional FP8 dense projections, graph decode, prefix cache and MTP | FP8 deployment templates use two or four nodes. Single-node NVFP4 serving maps the n-gram table from NVMe. Batched decode supports 64 rows, including sixteen slots at MTP depth 3 with scheduled verification disabled; opt-in fixed-depth compaction reduces wide graphs within their numerical dispatch range, with scalar fallback when no physical family fits. Compaction defaults off; without it deeper MTP uses fitting physical slot prefixes or scalar graphs. Opt-in prefill continuation gives decode a turn between chunks |
| GLM-4.7 | Paged GQA, partial RoPE, NVFP4 dense and expert weights, draft-layer requantization, graph decode, prefix cache and MTP | Four-node serving is measured. The engine supports up to 32 batched decode rows, including deeper MTP; the supplied default recipe uses depth 1 |
| GLM-5.3 (full) | MLA with decoupled RoPE and per-token DSA selection shared across layers, int4/int8 pack-quantized experts and attention, draft-layer requantization, graph decode, prefix cache and MTP | Four nodes at 99.3 GiB of weights per rank (48K bf16 / 96K fp8 latent cache at four slots); served 2026-09-12: T=1 51 ms/step, MTP 68–76 ms/pass at 1.8–2.0 tokens/pass, gsm8k 59/60, HumanEval 40/40. Batched decode up to sixteen rows (eight request slots at MTP depth 1, five at depth 2; the select in row groups of eight); packed experts and attention use tensor-core prefill from 128 rows; shorter prompts retain GEMV to preserve measured C1/MTP behavior |
| DeepSeek-V4.1-Flash | CED encoder/decoder, CSA2 sliding-window + compressed-KV attention with a two-level indexer, single-pass hyper-connections, Engram n-gram tables mapped from NVMe, the DSpark block draft (five drafts per pass), the MXFP4/FP8 checkpoint as shipped, graph decode, prefix cache and a bounded (SWA-replay) prefill | Four nodes at 72.94 GiB of weights per rank (128K context at two slots); served 2026-09-14: 76 ms/pass at 2.33 tokens/pass (32.5 ms/token), bounded prefill 1.6–2.4 ms/token, gsm8k 60/60, HumanEval 40/40, extract 30/30. `engine.prefill` chooses bounded (the default) or the exact 40-layer parity mode. Prefix caching is whole-block, so prompts shorter than 128 tokens are not cached yet |
| MiMo-V2.6-Flash | Hybrid sliding-window (128, sink-biased) / global GQA attention over 192/128-wide heads with partial RoPE, the fused pre-sharded fp8 qkv projection read in the checkpoint's chunk layout, MXFP4 routed experts without a shared expert, BF16 o_proj / head / eh_proj as 12-bit companions, one MTP draft layer, a per-layer-width paged K/V pool in BF16 or the fp8 row form, graph decode and prefix cache | Served 2026-09-22 on four nodes (50.06 GiB per rank at 128K: C1 71.0–84.4 tok/s at 23 ms/pass, cold prefill 1.83 / 7.42 / 36.6 s at 2K / 8K / 32K, HumanEval 151/164, GSM8K 294/300, extraction 100/100) and two nodes (97.2 GiB per rank with a 256K fp8 pool: C1 42–48 tok/s). The vision and audio encoders are not served; the sliding-window layers keep their full history in the paged cache (a 128-token ring is the recorded memory lever); the one-split prefill attention is the recorded prefill lever past 8K |
| DeepSeek-V4-Flash | 128-token sliding-window attention over one shared 512-wide latent with ratio-4 (64-head indexer, 512 entries per query) and ratio-128 compressed caches, two-pass hyper-connections over four streams, token-table routing on layers 0–2, MXFP4 routed experts with an FP8 shared expert, the DSpark block draft, graph decode, prefix cache, chunk-invariant prefill | Two or four nodes at the full 1M-token context (95.4 / 63.0 GiB per rank). The templates schedule the verify depth over the five-draft block; sampled requests draft greedily (`engine.mtp_draft`) and follow the schedule. Text only. |

World size comes from the configuration's node list. A single-node graph
world uses resident weights and identity collectives. A single-node run
without graph decode uses the eager streaming path. Memory planning at
startup checks whether the selected model and request capacity fit.

The text API supports streaming, tools, supported JSON schemas, reasoning
output, sampling controls, logprobs, stop strings and multiple chat choices.
Unsupported fields are rejected by name. GLM-5.3-Flash and Qwen3.8-Flash-Next
support PNG/JPEG/WebP image inputs with native vision encoders, journaled RGB
pixels, streaming and MTP. Image requests support prefix caching and
resumable prefill, and bypass grouped prefill; see
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
| M7 — prefix cache | Pool-aligned snapshots, an earlier long-document cut for changed tails, shared cache blocks, deterministic lookup and LRU eviction; MTP lookahead identity | Cached/cold target and draft parity, continuation cleanup, small-arena retention and [recipe sizing](docs/prefix-cache.md) |
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
- [MiMo-V2.6-Flash architecture and port](docs/mimo_v26_flash_plan.md):
  the hybrid SWA/GA attention with sinks, the pre-sharded fused projection,
  the MXFP4/fp8 checkpoint as shipped, the fp8 K/V form, the fixture
  ladder, the two defects met only on the real checkpoint, and the serving
  record.

## Qwen NVFP4 expert prefill

Qwen NVFP4 experts use W4A16 by default. `DGPP_MOE_W4A4=1` opts into
W4A4 for eligible grouped prefills and reserves the quantized activation
buffers. The host reference supports both output formats. Decode retains
W4A16. Issue #68 tracks the paired static/dynamic-scale teacher-forced and
task-evaluation gate required before revisiting this default; passing
kernel tests or a single BFCL comparison does not complete that gate. The
[PR #50 follow-up](benchmarks/results/2026-09-25-pr50-w4a4.md) records the
regression coverage and validation.

## Qwen FP8 vocabulary head

Compact serving logits now cover all six model families, including BF16 and
packed Qwen heads. The implementation keeps decode and MTP verification rows
and projects only prefill tails. The
[validation record](benchmarks/results/2026-10-01-compact-serving-heads.md)
documents memory savings, numerical error, real GLM-4.7 scoring, and the
unchanged strict cache and speculative-decoding contracts.

The NVFP4 deployment templates now select streaming MMA with
`engine.fp8_head: "mma"`. Matched real-checkpoint teacher-forced comparisons
passed on one and two Sparks, including the native eight-row recipe boundaries,
the sixteen-row deeper-MTP envelope and the factor-2 YaRN recipe. The
[numerical record](benchmarks/results/2026-09-21-qwen-fp8-head-numerics.md)
retains per-case results, exact repeat controls, short-prefill coverage and
serving checks. Configurations that omit the setting retain GEMV.

Plain FP8 prefills compute only their final vocabulary row while preserving
the full chunk's kernel selection and accumulation order. The
[prefill-head follow-up](benchmarks/results/2026-09-24-pr43-prefill-head.md)
covers the GEMV/tensor-core boundary and the full-head comparison switch.

The earlier Release-build ABBA workload measured 11.26% higher C4/MTP3
throughput with C1 effectively unchanged; its
[results and transcript limits](benchmarks/results/2026-09-20-qwen-fp8-head-e2e.md)
remain historical performance evidence. The numerical campaign does not
establish broad model-quality equivalence or long-context retrieval quality.

## Remaining work

The [Docker cross-build](docs/cross-compiling.md) provides x86-to-ARM64/GB10
compilation and install staging. Its [validation record](benchmarks/results/2026-09-19-spark-cross-build.md)
tracks host checks separately from target execution on idle Spark hardware.
Compiler versions are printed at build time. The follow-up based on upstream
`c5a6913` passed the two-Spark API and operation-stream checks recorded there;
four-node and full-suite hardware coverage remain separate gates.

Qwen graph serving can interleave prefill chunks with decode using
`engine.prefill_budget_tokens`; zero preserves monolithic admission.
An optional `engine.prefill_idle_budget_tokens` increases chunk size when
no request is actively decoding, including after a decoding peer retires.
Queued text of any size now joins existing fair shares while a non-group
prefill is in flight (issue #104), with one new reader per tick and unchanged
slot, KV and token budgets. Host scheduler regressions cover admission,
cache attachment and cancellation/failure cleanup; real-model latency and
MTP block-boundary validation remain separate acceptance gates.
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
partial arithmetic and decode kernels. The
[QSA warp-prefill kernel](benchmarks/results/2026-09-24-qsa-warp-review.md)
now handles larger prefills with tolerance-based comparison against the
one-, three- and eight-split references, including model-level dispatch checks. The
[long-context QSA selector](benchmarks/results/2026-09-21-qwen-qsa-select.md)
uses exact radix selection above 2048 pools, retaining the score arithmetic,
tie order and workspace. QSA scoring storage follows the per-request context
rather than the shared pool (2026-09-28), and no decode row is staged past the
positional ceiling. Prefill scoring launches follow the visible history, and
rows that select every pool bypass scoring and sorting. This shared Qwen path
has NVFP4 coverage with BF16 and FP8 dense projections; a bounded decode-grid
experiment was rejected after regressions on wider batches. See the
[launch and selection validation](benchmarks/results/2026-09-30-qsa-launch-selection.md).
Group admission is bitwise the prefills alone (2026-09-28:
cuts only where a snapshot can stand, the prefill head over the mirrored rows).
Grouped Qwen continuation and
GLM-Flash row expansion are the next targets in the
[performance plan](docs/performance_improvement_plan.md#10-next-priorities-after-the-first-delivery).

The metrics endpoints expose existing engine-lifetime MTP verification counters
under `scheduler.spec_decode`, including separate round and draft-token totals
and per-position attempts and accepts, including sampled exact-fallback
acceptance. See the
[metrics contract](docs/openai-compatibility.md#speculative-decoding-counters).

Sampled MTP now preserves the drafts each request verified before settling
another request's fallback. Live observer-based oracle checks cover grouped
and staggered admission, reused slots, final transcripts and counters, with
pipelining enabled and disabled. See the
[fallback isolation record](benchmarks/results/2026-09-21-mtp-fallback-isolation.md).

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

## Decode batch observability

Implemented graph launch, verification-row and padding counters with a capacity
histogram and retained last-launch shape in the JSON metrics snapshot. Host
contract tests, the native Spark suite and live two-rank metrics/API checks
passed; the physical four-node and unavailable checkpoint cases remain
untested. See the
[validation record](benchmarks/results/2026-09-19-decode-batch-telemetry.md).
