# Decode graph batch telemetry

## Scope

Based on upstream `444441feed0c4d7c256ff7ff7868b587823b737f`.
Extracts the decode-batch counters from historical commit `8825c4b`, excluding
its prefill tuning and performance experiments. It does not change graph
capacity, scheduling, MTP acceptance accounting or deployment configuration.

`SchedulerEngine` supplies zero defaults; the graph adapter records successful
launches, capacity, active requests and actual verification width. Scheduler
snapshots publish these values through both metrics aliases. The histogram
includes scalar fallback, and the last-launch fields persist after requests
close. A compile-time check keeps the histogram large enough for the picker.

## Workstation validation

Validation is performed in this isolated worktree with a disposable native
x86-64 build container derived from `dgpp-spark-cross:upstream-pr`, adding native
libibverbs, cuBLAS and clang-format packages. No production processes or
configuration are changed. Logs are retained locally under
`artifacts/decode-batch/`.

Commands and results:

- `cmake --preset ci`: configured with CUDA 13.0.88 and GCC 13.3.
- `cmake --build build-ci -j 4 --target serve_test scheduler_test qwen_engine_test`:
  all three targets built successfully, including CUDA kernels and the graph
  adapter instantiated by the Qwen test. This is native x86-64 compilation,
  not ARM64 execution.
- `DGPP_TEST_FILTER=serve_decodeBatchMetrics build-ci/serve_test`: one test passed.
  It covers zero defaults, publication from the engine, both endpoint aliases,
  every histogram bucket and retained last-launch shape at zero occupancy.
- `ctest --test-dir build-ci -R '^(serve_test|scheduler_test)$' --output-on-failure -j 1`:
  both suites passed (58 service and 57 scheduler cases) in 118.89 seconds.
- Changed C++ ranges formatted with clang-format 18; `git diff --check` passed.

The Qwen assertions exercise scalar MTP depths one and two, last-launch retention
at close, full six/eight-slot batches and padding after an interior slot retires.
Their target execution is recorded below.

## Native Spark and two-rank validation

Commit `8557e02df841` was cloned into `<checkout>/dgpp-decode-batch-pr` on
Spark 1 and built natively with `cmake --preset ci` followed by
`cmake --build --preset ci -j 4`. The complete build passed before testing.
The user confirmed exclusive handoff after another thread's maintenance.
Both production ranks then stopped cleanly, and both GPUs were checked idle.

With `CUDA_DEVICE_MAX_CONNECTIONS=32`, the serial command
`ctest --test-dir build-ci --output-on-failure -j 1 --timeout 300`
completed in 626.22 seconds: **101 passed, 10 skipped, zero failures**.
The Qwen engine, row-independent wide-batch and BF12 suites all passed,
executing the new scalar/depth-two, retained-last-launch and six/eight-slot
padding assertions. The other GPU/RDMA engine and bus suites also passed.

Skipped cases were `tokenizer_test`, `dsv41_tokenizer_test`,
`dsv41_prompt_test`, `glm4_tokenizer_test`, `glm_dsa_tokenizer_test`,
`chat_template_test`, `glm4_chat_template_test`, `glm_dsa_chat_template_test`,
`glm_vision_stream_test` and `glm_vision_frontend_test`; their optional
checkpoints were unavailable. The optional `glm_tp_forward_parity_real`
subcase also remained disabled because `DGPP_TP_REAL_MODEL` was not set.

The tested native binary was staged to both Sparks for a separate TP2
Qwen3.8-Flash-Next-NVFP4 deployment: concurrency 4, MTP depth 3, 65,536 KV
tokens, a 1 GiB prefix cache and HTTP port 30002. Both rank logs identify
`0.1.0+g8557e02df841`. Three serial health requests produced three scalar
launches, twelve verification rows and no padding. Four concurrent counting
requests with output limits 48, 96, 144 and 192 then produced these cumulative
counters:

| counter | value |
|---|---:|
| replays | 51 |
| verification rows | 648 |
| padded rows | 156 |
| one-slot replays | 13 |
| two-slot replays | 1 |
| three-slot replays | 1 |
| four-slot replays | 36 |

All other histogram buckets were zero. The row total equalled four times the
capacity-weighted histogram, and replay totals matched the histogram sum.
Both metrics aliases agreed after the service became idle. The retained shape
was one slot, one active request and four rows per request, while current
scheduler active and queued counts were both zero. There were no failed
requests or engine failures. `scripts/serve_api_check.py` passed all nine
stop, streaming, multiple-choice, logit-bias and usage checks.

After clean test shutdown, both ranks' operation streams had MD5
`2604c09be3fb0b5a324e0604e31e6408`.

The maintenance runner restored the original release
`0.1.0+g42b105d-mtpmetrics` on both ranks. Binary SHA-256 remained
`ab1b88a00de32dedb0de63d285e584651d69c99185c106c4e6d7e549b9585fae`;
resolved-config hashes on both ranks and the deployment/site-file hashes
matched their pre-test values. Production returned to concurrency 16, MTP
depth 3 and its 8 GiB prefix cache. Three repeated requests succeeded; the
second and third each reused 1,128 prompt tokens. Metrics reported 148 cache
slots and no engine failure, and both rank logs confirmed request retirement.
The stop-test-restore window ran from 23:21:42 to 23:33:25 UTC on 2026-09-19.

Raw build, CTest, metrics, API, shutdown and restoration evidence is retained
locally under `artifacts/decode-batch/server-validation/` (ignored by Git).

## Limits

These checks establish counter behavior in synthetic GPU/RDMA fixtures and a
physical two-Spark serving world. A physical four-node fabric run and the ten
unavailable checkpoint cases were not exercised. The counting workload is a
functional test, not a utilization or throughput benchmark.
