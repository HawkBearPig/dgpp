# Metrics follow-ups

## Scope

Issue #20 collects follow-ups from the merged decode-batch and speculative
decoding metrics changes. This work starts from upstream
`4f04f16dae597d4665475b2a771bf6b6725267de` and addresses the first six items:

- Cross-link the two metrics reference sections.
- Record the replay end event immediately after the CUDA graph launch, before
  updating host decode-batch counters.
- Retain the fixed sixteen-bucket histogram and document its stable shape for
  scrapers. The existing HTTP regression explicitly checks all sixteen keys.
- Assign port 29941 to the depth-one fallback-count gate, leaving the depth-two
  gate on 29935.
- Clarify that `num_drafts_total` counts request verification rounds.
- Replace the contributor-specific checkout path in the original telemetry
  record with a neutral placeholder, preserving its historical results.

The seventh item, measuring production sparse-slot padding, remains open.
No production measurement or performance claim is made here. Sparse compaction
and sampled-MTP correctness issue #15 are outside this change.

## Validation

Validation uses a local x86-64 container with CUDA 13, native libibverbs and
cuBLAS development libraries, and clang-format 18. Changed C++ ranges are
formatted. No running inference service is changed.

Static checks confirm that both documentation links resolve to existing
headings, port 29941 occurs only in the depth-one gate under `tests/cuda`, the
launch and end-event calls are adjacent, and the historical telemetry record
contains no home-directory path.

`cmake --preset ci` configured with warnings as errors.
`cmake --build build-ci -j 4 --target serve_test glm_tp_test qwen_engine_test`
passed for all three targets, compiling the changed GLM gate and the graph
adapter used by both model suites.
`ctest --test-dir build-ci -R '^serve_test$' --output-on-failure -j 1`
passed all 61 service cases in 129.00 seconds after rebuilding
`serve_test`. This includes decode-batch and speculative-decoding metrics
regressions through both HTTP aliases, including all sixteen histogram keys.
`git diff --check` passed.

The workstation checks do not establish GPU replay behavior or a timing
improvement from moving the event record. Target execution is recorded below.

## Two-Spark validation, 2026-09-21

Tested implementation commit `4172863817787caf86bebf343f8fb87365fd5ae8` on
GB10 hardware with CUDA 13. The same four native targets (`serve_test`,
`glm_tp_test`, `qwen_engine_test`, `dgpp_serve_app`) rebuilt successfully on
Spark 1 using the `ci` preset and warnings as errors before testing.

During the reserved maintenance window, both production ranks stopped cleanly
and both GPUs were verified idle. Serial focused CTest on Spark 1:

```sh
CUDA_DEVICE_MAX_CONNECTIONS=32 ctest --test-dir build-ci \
  -R '^(serve_test|glm_tp_test|qwen_engine_test)$' \
  --output-on-failure -j 1 --timeout 900
```

All three suites passed in 164.69 seconds: 46 GLM cases, 61 service cases and
6 Qwen cases. Both depth-one and depth-two sampled-MTP gates passed, including
the moved port. The optional `glm_tp_forward_parity_real` checkpoint subcase
was skipped because `DGPP_TP_REAL_MODEL` was unset. The same native Qwen binary
then passed all 6 cases on Spark 2. These fixture tests use loopback worlds;
the physical two-rank check follows.

The tested server ran Qwen3.8-Flash-Next-NVFP4 across both Sparks with concurrency
4, MTP depth 3, 65,536 KV tokens and a 1 GiB prefix cache. Both logs identify
`0.1.0+g417286381778`. Serial and concurrent requests exercised scalar, batched
and padded graph launches. After the telemetry workload, counters reported
51 replays, 544 verification rows and 52 padded rows; histogram buckets 1–4
were 15, 11, 1 and 24 respectively, with all remaining buckets zero. Replay
and row arithmetic, speculative-counter totals, both metrics aliases and idle
retention checks passed. This synthetic workload is not a production padding
frequency measurement.

All nine `serve_api_check.py` checks passed. Both rank logs contained no ERROR
lines, and clean shutdown produced identical operation-stream MD5s:
`1eecc55e04a2bf5c7ad2a3fe55db39da`.

The original production release `0.1.0+g5d98ea6b7b13` was restored on both
ranks. Executable and resolved-configuration hashes matched their pre-test
values; the deployment and site files were unchanged. Binary SHA-256:
`3563cebf2c3821e4990d69641827d4acb4376c0c0090d090873c04f97b67967d`.
Health and three real requests passed; the second and third reused 1,128
prompt tokens. Metrics reported 148 prefix-cache slots and no engine failure.
The stop-test-restore window ran from 08:04:00 to 08:08:14 UTC.

Raw logs and before/after manifests are retained locally under
`artifacts/issue20-hardware/`. The full repository suite, optional real-GLM
checkpoint gate and four-node fabric validation were not run. No timing or
full-model numerical-equivalence claim is made. Subsequent record-only commits
do not change the tested implementation.
