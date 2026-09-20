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

GPU execution and four-node fabric validation are not performed. In particular,
compiling the graph adapter does not establish replay behavior on target
hardware or a timing improvement from moving the event record.
