# API validation boundary coverage — 2026-09-13 UTC

Base: `c7609af8b8c1c4d4fede4eb8aa795a06ab4e122e`.

This test-only change pins existing HTTP/SSE behavior using the host service
fixture's fake engine and frontend. It adds two tests and three rows to the
existing request-validation test; it does not change serving behavior.

The stream test reads through the final HTTP chunk terminator before checking
that omitted stream options, an empty options object, and explicit
`include_usage:false` produce neither a usage object nor a usage-only chunk.
The validation test checks the HTTP 400 error type and exact field name for
`stream_options:false` and `stream_options:{"include_usage":null}`. The three
message rows reject missing/null user content and missing tool content while
leaving existing accepted assistant tool-call histories covered unchanged.

## Mutation findings

Each mutation was applied separately to `src/serve/generation_service.cpp`,
then `cmake --build build-ci --target serve_test -j1` succeeded and
`DGPP_TEST_FILTER=<test-name> build-ci/serve_test` exited 1 with one assertion
failure. These were deliberate regressions of correct behavior, not observed
production bugs. The mutation source was restored after the experiment.

| Mutation | Filter (without `serve_` prefix) | Observed failure |
| --- | --- | --- |
| Make the final `if (r->include_usage)` unconditional | `chatStream_usageRequiresOptIn` | Usage object emitted without opt-in |
| Disable the non-object stream-options guard | `streamOptions_invalidTypesNameTheField` | HTTP 200 instead of named 400 |
| Disable the non-boolean include-usage guard | `streamOptions_invalidTypesNameTheField` | HTTP 200 instead of named 400 |
| Allow omitted user content | `tools_requestSideRendersThroughTheTemplateAndRefusesByName` | HTTP 200 instead of named 400 |
| Allow null user content | `tools_requestSideRendersThroughTheTemplateAndRefusesByName` | HTTP 200 instead of named 400 |
| Allow omitted tool content | `tools_requestSideRendersThroughTheTemplateAndRefusesByName` | HTTP 200 instead of named 400 |

All six targeted mutations were detected. This is a selected mutation check,
not a repository-wide mutation score, throughput measurement or model-quality
claim. No production/runtime mutation is part of the patch.

## Final scoped validation

The user limited final validation to unit tests and lightweight fake-service,
HTTP and scheduler cases, not the complete host or integration suite. The HTTP
fixtures use loopback sockets in a private network namespace and a fake engine;
they do not contact an external service.

A new build directory was configured with the ci preset (RelWithDebInfo,
DGPP_WERROR=ON). The initial broader host build was deliberately stopped during
unit_tests compilation when the scope narrowed, before any CTest execution.
Its systemd wrapper returned zero despite SIGTERM; this is an interrupted build,
not a pass. The scoped build resumed in that newly created directory, reusing
only objects produced by this clean run, not the previous build-ci directory.

```sh
cmake --preset ci -B build-ci-final-clean
cmake --build build-ci-final-clean --target unit_tests http_server_test serve_test scheduler_test -j1
ctest --test-dir build-ci-final-clean -R '^(unit_tests|http_server_test|serve_test|scheduler_test)$' --output-on-failure -V -j1 --timeout 120
```

All four requested native targets completed. CTest reported **4/4 entries passed,
zero failed**, in **100.82 seconds**. Binary summaries were unit_tests: 155 cases,
http_server_test: 13, serve_test: 36, and scheduler_test: 42, each with zero failed.
These are harness case counts, not a claim that all cases exercised their subject:
unit_tests printed **three no-CUDA-device skips**; **eight optional Qwen/GLM4
checkpoint cases** return early with the empty HOME. CTest itself reported no
skipped entries. The duplicate unit alias was not run or counted again.

The validation service ran as the unprivileged agent user, capped at 2 GiB memory,
zero swap, one CPU, 128 tasks and nice 10, with a 540-second runtime bound.
Private networking, temporary storage and devices were enabled; the filesystem
was read-only except for the new build directory and private temp. HOME/HF cache
locations were new and empty, CUDA visibility was empty, and GPU/RDMA device
paths were checked absent. The service is an isolation wrapper, not a systemd
integration test. The resumed job completed in 2 min 22.735 s with peak memory
344.4 MiB and peak swap 0 B. The interrupted first phase peaked at 611.1 MiB.
These are job observations, not serving benchmarks.

`git diff --check` passed. Runtime source matched the base. The final source diff
remained 49 added lines with SHA-256
`e3e20122d7dee8649636b0aee160002e9315fb7366e31e1569038dcea59d8581`.
Touched additions had been formatted with clang-format 19.1.7 and the repository
style before this clean build; no whole-tree reformatting was performed.

Earlier evidence remains historical: a pre-format 12-entry selected host run
passed in 112.69 seconds, and all six targeted mutations above were detected.
Those results are not additional coverage from the final four-entry run.

## Limitations and review

The final run excludes portability_test, setup_tools_test, launcher_test,
fabric_serve_test, roster_selftest, other Python/system-integration suites,
checkpoint-labelled suites, GPU/RDMA suites and real models. The broader planned
selection never reached CTest. No deployment, installation, production restart,
cache drop, model inference or unfiltered full-suite run was performed.

The added tests retain the existing DGPP_TEST, require, ServiceRig and post_chat
conventions. Assertions cover complete-stream usage opt-out and named validation
errors, and preserve the existing assistant tool-history checks. This is scoped
coverage of existing behavior, not a production bug fix, comprehensive API
alignment proof, model-quality claim or performance measurement.

Prepared with GPT-6 Astra assistance. No commit, push, PR or public post was made;
human review and explicit publication approval remain separate gates.
