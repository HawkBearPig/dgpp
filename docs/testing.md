# Tests

`resident_image_*` unit tests cover malformed offsets and lengths, truncated
blobs, default payload verification, its explicit opt-out and recoverable
read/size failures. `glm_loader_rebuilds_rejected_resident_images` and
`qwen_loader_rebuilds_rejected_resident_images` corrupt an image's table and
verify that the affected layer rebuilds, its neighbor restores and the repaired
cache is reusable. Run the GPU loader suites on idle hardware.

`serve_preparation_*` checks the fixed preparation pool at configured engine
concurrencies of 1, 2, 4 and 16, completion ordering, responsive health/SSE
traffic during blocked preparation, count/payload bounds, disconnects, worker
errors and shutdown. The service suite runs in both stream-flush modes.
`image_preparation_concurrent_formats_and_errors` covers simultaneous PNG,
JPEG and WebP decoding; `glm_vision_frontend_test` also shares the real loaded
template/tokenizer across text/image preparation and output decoding.
Set `DGPP_TP_REAL_MODEL` to select another cached GLM-5.3 checkpoint for that
frontend test; its default is `unsloth/GLM-5.3-Flash-FP8`.

`minijson_number_*` and `minijson_document_*` cover numeric grammar, integer/double
representations and document versus prefix parsing. `serve_json_*` exercises both
completion routes, file preprocessing and string-encoded function arguments with
the fake engine, including refusal before admission and valid requests afterward.
Run the service cases with both polling and `DGPP_TEST_STREAM_WAKE=1`; file-input,
safetensors, cluster-config and tool-parser unit tests cover shared-parser callers.

`serve_admission_bench_test` checks the HTTP/SSE admission benchmark's transport,
native-overlap assertions, published-counter drain barrier and KV cleanup gate.
It also checks optional controller metrics sampling against request IDs, sampler
failure reporting and thread cleanup when a request fails.
The [benchmark guide](benchmarks.md#reproduce) describes serving measurements. The scheduler
suite covers short-reader admission, aligned-share donation, global chunk-grid
tails and failure cleanup; `qwen_engine_test` exercises admission and donation
with a real two-rank engine, including MTP3 and row-independent transcript parity.
Host policy checks cover automatic admission and explicit overrides.
`scheduler_growOnDemand_*` covers cache reclamation before shedding, minimum-only
growth, LRU order, attached-entry protection and youngest-first shedding after
true exhaustion. Its shared-block fake tracks physical block identities: an
eviction can free zero blocks until the last snapshot reference is released.
Repeated runs check engine-operation and prefix-digest agreement. The physical
reservation tests also cover speculative trailing rows and context clamping.
`qwen_decode_test` also checks a donated
share followed by full work budgets across automatic model-grid boundaries,
with a retained structural snapshot and exact target/draft state against the
same explicit chunk cuts.

`shared_prefill_test` checks the common continuation and mixed-row contract
across GLM, GLM4, GLM-DSA, MiMo, Qwen35, DeepSeek V4 and V4.1. It covers eager
and graph adapters, MTP on/off, compact batches and output storage, cancellation
and slot reuse, snapshot attachment, stale cursors, decoder rollback, and
bitwise isolation when a neighboring reader changes. Sparse graph checks assert
that compact execution selects the smaller bucket, then compare its tokens
against an independent decoder; setting a flag alone is not coverage of the
mapping. Qwen35 includes FP8,
mixed NVFP4 and per-tensor FP8; V4.1 includes exact and bounded prefill with
retained tails larger than a grouped walk's workspace. Cross-shape comparisons
use the existing session numerical tier and certify changed argmax near ties;
same-shape isolation and graph token comparisons remain exact. Qwen4Exp retains
its two-rank `qwen_engine_test` coverage. Run GPU checks serially with production
stopped, then restore and verify the original deployment.
`qwen35_moe_*` adds the ModelOpt MoE fixture, row/group invariance and
independent reference/teacher comparisons. `qwen35_tp_test` covers MoE
worlds 1/2/4, native MTP graphs at worlds 2/4, and DFlash v1 graph
proposals at worlds 1/2, with multiple
request slots, page-boundary reserves and a small reduction buffer.
`shared_prefill_qwen35_moe` in `shared_prefill_test` exercises the current
shared-prefill protocol and native MTP. Run the focused set with:

```bash
python3 scripts/site_env.py run-rank --rank 0 -- ctest --test-dir build-ci --output-on-failure -j 1 -R '^(unit_tests|dflash2_kernels_test|qwen35_moe_.*|qwen35_tp_test)$'
python3 scripts/site_env.py run-rank --rank 0 -- env DGPP_TEST_FILTER=shared_prefill_qwen35 ctest --test-dir build-ci --output-on-failure -j 1 -R '^shared_prefill_test$'
```

`qwen_image_prefill_test` also destroys the caller's prompt, image and boundary
buffers after starting a shared `PrefillTask`. Its yielded execution must retain
the target and MTP results, snapshot chain and KV cleanup behavior. Run it with
`glm_vision_stream_test` and `glm_vision_frontend_test` when changing that lifetime.
`qwen35_image_prefill_test` and `qwen35_mixed_image_prefill_test` cover the same
lifetime for the 27B model with synthetic FP8 and mixed NVFP4 checkpoints. They
also compare grouped and interleaved image requests against isolated execution,
check cancellation followed by text slot reuse, and resume an attached prefix
whose MTP catch-up token is the first image row. Build `qwen35_image_prefill_test`
and run `ctest --test-dir build-ci --output-on-failure -R '^qwen35_(mixed_)?image_prefill_test$'`.

For serving comparisons, use the same deployment, checkpoint revisions, prompts
and output lengths for the previous and candidate executable. Warm long prompts
before measuring solo and concurrent decode, prefill, mixed traffic and repeated
arrivals. Record total elapsed time, first-token latency and decode update gaps.
Keep probability tracing separate from throughput measurements.

The scheduler latency controller has deterministic cost-feedback tests for growth,
slowdowns, decoder-load changes, alignment and explicit caps. A regression checks
that short completed groups cannot replace a full-quantum cost estimate after
the fair allocator trims their budgets. Journal tests check
that workers use the leader's limit and reject a missing adaptive decision;
`qwen_engines_adaptive_prefill_scheduled_verify` exercises changing budgets through
real continuations, mixed graphs and scheduled verification. Serving tests check
latency-target misses against queue and token-pass timing.
`dsv4_engines_adaptive_prefill_mixed_cancel_and_reuse` covers the adaptive policy
with a different architecture in eager and graph modes: compressor-aligned
growth, early short-reader output, cancellation of a yielded reader, slot reuse
and full KV release. It uses a small generated checkpoint.
For policy-only changes, use representative deployments with different execution
costs and single-/multi-rank operation, plus an architecture adapter sanity check.
Repeat the broader checkpoint matrix when an executor, state layout or numerical
kernel changes. A shared scheduler still encounters different adapter geometry
and costs; one checkpoint alone does not cover those differences.
Include cold-controller startup and transitions between light traffic, bursts,
queue/KV pressure and recovery. Check that fitting requests progress behind a
KV-blocked reader, overload is reported, cancellations drain and learned budgets
recover after load subsides. Compare the final candidate directly against the
PR base in alternating deployment order; a same-binary policy comparison answers
a separate question about the controller's contribution.

Check cached continuations against isolated execution, repeat the loaded
workload, and verify isolated recovery and retained KV usage. Investigate changed
transcripts with token probabilities and matching execution shapes; do not treat
temperature zero as a guarantee of identical floating-point reductions.
The Qwen cached-tail regression compares resumed short-reader hidden states,
logits and a subsequent forced continuation with isolated execution, in BF16
and FP8 dense modes. It also checks native verification when the decoder forms
its own kernel group. The executor isolates requested outputs of resumed readers
of at most 128 rows for their projection shapes while retaining the established
wide mixed-batch recipe. This does not impose bitwise equality between arbitrary
decode batch widths. Keep probability and layer tracing out of throughput measurements.
A one-token completion can contain only a special token; record completion and
usage without inventing a first-visible-token timestamp.

When defaults change, report both default-versus-default and matched-budget
comparisons. Matched budgets mean explicitly setting the same busy and idle
prefill limits on both builds; this is a benchmark control, not a separate
serving mode. To isolate fixed token budgeting on a build with the adaptive
controller, also set both latency targets to zero. For an adaptive-policy
comparison, match and report the targets and controller tuning instead.
Record the resolved admission policy as well. Equal token caps alone do not
guarantee identical selected quanta, physical row batches or admission timing.

Run GPU and fabric checks serially on idle hardware, then restore and verify the
original configuration and executable on every rank. Inventory checkpoint and
resident caches before downloading missing weights. Finish downloads and peer
synchronization before benchmarking, and remove only test-created files after
their checks finish. Keep raw results and profiles in ignored local storage or
external artifacts; summarize coverage, measurements and limitations in the PR.
See [benchmark procedures](benchmarking.md) and the maintained serving clients
in [the benchmark guide](benchmarks.md#reproduce).

`prefix_snapshot_test` forces a four-block K/V pool full at snapshot time in
Qwen and GLM. It covers monolithic and resumable prefill, unchanged target/draft
logits, full-block pin cleanup, skipped hop replacements, timer/slot reuse, and
propagation of non-pool errors. Build it with `qwen_forward_test` and
`glm_forward_test`, then run CTest with
`-R '^(qwen_forward_fixture|glm_forward_fixture|prefix_snapshot_test)$'`.
`scheduler_test` also checks that skipped hops never become cache entries,
including replacements of older snapshots, and that skipped prefill slots and
the `skipped_no_block` counter are handled correctly.

The sampled-fallback hop regression (#103) runs with
`DGPP_TEST_FILTER=prefix_hop_sampled_fallback` in `glm_tp_test` (MTP depths
1 and 2) and `qwen_engine_test` (depths 1 and 3). These are two-rank GPU
tests: GLM checks the exact eager sampled oracle and requires an older
snapshot to survive a skipped capture; Qwen compares growing conversations
with the cache disabled and enabled at matching prefill cuts, including
concurrent requests, retirement and slot reuse. Both require the skipped-hop
event to occur and the ranks' prefix decisions to agree. Run GLM with
`DGPP_PIPELINE=0` as well
to cover the synchronous path. `scheduler_test` covers first-hop skips,
preserved older snapshots and their later attachment, failed replacements,
and inconsistent or missing outcomes releasing the rolling slot.

The long-document prefix-cache regression is covered by `scheduler_test`
(changed tails, lookahead identity, small arenas and cleanup), `qwen_decode_test`
and `glm_tp_test` (two snapshots, resumed target/draft logits and budget changes).
The entry floor and the head cut (2026-09-28) are covered by `scheduler_test`
(`entryFloorKeepsShortPromptsOutOfTheArena`, `headCutServesTheNextConversation…`,
`headCutYieldsToTheDeepestCutWhenSlotsAreScarce`, `headCutThroughChunkedPrefill`),
`cluster_config_test` (the keys) and `fabric_serve_test` (the journal records).
Real-checkpoint checks also cover 32K/260K prompts, KV sharing and the recipe
memory plan.

`serve_test` covers assistant thinking history with the
`DGPP_TEST_FILTER=serve_thinkingParts_` cases: mixed visible and reasoning
parts, redacted/empty thinking, explicit reasoning precedence, tool-call
history, invalid field types and role restrictions. The UTF-8 case keeps
heap-backed reasoning alive through later messages and template preparation;
run these cases under the `asan` preset to check the owned-string lifetime.

`DGPP_TEST_FILTER=serve_timings_ build-ci/serve_test` covers completion timing
compatibility: cached prompts on both endpoints, queue exclusion with unchanged
TTFT/prefill/TPOT histograms, resumed prefill, streaming with usage omitted,
false or true, multiple choices exceeding available slots, single-token
completions and speculative batches. It also checks that existing usage, JSON
metrics and Prometheus token counters keep their definitions.

Release and testing builds use separate CMake presets and directories:

| Preset | Directory | Purpose |
| --- | --- | --- |
| `release` | `build-release/` | Production server; `-O3`, standard assertions disabled, no debug symbols, executable stripped at link time. The build preset builds the server and its dependencies. |
| `ci` | `build-ci/` | Testing; `RelWithDebInfo` (`-O2 -g -DNDEBUG`) with warnings as errors. The build preset builds all targets. |
| `debug` | `build-debug/` | Host debugging with symbols and standard assertions enabled. |
| `asan`, `ubsan` | `build-asan/`, `build-ubsan/` | Testing with the selected host sanitizer and debug symbols. |

CUDA device optimization remains enabled in release and CI; neither preset
uses the device-debugging flag `-G`. Release retains the project's floating-point
settings for numerical parity. Leave `DGPP_BUILD_DIR` unset to preserve the
separate directories. `scripts/ci-local.sh` uses `build-<preset>` when selecting
another preset with `DGPP_PRESET`.

Build the targets you intend to test. For a first checkout, host and Python
checks do not need model weights or GPU execution:

```bash
cmake --preset ci
cmake --build build-ci -j 4 --target unit_tests http_server_test serve_test fabric_serve_test scheduler_test roster_check dgpp_serve_app
ctest --test-dir build-ci -L host -LE checkpoint --output-on-failure
ctest --test-dir build-ci -L python --output-on-failure
```

These selections include `cluster_process_swap_test`'s four cheap policy and
mocked propagation checks. Run the four systemd integration checks explicitly
on a Linux host with systemd user scopes:

```bash
ctest --test-dir build-ci -L systemd --output-on-failure -V
```

To run the remaining CTest groups while explicitly excluding real systemd
scopes, use `ctest --test-dir build-ci -LE systemd --output-on-failure` after
building their targets and satisfying their hardware/checkpoint requirements.

`cluster_process_swap_systemd_test` is excluded from the `host` and `python`
labels so those selections do not start real scopes. Positive checks report
skips when systemd tools, a user manager or cgroup v2 memory/swap controls are
unavailable. The missing-bus failure checks still run without a working user
manager; they verify that scope failure never falls back to an unprotected
command. Once a scope is launched, a behavioral failure remains a failure.
Run this group when changing process launch, swap policy or version probes;
directly running `tests/python/cluster_process_swap_test.py -v` retains all
eight checks and reports individual skips.

`serve_startup_test` uses the built server for two- and four-process settings
handshakes over loopback. It deliberately fails a local HTTP capacity check
after settings adoption, before model loading, CUDA initialization or RDMA
setup. It is registered when the server target is enabled (libibverbs required).

`DGPP_TEST_FILTER=<substring>` runs a subset of a binary's cases; the
loopback tests use fixed ports in the 299xx range. Run GPU/RDMA tests only
on idle test hardware, serially; do not run them alongside production serving.

The common native harness reports each completed case as `[ OK ]`, `[FAIL]`
or `[SKIP] name: reason`. Its summary includes separate counts, for example
`12 tests, 0 failed, 10 passed, 2 skipped`; the total includes skipped cases.
Missing optional devices, checkpoints or an applicable corpus are skips,
not passing checks. Assertion failures remain failures. A mixed or all-skipped
harness run retains exit zero when no case fails; failure or an unmatched
filter returns nonzero. CTest reports the aggregate binary result, not
individual case skips. Standalone suites that already return skip code 2
when their prerequisites are absent retain that behavior. Read the case
counts before treating a passing aggregate result as hardware/model coverage.

The [Chat API extension validation record](openai-api-validation.md) lists
focused file/custom-tool/reasoning/schema checks, sanitizer commands and
the distinction between host conformance tests and live model validation.

CTest labels are `host`, `python`, `checkpoint`, `gpu`, `rdma`, `fixture` and
`systemd`.
Use `ctest --test-dir build-ci -N -L LABEL` to inspect a group before running
it. Fixture producers are added automatically when a selected test requires
them. Build all targets before a full CTest run. The `checkpoint` label marks
host tokenizer/template suites requiring real cached metadata. Some GPU suites
also contain optional real-checkpoint cases: a passing process exit does not
mean every case ran. Read skip messages; missing metadata returns CTest's skip
code in the tokenizer suites. Do not count skipped tests as model validation.

Native tests do not read `.env`. To pass the head node's cache/NIC overrides,
run `python3 scripts/site_env.py run-rank --rank 0 -- ctest --test-dir build-ci ...`.
Reference generators need the optional dependencies listed in
[setup instructions](getting-started.md#optional-development-and-evaluation-tools). CPU fixture generators do not require PyTorch;
torch-backed reference modes do.

## Evaluation data and prefill fixtures

The [GLM reference-suite guide](experiments/glm-reference-suite.md) describes
reference dump generation, reduced-layer comparisons and route certification.
Keep generated dumps and traces outside the tracked documentation.

Datasets are not bundled and benchmarks never download them implicitly:

```bash
python3 scripts/prepare_data.py download --tasks gsm8k humaneval
python3 scripts/prepare_data.py tokens --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json --text /path/to/long-prompt.txt
```

Downloads use pinned upstream revisions from OpenAI's
[GSM8K](https://github.com/openai/grade-school-math) and
[HumanEval](https://github.com/openai/human-eval) repositories. Each dataset has
a `.source.json` recording its source/license URL, row count and SHA-256.
Existing different content is never overwritten. `DGPP_DATA_DIR` defaults to
the ignored `data/` directory; use `--out` to select another preparation path.
Token generation needs `tokenizers` and uses the selected model's tokenizer;
generate a sufficiently long input for the prefill lengths you want to test.
`fabric_prefill_repeat.sh --ids-file FILE` selects an existing CSV fixture.

`serve_eval.py --data DIR` overrides the dataset directory;
`serve_prefill_probe.py --data FILE` selects the GSM8K JSONL file directly.
Use `--tasks gsm8k,extract` for evaluation without generated-code execution.
HumanEval requires `--allow-code-execution`, including via the serving wrappers.
It runs model-produced Python with a timeout, **not a security sandbox**.
Use a disposable isolated evaluation machine/container with no secrets,
unneeded mounts or network access. The flag acknowledges the risk; it does
not provide isolation. Never run it on a production serving node.

## The stream collectives (2026-09-14, plan D9)

`bus_test`'s `scenario_allreduce_stream` pins the stream-launched fold
(`CollectiveBus::allreduce_stream`): 100 generations per pass on one
stream, three passes, at worlds 2 and 4 — every destination bitwise the
canonical chain, the gates (a host-driven collective and a handout
rejected while stream generations are outstanding), a host-driven
one-shot between passes. `dsv41_tp_test`'s group-prefill gate runs the
walk through both reducers (`BusBoundaryReducer`, `BusStreamReducer`) and
requires every layer's rows and the logits bitwise between them, at 12
and 51 rows (the wide group's folds take the bulk path under both).

For bulk, latency-tail and stream-reducer memcheck coverage, run `bus_test`
with one rank per GPU on separate nodes. Start the following on every rank,
replacing `RANK` with 0 through `WORLD-1` and `HEAD` with rank 0's address:

```bash
/usr/local/cuda/bin/compute-sanitizer --tool memcheck \
  --error-exitcode 86 --leak-check full "$BUILD/bus_test" \
  --bulk-rank RANK --world WORLD --peer HEAD --port 29924
```

Use the same binary, world, port and lane selection on every rank. Run worlds
2 and 4, each with and without `--one-lane`. Each run checks 20 sizes, including
short words and partial segment tails, through bulk, latency-chunked and stream
reduction against the canonical BF16 chain. The normal collective watchdogs
remain enabled. Require every process to exit successfully and report zero
sanitizer errors and leaks.

The normal `DGPP_TEST_FILTER=allreduce_bulk` loopback matrix still exercises
multiple ranks in one process. On the tested CUDA 13.0 / Compute Sanitizer
2025.3.1 stack, that shared-context memcheck run stalls at phase transitions;
moving allocations and pinning readback buffers did not resolve it. Separate
GPUs complete the same matrix under memcheck. A timed-out loopback run is not
a sanitizer pass, and its absence of memory-error reports is not sufficient
validation.

## The near-tie rule of the batched engine gates (2026-09-14)

The session-core families lower their dense sites by the rows of a launch
(`kernels/gemm.hpp dense_gemv_rows`), so a batched graph step's rows and
the eager scalar's are tolerance-equal, not bitwise, and on the
random-weight fixtures a near tie flips and every later token follows.
`glm4_engine_test` and `glm_dsa_engine_test` therefore hold a batched
transcript to the eager one with `tests/cuda/engine_test_ties.hpp`: the
first three decisions agree, or the first difference among them sits on a
near tie (top-2 margin under 0.1) of the world-1 reference's own decision
(recorded by `engine_ties::margin_pick`); later differences are logged with
the position and the reference margin. The scalar transcripts (one row: the
GEMV chain, whatever the batch) stay bitwise. `DGPP_DENSE_GEMV_ROWS=256`
restores the old lowering and the bitwise agreement. The group-prefill
gates (`glm4_decode_test`, `glm_dsa_decode_test`, `qwen_decode_test`, and
`glm_tp_test`'s `glm_tp_group_prefill_matches_prefills_alone` for the Flash
class) compare a group's rows to the prefills alone under the row compare's
l2 and near-tie rule and audit a decode off the group's cache against the
re-forward (the Flash gate: against the solo decode, token by token).

## Suite coverage

Use `ctest --test-dir build-ci -N` to list the tests in your configured
build. The suites cover:

- host unit cases covering logging/tracing, JSON, arenas, safetensors,
  FP8, the latent cache's fp8/fp4 codecs (the e2m1 grid and its
  round-to-even ties, the row quantizers' error bounds, zero rows, the
  padded fp4 row), shard plans, the HF cache resolver, the sampler against its
  centralized oracle, KDA/DSA geometry contracts (against DESIGN §7.2's
  transcribed literals), route-trace golden bytes shared with the python
  reader, the MoE route-flip certifier's rejection paths (near-tie
  accepted; far-rank, zero-noise, own-scores-inconsistent, and duplicate-id
  divergences rejected), and the TCP/roster control plane (seal, epoch
  bumps, eviction by death and by deadline, rejection reasons, coordinator
  loss).
- the M6 host suites: the tokenizer (55 byte-exact goldens per family,
  hash-keyed — GLM-5.3, Qwen3.8-Flash-Next and GLM-4.7 corpora), the chat
  template (26 / 20 / 26 goldens + 8 refusal negatives, the tool-call
  round trips and grammars over each real tokenizer), the scheduler
  (isolation, determinism, cancellation, bounded queue, tick ≡
  run_to_completion), the HTTP server's limit ladder, the OpenAI shapes
  over real HTTP with a fake engine, and the admission journal with two
  real peer loops over localhost.
- synthetic CUDA graph/eager parity.
- QSA warp prefill against one-, three- and eight-split references in
  `qsa_test`, including tile edges and empty lists. `qwen_qsa_prefill` runs
  separate warp-on/off processes on a synthetic model with twelve query heads
  per KV head and an indexer budget of 2048. It compares every logit at
  127/128/129/256/513/1024 rows, checks exact repeats and the unchanged short
  path, and exercises session prefill followed by decode. This is numerical
  regression coverage, not real-checkpoint quality or performance evidence.
- the second and third families on the shared cores (2026-09-09/10): the
  Qwen3.8-Flash-Next chain (config/binding units, the loader, GDN/QSA/GR/
  PLE/MoE kernel oracles, the fixture forward against the pure-python
  reference, decode sessions, loopback TP worlds 2 and 4, the graph engines
  at world 2) and the GLM-4.7 chain (`glm4_config`/`glm4_binding` units,
  `glm4_loader_test` with the draft's NVFP4 requant, `glm4_attn_test`
  bitwise against the tile-aware host reference, `glm4_forward_test` vs
  `tools/glm4_reference_dump.py`, `glm4_decode_test` — prefill == forward
  bitwise, interleaved slots, chunked prefill, snapshots at any position,
  the speculator through the draft — `glm4_tp_test` worlds 2 and 4
  (ports 29946/29947), `glm4_engine_test` (29948/29949)) and the
  DeepSeek-V4.1-Flash chain (config/binding units, `dsv41_loader_test`,
  the CSA2 kernel and layer oracles, `dsv41_engram_layer_test`,
  `dsv41_model_test`, `dsv41_forward_test` vs `tools/dsv41_reference_dump.py`
  — strict teacher-forced and relaxed end to end, the DSpark rows included,
  the chain run twice: the exact walk and the bounded prefill (`--prefill
  bounded` / `--bounded`, the bounded end-to-end gate's hard-ulp budget
  measured from the exact chain's states over the same rows) —
  `dsv41_decode_test` with its certified selection flips (its §8: the
  bounded prefill within the window bitwise the exact mode's, the chunked
  bounded walk with a tail vs the one-shot, bounded prefix snapshots, the
  speculator after a bounded prefill), `dsv41_tp_test`
  worlds 2 and 4 on ports 29958/29959 with the layer-local twin and, on
  29966/29967, the same gates over a 24-row prompt (the decode form of the
  dense projections: the streaming tensor-core GEMM's sharded slices —
  the 70-row prompt prefills through the tile kernels and never meets it),
  (`dsv41_decode_test`'s certified flips and `dsv41_model_test`'s
  decode-vs-prefill rows also read the coded index query of both paths,
  `IndexLogits::q_codes`: a flip whose e4m3 codes differ is the fp8 coding's
  discontinuity — a code step is hundreds of ulps — and needs no near-tie
  gap; a flip with identical codes must be a near tie; the relaxed
  end-to-end parity run bounds its kept rows at l2 0.03 / 2 % hard, the
  cascade of certified flips through the fp8/fp4 caches, while the strict
  layer-local run keeps 0 hard; `glm_moe_test`'s
  `moe_grouped_mma_fp4_mx_matches_the_oracle_per_segment` pins the MXFP4 form
  of the fp4 tile kernel — the family's prefill experts — to a host oracle
  over ragged segments and tiles),
  the group prefill's gates — `dsv41_model_test` (three prompts as one
  walk bitwise the prefills alone, first steps included, in the exact
  mode and in the bounded mode with spans wider than the window),
  `dsv41_tp_test`
  on 29968/29969 (world 2, per layer, a 12-row group inside one latency
  slot and a 51-row group on the bulk fold path), `dsv41_engine_test`'s
  phase 4 (the graph engine's group prefill then the three-slot batch,
  transcripts the eager engine's), `scheduler_test` (queued short prompts
  admit as one group; a prompt past the span limit admits alone) —
  `dsv41_engine_test` on 29961–29964 with the DSpark graph engine (its
  world-2-vs-world-1 rule: the first three decisions agree or flip on a
  world-1 top-2 margin under 0.1, recorded by the world-1 pick — the folds
  reassociate in bf16, so a bare prefix rule passes or fails on which side
  of a near tie a kernel's rounding falls; 29964:
  the scheduled verify depth — a forced, varied per-step depth over the
  reduced-row scalar variants, forced per-slot depths over the reduced-row
  batch variants (the compacted feeds and masks), then a scalar step over
  the batch's published confidence; every transcript the plain eager
  engine's), `glm4_engine_test` on 29953 and `glm_tp_test` on 29940 with
  the same schedule from the draft head's probabilities (depth-2 worlds
  with real sampler scratch; the GLM-5.3-Flash gate runs the eager
  speculator in lockstep at the engine's per-step depth),
  `dsv41_tokenizer_test` and `dsv41_prompt_test` against the snapshot's own
  tokenizer and encoder; the DSML tool grammar's walks in `unit_tests`'
  `tool_grammar_test`; the six-row sampled verdict oracle in
  `glm_pick_test`); the MiMo-V2.6-Flash chain (2026-09-22:
  `mimo_config`/`mimo_binding` units over the real config.json,
  `mimo_attn_test` — the fused-projection finish and the windowed sink
  attention bitwise against the host oracle over window / sink / split /
  hpk configurations, the one-launch decode attention bitwise the
  three-kernel chain (bf16 and fp8 caches, same-request batch rows,
  self-resetting counters) and the query-tiled tensor-core prefill
  attention against the split-KV chain at the oracle's statistic;
  `add_rmsnorm_test` — the fused residual add + norm bitwise the
  two-kernel chain over random rows of several dims and magnitude
  profiles; `fp4_gemv_test` additionally pins the MXFP4 f16 fast path's
  window boundaries against the double oracle; `mimo_loader_test` — every class byte-exact at
  worlds 1, 2 and 4 with the padded chunk stacking and the ignored
  encoders, `mimo_forward_test` vs `tools/mimo_reference_dump.py` (strict
  teacher-forced and relaxed end to end, the draft rows included),
  `mimo_decode_test` — prefill == forward bitwise, the batched rows bitwise
  their scalar rows through four rows, certified routing flips, the window
  verified at the session level (positions past it leave a sliding-window
  layer's rows bitwise unmoved), `mimo_tp_test` worlds 2 and 4 on
  29962/29963, `mimo_engine_test` on 29908/29909/29911/29913 with MTP
  depths 1 and 2 and the scheduled depth; `mimo_tokenizer_test` and
  `mimo_chat_template_test` against the snapshot's own tokenizer and
  template, the compact `<tool_call>` dialect in `tool_parser_test` /
  `tool_grammar_test`); the DeepSeek-V4-Flash chain (2026-10-01:
  `dsv4_config`/`dsv4_binding` units over the release's config and its
  72,317 tensors, `dsv4_loader_test` — every class byte-exact at worlds 1,
  2 and 4, the resident image round trip, a geometry the world does not
  divide refused by name — `dsv4_model_test` — prefill == forward, a
  chunked prefill bitwise the one-shot at any cut (128/256 and
  101/133/290), hot == cold prefix snapshots, the group prefill bitwise the
  prefills alone, the resumable prefill and its group advance (three
  prompts read in through one walk per tick beside a decoding peer, a
  cancelled cursor) bitwise the one-shot, verify and rollback with
  certified selection and routing flips, the DSpark draft rows and chain,
  the screened draft head's picks against the BF16 head's —
  `dsv4_tp_test` worlds 2 and 4 on 29991–29998 / 30003–30004 (a 150-row
  walk, grouped prefills through the stream-ordered reducer, the decode
  rows) and the prefill's fold overlap on one rank with a reducer that
  takes every asynchronous fold (a prompt and a group's spans in two row
  blocks bitwise the one-block walk), `dsv4_engine_test` on
  29986–29990 and 30005–30007 (graph == eager, DSpark depths == plain
  decode, the scheduled verify depth, a sampled slot under the schedule and
  two sampled slots batched through every depth with host fallbacks, six
  slots batched at depth 4, at depth 5 on the depth-capped family and at a
  fixed depth 5 split past the widest family), `dsv4_tokenizer_test` and
  `dsv4_prompt_test` against the snapshot's own tokenizer and
  `encoding_dsv4.py`; `dsa_test`'s 64-head release-shape decode select
  certified against fp64 logits, `mma_gemv_test`'s dense bf16-weight tile
  kernel (oracle budget, row-invariant across chunkings, the gated ring
  form bitwise the plain rows), `glm_moe_test`'s
  early shared-slot order bitwise the default one and the phased prefill
  (the router and the accumulation per row block) bitwise the one-call
  path, `scheduler_test`'s group advance (prompts queued together begin
  together, max-min fair then levelled shares, the busy quantum per
  reading prompt); the real checkpoint
  against the release's own `inference/model.py`, layer by layer, with
  `tools/dsv4_torch_reference.py`); the fp4 GEMV
  core at every compiled K (the MXFP4 set 2048 / 4096 included) and the
  MoE layer's NVFP4 shared expert.
- the KDA operator suite: conv/recurrent kernel parity against host
  fp32/fp64 references, chunked-vs-unchunked bitwise equivalence, decode
  graph replay, snapshot round-trip, head-slice TP readiness, and
  reference-dump parity against the pure-python oracle.
- the DSA suite (M3): pool compression and tail-ring continuation verified
  bitwise via hard-max gates (including multi-token decode == single-token),
  a 1,100-case bitwise pooled top-k fuzz around pool boundaries plus
  exact-tie and 512th-boundary constructions, the fused decode select (MTP
  multi-row, grid-size invariance, 100k-pool long-context stripes, graph
  capture/replay with changed position), split-KV absorbed attention against
  the host oracle at TP1/TP4 with empty-row and head-group coverage,
  latent/gather block-table round trip, the quantized latent cache (the
  fp8/fp4 append bitwise against the host codec at 512/256/32 wide; the
  split, listed-flash and dense-flash attention kernels over an fp8/fp4
  cache against the oracle fed the dequantized rows, within the bf16
  kernel's own tolerance; a whole layer on each quantized cache against
  the bf16 layer — same selection, output drift within the format's
  bound), multi-request decode with padding rows, kpool=2 generality, and
  the layer tests: state-pool block
  allocation and byte accounting at deployment scale, prefill/chunked-
  prefill/decode parity against the oracle with selection-aware near-tie
  certification (at select_k=16 and select_k=8), decode graph replay
  bitwise across positions, TP2 head-slice vs TP1, and a real-geometry
  chunked prefill + decode smoke.
- DSA reference-dump parity: a pure-python oracle dump (bit-exact fp8 codec
  cross-checked against the C++ encoder) exercised through the full layer —
  latent cache bitwise, index cache within one e4m3 ulp, top-k exact — plus
  the torch backend against real checkpoint slices (executed on this box at
  32 and 2,052 tokens, the latter crossing the top-k horizon with zero
  flips); any flipped row is certified as a measured boundary near tie by
  the audit, never absorbed by tolerance.
- the M4 assembly suites: the mHC stream module (Sinkhorn mixing, stream
  update, final mean vs the double oracle), the MoE router/expert/shared
  path (sigmoid router with the noaux tie rule, swiglu asymmetries, bf16
  accumulation order, near-tie certification on synthetic corpora), the
  scale-aware GEMM against cuBLASLt BF16 references and real-checkpoint
  block edges, and the assembled-forward chain: a synthetic mini-checkpoint
  written on disk, a full-stack pure-python reference over the same
  weights, then the engine compared (hidden ulp budgets, top-k exact, route
  ids/weights, determinism).
- CUDA system-scope flag ordering, payload visibility, inactivity watchdog,
  and post-watchdog recovery.
- the M5 CollectiveBus data plane (needs the fabric + ibverbs): loopback
  scenarios with real RC QPs — payload integrity via fold-hash, credit
  recycling, dual-lane striping asserted on both sides, latency under bulk
  contention, watchdog failures, config-mismatch rejection, the graph era
  with the mixed-era interlude, the bulk RS+AG machine, and orderly stop —
  plus the app-level smoke.
- the TP forward over loopback buses (`glm_tp_test`): per-layer isolated
  parity vs the world=1 oracle at worlds 2 and 4, decode sessions, the
  greedy generation loop, the device pick, the one-graph MTP step in
  lockstep with the eager speculator, and the T=1/MTP serving graph adapters
  through the real scheduler (including slot reuse, the in-graph draft
  checked against the eager speculator after every replay, the MTP depth-2
  and depth-3 greedy gates — plain transcript, device feed equal to the
  eager chain's — and the sampled depth-2 gate against the eager
  speculator through its fallbacks; observer events keep the live transcript
  checks active through retirement, including grouped/staggered admissions
  and slot reuse. `glm_mtp_sampling_no_pipeline` repeats the sampled gate
  with `DGPP_PIPELINE=0`. Every gate's eager oracle drains the
  pipelined engine before it steps); the pick/spec kernels against their
  host oracles, the T=2 and T=3 sampled verdict chains included
  (`glm_pick_test`); the bus's graph era including two replay windows
  armed at once (`bus_test`); the GEMV cores (`bf16_gemv_test`) and the
  loader's resident-image round trip (`glm_loader_test`).
- Python checkpoint classification, exact expert-occupancy tests, and the
  route-trace traffic-model contract.

`glm_tp_test` and `bus_test` run with
`CUDA_DEVICE_MAX_CONNECTIONS=32` and prefetch enabled. Capture-time
`check_decode_graph` rejects memcpy, memset, event-record and other
unsupported nodes. Kernel and empty nodes are allowed, along with a
model-declared number of host callbacks for Qwen's mapped n-gram gather.
The copy-engine restriction prevents dependency cycles between ranks in
one process. The [graph-stall investigation](batched_mtp_graph_stall.md)
records the reproducer and validation at 1 and 32 connections.

The recorded CUDA sanitizer runs cover `compute-sanitizer` memcheck (full
suite every milestone; racecheck and initcheck per-phase on the tests
exercising new kernel shapes). The multi-rank loopback worlds of
`glm_tp_test` are the exception: their budgets are liftable
(`DGPP_TEST_BUS_TIMEOUT_MS`, `DGPP_TEST_CONSUMER_DEADLINE_S`,
`DGPP_TEST_WAIT_TIMEOUT_MS`), but even lifted, the instrumented eager prefill
outlasts the first collective's kernel deadline (baseline and current alike,
2026-09-04 record entry), so those worlds report 0 errors and exit on a
budget rather than completing. The sanitizer findings that motivated this
(speculated loads past short-circuit guards, shared-memory reuse races
that pass by scheduling luck, undersized test buffers that made a graph test
pass vacuously) are pinned in `DESIGN.md` §12.

Cross-node RoCE and NIC→GPU checks are intentionally manual/deployment tests;
they require a peer and are documented in [benchmark procedures](benchmarking.md). The same
holds for the release checks: long, real-checkpoint cluster runs that no CI job
starts, each behind an explicit acknowledgement — today
`scripts/qwen_yarn_release_check.py` (`docs/qwen_yarn_release_check.md`), the
YaRN acceptance run near 256K and 512K.

## Vision arithmetic

`glm_vision_test` covers eager attention score rounding, softmax size boundaries,
batched GEMM layouts and strides, preservation of reductions across query tiles,
fused bias rounding, unbiased split-K cancellation, LayerNorm/GELU CUDA
reference hashes and attention row placement. The optional
`tools/glm_vision_norm_reference.py` regenerates the normalization hashes;
inspect backend changes before updating them. Run the CUDA tests with the existing
`bf16_gemv_test` after rebuilding the targets. Checkpoint-level validation uses
`tools/glm_vision_reference.py`; its default CUDA eager gate applies to the full
encoder as well as isolated stages. The pinned 30-case runner additionally
requires bitwise equality. Follow the [vision oracle procedure](vision.md)
for reference settings and numerical bounds.
