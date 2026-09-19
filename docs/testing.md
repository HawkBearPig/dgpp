# Tests

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
cmake --build build-ci -j 4 --target unit_tests http_server_test serve_test fabric_serve_test scheduler_test roster_check
ctest --test-dir build-ci -L host -LE checkpoint --output-on-failure
ctest --test-dir build-ci -L python --output-on-failure
```

`DGPP_TEST_FILTER=<substring>` runs a subset of a binary's cases; the
loopback tests use fixed ports in the 299xx range. Run GPU/RDMA tests only
on idle test hardware, serially; do not run them alongside production serving.

The [Chat API extension validation record](openai-api-validation.md) lists
focused file/custom-tool/reasoning/schema checks, sanitizer commands and
the distinction between host conformance tests and live model validation.

CTest labels are `host`, `python`, `checkpoint`, `gpu`, `rdma` and `fixture`.
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
  loss);
- the M6 host suites: the tokenizer (55 byte-exact goldens per family,
  hash-keyed — GLM-5.3, Qwen3.8-Flash-Next and GLM-4.7 corpora), the chat
  template (26 / 20 / 26 goldens + 8 refusal negatives, the tool-call
  round trips and grammars over each real tokenizer), the scheduler
  (isolation, determinism, cancellation, bounded queue, tick ≡
  run_to_completion), the HTTP server's limit ladder, the OpenAI shapes
  over real HTTP with a fake engine, and the admission journal with two
  real peer loops over localhost;
- synthetic CUDA graph/eager parity;
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
  `glm_pick_test`); the fp4 GEMV
  core at every compiled K and the MoE layer's NVFP4 shared expert;
- the KDA operator suite: conv/recurrent kernel parity against host
  fp32/fp64 references, chunked-vs-unchunked bitwise equivalence, decode
  graph replay, snapshot round-trip, head-slice TP readiness, and
  reference-dump parity against the pure-python oracle;
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
  chunked prefill + decode smoke;
- DSA reference-dump parity: a pure-python oracle dump (bit-exact fp8 codec
  cross-checked against the C++ encoder) exercised through the full layer —
  latent cache bitwise, index cache within one e4m3 ulp, top-k exact — plus
  the torch backend against real checkpoint slices (executed on this box at
  32 and 2,052 tokens, the latter crossing the top-k horizon with zero
  flips); any flipped row is certified as a measured boundary near tie by
  the audit, never absorbed by tolerance;
- the M4 assembly suites: the mHC stream module (Sinkhorn mixing, stream
  update, final mean vs the double oracle), the MoE router/expert/shared
  path (sigmoid router with the noaux tie rule, swiglu asymmetries, bf16
  accumulation order, near-tie certification on synthetic corpora), the
  scale-aware GEMM against cuBLASLt BF16 references and real-checkpoint
  block edges, and the assembled-forward chain: a synthetic mini-checkpoint
  written on disk, a full-stack pure-python reference over the same
  weights, then the engine compared (hidden ulp budgets, top-k exact, route
  ids/weights, determinism);
- CUDA system-scope flag ordering, payload visibility, inactivity watchdog,
  and post-watchdog recovery;
- the M5 CollectiveBus data plane (needs the fabric + ibverbs): loopback
  scenarios with real RC QPs — payload integrity via fold-hash, credit
  recycling, dual-lane striping asserted on both sides, latency under bulk
  contention, watchdog failures, config-mismatch rejection, the graph era
  with the mixed-era interlude, the bulk RS+AG machine, and orderly stop —
  plus the app-level smoke;
- the TP forward over loopback buses (`glm_tp_test`): per-layer isolated
  parity vs the world=1 oracle at worlds 2 and 4, decode sessions, the
  greedy generation loop, the device pick, the one-graph MTP step in
  lockstep with the eager speculator, and the T=1/MTP serving graph adapters
  through the real scheduler (including slot reuse, the in-graph draft
  checked against the eager speculator after every replay, the MTP depth-2
  and depth-3 greedy gates — plain transcript, device feed equal to the
  eager chain's — and the sampled depth-2 gate against the eager
  speculator through its fallbacks; every gate's eager oracle drains the
  pipelined engine before it steps); the pick/spec kernels against their
  host oracles, the T=2 and T=3 sampled verdict chains included
  (`glm_pick_test`); the bus's graph era including two replay windows
  armed at once (`bus_test`); the GEMV cores (`bf16_gemv_test`) and the
  loader's resident-image round trip (`glm_loader_test`);
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
they require a peer and are documented under `benchmarks/README.md`. The same
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
requires bitwise equality. Its reproducible procedure and
measured bounds are in the [numerical record](../benchmarks/results/2026-09-18-glm-vision-numerics.md).
