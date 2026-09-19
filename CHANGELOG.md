# Changelog

The history by milestone. The dated engineering record in
`benchmarks/results/` has every measurement and fix behind these lines;
`PLAN.md` has the milestones' exit gates.

## Unreleased

- **Streaming images and GLM prefill scheduling** (2026-09-19): remove
  history-wide image count/token caps. Stage visual embeddings through fixed
  single-image and chunk buffers, including MTP lookahead and cached suffixes.
  Bound decoded request pixels and retained prefix identities by bytes.
  Enable GLM-5.3-Flash graph prefill continuations: active decodes run between
  bounded chunks, with cancellation and prefix snapshots preserved. The
  four-rank deployment uses 256-token busy and 2,048-token idle budgets.
  Scan incoming journal frames incrementally so large pixel payloads do not
  delay admission with repeated scans of the entire buffered prefix.
- **Image-aware prefix caching** (2026-09-18): reuse GLM image prompts and
  generated continuations across turns instead of reprocessing the entire
  conversation. Compare pixels, geometry and token positions exactly, account
  for MTP lookahead, and restore uncached image embeddings during suffix prefill.
- **GLM-5.3-Flash image inputs** (2026-09-18): accept PNG/JPEG data URIs
  through Chat Completions, journal processed pixels across ranks, and run
  the native BF16 vision encoder before prompt prefill, including MTP.
  Report model input modalities and visual prompt-token usage. Add encoder reference checks, API tests
  and [usage documentation](docs/vision.md). Match the CUDA BF16 eager reference
  bitwise across the pinned regression corpus after correcting rounding and
  reduction arithmetic; see the [numerical investigation](benchmarks/results/2026-09-18-glm-vision-numerics.md).
- **Release packaging with bundled PCRE2** (2026-09-18): exclude the dependency's
  standalone install rules from DGPP's runtime package. Server-only builds
  no longer fail packaging because the unused `libpcre2-posix.a` was never
  built. Add a host-only build/install regression test using the actual
  dependency configuration.
- **Chat extension edge-case validation** (2026-09-18): cover malformed file
  inputs, storage quota races, multipart boundaries, custom-tool streaming
  and history, native reasoning mappings, schema bounds/references/formats,
  and malformed Lark grammars. Fix escaped file detection, corrupt metadata
  handling, array-bound overflow, missing pointer array traversal, impossible
  scalar enums, Lark escapes/cycles, leap-second validation, custom error
  fields/history IDs and duplicate thinking globals. Record reproducible
  checks and remaining limits in `docs/openai-api-validation.md`.
- **Document inputs and richer Chat tools** (2026-09-18): add inline and
  uploaded file inputs, bounded asynchronous PDF/text extraction and a
  configurable persistent file store. Add custom tools with text, regex and
  Lark formats; map reasoning effort through each model's native controls.
  Enforce local/recursive JSON Schema references, string patterns and formats,
  and exact decimal multiples during constrained decoding. Document limits,
  model mappings and the rank-0 `poppler-utils` dependency in
  `docs/openai-compatibility.md`.
- **OpenCode provider-option compatibility** (2026-09-18): restore acceptance
  of unknown top-level client extensions, including `preserveThinking`.
  The API audit's strict allowlist incorrectly rejected previously working
  requests. Supported options remain validated and known unsupported API
  features still produce explicit errors. Add streaming and one-shot coverage.
- **Chat API audit and live prefill metrics** (2026-09-18): align nullable
  options, developer messages, numeric bounds, streamed usage and obfuscation
  with the OpenAI reference; validate unsupported capabilities explicitly.
  Correct visible-content logprobs, transport error envelopes and HTTP request
  counters for multiple choices. Publish per-request prefill progress while a
  scheduler pass is still running. Document the supported text/tool profile
  and the DGPP metrics extension in `docs/openai-compatibility.md`.
- **Release and testing builds** (2026-09-18): the release preset builds the
  production server with explicit `-O3`, no debug information, and link-time
  stripping. Deployment defaults to `build-release/dgpp-serve`; testing keeps
  `build-ci/` and debug symbols, selectable with `--bin`. Release packaging
  rejects testing build directories and debug/sanitizer artifacts.
- **Configurable HTTP body limit** (2026-09-18): `http.max_body_bytes` in
  deployment JSON, or `--http-max-body-bytes`, controls the serialized request
  limit. The default rises from 4 MiB to 256 MiB for large document prefills
  and agent histories. HTTP 413 responses name the configured byte cap;
  this limit is independent of KV capacity and does not preallocate buffers.
- **Metrics endpoint alias** (2026-09-18): `/metrics` exposes the same JSON
  counters as `/v1/metrics`, which remains available for backward compatibility.
  Prometheus exposition is still a separate follow-up.
- **Agent API compatibility** (2026-09-17): JSON Schema numeric bounds now
  constrain `number` values, including fractions and exponents. Tools can
  accompany `response_format`: auto chooses a tool-call turn or a structured
  answer, including after tool results; required/named/none and parallel
  call controls retain their guarantees. `reasoning_effort: xhigh` is accepted
  through both API spellings and passed to the checkpoint template. Qwen
  renders low/medium directly and high/xhigh as xhigh.
- **Qwen wide batches and opt-in prefill continuation** (2026-09-15):
  sixteen decode rows support eight native-MTP requests. Small-row fused
  kernels and deployment defaults are preserved. Fitting physical slot
  prefixes remain batchable when a deeper full batch exceeds the model's
  row limit. `engine.prefill_budget_tokens` gives Qwen graph decoding a
  turn between bounded prefill chunks; the cursor retains target/draft
  state, reservations and snapshot ownership through cancellation and
  cache attach. The policy is journaled across ranks. Wide draft hidden
  projections use a tensor-core kernel to avoid cuBLASLt copy-engine nodes
  in collective graphs. Benchmark scripts now distinguish request-wall
  and legacy output-span rates, require server token usage, and provide
  repeated C1 comparison and long-prompt interference checks. See the
  [validation and measurements](benchmarks/results/2026-09-15-qwen-batching-prefill.md)
  for configuration tradeoffs, the unresolved C1 promotion gate, and remaining work.
- **One deployment template per model, quant and world** (2026-09-14, late;
  `deploy/README.md`): twenty-five templates became eight —
  `cluster_<model>_<quant>_w<n>.example.json`, each with MTP at the depth its
  family measured best, the decode graph, and the slot count and cache
  budget that measured at or above the other shapes (the six-slot DeepSeek
  shape, the eight-slot full GLM-5.3, the large-cache GLM-5.3-Flash on four
  nodes, the FP8 dense stack on the single-Spark Qwen). The retired shapes
  are knobs (`--no-mtp` is new; `--mtp-depth`, `--max-concurrency`,
  `--kv-capacity`, `--kv-dtype`, `--prefix-cache-gib`, `--dense-weights`
  were there) and the catalogue maps every retired name to its knobs. The
  default deployment (`scripts/site_env.py`) and the resolved-config fixture
  follow the four-node GLM-5.3-Flash template; historical changelog entries
  and benchmark records keep the old names.
- **The GPU-driven eager fold (plan D9)** (2026-09-14, late;
  `CollectiveBus::allreduce_stream` / `allreduce_settle`, `BusStreamReducer`):
  the eager walk's boundary reductions launch on the model's stream in the
  graph kernel form — no host drain of the model stream before each fold,
  no engine launch on the collective stream, no host notice of the finish
  before the model continues; the generation comes from the shared
  counter at submit, the engine posts from a FIFO through the replay
  walk's own flight, the model settles once per pass. DeepSeek-V4.1-Flash
  serves on it by default (`DGPP_DSV41_EAGER_FOLD=1` restores the
  host-driven reducer; the other families keep it). The fold is bitwise
  the host-driven one (bus_test's stream scenario against the oracle at
  worlds 2 and 4; dsv41_tp_test's group prefill through both reducers,
  every layer bitwise). Fabric, the six-slot config: TTFT 0.30 → 0.27 s at
  one stream and 0.84 → 0.73 s at six, the aggregates unchanged, the fold
  kernel 542 → 386 µs at 47 rows (docs/measurements.md).
- **The dense lowering of the session-core families follows the rows of a
  launch** (2026-09-14, `kernels/gemm.hpp dense_gemv_rows`, the site setting
  `DGPP_DENSE_GEMV_ROWS`, default 4; 256 restores the old lowering): GLM-4.7,
  the full GLM-5.3, Qwen3.8-Flash-Next and GLM-5.3-Flash lower their dense
  sites by row count — the row-independent GEMV chunks (and Qwen's fused
  multi-problem launches) to four rows, cuBLASLt's algorithm for bf16 rows
  above (at the weight-stream floor from six rows: the chunks re-read the
  weights per four rows, 2–8× the bytes at 6–32 rows: `bf16_gemv_test`'s
  table), the streaming tensor-core GEMM (`kernels/mma_gemv`, now with an
  asynchronous per-warp weight ring and a block width chosen by n) for fp8
  rows from five to 256 (the full GLM-5.3 DSA projections at sixteen rows:
  o_proj 320 → 109 µs, q_a 155 → 59, q_b 104 → 38; Qwen's FP8 dense stack
  and the shared expert; the Flash dense MLP) and the 128-row dense kernel
  above. T = 1 keeps its chain; a batched step's rows and the eager
  scalar's are tolerance-equal, not bitwise (the engine gates hold the
  first decisions exactly unless the flipped decision was a near tie of the
  world-1 reference — `tests/cuda/engine_test_ties.hpp` — and report the
  rest). DeepSeek-V4.1-Flash keeps its every-row tensor-core lowering.
  Fabric A/B: docs/measurements.md.
- **Group prefill for GLM-4.7, the full GLM-5.3, Qwen3.8-Flash-Next and
  GLM-5.3-Flash** (2026-09-14): the families expose
  `prefill_group_span_limit()`, so the scheduler admits queued cold prompts
  together and prefills them as the spans of one walk — GLM-4.7's attention
  rows carry their own request ids and positions already; the full
  GLM-5.3's DSA layer runs per span with a selection-scratch base per span
  (`DsaLayer::enqueue_prefill(row_base)`, the selection-reuse contract per
  base); Qwen's GDN scan and QSA attention run per span; GLM-5.3-Flash's
  own walk (`GlmDiagnosticModel::session_prefill_group`) runs its KDA scan
  and DSA attention per span and mirrors every span's last row (a group
  of at most kDecodeRows spans). Gates: a group's rows against the
  prefills alone (GLM-4.7 and Qwen bitwise on their fixtures; the DSA
  family's rows within 6.5e-3 relative l2 with the top-1 equal; the Flash
  fixture's within 2e-7, its decode off the group's cache bitwise the solo
  decode) and a decode off the group's cache against the re-forward.

- **Streaming tensor-core decode GEMM for DeepSeek-V4.1-Flash** (2026-09-14,
  `kernels/mma_gemv.{hpp,cu}`; default on for this family, `DGPP_DSV41_DENSE_GEMV=1`
  restores the chunks): the dense fp8 projections, the engram wkv, the
  draft's `main_proj` and the bf16 lm head at 1–32 decode rows run one
  launch that streams every weight once through `mma.sync` (the 4-row GEMV
  chunks re-read the weights per chunk: 82 ms of a 252 ms six-slot step).
  The dequantized values are exact for the e8m0 scales, so only the fp32
  order differs from the chunks (transcripts reorder, accuracy held), and a
  row's chain never depends on the rows sharing its launch (a batched row
  stays bitwise the row alone). Cold, fp8 [5120 × 5120]: 132/133/136/142 µs
  at 1/6/16/30 rows against 118/205/419/764; the prefill's rows take the
  same kernel in 64- and 128-row forms and 128-row groups above (192 / 288
  / 567 / 4,558 µs at 64 / 128 / 256 / 2,048 rows against the chunks' 1,507
  / 2,978 and the tile kernel's 3,533 / 23,360). Fabric: docs/measurements.md.
  The fixture gates that certify selection flips now read the coded index
  query of both paths (`Dsv41Model::IndexLogits::q_codes`): a flip whose
  e4m3 codes differ between the paths is the coding's discontinuity,
  whatever the reference gap.
- **Group prefill** (2026-09-14, `SchedulerEngine::prefill_group`,
  `SessionModel::session_prefill_group`, `Scheduler::admissible_group`):
  queued cold prompts (no prefix-cache attach or snapshot; any width up
  to a walk's rows) admit together and prefill as the spans of ONE forward
  — the dense sites, the MoE, the norms, the Engram and the head over
  every row, the CSA2 attention and kv publication per span, and under the
  bounded prefill each span's decoder segment (its last window rows)
  packed span after span — instead of one read-in per tick with a decode
  step between.
  A prompt prefilled in a group is bitwise the prompt alone (gates at the
  model, the TP world and the engine): the mHC dots take their tiled form
  at every prefill row count for this family (`mhc_set_tile_min_tokens`),
  and an opted-in cuBLASLt instance lowers every bf16 row count to the
  tensor-core form. Other families keep one prefill per tick.
- **Adaptive λ for the scheduled verify depth** (2026-09-14,
  `engine.mtp_schedule_adapt`, on by default with the schedule): λ follows
  an EWMA of each MTP step's committed tokens over its modeled time
  (base + rows·row), floored at the configured λ — replicated inputs, one
  fixed arithmetic, every rank the same value. One configuration now
  serves every concurrency: C1 48.2 / C2 66.8 / C6 87.0 on the six-slot
  world (docs/measurements.md).
- **The batched replay's verify depth is the batch's own Dinkelbach rule**
  (2026-09-14, `scheduled_verify_depth_batch`): one more draft position
  costs a verify row per live slot and yields the sum of the slots' prefix
  survivals, so the batch verifies position i while the MEAN survival over
  its slots beats λ·row — where the earlier rule took the deepest slot's
  own depth, verifying rows for every slot that one confident slot alone
  justified. Exact at any depth (each slot commits its greedy prefix);
  the test hook still forces per slot. Measured on the six-slot world:
  docs/measurements.md.
- **MXFP4 form of the fp4 tensor-core expert kernel** (2026-09-14,
  `moe_grouped_mma_fp4_ldm_kernel<OutT, kGroup>`, `fp4_group` on the grouped
  launchers): DeepSeek-V4.1-Flash's routed experts (e2m1 + e8m0 per 32, no
  global) prefill through the ldmatrix tile kernel instead of the 4-row
  GEMV core (the weights re-read per 4 rows: a third of a prompt's kernel
  time). The pair is decoded in fp32 — the e8m0 scale can leave f16's range
  — and is exact; `glm_moe_test` pins the grouped launch to a host oracle
  (gate within a bf16 rounding, down within 4e-5) and the z split bitwise.
- **Confidence-scheduled verify depth for DSpark** (2026-09-14,
  `engine.mtp_schedule`, off by default): a greedy request verifies only
  the leading drafts whose prefix-survival probability (the product of the
  confidence head's per-position acceptance probabilities) beats the value
  of a verify row, each depth on its own captured graph variant; the
  batched replay and sampled requests keep the whole block. The rule is
  the Dinkelbach optimum of aggregate tokens per second (a per-step ratio
  would be the wrong surrogate), its constants are the world's so every
  rank derives the same depth, and the committed transcript is the plain
  greedy one at every depth — proven on the eager speculator and on the
  graph engine over a forced, varied depth sequence
  (`tests/unit/verify_schedule_test.cpp`, `dsv41_decode_test` §7b,
  `dsv41_engine_test` on 29964). On the fabric (four nodes, greedy, 300
  tokens per class): chat 33.4 → 24.6, prose 27.4 → 22.7, code 21.9 → 18.9,
  json 21.9 → 19.1, math 21.8 → 19.8 ms/token with λ at the achieved
  throughput (0.045 tok/ms; the reservation-rate default is within 2–4 %),
  transcripts identical, op streams identical across the ranks. No change
  to any configuration that does not enable it: the captures and the
  replay path are the same (the baseline run reproduced the previous
  transcripts and timings). The DeepSeek-V4.1 deploy example enables it.
  Extended the same day to the batched replay (one depth per batch, the
  deepest a live slot asks for, on reduced-row batch variants over the
  compacted feeds) and to every MTP family: without a confidence head the
  engine takes the draft head's own probability of each draft from the
  device sampler (the draft picks report logprobs; the tokens are
  unchanged) — GLM-5.3-Flash, Qwen3.8-Flash-Next, GLM-4.7 and the full
  GLM-5.3 can schedule at `mtp_depth` 2 and above (`glm4_engine_test`,
  `glm_tp_test`). Measured: the batched DeepSeek replay at two live
  requests prose 34 → 51 and chat 37 → 54 tok/s aggregate (exact); the
  GLM families at depth 2 exact and throughput-neutral (one row at stake).
  A stale settle bound after a reduced-depth batch, found by the two-request
  probe, is fixed and gated.
- **DeepSeek-V4.1-Flash decode window attention split** (2026-09-14): an
  nsys profile found the decode window attention ran as a single
  latency-bound thread block (`n_split=1`), the largest non-GEMV cost in
  the step (~140 us a layer, the GEMVs themselves at 70-88 % of peak). The
  decode path now splits the window a fixed eight ways to fill the SMs
  (prefill stays unsplit and byte-identical): chat 38.3 to 34.3, code 25.3
  to 21.8 ms/token, gsm8k 60/60 unchanged, cross-rank determinism kept.
  Softmax attention cannot be split bitwise-invariantly, so decode's window
  is no longer bit-identical to the forward's block (about one bf16 ULP,
  which shifts greedy token sequences at near-ties without changing
  accuracy); the decode-vs-forward fixture checks were re-calibrated to a
  rounding budget plus an argmax-match assertion, the certified-flip decode
  audit unchanged.
- **DeepSeek-V4.1-Flash is a served family** (2026-09-14): the CED
  encoder/decoder with CSA2 attention (a 128-token sliding window plus a
  compressed-KV path selected by a two-level indexer), single-pass
  hyper-connections, the two Engram n-gram tables mapped from NVMe, and
  the DSpark block draft (five drafts verified per pass, `kSpecRows` 6)
  ride the shared loader, paged-block and session cores. The MXFP4/FP8
  checkpoint serves as shipped (the quantization study found nothing to
  gain beyond it). A bounded prefill (the model's own SWA-replay: the
  encoder over the whole prompt, the decoder over the last window) is the
  default, `engine.prefill` selects it or the exact 40-layer parity mode.
  The tokenizer is a three-stage pre-tokenizer; the prompt renderer
  follows the checkpoint's own encoder; tool calls use a constrained DSML
  grammar. Served on four nodes 2026-09-14: 76 ms/pass at 2.33 tokens/pass
  (32.5 ms/token), bounded prefill 1.6–2.4 ms/token, gsm8k 60/60,
  HumanEval 40/40, extract 30/30; the release's own layer code in fp32
  sits as far from the engine as from its own bf16 pipeline at every
  cross-checked layer. Prefix caching is whole-block, so a prompt shorter
  than 128 tokens is not cached yet (docs/deepseek_v41_flash_plan.md §8).
- **The launcher's port preflight binds as the servers do** (2026-09-13):
  the probe now sets `SO_REUSEADDR` like `tcp.cpp` and `http_server.cpp`,
  so a TIME_WAIT left by the previous world is not a conflict and a
  restart within a minute of a stop passes preflight (the failure drill's
  reboot had failed on the fabric and journal ports). The drill fetches
  rank 0's op stream from the deployment's log dir, where the launcher
  runs rank 0, instead of the repository root.
- **A served request leaves no history** (2026-09-13; reported by a
  third-party tester as GitHub issue #1): the scheduler kept every
  request's prompt, grammar, cache cuts and generated ids — and a second
  copy of the tokens in `results()` — for the process's lifetime, and
  every tick scanned the whole history; the op stream, too, sat in memory
  until the exit-time write. Measured against the built scheduler: 8 bytes
  per prompt token per request (1,000 requests of 16,384 tokens: 131 MB of
  RSS; the 2026-09-07 agent session's 85 requests: 21 MiB) and 37 bytes
  per generated token, all inside the 4 GiB headroom. A retired request
  now releases everything but its tombstone (id, status, counts); the
  service and the peers (`Scheduler::set_keep_retired(false)`) compact
  the tombstones and results away at the end of every tick — the same
  quantum on every rank, since the journal carries every tick — so after
  any tick, under any load, the scheduler holds exactly its live
  requests; `/v1/metrics` reports the
  records and ids held (`scheduler.records`, `record_tokens`, `terminal`
  now cumulative); and each rank's `serve_rank<N>.ops` is written as the
  run records it, flushed at every retire, so a killed rank keeps its
  evidence. Gates: `scheduler_test`'s release and compaction tests (the
  compaction moves no op under overlapping load; the live heap flat at
  +5 KiB from request 200 to request 40,000), `fabric_serve_test`'s
  file-mode observer.
- **The full GLM-5.3's decode batch: sixteen rows** (2026-09-13, plan D9):
  the family's cap was the fused decode select's eight rows of shared
  memory; the select now launches its rows in groups of eight (a group
  is the same work at any grouping), the pick kernels verdict sixteen
  request slots, the DSA layer sizes its attention tiles and (row, split)
  workspace from the count, and the absorb/vout projections keep their
  warp kernels for every decode shape (`kProjMmaMinRows` 16 → 32: the
  tensor-core forms are tolerance-equal, not bitwise, and a batched row
  must be bitwise the same row alone). Eight request slots at MTP depth 1,
  five at depth 2, four at depth 3; every recipe up to eight rows runs
  the launches it always did. Gates: `dsa_test`'s row groups (12 and 16
  rows bitwise the host mirror), `glm_dsa_engine_test`'s sixteen-row
  world (eight slots at depth 1, batched transcripts bitwise the eager
  engine's). Found on the way: the GEMM interface lowers bf16 calls up to
  the model's decode rows to the GEMV chain, so a sixteen-row deployment
  prefills 9–16-token prompts through it where an eight-row one used
  cuBLASLt (last-bit differences, deterministic within a deployment).
  The graph engine records 4- and 6-slot batch families for recipes wider
  than four slots (four live requests on an eight-slot world had replayed
  the sixteen-row every-slot family at 250–270 ms a step).
- **The bus fold at two and three decode rows** (2026-09-13): the graph
  all-reduce stages a payload in shared memory up to 80 KB (was 48 KB), so
  the full GLM-5.3's two-row depth-1 fold (6144 x 2 x 3 peers = 72 KB) no
  longer reads the NIC-placed rows twice, and both passes over peer memory
  keep several system loads in flight per thread. Bitwise the old fold;
  depth 1 66.9 ms/pass (from 68.0), depth 2 85.1 (from 88.2), T=1
  unchanged. The L2 weight-prefetch knobs (`DGPP_L2_PREFETCH=off`,
  `DGPP_L2_PREFETCH_MB`, `..._BOUNDARY`, `..._LAYER`) are site settings
  the launcher forwards to every rank, for A/Bs; the A/B on the full
  GLM-5.3 kept the prefetch on (off costs 4 ms a step at T=1 and 5 ms a
  pass at depth 1).
- **A bare `dgpp-cluster down` stops what is running** (2026-09-12):
  `down` and `status` without `--config` (or with `--all`) scan every
  deployment recorded under `DGPP_LOG_DIR/deployments` instead of the
  default deployment file's namespace; `status` lists the running ones
  with their ranks and the stopped ones as one line each with how their
  log ended, `down` stops the running ones and names them by model, world,
  namespace and deployment file (the staged namespace now records it in
  `deployment.path`). Before, a bare `down` looked only at the default
  file's namespace and reported success after stopping nothing while
  another deployment kept the ports, so the next `up` failed preflight.
  `down --config FILE` on a deployment with no live rank now says so.
- **The full GLM-5.3** (2026-09-12, `docs/glm53_plan.md`): a fourth model
  family, `glm_moe_dsa` (`GlmMoeDsaForCausalLM`, 78 layers, 754B), served
  from the int4/int8 group-64 pack-quantized release
  (`HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64`) at 99.3 GiB of weights per
  rank on four nodes. New in the engine: a packed-int GEMV core with the
  exact code x scale dequant behind the MoE slot and grouped paths, the
  DSA layer's decoupled interleaved RoPE (the rope key stored bf16 beside
  the latent in every cache format, the flash kernels at a 576-wide
  score), per-token selection with the relu'd indexer at select_k 2048,
  cross-layer selection sharing (21 indexers over 78 layers), a loader on
  the shared resident stream with the draft layer's experts requantized
  at load, the session-core model, the serving family, three deployment
  templates and the tokenizer / chat-template goldens (the template
  interpreter's `range` gained the one-argument form). Gates: the DSA
  layer against its host oracle at the full geometry, the fixture forward
  against a pure-python reference, decode / prefix / speculator, TP
  worlds 2 and 4, the graph engine at world 2, and the real layers 0–3
  against transformers' own layer code (per layer 0.002–0.0035 relative
  l2, the bf16 floor; MoE routing near-ties certified by margin).
  `tools/checkpoint_audit.py` learned the pack-quantized triple and writes
  `docs/checkpoint_budget_glm53.md` (99.30 GiB per rank at world 4, the
  memory plan's number). Served on the four nodes the same day
  (`scripts/fabric_glm_dsa_serve.sh`, `serve_mtp_classes.py`): T=1 51
  ms/step, MTP depth 1 68–76 ms/pass at 1.77–1.97 tokens/pass, gsm8k 59/60
  and HumanEval 40/40 with thinking on, MTP == T=1 transcripts; the
  templates carry 112K bf16 (plain) / 96K bf16 (MTP) / 160K fp8 latent
  caches. The prefill tile kernel for the packed formats is deferred
  (the GEMV chain serves prefill at 7–9.5 ms per prompt token).
- **`engine.embed_sharding: vocab`** (2026-09-13): the full GLM-5.3 can hold
  its lm-head slice of the embedding rows on each rank instead of the
  whole table — 1.33 GiB per rank back at world 4. A token's row is gathered
  by the rank that holds it and summed across ranks by one fold (a row plus
  zeros is exact in bf16), on the main path and again for the draft's
  lookup: worlds 2 and 4, eager and recorded, are bitwise the replicated
  worlds. The key rides the rank-0 settings record and the config digest;
  the other families keep their tables whole. Default `replicated`; the
  three GLM-5.3 templates set `vocab`.
- **The memory plan's headroom is 4 GiB, from 8** (2026-09-12/13): the growth
  after the plan check was measured on the full GLM-5.3 at world 4 (all
  four ranks sampled through a boot, a 32K prefill and four live requests)
  at a flat 5.5–6.0 GiB, 2.2 GiB of it the loader's pinned staging mirror.
  The shared resident stream's `release_sources()` now frees that mirror
  with the checkpoint mappings; the GLM-5.3 model materializes its stack in
  its constructor and releases there, before its caches exist and before
  any collective is in flight (Qwen and GLM-4.7 load lazily and keep their
  mirror: a release mid-forward waits on the bus's persistent kernels for a
  watchdog period), and every resident plan names the staging as an item. The residual after the check is 2.4–2.8
  GiB; 4 GiB keeps 1.2–1.8 of margin, held by a one-hour soak at the
  full GLM-5.3's 120K MTP shape (2,046 requests, none failed, memory flat
  on all four ranks, no allocation stall or direct reclaim anywhere, rank 0
  never under 2.7 GiB available). With the vocab-sharded embedding the
  full GLM-5.3 templates carry 144K bf16 plain / 120K bf16 MTP / 208K fp8.
  `serve_soak_run.sh` now honors a preset `DGPP_SERVE_KNOBS` (a soak at a
  deployment's real shape) and counts stalled collectives only inside the
  run (the peers' logs accumulate across boots of one deployment);
  `serve_run.sh` passes the selected deployment as `--config`, without
  which the launcher refused to stop a `--log-dir` world and the soak's
  teardown had been leaving its world serving.
- **`cluster_glm-5.3_int4-int8_w4_mtp2`** (2026-09-13): MTP depth 2 for the
  full GLM-5.3 at two request slots (the eight-row decode bound). Measured:
  86–89 ms/pass at 2.15–2.69 tokens/pass, a few percent per token either
  way by class (code and JSON gain, chat loses); depth 1 stays the default.
- **The memory ledger** (2026-09-13): one INFO line per boot phase (after
  the plan check, the bus, the loader, the globals, the resident stack, the
  caches, the model, the graph engine, the warm capture, at listening)
  with the process's resident split and the device's free memory, and one
  per graph variant with its executable's device memory. What it found on
  the full GLM-5.3: the executables are 0.39 GiB for fourteen variants
  (7–15 KB per node), the process grows 0.7 GiB on the host, and the rest
  of the 2.7 GiB the plan does not name — about 2 GiB — is CUDA's own
  (cuBLASLt's kernels and the context's growth at first launches), not a
  plan item that can shrink. The headroom stays 4 GiB.
- Setup now covers CUDA discovery outside `PATH`, head/peer dependencies,
  outbound connectivity, verified SSH login, offline preparation and storage
  budgeting. Empty explicit `--config` arguments are rejected.
- Doctor distinguishes failed probes from zero RoCE lanes and points checkpoint
  recovery to rank 0. RoCE discovery retains partial results and displays the
  selected lane order. Its `--json` output now contains `inventories`, `selections`
  and `errors` objects keyed by host, replacing the bare host-to-inventory map.

- **GLM-5.3-Flash on two nodes** (2026-09-12): the NVFP4/FP8 hybrid fits a
  two-rank world at 94.71 GiB of weights per rank, against 50.74 at world 4.
  `deploy/cluster_glm-5.3-flash_nvfp4-fp8_w2_mtp1.example.json` serves it with
  four request slots, MTP depth 1 and an FP8 latent cache at a 163,840-token
  context; the `_large-cache` variant trades two of those slots for 262,144
  tokens, because the draft block's per-position hidden cache costs 8 KiB per
  token per slot and is 83 % of what a context token costs. No engine change
  was needed: the TP slicing, the loader and the bus were already world-generic
  and the geometry divides by two. Measured (benchmarks.md §3 to §7): T=1
  45.68 ms/pass, greedy MTP 59.40 ms/pass at 1.734 tokens per pass for
  34.25 ms/token, per-class acceptance matching the four-node run, aggregate
  22.7 to 47.3 tokens/s across the class and concurrency matrix, procedure
  prefill 664 / 1,829 / 8,280 ms at 512 / 2,048 / 8,192, and HumanEval 158/164,
  GSM8K 291/300, schema extraction 100/100 on the four-node denominators. Decode
  costs 1.75x the four-node pace, prefill about 1.45x, quality nothing. Both
  ranks' op streams matched at shutdown and the MTP and T=1 runs produced the
  same rank-consistency digest.
- `serve_soak_run.sh`, `serve_stop_check.sh` and `serve_failure_drill.sh` take
  their rank count from the selected deployment instead of assuming four nodes,
  and refuse a single-node deployment by name (`dgpp_require_peers`). The
  drill's victim argument now accepts any rank of the configured world.
- Deployment filenames now identify the full model, quant, world size and
  decode mode. Settings are unchanged; see `deploy/README.md` for the rename
  table. Older entries below retain the filenames used at the time.

- **Qwen3.8-Flash-Next on a single Spark, and its dense stack in block FP8**
  (2026-09-10, docs/qwen38_single_spark.md): the NVFP4 checkpoint's experts
  through the shared fp4 core, the 47.7 GiB n-gram table left on the NVMe
  behind a memory mapping (`engine.ngram_table`, default resident) with each
  walk's rows gathered by a host node forked inside the walk, and the resident
  graph engine at world 1 over a world-of-one bus — 47 ms/token plain,
  31–38 ms/token with MTP (59 ms per pass), prefill 1.0–1.5 ms/token. Then the
  dense projections encoded as they load with the FP8 releases' own recipe
  (`engine.dense_weights = "fp8"`, default `"checkpoint"`; amax / 448 per
  128 x 128 block, round-to-nearest-even e4m3), read by the fp8 GEMV core at
  decode rows and dequantized into a BF16 bridge for the GEMM interface: T=1
  47 -> 31 ms per pass, MTP 59 -> 40 ms per pass (21–26 ms/token, 38–48 t/s),
  four in flight 122 -> 71 ms per step. Then the fused decode forms and a
  multi-problem fp8 GEMV — the GR site's norm-staged down GEMV with the inject
  rows riding it, the act-staged up GEMV, the batched down + inject, the shared
  expert's two-launch tail, and the GDN qkv + z / QSA q, k, v, indexer
  projections as one launch — each the BF16 fused kernel templated on the rows'
  form and bitwise the unfused fp8 chain: T=1 30.2–30.9 ms per pass, MTP
  39–40 ms per pass. Eval on the FP8 world: GSM8K 59/60, HumanEval 38–39/40,
  schema extraction 30/30 (the fabric's numbers); MTP transcripts identical to
  T=1. The world-1 profile prices every class against its byte budget (the
  experts and the multi-problem GEMVs at line rate, the draft head's second
  lm_head 2.5 ms a pass, the GR down GEMV at 55 %, ~2 ms of gaps), and the L2
  prefetcher stays on at world 1 (off costs 2 ms a T=1 pass). MTP depth 2 in
  this world: 48 ms a pass at 1.77–2.74 tokens, code/math/JSON 10–15 % faster
  per token, prose 5 % slower — depth 1 stays the default.

- **The MTP draft is drawn from the draft head and verified with the ratio
  rule, at any depth** (2026-09-10): the draft pick is a sampling pick on its
  own stream (`seed ^ kSampleDraftSeedMix`, the request's counter — the
  request's own draw accounting untouched), its final set travels as
  `DraftProposal` to the next step's verify, whose row 0 accepts with
  min(1, P/Q) and resamples the (P - Q)+ residual, so the emitted marginal is
  exactly P as before (`sampler_test`: marginal == P, acceptance ==
  1 - TV(P, Q); `glm_pick_test`: the device decision, token and draw count
  bitwise the host oracle's). A re-drafted row, an over-wide set or an
  unmatched token falls back to the plain rule, exact for any draft;
  `DGPP_SPEC_PROPOSAL=off` restores the argmax draft. Fabric world 4 at the
  sampled default: acceptance 41–51 % -> 60–68 %, 1.41–1.51 -> 1.60–1.69
  tokens per pass, 17.0–18.3 -> 15.3–16.1 ms/token at the same 25.8 ms/pass;
  greedy transcripts and the concurrency curve unchanged. The depth >= 2 chain
  picks are proposals too (each draft index its own key, so one step's drafts
  never share a draw; verify tests row t against proposal slot t; the host's
  re-drafts clear every slot): p2 30–34 -> 36–37 % at world 4 and 24–39 ->
  31–36 % at world 2.

- **Qwen3.8-Flash-Next takes the draft chain** (2026-09-10): `kDraftChain` and
  `kBatchedDraftChain` for the family the depth-2 entry below had excluded —
  the chain rows run the draft block forward past the first draft, and its QSA
  ring is copied aside before the first chain row and restored after the last,
  into a buffer of its own so the draft snapshot the fallback's rollback
  restores keeps the pre-draft ring. `qwen_engine_test` gates depth 2 on the
  loopback fixture: the greedy transcripts are the plain engine's, scalar and
  batched, on both ranks. Measured break-even on the fabric, so depth 1 stays
  the default — world 4 30.8 ms/pass at 1.93–2.03 tokens = 15.2–15.9 ms/token
  against depth 1's 25.9 ms/pass at 1.63–1.64 = 15.8–15.9; world 2 49.4 ms/pass
  at 1.81–2.08 against 40.4–41.0 at 1.63–1.73. The engine test's fixture
  directory joins the ignore list.

- **The GR inject dots leave the decode chain, and four levers measured and
  left behind knobs** (2026-09-10, docs/qwen38_optimization_plan.md): the
  gates depend on Rn alone, so at one row the hc inject rows ride the mix's
  down GEMV as the row space's last rows over the staged Rn (the same
  lane-strided chain as `combine_dots_kernel`, bitwise; `qwen_gr_test` pins the
  folded gates against the standalone kernel), and at more rows the same kernel
  runs on a low-priority side stream forked at the mix and joined before the
  apply — what was a 21 us one-block kernel between the boundary collective and
  the apply, 96 sites per step. Fabric world 4: T=1 22.12 -> 21.4–21.5 ms/step,
  MTP 26.5 -> 25.7 ms/pass, four live requests 114.6 -> 120.0 tok/s; world 2
  after both decode changes: T=1 31.38 ms/step, MTP 40.4 ms/pass at 23.7–25.2
  ms/token. Kept and off by default, each with its reading: the draft's
  temperature as a scale of the request's (`DGPP_SPEC_PROPOSAL_TEMP` — exact at
  any value, acceptance flat from 0.7 to 1.2, stays at 1); the small-k form of
  the fp8 expert tile kernel for the down projection (`DGPP_MOE_FP8_SMALLK`);
  the grouped fp8 expert kernel's m-sweep (`DGPP_MOE_FP8_SWEEP` — 5.87 vs
  6.87 ms per gate launch at 228 rows per expert, but 3.6 % on an 8K prefill
  against the byte model's 14 %: at four m-tiles it is issue-bound at 47
  TFLOP/s, so the chunk stays 2,048); and the prefetch-instruction form of the
  L2 prefetcher (`DGPP_L2_PREFETCH_FORM=prefetch` — 22.4–23.7 ms per T=1 step
  against the load form's 21.5, whose fold is what bounds the bytes in flight
  to what the collectives tolerate).

- **GLM-4.7 (`nvidia/GLM-4.7-NVFP4`) served** (2026-09-10, docs/glm47_plan.md):
  the Glm4MoeForCausalLM family — 92 pre-norm layers of biased GQA
  attention (96/8 heads, per-head q/k norms, half-split partial RoPE),
  three dense NVFP4 MLP layers then 160-expert sigmoid-routed MoE layers
  with an NVFP4 shared expert, an MTP draft layer whose BF16 experts are
  requantized to NVFP4 at load — on the same session core, engines,
  scheduler and service as GLM-5.3 and Qwen3.8-Flash-Next. New: the
  modelopt NVFP4 binding and loader (`src/models/glm4/`), the fp4 GEMV
  core generalized to any K multiple of 32 (power-of-two shapes bitwise
  unchanged), the NVFP4 shared expert as view-table entry E of the routed
  launches, paged split-KV GQA attention kernels with the fused bias /
  norm / RoPE / K-V append (`src/kernels/glm4_attn.cu`, bitwise a
  tile-aware host reference), `Glm4Model` with the paged K/V pool (no
  recurrent state: rollback positional, snapshots at any position, the
  draft's hidden window), the GLM-4.7 tokenizer (a Sequence[ByteLevel]
  post-processor), the template's `rstrip`/`lstrip` with a character
  argument, `deploy/cluster_glm47.json`, `scripts/fabric_glm4_serve.sh`;
  The partial RoPE is transformers' half-split rotate_half (pairs i, i+32
  over the first 64 dims), not the interleaved form of the older glm/glm4
  architectures: the interleaved reading passed every fixture gate (the
  python reference shared it) and left the real model coherent for ~30
  tokens before looping; found against transformers' own layer code on
  the real weights (`tools/glm4_torch_reference.py`, layer-0 relative l2
  0.036 -> 0.0025 after the fix). The bus recorder's collective-node budget per graph 128 -> 256
  (`kBusMaxGraphGens`: GLM-4.7's decode step records 186 — its first
  fabric boot refused the second graph variant at 128). The draft block
  takes the model's POST-final-norm hidden (vLLM's `glm4_moe_mtp`
  convention; the pre-norm residual gave 11–46 % draft acceptance on the
  fabric, the post-norm hidden 79–92 %).

- **The decode rows are the recipe's shape, and the batched depth-2 chain**
  (2026-09-10): the fixed decode batch's row ceiling is derived at boot —
  `max_concurrency x (1 + mtp_depth)`, floored at 8 so every existing recipe
  keeps its exact shape — and carried at runtime by the session core
  (`SessionParams::decode_rows`: the per-row scratch, the token feeds, the
  draft windows, the memory plan), the picker's tables, the bus's latency
  slot and the graph engine's batch families; the build-time bound is
  `kDecodeRowsMax` = 32 (the pick kernels' fixed arrays). GLM-4.7 takes the
  derived shape (`serve: decode rows 12 (4 slots x 3 rows ...)` in the boot
  log); GLM-5.3-Flash keeps its fixed 8, Qwen3.8-Flash-Next stays capped at
  8 until its kernels are gated past it. On the runtime rows the row batch
  carries depth >= 2: `session_graph_capture_draft_chain_batch` +
  `glm_spec_chain_rows_batched` run every slot's chain row in one draft-
  block run (the batched next-token feed carries every draft per request);
  a family opts in with `kBatchedDraftChain` (GLM-4.7). The GEMM interface
  lowers bf16 decode calls up to the model's decode rows to the row-
  independent GEMV core (`CublasLtGemm::set_decode_rows`; the first 9-row
  batch fell to an Lt algorithm with its own reduction order and flipped a
  near tie), so a batched request's rows keep the scalar order at any
  width — the depth-2 engine gate runs A scalar and B, C batched on a 9-row
  ceiling, bitwise the eager engine's. Also fixed: one slot's sampled MTP
  fallback dropped every slot's draft picks of a batched replay (the mirror
  went stale for the others). Measured on the fabric (`fabric_glm4_load.sh`):
  batched depth 2 loses to depth 1 at 2 and 4 live requests (37 vs 44.5,
  43 vs 47.5 tok/s) — every 4-row GEMV chunk past the first re-reads the
  6.3 GB of BF16 attention projections per rank (+45–60 ms per chunk: 4
  rows 79 ms, 6 rows 123, 8 rows 140, 12 rows 200); depth 2 remains a
  single-stream gain (+4–13 %). The lever for concurrency at either depth is
  the attention projections' bytes per pass: wider GEMV chunks or a row-
  independent tensor-core kernel, docs/measurements.md.

- **MTP depth 2 on the hidden-window families** (2026-09-10): the session
  core carries the GLM-5.3-Flash chain — one draft-block row per further
  draft at counter + index, fed the previous draft's pick and the block's
  own output row as its hidden, landed in the slot's window
  (`glm_spec_chain_row_window`); `session_draft_chain`,
  `session_graph_capture_draft_chain`, the multi-draft feed; a family opts
  in with `kDraftChain` (GLM-4.7 immediately; Qwen3.8-Flash-Next followed
  the same day, once its draft ring learned the chain snapshot — above).
  `deploy/cluster_glm47_d2.json`.
  GLM-4.7 at depth 2: 72 ms/pass at 2.3–2.6 tokens/pass (p2 48–65 %), 4–13 %
  more tokens/s single-stream than depth 1, +5 % at a 6.5K context;
  transcripts identical. Gate: `glm4_engine_test`'s depth-2 world (port
  29951); the row batch past depth 1 followed the same day (above).

- **`chat_template_kwargs.enable_thinking` is a knob of the templates that
  read it** (2026-09-10): the service asks the loaded template
  (`ChatTemplate::reads`, `ModelFrontend::template_reads`) and passes the
  boolean through for Qwen3.8-Flash-Next and GLM-4.7 (false closes the
  think block in the generation prompt — GLM-4.7's non-thinking mode; it
  ignores reasoning_effort and otherwise thinks to the token cap); GLM-5.3-
  Flash's template never reads it and the refusal stands. `serve_eval.py
  --no-think`; `serve_api_check.py` builds its stop material with thinking
  off where the template allows it.

- **Scheduler: a step cut short takes no retire-time prefix snapshot**
  (2026-09-10). When a multi-token MTP pass's earlier token completes the
  answer (the cap or EOS), the pass's remaining tokens are dropped and the
  model's state sits past the committed position; the retire path asked
  the arena for a live snapshot at the committed position and the arena
  refused (an engine failure on GLM-4.7's first fabric request — every
  position is snapshot-eligible at its align of 1; the aligned families
  hit it only when the cut lands on an aligned position). `Request::
  step_tail` records the dropped tail; the rolling entry stands as the
  close entry. `scheduler_prefixCache_aStepCutShortByTheCapTakesNoRetireSnapshot`.
  Gates: the python double reference (`tools/glm4_reference_dump.py`,
  final hidden l2 0.0035, top-1 and routing exact, the draft's rows),
  transformers' own layer code on the real weights (layer-0 relative l2
  0.0025), decode/sessions/speculator (60 steps across the pool's block
  boundary, per-step rolling snapshots), TP worlds 2 and 4, the graph
  engines at world 2 (scalar, batched and MTP across the boundary),
  tokenizer and chat-template goldens (HF tokenizers 0.23.1, jinja2
  3.1.2). Fabric (four nodes): boot 18–20 s from the resident image, T=1
  49.0 ms/step, MTP 60–61 ms/pass at 1.86–1.99 tokens/pass (31–33
  ms/token), MTP == T=1 transcripts, eval with thinking off gsm8k 60/60,
  HumanEval 39/40, extraction 30/30 (docs/measurements.md).

- **Shared cores extracted for the second and third families** (2026-09-09):
  `loaders/resident_stream.hpp` (the resident/streaming layer stream —
  bumps, staging, the image, byte reconciles, the digest — under a
  family's builders; Qwen and GLM-4.7 on it), `engine/paged_blocks.hpp`
  (the paged block table with its sharing and pinning protocol under both
  K/V pools), `engine/session_model.hpp` (the session core: positions,
  feeds, chunking, snapshots, the graph era, the draft plumbing, under
  `QwenModel` and `Glm4Model`). Every existing gate unchanged.

- The fp8 tile kernel ported to the ldmatrix form (32-deep stages, the
  block scale copied with the tile, fragments decoded e4m3 -> f32 x scale
  -> bf16 as the reference rounds them): bitwise the reference; the FP8
  checkpoint's steady-state prefill 740 / 1,725 / 7,463 -> 549 / 1,412 /
  6,280 ms at 512 / 2,048 / 8,192 tokens (with the mHC GEMM form), the
  hybrid's shared experts 440 / 1,290 / 5,769 -> 425 / 1,275 / 5,702.

- The prefill's mHC dots on tensor cores: a bf16 mma.sync GEMM over a
  cp.async ring with per-stage fp32 partials, the token's sum of squares
  gathered from the same tiles, then the standard finish; the sites no
  longer store the unread collapsed row. Within the oracle budgets (not
  bitwise the per-coefficient decode form; the tiled kernel stays behind
  `mhc_set_prefill_gemm(false)`): 1,196 -> 836 us per site at 2,048
  tokens; steady-state prefill 460 / 1,335 / 5,954 -> 440 / 1,290 / 5,769
  ms at 512 / 2,048 / 8,192; the 64-step transcript identical, one
  near-tie flip (0.049 logits) in a 512-token continuation.

- The fp4 prefill kernel, second pass: the ldmatrix tile kernel (64 x 128
  x 64, a three-slot cp.async ring of the activation tile and the raw fp4
  codes, B fragments decoded at fragment time, each weight row's next L2
  line prefetched ahead, the activation lines tagged evict-last, one m-tile
  per block with the m-tile the fastest grid index) is the production
  launcher on both expert shapes — bitwise the reference; steady-state
  prefill of the hybrid 563 / 1,502 / 6,572 -> 460 / 1,335 / 5,954 ms at
  512 / 2,048 / 8,192 tokens, generated ids unchanged. glm_moe_test's odd
  geometries caught misaligned copies at k = 208 (aligned-width fallbacks
  added). The FP8 path is untouched.

- The mHC site off the critical path: the fused finish leaves comb (the
  20-iteration Sinkhorn, 4.6 of its 8.6 us) to `launch_mhc_comb` on a
  low-priority side stream forked after the finish and joined before the
  stream update — the only reader — and reuses the dots block's register
  copy of the streams for the collapse; bitwise (glm_mhc_test pins the
  deferred form at 1/2/8 rows). The fused kernel 13.0 -> 6.5 us per site
  in the MTP graph; the hybrid's decode MTP 34.21 -> 33.89 ms/step (19.96
  -> 19.77 ms/token), T=1 26.46 -> 26.22. `mhc_site_bench` (new) times a
  site with the finish, Sinkhorn and norm separable. The router's register
  select and the collective's local phases were measured and parked
  (`docs/nvfp4_plan.md` §6a). DGPP_MHC_COMB_SIDE=0 keeps comb in the
  finish.

- The fp4 GEMV core decodes with GB10's own e2m1x2 -> f16x2 conversion and
  one exact f16 multiply by the group scale (bitwise the register decode;
  the build moves to the architecture-specific target `121a`), the
  activation window read once per chunk column; the decode slot path
  204 -> 195 us per MoE layer at one row, MTP 20.04 -> 19.96 ms/token on
  the hybrid. Hardware counters (Nsight Compute as root) and a no-arithmetic
  probe of the same access pattern put both fp4 slot kernels at 90-93 % of
  their streaming floor; a persistent grid-stride form with next-tile
  lookahead, the paired gate/up issue, more row steps, four blocks per SM,
  the router's unroll and block width, and programmatic dependent launch
  across the graph's interfaces were each measured and parked
  (`docs/nvfp4_plan.md` §6a).

- NVFP4 routed experts, phase 1 of `docs/nvfp4_plan.md` — the format is
  loadable. The composed checkpoint `dgpp/GLM-5.3-Flash-NVFP4-FP8`
  (`tools/compose_nvfp4_hybrid.py`: dabsLabs' NVFP4 experts for layers
  3–44 beside the FP8 release's bytes for every other tensor, the MTP
  layer included; 195.1 GB, built and verified byte-for-byte on all four
  nodes, identical shard hashes) declares `quantization_config.quant_method
  = "dgpp_mixed"`, which `GlmTextConfig` parses into a routed-expert format
  (`GlmExpertFormat`: FP8 block-128 as before, or NVFP4 group-16). Under the
  NVFP4 profile the expected-tensor table emits the triple `weight_packed`
  (U8 [N, K/2]), `weight_scale` (e4m3 [N, K/16]) and `weight_global_scale`
  (F32 [1]) for every main-stack routed expert and the validator binds it
  as a unit (every tensor now carries a `GlmTensorRole`, so an e4m3 block
  scale is never mistaken for an FP8 payload); the loader keeps the bytes
  untouched in `GlmFp4Matrix` views — this rank's inter slice as
  contiguous gate/up rows and nibble-granular down column packs, every
  matrix's global scale gathered into one per-layer array — with the same
  counting-mode byte formula, resident image and byte reconcile;
  `GlmTpViews` carves the same views from a full layer so the shard-parity
  gate pins the two paths bitwise on an NVFP4 fixture at worlds 2 and 4,
  streaming and resident. The MTP layer's experts, the shared experts, the
  dense MLPs and the attention keep the FP8 paths untouched under both
  profiles; the FP8 checkpoint's table and bytes are unchanged (the
  existing gates pin that). `glm_bind_check` on the hybrid: 112,049
  expected, 112,049 matched, 36,288 NVFP4 triples and 1,050 FP8 pairs
  bound; the resident formula at world 4 is 50.7 GiB per rank against
  ~82 GiB for the FP8 checkpoint.

- NVFP4 routed experts, phase 2 — the decode path runs on the packed
  bytes. `src/kernels/fp4_gemv.cuh` is the in-register e2m1 core: a lane
  loads 16 bytes (32 codes, two group-16 scales), places each code's bits
  into an fp16 field (`((code & 7) << 9) | ((code & 8) << 12)`, which reads
  as value x 2^-14 for every code) and multiplies by the scale x 2^14, all
  exact — an e2m1 times an e4m3 has at most six significant bits, so the
  weight enters the fp32 dot unrounded and the only inexact operation is
  the one division by the tensor's global scale in the epilogue
  (`tests/unit/fp4_test.cpp` pins the lemma over every code x scale pair
  and the bit-placement against the codec table). The core is templated
  on K so every index is compile-time (the first draft's lane-dependent
  accumulator index spilled to local memory and ran at 14 GB/s); it
  backs the single-matrix launcher (`launch_fp4_gemv_bf16/f32`), the fp4
  decode slot kernels (gate_up+swiglu and down, the shared expert running
  the fp8 core inside the same launch as before) and the fp4 grouped
  GEMV that `GlmMoeLayer` dispatches under the NVFP4 format; the host
  oracle dequantizes the triple exactly and applies the same divisor
  epilogue. Gates: `fp4_gemv_test` (the oracle across every supported
  geometry at l2_rel 0, row-count invariance, the f32 epilogue, NaN
  scales, the geometry contract, and two real slices of the hybrid at 0
  mismatches), `glm_moe_test`'s fp4 twins bitwise — the decode slot path,
  the host grouped path and the sliced-rank fold agree bit for bit as the
  FP8 ones do. `moe_slot_bench --format fp4` per layer on one rank's
  slice: rows=1 209 us against fp8's 273, rows=2 334 against 469, rows=8
  929 against 1,420. Prefill under NVFP4 uses the grouped GEMV chain
  (the tensor-core kernel is phase 3). The FP8 paths are untouched.

- NVFP4 routed experts, phase 3 — the prefill runs on tensor cores.
  `moe_grouped_mma_fp4_kernel` is the fp8 tile kernel's structure (128 x
  64 x 64 stages, the same activation tile and ascending-k16 bf16
  `mma.sync` chain) with the weight tile decoded from the NVFP4 triple:
  a thread's 16 consecutive k come from one 8-byte load of codes and one
  scale byte, decoded as e2m1 x scale, exact in bf16, so unlike the fp8
  tile nothing is rounded before the MMA; the global scale divides the
  finished dot once in the epilogue. Any k that is a multiple of 16 (the
  GEMV core's power-of-two set does not apply). The grouped form
  (segments through the row map) is bitwise the dense form per segment;
  `GlmMoeLayer` dispatches it for NVFP4 routed segments under the
  tensor-core kernel while the FP8 shared expert keeps the fp8 tile
  kernel; the fp8 kernel's source is untouched. Gates: grouped bitwise
  the dense form at I = 208 (a ragged n-tile and a ragged last k-stage)
  and 320, under the z split and through a permuted row map, gate bf16
  and down fp32; the whole layer within budget of the double oracle on
  four geometries beside the GEMV chain; both real checkpoint slices at 0
  mismatches with a row's bits independent of m. Fabric, steady-state
  prefill on the hybrid: 512 tokens 1,152 -> 635 ms (FP8 740), 2,048
  tokens 4,396 -> 1,563 (FP8 1,725), 8,192 tokens 18,246 -> 6,796 (FP8
  7,463); the 64-step transcript identical to the GEMV-chain build's, all
  ranks identical; the FP8 build's 2,048-token prefill unchanged (1,717
  ms). nsys says the tile kernels — fp8 and fp4 alike — sit at a third of
  the read floor (a 340 MB fp4 launch takes 4.6 ms): the activation tile
  is re-read once per 64-wide n-tile, the stages do not overlap, and a
  128-row m-tile is more than half padding at ~57 rows per expert; the
  restructure is `docs/nvfp4_plan.md` §6a (phase 5).

- `scripts/serve_eval.py`, the task-level eval through the served endpoint
  (HumanEval, GSM8K, schema extraction; greedy), run on both configs: the
  NVFP4 hybrid 157/164, 293/300, 100/100 against FP8's 155/164, 293/300,
  100/100 — the same items pass on both to within two HumanEval problems
  and three GSM8K flips each way, while the replies are word-identical on
  only two thirds of the items. `docs/nvfp4_plan.md` §6a.

- The DSA attention projections consumed as FP8 directly (the "bridge"
  item of `docs/nvfp4_plan.md`): q_a, kv_a, q_b and o_proj — FP8 pairs in
  every checkpoint — were dequantized to BF16 at load (the M3 interface) and
  read at twice their bytes on every step. The loader now keeps them as
  the checkpoint's pairs whenever this rank's q_b row slice and o_proj
  column slice start on the 128-wide scale grid (every power-of-two world
  of the production geometry; the bf16 bridge remains for misaligned
  slices — the test fixtures at world 4), `DsaLayerWeights` carries either
  form, and the layer's three projections run through the scale-aware
  GEMM (which gained an output row stride so the fused [q_a | kv_a]
  output takes two pairs into one buffer). `GlmTpViews` slices the pairs
  at aligned worlds and dequantizes the full pairs itself (the loader's
  kernel and rounding) for the bridge form at misaligned ones, so the
  shard-parity gate pins both forms bitwise against the sharded loader.
  The resident image format version is 2 (bridge-layout images are not
  restored). Gates: `dsa_test`'s new prefill case runs the same weights
  in both forms against the host reference; `glm_loader_test` checks the
  pairs byte-exact; shard parity at worlds 2 (direct) and 4 (bridge).
  Applies to the FP8 model and the hybrid alike: 0.35 GB per token per
  rank fewer over the 11 DSA layers.

- MTP depth 2 re-measured on the NVFP4 hybrid through the serve path: 4-6 %
  faster per token on 300-token prose, code and JSON answers at short
  context (2.5 tokens per step, the second draft standing 59-61 %), 6.6 %
  slower at a 7,368-token context (the second draft standing 41 %). Depth
  1 stays the default; the numbers are in `docs/nvfp4_plan.md` §6a.

- NVFP4 routed experts, phase 5, the decode core: one row step per warp in
  the fp4 GEMV core instead of two (`fp4_gemv::kSteps`) — twice the blocks
  per launch, four loads in flight per lane — after a sweep of the core's
  tunables on `moe_slot_bench` in which every "more per lane" direction
  lost; 5-6.5 % faster per layer at 1, 2 and 8 rows in an alternated A/B,
  outputs bitwise (a row's chain does not depend on the step count).
  Fabric, the hybrid: T=1 28.0 -> 27.0 ms/step, MTP 36.6 -> 35.4 ms/step
  (21.2 -> 20.5 ms/token), acceptance unchanged.

- NVFP4 routed experts, phase 5, the prefill tile kernel: a warp whose 16
  rows lie entirely past a segment's end skips its MMAs (its rows are
  never stored, so the outputs are bitwise), and `GlmMoeLayer` dispatches
  the fp4 tile kernels by shape — the synchronous 128 x 64 x 64 tile for
  gate/up, a pipelined 64 x 128 x 32 kernel (cp.async double-buffered
  activation stages, the next stage's codes decoded after the MMA) for
  the down projection, each the faster one on its shape. `moe_tile_bench`
  times every variant per launch at the production expert shape with
  uniform or ragged segments and checks the pipelined kernel bitwise
  against the reference; a decomposition with the loads stubbed out shows
  the gate launch is 58 % padded MMA + decode + barriers and the rest
  un-overlapped operand loads (`docs/nvfp4_plan.md` §6a records the
  numbers and what remains). Fabric, steady-state prefill on the hybrid:
  512 tokens 563 ms, 2,048 tokens 1,502, 8,192 tokens 6,572 (phase 3:
  635 / 1,563 / 6,796); transcript identical, all ranks identical.

- NVFP4 routed experts, phase 4 — the hybrid serves. A site keeps one
  cluster config per checkpoint and names it on the command line
  (`deploy/cluster.*.json` is git-ignored beside `deploy/cluster.json`;
  `docs/operations.md`): `deploy/cluster.nvfp4.json` selects
  `dgpp/GLM-5.3-Flash-NVFP4-FP8` with `kv_capacity` 786,432 and
  `prefix_cache_gib` 8, the 31 GiB per rank the experts free spent on
  context and cache (`--memory-plan`: 96.2 GiB + 8 headroom of 116.5).
  `scripts/dgpp-cluster up --config deploy/cluster.nvfp4.json` boots from
  the resident images in 18 s; `serve_api_check.py` passes every case;
  the world tears down with four identical op-stream md5s and no stall.
  The L2 prefetch defaults stand on the hybrid: the window is flat from 8
  to 24 MB at T=1 (28.0 ms/step) and under 1 % apart under MTP; the
  prefetcher itself is worth 2.0-2.3 ms/step, the full boundary rate
  costs 2 ms, as on FP8. `scripts/serve_bench.py` and `serve_soak.py`
  address the model the service reports (`GET /v1/models`) instead of a
  hard-coded FP8 id, with an optional argument to override.

- `glm_gen_check --mtp --decode-graph` accepted no draft since 2026-09-06
  (every class 0.0 %, the transcript still exact): the decode-step commit
  moved the scalar graph's fed tokens to the slot's persistent feed rows
  (`device_feed`, rows 8 onward) and the serving adapter followed, but the
  evidence app's recorded verify kept comparing row 0's winner with the
  eager rows at `device_tokens()`, which nothing rewrites during a replay.
  The app's recorded verify now reads the feed rows; the eager speculator
  (77 % accepted) and the serving path were never affected. Found by the
  NVFP4 evidence chain; bisected with the app built at 2b7a6c6 and
  f8b2d34 (both 0 %) against the eager path (77 %).

- The row batch is a family: a 2-slot (4-row) and a 3-slot (6-row) batch
  are recorded beside the full 8-row one, and a step replays the smallest
  whose slots cover the live requests. Two live requests used to be either
  two scalar replays (83 ms/step) or the 8-row batch with four padding
  rows (96.5 ms/step, the reason the crossover sat at four); the default
  crossover is now two. A single live request replays its scalar graph as
  before. Gated bitwise against independent scalar sessions and
  speculators through every occupancy and the hole cases, plain and MTP,
  on the loopback world.

- A prefix cache miss is explained on its own INFO line: the cuts probed,
  the entries held, and where the prompt parts from the entry it shares
  the most with (`PrefixCache::nearest`), so a client that edits the
  system prompt or compacts its history between turns is named on the
  spot; when the conversation's own entry was pushed out of the arena the
  line says that instead, from a ring of the last 256 evicted entries
  (`PrefixCache::ghost_at`). `scripts/serve_agentic_streams.py` reproduces an agent client's
  pattern — several concurrent multi-turn tool loops with the reasoning
  stripped from the history, side requests between turns, an optional
  system-prompt edit — and reads the cache's answer per turn from
  `usage.prompt_tokens_details.cached_tokens`; the scheduler suite pins
  two interleaved conversations hitting on every turn under a six-slot
  arena, one returning its reasoning (attaching at the close entry) and
  one stripping it (attaching at the previous prompt's header cut).

- An integer's `minimum` / `maximum` / `exclusiveMinimum` /
  `exclusiveMaximum` are enforced by constrained decoding, for a tool
  argument and for a `response_format` schema alike: the bounds compile
  to one inclusive 64-bit range (a fractional bound rounded inward, an
  exclusive one stepped by one), the JSON machine admits a digit only
  while some completion can still land inside the range and the value's
  closer only once the digits do, and the mask re-judges the pure-numeric
  tokens by the same arithmetic, proven equal to brute force by the walk
  gate. A `timeout` with `minimum: 1` can no longer come out as `0` or
  `-5`, a `limit` with `maximum: 2000` cannot exceed it, and the tool
  behind them never sees a value its schema excluded (the Hermes agent's
  definitions carry such bounds on every integer argument; before this
  the server logged one INFO line per bound saying it was not applied).
  A number that admits a fraction keeps its bound unenforced — refused
  under `strict`, noted with the reason for a non-strict tool — as does
  the draft-4 boolean form, a bound beyond int64, an empty range, and an
  enum no member of which fits; an enum beside a bound is filtered to the
  members inside it. The INFO line for the keywords that stay unenforced
  now carries the reason. Also fixed on the way: beside a bounded (or
  plain) integer alternative, an enum of integers let the mask admit a
  `.` every cursor then refused; and a top-level number under an `anyOf`
  is complete where some alternative accepts it, not only where all do.

- Fixed: the decode index selection's histogram sat at a call-dependent
  workspace offset (the call's rows of keys), so a ONE-row call — the
  sampled fallback's eager verify or re-draft — read its histogram out of
  row 1's keys, which every two-row replay writes: a garbage histogram, a
  partial best-list fill, stale shared-memory pool ids expanded into token
  ids, and the listed attention reading an unmapped page (an engine
  failure once in roughly ten fallbacks past 2,048 tokens of context; the
  first report was a live service at 22:13 on 2026-09-06). The histograms
  now sit at a fixed offset for the maximum rows and are zeroed every
  call; the best list starts empty and the expansion drops any pool past
  the row's visible pools; the listed gather checks the token and its
  block and zero-fills instead of faulting; both record their first
  anomaly, logged at the slot's close (`dsa_select_anomalies`,
  `dsa_attn_anomalies`). `DGPP_SYNC_EAGER=1` syncs an eager row after each
  stage and validates the selection list before the attention — the knob
  that found it. Verified by an 11-turn long-decode sampled soak (about 150
  fallbacks past 2K tokens of context) with no anomaly, where the
  unfixed select logged short fills on every request, and by the DSA
  suite.

- The pipelined replay: the engine's step returns at the verify's verdict
  (a kernel node publishes the slot's replay sequence to pinned memory)
  and the next replay is launched before the previous one's draft tail
  and window are settled, so the verdict read, the scheduler's
  bookkeeping, the rolling prefix snapshot and `cudaGraphLaunch` itself
  (~0.5 ms at 1,168 nodes) hide behind the tail. The token feed is per
  slot on the device, the pick's masks ride a pinned staging behind an
  in-graph handshake, the bus holds two live replay windows (a ring of
  arms, FIFO finish, in-order adoption) and every graph shape has two
  alternating variants. Eager bus work drains the replays in flight.
  `DGPP_PIPELINE=0` restores the settle-after-launch step. Measured on
  the fabric (one greedy request, the same binary pipelined vs
  `DGPP_PIPELINE=0`): 41.2 vs 41.7 ms per pass at a short prompt, 41.9 vs
  43.0 at 8K, 42.6 vs 43.3 at 32K — 0.5 to 1.1 ms per step, 44.1 / 45.3 /
  43.7 vs 43.4 / 44.8 / 43.1 tok/s on prose / code / JSON (+1.1 to
  +1.6 %); the greedy transcripts are byte-identical to the earlier
  plain, depth-1 and depth-2 runs. Three loopback gates run their eager
  oracles through the drain; the bus test arms two windows at once.

- MTP draft depth (`engine.mtp_depth`, `--mtp-depth`, 1–3): the verify
  runs 1 + depth rows and the draft block's chained rows (one more block
  row per draft, on its own output) propose the drafts after the first.
  The sampled verdict kernel decides T rows in a chain (draw for draw
  what the eager sampled speculator decides), the host fallback continues
  the chain from the row that fell back, the tail ring is guarded around
  the chain rows, the hop snapshot takes the rows the step committed past
  the position, and the device picker carries one slot per draft. Depth 1
  is byte-for-byte the two-row step as built; past depth 1 every step is
  a scalar replay. New gates: the T=3 verdict oracle, the greedy depth-2
  and depth-3 loopback gates (plain transcript, feed == the eager chain),
  the sampled depth-2 loopback gate (transcripts == the eager depth-2
  speculator's through 17 fallbacks). On the fabric the greedy transcripts
  of three prompts are byte-identical plain / depth 1 / depth 2; depth 1
  is unchanged at 42–43 ms/step; depth 2 is 54–56 ms/step with the second
  draft standing 45 % (prose) to 65 % (code) of the time — −4 % on prose,
  +4 % on code and JSON — so depth 1 stays the default
  (`scripts/mtp_depth_check.py` runs the transcript check).
- Per-position draft acceptance: the engine counts, per slot and overall,
  the steps that verified each draft position and the steps in which it
  stood; the stats line's `mtp` group prints `accept p1 74 % p2 61 %` and
  every MTP retire line carries the request's own `accept p1 78 %`.

- The KV cache's dtype is a config key (`engine.kv_dtype`, `--kv-dtype`):
  `bf16`, `fp8` (e4m3 + a row scale) or `fp4` (e2m1 in blocks of 16 with
  e4m3 block scales); the attention kernels dequantize on the load, the
  index cache and the selection are unchanged, the format rides the
  settings record and the config digest. New unit and CUDA tests pin the
  codecs, the append and the three attention kernels per format.
- The memory plan: every rank itemizes every byte it will allocate for its
  configured shape and refuses to boot when it does not fit the node's
  free memory, before the first allocation, naming the largest items and
  the largest `kv_capacity` that would fit; `dgpp-serve --memory-plan`
  runs the check alone. The model's per-forward row bound is now the
  prefill chunk rather than the whole context (the activations no longer
  grow with `kv_capacity`: a 262k-token context needs ~98 GiB per rank
  instead of ~300 GiB and boots), `max_context()` is the context bound.
- Every line the server prints is timestamped: the step-timing report and
  the MoE chain dump go through the logger, an uncaught exception leaves a
  stamped line before the abort, and the launcher stamps its own lines in
  the same format. The stamps are UTC on every rank (a peer's log used to
  carry its node's own zone, seven hours off the head's).
- The throughput line leads with what an operator watches — `decode 30.9
  tok/s, 31.4 ms/tok, 54.6 ms/step (178 steps / 309 tok, 97 % of wall) |
  mtp 1.74 tok/step/req = 74 % drafts accepted | prefill … | live 0,
  queued 0 | pool … | prefix cache … | requests …` — and every request's
  retire line carries its own numbers: the prompt and cached tokens, the
  prefill's ms, the decode tokens and passes, tok/s, ms/tok, ms/pass and
  MTP's tok/pass.
- A non-strict tool argument keeps its type under a keyword that only
  narrows the value (`minimum`, `maxLength`, `pattern`, `format`, …): the
  bound is not enforced and one INFO line says so, once; a schema outside
  the subset in shape still leaves the value free, with one WARN line,
  once (the same three WARN lines used to repeat before every request of
  an agent client). Strict tools refuse as before.
- The settings record carries the resolved row-batch threshold, so the
  pushed and the effective config print the same `batchmin`.
- The decode select kernel (DSA, the step's only context-scaled kernel)
  rewritten: every block scores its stripe and publishes the keys plus a
  radix histogram, the last `rows` blocks each find one row's boundary key
  by radix refinement, gather, and expand with a rank sort — the same set
  under the key's total order, so the bitwise gates are unchanged.
  `dsa_select_bench` at the MTP shape: 4K / 8K / 32K / 393K tokens of
  context 572 / 608 / 832 / 3700 us before, 25 / 33 / 67 / 509 after; the
  12 calls per step were 7–10 ms of the 54 ms step at agent-sized contexts.
  The workspace is `dsa_select_workspace_bytes` (keys and histogram per
  row, sized for the pool capacity), counter_ws is two words.
- The MoE down projection runs four rows per warp (`block_rows_multi`): the
  sliced k of 512 bytes gave a warp one load in flight; the per-row chain
  is factored (`consume_chunk`) so every row is bitwise the old one.
  `moe_slot_bench` at the per-rank geometry: the two-row decode chain
  487 -> 457 us per layer (1.3 ms per step), the eight-row 1598 -> 1412.
- On the fabric (one request, sampled MTP): 41.8 / 44.4 / 44.5 / 45.2
  ms per step at 26 / 8K / 18K / 32K tokens of context, against 43.4 /
  53.6 / 54.8 / 56.2 before the two kernels — 23–25 ms per token at every
  context an agent client uses.
- The decode attention partial kernel keeps each thread's q window and
  c accumulator in registers instead of shared memory: the block drops
  from 95 KB to ~40 KB, two blocks per SM, the 64-block decode grid in one
  wave — 2.24 -> 1.16 ms per step on rank 0's trace at 8K, bitwise the
  smem form (each thread always owned exactly that window). The gather
  loop is unrolled so its token -> block-table -> row chains overlap. The
  split count stays 32 (48 and 64 measured the same step on the fabric).
- The L2 prefetcher coalesces adjacent adds into one launch
  (`DGPP_L2_PREFETCH_MERGE=off` restores one per tensor): the decode
  graph goes from 1,595 to 1,168 kernel nodes and `cudaGraphLaunch` from
  730 to 520 us per step (the host enqueues ~0.45 us per node and the GPU
  idles through it — the interface the design record called hidden). The
  merged kernels stream through more of each collective's handshake
  (+3 us per collective on the timeline), so the step itself is within
  noise of the per-tensor form; kept for the node count.
- The mHC dots kernel's two streaming loops unroll 8 (bitwise; ~0.1 ms
  per step). Tried and reverted, the reasons in the code: batching the
  bus fold's loads (the fold span stayed 9.3 us), and decoding q/k to
  floats in the select kernel (four times the smem bytes per pool).
- The launcher forwards every `DGPP_*` knob in the head's environment to
  the peers.
- The graph collective's fold reads staged copies: the placement gate's
  hash pass, which reads every claimed payload anyway, writes it into
  dynamic shared memory (48 KB for the decode's three 16 KB payloads;
  wider vectors fold from the NIC-placed rows as before), and the fold
  after the last arrival is a shared-memory pass — 9.3 -> 5.9 us per
  collective on the bus timeline, bitwise the same chain. Staging needs
  every peer slot 16-byte aligned (element counts a multiple of 8): the
  step's small all-gathers are not, and the first fabric boot found that
  as a misaligned address the loopback gates' widths never hit.
- The decode attention runs on the tensor-core listed kernel
  (`DGPP_DSA_DECODE_MMA=off` restores the register split kernel): two rows
  x 16 local heads is one M-block, each slab its row's own selection with
  its own request's block table. The DSA layer at the rank's shape:
  0.563 -> 0.516 ms per row per layer at 8K. Numerics: the mma summation
  order, tolerance-equal — the teacher-forced gate over the 556-token
  text passed at a mean NLL delta of +0.0008 nat (bound 0.02), perplexity
  2.587 vs 2.585, no move over 1 nat.
- On the fabric after both: 41.9 / 42.7 / 43.2 / 43.5 ms per step at 26 /
  8K / 18K / 32K tokens of context (400-token generations).
- `DGPP_BUS_TIMELINE=1` writes the bus's per-window collective timeline
  (copy / handshake / skew / fold per collective, the wait histogram, each
  peer's lag and how often it arrived last) at INFO without the rest of
  the debug output, which perturbs what it measures; the launcher forwards
  it, and the head's `DGPP_LOG_LEVEL`, to the peers (the peers used to
  expand the level on their own environment and always ran at info).

## 0.1.0 — 2026-09-06

Productionizing: the model-agnostic code left the `glm` namespace
(`dgpp::serve`, `dgpp::sched`, `dgpp::sample`, `dgpp::text`; GLM under
`src/models/glm/`; the binary is `dgpp-serve`); one cluster config
(`deploy/cluster.json`, a site's copy of `deploy/cluster.example.json`)
read by the launcher and every rank, the head
pushing the world's settings to the peers over the journal before anything
builds; versioned releases (`scripts/release.sh`) installed once per node
(`dgpp-cluster install`), the version stamped into the binary and refused
across a mixed world; `scripts/dgpp-cluster` as the launcher.

The request contract gained `stop`, `n`, `logit_bias`, and the usage's
`cached_tokens` and `reasoning_tokens`. Every rank's log carries one
aggregate throughput line per 10 s; the per-tick lines moved to DEBUG.

## 2026-09-05 — M9 sign-off, prefix caching and service hardening

- M9: the continuous op-stream drift check on every journal record; a
  malformed-HTTP fuzzer under AddressSanitizer (three defects found and
  fixed); the one-hour mixed soak; the operator's page; the sign-off report
  (32K TTFT, MTP acceptance per class, the prefix cache's capacity curve);
  the ranked next-steps list.
- M7: the exact snapshot prefix cache — decisions rank-identical in the
  scheduler, the snapshot arena in the engines, hot == cold bitwise.
- The v1 failure semantics built, gated and drilled with kill −9 on the
  four nodes; the strict tool grammar's key ledger; cancellation under the
  one-graph step gated.
- Prefill rounds 4–9: the bulk collective as a cooperative kernel with paced
  senders, the MoE experts on grouped tensor-core kernels, the attention
  prefill as a flash kernel; 256 tokens 0.58 s, 2,048 tokens 1.7 s.

## 2026-09-04 — sampling, constrained output and shutdown handling

Tool calls and reasoning on the wire; constrained decoding as a guarantee
for `tool_choice` and `parallel_tool_calls`; `response_format` with JSON
schemas through the same masks; typed tool arguments; drain-on-stop;
grow-on-demand admission; exact sampling on the service with the on-device
verdict and its gather fallback; the sampling-width sweep; prefill rounds
1–3.

## 2026-09-03 — M8: transactional MTP; the on-device step

Greedy speculative decode with the checkpoint's MTP layer as one graph
replay per step, transcript identical to plain decode (31.3 → 22.45 ms per
token); the pick behind the head on the device; the adaptive scalar and
row-batched graph variants that removed the low-occupancy regression; the
kernels-only decode graph that closed the batched-MTP loopback stall; the
teacher-forced log-probability gate; fp32 logits.

## 2026-09-01 to 2026-09-02 — M6: generation, tokenizer, service

The incremental decode engine; the byte-exact tokenizer and the Jinja
chat-template interpreter; the deterministic scheduler; the
OpenAI-compatible service at world 1, then on the fabric behind the
admission journal; the decode step as a recorded graph with the
collectives as graph nodes; eleven optimization rounds on the T=1 step
(393 → 31.45 ms per token); the one-pass resident load and the per-rank
image cache (boot 15–25 s).

## 2026-08-29 to 2026-08-31 — M5: four-rank tensor parallelism

The RC/RoCE CollectiveBus with its slot pools, dual-lane striping and
watchdogs; the epoch-based roster; the sharded resident load; the fabric
TP=4 parity gate at real dimensions; the resident serving mode.

## 2026-08-27 to 2026-08-28 — M0 to M4

Platform, topology, transport and checkpoint facts; the core runtime, the
safetensors loader and the synthetic graph testbed; the KDA operators and
state manager; DSA/MLA sparse attention with its index pools; the
assembled 45-layer GLM forward with per-layer parity against torch
references.
