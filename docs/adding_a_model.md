# Adding a new model family

This page is the checklist for bringing a new architecture to the native
engine: a loader, a model, a deploy template and the test gates that make the
family maintainable. The per-family specifics live in that family's plan
document and [model card](model_cards/); the general change discipline
(tests, fabric evidence, engineering records, docs) is
[CONTRIBUTING.md](../CONTRIBUTING.md). The rules here are the ones ports have
actually been reviewed against — the Qwen3.8-27B port (PRs #79 and #80 and
the follow-up issues #84–#93) is the worked example.

## 1. Survey the checkpoint, then write the plan document

Read the checkpoint's `config.json` and safetensors headers before writing
code: layer types and their pattern, hidden/head geometry, every weight's
dtype, quantization format and scales, and what a decode step streams. The
plan document (`docs/<model>_plan.md`, the pattern of
[qwen38_flash_next_plan.md](qwen38_flash_next_plan.md)) carries the tensor
census, the architecture study, the placement decisions and the dated port
record. Reviewers judge the PR against it, so write it first. A checkpoint
inventory under `docs/checkpoint_budget*.md` is worth generating when memory
planning is tight.

Decide the world sizes now. If the loader carries TP slices but the model
refuses world > 1, that belongs in the PR text and the plan: the public
benchmark targets for most models are TP=2 and TP=4.

## 2. Architecture detection and the loader

- Map the architecture in `src/loaders/architecture.{hpp,cpp}` (the
  `ModelArchitecture` enum, the `arch`/`model_type` match, the name string).
- Write the binding table and streaming loader under `src/models/<family>/`.
  Load the checkpoint's native quantization; do not requantize at load time —
  anything lossy is an opt-in engine key (§4), not loader behavior.
- Every new weight class must be handled in the exhaustive `is_replicated`
  switch in `src/models/qwen/loader.cpp`. The `ci` preset builds with
  `-Werror=switch`, so an unhandled class fails every CI build, not just this
  family's.
- The resident image goes through the family's
  `set_resident_image_dir(GlmLayerStream::resident_image_dir())` in the serve
  wiring (§7) so `paths.resident_cache` applies, like every other family.
  Never write the image into the HF snapshot directory: on `setup.py`-managed
  nodes that directory is read-only, and a 20–30 GiB image lands inside the
  checkpoint cache as junk. The loader's own fallback dir is for tools and
  tests that load directly, never for serving.
- New quant kernels get kernel tests with a host reference
  (`tests/cuda/*_kernels_test.cu`). A kernel that ships off-by-default with
  env gates, debug macros and no test does not ship.

## 3. The model

Implement the model over the shared layer kernels and the shared engine
machinery — packed decode batching, the decode graph, chunked resumable
prefill, K/V and linear-attention state snapshots, prefix caching, the
speculative-decode framework — rather than beside them. Constraints the
engines enforce and ports have tripped on:

- **Chunk invariance.** The state a request leaves must not depend on how its
  prompt was chunked: a cold prefill and a resume from a snapshot run the same
  numerics for the same rows. A recipe that splits on row count (exact GEMV
  under 128 rows, a different path above or on resume) breaks this and cannot
  be the default.
- **Snapshot positions** must be chunk ends of the cut list the scheduler
  gives the engine (the `glm_tp_test` gates fail otherwise), and shared-system-
  prompt head snapshots exist so the next conversation attaches at the head —
  a per-family cost complaint is a keyed decision, not a scheduler rule.
- **Speculative steps return exactly the tokens decided that step.** On the
  plain path that is the verify's next token; a speculator that also returns
  the fed token duplicates the first token of every transcript and drops the
  last. Verify the contract by diffing a speculative world's transcripts
  against the plain world's on the same binary, not against a remembered
  reference.
- **Determinism across ranks**: anything a rank decides that another rank must
  agree with is a pure function of the journaled stream, checked by the
  op-stream fold (CONTRIBUTING).

## 4. Engine keys, exact defaults

Repo rule: **anything that changes what the engine computes is an `engine.*`
key, never an environment variable.** Diagnostics (`TRACE`-style env reads)
excepted. Every lever is plumbed the same seven ways:

1. parse and validate in `src/serve/cluster_config.cpp`,
2. the settings record the launcher carries to every peer rank, so all ranks
   parse the same value,
3. a `--flag` on `dgpp-serve`,
4. a static setter applied before the memory plan and the model build,
5. a `cluster_config_test` case (including a refusal when the family has no
   such path),
6. a row in the README key table,
7. a note in [operations.md](operations.md).

**The exact path is the default.** Anything lossy beyond the checkpoint — an
FP8 head requant, a per-tensor prefill recipe, activation quantization — is
opt-in and documented with its measured transcript cost, following the
`engine.dense_weights` / `engine.prefill_fp8_per_tensor` precedent. Measure
the default against the exact path before claiming bit-exactness: a lossy
default changing 4/4 battery transcripts within the first hundred tokens is
exactly what a review looks for.

## 5. Deploy templates

One template per model, quantization and world size:
`deploy/cluster_<model>_<quant>_w<n>.example.json`, with the speculative mode
on; the plain world is `--no-mtp`, not a second template. See
[deploy/README.md](../deploy/README.md) for the naming rules and variants. A
template is a template: `stats_interval_s: 10` like the others, and paths the
site fills in stay empty placeholders
(`"paths": {"resident_cache": ""}`, never a developer's own node paths). Then
register the model in `tests/python/portability_test.py`'s model map and the
`deploy/README.md` table — the test enforces both — and touch no other
model's template.

## 6. The forward fixture gate

Every family lands a `<family>_forward_test` (the `qwen35_forward_test`
pattern): `--write-fixture` writes a tiny random checkpoint **in the release's
own layout, generated from the binding table itself** so fixture and table
cannot disagree; a host-reference forward checks `--smoke`, `generate` and
`decode` rows, plus the verify/rollback parity of whatever speculation the
family serves. Kernel tests and "N/N transcripts" are not a substitute — a
host reference that encodes the same index bug as the kernel passes
forever. Wire the fixture/smoke/generate/decode cases into `CMakeLists.txt`
with the family's other tests.

## 7. Serve wiring

Register the family's `ServeFamily` in `apps/dgpp_serve.cpp`: the name, the
model build (including the resident-image dir, §2), the family checks for
keys it refuses, and `paths`/`engine` interactions. Do not add a new serve
wrapper script — `scripts/dgpp-cluster up|down|status --config FILE --bin BIN
--knobs=...` already serves any world, with logs under the deployment dir and
the op-stream digest at `down`. A duplicate launcher fails
`script_reports_test` and drifts. If you touch a shared script's contract
(a probe's return values, a report format), run the whole host python suite.

## 8. Validation before opening the PR

- `cmake --preset ci` builds clean (`-Werror`), and
  `cmake --build --preset ci -j 4` builds every target.
- Full `ctest --test-dir build-ci -j 1`, serial, on an idle GPU, **with master
  merged in**. Ports routinely break other families' suites — scheduler, GLM
  TP, portability, script reports — from changes that look local. A suite that
  passes by skipping checkpoint cases proves nothing.
- The greedy transcript battery at T=1, with speculation, and with the decode
  graph, bit-exact against a *named* reference (a plain world on the same
  binary is the usual one).
- Fabric evidence for anything on the path the ranks execute together: boot
  the world with `scripts/dgpp-cluster`, run the check, `down`, compare the
  op-stream md5s across ranks (CONTRIBUTING §What a change needs).
- A dated campaign under `benchmarks/results/` with the standard classes and
  concurrencies before quoting any number, reusing the campaign harness and
  `timed_load`; a change that moves a published number re-runs its procedure
  from [benchmarks.md](benchmarks.md) §9 and updates the row.

## 9. Docs landing

The CHANGELOG entry, the README rows (supported models, new keys, new
templates, this page's docs index), a `docs/model_cards/<model>.md` card,
operations notes, and `DESIGN.md` / `PLAN.md` status per CONTRIBUTING. A
CHANGELOG entry that documents env knobs instead of keys, or states
transcript identity without the defaults' numerics, is not done.

## What bites ports

- Shared code changed for one family: scheduler cut/snapshot rules, the shared
  GEMM row-band dispatch, `kSpecRows`. Family-specific behavior goes on the
  family's own objects (a model's `configure_gemm_rows`, a keyed decision).
  A genuinely shared change is **its own PR with op-stream replay evidence** —
  and it needs an issue so the deferral does not vanish.
- Test-name-level claims ("12/12 bit-exact") that turn out to compare the
  wrong worlds: check the transcript's first and last token, not just its
  digest count.
- Silent memory growth: a boot-requantized lever can add 20+ GiB resident;
  the memory plan must carry it and the template must size to it.
- Untracked local work: a fix that lives on one node's disk (or an
  unpushed branch) is not support. Push the branch, open the PR.
