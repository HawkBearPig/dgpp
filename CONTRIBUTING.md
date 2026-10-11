# Contributing

DGPP is developed against real hardware: a DGX Spark (GB10) node builds and
runs the single-node suites, and the four-node RoCE fabric runs the
multi-node checks and serving benchmarks. This page describes how to
prepare and validate a change.

## Environment

- A GB10 node with CUDA 13 (the toolkit, cuBLASLt, an SM 12.1 compiler),
  CMake 3.25+, a C++20 compiler, rdma-core with libibverbs, Python 3.10+.
  Keep libibverbs installed for serving, including single-node deployments.
  `-DDGPP_ENABLE_IBV=OFF` omits the server and RDMA targets; it is only useful
  for development that does not need them.
- The checkpoint in the Hugging Face cache (`~/.cache/huggingface/hub`)
  for the real-model checks; the synthetic fixtures the tests write need
  nothing.
- The fabric for anything under `src/net/`, the graph engine, or the
  service's multi-rank paths: the loopback gates cover the protocol on one
  node; four-node checks exercise the network.

Follow [Getting started](docs/getting-started.md) for head/peer dependencies,
CUDA discovery, site settings and checkpoint preparation. Use
`--config /path/to/deployment.json` explicitly for deployment commands.

## Build

For a first checkout, configure and build the host test targets:

```bash
cmake --preset ci
cmake --build build-ci -j 4 --target unit_tests http_server_test serve_test fabric_serve_test scheduler_test roster_check dgpp_serve_app
ctest --test-dir build-ci -L host -LE checkpoint --output-on-failure
```

These checks include the Python suites and do not need GPU execution or model
weights. The native build still needs the compiler and library dependencies
above. See [testing](docs/testing.md) for labels, fixtures and optional tools.

For a full run, reserve idle test hardware, prepare the required checkpoints
and fixtures, then build all targets before testing:

```bash
cmake --build --preset ci -j 4
ctest --test-dir build-ci --output-on-failure -j 1
```

`scripts/ci-local.sh` runs the full build and test sequence, including GPU/RDMA
suites. Use it only on idle test hardware. Production uses `cmake --preset release`
and `cmake --build --preset release -j 4`, producing the server in `build-release/`
with `-O3` and no debug symbols. The launcher defaults to that binary; select
`--bin build-ci/dgpp-serve` explicitly for a testing deployment. `scripts/release.sh`
packages the release build. The other testing presets are `debug`, `asan`
and `ubsan`; see [build profiles](docs/testing.md).

- **Rebuild the selected test targets before `ctest`; build everything before
  an unfiltered run.** The suite runs whatever binaries exist; an old binary
  can pass without testing your changes.
- **Run GPU/RDMA suites serially on idle test hardware, never alongside production
  serving.** Loopback worlds use GPU memory and fixed ports in the 299xx range.
  `DGPP_TEST_FILTER=<substring>` runs a subset of a binary. Read skip messages:
  a passing suite with skipped checkpoint cases does not validate those models.

## What a change needs

A new model family carries the extra gates in
[adding a model](docs/adding_a_model.md): the plan document, the forward
fixture gate, the deploy template registration and the engine-key plumbing.

1. **Tests.** New behavior gets a test that would fail without it, in the
   suite whose subject it is (`tests/unit` for pure logic, `tests/host` for
   the scheduler, the service and the journal with their fakes,
   `tests/cuda` for kernels and loopback worlds). Test invalid requests
   and configuration values as well, including whether the error names
   the offending field.
2. **Evidence on the fabric** when the change touches the path the ranks
   execute together: boot the world (`python3 scripts/dgpp-cluster up --config "$CONFIG"`), run the
   relevant check (`docs/operations.md` lists them; `scripts/serve_api_check.py`
   for the request contract), stop it with the same config
   (`python3 scripts/dgpp-cluster down --config "$CONFIG"`) and compare the
   op-stream md5s across all participating ranks. Set `CONFIG` to the deployment
   filename in the current shell. A numerics change is judged as `docs/numerics.md`
   describes, not by eye. A change that moves a published throughput number
   re-runs that number's procedure from `docs/benchmarks.md` §9 and updates the
   row there, with its date.
3. **Engineering records.** Changes with new measurements or debugging
   findings get a dated record in `benchmarks/results/` describing the change,
   validation and results. Continue an existing record only when it covers the
   same investigation; do not append unrelated work to an old milestone report.
   Preserve the context and names used in historical entries.
4. **Documentation.** Keep `DESIGN.md` aligned with the implementation and
   `PLAN.md` with its status. Update `docs/operations.md` and `README.md`
   when a change affects setup, usage or supported behavior.

## Style

- C++20, two-space indent, Google-derived formatting (`.clang-format`);
  format new and touched code with the `format` target (the tree has not
  been reformatted wholesale, so `format-check` over everything is not
  yet clean).
- Write documentation in direct, connected prose. Explain current behavior
  and its constraints before the details. Comments should explain intent,
  invariants or a non-obvious tradeoff. Put investigation chronology in a
  dated engineering record and link to it when the evidence helps explain
  a constraint; avoid dates and milestone labels in routine comments.
- Reject invalid configuration and protocol state with a useful error.
- Preserve determinism across ranks: anything a rank
  decides that another rank must agree with is a pure function of the
  journaled stream, and the op-stream fold checks it every tick.
- Operational configuration, launch and diagnostic scripts use the Python
  standard library. Hub downloads need `requirements-download.txt`; reference
  generators may need the optional dependencies described in the setup guide.

## Commits

One change per commit, with a subject that says what changed and why in
plain words; the record entry carries the detail. Commit to `master`
after the full build and the suite pass and the fabric evidence, when the
change needed it, is in the record.

## Reporting a problem

Include the version (`dgpp-serve --version`, or the `dgpp-serve 0.x` line
at the top of a rank's log), the `config:` line every rank logs at
startup, rank 0's log and the peers' (`dgpp-cluster down` fetches them
into the log dir), and the op-stream md5s if the world stopped cleanly.
