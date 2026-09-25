# Scripts

This directory contains the cluster launchers, test clients, fabric diagnostics,
and reports used to build and measure dgpp. Bash handles process orchestration
and log collection; Python handles HTTP clients, configuration, and analysis.
The tables below cover every source file in the directory, grouped by use.

A *world* is the set of ranks serving one deployment, usually one rank per node.
MTP means multi-token prediction; SSE is the streamed HTTP event format used by
the chat API.

## Before running a script

Run commands from the repository root unless a script says otherwise. Cluster
launchers run rank 0 locally, so invoke them on the head node: the first entry
in `DGPP_NODES`. Fabric runs need the relevant binaries built, passwordless SSH
to the peers, and the checkpoint available in each participating node's local
cache. Serving defaults to `build-release/dgpp-serve`; test and diagnostic
wrappers use binaries from `build-ci/`.

Shared node addresses, SSH user, ports, and staging/log/release paths belong in
the repository's `.env`; [`.env.example`](../.env.example) lists the settings.
Exported settings take precedence, and `DGPP_ENV_FILE` selects a different site
file. Deployment JSONs select the model, world size, and engine settings.
Select the deployment explicitly with `--config FILE`, or the positional
`CONFIG` argument for wrappers that use one. The world uses the first
N configured nodes. `.env` is parsed as data, not sourced as Bash, and unrelated
entries such as access tokens are not loaded by the site helper.

HTTP clients either take their endpoint as arguments or use the selected
deployment's HTTP settings. Shared defaults bind to localhost. Request model
IDs are discovered from `/v1/models` unless explicitly supplied. Family-specific
native checks still require an appropriate model. The failure, stop, and serving
soak procedures require four ranks and reject other worlds before starting.

Use an idle test cluster for launch-and-stop procedures. Cleanup targets
recorded process identities; the failure drill deliberately sends `SIGKILL`
to a recorded rank. `up --replace` explicitly stops the selected deployment;
`down` without `--config` stops every recorded deployment that is running.
Choose a fresh output directory to avoid overwriting previous results. The
HumanEval task in `serve_eval.py` executes generated Python locally with a
timeout, not a security sandbox; run it in an isolated evaluation environment
and pass `--allow-code-execution`. See [Getting started](../docs/getting-started.md)
and [test/data preparation](../docs/testing.md).

For exact arguments, read the usage header or use `--help` where supported.
Some older scripts start work immediately and do not implement `--help`.

## Build, configuration, and deployment

| File | What it does | When to use it |
| --- | --- | --- |
| [README.md](README.md) | This directory's file index and operating notes. | Start here to choose a launcher, client, or report. |
| [spark-cross](spark-cross) | Builds an x86 Docker toolchain and cross-compiles the ARM64/GB10 server. | Build on an x86 Linux workstation; see [cross-compiling](../docs/cross-compiling.md) for staging and target validation. |
| [setup.sh](setup.sh) | Finds `python3` and launches guided setup from any working directory (setup.py itself turns away interpreters older than 3.10). | Start here after cloning; `--help` lists unattended and read-only options. |
| [setup.py](setup.py) | Guides deployment/site and RoCE choices, checks prerequisites locally and on peers, optionally installs system packages, builds, downloads/syncs and runs final preflight. | Implementation of `setup.sh`; `--check` checks prerequisites without writes, `--configure-only` saves settings, and `--start` launches after success. |
| [ci-local.sh](ci-local.sh) | Configures CMake, builds, and runs CTest; defaults to the `ci` preset and `build-ci/`. | Validate a local code change. `DGPP_PRESET` selects `build-<preset>/`; `DGPP_BUILD_DIR` explicitly overrides the directory. |
| [release.sh](release.sh) | Builds or reuses the release build, stages the install layout, and packages a versioned tarball with checksums under `dist/`. Rejects testing build directories and binaries with debug information or sanitizer instrumentation. | Prepare an artifact for installation; `--no-build` repackages an existing release build. Uses `DGPP_BUILD_DIR` or `build-release/`. |
| [dgpp-cluster](dgpp-cluster) | Combines site settings with a deployment JSON; starts, stops, and inspects ranks, installs releases, and lists installed versions. | Manage a serving deployment. `resolve` prints the merged runtime JSON without SSH or startup. |
| [serve_run.sh](serve_run.sh) | Wraps `dgpp-cluster`, passing `DGPP_SERVE_KNOBS` as engine flags and `DGPP_SERVE_LOG` as the log directory. | Run `up`, `down`, or `status` through the environment-based interface used by the test procedures. |
| [cluster_env.sh](cluster_env.sh) | Loads site settings through `site_env.py` and defines Bash helpers for nodes, peers, user, ports, and config resolution. | Source it from an orchestration script that needs shared site information. |
| [site_env.py](site_env.py) | Parses allowlisted `.env` settings, applies exported overrides, and resolves deployment configuration. Exposes both Python functions and a CLI. | Reuse site parsing in Python or inspect the selected nodes and resolved configuration. |
| [cluster_doctor.py](cluster_doctor.py) | Read-only local/SSH probes for platform, GPU, cache completeness, libraries, ports, paths and RoCE selection. | Run through `dgpp-cluster doctor`, automatically invoked before `up`; `--local-only` omits SSH. |
| [cluster_process.py](cluster_process.py) | Starts isolated process sessions and records PID, owner, start time and boot identity. | Internal helper for launch/status/signals; prevents stale PID records from authorizing unrelated cleanup. |
| [download_model.py](download_model.py) | Downloads a complete checkpoint once on rank 0 into the HF cache, then syncs the selected snapshot and blobs to peers sequentially. | Run with `--config FILE`; `--sync-only` reuses an existing download, `--verify-only` checks every node without writes, and `--local-only` skips peers. `--model ORG/NAME` downloads locally without deployment settings. |
| [cache_sync.py](cache_sync.py) | Builds a snapshot/blob file list, transfers it with rsync content checksums, and validates the peer copy before updating `refs/main`. | Internal downloader helper; does not copy credentials or unrelated cached revisions. Peers need Python, SSH and rsync, not the HF Python package. |
| [discover_roce.py](discover_roce.py) | Lists local RoCE candidates and the deployment's configured/automatic lane order; retains reachable nodes' results when a peer fails. | Run locally without arguments or inspect deployment nodes with `--config FILE`; no network changes or RDMA traffic. `--json` returns `inventories`, `selections` and `errors` keyed by host; any probe error causes a nonzero exit. |
| [prepare_data.py](prepare_data.py) | Fetches pinned GSM8K/HumanEval data with provenance manifests or tokenizes a supplied text into CSV. | Prepare benchmark inputs explicitly before running evaluation or prefill checks. |
| [data_paths.py](data_paths.py) | Resolves `DGPP_DATA_DIR` and reports missing datasets with preparation instructions. | Shared by data-dependent clients and preparation tools. |
| [serve_client.py](serve_client.py) | Resolves the selected HTTP endpoint/log path and discovers the served model ID. | Shared by clients; its CLI prints `url`, `log`, or `model` for shell callers. |

For a configuration-only check from the repository root:

```bash
python3 scripts/dgpp-cluster resolve --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json
```

## Fabric runs and model checks

These scripts launch work on the nodes. The direct loader and forward checks
take a `WORLD` argument; the serving wrappers take a deployment `CONFIG`.

| File | What it does | When to use it |
| --- | --- | --- |
| [fabric_run.sh](fabric_run.sh) | Stages and launches a fabric application, starts the head before peers, monitors completion, and collects rank-consistency results. Defaults to `glm_gen_check`; supports `--app`, `--fetch-logs`, and `--node-probe`. | Run engine-level generation or another compatible fabric application without HTTP. `--force` stops only jobs recorded for the same output directory. |
| [fabric_bus_probe.sh](fabric_bus_probe.sh) | Runs `bus_check allreduce` across a chosen world size and collects eager and graph-replay collective timings. | Measure collective overhead independently of model execution. |
| [fabric_gr_probe.sh](fabric_gr_probe.sh) | Compares plain and MTP generation with and without extra all-reduces injected through `--gr-probe`. | Measure how additional collectives affect a real decode step. |
| [fabric_qwen_load.sh](fabric_qwen_load.sh) | Runs `qwen_load_check` independently on each rank and prints memory totals and digests; no collective bus is needed. | Check Qwen checkpoint loading and rank slicing before testing forward execution. |
| [fabric_qwen_forward.sh](fabric_qwen_forward.sh) | Runs the Qwen forward check on identical token IDs across ranks, checks digests, and prints the merged argmax IDs. | Validate tensor-parallel forward consistency; compare the printed argmax with a separate world-1 run. |
| [fabric_glm4_forward.sh](fabric_glm4_forward.sh) | Runs the equivalent cross-rank forward check for GLM-4.7 through `glm4_forward_check`. | Validate GLM-4.7 tensor-parallel digests and compare argmax IDs across world sizes. |
| [fabric_qwen_serve.sh](fabric_qwen_serve.sh) | Starts a Qwen deployment, runs greedy transcripts, client timing, and API checks, optionally evaluates tasks, then stops the world. | Validate a Qwen serving configuration; supply `--compare` to check a saved transcript baseline. |
| [fabric_glm4_serve.sh](fabric_glm4_serve.sh) | Runs the corresponding serving checks for GLM-4.7, with thinking disabled for optional task evaluations. | Validate a GLM-4.7 serving configuration or compare it with saved transcripts. |
| [fabric_glm_dsa_serve.sh](fabric_glm_dsa_serve.sh) | Runs the serving checks for the full GLM-5.3 (`glm_moe_dsa`): transcripts, pace, API check, MTP acceptance per prompt class, prefill by length, optional task evaluations (thinking stays on: the template has no switch). | Validate a full GLM-5.3 serving configuration or compare it with saved transcripts. |
| [fabric_v41bench.sh](fabric_v41bench.sh) | Boots a world and runs the vLLM DGX Spark recipe's own DeepSeek-V4.1-Flash benchmark client (its prompt set v1, thinking off, decode tok/s after the first token) against it at the given concurrency levels and prefill targets. | Compare a served DeepSeek-V4.1-Flash configuration like for like with the numbers that recipe reports on the developer forums. |
| [fabric_profile.sh](fabric_profile.sh) | Boots any family's world with rank 0 under nsys, runs a warm request and a profiled client (a single 400-token stream by default, or any client command such as a `serve_load` sweep at a concurrency), takes the world down and prints the per-step kernel breakdown between instances of a once-per-replay marker kernel. | Attribute a decode step's time to kernels for a served configuration, single-stream or batched. |
| [fabric_serve_load.sh](fabric_serve_load.sh) | Boots any family's world from a cluster config (with `--knobs` for the engine flags under test) and runs the greedy transcripts, the MTP acceptance per prompt class, and the concurrency probe, then takes the world down and prints the scheduled-depth summary. | Measure a served configuration's decode at one and two live requests, or A/B an engine flag such as `--mtp-schedule` against a reference run. |
| [fabric_mimo_serve.sh](fabric_mimo_serve.sh) | Runs the serving checks for MiMo-V2.6-Flash (`mimo_v2`): transcripts, pace, API check, MTP acceptance per prompt class, prefill by length, optional task evaluations with thinking disabled. | Validate a MiMo-V2.6-Flash serving configuration or compare it with saved transcripts. |
| [fabric_dsv41_serve.sh](fabric_dsv41_serve.sh) | Runs the serving checks for DeepSeek-V4.1-Flash (`deepseek_v41`): transcripts, pace, API check, DSpark acceptance per prompt class, bounded prefill by length, optional task evaluations. | Validate a DeepSeek-V4.1-Flash serving configuration or compare it with saved transcripts. |
| [fabric_qwen_pace.sh](fabric_qwen_pace.sh) | Starts a deployment, warms it up, takes two single-stream timing readings, collects server stats, and stops it. | Quickly compare decode pace between configurations or engine settings. |
| [fabric_qwen_prefill.sh](fabric_qwen_prefill.sh) | Starts a deployment, measures prefill at selected prompt lengths through `serve_prefill_probe.py`, and stops it. | Compare Qwen prefill settings or implementations at several context lengths. |
| [fabric_qwen_profile.sh](fabric_qwen_profile.sh) | Runs rank 0 under Nsight Systems, sends a decode or prefill workload, stops the world, and produces a kernel breakdown. | Capture a detailed GPU profile; requires `nsys`. |
| [fabric_glm4_load.sh](fabric_glm4_load.sh) | Starts a deployment with periodic stats, runs fixed-concurrency request phases, stops it, and matches server stats to those phases. | Measure GLM-4.7 throughput and MTP acceptance as concurrency changes. |
| [fabric_prefill_check.sh](fabric_prefill_check.sh) | Truncates a supplied token-ID file to exact lengths and runs short GLM generation jobs to report prefill time. | Measure prefill without HTTP or tokenizer-length estimation, using the selected deployment's model. |
| [fabric_prefill_repeat.sh](fabric_prefill_repeat.sh) | Reads `DGPP_DATA_DIR/hard_ids.csv` or `--ids-file FILE` and runs each prompt with three prefill passes. | Measure warmed-up prefill at selected lengths; prepare tokens with `prepare_data.py tokens` first. |
| [fabric_mtp_classes.sh](fabric_mtp_classes.sh) | Runs greedy graph-mode MTP generation for selected chat, code, prose, JSON, or math prompts and reports acceptance and rank consistency. | Compare MTP behavior across prompt types. |
| [fabric_glm_regression.sh](fabric_glm_regression.sh) | Combines a GLM MTP chat run, plain decode timing, and repeated prefill measurements. | Gather a consistent set of GLM regression measurements for comparison with a saved baseline. |

## Clients for an already-running service

These clients send requests but do not start or stop the world. Use a quiet
endpoint for timing and metric-delta measurements so unrelated requests do not
affect the results.

| File | What it does | When to use it |
| --- | --- | --- |
| [serve_api_check.py](serve_api_check.py) | Checks stop strings, multiple choices, logit bias, and usage details in streamed and non-streamed responses; fails if a check fails. | Verify request-field behavior after a server change. |
| [vision_api_check.py](vision_api_check.py) | Generates PNG inputs and checks colors, multiple images, streaming, choices, chunk boundaries and concurrent requests. | Verify a GLM-5.3-Flash image deployment on idle test hardware. |
| [serve_bench.py](serve_bench.py) | Times a streaming chat request and reports time to first token, decode pace, wall time, and usage. Can obtain its default endpoint from site settings. | Take a quick client-side latency reading. |
| [serve_load.py](serve_load.py) | Sends fixed-concurrency streaming request phases and reports per-request timing and aggregate throughput. Supports prompt-class sweeps. | Measure serving throughput as concurrency or workload changes. |
| [timed_load.py](timed_load.py) | Adds engine decode rates to `serve_load.py`, reconciling published counters with completed requests under a deadline. | Measure engine-call time on an otherwise idle server; uses the same CLI and also accounts for warmup and isolation requests. |
| [serve_mtp_classes.py](serve_mtp_classes.py) | Sends the five class prompts greedily and reads each request's retired line from rank 0's log: pass time, tokens per pass and draft acceptance as the engine counted them. | Compare MTP acceptance per prompt type through the service. |
| [serve_greedy_transcript.py](serve_greedy_transcript.py) | Saves fixed greedy prompts and replies to JSON, with optional exact comparison against a reference file. | Check whether plain/MTP modes or deployment changes preserve API-level output. |
| [mtp_depth_check.py](mtp_depth_check.py) | Writes separate prose, code, and JSON completions and prints acceptance information from a rank-0 log. | Collect outputs from separate MTP-depth launches for a directory diff. |
| [serve_eval.py](serve_eval.py) | Scores HumanEval, GSM8K, and synthetic structured extraction, saving per-item results and summaries. | Compare task accuracy between deployments. HumanEval executes generated code; GSM8K and HumanEval require local datasets. |
| [serve_prefill_probe.py](serve_prefill_probe.py) | Calibrates approximate prompt lengths, defeats prefix reuse with unique prefixes, and measures prefill metric deltas and first-token latency. | Isolate prefill cost through HTTP; requires `DGPP_DATA_DIR/gsm8k_test.jsonl`, or `--data FILE`. |
| [serve_prefill_interference.py](serve_prefill_interference.py) | Runs a long prefill beside an active decode stream and reports the decode client's update-gap distribution. | Measure how budgeted prefill affects an overlapping request on an otherwise idle server. |
| [serve_prefix_curve.py](serve_prefix_curve.py) | Interleaves three-turn conversations and reports cache hits, misses, evictions, saved tokens, and per-turn latency. | Measure prefix-cache capacity at a chosen conversation count. |
| [serve_nvme_cache_check.py](serve_nvme_cache_check.py) | Runs a long document through cold prefill, a changed question, eviction by other documents and a restore from the NVMe cache; reports each step's first-token latency, the transcript equality of the in-memory and restored hits, and the `nvme_cache` metric deltas. | Verify the cold tier on a booted world (boot with a small `--prefix-cache-gib` and the tier on). |
| [serve_agentic_streams.py](serve_agentic_streams.py) | Runs concurrent multi-turn tool or essay conversations, optional side requests, and system-prompt mutations; records cache reuse and timing. | Check prefix reuse under agent-style traffic and detect unexpected cache misses. Tool results are synthetic. |
| [serve_soak.py](serve_soak.py) | Runs timed mixed traffic: short conversations, long answers, disconnected streams, and admission bursts; saves requests, metrics, and latency summaries. | Test sustained serving behavior and tail-latency stability. |
| [qwen_yarn_release_check.py](qwen_yarn_release_check.py) | The opt-in YaRN release check: deterministic needle probes in 262 144- and 524 288-token documents, with cold prefill, prefix-cache reuse, incremental decode, concurrent streams, sampled peak memory and a short-context control, optionally replayed against a reference endpoint so retrieval misses are attributed. Refuses to run without `--run` and is never part of CI. | Accept a `engine.rope_scaling` build on real hardware before release; see [docs/qwen_yarn_release_check.md](../docs/qwen_yarn_release_check.md). |
| [step_probe.py](step_probe.py) | Sends streamed requests at selected context sizes, optionally with tool schemas, and joins their timing with rank-0 decode-step counts. | Separate token pace from engine step cost on an idle service; requires access to its rank-0 log. |
| [serve_c1_probe.py](serve_c1_probe.py) | Brackets each single-request run with scheduler counters and separates engine-call time from client-visible stream timing. | Measure repeated C1 engine throughput by prompt class on an idle server. |

## Serving lifecycle, stress, and sweep procedures

These procedures own the service lifecycle and change its engine settings for
the test. They are not passive checks against an existing production service.

| File | What it does | When to use it |
| --- | --- | --- |
| [serve_admission_check.sh](serve_admission_check.sh) | Runs the same request batch under full-reservation and grow-on-demand admission with a tight KV pool, then prints outcomes, counters, and op-stream hashes. | Compare admission policies and inspect their cross-rank decisions. |
| [serve_tools_check.sh](serve_tools_check.sh) | Starts the service, exercises tool choices, follow-up turns, reasoning, streaming, and structured JSON responses, then stops and summarizes it. | Inspect tool-calling and structured-output behavior end to end. |
| [vision_prefix_cache_check.py](vision_prefix_cache_check.py) | Checks image prefix reuse, continuation, streaming and pixel isolation against an idle GLM server. | Validate image-aware prefix caching through HTTP. |
| [prefill_fairness_check.py](prefill_fairness_check.py) | Checks that a long cold prefill yields to active streamed decodes, optionally including images. | Validate prefill continuation against an idle server. |
| [serve_stop_check.sh](serve_stop_check.sh) | Stops a four-rank world while streamed and non-streamed requests are active, then reports responses, drain logs, exits, and op-stream hashes. | Check graceful shutdown with in-flight work. |
| [serve_failure_drill.sh](serve_failure_drill.sh) | Kills a selected rank during streaming, checks failure propagation, restarts the service, and compares received text with replayed answers. | Exercise crash recovery on a dedicated four-rank test world. This deliberately uses `kill -9`. |
| [serve_soak_run.sh](serve_soak_run.sh) | Starts a four-rank serving world and per-node probes, runs `serve_soak.py`, then stops and collects logs, counters, and op-stream hashes. A preset `DGPP_SERVE_KNOBS` replaces the default 8K-token knobs, for a soak at a deployment's real shape. | Run a complete serving soak with host-level diagnostics. |
| [serve_prefix_curve_sweep.sh](serve_prefix_curve_sweep.sh) | Restarts the service for combinations of prefix-cache budget and conversation count, collecting a cache-capacity curve. | Choose a cache budget or compare eviction behavior. |
| [serve_width_sweep.sh](serve_width_sweep.sh) | Restarts for each sampling-candidate width and plain/MTP mode, sends seeded requests, and writes fallback counts, pace, and output hashes to `sweep.tsv`. | Measure sampling-width cost while checking that width changes preserve output. |

## Node and network diagnostics

| File | What it does | When to use it |
| --- | --- | --- |
| [node_probe.sh](node_probe.sh) | Samples local memory-pressure counters, GPU clocks/power/throttling, CPU utilization, and busy processes once per second. | Record host conditions alongside a latency or soak run; it runs until stopped. |
| [roce_counters.sh](roce_counters.sh) | Takes hardware-counter snapshots over SSH for the selected nodes, or delegates comparison of saved snapshots to `roce_report.py`. | Measure network traffic, retransmissions, congestion notifications, and discards around a run. |
| [soak.sh](soak.sh) | Repeatedly starts two-node `bus_check` sessions with concurrent bulk traffic and latency probes at rotating transfer sizes. | Stress collective transport independently of serving. Uses the first two site nodes; the peer binary must already be staged. |

## Offline analysis and shared helpers

These files work on saved artifacts or provide small functions to the other
scripts. They do not launch a cluster. `nsys_step_breakdown.py` may create or
refresh a SQLite export, and `width_sweep_collect.py` appends to its output TSV.

| File | What it does | When to use it |
| --- | --- | --- |
| [fabric_logs.py](fabric_logs.py) | Shared parsers for generation, teacher-forcing, and sampling-mass records in `rN.log`, plus a bf16 rounding helper. | Import it when adding fabric-log analysis; it is a library, not a report command. |
| [bench_stream.py](bench_stream.py) | Parses benchmark SSE streams and computes shared request-wall and output-span metrics. | Import it from serving benchmark clients that need consistent streaming measurements. |
| [fabric_xcript.py](fabric_xcript.py) | Compares two fabric transcripts and reports the first divergence with the global top-two logit margin. | Investigate whether changed greedy output occurs at a near-tied pick. |
| [fabric_logprob.py](fabric_logprob.py) | Joins teacher-forced rank logs to compute negative log-likelihood, perplexity, and top-1 accuracy, with optional reference thresholds. | Evaluate numerical changes on the same teacher text and prompt. |
| [qwen_head_compare.py](qwen_head_compare.py) | Joins vocabulary shards from paired `qwen_head_check` runs and enforces complete coverage, exact repeat controls, NLL budgets, and top-1 margins. | Validate the Qwen FP8 vocabulary-head dispatch on fixed teacher tokens; see [the procedure](../docs/numerics.md). |
| [fabric_sampling_profile.py](fabric_sampling_profile.py) | Validates cross-rank sampling-mass records and selects the smallest measured top-k width meeting a fallback-rate bound. | Size distributed sampling from complete `--sampling-profile` teacher runs. |
| [fabric_step_times.py](fabric_step_times.py) | Reads rank-0 generation timestamps, skips warmup, and reports mean and tail step times with any MTP summary. | Compare steady-state engine timing across saved fabric runs. |
| [bench_compare.py](bench_compare.py) | Compares matched `serve_load.py` JSON reports, checks exact C1 greedy transcripts, and reports median rate changes. | Gate a repeated baseline/candidate benchmark with an explicit slowdown tolerance. |
| [fabric_xrank.py](fabric_xrank.py) | Aligns rank logs by collective generation and reports step timing, host gaps, stalls, and lagging ranks; can correlate node probes. | Diagnose cross-rank latency imbalance. Record the run with `DGPP_LOG_LEVEL=debug`. |
| [bus_window_skew.py](bus_window_skew.py) | Summarizes per-rank collective copy, handshake, skew, and fold timings, wait histograms, and peer lag. | Analyze collected bus timelines; `DGPP_BUS_TIMELINE=1` records them at INFO level without full debug logging. |
| [nsys_step_breakdown.py](nsys_step_breakdown.py) | Exports a Nsight Systems report to SQLite and groups GPU kernel time by decode step or by the final kernel burst. | Find expensive kernels in a saved profile; use `--burst` for a prefill-oriented view. |
| [serve_pace.py](serve_pace.py) | Reads scheduler token timestamps to report per-request pace and estimated replays; `--waves` adds peak-occupancy measurements. Also exposes unformatted measurements. | Analyze a rank-0 serving log recorded at debug level, or reuse the measurements in another report. |
| [width_sweep_collect.py](width_sweep_collect.py) | Joins one sweep run's responses, slot-close fallback counts, and pace measurements into TSV rows. | Collect or rebuild results from a saved width/mode run; called by `serve_width_sweep.sh`. |
| [width_sweep_report.py](width_sweep_report.py) | Reads `sweep.tsv`, compares output hashes across widths, aggregates fallback rates and pace, and fits per-fallback cost. | Interpret a completed sampling-width sweep. |
| [forward_check_report.py](forward_check_report.py) | Checks digest agreement in `wWORLD_rR.log` files and merges rank-local argmax results, breaking ties by smallest token ID. | Re-read Qwen or GLM forward-check output; shared by both launchers. |
| [admission_report.py](admission_report.py) | Summarizes saved request outcomes, pool and admission metrics, and supplied batch start/end times. | Re-read one policy run from `serve_admission_check.sh`. |
| [load_phase_report.py](load_phase_report.py) | Matches timestamped concurrency phases in `load.log` to rank-0 stats in `world/serve_r0.log`. | Compare client load phases with server throughput measurements. |
| [node_probe_report.py](node_probe_report.py) | Totals memory-pressure counter deltas and lists observed throttle masks from one node-probe log. | Summarize host pressure after a soak or diagnostic run. |
| [prefill_report.py](prefill_report.py) | Builds repeated token-ID prompts or prints the first prefill timing found in a log. | Support the prefill wrappers or inspect saved prefill measurements. |
| [roce_report.py](roce_report.py) | Subtracts two counter snapshots and converts transmitted four-byte words to MiB. | Compare saved RoCE snapshots without loading site settings or contacting nodes. |
| [serve_streams.py](serve_streams.py) | Shared saved-SSE reader with commands for committed text, tool-call reconstruction, shutdown summaries, and post-error token checks. | Inspect captured streams; interrupted-text recovery tolerates truncated events while normal reports parse strictly. |
| [serve_response_report.py](serve_response_report.py) | Prints saved non-streamed replies, reasoning, tool calls, and the JSON/schema checks used by the tools procedure. | Inspect response artifacts from `serve_tools_check.sh`; schema verdicts are informational. |
| [failure_report.py](failure_report.py) | Builds drill request bodies, checks nonempty committed-text prefixes against replay, and compares surviving operation streams. | Reuse the failure drill's local checks without running the destructive procedure. |
| [run_helpers.py](run_helpers.py) | Provides localhost port checks, randomized short delays, and elapsed-time calculations. | Use the small CLI helpers called by fabric startup and failure-drill orchestration. |

The shared report modules can be imported without running their CLI. Several
older standalone clients and analysis scripts execute at module scope, so treat
those as commands rather than libraries.

For example, to re-read a saved stream without contacting the service:

```bash
python3 scripts/serve_streams.py summary /path/to/run/stream.sse
```

See [operations](../docs/operations.md) for deployment procedures,
[tools](../docs/tools.md) for the compiled binaries, and
[benchmarks](../docs/benchmarks.md) for recorded measurements and their workloads.
Offline regression tests for site configuration and the report helpers are in
[`tests/python/site_env_test.py`](../tests/python/site_env_test.py) and
[`tests/python/script_reports_test.py`](../tests/python/script_reports_test.py).
