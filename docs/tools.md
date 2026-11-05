# Tools, binaries and scripts

The main serving, diagnostic and validation commands are listed below.
The server is `dgpp-serve`, launched by `scripts/dgpp-cluster`. `tools/` holds the Python
checkpoint and reference tooling the tests use; `scripts/` holds the fabric
and serving operations. `docs/benchmarks.md` says which of these produced
each published number, and how to run them in the right order.

The Bash scripts handle process launches, SSH, requests and log collection.
Python parsing and reporting live in separate modules in `scripts/`, each with
a command-line entry point. They can also be imported without running a report.
Shared helpers include:

- `site_env.py`: shared site settings from `.env` and deployment resolution;
  Bash callers use `cluster_env.sh`.
- `serve_streams.py`: saved SSE events, committed text, tool-call reconstruction
  and shutdown summaries.
- `forward_check_report.py`: the cross-rank digest and argmax checks used by both
  the Qwen and GLM forward-check scripts.
- `serve_pace.py`: per-request pace measurements, used directly by
  `width_sweep_collect.py` as well as its own CLI.

The remaining report modules cover admission, load phases, prefill, node probes,
RoCE counters and failure drills. To re-read saved results without launching a
cluster, for example, run `python3 scripts/serve_streams.py summary STREAM.sse`
or `python3 scripts/forward_check_report.py LOG_DIR 4`. Use `--help` on the report
modules to see their arguments. Their offline regression tests run with
`python3 tests/python/script_reports_test.py` or CTest's `script_reports_test`.

| command | purpose |
|---|---|
| `dgppctl info` | CUDA and platform facts |
| `micro_mem_bw -s 4096` | LPDDR5x streaming patterns |
| `micro_gemm_peak` | best-of-cuBLASLt-heuristics FP8/BF16 shape sweep |
| `micro_zerocopy` | pinned/device GPU bandwidth, contention proxies, flag latency |
| `micro_gdr_probe` | informational direct-device MR probe and copy profile |
| `micro_ibv_smoke info` | enumerate verbs devices |
| `micro_ibv_smoke serve PORT --once --dev DEV` | one cross-node responder session |
| `micro_ibv_smoke ping --peer IP:PORT --dev DEV` | signaled RC SEND echo latency |
| `micro_ibv_smoke bw --peer IP:PORT --dev DEV` | signaled RC SEND bandwidth; repeat peer/device for both lanes |
| `micro_ibv_smoke verify --peer IP:PORT --dev DEV` | ordered NIC DMA payload/doorbell → GPU hash validation |
| `kda_bench` | KDA decode state-traffic and layer timing profile (M2) |
| `dsa_bench [--tp N]` | the DSA layer's decode kernels at a chosen context and TP width |
| `dsa_select_bench --ctx a,b,c [--rows R --grid G]` | the decode index selection alone, with its phase breakdown (the 2026-09-06 select rewrite's microbench) |
| `moe_slot_bench --rows R --inter I` | the MoE decode slot's expert GEMV at a chosen row count (the down kernel's row batching) |
| `gpt_doll --selftest` | synthetic eager/graph parity testbed |
| `glm_bind_check --config CONFIG --checkpoint-dir DIR` | validate a real GLM-5.3 checkpoint against the expected-tensor table (config parse, names, dtypes, shapes, FP8 scale pairing; headers only) |
| `glm_stream_check --config CONFIG --checkpoint-dir DIR` | stream real layers through the resident loader; bytes vs formula reconciled per layer |
| `glm_forward_check --config CONFIG --checkpoint-dir DIR --suite FILE` | curated reference suite: ISOLATED per-layer parity vs torch-reference dumps, head + routing agreement (DESIGN §7.5) |
| `glm_forward_check ... --trace-ids-file F --trace-out T` | engine-only real route-trace capture (deterministic) for the traffic model |
| `qwen_load_check` / `glm4_load_check --model ORG/NAME [--world W --rank R] [--streaming] [--mtp]` | load one rank's slice of a Qwen3.8-Flash-Next / GLM-4.7 checkpoint through the shared resident loader: per-layer bytes and times against the formulas, the replicated digest |
| `qwen_forward_check` / `glm4_forward_check --model ORG/NAME --ids 1,2,... [--world W --rank R --peer HOST]` | the world-1 (or fabric) diagnostic forward on a real checkpoint: per-layer residual digests, every position's top-k |
| `qwen_head_check --model ORG/NAME --requests FILE --fp8-head gemv\|mma [--decode-capacity 8\|16]` | repeated teacher-forced Qwen FP8-head scores through captured verification graphs and short prefills; compare complete GEMV/MMA runs with `scripts/qwen_head_compare.py` as described in [numerical checks](numerics.md) |
| `tools/qwen_reference_dump.py` / `tools/glm4_reference_dump.py gen-pure --checkpoint-dir DIR --out FILE` | the pure-python double references of the two families' forwards (and their draft blocks) the parity gates read |
| `tools/glm4_torch_reference.py --checkpoint-dir DIR --ids ... --layers N --engine-dump STATES` | the independent reference: transformers' own Glm4Moe layer code (torch, CPU) over the real checkpoint's first N layers against `glm4_forward_check --dump-states` — the gate that caught the RoPE pairing the python reference shared with the engine |
| `tools/route_trace_traffic.py TRACE` | route-trace traffic model: measured busiest-rank occupancy and corrected critical path (replaces the uniform-expert assumption) |
| `tools/checkpoint_audit.py [MODEL_DIR]` | regenerate inventory and checkpoint budget |
| `tools/make_shardspec.py MODEL_DIR` | generate the loader shard plan |
| `tools/kda_reference_dump.py` | KDA parity dumps: `selftest`, `gen-pure` (CI oracle), `gen-torch` (real checkpoint slices; needs torch) |
| `tools/dsa_reference_dump.py` | DSA parity dumps: `selftest`, `gen-pure` (CI oracle), `gen-torch` (real checkpoint slices; needs torch) |
| `tools/glm_reference_dump.py` | full-model parity dumps: `selftest`, `gen-pure` (CI oracle, mini checkpoint), `gen-torch` (real checkpoint, per-layer streams + router scores; `--layers N` for reduced budgets) |
| `roster_check coordinator/rank/selftest` | M5 control plane on real nodes: epoch-based roster startup, rank health, eviction on death/deadline; `selftest` is the loopback in-process smoke |
| `bus_check serve/ping/selftest` | M5 data plane on real nodes: RC/RoCE CollectiveBus — per-class slot pools, credit-grant RDMA writes, dual-lane striping, latency-under-bulk contention; `selftest` is the loopback in-process smoke; `--contend --soak-ms` is the duration-bounded soak |
| `bus_small_repro --pick-race` | the pick-path racer: the decode step's collective shape with an exact oracle after every collective under injected skew, loopback or fabric |
| `nic_regress mesh/pair/serve/selftest` | NIC→GPU visibility regression over every directed node pair on both lanes (rerun after driver/firmware changes) |
| `glm_tp_check --model ID --world W --rank R --peer HEAD` | fabric TP parity: final hidden/logits/routes/digests bitwise across ranks and against the loopback verdict dumps |
| `glm_shard_parity` | the sharded loader vs full-load + `GlmTpViews::bind`, bitwise on every bound surface |
| `glm_gen_check --model ID --chat TEXT [--system S] --steps N [--decode-graph] [--mtp]` | generation on the fabric (or `--text`, `--prompt IDS`): resident TP, distributed greedy pick, EOS stop; `--decode-graph` replays the step as one CUDA graph, `--mtp` adds speculative decode (`docs/mtp.md`); `--requests FILE` runs a JSONL manifest through the scheduler, `--sched-plan` prints the memory receipt without a GPU, `--teacher-file F` scores a text instead of generating, `--sampling-profile` adds the exact top-k-mass sizing probe, `--sample` (or any of `--temperature --top-p --top-k --min-p --repetition-penalty`, with `--seed N`) samples exactly at the checkpoint's defaults on the eager engines (with `--mtp`: the eager exact speculative sampler) instead of the default greedy loop, and `--step-timing` prints the per-phase budget |
| `dgpp-serve --model ID --port P [--world W --rank R --peer HEAD --journal-port J] [--max-concurrency N --kv-capacity T --queue-limit Q] [--decode-graph [--mtp] --graph-batch-min-live N] [--temperature X --top-p X --top-k N --min-p X --repetition-penalty X --seed N] [--sampling-candidates N] [--admission full|grow --admission-window N] [--reasoning-in-content] [--default-chat-template-kwargs JSON]` | the OpenAI-compatible service: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics` (JSON counters, also at `/v1/metrics`); rank 0 is the HTTP ingress and journals admissions to the peers; sampling defaults come from the checkpoint's `generation_config.json` (temperature 1.0 / top_p 0.95 for GLM-5.3-Flash-FP8) with the flags overriding them per process, requests may set `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `seed`, `top_k`, `min_p`, `repetition_penalty`, `logprobs` and `top_logprobs` (exact, the OpenAI shapes), and `/v1/models` reports the effective defaults; chat requests may carry `tools`, `tool_choice` (`auto`/`none`/`required`/a named function) and `parallel_tool_calls` — every call is well-formed by constrained decoding, a grammar mask on the pick applied identically on every rank (DESIGN §10): a declared function name, the argument keys the schema declares — every key closed to its `properties` whenever the schema declares any and does not opt out with an explicit `additionalProperties: true`, which is noted once naming the tool, and never the same key twice — argument values typed by their property schema (JSON-typed properties under the JSON machine — numeric `minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum` enforced digit by digit — enum strings from their texts, plain strings free; `function.strict: true` refuses a property outside the enforceable subset by keyword, a non-strict one under a keyword that only narrows its value without an automaton behind it — `pattern`, `format`, ... — keeps its type with one INFO line naming the keyword and the reason), and required, named and single-call turns as guarantees for closed keys (with free keys, repeated names make the parser flush the block as content, so required/named choices are best-effort) — `response_format` (`text` / `json_object` / `json_schema` with `name`, `schema`, `strict`: the content is one JSON text conforming to the schema's `type`, `properties` / `required` / `additionalProperties`, `items` / `minItems` / `maxItems`, `enum` / `const`, `anyOf`, and numeric `minimum` / `maximum` / `exclusiveMinimum` / `exclusiveMaximum`, enforced by the same masks; a strict schema outside that subset is refused naming the keyword, a non-strict one falls back to `json_object`; combinable with `tools`: auto chooses calls or a JSON answer, none forces JSON, required/named still require calls), `reasoning_effort`, `chat_template_kwargs` (native history controls `preserve_thinking`, `clear_thinking`, `drop_thinking`, `truncate_history_thinking` and `preserve_reasoning`, and `reasoning_effort`), assistant `tool_calls` and `tool` messages, all rendered through the checkpoint's template, and the response carries `reasoning_content` (or, with `--reasoning-in-content`, the reasoning folded into `content`), `content` and `tool_calls` parsed from the token ids on rank 0 with `finish_reason: "tool_calls"` (DESIGN §11); the eager engines and the `--decode-graph` graphs, with or without `--mtp`, sample exactly (DESIGN §10, the on-device verdict with the gather fallback between windows; under MTP the exact speculative accept test, DESIGN §9); graph mode requires the fabric and `max-concurrency * (mtp ? 2 : 1) <= 8`, warm-captures every graph variant at startup (the journal's `warm` record starts it on every rank together), and batches at `--graph-batch-min-live` live requests (default min(4, max-concurrency); must be in [1, max-concurrency]) |
| `scripts/fabric_run.sh [--stage-file F] [--fetch-logs] [--node-probe] -- APP-ARGS` | launches any app on the four-node fabric with the rendezvous discipline (head first, peers fire-and-forget, verified by pgrep, swept on head death); collects rank-invariant md5s and bus stats |
| `scripts/serve_run.sh up/down/status` | boots/stops the serving world (`DGPP_SERVE_KNOBS` overrides the engine flags on every rank, e.g. `--max-concurrency 1 ... --decode-graph --mtp`); `down` fetches and md5s every rank's op stream |
| `scripts/serve_api_check.py HOST PORT` (the request fields against a running world), `scripts/serve_soak_run.sh MINUTES OUT`, `scripts/serve_failure_drill.sh VICTIM`, `scripts/serve_prefix_curve_sweep.sh`, `scripts/fabric_prefill_repeat.sh`, `scripts/fabric_mtp_classes.sh` | the M9 evidence procedures: the mixed-workload soak with node probes and the four-way md5, the kill −9 drill, the prefix cache's capacity curve, the steady-state prefill per length, MTP acceptance per prompt class (`docs/operations.md`) |
| `scripts/serve_nvme_cache_check.py --port P`, `scripts/serve_nvme_cache_fuzz.py --config FILE --port P --requests N [--clients K --cancel-pct P --restart-every N --crash-every N]` | the NVMe cold tier against a booted world. The check drives one document through a cold prefill, a changed question, eviction by other documents and a restore from disk, reporting each step's first-token latency and whether the restored answer equals the in-memory one. The fuzz drives a random workload with pins, opt-outs, concurrent clients, mid-flight cancels, restarts and rank kills, checking every repeated answer and the tier's counters |
| `scripts/roce_counters.sh snapshot\|diff` | the fabric's RoCE hardware counters (sequence errors, adaptive retransmissions, CNPs, NIC ingress discards) per node and device, and the deltas between two snapshots — the wire-side view of a collective run |
| `scripts/serve_load.py HOST PORT [--concurrency 1,2,4] [--max-tokens N]`, `scripts/fabric_glm4_load.sh CONFIG OUT [--concurrency ...] [--tokens N]` | the fixed-concurrency decode reading (2026-09-10): c streaming long-answer requests at once per phase (thinking off, greedy), the client's per-request pace and aggregate tokens/s, and rank 0's stats lines inside each phase (the procedure boots the world with a 5-second stats line) — the batched depth-2 study's table |
| `scripts/serve_bench.py HOST PORT MAX_TOKENS LABEL [PROMPT]`, `scripts/serve_pace.py RANK_LOG [--waves]` | the service's pace: client-side SSE stamps; server-side per-request pace (from the per-token lines: run the service with `DGPP_LOG_LEVEL=debug`); and, with `--waves`, the steady peak-occupancy replay latency, tokens/replay, and aggregate tok/s used by the Phase-2 gate |
| `scripts/fabric_xcript.py`, `scripts/fabric_logprob.py`, `scripts/fabric_sampling_profile.py`, `scripts/fabric_xrank.py` | the judges (first divergence by bf16-ulp margin; teacher-forced perplexity delta; sampling-width evidence) and the cross-rank step/stall reader — `docs/numerics.md` |
| `scripts/step_probe.py [--max-tokens N] short|short+tools|long|ctx:N ...` | the decode-step probe against a running service: ttft, client ms/token, the engine's ms/step and tokens per step, the admission count, per mode |
| `scripts/mtp_depth_check.py --out DIR` | the greedy transcript of three prompts (prose, code, JSON) from a running service into DIR, for `diff -r` across launches (plain / MTP depth 1 / depth 2 / pipelined must be byte-identical) |
| `scripts/bus_window_skew.py LOG...` | the per-rank collective budget (copy / handshake / skew / fold) from the bus's window timeline lines (`DGPP_BUS_TIMELINE=1`) |

The GDR probe exits successfully when the probe itself completes, including
the expected “unsupported” result on GB10. It does not prescribe a bounce
copy; CollectiveBus uses registered pinned memory consumed directly by the
GPU.

Chat `reasoning_effort` accepts `minimal`, `low`, `medium`, `high` and
`xhigh`, either at the top level or in `chat_template_kwargs`. The Qwen
frontend maps minimal to low and high to xhigh; low, medium and xhigh pass
through. Qwen's checkpoint template adds short/extended thinking instructions
for low/xhigh, while medium uses its neutral thinking prompt. Other model
families receive the requested name unchanged and apply their own template.

History controls use native template kwargs, as in vLLM. For example, Qwen's
`chat_template_kwargs.preserve_thinking` is passed unchanged to Qwen's template;
it is not translated into another template's `clear_thinking` or other name.
`preserve_reasoning` is also passed through unchanged, without alias expansion.
The template decides which keys it reads and what they mean.

`--default-chat-template-kwargs '{"preserve_thinking":true}'` sets a server
default. A request's `chat_template_kwargs.preserve_thinking:false` overrides
that same key; differently named keys remain independent. Without a server or
request value, the checkpoint's own default applies. Top-level
`preserveThinking`, `preserve_thinking`, and `preserve_reasoning` are ignored,
even when template kwargs are absent. History kwargs must be booleans; defaults
are validated at startup and advertised in `/v1/models`. See
[API compatibility](openai-compatibility.md) for details and validation rules.

Tool key closure applies to top-level arguments only. `properties: {}` closes
to an empty set; a schema-valued `additionalProperties` also closes the declared
set in non-strict mode. Nested objects and `response_format: json_schema` keep
the JSON machine's open default. See [DESIGN §10](../DESIGN.md) for the policy.

MiMo compact XML calls preserve required top-level arguments even without
`strict: true`. The grammar consumes all bytes of tokens that span XML
fields, so a merged delimiter cannot bypass the next function/key/value
constraint. Plain string values remain free text, including empty strings
when their schema permits them.
Tokens that decode to no text leave the grammar state unchanged. Argument
names must remain distinct even when `additionalProperties: true` allows
undeclared names. Truncated or malformed calls still return as literal
content; reaching the generation limit retains `finish_reason: length`.

MiniMax M3.1 calls use a namespace before every recursive XML tag. Nested
objects, arrays and empty containers retain their schema types, including
numeric-looking strings. All invokes in a block are returned together;
malformed or truncated blocks remain literal content. Its grammar enforces
required arguments, typed values and distinct keys, and supports adaptive
reasoning before a call or JSON answer. The template omits null fields.
Required schemas without a finite non-null XML spelling, `anyOf` with
multiple non-null alternatives, and container `enum`/`const` are refused
with the offending field named. Nullable single-alternative schemas and
scalar enums are supported; JSON response schemas keep the normal JSON
grammar's capabilities.
