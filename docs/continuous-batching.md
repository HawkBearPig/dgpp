# Continuous batching and shared prefill

Serving uses one scheduler and shared continuation/mixed-row components across
all eight executor families. Early text admission and latency-adaptive prefill
are enabled by default, including with DFlash. Current controls and validation
procedures are in [operations](operations.md) and [testing](testing.md).

The problem statement and initial rules below record the original
multi-admission work. The later shared execution and adaptive policy are
described under [current execution](#current-shared-execution-and-policy);
the original soak alone did not establish those later changes.

## Original problem

Single-stream dgpp sits at its step floor, but per-request throughput collapses
under concurrency (measured: 45 → 9 tok/s @d8k c4). Root causes in the
scheduler (`src/sched/scheduler.cpp:quantum()`, `1124-1254`):

1. **≤1 admission per tick** (`1161-1165`): request #2 queues behind request
   #1's full prefill — head-of-line blocking. vLLM admits into the running
   batch immediately.
2. **One unfinished prefill at a time**: while any request is `kPrefilling`,
   `admit_arrival` is forced to -1 and every queued request waits for the
   whole chunked prefill to finish.
3. Decode cost scales ~7 ms/GPU-row with batch width while each request still
   earns ~1 token/step (kernel problem, out of scope here).

The full vLLM-style fix (a request joining mid-decode, with graph
revalidation and MTP rescheduling) is deliberately out of scope: it would
touch `validate_batch` (`src/engine/graph_engine.hpp:1665-1684`), the verify
schedule, and the fabric lockstep digests. The two items below capture most
of the c2–c4 win without it.

## Item 1: multi-admission per tick (done)

**Rule**: after the cancel sweep and `grow_reservations()`, admit fitting
queued requests in a loop *before* the decode slice is built. Each iteration
admits via the same dispatch as before (group → one-shot → chunked-start),
except:

- a request needing chunked prefill **ends** the loop (starts chunking only
  if nothing was admitted yet this tick — chunked read-ins stay one at a
  time; see item 2).
- the loop stops before grouping or admission when the tick's positive
  prefill budget is exhausted; zero remaining tokens must not be passed
  to the group planner, where zero means monolithic mode.
- with a zero policy budget, exactly one admission event (possibly a
  group) runs per tick, preserving the original monolithic policy.
- an oversized image on an engine without image chunking retains its
  monolithic fallback as the tick's only prefill admission. It waits if
  any prefill work already ran that tick, then consumes the whole cap.
- `next_admissible()` returning -1 (no fitting request / no free slot) ends
  the loop as before.

**Why it preserves the invariants**:

- `step_batch` builds its round-robin slice *after* admission
  (`scheduler.cpp:1212-1234`); `validate_batch` only requires the replay to
  include every live slot, which holds trivially for slots admitted earlier
  in the same tick — each admit path calls `reserve()` in `admit_finish`.
- Every function in the loop (`next_admissible`, `admissible_group`,
  `admit`, `admit_group`, `begin/advance_prefill`) depends only on replicated
  scheduler state, so all ranks compute the same loop. The fabric journal
  already carries `submits` as a list (`generation_service.cpp:3416`); no
  journal format change.
- Precedent: `admit_group` already admits N≥2 cold prompts in one forward.

**Docs updated with it**: `src/sched/scheduler.hpp:8-9` (tick policy line).

## Item 2: fair-share chunking across queued prefills (done)

**Rule**: the `prefill_arrival >= 0 → admit_arrival = -1` gate is gone.
Each tick with a prefill in flight now: collects the in-flight prefills in
arrival order, begins at most one new read-in (oldest fitting,
skip-fit, no eviction dance — a begin must not disturb the pool the
in-flight prefills hold), gated so every share keeps at least one aligned
chunk (`(n+1)*align <= budget`); splits the tick budget into equal
align-down shares and advances the selected prefills on their shares; then
admits fitting one-shots/groups into the align-down leftover (never a new
chunked start — chunked read-ins stay one at a time). Order derives from
arrival order and the last-advanced prefill only, so every rank agrees.

## Current shared execution and policy

Serving enables early text admission by default when the engine supports mixed
prefill and decode execution: queued text of
any size can join, including prompts shorter than one alignment unit. New
readers share a bounded grouped prefill and share their dense and expert
projections with active decoder verification.
Each decoder retains its own recurrent state, speculative rollback rows,
MTP drafts and sampling state. It receives at most one decode pass per tick.

`engine.prefill_early_admission=false` restores admission through leftovers.
The launcher does not inspect model architecture or drafter names. It carries
an automatic/on/off setting through the settings handshake, resolves it after
engine construction, and publishes the concrete policy in the warm record.
Backends without mixed execution retain their previous default; explicitly
enabling admission does not implement shared execution. Images
retain their existing eligibility and execution paths. Monolithic mode is
unchanged.

The shared execution components have separate responsibilities:

| Component | Responsibility |
| --- | --- |
| `PrefillTask` | Owns the prompt, images, snapshot plan and cursor for either adapter. |
| `PrefillContinuation` | Tracks logical cuts, work budgets, progress and suspension across yields. |
| `PrefillBatch` | Plans mixed rows, bounds physical work, commits positions and snapshots, catches up draft state and slices outputs. |
| `RowRun` | Describes request spans, the decoder prefix and selected output rows for a layer walk. |
| `GraphBatchMapping` | Maps compact graph rows to persistent request slots for feeds, positions and speculative commits. |
| `SessionModel` | Owns session positions, cache reservations, rollback, draft history and graph staging. |
| Layer executor | Applies its attention, recurrence, projections and draft implementation to those rows. |

Legacy GLM uses the same task, continuation, batch and row components through
its existing session facade. Every serving executor implements the
`MixedPrefillModel` contract: Qwen4Exp, Qwen35 (including DFlash), GLM, GLM4,
GLM-DSA, MiMo, DeepSeek V4 and V4.1. A drafter's name is not an eligibility
rule. The adapters check actual slot, row-capacity, context, continuation
workspace and graph-variant limits per batch.

The configured prefill budget bounds reader work. Decoder rows use spare
physical capacity without reducing that reader budget. When their combined
rows exceed the executor's scratch capacity, the planner trims readers at
their required alignment; short tails contribute their unused shares first.

Persistent reader state belongs to its request. In particular, V4.1's bounded
decoder retains each reader's tail separately and falls back to sequential
reader advancement when their combined final tails exceed scratch capacity.
Per-tensor activation scales also stay within a request. Decoder prefixes keep
their decode attention and recurrence paths, including speculative snapshots;
the rest of the layer walk can share projections where the numerical contract
permits it. Unfinished readers do not project or mirror unused logits.

The new reader still needs a free slot and its reservation under the existing
KV policy. Cached text attaches through the same prefix plan and computes
only its suffix. With early admission enabled, a short tail keeps its rounded-up aligned allocation and
donates the rest of its share to unfinished readers, split equally in
aligned units with arrival-order ties. This happens before the advances,
uses replicated positions and stays inside the original tick budget. When
all readers finish, unused capacity does not create another admission lane.
These allocations are caps: required snapshot/structural cuts can end a GPU
advance earlier. Executors that declare split-invariant prefill spend the cap
from the cursor's current position and omit redundant automatic chunk-grid
cuts; other executors retain their chunk-grid contracts.
A lone reader, or a set of readers that all fit their original shares, keeps
those shares unchanged, including its final tails.
With fixed budgeting, the busy cap protects decoders that started before a
reader. When every active decoder joined after all unfinished readers, and the engine
can share their execution, those readers retain the configured idle budget.
This avoids multiplying their weight-streaming passes because a short request
was admitted. Newly admitted streams can consequently have longer update gaps
while sharing those larger chunks. The budget remains bounded; slot and KV
pressure can still delay admission. Required cuts and decode rows can leave
less than the full prefill allowance in a shared walk.

The default adaptive policy instead learns pass cost by decoder-load range
and by the context range a chunk starts in, so a long prompt's late, expensive
chunks never size the next short prompt's (an unmeasured range borrows the
most conservative rate at the same context, then the deepest shallower one).
It starts at 256 tokens, grows within the aligned engine/idle ceiling and
applies the configured decode-gap target to both incumbent and newly admitted
decoders. Explicit positive budgets remain caps. Rank 0 journals its selected
limit; peers replay it and never use local timing to schedule. Fresh idle
arrivals receive the full first-token allowance; an arrival queued during
ongoing prefill with a free slot reserves half for its previous wait. These
soft targets do not predict when a slot- or KV-blocked request can start.
While nothing decodes and nothing waits for a free slot, the selected quantum
never drops below `prefill_idle_floor_tokens` (1024): a lone reader keeps the
engine's efficient chunk and an arrival during it waits at most that chunk —
without the floor a 522K prompt on the 27B read its tail in 256-token chunks
and took 39 % longer than at 4096.
See [policy defaults and metrics](operations.md) for the cost/recovery controls.

For example, with a 256-token budget and four-token alignment, a 23-token
arrival beside one long reader gets 24 allocated tokens and the long reader
gets 232. An engine consuming those caps computes 255 tokens in total;
equal shares offer at most 151 in that tick. This saves scheduled capacity;
GPU throughput
still depends on chunk shapes, additional walks and active decode width.
Deployments that need the same fixed prefill cap before and after admission
can choose equal busy and idle budgets and set both latency targets to zero.
With adaptive targets enabled, equal caps still allow smaller selected quanta.
Larger quanta can increase gaps between decode updates, so measure mixed loads.

When the budget drops from idle to busy, the in-flight count can exceed
`budget / align`. Advance only that many prefills, rotating through arrival
order from the last-advanced request. Every share remains at least one
aligned chunk without exceeding the total budget. Record compaction remaps
the prefill cursor just like the decode cursor, including when its request
was cancelled or retired. With enough budget, all prefills advance again.

**Audit results** (the pre-implementation open questions, all green): the
engine already supports ≥2 concurrent `kPrefilling` requests — per-slot
`prefills_` cursors suspend/resume by design (`session_model` advance marks
unfinished slots inert `0xff`), `reseed_live_feeds` touches live slots
only. Grow-mode shedding can retire younger `kActive` or `kPrefilling`
requests at the tick's initial reservation pass; the in-flight list is
built after that pass. Each surviving prefill retains its reservation
between chunks.

## Test plan (both items)

- `unit_tests` + `serve_test` green (scheduler determinism tests compare
  concurrent requests against independent runs).
- Scheduler regressions cover exhausted group budgets, chunked starts
  after one-shots, monolithic image fallback, idle-to-busy budget changes,
  fair rotation, cancellation, slot reuse and compaction determinism.
- Queued-text regressions additionally cover alignment and busy/idle budget
  boundaries, oldest-fitting order, three/four concurrent requests, full
  output reservations, cache-attached short totals and histories, and
  joined-prefill cancellation and engine failures. A fake with MTP trailing
  rows checks reservation forwarding and unwind, not physical GPU safety.
- Tail redistribution regressions cover multiple donors and recipients,
  all readers finishing together, twenty consecutive short arrivals, and
  preservation of decode progress and the total allocated budget. Begin
  failures release every private snapshot, including the head cut, before
  cancellation or retry. These scheduler cases also run under ASan/UBSan.
- Live: c1/c2/c4 @ d0/d4k/d8k benchy sweep; TTFT of request #2 must drop;
  per-request tok/s must not regress at c1.
- Item 2 additionally: c4 deep-context soak, KV pool metrics sane, no
  `validate_batch` / digest mismatches in logs.

The reproducible admission comparison is
`python3 scripts/serve_admission_bench.py --url http://127.0.0.1:18081
--label COMMIT-CONFIG --out RESULTS.json`. Run matched baseline/candidate
processes in alternating order on reserved hardware, using the same prompts,
output limits and configuration. Compare equal busy/idle budgets as well as
256/2048 to expose the budget-transition trade-off. Its seven scenarios cover
isolated prefill/decode, a short arrival with short or long output, sustained
arrivals with and without an existing decoder, and cancellation/recovery.
The first arrival must be dispatched while the long response's matching
native prefill ID is still present. Requests, usage, update times, output
hashes and before/after metrics are retained; unrelated traffic invalidates
a run. Aggregate rates include the full scenario wall time and client update
gaps are not per-token kernel latency. Cached conversations, image paths,
physical KV boundaries, semantic checks and multi-rank operation-stream
agreement remain separate live gates. Host passes do not establish GPU
throughput equivalence.

### Mixed-prefill controller investigation

[Issue #116](https://github.com/HawkBearPig/dgpp/issues/116) reports a 64K MiMo
prompt taking about 4.5 times longer beside short requests with the default
controller than with fixed budgets. The idle floor does not apply while any
request decodes. The estimate divides a whole pass's cost, including decode and
snapshot work, by computed prefill tokens. Fixed costs and less efficient small
chunks can therefore keep the learned limit small. The report does not establish
how much decoder responsiveness those smaller chunks buy; changing the default
requires both throughput and decode-gap measurements on this deployment.

On an available, otherwise idle two-Spark MiMo deployment, hold the model revision,
TP size, MTP settings, four slots and 262,144-token pool fixed. Use full admission
and the benchmark's cache-disabled requests to isolate this controller question
from prefix-cache memory pressure. Compare decode-gap/first-token targets of
1000/3000, 2000/3000, 3000/3000 and 0/0 ms with the same resolved busy/idle caps.
Restart between configurations, alternate run order, and record cold-controller
and warmed long-prompt results separately. The benchmark's automatic warmup is
only a short decode; use a separate long-prompt run for the warmed comparison.

For example, on each explicitly configured server:

```bash
python3 scripts/serve_admission_bench.py \
  --url http://127.0.0.1:18081 --label COMMIT-MIMO-TP2-TARGETS \
  --scenarios solo_prefill sustained_idle sustained_busy \
  --prompt-words 49152 --decode-tokens 16384 --arrival-tokens 2048 \
  --arrivals 8 --interval 0.25 --metrics-interval 0.25 \
  --trials 3 --timeout 1800 --out artifacts/mimo-admission.json
```

The word count is an initial estimate, not a 64K token guarantee: calibrate it
against recorded prompt usage, then keep the exact prompts and output caps
identical across comparisons. Use the reporter's scripts too when available;
these sustained-arrival scenarios are controlled probes, not an exact replay of
the simultaneous OpenCode workload. Keep decoders live throughout the long
prefill and check that condition in the sampled native progress/load trace.

Compare long-request first-token/wall time, short-request latency, per-request
SSE update-gap p95/max, aggregate output and prompt throughput, and target-miss
counter deltas. `metrics_samples` associates the selected adaptive limit and
computed/offered chunk counters with native reader IDs and elapsed time.
It is sampled evidence, not one record per engine pass; before/after snapshots
and final usage remain the workload accounting checks. Add a sample's `elapsed_s`
to the run's `start` to compare it with request and SSE timestamps; these use the
same monotonic clock. Both targets zero selects
fixed budgeting; increasing the idle floor alone cannot affect the mixed case.

Validate grow-mode cache reclamation separately with cached conversation
histories and the 393,216-token pool from the report. Fill the pool with idle
histories, then force a live request to grow while its full lifetime fits the
pool. Require cache evictions before any shed, complete output with no shed when
reclamation suffices, unchanged attached histories, and matching rank operation
streams. Also run genuine live oversubscription to retain youngest-first shedding.
The host regressions establish those decisions; hardware checks additionally
exercise real MiMo snapshot lifetimes, cross-rank execution, and throughput.

Measured cache-reclamation results and controller comparisons are recorded in
[the issue #116 investigation](experiments/issue116-admission.md), including
the workload differences and limits of those checks.

**Resolved 2026-10-22.** A second field report (one Spark, Qwen3.8-Flash-Next
NVFP4, five opencode agents, 2026-10-21) measured the same effect at the
rate level: under five active and four reading requests the adaptive limit
read 128 tokens — one amount per pass shared by the readers, about 32 each —
and prompt cost was 1.9–2.4 ms a token against 0.78 idle; fixed 1,024/2,048
budgets cut it to 1.01–1.08, the five-agent first-token p90 from 58–60 s to
12–15 s, and aggregate generation rose from 36–46 to 52–56 tok/s, while the
decoders running beside a probe's prefill went from 9.0 to 11.5 tok/s: the
small chunks bought no decode responsiveness at the rate level (inter-token
gaps were not recorded). The mechanism is the one named above, made exact:
the controller charged the whole tick — the decoders' pass included — to the
chunk's tokens. Growth is bounded at twice the last quantum, so a quantum
grows only when its pass is under half the allowance; with decoders live,
every quantum whose pass ran between 42.5 % and 85 % of the target was
therefore an absorbing state, and one slow pass (a snapshot cut, a cold
GEMM shape, a page-fault burst) walked the limit down the ladder for good.
The controller now learns the decoders' own pass from ticks that only step,
per decoder-load range, and charges a chunk only for what it adds; the target
less that pass is the room for tokens (`PrefillLatencyController::observe_decode`,
`scheduler.prefill_chunks.decode_pass_ms`). The per-reader overhead of a
walk is still charged to the tokens, which keeps a far smaller band: the
chunk's own cost must stay under half the room, and it does once the chunk
is a few hundred tokens. A lone reader is charged no decoders' pass and keeps
its idle floor.

Measured the same day on one Spark with the one-node Qwen3.8-Flash-Next NVFP4
template, 39K-token cold prompts, the binary before the change against the
one after it (two trials each, `scripts/serve_admission_bench.py` and a
four-reader probe). On the template as shipped (four slots, MTP depth 1,
bf16 K/V) nothing moved: the prompt's first token 27.2–27.4 s beside one
decoder before, 27.4–27.6 s after, the limit at 1024 both times, the decoder
at 38 tok/s; three 13K prompts reading together beside one decoder 29.3–33.8 s
before (limit 256–1024) and 26.3 s after (1024 throughout). On the field's shape (five slots, MTP depth 3, FP8 K/V, a bigger
decoders' pass) the ratchet appeared and went away: four 39K prompts reading
together beside one decoder took 252–267 s to their first tokens with the
limit at 128–512 (median 256) and three decode-gap misses; after the change
158–159 s, the limit at 512–1024 (median 512), no misses. One 39K prompt
beside one decoder went from 39.8–45.7 s to 32.6–38.2 s (limit median 512 →
1024, the decoder 45.8 → 49.6–50.7 tok/s), beside four decoders from
44.6–44.7 s to 39.0–39.5 s (aggregate 72–73 → 79 tok/s, decode-gap misses
4 → 0). Alone, 25.8–27.2 s both. The decoder beside four readers gets one
verify per pass either way, so its own rate during their prefill is set by
the pass length (1.7–2.0 tok/s here): the shared walk's trade, bounded by the
decode-gap target, not something the controller can remove.

### Shared execution timing

`mixed_steps` counts decode passes that also advance prefill. Their elapsed
`mixed_ms` is included in both `prefill_ms` and `step_ms`, because both logical
operations occupy that interval. Total engine time is therefore
`prefill_ms + step_ms - mixed_ms`; phase times are not independent throughput
measurements. Prometheus exposes the same overlap as `dgpp_mixed_seconds_total`
and `dgpp_mixed_steps_total`, including per-rank equivalents. Compare completed
work over wall time when evaluating mixed workloads.
