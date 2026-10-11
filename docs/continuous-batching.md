# Towards continuous batching: multi-admission + fair-share prefill

Status: original items 1 and 2 implemented in `Scheduler::quantum()` and
soak-tested live. The size-neutral queued-text follow-up for issue #104 is
host-tested; real-model latency and MTP boundary validation remain separate
acceptance gates, not established by the original soak.

## Problem

Single-stream dgpp matches or beats vLLM, but per-request throughput collapses
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
  time; see item 2);
- the loop stops before grouping or admission when the tick's positive
  prefill budget is exhausted; zero remaining tokens must not be passed
  to the group planner, where zero means monolithic mode;
- with a zero policy budget, exactly one admission event (possibly a
  group) runs per tick, preserving the original monolithic policy;
- an oversized image on an engine without image chunking retains its
  monolithic fallback as the tick's only prefill admission. It waits if
  any prefill work already ran that tick, then consumes the whole cap;
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

For non-group-advance engines, queued text of any total prompt length can
join these shares. The original rule admitted only prompts classified as
needing chunking; a fitting shorter prompt could wait until the long prefill
finished because the align-down leftover was zero. The issue #104 follow-up
removes that size exclusion without adding a short-request priority or a
second budget. Image and group-advance eligibility, monolithic mode and
admission without an in-flight prefill retain their existing paths.

The new reader still needs a free slot and its reservation under the existing
KV policy. Cached text attaches through the same prefix plan and computes
only its suffix. A short tail does not donate its unused share to other
readers. Early decoding may switch later ticks to the smaller busy budget,
delaying long-prompt completion; this policy does not promise unchanged
completion latency or starvation freedom under sustained KV pressure.

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
- Live: c1/c2/c4 @ d0/d4k/d8k benchy sweep; TTFT of request #2 must drop;
  per-request tok/s must not regress at c1.
- Item 2 additionally: c4 deep-context soak, KV pool metrics sane, no
  `validate_batch` / digest mismatches in logs.
