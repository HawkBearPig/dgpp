# Performance improvement plan from the DGX Spark comparison

Analysis date: 2026-09-15. Code reviewed at `3c537b6`.
Baseline: [dgpp-benchmark-comparison.md](../dgpp-benchmark-comparison.md),
the dated measurements, deployment templates, and the implementation.
The analysis below describes that baseline. Implementation now includes
the measurement tools, Qwen's 16-row decode path, fitting batch families,
and opt-in Qwen prefill continuation. See the [implementation report](../benchmarks/results/2026-09-15-qwen-batching-prefill.md)
for correctness gates and fabric measurements. Sections 1–9 retain the
original proposal; the follow-ups below and section 10 update delivery
status and remaining priorities.
The [C1 analysis](../benchmarks/results/2026-09-15-c1-analysis.md) puts the
default Qwen shift at about -0.25%, below this campaign's between-session
variation; the zero-threshold screen does not establish a material causal
regression. Section 10 updates priorities using the implementation results.
The [idle-prefill follow-up](../benchmarks/results/2026-09-15-qwen-idle-prefill.md)
adds a larger budget when no request is decoding. It reduces the measured
32K idle prefill from 40.0 to 24.7 seconds versus fixed 256-token chunks,
and the short-peer transition workload from 29.8 to 20.3 seconds. Qwen C1
shifts fit the measured unchanged-build variation; both budgets remain opt-in.
The [expert-kernel investigation](../benchmarks/results/2026-09-15-qwen-moe-prefill.md)
rejects smaller tiles, compact grids and persistent blocks for Qwen's small
prefill chunks. None established a useful service win. Recorded-route memory
counters are close to one read of the selected weights; production kernels
remain unchanged. Section 10 moves full GLM's missing packed prefill GEMM
ahead of further Qwen tile sweeps. The subsequent
[full-GLM implementation](../benchmarks/results/2026-09-15-glm-packed-prefill.md)
adds packed tensor-core prefill and independent FP64/likelihood checks.
It retains short-prompt GEMV because an earlier switch changed MTP acceptance
and reduced measured C1 throughput, despite passing the likelihood gates.
Section 10 lists the remaining priorities.

## Recommendation

Prioritize **mixed-workload scheduling, useful batch width, and prefill
kernels**, while prototyping **GLM-5.3-Flash DFlash2** as a separate,
checkpoint-backed project. Preserve the efficient one-request paths.

The most actionable gaps are:

1. Long prompt admission still stops all decoding. Group prefill improves
   bursts but does not solve this latency problem.
2. GLM-Flash and Qwen cannot currently batch more than four requests at
   MTP depth 1. Qwen is closer to wider batching than GLM-Flash.
3. Full GLM-5.3 still sends packed int4/int8 prefill through GEMV chunks.
   Other families need service-path profiles and better bulk collectives.
4. Speculation spends both draft work and target verification rows. The
   existing scheduler adapts verification, but cannot turn speculation off
   under load or assign different depths within one batch.
5. Some single-stream gaps compare different weight formats and memory
   placement. Establish those deployment variants before attributing the
   entire gap to kernels or drafting.

## 1. What the comparison establishes—and what has already changed

Use forum results to choose experiments, with matched workloads deciding
whether a change succeeds. Different concurrency, context, quantization,
thinking settings, prompt classes, and timing definitions are material.
The original forum pages were not retrievable during this review; their
figures remain attributed to the supplied comparison. Primary drafter
sources and deployment authors' repositories were checked separately.

Several statements in the comparison and older planning documents need
qualification against current code:

| Area | Current evidence | Planning consequence |
|---|---|---|
| DSpark | Already implemented in [Dsv41Model](../src/models/dsv41/model.hpp), including the block backbone, sequential correction and confidence output; [verify_schedule.hpp](../src/engine/verify_schedule.hpp) supplies adaptive verification | Improve the existing DSpark path; do not schedule a new DSpark port for DeepSeek |
| Group prefill | `Scheduler::admissible_group` and `admit_group` exist; all five serving families expose span support | Extend grouping to resumable work and cache-aware admissions; do not rebuild cold-prompt grouping |
| Dense batched kernels | September 14 changes select cuBLASLt or streaming tensor-core kernels above small row counts | Reprofile the current build before repeating the earlier dense-GEMV optimization |
| Qwen batch support | Runtime-sized session state and `kBatchedDraftChain = true` already exist; the serving cap is eight rows and some leaf kernels still enforce it | A bounded widening project, not a fresh batched-MTP implementation |
| GLM-Flash batch support | Its separate model fixes scratch at `kDecodeRows = 8` and declares `kBatchedDraftChain = false` | Requires state/scratch generalization plus a batched chain |
| Other families | GLM-4.7 supports up to 32 rows; full GLM supports 16; DeepSeek supports 32 | Run capacity/depth experiments on these paths now, subject to memory and graph limits |
| DeepSeek C6 | [Measurements](measurements.md) record 106.4 tok/s after group prefill, versus 131.86 from the recipe's own benchmark | The closest recorded target requires about 24% improvement; the forum's 214 figure is a separate reproduction target |

The four-slot restriction is conditional: eight plain-decode requests fit
an eight-row model, whereas four requests with one draft consume all eight
rows. More concurrency also increases the union of routed experts, cache
traffic and communication. It measures more than occupancy policy alone.

## 2. P0: establish a trustworthy performance baseline

### Concrete measurement fixes

Extend [serve_load.py](../scripts/serve_load.py) and
[serve_prefill_probe.py](../scripts/serve_prefill_probe.py), retaining the
historical output fields under explicit legacy names:

- `serve_load.phase` divides total completion tokens by the interval from
  the earliest first output to the latest last output. That interval
  includes later admissions' prefills and a changing number of live
  requests. Publish full request-wall throughput and a separately defined
  steady-state decode measurement.
- Its per-request "ms/token" uses the number of SSE content chunks.
  Chunks are not guaranteed to be individual tokens, especially with
  speculative bursts. Require usage/token accounting for token rates;
  report SSE inter-update latency separately. Obtain true token/commit
  timing from the server where necessary.
- Class prompt lists currently cycle modulo their length. Add at least
  sixteen distinct prompts per class for C6/C8/C12/C16, keep the existing
  C1 anchors, and expose cache-hit/cold status.
- The prefill probe uses `read(4096)` and treats any `choices` event as a
  first token. Use incremental reads and the first content/reasoning
  output; record server first-token-ready time independently.
- Prefill metrics currently include slot opening, picks and initial drafts
  inside `GraphEngine::open_slot`. Label that scope. Also, grouped
  admission credits the group's entire wall time to each request through
  `admit_finish`; summed request wait is not elapsed GPU prefill time.
  Track group execution time once and per-request wait separately.

### Baseline matrix

Start with Qwen TP2, GLM-Flash hybrid TP2/TP4, DeepSeek TP4, and full GLM
TP4. Add the remaining deployment lanes for regression checks.

| Dimension | Measurements |
|---|---|
| Concurrency | 1, 2, 4, 6, 8, then 12/16 where supported; record actual live slots and physical rows |
| Prompt length | 512, 2K, 8K, 32K; selected 128K runs after memory checks |
| Workload | Five existing classes; simultaneous bursts; sustained arrivals; a long prompt arriving during ongoing decode |
| Speculation | Plain, depth 1, deeper fixed depths, adaptive depth; greedy and sampled separately |
| Cache | Cold, warm weights with cold prefix cache, prefix hits; resident versus mmap tables |
| Report | Three-run medians and spread for throughput; enough requests for p50/p95/p99 TTFT and inter-update latency; failures, memory, clocks, acceptance and committed tokens/pass |

Pin commit, checkpoint revisions, quantization, templates/token IDs,
thinking settings, output lengths, cache policy and node count. Preserve
the existing five-class baseline and add the DeepSeek recipe's exact
client/prompts as a second named workload. Reject unhealthy-clock runs
and record them rather than silently selecting the best run.

For the service/procedure discrepancy, feed the **same rendered token IDs**
to both paths and separately vary prefix snapshots, MTP initialization,
chunk boundaries and cache warmth. Profile per-rank CPU waits, kernels,
route occupancy, memory traffic and collectives. The discrepancy is inside
the measured engine call too; HTTP overhead alone does not explain it.

**Exit:** a reproducible current baseline with timing scopes that remain
valid after chunk interleaving and group admission.

## 3. P1: resumable prefill with a budget per scheduler tick

### Code basis

[`Scheduler::quantum`](../src/sched/scheduler.cpp) calls `admit` or
`admit_group`, waits for the whole prefill, then executes decode.
[`SchedulerEngine`](../src/sched/scheduler.hpp) returns a first token from
`prefill`; it cannot represent partial progress.
[`SessionModel::session_prefill_chunks`](../src/engine/session_model.hpp)
already executes cuts, but owns the entire loop within one call.
GLM-Flash has an analogous separate implementation in
[decode.cpp](../src/models/glm/decode.cpp).

### Implementation

1. Add a `Prefilling` request state and engine operations equivalent to
   `begin_prefill`, `advance_prefill` and `finish_prefill`. A partial chunk
   returns progress, not a generated token. Factor the existing monolithic
   operations through the same implementation.
2. Give each in-progress request a persistent cursor: prompt end, next cut,
   cache attach/snapshot ownership, target/draft positions and accumulated
   execution time. Reserve/account for its resources before yielding;
   today the final reservation occurs after prefill completes.
3. Use a deterministic token budget and canonical request order per tick.
   Give active decode a bounded opportunity every tick, with age-based
   progress for waiting prefills. Tune aligned chunk sizes from profiles;
   128/256/512-token candidates are experiments, not universal defaults.
   Use larger chunks when there is no decode work if equivalence permits.
4. Journal the budget/policy and chunk plan, or derive the plan identically
   from journaled inputs. Never let each rank independently stop work based
   on its wall clock. A future latency-adaptive policy must distribute the
   head rank's decision before execution.
5. Drain graph replays before eager work and reseed live feeds **after each
   yielded chunk**. `GraphEngine::open_slot` currently does this only around
   the whole prompt; prefill can overwrite the token-feed storage.
6. Publish snapshots only at valid completed state boundaries. Retain
   partially owned cache entries across ticks and release them on cancel,
   error or retirement. Delay the first pick, grammar advance and draft
   initialization until finalization.
7. After single-prompt continuation passes, extend the existing span walker
   to grouped continuation chunks and eligible cache-hit suffixes.

### Model-specific traps

- Calling `session_prefill_resume` on every chunk is not sufficient: it is
  a prefix-attach API with alignment and MTP catch-up semantics.
- KDA/GDN recurrence, convolution histories, sparse-attention pool tails,
  and MTP's shifted token input must survive an arbitrary number of ticks.
  Additional cut sizes need numerical validation, not just matching offsets.
- DeepSeek's bounded prefill defers its decoder to a logical span's end.
  `tail_streams_`, `tail_pre_`, `tail_rows_`, `tail_end_` and `call_pos0_`
  are currently shared scratch. Make them per-prefilling-request or
  explicitly save/restore them before multiplexing prefills. Preserve
  `first_chunk`/`last_chunk` across yields and snapshot boundaries; do not
  rerun the decoder at every scheduler yield.

**Exit:** a 32K admission no longer creates a whole-prompt pause for other
streams. Measure the longest pause against one bounded prefill quantum
plus decode/transition cost. Check cancellation at every yield, admission
pressure, slot reuse, prefix equivalence, sampled state and rank agreement.
Report the TTFT/throughput tradeoff rather than claiming free throughput.

## 4. P1: widen useful batching, starting with Qwen

### Separate the limits

For S live requests and uniform draft depth K, a verify pass consumes
`R = S × (1 + K)` rows. Request capacity, physical row capacity and draft
depth should be separate controls.

[`GraphEngine`](../src/engine/graph_engine.hpp) currently marks batching
unavailable if the **configured slot count** times rows/request exceeds the
model limit. Consequently a deeper configuration can serialize even a
smaller live set that would fit. Build all fitting batch families and
advance larger live sets through deterministic sub-batches. Let the
scheduler see the true per-pass capacity, with fair rotation.

### Qwen first

- The cap in [`QwenFamily`](../apps/dgpp_serve.cpp) is eight rows, but
  `SessionModel` and much of Qwen's scratch already accept runtime sizes.
- Both shared-expert decode launchers in
  [qwen_moe.cu](../src/kernels/qwen_moe.cu) reject more than
  `2 * gemv::kMaxRows` (eight). Keep the efficient small-row fused path;
  use a wide shared-expert GEMM path above its measured crossover.
  Extending the four-row loop alone repeats weight reads.
- Audit QSA selection/attention, GDN/PLE rollback, GR fused dispatch,
  sampling scratch and feed buffers. Some GR entry points also enforce
  eight rows; confirm wider calls choose the general path.
- Validate 12/16 rows first (C6/C8 at depth 1), then 24/32 rows. Qwen
  already declares a batched draft chain: exercise and validate it.

### GLM-Flash next

Make its fixed arrays, recurrent snapshots, DSA scratch, pinned mirrors,
MTP windows and memory-plan formulas depend on runtime row capacity.
Use the shared session core as a reference without making a wholesale
model rewrite a prerequisite. Implement `kBatchedDraftChain` support.
Start at 16 rows, then 32 after kernel and fabric gates.

### Existing wider families

Benchmark GLM-4.7 C6/C8 at depth 1 now. Keep full GLM's eight-slot default
as a baseline. DeepSeek's default C6/depth 4 consumes 30 rows; C8/depth 3
and C16/depth 1 fit the arithmetic 32-row ceiling, but still need memory,
sampling and graph checks. These are experiments, not validated recipes.

### Graph and resource limits

The bus allows 64 graph variants; capture currently multiplies slots,
batch families, two parities and depth choices. The device pick supports
32 rows and 16 request verdicts. Add explicit capacity validation and
budget capture families; do not merely increase one constant. Preserve
small C1/C2 graphs and include boot time, graph memory, sampling candidate
width, KV reservation and recurrent-state memory in the evaluation.

**Exit:** real C6/C8 throughput gains with fair latency, accurate memory
plans, isolation under uneven completions/cancellation, and no material
regression at C1/C2. Capacity increases alone do not satisfy this gate.

## 5. P1/P2: prefill compute and communication

| Project | Code and proposed change | Evidence / gate |
|---|---|---|
| Packed int4/int8 GEMM | [packq_gemv.cu](../src/kernels/packq_gemv.cu), [glm_moe.cu](../src/kernels/glm_moe.cu), [full GLM model](../src/models/glm_dsa/model.cpp): dense and grouped expert tile kernels that unpack each weight tile once and use tensor cores across prompt rows | The documented 7–9.5 ms/token full-GLM prefill is a specific missing-kernel case. Preserve offset-code/group-scale semantics; test dequantization, accumulations, logits and task quality |
| Broaden stream-ordered folds | [tp_bus.hpp](../src/engine/tp_bus.hpp), [dgpp_serve.cpp](../apps/dgpp_serve.cpp): A/B the existing stream reducer on Qwen/GLM families | It is currently selected only for DeepSeek. Boundaries larger than one latency slot still synchronize and use host-driven bulk, so this primarily targets short/chunked prefill |
| Stream-ordered bulk collectives | [collective_bus.cpp](../src/net/collective_bus.cpp), [bus_kernel.cu](../src/net/bus_kernel.cu): remove host round trips at large boundaries, retain canonical reduction order and buffer lifetime | Measure network wait versus copy/fold/host time first. The current small-message reducer does not implement this |
| Expert prefill tiles | [glm_moe.cu](../src/kernels/glm_moe.cu): require evidence of avoidable traffic or instruction cost before another tile sweep; measure FP8 and FP4 separately | The [small-chunk FP8 investigation](../benchmarks/results/2026-09-15-qwen-moe-prefill.md) rejected three variants. Recorded median-route gate/down memory fills were within about 1% of one selected-weight payload. The earlier 8K-chunk experiment bought only 3.6% for 17 GiB/rank |
| Attention/recurrence | Qwen QSA and GDN, GLM-4.7 GQA: tiled attention and a chunk-parallel recurrence only where a new profile justifies them | Larger numerical projects; validate state continuation and target decisions, not only isolated kernel time |

Only after stream-ordered bulk works, test overlap of a row block's fold
with independent next-block work. Respect causal KV publication and
recurrence order; multiple streams do not remove those dependencies.

Qwen also runs its final mixer and vocabulary head on every prefill row
even when only selected last rows are consumed. A selective-row head can
reduce scratch and unnecessary work, but the code intentionally uses the
same `m=T` arithmetic as its forward oracle. Preserve a matching lowering
or explicitly validate the numerical change. Its earlier measured cost
was small, so treat it as a supporting optimization.

**Exit:** improved cold service prefill at 2K/8K/32K on the same workload,
with procedure/service differences attributed. Set numeric speed targets
after profiles establish the fraction of time each project can affect.

## 6. P2: improve high-concurrency expert work and speculation policy

### Expert reuse beyond ordering

[`GlmMoeLayer::enqueue_decode_impl`](../src/models/glm/moe_layer.cpp)
already orders slots by expert for locality. Sorting is not a new feature
to add. Investigate explicit multi-row reuse within each expert: form
device-side segments, stream a tile for several assigned rows, and write
results back to token/expert slots for canonical accumulation.

Measure experts touched per layer, rows per expert, actual DRAM reads and
launch overhead on captured real routes. Select slot GEMV for sparse
assignments and small grouped GEMM for sufficiently reused experts.
Reuse the existing prefill segmentation as a starting point, while avoiding
large empty tiles and host routing in a captured decode pass.

The DeepSeek record attributes roughly 110 ms of the historical 30-row
step to MoE and 33 ms to collectives. It also describes expert traffic near
the union-of-experts floor. If fresh profiling confirms that floor, more
tiling cannot eliminate compulsory reads: reduce wasted verify rows or
change the deployment/quantization instead. Do not promise a 2× kernel win.

### Extend the existing adaptive policy

The current policy uses prefix survival against a linear row-cost model,
adapts lambda using replicated modeled time, and chooses one depth for the
batch. It always computes the draft block, requires minimum depth 1, and
forces full depth for sampled/fresh members.

1. Fit an offline cost table by family/world, live slots, row shape and
   context bucket. Kernel crossovers make a single linear slope inaccurate.
   Distribute its version/settings across ranks; local clocks must not
   independently change collective schedules.
2. Compare a true plain-decode option against drafting **before** paying
   draft cost. This needs plain graph variants and safe draft-state
   catch-up/reinitialization; setting the existing minimum depth to zero
   is insufficient.
3. Try a small set of depth buckets before arbitrary per-slot depths.
   Compare saved rows with extra launches, weight reads and communication
   from splitting batches. Ultimately use packed per-request spans where
   it wins; preserve deterministic slot order and fair service.
4. Calibrate draft token probabilities against actual prefix acceptance
   for MTP families without a confidence head. A likely draft token is
   not itself a calibrated probability of matching target greedy output.
5. Extend scheduling to sampled traffic only after correct proposal
   probabilities, residual sampling, RNG positions, penalties and grammar
   masks are covered. Check mixed greedy/sampled batches explicitly.

**Exit:** better aggregate committed tokens per wall second across mixed
classes and arrival rates, including C1. Report draft cost, discarded
verification rows and depth distributions alongside acceptance.

## 7. P2: a scoped DFlash2 implementation for GLM-Flash

### Compatibility and scope

"K7" is a depth setting: seven drafts plus the pending/anchor token form
eight target verification rows. It is not a separate drafting algorithm.
The [GLM-Flash deployment author's recipe](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/blob/main/docs/DFLASH2-SPECULATIVE-DECODING.md)
uses exactly this interpretation.

A compatible [GLM-5.3-Flash DFlash2 checkpoint](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
exists. Its [configuration](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/blob/main/config.json)
specifies five draft layers, block size 8, target taps `[5,14,24,33,42]`,
top-16 selection, a rank-256 selector and two-tap convolutions. Those are
trained components. They cannot be obtained by deepening native MTP.
The card currently labels the weights CC BY-NC-ND 4.0 for research and
evaluation, with commercial licensing separate; engine support and shipping
those weights are distinct deliverables.

The [authors' DFlash2 description](https://inco.ai/blog/dflash2/) explains
parallel block prediction followed by lightweight candidate-path selection
and local convolutions. Implement the released architecture and reference
behavior rather than treating it as a generic small autoregressive model.

The released `z-lab/Qwen3.8-27B-DFlash2` checkpoint is target-matched to
Qwen/Qwen3.8-27B — its config names that base model — and is served by
the native engine today
(`deploy/cluster_qwen3.8-27b-fp8_w1_dflash2.example.json`,
[mtp.md](mtp.md#the-dflash2-block-drafter-qwen35-family)). No
target-matched drafter is established for Qwen3.8-Flash-Next or, from
the [DeepSpec released checkpoint table](https://github.com/deepseek-ai/DeepSpec#released-checkpoints),
for every dgpp model; each new lane needs its own checkpoint check.

### Implementation sequence

1. Introduce a narrow draft-provider contract around native MTP, existing
   DSpark and external DFlash2: required target features, prepare/propose,
   proposal probabilities, commit/rollback, cache state and memory cost.
   Keep the existing verifier and sampler responsible for exact acceptance.
2. Add device-resident target feature capture at the configured taps in
   GLM-Flash's forward/verify paths. Check layer numbering and mHC stream
   contraction against the reference implementation. The deployment recipe
   explicitly documents these as silent-acceptance-failure hazards.
3. Load the drafter independently of the target and include its weights,
   sliding-window KV, feature windows and scratch in the memory plan and
   prefix-state identity. Evaluate replicated versus TP draft placement;
   small draft layers can lose their benefit to extra collectives.
4. Implement the bidirectional block/sliding-window attention, convolution
   and candidate selector; begin with eager greedy C1 and a pinned
   reference oracle. Keep target verification causal.
5. Generalize the shared speculative limit from five drafts/six rows to
   seven/eight, including `kSpecMaxDrafts`, config validation, feed indexing,
   masks, pick scratch and snapshots. Keep DSpark's trained five-position
   block unchanged. Reuse GLM's wider row/state work from section 4.
6. Validate target and drafter rollback at every rejection position,
   especially KDA recurrence and DSA tails. Prefix attachment must restore
   or reconstruct the drafter's required context; never attach target state
   alone and silently draft from an unrelated history.
7. Capture C1, then batched graphs, then sampled verification with the
   selector's actual conditional proposal distribution. Add adaptive depth
   only after the fixed-depth implementation is correct.

At K7, 32 verification rows cover only four requests; C8 would need 64
rows for one full-width pass. Use shorter depths or sub-batches initially.
A global increase to 128 rows is not required to prove the C1 benefit.

**Exit:** greedy speculation matches the same target's plain path, sampled
verification matches its distribution, and per-class end-to-end results
beat the best native-MTP configuration at acceptable memory cost. High
acceptance alone is not success. Exact speculation does not undo quality
changes introduced by a different target quantization.

### Status (2026-10-01, Qwen3.8-27B lane)

Steps 3–6 are implemented for the Qwen3.5-family target
(`src/models/qwen/dflash2.*`, `src/kernels/dflash2.*`, the `dflash2_*`
methods of `Qwen35Model`, `DFlash2Speculator`): the drafter loads
independently with its weights and planes in the memory plan, taps
capture through the five `fc` slices with fp32 accumulation, the five
draft layers run bidirectional sliding-window attention over pool
planes with the 2-tap dynamic grouped convs, per-slot top-1 proposals
verify 2.5–6.1 tok/pass against MTP depth-2's 2.3–2.8 (the reference
chained selector walk is implemented exactly and kept behind
`DGPP_DFLASH2_WALK`, but verifies only ~1.2 here — see mtp.md), and
acceptance is the greedy verify/rollback of eight rows
(`kSpecRows`/`kSpecMaxDrafts` generalized 6/5 → 8/7; the DSpark block
is unchanged). The batched pass landed 2026-10-01: one physical target
verify covers every arriving slot (`SessionModel::session_verify_batch`,
slot-major rows and rollback bases; `EagerEngine::step_batch` caps the
batch to `floor(decode_rows/8)` slots — four at the 32-row ceiling, the
plan's K7 arithmetic — and a single-slot batch keeps the scalar kernel
sequence bit-for-bit). Drafts stay per-slot. Sampled verification (the
selector's conditional proposal) is open. The GLM-Flash lane (step
2's mHC-tap capture) is untouched.

Measured 2026-10-01 (Qwen3.8-27B-FP8, greedy, 12-prompt battery,
max_tokens 128): the block's per-slot top-1 proposals verify 2.5–6.1
tok/pass against MTP depth-2's 2.3–2.82 — the exit gate, met on every
prompt tried. The reference chained selector walk (implemented exactly,
kept behind `DGPP_DFLASH2_WALK`) verifies only ~1.2 here; either a
vLLM cross-check shows the reference picks the same drafts here
(head-domain behavior, proposal rule to revisit), or it picks better
ones (our block outputs differ somewhere top-1 doesn't see).
Transcripts match plain modulo single near-tie flips from
width-dependent GEMM numerics (T=8 verify vs T=1 steps). Concurrency
(2026-10-01): the batched speculative pass (one physical verify for
every arriving slot, `floor(decode_rows/8)` slots per pass) replaces
the flat ~8 t/s c4 line — same short-prompt harness gives dflash
25.7/41.3/52.4 agg tg at c1/c2/c4 vs MTP depth-2's 14.4/32.6/47.3,
with C1 transcripts 12/12 identical to the pre-batch build. The
stacked redraft is default-on (opt out with
`DGPP_DFLASH2_DRAFT_BATCH=0`); `DGPP_DFLASH2_DEPTH=k` stays opt-in.
Measured 2026-10-01: batch +5–9% at c4 with no errors; depth-5 keeps
τ 4.0 (80% window use). Headline correction the same day: the
throughput bench's traffic was *sampled* (benchy omits temperature;
the server inherits temp=1.0 from generation_config), so greedy-only
spec never engaged on either lane — all older throughput numbers are
sampled-plain decode, graphs vs eager. With `temperature=0` forced,
dflash throughput jumps +60–190% and wins every c1/c2 cell vs
graphed MTP (τ 4.5 vs ~2.5), ceding only c4, where graphs scale
linearly. That lane closed 2026-10-02: `DGPP_DFLASH2_VERIFY_GRAPH=1`
replays the batched verify as one static 8/16/32-row graph (the
slots' fed rows padded to full 8-row blocks; the drafter's context
K/V feed is a recorded node — first capture attempt starved the
planes and decayed acceptance), with the C1 capture bit-exact and the
graph the fastest path on that binary (35.9–36.1 vs 35.0–35.4 agg tg
at c4 on the short-prompt harness; see mtp.md "batched graph capture
with a drafter loaded"). A later default-knob sweep found that reading
had been graph-vs-graph: the shipped default (`DGPP_DFLASH2_VERIFY_GRAPH`
unset) had always graphed the multi-slot batch — levels 0 and 1 were
identical — so those "eager" numbers were the graph. Making level `0` a
true eager baseline and setting the default to `1` (the multi-slot
graph), re-measured against the real eager path (`BATCH_EAGER=1`): the
graph ties it at 8K c4 (14.9 vs 14.8–14.9) and leads it at
short-context c4/c8 (~54–55 vs ~51 agg tg, within that harness's
variance), never slower; the single-slot graph (level `2`) is the slowest
c1 option (20.1 vs the scalar path's 24.2), so lone slots stay scalar.
The rest of the sweep held: batched redrafts on (`DRAFT_BATCH`) 14.9 vs
14.3 off at 8K c4; full verify depth 14.9 vs 14.4 (k=4) vs 13.2 (k=2);
per-slot top-1 over walk (~1.2 tok/pass). Nothing performance-relevant
is opt-in on this lane (mtp.md "default knob sweep").

The 8K step profile (`DGPP_DFLASH2_PHASES=1` + nsys, 2026-10-02):
466 ms/pass at c4 = verify 195 + draft 260 + host ~11; the graph is
neutral at length (launches amortized), so the c4 gap to MTP there
(12.9 vs 14.8 same-workload agg tg; dflash still wins c1, 12.1 vs
7.8) sits in the draft block. The original kernel-sum read of that
window was wrong (`mma_gemv_kernel<(int)8, …>` is the mma form's
128-row tile, and those launches are the target's FP8 *prefill*
chunks); the corrected diagnosis (2026-10-02, dispatch trace via
`DGPP_MMA_TRACE`): the **BF16 drafter**'s wide GEMMs ran the
kernel-only 4-row GEMV chunks — m=32 as eight 4-row launches, eight
reads of the 5.8 GB of draft weights per pass — while the FP8 target
was already single-read. The lever was then implemented the same day:
`set_decode_mma(on, min_rows, max_rows)` plus the wide-decode opt-in
in `qwen_configure_gemm_rows` (17..128 rows → the streaming mma form,
one weight read; m ≤ 16 untouched, so C1 stays bit-exact 12/12, eager
and graphed). Result: the 8K draft phase 260 → 92 ms/pass, and the
same-workload head-to-head flips to **dflash 14.9 / 13.2 / 12.2 agg
tg at c4/c2/c1 against MTP's 14.8 / 12.7 / 7.6** — the §7 exit gate
("per-class end-to-end results beat the best native-MTP
configuration") is now met at every concurrency tried on the Qwen
lane (see mtp.md "wide-row GEMM dispatch"). Remaining in this lane:
the vLLM walk-parity dump (open), sampled verification (open), and
the c2 band (m=16 still Lt; the dsv41 table says Lt leads the mma
form below ~16 rows, so leaving it is the conservative call).

## 8. P1 experiments / P3 implementation: remaining single-stream wins

- **Qwen TP2 NVFP4 with a resident n-gram table:** first validate a recipe
  using the existing NVFP4 loader/sharding and resident-table setting.
  The checked-in TP2 template uses FP8 weights; TP1 NVFP4 uses mmap.
  Confirm format/geometry support, memory headroom and quality, then compare
  FP8/NVFP4 and resident/mmap as separate variables. Resident tables already
  exist and should not be presented as a new engine feature.
- **Native depth selection:** re-evaluate GLM-Flash TP2 depth 2 for C1;
  the latest measurements show code/math gains. Evaluate it after fixing
  the batch fallback, so a C1 preference does not serialize concurrent work.
- **Launch and collective gaps:** profile C1 on the current build, then
  fuse profitable producer/consumer sites and investigate peer skew using
  `scripts/bus_window_skew.py`. Graph replay and pipelined settle already
  exist. A more aggressive launch-before-verdict scheme is a separate
  rollback protocol project and should require a measured residual gap.
- **Memory-mapped tables:** measure page faults and gather latency before
  adding a bounded host row cache/prefetcher for Qwen or DeepSeek. CPU page
  cache, pinned staging and GPU allocations compete for Spark memory.
- **Additional quantization:** investigate INT4-AutoRound interoperability
  only with the actual checkpoint layout/scales and teacher-forced quality
  checks. Existing full-GLM packed-int support is useful infrastructure,
  not proof that a Qwen AutoRound checkpoint can load unchanged. EXL3 and
  a one-Spark GLM-Flash lane are larger capacity/format projects.

Avoid replaying measured losses without new evidence: larger prefill
chunks as the main optimization, all-row fused Qwen GR, indiscriminate
prefetch, and deeper speculation by default under concurrency.

## 9. Delivery order and validation

| Order | Reviewable deliverable | Relative scope | Acceptance |
|---|---|---|---|
| 1 | Measurement fixes, current profiles, feasible existing-recipe sweeps | Small/medium | Reproducible C1/C4/C6 and cold service prefill |
| 2 | Qwen wide shared-expert path and 16-row serving gates; fitting batch families | Medium | C6/C8 gains, C1/C2 preserved |
| 3 | Prefill continuation contract, then budgeted scheduler and journal | Large | Long-prompt interference bounded by a quantum; cancellation/cache/rank gates |
| 4 | Packed int4/int8 prefill kernels; short-message reducer expansion | Large / medium | Matched service TTFT improvements and numerical gates |
| 5 | GLM-Flash runtime rows and batched draft chain | Large | C6/C8 and concurrent deeper MTP work correctly |
| 6 | Expert decode reuse, measured-cost speculative policy, stream-ordered bulk | Separate medium/large changes | Profiled bottlenecks shrink in end-to-end runs |
| 7 | GLM-Flash DFlash2: feature oracle → eager C1 → graph → batch/sample | Large | Wins over native MTP, not just over plain decode |

The scheduler and kernel efforts can be developed independently after the
measurement baseline. DFlash2 checkpoint/feature validation can begin early;
its production integration depends on verifier/state and graph capacity work.
Scope labels indicate implementation risk, not delivery-date commitments.

For implementation, follow the repository's full-build requirement before
relevant suites. Extend existing tests rather than creating a parallel
validation framework:

- Scheduler/journal: `tests/host/scheduler_test.cpp`,
  `tests/host/fabric_serve_test.cpp`, service cancellation and prefix tests.
- State/batching/speculation: the model `*_decode_test` and `*_engine_test`
  suites, `tests/unit/verify_schedule_test.cpp`, forced rejection and slot
  reuse cases, and relevant TP tests.
- Kernels/communication: `glm_moe_test`, `qwen_moe_test`, `packq_gemv_test`,
  `mma_gemv_test`, `bus_test`, and real-checkpoint reference comparisons.
- Fabric: five-class greedy/sampled sweeps, transcript isolation, the
  applicable quality suites and a mixed-workload soak, with operation-stream
  agreement on every rank.

Keep three gates distinct: numerical correctness against a model reference,
speculative equivalence to the selected target path, and rank operation
agreement. An operation hash cannot establish model quality, and a quality
score cannot establish correct speculative rollback. When a new tensor-core
lowering changes reduction order, record the numerical/decision differences
and validate them under the repository's existing policy; do not hide a
speculation mismatch by widening a tolerance.

**First implementation recommendation:** deliver the measurement fixes and
Qwen 16-row support first, begin the resumable-prefill contract next, and
use the resulting profiles to choose between bulk-collective and expert
kernel work. Start DFlash2 with GLM feature-capture parity, which resolves
its largest compatibility uncertainty before a full integration.

## 10. Next priorities after the first delivery

The first delivery improves concurrency and bounds long-prompt interference.
The follow-up completes the 256/512/1024 budget sweep and optional larger
idle chunks. It recovers idle prefill efficiency and accelerates a pending
prompt after its decoding peer retires, while retaining the 256-token busy
budget. Grouped continuation remains unfinished.

Matched service profiles now point to routed-expert matrix multiplies as the
main small-chunk penalty: 1.38 seconds unbudgeted versus 4.68 seconds with a
256-token budget for the same profiled 8,281-token prompt. QSA attention is
also a substantial large-chunk cost. The subsequent
[expert-kernel investigation](../benchmarks/results/2026-09-15-qwen-moe-prefill.md)
tested smaller tiles, compact grids and persistent blocks. All were rejected:
the compact prototype moved service prefill by only 0.16–1.05%, while other
routing patterns regressed. The recorded median-route memory fills were
already close to a single read of the selected weights. Further Qwen tile
sweeps need a more specific source of avoidable work. The remaining priorities
below are proposed work, not measured wins. Full GLM's packed GEMM has since
been implemented; its [dated report](../benchmarks/results/2026-09-15-glm-packed-prefill.md)
records cold-service, C1 and quality evidence. It reuses packed tiles across
prompt rows without rounding the dequantized weights to BF16.

The subsequent [QSA prefill change](../benchmarks/results/2026-09-15-qwen-qsa-prefill.md)
shares each KV tile across twelve query heads and computes probabilities
cooperatively within a warp. It retains the original partial arithmetic and
the decode/verification kernels. Cold TP2 service prefill takes 8–14% less
time on the measured 2K/8K/32K prompts. Grouped continuation is the remaining
part of the first priority; evaluate it separately against this new baseline.

| Priority | Optimization | Concrete next step and success criterion |
|---|---|---|
| 1 | Qwen grouped continuation | Pack resumable spans from multiple requests where useful, preserving each cursor's positions, MTP state and snapshot cuts. Improve expert weight reuse at the same decode-pause budget; QSA tile reuse is now delivered |
| 2 | GLM-Flash concurrency | Generalize fixed recurrent/DSA/MTP state and scratch to 16 rows, then validate 32 and a batched draft chain. Reproduce useful C6/C8 gains with default C1 preserved |
| 3 | Less repeated expert traffic and collective overhead | Measure rows per expert and per-rank communication waits. Reuse weight tiles across rows assigned to the same expert; broaden the existing stream-ordered reducer where it helps, then tackle GPU-driven bulk collectives |
| 4 | Speculation matched to workload | Calibrate costs by batch width, context and draft depth. Improve existing DSpark/native-MTP depth selection and add a safe plain-decode choice before paying draft cost. Optimize committed tokens per wall second rather than acceptance alone |
| 5 | GLM-Flash DFlash2 | Establish target-feature parity against the compatible drafter first, then implement loading, proposal, rollback and verification. K7 means seven draft tokens; it needs verifier/state capacity changes and the trained DFlash2 architecture |
| 6 | Remaining C1 and client-latency improvements | Evaluate Qwen TP2 NVFP4 with resident n-gram tables, revisit per-class native MTP depth, profile launch/peer skew, and replace periodic HTTP output polling with a producer wakeup if measured delivery latency warrants it. Keep kernel throughput, service throughput and visible latency separate |

Arbitrary physical slot subsets and continuation for GLM/DeepSeek also
remain open. DeepSeek needs its shared bounded-prefill scratch made safe
across yields. Wider families such as GLM-4.7 can first receive matched
capacity/depth sweeps using their existing runtime row support.

The budget sweep, Qwen small-chunk expert investigation, full-GLM packed
prefill implementation and Qwen QSA tile reuse are complete. Continue with
grouped continuation and GLM-Flash row expansion. Full GLM can next use
profiles of the new path to identify the remaining attention/collective
cost; a lower packed cutoff needs measured
short-C1 gains as well as numerical validation.
DFlash2 feature validation can begin before committing to its full port.
Use repeated deployment A/B pairs for small C1 effects; consecutive prompt
repeats alone cannot separate implementation cost from session drift.
