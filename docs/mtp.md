# Speculative decoding with MTP

An MTP layer predicts draft tokens from the main model's hidden state and
the next input token. The main model verifies those drafts, commits the
accepted prefix and restores state after a rejection. This can produce
more than one output token per decode step.

For a GLM-5.3-Flash diagnostic run on the fabric:

```bash
scripts/fabric_run.sh -- --model unsloth/GLM-5.3-Flash-FP8 \
    --chat "Write a history of the Roman Republic." --steps 300 \
    --decode-graph --mtp
```

## Execution and correctness

At depth 1, a graph replay verifies two rows: the pending token and one
draft. Device kernels select tokens, decide acceptance, commit the accepted
rows, restore speculative state where needed, and run the draft block for
the next step. Each rank derives the same result from the gathered logits.
The host reads the verdict and updates request bookkeeping. Sampling that
cannot be resolved from the candidate table uses an exact gather fallback.

Greedy MTP must produce the same transcript as plain decode. The verify
rows preserve single-row arithmetic, and state tests cover rejection at
pool boundaries, slot reuse and scalar/batch transitions. Compare greedy
runs with `scripts/fabric_xcript.py PLAIN_DIR MTP_DIR`; the result must be
`IDENTICAL`.

Sampled MTP preserves the target distribution. GLM-5.3 uses a greedy draft;
Qwen also supports sampled drafts with an acceptance ratio and residual
sampling after rejection. A sampled transcript need not equal a plain
sampled run with the same seed. See `src/engine/speculative.hpp` and
the sampling tests for the acceptance rules.

## Serving configuration

Set `engine.decode_graph` and `engine.mtp` to true. The server requires
graph decode for MTP; the GLM diagnostic tool also has an eager speculative
path. Graph serving works on one node when the model fits, using identity
collectives.

`engine.mtp_depth` accepts 1–7 and defaults to 1 (DeepSeek-V4.1 defaults
to its DSpark depth). Each step verifies `1 + depth` rows. GLM-5.3 uses
scalar graphs beyond depth 1; Qwen and GLM-4.7 run batched draft chains at
every depth within their row limits (Qwen: sixteen slots at depth 3, the
64-row cap).
At depth 1, GLM-5.3 and Qwen can batch up to four requests. The engine
chooses among scalar and available batch graphs according to occupancy
and `graph_batch_min_live`.

Deeper drafts add verification work and state. Their benefit depends on
acceptance, prompt class, context and concurrency. Use the memory plan
before increasing capacity, and measure tokens per step as well as step
latency. [Operations](operations.md) describes the depth tradeoff, and
[benchmarks](benchmarks.md) records the results for each model.

Deeper drafts add verification work and state. Their benefit depends on
acceptance, prompt class, context and concurrency. Use the memory plan
before increasing capacity, and measure tokens per step as well as step
latency. [Operations](operations.md) describes the depth tradeoff, and
[benchmarks](benchmarks.md) records the results for each model.

## The DFlash2 block drafter (Qwen3.5 family)

DFlash2 (`src/models/qwen/dflash2.hpp`) replaces the MTP draft with an
external block drafter checkpoint (`z-lab/Qwen3.8-27B-DFlash2`): five
bidirectional Qwen3 layers that turn the target's tapped hidden states
(layers `[5,19,33,47,61]` through a fused `fc` + RMSNorm) and a block of
masked slots into seven drafts in one non-autoregressive pass. The
drafter shares the target's embedding and lm head and keeps its K/V in
five extra planes of the main pool, so prefix caching and rollback ride
the existing protocol.

Serve it with the drafter checkpoint in `engine.dflash_model` (or
`--dflash-model DIR_OR_ID`) and `engine.mtp` / `engine.decode_graph`
off — the drafter is the eager world-1 path (see
`deploy/cluster_qwen3.8-27b-fp8_w1_dflash2.example.json`;
`--no-dflash` runs the same recipe plain). Acceptance
is the ordinary greedy verify: the eight fed rows (pending token +
drafts) run through the target, the accepted prefix commits and the
rest rolls back, so the transcript stays exact. The throughput line
carries the drafter's per-position acceptance in the usual MTP group.
The lane's exit criterion (performance plan §7) is to beat the best
native-MTP configuration; until it does, the MTP recipe remains the
default.

### Proposal rule: per-slot top-1 over a full walk pass

Each mask slot's own top-1 proposes its draft; the block forward, head
and top-K run first, then the reference chained selector walk runs and
its picks are replaced by the per-slot top-1s
(`DGPP_DFLASH2_WALK=1` keeps the walk's own picks). This ordering is
load-bearing, not incidental: skipping the walk pass collapses
acceptance to ~1.0 tok/pass on every prompt tried, through a coupling
mechanism still open (the walk writes only its own buffers, yet its
absence deterministically changes the next verify's verdicts —
suspected GEMM/state coupling, under investigation). Measured on the
shipped form: 2.5–6.1 tok/pass at 22–45 tok/s against MTP depth-2's
2.3–2.8 on the 12-prompt greedy battery (3.4–6.1 on chat prompts,
2.5–5.1 on raw completions), and above vLLM-raw's 2.68. The block,
head, top-K and walk each check out individually (formula, kernel and
NumPy cross-checks; transcripts identical across proposal rules), so
the walk's own picks underperforming here is isolated to the trained
selector's pairwise edges on these prompts; the vLLM-side cross-check
(its selector table on the same prefixes) is open.

### Measured acceptance (2026-10-01, greedy, 12-prompt battery)

On Qwen3.8-27B-FP8, max_tokens 128, one stream: native MTP depth 2
verifies 2.3–2.8 tok/pass; the DFlash2 block's per-slot top-1 verifies
2.5–6.1 tok/pass at 22–45 tok/s (3.4–6.1 on chat prompts, 2.5–5.1 on
raw completions). The block/head/top-K chain is proven by the numbers;
the selector walk is off by default until its gap is understood.
Transcripts match the plain path except for single near-tie flips
inherent to width-dependent GEMM numerics (T=8 verify vs T=1 steps,
the same class the graph engine already shows against eager); all
flips observed were single-token cascades.

### Concurrency: the batched speculative pass

Every arriving slot's verify rows ride one physical target pass
(`SessionModel::session_verify_batch`, slot-major staging with per-slot
snapshot/rollback bases, riding the graph era's batched decode kernels;
`EagerEngine::step_batch`). One pass covers `floor(decode_rows / 8)`
slots — four at the family's 32-row ceiling; `serve` sizes the batch to
`max_concurrency x 8` clamped to it, and wider occupancy round-robins
two physical ticks. Drafts stay per-slot (the block forward is a few
percent of the target step). The single-slot case takes the scalar
path, so C1 keeps the scalar kernel sequence bit-for-bit (12/12
transcripts identical to the pre-batch build); multi-slot greedy
transcripts may show the same batched-kernel near-tie class the
graph engine shows against eager. Measured 2026-10-01 on the same
short-prompt harness (greedy, max_tokens 128, 4-slot configs):
dflash aggregate tg 25.7 / 41.3 / 52.4 at c1/c2/c4 against MTP
depth-2's 14.4 / 32.6 / 47.3 — the flat ~8 t/s c4 line is gone, and
the per-step cost is ~160 ms at c1 vs ~250 ms for a 4-slot batch
(one sweep instead of four). Per-request acceptance holds under
batching (4.5–5.7 tok/pass at c4).

### Long-context follow-ups: batched redrafts, verify-depth cap

The throughput bench (long-context thinking work) still trails MTP at
c4 (15.3 vs 37 at d0) while short prompts lead — the deficit is
per-step cost at 4–8K context, not acceptance. Two levers, both
opt-in and default-off (the shipped path is untouched):

- `DGPP_DFLASH2_DRAFT_BATCH=1` batches the redrafts: one stacked block
  forward per step (`Qwen35Model::dflash2_draft_batch` — row-wise
  GEMMs/norms/convs over S*8 rows in a single launch each, so the
  draft weights are read once; the grouped conv kernel was already
  block-boundary aware; KV appends, sliding-window attention, head,
  top-K and the selector walk stay per-slot at row offsets, the same
  calls as the scalar path). The engine splits each slot's commit into
  judge (`commit_verify`) and redraft, then assigns the batch result
  (`set_drafts`); a single spec slot keeps the scalar redraft.
- `DGPP_DFLASH2_DEPTH=k` verifies only the first k drafts per step
  (exact transcripts — unverified drafts re-draft next step, the same
  hook `GreedySpeculator` carries). Each tail row scores the full KV
  for a shrinking acceptance, so at long context fewer rows per
  accepted token can win back throughput.

Both need serving validation (host unit tests cover the speculator
halves in `dflash2_speculator_test`; the device batch path is
build-checked only — the GPU was under the user's bench).

### Measured on tool-eval-bench (2026-10-01, new binary)

Shipped path, spec-bench: τ 3.1–6.1, avg 4.5 (`...17-22-23...md`).
Shipped path, throughput: d0 7.7/11.8/14.9, d4096 7.4/10.1/11.6,
d8192 6.9/7.5/9.0 at c1/c2/c4 (`...17-38-10...md`).
`DRAFT_BATCH=1`, throughput: d0 8.1/12.0/15.7, d4096 7.7/10.3/12.3,
d8192 7.3/8.8/9.8 — +5–9% at c4, no errors (`...17-53-46...md`).
`DRAFT_BATCH=1 DEPTH=5`, spec-bench: window 5, τ 3.0–5.0 avg 4.0,
waste 40% vs 50% (`...17-57-38...md`); throughput: identical to
depth-7 within noise (`...18-18-05...md`).

The depth neutrality is explained, not a lever failure: of 117
requests in that server's log, 87 ran plain (~1.0 tok/pass) and
only 30 engaged spec (mean 3.86 tok/pass) — and the trace below
shows why: the bench's traffic was sampled, not greedy. Where
speculation never engages, no spec-side lever moves the headline
number. The remaining gap to graph-MTP there (15.7 vs 37 at d0 c4)
sits mostly in the plain path: the dflash config forces
`decode_graph=false`, so its plain steps are eager
while MTP's are graphed. Closing that gap means batched graphs with
a drafter loaded (plan §7 step 4), not further draft tuning.

### The temperature trap (2026-10-01): the bench was sampled all along

`DGPP_DFLASH2_TRACE` showed every bench slot-step ineligible with
`temp=1` — yet benchy sends no temperature field. Root cause: the
server's sampling defaults come from the checkpoint's
`generation_config.json` (DESIGN §10 — Qwen3.8-27B-FP8 declares
`temperature: 1.0, top_k: 20, top_p: 0.95, do_sample: true`), so any
request omitting temperature runs *sampled* — and greedy-only
speculation (MTP and dflash alike) never engages. All pre-2026-10-01
throughput numbers on both lanes are sampled-plain decode (graphs vs
eager), not spec numbers; vLLM's lead there is sampled-*spec* vs
no-spec, not drafter quality. The harness fix:
`--benchy-args "--extra-body temperature=0"` (benchy wants
`key=value`, not JSON). With greedy engaged, dflash throughput jumps
+60–190% per cell (`...19-26-43...md`): d0 20.1/31.8/25.1,
d4096 21.9/21.9/17.6, d8192 17.9/17.3/13.1 at c1/c2/c4.

### Fair fight, both greedy (2026-10-01): dflash wins c1/c2, graphs win c4

MTP depth-2 graphed at temp 0 (`...19-37-12...md`): d0
13.1/25.5/35.7, d4096 13.2/20.0/23.3, d8192 12.1/15.3/15.4. Against
the table above: dflash's higher acceptance (τ 4.5 vs ~2.5) wins
every c1/c2 cell with fewer passes, while MTP's graphed steps scale
linearly and take every c4 cell. The crossover is the whole story:
acceptance favors the block drafter, per-step machinery favors
graphs — i.e. batched graph capture with a drafter loaded is the
remaining lane, exactly as §7 step 4 orders it.

### Batched graph capture with a drafter loaded (2026-10-02)

§7 step 4 landed: `DGPP_DFLASH2_VERIFY_GRAPH=1` replays the batched
verify as a captured CUDA graph. One static replay per row size —
8 rows for a lone slot (`=2` extends capture to it; its op stream is
the scalar C1 step, so its transcripts stay bit-exact, 12/12), 16
for two slots, 32 otherwise. Every slot's fed rows are padded to a
full 8-row block; the kernels are row-independent for compute and
skip position -1 for every state write (KV appends, GDN recurrence,
snapshots), which the padded-eager control (`=3`, same staging, no
capture) proves: it matches the eager batch exactly. Drafts, judge,
rollback and redrafts stay eager between replays; any capture
breakage latches the eager batch for the life of the server.

The one capture bug found while validating: the drafter's context
K/V feed (`dflash2_store_features`) was gated `!run.capture`, so a
replay skipped the planes' feed of the verify rows and acceptance
decayed within a request (c4: 1.3–1.6 tok/pass, p1 6–31% against the
eager batch's 4.5–6, p1 79–100%). The feed is a recorded node now;
the graph is the fastest path: 35.9–36.1 vs 35.0–35.4 agg tg at c4
and 35.9–39.0 vs 35.9–36.1 at c2 (short-prompt harness, 4x64
batches). c2/c4 graph-vs-eager transcripts sit in the same near-tie
class as eager-vs-eager (7–8/12 on the mixed-12 harness — the
residual run-to-run wobble is the pre-existing sticky GEMM dispatch,
not the graph). The multi-slot graph is now the shipped default
(`DGPP_DFLASH2_VERIFY_GRAPH` unset = `1`): it ties the packed eager
batch at 8K and leads it at short context, and is never slower — see
the default knob sweep below; level `0` opts back out to the eager
batch, and `BATCH_EAGER=1` forces it at any level. The padded layout
(`=3`) costs one extra 8-row block of GEMM work per padded slot.

### 8K profile (2026-10-02): the step is GEMV, and the draft pays 4x

`DGPP_DFLASH2_PHASES=1` splits the batched step into fed / verify /
commit / draft wall times (rolling 25-step average). At 8.3K prompt +
128 decode, 4 slots, steady state: **466 ms/pass = verify 195 ms +
draft 260 ms + host ~11 ms** (fed 4.2, commit 6.8) — and the graph is
exactly neutral here (466.4 vs 465.0 ms eager): at this length the
~600 launches are amortized, so the graph's value is short-context
and low-occupancy, not long-context.

The nsys window named for the finding was misread (corrected
2026-10-02, below): `mma_gemv_kernel<(int)8, …>` is the mma form's
**128-row** tile, not an 8-row chunk, and those 400 launches are the
target's FP8 prefill chunks, not the decode GEMMs. The decode waste
was the **BF16 drafter's** 4-row GEMV chunks (the `bf16_gemv_kernel<4>`
family: 5580 launches, ~48/pass at c4, each of the 32 stacked rows'
GEMM re-reading its weights four-row at a time). The drafter —
`z-lab/Qwen3.8-27B-DFlash2`, 5 layers, 2.9B, **BF16** (5.8 GB) — ran
every wide GEMM through `CublasLtGemm::matmul`'s kernel-only band
(`qwen_configure_gemm_rows` sets `[17, 64]` for T > 16), where m=32
splits into eight 4-row GEMV launches: **eight reads of the 5.8 GB of
draft weights per draft pass** (~46 GB of weight traffic). The target
is FP8 and took the scale-GEMM streaming mma form already (one weight
read whatever m), which is why the verify phase did not move when the
drafter's did. The c4 gap at 8K was that re-reading — MTP does not pay
it (its drafter is one layer) — plus the draft's attention over the
long KV. Before/after: MTP graphed 14.8 / 12.8 / 7.8 agg tg at
c4/c2/c1 against dflash's 12.9 / 13.2 / 12.1 (dflash won c1, MTP won
c4, c2 tied). Secondary items in the window: the target's F32 lm head
(one 5.2 ms launch per step, ~2%) and ~14 ms of cutlass Lt GEMM;
attention itself is ~1.4 ms.

### Wide-row GEMM dispatch (2026-10-02): the draft pays 1x

The structural lever, implemented: `CublasLtGemm::set_decode_mma`
gains a min-rows bound (`gemm.hpp`, `gemm_cublaslt.cpp`:
`decode_mma_min_rows`, the dispatch condition now
`m >= min && m <= max`), and `qwen_configure_gemm_rows` opts the wide
decode in — `set_decode_mma(wide_decode, 17, kMmaGemvMaxRowsPerLaunch)`
— so a stacked verify/draft batch of 17..128 BF16 rows takes the
streaming tensor-core GEMM (`mma_gemv.hpp`: the weights read **once**
for every row of the launch, 16/32/64/128-row forms) instead of the
4-row GEMV chunks. The band's edges are the numerics boundary:

- **m ≤ 16 keeps its dispatch** (the GEMV rows at 1..4, Lt at 5..16),
  so the C1 gate's single-slot batch (m=8) and the c2 batch (m=16) are
  bit-for-bit what they were — C1 stays bit-exact 12/12, eager and
  graph replay alike (re-verified 2026-10-02 on the new binary).
- The kernel-only band stays as the guard (it still throws for a shape
  neither form takes) and the fallback for shapes the mma form cannot
  take; its 64-row top extends to the mma's 128, so the c8/c16 batches
  (m=64/128, previously sixteen GEMV chunks / the Lt algorithm) read
  once too.
- mma is tolerance-equal, not bitwise: the c4 batch's transcripts move
  with it into the documented cross-dispatch class (single near-tie
  flips, as c2/c4 already were).

The band is set per pass by `qwen_configure_gemm_rows` on the shared
`gemm_` — called by the MTP target pass (`mtp_run_rows`) and the
drafter's passes (`dflash2_draft`, `dflash2_draft_batch`). The DFlash2
target verify pass (`run_rows`) runs between them and inherits the
sticky state the last call left, which is harmless: every wide pass
sets the same [17,128] band, and the narrow C1/c2 taps keep their
GEMV/Lt dispatch. Prefill (decode=false) and the narrow batches clear
it. `dsv41` keeps its existing `set_decode_mma(on)` call (min defaults
to 1). The draft's FP8 GEMMs were already the streaming form through
the scale-GEMM `mma_from_rows`, so the change lands on the BF16 sites
— which, for this lane, are the drafter's.

Measured 2026-10-02 (Qwen3.8-27B lane, new binary, same rig as above):
8.3K prompt + 128 decode, 4 slots — **the draft phase drops from 260
to 92 ms/pass** (PHASES: total 466 → 295 ms; verify 195 → 192, the
FP8 target's GEMMs were already single-read), and the same-workload
head-to-head flips: **dflash 14.9 / 13.2 / 12.2 agg tg at c4/c2/c1
against MTP's 14.8 / 12.7 / 7.6** (three clean repeats at c4:
14.8–14.9). Short context, c4: 36 → 53.8–56.0 agg tg (the step there
was GEMV-dominated: the draft's 46 GB/pass of re-reads was most of a
~60 ms step); c8 lands at 56.9 agg tg. The graph stays exactly neutral
(graph 52.3 vs eager 53.8 vs padded-eager 49.3 agg tg at c4 short).
This closes the plan §7 exit gate on the Qwen lane: per-class
end-to-end, DFlash2 now beats the best native-MTP configuration at
every concurrency tried.

Default knob sweep (2026-10-02, current binary): the shipped defaults
were re-benchmarked against every env-gated alternative, so the
out-of-box configuration is the measured-best one in every cell. The
verify-graph default in particular was corrected this pass: the shipped
default is the **multi-slot verify graph** (`DGPP_DFLASH2_VERIFY_GRAPH=1`);
the true packed eager batch is level `0` (and `BATCH_EAGER=1` forces it
at any level). The earlier "eager is the default" reading was wrong —
level 0 and level 1 both graphed the multi-slot batch, so last pass's
eager-vs-graph A/B was graph-vs-graph. Measured against the real eager
baseline, the graph ties it at 8K c4 (14.9 vs 14.8–14.9) and leads it at
short-context c4/c8 (median ~54–55 vs ~51 agg tg, inside that harness's
run-to-run variance) and is never slower; the single-slot graph (level
`2`) is the slowest c1 option (20.1 vs the scalar path's 24.2), so lone
slots stay scalar at the default:

| knob | shipped default | alternative, 8K c4 / short c4 | verdict |
|---|---|---|---|
| `DGPP_DFLASH2_VERIFY_GRAPH` | `1` (multi-slot graph) | eager `0`: 14.8–14.9 / ~51; +single-slot `2`: — / 20.1 at c1 | graph ties 8K, leads short, never slower — ship `1` |
| `DGPP_DFLASH2_DRAFT_BATCH` | `1` (batched redrafts) | `0`: 14.3 at 8K c4 | on stays default |
| `DGPP_DFLASH2_DEPTH` | unset (full block) | `4`: 14.4; `2`: 13.2 at 8K c4 | full block stays default (the long-context cap hypothesis did not pay off at 8K) |
| `DGPP_DFLASH2_WALK` | unset (per-slot top-1) | `1`: ~1.2 vs 4.4–6.0 tok/pass | top-1 stays default |

The c1 lone slot is the scalar path at every level < 2 (the 2-slot gate
in `step_batch`), so it is level-invariant: 24.2 agg tg; only level `2`
(single-slot graph) changes it, to 20.1. The c2 batch (16 rows) rides
the same multi-slot graph as c4. The env gates stay in place as
bisection/diagnostic tools (`BATCH_EAGER`, `PHASES`, `TRACE`,
`DGPP_MMA_TRACE`); nothing performance-relevant is opt-in.

## Recorded GLM-5.3 result

On 2026-09-03 at TP=4, greedy depth-1 MTP accepted 88.7% of drafts on the
recorded coherent-text workload. It produced 1.89 tokens per 42.4 ms step:
22.45 ms/token, compared with 31.3 ms/token for plain decode. The draft
layer added about 7.3 GiB per rank. These figures describe that checkpoint
and workload; acceptance fell on the post-EOS text generated with
`--no-eos`.
