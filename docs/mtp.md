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

`engine.mtp_depth` accepts 1–5 and defaults to 1 (DeepSeek-V4.1 defaults
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

## The draft head's screen (every family)

An MTP draft step ends in the vocabulary head, and on Qwen3.8-Flash-Next
that read is 636 MB per draft token in the fp8 form, 2.6 ms of a 4 ms draft
step at one node, once per draft in the chain. The draft needs only the
head's argmax (or, sampled, the top of its distribution), so
`engine.draft_head: "screen"` replaces the full read with a screen
(`src/kernels/head_screen.hpp`):

1. the hidden row is rotated by a fixed random-sign block-128
   Walsh-Hadamard transform and rounded to int8 (the rotation spreads the
   hidden state's outlier channels so the rounding noise is Gaussian);
2. a 2-bit copy of the same rotated head, built once at load with one
   four-level quantizer per vocab row, is multiplied against it with
   dp4a, one CTA per 128-row block, and each block reports its
   approximate maximum;
3. the blocks whose maximum lies within three noise sigmas of the best
   block's are candidates, capped at `engine.draft_head_blocks` (32) by
   value;
4. the candidate blocks' logits are computed exactly from the served head
   by the production row chains (fp8, bf16 or the bf12 companion), every
   other logit is `-inf`, and the pick kernels run unchanged.

The screen is one shared object (`src/kernels/draft_head.hpp`,
`DraftHead`): a family hands it the head in whatever encoding it serves
(bf16, the bf12 companion, fp8 with block or channel scales, the int8
plane form) and calls it at its draft head site; the setting is process-wide
(`draft_head_configure`), so every family and rank reads the same cap.
Native MTP (GLM-5.3-Flash, the full GLM-5.3, GLM-4.7, MiMo, Qwen3.8-Flash-Next,
the 27B's MTP mode), the 27B's DFlash2 block (its mask rows' head) and
DeepSeek-V4.1's DSpark block all draft through it; DeepSeek-V4 keeps the
screened head it already had (an fp8 copy with an exact candidate pass
around its Markov bias). The verify's head calls never take the screen, so a
greedy transcript is the exact head's whatever the screen proposes, and a
sampled request's output distribution is exact for any proposal set. What can change is
which token is drafted: on 5,571 real draft inputs (2026-10-13) the screen
returned the exact argmax on 99.3 % of draft rows and the remaining rows
cost 0.001 % of the accepted probability mass; a sampled proposal's
top-64 mass trailed the exact one by 0.7 points at cap 32. The unit test
`head_screen_test` pins the build, the prep, the screen and the select to
a host oracle bit for bit and the rescore to fp64; `qwen_decode_test
--draft-screen` runs the fixture's speculator through the masked head and
requires the plain greedy transcript. The AutoRound hybrid's int8 plane
head is screenable (the rescore reads its planes through
`kernels/packq_planes.cuh`, the bit-plane read of `kernels/packq_head.hpp`
restricted to the candidate blocks), but its template keeps the exact head
until a single-Spark leg measures the screen on it.

## The batched draft chain on GLM-5.3-Flash

Past depth 1 the Flash's fixed batch carries every slot's chain rows
(`GlmDiagnosticModel::session_graph_capture_draft_chain_batch`, 2026-10-14):
one row per request slot at the slot's block counter plus the chain index,
its token the slot's previous draft pick and its hidden the block's own
previous output row (the batched draft run's last accepted row for the
first chain row, the previous chain run's row after it), copied into the
slot's per-position hidden cache where the block's input kernel reads it;
the slots' tail rings are snapshotted before the first chain row and
restored after the last, as the scalar chain does for one slot. The picks
read rows `0 .. k-1` of the chain run's head and the feed then carries
`[next, draft_1 .. draft_n]` per slot. Until then every live request of a
depth-2 recipe replayed a scalar graph (four requests: 108–114 → 59–67 wall
tok/s on four nodes). The gates: a sixteen-round greedy batch of two slots
at depth 2 against two eager speculators and the feed, and a 64-step
sampled co-admission at depth 2 (`glm_tp_test`).

Those gates are long enough to see the lowering: a batched row is bitwise
the row alone only while every dense decode site keeps the row-independent
GEMV chunks, so the chunks now stage up to eight rows a launch
(`kernels/gemv_wide.hpp`) and the Flash lowers its whole decode batch
through them (the CHANGELOG, 2026-10-14).

The Flash's decode rows are a runtime shape since the same day (the
constructor's `decode_rows`, the serving app's `max_concurrency x (1 +
mtp_depth)`, floored at the build-time eight and capped at 32 — the DSA
select bound; `README.md` "decode rows"): a four-slot recipe at depth 2
holds a twelve-row batch with the 2- and 3-slot families under it, so
every live set batches. Before, only a 2-slot family fit the eight rows and
the other live requests replayed scalar graphs. A batch past eight rows
lowers to an eight- and a four-row GEMV chunk of the same scalar chain —
the dense weights read once per chunk, the byte cost of that exactness —
and `glm_tp_serving_mtp_depth2_twelve_row_batch_matches_independent_speculators`
holds every family's rows to the slots' own depth-2 walks (tokens, feeds
and each slot's per-position hidden cache against an eager oracle built at
the same rows; a hole, the scalar tail). The bound covers prefill rows: a
prompt of at most the recipe's decode rows prefills through the GEMV
chunks (bitwise alone and as a mixed walk's decode prefix; the same tokens
inside a longer group walk take the tiled forms, as every prompt under
the GEMV bound always has).

With batch families past depth 1 the scheduled verify depth
(`engine.mtp_schedule`) records reduced-depth batch variants, so the
Flash's batch capture compacts a feed wider than its verify rows and its
draft writes the block's full width back (as the session-core families
do: `glm_spec_gather_feed`); without it the warm capture refused with
"this family has no reduced-depth feed".
`glm_tp_serving_mtp_depth2_scheduled_batch_matches_independent_speculators`
drives the 2-slot family through a forced per-step depth of 1 and 2
against two eager speculators at the same depths (a fresh slot's first
step is held at the full block).

## The DFlash2 block drafter (Qwen3.8-27B)

DFlash2 (`src/models/dflash2/dflash2.hpp`, `src/kernels/dflash2.cu`) replaces
the MTP draft with an external block-drafter checkpoint
(`z-lab/Qwen3.8-27B-DFlash2`): five bidirectional sliding-window Qwen3 layers
over a K/V context built from the target's tapped residual streams (layers
`[5, 19, 33, 47, 61]` through the split `fc` and `hidden_norm`), two-tap
dynamic grouped convolutions around attention and the MLP, and a codebook
path selector over each mask row's top-16 candidates. The drafter shares the
target's embedding and lm head and keeps its K/V in five extra planes of the
main pool, so block tables, prefix sharing and the positional overwrite
rollback apply to it verbatim. Every target pass (prefill chunks and verify
rows alike) feeds the planes at its positions; the block forward then runs
`[bonus, mask x 7]` and proposes seven tokens.

Serve it with `engine.dflash_model`, `decode_graph` on and `mtp` off
(`deploy/cluster_qwen3.8-27b_fp8_w1.example.json`; `--no-dflash` runs
the same recipe plain on the decode graph, `--no-dflash --mtp --mtp-depth 2`
as the MTP world). On the graph worlds (the fabric, and one Spark with the
decode graph — the one-node template since 2026-10-05) the block proposal
is recorded inside the graph step — see "On the graph worlds" below; with
`decode_graph` off on one Spark the eager engine runs the same proposal a
step (the same transcripts, chat 95 against 71 ms a token). A greedy request's acceptance is the ordinary greedy
verify: the fed rows (the pending token plus the drafts) run through the
target, the accepted prefix commits, the rest rolls back, and the step
returns the tokens it decided (the accepted drafts and the verify's next
token), so the transcript equals a plain world's of the same verify width:
4/4 identical to an MTP depth-4 world (both verify five or more rows, the
streaming mma class); a 3-row MTP verify or a T=1 step differs within the
family's cross-dispatch near-tie class (`mma_from_rows` 5). A sampled
request's block is verified with the sampled MTP rule (each draft stands
with its exact probability under the request's temperature, top-k / top-p
and penalties; the residual sample ends the step): the drafts are point
masses, so the target distribution is preserved as for sampled MTP.
Requests with logprobs, a logit bias or a grammar run the plain step. The throughput line carries the drafter's per-position
acceptance in the MTP group.

### Proposal rule: the selector walk

The reference's walk (`_score_edges` + `_selector_walk_kernel` at
temperature 0): per step `l`, `scores[l][p][c] = unary[l][c] +
<pred[id(l-1, p)] * hidden[l], succ[id(l, c)]>`, walked greedily from slot 0
(`token[l] = ids[l][argmax_c scores[l][prev][c]]`), the anchor token as every
slot's predecessor at step 0. The unary term is the candidate's logit; an
earlier form of the kernel added the predecessor's, which is constant over
`c`, so the walk ignored the logits (it verified ~1.2 tokens per pass and a
per-slot top-1 stood in). Fixed, the walk beats the top-1 on every prompt
class (2026-10-03, one GB10, C1): prose 2.88 vs 2.60, code 4.92 vs 4.57,
json 6.81 vs 6.40, math 5.42 vs 5.33, chat 2.73 vs 2.51 tokens per step at
the same 164 ms/step. The block, head, top-K and walk each have host
references in `dflash2_kernels_test`.

### The batched pass and its options

Every speculating slot's verify rows ride one physical target pass
(`SessionModel::session_verify_batch`, slot-major rows with per-slot rollback
bases; `EagerEngine::step_batch`), `floor(decode_rows / 8)` slots per pass
(four at the family's 32-row ceiling). A lone slot takes the scalar path, so
a C1 transcript keeps the scalar kernel sequence. Three options, all engine
keys with the measured-best as the default:

- `engine.dflash_verify_graph` (default true): multi-slot batches replay a
  captured static verify (one 16- or 32-row graph; every slot's rows padded
  to a full 8-row block, the padding rows at position -1 which every
  state-writing kernel skips). The drafter's plane feed is a recorded node.
  A capture failure falls back to the eager batch for the server's life.
- `engine.dflash_draft_batch` (default true): one stacked block forward
  redrafts every speculating slot (row-wise GEMMs, norms, convs and RoPE
  over the stacked rows; appends, attention, head, top-K and the walk per
  slot at row offsets) — the draft weights read once per step.
- `engine.dflash_depth` (default 0 = the block): verify only the first N
  drafts per step; unverified drafts re-draft next step, so transcripts are
  exact at any value.

The drafter's stacked forwards and the taps of a wide verify run their bf16
GEMMs through the model's own streaming tensor-core form from 17 rows
(`Qwen35Model::configure_gemm_rows`): the weights read once per launch
instead of once per 4-row GEMV chunk (the 8K profile's draft phase, 260 to
92 ms/pass). Other families' GEMM instances keep the shared rule.

### Measured (2026-10-03, one GB10, greedy, exact numerics)

Against the MTP depth-2 template on the same binary (`timed_load`, 320
tokens, wall tokens/s with prefill included; the step time from the engine
counters):

| class | DFlash2 C1 | C4 | MTP d2 C1 | C4 |
|---|---:|---:|---:|---:|
| prose | 15.4 | 41.5 | 13.0 | 52.6 |
| code | 27.3 | 83.1 | 15.1 | 65.8 |
| json | 38.0 | 83.6 | 16.3 | 68.9 |
| math | 31.5 | 75.5 | 15.8 | 64.1 |
| chat | 15.1 | 47.5 | 11.8 | 56.5 |

DFlash2 steps 164 ms at C1 (2.5–6.4 tokens per step by class), 176–179 at
C2 and 200–216 at C4 against a 125 ms byte floor (24.4 GB target + 1.3 GB
head + 3.85 GB drafter); MTP depth 2 steps 151 ms at C1 (2.1–2.9 tokens).
The author's `bench_qwen35` protocol (thinking on): Q&A 22.4, Code 32.2,
JSON 35.9, Math 31.0, LongCode 25.1 tokens/s (MTP: 13.6 / 14.4 / 16.4 /
14.2 / 13.7). The remaining gap is the per-pass cost at 4–8K context (the
one-warp-per-row attention kernels, #85) and the drafter's bf16 bytes.

### On the graph worlds (2026-10-04, #92)

`GraphEngineAdapter` hosts the block proposal beside the MTP draft chain:
the verify is the ordinary 8-row recorded step (the sampled device pick
with no proposal behind the drafts — the point-mass rule; the sampling
verdict's row limit `kSampleVerdictRows` went 6 → 8), and after the commit
one recorded block forward off the device verdict replaces the draft
picks (`Qwen35Model::session_graph_capture_block_draft`, batched per family):
the anchor and the block's positions come off the verdict and the committed
device position (`dflash2_stage_block`), the drafter's layers run with its
heads, kv heads and MLP rows sharded across the ranks and two boundary folds
a layer, the mask rows go through the vocab-sharded head, each rank's slice
top-16 is staged as 6-bit digits and merged through one boundary fold
(`dflash2_topk_stage` / `dflash2_topk_merge`, the pick's wire form) so every
rank walks the same proposal, and the drafts land in the slot's next feed
(`dflash2_block_feed`) and a pinned per-slot mirror the engine reads when
the replay settles. The block attention is the split-key form
(`dflash2_block_attn_split`: 32 ranges and a combine; the serial walk cost
~2 ms a layer at a 2K context). The eager draft (the slot's open, the
sampled fallback's re-draft) runs the same helpers.

The exactness gate is the same world's MTP depth-5 world (six verify rows,
the same GEMM dispatch class as the 8-row verify): greedy transcripts
identical 4/4 at worlds 1, 2 and 4; the 1-row plain world differs within
the family's row-count dispatch class (as at world 1 against a 3-row MTP
verify). World 1's graph engine equals the eager engine 4/4 and runs the
same tokens faster (chat 70.9 against 95.2 ms a token).

**Sampled requests (2026-10-05).** The recorded walk draws each draft from
the selector's softmax at the request's temperature (`engine.mtp_draft`
`sampled` / `auto`; the reference speculator's Gumbel walk, the draw keyed
on the draft's position so every rank walks the same chain) and writes the
set it drew from as the draft's proposal; the verify then tests the draft by
the ratio rule min(1, P/Q) with the (P − Q)+ residual, whose acceptance
rate is 1 − TV(P, Q) — against P(argmax) for the argmax walk's point
masses (`greedy`). Both are exact (the output distribution is the
target's); greedy requests take the argmax walk either way. The ratio rule
runs in both sampling regimes: the truncated one (top-k / top-p / min-p,
the materialized final set) and pure temperature sampling (a request that
sends only `temperature` — most clients, the arena harness, `timed_load
--temperature 1`), where the verify priced drawn drafts by the
deterministic P(draft) rule until 2026-10-05 (acceptance Σ Q·P, never
above the overlap Σ min(P, Q) the ratio rule reaches); in the pure regime
the residual's total is 1 − Σ min(P, Q), exact when the prefix holds every
id the proposal gives mass; a row with a proposed id outside the prefix
keeps the deterministic P(draft) rule (exact for a draft drawn from
anything).

**Block verification (2026-10-05, `engine.mtp_verify: block`).** The
sampled chain decided jointly (Sun et al. 2024, "Block Verification
Accelerates Speculative Decoding", Algorithm 2) instead of token by token:
p_i = min(1, p_{i−1} P_{i−1}(X_i) / Q_{i−1}(X_i)) along the drafts, each
sub-block of length i accepted with h_i = S_i / (S_i + 1 − p_i), S_i =
Σ_x max(p_i P_i(x) − Q_i(x), 0) (h_γ = p_γ), the longest accepted sub-block
kept and the residual (p_τ P_τ − Q_τ)+ drawn after it — exact for the output
distribution and never fewer accepted drafts in expectation than the token
rule (the paper's +7–10 % block efficiency at γ = 8). The device decides the
block when every row prices its draft (otherwise the step takes the token
rule); a row whose proposal has an id outside the pure regime's prefix
counts its draft as a point mass — choices the draws never see. `sample::block_verify_from_prefixes` is the
host oracle (bitwise the device's), `sample::block_residual_complete` the
host's draw when the pure regime's residual lies in the unseen tail.
Measured on one binary: four Sparks under the arena harness (pure
temperature sampling) 68.4 against the token rule's 65.2 tok/s over two
20-run pairs each (+5 %), MT-Bench think-sampled 3.45 against 3.41 tokens
a pass, one-Spark `timed_load` at temperature 1 within noise; the drafter
templates set it.

**Eight slots (2026-10-05).** The 27B decode batch holds 64 rows (eight
drafter slots at the block's eight verify rows in one replay; 32 before,
which put eight live slots on scalar replays), its GDN verify state takes
the checkpoint-and-replay form (the snapshot writes leave the step), and
the drafter's head, top-K and hidden projection run once over every
stacked row instead of once per slot. `engine.dflash_batch_rows` bounds a
batched step's verify rows: a family past it verifies the first drafts of
every slot's block through the reduced-depth variants (exact; the drafter
still proposes the whole block), which lifts prose and chat at eight slots
and costs the sharp classes their deep block — whole blocks by default. The
numbers are in the CHANGELOG's 2026-10-05 entry and the campaign record;
at eight slots on four Sparks the MTP depth-3 world (the template's
`--no-dflash --mtp --mtp-depth 3` mode) still leads the drafter on every
class, the drafter's edge being single-stream. Measured on
MT-Bench turn-1 prompts (writing / roleplay / humanities / stem, thinking
on, temperature 1.0, top-p 0.95, top-k 20, one Spark): the argmax walk
2.80 tokens per pass, the drawn proposals 3.27–3.34 (two runs; greedy on
the same prompts 3.30), writing 2.84 → 3.3–4.0, roleplay 3.18 → 3.4–3.5,
stem 2.49 → 3.0–3.3. `timed_load --temperature 1` C1 on one Spark, prose /
code / json / math / chat: argmax 15.5 / 28.1 / 38.6 / 33.3 / 15.2 tok/s,
drawn 16.9 / 29.3 / 37.2 / 31.3 / 17.1 — the flat classes gain 9–12 %, the
sharp ones (json, math) give back 4–6 % where the drafter's distribution is
flatter than the target's (Σ min(P, Q) under P(argmax)); four Sparks at
temperature 1: 49.0 / 91.0 / 115.2 / 91.1 / 50.9 drawn at the request's
temperature, 52.2 / 84.4 / 117.0 / 98.2 / 47.6 as the recipe ships (0.7),
against the greedy drafter mode's 54.9 / 86.3 / 120.8 / 101.2 / 52.6. The draft's temperature calibrates the overlap (`engine.mtp_draft_temperature`,
the draft's temperature as a fraction of the request's; exact at any value):
the same one-Spark `timed_load` at temperature 1 read 15.5 / 28.1 / 38.6 /
33.3 / 15.2 (argmax), 16.9 / 29.3 / 37.2 / 31.3 / 17.1 (1.0), 17.5 / 29.8 /
40.0 / 34.1 / 15.2 (0.7), 16.2 / 32.0 / 42.8 / 32.5 / 15.8 (0.5), and
MT-Bench think-sampled 2.80 / 3.27–3.34 / 3.41 at argmax / 1.0 / 0.7 — a
sharper draft keeps the argmax's rate on the sharp classes while the flat
ones keep the overlap gain; the drafter recipes ship 0.7.

The drafter's five block matrices (q|k|v, o, gate, up, down) serve as block
FP8 under `engine.dflash_weights: fp8` (the checkpoint's bf16 under
`checkpoint`, the default; block-128 e4m3 encoded at load by the loader's
own encoder). Lossy for the proposals only — the target's verify is exact
whatever the drafter proposes, so transcripts and sampling distributions
are unchanged and only the acceptance can move: measured on one Spark
(2026-10-05, `raw/fp8-drafter`), MT-Bench nothink-greedy 3.53 against the
checkpoint's 3.52 tokens a pass, greedy C1 20.1 / 32.3 / 44.3 / 35.2 / 19.4
against 19.9 / 31.6 / 44.4 / 34.5 / 18.8 tok/s at 142.2 against 145.5 ms a
pass, transcripts identical 4/4; the drafter recipes ship `fp8`.

The scheduled verify depth (`engine.mtp_schedule`, the README's engine keys; DeepSeek-V4.1-Flash's confidence head introduced it)
takes the block drafter too since 2026-10-05: the selector walk writes each
draft's confidence (the chosen candidate's softmax mass at the walk's
temperature, as a logit) beside the drafts, and a greedy slot verifies only
the leading drafts whose survival justifies another row — 1, 3, 5 or 7 of
the block — through the same reduced-depth variants; sampled slots hold the
whole block. The cost model from the 2026-10-05 traces: 2.0 ms a row over a
130 / 66 / 34 ms pass on one / two / four Sparks. Four Sparks, eight slots
greedy (`raw/sched4`): 183.7 / 273.3 / 326.3 / 284.4 / 195.6 tok/s at 106 ms
a step against the campaign's whole blocks 153.5 / 270.1 / 319.6 / 260.2 /
169.7 at 117 (the depth histogram at eight slots 3:118 5:234 7:1007 steps);
C1 level at every world; transcripts identical 4/4. The drafter recipes
ship it.

Exact at every depth, alone or in a batch (2026-10-05, `raw/isolation4`,
`raw/bisect5`, `raw/verify8`): the Qwen decode step's kernels are one chain
at every row count (the head, the fp8 projections and the GDN / attention
in-projections on the streaming form from one row — the in-projections'
fp8 GEMV core below five rows was the chain a scheduled 2- or 4-row step
took, `dense_gemv_rows` defaulting to 4 — the GDN a / b on the row-group
GEMV, split-K in 256-k units with the 33–64-row form splitting too; the
tiny-fixture test `qwen35_decode_rows_invariance` holds a 1..7-row
verify's rows bitwise to the eight-row verify's), and the recorded scalar
commit takes a reduced-depth step's own rows — captured with the model's
decode rows it retracted a step that accepted every row from a snapshot
its walk never wrote. A request joined by co-tenants now reads the same text
as alone on one, two and four Sparks, from the first request after boot,
and a scheduled request's text is the same at every depth mix tried (the
template, min-depth 3, a forced-shallow schedule: 4/4 against whole
blocks); prompts started together read the
same above the 128-row lowering bound (one pinned cuBLASLt algorithm per
shape, the GDN a / b GEMV at every row count) and, for shorter prompts,
under `engine.prefill_group: false` (one prompt per walk).
The drafter templates are level (W1 C1 20.1 / 32.8 / 45.5 / 35.1 / 19.6
tok/s at 142.0 ms, C8 195.2 ms a step; W4 C1 60.0 / 92.7 / 125.1 / 105.3 /
48.8 at 47.5, C8 205.3 / 299.2 / 356.7 / 310.7 / 218.9 at 91.5; W2 C1 32.6
/ 57.3 / 80.0 / 66.3 / 34.9 at 78.9). The plain T=1 world pays the one-row
streaming form on every fp8 site (W1 8.2 tok/s at 121.2 ms against the
campaign's 8.9 at 112.7; W4 28.0 at 35.3 against 31.0 at 32) — the one-row
form's rate, about 215 against the GEMV core's 240 GB/s and worse at the
four-node shard widths, is the next kernel item.

Measured 2026-10-04 (the 2026-10-04-qwen3.8-27b / raw / drafter-tp measurements
and `raw/prefetch`, the repository's `timed_load` workload, greedy C1, the
pass time from the engine's counters), prose / code / json / math / chat:

| world | recipe | ms/pass | greedy C1 tok/s |
|---|---|---:|---|
| 1 | drafter, graph engine | 176 | 17.9 / 28.6 / 40.4 / 32.5 / 17.1 |
| 2 | MTP depth 3 (the template) | 80 | 34.0 / 40.5 / 46.4 / 42.6 / 30.9 |
| 2 | drafter (bf12, sharded) | 94 | 32.1 / 50.3 / 71.0 / 57.6 / 30.3 |
| 4 | MTP depth 3 (the template) | 46 | 57.9 / 69.9 / 79.9 / 73.3 / 50.4 |
| 4 | drafter (bf12, sharded) | 52 | 54.9 / 86.3 / 120.8 / 101.2 / 52.6 |

The drafter leads on code, JSON and math by 29–58 % at four nodes and ties
prose and chat; the MTP template's 4-row verify keeps a 6 ms shorter pass,
so on prose-like traffic the two recipes were level and MTP stayed the two-
and four-node template on 2026-10-04. With the drawn proposals (above) the
drafter also leads sampled traffic, so since 2026-10-05 every Qwen3.8-27B
template is the drafter and the MTP world it replaced is the template's
mode (`--no-dflash --mtp --mtp-depth 3` on two and four Sparks, depth 2 on
one); the benchmarks page carries the campaign on the switched templates.
The nsys node trace of rank 0
at four nodes puts the target's fp8 GEMV at 35–38 ms a step for both
recipes before the prefetch windows (6.9 GB a rank; the GB10 reads at
233.6 GB/s and the kernels reach 89–97 % of that in isolation), the 128–140
bus folds at 7–10 ms, the sharded drafter's bf12 GEMMs at 3.8 ms. The
boundary prefetch windows (#93) hide part of the folds' idle DRAM time.

## Recorded GLM-5.3 result

On 2026-09-03 at TP=4, greedy depth-1 MTP accepted 88.7% of drafts on the
recorded coherent-text workload. It produced 1.89 tokens per 42.4 ms step:
22.45 ms/token, compared with 31.3 ms/token for plain decode. The draft
layer added about 7.3 GiB per rank. These figures describe that checkpoint
and workload; acceptance fell on the post-EOS text generated with
`--no-eos`.
