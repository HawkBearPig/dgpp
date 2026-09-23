# Issue #4: root-cause investigation plan

Objective: identify either a reproducible implementation defect or a causal
numerical/model difference, then validate a correction against the unchanged
original request. The reporter's unpublished patch is a separate review track.

## Current evidence and immediate work

The unchanged 261,120-token request still returns 5/6 correct associations on
master `c6ca191`. The independently reconstructed 261,290-token diagnostic
prefix still predicts `5` where the ledger requires `a`. Its token hash is
`fe207c412e6a70d70b00bbc3c5995488ed12ce964c4ac7eb90933ab61e82785d`.

The preceding [operator campaign](../2026-09-23-issue4-operators/README.md)
checked final-chunk attention/recurrence, dense projections and replicated
boundaries. Those replays use captured production inputs, including the GDN
state at position 260,096 and older sparse-attention cache entries. They cannot
validate how those inputs were produced.

The completed CPU experiment is `gr_replay.py`: reconstruct the attention
gated residual mixer from checkpoint weights and the preceding captured
residual. It covers 47 layers on both ranks, including layer zero's embedding.
It excludes zero-based layer 1, where PLE changes the residual first. This
closes a local coverage gap; it is not the proposed full-history comparison.
Its 94 comparisons have maximum relative L2 `1.8212e-3` and maximum
two-BF16-relative mismatch fraction `0.012109375`, within the existing
end-to-end GR mix limits (`2e-3` and `0.02`). See [the record](README.md).

## 1. Check history construction on the exact request

Prepare a capture build from the pinned baseline, keeping the original
projection call shapes and arithmetic. Run it on the same TP2 fixture and
require its response/logprobs to match the clean diagnostic control.

Record chunk identity (request, slot, absolute positions, row count), state
before/after, and logical-to-physical cache mapping for all 128 native chunks.
The checks must cover:

- GDN recurrent and convolution state: initial zero state; the next call's
  input equals the previous call's output for the same layer/request; no
  unintended reset, wrong slot, omitted row or repeated row.
- QSA cache: each logical token/pool comes from the intended projection row;
  completed earlier entries remain unchanged; compressed pools and incomplete
  tails map to the right positions. Validate values, not just the block table.
- PLE: independently derive n-gram IDs from the original token sequence,
  compare mmap gathers to the pinned table, and check its nine-row convolution
  history, including the start of each chunk. Capture its residual before and
  after injection, which was absent from the first campaign.

State continuity is an exact invariant. It is necessary but does not prove
that an update has the right numerical result. Therefore, for each investigated
stateful layer, feed its real input sequence into an independent reference
initialized at token zero. Carry the reference's own state across chunks;
never reseed it from DGPP at every boundary. Compare states and outputs at
chunk ends and retain full data around the earliest discrepancy.

Process layer input streams one at a time or consume them incrementally.
Dumping all activations across all layers would exceed the available disk
budget. Preserve manifests, compact checkpoint summaries and any discrepant
interval. Preserve a contiguous prefix when narrowing; independently thinning
the ledger changes the problem and does not establish a context-length threshold.

**Deliverable:** a table of covered layers/positions, initialization and
continuity verdicts, numerical errors, and the first unexplained discrepancy;
or an explicit account of which full-history checks passed and which remain.

## 2. Turn a discrepancy into a causal explanation

Once a chunk/layer is implicated, capture its projections, state update,
normalization, GR mixing/injection and PLE boundaries as applicable. Replay
identical inputs and weights independently. Separate expected floating-point
reassociation from an indexing, state-management, loading or arithmetic defect.
Use the relevant declared numerical contract; do not select tolerances just
to make the observation pass.

Find the smallest case that still exposes the discrepancy. Where possible,
replace only the implicated operation/state with the reference result in a
diagnostic build, leaving the rest unchanged. Check whether this changes the
failed association. A local bug may be real without causing #4; keep those
claims separate.

**Deliverable:** a failing focused test and a causal intervention, or evidence
that the discrepancy is incidental and investigation must continue upstream.

## 3. If construction and operator contracts hold, compare arithmetic

The supplied vLLM success uses W4A4 experts; DGPP uses W4A16. A hidden-state
difference between those configurations is expected and cannot itself identify
a DGPP implementation bug. Build/reference-check adapters that preserve the
pinned expert weight decoding, BF16 activations, FP8 dense representation,
normalization/gate staging, RoPE, and TP fold semantics for the comparison.
Pin the independent implementation's source/version and verify those choices
at frozen-input boundaries before interpreting an end-to-end difference.

Obtain a fresh reference control and compare layer/chunk checkpoints. Track
router choices, sparse selection, residuals and the `a` versus `5` logits.
If divergence is explained by a particular precision policy, vary that policy
alone and test both directions where feasible. Require the intervention to
explain the original request and additional retrieval controls before proposing
a default numerical change.

The reporter already tried checkpoint-dense weights (still failed), BF16 GDN
beta rounding (still failed), and FP32 QSA probabilities (same answer). Do not
repeat those switches without a new discriminating reason. The checkpoint-dense
saved verdict is available in the supplied evidence; the other controls are
reported in the issue discussion.

**Deliverable:** a matched-input implementation mismatch, or a demonstrated
precision/model sensitivity with appropriately limited conclusions. Absence
of a detected bug alone is not proof that the model is responsible.

## 4. Implement and validate the correction

Implement only after an implicated operation or integration invariant has a
failing test. The patch must make that test pass and preserve the relevant
kernel, state-continuation and TP tests. If it is a broader numerical change,
also run the repository's teacher-forced quality checks from `docs/numerics.md`.

For #4, rerun the untouched original request and oracle on fresh TP2 worlds,
with native and bounded prefill. Require all six associations correct, valid
completion, exact token identity and zero cached prompt tokens. Repeat the
run and exercise additional long-context retrieval controls to guard against
overfitting to one key. Check latency/memory if the correction changes either.

Publish the fix with its regression and retained evidence. If a confirmed
separate defect is fixed while the original request still fails, report that
explicitly and leave #4 open.

## Execution status

- Completed previously: exact free-generation reproduction, forced-prefix
  reconstruction, final-chunk operator captures, 408 dense-projection checks.
- Completed here: independent attention GR mixer replay using existing captures;
  94 comparisons, with no out-of-tolerance result under the existing mix limits.
- Completed: full-history state/cache/PLE instrumentation, exact-response
  capture, 98 arithmetic replays and independent PLE lookup audit. The audit
  found model EOS overwritten by generation stop IDs; see README.md.
- No reporter response is required. The serving EOS separation fix is being
  tested on the unchanged original request in the adjacent issue4-eos record.

GPU work remains serial on idle reserved hardware, with the production binary
and resolved configuration retained and restoration verified afterward. The
CPU replay in this record does not alter production serving.
