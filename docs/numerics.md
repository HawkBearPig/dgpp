# Judging a numerics change

For the Qwen FP8 vocabulary head, `qwen_head_check` scores fixed teacher
tokens through the production GEMV or MMA dispatch. Its JSON manifest is an
array of `{"name": "hard", "text": "..."}` objects; explicit `ids` arrays
are also accepted for fixtures. Build the target, then use the same manifest,
checkpoint, rank count and decode capacity for both modes:

```bash
python3 - <<'PY'
import json
from pathlib import Path
texts = [("quick", "teacher_text.txt"), ("hard", "teacher_text_hard.txt"),
         ("memorized", "teacher_text_memorized.txt")]
Path("/tmp/corpora.json").write_text(json.dumps([
    {"name": name, "text": (Path("benchmarks") / filename).read_text()}
    for name, filename in texts]) + "\n")
PY
scripts/fabric_run.sh --app build-ci/qwen_head_check --fetch-logs \
    --stage-file /tmp/corpora.json --log-dir /tmp/head-gemv -- \
    --model nvidia/Qwen3.8-Flash-Next-NVFP4 --requests /tmp/corpora.json \
    --image-dir /path/to/resident-cache --fp8-head gemv
# Repeat with --log-dir /tmp/head-mma and --fp8-head mma.
python3 scripts/qwen_head_compare.py /tmp/head-gemv /tmp/head-mma \
    --output /tmp/head-comparison.json
```

Select the deployment with `DGPP_CLUSTER_CONFIG`, as for other fabric runs.
On one Spark, run the binary directly and retain its output as `r0.log` in
each run directory. Each mode repeats the entire protocol twice. The default
decode capacity is 16: captured verification covers 4/6/8/12/16 rows, with
the complete text at 16 and the first 1024 tokens at the other widths. Capacity
8 (`--decode-capacity 8`) covers 4/6/8 rows, with the complete text at 8.
The text is partitioned into equal contiguous request streams; each has a
16-token prefill, and incomplete final verification groups are excluded.
Cold short-prefill forwards sample 32 evenly spaced excerpts per text at
lengths 1/4/5/8/9/16/17, scoring every row. `--yarn` uses the shipped
factor-2 Qwen YaRN configuration. These are head checks, not long-context
retrieval or speculative-decoder quality checks.

The analyzer requires complete vocabulary shards and case counts, matching
tokens and hidden states across modes, and exact logit-hash repeatability
within each mode. It also requires unchanged logits outside the optimized
interval. It joins the vocabulary slices before evaluating the 0.02-nat
mean-NLL budget, the 1% limit on token changes exceeding 1 nat, and top-1
changes within two BF16 ulps. A failed or incomplete run exits nonzero.

The same checker accepts `--compaction off` and `--compaction on` for
fixed-token comparisons of physical and compact Qwen graph layouts. Use the
same manifest and checkpoint for both, then pass the off/on directories to
`qwen_head_compare.py`. This protocol holds the default FP8 head and RoPE
settings fixed, allocates sixteen physical slots and captures up to 64 rows:

```bash
scripts/fabric_run.sh --app build-ci/qwen_head_check --fetch-logs \
    --stage-file /tmp/corpora.json --log-dir /tmp/compact-off -- \
    --model Qwen/Qwen3.8-Flash-Next-FP8 --requests /tmp/corpora.json \
    --image-dir /path/to/resident-cache --compaction off
# Repeat with --log-dir /tmp/compact-on and --compaction on.
python3 scripts/qwen_head_compare.py /tmp/compact-off /tmp/compact-on \
    --output /tmp/compact-comparison.json
```

The sparse cases place two requests in slots 15 and 0 at one, two and four
rows per request, and five requests in slots 15/0/7/3/12 at four rows each.
The checker uses the engine's numerical compatibility rule. The compact
policy selects sixteen, twelve and six groups for the two-request cases,
respectively, and six groups for the five-request case; unused groups carry
inactive padding. The physical policy uses sixteen groups. Small physical
graphs retain their width, and wider graphs shrink only within the lowering
range above sixteen verification rows. Four- and sixteen-slot
dense cases require identical hidden states and logits across modes. All
cases require exact repeats and rank agreement. Sparse cases allow hidden
states to change with the graph shape and use the same NLL and top-1 gates
above. Four-row cases score the complete corpus; other cases use its first
1024 tokens. This checks the teacher-forced verification backbone; sampled
fallback and draft-state transitions are exercised separately by the graph
engine tests.

Kernel work is allowed to change floating-point reduction order when it buys
latency, so two builds can legitimately produce different bits. These evidence
tools judge numerical changes and size sampling; all read run directories through
`scripts/fabric_logs.py`, the shared parser for the per-rank
`[gen]`/`[tf]`/`[sample_mass]` step lines — start there when writing the next
one:

- `scripts/fabric_xcript.py REF_DIR NEW_DIR` — for ordinary generation runs:
  finds the first token where the transcripts diverge and reports the global
  top-2 logit margin there in bf16 ulps. A flip at ≤ 1-2 ulp is a near-tie;
  a flip at a wide margin is a defect.
- `scripts/fabric_logprob.py NEW_DIR REF_DIR` — the quantitative gate. Run
  both builds with `--teacher-file benchmarks/teacher_text.txt` (the app
  scores the text's own tokens instead of generating; `--stage-file` puts the
  text on every rank), and the tool joins the ranks' vocabulary slices into
  log p(token) per position and reports the text's perplexity, the per-token
  deltas with a standard error, and a PASS/FAIL against a mean-NLL bound
  (default 0.02 nat ≈ 2% of perplexity):

  ```
  scripts/fabric_run.sh --stage-file benchmarks/teacher_text.txt -- \
      --model unsloth/GLM-5.3-Flash-FP8 --text "Encyclopedia article." \
      --teacher-file benchmarks/teacher_text.txt --decode-graph
  scripts/fabric_logprob.py /path/to/new_logs /path/to/ref_logs
  ```

  Repeats with the same binary, teacher tokens and execution geometry must
  show a delta of exactly 0. Hold chunk and batch shapes fixed for this control;
  adaptive scheduling under different load can select different geometry.
  The logits are fp32 (the head's accumulators,
  unrounded), but the residual stream feeding the head is bf16, so
  single-token deltas of ~0.1 nat are normal between builds that differ in
  reduction order; the mean over the text is the signal. Three
  texts ship: `teacher_text.txt` (556 tokens, a quick look),
  `teacher_text_hard.txt` (7,331 tokens of unmemorized technical prose,
  perplexity ~10.8 — the sensitive one; use it for the verdict) and
  `teacher_text_memorized.txt` (6,549 tokens of Conan Doyle, perplexity
  1.04 — the confident regime). Expect a handful of positions per thousand
  to move by more than 1 nat between any two rounding-level builds: a
  router's top-k boundary flipped an expert there. The tool bounds the
  *rate* of those, not the worst one.

- `scripts/fabric_logprob.py NEW_DIR REF_DIR --prefill` — full-GLM prefill
  reassociations. Use `glm_dsa_forward_check --teacher-file FILE` on both
  builds; every position is scored by the full forward instead of repeated
  decode steps. Fetch every rank's log. The reader joins vocabulary slices,
  merges local argmax values with global token-ID tie breaking, and rejects
  mismatched ranks, positions, targets or target ownership. Completion
  records must account for every rank and position; missing, duplicate or
  truncated score records fail. Reports distinguish changed argmax tokens
  from gains and losses in correct top-1 predictions:

  ```bash
  DGPP_CLUSTER_CONFIG=deploy/cluster_glm-5.3_int4-int8_w4.json \
  scripts/fabric_run.sh --app build-ci/glm_dsa_forward_check --fetch-logs \
      --stage-file benchmarks/teacher_text.txt --log-dir /tmp/prefill-new -- \
      --model HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64 --resident \
      --image-dir /path/to/resident-cache \
      --teacher-file benchmarks/teacher_text.txt
  python3 scripts/fabric_logprob.py /tmp/prefill-new /tmp/prefill-ref --prefill
  ```

  The forward holds prompt-wide scratch and vocabulary logits. Size the
  teacher text for the reserved hardware and retain identical text bytes
  when comparing builds. Teacher scoring skips saved per-layer residuals
  unless `--dump-states` is also requested. Use the same existing resident
  image cache as serving when available; direct checkpoint loading may
  take substantially longer and cause rank startup skew.

- `scripts/fabric_sampling_profile.py DIR...` — the M6 sampling-width gate.
  Make one teacher-forced run per shipped text with `--sampling-profile` and
  fetched rank logs, then pass all three directories together:

  ```
  scripts/fabric_run.sh --fetch-logs \
      --stage-file benchmarks/teacher_text.txt --log-dir /tmp/mass-quick -- \
      --model unsloth/GLM-5.3-Flash-FP8 --text "Encyclopedia article." \
      --teacher-file benchmarks/teacher_text.txt --sampling-profile \
      --decode-graph
  # Repeat as /tmp/mass-hard and /tmp/mass-memorized with the other texts.
  python3 scripts/fabric_sampling_profile.py \
      /tmp/mass-quick /tmp/mass-hard /tmp/mass-memorized
  ```

  The tool requires identical contiguous measurements on every rank, reports
  top-{32,64,128,256} mass and exact-gather fallback rate at T=1/top_p=0.95,
  and chooses the smallest k at or below 1%. The serving-side sweep is
  `scripts/serve_width_sweep.sh [OUT_DIR] [WIDTHS] [MODES]` (boots the world per
  width and graph mode over `--sampling-candidates`, the same seeded requests
  every time) with `scripts/width_sweep_report.py OUT_DIR/sweep.tsv` fitting
  the cost of one fallback. The extra diagnostic gather is
  intentionally outside the production path, so these runs are not throughput
  measurements.

## Compact serving heads

`compact_head_test` compares full and compact storage on all BF16-headed
families. It evaluates both projections against an FP64 dot product over
the same BF16 weights and hidden state. It also exercises nonuniform grouped
prompts, prefix restore, MTP drafts, and fixed-token continuation; hidden and
continuation checks remain exact. `bf16_gemv_test`'s
`compact_bf16_head_matches_fp64` covers long head dimensions, GEMV, MMA and
lossless BF12 companions with output-buffer sentinels. Qwen's prefill tests
retain bitwise gates for FP8 and packed heads and use a numerical budget for
BF16; the engine tests exercise compact allocation through captured decode.

`serving_head_check` scores real serving prefill tails at lengths
1/4/8/17/64/129/257/1024. Supply the same text, model, rank count and settings
to both runs. The text needs more than 1024 tokens. The probe prints the
normalized hidden-state digest for each scored row, alongside the local
vocabulary scores consumed by the existing logprob analyzer:

```bash
scripts/fabric_run.sh --app build-ci/serving_head_check --fetch-logs \
    --stage-file benchmarks/teacher_text_hard.txt --log-dir /tmp/head-full -- \
    --model nvidia/GLM-4.7-NVFP4 \
    --teacher-file benchmarks/teacher_text_hard.txt --full --trials 256
# Repeat with --log-dir /tmp/head-compact and --compact.
python3 scripts/fabric_logprob.py /tmp/head-compact /tmp/head-full --prefill \
    --max-mean-nll-delta 0.0001 --big-delta 0.001 --max-big-delta-rate 0
```

Require identical `[head_hidden]` records across modes and ranks. The tighter
head-only logprob thresholds above isolate accumulation rounding: this change
does not alter the attention, expert routing, or cached hidden states. The
probe uses resident weights, BF16 KV, and Qwen FP8 dense weights; it evaluates
head behavior rather than a complete task benchmark. Its aggregate time
includes lazy weight loading and is not a steady-state throughput benchmark.

## Greedy transcripts and chunk shape

A cold prefill's arithmetic depends on the shapes of the chunks it walks.
The attention sites of the DSA families are not split-invariant, so the
same prompt can answer differently at temperature 0 when its chunks differ,
with or without the prefix cache. The admission controller changes those
shapes: the first prompt after a boot walks its ramp from
`prefill_initial_tokens`, and readers in flight together share the budget
and get smaller chunks each. A cache hit reproduces the walk that made the
entry, which is why a restored entry answers like the in-memory one.

A transcript comparison across boots, restarts or concurrent clients is
therefore not evidence about the change under test unless the chunk shapes
are held fixed. Pin them by setting `engine.first_token_target_ms` and
`engine.decode_gap_target_ms` to 0 in the deployment, so every tick offers
the idle budget, and run one client at a time. Otherwise compare at the
kernel level, where the row-count and split invariance gates apply.
