# PR 95 follow-ups: correctness and dense NVFP4 prefill

This continues the independent review of PR head
`46715829a5de6ffa9934f17b585c0aaf9dbed010` on four GB10 nodes.
The first follow-up, `d92ee25d`, widened the fused FP4 tile
from 128 to 256 rows. The follow-ups preserve native checkpoint formats
and the FP32 dot product followed by one global-scale division.

The final matched campaign closes the measured prefill gap at worlds 1, 2
and 4. NVFP4 wins 59 of 60 decode comparisons; four-node single-request
prose remains 6.3% slower than FP8 because fewer DFlash2 drafts are accepted.
The strict all-cells performance gate is therefore **not passed**. The PR
remains open pending resolution of that exception or an explicit acceptance
of it. The original production service has been restored on all four nodes.

## Review fixes

- Context lookup now runs only on greedy MTP slots. The former sampled
  path repriced only changed tokens, which biases the accepted output;
  agreement-conditioned lookup also needs a separate distributional proof.
  Sampled slots retain their original MTP proposals. The existing sampled
  lookup metric remains present and zero.
- Lookup engine tests use a controlled zero-head fixture. They require
  lookup to fire and exercise rejection, accepted prefixes, batching and
  slot reuse. Seeded sampled requests verify that enabling lookup leaves
  the sampled MTP path unchanged at tail depths 0 and 3.
- Mixed resident initialization only harvests an MTP layer when MTP is
  enabled. The DFlash templates no longer materialize an unused 710 MiB
  draft layer at world 1. A test compares actual resident allocations with
  the memory plan for both MTP settings.

## Kernel and numerical checks

The 256x128x64 fused tile now uses eight warps, each covering 64x64 output
values. Each thread fetches 32 FP4 codes and applies the two 16-code scale
groups. Two activation stages and two decoded weight tiles use 96 KiB of
shared memory. The ascending-K FP32 accumulation and final division are
unchanged; the production decode kernel remains selected below 1024 rows.

The mixed fixture reference chain now also covers 3072-token prefills.
Tensor-parallel tests cover the FP8 fixture and both short and 2304-token
mixed forwards at worlds 2 and 4 against world 1. Their latency transport
slot holds a complete 1.125 MiB fold, so this numerical test does not
accidentally become a bulk-transfer timeout test.

A real-checkpoint probe scored 19,456 positions: overlapping first/last
4096-token windows of the hard and memorized teacher texts, plus a
3072-token stress input made by repeating the short teacher text. All
full-vocabulary per-row logit hashes matched the preceding 16-warp kernel;
mean/max NLL deltas and top-1 changes were zero. A separate process repeated
4096 positions bitwise. See [comparison](numerical-t256-comparison.json)
and [input manifest](numerical-manifest.json). This is a measured sample,
not a proof over every input.

A BF16-output cuBLAS bridge was also evaluated and rejected. It improved
speed but changed real-checkpoint numerics: the repeated-text case moved
mean NLL by +0.0769 nat/token, with 23.99% of positions moving by more than
one nat and 550 top-1 changes outside the two-ULP criterion. Synthetic
fixtures alone did not expose this. The [rejected comparison](numerical-comparison.json)
is retained; none of that bridge is in the implementation.

CUDA memcheck reported zero errors and racecheck reported zero hazards.
The full Release build passed before the suite. Of 215 non-checkpoint
CTest entries, 211 passed, two MiMo tokenizer/template cases skipped for
an absent checkpoint, and two bus cases failed because the serving site's
single-lane setting could not satisfy their explicit dual-lane assertions.
Both bus cases passed when rerun with the head node's two active NICs,
`rocep1s0f0 roceP2p1s0f0`, GID indices `3 3`: 213 passing entries and two
skips in total. No source change was required. The initial failure remains
in [validation status](validation-status.json); the corrected run is in
[bus retest](bus-retest.json).

The first four-node serving check completed 512/2K/8K/32K cold-prefill
buckets, three repetitions each, with matching rank operation streams,
identical deployed binary hashes and zero process swap on all four nodes.
The 8K median is 4594 ms, still 0.3% above the preceding FP8 measurement;
this preliminary run does not establish the final performance gate.
See [request measurements](eight-warp-w4/prefill.json).
This was followed by the paired-write candidate and base integration below;
the final matched serving matrix below determines the performance gate.

## Paired output writes

The next candidate packs adjacent BF16 columns into one conversion and
one aligned 32-bit store. Unaligned views, odd row strides and a final odd
column keep scalar stores. A sentinel test exercises those cases across
3073 rows, including the last partial M tile. For walks of at least 3072
rows, three M tiles per cache group replace four; shorter walks retain
four. This changes only block order and output writes, not a dot's
arithmetic.

All 18 targeted mixed-reference, tensor-parallel and FP4 kernel entries
passed after integrating the current base branch. The same 19,456 real
checkpoint positions and the 4096-position process repeat still match the
preceding 16-warp kernel's full-vocabulary hashes exactly; see the
[paired-write comparison](numerical-paired-comparison.json).

## Operational evidence

Raw logs, executable hashes, per-request JSON, numerical TSVs and rejected
experiments are retained locally in
`/home/stephen/dgpp/pr95-followups-20261011/`. The earlier review and cache
cleanup audit are in `/home/stephen/dgpp/pr95-review-20261010/`.

## Base integration and full validation

The branch includes master at `38d3af69485158a926eefb7eb46fbac475295df4`,
including the shared GLM kernel changes and startup diagnostics. The full
Release build completed before the serial 241-entry CTest run. It recorded
234 passes, six absent-checkpoint tokenizer/template skips and one stale
HTTP-default assertion inherited from master. That assertion now checks
`0.0.0.0` by default and an explicit localhost override. All three affected
Python suites passed on retest: 235 passing entries, six skips and no
unresolved failures. See [suite log](merged-suite.log),
[retest](binding-retest.log) and [resolution](full-suite-resolution.json).
The FP4 memcheck and racecheck runs also reported zero errors/hazards.

The paired-write four-node smoke measured median cold-prefill latencies
of 367.6 / 1261.7 / 4545.5 / 18955.6 ms at approximately 512 / 2K / 8K /
32K tokens. The preceding same-prompt FP8 medians were 415.8 / 1292.8 /
4566.0 / 19046.4 ms. These are preliminary comparisons: the small long-
prompt margin needs confirmation by the final matrix. All four paired-
write rank operation streams matched, deployed executable hashes agreed,
and process/cgroup swap stayed zero at the before/after checks.
See [NVFP4 requests](paired-w4/prefill.json),
[FP8 requests](eight-warp-fp8-w4/prefill.json) and
[rank streams](paired-w4/opstreams.json).

## Cache preparation and cleanup

The initial review cached `unsloth/Qwen3.8-27B-NVFP4` at revision
`f0b7c9e722f5565102fff8481c99e4d86ae099c7` on all four nodes. The indexed
checkpoint contains 23,417,592,488 bytes (21.8 GiB). The cache was downloaded
once, copied with checksums to the peers and verified there. DFlash2 at
revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` is also cached on every
node.

The cleanup removed only resident images tied to absent checkpoints or
superseded resident formats, after checking provenance, exact file identity
and that no process held them open. This reclaimed 303.89 GiB on each of
nodes 1 and 2 and 539.95 GiB on each of nodes 3 and 4: 1.65 TiB total.
It did not remove model checkpoints. The [per-node audit](cleanup-audit.json)
found no unexpected missing files or changed retained metadata. Its free-
space figures describe that earlier cleanup, before this follow-up run.

The [saved production binary comparison](production-fp8-comparison.json)
checks that the FP8 comparison path did not slow down in the PR build:
matched four-node 512/2K/8K/32K medians differ by less than 0.5%. Older
published bucket labels had different actual prompt lengths; in particular,
this campaign crosses 4096-token chunk boundaries that some historical
prompts did not. Prompt hashes, actual counts and zero-cache checks define
the matched comparison here.

The scaled FP4 weight needs at most six significant bits, not five:
`1.5 × 1.875 = 45/16` reaches that bound. The kernel comments and format
plan now state six. F16 and BF16 still represent every such finite product
exactly; this corrects the explanation and does not change arithmetic.
These comment-only edits follow the measured `c9a98ac7` build.

## Final matched serving campaign

Each world ran FP8, NVFP4 and FP8 again, with a fresh server per leg and
one world active at a time. All nine legs used the same Release executable
from `c9a98ac7`, SHA-256
`26e470006ae96ccc4716b10f1d4ce1253caa10e1ab6d240a105c3ce1ab785ffd`.
The subsequent source changes are comments only; their exact diff is in
[post-build comments](post-build-comments.patch). Templates use DFlash2,
eight slots, 256K BF16 KV and disabled context lookup. No CPU compilation
ran alongside GPU timing.

The decode procedure is `timed_load.py`, greedy, 256 completion tokens,
prose/code/JSON/math/chat, concurrency 1/2/4/8, three repetitions. Cold
prefill uses three repetitions at approximately 512/2K/8K/32K tokens, seed 7
and tag `pr95-matched`. The audit checks prompt hashes and actual counts
across formats, zero cached tokens, HTTP success and completion counts.
All API checks passed. Every world's operation streams matched, every live
rank had the expected executable hash, and process/cgroup swap stayed zero
with cgroup swap disabled at the before/after checks.

The following improvements compare NVFP4 against the **faster** of the two
FP8 medians for each cell, rather than choosing a favorable baseline.

| Nodes | 512-token prefill reduction | 2K | 8K | 32K | Decode throughput change across 20 cells |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 15.45% | 5.39% | 1.10% | 0.68% | +2.81% to +29.85% |
| 2 | 12.32% | 2.97% | 0.53% | 0.23% | +6.15% to +22.05% |
| 4 | 11.03% | 1.74% | 0.27% | 0.08% | -6.26% to +29.43% |

Every paired prefill repetition was faster, but the long-prompt margins
are small. Three repetitions establish these measurements, not a universal
speed guarantee. The [machine-readable audit](matched-audit.json) retains
all cells and baseline drift; its `performance_gate` remains false because
of the one decode regression. [Benchmark tables](../../../docs/benchmarks.md)
include the exception and link the raw request metrics.

The final [Nsight profile comparison](profile-comparison.json) uses the same
8,353-token uncached prompt as the original PR profile. Across 336 FP4
GEMMs, kernel time fell from 4432.22 to 3055.60 ms, a 31.06% reduction.
Instrumented request time was 8900.4 versus 7438.5 ms. The uninstrumented
matched campaign above is the serving-performance evidence.

## Remaining decode exception and rejected experiments

For four-node C1 prose, FP8 emits 255 timed decode tokens in 88 steps,
about 47.35 ms per step and 61.20 tokens/s. NVFP4 needs 105 steps at about
42.33 ms per step, reaching 57.37 tokens/s. NVFP4's step is faster, but its
2.43 committed tokens per step trail FP8's 2.90. The formats generate
different greedy text, so draft acceptance need not agree.

The [draft-policy sweep](draft-policy-comparison.json) tested full-block
verification, adaptive row costs from 0.5 to 32 ms, fixed depths 1/3/5 and
the original BF16 drafter weights. All greedy transcripts were unchanged.
None closed the exception; the best result was 58.44 tokens/s. The shipped
drafting settings remain unchanged.

Decode-tile experiments also retained exact kernel outputs. Narrowing a
64-row tile was slower. A 16-row tile saved about 25% in a cached-weight
microbenchmark but only about 2% in serving, reaching 58.54 tokens/s.
See [serving trials](narrow-serve-probe-status.json). These exploratory
link-wrapper builds were not substituted into the final measured matrix
or committed as production code.

The boundary-prefetch path does not currently add NVFP4 MLP weights. An
experiment added native payloads, scales and global factors to its window.
The existing 20 MiB/light default slightly slowed this case; 4/8/12 MiB
windows, full-rate reads, prefetch instructions and page touches also failed
to close the gap. [Window/rate trials](prefetch-tuning-status.json) and
[form trials](prefetch-form-status.json) retain the results. The experiment
was reverted. Skipping these reads is not a numerical omission: the
matmul itself loads every required weight.

## Output and quality checks

All twelve short greedy transcript cases (four per world) match the
original PR head; see [transcript regression](transcript-regression.json).
The full-vocabulary numerical probes described above separately constrain
the prefill optimization.

The four-node quality run used 100 GSM8K and 100 extraction cases, no
thinking, maximum 4096 output tokens, concurrency four and seed 20261010:

| Model | GSM8K | Extraction | Truncated GSM8K replies |
|---|---:|---:|---:|
| FP8 | 96/100 | 100/100 | 1 |
| NVFP4 | 95/100 | 100/100 | 0 |

Four GSM8K failures were shared; NVFP4 additionally missed case 62.
This small sample does not establish broad quality equivalence between
the quantized checkpoints. The [quality comparison](quality-comparison.json)
and per-leg summaries retain the scores and failure metadata. Raw response
files are retained locally and hashed in [raw manifest](raw-manifest.json).

## Final operational audit

The [final cache audit](final-cache-verification.json) verifies the indexed
shards, tensor maps, safetensors headers and file lengths for NVFP4 and
DFlash2 on every node. The NVFP4 index's `total_size` includes its headers;
its tensor payload is 23,417,339,744 bytes. This final check supplements the
content checksums verified during the original download and peer transfer.

At 04:02 UTC the original `Qwen/Qwen3.8-Flash-Next-FP8` service was restored
on port 18080 using the saved production executable, not the PR build.
The resolved deployment matches the original exactly. All four live
executables have the original hash, process and cgroup swap are zero, and
an HTTP smoke request returned the expected answer. See
[restoration audit](final-restoration-audit.json).

Reproduction scripts and command receipts accompany the nine legs. Raw
decode text and per-token client timing arrays are omitted from the compact
record; request hashes, response hashes, aggregate metrics and source-file
hashes are retained. Run GPU campaigns only on reserved, idle hardware;
the restored production service is currently using the cluster.

After reserving idle hardware, use a new output directory:

```bash
PR95_OUT=/path/to/empty/results PR95_BIN=/path/to/dgpp-serve \
  python3 benchmarks/results/2026-10-11-pr95-followups/final_matrix.py
```

The script deliberately exits unsuccessfully if any performance cell fails
the strict gate. The [final source audit](final-source-audit.json) confirms
that the exploratory decode changes were reverted and only line comments
differ from the measured executable's source.
