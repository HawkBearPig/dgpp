# MiniMax M3.1 Flash

DGPP loads [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
with its bundled `dspark/` drafter. Validation pins both to revision
`2f33a8b9ae49ce7b51c994df97166a7fd059bc2e`. The publisher describes approximately
428 billion parameters, 23 billion active per token, and text, image and video
inputs. See the [pinned upstream card](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e/README.md).

## Running

The initial deployment uses four GB10 nodes, tensor parallelism over RoCE,
and fully resident weights. The target and draft contain about 235.1 GiB of
tensor data before replication, caches and scratch. This checkpoint does not
fit resident on one Spark; a two-node deployment has not been validated.

Configure the cluster in `.env`, then download and synchronize the checkpoint:

```bash
python3 scripts/download_model.py --config deploy/cluster_minimax-m3.1-flash_nvfp4-mxfp8_w4.example.json
python3 scripts/dgpp-cluster up --config deploy/cluster_minimax-m3.1-flash_nvfp4-mxfp8_w4.example.json
```

The download includes the bundled draft. `--sync-only` reuses the head node's
download. All 73 files in the pinned snapshot have been checksum-verified on
the four development nodes.

The deployment template reserves eight request slots and a shared
2,097,152-token NVFP4 pool. Pool capacity is shared by active and cached requests;
it is not a per-request allowance at every concurrency. The checkpoint's
positional ceiling is 1,048,576. The larger shared pool provides concurrency
headroom; it does not extend that ceiling. The native pool has loaded on all
four nodes with zero process swap. The native campaign measures timing and
cache repeatability through 512K; it does not certify long-context retrieval
quality or the 1M ceiling. The BF16 control passes retrieval at all five tested
positions, prefix reuse and concurrent streams through 512K.

## Numerical recipe and drafting

The published NVFP4 expert and MXFP8 dense weights remain compressed in memory.
The deployment selects native activations and Q8KV4 attention/cache storage,
with FP32 routing. BF16 activations and BF16 main/index caches remain available
through the independent precision controls. Lossless BF12 vocabulary-head storage preserves the
checkpoint's exact BF16 values; the template retains the BF16 copy as well for
wider projections and prefill.

Set `engine.activations: "native"` for MXFP8 W8A8 dense/shared projections
and dynamic NVFP4 W4A4 experts. Independently, `engine.kv_dtype: "nvfp4"`
selects Q8KV4 attention and compressed main/index caches. These implement
the native quantized recipe for which the model was trained; differences
from the BF16 control alone do not establish an accuracy regression. The
full native likelihood check passes 28,329 positions with zero argmax
changes. Task-quality and throughput results are recorded below.

Native quantizers match the pinned SGLang/FlashInfer GPU producers byte for
byte. Native-cache attention uses the upstream per-page formula for prompts
and the online formula for decode and speculative verification. These have
different rounding boundaries: applying prefill during decode failed the
numerical gate and was corrected. Each phase preserves its own batch/chunk
consistency. Prompt and generated-state prefixes remain reusable, as in
upstream serving. A restored conversation preserves the arithmetic that
created its cache; recomputing its generated history as cold prefill can
produce different values. Snapshot validation compares restoration with
the same live conversation, rather than requiring the two upstream phases
to be interchangeable. Full upstream runtime equivalence has not been
established. The
[native precision contract](../minimax_native_precision.md) records these
boundaries, independent controls and validation scope.

A cancelled answer can retain committed state, but reuse still requires
the retained token prefix and position to match a cut in the next rendered
prompt. An arbitrary disconnect therefore does not guarantee an answer-cache
hit. Completed-answer reuse and safe cancellation accounting are validated
separately.

BF12 support currently covers the main vocabulary head. The embedding,
vision matrices and the drafter's BF16 matrices retain BF16 storage.

For the validated BF16 comparison recipe, set both `engine.activations`
and `engine.kv_dtype` to `bf16` and reduce `engine.kv_capacity` to `786432`.
The native template's 2M-token cache allocation is too large in BF16 on
four GB10 nodes. This changes activation/cache precision while retaining
the published quantized weights; it does not create a full-BF16 checkpoint.

Every layer uses MiniMax sparse attention: 128-token blocks, independent
selection per index head, and 16 selected blocks including the current block
inside that budget. The cache retains an index key for every token. The
implementation does not pool those keys, reduce top-k, or add RoPE scaling.

Native DSpark with confidence-based adaptive verification is the default,
with eight request slots. The idle prefill cap is 2,048 tokens, with adaptive
chunking while requests decode. Speculative speedup depends on proposal
acceptance, verification depth and concurrency.

The template's `engine.mtp: true` enables the bundled five-layer DSpark model.
Keep it at no more than eight request slots so its complete draft blocks fit
the graph. Its learned
block uses seven query rows `[anchor, six masks]` and predicts from all
seven rows, starting at the anchor, as in the pinned upstream
`sample_from_anchor: true` contract. Target verification consumes eight
rows `[anchor, seven proposals]`; `engine.mtp_depth` must remain seven. The common
verification schedule can check fewer proposals without changing the draft
block; fresh slots still verify the full block until confidence is available.
Use `--no-mtp` to return to plain decoding. With the template's automatic
draft policy, adaptive verification applies to greedy requests. Sampled
requests retain full-seven verification using the full proposal distribution
and rejection correction.

The four-node template enables confidence scheduling with the following
settings (shown as equivalent launcher knobs):

```text
--mtp-schedule --mtp-schedule-min-depth 1 --mtp-schedule-options 4
--mtp-schedule-row-ms 5.6 --mtp-schedule-base-ms 51.6
--mtp-schedule-lambda 0 --mtp-schedule-adapt
```

The row and base costs are 5.6 and 51.6 ms, derived from native C1 passes at
full-seven and reduced verification widths. The schedule adapts its time-value
estimate and chooses among one, three, five and seven verified proposals;
the physical draft still produces all seven. These costs are specific to
four GB10 nodes. Use `--no-mtp-schedule` to verify all seven proposals or
`--no-mtp` for plain decoding; `--no-mtp --max-concurrency 16` selects the
earlier sixteen-slot plain shape.

DSpark follows `engine.activations`: native mode uses MXFP8 inputs for
context K/V, QKV, output and all MLP projections. Context K/V shares one
quantized input across the five layers, and gate/up share their quantized
input. The feature projection, vocabulary head, Markov matrices and confidence
head retain their checkpoint BF16 operands; draft attention/cache also
remain BF16. Setting activations to `bf16` selects the projection control.

A fresh [early-position acceptance audit](../minimax_native_precision.md#dspark-early-position-acceptance-audit)
checks draft predictions and committed context separately from final target
output. Across 160 full-seven blocks, upstream rounding and an independent
backbone calculation leave almost all position-two and position-three
rejections in place. This is a bounded diagnostic, not full upstream serving
parity or a new throughput campaign.

## API and media

Text chat uses the checkpoint's template and adaptive `<mm:think>` output.
`reasoning_effort` supports the model's low, medium, high and max levels;
minimal maps to low and xhigh maps to max. The model does not support disabling
thinking through `enable_thinking` or `reasoning_effort: none`; these requests
receive an explicit field error. The benchmark clients record the effective
server-default mode when a thinking switch is unavailable.

Still images and native MP4/WebM video are supported through the chat API.
Video decoding requires the FFmpeg-enabled build; a build configured with
`DGPP_ENABLE_VIDEO=OFF` advertises no video modality. Inline video is limited
to 32 MiB; decoded input has a separate memory limit. Video windows contain
up to eight frames, preserving the checkpoint's temporal positions and
attention across all frames in a window. Pixel identity and all frame spans
participate in prefix-cache matching, including frames occurring later in
the same window.
See [image and video requests](../vision.md) for payload examples and limits.

Tool calls use the model's recursive XML representation, with nested objects
and arrays, streaming and tool-result history. Required nullable arguments
cannot be represented by its omission-based null convention and receive an
explicit schema error. Numeric strings retain their string type.

## Validation

The native phase-aware TP4 comparison passes all six raw/chat corpora:
28,329 scored positions, zero argmax changes and zero changes above 1 nat.
The largest absolute mean-NLL difference is `1.49e-7` nat and the largest
per-position difference is `1.20e-4` nat. The existing limits remain
`0.02` nat mean NLL and a 1% rate above 1 nat. Both engine passes repeat
exactly, hidden hashes match across ranks and input protocols match.
This is an independent reference using pinned upstream quantizers and
explicit arithmetic controls, not a full upstream serving execution.
The BF16 activation/cache control passes the same corpora with zero argmax
changes and a maximum absolute mean-NLL difference below `2.2e-7` nat.

The native adaptive-DSpark shipped-template campaign uses source
`a40e587a8608` and server SHA-256
`718a59bf8356b19c7d3ec63cd5bba644561cee2069ab4d3ac7fabb682ba6dbe3`.
Decode throughput is 28.8–48.2 tokens/s at C1 and 85.7–99.7 at C8;
these ranges span the five class medians. Cold prefill takes
2.176 / 7.816 / 33.341 seconds at approximately 2K / 8K / 32K tokens.
At 512K, cold prefill takes 919.523 seconds and median decode throughput
is 31.8 tokens/s. Index selection scans token keys: its work grows linearly
per decode query and quadratically over a complete cold prompt. See the
[benchmark tables](../benchmarks.md) for all concurrency and context rows,
with wall throughput reported separately. Context timing and repeatability
through 512K do not establish retrieval accuracy or validate the 1M ceiling.

Fresh C1 controls on the same binary and eight-slot configuration measure
21.5–21.7 tokens/s for plain decoding and 22.9–44.9 for full-seven DSpark.
Adaptive verification is faster in all five measured classes, at
1.33–2.23× the plain rate. This campaign replaces the earlier sixteen-slot
plain template's published rows.

| Task | Standard-cap result | Output cap | Larger-cap retry |
| --- | --- | --- | --- |
| HumanEval | 151/164 | 2,048 | Six capped; three completed and passed at 8,192, three remained in reasoning at the cap |
| GSM8K | 294/300 | 2,048 | No capped responses |
| Schema extraction | 99/100 | 512 | No capped responses |

The seven uncapped HumanEval misses contain coding errors, including an
incorrect regular expression, bounds and median indexing. The extraction
miss changes `Esme` to `EsmE`; its JSON schema and other fields are correct.
Scores retain all misses and the standard-cap verdicts. They do not claim
identical task scores to BF16 or to an upstream serving runtime.

All 225 load requests succeed, with 150 exact repeat/batch comparisons.
The separate three-repeat transcripts are stable in all five classes,
fresh plain/default transcripts match 5/5 classes, and solo/C8 isolation is
exact. All thirty timing-control replies match the adaptive sweep in text,
token count and finish reason. The 564 standard-cap quality replies and six
retries also match the earlier plain campaign exactly, including reasoning
token counts; task scores are unchanged. The API suite passes 9/9 checks.
All eighteen context
requests through 512K succeed; each bucket's three transcripts match and
all cold probes have zero cached tokens. All four campaign boots run the
recorded binary with zero swap on all four ranks and matching operation
streams. Production and head swap are restored once after validation, with
the original executable on all four ranks and successful health/chat checks.

The original campaign's DSpark timing is withdrawn: its predictions skipped
the anchor row, and its projections used BF16 activations even in native
mode. Exact final transcripts did not expose this defect because target
verification rejected the incorrect proposals. The corrected drafter's
geometry, numerical checks and fresh integration evidence are recorded in
the [implementation record](../minimax_m31_flash_plan.md).

The Release warnings-as-errors build and rebuilt targeted native tests pass,
including all four precision combinations, graph replay and generated-prefix
restoration followed by prompt extension and decode. Parallel native decode
matches its serial control bit for bit in the attention fixtures and a
256-token real-checkpoint diagnostic; CUDA memcheck and racecheck report
zero errors or hazards. Earlier feature validation includes all 381 local
CTest entries, with seventeen optional checkpoint cases supplied separately,
and video-enabled/disabled builds. Independent vision checks retain the
0.005 relative-RMS and 0.99998 cosine gates; eight real video cases have
exact final embeddings.

Native media integration passes eight video and six image cases against
plain decoding, with DSpark at C8, plus 24-image history reuse, changed-frame
invalidation and recursive streaming tools. The targeted cancellation check
passes all eight replays and both completed-answer reuse checks. Across it
and the earlier six-case diagnostic, all 32 cold/cached reply comparisons
are exact and all eight following turns reuse completed-answer state.
The earlier optional assertion demanding a cancelled-answer cache hit was
unmet; cancellation reuse has the cut-matching constraint described above.

Native NVMe validation commits 17 spills and two restores, with zero failures
or skipped spills and 46,056 restored tokens. The disk-restored continuation
matches the memory-hit continuation exactly. Both final cache boots have
matching operation streams, the expected executable and zero swap on every
rank.
