# MiniMax M3.1 Flash

Development status: the implementation and deployment are still undergoing
release validation. The final serving campaign and task-quality results are
pending; the template's tuning settings may change before release.

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

The deployment template reserves sixteen request slots and a shared
2,097,152-token NVFP4 pool. Pool capacity is shared by active and cached requests;
it is not a per-request allowance at every concurrency. The checkpoint's
positional ceiling is 1,048,576. The larger shared pool provides concurrency
headroom; it does not extend that ceiling. The native pool has loaded on all
four nodes with zero process swap. The BF16 control passes retrieval, prefix
reuse and concurrent streams through 512K; native long-context quality
validation remains pending.

## Numerical recipe and drafting

The published NVFP4 expert and MXFP8 dense weights remain compressed in memory.
The deployment selects native activations and Q8KV4 attention/cache storage,
with FP32 routing. BF16 activations and BF16 main/index caches remain available
through the independent precision controls. Lossless BF12 vocabulary-head storage preserves the
checkpoint's exact BF16 values; the template retains the BF16 copy as well for
wider projections and prefill.

Set `engine.activations: "native"` for MXFP8 W8A8 dense/shared projections
and dynamic NVFP4 W4A4 experts. Independently, `engine.kv_dtype: "nvfp4"`
selects Q8KV4 attention and compressed main/index caches. These paths are
integrated but remain under release validation. They implement the native
quantized recipe for which the model was trained; differences from the BF16
control alone do not establish an accuracy regression. The full native
likelihood check passes 28,329 positions with zero argmax changes; final
task-quality and throughput results remain pending.

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
boundaries, independent controls and outstanding comparisons.

BF12 support currently covers the main vocabulary head. The embedding,
vision matrices and the drafter's BF16 matrices retain BF16 storage.

Every layer uses MiniMax sparse attention: 128-token blocks, independent
selection per index head, and 16 selected blocks including the current block
inside that budget. The cache retains an index key for every token. The
implementation does not pool those keys, reduce top-k, or add RoPE scaling.

Plain decoding is the default. The C1/C8 and wider-batch tuning controls
favor it for throughput at load; adaptive drafting helps some single-request
classes but regresses others. The idle prefill cap is 2,048 tokens, with
adaptive chunking while requests decode.

`engine.mtp: true` enables the bundled five-layer DSpark model. Limit it to
eight request slots so its complete draft blocks fit the graph. From the
launcher, use `--knobs='--mtp --mtp-depth 7 --max-concurrency 8'`. Its learned
block proposes seven tokens; `engine.mtp_depth` must remain seven. The common
verification schedule can check fewer proposals without changing the draft
block; fresh slots still verify the full block until confidence is available.
Use `--no-mtp` to return to plain decoding. Sampled speculation uses
the full proposal distribution and rejection correction.

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

In the BF16 activation/cache control, the exact index optimization and
tensor-core vocabulary head pass the independent
fixed-order reference on six raw/chat corpora: 28,329 scored positions, zero
argmax changes, zero changes exceeding 1 nat, and a largest absolute mean NLL
difference below 0.00000022 nat against the 0.02-nat gate. Repeats are exact. The
unrestricted reference's arithmetic-order failures are retained in the
[implementation record](../minimax_m31_flash_plan.md); thresholds were not
relaxed.

Four-rank BF16-control API, image, video, streaming and cache tests pass, including C8
DSpark batch isolation. Independent vision checks retain the existing
0.005 relative-RMS and 0.99998 cosine limits. The real video cases produce
bit-exact final embeddings. Kernel diagnostics and microbenchmarks do not
establish deployment throughput; final results belong in
[the benchmark tables](../benchmarks.md) after the complete shipped-template
campaign.

The clean native release and warnings-as-errors builds pass. All 381 local
CTest entries have passing runs, including the 17 optional checkpoint tests
run separately with pinned metadata. Video-enabled and video-disabled builds
pass their capability and request checks. A 24-item large-pool quality pilot
passes eight each of HumanEval, GSM8K and extraction without truncation;
the complete task-quality campaign remains pending.

The corrected confidence interface passes the independent DSpark reference,
shared-engine regression tests and CUDA memcheck. A four-rank comparison
matches all 45 adaptive/plain request pairs. The sixteen-slot plain world
matches all 125 C1/C8/C16 requests against complete eight-row references,
including the extra prompts introduced by C16, with zero swap and matching
operation streams. The final serving campaign includes C16 automatically.
