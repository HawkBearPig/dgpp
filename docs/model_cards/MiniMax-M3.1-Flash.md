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

The development template reserves eight request slots and a shared
786,432-token BF16 pool. Pool capacity is shared by active and cached requests;
it is not a per-request allowance at every concurrency. The checkpoint's
positional ceiling is 1,048,576, but this template does not reserve that many
tokens. This pool has loaded successfully on all four nodes with zero
process swap. Long-context quality is still pending.

## Numerical recipe and drafting

The published NVFP4 expert and MXFP8 dense weights remain compressed in memory.
The current serving path uses BF16 activations, FP32 routing and attention
probabilities, and BF16 main K/V and index-key caches. Native activation
quantization and compressed caches are separate precision choices and are not
enabled by this template. Lossless BF12 vocabulary-head storage preserves the
checkpoint's exact BF16 values; the template retains the BF16 copy as well for
wider projections and prefill.

The tested MXFP8-dense and NVFP4-expert activation controls fail the unchanged
numerical limits. Their results and scope are recorded in the
[activation precision prescreen](../minimax_m31_flash_plan.md#activation-precision-prescreen).

Every layer uses MiniMax sparse attention: 128-token blocks, independent
selection per index head, and 16 selected blocks including the current block
inside that budget. The cache retains an index key for every token. The
implementation does not pool those keys, reduce top-k, or add RoPE scaling.

`engine.mtp: true` enables the bundled five-layer DSpark model. Its learned
block proposes seven tokens; `engine.mtp_depth` must remain seven. The common
verification schedule can check fewer proposals without changing the draft
block; fresh slots still verify the full block until confidence is available.
Use `--no-mtp` for the plain-decode control. Sampled speculation uses
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

The exact index optimization and tensor-core vocabulary head pass the independent
fixed-order reference on six raw/chat corpora: 28,329 scored positions, zero
argmax changes, zero changes exceeding 1 nat, and a largest absolute mean NLL
difference below 0.00000022 nat against the 0.02-nat gate. Native repeats are exact. The
unrestricted reference's arithmetic-order failures are retained in the
[implementation record](../minimax_m31_flash_plan.md); thresholds were not
relaxed.

Four-rank API, image, video, streaming and cache tests pass, including C8
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
