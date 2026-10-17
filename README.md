# DGPP

DGPP is a C++/CUDA inference engine for NVIDIA DGX Spark (GB10) systems.
It serves GLM-5.3-Flash, full GLM-5.3, Qwen3.8-Flash-Next, GLM-4.7 and
DeepSeek-V4.1-Flash through an OpenAI-compatible HTTP API, with tensor
parallelism over RoCE for multi-node deployments. Supported configurations
use one, two or four nodes, depending on the model and its memory requirements.

The repository contains the CUDA kernels, RDMA collectives, tokenizer,
Jinja chat-template interpreter, scheduler, prefix cache and HTTP service.
It uses the CUDA runtime, cuBLASLt and libibverbs. Rank 0 coordinates
requests through an admission journal; peers check their operation streams
against it throughout a run.

## Supported models and configurations

These serving configurations have deployment templates and recorded
measurements. World size is the number of ranks, with one DGX Spark per
rank; world 1 runs locally, while worlds 2 and 4 use tensor parallelism
over RoCE. Each quant links to its specific Hugging Face model card.

| Model | Quant / Hugging Face model card | World sizes | Example configuration |
|---|---|---|---|
| GLM-5.3-Flash | [unsloth/GLM-5.3-Flash-FP8](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8) | 4 | Copy the [base template](deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json) to `cluster_glm-5.3-flash_fp8_w4.json`, set `model` to the linked FP8 repository and lower `engine.kv_capacity` to 393216 (the FP8 experts are 31 GiB larger per rank; the startup memory plan refuses the base template's context) |
| GLM-5.3-Flash (hybrid) | [HawkBearPig/GLM-5.3-Flash-NVFP4-FP8](https://huggingface.co/HawkBearPig/GLM-5.3-Flash-NVFP4-FP8) | 2, 4 | [Two nodes](deploy/cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json), [four nodes](deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json) |
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | 2, 4 | [Two nodes](deploy/cluster_qwen-3.8-flash-next_fp8_w2.example.json), [four nodes](deploy/cluster_qwen-3.8-flash-next_fp8_w4.example.json) |
| Qwen3.8-Flash-Next | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | 1, 2 | [One node](deploy/cluster_qwen-3.8-flash-next_nvfp4_w1.example.json) (256K shared KV pool, 4K busy/idle prefill), [two nodes](deploy/cluster_qwen-3.8-flash-next_nvfp4_w2.example.json) (mapped n-gram table, dense projections FP8 at load) |
| Qwen3.8-Flash-Next | [Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN](https://huggingface.co/Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN) | 1 | [One node](deploy/cluster_qwen-3.8-flash-next_autoround-int4_w1.example.json) (AutoRound int4 g128 experts and an int8 g128 head served in their packed form, the n-gram table from the FP8 release's shards via `engine.ngram_table_model`; the template turns on the `prefill_bf16_partials` and `prefill_fp8_gemm` levers, whose measured quality cost is inside the default chain's band — see [accuracy and correctness](#accuracy-and-correctness)) |
| Qwen3.8-Flash-Next | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | 1, 2 | [One node](deploy/cluster_qwen-3.8-flash-next_nvfp4-radixark_w1.example.json) (tuned: 4K prefill chunks with matching prefill budgets, MTP depth 2), [two nodes](deploy/cluster_qwen-3.8-flash-next_nvfp4-radixark_w2.example.json) (same NVFP4 format and engine configuration as the NVIDIA release) |
| Qwen3.8-Flash-Next | [local-inference-lab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | 1 | [One node](deploy/cluster_qwen-3.8-flash-next_nvfp4-qad_w1.example.json) (the quantization-aware-distilled export served in its shipped forms: the MXFP8 dense stack and shared experts, the NVFP4 n-gram table resident on the Spark, F32 norms and weight-only NVFP4 draft experts; [card](docs/model_cards/Qwen3.8-Flash-Next-NVFP4-QAD.md)) |
| Qwen3.8-Flash-Next | [kaushikvira/Qwen3.8-Flash-Next-swift15-nvfp4-dgpp](https://huggingface.co/kaushikvira/Qwen3.8-Flash-Next-swift15-nvfp4-dgpp) | 2 | [Two nodes](deploy/cluster_qwen-3.8-flash-next_nvfp4-swift15_w2.example.json) (community NVFP4 conversion of the Swift 1.5 fine-tune, mapped n-gram table and BF16 fused MTP draft experts; [serving notes](docs/model_cards/Qwen3.8-Flash-Next-Swift1.5-NVFP4.md)) |
| GLM-4.7 | [nvidia/GLM-4.7-NVFP4](https://huggingface.co/nvidia/GLM-4.7-NVFP4) | 4 | [Four nodes](deploy/cluster_glm-4.7_nvfp4_w4.example.json) |
| GLM-5.3 | [HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64](https://huggingface.co/HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64) | 4 | [Four nodes](deploy/cluster_glm-5.3_int4-int8_w4.example.json) |
| GLM-5.3 | [HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128](https://huggingface.co/HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128) | 4 | [Four nodes](deploy/cluster_glm-5.3_nf4i8_w4.example.json) (272K FP8 K/V pool) |
| GLM-5.3 | [HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128](https://huggingface.co/HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128) | 4 | [Four nodes](deploy/cluster_glm-5.3_mixed346_w4.example.json) (272K FP8 K/V pool) |
| DeepSeek-V4.1-Flash | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 4 | [Four nodes](deploy/cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json) |
| MiMo-V2.6-Flash | [XiaomiMiMo/MiMo-V2.6-Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | 2, 4 | [Two nodes](deploy/cluster_mimo-v2.6-flash_mxfp4-fp8_w2.example.json), [four nodes](deploy/cluster_mimo-v2.6-flash_mxfp4-fp8_w4.example.json) |
| DeepSeek-V4-Flash | [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 2, 4 | [Two nodes](deploy/cluster_deepseek-v4-flash_mxfp4-fp8_w2.example.json), [four nodes](deploy/cluster_deepseek-v4-flash_mxfp4-fp8_w4.example.json) |
| Qwen3.8-27B | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 1, 2, 4 | [One node](deploy/cluster_qwen3.8-27b_fp8_w1.example.json) , [two nodes](deploy/cluster_qwen3.8-27b_fp8_w2.example.json) and [four nodes](deploy/cluster_qwen3.8-27b_fp8_w4.example.json) (the DFlash2 block drafter on every world) |
| Qwen3.8-27B NVFP4 | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | 1, 2, 4 | [One node](deploy/cluster_qwen3.8-27b_nvfp4_w1.example.json) , [two nodes](deploy/cluster_qwen3.8-27b_nvfp4_w2.example.json) and [four nodes](deploy/cluster_qwen3.8-27b_nvfp4_w4.example.json) (the mixed release: NVFP4 MLPs, channel-FP8 attention and head; [card](docs/model_cards/Qwen3.8-27B-NVFP4.md)) |

The Qwen NVFP4 templates select streaming MMA for the FP8 vocabulary head
with `engine.fp8_head: "mma"`, following matched one- and two-Spark
real-checkpoint numerical validation,
including the YaRN template. `--fp8-head gemv` restores the previous head
path. See [operation and numerical constraints](docs/operations.md).

The Qwen NVFP4 templates use `engine.ngram_table: "mmap"` to read the
n-gram table from NVMe, `engine.decode_graph: true` for resident graph serving,
and `engine.dense_weights: "fp8"` to encode dense projections at load. On two
Sparks, mapping saves 23.84 GiB per rank while staying within 3.6% of resident
decode throughput and 2.9% of resident prefill time in the matched campaign.
Use `--dense-weights checkpoint --fp8-head gemv` to retain the checkpoint's
BF16 dense stack. The quantization-aware-distilled export's template keeps
its 26.8 GiB NVFP4 n-gram table resident instead (`engine.ngram_table:
"resident"`, a 128K pool): its dense stack ships as MXFP8 and is served as
shipped, so `engine.dense_weights` reaches only the BF16 remainder there.
See the [single-node guide](docs/qwen38_single_spark.md) and
two-node benchmark.

The GLM-5.3 hybrid takes the main-stack routed experts from
[dabsLabs](https://huggingface.co/dabsLabs/GLM-5.3-Flash-NVFP4) and the
remaining tensors, including MTP, from
[Unsloth](https://huggingface.co/unsloth/GLM-5.3-Flash-FP8).
The GLM-5.3-Flash templates use
`HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`. The setup command below downloads it
once on rank 0 and syncs the selected snapshot to peers. To use the FP8 release
instead, set
`model` to `unsloth/GLM-5.3-Flash-FP8`.
To reproduce the hybrid from its sources, use
[the composition tool](tools/compose_nvfp4_hybrid.py) and
[the NVFP4 notes](docs/nvfp4_plan.md).

MiMo-V2.6-Flash is served as shipped — MXFP4 routed experts, fp8 block-128
dense projections, BF16 `o_proj` / head / `eh_proj` (resident in their
lossless 12-bit form under `engine.bf16_weights: "bf12"`) — with its hybrid
sliding-window (128, sink-biased) / global attention, one MTP draft layer
and the checkpoint's own chat template and `<tool_call>` format. The vision
and audio encoders in the checkpoint are not loaded; text prompts only. The
two-node template keeps its K/V cache in the fp8 row form
(`engine.kv_dtype: "fp8"`, 58 KiB per token per rank against 109 in BF16)
for a 256K-token pool; `--kv-dtype bf16` restores the BF16 cache. Decode
runs the attention as one launch per layer with the residual add fused
into each norm; prefill attention is query-tiled on the tensor cores (each
K/V tile staged — and under the fp8 cache dequantized — once per 64 query
vectors). See the [model card](docs/model_cards/MiMo-V2.6-Flash.md) and
[plan](docs/mimo_v26_flash_plan.md). Native blocks 0/1/2 MTP3 and prefill
work reductions are opt-in; see [MiMo operations](docs/operations.md#mimo-native-mtp-and-prefill-options)
for settings, memory implications and measured workload tradeoffs.

DeepSeek-V4-Flash (the 0731 release) is served as shipped — MXFP4 routed
experts, FP8 block-128 attention projections and shared expert, BF16
compressors, router and head — with its 128-token sliding window, the
ratio-4 (indexed, 512 entries per query) and ratio-128 compressed caches,
token-table routing on the first three layers, two-pass hyper-connections
and the DSpark block draft. Both templates serve the model's full 1M-token
context and schedule the verify depth from the draft's confidence
(`engine.mtp_schedule`): a pass verifies as many of the block's five drafts
as their survival pays for. A sampled request's drafts are the draft head's
most likely tokens (`engine.mtp_draft`), exact for the output distribution,
and follow the same schedule (`engine.mtp_schedule_sampled_scale`).
Prefill resumes at compressor-aligned cuts and groups compatible readers
into shared forwards. The common scheduler admits later arrivals alongside
active decoders and adapts its work budget to measured pass cost.
[Execution tests](docs/testing.md) cover chunking, snapshots and mixed batches.
Measured on four Sparks (2026-10-02): 65–126 tokens/s for one request by
prompt class (44 without the draft), 134–186 tokens/s across six, cold
prefill of a 2K prompt in 1.36 s, and on llama-benchy's sampled pp2048 /
tg128 64 tokens/s for one request and 91 / 121 in total at two and five
concurrent requests; on two Sparks 40–73 tokens/s for one request and
68–101 across four ([benchmarks](docs/benchmarks.md)). See the
[model card](docs/model_cards/DeepSeek-V4-Flash.md).

The linked templates select MTP or DFlash with deployment-specific tuning.
Plain decode, alternate speculation, cache formats and slot counts are launcher
knobs on these templates; [the deployment catalogue](deploy/README.md) maps the
supported shapes to their flags. [The benchmark tables](docs/benchmarks.md)
record the modes measured for each deployment.

Request preparation runs off the HTTP loop in a fixed pool sized to
`engine.max_concurrency`. Ready requests proceed in completion order; queue
and payload limits bound retained work. See [request preparation](docs/configuration.md#request-preparation)
for capacity, overload behavior and CPU/memory considerations.

## Highlights

- **Tensor-parallel resident serving** on four nodes: each rank holds its
  slice of the model resident on the GPU and typically boots from a per-rank
  image cache in 15–30 s, depending on the model.
- **Graph-based decode**, including collectives and MTP speculative
  decoding, with scalar or batched graphs selected for the active requests.
  Greedy MTP produces the same tokens as plain decode; sampled MTP
  preserves the target distribution.
- **Faster DeepSeek decoder selection**: parallel scoring and exact radix
  selection reduce the time spent choosing attention entries, especially at
  long context. Scores, tie rules and selected entries are preserved; existing
  recipes use the improvement automatically.
- **Row-aware tensor-core execution and grouped prefill**: dense kernels select
  their lowering from the active row count, and queued cold prompts can share a
  forward pass while retaining request-local attention and state.
- **Lossless 12-bit BF16 weights (`engine.bf16_weights`)**: the BF16 matrices
  a decode step streams are kept in a 12-bit form — the sign and mantissa
  byte plus a 4-bit exponent code per weight, exact side tables for the rare
  outliers — and the kernels rebuild the exact BF16 bits in registers. Outputs
  are bit-identical (greedy transcripts do not change) from 25% fewer bytes:
  +6–12% single-stream decode on GLM-5.3-Flash, GLM-4.7, the full GLM-5.3 and
  Qwen3.8-Flash-Next-FP8. The 12-bit form can also be the only resident one,
  which puts the model below its checkpoint's footprint.
- **Qwen3.8-Flash-Next AutoRound int4/int8 hybrid on one Spark**: the
  GPTQ-layout checkpoint (int4 group-128 experts, an int8 group-128 head)
  is served in its packed form with no dequantized copy; the head is read
  as bit planes with a provable argmax bound, so every greedy step touches
  75 % of its bytes for a bit-exact pick, and MTP runs at depth 3. Decode
  45–72 tok/s single-stream and 77–119 tok/s at four requests, prefill
  0.55 ms per token from 8K to 32K on the exact chain and 0.52–0.53 with
  the template's two prefill levers, with the decode step flat from 3K
  context up.
- **Model-specific prefill paths**: packed int4/int8 tensor-core prefill for
  full GLM-5.3, tiled QSA prefill for Qwen (W4A4 expert prefill is opt-in
  pending the gate in issue #68),
  and bounded grouped prefill for DeepSeek-V4.1-Flash. Qwen can optionally
  yield between prefill chunks so active decodes continue making progress.
- **Opt-in 512K context for Qwen3.8-Flash-Next** with `engine.rope_scaling`
  (YaRN): the [two-Spark NVFP4 template](deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json)
  supports a 524288-token request ceiling, with 5/5 retrieval probes passing
  at both 261K and 522K prompt tokens; the same ramp serves **Qwen3.8-27B**
  through its [one-node NVFP4 template](deploy/cluster_qwen3.8-27b_nvfp4_w1_yarn512k.example.json)
  (3/3 needles at 524 288 on one Spark and on four, 2026-10-15). See the
  [release-check procedure](docs/qwen_yarn_release_check.md).
- **Adaptive DSpark verification**: DeepSeek uses confidence-scheduled draft
  depth, including a batch-aware rule and an adaptive value of decode time.
- **Exact prefix caching**: matching token prefixes can reuse a stored
  session snapshot while preserving the cold-prefill result.
- **OpenAI-compatible text generation**: Chat Completions and legacy
  Completions, streaming, constrained tool calls, `response_format` with
  supported JSON schemas, `reasoning_content`, logprobs, `stop`, `n`,
  `logit_bias` and usage details; plus model, health and metrics endpoints.
  Assistant history accepts Anthropic-style thinking parts forwarded through
  LiteLLM.
  See the [API compatibility profile and live prefill metrics](docs/openai-compatibility.md)
  for supported options and model-dependent limitations.
- **Image inputs**: PNG/JPEG/WebP data URIs in Chat Completions for GLM-5.3-Flash
  and Qwen3.8-Flash-Next, using each checkpoint's native vision encoder.
  Multiple images, streaming and MTP work together, with image-aware prefix
  caching and resumable prefill; see [image inputs](docs/vision.md) for each
  family's preprocessing geometry, examples and memory requirements.
- **Deterministic across ranks**: admissions journaled from the head, every
  tick's operation-stream digest checked on every peer, and all ranks'
  complete streams compared at shutdown.
- **Process-failure detection**: a rank process exiting fails the service
  within seconds. Silent node loss is detected by the bus watchdog.
- **Deployment and monitoring**: a shared cluster config, versioned
  releases, periodic throughput logs and JSON counters at `/metrics`
  (also available at `/v1/metrics`), including
  [decode graph batch and padding counters](docs/operations.md#decode-graph-batch-counters)
  and [MTP acceptance counters](docs/openai-compatibility.md#speculative-decoding-counters).
  Startup checks the memory plan before allocation. Cache capacity is configurable,
  with BF16, FP8 or FP4 latent storage for GLM-5.3.
  All families size serving logits for decode and MTP verification, projecting
  only prefill tails. At a 2048-row prefill budget this saves about 301 MiB per
  rank for GLM-5.3-Flash on four nodes.

## Accuracy and correctness

dgpp's default serving chain computes what the checkpoint specifies, and it
computes it the same way every time: a greedy request returns the same
tokens run after run, across ranks (every admission is journaled from the
head and every tick's operation stream is digest-checked on every peer),
and with speculative decoding on or off (greedy MTP produces plain decode's
tokens; sampled MTP preserves the target distribution). Performance work
holds the cache, rank and speculative-decoding contracts exact. Representation
changes such as the packed int4/int8 tensor-core prefill, the 12-bit BF16
weight form, the bit-plane vocabulary head and the n-gram prestage have
bitwise regression gates. Accumulation-order changes, including compact BF16
prefill heads, use [numerical accuracy checks](docs/numerics.md) and matched
teacher-forced scoring; benign rounding differences need not reproduce the
previous build's bits. Weight storage changes must preserve the source values.

Reduced-precision levers include quantizing activations to FP8 for the tensor
cores, summing expert partials in bf16, and folding scales into the weight
values. dgpp supports those levers, but
never silently. Each is an explicit `engine.*` key, off by default, that a
deployment turns on in its own configuration; every rank runs the same
setting (the config digest carries it) and the startup log names it. Each
lever's kernel is gated in its test against the exact chain, and its cost
is measured on real prompts — the task evals in [benchmarks](docs/benchmarks.md)
and greedy transcript comparisons — before a template may enable it. A
template enables a lever only when that measured quality cost is inside
the run-to-run band of the default chain, and it says so; the
benchmarks list the default and the levers-on numbers as separate rows.
The current keys are `engine.prefill_bf16_partials`,
`engine.prefill_fold_scales` and `engine.prefill_fp8_gemm` (the
[configuration](docs/configuration.md#execution-and-performance) table); which levers pay, and
how to measure any change on your own workload, is in
[optimizing performance](docs/optimizing_performance.md).

## Performance

See [Benchmarks](docs/benchmarks.md) for current serving throughput, cold
prefill latency, quality scores and long-context measurements. The document
identifies the measured source revision, hardware, workload and timing scope,
and the reproduction commands.

## Status

As of 2026-10-10, the source tree has twenty-two measured deployment templates
covering eight model architectures on one, two or four Sparks. The shared
engine provides graph decode, transactional MTP, row-batched execution,
grouped prefill, prefix caching, deterministic multi-rank scheduling and the
OpenAI-compatible service. Current quantized paths cover FP8, NVFP4, MXFP4,
full GLM-5.3's packed int4/int8 format and its two GPTQ re-encodings (NF4I8:
a sixteen-level codebook per 128 in a Hadamard-rotated basis; Mixed346: 3/4/6-bit
codebook experts with int8 activation codes on int8 tensor cores), and the
Qwen3.8 AutoRound int4/int8 hybrid (GPTQ layout, group 128, served as packed). Qwen NVFP4 runs on one or two Sparks by
mapping its n-gram table from NVMe and encoding the dense stack to FP8 at load.

The Qwen sixteen-slot decode graphs and prefill continuation are implemented as
opt-in controls; the shipped Qwen templates retain four slots and monolithic
admission. DeepSeek ships at six slots with DSpark depth 4, confidence
scheduling, bounded grouped prefill and the stream-ordered eager collective.
Full GLM-5.3 ships at eight slots and uses packed tensor-core prefill from 128
rows. MiMo-V2.6-Flash ships at four slots with MTP depth 1 on two or four
Sparks. DeepSeek-V4-Flash (2026-10-02) ships at six slots on four Sparks and
four on two, both with the confidence-scheduled DSpark depth (greedy and
sampled requests) and the 1M-token context. Qwen3.8-27B (2026-10-03/04) ships the
DFlash2 block drafter on one Spark (one block proposal a step, eight request
slots, the FP8 head) and MTP depth 3 on two and four (tensor parallel over the
DeltaNet and attention heads, the MLP and the head); the drafter runs on
every world since 2026-10-04 (the block proposal recorded inside the graph
step, the ranks' top-16 lists merged through one fold, the drafter sharded)
and is the two- and four-node templates' mode. Version 0.1.0
remains the original GLM-5.3-Flash sign-off release; the current source has
advanced beyond that baseline.

[PLAN.md](PLAN.md) summarizes implementation status,
[CHANGELOG.md](CHANGELOG.md) records changes, and
[the remaining work](docs/next_steps.md) lists current limitations.

## Quickstart

Use an internet-connected Spark as rank 0. First get the source:

```bash
git clone https://github.com/HawkBearPig/dgpp.git
cd dgpp
```

For an x86 Linux build workstation, use the Docker-based
[Spark cross-build](docs/cross-compiling.md); run the resulting binaries on a Spark.

Run the guided setup on rank 0:

```bash
./scripts/setup.sh
```

The wizard helps you choose a supported model and node count, saves your site
settings, checks SSH and software dependencies on every node, and walks through
RoCE lane selection for multi-node deployments. It then builds the release
server, prepares the downloader, downloads the checkpoint once and syncs peers,
and runs the serving preflight. Existing deployment tuning and unrelated `.env`
entries are preserved. Reruns reuse the build and complete cached checkpoints.

Python 3.10+ is needed to run setup. On Ubuntu/DGX OS, the wizard can install
standard system packages with `sudo`; `--install-system-deps` requests this
up front. NVIDIA drivers/CUDA and physical network setup remain site prerequisites.
Expect substantial checkpoint storage and download time on a fresh machine.
The wizard reports each node's free cache space and explains the lane choices;
it cannot verify cabling or end-to-end RDMA connectivity.

Setup prints the commands to start, inspect and stop the selected deployment.
Add `--start` to launch after all checks pass:

```bash
./scripts/setup.sh --start
```

The API defaults to `0.0.0.0:18080` (all IPv4 interfaces). Setup asks for the
HTTP bind address; `--http-bind 127.0.0.1` selects local-only access. The server
has no authentication or TLS. Wait for `READY`
before sending requests. For unattended setup, read-only checks, offline cache
sync and the manual walkthrough, see [Getting started](docs/getting-started.md).
Use `./scripts/setup.sh --help` for all options.

Manual builds can opt into [table masks for free JSON strings](docs/json-mask-performance.md).
The build option is off by default; normal setup retains the existing behavior.

### Send a request

Ask the running model a question. This short example requests low reasoning
effort because reasoning tokens also count toward `max_tokens`:

```bash
MODEL=$(curl --fail -s http://127.0.0.1:18080/v1/models | jq -r '.data[0].id')
jq -n --arg model "$MODEL" \
  '{model:$model,max_tokens:160,reasoning_effort:"low",messages:[{role:"user",content:"Name three primary colors."}]}' \
  | curl --fail http://127.0.0.1:18080/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @-
```

Stop it when finished, using the same config:

```bash
python3 scripts/dgpp-cluster down --config /path/to/the/selected/deployment.json
```

## Release and install

A release is a versioned tarball installed once per node. Starting an
installed release stages the configuration and uses the installed binary.

```bash
CONFIG=/path/to/the/selected/deployment.json        # use the path printed by setup
scripts/release.sh                                 # build the release preset, stage, verify, pack
scripts/dgpp-cluster install dist/dgpp-VERSION.tar.zst --config "$CONFIG"
scripts/dgpp-cluster up --release VERSION --config "$CONFIG"
scripts/dgpp-cluster releases --config "$CONFIG"
```

Select a release with the config's `release` key or `--release`.

The version is the tree's, stamped at build time by `cmake/version.cmake`
into the binary (`dgpp-serve --version`; `0.1.0+g<sha12>`, `.dirty` when
the tree had uncommitted changes) and sent on the journal's settings
record, so a world of mixed versions refuses to form. Inside the tarball:

| path | what |
|---|---|
| `bin/dgpp-serve` | the server, rpath `$ORIGIN/../lib` |
| `lib/libcudart.so.13`, `lib/libcublasLt.so.13` | the CUDA runtime it was built against |
| `scripts/` | launcher, process/preflight/config helpers, checkpoint downloader and API check |
| `deploy/*.example.json` | all supported deployment templates |
| `README.md`, `docs/` | package-specific setup, dependency and networking guides |
| `MANIFEST`, `MANIFEST.sha256` | version, git sha, CUDA, build host and date; every other file's checksum |

Beyond the tarball a node needs the NVIDIA driver, rdma-core, libnl and
libstdc++, all part of the DGX OS image, plus the checkpoint and resident
image caches on its local NVMe. `install` unpacks under `paths.release_dir`
(`~/dgpp/releases/dgpp-<version>/`) and checks every file against the
manifest. Versions can be installed side by side; stop the running service
and select an older version to roll back. The launcher starts the service
on demand; no boot-time service units are included. When no release is
selected, `up` stages `build-release/dgpp-serve` to the peers. Use
`--bin build-ci/dgpp-serve` explicitly when deploying a testing build.

## Configuration

Use the [configuration reference](docs/configuration.md) for site settings,
deployment options, defaults and their effects on memory, response time and
answer quality. It includes a [guide to choosing settings](docs/configuration.md#find-the-setting-you-need)
and explains [how overrides are resolved](docs/configuration.md#how-settings-are-resolved).
Start with the [deployment template](deploy/README.md) for your model and node count.

## Source layout

Model implementations share the engine, service, scheduler and text stack:

| directory | namespace | what |
|---|---|---|
| `src/serve/` | `dgpp::serve` | the HTTP server, the OpenAI-compatible service, the admission journal, the throughput line, the text frontend interface |
| `src/sched/` | `dgpp::sched` | the scheduler (admission, the tick, cancel and stop), the prefix cache's index, the `SchedulerEngine` interface every model implements |
| `src/sample/` | `dgpp::sample` | the sampler's host math: penalties, the logit bias, masks, top-k/p, the exact prefix decision |
| `src/text/` | `dgpp::text` | the tokenizer (HF tokenizer.json), the chat template (Jinja), the tool grammar and parser, the JSON-schema grammar |
| `src/kernels/` | `dgpp` | CUDA kernels; `pick.cu` and `sample_pick.cu` are the device pick and sampler any model's logits can use, the rest carry their model's name |
| `src/net/` | `dgpp::net` | the RoCE collective bus, TCP, the roster |
| `src/engine/` | `dgpp` | shared session interfaces, eager and graph engines, speculative decoding, memory plans and prefix arenas |
| `src/models/qwen/`, `src/models/glm4/` | `dgpp` | Qwen3.8-Flash-Next and GLM-4.7 configuration, loaders, layers and sessions |
| `src/models/glm/` | `dgpp` | GLM-5.3-Flash: the forward pass, loader, MoE and mHC layers, session adapters and MTP draft |
| `src/models/glm_dsa/` | `dgpp` | full GLM-5.3 configuration, packed loader, MLA/DSA model and session implementation |
| `src/models/dsv41/` | `dgpp` | DeepSeek-V4.1-Flash configuration, MXFP4/FP8 loader, CSA2, Engram and DSpark session implementation |
| `src/models/dsv4/` | `dgpp` | DeepSeek-V4-Flash configuration, MXFP4/FP8 loader, sliding-window and compressed-cache attention with the 64-head indexer, token-table routing, two-pass hyper-connections and the DSpark session |
| `src/models/` | `dgpp` | the KDA and DSA layer libraries and the quantized matrix, shared by any model that uses them |
| `src/loaders/`, `src/core/`, `src/common/` | `dgpp` | safetensors, the HF cache, JSON; arenas and tracing; logging and process memory |

The server binary is `dgpp-serve` (`apps/dgpp_serve.cpp`); the GLM
tools keep their names (`glm_gen_check`, `glm_forward_check`, …).

`tools/` holds the Python checkpoint and reference tooling the tests use;
`scripts/` the fabric and serving operations; `apps/` the binaries;
`deploy/` the cluster
config.

## Documentation

| page | what |
|---|---|
| [Configuration](docs/configuration.md) | site and deployment settings, defaults, limits and the effects of each option |
| `docs/operations.md` | booting, stopping and watching the serving world; memory and the image cache on a node; failure semantics |
| `docs/benchmarks.md` | every measured serving number: per model, world, concurrency and prompt class, with the method to reproduce each |
| `docs/signoff_v1.md` | the v1 performance and hardening sign-off, with every measurement |
| `docs/next_steps.md` | what is worth doing next, ranked by cost and benefit |
| `docs/adding_a_model.md` | the checklist for a new model family: loader, model, engine keys, template, fixture gate, validation and the pitfalls ports have hit |
| `docs/tools.md` | serving, testing and diagnostic commands |
| `docs/testing.md` | the test suites and what each proves |
| `docs/numerics.md` | judging a numerics change; the sampling-width gate |
| `docs/optimizing_performance.md` | the levers that trade memory, latency, throughput and accuracy, which ones pay on which model, and how to measure a change before trusting it |
| `docs/mtp.md` | speculative decode with the MTP layer |
| `docs/measurements.md` | current validated platform, network, kernel and serving measurements, with historical observations separated |
| `docs/checkpoint_budget.md`, `docs/qwen38_checkpoint_budget.md`, `docs/checkpoint_budget_glm53.md`, `docs/checkpoint_budget_dsv41.md` | generated checkpoint inventories, per-rank residency and decode traffic budgets |
| `docs/batched_mtp_graph_stall.md` | a worked stall investigation, from symptom to root cause |
| `docs/qwen38_flash_next_plan.md` | Qwen3.8-Flash-Next's architecture, placement, decisions and status |
| `docs/glm47_plan.md` | GLM-4.7 NVFP4 architecture, modelopt format contract, decisions, gates and status |
| `docs/glm53_plan.md` | full GLM-5.3's packed checkpoint, DSA/MLA changes, placement and serving status |
| `docs/deepseek_v41_flash_plan.md` | DeepSeek-V4.1-Flash's CED, CSA2, Engram, DSpark and serving implementation |
| `docs/performance_improvement_plan.md` | current performance findings, completed changes and next measured targets |
| `DESIGN.md` | architecture and implementation contracts |
| `PLAN.md` | the milestones, their exit gates and status |
| [Benchmark procedures](docs/benchmarking.md), [serving measurements](docs/benchmarks.md) | benchmark methods, summarized results and reproduction commands |
| `CHANGELOG.md` | the history by milestone |

## Contributing

`CONTRIBUTING.md` describes the build presets, the test discipline, the
evidence a fabric change needs, the code style and where things go.

## License

Apache License 2.0. See `LICENSE`.
