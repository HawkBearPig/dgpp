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
[real-checkpoint numerical validation](benchmarks/results/2026-09-21-qwen-fp8-head-numerics.md),
including the YaRN template. `--fp8-head gemv` restores the previous head
path. See [operation and numerical constraints](docs/operations.md).

The Qwen NVFP4 templates use `engine.ngram_table: "mmap"` to read the
n-gram table from NVMe, `engine.decode_graph: true` for resident graph serving,
and `engine.dense_weights: "fp8"` to encode dense projections at load. On two
Sparks, mapping saves 23.84 GiB per rank while staying within 3.6% of resident
decode throughput and 2.9% of resident prefill time in the matched campaign.
Use `--dense-weights checkpoint --fp8-head gemv` to retain the checkpoint's
BF16 dense stack.
See the [single-node guide](docs/qwen38_single_spark.md) and
[two-node benchmark](benchmarks/results/2026-09-16-qwen-nvfp4-w2.md).

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
Prefill is bitwise the same for any chunking of a prompt; prompts that
arrive together are read in through one forward and start decoding
together, and a later arrival interleaves with the running decodes.
Measured on four Sparks (2026-10-02): 65–126 tokens/s for one request by
prompt class (44 without the draft), 134–186 tokens/s across six, cold
prefill of a 2K prompt in 1.36 s, and on llama-benchy's sampled pp2048 /
tg128 64 tokens/s for one request and 91 / 121 in total at two and five
concurrent requests; on two Sparks 40–73 tokens/s for one request and
68–101 across four ([benchmarks](docs/benchmarks.md)). See the
[model card](docs/model_cards/DeepSeek-V4-Flash.md).

The linked templates enable MTP at the depth measured best for that deployment.
Plain decode, deeper MTP, alternate cache formats and slot counts are launcher
knobs on these templates; [the deployment catalogue](deploy/README.md) maps the
supported shapes to their flags. [The benchmark tables](docs/benchmarks.md)
record the modes measured for each deployment.

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
  recipes use the improvement automatically. See the
  [comparison and validation](benchmarks/results/2026-09-21-deepseek-selection/README.md).
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
  context up. See the [record](benchmarks/results/2026-09-28-qwen-autoround-int4/README.md).
- **Model-specific prefill paths**: packed int4/int8 tensor-core prefill for
  full GLM-5.3, tiled QSA prefill for Qwen (W4A4 expert prefill is opt-in
  pending the gate in issue #68),
  and bounded grouped prefill for DeepSeek-V4.1-Flash. Qwen can optionally
  yield between prefill chunks so active decodes continue making progress.
- **Opt-in 512K context for Qwen3.8-Flash-Next** with `engine.rope_scaling`
  (YaRN): the [two-Spark NVFP4 template](deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json)
  supports a 524288-token request ceiling, with 5/5 retrieval probes passing
  at both 261K and 522K prompt tokens. See the [validation record](benchmarks/results/2026-09-20-qwen-yarn512k.md)
  and [release-check procedure](docs/qwen_yarn_release_check.md).
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
  rank for GLM-5.3-Flash on four nodes; see the
  [numerical validation](benchmarks/results/2026-10-01-compact-serving-heads.md).

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
[configuration](#execution-and-performance) table); which levers pay, and
how to measure any change on your own workload, is in
[optimizing performance](docs/optimizing_performance.md).

## Performance

See [Benchmarks](docs/benchmarks.md) for current serving throughput, cold
prefill latency, quality scores and long-context measurements. The document
identifies the measured source revision, hardware, workload and timing scope,
and links to the raw results and reproduction commands.

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

Site settings live in `.env` at the repository root, shared by every model
deployment. Scripts load it automatically through `scripts/site_env.py`
(shell scripts use `scripts/cluster_env.sh`). `DGPP_ENV_FILE` selects another
file; exported variables override file values. Values are literal, optionally
quoted: no shell commands or variable expansion are evaluated. The site helper
loads an allowlist of settings, so credentials such as `HF_ACCESS_TOKEN` stay out of the
scripts' environment and generated server config.

| `.env` key | purpose | default |
|---|---|---|
| `DGPP_NODES` | Space-separated SSH/control hostnames or IPv4 addresses in rank order, using management IPs, fabric IPs, or a mixture. Rank 0 runs locally, must reach peers over SSH, and must be reachable by peers at its listed address. A deployment uses the first `world_size` entries. See [network layouts](docs/networking.md#ssh-and-control-addresses). | required |
| `DGPP_SSH_USER` | Peer login for binary staging, process control, diagnostics and checkpoint sync. Needs SSH-key access and write access to the configured directories. | current login when empty or absent |
| `DGPP_CLUSTER_CONFIG` | Default deployment filename, saved by guided setup. An explicit `--config` takes precedence. | legacy four-node Flash hybrid filename |
| `DGPP_HTTP_PORT` | Default client-facing API TCP port on rank 0; deployment `http.port` overrides it. | 18080 |
| `DGPP_HTTP_BIND` | Default IPv4 listening address on rank 0; deployment `http.bind_host` overrides it. Setup prompts for it; `--http-bind` sets it unattended. | `0.0.0.0` |
| `DGPP_FABRIC_PORT` | Rank-0 TCP rendezvous listener used to establish the inter-node transport. Peers must reach it; clients do not use it. Keep it private to the cluster. | 29970 |
| `DGPP_JOURNAL_PORT` | Rank-0 TCP listener that distributes ordered scheduler operations to peers. Must differ from the fabric/API ports; keep it private to the cluster. | 29971 |
| `DGPP_LOG_DIR` | Base directory on rank 0 for logs, process records and collected peer logs. The launcher adds a deployment-specific subdirectory. | `~/dgpp/log` |
| `DGPP_STAGE_DIR` | Base directory on peers for staged development binaries, runtime config and logs. The launcher adds a deployment-specific subdirectory; it is not the model cache. | `/tmp/bus4` |
| `DGPP_RELEASE_DIR` | Base directory on each node for versioned installed server releases. Use a persistent, writable location. | `~/dgpp/releases` |
| `DGPP_BUILD_DIR` | Explicit build-directory override. Relative paths use the repository root. Leave unset to keep release and testing builds separate. | `build-release` for serving/packaging; `build-ci` for testing (`build-<preset>` with `DGPP_PRESET`) |
| `DGPP_DATA_DIR` | Evaluation-data location. Relative paths use the repository root. | `data` |
| `HF_HUB_CACHE`, `HF_HOME` | Downloaded checkpoints. An explicit hub cache wins; otherwise use `HF_HOME/hub`. Leave unset for the standard cache or choose a disk with room for the full checkpoint on each node. | `~/.cache/huggingface/hub` |
| `DGPP_RESIDENT_CACHE_DIR` | Per-node disk cache of prepacked weight images for faster reloads, separate from the HF checkpoint. Takes precedence over deployment `paths.resident_cache`. | `~/.cache/dgpp/resident` |
| `DGPP_ROCE_DEVICES` | One or two ordered local verbs device names, not Linux interface names. Use `discover_roce.py`; match lane subnets in the same order across nodes. | discover active Ethernet devices |
| `DGPP_ROCE_GID_INDICES` | One RoCE-v2 address-table index per explicitly selected device, in the same order. Pin these when a device offers several networks; discover the values on each host. | automatic RoCE-v2 selection |
| `DGPP_NODE_OVERRIDES` | JSON map keyed by exact `DGPP_NODES` entries, with per-node device/GID/hub-cache/resident-cache values. Use when peers have different NIC names or disks; see `.env.example`. | none |

Model settings live in deployment JSONs under `deploy/`. Each has a
`world_size` of 1, 2 or 4 and uses that many nodes from the beginning of
`DGPP_NODES`. Too few nodes is an error, not a fallback to a smaller world.
Select the deployment explicitly with `--config FILE`.
Wrappers that accept a config argument pass it through to their children.
Stage and release directories must be absolute or start with `~/`, without
spaces or shell syntax.
Log and staging directories are namespaced by deployment-file path;
`dgpp-cluster paths` prints the effective locations. An explicit `--log-dir`
is used as-is. `up` refuses an existing deployment unless `--replace` is given;
`down` without `--config` stops every recorded deployment that is running;
cleanup uses recorded process identity, never a binary-name kill.
During startup, the launcher prints the rank 0 log path and reports elapsed
time and the latest log line every ten seconds while waiting for readiness.
If rank 0 exits, it reports the exit status and full trailing log lines
within the two-second polling interval, then cleans up the deployment's peers.

The launcher resolves the deployment and site settings into
`<log_dir>/cluster.resolved.json` when starting the service. Both the head
and peers read that resolved config; the original JSON and `.env` are not
sent to peers. To inspect or use the runtime config with the native binary:

```bash
scripts/dgpp-cluster resolve --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json
# Save that JSON to a file before passing it to dgpp-serve --config.
```

The native binary reads resolved JSON, not `.env` or deployment templates.
Unknown keys and invalid types are rejected.

Rank 0 reads the model, ports and engine options, with command-line flags
after `--config` overriding the file. It sends the effective settings to
peers before model construction. Peers use their own files for bootstrap
addresses and local paths, then verify the shared settings by digest.
The defaults below come from `ClusterConfig::Engine`; deployment
templates set their serving options explicitly.

### Deployment and endpoint

| Key | Required | Purpose and when to change it | Default |
|---|---|---|---|
| `model` | yes | Exact Hugging Face repository ID, such as `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8`. Selects the checkpoint tensors, tokenizer and chat template. It is not a local filesystem path or a generic quant name. | — |
| `world_size` | yes | Number of participating nodes/ranks: 1, 2 or 4. Uses the first N entries of `DGPP_NODES`; each rank stores its tensor-parallel share in memory. Choose a supported model/world pair, not an arbitrary smaller number to save machines. | — |
| `http.bind_host` | no | IPv4 address on rank 0 that accepts API connections. `127.0.0.1` is local-only; a LAN address exposes that interface; `0.0.0.0` exposes all IPv4 interfaces. The server has no authentication/TLS. Does not select the RoCE interface. | `DGPP_HTTP_BIND`, otherwise `0.0.0.0` |
| `http.port` | no | TCP port clients use for the API, in 1–65535. Change it if the default is occupied, then update client URLs. Overrides the site HTTP port and must differ from fabric/journal ports in a multi-node deployment. | `DGPP_HTTP_PORT`, otherwise 18080 |
| `http.max_body_bytes` | no | Maximum serialized HTTP request body in bytes, as a positive integer. Allows large document prefills and agent histories; independent of the model's token/KV capacity. Oversized requests receive HTTP 413 based on Content-Length, before tokenization. Buffers grow with received data, so this does not preallocate the limit per connection. The binary flag `--http-max-body-bytes` overrides it. | 268435456 (256 MiB) |
| `http.sse_ping_interval` | no | Whole seconds of stream silence before sending an SSE keep-alive comment. Accepts 1–2147483647; `-1` disables pings. Covers queued requests, prefill and gaps between output chunks on rank 0. `--sse-ping-interval` overrides the file; request `sse_ping_interval` overrides the server setting. Engine deadlines are unchanged. | 30 |
| `release` | no | Installed software version to run on every node, not a model revision. Select a version under `DGPP_RELEASE_DIR/dgpp-VERSION`; use it to upgrade or roll back server binaries. `--release` overrides this field; `--bin` overrides binary selection. | Empty: use the development build |
| `paths.resident_cache` | no | Disk directory for prepacked per-rank weight images, which speed subsequent loads. This is separate from the Hugging Face download cache and consumes additional disk space. Prefer the shared `DGPP_RESIDENT_CACHE_DIR` site setting unless a deployment needs its own directory. | Empty: `~/.cache/dgpp/resident`; `DGPP_RESIDENT_CACHE_DIR` takes precedence |

### Request capacity and memory

For a first run, retain the template values. Tune `kv_capacity` for context
space and `max_concurrency` for simultaneous work; these are different limits.
Startup checks the combined memory plan before loading.

| Key | Required | Purpose and when to change it | Default |
|---|---|---|---|
| `engine.max_concurrency` | no | Maximum actively executing requests, not TCP connections or queued requests. More slots can improve aggregate throughput but use more state/scratch memory and may increase per-request latency. Allowed range is 1–16, subject to model/MTP row limits below. | 8 |
| `engine.kv_capacity` | no | Shared context-token pool on each rank, across active requests—not a separate allowance for every request. A prompt and its generated answer must fit. Increase for longer contexts or more simultaneous context; memory use increases and allocation is rounded to model block boundaries. | 8192 tokens |
| `engine.kv_dtype` | no | GLM-5.3 latent-cache precision: `bf16`, `fp8`, or `fp4`; the MiMo-V2.6-Flash K/V cache takes `bf16` or `fp8` (e4m3 rows with one scale per head row, half the bytes). Lower precision reduces cache storage at a numerical-accuracy cost; it does not quantize model weights. The GLM index cache stays FP8. Qwen, GLM-4.7 and DeepSeek K/V caches remain BF16. | `bf16` |
| `engine.mtp_head` | no | Full GLM-5.3 draft-block head: `checkpoint` reads the lm head as the main step does (its 12-bit form under `bf16_weights`); `fp8` reads a block-FP8 copy encoded at load (e4m3, 128×128 fp32 scales, the FP8 releases' recipe; +0.22 GiB per rank at world 4). The verifier keeps the main head, so the served distribution is unchanged; only the draft's acceptance can move (measured: unchanged, the draft head 1.36 → 1.05 ms per pass, 0.5 % of the step). Other families ignore it. | `checkpoint` |
| `engine.attention_weights` | no | Full GLM-5.3 attention projections' resident form: `checkpoint` as shipped (int8 g64); `int4` re-encodes q_a/kv_a, q_b and o_proj to int4 g64 at load with the RTN recipe (kv_b stays int8): −2.1 GB per MTP pass per rank and −0.9 GiB resident at world 4, NOT lossless — measure the quality cost with the teacher-forced gate before shipping it. Other families ignore it. | `checkpoint` |
| `engine.embed_sharding` | no | Full GLM-5.3 and DeepSeek embedding/head placement: `replicated` keeps the full table on every rank; `vocab` keeps each rank's vocabulary slice and folds token lookups. The full-GLM template uses `vocab` to save 1.33 GiB/rank; the DeepSeek template retains `replicated`. Other families ignore it. | `replicated` |
| `engine.default_max_tokens` | no | Answer-token budget for requests that omit `max_tokens`. Clients may supply their own value; this is not a global hard limit. A larger default also reserves more context space under full admission (`kv_capacity` must cover the prompt plus this budget, or the request is refused). Every deployment template sets 32768: agent clients such as Hermes send no `max_tokens`, and a thinking model's reasoning alone exceeds a few hundred tokens. | 256 |
| `engine.queue_limit` | no | Maximum requests waiting for an execution slot or memory budget. Additional arrivals receive HTTP 503 `overloaded`. Increase to tolerate bursts, at the cost of longer waits—not higher execution capacity. | 64 |
| `engine.max_connections` | no | Maximum simultaneously open HTTP connections on rank 0, including idle keep-alive connections and streams. Excess connections receive 503. Size this separately from active request slots. | 64 |
| `engine.prefix_cache_gib` | no | Memory budget per rank for reusable prefix-state snapshots. Long documents can reuse an earlier snapshot when their question changes. Snapshot slots and the KV token pool are separate limits; see [sizing and recipe capacities](docs/prefix-cache.md). Set 0 to disable. | 1.5 GiB |
| `engine.admission` | no | When to reserve context space. `full` reserves prompt plus the requested answer budget before admitting a request. `grow` starts with a smaller reservation and extends it during generation; if space runs out, the youngest request is shed. Use `full` for predictable reservations, `grow` to trade that guarantee for denser occupancy. | `full` |
| `engine.admission_window` | no | Answer-token reservation increment used by `grow` admission. Larger increments reduce growth frequency but reserve more space ahead of use. Has no effect under `full`. Must be positive. | 256 tokens |
| `engine.prefill_budget_tokens` | no | Maximum prefill tokens per scheduler tick. -1 selects an aligned budget near 256 on supported Qwen, GLM-5.3-Flash and DeepSeek-V4-Flash graph engines, with cancellation and decode between chunks. Positive values override it; 0 explicitly keeps full-prompt admission. Other engines retain full-prompt admission. DeepSeek-V4-Flash reads every in-flight prompt's next chunk in one forward: there the budget is a quantum per reading prompt (up to the forward's 4096 rows per tick), prompts queued together begin together, and near the end the shares level so they finish — and start decoding — together. Images on an engine without image chunking run monolithically as the tick's only prefill work when they exceed the budget. | -1 (automatic) |
| `engine.prefill_idle_budget_tokens` | no | Larger prefill budget when no request is actively decoding. Requires an enabled busy budget, must be at least that budget, aligned and within the same prefill limit. Rechecked after each chunk. 0 uses the busy budget for all chunks — except on DeepSeek-V4-Flash under the automatic busy budget, where it defaults to the whole forward (4096). | 0 (disabled) |
| `engine.admission_gather_ms` | no | How long rank 0 holds the first arrival at an idle engine for the rest of its burst. Requests sent together land a few milliseconds apart; the wait lets them be read in together instead of the first starting one tick ahead. Applies only when nothing is queued or running. 0 ticks at once. | 3 ms |

### Execution and performance

These settings change how the model runs. Use the matching deployment template
first; change one setting at a time and measure the effect on your workload
([optimizing performance](docs/optimizing_performance.md) has the protocol).
Every key here keeps the model's outputs exactly except the three
`engine.prefill_*` levers, which trade prefill arithmetic for speed and are
off unless a deployment turns them on (see
[accuracy and correctness](#accuracy-and-correctness)). Engine behavior is
set here, in the deployment file; an environment variable never selects a
kernel, a lever or a threshold.

| Key | Required | Purpose and when to change it | Default |
|---|---|---|---|
| `engine.decode_graph` | no | Use CUDA graph replay for decode to reduce CPU launch overhead. On one node, this selects resident graph serving instead of the eager streaming path. Required for MTP and the single-Spark Qwen templates; leave enabled for the documented serving configurations. | false |
| `engine.mtp` | no | Enable multi-token prediction: draft candidate tokens, then verify them with the main model. Can reduce decode time when drafts are accepted, but adds draft state and verification work. Requires `decode_graph`; not a larger request batch. | false |
| `engine.mtp_depth` | no | How many tokens to draft per speculative step, 1–5. Greater depth can accept more tokens per pass, but uses more verification rows and can waste work when drafts are rejected. Values above 1 require MTP; supported batching varies by model. DeepSeek's shipped template uses depth 4. | 1 |
| `engine.mtp_schedule` | no | Enable confidence-scheduled verify depth for a model with a confidence head, or with the DFlash2 block drafter (its selector walk's confidence for each draft; the drafter recipes ship it since 2026-10-05). Requests verify only the leading drafts whose survival probability justifies another row: greedy requests always, sampled requests when their drafts are the draft head's most likely tokens (`engine.mtp_draft` `greedy`; with drawn drafts they keep the configured depth). The depth changes pace only — a greedy transcript and a sampled request's distribution are the same at every depth. Unsupported for Qwen C16/MTP3: leave this false at sixteen slots and depth 3 because additional verification depths exceed the graph-variant limit. With the DFlash2 drafter the selector's own confidence (the chosen candidate's softmax mass per draft) drives the same schedule, so a confident block verifies deep and a prose block short at high concurrency; exclusive with `engine.dflash_batch_rows`. Exact at every depth since 2026-10-05: the recorded commit takes a reduced-depth step's own rows (it read the model's decode rows and retracted a step that accepted every row from a snapshot its walk never wrote), and the Qwen decode step's kernels are one chain at every row count. | false |
| `engine.mtp_schedule_row_ms`, `engine.mtp_schedule_base_ms` | no | Cost model for scheduled verification: milliseconds for another verify row and fixed work per pass. The Qwen3.8-27B drafter recipes ship 2.0 ms a row and 130 / 66 / 34 ms a pass on one / two / four Sparks (the 2026-10-05 traces). These values are deployment measurements; retain the DeepSeek template values unless re-profiling that world. | 8.0 / 28.0 |
| `engine.mtp_schedule_lambda`, `engine.mtp_schedule_min_depth`, `engine.mtp_schedule_adapt` | no | Floor for the value of decode time in tokens/ms, minimum verified draft depth, and whether the value adapts from committed tokens and modeled time. Lambda 0 derives the reservation rate. | 0 / 1 / true |
| `engine.mtp_schedule_sampled_scale` | no | Under `engine.mtp_schedule`, a sampled request's expected acceptance per drafted position as a fraction of the confidence head's. The head predicts a greedy match; a sampled request accepts a draft with the target's probability of it. Traced on DeepSeek-V4-Flash with the whole block verified every pass (3,140 prose passes, 819 over the five prompt classes), the head over-states a sampled request's first draft (0.75 predicted against 0.67 accepted on prose) and is near calibrated behind it; replayed over those passes, 0.93 commits 1.5–4 % more tokens per second than 0.8 and within 0.2 % of the best first-position-only correction. 0 makes sampled requests verify the whole block. Applies only with `engine.mtp_draft` `greedy`. | 0.93 |
| `engine.mtp_draft` | no | How a sampled request's speculative drafts are chosen: `sampled` draws them from the draft head's own distribution and verifies with the acceptance ratio; `greedy` takes the draft head's most likely token and accepts it with the target's probability of that token. Both preserve the target distribution exactly; they differ in tokens accepted per pass. `auto` uses the rule measured better for the model family (greedy for DeepSeek-V4-Flash's DSpark block, sampled elsewhere). With the DFlash2 block drafter the same key rules the recorded block: `sampled` / `auto` draw each draft from the selector's softmax at the request's temperature and verify by the ratio rule (the reference speculator's form), `greedy` walks the argmax under the P(draft) accept. The ratio rule applies in both sampling regimes — the truncated one (top-k / top-p / min-p) and pure temperature sampling (since 2026-10-05; a request that sends only `temperature`, as most clients and the arena harness do, is the pure regime). Greedy requests are unaffected. | `auto` |
| `engine.mtp_draft_temperature` | no | The drawn drafts' temperature as a fraction of the request's (the MTP draft pick's and the DFlash2 walk's proposal is the draft's distribution at this temperature). Exact at any value; it moves only the acceptance rate: a sharper draft keeps the argmax's rate where the target is sharp, the request's temperature keeps the overlap where it is flat. The Qwen3.8-27B drafter templates set the measured 0.7. | `1.0` |
| `engine.mtp_verify` | no | How a sampled request's speculative chain is judged: `block` runs block verification (Sun et al. 2024: the drafts decided jointly — p_i = min(1, p_{i-1} P/Q) along the block, each sub-block accepted with h_i = S_i / (S_i + 1 - p_i), the longest accepted sub-block kept, the residual (p_tau P - Q)+ drawn after it), exact for the output distribution and never fewer tokens in expectation than `token`, the token-by-token accept / residual test. Where a row's draft cannot be priced on the device (pure temperature sampling, the draft outside the held prefix) the step takes the token rule, and a proposal with an id outside the prefix counts as a point mass for its row; both choices the draws never see. Greedy requests are unaffected. The Qwen3.8-27B drafter templates set `block` (measured +5 % on four Sparks under sampling, neutral elsewhere). | `token` |
| `engine.dflash_batch_rows` | no | With the DFlash2 drafter: the verify rows a row-batched step may hold. A batch family whose slots times the eight-row block exceed it verifies the first floor(rows / slots) − 1 drafts of every slot's block (the drafter still proposes the whole block; verifying fewer drafts than proposed is exact), so many live slots take a short step instead of a deep one — eight slots at eight rows cost 142 ms a step on four Sparks against 84 at 32 rows. `0`: every family verifies the whole block. | `0` |
| `engine.lookup_draft` | no | Context-lookup drafting for MTP families: a match of the request's own token history (a suffix of at least `lookup_nmin` tokens seen before) replaces the MTP drafts within the same verify rows on greedy slots — a match of at least `lookup_nstrong` on its own, a shorter one only when its first `lookup_agree` tokens equal the device's own picks. Same rows verified — throughput only, never the greedy output. Sampled slots retain their MTP proposals; lookup fusion is disabled for them because fusion changes the proposal distribution. Firing steps reseed the slot's feed (stream-ordered, no drain); silent steps cost a host scan. Off by default; every rank takes the same setting through the journal. | false |
| `engine.lookup_nmin` / `engine.lookup_nstrong` / `engine.lookup_agree` | no | The lookup thresholds: propose from a suffix match of at least `nmin` tokens; take matches of at least `nstrong` on their own; take shorter matches only when their first `agree` tokens equal the MTP drafts (`agree` 0 disables the weak path). | `6` / `8` / `2` |
| `engine.lookup_tail` | no | Extra verify rows past the MTP block, a pinned mode for copy-heavy traffic: the device chain spans them as deeper MTP drafts, and a strong lookup match fuses over the whole width instead (up to 1 + `mtp_depth` + `lookup_tail` rows, within the 8-row verify cap and the 6-slot device picker — so at most 5 drafts). Fewer batch slots cover the wider block, and prose pays the wider attention every step whether it copies or not — the same bargain as HyperQwen's reproduction mode. Incompatible with the block drafter and the scheduled verify depth. | `0` |
| `engine.prefill` | no | DeepSeek prefill mode: `bounded` runs every prompt row through the encoder and only the final window through the decoder; `exact` runs every layer over every row for parity work. Other families ignore it. | `bounded` |
| `engine.ngram_table` | no | Qwen n-gram embedding-table placement. `resident` keeps it in device-accessible memory; `mmap` leaves it on local NVMe and fetches needed rows through the host page cache. The Qwen NVFP4 templates use `mmap`; the two-Spark campaign measured at most 3.6% lower decode throughput for 23.84 GiB less planned model memory per rank. DeepSeek's Engram tables use their own mapped checkpoint sidecar. | `resident` |
| `engine.dense_weights` | no | Qwen dense-projection storage. `checkpoint` retains the checkpoint's BF16 form; `fp8` converts dense projections at load time to reduce their memory footprint, with quantization error. Does not select another HF repository or change the expert quant; other families ignore it. On Qwen3.8-27B, whose projections ship as FP8, `fp8` requantizes the BF16 lm head to block FP8 (half its bytes per decode row: MTP depth-2 C1 151 → 132 ms/step at the same acceptance, HumanEval/GSM8K 39/40 each; greedy transcripts move within the first tokens). The Qwen3.8-27B templates ship it; `checkpoint` restores the BF16 head. | `checkpoint` |
| `engine.fp8_head` | no | Qwen FP8 vocabulary-head dispatch. `mma` takes the streaming tensor-core form at every width — one chain, so a request's logits are the same alone, at any verify depth, in a batch, and for a prompt's last row whatever the prompt's length (since 2026-10-05; before, the GEMV chunks below five rows and above the decode capacity were two more chains). Requires `dense_weights: "fp8"`; the NVFP4 templates select it after [matched numerical validation](benchmarks/results/2026-09-21-qwen-fp8-head-numerics.md). `gemv` keeps the row-chunked GEMV core. | `gemv` (NVFP4 templates: `mma`) |
| `engine.bf16_weights` | no | Resident form of the BF16 weights that decode streams (attention/linear-attention projections, the LM head; on Qwen3.8-27B the DFlash2 drafter's layers and fc taps, the MTP fc and a BF16 head, whose BF16 bytes stay resident under either packed mode). `checkpoint` keeps the BF16 bytes alone. `bf12` and `bf12+bf16` add a **lossless** 12-bit form of each matrix (sign+mantissa byte plus a 4-bit exponent code; exact side tables for the rare outliers): decode launches of up to eight rows read 0.75 of the bytes and produce bit-identical results, so transcripts do not change. This is a storage format, not quantization. `bf12+bf16` keeps both forms resident: prefill is untouched and the 12-bit copies cost about 0.75× those matrices in additional memory (GLM-5.3-Flash: +2.0 GiB/rank at four Sparks, +3.8 at two; GLM-4.7: +4.9). `bf12` keeps the 12-bit form **alone**: each matrix's BF16 bytes are returned to the node as its layer loads, the footprint drops below `checkpoint`'s (GLM-5.3-Flash −0.5 GiB/rank at four Sparks, −1.0 at two; GLM-4.7 −1.4), and a prefill GEMM expands the rows it reads into a small scratch first — the same bits, so the same results, for about 10 ms per prefill chunk on four-node GLM-5.3-Flash (+1–2% on 8K–32K prompts, +25–40 ms to a short prompt's first token). The memory plan includes either. Measured single-stream decode: GLM-5.3-Flash +6–7%, GLM-4.7 +11–12%, full GLM-5.3 +5–6.5%, Qwen3.8-Flash-Next-FP8 +6.4% on four Sparks and +9.0% on two ([benchmarks](docs/benchmarks.md)). On Qwen it packs the GDN/QSA projections and the head of the FP8 checkpoint and always keeps both forms resident (every Qwen recipe has the room); under `dense_weights: "fp8"` those matrices are already FP8 and nothing is packed. DeepSeek accepts the key and currently packs nothing. Templates with memory to spare ship `bf12+bf16`; the two sized to their nodes' memory (two-node GLM-5.3-Flash, full GLM-5.3) ship `bf12`. | `checkpoint` |
| `engine.draft_vocab` | no | Qwen3.8 AutoRound hybrid only: path to a one-dimensional int32/int64 `.npy` of token ids. The MTP draft head then scores only those rows of the head (every other id is `-inf` for the draft), so each draft step reads that slice instead of the whole head. The target verifies every draft, so outputs are unchanged; only the draft acceptance can move (a token outside the set is never proposed). `tools/build_draft_vocab.py TOKENIZER.json OUT.npy --size 65536` builds a set from the tokenizer's alphabet, added tokens and most frequent merges. Off by default; headline numbers are measured without it. | empty (whole vocabulary) |
| `engine.prefill_bf16_partials` | no | Qwen3.8 AutoRound hybrid prefill lever: the packed expert chain's down projection is written in bf16 and its per-expert partials summed from bf16 (half the bytes a prefill chunk writes and reads back; the reference stack's form). Not bitwise the default fp32 chain: served transcripts can differ within the quantized model's tolerance. Off by default; the AutoRound template turns it on (−3 to −7 % cold prefill, evals inside the default chain's band; [benchmarks](docs/benchmarks.md) list both rows). | false |
| `engine.prefill_fold_scales` | no | Qwen3.8 AutoRound hybrid prefill lever: the wide packed expert GEMM folds each group's scale into the bf16 weight values and keeps one fp32 accumulator across K (Marlin's form; no per-group fma). Not bitwise the default chain. Off by default and measured as no gain (+3 % at 32K): no template turns it on. | false |
| `engine.prefill_fp8_gemm` | no | Qwen prefill lever under `engine.dense_weights: "fp8"`: prefill-shaped dense projections run on the fp8 tensor cores from per-token 1×128 e4m3 activations with the checkpoint's 128×128 weight scales (the reference stack's blockwise GEMM) instead of dequantizing each matrix to bf16 for cuBLASLt. Not bitwise the dequantized chain (the activations are quantized). Off by default; requires `dense_weights` fp8; the AutoRound template turns it on beside `prefill_bf16_partials` (0 to −3 % alone). | false |
| `engine.prefill_fp8_per_tensor` | no | Qwen3.8-27B prefill lever: every FP8 projection's prefill GEMM (rows above the decode GEMV band, and every resumed chunk) runs on cuBLASLt's per-tensor-scale e4m3 kernels from boot-requantized per-tensor weights and per-call per-tensor activations — about 2x the dequantized bf16 GEMM's rate (0.59 vs 1.04 ms/token at 2K), +23 GiB resident at the 27B's shape. Lossy beyond the checkpoint: an 8K-prompt greedy completion differs from the exact path. Opt-in; the default prefill dequantizes each matrix to bf16 for cuBLASLt (exact). | false |
| `engine.dflash_model` | no | Qwen3.8-27B: a DFlash2 block-drafter checkpoint (`z-lab/Qwen3.8-27B-DFlash2`, an HF id or a directory) that replaces the MTP draft (`mtp` off). Five bidirectional draft layers fed by the target's layer taps propose seven tokens per step through the rank-256 selector walk. On one Spark without `decode_graph` it runs on the eager engine (the batched verify graph, the stacked redrafts); on the graph worlds — the fabric (`decode_graph` required) or one Spark with the decode graph — the block proposal is recorded inside the graph step on every rank: the drafter's heads and MLP rows sharded across the ranks with two folds a layer, each rank's vocab-slice top-16 merged through one boundary fold so every rank walks the same proposal, the drafts fed to the next replay on the device. The verify is exact: greedy transcripts equal the same world's MTP depth-5 world's (the same verify dispatch class; 4/4 at worlds 1, 2 and 4); sampled requests accept each draft with its exact probability under the request's temperature, top-k / top-p and penalties (the point-mass rule). Requests with logprobs, a logit bias or a grammar run plain. Past 4 slots the 8-row blocks exceed the 32-row decode batch: the engine batches the slots that fit and replays scalar graphs for a live set beyond them. | none |
| `engine.prefill_group` | no | Several cold prompts that arrive together are prefilled as the spans of one walk (the graph worlds). `false` prefills one prompt per walk: a prompt's prefill then never depends on who arrived with it — the walk's row count selects the GEMM lowering above 128 rows (one chain below it, one pinned cuBLASLt algorithm above since 2026-10-05), so a prompt shorter than 128 tokens started together with others can otherwise read differently from the same prompt alone — at the cost of a burst's prefill throughput. Prompts that join a running request, and prompts above 128 tokens in any group, read the same either way. | true |
| `engine.l2_prefetch` | no | The L2 weight prefetcher: a side stream pulls the next layer's weights (and a collective's neighbours) into L2 ahead of the chain. `false` disables it (one-node Qwen3.8-27B T=1 ran about 4 ms a step slower without it in the 2026-09 sweeps). Environment variables until 2026-10-05; every rank takes the same settings. | true |
| `engine.l2_prefetch_form`, `engine.l2_prefetch_window_mib`, `engine.l2_prefetch_boundary_window_mib`, `engine.l2_prefetch_boundary_rate`, `engine.l2_prefetch_layer_rate`, `engine.l2_prefetch_merge` | no | The prefetcher's levers: the form — `load` reads the bytes through L2, `lines` issues one L2 prefetch per 128-byte line, `touch` one line per 64 KB (the page walks only); the window budget in MiB (1–64) and the boundary windows' budget beside a collective (0–64, 0 takes the window budget; 20 measured on the fabric 2026-10-04); the rates (`off`, `light`, `full`) of the windows that overlap a collective and of those inside the attention layers (`full` slowed the small kernels it ran beside by as much as it saved); whether adjacent ranges merge into one launch (a few dozen instead of ~620 graph nodes a step). | `load` / 12 / 20 / `light` / `light` / true |
| `engine.dflash_verify_graph` | no | With a drafter: replay the multi-slot verify batch as a captured CUDA graph (one static 16- or 32-row replay; a lone slot stays on the scalar path). The measured best; `false` runs the packed eager batch. | true |
| `engine.dflash_draft_batch` | no | With a drafter: one stacked block forward redrafts every speculating slot (the draft weights read once per step); `false` redrafts slot by slot. | true |
| `engine.dflash_depth` | no | With a drafter on the eager engine (one node, `decode_graph: false`): verify only the first N drafts per step (1–7); unverified drafts are re-drafted next step, so transcripts are exact at any value. 0 verifies the whole block. The graph worlds verify the whole block or the scheduled depth (`engine.mtp_schedule`); this cap does not apply there. | 0 |
| `engine.dflash_weights` | no | With a drafter: how its five block matrices (q\|k\|v, o, gate, up, down) are served. `checkpoint` keeps the BF16 (packed lossless 12-bit under `bf16_weights: bf12`); `fp8` serves them as block-128 E4M3 (encoded at load; of the five only the k\|v rows keep a bf16 copy, for the context features' GEMM), half the bytes a decode block. Lossy for the PROPOSALS only — the target's verify is exact whatever the drafter proposes, so transcripts and sampling distributions are unchanged; only the acceptance rate can move. | `checkpoint` |
| `engine.expert_gemm` | no | Packed int4 expert GEMM form for prefill (Qwen3.8 AutoRound hybrid, full GLM-5.3): `wide` (the 64×128 tensor-core tile, eight warps), `wide3` (register decode, three stages), `wide4` / `wide4r` (the four-warp forms), `narrow` (the 32×64 kernel). All bitwise; `wide` is the measured best. | `wide` |
| `engine.expert_gemm_prefetch` | no | How many k-steps ahead the expert GEMM prefetches its weight and activation lines into L2, 0–16 (0 disables). Bitwise. | 3 |
| `engine.expert_tile_list` | no | Launch the expert GEMM over a compact list of the routed segments' tiles instead of a grid over the longest segment. Bitwise; off only for diagnosis. | true |
| `engine.expert_gemm_pair` | no | Run the expert gate and up projections as one launch. Bitwise; measured level, off by default. | false |
| `engine.ngram_prestage` | no | Qwen: gather the next prefill chunk's n-gram table rows while the current chunk runs (with `ngram_table: "mmap"`). Bitwise; off keeps the one-channel staging. | true |
| `engine.graph_batch_min_live` | no | Active-request count at which decode switches from scalar to batched graphs. A lower threshold starts batching earlier; batching may improve throughput while doing extra padded-row work. 0 chooses min(2, `max_concurrency`); explicit values must be 1 through `max_concurrency`. | 0 (automatic) |
| `engine.sampling_candidates` | no | Number of candidate tokens gathered per rank on the sampled-token fast path, 1–256. Smaller values reduce routine work but may trigger more full-gather fallbacks. The fallback preserves sampling correctness; this is not the client's `top_k` parameter. | 128 |
| `engine.bulk_pace_gbps` | no | Sender pacing rate per queue pair for bulk/prefill communication, in gigabits per second. Negative derives the rate from link speed; 0 disables pacing. Override only when measuring network contention—this is not an API throughput limit. | -1 (automatic) |
| `engine.bulk_inflight` | no | Maximum in-flight bulk stripes per lane. More can keep the link busy but increase pressure on buffers and competing traffic. -1 uses the transport default (4); normally leave automatic. | -1 (automatic) |
| `engine.rendezvous_timeout_ms` | no | How long ranks may take to join the transport rendezvous. Increase for slow cold starts or delayed peers; it does not increase an HTTP request's timeout. | 120000 ms |
| `engine.stats_interval_s` | no | Interval between periodic server throughput/state log lines. Shorter intervals give finer operational visibility and more log output. Set 0 to disable periodic statistics. | 10 seconds |
| `engine.reasoning_in_content` | no | Put reasoning text in the response's `content`, separated by the model's `</think>` marker, instead of a separate `reasoning_content` field. Use only for clients that need that combined format; it does not disable reasoning. | false |
| `engine.no_eos` | no | Ignore the model's end-of-sequence token so measurement runs continue to their token budget. Leave false for normal serving, where a model should be allowed to finish its answer. | false |

The application allows up to sixteen request slots. A speculative request uses
`1 + mtp_depth` physical decode rows. GLM-5.3-Flash supports eight batched
rows; full GLM-5.3 supports sixteen; GLM-4.7 and DeepSeek support
thirty-two; Qwen supports sixty-four (sixteen requests at MTP depth 3). Graph families capture the slot prefixes that fit, with scalar
fallback for an unsupported active shape. The startup log reports the selected
capacity and rejects a configuration that cannot fit its required rows.
Qwen's 17-64-token decode walks use kernel-only BF16 lowering, including
BF16 projections retained by FP8-dense storage. Smaller walks retain their
existing dispatch even when MTP expands a token into multiple matrix rows.
The shared 8/12-slot graph
families also affect other models above eight slots and can reduce the number
of scheduled verification depths that fit; see [operations](docs/operations.md).

Sampling defaults come from the checkpoint's `generation_config.json`
and per-request fields. Flags such as `--temperature` override them for
measurement runs. See [operations](docs/operations.md) for memory planning,
MTP depth tradeoffs and deployment checks.

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
`benchmarks/` the probes and the engineering record; `deploy/` the cluster
config.

## Documentation

| page | what |
|---|---|
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
| `benchmarks/README.md`, `benchmarks/results/` | benchmark probes, dated results and reproduction commands |
| `CHANGELOG.md` | the history by milestone |

## Contributing

`CONTRIBUTING.md` describes the build presets, the test discipline, the
evidence a fabric change needs, the code style and where things go.

## License

Apache License 2.0. See `LICENSE`.
