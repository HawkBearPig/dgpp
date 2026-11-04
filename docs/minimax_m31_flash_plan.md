# MiniMax M3.1 Flash implementation and optimization plan

Investigated 2026-11-04 against dgpp `403d3f987964f6cd548348316c86520fb700ec71`.
Status: implementation in progress on the feature branch; real-checkpoint text validation underway.

Implement `MiniMaxAI/MiniMax-M3.1-Flash` as a new model family on the shared
engine, initially at TP4. Keep its published NVFP4 experts and MXFP8 dense
weights resident. Establish plain decode and an independent numerical
reference, then add its bundled DSpark drafter, efficient block-sparse
attention, and measured native activation recipes. Preserve dgpp's exact
default and require explicit engine settings and accuracy evidence for
numerically different recipes.

Four Sparks are the practical first deployment. The published target and
draft contain **235.071 GiB of tensor data**, before replication, caches or
scratch. TP2 is a later capacity experiment; TP1 cannot hold this release
resident. Throughput needs measurement on GB10: upstream B200/B300/GB200
results do not predict performance over dgpp's RoCE fabric.

The port follows [adding a model](adding_a_model.md), the numerical and
state contracts in [continuous batching](continuous-batching.md), and
[benchmark workload and timing](benchmarks.md#workload-and-timing).

## Checkpoint and architecture

Pin the target and `dspark/` together to Hugging Face revision
`2f33a8b9ae49ce7b51c994df97166a7fd059bc2e`. The publisher describes about
428B parameters, 23B active per token, text/image/video input, and a 1M
context. All 60 attention layers are sparse in this release. The older M3
architecture and drafter are not interchangeable with it.
[Model card](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e/README.md).

| Property | Published configuration |
| --- | --- |
| Architecture | `MiniMaxM3SparseForConditionalGeneration`, `minimax_m3_vl`; nested `MiniMaxM3SparseForCausalLM` |
| Transformer | 60 layers, hidden 6144, vocabulary 200064, untied embedding/head |
| Attention | 64 query heads, 4 KV heads, head width 128 |
| Position encoding | RoPE theta 10000000, rotary width 64, 1048576 positions |
| Norms | Gemma RMSNorm, epsilon 1e-6; per-head Q/K norms |
| Feed-forward | First 3 layers dense, width 12288; remaining 57 MoE |
| MoE | 128 routed experts, top 4, width 3072; one shared expert of width 3072 |
| Routing | Sigmoid, correction bias, normalized selected probabilities, routed scale 2 |
| Activation | SwiGLU-OAI, alpha 1.702, limit 7 |
| MSA indexer | 4 query heads, width 128, one shared key head; block size 128; top 16 blocks; initial blocks 0, local blocks 1 |
| Index values | Disabled in every layer; no index-value/output branch |

[Pinned target configuration](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e/config.json).

The inventory below sums tensor data offsets from **all 48 target shard
headers and the one draft shard header**, without downloading the weight
payloads. It matches both indexes: target 250103411200 bytes / 89632
tensors, draft 2301852674 bytes / 97 tensors. GiB means 2^30 bytes.

| Tensor group | Tensors | GiB | Storage |
| --- | ---: | ---: | --- |
| Routed experts | 87552 | 216.422038 | NVFP4 packed U8, E4M3 group scales, F32 global scales |
| Text attention and index projections/norms | 960 | 6.389980 | MXFP8 and BF16 norms |
| Shared experts | 342 | 3.099792 | MXFP8 |
| First three dense MLPs | 18 | 0.652588 | MXFP8 |
| Embedding and output head | 2 | 4.579102 | BF16 |
| Routers and correction biases | 114 | 0.167019 | **F32**, including the router matrices |
| Remaining text norms | 121 | 0.001385 | BF16 |
| Vision tower and projectors | 523 | 1.615048 | BF16 |
| Bundled draft | 97 | 2.143767 | MXFP8 linears, BF16 auxiliary weights |

[Target tensor index](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e/model.safetensors.index.json),
[draft tensor index](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e/dspark/model.safetensors.index.json).

### Weight binding and numerical details

An MXFP8 linear stores E4M3 `weight[N,K]` and U8
`weight_scale_inv[N,K/32]`. Decode the latter as E8M0 powers of two;
the suffix does not mean the loader should invert those values.
Reuse `GlmQuantMatrix::scales_e8m0` and the scale-grid slicing machinery
in `src/models/qwen/loader.cpp`. Validate every pair's shape and dtype.

Each routed expert's `w1`/`w3` has logical shape `[3072,6144]`, and `w2`
has `[6144,3072]`. Each has `weight_packed[N,K/2]`,
`weight_scale[N,K/16]`, and scalar F32 `weight_global_scale` and
`input_global_scale`. The weight global is an **encode scale**: the
dequantized value is E2M1(code) times E4M3(group scale), divided by that
global. Preserve separate gate/up globals. Do not confuse this with a
ModelOpt export's multiplier convention. The supported upstream dynamic
activation path neutralizes static input globals; blindly treating these
sidecars as static activation scales would implement a different recipe.
[Upstream quantization implementation](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/layers/quantization/minimax_m3_nvfp4.py).

The shared MoE executor needs two explicit extensions. Its current router
accepts BF16 weights; this model's F32 gate must stay F32. Its clamped
SwiGLU also differs from MiniMax's operation:

```text
g = min(gate, 7)
u = clamp(up, -7, 7)
activation = g * sigmoid(1.702 * g) * (u + 1)
```

Implement the operation and its rounding sites in both grouped prefill and
slot decode, including fused quantization. Keep the existing families'
behavior unchanged. Shared experts need the same activation. Preserve
the placement of the routed scale relative to accumulation and rounding;
selection uses the biased scores, while mixture weights use the unbiased
scores.
[Activation reference](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/kernels/ops/activation/minimax_m3_swiglu.py),
[routing and layer reference](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/models/minimax_m3.py).

### Sparse attention semantics

For each query position and index head, score a block by the maximum
query/key dot product over its causally visible tokens. Select 16 blocks
per index head, forcing the current local block **inside that budget**.
The corresponding group of 16 main query heads attends to the selected
blocks through its KV head, with ordinary causal masking and scaled
softmax. A selected block contributes up to 128 tokens, so each main head
visits at most 2048 tokens. Define and test deterministic ties, short
prefixes, padded selections and partially filled blocks.
[Scoring and selection reference](https://github.com/vllm-project/vllm/blob/5225a1f17f9cf721efd5ad793e6b63798ca30306/vllm/models/minimax_m3/common/ops/index_topk.py).

The index cache contains a key **for every token**, not an average per
block. At TP4, each rank can own one index-query head and its main
query/KV group; replicate the single index-key projection/cache. This
avoids a new cross-rank index reduction. Selection stays separate per
head. dgpp's QSA compressed-key/ReLU-sum indexer and DeepSeek CSA2 are
useful implementation examples but do not implement this algorithm.
[Indexer cache reference](https://github.com/vllm-project/vllm/blob/5225a1f17f9cf721efd5ad793e6b63798ca30306/vllm/models/minimax_m3/common/indexer.py).

Sparse final attention does not make the whole model constant-cost in
context. Index scoring still scans the visible token keys for each query:
linear work per decode query and quadratic total indexing work during a
cold prefill. This must inform both kernel priorities and context claims.

### Bundled DSpark

The five-layer drafter uses dense 12288-wide MLPs, the target's hidden
width and GQA geometry, Gemma norms, partial RoPE and a 4096-token sliding
window. It consumes target features from layers `[3,17,31,45,59]` through
a BF16 `[6144,30720]` projection. Its two vocabulary-by-256 Markov matrices
and `[1,6400]` confidence projection remain BF16. It shares the target
embedding and output head. The configured block is **7 proposals**;
target verification uses the anchor plus those proposals, up to 8 rows.
Noise token ID is 200058.
[Draft configuration](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e/dspark/config.json).

Reuse the speculation protocol and feature-cache concepts, with a new
MiniMax draft adapter. `DFlash2Config::block_size` currently counts the
anchor, whereas this checkpoint's DSpark block size counts proposals;
passing it through unchanged is an off-by-one defect. DeepSeek's DSpark
adapter uses a different backbone and feature representation. MiniMax's
draft queries attend bidirectionally within the proposal block; target
verification remains causal. Context KV comes from projected target
features, not a replay of the draft transformer over the entire prompt.
[Draft execution reference](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/models/dspark_minimax.py).

Also evaluate `RadixArk/Minimax-M3.1-Flash-Dflash2`, revision
`923bf1d4b567d5ccb71b0488c6e61b5e4794e92a`, after the target is correct.
It uses the same feature taps, five layers and 8-row verification width,
with a DFlash2 selector and convolution components. Its 2.055B parameters
are BF16. The architecture is `MiniMaxDFlash2DraftModel`: despite its
`qwen3` model-type field, dgpp's existing Qwen-only DFlash2 execution needs
Gemma norms, partial RoPE and SwiGLU-OAI extensions to run it correctly.
Compare its memory, acceptance and total step cost against bundled DSpark;
the publisher's acceptance measurements at 16K context on GB300 do not
establish GB10 throughput or full-context acceptance.
[Alternative draft](https://huggingface.co/RadixArk/Minimax-M3.1-Flash-Dflash2),
[pinned alternative configuration](https://huggingface.co/RadixArk/Minimax-M3.1-Flash-Dflash2/blob/923bf1d4b567d5ccb71b0488c6e61b5e4794e92a/config.json).

## Placement and memory budget

Use the existing tensor-parallel placement: slice every expert's
intermediate dimension, rather than routing tokens to remote expert
owners. At TP4 the expert slice is 768, dense-MLP slice 3072, main query
count 16, main KV count 1 and index-query count 1. These slices align to
the 16/32-wide quantization groups. Fold attention and FFN output
partials with the existing boundary reducer and RoCE bus. Keep routers,
norms and embedding replicated; shard the vocabulary head. The initial
vision implementation replicates the encoder and broadcasts processed
pixels through the existing image admission journal. A head-only encoder
would need a new feature-delivery protocol and remains a measured follow-up.

Use 128-token physical pages to match MSA blocks. Every snapshot and
copy-on-write operation must cover the corresponding main KV and index-key
planes, plus their scales if quantized. Test partial-page copies and
rejected speculative rows so stale keys cannot enter later selection.

Estimated raw resident weights under that placement are 59.705 GiB per
TP4 rank for the text target, plus 1.615 GiB of vision weights on each rank. A
conservative initially replicated draft adds 2.144 GiB per rank: about
**61.85 GiB text / 63.47 GiB with vision**. Sharding the draft and using
validated lossless BF12 storage can reduce this. These are calculated
weight budgets, not measured process memory; include temporary loader
storage, resident-image construction and startup peak separately.

The same conservative layout at TP2 reaches about 119.05 GiB text /
120.67 GiB with vision, before any cache or workspace. Even after lossless
packing, useful context and zero-swap headroom need proof. Do not ship a
TP2 template by dividing the checkpoint size by two. Keep TP2-compatible
binding tests; reconsider deployment after the TP4 memory plan is measured.

For TP4, target cache payload per token per rank is:

```text
60 layers * (2 * 1 KV head * 128 + 1 index-key head * 128) * bytes/element
```

| Target cache representation | Bytes/token/rank | GiB at 128 Ki tokens | GiB at 1048576 tokens |
| --- | ---: | ---: | ---: |
| BF16 main KV and index keys | 46080 | 5.625 | 45.000 |
| FP8 main KV and index keys, payload only | 23040 | 2.813 | 22.500 |
| NVFP4 main KV and index keys, including g16 scale bytes | 12960 | 1.582 | 12.656 |

These are candidate layouts, not equivalent numerical recipes. Add scale
metadata where needed, draft cache/history, partial-block copies, prefix
snapshots, graph buffers, logits, loader peak and scratch to the plan.
The pool is shared across requests: 1M total cached tokens does not mean
1M tokens for every concurrent request. Index scoring alone reads up to
15 GiB of BF16 keys per TP4 rank at 1M context for a single query across
all layers, before accounting for reuse between verification rows.

Start with a 128K or 256K BF16 pool and expand only after memory and
long-context correctness checks. Avoid allocating all five draft layers'
KV over the entire target pool without accounting for it: at TP4 that
would add 2.5 GiB at 1M tokens in BF16. A bounded draft window can save
space, but needs explicit snapshot, reuse, rejection and rollback rules;
its 4096 attention mask alone does not prove that eviction is safe.

## Implementation sequence

Each milestone has a reviewable exit gate. Shared executor changes should
land separately with existing-family regression coverage.

| Stage | Changes | Exit gate |
| --- | --- | --- |
| 1. Numerical contract and bindings | Add `src/models/minimax/{config,binding,loader}.*`, architecture detection, format checks and memory accounting. Add independent Python reference and generated tiny checkpoints. Inventory nested `dspark/` in download/preflight/resident-cache identity. | All tensors accounted for or explicitly assigned to vision/draft; TP1/2/4 synthetic slices reconstruct original values; wrong formats/shapes fail. |
| 2. Shared arithmetic and MSA | Add F32 router input support, SwiGLU-OAI variants, Gemma norm/partial-RoPE checks, paged MSA score/select/attention kernels and reference. Preserve native payloads and distinct scale conventions. | Independent kernel and isolated-layer comparisons, including route/block decisions, 127/128/129 boundaries and near ties. |
| 3. Plain text engine | Implement `src/models/minimax/model.*` over `SessionModel`, `RowRun`, `PrefillContinuation`, `PrefillBatch` and `GraphBatchMapping`; add cache planes and TP boundaries. | Eager/graph forward and decode, mixed readers/decoders, prefix snapshots, cancellation, pool exhaustion and rollback; real TP4 plain serving. |
| 4. Text API | Wire `ServeFamily` in `apps/dgpp_serve.cpp`, tokenizer/template, MiniMax reasoning/tool dialect and grammar. Register settings, model map and a provisional TP4 template. | Byte-exact prompt/token fixtures, tools/structured output, streaming splits, stop/EOS/usage/logprobs and request isolation. |
| 5. DSpark | Add MiniMax draft execution, feature taps, context KV, Markov/confidence heads, graph proposals, sampled proposal bookkeeping and adaptive verification scheduling. | Plain/speculative greedy identity, forced rejection at every position, correct sampled acceptance, snapshot/resume and C1/C2/C4/C8 operation streams. |
| 6. Images and video | Add native MiniMax vision adapter and preprocessing; integrate existing frontend/image-prefill interfaces and the required video handling. | Processor/embedding parity, image/video ordering and position tests, multimodal task regressions, plain/speculative parity after media prompts. |
| 7. Optimization and release | Tune kernels and scheduler using the gates below; publish only measured template settings. Update model card, deployment catalogue, configuration, operations, README, CHANGELOG and design/status docs. | Full required CI/CTest plus one complete campaign per shipped template and restored production. |

Stage 3 can deliver a useful explicitly text-only checkpoint milestone;
full model support includes stage 6. Unsupported media must be rejected
explicitly until the corresponding frontend is available.

Every numerical setting must follow the repository's seven-part engine-key
contract: config validation, rank settings, CLI, pre-build setter, tests,
configuration reference and operations notes. Include exhaustive weight
class switches, resident-cache path wiring and Python portability checks.
Keep the template's site paths empty and use the common cluster launcher.

### Reference and tests

The checkpoint provides configuration and processors but no full modeling
implementation. Use pinned SGLang and vLLM code as executable semantics
references, plus an independently written small host/PyTorch oracle.
Pin the reference backend, precision, quantizer and runtime settings with
each dump; framework defaults differ and can introduce conversions.

Implement `minimax_forward_test` and `tools/minimax_reference_dump.py`
following the binding-generated fixture and smoke/generate/decode pattern.
Include arbitrary F32 router values, distinct gate/up global scales,
large E8M0 exponents, quantizer clipping/ties, all-selected short contexts,
and nontrivial MSA selections. An all-zero or all-local fixture would miss
the important failures.

For real weights, compare isolated sublayers starting from reference
inputs as well as an end-to-end teacher-forced trajectory: norms, Q/K and
index projections, block scores/IDs, attention, router scores/IDs/weights,
expert outputs, residuals, logits and logprobs. Use the existing
[route and top-1 certification approach](benchmarking.md#glm-reference-dumps-and-the-curated-suite-correctness-not-bandwidth)
for floating-point near ties. Set tolerances from the chosen arithmetic
and independent oracle before tuning; this repository does not provide a
single universal cosine or accuracy percentage that can certify this port.

### Prompt and media compatibility

Use the shipped Jinja template as the prompt oracle. It maps user-supplied
system/developer content into its developer section, has an optional root
role, preserves prior reasoning, emits `<mm:think>` markers and uses
namespaced recursive XML tool arguments. Its generation suffix is an
assistant header and its default thinking mode is adaptive; the template
does not implement an `enable_thinking` switch. Test the interpreter's
recursive macros and mapping/list operations, JSON arguments, null-field
omission, multiple calls/results, escaping and split streaming markers.
Add a dedicated tool dialect rather than forcing the existing Qwen/DSML
parsers to interpret it. Benchmark clients must record the actual reasoning
mode instead of claiming that an ignored flag disabled it.
[Chat template](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e/chat_template.jinja).

Vision uses a 32-layer, 1280-wide encoder, 14-pixel patches, 3D RoPE and
spatial/temporal patch merging. Port the published image normalization,
dynamic resolution, frame segmentation, feature ordering and projector
behavior; sharing an existing vision tower without these rules is
incorrect. Use the processor's effective limits: instantiating the pinned
fast image processor confirms `max_pixels=451584` (672²), or 576 merged
tokens. A 1008-square input becomes a 672-square canvas. The nested
`max_image_resolution=1008` metadata does not override this processor's
pixel budget. Video also needs native decoding/frame sampling
and API validation, beyond the existing image/PDF frontend.
[Image processor](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e/image_processor.py),
[processor settings](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e/preprocessor_config.json),
[video processor](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e/video_processor.py).

## Throughput optimization priorities

Optimize aggregate committed decode tokens/s at C1/C2/C4/C8, with C1
latency and cold prefill retained as separate objectives. There may be
different winning draft depths at different occupancy/context lengths.
Do not select a configuration by isolated GEMM speed or draft acceptance
alone.

1. **Preserve compressed residency and remove repeated traffic.** Keep
   MXFP8 E8M0 scales byte-sized and NVFP4 payloads packed; avoid full-matrix
   BF16 bridges or persistent duplicate expert layouts. Reuse dgpp's
   weight tiles, grouped expert execution and resident-image lifecycle.
   Use lossless BF12 only where the existing encoder validates exact
   reconstruction; keep the F32 routers and BF16 head numerics intact.

2. **Batch verification to amortize weight reads.** The selected routed
   weights alone total about 6.76 GiB per target token across the world
   at C1, calculated as 4/128 of the routed inventory. Dense projections,
   shared experts and the head add further traffic. DSpark can reuse
   weights and index-key tiles across multiple verify rows. Sweep plain
   and fixed verification depths 1/3/5/7, then fit the existing
   `verify_schedule.hpp` policy to this model's measured base/row costs.
   Retain the drafter's configured proposal width unless a smaller
   proposal block is independently validated. At higher occupancy,
   shallower verification or plain decode may win. Optimize
   `expected committed tokens / (draft + verify + bookkeeping time)`.

3. **Build MSA for SM121a.** Fuse Q/K and index normalization, partial
   RoPE and cache insertion where the reference's rounding survives.
   Compute exact token scores, reduce block maxima and maintain top-16
   candidates without an N-by-N score allocation. Tile prefill across
   queries to reuse each index-key tile; reuse it across speculative rows
   in decode. Start with a bounded score workspace and advance to fused
   streaming selection. Use stable selection and reduction ordering so
   shape changes cannot silently change the model.

4. **Add a true native MXFP8 activation path and MiniMax W4A4 path.**
   Existing E8M0 weight support is primarily W8A16; `fp8_gemm.hpp`'s
   activation quantizer uses 128-wide F32 scales, not MiniMax's 32-wide
   E8M0 activation groups. Implement the required quantizer and product,
   including rounding/saturation, and extend the existing NVFP4 GEMM's
   fused activation to SwiGLU-OAI. Verify SM121a instruction support and
   accumulation behavior rather than importing SM100-specific kernels.
   Benchmark small decode/verify rows and
   large prefill separately. Keep the W8A16/W4A16 checkpoint-weight
   reference. Ship native W8A8/W4A4 through an explicit, validated recipe;
   QAT is a reason to test it, not evidence of bitwise equivalence.

5. **Tune mixed execution and fabric costs.** Keep routes and selection
   on the GPU, use graph capture for decode/verification and the existing
   bounded prefill scheduler. Fuse residual/norm and local epilogues,
   and overlap shared/routed work only when accumulation remains fixed.
   Measure the roughly two target boundary reductions per layer and
   draft reductions before altering placement. TP4 already gives local
   index selection; expert-parallel all-to-all over 200 Gb RoCE is a
   later measured experiment, not the starting architecture.

6. **Evaluate cache precision separately.** BF16 main KV/index keys
   establish a conservative baseline. FP8/NVFP4 main KV, FP8/NVFP4 index
   keys and query quantization are distinct experiments. Index rounding
   can change block selection even when hidden-state error is small.
   The top-level checkpoint config does not fully specify cache formats,
   and upstream recipes choose different ones. Expose each numerical
   choice with an engine key and require long-context retrieval and
   quality evidence. Never lower top-16, enlarge blocks, pool index keys,
   truncate context, or drop the shared expert as an invisible speedup.

7. **Optimize proposal-only work and prefix reuse.** Once verification
   is exact, draft-head screening or cheaper draft arithmetic can be
   evaluated by acceptance and net throughput. For sampled decoding,
   record the actual proposal distribution and use correct rejection
   sampling; heuristic acceptance is not allowed. Cache target KV,
   index keys and required draft state through the common prefix/NVMe
   interfaces. Measure prefix reuse as service behavior; keep cold
   prefill probes uncached and active weights resident.

The upstream MSA package targets SM100, and its documented environment
is x86_64 with aarch64 untested. It is an algorithm/kernel reference for
this port, not a supported drop-in dependency for dgpp's SM121a build.
Likewise, upstream fused TP4 MonoKernel results use a different
interconnect and execution environment. Do not budget a second expert
copy on Spark to reproduce that implementation.
[MSA requirements](https://github.com/MiniMax-AI/MSA/blob/80434d7f67877c6570ca19cac444b84bc9855dac/README.md),
[SGLang deployment guidance](https://docs.sglang.io/cookbook/autoregressive/MiniMax/MiniMax-M3.1).

Profile each candidate with the existing step/fabric timing tools and
focused MSA/MoE microbenchmarks before the full campaign. Attribute time to
dense projections, routing/experts, index scan/select, selected attention,
collectives, vocabulary head and draft/context-KV work. Sweep occupancy,
context, verification depth and prefill chunk size separately before
combining winners. Record accepted tokens per verification, actual committed
tokens/s, TTFT, inter-token latency and peak memory. Include C16 only if
the memory plan and row capacity support it; retain C1/C2/C4/C8 as the
published comparison. Choose settings from end-to-end gains that survive
the correctness gates, and remeasure the final combined template.

## Accuracy and release gates

The accuracy standard has three separate parts; none substitutes for the
others:

- **Checkpoint fidelity:** correct formats, scales, F32 routers, activation
  formula, attention selection and independent forward comparisons. A
  plausible answer or high final hidden-state cosine cannot certify these.
- **Engine invariants:** identical greedy committed tokens for plain versus
  speculative execution under the same numerical recipe; repeatable eager
  and graph execution; chunk/resume/cache consistency; isolated mixed rows
  and correct request state. Compare first/last tokens as well as digests.
  Rank operation-stream equality proves schedule agreement, not numerical
  correctness. Investigate shape-dependent differences rather than treating
  them as normal MoE noise.
- **Task quality:** establish this checkpoint's reference HumanEval/GSM8K/
  extraction baseline using the maintained 300-item campaign setting,
  then inspect paired per-item changes for each candidate recipe. Add
  tool/reasoning and image/video cases, plus retrieval/position tests at
  32K/128K/256K/512K and the claimed maximum. Timing or successful memory
  allocation alone does not certify 1M-token accuracy. Follow the campaign's
  cap/retry accounting and use a larger held-out set when the 300-item set
  cannot resolve a proposed precision tradeoff.

There is no preauthorized accuracy-loss allowance. Keep numerical changes
opt-in with their measured transcript/quality costs; do not trade a
percentage of quality for throughput by inventing a new tolerance. Native
quantized execution must be compared both with its pinned upstream recipe
and with dgpp's conservative checkpoint-weight path. Greedy equivalence
is required for speculation within a recipe, not asserted between two
different precision recipes. Sampled verification must preserve the
target distribution, with exact small-distribution tests and end-to-end
statistical checks rather than a demand for identical random transcripts.

Before release, run the required CI build and full serial CTest on an idle
GPU, including the other families. Exercise the new fixture in the shared
prefill contract, and run TP4 restart/resident-cache, prefix/NVMe,
cancellation and pool-boundary checks. Cover block boundaries at
127/128/129 and 2047/2048/2049 tokens and draft-window boundaries around
4096, including rejection and prefix attachment at partial blocks.

The intended first template is
`deploy/cluster_minimax-m3.1-flash_nvfp4-mxfp8_w4.example.json`.
Its final cache capacity, slots, precision recipe and speculative depth
must come from the validated configuration. Run its complete published
row set with:

```sh
python3 scripts/serve_campaign.py run \
  --config deploy/cluster_minimax-m3.1-flash_nvfp4-mxfp8_w4.example.json
```

Extend the campaign's plain-mode cleanup for any new draft keys and make
its thinking-mode handling honest for this template. Use one binary for
comparisons and record source revision, binary hash, checkpoint revision,
resolved settings and every rank's zero swap before and after. Verify
zero cached tokens in every cold probe and identical rank operation
streams. Keep the fabric otherwise idle throughout; restore production
once at the end. Fix any defect the run exposes before publishing its row.

Report engine decode tokens/s by concurrency, cold prefill seconds, and
wall throughput only in its labelled service-level table. Keep generated
headers, dumps, benchmarks and logs untracked; reusable findings belong
here and validation summaries belong in the PR. This investigation has
not run the model or measured GB10 throughput; the remaining decisions
are kernel numerics, native-recipe quality, draft acceptance, and measured
memory/runtime costs.

## Implementation record: 2026-11-04 baseline

The pinned target and nested DSpark weights are downloaded and verified on
all four cluster nodes. The initial text implementation binds all 89,109
text tensors, retains FP32 routing weights, runs MXFP8 weights with BF16
activations and NVFP4 experts with BF16 activations, and implements paged
MSA including the independent per-head block selection. It uses the shared
session, prefix-cache and graph interfaces. The recursive tool-call parser
and grammar are implemented. DSpark and vision remain under development;
this is a feature-branch baseline, not a completed model release.

The synthetic four-layer checkpoint passes repeated forward, partial-block
prefix restore, continuation and request-isolation smoke tests. An
independent NumPy teacher-forced reference reports maximum per-layer relative
RMS error 0.003312, exact routed-expert selections, and output-head relative
RMS error 2.27e-7. Kernel and forward CUDA memchecks report zero errors. The
shared attention regression tests pass for Flint and MiMo. These fixture
results do not establish real-checkpoint quality or throughput.

Prompt validation covers the shipped Jinja template and HF tokenizer,
including recursive XML history, NFC, the case-sensitive split pattern,
contractions, Unicode and adaptive `<mm:think>` reasoning. The native
renderer adds the Jinja boolean test required by the shipped template.

### Real-checkpoint baseline investigation

The pinned target and bundled DSpark snapshot are downloaded and checksum-synced
to all four nodes. A TP4 baseline boots with 59.72 GiB of text weights per rank
and zero process swap on all ranks. Initial chat smoke tests run; throughput
figures are withheld until the complete deployment campaign and accuracy gates.
The smoke exposed an initial adaptive `</mm:think>` delimiter leaking into
content; the parser now handles that delimiter, with a regression case.

A five-token real-checkpoint probe records all 60 residuals and sublayer
boundaries. Every rank has identical residuals and expert selections. An
independent NumPy FP64 reference, with the same recorded sublayer inputs and
explicit TP4 BF16 folds, reports maxima of 0.005070 relative RMS for attention
and 0.000833 for the feed-forward outputs. All 57 routers select the same
experts from the recorded inputs; the final vocabulary head has relative RMS
6.37e-6. A whole-layer reference first diverges at layer 26 because a small
attention difference crosses a near-tied expert boundary. That original
whole-layer failure is retained as an investigation finding, not waived by
the isolated checks. Full-model NLL validation remains required.

`minimax_forward_check` records layer/substage outputs or scores a fixed
teacher stream through session verification. `tools/minimax_reference_dump.py`
checks independent whole-layer or isolated-stage math. The optional PyTorch
reference streams the real checkpoint one layer at a time without loading
dgpp kernels; it simulates TP folds and scores complete texts with FP32
attention. Its default follows the published fused normalization of the
unrounded residual sum. The explicit `--rounded-residual` diagnostic isolates
the initial baseline's BF16 residual-normalization boundary. These tools
provide numerical evidence; their elapsed times are not serving benchmarks.

The first independent full-text check (529 tokens, 528 scored positions,
quick teacher text) failed against the published normalization recipe:
absolute mean NLL delta 0.10734 nat and 7.01% of positions moving by over
1 nat. The baseline's NLL was lower, but the bidirectional numerical gate
still fails. The current baseline normalizes the unrounded FP32 residual
sum while storing the BF16 residual, as the published fused norm does.
Selected-block attention retains FP32 probabilities. The ordinary sink
attention variants keep their existing arithmetic.

The real probe also exposed a shape-dependent MXFP8 projection dispatch:
short verification and long prefill selected different reduction chains.
The MiniMax baseline now uses the streaming projection at every row count.
Full prefill and eight-row verification then produce exactly identical
teacher scores at all 528 positions. Two same-process trials and a separate
process repeat have identical logit hashes, with hidden-state agreement
across all four ranks. Synthetic fixtures additionally require exact
full-prefill versus incremental hidden states and verify that recording
layer states does not change the execution. CUDA memcheck reports zero
errors for the changed norm and the complete fixture/session smoke.

The corrected baseline still fails the independent PyTorch trajectory
comparison: absolute mean NLL delta 0.09451 nat, with 10.98% of positions
moving by more than 1 nat. An independent replay of all 60 layers using
the recorded inputs has maximum sublayer RMS 0.002023 and identical expert
selections at all 57 MoE layers. These isolated checks locate the discrepancy
without certifying end-to-end quality. The reference's `--projection-rows`
control measures its own matrix-shape sensitivity; `--check-engine-dir`
checks recorded sublayers separately. Changing only the PyTorch projection
batch size to eight rows also fails the same gate against its full-row
reference: absolute mean NLL delta 0.13521 nat and 10.98% large moves.
This control establishes that cuBLAS shape-dependent reductions alone can
produce discrepancies of this size on the quick text. It does not relax the
accuracy limits or certify dgpp. A fixed arithmetic reference is required
before this comparison can support a release verdict. Investigation remains
open and the feature branch is not ready to merge.

Short batches now compact nonempty expert segments before the three routed
matrix launches. The launch bound is `min(experts, rows * top_k)`; it changes
neither row offsets nor the order of any dot or expert accumulation. The
four-rank quick teacher run has identical logit and hidden hashes for every
chunk in both repeat trials, with exactly zero NLL delta. Fixture, mixed
reader/verifier/rollback graph checks and CUDA memcheck pass. The optional
`minimax_moe_bench` target compares both schedules on one real checkpoint
layer and recorded `ffn_norm` inputs under CUDA graphs. It requires exact
BF16 outputs before timing; its local layer timings are not deployment
throughput figures.

The graph audit exposed missing programmatic dependency waits in all nine
new MiniMax kernels. Every kernel now waits before reading its inputs. A
regression graph deliberately releases its dependent before its last
producer row is ready: it fails when the norm's wait is removed and passes
with the wait restored. All nine MiniMax tests, the mixed prefill/verify/
rollback graph contract, and kernel CUDA memcheck pass after the fix.
Earlier serving smoke tests used the unsafe graph binary and do not qualify
as release evidence.

The optional Triton reference fixes the projection reduction chain and
checks exact row-batch invariance, basis projections, and a condition-aware
FP32 error bound against FP64. Its raw quick-text comparison still fails:
absolute mean NLL delta 0.06404 nat and 9.28% large moves. Fixing reference
matrix dispatch alone does not resolve the whole-trajectory discrepancy.

Both numerical probes now support conditional scoring while retaining the
entire prompt in the forward pass. They record the complete token hash and
the first scored position, and test the scored suffix against full scoring.
With the shipped chat template, a 211-token prompt asking for an overview
of Lyon, and the same quick teacher article as 529 assistant target tokens,
the native TP4 mean NLL is 1.7241 versus 1.7250 for the independent PyTorch
reference. The likelihood gate passes (absolute mean delta 0.00087 nat,
zero positions moving by over 1 nat). Twelve argmax choices differ; this
likelihood result does not establish the separate greedy-flip limit, broad
task quality, or resolve the failed raw-text control. The branch remains
under development.

### Recursive tool calls

MiniMax uses its own namespaced XML parser and grammar. The parser retains
recursive parameter schemas, preserves raw scalar strings, handles arrays
and empty containers, and emits all invokes in a block atomically. Invalid
or truncated blocks retain their literal bytes and token provenance.
The grammar translates XML into the existing JSON schema machine, keeps
its cursor isolated across speculative copies, and supports tools alongside
structured JSON output and adaptive reasoning. Scalar masks reuse the JSON
vocabulary tables; structural masks visit only possible leading bytes.

Tests cover token splits across every tag, schema references, nullable
types, enums, required keys, numeric bounds, array limits, extra properties,
duplicate keys, literal JSON punctuation inside scalars, malformed fallback
and speculative cursor isolation. The pinned tokenizer/template corpus
round-trips recursive tool histories; real-vocabulary masks are checked
against independent per-token simulation. The existing tool and serving
regressions remain required alongside these family-specific checks.

The template omits null object fields and gives scalar strings no escaping
for its own structural delimiters. The grammar refuses required schemas
without a finite non-null XML spelling. It supports nullable `anyOf` with
one non-null alternative; multiple non-null alternatives and container
`enum`/`const` are refused by field at request preparation because their
ambiguous XML value typing is not implemented. Scalar enums, nested typed
objects/arrays and local references are supported. These are tool-format
constraints; JSON response schemas continue to use the existing JSON path.

### Four-rank API and graph checks

Source `2c5de769`, server SHA-256
`7bc941e22c28dadd625e01f9e3ff4e41eda9be99b13ad12debaff9bc88bccee2`,
passed the general API checks with BF16 KV, graph decode, and no draft.
The checks covered streaming stops, multiple completions, logit bias,
invalid-request errors, cold and reused prefixes, reasoning accounting,
and recursive tools with schema constraints. Nested object/array calls
passed both streaming and non-streaming validation, including preservation
of the string `"123"`, booleans, and integer arrays. Unsupported required
null tool parameters returned a field-specific 400 error.

Four deterministic 256-token transcripts (chat, code, arithmetic, JSON)
matched exactly between graph and eager decode, including raw reasoning,
content, finish reasons and token counts. Every rank was at zero process
swap with the same binary during these API and parity checks. The graph
world's operation streams matched at `ad363b2f8e421d3fa9be1f6e913f6a29`;
the eager world's matched at `0522926a6595ea7446c17dfd108544e8`. These
transcripts reached their token caps and establish execution parity, not
task-quality completion.

The separate maintained tool campaign also passed its structural checks
with matching rank streams, but rank 0 had swapped pages. Its timings are
excluded. These smoke tests supply no deployment throughput rows; those
require the complete shipped-template campaign.

### Bundled DSpark implementation

The native draft loader retains MXFP8 projections and expands only the
BF16 confidence weights to FP32. The shared block executor now supports
MiniMax Gemma norms, partial rotary embeddings, dense SwiGLU-OAI, a
vanilla Markov head, and learned acceptance confidence. Context features
come from the completed residual outputs of target layers 3, 17, 31, 45 and 59. Their
FC result rounds to BF16 before normalization. Draft K/V planes share the
target's block table, including prefix copies and the NVMe cache tier.

The initial proposal recipe considered the top 16 base-logit candidates
per mask position before adding the predecessor's Markov bias. After its
first serving probes, the native path now applies the bias across the full
vocabulary and gathers the corrected top 16 for each sequential proposal.
Greedy selection therefore scans every token. Sampled verification records
the actual truncated distribution, including request sampling settings;
this proposal distribution is not upstream's full-vocabulary sample.
The target verifier remains authoritative. Learned confidence drives the
existing verification scheduler; proposal-width and schedule tuning need
measured acceptance and complete serving campaigns.

The published 2.3 GB draft passed exact source-byte checks, FC column
reordering checks, BF16-to-FP32 confidence checks, and all TP2/TP4 tensor
slices. Synthetic tests additionally cover the 16:1 attention head ratio,
strided norm/RoPE, FP32 attention probabilities, context rounding, and
predecessor-conditioned confidence. The native fixture has an independent
FP64 forward oracle; its taps are checked against the target's completed
layer residuals, including the final layer before normalization. Release
validation remains in progress.

The initial fixture oracle reported FC relative RMS error `6.2e-8`, exact
context normalization, draft K/V relative RMS errors at most `0.000822`,
and final draft hidden-state relative RMS error `0.002163`. That run used
an incorrect feature-tap convention shared by the adapter and oracle;
it does not establish fidelity to the trained input features. The corrected
capture gate is described below. Output-head projections, vanilla Markov
selections and learned confidence agreed with the independent reference
at the observed BF16 hidden boundary. Sixteen selected CTest
gates pass, including all 18 shared drafter integration cases, plain versus
speculative greedy transcripts, every acceptance length, two-rank greedy
and sampled graph execution, prefix copy/restore and the NVMe tier.
The tier harness sizes staging chunks and restore positions from the
model's physical block size. Kernel and native draft-forward memchecks
report zero errors; the MiniMax shared-prefill test passes. Four-node real
model speculation and acceptance/throughput tuning remain separate gates.

The first real DSpark boot caught a remaining deployment-parser ceiling of
five draft tokens. The parser and CLI now admit seven; a family check
preserves the five-token ceiling for existing MTP chains and requires the
checkpoint's full physical block width for MiniMax. A deployment regression
test covers the seven-proposal configuration. MiniMax uses the verification
scheduler to reduce verified rows without changing that physical block.

The next real-model startup exposed five cuBLAS memset nodes in the
24-row verification graph, one per target feature projection. Those FC
projections now use the kernel-only streaming matrix multiply with one
accumulation chain for every row count, including prefill and resume.
A regression first failed on the previous dispatch and now requires
bitwise feature equality between full prefill, incremental verification,
short resumed readers, and two/three/four-request batches. All 21 shared
drafter integration cases pass, including greedy and sampled captures at
32/24/16/8 rows. The independent draft oracle and native kernel/loader
gates also pass. The real 6144-wide graph is checked separately at startup;
the tiny fixture alone cannot reproduce cuBLAS's shape-specific memset.

### Four-rank native draft smoke

Source `7b713fdf`, server SHA-256
`9ec1dc26122a01656b6ea5afcd5f1395014bfab7dff1729f28ae4f27282122d1`,
boots the full target and bundled DSpark with four slots and all 8/16/24/32
verification-row graphs. Plain and speculative greedy transcripts match
exactly for the four 256-token prompts, both sequentially and concurrently,
including raw reasoning/content, finish reasons and token counts. The
speculative world also passes the maintained API checks, cold/hot prefix
usage, recursive typed tool calls in one-shot and SSE responses, and the
unsupported-tool-schema refusal. Every rank is at zero swap before and
after both worlds, with matching binaries. Operation streams match at
`6210ef4df1bac9c0fb48d7ba48cd2a53` for speculation and
`fc1071765bff124585ab6e298979e03f` for plain decode.

These are integration probes, not published benchmark rows or completed
task-quality tests. The initial base-logit shortlist draft is slower than
plain decode on this battery; investigate the shortlist's effect on
Markov-conditioned proposals and tune verification depth before choosing
deployment defaults.

The full-vocabulary Markov candidate path reuses the existing FP32 DSpark
projection. It adds six small candidate gathers per draft block at TP4.
Its regression forces successive winners outside the base-logit top 16
and verifies the published predecessor and candidate records. The
single-step sampler is bitwise the existing selector's proposal
distribution and keyed draws for greedy and sampled requests, including
top-k/top-p/min-p truncation. Nine selected CTest gates pass, including
the independent draft oracle and all 21 shared drafter integration cases.
Full-checkpoint acceptance and runtime comparisons follow this change.

### Corrected target feature boundaries

The first full-vocabulary draft run preserved all four greedy transcripts
but still had weak acceptance. Tracing the capture configuration through
both upstream implementations exposed an off-by-one error: checkpoint tap
`L` means the output of target layer `L`, while the runtime captures the
entry of layer `L+1`. Both adapters add one to the configured IDs. The
initial dgpp adapter had read the entry of layer `L` directly. The final
tap therefore belongs after the last FFN, before final normalization.
[vLLM capture-ID conversion](https://github.com/vllm-project/vllm/blob/5225a1f17f9cf721efd5ad793e6b63798ca30306/vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py),
[SGLang capture-ID conversion](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/models/minimax_m3_vl.py).

The corrected oracle reads the named layers' completed residuals and
rejects the old implementation with FC relative RMS error `0.684383`.
The adapter now accumulates each tap after the next fused input norm has
saved that completed residual; the final norm supplies the last tap's
pre-normalization residual. This preserves the target's unrounded norm
input and avoids an extra rounding in the target path. Earlier draft
timings are excluded from release evidence.

With the corrected boundaries, the independent FP64 oracle reports FC
relative RMS error `1.6e-7`, context error `0.000119428`, draft K/V errors
at most `0.00164452`, and final hidden-state error `0.00303526`. The
output-head, full-vocabulary Markov walk and confidence checks pass.
All nine selected CTest gates pass, including all 21 shared drafter
integration cases and the row-invariant feature regression. Four-rank
acceptance and same-binary plain/speculative tests follow this correction.

Source `7e92bba9`, server SHA-256
`f3654de2c9a385cd2cfb43e46e6edcbaa2e71329dbd9ebae9a3396e88f397917`,
passes the four-rank real target/draft smoke after correcting the taps.
The four 256-token prompts match plain decoding exactly in raw content,
reasoning, finish reason and token counts at both C1 and C4. API, cold/hot
prefix usage and recursive one-shot/SSE tool checks pass. Every rank has
zero swap before and after both worlds, with matching executable hashes.
Operation streams match at `e59603ef596a35812ba71bfa45f172ab` for
speculation and `566256b1b51d228f32c148fbb0520371` for plain decode.
Acceptance remains too low to choose speculation as the deployment
default; these probes provide no published throughput rows.

### Native still-image encoder

The encoder loads all 523 published visual tensors: one FP32 patch
convolution and BF16 LayerNorm, attention, MLP, projector and merger weights.
The final memory plan adds 1,734,145,024 weight bytes and 407,793,664 workspace
bytes per rank. Buffers and BLAS handles are created before fabric execution;
request-time encoding does not allocate device storage. Image features enter
the target embedding stream before DSpark feature capture. Chunked and resumed
prefill, snapshots inside image spans, interleaved requests and plain versus
speculative decoding have dedicated synthetic-checkpoint regressions.

The precision recipe preserves the FP32 convolution, FP32 attention scores,
BF16 probabilities, exact GELU, and the 78 rotated dimensions of each 80-wide
attention head. Three arithmetic differences initially failed the existing
0.5% relative RMS / 0.99998 cosine gate and were corrected, without relaxing
that gate:

- FP32 patch projection uses SGEMM, matching the upstream CUDA linear
  lowering for patch-sized Conv3d inputs. cuBLASLt's alternative accumulation
  changes a few BF16 projections at larger patch counts.
- BF16 projections use the CUDA reference's 1 MiB cuBLASLt workspace cap.
  The 64 MiB alternative selects a different split-K/bias epilogue at the
  published MLP shape. The separate SGEMM workspace is 32 MiB.
- Rotary frequencies and trigonometry use the reference's FP32 device
  operations. Computing the power in host double precision before casting
  changes individual Q/K values; those differences accumulate over 32 layers.

[Pinned upstream convolution lowering](https://github.com/vllm-project/vllm/blob/5225a1f17f9cf721efd5ad793e6b63798ca30306/vllm/model_executor/layers/conv.py),
[PyTorch CUDA BLAS reference](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/cuda/CUDABlas.cpp).

`tools/minimax_vision_regression.py` runs the native harness followed by an
independent PyTorch CUDA oracle, serially on an idle GPU. Eight cases cover
28×28 through 896×896, rectangular images, noise and checkerboards, the
128-query attention tile boundary, the processor's 576-token image budget
and the encoder's 1,024-token allocation ceiling. All eight final embeddings
are bitwise identical with the real checkpoint and PyTorch 2.14.1+cu130.
The FP64 tiny-checkpoint oracle remains a separate structural check.
These are encoder regressions; cluster API checks and media task accuracy
are separate gates. Video decoding and temporal segmentation remain pending.

All 28 selected CTest gates pass, including shared prefill and all 21
shared drafter cases; the MiniMax unit filter passes all 19 cases.
Compute Sanitizer reports zero memory errors for the new kernels and
image-prefill/session regression, and zero race hazards for the tiled
encoder fixture. The PDL audit covers all 397 kernels, and documentation
checks pass.

Source `e96ea4ec`, server SHA-256
`d1499a8397b3fdb117c8072c45d5c912cfb7c039c602ff79483455b2f50b3d2b`,
passes four-rank still-image integration with both plain decoding and the
seven-proposal DSpark drafter. Six fixed cases cover color, image ordering,
a chunk-crossing large image, four-digit OCR and left/right layout. Complete
choices, including reasoning and finish reason, plus prompt/completion token
counts match exactly across plain/speculative execution and C1/C4.
The general vision API suite passes for streaming, multiple choices,
large images and concurrent isolation.

Both modes pass the full prefix-cache suite with a 65,536-token KV pool and
2 GiB cache budget: changed pixels miss, continuations and appended images
reuse the correct prefix, cached and cache-disabled answers agree, and
24-image histories exceed 14,000 prompt tokens. The plain history reuses
14,439 of 14,442 prompt tokens; the speculative run reuses 14,436 of 14,439
(the cache-test UUID changes tokenization between runs). Earlier prompts
remain reusable concurrently. A preliminary 32,768-token pool evicted those
earlier entries during the long cache-disabled control and therefore failed
the final retained-hit assertion; the 65,536-token rerun supplies the
required working-set capacity without changing that assertion.

All ranks have zero swap before and after both 65,536-token worlds and the
same executable hash. Operation streams match at
`e22f0665393a2178ab90e4a3c5b63dca` (plain) and
`e11d9bbe2a3a386e5f3a029e735f9103` (DSpark). These are correctness probes,
not published throughput measurements.

### Text and draft rotary frequency precision

The vision precision investigation exposed a related target/draft issue:
both initially built inverse frequencies with double-precision powers and
then cast to FP32. The independent CUDA text reference also explicitly used
FP64 for that operation. The pinned SGLang CUDA recipe computes the power
and reciprocal in FP32 and retains FP32 cosine/sine tables. At theta 1e7,
the two constructions differ in 14 of 32 frequencies. At position 1,048,575
the largest angle difference is 0.03125 radians, with a cosine difference
of 0.0167959. The earlier numerical results therefore describe the initial
frequency recipe rather than exact upstream rotary arithmetic.
[SGLang rotary construction](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/layers/rotary_embedding/base.py).

Target and MiniMax DSpark now build frequencies with the CUDA FP32 recipe;
other draft families keep their existing construction. The text oracle is
corrected independently and records the recipe in its metadata. A fixed
CUDA-reference vector rejects the old construction for both target and
draft table sizes. All 27 selected CTest gates pass, including the image
session tests and all 21 shared drafter cases; CUDA memcheck reports zero
errors and the PDL audit covers 398 kernels. Full-checkpoint teacher scoring
is repeated after this arithmetic correction.
