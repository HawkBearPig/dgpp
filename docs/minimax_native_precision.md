# MiniMax M3.1 Flash: native precision contract

The [publisher's model card](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
recommends SGLang, vLLM and TokenSpeed. It describes a model trained with
quantization awareness. The Blackwell serving paths quantize activations and
attention, in addition to reading the quantized checkpoint weights. The
current DGPP BF16 activation/cache path is a separate comparison recipe.
Failure of that recipe's parity limits does not establish failure of native
inference fidelity or task quality.

The native deployment now passes the phase-aware, independent TP4 reference
on all six corpora (28,329 scored positions). The sections below retain the
initial diagnostics and their corrections. The completed serving campaign
records task quality and throughput separately in the
[model card](model_cards/MiniMax-M3.1-Flash.md) and
[benchmark tables](benchmarks.md).

## Pinned reference

The initial native reference audit uses:

- SGLang `58484246d26523eaf1a7f16ab524deee288d2120`.
- FlashInfer 0.7.2, `d84335d8c54e0c8f9f7e22faa15c7e34a58abf94`, as required by
  that SGLang revision's
  [package manifest](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/pyproject.toml).
- MSA `c8e1013a16a708d2a7656805d5fc26f2b87715d1`, the tested revision named in
  the [SGLang deployment recipe](https://docs.sglang.io/cookbook/autoregressive/MiniMax/MiniMax-M3.1).

This pins source behavior; it is not a claim that the complete upstream
Blackwell stack has been executed on the four GB10 nodes. The upstream MSA
kernels target SM100/SM103 and use facilities absent from SM121. DGPP needs
compatible kernels and independent numerical validation on GB10.

## Projection boundaries

Dense quantized projections use MXFP8 weights and dynamic MXFP8 activations:
32 values share one E8M0 power-of-two scale; values use E4M3. The scale rounds
up to cover the group's maximum. Payloads and scales must be validated,
including zero groups and exponent boundaries, before comparing products.

The pinned SGLang dense/shared MLP path materializes BF16 gate/up outputs.
Its fused SwiGLU-OAI/MXFP8 producer also rounds the activation result to BF16
before quantizing it, preserving the unfused two-kernel recipe. This differs
from a separate no-BF16-rounding fused implementation in the same source
tree. The actual call path, not the availability of a similarly named
kernel, defines this reference.
[MLP dispatch](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/models/minimax_m3.py),
[fused producer](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/kernels/ops/activation/swiglu_mxfp8_swizzled.py).

That producer's sigmoid evaluation is also explicit: it multiplies the gate
by the sigmoid before multiplying by the up term. Its Triton-generated
exp2 and division are not replaced with `gate / (1 + expf(-alpha * gate))`.
DGPP's native dense/shared path now fuses this evaluation, BF16 rounding and
MXFP8 production into one kernel. Payloads and scales match the actual pinned
SGLang fused GPU producer byte for byte on 170,272 pairs, including all
65,280 finite BF16 gate values and a 6,176-wide partial tile. Independent
FP64 error bounds, row splits, graph replays and CUDA memcheck pass.

Routed experts use NVFP4 W4A4. Both GEMM inputs are quantized dynamically per
token. The MiniMax adapter explicitly neutralizes the checkpoint's static
input globals; those globals must not be reused as static activation scales.
The SwiGLU-OAI function uses alpha 1.702, an upper gate clamp of 7, an up
clamp of [-7, 7] and an up offset of 1. The per-token FlashInfer path fuses
the FP32 first GEMM accumulations with the activation and emits BF16 before
the explicit second input quantizer. The second GEMM emits BF16 before the
weighted expert reduction. In particular, it does not insert separate BF16
gate/up roundings. The earlier activation prescreen did not certify these
boundaries.
[MiniMax quantization adapter](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/layers/quantization/minimax_m3_nvfp4.py).
[FlashInfer per-token workspace and output types](https://github.com/flashinfer-ai/flashinfer/blob/d84335d8c54e0c8f9f7e22faa15c7e34a58abf94/csrc/trtllm_fused_moe_kernel_launcher.cu).

## Q8KV4 attention

After normalization and rotary embedding, attention queries and index
queries are converted from BF16 to E4M3 with unit outer scale. Main K/V and
index K are stored as NVFP4: a 128-value head has 64 payload bytes and eight
E4M3 scales. Each 16-value group uses `E4M3(amax / 6)`; the cache's outer
scale is fixed at one, unlike the expert activation row scale.
[Attention dispatch](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/layers/attention/minimax_sparse_nvfp4_backend.py),
[cache storage](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/mem_cache/minimax_m3_nvfp4_pool.py).

When consumed, E2M1 payloads are multiplied by E4M3 scales and **rounded to
E4M3 again**, including saturation. Index scoring uses these reconstructed
FP8 keys. Attention also quantizes unnormalized softmax weights to E4M3,
after multiplying them by 448. The softmax denominator retains the FP32
weights. Omitting either rounding implements a different formula.
[MSA conversion reference](https://github.com/MiniMax-AI/MSA/blob/c8e1013a16a708d2a7656805d5fc26f2b87715d1/tests/inference/dequant/nvfp4_to_fp8/reference.py).

The upstream phase formulas differ:

| Phase | Softmax/PV accumulation |
| --- | --- |
| Prefill | Normalize each selected 128-token page independently; quantize its weights; round each page output to BF16; combine pages using FP32 log-sum-exp weights and round the result to BF16. |
| Decode | Walk selected pages with an online maximum, an FP32 denominator and an FP32 output accumulator; quantize each page's weights relative to the running maximum; round the final output to BF16. |

These are not bitwise interchangeable: the probability quantization and
intermediate BF16 boundaries can change values. DGPP's batch/chunk
invariance requirement therefore needs explicit validation of the selected
execution formula. Matching the format names alone does not establish it.
[Prefill oracle](https://github.com/MiniMax-AI/MSA/blob/c8e1013a16a708d2a7656805d5fc26f2b87715d1/tests/inference/msa_v1/attention/prefill/q8kv4/reference.py),
[decode oracle](https://github.com/MiniMax-AI/MSA/blob/c8e1013a16a708d2a7656805d5fc26f2b87715d1/tests/inference/msa_v1/attention/decode/reference.py).

`tools/minimax_native_attention_reference.py` represents both formulas with
contiguous tensors and PyTorch FP32 arithmetic, independently of DGPP's
paged buffers and CUDA kernels. It is a mathematical reference, not an
upstream GPU execution or a performance measurement. Full-model validation
must record the chosen phase formula, projection reductions, TP boundaries
and quantizer implementation. Existing numerical limits remain unchanged;
BF16-vs-native task quality is a separate comparison.

Run the CPU cross-check against an MSA source checkout with:

```sh
python tools/minimax_native_attention_check.py --msa-source /path/to/MSA
```

The checker verifies the upstream file hashes before loading the reference
functions. It covers all 16 E2M1 codes crossed with all 127 nonnegative
finite E4M3 scales, signed-zero packing, zero groups, causal page boundaries,
more than sixteen candidate pages, permuted physical pages and query chunk
invariance within each formula. At 1, 129, 270 and 2,057 keys, both formulas
match their upstream CPU references exactly. They deliberately produce
different outputs from each other. The GPU checks below cover both phases; full-model results are recorded
separately from these isolated checks.

## Initial GB10 projection kernel validation

The new `mxfp8_gemm` kernel uses SM121's block-scaled E4M3 tensor-core
instruction with 32-value E8M0 groups. Its quantizer matches the actual
pinned SGLang Triton producer byte for byte on all 65,280 finite BF16 bit
patterns and another 18,528 inputs around scale-exponent boundaries, with
padded row strides and a partial final quantizer tile. Both payloads and
scales are compared; no tolerance is used for this check.

Four matrix cases cover 1 through 65 rows, incomplete output tiles and
reduction lengths through 6,144. They pass independent FP64 forward-error
bounds; the largest relative RMS error is `3.09e-6`. BF16 outputs are exactly
the rounding of FP32 outputs. Splitting the same queries into 1-, 8- and
17-row calls preserves every FP32 output bit. The warnings-as-errors build,
CUDA memcheck and racecheck pass, with zero errors or hazards.

These are isolated kernel checks. The kernels are connected to the target
model's serving path; full-model fidelity is recorded below. Reproduce the upstream quantizer check
with `tools/minimax_mxfp8_quantization_check.py`, providing the built
`mxfp8_gemm_test`, the pinned SGLang source and an untracked output directory.

## Initial GB10 Q8KV4 kernel validation

The initial native attention kernels implemented NVFP4 cache append, FP8
query conversion, sparse index scoring and the upstream **prefill** formula
at every query geometry. The subsequent full-model phase comparison below
found that applying this formula during decode fails the numerical gate.
The corrected path selects the published formula by execution phase:
prefill for prompt rows, online decode for generated and verification rows.
Batch size does not select the formula.

Five cases cover 1 through 65 query rows, mixed requests, permuted physical
pages, 1 through 4 KV heads, causal boundaries, more than sixteen candidate
pages, exact index ties and padded rows with invalid request IDs. Query and
cache bytes match the independent precision reference exactly. Selected
pages match, index scores satisfy FP64 forward-error bounds, and attention
relative RMS error is at most `1.74e-5`, within the existing `0.004` gate.
Splits into 1-, 8- and 17-row calls and two PDL CUDA-graph replays preserve
the output bits. CUDA memcheck and racecheck report zero errors or hazards.

Run `ctest --test-dir build-release -R '^minimax_native_attention_test$'
--output-on-failure` after building that target. The test generates temporary
fixtures and checks CUDA output against the independent CPU formula. These
tests certify the isolated path, not full-model fidelity or throughput.

## Initial native expert stage validation

The native expert quantizer matches the actual hash-pinned FlashInfer 0.7.2
CuTe GPU producer byte for byte: E2M1 payloads, E4M3 block scales and FP32
per-token globals. Cases include all 65,280 finite BF16 patterns, signed
zero rows, padded input strides, outliers and row widths through 32,768.
The exhaustive tiny inputs also reproduce upstream's approximate reciprocal
with subnormal flushing; they are not evaluated with a normal-range
nearest-format rounding bound.

Four expert chains cover 1 through 65 rows and a 6,144-wide hidden input.
Their first matrix products satisfy independent FP64 forward-error bounds;
SwiGLU has the required FP32-to-BF16 boundary; and the second product meets
its FP64 accumulation plus BF16-rounding bound. Splitting into 1-, 8- and
17-row calls preserves the stage outputs exactly. The new producers pass
PDL graph replay, CUDA memcheck and racecheck, with zero errors or hazards.

Run `minimax_native_moe_test` through CTest for the independent stage checks.
For the additional upstream GPU comparison, run
`tools/minimax_native_moe_regression.py --app <test-binary> --upstream` in an
environment containing the pinned FlashInfer source and its CuTe dependency.
The checker verifies the quantizer source hashes before execution. This is
not a full upstream MoE or full-model comparison.

## Target-model integration

`engine.activations: "native"` selects MXFP8 dense/shared projections and
dynamic NVFP4 routed experts for every target-model row, including decode.
`engine.kv_dtype: "nvfp4"` independently selects the Q8KV4 attention path.
Both settings default to `bf16`; the shipped deployment explicitly selects
`native` activations and `nvfp4` cache after native fidelity validation. The
vision encoder retains its existing precision. The optional DSpark drafter
also follows the activation setting: native MXFP8 inputs feed its context
K/V, QKV, output and MLP projections. Its BF16 feature projection, vocabulary
head, Markov matrices, confidence head and attention/cache retain their
checkpoint precision. The independent draft fixture checks the native and
BF16 projection recipes separately.

The cache pool exposes format-specific planes, and its memory planner,
partial-block prefix copies and NVMe byte counts use the actual storage
format. Activation precision travels in the fabric settings and contributes
to the effective configuration digest, preventing reuse across recipes.

Synthetic model checks cover native activations with native cache and each
mixed BF16 control. Repeated forwards, layer capture, incremental eight-row
verification and partial-block prefix restore preserve their expected
outputs exactly. Full-model CUDA graph replay also preserves hidden states
across changing eight-token feeds and cache-page boundaries in all four
precision combinations. The native integration passes CUDA memcheck and
racecheck with zero errors or hazards. A separate layer-by-layer reference executes the pinned
SGLang and FlashInfer GPU quantizers, decodes their operands independently,
and computes matrix products with FP32 accumulation. Its four-layer fixture
comparison passes the existing 1% sublayer gate (maximum relative RMS
`0.000714405`), with exact router selections. These checks are distinct from
full-trajectory likelihood, task quality and a complete upstream runtime.

The initial unrestricted FP32-product reference does **not** pass the full
synthetic trajectory gate: over 384 scored positions, absolute mean NLL
difference is `0.00100` nat, but 105 positions (27.34%) differ by more than
1 nat and 97 argmax choices differ. Small first-layer differences grow
through the fixture's attention and routed experts. This comparison uses
native quantization on both sides; it is not the earlier BF16-relative
prescreen. The arithmetic controls below explain this mismatch; isolated
stage passes alone do not establish end-to-end native accuracy.

The first real-checkpoint diagnostic at TP4 checks all 60 layers on five
tokens. Its stage comparison passes (maximum relative RMS `0.00142110`),
with exact router selections and the same binary hash and zero swap on all
four ranks. The four-position full trajectory fails the numerical gate
(absolute mean NLL difference `0.38348` nat; one position exceeds 1 nat).
These initial trajectory probes used the earlier dense sigmoid expression.
They remain recorded as failures; the corrected producer and phase-aware
comparisons below supply the subsequent evidence. Neither short diagnostic
establishes task quality.

Run the diagnostic with:

```sh
python tools/minimax_native_reference.py \
  --checkpoint-dir /path/to/checkpoint --sglang-source /path/to/sglang \
  --check-engine-dir /path/to/engine-trace --world 4 --output /path/to/results
```

The environment must contain the pinned FlashInfer quantizer and its CuTe
dependencies. Source hashes are checked before execution. `--manifest`
instead of `--check-engine-dir` runs full trajectories; `--attention-phase`
selects the prefill or decode formula explicitly. Reference output records
these choices and does not claim to execute the full upstream serving stack.

## Independent native arithmetic controls

The unrestricted decoded-operand reference is retained. Two additional
controls separate matrix and attention accumulation order from quantization:

- `tools/minimax_native_reference_gemm.py` uses Triton `dot_scaled` on the
  published byte payloads and scales. It imports no DGPP kernel or layout.
  MXFP8 and NVFP4 products satisfy FP64 forward-error bounds at reduction
  widths through 6,144; the largest relative RMS error is `2.74e-6`.
  Splitting into 1-, 8- and 17-row calls preserves every output bit.
- `tools/minimax_native_reference_attention.py` independently expresses
  FP8 index/product operations, the per-page BF16 boundary and ordered FP32
  softmax/page combination. Four geometries through 2,057 keys agree with
  the unrestricted mathematical oracle within `4.35e-5` relative RMS
  (existing gate `0.004`), with exact selections and row-geometry invariance.

Each file runs its own validation when executed directly. They are
arithmetic-order controls, not the upstream serving kernels. Select them
explicitly in the full reference with:

```sh
--projection-backend triton-native --elementwise-backend triton-fixed \
--attention-arithmetic triton-fixed
```

The attention control also supports the upstream online decode formula.
Its FP8 dot order and FP32 reductions remain fixed while it removes the
prefill page-output BF16 boundary and uses decode's running maximum and
accumulator. Both formulas pass the independent mathematical oracle's
`0.004` RMS gate and preserve results across 1-, 8- and 17-row splits.

For a full-model phase comparison, add `--attention-phase upstream` to a
manifest run. The manifest's `score_from` is the prompt length: prompt
rows use prefill, including the last prompt row that predicts the first
scored token, and subsequent rows use decode. `prefill` and `decode`
explicitly apply one formula throughout. This comparison isolates the
phase formulas; it still does not execute or certify the upstream GPU
serving stack.

With all three controls, the corrected native engine matches all 20
synthetic sublayer/residual traces bit for bit over 384 positions. Its
likelihood comparison has zero argmax changes and zero changes over 1 nat.
The five-token real-checkpoint TP4 diagnostic likewise matches all 300
sublayer/residual traces over its four scored positions, with zero argmax
changes; all four engine ranks have matching layer hashes. Using only the
matrix/elementwise controls left attention differences that grew through
the trajectory. These controls explain those short-probe discrepancies;
they do not turn the unrestricted-reference failures into passes or certify
full upstream runtime equivalence. Those short probes did not establish
broader corpus fidelity, task quality or the upstream decode formula;
the subsequent comparisons below address the corpus and phase checks.

The next real-checkpoint TP4 pilot covers complete raw-text and chat prompts:
528 and 529 scored positions respectively. Both pass the unchanged
likelihood gate against the three-control reference, with zero argmax
changes and zero changes above 1 nat. Absolute mean NLL deltas are
`4.42e-8` and `5.57e-7` nat; the largest per-position difference is
`3.70e-5` nat. Input protocols match, both engine trials repeat exactly,
and all four ranks agree on every recorded hidden-state hash. The binary
and zero-swap checks pass on every rank. These pilot results extend the
short-trace evidence; they do not replace the longer corpora or task-quality
checks, and do not claim to run the complete upstream stack.

The prompt-aware phase comparison fails the unchanged numerical gate on
both pilots. Applying prefill throughout, compared with prefill for the
prompt and decode for the continuation, raises mean NLL by `0.20743` nat
on raw text and `0.01922` nat on chat. Changes exceeding 1 nat occur at
128/528 (24.24%) and 60/529 (11.34%) positions; argmax changes number 129
and 66. The comparison holds native producers, matrix products and
elementwise arithmetic fixed. This is a material phase-formula mismatch,
not a BF16-versus-native comparison or a task-quality score. Earlier
same-formula passes do not certify the former serving path.

Generated-state prefix reuse follows upstream's cache policy. The pinned
SGLang radix cache retains output KV as well as prompt KV, splitting them
for eviction priority. DGPP likewise retains prompt, rolling and completed
conversation snapshots. An earlier provisional restriction suppressed
native generated-state snapshots to force cold/warm equivalence; that
restriction was removed because it imposed an extra constraint on the
publisher's phase-dependent recipe without evidence of a task-quality
benefit. Restore fidelity is checked against the live conversation that
created the state. Recomputing generated history as prefill can change its
values; this is reported separately from corruption or nondeterministic
restoration. See the existing
[chunk-shape policy](numerics.md#greedy-transcripts-and-chunk-shape) and
[upstream radix checkpointing](https://github.com/sgl-project/sglang/blob/58484246d26523eaf1a7f16ab524deee288d2120/python/sglang/srt/mem_cache/radix_cache.py).

The corrected kernels pass both phase oracles at all five isolated
geometries, exact row splits and PDL graph replays, CUDA memcheck with zero
errors and racecheck with zero hazards. Seven targeted CTest entries pass,
including the scheduler and all four target precision combinations.
Generated-prefix checks additionally compare restored prompt extensions
and subsequent decode with the same live conversation.

The corrected real-checkpoint TP4 pilot passes against the prompt-aware
native reference on all 1,057 raw/chat scored positions. There are zero
argmax changes and zero changes above 1 nat; absolute mean NLL deltas are
`3.41e-8` and `3.98e-8` nat, with a maximum per-position difference below
`3.70e-5` nat. Both trials repeat exactly, input protocols match, all four
ranks agree on hidden-state hashes, and every rank has the expected binary
and zero swap. This resolves the pilot's phase-formula mismatch. It remains
an independent fixed-arithmetic reference comparison, not a full upstream
serving execution or the broader native quality verdict.

The completed phase-aware TP4 comparison covers all six raw/chat corpora:
**28,329 scored positions**, zero argmax changes and zero changes above
1 nat. The largest absolute mean-NLL difference is `1.49e-7` nat; the
largest per-position difference is `1.20e-4` nat. The existing limits remain
`0.02` nat mean NLL and a 1% rate above 1 nat. Both engine passes are exact,
all ranks agree on hidden hashes, input protocols match, and every rank
runs the recorded binary with zero swap. Arithmetic source is `1c625c87`;
`4d2e4574` restores the standard generated-prefix policy without changing
those kernels. Its seven targeted tests pass, including a mixed
prefill/decode prefix restored before a new prompt extension and decode.
These results establish native fidelity against the independent control;
the serving campaign supplies task-quality and throughput evidence.

## Cache reconstruction and decode scheduling

Cache consumers reconstruct four FP8 values together. E2M1 multiplied by a
finite nonnegative E4M3 scale fits exactly in FP16, so paired FP16 products
preserve the required FP8 bytes, including saturation and signed zero. The
GPU conversion test exhausts all 256 packed code pairs crossed with all
127 finite nonnegative scales, exercises both word halves and all cache
groups, and compares with independent scalar FP32 reconstruction.

For at most sixteen query rows, the sparse indexer gives each row its own
CTA. This avoids serializing independent decode requests inside a shared
tile. Larger prefill calls retain sixteen-row tiles. Every score uses the
same four K32 FP8 products, and empty page ranges exit before staging data.
The existing mixed-request, selected-page, row-split and PDL graph tests
pass; the packed conversion and attention kernels pass CUDA memcheck, and
attention racecheck reports no hazards. Full-model native/cache smokes also
pass. Serving speedups require separate measurement.

## MXFP8 projection scheduling

Short batches use eight- or sixteen-column tiles with K512 or K256 staging,
respectively. Shared-down projections with short K and the 2,048-output
query projection select sixteen columns; other measured target projections
select eight. Larger batches retain the original 64-row/64-column tile.
Staging changes do not change the ordered K32 products, and incomplete
stages stop at the actual reduction width.

`mxfp8_gemm_bench` measures seven real projection shapes at 1, 8 and 16
rows. It rotates at least 64 MiB of weights, uses CUDA graphs and event
timing, rotates the order of four candidates over five trials, and requires
identical FP32 and BF16 outputs. Its measurements are isolated kernel
diagnostics. The regression test additionally checks padded outputs, tail
columns, short reductions, FP64 error bounds and 1/8/17-row splits across
all four column tiles. Full-model graph smokes pass in all four precision
combinations; CUDA memcheck and racecheck pass with no errors or hazards.

## Parallel native decode pages

A C1 diagnostic at roughly 7K context found the serial online attention
kernel used 31.7 ms of a 77.9 ms decode step, or 42% of GPU time. The
replacement computes QK and PV for selected pages in parallel, deriving
each page's running maximum from the page maxima. A final ordered FP32
fold preserves the original corrections, denominator and accumulator.
It reuses existing prefill scratch and adds no cache storage.

All five attention fixtures match the serial kernel bit for bit, including
row splits, padded rows, physical-page permutations and PDL graph replay.
The independent phase oracle and all four model precision combinations
pass. CUDA memcheck and racecheck report zero errors or hazards.

On the same 7,435-token prompt, the replacement reduces the diagnostic
decode step from 77.926 to 48.134 ms (38.2%) and attention from 31.731 to
2.303 ms. All 256 output tokens, text and finish reason match the serial
baseline exactly. The comparison uses sources `4d2e4574` and `3062ce1f`,
with the expected executable and zero swap on every rank and matching
operation streams. These profile timings explain the optimization; the
separate shipped-template campaign provides deployment throughput rows.

## DSpark early-position acceptance audit

After the anchor-row correction, positions two and three still warranted
direct prediction checks. Correct final target output cannot validate a
drafter: rejection can hide a broken proposal path.

`minimax_forward_check --draft-manifest <file> --draft-rounds 32 --dump-dir <dir>`
captures eager draft hidden states, vocabulary logits, K/V planes, proposals,
target verification tokens, and the target features committed between blocks.
The manifest is an array of objects with `name` and integer `ids` fields.
Use the normal checkpoint/residency/TP arguments and select
`--activations native --kv-dtype nvfp4` for the native recipe. Place the
manifest at the same path on all ranks and gather each rank's dump directory
before running the independent comparison:

```sh
.venv/bin/python tools/minimax_dspark_audit.py \
  --checkpoint-dir "$MODEL_DIR" --engine-dir "$AUDIT_DIR" \
  --sglang-source "$SGLANG_DIR" --world 4 --output "$REPORT_FILE"
```

This tool uses the pinned upstream MXFP8 producers and independent PyTorch
products. It checks target feature fusion, initial context K/V, committed
context positions and K/V, the five-layer draft block, the vocabulary head,
and the chained Markov selector. `--sampler-only` isolates selection on
captured hidden states. `--projection-backend triton-native
--elementwise-backend triton-fixed --blocks 2` provides an additional
arithmetic control. These reports are diagnostics, not new accuracy gates.

The 2026-11-05 audit used source
`50cbe346c92f806cae5f428061f6caa8a0170de6`, executable SHA256
`9104037abd34555da3b85312551476f5c6ac645e8feeb1148a3460181e44a8c4`,
checkpoint `2f33a8b9ae49ce7b51c994df97166a7fd059bc2e`, and native TP4
inference. Every rank had the expected executable and zero process swap.
Production was stopped during capture and reference execution. A repeat
capture reproduced all 9,760 original snapshot files byte for byte, and
2,040 replicated-state comparisons agreed across ranks. The diagnostic
probe also passed its native fixture smoke and CUDA memcheck (zero errors).

Each class used its first `serve_load.py` prompt, the published chat
template, and 32 full-seven greedy draft blocks. The table counts acceptance
**conditional on every earlier proposal being accepted**. This shorter
capture is not interchangeable with the earlier 256-token serving sample.

| Prompt class (generated tokens are thinking) | Position 2 | Position 3 |
| --- | ---: | ---: |
| Prose | 14/20 (70.0%) | 6/14 (42.9%) |
| Code | 20/25 (80.0%) | 11/20 (55.0%) |
| JSON | 19/31 (61.3%) | 12/19 (63.2%) |
| Math | 23/30 (76.7%) | 20/23 (87.0%) |
| Chat | 12/20 (60.0%) | 8/12 (66.7%) |

The independent FP32 head/Markov calculation reproduced all 1,120 proposals;
head relative RMS error was at most 5.11e-7. Initial feature-fusion error was
at most 1.26e-5. Initial context K/V error was below 0.00170, and the 155
transitions between captured blocks had correctly placed committed rows
with K/V error below 0.000631.

Upstream rounds base vocabulary logits to BF16. Its fused greedy Markov
path accumulates the bias and addition in FP32; its generic path rounds the
bias and sum to BF16 too. Applying either rounding path to the captured
states repaired none of the 38 position-two failures. The fused-style path
repaired one of 31 position-three failures; the generic path repaired none.
Recomputing the backbone independently, then applying fused-style sampling,
still rejected all 38 position-two failures and 30 of 31 position-three
failures. Consequently, these checks do not support rounding as the main
cause of the low early acceptance.

This is not full upstream serving parity or proof of the checkpoint's
maximum achievable acceptance. The independent backbone's hidden-state
relative RMS differences span 0.0162–0.0855, and both backbone calculations
consume the captured context cache, whose construction is checked separately.
Target labels are valid only along the captured proposal prefix: after a
changed proposal, later labels cannot be used to claim counterfactual
acceptance. No runtime precision change or accuracy-limit relaxation follows
from this audit, and it publishes no throughput row.

### Actual upstream draft-model comparison

The follow-up uses the actual SGLang `DSparkMiniMaxDraftModel.forward`, feature
projection, native linear methods, Gemma norms, RoPE, shared vocabulary head,
and `MarkovGreedyStep` from revision
`58484246d26523eaf1a7f16ab524deee288d2120`. It rebuilds the draft context from
the raw five target-layer taps for both prefill and generated tokens. It does
not reuse DGPP's feature projection or context K/V in the default mode.

On SM121, the harness selects FlashInfer's B12x MXFP8 GEMM backend and uses
noncausal FA2 attention over contiguous K/V. Those are hardware/allocation
adapters; the model forward and sampler are imported from the pinned source.
This is a one-GPU draft replay against DGPP's fixed target trajectory, not an
end-to-end distributed SGLang serving run or a throughput measurement. The
offline attention adapter rejects contexts that exceed the sliding window.

The new capture used source `3ddb36fd45d2fc79f96ab3ae16817f8baf5beda6` and
executable SHA256
`25c5b9d1b4e9eeb137c97c2558e233a5df05fd013b1cb150d704523f58302e71`,
with the same checkpoint, native TP4 recipe, five prompts and 32 blocks per
prompt. All four ranks had the expected binary and zero process swap. The
2,560 pre-existing rank-zero block files reproduced byte for byte in the new
capture. The final reusable tool reproduced all 160 upstream proposal blocks
from its initial run. The environment used Torch `2.14.1+cu130`, FlashInfer
`0.7.2`, and `sglang-kernel` `0.4.10`.

Acceptance is scored against the **committed greedy continuation**, not
verification logits conditioned on a rejected proposal. This permits a valid
comparison after either drafter changes its proposal. Unobserved continuation
tails are excluded at each position. Counts below are conditional on all
earlier proposals being accepted:

| Prompt class | DGPP position 2 | Upstream position 2 | DGPP position 3 | Upstream position 3 |
| --- | ---: | ---: | ---: | ---: |
| Prose | 14/20 | 14/20 | 6/14 | 7/14 |
| Code | 20/25 | 20/25 | 11/20 | 11/20 |
| JSON | 19/31 | 19/31 | 12/19 | 12/19 |
| Math | 23/30 | 23/30 | 20/23 | 20/23 |
| Chat | 12/20 | 12/20 | 8/12 | 8/12 |
| Total | 88/126 (69.8%) | 88/126 (69.8%) | 57/88 (64.8%) | 58/88 (65.9%) |

Across positions 1–7, the respective accepted-token counts are
`126, 88, 57, 41, 35, 28, 25` for DGPP and
`126, 88, 58, 41, 36, 29, 26` for upstream. An additional arithmetic control
partitions O/down GEMMs four ways, uses upstream native producers and
FlashInfer GEMMs for each partition, then folds the BF16 local outputs in
FP32. It preserves the same position-two and position-three accepted counts
as the one-GPU upstream run; its position-one count is 124 rather than 126.
This control is not presented as actual four-rank SGLang execution.

**Every scored token is inside `<mm:think>`; none is answer-phase code or
JSON.** The class describes the user's prompt, not the generated token type.
For example, both drafters predict ` carefully` after `Let me design this` in
the code prompt, whereas the greedy target chooses ` well`. In the math
prompt both predict `=` where the target chooses ` =`. These remain legitimate
greedy rejections. They do not establish code/JSON answer-phase acceptance.
The new report separates thinking, answer and unmarked positions so that a
longer continuation can measure those phases without conflating them.

The operation-level diagnostic in `tools/minimax_dspark_stages.py` compares
identical inputs with the independent native arithmetic reference. Across ten
blocks, its projections, norms, RoPE and TP folds agree exactly; attention
differs in 1,083 of 2,867,200 BF16 elements, with relative RMS at most 8.47e-5.
Small changes can cross later MXFP8 rounding boundaries. This trace localizes
the first difference against that arithmetic reference; it is not a claim of
bitwise SGLang parity. Actual upstream final hidden-state relative RMS is
0.0488–0.0877, despite the similar acceptance counts.

This evidence does not show a large remaining drafter-specific acceptance
loss on these prefixes. It also does not establish a universal acceptance
ceiling, answer-phase rates, or upstream target-model parity. No serving
precision change, accuracy-limit relaxation, or throughput claim follows.

To reproduce, install the pinned SGLang runtime dependencies in a separate
environment with the versions above, expose its complete checkout, and use
an ignored output directory for the JIT caches and reports:

```sh
TRITON_CACHE_DIR="$RUN_DIR/triton" \
FLASHINFER_WORKSPACE_BASE="$RUN_DIR/flashinfer" \
python tools/minimax_dspark_upstream.py \
  --checkpoint-dir "$MODEL_DIR" --sglang-source "$SGLANG_DIR" \
  --engine-dir "$AUDIT_DIR" --capture-receipt "$CAPTURE_RECEIPT" \
  --output "$REPORT_FILE"
```

The capture receipt records `source_revision`, `binary_sha256`, `command`
(including `--dump-dir`), `complete`, `returncode`, and per-rank `first`/`last`
samples with `sha256` and `swap_kib`. The tool verifies these, hashes the
complete upstream Python tree, validates anchors and original acceptance
against the continuation, and writes library versions and phase-specific
counts. `--context captured` isolates the backbone with DGPP K/V;
`--emulate-tp 4` enables the additional row-parallel arithmetic control.

Validation includes the eight DSpark CTest cases, the native three-round
capture fixture, CPU accounting tests for changed proposals and truncated
tails, and the full real-checkpoint upstream replays above. Production is
restored to its original frozen executable after the hardware window.
