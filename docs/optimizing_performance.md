# Optimizing performance

How to make a dgpp deployment faster on your own workload without losing
track of what it computes. The [README's configuration
tables](../README.md#configuration) define every key; this page says which
keys move which number, what each one trades, and how to measure a change
before trusting it. The numbers quoted are the recorded ones in
[benchmarks](benchmarks.md); yours will differ with prompts, context and
nodes, which is why the protocol matters more than the quotes.

## 1. Measure first

Four numbers describe a deployment, and they move independently:

| number | what it is | how to get it |
|---|---|---|
| single-stream decode | ms per pass and tokens per pass for one request; engine tokens/s is their ratio | `scripts/timed_load.py HOST PORT --concurrency 1 --classes all --repeat 3 --max-tokens 256` (greedy; medians of three) |
| loaded throughput | wall tokens/s with 2 or 4 requests live | the same with `--concurrency 1,2,4` |
| cold prefill | seconds to the first token for a 2K / 8K / 32K prompt the cache has not seen | `scripts/serve_prefill_probe.py HOST PORT 2048 8192 32768 --repeat 3 --seed 7 --tag NAME` (a new `--tag` per run, or the prefix cache answers instead) |
| quality | HumanEval, GSM8K and schema extraction through the endpoint | `scripts/serve_eval.py HOST PORT --out DIR --no-think --max-tokens 2048 --limit 300 --seed 20260908 --allow-code-execution` |

Rules that keep the numbers comparable:

- `export CUDA_DEVICE_MAX_CONNECTIONS=32` before the server boots (the
  templates assume it), and `python3 scripts/serve_bench.py HOST PORT 16
  warm` before timing, so the first requests do not pay the graphs' warm-up.
- One world on the fabric at a time. Stop the production deployment for
  the measurement (`scripts/dgpp-cluster down --config ...`) and restore it
  after; a second world sharing the GPUs or the RoCE links perturbs both.
- Compare on the same binary and the same prompts, A then B then A: the
  first leg after a boot is colder (its prefill reads the checkpoint's pages
  and the n-gram table from disk), so a closing baseline leg tells noise
  from effect. Read decode as ms per pass, not tokens/s, when comparing
  two builds; tokens/s folds in acceptance, which the prompts set.
- Greedy transcripts are the correctness check. A change that keeps the
  chain bitwise leaves a long prompt's greedy completion byte-identical
  (`sha256` of the text); a change that alters arithmetic may not, and then
  the evals say what that costs.
- Isolated kernel benchmarks predict the served step poorly on a Spark:
  the step's weight streams share one 273 GB/s memory pool with the L2
  prefetcher and the other kernels. A microbenchmark says whether a kernel
  is worth trying; only the served A/B decides.

## 2. The levers, by what they move

### Memory and context

| key | moves | trades |
|---|---|---|
| `engine.kv_capacity` | the context and request count the KV pool can hold | memory; the startup memory plan (`dgpp-serve --memory-plan`) says what fits before anything is allocated |
| `engine.kv_dtype` (GLM-5.3 family) | KV bytes per token: `bf16`, `fp8`, `fp4` | the quantized forms change the model's answers; `bf16` is the exact one |
| `engine.bf16_weights` | the resident form of the BF16 matrices decode streams: `checkpoint`, `bf12`, `bf12+bf16` | `bf12` is lossless (the kernels rebuild the exact bits) and both faster and smaller for decode; `bf12` alone slows a prefill chunk slightly, `bf12+bf16` keeps prefill on the BF16 bytes for 0.75 of those matrices' memory more |
| `engine.dense_weights` (Qwen) | the dense stack's storage: the checkpoint's BF16 or block-FP8 encoded at load | FP8 saves memory and decode bytes; it is a quantization of the dense projections, measured per template in the quality rows |
| `engine.ngram_table` (Qwen) | the n-gram table resident or mmap'ed from local NVMe | mmap saves its memory; the rows a chunk needs come through the page cache, gathered a chunk ahead |
| `engine.prefix_cache_gib`, `engine.prefix_min_tokens` | how much finished-request state is kept for exact prefix reuse | memory against repeated-prefix prefill; the reuse is exact (the cold-prefill result) |

### Decode latency

| key | moves | trades |
|---|---|---|
| `engine.decode_graph` | replays each decode step as a CUDA graph | launch overhead; keep it on |
| `engine.mtp`, `engine.mtp_depth` | draft tokens per pass | a deeper draft accepts more tokens per pass but every pass costs more rows; the template depth is the measured best for its model (depth 1 for GLM-5.3-Flash, 3 for the Qwen AutoRound hybrid); the [decode modes](benchmarks.md#decode-modes) table shows plain, default and deeper for every template |
| `engine.lookup_draft` (+ thresholds) | accepted tokens per pass on repeating work | a prompt-history match replaces MTP drafts within the same rows (strong matches on their own, shorter ones only when they agree with the device's picks); free on prose (no match, no cost), decisive while the answer quotes the prompt; greedy slots only |
| `engine.mtp_schedule*` (DeepSeek, the Qwen3.8-27B drafter) | confidence-scheduled verify depth | the cost model's two milliseconds are deployment measurements |
| `engine.fp8_head` (Qwen FP8 head) | `mma` streams the vocabulary head through the tensor cores at every width (one chain alone, at any verify depth and in a batch) | validated numerically per template; `gemv` restores the previous path |
| `engine.graph_batch_min_live` | when decode switches from scalar to batched graphs | earlier batching helps throughput under load and can cost a single stream |
| `engine.sampling_candidates` | the candidates gathered per rank on the sampled path | fewer means less routine work and more full-gather fallbacks |

### Prefill and time to first token

| key | moves | trades |
|---|---|---|
| `engine.prefill_budget_tokens`, `engine.prefill_idle_budget_tokens` (Qwen) | how many prompt rows a scheduler tick runs before yielding to live decodes | smaller budgets keep decodes moving under a long prompt; larger ones finish the prompt sooner; the idle budget applies when nothing else is live |
| `engine.admission`, `engine.admission_window` | grouping queued cold prompts into one forward pass | throughput against per-request latency; group admission is bitwise the prompts alone |
| `engine.prefill` (DeepSeek) | `bounded` (the model's own recipe) or `exact` | about half the prefill work against parity with the full walk |
| `engine.prefill_bf16_partials` (Qwen AutoRound hybrid) | the packed expert chain's down projection in bf16, partials summed from bf16 | −3 to −7 % cold prefill (2K to 32K); not bitwise the fp32 chain |
| `engine.prefill_fp8_gemm` (Qwen with `dense_weights` fp8) | the dense stack's prefill GEMMs on the fp8 tensor cores from per-token e4m3 activations | 0 to −3 %; not bitwise the dequantized chain |
| `engine.prefill_fold_scales` (Qwen AutoRound hybrid) | the expert GEMM's group scales folded into the bf16 weight values | measured no gain (+3 % at 32K); kept for completeness, leave it off |

The three `prefill_*` keys are the accuracy trades described in the
README's [accuracy and correctness](../README.md#accuracy-and-correctness)
section: off unless a deployment turns them on. The AutoRound hybrid's
template turns on the first two, whose evals stayed within the default
chain's run-to-run band; the fold is off everywhere.

### Throughput under load

| key | moves | trades |
|---|---|---|
| `engine.max_concurrency` | request slots (each speculative request holds `1 + mtp_depth` decode rows; the family's row cap bounds it) | more slots share the pass: loaded throughput rises while each stream slows; the memory plan sizes the KV pool for them |
| `engine.queue_limit`, `engine.max_connections` | how much waits behind the slots | back-pressure against acceptance |
| `engine.bulk_pace_gbps`, `engine.bulk_inflight` (multi-node) | the prefill's bulk transfers over RoCE | the fabric drops packets under incast without pacing; the derived default is the measured safe rate |

## 3. A worked example: the Qwen3.8 AutoRound hybrid on one Spark

The [template](../deploy/cluster_qwen-3.8-flash-next_autoround-int4_w1.example.json)
is the sum of measured choices: MTP depth 3 (52–54 ms a pass at 2.4–3.8
committed tokens; depth 4 measured slower per token), `bf16_weights:
"bf12+bf16"` (lossless, faster decode, prefill on the BF16 bytes),
`dense_weights: "fp8"` (the dense stack encoded at load; the quality rows
carry it), the n-gram table mmap'ed with the next chunk's rows staged
ahead, a 4,096-token prefill budget so decodes keep moving under a long
prompt, and the two prefill levers that paid without a measurable quality
cost. Its rows in [benchmarks](benchmarks.md) show the default chain and
the levers-on configuration separately; the [dated
record](../benchmarks/results/2026-09-28-qwen-autoround-int4/README.md)
holds every A/B behind those choices, including the ones that lost.

## 4. Where the floor is

A decode pass is bound by the bytes it streams (the resident weights a
token touches plus the KV it reads) over the Spark's memory bandwidth, plus
a launch floor per pass; a prefill chunk by its dense and expert GEMMs at
the tensor cores' rate. The per-model plans (`docs/*_plan.md`) state each
floor next to the measured step, and the gap between them is the work
list. When a change makes a number worse than that gap allows, suspect the
measurement before the code: a stale binary, an overlapping build, a
second world on the fabric, or a cold first leg.
