# Optimizing performance

How to make a dgpp deployment faster on your own workload without losing
track of what it computes. The [configuration
reference](configuration.md) defines each setting; this page says which
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
| `engine.kv_capacity` | the shared token capacity of the KV pool | memory; the startup memory plan (`dgpp-serve --memory-plan`) says what fits before anything is allocated |
| `engine.kv_dtype` (GLM-5.3 family, MiMo, Qwen3.5 releases) | KV bytes per token: `bf16`, `fp8`, and GLM-only `fp4` | the quantized forms change the model's answers; `bf16` is the exact one. On the Qwen3.8-27B NVFP4 release `fp8` is the release's own `kv_cache_scheme` (its shipped per-layer scales), halves the Full layers' pool and cuts long-context decode (262K: 188 -> 156 ms a pass); with the attention math kept in bf16 the cold prefill is slower (316 -> 352 s at 262K: the codes widened in shared memory) — see `engine.prefill_attention`. On the FP8 release (no shipped scales: unit scales, 2026-10-18) the same shape: decode at 256K 216 -> 181 ms a pass, cold prefill 395 -> 434 s under bf16 arithmetic and 325 with the fp8 attention, the suite at the bf16 floor |
| `engine.prefill_attention` (Qwen3.5 releases on `kv_dtype: "fp8"`) | the prefill attention's arithmetic over the fp8 K/V pool: `bf16` (the widened codes) or `fp8` (Q and the probabilities in e4m3 on the fp8 tensor cores) | `fp8` is the fast long prefill (262K cold prefill 242 s against 352 in bf16 over the same pool, the same decode); it adds a small deviation beyond the cache's own, measured at the bf16 cache's run-to-run floor in the [model card](model_cards/Qwen3.8-27B-NVFP4.md) |
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
| `engine.prefill_budget_tokens`, `engine.prefill_idle_budget_tokens` | shared reader-work policy for all serving families | automatic budgets learn pass cost; explicit positive values cap work. Smaller chunks yield sooner but repeat model walks. The idle ceiling is bounded by native capacity |
| `engine.decode_gap_target_ms`, `engine.first_token_target_ms` | soft decode-gap and arrival-to-first-token objectives, 1000 ms and 3000 ms respectively by default | tighter targets can reduce throughput; running chunks, long prompts and slot/KV pressure can exceed them. Both zero select fixed budgeting. On the mixed-prefill families the busy budget and the adaptive limit are one amount per pass shared by every reader, not per prompt. Until 2026-10-22 the controller charged the decoders' own pass to the chunk and could settle at 128 tokens beside several readers (a five-agent field deployment on one Spark read a 58–60 s first-token p90 against 12–15 s with fixed 1,024/2,048 budgets); it now learns that pass and charges the chunk only for what it adds |
| `engine.admission`, `engine.admission_window` | when and how much KV space a request reserves | `full` reserves prompt plus requested output; `grow` reserves incrementally and can shed a request on exhaustion. These settings do not select prefill grouping |
| `engine.prefill_group` | grouping compatible cold prompts into shared execution | amortizes model walks; batch geometry can change floating-point rounding |
| `engine.prefill` (DeepSeek) | `bounded` (the model's own recipe) or `exact` | about half the prefill work against parity with the full walk |
| `engine.prefill_bf16_partials` (Qwen AutoRound hybrid) | the packed expert chain's down projection in bf16, partials summed from bf16 | −3 to −7 % cold prefill (2K to 32K); not bitwise the fp32 chain |
| `engine.prefill_fp8_gemm` (Qwen with `dense_weights` fp8) | the dense stack's prefill GEMMs on the fp8 tensor cores from per-token e4m3 activations | 0 to −3 %; not bitwise the dequantized chain |
| `engine.prefill_fold_scales` (Qwen AutoRound hybrid) | the expert GEMM's group scales folded into the bf16 weight values | measured no gain (+3 % at 32K); kept for completeness, leave it off |

The numerical knobs `prefill_bf16_partials`, `prefill_fp8_gemm` and
`prefill_fold_scales` are the accuracy trades described in the README's [accuracy and correctness](../README.md#accuracy-and-correctness)
section: off unless a deployment turns them on. Scheduler budgets and latency
targets are separate controls; their [defaults and tuning](operations.md) apply
to every model family. The AutoRound hybrid's
template turns on the first two, whose evals stayed within the default
chain's run-to-run band; the fold is off everywhere.

### Tuning automatic prefill

Automatic budgeting chooses how much prompt processing runs before the next
scheduling opportunity. It starts at 256 tokens, adjusted to engine alignment,
then learns completed-pass cost, including decoding and snapshots. It keeps
separate estimates for different decoder loads and borrows the most conservative
known estimate for a new load range. The selected chunk leaves 15% time headroom
and grows by at most twice the last representative measured work. Slowdowns
update the estimate immediately; cheaper observations receive 25% weight during
recovery. Short final tails and completed prompt groups do not recalibrate bulk
prefill cost. Rank 0 journals the selected limit for every peer to replay.

The default allowance uses the 3,000 ms first-token target and, when decoding is
active, is also bounded by the 1,000 ms decode-gap target. An arrival queued
during an ongoing prefill with a free slot halves the first-token allowance to
reserve time for its preceding wait. This is a work-budget heuristic, not a
deadline computed from each request's age. A running GPU chunk is not preempted,
long prompts need multiple passes, and slot or KV pressure can exceed either
target. Initial requests can take extra passes while the controller learns.

The controller does not change concurrency, KV capacity, quantization or
speculative depth. Early admission lets fitting text requests join an ongoing
prefill; `prefill_early_admission: false` restores admission through leftovers.
This choice is independent of adaptive versus fixed budgeting. `mtp_schedule`
controls speculative verification depth separately. `admission: "full"` versus
`"grow"` chooses the memory-reservation policy, not the chunk-sizing policy.

To select the shared defaults, merge these fields into the deployment's
existing `engine` object, then restart:

```json
{
  "prefill_budget_tokens": -1,
  "prefill_idle_budget_tokens": 0,
  "decode_gap_target_ms": 1000,
  "first_token_target_ms": 3000
}
```

Omitting these fields has the same effect. Existing local files retain their
explicit overrides; updating an example does not update a running deployment.
Automatic mode uses an aligned ceiling of at most 4,096 tokens, bounded by native
capacity: the GLM/MiMo examples retain 2,048 and Qwen/DeepSeek retain 4,096.

| Configuration | Fields under `engine` | Use and tradeoff |
| --- | --- | --- |
| Automatic defaults | The four fields above, or omit them | Learn chunk sizes within native capacity. |
| More frequent decode opportunities | Automatic budgets, `decode_gap_target_ms: 500` | Aim for shorter output gaps; additional passes can reduce throughput. This is a tuning example, not a universal optimum. |
| Adaptive with explicit caps | `prefill_budget_tokens: 256`, `prefill_idle_budget_tokens: 2048`, retain latency targets | Keep configured work limits while allowing the controller to select smaller chunks. The caps bound the controller from above only: this exact configuration under five concurrent coding agents on one Spark produced the 58–60 s first-token p90 above (before the 2026-10-22 controller fix). For several concurrent long-prompt sessions, fixed budgets remain the measured configuration. |
| Fixed budgets | Positive busy/idle budgets, `decode_gap_target_ms: 0`, `first_token_target_ms: 0` | Disable adaptive resizing while retaining shared execution and early admission. Equal busy/idle values avoid the larger idle allowance. |
| Monolithic admission | `prefill_budget_tokens: 0` | Disable budgeted interleaving during admission; internal model and snapshot boundaries still apply. |

Positive budgets must meet the engine's alignment and capacity constraints;
the idle value must be at least the busy value. With an explicit busy budget,
an idle value of zero uses that budget as its ceiling. Disabling only one latency
target leaves the other objective active. Both must be zero for fixed budgeting.
The advanced controls are `prefill_initial_tokens` (256),
`prefill_latency_headroom_pct` (15), `prefill_growth_factor` (2), and
`prefill_cost_recovery_pct` (25). Change them after measuring: more headroom or
slower recovery is conservative; faster growth explores larger chunks sooner.

Tune in this order:

1. Size request slots and the shared KV pool for the workload. Queue capacity
   bounds accepted backlog; increasing it does not create compute capacity.
2. Choose acceptable first-token waits and output gaps. Begin with automatic
   defaults, then vary one target or cap at a time. Smaller chunks can help
   another stream while delaying the long prompt's own first token. For
   several concurrent agent sessions with long prompts, compare the defaults
   against fixed budgets (`prefill_budget_tokens: 1024`,
   `prefill_idle_budget_tokens: 2048`, both targets 0) on the deployment:
   the field evidence for that load is fixed budgets, and the busy budget is
   shared by every reader in a pass.
3. Compare fixed-work mixed loads using `scripts/serve_admission_bench.py`,
   including cold starts, steady traffic, arrivals, cancellation and recovery.
   Measure elapsed time and latency distributions alongside throughput.
4. Inspect `dgpp_first_token_target_misses_total`,
   `dgpp_decode_gap_target_misses_total`, `dgpp_prefill_adaptive_limit_tokens`
   and the prefill-work counters. High budget utilization alone does not
   establish good latency or throughput. Mixed prefill/decode phase times overlap;
   use completed work over wall time when comparing mixed workloads.

See [operations](operations.md) for setting ranges and metric definitions,
[deployment defaults](../deploy/README.md#shared-scheduler-defaults) for migration,
and [shared execution](continuous-batching.md#current-shared-execution-and-policy)
for fairness, state ownership and timing. Historical benchmark rows retain their
recorded fixed-budget settings; they are not measurements of the adaptive defaults.

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
ahead, the shared adaptive prefill policy with a 4,096-token ceiling,
and the two prefill levers that paid without a measurable quality
cost. Its rows in [benchmarks](benchmarks.md) show the default chain and
the levers-on configuration separately. The
[Qwen int4 plan](qwen38_autoround_int4_plan.md) explains the selected
optimizations and rejected experiments.

## 4. Where the floor is

A decode pass is bound by the bytes it streams (the resident weights a
token touches plus the KV it reads) over the Spark's memory bandwidth, plus
a launch floor per pass; a prefill chunk by its dense and expert GEMMs at
the tensor cores' rate. The per-model plans (`docs/*_plan.md`) state each
floor next to the measured step, and the gap between them is the work
list. When a change makes a number worse than that gap allows, suspect the
measurement before the code: a stale binary, an overlapping build, a
second world on the fabric, or a cold first leg.
