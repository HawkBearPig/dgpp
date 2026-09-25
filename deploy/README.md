# Deployment templates

One template per model, quantization and world size:
`cluster_<model>_<quant>_w<n>.example.json`. An optional trailing
`_<variant>` names a template that deviates from that shape in one documented
engine setting rather than in its size — today
`cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json`, the same two-Spark
NVFP4 deployment with the YaRN rope ramp switched on. Base shapes stay unique;
a variant may repeat a base only with that suffix. Local copies use the same name
without `.example` and stay Git-ignored; the launcher fills the nodes, the SSH
user and the ports from the site's `.env` (`scripts/site_env.py`).

- Model names are `glm-5.3-flash`, `glm-5.3` (the full model), `glm-4.7`,
  `qwen-3.8-flash-next`, `deepseek-v4.1-flash` and `mimo-v2.6-flash`.
- The quant names the checkpoint representation: `fp8`, `nvfp4`,
  `nvfp4-fp8` for the custom GLM-5.3-Flash hybrid, `int4-int8` for the full
  GLM-5.3's pack-quantized release (int4 group-64 routed experts, int8
  attention and shared experts), `mxfp4-fp8` for DeepSeek-V4.1-Flash as it
  ships (MXFP4 experts, FP8 dense and attention, FP8 Engram tables mapped
  from the NVMe) and for MiMo-V2.6-Flash as it ships (MXFP4 experts, FP8
  fused attention projections and dense MLPs, BF16 o_proj / head). The
  JSON's `model` field gives the exact Hugging Face repository.
- `w<n>` gives the participating node count. Nothing restricts that number to
  the counts in use today; a world is refused by the engine's geometry check or
  a rank's memory plan, not by a list of allowed sizes.

Every template enables MTP (the block draft on DeepSeek) at the depth the
family measured best, with the decode graph, at the request-slot count and
cache budget that measured at or above every other shape tried. The
long-context templates also carry a top-level `nvme_cache` block, off by
default: set `enabled` to true and size `capacity_gib` to the node's free
NVMe to keep evicted prefix entries on disk (the [README](../README.md#nvme-cache)
lists the keys; startup checks the space). The shapes
a template does not name are knobs appended at boot:
`scripts/dgpp-cluster up --config FILE --knobs "FLAGS"`.

| Template | Deployment | The shapes it replaced, as knobs |
|---|---|---|
| [cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json](cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json) | GLM-5.3-Flash hybrid on four nodes: MTP depth 1, bf16 latent cache, 768K context, an 8 GiB prefix arena, four request slots | the 8K-context template: `--kv-capacity 8192 --prefix-cache-gib 1.5`; T=1: `--no-mtp` |
| [cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json](cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json) | the same hybrid on two nodes: MTP depth 1, FP8 latent cache, 160K context, four request slots; the BF16 decode weights resident in their 12-bit form alone (`"bf12"`: 107.0 GiB per rank, 1 GiB under the BF16 plan, 4.8 GiB of the node left at boot) | both forms resident (no prefill cost, 132K context): `--bf16-weights bf12+bf16 --kv-capacity 135168`; the 256K-context two-slot shape: `--max-concurrency 2 --kv-capacity 262144 --prefix-cache-gib 2` |
| [cluster_qwen-3.8-flash-next_fp8_w4.example.json](cluster_qwen-3.8-flash-next_fp8_w4.example.json) | Qwen3.8-Flash-Next FP8 on four nodes: MTP depth 1, 256K context, four request slots | T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_fp8_w2.example.json](cluster_qwen-3.8-flash-next_fp8_w2.example.json) | the same on two nodes | T=1: `--no-mtp` |
| [cluster_qwen-3.8-flash-next_nvfp4_w1.example.json](cluster_qwen-3.8-flash-next_nvfp4_w1.example.json) | Qwen3.8-Flash-Next NVFP4 on one Spark: MTP depth 1, the dense projections FP8 at load (`dense_weights: "fp8"`: 31 ms/step T=1 and 21–26 ms/token against the BF16 stack's 38 and 31–38), the n-gram table mapped, 64K context | the BF16 dense stack: `--dense-weights checkpoint --fp8-head gemv`; T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_nvfp4_w2.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2.example.json) | Qwen3.8-Flash-Next NVFP4 on two Sparks: MTP depth 1, FP8 dense projections, the n-gram table mapped, 262K context and four request slots | the resident n-gram table: `--ngram-table resident`; T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_nvfp4-radixark_w1.example.json](cluster_qwen-3.8-flash-next_nvfp4-radixark_w1.example.json) | RadixArk's Qwen3.8-Flash-Next NVFP4 on one Spark: identical engine configuration to the NVIDIA release, MTP depth 1, dense projections FP8 at load, n-gram table mapped, 64K context | the BF16 dense stack: `--dense-weights checkpoint`; T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_nvfp4-radixark_w2.example.json](cluster_qwen-3.8-flash-next_nvfp4-radixark_w2.example.json) | RadixArk's Qwen3.8-Flash-Next NVFP4 on two Sparks: identical engine configuration to the NVIDIA release, MTP depth 1, FP8 dense projections, n-gram table mapped, 262K context and four request slots | the resident n-gram table: `--ngram-table resident`; T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json) | the two-Spark NVFP4 deployment with the opt-in YaRN ramp: `engine.rope_scaling` yarn ×2 over the checkpoint's 262 144 positions, so one request reaches 524 288 tokens; two request slots, a 532 480-token pool, the n-gram table mapped, FP8 dense projections, MTP depth 1 | two full-length streams at once: `--kv-capacity 1048576` (see [the YaRN notes](#the-512k-yarn-template-enginerope_scaling)); the plain 262K template: [cluster_qwen-3.8-flash-next_nvfp4_w2.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2.example.json) |
| [cluster_glm-4.7_nvfp4_w4.example.json](cluster_glm-4.7_nvfp4_w4.example.json) | GLM-4.7 NVFP4 on four nodes: MTP depth 1, 256K context, four request slots | T=1: `--no-mtp`; depth 2: `--mtp-depth 2` (single-stream +4–13 %, measured behind depth 1 under concurrency before the 2026-09-14 lowering) |
| [cluster_glm-5.3_int4-int8_w4.example.json](cluster_glm-5.3_int4-int8_w4.example.json) | the full GLM-5.3 (int4/int8 RTN) on four nodes: MTP depth 1, eight request slots (sixteen decode rows; c=4 the four-slot shape's 41–42 tok/s, c=8 48–50 aggregate), 120K bf16 context, the embedding vocab-sharded; the BF16 decode weights resident in their 12-bit form alone (`"bf12"`: 110.1 GiB per rank under the 4 GiB headroom — 0.3 GiB under the BF16 plan, where both forms resident stopped at 100K) | both forms resident (100K context): `--bf16-weights bf12+bf16 --kv-capacity 102400`; T=1: `--no-mtp` (144K context with `--kv-capacity 147456`); the fp8 latent cache at 208K: `--kv-dtype fp8 --kv-capacity 212992 --prefix-cache-gib 1.5`; depth 2 at two slots: `--mtp-depth 2 --max-concurrency 2` |
| [cluster_mimo-v2.6-flash_mxfp4-fp8_w4.example.json](cluster_mimo-v2.6-flash_mxfp4-fp8_w4.example.json) | MiMo-V2.6-Flash as shipped on four nodes: MTP depth 1 (the first of the release's three draft layers), the BF16 o_proj / head / eh_proj resident in their 12-bit form alone (`"bf12"`), 128K context (every layer's K/V paged, the sliding-window layers read theirs through the window), four request slots; text prompts only (the vision and audio encoders are not served) | T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_mimo-v2.6-flash_mxfp4-fp8_w2.example.json](cluster_mimo-v2.6-flash_mxfp4-fp8_w2.example.json) | the same on two nodes: MTP depth 1, the K/V cache in the fp8 row form (`"kv_dtype": "fp8"`: 58 KiB per token per rank against 109 in bf16), 256K context (97.5 GiB of the 121.6 GiB per rank), four request slots | T=1: `--no-mtp` |
| [cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json](cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json) | DeepSeek-V4.1-Flash as shipped on four nodes: six request slots at DSpark depth 4 (30 decode rows in one batched replay, the family's 32-row cap) with the confidence-scheduled verify depth (λ 0.045), the bounded prefill, 128K context — the six-stream shape the vLLM recipe reports its aggregate at; single and dual streams measure the same as the two-slot shapes did | the two-slot depth-5 shape: `--max-concurrency 2 --mtp-depth 5`; the two-slot depth-4 shape: `--max-concurrency 2`; T=1 at four slots: `--no-mtp --max-concurrency 4` |

The Qwen NVFP4 templates use `engine.fp8_head: "mma"` after matched
[real-checkpoint numerical checks](../benchmarks/results/2026-09-21-qwen-fp8-head-numerics.md)
on one and two Sparks, including the YaRN recipe. Streaming MMA applies above
the dense GEMV threshold and up to the configured decode-row capacity; short
prefills in that interval use it too. Pass `--knobs "--fp8-head gemv"` to
restore the previous head path. For BF16 dense weights, use
`--knobs "--dense-weights checkpoint --fp8-head gemv"`. The FP8-checkpoint
templates retain their BF16 dense/head stack and do not select this setting.

`engine.bf16_weights` (`"checkpoint"` by default; every template sets `"bf12"`
or `"bf12+bf16"`) keeps a lossless 12-bit form of each BF16 matrix the decode
GEMV streams — the sign+mantissa byte and a 4-bit exponent code, exact side
tables for the outliers — so a decode launch of up to eight rows reads 0.75 of
those bytes and returns the same bits (the transcripts do not move; it is a
storage format, not a quantization). Measured single-stream decode:
GLM-5.3-Flash +6–7 % (+4–6 % at two to four live requests), GLM-4.7 +11–12 %
(+5–7 % at four), the full GLM-5.3 +5–6.5 %. The two values differ in what stays
resident, and the memory plan carries either:

| value | resident | memory against `checkpoint` | prefill |
|---|---|---|---|
| `"bf12+bf16"` | both forms: decode streams the 12-bit one, prefill reads the BF16 bytes in place | GLM-5.3-Flash +2.0 GiB per rank on four nodes (+3.8 on two), GLM-4.7 +4.9, the full GLM-5.3 +1.3 | unchanged |
| `"bf12"` | the 12-bit form ALONE: each matrix's BF16 bytes go back to the node as its layer loads, and a prefill GEMM expands the rows it reads into a small scratch (the same bits, so the same results) | GLM-5.3-Flash −0.5 GiB per rank on four nodes (−1.0 on two), GLM-4.7 −1.4, the full GLM-5.3 −0.3 | about 10 ms per prefill chunk on four-node GLM-5.3-Flash (20 on two nodes): +1–2 % on 8K–32K prompts, +3–4 % at 2K, +25–40 ms to a short prompt's first token; within 1 % on the full GLM-5.3 |

The templates with memory to spare (four-node GLM-5.3-Flash, GLM-4.7) take
`"bf12+bf16"`; the two sized to their nodes' ceiling (two-node GLM-5.3-Flash,
the full GLM-5.3) take `"bf12"`, which is what returned their contexts to 160K
and 120K. Decode is the same either way — identical transcripts, measured on
every family. Under `"bf12"` a decode batch past eight rows (the full
GLM-5.3's sixteen-row shape) runs eight-row packed launches, level with the
BF16 algorithm it replaces. On Qwen3.8-Flash-Next-FP8 the key packs the GDN and
QSA projections, the draft block's and the head — 1.65 → 1.24 GiB per rank on
four nodes, 3.20 → 2.41 on two — for +6.4 / +3.8 / +2.4 / +1.3 % at one to
four live requests on four nodes and +9.0 / +6.7 / +6.6 / +4.6 % on two,
transcripts identical; that family keeps both forms resident under either
value (every Qwen recipe has the room), its GR sites, routers and shared
experts stay BF16 (they are read warm behind the prefetcher, where the packed
form loses), and under `dense_weights: "fp8"` (the NVFP4 templates) the
projections and the head are already FP8, so nothing is packed. DeepSeek
accepts the key and packs nothing yet (its BF16 sites ride the tensor-core
kernels). `--bf16-weights checkpoint` restores the BF16-only form.

`engine.embed_sharding` (`"replicated"` by default, `"vocab"` in the full
GLM-5.3 template) decides whether every rank holds the whole embedding table
or its lm-head slice of the rows: `vocab` frees 1.33 GiB per rank at world 4
for one small fold per token lookup and changes no number; the other families
ignore it. `kv_dtype` affects only the GLM-5.3 latent caches (bf16, fp8 or
fp4); Qwen's and GLM-4.7's K/V caches stay BF16.

## The 512K YaRN template (`engine.rope_scaling`)

[cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json)
is ready to run on the two-Spark kit: copy it without `.example`, fill the site
fields through `.env` as usual, and start the world with
`scripts/dgpp-cluster up --config deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.json`.
The one setting it adds over the plain two-Spark template is the rope ramp:

```json
"rope_scaling": {"rope_type": "yarn", "factor": 2.0,
                 "original_max_position_embeddings": 262144,
                 "beta_fast": 32, "beta_slow": 1, "attn_factor": 1.0,
                 "mrope_cache_factor": 4.0}
```

**YaRN raises the positional ceiling. It does not enlarge the K/V pool.** The
ramp re-derives the rope table's inverse frequencies so a token can be *placed*
at a position up to `original_max_position_embeddings × factor` = 524 288; the
pool the tokens are *stored* in is `engine.kv_capacity`, and the ramp moves no
byte of it (`docs/qwen38_flash_next_plan.md` §1.9.1; the plan check asserts the
total is identical with the knob on and off). Two numbers follow from that, and
`dgpp-serve` prints the smaller of them at startup and on `/v1/models` as the
request context limit:

- the **ceiling**: 524 288 positions with this template (262 144 without it),
- the **pool**: `kv_capacity` rounded up to the 64-token block, ~13 KB per
  token per rank at world 2 (`docs/qwen38_checkpoint_budget.md`).

The template sets `kv_capacity` 532 480 — the ceiling plus 8 192 tokens of room,
because a request reserves its prompt *and* its answer from the pool: a 524 288-token
prompt with the `default_max_tokens` 256 the template carries already needs
524 544, and a pool cut at exactly the ceiling would refuse the longest request
it exists to serve. With `max_concurrency` 2 that pool seats one full-length
stream, or two streams of up to ~266 000 tokens each, or any mix that fits; a
third request waits at the door (`queue_limit` 8) rather than sharing bytes. For
two concurrent 524 288-token streams set `--kv-capacity 1048576`.

Memory headroom: at TP=2 the resident weights are 44.1 GiB per rank, the pool is
6.6 GiB at 532 480 tokens (13.0 GiB at 1 048 576) and the plan's activations,
the prefix arena and the engine's tables come on top — 51.73 GiB per rank for
this shape, plus the 4 GiB headroom the pre-flight check adds
(`--memory-plan` prints the itemised total and refuses the configuration before
the first allocation if the node cannot cover it). The 1 048 576-token variant
is ~58.1 GiB plus headroom per rank. Nothing about the rope ramp changes those
numbers: they move with `kv_capacity`, `max_concurrency` and `prefix_cache_gib`.

The current 1.5 GiB prefix arena holds 27 snapshots per rank at TP=2.
Long prompts also retain an earlier chunk snapshot so changed questions can
share the document's KV blocks. Snapshot size does not grow with context;
KV-pool capacity limits how many independent long documents remain cached.
See [prefix-cache sizing](../docs/prefix-cache.md) for all recipes and retention
estimates before increasing the arena budget.

Two things to check on the boot log, both added with this knob:

- `serve: request context limit 524288 tokens — the lesser of the 524288-token
  positional ceiling (engine.rope_scaling yarn x2 over 262144 positions) and the
  532480-token K/V pool …` — if the ceiling it names is 262144 the knob did not
  reach the process,
- `engine.rope_scaling is set but the loaded model is the <family> family` is a
  **startup error**, not a note: the ramp is built for the Qwen3.8-Flash-Next
  QSA rope and no other family's, and the world refuses to start with an
  operator's setting left unapplied.

The checkpoint must be a plain one (`rope_parameters.rope_type: "default"`),
which the NVFP4 and FP8 releases are; a checkpoint that already bakes YaRN into
its own `rope_parameters` is rejected at load, for the reason
`docs/qwen38_flash_next_plan.md` §1.9.1 gives.
`scripts/qwen_yarn_release_check.py` is the opt-in, long-context acceptance run
for this template ([docs/qwen_yarn_release_check.md](../docs/qwen_yarn_release_check.md)).

## The consolidation (2026-09-14)

The earlier consolidation reduced twenty-five templates to eight. The new
two-Spark NVFP4 recipe brings the tracked set to nine. Every `_plain` variant
(MTP measured faster per token in every family), every `_mtp2` / `_mtp5`
variant (a depth is
a knob), the `_c6` / `_c8` slot variants (the wider shape measured at or above
the narrower one at every concurrency, so it is the template), the
`_large-cache` variants (the four-node GLM-5.3-Flash template took the large
cache; the two-node one and the full GLM-5.3's fp8 cache are knobs) and the
Qwen single-Spark BF16-dense variants (the FP8 dense stack measured faster).
The retired names map to the table's third column; historical changelog
entries and benchmark records keep the old names.

| Retired template | Now |
|---|---|
| `cluster_glm-5.3-flash_nvfp4-fp8_w4_mtp1` (8K context), `…_w4_mtp1_large-cache` | `cluster_glm-5.3-flash_nvfp4-fp8_w4` (the large cache) |
| `cluster_glm-5.3-flash_nvfp4-fp8_w2_mtp1`, `…_w2_mtp1_large-cache` | `cluster_glm-5.3-flash_nvfp4-fp8_w2` (four slots, 160K; 132K since the 2026-09-19 companions) |
| `cluster_qwen-3.8-flash-next_fp8_w4_{mtp1,plain}`, `…_w2_{mtp1,plain}` | `cluster_qwen-3.8-flash-next_fp8_w4`, `…_w2` |
| `cluster_qwen-3.8-flash-next_nvfp4_w1_{mtp1,plain,mtp1_dense-fp8,plain_dense-fp8,mtp2_dense-fp8}` | `cluster_qwen-3.8-flash-next_nvfp4_w1` (FP8 dense) |
| `cluster_glm-4.7_nvfp4_w4_{mtp1,plain,mtp2}` | `cluster_glm-4.7_nvfp4_w4` |
| `cluster_glm-5.3_int4-int8_w4_{plain,mtp1,mtp1_large-cache,mtp1_c8,mtp2}` | `cluster_glm-5.3_int4-int8_w4` (eight slots) |
| `cluster_deepseek-v4.1-flash_mxfp4-fp8_w4_{plain,mtp4,mtp4_c6,mtp5}` | `cluster_deepseek-v4.1-flash_mxfp4-fp8_w4` (six slots, depth 4) |

## Existing deployments and logs

Stop a running deployment using its old config path **before** renaming the
file. Log and staging namespaces are derived from the absolute config path;
the new filename gets a new namespace. Existing logs are not moved or deleted,
and a new filename does not take ownership of a process started with the old
one. Update saved commands and automation to use the new name.

The generic runtime files `cluster.resolved.json` and the peer's staged
`cluster.json` are unchanged: they are generated by the launcher, not templates.
