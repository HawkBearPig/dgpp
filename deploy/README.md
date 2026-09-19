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
  `qwen-3.8-flash-next` and `deepseek-v4.1-flash`.
- The quant names the checkpoint representation: `fp8`, `nvfp4`,
  `nvfp4-fp8` for the custom GLM-5.3-Flash hybrid, `int4-int8` for the full
  GLM-5.3's pack-quantized release (int4 group-64 routed experts, int8
  attention and shared experts), `mxfp4-fp8` for DeepSeek-V4.1-Flash as it
  ships (MXFP4 experts, FP8 dense and attention, FP8 Engram tables mapped
  from the NVMe). The JSON's `model` field gives the exact Hugging Face
  repository.
- `w<n>` gives the participating node count. Nothing restricts that number to
  the counts in use today; a world is refused by the engine's geometry check or
  a rank's memory plan, not by a list of allowed sizes.

Every template enables MTP (the block draft on DeepSeek) at the depth the
family measured best, with the decode graph, at the request-slot count and
cache budget that measured at or above every other shape tried. The shapes
a template does not name are knobs appended at boot:
`scripts/dgpp-cluster up --config FILE --knobs "FLAGS"`.

| Template | Deployment | The shapes it replaced, as knobs |
|---|---|---|
| [cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json](cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json) | GLM-5.3-Flash hybrid on four nodes: MTP depth 1, bf16 latent cache, 768K context, an 8 GiB prefix arena, four request slots | the 8K-context template: `--kv-capacity 8192 --prefix-cache-gib 1.5`; T=1: `--no-mtp` |
| [cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json](cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json) | the same hybrid on two nodes: MTP depth 1, FP8 latent cache, 160K context, four request slots | the 256K-context two-slot shape: `--max-concurrency 2 --kv-capacity 262144 --prefix-cache-gib 2` |
| [cluster_qwen-3.8-flash-next_fp8_w4.example.json](cluster_qwen-3.8-flash-next_fp8_w4.example.json) | Qwen3.8-Flash-Next FP8 on four nodes: MTP depth 1, 256K context, four request slots | T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_fp8_w2.example.json](cluster_qwen-3.8-flash-next_fp8_w2.example.json) | the same on two nodes | T=1: `--no-mtp` |
| [cluster_qwen-3.8-flash-next_nvfp4_w1.example.json](cluster_qwen-3.8-flash-next_nvfp4_w1.example.json) | Qwen3.8-Flash-Next NVFP4 on one Spark: MTP depth 1, the dense projections FP8 at load (`dense_weights: "fp8"`: 31 ms/step T=1 and 21–26 ms/token against the BF16 stack's 38 and 31–38), the n-gram table mapped, 64K context | the BF16 dense stack: `--dense-weights checkpoint`; T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_nvfp4_w2.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2.example.json) | Qwen3.8-Flash-Next NVFP4 on two Sparks: MTP depth 1, FP8 dense projections, the n-gram table mapped, 262K context and four request slots | the resident n-gram table: `--ngram-table resident`; T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json) | the two-Spark NVFP4 deployment with the opt-in YaRN ramp: `engine.rope_scaling` yarn ×2 over the checkpoint's 262 144 positions, so one request reaches 524 288 tokens; two request slots, a 532 480-token pool, the n-gram table mapped, FP8 dense projections, MTP depth 1 | two full-length streams at once: `--kv-capacity 1048576` (see [the YaRN notes](#the-512k-yarn-template-enginerope_scaling)); the plain 262K template: [cluster_qwen-3.8-flash-next_nvfp4_w2.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2.example.json) |
| [cluster_glm-4.7_nvfp4_w4.example.json](cluster_glm-4.7_nvfp4_w4.example.json) | GLM-4.7 NVFP4 on four nodes: MTP depth 1, 256K context, four request slots | T=1: `--no-mtp`; depth 2: `--mtp-depth 2` (single-stream +4–13 %, measured behind depth 1 under concurrency before the 2026-09-14 lowering) |
| [cluster_glm-5.3_int4-int8_w4.example.json](cluster_glm-5.3_int4-int8_w4.example.json) | the full GLM-5.3 (int4/int8 RTN) on four nodes: MTP depth 1, eight request slots (sixteen decode rows; c=4 the four-slot shape's 41–42 tok/s, c=8 48–50 aggregate), 120K bf16 context, the embedding vocab-sharded (110.4 GiB per rank under the 4 GiB headroom) | T=1: `--no-mtp` (144K context: `--kv-capacity 147456`); the fp8 latent cache at 208K: `--kv-dtype fp8 --kv-capacity 212992 --prefix-cache-gib 1.5`; depth 2 at two slots: `--mtp-depth 2 --max-concurrency 2` |
| [cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json](cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json) | DeepSeek-V4.1-Flash as shipped on four nodes: six request slots at DSpark depth 4 (30 decode rows in one batched replay, the family's 32-row cap) with the confidence-scheduled verify depth (λ 0.045), the bounded prefill, 128K context — the six-stream shape the vLLM recipe reports its aggregate at; single and dual streams measure the same as the two-slot shapes did | the two-slot depth-5 shape: `--max-concurrency 2 --mtp-depth 5`; the two-slot depth-4 shape: `--max-concurrency 2`; T=1 at four slots: `--no-mtp --max-concurrency 4` |

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
| `cluster_glm-5.3-flash_nvfp4-fp8_w2_mtp1`, `…_w2_mtp1_large-cache` | `cluster_glm-5.3-flash_nvfp4-fp8_w2` (four slots, 160K) |
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
