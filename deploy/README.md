# Deployment templates

One template per model, quantization and world size:
`cluster_<model>_<quant>_w<n>.example.json`. An optional trailing
`_<variant>` names a template that deviates from that shape in one documented
engine setting rather than in its size — today
`cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json`, the same two-Spark
NVFP4 deployment with the YaRN rope ramp switched on. Base shapes stay unique;
a variant may repeat a base only with that suffix. Local copies use the same name
without `.example` and stay Git-ignored; the launcher fills the nodes, the SSH
user and the HTTP/fabric/journal port defaults from the site's `.env`
(`scripts/site_env.py`). Set `"ports": {"metrics": 29972}` in a deployment
to enable each peer's Prometheus listener; an omitted port or `0` disables
it. Rank 0 serves metrics on its HTTP endpoint. See the
[per-rank metrics](../docs/openai-compatibility.md#per-rank-families).

- Model names are `glm-5.3-flash`, `glm-5.3` (the full model), `glm-4.7`,
  `qwen-3.8-flash-next`, `qwen3.8-27b`, `deepseek-v4-flash`,
  `deepseek-v4.1-flash` and `mimo-v2.6-flash`.
- The quant names the checkpoint representation: `fp8`, `nvfp4`,
  `nvfp4-fp8` for the custom GLM-5.3-Flash hybrid, `int4-int8` for the full
  GLM-5.3's pack-quantized release (int4 group-64 routed experts, int8
  attention and shared experts), `nf4i8` for its NF4I8 GPTQ H32 g128 re-encoding
  (codebook routed experts, bf16 scales per 128, the H32 input rotation),
  `mixed346` for its Mixed346 GPTQ H32 A8 g128 re-encoding (3/4/6-bit codebook
  experts with int8 activation codes), `mxfp4-fp8` for DeepSeek-V4.1-Flash as it
  ships (MXFP4 experts, FP8 dense and attention, FP8 Engram tables mapped
  from the NVMe) and for MiMo-V2.6-Flash as it ships (MXFP4 experts, FP8
  fused attention projections and dense MLPs, BF16 o_proj / head). The
  JSON's `model` field gives the exact Hugging Face repository.
- `w<n>` gives the participating node count. Nothing restricts that number to
  the counts in use today; a world is refused by the engine's geometry check or
  a rank's memory plan, not by a list of allowed sizes.

Every template enables MTP (the block draft on DeepSeek) at the depth the
family measured best, with the decode graph, at the request-slot count and
cache budget that measured at or above every other shape tried — except the
Qwen3.8-27B templates, which name `engine.dflash_model` instead of MTP: the
external block drafter replacing the MTP draft (the block proposal
recorded inside the graph step at every world; one Spark without the
decode graph runs it on the eager engine). The shapes
a template does not name are knobs appended at boot:
`scripts/dgpp-cluster up --config FILE --knobs "FLAGS"`.

| Template | Deployment | The shapes it replaced, as knobs |
|---|---|---|
| [cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json](cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json) | GLM-5.3-Flash hybrid on four nodes: MTP depth 1, bf16 latent cache, 768K context, a 22 GiB prefix arena (639 snapshots), four request slots | the 8K-context template: `--kv-capacity 8192 --prefix-cache-gib 1.5`; T=1: `--no-mtp` |
| [cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json](cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json) | the same hybrid on two nodes: MTP depth 1, FP8 latent cache, 160K context, four request slots; the BF16 decode weights resident in their 12-bit form alone (`"bf12"`: 110.0 GiB per rank with its 4.5 GiB prefix arena of 65 snapshots, 1.8 GiB of the node left at boot) | both forms resident (no prefill cost, 132K context): `--bf16-weights bf12+bf16 --kv-capacity 135168`; the 256K-context two-slot shape: `--max-concurrency 2 --kv-capacity 262144 --prefix-cache-gib 2` |
| [cluster_qwen-3.8-flash-next_fp8_w4.example.json](cluster_qwen-3.8-flash-next_fp8_w4.example.json) | Qwen3.8-Flash-Next FP8 on four nodes: MTP depth 1, 256K context, four request slots, a 1,052,672-token shared KV pool and a 40 GiB prefix arena (1476 snapshots) | T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_fp8_w2.example.json](cluster_qwen-3.8-flash-next_fp8_w2.example.json) | the same on two nodes | T=1: `--no-mtp` |
| [cluster_qwen-3.8-flash-next_nvfp4_w1.example.json](cluster_qwen-3.8-flash-next_nvfp4_w1.example.json) | Qwen3.8-Flash-Next NVFP4 on one Spark: MTP depth 1, the dense projections FP8 at load (`dense_weights: "fp8"`: 31 ms/step T=1 and 21–26 ms/token against the BF16 stack's 38 and 31–38), the n-gram table mapped, a 262144-token shared KV pool and the shared adaptive prefill policy | the BF16 dense stack: `--dense-weights checkpoint --fp8-head gemv`; T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_nvfp4_w2.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2.example.json) | Qwen3.8-Flash-Next NVFP4 on two Sparks: MTP depth 1, FP8 dense projections, the n-gram table mapped, 262K context, four request slots and a 58 GiB prefix arena (1074 snapshots) | the resident n-gram table: `--ngram-table resident`; T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_nvfp4-radixark_w1.example.json](cluster_qwen-3.8-flash-next_nvfp4-radixark_w1.example.json) | RadixArk's Qwen3.8-Flash-Next NVFP4 on one Spark: MTP depth 2, grow admission and the shared adaptive prefill policy, dense projections FP8 at load, n-gram table mapped, 256K context | the BF16 dense stack: `--dense-weights checkpoint`; T=1: `--no-mtp`; depth 1: `--mtp-depth 1` |
| [cluster_qwen-3.8-flash-next_nvfp4-radixark_w1_yarn512k.example.json](cluster_qwen-3.8-flash-next_nvfp4-radixark_w1_yarn512k.example.json) | the one-Spark RadixArk deployment with the opt-in YaRN ramp: `engine.rope_scaling` yarn ×2 over the checkpoint's 262 144 positions, so one request reaches 524 288 tokens; two request slots sharing a 565 248-token pool, MTP depth 2 with the lookup draft, full admission, the shared adaptive prefill policy, a 3 GiB prefix arena (27 snapshots: the one-node Qwen templates keep the arena small so the page cache serves the mmap'ed n-gram table); plan-verified from the checkpoint's config (100.9 GiB), not release-checked here | two full-length streams at once: `--kv-capacity 1114112` (see [the YaRN notes](#the-512k-yarn-templates-enginerope_scaling)); the plain 262K template: [cluster_qwen-3.8-flash-next_nvfp4-radixark_w1.example.json](cluster_qwen-3.8-flash-next_nvfp4-radixark_w1.example.json) |
| [cluster_qwen-3.8-flash-next_nvfp4-radixark_w2.example.json](cluster_qwen-3.8-flash-next_nvfp4-radixark_w2.example.json) | RadixArk's Qwen3.8-Flash-Next NVFP4 on two Sparks: MTP depth 4 with BF16 fused draft experts, FP8 dense projections, n-gram table mapped, a 524288-token shared KV pool, four request slots and a 32 GiB prefix arena | the resident n-gram table: `--ngram-table resident`; T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_qwen-3.8-flash-next_nvfp4-qad_w1.example.json](cluster_qwen-3.8-flash-next_nvfp4-qad_w1.example.json) | the quantization-aware-distilled export (`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`) on one Spark: MTP depth 1, the dense stack and shared experts in their shipped MXFP8, the NVFP4 n-gram table resident (26.8 GiB, no NVMe gather), the BF16 remainder (GR sites, PLE projections, head) FP8 at load as the NVIDIA recipe, a 128K shared pool and a 1 GiB prefix arena (the resident table takes the room the NVIDIA recipe spends on a 256K pool; `--ngram-table mmap --kv-capacity 262144 --prefix-cache-gib 3` trades the gather back for it) | the BF16 remainder as shipped: `--dense-weights checkpoint --fp8-head gemv`; T=1: `--no-mtp` |
| [cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json) | the two-Spark NVFP4 deployment with the opt-in YaRN ramp: `engine.rope_scaling` yarn ×2 over the checkpoint's 262 144 positions, so one request reaches 524 288 tokens; two request slots, a 565 248-token pool, the n-gram table mapped, FP8 dense projections, MTP depth 1 | two full-length streams at once: `--kv-capacity 1114112` (see [the YaRN notes](#the-512k-yarn-templates-enginerope_scaling)); the plain 262K template: [cluster_qwen-3.8-flash-next_nvfp4_w2.example.json](cluster_qwen-3.8-flash-next_nvfp4_w2.example.json) |
| [cluster_qwen-3.8-flash-next_autoround-int4_w1.example.json](cluster_qwen-3.8-flash-next_autoround-int4_w1.example.json) | the AutoRound int4/int8 hybrid (`Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN`) on one Spark in its packed form: MTP depth 3, the dense projections FP8 at load, a 256K BF16 KV pool, four request slots, a 3 GiB prefix cache; `engine.draft_vocab` (a `.npy` id set for the draft head) is opt-in and off | T=1: `--no-mtp` |
| [cluster_glm-4.7_nvfp4_w4.example.json](cluster_glm-4.7_nvfp4_w4.example.json) | GLM-4.7 NVFP4 on four nodes: MTP depth 1, 256K context, four request slots | T=1: `--no-mtp`; depth 2: `--mtp-depth 2` (single-stream +4–13 %, measured behind depth 1 under concurrency before the 2026-09-14 lowering) |
| [cluster_glm-5.3_int4-int8_w4.example.json](cluster_glm-5.3_int4-int8_w4.example.json) | the full GLM-5.3 (int4/int8 RTN) on four nodes: MTP depth 1, eight request slots (sixteen decode rows; c=4 the four-slot shape's 41–42 tok/s, c=8 48–50 aggregate), 120K bf16 context, the embedding vocab-sharded; the BF16 decode weights resident in their 12-bit form alone (`"bf12"`: 110.1 GiB per rank under the 4 GiB headroom — 0.3 GiB under the BF16 plan, where both forms resident stopped at 100K) | both forms resident (100K context): `--bf16-weights bf12+bf16 --kv-capacity 102400`; T=1: `--no-mtp` (144K context with `--kv-capacity 147456`); the fp8 latent cache at 208K: `--kv-dtype fp8 --kv-capacity 212992 --prefix-cache-gib 1.5`; depth 2 at two slots: `--mtp-depth 2 --max-concurrency 2` |
| [cluster_qwen3.8-27b_nvfp4_w1.example.json](cluster_qwen3.8-27b_nvfp4_w1.example.json) | Qwen3.8-27B NVFP4 (the mixed release: NVFP4 MLPs, channel-FP8 attention and head, BF16 draft layer) on one node, the DFlash2 block drafter in FP8 | the FP8 card's flags carry over; `--dense-weights fp8` is refused (the head is already FP8-resident) |
| [cluster_qwen3.8-27b_nvfp4_w1_yarn512k.example.json](cluster_qwen3.8-27b_nvfp4_w1_yarn512k.example.json) | the one-node 27B NVFP4 deployment with the opt-in YaRN ramp: `engine.rope_scaling` yarn ×2 over the checkpoint's 262 144 positions, so one request reaches 524 288 tokens; two request slots sharing a 565 248-token pool, the scheduled DFlash2 block draft, full admission, a 1.5 GiB prefix arena. Release-checked: 5/5 needles at 262 144 and 524 288, 100% prefix reuse, short-context quality 155/164 + 294/300 + 100/100 against plain's 153/290/100 (2026-10-12), 3/3 at 524 288 again on 2026-10-15 on one Spark (cold prefill 1 842 s with the adaptive policy's idle floor) and on four Sparks (713 s) with the four-node template plus this rope block | two full-length streams at once: `--kv-capacity 1114112` (see [the YaRN notes](#the-512k-yarn-templates-enginerope_scaling)); the plain 262K template: [cluster_qwen3.8-27b_nvfp4_w1.example.json](cluster_qwen3.8-27b_nvfp4_w1.example.json) |
| [cluster_qwen3.8-27b_nvfp4_w1_yarn768k.example.json](cluster_qwen3.8-27b_nvfp4_w1_yarn768k.example.json) | the same ramp at factor ×3 (ceiling 786 432, an 827 392-token pool that seats one full-length stream, 96.4 GiB planned). Release-checked on one Spark on 2026-10-15: the needle at depth 0.5 of a 783 780-token prompt found, the 4K control found, cold prefill 3 282 s (54.7 min, 4.2 ms/token on average: the full-attention cost grows with context, 1.0 ms/token at 8K to 7 at 780K) | the validated 512K template: [cluster_qwen3.8-27b_nvfp4_w1_yarn512k.example.json](cluster_qwen3.8-27b_nvfp4_w1_yarn512k.example.json) |
| [cluster_qwen3.8-27b_nvfp4_w2.example.json](cluster_qwen3.8-27b_nvfp4_w2.example.json) | the same on two nodes | ditto |
| [cluster_qwen3.8-27b_nvfp4_w4.example.json](cluster_qwen3.8-27b_nvfp4_w4.example.json) | the same on four nodes | ditto |
| [cluster_qwen3.5-122b-a10b_nvfp4_w1.example.json](cluster_qwen3.5-122b-a10b_nvfp4_w1.example.json) | Qwen3.5-122B-A10B NVFP4 (ModelOpt: routed experts NVFP4, rest BF16) on one node, MTP draft depth 2 (the per-expert BF16 draft via `engine.mtp_expert_format: bf16`; 2.59 tok/pass at 89 % p1 / 71 % p2 on the prose probe, +16 % over depth 1 with identical transcripts) | plain decode: `--no-mtp`; depth 1: `--mtp-depth 1`; depth 3: `--mtp-depth 3` |
| [cluster_qwen3.5-122b-a10b_nvfp4_w2.example.json](cluster_qwen3.5-122b-a10b_nvfp4_w2.example.json) | the same on two nodes | ditto |
| [cluster_qwen3.5-122b-a10b_nvfp4_w4.example.json](cluster_qwen3.5-122b-a10b_nvfp4_w4.example.json) | the same on four nodes | ditto |
| [cluster_glm-5.3_nf4i8_w4.example.json](cluster_glm-5.3_nf4i8_w4.example.json) | the full GLM-5.3 in the NF4I8 GPTQ H32 g128 checkpoint (`HawkBearPig/GLM-5.3-NF4I8-GPTQ-H32-g128`: sixteen-level codebook routed experts, int8 attention and shared expert) on four nodes: MTP depth 1, eight request slots, 272K fp8 context, the embedding vocab-sharded, the BF16 decode weights in their 12-bit form (93.7 GiB of weights per rank) | T=1: `--no-mtp`; depth 2 at two slots: `--mtp-depth 2 --max-concurrency 2` |
| [cluster_glm-5.3_mixed346_w4.example.json](cluster_glm-5.3_mixed346_w4.example.json) | the full GLM-5.3 in the Mixed346 GPTQ H32 A8 g128 checkpoint (`HawkBearPig/GLM-5.3-Mixed346-GPTQ-H32-A8-g128`: 3/4/6-bit codebook routed experts with int8 activation codes, 116 int4 fallback experts, int8 attention and shared expert) on four nodes: MTP depth 1, eight request slots, 272K fp8 context, the embedding vocab-sharded, the BF16 decode weights in their 12-bit form (92.1 GiB of weights per rank; 62.7 ms a pass at one request on 2026-10-10) | T=1: `--no-mtp`; depth 2 at two slots: `--mtp-depth 2 --max-concurrency 2` |
| [cluster_mimo-v2.6-flash_mxfp4-fp8_w4.example.json](cluster_mimo-v2.6-flash_mxfp4-fp8_w4.example.json) | MiMo-V2.6-Flash as shipped on four nodes: MTP depth 1 (the first of the release's three draft layers), the BF16 o_proj / head / eh_proj resident in their 12-bit form alone (`"bf12"`), 128K context (every layer's K/V paged, the sliding-window layers read theirs through the window), four request slots; text prompts only (the vision and audio encoders are not served) | T=1: `--no-mtp`; depth 2: `--mtp-depth 2` |
| [cluster_mimo-v2.6-flash_mxfp4-fp8_w2.example.json](cluster_mimo-v2.6-flash_mxfp4-fp8_w2.example.json) | the same on two nodes: MTP depth 1, the K/V cache in the fp8 row form (`"kv_dtype": "fp8"`: 58 KiB per token per rank against 109 in bf16), 256K context (97.5 GiB of the 121.6 GiB per rank), four request slots | T=1: `--no-mtp` |
| [cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json](cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json) | DeepSeek-V4.1-Flash as shipped on four nodes: six request slots at DSpark depth 4 (30 decode rows in one batched replay, the family's 32-row cap) with the confidence-scheduled verify depth (λ 0.045), the bounded prefill, 128K context — the six-stream shape the vLLM recipe reports its aggregate at; single and dual streams measure the same as the two-slot shapes did | the two-slot depth-5 shape: `--max-concurrency 2 --mtp-depth 5`; the two-slot depth-4 shape: `--max-concurrency 2`; T=1 at four slots: `--no-mtp --max-concurrency 4` |
| [cluster_deepseek-v4-flash_mxfp4-fp8_w4.example.json](cluster_deepseek-v4-flash_mxfp4-fp8_w4.example.json) | DeepSeek-V4-Flash (the 0731 release) as shipped on four nodes: MXFP4 routed experts and FP8 dense projections consumed as they are, the DSpark block draft (five drafts) under the confidence-scheduled verify depth (`"mtp_schedule": true` with this world's measured row and base costs: a pass verifies as many drafts as their survival pays for, any depth from one to five — mostly two or three on prose, the whole block on code and JSON), sampled requests scheduled too with the draft's most likely tokens (`engine.mtp_draft` greedy for this family, `engine.mtp_schedule_sampled_scale` 0.93), six request slots, the model's full 1M-token context (63.2 GiB per rank with the 14 GiB prefix cache; the compressed caches take 6.2 GiB per rank at 1M) | T=1: `--no-mtp`; a fixed depth instead of the schedule: set `"mtp_schedule": false` and `"mtp_depth"` (3 measured best fixed on prose, 5 on code and JSON); sampled requests drafting by draw: `--mtp-draft sampled` |
| [cluster_deepseek-v4-flash_mxfp4-fp8_w2.example.json](cluster_deepseek-v4-flash_mxfp4-fp8_w2.example.json) | the same on two nodes: the scheduled DSpark depth with the two-node costs, four request slots, the full 1M-token context (95.8 GiB of the 121.6 GiB per rank with an 8 GiB prefix cache) | T=1: `--no-mtp` |
| [cluster_qwen3.8-27b_fp8_w1.example.json](cluster_qwen3.8-27b_fp8_w1.example.json) | Qwen3.8-27B-FP8 on one Spark: the dense 27B (48 Gated-DeltaNet + 16 full-attention layers) from its native block-FP8 checkpoint with the DFlash2 block drafter (`engine.dflash_model: z-lab/Qwen3.8-27B-DFlash2`, five bidirectional draft layers off five target taps, a block of eight; the block proposal recorded inside the decode graph's step, no MTP), 256K context, eight request slots, the lm head requantized to block FP8 (`dense_weights: "fp8"`), the drafter's five block matrices served as block FP8 (`dflash_weights: "fp8"`: lossy for the PROPOSALS only — the verify is exact, transcripts bitwise — at the same acceptance, 145.5 → 142.2 ms a pass), its other bf16 matrices packed lossless 12-bit (`bf16_weights: "bf12"`), the rest exact. The drafter is the faster single-node recipe on every prompt class (C1 19–44 tok/s greedy against MTP depth 2's 16–22 in the 2026-10-05 campaign; the same model, an exact verify), so the MTP world ships as this template's mode, not a second file. The verify depth is scheduled a step at a time from the selector walk's confidence (`mtp_schedule`, 2.0 ms a row over a 130 ms pass): C1 level, eight slots a few percent ahead of whole blocks | plain decode: `--no-dflash`; MTP depth 2: `--no-dflash --mtp --mtp-depth 2`; the drafter on the eager engine: `engine.decode_graph: false` in the recipe (the same transcripts, chat 95 against 71 ms a token); the checkpoint's BF16 head: `--dense-weights checkpoint`; the opt-in prefill recipe `--prefill-fp8-per-tensor`; `--dflash-depth N` verifies the first N drafts (the eager engine); `--no-prefill-group`: one cold prompt per prefill walk (bitwise prompts in a burst) |
| [cluster_qwen3.8-27b_fp8_w2.example.json](cluster_qwen3.8-27b_fp8_w2.example.json) | Qwen3.8-27B-FP8 on two Sparks (tensor parallel: the GDN and attention heads, the MLP and the lm head split across the ranks, two boundary folds a layer) with the DFlash2 block drafter on the decode graph (the block proposal recorded in the graph step, the drafter sharded across the ranks, its block matrices as block FP8 (`dflash_weights: "fp8"`, lossy for the proposals only), `bf16_weights: "bf12"`; a sampled request's drafts drawn at 0.7 of its temperature under the ratio verify, `mtp_draft_temperature`), 256K context, eight request slots, the FP8 lm head. Greedy C1 35 / 55 / 77 / 63 / 33 tok/s (prose / code / JSON / math / chat) at 82 ms a pass in the 2026-10-05 campaign against MTP depth 3's 34 / 40 / 46 / 42 / 31 at 78; the drafter is the template since 2026-10-05, when its drawn proposals also took the sampled classes; the verify depth scheduled from the selector's confidence (`mtp_schedule`, 2.0 ms a row over a 66 ms pass) | plain decode: `--no-dflash`; MTP depth 3 (the template until 2026-10-05): `--no-dflash --mtp --mtp-depth 3`; the argmax walk for sampled requests: `--mtp-draft greedy`; the checkpoint's BF16 head: `--dense-weights checkpoint` |
| [cluster_qwen3.8-27b_fp8_w4.example.json](cluster_qwen3.8-27b_fp8_w4.example.json) | Qwen3.8-27B-FP8 on four Sparks (tensor parallel: the GDN and attention heads — one kv head a rank — the MLP and the lm head split across the ranks, two boundary folds a layer) with the DFlash2 block drafter on the decode graph (as the two-node template), 256K context, eight request slots, the FP8 lm head. Greedy C1 57 / 88 / 122 / 102 / 53 tok/s (prose / code / JSON / math / chat) at 50 ms a pass in the 2026-10-05 campaign against MTP depth 3's 58 / 69 / 78 / 72 / 51 at 45: ahead on code / JSON / math by 29–56 %, level on prose, 6 % ahead on chat, and ahead on sampled traffic with the drawn proposals. The verify depth is scheduled a step at a time from the selector walk's confidence (`mtp_schedule`, 2.0 ms a row over a 34 ms pass): eight slots 205 / 301 / 352 / 310 / 216 tok/s at 92 ms a step since the one-chain split-K (184 / 273 / 326 / 284 / 196 at 106 before it; 154 / 270 / 320 / 260 / 170 at 117 for whole blocks), C1 level; transcripts matched across the tested isolated and joining-co-tenant cases | plain decode: `--no-dflash`; MTP depth 3 (the template until 2026-10-05): `--no-dflash --mtp --mtp-depth 3`; the argmax walk for sampled requests: `--mtp-draft greedy`; the checkpoint's BF16 head: `--dense-weights checkpoint` |

The full GLM-5.3 templates keep MTP depth 1 for their eight slots: at
eight concurrent requests a second verify row costs more than it returns
(fixed depth 2: 51.7 → 47.2 wall tok/s on prose, −2..−9 % by class;
the scheduled depth 2, −1..−5 %), while one request gains +4..+11 % at
fixed depth 2 and +7..+13 % under the schedule (`engine.mtp_depth: 2`,
`engine.mtp_schedule: true`, `engine.mtp_schedule_min_depth: 1`,
`engine.mtp_schedule_row_ms: 14`, `engine.mtp_schedule_base_ms: 34`;
2026-10-14, four nodes, int4-int8). An interactive deployment of the full
model should set those five keys. GLM-5.3-Flash batches its requests
past depth 1 since 2026-10-14 (the batched draft chain and the runtime
decode rows: four slots at depth 2 are one twelve-row family) and gains
+5..+13 % at one request from fixed depth 2 and +3..+12 % under the
schedule, but four requests still trail depth 1 — 91–104 wall tok/s at
fixed depth 2 and 99–105 under the schedule against 107–112 at depth 1
(NVFP4/FP8, four nodes) — so its templates keep depth 1; an interactive
Flash deployment sets `engine.mtp_depth: 2`, `engine.mtp_schedule: true`,
`engine.mtp_schedule_min_depth: 1`, `engine.mtp_schedule_row_ms: 7` and
`engine.mtp_schedule_base_ms: 17`.

The Qwen3.8-Flash-Next, GLM-5.3-Flash, full GLM-5.3 and Qwen3.8-27B
templates set `engine.draft_head: "screen"` (2026-10-13; the AutoRound
hybrid keeps the exact head): each draft token reads a 2-bit rotated copy
of the head to find its candidate blocks and only those blocks exactly, so
greedy transcripts are unchanged and the pass is shorter — Flash-Next on
one node 39.8 → 38.1 ms at depth 1 and 55.8 → 50.5 at depth 3, on four
nodes 23.2 → 22.6 and 31.3 → 29.9; GLM-5.3-Flash on four nodes 31.3 → 30.9;
the full GLM-5.3 (int4-int8) on four nodes 63.5 → 62.5; the 27B NVFP4
drafter on four nodes 38.6 → 38.0. `--draft-head exact` restores the full
read. DeepSeek-V4.1's templates keep the exact head: DSpark adds its
Markov bias after the head read, and the screen's candidate selection does
not yet see it (measured: fewer accepted drafts). DeepSeek-V4 keeps its own
screened draft head.
The Qwen NVFP4 templates use `engine.fp8_head: "mma"` after matched
real-checkpoint numerical checks
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
ignore it. `kv_dtype` selects the GLM-5.3 latent-cache format (bf16, fp8 or
fp4) or MiMo K/V format (bf16 or fp8); Qwen and GLM-4.7 stay BF16, while
DeepSeek retains its architecture-defined compressed/window formats.

## Shared scheduler defaults

Every example inherits the shared adaptive prefill policy. The examples omit
`prefill_budget_tokens` and `prefill_idle_budget_tokens`, selecting automatic
values (`-1` and `0`). Early text admission resolves from execution capabilities,
including with DFlash. The controller starts at 256 tokens, adjusted to native
alignment, then learns pass cost up to the engine's aligned capacity or 4,096
tokens, whichever is smaller. The GLM and MiMo examples therefore retain their
2,048-token ceiling; Qwen and DeepSeek retain 4,096. Removing the old explicit
256-token busy cap lets the four-node GLM Flash example adapt during decoding.

The decode-gap target defaults to 1,000 ms and the first-token target to
3,000 ms. These are soft objectives: chunk execution, long prompts, occupied
slots and KV pressure can cause misses. The shared defaults reserve 15% latency headroom, grow by at most
2× observed work and weight cheaper observations by 25% during recovery.
All are deployment-level configurables; see [budget and latency settings](../docs/operations.md).
Smaller latency targets can reduce throughput by requiring more model walks.

The examples retain their measured model/world-specific precision, draft depth,
concurrency, KV and prefix-cache settings. They bound the admission queue at
eight and HTTP connections at 64, while inheriting the 3 ms idle gathering
window. `admission: "full"` reserves prompt plus requested output;
`"grow"` uses `admission_window` and can shed a request if KV runs out.
That reservation choice is independent of adaptive prefill and early admission.

Existing local deployment files keep their explicit overrides. To adopt the
shared policy, remove their old busy/idle budgets or set them to `-1` and `0`,
then restart with the updated serving binary. Positive budgets remain caps.
For fixed budgeting, set both latency targets to zero and supply the desired
positive busy/idle budgets. Example updates do not modify a running deployment.

## Single-Spark NVIDIA Qwen NVFP4 defaults

The NVIDIA NVFP4 `w1` example uses a 262144-token shared KV pool, four request
slots, MTP depth 1 and full admission. Its automatic prefill ceiling is 4,096
tokens under the policy above. The pool permits one request to approach the
checkpoint's 262144-token total context, including output; it does not reserve
256K for each of four simultaneous requests. Cached prefixes also occupy pool
blocks. Increase KV capacity separately when concurrent contexts need more
space, using the engine's startup memory plan.

Since 2026-09-28 the templates give the prefix arena (`prefix_cache_gib`) the
memory the node has left under the memory plan's 4 GiB headroom, less a 1 GiB
margin, because a snapshot slot is only ever spent on a prompt above
`engine.prefix_min_tokens` (1024 by default) and an idle GiB buys nothing. The
one-node Qwen templates are the exception: their n-gram table is mmap'ed, and
the memory they leave free is the page cache that keeps fresh-text prefill at
full speed, so they stay at 3 GiB (27 snapshots). The
[sizing guide](../docs/prefix-cache.md) lists representative capacities and the
entry policy's keys.

## The 512K YaRN templates (`engine.rope_scaling`)

The ramp serves two families with the same rope geometry (64 rotary dims of a
256-wide head, theta 1e7, mrope sections [11, 11, 10]): Qwen3.8-Flash-Next
(`qwen4_exp`, the QSA layers' table) and, since 2026-10-12, Qwen3.8-27B
(`qwen3_5`: the full-attention layers' table, the DFlash2 draft table and the
context ceiling all follow it). The one-node 27B templates are
[cluster_qwen3.8-27b_nvfp4_w1_yarn512k.example.json](cluster_qwen3.8-27b_nvfp4_w1_yarn512k.example.json)
(release-checked: 3/3 needles at 524 288 on one Spark and on four, 100 %
prefix reuse, 74.8 GiB planned) and the factor-3
[cluster_qwen3.8-27b_nvfp4_w1_yarn768k.example.json](cluster_qwen3.8-27b_nvfp4_w1_yarn768k.example.json)
(96.4 GiB planned, release-checked at 786K: needle and control found, a 55-minute cold prefill); the one-node RadixArk Flash-Next shape is
[cluster_qwen-3.8-flash-next_nvfp4-radixark_w1_yarn512k.example.json](cluster_qwen-3.8-flash-next_nvfp4-radixark_w1_yarn512k.example.json)
(100.9 GiB planned from its config). The pool and headroom arithmetic below
is written for the two-Spark Flash-Next shape and applies to each of them
with its own plan total.

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

The template sets `kv_capacity` 565 248 — the ceiling plus the 32 768-token
`default_max_tokens` the template carries plus 8 192 tokens of room, because a
request reserves its prompt *and* its answer from the pool: a 524 288-token
prompt from a client that omits `max_tokens` needs 557 056, and a pool cut at
the ceiling would refuse the longest request it exists to serve. With
`max_concurrency` 2 that pool seats one full-length stream, or two streams of up
to ~282 000 tokens each (~249 000 each when both omit `max_tokens`), or any mix
that fits; a third request waits at the door (`queue_limit` 8) rather than
sharing bytes. For two concurrent 524 288-token streams set `--kv-capacity
1114112` (1 048 576 when both clients send their own, smaller `max_tokens`).

Memory headroom: at TP=2 the resident weights are 44.1 GiB per rank, the pool is
7.45 GiB at 565 248 tokens (14.68 GiB at 1 114 112) and the plan's activations,
the prefix arena and the engine's tables come on top — 96.8 GiB per rank for
this shape with its 40 GiB arena (56.8 GiB without it, as the plan of
2026-09-28 measures it), plus the 4 GiB headroom the pre-flight check adds
(`--memory-plan` prints the itemised total and refuses the configuration before
the first allocation if the node cannot cover it). The 1 114 112-token variant
adds about 9.3 GiB per rank, which is why the arena stops at 40 GiB rather than
filling the node. Nothing about the rope ramp changes those numbers: they move
with `kv_capacity`, `max_concurrency` and `prefix_cache_gib`.

The 40 GiB prefix arena holds 741 snapshots per rank at TP=2.
Long prompts also retain an earlier chunk snapshot so changed questions can
share the document's KV blocks. Snapshot size does not grow with context;
KV-pool capacity limits how many independent long documents remain cached.
See [prefix-cache sizing](../docs/prefix-cache.md) for all recipes and retention
estimates before increasing the arena budget.

Two things to check on the boot log, both added with this knob:

- `serve: request context limit 524288 tokens — the lesser of the 524288-token
  positional ceiling (engine.rope_scaling yarn x2 over 262144 positions) and the
  565248-token K/V pool …` — if the ceiling it names is 262144 the knob did not
  reach the process,
- `engine.rope_scaling is set but the loaded model is the <family> family` is a
  **startup error**, not a note: the ramp is built for the Qwen3.8-Flash-Next
  QSA rope and the Qwen3.8-27B full-attention rope (the same geometry) and
  no other family's, and the world refuses to start with an operator's
  setting left unapplied.

The checkpoint must be a plain one (`rope_parameters.rope_type: "default"`),
which the NVFP4 and FP8 releases are; a checkpoint that already bakes YaRN into
its own `rope_parameters` is rejected at load, for the reason
`docs/qwen38_flash_next_plan.md` §1.9.1 gives.
`scripts/qwen_yarn_release_check.py` is the opt-in, long-context acceptance run
for this template ([docs/qwen_yarn_release_check.md](../docs/qwen_yarn_release_check.md)).

## The consolidation (2026-09-14)

The 2026-09-14 consolidation reduced the then-existing twenty-five templates
to eight; the subsequent two-Spark NVFP4 recipe brought that set to nine. More
families and quantizations have since been added to the table above. Every `_plain` variant
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
