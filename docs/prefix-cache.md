# Prefix reuse and memory sizing

`engine.prefix_cache_gib` reserves memory **per rank** for saved model state.
`engine.kv_capacity` sizes the separate KV token pool. Increasing the snapshot
budget helps only when snapshot slots are the limiting resource. Cached
documents and live requests still need room in the KV pool.
Evicted entries are not spilled to disk; NVMe-backed retention is tracked
separately in [enhancement #26](https://github.com/HawkBearPig/dgpp/issues/26).

## Changed questions after a long document

For prompts at least four prefill chunks long, the cache keeps one earlier
regular chunk snapshot as well as the final reusable cut. With 2048-token
chunks, the earlier cut leaves 2048–4095 tokens to prefill (4096–8191 on
Qwen, whose chunks are 4096 tokens). A new question
after an unchanged document can attach there; an identical repeat can still
use the deeper cut. This requires an exact token-prefix match, a valid cut
in the new prompt, and, with MTP, the same token immediately after the cut.
Changes earlier in the document can still miss. The policy does not discover
semantic document boundaries or cache every chunk.

The extra snapshot pins existing complete KV blocks by reference. Later
questions share those blocks and allocate their own suffix/answer blocks.
Short prompts keep the original policy; when slots are scarce the final
snapshot has priority. DeepSeek's bounded prefill keeps its original policy
because an additional snapshot would execute another decoder span. Exact
DeepSeek prefill supports the additional cut.

## Current recipe capacities

The following are per-rank memory-plan results for the checked-in recipes,
with their configured MTP settings, measured on 2026-09-21. Snapshot size is
independent of context length for these model families. A longer document
uses more **KV blocks**, not a larger snapshot slot.

| Recipe | Concurrent requests | KV token pool | Budget, GiB | MiB per snapshot | Slots |
|---|---:|---:|---:|---:|---:|
| Qwen NVFP4, one node | 4 | 65,536 | 1.5 | 110.317 | 13 |
| Qwen NVFP4 or FP8, two nodes | 4 | 262,144 | 1.5 | 55.263 | 27 |
| Qwen NVFP4 YaRN, two nodes | 2 | 532,480 | 1.5 | 55.263 | 27 |
| Qwen FP8, four nodes | 4 | 262,144 | 1.5 | 27.735 | 55 |
| GLM-5.3-Flash, two nodes | 4 | 163,840 | 1.5 | 70.422 | 21 |
| GLM-5.3-Flash, four nodes | 4 | 786,432 | 8 | 35.227 | 232 |
| GLM-4.7, four nodes | 4 | 262,144 | 1.5 | 0.010 | 4096 |
| GLM-5.3 full, four nodes | 8 | 122,880 | 1 | 0.012 | 4096 |
| DeepSeek-V4.1-Flash, four nodes | 6 | 131,072 | 1.5 | 3.476 | 441 |

GLM-4.7 and full GLM-5.3 reach the 4096-slot limit and allocate only about
40 and 48 MiB respectively, despite their larger configured ceilings.
The one-node Qwen memory plan includes MTP state when graph decode is enabled.

The existing budgets remain the defaults. Production GLM's 232 slots have
ample room for four active conversations. Qwen's two-node 27-slot arena
can retain a shared document and several question variants. Single-node
Qwen's 13 slots provide the least retention headroom: increase its budget
if the required history exceeds those slots and the memory plan still fits.
Concurrency alone does not specify how much historical cache to retain.

## Size for the working set

Compute slots as `min(4096, floor(prefix_cache_gib * 2^30 / snapshot_bytes))`.
The server logs the resulting slot count and actual allocated bytes during
startup and `--memory-plan`.

For a rough retention budget, allow one shared-document snapshot and up to
two entries per distinct question (final prompt and completed answer), plus
temporary slots for active prefills and rolling snapshots. For `D` documents,
`V` retained questions per document and `C` active requests, a conservative
starting estimate is `D * (1 + 2 * V) + 2 * C` slots. This is a working-set
estimate, not a minimum needed to serve requests: eviction and skipped
snapshots preserve serving when fewer slots are available. Alignment, shared
conversation prefixes, cancellation and response lengths change actual use.
The estimate assumes questions retain the same earlier document cut;
substantially different prompt lengths can retain additional body cuts.

For example, one document with five retained question/answer variants and
two active requests suggests 15 slots. On two-node Qwen that is about
0.81 GiB per rank, within the current 1.5 GiB. Four independent documents
with one retained question each and four active requests suggest 20 slots;
one-node Qwen would need about 2.16 GiB, so a 2.25 GiB budget is a reasonable
starting point **if that retention is required and its KV pool also fits**.

KV capacity must cover the union of distinct cached prefixes, private
suffix/answer blocks, live cold prompts, and growth headroom. Complete blocks
are shared; partial blocks may need copies. Five unrelated 261K-token
documents cannot all remain in a 532,480-token pool, regardless of the number
of snapshot slots. Five questions over the same document can share most of
their KV storage after an earlier snapshot has been retained.

Inspect `/v1/metrics` and the boot/retire logs before changing a budget:

- Snapshot entries at the slot limit, evictions and skipped snapshots can
  indicate arena pressure. Increase `prefix_cache_gib` only with enough memory
  left for the model, KV pool, activations and startup headroom.
- A nearly full `scheduler.pool_blocks_in_use` versus `pool_blocks_total`,
  with few snapshot entries, indicates KV pressure. More arena memory alone
  does not help. `prefix_blocks_pinned` counts references across entries and
  can count a shared block more than once; use the pool counters for physical
  occupancy.
- A miss naming an early token divergence needs a stable prompt prefix.
  A miss naming a changed MTP lookahead token correctly rejects stale draft
  state. Neither is resolved by a larger arena.

Resolve the site configuration before running a recipe's plan:

```bash
python3 scripts/dgpp-cluster resolve --config deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json > /tmp/qwen-resolved.json
build-release/dgpp-serve --config /tmp/qwen-resolved.json --rank 0 --memory-plan
```

See the [implementation and validation record](../benchmarks/results/2026-09-21-prefix-document-reuse.md)
for the measured reuse, capacity and correctness checks.
