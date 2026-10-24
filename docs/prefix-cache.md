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

## The entry floor and the head cut

Two policy keys (2026-09-28) decide which cuts become entries at all.
Both are world settings: rank 0's values reach every peer, and the
journal's decision digest checks that every rank made the same choices.

`engine.prefix_min_tokens` (`--prefix-min-tokens`, default 1024) is the
floor below which no snapshot is taken: not the prefill cut, not the head
or body cut, not a rolling or close entry. A prompt shorter than the floor
still attaches to any entry that matches; it just never takes a slot.
Every entry costs one slot whatever its position, and LRU eviction cannot
tell a probe's entry from a long conversation's. A single-Spark field log
replayed under a 13-slot arena lost every turn of a 66K-token conversation
because three 45-token requests followed each turn; with the floor the
same 13 slots served the whole log with two misses. Set the floor to 0 to
restore the old behavior.

`engine.prefix_head_snapshots` (`--prefix-head-snapshots`, default on)
adds one entry to a cold prefill at the prompt's first structural boundary
past its start, the aligned image of the first role marker after the
opening one, which is where a system prompt ends. The next conversation
under the same system prompt attaches there, and so does the turn after an
agent client compacted its history, which shares the system prompt and the
tool schemas with the conversation it replaced but nothing after them. In
the same field log those post-compaction turns prefilled 18K shared tokens
cold 24 times in 15 hours. The head cut obeys the floor, takes its slot
after the deepest cut and before the body cut, and is off on a family
whose prefill cannot take extra cuts (DeepSeek's bounded prefill).
`/v1/metrics` reports `min_tokens`, `head_cuts` and the `head_snapshots`
taken under `prefix_cache`.

## Current recipe capacities

The table covers selected checked-in recipes. KV pools and arena budgets
reflect their current JSON settings; snapshot sizes come from the 2026-09-28
memory-plan measurements, with slot counts adjusted for the current budgets.
It is not a new performance run. Snapshot size is independent of context length
for these model families. A longer document
uses more **KV blocks**, not a larger snapshot slot.

| Recipe | Concurrent requests | KV token pool | Budget, GiB | MiB per snapshot | Slots |
|---|---:|---:|---:|---:|---:|
| Qwen NVFP4, one node | 4 | 262,144 | 3 | 110.317 | 27 |
| Qwen NVFP4, two nodes | 4 | 262,144 | 58 | 55.263 | 1074 |
| Qwen FP8, two nodes | 4 | 262,144 | 6 | 55.263 | 111 |
| Qwen NVFP4 YaRN, two nodes | 2 | 565,248 | 40 | 55.263 | 741 |
| Qwen FP8, four nodes | 4 | 1,052,672 | 40 | 27.735 | 1476 |
| GLM-5.3-Flash, two nodes | 4 | 163,840 | 4.5 | 70.422 | 65 |
| GLM-5.3-Flash, four nodes | 4 | 786,432 | 22 | 35.227 | 639 |
| GLM-4.7, four nodes | 4 | 262,144 | 1.5 | 0.010 | 4096 |
| GLM-5.3 full, four nodes | 8 | 122,880 | 1 | 0.012 | 4096 |
| DeepSeek-V4.1-Flash, four nodes | 6 | 131,072 | 14 | 3.476 | 4096 |

GLM-4.7, full GLM-5.3 and DeepSeek reach the 4096-slot limit and allocate
about 40 MiB, 48 MiB and 13.9 GiB respectively.
The one-node Qwen memory plan includes MTP state when graph decode is enabled.

Since 2026-09-28 the recipes size the arena to what the node has left: the
memory plan's total plus its 4 GiB headroom, measured against the 115 GiB an
idle Spark reports free, minus a 1 GiB margin, rounded down to whole GiB (the
GLM-5.3-Flash two-node recipe, already within 4 GiB of the node, gains 3;
the YaRN recipe stops at 40 so its documented 1,114,112-token variant still
fits).
The one-node Qwen recipes are the exception on purpose: their 47.7 GiB n-gram
table is mmap'ed from the checkpoint and served through the page cache, and a
field log showed fresh text prefilling three to four times slower than text
the server had seen, so the memory not given to the arena is what keeps that
cache warm. Measure before assuming that: the five-agent deployment above
moved its arena from 6 to 12 GiB, which cut the page cache from 16.4–16.9 to
10.5–11.0 GiB and cost 4–7 % idle prefill, while loaded prefill, decode,
reclaim (8–35 MiB/s), major faults (2–8/s) and memory-stall time were
unchanged — the table fits in neither — and the arena, not the page cache,
was that workload's binding limit. Compare idle prefill and major faults
(`pgmajfault` in `/proc/vmstat`) before and after moving memory to the arena. The MiMo and RadixArk recipes could not be re-planned here (no
checkpoint on the fabric); the RadixArk two-node budget is an estimate from the
NVIDIA checkpoint's plan and the one-node one follows the page-cache rule.
Concurrency alone does not specify how much historical cache to retain: with
the entry floor a conversation costs about two slots per turn, and an arena
holds that many turns of history across every conversation it serves.
An agent client's turn is many calls (a field log: 1–34, median 4), and each
call leaves about three entries — the deepest cut, a body cut and the close
entry at retire — so an idle session survives about slots ÷ 3 other calls
(eviction is LRU with no time expiry). In that deployment (one Spark,
Qwen3.8-Flash-Next NVFP4, five coding agents, 2026-10-21) 55 slots kept a
session paused five minutes and lost one paused ten; 111 slots held ten
minutes (34–39 other calls) and not twenty (72). Size such an arena as three
times the other calls an idle session must survive, plus the active requests;
the document working-set estimate below (about 20 slots for that load)
undercounts agent turns.

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

MTP hop snapshots are captured before the exact sampled fallback runs. When
that fallback accepts additional tokens, it can pass an aligned position
whose snapshot was never taken. The engine reports the capture outcome
explicitly: the scheduler retains an unchanged older snapshot at its original
position, or keeps the slot empty if there was no previous snapshot. It does
not count either case as a new hop snapshot.

`prefix_cache.skipped_hop_fallback` counts those skipped captures;
`skipped_hop_mismatch` counts inconsistent outcomes or stored positions that
caused an optional snapshot to be dropped, with a warning. Prometheus exposes
them as `dgpp_prefix_cache_skipped_hop_fallback_total` and
`dgpp_prefix_cache_skipped_hop_mismatch_total`. A fallback skip is distinct
from `skipped_no_block`, which records pool pressure. If a hop replacement
runs out of pool blocks, the previous snapshot has also been released and
cannot become a close entry. Model-position checks and non-pool engine errors
still propagate.

Resolve the site configuration before running a recipe's plan:

```bash
python3 scripts/dgpp-cluster resolve --config deploy/cluster_qwen-3.8-flash-next_nvfp4_w2_yarn512k.example.json > /tmp/qwen-resolved.json
build-release/dgpp-serve --config /tmp/qwen-resolved.json --rank 0 --memory-plan
```
