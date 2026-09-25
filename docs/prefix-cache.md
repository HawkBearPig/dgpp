# Prefix reuse and memory sizing

`engine.prefix_cache_gib` reserves memory **per rank** for saved model state.
`engine.kv_capacity` sizes the separate KV token pool. Increasing the snapshot
budget helps only when snapshot slots are the limiting resource. Cached
documents and live requests still need room in the KV pool.
Entries evicted from memory are lost unless the [NVMe cold tier](#the-nvme-cold-tier)
below is enabled ([enhancement #26](https://github.com/HawkBearPig/dgpp/issues/26)).

## Changed questions after a long document

For prompts at least four prefill chunks long, the cache keeps one earlier
regular chunk snapshot as well as the final reusable cut. With 2048-token
chunks, the earlier cut leaves 2048–4095 tokens to prefill. A new question
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

## The NVMe cold tier

`nvme_cache` (a top-level deployment key; see the [README](../README.md#nvme-cache))
keeps entries the arena evicts on local NVMe and restores them for a later
matching request. Each rank owns one preallocated slab file,
`<path>/rank<N>.slab`, of `capacity_gib` GiB, holding its own shard of the
state. An entry on disk is complete: the snapshot blob (recurrent, ring and
draft state) and every KV block its metadata pins, so a restored entry needs
nothing recomputed. Restores read into fresh pool blocks and an arena slot,
so a restored prefix must still fit the memory pool; the tier expands the
retained history, not the active context.

**What a token costs.** The slab is pages of one KV block's bytes (rounded
to 4 KiB); a snapshot blob spans a run of pages. Startup logs the page size,
the page count and the blob's pages. Per rank, at the checked-in recipes:

| Recipe | KV bytes per token | 260K-token entry | Snapshot blob |
|---|---:|---:|---:|
| Qwen NVFP4 YaRN, two nodes | 13.8 KiB | 3.4 GiB | 55 MiB |
| GLM-5.3-Flash, four nodes | 12.4 KiB | 3.1 GiB | 35 MiB |

On a Spark's NVMe (direct I/O, measured 2026-09-25: 4.5 GB/s writes,
6.9 GB/s reads) a 260K-token Qwen entry spills in about a second and
restores in under one, against 216 s to recompute it. Measured on four
Sparks (GLM-5.3-Flash, a 25,150-token document, 2026-09-25): the cold
prefill took 18.8 s, the changed question over the in-memory document
2.45 s, and after three other documents had pushed it out of a four-slot
arena the same question came back from disk in 407 ms to first token
(restore 121 ms for 341 MiB), with the transcript identical to the
in-memory hit; see the
[validation record](../benchmarks/results/2026-09-25-nvme-cold-tier/README.md).

**Which entries.** Prefill snapshots (the deepest cut and the earlier
document cut) and the close-time entries a completed answer leaves spill
right after they are taken, once they are at least `min_tokens` long
(default: one prefill chunk). A spill or restore advances two 64 MiB
slices per scheduler tick — the device side of every slice is enqueued on
the model stream between decode steps, never on a second stream beside
the fabric's collectives — and a file worker moves the slices to and
from the slab. Between requests that is the NVMe's rate; during decode
about 3 GB/s (a 25K-token GLM entry, 340 MiB, spills in 0.4 s); during a
long prefill, whose ticks are one chunk long, a spill paces at its ticks
(the same entry took 16 s beside a 25K-token prefill). Measured 2026-09-25
on four Sparks, the decode step held 34 ms with spills in flight. Blocks that two entries
share in memory — a document and the questions attached to it — are stored
once, by physical identity. A memory entry stays attachable while its copy
is written; eviction from memory then keeps the disk copy. Retention on
disk is least-recently-used within the capacity, and the slab never grows
past it: a spill that cannot find room after evicting every unused entry
is skipped, not squeezed in.

**Restores.** Lookup uses the same cuts and lookahead rules as memory. When
the deepest usable state is on disk, the scheduler restores it (one restore
at a time; a second request wanting the same entry waits on the same one),
then attaches and prefills the suffix. Every page carries a CRC-32C; a
short read, a device error or a checksum mismatch fails the restore, and
the request prefills without it.

**Ranks.** Spills and restores are scheduler decisions, identical on every
rank; their outcomes are journaled: a peer reports each finished op to
rank 0 over the journal's return path, and rank 0 journals the world's
verdict once every rank has reported — ok only when all succeeded. A
failure on one rank is therefore the same cold miss everywhere, and the
prefix digest the journal checks every tick covers the tier's decisions.

**Restarts.** The slab is recreated at every start; entries are the
process's and are not reused across restarts.

**Sizing and checks.** `capacity_gib` must hold at least one entry at the
request context limit plus its blob (startup prints the minimum) and must
fit the filesystem's free space beside a 1 GiB reserve; both are checked on
every rank before anything is allocated, and by `--memory-plan`. The tier's
staging buffers (264 MiB per rank) are part of the memory plan. Watch
`nvme_cache` in `/v1/metrics`: `pages_used` against `pages_total`,
`spilled`, `restored`, `restore_failed`, `spill_skipped` (no room), and
`evictions` (disk retention at work).
