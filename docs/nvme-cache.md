# The NVMe cold tier

The prefix cache keeps a prompt's computed state in a snapshot arena on the
GPU, so a later request with the same prefix attaches to it instead of
recomputing it. The arena is small, and a few large documents push one
another out. The cold tier gives evicted entries a second home on each
node's local NVMe and brings one back before a request that matches it
prefills. It extends how much history a deployment remembers, not how much
context a request may use: a restored prefix still has to fit the KV pool.

Enable it with a top-level `nvme_cache` block in the deployment:

```json
"nvme_cache": {
  "enabled": true,
  "path": "~/dgpp/nvme-cache",
  "capacity_gib": 64
}
```

The [configuration reference](configuration.md#nvme-cache) lists every key.
Every long-context template in `deploy/` carries the block, off.

## Storage placement and containers

Startup and `--memory-plan` report the absolute resolved cache directory,
the existing directory probed, the containing mountpoint and the filesystem
type on each rank. Before creating the directory or allocating a slab, the
preflight checks placement, writability and capacity. A missing cache
directory is checked through its nearest existing ancestor. `--memory-plan`
does not create directories or probe files, and does not test direct I/O;
the slab opener checks that separately during normal startup.

In a container, `nvme_cache.path` is a path **inside the container**. Mount
the intended host storage there. Without that mount, cache files can consume
the container's writable layer and disappear when the container is removed,
even if restarting the same container preserved them. For example, after
creating `/mnt/local-nvme/dgpp-cache` on the intended host filesystem and
giving the container's serving user write access, add this to `docker run`:

```sh
--mount type=bind,src=/mnt/local-nvme/dgpp-cache,dst=/var/lib/dgpp-cache
```

Set the deployment's cache block to:

```json
"nvme_cache": {
  "enabled": true,
  "path": "/var/lib/dgpp-cache",
  "required_mount": "/var/lib/dgpp-cache",
  "capacity_gib": 64
}
```

Docker's `--mount` rejects a missing host source by default; `-v` can create
it on the fallback filesystem. See [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/).
A named Docker volume also works if its actual host backing is appropriate;
the volume name alone does not establish local NVMe storage. Configure the
mount on every rank's node and keep the same mount when replacing containers.
Cache adoption still requires a compatible [binding](#across-restarts).

`required_mount` is optional. When set, it names the exact containing
mountpoint after resolving existing symlinks. A cache subdirectory under
that mount is allowed. An ordinary directory at the expected mountpoint,
a symlink escaping it, or a nested mount with a different identity is
refused. Bind mounts are recognized even when their device number matches
the parent filesystem. An unavailable mount identity also refuses an explicit
assertion. Without an assertion, unavailable mount metadata is reported as a
warning; filesystem checks still run. Placement policy is local to each rank
and does not change the cache binding or numerical settings shared by rank 0.

Known RAM-backed filesystems (tmpfs, ramfs and hugetlbfs) are refused by
default, even on kernels that accept `O_DIRECT` there. The existing
`allow_buffered` option permits them for tests and experiments, with a warning.
OverlayFS, aufs and fuse-overlayfs produce a warning rather than a default
refusal: they can indicate a container writable layer but do not prove one.
Neither filesystem detection nor a mount assertion proves physical NVMe
backing, persistence, or that a host-side disk mount is present behind a bind
mount. Check the host storage too. Filesystems layered over RAM devices or
other hidden backing are outside this check's detection guarantees.

The slab opener rechecks placement before opening the cache and checks the
opened descriptor before truncation or preallocation. Placement refusals
preserve existing local cache files. This is a startup check, not a monitor
for mount changes during serving or a transaction across ranks.

## What it keeps and what it pays

An entry on disk is complete: the snapshot blob with the recurrent, ring and
draft state, and every KV block the snapshot pins. Nothing is recomputed on
the way back, and a restored entry answers bitwise like the in-memory hit.
Blocks a document shares with the questions attached to it are stored once.

Per rank at the checked-in recipes a token costs 12 to 14 KiB of slab, so a
260K-token entry takes about 3 GiB and a Spark's NVMe writes or reads it in
about a second. Measured against the cold prefill of the same document:

| World | Document | Cold prefill | Entry from disk |
|---|---:|---:|---:|
| GLM-5.3-Flash, four nodes | 25K tokens | 15 s | 340 MiB in about 120 ms |
| DeepSeek-V4.1-Flash, four nodes | 23K tokens | 9 s | 27 MiB in about 40 ms |
| Qwen3.8-27B with the DFlash2 drafter, one Spark | 31K tokens | 16 s | 1.75 GiB in about 0.5 s |

## How entries move

Every snapshot a prefill or a finished answer leaves is spilled as soon as it
is taken, if it is at least `min_tokens` long. The scheduler decides spills
and restores. The engine moves the bytes in slices of `chunk_mib`, two per
scheduler tick, enqueued on the model stream between decode steps so the
tier never competes with the fabric's collectives, while a file worker moves
each slice to or from the slab with direct I/O. Between requests a spill runs
at the NVMe's rate. Beside a long prefill it paces at the prefill's ticks,
and a decode in flight does not slow measurably at any slice size.

Retention on disk is least-recently-used within `capacity_gib`. The slab
never grows past it: a spill that finds no room after evicting every unused
entry is skipped. A memory entry stays attachable while its copy is written,
an entry being written or read is never evicted, and at most `max_ops`
operations are in flight.

Cancellation can retain an existing committed answer snapshot in memory,
but does not start a new spill for it. Its blocks remain evictable when
live requests need capacity; a disk write would pin them until completion.
Earlier prefill entries retain their ordinary disk-cache behavior.

A restore starts when a queued request's deepest usable state is on disk. It
reads into fresh pool blocks and a free arena slot, then the request attaches
and prefills only its suffix. One restore runs at a time, and a second
request wanting the same entry waits on it. If every arena slot is busy with
a spill, the request waits for a commit to free one rather than prefilling
cold.

## One decision on every rank

Each rank holds its own shard of the state, but the index of what is on disk
is scheduler state, identical everywhere. Spills and restores start as
decisions, and their outcomes are journaled: each rank reports a finished
operation to rank 0, and rank 0 journals the world's verdict once every rank
has reported, successful only when all succeeded. Every page carries a
checksum. A short read, a device error or a checksum mismatch on any rank is
the same cold miss on every rank, and the request prefills without the
entry. The operation-stream digest the journal checks every tick covers
these decisions.

## Across restarts

With `persist` (the default) the slab outlives the process. Its index and
checksums live beside it under a directory named by the automatic binding:
the build version (including its commit), the checkpoint path, the effective
settings and the slab geometry. By default, a build or settings change
therefore starts fresh. The previous binding's files remain on disk.

To keep using an existing cache across upgrades, set `nvme_cache.reuse_binding`
to its 16-character binding from the previous startup log (`binding ...`) or
its directory name under `nvme_cache.path`:

```json
"nvme_cache": {
  "enabled": true,
  "path": "~/dgpp/nvme-cache",
  "capacity_gib": 64,
  "persist": true,
  "reuse_binding": "0123456789abcdef"
}
```

Replace the example binding with the one belonging to this deployment and
keep its existing capacity. The server selects that directory and its stored
binding directly; it does not rename the cache, rewrite its identity, or
require a new cache format. Keep this setting on subsequent upgrades. Remove
it to return to automatic selection. Rank 0 sends the selection to every
rank; that binding must exist on each node under its local cache path.

Explicit reuse overrides the automatic build, checkpoint-path and settings
identity. It is the operator's assertion that the same model artifacts and
cached-state interpretation remain compatible. The old index does not store
the original inputs to its binding, so matching byte counts cannot establish
that assertion. Changes to weights, RoPE, quantization, tensor-parallel layout
or state layout can make a cache incompatible even at the same size.

The file formats, geometry, page checksums, exclusive slab lock and agreement
across ranks remain enforced. A missing binding, incompatible index or
geometry, or disagreement across ranks refuses startup without clearing the
selected cache. Existing checks still prune stale or unreadable entries;
successfully adopted caches resume normal writes and retention. `--memory-plan`
checks the selected files read-only; actual snapshot geometry and entry
checksums are checked when the engine opens the tier.

A start under the same binding validates every entry of the previous index
before offering it. An entry's pages must still carry the checksums its
spill recorded, and its token ids must read back. An entry whose pages a
later run reused before a crash, or whose ids are unreadable, is pruned. The
ranks then compare their candidates at the warm handshake and adopt only
when every rank holds the same one. Otherwise automatic selection starts
fresh and explicit reuse refuses startup, so
the world never runs with an index that differs by rank. Adopted entries
keep their pins, rank older than anything the new run makes, and restore
exactly as before. One world holds a slab, and a second process under the
same binding is refused at startup.

## Pins

A request's `prompt_cache_key` pins the entries it creates or attaches to,
in memory and on disk. Retention takes unpinned entries first, so an agent
session's documents survive a burst of other work. Pins protect while the
pinned entries hold at most `engine.prefix_pin_max_pct` of the arena's slots
and of the slab's pages. Past that share plain least-recently-used order
applies, so one client's keys cannot push everyone else out. The key never
changes matching, which stays by exact token prefix.

## Bounds and failure modes

- A filesystem that refuses direct I/O is refused at startup unless
  `allow_buffered` says otherwise, because buffered writes grow the page
  cache into memory the model plans on.
- An operation that makes no progress for `op_timeout_s` fails. A request
  waiting on that restore prefills cold, and the buffer the stuck call holds
  returns when the call does. While the file worker is inside a stuck call,
  a new restore is refused at once and its request prefills cold, and a
  queued spill fails at once with its entry still in memory.
- Startup names any other binding's slabs it finds in the directory, with
  their size. They belong to no running world and can be removed.

## Sizing, metrics and tuning

`capacity_gib` must hold one entry at the request context limit plus its
snapshot blob, and must fit the filesystem's free space beside
`reserve_gib`. Startup checks both on every rank, and `dgpp-serve
--memory-plan` runs the same check. The staging buffers, four times
`chunk_mib` plus 8 MiB, are part of the memory plan.

`/v1/metrics` reports `nvme_cache`: `pages_used` against `pages_total`,
`spilled` and `restored`, `restore_failed`, `spill_skipped` for entries that
found no room, `evictions` for retention at work, `pinned` and `adopted`.
Among the knobs, `chunk_mib` paces an operation against the decode step,
`max_ops` bounds the entries held busy, `op_timeout_s` bounds a stall and
`min_tokens` keeps short entries memory-only.
