# The NVMe cold tier for the prefix cache

Date: 2026-09-25
Issue: [#26](https://github.com/HawkBearPig/dgpp/issues/26)
Branch: `nvme-cold-tier` on master `ed0eba8`.

## What was built

Entries the snapshot arena evicts — the snapshot blob and every KV block
the entry pins — go to one preallocated slab file per rank on local NVMe
and come back before a later matching request prefills its suffix. The
index is scheduler state (identical on every rank); the bytes are each
rank's shard. Blocks shared in memory by a document and its attached
variants are stored once (by physical identity: block id + contents
generation); retention is LRU within `nvme_cache.capacity_gib`; every page
carries a CRC-32C. Spills and restores start as scheduler decisions and
their outcomes are journaled: the peers report each finished op to rank 0
over a new return path on the journal socket, and the tick record's `dk`
field carries the world's verdict (ok only when every rank succeeded),
applied before that tick everywhere. The device side of every slice runs
on the model stream from the engine thread (two 64 MiB staging slices per
tick); a file worker does the direct I/O. Configuration: the top-level
`nvme_cache` deployment block; startup checks the directory, the free space
beside a 1 GiB reserve and the minimum capacity (one entry at the request
context limit plus its blob) on every rank, `--memory-plan` too.

## The fabric numbers

### The NVMe itself (rank 0, direct I/O, `dd`, production world up)

| | |
|---|---:|
| sequential write, 1 GiB | 4.5 GB/s |
| sequential read, 1 GiB | 6.9 GB/s |
| PCIe link | Gen4 x4 |

### Four Sparks, GLM-5.3-Flash NVFP4/FP8 (`check-glm-w4.json`)

`deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json` booted with
`--knobs '--nvme-cache-path ~/dgpp/nvme-cache --nvme-cache-gib 32 --prefix-cache-gib 0.15'`
(a four-slot arena so eviction is quick; slab 32 GiB per rank = 21,129 pages of
1,626,112 bytes, 23 pages per snapshot blob, minimum 9.34 GiB at the 786,432-token
context limit). `scripts/serve_nvme_cache_check.py --records 700 --others 3`,
greedy, 48 answer tokens, TTFT to the first streamed delta:

| Step | Prompt tokens | Cached | TTFT |
|---|---:|---:|---:|
| 1. document A + Q0, cold | 25,150 | 0 | 18,784 ms |
| 2. document A + Q1, changed question (memory, the body cut) | 25,150 | 22,528 | 2,450 ms |
| 3. three other documents (each evicts from the four-slot arena) | 25,147–25,165 | 0 | 18,762–18,890 ms |
| 4. document A + Q1 again — a memory miss, **restored from disk** | 25,150 | 25,148 | **407 ms** |
| 5. document A + Q0 again (memory) | 25,150 | 25,148 | 257 ms |

The transcripts of steps 2 and 4 are identical. Counters at the end: 14 spills
committed, 2 restores committed, 0 spill or restore failures, 80 blocks shared,
2,916 of 21,129 pages in use, 4.42 GiB written, 0.67 GiB read. Per-op times
from the rank logs: restores 121 ms and 137 ms for 341 MiB (197 blocks + the
blob); spills 95–460 ms when the world was decoding or idle, and 1.5–17 s for
the spills that overlapped a 25K-token cold prefill (a prefill tick is one
2048-token chunk, and a spill advances two slices per tick — the entry stays
usable in memory meanwhile). The decode step held 34.1–34.8 ms/step with
spills in flight (the production line is 33.9). The four ranks' op streams
were identical (`down`'s md5s), so every spill, restore and commit was the
same decision on every rank.

### One Spark, Qwen3.8-Flash-Next NVFP4 (`check-qwen-w1.json`)

`deploy/cluster_qwen-3.8-flash-next_nvfp4_w1.json` with
`--nvme-cache-gib 16 --prefix-cache-gib 0.3` (two arena slots), `--records 200`:
the restore of the 9,040-token entry took 134 ms (349 MiB); the changed
question came back at 4.7 s (its body cut) and the original question at
231 ms from the restored entry; transcripts identical; 615 blocks shared
across the variants; no failures.

## The stall that was found and fixed

The first four-node run stalled at the first decode step after the
spilled prefill: on two ranks the spill's device work took exactly the
graph's 30 s stage-gate timeout and those ranks died on the gate. The
tier's first design ran its gather and copies on a second CUDA stream
from a worker thread, beside the fabric's device-side spin loops (the
stage gate, the graph collectives); the single-Spark world, which has no
such loops, passed the same flow. The tier now enqueues every byte of
device work on the model stream from the engine thread at a tick's top,
two staging slices at a time, and the worker touches only pinned memory
and the file. Runtime CUDA object creation was removed at the same time
(the events come from a ring made at boot), per the fabric's allocation
discipline.

## Gates

- `unit_tests`: `disk_cache_*` (the index: allocation, sharing, LRU,
  aborts) and `cluster_config_parses_nvme_cache`.
- `scheduler_test`: seven `scheduler_nvmeCache_*` cases (spill / evict /
  restore / attach, failed restore and spill, retention, two requests on
  one restore, a cancel while restoring, identical streams and digests);
  the 66 earlier cases unchanged.
- `fabric_serve_test`: the tick record's `dk`, the status-line codec, the
  commit collector, the loopback return path, and ok 10: spill, evict,
  restore and attach across three ranks over real sockets with identical
  op streams.
- `qwen_decode_test`: the tier block on the fixture (spill and restore
  bitwise — blob and every block plane — attach and decode parity with the
  in-memory entry, a corrupted page failing the restore, every block
  released).
- Python: `portability_test`'s resolver schema case; `site_env_test`'s
  resolved-fixture case updated for the new block.
- Startup checks exercised on the two-node YaRN recipe through
  `--memory-plan`: a 1 GiB capacity refused below the 6.96 GiB minimum,
  4000 GiB refused against 148 GiB free, 16 GiB accepted.

## Reproduce

```bash
scripts/dgpp-cluster up --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json \
    --knobs '--nvme-cache-path ~/dgpp/nvme-cache --nvme-cache-gib 32 --prefix-cache-gib 0.15'
python3 scripts/serve_nvme_cache_check.py --records 700 --others 3 --out check.json
scripts/dgpp-cluster down --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json
```
