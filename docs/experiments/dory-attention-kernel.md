# Dory attention kernel on GB10: what moves the deep-prefill wall

Context: Dory1.0-4B-NVFP4 prefill at 8K+ context is ~80% `attn_partial_kernel`
(`src/kernels/dory_attn.cu`) and costs ~2.1-2.3 ms/token, which is what makes
the adaptive-prefill chunk ticks starve decoding requests
(see `docs/continuous-batching.md` for the scheduler side).
Everything below was measured on the GB10 Spark box itself (sm_121: 48 SMs,
100 KB shared memory per SM, 99 KB opt-in per block, 24 MB L2, ~234 GB/s
streaming DRAM reads).

## The machine's ceilings (measured)

- contiguous uint4 grid-stride reads: **234 GB/s** (8 GiB working set)
- 512 B reads from every 2 KB page, grid-stride: **236 GB/s** — the paged-KV
  access pattern itself costs nothing on this hardware
- contiguous reads of a 21 MB (L2-resident) buffer: **1061-1194 GB/s**
- an L2 persisting access-policy window over that buffer: **no change** (already resident)

## What the kernel does, and why

Deep chunk (1024 rows at positions 9216+, full attention, `block_tokens=64`):
130-133 ms/launch, 10.2 GB of per-(row,key) traffic = **78 GB/s**. The same
staging loads with the math stripped: **94 ms (108 GB/s)**. The compute phases
alone (smem-resident tile, same instruction mix): **29 ms**. So the kernel is
a ~1.4x-latency + ~3x-memory story, and the memory path itself is the wall.

Little's law fits: 2 resident blocks/SM x 128 threads x 16 in-flight 16 B
loads = 64 KB in flight per SM; at ~600 ns DRAM latency that is ~110 GB/s,
which is exactly the stage-only ceiling. The committed kernel (37.6 KB smem,
2 blocks/SM) is at a local optimum: any change that raises in-flight bytes
also crosses the 50 KB-per-block line (1 block/SM) and loses.

## Variants tried (all bitwise-identical outputs; microbench A/B vs the lib kernel)

| variant | ms | verdict |
| --- | --- | --- |
| committed kernel (baseline) | 133 | - |
| uint2/float4 vectorized K score reads, 260-word rows | 133 | neutral, reverted |
| `n_split` 2-16 on prefill rows | 129-133 | neutral (split blocks are not wave-synchronized; also breaks chunk-invariance numerics) |
| `cp.async.cg` double-buffered K/V pipeline | 146-178 | worse (GB10 cp.async MLP is weak; 71 KB smem drops to 1 block/SM) |
| same with `cp.async.ca` | 146-189 | worse |
| normal-load end-of-tile pipelining + q in registers | 211-274 | worse (2 syncs/tile; q-in-regs occupancy cliff) |
| 8-deep register staging pipeline | 174 | worse (1 block/SM) |
| per-kv-head contiguous KV layout `[block][kv_head][token][dim]` | 134 | **neutral** — the page-efficiency theory was wrong (see ceilings above) |

## What this means

- The d8192 two-request wall time has a hard floor at (requests x prefill GPU
  seconds) ~ 46 s here; the scheduler already sits on it at `gap=1000`.
  Config knobs only redistribute between TTFT and tg, and warm-state behavior
  is unstable: `decode_gap_target_ms=200` measured tg 24.8/29.0 on a cold
  scheduler (wall 44 s) but 7.3/37.1 with wall 56.7 s once warm, i.e. worse
  than the default. The default templates keep 1000.
- Real prefill wins need a different data path, not scheduling: a
  wave-synchronized (cooperative or persistent-CTA) attention that walks the
  KV window across all rows together, so the in-flight working set is one
  window, not 296 diverged row sweeps. That converts the 10.2 GB per launch
  into ~21 MB of DRAM plus L2-resident reuse at ~1 TB/s. CUTLASS's Blackwell
  FMHA (or a port of a flash kernel with paged KV) is the other route.
  Upstream reference ports (vLLM #60149, llama.cpp #30306, both open at time
  of writing) have no fast deep-prefill path either.
