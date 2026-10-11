# Benchmarks

Serving throughput, decode modes and context scaling on the GB10 cluster. Each row identifies a model, weight format, node count and deployment configuration.

| Measurement set | Date (UTC) | Source revision | Scope and record |
|---|---|---|---|
| Qwen3.8-27B quant comparison | 2026-10-11 | `c9a98ac7`; [binary and source hashes](../benchmarks/results/2026-10-11-pr95-followups/final-build-manifest.json) | Matched FP8 / NVFP4 / FP8 at worlds 1/2/4, 256-token C1/C2/C4/C8 across five classes and cold 512/2K/8K/32K prefills; [record](../benchmarks/results/2026-10-11-pr95-followups/README.md) |
| GLM-5.3 Mixed346 checkpoint | 2026-10-10 | `41e3216d` plus the Mixed346 working tree | The rows for GLM-5.3 Mixed346 GPTQ H32 A8 g128 (C1–C8, cold prefill, context buckets, quality); measured on the four nodes with the fp8-KV template after the kernel round of 2026-10-10 (the int8 tensor-core tile, the 3/6-bit cooperative pieces, the fused norm); the Int4/Int8 and NF4I8 references are the 2026-10-09 rows on the same binary lineage |
| GLM-5.3 NF4I8 checkpoint | 2026-10-09 | `41e3216d` plus the NF4I8 working tree | The rows for GLM-5.3 NF4I8 H32 g128 (C1–C8, cold prefill, context buckets, quality), measured without the rank-0 CUDA-connections setting; the same binary measures the int4/int8 fp8-KV deployment at 3.896 / 17.866 / 76.439 s cold prefill and the same decode steps (its table rows are the 2026-10-07 binary's: 3.555 / 17.188 / 74.270) |
| Prefill benchmark completion | 2026-10-07–08 | `90ebbf1` plus both recorded kernel patches; [binary and source hashes](../benchmarks/results/2026-10-07-prefill-completion/manifest.json) | All 46 targeted groups complete across seven affected configurations; [measurement audit](../benchmarks/results/2026-10-07-prefill-completion/measurement-audit.json) |
| Prefill efficiency follow-up | 2026-10-07 | `90ebbf1` plus recorded kernel patches; binary hashes in each [run record](../benchmarks/results/2026-10-07-prefill-followup/README.md) | Matched before/after cold-prefill comparisons for two-node GLM Flash, full-GLM FP4 KV and two-node MiMo; corrected results retained below |
| Matrix refresh | 2026-10-06–07 | [`a3ebc21`](https://github.com/HawkBearPig/dgpp/commit/a3ebc21a7c245eb736cc799fcf87a2c97ed8f7c9); final runs use [`07a32fb`](https://github.com/HawkBearPig/dgpp/commit/07a32fb912a264de1ea52801ffee127d8bc05235) | All 29 deployments complete: C1/C2/C4/C8, decode modes, cold prefill and context scaling; [coverage](../benchmarks/results/2026-10-06-prefill-scaling/README.md) and [per-measurement source records](../benchmarks/results/2026-10-06-prefill-scaling/result-provenance.json) |
| DeepSeek-V4.1 prefill batching | 2026-10-07 | [`07a32fb`](https://github.com/HawkBearPig/dgpp/commit/07a32fb912a264de1ea52801ffee127d8bc05235); [validation](../benchmarks/results/2026-10-06-prefill-scaling/diagnostics/deepseek-capacity-prefill-investigation/README.md) | Both DeepSeek-V4.1 deployments were rerun. The [capacity retention catalog](../benchmarks/results/2026-10-06-prefill-scaling/capacity-retained-results.json) preserves unaffected model families |
| Full-GLM score fusion | 2026-10-07 | [`21b91a3`](https://github.com/HawkBearPig/dgpp/commit/21b91a39f999cef0c94579a3240b62b07854a80b), [`0b10cbe`](https://github.com/HawkBearPig/dgpp/commit/0b10cbeaed1b7c51d854ec2f3a23729cb7b91c35); [validation](../benchmarks/results/2026-10-06-prefill-scaling/diagnostics/full-glm-prefill-investigation/README.md) | Affected full-GLM prefill groups were rerun. The [retention catalog](../benchmarks/results/2026-10-06-prefill-scaling/retained-results.json) identifies unchanged measurements from the earlier fixed binary |
| Before the scaling fix | 2026-10-06 | `50309fd` + route-trace fix | Verified short-prompt GLM Flash measurements use an unchanged code path and pass the [reuse audit](../benchmarks/results/2026-10-06-prefill-scaling/reused-results.json). Earlier affected measurements are retained only as [baselines](../benchmarks/results/2026-10-05-matrix-refresh/README.md) |
| Original campaign | 2026-09-22 | `a3ed8994a34c` | GLM, Qwen Flash-Next and DeepSeek-V4.1 quality/correctness; [record](../benchmarks/results/2026-09-22-current/README.md) |
| MiMo | 2026-09-22/23 | `a895db96bb17` + recorded changes | Quality/correctness; [record](../benchmarks/results/2026-09-22-mimo-v26-flash-opt/README.md) |
| Qwen AutoRound | 2026-09-29 | Campaign revisions recorded per run | Quality/correctness; [record](../benchmarks/results/2026-09-28-qwen-autoround-int4/README.md) |
| DeepSeek-V4 | 2026-10-02 | `904f76c8519a` + recorded changes | Historical quality/correctness and llama-benchy; [record](../benchmarks/results/2026-10-01-deepseek-v4-flash/README.md) |
| Qwen 27B | 2026-10-05 | Morning: `5b57904274d3` + recorded changes; evening: `dc52f66` | Quality/correctness and earlier performance; [record](../benchmarks/results/2026-10-05-qwen3.8-27b/README.md), [evening measurements](../benchmarks/results/2026-10-05-qwen3.8-27b/notes.md) |

[Performance](#serving-performance) · [Configuration](#hardware-and-configuration) · [Method](#workload-and-timing) · [Decode modes](#decode-modes) · [Quality](#quality-and-correctness) · [Long context](#long-context) · [Reproduce](#reproduce)

## Serving performance

Rates are tokens per second. Ranges span the medians of five prompt classes, with three repetitions per class. **C** is the number of concurrent client requests. Every deployment is tested at **C1, C2, C4 and C8**.

| Metric | Timing scope |
|---|---|
| C1 engine tok/s | Decode steps only; excludes prefill and its first output token |
| C1/C2/C4/C8 wall tok/s | Total completed output divided by request-group wall time, including queueing and prefill |
| Cold prefill seconds | Median engine prefill time at each approximate target length |

Client concurrency and request slots are different. A C8 test on a two-slot deployment includes the time requests spend waiting for a slot. The deployment's slot count, KV pool and decode mode stay fixed throughout its concurrency sweep.

The seven configurations affected by the latest prefill fixes have updated serving and context measurements. Cold-prefill medians use three uncached probes, with two exceptions: linked diagnostic cells use two, and the GLM Flash 256K row uses two at ~2K/~8K and three at ~32K. Both two-node GLM Flash configurations use a 2,048-token idle budget.

The Qwen3.8-27B rows were refreshed on 2026-10-11. FP8 uses the first of two bracketing baselines; the comparison also checks the second baseline. NVFP4 leads in all 12 cold-prefill buckets and 59 of 60 decode cells; four-node single-request prose is 57.4 versus 61.2 tok/s (6.3% slower) because DFlash2 accepts fewer draft tokens. Long-prefill gains are small (0.08–1.1% at 8K/32K), so these are measured margins, not a guarantee for every input. Their cold probes use the same `pr95-matched` prompt hashes across formats. Actual lengths differ from the older matrix, including prompts on opposite sides of a 4096-token prefill chunk boundary, so compare within the new campaign. The historical decode-mode, quality and parcel-context rows retain their original evidence.

<!-- BEGIN serving -->

| Model / weights | Nodes | Options | C1 engine tok/s | C1 wall tok/s | C2 wall tok/s | C4 wall tok/s | C8 wall tok/s | Cold prefill s: ~2K / ~8K / ~32K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 30.6–36.0 | 29.6–34.5 | 43.0–46.8 | 55.3–61.2 | 56.4–61.8 | [2.057 / 8.384 / 35.312](../benchmarks/results/2026-10-07-prefill-followup/raw/glm-flash-hybrid-w2/prefill/idle2048/prefill.json) |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 30.7–36.1 | 29.6–34.6 | 43.1–46.9 | 44.7–48.1 | 44.5–47.2 | 1.976 / 8.178 / 35.025 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 56.8–62.2 | 54.8–59.7 | 79.5–85.4 | 106.2–117.7 | 105.7–116.0 | 1.182 / 5.075 / 21.843 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 46.7–51.5 | 44.8–49.2 | 60.1–64.4 | 77.5–86.2 | 77.6–84.9 | 1.334 / 5.686 / 24.491 |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | 46.8–51.6 | 44.9–49.3 | 60.2–64.4 | 77.8–86.3 | 77.5–85.8 | 1.356 / 5.716 / 24.507 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 41.7–49.6 | 40.7–48.1 | 62.1–73.1 | 86.5–97.7 | 84.3–98.9 | 1.363 / 4.522 / 18.054 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 31.2–38.0 | 30.6–37.1 | 52.0–60.9 | 74.2–84.8 | 73.5–85.6 | 1.392 / 6.116 / 19.055 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 62.3–74.9 | 60.5–72.5 | 93.3–109.3 | 126.1–146.1 | 124.9–147.3 | 2.130 / 7.873 / 28.829 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 65.0–75.9 | 63.0–73.0 | 94.5–110.2 | 94.7–111.2 | 93.0–109.6 | 1.830 / 7.145 / 28.795 |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | 44.2–61.0 | 42.9–58.4 | 56.1–76.8 | 80.6–106.3 | 78.3–107.9 | 1.358 / 4.602 / 18.234 |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | 52.1–96.9 | 50.6–92.3 | 71.9–136.4 | 100.8–161.5 | 95.0–177.6 | 1.870 / 7.179 / 28.922 |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | 44.0–52.9 | 43.0–51.3 | 68.2–81.3 | 90.6–103.0 | 81.0–100.3 | 2.027 / 8.061 / 33.457 |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | 70.8–84.5 | 68.7–81.1 | 110.6–128.3 | 141.5–162.2 | 139.8–164.1 | 1.582 / 5.614 / 23.039 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | 43.4–69.8 | 42.3–66.9 | 57.6–90.3 | 76.5–115.7 | 76.0–124.8 | 1.276 / 4.356 / 17.159 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | 43.1–69.0 | 42.1–66.0 | 55.6–91.8 | 78.9–120.1 | 77.1–124.4 | 1.182 / 4.077 / 16.682 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | 27.4–30.5 | 26.1–28.8 | 35.7–38.4 | 46.1–49.7 | 52.9–55.7 | 3.493 / 16.219 / 70.249 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV, 8 slots | 27.0–30.4 | 25.8–28.7 | 36.0–38.5 | 46.0–49.3 | 53.2–56.6 | 3.555 / 17.188 / 74.270 |
| GLM-5.3 int4/int8 | 4 | 256K FP4 KV, 8 slots | 27.4–30.5 | 26.1–28.8 | 36.4–37.7 | 45.4–49.6 | 52.4–56.9 | [3.770 / 17.180 / 74.988](../benchmarks/results/2026-10-07-prefill-followup/raw/glm53-fp4kv-256k-w4/prefill/native-conversion-clean/prefill.json) |
| GLM-5.3 NF4I8 H32 g128 | 4 | 272K FP8 KV, 8 slots | 27.1–30.6 | 25.9–28.9 | 36.4–39.0 | 46.2–49.3 | 52.6–56.4 | 3.872 / 17.915 / 76.522 |
| GLM-5.3 Mixed346 H32 A8 g128 | 4 | 272K FP8 KV, 8 slots | 28.1–31.6 | 26.8–29.8 | 37.3–39.1 | 47.3–52.0 | 56.0–60.0 | 3.906 / 17.943 / 76.714 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 42.1–77.9 | 40.9–73.2 | 62.9–102.0 | 79.1–109.3 | 66.8–95.3 | 1.483 / 5.037 / 19.869 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 41.8–78.2 | 40.8–73.7 | 62.4–102.2 | 78.8–111.2 | 67.1–97.9 | 1.472 / 4.946 / 19.545 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 40.8–73.6 | 39.5–68.6 | 54.0–93.0 | 68.2–101.4 | 65.7–97.3 | 2.006 / 7.586 / 36.786 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 65.7–128.2 | 63.1–117.2 | 89.8–157.7 | 120.9–169.3 | 109.5–158.0 | 1.309 / 5.226 / 27.327 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 43.4–49.0 | 41.5–46.2 | 57.0–63.6 | 75.1–84.7 | 74.4–83.2 | [1.567 / 5.512 / 25.825](../benchmarks/results/2026-10-07-prefill-followup/raw/mimo-w2/prefill/fp8-dispatch-fix/prefill.json) |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 40.5–47.0 | 38.8–44.5 | 54.9–62.1 | 74.9–84.1 | 73.1–82.1 | [1.559 / 5.573 / 27.945](../benchmarks/results/2026-10-07-prefill-followup/raw/mimo-w2-fp8kv/prefill/fp8-dispatch-fix/prefill.json) |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 74.4–85.1 | 70.9–80.5 | 99.3–114.0 | 135.5–146.5 | 130.9–147.7 | 1.105 / 3.966 / 17.627 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 74.8–85.7 | 71.5–80.9 | 99.4–115.1 | 136.1–148.4 | 131.3–148.7 | 1.087 / 3.887 / 17.454 |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 20.0–47.9 | 19.8–46.1 | 32.1–76.7 | 57.5–109.2 | 90.4–189.8 | [2.285 / 7.420 / 32.392](../benchmarks/results/2026-10-11-pr95-followups/final-w1-fp8a/prefill.json) |
| Qwen3.8-27B NVFP4 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 21.4–55.1 | 21.1–53.3 | 39.2–97.0 | 71.1–132.0 | 110.8–212.6 | [2.149 / 7.322 / 32.113](../benchmarks/results/2026-10-11-pr95-followups/final-w1-nvfp4/prefill.json) |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 33.7–83.2 | 33.1–78.9 | 55.7–129.9 | 95.7–176.2 | 149.4–273.9 | [1.684 / 5.710 / 24.348](../benchmarks/results/2026-10-11-pr95-followups/final-w2-fp8a/prefill.json) |
| Qwen3.8-27B NVFP4 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 36.3–95.0 | 35.6–90.1 | 64.7–156.4 | 111.8–202.4 | 172.6–306.6 | [1.601 / 5.648 / 24.098](../benchmarks/results/2026-10-11-pr95-followups/final-w2-nvfp4/prefill.json) |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 50.4–133.0 | 49.2–123.2 | 85.0–193.1 | 144.1–246.4 | 208.5–349.9 | [1.306 / 4.620 / 19.212](../benchmarks/results/2026-10-11-pr95-followups/final-w4-fp8a/prefill.json) |
| Qwen3.8-27B NVFP4 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 57.4–149.2 | 56.0–136.8 | 94.2–220.0 | 152.1–270.6 | 225.6–379.2 | [1.283 / 4.606 / 19.173](../benchmarks/results/2026-10-11-pr95-followups/final-w4-nvfp4/prefill.json) |

<!-- END serving -->

Configuration labels use **K = 1,024 tokens**. Nodes is the tensor-parallel world size; KV is the shared key/value-cache token pool.

The [cold-prefill follow-up](../benchmarks/results/2026-10-07-prefill-followup/README.md) found three avoidable costs. The table above incorporates the corrected cold-prefill measurements. These matched comparisons quantify each change, with two cold probes per length:

| Deployment | Change | ~8K prefill before → after |
|---|---|---:|
| GLM Flash, 2 nodes, 160K FP8 KV | Idle chunk: 256 → 2,048 tokens | 19.42 → 8.38 s |
| GLM Flash, 2 nodes, 256K FP8 KV | Idle chunk: 256 → 2,048 tokens | 19.32 → 8.18 s |
| Full GLM, 4 nodes, FP4 KV | Native FP4 attention-cache conversion | 24.50 → 17.18 s |
| MiMo, 2 nodes, BF16 KV | Route FP8 projections to the existing optimized GEMM | 12.93 → 5.51 s |
| MiMo, 2 nodes, FP8 KV | Same projection dispatch fix | 12.97 → 5.57 s |

The two-node GLM Flash example now selects the larger idle budget. The kernel fixes preserve the tested outputs and require no KV allocation reduction. Full-GLM BF16/FP8 paths are unchanged by these fixes; their published timings remain. Their profiles are dominated by attention, packed matrix products and collectives.

The [follow-up report](../benchmarks/results/2026-10-07-prefill-followup/README.md) records every comparison, correctness checks and profiling evidence. The [completion campaign](../benchmarks/results/2026-10-07-prefill-completion/README.md) supplies the remaining cold-prefill probes and all affected serving and context measurements, plus full-GLM FP4 KV decode modes. All 46 measurement groups passed request, rank-identity and zero-swap checks. Native FP8 attention evaluation is tracked in [issue #98](https://github.com/HawkBearPig/dgpp/issues/98).

## Hardware and configuration

| Component | Configuration |
|---|---|
| Rank 0 | MSI EdgeXpert, NVIDIA GB10 |
| Rank 1 | ASUS GX10, NVIDIA GB10 |
| Rank 2 | NVIDIA DGX Spark, NVIDIA GB10 |
| Rank 3 | AI TOP ATOM, NVIDIA GB10 |
| Memory | About 121 GiB of OS-visible unified memory per node |
| Network | 200 Gb/s RoCE fabric |
| Node counts | 1, 2 and 4 |
| Software | CUDA 13.0; NVIDIA driver 580.173.02; CMake `release` preset |
| Timing conditions | One benchmark deployment at a time; models and decode graphs loaded; no profiler |
| Telemetry | Host and serving-process memory every second on all serving ranks; GPU state every five seconds; rank 2 kernel journal on four-node launches |
| Rank validation | Recorded operation streams must match across all ranks after a clean shutdown before a launch contributes results |
| Memory validation | Each `dgpp-serve` process must report zero swap throughout monitoring; a nonzero sample stops the process and invalidates the run. OS swap remains enabled |
| Excluded runs | The [initial swapping attempt](../benchmarks/results/2026-10-05-matrix-refresh/invalid-swapping/INVALID.md) and [later four-node 512K attempt](../benchmarks/results/2026-10-05-matrix-refresh/diagnostics/swap-after-trace-fix.md) contribute no results |
| Swap prevention | Every new launch uses `DGPP_NO_SWAP=1`: each rank must have a cgroup swap limit of zero and report zero process and cgroup swap throughout monitoring |
| Earlier retained runs | The first two deployments preceded the cgroup policy; every rank reported zero process swap and no host swap-outs. [Audit](../benchmarks/results/2026-10-05-matrix-refresh/diagnostics/completed-before-swap-policy.json) |
| CUDA connections | `CUDA_DEVICE_MAX_CONNECTIONS=32` on rank 0; peers retain their login environment |
| Environment records | [Original comparison](../benchmarks/results/2026-09-22-current/environment.md); [refresh manifest](../benchmarks/results/2026-10-06-prefill-scaling/manifest.json) |

The KV pool is shared across requests. Its capacity does not mean every concurrent request can use that many tokens. The linked JSON files define the effective settings for each measured row.

Prefill tokens per tick are the resolved startup settings: **busy** applies while another request is decoding; **idle** applies when none is decoding. **0** means the scheduler submits the full prompt in one tick; the model may split it into internal batches. These settings affect prefill timing. The [startup records](../benchmarks/results/2026-10-06-prefill-scaling/diagnostics/resolved-prefill-budgets.json) identify their source logs.

<!-- BEGIN config -->

| Model / weights | Nodes | Options | Slots | KV pool tokens | KV format | Prefix cache GiB | Prefill tokens/tick (busy / idle) | Default decode | Configuration |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 4 | 163,840 | fp8 | 4.5 | 256 / 2,048 | MTP1 | [JSON](../benchmarks/results/2026-10-07-prefill-completion/configs/glm-flash-hybrid-w2.json) |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 2 | 262,144 | fp8 | 2 | 256 / 2,048 | MTP1 | [JSON](../benchmarks/results/2026-10-07-prefill-completion/configs/glm-flash-hybrid-256k-w2.json) |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 4 | 786,432 | bf16 | 22 | 256 / 2,048 | MTP1 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/glm-flash-hybrid-w4.json) |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 4 | 393,216 | bf16 | 8 | 256 / 2,048 | MTP1 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/glm-flash-fp8-w4.json) |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | 4 | 524,288 | bf16 | 2 | 256 / 2,048 | MTP1 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/glm-flash-fp8-512k-w4.json) |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 4 | 262,144 | bf16 | 3 | 4,096 / 4,096 | MTP1 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/qwen-nvfp4-w1.json) |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 4 | 262,144 | bf16 | 3 | 4,096 / 4,096 | MTP1 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/qwen-nvfp4-bf16-w1.json) |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 4 | 262,144 | bf16 | 58 | 256 / 256 | MTP1 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/qwen-nvfp4-w2.json) |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 2 | 565,248 | bf16 | 40 | 256 / 256 | MTP1 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/qwen-yarn-w2.json) |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | 4 | 262,144 | bf16 | 3 | 4,096 / 4,096 | MTP2 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/qwen-radixark-w1.json) |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | 4 | 524,288 | bf16 | 32 | 256 / 256 | MTP4 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/qwen-radixark-w2.json) |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | 4 | 262,144 | bf16 | 6 | 256 / 256 | MTP1 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/qwen-fp8-w2.json) |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | 4 | 262,144 | bf16 | 50 | 256 / 256 | MTP1 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/qwen-fp8-w4.json) |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | 4 | 262,144 | bf16 | 3 | 4,096 / 4,096 | MTP3 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/qwen-autoround-w1.json) |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | 4 | 262,144 | bf16 | 3 | 4,096 / 4,096 | MTP3 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/qwen-autoround-prefill-w1.json) |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | 8 | 122,880 | bf16 | 1.0 | 0 / 0 | MTP1 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/glm53-w4.json) |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV, 8 slots | 8 | 212,992 | fp8 | 1.5 | 0 / 0 | MTP1 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/glm53-fp8kv-w4.json) |
| GLM-5.3 int4/int8 | 4 | 256K FP4 KV, 8 slots | 8 | 262,144 | fp4 | 1.5 | 0 / 0 | MTP1 | [JSON](../benchmarks/results/2026-10-07-prefill-completion/configs/glm53-fp4kv-256k-w4.json) |
| GLM-5.3 NF4I8 H32 g128 | 4 | 272K FP8 KV, 8 slots | 8 | 278,528 | fp8 | 1.0 | 0 / 0 | MTP1 | [JSON](../deploy/cluster_glm-5.3_nf4i8_w4.example.json) |
| GLM-5.3 Mixed346 H32 A8 g128 | 4 | 272K FP8 KV, 8 slots | 8 | 278,528 | fp8 | 1.0 | 0 / 0 | MTP1 | [JSON](../deploy/cluster_glm-5.3_mixed346_w4.example.json) |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 6 | 131,072 | FP4 blocks / FP8 window | 14 | 0 / 0 | DSpark adaptive ≤4 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/deepseek-w4.json) |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 6 | 1,048,576 | FP4 blocks / FP8 window | 14 | 0 / 0 | DSpark adaptive ≤4 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/deepseek-1m-w4.json) |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 4 | 1,048,576 | BF16 (quantized values) | 8 | 256 / 4,096 | DSpark adaptive ≤5 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/dsv4-w2.json) |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 6 | 1,048,576 | BF16 (quantized values) | 14 | 256 / 4,096 | DSpark adaptive ≤5 | [JSON](../benchmarks/results/2026-10-06-prefill-scaling/configs/dsv4-w4.json) |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 4 | 131,072 | bf16 | 1.5 | 0 / 0 | MTP1 | [JSON](../benchmarks/results/2026-10-07-prefill-completion/configs/mimo-w2.json) |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 4 | 262,144 | fp8 | 1.5 | 0 / 0 | MTP1 | [JSON](../benchmarks/results/2026-10-07-prefill-completion/configs/mimo-w2-fp8kv.json) |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 4 | 131,072 | bf16 | 1.5 | 0 / 0 | MTP1 | [JSON](../benchmarks/results/2026-10-07-prefill-completion/configs/mimo-w4.json) |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 4 | 1,048,576 | bf16 | 1.5 | 0 / 0 | MTP1 | [JSON](../benchmarks/results/2026-10-07-prefill-completion/configs/mimo-1m-w4.json) |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 8 | 262,144 | bf16 | 1.5 | 0 / 0 | DFlash2 adaptive ≤7 | [JSON](../benchmarks/results/2026-10-11-pr95-followups/final-w1-fp8a/deployment.json) |
| Qwen3.8-27B NVFP4 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 8 | 262,144 | bf16 | 1.5 | 0 / 0 | DFlash2 adaptive ≤7 | [JSON](../benchmarks/results/2026-10-11-pr95-followups/final-w1-nvfp4/deployment.json) |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 8 | 262,144 | bf16 | 1.5 | 0 / 0 | DFlash2 adaptive ≤7 | [JSON](../benchmarks/results/2026-10-11-pr95-followups/final-w2-fp8a/deployment.json) |
| Qwen3.8-27B NVFP4 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 8 | 262,144 | bf16 | 1.5 | 0 / 0 | DFlash2 adaptive ≤7 | [JSON](../benchmarks/results/2026-10-11-pr95-followups/final-w2-nvfp4/deployment.json) |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 8 | 262,144 | bf16 | 1.5 | 0 / 0 | DFlash2 adaptive ≤7 | [JSON](../benchmarks/results/2026-10-11-pr95-followups/final-w4-fp8a/deployment.json) |
| Qwen3.8-27B NVFP4 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 8 | 262,144 | bf16 | 1.5 | 0 / 0 | DFlash2 adaptive ≤7 | [JSON](../benchmarks/results/2026-10-11-pr95-followups/final-w4-nvfp4/deployment.json) |

<!-- END config -->

| Option | Meaning |
|---|---|
| GLM Flash NVFP4/FP8 | [Mixed checkpoint](model_cards/GLM-5.3-Flash-NVFP4-FP8.md): NVFP4 routed experts; remaining tensors retain their FP8-release formats |
| GLM KV formats | BF16, FP8 and FP4 are selectable latent-cache formats. FP8/FP4 reduce storage and change attention values; model weights are unchanged. The added 256K FP4 configuration has performance measurements only, with no quality evaluation in this refresh |
| Qwen NVFP4, FP8/BF16 dense | Format of dense projections; expert weights stay NVFP4. The BF16 variant uses `dense_weights=checkpoint`, `fp8_head=gemv` |
| Qwen NVFP4, YaRN 512K | Extended-context RoPE configuration; n-gram tables remain mapped from NVMe, as in the other NVFP4 rows |
| Qwen RadixArk NVFP4 | Separate RadixArk checkpoint; one-node template uses MTP2 and 4K prefill chunks, two-node template uses MTP4 and a 512K shared KV pool. The benchmark follows the JSON templates |
| Qwen AutoRound | Single-node int4/int8 hybrid, FP8 dense projections, MTP3; optional draft-vocabulary slicing is off |
| Qwen AutoRound, prefill optimizations on | `prefill_bf16_partials` and `prefill_fp8_gemm` enabled, as in the template. These change prefill arithmetic; the separate row measures their effect |
| MiMo, two-node BF16 KV | Two-node template with `kv_dtype=bf16`, `kv_capacity=131072`; the FP8 row uses a 256K pool |
| DeepSeek-V4.1 | FP4 block format for the main KV cache and FP8 block format for the sliding window. No selectable KV format |
| DeepSeek-V4.1, prefill | `prefill=bounded`: decoder work is limited to the final window of each prompt |
| DeepSeek-V4 | 0731 checkpoint with DSpark; BF16 cache rows hold the release's FP8-quantized values. No selectable KV format |
| Qwen 27B | DFlash2 drafter and FP8 head; native MTP modes are measured separately |

## Workload and timing

| Setting | Serving / decode modes | Long context |
|---|---|---|
| Prompts | Repository prose, code, JSON, math and chat corpus | Deterministic parcel documents |
| Cold-prefill input | Prepared GSM8K text; two or three distinct prefixes per target, as specified in Serving performance | First request for each parcel document |
| Output budget | 256 tokens per request | 256 tokens per request |
| Repetitions | Three per class and concurrency | One cold request, then two identical requests |
| Sampling | Greedy, temperature zero | Greedy, temperature zero |
| Thinking | Disabled where supported; full GLM-5.3 uses its reasoning mode | Server default |
| Cold-prefill thinking | GLM probes use the server default; other families disable thinking | Server default |
| Prefix cache | Enabled; repetitions reuse prompts | Enabled; first request must report zero cached tokens |
| Concurrency | C1/C2/C4/C8; mode comparisons use C1 | C1 |
| Summary | Range across five class medians | Median across three requests; cold prefill is the first request |

- Each concurrency phase uses distinct prompts. Different concurrency levels use different subsets of the same class corpus.
- Engine rate is `1000 × (tokens_generated − prompts_prefilled) / step_ms`.
- Decode latency is `step_ms / decode_steps`. Speculative decoding can commit several tokens per pass, so latency and token throughput describe different quantities.
- Wall rate is total completion tokens divided by the interval from launching a request group until its last completion. SSE update counts are not token counts.
- Scheduler counters must reconcile with all completed requests before the next phase starts. Ranges describe prompt variation, not confidence intervals.

## Single-request decode by class

Engine tokens/s, greedy; median of three repetitions.

<!-- BEGIN classes -->

| Model / weights | Nodes | Options | Prose | Code | JSON | Math | Chat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 34.2 | 35.9 | 36.0 | 34.6 | 30.6 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 34.2 | 36.0 | 36.1 | 34.8 | 30.7 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 60.3 | 61.6 | 62.2 | 60.0 | 56.8 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 49.6 | 50.6 | 51.5 | 50.1 | 46.7 |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | 49.5 | 50.7 | 51.6 | 50.2 | 46.8 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 41.7 | 49.0 | 49.6 | 47.0 | 44.0 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 33.9 | 37.3 | 38.0 | 36.1 | 31.2 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 67.5 | 73.6 | 74.9 | 72.3 | 62.3 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 65.0 | 74.4 | 75.9 | 72.3 | 65.0 |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | 47.5 | 59.5 | 61.0 | 54.7 | 44.2 |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | 52.1 | 90.2 | 96.9 | 79.7 | 53.1 |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | 45.9 | 51.8 | 52.9 | 49.8 | 44.0 |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | 74.2 | 83.8 | 84.5 | 79.3 | 70.8 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | 43.4 | 59.7 | 69.8 | 60.9 | 45.6 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | 43.1 | 58.3 | 69.0 | 59.3 | 45.6 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | 29.5 | 30.5 | 30.5 | 29.3 | 27.4 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV, 8 slots | 30.1 | 30.4 | 30.4 | 29.8 | 27.0 |
| GLM-5.3 int4/int8 | 4 | 256K FP4 KV, 8 slots | 29.9 | 29.2 | 30.5 | 29.6 | 27.4 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 42.1 | 64.9 | 77.9 | 62.8 | 46.4 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 41.8 | 65.0 | 78.2 | 62.9 | 46.2 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 40.8 | 62.1 | 73.6 | 58.8 | 43.8 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 65.7 | 114.5 | 128.2 | 94.8 | 70.8 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 43.5 | 46.2 | 49.0 | 46.3 | 43.4 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 42.7 | 45.1 | 47.0 | 45.8 | 40.5 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 75.5 | 81.2 | 85.1 | 84.2 | 74.4 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 75.6 | 81.5 | 85.7 | 84.5 | 74.8 |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 20.5 | 34.3 | 47.9 | 39.5 | 20.0 |
| Qwen3.8-27B NVFP4 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 21.4 | 41.8 | 55.1 | 45.9 | 23.8 |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 33.7 | 60.0 | 83.2 | 69.3 | 35.8 |
| Qwen3.8-27B NVFP4 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 36.3 | 71.9 | 95.0 | 79.3 | 41.2 |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 61.2 | 96.9 | 133.0 | 106.1 | 50.4 |
| Qwen3.8-27B NVFP4 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 57.4 | 115.9 | 149.2 | 124.8 | 65.3 |

<!-- END classes -->

## Decode modes

C1 engine tokens/s; each range spans the five class medians. **MTP1**, **MTP2** and **MTP3** mean fixed chains of one, two and three draft tokens. DSpark is DeepSeek's block drafter; its numbered columns specify the number of drafts verified. DFlash2 is the separate Qwen 27B drafter. Adaptive columns use the deployment's confidence-based depth schedule.

**Not swept** identifies MTP4 where it is neither a default mode nor part of the earlier AutoRound comparison; MTP1/MTP2/MTP3 are tested for every native-MTP deployment. Exact settings, pass times and tokens per pass remain in each raw record.

<!-- BEGIN modes -->

### Native MTP

| Model / weights | Nodes | Options | Default | Plain | MTP1 | MTP2 | MTP3 | MTP4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | MTP1 | 24.2–24.3 | 30.6–36.0 | 31.6–40.0 | 27.6–37.8 | Not swept |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | MTP1 | 23.9–24.0 | 30.7–36.1 | 31.6–40.6 | 27.4–37.8 | Not swept |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | MTP1 | 39.6–39.7 | 56.8–62.2 | 58.8–68.9 | 53.2–66.3 | Not swept |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | MTP1 | 34.6–34.7 | 46.7–51.5 | 44.9–55.0 | 39.9–50.0 | Not swept |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | MTP1 | 34.3–34.4 | 46.8–51.6 | 44.9–55.0 | 39.9–50.1 | Not swept |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | MTP1 | 31.8–31.9 | 41.7–49.6 | 42.1–61.0 | 39.9–66.8 | Not swept |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | MTP1 | 24.2 | 31.2–38.0 | 36.1–47.7 | 33.0–54.6 | Not swept |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | MTP1 | 46.5–48.0 | 62.3–74.9 | 65.5–91.1 | 61.2–100.5 | Not swept |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | MTP1 | 47.9–48.0 | 65.0–75.9 | 66.2–91.4 | 61.2–100.5 | Not swept |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | MTP2 | 32.0–32.1 | 42.4–50.2 | 44.2–61.0 | 40.5–67.7 | Not swept |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | MTP4 | 46.8–47.5 | 64.1–76.3 | 66.7–91.2 | 64.6–102.7 | 52.1–96.9 |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | MTP1 | 34.6–35.3 | 44.0–52.9 | 46.4–62.7 | 43.4–72.9 | Not swept |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | MTP1 | 51.2–51.5 | 70.8–84.5 | 77.4–108.3 | 74.5–119.3 | Not swept |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | MTP3 | 30.2–30.3 | 42.1–48.9 | 45.3–61.1 | 43.4–69.8 | 35.7–66.5 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | MTP3 | 30.2 | 41.2–48.6 | 43.2–61.1 | 43.1–69.0 | 35.2–66.5 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | MTP1 | 19.7–20.0 | 27.4–30.5 | 28.9–33.6 | 26.3–33.1 | Not swept |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV, 8 slots | MTP1 | 20.0–20.1 | 27.0–30.4 | 25.8–34.9 | 23.8–33.1 | Not swept |
| GLM-5.3 int4/int8 | 4 | 256K FP4 KV, 8 slots | MTP1 | 20.0–20.1 | 27.4–30.5 | 27.0–34.3 | 23.2–34.1 | Not swept |

### MiMo: MTP and DFlash

| Model / weights | Nodes | Options | Default | Plain | MTP1 | MTP2 | MTP3 | DFlash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | MTP1 | 33.4–33.5 | 43.4–49.0 | 40.0–52.2 | 34.8–48.4 | Not loaded by engine |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | MTP1 | 33.6–33.7 | 40.5–47.0 | 38.2–51.6 | 32.1–47.1 | Not loaded by engine |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | MTP1 | 56.1–56.3 | 74.4–85.1 | 70.6–90.9 | 60.3–87.5 | Not loaded by engine |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | MTP1 | 56.8–57.1 | 74.8–85.7 | 71.2–92.0 | 60.4–87.6 | Not loaded by engine |

### DeepSeek DSpark

| Model / weights | Nodes | Options | Default | Plain | DSpark 1 | DSpark 2 | DSpark 3 | DSpark 5 | DSpark adaptive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | DSpark adaptive ≤4 | 31.8–32.1 | 39.4–47.5 | 43.0–61.4 | 40.3–70.1 | 34.2–84.9 | 42.1–77.9 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | DSpark adaptive ≤4 | 31.3–32.2 | 39.1–47.1 | 43.0–61.7 | 40.2–69.9 | 35.1–85.0 | 41.8–78.2 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | DSpark adaptive ≤5 | 29.6–29.8 | 37.5–43.5 | 40.6–56.8 | 39.3–64.0 | 33.1–75.8 | 40.8–73.6 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | DSpark adaptive ≤5 | 43.6–44.0 | 59.4–69.2 | 64.5–88.3 | 63.3–106.0 | 57.2–128.7 | 65.7–128.2 |

### Qwen 27B: MTP and DFlash2

| Model / weights | Nodes | Options | Default | Plain | MTP1 | MTP2 | MTP3 | DFlash2 fixed 7 | DFlash2 adaptive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | DFlash2 adaptive ≤7 | 8.2 | 13.7–15.3 | 15.3–20.8 | 16.9–25.2 | 19.8–47.5 | 19.9–47.5 |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | DFlash2 adaptive ≤7 | 15.0–15.3 | 25.5–28.5 | 28.4–38.8 | 31.1–45.8 | 33.0–84.2 | 33.3–82.3 |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | DFlash2 adaptive ≤7 | 28.4–28.5 | 46.2–52.4 | 50.7–69.1 | 50.6–80.5 | 48.6–133.0 | 49.9–131.5 |

<!-- END modes -->

## Quality and correctness

These are the original campaigns' quality results, retained separately from the performance refresh. Scores apply to their saved configurations, prompts, evaluator and output limits; current deployment settings may differ.

| Evaluation setting | Value |
|---|---|
| HumanEval | All 164 problems; returned code block executed independently in a network-disabled container without host mounts |
| GSM8K | First 300 test problems |
| Schema extraction | 100 deterministic records |
| Temperature | Zero |
| Concurrency | Deployment request-slot count |
| Output cap | 2,048 tokens; 512 for extraction |
| Reasoning effort | Low with thinking enabled; none with thinking disabled |
| Retained evidence | Raw responses, execution errors and token-cap counts |

| Model / weights | Nodes | Options | HumanEval | GSM8K | Schema extraction | Responses at token cap |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 157/164 | 292/300 | 100/100 | 0 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 158/164 | 292/300 | 100/100 | 0 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 157/164 | 292/300 | 100/100 | 0 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 154/164 | 295/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 160/164 | 294/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 158/164 | 291/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 159/164 | 293/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 161/164 | 293/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | 160/164 | 292/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | 158/164 | 291/300 | 100/100 | 0 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | 159/164 | 292/300 | 100/100 | 1 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | 159/164 | 291/300 | 100/100 | 2 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 159/164 | 292/300 | 100/100 | 0 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 160/164 | 293/300 | 100/100 | 0 |
| GLM-5.3 NF4I8 H32 g128 | 4 | 272K FP8 KV | 157/164 | 294/300 | 100/100 | 0 |
| GLM-5.3 Mixed346 H32 A8 g128 | 4 | 272K FP8 KV | 157/164 | 291/300 | 100/100 | 0 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 160/164 | 296/300 | 100/100 | 0 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 141/164 | 295/300 | 100/100 | 0 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 139/164 | 296/300 | 100/100 | 0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 154/164 | 295/300 | 100/100 | 0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 153/164 | 294/300 | 100/100 | 0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 154/164 | 294/300 | 100/100 | 0 |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 156/164 | 292/300 | 100/100 | 4 |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 157/164 | 292/300 | 100/100 | 5 |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 156/164 | 291/300 | 100/100 | 4 |

DeepSeek-V4's strict HumanEval scores count responses with unclosed code fences as failures. Executing those open blocks raises the two-node score to 153/164 and the four-node score to 152/164. Plain decoding produces the same answers; the checkpoint's reference implementation also prefers end-of-sequence at that position. [Investigation](../benchmarks/results/2026-10-01-deepseek-v4-flash/notes.md).

| Model / weights | Nodes | Options | Rank checks passed | Plain/default greedy match | Solo/batched greedy text |
|---|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 3/3 | 5/5 classes | different |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 3/3 | 5/5 classes | identical |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 7/7 | 5/5 classes | different |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 4/4 | 5/5 classes | identical |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 5/5 | 5/5 classes | identical |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | 4/4 | 5/5 classes | different |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | 5/5 | 5/5 classes | different |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 4/4 | 5/5 classes | identical |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 3/3 | 5/5 classes | identical |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 5/5 | 5/5 classes | identical |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 6/6 | 5/5 classes | identical |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 6/6 | 5/5 classes | identical |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 3/3 | 5/5 classes | identical |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 4/4 | 5/5 classes | identical |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 4/4 | 5/5 classes | identical |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 4/4 | 4/5 classes | different |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 4/4 | 4/5 classes | identical |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 4/4 | 4/5 classes | different |

| Check | What it establishes |
|---|---|
| Rank operation streams | Ranks executed matching operation sequences; this does not establish numerical equality or request isolation |
| Plain/default greedy match | All three repetitions in both modes produced identical text and token counts for a class |
| Solo/batched greedy text | Exact transcript agreement for the recorded workload; floating-point reduction order can change with batch shape |

Recorded operation streams matched for 54/54 launches in the original campaign, 11/11 in MiMo and 12/12 in DeepSeek-V4. Single-node launches have only one stream.

Later checks do not replace the historical verdicts above.

| Follow-up | Setting / workload | Recorded result |
|---|---|---|
| Qwen 27B, 2026-10-05 | Other clients join an existing request; one, two and four nodes | Matched the solo requests. [Notes](../benchmarks/results/2026-10-05-qwen3.8-27b/notes.md) |
| Qwen 27B, simultaneous starts | Pinned cuBLASLt algorithm | Prompts above 128 tokens matched across prefill groups |
| Qwen 27B, shorter prompts | `prefill_group=false` | Also covers the shorter prompts; prefill grouping explained the remaining simultaneous-start differences |
| GLM Flash NVFP4/FP8, four nodes | Original dispatch settings | Solo and C2 matched; C3/C4 diverged. [Procedure](../benchmarks/results/2026-09-22-current/isolation.md) |
| GLM Flash NVFP4/FP8, four nodes | `DGPP_DENSE_GEMV_ROWS=8` or `256` | Tested C1–C4 responses matched solo, supporting kernel dispatch as the cause. This does not establish isolation for every workload. [Issue #32](https://github.com/HawkBearPig/dgpp/issues/32) |
| GLM-4.7 | Excluded at the user's earlier request | Cold-prefill failures and node recoveries remain in the [interruption record](../benchmarks/results/2026-09-22-current/interruptions.md); the cause was not established |

## Long context

Decode uses the default mode listed in the configuration table. Speculative acceptance and adaptive draft depth affect both decode latency and token throughput.

Full GLM-5.3's remaining long-context decode cost and cold-prefill runtime are tracked in [issue #96](https://github.com/HawkBearPig/dgpp/issues/96), with the measured configuration and profiling plan.

One row per model/quantization and deployment configuration. **Context length (limit)** is the server's advertised per-request limit, including prompt and output; it can be lower than the model's architectural limit because of the configured KV pool.

| Checkpoint family | Declared context limit | Extension used in this matrix |
|---|---|---|
| GLM-5.3 and GLM-5.3-Flash | 1,048,576 tokens | None |
| DeepSeek-V4 and DeepSeek-V4.1 Flash | 1,048,576 tokens | Checkpoint's YaRN settings |
| MiMo-V2.6-Flash | 1,048,576 tokens | None |
| Qwen3.8 Flash-Next and 27B | 262,144 tokens | Flash-Next's separate YaRN row extends to 524,288 tokens |

Limits come from the [pinned checkpoint configurations](../benchmarks/results/2026-10-06-prefill-scaling/context-capabilities.json). A larger shared KV pool alone does not extend a model's positional range.

| Column / label | Definition |
|---|---|
| 0 context | Minimal parcel prompt, at most 256 input tokens; not a literal empty KV cache |
| 32K / 64K / 128K / 256K / 512K | Initial prompt-length buckets, K = 1,024. Actual inputs are within 1,024 tokens below the label, reserving room for the chat wrapper and 256 output tokens |
| Decode ms/pass | Median engine time per decode pass across three C1 requests |
| Engine tok/s | Median output rate over engine decode time at that context |
| Cold prefill (s), 32K | Engine prefill time for the first uncached 32K-bucket request; timings for all lengths are in the following table |
| Limit | Bucket exceeds this deployment's request limit |

Context grows as the request generates output. The actual input count, cached-token count, tokens per pass and every repetition are in the [campaign record](../benchmarks/results/2026-10-06-prefill-scaling/README.md). These measure timing; retrieval accuracy is evaluated separately.

The original matrix refresh fixed the three bottlenecks below. The [completion campaign](../benchmarks/results/2026-10-07-prefill-completion/README.md) reran every context bucket affected by the later idle-budget, FP4-conversion and MiMo-dispatch fixes. These parcel-document timings use different prompts from the serving table's GSM8K probes. Other retained paths are unchanged by the latest fixes; the [scope review](../benchmarks/results/2026-10-07-prefill-completion/scope.json) records that source-path check.

| Affected path | Bottleneck | Investigation |
|---|---|---|
| GLM DSA selection | Repeated top-k sorting at long contexts | [Selection scaling](../benchmarks/results/2026-10-05-matrix-refresh/diagnostics/prefill-scaling/README.md) |
| Full GLM DSA scoring | Materializing and rereading per-head scores | [Score fusion](../benchmarks/results/2026-10-06-prefill-scaling/diagnostics/full-glm-prefill-investigation/README.md) |
| DeepSeek-V4.1 CSA2 prefill | Query tiles sized for maximum KV capacity, underusing workspace at shorter contexts | [Query batching](../benchmarks/results/2026-10-06-prefill-scaling/diagnostics/deepseek-capacity-prefill-investigation/README.md) |


| Added configuration / capacity check | Validation |
|---|---|
| GLM Flash FP8, four nodes | 512K KV pool with a 2 GiB prefix cache; all measurements completed with zero serving-process swap |
| DeepSeek-V4.1, four nodes | 1M KV pool, six slots and a 14 GiB prefix cache; all serving, decode-mode and context measurements completed with zero serving-process swap |
| MiMo, four nodes | 1M BF16 KV pool; all serving, decode-mode and context measurements completed with zero process and cgroup swap on every rank |
| Full GLM-5.3 int4/int8, four nodes, 256K FP8 KV | Normal startup refused 112.27 GiB + 4 GiB reserve against 115.58 GiB available. No benchmark requests ran. The preliminary plan omitted allocations made earlier in normal startup. [Capacity record](../benchmarks/results/2026-10-06-prefill-scaling/diagnostics/full-glm-256k-capacity-review.json) |
| Full GLM-5.3 int4/int8, four nodes, 256K FP4 KV | Replaces the unmeasured FP8 case; planned allocation is 4.32 GiB lower. Eight slots, 1.5 GiB prefix-cache budget and 4 GiB reserve are unchanged. All serving, decode-mode and context measurements completed with zero serving-process swap |
| Full GLM-5.3 int4/int8, four nodes, 512K | Memory plan rejected: 125.53 GiB with FP8 KV or 116.89 GiB with FP4 KV, plus 4 GiB headroom, against 116.7 GiB available. The measured rows retain smaller pools. [Capacity-check records](../benchmarks/results/2026-10-05-matrix-refresh/memory-plans/) |

<!-- BEGIN context -->

| Model / weights | Nodes | Options | Context length (limit) | Cold prefill (s), 32K | Decode ms/pass (0) | Engine tok/s (0) | Decode ms/pass (32K) | Engine tok/s (32K) | Decode ms/pass (64K) | Engine tok/s (64K) | Decode ms/pass (128K) | Engine tok/s (128K) | Decode ms/pass (256K) | Engine tok/s (256K) | Decode ms/pass (512K) | Engine tok/s (512K) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 163,840 | 34.357 | 54.98 | 33.9 | 57.35 | 34.2 | 58.40 | 33.3 | 59.82 | 32.5 | Limit | Limit | Limit | Limit |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 262,144 | 33.915 | 55.09 | 34.0 | 57.24 | 34.3 | 58.80 | 33.4 | 59.63 | 32.6 | 62.62 | 31.3 | Limit | Limit |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 786,432 | 21.799 | 31.85 | 58.9 | 33.55 | 58.5 | 34.33 | 57.1 | 35.96 | 54.1 | 39.15 | 50.1 | 45.52 | 43.4 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 393,216 | 24.243 | 38.49 | 47.3 | 40.43 | 48.5 | 41.25 | 47.9 | 42.83 | 45.8 | 46.07 | 41.9 | Limit | Limit |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | 524,288 | 24.234 | 38.40 | 48.5 | 40.35 | 48.6 | 41.89 | 46.1 | 43.00 | 45.6 | 46.02 | 42.3 | 52.26 | 37.8 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 262,144 | 18.241 | 40.53 | 45.6 | 42.73 | 42.0 | 43.12 | 43.2 | 43.97 | 42.0 | 45.65 | 39.1 | Limit | Limit |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 262,144 | 17.776 | 52.62 | 34.4 | 54.78 | 33.5 | 55.24 | 33.2 | 57.01 | 33.1 | 58.08 | 31.4 | Limit | Limit |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 262,144 | 30.420 | 27.13 | 69.6 | 28.38 | 66.1 | 28.70 | 62.6 | 29.93 | 59.6 | 31.54 | 59.0 | Limit | Limit |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 524,288 | 30.052 | 26.43 | 70.4 | 28.12 | 65.2 | 28.46 | 65.4 | 29.46 | 61.8 | 31.35 | 58.9 | 37.12 | 49.8 |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | 262,144 | 19.703 | 48.77 | 51.3 | 50.94 | 48.6 | 51.55 | 48.5 | 52.85 | 47.3 | 54.87 | 47.9 | Limit | Limit |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | 262,144 | 30.213 | 47.52 | 70.6 | 50.32 | 68.5 | 50.89 | 80.8 | 52.28 | 70.7 | 54.80 | 62.9 | Limit | Limit |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | 262,144 | 35.545 | 37.60 | 50.6 | 40.22 | 45.0 | 39.80 | 46.8 | 40.77 | 45.3 | 42.61 | 43.1 | Limit | Limit |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | 262,144 | 24.077 | 23.55 | 78.5 | 25.35 | 72.9 | 25.68 | 73.6 | 26.51 | 70.2 | 28.33 | 63.4 | Limit | Limit |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | 262,144 | 17.514 | 54.88 | 55.3 | 58.08 | 51.1 | 58.42 | 50.2 | 59.72 | 52.1 | 61.92 | 48.4 | Limit | Limit |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | 262,144 | 16.869 | 55.44 | 56.8 | 58.53 | 51.9 | 58.82 | 51.6 | 60.59 | 47.3 | 62.81 | 51.4 | Limit | Limit |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | 122,880 | 70.959 | 64.37 | 29.1 | 76.25 | 25.7 | 80.89 | 24.4 | Limit | Limit | Limit | Limit | Limit | Limit |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV, 8 slots | 212,992 | 75.099 | 64.18 | 29.2 | 76.24 | 25.7 | 81.06 | 24.2 | 90.13 | 21.9 | Limit | Limit | Limit | Limit |
| GLM-5.3 int4/int8 | 4 | 256K FP4 KV, 8 slots | 262,144 | 74.567 | 64.08 | 29.0 | 75.99 | 26.0 | 80.81 | 24.5 | 89.94 | 22.1 | 109.90 | 18.0 | Limit | Limit |
| GLM-5.3 NF4I8 H32 g128 | 4 | 272K FP8 KV, 8 slots | 278,528 | 74.341 | 64.30 | 29.2 | 76.40 | 25.9 | 81.03 | 24.2 | 90.16 | 21.8 | 109.20 | 18.2 | Limit | Limit |
| GLM-5.3 Mixed346 H32 A8 g128 | 4 | 272K FP8 KV, 8 slots | 278,528 | 74.520 | 62.91 | 29.6 | 74.71 | 26.1 | 79.52 | 24.5 | 88.87 | 22.2 | 108.16 | 18.1 | Limit | Limit |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 131,072 | 19.131 | 51.99 | 61.3 | 53.67 | 55.2 | 53.88 | 51.4 | 56.32 | 50.3 | Limit | Limit | Limit | Limit |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 1,048,576 | 18.712 | 50.91 | 57.6 | 53.27 | 54.4 | 54.96 | 51.6 | 56.26 | 49.3 | 59.85 | 46.3 | 66.60 | 44.0 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 1,048,576 | 35.319 | 57.83 | 53.7 | 62.48 | 50.4 | 64.37 | 48.4 | 67.13 | 44.7 | 70.41 | 41.6 | 80.99 | 39.9 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 1,048,576 | 26.237 | 36.17 | 83.9 | 39.58 | 88.3 | 41.08 | 79.6 | 44.85 | 75.1 | 49.41 | 70.7 | 58.80 | 66.4 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 131,072 | 24.421 | 41.19 | 44.3 | 44.83 | 44.4 | 47.98 | 41.5 | 54.95 | 36.3 | Limit | Limit | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 262,144 | 26.812 | 41.99 | 45.3 | 47.89 | 41.6 | 53.35 | 37.3 | 64.24 | 31.0 | 86.26 | 23.1 | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 131,072 | 16.705 | 23.64 | 79.3 | 26.10 | 76.3 | 28.17 | 70.2 | 32.48 | 61.3 | Limit | Limit | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 1,048,576 | 16.632 | 23.42 | 78.3 | 25.90 | 76.9 | 28.10 | 70.9 | 32.29 | 61.7 | 41.41 | 48.1 | 58.40 | 34.1 |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 262,144 | 30.856 | 140.96 | 34.1 | 157.38 | 31.8 | 168.79 | 27.5 | 192.01 | 24.1 | 243.65 | 16.1 | Limit | Limit |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 262,144 | 21.066 | 79.19 | 63.0 | 88.52 | 56.5 | 95.09 | 47.9 | 108.53 | 36.1 | 135.91 | 28.9 | Limit | Limit |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 262,144 | 15.235 | 47.35 | 107.7 | 53.31 | 92.0 | 57.84 | 78.7 | 64.84 | 65.6 | 79.65 | 47.8 | Limit | Limit |

<!-- END context -->

Engine prefill seconds at each context bucket, one uncached sample per cell:

<!-- BEGIN context-prefill -->

| Model / weights | Nodes | Options | 0 | 32K | 64K | 128K | 256K | 512K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 0.383 | 34.357 | 74.549 | 166.938 | Limit | Limit |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 0.389 | 33.915 | 72.716 | 162.731 | 395.664 | Limit |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 0.236 | 21.799 | 48.172 | 114.093 | 303.271 | 925.281 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 0.331 | 24.243 | 52.909 | 123.326 | 317.875 | Limit |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | 0.322 | 24.234 | 53.010 | 123.934 | 317.535 | 928.123 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 0.340 | 18.241 | 38.461 | 85.217 | 201.651 | Limit |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 0.285 | 17.776 | 37.215 | 85.029 | 199.488 | Limit |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 0.210 | 30.420 | 64.056 | 138.480 | 314.459 | Limit |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 0.210 | 30.052 | 63.962 | 137.367 | 312.204 | 875.869 |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | 0.345 | 19.703 | 38.793 | 86.157 | 204.820 | Limit |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | 0.244 | 30.213 | 63.513 | 137.791 | 312.403 | Limit |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | 0.229 | 35.545 | 75.142 | 163.189 | 365.062 | Limit |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | 0.158 | 24.077 | 51.506 | 112.922 | 261.324 | Limit |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | 0.301 | 17.514 | 36.785 | 81.839 | 195.498 | Limit |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | 0.247 | 16.869 | 35.678 | 79.916 | 191.961 | Limit |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | 0.821 | 70.959 | 153.616 | Limit | Limit | Limit |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV, 8 slots | 0.840 | 75.099 | 161.000 | 366.933 | Limit | Limit |
| GLM-5.3 int4/int8 | 4 | 256K FP4 KV, 8 slots | 0.787 | 74.567 | 159.966 | 365.499 | 898.244 | Limit |
| GLM-5.3 NF4I8 H32 g128 | 4 | 272K FP8 KV, 8 slots | 0.945 | 74.341 | 160.953 | 364.771 | 893.299 | Limit |
| GLM-5.3 Mixed346 H32 A8 g128 | 4 | 272K FP8 KV, 8 slots | 0.715 | 74.520 | 161.336 | 364.648 | 893.491 | Limit |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 0.291 | 19.131 | 43.751 | 126.461 | Limit | Limit |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 0.294 | 18.712 | 40.931 | 105.382 | 367.554 | 1908.016 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 0.281 | 35.319 | 89.437 | 265.781 | 984.962 | 4679.601 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 0.190 | 26.237 | 70.847 | 225.675 | 897.362 | 4473.733 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 0.450 | 24.421 | 63.481 | 182.487 | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 0.464 | 26.812 | 72.294 | 218.280 | 725.764 | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 0.223 | 16.705 | 41.425 | 113.534 | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 0.231 | 16.632 | 41.529 | 113.838 | 346.439 | 1160.847 |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 0.692 | 30.856 | 71.119 | 180.085 | 512.813 | Limit |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 0.396 | 21.066 | 47.196 | 114.381 | 307.051 | Limit |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, 256K BF16 KV, 8 slots | 0.228 | 15.235 | 33.434 | 77.885 | 199.450 | Limit |

<!-- END context-prefill -->

### YaRN retrieval

The earlier YaRN check planted numeric codes at 5%, 25%, 50%, 75% and 95% of each document. The response budget was 768 tokens including reasoning. A hit means the expected code appeared after removing non-digit characters from the response. The two-stream check measures completion; admission can queue a request when the shared KV pool is full.

| Target tokens | Actual prompt tokens | Retrieval hits | Repeated prefix cached | Concurrent requests completed |
|---|---|---|---|---|
| 262,144 | 261,388 | 5/5 | 99.997% | 2/2 |
| 524,288 | 522,619 | 5/5 | 99.999% | 2/2 |

The ~4K control retrieved 5/5 codes. Prefix reuse repeats the final probe. [Full retrieval and concurrency record](../benchmarks/results/2026-09-22-current/raw/qwen-yarn-w2/long/default/retrieval.json).

## Selection microbenchmarks

| Setting | Qwen QSA / DeepSeek CSA2 |
|---|---|
| Hardware | Idle GB10; CUDA graphs |
| Iterations | Five warmups, thirty timed iterations per shape |
| QSA validation | Scores and selections compared with a host oracle |
| CSA2 validation | Selections compared with a host sort of production score keys; score arithmetic is not independently checked |
| Cache conditions | Repeated warm inputs; cold samples evict L2 |
| Scope | Kernel timing, separate from end-to-end service throughput |

[Selection timings and correctness results](../benchmarks/results/2026-09-22-current/README.md#selection-microbenchmarks).

## Historical external workload

This separate llama-benchy check predates the matrix refresh and uses temperature 1.

| Model / deployment | Workload | Repetitions | Decode tok/s | Prefill tok/s | Record |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash, four nodes | `pp2048 / tg128`, checkpoint sampling defaults, thinking on | 3 × 40 runs | 64.3–65.1 | About 1,380 | [DeepSeek-V4 notes](../benchmarks/results/2026-10-01-deepseek-v4-flash/notes.md) |

## Reproduce

Build source revision `07a32fb` in the primary checkout, preserving existing work when changing revisions. Apply the recorded [launcher swap-policy patch](../benchmarks/results/2026-10-06-prefill-scaling/diagnostics/launcher-process-swap.patch), use the pinned checkpoints and site settings, and reserve the cluster for one benchmark deployment at a time.

```bash
cmake --preset release
cmake --build --preset release -j 4 --target dgpp_serve_app
python3 scripts/prepare_data.py download
```

The saved runner resumes this campaign with its exact recorded binary. Rebuilding changes its version string and hash. Saved launch manifests and the source catalogs identify each retained binary. A separate future campaign needs its own result directory, binary and tool hashes, and initial checkpoint inventory for cleanup; resuming this session keeps its original inventory.

To inspect or resume the recorded campaign:

```bash
python3 benchmarks/results/2026-10-06-prefill-scaling/run.py --plan
python3 benchmarks/results/2026-10-06-prefill-scaling/run.py
python3 benchmarks/results/2026-10-06-prefill-scaling/summarize.py --require-complete
python3 benchmarks/results/2026-10-07-prefill-completion/results.py --require-complete
python3 benchmarks/results/2026-10-07-prefill-completion/update_document.py
python3 benchmarks/results/2026-10-07-prefill-completion/update_document.py --check
python3 benchmarks/results/2026-10-06-prefill-scaling/diagnostics/audit_measurements.py --require-complete
python3 benchmarks/results/2026-10-06-prefill-scaling/diagnostics/audit_document.py
```

The runner checks binary and tool hashes, preserves failed attempts, and skips verified measurements. It enables `DGPP_NO_SWAP=1` for every rank, starts and stops each deployment, runs the default and explicit decode modes, and saves commands, configurations and request-level results. Context measurements use `.venv/bin/python` with `requirements-tools.txt` installed.

The [original measurement audit](../benchmarks/results/2026-10-06-prefill-scaling/diagnostics/measurement-audit.json) and [completion audit](../benchmarks/results/2026-10-07-prefill-completion/measurement-audit.json) reconcile request counts, token usage and timing calculations, and check repetition grids, uncached inputs and context limits. Both must pass. Apply the completion updater after regenerating the original matrix; it incorporates the diagnostic prefill results and all validated replacements. The [diagnostic instructions](../benchmarks/results/2026-10-07-prefill-followup/README.md#reproduction-and-service-state) and [completion instructions](../benchmarks/results/2026-10-07-prefill-completion/README.md#reproduce-and-update) identify the binaries and retained evidence.

Disk checks and checkpoint cleanup are specific to this campaign. The saved initial inventory protects existing models. After every configuration and decode mode for a newly downloaded model passes, the runner removes its downloaded checkpoint and its separate prepacked-weight cache. Disk readings and removals are recorded in `storage.jsonl`.

Resident images created for existing checkpoints are tracked separately. Cleanup requires their model's tests to be complete, the files to be absent from the initial inventory, and no process to be using them.

| Generated cache record | Scope |
|---|---|
| [Qwen BF16 dense](../benchmarks/results/2026-10-06-prefill-scaling/diagnostics/new-bf16-resident-cache.json) | 72.6 GiB removed after the single-node variant passed |
| [Other resident images](../benchmarks/results/2026-10-06-prefill-scaling/diagnostics/generated-resident-cleanup.jsonl) | Inspections and completed removals; each record states whether removal was requested |

To run one deployment or a subset of stages:

```bash
python3 benchmarks/results/2026-10-06-prefill-scaling/run.py \
  --only qwen-nvfp4-w2 --stages greedy prefill context modes
```

To measure an already running server:

```bash
python3 scripts/timed_load.py 127.0.0.1 18080 \
  --concurrency 1,2,4,8 --classes all --repeat 3 --max-tokens 256 \
  --json-out /tmp/decode.json
.venv/bin/python scripts/serve_context_matrix.py 127.0.0.1 18080 \
  --model nvidia/Qwen3.8-Flash-Next-NVFP4 --tag unique-run-id \
  --json-out /tmp/context.json
```

Use the configured host and port. Add `--think` for the full GLM-5.3 serving sweep. Change the context tag for each new cold run. The historical [runner](../benchmarks/results/2026-09-22-current/run.py), [isolated evaluator](../benchmarks/results/2026-09-22-current/isolated_eval.py) and [long-context client](../benchmarks/results/2026-09-22-current/long_context.py) reproduce earlier campaigns; quality evaluation requires the Docker image pinned in their manifests.
