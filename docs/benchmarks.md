# Benchmarks

Serving throughput, decode modes and context scaling on the GB10 cluster. Each row identifies a model, weight format, node count and deployment configuration.

Every row was measured at the source revision and with the settings its
table names. The serving defaults changed on 2026-10-12: the prefill chunk
now grows to fit the latency targets instead of staying at a fixed token
budget, and `compact_batches` is on (see
[deployment defaults](../deploy/README.md#shared-scheduler-defaults)). The
rows below have not been re-measured since, so they describe deployments
with the fixed prefill budgets listed per row, not a current default
deployment. The Qwen3.8-Flash-Next QAD row (2026-10-14), the six
Qwen3.8-27B rows (2026-10-16), the Swift 1.5 two-node row (2026-10-17),
and the FP4-Local row (2026-10-21) are measured with the adaptive defaults.
The measurements behind that change are in
[PR #112](https://github.com/HawkBearPig/dgpp/pull/112). When defaults
change again, keep each row's recorded configuration with its numbers.

| Measurement set | Date (UTC) | Source revision | Scope |
|---|---|---|---|
| DeepSeek-V4.1-Flash-1017 EXL3 3.5bpw | 2026-10-23 | `b423d5db` plus the day's second program in the working tree (binary `9a179b15a6de02fa`) | The EXL3 3.5 bpw template as shipped on four nodes after the day's second program — sixteen slots on the family's 96-row decode batch (the morning's twelve on 64), the balanced expert slices, the fused split reduce, the deep slot ring, the 128-row dense tile, a 4,096-token first prefill chunk, the expert-run slot form with its 64-row run groups, the attention split groups merged in L2, the tiled router and mHC dots from sixteen rows, the tensor-core GEMV's one split chain to 128 rows, the verify schedule's row and base costs measured on the pack (2.95 / 19.9 ms) and the five-draft DSpark block under the schedule's three options (1, 3, 5; `engine.mtp_depth` 5, `engine.mtp_schedule_options` 3) — in one `serve_campaign.py run` on 2026-10-24 00:06–00:54 UTC (with the five DSpark mode boots: plain and fixed depths 1/2/3/5 through `--no-mtp-schedule`): five classes at C1/C2/C4/C8, three cold probes each at 512/2K/8K/32K, the context matrix to 512K, the 300-item quality set, the API, greedy-transcript and isolation checks, every rank at zero swap, zero cached tokens in the 12 cold probes, matching operation streams in 7/7 launches. The day's earlier campaigns (six slots on `c7efb833f88a06e3`, twelve on `dde85abfaa953cc1` and `6f8bccfa5b70efef`, sixteen at depth 4 on `9a179b15a6de02fa`) are superseded by this row set; the [model card](model_cards/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw.md) keeps them beside the release's |
| Flint FP8/NVFP4 | 2026-10-22 | `b3f9b538` (binary `15ac3372bb2e3230`) | The three templates as shipped (one, two and four nodes), every row from one `serve_campaign.py run` per template: C1/C2/C4/C8, twelve cold prefill probes, the context matrix to the 262,144 positional ceiling, the 300-item quality set with the first use of the retry beyond the cap, the API, greedy-transcript and isolation checks; every rank at zero swap, zero cached tokens in the cold probes, matching operation streams; the resident image cache off (boot path only). An earlier run of the one-node template on `4709d8d6` exposed the model-opened `<think>` defect fixed in `b3f9b538` and is not tabled |
| Qwen3.8-Flash-Next FP4-Local | 2026-10-21 | `7c3086939006` (binary `c8f6aaf59ce5d08a`); HumanEval scorer `9b7439b5` | The one-node template as shipped, all inference from one `serve_campaign.py run`: C1/C2/C4/C8, twelve cold prefill probes, context through 512K, 164 HumanEval / 300 GSM8K / 100 extraction cases, API and transcript checks. Saved HumanEval answers were re-scored after fixing supplied-context handling; answers and inference are unchanged. Both boots had zero swap and matching operation streams. |
| DeepSeek-V4.1-Flash-1017 | 2026-10-20 | `283a52f7` (binary `f4e9e9cdb16acf86`) | The 1017 template as shipped on four nodes, every row from one `serve_campaign.py run` (with the five DSpark mode boots: plain and fixed depths 1/2/3/5 through `--no-mtp-schedule`): five classes at C1/C2/C4/C8, three cold probes each at 512/2K/8K/32K, the context matrix to 512K, the 300-item quality set, the API, greedy-transcript and isolation checks, every rank at zero swap, zero cached tokens in the 12 cold probes, matching operation streams. The six-slot template's C8 cell is six live requests and two queued, with a three-repeat spread the [model card](model_cards/DeepSeek-V4.1-Flash-1017.md) measures ("The six-slot template at C6 and C8"); the 2026-10-19 rows (`0d69d565`, binary `9350b38a1e5b4908`) and the `engine.index_topk` 2048 set are priced in the model card, not tabled here |
| Qwen3.8-27B refresh | 2026-10-17/18 | `8e5e7d42` (the NVFP4 rows, 2026-10-17) / `aa88b8f7` (the FP8 rows, 2026-10-18, on the fp8 K/V pool) | FP8 and NVFP4 at worlds 1/2/4 on the templates as shipped (the fp8 K/V pool — the mixed release's shipped scales, unit scales on the FP8 release — the fp8 prefill attention and the native activation recipe on both; the pools and prefix arenas sized to the node), five classes at C1/C2/C4/C8, cold 2K/8K/32K prefill, the context matrix to 256K, the 300-item quality set, the API and greedy-transcript checks, the adaptive prefill defaults |
| Swift 1.5 Qwen3.8-Flash-Next NVFP4 | 2026-10-17 | `1cc9fefc` | Community checkpoint `a00c7c2f` on two nodes, MTP4, five classes at C1/C2/C4/C8, three cold probes each at 2K/8K/32K, adaptive prefill within 4K busy/idle caps |
| Qwen3.8-Flash-Next QAD export | 2026-10-14 | `347c386f` plus the QAD export working tree | The rows for the local-inference-lab QAD export on one node (C1–C8, cold prefill, decode modes, 0–128K context, quality), measured with the adaptive prefill defaults of 2026-10-12 |
| Qwen3.8-27B quant comparison | 2026-10-11 | `d12e7e9f` decode / `510542a4` prefill | Matched FP8 / NVFP4 / FP8 at worlds 1/2/4, five classes at C1/C2/C4/C8, cold prefill, all-rank profiling and four-node context decode |
| GLM-5.3 Mixed346 checkpoint | 2026-10-10 | `41e3216d` plus the Mixed346 working tree | The rows for GLM-5.3 Mixed346 GPTQ H32 A8 g128 (C1–C8, cold prefill, context buckets, quality); measured on the four nodes with the fp8-KV template after the kernel round of 2026-10-10 (the int8 tensor-core tile, the 3/6-bit cooperative pieces, the fused norm); the Int4/Int8 and NF4I8 references are the 2026-10-09 rows on the same binary lineage |
| GLM-5.3 NF4I8 checkpoint | 2026-10-09 | `41e3216d` plus the NF4I8 working tree | The rows for GLM-5.3 NF4I8 H32 g128 (C1–C8, cold prefill, context buckets, quality), measured without the rank-0 CUDA-connections setting; the same binary measures the int4/int8 fp8-KV deployment at 3.896 / 17.866 / 76.439 s cold prefill and the same decode steps (its table rows are the 2026-10-07 binary's: 3.555 / 17.188 / 74.270) |
| Prefill benchmark completion | 2026-10-07–08 | `90ebbf1` plus both recorded kernel patches | All 46 targeted groups complete across seven affected configurations |
| Prefill efficiency follow-up | 2026-10-07 | `90ebbf1` plus recorded kernel patches | Matched before/after cold-prefill comparisons for two-node GLM Flash, full-GLM FP4 KV and two-node MiMo; corrected results retained below |
| Matrix refresh | 2026-10-06–07 | [`a3ebc21`](https://github.com/HawkBearPig/dgpp/commit/a3ebc21a7c245eb736cc799fcf87a2c97ed8f7c9); final runs use [`07a32fb`](https://github.com/HawkBearPig/dgpp/commit/07a32fb912a264de1ea52801ffee127d8bc05235) | All 29 deployments complete: C1/C2/C4/C8, decode modes, cold prefill and context scaling |
| DeepSeek-V4.1 prefill batching | 2026-10-07 | [`07a32fb`](https://github.com/HawkBearPig/dgpp/commit/07a32fb912a264de1ea52801ffee127d8bc05235) | Both DeepSeek-V4.1 deployments were rerun. Unaffected model-family measurements were retained |
| Full-GLM score fusion | 2026-10-07 | [`21b91a3`](https://github.com/HawkBearPig/dgpp/commit/21b91a39f999cef0c94579a3240b62b07854a80b), [`0b10cbe`](https://github.com/HawkBearPig/dgpp/commit/0b10cbeaed1b7c51d854ec2f3a23729cb7b91c35) | Affected full-GLM prefill groups were rerun. Unchanged measurements from the earlier fixed binary were retained |
| Before the scaling fix | 2026-10-06 | `50309fd` + route-trace fix | Verified short-prompt GLM Flash measurements use an unchanged code path and pass the reuse audit. Earlier affected measurements are retained only as baselines |
| Original campaign | 2026-09-22 | `a3ed8994a34c` | GLM, Qwen Flash-Next and DeepSeek-V4.1 quality/correctness |
| MiMo | 2026-09-22/23 | `a895db96bb17` + recorded changes | Quality/correctness |
| Qwen AutoRound | 2026-09-29 | Campaign revisions recorded per run | Quality/correctness |
| DeepSeek-V4 | 2026-10-02 | `904f76c8519a` + recorded changes | Historical quality/correctness and llama-benchy |
| Qwen 27B | 2026-10-05 | Morning: `5b57904274d3` + recorded changes; evening: `dc52f66` | Quality/correctness and earlier performance |

[Performance](#serving-performance) · [Configuration](#hardware-and-configuration) · [Method](#workload-and-timing) · [Decode modes](#decode-modes) · [Quality](#quality-and-correctness) · [Long context](#long-context) · [Reproduce](#reproduce)

## Serving performance

Rates are tokens per second. Ranges span the medians of five prompt classes, with three repetitions per class. **C** is the number of concurrent client requests. Every deployment is tested at **C1, C2, C4 and C8**.

| Metric | Timing scope |
|---|---|
| Decode tok/s (C1 to C8) | The engine's decode steps only: every request's generated tokens over the decode time the group took; prefill, its first output token and queueing excluded. The throughput figure; the serving table's concurrency columns |
| Wall tok/s (C1 to C8) | Total completed output divided by the request group's wall time, prefill and queueing included: the service-level view, in its own table below the serving table, never in the decode columns' place |
| Cold prefill seconds | Median engine prefill time of three distinct uncached prompts at each approximate target length; the prefill figure, reported separately from decode |

Client concurrency and request slots are different. A C8 test on a two-slot deployment includes the time requests spend waiting for a slot. The deployment's slot count, KV pool and decode mode stay fixed throughout its concurrency sweep.

The seven configurations affected by the latest prefill fixes have updated serving and context measurements. Cold-prefill medians use three uncached probes, with two exceptions: the diagnostic follow-up rows listed below use two, and the GLM Flash 256K row uses two at ~2K/~8K and three at ~32K. Both two-node GLM Flash configurations use a 2,048-token idle budget.

Five rows were refreshed on 2026-10-13 with the draft-head screen (`engine.draft_head: "screen"`, now set in those templates; the drafts read a 2-bit copy of the vocabulary head and only the candidate blocks exactly, the verify reads the exact head, greedy transcripts matched the exact-head legs 4/4 and acceptance per class was identical): the C1 engine and wall cells, the class table and the default MTP1 column for Qwen3.8-Flash-Next NVFP4 on one node (FP8 dense), Qwen3.8-Flash-Next FP8 on four nodes, GLM-5.3-Flash NVFP4/FP8 on four nodes (768K), GLM-5.3 int4/int8 on four nodes (120K BF16 KV) and Qwen3.8-27B NVFP4 on four nodes, plus the MTP2 and MTP3 columns of the two Flash-Next rows. C2–C8, cold prefill (the screen does not touch prefill) and every other row keep their earlier evidence; the other templates that set the screen were not re-measured.

The Qwen3.8-27B rows were re-measured on 2026-10-17/18 at `8e5e7d42` (the NVFP4 rows, 2026-10-17) / `aa88b8f7` (the FP8 rows, 2026-10-18, on the fp8 K/V pool) on the templates as shipped, with the adaptive prefill defaults, one boot per template for the serving, long-context, quality and transcript rows (and a second boot without the drafter for the plain transcripts): the NVFP4 rows run the mixed release's fp8 K/V cache (`kv_dtype: "fp8"`), the prefill attention on the fp8 tensor cores (`prefill_attention: "fp8"`) and the native activation recipe (`prefill_activations: "native"`); the FP8 rows run the native activation recipe with the fp8 K/V pool at unit scales and the fp8 prefill attention; both carry the 2026-10-15/16 attention prefill kernels, and since 2026-10-17 every template's K/V pool and prefix arena are sized to the node (the configuration table has each). The FP8 templates took the native recipe during the 2026-10-16 refresh: measured first on the bf16 chain, the FP8 one-node cold prefill read 20 % slower than the 2026-10-11 row (2.70 / 8.30 / 35.3 s at 2K / 8K / 32K) because an idle prompt now runs in 4,096-token chunks and the chain re-dequantized every FP8 matrix into its bf16 bridge for each chunk (the same template and binary under the old one-pass budgets read 2.22 / 6.96 / 29.5 s in a control probe); the checkpoint's own blockwise fp8 product needs no bridge and the 300-item evals read the same under both chains (161/164, 294/300, 100/100). These rows replace the 2026-10-16 set (the 262K pools) and the 2026-10-11 matched set (fixed prefill budgets, the BF16 cache, the bf16 chain); compare within this set. The DFlash2 drafter draws its proposals, so a class's single-request rate moves run to run by several percent. Cold probes report zero cached tokens; every leg's ranks ran in the launcher's swap-limited scopes (`DGPP_NO_SWAP=1`, [operations](operations.md#memory-on-a-serving-node)), their operation streams matched and no serving process swapped. The NVFP4 legs' greedy transcripts first read different run to run (the drafter and plain decode alike); the cause was the dense W4A4 TMA form's pipeline drain, fixed the same day (`8e5e7d42`), and the NVFP4 rows are from the legs on it. The FP8 rows are from 2026-10-18 (`aa88b8f7`), when the FP8 templates took the fp8 K/V pool at unit scales with the fp8 prefill attention and their pools were re-sized for the halved cache; their earlier legs on the bf16 cache (`e771b7b1`) read 1.94 / 6.57 / 28.1 s cold prefill on one Spark and 400 s at 256K. The decode-mode rows keep their 2026-10-05 evidence.

The Swift 1.5 row was measured on 2026-10-17 with the
[two-node template](../deploy/cluster_qwen-3.8-flash-next_nvfp4-swift15_w2.example.json),
source `1cc9fefc` and checkpoint revision
`a00c7c2ff7029a1d6a63d78e98a1fdca4a576fba`. It retains the contributor's
MTP4 configuration, the exact draft head, a 1M-token shared BF16 KV pool
and a 42 GiB prefix cache. Adaptive prefill uses 4,096-token busy/idle caps.
The release build ran with resident-image caching off, zero-swap scopes on
both ranks and `CUDA_DEVICE_MAX_CONNECTIONS=32` on rank 0. Both rank
operation streams matched at shutdown. Five-second memory samples recorded
zero process and cgroup swap on both ranks, with no OOM events.

All 225 measured decode requests succeeded and their scheduler counters
reconciled; thinking was disabled. Cold-prefill input lengths were
2,022–2,049 / 7,999–8,215 / 32,282–32,344 tokens, with zero cached tokens
in all nine probes. These are absolute measurements of this configuration;
no matched NVIDIA comparison, draft-depth sweep or long-context campaign
was run. See the [checkpoint and serving notes](model_cards/Qwen3.8-Flash-Next-Swift1.5-NVFP4.md).

Decode throughput by concurrency (the engine's decode steps only: every request's generated tokens over the decode time the group took, prefill and queueing excluded) with cold prefill beside it. Rows measured before 2026-10-18 recorded the decode rate at every concurrency but kept only C1; their C2–C8 cells read — until each is re-measured.

The FP4-Local rows use the one-Spark template at `7c3086939006`, checkpoint
`b9ce18a6887e18c089fdea8509a33e63429a3300`, and server SHA-256
`c8f6aaf59ce5d08a54cfa54d31587baa46209ce474deb49789b453e83d082ff9`.
The fabric was otherwise idle; all cold probes had zero cached tokens,
and the original production deployment was restored after the campaign.
C8 means four live requests and four queued on this four-slot template.
The shared QSA scorer now reuses key stripes beyond L2 capacity; the indexer's
underlying quadratic cold-prefill work remains (see the
[checkpoint and shared-path notes](fp4local_plan.md)).

<!-- BEGIN serving -->

| Model / weights | Nodes | Options | C1 decode tok/s | C2 decode tok/s | C4 decode tok/s | C8 decode tok/s | Cold prefill s: ~2K / ~8K / ~32K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 30.6–36.0 | — | — | — | 2.057 / 8.384 / 35.312 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 30.7–36.1 | — | — | — | 1.976 / 8.178 / 35.025 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 58.1–63.7 | — | — | — | 1.182 / 5.075 / 21.843 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 46.7–51.5 | — | — | — | 1.334 / 5.686 / 24.491 |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | 46.8–51.6 | — | — | — | 1.356 / 5.716 / 24.507 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 43.5–51.9 | — | — | — | 1.363 / 4.522 / 18.054 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 31.2–38.0 | — | — | — | 1.392 / 6.116 / 19.055 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 62.3–74.9 | — | — | — | 2.130 / 7.873 / 28.829 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 65.0–75.9 | — | — | — | 1.830 / 7.145 / 28.795 |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | 44.2–61.0 | — | — | — | 1.358 / 4.602 / 18.234 |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | 52.1–96.9 | — | — | — | 1.870 / 7.179 / 28.922 |
| Qwen3.8-Flash-Next NVFP4 (Swift 1.5) | 2 | FP8 dense, 1M BF16 KV, 4 slots, MTP4 | 54.2–106.1 | — | — | — | 1.408 / 4.653 / 16.700 |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | 44.0–52.9 | — | — | — | 2.027 / 8.061 / 33.457 |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | 73.1–86.9 | — | — | — | 1.582 / 5.614 / 23.039 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | 43.4–69.8 | — | — | — | 1.276 / 4.356 / 17.159 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | 43.1–69.0 | — | — | — | 1.182 / 4.077 / 16.682 |
| Qwen3.8-Flash-Next NVFP4 (QAD) | 1 | MXFP8 dense, resident n-gram table, 128K BF16 KV, 4 slots | 41.1–50.2 | — | — | — | 1.412 / 5.236 / 21.477 |
| Qwen3.8-Flash-Next FP4-Local | 1 | MXFP8 dense, mapped n-gram table, 1M FP8 KV, 4 slots, MTP3 + lookup | 35.3–75.9 | 58.1–105.2 | 67.3–122.3 | 65.6–133.1 | 1.623 / 5.748 / 23.047 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | 28.3–31.4 | — | — | — | 3.493 / 16.219 / 70.249 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV, 8 slots | 27.0–30.4 | — | — | — | 3.555 / 17.188 / 74.270 |
| GLM-5.3 int4/int8 | 4 | 256K FP4 KV, 8 slots | 27.4–30.5 | — | — | — | 3.770 / 17.180 / 74.988 |
| GLM-5.3 NF4I8 H32 g128 | 4 | 272K FP8 KV, 8 slots | 27.1–30.6 | — | — | — | 3.872 / 17.915 / 76.522 |
| GLM-5.3 Mixed346 H32 A8 g128 | 4 | 272K FP8 KV, 8 slots | 28.1–31.6 | — | — | — | 3.906 / 17.943 / 76.714 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 42.1–77.9 | — | — | — | 1.483 / 5.037 / 19.869 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 41.8–78.2 | — | — | — | 1.472 / 4.946 / 19.545 |
| DeepSeek-V4.1-Flash-1017 MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 53.5–89.4 | 75.5–121.9 | 94.7–130.3 | 83.8–139.0 | 1.006 / 2.638 / 10.024 |
| DeepSeek-V4.1-Flash-1017 EXL3 3.5bpw | 4 | 1M compressed KV, 16 slots | 70.1–134.2 | 95.7–182.4 | 127.8–206.6 | 165.0–247.7 | 1.188 / 3.446 / 13.331 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 40.8–73.6 | — | — | — | 2.006 / 7.586 / 36.786 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 65.7–128.2 | — | — | — | 1.309 / 5.226 / 27.327 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 43.4–49.0 | — | — | — | 1.567 / 5.512 / 25.825 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 40.5–47.0 | — | — | — | 1.559 / 5.573 / 27.945 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 74.4–85.1 | — | — | — | 1.105 / 3.966 / 17.627 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 74.8–85.7 | — | — | — | 1.087 / 3.887 / 17.454 |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, native activations, 768K FP8 KV, fp8 prefill attention, 8 slots | 20.7–48.1 | 34.9–85.2 | 59.3–122.8 | 92.6–212.5 | 1.927 / 6.507 / 27.003 |
| Qwen3.8-27B NVFP4 | 1 | DFlash2 drafter, FP8 head, 1M FP8 KV, fp8 prefill attention, 8 slots | 25.2–59.8 | 42.5–107.6 | 73.8–147.3 | 118.5–244.0 | 1.198 / 3.943 / 16.634 |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 37.0–86.6 | 59.3–147.5 | 97.8–201.0 | 151.3–329.8 | 1.505 / 5.138 / 20.883 |
| Qwen3.8-27B NVFP4 | 2 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 44.8–105.4 | 72.8–177.5 | 122.2–224.7 | 175.2–368.9 | 1.198 / 4.232 / 17.285 |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 57.7–136.8 | 92.6–219.0 | 142.6–279.3 | 219.4–405.6 | 1.246 / 4.389 / 17.559 |
| Qwen3.8-27B NVFP4 | 4 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 64.0–165.5 | 113.3–257.7 | 177.9–304.3 | 244.7–430.5 | 1.058 / 3.897 / 15.646 |
| Flint FP8/NVFP4 | 1 | 512K BF16 KV, 8 slots | 56.2–58.0 | 99.9–101.2 | 146.2–148.0 | 197.8–204.0 | 0.484 / 1.951 / 12.077 |
| Flint FP8/NVFP4 | 2 | 512K BF16 KV, 8 slots | 104.3–105.2 | 175.1–177.5 | 253.7–258.0 | 342.5–352.5 | 0.413 / 1.570 / 8.661 |
| Flint FP8/NVFP4 | 4 | 512K BF16 KV, 8 slots | 154.8–158.9 | 256.2–260.8 | 393.9–399.7 | 524.8–542.9 | 0.358 / 1.297 / 6.468 |

<!-- END serving -->

Request wall throughput, the service-level view: total completed output over the request group's wall time, prefill and queueing included.

<!-- BEGIN wall -->

| Model / weights | Nodes | Options | C1 wall tok/s | C2 wall tok/s | C4 wall tok/s | C8 wall tok/s |
| --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 29.6–34.5 | 43.0–46.8 | 55.3–61.2 | 56.4–61.8 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 29.6–34.6 | 43.1–46.9 | 44.7–48.1 | 44.5–47.2 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 56.0–61.1 | 79.5–85.4 | 106.2–117.7 | 105.7–116.0 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 44.8–49.2 | 60.1–64.4 | 77.5–86.2 | 77.6–84.9 |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | 44.9–49.3 | 60.2–64.4 | 77.8–86.3 | 77.5–85.8 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 42.4–50.1 | 62.1–73.1 | 86.5–97.7 | 84.3–98.9 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 30.6–37.1 | 52.0–60.9 | 74.2–84.8 | 73.5–85.6 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 60.5–72.5 | 93.3–109.3 | 126.1–146.1 | 124.9–147.3 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 63.0–73.0 | 94.5–110.2 | 94.7–111.2 | 93.0–109.6 |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | 42.9–58.4 | 56.1–76.8 | 80.6–106.3 | 78.3–107.9 |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | 50.6–92.3 | 71.9–136.4 | 100.8–161.5 | 95.0–177.6 |
| Qwen3.8-Flash-Next NVFP4 (Swift 1.5) | 2 | FP8 dense, 1M BF16 KV, 4 slots, MTP4 | 52.7–100.2 | 85.0–155.8 | 116.5–200.1 | 112.5–205.1 |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | 43.0–51.3 | 68.2–81.3 | 90.6–103.0 | 81.0–100.3 |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | 70.8–83.5 | 110.6–128.3 | 141.5–162.2 | 139.8–164.1 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | 42.3–66.9 | 57.6–90.3 | 76.5–115.7 | 76.0–124.8 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | 42.1–66.0 | 55.6–91.8 | 78.9–120.1 | 77.1–124.4 |
| Qwen3.8-Flash-Next NVFP4 (QAD) | 1 | MXFP8 dense, resident n-gram table, 128K BF16 KV, 4 slots | 40.1–48.6 | 63.5–74.6 | 87.8–99.6 | 86.6–100.8 |
| Qwen3.8-Flash-Next FP4-Local | 1 | MXFP8 dense, mapped n-gram table, 1M FP8 KV, 4 slots, MTP3 + lookup | 34.6–72.0 | 56.5–99.6 | 65.8–117.2 | 65.0–130.2 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | 26.9–29.5 | 35.7–38.4 | 46.1–49.7 | 52.9–55.7 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV, 8 slots | 25.8–28.7 | 36.0–38.5 | 46.0–49.3 | 53.2–56.6 |
| GLM-5.3 int4/int8 | 4 | 256K FP4 KV, 8 slots | 26.1–28.8 | 36.4–37.7 | 45.4–49.6 | 52.4–56.9 |
| GLM-5.3 NF4I8 H32 g128 | 4 | 272K FP8 KV, 8 slots | 25.9–28.9 | 36.4–39.0 | 46.2–49.3 | 52.6–56.4 |
| GLM-5.3 Mixed346 H32 A8 g128 | 4 | 272K FP8 KV, 8 slots | 26.8–29.8 | 37.3–39.1 | 47.3–52.0 | 56.0–60.0 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 40.9–73.2 | 62.9–102.0 | 79.1–109.3 | 66.8–95.3 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 40.8–73.7 | 62.4–102.2 | 78.8–111.2 | 67.1–97.9 |
| DeepSeek-V4.1-Flash-1017 MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 52.0–84.6 | 73.4–115.8 | 92.2–125.2 | 81.7–132.7 |
| DeepSeek-V4.1-Flash-1017 EXL3 3.5bpw | 4 | 1M compressed KV, 16 slots | 67.6–124.3 | 92.9–170.4 | 124.3–195.6 | 160.3–235.8 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 39.5–68.6 | 54.0–93.0 | 68.2–101.4 | 65.7–97.3 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 63.1–117.2 | 89.8–157.7 | 120.9–169.3 | 109.5–158.0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 41.5–46.2 | 57.0–63.6 | 75.1–84.7 | 74.4–83.2 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 38.8–44.5 | 54.9–62.1 | 74.9–84.1 | 73.1–82.1 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 70.9–80.5 | 99.3–114.0 | 135.5–146.5 | 130.9–147.7 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 71.5–80.9 | 99.4–115.1 | 136.1–148.4 | 131.3–148.7 |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, native activations, 768K FP8 KV, fp8 prefill attention, 8 slots | 20.4–46.2 | 34.3–81.1 | 57.5–114.3 | 88.7–189.8 |
| Qwen3.8-27B NVFP4 | 1 | DFlash2 drafter, FP8 head, 1M FP8 KV, fp8 prefill attention, 8 slots | 24.8–57.8 | 41.8–102.3 | 71.9–138.6 | 113.0–222.4 |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 36.2–81.9 | 57.7–137.3 | 94.4–186.4 | 144.2–293.8 |
| Qwen3.8-27B NVFP4 | 2 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 43.8–99.0 | 71.0–165.0 | 118.2–209.8 | 167.0–332.3 |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 55.9–125.2 | 89.4–197.4 | 137.7–260.1 | 208.7–365.9 |
| Qwen3.8-27B NVFP4 | 4 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 62.0–150.2 | 108.6–230.4 | 171.3–283.6 | 232.7–390.0 |
| Flint FP8/NVFP4 | 1 | 512K BF16 KV, 8 slots | 54.8–56.9 | 97.0–99.1 | 142.7–144.7 | 193.1–199.5 |
| Flint FP8/NVFP4 | 2 | 512K BF16 KV, 8 slots | 101.2–101.8 | 169.4–171.9 | 247.0–251.2 | 333.5–344.0 |
| Flint FP8/NVFP4 | 4 | 512K BF16 KV, 8 slots | 149.1–152.6 | 245.9–250.2 | 381.8–387.3 | 511.5–528.4 |

<!-- END wall -->

Configuration labels use **K = 1,024 tokens**. Nodes is the tensor-parallel world size; KV is the shared key/value-cache token pool.

The cold-prefill follow-up found three avoidable costs. The table above incorporates the corrected cold-prefill measurements. These matched comparisons quantify each change, with two cold probes per length:

| Deployment | Change | ~8K prefill before → after |
|---|---|---:|
| GLM Flash, 2 nodes, 160K FP8 KV | Idle chunk: 256 → 2,048 tokens | 19.42 → 8.38 s |
| GLM Flash, 2 nodes, 256K FP8 KV | Idle chunk: 256 → 2,048 tokens | 19.32 → 8.18 s |
| Full GLM, 4 nodes, FP4 KV | Native FP4 attention-cache conversion | 24.50 → 17.18 s |
| MiMo, 2 nodes, BF16 KV | Route FP8 projections to the existing optimized GEMM | 12.93 → 5.51 s |
| MiMo, 2 nodes, FP8 KV | Same projection dispatch fix | 12.97 → 5.57 s |

The two-node GLM Flash example no longer sets a fixed idle budget; the adaptive
policy can grow its chunks to the same 2,048 tokens. The kernel fixes preserve the tested outputs and require no KV allocation reduction. Full-GLM BF16/FP8 paths are unchanged by these fixes; their published timings remain. Their profiles are dominated by attention, packed matrix products and collectives.

The October 7–8 completion campaign added the remaining cold-prefill probes, affected serving and context measurements, and full-GLM FP4 KV decode modes. All 46 measurement groups passed request, rank-identity and zero-swap checks. Native FP8 attention evaluation is tracked in [issue #98](https://github.com/HawkBearPig/dgpp/issues/98).

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
| Excluded runs | The initial swapping attempt and later four-node 512K attempt contribute no results |
| Swap prevention | Every new launch uses `DGPP_NO_SWAP=1`: each rank must have a cgroup swap limit of zero and report zero process and cgroup swap throughout monitoring |
| Earlier retained runs | The first two deployments preceded the cgroup policy; every rank reported zero process swap and no host swap-outs. |
| CUDA connections | `CUDA_DEVICE_MAX_CONNECTIONS=32` on rank 0; peers retain their login environment |

The KV pool is shared across requests. Its capacity does not mean every concurrent request can use that many tokens. The table below lists the effective settings for each measured row.

"Prefill tokens/tick" is the fixed prefill budget each row was measured with:
**busy** while another request was decoding, **idle** otherwise, and **0** for
whole-prompt admission. A current binary does not use fixed values by default
(it grows the chunk to fit the latency targets). To repeat a row on a current
binary, set `engine.prefill_budget_tokens` and
`engine.prefill_idle_budget_tokens` to the listed values and set both
`engine.decode_gap_target_ms` and `engine.first_token_target_ms` to 0; the
kernels have changed since as well, so only the recorded revision reproduces
a row exactly.

<!-- BEGIN config -->

| Model / weights | Nodes | Options | Slots | KV pool tokens | KV format | Prefix cache GiB | Prefill tokens/tick (busy / idle) | Recorded decode mode | Template at measurement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 4 | 163,840 | fp8 | 4.5 | 256 / 2,048 | MTP1 | — |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 2 | 262,144 | fp8 | 2 | 256 / 2,048 | MTP1 | — |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 4 | 786,432 | bf16 | 22 | 256 / 2,048 | MTP1 | — |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 4 | 393,216 | bf16 | 8 | 256 / 2,048 | MTP1 | — |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | 4 | 524,288 | bf16 | 2 | 256 / 2,048 | MTP1 | — |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 4 | 262,144 | bf16 | 3 | 4,096 / 4,096 | MTP1 | — |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 4 | 262,144 | bf16 | 3 | 4,096 / 4,096 | MTP1 | — |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 4 | 262,144 | bf16 | 58 | 256 / 256 | MTP1 | — |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 2 | 565,248 | bf16 | 40 | 256 / 256 | MTP1 | — |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | 4 | 262,144 | bf16 | 3 | 4,096 / 4,096 | MTP2 | — |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | 4 | 524,288 | bf16 | 32 | 256 / 256 | MTP4 | — |
| Qwen3.8-Flash-Next NVFP4 (Swift 1.5) | 2 | FP8 dense, 1M BF16 KV, 4 slots, MTP4 | 4 | 1,048,576 | bf16 | 42 | adaptive (4,096 / 4,096 caps) | MTP4 | [JSON](../deploy/cluster_qwen-3.8-flash-next_nvfp4-swift15_w2.example.json) |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | 4 | 262,144 | bf16 | 6 | 256 / 256 | MTP1 | — |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | 4 | 262,144 | bf16 | 50 | 256 / 256 | MTP1 | — |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | 4 | 262,144 | bf16 | 3 | 4,096 / 4,096 | MTP3 | — |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | 4 | 262,144 | bf16 | 3 | 4,096 / 4,096 | MTP3 | — |
| Qwen3.8-Flash-Next NVFP4 (QAD) | 1 | MXFP8 dense, resident n-gram table, 128K BF16 KV, 4 slots | 4 | 131,072 | bf16 | 1 | adaptive (default) | MTP1 | [JSON](../deploy/cluster_qwen-3.8-flash-next_nvfp4-qad_w1.example.json) |
| Qwen3.8-Flash-Next FP4-Local | 1 | MXFP8 dense, mapped n-gram table, 1M FP8 KV, 4 slots, MTP3 + lookup | 4 | 1,048,576 | fp8 | 3 | adaptive (default) | MTP3 + lookup | [JSON](../deploy/cluster_qwen-3.8-flash-next-fp4local_w1.example.json) |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | 8 | 122,880 | bf16 | 1.0 | 0 / 0 | MTP1 | — |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV, 8 slots | 8 | 212,992 | fp8 | 1.5 | 0 / 0 | MTP1 | — |
| GLM-5.3 int4/int8 | 4 | 256K FP4 KV, 8 slots | 8 | 262,144 | fp4 | 1.5 | 0 / 0 | MTP1 | — |
| GLM-5.3 NF4I8 H32 g128 | 4 | 272K FP8 KV, 8 slots | 8 | 278,528 | fp8 | 1.0 | 0 / 0 | MTP1 | [JSON](../deploy/cluster_glm-5.3_nf4i8_w4.example.json) |
| GLM-5.3 Mixed346 H32 A8 g128 | 4 | 272K FP8 KV, 8 slots | 8 | 278,528 | fp8 | 1.0 | 0 / 0 | MTP1 | [JSON](../deploy/cluster_glm-5.3_mixed346_w4.example.json) |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 6 | 131,072 | FP4 blocks / FP8 window | 14 | 0 / 0 | DSpark adaptive ≤4 | — |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 6 | 1,048,576 | FP4 blocks / FP8 window | 14 | 0 / 0 | DSpark adaptive ≤4 | — |
| DeepSeek-V4.1-Flash-1017 MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 6 | 1,048,576 | FP4 blocks / FP8 window | 14 | adaptive (default) | DSpark adaptive ≤4 | [JSON](../deploy/cluster_deepseek-v4.1-flash-1017_mxfp4-fp8_w4.example.json) |
| DeepSeek-V4.1-Flash-1017 EXL3 3.5bpw | 4 | 1M compressed KV, 16 slots | 16 | 1,048,576 | FP4 blocks / FP8 window | 14 | adaptive (default) | DSpark adaptive ≤5 | [JSON](../deploy/cluster_deepseek-v4.1-flash-1017_exl3-3.5bpw_w4.example.json) |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 4 | 1,048,576 | BF16 (quantized values) | 8 | 256 / 4,096 | DSpark adaptive ≤5 | — |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 6 | 1,048,576 | BF16 (quantized values) | 14 | 256 / 4,096 | DSpark adaptive ≤5 | — |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 4 | 131,072 | bf16 | 1.5 | 0 / 0 | MTP1 | — |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 4 | 262,144 | fp8 | 1.5 | 0 / 0 | MTP1 | — |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 4 | 131,072 | bf16 | 1.5 | 0 / 0 | MTP1 | — |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 4 | 1,048,576 | bf16 | 1.5 | 0 / 0 | MTP1 | — |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, native activations, 768K FP8 KV, fp8 prefill attention, 8 slots | 8 | 786,432 | fp8 | 20 | adaptive (default) | DFlash2 adaptive ≤7 | [JSON](../deploy/cluster_qwen3.8-27b_fp8_w1.example.json) |
| Qwen3.8-27B NVFP4 | 1 | DFlash2 drafter, FP8 head, 1M FP8 KV, fp8 prefill attention, 8 slots | 8 | 1,048,576 | fp8 | 16 | adaptive (default) | DFlash2 adaptive ≤7 | [JSON](../deploy/cluster_qwen3.8-27b_nvfp4_w1.example.json) |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 8 | 2,097,152 | fp8 | 24 | adaptive (default) | DFlash2 adaptive ≤7 | [JSON](../deploy/cluster_qwen3.8-27b_fp8_w2.example.json) |
| Qwen3.8-27B NVFP4 | 2 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 8 | 2,097,152 | fp8 | 32 | adaptive (default) | DFlash2 adaptive ≤7 | [JSON](../deploy/cluster_qwen3.8-27b_nvfp4_w2.example.json) |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 8 | 2,097,152 | fp8 | 48 | adaptive (default) | DFlash2 adaptive ≤7 | [JSON](../deploy/cluster_qwen3.8-27b_fp8_w4.example.json) |
| Qwen3.8-27B NVFP4 | 4 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 8 | 2,097,152 | fp8 | 32 | adaptive (default) | DFlash2 adaptive ≤7 | [JSON](../deploy/cluster_qwen3.8-27b_nvfp4_w4.example.json) |
| Flint FP8/NVFP4 | 1 | 512K BF16 KV, 8 slots | 8 | 524,288 | bf16 | 3 | adaptive (default) | plain | [JSON](../deploy/cluster_flint_fp8-nvfp4_w1.example.json) |
| Flint FP8/NVFP4 | 2 | 512K BF16 KV, 8 slots | 8 | 524,288 | bf16 | 3 | adaptive (default) | plain | [JSON](../deploy/cluster_flint_fp8-nvfp4_w2.example.json) |
| Flint FP8/NVFP4 | 4 | 512K BF16 KV, 8 slots | 8 | 524,288 | bf16 | 3 | adaptive (default) | plain | [JSON](../deploy/cluster_flint_fp8-nvfp4_w4.example.json) |

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
| Qwen NVFP4 (QAD) | [Quantization-aware distilled export](model_cards/Qwen3.8-Flash-Next-NVFP4-QAD.md) served as shipped: MXFP8 dense projections and shared experts, the NVFP4 n-gram table resident on the node, F32 norms; `dense_weights=fp8`, `fp8_head=mma` for the BF16 remainder. Measured with the adaptive prefill defaults |
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

### What a deployment's row set contains

One template, measured as shipped on the bench port, yields every row the tables above and below keep for it. `scripts/serve_campaign.py run --config <template> --out <dir>` runs the set in one command and `rows` prints the cells in table order:

| Table | Measurement | Client |
|---|---|---|
| Decode throughput by concurrency; request wall throughput; single-request decode by class | C1/C2/C4/C8 over the five prompt classes, three repetitions, 256 tokens; the engine's decode-only rate and the wall rate per phase | `timed_load.py` |
| Cold prefill | 512 / 2K / 8K / 32K, three distinct uncached prefixes each, thinking off | `serve_prefill_probe.py` |
| Long context (decode at context, prefill per bucket) | 0 / 32K / 64K / 128K / 256K / 512K parcel documents: one cold request, two identical | `serve_context_matrix.py` |
| Quality | the 300-item HumanEval / GSM8K / extraction set at the deployment's slot count; a response cut at the standard cap is asked once more at the 8,192 retry cap (counted in its own column, never scored) | `serve_eval.py` |
| Transcript checks | the API check; five classes × three greedy transcripts in the shipped mode and again in a second boot without the drafter (the plain/default match); prompt 0 alone against beside slots−1 others (the isolation check) | `serve_api_check.py`, `serve_load.py` |
| Decode modes | a boot per extra mode (`--mode NAME="<knobs>"`): the C1 sweep; the DSpark columns boot with `--mtp-depth N --no-mtp-schedule` (a fixed verify depth; a scheduled template refuses a fixed depth without the second flag) and the plain column with `--no-mtp` | `timed_load.py` |

### Validity

A row set is valid when, recorded beside the cells: the source revision and the binary's hash (one binary for a comparison set; a change that cannot touch a row's path may be noted instead of re-measured); the resolved deployment; every rank's swap at zero and its binary the one under test, before and after the measurements (the ranks run in the launcher's swap-limited scopes, `DGPP_NO_SWAP=1`); zero cached tokens in every cold probe; the ranks' operation streams identical at teardown. The fabric is otherwise idle: no build, no bench and no other world on any node while a row set runs, and production is down for the window and restored once after it. The template's memory plan must leave room at listening (a one-node boot ends 6–8 GiB below its plan, two nodes 4–5, four nodes 2): a rank that swaps invalidates the set.

### Reporting

Decode and prefill are independent attributes and are reported separately: decode tok/s by concurrency as the throughput figure, cold prefill in seconds, and the wall rate only as the labelled service-level table. Cells are medians (three repetitions; three requests per context bucket) and ranges span the five class medians. Every table names its source revision in the measurement-set table; a re-measured row replaces the old one rather than sitting beside it, and raw records stay out of Git.

## Single-request decode by class

Engine tokens/s, greedy; median of three repetitions.

<!-- BEGIN classes -->

| Model / weights | Nodes | Options | Prose | Code | JSON | Math | Chat |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 34.2 | 35.9 | 36.0 | 34.6 | 30.6 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 34.2 | 36.0 | 36.1 | 34.8 | 30.7 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 61.4 | 63.1 | 63.7 | 61.4 | 58.1 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 49.6 | 50.6 | 51.5 | 50.1 | 46.7 |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | 49.5 | 50.7 | 51.6 | 50.2 | 46.8 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 43.5 | 51.1 | 51.9 | 49.1 | 45.9 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 33.9 | 37.3 | 38.0 | 36.1 | 31.2 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 67.5 | 73.6 | 74.9 | 72.3 | 62.3 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 65.0 | 74.4 | 75.9 | 72.3 | 65.0 |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | 47.5 | 59.5 | 61.0 | 54.7 | 44.2 |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | 52.1 | 90.2 | 96.9 | 79.7 | 53.1 |
| Qwen3.8-Flash-Next NVFP4 (Swift 1.5) | 2 | FP8 dense, 1M BF16 KV, 4 slots, MTP4 | 61.4 | 96.9 | 106.1 | 87.8 | 54.2 |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | 45.9 | 51.8 | 52.9 | 49.8 | 44.0 |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | 76.7 | 86.7 | 86.9 | 81.7 | 73.1 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | 43.4 | 59.7 | 69.8 | 60.9 | 45.6 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | 43.1 | 58.3 | 69.0 | 59.3 | 45.6 |
| Qwen3.8-Flash-Next NVFP4 (QAD) | 1 | MXFP8 dense, resident n-gram table, 128K BF16 KV, 4 slots | 43.6 | 48.2 | 50.2 | 47.6 | 41.1 |
| Qwen3.8-Flash-Next FP4-Local | 1 | MXFP8 dense, mapped n-gram table, 1M FP8 KV, 4 slots, MTP3 + lookup | 36.4 | 66.6 | 75.9 | 64.0 | 35.3 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | 30.1 | 31.4 | 31.3 | 30.0 | 28.3 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV, 8 slots | 30.1 | 30.4 | 30.4 | 29.8 | 27.0 |
| GLM-5.3 int4/int8 | 4 | 256K FP4 KV, 8 slots | 29.9 | 29.2 | 30.5 | 29.6 | 27.4 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 42.1 | 64.9 | 77.9 | 62.8 | 46.4 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 41.8 | 65.0 | 78.2 | 62.9 | 46.2 |
| DeepSeek-V4.1-Flash-1017 MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 53.5 | 80.0 | 89.4 | 87.5 | 55.0 |
| DeepSeek-V4.1-Flash-1017 EXL3 3.5bpw | 4 | 1M compressed KV, 16 slots | 70.7 | 92.0 | 134.2 | 111.4 | 70.1 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 40.8 | 62.1 | 73.6 | 58.8 | 43.8 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 65.7 | 114.5 | 128.2 | 94.8 | 70.8 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 43.5 | 46.2 | 49.0 | 46.3 | 43.4 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 42.7 | 45.1 | 47.0 | 45.8 | 40.5 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 75.5 | 81.2 | 85.1 | 84.2 | 74.4 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 75.6 | 81.5 | 85.7 | 84.5 | 74.8 |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, native activations, 768K FP8 KV, fp8 prefill attention, 8 slots | 21.6 | 35.4 | 48.1 | 38.4 | 20.7 |
| Qwen3.8-27B NVFP4 | 1 | DFlash2 drafter, FP8 head, 1M FP8 KV, fp8 prefill attention, 8 slots | 25.2 | 44.3 | 59.8 | 49.8 | 26.3 |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 38.7 | 64.0 | 86.6 | 67.8 | 37.0 |
| Qwen3.8-27B NVFP4 | 2 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 44.8 | 84.3 | 105.4 | 88.3 | 46.7 |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 59.4 | 101.1 | 136.8 | 107.2 | 57.7 |
| Qwen3.8-27B NVFP4 | 4 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 64.0 | 112.1 | 165.5 | 135.9 | 69.2 |
| Flint FP8/NVFP4 | 1 | 512K BF16 KV, 8 slots | 58.0 | 57.5 | 57.2 | 56.2 | 56.6 |
| Flint FP8/NVFP4 | 2 | 512K BF16 KV, 8 slots | 104.3 | 105.2 | 105.1 | 104.8 | 104.5 |
| Flint FP8/NVFP4 | 4 | 512K BF16 KV, 8 slots | 155.3 | 156.9 | 154.8 | 158.9 | 158.8 |

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
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | MTP1 | 39.6–39.7 | 58.1–63.7 | 58.8–68.9 | 53.2–66.3 | Not swept |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | MTP1 | 34.6–34.7 | 46.7–51.5 | 44.9–55.0 | 39.9–50.0 | Not swept |
| GLM-5.3-Flash FP8 | 4 | 512K BF16 KV, 4 slots, 2 GiB prefix cache | MTP1 | 34.3–34.4 | 46.8–51.6 | 44.9–55.0 | 39.9–50.1 | Not swept |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | MTP1 | 31.8–31.9 | 43.5–51.9 | 45.5–66.1 | 44.2–73.8 | Not swept |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | MTP1 | 24.2 | 31.2–38.0 | 36.1–47.7 | 33.0–54.6 | Not swept |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | MTP1 | 46.5–48.0 | 62.3–74.9 | 65.5–91.1 | 61.2–100.5 | Not swept |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | MTP1 | 47.9–48.0 | 65.0–75.9 | 66.2–91.4 | 61.2–100.5 | Not swept |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 1 | FP8 dense, 256K BF16 KV, 4 slots, MTP2 | MTP2 | 32.0–32.1 | 42.4–50.2 | 44.2–61.0 | 40.5–67.7 | Not swept |
| Qwen3.8-Flash-Next NVFP4 (RadixArk) | 2 | FP8 dense, 512K BF16 KV, 4 slots, MTP4 | MTP4 | 46.8–47.5 | 64.1–76.3 | 66.7–91.2 | 64.6–102.7 | 52.1–96.9 |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | MTP1 | 34.6–35.3 | 44.0–52.9 | 46.4–62.7 | 43.4–72.9 | Not swept |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | MTP1 | 51.2–51.5 | 73.1–86.9 | 81.2–113.6 | 79.3–126.6 | Not swept |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | MTP3 | 30.2–30.3 | 42.1–48.9 | 45.3–61.1 | 43.4–69.8 | 35.7–66.5 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | MTP3 | 30.2 | 41.2–48.6 | 43.2–61.1 | 43.1–69.0 | 35.2–66.5 |
| Qwen3.8-Flash-Next NVFP4 (QAD) | 1 | MXFP8 dense, resident n-gram table, 128K BF16 KV, 4 slots | MTP1 | 30.4–30.5 | 41.1–50.2 | 44.1–63.6 | 42.3–73.7 | Not swept |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | MTP1 | 19.7–20.0 | 28.3–31.4 | 28.9–33.6 | 26.3–33.1 | Not swept |
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
| DeepSeek-V4.1-Flash-1017 MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | DSpark adaptive ≤4 | 41.7–41.9 | 50.2–59.5 | 50.1–76.5 | 47.8–82.6 | 39.6–96.7 | 53.5–89.4 |
| DeepSeek-V4.1-Flash-1017 EXL3 3.5bpw | 4 | 1M compressed KV, 16 slots | DSpark adaptive ≤5 | 52.6–53.0 | 65.0–76.6 | 69.2–99.3 | 67.0–110.9 | 58.3–135.1 | 70.1–134.2 |
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

These are the original campaigns' quality results, retained separately from the performance refresh; the six Qwen3.8-27B rows are from the 2026-10-17/18 refresh, and FP4-Local is from the 2026-10-21 campaign on its shipped template. Scores apply to their saved configurations, prompts, evaluator and output limits; current deployment settings may differ.

| Evaluation setting | Value |
|---|---|
| HumanEval | All 164 problems; the maintained `serve_eval.py` preserves supplied helpers/imports and executes each returned code block against the tests in a Python subprocess with a 10-second timeout (`--allow-code-execution`) |
| GSM8K | First 300 test problems |
| Schema extraction | 100 deterministic records |
| Temperature | Zero |
| Concurrency | Deployment request-slot count |
| Output cap | 2,048 tokens; 512 for extraction |
| Retry beyond the cap | Since 2026-10-22 a response the server cut at the cap is asked once more at 8,192 tokens (`serve_eval.py --retry-max-tokens`). The score columns keep the standard-cap verdicts; the last column counts the retried responses that completed within the larger cap, and in parentheses how many of those passed — a model that reaches the right answer only with more tokens is reported, not hidden. Rows measured before the retry show "—" when they had capped responses |
| Reasoning effort | Low with thinking enabled; none with thinking disabled |
| Retained evidence | Raw responses, execution errors, token-cap counts and the retries' replies |

| Model / weights | Nodes | Options | HumanEval | GSM8K | Schema extraction | Responses at token cap | Completed at the 8,192 retry (passed) |
|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash NVFP4/FP8 | 2 | 160K FP8 KV, 4 slots | 157/164 | 292/300 | 100/100 | 0 | 0 |
| GLM-5.3-Flash NVFP4/FP8 | 2 | 256K FP8 KV, 2 slots | 158/164 | 292/300 | 100/100 | 0 | 0 |
| GLM-5.3-Flash NVFP4/FP8 | 4 | 768K BF16 KV, 4 slots | 157/164 | 292/300 | 100/100 | 0 | 0 |
| GLM-5.3-Flash FP8 | 4 | 384K BF16 KV, 4 slots | 154/164 | 295/300 | 100/100 | 0 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 1 | FP8 dense | 160/164 | 294/300 | 100/100 | 0 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 1 | BF16 dense | 158/164 | 291/300 | 100/100 | 0 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense | 159/164 | 293/300 | 100/100 | 0 | 0 |
| Qwen3.8-Flash-Next NVFP4 | 2 | FP8 dense, YaRN 512K, 2 slots | 161/164 | 293/300 | 100/100 | 0 | 0 |
| Qwen3.8-Flash-Next FP8 | 2 | 256K BF16 KV, 4 slots | 160/164 | 292/300 | 100/100 | 0 | 0 |
| Qwen3.8-Flash-Next FP8 | 4 | 256K BF16 KV, 4 slots | 158/164 | 291/300 | 100/100 | 0 | 0 |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3 | 159/164 | 292/300 | 100/100 | 1 | — |
| Qwen3.8-Flash-Next AutoRound int4/int8 | 1 | 256K BF16 KV, 4 slots, MTP3, prefill optimizations on | 159/164 | 291/300 | 100/100 | 2 | — |
| Qwen3.8-Flash-Next NVFP4 (QAD) | 1 | MXFP8 dense, resident n-gram table, 128K BF16 KV, 4 slots | 161/164 | 297/300 | 100/100 | 1 | — |
| Qwen3.8-Flash-Next FP4-Local | 1 | MXFP8 dense, mapped n-gram table, 1M FP8 KV, 4 slots, MTP3 + lookup | 162/164 | 296/300 | 100/100 | 2 | — |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 159/164 | 292/300 | 100/100 | 0 | 0 |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 160/164 | 293/300 | 100/100 | 0 | 0 |
| GLM-5.3 NF4I8 H32 g128 | 4 | 272K FP8 KV | 157/164 | 294/300 | 100/100 | 0 | 0 |
| GLM-5.3 Mixed346 H32 A8 g128 | 4 | 272K FP8 KV | 157/164 | 291/300 | 100/100 | 0 | 0 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 160/164 | 296/300 | 100/100 | 0 | 0 |
| DeepSeek-V4.1-Flash-1017 MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 157/164 | 295/300 | 100/100 | 4 | — |
| DeepSeek-V4.1-Flash-1017 EXL3 3.5bpw | 4 | 1M compressed KV, 16 slots | 160/164 | 295/300 | 100/100 | 6 | 5 (3 pass) |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 141/164 | 295/300 | 100/100 | 0 | 0 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 139/164 | 296/300 | 100/100 | 0 | 0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 154/164 | 295/300 | 100/100 | 0 | 0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 153/164 | 294/300 | 100/100 | 0 | 0 |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 154/164 | 294/300 | 100/100 | 0 | 0 |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, native activations, 768K FP8 KV, fp8 prefill attention, 8 slots | 159/164 | 295/300 | 100/100 | 4 | — |
| Qwen3.8-27B NVFP4 | 1 | DFlash2 drafter, FP8 head, 1M FP8 KV, fp8 prefill attention, 8 slots | 158/164 | 295/300 | 100/100 | 4 | — |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 160/164 | 295/300 | 100/100 | 3 | — |
| Qwen3.8-27B NVFP4 | 2 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 159/164 | 295/300 | 100/100 | 4 | — |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 157/164 | 295/300 | 100/100 | 5 | — |
| Qwen3.8-27B NVFP4 | 4 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 160/164 | 293/300 | 100/100 | 4 | — |
| Flint FP8/NVFP4 | 1 | 512K BF16 KV, 8 slots | 137/164 | 255/300 | 94/100 | 73 | 65 (60 pass) |
| Flint FP8/NVFP4 | 2 | 512K BF16 KV, 8 slots | 133/164 | 257/300 | 96/100 | 77 | 63 (55 pass) |
| Flint FP8/NVFP4 | 4 | 512K BF16 KV, 8 slots | 137/164 | 255/300 | 100/100 | 72 | 59 (53 pass) |

Swift 1.5 received a smaller serving smoke evaluation on 2026-10-17 using
the two-node configuration above: 19/20 GSM8K problems and 20/20 schema
extraction records passed. `scripts/serve_eval.py` used temperature zero,
concurrency 4, low reasoning effort and output caps of 2,048 tokens for
GSM8K and 512 for extraction; none of the 40 responses were truncated.
This is not a full quality campaign or
a matched comparison with the original Qwen checkpoint; HumanEval was not run.

The Flint FP8/NVFP4 rows (2026-10-22) are the first measured with the retry
beyond the cap. The model thinks at length under the eval's low reasoning
effort (its template reads no effort control): a median 575 thinking tokens
on the HumanEval problems it finished. On the one-node leg 26 HumanEval
problems and 43 GSM8K problems reached the 2,048-token cap while still
thinking, and four extraction records the 512-token one; asked once more
at 8,192 tokens, 24 of the 26 HumanEval problems completed and 23 passed,
37 of the 43 GSM8K problems completed and 33 passed, and all four
extraction records completed and passed — the standard-cap scores stand as
tabled, the last column carries the retry. The two- and four-node legs
differ from it by a few items in each direction (the tensor-parallel split
reorders every projection's reduction, and the long-thinking items near a
greedy tie resolve differently); each row carries its own counts. The two
remaining one-node extraction misses are the model's: on
the two "Oona" records its reasoning drafts the email as `ona.…` one line
after reading `oona.…` correctly, with and without the grammar (the
tokenizer is byte-exact with the reference on those sentences), and one of
the two passes when asked alone — Flint's batched greedy text differs from
its solo text (the transcript table's isolation column), the bf16
accumulation sites reassociating across batch rows. Both pass on the
four-node leg (100/100), whose tensor-parallel split reorders the same
sums again. An earlier run of the
same template counted answers the scorer salvaged from thinking text that
leaked into content (HumanEval 145, GSM8K 276): that was the model-opened
`<think>` defect fixed in `b3f9b538`, and those numbers are not tabled.

The FP4-Local HumanEval result is 162/164 after rechecking all 164 unchanged
saved answers with the scorer fix in `9b7439b5`. The original scorer reported
161/164: it discarded the supplied `encode_cyclic` helper when the answer
contained the complete requested `decode_cyclic` function, falsely failing
HumanEval/38. That is the only changed verdict; original scores and answers
are retained in the run records. Older rows have not been re-scored. The two
remaining failures are an output-cap truncation and an incorrect answer.

The Qwen3.8-Flash-Next QAD row's and the six Qwen3.8-27B rows' HumanEval ran each returned code block in a local subprocess with a 10 s timeout (`scripts/serve_eval.py`, 2026-10-14), not in a network-disabled container; their truncated responses are the cap column.

DeepSeek-V4's strict HumanEval scores count responses with unclosed code fences as failures. Executing those open blocks raises the two-node score to 153/164 and the four-node score to 152/164. Plain decoding produces the same answers; the checkpoint's reference implementation also prefers end-of-sequence at that position.

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
| Qwen3.8-Flash-Next NVFP4 (QAD) | 1 | MXFP8 dense, resident n-gram table, 128K BF16 KV, 4 slots | 5/5 | 5/5 classes | different |
| Qwen3.8-Flash-Next FP4-Local | 1 | MXFP8 dense, mapped n-gram table, 1M FP8 KV, 4 slots, MTP3 + lookup | 9/9 | 5/5 classes | different |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV | 4/4 | 5/5 classes | identical |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV | 3/3 | 5/5 classes | identical |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 5/5 | 5/5 classes | identical |
| DeepSeek-V4.1-Flash-1017 MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 9/9 | 5/5 classes | identical |
| DeepSeek-V4.1-Flash-1017 EXL3 3.5bpw | 4 | 1M compressed KV, 16 slots | 9/9 | 5/5 classes | identical |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 6/6 | 5/5 classes | identical |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 6/6 | 5/5 classes | identical |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 3/3 | 5/5 classes | identical |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 4/4 | 5/5 classes | identical |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 4/4 | 5/5 classes | identical |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, native activations, 768K FP8 KV, fp8 prefill attention, 8 slots | 9/9 | 5/5 classes | identical |
| Qwen3.8-27B NVFP4 | 1 | DFlash2 drafter, FP8 head, 1M FP8 KV, fp8 prefill attention, 8 slots | 9/9 | 5/5 classes | identical |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 9/9 | 5/5 classes | identical |
| Qwen3.8-27B NVFP4 | 2 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 9/9 | 5/5 classes | identical |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 9/9 | 5/5 classes | identical |
| Qwen3.8-27B NVFP4 | 4 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 9/9 | 5/5 classes | identical |
| Flint FP8/NVFP4 | 1 | 512K BF16 KV, 8 slots | 9/9 | — | different |
| Flint FP8/NVFP4 | 2 | 512K BF16 KV, 8 slots | 9/9 | — | different |
| Flint FP8/NVFP4 | 4 | 512K BF16 KV, 8 slots | 9/9 | — | different |

| Check | What it establishes |
|---|---|
| Rank operation streams | Ranks executed matching operation sequences; this does not establish numerical equality or request isolation |
| Plain/default greedy match | All three repetitions in both modes produced identical text and token counts for a class |
| Solo/batched greedy text | Exact transcript agreement for the recorded workload; floating-point reduction order can change with batch shape |

Recorded operation streams matched for 54/54 launches in the original campaign, 11/11 in MiMo, 12/12 in DeepSeek-V4, 8/8 in the 2026-10-17/18 Qwen3.8-27B refresh, 2/2 in the 2026-10-18 DeepSeek-V4.1-Flash-1017 set and 7/7 in the 2026-10-23 DeepSeek-V4.1-Flash-1017 EXL3 3.5bpw set (the template, its plain boot and the five mode boots). Single-node launches have only one stream.

Later checks do not replace the historical verdicts above.

| Follow-up | Setting / workload | Recorded result |
|---|---|---|
| Qwen 27B, 2026-10-05 | Other clients join an existing request; one, two and four nodes | Matched the solo requests. |
| Qwen 27B, simultaneous starts | Pinned cuBLASLt algorithm | Prompts above 128 tokens matched across prefill groups |
| Qwen 27B, shorter prompts | `prefill_group=false` | Also covers the shorter prompts; prefill grouping explained the remaining simultaneous-start differences |
| GLM Flash NVFP4/FP8, four nodes | Original dispatch settings | Solo and C2 matched; C3/C4 diverged. |
| GLM Flash NVFP4/FP8, four nodes | `DGPP_DENSE_GEMV_ROWS=8` or `256` | Tested C1–C4 responses matched solo, supporting kernel dispatch as the cause. This does not establish isolation for every workload. [Issue #32](https://github.com/HawkBearPig/dgpp/issues/32) |
| Qwen3.8-Flash-Next QAD, 2026-10-14 | `timed_load --isolation 4` on one node with this build, then the NVIDIA NVFP4 export's one-node template on the same binary | Both exports' solo and batched transcripts diverged at a near-tie (the QAD export at character 173, the NVIDIA export at 149); the batch-shape dependence belongs to the build, not to the export's formats |
| Qwen3.8-Flash-Next FP4-Local, 2026-10-21 | Shipped one-node template at `7c3086939006`, solo versus three concurrent neighbors | Solo/batched text differed at character 161. Repeated C1 transcripts and plain/default decode matched in all five classes. The new long-context QSA traversal is inactive for this short-prompt probe. |
| GLM-4.7 | Excluded at the user's earlier request | Cold-prefill failures and node recoveries occurred; the cause was not established |

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

Limits reflect the checkpoint configurations used for these measurements. A larger shared KV pool alone does not extend a model's positional range.

| Column / label | Definition |
|---|---|
| 0 context | Minimal parcel prompt, at most 256 input tokens; not a literal empty KV cache |
| 32K / 64K / 128K / 256K / 512K | Initial prompt-length buckets, K = 1,024. Actual inputs are within 1,024 tokens below the label, reserving room for the chat wrapper and 256 output tokens |
| Decode ms/pass | Median engine time per decode pass across three C1 requests |
| Engine tok/s | Median output rate over engine decode time at that context |
| Cold prefill (s), 32K | Engine prefill time for the first uncached 32K-bucket request; timings for all lengths are in the following table |
| Limit | Bucket exceeds this deployment's request limit |

Context grows as the request generates output. These measure timing; retrieval accuracy is evaluated separately.

The original matrix refresh fixed the three bottlenecks below. The completion campaign reran every context bucket affected by the later idle-budget, FP4-conversion and MiMo-dispatch fixes. These parcel-document timings use different prompts from the serving table's GSM8K probes. Other retained paths were unchanged by those fixes.

| Affected path | Bottleneck | Investigation |
|---|---|---|
| GLM DSA selection | Repeated top-k sorting at long contexts | Selection scaling |
| Full GLM DSA scoring | Materializing and rereading per-head scores | Score fusion |
| DeepSeek-V4.1 CSA2 prefill | Query tiles sized for maximum KV capacity, underusing workspace at shorter contexts | Query batching |

| Added configuration / capacity check | Validation |
|---|---|
| GLM Flash FP8, four nodes | 512K KV pool with a 2 GiB prefix cache; all measurements completed with zero serving-process swap |
| DeepSeek-V4.1, four nodes | 1M KV pool, six slots and a 14 GiB prefix cache; all serving, decode-mode and context measurements completed with zero serving-process swap |
| MiMo, four nodes | 1M BF16 KV pool; all serving, decode-mode and context measurements completed with zero process and cgroup swap on every rank |
| Full GLM-5.3 int4/int8, four nodes, 256K FP8 KV | Normal startup refused 112.27 GiB + 4 GiB reserve against 115.58 GiB available. No benchmark requests ran. The preliminary plan omitted allocations made earlier in normal startup. |
| Full GLM-5.3 int4/int8, four nodes, 256K FP4 KV | Replaces the unmeasured FP8 case; planned allocation is 4.32 GiB lower. Eight slots, 1.5 GiB prefix-cache budget and 4 GiB reserve are unchanged. All serving, decode-mode and context measurements completed with zero serving-process swap |
| Full GLM-5.3 int4/int8, four nodes, 512K | Memory plan rejected: 125.53 GiB with FP8 KV or 116.89 GiB with FP4 KV, plus 4 GiB headroom, against 116.7 GiB available. The measured rows retain smaller pools. |

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
| Qwen3.8-Flash-Next NVFP4 (QAD) | 1 | MXFP8 dense, resident n-gram table, 128K BF16 KV, 4 slots | 131,072 | 20.595 | 39.82 | 46.1 | 41.99 | 44.0 | 42.33 | 44.0 | 43.30 | 41.8 | Limit | Limit | Limit | Limit |
| Qwen3.8-Flash-Next FP4-Local | 1 | MXFP8 dense, mapped n-gram table, 1M FP8 KV, 4 slots, MTP3 + lookup | 1,048,576 | 20.227 | 68.06 | 54.3 | 71.23 | 48.4 | 71.41 | 43.5 | 72.91 | 53.0 | 76.07 | 41.4 | 89.37 | 41.4 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | 122,880 | 70.959 | 64.37 | 29.1 | 76.25 | 25.7 | 80.89 | 24.4 | Limit | Limit | Limit | Limit | Limit | Limit |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV, 8 slots | 212,992 | 75.099 | 64.18 | 29.2 | 76.24 | 25.7 | 81.06 | 24.2 | 90.13 | 21.9 | Limit | Limit | Limit | Limit |
| GLM-5.3 int4/int8 | 4 | 256K FP4 KV, 8 slots | 262,144 | 74.567 | 64.08 | 29.0 | 75.99 | 26.0 | 80.81 | 24.5 | 89.94 | 22.1 | 109.90 | 18.0 | Limit | Limit |
| GLM-5.3 NF4I8 H32 g128 | 4 | 272K FP8 KV, 8 slots | 278,528 | 74.341 | 64.30 | 29.2 | 76.40 | 25.9 | 81.03 | 24.2 | 90.16 | 21.8 | 109.20 | 18.2 | Limit | Limit |
| GLM-5.3 Mixed346 H32 A8 g128 | 4 | 272K FP8 KV, 8 slots | 278,528 | 74.520 | 62.91 | 29.6 | 74.71 | 26.1 | 79.52 | 24.5 | 88.87 | 22.2 | 108.16 | 18.1 | Limit | Limit |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 131,072 | 19.131 | 51.99 | 61.3 | 53.67 | 55.2 | 53.88 | 51.4 | 56.32 | 50.3 | Limit | Limit | Limit | Limit |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 1,048,576 | 18.712 | 50.91 | 57.6 | 53.27 | 54.4 | 54.96 | 51.6 | 56.26 | 49.3 | 59.85 | 46.3 | 66.60 | 44.0 |
| DeepSeek-V4.1-Flash-1017 MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 1,048,576 | 9.684 | 43.12 | 73.0 | 44.82 | 72.9 | 45.64 | 69.5 | 48.49 | 64.9 | 51.99 | 53.3 | 60.35 | 51.5 |
| DeepSeek-V4.1-Flash-1017 EXL3 3.5bpw | 4 | 1M compressed KV, 16 slots | 1,048,576 | 12.470 | 34.70 | 91.2 | 37.80 | 95.0 | 38.14 | 85.7 | 40.67 | 81.4 | 44.39 | 74.6 | 52.38 | 65.8 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 1,048,576 | 35.319 | 57.83 | 53.7 | 62.48 | 50.4 | 64.37 | 48.4 | 67.13 | 44.7 | 70.41 | 41.6 | 80.99 | 39.9 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 1,048,576 | 26.237 | 36.17 | 83.9 | 39.58 | 88.3 | 41.08 | 79.6 | 44.85 | 75.1 | 49.41 | 70.7 | 58.80 | 66.4 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 131,072 | 24.421 | 41.19 | 44.3 | 44.83 | 44.4 | 47.98 | 41.5 | 54.95 | 36.3 | Limit | Limit | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 262,144 | 26.812 | 41.99 | 45.3 | 47.89 | 41.6 | 53.35 | 37.3 | 64.24 | 31.0 | 86.26 | 23.1 | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 131,072 | 16.705 | 23.64 | 79.3 | 26.10 | 76.3 | 28.17 | 70.2 | 32.48 | 61.3 | Limit | Limit | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 1,048,576 | 16.632 | 23.42 | 78.3 | 25.90 | 76.9 | 28.10 | 70.9 | 32.29 | 61.7 | 41.41 | 48.1 | 58.40 | 34.1 |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, native activations, 768K FP8 KV, fp8 prefill attention, 8 slots | 262,144 | 26.855 | 135.99 | 33.5 | 145.35 | 33.1 | 150.46 | 29.7 | 161.84 | 26.3 | 180.98 | 20.7 | Limit | Limit |
| Qwen3.8-27B NVFP4 | 1 | DFlash2 drafter, FP8 head, 1M FP8 KV, fp8 prefill attention, 8 slots | 262,144 | 16.407 | 107.36 | 39.6 | 116.18 | 42.2 | 121.35 | 29.6 | 131.25 | 34.7 | 152.08 | 31.1 | Limit | Limit |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 262,144 | 20.632 | 75.20 | 60.6 | 81.12 | 62.9 | 84.20 | 57.1 | 90.85 | 41.9 | 105.12 | 43.3 | Limit | Limit |
| Qwen3.8-27B NVFP4 | 2 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 262,144 | 17.182 | 59.87 | 70.9 | 65.83 | 63.5 | 70.53 | 61.3 | 75.79 | 54.3 | 89.93 | 42.3 | Limit | Limit |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 262,144 | 17.315 | 45.91 | 113.4 | 49.68 | 100.6 | 50.99 | 76.9 | 54.31 | 82.4 | 60.93 | 68.6 | Limit | Limit |
| Qwen3.8-27B NVFP4 | 4 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 262,144 | 15.458 | 37.24 | 120.1 | 41.27 | 116.6 | 42.18 | 100.8 | 45.81 | 89.8 | 52.39 | 68.6 | Limit | Limit |
| Flint FP8/NVFP4 | 1 | 512K BF16 KV, 8 slots | 262,144 | 11.720 | 17.37 | 57.6 | 21.03 | 47.5 | 25.23 | 39.6 | 34.03 | 29.4 | 51.46 | 19.4 | Limit | Limit |
| Flint FP8/NVFP4 | 2 | 512K BF16 KV, 8 slots | 262,144 | 8.510 | 9.58 | 104.4 | 11.93 | 83.8 | 14.16 | 70.6 | 18.38 | 54.4 | 26.79 | 37.3 | Limit | Limit |
| Flint FP8/NVFP4 | 4 | 512K BF16 KV, 8 slots | 262,144 | 6.308 | 6.48 | 154.4 | 7.87 | 127.1 | 9.06 | 110.3 | 11.17 | 89.5 | 15.40 | 64.9 | Limit | Limit |

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
| Qwen3.8-Flash-Next NVFP4 (QAD) | 1 | MXFP8 dense, resident n-gram table, 128K BF16 KV, 4 slots | 0.334 | 20.595 | 43.229 | 95.045 | Limit | Limit |
| Qwen3.8-Flash-Next FP4-Local | 1 | MXFP8 dense, mapped n-gram table, 1M FP8 KV, 4 slots, MTP3 + lookup | 0.354 | 20.227 | 42.303 | 93.128 | 224.228 | 595.753 |
| GLM-5.3 int4/int8 | 4 | 120K BF16 KV, 8 slots | 0.821 | 70.959 | 153.616 | Limit | Limit | Limit |
| GLM-5.3 int4/int8 | 4 | 208K FP8 KV, 8 slots | 0.840 | 75.099 | 161.000 | 366.933 | Limit | Limit |
| GLM-5.3 int4/int8 | 4 | 256K FP4 KV, 8 slots | 0.787 | 74.567 | 159.966 | 365.499 | 898.244 | Limit |
| GLM-5.3 NF4I8 H32 g128 | 4 | 272K FP8 KV, 8 slots | 0.945 | 74.341 | 160.953 | 364.771 | 893.299 | Limit |
| GLM-5.3 Mixed346 H32 A8 g128 | 4 | 272K FP8 KV, 8 slots | 0.715 | 74.520 | 161.336 | 364.648 | 893.491 | Limit |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 128K compressed KV, 6 slots | 0.291 | 19.131 | 43.751 | 126.461 | Limit | Limit |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 0.294 | 18.712 | 40.931 | 105.382 | 367.554 | 1908.016 |
| DeepSeek-V4.1-Flash-1017 MXFP4/FP8 | 4 | 1M compressed KV, 6 slots | 0.263 | 9.684 | 19.999 | 43.577 | 103.139 | 301.219 |
| DeepSeek-V4.1-Flash-1017 EXL3 3.5bpw | 4 | 1M compressed KV, 16 slots | 0.272 | 12.470 | 25.652 | 54.885 | 125.507 | 345.781 |
| DeepSeek-V4-Flash MXFP4/FP8 | 2 | 1M BF16 KV, 4 slots | 0.281 | 35.319 | 89.437 | 265.781 | 984.962 | 4679.601 |
| DeepSeek-V4-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 6 slots | 0.190 | 26.237 | 70.847 | 225.675 | 897.362 | 4473.733 |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 128K BF16 KV, 4 slots | 0.450 | 24.421 | 63.481 | 182.487 | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 2 | 256K FP8 KV, 4 slots | 0.464 | 26.812 | 72.294 | 218.280 | 725.764 | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 128K BF16 KV, 4 slots | 0.223 | 16.705 | 41.425 | 113.534 | Limit | Limit |
| MiMo-V2.6-Flash MXFP4/FP8 | 4 | 1M BF16 KV, 4 slots | 0.231 | 16.632 | 41.529 | 113.838 | 346.439 | 1160.847 |
| Qwen3.8-27B FP8 | 1 | DFlash2 drafter, FP8 head, native activations, 768K FP8 KV, fp8 prefill attention, 8 slots | 0.226 | 26.855 | 57.385 | 130.619 | 326.161 | Limit |
| Qwen3.8-27B NVFP4 | 1 | DFlash2 drafter, FP8 head, 1M FP8 KV, fp8 prefill attention, 8 slots | 0.160 | 16.407 | 36.921 | 90.520 | 242.918 | Limit |
| Qwen3.8-27B FP8 | 2 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 0.161 | 20.632 | 43.250 | 94.499 | 221.971 | Limit |
| Qwen3.8-27B NVFP4 | 2 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 0.133 | 17.182 | 36.987 | 81.357 | 194.735 | Limit |
| Qwen3.8-27B FP8 | 4 | DFlash2 drafter, FP8 head, native activations, 2M FP8 KV, fp8 prefill attention, 8 slots | 0.128 | 17.315 | 35.808 | 76.004 | 167.828 | Limit |
| Qwen3.8-27B NVFP4 | 4 | DFlash2 drafter, FP8 head, 2M FP8 KV, fp8 prefill attention, 8 slots | 0.115 | 15.458 | 32.102 | 68.695 | 153.222 | Limit |
| Flint FP8/NVFP4 | 1 | 512K BF16 KV, 8 slots | 0.128 | 11.720 | 36.445 | 133.208 | 503.100 | Limit |
| Flint FP8/NVFP4 | 2 | 512K BF16 KV, 8 slots | 0.088 | 8.510 | 24.437 | 78.201 | 276.200 | Limit |
| Flint FP8/NVFP4 | 4 | 512K BF16 KV, 8 slots | 0.064 | 6.308 | 16.485 | 47.691 | 153.814 | Limit |

<!-- END context-prefill -->

### YaRN retrieval

The earlier YaRN check planted numeric codes at 5%, 25%, 50%, 75% and 95% of each document. The response budget was 768 tokens including reasoning. A hit means the expected code appeared after removing non-digit characters from the response. The two-stream check measures completion; admission can queue a request when the shared KV pool is full.

| Target tokens | Actual prompt tokens | Retrieval hits | Repeated prefix cached | Concurrent requests completed |
|---|---|---|---|---|
| 262,144 | 261,388 | 5/5 | 99.997% | 2/2 |
| 524,288 | 522,619 | 5/5 | 99.999% | 2/2 |

The ~4K control retrieved 5/5 codes. Prefix reuse repeats the final probe.

## Selection microbenchmarks

| Setting | Qwen QSA / DeepSeek CSA2 |
|---|---|
| Hardware | Idle GB10; CUDA graphs |
| Iterations | Five warmups, thirty timed iterations per shape |
| QSA validation | Scores and selections compared with a host oracle |
| CSA2 validation | Selections compared with a host sort of production score keys; score arithmetic is not independently checked |
| Cache conditions | Repeated warm inputs; cold samples evict L2 |
| Scope | Kernel timing, separate from end-to-end service throughput |

The timings from that run stayed with its untracked run record; the benches
are `benchmarks/micro/qsa_index_bench.cu`, `qsa_prefill_bench.cu` and
`csa2_select_bench.cu`.

## Historical external workload

This separate llama-benchy check predates the matrix refresh and uses temperature 1.

| Model / deployment | Workload | Repetitions | Decode tok/s | Prefill tok/s | Record |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash, four nodes | `pp2048 / tg128`, checkpoint sampling defaults, thinking on | 3 × 40 runs | 64.3–65.1 | About 1,380 | Historical checkpoint-default run |

## Reproduce

The tables above summarize historical measurements at the listed revisions.
Raw run artifacts are not part of the project documentation. For new results,
record the source revision, executable hashes, checkpoint revisions, deployment
configuration, actual token counts and repeated timings. Compare old and new
executables on the same idle hardware and preserve failed attempts locally.

Build both executables in the primary checkout, preserving existing work when
changing revisions. Reserve the cluster for one deployment at a time. Keep
startup, downloads and warmup outside timed requests; verify no swapping and
matching operation streams across ranks. Follow [operations](operations.md)
for deployment and restore the original service after testing.

```bash
cmake --preset release
cmake --build build-release --target dgpp_serve_app -j 4
python3 scripts/prepare_data.py download
```

The whole row set for one template, in one command (records under the ignored `benchmarks/results/`; `summarize` and `rows` read them back):

```bash
python3 scripts/serve_campaign.py run --config deploy/cluster_qwen3.8-27b_fp8_w1.example.json \
  --out benchmarks/results/$(date -u +%Y%m%d)-qwen27b-fp8-w1
python3 scripts/serve_campaign.py rows benchmarks/results/<dir> --label "Qwen3.8-27B FP8" --nodes 1 --options "..."
```

Measure an already running server with the maintained clients:

```bash
python3 scripts/timed_load.py 127.0.0.1 18080 \
  --concurrency 1,2,4,8 --classes all --repeat 3 --max-tokens 256 \
  --json-out /tmp/decode.json
.venv/bin/python scripts/serve_context_matrix.py 127.0.0.1 18080 \
  --model nvidia/Qwen3.8-Flash-Next-NVFP4 --tag unique-run-id \
  --json-out /tmp/context.json
python3 scripts/serve_admission_bench.py --help
```

Use the configured host and port. Add `--think` for the full GLM-5.3 serving
sweep and a fresh context tag for each cold run. Install context-client
dependencies from `requirements-tools.txt`. Keep probability diagnostics
separate from timing and report latency alongside throughput.

Inventory checkpoints and resident caches before testing. Remove downloaded
weights and generated resident images only when they were absent from that
inventory and are no longer in use. Store detailed results in ignored local
storage or external artifacts; update the relevant table with its date,
configuration, comparison and limitations. See [testing](testing.md) for the
correctness checks and [benchmark procedures](benchmarking.md) for microbenchmarks.
