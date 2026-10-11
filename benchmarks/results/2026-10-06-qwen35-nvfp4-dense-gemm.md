# 2026-10-06 — Qwen3.8-27B-NVFP4 dense GEMM campaign (items 1–3)

Post-Item-1 A/B and the fp4w prefill kernel behind CHANGELOG's
2026-10-06 entries. All numbers GB10, TP=1 (`w1` template), greedy
(temp 0) unless noted.

## Microbench (single-GEMM, w1 shapes, μs)

gate/up [17408×5120], m = 1/8/32/64/128:

| form | 1 | 8 | 32 | 64 | 128 | roofline |
|---|---|---|---|---|---|---|
| fp4 GEMV (old decode) | 214 | 623 | 2456 | 4886 | 9999 | ~251 |
| grouped prod ldmatrix (item 1) | 270 | 239 | 257 | 295 | 513 | ~251 |
| fp4 streaming (tried, reverted) | 231 | 244 | 288 | 436 | 908 | ~251 |

down [5120×17408]: prod 221/232/270/353/535; streaming 225/239/330/447/951.

Prefill shapes, prod vs fp4w (weight-once per group):

| m | gate prod | gate fp4w | down prod | down fp4w |
|---|---|---|---|---|
| 256 | 977 | 1032 | 1075 | 972 |
| 448 | 1645 | 1824 (−11%) | 1686 | 1876 (−11%) |
| 512 | ~1870 | ~2000 (noisy) | 1973 | 1850 |
| 1024 | 3846 | 3386 (+12%) | 3988 | 3319 (+17%) |
| 2048 | 7389 | 6748 (+9%...14%) | 8239 | 7021 (+15–20%) |
| 4096 | 25829 | 14292 (+45%) | 26724 | 17412 (+35%...90% gate) |

Crossover ≈ 1024 rows: fp4w wired above 1024 only (all tests ≤ 296 rows
stay on the production kernel; decode ≤ 64 rows untouched).

## Serve A/B vs the FP8 release (identical transcripts + drafter acceptance)

- Short prompt (5+32 tok): NVFP4 29.1 tok/s, 119 ms/pass vs FP8 24.5, 141.
- pp448: NVFP4 720–748 ms vs FP8 931.
- pp2080: NVFP4 2445–2460 ms vs recipe baseline 2568 (~5%).

## Recipe matrix (throughput-only, post-fp4w, run 2026-10-06T01-46-56Z)

pp2048 tg128 @ d0: 842 t/s (was 780 pre-fp4w), TTFT 2438 ms (was 2568);
@ d4096: 823 (was 636); @ d8192: 825 (was 619). Gap to FP8's 878 is −4%.
Decode legs unchanged-or-better except one soft cell (d0 tg 28.4 vs 34.4;
siblings up; decode path byte-identical between the binaries — variance).
Deep-prefix + concurrent decode still collapses (d8192c4 tg 12.3) —
KV/attention-bound shared code, not NVFP4-specific.

## Reverted: fp4 streaming decode kernel

Correct (doubles oracle, bitwise m-invariant 1..128, sparse-vs-grouped
bitwise) but slower than prod-ldm on every production shape (table
above); the one m==1 win is unusable (splits the ≤128 chain the
solo-vs-batch gates require). Recipe kept in the model card.

## Gates run

unit_tests 317/317; ctest qwen35_mixed (7), qwen35_forward, qwen35_decode,
fp4w_gemm, glm_moe, fp4_gemv, mma_gemv, fp8w_gemm — all pass. Serve smoke:
coherent text, temp-0 transcript-identical to FP8 on both A/B prompts.
