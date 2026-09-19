# Qwen decode on two DGX Sparks

This branch adds 64-row Qwen decode, FP8 vocabulary-head weight reuse, and
1024-token prefill/cache boundaries. The x86-to-ARM64 build workflow is in
`scripts/spark-cross`. Validation used Qwen3.8-Flash-Next-NVFP4 on two GB10
ranks with FP8 dense weights, BF16 KV, mapped PLE, and MTP depth 3.

## Implementation

The shared decode, picker and GEMM lowering ceilings are 64 rows. Request
capacity remains 16: C16/MTP3 uses 16 x 4 verification rows. Intermediate
families cover 2/3/4/6/8/12 slots before the full family. Selection covers
the highest occupied slot, so sparse occupancy can select a larger graph
than the active count suggests. Other model families retain their caps.
All ranks must use matching binaries because picker layouts changed.

Position and draft kernels now launch enough threads for all 64 rows;
all picker winner sentinels initialize to -1. A 32-thread batched position
launch initially caused continuation and slot-isolation failures. The fix
passed the original assertions plus direct upper-half coverage.

At 48/64 rows the MTP embedding projection could capture a cuBLASLt memset
node, violating the kernels-only graph contract. Wide decode now uses the
existing kernel-only projection helper. Smaller shapes and prefill retain
the prior dispatch. Real hidden-size tests cover 12/16/32/48/64 tokens,
one/four branches, graph node types, replay stability and an FP64 reference.

The FP8 vocabulary head selects streaming MMA above the default four-row
GEMV threshold, bounded by configured decode rows. This reuses weights
across rows. It still converts weights for BF16 tensor arithmetic; it is
not native FP8 arithmetic. FP32 accumulation order changes, so outputs
need not be bit-identical. DGPP_DENSE_GEMV_ROWS=256 restores the diagnostic
GEMV path. Qwen prefill chunk boundaries are now 1024 rather than 2048.

## Validation and measurements

- ARM64 server and affected CUDA test targets built successfully.
- 64-position tests cover inactive upper-half requests and overflow.
- C16/MTP3 picker tests match a host oracle over a simulated four-rank world.
- Two-rank synthetic Qwen integration passes with both production and
  row-independent lowering, exercising 48/64-row replay, cancellation,
  slot reuse, sparse slots, cached continuation and yielded continuation.
- Full-model TP2 startup captured 48/64-row graphs. Sixteen simultaneous
  smoke requests completed with 2048 output tokens; observed concurrency
  reached 16, with no failed requests or engine failure.
- Startup rank-0 free device memory was 3.27 GiB; node available memory
  was 16.73 GiB. This is not a worst-case memory-headroom guarantee.
- Isolated vocabulary head (M=16, N=124160, K=2560): about 4.93 ms to
  1.32 ms. Relative L2 difference about 2.77e-6; sampled top-1 matched.
- A short cache-warm C4/MTP3 comparison, four fixed prompts, 512 output
  tokens each, one warmup and three measured batches per binary, improved
  mean aggregate throughput from 129.19 to 145.00 tokens/s. This measures
  the head change, not the benefit of widening to 64 rows.

The full suite and four-node hardware validation were not run for this
candidate. The production service remains running, so no final shutdown
op-stream digest comparison was collected. Live workloads differed;
there is no controlled C8-versus-C16 throughput claim or power comparison.
Private request payloads and deployment addresses are omitted.

## Reproduce focused tests

Build fresh binaries first. Run serially on idle Spark hardware:

```sh
DGPP_TEST_FILTER=rows64 ./glm_pick_test
DGPP_TEST_FILTER=wide DGPP_TEST_DENSE_FP8=1 DGPP_TEST_QWEN_ROWS64=1 DGPP_DENSE_GEMV_ROWS=4 ./qwen_engine_test
DGPP_TEST_FILTER=wide DGPP_TEST_DENSE_FP8=1 DGPP_TEST_QWEN_ROWS64=1 DGPP_DENSE_GEMV_ROWS=256 ./qwen_engine_test
```

Equivalent CTest entries are `qwen_engine_fp8_mtp3_rows64` and
`qwen_engine_mtp3_rows64_row_independent`. Use ROWS32 instead of ROWS64
for the C8/MTP3 test variant.
