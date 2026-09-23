# Issue #4: fresh exact vLLM reference

This uses the reporter's immutable ARM64 container image
`sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a`,
vLLM `0.1.dev20073+g8e685d198`, Torch `2.13.0+cu130`, Transformers `5.15.1`,
and FlashInfer `0.6.17`, with the exact provided PLE source overlay and pinned
checkpoint. It selects FLASHINFER_CUTLASS NVFP4 experts and Triton/FLA GDN
prefill, as the reporter's run did. The transport uses this cluster's verified
RoCE-v2 GID and interface. The peer cannot resolve Docker Hub, so the identical
image was streamed from the head with docker save/load and its ID checked.

The model is running on fresh TP2 containers, with one sequence, native 262144
context, 2048 prefill rows, BF16 caches, no prefix caching, no MTP and eager
execution. All 261120 native prompt token IDs match DGPP's original request
exactly, confirmed by the reference /tokenize endpoint before generation.
The first generation completed in 126.9 seconds and **failed 5/6**, returning
`val_5b9d2c3d7a95b986` for the same failed key. It used 190 output tokens and
stopped normally. The API omits cached-token details; prefix caching was
explicitly disabled. The raw generic DGPP scorer marks that missing field as a
protocol failure; verdict.json separately records the valid reference protocol
with its explicit no-cache configuration. This fresh run does not reproduce the
reporter's saved 6/6 success. It is not evidence that DGPP is correct or that the
model alone is responsible. Production restoration was independently verified. Source files extracted from the image,
commands and runtime logs are retained under raw/. The campaign restores the
original production deployment after stopping its own containers.

Both nodes' complete pinned snapshots were independently read and hashed. All 25 files per node match their content-addressed blobs (SHA256 for LFS, Git-blob SHA1 for ordinary files), totaling 132734506847 bytes per node. Both snapshots are identical. The model and quantization metadata also match the reporter's supplied JSON. See `snapshot-verification.json`.

Source tracing confirms the requested autotune cache is on the active path:
vLLM's `FlashInferExperts.apply` calls `flashinfer.cutlass_fused_moe`; that
function selects GEMM1/GEMM2 tactics using cache namespaces
`trtllm::fused_moe::gemm1` and `trtllm::fused_moe::gemm2`. The TRTLLM names
therefore do not mean these entries are unused by FLASHINFER_CUTLASS. This
establishes relevance of the requested cache, not that tactics caused the
reference outcome. Extracted source files are retained in raw/.

The pinned backend defaults to allowing fused expert finalization. The matching
NVIDIA epilogue uses atomic BF16 additions for BF16 outputs. When a selected
GEMM2 tactic uses that fusion, accumulation order can affect rounded results.
This is a source-supported hypothesis for reference request-to-request variation,
not yet a demonstrated cause or proof that any particular cached tactic uses it.
The current trace compares repeated model boundaries before changing the backend.
