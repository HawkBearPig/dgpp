#pragma once
// IGemm interface (DESIGN §§4 and 11): weights [N,K] fp8/bf16 against
// activations [M,K].
// Implementations must be deterministic run-to-run at fixed shapes so CUDA
// graph capture replays bitwise.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "kernels/bf12_gemv.hpp"

namespace dgpp {

enum class GemmOut : int { BF16, F32 };

class IGemm {
 public:
  virtual ~IGemm() = default;

  // Computes D[M,N] = Act[M,K] x W[N,K]^T, enqueued on `stream`.
  // act: row-major [M,K]; weight: row-major [N,K] contiguous.
  // act_row_stride: element stride between consecutive rows of `act`.
  // Pass k for a contiguous [M,K] tensor; wider strides read K-column slices
  // of a fused projection buffer (e.g. KDA f_a/g_a views).
  virtual void matmul(const void* act, const void* weight, void* out, int m,
                      int n, int k, DType io_dtype, GemmOut out_dtype,
                      size_t act_row_stride, void* workspace, size_t ws_bytes,
                      cudaStream_t stream) = 0;

  virtual size_t query_workspace_bytes(int m, int n, int k,
                                       DType io_dtype) = 0;

  // Prebuilds (and validates feasibility of) the plan for this shape without
  // executing it. Returns false when no heuristic exists — call at init, never
  // during capture.
  virtual bool ensure_plan(int m, int n, int k, DType io_dtype,
                           GemmOut out_dtype, size_t act_row_stride) = 0;

  // Whether this instance's decode-shaped bf16 calls take the streaming
  // tensor-core form (CublasLtGemm::set_decode_mma): the layers that own
  // fp8 projections next to their bf16 ones read it to lower those the
  // same way (scale_gemm.hpp's decode_mma), so one setting per model
  // decides every dense site. The bound (0: every row count) is the widest
  // row count the form takes; a fake instance says no.
  virtual bool decode_mma() const { return false; }
  virtual int decode_mma_max_rows() const { return 0; }

  // The bytes a matmul of m rows against `weight` will stream — what an L2
  // prefetch ahead of it must touch. The weight's own [weight, weight +
  // bytes) unless the instance holds a packed companion for it that the
  // m-row launch takes (CublasLtGemm::register_bf12).
  virtual void resident_view(const void* weight, size_t bytes, int /*m*/, const void** ptr,
                             size_t* view_bytes) const {
    *ptr = weight;
    *view_bytes = bytes;
  }
  // The packed companion registered for `weight` (CublasLtGemm::register_bf12),
  // or null: the fused decode launches that bypass matmul — the GDN's
  // multi-problem GEMV, the GR site's kernels — pick their packed twins by it.
  virtual const Bf12Matrix* bf12_lookup(const void* /*weight*/) const { return nullptr; }
};

// The decode shapes the interface lowers to the row-independent GEMV core: m up
// to the model's decode rows (set_decode_rows; kGemmDecodeRowsDefault, the
// old fixed batch, until a model says otherwise — a wider batch's rows
// keep the scalar reduction order whatever batch they ride in: the first
// 9-row batch fell to an Lt algorithm with its own reduction order,
// 2026-09-10), in chunks of at most gemv::kMaxRows. kGemmDecodeLoweringRows
// bounds it (engine/decode_outputs.hpp's kDecodeRowsMax). The interface cannot
// tell a decode call from a short prefill chunk, so the bound is the
// model's decode shape and nothing wider: a prefill of more rows keeps
// its Lt algorithm (and its transcripts). So a prompt no longer than the
// decode rows prefills through the GEMV chain, and two deployments with
// different decode rows (the full GLM-5.3's eight-row and sixteen-row
// shapes, 2026-09-13) prefill a 9–16-token prompt through different
// kernels — last-bit differences, deterministic within a deployment;
// glm_dsa_engine_test's sixteen-row gate configures both of its models
// alike for that reason.
constexpr int kGemmDecodeRowsDefault = 8;

// The dense sites' GEMV chunk bound for the session-core families
// (DGPP_DENSE_GEMV_ROWS, default 4; 2026-09-14): rows up to it take the
// row-independent GEMV chunks (one weight read per four rows — the T = 1
// floor and the fused multi-problem launches), bf16 rows above it take
// cuBLASLt's algorithm (at the weight-stream floor from six rows:
// bf16_gemv_test's table — the chunks re-read the weights per four rows,
// 2-8x the bytes at 6..32 rows), fp8 rows above it the streaming
// tensor-core GEMM to kScaleGemmMmaMaxRows (scale_gemm.hpp). A row's chain
// then depends on the rows sharing its launch across the bound (the
// chunks', the algorithm's and the streaming form's orders differ): the
// batched and the scalar transcripts of one request are tolerance-equal,
// not bitwise (the engine gates' near-tie rule). 256 restores the old
// lowering (every decode row count through the chunks, fp8 to 128 rows).
int dense_gemv_rows();
constexpr int kGemmDecodeLoweringRows = 64;

// cuBLASLt-backed implementation with per-shape heuristic caching. Decode-
// shaped bf16 calls (m <= the decode rows, k % 8 == 0, 16B-aligned weight)
// bypass Lt for one or more launches of the in-house row-independent
// bandwidth GEMV (bf16_gemv.hpp) — Lt's m=1 kernel runs at ~55% of the
// part's bandwidth.
class CublasLtGemm : public IGemm {
 public:
  CublasLtGemm();   // creates handle, unit-scale device constants
  ~CublasLtGemm() override;

  void matmul(const void* act, const void* weight, void* out, int m, int n,
              int k, DType io_dtype, GemmOut out_dtype,
              size_t act_row_stride, void* workspace, size_t ws_bytes,
              cudaStream_t stream) override;

  // Independent BF16 matrices with FP32 destinations. Batch strides are in
  // elements; rows within each matrix are contiguous. Always uses cuBLASLt,
  // including short query tiles (there is no decode/GEMV lowering).
  // plan_rows selects the algorithm for the untiled matrix while retaining
  // the actual tile layouts; this keeps query tiling from changing reductions.
  // weight_kn reads weights as row-major [K,N] instead of [N,K].
  void matmul_batched_bf16(const uint16_t* act, const uint16_t* weight, float* out, int m, int n,
                           int k, int batch, int64_t act_stride, int64_t weight_stride,
                           int64_t out_stride, void* workspace, size_t ws_bytes,
                           cudaStream_t stream, int plan_rows = 0, bool weight_kn = false);

  // BF16 linear projection with FP32 split-K reductions and one BF16 output
  // rounding. Optional bias uses the fused epilogue. Always uses cuBLASLt,
  // including small row counts; ordinary text matmul keeps its own dispatch.
  void matmul_linear_bf16(const uint16_t* act, const uint16_t* weight, const uint16_t* bias,
                        uint16_t* out, int m, int n, int k, void* workspace, size_t ws_bytes,
                        cudaStream_t stream);

  // E4M3 x E4M3 with per-call scalar scales (F32 device pointers) and BF16
  // output. Same D[M,N] = Act[M,K] x W[N,K]^T convention; `weight` holds
  // X/448 codes, `act_scale`/`weight_scale` the two grid maxabs values.
  // Reuses the cached F8 plan (its unit-scale pointers are overwritten per
  // call, the bias-pointer precedent) — the caller must have run
  // ensure_plan(m, n, k, F8_E4M3, BF16, ...) first.
  void matmul_fp8_scaled(const uint8_t* act, const uint8_t* weight,
                         const float* act_scale, const float* weight_scale,
                         uint16_t* out, int m, int n, int k, void* workspace,
                         size_t ws_bytes, cudaStream_t stream);

  size_t query_workspace_bytes(int m, int n, int k, DType io_dtype) override;

  bool ensure_plan(int m, int n, int k, DType io_dtype, GemmOut out_dtype,
                   size_t act_row_stride) override;

  // The widest decode shape this model runs (its fixed batch's rows): bf16
  // calls up to it take the GEMV core. [1, kGemmDecodeLoweringRows].
  void set_decode_rows(int rows);
  int decode_rows() const;
  // Force BF16 matmul in this inclusive row range through kernel-only
  // lowering. Other row counts retain their existing dispatch. The model
  // enables this for decode and clears it for prefill; (0, 0) disables it.
  // Unsupported shapes fail explicitly instead of falling back to cuBLASLt.
  void set_kernel_only_rows(int min_rows, int max_rows);
  // Decode-shaped bf16 calls take the streaming tensor-core GEMM
  // (mma_gemv.hpp) instead of the 4-row GEMV chunks: the weights read once
  // for every row of the launch, each row's chain the same whatever m. The
  // two forms are tolerance-equal, not bitwise, so a model opts in for all
  // its calls through this instance. Shapes the mma form cannot take keep
  // the GEMV chunks. max_rows bounds the form: 0 takes every row count
  // (DeepSeek: its group prefill's spans are then bitwise their prefills
  // alone), a bound hands wider calls to the Lt algorithm (the session-core
  // families: Lt is ahead of the streaming form's 128-row groups from a
  // dozen bf16 rows — bf16_gemv_test's table, 2026-09-14).
  void set_decode_mma(bool on, int max_rows = 0);
  bool decode_mma() const override;
  int decode_mma_max_rows() const override;

  // A lossless 12-bit companion of a bf16 weight (bf12_gemv.hpp): the GEMV
  // lowering's launches against `weight` stream the companion instead —
  // bitwise the bf16 GEMV's rows, 0.75 of its bytes — and under
  // set_bf12_wide the lowering widens to kBf12MaxRows rows for it (a
  // five-to-eight-row decode batch no longer takes an Lt algorithm over the
  // bf16 bytes). Wider calls (prefill, the batches past eight rows) read
  // the weight's own bytes — unless they were released (bf12_release_raw).
  // The caller owns the companion's memory and registers before any
  // capture; n and k must be the matmul's.
  // bf16 Lt calls of fewer rows than `rows` take the algorithm the heuristic
  // picks for `rows` (0: each call's own). A walk that runs a chunk's site
  // in row blocks (the prefill's fold overlap) sets the chunk's rows so a
  // row's reduction does not depend on the block it rode in: the blocks are
  // then bitwise the chunk (gemm_split_test). The caller resets it.
  void set_plan_rows(int rows);
  void register_bf12(const void* weight, const Bf12Matrix& packed);
  // The companions' five-to-eight-row launches: on only while the caller's
  // rows are a DECODE batch. The interface cannot tell a decode call from a
  // short prefill chunk, and a 5..8-row prefill chunk (a prefix-cache cut
  // leaves them) must keep the Lt algorithm its transcripts went through;
  // a decode batch of that width has no such history to keep — its rows
  // take the scalar chain instead, which is what the rows alone compute.
  void set_bf12_wide(bool on);
  size_t bf12_registered() const;
  const Bf12Matrix* bf12_lookup(const void* weight) const override;
  // bf12-ONLY residency (common/bf16_residency.hpp): `weight`'s own bytes
  // are gone — the loader returned them once the companion existed
  // (loaders/releasable_range.hpp) — and only its address remains, as the
  // key. Every GEMV-lowered launch already streams the companion. What is
  // left of the bf16 bytes' readers:
  //  * an Lt call (a prefill chunk): the rows it needs are expanded into the
  //    expansion scratch first — the exact bf16 bits, so the algorithm
  //    computes what it always did. A matrix larger than a scratch slot runs
  //    in WEIGHT-row blocks under the whole call's algorithm: bitwise the
  //    whole call (an output element's reduction does not depend on the
  //    weight rows sharing its call once the algorithm is pinned —
  //    bf12_gemv_test pins it on the head's shapes).
  //  * a decode batch past the lowering's rows (set_bf12_wide on, nine rows
  //    and up): kBf12MaxRows-row packed launches — the scalar chain,
  //    tolerance-equal to the Lt algorithm the other modes run at that
  //    width, 0.75 of the bytes per eight rows. A captured graph therefore
  //    never touches the scratch.
  void bf12_release_raw(const void* weight);
  bool bf12_raw_released(const void* weight) const;
  // `slots` scratch buffers of `bytes` each (replaces an earlier reserve).
  // While set_bf12_wide is off (a prefill walk), a matrix that fits a slot
  // stays expanded there until the slot is reused: the fold overlap's row
  // blocks call in, o, in, o per site and expand each once.
  void bf12_reserve_expand(int slots, size_t bytes);
  size_t bf12_expand_bytes() const;  // slots * bytes
  uint64_t bf12_expansions() const;  // expansions launched (tests, telemetry)
  void resident_view(const void* weight, size_t bytes, int m, const void** ptr,
                     size_t* view_bytes) const override;

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

}  // namespace dgpp
