#pragma once
// IGemm interface (DESIGN §§4 and 11): weights [N,K] fp8/bf16 against
// activations [M,K].
// Implementations must be deterministic run-to-run at fixed shapes so CUDA
// graph capture replays bitwise.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"

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

  size_t query_workspace_bytes(int m, int n, int k, DType io_dtype) override;

  bool ensure_plan(int m, int n, int k, DType io_dtype, GemmOut out_dtype,
                   size_t act_row_stride) override;

  // The widest decode shape this model runs (its fixed batch's rows): bf16
  // calls up to it take the GEMV core. [1, kGemmDecodeLoweringRows].
  void set_decode_rows(int rows);
  int decode_rows() const;
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

 private:
  struct Impl;
  Impl* impl_ = nullptr;
};

}  // namespace dgpp
