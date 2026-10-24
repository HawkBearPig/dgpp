#pragma once
#include <cstdint>

namespace dgpp {

// M5 tensor-parallel block-boundary interface (DESIGN §5.1): folds a partial
// hidden activation [rows, hidden] bf16 into its replicated value, in
// place. Called after the attention output projection and after the
// FFN/MoE — the two row-parallel sites whose local sums are partial —
// always with the producing kernels already quiesced on the model stream.
// world=1 constructs the model without a reducer and the interface is skipped.
struct BoundaryReducer {
  virtual ~BoundaryReducer() = default;

  // Optional staging interface (the §6.3 evolution): hands the producing GEMM
  // a pinned, device-writable destination for the boundary partial so the
  // collective sends it straight from there (no device→slot staging copy).
  // Returns nullptr when the shape does not fit (rows*hidden above the
  // latency slot — prefill-sized boundaries stay on the device path) or
  // when the transport has no pre-stage support. The returned pointer is
  // consumed by the following reduce() call.
  virtual uint16_t* stage(int /*rows*/, int /*hidden*/) { return nullptr; }

  virtual void reduce(uint16_t* partial, int rows, int hidden) = 0;

  // The stream-ordered form (2026-09-14, plan D9): reduce() launches the
  // fold on the model's stream after the producing kernels — the model
  // skips its host drain before the call — and settle() (once per pass,
  // before the host reads results) waits for the transport's verdict.
  // bind_stream() gives the reducer the model's stream (the model calls
  // it at construction). The defaults are the host-driven contract.
  virtual bool stream_ordered() const { return false; }
  virtual void bind_stream(void* /*cudaStream_t*/) {}
  virtual void settle() {}

  // The asynchronous form of a prefill-class fold (2026-09-19, the fold
  // overlap): begin_async() submits the fold of partial[rows, hidden] and
  // returns at once — these rows' producing kernels are quiesced, and the
  // caller may enqueue and run OTHER rows' work on its stream meanwhile —
  // and end_async() waits for it. One fold outstanding at a time. False
  // from begin_async(): this shape or transport folds synchronously only
  // (nothing was submitted; call reduce()).
  virtual bool begin_async(uint16_t* /*partial*/, int /*rows*/, int /*hidden*/) { return false; }
  virtual void end_async() {}
  // begin_async behind an EVENT (2026-10-18): the rows' producing kernels
  // are ordered before `ready` (a cudaEvent_t the caller recorded on its
  // stream), and the caller has already enqueued other rows' work behind
  // them — the implementation waits for the event, not the stream, so the
  // GPU keeps that work while the host submits the fold. The default folds
  // nothing (false): the caller drains and calls begin_async().
  virtual bool begin_async_after(uint16_t* /*partial*/, int /*rows*/, int /*hidden*/, void* /*ready*/) { return false; }

  // Measurement interface (2026-09-09, the Qwen plan's Q0): one extra collective
  // of `rows x cols` bf16 over a scratch buffer nobody reads, issued right
  // after a boundary fold — the shape and position of the all-reduce a
  // sliced GR gate would add per site. Default: nothing. Implementations
  // clamp the element count to their latency slot.
  virtual void probe(int /*rows*/, int /*cols*/) {}
};

}  // namespace dgpp
