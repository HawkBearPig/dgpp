#pragma once
// Composition interface (M5 deliverable 3): adapts BoundaryReducer to the
// CollectiveBus one-shot all-reduce. Header-only so the ungated models
// library never links ibverbs — only consumers that already link the bus
// (the loopback test, the fabric app) include it.
//
// Chunking: the bus all-reduce's v1 bound is one latency slot
// (lat_slot_bytes/2 bf16 elements — the 4096-hidden decode unit). A
// multi-row boundary folds row-major chunks of at most that many
// elements, sequentially (single outstanding, the v1 contract). Row-major
// chunking preserves the canonical per-element fold order, so every
// rank's destination stays bitwise identical across ranks.
//
// Pre-stage interface (§6.3 evolution): boundaries that fit one slot are handed
// the pinned staging buffer at stage() time — the producing GEMM writes
// the transport's send source directly and the collective runs with zero
// staging copies (the kernel only publishes ready). Boundaries above the
// slot stay on the device path (chunked, or the bulk collective class).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include <stdexcept>
#include <string>

#include "kernels/pick.hpp"
#include "kernels/sample_pick.hpp"
#include "text/tool_grammar.hpp"
#include "engine/boundary_reducer.hpp"
#include "sample/sampler.hpp"
#include "engine/step_timing.hpp"
#include "net/collective_bus.hpp"

namespace dgpp {

// One latency slot's bf16 capacity: the decode boundary's row ceiling.
// 8192-byte slots hold one hidden-4096 row (the classic decode unit);
// the Phase-2 fabric configuration holds kDecodeRows in a 64 KiB slot.
inline size_t bus_latency_slot_elems(const net::CollectiveBus& bus) {
  return bus.slot_bytes(net::BusMessageClass::kLatency) / 2;
}

struct BusBoundaryReducer final : BoundaryReducer {
  explicit BusBoundaryReducer(net::CollectiveBus& bus, int timeout_ms = 60000)
      : bus_(bus),
        timeout_ms_(timeout_ms),
        max_elems_(bus_latency_slot_elems(bus)) {}
  ~BusBoundaryReducer() override {
    if (probe_buf_) cudaFree(probe_buf_);
  }
  BusBoundaryReducer(const BusBoundaryReducer&) = delete;
  BusBoundaryReducer& operator=(const BusBoundaryReducer&) = delete;

  // The Q0 probe: one eager collective over a device scratch buffer, after
  // the fold (no handout may be held). Clamped to one latency slot.
  void probe(int rows, int cols) override {
    if (staged_ != nullptr)
      throw std::runtime_error("boundary probe: a staged handout is held");
    if (probe_buf_ == nullptr) {
      if (cudaMalloc(reinterpret_cast<void**>(&probe_buf_), max_elems_ * 2) !=
          cudaSuccess)
        throw std::runtime_error("boundary probe: scratch alloc failed");
      DGPP_CUDA_OK(cudaMemset(probe_buf_, 0, max_elems_ * 2));
    }
    const size_t elems = std::min(
        max_elems_, static_cast<size_t>(rows) * static_cast<size_t>(cols));
    const uint64_t id = submit_collective(probe_buf_, elems);
    wait_collective(id, "boundary probe");
  }

  uint16_t* stage(int rows, int hidden) override {
    if (hidden <= 0 || rows <= 0 || hidden % 2 != 0 ||
        static_cast<size_t>(rows) * static_cast<size_t>(hidden) > max_elems_)
      return nullptr;  // prefill-shaped boundary: device path
    std::string err;
    void* p = bus_.stage_next(&err);
    if (p == nullptr)
      throw std::runtime_error("boundary stage: handout rejected: " + err);
    staged_ = static_cast<uint16_t*>(p);
    return staged_;
  }

  // The bulk machine's submit without its wait (the fold overlap): only
  // prefill-class boundaries, and never beside a staged handout.
  bool begin_async(uint16_t* partial, int rows, int hidden) override {
    const size_t total = static_cast<size_t>(rows > 0 ? rows : 0) * static_cast<size_t>(hidden > 0 ? hidden : 0);
    if (staged_ != nullptr || async_id_ != 0 || hidden <= 0 || hidden % 2 != 0 ||
        static_cast<size_t>(hidden) > max_elems_ || total <= 2 * max_elems_)
      return false;
    std::string err;
    async_id_ = bus_.allreduce_bulk(partial, partial, total, &err);
    if (async_id_ == 0) throw std::runtime_error("boundary reduce: bulk rejected: " + err);
    return true;
  }
  bool begin_async_after(uint16_t* partial, int rows, int hidden, void* ready) override {
    const size_t total = static_cast<size_t>(rows > 0 ? rows : 0) * static_cast<size_t>(hidden > 0 ? hidden : 0);
    if (ready == nullptr || staged_ != nullptr || async_id_ != 0 || hidden <= 0 || hidden % 2 != 0 ||
        static_cast<size_t>(hidden) > max_elems_ || total <= 2 * max_elems_)
      return false;
    DGPP_CUDA_OK(cudaEventSynchronize(static_cast<cudaEvent_t>(ready)));
    return begin_async(partial, rows, hidden);
  }
  void end_async() override {
    if (async_id_ == 0) return;
    step_timing::Scope tick(step_timing::kFold);
    const uint64_t id = async_id_;
    async_id_ = 0;
    wait_collective(id, "boundary bulk (async)");
  }

  void reduce(uint16_t* partial, int rows, int hidden) override {
    step_timing::Scope tick(step_timing::kFold);
    if (async_id_ != 0)
      throw std::runtime_error("boundary reduce: an asynchronous fold is outstanding");
    if (hidden <= 0 || static_cast<size_t>(hidden) > max_elems_ ||
        hidden % 2 != 0)
      throw std::invalid_argument(
          "boundary reduce: hidden must be even and fit one latency slot");
    if (staged_ != nullptr) {
      if (partial != staged_)
        throw std::runtime_error(
            "boundary reduce: a staged handout is held (consume it first)");
      // The pre-staged submit consumes the handout; the fold runs in
      // place in the pinned buffer.
      std::string err;
      const size_t elems =
          static_cast<size_t>(rows) * static_cast<size_t>(hidden);
      const uint64_t id = bus_.allreduce_staged(elems, &err);
      if (id == 0)
        throw std::runtime_error("boundary reduce: staged submit rejected: " +
                                 err);
      staged_ = nullptr;
      wait_collective(id, "boundary staged reduce");
      return;
    }
    static thread_local int chunk_no = 0;
    const size_t total =
        static_cast<size_t>(rows) * static_cast<size_t>(hidden);
    // Prefill-class boundaries (well above a couple of latency chunks)
    // take the bulk machine: segment-quantized reduce-scatter + allgather,
    // the same canonical per-element chain (bitwise-equal to chunking —
    // the bus_test cross-path gate pins exactly that).
    if (total > 2 * max_elems_) {
      std::string err;
      const uint64_t id = bus_.allreduce_bulk(partial, partial, total, &err);
      if (id == 0)
        throw std::runtime_error("boundary reduce: bulk rejected: " + err);
      wait_collective(id, "boundary bulk");
      return;
    }
    // Floor: rows folded per collective (hidden itself when hidden fills
    // the slot — the decode shape, one collective per boundary).
    const int rows_per = static_cast<int>(max_elems_ / hidden);
    for (int row0 = 0; row0 < rows; row0 += rows_per) {
      const int n = std::min(rows_per, rows - row0);
      const size_t elems = static_cast<size_t>(n) * hidden;
      const uint64_t id = submit_collective(
          partial + static_cast<size_t>(row0) * hidden, elems);
      DGPP_LOG_DEBUG("boundary chunk {}: submit (rows {}..{})", chunk_no,
                     row0, row0 + n);
      wait_collective(id, "boundary chunk " + std::to_string(chunk_no));
      ++chunk_no;
    }
  }

 private:
  uint64_t submit_collective(uint16_t* at, size_t elems) {
    std::string err;
    const uint64_t id = bus_.allreduce(at, at, elems, &err);
    if (!id)
      throw std::runtime_error("boundary reduce: allreduce rejected: " + err);
    return id;
  }

  void wait_collective(uint64_t id, const std::string& what) {
    const net::BusAllReduceResult res = bus_.wait_allreduce(id, timeout_ms_);
    if (!res.ok)
      throw std::runtime_error("boundary reduce (" + what + "): " + res.error);
  }

  net::CollectiveBus& bus_;
  int timeout_ms_ = 60000;
  size_t max_elems_ = 0;        // one latency slot, in bf16
  uint16_t* staged_ = nullptr;  // held pre-stage handout, if any
  uint64_t async_id_ = 0;       // the outstanding asynchronous fold, if any
  uint16_t* probe_buf_ = nullptr;  // the Q0 probe's device scratch (lazy)
};

// ---------------------------------------------------------------------------
// The stream-ordered boundary reducer (2026-09-14, plan D9): the eager walk's
// folds as stream collectives — no host drain of the model stream before
// the fold, no engine launch on the collective stream, no host notice of
// the finish before the model may continue; the kernel form is the replay's
// (bitwise the eager machine's). The model skips the drain
// (stream_ordered) and calls settle() once per pass. Short boundaries use
// up to two latency slots. Larger boundaries drain the stream and fold
// their bulk prefix through the host, then queue any short tail.
// ---------------------------------------------------------------------------
class BusStreamReducer final : public BoundaryReducer {
 public:
  explicit BusStreamReducer(net::CollectiveBus& bus, int timeout_ms = 60000)
      : bus_(bus), timeout_ms_(timeout_ms), max_elems_(bus_latency_slot_elems(bus)) {}
  ~BusStreamReducer() override {
    if (probe_buf_) cudaFree(probe_buf_);
  }
  BusStreamReducer(const BusStreamReducer&) = delete;
  BusStreamReducer& operator=(const BusStreamReducer&) = delete;

  bool stream_ordered() const override { return stream_ != nullptr; }
  void bind_stream(void* stream) override { stream_ = static_cast<cudaStream_t>(stream); }

  void probe(int rows, int cols) override {
    if (probe_buf_ == nullptr) {
      if (cudaMalloc(reinterpret_cast<void**>(&probe_buf_), max_elems_ * 2) != cudaSuccess)
        throw std::runtime_error("boundary probe: scratch alloc failed");
      DGPP_CUDA_OK(cudaMemset(probe_buf_, 0, max_elems_ * 2));
    }
    const size_t elems = std::min(max_elems_, static_cast<size_t>(rows) * static_cast<size_t>(cols));
    issue(probe_buf_, elems);
  }

  // No pre-stage handout: the fold runs in place on the model's buffer.
  uint16_t* stage(int /*rows*/, int /*hidden*/) override { return nullptr; }

  void reduce(uint16_t* partial, int rows, int hidden) override {
    step_timing::Scope tick(step_timing::kFold);
    if (hidden <= 0 || hidden % 2 != 0)
      throw std::invalid_argument("boundary reduce: hidden must be even");
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(hidden);
    if (stream_ == nullptr) {
      // Unbound (no model stream yet): the host-driven one-shot / bulk.
      host_driven(partial, total);
      return;
    }
    // Keep short mixed walks on the model stream when a decoder pushes
    // a reader just past one latency slot. Like the host boundary reducer,
    // use bulk only beyond two slots; each chunk keeps canonical rank order.
    if (total <= 2 * max_elems_) {
      issue_pair(partial, total);
      return;
    }
    // Prefill-class boundary: the stream's folds settle, the stream
    // drains, and the bulk machine takes the buffer (its wait orders the
    // result for the host and the stream alike).
    settle();
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    // A few extra mixed-batch rows can spill a full bulk segment into a
    // second reduce-scatter/allgather wave. Fold a remainder of up to two
    // latency slots on the model stream instead. Both machines sum each
    // element in the same rank order; the completed bulk prefix and
    // stream-ordered tail are ready before the next model kernel consumes them.
    const size_t segment = bus_.bulk_segment_bytes() / 2;
    const size_t tail = segment ? total % segment : 0;
    if (total > segment && tail > 0 && tail <= 2 * max_elems_) {
      host_driven(partial, total - tail);
      issue_pair(partial + total - tail, tail);
    } else {
      host_driven(partial, total);
    }
  }

  void settle() override {
    std::string err;
    if (!bus_.allreduce_settle(timeout_ms_, &err))
      throw std::runtime_error("boundary settle: " + err);
  }

  // The fold overlap (2026-10-02, the DeepSeek-V4-Flash prefill): a
  // prefill-class fold submitted to the bulk machine without its wait —
  // the stream's folds settled and the stream drained first, as reduce()
  // does — so the model runs OTHER rows on its stream meanwhile.
  bool begin_async(uint16_t* partial, int rows, int hidden) override {
    const size_t total = static_cast<size_t>(rows > 0 ? rows : 0) * static_cast<size_t>(hidden > 0 ? hidden : 0);
    if (stream_ == nullptr || async_id_ != 0 || hidden <= 0 || hidden % 2 != 0 || total <= max_elems_) return false;
    settle();
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    std::string err;
    async_id_ = bus_.allreduce_bulk(partial, partial, total, &err);
    if (async_id_ == 0) throw std::runtime_error("boundary reduce: bulk rejected: " + err);
    return true;
  }
  // The same fold behind the producers' event: the stream's folds settle
  // and the host waits for the event alone, so the rows the caller
  // enqueued after it keep the GPU while the bulk machine takes the buffer.
  bool begin_async_after(uint16_t* partial, int rows, int hidden, void* ready) override {
    const size_t total = static_cast<size_t>(rows > 0 ? rows : 0) * static_cast<size_t>(hidden > 0 ? hidden : 0);
    if (ready == nullptr || stream_ == nullptr || async_id_ != 0 || hidden <= 0 || hidden % 2 != 0 || total <= max_elems_)
      return false;
    settle();
    DGPP_CUDA_OK(cudaEventSynchronize(static_cast<cudaEvent_t>(ready)));
    std::string err;
    async_id_ = bus_.allreduce_bulk(partial, partial, total, &err);
    if (async_id_ == 0) throw std::runtime_error("boundary reduce: bulk rejected: " + err);
    return true;
  }
  void end_async() override {
    if (async_id_ == 0) return;
    step_timing::Scope tick(step_timing::kFold);
    const uint64_t id = async_id_;
    async_id_ = 0;
    const net::BusAllReduceResult res = bus_.wait_allreduce(id, timeout_ms_);
    if (!res.ok) throw std::runtime_error("boundary bulk (async): " + res.error);
  }

 private:
  uint64_t async_id_ = 0;
  void issue(uint16_t* at, size_t elems) {
    std::string err;
    if (bus_.allreduce_stream(stream_, at, at, elems, &err) == 0)
      throw std::runtime_error("boundary reduce: stream collective rejected: " + err);
  }
  void issue_pair(uint16_t* at, size_t elems) {
    const size_t first = std::min(elems, max_elems_);
    issue(at, first);
    if (elems > first) issue(at + first, elems - first);
  }
  void host_driven(uint16_t* partial, size_t total) {
    std::string err;
    uint64_t id = 0;
    if (total > max_elems_) {
      id = bus_.allreduce_bulk(partial, partial, total, &err);
      if (id == 0) throw std::runtime_error("boundary reduce: bulk rejected: " + err);
    } else {
      id = bus_.allreduce(partial, partial, total, &err);
      if (id == 0) throw std::runtime_error("boundary reduce: allreduce rejected: " + err);
    }
    const net::BusAllReduceResult res = bus_.wait_allreduce(id, timeout_ms_);
    if (!res.ok) throw std::runtime_error("boundary reduce (host-driven): " + res.error);
  }

  net::CollectiveBus& bus_;
  int timeout_ms_ = 60000;
  size_t max_elems_ = 0;
  cudaStream_t stream_ = nullptr;
  uint16_t* probe_buf_ = nullptr;
};

// ---------------------------------------------------------------------------
// DESIGN §6.2: the boundary reducer's RECORD half — the decode walk's
// capture mode folds through the bus's graph session instead of the
// eager machine. Install it with model.set_boundary() for the capture
// only (between cudaStreamBeginCapture/EndCapture on the model's
// stream), then restore the eager reducer; the replay path never calls
// a reducer (the graph's collective nodes do the folds).
//
// stage() hands out one stable DEVICE buffer — the decode shape is
// strictly one-boundary-at-a-time, and within a recorded graph the
// producing kernel of boundary N+1 is stream-ordered behind boundary
// N's fold, so a single buffer is the whole lifetime contract. The
// baked src/dst addresses must outlive the graph. Device memory on
// purpose: only GPU kernels ever touch this buffer — the
// producing GEMV writes it, the recorded collective kernel snapshots it
// into the pinned staging rows and folds into it — and a GEMV whose 4096
// lane-0 stores land in cudaMallocHost memory pays ~20 us per launch for
// the fabric round trips (measured: N4096xK2048 76 -> 97 us), i.e. ~1.8
// ms/token across the 90 boundaries, for no reason at all.
// ---------------------------------------------------------------------------
struct GraphRecordReducer final : BoundaryReducer {
  GraphRecordReducer(net::CollectiveBus& bus, cudaStream_t capture_stream)
      : bus_(bus),
        stream_(capture_stream),
        max_elems_(bus_latency_slot_elems(bus)) {
    const cudaError_t alloc =
        cudaMalloc(reinterpret_cast<void**>(&stable_), max_elems_ * 2);
    if (alloc != cudaSuccess)
      throw std::runtime_error("graph record reducer: stable device "
                               "buffer alloc failed");
    // The Q0 probe's own stable buffer: a second baked address, so the
    // probe node never aliases the boundary's fold.
    if (cudaMalloc(reinterpret_cast<void**>(&probe_stable_), max_elems_ * 2) !=
        cudaSuccess)
      throw std::runtime_error("graph record reducer: probe buffer alloc failed");
    // Stream-ordered, never the legacy stream: a peer rank in the same
    // process (the loopback worlds) may already be capturing, and a
    // legacy-stream memset would try to synchronize with its capturing
    // stream (cudaErrorStreamCaptureImplicit, 2026-10-14).
    DGPP_CUDA_OK(cudaMemsetAsync(probe_stable_, 0, max_elems_ * 2, stream_));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }
  ~GraphRecordReducer() override {
    if (stable_) cudaFree(stable_);
    if (probe_stable_) cudaFree(probe_stable_);
  }

  void probe(int rows, int cols) override {
    const size_t elems = std::min(
        max_elems_, static_cast<size_t>(rows) * static_cast<size_t>(cols));
    std::string err;
    if (!bus_.allreduce_record(stream_, probe_stable_, probe_stable_, elems,
                               &err))
      throw std::runtime_error(
          "graph record reducer: probe allreduce_record rejected: " + err);
  }
  GraphRecordReducer(const GraphRecordReducer&) = delete;
  GraphRecordReducer& operator=(const GraphRecordReducer&) = delete;

  uint16_t* stage(int rows, int hidden) override {
    // Decode-shaped only: the capture walk is a few rows at hidden 4096
    // (one, or a speculative verify's kSpecRows), within one slot.
    // Anything else is the prefill shape, which never reaches this
    // reducer (the app captures after prefill, restore-before-prefill).
    if (hidden <= 0 || rows <= 0 || hidden % 2 != 0 ||
        static_cast<size_t>(rows) * static_cast<size_t>(hidden) > max_elems_)
      return nullptr;
    return stable_;
  }

  void reduce(uint16_t* partial, int rows, int hidden) override {
    if (partial != stable_)
      throw std::runtime_error(
          "graph record reducer: the capture walk must fold the staged "
          "buffer (the prefill-shaped device path cannot record)");
    if (hidden <= 0 || rows <= 0 || hidden % 2 != 0 ||
        static_cast<size_t>(rows) * static_cast<size_t>(hidden) > max_elems_)
      throw std::invalid_argument(
          "graph record reducer: boundary must be decode-shaped");
    std::string err;
    const size_t elems = static_cast<size_t>(rows) * static_cast<size_t>(hidden);
    if (!bus_.allreduce_record(stream_, partial, partial, elems, &err))
      throw std::runtime_error("graph record reducer: allreduce_record "
                               "rejected: " + err);
  }

 private:
  net::CollectiveBus& bus_;
  cudaStream_t stream_ = nullptr;
  size_t max_elems_ = 0;
  uint16_t* stable_ = nullptr;  // device; baked into every recorded node
  uint16_t* probe_stable_ = nullptr;  // the Q0 probe node's baked address
};

// ---------------------------------------------------------------------------
// M6 d3: the distributed greedy pick over the existing all-reduce.
// ---------------------------------------------------------------------------

// Exact greedy argmax across ranks with NO new transport surface. The
// bus all-reduce is an elementwise SUM in canonical rank order — and a
// sum over disjoint per-rank slots IS a gather: each rank's quadruple
// rides its own slots, the fold accumulates zeros elsewhere (bitwise-
// stable; bf16 values and small digits pass through exactly), and every
// rank then decodes an IDENTICAL candidate table. The pick is
// glm_sample's canonical (value desc, id asc) — rank-consistent by
// construction, the same discipline the unit tests pin.
//
// Encoding: everything travels as 6-bit digits (bf16 holds integers only
// 0..256 exactly): ids as three (vocab ids run past 150k), the fp32 logit
// as six — its 32 bits, exactly, since 2026-09-03 the logits are the
// head's unrounded accumulators and a bf16 on the wire would put the
// rounding back at the merge. Wire shape: one latency collective for the
// candidate gather, one for the winner broadcast — both padded to the
// boundary folds' 2048 elements (see the OPEN ENGINE BUG note inside). A
// proper (value, id) arg-max all-reduce is the M9 optimization; this is
// exact and rides the proven collective contract (single outstanding, no
// other latency traffic in flight — the pick is serialized behind the
// forward's collectives in every consumer of it).
//
// `scratch` is a device (managed) buffer of >= kPickScratchElems(world)
// bf16 elements, host-writable — caller-owned so this helper allocates
// nothing inside the decode loop. A speculative verify picks R rows in
// the same two collectives (row r's quadruple in slots [r*world + rank]).
// The digit constants live with the device half (kernels/glm_pick.hpp):
// both paths speak the same wire format.
constexpr size_t kPickScratchElems(int world) {
  return static_cast<size_t>(kPickMaxRows) * world * kPickSlotsPerRank;
}

inline void pick_encode_digits(uint16_t* slots, uint64_t value, int digits) {
  for (int d = 0; d < digits; ++d)
    slots[d] = static_cast<uint16_t>((value >> (6 * d)) & 63);
}
inline uint64_t pick_decode_digits(const uint16_t* slots, int digits) {
  uint64_t value = 0;
  for (int d = 0; d < digits; ++d)
    value |= static_cast<uint64_t>(slots[d] & 63) << (6 * d);
  return value;
}

inline std::vector<int32_t> bus_greedy_pick_rows(
    net::CollectiveBus& bus, int rank, int world,
    const std::vector<sample::Candidate>& locals, uint16_t* scratch,
    int timeout_ms) {
  step_timing::Scope tick(step_timing::kPick);
  const int rows = static_cast<int>(locals.size());
  if (rows < 1 || rows > kPickMaxRows)
    throw std::invalid_argument("bus_greedy_pick: row count out of range");
  for (const sample::Candidate& local : locals)
    if (local.id < 0 || local.id >= (1 << (6 * kPickIdDigits)))
      throw std::invalid_argument("bus_greedy_pick: token id outside the "
                                   "6-bit-triplet encoding range");
  // The historically-vulnerable shape, now the regression proof: a small
  // plain collective (world 4: 36 elems) after a run of staged ones.
  // Before the generation-gated claim this raced a peer's in-flight
  // collective kernel (corruption or stall, whichever way the claim
  // fell); the gate pins the claim to this collective's doorbells.
  const size_t gather_elems =
      static_cast<size_t>(rows) * world * kPickSlotsPerRank;
  const auto allreduce_wait = [&](std::string* err) -> uint64_t {
    const uint64_t id = bus.allreduce(scratch, scratch, gather_elems, err);
    if (id == 0) return 0;
    const net::BusAllReduceResult r = bus.wait_allreduce(id, timeout_ms);
    if (!r.ok) {
      *err = r.error;
      return 0;
    }
    return id;
  };
  const auto slot = [&](int row, int r) {
    return scratch + (static_cast<size_t>(row) * world + r) * kPickSlotsPerRank;
  };

  // ---- gather: every rank's (value, id) per row in its own slots ------
  std::memset(scratch, 0, gather_elems * 2);
  for (int row = 0; row < rows; ++row) {
    uint16_t* mine = slot(row, rank);
    uint32_t logit_bits = 0;
    std::memcpy(&logit_bits, &locals[row].logit, sizeof(logit_bits));
    pick_encode_digits(mine, logit_bits, kPickLogitDigits);
    pick_encode_digits(mine + kPickLogitDigits,
                       static_cast<uint64_t>(locals[row].id), kPickIdDigits);
  }
  std::string err;
  if (allreduce_wait(&err) == 0)
    throw std::runtime_error("bus_greedy_pick gather: " + err);
  std::vector<int32_t> winners(static_cast<size_t>(rows));
  for (int row = 0; row < rows; ++row) {
    std::vector<sample::Candidate> cands;
    cands.reserve(static_cast<size_t>(world));
    for (int r = 0; r < world; ++r) {
      const uint16_t* q = slot(row, r);
      sample::Candidate c;
      const uint32_t logit_bits =
          static_cast<uint32_t>(pick_decode_digits(q, kPickLogitDigits));
      std::memcpy(&c.logit, &logit_bits, sizeof(c.logit));
      c.id = static_cast<int32_t>(
          pick_decode_digits(q + kPickLogitDigits, kPickIdDigits));
      cands.push_back(c);
    }
    winners[static_cast<size_t>(row)] = sample::merge_greedy(cands);
  }

  // ---- broadcast: rank 0's winner digits reach every rank -----------
  std::memset(scratch, 0, gather_elems * 2);
  if (rank == 0)
    for (int row = 0; row < rows; ++row)
      pick_encode_digits(scratch + static_cast<size_t>(row) * kPickIdDigits,
                         static_cast<uint64_t>(winners[row]), kPickIdDigits);
  if (allreduce_wait(&err) == 0)
    throw std::runtime_error("bus_greedy_pick broadcast: " + err);
  for (int row = 0; row < rows; ++row) {
    const int32_t decoded = static_cast<int32_t>(pick_decode_digits(
        scratch + static_cast<size_t>(row) * kPickIdDigits, kPickIdDigits));
    // Load-bearing readback invariant: every rank folds an identical
    // candidate table (the allreduce is bitwise-stable by contract), so
    // every rank computes the same winner and every rank must decode rank
    // 0's broadcast of it. A mismatch means THIS rank's broadcast-phase
    // readback is corrupt — the 2026-09-01 fabric race folded
    // boundary-class bf16 into the digit slots on one rank while its peers
    // were correct; this check turns that silent corruption into a loud,
    // located failure at the exact collective, on the exact rank.
    if (decoded != winners[row]) {
      std::string words;
      for (size_t i = 0; i < gather_elems; ++i) {
        if (i) words += ",";
        words += std::format("{:#06x}", scratch[i]);
      }
      throw std::runtime_error(std::format(
          "bus_greedy_pick: broadcast readback corrupt on rank {} row {} "
          "(winner {} decoded {}): scratch[{}]",
          rank, row, winners[row], decoded, words));
    }
  }
  return winners;
}

inline int32_t bus_greedy_pick(net::CollectiveBus& bus, int rank, int world,
                               sample::Candidate local, uint16_t* scratch,
                               int timeout_ms) {
  return bus_greedy_pick_rows(bus, rank, world, {local}, scratch,
                              timeout_ms)[0];
}

// ---------------------------------------------------------------------------
// M6 6b sizing instrument: exact global top-k probability mass.
// ---------------------------------------------------------------------------
// This is deliberately a teacher-forced HOST profiler, not the production
// sampler. Each rank selects its exact local top-256 from the host-visible
// vocab slice, and one disjoint-slot bus fold gathers those candidates plus
// the slice's fp64 log-sum-exp on every rank. The union contains the exact
// global top-256; all ranks merge it in canonical order and report the full-
// distribution probability mass at k={32,64,128,256}. The wire uses the same
// six-bit-digit identity encoding as the proven greedy gather, so no float is
// rounded in transport. It adds one eager collective per teacher position
// only when --sampling-profile is requested.
inline constexpr std::array<int, 4> kSamplingProfileTopKs = {32, 64, 128,
                                                             256};
inline constexpr int kSamplingProfileMaxK = 256;
inline constexpr int kSamplingProfileLseDigits = 11;  // 66 bits carry fp64
inline constexpr size_t kSamplingProfileRankElems =
    static_cast<size_t>(kSamplingProfileMaxK) * kPickSlotsPerRank +
    kSamplingProfileLseDigits;

inline constexpr size_t sampling_profile_scratch_elems(int world) {
  const size_t elems = static_cast<size_t>(world) * kSamplingProfileRankElems;
  return elems + (elems & 1);  // CollectiveBus requires an even bf16 count.
}

// The decision digest rank 0 broadcasts after the fold — bus_greedy_pick's
// load-bearing readback invariant, for the sampler: every rank folds an
// identical table and so reaches an identical decision, and every rank must
// decode rank 0's digest of it. covered_mass is a function of every
// transported digit (all merged candidates and every slice lse), so a
// corrupt readback on one rank is loud at this collective rather than a
// silently divergent token a layer later.
inline constexpr size_t kSamplingPrefixDigestDigits =
    1 + kPickIdDigits + kPickLogitDigits + kSamplingProfileLseDigits;
inline constexpr size_t kSamplingPrefixDigestElems =
    kSamplingPrefixDigestDigits + (kSamplingPrefixDigestDigits & 1);

// Width-independent form used by the production-sampler bring-up. Keeping
// this dynamic is intentional: PLAN M6.6b does not fix the device candidate
// width until the three real teacher-text profiles have landed. The scratch
// also hosts the decision digest, hence the floor.
inline constexpr size_t sampling_prefix_scratch_elems(int world,
                                                      int candidate_k) {
  if (world <= 0 || candidate_k <= 0) return 0;
  const size_t rank_elems =
      static_cast<size_t>(candidate_k) * kPickSlotsPerRank +
      kSamplingProfileLseDigits;
  const size_t elems = static_cast<size_t>(world) * rank_elems;
  const size_t table = elems + (elems & 1);  // even bf16 count for the bus
  return table > kSamplingPrefixDigestElems ? table
                                            : kSamplingPrefixDigestElems;
}

inline std::array<double, kSamplingProfileTopKs.size()>
bus_sampling_topk_masses(net::CollectiveBus& bus, int rank, int world,
                         const float* logits, int vocab_count,
                         int vocab_begin, double local_logsumexp,
                         uint16_t* scratch, int timeout_ms) {
  if (world < 1 || world > kPickMaxWorld || rank < 0 || rank >= world)
    throw std::invalid_argument("sampling profile: rank/world");
  if (vocab_count < kSamplingProfileMaxK)
    throw std::invalid_argument(
        "sampling profile: every vocab shard must contain at least 256 ids");
  if (!std::isfinite(local_logsumexp))
    throw std::invalid_argument("sampling profile: non-finite slice lse");
  const size_t elems = sampling_profile_scratch_elems(world);
  if (elems * sizeof(uint16_t) >
      bus.slot_bytes(net::BusMessageClass::kLatency))
    throw std::invalid_argument(
        "sampling profile: candidate table exceeds one latency slot");

  const std::vector<sample::Candidate> local = sample::local_topk(
      logits, vocab_count, vocab_begin, kSamplingProfileMaxK);
  std::memset(scratch, 0, elems * sizeof(uint16_t));
  uint16_t* mine = scratch + static_cast<size_t>(rank) *
                                 kSamplingProfileRankElems;
  for (int i = 0; i < kSamplingProfileMaxK; ++i) {
    const sample::Candidate& candidate = local[static_cast<size_t>(i)];
    if (candidate.id < 0 || candidate.id >= (1 << (6 * kPickIdDigits)))
      throw std::invalid_argument(
          "sampling profile: token id outside the pick encoding");
    uint32_t logit_bits = 0;
    std::memcpy(&logit_bits, &candidate.logit, sizeof(logit_bits));
    uint16_t* encoded = mine + static_cast<size_t>(i) * kPickSlotsPerRank;
    pick_encode_digits(encoded, logit_bits, kPickLogitDigits);
    pick_encode_digits(encoded + kPickLogitDigits,
                       static_cast<uint64_t>(candidate.id), kPickIdDigits);
  }
  uint64_t lse_bits = 0;
  static_assert(sizeof(lse_bits) == sizeof(local_logsumexp));
  std::memcpy(&lse_bits, &local_logsumexp, sizeof(lse_bits));
  pick_encode_digits(mine + static_cast<size_t>(kSamplingProfileMaxK) *
                                kPickSlotsPerRank,
                     lse_bits, kSamplingProfileLseDigits);

  std::string err;
  const uint64_t id = bus.allreduce(scratch, scratch, elems, &err);
  if (id == 0)
    throw std::runtime_error("sampling profile gather: " + err);
  const net::BusAllReduceResult folded = bus.wait_allreduce(id, timeout_ms);
  if (!folded.ok)
    throw std::runtime_error("sampling profile gather: " + folded.error);

  std::vector<std::vector<sample::Candidate>> shards;
  std::vector<double> slice_lses;
  shards.reserve(static_cast<size_t>(world));
  slice_lses.reserve(static_cast<size_t>(world));
  for (int r = 0; r < world; ++r) {
    const uint16_t* encoded_rank =
        scratch + static_cast<size_t>(r) * kSamplingProfileRankElems;
    std::vector<sample::Candidate> candidates;
    candidates.reserve(kSamplingProfileMaxK);
    for (int i = 0; i < kSamplingProfileMaxK; ++i) {
      const uint16_t* encoded =
          encoded_rank + static_cast<size_t>(i) * kPickSlotsPerRank;
      sample::Candidate candidate;
      const uint32_t logit_bits = static_cast<uint32_t>(
          pick_decode_digits(encoded, kPickLogitDigits));
      std::memcpy(&candidate.logit, &logit_bits, sizeof(candidate.logit));
      candidate.id = static_cast<int32_t>(pick_decode_digits(
          encoded + kPickLogitDigits, kPickIdDigits));
      candidates.push_back(candidate);
    }
    shards.push_back(std::move(candidates));

    const uint64_t bits = pick_decode_digits(
        encoded_rank + static_cast<size_t>(kSamplingProfileMaxK) *
                           kPickSlotsPerRank,
        kSamplingProfileLseDigits);
    double lse = 0.0;
    std::memcpy(&lse, &bits, sizeof(lse));
    if (!std::isfinite(lse))
      throw std::runtime_error(
          "sampling profile: gathered a non-finite slice lse");
    slice_lses.push_back(lse);
  }

  const std::vector<sample::Candidate> global =
      sample::merge_topk(std::move(shards), kSamplingProfileMaxK);
  const double lse_top =
      *std::max_element(slice_lses.begin(), slice_lses.end());
  double lse_sum = 0.0;
  for (double lse : slice_lses) lse_sum += std::exp(lse - lse_top);
  const double global_lse = lse_top + std::log(lse_sum);
  const std::vector<double> masses = sample::topk_probability_masses(
      global, global_lse,
      std::vector<int>(kSamplingProfileTopKs.begin(),
                       kSamplingProfileTopKs.end()));
  std::array<double, kSamplingProfileTopKs.size()> out{};
  std::copy(masses.begin(), masses.end(), out.begin());
  return out;
}

// The bring-up candidate width of the host sampling path: PLAN M6 6b's
// planned k=128 per rank (~9.2 KB of table per row at world 4), to be FIXED
// by the three teacher-text profiles (scripts/fabric_sampling_profile.py).
// A width that resolves less often costs fallbacks, never correctness: the
// decision is width-independent by construction (glm_sampler.hpp).
inline constexpr int kSamplingCandidates = 128;

// Rank 0's decision digest, echoed to every rank through one small latency
// collective and compared against the rank's own — bus_greedy_pick's
// load-bearing readback invariant, for every stochastic decision (the
// prefix's and the fallback's). `resolved`, the token, the logprob bits and
// one fp64 `witness` (the prefix's covered mass — a function of every
// transported digit — or the fallback's normalizer) make the digest.
inline void bus_check_decision_digest(net::CollectiveBus& bus, int rank,
                                      bool resolved,
                                      const sample::Result& result,
                                      double witness, uint16_t* scratch,
                                      int timeout_ms, const char* what) {
  std::array<uint16_t, kSamplingPrefixDigestElems> mine{};
  pick_encode_digits(mine.data(), resolved ? 1u : 0u, 1);
  pick_encode_digits(mine.data() + 1,
                     static_cast<uint64_t>(static_cast<uint32_t>(result.token)) &
                         ((1ull << (6 * kPickIdDigits)) - 1),
                     kPickIdDigits);
  uint32_t logprob_bits = 0;
  std::memcpy(&logprob_bits, &result.logprob, sizeof(logprob_bits));
  pick_encode_digits(mine.data() + 1 + kPickIdDigits, logprob_bits,
                     kPickLogitDigits);
  uint64_t witness_bits = 0;
  std::memcpy(&witness_bits, &witness, sizeof(witness_bits));
  pick_encode_digits(mine.data() + 1 + kPickIdDigits + kPickLogitDigits,
                     witness_bits, kSamplingProfileLseDigits);
  std::memset(scratch, 0, kSamplingPrefixDigestElems * sizeof(uint16_t));
  if (rank == 0) std::copy(mine.begin(), mine.end(), scratch);
  std::string err;
  const uint64_t id =
      bus.allreduce(scratch, scratch, kSamplingPrefixDigestElems, &err);
  if (id == 0) throw std::runtime_error(std::string(what) + " digest: " + err);
  const net::BusAllReduceResult echoed = bus.wait_allreduce(id, timeout_ms);
  if (!echoed.ok)
    throw std::runtime_error(std::string(what) + " digest: " + echoed.error);
  if (!std::equal(mine.begin(), mine.end(), scratch)) {
    const bool peer_resolved = pick_decode_digits(scratch, 1) != 0;
    const int32_t peer_token =
        static_cast<int32_t>(pick_decode_digits(scratch + 1, kPickIdDigits));
    std::string words;
    for (size_t i = 0; i < kSamplingPrefixDigestElems; ++i) {
      if (i) words += ",";
      words += std::format("{:#06x}", scratch[i]);
    }
    throw std::runtime_error(std::format(
        "{}: decision digest mismatch on rank {} (mine: resolved={} token={} "
        "witness={:.17g}; rank 0's decode: resolved={} token={}): scratch[{}]",
        what, rank, resolved, result.token, witness, peer_resolved,
        peer_token, words));
  }
}

// ---------------------------------------------------------------------------
// The full-logit gather (DESIGN §10, the fallback): every rank's PENALIZED
// fp32 vocab slice to every rank, as one bulk-class collective between
// windows. The wire carries each fp32 as four 8-bit digits in bf16 words:
// bf16 holds every integer in [0, 256] exactly, exactly one rank writes any
// slot (the others contribute +0), and the fold's fp32 accumulation of
// x + 0 + ... + 0 followed by the bf16 store returns x — so no logit is
// rounded, and NaN/inf/denormal/-0 payloads (which raw bf16 halves would
// canonicalize or flush) survive as digits. Cost: 8 bytes per vocab id,
// ~1.24 MB per token for this vocabulary; the design budgets it at a <1%
// fallback rate. `scratch` is caller-owned pinned memory of
// sampling_gather_scratch_elems(vocab) words, allocated before the world
// forms.
// ---------------------------------------------------------------------------
inline constexpr int kGatherDigitsPerLogit = 4;

inline constexpr size_t sampling_gather_scratch_elems(int64_t vocab) {
  return vocab <= 0 ? 0
                    : static_cast<size_t>(vocab) * kGatherDigitsPerLogit;
}

inline void bus_gather_logits(net::CollectiveBus& bus, int rank, int world,
                              const float* slice, int vocab_count,
                              int vocab_begin, int vocab_size,
                              uint16_t* scratch, int timeout_ms,
                              std::vector<float>* full) {
  if (world < 1 || world > kPickMaxWorld || rank < 0 || rank >= world)
    throw std::invalid_argument("logit gather: rank/world");
  if (slice == nullptr || scratch == nullptr || full == nullptr ||
      vocab_count <= 0 || vocab_begin < 0 || vocab_size <= 0 ||
      vocab_begin + vocab_count > vocab_size)
    throw std::invalid_argument("logit gather: invalid vocab slice");
  const size_t elems = sampling_gather_scratch_elems(vocab_size);
  std::memset(scratch, 0, elems * sizeof(uint16_t));
  for (int i = 0; i < vocab_count; ++i) {
    uint32_t bits = 0;
    std::memcpy(&bits, slice + i, sizeof(bits));
    uint16_t* digits = scratch + static_cast<size_t>(vocab_begin + i) *
                                     kGatherDigitsPerLogit;
    for (int d = 0; d < kGatherDigitsPerLogit; ++d)
      digits[d] = static_cast<uint16_t>((bits >> (8 * d)) & 0xFFu);
  }
  std::string err;
  const uint64_t id = bus.allreduce_bulk(scratch, scratch, elems, &err);
  if (id == 0) throw std::runtime_error("logit gather: " + err);
  const net::BusAllReduceResult folded = bus.wait_allreduce(id, timeout_ms);
  if (!folded.ok) throw std::runtime_error("logit gather: " + folded.error);
  full->resize(static_cast<size_t>(vocab_size));
  for (int v = 0; v < vocab_size; ++v) {
    const uint16_t* digits =
        scratch + static_cast<size_t>(v) * kGatherDigitsPerLogit;
    uint32_t bits = 0;
    for (int d = 0; d < kGatherDigitsPerLogit; ++d) {
      if (digits[d] > 0xFFu)
        throw std::runtime_error(std::format(
            "logit gather: corrupt digit {:#06x} at vocab id {} on rank {}",
            digits[d], v, rank));
      bits |= static_cast<uint32_t>(digits[d]) << (8 * d);
    }
    std::memcpy(&(*full)[static_cast<size_t>(v)], &bits, sizeof(bits));
  }
}

// Host implementation of distributed sampling. Gather a global candidate
// prefix and each slice's normalizer, decide whether the prefix is
// sufficient, then compare rank 0's decision digest through another
// collective. The in-graph picker implements the corresponding device
// path with glm_pick_local/verdict and a digest in its verdict table.
// Tests exercise encoding, rank order, penalties, RNG counters and
// fallback behavior through the real bus.
//

// `scratch` is caller-owned pinned memory of at least
// sampling_prefix_scratch_elems(world, candidate_k) bf16 words. On fallback
// the RNG counter is unchanged so the later full-logit gather can consume the
// same draw — it must run sample_reference_sharded() over the same layout,
// which is this decision over the complete list. Candidate bytes, params,
// context, seed and counter are identical on every rank by contract, which
// is exactly the invariant the 2026-09-01 fabric race broke on one rank's
// readback; the digest makes a breach loud at the collective.
// The fold half: this rank's penalized slice, the exact global candidate
// prefix and the fold normalizer every rank now holds identically. The
// decisions (bus_sampling_prefix, bus_sample_row, bus_spec_accept) run
// over it; the penalized slice is the gather fallback's source.
struct SamplingFold {
  std::vector<sample::Candidate> prefix;
  double normalizer = 0.0;
  std::vector<float> adjusted;
  int vocab = 0;  // the decision's vocabulary: the mask's allowed count
                  // when constrained (M6 6g), else vocab_size
};

// `mask` (optional, M6 6g): the position's token mask — the slice's
// excluded ids become -inf (absent) before the local top-k and lse, the
// decision's vocabulary becomes the allowed count.
inline SamplingFold bus_sampling_fold(
    net::CollectiveBus& bus, int rank, int world, const float* logits,
    int vocab_count, int vocab_begin, int vocab_size,
    const sample::Params& params,
    const std::vector<int32_t>& context_ids, int candidate_k,
    uint16_t* scratch, int timeout_ms,
    const text::TokenMask* mask = nullptr, const float* bias = nullptr) {
  if (world < 1 || world > kPickMaxWorld || rank < 0 || rank >= world)
    throw std::invalid_argument("sampling prefix: rank/world");
  if (!(params.temperature > 0.0f) || !std::isfinite(params.temperature))
    throw std::invalid_argument(
        "sampling prefix: temperature must be finite and > 0 "
        "(temperature <= 0 is bus_greedy_pick's path)");
  if (logits == nullptr || scratch == nullptr || vocab_count <= 0 ||
      vocab_begin < 0 || vocab_size <= 0 ||
      vocab_begin + vocab_count > vocab_size)
    throw std::invalid_argument("sampling prefix: invalid vocab slice");
  if (candidate_k < 1 || candidate_k > kSamplingProfileMaxK)
    throw std::invalid_argument(
        "sampling prefix: candidate_k must be in [1, 256]");
  // A shard narrower than k transports every id it has; the remaining
  // candidate slots carry the EMPTY id (an id no vocabulary reaches, the
  // device kernel's convention too) and the decoder skips them.

  const size_t rank_elems =
      static_cast<size_t>(candidate_k) * kPickSlotsPerRank +
      kSamplingProfileLseDigits;
  const size_t elems = sampling_prefix_scratch_elems(world, candidate_k);
  if (elems * sizeof(uint16_t) >
      bus.slot_bytes(net::BusMessageClass::kLatency))
    throw std::invalid_argument(
        "sampling prefix: candidate table exceeds one latency slot");

  std::vector<float> adjusted(logits, logits + vocab_count);
  sample::apply_context_penalties(adjusted.data(), vocab_count, vocab_begin, params, context_ids);
  sample::apply_bias(adjusted.data(), vocab_count, vocab_begin, bias);
  const bool constrained = mask != nullptr && mask->constrained();
  if (constrained)
    sample::apply_mask(adjusted.data(), vocab_count, vocab_begin,
                           mask->words.data(), vocab_size);
  const std::vector<sample::Candidate> local = sample::local_topk(
      adjusted.data(), vocab_count, vocab_begin, candidate_k);
  const double local_lse = sample::slice_logsumexp(
      adjusted.data(), vocab_count, params.temperature);

  std::memset(scratch, 0, elems * sizeof(uint16_t));
  uint16_t* mine = scratch + static_cast<size_t>(rank) * rank_elems;
  for (int i = 0; i < candidate_k; ++i) {
    uint16_t* encoded = mine + static_cast<size_t>(i) * kPickSlotsPerRank;
    if (i >= static_cast<int>(local.size())) {
      pick_encode_digits(encoded + kPickLogitDigits, kSampleEmptyId,
                         kPickIdDigits);
      continue;
    }
    const sample::Candidate& candidate = local[static_cast<size_t>(i)];
    if (candidate.id < 0 || candidate.id >= vocab_size ||
        candidate.id >= (1 << (6 * kPickIdDigits)))
      throw std::invalid_argument(
          "sampling prefix: token id outside the pick encoding");
    uint32_t logit_bits = 0;
    std::memcpy(&logit_bits, &candidate.logit, sizeof(logit_bits));
    pick_encode_digits(encoded, logit_bits, kPickLogitDigits);
    pick_encode_digits(encoded + kPickLogitDigits,
                       static_cast<uint64_t>(candidate.id), kPickIdDigits);
  }
  uint64_t lse_bits = 0;
  std::memcpy(&lse_bits, &local_lse, sizeof(lse_bits));
  pick_encode_digits(
      mine + static_cast<size_t>(candidate_k) * kPickSlotsPerRank, lse_bits,
      kSamplingProfileLseDigits);

  std::string err;
  const uint64_t id = bus.allreduce(scratch, scratch, elems, &err);
  if (id == 0)
    throw std::runtime_error("sampling prefix gather: " + err);
  const net::BusAllReduceResult folded = bus.wait_allreduce(id, timeout_ms);
  if (!folded.ok)
    throw std::runtime_error("sampling prefix gather: " + folded.error);

  std::vector<std::vector<sample::Candidate>> shards;
  std::vector<double> slice_lses;
  shards.reserve(static_cast<size_t>(world));
  slice_lses.reserve(static_cast<size_t>(world));
  for (int r = 0; r < world; ++r) {
    const uint16_t* encoded_rank =
        scratch + static_cast<size_t>(r) * rank_elems;
    std::vector<sample::Candidate> candidates;
    candidates.reserve(static_cast<size_t>(candidate_k));
    for (int i = 0; i < candidate_k; ++i) {
      const uint16_t* encoded =
          encoded_rank + static_cast<size_t>(i) * kPickSlotsPerRank;
      sample::Candidate candidate;
      const uint32_t id = static_cast<uint32_t>(
          pick_decode_digits(encoded + kPickLogitDigits, kPickIdDigits));
      if (id >= static_cast<uint32_t>(vocab_size)) break;  // the empty tail
      const uint32_t logit_bits = static_cast<uint32_t>(
          pick_decode_digits(encoded, kPickLogitDigits));
      std::memcpy(&candidate.logit, &logit_bits, sizeof(candidate.logit));
      candidate.id = static_cast<int32_t>(id);
      candidates.push_back(candidate);
    }
    shards.push_back(std::move(candidates));

    const uint64_t bits = pick_decode_digits(
        encoded_rank + static_cast<size_t>(candidate_k) * kPickSlotsPerRank,
        kSamplingProfileLseDigits);
    double lse = 0.0;
    std::memcpy(&lse, &bits, sizeof(lse));
    slice_lses.push_back(lse);
  }

  SamplingFold fold;
  fold.prefix = sample::merge_topk(std::move(shards), candidate_k);
  fold.normalizer = sample::merge_logsumexp(slice_lses);
  fold.adjusted = std::move(adjusted);
  fold.vocab = constrained ? mask->allowed : vocab_size;
  return fold;
}

inline sample::PrefixDecision bus_sampling_prefix(
    net::CollectiveBus& bus, int rank, int world, const float* logits,
    int vocab_count, int vocab_begin, int vocab_size,
    const sample::Params& params, sample::Rng& rng,
    const std::vector<int32_t>& context_ids, int candidate_k,
    uint16_t* scratch, int timeout_ms) {
  const SamplingFold fold = bus_sampling_fold(
      bus, rank, world, logits, vocab_count, vocab_begin, vocab_size, params,
      context_ids, candidate_k, scratch, timeout_ms);
  const sample::PrefixDecision decision = sample::sample_from_prefix(
      fold.prefix, fold.vocab, fold.normalizer, params, rng);

  // ---- readback invariant: rank 0's decision digest reaches every rank --
  bus_check_decision_digest(bus, rank, decision.resolved, decision.result,
                            decision.covered_mass, scratch, timeout_ms,
                            "bus_sampling_prefix");
  return decision;
}

// One row's exact sample over the bus, fallback included (the eager
// engines' step): the fold, the prefix decision, else the gather of the
// penalized slices and the complete decision under the transported
// normalizer with the reserved draw; rank 0's digest either way.
// `mask` (optional, M6 6g): the position's token mask.
inline sample::Result bus_sample_row(
    net::CollectiveBus& bus, int rank, int world, const float* logits,
    int vocab_count, int vocab_begin, int vocab_size,
    const sample::Params& params, sample::Rng& rng,
    const std::vector<int32_t>& context_ids, int candidate_k,
    uint16_t* prefix_scratch, uint16_t* gather_scratch, int timeout_ms,
    std::vector<float>* gather_buffer,
    const text::TokenMask* mask = nullptr, const float* bias = nullptr) {
  if (params.temperature <= 0.0f) {
    // The greedy decision with logprobs, penalties or a mask: the fold at
    // temperature 1 (penalties and the mask applied), the canonical first
    // candidate under the raw normalizer — the greedy pick's token (the
    // masked argmax), no draw.
    sample::Params raw = params;
    raw.temperature = 1.0f;
    const SamplingFold fold = bus_sampling_fold(
        bus, rank, world, logits, vocab_count, vocab_begin, vocab_size, raw,
        context_ids, candidate_k, prefix_scratch, timeout_ms, mask, bias);
    const sample::Result r = sample::greedy_from_prefix(
        fold.prefix, fold.normalizer, params.logprobs);
    bus_check_decision_digest(bus, rank, true, r, fold.normalizer,
                              prefix_scratch, timeout_ms, "bus_sample_row greedy");
    return r;
  }
  const SamplingFold fold = bus_sampling_fold(
      bus, rank, world, logits, vocab_count, vocab_begin, vocab_size, params,
      context_ids, candidate_k, prefix_scratch, timeout_ms, mask, bias);
  const sample::PrefixDecision d = sample::sample_from_prefix(
      fold.prefix, fold.vocab, fold.normalizer, params, rng);
  if (d.resolved) {
    bus_check_decision_digest(bus, rank, true, d.result, d.covered_mass,
                              prefix_scratch, timeout_ms, "bus_sample_row");
    return d.result;
  }
  bus_gather_logits(bus, rank, world, fold.adjusted.data(), vocab_count,
                    vocab_begin, vocab_size, gather_scratch, timeout_ms,
                    gather_buffer);
  const sample::Result r = sample::sample_complete_logits(
      gather_buffer->data(), vocab_size, fold.normalizer, params, rng);
  bus_check_decision_digest(bus, rank, true, r, fold.normalizer,
                            prefix_scratch, timeout_ms,
                            "bus_sample_row fallback");
  return r;
}

// The speculative step's row-0 decision over the bus (DESIGN §9): accept
// the draft with its exact probability or sample the residual — the fold,
// spec_accept_from_prefix, else the gather and spec_accept_complete; the
// digest carries the accept flag as its `resolved` bit.
// `mask` (optional, M6 6g): row 0's token mask; a masked draft is rejected
// outright (its probability is 0 without the prefix having to show it).
inline sample::SpecPrefixDecision bus_spec_accept(
    net::CollectiveBus& bus, int rank, int world, const float* logits,
    int vocab_count, int vocab_begin, int vocab_size, int32_t draft,
    const sample::Params& params, sample::Rng& rng,
    const std::vector<int32_t>& context_ids, int candidate_k,
    uint16_t* prefix_scratch, uint16_t* gather_scratch, int timeout_ms,
    std::vector<float>* gather_buffer,
    const text::TokenMask* mask = nullptr, const float* bias = nullptr) {
  const SamplingFold fold = bus_sampling_fold(
      bus, rank, world, logits, vocab_count, vocab_begin, vocab_size, params,
      context_ids, candidate_k, prefix_scratch, timeout_ms, mask, bias);
  const bool draft_excluded =
      mask != nullptr && mask->constrained() && !mask->allows(draft);
  sample::SpecPrefixDecision d = sample::spec_accept_from_prefix(
      fold.prefix, fold.vocab, fold.normalizer, draft, params, rng,
      draft_excluded);
  if (!d.resolved) {
    bus_gather_logits(bus, rank, world, fold.adjusted.data(), vocab_count,
                      vocab_begin, vocab_size, gather_scratch, timeout_ms,
                      gather_buffer);
    d = sample::spec_accept_complete(gather_buffer->data(), vocab_size,
                                         fold.normalizer, draft, params, rng);
  }
  bus_check_decision_digest(bus, rank, d.accepted, d.result, d.normalizer,
                            prefix_scratch, timeout_ms, "bus_spec_accept");
  return d;
}

// ---------------------------------------------------------------------------
// The pick ON THE DEVICE (DESIGN §9, the on-device step): the same exact
// argmax as bus_greedy_pick_rows, as two kernels around one collective —
// glm_pick_local encodes this rank's per-row argmax into the wire table,
// the bus SUM-folds the table (a gather over disjoint slots), and
// glm_pick_verdict decodes every rank's identical table into the verdict
// (winners, accepted rows, next token). Recorded, the three are graph
// nodes behind the head GEMV, so a replayed step ends with its verdict in
// pinned memory and the host reads three ints instead of running a
// 38k-column scan, two collectives and a judge. The broadcast collective
// is gone: every rank computes the verdict itself; the readback invariant
// it carried is the digest group (see kernels/glm_pick.hpp), checked at
// the NEXT pick — one step late, still loud and located.
//
// SLOTS: a step may hold more than one pick before the host looks (the
// verify's and the in-graph draft's); each pick names a slot and its
// verdict/locals land in that slot's mirrors. The digest carry is one
// chain across every pick of every slot.
//
// Buffers are owned here and baked into the recorded nodes (device table,
// device carry digest, per-slot pinned verdict/locals and device verdict),
// so the picker must outlive the graphs it recorded. One picker per rank
// per bus; picks are stream-ordered, never concurrent (the decode
// contract).
// ---------------------------------------------------------------------------
class DevicePicker {
 public:
  // The verify's pick and one per draft position (kSpecRows - 1 chained
  // drafts at most, 2026-09-06; 6 since the DSpark block, 2026-09-13).
  static constexpr int kSlots = 6;

  // `sampling_candidates` > 0 arms the SAMPLING pick (kernels/
  // glm_sample_pick.hpp): a wider table (k candidates + the slice lse per
  // rank per row) and the sampling verdict, selected per record/run by
  // Inputs::specs. The table is allocated for the widest shape; whether a
  // given row count fits one latency slot is checked when it is recorded
  // (glm_sample_candidates_that_fit sizes k for the caller).
  // `max_rows`: the widest pick this picker records or runs (the engine's
  // fixed decode batch, engine/decode_outputs.hpp; at most kPickMaxRows) —
  // the wire table and the sampling scratch are allocated for it.
  DevicePicker(net::CollectiveBus& bus, int rank, int world,
                  int timeout_ms = 60000, int sampling_candidates = 0,
                  int max_rows = kPickMaxRows)
      : bus_(bus),
        rank_(rank),
        world_(world),
        timeout_ms_(timeout_ms),
        candidates_(sampling_candidates),
        max_rows_(max_rows) {
    if (world < 1 || world > kPickMaxWorld || rank < 0 || rank >= world)
      throw std::invalid_argument("DevicePicker: rank/world");
    if (candidates_ < 0 || candidates_ > kSampleMaxCandidates)
      throw std::invalid_argument("DevicePicker: sampling candidates");
    if (max_rows_ < 1 || max_rows_ > kPickMaxRows)
      throw std::invalid_argument("DevicePicker: max_rows outside [1, " +
                                  std::to_string(kPickMaxRows) + "]");
    size_t table_bytes = device_pick_table_elems(max_rows_, world) * 2;
    if (table_bytes > bus.slot_bytes(net::BusMessageClass::kLatency))
      throw std::invalid_argument(
          "DevicePicker: the pick table exceeds one latency slot");
    if (candidates_ > 0)
      table_bytes = std::max(
          table_bytes,
          device_sample_table_elems(max_rows_, world, candidates_) * 2);
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&table_), table_bytes));
    if (candidates_ > 0) {
      device_sample_verdict_prepare();  // the verdict kernel's dynamic shared memory, before any capture
      DGPP_CUDA_OK(cudaMallocHost(
          reinterpret_cast<void**>(&outcomes_),
          sizeof(SampleOutcome) * kSlots * kPickMaxRequests));
      for (int i = 0; i < kSlots * kPickMaxRequests; ++i)
        outcomes_[i] = SampleOutcome{};
      DGPP_CUDA_OK(cudaMalloc(
          reinterpret_cast<void**>(&sample_scratch_),
          sizeof(double) * device_sample_scratch_elems(
                               max_rows_, kSampleLseChunk * kSampleMaxChunks)));
    }
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&carry_), sizeof(uint64_t)));
    DGPP_CUDA_OK(cudaMemset(carry_, 0, sizeof(uint64_t)));
    DGPP_CUDA_OK(cudaMallocHost(
        reinterpret_cast<void**>(&verdict_),
        sizeof(PickVerdict) * kSlots * kPickMaxRequests));
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&device_verdict_),
                            sizeof(PickVerdict) * kSlots *
                                kPickMaxRequests));
    DGPP_CUDA_OK(cudaMemset(device_verdict_, 0,
                            sizeof(PickVerdict) * kSlots *
                                kPickMaxRequests));
    DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&locals_),
                                sizeof(PickLocal) * kPickMaxRows * kSlots));
    for (int i = 0; i < kSlots * kPickMaxRequests; ++i)
      verdict_[i] = PickVerdict{};
  }
  ~DevicePicker() {
    if (table_) cudaFree(table_);
    if (carry_) cudaFree(carry_);
    if (device_verdict_) cudaFree(device_verdict_);
    if (verdict_) cudaFreeHost(verdict_);
    if (locals_) cudaFreeHost(locals_);
    if (outcomes_) cudaFreeHost(outcomes_);
    if (sample_scratch_) cudaFree(sample_scratch_);
  }
  DevicePicker(const DevicePicker&) = delete;
  DevicePicker& operator=(const DevicePicker&) = delete;

  // The pick's inputs: this rank's fp32 logits [rows, vocab_count] on the
  // device (the model's head output), the slice's first vocab id, and the
  // rows' fed tokens on the device (the judge's right-hand side; any valid
  // pointer at one row per request, where row 0 always stands). `slot` names
  // the mirrors the verdict lands in. `row_select` makes a scalar pick, or
  // one candidate per request in a fixed batch, read that request's last
  // accepted logits row: the in-graph draft's head runs on the wider verify
  // layout and its pick selects one row from each group.
  struct Inputs {
    const float* logits = nullptr;
    int rows = 0;
    int vocab_count = 0;
    int vocab_begin = 0;
    const int64_t* fed = nullptr;
    int slot = 0;
    // One (default) or several independent request verdicts. Candidate rows
    // are packed request-major. positions marks fixed-shape padding; its
    // stride defaults to rows_per_request. A selected draft pick has one
    // candidate per request and reads from wider source_row_stride groups.
    int requests = 1;
    int rows_per_request = 0;
    const int64_t* positions = nullptr;
    int position_stride = 0;
    const PickVerdict* row_select = nullptr;
    int source_row_stride = 0;
    // The SAMPLING pick (a picker built with sampling_candidates > 0): the
    // per-request device specs, the [requests][vocab_size] count table and
    // the vocabulary size. Rows per request is 1 (T=1) or 2 (the MTP
    // verify); the logits are penalized IN PLACE for sampled requests.
    SampleSpec* specs = nullptr;
    const int32_t* request_map = nullptr;  // compact group -> physical sampling slot (-1: padding)
    int32_t* counts = nullptr;
    int vocab_size = 0;
    // Constrained decoding (M6 6g): the rows' token masks on the device,
    // device_sample_mask_words(vocab_size) words per row (null: none).
    const uint32_t* masks = nullptr;
    int mask_stride = 0;
    // The logit bias: [requests][vocab_size] floats, read for
    // the rows whose spec says `biased` (null: no request biases).
    const float* bias = nullptr;
    // The drafts' proposals (2026-09-10, kernels/sample_pick.hpp): what the
    // verify reads for its row-0 accept test (`proposals_in`) and what a
    // DRAFT pick writes for the next step (`proposals_out`, one row per
    // request, at [request][draft_index]); `proposals_out_host` is the
    // pinned mirror the host's fallback reads. A sampling draft pick also
    // sets `row_select`, which the verify never does.
    const DraftProposal* proposals_in = nullptr;
    DraftProposal* proposals_out = nullptr;
    DraftProposal* proposals_out_host = nullptr;
    int draft_index = 0;
    // The sampled chain's rule (2026-10-05): 1 = block verification
    // (sample::block_verify, the whole block decided jointly), 0 = the
    // token-by-token test. Baked into the captured verdict node.
    int block_verify = 0;
  };
  bool sampling() const { return candidates_ > 0; }
  int sampling_candidates() const { return candidates_; }

  // CAPTURE: enqueues the three nodes on `stream` (the caller is between
  // cudaStreamBeginCapture/EndCapture on it, inside the bus's record
  // session). The verdict is readable after the replay's stream sync and
  // graph_replay_finish, via verdict(slot).
  void record(cudaStream_t stream, const Inputs& in) {
    validate(in);
    std::string err;
    if (in.specs != nullptr) {
      validate_sampling(in);
      sample_local(stream, in);
      if (!bus_.allreduce_record(
              stream, table_, table_,
              device_sample_table_elems(in.rows, world_, candidates_), &err))
        throw std::runtime_error(
            "device sampling pick: allreduce_record rejected: " + err);
      sample_verdict(stream, in);
      return;
    }
    device_pick_local_batched(
        in.logits, in.rows, in.vocab_count, in.vocab_begin, rank_, world_,
        carry_, table_, locals_ + in.slot * kPickMaxRows, stream,
        in.row_select, in.requests, source_stride(in));
    if (!bus_.allreduce_record(stream, table_, table_,
                               device_pick_table_elems(in.rows, world_), &err))
      throw std::runtime_error("device pick: allreduce_record rejected: " +
                               err);
    device_pick_verdict_batched(
        table_, in.rows, world_, rank_, in.fed, in.positions, in.requests,
        rows_per_request(in), position_stride(in), verdict_slot(in.slot),
        device_verdict_slot(in.slot), carry_, stream);
  }

  // EAGER: the same three with the eager collective between (the draft
  // between graph windows, the non-graph paths). Returns after the stream
  // is synced; the verdict is checked.
  const PickVerdict& run(cudaStream_t stream, const Inputs& in) {
    step_timing::Scope tick(step_timing::kPick);
    validate(in);
    const bool sampling = in.specs != nullptr;
    if (sampling) {
      validate_sampling(in);
      sample_local(stream, in);
    } else {
      device_pick_local_batched(
          in.logits, in.rows, in.vocab_count, in.vocab_begin, rank_, world_,
          carry_, table_, locals_ + in.slot * kPickMaxRows, stream,
          in.row_select, in.requests, source_stride(in));
    }
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    std::string err;
    const size_t elems =
        sampling ? device_sample_table_elems(in.rows, world_, candidates_)
                 : device_pick_table_elems(in.rows, world_);
    const uint64_t id = bus_.allreduce(table_, table_, elems, &err);
    if (id == 0)
      throw std::runtime_error("device pick: allreduce rejected: " + err);
    const net::BusAllReduceResult res = bus_.wait_allreduce(id, timeout_ms_);
    if (!res.ok) throw std::runtime_error("device pick gather: " + res.error);
    if (sampling)
      sample_verdict(stream, in);
    else
      device_pick_verdict_batched(
          table_, in.rows, world_, rank_, in.fed, in.positions, in.requests,
          rows_per_request(in), position_stride(in), verdict_slot(in.slot),
          device_verdict_slot(in.slot), carry_, stream);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    return verdict(in.slot);
  }

  // The sampling verdict's outcome beside verdict(slot, request): fallback,
  // counter, normalizer, logprob. Valid once the pick's stream work
  // completed (pinned).
  const SampleOutcome& outcome(int slot = 0, int request = 0) const {
    if (!sampling())
      throw std::logic_error("device pick: no sampling outcome on a greedy "
                             "picker");
    check_slot(slot);
    check_request(request);
    return outcomes_[slot * kPickMaxRequests + request];
  }
  // The outcomes table itself (pinned; slot-major, kPickMaxRequests per
  // slot) for a kernel node that reads a pick's outcome on the device (the
  // draft-probability confidence, device_sample_draft_confidence).
  const SampleOutcome* outcomes_table() const { return outcomes_; }
  static constexpr int outcomes_slot_stride() { return kPickMaxRequests; }

  // Slot `slot`'s last verdict (pinned; valid once its stream work
  // completed). Throws when the digest group disagreed: some rank
  // computed a different verdict at the previous pick — its table was not
  // the others' table.
  const PickVerdict& verdict(int slot = 0, int request = 0) const {
    check_slot(slot);
    check_request(request);
    const PickVerdict& v =
        verdict_[slot * kPickMaxRequests + request];
    if (v.digest_mismatch != 0) {
      std::string digests;
      for (int k = 0; k < world_; ++k)
        digests += std::format("{}rank {}: {:#016x}", k ? ", " : "", k,
                               v.peer_digests[k]);
      throw std::runtime_error(std::format(
          "device pick: verdict digests diverged at the previous pick "
          "(rank {} sees mismatch mask {:#x} at slot {}): {}",
          rank_, v.digest_mismatch, slot, digests));
    }
    return v;
  }
  // Row `row`'s local argmax / runner-up in slot `slot` (the gen log's
  // fields).
  const PickLocal& local(int row, int slot = 0) const {
    check_slot(slot);
    if (row < 0 || row >= kPickMaxRows)
      throw std::out_of_range("device pick: local row");
    return locals_[slot * kPickMaxRows + row];
  }
  // Slot `slot`'s verdict on the device: the address the device-side
  // consumers (the model's commit kernel, the in-graph draft) read.
  const PickVerdict* device_verdict(int slot = 0) const {
    check_slot(slot);
    return device_verdict_slot(slot);
  }
  int rank() const { return rank_; }
  int world() const { return world_; }

 private:
  static void check_slot(int slot) {
    if (slot < 0 || slot >= kSlots)
      throw std::out_of_range("device pick: slot");
  }
  static void check_request(int request) {
    if (request < 0 || request >= kPickMaxRequests)
      throw std::out_of_range("device pick: request");
  }
  static int rows_per_request(const Inputs& in) {
    return in.rows_per_request > 0 ? in.rows_per_request : in.rows;
  }
  static int position_stride(const Inputs& in) {
    return in.position_stride > 0 ? in.position_stride
                                  : rows_per_request(in);
  }
  static int source_stride(const Inputs& in) {
    return in.source_row_stride > 0 ? in.source_row_stride
                                    : rows_per_request(in);
  }
  PickVerdict* verdict_slot(int slot) const {
    return verdict_ + slot * kPickMaxRequests;
  }
  PickVerdict* device_verdict_slot(int slot) const {
    return device_verdict_ + slot * kPickMaxRequests;
  }
  void validate(const Inputs& in) const {
    check_slot(in.slot);
    if (in.logits == nullptr || in.fed == nullptr)
      throw std::invalid_argument("device pick: null inputs");
    if (in.rows < 1 || in.rows > max_rows_)
      throw std::invalid_argument("device pick: rows outside [1, " +
                                  std::to_string(max_rows_) +
                                  "] (the picker's max_rows)");
    if (in.vocab_count < 1 || in.vocab_begin < 0)
      throw std::invalid_argument("device pick: vocab slice");
    if (in.requests < 1 || in.requests > kPickMaxRequests ||
        rows_per_request(in) < 1 ||
        in.requests * rows_per_request(in) != in.rows)
      throw std::invalid_argument("device pick: request shape");
    if (in.positions != nullptr &&
        position_stride(in) < rows_per_request(in))
      throw std::invalid_argument("device pick: position stride");
    if (in.row_select != nullptr &&
        (rows_per_request(in) != 1 || source_stride(in) < 1))
      throw std::invalid_argument(
          "device pick: row_select needs one candidate per request");
  }

  void validate_sampling(const Inputs& in) const {
    if (!sampling())
      throw std::logic_error(
          "device pick: sampling inputs on a picker built without "
          "sampling candidates");
    if (in.vocab_size < 1 ||
        (in.counts == nullptr && in.proposals_out == nullptr))
      throw std::invalid_argument(
          "device pick: sampling needs the count table and the vocabulary "
          "(only a draft pick, which commits no context, may omit the "
          "counts)");
    if (rows_per_request(in) < 1 || rows_per_request(in) > kSampleVerdictRows)
      throw std::invalid_argument(
          "device pick: the sampling verdict decides T=1 rows or an MTP "
          "verify of up to " + std::to_string(kSampleVerdictRows) + " rows");
    if (in.row_select != nullptr &&
        (rows_per_request(in) != 1 || in.source_row_stride < 1))
      throw std::invalid_argument(
          "device pick: a selected sampling row is a one-row draft pick");
    if (device_sample_table_elems(in.rows, world_, candidates_) * 2 >
        bus_.slot_bytes(net::BusMessageClass::kLatency))
      throw std::invalid_argument(
          "device pick: the sampling table for these rows exceeds one "
          "latency slot (glm_sample_candidates_that_fit sizes k)");
  }
  void sample_local(cudaStream_t stream, const Inputs& in) {
    device_sample_local(const_cast<float*>(in.logits), in.rows, in.vocab_count, in.vocab_begin,
                        in.vocab_size, rank_, world_, candidates_, in.specs, rows_per_request(in),
                        in.fed, in.positions, position_stride(in), in.counts, in.bias, in.masks,
                        in.mask_stride, carry_, table_, locals_ + in.slot * kPickMaxRows,
                        sample_scratch_, stream, in.row_select,
                        in.row_select != nullptr ? source_stride(in) : 0, in.request_map);
  }
  void sample_verdict(cudaStream_t stream, const Inputs& in) {
    device_sample_verdict(table_, in.rows, world_, rank_, candidates_, in.vocab_size, in.specs,
                          in.requests, rows_per_request(in), in.fed, in.positions,
                          position_stride(in), in.counts, in.masks, in.mask_stride,
                          verdict_slot(in.slot), device_verdict_slot(in.slot),
                          outcomes_ + in.slot * kPickMaxRequests, carry_, stream, in.proposals_in,
                          in.proposals_out, in.proposals_out_host, in.draft_index, in.request_map,
                          in.block_verify);
  }

  net::CollectiveBus& bus_;
  int rank_ = 0;
  int world_ = 1;
  int timeout_ms_ = 60000;
  int candidates_ = 0;
  int max_rows_ = kPickMaxRows;  // the widest pick (the tables' allocation)
  SampleOutcome* outcomes_ = nullptr;
  double* sample_scratch_ = nullptr;   // device: the local pick's partials
  uint16_t* table_ = nullptr;          // device: the wire table
  uint64_t* carry_ = nullptr;          // device: last verdict's digest
  PickVerdict* verdict_ = nullptr;  // pinned [kSlots][kPickMaxRequests]
  PickVerdict* device_verdict_ = nullptr;  // device, same shape
  PickLocal* locals_ = nullptr;     // pinned [kSlots][kPickMaxRows]
};

}  // namespace dgpp
