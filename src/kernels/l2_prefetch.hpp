#pragma once
// L2 weight prefetch. The decode step is one serial chain of
// kernels, and whenever the chain sits in a latency-bound phase — the bus
// all-reduce, the KDA recurrence, the DSA select/absorb, the router's
// top-k — the memory system idles. Weights, unlike routed experts, are
// known in advance, so a low-priority side stream reads the NEXT kernels'
// bytes into L2 (24 MB on the GB10) while the chain waits; the consuming
// GEMV then hits L2 for that prefix. Pure performance: nothing the model
// reads is written, so every result is bit-identical with it on or off.
//
// Capture-safe: the side stream forks from and joins the main stream
// through events, which under stream capture become graph edges.
#include <cstddef>
#include <string>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// How hard a prefetch pushes the memory system. On the GB10 the GPU, the
// CPU and the NIC share one LPDDR5X controller: a prefetch with megabytes
// in flight lifts EVERYONE's memory latency (Little's law — 3 MB in flight
// at 240 GB/s is 12 us of queue), which stalls the bus engine's posts and
// the peers' handshakes during a collective. Full keeps enough in flight
// for line rate; Light caps the in-flight bytes so the collective's
// latency-sensitive path stays fast while DRAM still moves ~150 GB/s.
enum class PrefetchRate { Off, Light, Full };

// Streams [ptr, ptr + bytes) through L2 (cache-global loads, no L1) with a
// modest grid so a concurrently running chain kernel still finds SM slots.
// bytes == 0 or ptr == nullptr is a no-op. ptr need not be aligned.
// `persisting`: the loads carry an L2 access-policy window over the range
// (cudaAccessPropertyPersisting), so the lines land in the persisting
// set-aside (l2_set_persisting_limit) and survive the streaming traffic
// that follows — the routed experts' — until later persisting fills
// replace them. Capture-safe (a launch attribute).
void launch_l2_prefetch(const void* ptr, size_t bytes, PrefetchRate rate,
                        cudaStream_t stream, bool persisting = false);

// The L2 set-aside for persisting accesses (2026-09-22, the MiMo decode
// step's early projection prefetch: docs/mimo_v26_flash_plan.md §7.1):
// the granted size in bytes (0: unavailable). While persisting lines hold
// it, normal accesses have the rest of the L2; l2_reset_persisting()
// returns the lines to normal (before a prefill walk, whose tile kernels
// want the whole L2).
size_t l2_set_persisting_limit(size_t bytes);
// [ptr, ptr + bytes) back to the normal eviction priority
// (applypriority.global.L2::evict_normal per line: 6 us over 14 MB, no
// reads) — the release of persisting lines, on a stream. NOT
// cudaCtxResetPersistingL2Cache: that one needs the device idle and
// deadlocks against a peer's spinning collective kernel (the world-2
// loopback engine tests, 2026-09-22).
void launch_l2_release(const void* ptr, size_t bytes, cudaStream_t stream);

// The prefetch scheduler. A WINDOW is opened at a point of the main stream
// (everything added to it starts once the main stream reaches that point)
// and carries a byte budget: a window's prefetch must fit in L2 beside the
// chain's own traffic, and must not outlast the latency it hides by much
// (past that it merely competes with the consumer for bandwidth). Layers
// take a pointer to this class and call prefetch_after() at the one point
// in their enqueue where the next weight is known and the chain is about
// to go latency-bound.
//
// The knobs are the engine's keys (engine.l2_prefetch*), applied once per
// process by l2_prefetch_configure before any model is built — every rank
// the same (2026-10-05; they were environment variables before): `enabled`
// (every call a no-op when off), the window budget, the boundary windows'
// budget (the models' windows beside a collective), the rate of the windows
// that overlap a collective (boundary_rate(); light), the rate of the
// windows inside the attention layers (layer_rate(); light — the Full rate
// slowed the small kernels it ran beside by as much as it saved on the
// projection) and whether adjacent ranges merge into one launch.
enum class L2PrefetchForm { Load, Lines, Touch };  // engine.l2_prefetch_form: load | lines | touch
// The full-rate window's shape (engine.l2_prefetch_full_form): 16 blocks x 8
// loads (512 KB in flight, the DRAM rate at the light form's SM footprint;
// DeepSeek-V4.1's gap kernels run unstretched beside it) or 48 x 4 (768 KB
// in flight on every SM; GLM-5.3's windows keep their share beside its
// streaming kernels — 2026-10-20, +0.3-0.8 % a pass at the narrow form).
enum class L2FullForm { Narrow16x8, Wide48x4 };
struct L2PrefetchSettings {
  bool enabled = true;
  bool merge = true;
  L2PrefetchForm form = L2PrefetchForm::Load;
  L2FullForm full_form = L2FullForm::Narrow16x8;
  size_t window_bytes = size_t{12} << 20;           // engine.l2_prefetch_window_mib
  size_t boundary_window_bytes = size_t{20} << 20;  // engine.l2_prefetch_boundary_window_mib (0: the window budget)
  PrefetchRate boundary_rate = PrefetchRate::Light;
  PrefetchRate layer_rate = PrefetchRate::Light;
};
void l2_prefetch_configure(const L2PrefetchSettings& s);
const L2PrefetchSettings& l2_prefetch_settings();
// "off" | "light" | "full" (throws on anything else).
PrefetchRate l2_prefetch_rate(const std::string& name);
// "load" | "lines" | "touch" (throws on anything else).
L2PrefetchForm l2_prefetch_form(const std::string& name);
// "16x8" | "48x4" (throws on anything else).
L2FullForm l2_prefetch_full_form(const std::string& name);
const char* l2_prefetch_rate_name(PrefetchRate rate);
class WeightPrefetcher {
 public:
  static constexpr size_t kDefaultWindowBytes = size_t{12} << 20;

  WeightPrefetcher();
  ~WeightPrefetcher();
  WeightPrefetcher(const WeightPrefetcher&) = delete;
  WeightPrefetcher& operator=(const WeightPrefetcher&) = delete;

  bool enabled() const { return enabled_; }
  size_t default_window_bytes() const { return window_bytes_; }
  // The rate for windows that run beside a collective (see PrefetchRate).
  PrefetchRate boundary_rate() const { return boundary_rate_; }
  // The rate for windows inside a layer (beside the chain's small kernels).
  PrefetchRate layer_rate() const { return layer_rate_; }

  // Opens a window at `main`'s current position with the given budget
  // (0 = the default) and rate. Adds before the next open_window() share
  // it. Rate Off opens a window that ignores every add().
  // `persisting`: every launch of the window carries the persisting
  // access policy (launch_l2_prefetch).
  void open_window(cudaStream_t main, size_t budget_bytes = 0,
                   PrefetchRate rate = PrefetchRate::Full, bool persisting = false);
  // Adds a range to the open window, clamped to its remaining budget.
  // Adjacent (or nearly adjacent) ranges coalesce into one launch
  //: a window's adds are a layer's tensors in consumption
  // order and they sit contiguously in the resident image, so what was
  // ~620 one-tensor kernels per decode step — 40 % of the graph's nodes,
  // and cudaGraphLaunch costs ~0.45 us per node on this host — becomes a
  // few dozen. The pending range launches at the next non-adjacent add,
  // at the next open_window(), or at join(); engine.l2_prefetch_merge false
  // restores one launch per add. A hole is bridged only inside the pending
  // range's own allocation — the driver's address range of its first byte
  // (2026-10-19: the DeepSeek-V4 draft head's fp8 copy, a separate
  // allocation that landed 1.9 MiB past its layer's image, was bridged,
  // and the load-form kernel read the unmapped hole; the layout a kernel
  // change shifts decides whether two allocations fall within kMergeGap).
  void add(const void* ptr, size_t bytes);
  // add() for a range that is its OWN allocation (a bf16 weight's packed
  // companion — IGemm::resident_view — not a span of the layer's image): it
  // never joins a pending range and nothing joins it. The coalescing bridges
  // holes of up to kMergeGap between adds, which is a read of whatever lies
  // between them — a layer's neighbouring tensors inside one image, but
  // between two allocations a hole can be unmapped (a released bf16 range,
  // loaders/releasable_range.hpp, stays reserved and unreadable; a
  // neighbouring image's edge is an out-of-bounds read, compute-sanitizer
  // 2026-09-09). One launch of its own.
  void add_isolated(const void* ptr, size_t bytes);
  // The matmul weight `weight` through its resident view: add() when the
  // view is the weight itself (its image's span), add_isolated() when it is
  // a companion.
  void add_view(const void* weight, const void* view, size_t view_bytes) {
    if (view == weight) add(view, view_bytes);
    else add_isolated(view, view_bytes);
  }
  // open_window + add in one call — the layers' idiom.
  void prefetch_after(cudaStream_t main, const void* ptr, size_t bytes,
                      size_t budget_bytes = 0,
                      PrefetchRate rate = PrefetchRate::Full);
  // `main` waits for every prefetch issued since the last join. Required
  // before a capture ends (a forked stream must rejoin the origin); a
  // no-op when nothing was forked.
  void join(cudaStream_t main);
  // Launches (and cumulative bytes) issued so far — the merge's evidence.
  size_t launches() const { return launches_; }
  // The persisting lines the recent persisting windows filled, released
  // to the normal priority on `main` (launch_l2_release over the last
  // kPersistRing persisting launches' ranges — the set-aside holds no
  // more than those): before a walk that wants the whole L2.
  void release_persisting(cudaStream_t main);

 private:
  // The coalescing range (see add): [pending_begin_, pending_end_) not
  // yet launched; merge_ is the knob; kMergeGap the largest hole bridged.
  static constexpr size_t kMergeGap = size_t{2} << 20;  // bridged gaps are charged to the budget
  void flush_pending();
  bool merge_ = true;
  uintptr_t pending_begin_ = 0, pending_end_ = 0;
  uintptr_t pending_alloc_end_ = 0;  // the end of the pending range's allocation (0: unknown — no hole bridged)
  size_t launches_ = 0;
  bool enabled_ = true;
  size_t window_bytes_ = kDefaultWindowBytes;
  PrefetchRate boundary_rate_ = PrefetchRate::Light;
  PrefetchRate layer_rate_ = PrefetchRate::Light;
  PrefetchRate rate_ = PrefetchRate::Full;  // the open window's
  bool persisting_ = false;                 // the open window's access policy
  static constexpr int kPersistRing = 4;
  struct Range {
    uintptr_t begin = 0;
    size_t bytes = 0;
  };
  Range persisted_[kPersistRing];
  int persisted_next_ = 0;
  void note_persisting(const void* ptr, size_t bytes);
  size_t remaining_ = 0;
  bool window_open_ = false;
  bool forked_ = false;  // a fork since the last join (join must follow)
  cudaStream_t side_ = nullptr;
  cudaEvent_t fork_ = nullptr;
  cudaEvent_t join_ = nullptr;
};

}  // namespace dgpp
