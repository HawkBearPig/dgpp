#include "kernels/l2_prefetch.hpp"
#include "kernels/pdl.cuh"

#include <cuda.h>

#include <format>
#include <stdexcept>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {
namespace {

constexpr int kThreads = 256;
// Full: 16 blocks x 8 loads x 16 B x 256 threads = 512 KB in flight — the
// DRAM ceiling (232 GB/s on a cold 18 MiB window) at the Light form's SM
// footprint. Its first form, 48 x 4 (768 KB), reached the same rate but
// sat on every SM: a gap-shaped latency chain (small L2-bound kernels and
// a spin) ran x1.21 beside it, x1.03 beside 16 x 8 — and on the fabric the
// boundary windows at that form lengthened the fold and update segments of
// a DeepSeek-V4.1 decode layer by 71 us against the 28 us they saved on the
// projections (2026-10-20). The stretch follows the footprint and the bytes
// in flight (24 x 8 and 32 x 8 at 768 KB-1 MB: x1.18-1.24), not the rate.
// Light: 16 x 2 = 128 KB in flight — 187 GB/s cold, x1.01 on the chain.
// (96 x 8 = 3 MB reached 215 GB/s and made every memory access on the box
// wait ~12 us: the bus handshake went 9 -> 26 us.)
// Both forms stay built: engine.l2_prefetch_full_form picks one per template
// (the narrow form is the default; GLM-5.3 ships the wide one).
constexpr int kNarrowBlocks = 16;
constexpr int kNarrowUnroll = 8;
constexpr int kWideBlocks = 48;
constexpr int kWideUnroll = 4;
constexpr int kLightUnroll = 2;
constexpr int kLightBlocksDefault = 16;
// DGPP_L2_PREFETCH_LIGHT_BLOCKS: the Light rate's grid (tuning knob).
int light_blocks() { return kLightBlocksDefault; }

// A device word the kernel compares its fold against. Nothing sets it and
// the fold is arbitrary, so the compare (almost) never matches — and when
// it does, the store is a harmless write to a sink. The compiler cannot
// prove either, so the loads survive optimization.
__device__ uint32_t g_prefetch_sink[2];

template <int kUnroll>
__global__ __launch_bounds__(kThreads) void l2_prefetch_kernel(
    const uint4* __restrict__ p, size_t vec_count) {
  dgpp_pdl_wait();
  const size_t stride = static_cast<size_t>(gridDim.x) * kThreads;
  size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x;
  uint32_t fold = 0;
  for (; i + (kUnroll - 1) * stride < vec_count; i += kUnroll * stride) {
    uint4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) v[u] = __ldcg(p + i + u * stride);
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) fold ^= v[u].x ^ v[u].w;
  }
  for (; i < vec_count; i += stride) {
    const uint4 v = __ldcg(p + i);
    fold ^= v.x ^ v.w;
  }
  if (fold == g_prefetch_sink[0] + 0x9E3779B9u) g_prefetch_sink[1] = fold;
}

// The prefetch-instruction form (2026-09-10, DGPP_L2_PREFETCH_FORM=prefetch):
// one `prefetch.global.L2` per 128-byte line per thread, grid-strided —
// no destination register, no scoreboard wait, so a warp keeps issuing
// lines instead of folding what the loads above brought back. The bytes
// in flight are then the memory system's to bound, not the fold's; the
// sweep says whether that lifts the sustainable rate beside the
// collectives or just their latency (the load form's light rate was
// tuned to exactly that trade).
__global__ __launch_bounds__(kThreads) void l2_prefetch_lines_kernel(
    const uint8_t* __restrict__ p, size_t lines) {
  dgpp_pdl_wait();
  const size_t stride = static_cast<size_t>(gridDim.x) * kThreads;
  for (size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x; i < lines; i += stride)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p + i * 128));
}

L2PrefetchSettings g_l2_settings;  // l2_prefetch_configure, before any model is built
bool full_form_wide() { return g_l2_settings.full_form == L2FullForm::Wide48x4; }
int full_blocks() { return full_form_wide() ? kWideBlocks : kNarrowBlocks; }
bool env_prefetch_form_lines() { return g_l2_settings.form == L2PrefetchForm::Lines; }

// The touch form (2026-09-29, the Qwen depth-3 profiles with the prefetch on
// against off): of the ~3.9 GB a pass prefetched, the consumers found
// 96 MB in L2 — yet they ran ~10 % slower without it. What the prefetch
// delivered was the TRANSLATIONS: a 65 GB working set walks 32K 2 MB
// pages per pass through a walker every SM shares, and the consumer that
// finds its pages already walked runs at the isolated rate. This form
// touches one 128-byte line per 64 KB (1/512 of the bytes) so the walks
// happen ahead of the consumer while the DRAM stays free for it. The form
// is the engine's key (engine.l2_prefetch_form touch); the shipped form
// loads the bytes.
bool env_prefetch_form_touch() { return g_l2_settings.form == L2PrefetchForm::Touch; }
size_t env_touch_stride_bytes() { return size_t{64} << 10; }
__global__ __launch_bounds__(kThreads) void l2_touch_pages_kernel(const uint8_t* __restrict__ p, size_t touches,
                                                                 size_t stride_bytes) {
  dgpp_pdl_wait();
  const size_t stride = static_cast<size_t>(gridDim.x) * kThreads;
  for (size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x; i < touches; i += stride)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p + i * stride_bytes));
}


const char* rate_name(PrefetchRate r) {
  switch (r) {
    case PrefetchRate::Off: return "off";
    case PrefetchRate::Light: return "light";
    case PrefetchRate::Full: return "full";
  }
  return "?";
}

}  // namespace

void l2_prefetch_configure(const L2PrefetchSettings& s) {
  if (s.window_bytes == 0 || s.window_bytes > (size_t{64} << 20))
    throw std::invalid_argument("l2_prefetch_configure: the window budget must be 1..64 MiB");
  if (s.boundary_window_bytes > (size_t{64} << 20))
    throw std::invalid_argument("l2_prefetch_configure: the boundary window budget must be 0..64 MiB");
  g_l2_settings = s;
}
const L2PrefetchSettings& l2_prefetch_settings() { return g_l2_settings; }
PrefetchRate l2_prefetch_rate(const std::string& name) {
  if (name == "off") return PrefetchRate::Off;
  if (name == "light") return PrefetchRate::Light;
  if (name == "full") return PrefetchRate::Full;
  throw std::invalid_argument("l2 prefetch rate: off, light or full (got '" + name + "')");
}
const char* l2_prefetch_rate_name(PrefetchRate rate) { return rate_name(rate); }
L2PrefetchForm l2_prefetch_form(const std::string& name) {
  if (name == "load") return L2PrefetchForm::Load;
  if (name == "lines") return L2PrefetchForm::Lines;
  if (name == "touch") return L2PrefetchForm::Touch;
  throw std::invalid_argument("l2 prefetch form: load, lines or touch (got '" + name + "')");
}
L2FullForm l2_prefetch_full_form(const std::string& name) {
  if (name == "16x8") return L2FullForm::Narrow16x8;
  if (name == "48x4") return L2FullForm::Wide48x4;
  throw std::invalid_argument("l2 prefetch full form: 16x8 or 48x4 (got '" + name + "')");
}

namespace {

}  // namespace

// The kernel launch with the persisting access-policy window over
// [begin, end) (cudaLaunchKernelEx: the attribute is captured into the
// graph node).
template <int kUnroll>
void launch_persisting(const uint4* p, size_t vecs, int blocks, cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(blocks));
  cfg.blockDim = dim3(kThreads);
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeAccessPolicyWindow;
  attr[0].val.accessPolicyWindow.base_ptr = const_cast<uint4*>(p);
  attr[0].val.accessPolicyWindow.num_bytes = vecs * 16;
  attr[0].val.accessPolicyWindow.hitRatio = 1.0f;
  attr[0].val.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr[0].val.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  DGPP_CUDA_OK(cudaLaunchKernelEx(&cfg, l2_prefetch_kernel<kUnroll>, p, vecs));
}

// The device allocation holding `p`: [base, end) from the driver. False for
// host, managed or unknown memory (nothing is clamped or bridged then).
static bool device_allocation(uintptr_t p, uintptr_t* base, uintptr_t* end) {
  cudaPointerAttributes attrs{};
  if (cudaPointerGetAttributes(&attrs, reinterpret_cast<const void*>(p)) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (attrs.type != cudaMemoryTypeDevice) return false;
  CUdeviceptr b = 0;
  size_t size = 0;
  if (cuMemGetAddressRange(&b, &size, static_cast<CUdeviceptr>(p)) != CUDA_SUCCESS) return false;
  *base = static_cast<uintptr_t>(b);
  *end = static_cast<uintptr_t>(b) + size;
  return true;
}

void launch_l2_prefetch(const void* ptr, size_t bytes, PrefetchRate rate,
                        cudaStream_t stream, bool persisting) {
  if (ptr == nullptr || bytes == 0 || rate == PrefetchRate::Off) return;
  // Widen to 16-byte boundaries: the read may cover a few bytes past either
  // end of the request, which is safe for any cudaMalloc'd range (256-byte
  // aligned and sized).
  const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr) & ~uintptr_t{15};
  const uintptr_t end =
      (reinterpret_cast<uintptr_t>(ptr) + bytes + 15) & ~uintptr_t{15};
  // The guard (2026-10-19): a range never reads past the allocation that
  // holds its first byte — the merge below keeps ranges inside one
  // allocation, and a caller's over-long range is clamped here with one
  // warning per process (the forms read every line they are given).
  uintptr_t end_clamped = end;
  {
    uintptr_t base = 0, alloc_end = 0;
    if (device_allocation(begin, &base, &alloc_end) && end > alloc_end) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        DGPP_LOG_WARN("l2 prefetch: a range at {:#x} ({} bytes) runs {} bytes past its allocation [{:#x}, {:#x}); clamped",
                      reinterpret_cast<uintptr_t>(ptr), bytes, end - alloc_end, base, alloc_end);
      }
      end_clamped = alloc_end & ~uintptr_t{15};
      if (end_clamped <= begin) return;
    }
  }
  const uint4* p = reinterpret_cast<const uint4*>(begin);
  const size_t vecs = (end_clamped - begin) / 16;
  if (persisting) {
    if (rate == PrefetchRate::Full && full_form_wide())
      launch_persisting<kWideUnroll>(p, vecs, kWideBlocks, stream);
    else if (rate == PrefetchRate::Full)
      launch_persisting<kNarrowUnroll>(p, vecs, kNarrowBlocks, stream);
    else
      launch_persisting<kLightUnroll>(p, vecs, light_blocks(), stream);
    return;
  }
  if (rate == PrefetchRate::Full) {
    if (full_form_wide())
      l2_prefetch_kernel<kWideUnroll><<<kWideBlocks, kThreads, 0, stream>>>(p, vecs);
    else
      l2_prefetch_kernel<kNarrowUnroll><<<kNarrowBlocks, kThreads, 0, stream>>>(p, vecs);
  } else if (env_prefetch_form_touch()) {
    const size_t sb = env_touch_stride_bytes();
    const size_t touches = (end_clamped - begin + sb - 1) / sb;
    l2_touch_pages_kernel<<<static_cast<unsigned>(std::min<size_t>(light_blocks(), (touches + kThreads - 1) / kThreads)),
                            kThreads, 0, stream>>>(reinterpret_cast<const uint8_t*>(begin), touches, sb);
  } else if (env_prefetch_form_lines()) {
    const size_t lines = (end_clamped - begin + 127) / 128;
    l2_prefetch_lines_kernel<<<light_blocks(), kThreads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(begin), lines);
  } else {
    l2_prefetch_kernel<kLightUnroll><<<light_blocks(), kThreads, 0, stream>>>(p, vecs);
  }
  DGPP_CUDA_OK(cudaGetLastError());
}

__global__ __launch_bounds__(kThreads) void l2_release_kernel(const uint8_t* p, size_t lines) {
  dgpp_pdl_wait();
  const size_t stride = static_cast<size_t>(gridDim.x) * kThreads;
  for (size_t i = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x; i < lines; i += stride)
    asm volatile("applypriority.global.L2::evict_normal [%0], 128;" ::"l"(p + i * 128) : "memory");
}

void launch_l2_release(const void* ptr, size_t bytes, cudaStream_t stream) {
  if (ptr == nullptr || bytes == 0) return;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr) & ~uintptr_t{127};
  const uintptr_t end = (reinterpret_cast<uintptr_t>(ptr) + bytes + 127) & ~uintptr_t{127};
  const size_t lines = (end - begin) / 128;
  l2_release_kernel<<<full_blocks(), kThreads, 0, stream>>>(reinterpret_cast<const uint8_t*>(begin), lines);
  DGPP_CUDA_OK(cudaGetLastError());
}

size_t l2_set_persisting_limit(size_t bytes) {
  int dev = 0;
  DGPP_CUDA_OK(cudaGetDevice(&dev));
  int max_bytes = 0;
  DGPP_CUDA_OK(cudaDeviceGetAttribute(&max_bytes, cudaDevAttrMaxPersistingL2CacheSize, dev));
  if (max_bytes <= 0) return 0;
  const size_t want = std::min(bytes, static_cast<size_t>(max_bytes));
  DGPP_CUDA_OK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want));
  size_t got = 0;
  DGPP_CUDA_OK(cudaDeviceGetLimit(&got, cudaLimitPersistingL2CacheSize));
  return got;
}

void WeightPrefetcher::note_persisting(const void* ptr, size_t bytes) {
  persisted_[persisted_next_] = Range{reinterpret_cast<uintptr_t>(ptr), bytes};
  persisted_next_ = (persisted_next_ + 1) % kPersistRing;
}

void WeightPrefetcher::release_persisting(cudaStream_t main) {
  for (Range& r : persisted_) {
    if (r.bytes == 0) continue;
    launch_l2_release(reinterpret_cast<const void*>(r.begin), r.bytes, main);
    r = Range{};
  }
}

WeightPrefetcher::WeightPrefetcher() {
  const L2PrefetchSettings& s = g_l2_settings;  // engine.l2_prefetch*, l2_prefetch_configure
  enabled_ = s.enabled;
  merge_ = s.merge;
  window_bytes_ = s.window_bytes;
  boundary_rate_ = s.boundary_rate;
  layer_rate_ = s.layer_rate;
  if (!enabled_) {
    DGPP_LOG_INFO("l2 prefetch: off (engine.l2_prefetch false)");
    return;
  }
  // Lowest priority: the chain's kernels (on a higher-priority stream)
  // take SM slots first; the prefetch fills what is left, which during a
  // latency-bound phase is nearly everything.
  int least = 0, greatest = 0;
  DGPP_CUDA_OK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  DGPP_CUDA_OK(cudaStreamCreateWithPriority(&side_, cudaStreamNonBlocking,
                                            least));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&join_, cudaEventDisableTiming));
  DGPP_LOG_INFO("l2 prefetch: on, window budget {} MiB, boundary rate {}, "
                "layer rate {}",
                window_bytes_ >> 20, rate_name(boundary_rate_),
                rate_name(layer_rate_));
}

WeightPrefetcher::~WeightPrefetcher() {
  if (join_) cudaEventDestroy(join_);
  if (fork_) cudaEventDestroy(fork_);
  if (side_) cudaStreamDestroy(side_);
}

void WeightPrefetcher::flush_pending() {
  if (pending_end_ > pending_begin_) {
    launch_l2_prefetch(reinterpret_cast<const void*>(pending_begin_),
                       pending_end_ - pending_begin_, rate_, side_, persisting_);
    if (persisting_) note_persisting(reinterpret_cast<const void*>(pending_begin_), pending_end_ - pending_begin_);
    ++launches_;
  }
  pending_begin_ = pending_end_ = 0;
  pending_alloc_end_ = 0;
}

void WeightPrefetcher::open_window(cudaStream_t main, size_t budget_bytes,
                                   PrefetchRate rate, bool persisting) {
  if (!enabled_) return;
  flush_pending();  // the previous window's tail, at its own rate and policy
  rate_ = rate;
  persisting_ = persisting;
  remaining_ = budget_bytes > 0 ? budget_bytes : window_bytes_;
  window_open_ = true;
  if (rate == PrefetchRate::Off) return;  // no fork: nothing will launch
  DGPP_CUDA_OK(cudaEventRecord(fork_, main));
  DGPP_CUDA_OK(cudaStreamWaitEvent(side_, fork_, 0));
  forked_ = true;
}

void WeightPrefetcher::add(const void* ptr, size_t bytes) {
  if (!enabled_ || !window_open_ || rate_ == PrefetchRate::Off ||
      ptr == nullptr || bytes == 0)
    return;
  if (!merge_) {
    const size_t take = std::min(bytes, remaining_);
    if (take == 0) return;
    remaining_ -= take;
    launch_l2_prefetch(ptr, take, rate_, side_, persisting_);
    if (persisting_) note_persisting(ptr, take);
    ++launches_;
    return;
  }
  const uintptr_t b = reinterpret_cast<uintptr_t>(ptr);
  if (pending_end_ > pending_begin_ && b >= pending_begin_ &&
      b <= pending_end_ + kMergeGap) {
    // Extend the pending range; the bridged gap (a neighbouring tensor of
    // the same layer, read soon anyway) is charged to the budget too — and
    // bridged only inside the pending range's allocation (an abutting
    // range joins whatever its allocation): the hole between two
    // allocations is unmapped.
    const size_t gap = b > pending_end_ ? b - pending_end_ : 0;
    const size_t take = std::min(bytes, remaining_ > gap ? remaining_ - gap : 0);
    if (take == 0) return;
    const bool inside = gap == 0 || (pending_alloc_end_ != 0 && b + take <= pending_alloc_end_);
    if (inside) {
      remaining_ -= gap + take;
      pending_end_ = std::max(pending_end_, b + take);
      return;
    }
  }
  const size_t take = std::min(bytes, remaining_);
  if (take == 0) return;
  remaining_ -= take;
  flush_pending();
  pending_begin_ = b;
  pending_end_ = b + take;
  uintptr_t base = 0, alloc_end = 0;
  pending_alloc_end_ = device_allocation(b, &base, &alloc_end) ? alloc_end : 0;
}

void WeightPrefetcher::add_isolated(const void* ptr, size_t bytes) {
  if (!enabled_ || !window_open_ || rate_ == PrefetchRate::Off || ptr == nullptr || bytes == 0)
    return;
  const size_t take = std::min(bytes, remaining_);
  if (take == 0) return;
  remaining_ -= take;
  flush_pending();  // nothing pending may grow into this range, nor this range into the next add
  launch_l2_prefetch(ptr, take, rate_, side_, persisting_);
  if (persisting_) note_persisting(ptr, take);
  ++launches_;
}

void WeightPrefetcher::prefetch_after(cudaStream_t main, const void* ptr,
                                      size_t bytes, size_t budget_bytes,
                                      PrefetchRate rate) {
  open_window(main, budget_bytes, rate);
  add(ptr, bytes);
}

void WeightPrefetcher::join(cudaStream_t main) {
  // Only a forked side stream may be joined: under capture, waiting on an
  // event the capture never recorded is a capture-isolation error.
  if (!enabled_ || !forked_) return;
  flush_pending();
  DGPP_CUDA_OK(cudaEventRecord(join_, side_));
  DGPP_CUDA_OK(cudaStreamWaitEvent(main, join_, 0));
  window_open_ = false;
  forked_ = false;
}

}  // namespace dgpp
