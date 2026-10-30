#pragma once
// Shared allocation, slicing and byte accounting for resident weights.
// Each model family supplies its tensor table, replicated-tensor
// classification and builders. Counting and copying use the same allocation
// sequence so the memory plan matches the constructed layout.
//
// Weight bump: DEVICE memory (cudaMalloc), built through a pinned HOST
// staging mirror of the same layout. On the GB10 the GPU streams
// cudaMallocManaged memory at ~160 GB/s whatever the advice/prefetch,
// pinned host memory at ~228 falling to ~180 once tens of GB are pinned
// (4 KB translations), and cudaMalloc at ~248 cold across 70 GiB
// (micro_mem_bw -m/-a/-f/-p, micro_gemv_bw -c) — the weights are the
// decode step's bytes. The build code writes with host memcpys into the
// mirror (host()), one H2D copy per layer lands the bytes, and the
// pointers handed out (alloc) are the DEVICE addresses the kernels consume.
// Counting mode walks the same grant sequence without touching memory —
// a family's byte formula and its allocator are the same build code, so
// the two cannot drift.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf12_gemv.hpp"
#include "loaders/fp8_quant.hpp"
#include "loaders/releasable_range.hpp"
#include "loaders/safetensors.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

constexpr size_t kWeightAllocAlign = 256;

inline size_t align_up_256(size_t b) {
  return (b + kWeightAllocAlign - 1) / kWeightAllocAlign * kWeightAllocAlign;
}

// FNV-1a over bytes, seeded (the replicated-weight digests fold with it).
inline uint64_t fnv1a(const void* data, size_t n, uint64_t h) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}
inline uint64_t fnv_mix(uint64_t h, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    h = (h ^ (v & 0xFF)) * 1099511628211ull;
    v >>= 8;
  }
  return h;
}

// A block-scale dequant of a compressed resident matrix into a bf16
// buffer alongside it (device -> device, run after the layer's H2D).
struct DequantJob {
  const uint8_t* payload;
  const float* scales;
  uint16_t* out;
  int64_t rows;
  int64_t cols;
};

// A strided 2-D pack: `rows` rows of `width` bytes from a source of pitch
// `src_pitch` into a destination of pitch `dst_pitch`. Host sources land
// in the staging mirror in phase one; device sources (a bridge buffer a
// dequant wrote) copy on the stream in phase three.
struct PackJob {
  const uint16_t* src;
  uint16_t* dst;  // device address inside the bump
  size_t src_pitch;
  size_t dst_pitch;
  size_t width;
  size_t rows;
  bool src_on_device;
};

// SIDE GRANTS (2026-09-19, bf12-only residency — common/bf16_residency.hpp).
// A bf16 matrix that a lossless 12-bit companion will replace is granted
// ASIDE: its own releasable range (loaders/releasable_range.hpp) instead of
// a span of the layer's allocation, so that its bytes can be returned once
// the companion exists (release_side) while its address stays a unique key.
// The STAGING layout does not change: `cursor` still walks every grant in
// order, so the mirror, the byte formula and the resident image are the
// same bytes whatever the mode (an image written in one mode restores in
// another). Only the device placement differs — `runs` maps the main
// allocation's spans to their staging offsets, `sides` the ranges'.
struct LayerBump {
  void* base = nullptr;   // device memory: the addresses alloc() hands out
  void* stage = nullptr;  // pinned host mirror for the build (grant order)
  size_t capacity = 0;    // device bytes behind base
  size_t cursor = 0;      // the layer's bytes in grant order (mirror, image, formula)
  bool counting = false;
  bool side_mode = false;  // alloc_side grants go aside (set before the first grant)
  size_t dev_cursor = 0;   // bytes granted from base (== cursor while nothing went aside)
  size_t side_bytes = 0;   // bytes granted aside
  // PLACEMENT (2026-10-30, the step engine's arena — MOONSHOT-1.md 2.4):
  // when set, main grant i lands at base + placement[i] instead of the
  // next free byte, and the staging mirror takes the same layout, so
  // host(), upload() and download() are unchanged and the image written
  // from such a bump holds the placed layout (its key carries the layout's
  // version). The grants are a permutation of the grant-order layout, so
  // `cursor`, the byte formula and the counting pass do not change. Set
  // before the first grant; not combined with side grants.
  std::vector<size_t> placement;
  size_t grant_index = 0;  // main grants handed out so far (indexes placement)

  struct SideGrant {
    size_t stage_off = 0;
    size_t bytes = 0;  // the 256-aligned grant
    ReleasableRange range;
  };
  struct Run {  // a maximal span of main grants
    size_t stage_off = 0, dev_off = 0, bytes = 0;
  };
  std::vector<SideGrant> sides;
  std::vector<Run> runs;

  ~LayerBump() {
    if (base) cudaFree(base);
  }
  LayerBump() = default;
  LayerBump(const LayerBump&) = delete;
  LayerBump& operator=(const LayerBump&) = delete;

  // `cap`: the bytes alloc() will grant (the layer's bytes less its side
  // grants when side_mode is on).
  void init(size_t cap) {
    if (cap > 0) DGPP_CUDA_OK(cudaMalloc(&base, cap));
    capacity = cap;
    cursor = dev_cursor = side_bytes = 0;
  }

  // The host-side address of a device grant: where the build writes.
  template <typename T>
  T* host(T* dev) const {
    if (!stage) throw std::logic_error("weight bump has no staging");
    const char* d = reinterpret_cast<const char*>(dev);
    if (!side_mode)
      return reinterpret_cast<T*>(static_cast<char*>(stage) + (d - static_cast<const char*>(base)));
    for (const SideGrant& g : sides) {
      const char* b = static_cast<const char*>(g.range.data());
      if (d >= b && d < b + g.bytes)
        return reinterpret_cast<T*>(static_cast<char*>(stage) + g.stage_off + (d - b));
    }
    const size_t off = static_cast<size_t>(d - static_cast<const char*>(base));
    for (const Run& r : runs)
      if (off >= r.dev_off && off < r.dev_off + r.bytes)
        return reinterpret_cast<T*>(static_cast<char*>(stage) + r.stage_off + (off - r.dev_off));
    throw std::logic_error("weight bump: host() of an address it did not grant");
  }

  // Every grant is 256-aligned, so a layer's total is exactly the sum of
  // per-tensor aligned sizes — no inter-allocation padding surprises.
  void* alloc(size_t bytes) {
    const size_t grant = align_up_256(bytes);
    if (dev_cursor + grant > capacity)
      throw std::runtime_error("weight bump OOM");
    size_t at = dev_cursor;
    if (!placement.empty()) {
      if (side_mode) throw std::logic_error("weight bump: a placed layout does not take side grants");
      if (grant_index >= placement.size())
        throw std::logic_error("weight bump: more grants than the placement names");
      at = placement[grant_index++];
      if (at % kWeightAllocAlign != 0 || at + grant > capacity)
        throw std::logic_error("weight bump: a placement outside the layer or unaligned");
    }
    void* p = counting ? nullptr : static_cast<char*>(base) + at;
    if (side_mode && !counting) {
      if (!runs.empty() && runs.back().stage_off + runs.back().bytes == cursor)
        runs.back().bytes += grant;
      else
        runs.push_back(Run{cursor, dev_cursor, grant});
    }
    cursor += grant;
    dev_cursor += grant;
    return p;
  }

  // A grant a packed companion may replace: aside when side_mode is on
  // (the caller checks the matrix is packable), alloc() otherwise.
  void* alloc_side(size_t bytes) {
    if (!side_mode) return alloc(bytes);
    const size_t grant = align_up_256(bytes);
    void* p = nullptr;
    if (!counting) {
      sides.push_back(SideGrant{cursor, grant, ReleasableRange(grant)});
      p = sides.back().range.data();
    }
    cursor += grant;
    side_bytes += grant;
    return p;
  }

  // The staging mirror's bytes onto the device (the layer's one H2D).
  void upload(cudaStream_t stream) const {
    if (!side_mode) {
      DGPP_CUDA_OK(cudaMemcpyAsync(base, stage, cursor, cudaMemcpyHostToDevice, stream));
      return;
    }
    for (const Run& r : runs)
      DGPP_CUDA_OK(cudaMemcpyAsync(static_cast<char*>(base) + r.dev_off,
                                   static_cast<const char*>(stage) + r.stage_off, r.bytes,
                                   cudaMemcpyHostToDevice, stream));
    for (const SideGrant& g : sides)
      if (g.range.mapped())
        DGPP_CUDA_OK(cudaMemcpyAsync(g.range.data(), static_cast<const char*>(stage) + g.stage_off,
                                     g.bytes, cudaMemcpyHostToDevice, stream));
  }

  // The device bytes back into `dst` in grant order (the image capture).
  // Synchronous; every side grant must still be mapped.
  void download(void* dst) const {
    if (!side_mode) {
      DGPP_CUDA_OK(cudaMemcpy(dst, base, cursor, cudaMemcpyDeviceToHost));
      return;
    }
    for (const Run& r : runs)
      DGPP_CUDA_OK(cudaMemcpy(static_cast<char*>(dst) + r.stage_off,
                              static_cast<const char*>(base) + r.dev_off, r.bytes,
                              cudaMemcpyDeviceToHost));
    for (const SideGrant& g : sides) {
      if (!g.range.mapped()) throw std::logic_error("weight bump: download after a side release");
      DGPP_CUDA_OK(cudaMemcpy(static_cast<char*>(dst) + g.stage_off, g.range.data(), g.bytes,
                              cudaMemcpyDeviceToHost));
    }
  }

  // Returns the bytes of the side grant at `dev` to the device (its address
  // stays reserved); 0 when `dev` is not a mapped side grant. The caller
  // guarantees nothing outstanding reads it.
  size_t release_side(const void* dev) {
    for (SideGrant& g : sides)
      if (g.range.data() == dev && g.range.mapped()) {
        const size_t freed = g.range.mapped_bytes();
        g.range.release();
        return freed;
      }
    return 0;
  }

  void reset() {
    cursor = dev_cursor = side_bytes = 0;
    grant_index = 0;
    sides.clear();
    runs.clear();
  }
};

inline void run_host_pack(const PackJob& j, const LayerBump& bump) {
  uint16_t* dst = bump.host(j.dst);
  if (j.width == j.dst_pitch && j.width == j.src_pitch) {
    std::memcpy(dst, j.src, j.width * j.rows);  // degenerate: contiguous
    return;
  }
  for (size_t r = 0; r < j.rows; ++r)
    std::memcpy(dst + r * (j.dst_pitch / 2), j.src + r * (j.src_pitch / 2),
                j.width);
}

inline void run_device_pack(const PackJob& j, cudaStream_t stream) {
  DGPP_CUDA_OK(cudaMemcpy2DAsync(j.dst, j.dst_pitch, j.src, j.src_pitch,
                                 j.width, j.rows, cudaMemcpyDeviceToDevice,
                                 stream));
}

// Load-boundary synchronization: waits for every outstanding READER of the
// bump. With a registered reader stream that is exactly two streams (the
// model's compute stream + the loader's dequant stream, idle since the
// previous load's exit sync). Without one, the conservative default: the
// whole device — standalone callers load with unknown readers, and no
// in-process peer exists to spin against. (A device-wide wait in a
// one-PROCESS multi-rank world deadlocks by construction: a peer's spinning
// collective kernel never completes.)
inline void sync_load_boundary(cudaStream_t reader, cudaStream_t dequant) {
  if (reader) {
    DGPP_CUDA_OK(cudaStreamSynchronize(reader));
    DGPP_CUDA_OK(cudaStreamSynchronize(dequant));
  } else {
    DGPP_CUDA_OK(cudaDeviceSynchronize());
  }
}

// /proc/meminfo MemAvailable in bytes (0 when unreadable): what the kernel
// will hand out once it reclaims the page cache — which cudaMemGetInfo's
// "free" on the GB10's unified pool does not count.
inline size_t host_mem_available_bytes() {
  std::ifstream in("/proc/meminfo");
  std::string key;
  uint64_t kib = 0;
  std::string unit;
  while (in >> key >> kib >> unit)
    if (key == "MemAvailable:") return static_cast<size_t>(kib) * 1024;
  return 0;
}

// Everything one layer build needs, minus the family. `copy` false = the
// counting pass: identical allocation sequence, no memcpys, no kernel
// launches (source lookups stay inside if (copy) — the counting pass has
// no tensors). Names and sizes come from the family's expected table
// (config-derived); mmap pointers only feed copies.
//
// `Expected` is the family's expected-tensor record: `.name`, `.shape`,
// `.numel()`, `.nbytes()`. The family classifies tensors through the
// virtuals: replicated() (a rank-invariant read, allowed verbatim at
// world > 1), full_read() (counted in the rank-invariant re-read set of
// the §5.2 byte reconcile), fp4_global() (the NVFP4 triple's F32 scalar).
template <class Expected>
struct WeightBuilder {
  const std::vector<Expected>& table;
  const std::unordered_map<std::string, const Expected*>& by_name;
  LayerBump& bump;
  const std::unordered_map<std::string, const TensorInfo*>& tensors;
  std::vector<DequantJob>& jobs;
  std::vector<PackJob>& packs;
  bool copy;
  int rank = 0;
  int world = 1;
  // Resident builds read each source tensor exactly once: stream it in
  // ahead of the copy and drop it from the mapping + page cache after
  // (SafetensorsFile::prefetch/discard). Streaming builds re-read layers
  // every forward and want the cache kept.
  bool one_pass_sources = false;
  uint64_t source_bytes = 0;    // checkpoint bytes touched (copy mode)
  uint64_t verbatim_bytes = 0;  // the rank-invariant re-read subset
  std::string who = "loader";   // the family's name in error messages

  WeightBuilder(const std::vector<Expected>& table_,
                const std::unordered_map<std::string, const Expected*>& by_name_,
                LayerBump& bump_,
                const std::unordered_map<std::string, const TensorInfo*>& tensors_,
                std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_,
                bool copy_, int rank_, int world_, std::string who_)
      : table(table_), by_name(by_name_), bump(bump_), tensors(tensors_),
        jobs(jobs_), packs(packs_), copy(copy_), rank(rank_), world(world_),
        who(std::move(who_)) {}
  virtual ~WeightBuilder() = default;

  virtual bool replicated(const Expected& e) const = 0;
  // A tensor the build may read in full at world > 1: the replicated set,
  // plus whatever a family dequantizes whole and slices afterwards.
  virtual bool verbatim_ok(const Expected& e) const { return replicated(e); }
  virtual bool full_read(const Expected& e) const { return replicated(e); }
  virtual bool fp4_global(const Expected&) const { return false; }

  bool sharded() const { return world > 1; }

  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error(who + ": " + what);
  }

  const Expected& expected(const std::string& name) const {
    auto it = by_name.find(name);
    if (it == by_name.end())
      fail("'" + name + "' missing from expected table (builder bug)");
    return *it->second;
  }

  const TensorInfo& source(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      fail("tensor not in checkpoint: " + name);
    const TensorInfo& t = *it->second;
    if (one_pass_sources && t.owner) t.owner->prefetch(t);
    return t;
  }
  // The read is complete: every byte this rank wants from `t` sits in the
  // staging mirror. A sliced read still consumed the whole tensor's pages
  // (column slices touch every row), so the whole tensor goes.
  void consumed(const TensorInfo& t) const {
    if (one_pass_sources && t.owner) t.owner->discard(t);
  }

  // Byte accounting chokepoint: every checkpoint byte this build touches
  // flows through here, with its tensor. Counted in both modes: a counting
  // build therefore PLANS the rank's source bytes, which a loader's
  // reconcile compares with what the copy build read.
  void note_read(const Expected& e, size_t bytes) {
    source_bytes += bytes;
    if (full_read(e)) verbatim_bytes += bytes;
  }

  void check_range(const std::string& name, int64_t start, int64_t count,
                   int64_t dim) const {
    if (start < 0 || count < 0 || count > dim || start > dim - count)
      fail("slice of '" + name + "' out of bounds");
  }

  // A bf16 [n, k] matrix's grant. `packable`: the model packs this matrix
  // into a 12-bit companion (kernels/bf12_gemv.hpp) — inside the packing
  // contract it goes ASIDE when the bump runs side grants (LayerBump), so
  // its bf16 bytes can be returned once the companion exists.
  void* grant_bf16(int64_t n, int64_t k, bool packable) {
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(k) * 2;
    const bool aside = packable && n > 0 && n <= INT32_MAX && k > 0 && k <= INT32_MAX &&
                       bf12_shape_ok(static_cast<int>(n), static_cast<int>(k));
    return aside ? bump.alloc_side(bytes) : bump.alloc(bytes);
  }

  // ---- verbatim loads (replicated at every world; world=1: everything) --
  const void* load_raw(const std::string& name, bool packable = false) {
    const Expected& e = expected(name);
    if (sharded() && !verbatim_ok(e))
      fail("TP read-class drift — '" + name +
           "' is sharded but was loaded verbatim (the slicing build and "
           "the replicated classifier disagree; fix one of them)");
    void* dst = packable && e.shape.size() == 2 && e.nbytes() == static_cast<size_t>(e.numel()) * 2
                    ? grant_bf16(e.shape[0], e.shape[1], true)
                    : bump.alloc(e.nbytes());
    if (copy) {
      const TensorInfo& t = source(name);
      std::memcpy(bump.host(dst), t.data, e.nbytes());
      consumed(t);
    }
    note_read(e, e.nbytes());
    return dst;
  }
  uint16_t* load_bf16(const std::string& name, bool packable = false) {
    return static_cast<uint16_t*>(const_cast<void*>(load_raw(name, packable)));
  }
  float* load_f32(const std::string& name) {
    return static_cast<float*>(const_cast<void*>(load_raw(name)));
  }

  // A BF16 vector widened to F32 (kernels that want fp32 parameters:
  // GLM's APE, the GDN's A_log/dt_bias). Read via memcpy: safetensors does
  // not align individual tensors in the file.
  float* load_bf16_as_f32(const std::string& name, int64_t start = 0,
                          int64_t count = -1) {
    const Expected& e = expected(name);
    const int64_t total = static_cast<int64_t>(e.numel());
    if (count < 0) count = total - start;
    check_range(name, start, count, total);
    float* dst = static_cast<float*>(bump.alloc(static_cast<size_t>(count) * 4));
    if (copy) {
      const TensorInfo& t = source(name);
      const uint8_t* src = static_cast<const uint8_t*>(t.data) + static_cast<size_t>(start) * 2;
      float* h = bump.host(dst);
      for (int64_t i = 0; i < count; ++i) {
        uint16_t bits;
        std::memcpy(&bits, src + static_cast<size_t>(i) * 2, 2);
        h[i] = bf16_bits_to_float(bits);
      }
      consumed(t);
    }
    note_read(e, static_cast<size_t>(count) * 2);
    return dst;
  }

  // ---- contiguous row/element slices (bf16/f32 — no scale grid) --------
  uint16_t* load_bf16_rows(const std::string& name, int64_t row_start,
                          int64_t rows, bool packable = false) {
    const Expected& e = expected(name);
    check_range(name, row_start, rows, e.shape[0]);
    const int64_t width = static_cast<int64_t>(e.numel()) / e.shape[0];
    uint16_t* dst = static_cast<uint16_t*>(grant_bf16(rows, width, packable));
    if (copy) {
      const TensorInfo& t = source(name);
      const uint8_t* src = static_cast<const uint8_t*>(t.data);
      std::memcpy(bump.host(dst), src + static_cast<size_t>(row_start) * width * 2,
                  static_cast<size_t>(rows) * width * 2);
      consumed(t);
    }
    note_read(e, static_cast<size_t>(rows) * width * 2);
    return dst;
  }

  // A column slice of a BF16 [rows, full_cols] matrix, packed contiguous
  // [rows, cols] (a host-source pack in phase one).
  uint16_t* load_bf16_cols(const std::string& name, int64_t col_start,
                          int64_t cols, bool packable = false) {
    const Expected& e = expected(name);
    if (e.shape.size() != 2) fail("'" + name + "' is not a matrix");
    const int64_t rows = e.shape[0], full = e.shape[1];
    check_range(name, col_start, cols, full);
    uint16_t* dst = static_cast<uint16_t*>(grant_bf16(rows, cols, packable));
    if (copy) {
      const TensorInfo& t = source(name);
      packs.push_back(PackJob{static_cast<const uint16_t*>(t.data) + col_start, dst,
                              static_cast<size_t>(full) * 2, static_cast<size_t>(cols) * 2,
                              static_cast<size_t>(cols) * 2, static_cast<size_t>(rows),
                              /*src_on_device=*/false});
      // Not consumed here: the pack reads the mmap in phase one, after the
      // builder returns; the family's stream discards sources afterwards.
    }
    note_read(e, static_cast<size_t>(rows) * static_cast<size_t>(cols) * 2);
    return dst;
  }

  // ---- BF16 matrices encoded to block FP8 at load -----------
  // The checkpoint's BF16 rows / columns slice, encoded on the host into the
  // resident form the fp8 GEMV core and the scale-GEMM tile read
  // (loaders/fp8_quant.hpp). The scale grid is anchored at the slice's
  // origin: a slice that starts on a 128 multiple carries the whole
  // matrix's blocks.
  GlmQuantMatrix encode_fp8(const uint16_t* host_src, size_t src_stride, int64_t rows, int64_t cols) {
    const int64_t sr = fp8_quant::scale_rows(rows), sc = fp8_quant::scale_cols(cols);
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols)));
    q.scales = static_cast<const float*>(bump.alloc(static_cast<size_t>(sr) * static_cast<size_t>(sc) * 4));
    if (copy)
      fp8_quant::encode_block128(host_src, src_stride, rows, cols, bump.host(const_cast<uint8_t*>(q.payload)),
                                 bump.host(const_cast<float*>(q.scales)));
    return q;
  }
  GlmQuantMatrix load_bf16_rows_fp8(const std::string& name, int64_t row_start, int64_t rows) {
    const Expected& e = expected(name);
    check_range(name, row_start, rows, e.shape[0]);
    const int64_t width = static_cast<int64_t>(e.numel()) / e.shape[0];
    const uint16_t* src = nullptr;
    if (copy) {
      const TensorInfo& t = source(name);
      src = static_cast<const uint16_t*>(t.data) + static_cast<size_t>(row_start) * width;
    }
    GlmQuantMatrix q = encode_fp8(src, static_cast<size_t>(width), rows, width);
    if (copy) consumed(source(name));
    note_read(e, static_cast<size_t>(rows) * width * 2);
    return q;
  }
  GlmQuantMatrix load_bf16_cols_fp8(const std::string& name, int64_t col_start, int64_t cols) {
    const Expected& e = expected(name);
    if (e.shape.size() != 2) fail("'" + name + "' is not a matrix");
    const int64_t rows = e.shape[0], full = e.shape[1];
    check_range(name, col_start, cols, full);
    const uint16_t* src = nullptr;
    if (copy) src = static_cast<const uint16_t*>(source(name).data) + col_start;
    GlmQuantMatrix q = encode_fp8(src, static_cast<size_t>(full), rows, cols);
    if (copy) consumed(source(name));
    note_read(e, static_cast<size_t>(rows) * static_cast<size_t>(cols) * 2);
    return q;
  }
  GlmQuantMatrix load_bf16_fp8(const std::string& name) {
    const Expected& e = expected(name);
    if (e.shape.size() != 2) fail("'" + name + "' is not a matrix");
    return load_bf16_rows_fp8(name, 0, e.shape[0]);
  }

  float* load_f32_range(const std::string& name, int64_t start, int64_t count) {
    const Expected& e = expected(name);
    check_range(name, start, count, e.shape[0]);
    float* dst = static_cast<float*>(bump.alloc(static_cast<size_t>(count) * 4));
    if (copy) {
      const TensorInfo& t = source(name);
      const uint8_t* src = static_cast<const uint8_t*>(t.data);
      std::memcpy(bump.host(dst), src + static_cast<size_t>(start) * 4,
                  static_cast<size_t>(count) * 4);
      consumed(t);
    }
    note_read(e, static_cast<size_t>(count) * 4);
    return dst;
  }

  // ---- quantized slices (E4M3 + 128x128 block scales) -----------------
  // The scale partner `<name>_scale_inv` may be F32 (GLM) or BF16 (the
  // Qwen release); the resident grid is always F32. Row slice: payload
  // rows are contiguous; the resident scale grid re-blocks the sliced
  // axis at `scale_block_rows` (128: the §5.2 contract, a 128-aligned
  // start; a divisor of 128: every local block's scale is its parent
  // 128-block's, exact for any start aligned to it — D2).
  GlmQuantMatrix load_quant_rows(const std::string& name, int64_t row_start,
                                 int64_t rows, int scale_block_rows = 128) {
    const Expected& e = expected(name);
    if (scale_block_rows <= 0 || 128 % scale_block_rows != 0)
      fail("quantized row slice of '" + name + "': scale block must divide 128");
    if (row_start % scale_block_rows != 0)
      fail("quantized row slice of '" + name +
           "' must start aligned to its scale block (scale-grid slice contract)");
    check_range(name, row_start, rows, e.shape[0]);
    const Expected& es = expected(name + "_scale_inv");
    const int64_t cols = e.shape[1];
    const int64_t sb = (cols + 127) / 128;  // scale blocks per row
    const int64_t scale_rows = (rows + scale_block_rows - 1) / scale_block_rows;
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_block_rows = scale_block_rows;
    q.payload = static_cast<const uint8_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols)));
    q.scales = static_cast<const float*>(
        bump.alloc(static_cast<size_t>(scale_rows) * sb * 4));
    if (copy) {
      const TensorInfo& tp = source(name);
      const TensorInfo& ts = source(es.name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(tp.data) + static_cast<size_t>(row_start) * cols,
                  static_cast<size_t>(rows) * cols);
      float* hs = bump.host(const_cast<float*>(q.scales));
      for (int64_t r = 0; r < scale_rows; ++r) {
        const int64_t src_row = (row_start + r * scale_block_rows) / 128;
        copy_scale_row(ts, src_row * sb, sb, hs + r * sb);
      }
      consumed(tp);
      consumed(ts);
    }
    note_read(e, static_cast<size_t>(rows) * cols +
                     static_cast<size_t>(scale_rows) * sb * dtype_size(es.dtype));
    return q;
  }

  // Column slice -> PACKED payload + packed scale columns, re-blocked at
  // `scale_block_cols` on the sliced axis (the same rule as the rows).
  GlmQuantMatrix load_quant_cols(const std::string& name, int64_t col_start,
                                 int64_t cols, int scale_block_cols = 128) {
    const Expected& e = expected(name);
    if (scale_block_cols <= 0 || 128 % scale_block_cols != 0)
      fail("quantized column slice of '" + name + "': scale block must divide 128");
    if (col_start % scale_block_cols != 0)
      fail("quantized column slice of '" + name +
           "' must start aligned to its scale block (scale-grid slice contract)");
    check_range(name, col_start, cols, e.shape[1]);
    const Expected& es = expected(name + "_scale_inv");
    const int64_t rows = e.shape[0];
    const int64_t full_cols = e.shape[1];
    const int64_t sb_full = (full_cols + 127) / 128;
    const int64_t sb_s = (cols + scale_block_cols - 1) / scale_block_cols;
    const int64_t scale_rows = (rows + 127) / 128;
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_block_cols = scale_block_cols;
    q.payload = static_cast<const uint8_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols)));
    q.scales = static_cast<const float*>(
        bump.alloc(static_cast<size_t>(scale_rows) * sb_s * 4));
    if (copy) {
      const TensorInfo& tp = source(name);
      const TensorInfo& ts = source(es.name);
      const uint8_t* sp = static_cast<const uint8_t*>(tp.data);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      float* hs = bump.host(const_cast<float*>(q.scales));
      if (cols == full_cols && col_start == 0) {
        std::memcpy(hp, sp, static_cast<size_t>(rows) * cols);
      } else {
        for (int64_t r = 0; r < rows; ++r)
          std::memcpy(hp + r * cols, sp + r * full_cols + col_start, cols);
      }
      for (int64_t r = 0; r < scale_rows; ++r)
        for (int64_t c = 0; c < sb_s; ++c) {
          const int64_t src_col = (col_start + c * scale_block_cols) / 128;
          copy_scale_row(ts, r * sb_full + src_col, 1, hs + r * sb_s + c);
        }
      consumed(tp);
      consumed(ts);
    }
    note_read(e, static_cast<size_t>(rows) * cols +
                     static_cast<size_t>(scale_rows) * sb_s * dtype_size(es.dtype));
    return q;
  }

  // Compressed residency: payload + scales, byte-for-byte (F32 scales; a
  // BF16 scale partner goes through load_quant_rows(name, 0, rows)).
  GlmQuantMatrix load_quant(const std::string& payload_name) {
    const Expected& e = expected(payload_name);
    const Expected& es = expected(payload_name + "_scale_inv");
    if (es.dtype != DType::F32) return load_quant_rows(payload_name, 0, e.shape[0]);
    GlmQuantMatrix q;
    q.rows = e.shape[0];
    q.cols = e.shape[1];
    q.payload = static_cast<const uint8_t*>(load_raw(payload_name));
    q.scales = static_cast<const float*>(load_raw(payload_name + "_scale_inv"));
    return q;
  }

  // Transient bridge for a bf16 interface: compressed copies land in the bump,
  // a dequant job writes the BF16 form alongside. Reads the full source.
  uint16_t* load_dequant_bf16(const std::string& payload_name) {
    const GlmQuantMatrix q = load_quant(payload_name);
    uint16_t* out = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(q.rows) * static_cast<size_t>(q.cols) * 2));
    jobs.push_back(DequantJob{q.payload, q.scales, out, q.rows, q.cols});
    return out;
  }

  // ---- NVFP4 slices (e2m1 pairs + e4m3 scales per 16 + F32 global) ----
  // `base` names the logical [N, K] matrix ("...gate_proj.weight"); the
  // triple is base_packed [N, K/2] U8, base_scale [N, K/16] F8_E4M3 and
  // base_global_scale [1] F32. Row slices need no alignment (every row
  // carries its own scales); column slices start and span whole 16-blocks.
  GlmFp4Matrix load_fp4_rows(const std::string& base, int64_t row_start,
                             int64_t rows, const float* global) {
    const Expected& ep = expected(base + "_packed");
    const Expected& es = expected(base + "_scale");
    const int64_t N = ep.shape[0];
    const int64_t cols = ep.shape[1] * 2;
    fp4_check_cols(cols, who.c_str());
    check_range(base, row_start, rows, N);
    if (es.shape[0] != N || es.shape[1] != cols / kFp4Group)
      fail("NVFP4 scale geometry mismatch on " + base);
    const size_t pc = static_cast<size_t>(cols / 2);
    const size_t sc = static_cast<size_t>(cols / kFp4Group);
    GlmFp4Matrix q;
    q.rows = rows;
    q.cols = cols;
    q.global_scale = global;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * sc));
    if (copy) {
      const TensorInfo& tp = source(ep.name);
      const TensorInfo& ts = source(es.name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(tp.data) + static_cast<size_t>(row_start) * pc,
                  static_cast<size_t>(rows) * pc);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.scales)),
                  static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(row_start) * sc,
                  static_cast<size_t>(rows) * sc);
      consumed(tp);
      consumed(ts);
    }
    note_read(ep, static_cast<size_t>(rows) * pc);
    note_read(es, static_cast<size_t>(rows) * sc);
    return q;
  }

  GlmFp4Matrix load_fp4_cols(const std::string& base, int64_t col_start,
                             int64_t cols, const float* global) {
    const Expected& ep = expected(base + "_packed");
    const Expected& es = expected(base + "_scale");
    const int64_t N = ep.shape[0];
    const int64_t full_cols = ep.shape[1] * 2;
    fp4_check_cols(full_cols, who.c_str());
    fp4_check_cols(cols, who.c_str());
    if (col_start % kFp4Group != 0)
      fail("NVFP4 column slice of '" + base + "' must start on a 16-element block boundary");
    check_range(base, col_start, cols, full_cols);
    if (es.shape[0] != N || es.shape[1] != full_cols / kFp4Group)
      fail("NVFP4 scale geometry mismatch on " + base);
    const size_t pc = static_cast<size_t>(cols / 2), pc_full = static_cast<size_t>(full_cols / 2);
    const size_t sc = static_cast<size_t>(cols / kFp4Group), sc_full = static_cast<size_t>(full_cols / kFp4Group);
    GlmFp4Matrix q;
    q.rows = N;
    q.cols = cols;
    q.global_scale = global;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(N) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(N) * sc));
    if (copy) {
      const TensorInfo& tp = source(ep.name);
      const TensorInfo& ts = source(es.name);
      const uint8_t* sp = static_cast<const uint8_t*>(tp.data);
      const uint8_t* ss = static_cast<const uint8_t*>(ts.data);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      uint8_t* hs = bump.host(const_cast<uint8_t*>(q.scales));
      if (cols == full_cols) {
        std::memcpy(hp, sp, static_cast<size_t>(N) * pc);
        std::memcpy(hs, ss, static_cast<size_t>(N) * sc);
      } else {
        for (int64_t r = 0; r < N; ++r) {
          std::memcpy(hp + r * pc, sp + r * pc_full + col_start / 2, pc);
          std::memcpy(hs + r * sc, ss + r * sc_full + col_start / kFp4Group, sc);
        }
      }
      consumed(tp);
      consumed(ts);
    }
    note_read(ep, static_cast<size_t>(N) * pc);
    note_read(es, static_cast<size_t>(N) * sc);
    return q;
  }

  // One matrix's F32 global scale into its slot of the layer's gathered
  // array (a device address inside the bump). Read whole by every rank.
  void load_fp4_global(const std::string& base, float* slot) {
    const Expected& eg = expected(base + "_global_scale");
    if (!fp4_global(eg) || eg.numel() != 1)
      fail("'" + eg.name + "' is not a global scale");
    if (copy) {
      const TensorInfo& t = source(eg.name);
      std::memcpy(bump.host(slot), t.data, 4);
      consumed(t);
    }
    note_read(eg, 4);
  }

 private:
  // `count` scale entries from element `index` of the source grid into
  // `dst` as F32 — memcpy for an F32 source, widened for BF16.
  void copy_scale_row(const TensorInfo& ts, int64_t index, int64_t count, float* dst) const {
    if (ts.dtype == DType::F32) {
      std::memcpy(dst, static_cast<const float*>(ts.data) + index,
                  static_cast<size_t>(count) * 4);
    } else if (ts.dtype == DType::BF16) {
      const uint8_t* src = static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(index) * 2;
      for (int64_t i = 0; i < count; ++i) {
        uint16_t bits;
        std::memcpy(&bits, src + static_cast<size_t>(i) * 2, 2);
        dst[i] = bf16_bits_to_float(bits);
      }
    } else {
      fail("scale tensor '" + ts.name + "' is neither F32 nor BF16");
    }
  }
};

}  // namespace dgpp
