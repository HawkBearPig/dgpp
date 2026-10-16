#pragma once
// Shared layer-loading lifecycle for Qwen and GLM-4.7. The stream owns
// checkpoint mappings, binding validation, staging, resident images, byte
// accounting and replicated-weight digests. A family provides its geometry
// and builders. Streaming mode reuses one layer allocation; resident mode
// retains an allocation per layer. GLM-5.3 has a separate loader.
//
// A family F provides:
//   using Config, Expected, LayerResident, GlobalsResident, Geometry;
//   struct Builder : WeightBuilder<Expected> — constructed as
//     Builder(cfg, geo, table, by_name, bump, out, tensors, jobs, packs, copy)
//     with void build_layer(int layer);
//   Geometry Geometry::from_config(cfg, rank, world, LoaderHeadSharding)
//     (with int lm_vocab_begin, lm_vocab_count);
//   using PresentMap = std::unordered_map<std::string, <dtype, shape>>;
//   static const char* who();                   // "qwen loader"
//   static uint64_t loader_format();            // the resident layout's version
//   static int max_layer(const Config&);        // main layers + the draft
//   static int main_layers(const Config&);      // the draft layers follow
//   static std::vector<Expected> layer_table(const Config&, int layer);
//   static std::vector<Expected> global_table(const Config&);
//   static void validate_binding(const Config&, const PresentMap&);       // throws
//   static void check_sources(const Config&, const LoaderTensorMap&);     // throws
//   static bool digest_included(const Expected&);   // replicated AND read
//   static bool discard_after_pack(const Expected&);  // packed-source classes
//   static void build_globals(const Config&, const Geometry&, const LoaderTensorMap&,
//                             LayerBump&, GlobalsResident&, uint64_t& source,
//                             uint64_t& verbatim, LoaderHeadSharding);
//   static size_t globals_bytes(const Config&, int rank, int world, LoaderHeadSharding);
//   static size_t extra_resident_bytes(const Config&, int rank, int world);
//   static size_t min_staging_bytes();          // a floor on the staging mirror
//   static void after_restore(const Config&, int layer, const LoaderTensorMap&,
//                             LayerResident&);  // host-side fields a restore re-reads
//   (optional) static size_t globals_side_bytes(const Config&, int rank, int world,
//                             LoaderHeadSharding);  // the globals' side grants
//     — a family whose model packs bf16 matrices into 12-bit companions
//     (common/bf16_residency.hpp) grants them with the builder's `packable`
//     loads / LayerBump::alloc_side, and its model returns their bytes with
//     release_packed() once the companions exist.
// LayerResident carries `int layer` (-1 when empty) and `size_t bytes`;
// GlobalsResident carries `size_t bytes` (0 when not loaded). A derived
// stream implements image_dir() and calls open_resident_image() from its
// constructor.
//
// Synchronization and residency contracts: load boundaries sync the
// registered reader stream + the loader's stream, resident cache hits read
// no storage, release_sources() drops the mappings after the load.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/bf16_residency.hpp"
#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "loaders/resident_image.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"

namespace dgpp {

// Eager parallel layer-load worker count (DGPP_PRELOAD_THREADS): unset or
// <= 1 keeps the serial loop. Shared by the serving families and the
// eagerly-loading model constructors; unset means historical behavior.
inline int loader_preload_threads() {
  int threads = 1;
  if (const char* e = std::getenv("DGPP_PRELOAD_THREADS")) {
    threads = std::atoi(e);
    if (threads < 0) threads = 0;
    if (threads > 16) threads = 16;
  }
  return threads;
}

namespace detail {
// A family may declare `static std::string extra_shard_dir()` and
// `static bool admit_extra_tensor(const std::string&)` (the Qwen n-gram
// table from another snapshot); the stream scans that directory only when
// both exist.
template <class F, class = void>
struct has_extra_shard_dir : std::false_type {};
template <class F>
struct has_extra_shard_dir<F, std::void_t<decltype(F::extra_shard_dir()),
                                          decltype(F::admit_extra_tensor(std::string{}))>>
    : std::true_type {};
}  // namespace detail

enum class LoaderResidency { Streaming, Resident };
enum class LoaderHeadSharding { Full, VocabSharded };

// Boot-time digest over the replicated weight source bytes, per layer plus
// the globals (FNV-1a over name then bytes, summed per layer,
// order-independent).
struct ReplicatedDigest {
  std::vector<uint64_t> layer;
  uint64_t globals = 0;
  uint64_t bytes = 0;
  uint64_t tensors = 0;
};

using LoaderTensorMap = std::unordered_map<std::string, const TensorInfo*>;

template <class F>
class ResidentLayerStream {
 public:
  using Config = typename F::Config;
  using Expected = typename F::Expected;
  using LayerResident = typename F::LayerResident;
  using GlobalsResident = typename F::GlobalsResident;
  using Geometry = typename F::Geometry;

  ResidentLayerStream(const Config& cfg, const std::string& checkpoint_dir, int rank, int world,
                      LoaderResidency residency, LoaderHeadSharding head, bool resident_mtp);
  virtual ~ResidentLayerStream();
  ResidentLayerStream(const ResidentLayerStream&) = delete;
  ResidentLayerStream& operator=(const ResidentLayerStream&) = delete;

  const LayerResident& load_layer(int layer);
  // Eagerly materializes [first, last) on `threads` workers (threads <= 1
  // is the serial load_layer loop). Resident mode only. Each worker owns
  // its pinned staging mirror and CUDA stream, so disk reads, host packs
  // and H2D uploads overlap across layers; the disk itself still
  // serializes at its line rate. Per-layer results are published by index,
  // so completion order does not matter. Throws (after joining) on the
  // first worker error; already-materialized layers stay resident.
  void load_layers_parallel(int first, int last_exclusive, int threads);
  const GlobalsResident& load_globals();
  void release_layer();
  void release_globals();
  void release_sources();
  bool sources_released() const { return sources_released_; }

  int image_layers_restored() const { return image_restored_; }
  int image_layers_captured() const { return image_captured_; }
  std::pair<const void*, size_t> resident_layer_span(int layer) const;
  // A resident layer's bytes in grant order — the staging mirror's and the
  // image's layout, whatever the device placement (side grants included;
  // all of them must still be mapped). `dst` holds layer_bytes. Synchronous.
  void copy_resident_layer(int layer, void* dst) const;

  // The byte formulas (counting builds and closed forms).
  static size_t layer_bytes(const Config& cfg, int layer, int rank = 0, int world = 1);
  static uint64_t planned_layer_source_bytes(const Config& cfg, int layer, int rank, int world);
  static size_t globals_bytes(const Config& cfg, int rank = 0, int world = 1,
                              LoaderHeadSharding head = LoaderHeadSharding::Full) {
    return F::globals_bytes(cfg, rank, world, head);
  }
  static size_t resident_bytes(const Config& cfg, int rank = 0, int world = 1,
                               LoaderHeadSharding head = LoaderHeadSharding::Full,
                               bool with_mtp = false);
  // bf12-only residency: of resident_bytes, the packable bf16 matrices a
  // resident stream grants ASIDE (LayerBump) — returned by release_packed()
  // once the model has packed them.
  static size_t layer_side_bytes(const Config& cfg, int layer, int rank = 0, int world = 1);
  static size_t globals_side_bytes(const Config& cfg, int rank = 0, int world = 1,
                                   LoaderHeadSharding head = LoaderHeadSharding::Full) {
    if constexpr (requires { F::globals_side_bytes(cfg, rank, world, head); })
      return F::globals_side_bytes(cfg, rank, world, head);
    else
      return 0;
  }
  static size_t side_bytes(const Config& cfg, int rank = 0, int world = 1,
                           LoaderHeadSharding head = LoaderHeadSharding::Full, bool with_mtp = false);
  bool side_grants() const { return side_grants_; }
  // Returns a packed matrix's bf16 bytes to the device (layer < 0: the
  // globals'); 0 when `weight` was not granted aside. Nothing outstanding
  // may read it.
  size_t release_packed(int layer, const void* weight);
  // The pinned staging mirror the load needs (the largest layer, the
  // globals, the family's floor): a plan item, because it lives beside the
  // resident weights until release_sources() frees it with the shard
  // mappings. A model that materializes its stack eagerly releases before
  // its caches exist; a lazily loading one keeps the mirror (releasing in
  // the middle of a forward would block on the bus's persistent kernels —
  // cudaFreeHost waits for the device — for a watchdog period).
  static size_t staging_plan_bytes(const Config& cfg, int rank = 0, int world = 1,
                                   LoaderHeadSharding head = LoaderHeadSharding::Full,
                                   bool with_mtp = false);
  bool staging_released() const { return staging_ == nullptr; }
  static int lm_vocab_count(const Config& cfg, int rank = 0, int world = 1,
                            LoaderHeadSharding head = LoaderHeadSharding::Full) {
    return Geometry::from_config(cfg, rank, world, head).lm_vocab_count;
  }

  void set_reader_stream(cudaStream_t reader) { reader_ = reader; }
  const Config& config() const { return cfg_; }
  const Geometry& geometry() const { return geo_; }
  LoaderResidency residency() const { return residency_; }
  size_t layer_capacity() const { return capacity_; }
  int rank() const { return rank_; }
  int world() const { return world_; }
  int max_layer() const { return F::max_layer(cfg_); }

  ReplicatedDigest hash_replicated() const;
  uint64_t source_bytes_read() const { return source_bytes_; }
  uint64_t verbatim_source_bytes() const { return verbatim_bytes_; }

 protected:
  // The resident image directory is per family (a process may host two).
  virtual const std::string& image_dir() const = 0;

  Config cfg_;
  Geometry geo_;
  int rank_ = 0;
  int world_ = 1;
  LoaderResidency residency_ = LoaderResidency::Streaming;
  LoaderHeadSharding head_ = LoaderHeadSharding::Full;
  bool resident_mtp_ = false;
  bool side_grants_ = false;  // packable matrices load into releasable ranges
  cudaStream_t reader_ = nullptr;
  uint64_t source_bytes_ = 0;
  uint64_t verbatim_bytes_ = 0;
  bool sources_released_ = false;
  std::string checkpoint_dir_;
  std::unique_ptr<ResidentImage> image_;
  int image_restored_ = 0;
  int image_captured_ = 0;
  std::vector<std::unique_ptr<SafetensorsFile>> shards_;
  LoaderTensorMap tensors_;
  std::unique_ptr<LayerBump> layer_bump_;  // streaming only
  std::unique_ptr<LayerBump> globals_bump_;
  std::vector<std::unique_ptr<LayerBump>> resident_bumps_;
  std::vector<LayerResident> resident_layers_;
  size_t capacity_ = 0;
  LayerResident resident_{};  // streaming: the active layer
  GlobalsResident globals_{};
  cudaStream_t stream_ = nullptr;
  void* staging_ = nullptr;
  size_t staging_bytes_ = 0;

  // Called by the derived constructor once its own state is ready (the
  // image directory is a virtual, unavailable inside this constructor).
  void open_resident_image();

 private:
  struct CountedBuild {
    size_t bytes = 0;
    uint64_t source_bytes = 0;
    size_t side_bytes = 0;  // of bytes: what a side-grant build places aside
  };
  static CountedBuild count_layer(const Config& cfg, int layer, int rank, int world);
  void check_resident_footprint_fits() const;
  uint64_t resident_image_key() const;
  ReplicatedDigest compute_replicated_digest() const;
  void restore_layer_from_image(int layer, LayerBump& bump, LayerResident& out);
  void capture_layer_to_image(int layer, const LayerBump& bump, size_t bytes);
  void build_layer_into(int layer, LayerBump& bump, LayerResident& out);
  // Parallel-load machinery: one worker owns a staging mirror and a CUDA
  // stream; LayerStats keeps a thread's counters for a post-join merge
  // (the members stay plain: build_globals takes them by reference).
  struct LoadWorker {
    void* staging = nullptr;
    cudaStream_t stream = nullptr;
  };
  struct LayerStats {
    int restored = 0;
    int captured = 0;
    uint64_t source_bytes = 0;
    uint64_t verbatim_bytes = 0;
  };
  LoadWorker make_worker() const;
  void free_worker(const LoadWorker& w) const;
  const LayerResident& load_one_layer(int layer, const LoadWorker& w, LayerStats& st);
  void restore_layer_from_image(int layer, LayerBump& bump, LayerResident& out,
                                const LoadWorker& w, LayerStats& st);
  void capture_layer_to_image(int layer, const LayerBump& bump, size_t bytes,
                              const LoadWorker& w, LayerStats& st);
  void build_layer_into(int layer, LayerBump& bump, LayerResident& out,
                        const LoadWorker& w, LayerStats& st);
  void sync_image_failure();
  std::atomic<bool> image_failed_{false};
};

// ---------------------------------------------------------------------------

template <class F>
ResidentLayerStream<F>::ResidentLayerStream(const Config& cfg, const std::string& checkpoint_dir,
                                            int rank, int world, LoaderResidency residency,
                                            LoaderHeadSharding head, bool resident_mtp)
    : cfg_(cfg),
      geo_(Geometry::from_config(cfg, rank, world, head)),
      rank_(rank),
      world_(world),
      residency_(residency),
      head_(head),
      resident_mtp_(resident_mtp),
      layer_bump_(std::make_unique<LayerBump>()),
      globals_bump_(std::make_unique<LayerBump>()) {
  namespace fs = std::filesystem;
  std::vector<fs::path> shard_paths;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir))
    if (entry.path().extension() == ".safetensors") shard_paths.push_back(entry.path());
  if (shard_paths.empty())
    throw std::runtime_error(std::string(F::who()) + ": no .safetensors shards in " + checkpoint_dir);
  std::sort(shard_paths.begin(), shard_paths.end());
  typename F::PresentMap present;
  for (const auto& path : shard_paths) {
    auto f = SafetensorsFile::open(path.string());
    f->for_each([&](const TensorInfo& t) {
      auto [it, inserted] = tensors_.emplace(t.name, &t);
      if (!inserted)
        throw std::runtime_error(std::string(F::who()) + ": duplicate tensor '" + t.name + "' in " +
                                 path.string());
      present.emplace(t.name, typename F::PresentMap::mapped_type{t.dtype, t.shape});
    });
    shards_.push_back(std::move(f));
  }
  // A family's extra shard directory (the Qwen n-gram table from another
  // snapshot): only the tensors its filter admits, and only when the
  // checkpoint's own shards did not carry them. Everything else in those
  // shards is ignored — never a duplicate, never unexpected.
  if constexpr (detail::has_extra_shard_dir<F>::value) {
    const std::string extra = F::extra_shard_dir();
    if (!extra.empty()) {
      if (!fs::is_directory(extra))
        throw std::runtime_error(std::string(F::who()) + ": the extra shard directory is not a directory: " + extra);
      std::vector<fs::path> extra_paths;
      for (const auto& entry : fs::directory_iterator(extra))
        if (entry.path().extension() == ".safetensors") extra_paths.push_back(entry.path());
      std::sort(extra_paths.begin(), extra_paths.end());
      size_t admitted = 0;
      for (const auto& path : extra_paths) {
        auto f = SafetensorsFile::open(path.string());
        bool used = false;
        f->for_each([&](const TensorInfo& t) {
          if (!F::admit_extra_tensor(t.name) || tensors_.count(t.name)) return;
          tensors_.emplace(t.name, &t);
          present.emplace(t.name, typename F::PresentMap::mapped_type{t.dtype, t.shape});
          used = true;
          ++admitted;
        });
        if (used) shards_.push_back(std::move(f));
      }
      if (admitted == 0)
        throw std::runtime_error(std::string(F::who()) + ": no admissible tensor in the extra shard directory " + extra);
    }
  }
  F::validate_binding(cfg_, present);
  F::check_sources(cfg_, tensors_);

  size_t capacity = 0;
  const int layers = max_layer();
  for (int i = 0; i < layers; ++i) capacity = std::max(capacity, layer_bytes(cfg_, i, rank_, world_));
  capacity_ = capacity;
  if (residency_ == LoaderResidency::Resident) {
    resident_bumps_.resize(static_cast<size_t>(layers));
    resident_layers_.assign(static_cast<size_t>(layers), LayerResident{});
    check_resident_footprint_fits();
  } else {
    layer_bump_->init(capacity);
  }
  // Side grants: resident stacks only — a streaming bump is rewritten every
  // forward and is never packed.
  side_grants_ = residency_ == LoaderResidency::Resident && bf16_side_grants();
  const size_t globals_cap = F::globals_bytes(cfg_, rank_, world_, head_);
  globals_bump_->side_mode = side_grants_;
  globals_bump_->init(globals_cap - (side_grants_ ? globals_side_bytes(cfg_, rank_, world_, head_) : 0));
  staging_bytes_ = std::max({capacity, globals_cap, F::min_staging_bytes()});
  DGPP_CUDA_OK(cudaHostAlloc(&staging_, staging_bytes_, cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaStreamCreate(&stream_));
  checkpoint_dir_ = checkpoint_dir;
}

template <class F>
ResidentLayerStream<F>::~ResidentLayerStream() {
  if (stream_) cudaStreamDestroy(stream_);
  if (staging_) cudaFreeHost(staging_);
}

template <class F>
typename ResidentLayerStream<F>::CountedBuild ResidentLayerStream<F>::count_layer(const Config& cfg,
                                                                                  int layer, int rank,
                                                                                  int world) {
  const Geometry geo = Geometry::from_config(cfg, rank, world, LoaderHeadSharding::Full);
  LayerBump bump;
  bump.counting = true;
  bump.side_mode = true;
  bump.capacity = SIZE_MAX;
  const std::vector<Expected> table = F::layer_table(cfg, layer);
  std::unordered_map<std::string, const Expected*> by_name;
  for (const auto& e : table) by_name.emplace(e.name, &e);
  LayerResident scratch{};
  std::vector<DequantJob> jobs;
  std::vector<PackJob> packs;
  LoaderTensorMap no_tensors;
  typename F::Builder ctx(cfg, geo, table, by_name, bump, scratch, no_tensors, jobs, packs, false);
  ctx.build_layer(layer);
  return CountedBuild{bump.cursor, ctx.source_bytes, bump.side_bytes};
}

template <class F>
size_t ResidentLayerStream<F>::layer_side_bytes(const Config& cfg, int layer, int rank, int world) {
  return count_layer(cfg, layer, rank, world).side_bytes;
}

template <class F>
size_t ResidentLayerStream<F>::side_bytes(const Config& cfg, int rank, int world, LoaderHeadSharding head,
                                          bool with_mtp) {
  size_t total = globals_side_bytes(cfg, rank, world, head);
  const int layers = with_mtp ? F::max_layer(cfg) : F::main_layers(cfg);
  for (int l = 0; l < layers; ++l) total += layer_side_bytes(cfg, l, rank, world);
  return total;
}

template <class F>
size_t ResidentLayerStream<F>::release_packed(int layer, const void* weight) {
  if (!side_grants_) return 0;
  if (layer < 0) return globals_bump_->release_side(weight);
  if (layer >= static_cast<int>(resident_bumps_.size()) || !resident_bumps_[static_cast<size_t>(layer)])
    return 0;
  return resident_bumps_[static_cast<size_t>(layer)]->release_side(weight);
}

template <class F>
size_t ResidentLayerStream<F>::layer_bytes(const Config& cfg, int layer, int rank, int world) {
  return count_layer(cfg, layer, rank, world).bytes;
}

template <class F>
uint64_t ResidentLayerStream<F>::planned_layer_source_bytes(const Config& cfg, int layer, int rank,
                                                            int world) {
  return count_layer(cfg, layer, rank, world).source_bytes;
}

template <class F>
size_t ResidentLayerStream<F>::resident_bytes(const Config& cfg, int rank, int world,
                                              LoaderHeadSharding head, bool with_mtp) {
  size_t total = F::globals_bytes(cfg, rank, world, head) + F::extra_resident_bytes(cfg, rank, world);
  const int main_layers = F::main_layers(cfg);
  for (int l = 0; l < main_layers; ++l) total += layer_bytes(cfg, l, rank, world);
  if (with_mtp)
    for (int l = main_layers; l < F::max_layer(cfg); ++l) total += layer_bytes(cfg, l, rank, world);
  return total;
}

template <class F>
size_t ResidentLayerStream<F>::staging_plan_bytes(const Config& cfg, int rank, int world,
                                                  LoaderHeadSharding head, bool with_mtp) {
  const int layers = with_mtp ? F::max_layer(cfg) : F::main_layers(cfg);
  size_t capacity = 0;
  for (int l = 0; l < layers; ++l) capacity = std::max(capacity, layer_bytes(cfg, l, rank, world));
  return std::max({capacity, F::globals_bytes(cfg, rank, world, head), F::min_staging_bytes()});
}

template <class F>
void ResidentLayerStream<F>::check_resident_footprint_fits() const {
  const size_t footprint = resident_bytes(cfg_, rank_, world_, head_, resident_mtp_);
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return;
  const size_t available = host_mem_available_bytes();
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  const size_t headroom = static_cast<size_t>(4 * kGiB);  // the serving plan's figure (2026-10-09; see models/glm/loader.cpp)
  DGPP_LOG_INFO("{}: rank {} resident footprint {:.1f} GiB; device free {:.1f} of {:.1f} GiB, host "
                "available {:.1f} GiB",
                F::who(), rank_, footprint / kGiB, free_bytes / kGiB, total_bytes / kGiB,
                available / kGiB);
  free_bytes = std::max(free_bytes, available);
  if (footprint + headroom > free_bytes)
    throw std::runtime_error(
        std::string(F::who()) + ": the resident model (" + std::to_string(footprint >> 30) +
        " GiB + 4 GiB headroom) does not fit in the device's free memory (" +
        std::to_string(free_bytes >> 30) +
        " GiB) — free memory on this node, use a larger world, or run in streaming residency");
}

// ---- the resident image ----------------------------------------------------

template <class F>
std::pair<const void*, size_t> ResidentLayerStream<F>::resident_layer_span(int layer) const {
  if (residency_ != LoaderResidency::Resident || layer < 0 ||
      layer >= static_cast<int>(resident_layers_.size()) ||
      resident_layers_[static_cast<size_t>(layer)].layer != layer)
    return {nullptr, 0};
  const LayerBump& b = *resident_bumps_[static_cast<size_t>(layer)];
  return {b.base, b.dev_cursor};
}

template <class F>
void ResidentLayerStream<F>::copy_resident_layer(int layer, void* dst) const {
  if (residency_ != LoaderResidency::Resident || layer < 0 ||
      layer >= static_cast<int>(resident_bumps_.size()) || !resident_bumps_[static_cast<size_t>(layer)])
    throw std::out_of_range(std::string(F::who()) + ": copy_resident_layer of a layer that is not resident");
  resident_bumps_[static_cast<size_t>(layer)]->download(dst);
}

template <class F>
uint64_t ResidentLayerStream<F>::resident_image_key() const {
  uint64_t h = 1469598103934665603ull;
  h = fnv_mix(h, ResidentImage::kFormatVersion);
  h = fnv_mix(h, F::loader_format());
  h = fnv_mix(h, static_cast<uint64_t>(world_));
  h = fnv_mix(h, static_cast<uint64_t>(rank_));
  h = fnv_mix(h, static_cast<uint64_t>(head_));
  {
    std::ifstream f(std::filesystem::path(checkpoint_dir_) / "config.json", std::ios::binary);
    char c;
    while (f.get(c)) h = (h ^ static_cast<uint8_t>(c)) * 1099511628211ull;
  }
  for (const auto& shard : shards_) {
    h = fnv_mix(h, shard->header_fold());
    h = fnv_mix(h, shard->map_size());
  }
  return h;
}

template <class F>
void ResidentLayerStream<F>::open_resident_image() {
  if (residency_ != LoaderResidency::Resident) return;
  const std::string& dir = image_dir();
  if (dir.empty()) return;
  try {
    image_ = std::make_unique<ResidentImage>(dir, resident_image_key(), max_layer());
  } catch (const std::exception& e) {
    DGPP_LOG_WARN("{}: rank {} resident image unavailable ({}) — building from the checkpoint",
                  F::who(), rank_, e.what());
    image_.reset();
    return;
  }
  DGPP_LOG_INFO("{}: rank {} resident image {} — {}/{} layers present, {} I/O", F::who(), rank_,
                image_->path(), image_->present(), image_->layers(),
                image_->direct_io() ? "direct" : "buffered");
}

template <class F>
void ResidentLayerStream<F>::restore_layer_from_image(int layer, LayerBump& bump,
                                                      LayerResident& out,
                                                      const LoadWorker& w, LayerStats& st) {
  bump.stage = w.staging;
  const std::vector<Expected> table = F::layer_table(cfg_, layer);
  std::unordered_map<std::string, const Expected*> by_name;
  for (const auto& e : table) by_name.emplace(e.name, &e);
  std::vector<DequantJob> jobs;
  std::vector<PackJob> packs;
  LoaderTensorMap no_tensors;
  typename F::Builder ctx(cfg_, geo_, table, by_name, bump, out, no_tensors, jobs, packs, false);
  ctx.build_layer(layer);
  const size_t bytes = bump.cursor;
  if (bytes != layer_bytes(cfg_, layer, rank_, world_))
    throw std::runtime_error(std::string(F::who()) + ": layout pass drifted from the byte formula on layer " +
                             std::to_string(layer));
  static const bool verify = [] {
    const char* v = std::getenv("DGPP_RESIDENT_CACHE_VERIFY");
    return v && *v && std::string(v) != "0";
  }();
  image_->read_layer(layer, w.staging, bytes, verify);
  sync_load_boundary(reader_, w.stream);
  bump.upload(w.stream);  // stage == w.staging: the blob is the mirror's layout
  DGPP_CUDA_OK(cudaStreamSynchronize(w.stream));
  F::after_restore(cfg_, layer, tensors_, out);
  out.bytes = bytes;
  ++st.restored;
}

template <class F>
void ResidentLayerStream<F>::capture_layer_to_image(int layer, const LayerBump& bump, size_t bytes,
                                                    const LoadWorker& w, LayerStats& st) {
  // A failed capture poisons the cache for this stream (flagged, reset by
  // sync_image_failure after the layer/pool completes) rather than racing
  // a null image_ against concurrent workers.
  if (image_failed_.load(std::memory_order_relaxed)) return;
  bump.download(w.staging);
  try {
    image_->write_layer(layer, w.staging, bytes);
    ++st.captured;
  } catch (const std::exception& e) {
    if (!image_failed_.exchange(true))
      DGPP_LOG_WARN("{}: rank {} could not capture layer {} to the resident image ({}) — cache "
                    "disabled for this stream", F::who(), rank_, layer, e.what());
  }
}

// ---- layers ------------------------------------------------------------------

template <class F>
void ResidentLayerStream<F>::build_layer_into(int layer, LayerBump& bump, LayerResident& out,
                                             const LoadWorker& w, LayerStats& st) {
  bump.stage = w.staging;
  const std::vector<Expected> table = F::layer_table(cfg_, layer);
  std::unordered_map<std::string, const Expected*> by_name;
  for (const auto& e : table) by_name.emplace(e.name, &e);
  std::vector<DequantJob> jobs;
  std::vector<PackJob> packs;
  typename F::Builder ctx(cfg_, geo_, table, by_name, bump, out, tensors_, jobs, packs, true);
  ctx.one_pass_sources = residency_ == LoaderResidency::Resident;
  ctx.build_layer(layer);
  for (const PackJob& j : packs)
    if (!j.src_on_device) run_host_pack(j, bump);
  if (ctx.one_pass_sources) {
    // The packed column slices read their sources in phase one, above;
    // drop those pages now (the builder could not: the pack ran after it).
    // Per-layer tensor names are disjoint across workers; a shared tensor
    // discarded twice is an idempotent madvise.
    for (const auto& e : table)
      if (F::discard_after_pack(e))
        if (auto it = tensors_.find(e.name); it != tensors_.end() && it->second && it->second->owner)
          it->second->owner->discard(*it->second);
  }
  bump.upload(w.stream);
  if (!jobs.empty())
    throw std::logic_error(std::string(F::who()) + ": no dequant bridges exist for this family");
  for (const PackJob& j : packs)
    if (j.src_on_device) run_device_pack(j, w.stream);
  DGPP_CUDA_OK(cudaStreamSynchronize(w.stream));
  const size_t used = bump.cursor;
  const size_t expected_bytes = layer_bytes(cfg_, layer, rank_, world_);
  if (used != expected_bytes)
    throw std::runtime_error(std::string(F::who()) + ": byte-formula drift on layer " +
                             std::to_string(layer) + ": used " + std::to_string(used) +
                             " != formula " + std::to_string(expected_bytes));
  const uint64_t planned = planned_layer_source_bytes(cfg_, layer, rank_, world_);
  if (ctx.source_bytes != planned)
    throw std::runtime_error(std::string(F::who()) + ": source-byte plan drift on layer " +
                             std::to_string(layer) + ": read " + std::to_string(ctx.source_bytes) +
                             " != planned " + std::to_string(planned));
  out.bytes = used;
  st.source_bytes += ctx.source_bytes;
  st.verbatim_bytes += ctx.verbatim_bytes;
}

template <class F>
const typename F::LayerResident& ResidentLayerStream<F>::load_layer(int layer) {
  if (residency_ != LoaderResidency::Resident) {
    if (resident_.layer == layer) return resident_;
    sync_load_boundary(reader_, stream_);
    layer_bump_->reset();
    resident_ = LayerResident{};
    LayerStats st;
    build_layer_into(layer, *layer_bump_, resident_, LoadWorker{staging_, stream_}, st);
    source_bytes_ += st.source_bytes;
    verbatim_bytes_ += st.verbatim_bytes;
    return resident_;
  }
  LayerStats st;
  const LayerResident& out = load_one_layer(layer, LoadWorker{staging_, stream_}, st);
  source_bytes_ += st.source_bytes;
  verbatim_bytes_ += st.verbatim_bytes;
  image_restored_ += st.restored;
  image_captured_ += st.captured;
  sync_image_failure();
  return out;
}

template <class F>
const typename F::LayerResident& ResidentLayerStream<F>::load_one_layer(int layer, const LoadWorker& w,
                                                                        LayerStats& st) {
  if (layer < 0 || layer >= static_cast<int>(resident_layers_.size()))
    throw std::out_of_range(std::string(F::who()) + ": layer index out of range: " + std::to_string(layer));
  LayerResident& slot = resident_layers_[static_cast<size_t>(layer)];
  if (slot.layer == layer) return slot;
  if (sources_released_)
    throw std::runtime_error(std::string(F::who()) + ": layer " + std::to_string(layer) +
                             " was never materialized before release_sources()");
  auto bump = std::make_unique<LayerBump>();
  bump->side_mode = side_grants_;
  const CountedBuild counted = count_layer(cfg_, layer, rank_, world_);
  const size_t cap = counted.bytes - (side_grants_ ? counted.side_bytes : 0);
  bump->init(cap);
  if (image_ && image_->has_layer(layer)) {
    restore_layer_from_image(layer, *bump, slot, w, st);
  } else {
    // Grants are 256-aligned but tensors are not: each grant's tail is
    // never written by the build. Zero the mirror first so those tails
    // (folded into the image entry, uploaded to the device) are
    // deterministic — otherwise they leak whatever staging held before
    // (the previous layer serially, garbage on a fresh worker).
    if (cap > 0) std::memset(w.staging, 0, cap);
    sync_load_boundary(reader_, w.stream);
    build_layer_into(layer, *bump, slot, w, st);
    if (image_) capture_layer_to_image(layer, *bump, slot.bytes, w, st);
  }
  resident_bumps_[static_cast<size_t>(layer)] = std::move(bump);
  return slot;
}

template <class F>
typename ResidentLayerStream<F>::LoadWorker ResidentLayerStream<F>::make_worker() const {
  LoadWorker w;
  DGPP_CUDA_OK(cudaHostAlloc(&w.staging, staging_bytes_, cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaStreamCreate(&w.stream));
  return w;
}

template <class F>
void ResidentLayerStream<F>::free_worker(const LoadWorker& w) const {
  if (w.stream) cudaStreamDestroy(w.stream);
  if (w.staging) cudaFreeHost(w.staging);
}

template <class F>
void ResidentLayerStream<F>::sync_image_failure() {
  if (image_failed_.exchange(false)) image_.reset();
}

template <class F>
void ResidentLayerStream<F>::load_layers_parallel(int first, int last_exclusive, int threads) {
  if (residency_ != LoaderResidency::Resident)
    throw std::logic_error(std::string(F::who()) + ": load_layers_parallel needs resident residency");
  if (threads <= 1 || last_exclusive - first <= 1) {
    for (int l = first; l < last_exclusive; ++l) load_layer(l);
    return;
  }
  threads = std::max(2, std::min(threads, 16));
  DGPP_LOG_INFO("{}: rank {} preloading layers [{}, {}) on {} workers (each {:.2f} GiB pinned staging)",
                F::who(), rank_, first, last_exclusive, threads,
                static_cast<double>(staging_bytes_) / (1024.0 * 1024.0 * 1024.0));
  std::vector<LoadWorker> workers;
  workers.reserve(static_cast<size_t>(threads));
  try {
    for (int i = 0; i < threads; ++i) workers.push_back(make_worker());
  } catch (...) {
    for (const LoadWorker& w : workers) free_worker(w);
    throw;
  }
  std::atomic<int> next(first);
  std::exception_ptr error;
  std::mutex error_mutex;
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(threads));
  for (int i = 0; i < threads; ++i) {
    pool.emplace_back([&, i] {
      LayerStats st;
      const auto merge = [&] {
        std::lock_guard<std::mutex> lk(error_mutex);
        source_bytes_ += st.source_bytes;
        verbatim_bytes_ += st.verbatim_bytes;
        image_restored_ += st.restored;
        image_captured_ += st.captured;
      };
      try {
        for (int l = next.fetch_add(1); l < last_exclusive; l = next.fetch_add(1)) {
          LayerStats one;
          load_one_layer(l, workers[static_cast<size_t>(i)], one);
          st.restored += one.restored;
          st.captured += one.captured;
          st.source_bytes += one.source_bytes;
          st.verbatim_bytes += one.verbatim_bytes;
        }
      } catch (...) {
        merge();
        std::lock_guard<std::mutex> lk(error_mutex);
        if (!error) error = std::current_exception();
        return;
      }
      // Merge under the error mutex: workers finish at different times,
      // the members stay plain and race-free.
      merge();
    });
  }
  for (auto& th : pool) th.join();
  for (const LoadWorker& w : workers) free_worker(w);
  sync_image_failure();
  if (error) std::rethrow_exception(error);
}

template <class F>
void ResidentLayerStream<F>::release_layer() {
  if (residency_ == LoaderResidency::Resident) return;
  resident_ = LayerResident{};
  layer_bump_->reset();
}

// ---- globals -----------------------------------------------------------------

template <class F>
const typename F::GlobalsResident& ResidentLayerStream<F>::load_globals() {
  if (globals_.bytes != 0) return globals_;
  if (sources_released_)
    throw std::runtime_error(std::string(F::who()) + ": load_globals after the checkpoint sources were released");
  sync_load_boundary(reader_, stream_);
  globals_bump_->reset();
  globals_bump_->stage = staging_;
  globals_ = GlobalsResident{};
  F::build_globals(cfg_, geo_, tensors_, *globals_bump_, globals_, source_bytes_, verbatim_bytes_, head_);
  globals_.bytes = globals_bump_->cursor;
  const size_t expected_bytes = F::globals_bytes(cfg_, rank_, world_, head_);
  if (globals_.bytes != expected_bytes)
    throw std::runtime_error(std::string(F::who()) + ": globals byte-formula drift: used " +
                             std::to_string(globals_.bytes) + " != formula " +
                             std::to_string(expected_bytes));
  globals_bump_->upload(stream_);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  return globals_;
}

template <class F>
void ResidentLayerStream<F>::release_globals() {
  globals_ = GlobalsResident{};
  globals_bump_->reset();
}

// ---- sources and digests --------------------------------------------------------

template <class F>
void ResidentLayerStream<F>::release_sources() {
  if (residency_ != LoaderResidency::Resident || sources_released_) return;
  sources_released_ = true;
  tensors_.clear();
  const size_t shard_count = shards_.size();
  uint64_t mapped_bytes = 0;
  for (auto& shard : shards_) {
    mapped_bytes += shard->map_size();
    shard->close_mapping(/*drop_page_cache=*/true);
  }
  shards_.clear();
  const size_t staging_freed = staging_bytes_;
  if (staging_) {
    DGPP_CUDA_OK(cudaFreeHost(staging_));
    staging_ = nullptr;
    staging_bytes_ = 0;
    globals_bump_->stage = nullptr;
  }
  DGPP_LOG_INFO("{}: rank {} resident load complete — released {} shard mappings ({:.1f} GiB) and "
                "evicted their page cache, freed the {:.2f} GiB pinned staging mirror; image: {} layers "
                "restored, {} captured",
                F::who(), rank_, shard_count, static_cast<double>(mapped_bytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(staging_freed) / (1024.0 * 1024.0 * 1024.0), image_restored_,
                image_captured_);
}

namespace resident_stream_detail {
constexpr uint64_t kDigestNoteFormat = 0x4447505044494731ull;  // "DGPPDIG1"
constexpr const char* kDigestNoteName = "digest";
inline std::vector<uint64_t> serialize_digest(const ReplicatedDigest& d) {
  std::vector<uint64_t> w = {kDigestNoteFormat, d.tensors, d.bytes, d.globals, d.layer.size()};
  w.insert(w.end(), d.layer.begin(), d.layer.end());
  return w;
}
inline bool deserialize_digest(const std::vector<uint64_t>& w, size_t layers, ReplicatedDigest& d) {
  if (w.size() != 5 + layers || w[0] != kDigestNoteFormat || w[4] != layers) return false;
  d.tensors = w[1];
  d.bytes = w[2];
  d.globals = w[3];
  d.layer.assign(w.begin() + 5, w.end());
  return true;
}
}  // namespace resident_stream_detail

template <class F>
ReplicatedDigest ResidentLayerStream<F>::hash_replicated() const {
  using namespace resident_stream_detail;
  if (sources_released_)
    throw std::runtime_error(std::string(F::who()) + ": hash_replicated after the checkpoint sources were released");
  const size_t layers = static_cast<size_t>(max_layer());
  if (image_) {
    std::vector<uint64_t> words(5 + layers);
    ReplicatedDigest d;
    if (image_->read_note(kDigestNoteName, words.data(), words.size() * sizeof(uint64_t)) &&
        deserialize_digest(words, layers, d))
      return d;
  }
  ReplicatedDigest d = compute_replicated_digest();
  if (image_) {
    try {
      const std::vector<uint64_t> words = serialize_digest(d);
      image_->write_note(kDigestNoteName, words.data(), words.size() * sizeof(uint64_t));
    } catch (const std::exception& e) {
      DGPP_LOG_WARN("{}: rank {} could not publish the digest note ({})", F::who(), rank_, e.what());
    }
  }
  return d;
}

template <class F>
ReplicatedDigest ResidentLayerStream<F>::compute_replicated_digest() const {
  const auto lookup = [this](const std::string& name) -> const TensorInfo& {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error(std::string(F::who()) + ": tensor not in checkpoint: " + name);
    return *it->second;
  };
  ReplicatedDigest d;
  const int layers = max_layer();
  d.layer.assign(static_cast<size_t>(layers), 0);
  for (int l = 0; l < layers; ++l) {
    uint64_t sum = 0;
    for (const Expected& e : F::layer_table(cfg_, l)) {
      if (!F::digest_included(e)) continue;
      const TensorInfo& t = lookup(e.name);
      const size_t bytes = e.nbytes();
      if (t.numel() * dtype_size(t.dtype) != bytes)
        throw std::runtime_error(std::string(F::who()) + ": replicated tensor shape drift on '" + e.name + "'");
      sum += fnv1a(t.data, bytes, fnv1a(e.name.data(), e.name.size(), 1469598103934665603ull));
      d.bytes += bytes;
      ++d.tensors;
    }
    d.layer[static_cast<size_t>(l)] = sum;
  }
  uint64_t g = 0;
  for (const Expected& e : F::global_table(cfg_)) {
    const TensorInfo& t = lookup(e.name);
    const size_t bytes = e.nbytes();
    g += fnv1a(t.data, bytes, fnv1a(e.name.data(), e.name.size(), 1469598103934665603ull));
    d.bytes += bytes;
    ++d.tensors;
  }
  d.globals = g;
  return d;
}

}  // namespace dgpp
