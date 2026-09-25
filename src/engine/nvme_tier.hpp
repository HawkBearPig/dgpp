#pragma once
// The prefix cache's NVMe cold tier (issue #26): a preallocated slab file
// per rank holding evicted entries' state — the snapshot blob and the
// cache blocks the entry's metadata pins — and a worker that copies
// entries out and back on its own stream, off the decode path.
//
// The slab is `pages` records of `page_bytes`, one cache block's planes
// (engine/cache_planes.hpp) rounded up to 4 KiB; a blob spans blob_pages
// contiguous pages. Which entry sits where is the scheduler's DiskCache
// (sched/disk_cache.hpp): identical on every rank, while this file's
// bytes are this rank's shard. The tier runs the ops it is handed:
//
//   spill   gather each block's planes into a device staging record,
//           copy it to pinned memory, write it to its page (direct I/O,
//           no page cache — the GB10's unified memory is the model's);
//           the blob likewise, in page runs.
//   restore read each page into pinned memory, verify its CRC-32C,
//           copy it to the device and scatter the planes into the fresh
//           block the restore acquired; the blob into the arena slot.
//
// Two staging buffers pipeline the GPU copies against the file I/O. An
// op's outcome (ok, or a short read/write, a device error, a checksum
// mismatch) is reported through poll(); rank 0 journals every rank's
// verdict as one commit, so a failure on any rank is the same cold miss
// everywhere. The worker finishes every device op before it reports, so
// a released restore never has a scatter in flight.
//
// Entries are the process's: the slab is recreated at every start (no
// reuse across restarts — a later version may bind records to the
// checkpoint and the configuration digest).
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/crc32c.hpp"
#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "engine/cache_planes.hpp"
#include "engine/prefix_arena.hpp"
#include "kernels/segment_copy.hpp"
#include "sched/scheduler.hpp"

namespace dgpp {

template <class Model>
class NvmeTier {
 public:
  using Config = sched::SchedulerEngine::DiskEnable;
  static constexpr size_t kAlign = 4096;

  // The host and device bytes the tier allocates for a chunk size (the
  // memory plan): two device and two pinned staging buffers plus the
  // segment tables.
  static size_t staging_bytes(size_t chunk_bytes) { return 4 * chunk_bytes + (size_t{8} << 20); }

  NvmeTier(Model* model, PrefixArena<Model>* arena, const Config& cfg)
      : model_(model), arena_(arena), cfg_(cfg) {
    if (model_ == nullptr || arena_ == nullptr) throw std::invalid_argument("NvmeTier: null model or arena");
    if (arena_->slots() <= 0) throw std::invalid_argument("NvmeTier: the prefix arena has no slots");
    if (cfg_.chunk_bytes < kAlign || cfg_.chunk_bytes % kAlign != 0)
      throw std::invalid_argument("NvmeTier: the chunk must be a positive 4 KiB multiple");
    planes_ = model_->cache_planes();
    size_t off = 0;
    for (const CachePlane& p : planes_) {
      if (p.base == nullptr || p.block_bytes == 0) throw std::invalid_argument("NvmeTier: an empty cache plane");
      plane_off_.push_back(off);
      off += p.block_bytes;
    }
    block_bytes_ = off;
    page_bytes_ = std::max<size_t>(kAlign, (block_bytes_ + kAlign - 1) / kAlign * kAlign);
    if (page_bytes_ > cfg_.chunk_bytes)
      throw std::invalid_argument("NvmeTier: one cache block (" + std::to_string(page_bytes_) +
                                  " bytes) exceeds the staging chunk");
    blob_bytes_ = arena_->bytes();
    blob_pages_ = static_cast<int64_t>((blob_bytes_ + page_bytes_ - 1) / page_bytes_);
    if (blob_pages_ < 1) blob_pages_ = 1;
    pages_ = static_cast<int64_t>(cfg_.capacity_bytes / page_bytes_);
    if (pages_ < blob_pages_ + 1)
      throw std::invalid_argument("NvmeTier: a capacity of " + std::to_string(cfg_.capacity_bytes) +
                                  " bytes holds fewer than one blob (" + std::to_string(blob_pages_) +
                                  " pages of " + std::to_string(page_bytes_) + " bytes) and one block");
    per_chunk_ = static_cast<int64_t>(cfg_.chunk_bytes / page_bytes_);
    open_slab();
    DGPP_CUDA_OK(cudaGetDevice(&device_));
    DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    const size_t max_segments = static_cast<size_t>(per_chunk_) * std::max<size_t>(1, planes_.size());
    for (int b = 0; b < 2; ++b) {
      DGPP_CUDA_OK(cudaMalloc(&d_stage_[b], cfg_.chunk_bytes));
      DGPP_CUDA_OK(cudaHostAlloc(&h_stage_[b], cfg_.chunk_bytes, cudaHostAllocDefault));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_segs_[b]), max_segments * sizeof(CopySegment)));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_segs_[b]), max_segments * sizeof(CopySegment),
                                 cudaHostAllocDefault));
      DGPP_CUDA_OK(cudaEventCreateWithFlags(&done_[b], cudaEventDisableTiming));
    }
    crc_.assign(static_cast<size_t>(pages_), 0);
    worker_ = std::thread([this] { worker(); });
    DGPP_LOG_INFO(
        "rank {}: NVMe cache open — {} ({:.2f} GiB): {} pages of {} bytes ({} bytes of cache block in {} planes), "
        "{} pages per snapshot blob ({:.1f} MiB), {} MiB staging chunks, {} I/O",
        cfg_.rank, cfg_.slab_path, static_cast<double>(cfg_.capacity_bytes) / (1024.0 * 1024.0 * 1024.0), pages_,
        page_bytes_, block_bytes_, planes_.size(), blob_pages_, static_cast<double>(blob_bytes_) / (1024.0 * 1024.0),
        cfg_.chunk_bytes >> 20, direct_ ? "direct" : "buffered");
  }
  ~NvmeTier() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    for (int b = 0; b < 2; ++b) {
      if (done_[b]) cudaEventDestroy(done_[b]);
      if (d_stage_[b]) cudaFree(d_stage_[b]);
      if (h_stage_[b]) cudaFreeHost(h_stage_[b]);
      if (d_segs_[b]) cudaFree(d_segs_[b]);
      if (h_segs_[b]) cudaFreeHost(h_segs_[b]);
    }
    if (stream_) cudaStreamDestroy(stream_);
    if (fd_ >= 0) {
      ::close(fd_);
      ::unlink(cfg_.slab_path.c_str());
    }
  }
  NvmeTier(const NvmeTier&) = delete;
  NvmeTier& operator=(const NvmeTier&) = delete;

  int64_t pages() const { return pages_; }
  size_t page_bytes() const { return page_bytes_; }
  int64_t blob_pages() const { return blob_pages_; }
  size_t block_bytes() const { return block_bytes_; }

  sched::SchedulerEngine::DiskInfo info() const {
    sched::SchedulerEngine::DiskInfo i;
    i.enabled = true;
    i.pages = pages_;
    i.page_bytes = page_bytes_;
    i.blob_pages = blob_pages_;
    i.min_tokens = cfg_.min_tokens;
    return i;
  }

  sched::SchedulerEngine::DiskBlocks entry_blocks(int slot) const {
    const auto& meta = arena_->meta(slot);
    if (!arena_->filled(slot)) throw std::logic_error("NvmeTier: the arena slot is empty");
    sched::SchedulerEngine::DiskBlocks out;
    out.identities.reserve(meta.full_blocks.size());
    for (const int32_t b : meta.full_blocks) out.identities.push_back(model_->cache_block_identity(b));
    out.partial_identity = meta.partial_block >= 0 ? model_->cache_block_identity(meta.partial_block) : 0;
    out.mtp_position = meta.mtp_position;
    return out;
  }

  void spill_begin(const sched::SchedulerEngine::DiskSpill& spill) {
    if (!arena_->filled(spill.slot)) throw std::logic_error("NvmeTier: spilling an empty arena slot");
    const auto& meta = arena_->meta(spill.slot);
    if (spill.pages.size() != meta.full_blocks.size())
      throw std::invalid_argument("NvmeTier: the spill's pages do not cover the entry's blocks");
    if ((spill.partial_page >= 0) != (meta.partial_block >= 0))
      throw std::invalid_argument("NvmeTier: the spill's partial page does not match the entry");
    Job job;
    job.op = spill.op;
    job.restore = false;
    job.slot = spill.slot;
    job.blob_page = spill.blob_page;
    for (size_t i = 0; i < spill.pages.size(); ++i)
      if (spill.pages[i] >= 0) job.blocks.emplace_back(meta.full_blocks[i], spill.pages[i]);
    if (spill.partial_page >= 0) job.blocks.emplace_back(meta.partial_block, spill.partial_page);
    check_pages(job);
    enqueue(std::move(job));
  }

  sched::SchedulerEngine::DiskBlocks restore_begin(const sched::SchedulerEngine::DiskRestore& restore) {
    if (restore.position <= 0) throw std::invalid_argument("NvmeTier: restore at position 0");
    const int64_t bt = model_->kv_block_tokens();
    const int64_t n_full = bt > 0 ? restore.position / bt : 0;
    if (static_cast<int64_t>(restore.pages.size()) != n_full)
      throw std::invalid_argument("NvmeTier: the restore's pages do not cover the position's blocks");
    const bool partial = bt > 0 && restore.position % bt != 0;
    if (partial != (restore.partial_page >= 0))
      throw std::invalid_argument("NvmeTier: the restore's partial page does not match the position");
    typename Model::SessionSnapshotMeta meta;
    meta.position = restore.position;
    meta.mtp_position = restore.mtp_position;
    std::vector<int32_t> taken;
    const auto take = [&]() -> int32_t {
      const int32_t b = model_->cache_acquire_block();
      if (b < 0) {
        model_->cache_release_blocks(taken.data(), static_cast<int64_t>(taken.size()));
        throw std::runtime_error("NvmeTier: the pool has no free block for the restore");
      }
      taken.push_back(b);
      return b;
    };
    Job job;
    job.op = restore.op;
    job.restore = true;
    job.slot = restore.slot;
    job.blob_page = restore.blob_page;
    sched::SchedulerEngine::DiskBlocks out;
    for (int64_t i = 0; i < n_full; ++i) {
      const int32_t b = take();
      meta.full_blocks.push_back(b);
      job.blocks.emplace_back(b, restore.pages[static_cast<size_t>(i)]);
      out.identities.push_back(model_->cache_block_identity(b));
    }
    if (partial) {
      const int32_t b = take();
      meta.partial_block = b;
      job.blocks.emplace_back(b, restore.partial_page);
      out.partial_identity = model_->cache_block_identity(b);
    }
    out.mtp_position = restore.mtp_position;
    check_pages(job);
    // The slot owns the blocks from here: a release (a failed restore, an
    // eviction) returns them exactly as it returns a snapshot's.
    arena_->adopt(restore.slot, std::move(meta));
    enqueue(std::move(job));
    return out;
  }

  std::vector<sched::SchedulerEngine::DiskCompletion> poll() {
    std::vector<sched::SchedulerEngine::DiskCompletion> out;
    std::lock_guard<std::mutex> lock(mutex_);
    out.swap(completions_);
    return out;
  }

  sched::SchedulerEngine::DiskEngineStats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    sched::SchedulerEngine::DiskEngineStats s = stats_;
    s.pending = static_cast<int64_t>(jobs_.size()) + (running_ ? 1 : 0);
    return s;
  }

 private:
  struct Job {
    uint64_t op = 0;
    bool restore = false;
    int slot = -1;
    int64_t blob_page = -1;
    std::vector<std::pair<int32_t, int64_t>> blocks;  // (physical block, page)
    cudaEvent_t start = nullptr;  // the model stream's position when the op began
  };

  void check_pages(const Job& job) const {
    const auto bad = [&](int64_t page) { return page < 0 || page >= pages_; };
    if (bad(job.blob_page) || job.blob_page + blob_pages_ > pages_)
      throw std::invalid_argument("NvmeTier: the blob's pages are outside the slab");
    for (const auto& [block, page] : job.blocks) {
      (void)block;
      if (bad(page)) throw std::invalid_argument("NvmeTier: a block's page is outside the slab");
    }
  }

  void open_slab() {
    // Direct I/O first: the page cache would grow into the memory the
    // model plans on. A filesystem that refuses it (tmpfs, the tests) gets
    // buffered I/O, said out loud.
    const int flags = O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC;
    fd_ = ::open(cfg_.slab_path.c_str(), flags | O_DIRECT, 0600);
    direct_ = fd_ >= 0;
    if (fd_ < 0 && errno == EINVAL) fd_ = ::open(cfg_.slab_path.c_str(), flags, 0600);
    if (fd_ < 0)
      throw std::runtime_error("NvmeTier: cannot create " + cfg_.slab_path + ": " + std::strerror(errno));
    const off_t bytes = static_cast<off_t>(static_cast<size_t>(pages_) * page_bytes_);
    // The whole capacity claimed now (unwritten extents): a full disk is a
    // startup error, never a torn write later.
    if (::posix_fallocate(fd_, 0, bytes) != 0) {
      const int err = errno;
      if (::ftruncate(fd_, bytes) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("NvmeTier: cannot size " + cfg_.slab_path + " to " + std::to_string(bytes) +
                                 " bytes: " + std::strerror(err));
      }
      DGPP_LOG_WARN("rank {}: NVMe cache: the filesystem cannot preallocate {} ({}); the slab is sparse and a "
                    "full disk would fail a spill instead",
                    cfg_.rank, cfg_.slab_path, std::strerror(err));
    }
  }

  void enqueue(Job job) {
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&job.start, cudaEventDisableTiming));
    DGPP_CUDA_OK(cudaEventRecord(job.start, model_->stream()));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      jobs_.push_back(std::move(job));
    }
    cv_.notify_one();
  }

  void worker() {
    cudaSetDevice(device_);
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
        if (stop_ && jobs_.empty()) return;
        job = std::move(jobs_.front());
        jobs_.pop_front();
        running_ = true;
      }
      const auto t0 = std::chrono::steady_clock::now();
      bool ok = false;
      std::string why;
      try {
        ok = job.restore ? run_restore(job, &why) : run_spill(job, &why);
      } catch (const std::exception& e) {
        why = e.what();
        ok = false;
      }
      // Every device op of this job finishes before its verdict: a
      // released slot never has a scatter in flight.
      if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        ok = false;
        why = "the tier's stream failed";
      }
      if (job.start) cudaEventDestroy(job.start);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (!ok)
        DGPP_LOG_ERROR("rank {}: NVMe cache {} op {} failed: {}", cfg_.rank, job.restore ? "restore" : "spill", job.op,
                       why);
      else
        DGPP_LOG_DEBUG("rank {}: NVMe cache {} op {} done in {:.1f} ms ({} blocks)", cfg_.rank,
                       job.restore ? "restore" : "spill", job.op, ms, job.blocks.size());
      std::lock_guard<std::mutex> lock(mutex_);
      running_ = false;
      if (job.restore) {
        ++stats_.restores;
        stats_.restore_ms += ms;
      } else {
        ++stats_.spills;
        stats_.spill_ms += ms;
      }
      if (!ok) ++stats_.failures;
      completions_.push_back({job.op, ok});
    }
  }

  // ---- the file --------------------------------------------------------
  bool write_page(int64_t page, const void* buf, std::string* why) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t left = page_bytes_;
    off_t off = static_cast<off_t>(static_cast<size_t>(page) * page_bytes_);
    while (left > 0) {
      const ssize_t n = ::pwrite(fd_, p, left, off);
      if (n < 0) {
        if (errno == EINTR) continue;
        *why = "write of page " + std::to_string(page) + ": " + std::strerror(errno);
        return false;
      }
      if (n == 0) {
        *why = "short write of page " + std::to_string(page);
        return false;
      }
      p += n;
      off += n;
      left -= static_cast<size_t>(n);
    }
    crc_[static_cast<size_t>(page)] = crc32c(buf, page_bytes_);
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.bytes_written += static_cast<int64_t>(page_bytes_);
    return true;
  }
  bool read_page(int64_t page, void* buf, std::string* why) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t left = page_bytes_;
    off_t off = static_cast<off_t>(static_cast<size_t>(page) * page_bytes_);
    while (left > 0) {
      const ssize_t n = ::pread(fd_, p, left, off);
      if (n < 0) {
        if (errno == EINTR) continue;
        *why = "read of page " + std::to_string(page) + ": " + std::strerror(errno);
        return false;
      }
      if (n == 0) {
        *why = "short read of page " + std::to_string(page);
        return false;
      }
      p += n;
      off += n;
      left -= static_cast<size_t>(n);
    }
    if (crc32c(buf, page_bytes_) != crc_[static_cast<size_t>(page)]) {
      *why = "checksum mismatch on page " + std::to_string(page);
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.bytes_read += static_cast<int64_t>(page_bytes_);
    return true;
  }

  // ---- the ops ---------------------------------------------------------
  // The blocks in batches of per_chunk_ records: batch k gathers on buffer
  // k % 2 while batch k - 1's pages are written from the other buffer.
  bool run_spill(const Job& job, std::string* why) {
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream_, job.start, 0));
    const int64_t n = static_cast<int64_t>(job.blocks.size());
    int64_t issued = 0;  // batches issued
    const auto issue = [&](int64_t first, int64_t count, int buf) {
      CopySegment* segs = h_segs_[buf];
      size_t ns = 0;
      for (int64_t j = 0; j < count; ++j) {
        const int32_t block = job.blocks[static_cast<size_t>(first + j)].first;
        uint8_t* record = static_cast<uint8_t*>(d_stage_[buf]) + static_cast<size_t>(j) * page_bytes_;
        for (size_t p = 0; p < planes_.size(); ++p) {
          segs[ns].src = planes_[p].base + static_cast<size_t>(block) * planes_[p].block_bytes;
          segs[ns].dst = record + plane_off_[p];
          segs[ns].bytes = planes_[p].block_bytes;
          ++ns;
        }
      }
      DGPP_CUDA_OK(cudaMemcpyAsync(d_segs_[buf], segs, ns * sizeof(CopySegment), cudaMemcpyHostToDevice, stream_));
      segment_copy(d_segs_[buf], static_cast<int>(ns), stream_);
      DGPP_CUDA_OK(cudaMemcpyAsync(h_stage_[buf], d_stage_[buf], static_cast<size_t>(count) * page_bytes_,
                                   cudaMemcpyDeviceToHost, stream_));
      DGPP_CUDA_OK(cudaEventRecord(done_[buf], stream_));
    };
    const auto flush = [&](int64_t first, int64_t count, int buf) -> bool {
      DGPP_CUDA_OK(cudaEventSynchronize(done_[buf]));
      for (int64_t j = 0; j < count; ++j)
        if (!write_page(job.blocks[static_cast<size_t>(first + j)].second,
                        static_cast<const uint8_t*>(h_stage_[buf]) + static_cast<size_t>(j) * page_bytes_, why))
          return false;
      return true;
    };
    int64_t prev_first = 0, prev_count = 0;
    for (int64_t first = 0; first < n; first += per_chunk_) {
      const int64_t count = std::min<int64_t>(per_chunk_, n - first);
      const int buf = static_cast<int>(issued % 2);
      issue(first, count, buf);
      if (issued > 0 && !flush(prev_first, prev_count, 1 - buf)) return false;
      prev_first = first;
      prev_count = count;
      ++issued;
    }
    if (issued > 0 && !flush(prev_first, prev_count, static_cast<int>((issued - 1) % 2))) return false;
    // The blob, in page runs.
    const uint8_t* blob = static_cast<const uint8_t*>(arena_->slot_data(job.slot));
    for (int64_t first = 0; first < blob_pages_; first += per_chunk_) {
      const int64_t count = std::min<int64_t>(per_chunk_, blob_pages_ - first);
      const size_t offset = static_cast<size_t>(first) * page_bytes_;
      const size_t bytes = std::min(static_cast<size_t>(count) * page_bytes_, blob_bytes_ - offset);
      DGPP_CUDA_OK(cudaMemcpyAsync(h_stage_[0], blob + offset, bytes, cudaMemcpyDeviceToHost, stream_));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      for (int64_t j = 0; j < count; ++j)
        if (!write_page(job.blob_page + first + j,
                        static_cast<const uint8_t*>(h_stage_[0]) + static_cast<size_t>(j) * page_bytes_, why))
          return false;
    }
    return true;
  }

  bool run_restore(const Job& job, std::string* why) {
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream_, job.start, 0));
    const int64_t n = static_cast<int64_t>(job.blocks.size());
    int64_t issued = 0;
    for (int64_t first = 0; first < n; first += per_chunk_) {
      const int64_t count = std::min<int64_t>(per_chunk_, n - first);
      const int buf = static_cast<int>(issued % 2);
      // The buffer's previous scatter must have consumed it.
      DGPP_CUDA_OK(cudaEventSynchronize(done_[buf]));
      for (int64_t j = 0; j < count; ++j)
        if (!read_page(job.blocks[static_cast<size_t>(first + j)].second,
                       static_cast<uint8_t*>(h_stage_[buf]) + static_cast<size_t>(j) * page_bytes_, why))
          return false;
      DGPP_CUDA_OK(cudaMemcpyAsync(d_stage_[buf], h_stage_[buf], static_cast<size_t>(count) * page_bytes_,
                                   cudaMemcpyHostToDevice, stream_));
      CopySegment* segs = h_segs_[buf];
      size_t ns = 0;
      for (int64_t j = 0; j < count; ++j) {
        const int32_t block = job.blocks[static_cast<size_t>(first + j)].first;
        const uint8_t* record = static_cast<const uint8_t*>(d_stage_[buf]) + static_cast<size_t>(j) * page_bytes_;
        for (size_t p = 0; p < planes_.size(); ++p) {
          segs[ns].src = record + plane_off_[p];
          segs[ns].dst = planes_[p].base + static_cast<size_t>(block) * planes_[p].block_bytes;
          segs[ns].bytes = planes_[p].block_bytes;
          ++ns;
        }
      }
      DGPP_CUDA_OK(cudaMemcpyAsync(d_segs_[buf], segs, ns * sizeof(CopySegment), cudaMemcpyHostToDevice, stream_));
      segment_copy(d_segs_[buf], static_cast<int>(ns), stream_);
      DGPP_CUDA_OK(cudaEventRecord(done_[buf], stream_));
      ++issued;
    }
    // The blob into the arena slot, in page runs (the last page's tail
    // beyond the blob is not copied).
    uint8_t* blob = static_cast<uint8_t*>(arena_->slot_data_mutable(job.slot));
    for (int64_t first = 0; first < blob_pages_; first += per_chunk_) {
      const int64_t count = std::min<int64_t>(per_chunk_, blob_pages_ - first);
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));  // h_stage_[0] is free
      for (int64_t j = 0; j < count; ++j)
        if (!read_page(job.blob_page + first + j,
                       static_cast<uint8_t*>(h_stage_[0]) + static_cast<size_t>(j) * page_bytes_, why))
          return false;
      const size_t offset = static_cast<size_t>(first) * page_bytes_;
      const size_t bytes = std::min(static_cast<size_t>(count) * page_bytes_, blob_bytes_ - offset);
      DGPP_CUDA_OK(cudaMemcpyAsync(blob + offset, h_stage_[0], bytes, cudaMemcpyHostToDevice, stream_));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    }
    return true;
  }

  Model* model_ = nullptr;
  PrefixArena<Model>* arena_ = nullptr;
  Config cfg_;
  std::vector<CachePlane> planes_;
  std::vector<size_t> plane_off_;
  size_t block_bytes_ = 0;
  size_t page_bytes_ = 0;
  size_t blob_bytes_ = 0;
  int64_t blob_pages_ = 0;
  int64_t pages_ = 0;
  int64_t per_chunk_ = 0;
  int fd_ = -1;
  bool direct_ = false;
  int device_ = 0;
  cudaStream_t stream_ = nullptr;
  void* d_stage_[2] = {nullptr, nullptr};
  void* h_stage_[2] = {nullptr, nullptr};
  CopySegment* d_segs_[2] = {nullptr, nullptr};
  CopySegment* h_segs_[2] = {nullptr, nullptr};
  cudaEvent_t done_[2] = {nullptr, nullptr};
  std::vector<uint32_t> crc_;  // per page: the worker's alone
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Job> jobs_;
  std::vector<sched::SchedulerEngine::DiskCompletion> completions_;
  sched::SchedulerEngine::DiskEngineStats stats_;
  bool running_ = false;
  bool stop_ = false;
  std::thread worker_;
};

}  // namespace dgpp
