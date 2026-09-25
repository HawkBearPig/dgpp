#pragma once
// The prefix cache's NVMe cold tier (issue #26): a preallocated slab file
// per rank holding evicted entries' state — the snapshot blob and the
// cache blocks the entry's metadata pins — a pump that moves that state
// between the device and pinned staging on the MODEL stream, and a file
// worker that moves it between pinned staging and the slab.
//
// The slab is `pages` records of `page_bytes`, one cache block's planes
// (engine/cache_planes.hpp) rounded up to 4 KiB; a blob spans blob_pages
// contiguous pages. Which entry sits where is the scheduler's DiskCache
// (sched/disk_cache.hpp): identical on every rank, while this file's
// bytes are this rank's shard. The tier runs the ops it is handed:
//
//   spill   per tick, the pump gathers the next chunk of blocks (every
//           plane of each block into one contiguous record) into device
//           staging and copies it to a pinned buffer; the worker writes
//           the records to their pages (direct I/O — no page cache on the
//           GB10's unified memory) once the copy has landed. The blob
//           follows in page runs.
//   restore the worker reads pages into a pinned buffer; the pump copies
//           the buffer to the device and scatters each record's planes
//           into the fresh block the restore acquired; the blob into the
//           arena slot.
//
// Every byte of device work is enqueued by the engine thread on the model
// stream at a tick's top, in bounded slices (two staging chunks per
// tick), exactly as the prefix arena's own copies interleave with the
// decode replays; the worker never calls into CUDA. A second stream
// beside the fabric's device-side spin loops (the graph replays' stage
// gate and collectives) stalled them for their whole timeout (2026-09-25:
// the spill took 30 s on two ranks and those ranks died), which is why
// nothing here runs concurrently with the model stream.
//
// An op's outcome (ok, or a short read/write, a device error, a checksum
// mismatch on a page's CRC-32C) is reported through poll() once every
// slice of its device work has completed, so a released restore never
// has a scatter in flight; rank 0 journals every rank's verdict as one
// commit, so a failure on any rank is the same cold miss everywhere.
//
// Entries are the process's: the slab is recreated at every start (no
// reuse across restarts — a later version may bind records to the
// checkpoint and the configuration digest).
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
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
  static constexpr int kBuffers = 2;

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
    // Every CUDA object now, before the world's collectives (the fabric's
    // rule: nothing creates or frees one between them).
    const size_t max_segments = static_cast<size_t>(per_chunk_) * std::max<size_t>(1, planes_.size());
    for (int b = 0; b < kBuffers; ++b) {
      Buffer& buf = buffers_[b];
      DGPP_CUDA_OK(cudaMalloc(&buf.device, cfg_.chunk_bytes));
      DGPP_CUDA_OK(cudaHostAlloc(&buf.pinned, cfg_.chunk_bytes, cudaHostAllocDefault));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&buf.d_segs), max_segments * sizeof(CopySegment)));
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&buf.h_segs), max_segments * sizeof(CopySegment),
                                 cudaHostAllocDefault));
      DGPP_CUDA_OK(cudaEventCreateWithFlags(&buf.done, cudaEventDisableTiming));
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
    // Device work still in flight for a job settles before the buffers go.
    for (Buffer& buf : buffers_)
      if (buf.done) cudaEventSynchronize(buf.done);
    for (Buffer& buf : buffers_) {
      if (buf.done) cudaEventDestroy(buf.done);
      if (buf.device) cudaFree(buf.device);
      if (buf.pinned) cudaFreeHost(buf.pinned);
      if (buf.d_segs) cudaFree(buf.d_segs);
      if (buf.h_segs) cudaFreeHost(buf.h_segs);
    }
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
    jobs_.push_back(std::move(job));
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
    jobs_.push_back(std::move(job));
    return out;
  }

  // The per-tick pump (the engine thread): finished device slices go to
  // the worker (spill) or free their buffer (restore); the next slices of
  // the running job are enqueued on the model stream; a finished job
  // reports.
  void pump() {
    if (!active_ && !jobs_.empty()) start_job();
    if (!active_) return;
    Job& job = *active_;
    // Slices whose device work landed (kDevice is the pump's own state to
    // set and clear, so the read needs no lock).
    for (Buffer& buf : buffers_) {
      if (buf.state != Buffer::kDevice) continue;
      const cudaError_t q = cudaEventQuery(buf.done);
      if (q == cudaErrorNotReady) continue;
      if (q != cudaSuccess) {
        (void)cudaGetLastError();
        job.failed = true;
        job.why = "the model stream failed on a slice";
      }
      if (job.restore || job.failed) {
        release_buffer(buf);
      } else {
        std::lock_guard<std::mutex> lock(mutex_);
        buf.state = Buffer::kWorker;
        cv_.notify_one();
      }
    }
    // Slices the worker finished (a restore's reads, a spill's writes).
    std::vector<Buffer*> ready;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!job.why.empty() && !job.failed) job.failed = true;
      for (Buffer& buf : buffers_)
        if (buf.state == Buffer::kReady) ready.push_back(&buf);
    }
    for (Buffer* buf : ready) {
      if (job.restore && !job.failed) {
        issue_restore_slice(job, *buf);
      } else {
        release_buffer(*buf);
      }
    }
    // The next slices: a spill gathers into free buffers; a restore hands
    // free buffers to the worker to read into.
    if (!job.failed) {
      for (Buffer& buf : buffers_) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (buf.state != Buffer::kFree || job.next_slice >= job.slices) continue;
        }
        if (job.restore) {
          std::lock_guard<std::mutex> lock(mutex_);
          buf.slice = job.next_slice++;
          buf.state = Buffer::kWorker;
          cv_.notify_one();
        } else {
          issue_spill_slice(job, buf, job.next_slice++);
        }
      }
    }
    // Done: every slice issued and every buffer back, or failed with the
    // buffers quiet (no device work in flight, no worker on them).
    bool quiet = true;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const Buffer& buf : buffers_) quiet = quiet && buf.state == Buffer::kFree;
    }
    if (!quiet) return;
    if (!job.failed && job.next_slice < job.slices) return;
    finish_job();
  }

  std::vector<sched::SchedulerEngine::DiskCompletion> poll() {
    std::vector<sched::SchedulerEngine::DiskCompletion> out;
    out.swap(completions_);
    return out;
  }

  sched::SchedulerEngine::DiskEngineStats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    sched::SchedulerEngine::DiskEngineStats s = stats_;
    s.pending = static_cast<int64_t>(jobs_.size()) + (active_ ? 1 : 0);
    return s;
  }

 private:
  struct Job {
    uint64_t op = 0;
    bool restore = false;
    int slot = -1;
    int64_t blob_page = -1;
    std::vector<std::pair<int32_t, int64_t>> blocks;  // (physical block, page)
    int64_t block_slices = 0;  // ceil(blocks / per_chunk)
    int64_t slices = 0;        // block slices + blob slices
    int64_t next_slice = 0;
    bool failed = false;
    std::string why;
    std::chrono::steady_clock::time_point started;
  };
  // A staging buffer's state machine. Spill: kFree -> kDevice (gather +
  // D2H on the model stream) -> kWorker (writing) -> kFree. Restore: kFree
  // -> kWorker (reading) -> kReady -> kDevice (H2D + scatter) -> kFree.
  struct Buffer {
    enum State { kFree, kDevice, kWorker, kReady };
    State state = kFree;
    int64_t slice = -1;
    void* device = nullptr;
    void* pinned = nullptr;
    CopySegment* d_segs = nullptr;
    CopySegment* h_segs = nullptr;
    cudaEvent_t done = nullptr;
  };

  // A slice's records: block slices first, then the blob's page runs.
  struct Slice {
    bool blob = false;
    int64_t first = 0;   // first block index, or first blob page offset
    int64_t count = 0;   // records in the slice
  };
  Slice slice_of(const Job& job, int64_t s) const {
    Slice out;
    if (s < job.block_slices) {
      out.first = s * per_chunk_;
      out.count = std::min<int64_t>(per_chunk_, static_cast<int64_t>(job.blocks.size()) - out.first);
    } else {
      out.blob = true;
      out.first = (s - job.block_slices) * per_chunk_;
      out.count = std::min<int64_t>(per_chunk_, blob_pages_ - out.first);
    }
    return out;
  }

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

  void start_job() {
    active_ = std::move(jobs_.front());
    jobs_.pop_front();
    Job& job = *active_;
    job.block_slices = (static_cast<int64_t>(job.blocks.size()) + per_chunk_ - 1) / per_chunk_;
    job.slices = job.block_slices + (blob_pages_ + per_chunk_ - 1) / per_chunk_;
    job.next_slice = 0;
    job.started = std::chrono::steady_clock::now();
  }

  void finish_job() {
    Job& job = *active_;
    const bool ok = !job.failed;
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - job.started).count();
    if (!ok)
      DGPP_LOG_ERROR("rank {}: NVMe cache {} op {} failed: {}", cfg_.rank, job.restore ? "restore" : "spill", job.op,
                     job.why);
    else
      DGPP_LOG_INFO("rank {}: NVMe cache {} op {} done in {:.1f} ms ({} blocks, {:.1f} MiB)", cfg_.rank,
                    job.restore ? "restore" : "spill", job.op, ms, job.blocks.size(),
                    static_cast<double>((job.blocks.size() + static_cast<size_t>(blob_pages_)) * page_bytes_) /
                        (1024.0 * 1024.0));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (job.restore) {
        ++stats_.restores;
        stats_.restore_ms += ms;
      } else {
        ++stats_.spills;
        stats_.spill_ms += ms;
      }
      if (!ok) ++stats_.failures;
    }
    completions_.push_back({job.op, ok});
    active_.reset();
  }

  void release_buffer(Buffer& buf) {
    std::lock_guard<std::mutex> lock(mutex_);
    buf.state = Buffer::kFree;
    buf.slice = -1;
  }

  // A spill slice: the blocks' planes gathered into the device buffer as
  // contiguous records (or the blob's run copied), then to the pinned
  // buffer; the worker writes it once the event lands.
  void issue_spill_slice(Job& job, Buffer& buf, int64_t s) {
    const Slice sl = slice_of(job, s);
    cudaStream_t stream = model_->stream();
    if (sl.blob) {
      const uint8_t* blob = static_cast<const uint8_t*>(arena_->slot_data(job.slot));
      const size_t offset = static_cast<size_t>(sl.first) * page_bytes_;
      const size_t bytes = std::min(static_cast<size_t>(sl.count) * page_bytes_, blob_bytes_ - offset);
      DGPP_CUDA_OK(cudaMemcpyAsync(buf.pinned, blob + offset, bytes, cudaMemcpyDeviceToHost, stream));
    } else {
      size_t ns = 0;
      for (int64_t j = 0; j < sl.count; ++j) {
        const int32_t block = job.blocks[static_cast<size_t>(sl.first + j)].first;
        uint8_t* record = static_cast<uint8_t*>(buf.device) + static_cast<size_t>(j) * page_bytes_;
        for (size_t p = 0; p < planes_.size(); ++p) {
          buf.h_segs[ns].src = planes_[p].base + static_cast<size_t>(block) * planes_[p].block_bytes;
          buf.h_segs[ns].dst = record + plane_off_[p];
          buf.h_segs[ns].bytes = planes_[p].block_bytes;
          ++ns;
        }
      }
      DGPP_CUDA_OK(cudaMemcpyAsync(buf.d_segs, buf.h_segs, ns * sizeof(CopySegment), cudaMemcpyHostToDevice, stream));
      segment_copy(buf.d_segs, static_cast<int>(ns), stream);
      DGPP_CUDA_OK(cudaMemcpyAsync(buf.pinned, buf.device, static_cast<size_t>(sl.count) * page_bytes_,
                                   cudaMemcpyDeviceToHost, stream));
    }
    DGPP_CUDA_OK(cudaEventRecord(buf.done, stream));
    std::lock_guard<std::mutex> lock(mutex_);
    buf.slice = s;
    buf.state = Buffer::kDevice;
  }

  // A restore slice the worker read: to the device, then each record's
  // planes into its block (or the blob's run into the arena slot).
  void issue_restore_slice(Job& job, Buffer& buf) {
    const Slice sl = slice_of(job, buf.slice);
    cudaStream_t stream = model_->stream();
    if (sl.blob) {
      uint8_t* blob = static_cast<uint8_t*>(arena_->slot_data_mutable(job.slot));
      const size_t offset = static_cast<size_t>(sl.first) * page_bytes_;
      const size_t bytes = std::min(static_cast<size_t>(sl.count) * page_bytes_, blob_bytes_ - offset);
      DGPP_CUDA_OK(cudaMemcpyAsync(blob + offset, buf.pinned, bytes, cudaMemcpyHostToDevice, stream));
    } else {
      DGPP_CUDA_OK(cudaMemcpyAsync(buf.device, buf.pinned, static_cast<size_t>(sl.count) * page_bytes_,
                                   cudaMemcpyHostToDevice, stream));
      size_t ns = 0;
      for (int64_t j = 0; j < sl.count; ++j) {
        const int32_t block = job.blocks[static_cast<size_t>(sl.first + j)].first;
        const uint8_t* record = static_cast<const uint8_t*>(buf.device) + static_cast<size_t>(j) * page_bytes_;
        for (size_t p = 0; p < planes_.size(); ++p) {
          buf.h_segs[ns].src = record + plane_off_[p];
          buf.h_segs[ns].dst = planes_[p].base + static_cast<size_t>(block) * planes_[p].block_bytes;
          buf.h_segs[ns].bytes = planes_[p].block_bytes;
          ++ns;
        }
      }
      DGPP_CUDA_OK(cudaMemcpyAsync(buf.d_segs, buf.h_segs, ns * sizeof(CopySegment), cudaMemcpyHostToDevice, stream));
      segment_copy(buf.d_segs, static_cast<int>(ns), stream);
    }
    DGPP_CUDA_OK(cudaEventRecord(buf.done, stream));
    std::lock_guard<std::mutex> lock(mutex_);
    buf.state = Buffer::kDevice;
  }

  // ---- the file worker: pinned staging <-> the slab, nothing else -------
  void worker() {
    for (;;) {
      Buffer* buf = nullptr;
      bool restore = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
          if (stop_) return true;
          for (Buffer& b : buffers_)
            if (b.state == Buffer::kWorker) return true;
          return false;
        });
        if (stop_) return;
        for (Buffer& b : buffers_)
          if (b.state == Buffer::kWorker && buf == nullptr) buf = &b;
        restore = active_ && active_->restore;
      }
      // The active job is the engine thread's; the worker reads only its
      // slice geometry, which does not change while a buffer is its.
      const Job& job = *active_;
      const Slice sl = slice_of(job, buf->slice);
      std::string why;
      bool ok = true;
      for (int64_t j = 0; j < sl.count && ok; ++j) {
        const int64_t page = sl.blob ? job.blob_page + sl.first + j : job.blocks[static_cast<size_t>(sl.first + j)].second;
        uint8_t* record = static_cast<uint8_t*>(buf->pinned) + static_cast<size_t>(j) * page_bytes_;
        ok = restore ? read_page(page, record, &why) : write_page(page, record, &why);
      }
      std::lock_guard<std::mutex> lock(mutex_);
      if (!ok && active_->why.empty()) active_->why = why;
      // A finished read is the pump's to scatter; a finished write frees
      // the buffer; a failure frees it either way.
      buf->state = ok && restore ? Buffer::kReady : Buffer::kFree;
      if (!ok || !restore) buf->slice = -1;
    }
  }

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
  std::array<Buffer, kBuffers> buffers_;
  std::vector<uint32_t> crc_;  // per page: the worker's alone
  // The jobs (the engine thread's), the active one and the completions:
  // the worker touches only the buffers' states and the job's failure
  // note, under the mutex.
  std::deque<Job> jobs_;
  std::optional<Job> active_;
  std::vector<sched::SchedulerEngine::DiskCompletion> completions_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  sched::SchedulerEngine::DiskEngineStats stats_;
  bool stop_ = false;
  std::thread worker_;
};

}  // namespace dgpp
