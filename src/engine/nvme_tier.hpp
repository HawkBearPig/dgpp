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
// The slab outlives the process (nvme_cache.persist): the token ids go to
// the slab beside the blob, the page CRCs to rank<N>.crc and the
// scheduler's index to `index`, all under a directory named by the
// deployment's binding (checkpoint, settings, slab geometry). A start
// that finds the three files for its binding offers the index as a
// candidate; the world adopts it only when every rank's candidate is the
// same bytes (the warm handshake), else the slab is recreated. The index
// file also carries, per entry, this rank's digest of the entry's page
// CRCs when its spill committed: at open every entry is checked against
// the CRC table as it stands and its ids run is read, and an entry whose
// pages were reused by a later run that crashed before its index landed,
// or whose ids are unreadable, is pruned from the candidate — a pruned
// candidate differs from the other ranks' and the world starts fresh,
// never with an entry whose checksums pass over someone else's bytes.
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/crc32c.hpp"
#include "common/cache_storage.hpp"
#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "engine/cache_planes.hpp"
#include "engine/pool_exhausted.hpp"
#include "engine/prefix_arena.hpp"
#include "kernels/segment_copy.hpp"
#include "sched/scheduler.hpp"
#include "sched/nvme_index.hpp"

namespace dgpp {

// The host and device bytes the tier allocates for a chunk size (the
// memory plan): two device and two pinned staging buffers plus the
// segment tables.
inline size_t nvme_tier_staging_bytes(size_t chunk_bytes) { return 4 * chunk_bytes + (size_t{8} << 20); }

template <class Model>
class NvmeTier {
 public:
  using Config = sched::SchedulerEngine::DiskEnable;
  static constexpr size_t kAlign = 4096;
  static constexpr int kBuffers = 2;
  static size_t staging_bytes(size_t chunk_bytes) { return nvme_tier_staging_bytes(chunk_bytes); }

  NvmeTier(Model* model, PrefixArena<Model>* arena, const Config& cfg)
      : model_(model), arena_(arena), cfg_(cfg) {
    if (model_ == nullptr || arena_ == nullptr) throw std::invalid_argument("NvmeTier: null model or arena");
    if (arena_->slots() <= 0) throw std::invalid_argument("NvmeTier: the prefix arena has no slots");
    if (cfg_.chunk_bytes < kAlign || cfg_.chunk_bytes % kAlign != 0)
      throw std::invalid_argument("NvmeTier: the chunk must be a positive 4 KiB multiple");
    if (cfg_.require_existing && (!cfg_.persist || cfg_.binding.empty()))
      throw std::invalid_argument("NvmeTier: explicit reuse requires persistence and a binding");
    const auto parent = std::filesystem::path(cfg_.slab_path).parent_path();
    const auto storage = inspect_cache_directory(parent.empty() ? "." : parent.string());
    if (const auto error = check_cache_storage(storage, cfg_.required_mount, cfg_.allow_buffered); !error.empty())
      throw std::runtime_error("NvmeTier: " + error);
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
    // The adoption's read buffer, before open_slab() reads the candidate's
    // ids runs with it (direct I/O needs the alignment).
    void* aligned_page = nullptr;
    if (::posix_memalign(&aligned_page, kAlign, page_bytes_) != 0 || aligned_page == nullptr)
      throw std::runtime_error("NvmeTier: cannot allocate an aligned page");
    // open_slab() can refuse a duplicate before this object's destructor runs.
    aligned_page_.reset(aligned_page);
    try {
      open_slab();
    } catch (...) {
      // A rejected existing cache can fail after taking the slab lock.
      if (fd_ >= 0) ::close(fd_);
      if (crc_fd_ >= 0) ::close(crc_fd_);
      throw;
    }
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
    worker_ = std::thread([this] { worker(); });
    DGPP_LOG_INFO(
        "rank {}: NVMe cache open — {} ({:.2f} GiB): {} pages of {} bytes ({} bytes of cache block in {} planes), "
        "{} pages per snapshot blob ({:.1f} MiB), {} MiB staging chunks, {} I/O",
        cfg_.rank, cfg_.slab_path, static_cast<double>(cfg_.capacity_bytes) / (1024.0 * 1024.0 * 1024.0), pages_,
        page_bytes_, block_bytes_, planes_.size(), blob_pages_, static_cast<double>(blob_bytes_) / (1024.0 * 1024.0),
        cfg_.chunk_bytes >> 20, direct_ ? "direct" : "buffered");
    if (cfg_.persist)
      DGPP_LOG_INFO("rank {}: NVMe cache: the slab outlives this process ({}, binding {}); {}", cfg_.rank,
                    index_path_, cfg_.binding,
                    candidate_.empty() ? "no previous index for this binding" : "a previous index is the candidate");
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
    if (crc_fd_ >= 0) ::close(crc_fd_);
    if (fd_ >= 0) {
      ::close(fd_);
      // A slab that outlives the process stays, with its index and CRC
      // table beside it; otherwise nothing is left behind.
      if (!cfg_.persist) {
        ::unlink(cfg_.slab_path.c_str());
        ::unlink(crc_path_.c_str());
        ::unlink(index_path_.c_str());
      }
    }
  }
  NvmeTier(const NvmeTier&) = delete;
  NvmeTier& operator=(const NvmeTier&) = delete;

  int64_t pages() const { return pages_; }
  size_t page_bytes() const { return page_bytes_; }
  int64_t blob_pages() const { return blob_pages_; }
  size_t block_bytes() const { return block_bytes_; }

  // ---- across restarts ------------------------------------------------------
  // The index a previous run left for this binding (empty: none), and its
  // digest for the world's agreement ("0": none).
  const std::vector<uint8_t>& candidate_index() const { return candidate_; }
  std::string candidate_digest() const {
    if (candidate_.empty()) return "0";
    uint64_t h = 0xcbf29ce484222325ull;
    for (const uint8_t b : candidate_) {
      h ^= b;
      h *= 0x100000001b3ull;
    }
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
  }
  // The world's verdict: keep the candidate for the scheduler to adopt, or
  // start fresh (the slab recreated, the files gone). Before any op.
  void adopt(bool keep) {
    if (keep || candidate_.empty()) return;
    candidate_.clear();
    adopted_digests_.clear();
    adopted_ids_.clear();
    recreate_slab();
    DGPP_LOG_INFO("rank {}: NVMe cache: the world did not agree on the previous index; the slab starts fresh",
                  cfg_.rank);
  }
  // `count` pages from `page` into `dst` (count x page bytes), each checked
  // against its CRC: the adoption's reads of the token ids, on the engine
  // thread with the worker idle. False on any failure.
  bool read_pages(int64_t page, int64_t count, void* dst) {
    if (page < 0 || count <= 0 || page + count > pages_) return false;
    // The ids the open already read and checked (an adopted entry's run).
    if (const auto it = adopted_ids_.find(page); it != adopted_ids_.end()) {
      const size_t bytes = static_cast<size_t>(count) * page_bytes_;
      std::memset(dst, 0, bytes);
      std::memcpy(dst, it->second.data(), std::min(bytes, it->second.size()));
      return true;
    }
    for (int64_t i = 0; i < count; ++i) {
      std::string why;
      if (!read_page(page + i, aligned_page_.get(), &why)) {
        DGPP_LOG_WARN("rank {}: NVMe cache: an adopted entry's ids run is unreadable ({}); the entry is dropped",
                      cfg_.rank, why);
        return false;
      }
      std::memcpy(static_cast<uint8_t*>(dst) + static_cast<size_t>(i) * page_bytes_, aligned_page_.get(), page_bytes_);
    }
    return true;
  }
  // The committed index with its entries' content digests, for the worker
  // to write beside the slab (the newest bytes win; a write in progress is
  // followed by another).
  void persist_index(std::vector<uint8_t> bytes, std::vector<uint64_t> digests) {
    if (!cfg_.persist) return;
    std::lock_guard<std::mutex> lock(mutex_);
    pending_index_ = std::move(bytes);
    pending_digests_ = std::move(digests);
    has_pending_index_ = true;
    cv_.notify_one();
  }
  // The adopted index's content digests, in its order.
  const std::vector<uint64_t>& adopted_digests() const { return adopted_digests_; }
  // This rank's digest of the pages' CRCs as the table stands: the index
  // binds an entry to the bytes its spill wrote with it. Distinct pages
  // are distinct elements of the table, so the worker's writes of other
  // pages race nothing here.
  uint64_t pages_digest(const std::vector<int64_t>& pages, uint64_t salt) const {
    uint64_t h = 0xcbf29ce484222325ull;
    for (int i = 0; i < 8; ++i) {
      h ^= (salt >> (8 * i)) & 0xffu;
      h *= 0x100000001b3ull;
    }
    for (const int64_t p : pages) {
      const uint32_t crc = p >= 0 && p < pages_ ? crc_[static_cast<size_t>(p)] : 0;
      for (int i = 0; i < 4; ++i) {
        h ^= (crc >> (8 * i)) & 0xffu;
        h *= 0x100000001b3ull;
      }
    }
    return h;
  }

  sched::SchedulerEngine::DiskInfo info() const {
    sched::SchedulerEngine::DiskInfo i;
    i.enabled = true;
    i.pages = pages_;
    i.page_bytes = page_bytes_;
    i.blob_pages = blob_pages_;
    i.min_tokens = cfg_.min_tokens;
    i.max_ops = cfg_.max_ops;
    i.persist = cfg_.persist;
    i.op_timeout_s = cfg_.op_timeout_s;
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
    // The token ids' run (the scheduler's index names it; a spill without
    // ids — a test driving the tier alone — has no run and leaves the slab
    // nothing to describe the entry by).
    const int64_t ids_bytes = static_cast<int64_t>(spill.ids.size()) * static_cast<int64_t>(sizeof(int64_t));
    const int64_t ids_pages = (ids_bytes + static_cast<int64_t>(page_bytes_) - 1) / static_cast<int64_t>(page_bytes_);
    if (ids_pages != spill.ids_pages || (ids_pages > 0 && (spill.ids_page < 0 || spill.ids_page + ids_pages > pages_)))
      throw std::invalid_argument("NvmeTier: the spill's ids run does not cover its token ids");
    Job job;
    job.op = spill.op;
    job.restore = false;
    job.slot = spill.slot;
    job.blob_page = spill.blob_page;
    job.ids = spill.ids;
    job.ids_page = spill.ids_page;
    job.ids_pages = ids_pages;
    for (size_t i = 0; i < spill.pages.size(); ++i)
      if (spill.pages[i] >= 0) job.blocks.emplace_back(meta.full_blocks[i], spill.pages[i]);
    if (spill.partial_page >= 0) job.blocks.emplace_back(meta.partial_block, spill.partial_page);
    check_pages(job);
    jobs_.push_back(std::move(job));
  }

  sched::SchedulerEngine::DiskBlocks restore_begin(const sched::SchedulerEngine::DiskRestore& restore) {
    if (restore.position <= 0) throw std::invalid_argument("NvmeTier: restore at position 0");
    // The file worker is one thread: while it is inside the I/O of an op
    // the timeout abandoned, nothing can be read from the slab. A restore
    // asked for now is refused the way a full pool refuses one (the
    // scheduler keeps the entry and admits the request cold at once)
    // instead of waiting out another timeout and dropping the entry.
    if (worker_stuck())
      throw CachePoolExhausted("NvmeTier: the file worker is stuck in an earlier op's I/O; the restore is refused");
    const int64_t bt = model_->kv_block_tokens();
    const int64_t n_full = bt > 0 ? restore.position / bt : 0;
    if (static_cast<int64_t>(restore.pages.size()) != n_full)
      throw std::invalid_argument("NvmeTier: the restore's pages do not cover the position's blocks");
    const bool partial = bt > 0 && restore.position % bt != 0;
    if (partial != (restore.partial_page >= 0))
      throw std::invalid_argument("NvmeTier: the restore's partial page does not match the position");
    // The pages first: a plan outside the slab is the scheduler's error and
    // must not leave pool blocks taken.
    check_pages(restore.blob_page, restore.pages, restore.partial_page);
    typename Model::SessionSnapshotMeta meta;
    meta.position = restore.position;
    meta.mtp_position = restore.mtp_position;
    std::vector<int32_t> taken;
    const auto take = [&]() -> int32_t {
      const int32_t b = model_->cache_acquire_block();
      if (b < 0) {
        model_->cache_release_blocks(taken.data(), static_cast<int64_t>(taken.size()));
        throw CachePoolExhausted("NvmeTier: the pool has no free block for the restore");
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
    // The scheduler adopted the index before the first tick; the ids kept
    // for it are not needed again.
    if (!adopted_ids_.empty()) adopted_ids_.clear();
    if (!active_ && !jobs_.empty()) start_job();
    if (!active_) return;
    Job& job = *active_;
    // An op that makes no progress (a stuck write, a device that never
    // signals): the scheduler holds the requests waiting on it and its
    // deadlock diagnosis stands aside while ops pend, so a stall is said
    // out loud — 30 s without a slice issued or returned, then every
    // minute. Age alone is not the signal: two slices move per tick, and
    // a long prefill's ticks are chunk-long, so a 1.75 GiB spill beside a
    // 31K-token prefill legitimately runs for half a minute.
    const auto now = std::chrono::steady_clock::now();
    const double stall_s = std::chrono::duration<double>(now - job.progressed).count();
    if (stall_s >= job.next_stall_note_s) {
      DGPP_LOG_WARN("rank {}: NVMe cache {} op {} has made no progress for {:.0f} s ({} of {} slices issued, "
                    "{:.0f} s old)",
                    cfg_.rank, job.restore ? "restore" : "spill", job.op, stall_s, job.next_slice, job.slices,
                    std::chrono::duration<double>(now - job.started).count());
      job.next_stall_note_s = stall_s + 60.0;
    }
    // The bound: an op without progress for op_timeout_s fails, and the
    // buffers a stuck read or write holds are abandoned to the worker —
    // it frees them when the call returns (they belong to no job by then)
    // — so the requests waiting on the op admit cold instead of never.
    if (cfg_.op_timeout_s > 0 && stall_s >= cfg_.op_timeout_s && !job.failed) {
      std::lock_guard<std::mutex> lock(mutex_);
      job.failed = true;
      job.abandoned = true;
      if (job.why.empty())
        job.why = "no progress for " + std::to_string(static_cast<int64_t>(stall_s)) + " s (nvme_cache.op_timeout_s " +
                  std::to_string(cfg_.op_timeout_s) + ")";
    }
    bool progressed = false;
    // Slices whose device work landed (kDevice is the pump's own state to
    // set and clear, so the read needs no lock).
    for (Buffer& buf : buffers_) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buf.state != Buffer::kDevice) continue;
      }
      const cudaError_t q = cudaEventQuery(buf.done);
      if (q == cudaErrorNotReady) continue;
      progressed = true;
      if (q != cudaSuccess) {
        (void)cudaGetLastError();
        // The note is shared with the worker (it writes its own under the
        // mutex when a read or write fails): take the lock for ours too.
        std::lock_guard<std::mutex> lock(mutex_);
        job.failed = true;
        if (job.why.empty()) job.why = "the model stream failed on a slice";
      }
      if (job.restore || job.failed) {
        release_buffer(buf);
      } else {
        std::lock_guard<std::mutex> lock(mutex_);
        buf.state = Buffer::kWorker;
        cv_.notify_one();
      }
    }
    // Slices the worker finished (a restore's reads, a spill's writes). A
    // read an abandoned op's worker finished late belongs to no job.
    std::vector<Buffer*> ready;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!job.why.empty() && !job.failed) job.failed = true;
      for (Buffer& buf : buffers_)
        if (buf.state == Buffer::kReady) ready.push_back(&buf);
    }
    for (Buffer* buf : ready) {
      progressed = true;
      if (job.restore && !job.failed && buf->job_seq == job.seq) {
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
        progressed = true;
        if (job.restore) {
          std::lock_guard<std::mutex> lock(mutex_);
          buf.slice = job.next_slice++;
          buf.job_seq = job.seq;
          buf.state = Buffer::kWorker;
          cv_.notify_one();
        } else {
          issue_spill_slice(job, buf, job.next_slice++);
        }
      }
    }
    if (progressed) {
      job.progressed = now;
      job.next_stall_note_s = 30.0;
    }
    // Done: every slice issued and every buffer back, or failed with the
    // buffers quiet (no device work in flight, no worker on them) — an
    // abandoned op's buffers with the worker count as quiet: the job is
    // over for them, the worker frees them when its call returns.
    bool quiet = true;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (const Buffer& buf : buffers_)
        quiet = quiet && (buf.state == Buffer::kFree ||
                          (buf.state == Buffer::kWorker && (job.abandoned || buf.job_seq != job.seq)));
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
    std::vector<int64_t> ids;  // spill: the entry's token ids, written behind the blob
    int64_t ids_page = -1;
    int64_t ids_pages = 0;
    int64_t block_slices = 0;  // ceil(blocks / per_chunk)
    int64_t blob_slices = 0;
    int64_t slices = 0;        // block slices + blob slices (+ ids slices on a spill)
    int64_t next_slice = 0;
    bool failed = false;
    bool abandoned = false;  // failed by the timeout: its buffers with the worker are the worker's to free
    uint64_t seq = 0;        // the job's number: a buffer carries the one it serves
    std::string why;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point progressed;  // the last slice issued or returned
    double next_stall_note_s = 30.0;  // the next stall length the pump reports at
  };
  // A staging buffer's state machine. Spill: kFree -> kDevice (gather +
  // D2H on the model stream) -> kWorker (writing) -> kFree. Restore: kFree
  // -> kWorker (reading) -> kReady -> kDevice (H2D + scatter) -> kFree.
  struct Buffer {
    enum State { kFree, kDevice, kWorker, kReady };
    State state = kFree;
    int64_t slice = -1;
    uint64_t job_seq = 0;  // the job the slice belongs to (the worker returns a stale one unused)
    void* device = nullptr;
    void* pinned = nullptr;
    CopySegment* d_segs = nullptr;
    CopySegment* h_segs = nullptr;
    cudaEvent_t done = nullptr;
  };

  // A slice's records: block slices first, then the blob's page runs,
  // then (a spill) the token ids' pages, host bytes with no device work.
  struct Slice {
    enum Kind { kBlock, kBlob, kIds };
    Kind kind = kBlock;
    int64_t first = 0;   // first block index, or first page offset in the run
    int64_t count = 0;   // records in the slice
  };
  Slice slice_of(const Job& job, int64_t s) const {
    Slice out;
    if (s < job.block_slices) {
      out.kind = Slice::kBlock;
      out.first = s * per_chunk_;
      out.count = std::min<int64_t>(per_chunk_, static_cast<int64_t>(job.blocks.size()) - out.first);
    } else if (s < job.block_slices + job.blob_slices) {
      out.kind = Slice::kBlob;
      out.first = (s - job.block_slices) * per_chunk_;
      out.count = std::min<int64_t>(per_chunk_, blob_pages_ - out.first);
    } else {
      out.kind = Slice::kIds;
      out.first = (s - job.block_slices - job.blob_slices) * per_chunk_;
      out.count = std::min<int64_t>(per_chunk_, job.ids_pages - out.first);
    }
    return out;
  }

  void check_pages(int64_t blob_page, const std::vector<int64_t>& pages, int64_t partial_page) const {
    const auto bad = [&](int64_t page) { return page < 0 || page >= pages_; };
    if (bad(blob_page) || blob_page + blob_pages_ > pages_)
      throw std::invalid_argument("NvmeTier: the blob's pages are outside the slab");
    for (const int64_t page : pages)
      if (page >= 0 && bad(page)) throw std::invalid_argument("NvmeTier: a block's page is outside the slab");
    if (partial_page >= 0 && bad(partial_page))
      throw std::invalid_argument("NvmeTier: the partial block's page is outside the slab");
  }
  void check_pages(const Job& job) const {
    std::vector<int64_t> pages;
    pages.reserve(job.blocks.size());
    for (const auto& [block, page] : job.blocks) {
      (void)block;
      pages.push_back(page);
    }
    check_pages(job.blob_page, pages, -1);
  }

  void open_slab() {
    namespace fs = std::filesystem;
    const fs::path slab(cfg_.slab_path);
    const fs::path dir = slab.parent_path();
    crc_path_ = (dir / ("rank" + std::to_string(cfg_.rank) + ".crc")).string();
    // This rank's: the scheduler's bytes are the world's, the digests
    // behind them this rank's, so ranks sharing a directory (one host, a
    // shared filesystem) never overwrite each other's.
    index_path_ = (dir / ("rank" + std::to_string(cfg_.rank) + ".index")).string();
    // Other ranks' slabs beside this one, and other bindings' directories
    // beside this one (another model or configuration served from this
    // node) — nobody's to reclaim now; name them and their size so the
    // operator can.
    {
      std::error_code ec;
      const std::string own = slab.filename().string();
      size_t stale_bytes = 0;
      std::string stale;
      for (const auto& entry : fs::directory_iterator(dir, ec)) {
        const std::string name = entry.path().filename().string();
        if (name == own || name.rfind("rank", 0) != 0 || entry.path().extension() != ".slab") continue;
        if (!entry.is_regular_file(ec)) continue;
        stale_bytes += static_cast<size_t>(entry.file_size(ec));
        stale += (stale.empty() ? "" : ", ") + name;
      }
      for (const auto& sibling : fs::directory_iterator(dir.parent_path(), ec)) {
        if (!sibling.is_directory(ec) || sibling.path() == dir) continue;
        size_t bytes = 0;
        for (const auto& entry : fs::directory_iterator(sibling.path(), ec))
          if (entry.is_regular_file(ec) && entry.path().extension() == ".slab") bytes += static_cast<size_t>(entry.file_size(ec));
        if (bytes == 0) continue;
        stale_bytes += bytes;
        stale += (stale.empty() ? "" : ", ") + sibling.path().filename().string() + "/";
      }
      if (!stale.empty())
        DGPP_LOG_WARN("rank {}: NVMe cache: {} ({:.1f} GiB) beside {} belong to no rank of this deployment — "
                      "another binding's slabs or a crash's leftovers; remove them to free the disk",
                      cfg_.rank, stale, static_cast<double>(stale_bytes) / (1024.0 * 1024.0 * 1024.0), own);
    }
    // A previous run's slab for this binding: the slab of the right size,
    // its CRC table of the right size and an index written for the same
    // binding — the index becomes the candidate; the slab opens as it is.
    bool adopt = false;
    if (cfg_.persist) {
      std::error_code ec;
      const uintmax_t slab_size = fs::is_regular_file(slab, ec) ? fs::file_size(slab, ec) : 0;
      const uintmax_t crc_size = fs::is_regular_file(crc_path_, ec) ? fs::file_size(crc_path_, ec) : 0;
      if (slab_size == static_cast<uintmax_t>(pages_) * page_bytes_ &&
          crc_size == static_cast<uintmax_t>(pages_) * sizeof(uint32_t) && fs::is_regular_file(index_path_, ec)) {
        const std::vector<uint8_t> file = read_file(index_path_);
        adopt = sched::parse_nvme_index(file, cfg_.binding, &candidate_, &adopted_digests_);
      }
    }
    if (!adopt && cfg_.require_existing)
      throw std::runtime_error("NvmeTier: explicitly selected cache is missing or its index, binding or size "
                               "does not match; existing cache left untouched: " + cfg_.slab_path);
    open_fd(adopt);
    if (adopt) {
      crc_.assign(static_cast<size_t>(pages_), 0);
      const int fd = ::open(crc_path_.c_str(), O_RDONLY | O_CLOEXEC);
      bool ok = fd >= 0;
      size_t done = 0;
      const size_t want = static_cast<size_t>(pages_) * sizeof(uint32_t);
      while (ok && done < want) {
        const ssize_t n = ::pread(fd, reinterpret_cast<uint8_t*>(crc_.data()) + done, want - done,
                                  static_cast<off_t>(done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) ok = false;
        else done += static_cast<size_t>(n);
      }
      if (fd >= 0) ::close(fd);
      if (!ok) {
        DGPP_LOG_WARN("rank {}: NVMe cache: the CRC table beside {} is unreadable; {}",
                      cfg_.rank, cfg_.slab_path, cfg_.require_existing ? "explicit reuse refused" : "the slab starts fresh");
        candidate_.clear();
        adopted_digests_.clear();
        recreate_slab();
      } else {
        (void)validate_candidate();  // a candidate that does not describe the slab recreates it
      }
    } else {
      fresh_files();
    }
    if (cfg_.persist) {
      crc_fd_ = ::open(crc_path_.c_str(), O_RDWR | O_CLOEXEC | (cfg_.require_existing ? 0 : O_CREAT), 0600);
      if (crc_fd_ < 0)
        throw std::runtime_error("NvmeTier: cannot open " + crc_path_ + ": " + std::strerror(errno));
    }
  }

  // The candidate against the slab as it stands: an entry stays only when
  // its pages' CRCs still fold to the digest its commit recorded and its
  // ids run reads back (checked, and kept for the scheduler's adoption);
  // the rest are pruned. False when nothing parses: the slab starts fresh.
  bool validate_candidate() {
    sched::DiskCache::Header header;
    std::vector<sched::DiskCache::Record> records;
    if (!sched::DiskCache::parse(candidate_, &header, &records) || records.size() != adopted_digests_.size() ||
        header.pages != static_cast<uint64_t>(pages_) || header.page_bytes != page_bytes_ ||
        header.blob_pages != static_cast<uint64_t>(blob_pages_) ||
        header.block_tokens != static_cast<uint64_t>(model_->kv_block_tokens())) {
      DGPP_LOG_WARN("rank {}: NVMe cache: the index beside {} does not describe this slab; {}",
                    cfg_.rank, cfg_.slab_path, cfg_.require_existing ? "explicit reuse refused" : "the slab starts fresh");
      candidate_.clear();
      adopted_digests_.clear();
      recreate_slab();
      return false;
    }
    std::vector<sched::DiskCache::Record> kept;
    std::vector<uint64_t> kept_digests;
    int stale = 0, unreadable = 0, invalid = 0;
    for (size_t i = 0; i < records.size(); ++i) {
      const sched::DiskCache::Record& rec = records[i];
      // The record's counts against the slab before any list is built from
      // them: a corrupted ids_pages must prune the entry, not allocate.
      bool ok = rec.position > 0 && rec.blob_page >= 0 && rec.blob_page < pages_ &&
                static_cast<int64_t>(rec.ids_pages) <= pages_ &&
                rec.blob_page + blob_pages_ + static_cast<int64_t>(rec.ids_pages) <= pages_ &&
                static_cast<int64_t>(rec.block_pages.size()) <= pages_ && rec.partial_page < pages_;
      std::vector<int64_t> pages;
      if (ok) pages = sched::DiskCache::record_pages(rec, blob_pages_);
      for (const int64_t p : pages) ok = ok && p >= 0 && p < pages_;
      if (!ok) ++invalid;
      if (ok && pages_digest(pages, static_cast<uint64_t>(rec.position)) != adopted_digests_[i]) {
        ok = false;
        ++stale;
      }
      std::vector<uint8_t> ids;
      if (ok) {
        const size_t want = static_cast<size_t>(rec.position) * sizeof(int64_t);
        for (uint32_t q = 0; q < rec.ids_pages && ok && ids.size() < want; ++q) {
          std::string why;
          ok = read_page(rec.blob_page + blob_pages_ + q, aligned_page_.get(), &why);
          if (ok) {
            const auto* bytes = static_cast<const uint8_t*>(aligned_page_.get());
            ids.insert(ids.end(), bytes, bytes + std::min(page_bytes_, want - ids.size()));
          }
        }
        if (ok && ids.size() < want) ok = false;
        if (!ok) ++unreadable;
      }
      if (!ok) continue;
      adopted_ids_[rec.blob_page + blob_pages_] = std::move(ids);
      kept.push_back(rec);
      kept_digests.push_back(adopted_digests_[i]);
    }
    if (kept.size() != records.size()) candidate_ = sched::DiskCache::write(header, kept);
    adopted_digests_ = std::move(kept_digests);
    DGPP_LOG_INFO("rank {}: NVMe cache: a previous run's index for this binding is the candidate — {} entries kept, "
                  "{} pruned (their pages hold other bytes now), {} unreadable, {} outside the slab",
                  cfg_.rank, kept.size(), stale, unreadable, invalid);
    return true;
  }

  static std::vector<uint8_t> read_file(const std::string& path) {
    std::vector<uint8_t> out;
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return out;
    uint8_t buf[65536];
    for (;;) {
      const ssize_t n = ::read(fd, buf, sizeof buf);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) break;
      out.insert(out.end(), buf, buf + n);
    }
    ::close(fd);
    return out;
  }

  // The slab's descriptor: direct I/O first (the page cache would grow
  // into the memory the model plans on); RAM-backed storage and filesystems
  // that refuse it require the explicit test allowance. A fresh slab is
  // truncated and its whole capacity claimed now (unwritten extents): a
  // full disk is a startup error, never a torn write later.
  void open_fd(bool keep_contents) {
    if (fd_ >= 0 && !keep_contents) {
      // A recreation on the descriptor already held: the contents go, the
      // lock stays — a duplicate deployment finds no gap to take it in.
      if (::ftruncate(fd_, 0) != 0)
        throw std::runtime_error("NvmeTier: cannot truncate " + cfg_.slab_path + ": " + std::strerror(errno));
      size_slab();
      return;
    }
    if (fd_ >= 0) ::close(fd_);
    const int flags = O_RDWR | O_CLOEXEC | (cfg_.require_existing ? 0 : O_CREAT);
    fd_ = ::open(cfg_.slab_path.c_str(), flags | O_DIRECT, 0600);
    direct_ = fd_ >= 0;
    if (fd_ < 0 && errno == EINVAL) fd_ = ::open(cfg_.slab_path.c_str(), flags, 0600);
    if (fd_ < 0)
      throw std::runtime_error("NvmeTier: cannot create " + cfg_.slab_path + ": " + std::strerror(errno));
    const auto storage = inspect_cache_storage_fd(fd_);
    if (const auto error = check_cache_storage(storage, cfg_.required_mount, cfg_.allow_buffered); !error.empty()) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("NvmeTier: " + error);
    }
    // Modern tmpfs can accept O_DIRECT while still keeping all bytes in RAM.
    if (storage.ram_backed) direct_ = false;
    // One world per slab: a second process on the same binding (a duplicate
    // deployment) would truncate and write under the first.
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
      const int err = errno;
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("NvmeTier: " + cfg_.slab_path + " is held by another process (" + std::strerror(err) +
                               "): one world per slab — stop the other deployment or give this one another "
                               "nvme_cache.path");
    }
    if (!direct_ && !cfg_.allow_buffered) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("NvmeTier: " + cfg_.slab_path +
                               " refuses direct I/O (the filesystem buffers it, and the page cache would grow into "
                               "the memory the model plans on): put nvme_cache.path on a local filesystem, or set "
                               "nvme_cache.allow_buffered");
    }
    if (keep_contents) return;
    // Opening with O_TRUNC would destroy another world's slab before the
    // advisory lock can refuse us. Clear it only after acquiring ownership.
    if (::ftruncate(fd_, 0) != 0) {
      const int err = errno;
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("NvmeTier: cannot truncate " + cfg_.slab_path + ": " +
                               std::strerror(err));
    }
    size_slab();
  }
  // The whole capacity claimed now (unwritten extents): a full disk is a
  // startup error, never a torn write later.
  void size_slab() {
    const off_t bytes = static_cast<off_t>(static_cast<size_t>(pages_) * page_bytes_);
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
  // A fresh slab's files: no index, a zeroed CRC table (sparse).
  void fresh_files() {
    crc_.assign(static_cast<size_t>(pages_), 0);
    ::unlink(index_path_.c_str());
    ::unlink((index_path_ + ".tmp").c_str());
    if (!cfg_.persist) {
      ::unlink(crc_path_.c_str());
      return;
    }
    const int fd = ::open(crc_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(static_cast<size_t>(pages_) * sizeof(uint32_t))) != 0)
      throw std::runtime_error("NvmeTier: cannot create " + crc_path_ + ": " + std::strerror(errno));
    ::close(fd);
  }
  void recreate_slab() {
    if (cfg_.require_existing)
      throw std::runtime_error("NvmeTier: explicitly selected cache failed validation or world agreement; "
                               "existing cache left untouched: " + cfg_.slab_path);
    open_fd(/*keep_contents=*/false);
    fresh_files();
  }

  // A buffer with the worker that belongs to no running job: the worker is
  // inside a stuck call for it (it frees a stale buffer the moment the call
  // returns), so no slice of any op can move until then.
  bool worker_stuck() const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const Buffer& buf : buffers_)
      if (buf.state == Buffer::kWorker && (!active_ || buf.job_seq != active_->seq)) return true;
    return false;
  }

  void start_job() {
    // The job is whole before the worker can see it: a buffer abandoned to
    // the worker by an earlier op keeps the worker reading `active_` under
    // the lock at any time, so the publication is under the lock too.
    Job job = std::move(jobs_.front());
    jobs_.pop_front();
    job.seq = ++job_seq_;
    job.block_slices = (static_cast<int64_t>(job.blocks.size()) + per_chunk_ - 1) / per_chunk_;
    job.blob_slices = (blob_pages_ + per_chunk_ - 1) / per_chunk_;
    const int64_t ids_slices = job.restore ? 0 : (job.ids_pages + per_chunk_ - 1) / per_chunk_;
    job.slices = job.block_slices + job.blob_slices + ids_slices;
    job.next_slice = 0;
    job.started = std::chrono::steady_clock::now();
    job.progressed = job.started;
    std::lock_guard<std::mutex> lock(mutex_);
    // A buffer with the worker on an abandoned op: the worker is inside
    // that call and nothing can move until it returns, which may be never
    // — the op fails now (a spill's entry stays in memory; a restore
    // queued before the stall is a cold miss) instead of waiting out the
    // timeout too.
    bool stuck = false;
    for (const Buffer& buf : buffers_) stuck = stuck || buf.state == Buffer::kWorker;
    if (stuck) {
      job.failed = true;
      job.abandoned = true;
      job.why = "the file worker is stuck in an earlier op's I/O";
    }
    active_ = std::move(job);
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
    std::lock_guard<std::mutex> lock(mutex_);
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
    if (sl.kind == Slice::kIds) {
      // Host bytes: into the pinned buffer and to the worker, no event.
      const uint8_t* src = reinterpret_cast<const uint8_t*>(job.ids.data());
      const size_t total = job.ids.size() * sizeof(int64_t);
      const size_t offset = static_cast<size_t>(sl.first) * page_bytes_;
      const size_t bytes = std::min(static_cast<size_t>(sl.count) * page_bytes_, total - offset);
      std::memcpy(buf.pinned, src + offset, bytes);
      std::lock_guard<std::mutex> lock(mutex_);
      buf.slice = s;
      buf.job_seq = job.seq;
      buf.state = Buffer::kWorker;
      cv_.notify_one();
      return;
    }
    cudaStream_t stream = model_->stream();
    if (sl.kind == Slice::kBlob) {
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
    buf.job_seq = job.seq;
    buf.state = Buffer::kDevice;
  }

  // A restore slice the worker read: to the device, then each record's
  // planes into its block (or the blob's run into the arena slot).
  void issue_restore_slice(Job& job, Buffer& buf) {
    const Slice sl = slice_of(job, buf.slice);
    cudaStream_t stream = model_->stream();
    if (sl.kind == Slice::kBlob) {
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

  // ---- the file worker: pinned staging <-> the slab, the files beside it --
  void worker() {
    for (;;) {
      Buffer* buf = nullptr;
      bool restore = false;
      uint64_t seq = 0;
      std::vector<int64_t> pages;  // the slice's pages, taken under the lock: the job may end before the I/O does
      std::vector<uint8_t> index;
      std::vector<uint64_t> digests;
      bool write_index = false;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
          if (stop_ || has_pending_index_) return true;
          for (Buffer& b : buffers_)
            if (b.state == Buffer::kWorker) return true;
          return false;
        });
        for (Buffer& b : buffers_)
          if (b.state == Buffer::kWorker && buf == nullptr) buf = &b;
        if (has_pending_index_ && (buf == nullptr || stop_)) {
          index.swap(pending_index_);
          digests.swap(pending_digests_);
          has_pending_index_ = false;
          write_index = true;
        }
        if (stop_ && buf == nullptr) {
          lock.unlock();
          if (write_index) write_index_file(index, digests);  // the last committed index lands before exit
          return;
        }
        if (buf != nullptr) {
          // A buffer whose job is over (abandoned by the timeout before the
          // worker reached it) goes back unused.
          if (!active_ || active_->seq != buf->job_seq) {
            buf->state = Buffer::kFree;
            buf->slice = -1;
            continue;
          }
          const Job& job = *active_;
          restore = job.restore;
          seq = job.seq;
          const Slice sl = slice_of(job, buf->slice);
          for (int64_t j = 0; j < sl.count; ++j)
            pages.push_back(sl.kind == Slice::kBlock ? job.blocks[static_cast<size_t>(sl.first + j)].second
                            : sl.kind == Slice::kBlob ? job.blob_page + sl.first + j
                                                      : job.ids_page + sl.first + j);
        }
      }
      if (write_index) {
        write_index_file(index, digests);
        continue;
      }
      std::string why;
      bool ok = true;
      std::vector<int64_t> written;
      for (size_t j = 0; j < pages.size() && ok; ++j) {
        uint8_t* record = static_cast<uint8_t*>(buf->pinned) + j * page_bytes_;
        ok = restore ? read_page(pages[j], record, &why) : write_page(pages[j], record, &why);
        if (ok && !restore) written.push_back(pages[j]);
      }
      // The pages' CRCs reach the file before the slice counts as
      // written, so a commit never names a page whose check is not on
      // disk (a crash between leaves the entry unreadable, never wrong).
      if (ok && !restore && crc_fd_ >= 0) ok = write_crc_entries(written, &why);
      std::lock_guard<std::mutex> lock(mutex_);
      // The job may have ended (abandoned) while the I/O ran: its buffer
      // is free again and its outcome nobody's.
      if (!active_ || active_->seq != seq) {
        buf->state = Buffer::kFree;
        buf->slice = -1;
        continue;
      }
      if (!ok && active_->why.empty()) active_->why = why;
      // A finished read is the pump's to scatter; a finished write frees
      // the buffer; a failure frees it either way.
      buf->state = ok && restore ? Buffer::kReady : Buffer::kFree;
      if (!ok || !restore) buf->slice = -1;
    }
  }

  bool write_crc_entries(const std::vector<int64_t>& pages, std::string* why) {
    for (const int64_t page : pages) {
      const uint32_t crc = crc_[static_cast<size_t>(page)];
      if (::pwrite(crc_fd_, &crc, sizeof crc, static_cast<off_t>(page) * static_cast<off_t>(sizeof crc)) !=
          static_cast<ssize_t>(sizeof crc)) {
        *why = "write of the CRC of page " + std::to_string(page) + ": " + std::strerror(errno);
        return false;
      }
    }
    if (::fdatasync(crc_fd_) != 0) {
      *why = std::string("sync of the CRC table: ") + std::strerror(errno);
      return false;
    }
    return true;
  }

  // The index file: magic, the binding, the scheduler's bytes — written
  // whole to a sibling and renamed over the old one, so a reader sees one
  // or the other, never a torn file.
  void write_index_file(const std::vector<uint8_t>& bytes, const std::vector<uint64_t>& digests) {
    const std::string tmp = index_path_ + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) {
      DGPP_LOG_WARN("rank {}: NVMe cache: cannot write {}: {}", cfg_.rank, tmp, std::strerror(errno));
      return;
    }
    const std::vector<uint8_t> out = sched::write_nvme_index(cfg_.binding, bytes, digests);
    size_t done = 0;
    bool ok = true;
    while (ok && done < out.size()) {
      const ssize_t w = ::write(fd, out.data() + done, out.size() - done);
      if (w < 0 && errno == EINTR) continue;
      if (w <= 0) ok = false;
      else done += static_cast<size_t>(w);
    }
    ok = ok && ::fdatasync(fd) == 0;
    ::close(fd);
    ok = ok && ::rename(tmp.c_str(), index_path_.c_str()) == 0;
    if (!ok)
      DGPP_LOG_WARN("rank {}: NVMe cache: the index did not reach {}: {}", cfg_.rank, index_path_,
                    std::strerror(errno));
  }

  // The test hook: a delay or a failure injected before the call.
  bool hook(bool write, int64_t page, std::string* why) {
    if (!cfg_.io_hook) return true;
    const int r = cfg_.io_hook(write, page);
    if (r < 0) {
      *why = std::string("injected failure on the ") + (write ? "write" : "read") + " of page " + std::to_string(page);
      return false;
    }
    if (r > 0) std::this_thread::sleep_for(std::chrono::milliseconds(r));
    return true;
  }
  bool write_page(int64_t page, const void* buf, std::string* why) {
    if (!hook(true, page, why)) return false;
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
    if (!hook(false, page, why)) return false;
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
  int crc_fd_ = -1;            // the CRC table beside the slab (persist)
  std::string crc_path_, index_path_;
  std::vector<uint8_t> candidate_;  // a previous run's index for this binding, until adopted or refused
  std::vector<uint64_t> adopted_digests_;  // its entries' content digests, in its order
  std::map<int64_t, std::vector<uint8_t>> adopted_ids_;  // its entries' ids by their run's first page, until the first tick
  // The adoption's aligned read buffer, also released if construction fails.
  std::unique_ptr<void, decltype(&std::free)> aligned_page_{nullptr, &std::free};
  std::vector<uint8_t> pending_index_;  // the worker's next index write
  std::vector<uint64_t> pending_digests_;
  bool has_pending_index_ = false;
  uint64_t job_seq_ = 0;
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
