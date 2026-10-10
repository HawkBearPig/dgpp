#pragma once

// ResidentImage: the byte image of one rank's resident layers, cached on
// the node's disk so the next startup streams device bytes instead of
// rebuilding them from the checkpoint.
//
// WHY. A rank's resident layer is the END of a pipeline — mmap the shard,
// slice the rank's rows/columns out of each tensor (a column slice pages
// in the whole matrix for a quarter of its bytes), copy to pinned staging,
// H2D, dequantize the bridge tensors, pack. At GLM that is ~130-180 s per
// rank of fault-driven, single-threaded work over 305 GB of mapped shards.
// The pipeline's OUTPUT is a flat byte range per layer (the layer bump:
// weights, scales, dequantized bridges, packs — no pointers), and the
// layout of the views INTO that range is reproducible without touching a
// source byte (the build's counting mode lays out every grant). So: dump
// each bump once, and next time lay out the views, pread the blob into
// staging, H2D. One sequential read of ~82 GiB per rank, no slicing, no
// dequant — the disk's rate is the load time.
//
// FILE. One file per (checkpoint, config, world, rank, head sharding,
// loader format version) — the key is the caller's; a mismatch replaces
// the file. Header + a fixed table (one entry per layer) + 4 KiB-aligned
// blobs. A layer's entry is published AFTER its blob is fully written and
// synced, so a crash mid-write leaves that layer absent (rebuilt and
// re-appended next time) and never a torn blob with a valid entry. Every
// entry carries a 64-bit word fold of its blob; verification on read is
// opt-in (it costs a pass over 82 GiB).
//
// Bitwise: the restored bytes ARE the built bytes — glm_loader_test pins a
// build-vs-restore round trip byte-for-byte on the fixture.
//
// I/O. Blobs move with O_DIRECT when the filesystem allows it: on the GB10
// nodes a buffered stream tops out at 1.2 GB/s read / 1.8 GB/s write (the
// page-cache copy plus the reclaim it forces at the memory watermark), a
// direct stream at 5.6 / 5.0 GB/s — the NVMe's line rate, on one thread.
// The blob's 4 KiB-aligned prefix goes direct; the sub-page tail and the
// table go through the buffered descriptor. A filesystem that refuses
// O_DIRECT (tmpfs) silently gets the buffered path for everything.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace dgpp {

class ResidentImage {
 public:
  static constexpr uint32_t kFormatVersion = 2;  // 2: the DSA projections resident as FP8 pairs where aligned

  // Opens `dir/<key hex>.img`, creating the directory and the file when
  // absent; a file whose header disagrees (version, key, layer count) is
  // replaced. Throws std::runtime_error on I/O failure.
  ResidentImage(const std::string& dir, uint64_t key, int layers);
  ~ResidentImage();
  ResidentImage(const ResidentImage&) = delete;
  ResidentImage& operator=(const ResidentImage&) = delete;

  int layers() const { return static_cast<int>(entries_.size()); }
  int present() const;  // layers with a published blob
  bool has_layer(int layer) const;
  size_t layer_bytes(int layer) const;  // 0 when absent
  const std::string& path() const { return path_; }

  // pread the layer's blob into `dst` (`bytes` must equal layer_bytes).
  // `verify` re-folds the bytes against the entry. Throws on mismatch.
  // `dst` should be 4 KiB-aligned to take the direct path (pinned
  // allocations are); anything else reads buffered, correctly but slower.
  // Thread safety: concurrent read_layer calls for DIFFERENT layers are
  // safe (pread is thread-safe; the entry snapshot is taken under the
  // mutex and the blob moves outside it). Concurrent write_layer calls
  // are safe: each reserves its file range under the mutex (ftruncate),
  // writes outside it, then publishes its entry under the mutex — the
  // file order across layers is nondeterministic, the entries are exact.
  // A write racing a read of the SAME layer is not supported; callers
  // give each layer to one worker.
  void read_layer(int layer, void* dst, size_t bytes, bool verify) const;

  // Appends the blob at the next 4 KiB boundary, fdatasyncs, then
  // publishes the entry. An already-present layer is rewritten (new blob,
  // old bytes orphaned — rebuilds are rare and the file is a cache).
  void write_layer(int layer, const void* src, size_t bytes);

  bool direct_io() const { return direct_fd_ >= 0; }

  // Notes: small named sidecars (`dir/<key hex>.<name>`) that share the
  // image's key and trust level — used for the boot digest, which is a
  // function of the same checkpoint bytes the layers came from and would
  // otherwise cost a 3 GiB pass over the shards on every start. A note is
  // published atomically (temp file + rename); read_note returns false
  // when absent or of a different size, so callers fall back to computing.
  bool read_note(const std::string& name, void* dst, size_t bytes) const;
  void write_note(const std::string& name, const void* src, size_t bytes) const;

  // 64-bit word fold (xor-multiply, the bus's shape); a byte tail is folded
  // as a partial word. Streams at memory speed, unlike a byte-serial hash.
  static uint64_t fold(const void* data, size_t bytes);

 private:
  struct Entry {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t fold = 0;
    uint64_t reserved = 0;
  };
  void write_fresh(uint64_t key);
  void write_entry(int layer) const;
  uint64_t table_offset(int layer) const;
  void read_blob(void* dst, size_t bytes, uint64_t offset) const;
  void write_blob(const void* src, size_t bytes, uint64_t offset) const;
  std::string note_path(const std::string& name) const;

  std::string path_;
  std::string stem_;    // path_ without ".img": the notes' prefix
  int fd_ = -1;         // buffered: header, table, blob tails
  int direct_fd_ = -1;  // O_DIRECT: blob bodies; -1 when unsupported
  std::vector<Entry> entries_;
  mutable std::mutex mutex_;  // the entry table + write offset allocation
};

}  // namespace dgpp
