#pragma once
// The NVMe cold tier's pre-flight (issue #26): before a rank allocates
// anything, the configured slab must have a directory it can write, room
// on that filesystem for its whole capacity beside a reserve, and a
// capacity that holds at least one entry at the request context limit —
// a tier that cannot hold the prompts it exists for is a configuration
// error, not a quiet degradation. The same check runs under
// --memory-plan and prints the plan line.
#include <cstddef>
#include <cstdint>
#include <string>

#include "common/cache_storage.hpp"

namespace dgpp::serve {

struct NvmeCachePlan {
  std::string directory;      // absolute directory, existing symlinks resolved
  std::string slab;           // <directory>/rank<r>.slab
  CacheStorage storage;
  size_t capacity_bytes = 0;  // the configured capacity
  size_t page_bytes = 0;      // one cache block's record (4 KiB multiple)
  int64_t pages = 0;          // records the capacity holds
  int64_t blob_pages = 0;     // pages one snapshot blob spans
  size_t minimum_bytes = 0;   // one entry at the context limit plus its blob
  size_t free_bytes = 0;      // the filesystem's free bytes (an existing slab counted as free)
  size_t reserve_bytes = 0;   // the free space the tier leaves alone
  std::string error;          // empty: the plan fits
  std::string describe() const;
};

// The reserve the tier never takes from a filesystem (nvme_cache.reserve_gib): 1 GiB by default.
constexpr size_t kNvmeCacheReserveBytes = size_t{1} << 30;

// Existing automatically generated bindings are sixteen lowercase hex digits.
bool valid_nvme_cache_binding(const std::string& binding);

// Keep the legacy automatic binding byte-for-byte when no reuse is requested.
std::string nvme_cache_binding(const std::string& version, const std::string& config_digest,
                               const std::string& checkpoint, size_t block_bytes, size_t snapshot_bytes,
                               double capacity_gib, int64_t block_tokens, int world,
                               const std::string& reuse_binding = "");

// Read-only preflight for explicit reuse; never creates or clears a cache.
// The tier checks the actual snapshot geometry and page checksums at open.
std::string check_nvme_cache_reuse(const NvmeCachePlan& plan, const std::string& binding,
                                  int rank, int64_t block_tokens);

// Plans the slab for a rank: `block_bytes` one cache block's bytes across
// its planes (0: a family without a paged cache — blobs alone),
// `blob_bytes` the snapshot size, `block_tokens` the pool's block (0: no
// pool) and `context_limit` the longest request. Placement and free space
// are checked on the nearest existing ancestor before creating anything.
// With `create_directory`, mkdir -p follows a successful check; without it
// (--memory-plan) the plan leaves no trace. The plan's `error` names what
// to change when it does not fit.
// With a `binding` (nvme_cache.persist) the slab lives under
// <path>/<binding>/, so another checkpoint's or configuration's slab is
// never mistaken for this deployment's.
// `chunk_bytes` (nvme_cache.chunk_mib) must hold one page: a family whose
// block outgrows the slice is refused here, not after the model loaded.
NvmeCachePlan plan_nvme_cache(const std::string& path, double capacity_gib, int rank, size_t block_bytes,
                              size_t blob_bytes, int64_t block_tokens, int64_t context_limit,
                              bool create_directory = true, size_t reserve_bytes = kNvmeCacheReserveBytes,
                              const std::string& binding = "", size_t chunk_bytes = 0,
                              const std::string& required_mount = "", bool allow_buffered = false);

}  // namespace dgpp::serve
