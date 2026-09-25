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

namespace dgpp::serve {

struct NvmeCachePlan {
  std::string directory;      // the configured path, ~ expanded
  std::string slab;           // <directory>/rank<r>.slab
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

// The reserve the tier never takes from a filesystem: 1 GiB.
constexpr size_t kNvmeCacheReserveBytes = size_t{1} << 30;

// Plans the slab for a rank: `block_bytes` one cache block's bytes across
// its planes (0: a family without a paged cache — blobs alone),
// `blob_bytes` the snapshot size, `block_tokens` the pool's block (0: no
// pool) and `context_limit` the longest request. Creates the directory
// (mkdir -p) so the free-space probe sees the filesystem the slab will
// live on. The plan's `error` names what to change when it does not fit.
NvmeCachePlan plan_nvme_cache(const std::string& path, double capacity_gib, int rank, size_t block_bytes,
                              size_t blob_bytes, int64_t block_tokens, int64_t context_limit);

}  // namespace dgpp::serve
