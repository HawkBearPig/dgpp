#pragma once
// A paged cache plane as the NVMe cold tier sees it (issue #26): one
// contiguous device region in which physical block `b` occupies bytes
// [b * block_bytes, (b + 1) * block_bytes). Every family pool lists its
// planes (a layer's K rows, its V rows, its index keys, their scales ...)
// so a block's complete state can be gathered into one record and
// scattered back without the tier knowing the family's layout.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dgpp {

struct CachePlane {
  uint8_t* base = nullptr;   // the plane's device base
  size_t block_bytes = 0;    // one physical block's bytes in this plane
};

// The bytes one block takes across every plane: a record's payload.
inline size_t cache_block_bytes(const std::vector<CachePlane>& planes) {
  size_t n = 0;
  for (const CachePlane& p : planes) n += p.block_bytes;
  return n;
}

}  // namespace dgpp
