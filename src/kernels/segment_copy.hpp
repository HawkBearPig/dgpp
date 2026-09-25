#pragma once
// A batched device copy: many (src, dst, bytes) segments in one launch —
// the NVMe cold tier's gather of a cache block's planes into a contiguous
// staging record and the scatter back (engine/nvme_tier.hpp). One cache
// block is dozens of small strided ranges (every layer's K, V and index
// rows); a memcpy per range would be tens of thousands of launches per
// restored document.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

struct CopySegment {
  const void* src = nullptr;
  void* dst = nullptr;
  uint64_t bytes = 0;
};

// Copies every segment of the DEVICE table `segs[0..n)`; stream-ordered.
// Segments whose ends are 16-byte aligned copy in 16-byte units, the rest
// byte by byte. Segments must not overlap.
void segment_copy(const CopySegment* segs, int n, cudaStream_t stream);

}  // namespace dgpp
