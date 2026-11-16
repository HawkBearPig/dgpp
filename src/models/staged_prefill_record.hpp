// A model's shutdown record of the prefill tiles' staged listed attention
// (kernels/dsa.hpp dsa_attn_listed_staged). A tile that lists more distinct
// cache rows than the staged copy holds reads the cache instead — the same
// output without the saving — so a workload that does so often shows in the
// rank's log. The counters are the process's.
#pragma once

#include <cuda_runtime.h>

#include "common/log.hpp"
#include "kernels/dsa.hpp"

namespace dgpp {

inline void log_staged_prefill_tiles(const char* family, cudaStream_t stream) noexcept {
  try {
    unsigned long long st[4] = {0, 0, 0, 0};
    dsa_attn_stage_stats(st, /*clear=*/false, stream);
    if (st[0] != 0)
      DGPP_LOG_INFO("{}: {} prefill attention launches read a staged copy of their distinct cache rows and {} read the "
                    "cache (more distinct rows than the copy holds); distinct rows a launch: {} on average, {} at most",
                    family, st[0] - st[1], st[1], st[2] / st[0], st[3]);
  } catch (...) {
  }
}

}  // namespace dgpp
