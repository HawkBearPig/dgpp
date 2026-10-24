#include "kernels/glm_spec.hpp"
#include "kernels/pdl.cuh"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <cuda/atomic>

#include "common/cuda_check.hpp"

namespace dgpp {

namespace {

constexpr int kUploadMaxWords = kPickMaxRows;  // the batch map's requests x rows (16 slots x 5 rows = 80 since 2026-10-23)

template <typename Word>
__global__ void upload_words_kernel(const Word* __restrict__ src,
                                    Word* __restrict__ dst, int count) {
  dgpp_pdl_wait();
  const int i = static_cast<int>(threadIdx.x);
  if (i >= count) return;
  cuda::atomic_ref<Word, cuda::thread_scope_system> ref(
      *const_cast<Word*>(src + i));
  dst[i] = ref.load(cuda::memory_order_relaxed);
}

template <typename Word>
void launch_upload_words(const char* who, const Word* pinned_src, Word* dst,
                         int count, cudaStream_t stream) {
  if (pinned_src == nullptr || dst == nullptr)
    throw std::invalid_argument(std::string(who) + ": null argument");
  if (count < 1 || count > kUploadMaxWords)
    throw std::invalid_argument(std::string(who) + ": count outside [1, " +
                                std::to_string(kUploadMaxWords) + "]");
  upload_words_kernel<Word><<<1, kUploadMaxWords, 0, stream>>>(pinned_src,
                                                               dst, count);
  DGPP_CUDA_OK(cudaGetLastError());
}

constexpr int kCopyThreads = 256;
constexpr size_t kUnit = 16;  // one uint4 per thread per iteration

// grid.y = segment, grid.x = chunks of the LARGEST segment; blocks past a
// smaller segment's end exit. Block (0,0)'s thread 0 owns the position
// advance so it happens exactly once whether or not anything copies.
__global__ void spec_commit_kernel(const PickVerdict* __restrict__ verdict, int rows,
                                   GlmSpecSegments segments, int64_t* __restrict__ session_pos,
                                   const int32_t* request_map, int batch_index) {
  dgpp_pdl_wait();
  const int req = request_map ? request_map[batch_index] : 0;
  if (req < 0) return;
  const int accepted = verdict->accepted;
  if (accepted <= 0) return;  // fixed-batch padding slot
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    session_pos[req] += accepted;
    if (segments.replay_pending != nullptr) segments.replay_pending[req] = accepted;
  }
  if (segments.replay_dst != nullptr && blockIdx.y == static_cast<unsigned>(segments.count)) {
    // The replay family's rows: this pass's saved inputs become the next
    // pass's replay source, whatever the verdict (it replays `accepted`).
    const size_t units = segments.replay_bytes / kUnit;
    const uint4* src = reinterpret_cast<const uint4*>(
        static_cast<const char*>(segments.replay_src) + static_cast<size_t>(req) * segments.replay_request_stride_bytes);
    uint4* dst = reinterpret_cast<uint4*>(static_cast<char*>(segments.replay_dst) +
                                          static_cast<size_t>(req) * segments.replay_request_stride_bytes);
    for (size_t i = static_cast<size_t>(blockIdx.x) * kCopyThreads + threadIdx.x;
         i < units; i += static_cast<size_t>(gridDim.x) * kCopyThreads)
      dst[i] = src[i];
    return;
  }
  if (accepted >= rows) return;  // every row stood: nothing to retract
  if (blockIdx.y >= static_cast<unsigned>(segments.count)) return;  // position-only configuration

  const GlmSpecSegment& s = segments.seg[blockIdx.y];
  const size_t units = s.bytes / kUnit;
  const uint4* src = reinterpret_cast<const uint4*>(
      static_cast<const char*>(s.snapshots) +
      static_cast<size_t>(accepted - 1) * s.row_stride_bytes);
  uint4* dst = reinterpret_cast<uint4*>(static_cast<char*>(s.dst) + req * s.request_stride_bytes);
  for (size_t i = static_cast<size_t>(blockIdx.x) * kCopyThreads + threadIdx.x;
       i < units; i += static_cast<size_t>(gridDim.x) * kCopyThreads)
    dst[i] = src[i];
}

// A row at or past the context ceiling is padding (-1), like a closed
// slot's rows: a request's lifetime reservation is capped at the ceiling,
// so the fixed-width verify's trailing rows past it must land in no K/V
// block, index stripe or state row. Their verdicts are past the request's
// budget by construction (2026-09-28).
__global__ void spec_positions_kernel(const int64_t* __restrict__ session_pos,
                                      int rows, int64_t max_context,
                                      int64_t* __restrict__ step_pos) {
  dgpp_pdl_wait();
  const int r = threadIdx.x;
  if (r >= rows) return;
  const int64_t p = *session_pos + r;
  step_pos[r] = p < max_context ? p : -1;
}

__global__ void spec_positions_batched_kernel(
    const int64_t* __restrict__ session_pos,
    const int32_t* __restrict__ request_ids, int rows, int rows_per_request,
    int64_t max_context, int64_t* __restrict__ step_pos) {
  dgpp_pdl_wait();
  const int r = threadIdx.x + blockIdx.x * blockDim.x;
  if (r >= rows) return;
  const int req = request_ids[r];
  const int64_t base = req >= 0 ? session_pos[req] : 0;
  const int64_t p = base + (r % rows_per_request);
  step_pos[r] = base > 0 && p < max_context ? p : -1;
}

__global__ void spec_draft_rows_kernel(const PickVerdict* __restrict__ verdict,
                                       int rows, int64_t* __restrict__ block_pos,
                                       int64_t* __restrict__ step_pos,
                                       int64_t* __restrict__ tokens,
                                       int64_t* __restrict__ next_out) {
  dgpp_pdl_wait();
  const int accepted = verdict->accepted;
  const int r = threadIdx.x;
  if (r < rows) {
    const bool real = r < accepted;
    step_pos[r] = real ? *block_pos + r : -1;
    tokens[r] = verdict->winners[real ? r : 0];
  }
  __syncthreads();  // every row read *block_pos before it moves
  if (r == 0) {
    *block_pos += accepted;
    *next_out = verdict->next;
  }
}

__global__ void spec_next_tokens_kernel(const int64_t* __restrict__ next,
                                        GlmSpecDrafts drafts,
                                        int64_t* __restrict__ tokens) {
  dgpp_pdl_wait();
  tokens[0] = *next;
  for (int c = 0; c < drafts.count; ++c) {
    const int32_t id = drafts.v[c]->next;
    tokens[1 + c] = id >= 0 ? id : 0;  // any valid id; a bad draft never stands
  }
}

__global__ void spec_chain_row_kernel(
    const PickVerdict* __restrict__ verify, int src_row,
    const PickVerdict* __restrict__ draft, const uint16_t* __restrict__ block_x,
    int hidden, uint16_t* __restrict__ hidden_cache,
    const int64_t* __restrict__ block_pos, int chain_index, int64_t max_context,
    int64_t* __restrict__ step_pos, int64_t* __restrict__ tokens,
    int32_t* __restrict__ req_spans) {
  dgpp_pdl_wait();
  const int row = verify != nullptr ? max(verify->accepted - 1, 0) : src_row;
  const int64_t pos = *block_pos + chain_index;
  const bool fits = pos >= 0 && pos < max_context;
  if (fits) {
    // The hidden row into the position cache, 16 bytes per thread-step.
    const uint4* src = reinterpret_cast<const uint4*>(
        block_x + static_cast<size_t>(row) * hidden);
    uint4* dst = reinterpret_cast<uint4*>(
        hidden_cache + static_cast<size_t>(pos) * hidden);
    const int units = hidden / 8;
    for (int i = threadIdx.x; i < units; i += blockDim.x) dst[i] = src[i];
  }
  if (threadIdx.x == 0) {
    step_pos[0] = fits ? pos : -1;
    const int32_t id = draft->next;
    tokens[0] = id >= 0 ? id : 0;
    req_spans[0] = 0;
    req_spans[1] = 1;
  }
}

// The chain row for the families that keep the draft's hidden in a per-slot
// WINDOW of window_rows rows by position (Qwen3.8-Flash-Next, GLM-4.7;
// 2026-09-10) instead of a per-position cache: the block's output row lands
// at window[pos % window_rows]; the rest is spec_chain_row_kernel's.
__global__ void spec_chain_row_window_kernel(
    const PickVerdict* __restrict__ verify, int src_row,
    const PickVerdict* __restrict__ draft, const uint16_t* __restrict__ block_x,
    int hidden, uint16_t* __restrict__ window, int window_rows,
    const int64_t* __restrict__ block_pos, int chain_index, int64_t max_context,
    int64_t* __restrict__ step_pos, int64_t* __restrict__ tokens,
    int32_t* __restrict__ req_spans) {
  dgpp_pdl_wait();
  const int row = verify != nullptr ? max(verify->accepted - 1, 0) : src_row;
  const int64_t pos = *block_pos + chain_index;
  const bool fits = pos >= 0 && pos < max_context;
  if (fits) {
    const uint4* src = reinterpret_cast<const uint4*>(
        block_x + static_cast<size_t>(row) * hidden);
    uint4* dst = reinterpret_cast<uint4*>(
        window + static_cast<size_t>(pos % window_rows) * hidden);
    const int units = hidden / 8;
    for (int i = threadIdx.x; i < units; i += blockDim.x) dst[i] = src[i];
  }
  if (threadIdx.x == 0) {
    step_pos[0] = fits ? pos : -1;
    const int32_t id = draft->next;
    tokens[0] = id >= 0 ? id : 0;
    req_spans[0] = 0;
    req_spans[1] = 1;
  }
}

__global__ void spec_draft_rows_batched_kernel(
    const PickVerdict* __restrict__ verdicts, int rows_per_request, int64_t* __restrict__ block_pos,
    int64_t* __restrict__ step_pos, int64_t* __restrict__ tokens, int64_t* __restrict__ next_out,
    const int32_t* request_map) {
  dgpp_pdl_wait();
  const int q = blockIdx.x;
  const int req = request_map ? request_map[q] : q;
  const PickVerdict& verdict = verdicts[q];
  const int accepted = verdict.accepted;
  const bool active = req >= 0 && accepted > 0;
  const int r = threadIdx.x;
  const int row = q * rows_per_request + r;
  if (r < rows_per_request) {
    const bool real = active && r < accepted;
    step_pos[row] = real ? block_pos[req] + r : -1;
    tokens[row] = active ? verdict.winners[real ? r : 0] : 0;
  }
  __syncthreads();
  if (r == 0 && active) {
    block_pos[req] += accepted;
    next_out[req] = verdict.next;
  }
}

__global__ void spec_verify_next_tokens_batched_kernel(const PickVerdict* __restrict__ verify,
                                                       int rows_per_request,
                                                       int64_t* __restrict__ tokens,
                                                       const int32_t* request_map) {
  dgpp_pdl_wait();
  const int q = blockIdx.x;
  const int req = request_map ? request_map[q] : q;
  const bool active = req >= 0 && verify[q].accepted > 0;
  if (req < 0) return;
  for (int r = threadIdx.x; r < rows_per_request; r += blockDim.x)
    tokens[req * rows_per_request + r] = active && r == 0 ? verify[q].next : 0;
}

__global__ void spec_next_tokens_batched_kernel(const int64_t* __restrict__ next,
                                                GlmSpecDrafts drafts, int rows_per_request,
                                                int64_t* __restrict__ tokens,
                                                const int32_t* request_map) {
  dgpp_pdl_wait();
  const int q = blockIdx.x;
  const int req = request_map ? request_map[q] : q;
  const bool active = req >= 0 && drafts.v[0][q].accepted > 0;
  for (int r = threadIdx.x; r < rows_per_request; r += blockDim.x) {
    int64_t token = 0;
    if (active && r == 0) token = next[req];
    if (active && r >= 1 && r - 1 < drafts.count) {
      const int32_t id = drafts.v[r - 1][q].next;
      token = id >= 0 ? id : 0;  // any valid id; a bad draft never stands
    }
    if (req >= 0) tokens[req * rows_per_request + r] = token;
  }
}

// The fixed batch's chain rows: block q stages request q's
// row and copies its hidden into the slot's window (the window form of
// spec_chain_row_window_kernel, one block per request).
__global__ void spec_chain_rows_batched_kernel(
    const PickVerdict* __restrict__ verify, const PickVerdict* __restrict__ draft,
    int rows_per_request, const uint16_t* __restrict__ block_x, int hidden,
    uint16_t* __restrict__ window, int window_rows, size_t window_stride,
    const int64_t* __restrict__ block_pos, int chain_index, int64_t max_context,
    int64_t* __restrict__ step_pos, int64_t* __restrict__ tokens, int32_t* __restrict__ req_ids,
    int32_t* __restrict__ req_spans, const int32_t* request_map) {
  dgpp_pdl_wait();
  const int q = blockIdx.x;
  const int req = request_map ? request_map[q] : q;
  const bool active = req >= 0 && verify[q].accepted > 0;
  const int row = chain_index == 0
                      ? q * rows_per_request + max(verify[q].accepted - 1, 0)
                      : q;
  const int64_t pos = active ? block_pos[req] + chain_index : -1;
  const bool fits = active && pos >= 0 && pos < max_context;
  if (fits) {
    const uint4* src = reinterpret_cast<const uint4*>(
        block_x + static_cast<size_t>(row) * hidden);
    uint4* dst = reinterpret_cast<uint4*>(window + static_cast<size_t>(req) * window_stride +
                                          static_cast<size_t>(pos % window_rows) * hidden);
    const int units = hidden / 8;
    for (int i = threadIdx.x; i < units; i += blockDim.x) dst[i] = src[i];
  }
  if (threadIdx.x == 0) {
    step_pos[q] = fits ? pos : -1;
    const int32_t id = active ? draft[q].next : -1;
    tokens[q] = id >= 0 ? id : 0;
    req_ids[q] = req;
    req_spans[2 * q] = q;
    req_spans[2 * q + 1] = 1;
  }
}

bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

__device__ __forceinline__ uint64_t ld_acquire_sys_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

__global__ void stage_wait_kernel(const uint64_t* __restrict__ pinned_seq,
                                  uint64_t* __restrict__ device_seq,
                                  uint32_t* __restrict__ pinned_late,
                                  int64_t timeout_ns) {
  dgpp_pdl_wait();
  if (threadIdx.x != 0) return;
  const uint64_t need = *device_seq + 1;
  *device_seq = need;
  uint64_t t0 = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  while (ld_acquire_sys_u64(pinned_seq) < need) {
    uint64_t now = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    if (static_cast<int64_t>(now - t0) > timeout_ns) {
      *pinned_late = 1u;
      __threadfence_system();
      break;
    }
    __nanosleep(2000);
  }
}

__global__ void publish_seq_kernel(uint64_t* __restrict__ device_seq,
                                   uint64_t* __restrict__ pinned_out) {
  dgpp_pdl_wait();
  if (threadIdx.x != 0) return;
  const uint64_t v = *device_seq + 1;
  *device_seq = v;
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(pinned_out), "l"(v) : "memory");
}

__global__ void publish_f32_kernel(const float* __restrict__ src,
                                   float* __restrict__ pinned_dst, int count,
                                   uint64_t* __restrict__ device_seq,
                                   uint64_t* __restrict__ pinned_seq) {
  dgpp_pdl_wait();
  for (int i = static_cast<int>(threadIdx.x); i < count; i += static_cast<int>(blockDim.x)) {
    const float v = src[i];
    asm volatile("st.relaxed.sys.global.f32 [%0], %1;" ::"l"(pinned_dst + i), "f"(v) : "memory");
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint64_t v = *device_seq + 1;
    *device_seq = v;
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(pinned_seq), "l"(v) : "memory");
  }
}

__global__ void gather_feed_kernel(const int64_t* __restrict__ feeds, int requests, int feed_rows,
                                   int rows_per_request, int64_t* __restrict__ out,
                                   const int32_t* request_map) {
  dgpp_pdl_wait();
  const int n = requests * rows_per_request;
  for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
    const int q = i / rows_per_request;
    const int t = i - q * rows_per_request;
    const int req = request_map ? request_map[q] : q;
    out[i] = req >= 0 ? feeds[req * feed_rows + t] : 0;
  }
}

__global__ void upload_words_kernel(const uint32_t* __restrict__ src,
                                    uint32_t* __restrict__ dst, size_t count) {
  dgpp_pdl_wait();
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(src + i) : "memory");
    dst[i] = v;
  }
}

}  // namespace

void glm_stage_wait(const uint64_t* pinned_stage_seq, uint64_t* device_seq,
                    uint32_t* pinned_late, int64_t timeout_ns,
                    cudaStream_t stream) {
  if (pinned_stage_seq == nullptr || device_seq == nullptr ||
      pinned_late == nullptr || timeout_ns < 1)
    throw std::invalid_argument("glm_stage_wait: null argument/timeout");
  stage_wait_kernel<<<1, 32, 0, stream>>>(pinned_stage_seq, device_seq,
                                          pinned_late, timeout_ns);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_publish_seq(uint64_t* device_seq, uint64_t* pinned_out,
                     cudaStream_t stream) {
  if (device_seq == nullptr || pinned_out == nullptr)
    throw std::invalid_argument("glm_publish_seq: null argument");
  publish_seq_kernel<<<1, 32, 0, stream>>>(device_seq, pinned_out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_publish_f32(const float* src, float* pinned_dst, int count,
                     uint64_t* device_seq, uint64_t* pinned_seq,
                     cudaStream_t stream) {
  if (src == nullptr || pinned_dst == nullptr || device_seq == nullptr ||
      pinned_seq == nullptr || count < 1 || count > 32)
    throw std::invalid_argument("glm_publish_f32: null argument/count");
  publish_f32_kernel<<<1, 32, 0, stream>>>(src, pinned_dst, count, device_seq,
                                           pinned_seq);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_gather_feed(const int64_t* feeds, int requests, int feed_rows, int rows_per_request,
                          int64_t* out, cudaStream_t stream, const int32_t* request_map) {
  if (feeds == nullptr || out == nullptr || requests < 1 || rows_per_request < 1 ||
      feed_rows < rows_per_request || requests * rows_per_request > 1024)
    throw std::invalid_argument("glm_spec_gather_feed: null argument/shape");
  gather_feed_kernel<<<1, 256, 0, stream>>>(feeds, requests, feed_rows, rows_per_request, out,
                                            request_map);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_upload_words(const uint32_t* pinned_src, uint32_t* dst, size_t count,
                      cudaStream_t stream) {
  if (pinned_src == nullptr || dst == nullptr || count == 0)
    throw std::invalid_argument("glm_upload_words: null argument/count");
  const unsigned blocks = static_cast<unsigned>(
      std::min<size_t>((count + 255) / 256, 256));
  upload_words_kernel<<<blocks, 256, 0, stream>>>(pinned_src, dst, count);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_commit(const PickVerdict* verdict, int rows, const GlmSpecSegments& segments,
                     int64_t* session_pos, cudaStream_t stream, const int32_t* request_map,
                     int batch_index) {
  if (verdict == nullptr || session_pos == nullptr)
    throw std::invalid_argument("glm_spec_commit: null verdict/position");
  if (rows < 1 || rows > kPickMaxRows)
    throw std::invalid_argument("glm_spec_commit: rows outside [1, " +
                                std::to_string(kPickMaxRows) + "]");
  if (segments.count < 0 || segments.count > kSpecMaxSegments)
    throw std::invalid_argument("glm_spec_commit: segment count");
  size_t max_units = 0;
  for (int i = 0; i < segments.count; ++i) {
    const GlmSpecSegment& s = segments.seg[i];
    if (!aligned16(s.dst) || !aligned16(s.snapshots) ||
        s.bytes % kUnit != 0 || s.row_stride_bytes % kUnit != 0)
      throw std::invalid_argument(
          "glm_spec_commit: segment " + std::to_string(i) +
          " must be 16-byte aligned with 16-byte-multiple sizes");
    max_units = std::max(max_units, s.bytes / kUnit);
  }
  const bool replay = segments.replay_dst != nullptr;
  if (replay) {
    if (!aligned16(segments.replay_dst) || !aligned16(segments.replay_src) || segments.replay_bytes % kUnit != 0 ||
        segments.replay_request_stride_bytes % kUnit != 0 || segments.replay_pending == nullptr)
      throw std::invalid_argument("glm_spec_commit: the replay rows must be 16-byte aligned with a pending counter");
    max_units = std::max(max_units, segments.replay_bytes / kUnit);
  }
  // A single block still runs (the position advance) when no segment
  // exists; otherwise enough blocks to stream the largest segment. The
  // replay family's copy runs as one more y-slice (index `count`).
  const unsigned chunks = static_cast<unsigned>(
      std::min<size_t>(1024, (max_units + kCopyThreads - 1) / kCopyThreads));
  const dim3 grid(std::max(1u, chunks),
                  static_cast<unsigned>(std::max(1, segments.count + (replay ? 1 : 0))));
  spec_commit_kernel<<<grid, kCopyThreads, 0, stream>>>(verdict, rows, segments, session_pos,
                                                        request_map, batch_index);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_upload_i32(const int32_t* pinned_src, int32_t* dst, int count,
                    cudaStream_t stream) {
  launch_upload_words("glm_upload_i32", pinned_src, dst, count, stream);
}

void glm_upload_i64(const int64_t* pinned_src, int64_t* dst, int count,
                    cudaStream_t stream) {
  launch_upload_words("glm_upload_i64", pinned_src, dst, count, stream);
}

namespace {
__global__ void device_copy_kernel(uint4* __restrict__ dst,
                                   const uint4* __restrict__ src,
                                   size_t units) {
  dgpp_pdl_wait();
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < units; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    dst[i] = src[i];
}
}  // namespace

void glm_device_copy(void* dst, const void* src, size_t bytes,
                     cudaStream_t stream) {
  if (dst == nullptr || src == nullptr || bytes == 0 || bytes % 16 != 0 ||
      (reinterpret_cast<uintptr_t>(dst) & 15) != 0 ||
      (reinterpret_cast<uintptr_t>(src) & 15) != 0)
    throw std::invalid_argument("glm_device_copy: alignment/size");
  const size_t units = bytes / 16;
  const unsigned blocks = static_cast<unsigned>(
      std::min<size_t>((units + 255) / 256, 1024));
  device_copy_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<uint4*>(dst), static_cast<const uint4*>(src), units);
  DGPP_CUDA_OK(cudaGetLastError());
}

namespace {
__global__ void store_u8x2_kernel(uint8_t* __restrict__ dst, uint8_t a, uint8_t b) {
  dgpp_pdl_wait();
  dst[0] = a;
  dst[1] = b;
}
}  // namespace

void glm_device_store_u8x2(uint8_t* dst, uint8_t a, uint8_t b, cudaStream_t stream) {
  if (dst == nullptr) throw std::invalid_argument("glm_device_store_u8x2: null");
  store_u8x2_kernel<<<1, 1, 0, stream>>>(dst, a, b);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_positions(const int64_t* session_pos, int rows, int64_t max_context,
                        int64_t* step_pos, cudaStream_t stream) {
  if (rows < 1 || rows > kPickMaxRows) throw std::invalid_argument("glm_spec_positions: rows");
  if (max_context < 1) throw std::invalid_argument("glm_spec_positions: context ceiling");
  spec_positions_kernel<<<1, kPickMaxRows, 0, stream>>>(session_pos, rows, max_context, step_pos);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_positions_batched(const int64_t* session_pos,
                                const int32_t* request_ids, int rows,
                                int rows_per_request, int64_t max_context,
                                int64_t* step_pos, cudaStream_t stream) {
  if (session_pos == nullptr || request_ids == nullptr || step_pos == nullptr)
    throw std::invalid_argument("glm_spec_positions_batched: null argument");
  if (rows < 1 || rows > kPickMaxRows || rows_per_request < 1 ||
      rows % rows_per_request != 0)
    throw std::invalid_argument("glm_spec_positions_batched: row shape");
  if (max_context < 1) throw std::invalid_argument("glm_spec_positions_batched: context ceiling");
  spec_positions_batched_kernel<<<1, kPickMaxRows, 0, stream>>>(
      session_pos, request_ids, rows, rows_per_request, max_context, step_pos);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_draft_rows(const PickVerdict* verdict, int rows,
                         int64_t* block_pos, int64_t* step_pos, int64_t* tokens,
                         int64_t* next_out, cudaStream_t stream) {
  if (verdict == nullptr || block_pos == nullptr || step_pos == nullptr ||
      tokens == nullptr || next_out == nullptr)
    throw std::invalid_argument("glm_spec_draft_rows: null argument");
  if (rows < 1 || rows > kPickMaxRows)
    throw std::invalid_argument("glm_spec_draft_rows: rows outside [1, " +
                                std::to_string(kPickMaxRows) + "]");
  spec_draft_rows_kernel<<<1, kPickMaxRows, 0, stream>>>(verdict, rows, block_pos, step_pos, tokens,
                                                         next_out);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_draft_rows_batched(const PickVerdict* verdicts, int requests, int rows_per_request,
                                 int64_t* block_pos, int64_t* step_pos, int64_t* tokens,
                                 int64_t* next_out, cudaStream_t stream,
                                 const int32_t* request_map) {
  if (verdicts == nullptr || block_pos == nullptr || step_pos == nullptr ||
      tokens == nullptr || next_out == nullptr)
    throw std::invalid_argument("glm_spec_draft_rows_batched: null argument");
  if (requests < 1 || requests > kPickMaxRequests ||
      rows_per_request < 1 || requests * rows_per_request > kPickMaxRows)
    throw std::invalid_argument("glm_spec_draft_rows_batched: request shape");
  spec_draft_rows_batched_kernel<<<requests, 32, 0, stream>>>(
      verdicts, rows_per_request, block_pos, step_pos, tokens, next_out, request_map);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_next_tokens(const int64_t* next, const GlmSpecDrafts& drafts,
                          int64_t* tokens, cudaStream_t stream) {
  if (next == nullptr || tokens == nullptr)
    throw std::invalid_argument("glm_spec_next_tokens: null argument");
  if (drafts.count < 1 || drafts.count > kSpecMaxDrafts)
    throw std::invalid_argument("glm_spec_next_tokens: draft count");
  for (int c = 0; c < drafts.count; ++c)
    if (drafts.v[c] == nullptr)
      throw std::invalid_argument("glm_spec_next_tokens: null draft verdict");
  spec_next_tokens_kernel<<<1, 1, 0, stream>>>(next, drafts, tokens);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_chain_row(const PickVerdict* verify_verdict, int src_row,
                        const PickVerdict* draft_verdict,
                        const uint16_t* block_x, int hidden,
                        uint16_t* hidden_cache, const int64_t* block_pos,
                        int chain_index, int64_t max_context,
                        int64_t* step_pos, int64_t* tokens, int32_t* req_spans,
                        cudaStream_t stream) {
  if (draft_verdict == nullptr || block_x == nullptr ||
      hidden_cache == nullptr || block_pos == nullptr || step_pos == nullptr ||
      tokens == nullptr || req_spans == nullptr)
    throw std::invalid_argument("glm_spec_chain_row: null argument");
  if (hidden < 8 || hidden % 8 != 0 || !aligned16(block_x) ||
      !aligned16(hidden_cache))
    throw std::invalid_argument("glm_spec_chain_row: hidden alignment");
  if (verify_verdict == nullptr && (src_row < 0 || src_row >= kPickMaxRows))
    throw std::invalid_argument("glm_spec_chain_row: source row");
  if (chain_index < 0 || chain_index >= kSpecMaxDrafts || max_context < 1)
    throw std::invalid_argument("glm_spec_chain_row: chain index/context");
  spec_chain_row_kernel<<<1, 256, 0, stream>>>(
      verify_verdict, src_row, draft_verdict, block_x, hidden, hidden_cache,
      block_pos, chain_index, max_context, step_pos, tokens, req_spans);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_chain_row_window(const PickVerdict* verify_verdict, int src_row,
                               const PickVerdict* draft_verdict,
                               const uint16_t* block_x, int hidden,
                               uint16_t* window, int window_rows,
                               const int64_t* block_pos, int chain_index,
                               int64_t max_context, int64_t* step_pos,
                               int64_t* tokens, int32_t* req_spans,
                               cudaStream_t stream) {
  if (draft_verdict == nullptr || block_x == nullptr || window == nullptr ||
      block_pos == nullptr || step_pos == nullptr || tokens == nullptr ||
      req_spans == nullptr)
    throw std::invalid_argument("glm_spec_chain_row_window: null argument");
  if (hidden < 8 || hidden % 8 != 0 || !aligned16(block_x) || !aligned16(window))
    throw std::invalid_argument("glm_spec_chain_row_window: hidden alignment");
  if (window_rows < 1)
    throw std::invalid_argument("glm_spec_chain_row_window: window rows");
  if (verify_verdict == nullptr && (src_row < 0 || src_row >= kPickMaxRows))
    throw std::invalid_argument("glm_spec_chain_row_window: source row");
  if (chain_index < 0 || chain_index >= kSpecMaxDrafts || max_context < 1)
    throw std::invalid_argument("glm_spec_chain_row_window: chain index/context");
  spec_chain_row_window_kernel<<<1, 256, 0, stream>>>(
      verify_verdict, src_row, draft_verdict, block_x, hidden, window, window_rows,
      block_pos, chain_index, max_context, step_pos, tokens, req_spans);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_verify_next_tokens_batched(const PickVerdict* verify_verdicts, int requests,
                                         int rows_per_request, int64_t* tokens, cudaStream_t stream,
                                         const int32_t* request_map) {
  if (verify_verdicts == nullptr || tokens == nullptr)
    throw std::invalid_argument(
        "glm_spec_verify_next_tokens_batched: null argument");
  if (requests < 1 || requests > kPickMaxRequests ||
      rows_per_request < 1 || requests * rows_per_request > kPickMaxRows)
    throw std::invalid_argument(
        "glm_spec_verify_next_tokens_batched: request shape");
  spec_verify_next_tokens_batched_kernel<<<requests, 32, 0, stream>>>(
      verify_verdicts, rows_per_request, tokens, request_map);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_next_tokens_batched(const int64_t* next, const GlmSpecDrafts& drafts, int requests,
                                  int rows_per_request, int64_t* tokens, cudaStream_t stream,
                                  const int32_t* request_map) {
  if (next == nullptr || tokens == nullptr)
    throw std::invalid_argument("glm_spec_next_tokens_batched: null argument");
  if (drafts.count < 1 || drafts.count > kSpecMaxDrafts)
    throw std::invalid_argument("glm_spec_next_tokens_batched: draft count");
  for (int c = 0; c < drafts.count; ++c)
    if (drafts.v[c] == nullptr)
      throw std::invalid_argument(
          "glm_spec_next_tokens_batched: null draft verdicts");
  if (requests < 1 || requests > kPickMaxRequests ||
      rows_per_request != 1 + drafts.count ||
      requests * rows_per_request > kPickMaxRows)
    throw std::invalid_argument(
        "glm_spec_next_tokens_batched: the feed is [next, draft_1 .. "
        "draft_n] per request (rows_per_request = 1 + drafts)");
  spec_next_tokens_batched_kernel<<<requests, 32, 0, stream>>>(next, drafts, rows_per_request,
                                                               tokens, request_map);
  DGPP_CUDA_OK(cudaGetLastError());
}

void glm_spec_chain_rows_batched(const PickVerdict* verify_verdicts,
                                 const PickVerdict* draft_verdicts, int requests,
                                 int rows_per_request, const uint16_t* block_x, int hidden,
                                 uint16_t* window, int window_rows, size_t window_stride_elems,
                                 const int64_t* block_pos, int chain_index, int64_t max_context,
                                 int64_t* step_pos, int64_t* tokens, int32_t* req_ids,
                                 int32_t* req_spans, cudaStream_t stream,
                                 const int32_t* request_map) {
  if (verify_verdicts == nullptr || draft_verdicts == nullptr ||
      block_x == nullptr || window == nullptr || block_pos == nullptr ||
      step_pos == nullptr || tokens == nullptr || req_ids == nullptr ||
      req_spans == nullptr)
    throw std::invalid_argument("glm_spec_chain_rows_batched: null argument");
  if (requests < 1 || requests > kPickMaxRequests || rows_per_request < 2 ||
      requests * rows_per_request > kPickMaxRows)
    throw std::invalid_argument("glm_spec_chain_rows_batched: request shape");
  if (hidden < 8 || hidden % 8 != 0 || !aligned16(block_x) || !aligned16(window))
    throw std::invalid_argument("glm_spec_chain_rows_batched: hidden alignment");
  if (window_rows < 1 ||
      window_stride_elems < static_cast<size_t>(window_rows) * hidden ||
      window_stride_elems % 8 != 0)
    throw std::invalid_argument("glm_spec_chain_rows_batched: window shape");
  if (chain_index < 0 || chain_index >= kSpecMaxDrafts || max_context < 1)
    throw std::invalid_argument("glm_spec_chain_rows_batched: chain index/context");
  spec_chain_rows_batched_kernel<<<requests, 256, 0, stream>>>(
      verify_verdicts, draft_verdicts, rows_per_request, block_x, hidden, window, window_rows,
      window_stride_elems, block_pos, chain_index, max_context, step_pos, tokens, req_ids,
      req_spans, request_map);
  DGPP_CUDA_OK(cudaGetLastError());
}

__global__ void batch_rows_kernel(const int32_t* map, int requests, int rows, int32_t* ids,
                                  int32_t* spans) {
  dgpp_pdl_wait();
  const int i = threadIdx.x;
  if (i < requests * rows) ids[i] = map[i / rows];
  if (i < requests) {
    spans[2 * i] = i * rows;
    spans[2 * i + 1] = rows;
  }
}
void glm_batch_rows(const int32_t* map, int requests, int rows, int32_t* ids, int32_t* spans,
                    cudaStream_t stream) {
  if (!map || !ids || !spans || requests < 1 || requests > kPickMaxRequests || rows < 1 ||
      requests * rows > kPickMaxRows)
    throw std::invalid_argument("glm_batch_rows: invalid map/shape");
  batch_rows_kernel<<<1, kPickMaxRows, 0, stream>>>(map, requests, rows, ids, spans);
  DGPP_CUDA_OK(cudaGetLastError());
}
}  // namespace dgpp
