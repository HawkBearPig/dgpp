// DSA / CSA2 score-selection benchmark with production paging and exact selection checks.
// Run only on idle hardware; cold samples evict L2 outside the timer.
// Selection is checked against a host sort of production score keys.
// This is not an independent oracle for the score arithmetic.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/csa2.hpp"
#include "kernels/dsa.hpp"
#include "kernels/topk_select.cuh"

namespace {
template <class T>
struct Buffer {
  T* p = nullptr;
  size_t size;
  explicit Buffer(size_t n) : size(n) { DGPP_CUDA_OK(cudaMalloc(&p, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void upload(const std::vector<T>& v) {
    DGPP_CUDA_OK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> v(size);
    DGPP_CUDA_OK(cudaMemcpy(v.data(), p, size * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};

__global__ void evict_l2(uint4* data, size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    data[i] = make_uint4(i, i + 1, i + 2, i + 3);
}

struct Launch {
  std::function<void()> fn;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  Launch(std::function<void()> f, bool capture, cudaStream_t stream) : fn(std::move(f)) {
    if (capture) {
      DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      fn();
      DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
      DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    }
  }
  ~Launch() {
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
  }
  void run(cudaStream_t stream) const {
    if (exec)
      DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
    else
      fn();
  }
};

std::vector<double> measure(const Launch& launch, bool cold, Buffer<uint4>& eviction, int iters,
                            int warmup, cudaStream_t stream) {
  for (int i = 0; i < warmup; ++i) launch.run(stream);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  cudaEvent_t start, end;
  DGPP_CUDA_OK(cudaEventCreate(&start));
  DGPP_CUDA_OK(cudaEventCreate(&end));
  std::vector<double> times;
  for (int i = 0; i < iters; ++i) {
    if (cold) evict_l2<<<1024, 256, 0, stream>>>(eviction.p, eviction.size);
    DGPP_CUDA_OK(cudaEventRecord(start, stream));
    launch.run(stream);
    DGPP_CUDA_OK(cudaEventRecord(end, stream));
    DGPP_CUDA_OK(cudaEventSynchronize(end));
    float elapsed;
    DGPP_CUDA_OK(cudaEventElapsedTime(&elapsed, start, end));
    times.push_back(elapsed * 1000.0);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  std::sort(times.begin(), times.end());
  return times;
}

void require(bool condition, const std::string& what) {
  if (!condition) throw std::runtime_error(what);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string family = "deepseek", only;
    int rows = 5, ctx = 131072, iters = 30, warmup = 5, select_k_arg = 0;
    bool graph = true, cold_only = false;
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--eager") {
        graph = false;
        continue;
      }
      if (arg == "--cold-only") {
        cold_only = true;
        continue;
      }
      require(i + 1 < argc, "missing argument");
      std::string value = argv[++i];
      if (arg == "--family")
        family = value;
      else if (arg == "--stage")
        only = value;
      else if (arg == "--ctx")
        ctx = std::stoi(value);
      else if (arg == "--rows")
        rows = std::stoi(value);
      else if (arg == "--iters")
        iters = std::stoi(value);
      else if (arg == "--warmup")
        warmup = std::stoi(value);
      else if (arg == "--select-k")  // the index budget (engine.index_topk); 0: the family's default
        select_k_arg = std::stoi(value);
      else
        throw std::invalid_argument("unknown argument " + arg);
    }
    require(family == "flash" || family == "full" || family == "deepseek" ||
                family == "deepseek-encoder",
            "bad family");
    require(rows > 0 && rows <= 32 && ctx >= 64 && ctx >= rows && ctx < (1 << 21) && iters > 0 &&
                warmup >= 0,
            "bad size");
    const bool csa = family == "deepseek";
    const int compression = family == "deepseek-encoder" ? 2 : 1;
    const int kpool = family == "flash" ? 4 : 1;
    const int ppb = 128 / kpool / compression;
    const int select_k = select_k_arg > 0 ? select_k_arg : (family == "full" ? 2048 : 512);
    require(select_k > 0 && (select_k & (select_k - 1)) == 0 && select_k <= dgpp::kSelectMaxK, "bad select_k");
    const int reference_k = csa ? 1 : select_k;
    const int max_selected = select_k * kpool + kpool - 1;
    const int entries = ctx / kpool / compression;
    const int blocks = (entries + ppb - 1) / ppb, stride = blocks * ppb;
    std::mt19937 rng(20260921);
    std::vector<uint8_t> q(size_t(rows) * 32 * 128), cache(size_t(stride) * 128);
    for (auto* v : {&q, &cache})
      for (auto& x : *v) {
        x = uint8_t(rng());
        if ((x & 0x7f) == 0x7f) x ^= 1;
      }
    std::vector<float> w(size_t(rows) * 32), scales(stride);
    for (auto& x : w) x = (float(rng() % 2001) - 1000.f) / 100000.f;
    for (auto& x : scales) x = std::ldexp(1.f, -6 - int(rng() % 3));
    std::vector<int32_t> table(blocks), req(rows, 0);
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<int64_t> pos(rows);
    for (int r = 0; r < rows; ++r) pos[r] = (ctx - rows + r + 1) / compression - 1;
    Buffer<uint8_t> dq(q.size()), dk(cache.size());
    Buffer<float> dw(w.size()), ds(scales.size());
    Buffer<int32_t> dt(table.size()), dr(rows), top(size_t(rows) * max_selected), counts(rows),
        ctr(2);
    Buffer<int64_t> dp(rows);
    Buffer<uint8_t> ws(dgpp::dsa_select_workspace_bytes(rows, stride));
    dq.upload(q);
    dk.upload(cache);
    dw.upload(w);
    ds.upload(scales);
    dt.upload(table);
    dr.upload(req);
    dp.upload(pos);
    DGPP_CUDA_OK(cudaMemset(ws.p, 0, ws.size));
    DGPP_CUDA_OK(cudaMemset(ctr.p, 0, 8));
    cudaStream_t stream;
    DGPP_CUDA_OK(cudaStreamCreate(&stream));
    auto dsa = [&] {
      dgpp::dsa_select_decode(dq.p, dw.p, dr.p, dp.p, rows, dt.p, blocks, dk.p, ds.p, ppb, 32, 128,
                              reference_k, kpool, max_selected, top.p, counts.p, ws.p, stride,
                              ctr.p, 0, stream, family != "flash");
    };
    dsa();
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    // Independent host sorting of the production score keys verifies selection,
    // ties, causal bounds, ascending expansion and padding. It is not an independent score oracle.
    std::vector<uint64_t> keys(size_t(rows) * stride);
    DGPP_CUDA_OK(cudaMemcpy(keys.data(), ws.p, keys.size() * 8, cudaMemcpyDeviceToHost));
    auto check_dsa = [&] {
      auto actual = top.download(), n = counts.download();
      for (int r = 0; r < rows; ++r) {
        const int visible = (pos[r] + 1) / kpool;
        std::vector<uint64_t> sorted(keys.begin() + size_t(r) * stride,
                                     keys.begin() + size_t(r) * stride + visible);
        std::sort(sorted.begin(), sorted.end());
        std::vector<int> ids;
        for (int i = 0; i < std::min(reference_k, visible); ++i)
          ids.push_back(visible <= reference_k ? i : int(sorted[i] & dgpp::kIdxMask));
        std::sort(ids.begin(), ids.end());
        std::vector<int> want;
        for (int id : ids)
          for (int j = 0; j < kpool; ++j) want.push_back(id * kpool + j);
        for (int p = visible * kpool; p <= pos[r]; ++p) want.push_back(p);
        require(n[r] == int(want.size()), "DSA count");
        want.resize(max_selected, -1);
        for (int i = 0; i < max_selected; ++i)
          require(actual[size_t(r) * max_selected + i] == want[i], "DSA selection");
      }
    };
    check_dsa();
    uint64_t phases[8]{};
    dgpp::dsa_select_debug_phases(phases);
    constexpr int candidate_blocks = 2048, block_size = 8;
    Buffer<int32_t> candidates(size_t(rows) * candidate_blocks), candidate_counts(rows);
    Buffer<float> logits(size_t(rows) * stride);
    dgpp::csa2_prepare_kernel_smem();
    auto candidate = [&] {
      dgpp::csa2_select_candidates_decode(dq.p, dw.p, dr.p, dp.p, rows, dt.p, blocks, dk.p, ds.p,
                                          ppb, 32, block_size, candidate_blocks,
                                          reinterpret_cast<uint64_t*>(ws.p), stride, candidates.p,
                                          candidate_counts.p, stream);
    };
    auto listed = [&] {
      dgpp::csa2_select_listed_decode(dq.p, dw.p, dr.p, dp.p, rows, dt.p, blocks, dk.p, ds.p, ppb,
                                      32, candidates.p, candidate_blocks, candidate_counts.p,
                                      block_size, select_k, reinterpret_cast<uint64_t*>(ws.p),
                                      stride, top.p, counts.p, stream);
    };
    // Diagnostic: the unchanged prefill streaming selector over precomputed
    // scores. This is not a stage of the new decoder.
    auto select_only = [&] {
      dgpp::csa2_select_rows_prefill(logits.p, stride, dp.p, rows, select_k, candidates.p,
                                     candidate_blocks, candidate_counts.p, block_size, top.p,
                                     counts.p, stream);
    };
    // The fused prefill scorer over the cache read as a contiguous [stride,
    // 128] key buffer (the rows' visible entries bound the reads): the
    // per-row logits of the prefill selection, timed per `rows` queries at
    // this context. Runs last — it overwrites the logits buffer.
    auto score_prefill = [&] {
      dgpp::csa2_score_prefill(dq.p, dk.p, dw.p, ds.p, dp.p, rows, stride, 32, logits.p, stride, stream);
    };
    std::vector<int32_t> expected_top, expected_counts, expected_candidates,
        expected_candidate_counts;
    if (csa) {
      std::vector<float> values(keys.size());
      for (int r = 0; r < rows; ++r)
        for (int p = 0; p <= pos[r]; ++p) {
          uint32_t sortable = ~uint32_t(keys[size_t(r) * stride + p] >> dgpp::kIdxBits);
          uint32_t bits = (sortable & 0x80000000u) ? sortable ^ 0x80000000u : ~sortable;
          std::memcpy(&values[size_t(r) * stride + p], &bits, 4);
        }
      logits.upload(values);
      candidate();
      listed();
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      auto got_c = candidates.download(), got_nc = candidate_counts.download();
      expected_top.assign(size_t(rows) * select_k, -1);
      expected_counts.resize(rows);
      for (int r = 0; r < rows; ++r) {
        const int visible = pos[r] + 1, nb = visible / block_size;
        std::vector<uint64_t> ck;
        for (int b = 0; b < nb; ++b) {
          uint64_t best = UINT64_MAX;
          for (int j = 0; j < block_size; ++j)
            best = std::min(best, keys[size_t(r) * stride + b * block_size + j]);
          best = (best & ~dgpp::kIdxMask) | b;
          if (visible % block_size == 0 && b == nb - 1)
            best = (uint64_t(0x007fffff) << dgpp::kIdxBits) | b;
          ck.push_back(best);
        }
        std::sort(ck.begin(), ck.end());
        std::vector<int> cb;
        for (int i = 0; i < std::min(nb, candidate_blocks); ++i)
          cb.push_back(int(ck[i] & dgpp::kIdxMask));
        std::sort(cb.begin(), cb.end());
        require(got_nc[r] == int(cb.size()), "candidate count");
        for (int i = 0; i < candidate_blocks; ++i)
          require(got_c[size_t(r) * candidate_blocks + i] == (i < int(cb.size()) ? cb[i] : -1),
                  "candidate blocks");
        std::vector<uint64_t> eligible;
        for (int b : cb)
          for (int j = 0; j < block_size; ++j)
            eligible.push_back(keys[size_t(r) * stride + b * block_size + j]);
        for (int p = nb * block_size; p < visible; ++p)
          eligible.push_back(keys[size_t(r) * stride + p]);
        std::sort(eligible.begin(), eligible.end());
        std::vector<int> ids;
        for (int i = 0; i < std::min(select_k, int(eligible.size())); ++i)
          ids.push_back(int(eligible[i] & dgpp::kIdxMask));
        std::sort(ids.begin(), ids.end());
        expected_counts[r] = int(ids.size());
        std::copy(ids.begin(), ids.end(), expected_top.begin() + size_t(r) * select_k);
      }
      expected_candidates = got_c;
      expected_candidate_counts = got_nc;
      require(top.download() == expected_top, "listed selection");
      require(counts.download() == expected_counts, "listed counts");
      select_only();
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      require(top.download() == expected_top, "selection-only oracle");
    }
    std::fprintf(stderr, "PASS selections family=%s ctx=%d rows=%d\n", family.c_str(), ctx, rows);
    cudaDeviceProp prop{};
    DGPP_CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    Buffer<uint4> eviction(std::max<size_t>(64ull << 20, 4ull * prop.l2CacheSize) / sizeof(uint4));
    std::vector<std::pair<std::string, std::function<void()>>> stages{{"dsa", dsa}};
    if (csa)
      stages = {{"candidate", candidate},
                {"restricted", listed},
                {"restricted_select_only", select_only},
                {"combined", [&] {
                   candidate();
                   listed();
                 }},
                {"score_prefill", score_prefill}};
    require(only.empty() || std::any_of(stages.begin(), stages.end(),
                                        [&](const auto& stage) { return stage.first == only; }),
            "unknown stage for family");
    for (auto& [name, fn] : stages) {
      if (!only.empty() && name != only) continue;
      Launch launch(fn, graph, stream);
      for (bool cold : {false, true}) {
        if (cold_only && !cold) continue;
        auto times = measure(launch, cold, eviction, iters, warmup, stream);
        DGPP_CUDA_OK(cudaGetLastError());
        if (csa) {
          require(candidates.download() == expected_candidates, "repeat candidate IDs");
          require(candidate_counts.download() == expected_candidate_counts,
                  "repeat candidate counts");
        }
        if (csa &&
            (name == "restricted" || name == "restricted_select_only" || name == "combined")) {
          require(top.download() == expected_top, "repeat selection");
          require(counts.download() == expected_counts, "repeat selection counts");
        }
        if (!csa) {
          check_dsa();
          dgpp::dsa_select_debug_phases(phases);
        }
        if (csa) std::fill(std::begin(phases), std::end(phases), uint64_t(0));
        std::printf(
            "{\"family\":\"%s\",\"ctx\":%d,\"rows\":%d,\"stage\":\"%s\",\"cache\":\"%s\",\"graph\":"
            "%s,\"iters\":%d,\"median_us\":%.3f,\"min_us\":%.3f,\"max_us\":%.3f,\"dsa_last_block_"
            "score_us\":%.3f,\"dsa_last_block_wait_us\":%.3f,\"dsa_selection_sum_us\":%.3f,\"dsa_"
            "expand_sum_us\":%.3f}\n",
            family.c_str(), ctx, rows, name.c_str(), cold ? "cold" : "warm",
            graph ? "true" : "false", iters, times[times.size() / 2], times.front(), times.back(),
            (phases[1] - phases[0]) / 1000., (phases[2] - phases[1]) / 1000., phases[3] / 1000.,
            phases[4] / 1000.);
        std::fflush(stdout);
      }
    }
    DGPP_CUDA_OK(cudaStreamDestroy(stream));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
