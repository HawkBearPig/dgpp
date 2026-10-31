// The MXFP4 expert consumer's gate (MOONSHOT-1.md section 5, phase 3):
// against an fp32 reference with exact dequantization over the same
// tile-major bytes, the gate and up pair's SwiGLU output is within two bf16
// ulps and the down partials within 1e-3; the form is bitwise its own
// across launches with a changing sentinel and across group sizes (a slot
// computed in a group of one equals the same slot in a group of 29); and
// the plain grids' cold streaming rate is reported. DeepSeek-V4.1's
// world-4 shape: 576 by 5120 experts.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/step/fp8_dense.cuh"
#include "kernels/step/mhc.cuh"
#include "kernels/glm_mhc_launch.hpp"
#include "models/glm/mhc.hpp"
#include "kernels/step/mxfp4_expert.cuh"
#include "kernels/step/mxfp4_row.cuh"
#include "step_reference.cuh"

namespace {

using namespace dgpp;
using namespace dgpp::step;

constexpr int kH = 5120, kI = 576;
constexpr int kTopK = 6, kSlotsPerToken = kTopK + 1;
constexpr size_t kPair = tile_major_bytes(kI, kH), kPairScales = static_cast<size_t>(kI) * (kH / 32);
constexpr size_t kDown = tile_major_bytes(kH, kI), kDownScales = static_cast<size_t>(kH) * (kI / 32);
constexpr size_t kAlign = 256;
constexpr size_t align_up(size_t b) { return (b + kAlign - 1) / kAlign * kAlign; }
constexpr size_t kOffGateSc = align_up(kPair), kOffUp = kOffGateSc + align_up(kPairScales), kOffUpSc = kOffUp + align_up(kPair),
                 kOffDown = kOffUpSc + align_up(kPairScales), kOffDownSc = kOffDown + align_up(kDown),
                 kExpertBytes = kOffDownSc + align_up(kDownScales);

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

__host__ __device__ Mxfp4Expert expert_at(const uint8_t* arena, int e) {
  const uint8_t* b = arena + static_cast<size_t>(e) * kExpertBytes;
  Mxfp4Expert x;
  x.gate = b;
  x.gate_scales = b + kOffGateSc;
  x.up = b + kOffUp;
  x.up_scales = b + kOffUpSc;
  x.down = b + kOffDown;
  x.down_scales = b + kOffDownSc;
  return x;
}
// Partials token-major: the slots of one output element in one sector.
struct OutIndex {
  __host__ __device__ size_t operator()(int slot, int row) const {
    return (static_cast<size_t>(slot / kSlotsPerToken) * kH + row) * kSlotsPerToken + (slot % kSlotsPerToken);
  }
};

// Payload bytes hashed, e8m0 scales in [112, 136): both decode paths cover
// the magnitudes without overflow.
__global__ void init_experts(uint8_t* arena, int experts) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(experts) * kExpertBytes) return;
  const size_t off = i % kExpertBytes;
  const bool scale = (off >= kOffGateSc && off < kOffUp) || (off >= kOffUpSc && off < kOffDown) || off >= kOffDownSc;
  unsigned h = static_cast<unsigned>(i * 2654435761ull) ^ static_cast<unsigned>(i >> 17) ^ 0x9e3779b9u;
  h ^= h >> 13; h *= 0x85ebca6bu; h ^= h >> 16;
  arena[i] = scale ? static_cast<uint8_t>(112 + (h % 24)) : static_cast<uint8_t>(h);
}
__global__ void init_x(uint16_t* x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned h = static_cast<unsigned>(i) * 2654435761u; h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12;
  x[i] = float_to_bf16_bits(((h & 0xffff) / 65536.0f - 0.5f) * 0.25f);
}

struct GroupTable {
  const int32_t* slots;   // [groups * kMxfp4GroupRows]
  const int32_t* tokens;
  const int32_t* count;   // [groups]
  const int32_t* expert;  // [groups]
};
__device__ Mxfp4Group group_of(const GroupTable& t, int g) {
  Mxfp4Group grp;
  grp.slots = t.slots + g * kMxfp4GroupRows;
  grp.tokens = t.tokens + g * kMxfp4GroupRows;
  grp.count = t.count[g];
  return grp;
}

__global__ void __launch_bounds__(kMxfp4Threads) gate_up_kernel(const uint8_t* arena, GroupTable t, const uint16_t* x, uint16_t* act) {
  extern __shared__ __align__(128) uint8_t smem[];
  const Mxfp4Group g = group_of(t, blockIdx.y);
  mxfp4_gate_up<kH, kI>(expert_at(arena, t.expert[blockIdx.y]), g, blockIdx.x, x, act, smem);
}
__global__ void __launch_bounds__(kMxfp4Threads) down_kernel(const uint8_t* arena, GroupTable t, const uint16_t* act, float* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  const Mxfp4Group g = group_of(t, blockIdx.y);
  mxfp4_stage_down_act<kI>(act, g, smem);
  __syncthreads();
  mxfp4_down<kH, kI>(expert_at(arena, t.expert[blockIdx.y]), g, blockIdx.x, smem, out, OutIndex{});
}

// The reference: exact dequantization of the same bytes, accumulated in
// double, so the consumer's own fp32 rounding is the only difference. The
// down reference also writes the sum of the terms' magnitudes, the scale
// an fp32 accumulation's error is bounded by.
__device__ float e2m1_value(uint32_t code) {
  const int em = (code >> 1) & 3, mm = code & 1;
  float v = em == 0 ? 0.5f * mm : ldexpf(1.0f + 0.5f * mm, em - 1);
  return (code & 8) ? -v : v;
}
__global__ void ref_gate_up(const uint8_t* arena, GroupTable t, const uint16_t* x, uint16_t* act, float* gu, float* mag) {
  const Mxfp4Group g = group_of(t, blockIdx.y);
  const Mxfp4Expert e = expert_at(arena, t.expert[blockIdx.y]);
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= kI) return;
  for (int r = 0; r < g.count; ++r) {
    const int slot = g.slots[r], tok = g.tokens[r];
    float d[2];
    for (int m = 0; m < 2; ++m) {
      const uint8_t* pay = m ? e.up : e.gate;
      const uint8_t* sc = (m ? e.up_scales : e.gate_scales) + static_cast<size_t>(row) * (kH / 32);
      double a = 0.0, mg = 0.0;
      for (int k = 0; k < kH; ++k) {
        const uint32_t code = (pay[tile_major_offset(row, k, kH)] >> (4 * (k & 1))) & 0xF;
        const double term = static_cast<double>(ldexpf(e2m1_value(code), static_cast<int>(sc[k >> 5]) - 127)) *
                            static_cast<double>(bf16_bits_to_float(x[static_cast<size_t>(tok) * kH + k]));
        a += term;
        mg += fabs(term);
      }
      d[m] = static_cast<float>(a);
      gu[(static_cast<size_t>(slot) * kI + row) * 2 + m] = d[m];
      mag[(static_cast<size_t>(slot) * kI + row) * 2 + m] = static_cast<float>(mg);
    }
    act[static_cast<size_t>(slot) * kI + row] = swiglu_bf16(d[0], d[1]);
  }
}
__global__ void ref_down(const uint8_t* arena, GroupTable t, const uint16_t* act, float* out, float* abs_sum) {
  const Mxfp4Group g = group_of(t, blockIdx.y);
  const Mxfp4Expert e = expert_at(arena, t.expert[blockIdx.y]);
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= kH) return;
  const uint8_t* sc = e.down_scales + static_cast<size_t>(row) * (kI / 32);
  for (int r = 0; r < g.count; ++r) {
    const int slot = g.slots[r];
    double a = 0.0, m = 0.0;
    for (int k = 0; k < kI; ++k) {
      const uint32_t code = (e.down[tile_major_offset(row, k, kI)] >> (4 * (k & 1))) & 0xF;
      const double term = static_cast<double>(ldexpf(e2m1_value(code), static_cast<int>(sc[k >> 5]) - 127)) *
                          static_cast<double>(bf16_bits_to_float(act[static_cast<size_t>(slot) * kI + k]));
      a += term;
      m += fabs(term);
    }
    out[OutIndex{}(slot, row)] = static_cast<float>(a);
    abs_sum[OutIndex{}(slot, row)] = static_cast<float>(m);
  }
}

struct Rig {
  int experts = 0, rows = 0, groups = 0;
  uint8_t* arena = nullptr;
  uint16_t* x = nullptr;
  uint16_t* act = nullptr;
  float* out = nullptr;
  int32_t *slots = nullptr, *tokens = nullptr, *count = nullptr, *expert = nullptr;
  std::vector<int32_t> h_slots, h_tokens, h_count, h_expert;
  size_t act_elems = 0, out_elems = 0;
  ~Rig() {
    cudaFree(arena); cudaFree(x); cudaFree(act); cudaFree(out);
    cudaFree(slots); cudaFree(tokens); cudaFree(count); cudaFree(expert);
  }
  GroupTable table() const { return GroupTable{slots, tokens, count, expert}; }
};

// `rows` tokens pick kTopK distinct experts of `experts`; the slots of one
// expert form a group of at most `max_group` slots (1: every slot alone).
std::unique_ptr<Rig> make_rig(int experts, int rows, int max_group, unsigned seed) {
  auto r = std::make_unique<Rig>();
  r->experts = experts;
  r->rows = rows;
  size_t free_b = 0, total = 0;
  DGPP_CUDA_OK(cudaMemGetInfo(&free_b, &total));
  const size_t need = static_cast<size_t>(experts) * kExpertBytes + (size_t{128} << 20);
  if (free_b < need) dgpp::test::skip("needs " + std::to_string(need >> 20) + " MB free, " + std::to_string(free_b >> 20));
  DGPP_CUDA_OK(cudaMalloc(&r->arena, static_cast<size_t>(experts) * kExpertBytes));
  init_experts<<<static_cast<unsigned>((static_cast<size_t>(experts) * kExpertBytes + 255) / 256), 256>>>(r->arena, experts);
  DGPP_CUDA_OK(cudaMalloc(&r->x, static_cast<size_t>(rows) * kH * 2));
  init_x<<<(rows * kH + 255) / 256, 256>>>(r->x, rows * kH);
  const int slots_n = rows * kSlotsPerToken;
  r->act_elems = static_cast<size_t>(slots_n) * kI;
  r->out_elems = static_cast<size_t>(slots_n) * kH;
  DGPP_CUDA_OK(cudaMalloc(&r->act, r->act_elems * 2));
  DGPP_CUDA_OK(cudaMalloc(&r->out, r->out_elems * 4));
  // The picks, then the slots by expert.
  std::vector<std::vector<int>> by_expert(static_cast<size_t>(experts));
  unsigned h = seed * 2654435761u + 12345u;
  for (int t = 0; t < rows; ++t) {
    std::vector<int> picked;
    while (static_cast<int>(picked.size()) < kTopK) {
      h = h * 1664525u + 1013904223u;
      const int e = static_cast<int>((h >> 8) % static_cast<unsigned>(experts));
      if (std::find(picked.begin(), picked.end(), e) == picked.end()) picked.push_back(e);
    }
    for (int j = 0; j < kTopK; ++j) by_expert[static_cast<size_t>(picked[static_cast<size_t>(j)])].push_back(t * kSlotsPerToken + j);
  }
  for (int e = 0; e < experts; ++e) {
    const auto& v = by_expert[static_cast<size_t>(e)];
    for (size_t q = 0; q < v.size(); q += static_cast<size_t>(max_group)) {
      const int cnt = static_cast<int>(std::min<size_t>(static_cast<size_t>(max_group), v.size() - q));
      r->h_count.push_back(cnt);
      r->h_expert.push_back(e);
      for (int i = 0; i < kMxfp4GroupRows; ++i) {
        const int slot = i < cnt ? v[q + static_cast<size_t>(i)] : v[q];
        r->h_slots.push_back(slot);
        r->h_tokens.push_back(slot / kSlotsPerToken);
      }
    }
  }
  r->groups = static_cast<int>(r->h_count.size());
  auto up = [](int32_t** d, const std::vector<int32_t>& v) {
    DGPP_CUDA_OK(cudaMalloc(d, v.size() * 4));
    DGPP_CUDA_OK(cudaMemcpy(*d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  };
  up(&r->slots, r->h_slots);
  up(&r->tokens, r->h_tokens);
  up(&r->count, r->h_count);
  up(&r->expert, r->h_expert);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  return r;
}

constexpr int kSmem = mxfp4_unit_smem_bytes<kI>();

void run_consumer(const Rig& r, cudaStream_t s) {
  gate_up_kernel<<<dim3((kI + kMxfp4UnitRows - 1) / kMxfp4UnitRows, static_cast<unsigned>(r.groups)), kMxfp4Threads, kSmem, s>>>(
      r.arena, r.table(), r.x, r.act);
  down_kernel<<<dim3(kH / kMxfp4UnitRows, static_cast<unsigned>(r.groups)), kMxfp4Threads, kSmem, s>>>(r.arena, r.table(), r.act, r.out);
}
// The host's copy of the epilogue, for the error envelope.
uint16_t host_swiglu(float gate, float up) {
  float g = bf16_bits_to_float(float_to_bf16_bits(gate)), u = bf16_bits_to_float(float_to_bf16_bits(up));
  if (g > kMxfp4SwigluLimit) g = kMxfp4SwigluLimit;
  u = std::min(std::max(u, -kMxfp4SwigluLimit), kMxfp4SwigluLimit);
  const uint16_t t = float_to_bf16_bits(g * (1.0f / (1.0f + std::exp(-g))));
  return float_to_bf16_bits(bf16_bits_to_float(t) * u);
}
float bf16_ulp(float v) {
  int e = 0;
  std::frexp(std::fabs(v), &e);
  return std::ldexp(1.0f, e - 8);
}
template <typename T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
// Which slots the rig computes (the others are untouched).
std::vector<bool> live_slots(const Rig& r) {
  std::vector<bool> live(static_cast<size_t>(r.rows) * kSlotsPerToken, false);
  for (int g = 0; g < r.groups; ++g)
    for (int i = 0; i < r.h_count[static_cast<size_t>(g)]; ++i) live[static_cast<size_t>(r.h_slots[static_cast<size_t>(g) * kMxfp4GroupRows + i])] = true;
  return live;
}

}  // namespace

DGPP_TEST(mxfp4_consumer_matches_the_fp32_reference) {
  auto rig = make_rig(/*experts=*/48, /*rows=*/8, /*max_group=*/32, 3);
  DGPP_CUDA_OK(cudaFuncSetAttribute(gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  DGPP_CUDA_OK(cudaFuncSetAttribute(down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  DGPP_CUDA_OK(cudaMemset(rig->act, 0, rig->act_elems * 2));
  DGPP_CUDA_OK(cudaMemset(rig->out, 0, rig->out_elems * 4));
  run_consumer(*rig, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> act = download(rig->act, rig->act_elems);
  const std::vector<float> out = download(rig->out, rig->out_elems);
  float *gu = nullptr, *gu_mag = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&gu, rig->act_elems * 2 * 4));
  DGPP_CUDA_OK(cudaMalloc(&gu_mag, rig->act_elems * 2 * 4));
  ref_gate_up<<<dim3((kI + 127) / 128, static_cast<unsigned>(rig->groups)), 128>>>(rig->arena, rig->table(), rig->x, rig->act, gu, gu_mag);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> act_r = download(rig->act, rig->act_elems);
  const std::vector<float> gu_h = download(gu, rig->act_elems * 2);
  const std::vector<float> gu_m = download(gu_mag, rig->act_elems * 2);
  cudaFree(gu);
  cudaFree(gu_mag);
  // The down stage is judged on the consumer's own act.
  DGPP_CUDA_OK(cudaMemcpy(rig->act, act.data(), rig->act_elems * 2, cudaMemcpyHostToDevice));
  float* abs_sum = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&abs_sum, rig->out_elems * 4));
  ref_down<<<dim3(kH / 128, static_cast<unsigned>(rig->groups)), 128>>>(rig->arena, rig->table(), rig->act, rig->out, abs_sum);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> out_r = download(rig->out, rig->out_elems);
  const std::vector<float> mag = download(abs_sum, rig->out_elems);
  cudaFree(abs_sum);
  const std::vector<bool> live = live_slots(*rig);
  // Within two bf16 ulps of the exact result, or inside the envelope the
  // gate's and up's own fp32 accumulation error allows (4e-6 of the terms'
  // magnitude each, through the epilogue) widened by two ulps.
  size_t exact = 0, ulp = 0, enveloped = 0, bad = 0, n = 0, shown = 0;
  for (size_t i = 0; i < act.size(); ++i) {
    if (!live[i / kI]) continue;
    ++n;
    if (act[i] == act_r[i]) { ++exact; continue; }
    if (std::abs(static_cast<int>(act[i]) - static_cast<int>(act_r[i])) <= 2) { ++ulp; continue; }
    const float g = gu_h[2 * i], u = gu_h[2 * i + 1];
    const float dg = gu_m[2 * i] * 4e-6f + 1e-7f, du = gu_m[2 * i + 1] * 4e-6f + 1e-7f;
    float lo = 1e30f, hi = -1e30f;
    for (int c = 0; c < 4; ++c) {
      const float v = bf16_bits_to_float(host_swiglu(g + ((c & 1) ? dg : -dg), u + ((c & 2) ? du : -du)));
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    const float slack = 2.0f * bf16_ulp(std::max(std::fabs(lo), std::fabs(hi)));
    const float got = bf16_bits_to_float(act[i]);
    if (got >= lo - slack && got <= hi + slack) { ++enveloped; continue; }
    ++bad;
    if (shown++ < 5)
      std::printf("      act[slot %zu][row %zu]: got %g ref %g (gate %g of %g, up %g of %g; envelope %g .. %g)\n", i / kI, i % kI, got,
                  bf16_bits_to_float(act_r[i]), g, gu_m[2 * i], u, gu_m[2 * i + 1], lo, hi);
  }
  // An fp32 accumulation of n terms is within about log2(n) ulps of the
  // sum of the terms' magnitudes; the consumer sums 576 terms in blocks.
  size_t out_bad = 0, out_n = 0;
  double worst = 0;
  shown = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const int slot = static_cast<int>((i / kSlotsPerToken) / kH) * kSlotsPerToken + static_cast<int>(i % kSlotsPerToken);
    if (!live[static_cast<size_t>(slot)]) continue;
    ++out_n;
    const double d = std::fabs(static_cast<double>(out[i]) - out_r[i]);
    const double scale = static_cast<double>(mag[i]) * 4e-6 + 1e-7;
    worst = std::max(worst, d / scale);
    if (d > scale) {
      ++out_bad;
      if (shown++ < 5) std::printf("      out[%zu]: got %g ref %g, terms' magnitude %g\n", i, out[i], out_r[i], mag[i]);
    }
  }
  std::printf("   %d groups of %d rows: act %zu exact, %zu within 2 ulps, %zu inside the accumulation envelope, %zu beyond of %zu; down partials %zu beyond 4e-6 of their terms' magnitude of %zu (worst %.2f of the bound)\n",
              rig->groups, rig->rows, exact, ulp, enveloped, bad, n, out_bad, out_n, worst);
  require(bad == 0, "gate and up outputs beyond two bf16 ulps of the reference");
  require(out_bad == 0, "down partials beyond the fp32 accumulation bound");
}

DGPP_TEST(mxfp4_consumer_is_bitwise_across_launches_and_group_sizes) {
  auto grouped = make_rig(48, 8, 32, 5);
  auto alone = make_rig(48, 8, 1, 5);  // the same picks, every slot its own group
  DGPP_CUDA_OK(cudaFuncSetAttribute(gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  DGPP_CUDA_OK(cudaFuncSetAttribute(down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  auto run = [&](Rig& r, uint8_t sentinel) {
    DGPP_CUDA_OK(cudaMemset(r.act, sentinel, r.act_elems * 2));
    DGPP_CUDA_OK(cudaMemset(r.out, sentinel, r.out_elems * 4));
    run_consumer(r, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    return std::make_pair(download(r.act, r.act_elems), download(r.out, r.out_elems));
  };
  const auto first = run(*grouped, 0x00), second = run(*grouped, 0x7f), third = run(*grouped, 0xff);
  const std::vector<bool> live = live_slots(*grouped);
  auto same_live = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const std::vector<float>& oa, const std::vector<float>& ob) {
    for (size_t i = 0; i < a.size(); ++i)
      if (live[i / kI] && a[i] != b[i]) return false;
    for (size_t i = 0; i < oa.size(); ++i) {
      const int slot = static_cast<int>((i / kSlotsPerToken) / kH) * kSlotsPerToken + static_cast<int>(i % kSlotsPerToken);
      if (live[static_cast<size_t>(slot)] && std::memcmp(&oa[i], &ob[i], 4) != 0) return false;
    }
    return true;
  };
  require(same_live(first.first, second.first, first.second, second.second) && same_live(first.first, third.first, first.second, third.second),
          "repeat launches with a changing sentinel differ");
  const auto solo = run(*alone, 0x00);
  require(same_live(first.first, solo.first, first.second, solo.second), "a slot in a group of one differs from the same slot grouped");
  std::printf("   %d grouped units and %d single-slot units agree bitwise over three sentinels\n", grouped->groups, alone->groups);
}

DGPP_TEST(mxfp4_consumer_plain_grid_rate) {
  // Cold experts: more than the L2 holds, one group of one token each.
  auto rig = make_rig(/*experts=*/192, /*rows=*/32, 32, 11);
  DGPP_CUDA_OK(cudaFuncSetAttribute(gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  DGPP_CUDA_OK(cudaFuncSetAttribute(down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  cudaEvent_t e0, e1, e2;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  DGPP_CUDA_OK(cudaEventCreate(&e2));
  std::vector<int> distinct_experts;
  for (int e : rig->h_expert) if (std::find(distinct_experts.begin(), distinct_experts.end(), e) == distinct_experts.end()) distinct_experts.push_back(e);
  const double pair_mb = distinct_experts.size() * 2.0 * (kPair + kPairScales) / 1e6, down_mb = distinct_experts.size() * (kDown + kDownScales) / 1e6;
  for (int rep = 0; rep < 2; ++rep) {
    DGPP_CUDA_OK(cudaEventRecord(e0));
    gate_up_kernel<<<dim3((kI + kMxfp4UnitRows - 1) / kMxfp4UnitRows, static_cast<unsigned>(rig->groups)), kMxfp4Threads, kSmem>>>(
        rig->arena, rig->table(), rig->x, rig->act);
    DGPP_CUDA_OK(cudaEventRecord(e1));
    down_kernel<<<dim3(kH / kMxfp4UnitRows, static_cast<unsigned>(rig->groups)), kMxfp4Threads, kSmem>>>(rig->arena, rig->table(), rig->act, rig->out);
    DGPP_CUDA_OK(cudaEventRecord(e2));
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    float m1 = 0, m2 = 0;
    DGPP_CUDA_OK(cudaEventElapsedTime(&m1, e0, e1));
    DGPP_CUDA_OK(cudaEventElapsedTime(&m2, e1, e2));
    std::printf("   plain grids, %s, %zu distinct experts in %d groups: gate and up %.0f us = %.0f GB/s (%.0f MB) | down %.0f us = %.0f GB/s (%.0f MB)\n",
                rep ? "L2-warm where it fits" : "cold", distinct_experts.size(), rig->groups, m1 * 1e3, pair_mb / m1, pair_mb, m2 * 1e3, down_mb / m2, down_mb);
  }
}

// ---- the row-major fp8 pair and down with F32 scales on 32 x 32 blocks (the shared expert's form)

namespace {

constexpr int kFp8Block = 32;
constexpr size_t kFp8Pair = static_cast<size_t>(kI) * kH, kFp8PairSc = static_cast<size_t>(kI / kFp8Block) * (kH / kFp8Block) * 4;
constexpr size_t kFp8Down = static_cast<size_t>(kH) * kI, kFp8DownSc = static_cast<size_t>(kH / kFp8Block) * (kI / kFp8Block) * 4;
constexpr size_t kF8OffGateSc = align_up(kFp8Pair), kF8OffUp = kF8OffGateSc + align_up(kFp8PairSc), kF8OffUpSc = kF8OffUp + align_up(kFp8Pair),
                 kF8OffDown = kF8OffUpSc + align_up(kFp8PairSc), kF8OffDownSc = kF8OffDown + align_up(kFp8Down),
                 kFp8ExpertBytes = kF8OffDownSc + align_up(kFp8DownSc);

__host__ __device__ Fp8Expert fp8_expert_at(const uint8_t* arena, int e) {
  const uint8_t* b = arena + static_cast<size_t>(e) * kFp8ExpertBytes;
  Fp8Expert x;
  x.gate = b;
  x.gate_scales = reinterpret_cast<const float*>(b + kF8OffGateSc);
  x.up = b + kF8OffUp;
  x.up_scales = reinterpret_cast<const float*>(b + kF8OffUpSc);
  x.down = b + kF8OffDown;
  x.down_scales = reinterpret_cast<const float*>(b + kF8OffDownSc);
  return x;
}
// e4m3 normals (exponent 1..13, no NaN, no subnormals) and F32 scales in [0.5, 2).
__global__ void init_fp8_experts(uint8_t* arena, int experts) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<size_t>(experts) * kFp8ExpertBytes) return;
  const size_t off = i % kFp8ExpertBytes;
  const bool scale = (off >= kF8OffGateSc && off < kF8OffUp) || (off >= kF8OffUpSc && off < kF8OffDown) || off >= kF8OffDownSc;
  unsigned h = static_cast<unsigned>(i * 2654435761ull) ^ static_cast<unsigned>(i >> 17) ^ 0x7f4a7c15u;
  h ^= h >> 13; h *= 0x85ebca6bu; h ^= h >> 16;
  if (!scale) {
    arena[i] = static_cast<uint8_t>((h & 0x80) | (((h >> 8) % 15) << 3) | ((h >> 16) & 7));  // exponents 0..14: subnormals and zeros in, the NaN codes out
    return;
  }
  if ((off & 3) == 0) {
    const size_t w = i & ~size_t{3};
    unsigned hw = static_cast<unsigned>(w * 2654435761ull) ^ 0x1234567u;
    hw ^= hw >> 13; hw *= 0x85ebca6bu; hw ^= hw >> 16;
    const float sv = 0.5f + (hw % 1536) / 1024.0f;
    uint32_t u;
    memcpy(&u, &sv, 4);
    arena[i] = u & 0xff; arena[i + 1] = (u >> 8) & 0xff; arena[i + 2] = (u >> 16) & 0xff; arena[i + 3] = (u >> 24) & 0xff;
  }
}
__global__ void __launch_bounds__(kMxfp4Threads) fp8_gate_up_kernel(const uint8_t* arena, GroupTable t, const uint16_t* x, uint16_t* act) {
  extern __shared__ __align__(128) uint8_t smem[];
  const Mxfp4Group g = group_of(t, blockIdx.y);
  fp8_gate_up<kH, kI, kFp8Block>(fp8_expert_at(arena, t.expert[blockIdx.y]), g, blockIdx.x, x, act, smem);
}
__global__ void __launch_bounds__(kMxfp4Threads) fp8_down_kernel(const uint8_t* arena, GroupTable t, const uint16_t* act, float* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  const Mxfp4Group g = group_of(t, blockIdx.y);
  mxfp4_stage_down_act<kI>(act, g, smem);
  __syncthreads();
  fp8_down<kH, kI, kFp8Block>(fp8_expert_at(arena, t.expert[blockIdx.y]), g, blockIdx.x, smem, out, OutIndex{});
}

struct Fp8Rig {
  std::unique_ptr<Rig> base;  // the slots, groups, x, act and out buffers
  uint8_t* arena = nullptr;
  ~Fp8Rig() { cudaFree(arena); }
};
std::unique_ptr<Fp8Rig> make_fp8_rig(int experts, int rows, int max_group, unsigned seed) {
  auto r = std::make_unique<Fp8Rig>();
  r->base = make_rig(experts, rows, max_group, seed);
  size_t free_b = 0, total = 0;
  DGPP_CUDA_OK(cudaMemGetInfo(&free_b, &total));
  const size_t need = static_cast<size_t>(experts) * kFp8ExpertBytes + (size_t{64} << 20);
  if (free_b < need) dgpp::test::skip("needs " + std::to_string(need >> 20) + " MB free, " + std::to_string(free_b >> 20));
  DGPP_CUDA_OK(cudaMalloc(&r->arena, static_cast<size_t>(experts) * kFp8ExpertBytes));
  init_fp8_experts<<<static_cast<unsigned>((static_cast<size_t>(experts) * kFp8ExpertBytes + 255) / 256), 256>>>(r->arena, experts);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  return r;
}
void run_fp8_consumer(const Fp8Rig& f) {
  const Rig& r = *f.base;
  fp8_gate_up_kernel<<<dim3((kI + kMxfp4UnitRows - 1) / kMxfp4UnitRows, static_cast<unsigned>(r.groups)), kMxfp4Threads, kSmem>>>(f.arena, r.table(), r.x, r.act);
  fp8_down_kernel<<<dim3(kH / kMxfp4UnitRows, static_cast<unsigned>(r.groups)), kMxfp4Threads, kSmem>>>(f.arena, r.table(), r.act, r.out);
}

}  // namespace

DGPP_TEST(fp8_consumer_matches_the_reference) {
  auto f = make_fp8_rig(48, 8, 32, 17);
  const Rig& rig = *f->base;
  DGPP_CUDA_OK(cudaFuncSetAttribute(fp8_gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  DGPP_CUDA_OK(cudaFuncSetAttribute(fp8_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  DGPP_CUDA_OK(cudaMemset(rig.act, 0, rig.act_elems * 2));
  DGPP_CUDA_OK(cudaMemset(rig.out, 0, rig.out_elems * 4));
  run_fp8_consumer(*f);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> act = download(rig.act, rig.act_elems);
  const std::vector<float> out = download(rig.out, rig.out_elems);
  float *gu = nullptr, *gu_mag = nullptr, *abs_sum = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&gu, rig.act_elems * 8));
  DGPP_CUDA_OK(cudaMalloc(&gu_mag, rig.act_elems * 8));
  DGPP_CUDA_OK(cudaMalloc(&abs_sum, rig.out_elems * 4));
  for (int g = 0; g < rig.groups; ++g) {
    Mxfp4Group grp;
    grp.slots = rig.slots + g * kMxfp4GroupRows;
    grp.tokens = rig.tokens + g * kMxfp4GroupRows;
    grp.count = rig.h_count[static_cast<size_t>(g)];
    step_ref::fp8_gate_up<<<(kI + 127) / 128, 128>>>(fp8_expert_at(f->arena, rig.h_expert[static_cast<size_t>(g)]), grp, rig.x, kH, kI, kFp8Block, rig.act, gu, gu_mag);
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> act_r = download(rig.act, rig.act_elems);
  const std::vector<float> gu_h = download(gu, rig.act_elems * 2), gu_m = download(gu_mag, rig.act_elems * 2);
  DGPP_CUDA_OK(cudaMemcpy(rig.act, act.data(), rig.act_elems * 2, cudaMemcpyHostToDevice));  // the down judged on the consumer's act
  for (int g = 0; g < rig.groups; ++g) {
    Mxfp4Group grp;
    grp.slots = rig.slots + g * kMxfp4GroupRows;
    grp.tokens = rig.tokens + g * kMxfp4GroupRows;
    grp.count = rig.h_count[static_cast<size_t>(g)];
    step_ref::fp8_down<<<kH / 128, 128>>>(fp8_expert_at(f->arena, rig.h_expert[static_cast<size_t>(g)]), grp, rig.act, kH, kI, kFp8Block, kSlotsPerToken, rig.out, abs_sum);
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> out_r = download(rig.out, rig.out_elems), mag = download(abs_sum, rig.out_elems);
  cudaFree(gu); cudaFree(gu_mag); cudaFree(abs_sum);
  const std::vector<bool> live = live_slots(rig);
  size_t exact = 0, ok = 0, bad = 0, n = 0, out_bad = 0, out_n = 0;
  for (size_t i = 0; i < act.size(); ++i) {
    if (!live[i / kI]) continue;
    ++n;
    if (act[i] == act_r[i]) ++exact;
    else if (step_ref::act_ok(act[i], act_r[i], gu_h[2 * i], gu_h[2 * i + 1], gu_m[2 * i], gu_m[2 * i + 1])) ++ok;
    else ++bad;
  }
  for (size_t i = 0; i < out.size(); ++i) {
    const int slot = static_cast<int>((i / kSlotsPerToken) / kH) * kSlotsPerToken + static_cast<int>(i % kSlotsPerToken);
    if (!live[static_cast<size_t>(slot)]) continue;
    ++out_n;
    if (!step_ref::partial_ok(out[i], out_r[i], mag[i])) ++out_bad;
  }
  std::printf("   fp8 row-major, %d groups of %d rows: act %zu exact, %zu within the envelope, %zu beyond of %zu; partials %zu beyond the bound of %zu\n",
              rig.groups, rig.rows, exact, ok, bad, n, out_bad, out_n);
  require(bad == 0, "fp8 gate and up outputs beyond the reference's envelope");
  require(out_bad == 0, "fp8 down partials beyond the fp32 bound");
}

DGPP_TEST(fp8_consumer_is_bitwise_across_launches_and_group_sizes) {
  auto grouped = make_fp8_rig(48, 8, 32, 19);
  auto alone = make_fp8_rig(48, 8, 1, 19);
  DGPP_CUDA_OK(cudaFuncSetAttribute(fp8_gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  DGPP_CUDA_OK(cudaFuncSetAttribute(fp8_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  auto run = [&](Fp8Rig& f, uint8_t sentinel) {
    DGPP_CUDA_OK(cudaMemset(f.base->act, sentinel, f.base->act_elems * 2));
    DGPP_CUDA_OK(cudaMemset(f.base->out, sentinel, f.base->out_elems * 4));
    run_fp8_consumer(f);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    return std::make_pair(download(f.base->act, f.base->act_elems), download(f.base->out, f.base->out_elems));
  };
  const auto first = run(*grouped, 0x00), second = run(*grouped, 0x7f);
  // The two rigs hold the same bytes (the same seed and shape): the lone-slot run reads the grouped arena.
  cudaFree(alone->arena);
  alone->arena = grouped->arena;
  const auto solo = run(*alone, 0x00);
  alone->arena = nullptr;
  const std::vector<bool> live = live_slots(*grouped->base);
  auto same_live = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const std::vector<float>& oa, const std::vector<float>& ob) {
    for (size_t i = 0; i < a.size(); ++i)
      if (live[i / kI] && a[i] != b[i]) return false;
    for (size_t i = 0; i < oa.size(); ++i) {
      const int slot = static_cast<int>((i / kSlotsPerToken) / kH) * kSlotsPerToken + static_cast<int>(i % kSlotsPerToken);
      if (live[static_cast<size_t>(slot)] && std::memcmp(&oa[i], &ob[i], 4) != 0) return false;
    }
    return true;
  };
  require(same_live(first.first, second.first, first.second, second.second), "fp8: repeat launches with a changing sentinel differ");
  require(same_live(first.first, solo.first, first.second, solo.second), "fp8: a slot in a group of one differs from the same slot grouped");
  std::printf("   fp8: %d grouped and %d single-slot units agree bitwise\n", grouped->base->groups, alone->base->groups);
}

DGPP_TEST(fp8_consumer_plain_grid_rate) {
  auto f = make_fp8_rig(64, 32, 32, 23);  // 64 pairs of 8.85 MB: cold
  const Rig& rig = *f->base;
  DGPP_CUDA_OK(cudaFuncSetAttribute(fp8_gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  DGPP_CUDA_OK(cudaFuncSetAttribute(fp8_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  cudaEvent_t e0, e1, e2;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  DGPP_CUDA_OK(cudaEventCreate(&e2));
  std::vector<int> distinct;
  for (int e : rig.h_expert) if (std::find(distinct.begin(), distinct.end(), e) == distinct.end()) distinct.push_back(e);
  const double pair_mb = distinct.size() * 2.0 * (kFp8Pair + kFp8PairSc) / 1e6, down_mb = distinct.size() * (kFp8Down + kFp8DownSc) / 1e6;
  for (int rep = 0; rep < 2; ++rep) {
    DGPP_CUDA_OK(cudaEventRecord(e0));
    fp8_gate_up_kernel<<<dim3((kI + kMxfp4UnitRows - 1) / kMxfp4UnitRows, static_cast<unsigned>(rig.groups)), kMxfp4Threads, kSmem>>>(f->arena, rig.table(), rig.x, rig.act);
    DGPP_CUDA_OK(cudaEventRecord(e1));
    fp8_down_kernel<<<dim3(kH / kMxfp4UnitRows, static_cast<unsigned>(rig.groups)), kMxfp4Threads, kSmem>>>(f->arena, rig.table(), rig.act, rig.out);
    DGPP_CUDA_OK(cudaEventRecord(e2));
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    float m1 = 0, m2 = 0;
    DGPP_CUDA_OK(cudaEventElapsedTime(&m1, e0, e1));
    DGPP_CUDA_OK(cudaEventElapsedTime(&m2, e1, e2));
    std::printf("   fp8 plain grids, %s, %zu distinct pairs in %d groups: gate and up %.0f us = %.0f GB/s (%.0f MB) | down %.0f us = %.0f GB/s (%.0f MB)\n",
                rep ? "L2-warm where it fits" : "cold", distinct.size(), rig.groups, m1 * 1e3, pair_mb / m1, pair_mb, m2 * 1e3, down_mb / m2, down_mb);
  }
}

// ---- the MXFP4 pair and down over the checkpoint's row-major layout, through the L2 bridge

namespace {

__global__ void __launch_bounds__(kMxfp4Threads) row_gate_up_kernel(const uint8_t* arena, GroupTable t, const uint16_t* x, uint16_t* act) {
  extern __shared__ __align__(128) uint8_t smem[];
  const Mxfp4Group g = group_of(t, blockIdx.y);
  mxfp4_row_gate_up<kH, kI>(expert_at(arena, t.expert[blockIdx.y]), g, blockIdx.x, x, act, smem);
}
__global__ void __launch_bounds__(kMxfp4Threads) row_down_kernel(const uint8_t* arena, GroupTable t, const uint16_t* act, float* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  const Mxfp4Group g = group_of(t, blockIdx.y);
  mxfp4_stage_down_act<kI>(act, g, smem);
  __syncthreads();
  mxfp4_row_down<kH, kI>(expert_at(arena, t.expert[blockIdx.y]), g, blockIdx.x, smem, out, OutIndex{});
}
void run_row_consumer(const Rig& r) {
  row_gate_up_kernel<<<dim3((kI + kMxfp4UnitRows - 1) / kMxfp4UnitRows, static_cast<unsigned>(r.groups)), kMxfp4Threads, kSmem>>>(
      r.arena, r.table(), r.x, r.act);
  row_down_kernel<<<dim3(kH / kMxfp4UnitRows, static_cast<unsigned>(r.groups)), kMxfp4Threads, kSmem>>>(r.arena, r.table(), r.act, r.out);
}

}  // namespace

DGPP_TEST(mxfp4_row_major_consumer_matches_the_reference) {
  auto rig = make_rig(48, 8, 32, 29);  // the same bytes read as row-major [rows, K / 2] payloads
  DGPP_CUDA_OK(cudaFuncSetAttribute(row_gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  DGPP_CUDA_OK(cudaFuncSetAttribute(row_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  DGPP_CUDA_OK(cudaMemset(rig->act, 0, rig->act_elems * 2));
  DGPP_CUDA_OK(cudaMemset(rig->out, 0, rig->out_elems * 4));
  run_row_consumer(*rig);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> act = download(rig->act, rig->act_elems);
  const std::vector<float> out = download(rig->out, rig->out_elems);
  float *gu = nullptr, *gu_mag = nullptr, *abs_sum = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&gu, rig->act_elems * 8));
  DGPP_CUDA_OK(cudaMalloc(&gu_mag, rig->act_elems * 8));
  DGPP_CUDA_OK(cudaMalloc(&abs_sum, rig->out_elems * 4));
  for (int g = 0; g < rig->groups; ++g) {
    Mxfp4Group grp;
    grp.slots = rig->slots + g * kMxfp4GroupRows;
    grp.tokens = rig->tokens + g * kMxfp4GroupRows;
    grp.count = rig->h_count[static_cast<size_t>(g)];
    step_ref::mxfp4_row_gate_up<<<(kI + 127) / 128, 128>>>(expert_at(rig->arena, rig->h_expert[static_cast<size_t>(g)]), grp, rig->x, kH, kI, rig->act, gu, gu_mag);
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<uint16_t> act_r = download(rig->act, rig->act_elems);
  const std::vector<float> gu_h = download(gu, rig->act_elems * 2), gu_m = download(gu_mag, rig->act_elems * 2);
  DGPP_CUDA_OK(cudaMemcpy(rig->act, act.data(), rig->act_elems * 2, cudaMemcpyHostToDevice));
  for (int g = 0; g < rig->groups; ++g) {
    Mxfp4Group grp;
    grp.slots = rig->slots + g * kMxfp4GroupRows;
    grp.tokens = rig->tokens + g * kMxfp4GroupRows;
    grp.count = rig->h_count[static_cast<size_t>(g)];
    step_ref::mxfp4_row_down<<<kH / 128, 128>>>(expert_at(rig->arena, rig->h_expert[static_cast<size_t>(g)]), grp, rig->act, kH, kI, kSlotsPerToken, rig->out, abs_sum);
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const std::vector<float> out_r = download(rig->out, rig->out_elems), mag = download(abs_sum, rig->out_elems);
  cudaFree(gu); cudaFree(gu_mag); cudaFree(abs_sum);
  const std::vector<bool> live = live_slots(*rig);
  size_t exact = 0, ok = 0, bad = 0, n = 0, out_bad = 0, out_n = 0;
  for (size_t i = 0; i < act.size(); ++i) {
    if (!live[i / kI]) continue;
    ++n;
    if (act[i] == act_r[i]) ++exact;
    else if (step_ref::act_ok(act[i], act_r[i], gu_h[2 * i], gu_h[2 * i + 1], gu_m[2 * i], gu_m[2 * i + 1])) ++ok;
    else ++bad;
  }
  for (size_t i = 0; i < out.size(); ++i) {
    const int slot = static_cast<int>((i / kSlotsPerToken) / kH) * kSlotsPerToken + static_cast<int>(i % kSlotsPerToken);
    if (!live[static_cast<size_t>(slot)]) continue;
    ++out_n;
    if (!step_ref::partial_ok(out[i], out_r[i], mag[i])) ++out_bad;
  }
  std::printf("   MXFP4 row-major, %d groups of %d rows: act %zu exact, %zu within the envelope, %zu beyond of %zu; partials %zu beyond the bound of %zu\n",
              rig->groups, rig->rows, exact, ok, bad, n, out_bad, out_n);
  require(bad == 0, "row-major MXFP4 gate and up outputs beyond the reference's envelope");
  require(out_bad == 0, "row-major MXFP4 down partials beyond the fp32 bound");
}

DGPP_TEST(mxfp4_row_major_consumer_is_bitwise_across_launches_and_group_sizes) {
  auto grouped = make_rig(48, 8, 32, 31);
  auto alone = make_rig(48, 8, 1, 31);
  DGPP_CUDA_OK(cudaFuncSetAttribute(row_gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  DGPP_CUDA_OK(cudaFuncSetAttribute(row_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  auto run = [&](Rig& r, uint8_t sentinel) {
    DGPP_CUDA_OK(cudaMemset(r.act, sentinel, r.act_elems * 2));
    DGPP_CUDA_OK(cudaMemset(r.out, sentinel, r.out_elems * 4));
    run_row_consumer(r);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    return std::make_pair(download(r.act, r.act_elems), download(r.out, r.out_elems));
  };
  const auto first = run(*grouped, 0x00), second = run(*grouped, 0x7f), solo = run(*alone, 0x00);
  const std::vector<bool> live = live_slots(*grouped);
  auto same_live = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const std::vector<float>& oa, const std::vector<float>& ob) {
    for (size_t i = 0; i < a.size(); ++i)
      if (live[i / kI] && a[i] != b[i]) return false;
    for (size_t i = 0; i < oa.size(); ++i) {
      const int slot = static_cast<int>((i / kSlotsPerToken) / kH) * kSlotsPerToken + static_cast<int>(i % kSlotsPerToken);
      if (live[static_cast<size_t>(slot)] && std::memcmp(&oa[i], &ob[i], 4) != 0) return false;
    }
    return true;
  };
  require(same_live(first.first, second.first, first.second, second.second), "row-major MXFP4: repeat launches differ");
  require(same_live(first.first, solo.first, first.second, solo.second), "row-major MXFP4: a lone slot differs from the grouped one");
  std::printf("   MXFP4 row-major: %d grouped and %d single-slot units agree bitwise\n", grouped->groups, alone->groups);
}

DGPP_TEST(mxfp4_row_major_consumer_plain_grid_rate) {
  auto rig = make_rig(192, 32, 32, 37);
  DGPP_CUDA_OK(cudaFuncSetAttribute(row_gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  DGPP_CUDA_OK(cudaFuncSetAttribute(row_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  cudaEvent_t e0, e1, e2;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  DGPP_CUDA_OK(cudaEventCreate(&e2));
  std::vector<int> distinct;
  for (int e : rig->h_expert) if (std::find(distinct.begin(), distinct.end(), e) == distinct.end()) distinct.push_back(e);
  const double pair_mb = distinct.size() * 2.0 * (kPair + kPairScales) / 1e6, down_mb = distinct.size() * (kDown + kDownScales) / 1e6;
  for (int rep = 0; rep < 2; ++rep) {
    DGPP_CUDA_OK(cudaEventRecord(e0));
    row_gate_up_kernel<<<dim3((kI + kMxfp4UnitRows - 1) / kMxfp4UnitRows, static_cast<unsigned>(rig->groups)), kMxfp4Threads, kSmem>>>(
        rig->arena, rig->table(), rig->x, rig->act);
    DGPP_CUDA_OK(cudaEventRecord(e1));
    row_down_kernel<<<dim3(kH / kMxfp4UnitRows, static_cast<unsigned>(rig->groups)), kMxfp4Threads, kSmem>>>(rig->arena, rig->table(), rig->act, rig->out);
    DGPP_CUDA_OK(cudaEventRecord(e2));
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    float m1 = 0, m2 = 0;
    DGPP_CUDA_OK(cudaEventElapsedTime(&m1, e0, e1));
    DGPP_CUDA_OK(cudaEventElapsedTime(&m2, e1, e2));
    std::printf("   MXFP4 row-major plain grids, %s, %zu distinct experts in %d groups: gate and up %.0f us = %.0f GB/s (%.0f MB) | down %.0f us = %.0f GB/s (%.0f MB)\n",
                rep ? "L2-warm where it fits" : "cold", distinct.size(), rig->groups, m1 * 1e3, pair_mb / m1, pair_mb, m2 * 1e3, down_mb / m2, down_mb);
  }
}

// ---- one fp8 projection unit's own rate: a [1280, 5120] matrix for five tokens

namespace {

constexpr int kPK = 5120, kPN = 1280;
constexpr size_t kPBytes = static_cast<size_t>(kPN) * kPK, kPScales = static_cast<size_t>(kPN / 32) * (kPK / 32) * 4;
constexpr size_t kPStride = align_up(kPBytes) + align_up(kPScales);

__global__ void stream_cs_kernel(const uint4* __restrict__ p, size_t n, unsigned* sink) {
  unsigned acc = 0;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const uint4 v = __ldcs(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) *sink = acc;
}
__global__ void stage_cg_kernel(const uint4* __restrict__ p, size_t n, unsigned* sink) {
  unsigned acc = 0;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const uint4 v = __ldcg(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) *sink = acc;
}

// A projection matrix of kPN rows by kPK codes (exponents 0..14, the NaN codes out) with F32 scales in [0.5, 2).
__global__ void init_fp8_matrix(uint8_t* w) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= kPStride) return;
  unsigned h = static_cast<unsigned>(i * 2654435761ull) ^ static_cast<unsigned>(i >> 17) ^ 0x51ed270bu;
  h ^= h >> 13; h *= 0x85ebca6bu; h ^= h >> 16;
  if (i < kPBytes) { w[i] = static_cast<uint8_t>((h & 0x80) | (((h >> 8) % 15) << 3) | ((h >> 16) & 7)); return; }
  const size_t so = i - align_up(kPBytes);
  if (so < kPScales && (so & 3) == 0) {
    const float sc = 0.5f + static_cast<float>((h >> 8) & 0xFFFF) / 65536.0f * 1.5f;
    *reinterpret_cast<float*>(w + i) = sc;
  }
}

// The windowed (shuffled) form of the same unit, for the bitwise comparison.
__global__ void __launch_bounds__(kMxfp4Threads) proj_windowed_kernel(const uint8_t* w, const float* sc, const int32_t* tokens, int count,
                                                                      const uint16_t* x, uint16_t* y) {
  extern __shared__ __align__(128) uint8_t smem[];
  Mxfp4Group g;
  g.slots = tokens;
  g.tokens = tokens;
  g.count = count;
  g.x_stride = kPK;
  fp8_matrix_unit<kPK, kPN, 32, 1>(w, sc, g, blockIdx.x, x, y, nullptr, smem, 0, kPK / kMxfp4WindowCodes, kMxfp4UnitRows);
}

__global__ void __launch_bounds__(kMxfp4Threads) proj_kernel(const uint8_t* arena, int matrices, const int32_t* tokens, int count, const uint16_t* x,
                                                             uint16_t* y, int slabs) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int m = blockIdx.y % matrices;
  const uint8_t* w = arena + static_cast<size_t>(m) * kPStride;
  const float* sc = reinterpret_cast<const float*>(w + align_up(kPBytes));
  Mxfp4Group g;
  g.slots = tokens;
  g.tokens = tokens;
  g.count = count;
  g.x_stride = kPK;
  const int windows = kPK / kMxfp4WindowCodes, per = windows / slabs;
  const int rb = blockIdx.x / slabs, slab = blockIdx.x % slabs;
  fp8_matrix<kPK, kPN, 32>(w, sc, g, rb, x, y + static_cast<size_t>(blockIdx.y) * 32 * kPN, slabs > 1 ? nullptr : nullptr, smem, slab * per, (slab + 1) * per);
}

}  // namespace

DGPP_TEST(fp8_projection_unit_rate) {
  const int matrices = 47;
  uint8_t* arena = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&arena, kPStride * matrices));
  // e4m3 normals and unit scales are enough for a rate.
  DGPP_CUDA_OK(cudaMemset(arena, 0x38, kPStride * matrices));
  uint16_t* x = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&x, 32 * kPK * 2));
  init_x<<<(32 * kPK + 255) / 256, 256>>>(x, 32 * kPK);
  int32_t* tokens = nullptr;
  std::vector<int32_t> h_tok = {0, 1, 2, 3, 4};
  DGPP_CUDA_OK(cudaMalloc(&tokens, 5 * 4));
  DGPP_CUDA_OK(cudaMemcpy(tokens, h_tok.data(), 20, cudaMemcpyHostToDevice));
  uint16_t* y = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&y, static_cast<size_t>(matrices) * 32 * kPN * 2));
  constexpr int kProjSmem = kFp8StagedSmemCap;
  DGPP_CUDA_OK(cudaFuncSetAttribute(proj_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kProjSmem));
  cudaEvent_t e0, e1;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  proj_kernel<<<dim3(kPN / 128, 1), kMxfp4Threads, kProjSmem>>>(arena, matrices, tokens, 5, x, y, 1);  // the first launch's costs
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (int slabs : {1, 4}) {
    for (int mats : {1, 8, 47}) {
      const dim3 grid((kPN / 128) * slabs, mats);
      std::vector<float> ms(3);
      for (float& m : ms) {
        DGPP_CUDA_OK(cudaEventRecord(e0));
        proj_kernel<<<grid, kMxfp4Threads, kProjSmem>>>(arena, matrices, tokens, 5, x, y, slabs);
        DGPP_CUDA_OK(cudaEventRecord(e1));
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        DGPP_CUDA_OK(cudaEventElapsedTime(&m, e0, e1));
      }
      std::sort(ms.begin(), ms.end());
      const double mb = static_cast<double>(mats) * (kPBytes + kPScales) / 1e6;
      const int blocks = static_cast<int>(grid.x * grid.y);
      const double per_block = mb / ms[1] / std::min(blocks, 48);
      std::printf("   fp8 [1280 x 5120] x %2d matrices, %d slab(s), %4d blocks: %6.0f us (median of 3) = %5.1f GB/s aggregate, %5.1f GB/s an SM\n",
                  mats, slabs, blocks, ms[1] * 1e3, mb / ms[1], per_block);
    }
  }
  // The lead's staging as the streamer does it: the matrix read into L2 with cache-global loads by 47 blocks after a
  // 150 MB evict-first stream, then the unit over it once. Hot means the staged lines serve the consumer.
  {
    uint4* flushbuf = nullptr;
    const size_t flush_n = (150u << 20) / 16;
    DGPP_CUDA_OK(cudaMalloc(&flushbuf, flush_n * 16));
    DGPP_CUDA_OK(cudaMemset(flushbuf, 3, flush_n * 16));
    unsigned* sink = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&sink, 4));
    for (int staged = 0; staged < 3; ++staged) {
      stream_cs_kernel<<<47, 256>>>(flushbuf, flush_n, sink);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      if (staged == 1) stage_cg_kernel<<<47, 256>>>(reinterpret_cast<const uint4*>(arena), kPStride / 16, sink);
      if (staged == 2) stream_cs_kernel<<<47, 256>>>(reinterpret_cast<const uint4*>(arena), kPStride / 16, sink);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaEventRecord(e0));
      proj_kernel<<<dim3(kPN / 128, 1), kMxfp4Threads, kProjSmem>>>(arena, 1, tokens, 5, x, y, 1);
      DGPP_CUDA_OK(cudaEventRecord(e1));
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      float ms = 0;
      DGPP_CUDA_OK(cudaEventElapsedTime(&ms, e0, e1));
      std::printf("   fp8 [1280 x 5120] x 1, 1 slab, 10 blocks, %s: %6.0f us = %5.1f GB/s aggregate\n",
                  staged == 0 ? "cold after a 150 MB stream" : staged == 1 ? "staged by cache-global loads (the lead's)" : "staged by evict-first loads", ms * 1e3,
                  (kPBytes + kPScales) / (ms * 1e6));
    }
    cudaFree(flushbuf); cudaFree(sink);
  }
  // The same matrix eight times, L2-hot: the unit's compute rate, at one block an SM with the one-slab slice
  // staged and at the blocks the smaller four-slab slice lets the registers and shared memory co-reside.
  for (int slabs : {1, 4}) {
    const int smem = slabs == 1 ? kProjSmem : fp8_staged_smem_bytes<kPK, 32>(kPK / slabs);
    const dim3 grid((kPN / 128) * slabs, 8);
    std::vector<float> ms(3);
    for (float& m : ms) {
      DGPP_CUDA_OK(cudaEventRecord(e0));
      proj_kernel<<<grid, kMxfp4Threads, smem>>>(arena, 1, tokens, 5, x, y, slabs);
      DGPP_CUDA_OK(cudaEventRecord(e1));
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaEventElapsedTime(&m, e0, e1));
    }
    std::sort(ms.begin(), ms.end());
    const double mb = 8.0 * (kPBytes + kPScales) / 1e6;
    int per_sm = 0;
    DGPP_CUDA_OK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, proj_kernel, kMxfp4Threads, smem));
    std::printf("   fp8 [1280 x 5120] hot x 8, %d slab(s), %4d blocks, %d a SM by occupancy, %d KB smem: %6.0f us = %5.1f GB/s aggregate\n",
                slabs, static_cast<int>(grid.x * grid.y), per_sm, smem / 1024, ms[1] * 1e3, mb / ms[1]);
  }
  cudaFree(arena); cudaFree(x); cudaFree(tokens); cudaFree(y);
}

// ---- the split form of one fp8 matrix against the reference

namespace {

__global__ void __launch_bounds__(kMxfp4Threads) split_kernel(const uint8_t* w, const float* sc, const int32_t* tokens, int count, const uint16_t* x,
                                                              uint16_t* y) {
  extern __shared__ __align__(128) uint8_t smem[];
  Mxfp4Group g;
  g.slots = tokens;
  g.tokens = tokens;
  g.count = count;
  g.x_stride = kPK;
  fp8_matrix_split_unit<kPK, kPN, 32, 2>(w, sc, g, blockIdx.x, x, y, smem);
}

}  // namespace

DGPP_TEST(fp8_split_form_matches_the_reference) {
  uint8_t* arena = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&arena, kPStride));
  init_fp8_matrix<<<static_cast<unsigned>((kPStride + 255) / 256), 256>>>(arena);
  uint16_t* x = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&x, 32 * kPK * 2));
  init_x<<<(32 * kPK + 255) / 256, 256>>>(x, 32 * kPK);
  int32_t* tokens = nullptr;
  std::vector<int32_t> h_tok = {0, 1, 2, 3, 4};
  DGPP_CUDA_OK(cudaMalloc(&tokens, 5 * 4));
  DGPP_CUDA_OK(cudaMemcpy(tokens, h_tok.data(), 20, cudaMemcpyHostToDevice));
  const size_t elems = 5 * static_cast<size_t>(kPN);
  uint16_t *y = nullptr, *y_ref = nullptr;
  float *yf = nullptr, *mag = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&y, elems * 2));
  DGPP_CUDA_OK(cudaMalloc(&y_ref, elems * 2));
  DGPP_CUDA_OK(cudaMalloc(&yf, elems * 4));
  DGPP_CUDA_OK(cudaMalloc(&mag, elems * 4));
  DGPP_CUDA_OK(cudaMemset(y, 0, elems * 2));
  const uint8_t* w = arena;
  const float* sc = reinterpret_cast<const float*>(arena + align_up(kPBytes));
  constexpr int smem = fp8_split_smem_bytes<kPK, 2>();
  DGPP_CUDA_OK(cudaFuncSetAttribute(split_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  split_kernel<<<kPN / 32, kMxfp4Threads, smem>>>(w, sc, tokens, 5, x, y);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  Mxfp4Group g;
  g.slots = tokens;
  g.tokens = tokens;
  g.count = 5;
  step_ref::fp8_matrix<<<(kPN + 127) / 128, 128>>>(w, sc, g, x, kPK, kPK, kPN, 32, y_ref, yf, mag);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(elems), ref(elems);
  std::vector<float> exact(elems), m(elems);
  DGPP_CUDA_OK(cudaMemcpy(got.data(), y, elems * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(ref.data(), y_ref, elems * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(exact.data(), yf, elems * 4, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(m.data(), mag, elems * 4, cudaMemcpyDeviceToHost));
  size_t ok = 0, bad = 0, shown = 0;
  for (size_t i = 0; i < elems; ++i) {
    if (got[i] == ref[i] || step_ref::value_ok(got[i], exact[i], m[i])) ++ok;
    else {
      ++bad;
      if (shown++ < 6)
        std::printf("      y[slot %zu][row %zu]: got %g ref %g (magnitude %g)\n", i / kPN, i % kPN, bf16_bits_to_float(got[i]), exact[i], m[i]);
    }
  }
  std::printf("   fp8 split form [1280 x 5120], 5 tokens: %zu ok, %zu beyond of %zu\n", ok, bad, elems);
  cudaFree(arena); cudaFree(x); cudaFree(tokens); cudaFree(y); cudaFree(y_ref); cudaFree(yf); cudaFree(mag);
  require(bad == 0, "the split form disagrees with the reference");
}

// ---- the one-token-group (staged) form of one fp8 matrix against the reference, and bitwise across launches

DGPP_TEST(fp8_staged_form_matches_the_reference) {
  uint8_t* arena = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&arena, kPStride));
  init_fp8_matrix<<<static_cast<unsigned>((kPStride + 255) / 256), 256>>>(arena);
  uint16_t* x = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&x, 32 * kPK * 2));
  init_x<<<(32 * kPK + 255) / 256, 256>>>(x, 32 * kPK);
  int32_t* tokens = nullptr;
  std::vector<int32_t> h_tok = {3, 0, 4, 1, 2};
  DGPP_CUDA_OK(cudaMalloc(&tokens, 5 * 4));
  DGPP_CUDA_OK(cudaMemcpy(tokens, h_tok.data(), 20, cudaMemcpyHostToDevice));
  const size_t elems = 5 * static_cast<size_t>(kPN);
  uint16_t *y = nullptr, *y2 = nullptr, *y_ref = nullptr;
  float *yf = nullptr, *mag = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&y, elems * 2));
  DGPP_CUDA_OK(cudaMalloc(&y2, elems * 2));
  DGPP_CUDA_OK(cudaMalloc(&y_ref, elems * 2));
  DGPP_CUDA_OK(cudaMalloc(&yf, elems * 4));
  DGPP_CUDA_OK(cudaMalloc(&mag, elems * 4));
  DGPP_CUDA_OK(cudaMemset(y, 0, elems * 2));
  DGPP_CUDA_OK(cudaMemset(y2, 0xff, elems * 2));
  const uint8_t* w = arena;
  const float* sc = reinterpret_cast<const float*>(arena + align_up(kPBytes));
  constexpr int smem = kFp8StagedSmemCap;
  static_assert(fp8_staged_smem_bytes<kPK, 32>(kPK) <= kFp8StagedSmemCap, "the [1280 x 5120] unit takes the staged form at one slab");
  DGPP_CUDA_OK(cudaFuncSetAttribute(proj_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  proj_kernel<<<dim3(kPN / 128, 1), kMxfp4Threads, smem>>>(arena, 1, tokens, 5, x, y, 1);
  proj_kernel<<<dim3(kPN / 128, 1), kMxfp4Threads, smem>>>(arena, 1, tokens, 5, x, y2, 1);
  uint16_t* y_win = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&y_win, elems * 2));
  DGPP_CUDA_OK(cudaMemset(y_win, 0, elems * 2));
  DGPP_CUDA_OK(cudaFuncSetAttribute(proj_windowed_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  proj_windowed_kernel<<<kPN / 128, kMxfp4Threads, kSmem>>>(w, sc, tokens, 5, x, y_win);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  Mxfp4Group g;
  g.slots = tokens;
  g.tokens = tokens;
  g.count = 5;
  step_ref::fp8_matrix<<<(kPN + 127) / 128, 128>>>(w, sc, g, x, kPK, kPK, kPN, 32, y_ref, yf, mag);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(elems), again(elems), ref(elems);
  std::vector<float> exact(elems), m(elems);
  DGPP_CUDA_OK(cudaMemcpy(got.data(), y, elems * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(again.data(), y2, elems * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(ref.data(), y_ref, elems * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(exact.data(), yf, elems * 4, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(m.data(), mag, elems * 4, cudaMemcpyDeviceToHost));
  std::vector<uint16_t> win(elems);
  DGPP_CUDA_OK(cudaMemcpy(win.data(), y_win, elems * 2, cudaMemcpyDeviceToHost));
  size_t ok = 0, bad = 0, shown = 0, differ = 0, differ_win = 0;
  for (size_t i = 0; i < elems; ++i) {
    if (got[i] != again[i]) ++differ;
    if (got[i] != win[i]) ++differ_win;
    if (got[i] == ref[i] || step_ref::value_ok(got[i], exact[i], m[i])) ++ok;
    else {
      ++bad;
      if (shown++ < 6)
        std::printf("      y[slot %zu][row %zu]: got %g ref %g (magnitude %g)\n", i / kPN, i % kPN, bf16_bits_to_float(got[i]), exact[i], m[i]);
    }
  }
  std::printf("   fp8 staged form [1280 x 5120], 5 tokens in slot order 3 0 4 1 2: %zu ok, %zu beyond of %zu, %zu differ between launches, %zu from the windowed form\n",
              ok, bad, elems, differ, differ_win);
  cudaFree(arena); cudaFree(x); cudaFree(tokens); cudaFree(y); cudaFree(y2); cudaFree(y_win); cudaFree(y_ref); cudaFree(yf); cudaFree(mag);
  require(bad == 0, "the staged form disagrees with the reference");
  require(differ == 0, "the staged form is not bitwise across launches");
  require(differ_win == 0, "the staged form is not bitwise the windowed form");
}

// ---- the hyper-connection site against the production kernels

namespace {
constexpr int kMhcH = 5120, kMhcT = 5;
__global__ void mhc_dots_test_kernel(const uint16_t* streams, const uint16_t* fn, float* logits, int hidden, float eps) {
  __shared__ float scratch[8];
  const int c = blockIdx.x, t = blockIdx.y;
  const float d = mhc_dot(streams + static_cast<size_t>(t) * 4 * hidden, fn + static_cast<size_t>(c) * 4 * hidden, hidden, eps, scratch);
  if (threadIdx.x == 0) logits[t * kMhcCoeffs + c] = d;
}
__global__ void mhc_finish_test_kernel(const uint16_t* streams, const float* logits, MhcSite site, int hidden, const float* pre_in, float* pre_out,
                                       float* post, float* comb, uint16_t* normed) {
  __shared__ float smem[kMhcN + kMhcN * kMhcN + 8];
  const int t = blockIdx.x;
  mhc_finish(streams + static_cast<size_t>(t) * 4 * hidden, logits + t * kMhcCoeffs, site, hidden, pre_in + t * 4, pre_out + t * 4, post + t * 4,
             comb + t * 16, normed + static_cast<size_t>(t) * hidden, smem);
}
__global__ void mhc_update_test_kernel(const float* post, const float* comb, const uint16_t* h, const uint16_t* in, uint16_t* out, int hidden, int total) {
  const int i0 = blockIdx.x * 512, i1 = min(i0 + 512, total);
  mhc_update_chunk(post, comb, h, in, out, hidden, i0, i1);
}
__global__ void init_f32(float* p, int n, float lo, float span, unsigned seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned hh = static_cast<unsigned>(i) * 2654435761u + seed; hh ^= hh >> 15; hh *= 0x2c1b3c6du; hh ^= hh >> 12;
  p[i] = lo + (hh & 0xffff) / 65536.0f * span;
}
template <typename T> size_t count_diff(const T* a, const T* b, size_t n) {
  std::vector<T> ha(n), hb(n);
  DGPP_CUDA_OK(cudaMemcpy(ha.data(), a, n * sizeof(T), cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(hb.data(), b, n * sizeof(T), cudaMemcpyDeviceToHost));
  size_t d = 0;
  for (size_t i = 0; i < n; ++i) if (std::memcmp(&ha[i], &hb[i], sizeof(T)) != 0) ++d;
  return d;
}
}  // namespace

// The site unit on its own: 40 blocks (five tokens by eight units of three
// coefficients), each over 40 KB of streams and 120 KB of coefficient rows,
// cold after a 150 MB stream and with the rows in L2 as the lead stages them.
namespace {
__global__ void mhc_unit_rate_kernel(const uint16_t* streams, const uint16_t* fn, float* logits, int hidden, float eps, int units_per_token) {
  __shared__ float scratch[8];
  const int t = blockIdx.x / units_per_token, u = blockIdx.x % units_per_token, c0 = u * kMhcUnitCoeffs;
  float lg[kMhcUnitCoeffs];
  mhc_dot_n<kMhcUnitCoeffs>(streams + static_cast<size_t>(t) * 4 * hidden, fn + static_cast<size_t>(c0) * 4 * hidden, hidden, eps, scratch, lg);
  if (threadIdx.x < kMhcUnitCoeffs) logits[t * kMhcCoeffs + c0 + threadIdx.x] = lg[threadIdx.x];
}
}  // namespace

DGPP_TEST(mhc_unit_rate) {
  const int H = kMhcH, T = kMhcT, units = kMhcCoeffs / kMhcUnitCoeffs;
  uint16_t *streams = nullptr, *fn = nullptr;
  float* lg = nullptr;
  const size_t sn = static_cast<size_t>(T) * 4 * H, fnn = static_cast<size_t>(kMhcCoeffs) * 4 * H;
  DGPP_CUDA_OK(cudaMalloc(&streams, sn * 2));
  DGPP_CUDA_OK(cudaMalloc(&fn, fnn * 2));
  DGPP_CUDA_OK(cudaMalloc(&lg, T * kMhcCoeffs * 4));
  init_x<<<static_cast<unsigned>((sn + 255) / 256), 256>>>(streams, static_cast<int>(sn));
  init_x<<<static_cast<unsigned>((fnn + 255) / 256), 256>>>(fn, static_cast<int>(fnn));
  uint4* flushbuf = nullptr;
  const size_t flush_n = (150u << 20) / 16;
  DGPP_CUDA_OK(cudaMalloc(&flushbuf, flush_n * 16));
  DGPP_CUDA_OK(cudaMemset(flushbuf, 3, flush_n * 16));
  unsigned* sink = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&sink, 4));
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  DGPP_CUDA_OK(cudaEventCreate(&e0));
  DGPP_CUDA_OK(cudaEventCreate(&e1));
  mhc_unit_rate_kernel<<<T * units, kMhcThreads>>>(streams, fn, lg, H, 1e-6f, units);  // the first launch's costs
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const double unit_kb = (4.0 * H * 2 + kMhcUnitCoeffs * 4.0 * H * 2) / 1e3;
  for (int mode = 0; mode < 3; ++mode) {
    std::vector<float> ms(3);
    for (float& m : ms) {
      stream_cs_kernel<<<47, 256>>>(flushbuf, flush_n, sink);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      if (mode == 1) stage_cg_kernel<<<47, 256>>>(reinterpret_cast<const uint4*>(fn), fnn * 2 / 16, sink);
      if (mode == 2) { stage_cg_kernel<<<47, 256>>>(reinterpret_cast<const uint4*>(fn), fnn * 2 / 16, sink); stage_cg_kernel<<<47, 256>>>(reinterpret_cast<const uint4*>(streams), sn * 2 / 16, sink); }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaEventRecord(e0));
      mhc_unit_rate_kernel<<<T * units, kMhcThreads>>>(streams, fn, lg, H, 1e-6f, units);
      DGPP_CUDA_OK(cudaEventRecord(e1));
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaEventElapsedTime(&m, e0, e1));
    }
    std::sort(ms.begin(), ms.end());
    std::printf("   mhc unit (3 coefficients, 160 KB), %d blocks, %s: %5.1f us (median of 3) = %5.1f GB/s a block\n", T * units,
                mode == 0 ? "cold after a 150 MB stream" : mode == 1 ? "coefficient rows staged in L2" : "rows and streams staged in L2", ms[1] * 1e3,
                unit_kb / ms[1] / 1e3);
  }
  // One coefficient a unit (mhc_dot, 120 blocks) and the finish alone (five blocks), rows in L2, for the unit shape's choice.
  {
    float *pre_in = nullptr, *pre_out = nullptr, *post = nullptr, *comb = nullptr, *base = nullptr, *scale = nullptr;
    uint16_t *ln = nullptr, *normed = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&pre_in, T * 4 * 4)); DGPP_CUDA_OK(cudaMalloc(&pre_out, T * 4 * 4)); DGPP_CUDA_OK(cudaMalloc(&post, T * 4 * 4));
    DGPP_CUDA_OK(cudaMalloc(&comb, T * 16 * 4)); DGPP_CUDA_OK(cudaMalloc(&base, kMhcCoeffs * 4)); DGPP_CUDA_OK(cudaMalloc(&scale, 3 * 4));
    DGPP_CUDA_OK(cudaMalloc(&ln, H * 2)); DGPP_CUDA_OK(cudaMalloc(&normed, static_cast<size_t>(T) * H * 2));
    DGPP_CUDA_OK(cudaMemset(pre_in, 0, T * 4 * 4));
    init_f32<<<1, 32>>>(base, kMhcCoeffs, -1.f, 2.f, 11u);
    init_f32<<<1, 32>>>(scale, 3, -1.5f, 3.f, 13u);
    init_x<<<(H + 255) / 256, 256>>>(ln, H);
    MhcSite site; site.base = base; site.scale = scale; site.ln = ln; site.hc_eps = 1e-6f; site.sinkhorn_iters = 20; site.ln_eps = 1e-6f;
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    for (int which = 0; which < 2; ++which) {
      std::vector<float> ms(3);
      for (float& m : ms) {
        stream_cs_kernel<<<47, 256>>>(flushbuf, flush_n, sink);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        stage_cg_kernel<<<47, 256>>>(reinterpret_cast<const uint4*>(fn), fnn * 2 / 16, sink);
        stage_cg_kernel<<<47, 256>>>(reinterpret_cast<const uint4*>(streams), sn * 2 / 16, sink);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        DGPP_CUDA_OK(cudaEventRecord(e0));
        if (which == 0) mhc_dots_test_kernel<<<dim3(kMhcCoeffs, T), kMhcThreads>>>(streams, fn, lg, H, 1e-6f);
        else mhc_finish_test_kernel<<<T, kMhcThreads>>>(streams, lg, site, H, pre_in, pre_out, post, comb, normed);
        DGPP_CUDA_OK(cudaEventRecord(e1));
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        DGPP_CUDA_OK(cudaEventElapsedTime(&m, e0, e1));
      }
      std::sort(ms.begin(), ms.end());
      if (which == 0) std::printf("   mhc unit (1 coefficient, 80 KB), %d blocks, rows and streams staged in L2: %5.1f us (median of 3)\n", kMhcCoeffs * T, ms[1] * 1e3);
      else std::printf("   mhc finish alone, %d blocks, streams staged in L2: %5.1f us (median of 3)\n", T, ms[1] * 1e3);
    }
    cudaFree(pre_in); cudaFree(pre_out); cudaFree(post); cudaFree(comb); cudaFree(base); cudaFree(scale); cudaFree(ln); cudaFree(normed);
  }
  cudaFree(streams); cudaFree(fn); cudaFree(lg); cudaFree(flushbuf); cudaFree(sink);
}

DGPP_TEST(mhc_site_matches_the_production_kernels) {
  const int H = kMhcH, T = kMhcT;
  uint16_t *streams = nullptr, *fn = nullptr, *ln = nullptr, *h = nullptr, *normed_p = nullptr, *normed_s = nullptr, *out_p = nullptr, *out_s = nullptr, *post_bf16 = nullptr, *comb_bf16 = nullptr;
  float *base = nullptr, *scale = nullptr, *pre_in = nullptr, *pre_p = nullptr, *pre_s = nullptr, *post_p = nullptr, *post_s = nullptr, *comb_p = nullptr, *comb_s = nullptr, *lg_p = nullptr, *lg_s = nullptr;
  int* counters = nullptr;
  const size_t sn = static_cast<size_t>(T) * 4 * H;
  DGPP_CUDA_OK(cudaMalloc(&streams, sn * 2)); DGPP_CUDA_OK(cudaMalloc(&fn, static_cast<size_t>(kMhcCoeffs) * 4 * H * 2)); DGPP_CUDA_OK(cudaMalloc(&ln, H * 2));
  DGPP_CUDA_OK(cudaMalloc(&h, static_cast<size_t>(T) * H * 2)); DGPP_CUDA_OK(cudaMalloc(&normed_p, static_cast<size_t>(T) * H * 2)); DGPP_CUDA_OK(cudaMalloc(&normed_s, static_cast<size_t>(T) * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&out_p, sn * 2)); DGPP_CUDA_OK(cudaMalloc(&out_s, sn * 2)); DGPP_CUDA_OK(cudaMalloc(&post_bf16, T * 4 * 2)); DGPP_CUDA_OK(cudaMalloc(&comb_bf16, T * 16 * 2));
  DGPP_CUDA_OK(cudaMalloc(&base, kMhcCoeffs * 4)); DGPP_CUDA_OK(cudaMalloc(&scale, 3 * 4)); DGPP_CUDA_OK(cudaMalloc(&pre_in, T * 4 * 4));
  DGPP_CUDA_OK(cudaMalloc(&pre_p, T * 4 * 4)); DGPP_CUDA_OK(cudaMalloc(&pre_s, T * 4 * 4)); DGPP_CUDA_OK(cudaMalloc(&post_p, T * 4 * 4)); DGPP_CUDA_OK(cudaMalloc(&post_s, T * 4 * 4));
  DGPP_CUDA_OK(cudaMalloc(&comb_p, T * 16 * 4)); DGPP_CUDA_OK(cudaMalloc(&comb_s, T * 16 * 4)); DGPP_CUDA_OK(cudaMalloc(&lg_p, T * kMhcCoeffs * 4)); DGPP_CUDA_OK(cudaMalloc(&lg_s, T * kMhcCoeffs * 4));
  DGPP_CUDA_OK(cudaMalloc(&counters, T * 4)); DGPP_CUDA_OK(cudaMemset(counters, 0, T * 4));
  init_x<<<static_cast<unsigned>((sn + 255) / 256), 256>>>(streams, static_cast<int>(sn));
  init_x<<<static_cast<unsigned>((static_cast<size_t>(kMhcCoeffs) * 4 * H + 255) / 256), 256>>>(fn, kMhcCoeffs * 4 * H);
  init_x<<<static_cast<unsigned>((static_cast<size_t>(T) * H + 255) / 256), 256>>>(h, T * H);
  init_f32<<<1, 32>>>(base, kMhcCoeffs, -1.f, 2.f, 11u);
  init_f32<<<1, 32>>>(scale, 3, -1.5f, 3.f, 13u);
  {  // ln around one, pre_in one-hot (the first site's seed, as the model's)
    std::vector<uint16_t> hl(H);
    for (int i = 0; i < H; ++i) hl[static_cast<size_t>(i)] = float_to_bf16_bits(0.9f + 0.2f * ((i * 7919) % 101) / 100.0f);
    DGPP_CUDA_OK(cudaMemcpy(ln, hl.data(), H * 2, cudaMemcpyHostToDevice));
    std::vector<float> oh(static_cast<size_t>(T) * 4, 0.f);
    for (int t = 0; t < T; ++t) oh[static_cast<size_t>(t) * 4] = 1.f;
    DGPP_CUDA_OK(cudaMemcpy(pre_in, oh.data(), oh.size() * 4, cudaMemcpyHostToDevice));
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  // The production site: the fused decode form with the one-rounding norm, the single pass, the deferred comb.
  GlmMhcConfig cfg;
  cfg.hc_mult = 4; cfg.hidden = H; cfg.sinkhorn_iters = 20; cfg.hc_eps = 1e-6f; cfg.norm_eps = 1e-6f;
  GlmMhcWeights w; w.fn = fn; w.base = base; w.scale = scale;
  MhcSinglePass sp; sp.pre_in = pre_in; sp.pre_out = pre_p; sp.post_f32 = post_p; sp.comb_f32 = nullptr; sp.one_round_norm = true;
  const float ln_eps = 1e-6f;
  const bool deferred = launch_mhc_compute_normed(streams, w, cfg, nullptr, post_bf16, comb_bf16, lg_p, ln, normed_p, ln_eps, T, nullptr, counters, true, &sp, true);
  require(deferred, "the production site defers its comb");
  launch_mhc_comb(lg_p, w, cfg, comb_bf16, T, nullptr, comb_p);
  launch_mhc_stream_update_f32(post_p, comb_p, h, streams, out_p, cfg, T, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  // The step's units.
  mhc_dots_test_kernel<<<dim3(kMhcCoeffs, T), kMhcThreads>>>(streams, fn, lg_s, H, cfg.norm_eps);
  MhcSite site; site.base = base; site.scale = scale; site.ln = ln; site.hc_eps = cfg.hc_eps; site.sinkhorn_iters = cfg.sinkhorn_iters; site.ln_eps = ln_eps;
  mhc_finish_test_kernel<<<T, kMhcThreads>>>(streams, lg_s, site, H, pre_in, pre_s, post_s, comb_s, normed_s);
  mhc_update_test_kernel<<<static_cast<unsigned>((T * H + 511) / 512), kMhcThreads>>>(post_s, comb_s, h, streams, out_s, H, T * H);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const size_t d_lg = count_diff(lg_p, lg_s, static_cast<size_t>(T) * kMhcCoeffs), d_pre = count_diff(pre_p, pre_s, static_cast<size_t>(T) * 4),
               d_post = count_diff(post_p, post_s, static_cast<size_t>(T) * 4), d_comb = count_diff(comb_p, comb_s, static_cast<size_t>(T) * 16),
               d_norm = count_diff(normed_p, normed_s, static_cast<size_t>(T) * H), d_out = count_diff(out_p, out_s, sn);
  std::printf("   mhc site, %d tokens: logits %zu differ of %d, pre %zu, post %zu, comb %zu, normed %zu of %d, updated streams %zu of %zu\n", T, d_lg,
              T * kMhcCoeffs, d_pre, d_post, d_comb, d_norm, T * H, d_out, sn);
  require(d_lg == 0 && d_pre == 0 && d_post == 0 && d_comb == 0, "the step's coefficients differ from the production site's");
  require(d_norm == 0, "the step's normed rows differ from the production site's");
  require(d_out == 0, "the step's stream update differs from the production kernel's");
}

int main() { return dgpp::test::run_all(); }
