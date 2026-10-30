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
#include "kernels/step/mxfp4_expert.cuh"

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

int main() { return dgpp::test::run_all(); }
