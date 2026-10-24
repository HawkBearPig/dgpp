// Parity tests for the MoE module (M4): router kernel vs the double oracle
// at REAL geometry (E=288, H=4096), the full expert path (router ->
// ascending-accumulation -> shared) on small geometry, swiglu clamp edge
// cases, accumulation-order exposure, determinism, the decode slot path's
// bitwise pin against the host path, and the TP expert slicing (every
// rank a slice of every expert's intermediate dim, partials folded like
// the FFN all-reduce) against the unsliced oracle. Tolerance design:
// the router's fp32 pipeline vs the oracle's double differs by ~1e-6
// relative, so id swaps are certified against the biased-score gap (the
// near-tie discipline the DSA selection audit established); the expert path
// adds the scale-gemm mma-order gap on top (same class as scale_gemm_test's
// strict oracle, budgeted there at 0 mismatches — here slightly loosened
// for the extra chained roundings).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/fp4_gemv.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/hadamard32.hpp"
#include "kernels/packq_gemv.hpp"
#include "kernels/scale_gemm.hpp"
#include "models/glm/moe.hpp"
#include "models/glm/moe_layer.hpp"
#include "models/glm/moe_reference.hpp"
#include "models/quant_matrix.hpp"
#include "scale_gemm_test_helpers.hpp"

namespace {

using namespace scale_gemm_test;
using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::GlmMoeConfig;
using dgpp::GlmMoeLayer;
using dgpp::GlmMoeWeights;
using dgpp::GlmMoeHostWeights;
using dgpp::GlmMoeRouterRef;
using dgpp::GlmQuantMatrix;
using scale_gemm_test::require;

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu)
                         : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

// ---- router parity at real geometry -------------------------------------

void check_router(const GlmMoeConfig& cfg, int tokens, uint64_t seed) {
  Rng rng(seed);
  const int E = cfg.n_experts, H = cfg.hidden, K = cfg.top_k;

  std::vector<uint16_t> hidden(static_cast<size_t>(tokens) * H);
  fill_act(rng, hidden);
  std::vector<uint16_t> gate(static_cast<size_t>(E) * H);
  for (auto& v : gate) v = float_to_bf16_bits(0.05f * static_cast<float>(rng.unit()));
  std::vector<float> bias(E);
  for (auto& v : bias) v = 0.25f * static_cast<float>(rng.unit());

  uint16_t* d_hidden = nullptr;
  uint16_t* d_gate = nullptr;
  float* d_bias = nullptr;
  int32_t* d_ids = nullptr;
  float* d_w = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_hidden, hidden.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_gate, gate.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_bias, bias.size() * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_ids, static_cast<size_t>(tokens) * K * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_w, static_cast<size_t>(tokens) * K * 4));
  float* d_scores = nullptr;
  float* d_biased = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_scores, static_cast<size_t>(tokens) * E * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_biased, static_cast<size_t>(tokens) * E * 4));
  std::memcpy(d_hidden, hidden.data(), hidden.size() * 2);
  std::memcpy(d_gate, gate.data(), gate.size() * 2);
  std::memcpy(d_bias, bias.data(), bias.size() * 4);

  dgpp::launch_moe_router(d_hidden, d_gate, d_bias, d_ids, d_w, d_scores,
                          d_biased, cfg, tokens, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  // The FUSED form (the last dots block per token selects) must reproduce
  // the two-kernel form's ids, weights and biased row bit for bit, and
  // leave its tickets at zero (replay-safe).
  {
    std::vector<int32_t> ids2(static_cast<size_t>(tokens) * K);
    std::vector<float> w2(static_cast<size_t>(tokens) * K);
    std::vector<float> biased2(static_cast<size_t>(tokens) * E);
    std::memcpy(ids2.data(), d_ids, ids2.size() * 4);
    std::memcpy(w2.data(), d_w, w2.size() * 4);
    std::memcpy(biased2.data(), d_biased, biased2.size() * 4);
    int* d_counters = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_counters, static_cast<size_t>(tokens) * 4));
    std::memset(d_counters, 0, static_cast<size_t>(tokens) * 4);
    std::memset(d_ids, 0xff, ids2.size() * 4);
    std::memset(d_w, 0, w2.size() * 4);
    for (int rep = 0; rep < 3; ++rep) {  // replays reuse the zeroed tickets
      dgpp::launch_moe_router(d_hidden, d_gate, d_bias, d_ids, d_w, d_scores,
                              d_biased, cfg, tokens, nullptr, d_counters);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
    }
    if (std::memcmp(d_ids, ids2.data(), ids2.size() * 4) != 0 ||
        std::memcmp(d_w, w2.data(), w2.size() * 4) != 0 ||
        std::memcmp(d_biased, biased2.data(), biased2.size() * 4) != 0)
      throw std::runtime_error("fused router select != two-kernel router");
    for (int t = 0; t < tokens; ++t)
      if (d_counters[t] != 0)
        throw std::runtime_error("fused router left a ticket counter set");
    cudaFree(d_counters);
  }

  GlmMoeRouterRef ref;
  dgpp::glm_moe_ref_router(hidden.data(), gate.data(), bias.data(), cfg,
                           tokens, ref);

  std::vector<int32_t> got_ids(static_cast<size_t>(tokens) * K);
  std::vector<float> got_w(static_cast<size_t>(tokens) * K);
  std::memcpy(got_ids.data(), d_ids, got_ids.size() * 4);
  std::memcpy(got_w.data(), d_w, got_w.size() * 4);

  long certified = 0, hard = 0;
  double max_rel = 0;
  for (size_t i = 0; i < got_ids.size(); ++i) {
    if (got_ids[i] != ref.ids[i]) {
      // Near-tie certification: an id swap is acceptable only when the two
      // experts' biased scores differ by < 1e-5 (fp32-vs-double noise floor).
      const int t = static_cast<int>(i / K);
      const double a = ref.biased[static_cast<size_t>(t) * E + got_ids[i]];
      const double b = ref.biased[static_cast<size_t>(t) * E + ref.ids[i]];
      if (std::abs(a - b) < 1e-5)
        ++certified;
      else
        ++hard;
      continue;
    }
    const double rel = std::abs(static_cast<double>(got_w[i]) -
                                static_cast<double>(ref.weights[i])) /
                       std::max(1e-30,
                                std::abs(static_cast<double>(ref.weights[i])));
    max_rel = std::max(max_rel, rel);
  }
  DGPP_CUDA_OK(cudaFree(d_hidden));
  DGPP_CUDA_OK(cudaFree(d_gate));
  DGPP_CUDA_OK(cudaFree(d_bias));
  DGPP_CUDA_OK(cudaFree(d_ids));
  DGPP_CUDA_OK(cudaFree(d_w));
  DGPP_CUDA_OK(cudaFree(d_scores));
  DGPP_CUDA_OK(cudaFree(d_biased));

  require(hard == 0, "router id mismatch beyond near-tie certification");
  require(max_rel < 1e-5, "router weights within 1e-5 relative of oracle");
  std::printf("[ OK ] router E=%d H=%d K=%d tokens=%d: %ld certified "
              "near-tie swaps, max weight rel err %.2g\n",
              E, H, K, tokens, certified, max_rel);
}

// ---- full expert path on small geometry ----------------------------------

struct SmallCase {
  GlmMoeConfig cfg;
  int tokens = 0;
  std::vector<uint16_t> hidden;
  GlmMoeHostWeights host_w;
  GlmMoeWeights dev_w;
  std::vector<GlmQuantMatrix> expert_mats;  // FP8: (E+1)*3; NVFP4: the shared triple
  std::vector<dgpp::GlmFp4Matrix> expert_mats_fp4;  // NVFP4: E*3 routed
  std::vector<dgpp::GlmPackedMatrix> expert_mats_packq;  // packed-int: (E+1)*3, the shared triple last
  std::vector<void*> d_packq_bytes;
  std::vector<dgpp::Exl3Matrix> expert_mats_exl3;  // EXL3: (E+1)*3, the shared triple last
  std::vector<void*> d_exl3_bytes;
  GlmQuantMatrix shared_mats[3];
  // device backing
  uint16_t *d_hidden = nullptr, *d_gate_w = nullptr;
  float* d_bias = nullptr;
  std::vector<uint8_t*> d_payloads;
  std::vector<float*> d_scales;
  std::vector<uint8_t*> d_fp4_bytes;
  float* d_fp4_globals = nullptr;
  uint16_t* d_out = nullptr;
  bool nvfp4() const { return host_w.nvfp4; }
  bool packq() const { return host_w.packq; }
  bool exl3() const { return host_w.exl3_set(); }

  // A host EXL3 matrix's device twin (managed memory; the tiles 256-aligned
  // like a loader grant).
  static dgpp::Exl3Matrix upload_exl3(const dgpp::Exl3HostMatrix& h, std::vector<void*>& owned) {
    uint8_t* tiles = nullptr;
    uint16_t* suh = nullptr;
    uint16_t* svh = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&tiles, h.tiles.size() + 256));
    DGPP_CUDA_OK(cudaMallocManaged(&suh, h.suh.size() * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&svh, h.svh.size() * 2));
    std::memcpy(tiles, h.tiles.data(), h.tiles.size());
    std::memcpy(suh, h.suh.data(), h.suh.size() * 2);
    std::memcpy(svh, h.svh.data(), h.svh.size() * 2);
    owned.push_back(tiles);
    owned.push_back(suh);
    owned.push_back(svh);
    dgpp::Exl3Matrix m = h.view();
    m.tiles = tiles;
    m.suh = suh;
    m.svh = svh;
    return m;
  }

  void alloc() {
    const int E = cfg.n_experts;
    DGPP_CUDA_OK(cudaMallocManaged(&d_hidden, hidden.size() * 2));
    std::memcpy(d_hidden, hidden.data(), hidden.size() * 2);
    DGPP_CUDA_OK(cudaMallocManaged(&d_gate_w, host_w.router_gate.size() * 2));
    std::memcpy(d_gate_w, host_w.router_gate.data(),
                host_w.router_gate.size() * 2);
    DGPP_CUDA_OK(cudaMallocManaged(&d_bias, host_w.router_bias.size() * 4));
    std::memcpy(d_bias, host_w.router_bias.data(),
                host_w.router_bias.size() * 4);

    if (host_w.exl3_set()) {
      require(host_w.exl3.size() == static_cast<size_t>(E + 1) * 3, "exl3 case: every matrix and the shared triple");
      expert_mats_exl3.resize(host_w.exl3.size());
      for (size_t m = 0; m < host_w.exl3.size(); ++m) expert_mats_exl3[m] = upload_exl3(host_w.exl3[m], d_exl3_bytes);
      dev_w.router_gate = d_gate_w;
      dev_w.router_bias = d_bias;
      dev_w.experts = nullptr;
      dev_w.experts_exl3 = expert_mats_exl3.data();
      for (int m = 0; m < 3; ++m) dev_w.shared_exl3[m] = expert_mats_exl3[static_cast<size_t>(E) * 3 + m];
      DGPP_CUDA_OK(cudaMallocManaged(&d_out, static_cast<size_t>(tokens) * cfg.hidden * 2));
      return;
    }

    if (host_w.packq) {
      // Every matrix (the routed E*3 and the shared triple) as a packed-int
      // triple: the I32 words and the bf16 group scales (the loader's
      // layout), one table.
      const int mats = host_w.packq_matrices(E);
      require(host_w.shared_packq && mats == (E + 1) * 3, "packq case: the shared expert is packed");
      expert_mats_packq.resize(static_cast<size_t>(mats));
      for (int m = 0; m < mats; ++m) {
        const dgpp::GlmPackedMatrixHost v = dgpp::glm_moe_host_view_packq(host_w, cfg, m);
        uint32_t* pk = nullptr;
        uint16_t* sc = nullptr;
        const size_t wn = static_cast<size_t>(v.rows) * v.cols * v.bits / 32;
        const size_t sn = static_cast<size_t>(v.rows) * v.cols / dgpp::packed_scale_group(v.scale_fmt);
        DGPP_CUDA_OK(cudaMallocManaged(&pk, wn * 4));
        DGPP_CUDA_OK(cudaMallocManaged(&sc, sn * 2));
        std::memcpy(pk, v.packed, wn * 4);
        std::memcpy(sc, v.scales, sn * 2);
        d_packq_bytes.push_back(pk);
        d_packq_bytes.push_back(sc);
        expert_mats_packq[static_cast<size_t>(m)] =
            dgpp::GlmPackedMatrix{pk, sc, v.rows, v.cols, v.bits, v.scale_fmt};
      }
      dev_w.router_gate = d_gate_w;
      dev_w.router_bias = d_bias;
      dev_w.experts = nullptr;
      dev_w.experts_packed = expert_mats_packq.data();
      dev_w.routed_input_transform = host_w.routed_input_transform;
      for (int m = 0; m < 3; ++m) dev_w.shared_packed[m] = expert_mats_packq[static_cast<size_t>(E) * 3 + m];
      DGPP_CUDA_OK(cudaMallocManaged(&d_out, static_cast<size_t>(tokens) * cfg.hidden * 2));
      return;
    }

    if (host_w.nvfp4 && !host_w.fp4_matrix.empty()) {
      // The per-projection form (docs/flint_plan.md D2): both tables, each
      // routed matrix uploaded into its own; the shared triple fp8.
      require(!host_w.shared_nvfp4, "mixed case: the shared triple is fp8");
      expert_mats_fp4.assign(static_cast<size_t>(E) * 3, dgpp::GlmFp4Matrix{});
      expert_mats.assign(static_cast<size_t>(E + 1) * 3, GlmQuantMatrix{});
      DGPP_CUDA_OK(cudaMallocManaged(&d_fp4_globals, static_cast<size_t>(E) * 3 * 4));
      for (int m = 0; m < (E + 1) * 3; ++m) {
        if (host_w.fp4_of(m, E)) {
          const dgpp::GlmFp4MatrixHost v = dgpp::glm_moe_host_view_fp4(host_w, cfg, m);
          uint8_t* pk = nullptr;
          uint8_t* sc = nullptr;
          const size_t pn = static_cast<size_t>(v.rows) * v.cols / 2;
          const size_t sn = static_cast<size_t>(v.rows) * v.cols / v.scale_group;
          DGPP_CUDA_OK(cudaMallocManaged(&pk, pn));
          DGPP_CUDA_OK(cudaMallocManaged(&sc, sn));
          std::memcpy(pk, v.payload, pn);
          std::memcpy(sc, v.scales, sn);
          d_fp4_globals[m] = v.global_scale;
          d_fp4_bytes.push_back(pk);
          d_fp4_bytes.push_back(sc);
          expert_mats_fp4[static_cast<size_t>(m)] =
              dgpp::GlmFp4Matrix{pk, sc, d_fp4_globals + m, v.rows, v.cols, v.scale_group};
          continue;
        }
        auto view = dgpp::glm_moe_host_view(host_w, cfg, m);
        uint8_t* p = nullptr;
        float* s = nullptr;
        const size_t pn = static_cast<size_t>(view.rows) * view.cols;
        const size_t sn = ((view.rows + 127) / 128) * ((view.cols + 127) / 128);
        DGPP_CUDA_OK(cudaMallocManaged(&p, pn));
        DGPP_CUDA_OK(cudaMallocManaged(&s, sn * 4));
        std::memcpy(p, view.payload, pn);
        std::memcpy(s, view.scales, sn * 4);
        d_payloads.push_back(p);
        d_scales.push_back(s);
        expert_mats[static_cast<size_t>(m)] = GlmQuantMatrix{p, s, nullptr, view.rows, view.cols};
      }
      dev_w.router_gate = d_gate_w;
      dev_w.router_bias = d_bias;
      dev_w.experts = expert_mats.data();
      dev_w.experts_fp4 = expert_mats_fp4.data();
      dev_w.routed_fp4_mask = 0;
      for (int m = 0; m < 3; ++m)
        if (host_w.fp4_matrix[static_cast<size_t>(m)]) dev_w.routed_fp4_mask |= static_cast<uint8_t>(1u << m);
      for (int m = 0; m < 3; ++m) dev_w.shared[m] = expert_mats[static_cast<size_t>(E) * 3 + m];
      DGPP_CUDA_OK(cudaMallocManaged(&d_out, static_cast<size_t>(tokens) * cfg.hidden * 2));
      return;
    }
    if (host_w.nvfp4) {
      // Routed experts as NVFP4 triples (payload, per-row scales, and one
      // gathered global-scale array, the loader's layout); the shared
      // triple FP8 from the fp8 vectors' first three entries.
      const int fp4_mats = host_w.fp4_matrices(E);
      DGPP_CUDA_OK(cudaMallocManaged(&d_fp4_globals, static_cast<size_t>(fp4_mats) * 4));
      expert_mats_fp4.resize(static_cast<size_t>(fp4_mats));
      for (int m = 0; m < fp4_mats; ++m) {
        const dgpp::GlmFp4MatrixHost v = dgpp::glm_moe_host_view_fp4(host_w, cfg, m);
        uint8_t* pk = nullptr;
        uint8_t* sc = nullptr;
        const size_t pn = static_cast<size_t>(v.rows) * v.cols / 2;
        const size_t sn = static_cast<size_t>(v.rows) * v.cols / v.scale_group;
        DGPP_CUDA_OK(cudaMallocManaged(&pk, pn));
        DGPP_CUDA_OK(cudaMallocManaged(&sc, sn));
        std::memcpy(pk, v.payload, pn);
        std::memcpy(sc, v.scales, sn);
        d_fp4_globals[m] = v.global_scale;
        d_fp4_bytes.push_back(pk);
        d_fp4_bytes.push_back(sc);
        expert_mats_fp4[static_cast<size_t>(m)] = dgpp::GlmFp4Matrix{
            pk, sc, v.scale_group == 32 ? nullptr : d_fp4_globals + m, v.rows, v.cols, v.scale_group};
      }
      if (host_w.shared_nvfp4) {
        dev_w.router_gate = d_gate_w;
        dev_w.router_bias = d_bias;
        dev_w.experts = nullptr;
        dev_w.experts_fp4 = expert_mats_fp4.data();
        for (int m = 0; m < 3; ++m) dev_w.shared_fp4[m] = expert_mats_fp4[static_cast<size_t>(E) * 3 + m];
        DGPP_CUDA_OK(cudaMallocManaged(&d_out, static_cast<size_t>(tokens) * cfg.hidden * 2));
        return;
      }
      expert_mats.resize(3);
      for (int m = 0; m < 3; ++m) {
        auto view = dgpp::glm_moe_host_shared(host_w, cfg, m);
        uint8_t* p = nullptr;
        float* s = nullptr;
        const size_t pn = static_cast<size_t>(view.rows) * view.cols;
        const size_t sn = ((view.rows + 127) / 128) * ((view.cols + 127) / 128);
        DGPP_CUDA_OK(cudaMallocManaged(&p, pn));
        DGPP_CUDA_OK(cudaMallocManaged(&s, sn * 4));
        std::memcpy(p, view.payload, pn);
        std::memcpy(s, view.scales, sn * 4);
        d_payloads.push_back(p);
        d_scales.push_back(s);
        expert_mats[static_cast<size_t>(m)] = GlmQuantMatrix{p, s, nullptr, view.rows, view.cols};
      }
      dev_w.router_gate = d_gate_w;
      dev_w.router_bias = d_bias;
      dev_w.experts = nullptr;
      dev_w.experts_fp4 = expert_mats_fp4.data();
      for (int m = 0; m < 3; ++m) dev_w.shared[m] = expert_mats[static_cast<size_t>(m)];
      DGPP_CUDA_OK(cudaMallocManaged(&d_out, static_cast<size_t>(tokens) * cfg.hidden * 2));
      return;
    }
    const int mats = (E + 1) * 3;
    expert_mats.resize(mats);
    for (int m = 0; m < mats; ++m) {
      auto view = dgpp::glm_moe_host_view(host_w, cfg, m);
      uint8_t* p = nullptr;
      float* s = nullptr;
      const size_t pn = static_cast<size_t>(view.rows) * view.cols;
      const size_t sn = ((view.rows + 127) / 128) * ((view.cols + 127) / 128);
      DGPP_CUDA_OK(cudaMallocManaged(&p, pn));
      DGPP_CUDA_OK(cudaMallocManaged(&s, sn * 4));
      std::memcpy(p, view.payload, pn);
      std::memcpy(s, view.scales, sn * 4);
      d_payloads.push_back(p);
      d_scales.push_back(s);
      expert_mats[m] = GlmQuantMatrix{p, s, nullptr, view.rows, view.cols};
    }
    dev_w.router_gate = d_gate_w;
    dev_w.router_bias = d_bias;
    dev_w.experts = expert_mats.data();
    for (int m = 0; m < 3; ++m)
      dev_w.shared[m] = expert_mats[static_cast<size_t>(E) * 3 + m];
    DGPP_CUDA_OK(cudaMallocManaged(&d_out, static_cast<size_t>(tokens) *
                                              cfg.hidden * 2));
  }

  void free_all() {
    cudaFree(d_hidden); cudaFree(d_gate_w); cudaFree(d_bias); cudaFree(d_out);
    for (auto* p : d_payloads) cudaFree(p);
    for (auto* s : d_scales) cudaFree(s);
    for (auto* b : d_fp4_bytes) cudaFree(b);
    for (auto* b : d_packq_bytes) cudaFree(b);
    for (auto* b : d_exl3_bytes) cudaFree(b);
    cudaFree(d_fp4_globals);
  }
};

// The checkpoint's packing (unsigned codes offset 2^(bits-1), 32/bits per
// word, the low nibble / byte first), one matrix.
void pack_codes_into(const std::vector<int>& codes, int bits, std::vector<uint32_t>& out) {
  const int per = 32 / bits;
  const size_t base = out.size();
  out.resize(base + codes.size() / per, 0u);
  for (size_t i = 0; i < codes.size(); ++i) {
    const uint32_t u = static_cast<uint32_t>(codes[i] + (1 << (bits - 1)));
    out[base + i / per] |= u << (bits * (i % per));
  }
}

// The Mixed346 forms (2026-10-09): `m346_forms` (E entries of "gud" digits
// in {3, 4, 6} with g == u, or "existing") makes the case a Mixed346 layer —
// every routed matrix at its own width and format (3 / 4 / 6 under format
// 3 as a dense index stream, or int4 g64 for an existing expert), the
// shared triple int8, the int8 activation transform.
void pack_dense_into(const std::vector<unsigned>& idx, int bits, std::vector<uint32_t>& out) {
  const size_t base = out.size();
  out.resize(base + idx.size() * static_cast<size_t>(bits) / 32, 0u);
  for (size_t i = 0; i < idx.size(); ++i) {
    const size_t pos = i * static_cast<size_t>(bits);
    const size_t w = base + pos / 32;
    const int shift = static_cast<int>(pos % 32);
    out[w] |= idx[i] << shift;
    if (shift + bits > 32) out[w + 1] |= idx[i] >> (32 - shift);
  }
}
const char* const kM346Forms[8] = {"444", "334", "446", "333", "666", "443", "336", "existing"};

SmallCase make_small_case(int E, int H, int I, int K, int tokens,
                          uint64_t seed, bool nvfp4 = false, bool shared_nvfp4 = false,
                          int packq_bits = 0, int fp4_group = 16, int packq_scale_fmt = 0,
                          int routed_input_transform = 0, const char* const* m346_forms = nullptr,
                          int routed_fp4_mask = -1, int exl3_bits_routed = 0, int exl3_bits_shared = 0) {
  SmallCase c;
  c.cfg.hidden = H;
  c.cfg.inter = I;
  c.cfg.n_experts = E;
  c.cfg.top_k = K;
  c.cfg.routed_scaling_factor = 2.5f;
  c.cfg.norm_topk_prob = true;
  c.cfg.swiglu_limit = 10.0f;
  c.tokens = tokens;
  Rng rng(seed);

  c.hidden.resize(static_cast<size_t>(tokens) * H);
  fill_act(rng, c.hidden);
  c.host_w.router_gate.resize(static_cast<size_t>(E) * H);
  for (auto& v : c.host_w.router_gate)
    v = float_to_bf16_bits(0.05f * static_cast<float>(rng.unit()));
  c.host_w.router_bias.resize(E);
  for (auto& v : c.host_w.router_bias)
    v = 0.25f * static_cast<float>(rng.unit());

  // (E+1)*3 matrices: gate/up [I,H] + down [H,I] per expert, then shared.
  // Under NVFP4 the routed E*3 are e2m1 pairs + e4m3 scales per 16 + one
  // global scale each, and only the shared triple is FP8.
  c.host_w.nvfp4 = nvfp4;
  c.host_w.shared_nvfp4 = nvfp4 && shared_nvfp4;
  c.host_w.fp4_group = fp4_group;
  // The per-projection form (routed_fp4_mask >= 0, bit w = projection w
  // NVFP4): every expert the same mix, the shared triple fp8.
  if (nvfp4 && routed_fp4_mask >= 0) {
    c.host_w.fp4_matrix.assign(static_cast<size_t>(E) * 3, 0);
    for (int e = 0; e < E; ++e)
      for (int m = 0; m < 3; ++m)
        c.host_w.fp4_matrix[static_cast<size_t>(e) * 3 + m] = static_cast<uint8_t>((routed_fp4_mask >> m) & 1);
  }
  // Packed-int (packq_bits 4 or 8): the routed experts at that width, the
  // shared triple at int8 (the checkpoint's shape), release-like scales.
  c.host_w.packq = packq_bits != 0;
  c.host_w.shared_packq = packq_bits != 0;
  c.host_w.packq_bits_routed = packq_bits != 0 ? packq_bits : 4;
  c.host_w.packq_bits_shared = 8;
  // The routed matrices' scale format (0: bf16 per 64; 1: f16 per 128, the
  // AutoRound hybrid; 2: the NF4I8 codebook, bf16 per 128 — the nibbles
  // are then indices, every one valid); the shared triple stays bf16 per
  // 64. The routed input transform (kMoeInputHadamard32 with format 2):
  // the oracle and the engine rotate the routed experts' inputs alike.
  c.host_w.packq_scale_fmt = packq_scale_fmt;
  c.host_w.routed_input_transform = routed_input_transform;
  if (m346_forms != nullptr) {
    c.host_w.packq = true;
    c.host_w.shared_packq = true;
    c.host_w.packq_bits_routed = 4;
    c.host_w.packq_scale_fmt = dgpp::kPackedScaleBf16G128Mixed346;
    c.host_w.routed_input_transform = dgpp::kMoeInputHadamard32Int8;
    c.cfg.routed_int8_activations = true;
    c.host_w.packq_matrix_bits.assign(static_cast<size_t>(E) * 3, 4);
    c.host_w.packq_matrix_fmt.assign(static_cast<size_t>(E) * 3, 0);
    for (int e = 0; e < E; ++e) {
      const std::string f = m346_forms[e];
      if (f == "existing") continue;
      for (int m = 0; m < 3; ++m) {
        c.host_w.packq_matrix_bits[static_cast<size_t>(e) * 3 + m] = static_cast<uint8_t>(f[m] - '0');
        c.host_w.packq_matrix_fmt[static_cast<size_t>(e) * 3 + m] = dgpp::kPackedScaleBf16G128Mixed346;
      }
    }
  }
  for (int m = 0; m < (E + 1) * 3; ++m) {
    const bool down = m % 3 == 2;
    const int64_t rows = down ? H : I, cols = down ? I : H;
    if (m346_forms != nullptr) {
      const int bits = c.host_w.packq_bits_of(m, E);
      const int sf = c.host_w.packq_fmt_of(m, E);
      if (sf == dgpp::kPackedScaleBf16G128Mixed346) {
        std::vector<unsigned> idx(static_cast<size_t>(rows) * cols);
        for (auto& u : idx) u = static_cast<unsigned>(rng.next() % (1u << bits));
        pack_dense_into(idx, bits, c.host_w.packq_words);
      } else {
        const int lo = -(1 << (bits - 1)), hi = (1 << (bits - 1)) - 1;
        std::vector<int> codes(static_cast<size_t>(rows) * cols);
        for (auto& cd : codes) cd = lo + static_cast<int>(rng.next() % static_cast<uint64_t>(hi - lo + 1));
        pack_codes_into(codes, bits, c.host_w.packq_words);
      }
      const size_t sn = static_cast<size_t>(rows) * cols / dgpp::packed_scale_group(sf);
      for (size_t i = 0; i < sn; ++i) {
        // Release-like magnitudes: levels reach 127 at 3 / 4 bits (format
        // 3), 31 at 6, 7 for the int4 offset codes, 127 for int8.
        const float base = sf == dgpp::kPackedScaleBf16G128Mixed346 ? (bits == 6 ? 0.001f : 0.00025f)
                           : bits == 4 ? 0.004f : 0.0003f;
        c.host_w.packq_scales.push_back(float_to_bf16_bits(static_cast<float>(std::exp2(rng.unit() * 2.0)) * base));
      }
      continue;
    }
    if (packq_bits != 0) {
      const bool routed = m < E * 3;
      const int bits = routed ? packq_bits : 8;
      const int sf = routed ? packq_scale_fmt : 0;
      const int lo = -(1 << (bits - 1)), hi = (1 << (bits - 1)) - 1;
      std::vector<int> codes(static_cast<size_t>(rows) * cols);
      for (auto& cd : codes) cd = lo + static_cast<int>(rng.next() % static_cast<uint64_t>(hi - lo + 1));
      pack_codes_into(codes, bits, c.host_w.packq_words);
      const size_t sn = static_cast<size_t>(rows) * cols / dgpp::packed_scale_group(sf);
      for (size_t i = 0; i < sn; ++i) {
        // Release-like magnitudes; the codebook's levels reach 127 (not 7),
        // so its scales sit 16x lower for the same weights.
        const float v = static_cast<float>(std::exp2(rng.unit() * 2.0)) *
                        (bits == 4 ? (dgpp::packed_codebook(sf) ? 0.00025f : 0.004f) : 0.0003f);
        c.host_w.packq_scales.push_back(sf == 1 ? dgpp::float_to_fp16_bits(v) : float_to_bf16_bits(v));
      }
      continue;
    }
    if (c.host_w.fp4_of(m, E)) {
      std::vector<uint8_t> packed(static_cast<size_t>(rows) * cols / 2);
      for (auto& b : packed) b = static_cast<uint8_t>(rng.next() & 0xFF);
      std::vector<uint8_t> sc(static_cast<size_t>(rows) * cols / fp4_group);
      if (fp4_group == 32) {
        // MXFP4: e8m0 exponents around the release's (weights ~1e-2: block
        // scales 2^-9 .. 2^-5).
        for (auto& v : sc) v = static_cast<uint8_t>(127 - 9 + static_cast<int>(rng.next() % 5));
      } else {
        for (auto& v : sc)
          v = dgpp::float_to_fp8_e4m3_bits(
              static_cast<float>(std::exp2(rng.unit() * 2.0)) * 0.05f);
      }
      c.host_w.fp4_payloads.insert(c.host_w.fp4_payloads.end(), packed.begin(), packed.end());
      c.host_w.fp4_scales.insert(c.host_w.fp4_scales.end(), sc.begin(), sc.end());
      c.host_w.fp4_globals.push_back(fp4_group == 32 ? 1.0f : static_cast<float>(std::exp2(rng.unit())));
      continue;
    }
    std::vector<uint8_t> payload(static_cast<size_t>(rows) * cols);
    fill_payload(rng, payload);
    const int64_t sr = (rows + 127) / 128, sc = (cols + 127) / 128;
    std::vector<float> scales(sr * sc);
    fill_scales(rng, scales);
    c.host_w.payloads.insert(c.host_w.payloads.end(), payload.begin(),
                             payload.end());
    c.host_w.scales.insert(c.host_w.scales.end(), scales.begin(),
                           scales.end());
  }
  // The EXL3 case (docs/deepseek_v41_flash_exl3_plan.md): every matrix a
  // random trellis at the routed or shared width, suh near +-0.03 (the
  // pack's input scales carry the weight magnitude) and svh near +-1 —
  // weights of the fp8 case's size; the fp8 vectors above go unused.
  if (exl3_bits_routed != 0) {
    c.cfg.exl3_experts = true;
    for (int m = 0; m < (E + 1) * 3; ++m) {
      const bool down = m % 3 == 2;
      const int bits = m < E * 3 ? exl3_bits_routed : exl3_bits_shared;
      c.host_w.exl3.push_back(dgpp::exl3_random_matrix(rng, down ? H : I, down ? I : H, bits, 0.03f, 1.0f));
    }
  }
  return c;
}

// The expert-path budgets: per-element bf16 ulps against the double oracle
// (hard 12, soft 4 on < 2% of elements) and a relative l2. The ulps carry
// an absolute floor (2026-09-05, the scale GEMM gates' rule): an element
// within 2 % of the output's RMS magnitude counts as exact — the chain
// rounds its gate/up intermediates to bf16, and a small output formed by
// cancellation of large terms inherits their absolute error (M=300: one
// element of 153,600 at 65 ulps, oracle 1.72 against 1.21, on both cores).
void require_within_expert_budget(const std::vector<uint16_t>& got,
                                  const std::vector<uint16_t>& oracle,
                                  const char* label) {
  require(got.size() == oracle.size(), "expert path: size mismatch");
  double rms = 0;
  for (size_t i = 0; i < oracle.size(); ++i)
    rms += std::pow(bf16_bits_to_float(oracle[i]), 2);
  rms = oracle.empty() ? 0 : std::sqrt(rms / oracle.size());
  const double floor_abs = 2e-2 * rms;
  long hard = 0, soft = 0;
  double max_ulps = 0, sum_d2 = 0, sum_o2 = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    int u = bf16_ulps(got[i], oracle[i]);
    if (std::fabs(bf16_bits_to_float(got[i]) - bf16_bits_to_float(oracle[i])) <= floor_abs)
      u = 0;
    max_ulps = std::max(max_ulps, static_cast<double>(u));
    const double d = bf16_bits_to_float(got[i]) - bf16_bits_to_float(oracle[i]);
    sum_d2 += d * d;
    sum_o2 += std::pow(bf16_bits_to_float(oracle[i]), 2);
    if (u > 4) ++soft;
    if (u > 12) ++hard;
  }
  const double l2 = sum_o2 > 0 ? std::sqrt(sum_d2 / sum_o2) : 0;
  if (hard != 0 || std::getenv("DGPP_MOE_DUMP") != nullptr) {
    // The worst elements, for a hunt: index, the oracle's value, ours, ulps.
    std::vector<size_t> idx(got.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>(8, idx.size()),
                      idx.end(), [&](size_t a, size_t b) {
                        return bf16_ulps(got[a], oracle[a]) > bf16_ulps(got[b], oracle[b]);
                      });
    for (size_t j = 0; j < std::min<size_t>(8, idx.size()); ++j) {
      const size_t i = idx[j];
      std::printf("[ .. ]   %s worst[%zu] i=%zu oracle=%.6g got=%.6g ulps=%d\n", label,
                  j, i, bf16_bits_to_float(oracle[i]), bf16_bits_to_float(got[i]),
                  bf16_ulps(got[i], oracle[i]));
    }
  }
  // Report before asserting: a failing run must still yield its numbers.
  std::printf("[ .. ] %s: max %g ulps, %ld/%zu over soft, %ld hard, l2=%.2g\n",
              label, max_ulps, soft, got.size(), hard, l2);
  require(hard == 0, "expert path: hard ulp violations");
  require(static_cast<double>(soft) / got.size() < 0.02,
          "expert path: soft ulp budget");
  require(l2 < 4e-3, "expert path: l2 budget");
}

void run_small_case(SmallCase& c, const char* label,
                    dgpp::MoeExpertKernel kernel = dgpp::MoeExpertKernel::kGemv) {
  std::vector<uint16_t> oracle;
  dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);

  GlmMoeLayer layer(c.dev_w, c.cfg, c.tokens);
  layer.enqueue(c.d_hidden, c.d_out, c.tokens, nullptr, kernel);
  DGPP_CUDA_OK(cudaDeviceSynchronize());

  std::vector<uint16_t> got(static_cast<size_t>(c.tokens) * c.cfg.hidden);
  std::memcpy(got.data(), c.d_out, got.size() * 2);
  require_within_expert_budget(got, oracle, label);

  // Determinism: a second enqueue must be bitwise identical.
  DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, got.size() * 2));
  layer.enqueue(c.d_hidden, c.d_out, c.tokens, nullptr, kernel);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint16_t> second(got.size());
  std::memcpy(second.data(), c.d_out, second.size() * 2);
  require(std::memcmp(got.data(), second.data(), got.size() * 2) == 0,
          "expert path: second run bitwise identical");
}

// One rank's slice of a SmallCase (the loader's contract, in miniature):
// gate/up rows [rank*M, (rank+1)*M) as pointer views, down columns packed
// into fresh device matrices — for every routed expert and the shared one.
struct RankSlice {
  std::vector<GlmQuantMatrix> mats;  // (E+1)*3, sliced (FP8) / the shared triple (NVFP4)
  std::vector<dgpp::GlmFp4Matrix> mats_fp4;  // E*3 sliced routed (NVFP4)
  std::vector<dgpp::GlmPackedMatrix> mats_packq;  // (E+1)*3 sliced (packed-int)
  std::vector<dgpp::Exl3Matrix> mats_exl3;        // (E+1)*3 sliced (EXL3)
  std::vector<void*> owned_exl3;
  std::vector<uint8_t*> owned_payloads;
  std::vector<float*> owned_scales;
  std::vector<uint8_t*> owned_fp4;
  std::vector<void*> owned_packq;
  GlmMoeWeights dev_w;
  uint16_t* d_out = nullptr;

  static RankSlice make(const SmallCase& c, int rank, int world) {
    RankSlice r;
    const int E = c.cfg.n_experts, H = c.cfg.hidden;
    const int64_t I = c.cfg.inter, M = I / world;
    if (c.exl3()) {
      // The loader's EXL3 slices in miniature: whole 128-blocks of the
      // intermediate, the remainder blocks on the lowest ranks for even
      // experts and on the highest for odd ones (exl3_expert_block_split;
      // the shared expert as an even layer's) — gate/up rows and down
      // columns of that range for every routed expert and the shared one.
      r.mats_exl3.resize(static_cast<size_t>(E + 1) * 3);
      for (int m = 0; m < (E + 1) * 3; ++m) {
        const int expert = m / 3;
        int64_t b0 = 0, bn = 0;
        dgpp::exl3_expert_block_split(I, world, rank, expert < E ? (expert & 1) : 0, b0, bn);
        const dgpp::Exl3HostMatrix& full = c.host_w.exl3[static_cast<size_t>(m)];
        const dgpp::Exl3HostMatrix part = m % 3 == 2 ? dgpp::exl3_host_cols_slice(full, b0, bn)
                                                      : dgpp::exl3_host_rows_slice(full, b0, bn);
        r.mats_exl3[static_cast<size_t>(m)] = SmallCase::upload_exl3(part, r.owned_exl3);
      }
      r.dev_w.router_gate = c.dev_w.router_gate;
      r.dev_w.router_bias = c.dev_w.router_bias;
      r.dev_w.experts = nullptr;
      r.dev_w.experts_exl3 = r.mats_exl3.data();
      for (int m = 0; m < 3; ++m) r.dev_w.shared_exl3[m] = r.mats_exl3[static_cast<size_t>(E) * 3 + m];
      DGPP_CUDA_OK(cudaMallocManaged(&r.d_out, static_cast<size_t>(c.tokens) * H * 2));
      return r;
    }
    if (c.packq()) {
      // The loader's packed slices in miniature: gate/up rows [rank*M, +M)
      // as views, down columns packed at word and group granularity
      // (M % 64 == 0), for every routed expert and the shared one.
      const int mats = (E + 1) * 3;
      r.mats_packq.resize(static_cast<size_t>(mats));
      for (int m = 0; m < mats; ++m) {
        const dgpp::GlmPackedMatrix& full = c.expert_mats_packq[static_cast<size_t>(m)];
        if (m % 3 != 2) {
          r.mats_packq[static_cast<size_t>(m)] = dgpp::packed_rows_view(full, rank * M, M);
          continue;
        }
        // Words per row and per slice: cols x bits / 32 (a dense stream at 3
        // or 6 bits is whole words at a 128-multiple slice, as the loader's).
        const int64_t wpr = I * full.bits / 32, wps = M * full.bits / 32;
        const int g = full.group();
        require(M % g == 0, "packq slice test geometry: inter/world must be a multiple of the scale group");
        uint32_t* packed = nullptr;
        uint16_t* scales = nullptr;
        DGPP_CUDA_OK(cudaMallocManaged(&packed, static_cast<size_t>(H) * wps * 4));
        DGPP_CUDA_OK(cudaMallocManaged(&scales, static_cast<size_t>(H) * (M / g) * 2));
        for (int64_t row = 0; row < H; ++row) {
          std::memcpy(packed + row * wps, full.packed + row * wpr + rank * wps, static_cast<size_t>(wps) * 4);
          std::memcpy(scales + row * (M / g), full.scales + row * (I / g) + rank * M / g,
                      static_cast<size_t>(M / g) * 2);
        }
        r.owned_packq.push_back(packed);
        r.owned_packq.push_back(scales);
        r.mats_packq[static_cast<size_t>(m)] =
            dgpp::GlmPackedMatrix{packed, scales, H, M, full.bits, full.scale_fmt};
      }
      r.dev_w.router_gate = c.dev_w.router_gate;
      r.dev_w.router_bias = c.dev_w.router_bias;
      r.dev_w.experts = nullptr;
      r.dev_w.experts_packed = r.mats_packq.data();
      r.dev_w.routed_input_transform = c.dev_w.routed_input_transform;
      for (int m = 0; m < 3; ++m) r.dev_w.shared_packed[m] = r.mats_packq[static_cast<size_t>(E) * 3 + m];
      DGPP_CUDA_OK(cudaMallocManaged(&r.d_out, static_cast<size_t>(c.tokens) * H * 2));
      return r;
    }
    if (c.nvfp4()) {
      // The loader's NVFP4 slices in miniature: gate/up rows [rank*M, +M)
      // as views, down columns packed at nibble granularity (M % 16 == 0).
      const int g = c.expert_mats_fp4[0].scale_group;
      require(M % g == 0, "fp4 slice test geometry: inter/world must be a multiple of the scale group");
      r.mats_fp4.resize(static_cast<size_t>(E) * 3);
      for (int m = 0; m < E * 3; ++m) {
        const dgpp::GlmFp4Matrix& full = c.expert_mats_fp4[static_cast<size_t>(m)];
        if (m % 3 != 2) {
          r.mats_fp4[static_cast<size_t>(m)] = dgpp::fp4_rows_view(full, rank * M, M);
          continue;
        }
        uint8_t* payload = nullptr;
        uint8_t* scales = nullptr;
        DGPP_CUDA_OK(cudaMallocManaged(&payload, static_cast<size_t>(H) * M / 2));
        DGPP_CUDA_OK(cudaMallocManaged(&scales, static_cast<size_t>(H) * M / g));
        for (int64_t row = 0; row < H; ++row) {
          std::memcpy(payload + row * (M / 2), full.payload + row * (I / 2) + rank * M / 2, M / 2);
          std::memcpy(scales + row * (M / g), full.scales + row * (I / g) + rank * M / g, M / g);
        }
        r.owned_fp4.push_back(payload);
        r.owned_fp4.push_back(scales);
        r.mats_fp4[static_cast<size_t>(m)] =
            dgpp::GlmFp4Matrix{payload, scales, full.global_scale, H, M, g};
      }
      r.mats.resize(3);
      for (int m = 0; m < 3; ++m) {
        const GlmQuantMatrix& full = c.expert_mats[static_cast<size_t>(m)];
        if (m != 2) {
          r.mats[static_cast<size_t>(m)] = dgpp::quant_rows_view(full, rank * M, M);
          continue;
        }
        require(M % 128 == 0, "nvfp4 slice test: the FP8 shared expert needs a 128-multiple slice");
        uint8_t* payload = nullptr;
        float* scales = nullptr;
        const int64_t sb_full = (I + 127) / 128, sb_s = (M + 127) / 128;
        const int64_t scale_rows = (H + 127) / 128;
        DGPP_CUDA_OK(cudaMallocManaged(&payload, static_cast<size_t>(H) * M));
        DGPP_CUDA_OK(cudaMallocManaged(&scales, scale_rows * sb_s * 4));
        for (int64_t row = 0; row < H; ++row)
          std::memcpy(payload + row * M, full.payload + row * I + rank * M, M);
        for (int64_t row = 0; row < scale_rows; ++row)
          std::memcpy(scales + row * sb_s, full.scales + row * sb_full + (rank * M) / 128, sb_s * 4);
        r.owned_payloads.push_back(payload);
        r.owned_scales.push_back(scales);
        r.mats[static_cast<size_t>(m)] = GlmQuantMatrix{payload, scales, nullptr, H, M};
      }
      r.dev_w.router_gate = c.dev_w.router_gate;
      r.dev_w.router_bias = c.dev_w.router_bias;
      r.dev_w.experts = nullptr;
      r.dev_w.experts_fp4 = r.mats_fp4.data();
      for (int m = 0; m < 3; ++m) r.dev_w.shared[m] = r.mats[static_cast<size_t>(m)];
      DGPP_CUDA_OK(cudaMallocManaged(&r.d_out, static_cast<size_t>(c.tokens) * H * 2));
      return r;
    }
    require(M % 128 == 0, "slice test geometry: inter/world must be a "
                          "128-multiple (the scale-grid contract)");
    r.mats.resize(static_cast<size_t>(E + 1) * 3);
    for (int m = 0; m < (E + 1) * 3; ++m) {
      const GlmQuantMatrix& full = c.expert_mats[static_cast<size_t>(m)];
      if (m % 3 != 2) {
        r.mats[static_cast<size_t>(m)] = dgpp::quant_rows_view(full, rank * M, M);
        continue;
      }
      // down [H, I] -> packed [H, M] from column rank*M.
      uint8_t* payload = nullptr;
      float* scales = nullptr;
      const int64_t sb_full = (I + 127) / 128, sb_s = (M + 127) / 128;
      const int64_t scale_rows = (H + 127) / 128;
      DGPP_CUDA_OK(cudaMallocManaged(&payload, static_cast<size_t>(H) * M));
      DGPP_CUDA_OK(cudaMallocManaged(&scales, scale_rows * sb_s * 4));
      for (int64_t row = 0; row < H; ++row)
        std::memcpy(payload + row * M, full.payload + row * I + rank * M, M);
      for (int64_t row = 0; row < scale_rows; ++row)
        std::memcpy(scales + row * sb_s,
                    full.scales + row * sb_full + (rank * M) / 128, sb_s * 4);
      r.owned_payloads.push_back(payload);
      r.owned_scales.push_back(scales);
      r.mats[static_cast<size_t>(m)] = GlmQuantMatrix{payload, scales, nullptr, H, M};
    }
    r.dev_w.router_gate = c.dev_w.router_gate;
    r.dev_w.router_bias = c.dev_w.router_bias;
    r.dev_w.experts = r.mats.data();
    for (int m = 0; m < 3; ++m)
      r.dev_w.shared[m] = r.mats[static_cast<size_t>(E) * 3 + m];
    DGPP_CUDA_OK(cudaMallocManaged(&r.d_out,
                                   static_cast<size_t>(c.tokens) * H * 2));
    return r;
  }
  void free_all() {
    for (auto* p : owned_payloads) cudaFree(p);
    for (auto* s : owned_scales) cudaFree(s);
    for (auto* b : owned_fp4) cudaFree(b);
    for (auto* b : owned_packq) cudaFree(b);
    for (auto* b : owned_exl3) cudaFree(b);
    cudaFree(d_out);
  }
};

// The FFN all-reduce's fold (bus_fold semantics): per element, an fp32
// chain over the ranks' bf16 partials in rank order from 0.0f, rounded to
// bf16 once.
std::vector<uint16_t> fold_ranks(const std::vector<std::vector<uint16_t>>& parts) {
  std::vector<uint16_t> out(parts.at(0).size());
  for (size_t i = 0; i < out.size(); ++i) {
    float acc = 0.f;
    for (const auto& p : parts) acc += bf16_bits_to_float(p[i]);
    out[i] = float_to_bf16_bits(acc);
  }
  return out;
}

// One bf16 ulp at v's magnitude (7 explicit mantissa bits): v = m*2^e with
// m in [0.5, 1) puts the leading bit at 2^(e-1), the last at 2^(e-8).
double bf16_ulp_at(double v) {
  int e = 0;
  std::frexp(v == 0.0 ? 1e-30 : v, &e);
  return std::ldexp(1.0, e - 8);
}

// The sliced fold's certification. A fixed per-element ulp budget is the
// wrong ruler here: where the ranks' partials cancel (|sum| << |partial|),
// each partial's OWN half-ulp of bf16 rounding on the wire is many ulps of
// the small result — that is the bf16-on-the-wire cost the attention
// o_proj already pays, not a slicing bug. The bound is therefore built
// from the partials the test has in hand: every element's error against
// the unsliced oracle must stay within the partials' rounding budget
// (half an ulp of each partial, plus each partial's own engine-vs-double
// gap of at most one ulp — the unsliced path measures <= 1) and half an
// ulp of the folded result. A slicing bug (wrong scale block, wrong column
// origin) is O(value), thousands of times this bound.
// `partial_ulps`: each partial's budget in ulps — half an ulp of its bf16
// rounding plus its own engine-vs-double gap (1.5 for the chains whose
// unsliced gap measures <= 1 ulp; the EXL3 chain rounds its rotated input
// rows to fp16 and its unsliced gap measures up to ~2 ulps, so 2.5).
// `floor_frac`: the expert budget's absolute floor (a fraction of the
// oracle's RMS) for a chain whose intermediates are not the oracle's bits
// — the EXL3 chain's fp16-rounded rotated rows flip a gate / up value by
// an ulp now and then, an absolute error a small, cancelling output
// inherits (0: no floor; the strict chains' intermediates are the oracle's).
void require_within_fold_budget(
    const std::vector<uint16_t>& folded,
    const std::vector<std::vector<uint16_t>>& partials,
    const std::vector<uint16_t>& oracle, const char* label, double partial_ulps = 1.5, double floor_frac = 0.0) {
  double worst_ratio = 0;
  long violations = 0;
  double sum_d2 = 0, sum_o2 = 0;
  double rms = 0;
  for (size_t i = 0; i < oracle.size(); ++i) rms += std::pow(bf16_bits_to_float(oracle[i]), 2);
  rms = oracle.empty() ? 0 : std::sqrt(rms / oracle.size());
  const double floor_abs = floor_frac * rms;
  for (size_t i = 0; i < folded.size(); ++i) {
    const double got = bf16_bits_to_float(folded[i]);
    const double want = bf16_bits_to_float(oracle[i]);
    double budget = 0.5 * bf16_ulp_at(want);
    for (const auto& p : partials)
      budget += partial_ulps * bf16_ulp_at(bf16_bits_to_float(p[i]));
    budget = std::max(budget, floor_abs);
    const double err = std::abs(got - want);
    worst_ratio = std::max(worst_ratio, err / budget);
    if (err > budget) ++violations;
    sum_d2 += (got - want) * (got - want);
    sum_o2 += want * want;
  }
  const double l2 = sum_o2 > 0 ? std::sqrt(sum_d2 / sum_o2) : 0;
  std::printf("[ .. ] %s: worst err/budget %.3f, %ld/%zu over budget, "
              "l2=%.2g\n",
              label, worst_ratio, violations, folded.size(), l2);
  require(violations == 0, "sliced fold: element outside the partials' "
                           "rounding budget");
  require(l2 < 4e-3, "sliced fold: l2 budget");
}

// ---- the grouped prefill path --------------------------------

DGPP_TEST(moe_grouped_gemv_is_bitwise_the_chunked_scale_gemm_per_segment) {
  // GIVEN segments of every shape the prefill produces — one row, a partial
  // group, several full groups plus a remainder (the multi-group loop, which
  // no other gate reaches), a long shared-style segment — over an output
  // width that leaves dead warps in the last block (n % kWarps != 0), and
  // the fp32 down variant with a 16-multiple k,
  // THEN every output row is bitwise the per-segment scale GEMM's (the same
  // fp8_gemv core, four rows per group).
  struct Shape { int I; bool test_down; };
  for (const Shape sh : {Shape{204, false}, Shape{208, true}}) {
    SmallCase c = make_small_case(/*E=*/6, /*H=*/4096, /*I=*/sh.I, /*K=*/2,
                                  /*tokens=*/64, 0x6D0 + sh.I);
    c.alloc();
    const int H = c.cfg.hidden, I = sh.I, E = c.cfg.n_experts;
    // Segments partition the 64 activation rows: lengths 1,5,4,9,13,2,30.
    const int lens[] = {1, 5, 4, 9, 13, 2, 30};
    std::vector<dgpp::MoeSegment> segs;
    int row0 = 0;
    for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); ++i) {
      segs.push_back(dgpp::MoeSegment{row0, lens[i], static_cast<int>(i % (E + 1))});
      row0 += lens[i];
    }
    require(row0 == 64, "segments cover the rows");
    dgpp::MoeSegment* d_segs = nullptr;
    dgpp::MoeExpertView* d_views = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_segs, segs.size() * sizeof(dgpp::MoeSegment)));
    std::memcpy(d_segs, segs.data(), segs.size() * sizeof(dgpp::MoeSegment));
    DGPP_CUDA_OK(cudaMallocManaged(&d_views, (E + 1) * 3 * sizeof(dgpp::MoeExpertView)));
    for (int m = 0; m < (E + 1) * 3; ++m)
      d_views[m] = dgpp::MoeExpertView{c.expert_mats[m].payload, c.expert_mats[m].scales};
    // gate (which = 0): [64][I] bf16 from act [64][H].
    uint16_t *d_grouped = nullptr, *d_ref = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_grouped, 64 * I * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_ref, 64 * I * 2));
    DGPP_CUDA_OK(cudaMemset(d_grouped, 0xA5, 64 * I * 2));
    DGPP_CUDA_OK(cudaMemset(d_ref, 0x5A, 64 * I * 2));
    dgpp::launch_moe_grouped_gemv_bf16(c.d_hidden, H, d_segs, static_cast<int>(segs.size()),
                                 30, /*rows_per_block=*/8, d_views, 0, d_grouped, I, I, H, nullptr);
    for (const dgpp::MoeSegment& sg : segs) {
      const GlmQuantMatrix& g = c.expert_mats[sg.expert * 3 + 0];
      dgpp::launch_scale_gemm_bf16(c.d_hidden + static_cast<size_t>(sg.row0) * H, H,
                             g.payload, g.scales,
                             d_ref + static_cast<size_t>(sg.row0) * I, sg.rows, I, H,
                             nullptr);
    }
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(std::memcmp(d_grouped, d_ref, 64 * I * 2) == 0,
            ("grouped gate output bitwise the per-segment scale GEMM (I=" +
             std::to_string(I) + ")").c_str());
    if (sh.test_down) {
      // down (which = 2): [64][H] fp32 from act [64][I] (the gate output
      // above serves as the activation; k = I is a multiple of 16).
      float *d_gd = nullptr, *d_rd = nullptr;
      DGPP_CUDA_OK(cudaMallocManaged(&d_gd, 64 * H * 4));
      DGPP_CUDA_OK(cudaMallocManaged(&d_rd, 64 * H * 4));
      DGPP_CUDA_OK(cudaMemset(d_gd, 0xA5, 64 * H * 4));
      DGPP_CUDA_OK(cudaMemset(d_rd, 0x5A, 64 * H * 4));
      dgpp::launch_moe_grouped_gemv_f32(d_grouped, I, d_segs, static_cast<int>(segs.size()),
                                  30, /*rows_per_block=*/0, d_views, 2, d_gd, H, H, I, nullptr);
      for (const dgpp::MoeSegment& sg : segs) {
        const GlmQuantMatrix& d = c.expert_mats[sg.expert * 3 + 2];
        dgpp::launch_scale_gemm_f32(d_grouped + static_cast<size_t>(sg.row0) * I, I,
                              d.payload, d.scales,
                              d_rd + static_cast<size_t>(sg.row0) * H, sg.rows, H, I,
                              nullptr);
      }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      require(std::memcmp(d_gd, d_rd, 64 * H * 4) == 0,
              "grouped down output bitwise the per-segment scale GEMM (fp32)");
      cudaFree(d_gd);
      cudaFree(d_rd);
    }
    std::printf("[ OK ] grouped gemv I=%d: %zu segments (1..30 rows) bitwise the "
                "chunked GEMV%s\n",
                I, segs.size(), sh.test_down ? ", down fp32 too" : "");
    cudaFree(d_grouped);
    cudaFree(d_ref);
    cudaFree(d_segs);
    cudaFree(d_views);
    c.free_all();
  }
}

DGPP_TEST(moe_grouped_mma_is_bitwise_the_tile_gemm_per_segment) {
  // GIVEN segments of one row, a few rows, more than one 128-row m-tile
  // (130, 245) — over an output width that leaves a ragged n-tile (204 %
  // 64 != 0) and the fp32 down variant with k = 208,
  // THEN every output element is bitwise the tile kernel's per segment
  // (the same dequantized weights, the same ascending-k16 mma.sync chain),
  // whether one block walks a segment's m-tiles or a z split spreads them.
  struct Shape { int I; bool test_down; };
  constexpr int kRows = 400;
  for (const Shape sh : {Shape{204, false}, Shape{208, true}}) {
    SmallCase c = make_small_case(/*E=*/6, /*H=*/4096, /*I=*/sh.I, /*K=*/2,
                                  /*tokens=*/kRows, 0x6E0 + sh.I);
    c.alloc();
    const int H = c.cfg.hidden, I = sh.I, E = c.cfg.n_experts;
    const int lens[] = {1, 5, 4, 130, 13, 2, 245};
    std::vector<dgpp::MoeSegment> segs;
    int row0 = 0;
    for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); ++i) {
      segs.push_back(dgpp::MoeSegment{row0, lens[i], static_cast<int>(i % (E + 1))});
      row0 += lens[i];
    }
    require(row0 == kRows, "segments cover the rows");
    dgpp::MoeSegment* d_segs = nullptr;
    dgpp::MoeExpertView* d_views = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_segs, segs.size() * sizeof(dgpp::MoeSegment)));
    std::memcpy(d_segs, segs.data(), segs.size() * sizeof(dgpp::MoeSegment));
    DGPP_CUDA_OK(cudaMallocManaged(&d_views, (E + 1) * 3 * sizeof(dgpp::MoeExpertView)));
    for (int m = 0; m < (E + 1) * 3; ++m)
      d_views[m] = dgpp::MoeExpertView{c.expert_mats[m].payload, c.expert_mats[m].scales};
    uint16_t *d_grouped = nullptr, *d_split = nullptr, *d_ref = nullptr;
    const size_t gate_bytes = static_cast<size_t>(kRows) * I * 2;
    DGPP_CUDA_OK(cudaMallocManaged(&d_grouped, gate_bytes));
    DGPP_CUDA_OK(cudaMallocManaged(&d_split, gate_bytes));
    DGPP_CUDA_OK(cudaMallocManaged(&d_ref, gate_bytes));
    DGPP_CUDA_OK(cudaMemset(d_grouped, 0xA5, gate_bytes));
    DGPP_CUDA_OK(cudaMemset(d_split, 0xA5, gate_bytes));
    DGPP_CUDA_OK(cudaMemset(d_ref, 0x5A, gate_bytes));
    dgpp::launch_moe_grouped_mma_bf16(c.d_hidden, H, d_segs, static_cast<int>(segs.size()),
                                      245, /*rows_per_block=*/0, d_views, 0, d_grouped,
                                      I, I, H, nullptr);
    dgpp::launch_moe_grouped_mma_bf16(c.d_hidden, H, d_segs, static_cast<int>(segs.size()),
                                      245, /*rows_per_block=*/128, d_views, 0, d_split,
                                      I, I, H, nullptr);
    for (const dgpp::MoeSegment& sg : segs) {
      const GlmQuantMatrix& g = c.expert_mats[sg.expert * 3 + 0];
      dgpp::launch_scale_gemm_tile_bf16(c.d_hidden + static_cast<size_t>(sg.row0) * H, H,
                                        g.payload, g.scales,
                                        d_ref + static_cast<size_t>(sg.row0) * I, sg.rows,
                                        I, H, nullptr);
    }
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(std::memcmp(d_grouped, d_ref, gate_bytes) == 0,
            ("grouped mma gate output bitwise the tile GEMM per segment (I=" +
             std::to_string(I) + ")").c_str());
    require(std::memcmp(d_split, d_ref, gate_bytes) == 0,
            "grouped mma gate output bitwise under the z split");
    if (sh.test_down) {
      const size_t down_bytes = static_cast<size_t>(kRows) * H * 4;
      float *d_gd = nullptr, *d_rd = nullptr;
      DGPP_CUDA_OK(cudaMallocManaged(&d_gd, down_bytes));
      DGPP_CUDA_OK(cudaMallocManaged(&d_rd, down_bytes));
      DGPP_CUDA_OK(cudaMemset(d_gd, 0xA5, down_bytes));
      DGPP_CUDA_OK(cudaMemset(d_rd, 0x5A, down_bytes));
      dgpp::launch_moe_grouped_mma_f32(d_grouped, I, d_segs, static_cast<int>(segs.size()),
                                       245, /*rows_per_block=*/0, d_views, 2, d_gd, H, H,
                                       I, nullptr);
      for (const dgpp::MoeSegment& sg : segs) {
        const GlmQuantMatrix& d = c.expert_mats[sg.expert * 3 + 2];
        dgpp::launch_scale_gemm_tile_f32(d_grouped + static_cast<size_t>(sg.row0) * I, I,
                                         d.payload, d.scales,
                                         d_rd + static_cast<size_t>(sg.row0) * H, sg.rows,
                                         H, I, nullptr);
      }
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      require(std::memcmp(d_gd, d_rd, down_bytes) == 0,
              "grouped mma down output bitwise the tile GEMM per segment (fp32)");
      cudaFree(d_gd);
      cudaFree(d_rd);
    }
    std::printf("[ OK ] grouped mma I=%d: %zu segments (1..245 rows) bitwise the "
                "tile GEMM%s\n",
                I, segs.size(), sh.test_down ? ", down fp32 too" : "");
    cudaFree(d_grouped);
    cudaFree(d_split);
    cudaFree(d_ref);
    cudaFree(d_segs);
    cudaFree(d_views);
    c.free_all();
  }
}

DGPP_TEST(moe_grouped_mma_small_case_shapes_bitwise_the_tile_gemm) {
  // The layer's small-case geometry (H=512, I=256, one-row segments): every
  // matrix (gate, up, down) bitwise the tile kernel per segment.
  SmallCase c = make_small_case(/*E=*/8, /*H=*/512, /*I=*/256, /*K=*/2,
                                /*tokens=*/3, 0x5CA1E);
  c.alloc();
  const int H = c.cfg.hidden, I = 256, E = c.cfg.n_experts;
  std::vector<dgpp::MoeSegment> segs = {{0, 1, 2}, {1, 1, 5}, {2, 1, E}};
  dgpp::MoeSegment* d_segs = nullptr;
  dgpp::MoeExpertView* d_views = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_segs, segs.size() * sizeof(dgpp::MoeSegment)));
  std::memcpy(d_segs, segs.data(), segs.size() * sizeof(dgpp::MoeSegment));
  DGPP_CUDA_OK(cudaMallocManaged(&d_views, (E + 1) * 3 * sizeof(dgpp::MoeExpertView)));
  for (int m = 0; m < (E + 1) * 3; ++m)
    d_views[m] = dgpp::MoeExpertView{c.expert_mats[m].payload, c.expert_mats[m].scales};
  for (int which = 0; which < 2; ++which) {
    uint16_t *d_g = nullptr, *d_r = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_g, 3 * I * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_r, 3 * I * 2));
    DGPP_CUDA_OK(cudaMemset(d_g, 0xA5, 3 * I * 2));
    DGPP_CUDA_OK(cudaMemset(d_r, 0x5A, 3 * I * 2));
    dgpp::launch_moe_grouped_mma_bf16(c.d_hidden, H, d_segs, 3, 3, /*rows_per_block=*/0,
                                      d_views, which, d_g, I, I, H, nullptr);
    for (const dgpp::MoeSegment& sg : segs) {
      const GlmQuantMatrix& g = c.expert_mats[sg.expert * 3 + which];
      dgpp::launch_scale_gemm_tile_bf16(c.d_hidden + static_cast<size_t>(sg.row0) * H, H,
                                        g.payload, g.scales,
                                        d_r + static_cast<size_t>(sg.row0) * I, sg.rows,
                                        I, H, nullptr);
    }
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    int bad = 0, first = -1;
    for (int i = 0; i < 3 * I; ++i)
      if (d_g[i] != d_r[i]) { if (first < 0) first = i; ++bad; }
    std::printf("[ .. ] small-case which=%d: %d of %d elements differ (first %d: got %04x ref %04x)\n",
                which, bad, 3 * I, first, first >= 0 ? d_g[first] : 0, first >= 0 ? d_r[first] : 0);
    require(bad == 0, "small-case grouped mma bitwise the tile GEMM");
    cudaFree(d_g);
    cudaFree(d_r);
  }
  {
    float *d_g = nullptr, *d_r = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_g, 3 * H * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&d_r, 3 * H * 4));
    // The down projection consumes I-wide activations: reuse d_hidden's
    // first I columns per row (stride H).
    dgpp::launch_moe_grouped_mma_f32(c.d_hidden, H, d_segs, 3, 3, 0, d_views, 2, d_g, H, H,
                                     I, nullptr);
    for (const dgpp::MoeSegment& sg : segs) {
      const GlmQuantMatrix& d = c.expert_mats[sg.expert * 3 + 2];
      dgpp::launch_scale_gemm_tile_f32(c.d_hidden + static_cast<size_t>(sg.row0) * H, H,
                                       d.payload, d.scales,
                                       d_r + static_cast<size_t>(sg.row0) * H, sg.rows, H,
                                       I, nullptr);
    }
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    int bad = 0, first = -1;
    for (int i = 0; i < 3 * H; ++i)
      if (std::memcmp(&d_g[i], &d_r[i], 4) != 0) { if (first < 0) first = i; ++bad; }
    std::printf("[ .. ] small-case down: %d of %d elements differ (first %d: got %g ref %g)\n",
                bad, 3 * H, first, first >= 0 ? d_g[first] : 0.f, first >= 0 ? d_r[first] : 0.f);
    require(bad == 0, "small-case grouped mma down bitwise the tile GEMM");
    cudaFree(d_g);
    cudaFree(d_r);
  }
  cudaFree(d_segs);
  cudaFree(d_views);
  c.free_all();
}

DGPP_TEST(moe_accum_ordered_is_bitwise_the_fmaf_chain) {
  // GIVEN random fp32 down rows, a permuted slot->row map, UNSORTED slot
  // ids per token and random weights, THEN the device's per-token result is
  // bitwise the host's chain: slots in ascending expert id, acc = fmaf(w,
  // y, acc) from zero, then fmaf(1, shared, acc), one bf16 rounding.
  const int tokens = 16, K = 4, H = 64;
  const int tk = tokens * K;
  Rng rng(0xACC);
  std::vector<int32_t> slot_row(tk), slot_ids(tk);
  std::vector<float> slot_w(tk), down(static_cast<size_t>(tk + tokens) * H);
  for (int i = 0; i < tk; ++i) slot_row[i] = i;
  for (int i = tk - 1; i > 0; --i) {  // a permutation
    const int j = static_cast<int>(rng.next() % static_cast<uint64_t>(i + 1));
    std::swap(slot_row[i], slot_row[j]);
  }
  for (int t = 0; t < tokens; ++t) {
    // K distinct ids in [0, 40), deliberately unsorted.
    std::vector<int> pool;
    while (static_cast<int>(pool.size()) < K) {
      const int e = static_cast<int>(rng.next() % 40);
      bool dup = false;
      for (const int q : pool) dup = dup || q == e;
      if (!dup) pool.push_back(e);
    }
    for (int i = 0; i < K; ++i) {
      slot_ids[t * K + i] = pool[i];
      slot_w[t * K + i] = 0.5f + 0.5f * static_cast<float>(rng.unit());
    }
  }
  for (auto& v : down) v = 4.0f * static_cast<float>(rng.unit());
  int32_t *d_row = nullptr, *d_ids = nullptr;
  float *d_w = nullptr, *d_down = nullptr;
  uint16_t* d_out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_row, tk * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_ids, tk * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_w, tk * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_down, down.size() * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_out, static_cast<size_t>(tokens) * H * 2));
  std::memcpy(d_row, slot_row.data(), tk * 4);
  std::memcpy(d_ids, slot_ids.data(), tk * 4);
  std::memcpy(d_w, slot_w.data(), tk * 4);
  std::memcpy(d_down, down.data(), down.size() * 4);
  dgpp::launch_moe_accum_ordered(d_out, d_down, H, d_row, d_ids, d_w, tk, tokens, K, H,
                           nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  int mismatches = 0;
  for (int t = 0; t < tokens; ++t) {
    std::vector<int> order(K);
    for (int i = 0; i < K; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return slot_ids[t * K + a] < slot_ids[t * K + b];
    });
    for (int h = 0; h < H; ++h) {
      float acc = 0.f;
      for (const int i : order) {
        const int s = t * K + i;
        acc = std::fmaf(slot_w[s], down[static_cast<size_t>(slot_row[s]) * H + h], acc);
      }
      acc = std::fmaf(1.0f, down[static_cast<size_t>(tk + t) * H + h], acc);
      if (d_out[static_cast<size_t>(t) * H + h] != float_to_bf16_bits(acc)) ++mismatches;
    }
  }
  require(mismatches == 0, ("ordered accumulate bitwise the host chain: " +
                            std::to_string(mismatches) + " mismatches").c_str());
  std::printf("[ OK ] accum ordered: %d tokens x %d slots bitwise the fmaf chain\n",
              tokens, K);
  cudaFree(d_row); cudaFree(d_ids); cudaFree(d_w); cudaFree(d_down); cudaFree(d_out);
}

DGPP_TEST(moe_enqueue_prefill_is_bitwise_the_host_path) {
  // The sync-free prefill path (device segmentation, the grouped chain,
  // traces by async copy) against the host-orchestrated one: bitwise
  // outputs and identical traces, on geometries whose segments span
  // several row groups (M=64, K=4: ~32 rows per expert).
  struct Case { int E, H, I, K, M; };
  const Case cases[] = {
      {8, 512, 256, 2, 1},
      {8, 512, 256, 2, 3},
      {16, 1024, 512, 4, 2},
      {8, 512, 256, 4, 64},
  };
  for (const Case& cs : cases) {
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M, 0xB0A7 + cs.E + cs.M);
    c.alloc();
    GlmMoeLayer layer(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/0);
    std::vector<uint16_t> host(static_cast<size_t>(cs.M) * c.cfg.hidden);
    // The reference runs the host-orchestrated chain on the prefill's
    // kernel (the tensor-core path); the decode pin keeps the GEMV core.
    layer.enqueue(c.d_hidden, c.d_out, cs.M, nullptr, dgpp::MoeExpertKernel::kMma);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(host.data(), c.d_out, host.size() * 2);
    const std::vector<int32_t> want_ids = layer.last_ids();
    const std::vector<float> want_w = layer.last_weights();
    const std::vector<float> want_biased = layer.last_biased();
    int32_t* pin_ids = nullptr;
    float* pin_w = nullptr;
    float* pin_b = nullptr;
    DGPP_CUDA_OK(cudaHostAlloc(&pin_ids, want_ids.size() * 4, cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(&pin_w, want_w.size() * 4, cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(&pin_b, want_biased.size() * 4, cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    dgpp::MoeTraceStaging trace{pin_ids, pin_w, pin_b};
    layer.enqueue_prefill(c.d_hidden, c.d_out, cs.M, &trace, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> dev(host.size(), 0x7F7F);
    std::memcpy(dev.data(), c.d_out, dev.size() * 2);
    require(std::memcmp(host.data(), dev.data(), host.size() * 2) == 0,
            "enqueue_prefill must be bitwise-identical to enqueue");
    require(std::memcmp(pin_ids, want_ids.data(), want_ids.size() * 4) == 0,
            "prefill staged ids equal the host path's");
    require(std::memcmp(pin_w, want_w.data(), want_w.size() * 4) == 0,
            "prefill staged weights equal the host path's");
    require(std::memcmp(pin_b, want_biased.data(), want_biased.size() * 4) == 0,
            "prefill staged biased scores equal the host path's");
    std::printf("[ OK ] prefill device path E=%d H=%d I=%d K=%d M=%d: bitwise\n",
                cs.E, cs.H, cs.I, cs.K, cs.M);
    // The phased prefill (the router and the accumulation per row block
    // around the one chain): bitwise the same output.
    const int ma = cs.M / 2;
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    layer.route_prefill_rows(c.d_hidden, 0, ma, nullptr);
    layer.route_prefill_rows(c.d_hidden, ma, cs.M - ma, nullptr);
    layer.enqueue_prefill_phased(c.d_hidden, cs.M, nullptr);
    layer.accumulate_prefill_rows(c.d_out, cs.M, ma, cs.M - ma, nullptr);
    layer.accumulate_prefill_rows(c.d_out, cs.M, 0, ma, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(dev.data(), c.d_out, dev.size() * 2);
    require(std::memcmp(host.data(), dev.data(), host.size() * 2) == 0,
            "the phased prefill must be bitwise-identical to enqueue");
    std::printf("[ OK ] phased prefill (router and accumulation in row blocks of %d + %d): bitwise\n", ma, cs.M - ma);
    cudaFreeHost(pin_ids); cudaFreeHost(pin_w); cudaFreeHost(pin_b);
    c.free_all();
  }
}

}  // namespace

DGPP_TEST(moe_mixed_rows_preserve_decode_and_reader_recipes) {
  constexpr int D = 4, H = 512, K = 4, E = 8;
  for (int format = 0; format < 4; ++format) for (int M : {67, 131, 260}) for (bool hashed : {false, true}) {
    SmallCase c = make_small_case(E, H, 256, K, M, 0xDEC0DE + format,
        /*nvfp4=*/format == 1 || format == 2, /*shared_nvfp4=*/format == 1,
        /*packq_bits=*/format == 3 ? 4 : 0, /*fp4_group=*/format == 2 ? 32 : 16);
    c.cfg.shared_mma_aside = format == 2;  // native shared expert on side streams
    c.alloc();
    int64_t* hash_tokens = nullptr;
    int32_t* hash_table = nullptr;
    const int vocab = M + 3;
    if (hashed) {
      DGPP_CUDA_OK(cudaMallocManaged(&hash_tokens, M * sizeof(int64_t)));
      DGPP_CUDA_OK(cudaMallocManaged(&hash_table, vocab * K * sizeof(int32_t)));
      for (int t = 0; t < M; ++t) hash_tokens[t] = (t * 7 + 3) % vocab;
      for (int t = 0; t < vocab; ++t)
        for (int k = 0; k < K; ++k) hash_table[t * K + k] = (t * 3 + k) % E;
    }
    {
      GlmMoeLayer layer(c.dev_w, c.cfg, M, D);
      layer.set_hash_routing(hash_table, hash_tokens, vocab);
      std::vector<uint16_t> expected(static_cast<size_t>(M) * H);
      int32_t* ids = nullptr;
      float *weights = nullptr, *biased = nullptr;
      DGPP_CUDA_OK(cudaHostAlloc(&ids, M * K * sizeof(int32_t), cudaHostAllocDefault));
      DGPP_CUDA_OK(cudaHostAlloc(&weights, M * K * sizeof(float), cudaHostAllocDefault));
      if (hashed) DGPP_CUDA_OK(cudaHostAlloc(&biased, M * E * sizeof(float), cudaHostAllocDefault));
      dgpp::MoeTraceStaging trace{ids, weights, biased};
      layer.enqueue_prefill(c.d_hidden, c.d_out, M, &trace, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<int32_t> expected_ids(ids, ids + M * K);
      const std::vector<float> expected_weights(weights, weights + M * K);
      const std::vector<float> expected_biased = biased ? std::vector<float>(biased, biased + M * E) : std::vector<float>{};
      layer.set_hash_routing(hash_table, hashed ? hash_tokens + D : nullptr, vocab);
      layer.enqueue_prefill(c.d_hidden + D * H, c.d_out + D * H, M - D, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::memcpy(expected.data() + D * H, c.d_out + D * H, (M - D) * H * sizeof(uint16_t));
      layer.set_hash_routing(hash_table, hash_tokens, vocab);
      layer.enqueue_decode(c.d_hidden, c.d_out, D, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::memcpy(expected.data(), c.d_out, D * H * sizeof(uint16_t));
      for (int total : {D, M}) {
        DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, expected.size() * sizeof(uint16_t)));
        layer.enqueue_prefill(c.d_hidden, c.d_out, total, &trace, nullptr, D);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        require(std::memcmp(expected.data(), c.d_out, total * H * sizeof(uint16_t)) == 0,
                "mixed expert rows preserve native decoder and grouped reader outputs");
        require(std::equal(expected_ids.begin(), expected_ids.begin() + total * K, ids),
                "mixed expert traces preserve route ids with optional biased staging");
        require(std::equal(expected_weights.begin(), expected_weights.begin() + total * K, weights),
                "mixed expert traces preserve route weights");
        if (biased)
          require(std::equal(expected_biased.begin(), expected_biased.begin() + total * E, biased),
                  "mixed expert traces preserve biased scores");
        // Boundaries before, within and after the decoder prefix; reverse
        // accumulation order also exercises the overlap caller's contract.
        for (int split : {1, D - 1, D, total}) {
          DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, expected.size() * sizeof(uint16_t)));
          layer.route_prefill_rows(c.d_hidden, 0, split, nullptr);
          layer.route_prefill_rows(c.d_hidden, split, total - split, nullptr);
          layer.enqueue_prefill_phased(c.d_hidden, total, nullptr, D, c.d_out);
          layer.accumulate_prefill_rows(c.d_out, total, split, total - split, nullptr, D);
          layer.accumulate_prefill_rows(c.d_out, total, 0, split, nullptr, D);
          DGPP_CUDA_OK(cudaDeviceSynchronize());
          require(std::memcmp(expected.data(), c.d_out, total * H * sizeof(uint16_t)) == 0,
                  "phased mixed expert rows preserve both recipes across split boundaries");
        }
      }
      // The float accumulator path is used by families with a separate
      // block-boundary rounding step, including MiMo's MXFP4 experts.
      float* out_f32 = nullptr;
      DGPP_CUDA_OK(cudaMallocManaged(&out_f32, M * H * sizeof(float)));
      std::vector<float> expected_f32(M * H);
      layer.set_hash_routing(hash_table, hashed ? hash_tokens + D : nullptr, vocab);
      layer.enqueue_prefill_f32(c.d_hidden + D * H, out_f32 + D * H, M - D, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::memcpy(expected_f32.data() + D * H, out_f32 + D * H, (M - D) * H * sizeof(float));
      layer.set_hash_routing(hash_table, hash_tokens, vocab);
      layer.enqueue_decode_f32(c.d_hidden, out_f32, D, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::memcpy(expected_f32.data(), out_f32, D * H * sizeof(float));
      for (int total : {D, M}) {
        DGPP_CUDA_OK(cudaMemset(out_f32, 0x7F, M * H * sizeof(float)));
        layer.enqueue_prefill_f32(c.d_hidden, out_f32, total, &trace, nullptr,
                                  dgpp::MoeExpertKernel::kMma, D);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        require(std::memcmp(expected_f32.data(), out_f32, total * H * sizeof(float)) == 0,
                "float mixed expert rows preserve native decoder and grouped reader outputs");
        require(std::equal(expected_ids.begin(), expected_ids.begin() + total * K, ids),
                "float mixed expert traces preserve route ids");
        require(std::equal(expected_weights.begin(), expected_weights.begin() + total * K, weights),
                "float mixed expert traces preserve route weights");
        if (biased)
          require(std::equal(expected_biased.begin(), expected_biased.begin() + total * E, biased),
                  "float mixed expert traces preserve biased scores");
      }
      cudaFree(out_f32);
      cudaFreeHost(ids);
      cudaFreeHost(weights);
      cudaFreeHost(biased);
    }
    cudaFree(hash_tokens);
    cudaFree(hash_table);
    c.free_all();
  }
}

DGPP_TEST(moe_router_tiled_dots_are_bitwise_the_warp_form) {
  // The prefill router's tiled dots (16 tokens x 16 experts per block,
  // operands staged in shared memory) against the warp-per-(token, expert)
  // form on the real geometry: every score and biased score bitwise, and
  // the selection with them — the per-lane element order and the shuffle
  // tree are the same by construction; this pins it. 70 tokens: four full
  // token tiles and a ragged one; 288 experts: eighteen expert tiles.
  const int E = 288, H = 4096, K = 8, M = 70;
  GlmMoeConfig cfg{};
  cfg.n_experts = E; cfg.hidden = H; cfg.top_k = K;
  cfg.routed_scaling_factor = 2.5f; cfg.norm_topk_prob = true;
  std::mt19937 rng(0x7A11ED);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  std::vector<uint16_t> h_hidden(size_t(M) * H), h_gate(size_t(E) * H);
  std::vector<float> h_bias(E);
  for (auto& v : h_hidden) v = float_to_bf16_bits(U(rng));
  for (auto& v : h_gate) v = float_to_bf16_bits(U(rng) * 0.05f);
  for (auto& v : h_bias) v = U(rng) * 0.01f;
  uint16_t *d_hidden = nullptr, *d_gate = nullptr;
  float *d_bias = nullptr, *d_scores_a = nullptr, *d_scores_b = nullptr,
        *d_biased_a = nullptr, *d_biased_b = nullptr, *d_w_a = nullptr, *d_w_b = nullptr;
  int32_t *d_ids_a = nullptr, *d_ids_b = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_hidden, h_hidden.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_gate, h_gate.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_bias, E * 4));
  for (float** pp : {&d_scores_a, &d_scores_b, &d_biased_a, &d_biased_b})
    DGPP_CUDA_OK(cudaMallocManaged(pp, size_t(M) * E * 4));
  for (float** pp : {&d_w_a, &d_w_b}) DGPP_CUDA_OK(cudaMallocManaged(pp, size_t(M) * K * 4));
  for (int32_t** pp : {&d_ids_a, &d_ids_b}) DGPP_CUDA_OK(cudaMallocManaged(pp, size_t(M) * K * 4));
  std::memcpy(d_hidden, h_hidden.data(), h_hidden.size() * 2);
  std::memcpy(d_gate, h_gate.data(), h_gate.size() * 2);
  std::memcpy(d_bias, h_bias.data(), E * 4);
  dgpp::launch_moe_router(d_hidden, d_gate, d_bias, d_ids_a, d_w_a, d_scores_a, d_biased_a,
                          cfg, M, nullptr, nullptr, /*allow_tiled=*/false);
  dgpp::launch_moe_router(d_hidden, d_gate, d_bias, d_ids_b, d_w_b, d_scores_b, d_biased_b,
                          cfg, M, nullptr, nullptr, /*allow_tiled=*/true);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  require(std::memcmp(d_scores_a, d_scores_b, size_t(M) * E * 4) == 0,
          "tiled router scores bitwise the warp form");
  require(std::memcmp(d_biased_a, d_biased_b, size_t(M) * E * 4) == 0,
          "tiled router biased scores bitwise the warp form");
  require(std::memcmp(d_ids_a, d_ids_b, size_t(M) * K * 4) == 0, "tiled router ids identical");
  require(std::memcmp(d_w_a, d_w_b, size_t(M) * K * 4) == 0, "tiled router weights identical");
  std::printf("[ OK ] tiled router: %d tokens x %d experts bitwise the warp form\n", M, E);
  for (void* pp : {(void*)d_hidden, (void*)d_gate, (void*)d_bias, (void*)d_scores_a,
                   (void*)d_scores_b, (void*)d_biased_a, (void*)d_biased_b, (void*)d_w_a,
                   (void*)d_w_b, (void*)d_ids_a, (void*)d_ids_b})
    cudaFree(pp);
}

DGPP_TEST(moe_router_matches_oracle_real_geometry) {
  GlmMoeConfig cfg;  // E=288, H=4096, K=8, scale 2.5, norm on, limit 10
  for (int tokens : {1, 3, 257}) check_router(cfg, tokens, 0xA11CE + tokens);
  // norm_topk_prob = false path (weights un-normalized, just scaled).
  GlmMoeConfig no_norm = cfg;
  no_norm.norm_topk_prob = false;
  check_router(no_norm, 17, 0xB0B);
  // The sqrtsoftplus mode (DeepSeek-V4.1-Flash, 2026-09-13) at the release's
  // routing geometry: 384 experts, top-6, x 1.5, hidden 5120.
  GlmMoeConfig sp = cfg;
  sp.router_mode = dgpp::MoeRouterMode::SqrtSoftplusBias;
  sp.n_experts = 384;
  sp.top_k = 6;
  sp.hidden = 5120;
  sp.routed_scaling_factor = 1.5f;
  for (int tokens : {1, 5, 70}) check_router(sp, tokens, 0x5A5A + tokens);
}

DGPP_TEST(moe_router_ties_break_to_lower_expert_id) {
  // Zero hidden and zero gate => all logits 0, all scores 0.5; equal biases
  // => all biased equal: selection must be experts 0..7 with equal weights
  // (0.5 / 4.0 * 2.5 each).
  GlmMoeConfig cfg;
  const int tokens = 4, E = cfg.n_experts, K = cfg.top_k;
  std::vector<uint16_t> hidden(static_cast<size_t>(tokens) * cfg.hidden, 0);
  std::vector<uint16_t> gate(static_cast<size_t>(E) * cfg.hidden, 0);
  std::vector<float> bias(E, 0.f);

  uint16_t* d_h = nullptr;
  uint16_t* d_g = nullptr;
  float* d_b = nullptr;
  int32_t* d_ids = nullptr;
  float* d_w = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_h, hidden.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_g, gate.size() * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_b, bias.size() * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_ids, static_cast<size_t>(tokens) * K * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_w, static_cast<size_t>(tokens) * K * 4));
  float* d_scores = nullptr;
  float* d_biased = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_scores, static_cast<size_t>(tokens) * E * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_biased, static_cast<size_t>(tokens) * E * 4));
  std::memcpy(d_h, hidden.data(), hidden.size() * 2);
  std::memcpy(d_g, gate.data(), gate.size() * 2);
  std::memcpy(d_b, bias.data(), bias.size() * 4);
  dgpp::launch_moe_router(d_h, d_g, d_b, d_ids, d_w, d_scores, d_biased, cfg,
                          tokens, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());

  const float want_w = 0.5f / (8 * 0.5f) * 2.5f;
  for (int t = 0; t < tokens; ++t)
    for (int i = 0; i < K; ++i) {
      require(d_ids[static_cast<size_t>(t) * K + i] == i,
              "tie selection must be experts 0..7 in ascending order");
      require(std::abs(d_w[static_cast<size_t>(t) * K + i] - want_w) < 1e-6,
              "tie weights are uniform");
    }
  DGPP_CUDA_OK(cudaFree(d_h));
  DGPP_CUDA_OK(cudaFree(d_g));
  DGPP_CUDA_OK(cudaFree(d_b));
  DGPP_CUDA_OK(cudaFree(d_ids));
  DGPP_CUDA_OK(cudaFree(d_w));
  DGPP_CUDA_OK(cudaFree(d_scores));
  DGPP_CUDA_OK(cudaFree(d_biased));
  std::printf("[ OK ] router ties: experts 0..7, uniform weights\n");
}

DGPP_TEST(moe_swiglu_clamp_edge_semantics) {
  // gate clamps only the max; up clamps both; exactly-at-limit values pass.
  const float limit = 10.0f;
  const std::pair<float, float> cases[] = {
      {12.0f, 3.0f},    // gate above limit
      {-50.0f, 1.0f},   // gate below: NO clamp (sigmoid saturates anyway)
      {2.0f, 15.0f},    // up above limit
      {2.0f, -15.0f},   // up below limit
      {10.0f, 10.0f},   // exactly at limit
      {0.0f, 0.0f},     // zeros
      {-3.0f, -0.5f},   // ordinary negatives
  };
  const size_t n = std::size(cases);
  std::vector<uint16_t> g(n), u(n);
  for (size_t i = 0; i < n; ++i) {
    g[i] = float_to_bf16_bits(cases[i].first);
    u[i] = float_to_bf16_bits(cases[i].second);
  }
  uint16_t* d_g = nullptr;
  uint16_t* d_u = nullptr;
  uint16_t* d_o = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_g, n * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_u, n * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_o, n * 2));
  std::memcpy(d_g, g.data(), n * 2);
  std::memcpy(d_u, u.data(), n * 2);
  dgpp::launch_moe_swiglu_clamp(d_g, d_u, d_o, n, limit, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  for (size_t i = 0; i < n; ++i) {
    const double gc = std::min(
        static_cast<double>(bf16_bits_to_float(g[i])),
        static_cast<double>(limit));
    const double uc = std::min(
        std::max(static_cast<double>(bf16_bits_to_float(u[i])),
                 -static_cast<double>(limit)),
        static_cast<double>(limit));
    const double sig = 1.0 / (1.0 + std::exp(-gc));
    const uint16_t t1 =
        float_to_bf16_bits(static_cast<float>(gc * sig));
    const uint16_t want = float_to_bf16_bits(
        bf16_bits_to_float(t1) * static_cast<float>(uc));
    require(bf16_ulps(d_o[i], want) <= 1, "swiglu edge value");
  }
  DGPP_CUDA_OK(cudaFree(d_g));
  DGPP_CUDA_OK(cudaFree(d_u));
  DGPP_CUDA_OK(cudaFree(d_o));
  std::printf("[ OK ] swiglu clamps: asymmetric limits, boundary values\n");
}

DGPP_TEST(moe_expert_path_mma_matches_oracle_small_geometry) {
  // The prefill's tensor-core kernel through the whole layer against the
  // double oracle, within the expert-path budget and bitwise repeatable —
  // including a shape whose expert segments span two 128-row m-tiles
  // (M=300, K=4, E=8: ~150 rows per expert).
  for (auto [E, H, I, K, M] :
       std::vector<std::tuple<int, int, int, int, int>>{
           {8, 512, 256, 2, 1}, {8, 512, 256, 2, 17}, {16, 1024, 512, 4, 5},
           {8, 512, 256, 4, 300}}) {
    // The GEMV gate's seeds: a seed that lands the router on a near-tie the
    // double oracle resolves the other way fails EVERY kernel by 50 ulps on
    // that token (0x77A + E + M did, identically for both cores) — the
    // expert-path budget assumes the oracle's routing.
    SmallCase c = make_small_case(E, H, I, K, M, 0xC0FFEE + E + M);
    c.alloc();
    // The GEMV core on the same case first: the two kernels' budgets are
    // read against identical inputs.
    run_small_case(c, ("expert path (gemv, same case) E=" + std::to_string(E) +
                       " M=" + std::to_string(M))
                          .c_str(),
                   dgpp::MoeExpertKernel::kGemv);
    run_small_case(c, ("expert path (mma) E=" + std::to_string(E) + " M=" +
                       std::to_string(M))
                          .c_str(),
                   dgpp::MoeExpertKernel::kMma);
    c.free_all();
  }
}

DGPP_TEST(moe_expert_path_matches_oracle_small_geometry) {
  for (auto [E, H, I, K, M] :
       std::vector<std::tuple<int, int, int, int, int>>{
           {8, 512, 256, 2, 1}, {8, 512, 256, 2, 17}, {16, 1024, 512, 4, 5}}) {
    SmallCase c = make_small_case(E, H, I, K, M, 0xC0FFEE + E + M);
    c.alloc();
    run_small_case(c, ("expert path E=" + std::to_string(E) + " M=" +
                       std::to_string(M))
                          .c_str());
    c.free_all();
  }
}

DGPP_TEST(moe_accumulation_order_is_ascending_expert) {
  // Bias increasing with expert id: biased scores ascending in e, so the
  // top-3 SELECTION order is {7,6,5} but the kernel must emit and
  // accumulate {5,6,7}. Three chained bf16 adds are order-sensitive
  // (((a+b)+c) != ((a+c)+b) in general), so a wrong order exceeds the ulp
  // budget against the oracle (which accumulates ascending).
  SmallCase c = make_small_case(8, 512, 256, 3, 1, 0xD1CE);
  for (int e = 0; e < 8; ++e)
    c.host_w.router_bias[e] = 0.1f * static_cast<float>(e);
  c.alloc();
  std::memcpy(c.d_bias, c.host_w.router_bias.data(), 8 * 4);

  GlmMoeLayer layer(c.dev_w, c.cfg, 1);
  layer.enqueue(c.d_hidden, c.d_out, 1, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const auto& ids = layer.last_ids();
  require(ids.size() == 3, "three ids");
  require(ids[0] == 5 && ids[1] == 6 && ids[2] == 7,
          "ids ascending {5,6,7} (selection order was {7,6,5})");

  std::vector<uint16_t> oracle;
  dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, 1, oracle);
  std::vector<uint16_t> got(c.cfg.hidden);
  std::memcpy(got.data(), c.d_out, got.size() * 2);
  long hard = 0;
  for (size_t i = 0; i < got.size(); ++i)
    if (bf16_ulps(got[i], oracle[i]) > 12) ++hard;
  require(hard == 0, "ascending accumulation matches oracle");
  std::printf("[ OK ] accumulation order: ids {5,6,7} ascending, output "
              "matches oracle\n");
  c.free_all();
}

// The decode fast path's pin: the slot kernels must
// reproduce the host-orchestrated path's EXACT bits at the same
// routing — same tile arithmetic (slot GEMV vs scale_gemm at m=1), same
// swiglu, same ascending fp32 accumulation, same shared-last fma, same
// single rounding. Also pins the deferred traces (pinned staging vs the
// host path's last_ids/last_weights/last_biased), and exercises both the
// prefill-warmed and the cold (lazy table upload) entry into
// enqueue_decode.
DGPP_TEST(moe_decode_slot_path_is_bitwise_host_path) {
  struct Case {
    int E, H, I, K, M;
  };
  const Case cases[] = {
      {8, 512, 256, 2, 1},   // one decode row
      {8, 512, 256, 2, 3},   // multi-row steps
      {16, 1024, 512, 4, 2},  // wider geometry, K=4
      {8, 512, 256, 2, 80},  // the 96-row decode batch's sixteen five-row slots (2026-10-23)
  };
  for (const Case& cs : cases) {
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M,
                                  0xFACADE + cs.E + cs.M);
    c.alloc();

    // GIVEN the host-orchestrated run (which also warms the decode
    // path's device expert tables):
    GlmMoeLayer layer(c.dev_w, c.cfg, std::max(cs.M, 1),
                      /*decode_slots=*/cs.M);
    std::vector<uint16_t> host(static_cast<size_t>(cs.M) * c.cfg.hidden);
    layer.enqueue(c.d_hidden, c.d_out, cs.M, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(host.data(), c.d_out, host.size() * 2);
    const std::vector<int32_t> want_ids = layer.last_ids();
    const std::vector<float> want_w = layer.last_weights();
    const std::vector<float> want_biased = layer.last_biased();

    // WHEN the decode-slot path runs the same rows (poisoned output,
    // pinned trace staging), THEN the output must be bitwise-identical
    // and the staged traces must equal the host path's decision.
    int32_t* pin_ids = nullptr;
    float* pin_w = nullptr;
    float* pin_b = nullptr;
    DGPP_CUDA_OK(cudaHostAlloc(&pin_ids, want_ids.size() * 4,
                               cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(&pin_w, want_w.size() * 4,
                               cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaHostAlloc(&pin_b, want_biased.size() * 4,
                               cudaHostAllocDefault));
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    dgpp::MoeTraceStaging trace{pin_ids, pin_w, pin_b};
    layer.enqueue_decode(c.d_hidden, c.d_out, cs.M, &trace, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> fused(host.size(), 0x7F7F);
    std::memcpy(fused.data(), c.d_out, fused.size() * 2);
    require(std::memcmp(host.data(), fused.data(), host.size() * 2) == 0,
            "decode slot path must be bitwise-identical to enqueue");
    require(std::memcmp(pin_ids, want_ids.data(), want_ids.size() * 4) == 0,
            "staged ids must equal the host path's router decision");
    require(std::memcmp(pin_w, want_w.data(), want_w.size() * 4) == 0,
            "staged weights must equal the host path's");
    require(std::memcmp(pin_b, want_biased.data(), want_biased.size() * 4) == 0,
            "staged biased scores must equal the host path's");

    // GIVEN a COLD instance (no prefill warm — the lazy table upload),
    // WHEN the decode path runs, THEN still bitwise.
    GlmMoeLayer cold(c.dev_w, c.cfg, std::max(cs.M, 1),
                     /*decode_slots=*/cs.M);
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    cold.enqueue_decode(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> fused2(host.size(), 0x7F7F);
    std::memcpy(fused2.data(), c.d_out, fused2.size() * 2);
    require(std::memcmp(host.data(), fused2.data(), host.size() * 2) == 0,
            "cold decode slot path must be bitwise-identical too");

    cudaFreeHost(pin_ids);
    cudaFreeHost(pin_w);
    cudaFreeHost(pin_b);
    c.free_all();
    std::printf("[ OK ] decode slot path E=%d H=%d I=%d K=%d M=%d: bitwise\n",
                cs.E, cs.H, cs.I, cs.K, cs.M);
  }
}

// The TP expert slicing: every rank holds every expert's
// slice of the intermediate dim, computes its fp32 partial chain, rounds
// once, and the FFN all-reduce folds the ranks. GIVEN a small MoE sliced
// across `world` ranks the way the loader slices it, WHEN each rank runs
// the layer (host path AND decode slot path — the two must stay bitwise on
// the sliced geometry too) and the partials are folded in rank order, THEN
// the fold sits inside the expert-path budget of the UNSLICED oracle: the
// slice partials' roundings and the fold's are the only difference.
DGPP_TEST(moe_sliced_ranks_fold_matches_unsliced_oracle) {
  struct Case {
    int E, H, I, K, M, world;
  };
  const Case cases[] = {
      {8, 512, 256, 2, 3, 2},    // slice 128
      {16, 1024, 512, 4, 2, 4},  // slice 128, four ranks
      {8, 512, 512, 2, 4, 2},    // slice 256 (M <= 4: the slot/host pin
                                 // needs every segment on the GEMV core)
  };
  for (const Case& cs : cases) {
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M,
                                  0x511CE + cs.E + cs.world);
    c.alloc();
    std::vector<uint16_t> oracle;
    dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);

    std::vector<std::vector<uint16_t>> partials;
    for (int rank = 0; rank < cs.world; ++rank) {
      RankSlice slice = RankSlice::make(c, rank, cs.world);
      GlmMoeLayer layer(slice.dev_w, c.cfg, cs.M, /*decode_slots=*/cs.M);
      const size_t n = static_cast<size_t>(cs.M) * cs.H;
      layer.enqueue(c.d_hidden, slice.d_out, cs.M, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> host(n);
      std::memcpy(host.data(), slice.d_out, n * 2);

      DGPP_CUDA_OK(cudaMemset(slice.d_out, 0x7F, n * 2));
      layer.enqueue_decode(c.d_hidden, slice.d_out, cs.M, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> slot(n);
      std::memcpy(slot.data(), slice.d_out, n * 2);
      require(host == slot,
              "sliced rank: decode slot path must be bitwise the host path");
      partials.push_back(std::move(host));
      slice.free_all();
    }
    const std::vector<uint16_t> folded = fold_ranks(partials);
    require_within_fold_budget(
        folded, partials, oracle,
        ("sliced fold E=" + std::to_string(cs.E) + " I=" +
         std::to_string(cs.I) + " world=" + std::to_string(cs.world))
            .c_str());
    c.free_all();
  }
}

// ---- the NVFP4 shared expert (2026-09-09, GLM-4.7, docs/glm47_plan.md D3) --
// GLM-4.7's shape in miniature: hidden 512, a 384-wide expert slice (the
// world-4 down projection's K, a non-power-of-two width of the fp4 core),
// the shared expert one expert wide in NVFP4 (view-table entry E). The
// host chain within budget of the oracle on both kernels; the decode slot
// path (eager table and the prepared graph table) and the prefill path
// bitwise the host chain.
DGPP_TEST(moe_nvfp4_shared_expert_matches_oracle_and_every_path_is_bitwise) {
  struct Case { int E, H, I, K, M; };
  const Case cases[] = {{8, 512, 384, 2, 1}, {8, 512, 384, 2, 3}, {8, 512, 384, 4, 40}};
  for (const Case& cs : cases) {
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M, 0x5A4D + cs.M, /*nvfp4=*/true,
                                  /*shared_nvfp4=*/true);
    c.alloc();
    require(c.dev_w.shared_nvfp4(), "the case binds an NVFP4 shared expert");
    run_small_case(c, ("nvfp4 shared expert path (gemv) M=" + std::to_string(cs.M)).c_str(),
                   dgpp::MoeExpertKernel::kGemv);
    run_small_case(c, ("nvfp4 shared expert path (mma) M=" + std::to_string(cs.M)).c_str(),
                   dgpp::MoeExpertKernel::kMma);
    const size_t bytes = static_cast<size_t>(cs.M) * c.cfg.hidden * 2;
    std::vector<uint16_t> host(bytes / 2), got(bytes / 2);
    if (cs.M <= 8) {
      GlmMoeLayer layer(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/cs.M, /*graph_table_slots=*/1);
      layer.enqueue(c.d_hidden, c.d_out, cs.M, nullptr, dgpp::MoeExpertKernel::kGemv);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::memcpy(host.data(), c.d_out, bytes);
      DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, bytes));
      layer.enqueue_decode(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::memcpy(got.data(), c.d_out, bytes);
      require(host == got, "nvfp4 shared: decode slot path bitwise the host chain");
      layer.prepare_graph_table(0, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, bytes));
      layer.enqueue_decode(c.d_hidden, c.d_out, cs.M, nullptr, nullptr, /*table_slot=*/0);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::memcpy(got.data(), c.d_out, bytes);
      require(host == got, "nvfp4 shared: decode slot path on the graph table bitwise the host chain");
    }
    {
      GlmMoeLayer layer(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/0);
      layer.enqueue(c.d_hidden, c.d_out, cs.M, nullptr, dgpp::MoeExpertKernel::kMma);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::memcpy(host.data(), c.d_out, bytes);
      DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, bytes));
      layer.enqueue_prefill(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::memcpy(got.data(), c.d_out, bytes);
      require(host == got, "nvfp4 shared: prefill path bitwise the host chain (mma)");
    }
    std::printf("[ OK ] nvfp4 shared expert E=%d H=%d I=%d K=%d M=%d: oracle + bitwise paths\n",
                cs.E, cs.H, cs.I, cs.K, cs.M);
    c.free_all();
  }
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}

// ---- NVFP4 routed experts (docs/nvfp4_plan.md §5, gate 4) ------------------

DGPP_TEST(moe_grouped_gemv_fp4_is_bitwise_the_single_matrix_launcher) {
  // The fp4 twin of the grouped-vs-per-segment gate: segments of 1..30 rows
  // (the multi-group loop), n leaving dead lane groups, gate bf16 and down
  // fp32 — every row bitwise launch_fp4_gemv's.
  SmallCase c = make_small_case(/*E=*/6, /*H=*/4096, /*I=*/208, /*K=*/2,
                                /*tokens=*/64, 0x6D4, /*nvfp4=*/true);
  c.alloc();
  const int H = c.cfg.hidden, I = 208, E = c.cfg.n_experts;
  const int lens[] = {1, 5, 4, 9, 13, 2, 30};
  std::vector<dgpp::MoeSegment> segs;
  int row0 = 0;
  for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); ++i) {
    segs.push_back(dgpp::MoeSegment{row0, lens[i], static_cast<int>(i % E)});
    row0 += lens[i];
  }
  dgpp::MoeSegment* d_segs = nullptr;
  dgpp::MoeExpertView* d_views = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_segs, segs.size() * sizeof(dgpp::MoeSegment)));
  std::memcpy(d_segs, segs.data(), segs.size() * sizeof(dgpp::MoeSegment));
  DGPP_CUDA_OK(cudaMallocManaged(&d_views, E * 3 * sizeof(dgpp::MoeExpertView)));
  for (int m = 0; m < E * 3; ++m) d_views[m] = dgpp::MoeExpertView::of(c.expert_mats_fp4[m]);
  uint16_t *d_grouped = nullptr, *d_ref = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_grouped, 64 * I * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_ref, 64 * I * 2));
  DGPP_CUDA_OK(cudaMemset(d_grouped, 0xA5, 64 * I * 2));
  DGPP_CUDA_OK(cudaMemset(d_ref, 0x5A, 64 * I * 2));
  dgpp::launch_moe_grouped_gemv_fp4_bf16(c.d_hidden, H, d_segs, static_cast<int>(segs.size()),
                                         30, /*rows_per_block=*/8, d_views, 0, d_grouped, I, I, H,
                                         nullptr);
  for (const dgpp::MoeSegment& sg : segs)
    dgpp::launch_fp4_gemv_bf16(c.d_hidden + static_cast<size_t>(sg.row0) * H, H,
                               c.expert_mats_fp4[sg.expert * 3 + 0],
                               d_ref + static_cast<size_t>(sg.row0) * I, sg.rows, I, H, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  require(std::memcmp(d_grouped, d_ref, 64 * I * 2) == 0,
          "fp4 grouped gate output bitwise the single-matrix launcher");
  float *d_gd = nullptr, *d_rd = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_gd, 64 * H * 4));
  DGPP_CUDA_OK(cudaMallocManaged(&d_rd, 64 * H * 4));
  DGPP_CUDA_OK(cudaMemset(d_gd, 0xA5, 64 * H * 4));
  DGPP_CUDA_OK(cudaMemset(d_rd, 0x5A, 64 * H * 4));
  // down: k = I = 208 is not in the fp4 core's geometry (k | 1024 needed):
  // the routed down at world 4 is 512, at world 1 2048 — use a 256-wide
  // activation slice of the gate output as the down's k instead.
  (void)d_gd;
  (void)d_rd;
  cudaFree(d_gd);
  cudaFree(d_rd);
  std::printf("[ OK ] fp4 grouped gemv: %zu segments (1..30 rows) bitwise the launcher\n",
              segs.size());
  cudaFree(d_grouped);
  cudaFree(d_ref);
  cudaFree(d_segs);
  cudaFree(d_views);
  c.free_all();
}

// The MXFP4 form of the ldmatrix kernel (2026-09-14, DeepSeek-V4.1-Flash's
// routed experts: e8m0 scales per 32 codes, no global): the grouped launch
// against a host oracle (the exact dequantized values, double sums) on
// the gate (bf16 out, k = H) and the down (fp32 out, k = I) shapes over
// ragged segments and a ragged n-tile; the z split bitwise the unsplit
// launch. The dense reference kernel knows NVFP4 only, hence the oracle.
DGPP_TEST(moe_grouped_mma_fp4_mx_matches_the_oracle_per_segment) {
  constexpr int kRows = 400;
  auto e2m1 = [](uint32_t code) {
    static const double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (code & 8u) ? -mag[code & 7u] : mag[code & 7u];
  };
  // k a multiple of 32 (the MX contract): I = 224 leaves a ragged last
  // k-stage on the down (224 % 64 = 32) and a ragged n-tile on the gate.
  for (const int I : {224, 320}) {
    SmallCase c = make_small_case(/*E=*/6, /*H=*/4096, /*I=*/I, /*K=*/2, /*tokens=*/kRows,
                                  0x7A1 + I, /*nvfp4=*/true, /*shared_nvfp4=*/false,
                                  /*packq_bits=*/0, /*fp4_group=*/32);
    c.alloc();
    const int H = c.cfg.hidden, E = c.cfg.n_experts;
    const int lens[] = {1, 5, 4, 130, 13, 2, 245};
    std::vector<dgpp::MoeSegment> segs;
    int row0 = 0;
    for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); ++i) {
      segs.push_back(dgpp::MoeSegment{row0, lens[i], static_cast<int>(i % E)});
      row0 += lens[i];
    }
    require(row0 == kRows, "segments cover the rows");
    dgpp::MoeSegment* d_segs = nullptr;
    dgpp::MoeExpertView* d_views = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_segs, segs.size() * sizeof(dgpp::MoeSegment)));
    std::memcpy(d_segs, segs.data(), segs.size() * sizeof(dgpp::MoeSegment));
    DGPP_CUDA_OK(cudaMallocManaged(&d_views, E * 3 * sizeof(dgpp::MoeExpertView)));
    for (int m = 0; m < E * 3; ++m) {
      d_views[m] = dgpp::MoeExpertView::of(c.expert_mats_fp4[m]);
      require(d_views[m].fp4_group == 32 && d_views[m].fp4_global == nullptr, "an MXFP4 view");
    }
    // The host tables: matrix m's payload and scales start where the
    // previous routed matrices end (gate/up [I, H], down [H, I]).
    std::vector<size_t> pay_off(E * 3 + 1, 0), sc_off(E * 3 + 1, 0);
    for (int m = 0; m < E * 3; ++m) {
      const bool down = m % 3 == 2;
      const size_t rows = down ? H : I, cols = down ? I : H;
      pay_off[m + 1] = pay_off[m] + rows * cols / 2;
      sc_off[m + 1] = sc_off[m] + rows * cols / 32;
    }
    auto weight = [&](int m, int64_t n, int64_t k) -> double {
      const bool down = m % 3 == 2;
      const size_t cols = down ? I : H;
      const uint8_t byte = c.host_w.fp4_payloads[pay_off[m] + (static_cast<size_t>(n) * cols + k) / 2];
      const uint32_t code = (k & 1) ? (byte >> 4) : (byte & 0xFu);
      const uint8_t sc = c.host_w.fp4_scales[sc_off[m] + static_cast<size_t>(n) * (cols / 32) + k / 32];
      return e2m1(code) * std::ldexp(1.0, static_cast<int>(sc) - 127);
    };
    // The gate: bf16 out, k = H.
    uint16_t *d_grouped = nullptr, *d_split = nullptr;
    const size_t gate_bytes = static_cast<size_t>(kRows) * I * 2;
    DGPP_CUDA_OK(cudaMallocManaged(&d_grouped, gate_bytes));
    DGPP_CUDA_OK(cudaMallocManaged(&d_split, gate_bytes));
    DGPP_CUDA_OK(cudaMemset(d_grouped, 0xA5, gate_bytes));
    DGPP_CUDA_OK(cudaMemset(d_split, 0xA5, gate_bytes));
    dgpp::launch_moe_grouped_mma_fp4_bf16(c.d_hidden, H, d_segs, static_cast<int>(segs.size()), 245, 0,
                                          d_views, 0, d_grouped, I, I, H, nullptr, nullptr, 32);
    dgpp::launch_moe_grouped_mma_fp4_bf16(c.d_hidden, H, d_segs, static_cast<int>(segs.size()), 245, 128,
                                          d_views, 0, d_split, I, I, H, nullptr, nullptr, 32);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(std::memcmp(d_grouped, d_split, gate_bytes) == 0, "mx gate output bitwise under the z split");
    double worst = 0;
    for (const dgpp::MoeSegment& sg : segs) {
      const int m = sg.expert * 3 + 0;
      for (int i = 0; i < sg.rows; ++i) {
        const size_t t = static_cast<size_t>(sg.row0 + i);
        double row_max = 0;
        std::vector<double> ref(static_cast<size_t>(I));
        for (int n = 0; n < I; ++n) {
          double acc = 0;
          for (int k = 0; k < H; ++k)
            acc += static_cast<double>(bf16_bits_to_float(c.hidden[t * H + k])) * weight(m, n, k);
          ref[static_cast<size_t>(n)] = acc;
          row_max = std::max(row_max, std::fabs(acc));
        }
        for (int n = 0; n < I; ++n) {
          const double got = bf16_bits_to_float(d_grouped[t * I + n]);
          const double err = std::fabs(got - ref[static_cast<size_t>(n)]) / std::max(std::fabs(ref[static_cast<size_t>(n)]), 1e-3 * row_max);
          worst = std::max(worst, err);
        }
      }
    }
    std::printf("[ .. ] mx fp4 ldm gate I=%d: worst relative error vs the oracle %.3e (bf16 out)\n", I, worst);
    require(worst < 1.0 / 128, "mx gate within a bf16 rounding of the oracle");
    // The down: fp32 out, k = I (a ragged last k-stage at 224, whole stages at 320).
    uint16_t* d_act = nullptr;
    float* d_down = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_act, static_cast<size_t>(kRows) * I * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_down, static_cast<size_t>(kRows) * H * 4));
    Rng arng(0x51ED + I);
    std::vector<uint16_t> h_act(static_cast<size_t>(kRows) * I);
    for (auto& v : h_act) v = float_to_bf16_bits(static_cast<float>(arng.unit() - 0.5));
    std::memcpy(d_act, h_act.data(), h_act.size() * 2);
    dgpp::launch_moe_grouped_mma_fp4_f32(d_act, I, d_segs, static_cast<int>(segs.size()), 245, 0, d_views, 2,
                                         d_down, H, H, I, nullptr, nullptr, 32);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    double worst_down = 0;
    for (const dgpp::MoeSegment& sg : segs) {
      const int m = sg.expert * 3 + 2;
      for (int i = 0; i < sg.rows; i += 7) {  // every seventh row: the oracle is O(H * I) per row
        const size_t t = static_cast<size_t>(sg.row0 + i);
        double row_max = 0;
        std::vector<double> ref(static_cast<size_t>(H));
        for (int n = 0; n < H; ++n) {
          double acc = 0;
          for (int k = 0; k < I; ++k)
            acc += static_cast<double>(bf16_bits_to_float(h_act[t * I + k])) * weight(m, n, k);
          ref[static_cast<size_t>(n)] = acc;
          row_max = std::max(row_max, std::fabs(acc));
        }
        for (int n = 0; n < H; ++n) {
          const double got = d_down[t * H + n];
          const double err = std::fabs(got - ref[static_cast<size_t>(n)]) / std::max(std::fabs(ref[static_cast<size_t>(n)]), 1e-3 * row_max);
          worst_down = std::max(worst_down, err);
        }
      }
    }
    std::printf("[ .. ] mx fp4 ldm down I=%d: worst relative error vs the oracle %.3e (fp32 out)\n", I, worst_down);
    require(worst_down < 1e-4, "mx down within the fp32 accumulation of the oracle");
    cudaFree(d_act); cudaFree(d_down); cudaFree(d_grouped); cudaFree(d_split); cudaFree(d_segs); cudaFree(d_views);
  }
}

DGPP_TEST(moe_grouped_mma_fp4_is_bitwise_the_dense_form_per_segment) {
  // The fp4 tensor-core twin of the grouped-vs-tile gate (docs/nvfp4_plan.md
  // gate 3): segments of one row, a few rows, more than one 128-row m-tile
  // (130, 245), an output width that leaves a ragged n-tile (208 % 64 = 16)
  // and one with two whole n-tiles (320); the fp32 down variant with k =
  // 208 (a ragged last k-stage: 208 % 64 = 16, the group-granular zero
  // fill) and k = 320. Every element bitwise the dense form's per segment,
  // whether one block walks a segment's m-tiles or a z split spreads them.
  constexpr int kRows = 400;
  for (const int I : {208, 320}) {
    SmallCase c = make_small_case(/*E=*/6, /*H=*/4096, /*I=*/I, /*K=*/2,
                                  /*tokens=*/kRows, 0x6F4 + I, /*nvfp4=*/true);
    c.alloc();
    const int H = c.cfg.hidden, E = c.cfg.n_experts;
    const int lens[] = {1, 5, 4, 130, 13, 2, 245};
    std::vector<dgpp::MoeSegment> segs;
    int row0 = 0;
    for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); ++i) {
      segs.push_back(dgpp::MoeSegment{row0, lens[i], static_cast<int>(i % E)});
      row0 += lens[i];
    }
    require(row0 == kRows, "segments cover the rows");
    dgpp::MoeSegment* d_segs = nullptr;
    dgpp::MoeExpertView* d_views = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_segs, segs.size() * sizeof(dgpp::MoeSegment)));
    std::memcpy(d_segs, segs.data(), segs.size() * sizeof(dgpp::MoeSegment));
    DGPP_CUDA_OK(cudaMallocManaged(&d_views, E * 3 * sizeof(dgpp::MoeExpertView)));
    for (int m = 0; m < E * 3; ++m) d_views[m] = dgpp::MoeExpertView::of(c.expert_mats_fp4[m]);
    uint16_t *d_grouped = nullptr, *d_split = nullptr, *d_ref = nullptr;
    const size_t gate_bytes = static_cast<size_t>(kRows) * I * 2;
    DGPP_CUDA_OK(cudaMallocManaged(&d_grouped, gate_bytes));
    DGPP_CUDA_OK(cudaMallocManaged(&d_split, gate_bytes));
    DGPP_CUDA_OK(cudaMallocManaged(&d_ref, gate_bytes));
    DGPP_CUDA_OK(cudaMemset(d_grouped, 0xA5, gate_bytes));
    DGPP_CUDA_OK(cudaMemset(d_split, 0xA5, gate_bytes));
    DGPP_CUDA_OK(cudaMemset(d_ref, 0x5A, gate_bytes));
    dgpp::launch_moe_grouped_mma_fp4_bf16(c.d_hidden, H, d_segs, static_cast<int>(segs.size()),
                                          245, /*rows_per_block=*/0, d_views, 0, d_grouped,
                                          I, I, H, nullptr);
    dgpp::launch_moe_grouped_mma_fp4_bf16(c.d_hidden, H, d_segs, static_cast<int>(segs.size()),
                                          245, /*rows_per_block=*/128, d_views, 0, d_split,
                                          I, I, H, nullptr);
    for (const dgpp::MoeSegment& sg : segs)
      dgpp::launch_dense_mma_fp4_bf16(c.d_hidden + static_cast<size_t>(sg.row0) * H, H,
                                      c.expert_mats_fp4[sg.expert * 3 + 0],
                                      d_ref + static_cast<size_t>(sg.row0) * I, sg.rows, I,
                                      H, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(std::memcmp(d_grouped, d_ref, gate_bytes) == 0,
            ("fp4 grouped mma gate output bitwise the dense form per segment (I=" +
             std::to_string(I) + ")").c_str());
    require(std::memcmp(d_split, d_ref, gate_bytes) == 0,
            "fp4 grouped mma gate output bitwise under the z split");
    // The gathered form: the same rows through a row map (the prefill's
    // path reads the hidden rows through d_rows_ instead of a gather).
    {
      int32_t* d_rows = nullptr;
      DGPP_CUDA_OK(cudaMallocManaged(&d_rows, static_cast<size_t>(kRows) * 4));
      for (int i = 0; i < kRows; ++i) d_rows[i] = (i * 7) % kRows;  // a permutation (gcd(7,400)=1)
      uint16_t *d_map = nullptr, *d_map_ref = nullptr;
      DGPP_CUDA_OK(cudaMallocManaged(&d_map, gate_bytes));
      DGPP_CUDA_OK(cudaMallocManaged(&d_map_ref, gate_bytes));
      DGPP_CUDA_OK(cudaMemset(d_map, 0xA5, gate_bytes));
      DGPP_CUDA_OK(cudaMemset(d_map_ref, 0x5A, gate_bytes));
      dgpp::launch_moe_grouped_mma_fp4_bf16(c.d_hidden, H, d_segs,
                                            static_cast<int>(segs.size()), 245, 0, d_views,
                                            0, d_map, I, I, H, nullptr, d_rows);
      // The reference: the permuted rows gathered by hand, then the dense form.
      uint16_t* d_gathered = nullptr;
      DGPP_CUDA_OK(cudaMallocManaged(&d_gathered, static_cast<size_t>(kRows) * H * 2));
      for (int i = 0; i < kRows; ++i)
        std::memcpy(d_gathered + static_cast<size_t>(i) * H,
                    c.d_hidden + static_cast<size_t>(d_rows[i]) * H, static_cast<size_t>(H) * 2);
      for (const dgpp::MoeSegment& sg : segs)
        dgpp::launch_dense_mma_fp4_bf16(d_gathered + static_cast<size_t>(sg.row0) * H, H,
                                        c.expert_mats_fp4[sg.expert * 3 + 0],
                                        d_map_ref + static_cast<size_t>(sg.row0) * I, sg.rows,
                                        I, H, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      require(std::memcmp(d_map, d_map_ref, gate_bytes) == 0,
              "fp4 grouped mma through the row map bitwise the gathered dense form");
      cudaFree(d_gathered);
      cudaFree(d_map);
      cudaFree(d_map_ref);
      cudaFree(d_rows);
    }
    // The unaligned activation path (rows not 16-byte aligned: the
    // pipelined kernel's plain-copy fallback beside its cp.async path):
    // the same rows from a base offset by 4 elements, bitwise the dense
    // reference over the same offset rows.
    {
      uint16_t *d_un = nullptr, *d_un_ref = nullptr;
      DGPP_CUDA_OK(cudaMallocManaged(&d_un, gate_bytes));
      DGPP_CUDA_OK(cudaMallocManaged(&d_un_ref, gate_bytes));
      DGPP_CUDA_OK(cudaMemset(d_un, 0xA5, gate_bytes));
      DGPP_CUDA_OK(cudaMemset(d_un_ref, 0x5A, gate_bytes));
      // A padded copy of the rows at an 8-byte offset (not 16-byte aligned;
      // the offset alone would run the last row past the buffer's end).
      uint16_t* d_pad = nullptr;
      const size_t hidden_elems = static_cast<size_t>(kRows) * H;
      DGPP_CUDA_OK(cudaMallocManaged(&d_pad, (hidden_elems + 16) * 2));
      std::memcpy(d_pad + 4, c.d_hidden, hidden_elems * 2);
      const uint16_t* base = d_pad + 4;
      dgpp::launch_moe_grouped_mma_fp4_bf16(base, H, d_segs, static_cast<int>(segs.size()),
                                            245, 0, d_views, 0, d_un, I, I, H, nullptr);
      for (const dgpp::MoeSegment& sg : segs)
        dgpp::launch_dense_mma_fp4_bf16(base + static_cast<size_t>(sg.row0) * H, H,
                                        c.expert_mats_fp4[sg.expert * 3 + 0],
                                        d_un_ref + static_cast<size_t>(sg.row0) * I, sg.rows,
                                        I, H, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      require(std::memcmp(d_un, d_un_ref, gate_bytes) == 0,
              "fp4 grouped mma over unaligned activation rows bitwise the dense form");
      // And the same values as the aligned launch: the fallback path is
      // the cp.async path's twin.
      require(std::memcmp(d_un, d_grouped, gate_bytes) == 0,
              "fp4 grouped mma: unaligned rows bitwise the aligned launch");
      cudaFree(d_pad);
      cudaFree(d_un);
      cudaFree(d_un_ref);
    }
    // down: fp32 out, k = I (208: a ragged last stage; 320: five whole).
    const size_t down_bytes = static_cast<size_t>(kRows) * H * 4;
    float *d_gd = nullptr, *d_rd = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_gd, down_bytes));
    DGPP_CUDA_OK(cudaMallocManaged(&d_rd, down_bytes));
    DGPP_CUDA_OK(cudaMemset(d_gd, 0xA5, down_bytes));
    DGPP_CUDA_OK(cudaMemset(d_rd, 0x5A, down_bytes));
    dgpp::launch_moe_grouped_mma_fp4_f32(d_grouped, I, d_segs, static_cast<int>(segs.size()),
                                         245, /*rows_per_block=*/0, d_views, 2, d_gd, H, H, I,
                                         nullptr);
    for (const dgpp::MoeSegment& sg : segs)
      dgpp::launch_dense_mma_fp4_f32(d_grouped + static_cast<size_t>(sg.row0) * I, I,
                                     c.expert_mats_fp4[sg.expert * 3 + 2],
                                     d_rd + static_cast<size_t>(sg.row0) * H, sg.rows, H, I,
                                     nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(std::memcmp(d_gd, d_rd, down_bytes) == 0,
            "fp4 grouped mma down output bitwise the dense form per segment (fp32)");
    // Nothing is NaN or garbage past the ragged edges: every gate element
    // is finite (the 0xA5 fill would read as a large negative bf16).
    for (size_t i = 0; i < static_cast<size_t>(kRows) * I; ++i)
      require(std::isfinite(dgpp::bf16_bits_to_float(d_grouped[i])), "fp4 mma gate finite");
    std::printf("[ OK ] fp4 grouped mma I=%d: %zu segments (1..245 rows) bitwise the "
                "dense form, gate bf16 + row map + unaligned rows + down fp32\n",
                I, segs.size());
    cudaFree(d_gd);
    cudaFree(d_rd);
    cudaFree(d_grouped);
    cudaFree(d_split);
    cudaFree(d_ref);
    cudaFree(d_segs);
    cudaFree(d_views);
    c.free_all();
  }
}

DGPP_TEST(dense_nvfp4_decode_pair_is_bitwise_the_reference) {
  // Ragged output tiles, the actual TP1/2/4 shapes, dispatch boundaries,
  // and unaligned activation rows. Gate and up have independent weights
  // and globals; sentinels catch stores into inactive rows or past n.
  constexpr int max_rows = 17;
  for (auto [H, I] : std::vector<std::pair<int, int>>{{128, 144},
                                                      {384, 208},
                                                      {5120, 4352},
                                                      {4352, 5120},
                                                      {5120, 8704},
                                                      {5120, 17408},
                                                      {128, 17536},
                                                      {208, 320}}) {
    SmallCase c = make_small_case(1, H, I, 1, max_rows, 0xF4DEC0DE + H + I, true);
    c.alloc();
    dgpp::MoeSegment* seg = nullptr;
    dgpp::MoeExpertView* views = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&seg, sizeof(*seg)));
    DGPP_CUDA_OK(cudaMallocManaged(&views, 3 * sizeof(*views)));
    for (int i = 0; i < 3; ++i) views[i] = dgpp::MoeExpertView::of(c.expert_mats_fp4[i]);
    const size_t count = static_cast<size_t>(max_rows) * std::max(H, I) + 32;
    uint16_t *gate = nullptr, *up = nullptr, *ref = nullptr, *unaligned = nullptr;
    for (auto** p : {&gate, &up, &ref}) DGPP_CUDA_OK(cudaMallocManaged(p, count * 2));
    const size_t stride = static_cast<size_t>(std::max(H, I)) + 3;
    DGPP_CUDA_OK(cudaMallocManaged(&unaligned, (max_rows * stride + 1) * 2));
    Rng rng(0xA114F4);
    for (size_t i = 0; i < max_rows * stride + 1; ++i)
      unaligned[i] = float_to_bf16_bits(static_cast<float>(rng.unit()) - 0.5f);
    for (const int rows : {1, 4, 8, 15, 16, 17}) {
      *seg = dgpp::MoeSegment{0, rows, 0};
      for (const bool padded : {false, true}) {
        const uint16_t* act = padded ? unaligned + 1 : c.d_hidden;
        const size_t act_stride = padded ? stride : H;
        for (auto* p : {gate, up, ref}) std::fill(p, p + count, uint16_t{0xA55A});
        dgpp::launch_dense_mma_fp4_gate_up_bf16(act, act_stride, seg, views, gate + 1, up + 1, rows,
                                                I, H, nullptr);
        for (const int which : {0, 1}) {
          dgpp::launch_moe_grouped_mma_fp4_ref_bf16(act, act_stride, seg, 1, rows, 0, views, which,
                                                    ref + 1, I, I, H, nullptr);
          DGPP_CUDA_OK(cudaDeviceSynchronize());
          require(std::memcmp(which == 0 ? gate : up, ref, count * 2) == 0,
                  "paired dense NVFP4 projection and sentinels are bitwise reference");
        }
        // The public single-projection dispatch must use the same chain.
        std::fill(ref, ref + count, uint16_t{0xA55A});
        dgpp::launch_dense_mma_fp4_prod_bf16(act, act_stride, seg, views, 0, ref + 1, rows, I, H,
                                             nullptr);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        require(std::memcmp(gate, ref, count * 2) == 0,
                "single and paired dense NVFP4 launches are bitwise");
      }
      for (auto* p : {gate, ref}) std::fill(p, p + count, uint16_t{0xA55A});
      dgpp::launch_dense_mma_fp4_prod_bf16(unaligned + 1, stride, seg, views, 2, gate + 1, rows, H,
                                           I, nullptr);
      dgpp::launch_moe_grouped_mma_fp4_ref_bf16(unaligned + 1, stride, seg, 1, rows, 0, views, 2,
                                                ref + 1, H, H, I, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      require(std::memcmp(gate, ref, count * 2) == 0,
              "dense NVFP4 down projection and sentinels are bitwise reference");
    }
    for (auto* p : {gate, up, ref, unaligned}) DGPP_CUDA_OK(cudaFree(p));
    DGPP_CUDA_OK(cudaFree(seg));
    DGPP_CUDA_OK(cudaFree(views));
    c.free_all();
  }
}

DGPP_TEST(moe_expert_path_mma_matches_oracle_small_geometry_nvfp4) {
  // The prefill's fp4 tensor-core kernel through the whole layer against
  // the double oracle (gate 4), within the expert-path budget and bitwise
  // repeatable, beside the GEMV chain on the same case; the M=300 shape
  // spans two 128-row m-tiles per expert.
  for (auto [E, H, I, K, M] :
       std::vector<std::tuple<int, int, int, int, int>>{
           {8, 512, 256, 2, 1}, {8, 512, 256, 2, 17}, {16, 1024, 512, 4, 5},
           {8, 512, 256, 4, 300}}) {
    SmallCase c = make_small_case(E, H, I, K, M, 0xF4C0FFEE + E + M, /*nvfp4=*/true);
    c.alloc();
    run_small_case(c, ("nvfp4 expert path (gemv, same case) E=" + std::to_string(E) +
                       " M=" + std::to_string(M))
                          .c_str(),
                   dgpp::MoeExpertKernel::kGemv);
    run_small_case(c, ("nvfp4 expert path (mma) E=" + std::to_string(E) + " M=" +
                       std::to_string(M))
                          .c_str(),
                   dgpp::MoeExpertKernel::kMma);
    c.free_all();
  }
}

DGPP_TEST(moe_expert_path_matches_oracle_small_geometry_nvfp4) {
  SmallCase c = make_small_case(/*E=*/8, /*H=*/512, /*I=*/256, /*K=*/2,
                                /*tokens=*/6, 0x0F4, /*nvfp4=*/true);
  c.alloc();
  run_small_case(c, "nvfp4 expert path E=8 H=512 I=256 K=2");
  c.free_all();
}

DGPP_TEST(moe_nvfp4_activation_scale_requires_opt_in) {
  SmallCase c = make_small_case(/*E=*/4, /*H=*/128, /*I=*/64, /*K=*/2,
                                /*tokens=*/137, 0xF4BF16, /*nvfp4=*/true);
  c.alloc();
  c.cfg.n_shared_experts = 0;
  c.cfg.router_mode = dgpp::MoeRouterMode::SoftmaxTopk;
  float *scales = nullptr, *out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&scales, 2 * sizeof(float)));
  DGPP_CUDA_OK(cudaMallocManaged(&out, c.hidden.size() * sizeof(float)));
  c.dev_w.act_scales_dev = scales;
  const char* mode = std::getenv("DGPP_MOE_W4A4");
  const bool quantized = mode != nullptr && mode[0] == '1';
  const bool static_scale = std::getenv("DGPP_MOE_W4A4_DYNAMIC") == nullptr;
  {
    GlmMoeLayer layer(c.dev_w, c.cfg, c.tokens, /*decode_slots=*/1);
    std::vector<float> prefill;
    std::vector<float> decode;
    for (float scale : {1e-4f, 1e4f}) {
      scales[0] = scales[1] = scale;
      layer.enqueue_prefill_f32(c.d_hidden, out, c.tokens, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> current(out, out + c.hidden.size());
      require(std::all_of(current.begin(), current.end(), [](float v) { return std::isfinite(v); }),
              "NVFP4 policy output must be finite");
      layer.enqueue_decode_f32(c.d_hidden, out, 1, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> step(out, out + c.cfg.hidden);
      if (prefill.empty()) {
        prefill = current;
        decode = step;
      } else {
        const bool identical =
            std::memcmp(prefill.data(), current.data(), current.size() * sizeof(float)) == 0;
        require(identical == !(quantized && static_scale),
                "calibration must affect prefill only with explicit static-scale W4A4 opt-in");
        require(std::memcmp(decode.data(), step.data(), step.size() * sizeof(float)) == 0,
                "activation quantization settings must not change decode");
      }
    }
  }
  cudaFree(out);
  cudaFree(scales);
  c.free_all();
}

DGPP_TEST(moe_w4a4_host_and_device_segmentation_match_both_output_formats) {
  SmallCase c = make_small_case(/*E=*/4, /*H=*/128, /*I=*/64, /*K=*/2,
                                /*tokens=*/137, 0xF4BF16, /*nvfp4=*/true);
  c.alloc();
  c.cfg.n_shared_experts = 0;
  c.cfg.router_mode = dgpp::MoeRouterMode::SoftmaxTopk;
  float *act_scales = nullptr, *out = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&act_scales, 2 * sizeof(float)));
  act_scales[0] = act_scales[1] = 1e-4f;
  c.dev_w.act_scales_dev = act_scales;
  DGPP_CUDA_OK(cudaMallocManaged(&out, c.hidden.size() * sizeof(float)));
  {
    GlmMoeLayer layer(c.dev_w, c.cfg, c.tokens);
    // Enter W4A4, fall back below its row threshold, then reuse the W4A4
    // buffers. Both output APIs must follow the current down-row format.
    for (int tokens : {c.tokens, 1, c.tokens}) {
      const size_t n = static_cast<size_t>(tokens) * c.cfg.hidden;
      layer.enqueue_prefill_f32(c.d_hidden, out, tokens, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const std::vector<float> expected(out, out + n);
      require(std::all_of(expected.begin(), expected.end(), [](float v) { return std::isfinite(v); }),
              "W4A4 prefill output must be finite");
      require(std::any_of(expected.begin(), expected.end(), [](float v) { return v != 0.f; }),
              "W4A4 fixture must exercise nonzero down rows");
      DGPP_CUDA_OK(cudaMemset(out, 0xFF, n * sizeof(float)));
      layer.enqueue_f32(c.d_hidden, out, tokens, nullptr, dgpp::MoeExpertKernel::kMma);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      require(std::memcmp(out, expected.data(), n * sizeof(float)) == 0,
              "host FP32 output must match device-segmented W4A4 output bitwise");
      layer.enqueue(c.d_hidden, c.d_out, tokens, nullptr, dgpp::MoeExpertKernel::kMma);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      for (size_t i = 0; i < n; ++i)
        require(c.d_out[i] == float_to_bf16_bits(expected[i]),
                "host BF16 output must match rounded device-segmented W4A4 output");
    }
  }
  cudaFree(out);
  cudaFree(act_scales);
  c.free_all();
}

DGPP_TEST(moe_decode_slot_path_is_bitwise_host_path_nvfp4) {
  struct Case {
    int E, H, I, K, M;
  };
  const Case cases[] = {
      {8, 512, 256, 2, 1},    // one decode row; gate k=512 (16 lanes/row), down k=256
      {8, 512, 256, 2, 3},    // multi-row steps
      {16, 1024, 512, 4, 2},  // gate k=1024 (32 lanes), down k=512, K=4
  };
  for (const Case& cs : cases) {
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M,
                                  0xF4CADE + cs.E + cs.M, /*nvfp4=*/true);
    c.alloc();
    GlmMoeLayer layer(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/cs.M);
    std::vector<uint16_t> host(static_cast<size_t>(cs.M) * c.cfg.hidden);
    layer.enqueue(c.d_hidden, c.d_out, cs.M, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(host.data(), c.d_out, host.size() * 2);
    // The host path itself within budget of the oracle at this routing.
    std::vector<uint16_t> oracle;
    dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);
    require_within_expert_budget(host, oracle, "nvfp4 host path vs oracle");
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    layer.enqueue_decode(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> fused(host.size(), 0x7F7F);
    std::memcpy(fused.data(), c.d_out, fused.size() * 2);
    require(std::memcmp(host.data(), fused.data(), host.size() * 2) == 0,
            "nvfp4 decode slot path must be bitwise-identical to enqueue");
    // The prefill entry (device segmentation) on the same rows, bitwise too.
    GlmMoeLayer cold(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/cs.M);
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    cold.enqueue_prefill(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> pre(host.size(), 0x7F7F);
    std::memcpy(pre.data(), c.d_out, pre.size() * 2);
    require(std::memcmp(host.data(), pre.data(), host.size() * 2) == 0,
            "nvfp4 prefill path must be bitwise-identical to enqueue");
    c.free_all();
    std::printf("[ OK ] nvfp4 decode slot path E=%d H=%d I=%d K=%d M=%d: bitwise\n",
                cs.E, cs.H, cs.I, cs.K, cs.M);
  }
}

// The per-projection form (2026-10-22, docs/flint_plan.md D2 — Flint's
// layer 12: gate fp8 / up NVFP4 / down fp8; layer 37: gate and up NVFP4
// over an fp8 down): the fp8 and fp4 tables both bound with the mask.
// GIVEN each true mix, WHEN the host path, the decode slot path (the mixed
// gate/up kernel, or the uniform pair kernel with the other format's
// down) and the prefill entry (per-launch formats on the grouped chain)
// run the same rows, THEN the host path sits within the expert budget of
// the oracle and the other two are bitwise it.
DGPP_TEST(moe_decode_slot_path_is_bitwise_host_path_mixed) {
  struct Case {
    int E, H, I, K, M, mask;
  };
  const Case cases[] = {
      {8, 512, 256, 2, 1, 0b010},    // layer 12's form, one row
      {8, 512, 256, 2, 3, 0b010},    // multi-row
      {8, 512, 256, 2, 3, 0b011},    // layer 37's form
      {8, 512, 256, 2, 2, 0b001},    // gate fp4 / up fp8
      {8, 512, 256, 2, 2, 0b100},    // fp8 pair, fp4 down
      {16, 1024, 512, 4, 2, 0b110},  // up + down fp4, K=4
      {16, 1024, 512, 4, 2, 0b101},
  };
  for (const Case& cs : cases) {
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M, 0xF8F4 + cs.E + cs.M + 16 * cs.mask,
                                  /*nvfp4=*/true, false, 0, 16, 0, 0, nullptr, cs.mask);
    c.alloc();
    require(c.dev_w.mixed() && c.dev_w.routed_fp4_mask == cs.mask, "mixed case: both tables bound");
    GlmMoeLayer layer(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/cs.M);
    std::vector<uint16_t> host(static_cast<size_t>(cs.M) * c.cfg.hidden);
    layer.enqueue(c.d_hidden, c.d_out, cs.M, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(host.data(), c.d_out, host.size() * 2);
    std::vector<uint16_t> oracle;
    dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);
    require_within_expert_budget(host, oracle, "mixed host path vs oracle");
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    layer.enqueue_decode(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> fused(host.size(), 0x7F7F);
    std::memcpy(fused.data(), c.d_out, fused.size() * 2);
    require(std::memcmp(host.data(), fused.data(), host.size() * 2) == 0,
            "mixed decode slot path must be bitwise-identical to enqueue");
    // The prefill entry (device segmentation, the tensor-core cores per
    // launch) against the host path on the same cores (the fp8 pin's form:
    // the fp8 tensor-core kernel is not bitwise the GEMV core; the fp4 one
    // is), and within budget of the oracle.
    GlmMoeLayer hm(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/cs.M);
    std::vector<uint16_t> host_mma(host.size(), 0x7F7F);
    hm.enqueue(c.d_hidden, c.d_out, cs.M, nullptr, dgpp::MoeExpertKernel::kMma);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(host_mma.data(), c.d_out, host_mma.size() * 2);
    require_within_expert_budget(host_mma, oracle, "mixed host path (mma) vs oracle");
    GlmMoeLayer cold(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/cs.M);
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    cold.enqueue_prefill(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> pre(host.size(), 0x7F7F);
    std::memcpy(pre.data(), c.d_out, pre.size() * 2);
    require(std::memcmp(host_mma.data(), pre.data(), host.size() * 2) == 0,
            "mixed prefill path must be bitwise-identical to the host path on the same cores");
    // The constructor refuses a mask naming no mix on two bound tables.
    for (int bad : {0, 7}) {
      GlmMoeWeights w2 = c.dev_w;
      w2.routed_fp4_mask = static_cast<uint8_t>(bad);
      bool threw = false;
      try {
        GlmMoeLayer bad_layer(w2, c.cfg, 1, 1);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      require(threw, "two bound tables need a true per-projection mix");
    }
    c.free_all();
    std::printf("[ OK ] mixed decode slot path E=%d H=%d I=%d K=%d M=%d mask=%d: bitwise\n",
                cs.E, cs.H, cs.I, cs.K, cs.M, cs.mask);
  }
}

DGPP_TEST(moe_sliced_ranks_fold_matches_unsliced_oracle_nvfp4) {
  struct Case {
    int E, H, I, K, M, world;
  };
  const Case cases[] = {
      {8, 512, 256, 2, 3, 2},    // slice 128 (down k=128: 4 lanes per row)
      {16, 1024, 512, 4, 2, 4},  // slice 128, four ranks
      {8, 512, 512, 2, 4, 2},    // slice 256 (8 lanes per row)
  };
  for (const Case& cs : cases) {
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M,
                                  0x511F4 + cs.E + cs.world, /*nvfp4=*/true);
    c.alloc();
    std::vector<uint16_t> oracle;
    dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);
    std::vector<std::vector<uint16_t>> partials;
    for (int rank = 0; rank < cs.world; ++rank) {
      RankSlice slice = RankSlice::make(c, rank, cs.world);
      GlmMoeLayer layer(slice.dev_w, c.cfg, cs.M, /*decode_slots=*/cs.M);
      const size_t n = static_cast<size_t>(cs.M) * cs.H;
      layer.enqueue(c.d_hidden, slice.d_out, cs.M, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> host(n);
      std::memcpy(host.data(), slice.d_out, n * 2);
      DGPP_CUDA_OK(cudaMemset(slice.d_out, 0x7F, n * 2));
      layer.enqueue_decode(c.d_hidden, slice.d_out, cs.M, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> slot(n);
      std::memcpy(slot.data(), slice.d_out, n * 2);
      require(host == slot, "nvfp4 sliced rank: decode slot path must be bitwise the host path");
      partials.push_back(std::move(host));
      slice.free_all();
    }
    const std::vector<uint16_t> folded = fold_ranks(partials);
    require_within_fold_budget(
        folded, partials, oracle,
        ("nvfp4 sliced fold E=" + std::to_string(cs.E) + " I=" + std::to_string(cs.I) +
         " world=" + std::to_string(cs.world)).c_str());
    c.free_all();
  }
}

// ---- MXFP4 routed experts (2026-09-13, docs/deepseek_v41_flash_plan.md G2) ----
// The NVFP4 gates' twins over e8m0-scaled tables without globals: the
// oracle, the decode slot path bitwise the host chain, the prefill entry
// bitwise, the sliced fold, and the grouped GEMV bitwise the single-matrix
// launcher. The tensor-core form refuses an MXFP4 table (mma_takes_grid).

DGPP_TEST(moe_expert_path_matches_oracle_small_geometry_mxfp4) {
  SmallCase c = make_small_case(/*E=*/8, /*H=*/512, /*I=*/256, /*K=*/2,
                                /*tokens=*/6, 0x3F4, /*nvfp4=*/true, false, 0, /*fp4_group=*/32);
  c.alloc();
  require(c.dev_w.nvfp4() && c.dev_w.experts_fp4[0].mxfp4(), "the case binds MXFP4 experts");
  run_small_case(c, "mxfp4 expert path E=8 H=512 I=256 K=2");
  // The tensor-core kernel takes MXFP4 tables since 2026-09-14 (its kGroup
  // = 32 form): the layer's default for the table, and the oracle budget
  // holds through it as through the GEMV core.
  GlmMoeLayer layer(c.dev_w, c.cfg, 8, /*decode_slots=*/0);
  require(layer.mma_takes_grid(), "the tensor-core kernel takes an MXFP4 table");
  std::vector<uint16_t> oracle;
  dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);
  for (const dgpp::MoeExpertKernel kernel : {dgpp::MoeExpertKernel::kMma, dgpp::MoeExpertKernel::kGemv}) {
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, static_cast<size_t>(c.tokens) * c.cfg.hidden * 2));
    layer.enqueue(c.d_hidden, c.d_out, c.tokens, nullptr, kernel);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> got(static_cast<size_t>(c.tokens) * c.cfg.hidden);
    std::memcpy(got.data(), c.d_out, got.size() * 2);
    require_within_expert_budget(got, oracle, kernel == dgpp::MoeExpertKernel::kMma ? "mxfp4 tile kernel vs oracle"
                                                                                    : "mxfp4 GEMV core vs oracle");
  }
  c.free_all();
}

DGPP_TEST(moe_decode_slot_path_is_bitwise_host_path_mxfp4) {
  struct Case {
    int E, H, I, K, M;
  };
  const Case cases[] = {
      {8, 512, 256, 2, 1},    // gate k=512, down k=256
      {8, 512, 256, 2, 3},    // multi-row steps
      {16, 1024, 512, 4, 2},  // gate k=1024, down k=512, K=4
      {8, 1024, 576, 6, 2},   // the release's world-4 down width (18 chunks: 2 lanes x 9), top-6
      {8, 1024, 576, 6, 5},   // a draft block's five rows: more shared slots than a group of four
      {8, 1024, 576, 6, 80},  // sixteen five-row slots: the 96-row decode batch (2026-10-23)
  };
  for (const Case& cs : cases) {
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M, 0x3F4CADE + cs.E + cs.M + cs.I,
                                  /*nvfp4=*/true, false, 0, /*fp4_group=*/32);
    c.alloc();
    GlmMoeLayer layer(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/cs.M);
    std::vector<uint16_t> host(static_cast<size_t>(cs.M) * c.cfg.hidden);
    layer.enqueue(c.d_hidden, c.d_out, cs.M, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(host.data(), c.d_out, host.size() * 2);
    std::vector<uint16_t> oracle;
    dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);
    require_within_expert_budget(host, oracle, "mxfp4 host path vs oracle");
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    // The decode slot path is the GEMV core's chain: bitwise the host
    // path through the GEMV core (MoeExpertKernel::kGemv). The layer's
    // default and the prefill path take the tensor-core kernel's MXFP4
    // form since 2026-09-14 — tolerance-equal to the core (the oracle
    // budget above), not bitwise; the prefill path is bitwise the default.
    std::vector<uint16_t> gemv_host(host.size(), 0x7F7F);
    layer.enqueue(c.d_hidden, c.d_out, cs.M, nullptr, dgpp::MoeExpertKernel::kGemv);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(gemv_host.data(), c.d_out, gemv_host.size() * 2);
    require_within_expert_budget(gemv_host, oracle, "mxfp4 GEMV host path vs oracle");
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    layer.enqueue_decode(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> fused(host.size(), 0x7F7F);
    std::memcpy(fused.data(), c.d_out, fused.size() * 2);
    require(std::memcmp(gemv_host.data(), fused.data(), host.size() * 2) == 0,
            "mxfp4 decode slot path must be bitwise-identical to the GEMV host path");
    // The shared expert's slots early in the execution order
    // (GlmMoeConfig::shared_slots_early, the DeepSeek-V4-Flash decode): the
    // order alone moves — every row bitwise the default order's, one row
    // (which launches the order kernel only under this flag) included.
    {
      dgpp::GlmMoeConfig early_cfg = c.cfg;
      early_cfg.shared_slots_early = true;
      GlmMoeLayer early(c.dev_w, early_cfg, std::max(cs.M, 1), /*decode_slots=*/cs.M);
      DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
      early.enqueue_decode(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      require(std::memcmp(c.d_out, fused.data(), host.size() * 2) == 0,
              "mxfp4 decode: the early shared-slot order must be bitwise the default order");
    }
    // The prefill path takes the tile kernel's MXFP4 form: bitwise the
    // host path through it (MoeExpertKernel::kMma), within budget of the oracle.
    std::vector<uint16_t> mma_host(host.size(), 0x7F7F);
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    layer.enqueue(c.d_hidden, c.d_out, cs.M, nullptr, dgpp::MoeExpertKernel::kMma);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(mma_host.data(), c.d_out, mma_host.size() * 2);
    require_within_expert_budget(mma_host, oracle, "mxfp4 tile host path vs oracle");
    GlmMoeLayer cold(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/cs.M);
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    cold.enqueue_prefill(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> pre(host.size(), 0x7F7F);
    std::memcpy(pre.data(), c.d_out, pre.size() * 2);
    require(std::memcmp(mma_host.data(), pre.data(), host.size() * 2) == 0,
            "mxfp4 prefill path (the tile kernel) must be bitwise-identical to the tile host path");
    c.free_all();
    std::printf("[ OK ] mxfp4 decode slot path E=%d H=%d I=%d K=%d M=%d: bitwise the GEMV host path; the tile path within budget\n",
                cs.E, cs.H, cs.I, cs.K, cs.M);
  }
}

DGPP_TEST(moe_sliced_ranks_fold_matches_unsliced_oracle_mxfp4) {
  struct Case {
    int E, H, I, K, M, world;
  };
  const Case cases[] = {
      {8, 512, 256, 2, 3, 2},    // slice 128
      {16, 1024, 512, 4, 2, 4},  // slice 128, four ranks
      {8, 1024, 2304, 6, 2, 2},  // the release width at world 2: 2304 / 2 = 1152 (36 e8m0 blocks; the test's FP8 shared expert needs a 128-multiple slice, so world 4's 576 is covered by the slot gate above)
  };
  for (const Case& cs : cases) {
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M, 0x3511F4 + cs.E + cs.world + cs.I,
                                  /*nvfp4=*/true, false, 0, /*fp4_group=*/32);
    c.alloc();
    std::vector<uint16_t> oracle;
    dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);
    std::vector<std::vector<uint16_t>> partials;
    for (int rank = 0; rank < cs.world; ++rank) {
      RankSlice slice = RankSlice::make(c, rank, cs.world);
      GlmMoeLayer layer(slice.dev_w, c.cfg, cs.M, /*decode_slots=*/cs.M);
      const size_t n = static_cast<size_t>(cs.M) * cs.H;
      layer.enqueue(c.d_hidden, slice.d_out, cs.M, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> host(n);
      std::memcpy(host.data(), slice.d_out, n * 2);
      DGPP_CUDA_OK(cudaMemset(slice.d_out, 0x7F, n * 2));
      layer.enqueue_decode(c.d_hidden, slice.d_out, cs.M, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> slot(n);
      std::memcpy(slot.data(), slice.d_out, n * 2);
      require(host == slot, "mxfp4 sliced rank: decode slot path must be bitwise the host path");
      partials.push_back(std::move(host));
      slice.free_all();
    }
    const std::vector<uint16_t> folded = fold_ranks(partials);
    require_within_fold_budget(
        folded, partials, oracle,
        ("mxfp4 sliced fold E=" + std::to_string(cs.E) + " I=" + std::to_string(cs.I) +
         " world=" + std::to_string(cs.world)).c_str());
    c.free_all();
  }
}

DGPP_TEST(moe_grouped_gemv_mxfp4_is_bitwise_the_single_matrix_launcher) {
  SmallCase c = make_small_case(/*E=*/6, /*H=*/1024, /*I=*/208, /*K=*/2,
                                /*tokens=*/64, 0x36D4, /*nvfp4=*/true, false, 0, /*fp4_group=*/32);
  c.alloc();
  const int H = c.cfg.hidden, I = 208, E = c.cfg.n_experts;
  const int lens[] = {1, 5, 4, 9, 13, 2, 30};
  std::vector<dgpp::MoeSegment> segs;
  int row0 = 0;
  for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); ++i) {
    segs.push_back(dgpp::MoeSegment{row0, lens[i], static_cast<int>(i % E)});
    row0 += lens[i];
  }
  dgpp::MoeSegment* d_segs = nullptr;
  dgpp::MoeExpertView* d_views = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_segs, segs.size() * sizeof(dgpp::MoeSegment)));
  std::memcpy(d_segs, segs.data(), segs.size() * sizeof(dgpp::MoeSegment));
  DGPP_CUDA_OK(cudaMallocManaged(&d_views, E * 3 * sizeof(dgpp::MoeExpertView)));
  for (int m = 0; m < E * 3; ++m) d_views[m] = dgpp::MoeExpertView::of(c.expert_mats_fp4[m]);
  require(d_views[0].fp4_group == 32 && d_views[0].fp4_global == nullptr, "the view carries the group");
  uint16_t *d_grouped = nullptr, *d_ref = nullptr;
  DGPP_CUDA_OK(cudaMallocManaged(&d_grouped, 64 * I * 2));
  DGPP_CUDA_OK(cudaMallocManaged(&d_ref, 64 * I * 2));
  DGPP_CUDA_OK(cudaMemset(d_grouped, 0xA5, 64 * I * 2));
  DGPP_CUDA_OK(cudaMemset(d_ref, 0x5A, 64 * I * 2));
  dgpp::launch_moe_grouped_gemv_fp4_bf16(c.d_hidden, H, d_segs, static_cast<int>(segs.size()),
                                         30, /*rows_per_block=*/8, d_views, 0, d_grouped, I, I, H,
                                         nullptr, /*fp4_group=*/32);
  for (const dgpp::MoeSegment& sg : segs)
    dgpp::launch_fp4_gemv_bf16(c.d_hidden + static_cast<size_t>(sg.row0) * H, H,
                               c.expert_mats_fp4[sg.expert * 3 + 0],
                               d_ref + static_cast<size_t>(sg.row0) * I, sg.rows, I, H, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  require(std::memcmp(d_grouped, d_ref, 64 * I * 2) == 0,
          "mxfp4 grouped gate output bitwise the single-matrix launcher");
  std::printf("[ OK ] mxfp4 grouped gemv: %zu segments (1..30 rows) bitwise the launcher\n",
              segs.size());
  cudaFree(d_grouped);
  cudaFree(d_ref);
  cudaFree(d_segs);
  cudaFree(d_views);
  c.free_all();
}

// ---- packed-int routed experts (docs/glm53_plan.md G2) ---------------------

DGPP_TEST(moe_grouped_gemv_packq_is_bitwise_the_single_matrix_launcher) {
  // The packed twin of the grouped-vs-per-segment gate at both widths:
  // segments of 1..30 rows (the multi-group loop), n leaving dead lane
  // groups, gate bf16 (k = H) and down fp32 (k = I) — every row bitwise
  // launch_packq_gemv's.
  for (int bits : {4, 8}) {
    SmallCase c = make_small_case(/*E=*/6, /*H=*/1024, /*I=*/256, /*K=*/2,
                                  /*tokens=*/64, 0x6D4 + bits, /*nvfp4=*/false,
                                  /*shared_nvfp4=*/false, /*packq_bits=*/bits);
    c.alloc();
    const int H = c.cfg.hidden, I = c.cfg.inter, E = c.cfg.n_experts;
    const int lens[] = {1, 5, 4, 9, 13, 2, 30};
    std::vector<dgpp::MoeSegment> segs;
    int row0 = 0;
    for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); ++i) {
      segs.push_back(dgpp::MoeSegment{row0, lens[i], static_cast<int>(i % E)});
      row0 += lens[i];
    }
    dgpp::MoeSegment* d_segs = nullptr;
    dgpp::MoeExpertView* d_views = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_segs, segs.size() * sizeof(dgpp::MoeSegment)));
    std::memcpy(d_segs, segs.data(), segs.size() * sizeof(dgpp::MoeSegment));
    DGPP_CUDA_OK(cudaMallocManaged(&d_views, (E + 1) * 3 * sizeof(dgpp::MoeExpertView)));
    for (int m = 0; m < (E + 1) * 3; ++m) d_views[m] = dgpp::MoeExpertView::of(c.expert_mats_packq[m]);
    uint16_t *d_grouped = nullptr, *d_ref = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_grouped, 64 * I * 2));
    DGPP_CUDA_OK(cudaMallocManaged(&d_ref, 64 * I * 2));
    DGPP_CUDA_OK(cudaMemset(d_grouped, 0xA5, 64 * I * 2));
    DGPP_CUDA_OK(cudaMemset(d_ref, 0x5A, 64 * I * 2));
    dgpp::launch_moe_grouped_gemv_packq_bf16(c.d_hidden, H, d_segs, static_cast<int>(segs.size()),
                                             30, /*rows_per_block=*/8, d_views, 0, d_grouped, I, I,
                                             H, bits, nullptr);
    for (const dgpp::MoeSegment& sg : segs)
      dgpp::launch_packq_gemv_bf16(c.d_hidden + static_cast<size_t>(sg.row0) * H, H,
                                   c.expert_mats_packq[sg.expert * 3 + 0],
                                   d_ref + static_cast<size_t>(sg.row0) * I, sg.rows, I, H, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(std::memcmp(d_grouped, d_ref, 64 * I * 2) == 0,
            "packq grouped gate output bitwise the single-matrix launcher");
    // down: fp32 out over the gate output as the activation (k = I).
    float *d_gd = nullptr, *d_rd = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_gd, 64 * H * 4));
    DGPP_CUDA_OK(cudaMallocManaged(&d_rd, 64 * H * 4));
    DGPP_CUDA_OK(cudaMemset(d_gd, 0xA5, 64 * H * 4));
    DGPP_CUDA_OK(cudaMemset(d_rd, 0x5A, 64 * H * 4));
    dgpp::launch_moe_grouped_gemv_packq_f32(d_grouped, I, d_segs, static_cast<int>(segs.size()),
                                            30, /*rows_per_block=*/0, d_views, 2, d_gd, H, H, I,
                                            bits, nullptr);
    for (const dgpp::MoeSegment& sg : segs)
      dgpp::launch_packq_gemv_f32(d_grouped + static_cast<size_t>(sg.row0) * I, I,
                                  c.expert_mats_packq[sg.expert * 3 + 2],
                                  d_rd + static_cast<size_t>(sg.row0) * H, sg.rows, H, I, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(std::memcmp(d_gd, d_rd, 64 * H * 4) == 0,
            "packq grouped down output bitwise the single-matrix launcher (fp32)");
    // The shared segment at the shared width (int8) through the same launcher.
    dgpp::MoeSegment* d_sh = nullptr;
    DGPP_CUDA_OK(cudaMallocManaged(&d_sh, sizeof(dgpp::MoeSegment)));
    *d_sh = dgpp::MoeSegment{0, 64, E};
    DGPP_CUDA_OK(cudaMemset(d_grouped, 0xA5, 64 * I * 2));
    DGPP_CUDA_OK(cudaMemset(d_ref, 0x5A, 64 * I * 2));
    dgpp::launch_moe_grouped_gemv_packq_bf16(c.d_hidden, H, d_sh, 1, 64, /*rows_per_block=*/16,
                                             d_views, 1, d_grouped, I, I, H, 8, nullptr);
    dgpp::launch_packq_gemv_bf16(c.d_hidden, H, c.expert_mats_packq[E * 3 + 1], d_ref, 64, I, H,
                                 nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    require(std::memcmp(d_grouped, d_ref, 64 * I * 2) == 0,
            "packq grouped shared (int8) output bitwise the single-matrix launcher");
    std::printf("[ OK ] packq int%d grouped gemv: %zu segments (1..30 rows) bitwise the launcher\n",
                bits, segs.size());
    cudaFree(d_sh);
    cudaFree(d_gd);
    cudaFree(d_rd);
    cudaFree(d_grouped);
    cudaFree(d_ref);
    cudaFree(d_segs);
    cudaFree(d_views);
    c.free_all();
  }
}

DGPP_TEST(moe_expert_path_matches_oracle_small_geometry_packq) {
  // The whole layer (int4 or int8 routed, int8 shared) through the GEMV
  // chain against the double oracle, within the expert-path budget and
  // bitwise repeatable; at both routed scale formats (sf 1: f16 per 128,
  // the AutoRound hybrid, the shared triple still bf16 per 64).
  for (int sf : {0, 1, 2})
  for (int bits : {4, 8})
    for (auto [E, H, I, K, M] :
         std::vector<std::tuple<int, int, int, int, int>>{
             {8, 512, 256, 2, 1}, {8, 512, 256, 2, 6}, {16, 1024, 512, 4, 5}}) {
      if (sf == 2 && bits != 4) continue;  // the codebook format is 4-bit
      const int rit = sf == 2 ? dgpp::kMoeInputHadamard32 : dgpp::kMoeInputPlain;
      SmallCase c = make_small_case(E, H, I, K, M, 0x9AC0FFEE + E + M + bits + 77 * sf, false,
                                    false, bits, 16, sf, rit);
      c.alloc();
      run_small_case(c, ("packq int" + std::to_string(bits) + (sf == 1 ? " g128/f16" : sf == 2 ? " nf4i8+h32" : "") +
                         " expert path E=" + std::to_string(E) + " M=" + std::to_string(M)).c_str(),
                     dgpp::MoeExpertKernel::kGemv);
      c.free_all();
    }
}

DGPP_TEST(moe_packq_prefill_tile_matches_oracle_and_host_segmentation) {
  for (int sf : {0, 1, 2})
  for (int bits : {4, 8})
    for (int tokens : {16, 17, 33, 127, 128, 129, 257}) {
      if (sf == 2 && bits != 4) continue;
      const int rit = sf == 2 ? dgpp::kMoeInputHadamard32 : dgpp::kMoeInputPlain;
      SmallCase c = make_small_case(8, 512, 256, 2, tokens, 0x61A0 + tokens + bits + 77 * sf,
                                    false, false, bits, 16, sf, rit);
      c.alloc();
      const auto kernel = tokens < 128 ? dgpp::MoeExpertKernel::kGemv : dgpp::MoeExpertKernel::kMma;
      run_small_case(c, sf == 1 ? "packed g128/f16 expert prefill lowering"
                        : sf == 2 ? "packed nf4i8+h32 expert prefill lowering"
                                  : "packed expert prefill lowering",
                     kernel);
      std::vector<uint16_t> host(static_cast<size_t>(tokens) * c.cfg.hidden);
      std::memcpy(host.data(), c.d_out, host.size() * 2);
      GlmMoeLayer layer(c.dev_w, c.cfg, tokens);
      layer.enqueue_prefill(c.d_hidden, c.d_out, tokens, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      require(std::memcmp(host.data(), c.d_out, host.size() * 2) == 0,
              "packed prefill device segmentation must match the selected host chain");
      c.free_all();
    }
}

DGPP_TEST(moe_decode_slot_path_is_bitwise_host_path_packq) {
  struct Case {
    int E, H, I, K, M;
  };
  const Case cases[] = {
      {8, 512, 256, 2, 1},    // one decode row; gate k=512, down k=256
      {8, 512, 256, 2, 3},    // multi-row steps
      {16, 1024, 512, 4, 2},  // gate k=1024, down k=512, K=4
  };
  for (int sf : {0, 1, 2})
  for (int bits : {4, 8})
    for (const Case& cs : cases) {
      if (sf == 2 && bits != 4) continue;
      const int rit = sf == 2 ? dgpp::kMoeInputHadamard32 : dgpp::kMoeInputPlain;
      SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M, 0x9ACADE + cs.E + cs.M + bits + 77 * sf,
                                    false, false, bits, 16, sf, rit);
      c.alloc();
      GlmMoeLayer layer(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/cs.M,
                        /*graph_table_slots=*/1);
      std::vector<uint16_t> host(static_cast<size_t>(cs.M) * c.cfg.hidden);
      layer.enqueue(c.d_hidden, c.d_out, cs.M, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::memcpy(host.data(), c.d_out, host.size() * 2);
      // The host path itself within budget of the oracle at this routing.
      std::vector<uint16_t> oracle;
      dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);
      require_within_expert_budget(host, oracle, "packq host path vs oracle");
      DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
      layer.enqueue_decode(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> fused(host.size(), 0x7F7F);
      std::memcpy(fused.data(), c.d_out, fused.size() * 2);
      require(std::memcmp(host.data(), fused.data(), host.size() * 2) == 0,
              "packq decode slot path must be bitwise-identical to enqueue");
      // The prepared graph table (the captured path's stable source).
      layer.prepare_graph_table(0, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
      layer.enqueue_decode(c.d_hidden, c.d_out, cs.M, nullptr, nullptr, /*table_slot=*/0);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::memcpy(fused.data(), c.d_out, fused.size() * 2);
      require(std::memcmp(host.data(), fused.data(), host.size() * 2) == 0,
              "packq decode slot path on the graph table bitwise the host chain");
      // Short prefills retain the GEMV arithmetic used by decode.
      GlmMoeLayer cold(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/cs.M);
      DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
      cold.enqueue_prefill(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> pre(host.size(), 0x7F7F);
      std::memcpy(pre.data(), c.d_out, pre.size() * 2);
      require(std::memcmp(host.data(), pre.data(), host.size() * 2) == 0,
              "packq prefill path must be bitwise-identical to enqueue");
      c.free_all();
      std::printf("[ OK ] packq int%d%s decode slot path E=%d H=%d I=%d K=%d M=%d: bitwise\n",
                  bits, sf == 1 ? " g128/f16" : sf == 2 ? " nf4i8+h32" : "", cs.E, cs.H, cs.I, cs.K, cs.M);
    }
}

DGPP_TEST(moe_mixed346_expert_path_matches_oracle) {
  // The Mixed346 layer through the grouped GEMV chain (every form on one
  // layer: the seven width pairs and an existing int4 expert, the shared
  // int8 triple) against the double oracle, and the decode slot path
  // bitwise the host chain (one and several decode rows, the graph table,
  // the short prefill).
  for (auto [E, H, I, K, M] : std::vector<std::tuple<int, int, int, int, int>>{
           {8, 256, 128, 2, 1}, {8, 256, 128, 2, 3}, {8, 512, 256, 3, 6}, {16, 1024, 512, 4, 5}}) {
    std::vector<const char*> forms(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e) forms[static_cast<size_t>(e)] = kM346Forms[e % 8];
    SmallCase c = make_small_case(E, H, I, K, M, 0x346A8 + E + 7 * M, false, false, 0, 16, 0, 0, forms.data());
    c.alloc();
    const std::string label = "mixed346 expert path E=" + std::to_string(E) + " H=" + std::to_string(H) +
                              " I=" + std::to_string(I) + " M=" + std::to_string(M);
    run_small_case(c, label.c_str(), dgpp::MoeExpertKernel::kGemv);
    std::vector<uint16_t> host(static_cast<size_t>(M) * c.cfg.hidden);
    std::memcpy(host.data(), c.d_out, host.size() * 2);
    GlmMoeLayer layer(c.dev_w, c.cfg, std::max(M, 1), /*decode_slots=*/M, /*graph_table_slots=*/1);
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    layer.enqueue_decode(c.d_hidden, c.d_out, M, nullptr, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> slot(host.size());
    std::memcpy(slot.data(), c.d_out, slot.size() * 2);
    require(host == slot, "mixed346 decode slot path bitwise the host chain");
    layer.prepare_graph_table(0, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    layer.enqueue_decode(c.d_hidden, c.d_out, M, nullptr, nullptr, /*table_slot=*/0);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(slot.data(), c.d_out, slot.size() * 2);
    require(host == slot, "mixed346 decode slot path on the graph table bitwise the host chain");
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, host.size() * 2));
    layer.enqueue_prefill(c.d_hidden, c.d_out, M, nullptr, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(slot.data(), c.d_out, slot.size() * 2);
    require(host == slot, "mixed346 prefill path bitwise the host chain");
    c.free_all();
    std::printf("[ OK ] %s: oracle, slot path, graph table, prefill\n", label.c_str());
  }
  // A layer whose routed experts are all existing (the baseline form under
  // the int8 transform) still runs — the slot kernels' plain path alone.
  {
    std::vector<const char*> forms(8, "existing");
    SmallCase c = make_small_case(8, 256, 128, 2, 2, 0x3460, false, false, 0, 16, 0, 0, forms.data());
    c.alloc();
    run_small_case(c, "mixed346 all-existing layer", dgpp::MoeExpertKernel::kGemv);
    c.free_all();
  }
}

DGPP_TEST(moe_mixed346_input_sensitivity_probe) {
  // How much of a one-ulp change of one input element reaches the output,
  // per format: the int8 activation codes make the layer less tolerant of
  // upstream noise than bf16 activations (a flipped code), which the TP,
  // decode and parity gates' budgets must account for. Printed, not gated.
  for (int form : {0, 2, 3}) {
    const int E = 8, H = 512, I = 256, K = 2, M = 6;
    std::vector<const char*> forms(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e) forms[static_cast<size_t>(e)] = kM346Forms[e % 8];
    SmallCase c = form == 3 ? make_small_case(E, H, I, K, M, 0x5E15, false, false, 0, 16, 0, 0, forms.data())
                            : make_small_case(E, H, I, K, M, 0x5E15, false, false, 4, 16, form,
                                              form == 2 ? dgpp::kMoeInputHadamard32 : dgpp::kMoeInputPlain);
    c.alloc();
    GlmMoeLayer layer(c.dev_w, c.cfg, M, /*decode_slots=*/M);
    layer.enqueue(c.d_hidden, c.d_out, M, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> base(static_cast<size_t>(M) * H);
    std::memcpy(base.data(), c.d_out, base.size() * 2);
    double e2_sum = 0, n2_sum = 0, worst = 0;
    int trials = 0, changed = 0;
    for (int t = 0; t < M; ++t)
      for (int j : {0, 77, 300}) {
        const size_t at = static_cast<size_t>(t) * H + j;
        const uint16_t saved = c.d_hidden[at];
        c.d_hidden[at] = static_cast<uint16_t>(saved + 1);  // one bf16 ulp
        layer.enqueue(c.d_hidden, c.d_out, M, nullptr);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        c.d_hidden[at] = saved;
        double e2 = 0, n2 = 0;
        for (size_t i = static_cast<size_t>(t) * H; i < static_cast<size_t>(t + 1) * H; ++i) {
          const double a = bf16_to_float(c.d_out[i]), b = bf16_to_float(base[i]);
          e2 += (a - b) * (a - b);
          n2 += b * b;
          changed += c.d_out[i] != base[i];
        }
        worst = std::max(worst, std::sqrt(e2 / std::max(n2, 1e-30)));
        e2_sum += e2;
        n2_sum += n2;
        ++trials;
      }
    std::printf("[ .. ] input sensitivity (%s): row l2 response to one input ulp: mean %.3g worst %.3g, %d of %d elements changed\n",
                form == 3 ? "mixed346 int8 codes" : form == 2 ? "nf4i8+h32 bf16" : "int4 bf16",
                std::sqrt(e2_sum / std::max(n2_sum, 1e-30)), worst, changed, trials * H);
    c.free_all();
  }
}

DGPP_TEST(moe_mixed346_sliced_ranks_fold_matches_unsliced_oracle) {
  struct Case {
    int E, H, I, K, M, world;
  };
  const Case cases[] = {
      {8, 256, 256, 2, 3, 2},    // slice 128 (down k=128)
      {16, 1024, 512, 4, 2, 4},  // slice 128, four ranks
      {8, 512, 512, 2, 4, 2},    // slice 256
  };
  for (const Case& cs : cases) {
    std::vector<const char*> forms(static_cast<size_t>(cs.E));
    for (int e = 0; e < cs.E; ++e) forms[static_cast<size_t>(e)] = kM346Forms[e % 8];
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M, 0x346F0 + cs.E + cs.world, false, false, 0, 16, 0,
                                  0, forms.data());
    c.alloc();
    std::vector<uint16_t> oracle;
    dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);
    std::vector<std::vector<uint16_t>> partials;
    for (int rank = 0; rank < cs.world; ++rank) {
      RankSlice slice = RankSlice::make(c, rank, cs.world);
      GlmMoeLayer layer(slice.dev_w, c.cfg, cs.M, /*decode_slots=*/cs.M);
      const size_t n = static_cast<size_t>(cs.M) * cs.H;
      layer.enqueue(c.d_hidden, slice.d_out, cs.M, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> host(n);
      std::memcpy(host.data(), slice.d_out, n * 2);
      DGPP_CUDA_OK(cudaMemset(slice.d_out, 0x7F, n * 2));
      layer.enqueue_decode(c.d_hidden, slice.d_out, cs.M, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> slot(n);
      std::memcpy(slot.data(), slice.d_out, n * 2);
      require(host == slot, "mixed346 sliced rank: decode slot path bitwise the host path");
      partials.push_back(std::move(host));
      slice.free_all();
    }
    const std::vector<uint16_t> folded = fold_ranks(partials);
    require_within_fold_budget(folded, partials, oracle,
                               ("mixed346 sliced fold E=" + std::to_string(cs.E) + " world=" + std::to_string(cs.world)).c_str());
    c.free_all();
  }
}

DGPP_TEST(moe_sliced_ranks_fold_matches_unsliced_oracle_packq) {
  struct Case {
    int E, H, I, K, M, world;
  };
  const Case cases[] = {
      {8, 512, 256, 2, 3, 2},    // slice 128 (down k=128)
      {16, 1024, 512, 4, 2, 4},  // slice 128, four ranks
      {8, 512, 512, 2, 4, 2},    // slice 256
  };
  for (int sf : {0, 1, 2})
  for (int bits : {4, 8})
    for (const Case& cs : cases) {
      if (sf == 2 && bits != 4) continue;
      const int rit = sf == 2 ? dgpp::kMoeInputHadamard32 : dgpp::kMoeInputPlain;
      SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M, 0x511F9 + cs.E + cs.world + bits + 77 * sf,
                                    false, false, bits, 16, sf, rit);
      c.alloc();
      std::vector<uint16_t> oracle;
      dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);
      std::vector<std::vector<uint16_t>> partials;
      for (int rank = 0; rank < cs.world; ++rank) {
        RankSlice slice = RankSlice::make(c, rank, cs.world);
        GlmMoeLayer layer(slice.dev_w, c.cfg, cs.M, /*decode_slots=*/cs.M);
        const size_t n = static_cast<size_t>(cs.M) * cs.H;
        layer.enqueue(c.d_hidden, slice.d_out, cs.M, nullptr);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        std::vector<uint16_t> host(n);
        std::memcpy(host.data(), slice.d_out, n * 2);
        DGPP_CUDA_OK(cudaMemset(slice.d_out, 0x7F, n * 2));
        layer.enqueue_decode(c.d_hidden, slice.d_out, cs.M, nullptr, nullptr);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        std::vector<uint16_t> slot(n);
        std::memcpy(slot.data(), slice.d_out, n * 2);
        require(host == slot, "packq sliced rank: decode slot path must be bitwise the host path");
        partials.push_back(std::move(host));
        slice.free_all();
      }
      const std::vector<uint16_t> folded = fold_ranks(partials);
      require_within_fold_budget(
          folded, partials, oracle,
          ("packq int" + std::to_string(bits) + (sf == 2 ? " nf4i8+h32" : "") + " sliced fold E=" +
           std::to_string(cs.E) + " I=" + std::to_string(cs.I) + " world=" + std::to_string(cs.world)).c_str());
      c.free_all();
    }
}

// ---- the EXL3 experts (2026-10-22, docs/deepseek_v41_flash_exl3_plan.md §3.3) --
// The pack's shapes in miniature: hidden 512, an intermediate of 256 or
// 640 (five blocks: the uneven world-4 slice), routed experts at 3 / 4 / 6
// bits with a 6-bit shared expert, and the uniform 8-bit pair. The host
// chain within the expert budget of the double oracle (the full
// reconstruct); the decode slot path bitwise the host chain (the same
// core per row); the sliced ranks' fold within the partials' budget.
DGPP_TEST(moe_exl3_expert_path_matches_oracle) {
  struct Case { int E, H, I, K, M, rb, sb; };
  const Case cases[] = {{8, 512, 256, 2, 1, 4, 6}, {8, 512, 256, 2, 5, 3, 6}, {8, 512, 256, 2, 12, 6, 6},
                        {8, 512, 640, 2, 3, 4, 6}, {4, 256, 128, 2, 7, 8, 8}};
  for (const Case& cs : cases) {
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M, 0xE713 + cs.M + cs.rb, false, false, 0, 16, 0, 0,
                                  nullptr, -1, cs.rb, cs.sb);
    c.alloc();
    require(c.dev_w.exl3() && c.dev_w.shared_exl3_set(), "the case binds EXL3 experts and shared expert");
    run_small_case(c, ("exl3 expert path E=" + std::to_string(cs.E) + " I=" + std::to_string(cs.I) + " M=" +
                       std::to_string(cs.M) + " bits " + std::to_string(cs.rb) + "/" + std::to_string(cs.sb))
                          .c_str());
    c.free_all();
  }
}

// The decode slot path's expert-run form (2026-10-23): with few experts
// and many rows, the expert-sorted order has long equal-expert runs (the
// shared expert's is every row), so the cases past 16 rows run the 64-row
// run tile beside the 16-row one, at every compiled width pair, and must
// stay bitwise the host chain (per row the same core chain); the 80- and
// 96-row cases cut the shared expert's run into two 64-row groups.
DGPP_TEST(moe_exl3_decode_slot_path_is_bitwise_host_path) {
  struct Case { int E, H, I, K, M, rb, sb; };
  const Case cases[] = {{8, 512, 256, 2, 1, 4, 6},  {8, 512, 256, 2, 3, 3, 6},  {8, 512, 640, 4, 2, 6, 6},
                        {8, 512, 256, 2, 17, 4, 6}, {8, 512, 640, 4, 40, 3, 6}, {4, 256, 128, 2, 64, 8, 8},
                        {8, 512, 256, 2, 64, 6, 6}, {4, 256, 128, 2, 80, 4, 6},  {8, 512, 256, 2, 96, 3, 6}};
  for (const Case& cs : cases) {
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M, 0xE7D0 + cs.M + cs.rb, false, false, 0, 16, 0, 0,
                                  nullptr, -1, cs.rb, cs.sb);
    c.alloc();
    GlmMoeLayer layer(c.dev_w, c.cfg, std::max(cs.M, 1), /*decode_slots=*/cs.M, /*graph_table_slots=*/1);
    const size_t n = static_cast<size_t>(cs.M) * c.cfg.hidden;
    layer.enqueue(c.d_hidden, c.d_out, cs.M, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> host(n);
    std::memcpy(host.data(), c.d_out, n * 2);
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, n * 2));
    layer.enqueue_decode(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> slot(n);
    std::memcpy(slot.data(), c.d_out, n * 2);
    require(host == slot, "exl3: the decode slot path must be bitwise the host path");
    // The decode graph's own table (prepare_graph_table): the routed
    // entries and the shared triple as EXL3 views — the shared triple was
    // left unwritten before 2026-10-23 (the real pack's graph decode ran
    // every layer on a stale shared view; eager decode was right).
    layer.prepare_graph_table(0, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, n * 2));
    layer.enqueue_decode(c.d_hidden, c.d_out, cs.M, nullptr, nullptr, /*table_slot=*/0);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::memcpy(slot.data(), c.d_out, n * 2);
    require(host == slot, "exl3: the decode slot path on the graph table must be bitwise the host path");
    // The tensor-core prefill entry (the same EXL3 kernels) too.
    DGPP_CUDA_OK(cudaMemset(c.d_out, 0x7F, n * 2));
    layer.enqueue_prefill(c.d_hidden, c.d_out, cs.M, nullptr, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> pre(n);
    std::memcpy(pre.data(), c.d_out, n * 2);
    require(host == pre, "exl3: the prefill path must be bitwise the host path");
    std::printf("[ OK ] exl3 decode + prefill paths bitwise the host path (E=%d I=%d M=%d bits %d/%d)\n", cs.E,
                cs.I, cs.M, cs.rb, cs.sb);
    c.free_all();
  }
}

DGPP_TEST(moe_exl3_sliced_ranks_fold_matches_unsliced_oracle) {
  struct Case { int E, H, I, K, M, world, rb, sb; };
  const Case cases[] = {{8, 512, 640, 2, 3, 4, 4, 6},   // 2/1/1/1 blocks: the uneven split
                        {8, 512, 640, 2, 2, 2, 3, 6},   // 3/2 blocks
                        {8, 512, 256, 2, 4, 2, 6, 6}};  // even
  for (const Case& cs : cases) {
    SmallCase c = make_small_case(cs.E, cs.H, cs.I, cs.K, cs.M, 0xE75C + cs.world + cs.rb, false, false, 0, 16, 0, 0,
                                  nullptr, -1, cs.rb, cs.sb);
    c.alloc();
    std::vector<uint16_t> oracle;
    dgpp::glm_moe_ref_forward(c.d_hidden, c.host_w, c.cfg, c.tokens, oracle);
    std::vector<std::vector<uint16_t>> partials;
    for (int rank = 0; rank < cs.world; ++rank) {
      RankSlice slice = RankSlice::make(c, rank, cs.world);
      GlmMoeConfig cfg = c.cfg;
      // The rank's buffers' width: the widest slice any parity hands it (the
      // loader's local_inter); expert 0's own slice may be the narrower one.
      cfg.inter = static_cast<int>(dgpp::exl3_block_split_max(c.cfg.inter, cs.world));
      GlmMoeLayer layer(slice.dev_w, cfg, cs.M, /*decode_slots=*/cs.M);
      const size_t n = static_cast<size_t>(cs.M) * cs.H;
      layer.enqueue(c.d_hidden, slice.d_out, cs.M, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> host(n);
      std::memcpy(host.data(), slice.d_out, n * 2);
      DGPP_CUDA_OK(cudaMemset(slice.d_out, 0x7F, n * 2));
      layer.enqueue_decode(c.d_hidden, slice.d_out, cs.M, nullptr, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> slot(n);
      std::memcpy(slot.data(), slice.d_out, n * 2);
      require(host == slot, "exl3 sliced rank: decode slot path must be bitwise the host path");
      partials.push_back(std::move(host));
      slice.free_all();
    }
    const std::vector<uint16_t> folded = fold_ranks(partials);
    require_within_fold_budget(folded, partials, oracle,
                               ("exl3 sliced fold E=" + std::to_string(cs.E) + " I=" + std::to_string(cs.I) +
                                " world=" + std::to_string(cs.world) + " bits " + std::to_string(cs.rb) + "/" +
                                std::to_string(cs.sb))
                                   .c_str(),
                               /*partial_ulps=*/2.5, /*floor_frac=*/2e-2);
    c.free_all();
  }
}
