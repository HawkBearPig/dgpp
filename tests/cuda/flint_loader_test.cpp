// The Flint loader over the tiny fixture (docs/flint_plan.md G2): every
// class's slice byte-exact against the fixture's source bytes at worlds 1,
// 2 and 4 — the fused [Q | K | V] projection's rows, o_proj's packed
// columns, the sink widened to fp32, the replicated norms and router with
// its synthesized zero bias, the dense and shared MLP slices, the fp8
// experts' row / column slices with their re-blocked scales, the NVFP4
// experts' packed rows / columns, scales and globals, the globals with the
// vocab-sharded head — and the byte formulas (layer_bytes, resident_bytes)
// against the counted grants, the resident image round trip.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "flint_fixture.hpp"
#include "models/flint/binding.hpp"
#include "models/flint/loader.hpp"

using dgpp::FlintExpectedTensor;
using dgpp::FlintLayerStream;
using dgpp::FlintLocalGeometry;
using dgpp::FlintTextConfig;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  FlintTextConfig cfg;
  std::string dir;
  std::unordered_map<std::string, FlintExpectedTensor> table;
  Fixture() {
    cfg = flintfx::tiny_config();
    dir = (std::filesystem::temp_directory_path() / "dgpp_flint_loader_fixture").string();
    flintfx::write_fixture(cfg, dir);
    for (auto& e : dgpp::flint_expected_text_tensors(cfg)) table[e.name] = e;
  }
  const FlintExpectedTensor& expected(const std::string& name) const {
    auto it = table.find(name);
    require(it != table.end(), "fixture has no tensor " + name);
    return it->second;
  }
  std::vector<uint8_t> source(const std::string& name) const { return flintfx::tensor_bytes(cfg, expected(name)); }
};

std::vector<uint8_t> device_bytes(const void* p, size_t n) {
  std::vector<uint8_t> v(n);
  if (n) DGPP_CUDA_OK(cudaMemcpy(v.data(), p, n, cudaMemcpyDeviceToHost));
  return v;
}

// Rows [r0, r0 + n) of a row-major [rows, row_bytes] source.
std::vector<uint8_t> rows_of(const std::vector<uint8_t>& src, size_t row_bytes, size_t r0, size_t n) {
  return std::vector<uint8_t>(src.begin() + static_cast<std::ptrdiff_t>(r0 * row_bytes),
                              src.begin() + static_cast<std::ptrdiff_t>((r0 + n) * row_bytes));
}
// Byte columns [c0, c0 + n) of every row of a [rows, row_bytes] source, packed.
std::vector<uint8_t> cols_of(const std::vector<uint8_t>& src, size_t row_bytes, size_t c0, size_t n) {
  const size_t rows = src.size() / row_bytes;
  std::vector<uint8_t> out;
  out.reserve(rows * n);
  for (size_t r = 0; r < rows; ++r)
    out.insert(out.end(), src.begin() + static_cast<std::ptrdiff_t>(r * row_bytes + c0),
               src.begin() + static_cast<std::ptrdiff_t>(r * row_bytes + c0 + n));
  return out;
}

void check_layer(const Fixture& fx, FlintLayerStream& s, int layer, int rank, int world) {
  const FlintTextConfig& cfg = fx.cfg;
  const auto& r = s.load_layer(layer);
  const std::string tag = "layer " + std::to_string(layer) + " rank " + std::to_string(rank) + "/" +
                          std::to_string(world) + ": ";
  const FlintLocalGeometry geo = FlintLocalGeometry::from_config(cfg, rank, world, world > 1
                                                                                         ? dgpp::FlintHeadSharding::VocabSharded
                                                                                         : dgpp::FlintHeadSharding::Full);
  const std::string p = dgpp::flint_layer_prefix(layer);
  const size_t H = static_cast<size_t>(cfg.hidden_size), D = 128;
  const size_t lh = static_cast<size_t>(geo.local_heads), lkv = static_cast<size_t>(geo.local_kv_heads);
  require(r.layer == layer && r.moe == cfg.is_moe_layer(layer) && r.swa == cfg.is_swa_layer(layer) &&
              r.rope == cfg.rope_of(layer),
          tag + "layer kinds");
  require(r.bytes == FlintLayerStream::layer_bytes(cfg, layer, rank, world), tag + "layer bytes formula");
  // The replicated norms.
  require(device_bytes(r.input_norm, H * 2) == fx.source(p + "input_layernorm.weight"), tag + "input norm");
  require(device_bytes(r.post_norm, H * 2) == fx.source(p + "post_attention_layernorm.weight"), tag + "post norm");
  // The fused projection: the rank's q rows, then its k rows, then its v rows.
  const auto& a = r.attn;
  require(a.local_heads == geo.local_heads && a.local_kv_heads == geo.local_kv_heads && a.head_begin == geo.head_begin,
          tag + "attention geometry");
  {
    std::vector<uint8_t> want = rows_of(fx.source(p + "self_attn.q_proj.weight"), H * 2, static_cast<size_t>(geo.head_begin) * D, lh * D);
    const auto k = rows_of(fx.source(p + "self_attn.k_proj.weight"), H * 2, static_cast<size_t>(geo.kv_head_begin) * D, lkv * D);
    const auto v = rows_of(fx.source(p + "self_attn.v_proj.weight"), H * 2, static_cast<size_t>(geo.kv_head_begin) * D, lkv * D);
    want.insert(want.end(), k.begin(), k.end());
    want.insert(want.end(), v.begin(), v.end());
    require(device_bytes(a.qkv, want.size()) == want, tag + "fused [Q | K | V] rows");
  }
  require(device_bytes(a.o_proj, H * lh * D * 2) ==
              cols_of(fx.source(p + "self_attn.o_proj.weight"), static_cast<size_t>(cfg.num_attention_heads) * D * 2,
                      static_cast<size_t>(geo.head_begin) * D * 2, lh * D * 2),
          tag + "o_proj packed columns");
  require(device_bytes(a.q_norm, D * 2) == fx.source(p + "self_attn.q_norm.weight"), tag + "q_norm");
  require(device_bytes(a.k_norm, D * 2) == fx.source(p + "self_attn.k_norm.weight"), tag + "k_norm");
  {
    // The sink's heads widened to fp32.
    const auto src = fx.source(p + "self_attn.softmax_offset");
    const auto got = device_bytes(a.sink, lh * 4);
    for (size_t h = 0; h < lh; ++h) {
      uint16_t bits = 0;
      std::memcpy(&bits, src.data() + (static_cast<size_t>(geo.head_begin) + h) * 2, 2);
      float f = 0;
      std::memcpy(&f, got.data() + h * 4, 4);
      require(f == dgpp::bf16_bits_to_float(bits), tag + "sink head " + std::to_string(h));
    }
  }
  const auto check_bf16_mlp = [&](const std::string& mp, const dgpp::FlintBf16MlpResident& m, int64_t I_full) {
    const size_t I = static_cast<size_t>(m.local_inter);
    require(static_cast<int64_t>(I) * world == I_full, tag + mp + "inter slice");
    const size_t r0 = static_cast<size_t>(rank) * I;
    require(device_bytes(m.gate, I * H * 2) == rows_of(fx.source(mp + "gate_proj.weight"), H * 2, r0, I), tag + mp + "gate rows");
    require(device_bytes(m.up, I * H * 2) == rows_of(fx.source(mp + "up_proj.weight"), H * 2, r0, I), tag + mp + "up rows");
    require(device_bytes(m.down, H * I * 2) == cols_of(fx.source(mp + "down_proj.weight"), static_cast<size_t>(I_full) * 2, r0 * 2, I * 2),
            tag + mp + "down columns");
  };
  if (!r.moe) {
    check_bf16_mlp(p + "mlp.", r.dense, cfg.intermediate_size);
    return;
  }
  const auto& m = r.moe_w;
  const size_t E = static_cast<size_t>(cfg.n_routed_experts);
  require(device_bytes(m.router, E * H * 2) == fx.source(p + "mlp.gate.weight"), tag + "router");
  {
    const auto bias = device_bytes(m.router_bias, E * 4);
    for (uint8_t b : bias) require(b == 0, tag + "the synthesized router bias is zero");
  }
  check_bf16_mlp(p + "mlp.shared_expert.", m.shared, cfg.shared_expert_intermediate_size);
  require(m.fp4_mask == cfg.expert_fp4_mask[static_cast<size_t>(layer)], tag + "fp4 mask");
  const size_t I = static_cast<size_t>(m.local_inter), r0 = static_cast<size_t>(rank) * I;
  const size_t I_full = static_cast<size_t>(cfg.moe_intermediate_size);
  for (size_t e = 0; e < E; e += E - 1) {  // the first and the last expert
    const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
    const char* names[3] = {"gate_proj.weight", "up_proj.weight", "down_proj.weight"};
    for (int w = 0; w < 3; ++w) {
      const size_t at = e * 3 + static_cast<size_t>(w);
      const std::string base = ep + names[w];
      const bool down = w == 2;
      const std::string what = tag + "expert " + std::to_string(e) + " " + names[w] + " ";
      if (m.fp4_of(w)) {
        const auto& q = m.experts_fp4[at];
        require(q.payload && q.scales && q.global_scale && q.scale_group == 16, what + "fp4 bound");
        require(static_cast<size_t>(q.rows) == (down ? H : I) && static_cast<size_t>(q.cols) == (down ? I : H), what + "fp4 shape");
        const auto packed = fx.source(base + "_packed");
        const auto scales = fx.source(base + "_scale");
        if (down) {
          require(device_bytes(q.payload, H * I / 2) == cols_of(packed, I_full / 2, r0 / 2, I / 2), what + "fp4 packed columns");
          require(device_bytes(q.scales, H * I / 16) == cols_of(scales, I_full / 16, r0 / 16, I / 16), what + "fp4 scale columns");
        } else {
          require(device_bytes(q.payload, I * H / 2) == rows_of(packed, H / 2, r0, I), what + "fp4 packed rows");
          require(device_bytes(q.scales, I * H / 16) == rows_of(scales, H / 16, r0, I), what + "fp4 scale rows");
        }
        require(device_bytes(q.global_scale, 4) == fx.source(base + "_global_scale"), what + "fp4 global");
        require(!m.experts_fp8[at].payload, what + "no fp8 twin");
      } else {
        const auto& q = m.experts_fp8[at];
        require(q.payload && q.scales, what + "fp8 bound");
        require(static_cast<size_t>(q.rows) == (down ? H : I) && static_cast<size_t>(q.cols) == (down ? I : H), what + "fp8 shape");
        const auto payload = fx.source(base);
        const auto scales = fx.source(base + "_scale_inv");
        if (down) {
          require(device_bytes(q.payload, H * I) == cols_of(payload, I_full, r0, I), what + "fp8 columns");
          // The scale grid: [H/128, I_full/128] -> the rank's I/128 columns at a 128-row grid.
          require(q.scale_block_rows == 128 && q.scale_block_cols == 128, what + "fp8 down grid");
          require(device_bytes(q.scales, (H / 128) * (I / 128) * 4) == cols_of(scales, (I_full / 128) * 4, (r0 / 128) * 4, (I / 128) * 4),
                  what + "fp8 scale columns");
        } else {
          require(device_bytes(q.payload, I * H) == rows_of(payload, H, r0, I), what + "fp8 rows");
          require(q.scale_block_rows == 128 && q.scale_block_cols == 128, what + "fp8 gate/up grid");
          require(device_bytes(q.scales, (I / 128) * (H / 128) * 4) == rows_of(scales, (H / 128) * 4, r0 / 128, I / 128),
                  what + "fp8 scale rows");
        }
        require(!m.experts_fp4[at].payload, what + "no fp4 twin");
      }
    }
  }
}

}  // namespace

DGPP_TEST(flint_loader_geometry_at_worlds_1_2_4) {
  const FlintTextConfig cfg = flintfx::tiny_config();
  for (int world : {1, 2, 4})
    for (int rank = 0; rank < world; ++rank) {
      const FlintLocalGeometry g = FlintLocalGeometry::from_config(cfg, rank, world, dgpp::FlintHeadSharding::VocabSharded);
      require(g.local_heads * world == cfg.num_attention_heads && g.head_begin == rank * g.local_heads, "query heads");
      require(g.local_kv_heads * world == cfg.num_key_value_heads, "kv heads");
      require(g.local_inter * world == cfg.moe_intermediate_size && g.local_inter % 128 == 0, "routed slice");
      require(g.local_dense_inter * world == cfg.intermediate_size, "dense slice");
      require(g.local_shared_inter * world == cfg.shared_expert_intermediate_size, "shared slice");
    }
  bool threw = false;
  try {
    (void)FlintLocalGeometry::from_config(cfg, 0, 8, dgpp::FlintHeadSharding::Full);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "world 8 does not divide the kv heads");
}

DGPP_TEST(flint_loader_slices_every_class_byte_exact_at_worlds_1_2_4) {
  Fixture fx;
  for (int world : {1, 2, 4})
    for (int rank = 0; rank < world; ++rank) {
      FlintLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::FlintResidency::Streaming,
                         world > 1 ? dgpp::FlintHeadSharding::VocabSharded : dgpp::FlintHeadSharding::Full);
      for (int l = 0; l < fx.cfg.num_hidden_layers; ++l) check_layer(fx, s, l, rank, world);
      // The globals: the embedding replicated, the head vocab-sharded.
      const auto& g = s.load_globals();
      const size_t H = static_cast<size_t>(fx.cfg.hidden_size);
      require(device_bytes(g.embed, static_cast<size_t>(fx.cfg.vocab_size) * H * 2) == fx.source("model.embed_tokens.weight"), "embed");
      require(device_bytes(g.final_norm, H * 2) == fx.source("model.norm.weight"), "final norm");
      const int count = FlintLayerStream::lm_vocab_count(fx.cfg, rank, world,
                                                         world > 1 ? dgpp::FlintHeadSharding::VocabSharded : dgpp::FlintHeadSharding::Full);
      require(g.lm_vocab_count == count && (world == 1 ? g.lm_vocab_begin == 0 : true), "head slice geometry");
      require(device_bytes(g.lm_head, static_cast<size_t>(count) * H * 2) ==
                  rows_of(fx.source("lm_head.weight"), H * 2, static_cast<size_t>(g.lm_vocab_begin), static_cast<size_t>(count)),
              "head rows");
      std::printf("[ OK ] flint loader rank %d/%d: every class byte-exact\n", rank, world);
    }
}

DGPP_TEST(flint_loader_resident_mode_and_byte_formulas) {
  Fixture fx;
  const int layers = fx.cfg.num_hidden_layers;
  FlintLayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::FlintResidency::Resident, dgpp::FlintHeadSharding::VocabSharded);
  for (int l = 0; l < layers; ++l) check_layer(fx, s, l, 1, 2);
  size_t total = s.load_globals().bytes;
  for (int l = 0; l < layers; ++l) total += s.load_layer(l).bytes;
  require(FlintLayerStream::resident_bytes(fx.cfg, 1, 2, dgpp::FlintHeadSharding::VocabSharded, false) == total,
          "resident bytes formula");
  // A resident layer is stable across loads (the same view).
  const auto* first = s.load_layer(3).attn.qkv;
  require(s.load_layer(3).attn.qkv == first, "resident layer views are stable");
}

int main() { return dgpp::test::run_all(); }
