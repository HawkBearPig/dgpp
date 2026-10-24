// The Engram layer (models/dsv41/engram_layer.hpp) on the loader fixture:
// the host-node staging from the mmap'ed tables (the hash of two request
// spans with stored contexts, the rows of both Engram layers gathered in
// one callback, converted to bf16) bitwise a host hash-and-lookup over
// the fixture's table bytes at world 1 and every rank of world 4; the
// per-row context snapshots; the TP-4 wkv partials summing to the world-1
// projection; the gate running on the streams.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "dsv41_fixture.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/latent_format.hpp"
#include "models/dsv41/engram_layer.hpp"
#include "models/dsv41/loader.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::kda_test::DevBuf;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}
template <typename T>
DevBuf upload(const std::vector<T>& v) {
  DevBuf b(v.size() * sizeof(T));
  b.upload(v.data(), v.size() * sizeof(T));
  return b;
}
template <typename T>
std::vector<T> download(const void* p, size_t n) {
  std::vector<T> v(n);
  DGPP_CUDA_OK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}

struct Fixture {
  dgpp::Dsv41TextConfig cfg;
  std::string dir;
  std::vector<dgpp::Dsv41ExpectedTensor> table;
  dgpp::Dsv41EngramSidecar sidecar;
  std::vector<uint8_t> bytes(const std::string& name) const {
    for (const auto& e : table)
      if (e.name == name) return dsv41fx::fixture_bytes(e);
    throw std::runtime_error("fixture: no tensor " + name);
  }
};
Fixture make_fixture() {
  Fixture fx;
  fx.cfg = dsv41fx::tiny_config();
  fx.dir = (fs::current_path() / "dsv41_engram_layer_fixture").string();
  dsv41fx::write_fixture(fx.cfg, fx.dir);
  fx.table = dgpp::dsv41_expected_text_tensors(fx.cfg);
  fx.sidecar = dsv41fx::fixture_sidecar(fx.cfg, fx.dir);
  return fx;
}

// The reference hash of one row per layer: y[0] the token's class, y[k]
// the class k tokens back (the context), the products XORed one lookback
// at a time, each (n-gram, head) bucket = rolling mod prime + offset.
std::vector<int32_t> hash_row(const dgpp::Dsv41EngramSidecar& sc, const std::vector<int64_t>& y) {
  std::vector<int32_t> out;
  const int G = sc.ngrams(), Hh = sc.n_heads;
  for (int l = 0; l < sc.layers(); ++l) {
    const int64_t* mult = sc.multipliers.data() + static_cast<size_t>(l) * sc.max_ngram_size;
    uint64_t rolling = static_cast<uint64_t>(y[0]) * static_cast<uint64_t>(mult[0]);
    for (int n = 1; n < sc.max_ngram_size; ++n) {
      rolling ^= static_cast<uint64_t>(y[static_cast<size_t>(n)]) * static_cast<uint64_t>(mult[n]);
      for (int h = 0; h < Hh; ++h) {
        const size_t at = (static_cast<size_t>(l) * G + (n - 1)) * Hh + h;
        int64_t r = static_cast<int64_t>(rolling) % sc.primes[at];
        if (r < 0) r += sc.primes[at];
        out.push_back(static_cast<int32_t>(r + sc.offsets[at]));
      }
    }
  }
  return out;
}
}  // namespace

DGPP_TEST(dsv41_engram_layer_stages_rows_from_the_mmap_bitwise_the_host_lookup) {
  const Fixture fx = make_fixture();
  const dgpp::Dsv41TextConfig& cfg = fx.cfg;
  const int L = cfg.engram_index(cfg.engram_layer_ids.back()) + 1, G = cfg.engram_max_ngram_size - 1, Hh = cfg.engram_n_heads;
  const int hd = cfg.engram_head_dim, entries = G * Hh;
  // Two requests: request 0 at positions 0..4 (a fresh context: pad
  // classes), request 1 at 10..12 with a stored context of three classes;
  // a padding row at the end of request 0's span.
  const std::vector<int64_t> tokens = {5, 17, 300, 511, 42, 7, 9, 128, 77};
  const std::vector<int32_t> req_ids = {0, 0, 0, 0, 0, 0, 1, 1, 1};
  const std::vector<int64_t> pos = {0, 1, 2, 3, 4, -1, 10, 11, 12};
  const std::vector<int32_t> spans = {0, 6, 6, 3};
  const int rows = int(tokens.size());
  std::vector<int32_t> ctx(2 * 4, fx.sidecar.pad_class);
  ctx[4] = 33; ctx[5] = 120; ctx[6] = 7; ctx[7] = fx.sidecar.pad_class;  // request 1: the classes of its 3 previous tokens (newest first)
  // The oracle: per row, the ids and the context after it.
  std::vector<std::vector<int32_t>> want_ids(static_cast<size_t>(rows));
  std::vector<std::vector<int32_t>> want_ctx(size_t(rows), std::vector<int32_t>(4, fx.sidecar.pad_class));
  {
    std::vector<int32_t> hist0(ctx.begin(), ctx.begin() + 4), hist1(ctx.begin() + 4, ctx.end());
    for (int q = 0; q < 2; ++q) {
      std::vector<int32_t>& hist = q == 0 ? hist0 : hist1;
      for (int t = spans[size_t(2 * q)]; t < spans[size_t(2 * q)] + spans[size_t(2 * q + 1)]; ++t) {
        const int32_t c = fx.sidecar.token_map[size_t(tokens[size_t(t)])];
        std::vector<int64_t> y = {c, hist[0], hist[1], hist[2]};
        want_ids[size_t(t)] = hash_row(fx.sidecar, y);
        if (pos[size_t(t)] >= 0) hist = {c, hist[0], hist[1], hist[2]};  // a padding row leaves the context
        want_ctx[size_t(t)] = {hist[0], hist[1], hist[2], hist[3]};
      }
    }
  }
  std::vector<std::vector<uint16_t>> kv_partials;  // world 4's, [rows, (hc+1)H]
  std::vector<uint16_t> kv_world1;
  for (const int world : {1, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      dgpp::Dsv41LayerStream s(cfg, fx.dir, rank, world, dgpp::Dsv41Residency::Streaming,
                               world > 1 ? dgpp::Dsv41HeadSharding::VocabSharded : dgpp::Dsv41HeadSharding::Full);
      const dgpp::Dsv41EngramTables& tables = s.load_engram_tables();
      dgpp::Dsv41EngramLayer layer(cfg, tables, fx.sidecar, 16, 2);
      DevBuf dtok = upload(tokens), dri = upload(req_ids), dpos = upload(pos), dsp = upload(spans), dctx = upload(ctx),
          dctx_rows(size_t(rows) * 16);
      layer.stage(static_cast<const int64_t*>(dtok.p), rows, static_cast<const int32_t*>(dri.p), static_cast<const int64_t*>(dpos.p),
                  static_cast<const int32_t*>(dsp.p), 2, static_cast<const int32_t*>(dctx.p), 0);
      layer.context_rows(static_cast<const int64_t*>(dtok.p), rows, static_cast<const int32_t*>(dri.p),
                         static_cast<const int64_t*>(dpos.p), static_cast<const int32_t*>(dsp.p), 2, static_cast<int32_t*>(dctx.p),
                         static_cast<int32_t*>(dctx_rows.p), 0);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      layer.check_staged();
      const std::string tag = "world " + std::to_string(world) + " rank " + std::to_string(rank) + ": ";
      // The ids (pinned) against the oracle.
      const int32_t* ids = layer.ids();
      for (int t = 0; t < rows; ++t)
        for (int i = 0; i < L * entries; ++i)
          require(ids[size_t(t) * L * entries + i] == want_ids[size_t(t)][size_t(i)], tag + "hash id row " + std::to_string(t));
      // The contexts after every row and in place.
      // The context holds max_ngram - 1 words (the fourth is unused).
      const auto ctx_rows = download<int32_t>(dctx_rows.p, size_t(rows) * 4);
      for (int t = 0; t < rows; ++t)
        for (int k = 0; k < G; ++k) require(ctx_rows[size_t(t) * 4 + k] == want_ctx[size_t(t)][size_t(k)], tag + "context row " + std::to_string(t));
      const auto ctx_after = download<int32_t>(dctx.p, 8);
      for (int k = 0; k < G; ++k) {
        require(ctx_after[size_t(k)] == want_ctx[4][size_t(k)], tag + "request 0's context in place (its last real row)");
        require(ctx_after[size_t(4 + k)] == want_ctx[8][size_t(k)], tag + "request 1's context in place");
      }
      // The staged rows of both layers against the fixture's table bytes.
      for (int li = 0; li < L; ++li) {
        layer.embed(li, rows, 0);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        const auto e = download<uint16_t>(layer.embedded(), size_t(rows) * layer.width_local());
        const std::string p = dgpp::dsv41_layer_prefix(cfg, cfg.engram_layer_ids[size_t(li)]) + "engram.embed.";
        const std::vector<uint8_t> payload = fx.bytes(p + "weight"), scales = fx.bytes(p + "scale");
        for (int t = 0; t < rows; ++t)
          for (int n = 0; n < G; ++n)
            for (int hl = 0; hl < tables.heads_local; ++hl) {
              const int col = n * Hh + tables.head_begin + hl;
              const int32_t id = want_ids[size_t(t)][size_t(li * entries + col)];
              const int j = n * tables.heads_local + hl;
              for (int d = 0; d < hd; ++d) {
                const float v = dgpp::fp8_e4m3_bits_to_float(payload[size_t(id) * hd + d]) *
                                dgpp::e8m0_byte_to_float(scales[size_t(id) * (hd / 32) + d / 32]);
                require(e[(size_t(t) * tables.rows_local() + j) * hd + d] == float_to_bf16_bits(v),
                        tag + "staged row " + std::to_string(t) + " layer " + std::to_string(li));
              }
            }
      }
      // The projection (the first Engram layer's wkv slice) and the gate:
      // stage again (the last embed released the staging), embed layer 0.
      const dgpp::Dsv41LayerResident& r = s.load_layer(cfg.engram_layer_ids[0]);
      dgpp::Dsv41EngramLayerWeights w;
      w.wkv = r.engram.wkv;
      w.wkv_x = r.engram.wkv_x;  // the EXL3 fixture's (DGPP_DSV41_FIXTURE=exl3)
      w.q_weight = r.engram.q_weight;
      w.k_weight = r.engram.k_weight;
      w.table_index = 0;
      layer.stage(static_cast<const int64_t*>(dtok.p), rows, static_cast<const int32_t*>(dri.p), static_cast<const int64_t*>(dpos.p),
                  static_cast<const int32_t*>(dsp.p), 2, static_cast<const int32_t*>(dctx.p), 0);
      layer.embed(0, rows, 0);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const size_t N = size_t(cfg.hc_mult + 1) * cfg.hidden_size;
      layer.project(w, rows, nullptr, 0);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const auto kv = download<uint16_t>(layer.kv_partial(), size_t(rows) * N);
      if (world == 1) kv_world1 = kv;
      else kv_partials.push_back(kv);
      auto x = dgpp::kda_test::random_bf16_bits(0xE6 + rank, size_t(rows) * cfg.hc_mult * cfg.hidden_size, -3, 1);
      DevBuf dx = upload(x);
      layer.gate(w, static_cast<uint16_t*>(dx.p), layer.kv_partial(), rows, 0);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const auto xg = download<uint16_t>(dx.p, x.size());
      int changed = 0;
      for (size_t i = 0; i < x.size(); ++i) {
        require(std::isfinite(bf16_bits_to_float(xg[i])), tag + "gated stream finite");
        changed += xg[i] != x[i];
      }
      require(changed > int(x.size() / 2), tag + "the gate updated the streams");
    }
  }
  // World 4's partials sum to world 1's projection (fp32 sums of bf16
  // partials against the one-rank bf16 result: the fold's rounding).
  require(kv_partials.size() == 4, "four partials");
  std::vector<uint16_t> summed(kv_world1.size());
  for (size_t i = 0; i < summed.size(); ++i) {
    float acc = 0.0f;
    for (const auto& p : kv_partials) acc += bf16_bits_to_float(p[i]);
    summed[i] = float_to_bf16_bits(acc);
  }
  dgpp::kda_test::require_bf16("TP-4 Engram partials vs the world-1 projection",
                               dgpp::kda_test::compare_bf16(summed, kv_world1, 8), 0.01, 0.05);
}

int main() { return dgpp::test::run_all(); }
