// The DeepSeek-V4.1-Flash resident loader on the synthetic fixture: every
// class lands byte-exact at worlds 1, 2 and 4 as the slice formulas of
// docs/deepseek_v41_flash_plan.md §2 say (wq_b head blocks, wo_a group
// rows, wo_b packed group columns, the sink per head, the fp8 32 x 32
// scale grids converted e8m0 -> fp32 alongside their slices, the MXFP4
// experts sliced on the intermediate dim with their per-row e8m0 scales,
// the Engram wkv's K columns of the rank's hash heads, the mHC
// coefficients rounded fp32 -> bf16 once, the replicated set verbatim,
// the draft's head tensors whole); the byte formulas equal actual
// usage and the source-byte plan equals the bytes read; the globals; the
// Engram tables mmap'ed and gathered from the shard; resident cache hits
// and the image round trip; a geometry the world does not divide and a
// checkpoint that does not bind are refused by name.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "dsv41_fixture.hpp"
#include "glm_rng.hpp"
#include "loaders/safetensors.hpp"
#include "models/dsv41/binding.hpp"
#include "models/dsv41/config.hpp"
#include "models/dsv41/engram_tables.hpp"
#include "models/dsv41/loader.hpp"
#include "models/exl3_matrix.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::Dsv41ExpectedTensor;
using dgpp::Dsv41LayerStream;
using dgpp::Dsv41TextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  Dsv41TextConfig cfg;
  std::string dir;
  std::vector<Dsv41ExpectedTensor> table;
  const Dsv41ExpectedTensor& expected(const std::string& name) const {
    for (const auto& e : table)
      if (e.name == name) return e;
    throw std::runtime_error("fixture: no expected tensor " + name);
  }
  std::vector<uint8_t> bytes(const std::string& name) const { return dsv41fx::fixture_bytes(expected(name)); }
};

Fixture write_fixture() {
  Fixture fx;
  fx.cfg = dsv41fx::tiny_config();
  fx.dir = (fs::current_path() / "dsv41_loader_fixture").string();
  dsv41fx::write_fixture(fx.cfg, fx.dir);
  fx.table = dgpp::dsv41_expected_text_tensors(fx.cfg);
  return fx;
}

std::vector<uint8_t> device_bytes(const void* dev, size_t n) {
  std::vector<uint8_t> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), dev, n, cudaMemcpyDeviceToHost));
  return h;
}
std::vector<uint8_t> host_rows(const std::vector<uint8_t>& t, size_t row_bytes, int64_t r0, int64_t rows) {
  return std::vector<uint8_t>(t.begin() + r0 * row_bytes, t.begin() + (r0 + rows) * row_bytes);
}
std::vector<uint8_t> host_cols(const std::vector<uint8_t>& t, int64_t rows, size_t full, size_t c0, size_t w) {
  std::vector<uint8_t>
      out;
  out.reserve(rows * w);
  for (int64_t r = 0; r < rows; ++r)
    out.insert(out.end(), t.begin() + r * full + c0, t.begin() + r * full + c0 + w);
  return out;
}
void expect_device_equals(const void* dev, const std::vector<uint8_t>& want, const std::string& what) {
  require(dev != nullptr, what + ": null resident pointer");
  const std::vector<uint8_t> got = device_bytes(dev, want.size());
  require(got == want, what + ": resident bytes differ from the checkpoint slice");
}
float e8m0_value(uint8_t b) { return std::ldexp(1.0f, static_cast<int>(b) - 127); }
std::vector<uint8_t> f32_bytes(const std::vector<float>& v) {
  std::vector<uint8_t> out(v.size() * 4);
  std::memcpy(out.data(), v.data(), out.size());
  return out;
}

// An fp8 pair against the checkpoint: rows [row0, +rows) of every column
// (`ranges` empty), or the packed column ranges of every row. The scale
// grid: the e8m0 bytes of the slice's blocks as fp32.
using Ranges = std::vector<std::pair<int64_t, int64_t>>;
void check_fp8(const Fixture& fx, const dgpp::GlmQuantMatrix& q, const std::string& base, int64_t row0, int64_t rows,
               const Ranges& ranges, const std::string& tag) {
  const auto& ew = fx.expected(base + ".weight");
  const auto& es = fx.expected(base + ".scale");
  const int64_t N = ew.shape[0], K = ew.shape[1], SR = es.shape[0], SC = es.shape[1];
  const std::vector<uint8_t> w = fx.bytes(base + ".weight");
  const std::vector<uint8_t> s = fx.bytes(base + ".scale");
  require(q.scale_block_rows == 32 && q.scale_block_cols == 32, tag + " scale grid");
  std::vector<uint8_t> want_w;
  std::vector<float> want_s;
  if (ranges.empty()) {
    require(q.rows == rows && q.cols == K, tag + " geometry");
    want_w = host_rows(w, static_cast<size_t>(K), row0, rows);
    const int64_t sr = (rows + 31) / 32;
    for (int64_t r = 0; r < sr; ++r)
      for (int64_t c = 0; c < SC; ++c) want_s.push_back(e8m0_value(s[static_cast<size_t>((row0 / 32 + r) * SC + c)]));
  } else {
    int64_t cols = 0;
    for (const auto& [start, count] : ranges) cols += count;
    require(q.rows == N && q.cols == cols, tag + " geometry");
    for (int64_t r = 0; r < N; ++r)
      for (const auto& [start, count] : ranges)
        want_w.insert(want_w.end(), w.begin() + r * K + start, w.begin() + r * K + start + count);
    for (int64_t r = 0; r < SR; ++r)
      for (const auto& [start, count] : ranges)
        for (int64_t c = 0; c < count / 32; ++c) want_s.push_back(e8m0_value(s[static_cast<size_t>(r * SC + start / 32 + c)]));
  }
  expect_device_equals(q.payload, want_w, tag + " payload");
  expect_device_equals(q.scales, f32_bytes(want_s), tag + " scales");
}
void check_fp8_full(const Fixture& fx, const dgpp::GlmQuantMatrix& q, const std::string& base, const std::string& tag) {
  check_fp8(fx, q, base, 0, fx.expected(base + ".weight").shape[0], {}, tag);
}

// An MXFP4 matrix against the checkpoint's row / column slice.
void check_mxfp4(const Fixture& fx, const dgpp::GlmFp4Matrix& m, const std::string& base, bool rows_slice, int64_t start,
                 int64_t count, const std::string& tag) {
  const auto& ew = fx.expected(base + ".weight");
  const int64_t N = ew.shape[0], K = ew.shape[1] * 2;
  const size_t pr = static_cast<size_t>(K / 2), sr = static_cast<size_t>(K / 32);
  require(m.scale_group == 32 && m.mxfp4() && m.global_scale == nullptr, tag + " format");
  if (rows_slice) {
    require(m.rows == count && m.cols == K, tag + " geometry");
    expect_device_equals(m.payload, host_rows(fx.bytes(base + ".weight"), pr, start, count), tag + " payload rows");
    expect_device_equals(m.scales, host_rows(fx.bytes(base + ".scale"), sr, start, count), tag + " scale rows");
  } else {
    require(m.rows == N && m.cols == count, tag + " geometry");
    expect_device_equals(m.payload, host_cols(fx.bytes(base + ".weight"), N, pr, start / 2, count / 2), tag + " payload cols");
    expect_device_equals(m.scales, host_cols(fx.bytes(base + ".scale"), N, sr, start / 32, count / 32), tag + " scale cols");
  }
}

std::vector<uint8_t> f32_as_bf16(const std::vector<uint8_t>& f32) {
  std::vector<uint8_t> out(f32.size() / 2);
  for (size_t i = 0; i < f32.size() / 4; ++i) {
    float v;
    std::memcpy(&v, f32.data() + i * 4, 4);
    const uint16_t b = dgpp::float_to_bf16_bits(v);
    std::memcpy(out.data() + i * 2, &b, 2);
  }
  return out;
}
std::vector<uint8_t> bf16_as_f32(const std::vector<uint8_t>& bf16) {
  std::vector<uint8_t> out(bf16.size() * 2);
  for (size_t i = 0; i < bf16.size() / 2; ++i) {
    uint16_t b;
    std::memcpy(&b, bf16.data() + i * 2, 2);
    const float v = dgpp::bf16_bits_to_float(b);
    std::memcpy(out.data() + i * 4, &v, 4);
  }
  return out;
}

void check_layer(const Fixture& fx, const Dsv41LayerStream& s, const dgpp::Dsv41LayerResident& r) {
  const Dsv41TextConfig& cfg = fx.cfg;
  const dgpp::Dsv41LocalGeometry& g = s.geometry();
  const int world = s.world(), rank = s.rank();
  const std::string p = dgpp::dsv41_layer_prefix(cfg, r.layer);
  const std::string tag = "world " + std::to_string(world) + " rank " + std::to_string(rank) + " layer " +
                          std::to_string(r.layer) + " ";
  const bool draft = cfg.is_draft(r.layer);
  expect_device_equals(r.attn_norm, fx.bytes(p + "attn_norm.weight"), tag + "attn norm");
  expect_device_equals(r.ffn_norm, fx.bytes(p + "ffn_norm.weight"), tag + "ffn norm");
  // mHC: fn rounded once to bf16, base / scale fp32 verbatim.
  expect_device_equals(r.mhc.attn_fn, f32_as_bf16(fx.bytes(p + "hc_attn_fn")), tag + "hc attn fn");
  expect_device_equals(r.mhc.attn_base, fx.bytes(p + "hc_attn_base"), tag + "hc attn base");
  expect_device_equals(r.mhc.attn_scale, fx.bytes(p + "hc_attn_scale"), tag + "hc attn scale");
  expect_device_equals(r.mhc.ffn_fn, f32_as_bf16(fx.bytes(p + "hc_ffn_fn")), tag + "hc ffn fn");
  expect_device_equals(r.mhc.ffn_base, fx.bytes(p + "hc_ffn_base"), tag + "hc ffn base");
  expect_device_equals(r.mhc.ffn_scale, fx.bytes(p + "hc_ffn_scale"), tag + "hc ffn scale");
  // Attention.
  const std::string ap = p + "attn.";
  const int64_t hd = cfg.head_dim, ol = cfg.o_lora_rank;
  require(r.attn.local_heads == g.local_heads && r.attn.head_begin == g.head_begin, tag + "head geometry");
  require(r.attn.local_groups == g.local_groups && r.attn.group_begin == g.group_begin, tag + "group geometry");
  require(g.local_heads == cfg.num_attention_heads / world && g.local_groups == cfg.o_groups / world, tag + "geometry formulas");
  expect_device_equals(r.attn.q_norm, fx.bytes(ap + "q_norm.weight"), tag + "q norm");
  expect_device_equals(r.attn.kv_norm, fx.bytes(ap + "kv_norm.weight"), tag + "kv norm");
  expect_device_equals(r.attn.attn_sink, host_rows(fx.bytes(ap + "attn_sink"), 4, g.head_begin, g.local_heads), tag + "sink");
  check_fp8_full(fx, r.attn.wq_a, ap + "wq_a", tag + "wq_a");
  check_fp8_full(fx, r.attn.wkv, ap + "wkv", tag + "wkv");
  check_fp8(fx, r.attn.wq_b, ap + "wq_b", g.head_begin * hd, g.local_heads * hd, {}, tag + "wq_b");
  check_fp8(fx, r.attn.wo_a, ap + "wo_a", g.group_begin * ol, g.local_groups * ol, {}, tag + "wo_a");
  check_fp8(fx, r.attn.wo_b, ap + "wo_b", 0, 0, {{g.group_begin * ol, g.local_groups * ol}}, tag + "wo_b");
  if (cfg.is_index_source(r.layer)) {
    require(r.attn.index_source(), tag + "indexer present");
    check_fp8_full(fx, r.attn.idx_wq_b, ap + "indexer.wq_b", tag + "indexer wq_b");
    expect_device_equals(r.attn.idx_wp, fx.bytes(ap + "indexer.weights_proj.weight"), tag + "indexer weights_proj");
  } else {
    require(!r.attn.index_source() && r.attn.idx_wp == nullptr, tag + "no indexer");
  }
  if (cfg.is_kv_source(r.layer)) {
    require(r.attn.kv_source(), tag + "kv source present");
    expect_device_equals(r.attn.idx_wk, fx.bytes(ap + "indexer.wk.weight"), tag + "index keys wk");
    expect_device_equals(r.attn.idx_k_norm, fx.bytes(ap + "indexer.k_norm.weight"), tag + "index keys norm");
    expect_device_equals(r.attn.comp_wkv, fx.bytes(ap + "compressor.wkv.weight"), tag + "compressor wkv");
    expect_device_equals(r.attn.comp_norm, fx.bytes(ap + "compressor.norm.weight"), tag + "compressor norm");
    if (cfg.compress_ratio(r.layer) > 1)
      expect_device_equals(r.attn.comp_wgate, fx.bytes(ap + "compressor.wgate.weight"), tag + "compressor wgate");
    else
      require(r.attn.comp_wgate == nullptr, tag + "no gate at ratio 1");
  } else {
    require(!r.attn.kv_source() && r.attn.idx_wk == nullptr && r.attn.comp_wgate == nullptr, tag + "no compressor");
  }
  // MoE.
  const std::string mp = p + "ffn.";
  expect_device_equals(r.moe.router, fx.bytes(mp + "gate.weight"), tag + "router");
  expect_device_equals(r.moe.router_bias, fx.bytes(mp + "gate.bias"), tag + "router bias");
  const int64_t I = g.local_inter, S = g.local_shared_inter;
  require(I == cfg.moe_intermediate_size / world && S == cfg.shared_expert_inter() / world, tag + "inter formulas");
  require(r.moe.local_inter == I && r.moe.local_shared_inter == S, tag + "moe inter");
  const int E = draft ? cfg.dspark_n_routed_experts : cfg.n_routed_experts;
  require(r.moe.n_experts == E && static_cast<int>(r.moe.experts.size()) == E * 3, tag + "expert count");
  for (int e = 0; e < E; ++e) {
    const std::string ep = mp + "experts." + std::to_string(e) + ".";
    check_mxfp4(fx, r.moe.expert(e, 0), ep + "w1", true, rank * I, I, tag + "expert gate");
    check_mxfp4(fx, r.moe.expert(e, 1), ep + "w3", true, rank * I, I, tag + "expert up");
    check_mxfp4(fx, r.moe.expert(e, 2), ep + "w2", false, rank * I, I, tag + "expert down");
  }
  const std::string sp = mp + "shared_experts.";
  check_fp8(fx, r.moe.shared[0], sp + "w1", rank * S, S, {}, tag + "shared gate");
  check_fp8(fx, r.moe.shared[1], sp + "w3", rank * S, S, {}, tag + "shared up");
  check_fp8(fx, r.moe.shared[2], sp + "w2", 0, 0, {{rank * S, S}}, tag + "shared down");
  // Engram.
  if (cfg.has_engram(r.layer)) {
    require(r.engram.present && r.engram.table_index == cfg.engram_index(r.layer), tag + "engram present");
    Ranges ranges;
    const int64_t ehd = cfg.engram_head_dim;
    for (int n = 0; n < cfg.engram_max_ngram_size - 1; ++n)
      ranges.emplace_back((static_cast<int64_t>(n) * cfg.engram_n_heads + g.engram_head_begin) * ehd,
                          static_cast<int64_t>(g.engram_heads_local) * ehd);
    check_fp8(fx, r.engram.wkv, p + "engram.wkv", 0, 0, ranges, tag + "engram wkv");
    expect_device_equals(r.engram.q_weight, fx.bytes(p + "engram.q_weight"), tag + "engram q");
    expect_device_equals(r.engram.k_weight, fx.bytes(p + "engram.k_weight"), tag + "engram k");
  } else {
    require(!r.engram.present && r.engram.wkv.payload == nullptr, tag + "no engram");
  }
  // Draft.
  if (draft) {
    const int stage = cfg.draft_stage(r.layer);
    if (stage == 0) {
      check_fp8_full(fx, r.draft.main_proj, p + "main_proj", tag + "main_proj");
      expect_device_equals(r.draft.main_norm, fx.bytes(p + "main_norm.weight"), tag + "main norm");
    } else {
      require(r.draft.main_proj.payload == nullptr && r.draft.main_norm == nullptr, tag + "no main_proj past stage 0");
    }
    if (stage == cfg.num_nextn_predict_layers - 1) {
      expect_device_equals(r.draft.norm, fx.bytes(p + "norm.weight"), tag + "draft norm");
      expect_device_equals(r.draft.markov_embed, fx.bytes(p + "markov_head.embed.weight"), tag + "markov embed");
      expect_device_equals(r.draft.markov_head, fx.bytes(p + "markov_head.head.weight"), tag + "markov head");
      expect_device_equals(r.draft.confidence, bf16_as_f32(fx.bytes(p + "confidence_head.proj.weight")), tag + "confidence");
    } else {
      require(r.draft.norm == nullptr && r.draft.markov_head == nullptr, tag + "no head before the last stage");
    }
  } else {
    require(r.draft.main_proj.payload == nullptr && r.draft.markov_head == nullptr, tag + "no draft head on a main layer");
  }
  require(r.bytes == Dsv41LayerStream::layer_bytes(cfg, r.layer, rank, world), tag + "layer bytes formula");
}

}  // namespace

DGPP_TEST(dsv41_loader_slices_every_class_byte_exact_at_worlds_1_2_4) {
  const Fixture fx = write_fixture();
  const int layers = fx.cfg.max_layer();
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::Dsv41HeadSharding::VocabSharded : dgpp::Dsv41HeadSharding::Full;
      Dsv41LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::Dsv41Residency::Streaming, head, /*resident_mtp=*/true);
      for (int l = 0; l < layers; ++l) {
        const uint64_t before = s.source_bytes_read();
        const auto& r = s.load_layer(l);
        require(r.layer == l, "layer index");
        check_layer(fx, s, r);
        require(s.source_bytes_read() - before == Dsv41LayerStream::planned_layer_source_bytes(fx.cfg, l, rank, world),
                "the source-byte plan equals the bytes read");
        s.release_layer();
      }
    }
  }
}

DGPP_TEST(dsv41_loader_globals_slices) {
  const Fixture fx = write_fixture();
  const size_t H2 = static_cast<size_t>(fx.cfg.hidden_size) * 2;
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::Dsv41HeadSharding::VocabSharded : dgpp::Dsv41HeadSharding::Full;
      Dsv41LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::Dsv41Residency::Streaming, head);
      const auto& g = s.load_globals();
      expect_device_equals(g.embed, fx.bytes("embed.weight"), "embed");
      expect_device_equals(g.final_norm, fx.bytes("norm.weight"), "final norm");
      expect_device_equals(g.lm_head, host_rows(fx.bytes("head.weight"), H2, g.lm_vocab_begin, g.lm_vocab_count), "lm head slice");
      require(g.lm_vocab_count == Dsv41LayerStream::lm_vocab_count(fx.cfg, rank, world, head), "lm head count");
      require(g.lm_vocab_count == (world > 1 ? fx.cfg.vocab_size / world : fx.cfg.vocab_size), "lm head formula");
      require(g.embed_vocab_begin == 0 && g.embed_vocab_count == fx.cfg.vocab_size, "the embedding is whole by default");
      require(g.bytes == Dsv41LayerStream::globals_bytes(fx.cfg, rank, world, head), "globals formula");
    }
  }
  Dsv41LayerStream::set_embed_vocab_sharded(true);
  for (const int world : {1, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::Dsv41HeadSharding::VocabSharded : dgpp::Dsv41HeadSharding::Full;
      Dsv41LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::Dsv41Residency::Streaming, head);
      const auto& g = s.load_globals();
      if (world == 1)
        require(g.embed_vocab_begin == 0 && g.embed_vocab_count == fx.cfg.vocab_size, "world 1 keeps the whole table");
      else
        require(g.embed_vocab_begin == g.lm_vocab_begin && g.embed_vocab_count == g.lm_vocab_count, "the embedding slice is the lm head's");
      expect_device_equals(g.embed, host_rows(fx.bytes("embed.weight"), H2, g.embed_vocab_begin, g.embed_vocab_count), "embed slice");
      require(g.bytes == Dsv41LayerStream::globals_bytes(fx.cfg, rank, world, head), "globals formula (sharded)");
    }
  }
  Dsv41LayerStream::set_embed_vocab_sharded(false);
}

DGPP_TEST(dsv41_loader_engram_tables_mmap_and_gather) {
  const Fixture fx = write_fixture();
  const Dsv41TextConfig& cfg = fx.cfg;
  const int L = static_cast<int>(cfg.engram_layer_ids.size()), G = cfg.engram_max_ngram_size - 1, Hh = cfg.engram_n_heads;
  const int row_bytes = cfg.engram_head_dim + cfg.engram_head_dim / 32;
  for (const int world : {1, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      Dsv41LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::Dsv41Residency::Resident,
                         world > 1 ? dgpp::Dsv41HeadSharding::VocabSharded : dgpp::Dsv41HeadSharding::Full);
      const dgpp::Dsv41EngramTables& t = s.load_engram_tables();
      require(static_cast<int>(t.tables.size()) == L && t.heads == Hh && t.heads_local == Hh / world &&
                  t.head_begin == rank * (Hh / world) && t.ngrams == G && t.rows_local() == G * (Hh / world),
              "table geometry");
      for (int n = 0; n < G; ++n)
        for (int h = 0; h < t.heads_local; ++h)
          require(t.sel[static_cast<size_t>(n * t.heads_local + h)] == n * Hh + t.head_begin + h, "row selection");
      // A large gather (the multi-threaded path at world 1: 64 x 12 rows).
      const int tokens = 64;
      const int ids_stride = L * G * Hh;
      std::vector<int32_t> ids(static_cast<size_t>(tokens) * ids_stride);
      glmrng::Rng rng(0xE1A + world * 16 + rank);
      for (int li = 0; li < L; ++li) {
        const auto& tab = *t.tables[static_cast<size_t>(li)];
        require(tab.rows() == cfg.engram_num_embeddings[static_cast<size_t>(li)] && tab.head_dim() == cfg.engram_head_dim &&
                    tab.row_bytes() == row_bytes,
                "table rows");
        for (int tk = 0; tk < tokens; ++tk)
          for (int c = 0; c < G * Hh; ++c)
            ids[static_cast<size_t>(tk) * ids_stride + li * G * Hh + c] = static_cast<int32_t>(rng.next() % static_cast<uint64_t>(tab.rows()));
      }
      const std::string p1 = dgpp::dsv41_layer_prefix(cfg, cfg.engram_layer_ids[0]) + "engram.embed.";
      const std::vector<uint8_t> payload = fx.bytes(p1 + "weight"), scales = fx.bytes(p1 + "scale");
      std::vector<uint8_t> staged(static_cast<size_t>(tokens) * t.rows_local() * row_bytes);
      t.tables[0]->gather(ids.data(), ids_stride, t.sel.data(), t.rows_local(), tokens, staged.data());
      for (int tk = 0; tk < tokens; ++tk)
        for (int j = 0; j < t.rows_local(); ++j) {
          const int32_t id = ids[static_cast<size_t>(tk) * ids_stride + t.sel[static_cast<size_t>(j)]];
          const uint8_t* got = staged.data() + (static_cast<size_t>(tk) * t.rows_local() + j) * row_bytes;
          require(std::memcmp(got, payload.data() + static_cast<size_t>(id) * cfg.engram_head_dim, cfg.engram_head_dim) == 0,
                  "gathered payload row");
          require(std::memcmp(got + cfg.engram_head_dim, scales.data() + static_cast<size_t>(id) * (cfg.engram_head_dim / 32),
                              cfg.engram_head_dim / 32) == 0,
                  "gathered scale row");
        }
      // The mappings outlive the checkpoint release (a decode step reads
      // them for the whole run), and a row past the table is refused.
      for (int l = 0; l < cfg.max_layer(); ++l) (void)s.load_layer(l);
      s.release_sources();
      std::vector<uint8_t> one(static_cast<size_t>(t.rows_local()) * row_bytes);
      t.tables[1]->gather(ids.data(), ids_stride, t.sel.data(), t.rows_local(), 1, one.data());
      bool refused = false;
      try {
        (void)t.tables[1]->payload_row(t.tables[1]->rows());
      } catch (const std::out_of_range&) {
        refused = true;
      }
      require(refused, "a row past the table is refused");
      require(t.mapped_bytes() > 0, "mapped bytes reported");
    }
  }
}

DGPP_TEST(dsv41_loader_resident_mode_and_image_round_trip) {
  const Fixture fx = write_fixture();
  const fs::path cache = fs::current_path() / "dsv41_loader_image_cache";
  fs::remove_all(cache);
  const std::string saved = Dsv41LayerStream::resident_image_dir();
  Dsv41LayerStream::set_resident_image_dir(cache.string());
  const int layers = fx.cfg.max_layer();
  std::vector<std::vector<uint8_t>> built(static_cast<size_t>(layers));
  {
    Dsv41LayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::Dsv41Residency::Resident, dgpp::Dsv41HeadSharding::VocabSharded,
                       /*resident_mtp=*/true);
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      check_layer(fx, s, r);
      const auto span = s.resident_layer_span(l);
      require(span.first != nullptr && span.second == r.bytes, "resident span");
      built[static_cast<size_t>(l)] = device_bytes(span.first, span.second);
    }
    (void)s.load_globals();
    require(s.image_layers_captured() == layers, "every layer captured");
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) (void)s.load_layer(l);
    require(s.source_bytes_read() == before, "cache hits read no storage");
    require(Dsv41LayerStream::resident_bytes(fx.cfg, 1, 2, dgpp::Dsv41HeadSharding::VocabSharded, true) ==
                [&] { size_t t = s.load_globals().bytes; for (int l = 0; l < layers; ++l) t += s.load_layer(l).bytes; return t; }(),
            "resident bytes formula");
    (void)s.load_engram_tables();
    s.release_sources();
    require(s.sources_released() && s.staging_released(), "the mappings and the staging mirror go together");
  }
  {
    Dsv41LayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::Dsv41Residency::Resident, dgpp::Dsv41HeadSharding::VocabSharded,
                       /*resident_mtp=*/true);
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      const auto span = s.resident_layer_span(l);
      require(device_bytes(span.first, span.second) == built[static_cast<size_t>(l)], "restored layer bitwise the built one");
      check_layer(fx, s, r);
    }
    require(s.image_layers_restored() == layers && s.source_bytes_read() == before,
            "every layer restored from the image, no source reads");
    const dgpp::Dsv41ReplicatedDigest d = s.hash_replicated();
    require(d.layer.size() == static_cast<size_t>(layers) && d.tensors > 0, "digest shape");
    // The digest is rank-invariant: rank 0's equals rank 1's.
    Dsv41LayerStream s0(fx.cfg, fx.dir, 0, 2, dgpp::Dsv41Residency::Streaming, dgpp::Dsv41HeadSharding::VocabSharded, true);
    const dgpp::Dsv41ReplicatedDigest d0 = s0.hash_replicated();
    require(d0.globals == d.globals && d0.layer == d.layer && d0.bytes == d.bytes, "the replicated digest is rank-invariant");
  }
  Dsv41LayerStream::set_resident_image_dir(saved);
  fs::remove_all(cache);
}

DGPP_TEST(dsv41_loader_refuses_bad_geometry_and_binding_by_name) {
  const Fixture fx = write_fixture();
  {
    bool refused = false;
    try {
      Dsv41LayerStream s(fx.cfg, fx.dir, 0, 3, dgpp::Dsv41Residency::Streaming, dgpp::Dsv41HeadSharding::VocabSharded);
    } catch (const std::invalid_argument& e) {
      refused = std::string(e.what()).find("world") != std::string::npos;
    }
    require(refused, "a world that does not divide the heads is refused by name");
  }
  {
    Dsv41TextConfig cfg = fx.cfg;
    cfg.n_routed_experts = 16;  // experts 8..15 are not in the shard
    bool refused = false;
    try {
      Dsv41LayerStream s(cfg, fx.dir, 0, 1);
    } catch (const std::runtime_error& e) {
      refused = std::string(e.what()).find("binding failed") != std::string::npos;
    }
    require(refused, "a checkpoint that does not bind is refused by name");
  }
  {
    // The Engram tables must be asked for before the release.
    Dsv41LayerStream s(fx.cfg, fx.dir, 0, 1, dgpp::Dsv41Residency::Resident);
    for (int l = 0; l < fx.cfg.max_layer(); ++l) (void)s.load_layer(l);
    s.release_sources();
    bool refused = false;
    try {
      (void)s.load_engram_tables();
    } catch (const std::runtime_error& e) {
      refused = std::string(e.what()).find("released") != std::string::npos;
    }
    require(refused, "tables after the release are refused by name");
  }
}

namespace {
void write_text(const fs::path& p, const std::string& text) {
  std::FILE* f = std::fopen(p.c_str(), "wb");
  require(f != nullptr, "open " + p.string());
  std::fwrite(text.data(), 1, text.size(), f);
  std::fclose(f);
}
}  // namespace

DGPP_TEST(dsv41_loader_engram_sidecar_loads_and_refuses_by_name) {
  const Fixture fx = write_fixture();
  const fs::path p = fs::path(fx.dir) / dgpp::kDsv41EngramSidecarName;
  write_text(p, dsv41fx::sidecar_json(fx.cfg, false));
  const dgpp::Dsv41EngramSidecar sc = dgpp::dsv41_load_engram_sidecar_for(fx.dir, fx.cfg);
  require(sc.layers() == 2 && sc.ngrams() == 3 && sc.n_heads == 4 && sc.token_map.size() == 512 && sc.pad_class == 2,
          "sidecar geometry");
  require(sc.prime(0, 0, 0) == 79 && sc.offset(0, 0, 1) == 79 && sc.prime(1, 2, 3) == 197 && sc.offset(1, 2, 3) == 2040 - 197,
          "sidecar indexing");
  write_text(p, dsv41fx::sidecar_json(fx.cfg, true));
  bool refused = false;
  try {
    (void)dgpp::dsv41_load_engram_sidecar_for(fx.dir, fx.cfg);
  } catch (const std::runtime_error& e) {
    refused = std::string(e.what()).find("offsets") != std::string::npos;
  }
  require(refused, "offsets off the running sum are refused by name");
  fs::remove(p);
  refused = false;
  try {
    (void)dgpp::dsv41_load_engram_sidecar_for(fx.dir, fx.cfg);
  } catch (const std::runtime_error& e) {
    refused = std::string(e.what()).find("sidecar missing") != std::string::npos;
  }
  require(refused, "a missing sidecar names the generator");
}

int main() { return dgpp::test::run_all(); }

// ---- the EXL3 pack (2026-10-22, docs/deepseek_v41_flash_exl3_plan.md §4-5) ----
// The tiny EXL3 pack: every matrix an EXL3 quadruple re-tiled n-tile-major
// into its slice — wq_b's head rows, wo_a one matrix per local group, wo_b's
// group columns, the experts' and the shared expert's 128-block slices of
// the intermediate (2/1/1/1 blocks of 640 at world 4, 3/2 at world 2), the
// Engram wkv's K ranges of the rank's hash heads, main_proj and the
// compressors / indexer whole, the head's 128-aligned vocabulary rows —
// with their suh / svh ranges; the pack's F16 natives as bf16 (checked)
// or f32; the byte formulas; the resident image; the tables admitted from
// a second directory; the refusals by name.
namespace {

std::vector<uint8_t> f16_as_bf16(const std::vector<uint8_t>& f16) {
  std::vector<uint8_t> out(f16.size());
  for (size_t i = 0; i < f16.size() / 2; ++i) {
    uint16_t h;
    std::memcpy(&h, f16.data() + i * 2, 2);
    const uint16_t b = dgpp::float_to_bf16_bits(dgpp::fp16_bits_to_float(h));
    std::memcpy(out.data() + i * 2, &b, 2);
  }
  return out;
}
std::vector<uint8_t> f16_as_f32(const std::vector<uint8_t>& f16) {
  std::vector<uint8_t> out(f16.size() * 2);
  for (size_t i = 0; i < f16.size() / 2; ++i) {
    uint16_t h;
    std::memcpy(&h, f16.data() + i * 2, 2);
    const float v = dgpp::fp16_bits_to_float(h);
    std::memcpy(out.data() + i * 4, &v, 4);
  }
  return out;
}

Fixture write_exl3_fixture(const char* name = "dsv41_loader_exl3_fixture") {
  Fixture fx;
  fx.cfg = dsv41fx::tiny_exl3_config();
  fx.dir = (fs::current_path() / name).string();
  dsv41fx::write_fixture(fx.cfg, fx.dir);
  fx.table = dgpp::dsv41_expected_text_tensors(fx.cfg);
  return fx;
}

// An EXL3 matrix against the checkpoint: output rows [row0, +rows) and the
// packed input column ranges (empty: every column), re-tiled as the loader
// re-tiles (exl3_retile over the k-tile ranges), with the suh / svh ranges.
void check_exl3(const Fixture& fx, const dgpp::Exl3Matrix& m, const std::string& base, int64_t row0, int64_t rows,
                const Ranges& ranges_in, const std::string& tag) {
  const auto& et = fx.expected(base + ".trellis");
  const int64_t K = et.shape[0] * 16, N = et.shape[1] * 16;
  const int bits = static_cast<int>(et.shape[2] / 16);
  Ranges ranges = ranges_in;
  if (ranges.empty()) ranges.emplace_back(0, K);
  int64_t cols = 0;
  for (const auto& [start, count] : ranges) cols += count;
  require(m.rows == rows && m.cols == cols && m.bits == bits, tag + " geometry");
  const std::vector<uint8_t> trellis = fx.bytes(base + ".trellis");
  const size_t tb = dgpp::exl3_tile_bytes(bits);
  std::vector<uint8_t> want_tiles(static_cast<size_t>(rows / 16) * static_cast<size_t>(cols / 16) * tb);
  {
    size_t at = 0;
    for (int64_t nt = row0 / 16; nt < (row0 + rows) / 16; ++nt)
      for (const auto& [start, count] : ranges)
        for (int64_t kt = start / 16; kt < (start + count) / 16; ++kt, ++at)
          std::memcpy(want_tiles.data() + at * tb,
                      trellis.data() + (static_cast<size_t>(kt) * static_cast<size_t>(N / 16) + static_cast<size_t>(nt)) * tb, tb);
  }
  std::vector<uint8_t> want_suh;
  const std::vector<uint8_t> suh = fx.bytes(base + ".suh");
  for (const auto& [start, count] : ranges)
    want_suh.insert(want_suh.end(), suh.begin() + start * 2, suh.begin() + (start + count) * 2);
  expect_device_equals(m.tiles, want_tiles, tag + " tiles");
  expect_device_equals(m.suh, want_suh, tag + " suh");
  expect_device_equals(m.svh, host_rows(fx.bytes(base + ".svh"), 2, row0, rows), tag + " svh");
}
void check_exl3_full(const Fixture& fx, const dgpp::Exl3Matrix& m, const std::string& base, const std::string& tag) {
  check_exl3(fx, m, base, 0, fx.expected(base + ".trellis").shape[1] * 16, {}, tag);
}

void check_layer_exl3(const Fixture& fx, const Dsv41LayerStream& s, const dgpp::Dsv41LayerResident& r) {
  const Dsv41TextConfig& cfg = fx.cfg;
  const dgpp::Dsv41LocalGeometry& g = s.geometry();
  const int world = s.world(), rank = s.rank();
  const std::string p = dgpp::dsv41_layer_prefix(cfg, r.layer);
  const std::string tag = "exl3 world " + std::to_string(world) + " rank " + std::to_string(rank) + " layer " +
                          std::to_string(r.layer) + " ";
  const bool draft = cfg.is_draft(r.layer);
  require(r.attn.exl3() && r.moe.exl3(), tag + "the EXL3 views are bound");
  require(r.attn.wq_a.payload == nullptr && r.moe.experts.empty() && r.moe.shared[0].payload == nullptr,
          tag + "the release-format views stay unset");
  // Attention.
  const std::string ap = p + "attn.";
  const int64_t hd = cfg.head_dim, ol = cfg.o_lora_rank;
  check_exl3_full(fx, r.attn.wq_a_x, ap + "wq_a", tag + "wq_a");
  check_exl3_full(fx, r.attn.wkv_x, ap + "wkv", tag + "wkv");
  check_exl3(fx, r.attn.wq_b_x, ap + "wq_b", g.head_begin * hd, g.local_heads * hd, {}, tag + "wq_b");
  require(static_cast<int>(r.attn.wo_a_x.size()) == g.local_groups, tag + "wo_a one matrix per local group");
  for (int gi = 0; gi < g.local_groups; ++gi)
    check_exl3_full(fx, r.attn.wo_a_x[static_cast<size_t>(gi)], ap + "wo_a.slice." + std::to_string(g.group_begin + gi),
                    tag + "wo_a group " + std::to_string(gi));
  check_exl3(fx, r.attn.wo_b_x, ap + "wo_b", 0, cfg.hidden_size, {{g.group_begin * ol, g.local_groups * ol}}, tag + "wo_b");
  if (cfg.is_index_source(r.layer)) {
    require(r.attn.index_source(), tag + "indexer present");
    check_exl3_full(fx, r.attn.idx_wq_b_x, ap + "indexer.wq_b", tag + "indexer wq_b");
    expect_device_equals(r.attn.idx_wp, f16_as_bf16(fx.bytes(ap + "indexer.weights_proj.weight")), tag + "weights_proj as bf16");
  } else {
    require(!r.attn.index_source() && r.attn.idx_wp == nullptr, tag + "no indexer");
  }
  if (cfg.is_kv_source(r.layer)) {
    require(r.attn.kv_source(), tag + "kv source present");
    check_exl3_full(fx, r.attn.idx_wk_x, ap + "indexer.wk", tag + "index keys wk");
    require(r.attn.idx_wk_x.bits == cfg.exl3.indexer_wk, tag + "wk at the indexer width");
    expect_device_equals(r.attn.idx_k_norm, fx.bytes(ap + "indexer.k_norm.weight"), tag + "index keys norm");
    check_exl3_full(fx, r.attn.comp_wkv_x, ap + "compressor.wkv", tag + "compressor wkv");
    expect_device_equals(r.attn.comp_norm, fx.bytes(ap + "compressor.norm.weight"), tag + "compressor norm");
    if (cfg.compress_ratio(r.layer) > 1)
      check_exl3_full(fx, r.attn.comp_wgate_x, ap + "compressor.wgate", tag + "compressor wgate");
    else
      require(r.attn.comp_wgate_x.tiles == nullptr, tag + "no gate at ratio 1");
  } else {
    require(!r.attn.kv_source() && r.attn.idx_wk_x.tiles == nullptr, tag + "no compressor");
  }
  // MoE: the F16 router as bf16, the F16 bias as f32, the 128-block slices.
  const std::string mp = p + "ffn.";
  expect_device_equals(r.moe.router, f16_as_bf16(fx.bytes(mp + "gate.weight")), tag + "router as bf16");
  expect_device_equals(r.moe.router_bias, f16_as_f32(fx.bytes(mp + "gate.bias")), tag + "router bias as f32");
  // The geometry: the even slice's begin, the widest slice's width; each
  // expert's own slice alternates by parity (exl3_expert_block_split), the
  // shared expert's by the layer's.
  int64_t I0 = 0, I = 0, S0 = 0, S = 0;
  dgpp::exl3_block_split(cfg.moe_intermediate_size, world, rank, I0, I);
  dgpp::exl3_block_split(cfg.shared_expert_inter(), world, rank, S0, S);
  const int64_t Imax = dgpp::exl3_block_split_max(cfg.moe_intermediate_size, world);
  const int64_t Smax = dgpp::exl3_block_split_max(cfg.shared_expert_inter(), world);
  require(g.inter_begin == I0 && g.local_inter == Imax && g.shared_inter_begin == S0 && g.local_shared_inter == Smax,
          tag + "the block-split geometry");
  require(r.moe.local_inter == Imax && r.moe.local_shared_inter == Smax, tag + "moe inter");
  const int E = draft ? cfg.dspark_n_routed_experts : cfg.n_routed_experts;
  require(r.moe.n_experts == E && static_cast<int>(r.moe.experts_x.size()) == E * 3, tag + "expert count");
  const int rb = cfg.exl3_bits_routed(r.layer), sb = cfg.exl3_bits_shared(r.layer);
  for (int e = 0; e < E; ++e) {
    const std::string ep = mp + "experts." + std::to_string(e) + ".";
    const dgpp::Exl3Matrix* t = r.moe.experts_x.data() + static_cast<size_t>(e) * 3;
    int64_t e0 = 0, en = 0;
    dgpp::exl3_expert_block_split(cfg.moe_intermediate_size, world, rank, e & 1, e0, en);
    check_exl3(fx, t[0], ep + "w1", e0, en, {}, tag + "expert gate");
    check_exl3(fx, t[1], ep + "w3", e0, en, {}, tag + "expert up");
    check_exl3(fx, t[2], ep + "w2", 0, cfg.hidden_size, {{e0, en}}, tag + "expert down");
    require(t[0].bits == rb && t[2].bits == rb, tag + "expert width");
  }
  int64_t s0 = 0, sn = 0;
  dgpp::exl3_expert_block_split(cfg.shared_expert_inter(), world, rank, r.layer & 1, s0, sn);
  const std::string sp = mp + "shared_experts.";
  check_exl3(fx, r.moe.shared_x[0], sp + "w1", s0, sn, {}, tag + "shared gate");
  check_exl3(fx, r.moe.shared_x[1], sp + "w3", s0, sn, {}, tag + "shared up");
  check_exl3(fx, r.moe.shared_x[2], sp + "w2", 0, cfg.hidden_size, {{s0, sn}}, tag + "shared down");
  require(r.moe.shared_x[0].bits == sb, tag + "shared width");
  // Engram: the wkv's K ranges, the F32 q / k weights as bf16.
  if (cfg.has_engram(r.layer)) {
    require(r.engram.present, tag + "engram present");
    Ranges ranges;
    const int64_t ehd = cfg.engram_head_dim;
    for (int n = 0; n < cfg.engram_max_ngram_size - 1; ++n)
      ranges.emplace_back((static_cast<int64_t>(n) * cfg.engram_n_heads + g.engram_head_begin) * ehd,
                          static_cast<int64_t>(g.engram_heads_local) * ehd);
    check_exl3(fx, r.engram.wkv_x, p + "engram.wkv", 0, static_cast<int64_t>(cfg.hidden_size) * (cfg.hc_mult + 1), ranges,
               tag + "engram wkv");
    expect_device_equals(r.engram.q_weight, f32_as_bf16(fx.bytes(p + "engram.q_weight")), tag + "engram q as bf16");
    expect_device_equals(r.engram.k_weight, f32_as_bf16(fx.bytes(p + "engram.k_weight")), tag + "engram k as bf16");
  } else {
    require(!r.engram.present && r.engram.wkv_x.tiles == nullptr, tag + "no engram");
  }
  // Draft: main_proj EXL3 at the mtp width, the F16 Markov head and confidence.
  if (draft) {
    const int stage = cfg.draft_stage(r.layer);
    if (stage == 0) {
      check_exl3_full(fx, r.draft.main_proj_x, p + "main_proj", tag + "main_proj");
      require(r.draft.main_proj_x.bits == cfg.exl3.mtp, tag + "main_proj at the mtp width");
    } else {
      require(r.draft.main_proj_x.tiles == nullptr, tag + "no main_proj past stage 0");
    }
    if (stage == cfg.num_nextn_predict_layers - 1) {
      expect_device_equals(r.draft.markov_head, f16_as_bf16(fx.bytes(p + "markov_head.head.weight")), tag + "markov head as bf16");
      expect_device_equals(r.draft.confidence, f16_as_f32(fx.bytes(p + "confidence_head.proj.weight")), tag + "confidence as f32");
    }
  }
  require(r.bytes == Dsv41LayerStream::layer_bytes(cfg, r.layer, rank, world), tag + "layer bytes formula");
}

// Patches one f16 element of a tensor in the fixture's shard.
void patch_f16(const std::string& dir, const std::string& name, size_t index, uint16_t bits) {
  auto f = dgpp::SafetensorsFile::open((fs::path(dir) / "model.safetensors").string());
  uint64_t at = 0;
  bool found = false;
  f->for_each([&](const dgpp::TensorInfo& t) {
    if (t.name == name) {
      at = t.data_begin + index * 2;
      found = true;
    }
  });
  require(found, "patch: tensor present");
  f.reset();
  std::fstream io((fs::path(dir) / "model.safetensors").string(), std::ios::in | std::ios::out | std::ios::binary);
  io.seekp(static_cast<std::streamoff>(at));
  io.write(reinterpret_cast<const char*>(&bits), 2);
}

}  // namespace

DGPP_TEST(dsv41_loader_exl3_slices_every_class_byte_exact_at_worlds_1_2_4) {
  const Fixture fx = write_exl3_fixture();
  require(fx.cfg.weights_exl3, "the fixture is the EXL3 pack");
  const int layers = fx.cfg.max_layer();
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::Dsv41HeadSharding::VocabSharded : dgpp::Dsv41HeadSharding::Full;
      Dsv41LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::Dsv41Residency::Streaming, head, /*resident_mtp=*/true);
      for (int l = 0; l < layers; ++l) {
        const uint64_t before = s.source_bytes_read();
        const auto& r = s.load_layer(l);
        check_layer_exl3(fx, s, r);
        require(s.source_bytes_read() - before == Dsv41LayerStream::planned_layer_source_bytes(fx.cfg, l, rank, world),
                "exl3: the source-byte plan equals the bytes read");
        s.release_layer();
      }
    }
  }
}

DGPP_TEST(dsv41_loader_exl3_globals_head_rows_on_blocks) {
  const Fixture fx = write_exl3_fixture();
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      const auto head = world > 1 ? dgpp::Dsv41HeadSharding::VocabSharded : dgpp::Dsv41HeadSharding::Full;
      Dsv41LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::Dsv41Residency::Streaming, head);
      const auto& g = s.load_globals();
      expect_device_equals(g.embed, fx.bytes("embed.weight"), "exl3 embed");
      expect_device_equals(g.final_norm, fx.bytes("norm.weight"), "exl3 final norm");
      require(g.lm_head == nullptr && g.lm_head_x.tiles != nullptr, "exl3: the head is the EXL3 view");
      int64_t b = 0, n = 0;
      if (world > 1) dgpp::exl3_block_split(fx.cfg.vocab_size, world, rank, b, n);
      else n = fx.cfg.vocab_size;
      require(g.lm_vocab_begin == b && g.lm_vocab_count == n, "exl3 head rows on 128-blocks");
      check_exl3(fx, g.lm_head_x, "head", b, n, {}, "exl3 head slice");
      require(g.bytes == Dsv41LayerStream::globals_bytes(fx.cfg, rank, world, head), "exl3 globals formula");
    }
  }
}

DGPP_TEST(dsv41_loader_exl3_resident_image_round_trip) {
  const Fixture fx = write_exl3_fixture();
  const fs::path cache = fs::current_path() / "dsv41_loader_exl3_image_cache";
  fs::remove_all(cache);
  const std::string saved = Dsv41LayerStream::resident_image_dir();
  Dsv41LayerStream::set_resident_image_dir(cache.string());
  const int layers = fx.cfg.max_layer();
  std::vector<std::vector<uint8_t>> built(static_cast<size_t>(layers));
  {
    Dsv41LayerStream s(fx.cfg, fx.dir, 3, 4, dgpp::Dsv41Residency::Resident, dgpp::Dsv41HeadSharding::VocabSharded,
                       /*resident_mtp=*/true);
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      check_layer_exl3(fx, s, r);
      const auto span = s.resident_layer_span(l);
      built[static_cast<size_t>(l)] = device_bytes(span.first, span.second);
    }
    (void)s.load_globals();
    require(s.image_layers_captured() == layers, "exl3: every layer captured");
  }
  {
    Dsv41LayerStream s(fx.cfg, fx.dir, 3, 4, dgpp::Dsv41Residency::Resident, dgpp::Dsv41HeadSharding::VocabSharded,
                       /*resident_mtp=*/true);
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      const auto span = s.resident_layer_span(l);
      require(device_bytes(span.first, span.second) == built[static_cast<size_t>(l)], "exl3: the image restores the bytes");
      check_layer_exl3(fx, s, r);
    }
    require(s.image_layers_restored() == layers, "exl3: every layer restored");
  }
  Dsv41LayerStream::set_resident_image_dir(saved);
  fs::remove_all(cache);
}

DGPP_TEST(dsv41_loader_exl3_engram_tables_from_the_source_directory) {
  // The pack's shape: no tables in the checkpoint, the tables alone in a
  // second directory named as the Engram source.
  Fixture fx;
  fx.cfg = dsv41fx::tiny_exl3_config();
  fx.dir = (fs::current_path() / "dsv41_loader_exl3_split_fixture").string();
  const std::string tables = (fs::current_path() / "dsv41_loader_exl3_split_tables").string();
  dsv41fx::write_fixture_split(fx.cfg, fx.dir, tables);
  fx.table = dgpp::dsv41_expected_text_tensors(fx.cfg);
  // Without the source the binding fails naming the tables.
  bool refused = false;
  try {
    Dsv41LayerStream s(fx.cfg, fx.dir, 0, 1, dgpp::Dsv41Residency::Streaming, dgpp::Dsv41HeadSharding::Full);
  } catch (const std::exception& e) {
    refused = std::string(e.what()).find("engram.embed") != std::string::npos;
  }
  require(refused, "exl3: a pack without the tables and no Engram source is refused naming them");
  Dsv41LayerStream::set_engram_source_dir(tables);
  for (const int world : {1, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      Dsv41LayerStream s(fx.cfg, fx.dir, rank, world, dgpp::Dsv41Residency::Resident,
                         world > 1 ? dgpp::Dsv41HeadSharding::VocabSharded : dgpp::Dsv41HeadSharding::Full);
      const dgpp::Dsv41EngramTables& t = s.load_engram_tables();
      require(t.tables.size() == fx.cfg.engram_layer_ids.size(), "exl3: the tables mapped from the source");
      const std::string p1 = dgpp::dsv41_layer_prefix(fx.cfg, fx.cfg.engram_layer_ids[0]) + "engram.embed.";
      const std::vector<uint8_t> payload = fx.bytes(p1 + "weight");
      require(std::memcmp(t.tables[0]->payload_row(7), payload.data() + 7 * fx.cfg.engram_head_dim, fx.cfg.engram_head_dim) == 0,
              "exl3: a table row read from the source directory");
      for (int l = 0; l < fx.cfg.max_layer(); ++l) check_layer_exl3(fx, s, s.load_layer(l));
    }
  }
  Dsv41LayerStream::set_engram_source_dir("");
}

DGPP_TEST(dsv41_loader_exl3_refuses_inexact_natives_and_a_pack_without_its_k_map) {
  Fixture fx = write_exl3_fixture("dsv41_loader_exl3_refusal_fixture");
  // The k-map beside config.json is what the file form reads.
  const Dsv41TextConfig from_file = Dsv41TextConfig::from_json_file((fs::path(fx.dir) / "config.json").string());
  require(from_file.weights_exl3 && from_file.exl3.routed == fx.cfg.exl3.routed && from_file.exl3.head == 6,
          "exl3: from_json_file reads the k-map");
  fs::remove(fs::path(fx.dir) / "exl3_k_map.json");
  bool refused = false;
  try {
    (void)Dsv41TextConfig::from_json_file((fs::path(fx.dir) / "config.json").string());
  } catch (const std::exception& e) {
    refused = std::string(e.what()).find("exl3_k_map") != std::string::npos;
  }
  require(refused, "exl3: a pack without its k-map is refused naming it");
  // An f16 router value that is not bf16-exact: the checked load refuses it by name.
  patch_f16(fx.dir, "layers.0.ffn.gate.weight", 5, 0x3C01);  // 1.0009765625: three more mantissa bits than bf16
  refused = false;
  try {
    Dsv41LayerStream s(fx.cfg, fx.dir, 0, 1, dgpp::Dsv41Residency::Streaming, dgpp::Dsv41HeadSharding::Full);
    (void)s.load_layer(0);
  } catch (const std::exception& e) {
    refused = std::string(e.what()).find("layers.0.ffn.gate.weight") != std::string::npos &&
              std::string(e.what()).find("bf16-exact") != std::string::npos;
  }
  require(refused, "exl3: an f16 native that is not bf16-exact is refused naming the tensor");
}
