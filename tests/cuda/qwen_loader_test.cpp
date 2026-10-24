// The Qwen3.8-Flash-Next resident loader on the synthetic
// fixture: every class lands byte-exact at worlds 1, 2 and 4 as the slice
// formulas of docs/qwen38_flash_next_plan.md §2.1 say (GDN segments, QSA
// heads and the kv pairing, the experts' inter slices on the re-blocked
// scale grid, the PLE's column slices, the replicated set verbatim); the
// byte formulas equal actual usage and the source-byte plan equals the
// bytes read; the globals and the n-gram table slice; resident cache hits
// and the image round trip; and a tampered hash buffer is refused.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/packq_head.hpp"
#include "loaders/packq_quant.hpp"
#include "models/qwen/binding.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/loader.hpp"
#include "qwen_fixture.hpp"
#include "resident_image_fixture.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::QwenExpectedTensor;
using dgpp::QwenLayerStream;
using dgpp::QwenTextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  QwenTextConfig cfg;
  std::string dir;
  std::vector<QwenExpectedTensor> table;
  const QwenExpectedTensor& expected(const std::string& name) const {
    for (const auto& e : table)
      if (e.name == name) return e;
    throw std::runtime_error("fixture: no expected tensor " + name);
  }
  std::vector<uint8_t> bytes(const std::string& name) const {
    return qwenfx::tensor_bytes(cfg, expected(name));
  }
};

Fixture write_fixture() {
  Fixture fx;
  fx.cfg = qwenfx::tiny_config();
  fx.dir = (fs::current_path() / "qwen_loader_fixture").string();
  qwenfx::write_fixture(fx.cfg, fx.dir);
  fx.table = dgpp::qwen_expected_text_tensors(fx.cfg);
  return fx;
}

Fixture write_nvfp4_fixture() {
  Fixture fx;
  fx.cfg = qwenfx::tiny_nvfp4_config();
  fx.dir = (fs::current_path() / "qwen_loader_nvfp4_fixture").string();
  qwenfx::write_fixture(fx.cfg, fx.dir, qwenfx::tiny_text_json(),
                        qwenfx::tiny_nvfp4_quant_json());
  fx.table = dgpp::qwen_expected_text_tensors(fx.cfg);
  return fx;
}

Fixture write_qad_fixture() {
  Fixture fx;
  fx.cfg = qwenfx::tiny_qad_config();
  fx.dir = (fs::current_path() / "qwen_loader_qad_fixture").string();
  qwenfx::write_fixture(fx.cfg, fx.dir, qwenfx::tiny_qad_text_json(), qwenfx::tiny_qad_quant_json(), {});
  fx.table = dgpp::qwen_expected_text_tensors(fx.cfg);
  return fx;
}

Fixture write_gptq_fixture() {
  Fixture fx;
  fx.cfg = qwenfx::tiny_gptq_config();
  fx.dir = (fs::current_path() / "qwen_loader_gptq_fixture").string();
  qwenfx::write_fixture(fx.cfg, fx.dir, qwenfx::tiny_gptq_text_json(),
                        qwenfx::tiny_gptq_quant_json());
  fx.table = dgpp::qwen_expected_text_tensors(fx.cfg);
  return fx;
}

// The packed-core words of the fixture's GPTQ triple `base` ([N, K] at
// `bits`): qweight I32 [K*bits/32, N] transposed to [N, K*bits/32].
std::vector<uint8_t> gptq_words_transposed(const Fixture& fx, const std::string& base, int64_t N,
                                           int64_t K, int bits) {
  const std::vector<uint8_t> src = fx.bytes(base + ".qweight");
  const int64_t wr = K * bits / 32;
  std::vector<uint32_t> want(static_cast<size_t>(N) * wr);
  for (int64_t i = 0; i < wr; ++i)
    for (int64_t n = 0; n < N; ++n)
      std::memcpy(&want[static_cast<size_t>(n) * wr + i], src.data() + (static_cast<size_t>(i) * N + n) * 4, 4);
  std::vector<uint8_t> out(want.size() * 4);
  std::memcpy(out.data(), want.data(), out.size());
  return out;
}
std::vector<uint8_t> gptq_scales_transposed(const Fixture& fx, const std::string& base, int64_t N,
                                            int64_t K, int group) {
  const std::vector<uint8_t> src = fx.bytes(base + ".scales");
  const int64_t gr = K / group;
  std::vector<uint8_t> out(static_cast<size_t>(N) * gr * 2);
  for (int64_t g = 0; g < gr; ++g)
    for (int64_t n = 0; n < N; ++n)
      std::memcpy(out.data() + (static_cast<size_t>(n) * gr + g) * 2, src.data() + (static_cast<size_t>(g) * N + n) * 2, 2);
  return out;
}
// GPTQ's own dequant of element (n, k): (code - 2^(bits-1)) x f16(scale).
float gptq_dequant(const Fixture& fx, const std::string& base, int64_t N, int64_t K, int bits, int group,
                   int64_t n, int64_t k) {
  (void)K;  // the word index needs only N (the GPTQ row stride)
  const std::vector<uint8_t> qw = fx.bytes(base + ".qweight");
  const std::vector<uint8_t> sc = fx.bytes(base + ".scales");
  const int per = 32 / bits;
  uint32_t word;
  std::memcpy(&word, qw.data() + (static_cast<size_t>(k / per) * N + n) * 4, 4);
  const int code = static_cast<int>((word >> (bits * (k % per))) & ((1u << bits) - 1u)) - (1 << (bits - 1));
  uint16_t h;
  std::memcpy(&h, sc.data() + (static_cast<size_t>(k / group) * N + n) * 2, 2);
  return static_cast<float>(code) * dgpp::fp16_bits_to_float(h);
}

std::vector<uint8_t> device_bytes(const void* dev, size_t n) {
  std::vector<uint8_t> h(n);
  DGPP_CUDA_OK(cudaMemcpy(h.data(), dev, n, cudaMemcpyDeviceToHost));
  return h;
}

// Rows [r0, +rows) of a [N, W-byte] host tensor.
std::vector<uint8_t> host_rows(const std::vector<uint8_t>& t, size_t row_bytes, int64_t r0, int64_t rows) {
  return std::vector<uint8_t>(t.begin() + r0 * row_bytes, t.begin() + (r0 + rows) * row_bytes);
}
// Columns [c0, +w) bytes of every row of a [rows, full] host matrix, packed.
std::vector<uint8_t> host_cols(const std::vector<uint8_t>& t, int64_t rows, size_t full, size_t c0, size_t w) {
  std::vector<uint8_t> out;
  out.reserve(rows * w);
  for (int64_t r = 0; r < rows; ++r)
    out.insert(out.end(), t.begin() + r * full + c0, t.begin() + r * full + c0 + w);
  return out;
}
std::vector<uint8_t> widened_f32(const std::vector<uint8_t>& bf16, int64_t start, int64_t count) {
  std::vector<uint8_t> out(count * 4);
  for (int64_t i = 0; i < count; ++i) {
    uint16_t bits;
    std::memcpy(&bits, bf16.data() + (start + i) * 2, 2);
    const float f = dgpp::bf16_bits_to_float(bits);
    std::memcpy(out.data() + i * 4, &f, 4);
  }
  return out;
}
void expect_device_equals(const void* dev, const std::vector<uint8_t>& want, const std::string& what) {
  const std::vector<uint8_t> got = device_bytes(dev, want.size());
  require(got == want, what + ": resident bytes differ from the checkpoint slice");
}

// The expected re-blocked F32 scale grid of a sliced quantized matrix.
std::vector<uint8_t> reblocked_scales(const std::vector<uint8_t>& src_bf16, int64_t src_cols,
                                      bool row_slice, int64_t start, int64_t count, int sb) {
  std::vector<uint8_t> out;
  auto at = [&](int64_t r, int64_t c) {
    uint16_t bits;
    std::memcpy(&bits, src_bf16.data() + (r * src_cols + c) * 2, 2);
    const float f = dgpp::bf16_bits_to_float(bits);
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&f);
    out.insert(out.end(), p, p + 4);
  };
  const int64_t src_rows = static_cast<int64_t>(src_bf16.size() / 2) / src_cols;
  if (row_slice) {
    const int64_t n = (count + sb - 1) / sb;
    for (int64_t i = 0; i < n; ++i)
      for (int64_t c = 0; c < src_cols; ++c) at((start + i * sb) / 128, c);
  } else {
    const int64_t n = (count + sb - 1) / sb;
    for (int64_t r = 0; r < src_rows; ++r)
      for (int64_t j = 0; j < n; ++j) at(r, (start + j * sb) / 128);
  }
  return out;
}

void check_layer(const Fixture& fx, const QwenLayerStream& s, const dgpp::QwenLayerResident& r) {
  const QwenTextConfig& cfg = fx.cfg;
  const dgpp::QwenLocalGeometry& g = s.geometry();
  const int world = s.world(), rank = s.rank();
  const std::string p = dgpp::qwen_layer_prefix(cfg, r.layer);
  const size_t H2 = static_cast<size_t>(cfg.hidden_size) * 2;
  const std::string tag = "world " + std::to_string(world) + " rank " + std::to_string(rank) +
                          " layer " + std::to_string(r.layer) + " ";
  // GR sites: verbatim.
  for (const auto& [gp, gr] : {std::pair{p + "attn_hyper_connection.", &r.attn_gr},
                               std::pair{p + "mlp_hyper_connection.", &r.mlp_gr}}) {
    expect_device_equals(gr->hc_norm, fx.bytes(gp + "hc_norm.weight"), tag + "hc_norm");
    expect_device_equals(gr->down, fx.bytes(gp + "input_mix_weight_down.weight"), tag + "gr down");
    expect_device_equals(gr->up, fx.bytes(gp + "input_mix_weight_up.weight"), tag + "gr up");
    expect_device_equals(gr->inject, fx.bytes(gp + "block_inject_weight.weight"), tag + "gr inject");
  }
  if (r.kind == dgpp::QwenLayerKind::Gdn) {
    const std::string gp = p + "linear_attn.";
    const int64_t dk = cfg.gdn_key_head_dim, dv = cfg.gdn_value_head_dim;
    const int64_t K = static_cast<int64_t>(cfg.gdn_key_heads) * dk;
    const int64_t lk = g.local_key_heads, lv = g.local_value_heads;
    const std::vector<uint8_t> qkv = fx.bytes(gp + "in_proj_qkv.weight");
    std::vector<uint8_t> want = host_rows(qkv, H2, rank * lk * dk, lk * dk);
    auto k = host_rows(qkv, H2, K + rank * lk * dk, lk * dk);
    auto v = host_rows(qkv, H2, 2 * K + rank * lv * dv, lv * dv);
    want.insert(want.end(), k.begin(), k.end());
    want.insert(want.end(), v.begin(), v.end());
    expect_device_equals(r.gdn.in_proj_qkv, want, tag + "gdn qkv segments");
    const size_t cw = static_cast<size_t>(cfg.gdn_conv_width) * 2;
    const std::vector<uint8_t> conv = fx.bytes(gp + "conv1d.weight");
    want = host_rows(conv, cw, rank * lk * dk, lk * dk);
    k = host_rows(conv, cw, K + rank * lk * dk, lk * dk);
    v = host_rows(conv, cw, 2 * K + rank * lv * dv, lv * dv);
    want.insert(want.end(), k.begin(), k.end());
    want.insert(want.end(), v.begin(), v.end());
    expect_device_equals(r.gdn.conv, want, tag + "gdn conv segments");
    expect_device_equals(r.gdn.in_proj_z, host_rows(fx.bytes(gp + "in_proj_z.weight"), H2, rank * lv * dv, lv * dv), tag + "gdn z");
    expect_device_equals(r.gdn.in_proj_a, host_rows(fx.bytes(gp + "in_proj_a.weight"), H2, rank * lv, lv), tag + "gdn a");
    expect_device_equals(r.gdn.in_proj_b, host_rows(fx.bytes(gp + "in_proj_b.weight"), H2, rank * lv, lv), tag + "gdn b");
    expect_device_equals(r.gdn.a_log, widened_f32(fx.bytes(gp + "A_log"), rank * lv, lv), tag + "gdn A_log");
    expect_device_equals(r.gdn.dt_bias, widened_f32(fx.bytes(gp + "dt_bias"), rank * lv, lv), tag + "gdn dt_bias");
    expect_device_equals(r.gdn.norm, fx.bytes(gp + "norm.weight"), tag + "gdn norm");
    const int64_t V = static_cast<int64_t>(cfg.gdn_value_heads) * dv;
    expect_device_equals(r.gdn.out_proj, host_cols(fx.bytes(gp + "out_proj.weight"), cfg.hidden_size, V * 2, rank * lv * dv * 2, lv * dv * 2), tag + "gdn out_proj cols");
    require(r.gdn.local_key_heads == lk && r.gdn.local_value_heads == lv, tag + "gdn head counts");
  } else {
    const std::string ap = p + "self_attn.";
    const int64_t d = cfg.head_dim;
    expect_device_equals(r.qsa.q_proj, host_rows(fx.bytes(ap + "q_proj.weight"), H2, g.head_begin * 2 * d, g.local_heads * 2 * d), tag + "qsa q rows");
    expect_device_equals(r.qsa.k_proj, host_rows(fx.bytes(ap + "k_proj.weight"), H2, g.kv_head_begin * d, g.local_kv_heads * d), tag + "qsa k rows");
    expect_device_equals(r.qsa.v_proj, host_rows(fx.bytes(ap + "v_proj.weight"), H2, g.kv_head_begin * d, g.local_kv_heads * d), tag + "qsa v rows");
    const int64_t Q = static_cast<int64_t>(cfg.num_attention_heads) * d;
    expect_device_equals(r.qsa.o_proj, host_cols(fx.bytes(ap + "o_proj.weight"), cfg.hidden_size, Q * 2, g.head_begin * d * 2, g.local_heads * d * 2), tag + "qsa o cols");
    expect_device_equals(r.qsa.q_norm, fx.bytes(ap + "q_norm.weight"), tag + "q_norm");
    expect_device_equals(r.qsa.k_norm, fx.bytes(ap + "k_norm.weight"), tag + "k_norm");
    expect_device_equals(r.qsa.index_qk_proj, fx.bytes(ap + "indexer.index_qk_proj.weight"), tag + "indexer proj");
    expect_device_equals(r.qsa.index_q_norm, fx.bytes(ap + "indexer.q_layernorm.weight"), tag + "indexer q norm");
    expect_device_equals(r.qsa.index_k_norm, fx.bytes(ap + "indexer.k_layernorm.weight"), tag + "indexer k norm");
    // The kv pairing: at world 4 ranks 0,1 share head 0 and 2,3 head 1.
    if (world > cfg.num_key_value_heads)
      require(r.qsa.kv_head_begin == rank / (world / cfg.num_key_value_heads), tag + "kv pairing");
  }
  // MoE: router and shared gate verbatim; shared expert BF16 slices; experts
  // e4m3 slices with the re-blocked scale grid.
  const std::string mp = p + "mlp.";
  expect_device_equals(r.moe.router, fx.bytes(mp + "gate.weight"), tag + "router");
  expect_device_equals(r.moe.shared_gate, fx.bytes(mp + "shared_expert_gate.weight"), tag + "shared gate");
  const int64_t S = g.local_shared_inter, I = g.local_inter;
  expect_device_equals(r.moe.shared[0], host_rows(fx.bytes(mp + "shared_expert.gate_proj.weight"), H2, rank * S, S), tag + "shared gate_proj rows");
  expect_device_equals(r.moe.shared[1], host_rows(fx.bytes(mp + "shared_expert.up_proj.weight"), H2, rank * S, S), tag + "shared up rows");
  expect_device_equals(r.moe.shared[2], host_cols(fx.bytes(mp + "shared_expert.down_proj.weight"), cfg.hidden_size, cfg.shared_expert_intermediate_size * 2, rank * S * 2, S * 2), tag + "shared down cols");
  require(r.moe.scale_block == std::gcd(128, static_cast<int>(I)), tag + "scale block");
  const int64_t sb_cols = (cfg.hidden_size + 127) / 128;
  for (int e = 0; e < cfg.num_experts; ++e) {
    const std::string ep = mp + "experts." + std::to_string(e) + ".";
    const dgpp::GlmQuantMatrix& gate = r.moe.expert(e, 0);
    const dgpp::GlmQuantMatrix& up = r.moe.expert(e, 1);
    const dgpp::GlmQuantMatrix& down = r.moe.expert(e, 2);
    require(gate.rows == I && gate.cols == cfg.hidden_size && gate.scale_block_rows == r.moe.scale_block && gate.scale_block_cols == 128, tag + "gate geometry");
    require(down.rows == cfg.hidden_size && down.cols == I && down.scale_block_cols == r.moe.scale_block && down.scale_block_rows == 128, tag + "down geometry");
    expect_device_equals(gate.payload, host_rows(fx.bytes(ep + "gate_proj.weight"), cfg.hidden_size, rank * I, I), tag + "expert gate payload");
    expect_device_equals(up.payload, host_rows(fx.bytes(ep + "up_proj.weight"), cfg.hidden_size, rank * I, I), tag + "expert up payload");
    expect_device_equals(down.payload, host_cols(fx.bytes(ep + "down_proj.weight"), cfg.hidden_size, cfg.moe_intermediate_size, rank * I, I), tag + "expert down payload");
    expect_device_equals(gate.scales, reblocked_scales(fx.bytes(ep + "gate_proj.weight_scale_inv"), sb_cols, true, rank * I, I, r.moe.scale_block), tag + "expert gate scales");
    expect_device_equals(up.scales, reblocked_scales(fx.bytes(ep + "up_proj.weight_scale_inv"), sb_cols, true, rank * I, I, r.moe.scale_block), tag + "expert up scales");
    expect_device_equals(down.scales, reblocked_scales(fx.bytes(ep + "down_proj.weight_scale_inv"), (cfg.moe_intermediate_size + 127) / 128, false, rank * I, I, r.moe.scale_block), tag + "expert down scales");
  }
  if (r.has_ple) {
    const std::string lp = p + "ple.";
    const dgpp::QwenNgramGeometry ng = cfg.ngram_geometry();
    const size_t E2 = static_cast<size_t>(cfg.ple_embed_dim) * 2;
    const size_t c0 = static_cast<size_t>(g.hash_head_begin) * ng.head_dim * 2;
    const size_t cw = static_cast<size_t>(g.hash_heads) * ng.head_dim * 2;
    expect_device_equals(r.ple.key_proj, host_cols(fx.bytes(lp + "key_proj.weight"), cfg.hyper_width(), E2, c0, cw), tag + "ple key cols");
    expect_device_equals(r.ple.value_proj, host_cols(fx.bytes(lp + "value_proj.weight"), cfg.hidden_size, E2, c0, cw), tag + "ple value cols");
    expect_device_equals(r.ple.norm_key, fx.bytes(lp + "norm_key.weight"), tag + "ple norm_key");
    expect_device_equals(r.ple.norm_query, fx.bytes(lp + "norm_query.weight"), tag + "ple norm_query");
    expect_device_equals(r.ple.norm_conv, fx.bytes(lp + "norm_conv.weight"), tag + "ple norm_conv");
    expect_device_equals(r.ple.conv, fx.bytes(lp + "conv1d.weight"), tag + "ple conv");
    require(r.ple.table_scale == dgpp::bf16_bits_to_float(dgpp::float_to_bf16_bits(0.02f)), tag + "ple table scale");
    require(r.ple.row_begin == ng.head_offset[static_cast<size_t>(g.hash_head_begin)], tag + "ple row begin");
  }
  require(r.bytes == QwenLayerStream::layer_bytes(cfg, r.layer, rank, world), tag + "layer bytes formula");
}

}  // namespace

DGPP_TEST(qwen_loader_slices_every_class_byte_exact_at_worlds_1_2_4) {
  const Fixture fx = write_fixture();
  const int layers = fx.cfg.num_hidden_layers + 1;
  for (const int world : {1, 2, 4}) {
    uint64_t sum_source = 0;
    for (int rank = 0; rank < world; ++rank) {
      QwenLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::QwenResidency::Streaming,
                        dgpp::QwenHeadSharding::Full, /*resident_mtp=*/true);
      for (int l = 0; l < layers; ++l) {
        const uint64_t before = s.source_bytes_read();
        const auto& r = s.load_layer(l);
        require(r.layer == l, "layer index");
        check_layer(fx, s, r);
        require(s.source_bytes_read() - before ==
                    QwenLayerStream::planned_layer_source_bytes(fx.cfg, l, rank, world),
                "the source-byte plan equals the bytes read");
        s.release_layer();
      }
      sum_source += s.source_bytes_read();
    }
    std::printf("world %d: %.2f MB of source read across the ranks\n", world,
                static_cast<double>(sum_source) / 1048576.0);
  }
}

DGPP_TEST(qwen_loader_globals_and_ngram_table_slices) {
  const Fixture fx = write_fixture();
  const dgpp::QwenNgramGeometry ng = fx.cfg.ngram_geometry();
  // The whole table's bytes in shard order.
  std::vector<uint8_t> table;
  const std::string tp = dgpp::qwen_layer_prefix(fx.cfg, fx.cfg.ple_layer()) + "ple.ple_embedding.ngram_embedding.";
  for (int sh = 0; sh < fx.cfg.split_ngram_parts; ++sh) {
    const auto b = fx.bytes(tp + "shard_" + std::to_string(sh) + ".weight");
    table.insert(table.end(), b.begin(), b.end());
  }
  require(static_cast<int64_t>(table.size()) == ng.padded_rows * ng.head_dim, "fixture table size");
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      QwenLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::QwenResidency::Streaming,
                        world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full);
      const auto& g = s.load_globals();
      const size_t H2 = static_cast<size_t>(fx.cfg.hidden_size) * 2;
      expect_device_equals(g.embed, fx.bytes("model.language_model.embed_tokens.weight"), "embed");
      expect_device_equals(g.lm_head, host_rows(fx.bytes("lm_head.weight"), H2, g.lm_vocab_begin, g.lm_vocab_count), "lm head slice");
      require(g.lm_vocab_count == QwenLayerStream::lm_vocab_count(fx.cfg, rank, world, world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full), "lm head count");
      expect_device_equals(g.mixer.down, fx.bytes("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight"), "mixer down");
      require(g.mixer.inject == nullptr, "the mixer has no inject");
      expect_device_equals(g.mtp_fc_hidden, fx.bytes("mtp.fc_hidden.weight"), "mtp fc_hidden");
      expect_device_equals(g.mtp_pre_fc_norm_hidden, fx.bytes("mtp.pre_fc_norm_hidden.weight"), "mtp pre_fc_norm_hidden");
      expect_device_equals(g.mtp_mixer.up, fx.bytes("mtp.hyper_connection_mixer.input_mix_weight_up.weight"), "mtp mixer up");
      require(g.bytes == QwenLayerStream::globals_bytes(fx.cfg, rank, world, world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full), "globals formula");
      const auto& t = s.load_ngram_table();
      const int64_t hb = s.geometry().hash_head_begin, hn = s.geometry().hash_heads;
      const int64_t rb = ng.head_offset[static_cast<size_t>(hb)];
      const int64_t re = hb + hn < ng.heads ? ng.head_offset[static_cast<size_t>(hb + hn)] : ng.total_rows;
      require(t.row_begin == rb && t.rows == re - rb, "table slice range");
      expect_device_equals(t.rows_e4m3, host_rows(table, static_cast<size_t>(ng.head_dim), rb, re - rb), "table rows");
      require(t.bytes == QwenLayerStream::ngram_table_bytes(fx.cfg, rank, world), "table bytes formula");
    }
  }
}

// The mmap'ed table: the stream maps the shard(s) instead of
// copying the rows; every row read through the mapping is the shard's,
// the gather lays a rank's heads out as the device gather would, the
// plan's table bytes are zero, and the mode is off unless asked for.
// The dense stack encoded to block FP8 at load (2026-09-10,
// engine.dense_weights = "fp8"): off by default; on, a rows slice, a
// columns slice and the merged GDN qkv carry the recipe's codes — scale
// amax / 448 per 128 x 128 block, round-to-nearest-even e4m3 of w / scale
// — bit for bit against a scalar reference, dequantize back within the
// format's precision, and the layer's resident bytes shrink.
namespace {
void reference_fp8(const std::vector<uint16_t>& w, int64_t rows, int64_t cols, std::vector<uint8_t>* codes,
                   std::vector<float>* scales) {
  const int64_t sr = (rows + 127) / 128, sc = (cols + 127) / 128;
  codes->assign(static_cast<size_t>(rows) * cols, 0);
  scales->assign(static_cast<size_t>(sr) * sc, 0.f);
  for (int64_t br = 0; br < sr; ++br)
    for (int64_t bc = 0; bc < sc; ++bc) {
      float amax = 0.f;
      for (int64_t r = br * 128; r < std::min(rows, br * 128 + 128); ++r)
        for (int64_t c = bc * 128; c < std::min(cols, bc * 128 + 128); ++c)
          amax = std::max(amax, std::fabs(dgpp::bf16_bits_to_float(w[static_cast<size_t>(r) * cols + c])));
      const float s = amax > 0.f ? amax / 448.f : 1.f;
      (*scales)[static_cast<size_t>(br) * sc + bc] = s;
      for (int64_t r = br * 128; r < std::min(rows, br * 128 + 128); ++r)
        for (int64_t c = bc * 128; c < std::min(cols, bc * 128 + 128); ++c)
          (*codes)[static_cast<size_t>(r) * cols + c] =
              dgpp::float_to_fp8_e4m3_bits(dgpp::bf16_bits_to_float(w[static_cast<size_t>(r) * cols + c]) / s);
    }
}
void expect_fp8_matches(const dgpp::GlmQuantMatrix& q, const std::vector<uint16_t>& w, const std::string& what) {
  std::vector<uint8_t> codes;
  std::vector<float> scales;
  reference_fp8(w, q.rows, q.cols, &codes, &scales);
  const std::vector<uint8_t> got = device_bytes(q.payload, codes.size());
  require(got == codes, what + ": the codes differ from the reference recipe");
  const std::vector<uint8_t> sb = device_bytes(q.scales, scales.size() * 4);
  std::vector<float> got_s(scales.size());
  std::memcpy(got_s.data(), sb.data(), sb.size());
  require(got_s == scales, what + ": the scales differ from the reference recipe");
  // Dequantized back: within the e4m3 grid's half ulp (2^-4 relative;
  // the subnormal zone's absolute 2^-10 of the scale).
  const int64_t sc = (q.cols + 127) / 128;
  for (int64_t r = 0; r < q.rows; ++r)
    for (int64_t c = 0; c < q.cols; ++c) {
      const float s = scales[static_cast<size_t>(r / 128) * sc + c / 128];
      const float x = dgpp::bf16_bits_to_float(w[static_cast<size_t>(r) * q.cols + c]);
      const float dq = dgpp::fp8_e4m3_bits_to_float(got[static_cast<size_t>(r) * q.cols + c]) * s;
      const float tol = std::fabs(x) / 16.f + s / 1024.f + 1e-30f;
      require(std::fabs(dq - x) <= tol, what + ": dequantized value outside the format's precision at (" +
                                             std::to_string(r) + ", " + std::to_string(c) + ")");
    }
}
}  // namespace

DGPP_TEST(qwen_loader_dense_fp8_at_load_is_the_reference_recipe) {
  const Fixture fx = write_fixture();
  require(!QwenLayerStream::dense_weights_fp8(), "the dense stack is the checkpoint's by default");
  const std::string p = dgpp::qwen_layer_prefix(fx.cfg, 0);
  const size_t bf16_bytes = QwenLayerStream::layer_bytes(fx.cfg, 0, 0, 1);
  QwenLayerStream::set_dense_weights_fp8(true);
  struct Reset { ~Reset() { QwenLayerStream::set_dense_weights_fp8(false); } } reset;
  require(QwenLayerStream::layer_bytes(fx.cfg, 0, 0, 1) < bf16_bytes, "the FP8 layer is smaller in the plan");
  QwenLayerStream s(fx.cfg, fx.dir, 0, 1, dgpp::QwenResidency::Streaming, dgpp::QwenHeadSharding::Full);
  const auto& l = s.load_layer(0);
  require(l.kind == dgpp::QwenLayerKind::Gdn, "layer 0 of the fixture is a GDN layer");
  require(l.gdn.in_proj_z == nullptr && l.gdn.in_proj_z_fp8.payload != nullptr, "z is FP8 only");
  require(l.gdn.in_proj_a != nullptr, "a stays BF16");
  // A rows slice (the whole matrix at world 1).
  {
    const auto w = fx.bytes(p + "linear_attn.in_proj_z.weight");
    std::vector<uint16_t> h(w.size() / 2);
    std::memcpy(h.data(), w.data(), w.size());
    expect_fp8_matches(l.gdn.in_proj_z_fp8, h, "in_proj_z");
  }
  // A columns slice (the whole matrix at world 1) and the merged qkv.
  {
    const auto w = fx.bytes(p + "linear_attn.out_proj.weight");
    std::vector<uint16_t> h(w.size() / 2);
    std::memcpy(h.data(), w.data(), w.size());
    expect_fp8_matches(l.gdn.out_proj_fp8, h, "out_proj");
  }
  {
    const auto w = fx.bytes(p + "linear_attn.in_proj_qkv.weight");
    std::vector<uint16_t> h(w.size() / 2);
    std::memcpy(h.data(), w.data(), w.size());
    expect_fp8_matches(l.gdn.in_proj_qkv_fp8, h, "in_proj_qkv");
  }
  require(l.attn_gr.down == nullptr && l.attn_gr.down_fp8.payload != nullptr, "the GR site is FP8");
  require(l.moe.shared[0] == nullptr && l.moe.shared_fp8[0].payload != nullptr, "the shared expert is FP8");
  const auto& g = s.load_globals();
  require(g.lm_head == nullptr && g.lm_head_fp8.payload != nullptr && g.lm_head_fp8.rows == g.lm_vocab_count,
          "lm_head is FP8");
  require(g.bytes == QwenLayerStream::globals_bytes(fx.cfg, 0, 1, dgpp::QwenHeadSharding::Full), "globals formula (fp8)");
}

DGPP_TEST(qwen_loader_mmap_ngram_table_reads_the_shards_rows) {
  const Fixture fx = write_fixture();
  const dgpp::QwenNgramGeometry ng = fx.cfg.ngram_geometry();
  std::vector<uint8_t> table;
  const std::string tp = dgpp::qwen_layer_prefix(fx.cfg, fx.cfg.ple_layer()) + "ple.ple_embedding.ngram_embedding.";
  for (int sh = 0; sh < fx.cfg.split_ngram_parts; ++sh) {
    const auto b = fx.bytes(tp + "shard_" + std::to_string(sh) + ".weight");
    table.insert(table.end(), b.begin(), b.end());
  }
  require(!QwenLayerStream::ngram_table_mmap(), "the mmap mode is off by default");
  QwenLayerStream::set_ngram_table_mmap(true);
  struct Reset { ~Reset() { QwenLayerStream::set_ngram_table_mmap(false); } } reset;
  const size_t hd = static_cast<size_t>(ng.head_dim);
  for (const int world : {1, 2}) {
    for (int rank = 0; rank < world; ++rank) {
      QwenLayerStream s(fx.cfg, fx.dir, rank, world, dgpp::QwenResidency::Streaming,
                        world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full);
      require(QwenLayerStream::ngram_table_bytes(fx.cfg, rank, world) == 0, "no table bytes under mmap");
      const auto& t = s.load_ngram_table();
      require(t.rows_e4m3 == nullptr && t.mmap != nullptr && t.bytes == 0, "the mmap'ed view");
      require(t.mmap->capacity() == dgpp::qwen_ngram_shard_capacity(fx.cfg) && t.mmap->row_bytes() == ng.head_dim, "geometry");
      for (int64_t r : {int64_t{0}, int64_t{1}, ng.total_rows / 2, ng.total_rows - 1})
        require(std::memcmp(t.mmap->row(r), table.data() + static_cast<size_t>(r) * hd, hd) == 0, "row " + std::to_string(r));
      // A gather of the rank's heads: ids [n, heads] → dst [n, heads_local, hd].
      const int hb = s.geometry().hash_head_begin, hn = s.geometry().hash_heads;
      const int n = 5;
      std::vector<int32_t> ids(static_cast<size_t>(n) * ng.heads);
      uint64_t x = 0x9E3779B97F4A7C15ull ^ static_cast<uint64_t>(rank * 7 + world);
      for (int tkn = 0; tkn < n; ++tkn)
        for (int h = 0; h < ng.heads; ++h) {
          x ^= x << 13; x ^= x >> 7; x ^= x << 17;
          ids[static_cast<size_t>(tkn) * ng.heads + h] = static_cast<int32_t>(ng.head_offset[h] + static_cast<int64_t>(x % static_cast<uint64_t>(ng.head_vocab[h])));
        }
      std::vector<uint8_t> dst(static_cast<size_t>(n) * hn * hd);
      t.mmap->gather(ids.data(), n, ng.heads, hb, hn, dst.data());
      for (int tkn = 0; tkn < n; ++tkn)
        for (int hl = 0; hl < hn; ++hl) {
          const int32_t id = ids[static_cast<size_t>(tkn) * ng.heads + hb + hl];
          require(std::memcmp(dst.data() + (static_cast<size_t>(tkn) * hn + hl) * hd, table.data() + static_cast<size_t>(id) * hd, hd) == 0,
                  "gathered row");
        }
    }
  }
}

DGPP_TEST(qwen_loader_repacks_the_autoround_hybrid_exactly) {
  // The hybrid (docs/qwen38_autoround_int4_plan.md D2): every expert
  // matrix's words are the fixture's GPTQ words transposed, its scales the
  // fixture's f16 scales transposed, the dequant is GPTQ's own; the dense
  // classes' shipped fp8 codes and F32 scales land as is; the head is the
  // int8 triple transposed; the zeros never become resident.
  const Fixture fx = write_gptq_fixture();
  require(fx.cfg.experts_gptq_int4 && fx.cfg.lm_head_gptq_int8 && fx.cfg.dense_fp8_shipped,
          "the fixture selects the hybrid");
  const bool saved = QwenLayerStream::dense_weights_fp8();
  QwenLayerStream::set_dense_weights_fp8(true);
  const auto& cfg = fx.cfg;
  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size, E = cfg.num_experts;
  {
    QwenLayerStream s(cfg, fx.dir, 0, 1, dgpp::QwenResidency::Streaming, dgpp::QwenHeadSharding::Full);
    for (const int layer : {0, 1, 2, cfg.mtp_layer()}) {
      const auto& r = s.load_layer(layer);
      const std::string tag = "hybrid layer " + std::to_string(layer) + ": ";
      require(r.moe.packq() && r.moe.experts_packed.size() == static_cast<size_t>(E) * 3 &&
                  r.moe.experts.empty() && r.moe.experts_fp4.empty(),
              tag + "every expert packed, no other form");
      const std::string p = dgpp::qwen_layer_prefix(cfg, layer);
      const std::string mp = p + "mlp.";
      for (int e = 0; e < E; ++e)
        for (int which = 0; which < 3; ++which) {
          const char* names[3] = {"gate_proj", "up_proj", "down_proj"};
          const std::string base = mp + "experts." + std::to_string(e) + "." + names[which];
          const int64_t N = which == 2 ? H : I, K = which == 2 ? I : H;
          const dgpp::GlmPackedMatrix& m = r.moe.experts_packed[static_cast<size_t>(e) * 3 + which];
          require(m.rows == N && m.cols == K && m.bits == 4 && m.scale_fmt == dgpp::kPackedScaleF16G128,
                  tag + "packed geometry " + base);
          expect_device_equals(m.packed, gptq_words_transposed(fx, base, N, K, 4), tag + "words " + base);
          expect_device_equals(m.scales, gptq_scales_transposed(fx, base, N, K, cfg.gptq_group), tag + "scales " + base);
          if (e == 0 || e == E - 1) {
            const std::vector<uint8_t> w = device_bytes(m.packed, m.packed_bytes());
            const std::vector<uint8_t> sc = device_bytes(m.scales, m.scale_bytes());
            for (const auto& [n, k] : {std::pair<int64_t, int64_t>{0, 0}, {N - 1, K - 1}, {N / 2, 129 % K}, {3, K - 5}}) {
              const float got = dgpp::packq_decode(reinterpret_cast<const uint32_t*>(w.data()),
                                                   reinterpret_cast<const uint16_t*>(sc.data()), K, 4, n, k,
                                                   dgpp::kPackedScaleF16G128);
              require(got == gptq_dequant(fx, base, N, K, 4, cfg.gptq_group, n, k), tag + "dequant == GPTQ's " + base);
            }
          }
        }
      // The dense classes as shipped: codes and F32 scales byte for byte.
      if (r.kind == dgpp::QwenLayerKind::Gdn) {
        const std::string gp = p + "linear_attn.";
        require(r.gdn.in_proj_qkv == nullptr && r.gdn.in_proj_qkv_fp8.payload != nullptr, tag + "qkv fp8 as shipped");
        expect_device_equals(r.gdn.in_proj_qkv_fp8.payload, fx.bytes(gp + "in_proj_qkv.weight"), tag + "qkv codes");
        expect_device_equals(r.gdn.in_proj_qkv_fp8.scales, fx.bytes(gp + "in_proj_qkv.weight_scale_inv"), tag + "qkv scales");
        expect_device_equals(r.gdn.in_proj_z_fp8.payload, fx.bytes(gp + "in_proj_z.weight"), tag + "z codes");
        expect_device_equals(r.gdn.out_proj_fp8.payload, fx.bytes(gp + "out_proj.weight"), tag + "out codes");
        expect_device_equals(r.gdn.out_proj_fp8.scales, fx.bytes(gp + "out_proj.weight_scale_inv"), tag + "out scales");
      } else if (layer == cfg.mtp_layer()) {
        // The hybrid's draft layer ships its projections and shared expert in BF16.
        const std::string ap = p + "self_attn.";
        require(r.qsa.q_proj != nullptr && r.qsa.q_proj_fp8.payload == nullptr, tag + "draft q BF16 as shipped");
        expect_device_equals(r.qsa.q_proj, fx.bytes(ap + "q_proj.weight"), tag + "draft q bf16");
        require(r.moe.shared[0] != nullptr && r.moe.shared_fp8[0].payload == nullptr, tag + "draft shared BF16 as shipped");
        expect_device_equals(r.moe.shared[0], fx.bytes(mp + "shared_expert.gate_proj.weight"), tag + "draft shared gate bf16");
      } else {
        const std::string ap = p + "self_attn.";
        require(r.qsa.q_proj == nullptr && r.qsa.q_proj_fp8.payload != nullptr, tag + "q fp8 as shipped");
        expect_device_equals(r.qsa.q_proj_fp8.payload, fx.bytes(ap + "q_proj.weight"), tag + "q codes");
        expect_device_equals(r.qsa.q_proj_fp8.scales, fx.bytes(ap + "q_proj.weight_scale_inv"), tag + "q scales");
        expect_device_equals(r.qsa.o_proj_fp8.payload, fx.bytes(ap + "o_proj.weight"), tag + "o codes");
        require(r.qsa.index_qk_proj != nullptr && r.qsa.index_qk_proj_fp8.payload == nullptr,
                tag + "the indexer stays BF16 as shipped");
        expect_device_equals(r.qsa.index_qk_proj, fx.bytes(ap + "indexer.index_qk_proj.weight"), tag + "indexer bf16");
      }
      require(r.attn_gr.down != nullptr && r.attn_gr.down_fp8.payload == nullptr && r.mlp_gr.up != nullptr,
              tag + "the GR sites stay BF16 as shipped");
      if (r.has_ple)
        require(r.ple.key_proj != nullptr && r.ple.key_proj_fp8.payload == nullptr && r.ple.value_proj != nullptr,
                tag + "the PLE projections stay BF16 as shipped");
      if (layer != cfg.mtp_layer()) {
        expect_device_equals(r.moe.shared_fp8[0].payload, fx.bytes(mp + "shared_expert.gate_proj.weight"), tag + "shared gate codes");
        expect_device_equals(r.moe.shared_fp8[0].scales, fx.bytes(mp + "shared_expert.gate_proj.weight_scale_inv"), tag + "shared gate scales");
        expect_device_equals(r.moe.shared_fp8[2].payload, fx.bytes(mp + "shared_expert.down_proj.weight"), tag + "shared down codes");
      }
      require(r.bytes == QwenLayerStream::layer_bytes(cfg, r.layer, 0, 1), tag + "layer bytes formula");
    }
    const auto& g = s.load_globals();
    require(g.lm_head == nullptr && g.lm_head_fp8.payload == nullptr && g.lm_head_packed.packed != nullptr,
            "the head is packed and nothing else");
    require(g.lm_head_packed.rows == cfg.vocab_size && g.lm_head_packed.cols == H && g.lm_head_packed.bits == 8 &&
                g.lm_head_packed.scale_fmt == dgpp::kPackedScaleF16G128,
            "packed head geometry");
    // The resident head words: the checkpoint's rows, or (the default since
    // the bit-plane head, 2026-09-29) those rows permuted into per-row
    // planes — the host permutation of the same bytes.
    const std::vector<uint8_t> head_rows = gptq_words_transposed(fx, "lm_head", cfg.vocab_size, H, 8);
    {
      std::vector<uint8_t> want = head_rows;
      if (g.lm_head_packed.layout == dgpp::kPackedLayoutPlanes8)
        dgpp::packq_planes_permute_int8(want.data(), cfg.vocab_size, H);
      expect_device_equals(g.lm_head_packed.packed, want, "head words");
    }
    expect_device_equals(g.lm_head_packed.scales, gptq_scales_transposed(fx, "lm_head", cfg.vocab_size, H, cfg.gptq_group), "head scales");
    {
      const std::vector<uint8_t>& w = head_rows;  // the row layout the decode reads (== the resident bytes, above)
      const std::vector<uint8_t> sc = device_bytes(g.lm_head_packed.scales, g.lm_head_packed.scale_bytes());
      for (const auto& [n, k] : {std::pair<int64_t, int64_t>{0, 0}, {cfg.vocab_size - 1, H - 1}, {17, 131}}) {
        const float got = dgpp::packq_decode(reinterpret_cast<const uint32_t*>(w.data()),
                                             reinterpret_cast<const uint16_t*>(sc.data()), H, 8, n, k,
                                             dgpp::kPackedScaleF16G128);
        require(got == gptq_dequant(fx, "lm_head", cfg.vocab_size, H, 8, cfg.gptq_group, n, k), "head dequant == GPTQ's");
      }
    }
  }
  // World 4: the down projection's K slice (64) is not a whole group — refused by name.
  {
    bool refused = false;
    try {
      QwenLayerStream s2(cfg, fx.dir, 0, 4, dgpp::QwenResidency::Streaming, dgpp::QwenHeadSharding::VocabSharded);
      (void)s2.load_layer(0);
    } catch (const std::runtime_error& e) {
      refused = std::string(e.what()).find("whole groups") != std::string::npos;
    }
    require(refused, "world 4 refused on the group grid");
  }
  // Without engine.dense_weights = fp8 the shipped fp8 stack is refused by name.
  QwenLayerStream::set_dense_weights_fp8(false);
  {
    bool refused = false;
    try {
      QwenLayerStream s3(cfg, fx.dir, 0, 1, dgpp::QwenResidency::Streaming, dgpp::QwenHeadSharding::Full);
      (void)s3.load_layer(0);
    } catch (const std::runtime_error& e) {
      refused = std::string(e.what()).find("dense_weights") != std::string::npos;
    }
    require(refused, "the shipped fp8 stack needs dense_weights fp8");
  }
  QwenLayerStream::set_dense_weights_fp8(true);
  // A zero word off the symmetric constant is refused before anything is served.
  {
    const std::string name = dgpp::qwen_layer_prefix(cfg, 0) + "mlp.experts.1.up_proj.qzeros";
    uint64_t at = 0;
    std::string shard;
    for (const auto& entry : fs::directory_iterator(fx.dir))
      if (entry.path().extension() == ".safetensors") {
        auto f = dgpp::SafetensorsFile::open(entry.path().string());
        f->for_each([&](const dgpp::TensorInfo& t) {
          if (t.name == name) {
            at = t.data_begin + 4;
            shard = entry.path().string();
          }
        });
      }
    require(!shard.empty(), "the zeros tensor is in the fixture");
    {
      std::fstream f(shard, std::ios::in | std::ios::out | std::ios::binary);
      const uint32_t bad = 0x77777767u;
      f.seekp(static_cast<std::streamoff>(at));
      f.write(reinterpret_cast<const char*>(&bad), 4);
    }
    bool refused = false;
    try {
      QwenLayerStream s4(cfg, fx.dir, 0, 1, dgpp::QwenResidency::Streaming, dgpp::QwenHeadSharding::Full);
      (void)s4.load_layer(0);
    } catch (const std::runtime_error& e) {
      refused = std::string(e.what()).find("symmetric") != std::string::npos;
    }
    require(refused, "a non-symmetric zero word is refused");
  }
  QwenLayerStream::set_dense_weights_fp8(saved);
}

DGPP_TEST(qwen_loader_nvfp4_metadata_is_replicated_at_world_two) {
  const Fixture fx = write_nvfp4_fixture();
  require(fx.cfg.experts_nvfp4, "the fixture selects NVFP4 experts");
  for (int rank = 0; rank < 2; ++rank) {
    QwenLayerStream s(fx.cfg, fx.dir, rank, 2, dgpp::QwenResidency::Streaming,
                      dgpp::QwenHeadSharding::VocabSharded);
    const auto& layer = s.load_layer(0);
    require(layer.moe.nvfp4(), "the backbone experts stay NVFP4");
    require(layer.moe.experts_fp4.size() == static_cast<size_t>(fx.cfg.num_experts) * 3,
            "every local NVFP4 expert matrix is present");
    require(layer.moe.expert_globals != nullptr, "the replicated weight scales are loaded");
    float scales[2] = {};
    require(layer.moe.act_scales != nullptr, "the activation scales are in the layer image");
    DGPP_CUDA_OK(cudaMemcpy(scales, layer.moe.act_scales, sizeof(scales), cudaMemcpyDeviceToHost));
    require(scales[0] == 1.f && scales[1] == 1.f,
            "both replicated activation scales match the fixture");
  }
}

DGPP_TEST(qwen_loader_nvfp4_activation_scales_survive_image_restore) {
  const Fixture fx = write_nvfp4_fixture();
  const fs::path cache = fs::current_path() / "qwen_loader_nvfp4_image_cache";
  fs::remove_all(cache);
  const std::string saved = QwenLayerStream::resident_image_dir();
  QwenLayerStream::set_resident_image_dir(cache.string());
  for (int pass = 0; pass < 2; ++pass) {
    QwenLayerStream stream(fx.cfg, fx.dir, 1, 2, dgpp::QwenResidency::Resident);
    const auto& layer = stream.load_layer(0);
    float scales[2] = {};
    DGPP_CUDA_OK(cudaMemcpy(scales, layer.moe.act_scales, sizeof(scales), cudaMemcpyDeviceToHost));
    require(scales[0] == 1.f && scales[1] == 1.f,
            "activation globals survive a resident-image round trip");
    if (pass == 0)
      require(stream.image_layers_captured() == 1, "the cold load captures the NVFP4 image");
    else {
      require(stream.image_layers_restored() == 1, "the second load restores the NVFP4 image");
      require(
          layer.moe.act_scale_w13 == 0.f && layer.moe.act_scale_w2 == 0.f,
          "restore obtains activation globals from the image without rereading them on the host");
    }
  }
  QwenLayerStream::set_resident_image_dir(saved);
  fs::remove_all(cache);
}

DGPP_TEST(qwen_loader_resident_mode_and_image_round_trip) {
  const Fixture fx = write_fixture();
  const fs::path cache = fs::current_path() / "qwen_loader_image_cache";
  fs::remove_all(cache);
  const std::string saved = QwenLayerStream::resident_image_dir();
  QwenLayerStream::set_resident_image_dir(cache.string());
  const int layers = fx.cfg.num_hidden_layers + 1;
  std::vector<std::vector<uint8_t>> built(static_cast<size_t>(layers));
  {
    QwenLayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::QwenResidency::Resident,
                      dgpp::QwenHeadSharding::VocabSharded, /*resident_mtp=*/true);
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      check_layer(fx, s, r);
      const auto span = s.resident_layer_span(l);
      require(span.first != nullptr && span.second == r.bytes, "resident span");
      built[static_cast<size_t>(l)] = device_bytes(span.first, span.second);
    }
    (void)s.load_globals();
    (void)s.load_ngram_table();
    require(s.image_layers_captured() == layers, "every layer captured");
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) (void)s.load_layer(l);
    require(s.source_bytes_read() == before, "cache hits read no storage");
    s.release_sources();
    require(s.sources_released(), "sources released");
    require(s.staging_released(), "the pinned staging mirror goes with the sources");
    bool refused = false;
    try {
      (void)s.hash_replicated();
    } catch (const std::runtime_error&) {
      refused = true;
    }
    require(refused, "the digest is a boot check");
  }
  {
    QwenLayerStream s(fx.cfg, fx.dir, 1, 2, dgpp::QwenResidency::Resident,
                      dgpp::QwenHeadSharding::VocabSharded, /*resident_mtp=*/true);
    const uint64_t before = s.source_bytes_read();
    for (int l = 0; l < layers; ++l) {
      const auto& r = s.load_layer(l);
      const auto span = s.resident_layer_span(l);
      require(device_bytes(span.first, span.second) == built[static_cast<size_t>(l)],
              "restored layer bitwise the built one");
      check_layer(fx, s, r);
    }
    require(s.image_layers_restored() == layers && s.source_bytes_read() == before,
            "every layer restored from the image, no source reads");
    // The digest note round trip.
    const dgpp::QwenReplicatedDigest d = s.hash_replicated();
    require(d.layer.size() == static_cast<size_t>(layers) && d.tensors > 0, "digest shape");
  }
  QwenLayerStream::set_resident_image_dir(saved);
  fs::remove_all(cache);
}

DGPP_TEST(qwen_loader_rebuilds_rejected_resident_images) {
  const Fixture fx = write_fixture();
  imagefx::Cache<QwenLayerStream> cache(fs::path(fx.dir) / "rejected-cache");
  for (const size_t field : {0, 8, 16}) {
    std::vector<uint8_t> expected;
    {
      QwenLayerStream stream(fx.cfg, fx.dir, 1, 2, dgpp::QwenResidency::Resident);
      (void)stream.load_layer(0);
      (void)stream.load_layer(1);
      const auto [base, bytes] = stream.resident_layer_span(0);
      expected = device_bytes(base, bytes);
    }
    cache.corrupt(field);
    {
      QwenLayerStream stream(fx.cfg, fx.dir, 1, 2, dgpp::QwenResidency::Resident);
      const auto& layer = stream.load_layer(0);
      (void)stream.load_layer(1);
      check_layer(fx, stream, layer);
      const auto [base, bytes] = stream.resident_layer_span(0);
      require(device_bytes(base, bytes) == expected, "rebuilt Qwen layer matches checkpoint");
      require(stream.image_layers_captured() == 1 && stream.image_layers_restored() == 1,
              "rebuild rejected Qwen layer and restore its neighbor");
      require(stream.source_bytes_read() > 0, "rebuild reads source weights");
    }
    {
      QwenLayerStream stream(fx.cfg, fx.dir, 1, 2, dgpp::QwenResidency::Resident);
      (void)stream.load_layer(0);
      require(stream.image_layers_restored() == 1 && stream.source_bytes_read() == 0,
              "repaired Qwen layer restores on next startup");
    }
  }
}

DGPP_TEST(qwen_loader_refuses_tampered_hash_buffers) {
  const Fixture fx = write_fixture();
  // Flip one byte of layer_multipliers inside the shard.
  const fs::path shard = fs::path(fx.dir) / "model.safetensors";
  const std::string name = dgpp::qwen_layer_prefix(fx.cfg, fx.cfg.ple_layer()) + "ple.ple_embedding.layer_multipliers";
  std::FILE* f = std::fopen(shard.c_str(), "r+b");
  require(f != nullptr, "open shard");
  uint64_t hlen = 0;
  require(std::fread(&hlen, 8, 1, f) == 1, "header length");
  std::string header(hlen, '\0');
  require(std::fread(header.data(), 1, hlen, f) == hlen, "header");
  const size_t at = header.find("\"" + name + "\"");
  require(at != std::string::npos, "tensor in header");
  const size_t off_at = header.find("\"data_offsets\":[", at);
  const uint64_t begin = std::stoull(header.substr(off_at + 16));
  std::fseek(f, static_cast<long>(8 + hlen + begin), SEEK_SET);
  uint8_t byte = 0;
  require(std::fread(&byte, 1, 1, f) == 1, "read byte");
  byte ^= 1;
  std::fseek(f, static_cast<long>(8 + hlen + begin), SEEK_SET);
  require(std::fwrite(&byte, 1, 1, f) == 1, "write byte");
  std::fclose(f);
  bool refused = false;
  try {
    QwenLayerStream s(fx.cfg, fx.dir, 0, 1);
  } catch (const std::runtime_error& e) {
    refused = std::string(e.what()).find("hash buffers") != std::string::npos;
  }
  require(refused, "a tampered multiplier is refused by name");
}

int main() { return dgpp::test::run_all(); }

// ---- the QAD export's twin (issue #113) --------------------------------------------
namespace {
// e8m0 bytes [start, +count) of a scale tensor as the resident F32 grid.
std::vector<uint8_t> widened_e8m0(const std::vector<uint8_t>& bytes) {
  std::vector<uint8_t> out(bytes.size() * 4);
  for (size_t i = 0; i < bytes.size(); ++i) {
    const float f = dgpp::e8m0_byte_to_float(bytes[i]);
    std::memcpy(out.data() + i * 4, &f, 4);
  }
  return out;
}
// bf16(e4m3 x 2^k) of a payload row range: the exact decode of an MXFP8 row slice.
std::vector<uint8_t> decoded_mxfp8_rows(const std::vector<uint8_t>& payload, const std::vector<uint8_t>& scales,
                                        int64_t cols, int64_t r0, int64_t rows) {
  std::vector<uint8_t> out(static_cast<size_t>(rows) * cols * 2);
  const int64_t sb = cols / 32;
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c) {
      const uint8_t code = payload[static_cast<size_t>((r0 + r) * cols + c)];
      const uint8_t sc = scales[static_cast<size_t>((r0 + r) * sb + c / 32)];
      const uint16_t v = dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(code) * dgpp::e8m0_byte_to_float(sc));
      std::memcpy(out.data() + (static_cast<size_t>(r) * cols + c) * 2, &v, 2);
    }
  return out;
}
// An MXFP8 resident matrix against the fixture: a row slice [r0, +rows) or a
// column slice [c0, +cols) of `base` (payload e4m3 [N, K], scales e8m0 [N, K/32]).
// bytes: the attention-side matrices keep the e8m0 bytes (their k a multiple
// of 64); the shared experts carry the grid widened to F32.
void expect_mxfp8(const Fixture& fx, const dgpp::GlmQuantMatrix& m, const std::string& base, bool row_slice,
                  int64_t start, int64_t count, const std::string& tag, bool bytes = true) {
  const QwenExpectedTensor& e = fx.expected(base + ".weight");
  const int64_t N = e.shape[0], K = e.shape[1];
  require(m.scale_block_rows == 1 && m.scale_block_cols == 32, tag + ": the 1 x 32 grid");
  require((m.scales_e8m0 != nullptr) == bytes && (m.scales != nullptr) == !bytes, tag + ": the scale form");
  const std::vector<uint8_t> payload = fx.bytes(base + ".weight");
  const std::vector<uint8_t> scales = fx.bytes(base + ".weight_scale");
  auto expect_scales = [&](const std::vector<uint8_t>& want_bytes, const std::string& what) {
    if (bytes) expect_device_equals(m.scales_e8m0, want_bytes, what);
    else expect_device_equals(m.scales, widened_e8m0(want_bytes), what);
  };
  if (row_slice) {
    require(m.rows == count && m.cols == K, tag + ": row slice geometry");
    expect_device_equals(m.payload, host_rows(payload, static_cast<size_t>(K), start, count), tag + " payload rows");
    expect_scales(host_rows(scales, static_cast<size_t>(K / 32), start, count), tag + " scale rows");
  } else {
    require(m.rows == N && m.cols == count, tag + ": column slice geometry");
    expect_device_equals(m.payload, host_cols(payload, N, static_cast<size_t>(K), static_cast<size_t>(start), static_cast<size_t>(count)), tag + " payload cols");
    expect_scales(host_cols(scales, N, static_cast<size_t>(K / 32), static_cast<size_t>(start / 32), static_cast<size_t>(count / 32)), tag + " scale cols");
  }
}
// An NVFP4 resident matrix (modelopt triple) against the fixture's slice.
void expect_fp4(const Fixture& fx, const dgpp::GlmFp4Matrix& m, const std::string& base, bool row_slice, int64_t start,
                int64_t count, const std::string& tag) {
  const QwenExpectedTensor& e = fx.expected(base + ".weight");
  const int64_t N = e.shape[0], K2 = e.shape[1];  // codes [N, K/2]
  const std::vector<uint8_t> codes = fx.bytes(base + ".weight");
  const std::vector<uint8_t> scales = fx.bytes(base + ".weight_scale");
  if (row_slice) {
    require(m.rows == count && m.cols == K2 * 2, tag + ": fp4 row slice geometry");
    expect_device_equals(m.payload, host_rows(codes, static_cast<size_t>(K2), start, count), tag + " fp4 codes rows");
    expect_device_equals(m.scales, host_rows(scales, static_cast<size_t>(K2 * 2 / 16), start, count), tag + " fp4 scale rows");
  } else {
    require(m.rows == N && m.cols == count, tag + ": fp4 column slice geometry");
    expect_device_equals(m.payload, host_cols(codes, N, static_cast<size_t>(K2), static_cast<size_t>(start / 2), static_cast<size_t>(count / 2)), tag + " fp4 codes cols");
    expect_device_equals(m.scales, host_cols(scales, N, static_cast<size_t>(K2 * 2 / 16), static_cast<size_t>(start / 16), static_cast<size_t>(count / 16)), tag + " fp4 scale cols");
  }
  float inv = 0.f;
  DGPP_CUDA_OK(cudaMemcpy(&inv, m.global_scale, 4, cudaMemcpyDeviceToHost));
  require(inv == 32.0f, tag + ": the global's reciprocal (the fixture's weight_scale_2 is 2^-5)");
}
}  // namespace

static void check_qad_loader(bool csf) {
  const Fixture fx = write_qad_fixture();
  const QwenTextConfig cfg = csf ? qwenfx::tiny_fp4local_config() : fx.cfg;
  if (csf) qwenfx::write_fixture_for(cfg, fx.dir);
  require(cfg.dense_mxfp8_shipped && cfg.ngram_table_nvfp4 && cfg.mtp_experts_nvfp4 && cfg.norms_f32,
          "the fixture selects the QAD export's forms");
  require(!QwenLayerStream::dense_weights_fp8(), "the dense stack is served as shipped, no engine key");
  const int layers = cfg.num_hidden_layers + 1;
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  for (const int world : {1, 2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      QwenLayerStream s(cfg, fx.dir, rank, world, dgpp::QwenResidency::Streaming,
                        world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full,
                        /*resident_mtp=*/true);
      const dgpp::QwenLocalGeometry& g = s.geometry();
      for (int l = 0; l < layers; ++l) {
        const uint64_t before = s.source_bytes_read();
        const auto& r = s.load_layer(l);
        const bool mtp = l == cfg.mtp_layer();
        const std::string p = dgpp::qwen_layer_prefix(cfg, l);
        const std::string tag = "world " + std::to_string(world) + " rank " + std::to_string(rank) + " layer " +
                                std::to_string(l) + " ";
        // The norms: F32 on the backbone, BF16 on the draft layer.
        for (const auto& [gp, gr] : {std::pair{p + "attn_hyper_connection.", &r.attn_gr},
                                     std::pair{p + "mlp_hyper_connection.", &r.mlp_gr}}) {
          if (mtp) {
            require(gr->hc_norm != nullptr && gr->hc_norm_f32 == nullptr, tag + "draft hc_norm BF16");
            expect_device_equals(gr->hc_norm, fx.bytes(gp + "hc_norm.weight"), tag + "draft hc_norm");
          } else {
            require(gr->hc_norm == nullptr && gr->hc_norm_f32 != nullptr, tag + "hc_norm F32");
            expect_device_equals(gr->hc_norm_f32, fx.bytes(gp + "hc_norm.weight"), tag + "hc_norm f32 bytes");
          }
          expect_device_equals(gr->down, fx.bytes(gp + "input_mix_weight_down.weight"), tag + "gr down");
        }
        if (r.kind == dgpp::QwenLayerKind::Gdn) {
          const std::string gp = p + "linear_attn.";
          const int64_t dk = cfg.gdn_key_head_dim, dv = cfg.gdn_value_head_dim;
          const int64_t K = static_cast<int64_t>(cfg.gdn_key_heads) * dk;
          const int64_t lk = g.local_key_heads, lv = g.local_value_heads;
          require(r.gdn.in_proj_qkv == nullptr && r.gdn.in_proj_qkv_fp8.payload != nullptr, tag + "qkv MXFP8");
          // The three segments' payload rows and scale rows.
          const std::vector<uint8_t> qkv = fx.bytes(gp + "in_proj_qkv.weight");
          const std::vector<uint8_t> qkv_s = fx.bytes(gp + "in_proj_qkv.weight_scale");
          std::vector<uint8_t> want = host_rows(qkv, H, rank * lk * dk, lk * dk);
          auto k = host_rows(qkv, H, K + rank * lk * dk, lk * dk);
          auto v = host_rows(qkv, H, 2 * K + rank * lv * dv, lv * dv);
          want.insert(want.end(), k.begin(), k.end());
          want.insert(want.end(), v.begin(), v.end());
          expect_device_equals(r.gdn.in_proj_qkv_fp8.payload, want, tag + "qkv payload segments");
          std::vector<uint8_t> ws = host_rows(qkv_s, H / 32, rank * lk * dk, lk * dk);
          k = host_rows(qkv_s, H / 32, K + rank * lk * dk, lk * dk);
          v = host_rows(qkv_s, H / 32, 2 * K + rank * lv * dv, lv * dv);
          ws.insert(ws.end(), k.begin(), k.end());
          ws.insert(ws.end(), v.begin(), v.end());
          require(r.gdn.in_proj_qkv_fp8.scales_e8m0 != nullptr && r.gdn.in_proj_qkv_fp8.scales == nullptr, tag + "qkv byte scales");
          expect_device_equals(r.gdn.in_proj_qkv_fp8.scales_e8m0, ws, tag + "qkv scale segments");
          require(r.gdn.in_proj_qkv_fp8.scale_block_rows == 1 && r.gdn.in_proj_qkv_fp8.scale_block_cols == 32, tag + "qkv grid");
          expect_mxfp8(fx, r.gdn.in_proj_z_fp8, gp + "in_proj_z", true, rank * lv * dv, lv * dv, tag + "z");
          expect_mxfp8(fx, r.gdn.out_proj_fp8, gp + "out_proj", false, rank * lv * dv, lv * dv, tag + "out_proj");
          // a and b: the exact bf16 decode of their MXFP8 rows.
          expect_device_equals(r.gdn.in_proj_a, decoded_mxfp8_rows(fx.bytes(gp + "in_proj_a.weight"), fx.bytes(gp + "in_proj_a.weight_scale"), H, rank * lv, lv), tag + "a decoded");
          expect_device_equals(r.gdn.in_proj_b, decoded_mxfp8_rows(fx.bytes(gp + "in_proj_b.weight"), fx.bytes(gp + "in_proj_b.weight_scale"), H, rank * lv, lv), tag + "b decoded");
        } else if (!mtp) {
          const std::string ap = p + "self_attn.";
          const int64_t d = cfg.head_dim;
          require(r.qsa.q_proj == nullptr && r.qsa.q_proj_fp8.payload != nullptr, tag + "q MXFP8");
          expect_mxfp8(fx, r.qsa.q_proj_fp8, ap + "q_proj", true, g.head_begin * 2 * d, g.local_heads * 2 * d, tag + "q");
          expect_mxfp8(fx, r.qsa.k_proj_fp8, ap + "k_proj", true, g.kv_head_begin * d, g.local_kv_heads * d, tag + "k");
          expect_mxfp8(fx, r.qsa.v_proj_fp8, ap + "v_proj", true, g.kv_head_begin * d, g.local_kv_heads * d, tag + "v");
          expect_mxfp8(fx, r.qsa.o_proj_fp8, ap + "o_proj", false, g.head_begin * d, g.local_heads * d, tag + "o");
          const QwenExpectedTensor& ei = fx.expected(ap + "indexer.index_qk_proj.weight");
          require(r.qsa.index_qk_proj == nullptr, tag + "indexer MXFP8");
          expect_mxfp8(fx, r.qsa.index_qk_proj_fp8, ap + "indexer.index_qk_proj", true, 0, ei.shape[0], tag + "indexer");
        } else {
          // The draft layer's projections stay BF16.
          require(r.qsa.q_proj != nullptr && r.qsa.q_proj_fp8.payload == nullptr, tag + "draft q BF16");
          require(r.qsa.index_qk_proj != nullptr, tag + "draft indexer BF16");
        }
        const std::string mp = p + "mlp.";
        const int64_t S = g.local_shared_inter, I = g.local_inter;
        if (!mtp) {
          require(r.moe.shared[0] == nullptr && r.moe.shared_fp8[0].payload != nullptr, tag + "shared MXFP8");
          expect_mxfp8(fx, r.moe.shared_fp8[0], mp + "shared_expert.gate_proj", true, rank * S, S, tag + "shared gate", /*bytes=*/false);
          expect_mxfp8(fx, r.moe.shared_fp8[1], mp + "shared_expert.up_proj", true, rank * S, S, tag + "shared up", /*bytes=*/false);
          expect_mxfp8(fx, r.moe.shared_fp8[2], mp + "shared_expert.down_proj", false, rank * S, S, tag + "shared down", /*bytes=*/false);
        } else {
          require(r.moe.shared[0] != nullptr && r.moe.shared_fp8[0].payload == nullptr, tag + "draft shared BF16");
        }
        // The routed experts: NVFP4 on every layer; the draft's weight-only.
        require(r.moe.nvfp4() && r.moe.experts_fp4.size() == static_cast<size_t>(cfg.num_experts) * 3, tag + "nvfp4 experts");
        require(r.moe.experts_weight_only == mtp, tag + "weight-only on the draft layer alone");
        require((r.moe.act_scales == nullptr) == mtp, tag + "activation scales absent on the draft layer alone");
        for (int e = 0; e < cfg.num_experts; e += 3) {
          const std::string ep = mp + "experts." + std::to_string(e) + ".";
          expect_fp4(fx, r.moe.experts_fp4[static_cast<size_t>(e) * 3 + 0], ep + "gate_proj", true, rank * I, I, tag + "expert gate");
          expect_fp4(fx, r.moe.experts_fp4[static_cast<size_t>(e) * 3 + 2], ep + "down_proj", false, rank * I, I, tag + "expert down");
        }
        if (r.has_ple) {
          const std::string lp = p + "ple.";
          require(r.ple.norm_key == nullptr && r.ple.norm_key_f32 != nullptr, tag + "ple norms F32");
          expect_device_equals(r.ple.norm_key_f32, fx.bytes(lp + "norm_key.weight"), tag + "ple norm_key f32");
          expect_device_equals(r.ple.norm_query_f32, fx.bytes(lp + "norm_query.weight"), tag + "ple norm_query f32");
          expect_device_equals(r.ple.norm_conv_f32, fx.bytes(lp + "norm_conv.weight"), tag + "ple norm_conv f32");
          require(r.ple.table_scale == 0.02f, tag + "ple table global (F32, exact)");
        }
        require(r.bytes == QwenLayerStream::layer_bytes(cfg, l, rank, world), tag + "layer bytes formula");
        require(s.source_bytes_read() - before == QwenLayerStream::planned_layer_source_bytes(cfg, l, rank, world),
                tag + "the source-byte plan equals the bytes read");
        s.release_layer();
      }
      // The globals: the model mixer's norm F32, the draft mixer's BF16.
      const auto& gl = s.load_globals();
      require(gl.mixer.hc_norm == nullptr && gl.mixer.hc_norm_f32 != nullptr, "mixer norm F32");
      expect_device_equals(gl.mixer.hc_norm_f32, fx.bytes("model.language_model.hyper_connection_mixer.hc_norm.weight"), "mixer norm bytes");
      require(gl.mtp_mixer.hc_norm != nullptr && gl.mtp_mixer.hc_norm_f32 == nullptr, "draft mixer norm BF16");
      require(gl.bytes == QwenLayerStream::globals_bytes(cfg, rank, world, world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full),
              "globals formula (QAD)");
    }
  }
}

DGPP_TEST(qwen_loader_loads_the_qad_export_byte_exact_at_worlds_1_2_4) { check_qad_loader(false); }
DGPP_TEST(qwen_loader_loads_nested_csf_export_byte_exact_at_worlds_1_2_4) { check_qad_loader(true); }

DGPP_TEST(qwen_loader_qad_ngram_table_resident_and_mmap) {
  const Fixture fx = write_qad_fixture();
  const QwenTextConfig& cfg = fx.cfg;
  const dgpp::QwenNgramGeometry ng = cfg.ngram_geometry();
  const int hd = ng.head_dim;
  const size_t cb = static_cast<size_t>(hd / 2), sb = static_cast<size_t>(hd / 16);
  // The whole table's codes and scales in shard order.
  std::vector<uint8_t> codes, scales;
  const std::string tp = dgpp::qwen_layer_prefix(cfg, cfg.ple_layer()) + "ple.ple_embedding.ngram_embedding.";
  for (int sh = 0; sh < cfg.split_ngram_parts; ++sh) {
    const auto c = fx.bytes(tp + "shard_" + std::to_string(sh) + ".weight");
    const auto sc = fx.bytes(tp + "shard_" + std::to_string(sh) + ".weight_scale");
    codes.insert(codes.end(), c.begin(), c.end());
    scales.insert(scales.end(), sc.begin(), sc.end());
  }
  require(static_cast<int64_t>(codes.size()) == ng.padded_rows * static_cast<int64_t>(cb), "fixture codes size");
  require(static_cast<int64_t>(scales.size()) == ng.padded_rows * static_cast<int64_t>(sb), "fixture scales size");
  for (const int world : {1, 2}) {
    for (int rank = 0; rank < world; ++rank) {
      // Resident.
      {
        QwenLayerStream s(cfg, fx.dir, rank, world, dgpp::QwenResidency::Streaming,
                          world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full);
        const auto& t = s.load_ngram_table();
        const int64_t rb = t.row_begin, rows = t.rows;
        require(t.fp4 && t.rows_e4m3 == nullptr && t.rows_fp4 != nullptr && t.rows_fp4_scales != nullptr, "the NVFP4 resident view");
        require(t.scale == 0.02f, "the table's F32 global");
        require(t.bytes == static_cast<size_t>(rows) * (cb + sb), "table bytes: codes + scales");
        require(t.bytes == QwenLayerStream::ngram_table_bytes(cfg, rank, world), "table bytes formula");
        expect_device_equals(t.rows_fp4, host_rows(codes, cb, rb, rows), "table codes rows");
        expect_device_equals(t.rows_fp4_scales, host_rows(scales, sb, rb, rows), "table scale rows");
      }
      // Mapped.
      QwenLayerStream::set_ngram_table_mmap(true);
      struct Reset { ~Reset() { QwenLayerStream::set_ngram_table_mmap(false); } } reset;
      {
        QwenLayerStream s(cfg, fx.dir, rank, world, dgpp::QwenResidency::Streaming,
                          world > 1 ? dgpp::QwenHeadSharding::VocabSharded : dgpp::QwenHeadSharding::Full);
        const auto& t = s.load_ngram_table();
        require(t.fp4 && t.rows_fp4 == nullptr && t.mmap != nullptr && t.bytes == 0, "the mmap'ed NVFP4 view");
        require(t.mmap->has_scales() && t.mmap->row_bytes() == static_cast<int>(cb) && t.mmap->scale_row_bytes() == static_cast<int>(sb), "mmap geometry");
        for (int64_t r = 0; r < ng.padded_rows; r += 37) {
          require(std::memcmp(t.mmap->row(r), codes.data() + static_cast<size_t>(r) * cb, cb) == 0, "codes row " + std::to_string(r));
          require(std::memcmp(t.mmap->scale_row(r), scales.data() + static_cast<size_t>(r) * sb, sb) == 0, "scale row " + std::to_string(r));
        }
        // A gather of this rank's heads with the scale staging alongside.
        const int n = 5, hb = s.geometry().hash_head_begin, hn = s.geometry().hash_heads;
        std::vector<int32_t> ids(static_cast<size_t>(n) * ng.heads);
        for (int tkn = 0; tkn < n; ++tkn)
          for (int h = 0; h < ng.heads; ++h)
            ids[static_cast<size_t>(tkn) * ng.heads + h] = static_cast<int32_t>(ng.head_offset[h] + (tkn * 7 + h) % ng.head_vocab[h]);
        std::vector<uint8_t> dst(static_cast<size_t>(n) * hn * cb), dst_s(static_cast<size_t>(n) * hn * sb);
        t.mmap->gather(ids.data(), n, ng.heads, hb, hn, dst.data(), dst_s.data());
        for (int tkn = 0; tkn < n; ++tkn)
          for (int hl = 0; hl < hn; ++hl) {
            const size_t row = static_cast<size_t>(ids[static_cast<size_t>(tkn) * ng.heads + hb + hl]);
            const size_t pair = static_cast<size_t>(tkn) * hn + hl;
            require(std::memcmp(dst.data() + pair * cb, codes.data() + row * cb, cb) == 0, "gathered codes");
            require(std::memcmp(dst_s.data() + pair * sb, scales.data() + row * sb, sb) == 0, "gathered scales");
          }
        bool refused = false;
        try {
          t.mmap->gather(ids.data(), n, ng.heads, hb, hn, dst.data());
        } catch (const std::invalid_argument&) {
          refused = true;
        }
        require(refused, "a gather without the scale staging is refused");
      }
    }
  }
}
