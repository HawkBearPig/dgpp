#include "kernels/minimax.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "engine/verify_schedule.hpp"
#include "kernels/dflash2.hpp"
#include "kernels/dsv41_dspark.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/sink_attn.hpp"
#include "minimax_fixture.hpp"
#include "models/minimax/dspark.hpp"
#include "models/minimax/forward.hpp"
#include "models/minimax/reference.hpp"

void minimax_pdl_test();

namespace {
using namespace dgpp;
void require(bool ok, const std::string& msg) {
  if (!ok) throw std::runtime_error(msg);
}
template <class T>
struct Device {
  T* p = nullptr;
  size_t size;
  explicit Device(size_t n) : size(n) {
    DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
  }
  explicit Device(const std::vector<T>& v) : Device(v.size()) { upload(v); }
  ~Device() { cudaFree(p); }
  Device(const Device&) = delete;
  void upload(const std::vector<T>& v) {
    require(v.size() == size, "device upload size");
    DGPP_CUDA_OK(cudaMemcpy(p, v.data(), size * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> get() const {
    std::vector<T> v(size);
    DGPP_CUDA_OK(cudaMemcpy(v.data(), p, size * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
  }
};
uint16_t b(float x) {
  return float_to_bf16_bits(x);
}
template <class T>
std::vector<T> copy_device(const T* p, size_t n) {
  std::vector<T> out(n);
  DGPP_CUDA_OK(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
  return out;
}
void rectangular_slice(const uint8_t* local, int rows, int cols, const uint8_t* full, int full_rows,
                       int full_cols, int row0, int col0) {
  const auto a = copy_device(local, static_cast<size_t>(rows) * cols);
  const auto b = copy_device(full, static_cast<size_t>(full_rows) * full_cols);
  for (int r = 0; r < rows; ++r)
    require(std::equal(a.begin() + static_cast<size_t>(r) * cols,
                       a.begin() + static_cast<size_t>(r + 1) * cols,
                       b.begin() + static_cast<size_t>(r + row0) * full_cols + col0),
            "TP slice changed checkpoint bytes");
}
void matrix_slice(const GlmQuantMatrix& a, const GlmQuantMatrix& b, int rank, bool columns,
                  bool replicated = false) {
  const int row0 = !columns && !replicated ? rank * a.rows : 0;
  const int col0 = columns && !replicated ? rank * a.cols : 0;
  rectangular_slice(a.payload, a.rows, a.cols, b.payload, b.rows, b.cols, row0, col0);
  rectangular_slice(a.scales_e8m0, a.rows, a.cols / 32, b.scales_e8m0, b.rows, b.cols / 32, row0,
                    col0 / 32);
}
void loader_slices(const MinimaxConfig& cfg, const std::string& path) {
  MinimaxLayerStream full(cfg, path, 0, 1, LoaderResidency::Resident);
  const auto globals = full.load_globals();
  for (int layer = 0; layer < cfg.num_hidden_layers; ++layer) full.load_layer(layer);
  for (int world : {2, 4}) {
    for (int rank = 0; rank < world; ++rank) {
      MinimaxLayerStream part(cfg, path, rank, world, LoaderResidency::Resident,
                              LoaderHeadSharding::VocabSharded);
      const auto g = part.load_globals();
      require(copy_device(g.embed, size_t(cfg.vocab_size) * cfg.hidden_size) ==
                  copy_device(globals.embed, size_t(cfg.vocab_size) * cfg.hidden_size),
              "embedding must remain replicated");
      rectangular_slice(reinterpret_cast<const uint8_t*>(g.lm_head), g.lm_vocab_count,
                        cfg.hidden_size * 2, reinterpret_cast<const uint8_t*>(globals.lm_head),
                        cfg.vocab_size, cfg.hidden_size * 2, g.lm_vocab_begin, 0);
      for (int layer = 0; layer < cfg.num_hidden_layers; ++layer) {
        const auto& a = part.load_layer(layer);
        const auto& b = full.load_layer(layer);
        for (auto [x, y] : {std::pair{a.attn.q, b.attn.q},
                            {a.attn.k, b.attn.k},
                            {a.attn.v, b.attn.v},
                            {a.attn.index_q, b.attn.index_q}})
          matrix_slice(x, y, rank, false);
        matrix_slice(a.attn.o, b.attn.o, rank, true);
        matrix_slice(a.attn.index_k, b.attn.index_k, rank, false, true);
        const auto mlp = [&](const MinimaxMlpResident& x, const MinimaxMlpResident& y) {
          matrix_slice(x.gate, y.gate, rank, false);
          matrix_slice(x.up, y.up, rank, false);
          matrix_slice(x.down, y.down, rank, true);
        };
        if (!a.moe) {
          mlp(a.dense, b.dense);
          continue;
        }
        mlp(a.moe_w.shared, b.moe_w.shared);
        require(copy_device(a.moe_w.router, size_t(cfg.num_local_experts) * cfg.hidden_size) ==
                    copy_device(b.moe_w.router, size_t(cfg.num_local_experts) * cfg.hidden_size),
                "router must remain replicated FP32");
        for (size_t e = 0; e < a.moe_w.experts.size(); ++e) {
          const auto& x = a.moe_w.experts[e];
          const auto& y = b.moe_w.experts[e];
          const int r0 = e % 3 == 2 ? 0 : rank * x.rows;
          const int c0 = e % 3 == 2 ? rank * x.cols : 0;
          rectangular_slice(x.payload, x.rows, x.cols / 2, y.payload, y.rows, y.cols / 2, r0,
                            c0 / 2);
          rectangular_slice(x.scales, x.rows, x.cols / 16, y.scales, y.rows, y.cols / 16, r0,
                            c0 / 16);
          require(copy_device(x.global_scale, 1) == copy_device(y.global_scale, 1),
                  "FP4 global scale");
          require(
              copy_device(a.moe_w.input_globals[e], 1) == copy_device(b.moe_w.input_globals[e], 1),
              "FP4 activation global scale");
        }
      }
    }
  }
  std::puts("MiniMax TP2/TP4 checkpoint slices exact");
}
float f(uint16_t x) {
  return bf16_bits_to_float(x);
}
void near(const std::vector<uint16_t>& got, const std::vector<uint16_t>& want, double limit,
          const char* name) {
  require(got.size() == want.size(), "comparison sizes");
  double err = 0, ref = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    require(std::isfinite(f(got[i])), std::string(name) + " nonfinite");
    double d = double(f(got[i])) - f(want[i]);
    err += d * d;
    ref += double(f(want[i])) * f(want[i]);
  }
  const double relative = std::sqrt(err / std::max(ref, 1e-30));
  std::printf("%s relative RMS %.8g\n", name, relative);
  require(relative <= limit, std::string(name) + " numerical mismatch");
}
void dspark_loader(const std::string& dir) {
  const auto c = minimax_dspark_config_file(dir + "/config.json");
  auto full = load_minimax_dspark(c, dir, nullptr, 0, 1);
  require(full.bytes == minimax_dspark_weights_bytes(c, 1),
          "DSpark memory plan reconciles full weights");
  std::vector<std::unique_ptr<SafetensorsFile>> files;
  for (const auto& e : std::filesystem::directory_iterator(dir))
    if (e.path().extension() == ".safetensors")
      files.push_back(SafetensorsFile::open(e.path().string()));
  auto source = [&](const std::string& name) -> const TensorInfo& {
    for (const auto& f : files)
      if (auto* t = f->find("language_model.model.dspark." + name)) return *t;
    throw std::runtime_error("missing source " + name);
  };
  auto verbatim = [&](const auto* data, const std::string& name) {
    const auto& t = source(name);
    const auto got = copy_device(reinterpret_cast<const uint8_t*>(data), t.nbytes());
    require(std::memcmp(got.data(), t.data, t.nbytes()) == 0, "DSpark changed source " + name);
  };
  const int H = c.hidden_size, Q = c.q_row(), K = c.kv_row(), I = c.intermediate_size;
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    const auto& w = full.layers[l];
    const auto p = "layers." + std::to_string(l) + ".decoder_layer.";
    verbatim(w.input_norm, p + "input_layernorm.weight");
    verbatim(w.post_norm, p + "post_attention_layernorm.weight");
    verbatim(w.q_norm, p + "self_attn.q_norm.weight");
    verbatim(w.k_norm, p + "self_attn.k_norm.weight");
    verbatim(w.qkv_fp8, p + "self_attn.q_proj.weight");
    verbatim(w.qkv_e8m0, p + "self_attn.q_proj.weight_scale_inv");
    verbatim(w.qkv_fp8 + size_t(Q) * H, p + "self_attn.k_proj.weight");
    verbatim(w.qkv_e8m0 + size_t(Q) * H / 32, p + "self_attn.k_proj.weight_scale_inv");
    verbatim(w.qkv_fp8 + size_t(Q + K) * H, p + "self_attn.v_proj.weight");
    verbatim(w.qkv_e8m0 + size_t(Q + K) * H / 32, p + "self_attn.v_proj.weight_scale_inv");
    for (const auto& [name, payload, scales] : {std::tuple{"self_attn.o_proj", w.o_fp8, w.o_e8m0},
                                                {"mlp.gate_proj", w.gate_fp8, w.gate_e8m0},
                                                {"mlp.up_proj", w.up_fp8, w.up_e8m0},
                                                {"mlp.down_proj", w.down_fp8, w.down_e8m0}}) {
      verbatim(payload, p + name + ".weight");
      verbatim(scales, p + name + ".weight_scale_inv");
    }
  }
  verbatim(full.hidden_norm, "hidden_norm.weight");
  verbatim(full.norm, "final_norm.weight");
  verbatim(full.pred_codebook, "markov_head.markov_w1.weight");
  verbatim(full.succ_codebook, "markov_head.markov_w2.weight");
  const auto fc = copy_device(full.fc, c.target_layer_ids.size() * H * H);
  const auto* original = static_cast<const uint16_t*>(source("fc.weight").data);
  for (size_t tap = 0; tap < c.target_layer_ids.size(); ++tap)
    for (int r = 0; r < H; ++r)
      require(std::memcmp(fc.data() + (tap * H + r) * H, original + size_t(r) * c.fc_in() + tap * H,
                          H * 2) == 0,
              "DSpark FC tap column order");
  for (const auto& [ptr, name] : {std::pair{full.confidence_weight, "confidence_head.proj.weight"},
                                  {full.confidence_bias, "confidence_head.proj.bias"}}) {
    const auto& t = source(name);
    const auto values = copy_device(ptr, t.numel());
    for (size_t i = 0; i < t.numel(); ++i)
      require(values[i] == f(static_cast<const uint16_t*>(t.data)[i]),
              "DSpark confidence expands BF16 exactly to FP32");
  }
  files.clear();
  for (int world : {2, 4})
    for (int rank = 0; rank < world; ++rank) {
      auto part = load_minimax_dspark(c, dir, nullptr, rank, world);
      require(part.bytes == minimax_dspark_weights_bytes(c, world),
              "DSpark memory plan reconciles sharded weights");
      const int q = Q / world, k = K / world, i = I / world;
      for (int l = 0; l < c.num_hidden_layers; ++l) {
        const auto& a = part.layers[l];
        const auto& b = full.layers[l];
        int local = 0, global = 0;
        for (auto [nr, fullrows] : {std::pair{q, Q}, {k, K}, {k, K}}) {
          rectangular_slice(a.qkv_fp8 + size_t(local) * H, nr, H, b.qkv_fp8 + size_t(global) * H,
                            fullrows, H, rank * nr, 0);
          rectangular_slice(a.qkv_e8m0 + size_t(local) * H / 32, nr, H / 32,
                            b.qkv_e8m0 + size_t(global) * H / 32, fullrows, H / 32, rank * nr, 0);
          local += nr;
          global += fullrows;
        }
        rectangular_slice(a.o_fp8, H, q, b.o_fp8, H, Q, 0, rank * q);
        rectangular_slice(a.o_e8m0, H, q / 32, b.o_e8m0, H, Q / 32, 0, rank * q / 32);
        for (auto [ap, as, bp, bs] : {std::tuple{a.gate_fp8, a.gate_e8m0, b.gate_fp8, b.gate_e8m0},
                                      {a.up_fp8, a.up_e8m0, b.up_fp8, b.up_e8m0}}) {
          rectangular_slice(ap, i, H, bp, I, H, rank * i, 0);
          rectangular_slice(as, i, H / 32, bs, I, H / 32, rank * i, 0);
        }
        rectangular_slice(a.down_fp8, H, i, b.down_fp8, H, I, 0, rank * i);
        rectangular_slice(a.down_e8m0, H, i / 32, b.down_e8m0, H, I / 32, 0, rank * i / 32);
        require(copy_device(a.input_norm, H) == copy_device(b.input_norm, H),
                "DSpark norm replication");
      }
      require(copy_device(part.fc, fc.size()) == fc, "DSpark FC replication");
      require(copy_device(part.pred_codebook, size_t(c.vocab_size) * c.selector_rank) ==
                  copy_device(full.pred_codebook, size_t(c.vocab_size) * c.selector_rank),
              "DSpark predecessor replication");
      require(copy_device(part.succ_codebook, size_t(c.vocab_size) * c.selector_rank) ==
                  copy_device(full.succ_codebook, size_t(c.vocab_size) * c.selector_rank),
              "DSpark successor replication");
    }
  std::printf(
      "DSpark source bytes, FC columns, FP32 confidence and TP2/TP4 slices exact; full arena %zu "
      "bytes\n",
      full.bytes);
}

void dspark_markov_candidates() {
  constexpr int V = 64, M = 256, S = 2, QR = 8, K = 16;
  std::vector<float> base(S * QR * V, -20.f);
  for (int r = 0; r < S * QR; ++r)
    for (int v = 0; v < K; ++v) base[r * V + v] = 8.f - .25f * v;
  std::vector<uint16_t> pred(V * M, 0), succ(V * M, 0);
  const int predecessors[] = {7, 11, 47, 55}, successors[] = {47, 55, 41, 57};
  for (int i = 0; i < 4; ++i) {
    pred[predecessors[i] * M + i] = b(1);
    succ[successors[i] * M + i] = b(50);
  }
  Device<float> db(base), logits(S * V), scores(S * K);
  Device<uint16_t> dp(pred), ds(succ);
  Device<int64_t> anchors(std::vector<int64_t>{7, 11}), next(S);
  Device<int32_t> ids(S * K), record(S * K), tokens(S);
  for (int step = 0; step < 2; ++step) {
    dsv41_dspark_markov_bias(db.p, QR * V, step + 1, dp.p, ds.p, M, 0, V, step ? next.p : anchors.p,
                             1, S, logits.p, V, 1, nullptr);
    const auto got = logits.get();
    for (int slot = 0; slot < S; ++slot)
      for (int v = 0; v < V; ++v) {
        const int wanted = successors[step * S + slot];
        require(got[slot * V + v] == base[(slot * QR + step + 1) * V + v] + (v == wanted ? 50 : 0),
                "full-vocabulary Markov projection");
      }
    dflash2_topk_f32(logits.p, ids.p, scores.p, V, S, K, nullptr, nullptr, 0);
    for (int slot = 0; slot < S; ++slot)
      dflash2_sample_candidates(ids.p + slot * K, scores.p + slot * K, K, step, tokens.p + slot,
                                next.p + slot, record.p + slot * K, nullptr);
    const auto chosen = tokens.get();
    const auto prev = next.get();
    require(record.get() == ids.get(), "corrected candidates retained for diagnostics");
    for (int slot = 0; slot < S; ++slot)
      require(chosen[slot] == successors[step * S + slot] && prev[slot] == chosen[slot],
              "Markov winner outside the base top-16 or chained predecessor was lost");
  }
}
void dspark_kernels() {
  dspark_markov_candidates();
  constexpr int R = 7, H = 640, M = 256, V = 32;
  std::vector<float> x(R * H), weight(H + M);
  std::vector<uint16_t> norm(H), rounded(x.size()), want(x.size()), hidden(x.size()), pred(V * M);
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = std::sin(float(i) * .21f) * 2.017f;
    rounded[i] = b(x[i]);
    hidden[i] = b(std::cos(float(i) * .31f));
  }
  for (int i = 0; i < H; ++i) norm[i] = b(std::sin(float(i)) * .13f);
  for (int i = 0; i < H + M; ++i) weight[i] = std::cos(float(i) * .13f) * .04f;
  for (size_t i = 0; i < pred.size(); ++i) pred[i] = b(std::sin(float(i) * .11f));
  for (int r = 0; r < R; ++r) {
    double ss = 0;
    for (int d = 0; d < H; ++d) ss += double(f(rounded[r * H + d])) * f(rounded[r * H + d]);
    const float inv = 1 / std::sqrt(float(ss / H) + 1e-6f);
    for (int d = 0; d < H; ++d) want[r * H + d] = b(f(rounded[r * H + d]) * inv * (1 + f(norm[d])));
  }
  Device<float> dx(x), dw(weight), db(std::vector<float>{.31f}), dc(R);
  Device<uint16_t> dn(norm), dy(x.size()), dh(hidden), dp(pred);
  Device<int64_t> anchor(std::vector<int64_t>{9});
  const std::vector<int32_t> tokens{3, 8, 2, 13, 18, 1, 6};
  Device<int32_t> dt(tokens);
  minimax_context_norm(dx.p, dn.p, dy.p, R, H, 1e-6f, nullptr);
  near(dy.get(), want, .0001, "DSpark FC rounds before Gemma norm");
  minimax_dspark_confidence(dh.p, dp.p, anchor.p, dt.p, dw.p, db.p, dc.p, R, H, M, V, nullptr);
  const auto confidence = dc.get();
  for (int r = 0; r < R; ++r) {
    double dot = .31f;
    for (int i = 0; i < H; ++i) dot += double(f(hidden[r * H + i])) * weight[i];
    const int previous = r == 0 ? 9 : tokens[r - 1];
    for (int i = 0; i < M; ++i) dot += double(f(pred[previous * M + i])) * weight[H + i];
    require(std::abs(confidence[r] - dot) < 2e-6, "DSpark learned predecessor confidence logit");
  }
  // Exercise the producer/consumer boundary. Applying sigmoid in both places
  // would turn a low-confidence draft into a roughly 50%-confidence one.
  dw.upload(std::vector<float>(H + M, 0));
  for (auto [bias, expected_depth] : {std::pair{-4.f, 0}, {0.f, 2}, {4.f, 7}}) {
    db.upload(std::vector<float>{bias});
    minimax_dspark_confidence(dh.p, dp.p, anchor.p, dt.p, dw.p, db.p, dc.p, R, H, M, V, nullptr);
    const auto logits = dc.get();
    require(scheduled_verify_depth(logits.data(), R, 1.f, .2f) == expected_depth,
            "DSpark confidence must cross into the shared schedule as logits");
  }
  anchor.upload(std::vector<int64_t>{-1});
  minimax_dspark_confidence(dh.p, dp.p, anchor.p, dt.p, dw.p, db.p, dc.p, R, H, M, V, nullptr);
  const auto invalid = dc.get();
  require(std::isinf(invalid[0]) && invalid[0] < 0 &&
              scheduled_verify_depth(invalid.data(), R, 1.f, .2f) == 0,
          "invalid DSpark predecessor has zero acceptance probability");
  // The native 16:1 GQA path, a noncontiguous page table, window and
  // padded query rows against a full FP64 softmax reference.
  constexpr int QH = 16, KH = 1, HD = 128, QR = 8, SLOTS = 384, BPR = 3;
  const int64_t start = 257, window = 129;
  const std::vector<int32_t> table{2, 0, 1};
  std::vector<int64_t> pos(QR);
  std::vector<uint16_t> q(QR * QH * HD), k(SLOTS * KH * HD), v(k.size()), expected(q.size());
  for (int r = 0; r < QR; ++r) pos[r] = r < 6 ? start + r : -1;
  for (size_t i = 0; i < q.size(); ++i) q[i] = b(std::sin(float(i) * .18f));
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = b(std::cos(float(i) * .07f));
    v[i] = b(std::sin(float(i) * .31f));
  }
  for (int r = 0; r < 6; ++r)
    for (int h = 0; h < QH; ++h) {
      std::vector<double> scores;
      for (int64_t p = std::max<int64_t>(0, pos[r] - window + 1); p < start + 6; ++p) {
        const auto at = (table[p / 128] * 128 + p % 128) * HD;
        double dot = 0;
        for (int d = 0; d < HD; ++d) dot += double(f(q[(r * QH + h) * HD + d])) * f(k[at + d]);
        scores.push_back(dot / std::sqrt(128.0));
      }
      const double mx = *std::max_element(scores.begin(), scores.end());
      double den = 0;
      for (double& score : scores) {
        score = std::exp(score - mx);
        den += score;
      }
      for (int d = 0; d < HD; ++d) {
        double value = 0;
        size_t i = 0;
        for (int64_t p = std::max<int64_t>(0, pos[r] - window + 1); p < start + 6; ++p)
          value += scores[i++] * f(v[(table[p / 128] * 128 + p % 128) * HD + d]);
        expected[(r * QH + h) * HD + d] = b(value / den);
      }
    }
  Device<uint16_t> dq(q), dk(k), dv(v), out(q.size());
  Device<int32_t> tables(table);
  Device<int64_t> positions(pos);
  Device<float> partial(dflash2_block_attn_partials_bytes(QR, QH) / sizeof(float));
  dflash2_block_attn_split(dq.p, QH * HD, dk.p, dv.p, tables.p, 128, BPR, QR, window, positions.p,
                           QR, QH, KH, HD, 1 / std::sqrt(128.0f), partial.p, out.p, nullptr, true);
  near(out.get(), expected, .0002, "DSpark FP32-P 16:1 block attention");
}
void rotary_frequencies() {
  // Independent CUDA PyTorch float32 recipe at the published theta.
  // Double pow -> float differs in 14 of these 32 entries and drifts by
  // up to 0.03125 radians at the million-token position limit.
  constexpr uint32_t expected[] = {
      0x3f800000u, 0x3f1ab32bu, 0x3ebaf81bu, 0x3e61f835u, 0x3e088d77u, 0x3da50956u, 0x3d47763fu,
      0x3cf11177u, 0x3c91ad39u, 0x3c301052u, 0x3bd4ca15u, 0x3b80967du, 0x3b1b690du, 0x3abbd3edu,
      0x3a6301e2u, 0x3a092e02u, 0x39a5cb60u, 0x394860c1u, 0x38f22ce2u, 0x3892587eu, 0x3830df52u,
      0x37d5c441u, 0x37812dabu, 0x371c1fc4u, 0x36bcb0c1u, 0x36640cc6u, 0x3609cf4au, 0x35a68e4cu,
      0x35494c56u, 0x34f3499du, 0x3493048eu, 0x3431af44u};
  // Exercise both the target's half table and the draft's overallocated
  // full-head table. The used frequencies must be identical.
  for (int count : {32, 64}) {
    Device<float> table(count);
    minimax_rope_inv_freq(table.p, count, 64, 10000000, nullptr);
    const auto values = table.get();
    for (int i = 0; i < 32; ++i)
      require(std::bit_cast<uint32_t>(values[i]) == expected[i],
              "MiniMax CUDA rotary frequency rounding");
  }
}
void index_score_schedule() {
  constexpr int blocks = 257, heads = 4, requests = 3, topk = 16;
  std::vector<int32_t> table(requests * blocks);
  for (int req = 0; req < requests; ++req)
    for (int i = 0; i < blocks; ++i) table[req * blocks + i] = req * blocks + (i * 29) % blocks;
  std::vector<uint16_t> keys(size_t(requests) * blocks * 128 * 128);
  for (size_t i = 0; i < keys.size(); ++i)
    keys[i] = b(float(int((i * 17 + i / 131) % 131) - 65) / 32);
  Device<uint16_t> dk(keys);
  Device<int32_t> dt(table);
  const int64_t boundaries[]{0, 1, 127, 128, 2047, 2048, 32767, 32895, -1};
  for (bool mixed : {false, true})
    for (int rows : {1, 8, 17, 64, 129}) {
      std::vector<int64_t> pos(rows);
      std::vector<int32_t> req(rows);
      std::vector<uint16_t> query(size_t(rows) * heads * 128);
      for (size_t i = 0; i < query.size(); ++i)
        query[i] = b(float(int((i * 13 + i / 43) % 97) - 48) / 16);
      for (int r = 0; r < rows; ++r) {
        pos[r] = boundaries[r % 9];
        req[r] = pos[r] < 0 ? -1 : mixed ? r % requests : 0;
      }
      Device<int64_t> dp(pos);
      Device<int32_t> dr(req), selected(size_t(rows) * heads * topk);
      Device<uint16_t> dq(query);
      Device<float> original(size_t(rows) * heads * blocks), bounded(original.size);
      cudaStream_t stream;
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      DGPP_CUDA_OK(cudaStreamCreate(&stream));
      DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      minimax_index_score(dq.p, dk.p, dr.p, dp.p, dt.p, blocks, rows, heads, blocks, bounded.p,
                          stream);
      minimax_index_select(bounded.p, dp.p, rows, heads, blocks, topk, selected.p, stream);
      DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
      DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      // Reuse the graph and score storage as visibility grows, shrinks and
      // becomes padding. Previously visible future scores must never leak.
      for (int pass = 0; pass < 4; ++pass) {
        if (pass == 3) std::fill(pos.begin(), pos.end(), -1);
        dp.upload(pos);
        dr.upload(req);
        DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        const auto got = bounded.get();
        const auto got_ids = selected.get();
        minimax_index_score(dq.p, dk.p, dr.p, dp.p, dt.p, blocks, rows, heads, blocks, original.p,
                            stream, MinimaxIndexScan::Reference);
        minimax_index_select(original.p, dp.p, rows, heads, blocks, topk, selected.p, stream);
        DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        const auto expected = original.get();
        require(got_ids == selected.get(), "bounded index scan changed causal block selection");
        for (int r = 0; r < rows; ++r)
          for (int h = 0; h < heads; ++h)
            for (int block = 0; block <= pos[r] / 128 && pos[r] >= 0; ++block) {
              const size_t at = (size_t(r) * heads + h) * blocks + block;
              require(std::bit_cast<uint32_t>(got[at]) == std::bit_cast<uint32_t>(expected[at]),
                      "bounded index scan changed a visible FP32 score");
            }
        for (int r = 0; r < rows; ++r) {
          pos[r] = boundaries[(r + pass * 3 + 2) % 9];
          req[r] = pos[r] < 0 ? -1 : mixed ? (r + pass) % requests : pass % requests;
        }
      }
      DGPP_CUDA_OK(cudaGraphExecDestroy(exec));
      DGPP_CUDA_OK(cudaGraphDestroy(graph));
      DGPP_CUDA_OK(cudaStreamDestroy(stream));
    }
}
void kernels() {
  rotary_frequencies();
  index_score_schedule();
  minimax_pdl_test();
  dspark_kernels();
  {
    std::vector<MoeSegment> segments(128);
    for (int e : {0, 7, 32, 127}) segments[e] = MoeSegment{e * 3, 1, e};
    Device<MoeSegment> source(segments), compact(4);
    minimax_compact_segments(source.p, compact.p, 128, 4, nullptr);
    auto got = compact.get();
    int at = 0;
    for (const auto& s : segments)
      if (s.rows) {
        require(got[at].row0 == s.row0 && got[at].rows == s.rows && got[at].expert == s.expert,
                "segment compaction preserves row offsets and expert order");
        ++at;
      }
    segments[7].rows = segments[127].rows = 0;
    source.upload(segments);
    minimax_compact_segments(source.p, compact.p, 128, 4, nullptr);
    got = compact.get();
    require(got[0].expert == 0 && got[1].expert == 32 && got[2].rows == 0 && got[3].rows == 0,
            "segment compaction clears unused entries when the route set shrinks");
  }
  {
    const int rows = 3, hidden = 256;
    std::vector<uint16_t> x(rows * hidden), add(x.size()), weight(hidden), expected(x.size()),
        residual(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      x[i] = b(std::sin(float(i) * .21f));
      add[i] = b(std::cos(float(i) * .37f) * .3f);
    }
    for (int d = 0; d < hidden; ++d) weight[d] = b(std::sin(float(d)) * .1f);
    for (int r = 0; r < rows; ++r) {
      double sum = 0;
      for (int d = 0; d < hidden; ++d) {
        const float v = f(x[r * hidden + d]) + f(add[r * hidden + d]);
        sum += double(v) * v;
        residual[r * hidden + d] = b(v);
      }
      const float inv = 1 / std::sqrt(float(sum / hidden) + 1e-6f);
      for (int d = 0; d < hidden; ++d)
        expected[r * hidden + d] =
            b((f(x[r * hidden + d]) + f(add[r * hidden + d])) * inv * (1 + f(weight[d])));
    }
    Device<uint16_t> dx(x), da(add), dw(weight), dy(x.size());
    minimax_add_norm(dx.p, da.p, dw.p, dy.p, rows, hidden, 1e-6f, nullptr);
    require(dx.get() == residual, "fused norm residual must round exactly once");
    near(dy.get(), expected, .0001, "FP32 residual Gemma norm");
  }
  // Activation extremes distinguish all three parts of SwiGLU-OAI.
  std::vector<uint16_t> g(513), u(g.size()), want(g.size());
  for (size_t i = 0; i < g.size(); ++i) {
    g[i] = b(float(int(i % 51) - 25) * .5f);
    u[i] = b(float(int(i % 37) - 18) * .8f);
    want[i] = b(minimax_ref::swiglu(f(g[i]), f(u[i])));
  }
  Device<uint16_t> dg(g), du(u), dy(g.size());
  minimax_swiglu(dg.p, du.p, dy.p, g.size(), 1.702f, 7, nullptr);
  near(dy.get(), want, .001, "SwiGLU-OAI");
  // Partial RoPE includes the context ceiling and padding. In-place must
  // be bitwise identical to out-of-place, including the nonrotary tail.
  const int R = 7, heads = 4;
  std::vector<int64_t> positions{0, 1, 127, 128, 2049, 1048575, -1};
  std::vector<uint16_t> x(R * heads * 128), w(128), gold(x.size());
  for (size_t i = 0; i < x.size(); ++i) x[i] = b(std::sin(float(i) * .13f) * 2);
  for (int d = 0; d < 128; ++d) w[d] = b(std::cos(float(d)) * .1f);
  std::vector<float> inv(32);
  for (int d = 0; d < 32; ++d) inv[d] = static_cast<float>(1 / std::pow(1e7, double(2 * d) / 64));
  for (int r = 0; r < R - 1; ++r)
    for (int h = 0; h < heads; ++h)
      minimax_ref::norm_rope(x.data() + (r * heads + h) * 128, w.data(),
                             gold.data() + (r * heads + h) * 128, 128, 64, 1e-6f, 1e7,
                             positions[r]);
  Device<uint16_t> dx(x), dw(w), do_(x.size());
  Device<int64_t> dp(positions);
  Device<float> di(inv);
  minimax_norm_rope(dx.p, dw.p, do_.p, dp.p, di.p, R, heads, 64, 1e-6f, nullptr);
  const auto separate = do_.get();
  near(separate, gold, .002, "Gemma norm and partial RoPE");
  minimax_norm_rope(dx.p, dw.p, dx.p, dp.p, di.p, R, heads, 64, 1e-6f, nullptr);
  require(dx.get() == separate, "in-place norm/RoPE");
  {
    const int stride = (heads + 2) * 128;
    std::vector<uint16_t> strided(R * stride, b(99));
    for (int r = 0; r < R; ++r)
      std::copy_n(x.data() + r * heads * 128, heads * 128, strided.data() + r * stride + 128);
    Device<uint16_t> ds(strided);
    minimax_norm_rope(ds.p + 128, dw.p, ds.p + 128, dp.p, di.p, R, heads, 64, 1e-6f, nullptr,
                      stride, stride);
    const auto got = ds.get();
    for (int r = 0; r < R; ++r) {
      require(
          std::equal(got.begin() + r * stride + 128, got.begin() + r * stride + (heads + 1) * 128,
                     separate.begin() + r * heads * 128),
          "strided QKV norm/RoPE exact");
      for (int d = 0; d < 128; ++d)
        require(got[r * stride + d] == b(99) && got[(r + 1) * stride - 128 + d] == b(99),
                "strided norm/RoPE guard rows");
    }
  }

  // Logical pages are deliberately noncontiguous; requests have different
  // permutations, four index heads choose different block lists.
  const int B = 18, N = 4, IH = 4, QH = 8, KH = 4, TOP = 3, SLOTS = 2 * B * 128;
  std::vector<int32_t> table(2 * B), req{0, 1, 0, -1};
  for (int i = 0; i < B; ++i) {
    table[i] = (i * 5) % B;
    table[B + i] = B + (i * 7) % B;
  }
  std::vector<int64_t> pos{0, 127, 2176, -1};
  std::vector<uint16_t> iq(N * IH * 128, b(0)), ik(SLOTS * 128, b(0)), q(N * QH * 128),
      k(SLOTS * KH * 128), v(k.size());
  for (int r = 0; r < N; ++r)
    for (int h = 0; h < IH; ++h) iq[(r * IH + h) * 128 + h] = b(1);
  for (int request = 0; request < 2; ++request)
    for (int block = 0; block < B; ++block)
      for (int t = 0; t < 128; ++t)
        for (int h = 0; h < IH; ++h)
          ik[(table[request * B + block] * 128 + t) * 128 + h] =
              b(float((block * (h + 3) + t % 13) % 31) - 15);
  for (size_t i = 0; i < q.size(); ++i) q[i] = b(std::sin(float(i) * .19f) * .3f);
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = b(std::cos(float(i) * .013f) * .3f);
    v[i] = b(std::sin(float(i) * .021f));
  }
  Device<uint16_t> dIq(iq), dIk(ik), dQ(q), dK(k), dV(v), dOut(N * QH * 128);
  Device<int32_t> dTable(table), dReq(req), dSelected(N * IH * TOP);
  Device<int64_t> dPos(pos);
  Device<float> dScore(N * IH * B), dM(N * TOP * QH), dL(N * TOP * QH), dC(N * TOP * QH * 128);
  minimax_index_score(dIq.p, dIk.p, dReq.p, dPos.p, dTable.p, B, N, IH, B, dScore.p, nullptr);
  minimax_index_select(dScore.p, dPos.p, N, IH, B, TOP, dSelected.p, nullptr);
  const auto selected = dSelected.get();
  std::vector<uint16_t> expected(N * QH * 128, b(0));
  for (int r = 0; r < N; ++r) {
    std::vector<uint16_t> logical(B * 128 * 128), lk(B * 128 * KH * 128), lv(lk.size());
    if (pos[r] >= 0)
      for (int t = 0; t < B * 128; ++t) {
        const int physical = table[req[r] * B + t / 128] * 128 + t % 128;
        std::copy_n(ik.data() + physical * 128, 128, logical.data() + t * 128);
        std::copy_n(k.data() + physical * KH * 128, KH * 128, lk.data() + t * KH * 128);
        std::copy_n(v.data() + physical * KH * 128, KH * 128, lv.data() + t * KH * 128);
      }
    for (int h = 0; h < IH; ++h) {
      const auto ids =
          minimax_ref::select_blocks(iq.data() + (r * IH + h) * 128, logical.data(), pos[r], TOP);
      for (int j = 0; j < TOP; ++j)
        require(selected[(r * IH + h) * TOP + j] == (j < int(ids.size()) ? ids[j] : -1),
                "paged causal MSA block selection");
      if (pos[r] < 0) continue;
      for (int head = h * (QH / IH); head < (h + 1) * (QH / IH); ++head) {
        const auto result = minimax_ref::attention(q.data() + r * QH * 128, lk.data(), lv.data(),
                                                   head, QH, KH, pos[r], ids);
        for (int d = 0; d < 128; ++d) expected[(r * QH + head) * 128 + d] = b(result[d]);
      }
    }
  }
  minimax_attn_partial(dQ.p, dK.p, dV.p, dReq.p, dPos.p, dTable.p, B, dSelected.p, N, QH, KH, TOP,
                       dM.p, dL.p, dC.p, nullptr);
  sink_attn_combine(dM.p, dL.p, dC.p, N, TOP, QH, dOut.p, nullptr);
  near(dOut.get(), expected, .004, "Paged per-head sparse attention");
  // A graph replay must not retain scores/selection from a previous position.
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  minimax_index_score(dIq.p, dIk.p, dReq.p, dPos.p, dTable.p, B, N, IH, B, dScore.p, stream);
  minimax_index_select(dScore.p, dPos.p, N, IH, B, TOP, dSelected.p, stream);
  DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graph));
  DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  require(dSelected.get() == selected, "index graph replay");
  pos = {-1, -1, -1, -1};
  dPos.upload(pos);
  DGPP_CUDA_OK(cudaGraphLaunch(exec, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  for (int id : dSelected.get()) require(id == -1, "padding graph retains no selections");
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);

  // FP32 router low bits determine an otherwise tied selection.
  std::vector<uint16_t> rx(128, b(0));
  rx[0] = b(1);
  std::vector<float> rw(4 * 128, 0), bias(4, 0), logits{.5001f, .5002f, -1, -2};
  for (int e = 0; e < 4; ++e) rw[e * 128] = logits[e];
  Device<uint16_t> dRx(rx);
  Device<float> dRw(rw), dBias(bias), dRs(4), dWeights(1);
  Device<int32_t> dIds(1);
  minimax_router(dRx.p, dRw.p, dBias.p, 1, 128, 4, 1, 2, dRs.p, dIds.p, dWeights.p, nullptr);
  require(dIds.get()[0] == 1 && dWeights.get()[0] == 2, "router must retain FP32 low bits");
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::puts("MiniMax kernel gates passed");
}
template <class T>
void write(const std::filesystem::path& p, const std::vector<T>& x) {
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(x.data()), x.size() * sizeof(T));
  if (!f) throw std::runtime_error("cannot write " + p.string());
}
void dspark_oracle(const std::string& path, const std::string& dump, bool native) {
  const auto cfg = MinimaxConfig::from_json_file(path + "/config.json");
  require(cfg.dspark != nullptr && !dump.empty(), "DSpark oracle needs model and dump directory");
  std::vector<int64_t> ids{3, 12, 25, 7, 91, 17, 6, 7, 47, 3, 19, 83, 71, 8, 12, 9, 31};
  MinimaxModel::Outputs plain;
  {
    MinimaxModel target(cfg, path, 256, 512, LoaderResidency::Resident, nullptr, 0, 1, 1,
                         false, 16, LatentFormat::kBf16, false, native);
    plain = target.forward(ids, true);
  }
  MinimaxModel model(cfg, path, 256, 512, LoaderResidency::Resident, nullptr, 0, 1, 1, true, 16,
                       LatentFormat::kBf16, false, native);
  (void)model.session_prefill(0, ids);
  std::vector<int32_t> proposals;
  require(model.dflash2_draft(0, 7, &proposals), "DSpark oracle draft");
  DGPP_CUDA_OK(cudaStreamSynchronize(model.stream()));
  const auto& df = model.dspark_drafter();
  const auto& c = df.cfg();
  const size_t H = c.hidden_size, T = ids.size(), R = c.query_rows();
  const std::filesystem::path dir(dump);
  std::filesystem::create_directories(dir);
  for (size_t l = 0; l < plain.layer_states.size(); ++l)
    write(dir / ("target-layer-" + std::to_string(l) + ".bf16"), plain.layer_states[l]);
  write(dir / "tokens.i64", ids);
  write(dir / "features.f32", copy_device(df.acc(), T * H));
  write(dir / "context.bf16", copy_device(df.context_norm(), T * H));
  write(dir / "hidden.bf16", copy_device(df.block_hidden(), R * H));
  write(dir / "logits.f32", copy_device(df.block_logits(), R * c.vocab_size));
  write(dir / "proposals.i32", proposals);
  write(dir / "confidence.f32", copy_device(df.confidence(), c.drafts()));
  const auto* table = model.pool().request_table_row(0);
  for (int l = 0; l < c.num_hidden_layers; ++l) {
    const auto plane = model.pool().draft_view(l);
    for (const auto& [ptr, name] : {std::pair{plane.k, "k"}, {plane.v, "v"}}) {
      std::vector<uint16_t> logical((T + R) * c.kv_row());
      for (size_t r = 0; r < T + R; ++r) {
        const size_t physical = size_t(table[r / 128]) * 128 + r % 128;
        DGPP_CUDA_OK(cudaMemcpy(logical.data() + r * c.kv_row(), ptr + physical * c.kv_row(),
                                c.kv_row() * 2, cudaMemcpyDeviceToHost));
      }
      write(dir / (std::string(name) + "-" + std::to_string(l) + ".bf16"), logical);
    }
  }
  model.session_close(0);
}
void smoke(const std::string& path, const std::string& dump, bool native, LatentFormat format) {
  const auto cfg = MinimaxConfig::from_json_file(path + "/config.json");
  loader_slices(cfg, path);
  require(cfg.hidden_size <= 1024, "smoke is a synthetic-fixture gate");
  std::vector<int64_t> tokens(385);
  for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = (i * 17 + 3) % cfg.vocab_size;
  MinimaxModel m(cfg, path, 512, 1536, LoaderResidency::Resident, nullptr, 0, 1, 3, false, 0,
                   format, false, native);
  require(m.kv_format() == format, "model reports its actual cache format");
  size_t block_bytes = 0;
  for (const auto& plane : m.kv_pool().planes()) block_bytes += plane.block_bytes;
  require(block_bytes == MinimaxModel::kv_block_bytes_static(cfg, 0, 1, format),
          "native cache plane sizes match the NVMe/prefix memory plan");
  const auto cache = m.kv_pool().view(0);
  require(format == LatentFormat::kBf16 ? cache.k_cache && !cache.k_nvfp4 : cache.k_nvfp4 && !cache.k_cache,
          "only the configured cache pointers are exposed");
  m.graph_prepare();
  m.set_decode_row_traces(true);
  if (!dump.empty()) {
    std::filesystem::create_directories(dump);
    m.set_observer([&](int layer, std::string_view stage, const uint16_t* data, int rows,
                       cudaStream_t stream) {
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      write(std::filesystem::path(dump) /
                (std::string(stage) + "-" + std::to_string(layer) + ".bf16"),
            copy_device(data, static_cast<size_t>(rows) * cfg.hidden_size));
    });
  }
  auto first = m.forward(tokens, true), second = m.forward(tokens, true);
  require(first.logits == second.logits && first.layer_states == second.layer_states,
          "repeated forward must be bitwise identical");
  require(first.logits == m.forward(tokens, false).logits,
          "diagnostic layer capture must not change arithmetic");
  m.set_observer({});
  std::vector<uint16_t> decoded_hidden;
  {
    auto decoded = m.session_prefill(0, {tokens.front()});
    require(std::equal(decoded.final_hidden_bits.begin(), decoded.final_hidden_bits.end(),
                       first.final_hidden_bits.begin()),
            "first decode hidden differs from full prefill");
    decoded_hidden = decoded.final_hidden_bits;
    for (size_t at = 1; at < tokens.size();) {
      const int rows = std::min<size_t>(8, tokens.size() - at);
      decoded = m.session_verify(0, {tokens.begin() + at, tokens.begin() + at + rows});
      if (format != LatentFormat::kFp4)
        require(std::equal(decoded.final_hidden_bits.begin(), decoded.final_hidden_bits.end(),
                           first.final_hidden_bits.begin() + at * cfg.hidden_size),
                "incremental verification hidden differs from full prefill at " + std::to_string(at));
      decoded_hidden.insert(decoded_hidden.end(), decoded.final_hidden_bits.begin(), decoded.final_hidden_bits.end());
      m.session_rollback(0, rows, rows);
      at += rows;
    }
    m.session_close(0);
  }
  // The native cache has distinct published phase formulas. Verify decode
  // chunk invariance against its own trajectory, retaining the cold-prefill
  // equality gate for the BF16 cache.
  for (int chunk : {1, 4}) {
    (void)m.session_prefill(0, {tokens.front()});
    for (size_t at = 1; at < tokens.size();) {
      const int rows = std::min<size_t>(chunk, tokens.size() - at);
      const auto decoded = m.session_verify(0, {tokens.begin() + at, tokens.begin() + at + rows});
      require(std::equal(decoded.final_hidden_bits.begin(), decoded.final_hidden_bits.end(),
                         decoded_hidden.begin() + at * cfg.hidden_size),
              "verification hidden changes with row geometry at " + std::to_string(at));
      m.session_rollback(0, rows, rows);
      at += rows;
    }
    m.session_close(0);
  }
  // Reuse the complete target graph while feeds and positions change across
  // cache pages. This also exercises both grouped expert GEMMs under capture.
  {
    (void)m.session_prefill(0, {tokens.front()});
    m.session_reserve_blocks(0, tokens.size());
    m.session_graph_prepare();
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    DGPP_CUDA_OK(cudaStreamBeginCapture(m.stream(), cudaStreamCaptureModeThreadLocal));
    m.session_graph_capture_step(0, std::vector<int64_t>(tokens.begin() + 1, tokens.begin() + 9));
    DGPP_CUDA_OK(cudaStreamEndCapture(m.stream(), &graph));
    DGPP_CUDA_OK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (size_t at = 1; at < tokens.size(); at += 8) {
      m.session_graph_stage(0, std::vector<int64_t>(tokens.begin() + at, tokens.begin() + at + 8));
      DGPP_CUDA_OK(cudaGraphLaunch(exec, m.stream()));
      DGPP_CUDA_OK(cudaStreamSynchronize(m.stream()));
      const auto replay = m.session_graph_collect(0);
      require(std::equal(replay.final_hidden_bits.begin(), replay.final_hidden_bits.end(),
                         decoded_hidden.begin() + at * cfg.hidden_size),
              "target graph hidden differs from eager decode at " + std::to_string(at));
    }
    DGPP_CUDA_OK(cudaGraphExecDestroy(exec));
    DGPP_CUDA_OK(cudaGraphDestroy(graph));
    m.session_close(0);
  }
  for (float x : first.logits) require(std::isfinite(x), "finite logits");
  if (!dump.empty()) {
    std::filesystem::create_directories(dump);
    write(std::filesystem::path(dump) / "logits.f32", first.logits);
    int ordinal = 0;
    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
      write(std::filesystem::path(dump) / ("layer-" + std::to_string(l) + ".bf16"),
            first.layer_states[l]);
      if (cfg.is_moe_layer(l))
        write(std::filesystem::path(dump) / ("routes-" + std::to_string(l) + ".i32"),
              first.route_ids[ordinal++]);
    }
    std::ofstream meta(std::filesystem::path(dump) / "metadata.json");
    meta << "{\"residual_norm\":\"fp32\",\"rope_inv_freq\":\"cuda-fp32\",\"activations\":\"" << (native ? "native" : "bf16")
         << "\",\"kv_dtype\":\"" << latent_format_name(format) << "\",\"tokens\":[";
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (i) meta << ',';
      meta << tokens[i];
    }
    meta << "]}\n";
  }
  // Prefix reuse copies a partial block's main K, V and index K planes.
  Device<uint8_t> snapshot(m.session_snapshot_bytes());
  MinimaxModel::SessionSnapshotMeta meta;
  MinimaxModel::SnapshotRequest request;
  request.position = 129;
  request.dst = snapshot.p;
  request.meta = &meta;
  const auto cold = m.session_prefill(0, tokens, {129}, &request);
  require(request.taken, "partial-block prefix snapshot");
  const auto cold_step = m.session_step(0, 7);
  m.session_close(0);
  m.session_attach(1, snapshot.p, meta);
  const auto hot =
      m.session_prefill_resume(1, std::vector<int64_t>(tokens.begin() + 129, tokens.end()), {129});
  require(cold.logits == hot.logits, "hot prefix must equal the same cold chunks bitwise");
  const auto hot_step = m.session_step(1, 7);
  require(cold_step.logits == hot_step.logits, "index cache survives prefix copy");
  m.session_close(1);
  m.session_release_snapshot(meta);
  require(m.kv_blocks_in_use() == 0, "closed requests and released prefix leave no blocks");
  // A conversation prefix can contain both prefilled and generated tokens.
  // Its restore must preserve that history exactly; a wholly cold prefill
  // uses different upstream phase arithmetic in the native KV4 recipe.
  (void)m.session_prefill(0, {tokens.begin(), tokens.begin() + 129});
  for (size_t at = 129; at < 257; at += 8) {
    (void)m.session_verify(0, {tokens.begin() + at, tokens.begin() + at + 8});
    m.session_rollback(0, 8, 8);
  }
  meta = m.session_snapshot(0, snapshot.p);
  const std::vector<int64_t> suffix(tokens.begin() + 257, tokens.end());
  const auto live_turn = m.session_prefill_resume(0, suffix, {257});
  const auto live_step = m.session_step(0, 7);
  m.session_close(0);
  m.session_attach(1, snapshot.p, meta);
  const auto restored_turn = m.session_prefill_resume(1, suffix, {257});
  require(live_turn.logits == restored_turn.logits,
          "generated-prefix restore preserves the live conversation's prefill");
  const auto restored_step = m.session_step(1, 7);
  require(live_step.logits == restored_step.logits,
          "generated-prefix restore preserves the following decode");
  m.session_close(1);
  m.session_release_snapshot(meta);
  require(m.kv_blocks_in_use() == 0, "generated-prefix snapshots release all blocks");
  // Grouped requests exercise distinct cache rows at different lengths.
  std::vector<int64_t> a(tokens.begin(), tokens.begin() + 129),
      b_(tokens.begin() + 17, tokens.begin() + 34);
  const auto solo_a = m.session_prefill(0, a);
  m.session_close(0);
  const auto solo_b = m.session_prefill(1, b_);
  m.session_close(1);
  const auto group = m.session_prefill_group({0, 1}, {&a, &b_});
  auto relative = [](const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, n = 0;
    require(a.size() == b.size(), "logits shapes");
    for (size_t i = 0; i < a.size(); ++i) {
      d += double(a[i] - b[i]) * (a[i] - b[i]);
      n += double(a[i]) * a[i];
    }
    return std::sqrt(d / std::max(n, 1e-30));
  };
  require(relative(solo_a.logits, group[0].logits) < .01 &&
              relative(solo_b.logits, group[1].logits) < .01,
          "grouped request isolation");
  m.session_close(0);
  m.session_close(1);
  std::puts("MiniMax forward/session smoke passed");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    std::string checkpoint, dump, fixture, config, dspark, dspark_model;
    bool run_kernels = false, native = false;
    LatentFormat format = LatentFormat::kBf16;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--kernels")
        run_kernels = true;
      else if (a == "--native-activations")
        native = true;
      else if (a == "--native-cache")
        format = LatentFormat::kFp4;
      else if (a == "--dspark-model-dir" && i + 1 < argc)
        dspark_model = argv[++i];
      else if (a == "--dspark-dir" && i + 1 < argc)
        dspark = argv[++i];
      else if (a == "--write-fixture" && i + 1 < argc)
        fixture = argv[++i];
      else if (a == "--config" && i + 1 < argc)
        config = argv[++i];
      else if ((a == "--checkpoint-dir" || a == "--dump") && i + 1 < argc) {
        const auto v = std::string(argv[++i]);
        if (a == "--dump")
          dump = v;
        else
          checkpoint = v;
      } else
        throw std::runtime_error("invalid argument: " + a);
    }
    if (!fixture.empty()) {
      require(!config.empty(), "--write-fixture requires --config");
      minimaxfx::write(fixture, config);
      return 0;
    }
    if (!dspark_model.empty()) dspark_oracle(dspark_model, dump, native);
    if (!dspark.empty()) dspark_loader(dspark);
    if (run_kernels) kernels();
    if (!checkpoint.empty()) smoke(checkpoint, dump, native, format);
    if (!run_kernels && checkpoint.empty() && dspark.empty() && dspark_model.empty())
      throw std::runtime_error("use --kernels or --checkpoint-dir");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "minimax_test: %s\n", e.what());
    return 1;
  }
}
