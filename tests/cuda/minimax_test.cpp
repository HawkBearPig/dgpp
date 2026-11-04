#include "kernels/minimax.hpp"

#include <algorithm>
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
#include "kernels/sink_attn.hpp"
#include "minimax_fixture.hpp"
#include "models/minimax/forward.hpp"
#include "models/minimax/reference.hpp"

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
void kernels() {
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
void smoke(const std::string& path, const std::string& dump) {
  const auto cfg = MinimaxConfig::from_json_file(path + "/config.json");
  loader_slices(cfg, path);
  require(cfg.hidden_size <= 1024, "smoke is a synthetic-fixture gate");
  std::vector<int64_t> tokens(385);
  for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = (i * 17 + 3) % cfg.vocab_size;
  MinimaxModel m(cfg, path, 512, 1536, LoaderResidency::Resident, nullptr, 0, 1, 3);
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
  {
    auto decoded = m.session_prefill(0, {tokens.front()});
    require(std::equal(decoded.final_hidden_bits.begin(), decoded.final_hidden_bits.end(),
                       first.final_hidden_bits.begin()),
            "first decode hidden differs from full prefill");
    for (size_t at = 1; at < tokens.size();) {
      const int rows = std::min<size_t>(8, tokens.size() - at);
      decoded = m.session_verify(0, {tokens.begin() + at, tokens.begin() + at + rows});
      require(std::equal(decoded.final_hidden_bits.begin(), decoded.final_hidden_bits.end(),
                         first.final_hidden_bits.begin() + at * cfg.hidden_size),
              "incremental verification hidden differs from full prefill at " + std::to_string(at));
      m.session_rollback(0, rows, rows);
      at += rows;
    }
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
    meta << "{\"residual_norm\":\"fp32\",\"tokens\":[";
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
    std::string checkpoint, dump, fixture, config;
    bool run_kernels = false;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--kernels")
        run_kernels = true;
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
    if (run_kernels) kernels();
    if (!checkpoint.empty()) smoke(checkpoint, dump);
    if (!run_kernels && checkpoint.empty())
      throw std::runtime_error("use --kernels or --checkpoint-dir");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "minimax_test: %s\n", e.what());
    return 1;
  }
}
