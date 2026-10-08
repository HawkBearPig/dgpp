// The Qwen3.8-27B (qwen3_5) world-1 forward against its reference (#84,
// 2026-10-03).
//
// Modes:
//   --write-fixture DIR          the tiny synthetic checkpoint (tests/cuda/qwen35_fixture.hpp)
//   --smoke DIR                  Qwen35Model over the fixture: finite outputs, bitwise repeat
//   --checkpoint-dir DIR
//   --dump-file FILE             Qwen35Model against tools/qwen35_reference_dump.py's numpy
//                                double reference: every layer's residual, the final
//                                read, the per-token top-k logits, the draft block's rows
//   --engine-states FILE         (with --dump-file) write the engine's layer residuals and
//                                post-final-norm hidden for the teacher-forced reference
//   --relaxed                    the end-to-end budgets
//
// The ctest chain: qwen35_forward_fixture -> qwen35_forward_smoke, and
// qwen35_forward_fixture -> qwen35_forward_generate (python, end to end) ->
// qwen35_forward_test_e2e (--relaxed, writes the engine's states) ->
// qwen35_forward_generate_teacher (python, --teacher: every reference layer
// fed the engine's own input) -> qwen35_forward_test (strict).
//
// Budgets: the MiMo / GLM-4.7 gates' (16 / 128 bf16 ulps soft / hard with a
// 2 % RMS cancellation floor, l2 1 %, 0.5 % hard elements; top-1 exact
// outside certified near ties) on the teacher-forced dump, where each
// layer's error is one layer's worth of the fp32 order against the
// reference's doubles (the GDN recurrence is the chunked kernel against the
// scalar form; the attention the tile chain against the same chain in
// doubles). The end-to-end run is the relaxed yardstick (l2 3 %, 2 % hard):
// there is no routing here, so it has no flip exemption.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/config35.hpp"
#include "models/qwen/model35.hpp"
#include "qwen35_fixture.hpp"

namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::Qwen35Model;
using dgpp::Qwen35TextConfig;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

std::vector<int64_t> smoke_tokens(const Qwen35TextConfig& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  if (n > 9) t[9] = cfg.eos_token_ids.empty() ? 0 : cfg.eos_token_ids[0];
  return t;
}

// Per row: the k largest logits, ties to the lower id.
std::vector<std::vector<std::pair<int, float>>> topk(const std::vector<float>& logits, int rows, int vocab, int k) {
  std::vector<std::vector<std::pair<int, float>>> out(static_cast<size_t>(rows));
  for (int r = 0; r < rows; ++r) {
    const float* row = logits.data() + static_cast<size_t>(r) * vocab;
    std::vector<int> ids(static_cast<size_t>(vocab));
    for (int i = 0; i < vocab; ++i) ids[static_cast<size_t>(i)] = i;
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                      [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
    for (int i = 0; i < k; ++i) out[static_cast<size_t>(r)].emplace_back(ids[static_cast<size_t>(i)], row[ids[static_cast<size_t>(i)]]);
  }
  return out;
}

Qwen35Model make_model(const Qwen35TextConfig& cfg, const std::string& dir, int T, bool mtp,
                       dgpp::LoaderResidency residency = dgpp::LoaderResidency::Streaming, int max_requests = 1) {
  return Qwen35Model(cfg, dir, /*max_tokens=*/std::max(256, T), /*max_cache_tokens=*/1024, residency,
                     /*boundary=*/nullptr, /*rank=*/0, /*world=*/1, max_requests, /*decode_rows=*/8, mtp);
}

// ---- the reference dump ------------------------------------------------------
struct Dump {
  struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
  };
  std::vector<uint8_t> bytes;
  std::map<std::string, Tensor> tensors;
  int hidden = 0, vocab = 0, num_layers = 0, top_k = 0;
  int64_t token_count = 0;
  static Dump load(const std::string& path) {
    Dump d;
    std::ifstream f(path, std::ios::binary);
    require(f.good(), "dump: cannot open " + path);
    d.bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    require(d.bytes.size() >= 16 && std::memcmp(d.bytes.data(), "DGPPQW35", 8) == 0, "dump: bad magic");
    uint32_t version = 0, header_len = 0;
    std::memcpy(&version, d.bytes.data() + 8, 4);
    std::memcpy(&header_len, d.bytes.data() + 12, 4);
    require(version == 1, "dump: unsupported version");
    require(16 + static_cast<size_t>(header_len) <= d.bytes.size(), "dump: header exceeds file");
    const std::string_view header(reinterpret_cast<const char*>(d.bytes.data() + 16), header_len);
    dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(header);
    const dgpp::minijson::Value& root = parsed.root;
    const dgpp::minijson::Value& cfg = root.at("config");
    d.hidden = static_cast<int>(cfg.at("hidden").as_int());
    d.vocab = static_cast<int>(cfg.at("vocab").as_int());
    d.num_layers = static_cast<int>(cfg.at("num_layers").as_int());
    d.top_k = static_cast<int>(cfg.at("top_k").as_int());
    d.token_count = cfg.at("tokens").as_int();
    const size_t payload_base = 16 + header_len;
    for (const auto& m : root.at("tensors").members()) {
      Tensor t;
      t.dtype = std::string(m.value.at("dtype").as_string());
      for (const auto& dim : m.value.at("shape").items()) t.shape.push_back(dim.as_int());
      const uint64_t offset = m.value.at("offset").as_int();
      t.nbytes = static_cast<size_t>(m.value.at("nbytes").as_int());
      require(payload_base + offset + t.nbytes <= d.bytes.size(), "dump: tensor exceeds payload");
      t.data = d.bytes.data() + payload_base + offset;
      d.tensors[m.key] = t;
    }
    return d;
  }
  const Tensor& tensor(const std::string& name) const {
    auto it = tensors.find(name);
    require(it != tensors.end(), "dump: missing tensor " + name);
    return it->second;
  }
};

struct Stats {
  double l2 = 0, max_ulps = 0;
  long soft = 0, hard = 0, total = 0;
};

// bf16 ulps with a cancellation floor at 2 % of the reference's RMS, the
// soft/hard counts and the relative l2.
Stats compare_bf16(const uint16_t* got, const uint16_t* want, size_t n, int soft, int hard) {
  Stats s;
  s.total = static_cast<long>(n);
  double rms = 0;
  for (size_t i = 0; i < n; ++i) rms += std::pow(bf16_bits_to_float(want[i]), 2);
  rms = std::sqrt(rms / std::max<size_t>(n, 1));
  const double floor_abs = 0.02 * rms;
  double sum_d2 = 0, sum_o2 = 0;
  for (size_t i = 0; i < n; ++i) {
    const double g = bf16_bits_to_float(got[i]), w = bf16_bits_to_float(want[i]);
    int u = bf16_ulps(got[i], want[i]);
    if (std::fabs(g - w) <= floor_abs) u = 0;
    s.max_ulps = std::max(s.max_ulps, static_cast<double>(u));
    sum_d2 += (g - w) * (g - w);
    sum_o2 += w * w;
    if (u > soft) ++s.soft;
    if (u > hard) ++s.hard;
  }
  s.l2 = std::sqrt(sum_d2) / std::sqrt(sum_o2 + 1e-30);
  return s;
}

int run_smoke(const std::string& dir) {
  const Qwen35TextConfig cfg = Qwen35TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  const int T = 72;  // past two 32-token attention tiles
  const std::vector<int64_t> tokens = smoke_tokens(cfg, T);
  Qwen35Model model = make_model(cfg, dir, T, false);
  const Qwen35Model::Outputs out = model.forward(tokens, true);
  require(out.layer_states.size() == static_cast<size_t>(cfg.num_hidden_layers), "smoke: layer captures");
  for (size_t l = 0; l < out.layer_states.size(); ++l) {
    double rms = 0, mx = 0;
    for (uint16_t v : out.layer_states[l]) {
      const float f = bf16_bits_to_float(v);
      require(std::isfinite(f), "smoke: non-finite residual at layer " + std::to_string(l));
      rms += static_cast<double>(f) * f;
      mx = std::max(mx, static_cast<double>(std::fabs(f)));
    }
    rms = std::sqrt(rms / static_cast<double>(out.layer_states[l].size()));
    std::printf("[ .. ] layer %zu: h rms %.4g max %.4g\n", l, rms, mx);
  }
  require(out.logits.size() == static_cast<size_t>(T) * cfg.vocab_size, "smoke: every row's logits");
  for (float v : out.logits) require(std::isfinite(v), "smoke: non-finite logit");
  const Qwen35Model::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "smoke: forward is not deterministic across calls");
  const auto top = topk(out.logits, T, cfg.vocab_size, 4);
  std::printf("[ .. ] smoke: %d tokens, %d layers, top-1 of the last row %d (%.4g); deterministic\n", T,
              cfg.num_hidden_layers, top.back()[0].first, top.back()[0].second);
  std::printf("[ OK ] qwen35_forward_smoke\n");
  return 0;
}

int run_dump_parity(const std::string& dir, const std::string& dump_path, const std::string& states_path,
                    bool relaxed) {
  const Dump dump = Dump::load(dump_path);
  const Qwen35TextConfig cfg = Qwen35TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  require(cfg.hidden_size == dump.hidden && cfg.vocab_size == dump.vocab && cfg.num_hidden_layers == dump.num_layers,
          "dump config disagrees with the checkpoint config");
  const Dump::Tensor& tok = dump.tensor("tokens");
  require(tok.dtype == "I64", "dump: tokens dtype");
  const int T = static_cast<int>(dump.token_count);
  std::vector<int64_t> tokens(static_cast<size_t>(T));
  std::memcpy(tokens.data(), tok.data, static_cast<size_t>(T) * 8);

  Qwen35Model model = make_model(cfg, dir, T, false);
  const Qwen35Model::Outputs out = model.forward(tokens, true);
  const Qwen35Model::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "forward is not deterministic across calls");
  const int H = cfg.hidden_size;
  if (!states_path.empty()) {
    // The engine's layer residuals for the teacher-forced reference
    // (tools/qwen35_reference_dump.py --teacher): int32 L, T, H then bf16
    // rows per layer, the post-final-norm hidden after them.
    std::ofstream f(states_path, std::ios::binary);
    const int32_t hdr[3] = {cfg.num_hidden_layers, T, H};
    f.write(reinterpret_cast<const char*>(hdr), 12);
    for (const auto& st : out.layer_states)
      f.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size() * 2));
    f.write(reinterpret_cast<const char*>(out.final_hidden_bits.data()),
            static_cast<std::streamsize>(out.final_hidden_bits.size() * 2));
  }

  bool ok = true;
  const double l2_budget = relaxed ? 0.03 : 0.01, hard_budget = relaxed ? 0.02 : 0.005;
  // ---- per-layer residual states ---------------------------------------------
  const Dump::Tensor& ls = dump.tensor("layer_states");
  require(ls.dtype == "BF16" && ls.shape.size() == 3 && ls.shape[0] == cfg.num_hidden_layers && ls.shape[1] == T &&
              ls.shape[2] == H,
          "dump: layer_states shape");
  require(out.layer_states.size() == static_cast<size_t>(cfg.num_hidden_layers), "the engine captured every layer");
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const uint16_t* want = reinterpret_cast<const uint16_t*>(ls.data) + static_cast<size_t>(l) * T * H;
    const uint16_t* got = out.layer_states[static_cast<size_t>(l)].data();
    const Stats s = compare_bf16(got, want, static_cast<size_t>(T) * H, 16, 128);
    std::printf("[ .. ] layer %d (%s) h: l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", l,
                cfg.layers[static_cast<size_t>(l)] == dgpp::Qwen35LayerKind::Gdn ? "gdn" : "full", s.l2, s.max_ulps,
                s.soft, s.hard, s.total);
    std::vector<std::pair<double, int>> rows;
    for (int t = 0; t < T; ++t) {
      const Stats r = compare_bf16(got + static_cast<size_t>(t) * H, want + static_cast<size_t>(t) * H,
                                   static_cast<size_t>(H), 16, 128);
      rows.emplace_back(r.l2, t);
    }
    std::sort(rows.rbegin(), rows.rend());
    std::printf("[ .. ]   worst rows:");
    for (int i = 0; i < 4 && i < T; ++i)
      std::printf(" t%d %.3g", rows[static_cast<size_t>(i)].second, rows[static_cast<size_t>(i)].first);
    std::printf("\n");
    if (s.hard > 0 && s.hard <= 8) {
      double rms = 0;
      for (size_t i = 0; i < static_cast<size_t>(T) * H; ++i) rms += std::pow(bf16_bits_to_float(want[i]), 2);
      rms = std::sqrt(rms / (static_cast<double>(T) * H));
      for (size_t i = 0, shown = 0; i < static_cast<size_t>(T) * H && shown < 8; ++i) {
        const float g = bf16_bits_to_float(got[i]), w = bf16_bits_to_float(want[i]);
        if (bf16_ulps(got[i], want[i]) > 128 && std::fabs(g - w) > 0.02 * rms) {
          std::printf("[ .. ]   hard element t%zu h%zu: got %g want %g\n", i / H, i % H, g, w);
          ++shown;
        }
      }
    }
    if (s.total > 0 && (s.l2 > l2_budget || static_cast<double>(s.hard) / s.total > hard_budget)) ok = false;
  }
  // ---- the final read ---------------------------------------------------------
  const Dump::Tensor& fh = dump.tensor("final_hidden");
  require(fh.dtype == "BF16" && fh.shape.size() == 2 && fh.shape[0] == T && fh.shape[1] == H, "dump: final_hidden shape");
  const Stats fs_ = compare_bf16(out.final_hidden_bits.data(), reinterpret_cast<const uint16_t*>(fh.data),
                                 static_cast<size_t>(T) * H, 16, 128);
  std::printf("[ .. ] final hidden: l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", fs_.l2, fs_.max_ulps, fs_.soft,
              fs_.hard, fs_.total);
  if (fs_.total > 0 && (fs_.l2 > l2_budget || static_cast<double>(fs_.hard) / fs_.total > hard_budget)) ok = false;
  // ---- logits: top-1 (near ties certified), top-k overlap --------------------
  const Dump::Tensor& tid = dump.tensor("topk_ids");
  const Dump::Tensor& tlog = dump.tensor("topk_logits");
  const int k = dump.top_k;
  require(tid.dtype == "I32" && tlog.dtype == "F32" && tid.shape[0] == T && tid.shape[1] == k, "dump: topk shape");
  const int32_t* ref_ids = reinterpret_cast<const int32_t*>(tid.data);
  const float* ref_vals = reinterpret_cast<const float*>(tlog.data);
  const auto got = topk(out.logits, T, cfg.vocab_size, k);
  int top1_hard = 0, top1_soft = 0, set_miss = 0;
  double max_rel = 0;
  for (int t = 0; t < T; ++t) {
    const int32_t* rid = ref_ids + static_cast<size_t>(t) * k;
    const float* rval = ref_vals + static_cast<size_t>(t) * k;
    const double margin = std::fabs(rval[0] - rval[1]) / (std::fabs(rval[0]) + 1e-30);
    if (got[static_cast<size_t>(t)][0].first != rid[0]) {
      if (margin < 0.02) ++top1_soft; else ++top1_hard;
    }
    for (int i = 0; i < k; ++i) {
      const int32_t id = got[static_cast<size_t>(t)][i].first;
      const int32_t* p = std::find(rid, rid + k, id);
      if (p == rid + k) { ++set_miss; continue; }
      max_rel = std::max(max_rel, std::fabs(rval[p - rid] - got[static_cast<size_t>(t)][i].second) /
                                      (std::fabs(rval[p - rid]) + 1e-30));
    }
  }
  std::printf("[ .. ] logits: top-1 hard mismatches %d, near-tie %d of %d rows; top-%d set misses %d of %d; "
              "matched logits within %.3g relative\n",
              top1_hard, top1_soft, T, k, set_miss, T * k, max_rel);
  if (top1_hard != 0 || set_miss > T * k / 4) ok = false;
  // ---- the draft block: its rows over the prompt ---------------------------------
  if (dump.tensors.count("mtp_topk_ids") && cfg.mtp_layer() >= 0) {
    Qwen35Model mtp = make_model(cfg, dir, T, true);
    const Qwen35Model::Outputs d = mtp.mtp_forward(tokens);
    const int R = T - 1;
    const Dump::Tensor& mh = dump.tensor("mtp_final_hidden");
    require(mh.dtype == "BF16" && mh.shape.size() == 2 && mh.shape[0] == R && mh.shape[1] == H, "dump: mtp_final_hidden shape");
    require(d.final_hidden_bits.size() == static_cast<size_t>(R) * H, "mtp_forward: hidden rows");
    const Stats ms = compare_bf16(d.final_hidden_bits.data(), reinterpret_cast<const uint16_t*>(mh.data),
                                  static_cast<size_t>(R) * H, 16, 128);
    std::printf("[ .. ] draft final hidden: l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", ms.l2, ms.max_ulps, ms.soft,
                ms.hard, ms.total);
    if (ms.total > 0 && (ms.l2 > l2_budget || static_cast<double>(ms.hard) / ms.total > hard_budget)) ok = false;
    const Dump::Tensor& mid = dump.tensor("mtp_topk_ids");
    const Dump::Tensor& mlog = dump.tensor("mtp_topk_logits");
    require(mid.dtype == "I32" && mlog.dtype == "F32" && mid.shape[0] == R && mid.shape[1] == k, "dump: mtp topk shape");
    const int32_t* mref_ids = reinterpret_cast<const int32_t*>(mid.data);
    const float* mref_vals = reinterpret_cast<const float*>(mlog.data);
    const auto mgot = topk(d.logits, R, cfg.vocab_size, k);
    int mhard = 0, msoft = 0;
    for (int t = 0; t < R; ++t) {
      const int32_t r1 = mref_ids[static_cast<size_t>(t) * k];
      const float v1 = mref_vals[static_cast<size_t>(t) * k], v2 = mref_vals[static_cast<size_t>(t) * k + 1];
      const double margin = std::fabs(v1 - v2) / (std::fabs(v1) + 1e-30);
      if (mgot[static_cast<size_t>(t)][0].first != r1) {
        if (margin < 0.02) ++msoft; else ++mhard;
      }
    }
    std::printf("[ .. ] draft logits: top-1 hard mismatches %d, near-tie %d of %d rows\n", mhard, msoft, R);
    if (mhard != 0) ok = false;
  }
  std::printf("[ %s ] qwen35_forward_dump_parity (%s)\n", ok ? "OK" : "FAIL",
              relaxed ? "end-to-end, relaxed" : "teacher-forced, strict");
  return ok ? 0 : 1;
}

}  // namespace

// A decode step's rows are bitwise the same whatever the step's row count
// (2026-10-05, the scheduled verify depth: a slot verifies 2 / 4 / 6 / 8
// rows a step and its text must not move with the depth): the same context
// verified as 8 rows and as 1..7 rows, the first rows' logits and hidden
// bits compared bitwise, on the tiny fixture with the production recipe's
// fp8 head.
int run_rows_invariance(const std::string& dir, bool fp8_head) {
  const Qwen35TextConfig cfg = Qwen35TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  // The mixed release's head is already fp8-channel resident: the FP8
  // release's requantizing lever is refused there (the model's ctor). The
  // MoE release likewise serves its own formats.
  if (cfg.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Mixed ||
      cfg.quant_kind == dgpp::Qwen35QuantKind::Nvfp4Moe)
    fp8_head = false;
  Qwen35Model::set_dense_weights_fp8(fp8_head);
  Qwen35Model::set_session_capture_layers(true);
  // Resident: the fp8 head requantizes only on a resident stack (the
  // serving configuration); the streaming stack's bf16 head runs on
  // cuBLASLt, whose kernels change with the row count.
  Qwen35Model model = make_model(cfg, dir, 64, false, dgpp::LoaderResidency::Resident);
  const std::vector<int64_t> prompt = smoke_tokens(cfg, 37);
  std::vector<int64_t> block = smoke_tokens(cfg, 16);
  block.erase(block.begin(), block.begin() + 8);  // eight tokens unlike the prompt's head
  const int V = cfg.vocab_size, H = cfg.hidden_size;
  const auto run = [&](int T) {
    model.session_close(0);
    (void)model.session_prefill(0, prompt);
    return model.session_verify(0, std::vector<int64_t>(block.begin(), block.begin() + T));
  };
  const Qwen35Model::Outputs full = run(8);
  require(full.logits.size() == static_cast<size_t>(8) * V, "rows: the eight-row verify's logits");
  const Qwen35Model::Outputs again = run(8);
  require(again.logits == full.logits && again.final_hidden_bits == full.final_hidden_bits,
          "rows: the eight-row verify is not deterministic across calls");
  // The MoE release's BF16 attention projections take the GEMV chain at
  // <= 4 rows and the streaming MMA above: the two agree to 1 ulp (the
  // bf16 residuals are bitwise; only the fp32 head accumulators preserve
  // it, max |d| 1e-6 observed). Logits compare under 1e-5 there; hidden
  // states stay bitwise everywhere.
  const bool rows_exact = cfg.quant_kind != dgpp::Qwen35QuantKind::Nvfp4Moe;
  constexpr double kRowsTol = 1e-5;
  std::vector<int> failures;
  for (const int T : {1, 2, 3, 4, 5, 6, 7}) {
    const Qwen35Model::Outputs got = run(T);
    require(got.logits.size() == static_cast<size_t>(T) * V, "rows: the verify's logits");
    int bad_row = -1;
    std::string where;
    for (int r = 0; r < T && bad_row < 0; ++r) {
      const bool hid = std::memcmp(got.final_hidden_bits.data() + static_cast<size_t>(r) * H,
                                   full.final_hidden_bits.data() + static_cast<size_t>(r) * H,
                                   static_cast<size_t>(H) * 2) != 0;
      size_t nlog = 0;
      double maxd = 0;
      for (int c = 0; c < V; ++c) {
        const float a = got.logits[static_cast<size_t>(r) * V + c], b = full.logits[static_cast<size_t>(r) * V + c];
        const bool differ = rows_exact ? a != b : std::fabs(a - b) > kRowsTol;
        if (differ) { ++nlog; maxd = std::max(maxd, static_cast<double>(std::fabs(a - b))); }
      }
      if (hid || nlog > 0) {
        bad_row = r;
        int first_layer = -1;
        for (size_t l = 0; l < got.layer_states.size() && first_layer < 0; ++l)
          if (std::memcmp(got.layer_states[l].data() + static_cast<size_t>(r) * H,
                          full.layer_states[l].data() + static_cast<size_t>(r) * H, static_cast<size_t>(H) * 2) != 0)
            first_layer = static_cast<int>(l);
        std::string stage = "none";
        for (size_t st = 0; st < got.debug_stages.size() && stage == "none"; ++st) {
          const size_t w = got.debug_stages[st].size() / static_cast<size_t>(T);
          const uint16_t* gr = got.debug_stages[st].data() + static_cast<size_t>(r) * w;
          if (std::memcmp(gr, full.debug_stages[st].data() + static_cast<size_t>(r) * w, w * 2) != 0) {
            stage = got.debug_stage_names[st];
            size_t nd = 0; double md = 0;
            for (size_t c = 0; c < w; ++c) {
              const float a = bf16_bits_to_float(gr[c]), b = bf16_bits_to_float(full.debug_stages[st][static_cast<size_t>(r) * w + c]);
              if (a != b) { ++nd; md = std::max(md, static_cast<double>(std::fabs(a - b))); }
            }
            int same_as = -1;  // the row of the eight-row walk this row equals, if any (a stride / staging slip)
            for (int rr = 0; rr < 8 && same_as < 0; ++rr)
              if (std::memcmp(gr, full.debug_stages[st].data() + static_cast<size_t>(rr) * w, w * 2) == 0) same_as = rr;
            stage += " (" + std::to_string(nd) + " of " + std::to_string(w) + " differ, max |d| " + std::to_string(md) +
                     ", equals the eight-row walk's row " + std::to_string(same_as) + ")";
            for (size_t c = 0; c < w; ++c) {
              const uint16_t a = gr[c], b = full.debug_stages[st][static_cast<size_t>(r) * w + c];
              if (a != b)
                std::printf("[ .. ]   %s row %d col %zu: %d-row walk 0x%04x (%.6g) vs eight-row 0x%04x (%.6g)\n",
                            got.debug_stage_names[st].c_str(), r, c, T, a, bf16_bits_to_float(a), b, bf16_bits_to_float(b));
            }
            // Determinism of this row count: the same walk again.
            const Qwen35Model::Outputs rerun = run(T);
            std::printf("[ .. ]   the %d-row walk repeated: stage %s %s, logits %s\n", T, got.debug_stage_names[st].c_str(),
                        rerun.debug_stages[st] == got.debug_stages[st] ? "bitwise" : "DIFFERENT",
                        rerun.logits == got.logits ? "bitwise" : "DIFFERENT");
          }
        }
        where = (hid ? "the final hidden" : "the final hidden the same") + std::string(", ") + std::to_string(nlog) +
                " of " + std::to_string(V) + " logits differ, max |d| " + std::to_string(maxd) +
                ", first differing layer " + std::to_string(first_layer) +
                (first_layer >= 0 ? (cfg.layers[static_cast<size_t>(first_layer)] == dgpp::Qwen35LayerKind::Gdn ? " (gdn)" : " (full)") : " (none)") +
                ", first differing stage of GDN layer 0: " + stage;
      }
    }
    if (bad_row >= 0) {
      std::printf("[ !! ] a %d-row verify's row %d differs from the eight-row verify's (%s)\n", T, bad_row, where.c_str());
      failures.push_back(T);
    } else {
      std::printf("[ .. ] a %d-row verify is bitwise the eight-row verify's first rows\n", T);
    }
  }
  require(failures.empty(), "rows: " + std::to_string(failures.size()) + " row counts differ from the eight-row verify");
  std::printf("[ OK ] qwen35_decode_rows_invariance\n");
  return 0;
}

// A prompt prefilled as one span of a group walk (several cold prompts in
// one forward, session_prefill_group) against the same prompt prefilled
// alone: its last-row logits and its first decode step must be bitwise
// (2026-10-05: prompts started together read DIFFERENT at char 26 on the
// fabric while prompts joining later read identical). Groups of two (74
// rows: the GDN chunked form's threshold) and four (148 rows: past the
// scale GEMM's 128-row lowering, the dequant bridge on cuBLASLt).
int run_group_invariance(const std::string& dir) {
  const Qwen35TextConfig cfg = Qwen35TextConfig::from_json_file((fs::path(dir) / "config.json").string());
  // No requantizing levers on the NVFP4 releases (the model's ctor refuses).
  const bool fp8_head = cfg.quant_kind != dgpp::Qwen35QuantKind::Nvfp4Mixed &&
                        cfg.quant_kind != dgpp::Qwen35QuantKind::Nvfp4Moe;
  Qwen35Model::set_dense_weights_fp8(fp8_head);
  Qwen35Model model = make_model(cfg, dir, 512, false, dgpp::LoaderResidency::Resident, /*max_requests=*/8);
  const int V = cfg.vocab_size;
  std::vector<int64_t> prompts[8];
  for (int i = 0; i < 8; ++i) {
    std::vector<int64_t> t = smoke_tokens(cfg, 37 * (i + 1));
    prompts[i].assign(t.end() - 37, t.end());  // four unlike 37-token prompts
  }
  const std::vector<int64_t> next = {7};
  const auto solo = [&](int req) {
    model.session_close(req);
    const Qwen35Model::Outputs p = model.session_prefill(req, prompts[req]);
    const Qwen35Model::Outputs d = model.session_verify(req, next);
    return std::make_pair(p.logits, d.logits);
  };
  const auto ref = solo(0);
  require(ref.first.size() == static_cast<size_t>(V), "group: the prefill's last-row logits");
  const auto again = solo(0);
  require(again.first == ref.first && again.second == ref.second, "group: a solo prefill is not deterministic");
  // The gate: a group inside the scale GEMM's 128-row lowering (two
  // prompts, 74 rows: the streaming form, as the solo walk) is bitwise the
  // solo prefill; groups past it (four and eight: the dequant bridge on
  // cuBLASLt) are bitwise each other (one pinned algorithm per shape,
  // CublasLtGemm::set_pinned_rows). A group past the bound against the
  // solo walk is a different chain by construction (reported, not
  // required): engine.prefill_group false is the exact mode for it.
  // The MoE release holds the same gate under a tight numerical budget
  // (max |d| 1e-5): its grouped expert kernels reduce in batch-shape order
  // (fp32 noise ~2e-6 observed once, no flips, no path switch), which a
  // bitwise comparison cannot survive if it recurs; a real chunk-dependence
  // (wrong rows, wrong experts, a switched algorithm) moves logits by
  // orders of magnitude more. The current fixture passes bitwise.
  const bool group_exact = cfg.quant_kind != dgpp::Qwen35QuantKind::Nvfp4Moe;
  constexpr double kGroupTol = 1e-5;
  std::vector<float> four_logits, four_step;
  for (const int n : {2, 4, 8}) {
    std::vector<int> reqs;
    std::vector<const std::vector<int64_t>*> ps;
    for (int i = 0; i < n; ++i) { model.session_close(i); reqs.push_back(i); ps.push_back(&prompts[i]); }
    const std::vector<Qwen35Model::Outputs> outs = model.session_prefill_group(reqs, ps);
    const Qwen35Model::Outputs d = model.session_verify(0, next);
    size_t np = 0, nd = 0; double mp = 0, md = 0;
    for (int c = 0; c < V; ++c) {
      const bool pdiff = group_exact ? outs[0].logits[c] != ref.first[c]
                                     : std::fabs(outs[0].logits[c] - ref.first[c]) > kGroupTol;
      const bool ddiff = group_exact ? d.logits[c] != ref.second[c]
                                     : std::fabs(d.logits[c] - ref.second[c]) > kGroupTol;
      if (pdiff) { ++np; mp = std::max(mp, static_cast<double>(std::fabs(outs[0].logits[c] - ref.first[c]))); }
      if (ddiff) { ++nd; md = std::max(md, static_cast<double>(std::fabs(d.logits[c] - ref.second[c]))); }
    }
    std::printf("[ .. ] a group of %d (%d rows): prompt 0's prefill logits %s its solo walk's (%zu of %d differ, max |d| %.3g; first step %zu, max |d| %.3g)\n",
                n, 37 * n, np == 0 && nd == 0 ? "bitwise" : "differ from", np, V, mp, nd, md);
    if (n == 2) require(np == 0 && nd == 0, "group: a group inside the lowering bound must be bitwise the solo prefill");
    if (n == 4) { four_logits = outs[0].logits; four_step = d.logits; }
    if (n == 8) {
      bool same;
      if (group_exact) {
        same = outs[0].logits == four_logits && d.logits == four_step;
      } else {
        same = outs[0].logits.size() == four_logits.size() && d.logits.size() == four_step.size();
        for (size_t c = 0; same && c < outs[0].logits.size(); ++c)
          same = std::fabs(outs[0].logits[c] - four_logits[c]) <= kGroupTol;
        for (size_t c = 0; same && c < d.logits.size(); ++c)
          same = std::fabs(d.logits[c] - four_step[c]) <= kGroupTol;
      }
      std::printf("[ .. ] a group of 8 (296 rows) against the group of 4 (148 rows), both past the 128-row bound: %s\n",
                  same ? "bitwise" : "DIFFERENT");
      require(same, "group: groups past the lowering bound must be bitwise each other (the pinned cuBLASLt algorithm)");
    }
  }
  std::printf("[ OK ] qwen35_prefill_group_invariance\n");
  return 0;
}

int main(int argc, char** argv) {
  std::string fixture, fixture_mixed, fixture_moe, smoke, rows, group, checkpoint, dump, states;
  bool relaxed = false, bf16_head = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--write-fixture" && i + 1 < argc) fixture = argv[++i];
    else if (a == "--write-fixture-mixed" && i + 1 < argc) fixture_mixed = argv[++i];
    else if (a == "--write-fixture-moe" && i + 1 < argc) fixture_moe = argv[++i];
    else if (a == "--smoke" && i + 1 < argc) smoke = argv[++i];
    else if (a == "--rows-invariance" && i + 1 < argc) rows = argv[++i];
    else if (a == "--group-invariance" && i + 1 < argc) group = argv[++i];
    else if (a == "--bf16-head") bf16_head = true;
    else if (a == "--checkpoint-dir" && i + 1 < argc) checkpoint = argv[++i];
    else if (a == "--dump-file" && i + 1 < argc) dump = argv[++i];
    else if (a == "--engine-states" && i + 1 < argc) states = argv[++i];
    else if (a == "--relaxed") relaxed = true;
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  try {
    if (!fixture.empty()) {
      qwen35fx::write_fixture(fixture);
      std::printf("[ OK ] wrote the fixture to %s\n", fixture.c_str());
      return 0;
    }
    if (!fixture_mixed.empty()) {
      qwen35fx::write_fixture(fixture_mixed, /*mixed=*/true);
      std::printf("[ OK ] wrote the mixed fixture to %s\n", fixture_mixed.c_str());
      return 0;
    }
    if (!fixture_moe.empty()) {
      qwen35fx::write_moe_fixture(fixture_moe);
      std::printf("[ OK ] wrote the MoE fixture to %s\n", fixture_moe.c_str());
      return 0;
    }
    if (!smoke.empty()) return run_smoke(smoke);
    if (!rows.empty()) return run_rows_invariance(rows, !bf16_head);
    if (!group.empty()) return run_group_invariance(group);
    if (!checkpoint.empty() && !dump.empty()) return run_dump_parity(checkpoint, dump, states, relaxed);
    std::fprintf(stderr,
                 "usage: --write-fixture DIR | --write-fixture-mixed DIR | --write-fixture-moe DIR | --smoke DIR | --rows-invariance DIR | "
                 "--checkpoint-dir DIR --dump-file FILE "
                 "[--engine-states FILE] [--relaxed]\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
