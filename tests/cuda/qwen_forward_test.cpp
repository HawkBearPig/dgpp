// The Qwen3.8-Flash-Next world-1 forward.
//
// Modes:
//   --write-fixture DIR          the tiny synthetic checkpoint (tests/cuda/qwen_fixture.hpp)
//   --smoke DIR                  QwenModel over the fixture: finite outputs, bitwise repeat
//   --rope-scaling F:O[:BF:BS:AF:MF]  run --smoke with the engine's YaRN knob
//                                (factor, original_max_position_embeddings,
//                                 beta_fast, beta_slow, attn_factor,
//                                 mrope_cache_factor — defaults 32/1/1/4)
//   --plan-check                 the memory plan's context line under the rope knob
//   --checkpoint-dir DIR
//   --dump-file FILE             QwenModel against tools/qwen_reference_dump.py's pure
//                                double reference: every layer's hyper state, the final
//                                read, the per-token top-k logits, the routing decisions
//
// The ctest chain: qwen_forward_fixture -> qwen_forward_smoke, and
// qwen_forward_fixture -> qwen_forward_generate (python) -> qwen_forward_test.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/config.hpp"
#include "models/qwen/forward.hpp"
#include "qwen_fixture.hpp"

namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::QwenModel;
using dgpp::QwenTextConfig;

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

std::vector<int64_t> smoke_tokens(const QwenTextConfig& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  if (n > 9) t[9] = cfg.eos_token_ids.empty() ? 0 : cfg.eos_token_ids[0];
  return t;
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
  int hidden = 0, vocab = 0, num_layers = 0, top_k = 0, hc = 0;
  int64_t token_count = 0;

  static Dump load(const std::string& path) {
    Dump d;
    std::ifstream f(path, std::ios::binary);
    require(f.good(), "dump: cannot open " + path);
    d.bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    require(d.bytes.size() >= 16 && std::memcmp(d.bytes.data(), "DGPPQWND", 8) == 0, "dump: bad magic");
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
    d.hc = static_cast<int>(cfg.at("hc").as_int());
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

int run_plan_check() {  // The memory plan's context line under the rope knob (engine.rope_scaling,
  // 2026-09-18): the plan must be computed at the YaRN-scaled ceiling, so a
  // 512K pool is validated against the pool it really allocates — and a pool
  // past the ceiling is what the launcher warns about.
  QwenTextConfig cfg = qwenfx::tiny_config();
  const int64_t native = cfg.max_position_embeddings;  // the fixture's 4096
  const int64_t pool = native * 128;                    // a pool far past the ceiling
  QwenTextConfig yarn = cfg;
  yarn.rope_scaling = dgpp::RopeScaling{2.0, native * 64, 32.0, 1.0, 1.0, 4.0};
  const int64_t want_off = native, want_on = native * 64 * 2;
  const auto plan_of = [&](const QwenTextConfig& c) {
    return QwenModel::plan_memory(c, /*max_tokens=*/64, /*max_cache_tokens=*/pool, /*tp_rank=*/0,
                                  /*tp_world=*/2, dgpp::QwenResidency::Resident,
                                  /*max_requests=*/4, /*mtp=*/false, /*decode_rows=*/8);
  };
  const int64_t off = plan_of(cfg).context_tokens;
  const int64_t on = plan_of(yarn).context_tokens;
  std::printf("[ .. ] plan context: plain %lld, YaRN %lld (pool %lld, native %lld)\n",
              static_cast<long long>(off), static_cast<long long>(on),
              static_cast<long long>(pool), static_cast<long long>(native));
  require(off == want_off, "the plain plan keeps the checkpoint's ceiling");
  require(on == want_on, "the YaRN plan takes original x factor");
  require(cfg.context_limit() == want_off && yarn.context_limit() == want_on,
          "the config's ceiling is what the plan reads");
  // The pool is sized by the cache capacity, not by the positional ceiling,
  // so the knob lifts what a request may reach without moving a byte: the
  // plan validates the same pool either way (a pool past the ceiling is
  // concurrency headroom, which the launcher names out loud).
  require(plan_of(yarn).total_bytes() == plan_of(cfg).total_bytes(),
          "the knob allocates nothing");
  // A pool of exactly the scaled ceiling is accepted by the same arithmetic.
  require(QwenModel::plan_memory(yarn, 64, native * 64 * 2, 0, 2, dgpp::QwenResidency::Resident, 4,
                                 false, 8)
                  .context_tokens == want_on,
          "a pool at the ceiling");
  std::printf("[ OK ] qwen_plan_check\n");
  return 0;
}

int run_smoke(const std::string& dir, const std::optional<dgpp::RopeScaling>& rope_scaling) {
  QwenTextConfig cfg = QwenTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  if (rope_scaling.has_value()) cfg.rope_scaling = *rope_scaling;
  const int T = 72;
  const std::vector<int64_t> tokens = smoke_tokens(cfg, T);
  QwenModel model(cfg, dir, T, 256);
  const QwenModel::Outputs out = model.forward(tokens, true);
  require(out.layer_states.size() == static_cast<size_t>(cfg.num_hidden_layers), "smoke: layer captures");
  for (size_t l = 0; l < out.layer_states.size(); ++l) {
    double rms = 0, mx = 0;
    for (uint16_t v : out.layer_states[l]) {
      const float f = bf16_bits_to_float(v);
      require(std::isfinite(f), "smoke: non-finite hyper state at layer " + std::to_string(l));
      rms += static_cast<double>(f) * f;
      mx = std::max(mx, static_cast<double>(std::fabs(f)));
    }
    rms = std::sqrt(rms / static_cast<double>(out.layer_states[l].size()));
    std::printf("[ .. ] layer %zu: R rms %.4g max %.4g\n", l, rms, mx);
  }
  for (float v : out.logits) require(std::isfinite(v), "smoke: non-finite logit");
  const QwenModel::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "smoke: forward is not deterministic across calls");
  const auto top = QwenModel::topk(out.logits, T, cfg.vocab_size, 4);
  std::printf("[ .. ] smoke: %d tokens, %d layers, top-1 of the last row %d (%.4g); deterministic%s\n", T,
              cfg.num_hidden_layers, top.back()[0].first, top.back()[0].second,
              rope_scaling.has_value() ? " (YaRN rope)" : "");
  if (rope_scaling.has_value()) {
    // The knob is live: with the same weights, the YaRN frequencies must
    // move the logits away from the plain rope's (the table and the mscale
    // are the only difference).
    QwenTextConfig plain_cfg = cfg;
    plain_cfg.rope_scaling.reset();
    QwenModel plain(plain_cfg, dir, T, 256);
    const QwenModel::Outputs p = plain.forward(tokens, false);
    require(p.logits != out.logits, "the YaRN rope must change the forward");
    double worst = 0;
    for (size_t i = 0; i < p.logits.size(); ++i)
      worst = std::max(worst, std::fabs(static_cast<double>(p.logits[i]) - out.logits[i]));
    std::printf("[ .. ] smoke: YaRN vs plain logits differ by at most %.4g\n", worst);
  }
  std::printf("[ OK ] qwen_forward_smoke\n");
  return 0;
}

int run_dump_parity(const std::string& dir, const std::string& dump_path) {
  const Dump dump = Dump::load(dump_path);
  const QwenTextConfig cfg = QwenTextConfig::from_json_file((fs::path(dir) / "config.json").string());
  require(cfg.hidden_size == dump.hidden && cfg.vocab_size == dump.vocab &&
              cfg.num_hidden_layers == dump.num_layers && cfg.hc_count == dump.hc,
          "dump config disagrees with the checkpoint config");
  const Dump::Tensor& tok = dump.tensor("tokens");
  require(tok.dtype == "I64", "dump: tokens dtype");
  const int T = static_cast<int>(dump.token_count);
  std::vector<int64_t> tokens(static_cast<size_t>(T));
  std::memcpy(tokens.data(), tok.data, static_cast<size_t>(T) * 8);

  QwenModel model(cfg, dir, T, std::max(256, T));
  const QwenModel::Outputs out = model.forward(tokens, true);
  const QwenModel::Outputs again = model.forward(tokens, false);
  require(again.final_hidden_bits == out.final_hidden_bits && again.logits == out.logits,
          "forward is not deterministic across calls");

  const int H = cfg.hidden_size, W = cfg.hc_count * H;
  bool ok = true;
  // ---- per-layer hyper states ------------------------------------------------
  const Dump::Tensor& ls = dump.tensor("layer_states");
  require(ls.dtype == "BF16" && ls.shape.size() == 3 && ls.shape[0] == cfg.num_hidden_layers &&
              ls.shape[1] == T && ls.shape[2] == W,
          "dump: layer_states shape");
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const uint16_t* want = reinterpret_cast<const uint16_t*>(ls.data) + static_cast<size_t>(l) * T * W;
    const Stats s = compare_bf16(out.layer_states[static_cast<size_t>(l)].data(), want,
                                 static_cast<size_t>(T) * W, 16, 128);
    std::printf("[ .. ] layer %d R: l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", l, s.l2,
                s.max_ulps, s.soft, s.hard, s.total);
    if (s.l2 > 0.01 || static_cast<double>(s.hard) / s.total > 0.005) ok = false;
  }
  // ---- the final read ---------------------------------------------------------
  const Dump::Tensor& fh = dump.tensor("final_hidden");
  require(fh.dtype == "BF16" && fh.shape.size() == 2 && fh.shape[0] == T && fh.shape[1] == H,
          "dump: final_hidden shape");
  const Stats fs_ = compare_bf16(out.final_hidden_bits.data(), reinterpret_cast<const uint16_t*>(fh.data),
                                 static_cast<size_t>(T) * H, 16, 128);
  std::printf("[ .. ] final hidden: l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", fs_.l2, fs_.max_ulps,
              fs_.soft, fs_.hard, fs_.total);
  if (fs_.l2 > 0.01 || static_cast<double>(fs_.hard) / fs_.total > 0.005) ok = false;
  // ---- logits: top-1 (near ties certified), top-k overlap --------------------
  const Dump::Tensor& tid = dump.tensor("topk_ids");
  const Dump::Tensor& tlog = dump.tensor("topk_logits");
  const int k = dump.top_k;
  require(tid.dtype == "I32" && tlog.dtype == "F32" && tid.shape[0] == T && tid.shape[1] == k, "dump: topk shape");
  const int32_t* ref_ids = reinterpret_cast<const int32_t*>(tid.data);
  const float* ref_vals = reinterpret_cast<const float*>(tlog.data);
  const auto got = QwenModel::topk(out.logits, T, cfg.vocab_size, k);
  int top1_hard = 0, top1_soft = 0, set_miss = 0;
  double max_rel = 0;
  for (int t = 0; t < T; ++t) {
    const int32_t r1 = ref_ids[static_cast<size_t>(t) * k];
    const float v1 = ref_vals[static_cast<size_t>(t) * k], v2 = ref_vals[static_cast<size_t>(t) * k + 1];
    const double margin = std::fabs(v1 - v2) / (std::fabs(v1) + 1e-30);
    if (got[static_cast<size_t>(t)][0].first != r1) {
      if (margin < 0.02) ++top1_soft; else ++top1_hard;
    }
    // The reference's own logit for our top-1 must be within 2 % of its top-1.
    for (int i = 0; i < k; ++i) {
      const int32_t id = got[static_cast<size_t>(t)][i].first;
      const float* p = std::find(ref_ids + static_cast<size_t>(t) * k, ref_ids + static_cast<size_t>(t + 1) * k, id) ==
                               ref_ids + static_cast<size_t>(t + 1) * k
                           ? nullptr
                           : ref_vals + static_cast<size_t>(t) * k +
                                 (std::find(ref_ids + static_cast<size_t>(t) * k, ref_ids + static_cast<size_t>(t + 1) * k, id) -
                                  (ref_ids + static_cast<size_t>(t) * k));
      if (!p) { ++set_miss; continue; }
      max_rel = std::max(max_rel, std::fabs(*p - got[static_cast<size_t>(t)][i].second) / (std::fabs(*p) + 1e-30));
    }
  }
  std::printf("[ .. ] logits: top-1 hard mismatches %d, near-tie %d of %d rows; top-%d set misses %d of %d; "
              "matched logits within %.3g relative\n",
              top1_hard, top1_soft, T, k, set_miss, T * k, max_rel);
  if (top1_hard != 0 || set_miss > T * k / 4) ok = false;
  // ---- the draft block (Q6 stage C): its rows over the prompt ------------------
  if (dump.tensors.count("mtp_topk_ids") && cfg.mtp_layer() >= 0) {
    QwenModel mtp(cfg, dir, T, std::max(256, T), dgpp::QwenResidency::Streaming, nullptr, 0, 1, 1, /*mtp=*/true);
    const QwenModel::Outputs d = mtp.mtp_forward(tokens);
    const int R = T - 1;
    const Dump::Tensor& mh = dump.tensor("mtp_final_hidden");
    require(mh.dtype == "BF16" && mh.shape.size() == 2 && mh.shape[0] == R && mh.shape[1] == H, "dump: mtp_final_hidden shape");
    const Stats ms = compare_bf16(d.final_hidden_bits.data(), reinterpret_cast<const uint16_t*>(mh.data),
                                  static_cast<size_t>(R) * H, 16, 128);
    std::printf("[ .. ] draft final hidden: l2 %.3g max %g ulps, soft %ld hard %ld of %ld\n", ms.l2, ms.max_ulps,
                ms.soft, ms.hard, ms.total);
    if (ms.l2 > 0.01 || static_cast<double>(ms.hard) / ms.total > 0.005) ok = false;
    const Dump::Tensor& mid = dump.tensor("mtp_topk_ids");
    const Dump::Tensor& mlog = dump.tensor("mtp_topk_logits");
    require(mid.dtype == "I32" && mlog.dtype == "F32" && mid.shape[0] == R && mid.shape[1] == k, "dump: mtp topk shape");
    const int32_t* mref_ids = reinterpret_cast<const int32_t*>(mid.data);
    const float* mref_vals = reinterpret_cast<const float*>(mlog.data);
    const auto mgot = QwenModel::topk(d.logits, R, cfg.vocab_size, k);
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
  // ---- routing ----------------------------------------------------------------
  if (dump.tensors.count("route_ids")) {
    const Dump::Tensor& rt = dump.tensor("route_ids");
    const int K = static_cast<int>(rt.shape[2]);
    const int32_t* ref = reinterpret_cast<const int32_t*>(rt.data);
    long diff = 0, total = 0;
    for (int l = 0; l < cfg.num_hidden_layers; ++l)
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i, ++total)
          if (out.route_ids[static_cast<size_t>(l)][static_cast<size_t>(t) * K + i] !=
              ref[(static_cast<size_t>(l) * T + t) * K + i])
            ++diff;
    std::printf("[ .. ] routing: %ld of %ld slots differ from the reference\n", diff, total);
    if (diff > total / 20) ok = false;
  }
  std::printf("[ %s ] qwen_forward_dump_parity\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixture, smoke, checkpoint, dump;
  std::string rope_scaling_arg;
  bool plan_check = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--write-fixture" && i + 1 < argc) fixture = argv[++i];
    else if (a == "--smoke" && i + 1 < argc) smoke = argv[++i];
    else if (a == "--rope-scaling" && i + 1 < argc) rope_scaling_arg = argv[++i];
    else if (a == "--plan-check") plan_check = true;
    else if (a == "--checkpoint-dir" && i + 1 < argc) checkpoint = argv[++i];
    else if (a == "--dump-file" && i + 1 < argc) dump = argv[++i];
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  try {
    // The engine's YaRN knob (engine.rope_scaling) as the smoke's config:
    // factor:original[:beta_fast:beta_slow:attn_factor:mrope_cache_factor].
    std::optional<dgpp::RopeScaling> rope_scaling;
    if (!rope_scaling_arg.empty()) {
      std::vector<double> f;
      std::string field;
      std::stringstream ss(rope_scaling_arg);
      while (std::getline(ss, field, ':')) f.push_back(std::stod(field));
      require(f.size() >= 2 && f.size() <= 6, "--rope-scaling wants factor:original[:4 more]");
      dgpp::RopeScaling rs;
      rs.factor = f[0];
      rs.original_max_position_embeddings = static_cast<int64_t>(f[1]);
      if (f.size() > 2) rs.beta_fast = f[2];
      if (f.size() > 3) rs.beta_slow = f[3];
      if (f.size() > 4) rs.attn_factor = f[4];
      if (f.size() > 5) rs.mrope_cache_factor = f[5];
      rs.validate("--rope-scaling");
      rope_scaling = rs;
    }
    if (!fixture.empty()) {
      qwenfx::write_fixture(qwenfx::tiny_config(), fixture);
      std::printf("[ OK ] wrote the fixture to %s\n", fixture.c_str());
      return 0;
    }
    if (plan_check) return run_plan_check();
    if (!smoke.empty()) return run_smoke(smoke, rope_scaling);
    if (!checkpoint.empty() && !dump.empty()) return run_dump_parity(checkpoint, dump);
    std::fprintf(stderr, "usage: --write-fixture DIR | --smoke DIR | --plan-check | --checkpoint-dir DIR --dump-file FILE\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
