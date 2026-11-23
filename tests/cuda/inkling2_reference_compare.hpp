#pragma once
// The Inkling 2 engine against a reference dump (tools/inkling2_reference_dump.py,
// or another implementation's dump in the same layout): the dump reader and
// the per-visit comparison the world-1 and tensor-parallel tests share.
//
// Budgets: 16 / 128 bf16 ulps soft / hard with a 2 % RMS cancellation floor,
// relative l2 1 %, 0.5 % hard elements, top-1 exact outside near ties and
// routing flips under 5 % of the slots on a teacher-forced dump. The
// end-to-end run is the relaxed yardstick (l2 3 %, 2 % hard) with the visits
// from the first routing flip on reported, not judged. Under the native
// recipe the teacher also supplies every visit's expert input, and the
// end-to-end run judges the first visit up to its expert input only.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "loaders/minijson.hpp"
#include "models/inkling2/config.hpp"

namespace ink2ref {

using dgpp::bf16_bits_to_float;
using dgpp::Inkling2Config;

inline void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

inline int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

struct Dump {
  struct Tensor {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
  };
  std::vector<uint8_t> bytes;
  std::map<std::string, Tensor> tensors;
  int hidden = 0, vocab = 0, visits = 0, top_k = 0;
  int64_t token_count = 0;
  std::string activations;

  static Dump load(const std::string& path) {
    Dump d;
    std::ifstream f(path, std::ios::binary);
    require(f.good(), "dump: cannot open " + path);
    d.bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    require(d.bytes.size() >= 16 && std::memcmp(d.bytes.data(), "DGPPINK2", 8) == 0, "dump: bad magic");
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
    d.visits = static_cast<int>(cfg.at("visits").as_int());
    d.top_k = static_cast<int>(cfg.at("top_k").as_int());
    d.token_count = cfg.at("tokens").as_int();
    d.activations = std::string(cfg.at("activations").as_string());
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
// soft / hard counts and the relative l2.
inline Stats compare_bf16(const uint16_t* got, const uint16_t* ref, size_t n) {
  Stats s;
  s.total = static_cast<long>(n);
  double ss = 0, dd = 0;
  for (size_t i = 0; i < n; ++i) {
    const double r = bf16_bits_to_float(ref[i]);
    ss += r * r;
  }
  const double rms = std::sqrt(ss / static_cast<double>(std::max<size_t>(n, 1)));
  const double floor = 0.02 * rms;
  for (size_t i = 0; i < n; ++i) {
    const double g = bf16_bits_to_float(got[i]), r = bf16_bits_to_float(ref[i]);
    dd += (g - r) * (g - r);
    if (std::fabs(g - r) <= floor) continue;
    const int u = bf16_ulps(got[i], ref[i]);
    s.max_ulps = std::max<double>(s.max_ulps, u);
    if (u > 16) ++s.soft;
    if (u > 128) ++s.hard;
  }
  s.l2 = ss > 0 ? std::sqrt(dd / ss) : std::sqrt(dd);
  return s;
}

// One engine run in the dump's shape.
struct EngineRun {
  std::vector<std::vector<uint16_t>> states;      // per visit [T, H]: the partial state after the visit
  std::vector<std::vector<uint16_t>> attention;   // per visit [T, H]: the attention contribution
  std::vector<std::vector<uint16_t>> mlp_input;   // per visit [T, H]: the experts' input
  std::vector<std::vector<int32_t>> route_ids;    // per visit [T, K]
  std::vector<uint16_t> final_hidden;             // [T, H]
  std::vector<float> logits;                      // [T, vocab], softcapped
};

// The teacher's file: int32 V, T, H, then the states and the expert inputs.
inline void write_states(const std::string& path, const EngineRun& run, int T, int H) {
  std::ofstream f(path, std::ios::binary);
  const int32_t header[3] = {static_cast<int32_t>(run.states.size()), T, H};
  f.write(reinterpret_cast<const char*>(header), 12);
  for (const auto& s : run.states) f.write(reinterpret_cast<const char*>(s.data()), static_cast<std::streamsize>(s.size() * 2));
  for (const auto& s : run.mlp_input) f.write(reinterpret_cast<const char*>(s.data()), static_cast<std::streamsize>(s.size() * 2));
  require(f.good(), "cannot write " + path);
  std::printf("wrote the engine's states to %s\n", path.c_str());
}

// 0 when the run is within budget (or `report`: printed, not judged).
inline int judge(const Inkling2Config& cfg, const Dump& d, const EngineRun& run, bool relaxed, bool native, bool report,
                 const char* label) {
  require(d.hidden == cfg.hidden_size && d.vocab == cfg.vocab_size && d.visits == cfg.num_visits(),
          "dump: the geometry is not this checkpoint's");
  require(report || d.activations == (native ? "native" : "bf16"), "dump: the activation recipe differs from the run's");
  const int T = static_cast<int>(d.token_count), H = cfg.hidden_size, V = cfg.num_visits();
  const int K = cfg.num_experts_per_tok;
  require(static_cast<int>(run.states.size()) == V, "engine: one state per visit");
  const double l2_budget = relaxed ? 0.03 : 0.01;
  const double hard_budget = relaxed ? 0.02 : 0.005;
  int failures = 0;
  const size_t plane = static_cast<size_t>(T) * H;
  const uint16_t* ref_states = reinterpret_cast<const uint16_t*>(d.tensor("visit_states").data);
  const uint16_t* ref_attn = reinterpret_cast<const uint16_t*>(d.tensor("visit_attn").data);
  const int32_t* ref_routes = reinterpret_cast<const int32_t*>(d.tensor("route_ids").data);
  // Routing: flips per visit. End to end, visits from the first flip on are
  // reported and not judged (a flipped route moves everything after it).
  long flips = 0, slots = 0;
  int first_flip_visit = -1;
  for (int v = 0; v < V; ++v) {
    const auto& ids = run.route_ids[static_cast<size_t>(v)];
    long visit_flips = 0;
    for (int t = 0; t < T; ++t)
      for (int k = 0; k < K; ++k) {
        const int32_t want = ref_routes[(static_cast<size_t>(v) * T + t) * K + k];
        bool found = false;
        for (int j = 0; j < K; ++j) found = found || ids[static_cast<size_t>(t) * K + j] == want;
        visit_flips += found ? 0 : 1;
      }
    flips += visit_flips;
    slots += static_cast<long>(T) * K;
    if (visit_flips > 0 && first_flip_visit < 0) first_flip_visit = v;
  }
  // The experts' input is in the reference's own dumps. The expert chain is
  // judged on the engine's input (the teacher supplies it), so this row is
  // what holds the mix and its norm to the reference.
  const uint16_t* ref_mlp_in =
      d.tensors.count("visit_mlp_in") ? reinterpret_cast<const uint16_t*>(d.tensor("visit_mlp_in").data) : nullptr;
  for (int v = 0; v < V; ++v) {
    const dgpp::Inkling2Visit vi = cfg.visit(v);
    const Stats a = compare_bf16(run.attention[static_cast<size_t>(v)].data(), ref_attn + static_cast<size_t>(v) * plane, plane);
    const Stats s = compare_bf16(run.states[static_cast<size_t>(v)].data(), ref_states + static_cast<size_t>(v) * plane, plane);
    Stats x;
    if (ref_mlp_in)
      x = compare_bf16(run.mlp_input[static_cast<size_t>(v)].data(), ref_mlp_in + static_cast<size_t>(v) * plane, plane);
    // End to end under the native recipe only the first visit's attention
    // and expert input are judged: the activation quantizers turn a one-ulp
    // input difference into whole quantization steps, so everything from the
    // first expert output on is a different trajectory, reported here and
    // judged teacher-forced.
    const bool native_e2e = relaxed && native;
    const bool judged = native_e2e ? v == 0 : (!relaxed || first_flip_visit < 0 || v < first_flip_visit);
    const auto within = [&](const Stats& t) {
      return t.l2 <= l2_budget && static_cast<double>(t.hard) <= hard_budget * static_cast<double>(t.total);
    };
    const bool ok = within(a) && within(x) && (native_e2e || within(s));
    std::printf("visit %2d (layer %2d pass %d %s): attention l2 %.5f max %3.0f ulps hard %ld | expert input l2 %.5f max %3.0f "
                "ulps hard %ld | state l2 %.5f max %3.0f ulps soft %ld hard %ld of %ld%s\n",
                v, vi.layer, vi.pass, cfg.is_local_layer(vi.layer) ? "local" : "full ", a.l2, a.max_ulps, a.hard, x.l2,
                x.max_ulps, x.hard, s.l2, s.max_ulps, s.soft, s.hard, s.total,
                ok ? "" : (judged ? "  FAIL" : "  (not judged)"));
    if (!ok && judged) ++failures;
  }
  // End to end, a visit before the first routing flip must exist: a run whose
  // first visit already routes differently has judged nothing.
  if (relaxed && !native && first_flip_visit == 0) {
    std::printf("FAIL: the first visit routes differently from the reference\n");
    ++failures;
  }
  std::printf("routing: %ld of %ld slots differ (first at visit %d)\n", flips, slots, first_flip_visit);
  if (!relaxed && static_cast<double>(flips) > 0.05 * static_cast<double>(slots)) {
    std::printf("FAIL: routing flips above 5 %%\n");
    ++failures;
  }

  // The final hidden state and the logits.
  const bool tail_judged = !relaxed || (!native && first_flip_visit < 0);
  const Stats fh = compare_bf16(run.final_hidden.data(), reinterpret_cast<const uint16_t*>(d.tensor("final_hidden").data), plane);
  std::printf("final hidden: l2 %.5f max %.0f ulps soft %ld hard %ld\n", fh.l2, fh.max_ulps, fh.soft, fh.hard);
  if (tail_judged && (fh.l2 > l2_budget || static_cast<double>(fh.hard) > hard_budget * static_cast<double>(fh.total))) {
    std::printf("FAIL: final hidden\n");
    ++failures;
  }
  const int topk = d.top_k;
  const int32_t* ref_ids = reinterpret_cast<const int32_t*>(d.tensor("topk_ids").data);
  const float* ref_capped = reinterpret_cast<const float*>(d.tensor("topk_capped").data);
  int top1_miss = 0, near_ties = 0;
  double max_logit_err = 0;
  for (int t = 0; t < T; ++t) {
    const float* row = run.logits.data() + static_cast<size_t>(t) * cfg.vocab_size;
    int best = 0;
    for (int i = 1; i < cfg.vocab_size; ++i)
      if (row[i] > row[best]) best = i;
    const int32_t want = ref_ids[static_cast<size_t>(t) * topk];
    for (int k = 0; k < topk; ++k) {
      const int32_t id = ref_ids[static_cast<size_t>(t) * topk + k];
      max_logit_err = std::max<double>(max_logit_err, std::fabs(row[id] - ref_capped[static_cast<size_t>(t) * topk + k]));
    }
    if (best != want) {
      // A near tie: the engine's pick is within 1e-2 of the reference's best.
      if (std::fabs(row[best] - row[want]) <= 1e-2f) ++near_ties;
      else ++top1_miss;
    }
  }
  std::printf("logits: top-1 misses %d, near ties %d of %d rows, max |logit error| over the reference top-%d %.5f\n",
              top1_miss, near_ties, T, topk, max_logit_err);
  if (tail_judged && top1_miss > 0) {
    std::printf("FAIL: top-1\n");
    ++failures;
  }
  if (report) {
    std::printf("%s (%s activations against a %s dump): reported, not judged\n", label, native ? "native" : "bf16",
                d.activations.c_str());
    return 0;
  }
  if (failures) {
    std::printf("%s: %d failure(s)\n", label, failures);
    return 1;
  }
  std::printf("%s (%s activations, %s): PASS\n", label, native ? "native" : "bf16", relaxed ? "end to end" : "teacher-forced");
  return 0;
}

// Token ids from a text file: integers separated by white space or commas.
inline std::vector<int64_t> read_ids(const std::string& path) {
  std::ifstream f(path);
  require(f.good(), "ids: cannot open " + path);
  std::vector<int64_t> ids;
  std::string token;
  char c;
  const auto flush = [&] {
    if (!token.empty()) ids.push_back(std::stoll(token));
    token.clear();
  };
  while (f.get(c)) {
    if ((c >= '0' && c <= '9') || c == '-') token.push_back(c);
    else flush();
  }
  flush();
  require(ids.size() >= 2, "ids: fewer than two token ids in " + path);
  return ids;
}

// The teacher-forced likelihood of a token sequence under the engine's
// logits: row t scores token t + 1. Two runs over the same ids (another
// world size, another activation recipe) are compared position by position
// from the file this writes: one line per scored position with the negative
// log likelihood in nats, the row's best id and the id that followed.
inline int likelihood(const Inkling2Config& cfg, const std::vector<float>& logits, const std::vector<int64_t>& tokens,
                      const std::string& out_path, const char* label) {
  const int T = static_cast<int>(tokens.size()), V = cfg.vocab_size;
  require(logits.size() == static_cast<size_t>(T) * V, "likelihood: the logits are not one row per token");
  std::FILE* out = out_path.empty() ? nullptr : std::fopen(out_path.c_str(), "w");
  require(out_path.empty() || out != nullptr, "likelihood: cannot write " + out_path);
  double sum = 0;
  int top1 = 0;
  for (int t = 0; t + 1 < T; ++t) {
    const float* row = logits.data() + static_cast<size_t>(t) * V;
    int best = 0;
    for (int i = 1; i < V; ++i)
      if (row[i] > row[best]) best = i;
    double lse = 0;
    for (int i = 0; i < V; ++i) lse += std::exp(static_cast<double>(row[i]) - static_cast<double>(row[best]));
    const int64_t next = tokens[static_cast<size_t>(t) + 1];
    require(next >= 0 && next < V, "likelihood: a token id is outside the vocabulary");
    const double nll = std::log(lse) + static_cast<double>(row[best]) - static_cast<double>(row[next]);
    sum += nll;
    top1 += best == next ? 1 : 0;
    if (out) std::fprintf(out, "%d %.9g %d %lld\n", t, nll, best, static_cast<long long>(next));
  }
  if (out) std::fclose(out);
  const int n = T - 1;
  std::printf("%s: %d scored positions, mean negative log likelihood %.6f nats, perplexity %.4f, top-1 accuracy %.4f\n", label,
              n, sum / n, std::exp(sum / n), static_cast<double>(top1) / n);
  return 0;
}

}  // namespace ink2ref
