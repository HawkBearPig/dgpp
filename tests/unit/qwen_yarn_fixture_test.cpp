// The vLLM YaRN oracle, offline (2026-09-19). tests/python/qwen_yarn_oracle.py
// --emit runs the recipe's own container and writes
// tests/data/qwen_yarn_oracle.json: the YaRN inverse-frequency table, the
// attention mscale, the bf16 cos/sin cache rows and ROTATED Q/K rows (with
// one attention logit) at several positions, for the recipe and for four
// non-default parameter combinations, under the digest of the container that
// produced them. This gate reads that file and compares the ENGINE's numbers
// against it — dgpp::yarn_rope_inv_freq_host, dgpp::RopeScaling's mscale and
// correction band, and the qsa_reference rotation the QSA kernels mirror — so
// the vLLM parity is a normal unit test: no vLLM, no container, no GPU.
//
// What makes a rotated row a check on vLLM rather than on this file: the
// input rows come from the seed the fixture records, drawn straight to bf16
// BITS (splitmix64, one draw a lane — the fixture's own "spec" block spells
// that out and its "input_fnv" catches a drift in the drawing), and the
// arithmetic the fixture's rows were built with is the reference's, one fp32
// operation at a time with nothing fused (the repo builds with
// -ffp-contract=off, so the engine never fuses either). The vLLM half of the
// golden is the table, the mscale and the cache ROW VALUES; run those through
// the engine's own builder and rotation and the bits either match or this
// gate names the case, the position and the lane.
//
// The fixture's provenance is part of the contract, so the pinned image and
// its digest are checked too: a fixture nobody can attribute is a fixture
// nobody can trust. Refresh it inside the container — the oracle script's
// docstring carries the docker line — whenever vLLM's yarn code or the recipe
// changes, and --check there says whether the file still says what vLLM says.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/rope_scaling.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/qsa_reference.hpp"

namespace {

using dgpp::minijson::Value;

std::string g_fixture_path =
#ifdef DGPP_SOURCE_DIR
    std::string(DGPP_SOURCE_DIR) + "/tests/data/qwen_yarn_oracle.json"
#else
    "tests/data/qwen_yarn_oracle.json"
#endif
    ;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// ---- reading the fixture ----

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open the fixture " + path + " (regenerate it with "
                             "tests/python/qwen_yarn_oracle.py --emit, inside the pinned container)");
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

const Value& fixture() {
  static const std::string text = read_file(g_fixture_path);  // the tree views into it
  static const Value root = dgpp::minijson::parse(text).root;
  return root;
}

std::string str(const Value& v) { return std::string(v.as_string()); }

int nibble(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// "0x…" (a single value) or a packed run of `width` hex characters a value.
std::vector<uint32_t> unhex(const std::string& s, int width) {
  std::string h = s;
  if (h.rfind("0x", 0) == 0 || h.rfind("0X", 0) == 0) h = h.substr(2);
  if (h.empty() || h.size() % width != 0)
    throw std::runtime_error("fixture bit string \"" + s + "\" is not a whole number of " +
                             std::to_string(width) + "-character values");
  std::vector<uint32_t> out;
  for (size_t i = 0; i < h.size(); i += static_cast<size_t>(width)) {
    uint32_t v = 0;
    for (int k = 0; k < width; ++k) {
      const int n = nibble(h[i + static_cast<size_t>(k)]);
      require(n >= 0, "fixture bit string \"" + s + "\" has a non-hex character");
      v = (v << 4) | static_cast<uint32_t>(n);
    }
    out.push_back(v);
  }
  return out;
}

std::vector<uint16_t> unhex_bf16(const std::string& s) {
  std::vector<uint16_t> out;
  for (const uint32_t v : unhex(s, 4)) out.push_back(static_cast<uint16_t>(v));
  return out;
}

uint32_t bits_of(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  return u;
}

float float_of(uint32_t u) {
  float f = 0.f;
  std::memcpy(&f, &u, 4);
  return f;
}

std::string hex(uint32_t u, int width = 8) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%0*x", width, u);
  return std::string(buf);
}

std::string hex64(uint64_t u) { return hex(static_cast<uint32_t>(u >> 32)) + hex(static_cast<uint32_t>(u)); }

// ---- the fixture's deterministic input (its "spec" block) ----

struct SplitMix64 {
  uint64_t s;
  uint64_t next() {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
};

// x[rows][dim] then w[rows][dim], one draw a lane, straight to bf16 bits.
void fixture_input(uint64_t seed, int rows, int dim, std::vector<uint16_t>& x,
                   std::vector<uint16_t>& w) {
  SplitMix64 r{seed};
  x.assign(static_cast<size_t>(rows) * static_cast<size_t>(dim), 0);
  w.assign(static_cast<size_t>(rows) * static_cast<size_t>(dim), 0);
  for (auto& v : x) {
    const uint64_t z = r.next();
    v = static_cast<uint16_t>(((z >> 63) << 15) | ((124 + (z % 5)) << 7) | ((z >> 11) & 0x7F));
  }
  for (auto& v : w) {
    const uint64_t z = r.next();
    v = static_cast<uint16_t>(((z >> 63) << 15) | (125ull << 7) | ((z >> 11) & 0x7F));
  }
  // Every lane has to be a plain finite number: the fixture is a gate on the
  // rope, not on what bf16 does to an infinity.
  for (const uint16_t v : x)
    require(std::isfinite(dgpp::bf16_bits_to_float(v)), "the drawn input has a non-finite lane");
  for (const uint16_t v : w)
    require(std::isfinite(dgpp::bf16_bits_to_float(v)), "the drawn weights have a non-finite lane");
}

uint64_t fnv64(const std::vector<uint16_t>& v) {
  uint64_t h = 0xCBF29CE484222325ull;
  for (const uint16_t x : v)
    for (const uint32_t b : {static_cast<uint32_t>(x & 0xFF), static_cast<uint32_t>(x >> 8)})
      h = (h ^ b) * 0x100000001B3ull;
  return h;
}

struct Fixture {
  int rows = 0, dim = 0, rotary = 0;
  double theta = 0.0;
  float eps = 0.f;
  uint64_t seed = 0;
  std::vector<uint16_t> x, w;  // [rows][dim]
  const Value* cases = nullptr;
};

Fixture load_fixture() {
  const Value& fx = fixture();
  require(fx.at("format").as_int() == 1, "the fixture's format is not 1 (this gate is older than it)");
  Fixture f;
  f.rows = static_cast<int>(fx.at("spec").at("rows").as_int());
  f.dim = static_cast<int>(fx.at("checkpoint").at("head_dim").as_int());
  f.rotary = static_cast<int>(fx.at("checkpoint").at("rotary_dim").as_int());
  f.theta = fx.at("checkpoint").at("theta").as_double();
  f.eps = float_of(unhex(str(fx.at("checkpoint").at("rms_norm_eps_bits")), 8)[0]);
  f.seed = std::stoull(str(fx.at("spec").at("seed")), nullptr, 16);
  fixture_input(f.seed, f.rows, f.dim, f.x, f.w);
  std::vector<uint16_t> both = f.x;
  both.insert(both.end(), f.w.begin(), f.w.end());
  const std::string want = str(fx.at("spec").at("input_fnv"));
  require(hex64(fnv64(both)) == (want.rfind("0x", 0) == 0 ? want.substr(2) : want),
          "the fixture's input spec no longer draws the rows it recorded (seed 0x" + hex64(f.seed) + ")");
  f.cases = fx.find("cases");
  require(f.cases != nullptr && f.cases->is_array() && !f.cases->items().empty(),
          "the fixture has no cases");
  return f;
}

// The knob a case describes, as the cluster config would hand it over.
dgpp::RopeScaling scaling_of(const Value& params) {
  dgpp::RopeScaling rs;
  rs.factor = params.at("factor").as_double();
  rs.original_max_position_embeddings = params.at("original_max_position_embeddings").as_int();
  rs.beta_fast = params.at("beta_fast").as_double();
  rs.beta_slow = params.at("beta_slow").as_double();
  rs.attn_factor = params.at("attn_factor").as_double();
  rs.mrope_cache_factor = params.at("mrope_cache_factor").as_double();
  return rs;
}

// The engine's cos/sin for one position: the two lines qsa_reference.cpp's
// rotation and kernels/qsa.cu's norm_rope_thread build, written out here so a
// failure can name the lane instead of a rotated tensor.
void engine_cos_sin(const std::vector<float>& inv, float mscale, int64_t pos, int rotary,
                    std::vector<uint16_t>& out) {
  out.assign(static_cast<size_t>(rotary), 0);
  const int half = rotary / 2;
  for (int i = 0; i < half; ++i) {
    const float ang = static_cast<float>(pos) * inv[static_cast<size_t>(i)];
    out[static_cast<size_t>(i)] = dgpp::float_to_bf16_bits(std::cos(ang) * mscale);
    out[static_cast<size_t>(half + i)] = dgpp::float_to_bf16_bits(std::sin(ang) * mscale);
  }
}

// The fixture's logit: the fp32 dot of two rotated rows in lane order, times
// head_dim**-0.5 — one number that carries the mscale^2 the rotated lanes
// pick up. (head_dim is 256, so the 1/16 scale is exact either way.)
float logit_of(const std::vector<uint16_t>& q, const std::vector<uint16_t>& k, int dim) {
  float acc = 0.f;
  for (int d = 0; d < dim; ++d)
    acc += dgpp::bf16_bits_to_float(q[static_cast<size_t>(d)]) *
           dgpp::bf16_bits_to_float(k[static_cast<size_t>(d)]);
  return acc / std::sqrt(static_cast<float>(dim));
}

// The table a case's knob builds, from the engine's own builder.
std::vector<float> engine_table(const Fixture& f, const dgpp::RopeScaling& rs) {
  std::vector<float> inv(static_cast<size_t>(f.rotary / 2), 0.f);
  dgpp::yarn_rope_inv_freq_host(f.rotary, f.theta, rs.correction_max_position(), rs.factor, rs.beta_fast,
                                rs.beta_slow, inv.data());
  return inv;
}

}  // namespace

DGPP_TEST(qwen_yarn_fixture_provenance_is_the_pinned_container) {
  const Value& ctr = fixture().at("container");
  require(str(ctr.at("image")) == "vllm/vllm-openai:qwen38-flash-next",
          "the fixture came from another image than the recipe's: " + str(ctr.at("image")));
  const std::string digest = str(ctr.at("image_digest"));
  require(digest.rfind("sha256:", 0) == 0 && digest.size() == 71,
          "the fixture recorded no usable container digest (regenerate it with --image-digest or "
          "$DGPP_ORACLE_IMAGE_DIGEST): \"" + digest + "\"");
  for (size_t i = 7; i < digest.size(); ++i)
    require(nibble(digest[i]) >= 0, "the fixture's container digest is not hex");
  std::printf("[ .. ] yarn fixture: %s @ %.19s…  vllm %s, torch %s\n", str(ctr.at("image")).c_str(),
              digest.c_str() + 7, str(ctr.at("vllm_version")).c_str(), str(ctr.at("torch_version")).c_str());
}

DGPP_TEST(qwen_yarn_fixture_input_is_the_one_the_fixture_recorded) {
  const Fixture f = load_fixture();
  require(f.dim == 256 && f.rotary == 64, "the fixture's geometry is not the checkpoint's 256/64");
  require(f.theta == 1e7, "the fixture's rope_theta is not the checkpoint's 1e7");
  require(bits_of(f.eps) == 0x358637BDu, "the fixture's rms_norm_eps is not the checkpoint's 1e-6");
  std::printf("[ .. ] yarn fixture: seed 0x%s drew %d rows x %d lanes\n", hex64(f.seed).c_str(), f.rows,
              f.dim);
}

DGPP_TEST(qwen_yarn_fixture_table_mscale_and_band_match_vllm) {
  const Fixture f = load_fixture();
  int lanes = 0;
  for (const Value& c : f.cases->items()) {
    const std::string name = str(c.at("name"));
    const dgpp::RopeScaling rs = scaling_of(c.at("params"));
    rs.validate("fixture case " + name);
    require(bits_of(rs.mscale()) == unhex(str(c.at("mscale_bits")), 8)[0],
            "case " + name + ": mscale() is 0x" + hex(bits_of(rs.mscale())) +
                ", vLLM built its cache with 0x" + hex(unhex(str(c.at("mscale_bits")), 8)[0]));
    const int64_t band = c.at("band").at("correction_max_position").as_int();
    require(rs.correction_max_position() == band,
            "case " + name + ": the knob's correction band is " + std::to_string(rs.correction_max_position()) +
                ", vLLM's rope's own max_position_embeddings is " + std::to_string(band));
    const std::vector<float> inv = engine_table(f, rs);
    const std::vector<uint32_t> want = unhex(str(c.at("inv_freq")), 8);
    require(want.size() == inv.size(), "case " + name + ": the fixture recorded " + std::to_string(want.size()) +
                                           " table lanes, the checkpoint's rotary_dim has " +
                                           std::to_string(inv.size()));
    for (size_t i = 0; i < want.size(); ++i)
      require(bits_of(inv[i]) == want[i], "case " + name + ": table lane " + std::to_string(i) + " is 0x" +
                                              hex(bits_of(inv[i])) + ", vLLM built 0x" + hex(want[i]));
    lanes += static_cast<int>(want.size());
    const int64_t last_pos = c.at("rows").items().back().at("pos").as_int();
    require(rs.context_limit() > last_pos, "case " + name + ": the fixture sampled position " +
                                               std::to_string(last_pos) + " at or past the knob's context limit " +
                                               std::to_string(rs.context_limit()));
  }
  std::printf("[ .. ] yarn fixture: %d table lanes, %d mscale and band pairs match vLLM\n", lanes,
              static_cast<int>(f.cases->items().size()));
}

DGPP_TEST(qwen_yarn_fixture_cos_sin_rows_match_the_vllm_cache) {
  const Fixture f = load_fixture();
  int rows = 0;
  for (const Value& c : f.cases->items()) {
    const std::string name = str(c.at("name"));
    const dgpp::RopeScaling rs = scaling_of(c.at("params"));
    const std::vector<float> inv = engine_table(f, rs);
    for (const Value& r : c.at("rows").items()) {
      const int64_t pos = r.at("pos").as_int();
      const std::vector<uint16_t> want = unhex_bf16(str(r.at("cache_row")));
      require(want.size() == static_cast<size_t>(f.rotary),
              "case " + name + ": the recorded cache row is not rotary-wide");
      std::vector<uint16_t> got;
      engine_cos_sin(inv, rs.mscale(), pos, f.rotary, got);
      for (size_t i = 0; i < want.size(); ++i)
        require(got[i] == want[i], "case " + name + " pos " + std::to_string(pos) + " lane " + std::to_string(i) +
                                       ": the engine's cos/sin is 0x" + hex(got[i], 4) +
                                       ", vLLM's cache holds 0x" + hex(want[i], 4));
      ++rows;
    }
  }
  std::printf("[ .. ] yarn fixture: %d bf16 cos/sin rows match vLLM's cache\n", rows);
}

DGPP_TEST(qwen_yarn_fixture_rotated_q_k_and_their_logits_match_vllm) {
  const Fixture f = load_fixture();
  int rows = 0;
  for (const Value& c : f.cases->items()) {
    const std::string name = str(c.at("name"));
    const dgpp::RopeScaling rs = scaling_of(c.at("params"));
    const std::vector<float> inv = engine_table(f, rs);
    const float mscale = rs.mscale();
    for (const Value& r : c.at("rows").items()) {
      const int64_t pos = r.at("pos").as_int();
      const std::string where = "case " + name + " pos " + std::to_string(pos);
      const char* label[2] = {"q", "k"};
      std::vector<uint16_t> got[2];
      for (int row = 0; row < f.rows && row < 2; ++row) {
        got[row].assign(static_cast<size_t>(f.dim), 0);
        // The engine's rotation path: the host reference the QSA kernels are
        // gated against, on the YaRN table and the mscale it rides.
        dgpp::qwen_ref::qsa_norm_rope(f.x.data() + static_cast<size_t>(row) * f.dim,
                                      f.w.data() + static_cast<size_t>(row) * f.dim, pos, inv.data(),
                                      got[row].data(), f.dim, f.rotary, f.eps, mscale);
        const std::vector<uint16_t> want = unhex_bf16(str(r.at(label[row])));
        require(want.size() == got[row].size(), where + ": the fixture's " + label[row] +
                                                    " row is not head_dim wide");
        for (size_t d = 0; d < want.size(); ++d)
          require(want[d] == got[row][d], where + " " + label[row] + " lane " + std::to_string(d) +
                                              ": the engine rotated 0x" + hex(got[row][d], 4) +
                                              ", vLLM's table rotates to 0x" + hex(want[d], 4));
      }
      require(bits_of(logit_of(got[0], got[1], f.dim)) == unhex(str(r.at("logit")), 8)[0],
              where + ": the attention logit is 0x" + hex(bits_of(logit_of(got[0], got[1], f.dim))) +
                  ", vLLM's table gives 0x" + hex(unhex(str(r.at("logit")), 8)[0]));
      ++rows;
    }
  }
  std::printf("[ .. ] yarn fixture: %d rotated Q/K rows and their logits match vLLM\n", rows);
}

int main(int argc, char** argv) {
  if (argc > 1) g_fixture_path = argv[1];
  return dgpp::test::run_all();
}
