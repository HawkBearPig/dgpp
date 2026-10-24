// Route-trace format tests (host-only): write->read roundtrip, corruption
// rejection, and the golden bytes shared with tests/python/route_trace_test.py
// so the C++ writer and the python reader cannot drift apart.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/temp_dir.hpp"
#include "common/test.hpp"
#include "models/glm/trace.hpp"

namespace {

using dgpp::GlmRouteTraceLayer;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::vector<GlmRouteTraceLayer> sample_trace() {
  GlmRouteTraceLayer l;
  l.layer_idx = 3;
  l.top_k = 2;
  l.tokens = 3;
  l.ids = {0, 5, 5, 0, 7, 3};
  l.weights = {0.25f, 0.75f, 0.5f, 0.5f, 1.0f, 0.0f};
  return {l};
}

}  // namespace

DGPP_TEST(glm_trace_roundtrip_write_read) {
  const dgpp::test::TempDir fixture("dgpp_glm_trace");
  const auto path = fixture.path() / "dgpp_glm_trace_rt.bin";
  const auto layers = sample_trace();
  dgpp::glm_trace_write(path.string(), layers);
  const auto back = dgpp::glm_trace_read(path.string());
  require(back.size() == 1, "one layer");
  require(back[0].layer_idx == 3 && back[0].top_k == 2 && back[0].tokens == 3,
          "header roundtrip");
  require(back[0].ids == layers[0].ids, "ids roundtrip");
  require(back[0].weights == layers[0].weights, "weights roundtrip");
  std::filesystem::remove(path);
}

DGPP_TEST(glm_trace_writer_pins_the_golden_bytes) {
  // Byte-for-byte the same record the python reader test parses. If this
  // assertion fails, the C++ writer changed the format; if the python one
  // fails, the reader did. They must fail together or not at all.
  const dgpp::test::TempDir fixture("dgpp_glm_trace");
  const auto path = fixture.path() / "dgpp_glm_trace_golden.bin";
  dgpp::glm_trace_write(path.string(), sample_trace());

  const uint8_t want[] = {
      'D', 'G', 'P', 'P', 'T', 'C', '1', 0x01,
      0x01, 0x00, 0x00, 0x00,             // num_layers = 1
      0x03, 0x00, 0x00, 0x00,             // layer_idx = 3
      0x02, 0x00, 0x00, 0x00,             // top_k = 2
      0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // tokens = 3
      // ids: 0, 5, 5, 0, 7, 3 (i32 LE)
      0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
      0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x07, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
      // weights: 0.25, 0.75, 0.5, 0.5, 1.0, 0.0 (f32 LE)
      0x00, 0x00, 0x80, 0x3E, 0x00, 0x00, 0x40, 0x3F,
      0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x3F,
      0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x00,
  };
  std::FILE* f = std::fopen(path.c_str(), "rb");
  require(f != nullptr, "open golden file");
  uint8_t got[sizeof(want)];
  const size_t n = std::fread(got, 1, sizeof(got), f);
  std::fclose(f);
  std::filesystem::remove(path);
  require(n == sizeof(want), "golden size");
  require(std::memcmp(got, want, sizeof(want)) == 0,
          "golden bytes: format drift between C++ writer and spec");
}

DGPP_TEST(glm_trace_rejects_corruption) {
  const dgpp::test::TempDir fixture("dgpp_glm_trace");
  const auto path = fixture.path() / "dgpp_glm_trace_bad.bin";
  dgpp::glm_trace_write(path.string(), sample_trace());

  // Truncated file.
  {
    bool threw = false;
    try {
      std::FILE* f = std::fopen(path.c_str(), "rb");
      std::vector<uint8_t> half(32);
      const size_t n = std::fread(half.data(), 1, 32, f);
      std::fclose(f);
      std::FILE* w = std::fopen(path.c_str(), "wb");
      std::fwrite(half.data(), 1, n, w);
      std::fclose(w);
      (void)dgpp::glm_trace_read(path.string());
    } catch (const std::runtime_error&) {
      threw = true;
    }
    require(threw, "truncated trace rejected");
  }
  // Bad magic.
  {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    const uint8_t junk[16] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::fwrite(junk, 1, 16, f);
    std::fclose(f);
    bool threw = false;
    try {
      (void)dgpp::glm_trace_read(path.string());
    } catch (const std::runtime_error&) {
      threw = true;
    }
    require(threw, "bad magic rejected");
  }
  std::filesystem::remove(path);
}
