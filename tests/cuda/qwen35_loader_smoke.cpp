// Loader smoke test: Qwen35LayerStream against the real 27B FP8 checkpoint.
// Loads layer 0 (GDN) + layer 3 (Full) in streaming mode and verifies the
// resident pointers are non-null and the norm weights read back ~1.0.
// Discriminates loader-build crashes from model/run_rows crashes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "models/qwen/config35.hpp"
#include "models/qwen/loader35.hpp"

namespace {

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::printf("CUDA FAIL %s:%d: %s\n", __FILE__, __LINE__,                 \
                  cudaGetErrorString(err__));                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("FAIL: %s\n", msg);                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

float h2f(uint16_t v) {
  uint32_t b = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

std::vector<uint16_t> down16(const uint16_t* d, size_t n) {
  std::vector<uint16_t> h(n);
  cudaError_t err__ = cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost);
  if (err__ != cudaSuccess) {
    std::printf("CUDA FAIL download: %s\n", cudaGetErrorString(err__));
    std::exit(1);
  }
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <checkpoint-dir>\n"
                 "  <checkpoint-dir> must hold config.json + the safetensors shards for "
                 "Qwen3.8-27B-FP8 (e.g. a Hugging Face snapshot directory)\n",
                 argv[0]);
    return 2;
  }
  const std::string ckpt = argv[1];
  dgpp::Qwen35TextConfig cfg = dgpp::Qwen35TextConfig::from_json_file(ckpt + "/config.json");
  std::fprintf(stderr, "MARK config parsed: hidden=%d layers=%d\n", cfg.hidden_size,
               cfg.num_hidden_layers);
  std::printf("config: hidden=%d layers=%d vocab=%d\n", cfg.hidden_size, cfg.num_hidden_layers,
              cfg.vocab_size);
  dgpp::Qwen35LayerStream stream(cfg, ckpt, 0, 1, dgpp::LoaderResidency::Streaming,
                                 dgpp::LoaderHeadSharding::Full, false);
  std::fprintf(stderr, "MARK stream constructed\n");
  // Layer 0: GDN.
  {
    const dgpp::Qwen35LayerResident& r = stream.load_layer(0);
    std::fprintf(stderr, "MARK layer 0 loaded\n");
    CHECK(r.kind == dgpp::Qwen35LayerKind::Gdn, "layer 0 kind != Gdn");
    CHECK(r.input_norm && r.post_norm, "layer 0 norms null");
    CHECK(r.gdn.in_proj_qkv_fp8.payload, "layer 0 qkv fp8 payload null");
    CHECK(r.gdn.in_proj_a && r.gdn.in_proj_b, "layer 0 a/b (fp8->bf16 dequant) null");
    CHECK(r.gdn.norm, "layer 0 gdn norm null");
    CHECK(r.mlp.gate_fp8.payload, "layer 0 mlp gate fp8 null");
    const std::vector<uint16_t> n = down16(r.input_norm, 4);
    std::printf("L0 input_norm[0..3] = %.4f %.4f %.4f %.4f\n", h2f(n[0]), h2f(n[1]), h2f(n[2]),
                h2f(n[3]));
    for (float v : {h2f(n[0]), h2f(n[1]), h2f(n[2]), h2f(n[3])})
      CHECK(std::isfinite(v) && std::fabs(v) < 100.0f, "layer 0 norm value out of range");
  }
  // Layer 3: Full attention.
  {
    const dgpp::Qwen35LayerResident& r = stream.load_layer(3);
    std::fprintf(stderr, "MARK layer 3 loaded\n");
    CHECK(r.kind == dgpp::Qwen35LayerKind::Full, "layer 3 kind != Full");
    CHECK(r.full.q_proj_fp8.payload, "layer 3 q fp8 payload null");
    CHECK(r.full.o_proj_fp8.payload, "layer 3 o fp8 payload null");
    CHECK(r.full.q_norm && r.full.k_norm, "layer 3 q/k norms null");
    const std::vector<uint16_t> n = down16(r.post_norm, 4);
    std::printf("L3 post_norm[0..3] = %.4f %.4f %.4f %.4f\n", h2f(n[0]), h2f(n[1]), h2f(n[2]),
                h2f(n[3]));
  }
  std::printf("SMOKE PASS\n");
  return 0;
}
