// The EXL3 host codec (models/exl3_matrix.hpp) pinned against the numpy
// reference that decoded the LibertAIDAI DeepSeek-V4.1-Flash-1017 3.5 bpw
// pack on 2026-10-22 (docs/deepseek_v41_flash_exl3_plan.md §2.4): the mul1
// codebook's values, the full table's statistics, one real 6-bit tile of
// `layers.0.attn.wq_a` (its stream bytes embedded) decoded to the states
// and values that reference produced, the n-tile-major re-tiling, the
// 128-point rotation's identities, and the strict oracle against the full
// reconstruct. CUDA-free.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "models/exl3_matrix.hpp"

namespace {

using dgpp::bf16_bits_to_float;
using dgpp::Exl3HostMatrix;
using dgpp::float_to_bf16_bits;
using dgpp::fp16_bits_to_float;
using dgpp::kExl3Block;
using dgpp::kExl3Tile;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed | 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  double unit() { return static_cast<double>(next() >> 11) / static_cast<double>(1ull << 52) - 1.0; }
};

// Tile (kt 0, nt 0) of layers.0.attn.wq_a.trellis (I16 [320, 80, 96]): 96
// int16 words = 192 bytes as stored.
const uint8_t kRealTile[192] = {
    0x14, 0x2b, 0x6d, 0x94, 0xec, 0x2b, 0xda, 0x2d, 0x83, 0x46, 0x74, 0x83, 0x0f, 0x27, 0x76, 0x65, 0xb2, 0x24, 0x21,
    0x98, 0x72, 0xa3, 0x91, 0x87, 0xc5, 0xcd, 0xa4, 0x6c, 0x35, 0x0a, 0xc1, 0x89, 0x99, 0xcf, 0x50, 0xa6, 0x6e, 0xca,
    0xb1, 0xc0, 0x77, 0x79, 0xee, 0xfe, 0x25, 0xcf, 0x76, 0x74, 0xa0, 0x9b, 0xd9, 0xef, 0xbe, 0x7b, 0x39, 0x10, 0xef,
    0x20, 0x2a, 0xad, 0xf1, 0xbb, 0x7d, 0x30, 0x3c, 0xa8, 0x96, 0x33, 0x53, 0xdd, 0x0e, 0xe6, 0xb3, 0xae, 0xe3, 0x6a,
    0xe6, 0x16, 0xf3, 0xf6, 0x42, 0x4e, 0xff, 0x82, 0xd0, 0x85, 0xdb, 0x80, 0x77, 0x0e, 0x9c, 0x4b, 0x74, 0x22, 0x26,
    0xed, 0x9f, 0x64, 0xd4, 0x29, 0xa1, 0x6c, 0xf3, 0x58, 0xde, 0x9a, 0xc7, 0xa7, 0x9c, 0x0b, 0xe0, 0x96, 0x28, 0xa0,
    0x4f, 0xbf, 0xe8, 0xc3, 0x77, 0xfe, 0xb2, 0x1a, 0xd5, 0x99, 0x78, 0x08, 0x5f, 0x40, 0xd1, 0x2d, 0x15, 0x13, 0x92,
    0x81, 0xd4, 0x1a, 0x99, 0x11, 0x5c, 0xc3, 0x35, 0x75, 0xe5, 0x53, 0x39, 0xf8, 0xab, 0xde, 0xc3, 0x6d, 0xf7, 0xc6,
    0x39, 0x53, 0xc4, 0x69, 0x7f, 0x1b, 0x72, 0x06, 0x21, 0x29, 0xc1, 0xf8, 0x5e, 0x76, 0x39, 0x96, 0x7c, 0xff, 0x8b,
    0x59, 0x71, 0xb0, 0x44, 0x00, 0x75, 0x10, 0xa1, 0x01, 0x27, 0xc5, 0x17, 0x8b, 0xa3, 0xa7, 0xd7, 0x78, 0x2b, 0x74,
    0x29, 0x0f};

void test_mul1_pins() {
  struct Pin {
    uint32_t state;
    float value;
    uint16_t bits;
  };
  const Pin pins[] = {{0, -3.453125f, 0xc2e8},          {1, 0.64111328125f, 0x3921},   {2, -0.448486328125f, 0xb72d},
                      {255, 0.83740234375f, 0x3ab3},    {256, -0.2454833984375f, 0xb3db}, {4096, -0.53662109375f, 0xb84b},
                      {12345, -1.4091796875f, 0xbda3},  {32768, -0.8681640625f, 0xbaf2}, {54321, -0.84765625f, 0xbac8},
                      {65535, -0.6513671875f, 0xb936}};
  for (const Pin& p : pins) {
    const float v = dgpp::exl3_mul1_value(p.state);
    require(v == p.value, "mul1 value of state " + std::to_string(p.state) + ": " + std::to_string(v));
    require(dgpp::float_to_fp16_bits(v) == p.bits, "mul1 fp16 bits of state " + std::to_string(p.state));
  }
  const std::vector<float>& t = dgpp::exl3_mul1_table();
  require(t.size() == 65536, "table size");
  double sum = 0.0;
  std::vector<float> sorted(t);
  for (float v : t) {
    require(std::isfinite(v) && std::fabs(v) < 3.6f, "a codebook value outside the fp16 range the reference has");
    sum += v;
  }
  std::sort(sorted.begin(), sorted.end());
  size_t distinct = 0;
  for (size_t i = 0; i < sorted.size(); ++i)
    if (i == 0 || sorted[i] != sorted[i - 1]) ++distinct;
  require(distinct == 913, "distinct codebook values: " + std::to_string(distinct));
  require(std::fabs(sum - (-126.55587005615234)) < 1e-6, "codebook sum: " + std::to_string(sum));
}

void test_real_tile() {
  const int bits = 6;
  // The states the reference extracted at positions 0..7 and 248..255.
  const uint32_t head[8] = {2789, 47430, 20916, 27947, 19141, 45378, 20663, 11738};
  const uint32_t tail[8] = {13801, 31354, 40588, 41743, 50122, 62103, 42448, 29739};
  for (int j = 0; j < 8; ++j) {
    require(dgpp::exl3_tile_state(kRealTile, bits, j) == head[j], "state at position " + std::to_string(j));
    require(dgpp::exl3_tile_state(kRealTile, bits, 248 + j) == tail[j], "state at position " + std::to_string(248 + j));
  }
  // The raw stream words as the reference read them (LE uint32).
  require(dgpp::exl3_tile_word(kRealTile, 0) == 0x946d2b14u && dgpp::exl3_tile_word(kRealTile, 1) == 0x2dda2becu &&
              dgpp::exl3_tile_word(kRealTile, 2) == 0x83744683u && dgpp::exl3_tile_word(kRealTile, 3) == 0x6576270fu,
          "stream words");
  float w[kExl3Tile * kExl3Tile];
  dgpp::exl3_decode_tile(kRealTile, bits, w);
  const float row0[8] = {0.7021484375f, -0.6376953125f, 0.8984375f, 1.0947265625f,
                         1.744140625f,  -0.10333251953125f, 1.6630859375f, -0.42822265625f};
  const float row15[8] = {-0.556640625f, -0.3671875f, -0.33349609375f, 0.11322021484375f,
                          1.337890625f,  -0.4755859375f, -0.394287109375f, 0.282470703125f};
  for (int c = 0; c < 8; ++c) {
    require(w[0 * kExl3Tile + c] == row0[c], "w_inner[0][" + std::to_string(c) + "] = " + std::to_string(w[c]));
    require(w[15 * kExl3Tile + 8 + c] == row15[c], "w_inner[15][" + std::to_string(8 + c) + "]");
  }
  // The permutation is a bijection onto the tile.
  std::vector<int> seen(256, 0);
  for (int p = 0; p < 256; ++p) {
    int r, c;
    dgpp::exl3_tile_position(p, r, c);
    require(r >= 0 && r < 16 && c >= 0 && c < 16, "position out of the tile");
    seen[r * 16 + c] += 1;
  }
  for (int i = 0; i < 256; ++i) require(seen[i] == 1, "the tile permutation is not a bijection");
}

void test_retile() {
  // A [kt 3, nt 5] trellis at 3 bits whose tile bytes name the tile.
  const int bits = 3, kt_full = 3, nt_full = 5;
  const size_t tb = dgpp::exl3_tile_bytes(bits);
  std::vector<uint8_t> src(static_cast<size_t>(kt_full) * nt_full * tb);
  for (int kt = 0; kt < kt_full; ++kt)
    for (int nt = 0; nt < nt_full; ++nt)
      for (size_t i = 0; i < tb; ++i) src[(static_cast<size_t>(kt) * nt_full + nt) * tb + i] = static_cast<uint8_t>(kt * 16 + nt + (i & 1 ? 128 : 0));
  std::vector<uint8_t> dst(static_cast<size_t>(2) * 2 * tb);
  dgpp::exl3_retile(src.data(), kt_full, nt_full, bits, /*kt0=*/1, /*kt1=*/3, /*nt0=*/3, /*nt1=*/5, dst.data());
  for (int nt = 3; nt < 5; ++nt)
    for (int kt = 1; kt < 3; ++kt) {
      const uint8_t* t = dst.data() + (static_cast<size_t>(nt - 3) * 2 + static_cast<size_t>(kt - 1)) * tb;
      require(t[0] == kt * 16 + nt && t[1] == kt * 16 + nt + 128 && t[tb - 2] == kt * 16 + nt, "retiled tile bytes");
    }
  bool threw = false;
  try {
    dgpp::exl3_retile(src.data(), kt_full, nt_full, bits, 0, 4, 0, 1, dst.data());
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "an out-of-range tile range is refused");
}

void test_hadamard() {
  Rng rng(7);
  float v[kExl3Block], w[kExl3Block];
  for (int j = 0; j < kExl3Block; ++j) v[j] = w[j] = static_cast<float>(rng.unit());
  // The explicit Sylvester matrix over sqrt(128): H[a][b] = (-1)^popcount(a & b) / sqrt(128).
  double ref[kExl3Block];
  for (int a = 0; a < kExl3Block; ++a) {
    double s = 0.0;
    for (int b = 0; b < kExl3Block; ++b) s += (__builtin_popcount(a & b) & 1 ? -1.0 : 1.0) * w[b];
    ref[a] = s / std::sqrt(128.0);
  }
  dgpp::exl3_hadamard128_f32(v, dgpp::kExl3HadamardScale);
  for (int j = 0; j < kExl3Block; ++j)
    require(std::fabs(v[j] - ref[j]) < 1e-5, "rotation vs the Sylvester matrix at " + std::to_string(j));
  dgpp::exl3_hadamard128_f32(v, dgpp::kExl3HadamardScale);
  for (int j = 0; j < kExl3Block; ++j) require(std::fabs(v[j] - w[j]) < 1e-5, "the rotation is an involution");
}

void test_oracle_vs_reconstruct() {
  Rng rng(11);
  for (int bits : {3, 4, 6, 8}) {
    const int64_t N = 256, K = 384;
    Exl3HostMatrix m = dgpp::exl3_random_matrix(rng, N, K, bits, 0.03f, 1.0f);
    const std::vector<double> inner = dgpp::exl3_inner_weights(m);
    const std::vector<double> W = dgpp::exl3_reconstruct(m);  // [N, K]
    std::vector<uint16_t> x(static_cast<size_t>(K));
    for (auto& e : x) e = float_to_bf16_bits(static_cast<float>(rng.unit()) * 4.0f);
    std::vector<double> y;
    dgpp::exl3_oracle_row(m, inner, x.data(), y);
    double num = 0.0, den = 0.0;
    for (int64_t n = 0; n < N; ++n) {
      double ref = 0.0;
      for (int64_t k = 0; k < K; ++k) ref += bf16_bits_to_float(x[k]) * W[static_cast<size_t>(n) * K + k];
      num += (y[n] - ref) * (y[n] - ref);
      den += ref * ref;
    }
    const double rel = std::sqrt(num / den);
    // The oracle rounds the rotated input to fp16 (2^-12 relative per
    // element, uncorrelated); the reconstruct does not.
    require(rel < 1e-3, "oracle vs reconstruct at " + std::to_string(bits) + " bits: rel " + std::to_string(rel));
    require(rel > 0.0, "the oracle equals the reconstruct exactly (the fp16 rounding is missing)");
  }
}

void test_geometry() {
  bool threw = false;
  try {
    dgpp::exl3_check_geometry(256, 192, 4, "test");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "K not a multiple of 128 is refused");
  threw = false;
  try {
    dgpp::exl3_check_geometry(256, 256, 9, "test");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "9 bits is refused");
  Rng rng(3);
  Exl3HostMatrix m = dgpp::exl3_random_matrix(rng, 384, 128, 3, 0.03f, 1.0f);
  const dgpp::Exl3Matrix v = dgpp::exl3_rows_view(m.view(), 128, 256);
  require(v.rows == 256 && v.cols == 128 && v.svh == m.svh.data() + 128 &&
              v.tiles == m.tiles.data() + 8 * m.view().n_tile_bytes(),
          "the rows view's pointers");
  threw = false;
  try {
    dgpp::exl3_rows_view(m.view(), 64, 128);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "a misaligned rows view is refused");
}

}  // namespace

int main() {
  struct Case {
    const char* name;
    void (*fn)();
  };
  const Case cases[] = {{"mul1 pins", test_mul1_pins},   {"real tile", test_real_tile},
                        {"retile", test_retile},         {"hadamard", test_hadamard},
                        {"oracle vs reconstruct", test_oracle_vs_reconstruct}, {"geometry", test_geometry}};
  int failed = 0;
  for (const Case& c : cases) {
    try {
      c.fn();
      std::printf("[ OK ] %s\n", c.name);
    } catch (const std::exception& e) {
      std::printf("[FAIL] %s: %s\n", c.name, e.what());
      ++failed;
    }
  }
  return failed ? 1 : 0;
}
