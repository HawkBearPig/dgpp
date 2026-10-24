#pragma once
// The EXL3 trellis weight format (2026-10-22, docs/deepseek_v41_flash_exl3_plan.md
// §2): exllamav3's `mul1` codebook trellis as the LibertAIDAI
// DeepSeek-V4.1-Flash-1017 3.5 bpw pack stores it, the resident view the
// kernels read, and the host codec + oracle every kernel test is pinned
// against. CUDA-free: the host-only model libraries and the unit gate
// include it.
//
// The checkpoint's matrix X with input width K and output width N (the HF
// weight is [N, K]) is four tensors:
//   X.trellis  I16 [K/16, N/16, 16 * bits]  one 16x16 tile per (kt, nt)
//   X.suh      F16 [K]                      the input-side per-channel factor
//   X.svh      F16 [N]                      the output-side per-channel factor
//   X.mul1     I32 []                       the codebook marker kExl3Mul1Marker
// A tile's 256 weights sit in the mma fragment order (exl3_tile_perm):
// position 8 t + j of lane t is tile element (r, c) with r along K and c
// along N. The tile's 16 * bits int16 words, read as little-endian uint32
// words, form a 256 * bits-bit stream MSB-first; position p's 16-bit
// state is the stream's bits [(p + 1) bits - 16, (p + 1) bits) modulo the
// stream length (a tail-biting ring), the low `bits` of which are p's own
// (exllamav3 pack.cu / exl3_dq.cuh). A state's value (exl3_mul1_value):
//   x = state * 0x83DCD12D (mod 2^32); h = 1024 + bytesum(x)  — an fp16 bit
//   pattern in [0x6400, 0x67FF]; value = hfma(h, 0x1eee, 0xc931) in fp16.
// With w_inner[k, n] the decoded values and H the 128-point Sylvester
// Hadamard over sqrt(128) applied per 128-block (exl3_hadamard128_f32's
// stage order), the weight and the forward are
//   W   = D_suh . H_K . w_inner . H_N . D_svh            (HF weight = W^T)
//   xh  = (x o suh) H_K ;  y = (xh . w_inner) H_N o svh.
// The engine's chain (the plan's §2.2): x bf16 widened, times suh in fp32,
// the rotation in fp32, times kExl3HadamardScale, rounded to fp16 once; the
// mma on fp16 A and the fp16 codebook values with fp32 accumulation in
// ascending k; the output rotation in fp32, times the scale and svh in
// fp32; the site's rounding last.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"

namespace dgpp {

constexpr int kExl3Tile = 16;                        // a tile's side, along k and along n
constexpr int kExl3Block = 128;                      // the Hadamard block (every K and N a multiple)
constexpr uint32_t kExl3Mul1Multiplier = 0x83DCD12Du;
constexpr int32_t kExl3Mul1Marker = -2082680531;     // the I32 marker tensor's value (= the multiplier)
constexpr uint16_t kExl3Mul1KInv = 0x1eee;           // fp16 1/147.7
constexpr uint16_t kExl3Mul1KBias = 0xc931;          // fp16 -10.39
constexpr float kExl3HadamardScale = 0.088388347648f;  // the reference's fp32 1/sqrt(128)

inline bool exl3_bits_ok(int bits) { return bits >= 1 && bits <= 8; }
// Bytes of one 16x16 tile: 256 states x bits / 8.
inline size_t exl3_tile_bytes(int bits) { return static_cast<size_t>(32) * static_cast<size_t>(bits); }

// Resident view of one EXL3 matrix: the checkpoint's tile bytes re-ordered
// n-tile-major (an output tile's k tiles contiguous — what a warp streams),
// each tile's bytes untouched; suh / svh as stored (f16). A TP slice holds
// whole 128-blocks of its sliced axis (the two rotations act per block):
// an output slice takes n-tiles and the svh range, an input slice k-tiles
// and the suh range.
struct Exl3Matrix {
  const uint8_t* tiles = nullptr;  // [rows/16][cols/16][32 * bits]
  const uint16_t* suh = nullptr;   // f16 [cols]
  const uint16_t* svh = nullptr;   // f16 [rows]
  int64_t rows = 0;                // N (this slice's)
  int64_t cols = 0;                // K (this slice's)
  int bits = 0;

  int64_t n_tiles() const { return rows / kExl3Tile; }
  int64_t k_tiles() const { return cols / kExl3Tile; }
  size_t tile_bytes() const { return exl3_tile_bytes(bits); }
  size_t n_tile_bytes() const { return static_cast<size_t>(k_tiles()) * tile_bytes(); }  // one output tile's stream
  size_t tiles_bytes() const { return static_cast<size_t>(n_tiles()) * n_tile_bytes(); }
  size_t suh_bytes() const { return static_cast<size_t>(cols) * 2; }
  size_t svh_bytes() const { return static_cast<size_t>(rows) * 2; }
};

// The geometry every view must satisfy: K and N positive multiples of the
// Hadamard block, an integer bit width 1..8 (the half-integer trellis
// widths exllamav3 also writes are refused: this pack has none).
inline void exl3_check_geometry(int64_t rows, int64_t cols, int bits, const char* who) {
  if (!exl3_bits_ok(bits))
    throw std::invalid_argument(std::string(who) + ": EXL3 bit width " + std::to_string(bits) +
                                " outside 1..8 (the trellis width must be 16 * bits int16 words)");
  if (rows <= 0 || cols <= 0 || rows % kExl3Block != 0 || cols % kExl3Block != 0)
    throw std::invalid_argument(std::string(who) + ": EXL3 K and N must be positive multiples of " +
                                std::to_string(kExl3Block) + " (the Hadamard block), got [" +
                                std::to_string(rows) + ", " + std::to_string(cols) + "]");
}

// A whole-block output range of a view (the layer's uses: a wo_a group of
// a two-group slice, a head range): rows aligned to the Hadamard block.
inline Exl3Matrix exl3_rows_view(const Exl3Matrix& m, int64_t row_start, int64_t rows) {
  if (row_start < 0 || rows <= 0 || rows > m.rows || row_start > m.rows - rows)
    throw std::invalid_argument("exl3_rows_view: range out of bounds");
  if (row_start % kExl3Block != 0 || rows % kExl3Block != 0)
    throw std::invalid_argument("exl3_rows_view: the range must be whole 128-row blocks (the output rotation)");
  Exl3Matrix v = m;
  v.tiles = m.tiles + static_cast<size_t>(row_start / kExl3Tile) * m.n_tile_bytes();
  v.svh = m.svh + row_start;
  v.rows = rows;
  return v;
}

// ---- the host codec (the kernels' arithmetic exactly) ----------------------

// The value of one 16-bit trellis state under the mul1 codebook, as the
// kernel computes it (imul, dp4a byte sum into the fp16 pattern 0x6400 +
// sum, one fp16 fma): the fp16 fma is one rounding of the exact result,
// which the double arithmetic below reproduces (h, 1/147.7 and -10.39 are
// exact in double and the product has 22 significant bits).
inline float exl3_mul1_value(uint32_t state) {
  const uint32_t x = (state & 0xFFFFu) * kExl3Mul1Multiplier;
  const uint32_t sum = 0x6400u + (x & 0xFFu) + ((x >> 8) & 0xFFu) + ((x >> 16) & 0xFFu) + ((x >> 24) & 0xFFu);
  const double h = fp16_bits_to_float(static_cast<uint16_t>(sum));
  const double v = h * static_cast<double>(fp16_bits_to_float(kExl3Mul1KInv)) +
                   static_cast<double>(fp16_bits_to_float(kExl3Mul1KBias));
  return fp16_bits_to_float(float_to_fp16_bits(static_cast<float>(v)));
}

// The 1,024-entry value table by the state's high bits is not what the
// codebook is (every 16-bit state has its own value); this is the full
// 65,536-entry table for the oracles.
inline const std::vector<float>& exl3_mul1_table() {
  static const std::vector<float> t = [] {
    std::vector<float> v(65536);
    for (uint32_t s = 0; s < 65536; ++s) v[s] = exl3_mul1_value(s);
    return v;
  }();
  return t;
}

// Position p (0..255) of a tile -> (r, c): the mma B-fragment order. Lane
// t = p / 8 holds, for j = p % 8: rows (t % 4) * 2 + {0, 1, 8, 9}[j % 4]
// of column t / 4 + 8 * (j / 4).
inline void exl3_tile_position(int p, int& r, int& c) {
  const int t = p / 8, j = p % 8;
  static const int kRow[4] = {0, 1, 8, 9};
  r = (t % 4) * 2 + kRow[j % 4];
  c = t / 4 + 8 * (j / 4);
}

// The stream word w of a tile (its int16 words as little-endian uint32).
inline uint32_t exl3_tile_word(const uint8_t* tile, int w) {
  uint32_t v;
  std::memcpy(&v, tile + static_cast<size_t>(w) * 4, 4);
  return v;
}
// Position p's 16-bit state: the stream bits [(p + 1) bits - 16, (p + 1)
// bits) modulo 256 * bits, MSB-first inside each word.
inline uint32_t exl3_tile_state(const uint8_t* tile, int bits, int p) {
  const int length = 256 * bits;
  const int end = ((p + 1) * bits) % length;  // exclusive end bit; 0 means the very end
  uint32_t s = 0;
  for (int i = 0; i < 16; ++i) {
    const int b = (((end - 16 + i) % length) + length) % length;
    const uint32_t word = exl3_tile_word(tile, b / 32);
    s = (s << 1) | ((word >> (31 - (b % 32))) & 1u);
  }
  return s;
}
// One tile decoded to its values in (k, n) order: out[r * 16 + c].
inline void exl3_decode_tile(const uint8_t* tile, int bits, float* out) {
  const std::vector<float>& lut = exl3_mul1_table();
  for (int p = 0; p < 256; ++p) {
    int r, c;
    exl3_tile_position(p, r, c);
    out[r * kExl3Tile + c] = lut[exl3_tile_state(tile, bits, p)];
  }
}

// The checkpoint's [K/16, N/16, 16 * bits] trellis (host bytes) re-ordered
// n-tile-major over the tile ranges [nt0, nt1) x [kt0, kt1): dst holds
// (nt1 - nt0) * (kt1 - kt0) * 32 * bits bytes, output tile i's k tiles
// contiguous. A pure permutation of whole tiles.
inline void exl3_retile(const uint8_t* src, int64_t k_tiles_full, int64_t n_tiles_full, int bits, int64_t kt0,
                        int64_t kt1, int64_t nt0, int64_t nt1, uint8_t* dst) {
  if (kt0 < 0 || kt1 > k_tiles_full || kt0 >= kt1 || nt0 < 0 || nt1 > n_tiles_full || nt0 >= nt1)
    throw std::invalid_argument("exl3_retile: tile range out of bounds");
  const size_t tb = exl3_tile_bytes(bits);
  for (int64_t nt = nt0; nt < nt1; ++nt)
    for (int64_t kt = kt0; kt < kt1; ++kt)
      std::memcpy(dst + (static_cast<size_t>(nt - nt0) * static_cast<size_t>(kt1 - kt0) + static_cast<size_t>(kt - kt0)) * tb,
                  src + (static_cast<size_t>(kt) * static_cast<size_t>(n_tiles_full) + static_cast<size_t>(nt)) * tb, tb);
}

// The 128-point rotation of the reference, in fp32, in place: seven
// butterfly stages at strides 1, 2, 4, 8, 16, 32, 64 — element j takes
// v_j + v_{j^s} when bit s of j is clear and v_{j^s} - v_j when it is set
// (exllamav3 hadamard_inner.cuh: the in-lane four-element stage, then the
// shuffle stages) — then one multiply by the fp32 scale. The kernels run
// these operations in this order.
inline void exl3_hadamard128_f32(float* v, float scale) {
  float next[kExl3Block];
  for (int s = 1; s < kExl3Block; s *= 2) {
    for (int j = 0; j < kExl3Block; ++j) {
      const float other = v[j ^ s];
      next[j] = (j & s) ? other - v[j] : v[j] + other;
    }
    for (int j = 0; j < kExl3Block; ++j) v[j] = next[j];
  }
  for (int j = 0; j < kExl3Block; ++j) v[j] *= scale;
}

// The engine's A operand for one activation row (bf16 [k]): per block,
// x * suh in fp32, the rotation, the scale, rounded to fp16 (bits out).
inline void exl3_rotate_row_in(const uint16_t* x_bf16, const uint16_t* suh_f16, int64_t k, uint16_t* xh_f16) {
  float v[kExl3Block];
  for (int64_t b = 0; b < k; b += kExl3Block) {
    for (int j = 0; j < kExl3Block; ++j)
      v[j] = bf16_bits_to_float(x_bf16[b + j]) * fp16_bits_to_float(suh_f16[b + j]);
    exl3_hadamard128_f32(v, kExl3HadamardScale);
    for (int j = 0; j < kExl3Block; ++j) xh_f16[b + j] = float_to_fp16_bits(v[j]);
  }
}

// A host copy of a matrix in the resident layout (the tests' and the
// fixture's), its view, and the full reconstruct for the model-level
// oracles.
struct Exl3HostMatrix {
  std::vector<uint8_t> tiles;   // n-tile-major, as Exl3Matrix
  std::vector<uint16_t> suh;    // f16 [cols]
  std::vector<uint16_t> svh;    // f16 [rows]
  int64_t rows = 0;
  int64_t cols = 0;
  int bits = 0;
  Exl3Matrix view() const {
    Exl3Matrix m;
    m.tiles = tiles.data();
    m.suh = suh.data();
    m.svh = svh.data();
    m.rows = rows;
    m.cols = cols;
    m.bits = bits;
    return m;
  }
  const uint8_t* tile(int64_t nt, int64_t kt) const {
    return tiles.data() + (static_cast<size_t>(nt) * static_cast<size_t>(cols / kExl3Tile) + static_cast<size_t>(kt)) *
                              exl3_tile_bytes(bits);
  }
};

// The TP split of an axis of `total` elements (a multiple of the block)
// into whole 128-blocks: rank r of `world` gets blocks [begin, begin +
// count) with the remainder blocks on the lowest ranks (2304 at world 4:
// 640 / 640 / 512 / 512; the plan's §2.3 — the experts alternate this with
// its mirror, exl3_expert_block_split below). Every rank gets at least
// one block when world <= total / 128.
inline void exl3_block_split(int64_t total, int world, int rank, int64_t& begin, int64_t& count) {
  if (total <= 0 || total % kExl3Block != 0 || world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("exl3_block_split: bad geometry");
  const int64_t blocks = total / kExl3Block;
  if (blocks < world) throw std::invalid_argument("exl3_block_split: fewer 128-blocks than ranks");
  const int64_t base = blocks / world, rem = blocks % world;
  const int64_t b0 = static_cast<int64_t>(rank) * base + std::min<int64_t>(rank, rem);
  const int64_t bn = base + (rank < rem ? 1 : 0);
  begin = b0 * kExl3Block;
  count = bn * kExl3Block;
}
// The widest slice exl3_block_split hands any rank (the buffers' width).
inline int64_t exl3_block_split_max(int64_t total, int world) {
  if (total <= 0 || total % kExl3Block != 0 || world < 1) throw std::invalid_argument("exl3_block_split_max: bad geometry");
  const int64_t blocks = total / kExl3Block;
  return ((blocks + world - 1) / world) * kExl3Block;
}
// An expert's slice (2026-10-23): the remainder blocks on the LOWEST ranks
// for an even parity (exl3_block_split), on the HIGHEST for an odd one —
// the mirror image. Routed experts take their index's parity, a layer's
// shared expert the layer's, so a decode row's six experts split their
// remainder blocks between the two halves of the world and the expert
// bytes a step streams per rank even out: 2304 = 18 blocks at world 4 is
// 5/5/4/4 on even experts and 4/4/5/5 on odd ones (one rule for all
// experts put 11 % more expert bytes on ranks 0 and 1 than an even split,
// and the step waits for the slowest rank at every fold). Contiguous,
// disjoint, covering, whole blocks either way.
inline void exl3_expert_block_split(int64_t total, int world, int rank, int parity, int64_t& begin, int64_t& count) {
  if ((parity & 1) == 0) {
    exl3_block_split(total, world, rank, begin, count);
    return;
  }
  int64_t b = 0, c = 0;
  exl3_block_split(total, world, world - 1 - rank, b, c);
  count = c;
  begin = total - (b + c);
}

// Slices of a host matrix in the resident layout (the tests' and the
// loader's TP slices in miniature): whole 128-blocks of the output axis
// (n-tiles + svh) or of the input axis (k-tiles + suh).
inline Exl3HostMatrix exl3_host_rows_slice(const Exl3HostMatrix& m, int64_t row0, int64_t rows) {
  if (row0 < 0 || rows <= 0 || row0 + rows > m.rows || row0 % kExl3Block != 0 || rows % kExl3Block != 0)
    throw std::invalid_argument("exl3_host_rows_slice: the range must be whole 128-blocks inside the matrix");
  Exl3HostMatrix o;
  o.rows = rows;
  o.cols = m.cols;
  o.bits = m.bits;
  const size_t ntb = static_cast<size_t>(m.cols / kExl3Tile) * exl3_tile_bytes(m.bits);
  o.tiles.assign(m.tiles.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(row0 / kExl3Tile) * ntb),
                 m.tiles.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>((row0 + rows) / kExl3Tile) * ntb));
  o.suh = m.suh;
  o.svh.assign(m.svh.begin() + row0, m.svh.begin() + row0 + rows);
  return o;
}
inline Exl3HostMatrix exl3_host_cols_slice(const Exl3HostMatrix& m, int64_t col0, int64_t cols) {
  if (col0 < 0 || cols <= 0 || col0 + cols > m.cols || col0 % kExl3Block != 0 || cols % kExl3Block != 0)
    throw std::invalid_argument("exl3_host_cols_slice: the range must be whole 128-blocks inside the matrix");
  Exl3HostMatrix o;
  o.rows = m.rows;
  o.cols = cols;
  o.bits = m.bits;
  const size_t tb = exl3_tile_bytes(m.bits);
  const int64_t kt0 = col0 / kExl3Tile, ktn = cols / kExl3Tile, ktf = m.cols / kExl3Tile;
  o.tiles.resize(static_cast<size_t>(m.rows / kExl3Tile) * static_cast<size_t>(ktn) * tb);
  for (int64_t nt = 0; nt < m.rows / kExl3Tile; ++nt)
    std::memcpy(o.tiles.data() + static_cast<size_t>(nt) * static_cast<size_t>(ktn) * tb,
                m.tiles.data() + (static_cast<size_t>(nt) * static_cast<size_t>(ktf) + static_cast<size_t>(kt0)) * tb,
                static_cast<size_t>(ktn) * tb);
  o.suh.assign(m.suh.begin() + col0, m.suh.begin() + col0 + cols);
  o.svh = m.svh;
  return o;
}

// w_inner[k, n] (double, [cols x rows]) of a host matrix.
inline std::vector<double> exl3_inner_weights(const Exl3HostMatrix& m) {
  std::vector<double> w(static_cast<size_t>(m.cols) * static_cast<size_t>(m.rows));
  float tile[kExl3Tile * kExl3Tile];
  for (int64_t nt = 0; nt < m.rows / kExl3Tile; ++nt)
    for (int64_t kt = 0; kt < m.cols / kExl3Tile; ++kt) {
      exl3_decode_tile(m.tile(nt, kt), m.bits, tile);
      for (int r = 0; r < kExl3Tile; ++r)
        for (int c = 0; c < kExl3Tile; ++c)
          w[static_cast<size_t>(kt * kExl3Tile + r) * static_cast<size_t>(m.rows) + static_cast<size_t>(nt * kExl3Tile + c)] =
              tile[r * kExl3Tile + c];
    }
  return w;
}

// The full weight W^T in the HF layout ([rows, cols] = [N, K], double): the
// model-level oracles' matrix (the "semantic" reconstruct, every rotation
// in double).
inline std::vector<double> exl3_reconstruct(const Exl3HostMatrix& m) {
  const std::vector<double> inner = exl3_inner_weights(m);
  const int64_t K = m.cols, N = m.rows;
  std::vector<double> w(static_cast<size_t>(K) * static_cast<size_t>(N));
  // H_K on the k axis per 128-block, then suh.
  std::vector<double> col(kExl3Block), tmp(kExl3Block);
  auto had = [&](std::vector<double>& v) {
    for (int s = 1; s < kExl3Block; s *= 2) {
      for (int j = 0; j < kExl3Block; ++j) tmp[j] = (j & s) ? v[j ^ s] - v[j] : v[j] + v[j ^ s];
      v = tmp;
    }
    const double scale = 1.0 / std::sqrt(static_cast<double>(kExl3Block));
    for (int j = 0; j < kExl3Block; ++j) v[j] *= scale;
  };
  for (int64_t n = 0; n < N; ++n)
    for (int64_t kb = 0; kb < K; kb += kExl3Block) {
      for (int j = 0; j < kExl3Block; ++j) col[j] = inner[static_cast<size_t>(kb + j) * N + n];
      had(col);
      for (int j = 0; j < kExl3Block; ++j)
        w[static_cast<size_t>(kb + j) * N + n] = col[j] * fp16_bits_to_float(m.suh[kb + j]);
    }
  // H_N on the n axis per 128-block, then svh.
  std::vector<double> out(static_cast<size_t>(N) * static_cast<size_t>(K));
  for (int64_t k = 0; k < K; ++k)
    for (int64_t nb = 0; nb < N; nb += kExl3Block) {
      for (int j = 0; j < kExl3Block; ++j) col[j] = w[static_cast<size_t>(k) * N + nb + j];
      had(col);
      for (int j = 0; j < kExl3Block; ++j)
        out[static_cast<size_t>(nb + j) * K + k] = col[j] * fp16_bits_to_float(m.svh[nb + j]);
    }
  return out;
}

// The strict oracle of the engine's chain for one bf16 activation row:
// the A operand as the kernel rounds it (exl3_rotate_row_in), the dot
// products in double over the fp16 codebook values (exact products), the
// output rotation and the svh scale in double. A correct kernel differs
// from it by its fp32 accumulation and rotation order only.
inline void exl3_oracle_row(const Exl3HostMatrix& m, const std::vector<double>& inner, const uint16_t* x_bf16,
                            std::vector<double>& y) {
  const int64_t K = m.cols, N = m.rows;
  std::vector<uint16_t> xh(static_cast<size_t>(K));
  exl3_rotate_row_in(x_bf16, m.suh.data(), K, xh.data());
  std::vector<double> xd(static_cast<size_t>(K));
  for (int64_t k = 0; k < K; ++k) xd[k] = fp16_bits_to_float(xh[k]);
  y.assign(static_cast<size_t>(N), 0.0);
  for (int64_t k = 0; k < K; ++k) {
    const double a = xd[k];
    if (a == 0.0) continue;
    const double* wrow = inner.data() + static_cast<size_t>(k) * N;
    for (int64_t n = 0; n < N; ++n) y[n] += a * wrow[n];
  }
  std::vector<double> v(kExl3Block), tmp(kExl3Block);
  const double scale = 1.0 / std::sqrt(static_cast<double>(kExl3Block));
  for (int64_t nb = 0; nb < N; nb += kExl3Block) {
    for (int j = 0; j < kExl3Block; ++j) v[j] = y[nb + j];
    for (int s = 1; s < kExl3Block; s *= 2) {
      for (int j = 0; j < kExl3Block; ++j) tmp[j] = (j & s) ? v[j ^ s] - v[j] : v[j] + v[j ^ s];
      v = tmp;
    }
    for (int j = 0; j < kExl3Block; ++j) y[nb + j] = v[j] * scale * fp16_bits_to_float(m.svh[nb + j]);
  }
}

// A deterministic host matrix for tests and fixtures: random trellis
// bytes (every bit pattern is a valid trellis), suh / svh drawn as given.
// `suh_scale`: suh values in +-[suh_scale/2, suh_scale]; svh in
// +-[svh_scale/2, svh_scale] (the pack's suh carries the weight magnitude,
// ~0.03; its svh sits near +-1).
template <class Rng>
inline Exl3HostMatrix exl3_random_matrix(Rng& rng, int64_t rows, int64_t cols, int bits, float suh_scale,
                                         float svh_scale) {
  exl3_check_geometry(rows, cols, bits, "exl3_random_matrix");
  Exl3HostMatrix m;
  m.rows = rows;
  m.cols = cols;
  m.bits = bits;
  m.tiles.resize(static_cast<size_t>(rows / kExl3Tile) * static_cast<size_t>(cols / kExl3Tile) * exl3_tile_bytes(bits));
  for (auto& b : m.tiles) b = static_cast<uint8_t>(rng.next());
  auto draw = [&](float s) {
    const float u = static_cast<float>(rng.unit());  // [-1, 1)
    const float mag = s * (0.5f + 0.5f * std::fabs(u));
    return float_to_fp16_bits(u < 0 ? -mag : mag);
  };
  m.suh.resize(static_cast<size_t>(cols));
  for (auto& v : m.suh) v = draw(suh_scale);
  m.svh.resize(static_cast<size_t>(rows));
  for (auto& v : m.svh) v = draw(svh_scale);
  return m;
}

}  // namespace dgpp
