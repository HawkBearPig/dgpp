#include "models/dsv41/loader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {
namespace {

bool contains(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

// One e8m0 scale byte as fp32: 2^(byte - 127); 255 is NaN.
float e8m0_to_float(uint8_t b) {
  if (b == 255) return std::nanf("");
  return std::ldexp(1.0f, static_cast<int>(b) - 127);
}

// The replicated set (rank-invariant reads at world > 1): the norms, the
// routers and their biases, the indexers, the compressors, the mHC
// coefficients, the Engram gates, the embedding, the final norm, wq_a and
// wkv (every rank's latent), the draft's head tensors (main_proj, the
// Markov embedding and head, the confidence head). Everything else is a
// slice: wq_b, wo_a, wo_b, the sink, every expert, the Engram wkv, the lm
// head.
bool is_replicated(const Dsv41ExpectedTensor& e) {
  switch (e.cls) {
    case Dsv41WeightClass::Embed:
    case Dsv41WeightClass::FinalNorm:
    case Dsv41WeightClass::LayerNorm:
    case Dsv41WeightClass::Indexer:
    case Dsv41WeightClass::Compressor:
    case Dsv41WeightClass::Router:
    case Dsv41WeightClass::Mhc:
    case Dsv41WeightClass::Draft:
      return true;
    case Dsv41WeightClass::Attention:
      return contains(e.name, ".wq_a.") || contains(e.name, ".wkv.");
    case Dsv41WeightClass::Engram:
      return !contains(e.name, ".wkv.");
    case Dsv41WeightClass::EngramTable:
    case Dsv41WeightClass::Vision:
    case Dsv41WeightClass::LmHead:
    case Dsv41WeightClass::SharedExpert:
    case Dsv41WeightClass::RoutedExpert:
      return false;
  }
  return false;
}

}  // namespace

// The per-class builders (loaders/weight_build.hpp's primitives).
struct Dsv41LoaderFamily::Builder : WeightBuilder<Dsv41ExpectedTensor> {
  const Dsv41TextConfig& cfg;
  const Dsv41LocalGeometry& geo;
  Dsv41LayerResident& out;

  Builder(const Dsv41TextConfig& cfg_, const Dsv41LocalGeometry& geo_,
          const std::vector<Dsv41ExpectedTensor>& table_,
          const std::unordered_map<std::string, const Dsv41ExpectedTensor*>& by_name_, LayerBump& bump_,
          Dsv41LayerResident& out_, const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<Dsv41ExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_,
                                           geo_.rank, geo_.world, "dsv41 loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const Dsv41ExpectedTensor& e) const override { return is_replicated(e); }

  // ---- fp8 pairs on the 32 x 32 grid (e8m0 -> fp32 scales) -----------------
  struct Fp8Source {
    const Dsv41ExpectedTensor* w;
    const Dsv41ExpectedTensor* s;
    int64_t N, K, SR, SC;
  };
  Fp8Source fp8_source(const std::string& base) {
    Fp8Source f;
    f.w = &expected(base + ".weight");
    f.s = &expected(base + ".scale");
    if (f.w->role != Dsv41TensorRole::Fp8Payload || f.s->role != Dsv41TensorRole::Fp8Scale ||
        f.w->shape.size() != 2 || f.s->shape.size() != 2)
      fail("'" + base + "' is not an fp8 pair");
    f.N = f.w->shape[0];
    f.K = f.w->shape[1];
    f.SR = f.s->shape[0];
    f.SC = f.s->shape[1];
    const int b = kDsv41Fp8Block;
    if (f.SR != (f.N + b - 1) / b || f.SC != (f.K + b - 1) / b)
      fail("fp8 scale geometry mismatch on " + base);
    return f;
  }
  GlmQuantMatrix alloc_fp8(int64_t rows, int64_t cols) {
    const int b = kDsv41Fp8Block;
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_block_rows = b;
    q.scale_block_cols = b;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols)));
    q.scales = static_cast<const float*>(
        bump.alloc(static_cast<size_t>((rows + b - 1) / b) * static_cast<size_t>((cols + b - 1) / b) * 4));
    return q;
  }
  // Scale entries [col0, col0 + count) of source scale row `src_row` into `dst`.
  void convert_scales(const TensorInfo& ts, const Fp8Source& f, int64_t src_row, int64_t col0, int64_t count,
                      float* dst) const {
    const uint8_t* s = static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(src_row) * f.SC + col0;
    for (int64_t i = 0; i < count; ++i) dst[i] = e8m0_to_float(s[i]);
  }
  // Rows [row_start, +rows): row_start on a block boundary (every row's
  // scale row is then its own block row).
  GlmQuantMatrix load_fp8_rows(const std::string& base, int64_t row_start, int64_t rows) {
    const Fp8Source f = fp8_source(base);
    const int b = kDsv41Fp8Block;
    if (row_start % b != 0) fail("fp8 row slice of '" + base + "' must start on a 32-row block");
    check_range(base, row_start, rows, f.N);
    GlmQuantMatrix q = alloc_fp8(rows, f.K);
    const int64_t sr = (rows + b - 1) / b;
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(tw.data) + static_cast<size_t>(row_start) * f.K,
                  static_cast<size_t>(rows) * f.K);
      float* hs = bump.host(const_cast<float*>(q.scales));
      for (int64_t r = 0; r < sr; ++r) convert_scales(ts, f, row_start / b + r, 0, f.SC, hs + r * f.SC);
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(rows) * f.K);
    note_read(*f.s, static_cast<size_t>(sr) * f.SC);
    return q;
  }
  GlmQuantMatrix load_fp8(const std::string& base) {
    const Fp8Source f = fp8_source(base);
    return load_fp8_rows(base, 0, f.N);
  }
  // Column ranges [(start, count)] of every row, packed in order: every
  // range on 32-column blocks (the scale columns follow exactly).
  GlmQuantMatrix load_fp8_col_ranges(const std::string& base,
                                     const std::vector<std::pair<int64_t, int64_t>>& ranges) {
    const Fp8Source f = fp8_source(base);
    const int b = kDsv41Fp8Block;
    int64_t cols = 0;
    for (const auto& [start, count] : ranges) {
      if (start % b != 0 || count % b != 0 || count <= 0)
        fail("fp8 column slice of '" + base + "' must be whole 32-column blocks");
      check_range(base, start, count, f.K);
      cols += count;
    }
    GlmQuantMatrix q = alloc_fp8(f.N, cols);
    const int64_t sc = cols / b;
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      const uint8_t* sp = static_cast<const uint8_t*>(tw.data);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      float* hs = bump.host(const_cast<float*>(q.scales));
      if (ranges.size() == 1 && ranges[0].first == 0 && cols == f.K) {
        std::memcpy(hp, sp, static_cast<size_t>(f.N) * f.K);
      } else {
        for (int64_t r = 0; r < f.N; ++r) {
          int64_t at = 0;
          for (const auto& [start, count] : ranges) {
            std::memcpy(hp + r * cols + at, sp + r * f.K + start, static_cast<size_t>(count));
            at += count;
          }
        }
      }
      for (int64_t r = 0; r < f.SR; ++r) {
        int64_t at = 0;
        for (const auto& [start, count] : ranges) {
          convert_scales(ts, f, r, start / b, count / b, hs + r * sc + at);
          at += count / b;
        }
      }
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(f.N) * static_cast<size_t>(cols));
    note_read(*f.s, static_cast<size_t>(f.SR) * static_cast<size_t>(sc));
    return q;
  }
  GlmQuantMatrix load_fp8_cols(const std::string& base, int64_t col_start, int64_t cols) {
    return load_fp8_col_ranges(base, {{col_start, cols}});
  }

  // ---- MXFP4 pairs (e2m1 pairs + e8m0 per 32) ------------------------------
  struct Fp4Source {
    const Dsv41ExpectedTensor* w;
    const Dsv41ExpectedTensor* s;
    int64_t N, K;
  };
  Fp4Source fp4_source(const std::string& base) {
    Fp4Source f;
    f.w = &expected(base + ".weight");
    f.s = &expected(base + ".scale");
    if (f.w->role != Dsv41TensorRole::Fp4Payload || f.s->role != Dsv41TensorRole::Fp4Scale ||
        f.w->shape.size() != 2 || f.s->shape.size() != 2)
      fail("'" + base + "' is not an MXFP4 pair");
    f.N = f.w->shape[0];
    f.K = f.w->shape[1] * 2;
    fp4_check_cols(f.K, who.c_str(), kMxfp4Group);
    if (f.s->shape[0] != f.N || f.s->shape[1] != f.K / kMxfp4Group)
      fail("MXFP4 scale geometry mismatch on " + base);
    return f;
  }
  GlmFp4Matrix alloc_mxfp4(int64_t rows, int64_t cols) {
    GlmFp4Matrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_group = kMxfp4Group;
    q.global_scale = nullptr;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols / 2)));
    q.scales = static_cast<const uint8_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols / kMxfp4Group)));
    return q;
  }
  GlmFp4Matrix load_mxfp4_rows(const std::string& base, int64_t row_start, int64_t rows) {
    const Fp4Source f = fp4_source(base);
    check_range(base, row_start, rows, f.N);
    const size_t pc = static_cast<size_t>(f.K / 2), sc = static_cast<size_t>(f.K / kMxfp4Group);
    GlmFp4Matrix q = alloc_mxfp4(rows, f.K);
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(tw.data) + static_cast<size_t>(row_start) * pc,
                  static_cast<size_t>(rows) * pc);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.scales)),
                  static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(row_start) * sc,
                  static_cast<size_t>(rows) * sc);
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(rows) * pc);
    note_read(*f.s, static_cast<size_t>(rows) * sc);
    return q;
  }
  GlmFp4Matrix load_mxfp4_cols(const std::string& base, int64_t col_start, int64_t cols) {
    const Fp4Source f = fp4_source(base);
    fp4_check_cols(cols, who.c_str(), kMxfp4Group);
    if (col_start % kMxfp4Group != 0)
      fail("MXFP4 column slice of '" + base + "' must start on a 32-element block boundary");
    check_range(base, col_start, cols, f.K);
    const size_t pc = static_cast<size_t>(cols / 2), pc_full = static_cast<size_t>(f.K / 2);
    const size_t sc = static_cast<size_t>(cols / kMxfp4Group), sc_full = static_cast<size_t>(f.K / kMxfp4Group);
    GlmFp4Matrix q = alloc_mxfp4(f.N, cols);
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      const uint8_t* sp = static_cast<const uint8_t*>(tw.data);
      const uint8_t* ss = static_cast<const uint8_t*>(ts.data);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      uint8_t* hs = bump.host(const_cast<uint8_t*>(q.scales));
      if (cols == f.K) {
        std::memcpy(hp, sp, static_cast<size_t>(f.N) * pc);
        std::memcpy(hs, ss, static_cast<size_t>(f.N) * sc);
      } else {
        for (int64_t r = 0; r < f.N; ++r) {
          std::memcpy(hp + r * pc, sp + r * pc_full + col_start / 2, pc);
          std::memcpy(hs + r * sc, ss + r * sc_full + col_start / kMxfp4Group, sc);
        }
      }
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(f.N) * pc);
    note_read(*f.s, static_cast<size_t>(f.N) * sc);
    return q;
  }

  // An F32 tensor rounded to bf16 at load (the mHC coefficient matrices,
  // plan D10: RNE, the kernel's bf16 form).
  uint16_t* load_f32_as_bf16(const std::string& name) {
    const Dsv41ExpectedTensor& e = expected(name);
    if (e.dtype != DType::F32) fail("'" + name + "' is not F32");
    const size_t n = e.numel();
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(n * 2));
    if (copy) {
      const TensorInfo& t = source(name);
      const float* src = static_cast<const float*>(t.data);
      uint16_t* h = bump.host(dst);
      for (size_t i = 0; i < n; ++i) {
        float v;
        std::memcpy(&v, src + i, 4);
        h[i] = float_to_bf16_bits(v);
      }
      consumed(t);
    }
    note_read(e, n * 4);
    return dst;
  }

  // ---- EXL3 quadruples (docs/deepseek_v41_flash_exl3_plan.md §3.1, §4) ------
  struct Exl3Source {
    const Dsv41ExpectedTensor* trellis;
    const Dsv41ExpectedTensor* suh;
    const Dsv41ExpectedTensor* svh;
    const Dsv41ExpectedTensor* marker;
    int64_t K, N;
    int bits;
  };
  Exl3Source exl3_source(const std::string& base) {
    Exl3Source f;
    f.trellis = &expected(base + ".trellis");
    f.suh = &expected(base + ".suh");
    f.svh = &expected(base + ".svh");
    f.marker = &expected(base + ".mul1");
    if (f.trellis->role != Dsv41TensorRole::Exl3Trellis || f.suh->role != Dsv41TensorRole::Exl3Suh ||
        f.svh->role != Dsv41TensorRole::Exl3Svh || f.marker->role != Dsv41TensorRole::Exl3Marker ||
        f.trellis->shape.size() != 3)
      fail("'" + base + "' is not an EXL3 quadruple");
    f.K = f.trellis->shape[0] * kExl3Tile;
    f.N = f.trellis->shape[1] * kExl3Tile;
    f.bits = static_cast<int>(f.trellis->shape[2] / kExl3Tile);
    exl3_check_geometry(f.N, f.K, f.bits, who.c_str());
    if (f.suh->shape != std::vector<int64_t>{f.K} || f.svh->shape != std::vector<int64_t>{f.N})
      fail("EXL3 suh / svh geometry mismatch on " + base);
    return f;
  }
  Exl3Matrix alloc_exl3(int64_t rows, int64_t cols, int bits) {
    Exl3Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.bits = bits;
    m.tiles = static_cast<const uint8_t*>(bump.alloc(m.tiles_bytes()));
    m.suh = static_cast<const uint16_t*>(bump.alloc(m.suh_bytes()));
    m.svh = static_cast<const uint16_t*>(bump.alloc(m.svh_bytes()));
    return m;
  }
  // The checkpoint's marker tensor must be the mul1 codebook's.
  void check_marker(const Exl3Source& f, const std::string& base) {
    const TensorInfo& t = source(f.marker->name);
    int32_t v = 0;
    std::memcpy(&v, t.data, 4);
    if (v != kExl3Mul1Marker)
      fail("'" + base + "' is not a mul1-codebook EXL3 matrix (marker " + std::to_string(v) + ")");
    consumed(t);
  }
  // Output rows [row0, +rows) (whole 128-blocks: n-tiles + the svh range)
  // and input column ranges [(start, count)] (whole 128-blocks: k-tiles +
  // the suh ranges), re-tiled n-tile-major into the grant.
  Exl3Matrix load_exl3_slice(const std::string& base, int64_t row0, int64_t rows,
                             const std::vector<std::pair<int64_t, int64_t>>& ranges) {
    const Exl3Source f = exl3_source(base);
    check_range(base, row0, rows, f.N);
    if (row0 % kExl3Block != 0 || rows % kExl3Block != 0)
      fail("EXL3 row slice of '" + base + "' must be whole 128-blocks");
    int64_t cols = 0;
    for (const auto& [start, count] : ranges) {
      if (start % kExl3Block != 0 || count % kExl3Block != 0 || count <= 0)
        fail("EXL3 column slice of '" + base + "' must be whole 128-blocks");
      check_range(base, start, count, f.K);
      cols += count;
    }
    Exl3Matrix m = alloc_exl3(rows, cols, f.bits);
    const size_t tb = exl3_tile_bytes(f.bits);
    const int64_t kt_full = f.K / kExl3Tile, nt_full = f.N / kExl3Tile, ktn = cols / kExl3Tile;
    if (copy) {
      const TensorInfo& tt = source(f.trellis->name);
      const TensorInfo& ts = source(f.suh->name);
      const TensorInfo& tv = source(f.svh->name);
      check_marker(f, base);
      uint8_t* ht = bump.host(const_cast<uint8_t*>(m.tiles));
      const uint8_t* src = static_cast<const uint8_t*>(tt.data);
      for (int64_t nt = row0 / kExl3Tile; nt < (row0 + rows) / kExl3Tile; ++nt) {
        uint8_t* dst_row = ht + static_cast<size_t>(nt - row0 / kExl3Tile) * static_cast<size_t>(ktn) * tb;
        int64_t at = 0;
        for (const auto& [start, count] : ranges) {
          for (int64_t kt = start / kExl3Tile; kt < (start + count) / kExl3Tile; ++kt, ++at)
            std::memcpy(dst_row + static_cast<size_t>(at) * tb,
                        src + (static_cast<size_t>(kt) * static_cast<size_t>(nt_full) + static_cast<size_t>(nt)) * tb, tb);
        }
      }
      uint16_t* hs = bump.host(const_cast<uint16_t*>(m.suh));
      int64_t at = 0;
      for (const auto& [start, count] : ranges) {
        std::memcpy(hs + at, static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(start) * 2,
                    static_cast<size_t>(count) * 2);
        at += count;
      }
      std::memcpy(bump.host(const_cast<uint16_t*>(m.svh)), static_cast<const uint8_t*>(tv.data) + static_cast<size_t>(row0) * 2,
                  static_cast<size_t>(rows) * 2);
      consumed(tt);
      consumed(ts);
      consumed(tv);
    }
    (void)kt_full;
    note_read(*f.trellis, static_cast<size_t>(rows / kExl3Tile) * static_cast<size_t>(ktn) * tb);
    note_read(*f.suh, static_cast<size_t>(cols) * 2);
    note_read(*f.svh, static_cast<size_t>(rows) * 2);
    note_read(*f.marker, 4);
    return m;
  }
  Exl3Matrix load_exl3(const std::string& base) {
    const Exl3Source f = exl3_source(base);
    return load_exl3_slice(base, 0, f.N, {{0, f.K}});
  }
  Exl3Matrix load_exl3_rows(const std::string& base, int64_t row0, int64_t rows) {
    const Exl3Source f = exl3_source(base);
    return load_exl3_slice(base, row0, rows, {{0, f.K}});
  }
  Exl3Matrix load_exl3_cols(const std::string& base, int64_t col0, int64_t cols) {
    const Exl3Source f = exl3_source(base);
    return load_exl3_slice(base, 0, f.N, {{col0, cols}});
  }
  Exl3Matrix load_exl3_col_ranges(const std::string& base, const std::vector<std::pair<int64_t, int64_t>>& ranges) {
    const Exl3Source f = exl3_source(base);
    return load_exl3_slice(base, 0, f.N, ranges);
  }

  // ---- the pack's F16 / F32 natives (plan §1.2) ---------------------------------
  // An F16 tensor as bf16 — accepted only when every value is
  // bf16-representable (the pack's routers, weights_proj and Markov head
  // are: bf16 originals widened to f16), refused otherwise: no silent
  // rounding of a weight.
  uint16_t* load_f16_as_bf16_checked(const std::string& name) {
    const Dsv41ExpectedTensor& e = expected(name);
    if (e.dtype != DType::F16) fail("'" + name + "' is not F16");
    const size_t n = e.numel();
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(n * 2));
    if (copy) {
      const TensorInfo& t = source(name);
      const uint16_t* src = static_cast<const uint16_t*>(t.data);
      uint16_t* h = bump.host(dst);
      for (size_t i = 0; i < n; ++i) {
        uint16_t bits;
        std::memcpy(&bits, src + i, 2);
        const float v = fp16_bits_to_float(bits);
        const uint16_t b = float_to_bf16_bits(v);
        if (bf16_bits_to_float(b) != v)
          fail("'" + name + "' holds an f16 value that is not bf16-exact (element " + std::to_string(i) +
               "); the loader does not round weights");
        h[i] = b;
      }
      consumed(t);
    }
    note_read(e, n * 2);
    return dst;
  }
  // An F16 tensor widened to F32 (exact).
  float* load_f16_as_f32(const std::string& name) {
    const Dsv41ExpectedTensor& e = expected(name);
    if (e.dtype != DType::F16) fail("'" + name + "' is not F16");
    const size_t n = e.numel();
    float* dst = static_cast<float*>(bump.alloc(n * 4));
    if (copy) {
      const TensorInfo& t = source(name);
      const uint16_t* src = static_cast<const uint16_t*>(t.data);
      float* h = bump.host(dst);
      for (size_t i = 0; i < n; ++i) {
        uint16_t bits;
        std::memcpy(&bits, src + i, 2);
        h[i] = fp16_bits_to_float(bits);
      }
      consumed(t);
    }
    note_read(e, n * 2);
    return dst;
  }
  // An F32 tensor as bf16 — accepted only when every value is bf16-exact
  // (the pack's Engram q / k weights: the base's bf16 widened).
  uint16_t* load_f32_as_bf16_checked(const std::string& name) {
    const Dsv41ExpectedTensor& e = expected(name);
    if (e.dtype != DType::F32) fail("'" + name + "' is not F32");
    const size_t n = e.numel();
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(n * 2));
    if (copy) {
      const TensorInfo& t = source(name);
      const float* src = static_cast<const float*>(t.data);
      uint16_t* h = bump.host(dst);
      for (size_t i = 0; i < n; ++i) {
        float v;
        std::memcpy(&v, src + i, 4);
        const uint16_t b = float_to_bf16_bits(v);
        if (bf16_bits_to_float(b) != v)
          fail("'" + name + "' holds an f32 value that is not bf16-exact (element " + std::to_string(i) +
               "); the loader does not round weights");
        h[i] = b;
      }
      consumed(t);
    }
    note_read(e, n * 4);
    return dst;
  }

  // ---- the classes ---------------------------------------------------------
  void build_attention(const std::string& p, int layer) {
    Dsv41AttnResident& a = out.attn;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_groups = geo.local_groups;
    a.group_begin = geo.group_begin;
    const int64_t hd = cfg.head_dim, ol = cfg.o_lora_rank;
    a.q_norm = load_bf16(p + "q_norm.weight");
    a.kv_norm = load_bf16(p + "kv_norm.weight");
    a.attn_sink = load_f32_range(p + "attn_sink", geo.head_begin, geo.local_heads);
    if (cfg.weights_exl3) {
      a.wq_a_x = load_exl3(p + "wq_a");
      a.wkv_x = load_exl3(p + "wkv");
      a.wq_b_x = load_exl3_rows(p + "wq_b", static_cast<int64_t>(geo.head_begin) * hd,
                                static_cast<int64_t>(geo.local_heads) * hd);
      a.wo_a_x.clear();
      for (int g = 0; g < geo.local_groups; ++g)
        a.wo_a_x.push_back(load_exl3(p + "wo_a.slice." + std::to_string(geo.group_begin + g)));
      a.wo_b_x = load_exl3_cols(p + "wo_b", static_cast<int64_t>(geo.group_begin) * ol,
                                static_cast<int64_t>(geo.local_groups) * ol);
      if (cfg.is_index_source(layer)) {
        const std::string ip = p + "indexer.";
        a.idx_wq_b_x = load_exl3(ip + "wq_b");
        a.idx_wp = load_f16_as_bf16_checked(ip + "weights_proj.weight");
        if (cfg.is_kv_source(layer)) {
          a.idx_wk_x = load_exl3(ip + "wk");
          a.idx_k_norm = load_bf16(ip + "k_norm.weight");
        }
      }
      if (cfg.is_kv_source(layer)) {
        const std::string cp = p + "compressor.";
        a.comp_wkv_x = load_exl3(cp + "wkv");
        if (cfg.compress_ratio(layer) > 1) a.comp_wgate_x = load_exl3(cp + "wgate");
        a.comp_norm = load_bf16(cp + "norm.weight");
      }
      return;
    }
    a.wq_a = load_fp8(p + "wq_a");
    a.wkv = load_fp8(p + "wkv");
    a.wq_b = load_fp8_rows(p + "wq_b", static_cast<int64_t>(geo.head_begin) * hd,
                           static_cast<int64_t>(geo.local_heads) * hd);
    a.wo_a = load_fp8_rows(p + "wo_a", static_cast<int64_t>(geo.group_begin) * ol,
                           static_cast<int64_t>(geo.local_groups) * ol);
    a.wo_b = load_fp8_cols(p + "wo_b", static_cast<int64_t>(geo.group_begin) * ol,
                           static_cast<int64_t>(geo.local_groups) * ol);
    if (cfg.is_index_source(layer)) {
      const std::string ip = p + "indexer.";
      a.idx_wq_b = load_fp8(ip + "wq_b");
      a.idx_wp = load_bf16(ip + "weights_proj.weight");
      if (cfg.is_kv_source(layer)) {
        a.idx_wk = load_bf16(ip + "wk.weight");
        a.idx_k_norm = load_bf16(ip + "k_norm.weight");
      }
    }
    if (cfg.is_kv_source(layer)) {
      const std::string cp = p + "compressor.";
      a.comp_wkv = load_bf16(cp + "wkv.weight");
      if (cfg.compress_ratio(layer) > 1) a.comp_wgate = load_bf16(cp + "wgate.weight");
      a.comp_norm = load_bf16(cp + "norm.weight");
    }
  }

  void build_moe(const std::string& p, bool draft, int layer) {
    Dsv41MoeResident& m = out.moe;
    // gate.bias_vl (image tokens) is present and never loaded (plan D9).
    const int64_t I = geo.local_inter, S = geo.local_shared_inter;
    const int64_t I0 = geo.inter_begin, S0 = geo.shared_inter_begin;
    m.local_inter = I;
    m.local_shared_inter = S;
    const int E = draft ? cfg.dspark_n_routed_experts : cfg.n_routed_experts;
    m.n_experts = E;
    if (cfg.weights_exl3) {
      // The pack's F16 router (bf16-exact, checked) and F16 bias (widened);
      // every expert's gate / up rows and down columns of this rank's
      // 128-block slice of that expert (exl3_expert_block_split: the
      // remainder blocks alternate between the halves of the world by the
      // expert's parity), the shared expert's by the layer's parity. The
      // resident widths (local_inter) are the widest slice; a matrix's
      // own rows / cols carry its slice.
      m.router = load_f16_as_bf16_checked(p + "gate.weight");
      m.router_bias = load_f16_as_f32(p + "gate.bias");
      m.experts_x.resize(static_cast<size_t>(E) * 3);
      for (int e = 0; e < E; ++e) {
        const std::string ep = p + "experts." + std::to_string(e) + ".";
        Exl3Matrix* t = m.experts_x.data() + static_cast<size_t>(e) * 3;
        int64_t e0 = 0, en = 0;
        exl3_expert_block_split(cfg.moe_intermediate_size, geo.world, geo.rank, e & 1, e0, en);
        t[0] = load_exl3_rows(ep + "w1", e0, en);  // gate
        t[1] = load_exl3_rows(ep + "w3", e0, en);  // up
        t[2] = load_exl3_cols(ep + "w2", e0, en);  // down
      }
      int64_t s0 = 0, sn = 0;
      exl3_expert_block_split(cfg.shared_expert_inter(), geo.world, geo.rank, layer & 1, s0, sn);
      const std::string sp = p + "shared_experts.";
      m.shared_x[0] = load_exl3_rows(sp + "w1", s0, sn);
      m.shared_x[1] = load_exl3_rows(sp + "w3", s0, sn);
      m.shared_x[2] = load_exl3_cols(sp + "w2", s0, sn);
      (void)I0;
      (void)S0;
      return;
    }
    m.router = load_bf16(p + "gate.weight");
    m.router_bias = load_f32(p + "gate.bias");
    m.experts.resize(static_cast<size_t>(E) * 3);
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      GlmFp4Matrix* t = m.experts.data() + static_cast<size_t>(e) * 3;
      t[0] = load_mxfp4_rows(ep + "w1", I0, I);  // gate
      t[1] = load_mxfp4_rows(ep + "w3", I0, I);  // up
      t[2] = load_mxfp4_cols(ep + "w2", I0, I);  // down
    }
    const std::string sp = p + "shared_experts.";
    m.shared[0] = load_fp8_rows(sp + "w1", S0, S);
    m.shared[1] = load_fp8_rows(sp + "w3", S0, S);
    m.shared[2] = load_fp8_cols(sp + "w2", S0, S);
  }

  void build_mhc(const std::string& p) {
    Dsv41MhcResident& h = out.mhc;
    h.attn_fn = load_f32_as_bf16(p + "hc_attn_fn");
    h.attn_base = load_f32(p + "hc_attn_base");
    h.attn_scale = load_f32(p + "hc_attn_scale");
    h.ffn_fn = load_f32_as_bf16(p + "hc_ffn_fn");
    h.ffn_base = load_f32(p + "hc_ffn_base");
    h.ffn_scale = load_f32(p + "hc_ffn_scale");
  }

  void build_engram(const std::string& p, int layer) {
    Dsv41EngramResident& g = out.engram;
    g.present = true;
    g.table_index = cfg.engram_index(layer);
    // wkv's K columns: for every n-gram size, this rank's hash heads' rows
    // of the concatenated embedding (heads_local x head_dim per size).
    std::vector<std::pair<int64_t, int64_t>> ranges;
    const int64_t hd = cfg.engram_head_dim;
    for (int n = 0; n < cfg.engram_max_ngram_size - 1; ++n)
      ranges.emplace_back((static_cast<int64_t>(n) * cfg.engram_n_heads + geo.engram_head_begin) * hd,
                          static_cast<int64_t>(geo.engram_heads_local) * hd);
    if (cfg.weights_exl3) {
      g.wkv_x = load_exl3_col_ranges(p + "wkv", ranges);
      g.q_weight = load_f32_as_bf16_checked(p + "q_weight");
      g.k_weight = load_f32_as_bf16_checked(p + "k_weight");
    } else {
      g.wkv = load_fp8_col_ranges(p + "wkv", ranges);
      g.q_weight = load_bf16(p + "q_weight");
      g.k_weight = load_bf16(p + "k_weight");
    }
    // The table itself (embed.weight / embed.scale) stays in its shard:
    // Dsv41LayerStream::load_engram_tables maps it.
  }

  void build_draft(const std::string& p, int layer) {
    Dsv41DraftResident& d = out.draft;
    const int stage = cfg.draft_stage(layer);
    if (stage == 0) {
      if (cfg.weights_exl3)
        d.main_proj_x = load_exl3(p + "main_proj");
      else
        d.main_proj = load_fp8(p + "main_proj");
      d.main_norm = load_bf16(p + "main_norm.weight");
    }
    if (stage == cfg.num_nextn_predict_layers - 1) {
      d.norm = load_bf16(p + "norm.weight");
      d.markov_embed = load_bf16(p + "markov_head.embed.weight");
      if (cfg.weights_exl3) {
        d.markov_head = load_f16_as_bf16_checked(p + "markov_head.head.weight");
        d.confidence = load_f16_as_f32(p + "confidence_head.proj.weight");
      } else {
        d.markov_head = load_bf16(p + "markov_head.head.weight");
        d.confidence = load_bf16_as_f32(p + "confidence_head.proj.weight");
      }
    }
  }

  void build_layer(int layer) {
    if (layer < 0 || layer >= cfg.max_layer()) fail("layer index out of range: " + std::to_string(layer));
    const bool draft = cfg.is_draft(layer);
    const std::string p = dsv41_layer_prefix(cfg, layer);
    out.layer = layer;
    out.attn_norm = load_bf16(p + "attn_norm.weight");
    out.ffn_norm = load_bf16(p + "ffn_norm.weight");
    build_mhc(p);
    build_attention(p + "attn.", layer);
    build_moe(p + "ffn.", draft, layer);
    if (cfg.has_engram(layer)) build_engram(p + "engram.", layer);
    if (draft) build_draft(p, layer);
  }
};

namespace {

std::pair<int, int> lm_head_slice(const Dsv41TextConfig& cfg, int rank, int world) {
  const int64_t V = cfg.vocab_size;
  if (cfg.weights_exl3) {
    // Whole 128-row blocks (the EXL3 head's output rotation), the
    // remainder blocks on the lowest ranks.
    int64_t b = 0, n = 0;
    exl3_block_split(V, world, rank, b, n);
    return {static_cast<int>(b), static_cast<int>(n)};
  }
  const int64_t begin = V * rank / world;
  const int64_t end = V * (rank + 1) / world;
  return {static_cast<int>(begin), static_cast<int>(end - begin)};
}

std::string& engram_source_dir_storage() {
  static std::string dir;
  return dir;
}

std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}

bool g_embed_vocab_sharded = false;

}  // namespace

// ---------------------------------------------------------------------------

Dsv41LocalGeometry Dsv41LocalGeometry::from_config(const Dsv41TextConfig& cfg, int rank, int world,
                                                   Dsv41HeadSharding head) {
  if (world > 1) dsv41_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("dsv41 loader: rank/world out of range");
  Dsv41LocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  g.local_groups = cfg.o_groups / world;
  g.group_begin = g.local_groups * rank;
  if (cfg.weights_exl3) {
    // The even-parity slice's begin (expert 0's, the shared expert's on
    // even layers) and the widest slice any parity hands this rank — the
    // buffers' width; each expert's own slice is exl3_expert_block_split's.
    exl3_block_split(cfg.moe_intermediate_size, world, rank, g.inter_begin, g.local_inter);
    g.local_inter = exl3_block_split_max(cfg.moe_intermediate_size, world);
    exl3_block_split(cfg.shared_expert_inter(), world, rank, g.shared_inter_begin, g.local_shared_inter);
    g.local_shared_inter = exl3_block_split_max(cfg.shared_expert_inter(), world);
  } else {
    g.local_inter = cfg.moe_intermediate_size / world;
    g.inter_begin = g.local_inter * rank;
    g.local_shared_inter = cfg.shared_expert_inter() / world;
    g.shared_inter_begin = g.local_shared_inter * rank;
  }
  g.engram_heads_local = cfg.engram_layer_ids.empty() ? 0 : cfg.engram_n_heads / world;
  g.engram_head_begin = g.engram_heads_local * rank;
  if (head == Dsv41HeadSharding::VocabSharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.lm_vocab_begin = b;
    g.lm_vocab_count = n;
  } else {
    g.lm_vocab_begin = 0;
    g.lm_vocab_count = cfg.vocab_size;
  }
  if (world > 1 && g_embed_vocab_sharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.embed_vocab_begin = b;
    g.embed_vocab_count = n;
  } else {
    g.embed_vocab_begin = 0;
    g.embed_vocab_count = cfg.vocab_size;
  }
  return g;
}

// ---- the family hooks (loaders/resident_stream.hpp) -------------------------

void Dsv41LoaderFamily::validate_binding(const Dsv41TextConfig& cfg, const PresentMap& present) {
  const Dsv41BindReport rep = dsv41_validate_text_binding(cfg, present);
  if (rep.ok()) return;
  std::string msg = "dsv41 loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

// The digest covers what a rank actually reads verbatim: the replicated
// set minus the Engram tables (mapped, never read whole) and the vision
// tensors (never read).
bool Dsv41LoaderFamily::digest_included(const Dsv41ExpectedTensor& e) {
  if (e.cls == Dsv41WeightClass::EngramTable || e.cls == Dsv41WeightClass::Vision) return false;
  if (contains(e.name, "gate.bias_vl")) return false;
  return is_replicated(e);
}

size_t Dsv41LoaderFamily::globals_bytes(const Dsv41TextConfig& cfg, int rank, int world,
                                        LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const Dsv41LocalGeometry geo = Dsv41LocalGeometry::from_config(cfg, rank, world, head);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(geo.embed_vocab_count) * H * 2);
  b += align_up_256(H * 2);
  if (cfg.weights_exl3) {
    Exl3Matrix m;
    m.rows = geo.lm_vocab_count;
    m.cols = cfg.hidden_size;
    m.bits = cfg.exl3.head;
    b += align_up_256(m.tiles_bytes()) + align_up_256(m.suh_bytes()) + align_up_256(m.svh_bytes());
  } else {
    b += align_up_256(static_cast<size_t>(geo.lm_vocab_count) * H * 2);
  }
  return b;
}

std::string Dsv41LoaderFamily::extra_shard_dir() { return engram_source_dir_storage(); }
bool Dsv41LoaderFamily::admit_extra_tensor(const std::string& name) {
  // The Engram tables alone (layers.L.engram.embed.weight / .scale).
  return name.rfind("layers.", 0) == 0 && name.find(".engram.embed.") != std::string::npos;
}

void Dsv41LoaderFamily::build_globals(const Dsv41TextConfig& cfg, const Dsv41LocalGeometry& geo,
                                      const LoaderTensorMap& tensors, LayerBump& bump,
                                      Dsv41GlobalsResident& out, uint64_t& source_bytes,
                                      uint64_t& verbatim_bytes, LoaderHeadSharding head) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("dsv41 loader: global tensor missing: " + name);
    return *it->second;
  };
  const size_t row_bytes = static_cast<size_t>(cfg.hidden_size) * 2;
  {
    const TensorInfo& e = lookup("embed.weight");
    const int ebegin = geo.embed_vocab_begin, ecount = geo.embed_vocab_count;
    if (ebegin < 0 || ecount <= 0 || static_cast<size_t>(ebegin + ecount) * row_bytes > e.nbytes())
      throw std::runtime_error("dsv41 loader: the embedding slice does not fit the table");
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(ecount) * row_bytes));
    std::memcpy(bump.host(dst), static_cast<const uint8_t*>(e.data) + static_cast<size_t>(ebegin) * row_bytes,
                static_cast<size_t>(ecount) * row_bytes);
    source_bytes += static_cast<size_t>(ecount) * row_bytes;
    if (ecount == cfg.vocab_size) verbatim_bytes += e.nbytes();
    out.embed = dst;
    out.embed_vocab_begin = ebegin;
    out.embed_vocab_count = ecount;
  }
  {
    const TensorInfo& t = lookup("norm.weight");
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(t.nbytes()));
    std::memcpy(bump.host(dst), t.data, t.nbytes());
    source_bytes += t.nbytes();
    verbatim_bytes += t.nbytes();
    out.final_norm = dst;
  }
  const int begin = geo.lm_vocab_begin, count = geo.lm_vocab_count;
  if (cfg.weights_exl3) {
    // The EXL3 head: this rank's vocabulary rows (whole 128-blocks) of
    // the trellis re-tiled n-tile-major, the svh range, the whole suh.
    const TensorInfo& tt = lookup("head.trellis");
    const TensorInfo& ts = lookup("head.suh");
    const TensorInfo& tv = lookup("head.svh");
    const TensorInfo& tm = lookup("head.mul1");
    int32_t marker = 0;
    std::memcpy(&marker, tm.data, 4);
    if (marker != kExl3Mul1Marker) throw std::runtime_error("dsv41 loader: head.mul1 is not the mul1 codebook marker");
    if (tt.shape.size() != 3 || tt.shape[0] * kExl3Tile != cfg.hidden_size || tt.shape[1] * kExl3Tile != cfg.vocab_size ||
        tt.shape[2] != 16LL * cfg.exl3.head)
      throw std::runtime_error("dsv41 loader: head.trellis geometry mismatch");
    Exl3Matrix m;
    m.rows = count;
    m.cols = cfg.hidden_size;
    m.bits = cfg.exl3.head;
    uint8_t* tiles = static_cast<uint8_t*>(bump.alloc(m.tiles_bytes()));
    uint16_t* suh = static_cast<uint16_t*>(bump.alloc(m.suh_bytes()));
    uint16_t* svh = static_cast<uint16_t*>(bump.alloc(m.svh_bytes()));
    exl3_retile(static_cast<const uint8_t*>(tt.data), tt.shape[0], tt.shape[1], m.bits, 0, tt.shape[0],
                begin / kExl3Tile, (begin + count) / kExl3Tile, bump.host(tiles));
    std::memcpy(bump.host(suh), ts.data, m.suh_bytes());
    std::memcpy(bump.host(svh), static_cast<const uint8_t*>(tv.data) + static_cast<size_t>(begin) * 2, m.svh_bytes());
    m.tiles = tiles;
    m.suh = suh;
    m.svh = svh;
    source_bytes += m.tiles_bytes() + m.suh_bytes() + m.svh_bytes();
    if (head == LoaderHeadSharding::Full) verbatim_bytes += m.tiles_bytes() + m.suh_bytes() + m.svh_bytes();
    out.lm_head_x = m;
    out.lm_vocab_begin = begin;
    out.lm_vocab_count = count;
    return;
  }
  const TensorInfo& t = lookup("head.weight");
  uint16_t* dst = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(count) * row_bytes));
  std::memcpy(bump.host(dst), static_cast<const uint8_t*>(t.data) + static_cast<size_t>(begin) * row_bytes,
              static_cast<size_t>(count) * row_bytes);
  source_bytes += static_cast<size_t>(count) * row_bytes;
  if (head == LoaderHeadSharding::Full) verbatim_bytes += static_cast<size_t>(count) * row_bytes;
  out.lm_head = dst;
  out.lm_vocab_begin = begin;
  out.lm_vocab_count = count;
}

template class ResidentLayerStream<Dsv41LoaderFamily>;

// ---- the Engram table mapping --------------------------------------------------

Dsv41EngramTableMmap::Dsv41EngramTableMmap(const std::string& path, uint64_t payload_begin,
                                           uint64_t scale_begin, int64_t rows, int head_dim)
    : payload_begin_(payload_begin), scale_begin_(scale_begin), rows_(rows), head_dim_(head_dim) {
  if (rows <= 0 || head_dim <= 0 || head_dim % 32 != 0)
    throw std::invalid_argument("Dsv41EngramTableMmap: bad geometry");
  fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd_ < 0) throw std::runtime_error("Dsv41EngramTableMmap: cannot open " + path);
  struct stat st {};
  if (fstat(fd_, &st) != 0) {
    ::close(fd_);
    throw std::runtime_error("Dsv41EngramTableMmap: fstat " + path);
  }
  len_ = static_cast<size_t>(st.st_size);
  const uint64_t pend = payload_begin + static_cast<uint64_t>(rows) * head_dim;
  const uint64_t send = scale_begin + static_cast<uint64_t>(rows) * (head_dim / 32);
  if (pend > len_ || send > len_) {
    ::close(fd_);
    throw std::runtime_error("Dsv41EngramTableMmap: the table runs past the end of " + path);
  }
  void* map = mmap(nullptr, len_, PROT_READ, MAP_SHARED, fd_, 0);
  if (map == MAP_FAILED) {
    ::close(fd_);
    throw std::runtime_error("Dsv41EngramTableMmap: mmap " + path);
  }
  base_ = static_cast<uint8_t*>(map);
  // Rows are read one at a time from anywhere in ~95 GB: no readahead.
  madvise(base_, len_, MADV_RANDOM);
}

Dsv41EngramTableMmap::~Dsv41EngramTableMmap() {
  if (base_) munmap(base_, len_);
  if (fd_ >= 0) ::close(fd_);
}

const uint8_t* Dsv41EngramTableMmap::payload_row(int64_t row) const {
  if (row < 0 || row >= rows_)
    throw std::out_of_range("Dsv41EngramTableMmap: row " + std::to_string(row) + " outside the table");
  return base_ + payload_begin_ + static_cast<size_t>(row) * head_dim_;
}

const uint8_t* Dsv41EngramTableMmap::scale_row(int64_t row) const {
  if (row < 0 || row >= rows_)
    throw std::out_of_range("Dsv41EngramTableMmap: row " + std::to_string(row) + " outside the table");
  return base_ + scale_begin_ + static_cast<size_t>(row) * (head_dim_ / 32);
}

void Dsv41EngramTableMmap::gather(const int32_t* ids, int64_t ids_stride, const int32_t* sel,
                                  int rows_local, int n, uint8_t* dst) const {
  if (n <= 0 || rows_local <= 0) return;
  const int64_t total = static_cast<int64_t>(n) * rows_local;
  const size_t rb = static_cast<size_t>(row_bytes());
  const size_t sb = static_cast<size_t>(head_dim_ / 32);
  auto copy_range = [&](int64_t lo, int64_t hi) {
    for (int64_t pair = lo; pair < hi; ++pair) {
      const int64_t t = pair / rows_local, j = pair - t * rows_local;
      const int32_t id = ids[t * ids_stride + sel[j]];
      uint8_t* d = dst + static_cast<size_t>(pair) * rb;
      std::memcpy(d, payload_row(id), static_cast<size_t>(head_dim_));
      std::memcpy(d + head_dim_, scale_row(id), sb);
    }
  };
  // Every row's pages asked for up front (the faults in flight together),
  // then the copies: a decode step's few dozen rows in one thread, a
  // prefill chunk's thousands across a few.
  for (int64_t pair = 0; pair < total; ++pair) {
    const int64_t t = pair / rows_local, j = pair - t * rows_local;
    const int32_t id = ids[t * ids_stride + sel[j]];
    for (const uint8_t* p : {payload_row(id), scale_row(id)}) {
      const uintptr_t page = reinterpret_cast<uintptr_t>(p) & ~uintptr_t{4095};
      madvise(reinterpret_cast<void*>(page), 4096 + static_cast<size_t>(head_dim_), MADV_WILLNEED);
    }
  }
  constexpr int64_t kPerThread = 256;
  if (total <= kPerThread) {
    copy_range(0, total);
    return;
  }
  const int threads = static_cast<int>(std::min<int64_t>(16, (total + kPerThread - 1) / kPerThread));
  const int64_t span = (total + threads - 1) / threads;
  std::vector<std::thread> pool;
  for (int w = 0; w < threads; ++w) {
    const int64_t lo = w * span, hi = std::min(total, lo + span);
    if (lo < hi) pool.emplace_back(copy_range, lo, hi);
  }
  for (auto& th : pool) th.join();
}

// ---------------------------------------------------------------------------

void Dsv41LayerStream::set_embed_vocab_sharded(bool on) { g_embed_vocab_sharded = on; }
bool Dsv41LayerStream::embed_vocab_sharded() { return g_embed_vocab_sharded; }

Dsv41LayerStream::Dsv41LayerStream(const Dsv41TextConfig& cfg, const std::string& checkpoint_dir, int rank,
                                   int world, Dsv41Residency residency, Dsv41HeadSharding head,
                                   bool resident_mtp)
    : ResidentLayerStream<Dsv41LoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                             resident_mtp) {
  open_resident_image();
}

void Dsv41LayerStream::set_resident_image_dir(const std::string& dir) { resident_image_dir_storage() = dir; }
const std::string& Dsv41LayerStream::resident_image_dir() { return resident_image_dir_storage(); }
void Dsv41LayerStream::set_engram_source_dir(const std::string& dir) { engram_source_dir_storage() = dir; }
const std::string& Dsv41LayerStream::engram_source_dir() { return engram_source_dir_storage(); }

const Dsv41EngramTables& Dsv41LayerStream::load_engram_tables() {
  if (engram_loaded_ || cfg_.engram_layer_ids.empty()) return engram_;
  if (sources_released_)
    throw std::runtime_error("dsv41 loader: load_engram_tables after the checkpoint sources were released");
  engram_.head_dim = cfg_.engram_head_dim;
  engram_.heads = cfg_.engram_n_heads;
  engram_.heads_local = geo_.engram_heads_local;
  engram_.head_begin = geo_.engram_head_begin;
  engram_.ngrams = cfg_.engram_max_ngram_size - 1;
  engram_.sel.clear();
  for (int n = 0; n < engram_.ngrams; ++n)
    for (int hl = 0; hl < engram_.heads_local; ++hl)
      engram_.sel.push_back(n * engram_.heads + engram_.head_begin + hl);
  engram_.tables.clear();
  for (size_t i = 0; i < cfg_.engram_layer_ids.size(); ++i) {
    const std::string p = dsv41_layer_prefix(cfg_, cfg_.engram_layer_ids[i]) + "engram.embed.";
    auto it_w = tensors_.find(p + "weight");
    auto it_s = tensors_.find(p + "scale");
    if (it_w == tensors_.end() || !it_w->second || !it_w->second->owner || it_s == tensors_.end() ||
        !it_s->second || !it_s->second->owner)
      throw std::runtime_error("dsv41 loader: Engram table tensors not in the checkpoint: " + p + "weight");
    const TensorInfo& tw = *it_w->second;
    const TensorInfo& ts = *it_s->second;
    if (tw.owner->path() != ts.owner->path())
      throw std::runtime_error("dsv41 loader: an Engram table's payload and scales must share a shard");
    if (tw.shape.size() != 2 || tw.shape[1] != cfg_.engram_head_dim || tw.shape[0] != cfg_.engram_num_embeddings[i])
      throw std::runtime_error("dsv41 loader: Engram table geometry mismatch on " + tw.name);
    engram_.tables.push_back(std::make_unique<Dsv41EngramTableMmap>(tw.owner->path(), tw.data_begin, ts.data_begin,
                                                                    tw.shape[0], cfg_.engram_head_dim));
  }
  engram_loaded_ = true;
  DGPP_LOG_INFO("dsv41 loader: rank {} Engram tables mmap'ed from the checkpoint ({} tables, {:.2f} GiB mapped, "
                "{} rows per token: {} heads x {} n-gram sizes)",
                rank_, engram_.tables.size(), engram_.mapped_bytes() / (1024.0 * 1024.0 * 1024.0),
                engram_.rows_local(), engram_.heads_local, engram_.ngrams);
  return engram_;
}

}  // namespace dgpp
