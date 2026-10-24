#include "models/qwen/loader.hpp"
#include "kernels/packq_head.hpp"

#include "loaders/fp8_quant.hpp"
#include "kernels/latent_format.hpp"

#include <sys/mman.h>
#include <sys/uio.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include <initializer_list>
#include <utility>
#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "loaders/gptq_repack.hpp"

namespace dgpp {
namespace {

bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// The replicated set (rank-invariant reads at world > 1): the gated
// residual sites and the final mixers, routers and shared gates, the whole
// indexer, the PLE's norms/conv/buffers and the table's scale, the head
// norms of GDN and QSA, the draft head's own tensors, and the embedding
// (a row gather; sharding it buys nothing in v1). Everything else is a
// slice: GDN/QSA projections, both expert classes, the PLE projections,
// the n-gram shards, the lm head under VocabSharded.
bool is_replicated(const QwenExpectedTensor& e) {
  // ModelOpt stores one scalar weight scale and one scalar activation scale
  // beside every NVFP4 matrix. The payload and per-group scales are sliced,
  // but these two scalars are rank-invariant metadata read by every rank.
  if (e.role == QwenTensorRole::Fp4Global || e.role == QwenTensorRole::InputScale)
    return true;
  switch (e.cls) {
    case QwenWeightClass::Embed:
    case QwenWeightClass::Mixer:
    case QwenWeightClass::Gr:
    case QwenWeightClass::Router:
    case QwenWeightClass::QsaIndexer:
    case QwenWeightClass::Ple:
    case QwenWeightClass::Mtp:
      return true;
    case QwenWeightClass::PleTable:
      return e.role == QwenTensorRole::NgramScale;
    case QwenWeightClass::Gdn:
      return ends_with(e.name, "linear_attn.norm.weight");
    case QwenWeightClass::Qsa:
      return ends_with(e.name, "q_norm.weight") || ends_with(e.name, "k_norm.weight");
    case QwenWeightClass::LmHead:
    case QwenWeightClass::SharedExpert:
    case QwenWeightClass::RoutedExpert:
      return false;
    case QwenWeightClass::Norm:
    case QwenWeightClass::FullAttn:
    case QwenWeightClass::DenseMlp:
      // qwen3_5 classes: this family's table never carries them (loader35
      // owns their replication rule).
      return false;
  }
  return false;
}

int gcd_int(int a, int b) { return std::gcd(a, b); }

// The n-gram table's per-tensor scale: BF16 `weight_scale` (the FP8
// table), or the NVFP4 table's F32 `weight_scale_2` (the e2m1 x e4m3
// product's global).
std::string qwen_ngram_table_scale_name(const QwenTextConfig& cfg, const std::string& ple_embedding_prefix) {
  return ple_embedding_prefix + (cfg.ngram_table_nvfp4 ? "ngram_embedding.weight_scale_2" : "ngram_embedding.weight_scale");
}
float qwen_ngram_table_scale(const QwenTextConfig& cfg, const TensorInfo& t) {
  if (cfg.ngram_table_nvfp4) {
    float v;
    std::memcpy(&v, t.data, 4);
    if (!(v > 0.0f) || !std::isfinite(v))
      throw std::runtime_error("qwen loader: '" + t.name + "' is not a positive finite scale");
    return v;
  }
  uint16_t bits;
  std::memcpy(&bits, t.data, 2);
  return bf16_bits_to_float(bits);
}
// Bytes of one table row as stored: head_dim e4m3 codes, or the NVFP4
// table's codes (head_dim / 2) and scales (head_dim / 16) — summed.
size_t qwen_ngram_row_bytes(const QwenTextConfig& cfg) {
  const int hd = cfg.ngram_geometry().head_dim;
  return cfg.ngram_table_nvfp4 ? static_cast<size_t>(hd / 2 + hd / kQwenNvfp4Group) : static_cast<size_t>(hd);
}

std::pair<int, int> lm_head_slice(const QwenTextConfig& cfg, int rank, int world) {
  const int64_t V = cfg.vocab_size;
  const int64_t begin = V * rank / world;
  const int64_t end = V * (rank + 1) / world;
  return {static_cast<int>(begin), static_cast<int>(end - begin)};
}

std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}

// The n-gram table's residency: process-wide, set before
// any stream is built (the memory plan reads it too).
bool g_ngram_table_mmap = false;
std::string& ngram_table_dir_storage() {
  static std::string dir;
  return dir;
}
// The dense stack's form (engine.dense_weights = "fp8", 2026-09-10).
bool g_dense_weights_fp8 = false;
std::vector<int32_t> g_draft_vocab_ids;

// A one-dimensional int32/int64 .npy (format 1.0 or 2.0, little-endian, C order).
bool read_npy_ids(const std::string& path, std::vector<int32_t>* ids, std::string* err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { *err = "cannot open"; return false; }
  char magic[8];
  f.read(magic, 8);
  if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0) { *err = "not a .npy file"; return false; }
  const int major = static_cast<unsigned char>(magic[6]);
  uint32_t hlen = 0;
  if (major == 1) {
    unsigned char b[2];
    f.read(reinterpret_cast<char*>(b), 2);
    hlen = b[0] | (b[1] << 8);
  } else {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    hlen = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<uint32_t>(b[3]) << 24);
  }
  std::string header(hlen, '\0');
  f.read(header.data(), hlen);
  if (!f) { *err = "truncated header"; return false; }
  int width = 0;
  if (header.find("'<i4'") != std::string::npos) width = 4;
  else if (header.find("'<i8'") != std::string::npos) width = 8;
  else { *err = "dtype must be little-endian int32 or int64"; return false; }
  if (header.find("'fortran_order': False") == std::string::npos) { *err = "fortran order"; return false; }
  const size_t sp = header.find("'shape': (");
  if (sp == std::string::npos) { *err = "no shape"; return false; }
  const size_t n = static_cast<size_t>(std::strtoull(header.c_str() + sp + 10, nullptr, 10));
  const size_t close = header.find(')', sp);
  if (close == std::string::npos || header.substr(sp + 10, close - sp - 10).find(',') != std::string::npos) {
    const std::string inner = close == std::string::npos ? "" : header.substr(sp + 10, close - sp - 10);
    // "(N,)" is one-dimensional; "(N, M)" is not.
    size_t commas = 0;
    for (char c : inner) commas += c == ',';
    if (commas != 1 || inner.find_first_not_of("0123456789, ") != std::string::npos) { *err = "shape must be (N,)"; return false; }
  }
  if (n == 0) { *err = "empty"; return false; }
  std::vector<char> raw(n * width);
  f.read(raw.data(), static_cast<std::streamsize>(raw.size()));
  if (!f) { *err = "truncated data"; return false; }
  ids->resize(n);
  for (size_t i = 0; i < n; ++i) {
    int64_t v;
    if (width == 4) { int32_t x; std::memcpy(&x, raw.data() + i * 4, 4); v = x; }
    else { std::memcpy(&v, raw.data() + i * 8, 8); }
    if (v < 0 || v > 0x7FFFFFFF) { *err = "an id is outside int32"; return false; }
    (*ids)[i] = static_cast<int32_t>(v);
  }
  std::sort(ids->begin(), ids->end());
  ids->erase(std::unique(ids->begin(), ids->end()), ids->end());
  return true;
}
// The RadixArk MTP expert format (engine.mtp_expert_format = "bf16_fused"):
// fused BF16 gate_up_proj + down_proj instead of per-expert FP8 tensors.
bool g_mtp_experts_bf16_fused = false;
}  // namespace

// Expand ModelOpt NVFP4 CSF sidecars (codec
// byte-window4-fixed-stream-u24-exceptions/1, per the vLLM csf_scales
// reader) to a plain [rows, cols] E4M3 scale plane: per 16-row slab, one
// base byte per row, then the rows' 4-bit codes low-nibble-first; a scale
// byte is base + code, and each exception word's high 8 bits replace the
// byte at its low-24-bit flat position. Pure host integer arithmetic;
// throws std::runtime_error on any geometry or content violation.
std::vector<uint8_t> qwen_csf_scales_expand(const uint8_t* fixed, size_t fixed_bytes,
                                            const uint32_t* exc, size_t exc_count, int64_t rows,
                                            int64_t cols) {
  auto bad = [](const std::string& what) -> std::vector<uint8_t> {
    throw std::runtime_error("CSF scale expansion: " + what);
  };
  if (rows <= 0 || cols <= 0 || rows % 16 != 0 || cols % 2 != 0)
    bad("plane [" + std::to_string(rows) + ", " + std::to_string(cols) + "] is not 16-row / even-col");
  if (fixed == nullptr || (exc_count != 0 && exc == nullptr)) bad("null sidecar data");
  const int64_t slab = 16 * (1 + cols / 2);
  if (fixed_bytes != static_cast<size_t>(rows / 16) * static_cast<size_t>(slab))
    bad("fixed stream has " + std::to_string(fixed_bytes) + " bytes, want " +
        std::to_string(rows / 16 * slab));
  std::vector<uint8_t> out(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int64_t g = 0; g < rows / 16; ++g) {
    const uint8_t* base = fixed + static_cast<size_t>(g) * static_cast<size_t>(slab);
    const uint8_t* packed = base + 16;
    for (int64_t r = 0; r < 16; ++r) {
      const unsigned b = base[static_cast<size_t>(r)];
      if (b > 240) bad("row base " + std::to_string(b) + " exceeds 240");
      uint8_t* dst = out.data() + static_cast<size_t>(g * 16 + r) * static_cast<size_t>(cols);
      for (int64_t c = 0; c < cols / 2; ++c) {
        const unsigned p = packed[static_cast<size_t>(r) * static_cast<size_t>(cols / 2) +
                                  static_cast<size_t>(c)];
        dst[2 * static_cast<size_t>(c)] = static_cast<uint8_t>(b + (p & 0xF));
        dst[2 * static_cast<size_t>(c) + 1] = static_cast<uint8_t>(b + (p >> 4));
      }
    }
  }
  const size_t n = out.size();
  for (size_t i = 0; i < exc_count; ++i) {
    // The exceptions stream is a safetensors tensor and is not word-aligned,
    // so read each little-endian word bytewise rather than dereferencing the
    // caller's uint32_t* (UB / a fault on strict-alignment targets).
    uint32_t w;
    std::memcpy(&w, reinterpret_cast<const uint8_t*>(exc) + i * 4, 4);
    const size_t pos = static_cast<size_t>(w & 0xFFFFFFu);
    if (pos >= n) bad("exception position " + std::to_string(pos) + " outside " + std::to_string(n));
    out[pos] = static_cast<uint8_t>(w >> 24);
  }
  return out;
}

// The per-class builders (loaders/weight_build.hpp's primitives).
struct QwenLoaderFamily::Builder : WeightBuilder<QwenExpectedTensor> {
  const QwenTextConfig& cfg;
  const QwenLocalGeometry& geo;
  QwenLayerResident& out;

  Builder(const QwenTextConfig& cfg_, const QwenLocalGeometry& geo_,
           const std::vector<QwenExpectedTensor>& table_,
           const std::unordered_map<std::string, const QwenExpectedTensor*>& by_name_,
           LayerBump& bump_, QwenLayerResident& out_,
           const std::unordered_map<std::string, const TensorInfo*>& tensors_,
           std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<QwenExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_,
                                          copy_, geo_.rank, geo_.world, "qwen loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const QwenExpectedTensor& e) const override { return is_replicated(e); }

  // A contiguous BF16 row range of a source matrix into a running
  // destination (the GDN's segmented q|k|v merge): rows [src_row, +rows)
  // of `name` land at `dst` + dst_row * width. Returns the source's row
  // width in elements.
  int64_t copy_rows_into(const std::string& name, int64_t src_row, int64_t rows,
                         uint16_t* dst, int64_t dst_row) {
    const QwenExpectedTensor& e = expected(name);
    const int64_t width = static_cast<int64_t>(e.numel()) / e.shape[0];
    check_range(name, src_row, rows, e.shape[0]);
    if (copy) {
      const TensorInfo& t = source(name);
      std::memcpy(bump.host(dst) + static_cast<size_t>(dst_row) * width,
                  static_cast<const uint8_t*>(t.data) + static_cast<size_t>(src_row) * width * 2,
                  static_cast<size_t>(rows) * width * 2);
      // Not consumed: a later segment of the same tensor may follow.
    }
    note_read(e, static_cast<size_t>(rows) * width * 2);
    return width;
  }

  // ---- the AutoRound hybrid (docs/qwen38_autoround_int4_plan.md) ------------
  // The checkpoint ships its dense stack as block FP8: the model's fp8 paths
  // (engine.dense_weights = fp8) are the only ones that can read it.
  void require_dense_fp8_mode() const {
    if (!g_dense_weights_fp8)
      fail("the checkpoint ships its dense stack as block FP8 (the AutoRound hybrid): "
           "set engine.dense_weights = \"fp8\" (the loader takes the shipped codes and scales as is)");
  }

  // A pre-encoded block-FP8 matrix's row segments (the GDN's q | k | v at
  // the local geometry) into one resident matrix: payload rows and their
  // 128-row scale rows concatenated. Every segment starts and spans whole
  // 128-row blocks.
  GlmQuantMatrix load_quant_row_segments(const std::string& name,
                                         std::initializer_list<std::pair<int64_t, int64_t>> segs) {
    const QwenExpectedTensor& e = expected(name);
    const QwenExpectedTensor& es = expected(name + "_scale_inv");
    const int64_t cols = e.shape[1];
    const int64_t sb = (cols + 127) / 128;
    int64_t total = 0;
    for (const auto& [s, n] : segs) {
      if (s % 128 != 0 || n % 128 != 0)
        fail("quantized row segment of '" + name + "' must start and span whole 128-row blocks");
      check_range(name, s, n, e.shape[0]);
      total += n;
    }
    GlmQuantMatrix q;
    q.rows = total;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(total) * cols));
    q.scales = static_cast<const float*>(bump.alloc(static_cast<size_t>(total / 128) * sb * 4));
    if (copy) {
      const TensorInfo& tp = source(name);
      const TensorInfo& ts = source(es.name);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      float* hs = bump.host(const_cast<float*>(q.scales));
      int64_t dst = 0;
      for (const auto& [s, n] : segs) {
        std::memcpy(hp + dst * cols, static_cast<const uint8_t*>(tp.data) + s * cols,
                    static_cast<size_t>(n) * cols);
        for (int64_t r = 0; r < n / 128; ++r) {
          const int64_t src_row = s / 128 + r, dst_row = dst / 128 + r;
          if (ts.dtype == DType::F32) {
            std::memcpy(hs + dst_row * sb, static_cast<const float*>(ts.data) + src_row * sb, static_cast<size_t>(sb) * 4);
          } else if (ts.dtype == DType::BF16) {
            for (int64_t c = 0; c < sb; ++c) {
              uint16_t bits;
              std::memcpy(&bits, static_cast<const uint8_t*>(ts.data) + (src_row * sb + c) * 2, 2);
              hs[dst_row * sb + c] = bf16_bits_to_float(bits);
            }
          } else {
            fail("scale tensor '" + es.name + "' is neither F32 nor BF16");
          }
        }
        dst += n;
      }
      consumed(tp);
      consumed(ts);
    }
    note_read(e, static_cast<size_t>(total) * cols + static_cast<size_t>(total / 128) * sb * dtype_size(es.dtype));
    return q;
  }

  // ---- the QAD export's MXFP8 dense stack (config dense_mxfp8_shipped) ----
  // `name` is "...proj.weight", e4m3 [N, K]; its scale partner
  // `base.weight_scale` holds one e8m0 byte per 32 columns of a row (U8
  // [N, K/32]). The resident form is the fp8 cores' GlmQuantMatrix on a
  // 1 x 32 grid (rs = 0, cs = 5): the scale bytes as shipped for the
  // attention-side projections (the streaming MMA and the dequant bridge
  // read the byte grid; the F32 grid cost 12.5 % of the payload's bytes a
  // step), widened to F32 (2^(b - 127), exact) for the shared experts,
  // whose fused decode kernels take F32. Rows slice freely (a scale row per
  // weight row); a column slice spans whole 32-groups. A 255 byte (NaN) is
  // refused.
  static std::string mxfp8_scale_name(const std::string& name) {
    return name.substr(0, name.size() - std::string(".weight").size()) + ".weight_scale";
  }
  void check_mxfp8_pair(const QwenExpectedTensor& e, const QwenExpectedTensor& es,
                        const std::string& name) const {
    if (e.role != QwenTensorRole::Mxfp8Payload || es.role != QwenTensorRole::Mxfp8Scale)
      fail("'" + name + "' is not an MXFP8 pair in the binding");
    if (e.shape.size() != 2 || es.shape.size() != 2 || es.shape[0] != e.shape[0] ||
        es.shape[1] * kQwenMxfp8Group != e.shape[1])
      fail("MXFP8 scale geometry mismatch on " + name);
  }
  void widen_e8m0(const TensorInfo& ts, int64_t index, int64_t count, float* dst) const {
    const uint8_t* src = static_cast<const uint8_t*>(ts.data) + index;
    for (int64_t i = 0; i < count; ++i) {
      if (src[i] == 255u) fail("'" + ts.name + "' holds a NaN e8m0 scale");
      dst[i] = e8m0_byte_to_float(src[i]);
    }
  }
  // The scale bytes kept as shipped (the byte grid the streaming MMA and the
  // dequant bridge read); a NaN byte is refused as above.
  void copy_e8m0(const TensorInfo& ts, int64_t index, int64_t count, uint8_t* dst) const {
    const uint8_t* src = static_cast<const uint8_t*>(ts.data) + index;
    for (int64_t i = 0; i < count; ++i)
      if (src[i] == 255u) fail("'" + ts.name + "' holds a NaN e8m0 scale");
    std::memcpy(dst, src, static_cast<size_t>(count));
  }
  // The resident scale form of an MXFP8 matrix: the e8m0 bytes themselves
  // when the streaming MMA can take the matrix (its k a multiple of 64 —
  // every attention-side projection at worlds 1, 2 and 4), the F32 grid
  // otherwise (and for the shared experts, whose fused decode kernels read
  // F32 scales).
  static bool mxfp8_bytes_ok(int64_t cols, bool prefer_bytes) { return prefer_bytes && cols % 64 == 0; }
  GlmQuantMatrix mxfp8_matrix(int64_t rows, int64_t cols, bool bytes) {
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_block_rows = 1;
    q.scale_block_cols = kQwenMxfp8Group;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols)));
    const size_t grid = static_cast<size_t>(rows) * static_cast<size_t>(cols / kQwenMxfp8Group);
    if (bytes)
      q.scales_e8m0 = static_cast<const uint8_t*>(bump.alloc(grid));
    else
      q.scales = static_cast<const float*>(bump.alloc(grid * 4));
    return q;
  }
  // The scale slice [index, +count) of the source into the resident grid, in its form.
  void mxfp8_scales_into(const GlmQuantMatrix& q, const TensorInfo& ts, int64_t index, int64_t count, int64_t dst_index) {
    if (q.scales_e8m0)
      copy_e8m0(ts, index, count, bump.host(const_cast<uint8_t*>(q.scales_e8m0)) + dst_index);
    else
      widen_e8m0(ts, index, count, bump.host(const_cast<float*>(q.scales)) + dst_index);
  }
  // Row segments of an MXFP8 matrix concatenated (the GDN's q | k | v at
  // the local geometry; one segment = a plain row slice).
  GlmQuantMatrix load_mxfp8_row_segments(const std::string& name,
                                         std::initializer_list<std::pair<int64_t, int64_t>> segs,
                                         bool prefer_bytes = true) {
    const QwenExpectedTensor& e = expected(name);
    const QwenExpectedTensor& es = expected(mxfp8_scale_name(name));
    check_mxfp8_pair(e, es, name);
    const int64_t cols = e.shape[1];
    const int64_t sb = cols / kQwenMxfp8Group;
    int64_t total = 0;
    for (const auto& [st, n] : segs) {
      check_range(name, st, n, e.shape[0]);
      total += n;
    }
    GlmQuantMatrix q = mxfp8_matrix(total, cols, mxfp8_bytes_ok(cols, prefer_bytes));
    if (copy) {
      const TensorInfo& tp = source(name);
      const TensorInfo& ts = source(es.name);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      int64_t dst = 0;
      for (const auto& [st, n] : segs) {
        std::memcpy(hp + dst * cols, static_cast<const uint8_t*>(tp.data) + st * cols,
                    static_cast<size_t>(n) * cols);
        mxfp8_scales_into(q, ts, st * sb, n * sb, dst * sb);
        dst += n;
      }
      consumed(tp);
      consumed(ts);
    }
    note_read(e, static_cast<size_t>(total) * cols);
    note_read(es, static_cast<size_t>(total) * sb);
    return q;
  }
  GlmQuantMatrix load_mxfp8_rows(const std::string& name, int64_t row_start, int64_t rows, bool prefer_bytes = true) {
    return load_mxfp8_row_segments(name, {{row_start, rows}}, prefer_bytes);
  }
  // A packed column slice [N, cols] of an MXFP8 [N, K]: whole 32-groups.
  GlmQuantMatrix load_mxfp8_cols(const std::string& name, int64_t col_start, int64_t cols, bool prefer_bytes = true) {
    const QwenExpectedTensor& e = expected(name);
    const QwenExpectedTensor& es = expected(mxfp8_scale_name(name));
    check_mxfp8_pair(e, es, name);
    const int64_t rows = e.shape[0], full = e.shape[1];
    if (col_start % kQwenMxfp8Group != 0 || cols % kQwenMxfp8Group != 0 || cols <= 0)
      fail("MXFP8 column slice of '" + name + "' must start and span whole 32-groups");
    check_range(name, col_start, cols, full);
    const int64_t sb_full = full / kQwenMxfp8Group, sb = cols / kQwenMxfp8Group;
    GlmQuantMatrix q = mxfp8_matrix(rows, cols, mxfp8_bytes_ok(cols, prefer_bytes));
    if (copy) {
      const TensorInfo& tp = source(name);
      const TensorInfo& ts = source(es.name);
      const uint8_t* sp = static_cast<const uint8_t*>(tp.data);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      if (cols == full) {
        std::memcpy(hp, sp, static_cast<size_t>(rows) * cols);
        mxfp8_scales_into(q, ts, 0, rows * sb, 0);
      } else {
        for (int64_t r = 0; r < rows; ++r) {
          std::memcpy(hp + r * cols, sp + r * full + col_start, static_cast<size_t>(cols));
          mxfp8_scales_into(q, ts, r * sb_full + col_start / kQwenMxfp8Group, sb, r * sb);
        }
      }
      consumed(tp);
      consumed(ts);
    }
    note_read(e, static_cast<size_t>(rows) * cols);
    note_read(es, static_cast<size_t>(rows) * sb);
    return q;
  }
  // A row slice of an MXFP8 matrix decoded to BF16 — bf16(e4m3 x 2^k), the
  // value every fp8 core's dequant bridge produces, exact — for the GDN's
  // a and b projections ([lv, H], 0.12 MB each), which ride the bf16 GEMV
  // beside the fp8 qkv / z launch (the a/b kernels take BF16 rows).
  uint16_t* load_mxfp8_rows_as_bf16(const std::string& name, int64_t row_start, int64_t rows) {
    const QwenExpectedTensor& e = expected(name);
    const QwenExpectedTensor& es = expected(mxfp8_scale_name(name));
    check_mxfp8_pair(e, es, name);
    const int64_t cols = e.shape[1];
    const int64_t sb = cols / kQwenMxfp8Group;
    check_range(name, row_start, rows, e.shape[0]);
    uint16_t* dst = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols) * 2));
    if (copy) {
      const TensorInfo& tp = source(name);
      const TensorInfo& ts = source(es.name);
      const uint8_t* sp = static_cast<const uint8_t*>(tp.data) + static_cast<size_t>(row_start) * cols;
      const uint8_t* ss = static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(row_start) * sb;
      uint16_t* h = bump.host(dst);
      for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < cols; ++c) {
          const uint8_t sc = ss[r * sb + c / kQwenMxfp8Group];
          if (sc == 255u) fail("'" + es.name + "' holds a NaN e8m0 scale");
          h[r * cols + c] = float_to_bf16_bits(fp8_e4m3_bits_to_float(sp[r * cols + c]) * e8m0_byte_to_float(sc));
        }
      consumed(tp);
      consumed(ts);
    }
    note_read(e, static_cast<size_t>(rows) * cols);
    note_read(es, static_cast<size_t>(rows) * sb);
    return dst;
  }
  // The MXFP8 stack is the checkpoint's own form: served as shipped, no
  // engine key — engine.dense_weights applies to the BF16 remainder (the
  // GR sites, the PLE projections, lm_head, the draft layer).
  bool mxfp8_layer() const { return cfg.dense_mxfp8_shipped && out.layer != cfg.mtp_layer(); }

  // The GPTQ triple of the logical [N, K] matrix `base` — qweight I32
  // [K*bits/32, N], scales F16 [K/g, N], qzeros I32 [K/g, N*bits/32] — as
  // the packed core's [n, K*bits/32] words and [n, K/g] f16 scales for the
  // N range [n0, +n) and the K range [k0, +kn) (whole groups): the same
  // words and scales, transposed at load (loaders/gptq_repack.hpp), the
  // zeros verified and dropped. The copies are deferred to gptq_jobs_ and
  // run in parallel by run_gptq_jobs() (the bump's grant order stays
  // sequential, as the counting build walks it).
  std::vector<GptqRepackJob> gptq_jobs_;
  std::vector<std::string> gptq_job_names_;
  GlmPackedMatrix gptq_slice(const std::string& base, int bits, int64_t n0, int64_t n, int64_t k0,
                             int64_t kn) {
    const int g = cfg.gptq_group;
    const int per = 32 / bits;
    const QwenExpectedTensor& ew = expected(base + ".qweight");
    const QwenExpectedTensor& es = expected(base + ".scales");
    const QwenExpectedTensor& ez = expected(base + ".qzeros");
    const int64_t N = ew.shape[1], K = ew.shape[0] * per;
    if (es.shape[0] != K / g || es.shape[1] != N || ez.shape[0] != K / g || ez.shape[1] != N * bits / 32)
      fail("GPTQ triple geometry mismatch on " + base);
    check_range(base, n0, n, N);
    check_range(base, k0, kn, K);
    if (k0 % g != 0 || kn % g != 0 || kn <= 0)
      fail("GPTQ column slice of '" + base + "' must start and span whole groups of " +
           std::to_string(g) + " (world 1 serves the AutoRound hybrid; its down projection's K "
           "does not slice on the group grid)");
    GlmPackedMatrix m;
    m.rows = n;
    m.cols = kn;
    m.bits = bits;
    m.scale_fmt = kPackedScaleF16G128;
    const size_t word_rows = static_cast<size_t>(kn / per);
    const size_t groups = static_cast<size_t>(kn / g);
    m.packed = static_cast<const uint32_t*>(bump.alloc(static_cast<size_t>(n) * word_rows * 4));
    m.scales = static_cast<const uint16_t*>(bump.alloc(static_cast<size_t>(n) * groups * 2));
    if (copy) {
      const TensorInfo& tw = source(ew.name);
      const TensorInfo& ts = source(es.name);
      const TensorInfo& tz = source(ez.name);
      GptqRepackJob j;
      j.qweight = static_cast<const uint32_t*>(tw.data);
      j.qw_stride = N;
      j.word_row0 = k0 / per;
      j.word_rows = kn / per;
      j.n0 = n0;
      j.n = n;
      j.scales = static_cast<const uint16_t*>(ts.data);
      j.sc_stride = N;
      j.group0 = k0 / g;
      j.groups = kn / g;
      j.qzeros = static_cast<const uint32_t*>(tz.data);
      j.qz_stride = N * bits / 32;
      j.bits = bits;
      j.dst_words = bump.host(const_cast<uint32_t*>(m.packed));
      j.dst_scales = bump.host(const_cast<uint16_t*>(m.scales));
      gptq_jobs_.push_back(j);
      gptq_job_names_.push_back(base);
    }
    note_read(ew, static_cast<size_t>(n) * word_rows * 4);
    note_read(es, static_cast<size_t>(n) * groups * 2);
    note_read(ez, groups * (static_cast<size_t>(n) * bits / 32) * 4);
    return m;
  }
  GlmPackedMatrix load_gptq_rows(const std::string& base, int64_t row_start, int64_t rows, int bits) {
    const QwenExpectedTensor& ew = expected(base + ".qweight");
    return gptq_slice(base, bits, row_start, rows, 0, ew.shape[0] * (32 / bits));
  }
  GlmPackedMatrix load_gptq_cols(const std::string& base, int64_t col_start, int64_t cols, int bits) {
    const QwenExpectedTensor& ew = expected(base + ".qweight");
    return gptq_slice(base, bits, 0, ew.shape[1], col_start, cols);
  }
  void run_gptq_jobs() {
    if (gptq_jobs_.empty()) return;
    const int64_t bad = gptq_repack_all(gptq_jobs_, 16);
    if (bad >= 0)
      fail("'" + gptq_job_names_[static_cast<size_t>(bad)] +
           "': a qzeros word is not the symmetric constant (the packed core's fixed offset "
           "needs zero = 2^(bits-1); an asymmetric GPTQ checkpoint is not served)");
    for (const std::string& base : gptq_job_names_) {
      consumed(source(base + ".qweight"));
      consumed(source(base + ".scales"));
      consumed(source(base + ".qzeros"));
    }
    gptq_jobs_.clear();
    gptq_job_names_.clear();
  }

  void build_gr(const std::string& p, QwenGrResident& g, bool combine) {
    // The QAD export's trained norms are F32 on the backbone (the draft
    // layer's stay BF16): one of the two pointers is set.
    if (cfg.norms_f32 && out.layer != cfg.mtp_layer())
      g.hc_norm_f32 = load_f32(p + "hc_norm.weight");
    else
      g.hc_norm = load_bf16(p + "hc_norm.weight");
    // The at-load fp8 encode applies to the BF16 releases only: the hybrid
    // ships this class in BF16 and it stays so (as shipped, D4).
    if (g_dense_weights_fp8 && !cfg.dense_fp8_shipped) {
      g.down_fp8 = load_bf16_fp8(p + "input_mix_weight_down.weight");
      g.up_fp8 = load_bf16_fp8(p + "input_mix_weight_up.weight");
    } else {
      g.down = load_bf16(p + "input_mix_weight_down.weight");
      g.up = load_bf16(p + "input_mix_weight_up.weight");
    }
    g.inject = combine ? load_bf16(p + "block_inject_weight.weight") : nullptr;
  }

  void build_gdn(const std::string& p) {
    const int64_t H = cfg.hidden_size;
    const int64_t dk = cfg.gdn_key_head_dim, dv = cfg.gdn_value_head_dim;
    const int64_t K = static_cast<int64_t>(cfg.gdn_key_heads) * dk;
    const int64_t lk = geo.local_key_heads, lv = geo.local_value_heads;
    const int64_t r = rank;
    QwenGdnResident& g = out.gdn;
    g.local_key_heads = static_cast<int>(lk);
    g.local_value_heads = static_cast<int>(lv);
    // in_proj_qkv rows [q | k | v] at the local geometry: this rank's key
    // heads' q and k rows, its value heads' v rows. world=1: the whole
    // matrix, byte for byte.
    const int64_t local_rows = 2 * lk * dk + lv * dv;
    const std::string qkv_name = p + "in_proj_qkv.weight";
    if (mxfp8_layer()) {
      // The QAD export's MXFP8 stack: the three segments' codes and scale
      // rows as shipped (a scale row per weight row: any segment boundary).
      g.in_proj_qkv_fp8 = load_mxfp8_row_segments(
          qkv_name, {{r * lk * dk, lk * dk}, {K + r * lk * dk, lk * dk}, {2 * K + r * lv * dv, lv * dv}});
    } else if (cfg.dense_fp8_shipped) {
      // The hybrid's pre-encoded block FP8: the three segments' codes and
      // scale rows as shipped.
      require_dense_fp8_mode();
      g.in_proj_qkv_fp8 = load_quant_row_segments(
          qkv_name, {{r * lk * dk, lk * dk}, {K + r * lk * dk, lk * dk}, {2 * K + r * lv * dv, lv * dv}});
    } else if (g_dense_weights_fp8 && !cfg.dense_fp8_shipped) {
      // The three segments assembled on the host, encoded into the bump.
      std::vector<uint16_t> merged;
      if (copy) {
        merged.resize(static_cast<size_t>(local_rows) * static_cast<size_t>(H));
        const uint16_t* src = static_cast<const uint16_t*>(source(qkv_name).data);
        const auto seg = [&](int64_t src_row, int64_t rows, int64_t dst_row) {
          std::memcpy(merged.data() + static_cast<size_t>(dst_row) * H, src + static_cast<size_t>(src_row) * H,
                      static_cast<size_t>(rows) * H * 2);
        };
        seg(r * lk * dk, lk * dk, 0);
        seg(K + r * lk * dk, lk * dk, lk * dk);
        seg(2 * K + r * lv * dv, lv * dv, 2 * lk * dk);
      }
      note_read(expected(qkv_name), static_cast<size_t>(local_rows) * H * 2);
      g.in_proj_qkv_fp8 = encode_fp8(merged.data(), static_cast<size_t>(H), local_rows, H);
      if (copy) consumed(source(qkv_name));
    } else {
      uint16_t* qkv = static_cast<uint16_t*>(
          bump.alloc(static_cast<size_t>(local_rows) * static_cast<size_t>(H) * 2));
      copy_rows_into(qkv_name, r * lk * dk, lk * dk, qkv, 0);
      copy_rows_into(qkv_name, K + r * lk * dk, lk * dk, qkv, lk * dk);
      copy_rows_into(qkv_name, 2 * K + r * lv * dv, lv * dv, qkv, 2 * lk * dk);
      if (copy) consumed(source(qkv_name));
      g.in_proj_qkv = qkv;
    }
    // The conv channels follow the same three segments ([C, 1, w] rows).
    const int64_t w = cfg.gdn_conv_width;
    uint16_t* conv = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(local_rows) * static_cast<size_t>(w) * 2));
    const std::string conv_name = p + "conv1d.weight";
    copy_rows_into(conv_name, r * lk * dk, lk * dk, conv, 0);
    copy_rows_into(conv_name, K + r * lk * dk, lk * dk, conv, lk * dk);
    copy_rows_into(conv_name, 2 * K + r * lv * dv, lv * dv, conv, 2 * lk * dk);
    if (copy) consumed(source(conv_name));
    g.conv = conv;
    if (mxfp8_layer())
      g.in_proj_z_fp8 = load_mxfp8_rows(p + "in_proj_z.weight", r * lv * dv, lv * dv);
    else if (cfg.dense_fp8_shipped)
      g.in_proj_z_fp8 = load_quant_rows(p + "in_proj_z.weight", r * lv * dv, lv * dv);
    else if (g_dense_weights_fp8 && !cfg.dense_fp8_shipped)
      g.in_proj_z_fp8 = load_bf16_rows_fp8(p + "in_proj_z.weight", r * lv * dv, lv * dv);
    else
      g.in_proj_z = load_bf16_rows(p + "in_proj_z.weight", r * lv * dv, lv * dv);
    if (mxfp8_layer()) {
      g.in_proj_a = load_mxfp8_rows_as_bf16(p + "in_proj_a.weight", r * lv, lv);
      g.in_proj_b = load_mxfp8_rows_as_bf16(p + "in_proj_b.weight", r * lv, lv);
    } else {
      g.in_proj_a = load_bf16_rows(p + "in_proj_a.weight", r * lv, lv);
      g.in_proj_b = load_bf16_rows(p + "in_proj_b.weight", r * lv, lv);
    }
    g.a_log = load_bf16_as_f32(p + "A_log", r * lv, lv);
    g.dt_bias = load_bf16_as_f32(p + "dt_bias", r * lv, lv);
    g.norm = load_bf16(p + "norm.weight");
    if (mxfp8_layer())
      g.out_proj_fp8 = load_mxfp8_cols(p + "out_proj.weight", r * lv * dv, lv * dv);
    else if (cfg.dense_fp8_shipped)
      g.out_proj_fp8 = load_quant_cols(p + "out_proj.weight", r * lv * dv, lv * dv);
    else if (g_dense_weights_fp8 && !cfg.dense_fp8_shipped)
      g.out_proj_fp8 = load_bf16_cols_fp8(p + "out_proj.weight", r * lv * dv, lv * dv);
    else
      g.out_proj = load_bf16_cols(p + "out_proj.weight", r * lv * dv, lv * dv);
  }

  void build_qsa(const std::string& p) {
    const int64_t d = cfg.head_dim;
    QwenQsaResident& a = out.qsa;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_kv_heads = geo.local_kv_heads;
    a.kv_head_begin = geo.kv_head_begin;
    const int64_t q0 = static_cast<int64_t>(geo.head_begin) * 2 * d, qn = static_cast<int64_t>(geo.local_heads) * 2 * d;
    const int64_t kv0 = static_cast<int64_t>(geo.kv_head_begin) * d, kvn = static_cast<int64_t>(geo.local_kv_heads) * d;
    const int64_t o0 = static_cast<int64_t>(geo.head_begin) * d, on = static_cast<int64_t>(geo.local_heads) * d;
    if (mxfp8_layer()) {
      a.q_proj_fp8 = load_mxfp8_rows(p + "q_proj.weight", q0, qn);
      a.k_proj_fp8 = load_mxfp8_rows(p + "k_proj.weight", kv0, kvn);
      a.v_proj_fp8 = load_mxfp8_rows(p + "v_proj.weight", kv0, kvn);
      a.o_proj_fp8 = load_mxfp8_cols(p + "o_proj.weight", o0, on);
    } else if (cfg.dense_fp8_shipped && out.layer != cfg.mtp_layer()) {
      require_dense_fp8_mode();
      a.q_proj_fp8 = load_quant_rows(p + "q_proj.weight", q0, qn);
      a.k_proj_fp8 = load_quant_rows(p + "k_proj.weight", kv0, kvn);
      a.v_proj_fp8 = load_quant_rows(p + "v_proj.weight", kv0, kvn);
      a.o_proj_fp8 = load_quant_cols(p + "o_proj.weight", o0, on);
    } else if (g_dense_weights_fp8 && !cfg.dense_fp8_shipped) {
      a.q_proj_fp8 = load_bf16_rows_fp8(p + "q_proj.weight", q0, qn);
      a.k_proj_fp8 = load_bf16_rows_fp8(p + "k_proj.weight", kv0, kvn);
      a.v_proj_fp8 = load_bf16_rows_fp8(p + "v_proj.weight", kv0, kvn);
      a.o_proj_fp8 = load_bf16_cols_fp8(p + "o_proj.weight", o0, on);
    } else {
      a.q_proj = load_bf16_rows(p + "q_proj.weight", q0, qn);
      a.k_proj = load_bf16_rows(p + "k_proj.weight", kv0, kvn);
      a.v_proj = load_bf16_rows(p + "v_proj.weight", kv0, kvn);
      a.o_proj = load_bf16_cols(p + "o_proj.weight", o0, on);
    }
    a.q_norm = load_bf16(p + "q_norm.weight");
    a.k_norm = load_bf16(p + "k_norm.weight");
    if (mxfp8_layer()) {
      const QwenExpectedTensor& ei = expected(p + "indexer.index_qk_proj.weight");
      a.index_qk_proj_fp8 = load_mxfp8_rows(p + "indexer.index_qk_proj.weight", 0, ei.shape[0]);
    } else if (g_dense_weights_fp8 && !cfg.dense_fp8_shipped)
      a.index_qk_proj_fp8 = load_bf16_fp8(p + "indexer.index_qk_proj.weight");
    else
      a.index_qk_proj = load_bf16(p + "indexer.index_qk_proj.weight");
    a.index_q_norm = load_bf16(p + "indexer.q_layernorm.weight");
    a.index_k_norm = load_bf16(p + "indexer.k_layernorm.weight");
  }

  void build_moe(const std::string& p) {
    QwenMoeResident& m = out.moe;
    m.router = load_bf16(p + "gate.weight");
    m.shared_gate = load_bf16(p + "shared_expert_gate.weight");
    const int64_t S = geo.local_shared_inter, I = geo.local_inter;
    const int64_t r = rank;
    m.local_inter = I;
    m.local_shared_inter = S;
    m.scale_block = geo.scale_block;
    const std::string sp = p + "shared_expert.";
    if (mxfp8_layer()) {
      // The shared expert's fused decode kernels read F32 scales: the grid
      // widened here (0.2 MB a matrix beside 1.6 MB of codes).
      m.shared_fp8[0] = load_mxfp8_rows(sp + "gate_proj.weight", r * S, S, /*prefer_bytes=*/false);
      m.shared_fp8[1] = load_mxfp8_rows(sp + "up_proj.weight", r * S, S, /*prefer_bytes=*/false);
      m.shared_fp8[2] = load_mxfp8_cols(sp + "down_proj.weight", r * S, S, /*prefer_bytes=*/false);
    } else if (cfg.dense_fp8_shipped && out.layer != cfg.mtp_layer()) {
      require_dense_fp8_mode();
      const int sblk = gcd_int(128, static_cast<int>(S));
      m.shared_fp8[0] = load_quant_rows(sp + "gate_proj.weight", r * S, S, sblk);
      m.shared_fp8[1] = load_quant_rows(sp + "up_proj.weight", r * S, S, sblk);
      m.shared_fp8[2] = load_quant_cols(sp + "down_proj.weight", r * S, S, sblk);
    } else if (g_dense_weights_fp8 && !cfg.dense_fp8_shipped) {
      m.shared_fp8[0] = load_bf16_rows_fp8(sp + "gate_proj.weight", r * S, S);
      m.shared_fp8[1] = load_bf16_rows_fp8(sp + "up_proj.weight", r * S, S);
      m.shared_fp8[2] = load_bf16_cols_fp8(sp + "down_proj.weight", r * S, S);
    } else {
      m.shared[0] = load_bf16_rows(sp + "gate_proj.weight", r * S, S);
      m.shared[1] = load_bf16_rows(sp + "up_proj.weight", r * S, S);
      m.shared[2] = load_bf16_cols(sp + "down_proj.weight", r * S, S);
    }
    const int E = cfg.num_experts;
    // The AutoRound hybrid: every layer's experts (the draft layer's too) as
    // int4 g128 GPTQ triples, transposed into the packed core's rows.
    if (cfg.experts_gptq_int4) {
      m.experts_packed.resize(static_cast<size_t>(E) * 3);
      for (int e = 0; e < E; ++e) {
        const std::string ep = p + "experts." + std::to_string(e) + ".";
        m.experts_packed[static_cast<size_t>(e) * 3 + 0] = load_gptq_rows(ep + "gate_proj", r * I, I, 4);
        m.experts_packed[static_cast<size_t>(e) * 3 + 1] = load_gptq_rows(ep + "up_proj", r * I, I, 4);
        m.experts_packed[static_cast<size_t>(e) * 3 + 2] = load_gptq_cols(ep + "down_proj", r * I, I, 4);
      }
      return;
    }
    // The NVFP4 release's backbone experts (the MTP layer's stay FP8): the
    // modelopt triple per matrix, sliced on the intermediate axis like the
    // FP8 form — rows for gate/up, whole 16-blocks of columns for down. The
    // QAD export's draft experts take the same path weight-only: no
    // input_scale shipped, so no activation scales in the image and the
    // layer marked so (bf16 activations whatever the W4A4 policy).
    const bool is_mtp_layer = p.rfind("mtp.", 0) == 0;
    const bool weight_only = cfg.mtp_experts_nvfp4 && is_mtp_layer;
    if ((cfg.experts_nvfp4 && !is_mtp_layer) || weight_only) {
      m.experts_fp4.resize(static_cast<size_t>(E) * 3);
      m.expert_globals = static_cast<float*>(bump.alloc(static_cast<size_t>(E) * 3 * sizeof(float)));
      m.experts_weight_only = weight_only;
      if (!weight_only) m.act_scales = static_cast<float*>(bump.alloc(2 * sizeof(float)));
      // The scale form follows the scope: the FP4-Local export wraps the
      // backbone's expert scales as CSF and keeps the draft's plain.
      const bool mo_csf = is_mtp_layer ? cfg.mtp_experts_nvfp4_csf : cfg.experts_nvfp4_csf;
      for (int e = 0; e < E; ++e) {
        const std::string ep = p + "experts." + std::to_string(e) + ".";
        float* g = m.expert_globals + static_cast<size_t>(e) * 3;
        m.experts_fp4[static_cast<size_t>(e) * 3 + 0] = load_fp4_rows_mo(ep + "gate_proj", r * I, I, g + 0, !weight_only, mo_csf);
        m.act_scale_w13 = std::max(m.act_scale_w13, last_input_scale_);
        m.experts_fp4[static_cast<size_t>(e) * 3 + 1] = load_fp4_rows_mo(ep + "up_proj", r * I, I, g + 1, !weight_only, mo_csf);
        m.act_scale_w13 = std::max(m.act_scale_w13, last_input_scale_);
        m.experts_fp4[static_cast<size_t>(e) * 3 + 2] = load_fp4_cols_mo(ep + "down_proj", r * I, I, g + 2, !weight_only, mo_csf);
        m.act_scale_w2 = std::max(m.act_scale_w2, last_input_scale_);
      }
      // The layer's activation scales go into the image with the weights: a
      // resident-image restore lays the views out without the copy pass, so
      // host-side values would come back as 0.
      if (copy && m.act_scales) {
        const float v[2] = {m.act_scale_w13, m.act_scale_w2};
        std::memcpy(bump.host(m.act_scales), v, sizeof(v));
      }
      return;
    }
     // The RadixArk release stores the MTP layer's 512 experts as a single
     // fused BF16 pair: gate_up_proj [E, 2*I_full, H] (gate || up along the
     // intermediate axis) and down_proj [E, H, I_full]. The loader encodes
     // each expert's row/column slice to the resident FP8 form at load; the
     // resident GlmQuantMatrix is identical to the per-expert FP8 path, so
     // the moE kernels are format-agnostic about the source. The two fused
     // tensors are consumed once and note_read'd once each (counting mode
     // allocates the same from the bump with a null host_src).
     if (cfg.mtp_experts_bf16_fused && p.rfind("mtp.", 0) == 0) {
       const int64_t I_full = cfg.moe_intermediate_size, H = cfg.hidden_size;
       m.experts.resize(static_cast<size_t>(E) * 3);
       const std::string gup = p + "experts.gate_up_proj";
       const std::string dwn = p + "experts.down_proj";
       const QwenExpectedTensor& eg = expected(gup);
       const QwenExpectedTensor& ed = expected(dwn);
       const uint16_t* gup_src = nullptr;
       const uint16_t* dwn_src = nullptr;
       if (copy) {
         gup_src = static_cast<const uint16_t*>(source(gup).data);
         dwn_src = static_cast<const uint16_t*>(source(dwn).data);
       }
       for (int e = 0; e < E; ++e) {
         const size_t eo = static_cast<size_t>(e) * 2 * I_full * H;
         m.experts[static_cast<size_t>(e) * 3 + 0] =
             encode_fp8(gup_src + eo + static_cast<size_t>(r) * I * H, H, I, H);
         m.experts[static_cast<size_t>(e) * 3 + 1] =
             encode_fp8(gup_src + eo + static_cast<size_t>(I_full) * H + static_cast<size_t>(r) * I * H,
                        H, I, H);
         const size_t doff = static_cast<size_t>(e) * H * I_full + static_cast<size_t>(r) * I;
         m.experts[static_cast<size_t>(e) * 3 + 2] =
             encode_fp8(dwn_src + doff, I_full, H, I);
       }
       if (copy) {
         consumed(source(gup));
         consumed(source(dwn));
       }
       note_read(eg, static_cast<size_t>(E) * 2 * I * H * 2);
       note_read(ed, static_cast<size_t>(E) * H * I * 2);
       return;
     }
     m.experts.resize(static_cast<size_t>(E) * 3);
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      m.experts[static_cast<size_t>(e) * 3 + 0] =
          load_quant_rows(ep + "gate_proj.weight", r * I, I, geo.scale_block);
      m.experts[static_cast<size_t>(e) * 3 + 1] =
          load_quant_rows(ep + "up_proj.weight", r * I, I, geo.scale_block);
      m.experts[static_cast<size_t>(e) * 3 + 2] =
          load_quant_cols(ep + "down_proj.weight", r * I, I, geo.scale_block);
    }
  }

  // ---- NVFP4 slices in the modelopt layout (as models/glm4/loader.cpp) ------
  // `base` names the matrix ("...gate_proj"): base.weight U8 [N, K/2],
  // base.weight_scale e4m3 [N, K/16], base.weight_scale_2 F32 [] — stored
  // in `global` as its reciprocal (the kernels divide by it once), and the
  // unused base.input_scale consumed so the checkpoint reconciles.
  float last_input_scale_ = 0.0f;  // the last NVFP4 matrix's input_scale (0: not read)
  void load_global_reciprocal_into(const std::string& base, float* slot, bool with_input_scale = true) {
    const QwenExpectedTensor& eg = expected(base + ".weight_scale_2");
    if (copy) {
      const TensorInfo& t = source(eg.name);
      float ws2;
      std::memcpy(&ws2, t.data, 4);
      if (!(ws2 > 0.0f) || !std::isfinite(ws2))
        fail("'" + eg.name + "' is not a positive finite scale");
      const float inv = 1.0f / ws2;
      std::memcpy(bump.host(slot), &inv, 4);
      consumed(t);
    }
    note_read(eg, 4);
    // The activation scale (modelopt input_scale, amax / (6 * 448) from
    // calibration): kept host-side for the W4A4 prefill path, which quantizes
    // with the layer's maximum like SGLang's flashinfer_cutlass MoE.
    last_input_scale_ = 0.0f;
    if (!with_input_scale) return;  // the weight-only form ships none
    if (copy) std::memcpy(&last_input_scale_, source(base + ".input_scale").data, 4);
    (void)load_raw(base + ".input_scale");
  }
  // A CSF-wrapped NVFP4 scale plane (the caller's CSF form: config
  // experts_nvfp4_csf for the backbone, mtp_experts_nvfp4_csf for the
  // draft): the full [N, full_cols/16] E4M3 plane for `base`, expanded
  // once from its fixed + exceptions sidecars. Empty in plan mode
  // (accounting only), where the tabled tensors do not exist yet: the
  // fixed stream's size is analytic (16 row-base bytes + 8 bytes of
  // low-nibble-first codes per 16-row slab, the expander's contract —
  // 16 * (1 + cols/2) per slab), so both modes count exactly that; the
  // exceptions stream is data-dependent (a property of the quantized
  // values, not of the geometry), so it is read in copy mode but not
  // part of the source-byte plan.
  //
  // The whole plane is expanded even for a row slice (the FP4 row loader):
  // the exception words address the full matrix by flat position, so a
  // windowed expansion would have to filter and rebase them. Every rank
  // therefore expands its own copy of each expert's sidecars under tensor
  // parallelism; the disk read is one pass regardless (consumed() drops the
  // whole tensor), and the plan-vs-read invariant was validated at world 1.
  std::vector<uint8_t> load_csf_plane(const std::string& base, int64_t N, int64_t full_cols) {
    const int64_t C = full_cols / kFp4Group;
    source_bytes += static_cast<size_t>(N / 16) * static_cast<size_t>(16 + full_cols / 2);
    std::vector<uint8_t> plane;
    if (!copy) return plane;
    const TensorInfo& tf = source(base + ".weight_scale.nvfp4_csf_fixed");
    const TensorInfo& te = source(base + ".weight_scale.nvfp4_csf_exceptions");
    if (tf.dtype != DType::U8) fail("'" + tf.name + "' is not U8");
    if (te.dtype != DType::U32) fail("'" + te.name + "' is not U32");
    if (tf.data_end < tf.data_begin || te.data_end < te.data_begin)
      fail("sidecar offsets of '" + base + "' run backwards");
    const size_t ebytes = static_cast<size_t>(te.data_end - te.data_begin);
    if (ebytes % 4 != 0) fail("'" + te.name + "' is not a whole U32 count");
    plane = qwen_csf_scales_expand(static_cast<const uint8_t*>(tf.data),
                                   static_cast<size_t>(tf.data_end - tf.data_begin),
                                   static_cast<const uint32_t*>(te.data), ebytes / 4, N, C);
    consumed(tf);
    consumed(te);
    return plane;
  }
  GlmFp4Matrix load_fp4_rows_mo(const std::string& base, int64_t row_start, int64_t rows, float* global,
                                bool with_input_scale, bool csf) {
    const QwenExpectedTensor& ep = expected(base + ".weight");
    const int64_t N = ep.shape[0];
    const int64_t cols = ep.shape[1] * 2;
    fp4_check_cols(cols, who.c_str());
    check_range(base, row_start, rows, N);
    const size_t pc = static_cast<size_t>(cols / 2), sc = static_cast<size_t>(cols / kFp4Group);
    // The scale source: the tabled plain weight_scale, or the CSF sidecars
    // expanded once (the caller's csf form).
    std::vector<uint8_t> csf_plane;
    const QwenExpectedTensor* es = nullptr;
    if (csf) {
      csf_plane = load_csf_plane(base, N, cols);
    } else {
      es = &expected(base + ".weight_scale");
      if (es->shape[0] != N || es->shape[1] != cols / kFp4Group)
        fail("NVFP4 scale geometry mismatch on " + base);
    }
    GlmFp4Matrix q;
    q.rows = rows;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * sc));
    if (copy) {
      const TensorInfo& tp = source(ep.name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(tp.data) + static_cast<size_t>(row_start) * pc,
                  static_cast<size_t>(rows) * pc);
      if (csf) {
        std::memcpy(bump.host(const_cast<uint8_t*>(q.scales)),
                    csf_plane.data() + static_cast<size_t>(row_start) * sc,
                    static_cast<size_t>(rows) * sc);
      } else {
        const TensorInfo& ts = source(es->name);
        std::memcpy(bump.host(const_cast<uint8_t*>(q.scales)),
                    static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(row_start) * sc,
                    static_cast<size_t>(rows) * sc);
        consumed(ts);
      }
      consumed(tp);
    }
    note_read(ep, static_cast<size_t>(rows) * pc);
    if (!csf) note_read(*es, static_cast<size_t>(rows) * sc);
    load_global_reciprocal_into(base, global, with_input_scale);
    q.global_scale = global;
    return q;
  }
  GlmFp4Matrix load_fp4_cols_mo(const std::string& base, int64_t col_start, int64_t cols, float* global,
                                bool with_input_scale, bool csf) {
    const QwenExpectedTensor& ep = expected(base + ".weight");
    const int64_t N = ep.shape[0];
    const int64_t full = ep.shape[1] * 2;
    fp4_check_cols(full, who.c_str());
    fp4_check_cols(cols, who.c_str());
    if (col_start % kFp4Group != 0) fail("NVFP4 column slice of " + base + " is not 16-aligned");
    check_range(base, col_start, cols, full);
    std::vector<uint8_t> csf_plane;
    const QwenExpectedTensor* es = nullptr;
    if (csf) {
      csf_plane = load_csf_plane(base, N, full);
    } else {
      es = &expected(base + ".weight_scale");
      if (es->shape[0] != N || es->shape[1] != full / kFp4Group)
        fail("NVFP4 scale geometry mismatch on " + base);
    }
    const size_t pc_full = static_cast<size_t>(full / 2), sc_full = static_cast<size_t>(full / kFp4Group);
    const size_t pc = static_cast<size_t>(cols / 2), sc = static_cast<size_t>(cols / kFp4Group);
    GlmFp4Matrix q;
    q.rows = N;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(N) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(N) * sc));
    if (copy) {
      const TensorInfo& tp = source(ep.name);
      const uint8_t* sp = static_cast<const uint8_t*>(tp.data);
      const uint8_t* ss;
      const TensorInfo* scale_source = nullptr;
      if (csf) {
        ss = csf_plane.data();
      } else {
        const TensorInfo& ts = source(es->name);
        ss = static_cast<const uint8_t*>(ts.data);
        scale_source = &ts;
      }
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      uint8_t* hs = bump.host(const_cast<uint8_t*>(q.scales));
      if (cols == full) {
        std::memcpy(hp, sp, static_cast<size_t>(N) * pc);
        std::memcpy(hs, ss, static_cast<size_t>(N) * sc);
      } else {
        for (int64_t row = 0; row < N; ++row) {
          std::memcpy(hp + row * pc, sp + row * pc_full + col_start / 2, pc);
          std::memcpy(hs + row * sc, ss + row * sc_full + col_start / kFp4Group, sc);
        }
      }
      // Keep the prefetched scale pages mapped until the column copy is
      // complete. Discarding earlier refaults them and leaves them cached.
      if (scale_source) consumed(*scale_source);
      consumed(tp);
    }
    note_read(ep, static_cast<size_t>(N) * pc);
    if (!csf) note_read(*es, static_cast<size_t>(N) * sc);
    load_global_reciprocal_into(base, global, with_input_scale);
    q.global_scale = global;
    return q;
  }

  void build_ple(const std::string& p) {
    QwenPleResident& l = out.ple;
    const QwenNgramGeometry g = cfg.ngram_geometry();
    const int64_t hd = g.head_dim;
    l.hash_heads = geo.hash_heads;
    l.hash_head_begin = geo.hash_head_begin;
    l.row_begin = geo.table_row_begin;
    l.rows = geo.table_rows;
    const int64_t c0 = static_cast<int64_t>(geo.hash_head_begin) * hd;
    const int64_t cn = static_cast<int64_t>(geo.hash_heads) * hd;
    if (g_dense_weights_fp8 && !cfg.dense_fp8_shipped) {
      l.key_proj_fp8 = load_bf16_cols_fp8(p + "key_proj.weight", c0, cn);
      l.value_proj_fp8 = load_bf16_cols_fp8(p + "value_proj.weight", c0, cn);
    } else {
      l.key_proj = load_bf16_cols(p + "key_proj.weight", c0, cn);
      l.value_proj = load_bf16_cols(p + "value_proj.weight", c0, cn);
    }
    if (cfg.norms_f32) {
      l.norm_key_f32 = load_f32(p + "norm_key.weight");
      l.norm_query_f32 = load_f32(p + "norm_query.weight");
      l.norm_conv_f32 = load_f32(p + "norm_conv.weight");
    } else {
      l.norm_key = load_bf16(p + "norm_key.weight");
      l.norm_query = load_bf16(p + "norm_query.weight");
      l.norm_conv = load_bf16(p + "norm_conv.weight");
    }
    l.conv = load_bf16(p + "conv1d.weight");
    // The hash buffers ride along (replicated; the stream verified them
    // against the config at construction), and the table's scale.
    const std::string ep = p + "ple_embedding.";
    (void)load_raw(ep + "layer_multipliers");
    (void)load_raw(ep + "ngram_heads_offsets");
    (void)load_raw(ep + "ngram_heads_vocab_sizes");
    const std::string scale_name = qwen_ngram_table_scale_name(cfg, ep);
    (void)load_raw(scale_name);
    if (copy) l.table_scale = qwen_ngram_table_scale(cfg, *tensors.at(scale_name));
  }

  void build_layer(int layer) {
    const int max_layer = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    if (layer < 0 || layer >= max_layer)
      fail("layer index out of range: " + std::to_string(layer));
    const bool is_mtp = layer == cfg.mtp_layer();
    const std::string p = qwen_layer_prefix(cfg, layer);
    out.layer = layer;
    out.kind = is_mtp ? QwenLayerKind::Qsa : cfg.layers[layer];
    out.has_ple = !is_mtp && layer == cfg.ple_layer();
    if (out.has_ple) build_ple(p + "ple.");
    build_gr(p + "attn_hyper_connection.", out.attn_gr, true);
    if (out.kind == QwenLayerKind::Gdn)
      build_gdn(p + "linear_attn.");
    else
      build_qsa(p + "self_attn.");
    build_gr(p + "mlp_hyper_connection.", out.mlp_gr, true);
    build_moe(p + "mlp.");
    run_gptq_jobs();
  }
};

// ---------------------------------------------------------------------------

QwenLocalGeometry QwenLocalGeometry::from_config(const QwenTextConfig& cfg, int rank, int world,
                                                 QwenHeadSharding head) {
  if (world > 1) qwen_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("qwen loader: rank/world out of range");
  QwenLocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_key_heads = cfg.gdn_key_heads / world;
  g.local_value_heads = cfg.gdn_value_heads / world;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  if (cfg.num_key_value_heads >= world) {
    g.local_kv_heads = cfg.num_key_value_heads / world;
    g.kv_head_begin = g.local_kv_heads * rank;
  } else {
    g.local_kv_heads = 1;
    g.kv_head_begin = rank / (world / cfg.num_key_value_heads);
  }
  // The rank's query heads must belong to its kv head(s).
  const int q_per_kv = cfg.num_attention_heads / cfg.num_key_value_heads;
  if (g.head_begin / q_per_kv != g.kv_head_begin ||
      (g.head_begin + g.local_heads - 1) / q_per_kv != g.kv_head_begin + g.local_kv_heads - 1)
    throw std::invalid_argument("qwen loader: the query heads of a rank straddle kv heads");
  g.local_inter = cfg.moe_intermediate_size / world;
  g.local_shared_inter = cfg.shared_expert_intermediate_size / world;
  g.scale_block = gcd_int(128, static_cast<int>(g.local_inter));
  if (!cfg.ple_layer_ids.empty()) {
    const QwenNgramGeometry ng = cfg.ngram_geometry();
    g.hash_heads = ng.heads / world;
    g.hash_head_begin = g.hash_heads * rank;
    g.table_row_begin = ng.head_offset[static_cast<size_t>(g.hash_head_begin)];
    const int last = g.hash_head_begin + g.hash_heads;  // one past
    const int64_t end = last < ng.heads ? ng.head_offset[static_cast<size_t>(last)] : ng.total_rows;
    g.table_rows = end - g.table_row_begin;
  }
  if (head == QwenHeadSharding::VocabSharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.lm_vocab_begin = b;
    g.lm_vocab_count = n;
  } else {
    g.lm_vocab_begin = 0;
    g.lm_vocab_count = cfg.vocab_size;
  }
  return g;
}

// ---- the family hooks (loaders/resident_stream.hpp) -------------------------

void QwenLoaderFamily::validate_binding(const QwenTextConfig& cfg, const PresentMap& present) {
  const QwenBindReport rep = qwen_validate_text_binding(cfg, present);
  if (rep.ok()) return;
  std::string msg = "qwen loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

// The checkpoint's stored hash buffers must equal the config's derivation
// (docs §1.7): a disagreement means a different hash, i.e. a different
// model, and is refused before a byte of weights moves.
void QwenLoaderFamily::check_sources(const QwenTextConfig& cfg_, const LoaderTensorMap& tensors_) {
  if (cfg_.ple_layer_ids.empty()) return;
  const QwenNgramGeometry g = cfg_.ngram_geometry();
  const std::string p = qwen_layer_prefix(cfg_, cfg_.ple_layer()) + "ple.ple_embedding.";
  auto read_i64 = [&](const std::string& name, size_t n) {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("qwen loader: tensor not in checkpoint: " + name);
    const TensorInfo& t = *it->second;
    if (t.numel() != n) throw std::runtime_error("qwen loader: '" + name + "' has the wrong length");
    std::vector<int64_t> v(n);
    std::memcpy(v.data(), t.data, n * 8);
    return v;
  };
  const std::vector<int64_t> mult = read_i64(p + "layer_multipliers", static_cast<size_t>(cfg_.ngram_size));
  const std::vector<int64_t> sizes = read_i64(p + "ngram_heads_vocab_sizes", static_cast<size_t>(g.heads));
  const std::vector<int64_t> offs = read_i64(p + "ngram_heads_offsets", static_cast<size_t>(g.heads));
  if (mult != g.multipliers || sizes != g.head_vocab || offs != g.head_offset)
    throw std::runtime_error(
        "qwen loader: the checkpoint's n-gram hash buffers (layer_multipliers, "
        "ngram_heads_vocab_sizes, ngram_heads_offsets) disagree with the config's "
        "derivation — a different hash, refused");
}

bool QwenLoaderFamily::digest_included(const QwenExpectedTensor& e) { return is_replicated(e); }

// The packed column slices (a pack reads its source after the builder
// returns): the classes whose sources the one-pass load drops afterwards.
bool QwenLoaderFamily::discard_after_pack(const QwenExpectedTensor& e) {
  return e.cls == QwenWeightClass::Ple || e.cls == QwenWeightClass::Gdn ||
         e.cls == QwenWeightClass::Qsa || e.cls == QwenWeightClass::SharedExpert;
}

size_t QwenLoaderFamily::globals_bytes(const QwenTextConfig& cfg, int rank, int world,
                                       LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t W = static_cast<size_t>(cfg.hyper_width());
  const size_t r = static_cast<size_t>(cfg.hc_lowrank);
  // A dense [n, k] matrix's resident bytes: BF16, or block FP8 (codes + the
  // fp32 scale grid) under engine.dense_weights = "fp8".
  const auto dense = [&cfg](size_t n, size_t k) -> size_t {
    if (!g_dense_weights_fp8 || cfg.dense_fp8_shipped) return align_up_256(n * k * 2);
    return align_up_256(n * k) +
           align_up_256(static_cast<size_t>(fp8_quant::scale_rows(static_cast<int64_t>(n))) *
                        static_cast<size_t>(fp8_quant::scale_cols(static_cast<int64_t>(k))) * 4);
  };
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(cfg.vocab_size) * H * 2);                  // embed
  const size_t head_rows = static_cast<size_t>(QwenLocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count);
  if (cfg.lm_head_gptq_int8) {  // the hybrid's int8 g128 head: packed words + f16 scales
    b += align_up_256(head_rows * H) + align_up_256(head_rows * (H / static_cast<size_t>(cfg.gptq_group)) * 2);
    if (!g_draft_vocab_ids.empty()) {  // the draft vocabulary slice: rows, scales, ids
      const size_t n = g_draft_vocab_ids.size();
      b += align_up_256(n * H) + align_up_256(n * (H / static_cast<size_t>(cfg.gptq_group)) * 2) + align_up_256(n * 4);
    }
  } else {
    b += dense(head_rows, H);
  }
  b += align_up_256(W * (cfg.norms_f32 ? 4 : 2)) + dense(r, W) + dense(W, r);      // mixer (its norm F32 on the QAD export)
  if (cfg.mtp_layer() >= 0) {
    b += 2 * align_up_256(H * H * 2);                          // fc_embedding, fc_hidden
    b += align_up_256(H * 2) + align_up_256(W * 2);            // pre_fc norms
    b += align_up_256(W * 2) + dense(r, W) + dense(W, r);                        // mtp mixer
  }
  return b;
}

size_t QwenLoaderFamily::extra_resident_bytes(const QwenTextConfig& cfg, int rank, int world) {
  return QwenLayerStream::ngram_table_bytes(cfg, rank, world);
}

// The restored PLE layer's host-side scale: from the source when mapped.
void QwenLoaderFamily::after_restore(const QwenTextConfig& cfg, int layer,
                                     const LoaderTensorMap& tensors, QwenLayerResident& out) {
  if (!out.has_ple || tensors.empty()) return;
  const std::string name = qwen_ngram_table_scale_name(cfg, qwen_layer_prefix(cfg, layer) + "ple.ple_embedding.");
  out.ple.table_scale = qwen_ngram_table_scale(cfg, *tensors.at(name));
}

void QwenLoaderFamily::build_globals(const QwenTextConfig& cfg_, const QwenLocalGeometry& geo_,
                                     const LoaderTensorMap& tensors_, LayerBump& bump,
                                     QwenGlobalsResident& globals_, uint64_t& source_bytes_,
                                     uint64_t& verbatim_bytes_, LoaderHeadSharding head_) {
  LayerBump* globals_bump_ = &bump;
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("qwen loader: global tensor missing: " + name);
    return *it->second;
  };
  auto copy_global = [&](const std::string& name) -> uint16_t* {
    const TensorInfo& t = lookup(name);
    uint16_t* dst = static_cast<uint16_t*>(globals_bump_->alloc(t.nbytes()));
    std::memcpy(globals_bump_->host(dst), t.data, t.nbytes());
    source_bytes_ += t.nbytes();
    verbatim_bytes_ += t.nbytes();
    return dst;
  };
  // A BF16 global encoded to block FP8 into the globals bump (dense_weights fp8).
  auto encode_global_fp8 = [&](const std::string& name) -> GlmQuantMatrix {
    const TensorInfo& t = lookup(name);
    if (t.shape.size() != 2) throw std::runtime_error("qwen loader: '" + name + "' is not a matrix");
    const int64_t rows = t.shape[0], cols = t.shape[1];
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(globals_bump_->alloc(static_cast<size_t>(rows) * cols));
    q.scales = static_cast<const float*>(globals_bump_->alloc(
        static_cast<size_t>(fp8_quant::scale_rows(rows)) * fp8_quant::scale_cols(cols) * 4));
    fp8_quant::encode_block128(static_cast<const uint16_t*>(t.data), static_cast<size_t>(cols), rows, cols,
                               globals_bump_->host(const_cast<uint8_t*>(q.payload)),
                               globals_bump_->host(const_cast<float*>(q.scales)));
    source_bytes_ += t.nbytes();
    return q;
  };
  auto copy_gr = [&](const std::string& p, QwenGrResident& g, bool norm_f32) {
    // copy_global moves the tensor's bytes whatever their dtype; the F32
    // norm (the QAD export's model mixer) lands in its own pointer.
    if (norm_f32)
      g.hc_norm_f32 = reinterpret_cast<const float*>(copy_global(p + "hc_norm.weight"));
    else
      g.hc_norm = copy_global(p + "hc_norm.weight");
    if (g_dense_weights_fp8 && !cfg_.dense_fp8_shipped) {
      g.down_fp8 = encode_global_fp8(p + "input_mix_weight_down.weight");
      g.up_fp8 = encode_global_fp8(p + "input_mix_weight_up.weight");
    } else {
      g.down = copy_global(p + "input_mix_weight_down.weight");
      g.up = copy_global(p + "input_mix_weight_up.weight");
    }
    g.inject = nullptr;
  };
  globals_.embed = copy_global("model.language_model.embed_tokens.weight");
  if (cfg_.lm_head_gptq_int8) {
    // The hybrid's int8 g128 head: the vocab slice's columns of the GPTQ
    // triple transposed into packed rows (loaders/gptq_repack.hpp), the
    // scales untouched, the zeros verified. Every boot (the globals are
    // not in the resident image); 16 threads over the columns.
    const TensorInfo& tw = lookup("lm_head.qweight");
    const TensorInfo& ts = lookup("lm_head.scales");
    const TensorInfo& tz = lookup("lm_head.qzeros");
    const int64_t H = cfg_.hidden_size, V = cfg_.vocab_size, g = cfg_.gptq_group;
    constexpr int bits = 8, per = 4;
    if (tw.shape != std::vector<int64_t>{H / per, V} || ts.shape != std::vector<int64_t>{H / g, V} ||
        tz.shape != std::vector<int64_t>{H / g, V / per})
      throw std::runtime_error("qwen loader: the lm_head GPTQ triple has the wrong geometry");
    const int begin = geo_.lm_vocab_begin, count = geo_.lm_vocab_count;
    GlmPackedMatrix m;
    m.rows = count;
    m.cols = H;
    m.bits = bits;
    m.scale_fmt = kPackedScaleF16G128;
    const size_t word_rows = static_cast<size_t>(H / per), groups = static_cast<size_t>(H / g);
    m.packed = static_cast<const uint32_t*>(globals_bump_->alloc(static_cast<size_t>(count) * word_rows * 4));
    m.scales = static_cast<const uint16_t*>(globals_bump_->alloc(static_cast<size_t>(count) * groups * 2));
    GptqRepackJob j;
    j.qweight = static_cast<const uint32_t*>(tw.data);
    j.qw_stride = V;
    j.word_row0 = 0;
    j.word_rows = static_cast<int64_t>(word_rows);
    j.n0 = begin;
    j.n = count;
    j.scales = static_cast<const uint16_t*>(ts.data);
    j.sc_stride = V;
    j.group0 = 0;
    j.groups = static_cast<int64_t>(groups);
    j.qzeros = static_cast<const uint32_t*>(tz.data);
    j.qz_stride = V / per;
    j.bits = bits;
    j.dst_words = globals_bump_->host(const_cast<uint32_t*>(m.packed));
    j.dst_scales = globals_bump_->host(const_cast<uint16_t*>(m.scales));
    if (gptq_repack_all(gptq_repack_split(j, 16), 16) >= 0)
      throw std::runtime_error(
          "qwen loader: 'lm_head.qzeros' holds a word that is not the symmetric constant "
          "(the packed core's fixed offset needs zero = 128)");
    // The bit-plane layout (kernels/packq_head.hpp, 2026-09-29): the
    // argmax-only head reads three of every four sectors. Default on;
    // DGPP_QWEN_HEAD_PLANES=off keeps the row layout (the A/B knob).
    if (!g_draft_vocab_ids.empty()) {
      // The draft's slice: the set's rows of the row-layout head (and their
      // scales) into their own matrix, in the plane layout, plus the ids.
      const std::vector<int32_t>& ids = g_draft_vocab_ids;
      for (int32_t id : ids)
        if (id < begin || id >= begin + count)
          throw std::runtime_error("qwen loader: engine.draft_vocab holds an id outside this rank's vocab slice");
      const size_t n_slice = ids.size();
      GlmPackedMatrix d = m;
      d.rows = static_cast<int64_t>(n_slice);
      d.layout = kPackedLayoutPlanes8;
      d.packed = static_cast<const uint32_t*>(globals_bump_->alloc(n_slice * static_cast<size_t>(H)));
      d.scales = static_cast<const uint16_t*>(globals_bump_->alloc(n_slice * groups * 2));
      const int32_t* dev_ids = static_cast<const int32_t*>(globals_bump_->alloc(n_slice * 4));
      uint8_t* dst = reinterpret_cast<uint8_t*>(globals_bump_->host(const_cast<uint32_t*>(d.packed)));
      uint16_t* dsc = globals_bump_->host(const_cast<uint16_t*>(d.scales));
      int32_t* hid = globals_bump_->host(const_cast<int32_t*>(dev_ids));
      const uint8_t* src = reinterpret_cast<const uint8_t*>(j.dst_words);
      const uint16_t* ssc = j.dst_scales;
      for (size_t r = 0; r < n_slice; ++r) {
        const size_t row = static_cast<size_t>(ids[r] - begin);
        std::memcpy(dst + r * static_cast<size_t>(H), src + row * static_cast<size_t>(H), static_cast<size_t>(H));
        std::memcpy(dsc + r * groups, ssc + row * groups, groups * 2);
        hid[r] = ids[r];
      }
      packq_planes_permute_int8(dst, static_cast<int64_t>(n_slice), H);
      globals_.draft_head_packed = d;
      globals_.draft_vocab_ids = dev_ids;
      globals_.draft_vocab_count = static_cast<int>(n_slice);
      DGPP_LOG_INFO("qwen loader: draft vocabulary slice: {} of {} head rows ({:.0f} MiB)", n_slice,
                    count, static_cast<double>(n_slice * (static_cast<size_t>(H) + groups * 2)) / (1 << 20));
    }
    {
      const char* e = std::getenv("DGPP_QWEN_HEAD_PLANES");
      const bool planes = !(e != nullptr && (std::string(e) == "off" || std::string(e) == "0"));
      if (planes) {
        packq_planes_permute_int8(reinterpret_cast<uint8_t*>(j.dst_words), count, H);
        m.layout = kPackedLayoutPlanes8;
      }
    }
    source_bytes_ += static_cast<size_t>(count) * word_rows * 4 + static_cast<size_t>(count) * groups * 2 +
                     groups * (static_cast<size_t>(count) / per) * 4;
    globals_.lm_head_packed = m;
    globals_.lm_vocab_begin = begin;
    globals_.lm_vocab_count = count;
  } else {
    const TensorInfo& t = lookup("lm_head.weight");
    const size_t row_bytes = static_cast<size_t>(cfg_.hidden_size) * 2;
    const int begin = geo_.lm_vocab_begin, count = geo_.lm_vocab_count;
    if (g_dense_weights_fp8) {
      const int64_t H = cfg_.hidden_size;
      GlmQuantMatrix q;
      q.rows = count;
      q.cols = H;
      q.payload = static_cast<const uint8_t*>(globals_bump_->alloc(static_cast<size_t>(count) * H));
      q.scales = static_cast<const float*>(globals_bump_->alloc(
          static_cast<size_t>(fp8_quant::scale_rows(count)) * fp8_quant::scale_cols(H) * 4));
      fp8_quant::encode_block128(static_cast<const uint16_t*>(t.data) + static_cast<size_t>(begin) * H,
                                 static_cast<size_t>(H), count, H,
                                 globals_bump_->host(const_cast<uint8_t*>(q.payload)),
                                 globals_bump_->host(const_cast<float*>(q.scales)));
      globals_.lm_head_fp8 = q;
    } else {
      uint16_t* dst = static_cast<uint16_t*>(globals_bump_->alloc(static_cast<size_t>(count) * row_bytes));
      std::memcpy(globals_bump_->host(dst),
                  static_cast<const uint8_t*>(t.data) + static_cast<size_t>(begin) * row_bytes,
                  static_cast<size_t>(count) * row_bytes);
      if (head_ == LoaderHeadSharding::Full) verbatim_bytes_ += static_cast<size_t>(count) * row_bytes;
      globals_.lm_head = dst;
    }
    source_bytes_ += static_cast<size_t>(count) * row_bytes;
    globals_.lm_vocab_begin = begin;
    globals_.lm_vocab_count = count;
  }
  copy_gr("model.language_model.hyper_connection_mixer.", globals_.mixer, cfg_.norms_f32);
  if (cfg_.mtp_layer() >= 0) {
    globals_.mtp_fc_embedding = copy_global("mtp.fc_embedding.weight");
    globals_.mtp_fc_hidden = copy_global("mtp.fc_hidden.weight");
    globals_.mtp_pre_fc_norm_embedding = copy_global("mtp.pre_fc_norm_embedding.weight");
    globals_.mtp_pre_fc_norm_hidden = copy_global("mtp.pre_fc_norm_hidden.weight");
    copy_gr("mtp.hyper_connection_mixer.", globals_.mtp_mixer, /*norm_f32=*/false);  // the draft's norms stay BF16
  }
}

template class ResidentLayerStream<QwenLoaderFamily>;

// ---------------------------------------------------------------------------

QwenLayerStream::QwenLayerStream(const QwenTextConfig& cfg, const std::string& checkpoint_dir,
                                 int rank, int world, QwenResidency residency,
                                 QwenHeadSharding head, bool resident_mtp)
    : ResidentLayerStream<QwenLoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                            resident_mtp) {
  open_resident_image();
}

QwenLayerStream::~QwenLayerStream() {
  if (table_device_) cudaFree(table_device_);
}

size_t QwenLayerStream::ngram_table_bytes(const QwenTextConfig& cfg, int rank, int world) {
  if (g_ngram_table_mmap) return 0;
  if (cfg.ple_layer_ids.empty()) return 0;
  const QwenLocalGeometry g = QwenLocalGeometry::from_config(cfg, rank, world, QwenHeadSharding::Full);
  return static_cast<size_t>(g.table_rows) * qwen_ngram_row_bytes(cfg);
}

void QwenLayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage() = dir;
}
const std::string& QwenLayerStream::resident_image_dir() { return resident_image_dir_storage(); }

// ---- the n-gram table ----------------------------------------------------------

void QwenLayerStream::set_ngram_table_mmap(bool on) { g_ngram_table_mmap = on; }
bool QwenLayerStream::ngram_table_mmap() { return g_ngram_table_mmap; }
void QwenLayerStream::set_ngram_table_dir(const std::string& dir) { ngram_table_dir_storage() = dir; }
const std::string& QwenLayerStream::ngram_table_dir() { return ngram_table_dir_storage(); }
std::string QwenLoaderFamily::extra_shard_dir() { return ngram_table_dir_storage(); }
bool QwenLoaderFamily::admit_extra_tensor(const std::string& name) {
  return name.find("ple.ple_embedding.ngram_embedding.") != std::string::npos;
}

// ---- the dense stack's form (engine.dense_weights) ------------------------------

void QwenLayerStream::set_dense_weights_fp8(bool on) { g_dense_weights_fp8 = on; }
bool QwenLayerStream::set_draft_vocab(const std::string& npy_path, std::string* err) {
  std::vector<int32_t> ids;
  std::string e;
  if (!read_npy_ids(npy_path, &ids, &e)) {
    if (err) *err = e;
    return false;
  }
  g_draft_vocab_ids = std::move(ids);
  return true;
}
int QwenLayerStream::draft_vocab_count() { return static_cast<int>(g_draft_vocab_ids.size()); }
const std::vector<int32_t>& QwenLayerStream::draft_vocab_ids() { return g_draft_vocab_ids; }
bool QwenLayerStream::dense_weights_fp8() { return g_dense_weights_fp8; }
namespace {
bool g_prefill_fp8_gemm = false;
}
void QwenLayerStream::set_prefill_fp8_gemm(bool on) { g_prefill_fp8_gemm = on; }
bool QwenLayerStream::prefill_fp8_gemm() { return g_prefill_fp8_gemm; }
namespace {
bool g_ngram_prestage = true;
}
void QwenLayerStream::set_ngram_prestage(bool on) { g_ngram_prestage = on; }
bool QwenLayerStream::ngram_prestage() { return g_ngram_prestage; }
// Bit 8: the NVFP4 experts' activation scales live in the layer image (a
// resident image written without them is rebuilt, not misread).
// The QAD export's forms need no bit of their own: the image key already
// folds config.json and every shard header, which differ for that
// checkpoint, and the layout of the existing checkpoints' images is
// unchanged (nothing is recaptured).
uint64_t QwenLoaderFamily::loader_format() {
  return (g_dense_weights_fp8 ? 2 : 1) | (g_mtp_experts_bf16_fused ? 4 : 0) | 8;
}

void QwenLayerStream::set_mtp_expert_format(bool on) { g_mtp_experts_bf16_fused = on; }
bool QwenLayerStream::mtp_experts_bf16_fused() { return g_mtp_experts_bf16_fused; }

// ---- QwenNgramTableMmap ---------------------------------------------------------

void QwenNgramTableMmap::map_plane(Plane& plane, std::unordered_map<std::string, size_t>& by_path,
                                   const char* what) {
  plane.part_base.resize(plane.parts.size(), nullptr);
  for (size_t s = 0; s < plane.parts.size(); ++s) {
    const Part& part = plane.parts[s];
    auto it = by_path.find(part.path);
    if (it == by_path.end()) {
      Mapping m;
      m.fd = ::open(part.path.c_str(), O_RDONLY | O_CLOEXEC);
      if (m.fd < 0) throw std::runtime_error("QwenNgramTableMmap: cannot open " + part.path);
      struct stat st {};
      if (fstat(m.fd, &st) != 0) { ::close(m.fd); throw std::runtime_error("QwenNgramTableMmap: fstat " + part.path); }
      m.len = static_cast<size_t>(st.st_size);
      void* map = mmap(nullptr, m.len, PROT_READ, MAP_SHARED, m.fd, 0);
      if (map == MAP_FAILED) { ::close(m.fd); throw std::runtime_error("QwenNgramTableMmap: mmap " + part.path); }
      m.base = static_cast<uint8_t*>(map);
      // Rows are read one record at a time from anywhere in tens of GB:
      // no readahead, or every fault pulls 128 KB for 160 bytes.
      madvise(m.base, m.len, MADV_RANDOM);
      mapped_bytes_ += m.len;
      maps_.push_back(m);
      it = by_path.emplace(part.path, maps_.size() - 1).first;
    }
    const Mapping& m = maps_[it->second];
    if (part.data_begin + static_cast<uint64_t>(part.rows) * static_cast<uint64_t>(plane.row_bytes) > m.len)
      throw std::runtime_error(std::string("QwenNgramTableMmap: ") + what + " part " + std::to_string(s) +
                               " past the end of " + part.path);
    plane.part_base[s] = m.base + part.data_begin;
  }
}

QwenNgramTableMmap::QwenNgramTableMmap(std::vector<Part> parts, int64_t capacity, int row_bytes,
                                       std::vector<Part> scale_parts, int scale_row_bytes)
    : capacity_(capacity), row_bytes_(row_bytes), scale_row_bytes_(scale_row_bytes) {
  if (capacity_ <= 0 || row_bytes_ <= 0 || parts.empty())
    throw std::invalid_argument("QwenNgramTableMmap: empty geometry");
  if ((scale_row_bytes_ > 0) != !scale_parts.empty() || (scale_row_bytes_ > 0 && scale_parts.size() != parts.size()))
    throw std::invalid_argument("QwenNgramTableMmap: the scale plane must mirror the code parts");
  for (size_t i = 0; i < scale_parts.size(); ++i)
    if (scale_parts[i].rows != parts[i].rows)
      throw std::invalid_argument("QwenNgramTableMmap: a scale part's rows differ from its codes'");
  codes_.parts = std::move(parts);
  codes_.row_bytes = row_bytes_;
  scales_.parts = std::move(scale_parts);
  scales_.row_bytes = scale_row_bytes_;
  std::unordered_map<std::string, size_t> by_path;
  map_plane(codes_, by_path, "codes");
  if (scale_row_bytes_ > 0) map_plane(scales_, by_path, "scales");
}

QwenNgramTableMmap::~QwenNgramTableMmap() {
  for (Mapping& m : maps_) {
    if (m.base) munmap(m.base, m.len);
    if (m.fd >= 0) ::close(m.fd);
  }
}

const uint8_t* QwenNgramTableMmap::plane_row(const Plane& plane, int64_t row) const {
  const int64_t s = row / capacity_, r = row - s * capacity_;
  if (row < 0 || s >= static_cast<int64_t>(plane.parts.size()) || r >= plane.parts[static_cast<size_t>(s)].rows)
    throw std::out_of_range("QwenNgramTableMmap: row " + std::to_string(row) + " outside the table");
  return plane.part_base[static_cast<size_t>(s)] + static_cast<size_t>(r) * static_cast<size_t>(plane.row_bytes);
}

const uint8_t* QwenNgramTableMmap::row(int64_t row) const { return plane_row(codes_, row); }
const uint8_t* QwenNgramTableMmap::scale_row(int64_t row) const {
  return scale_row_bytes_ > 0 ? plane_row(scales_, row) : nullptr;
}

void QwenNgramTableMmap::gather(const int32_t* ids, int n, int heads, int head_begin, int heads_local,
                                uint8_t* dst, uint8_t* dst_scales) const {
  if (n <= 0) return;
  if (scale_row_bytes_ > 0 && dst_scales == nullptr)
    throw std::invalid_argument("QwenNgramTableMmap: the NVFP4 table's gather needs a scale staging");
  const int64_t total = static_cast<int64_t>(n) * heads_local;
  const size_t rb = static_cast<size_t>(row_bytes_), sb = static_cast<size_t>(scale_row_bytes_);
  // Every row's page asked for up front (the kernel issues the reads in
  // parallel), then the copies: a decode step's 32 rows in one thread, a
  // prefill chunk's tens of thousands across a few. The NVFP4 table's
  // scale row rides along (another 10 bytes; its own page, usually).
  auto copy_range = [&](int64_t lo, int64_t hi) {
    for (int64_t pair = lo; pair < hi; ++pair) {
      const int64_t t = pair / heads_local, hl = pair - t * heads_local;
      const int32_t id = ids[t * heads + head_begin + hl];
      std::memcpy(dst + static_cast<size_t>(pair) * rb, row(id), rb);
      if (sb) std::memcpy(dst_scales + static_cast<size_t>(pair) * sb, scale_row(id), sb);
    }
  };
  static const bool timing = [] {
    const char* v = std::getenv("DGPP_QWEN_PLE_STAGE_TIMING");
    return v && *v && std::string(v) != "0";
  }();
  const auto t0 = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  // The advice for every row's page in ONE syscall (process_madvise on our
  // own pidfd; 2026-09-29: the per-row madvise loop was 249 us of a 440 us
  // decode gather — 64 syscalls at ~4 us). The kernel issues the reads
  // asynchronously; the copies below then fault only on what is not yet
  // resident. Falls back to the per-row loop if the batched call is refused.
  static const int pidfd = static_cast<int>(syscall(SYS_pidfd_open, getpid(), 0));
  static std::atomic<bool> batched_ok{pidfd >= 0};
  bool advised = false;
  // A prefill chunk's tens of thousands of rows batch the same way (2026-09-29:
  // the per-row loop below was the 56 ms stage wait of every 4096-token chunk
  // in the 32K profile — 4 us a syscall).
  if (batched_ok.load(std::memory_order_relaxed)) {
    const int planes = sb ? 2 : 1;
    std::vector<iovec> iov(static_cast<size_t>(total) * planes);
    for (int64_t pair = 0; pair < total; ++pair) {
      const int64_t t = pair / heads_local, hl = pair - t * heads_local;
      const int32_t id = ids[t * heads + head_begin + hl];
      const uint8_t* p = row(id);
      const uintptr_t page = reinterpret_cast<uintptr_t>(p) & ~uintptr_t{4095};
      iov[static_cast<size_t>(pair) * planes].iov_base = reinterpret_cast<void*>(page);
      iov[static_cast<size_t>(pair) * planes].iov_len = 4096 + rb;
      if (sb) {
        const uintptr_t spage = reinterpret_cast<uintptr_t>(scale_row(id)) & ~uintptr_t{4095};
        iov[static_cast<size_t>(pair) * planes + 1].iov_base = reinterpret_cast<void*>(spage);
        iov[static_cast<size_t>(pair) * planes + 1].iov_len = 4096 + sb;
      }
    }
    // One call covers up to IOV_MAX (1024) ranges.
    advised = true;
    for (size_t off = 0; off < iov.size(); off += 1024) {
      const size_t n = std::min<size_t>(1024, iov.size() - off);
      const long r = syscall(SYS_process_madvise, pidfd, iov.data() + off, n, MADV_WILLNEED, 0);
      if (r < 0) { advised = false; batched_ok.store(false, std::memory_order_relaxed); break; }
    }
  }
  if (!advised) {
    for (int64_t pair = 0; pair < total; ++pair) {
      const int64_t t = pair / heads_local, hl = pair - t * heads_local;
      const int32_t id = ids[t * heads + head_begin + hl];
      const uint8_t* p = row(id);
      const uintptr_t page = reinterpret_cast<uintptr_t>(p) & ~uintptr_t{4095};
      madvise(reinterpret_cast<void*>(page), 4096 + rb, MADV_WILLNEED);
      if (sb) {
        const uintptr_t spage = reinterpret_cast<uintptr_t>(scale_row(id)) & ~uintptr_t{4095};
        madvise(reinterpret_cast<void*>(spage), 4096 + sb, MADV_WILLNEED);
      }
    }
  }
  const auto t1 = timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  constexpr int64_t kPerThread = 256;
  if (total <= kPerThread) {
    copy_range(0, total);
    if (timing) {
      // The two phases of a decode gather, every 200th call: the advice
      // loop (one madvise per row) and the copies (the faults land here).
      static std::atomic<long long> calls{0}, adv_us{0}, copy_us{0};
      const auto t2 = std::chrono::steady_clock::now();
      const long long n = calls.fetch_add(1) + 1;
      adv_us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
      copy_us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
      if (n % 200 == 0)
        DGPP_LOG_INFO("qwen ple: gather phases x{} — advise {} us, copy {} us (means)", n, adv_us.load() / n,
                      copy_us.load() / n);
    }
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

const QwenNgramTableResident& QwenLayerStream::load_ngram_table() {
  if (table_.rows_e4m3 || table_.rows_fp4 || mmap_table_ || cfg_.ple_layer_ids.empty()) return table_;
  if (sources_released_)
    throw std::runtime_error("qwen loader: load_ngram_table after the checkpoint sources were released");
  const QwenNgramGeometry g = cfg_.ngram_geometry();
  const int64_t hd = g.head_dim;
  const int64_t row_begin = geo_.table_row_begin, rows = geo_.table_rows;
  const bool fp4 = cfg_.ngram_table_nvfp4;
  // The stored row: head_dim e4m3 codes, or the NVFP4 table's head_dim / 2
  // code bytes with a separate head_dim / 16 scale tensor per shard.
  const size_t code_bytes = fp4 ? static_cast<size_t>(hd / 2) : static_cast<size_t>(hd);
  const size_t scale_bytes = fp4 ? static_cast<size_t>(hd / kQwenNvfp4Group) : 0;
  const std::string p = qwen_layer_prefix(cfg_, cfg_.ple_layer()) + "ple.ple_embedding.";
  auto tensor = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second) throw std::runtime_error("qwen loader: tensor not in checkpoint: " + name);
    return *it->second;
  };
  table_.fp4 = fp4;
  table_.scale = qwen_ngram_table_scale(cfg_, tensor(qwen_ngram_table_scale_name(cfg_, p)));
  table_.row_begin = row_begin;
  table_.rows = rows;
  table_.head_dim = static_cast<int>(hd);
  if (g_ngram_table_mmap) {
    // The table stays in its shard(s): the parts' file offsets, mapped by
    // the table object; the resident view keeps the geometry and the scale
    // (the staged gather's), the device pointers null.
    std::vector<QwenNgramTableMmap::Part> parts, scale_parts;
    for (int s = 0; s < cfg_.split_ngram_parts; ++s) {
      const std::string sp = p + "ngram_embedding.shard_" + std::to_string(s);
      const TensorInfo& t = tensor(sp + ".weight");
      if (!t.owner) throw std::runtime_error("qwen loader: tensor not in checkpoint: " + sp + ".weight");
      parts.push_back(QwenNgramTableMmap::Part{t.owner->path(), t.data_begin, t.shape[0]});
      if (fp4) {
        const TensorInfo& ts = tensor(sp + ".weight_scale");
        if (!ts.owner) throw std::runtime_error("qwen loader: tensor not in checkpoint: " + sp + ".weight_scale");
        scale_parts.push_back(QwenNgramTableMmap::Part{ts.owner->path(), ts.data_begin, ts.shape[0]});
      }
    }
    mmap_table_ = std::make_unique<QwenNgramTableMmap>(std::move(parts), qwen_ngram_shard_capacity(cfg_),
                                                       static_cast<int>(code_bytes), std::move(scale_parts),
                                                       static_cast<int>(scale_bytes));
    table_.rows_e4m3 = nullptr;
    table_.rows_fp4 = nullptr;
    table_.rows_fp4_scales = nullptr;
    table_.mmap = mmap_table_.get();
    table_.bytes = 0;
    DGPP_LOG_INFO("qwen loader: rank {} n-gram table rows [{}, {}) mmap'ed from the checkpoint ({:.2f} GiB mapped, "
                  "{}, scale {})",
                  rank_, row_begin, row_begin + rows, mmap_table_->mapped_bytes() / (1024.0 * 1024.0 * 1024.0),
                  fp4 ? "NVFP4 codes + scales" : "e4m3", table_.scale);
    return table_;
  }
  const size_t bytes = static_cast<size_t>(rows) * (code_bytes + scale_bytes);
  DGPP_CUDA_OK(cudaMalloc(&table_device_, bytes));
  const int64_t cap = qwen_ngram_shard_capacity(cfg_);
  const int64_t row_end = row_begin + rows;
  uint8_t* dst_codes = static_cast<uint8_t*>(table_device_);
  uint8_t* dst_scales = dst_codes + static_cast<size_t>(rows) * code_bytes;  // the scale plane after the codes
  size_t done = 0;
  // One tensor's row range [lo, hi) of shard s into dst (row_bytes a row)
  // through the staging mirror.
  auto copy_rows = [&](const TensorInfo& t, int64_t s_begin, int64_t lo, int64_t hi, size_t row_bytes, uint8_t* dst) {
    if (t.owner) t.owner->prefetch(t);
    const uint8_t* src = static_cast<const uint8_t*>(t.data) + static_cast<size_t>(lo - s_begin) * row_bytes;
    size_t remaining = static_cast<size_t>(hi - lo) * row_bytes;
    uint8_t* d = dst + static_cast<size_t>(lo - row_begin) * row_bytes;
    while (remaining > 0) {
      const size_t chunk = std::min(remaining, staging_bytes_);
      std::memcpy(staging_, src, chunk);
      DGPP_CUDA_OK(cudaMemcpyAsync(d, staging_, chunk, cudaMemcpyHostToDevice, stream_));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      src += chunk;
      d += chunk;
      done += chunk;
      remaining -= chunk;
    }
    if (residency_ == QwenResidency::Resident && t.owner) t.owner->discard(t);
  };
  for (int s = 0; s < cfg_.split_ngram_parts; ++s) {
    const int64_t s_begin = static_cast<int64_t>(s) * cap;
    const int64_t s_end = s_begin + qwen_ngram_shard_rows(cfg_, s);
    const int64_t lo = std::max(row_begin, s_begin), hi = std::min(row_end, s_end);
    if (lo >= hi) continue;
    const std::string sp = p + "ngram_embedding.shard_" + std::to_string(s);
    copy_rows(tensor(sp + ".weight"), s_begin, lo, hi, code_bytes, dst_codes);
    if (fp4) copy_rows(tensor(sp + ".weight_scale"), s_begin, lo, hi, scale_bytes, dst_scales);
  }
  if (done != bytes)
    throw std::runtime_error("qwen loader: the n-gram shards did not cover this rank's rows");
  source_bytes_ += bytes;
  if (fp4) {
    table_.rows_fp4 = dst_codes;
    table_.rows_fp4_scales = dst_scales;
  } else {
    table_.rows_e4m3 = dst_codes;
  }
  table_.bytes = bytes;
  DGPP_LOG_INFO("qwen loader: rank {} n-gram table rows [{}, {}) resident ({:.2f} GiB, {}, scale {})",
                rank_, row_begin, row_end, bytes / (1024.0 * 1024.0 * 1024.0), fp4 ? "NVFP4 codes + scales" : "e4m3",
                table_.scale);
  return table_;
}

// ---- sources and digests --------------------------------------------------------

}  // namespace dgpp
