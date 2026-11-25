// The Inkling 2 row kernels against host statements (docs/inkling2_mini_plan.md G2).
//
//   quantizers   the NVFP4 four-over-six and MXFP8 activation quantizers: the device kernels
//                equal their host statements bit for bit over rows of every magnitude, and
//                --dump-quant FILE writes the rows and results for the numpy reference
//                (tools/inkling2_reference_dump.py check-quant)
//   convolution  one launch over a run, the same run split across launches and one row at a
//                time give the same bits (the ring carries the earlier positions)
//   snapshot     a ring restored from a prefix snapshot continues with the same bits
//   mix          the mix with its norm against a double-precision statement
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/device_copy.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/inkling2.hpp"

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  float unit() { return static_cast<float>(static_cast<double>(next() >> 11) / 9007199254740992.0 * 2.0 - 1.0); }
  float normal3() { return (unit() + unit() + unit()) * (1.0f / 3.0f); }
};

template <class T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t count) : n(count) {
    DGPP_CUDA_OK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    DGPP_CUDA_OK(cudaMemset(p, 0, std::max<size_t>(count, 1) * sizeof(T)));
  }
  explicit Dev(const std::vector<T>& host) : Dev(host.size()) { upload(host); }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  void upload(const std::vector<T>& host) {
    require(host.size() <= n, "upload: too large");
    DGPP_CUDA_OK(cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> download() const {
    std::vector<T> host(n);
    DGPP_CUDA_OK(cudaMemcpy(host.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
  }
};

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

// ---- the quantizers -----------------------------------------------------------------

std::vector<uint16_t> quantizer_rows(int rows, int k) {
  std::vector<uint16_t> x(static_cast<size_t>(rows) * k);
  Rng rng(11);
  for (int r = 0; r < rows; ++r) {
    // Rows from 2^-10 to 2^10, blocks inside a row spread over another 2^6.
    const float row_scale = std::exp2(10.0f * rng.unit());
    for (int c = 0; c < k; ++c) {
      const float block_scale = std::exp2(3.0f * static_cast<float>(((c / 16) * 7 + r) % 5 - 2));
      float v = row_scale * block_scale * rng.normal3();
      if (r == 0) v = 0.0f;                                  // an all-zero row
      if (r == 1) v = c == 37 ? 900.0f : 1e-3f * rng.unit(); // one outlier
      if (r == 2 && (c / 16) % 2 == 0) v = 0.0f;             // zero blocks
      if (r == 3) v = (c % 2 ? 6.0f : -4.0f);                // values on the grid
      if (r == 4) v = 1e-30f * rng.unit();                   // near the bottom of the format
      x[static_cast<size_t>(r) * k + c] = float_to_bf16_bits(v);
    }
  }
  return x;
}

void test_quantizers(const std::string& dump_path) {
  const int rows = 96, k = 256;
  const std::vector<uint16_t> x = quantizer_rows(rows, k);
  std::vector<uint16_t> host_inner(x.size()), host_mx(x.size());
  std::vector<float> host_outer(static_cast<size_t>(rows));
  for (int r = 0; r < rows; ++r) {
    dgpp::ink2_quantize_nvfp4_host(x.data() + static_cast<size_t>(r) * k, k, host_inner.data() + static_cast<size_t>(r) * k,
                                   &host_outer[static_cast<size_t>(r)]);
    dgpp::ink2_requantize_mxfp8_host(x.data() + static_cast<size_t>(r) * k, k, host_mx.data() + static_cast<size_t>(r) * k);
  }
  Dev<uint16_t> dx(x), dinner(x.size()), dmx(x.size());
  Dev<float> douter(static_cast<size_t>(rows));
  dgpp::ink2_quantize_nvfp4(dx.p, k, rows, k, dinner.p, k, douter.p, nullptr, nullptr);
  dgpp::ink2_requantize_mxfp8(dx.p, k, rows, k, dmx.p, k, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  const auto inner = dinner.download();
  const auto mx = dmx.download();
  const auto outer = douter.download();
  size_t inner_diff = 0, mx_diff = 0, outer_diff = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    inner_diff += inner[i] != host_inner[i];
    mx_diff += mx[i] != host_mx[i];
  }
  for (int r = 0; r < rows; ++r) outer_diff += std::memcmp(&outer[static_cast<size_t>(r)], &host_outer[static_cast<size_t>(r)], 4) != 0;
  std::printf("quantizers: device vs host, nvfp4 inner %zu, nvfp4 outer %zu, mxfp8 %zu differing of %zu\n", inner_diff,
              outer_diff, mx_diff, x.size());
  require(inner_diff == 0 && outer_diff == 0 && mx_diff == 0, "quantizers: the device kernels differ from the host statements");
  if (!dump_path.empty()) {
    std::ofstream f(dump_path, std::ios::binary);
    const int32_t header[2] = {rows, k};
    f.write(reinterpret_cast<const char*>(header), 8);
    f.write(reinterpret_cast<const char*>(x.data()), static_cast<std::streamsize>(x.size() * 2));
    f.write(reinterpret_cast<const char*>(inner.data()), static_cast<std::streamsize>(inner.size() * 2));
    f.write(reinterpret_cast<const char*>(outer.data()), static_cast<std::streamsize>(outer.size() * 4));
    f.write(reinterpret_cast<const char*>(mx.data()), static_cast<std::streamsize>(mx.size() * 2));
    require(f.good(), "quantizers: cannot write " + dump_path);
    std::printf("quantizers: wrote %s\n", dump_path.c_str());
  }
  // The reconstruction error. NVFP4: a quarter of the block maximum, or for
  // a block far under the row maximum the smallest block scale's step (the
  // E4M3 scale bottoms out at 2^-9 of the row's outer scale). MXFP8: one
  // E4M3 step of the block maximum.
  double worst4 = 0, worst8 = 0;
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < k / 16; ++b) {
      double bmax = 0, e4 = 0;
      for (int c = 0; c < 16; ++c) {
        const size_t i = static_cast<size_t>(r) * k + b * 16 + c;
        const double v = bf16_bits_to_float(x[i]);
        bmax = std::max(bmax, std::fabs(v));
        e4 = std::max(e4, std::fabs(static_cast<double>(bf16_bits_to_float(inner[i])) * outer[static_cast<size_t>(r)] - v));
      }
      const double floor = static_cast<double>(outer[static_cast<size_t>(r)]) * std::exp2(-9.0);
      if (bmax > 0) worst4 = std::max(worst4, e4 / (bmax + 4.0 * floor));
    }
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < k / 32; ++b) {
      double bmax = 0, e8 = 0;
      for (int c = 0; c < 32; ++c) {
        const size_t i = static_cast<size_t>(r) * k + b * 32 + c;
        const double v = bf16_bits_to_float(x[i]);
        bmax = std::max(bmax, std::fabs(v));
        e8 = std::max(e8, std::fabs(static_cast<double>(bf16_bits_to_float(mx[i])) - v));
      }
      if (bmax > 1e-30) worst8 = std::max(worst8, e8 / bmax);
    }
  std::printf("quantizers: worst error over the block maximum, nvfp4 %.4f, mxfp8 %.4f\n", worst4, worst8);
  require(worst4 <= 0.26 && worst8 <= 0.07, "quantizers: reconstruction error out of range");
}

// ---- the convolution ring ---------------------------------------------------------------

struct ConvCase {
  int channels = 96, requests = 2, T = 40;
  int64_t slot_stride = 96 + 32;   // another stream's channels beside this one
  int64_t ring_offset = 32;
  std::vector<uint16_t> x;         // [T, channels]
  std::vector<float> taps;         // [channels, 4]
  ConvCase() {
    Rng rng(5);
    x.resize(static_cast<size_t>(T) * channels);
    for (auto& v : x) v = float_to_bf16_bits(rng.normal3());
    taps.resize(static_cast<size_t>(channels) * 4);
    for (auto& v : taps) v = 0.6f * rng.normal3();
  }
};

// Runs positions [0, T) of request `req` in launches of the given sizes
// (decode launches store in the kernel, the others with ink2_ring_store).
std::vector<uint16_t> run_conv(const ConvCase& c, const std::vector<int>& launches, bool decode, int req,
                               Dev<uint16_t>& ring) {
  Dev<uint16_t> dx(c.x), out(c.x.size());
  Dev<float> taps(c.taps);
  std::vector<int32_t> req_ids(static_cast<size_t>(c.T), req);
  std::vector<int64_t> pos(static_cast<size_t>(c.T));
  for (int t = 0; t < c.T; ++t) pos[static_cast<size_t>(t)] = t;
  Dev<int32_t> dreq(req_ids);
  Dev<int64_t> dpos(pos);
  int row0 = 0;
  for (int n : launches) {
    dgpp::Ink2ConvArgs a;
    a.x = dx.p + static_cast<size_t>(row0) * c.channels;
    a.x_stride = c.channels;
    a.taps = taps.p;
    a.scale = static_cast<float>(1.0 / std::sqrt(5.0));
    a.ring = ring.p;
    a.ring_request_stride = dgpp::kInk2RingSlots * c.slot_stride;
    a.ring_slot_stride = c.slot_stride;
    a.ring_offset = c.ring_offset;
    a.req_ids = dreq.p + row0;
    a.pos = dpos.p + row0;
    a.rows = n;
    a.channels = c.channels;
    a.store_inline = decode;
    a.out = out.p + static_cast<size_t>(row0) * c.channels;
    a.out_stride = c.channels;
    dgpp::ink2_conv(a, nullptr);
    if (!decode)
      dgpp::ink2_ring_store(a.x, c.channels, 0, ring.p, a.ring_request_stride, c.slot_stride, c.ring_offset, a.req_ids,
                            a.pos, n, c.channels, nullptr);
    row0 += n;
  }
  require(row0 == c.T, "run_conv: the launches do not cover the run");
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  return out.download();
}

void test_convolution() {
  const ConvCase c;
  const size_t ring_elems = static_cast<size_t>(c.requests) * dgpp::kInk2RingSlots * static_cast<size_t>(c.slot_stride);
  // The host statement.
  std::vector<uint16_t> want(c.x.size());
  for (int t = 0; t < c.T; ++t)
    for (int ch = 0; ch < c.channels; ++ch) {
      float acc = 0.0f;
      for (int lag = 3; lag >= 1; --lag)
        if (t - lag >= 0)
          acc += c.taps[static_cast<size_t>(ch) * 4 + lag] * bf16_bits_to_float(c.x[static_cast<size_t>(t - lag) * c.channels + ch]);
      const float self = bf16_bits_to_float(c.x[static_cast<size_t>(t) * c.channels + ch]);
      acc += c.taps[static_cast<size_t>(ch) * 4] * self;
      want[static_cast<size_t>(t) * c.channels + ch] = float_to_bf16_bits((acc + self) * static_cast<float>(1.0 / std::sqrt(5.0)));
    }
  Dev<uint16_t> ring_a(ring_elems), ring_b(ring_elems), ring_c(ring_elems), ring_d(ring_elems);
  const auto whole = run_conv(c, {c.T}, false, 1, ring_a);
  const auto split = run_conv(c, {7, 1, 13, 19}, false, 1, ring_b);
  const auto rows = run_conv(c, std::vector<int>(static_cast<size_t>(c.T), 1), true, 1, ring_c);
  const auto spec = run_conv(c, {5, 8, 3, 8, 8, 8}, true, 1, ring_d);
  int worst = 0;
  for (size_t i = 0; i < want.size(); ++i) worst = std::max(worst, bf16_ulps(whole[i], want[i]));
  std::printf("convolution: one launch vs the host statement, worst %d ulps\n", worst);
  require(worst <= 1, "convolution: the kernel differs from the host statement");
  require(whole == split, "convolution: a split run differs from one launch");
  require(whole == rows, "convolution: single rows differ from one launch");
  require(whole == spec, "convolution: decode launches differ from one launch");
  // The rings hold the same last positions whichever way they were written,
  // and nothing outside the stream's channels or the request was touched.
  const auto ra = ring_a.download(), rc = ring_c.download();
  require(ra == rc, "convolution: the rings differ between prefill and decode stores");
  for (size_t i = 0; i < ra.size(); ++i) {
    const size_t req = i / (dgpp::kInk2RingSlots * static_cast<size_t>(c.slot_stride));
    const size_t in_slot = i % static_cast<size_t>(c.slot_stride);
    if (req != 1 || in_slot < static_cast<size_t>(c.ring_offset)) require(ra[i] == 0, "convolution: a store outside the stream");
  }
  std::printf("convolution: split, single-row and decode launches are bit-identical to one launch\n");
}

void test_snapshot() {
  ConvCase c;
  c.slot_stride = c.channels;
  c.ring_offset = 0;
  const size_t ring_elems = static_cast<size_t>(c.requests) * dgpp::kInk2RingSlots * static_cast<size_t>(c.slot_stride);
  const int64_t request_stride = dgpp::kInk2RingSlots * c.slot_stride;
  Dev<uint16_t> ring(ring_elems);
  const auto whole = run_conv(c, std::vector<int>(static_cast<size_t>(c.T), 1), true, 0, ring);
  // Rebuild: run the first `cut` positions, snapshot, run on (the ring moves
  // past the snapshot), restore into the other request and continue there.
  for (const int cut : {0, 1, 2, 3, 17, 29}) {
    Dev<uint16_t> r2(ring_elems);
    Dev<uint8_t> snap(dgpp::ink2_ring_snapshot_bytes(c.slot_stride));
    ConvCase head = c;
    head.T = cut;
    head.x.resize(static_cast<size_t>(cut) * c.channels);
    if (cut > 0) (void)run_conv(head, std::vector<int>(static_cast<size_t>(cut), 1), true, 0, r2);
    dgpp::ink2_ring_snapshot(r2.p, c.slot_stride, cut, snap.p, nullptr);
    // Twenty more positions overwrite every slot.
    ConvCase more = c;
    if (cut + 20 <= c.T) {
      Dev<uint16_t> dx(c.x), out(c.x.size());
      Dev<float> taps(c.taps);
      std::vector<int32_t> req_ids(20, 0);
      std::vector<int64_t> pos(20);
      for (int i = 0; i < 20; ++i) pos[static_cast<size_t>(i)] = cut + i;
      Dev<int32_t> dreq(req_ids);
      Dev<int64_t> dpos(pos);
      for (int i = 0; i < 20; ++i) {
        dgpp::Ink2ConvArgs a;
        a.x = dx.p + static_cast<size_t>(cut + i) * c.channels;
        a.x_stride = c.channels;
        a.taps = taps.p;
        a.scale = 0.4472136f;
        a.ring = r2.p;
        a.ring_request_stride = request_stride;
        a.ring_slot_stride = c.slot_stride;
        a.req_ids = dreq.p + i;
        a.pos = dpos.p + i;
        a.rows = 1;
        a.channels = c.channels;
        a.store_inline = true;
        a.out = out.p;
        a.out_stride = c.channels;
        dgpp::ink2_conv(a, nullptr);
      }
    }
    dgpp::ink2_ring_restore(snap.p, c.slot_stride, r2.p + request_stride, nullptr);
    // Continue from `cut` in request 1.
    Dev<uint16_t> dx(c.x), out(c.x.size());
    Dev<float> taps(c.taps);
    const int rest = c.T - cut;
    std::vector<int32_t> req_ids(static_cast<size_t>(rest), 1);
    std::vector<int64_t> pos(static_cast<size_t>(rest));
    for (int i = 0; i < rest; ++i) pos[static_cast<size_t>(i)] = cut + i;
    Dev<int32_t> dreq(req_ids);
    Dev<int64_t> dpos(pos);
    dgpp::Ink2ConvArgs a;
    a.x = dx.p + static_cast<size_t>(cut) * c.channels;
    a.x_stride = c.channels;
    a.taps = taps.p;
    a.scale = static_cast<float>(1.0 / std::sqrt(5.0));
    a.ring = r2.p;
    a.ring_request_stride = request_stride;
    a.ring_slot_stride = c.slot_stride;
    a.req_ids = dreq.p;
    a.pos = dpos.p;
    a.rows = rest;
    a.channels = c.channels;
    a.out = out.p;
    a.out_stride = c.channels;
    dgpp::ink2_conv(a, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const auto got = out.download();
    for (size_t i = 0; i < static_cast<size_t>(rest) * c.channels; ++i)
      require(got[i] == whole[static_cast<size_t>(cut) * c.channels + i],
              "snapshot: a restored ring continues differently at cut " + std::to_string(cut));
  }
  std::printf("snapshot: a ring restored from a prefix snapshot continues with the same bits\n");
}

// ---- the mix -----------------------------------------------------------------------------

void test_mix() {
  const int rows = 9, H = 256, blocks = 3, queries = 3;
  Rng rng(3);
  std::vector<std::vector<uint16_t>> src(static_cast<size_t>(blocks) + 2, std::vector<uint16_t>(static_cast<size_t>(rows) * H));
  for (size_t n = 0; n < src.size(); ++n)
    for (auto& v : src[n]) v = float_to_bf16_bits((n == 1 ? 6.0f : 1.0f) * rng.normal3());
  std::vector<float> q(static_cast<size_t>(queries) * H);
  for (auto& v : q) v = 4.0f * rng.normal3() / std::sqrt(static_cast<float>(H));
  std::vector<uint16_t> w(static_cast<size_t>(queries) * H);
  for (auto& v : w) v = float_to_bf16_bits(1.0f + 0.1f * rng.unit());
  const float eps = 1e-6f;
  for (const bool with_addend : {false, true}) {
    std::vector<Dev<uint16_t>*> dsrc;
    for (auto& s : src) dsrc.push_back(new Dev<uint16_t>(s));
    Dev<float> dq(q);
    Dev<uint16_t> dw(w);
    Dev<uint16_t> out(static_cast<size_t>(queries) * rows * H);
    Dev<float> out_f(static_cast<size_t>(queries) * rows * H);
    dgpp::Ink2MixArgs m;
    for (int n = 0; n < blocks; ++n) m.stack[n] = dsrc[static_cast<size_t>(n)]->p;
    m.n_stack = blocks;
    m.last = dsrc[static_cast<size_t>(blocks)]->p;
    m.addend = with_addend ? dsrc[static_cast<size_t>(blocks) + 1]->p : nullptr;
    m.queries = dq.p;
    m.n_queries = queries;
    m.norm_eps = eps;
    m.rows = rows;
    m.hidden = H;
    for (int j = 0; j < queries; ++j) {
      m.norm_w[j] = dw.p + static_cast<size_t>(j) * H;
      m.out_bf16[j] = out.p + static_cast<size_t>(j) * rows * H;
    }
    dgpp::ink2_mix(m, nullptr);
    dgpp::Ink2MixArgs f = m;
    for (int j = 0; j < queries; ++j) {
      f.norm_w[j] = nullptr;
      f.out_bf16[j] = nullptr;
      f.out_f32[j] = out_f.p + static_cast<size_t>(j) * rows * H;
    }
    dgpp::ink2_mix(f, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const auto got = out.download();
    const auto got_f = out_f.download();
    int worst = 0;
    double worst_f = 0;
    for (int j = 0; j < queries; ++j)
      for (int r = 0; r < rows; ++r) {
        // The sources of this row in double precision.
        std::vector<std::vector<double>> x(static_cast<size_t>(blocks) + 1, std::vector<double>(static_cast<size_t>(H)));
        for (int n = 0; n <= blocks; ++n)
          for (int c = 0; c < H; ++c) {
            double v = bf16_bits_to_float(src[static_cast<size_t>(n)][static_cast<size_t>(r) * H + c]);
            if (n == blocks && with_addend) v += bf16_bits_to_float(src[static_cast<size_t>(blocks) + 1][static_cast<size_t>(r) * H + c]);
            x[static_cast<size_t>(n)][static_cast<size_t>(c)] = v;
          }
        std::vector<double> rstd(static_cast<size_t>(blocks) + 1), score(static_cast<size_t>(blocks) + 1);
        double top = -1e300;
        for (int n = 0; n <= blocks; ++n) {
          double ss = 0, dot = 0;
          for (int c = 0; c < H; ++c) {
            ss += x[static_cast<size_t>(n)][static_cast<size_t>(c)] * x[static_cast<size_t>(n)][static_cast<size_t>(c)];
            dot += x[static_cast<size_t>(n)][static_cast<size_t>(c)] * q[static_cast<size_t>(j) * H + c];
          }
          rstd[static_cast<size_t>(n)] = 1.0 / std::sqrt(ss / H + static_cast<double>(dgpp::kInk2MixEps));
          score[static_cast<size_t>(n)] = dot * rstd[static_cast<size_t>(n)];
          top = std::max(top, score[static_cast<size_t>(n)]);
        }
        double z = 0;
        for (int n = 0; n <= blocks; ++n) z += std::exp(score[static_cast<size_t>(n)] - top);
        std::vector<double> mix(static_cast<size_t>(H), 0.0);
        double ss = 0;
        for (int c = 0; c < H; ++c) {
          for (int n = 0; n <= blocks; ++n)
            mix[static_cast<size_t>(c)] += std::exp(score[static_cast<size_t>(n)] - top) / z * x[static_cast<size_t>(n)][static_cast<size_t>(c)] *
                                           rstd[static_cast<size_t>(n)];
          ss += mix[static_cast<size_t>(c)] * mix[static_cast<size_t>(c)];
        }
        const double norm = 1.0 / std::sqrt(ss / H + static_cast<double>(eps));
        for (int c = 0; c < H; ++c) {
          const size_t i = (static_cast<size_t>(j) * rows + r) * H + c;
          const double want = mix[static_cast<size_t>(c)] * norm * bf16_bits_to_float(w[static_cast<size_t>(j) * H + c]);
          worst = std::max(worst, bf16_ulps(got[i], float_to_bf16_bits(static_cast<float>(want))));
          worst_f = std::max(worst_f, std::fabs(got_f[i] - mix[static_cast<size_t>(c)]));
        }
      }
    for (auto* d : dsrc) delete d;
    std::printf("mix (%s): worst %d ulps with the norm, %.2e absolute on the FP32 mix\n",
                with_addend ? "partial plus addend" : "partial", worst, worst_f);
    require(worst <= 1 && worst_f <= 2e-5, "mix: the kernel differs from the double-precision statement");
  }
}

}  // namespace

// The kernel copy (kernels/device_copy.hpp) on spans that are not whole
// 16-byte units: three logits rows of a 200,058-value vocabulary from an odd
// first row (4-byte words), and a span that starts and ends on odd bytes.
// Nothing around a span moves.
void test_device_copy() {
  constexpr size_t kVocab = 200058, kRows = 8;
  std::vector<float> src(kVocab * kRows), dst(kVocab * kRows, -7.0f), got(kVocab * kRows);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<float>(i % 65521) * 0.25f - 3.0f;
  float *d_src = nullptr, *d_dst = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&d_src, src.size() * sizeof(float)));
  DGPP_CUDA_OK(cudaMalloc(&d_dst, dst.size() * sizeof(float)));
  DGPP_CUDA_OK(cudaMemcpy(d_src, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_dst, dst.data(), dst.size() * sizeof(float), cudaMemcpyHostToDevice));
  const size_t first_src = 1, first_dst = 3, rows = 3;
  dgpp::device_copy(d_dst + first_dst * kVocab, d_src + first_src * kVocab, rows * kVocab * sizeof(float), nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  DGPP_CUDA_OK(cudaMemcpy(got.data(), d_dst, got.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < got.size(); ++i) {
    const bool inside = i >= first_dst * kVocab && i < (first_dst + rows) * kVocab;
    const float want = inside ? src[i - first_dst * kVocab + first_src * kVocab] : -7.0f;
    require(got[i] == want, "device copy: value " + std::to_string(i) + (inside ? " was not copied" : " was overwritten"));
  }
  // Bytes: 1,001 of them from byte 3 of the source to byte 5 of the target.
  std::vector<uint8_t> before(4096), after(4096);
  DGPP_CUDA_OK(cudaMemcpy(before.data(), d_dst, before.size(), cudaMemcpyDeviceToHost));
  dgpp::device_copy(reinterpret_cast<uint8_t*>(d_dst) + 5, reinterpret_cast<const uint8_t*>(d_src) + 3, 1001, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  DGPP_CUDA_OK(cudaMemcpy(after.data(), d_dst, after.size(), cudaMemcpyDeviceToHost));
  const auto* raw = reinterpret_cast<const uint8_t*>(src.data());
  for (size_t i = 0; i < after.size(); ++i) {
    const bool inside = i >= 5 && i < 5 + 1001;
    require(after[i] == (inside ? raw[i - 5 + 3] : before[i]), "device copy: byte " + std::to_string(i));
  }
  cudaFree(d_src);
  cudaFree(d_dst);
  std::printf("device copy: word and byte spans off the 16-byte grid, neighbours untouched\n");
}

// Padding rows read no experts: with the rows' positions given, the segment
// builder leaves a padding token in no expert's row list, an expert only
// padding tokens selected has no rows, the live tokens' slots are listed as
// without positions, and the padding slots take the gathered rows behind
// the lists, one each, so no two slots share a row.
void test_padding_segments() {
  constexpr int kTokens = 12, kTopK = 8, kExperts = 128, kRouted = kTokens * kTopK;
  const std::vector<int64_t> pos = {-1, -1, 40, 41, -1, 90, -1, -1, 7, -1, -1, -1};
  std::vector<int32_t> ids(kRouted), rows(kRouted + kTokens), slot_row(kRouted);
  for (int t = 0; t < kTokens; ++t)
    for (int j = 0; j < kTopK; ++j)
      // Distinct experts within a token. Padding tokens alone select 120..127.
      ids[t * kTopK + j] = pos[t] >= 0 ? (t * 11 + j * 13) % 120 : 120 + j;
  int64_t* d_pos = nullptr;
  int32_t *d_ids = nullptr, *d_rows = nullptr, *d_slot_row = nullptr;
  dgpp::MoeSegment* d_segs = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&d_pos, pos.size() * sizeof(int64_t)));
  DGPP_CUDA_OK(cudaMalloc(&d_ids, ids.size() * sizeof(int32_t)));
  DGPP_CUDA_OK(cudaMalloc(&d_rows, rows.size() * sizeof(int32_t)));
  DGPP_CUDA_OK(cudaMalloc(&d_slot_row, slot_row.size() * sizeof(int32_t)));
  DGPP_CUDA_OK(cudaMalloc(&d_segs, (kExperts + 1) * sizeof(dgpp::MoeSegment)));
  DGPP_CUDA_OK(cudaMemcpy(d_pos, pos.data(), pos.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(d_rows, 0xFF, rows.size() * sizeof(int32_t)));
  DGPP_CUDA_OK(cudaMemset(d_slot_row, 0xFF, slot_row.size() * sizeof(int32_t)));
  dgpp::launch_moe_segment(d_ids, kTokens, kTopK, kExperts, d_rows, d_slot_row, d_segs, nullptr, d_pos);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<dgpp::MoeSegment> segs(kExperts + 1);
  DGPP_CUDA_OK(cudaMemcpy(rows.data(), d_rows, rows.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(slot_row.data(), d_slot_row, slot_row.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(segs.data(), d_segs, segs.size() * sizeof(dgpp::MoeSegment), cudaMemcpyDeviceToHost));
  int live_slots = 0;
  for (int t = 0; t < kTokens; ++t) live_slots += pos[t] >= 0 ? kTopK : 0;
  int listed = 0;
  for (int e = 0; e < kExperts; ++e) {
    require(segs[e].row0 == listed && segs[e].expert == e, "padding segments: the experts' row lists are not contiguous");
    if (e >= 120) require(segs[e].rows == 0, "padding segments: an expert only padding tokens selected has rows");
    for (int r = segs[e].row0; r < segs[e].row0 + segs[e].rows; ++r)
      require(pos[rows[r]] >= 0, "padding segments: an expert's row list holds a padding token");
    listed += segs[e].rows;
  }
  require(listed == live_slots, "padding segments: the experts list every live slot once");
  std::vector<int> seen(kRouted, 0);
  for (int i = 0; i < kRouted; ++i) {
    const int r = slot_row[i], t = i / kTopK;
    if (pos[t] >= 0) {
      require(r >= 0 && r < listed && rows[r] == t, "padding segments: a live slot names no gathered row of its token");
      const dgpp::MoeSegment& seg = segs[ids[i]];
      require(r >= seg.row0 && r < seg.row0 + seg.rows, "padding segments: a live slot sits outside its expert's rows");
    } else {
      require(r >= listed && r < kRouted && rows[r] == t, "padding segments: a padding slot names no row of its token behind the lists");
    }
    require(++seen[r] == 1, "padding segments: two slots share a gathered row");
  }
  for (int r = 0; r < kRouted; ++r) require(seen[r] == 1, "padding segments: a gathered row belongs to no slot");
  for (int t = 0; t < kTokens; ++t) require(rows[kRouted + t] == t, "padding segments: the shared rows moved");
  // Without positions every slot is listed, the padding tokens' included.
  dgpp::launch_moe_segment(d_ids, kTokens, kTopK, kExperts, d_rows, d_slot_row, d_segs, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  DGPP_CUDA_OK(cudaMemcpy(segs.data(), d_segs, segs.size() * sizeof(dgpp::MoeSegment), cudaMemcpyDeviceToHost));
  int all = 0;
  for (int e = 0; e < kExperts; ++e) all += segs[e].rows;
  require(all == kRouted, "padding segments: without positions every slot is listed");
  cudaFree(d_pos);
  cudaFree(d_ids);
  cudaFree(d_rows);
  cudaFree(d_slot_row);
  cudaFree(d_segs);
  std::printf("[ OK ] padding rows are in no expert's row list\n");
}

// ---- loop guidance --------------------------------------------------------------------
// The guided logits against the host statement, bit for bit, each row at
// its own request's strength (0 leaves a row untouched, -1 returns the
// earlier logit).
void test_loop_guidance() {
  const int kRows = 6, kRequests = 4;
  const int64_t kVocab = 4099;  // not a multiple of the kernel's threads
  Rng rng(29);
  std::vector<float> z(static_cast<size_t>(kRows) * kVocab), zr(z.size());
  for (int r = 0; r < kRows; ++r)
    for (int64_t v = 0; v < kVocab; ++v) {
      const size_t i = static_cast<size_t>(r) * kVocab + v;
      // Row 0 is close to uniform, row 1 has one dominant token, row 2 a tie
      // of two, the others a spread.
      float value = 6.0f * rng.normal3();
      if (r == 0) value = 0.01f * rng.unit();
      if (r == 1 && v == 17) value = 25.0f;
      if (r == 2 && (v == 5 || v == 4000)) value = 20.0f;
      z[i] = value;
      zr[i] = value + 2.0f * rng.normal3();
    }
  // Rows 0 to 5 belong to requests 0, 1, 2, 3, 1 and one outside the table.
  const std::vector<int32_t> row_request = {0, 1, 2, 3, 1, 9};
  const std::vector<float> strength = {0.0f, -1.0f, 0.3f, 1.0f};
  const auto row_strength = [&](int r) {
    const int q = row_request[static_cast<size_t>(r)];
    return q >= 0 && q < kRequests ? strength[static_cast<size_t>(q)] : 0.0f;
  };
  Dev<int32_t> d_request(row_request);
  Dev<float> d_strength(strength);
  {
    Dev<float> d_z(z), d_ref(zr);
    dgpp::ink2_logits_guide(d_z.p, d_ref.p, kRows, kVocab, d_request.p, d_strength.p, kRequests, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    const std::vector<float> got = d_z.download();
    for (int r = 0; r < kRows; ++r) {
      const float w = row_strength(r);
      for (int64_t v = 0; v < kVocab; ++v) {
        const size_t i = static_cast<size_t>(r) * kVocab + v;
        const float want = w == 0.0f ? z[i] : dgpp::ink2_guided_value_host(w, z[i], zr[i]);
        require(std::memcmp(&got[i], &want, 4) == 0, "loop guidance: a guided logit differs from the host statement");
        if (w == -1.0f) require(std::memcmp(&got[i], &zr[i], 4) == 0, "loop guidance: strength -1 is not the earlier logit");
      }
    }
  }
  std::printf("[ OK ] loop guidance: every row at its own request's strength\n");
}

int main(int argc, char** argv) {
  try {
    std::string dump;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--dump-quant" && i + 1 < argc) dump = argv[++i];
      else throw std::runtime_error("unknown argument " + a);
    }
    test_quantizers(dump);
    test_convolution();
    test_snapshot();
    test_mix();
    test_device_copy();
    test_padding_segments();
    test_loop_guidance();
    std::printf("inkling2 kernels: PASS\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "inkling2_kernels_test: %s\n", e.what());
    return 1;
  }
}
