// The int8 head in bit planes (kernels/packq_head.hpp): the full form is
// bitwise the row layout's packed GEMV; the argmax form gives every row the
// full form's argmax and its logit bit for bit (every written logit is the
// full form's), with -inf elsewhere; both across the launcher's row chunks
// and the row table that names each row's slot.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/packq_gemm.hpp"
#include "kernels/packq_gemv.hpp"
#include "kernels/packq_head.hpp"
#include "models/quant_matrix.hpp"
#include "tests/cuda/kda_test_helpers.hpp"

using namespace dgpp::kda_test;

namespace {

struct Head {
  int n = 0, k = 0;
  std::vector<uint8_t> codes;    // [n][k] row-major, the packed core's byte order
  std::vector<uint16_t> scales;  // [n][k/128] f16 (signed, like the AutoRound head's)
};

Head make_head(uint64_t seed, int n, int k) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.f, 40.f);
  std::uniform_real_distribution<float> sd(2e-4f, 6e-4f);
  Head h;
  h.n = n;
  h.k = k;
  h.codes.resize(static_cast<size_t>(n) * k);
  h.scales.resize(static_cast<size_t>(n) * (k / 128));
  for (size_t i = 0; i < h.codes.size(); ++i) {
    const float v = std::round(nd(rng)) + 128.f;
    h.codes[i] = static_cast<uint8_t>(std::min(255.f, std::max(0.f, v)));
  }
  for (size_t i = 0; i < h.scales.size(); ++i) {
    const float s = sd(rng) * ((rng() & 1u) ? 1.f : -1.f);
    h.scales[i] = dgpp::float_to_fp16_bits(s);
  }
  return h;
}

std::vector<uint16_t> make_act(uint64_t seed, int m, int k) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.5f);
  std::vector<uint16_t> x(static_cast<size_t>(m) * k);
  for (auto& v : x) v = dgpp::float_to_bf16_bits(nd(rng));
  return x;
}

// The packed core's word layout of a row's codes: word q holds codes
// 4q..4q+3, code j at bits 8 (j % 4) — i.e. the bytes in order.
struct Device {
  DevBuf packed, scales, act, out_ref, out;
  dgpp::GlmPackedMatrix m;
};

int ref_argmax(const std::vector<float>& logits, size_t off, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (logits[off + i] > logits[off + best]) best = i;  // ties: the smallest id
  return best;
}

// `row_slots`: the launch's per-row slots (empty: every row is slot 0).
// `expect_argmax`: the launch must leave -inf somewhere (the argmax form ran).
void run_case(int n, int k, int m, bool greedy, const std::vector<uint8_t>& flags,
              const std::vector<int32_t>& row_slots, const char* label, bool expect_argmax = false) {
  if (!row_slots.empty() && static_cast<int>(row_slots.size()) != m)
    throw std::runtime_error(std::string(label) + ": one slot per row");
  cudaStream_t st = test_stream();
  const Head h = make_head(7 + n + m, n, k);
  const std::vector<uint16_t> x = make_act(11 + m, m, k);
  // The row layout and its reference.
  DevBuf d_rows(h.codes.size()), d_sc(h.scales.size() * 2), d_x(x.size() * 2);
  DevBuf d_ref(static_cast<size_t>(m) * n * 4), d_out(static_cast<size_t>(m) * n * 4);
  d_rows.upload(h.codes.data(), h.codes.size());
  d_sc.upload(h.scales.data(), h.scales.size() * 2);
  d_x.upload(x.data(), x.size() * 2);
  dgpp::GlmPackedMatrix rows;
  rows.packed = d_rows.as<uint32_t>();
  rows.scales = d_sc.as<uint16_t>();
  rows.rows = n;
  rows.cols = k;
  rows.bits = 8;
  rows.scale_fmt = dgpp::kPackedScaleF16G128;
  dgpp::launch_packq_gemv_f32(d_x.as<uint16_t>(), static_cast<size_t>(k), rows, d_ref.as<float>(), m, n, k, st);
  // The planes.
  std::vector<uint8_t> planes = h.codes;
  dgpp::packq_planes_permute_int8(planes.data(), n, k);
  DevBuf d_planes(planes.size());
  d_planes.upload(planes.data(), planes.size());
  dgpp::GlmPackedMatrix pm = rows;
  pm.packed = d_planes.as<uint32_t>();
  pm.layout = dgpp::kPackedLayoutPlanes8;
  DevBuf d_ab(static_cast<size_t>(4) * n * 4), d_lo(4 * 4), d_flags(std::max<size_t>(1, flags.size())),
      d_slots(std::max<size_t>(1, row_slots.size()) * 4), d_cand(static_cast<size_t>(n) * 4), d_cc(4);
  if (!flags.empty()) d_flags.upload(flags.data(), flags.size());
  if (!row_slots.empty()) d_slots.upload(row_slots.data(), row_slots.size() * 4);
  dgpp::PackqHeadMode mode;
  if (greedy) {
    mode.greedy = d_flags.as<uint8_t>();
    mode.row_slots = row_slots.empty() ? nullptr : d_slots.as<int32_t>();
    mode.req0 = 0;
  }
  dgpp::PackqHeadScratch scratch{d_ab.as<float>(), d_lo.as<int32_t>(), d_cand.as<int32_t>(), d_cc.as<int32_t>()};
  dgpp::launch_packq_head_f32(d_x.as<uint16_t>(), static_cast<size_t>(k), pm, d_out.as<float>(), m, n, k, mode,
                              scratch, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  std::vector<float> ref(static_cast<size_t>(m) * n), out(ref.size());
  DGPP_CUDA_OK(cudaMemcpy(ref.data(), d_ref.p, ref.size() * 4, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(out.data(), d_out.p, out.size() * 4, cudaMemcpyDeviceToHost));
  size_t written = 0, differ = 0;
  for (int a = 0; a < m; ++a) {
    const size_t off = static_cast<size_t>(a) * n;
    // Every written logit is the full form's, bit for bit.
    size_t row_written = 0;
    for (int i = 0; i < n; ++i) {
      if (std::isinf(out[off + i]) && out[off + i] < 0) continue;
      ++row_written;
      if (std::memcmp(&out[off + i], &ref[off + i], 4) != 0) ++differ;
    }
    written += row_written;
    // A row whose own slot is sampled, or a padding row, is a full read
    // whatever its neighbours' slots are.
    const int slot = row_slots.empty() ? 0 : row_slots[static_cast<size_t>(a)];
    const bool sampled = greedy && (slot < 0 || flags[static_cast<size_t>(slot)] == 0);
    if (sampled && row_written != static_cast<size_t>(n))
      throw std::runtime_error(std::string(label) + ": row " + std::to_string(a) + " of slot " +
                               std::to_string(slot) + " is not a full read (" + std::to_string(row_written) +
                               " of " + std::to_string(n) + " logits)");
    // The argmax and its logit.
    const int ra = ref_argmax(ref, off, n), oa = ref_argmax(out, off, n);
    if (ra != oa || std::memcmp(&out[off + oa], &ref[off + ra], 4) != 0)
      throw std::runtime_error(std::string(label) + ": row " + std::to_string(a) + " argmax " + std::to_string(oa) +
                               " vs " + std::to_string(ra));
  }
  if (differ != 0) throw std::runtime_error(std::string(label) + ": written logits differ from the row layout's");
  // Which rows took the argmax form: a chunk of rows is argmax-only when
  // every row's slot is greedy — those rows carry -inf somewhere; a full
  // chunk writes every logit.
  const size_t total = static_cast<size_t>(m) * n;
  std::printf("[ OK ] %s: m=%d n=%d written %zu of %zu logits (%.2f%%), argmax + logit bitwise\n", label, m, n,
              written, total, 100.0 * written / total);
  if (!greedy && written != total) throw std::runtime_error(std::string(label) + ": the full form left -inf");
  if (expect_argmax && written == total)
    throw std::runtime_error(std::string(label) + ": no row took the argmax form");
}

}  // namespace

DGPP_TEST(packq_head_full_form_is_bitwise_the_row_layout) {
  for (int m : {1, 3, 4, 6}) run_case(4096, 2560, m, false, {}, {}, "full");
}

DGPP_TEST(packq_head_argmax_form_keeps_the_argmax_and_its_logit) {
  // One greedy slot: every chunk argmax-only.
  for (int m : {1, 2, 4, 6}) run_case(8192, 2560, m, true, {1}, {}, "argmax", true);
  // A sampled slot among greedy ones through the row table: rows of the
  // sampled slot's chunk take the full form, the others the argmax form.
  run_case(4096, 2560, 8, true, {1, 0}, {0, 0, 0, 0, 1, 1, 1, 1}, "argmax+sampled", true);
  run_case(4096, 2560, 8, true, {1, 1}, {0, 0, 0, 0, 1, 1, 1, 1}, "two greedy slots", true);
  // A narrower K of the compiled set (the test geometries).
  run_case(2048, 1024, 4, true, {1}, {}, "argmax k=1024", true);
}

// A row's slot is its entry of the row table, whatever the launch's shape:
// a sampled slot never reads the argmax form because a neighbouring or an
// earlier slot is greedy.
DGPP_TEST(packq_head_row_table_names_each_rows_slot) {
  // A sampled slot's own launch beside a greedy slot 0: the one-row eager
  // verify and the two-row scalar step of slot 1.
  run_case(4096, 2560, 1, true, {1, 0}, {1}, "sampled slot 1, one row");
  run_case(4096, 2560, 2, true, {1, 0}, {1, 1}, "sampled slot 1, two rows");
  // The same launches of a greedy slot 1 beside a sampled slot 0.
  run_case(4096, 2560, 1, true, {0, 1}, {1}, "greedy slot 1, one row", true);
  run_case(4096, 2560, 2, true, {0, 1}, {1, 1}, "greedy slot 1, two rows", true);
  // One row per request (the chained drafts of a batch).
  run_case(4096, 2560, 4, true, {1, 0, 1, 1}, {0, 1, 2, 3}, "one row a request, one sampled");
  run_case(4096, 2560, 4, true, {1, 1, 1, 1}, {0, 1, 2, 3}, "one row a request, all greedy", true);
  // A batch over slots that are not a prefix, and its padding rows.
  run_case(4096, 2560, 6, true, {1, 1, 0, 1}, {1, 1, 1, 1, 3, 3}, "slots 1 and 3", true);
  run_case(4096, 2560, 4, true, {1, 1}, {0, 0, -1, -1}, "padding rows");
  // Three rows a request: the chunk boundary falls inside a request.
  run_case(4096, 2560, 6, true, {1, 0}, {0, 0, 0, 1, 1, 1}, "three rows a request");
}

DGPP_TEST(packq_head_planes_tile_gemm_is_bitwise_the_row_layout_tile) {
  // The prefill tile kernel over the plane layout (the 128-code k-step)
  // against the row layout's tile: the same codes, the same MMA chain.
  cudaStream_t st = test_stream();
  const int n = 2048, k = 2560, m = 256;
  const Head h = make_head(99, n, k);
  const std::vector<uint16_t> x = make_act(5, m, k);
  std::vector<uint8_t> planes = h.codes;
  dgpp::packq_planes_permute_int8(planes.data(), n, k);
  DevBuf d_rows(h.codes.size()), d_planes(planes.size()), d_sc(h.scales.size() * 2), d_x(x.size() * 2);
  DevBuf d_ref(static_cast<size_t>(m) * n * 4), d_out(static_cast<size_t>(m) * n * 4);
  d_rows.upload(h.codes.data(), h.codes.size());
  d_planes.upload(planes.data(), planes.size());
  d_sc.upload(h.scales.data(), h.scales.size() * 2);
  d_x.upload(x.data(), x.size() * 2);
  dgpp::GlmPackedMatrix rows;
  rows.packed = d_rows.as<uint32_t>();
  rows.scales = d_sc.as<uint16_t>();
  rows.rows = n;
  rows.cols = k;
  rows.bits = 8;
  rows.scale_fmt = dgpp::kPackedScaleF16G128;
  dgpp::GlmPackedMatrix pm = rows;
  pm.packed = d_planes.as<uint32_t>();
  pm.layout = dgpp::kPackedLayoutPlanes8;
  dgpp::launch_packq_gemm_f32(d_x.as<uint16_t>(), static_cast<size_t>(k), rows, d_ref.as<float>(), m, n, k, st);
  dgpp::launch_packq_gemm_f32(d_x.as<uint16_t>(), static_cast<size_t>(k), pm, d_out.as<float>(), m, n, k, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  std::vector<float> ref(static_cast<size_t>(m) * n), out(ref.size());
  DGPP_CUDA_OK(cudaMemcpy(ref.data(), d_ref.p, ref.size() * 4, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(out.data(), d_out.p, out.size() * 4, cudaMemcpyDeviceToHost));
  if (std::memcmp(ref.data(), out.data(), ref.size() * 4) != 0)
    throw std::runtime_error("the plane tile differs from the row tile");
  std::printf("[ OK ] the plane tile GEMM is bitwise the row layout's at m=%d\n", m);
}

DGPP_TEST(packq_head_slice_scatters_the_argmax_to_the_ids) {
  // A 2048-row slice of an 8192-row head scattered into 8192-wide output
  // rows: the slice rows' logits are the slice's own full form's, bit for
  // bit where written, the argmax of the wide row is the slice argmax at
  // its id, and every other column is -inf.
  cudaStream_t st = test_stream();
  const int n_full = 8192, n = 2048, k = 2560, m = 4;
  const Head h = make_head(31, n_full, k);
  const std::vector<uint16_t> x = make_act(17, m, k);
  std::mt19937_64 rng(5);
  std::vector<int32_t> ids(n_full);
  for (int i = 0; i < n_full; ++i) ids[i] = i;
  std::shuffle(ids.begin(), ids.end(), rng);
  ids.resize(n);
  std::sort(ids.begin(), ids.end());
  Head sl;
  sl.n = n;
  sl.k = k;
  sl.codes.resize(static_cast<size_t>(n) * k);
  sl.scales.resize(static_cast<size_t>(n) * (k / 128));
  for (int r = 0; r < n; ++r) {
    std::copy_n(h.codes.begin() + static_cast<size_t>(ids[r]) * k, k, sl.codes.begin() + static_cast<size_t>(r) * k);
    std::copy_n(h.scales.begin() + static_cast<size_t>(ids[r]) * (k / 128), k / 128,
                sl.scales.begin() + static_cast<size_t>(r) * (k / 128));
  }
  std::vector<uint8_t> planes = sl.codes;
  dgpp::packq_planes_permute_int8(planes.data(), n, k);
  DevBuf d_rows(sl.codes.size()), d_planes(planes.size()), d_sc(sl.scales.size() * 2), d_x(x.size() * 2);
  DevBuf d_ref(static_cast<size_t>(m) * n * 4), d_out(static_cast<size_t>(m) * n_full * 4), d_ids(n * 4);
  DevBuf d_ab(static_cast<size_t>(4) * n * 4), d_lo(16), d_flags(1), d_cand(static_cast<size_t>(n) * 4), d_cc(4);
  d_rows.upload(sl.codes.data(), sl.codes.size());
  d_planes.upload(planes.data(), planes.size());
  d_sc.upload(sl.scales.data(), sl.scales.size() * 2);
  d_x.upload(x.data(), x.size() * 2);
  d_ids.upload(ids.data(), ids.size() * 4);
  const uint8_t one = 1;
  d_flags.upload(&one, 1);
  dgpp::GlmPackedMatrix rows;
  rows.packed = d_rows.as<uint32_t>();
  rows.scales = d_sc.as<uint16_t>();
  rows.rows = n;
  rows.cols = k;
  rows.bits = 8;
  rows.scale_fmt = dgpp::kPackedScaleF16G128;
  dgpp::GlmPackedMatrix pm = rows;
  pm.packed = d_planes.as<uint32_t>();
  pm.layout = dgpp::kPackedLayoutPlanes8;
  dgpp::launch_packq_gemv_f32(d_x.as<uint16_t>(), static_cast<size_t>(k), rows, d_ref.as<float>(), m, n, k, st);
  for (const bool greedy : {false, true}) {
    dgpp::PackqHeadMode mode;
    if (greedy) {
      mode.greedy = d_flags.as<uint8_t>();
    }
    dgpp::PackqHeadScratch scratch{d_ab.as<float>(), d_lo.as<int32_t>(), d_cand.as<int32_t>(), d_cc.as<int32_t>()};
    DGPP_CUDA_OK(cudaMemset(d_out.p, 0, d_out.bytes));
    dgpp::launch_packq_head_f32(d_x.as<uint16_t>(), static_cast<size_t>(k), pm, d_out.as<float>(), m, n, k, mode,
                                scratch, st, d_ids.as<int32_t>(), n_full);
    DGPP_CUDA_OK(cudaStreamSynchronize(st));
    std::vector<float> ref(static_cast<size_t>(m) * n), out(static_cast<size_t>(m) * n_full);
    DGPP_CUDA_OK(cudaMemcpy(ref.data(), d_ref.p, ref.size() * 4, cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(out.data(), d_out.p, out.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<char> in_slice(n_full, 0);
    for (int32_t id : ids) in_slice[id] = 1;
    size_t written = 0;
    for (int a = 0; a < m; ++a) {
      const size_t o = static_cast<size_t>(a) * n_full, r0 = static_cast<size_t>(a) * n;
      for (int c = 0; c < n_full; ++c) {
        const float v = out[o + c];
        if (!in_slice[c] && !(std::isinf(v) && v < 0)) throw std::runtime_error("slice: a column outside the set is not -inf");
      }
      for (int r = 0; r < n; ++r) {
        const float v = out[o + ids[r]];
        if (std::isinf(v) && v < 0) continue;
        ++written;
        if (std::memcmp(&v, &ref[r0 + r], 4) != 0) throw std::runtime_error("slice: a written logit differs");
      }
      const int ra = ref_argmax(ref, r0, n), oa = ref_argmax(out, o, n_full);
      if (oa != ids[ra]) throw std::runtime_error("slice: the wide row's argmax is not the slice argmax's id");
    }
    std::printf("[ OK ] slice %s: %zu of %zu slice logits written, argmax at its id, the rest -inf\n",
                greedy ? "argmax form" : "full form", written, static_cast<size_t>(m) * n);
  }
}

int main() { return dgpp::test::run_all(); }
