// The Qwen norm kernels against the host oracles: the
// (1 + w) norm, its grouped form over the 4-branch hyper state, and the
// GDN gated norm with the reference's three roundings. The device sum of
// squares is a tree, the host's a chain: rstd may differ by an fp32 ulp,
// so the outputs are held to one bf16 ulp on a small fraction of elements.
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/qwen_norm.hpp"
#include "models/qwen/gdn_reference.hpp"
#include "models/qwen/norm_reference.hpp"

using namespace dgpp::kda_test;

namespace {

std::vector<uint16_t> device_norm(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
                                  int64_t rows, int dim, int groups, float eps) {
  cudaStream_t s = test_stream();
  DevBuf dx(x.size() * 2), dw(w.size() * 2), dy(x.size() * 2);
  dx.upload(x.data(), x.size() * 2);
  dw.upload(w.data(), w.size() * 2);
  if (groups == 1)
    dgpp::qwen_rmsnorm_bf16(dx.p, dw.p, dy.p, rows, dim, eps, s);
  else
    dgpp::qwen_group_rmsnorm_bf16(dx.p, dw.p, dy.p, rows, groups, dim, eps, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<uint16_t> y(x.size());
  dy.download(y.data(), y.size() * 2);
  return y;
}

}  // namespace

DGPP_TEST(qwen_rmsnorm_matches_the_oracle_within_an_ulp) {
  for (const int dim : {128, 256, 2560}) {
    const int64_t rows = 9;
    const std::vector<uint16_t> x = random_bf16_normal(1, rows * dim, 1.5f);
    const std::vector<uint16_t> w = random_bf16_uniform(2, dim, 0.3f);
    const std::vector<uint16_t> got = device_norm(x, w, rows, dim, 1, 1e-6f);
    std::vector<uint16_t> want(x.size());
    dgpp::qwen_ref::rmsnorm(x.data(), w.data(), want.data(), rows, dim, 1e-6f);
    require_bf16("rmsnorm dim " + std::to_string(dim), compare_bf16(got, want, 1), 1e-3, 0.01);
  }
}

DGPP_TEST(qwen_group_rmsnorm_normalizes_every_branch_on_its_own) {
  const int groups = 4, dim = 2560;
  const int64_t rows = 3;
  const std::vector<uint16_t> x = random_bf16_normal(3, rows * groups * dim, 2.0f);
  const std::vector<uint16_t> w = random_bf16_uniform(4, groups * dim, 0.3f);
  const std::vector<uint16_t> got = device_norm(x, w, rows, dim, groups, 1e-6f);
  std::vector<uint16_t> want(x.size());
  dgpp::qwen_ref::group_rmsnorm(x.data(), w.data(), want.data(), rows, groups, dim, 1e-6f);
  require_bf16("group rmsnorm", compare_bf16(got, want, 1), 1e-3, 0.01);
  // The grouped form is the plain form per group with that group's weights.
  std::vector<uint16_t> flat(x.size());
  for (int64_t r = 0; r < rows; ++r)
    for (int g = 0; g < groups; ++g) {
      const int64_t base = (r * groups + g) * dim;
      dgpp::qwen_ref::rmsnorm(x.data() + base, w.data() + g * dim, flat.data() + base, 1, dim, 1e-6f);
    }
  require_bitwise("grouped == plain per group", want.data(), flat.data(), want.size() * 2);
}

DGPP_TEST(gdn_gated_rmsnorm_keeps_the_reference_roundings) {  const int dim = 128;
  const int64_t rows = 12 * 7;
  const std::vector<uint16_t> x = random_bf16_normal(5, rows * dim, 1.0f);
  const std::vector<uint16_t> g = random_bf16_normal(6, rows * dim, 2.0f);
  const std::vector<uint16_t> w = random_bf16_uniform(7, dim, 0.5f);
  cudaStream_t s = test_stream();
  DevBuf dx(x.size() * 2), dg(g.size() * 2), dw(w.size() * 2), dy(x.size() * 2);
  dx.upload(x.data(), x.size() * 2);
  dg.upload(g.data(), g.size() * 2);
  dw.upload(w.data(), w.size() * 2);
  dgpp::gdn_gated_rmsnorm_bf16(dx.p, dg.p, dw.p, dy.p, rows, dim, 1e-6f, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<uint16_t> got(x.size()), want(x.size());
  dy.download(got.data(), got.size() * 2);
  dgpp::gdn_ref::gated_rmsnorm_sigmoid(x.data(), g.data(), w.data(), want.data(), rows, dim, 1e-6f);
  require_bf16("gdn gated norm", compare_bf16(got, want, 1), 1e-3, 0.01);
}

DGPP_TEST(gdn_gated_rmsnorm_swish_keeps_the_reference_roundings) {
  const int dim = 128;
  const int64_t rows = 12 * 7;
  const std::vector<uint16_t> x = random_bf16_normal(15, rows * dim, 1.0f);
  const std::vector<uint16_t> g = random_bf16_normal(16, rows * dim, 2.0f);
  const std::vector<uint16_t> w = random_bf16_uniform(17, dim, 0.5f);
  cudaStream_t s = test_stream();
  DevBuf dx(x.size() * 2), dg(g.size() * 2), dw(w.size() * 2), dy(x.size() * 2);
  dx.upload(x.data(), x.size() * 2);
  dg.upload(g.data(), g.size() * 2);
  dw.upload(w.data(), w.size() * 2);
  dgpp::gdn_gated_rmsnorm_swish_bf16(dx.p, dg.p, dw.p, dy.p, rows, dim, 1e-6f, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<uint16_t> got(x.size()), want(x.size());
  dy.download(got.data(), got.size() * 2);
  dgpp::gdn_ref::gated_rmsnorm_swish(x.data(), g.data(), w.data(), want.data(), rows, dim, 1e-6f);
  require_bf16("gdn gated norm swish", compare_bf16(got, want, 1), 1e-3, 0.01);
  // Swish is not sigmoid: the two gates must differ almost everywhere.
  std::vector<uint16_t> sig(x.size());
  dgpp::gdn_ref::gated_rmsnorm_sigmoid(x.data(), g.data(), w.data(), sig.data(), rows, dim, 1e-6f);
  size_t same = 0;
  for (size_t i = 0; i < got.size(); ++i) same += (got[i] == sig[i]);
  if (same * 20 >= got.size())
    throw std::runtime_error("swish gate output suspiciously close to sigmoid");
}

int main() { return dgpp::test::run_all(); }
