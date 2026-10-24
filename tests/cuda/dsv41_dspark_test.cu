// The DSpark Markov bias (kernels/dsv41_dspark.hpp): the biased row is the base row plus the
// prev token's Markov dot — one fmaf chain in r order per vocab entry —
// and the kernel's coalesced form (rank a multiple of 8, aligned) and its
// per-thread form both match the host's chain bit for bit; a padding
// group (token -1) adds nothing; the block maxima are the rescore
// kernel's per-256-entry layout.
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
#include "kernels/dsv41_dspark.hpp"

namespace {

void require(bool c, const std::string& w) { if (!c) throw std::runtime_error(w); }

void check(int rank, int count, int vocab_begin, int vocab, int groups, int rows_out) {
  std::mt19937 rng(static_cast<unsigned>(rank * 7 + count));
  std::normal_distribution<float> nd(0.f, 0.05f);
  std::vector<uint16_t> embed(static_cast<size_t>(vocab) * rank), head(static_cast<size_t>(vocab) * rank);
  for (auto& x : embed) x = dgpp::float_to_bf16_bits(nd(rng) * 10.f);
  for (auto& x : head) x = dgpp::float_to_bf16_bits(nd(rng));
  const int block_row = 1, block = 3;
  std::vector<float> base(static_cast<size_t>(groups) * block * count);
  for (auto& x : base) x = nd(rng) * 100.f;
  std::vector<int64_t> tok(static_cast<size_t>(groups));
  for (int g = 0; g < groups; ++g)
    tok[static_cast<size_t>(g)] = g == groups - 1 ? -1 : static_cast<int64_t>(rng() % static_cast<unsigned>(vocab));
  uint16_t* de; uint16_t* dh; float* dbase; int64_t* dtok; float* dout; float* dmax;
  DGPP_CUDA_OK(cudaMalloc(&de, embed.size() * 2)); DGPP_CUDA_OK(cudaMemcpy(de, embed.data(), embed.size() * 2, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMalloc(&dh, head.size() * 2)); DGPP_CUDA_OK(cudaMemcpy(dh, head.data(), head.size() * 2, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMalloc(&dbase, base.size() * 4)); DGPP_CUDA_OK(cudaMemcpy(dbase, base.data(), base.size() * 4, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMalloc(&dtok, tok.size() * 8)); DGPP_CUDA_OK(cudaMemcpy(dtok, tok.data(), tok.size() * 8, cudaMemcpyHostToDevice));
  const size_t out_n = static_cast<size_t>(groups) * rows_out * count;
  DGPP_CUDA_OK(cudaMalloc(&dout, out_n * 4)); DGPP_CUDA_OK(cudaMemset(dout, 0x7F, out_n * 4));
  const int blocks = (count + 255) / 256;
  DGPP_CUDA_OK(cudaMalloc(&dmax, static_cast<size_t>(groups) * blocks * 4));
  dgpp::dsv41_dspark_markov_bias(dbase, static_cast<int64_t>(block) * count, block_row, de, dh, rank, vocab_begin, count, dtok, 1,
                                 groups, dout, static_cast<int64_t>(rows_out) * count, rows_out, nullptr, dmax);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> got(out_n), gmax(static_cast<size_t>(groups) * blocks);
  DGPP_CUDA_OK(cudaMemcpy(got.data(), dout, out_n * 4, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(gmax.data(), dmax, gmax.size() * 4, cudaMemcpyDeviceToHost));
  for (int g = 0; g < groups; ++g) {
    const int64_t t = tok[static_cast<size_t>(g)];
    std::vector<float> bmax(static_cast<size_t>(blocks), -INFINITY);
    for (int v = 0; v < count; ++v) {
      float acc = 0.f;
      if (t >= 0)
        for (int r = 0; r < rank; ++r)
          acc = std::fmaf(dgpp::bf16_bits_to_float(embed[static_cast<size_t>(t) * rank + r]),
                          dgpp::bf16_bits_to_float(head[(static_cast<size_t>(vocab_begin) + v) * rank + r]), acc);
      const float want = base[(static_cast<size_t>(g) * block + block_row) * count + v] + acc;
      for (int j = 0; j < rows_out; ++j) {
        const float have = got[(static_cast<size_t>(g) * rows_out + j) * count + v];
        require(std::memcmp(&have, &want, 4) == 0, "bias differs at group " + std::to_string(g) + " entry " + std::to_string(v) +
                                                       " rank " + std::to_string(rank) + ": " + std::to_string(have) + " vs " + std::to_string(want));
      }
      bmax[static_cast<size_t>(v / 256)] = std::fmax(bmax[static_cast<size_t>(v / 256)], want);
    }
    for (int x = 0; x < blocks; ++x)
      require(gmax[static_cast<size_t>(g) * blocks + x] == bmax[static_cast<size_t>(x)], "block maximum differs at block " + std::to_string(x));
  }
  cudaFree(de); cudaFree(dh); cudaFree(dbase); cudaFree(dtok); cudaFree(dout); cudaFree(dmax);
}

DGPP_TEST(markov_bias_rank_256_coalesced_is_the_host_chain) { check(256, 32320, 32320, 64640, 2, 3); }
DGPP_TEST(markov_bias_rank_64_short_slice) { check(64, 1000, 8, 2000, 3, 1); }
DGPP_TEST(markov_bias_rank_512_two_units_a_lane) { check(512, 700, 0, 700, 1, 2); }
DGPP_TEST(markov_bias_rank_252_per_thread_form) { check(252, 600, 100, 800, 2, 1); }

}  // namespace

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
