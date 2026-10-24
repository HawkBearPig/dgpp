#include "models/dsv41/csa2_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/cuda_check.hpp"
#include "kernels/dsa.hpp"
#include "kernels/exl3_gemv.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {
namespace {
size_t align256(size_t b) { return (b + 255) / 256 * 256; }
int64_t round_up_to(int64_t v, int64_t g) { return (v + g - 1) / g * g; }
}  // namespace

void Csa2Config::validate(const Csa2Config& c) {
  auto fail = [](const char* what) { throw std::invalid_argument(std::string("csa2 layer: ") + what); };
  if (c.hidden <= 0 || c.hidden % 32 != 0) fail("hidden must be a positive multiple of 32");
  if (c.q_lora <= 0 || c.q_lora % 32 != 0) fail("q_lora must be a positive multiple of 32");
  if (c.o_lora <= 0 || c.o_lora % 32 != 0) fail("o_lora must be a positive multiple of 32");
  if (c.tp <= 0 || c.num_heads % c.tp != 0 || c.o_groups % c.tp != 0 || c.num_heads % c.o_groups != 0)
    fail("heads and groups must divide by tp, heads by groups");
  if (c.local_heads() < 4 || (c.local_heads() & (c.local_heads() - 1)) != 0)
    fail("local heads must be a power of two >= 4 (the attention head-group tiling)");
  if (c.index_heads != 32) fail("the selection kernels pin 32 index heads");
  // The select networks' bound (kernels/topk_select.cuh): the release's 512,
  // or the 2048 a deployment widens the index budget to (engine.index_topk).
  if (c.index_topk <= 0 || (c.index_topk & (c.index_topk - 1)) != 0 || c.index_topk > kDsaSelectMaxK)
    fail("index_topk must be a power of two <= 2048 (the select networks)");
  if (c.candidate_block <= 0 || c.candidate_blocks <= 0 || c.candidate_blocks > kCsa2CandidateMaxBlocks ||
      (c.candidate_blocks & (c.candidate_blocks - 1)) != 0)
    fail("candidate_blocks must be a power of two <= 2048");
  if (c.window <= 0 || c.ring_slots < c.window + 16) fail("the ring must hold the window plus the spec rows");
  if (c.block_tokens <= 0 || c.block_tokens % 2 != 0) fail("block_tokens must be a positive even number");
  if (!(c.eps >= 0.f)) fail("eps");
}

Csa2Layer::Layout Csa2Layer::layout(const Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens,
                                    int max_decode_rows, int decode_n_split, size_t dot_budget) {
  Csa2Config::validate(cfg);
  if (max_tokens <= 0 || max_cache_tokens <= 0) throw std::invalid_argument("csa2 layer: max_tokens / max_cache_tokens");
  // The decode batch's bound mirrors the engine's (dgpp::kDecodeRowsMax, 96
  // — sixteen DSpark slots at the five-row block since 2026-10-23, twelve
  // at 64 earlier that day; the header stays out of this CUDA unit); the
  // select workspace scales with it.
  constexpr int kCsa2MaxDecodeRows = 96;
  if (max_decode_rows <= 0 || max_decode_rows > kCsa2MaxDecodeRows || max_decode_rows > max_tokens)
    throw std::invalid_argument("csa2 layer: max_decode_rows must be in [1, min(96, max_tokens)]");
  if (decode_n_split <= 0 || dot_budget == 0) throw std::invalid_argument("csa2 layer: decode_n_split / dot_budget");
  const int lh = cfg.local_heads(), lg = cfg.local_groups();
  const size_t T = static_cast<size_t>(max_tokens);
  Layout L;
  size_t off = 0;
  const auto alloc = [&](size_t bytes) {
    off = align256(off);
    const size_t at = off;
    off += std::max<size_t>(bytes, 16);
    return at;
  };
  L.max_entries = round_up_to(max_cache_tokens, kEntryPad);
  // The prefill scorer's logits tile: `dot_budget` bytes of fp32 rows (the
  // fused scorer writes per-row logits only; the per-head dot buffer is
  // gone), at least one row at the pool's full width.
  int tile_cap = static_cast<int>(dot_budget / (size_t(L.max_entries) * 4));
  tile_cap = std::max(1, std::min(tile_cap, max_tokens));
  L.tile_cap = tile_cap;
  L.ws_slots = std::max(std::max(max_decode_rows, 8) * std::max(decode_n_split, 8), kPrefillAttnRows * kPrefillSplit);
  L.ws_win_rows = std::max(std::max(max_decode_rows, 8), kPrefillAttnRows);
  L.qr = alloc(T * cfg.q_lora * 2);
  L.kv = alloc(T * kCsa2Latent * 2);
  L.q = alloc(T * lh * kCsa2Latent * 2);
  L.o = alloc(T * lh * kCsa2Latent * 2);
  L.oa = alloc(T * lg * cfg.o_lora * 2);
  L.idx_q = alloc(T * cfg.index_heads * kCsa2IndexDim * 2);
  L.q_fp8 = alloc(T * cfg.index_heads * kCsa2IndexDim);
  L.q_scale = alloc(T * cfg.index_heads * 4);
  L.w = alloc(T * cfg.index_heads * 2);
  L.w_folded = alloc(T * cfg.index_heads * 4);
  L.comp_kv = alloc(T * kCsa2Latent * 4);
  L.comp_score = alloc(T * kCsa2Latent * 4);
  L.latent = alloc(T * kCsa2Latent * 2);
  L.ik = alloc(T * kCsa2IndexDim * 2);
  L.pos = alloc(T * 8);
  L.req_ids = alloc(T * 4);
  L.req_zero = alloc(T * 4);
  L.slots = alloc(T * 8);
  L.pos_sel = alloc(T * 8);
  L.entries = alloc(T * 8);
  L.ent_pos = alloc(T * 8);
  L.scratch_pos = alloc(T * 8);
  L.iota = alloc(T * 8);
  L.one_block = alloc(16);
  L.wlist = alloc(T * cfg.window * 4);
  L.wcounts = alloc(T * 4);
  L.dlist = alloc(size_t(max_decode_rows) * size_t(cfg.window + kMaxDraftBlock) * 4);
  L.dcounts = alloc(size_t(max_decode_rows) * 4);
  L.wscratch = alloc((size_t(cfg.window) - 1 + T) * latent_row_bytes(Csa2StatePool::kRingFormat, kCsa2Latent));
  L.topk = alloc(T * cfg.index_topk * 4);
  L.counts = alloc(T * 4);
  L.cand = alloc(T * cfg.candidate_blocks * 4);
  L.cand_counts = alloc(T * 4);
  L.m_main = alloc(size_t(L.ws_slots) * lh * 4);
  L.l_main = alloc(size_t(L.ws_slots) * lh * 4);
  L.c_main = alloc(size_t(L.ws_slots) * lh * kCsa2Latent * 4);
  // The window partials carry a split dimension too (the decode fix below):
  // sized like the main source so a small-row window can fill the GPU.
  L.m_win = alloc(size_t(L.ws_slots) * lh * 4);
  L.l_win = alloc(size_t(L.ws_slots) * lh * 4);
  L.c_win = alloc(size_t(L.ws_slots) * lh * kCsa2Latent * 4);
  // The split groups' partials (one per kAttnSplitGroup splits) and the
  // groups' tickets per row pair (the listed kernel's 32 M-rows), both
  // sources (2026-10-23).
  const size_t grp_slots = size_t(L.ws_slots) / kAttnSplitGroup + 1;
  L.m_grp_main = alloc(grp_slots * lh * 4);
  L.l_grp_main = alloc(grp_slots * lh * 4);
  L.c_grp_main = alloc(grp_slots * lh * kCsa2Latent * 4);
  L.m_grp_win = alloc(grp_slots * lh * 4);
  L.l_grp_win = alloc(grp_slots * lh * 4);
  L.c_grp_win = alloc(grp_slots * lh * kCsa2Latent * 4);
  const size_t row_pairs = (size_t(std::max(max_decode_rows, 8)) * lh + 31) / 32;
  L.tickets_per_source = row_pairs * (size_t(std::max(decode_n_split, 8)) / kAttnSplitGroup + 1);
  L.tickets = alloc(2 * L.tickets_per_source * 4);
  L.gather_k = alloc(size_t(L.max_entries) * kCsa2IndexDim);
  L.gather_scale = alloc(size_t(L.max_entries) * 4);
  L.logits = alloc(size_t(tile_cap) * size_t(L.max_entries) * 4);
  L.select_ws = alloc(dsa_select_workspace_bytes(max_decode_rows, L.max_entries));
  L.counter = alloc(16);
  L.violations = alloc(16);
  if (cfg.exl3) {
    // The EXL3 launchers' workspace: the widest of the layer's sites (the
    // rotated rows at 64 x k fp16 and the split partials the narrow shapes
    // take; kernels/exl3_gemv.hpp exl3_gemm_ws_bytes_for).
    const int hpg = cfg.heads_per_group();
    const std::pair<int, int> sites[] = {
        {cfg.q_lora, cfg.hidden},          {kCsa2Latent, cfg.hidden},           {lh * kCsa2Latent, cfg.q_lora},
        {cfg.index_heads * kCsa2IndexDim, cfg.q_lora}, {cfg.o_lora, hpg * kCsa2Latent}, {cfg.hidden, lg * cfg.o_lora},
        {kCsa2IndexDim, kCsa2Latent}};
    size_t bytes = 0;
    for (const auto& nk : sites) bytes = std::max(bytes, exl3_gemm_ws_bytes_for(nk.first, nk.second));
    L.exl3_ws_bytes = bytes;
    L.exl3_ws = alloc(bytes);
  }
  L.total = align256(off);
  return L;
}

size_t Csa2Layer::scratch_bytes(const Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens, int max_decode_rows,
                                int decode_n_split, size_t dot_budget) {
  return layout(cfg, max_tokens, max_cache_tokens, max_decode_rows, decode_n_split, dot_budget).total;
}

Csa2Layer::Csa2Layer(IGemm& gemm, const Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens, void* scratch,
                     size_t scratch_capacity, void* gemm_workspace, size_t gemm_ws_bytes, int max_decode_rows,
                     int decode_n_split, size_t dot_budget)
    : gemm_(gemm), cfg_(cfg), max_tokens_(max_tokens), max_cache_tokens_(max_cache_tokens),
      max_decode_rows_(max_decode_rows), decode_n_split_(decode_n_split), gemm_ws_(gemm_workspace),
      gemm_ws_bytes_(gemm_ws_bytes) {
  const Layout L = layout(cfg, max_tokens, max_cache_tokens, max_decode_rows, decode_n_split, dot_budget);
  if (scratch == nullptr || scratch_capacity < L.total) throw std::invalid_argument("csa2 layer: scratch too small");
  scratch_ = static_cast<uint8_t*>(scratch);
  tile_cap_ = L.tile_cap;
  max_entries_ = L.max_entries;
  ws_slots_ = L.ws_slots;
  ws_win_rows_ = L.ws_win_rows;
  attn_scale_ = static_cast<float>(std::pow(static_cast<double>(kCsa2Latent), -0.5));
  const auto at = [&](size_t off) { return scratch_ + off; };
  qr_ = reinterpret_cast<uint16_t*>(at(L.qr));
  kv_ = reinterpret_cast<uint16_t*>(at(L.kv));
  q_ = reinterpret_cast<uint16_t*>(at(L.q));
  o_ = reinterpret_cast<uint16_t*>(at(L.o));
  oa_ = reinterpret_cast<uint16_t*>(at(L.oa));
  idx_q_ = reinterpret_cast<uint16_t*>(at(L.idx_q));
  q_fp8_ = at(L.q_fp8);
  q_scale_ = reinterpret_cast<float*>(at(L.q_scale));
  iw_ = reinterpret_cast<uint16_t*>(at(L.w));
  w_folded_ = reinterpret_cast<float*>(at(L.w_folded));
  comp_kv_ = reinterpret_cast<float*>(at(L.comp_kv));
  comp_score_ = reinterpret_cast<float*>(at(L.comp_score));
  latent_ = reinterpret_cast<uint16_t*>(at(L.latent));
  ik_ = reinterpret_cast<uint16_t*>(at(L.ik));
  pos_ = reinterpret_cast<int64_t*>(at(L.pos));
  req_ids_ = reinterpret_cast<int32_t*>(at(L.req_ids));
  req_zero_ = reinterpret_cast<int32_t*>(at(L.req_zero));
  slots_ = reinterpret_cast<int64_t*>(at(L.slots));
  pos_sel_ = reinterpret_cast<int64_t*>(at(L.pos_sel));
  entries_ = reinterpret_cast<int64_t*>(at(L.entries));
  ent_pos_ = reinterpret_cast<int64_t*>(at(L.ent_pos));
  scratch_pos_ = reinterpret_cast<int64_t*>(at(L.scratch_pos));
  iota_ = reinterpret_cast<int64_t*>(at(L.iota));
  one_block_ = reinterpret_cast<int32_t*>(at(L.one_block));
  wlist_ = reinterpret_cast<int32_t*>(at(L.wlist));
  wcounts_ = reinterpret_cast<int32_t*>(at(L.wcounts));
  dlist_ = reinterpret_cast<int32_t*>(at(L.dlist));
  dcounts_ = reinterpret_cast<int32_t*>(at(L.dcounts));
  wscratch_ = at(L.wscratch);
  topk_ = reinterpret_cast<int32_t*>(at(L.topk));
  counts_ = reinterpret_cast<int32_t*>(at(L.counts));
  cand_ = reinterpret_cast<int32_t*>(at(L.cand));
  cand_counts_ = reinterpret_cast<int32_t*>(at(L.cand_counts));
  m_main_ = reinterpret_cast<float*>(at(L.m_main));
  l_main_ = reinterpret_cast<float*>(at(L.l_main));
  c_main_ = reinterpret_cast<float*>(at(L.c_main));
  m_win_ = reinterpret_cast<float*>(at(L.m_win));
  l_win_ = reinterpret_cast<float*>(at(L.l_win));
  c_win_ = reinterpret_cast<float*>(at(L.c_win));
  m_grp_main_ = reinterpret_cast<float*>(at(L.m_grp_main));
  l_grp_main_ = reinterpret_cast<float*>(at(L.l_grp_main));
  c_grp_main_ = reinterpret_cast<float*>(at(L.c_grp_main));
  m_grp_win_ = reinterpret_cast<float*>(at(L.m_grp_win));
  l_grp_win_ = reinterpret_cast<float*>(at(L.l_grp_win));
  c_grp_win_ = reinterpret_cast<float*>(at(L.c_grp_win));
  tickets_main_ = reinterpret_cast<int32_t*>(at(L.tickets));
  tickets_win_ = tickets_main_ + L.tickets_per_source;
  gather_k_ = at(L.gather_k);
  gather_scale_ = reinterpret_cast<float*>(at(L.gather_scale));
  logits_ = reinterpret_cast<float*>(at(L.logits));
  select_ws_ = at(L.select_ws);
  counter_ws_ = reinterpret_cast<int32_t*>(at(L.counter));
  violations_ = reinterpret_cast<unsigned*>(at(L.violations));
  if (cfg.exl3) {
    exl3_ws_ = at(L.exl3_ws);
    exl3_ws_bytes_ = L.exl3_ws_bytes;
  }
  // The constants: the iota, the zero request ids, the one-block table,
  // the select workspace and counters (zeroed once).
  std::vector<int64_t> iota(static_cast<size_t>(max_tokens));
  for (int i = 0; i < max_tokens; ++i) iota[static_cast<size_t>(i)] = i;
  DGPP_CUDA_OK(cudaMemcpy(iota_, iota.data(), iota.size() * 8, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(req_zero_, 0, static_cast<size_t>(max_tokens) * 4));
  DGPP_CUDA_OK(cudaMemset(one_block_, 0, 16));
  DGPP_CUDA_OK(cudaMemset(select_ws_, 0, dsa_select_workspace_bytes(max_decode_rows, max_entries_)));
  DGPP_CUDA_OK(cudaMemset(counter_ws_, 0, 16));
  DGPP_CUDA_OK(cudaMemset(tickets_main_, 0, 2 * L.tickets_per_source * 4));  // the group tickets start at zero
  DGPP_CUDA_OK(cudaMemset(violations_, 0, 16));
  if (cfg.exl3) DGPP_CUDA_OK(cudaMemset(exl3_ws_, 0, exl3_ws_bytes_));  // the GEMV's arrival counters start at zero
  DGPP_CUDA_OK(cudaMemset(wscratch_, 0, (size_t(cfg.window) - 1 + size_t(max_tokens)) * latent_row_bytes(Csa2StatePool::kRingFormat, kCsa2Latent)));
}

void Csa2Layer::rebind(const Csa2LayerWeights& w, int layer) {
  const bool exl3 = w.exl3();
  if (exl3 != cfg_.exl3)
    throw std::invalid_argument("csa2 layer: the weights' format (EXL3 / fp8) disagrees with the layer's config");
  if (w.q_norm == nullptr || w.kv_norm == nullptr || w.attn_sink == nullptr || w.inv_freq == nullptr)
    throw std::invalid_argument("csa2 layer: a null norm, sink or rotary table");
  if (exl3) {
    if (w.wkv_x.tiles == nullptr || w.wq_b_x.tiles == nullptr || w.wo_b_x.tiles == nullptr ||
        w.wo_a_x.size() != size_t(cfg_.local_groups()))
      throw std::invalid_argument("csa2 layer: a null EXL3 projection (or the wo_a groups are not the rank's)");
    for (const Exl3Matrix& g : w.wo_a_x)
      if (g.tiles == nullptr || g.rows != cfg_.o_lora || g.cols != int64_t(cfg_.heads_per_group()) * kCsa2Latent)
        throw std::invalid_argument("csa2 layer: an EXL3 wo_a group is null or off the config's geometry");
    if (w.wq_a_x.rows != cfg_.q_lora || w.wq_a_x.cols != cfg_.hidden || w.wkv_x.rows != kCsa2Latent ||
        w.wkv_x.cols != cfg_.hidden || w.wq_b_x.rows != int64_t(cfg_.local_heads()) * kCsa2Latent ||
        w.wq_b_x.cols != cfg_.q_lora || w.wo_b_x.rows != cfg_.hidden ||
        w.wo_b_x.cols != int64_t(cfg_.local_groups()) * cfg_.o_lora)
      throw std::invalid_argument("csa2 layer: EXL3 projection geometry disagrees with the config");
    if (w.kv_source && (w.comp_wkv_x.tiles == nullptr || w.comp_wkv_x.rows != kCsa2Latent ||
                        w.comp_wkv_x.cols != cfg_.hidden || w.idx_wk_x.tiles == nullptr ||
                        w.idx_wk_x.rows != kCsa2IndexDim || w.idx_wk_x.cols != kCsa2Latent ||
                        (w.ratio == 2 && (w.comp_wgate_x.tiles == nullptr || w.comp_wgate_x.rows != kCsa2Latent ||
                                          w.comp_wgate_x.cols != cfg_.hidden))))
      throw std::invalid_argument("csa2 layer: a kv source needs its EXL3 compressor and index-key matrices");
    if (w.index_source && (w.idx_wq_b_x.tiles == nullptr || w.idx_wq_b_x.rows != int64_t(cfg_.index_heads) * kCsa2IndexDim ||
                           w.idx_wq_b_x.cols != cfg_.q_lora))
      throw std::invalid_argument("csa2 layer: an index source needs its EXL3 indexer query matrix");
  } else if (w.wq_a.payload == nullptr || w.wkv.payload == nullptr || w.wq_b.payload == nullptr ||
             w.wo_a.payload == nullptr || w.wo_b.payload == nullptr) {
    throw std::invalid_argument("csa2 layer: a null projection");
  }
  if (w.ratio != 0 && w.ratio != 1 && w.ratio != 2) throw std::invalid_argument("csa2 layer: ratio must be 0, 1 or 2");
  if (w.ratio > 0 && w.cache_ord < 0) throw std::invalid_argument("csa2 layer: a compressing layer needs its cache");
  if (w.ratio == 0 && (w.kv_source || w.index_source || w.candidate_source || w.uses_candidates))
    throw std::invalid_argument("csa2 layer: a window-only layer owns no compressor, indexer or candidates");
  if (w.kv_source && ((!exl3 && (w.comp_wkv == nullptr || w.idx_wk == nullptr)) || w.comp_norm == nullptr ||
                      w.idx_k_norm == nullptr || (w.ratio == 2 && ((!exl3 && w.comp_wgate == nullptr) || w.tail_ord < 0))))
    throw std::invalid_argument("csa2 layer: a kv source needs its compressor and index-key weights");
  if (w.index_source && ((!exl3 && w.idx_wq_b.payload == nullptr) || w.idx_wp == nullptr))
    throw std::invalid_argument("csa2 layer: an index source needs its indexer weights");
  if ((w.candidate_source || w.uses_candidates) && !w.index_source)
    throw std::invalid_argument("csa2 layer: the candidate stages belong to index sources");
  if (w.candidate_source && w.uses_candidates) throw std::invalid_argument("csa2 layer: a candidate source uses no pool");
  if (!exl3 && (w.wq_a.rows != cfg_.q_lora || w.wq_a.cols != cfg_.hidden || w.wkv.rows != kCsa2Latent ||
                w.wkv.cols != cfg_.hidden || w.wq_b.rows != int64_t(cfg_.local_heads()) * kCsa2Latent ||
                w.wq_b.cols != cfg_.q_lora || w.wo_a.rows != int64_t(cfg_.local_groups()) * cfg_.o_lora ||
                w.wo_a.cols != int64_t(cfg_.heads_per_group()) * kCsa2Latent || w.wo_b.rows != cfg_.hidden ||
                w.wo_b.cols != int64_t(cfg_.local_groups()) * cfg_.o_lora))
    throw std::invalid_argument("csa2 layer: projection geometry disagrees with the config");
  if (!exl3 && w.index_source &&
      (w.idx_wq_b.rows != int64_t(cfg_.index_heads) * kCsa2IndexDim || w.idx_wq_b.cols != cfg_.q_lora))
    throw std::invalid_argument("csa2 layer: indexer geometry disagrees with the config");
  w_ = w;
  layer_ = layer;
}

bool Csa2Layer::prepare(int tokens) {
  if (tokens <= 0 || tokens > max_tokens_) throw std::invalid_argument("csa2 layer: prepare rows out of range");
  dsa_prepare_kernel_smem();
  csa2_prepare_kernel_smem();
  bool ok = true;
  ok &= gemm_.ensure_plan(tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32, size_t(cfg_.hidden));
  ok &= gemm_.ensure_plan(tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::BF16, size_t(cfg_.hidden));
  ok &= gemm_.ensure_plan(tokens, kCsa2IndexDim, kCsa2Latent, DType::BF16, GemmOut::BF16, size_t(kCsa2Latent));
  ok &= gemm_.ensure_plan(tokens, cfg_.index_heads, cfg_.hidden, DType::BF16, GemmOut::BF16, size_t(cfg_.hidden));
  return ok;
}

unsigned Csa2Layer::index_violations() const {
  unsigned v = 0;
  DGPP_CUDA_OK(cudaMemcpy(&v, violations_, 4, cudaMemcpyDeviceToHost));
  return v;
}

void Csa2Layer::require_shape(const SelShape& have, const SelShape& want, const char* what) const {
  if (have.kind != want.kind || have.rows != want.rows || have.start != want.start || have.req != want.req ||
      have.pos != want.pos)
    throw std::logic_error(std::string("csa2 layer: ") + what +
                           " reused by a call of another shape (the producing layer's enqueue must precede it "
                           "with the same rows, chunk and request)");
}

void Csa2Layer::validate_pool(const Csa2StatePool& pool) const {
  if (!pool.initialized()) throw std::invalid_argument("csa2 layer: the pool is not initialized");
  if (pool.shape().ring_slots != cfg_.ring_slots || pool.shape().block_tokens != cfg_.block_tokens)
    throw std::invalid_argument("csa2 layer: the pool's ring / block geometry disagrees with the config");
  if (layer_ < 0 || layer_ >= pool.shape().layers) throw std::invalid_argument("csa2 layer: rebind a layer first");
  if (w_.ratio > 0 && (w_.cache_ord >= pool.caches() || pool.cache_ratio(w_.cache_ord) != w_.ratio))
    throw std::invalid_argument("csa2 layer: the layer's cache ordinal / ratio disagrees with the pool");
  if (w_.ratio == 2 && w_.kv_source && w_.tail_ord >= pool.shape().tail_ordinals)
    throw std::invalid_argument("csa2 layer: tail ordinal out of the pool's range");
}

// ---- the projections -------------------------------------------------------------------------

// The decode projections' split-K (2026-10-19): the narrow matrices (wkv's
// 512 rows, wq_a's 1,280, a wo_a group's 1,024) under-fill the grid unsplit
// — in isolation 86, 161 and 136 GB/s at four rows against the 233.6 GB/s
// ceiling — and take the workspace so the k range splits across blocks
// (157, 168, 180 GB/s); wq_b, the indexer's wq_b and wo_b fill the grid
// already and lose 2–15 % to the split's reduce, so they stay unsplit
// (micro_mma_gemv_shapes -d). A row's chain is the split form's, the same
// whatever rows ride in the launch.
void Csa2Layer::project_kv(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream) {
  const uint16_t* h = static_cast<const uint16_t*>(hidden_in);
  if (w_.exl3())
    launch_exl3_gemm_bf16(h, size_t(cfg_.hidden), w_.wkv_x, kv_, tokens, kCsa2Latent, cfg_.hidden, 0, stream, exl3_ws_,
                          exl3_ws_bytes_);
  else
    launch_scale_gemm_grid_bf16(h, size_t(cfg_.hidden), w_.wkv.payload, w_.wkv.scales, kv_, tokens, kCsa2Latent,
                                cfg_.hidden, stream, 0, 5, 5, cfg_.dense_mma, gemm_ws_, gemm_ws_bytes_, prefill_tile_);
  csa2_rmsnorm_bf16(kv_, kCsa2Latent, w_.kv_norm, kv_, kCsa2Latent, tokens, kCsa2Latent, cfg_.eps, stream);
  csa2_rope_apply(kv_ + (kCsa2Latent - kCsa2Rope), kCsa2Latent, kCsa2Latent, 1, kCsa2Rope, pos, w_.inv_freq, false,
                  tokens, stream);
}

void Csa2Layer::project_q_kv(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream, bool rope_q) {
  const uint16_t* h = static_cast<const uint16_t*>(hidden_in);
  const int lh = cfg_.local_heads();
  // wq_a and wkv read the same hidden: one decode launch for both when the
  // pair form holds (mma_gemv_pair_ok: the same split alone, whole blocks),
  // each row's chain its own launch's — one ramp and drain instead of two
  // (2026-10-20); otherwise the two launches.
  const bool paired = !w_.exl3() && cfg_.dense_mma && !prefill_tile_ &&
                      mma_gemv_pair_ok(w_.wq_a.payload, w_.wkv.payload, h, h, size_t(cfg_.hidden), tokens, cfg_.q_lora,
                                       kCsa2Latent, cfg_.hidden, gemm_ws_, gemm_ws_bytes_);
  if (w_.exl3()) {
    // The EXL3 launchers: wq_a, its norm, then wkv's chain (project_kv).
    launch_exl3_gemm_bf16(h, size_t(cfg_.hidden), w_.wq_a_x, qr_, tokens, cfg_.q_lora, cfg_.hidden, 0, stream, exl3_ws_,
                          exl3_ws_bytes_);
    csa2_rmsnorm_bf16(qr_, cfg_.q_lora, w_.q_norm, qr_, cfg_.q_lora, tokens, cfg_.q_lora, cfg_.eps, stream);
    project_kv(hidden_in, tokens, pos, stream);
  } else if (paired) {
    launch_mma_gemv_fp8_bf16_pair(h, w_.wq_a.payload, w_.wq_a.scales, qr_, cfg_.q_lora, size_t(cfg_.q_lora), h,
                                  w_.wkv.payload, w_.wkv.scales, kv_, kCsa2Latent, size_t(kCsa2Latent), size_t(cfg_.hidden),
                                  tokens, cfg_.hidden, 5, 5, stream, gemm_ws_, gemm_ws_bytes_);
    // The two norms and the kv rope in one launch (csa2_norms_rope, 2026-10-20).
    csa2_norms_rope(qr_, cfg_.q_lora, w_.q_norm, cfg_.q_lora, kv_, kCsa2Latent, w_.kv_norm, kCsa2Latent, cfg_.eps,
                    kCsa2Latent - kCsa2Rope, kCsa2Rope, pos, w_.inv_freq, tokens, stream);
  } else {
    launch_scale_gemm_grid_bf16(h, size_t(cfg_.hidden), w_.wq_a.payload, w_.wq_a.scales, qr_, tokens, cfg_.q_lora,
                                cfg_.hidden, stream, 0, 5, 5, cfg_.dense_mma, gemm_ws_, gemm_ws_bytes_, prefill_tile_);
    csa2_rmsnorm_bf16(qr_, cfg_.q_lora, w_.q_norm, qr_, cfg_.q_lora, tokens, cfg_.q_lora, cfg_.eps, stream);
    project_kv(hidden_in, tokens, pos, stream);
  }
  if (w_.exl3())
    launch_exl3_gemm_bf16(qr_, size_t(cfg_.q_lora), w_.wq_b_x, q_, tokens, lh * kCsa2Latent, cfg_.q_lora, 0, stream,
                          exl3_ws_, exl3_ws_bytes_);
  else
    launch_scale_gemm_grid_bf16(qr_, size_t(cfg_.q_lora), w_.wq_b.payload, w_.wq_b.scales, q_, tokens,
                                lh * kCsa2Latent, cfg_.q_lora, stream, 0, 5, 5, cfg_.dense_mma, nullptr, 0, prefill_tile_);
  if (rope_q)
    csa2_rope_apply(q_ + (kCsa2Latent - kCsa2Rope), int64_t(lh) * kCsa2Latent, kCsa2Latent, lh, kCsa2Rope, pos,
                    w_.inv_freq, false, tokens, stream);
}

void Csa2Layer::indexer_query(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream) {
  const int heads = cfg_.index_heads;
  if (w_.exl3())
    launch_exl3_gemm_bf16(qr_, size_t(cfg_.q_lora), w_.idx_wq_b_x, idx_q_, tokens, heads * kCsa2IndexDim, cfg_.q_lora, 0,
                          stream, exl3_ws_, exl3_ws_bytes_);
  else
    launch_scale_gemm_grid_bf16(qr_, size_t(cfg_.q_lora), w_.idx_wq_b.payload, w_.idx_wq_b.scales, idx_q_, tokens,
                                heads * kCsa2IndexDim, cfg_.q_lora, stream, 0, 5, 5, cfg_.dense_mma, nullptr, 0, prefill_tile_);
  csa2_rope_apply(idx_q_ + (kCsa2IndexDim - kCsa2Rope), int64_t(heads) * kCsa2IndexDim, kCsa2IndexDim, heads,
                  kCsa2Rope, pos, w_.inv_freq, false, tokens, stream);
  csa2_index_q_quant(idx_q_, tokens, heads, q_fp8_, q_scale_, violations_, stream);
  gemm_.matmul(hidden_in, w_.idx_wp, iw_, tokens, heads, cfg_.hidden, DType::BF16, GemmOut::BF16,
               size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
  csa2_fold_weights(iw_, q_scale_, w_folded_, int64_t(tokens) * heads, stream);
}

// The compressor's projections: the kv and score rows (fp32, ratio 2) or
// the latent rows (bf16, ratio 1) of `tokens` hidden rows — the bf16
// matmul of the release weights or the EXL3 launchers.
void Csa2Layer::compress_project(const void* hidden_in, int tokens, float* kv, float* score, cudaStream_t stream) {
  if (w_.exl3()) {
    const uint16_t* h = static_cast<const uint16_t*>(hidden_in);
    launch_exl3_gemm_f32(h, size_t(cfg_.hidden), w_.comp_wkv_x, kv, tokens, kCsa2Latent, cfg_.hidden, 0, stream, exl3_ws_,
                         exl3_ws_bytes_);
    launch_exl3_gemm_f32(h, size_t(cfg_.hidden), w_.comp_wgate_x, score, tokens, kCsa2Latent, cfg_.hidden, 0, stream,
                         exl3_ws_, exl3_ws_bytes_);
    return;
  }
  gemm_.matmul(hidden_in, w_.comp_wkv, kv, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
               size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
  gemm_.matmul(hidden_in, w_.comp_wgate, score, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
               size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
}

void Csa2Layer::compress_latent(const void* hidden_in, int tokens, cudaStream_t stream) {
  if (w_.exl3()) {
    launch_exl3_gemm_bf16(static_cast<const uint16_t*>(hidden_in), size_t(cfg_.hidden), w_.comp_wkv_x, latent_, tokens,
                          kCsa2Latent, cfg_.hidden, 0, stream, exl3_ws_, exl3_ws_bytes_);
    return;
  }
  gemm_.matmul(hidden_in, w_.comp_wkv, latent_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::BF16,
               size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
}

void Csa2Layer::publish_entries(Csa2StatePool& pool, const int32_t* req_ids, int n, cudaStream_t stream) {
  if (n <= 0) return;
  const int ord = w_.cache_ord;
  const int epb = pool.entries_per_block(ord);
  // The index keys from the unrotated latents: wk -> RMSNorm -> the tail
  // rotated at the entry's position -> the planar form.
  if (w_.exl3())
    launch_exl3_gemm_bf16(latent_, size_t(kCsa2Latent), w_.idx_wk_x, ik_, n, kCsa2IndexDim, kCsa2Latent, 0, stream,
                          exl3_ws_, exl3_ws_bytes_);
  else
    gemm_.matmul(latent_, w_.idx_wk, ik_, n, kCsa2IndexDim, kCsa2Latent, DType::BF16, GemmOut::BF16,
                 size_t(kCsa2Latent), gemm_ws_, gemm_ws_bytes_, stream);
  csa2_rmsnorm_bf16(ik_, kCsa2IndexDim, w_.idx_k_norm, ik_, kCsa2IndexDim, n, kCsa2IndexDim, cfg_.eps, stream);
  csa2_rope_apply(ik_ + (kCsa2IndexDim - kCsa2Rope), kCsa2IndexDim, kCsa2IndexDim, 1, kCsa2Rope, ent_pos_, w_.inv_freq,
                  false, n, stream);
  csa2_index_k_append(ik_, req_ids, entries_, n, pool.block_tables(), int(pool.total_blocks()), epb, pool.index_k(ord),
                      pool.index_scale(ord), violations_, stream);
  // The main rows: the latent's tail rotated, then the fp4_block append.
  csa2_rope_apply(latent_ + (kCsa2Latent - kCsa2Rope), kCsa2Latent, kCsa2Latent, 1, kCsa2Rope, ent_pos_, w_.inv_freq,
                  false, n, stream);
  dsa_latent_append(latent_, req_ids, entries_, n, pool.block_tables(), int(pool.total_blocks()), epb, pool.main(ord),
                    kCsa2Latent, stream, Csa2StatePool::kMainFormat);
}

void Csa2Layer::attend_rows(Csa2StatePool& pool, const int32_t* req_ids_win, const uint8_t* win_cache,
                            int win_block_tokens, const int32_t* win_table, const int32_t* req_ids_main,
                            const int64_t* pos, int row0, int rows, int n_split_main, cudaStream_t stream,
                            const int32_t* list, int list_stride, const int32_t* counts, bool split_window,
                            int sel_base, bool flash_window) {
  const int lh = cfg_.local_heads();
  if (rows > ws_win_rows_ || rows * n_split_main > ws_slots_) throw std::logic_error("csa2 layer: attention tile too wide");
  const uint16_t* q = q_ + size_t(row0) * lh * kCsa2Latent;
  // The window source. At decode (one or a few rows) an unsplit window is a
  // single thread block per head group — latency-bound at ~4 warps while the
  // rest of the GPU idles (the 2026-09-14 profile: 140 us for 128 keys). The
  // decode caller (`split_window`) splits it across the key dimension the way
  // the compressed source is, sized to fill the SMs. Prefill's row tile
  // already fills the GPU, so it stays one split and is byte-identical to
  // before (the reference-parity gates unchanged); decode's window is then no
  // longer bit-identical to the forward's single block — a different fp32
  // combine order, the rounding class the decode audit certifies.
  // n_split_win <= n_split_main, so it fits ws_slots.
  // The prefill's window on the flash listed kernel (2026-10-18,
  // engine.prefill_window flash): the window scratch is a one-block cache
  // and the slot lists are per-row selections, the listed kernel's inputs;
  // it reads a tile's keys once into shared memory per slab instead of
  // once per thread of the scalar kernel (287 us per 128-row tile in the
  // 32K prefill's ledger, 12 % of the walk). Its S sums in the mma's
  // fragment order — the rounding class the forward-parity gates certify
  // for the main source; the model test gates it against the scalar form.
  // The decode window on the same kernel unsplit (2026-10-19,
  // engine.decode_window flash): the 128-key window is latency, not bytes
  // — the scalar kernel at eight splits took 27-29 us a layer at C1
  // where the flash kernel's one block per row and head group takes ~7 —
  // and it runs on the side stream beside the main source's kernel when
  // the layer has one (the two read disjoint caches and write disjoint
  // partials; the finish joins them). The draft block's rows take the same
  // form (2026-10-20; they kept the scalar kernel before: 131-170 us a
  // draft layer).
  const bool flash_try = lh >= 16 && lh % 16 == 0 && (split_window ? decode_window_flash_ : flash_window);
  const bool beside = split_window && side_stream_ != nullptr && w_.ratio > 0;
  cudaStream_t wstream = stream;
  if (beside) {
    DGPP_CUDA_OK(cudaEventRecord(side_fork_, stream));
    DGPP_CUDA_OK(cudaStreamWaitEvent(side_stream_, side_fork_, 0));
    wstream = side_stream_;
  }
  // Decode's window is split like the scalar form's (eight splits of a
  // 128-key window: one 16-token tile per block — the unsplit flash
  // kernel walked the eight tiles in one block at 36 µs a layer, the
  // scalar kernel's 29; prefill keeps one split, its row tile fills the
  // GPU).
  const int n_split_win = split_window ? std::min(n_split_main, kWinDecodeSplit) : 1;
  // The decode splits in groups (2026-10-23, kernels/dsa.hpp
  // kAttnSplitGroup): each group of eight consecutive splits publishes one
  // merged partial (the last block of the group folds the eight from L2
  // and discards them), so the finish reads n_split / 8 per source and the
  // per-split partials — fifty megabytes a layer at forty rows — never
  // write back. The scalar fallback keeps one per split.
  const int group = split_window && n_split_win % kAttnSplitGroup == 0 && n_split_main % kAttnSplitGroup == 0
                        ? kAttnSplitGroup : 1;
  // The window table has one entry per request (the ring: request r's
  // region r; the prefill scratch: block 0): the physical bound is the
  // ring's region count, not the row width.
  const int win_blocks_total = split_window ? pool.shape().max_requests : 1;
  int n_win_pub = n_split_win;
  const float *m_win_pub = m_win_, *l_win_pub = l_win_, *c_win_pub = c_win_;
  if (flash_try &&
      dsa_attn_listed(q, win_cache, req_ids_win, list + size_t(row0) * list_stride, list_stride, counts + row0, rows,
                      n_split_win, lh, kCsa2Latent, win_block_tokens, win_table, 1, attn_scale_, m_win_, l_win_, c_win_,
                      wstream, Csa2StatePool::kRingFormat, nullptr, 0, win_blocks_total, group, tickets_win_, m_grp_win_,
                      l_grp_win_, c_grp_win_)) {
    n_win_pub = n_split_win / group;
    if (group > 1) {
      m_win_pub = m_grp_win_;
      l_win_pub = l_grp_win_;
      c_win_pub = c_grp_win_;
    }
  } else {
    dsa_attn_partial(q, win_cache, req_ids_win, list + size_t(row0) * list_stride, list_stride, counts + row0, rows,
                     n_split_win, lh, kCsa2Latent, win_block_tokens, win_table, 1, attn_scale_, m_win_, l_win_, c_win_,
                     wstream, Csa2StatePool::kRingFormat, nullptr, 0);
  }
  int n_main_pub = 0;
  const float *m_main_pub = m_main_, *l_main_pub = l_main_, *c_main_pub = c_main_;
  if (w_.ratio > 0) {
    const int ord = w_.cache_ord;
    const int epb = pool.entries_per_block(ord);
    const int32_t* topk = topk_ + size_t(sel_base + row0) * cfg_.index_topk;
    const int32_t* counts = counts_ + sel_base + row0;
    const int n_main = n_split_main;
    if (lh >= 16 && lh % 16 == 0 &&
        dsa_attn_listed(q, pool.main(ord), req_ids_main, topk, cfg_.index_topk, counts, rows, n_main, lh, kCsa2Latent,
                        epb, pool.block_tables(), int(pool.total_blocks()), attn_scale_, m_main_, l_main_, c_main_,
                        stream, Csa2StatePool::kMainFormat, nullptr, 0, -1, group, tickets_main_, m_grp_main_,
                        l_grp_main_, c_grp_main_)) {
      n_main_pub = n_main / group;
      if (group > 1) {
        m_main_pub = m_grp_main_;
        l_main_pub = l_grp_main_;
        c_main_pub = c_grp_main_;
      }
    } else {
      dsa_attn_partial(q, pool.main(ord), req_ids_main, topk, cfg_.index_topk, counts, rows, n_main, lh, kCsa2Latent,
                       epb, pool.block_tables(), int(pool.total_blocks()), attn_scale_, m_main_, l_main_, c_main_,
                       stream, Csa2StatePool::kMainFormat, nullptr, 0);
      n_main_pub = n_main;
    }
  }
  if (beside) {
    DGPP_CUDA_OK(cudaEventRecord(side_join_, side_stream_));
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream, side_join_, 0));
  }
  csa2_attn_finish(n_main_pub ? m_main_pub : nullptr, n_main_pub ? l_main_pub : nullptr,
                   n_main_pub ? c_main_pub : nullptr, n_main_pub, m_win_pub, l_win_pub, c_win_pub, n_win_pub,
                   w_.attn_sink, rows, lh, pos, w_.inv_freq, o_ + size_t(row0) * lh * kCsa2Latent, stream);
}

void Csa2Layer::project_out(int tokens, void* out, cudaStream_t stream) {
  const int lh = cfg_.local_heads(), lg = cfg_.local_groups(), hpg = cfg_.heads_per_group();
  const int K = hpg * kCsa2Latent;
  // wo_a is block-diagonal over the groups: group g projects its own heads.
  // Two groups take one decode launch (the pair form, 2026-10-20: the same
  // k, each group's chain its own launch's); otherwise one launch per group.
  const size_t act_stride = size_t(lh) * kCsa2Latent;
  if (w_.exl3()) {
    // One EXL3 launch per group (its matrix is its own pack tensor), then wo_b.
    for (int g = 0; g < lg; ++g)
      launch_exl3_gemm_bf16(o_ + size_t(g) * K, act_stride, w_.wo_a_x[size_t(g)], oa_ + size_t(g) * cfg_.o_lora, tokens,
                            cfg_.o_lora, K, size_t(lg) * cfg_.o_lora, stream, exl3_ws_, exl3_ws_bytes_);
    launch_exl3_gemm_bf16(oa_, size_t(lg) * cfg_.o_lora, w_.wo_b_x, static_cast<uint16_t*>(out), tokens, cfg_.hidden,
                          lg * cfg_.o_lora, 0, stream, exl3_ws_, exl3_ws_bytes_);
    return;
  }
  const size_t row_off1 = size_t(cfg_.o_lora), scale_off1 = (row_off1 / 32) * (size_t(K) / 32);
  const bool paired = lg == 2 && cfg_.dense_mma && !prefill_tile_ &&
                      mma_gemv_pair_ok(w_.wo_a.payload, w_.wo_a.payload + row_off1 * K, o_, o_ + K, act_stride, tokens,
                                       cfg_.o_lora, cfg_.o_lora, K, gemm_ws_, gemm_ws_bytes_);
  if (paired) {
    launch_mma_gemv_fp8_bf16_pair(o_, w_.wo_a.payload, w_.wo_a.scales, oa_, cfg_.o_lora, size_t(lg) * cfg_.o_lora,
                                  o_ + K, w_.wo_a.payload + row_off1 * K, w_.wo_a.scales + scale_off1, oa_ + row_off1,
                                  cfg_.o_lora, size_t(lg) * cfg_.o_lora, act_stride, tokens, K, 5, 5, stream, gemm_ws_,
                                  gemm_ws_bytes_);
  } else {
    for (int g = 0; g < lg; ++g) {
      const size_t row_off = size_t(g) * cfg_.o_lora;
      launch_scale_gemm_grid_bf16(o_ + size_t(g) * K, act_stride, w_.wo_a.payload + row_off * K,
                                  w_.wo_a.scales + (row_off / 32) * (size_t(K) / 32), oa_ + row_off, tokens, cfg_.o_lora, K,
                                  stream, size_t(lg) * cfg_.o_lora, 5, 5, cfg_.dense_mma, gemm_ws_, gemm_ws_bytes_, prefill_tile_);
    }
  }
  launch_scale_gemm_grid_bf16(oa_, size_t(lg) * cfg_.o_lora, w_.wo_b.payload, w_.wo_b.scales,
                              static_cast<uint16_t*>(out), tokens, cfg_.hidden, lg * cfg_.o_lora, stream, 0, 5, 5, cfg_.dense_mma, nullptr, 0, prefill_tile_);
}

// ---- the decode walk's prefetch windows --------------------------------------------------------
// A window forks from the main stream where it is opened: its bytes start
// streaming once the chain reaches that point, behind whatever the side
// stream still carries. The output projections are the layer's last
// weights and the selection and the attention before them are the chain's
// latency-bound phase, so the window opens behind the last projection the
// chain reads before them: the q/kv projections on a window-only layer,
// the compressor's on a reusing kv source, the indexer's on an index
// source (the compressor and indexer projections take no window of their
// own: they are DRAM-bound launches right behind the q projection, and a
// window a few microseconds ahead of its consumer reads the same lines
// beside it — measured level on DeepSeek-V4, 2026-10-01).

void Csa2Layer::window_fp8(const GlmQuantMatrix& m) {
  if (m.payload) prefetch_->add(m.payload, size_t(m.rows) * size_t(m.cols));
  if (m.scales) prefetch_->add(m.scales, m.scale_bytes());
}

void Csa2Layer::window_exl3(const Exl3Matrix& m) {
  if (m.tiles) prefetch_->add(m.tiles, m.tiles_bytes());
}

void Csa2Layer::open_output_window(cudaStream_t stream) {
  if (!windows()) return;
  prefetch_->open_window(stream, kWindowBudget, prefetch_->layer_rate());
  if (w_.exl3()) {
    for (const Exl3Matrix& g : w_.wo_a_x) window_exl3(g);
    window_exl3(w_.wo_b_x);
    return;
  }
  window_fp8(w_.wo_a);
  window_fp8(w_.wo_b);
}

// ---- decode ---------------------------------------------------------------------------------------

void Csa2Layer::enqueue_decode(const void* hidden_in, Csa2StatePool& pool, const int32_t* req_ids, const int64_t* pos,
                               const int32_t* req_spans, int num_requests, int tokens, void* out, cudaStream_t stream,
                               float* tail_snapshots, bool prefetch) {
  validate_pool(pool);
  if (tokens <= 0 || tokens > max_decode_rows_) throw std::invalid_argument("csa2 layer: decode rows out of range");
  if (num_requests <= 0 || num_requests > pool.shape().max_requests)
    throw std::invalid_argument("csa2 layer: request count out of range");
  if (!hidden_in || !req_ids || !pos || !req_spans || !out) throw std::invalid_argument("csa2 layer: null buffer");
  const SelShape shape{SelKind::kDecode, tokens, -1, -1, pos};
  prefetch_windows_ = prefetch;
  struct WindowScope {
    bool& windows;
    ~WindowScope() { windows = false; }
  } window_scope{prefetch_windows_};

  const int lh = cfg_.local_heads();
  project_q_kv(hidden_in, tokens, pos, stream, /*rope_q=*/false);  // the q rope rides the decode tail below
  // The output window of a layer without an index source (2026-10-20):
  // wo_b alone, opened behind the q/kv projections at the layer rate — it
  // streams under the appends, the selection, the attention and the first
  // output GEMV (whose lines it does not share) and lands for wo_b. At the
  // full rate it landed too (wo_b 372 GB/s) but stretched the small kernels
  // behind the projections by 43 us a layer, the level pass again. Opened at the layer's start it shared
  // L2 with the attention-side window's wq_b and displaced it (the
  // per-layer instrument: wq_b 295 -> 148 GB/s, wo_b 162 -> 368, the layer
  // level); behind the projections wq_b's lines are consumed. wo_a stays
  // out: it is the first output GEMV and a window still streaming its
  // lines when it starts reads them beside it (measured 2026-10-19).
  if (windows() && !(w_.ratio > 0 && w_.index_source)) {
    prefetch_->open_window(stream, kWindowBudget, prefetch_->layer_rate());
    if (w_.exl3()) window_exl3(w_.wo_b_x);
    else window_fp8(w_.wo_b);
  }
  // A window-only layer opens no output window (2026-10-19): with the
  // decode window beside the main source its attention is 12-20 us, and
  // the 1.5 MB of wo_a and wo_b have no cover there — at the light rate
  // they were still streaming when the first output GEMV read the same
  // lines (150 us GEMVs), at the full rate they slowed the attention's
  // own cache reads, and at the layer's start they slowed the q and kv
  // projections. The GEMVs read them from DRAM; an index-source layer's
  // window keeps its cover under the selection and the attention.
  // The decode tail in one launch per row (dsa_decode_tail_fp8block,
  // 2026-10-20): the q rope, the ring slot, the window slot list, the entry
  // position on a layer with a ratio, and the window row's append into the
  // ring (this batch's rows are visible to its own queries: the ring is
  // read after the append) — the five launches it replaces, bitwise.
  static_assert(Csa2StatePool::kRingFormat == LatentFormat::kFp8Block, "the decode tail appends the ring's fp8-block rows");
  dsa_decode_tail_fp8block(q_ + (kCsa2Latent - kCsa2Rope), int64_t(lh) * kCsa2Latent, kCsa2Latent, lh, kCsa2Rope, pos,
                           w_.inv_freq, slots_, cfg_.ring_slots, cfg_.window, wlist_, wcounts_,
                           w_.ratio > 0 ? pos_sel_ : nullptr, w_.ratio, kv_, req_ids, pool.ring_table(), pool.ring(layer_),
                           kCsa2Latent, tokens, stream);

  if (w_.ratio > 0) {
    const int ord = w_.cache_ord;
    if (w_.kv_source) {
      if (w_.ratio == 2) {
        compress_project(hidden_in, tokens, comp_kv_, comp_score_, stream);
        csa2_compress_decode_update(comp_kv_, comp_score_, req_ids, pos, req_spans, num_requests, w_.comp_norm, cfg_.eps,
                                    pool.tails(w_.tail_ord), latent_, entries_, tokens, tail_snapshots, stream);
        csa2_scaled_positions(entries_, ent_pos_, tokens, 2, 0, stream);
      } else {
        compress_latent(hidden_in, tokens, stream);
        csa2_rmsnorm_bf16(latent_, kCsa2Latent, w_.comp_norm, latent_, kCsa2Latent, tokens, kCsa2Latent, cfg_.eps, stream);
        csa2_scaled_positions(pos, entries_, tokens, 1, 0, stream);
        csa2_scaled_positions(pos, ent_pos_, tokens, 1, 0, stream);
      }
      publish_entries(pool, req_ids, tokens, stream);
    }
    // (the entry positions came with the decode tail above)
    if (w_.index_source) {
      indexer_query(hidden_in, tokens, pos, stream);
      open_output_window(stream);  // behind the indexer's projection, under the selection and the attention
      const int epb = pool.entries_per_block(ord);
      if (w_.candidate_source) {
        csa2_select_candidates_decode(
            q_fp8_, w_folded_, req_ids, pos_sel_, tokens, pool.block_tables(),
            int(pool.total_blocks()), pool.index_k(ord), pool.index_scale(ord), epb,
            cfg_.index_heads, cfg_.candidate_block, cfg_.candidate_blocks,
            static_cast<uint64_t*>(select_ws_), max_entries_, cand_, cand_counts_, stream);
        cand_at(0) = shape;
      } else if (w_.uses_candidates) {
        require_shape(cand_at(0), shape, "the candidate pool");
      }
      if (w_.candidate_source || w_.uses_candidates)
        csa2_select_listed_decode(q_fp8_, w_folded_, req_ids, pos_sel_, tokens, pool.block_tables(),
                                  int(pool.total_blocks()), pool.index_k(ord),
                                  pool.index_scale(ord), epb, cfg_.index_heads, cand_,
                                  cfg_.candidate_blocks, cand_counts_, cfg_.candidate_block,
                                  cfg_.index_topk, static_cast<uint64_t*>(select_ws_), max_entries_,
                                  topk_, counts_, stream);
      else
        dsa_select_decode(q_fp8_, w_folded_, req_ids, pos_sel_, tokens, pool.block_tables(), int(pool.total_blocks()),
                          pool.index_k(ord), pool.index_scale(ord), epb, cfg_.index_heads, kCsa2IndexDim,
                          cfg_.index_topk, 1, cfg_.index_topk, topk_, counts_, select_ws_, max_entries_, counter_ws_, 0,
                          stream, true);
      sel_at(0) = shape;
    } else {
      require_shape(sel_at(0), shape, "the selection");
    }
  }
  attend_rows(pool, req_ids, pool.ring(layer_), cfg_.ring_slots, pool.ring_table(), req_ids, pos, 0, tokens,
              decode_n_split_, stream, wlist_, cfg_.window, wcounts_, /*split_window=*/true);
  project_out(tokens, out, stream);
}

// ---- the DSpark draft --------------------------------------------------------------------------------

void Csa2Layer::append_window_rows(const void* x, Csa2StatePool& pool, const int32_t* req_ids, const int64_t* pos, int n,
                                   cudaStream_t stream) {
  validate_pool(pool);
  if (n <= 0 || n > max_tokens_ || n > cfg_.ring_slots) throw std::invalid_argument("csa2 layer: window rows out of range");
  if (!x || !req_ids || !pos) throw std::invalid_argument("csa2 layer: null buffer");
  project_kv(x, n, pos, stream);
  csa2_ring_slot_positions(pos, slots_, n, cfg_.ring_slots, stream);
  dsa_latent_append(kv_, req_ids, slots_, n, pool.ring_table(), 1, cfg_.ring_slots, pool.ring(layer_), kCsa2Latent,
                    stream, Csa2StatePool::kRingFormat);
}

void Csa2Layer::enqueue_draft_block(const void* hidden_in, Csa2StatePool& pool, const int32_t* req_ids, const int64_t* pos,
                                    int rows, int block, void* out, cudaStream_t stream) {
  validate_pool(pool);
  if (w_.ratio != 0) throw std::logic_error("csa2 layer: the draft block runs on a window-only layer");
  if (block <= 0 || block > kMaxDraftBlock || rows <= 0 || rows % block != 0 || rows > max_decode_rows_)
    throw std::invalid_argument("csa2 layer: draft block rows out of range");
  if (cfg_.ring_slots < cfg_.window + block) throw std::invalid_argument("csa2 layer: the ring must hold window + block");
  if (!hidden_in || !req_ids || !pos || !out) throw std::invalid_argument("csa2 layer: null buffer");
  prefetch_windows_ = true;
  struct WindowScope {
    bool& windows;
    ~WindowScope() { windows = false; }
  } window_scope{prefetch_windows_};
  project_q_kv(hidden_in, rows, pos, stream);
  open_output_window(stream);  // the draft stage is window-only: nothing between q and the attention
  csa2_ring_slot_positions(pos, slots_, rows, cfg_.ring_slots, stream);
  dsa_latent_append(kv_, req_ids, slots_, rows, pool.ring_table(), 1, cfg_.ring_slots, pool.ring(layer_), kCsa2Latent,
                    stream, Csa2StatePool::kRingFormat);
  csa2_dspark_window_slots(pos, rows, block, cfg_.window, cfg_.ring_slots, dlist_, dcounts_, stream);
  // The block's rows attend like the decode rows (2026-10-20): the listed
  // flash kernel split like the scalar form under engine.decode_window —
  // the scalar kernel took 131-170 us a draft layer at C1 (five rows, one
  // block per row) against the decode rows' ~20; the lists are the
  // builder's (its stride window + block).
  attend_rows(pool, req_ids, pool.ring(layer_), cfg_.ring_slots, pool.ring_table(), req_ids, pos, 0, rows,
              decode_n_split_, stream, dlist_, cfg_.window + block, dcounts_, /*split_window=*/true);
  project_out(rows, out, stream);
}

// ---- prefill --------------------------------------------------------------------------------------

void Csa2Layer::stage_prefill_rows(int req, int64_t pos0, int tokens, cudaStream_t stream) {
  // Host staging (prefill is never captured): positions and request ids.
  host_i64_.assign(static_cast<size_t>(tokens), 0);
  host_i32_.assign(static_cast<size_t>(tokens), req);
  for (int i = 0; i < tokens; ++i) host_i64_[static_cast<size_t>(i)] = pos0 + i;
  DGPP_CUDA_OK(cudaMemcpyAsync(pos_, host_i64_.data(), static_cast<size_t>(tokens) * 8, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(req_ids_, host_i32_.data(), static_cast<size_t>(tokens) * 4, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));  // the staging vectors are reused by the callers
}

void Csa2Layer::publish_rows(const void* hidden_in, Csa2StatePool& pool, int req, int64_t pos0, int tokens,
                             cudaStream_t stream) {
  int n_entries = 0;
  if (w_.ratio == 2) {
    compress_project(hidden_in, tokens, comp_kv_, comp_score_, stream);
    csa2_compress_pairs_prefill(comp_kv_, comp_score_, tokens, w_.comp_norm, cfg_.eps, latent_,
                                pool.tails(w_.tail_ord) + size_t(req) * 2 * kCsa2Latent, stream);
    n_entries = tokens / 2;
    csa2_scaled_positions(iota_, entries_, n_entries, 1, pos0 / 2, stream);
    csa2_scaled_positions(entries_, ent_pos_, n_entries, 2, 0, stream);
  } else {
    compress_latent(hidden_in, tokens, stream);
    csa2_rmsnorm_bf16(latent_, kCsa2Latent, w_.comp_norm, latent_, kCsa2Latent, tokens, kCsa2Latent, cfg_.eps, stream);
    n_entries = tokens;
    csa2_scaled_positions(pos_, entries_, tokens, 1, 0, stream);
    csa2_scaled_positions(pos_, ent_pos_, tokens, 1, 0, stream);
  }
  publish_entries(pool, req_ids_, n_entries, stream);
}

void Csa2Layer::publish_prefill(const void* hidden_in, Csa2StatePool& pool, int req, int64_t pos0, int tokens,
                                cudaStream_t stream) {
  validate_pool(pool);
  prefill_tile_ = prefill_dense_tile_;
  struct TileScope {
    bool& tile;
    ~TileScope() { tile = false; }
  } tile_scope{prefill_tile_};
  if (!w_.kv_source || w_.ratio <= 0) throw std::logic_error("csa2 layer: publish_prefill on a layer that is not a kv source");
  if (tokens <= 0 || tokens > max_tokens_) throw std::invalid_argument("csa2 layer: prefill rows out of range");
  if (req < 0 || req >= pool.shape().max_requests) throw std::invalid_argument("csa2 layer: request out of range");
  if (pos0 < 0 || pos0 % cfg_.block_tokens != 0)
    throw std::invalid_argument("csa2 layer: a prefill chunk starts on a block boundary");
  if (pool.request_blocks(req) * cfg_.block_tokens < pos0 + tokens)
    throw std::invalid_argument("csa2 layer: the request's blocks do not cover the chunk");
  if (!hidden_in) throw std::invalid_argument("csa2 layer: null buffer");
  stage_prefill_rows(req, pos0, tokens, stream);
  publish_rows(hidden_in, pool, req, pos0, tokens, stream);
}

void Csa2Layer::enqueue_prefill(const void* hidden_in, Csa2StatePool& pool, int req, int64_t pos0, int tokens,
                                void* out, cudaStream_t stream, int64_t floor, bool publish, int row_base) {
  validate_pool(pool);
  prefill_tile_ = prefill_dense_tile_;
  struct TileScope {
    bool& tile;
    ~TileScope() { tile = false; }
  } tile_scope{prefill_tile_};
  if (tokens <= 0 || tokens > max_tokens_) throw std::invalid_argument("csa2 layer: prefill rows out of range");
  if (row_base < 0 || row_base + tokens > max_tokens_)
    throw std::invalid_argument("csa2 layer: the chunk's row base and rows exceed max_tokens");
  if (req < 0 || req >= pool.shape().max_requests) throw std::invalid_argument("csa2 layer: request out of range");
  if (pos0 < 0 || (publish && pos0 % cfg_.block_tokens != 0))
    throw std::invalid_argument("csa2 layer: a prefill chunk starts on a block boundary");
  if (floor < 0 || floor > pos0) throw std::invalid_argument("csa2 layer: the replay floor lies at or before the chunk");
  if (pool.request_blocks(req) * cfg_.block_tokens < pos0 + tokens)
    throw std::invalid_argument("csa2 layer: the request's blocks do not cover the chunk");
  if (!hidden_in || !out) throw std::invalid_argument("csa2 layer: null buffer");
  const SelShape shape{SelKind::kPrefill, tokens, pos0, req, nullptr};
  stage_prefill_rows(req, pos0, tokens, stream);

  project_q_kv(hidden_in, tokens, pos_, stream);
  // The window scratch: the ring's last window - 1 rows (those at or past
  // the floor), then the chunk's rows as a one-block cache (scratch row
  // window - 1 + i).
  uint8_t* ring = pool.ring(layer_) + size_t(req) * pool.ring_bytes_per_request();
  const size_t rb = pool.ring_row_bytes();
  const int win_rows = cfg_.window - 1 + tokens;
  csa2_window_scratch_prologue(ring, cfg_.ring_slots, pos0, cfg_.window, rb, wscratch_, stream, floor);
  csa2_scaled_positions(iota_, scratch_pos_, tokens, 1, cfg_.window - 1, stream);
  dsa_latent_append(kv_, req_zero_, scratch_pos_, tokens, one_block_, 1, win_rows, wscratch_, kCsa2Latent, stream,
                    Csa2StatePool::kRingFormat);
  csa2_window_slots_prefill(pos0, tokens, cfg_.window, wlist_, wcounts_, stream, floor);

  if (w_.ratio > 0) {
    const int ord = w_.cache_ord;
    const int epb = pool.entries_per_block(ord);
    if (w_.kv_source && publish) publish_rows(hidden_in, pool, req, pos0, tokens, stream);
    csa2_entry_positions(pos_, pos_sel_, tokens, w_.ratio, stream);
    if (w_.index_source) {
      indexer_query(hidden_in, tokens, pos_, stream);
      // The visible entries of the chunk's last row, gathered contiguous
      // for the dot GEMM (the padded tail zeroed once for the sanitizer).
      const int64_t n_gather = (pos0 + tokens) / w_.ratio;
      const int64_t padded_n = n_gather > 0 ? round_up_to(n_gather, kEntryPad) : 0;
      if (padded_n > max_entries_) throw std::invalid_argument("csa2 layer: the context exceeds max_cache_tokens");
      if (padded_n > gather_zeroed_) {
        DGPP_CUDA_OK(cudaMemsetAsync(gather_k_ + gather_zeroed_ * kCsa2IndexDim, 0,
                                     size_t(padded_n - gather_zeroed_) * kCsa2IndexDim, stream));
        DGPP_CUDA_OK(cudaMemsetAsync(gather_scale_ + gather_zeroed_, 0, size_t(padded_n - gather_zeroed_) * 4, stream));
        gather_zeroed_ = padded_n;
      }
      if (n_gather > 0)
        dsa_gather_index_pools(pool.block_tables() + size_t(req) * pool.total_blocks(), epb, pool.index_k(ord),
                               pool.index_scale(ord), n_gather, gather_k_, gather_scale_, kCsa2IndexDim, stream);
      // The logits tile holds tile_cap_ rows at max_entries_; at this
      // context's padded_n it holds proportionally more rows, so a row
      // tile reads the gathered keys once for as many queries as the
      // budget allows (the fused scorer, csa2_score_prefill: no per-head
      // dot buffer, the keys from L2 across the tile's rows).
      const int tile_rows = padded_n > 0
          ? int(std::min<size_t>(size_t(tokens),
                                size_t(tile_cap_) * size_t(max_entries_) / size_t(padded_n)))
          : tokens;
      for (int row0 = 0; row0 < tokens; row0 += tile_rows) {
        const int rows = std::min(tile_rows, tokens - row0);
        if (padded_n > 0)
          csa2_score_prefill(q_fp8_ + size_t(row0) * cfg_.index_heads * kCsa2IndexDim, gather_k_,
                             w_folded_ + size_t(row0) * cfg_.index_heads, gather_scale_, pos_sel_ + row0, rows, n_gather,
                             cfg_.index_heads, logits_, padded_n, stream);
        const int64_t stride = std::max<int64_t>(padded_n, 1);
        dbg_logits_rows_ = padded_n > 0 ? rows : 0;
        dbg_logits_stride_ = stride;
        dbg_logits_entries_ = n_gather;
        const size_t srow = size_t(row_base + row0);  // the selection state's row
        if (w_.candidate_source)
          csa2_select_candidates_prefill(logits_, stride, pos_sel_ + row0, rows, cfg_.candidate_block, cfg_.candidate_blocks,
                                         cand_ + srow * cfg_.candidate_blocks, cand_counts_ + srow, stream);
        const bool restricted = w_.candidate_source || w_.uses_candidates;
        csa2_select_rows_prefill(logits_, stride, pos_sel_ + row0, rows, cfg_.index_topk,
                                 restricted ? cand_ + srow * cfg_.candidate_blocks : nullptr, cfg_.candidate_blocks,
                                 restricted ? cand_counts_ + srow : nullptr, cfg_.candidate_block,
                                 topk_ + srow * cfg_.index_topk, counts_ + srow, stream);
      }
      if (w_.candidate_source) cand_at(row_base) = shape;
      else if (w_.uses_candidates) require_shape(cand_at(row_base), shape, "the candidate pool");
      sel_at(row_base) = shape;
    } else {
      require_shape(sel_at(row_base), shape, "the selection");
    }
  }
  for (int row0 = 0; row0 < tokens; row0 += kPrefillAttnRows) {
    const int rows = std::min(kPrefillAttnRows, tokens - row0);
    attend_rows(pool, req_zero_, wscratch_, win_rows, one_block_, req_ids_, pos_ + row0, row0, rows, kPrefillSplit,
                stream, wlist_, cfg_.window, wcounts_, false, row_base, prefill_window_flash_);
  }
  csa2_window_ring_writeback(wscratch_, cfg_.window, pos0, tokens, cfg_.ring_slots, rb, ring, stream);
  project_out(tokens, out, stream);
}

}  // namespace dgpp
