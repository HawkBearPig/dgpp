#include "models/dsv4/attn_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "kernels/csa2.hpp"
#include "kernels/dense_mma_bf16w.hpp"
#include "kernels/dsa.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {
namespace {
size_t align256(size_t b) { return (b + 255) / 256 * 256; }
int64_t round_up_to(int64_t v, int64_t g) { return (v + g - 1) / g * g; }
constexpr LatentFormat kRowFormat = LatentFormat::kBf16;
// The fp8 matrices' scale grid as log2 block sizes (the release's 128 x 128).
constexpr int kGrid = 7;
constexpr int kCompWideMax = 2 * kDsv4Latent;
}  // namespace

void Dsv4AttnConfig::validate(const Dsv4AttnConfig& c) {
  auto fail = [](const char* what) { throw std::invalid_argument(std::string("dsv4 attention: ") + what); };
  if (c.hidden <= 0 || c.hidden % 128 != 0) fail("hidden must be a positive multiple of 128");
  if (c.q_lora <= 0 || c.q_lora % 128 != 0) fail("q_lora must be a positive multiple of 128");
  if (c.o_lora <= 0 || c.o_lora % 128 != 0) fail("o_lora must be a positive multiple of 128");
  if (c.tp <= 0 || c.num_heads % c.tp != 0 || c.o_groups % c.tp != 0 || c.num_heads % c.o_groups != 0)
    fail("heads and groups must divide by tp, heads by groups");
  if (c.local_heads() < 4 || (c.local_heads() & (c.local_heads() - 1)) != 0)
    fail("local heads must be a power of two >= 4 (the attention head-group tiling)");
  if (c.index_heads != 32 && c.index_heads != 64) fail("the selection kernels take 32 or 64 index heads");
  if (c.index_topk <= 0 || (c.index_topk & (c.index_topk - 1)) != 0 || c.index_topk > 1024)
    fail("index_topk must be a power of two <= 1024");
  if (c.window <= 0 || c.ring_slots < c.window + 16) fail("the ring must hold the window plus the spec rows");
  if (c.block_tokens != 128) fail("the pool's block is 128 tokens (a whole ratio-128 entry)");
  if (!(c.eps > 0.f)) fail("eps");
}

Dsv4AttnLayer::Layout Dsv4AttnLayer::layout(const Dsv4AttnConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                            int max_decode_rows, int decode_n_split, size_t dot_budget) {
  Dsv4AttnConfig::validate(cfg);
  if (max_tokens <= 0 || max_cache_tokens <= 0) throw std::invalid_argument("dsv4 attention: max_tokens / max_cache_tokens");
  if (max_decode_rows <= 0 || max_decode_rows > 32 || max_decode_rows > max_tokens)
    throw std::invalid_argument("dsv4 attention: max_decode_rows must be in [1, min(32, max_tokens)]");
  if (decode_n_split <= 0 || dot_budget == 0) throw std::invalid_argument("dsv4 attention: decode_n_split / dot_budget");
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
  L.max_entries = round_up_to((max_cache_tokens + 3) / 4, kEntryPad);
  L.dense_stride = static_cast<int>(round_up_to((max_cache_tokens + 127) / 128, 16));
  int tile_cap = static_cast<int>(dot_budget / (size_t(cfg.index_heads) * size_t(L.max_entries) * 4));
  tile_cap = std::max(1, std::min(tile_cap, max_tokens));
  L.tile_cap = tile_cap;
  L.ws_slots = std::max(std::max(max_decode_rows, 8) * std::max(decode_n_split, 8), kPrefillAttnRows * kPrefillSplit);
  L.ws_win_rows = std::max(std::max(max_decode_rows, 8), kPrefillAttnRows);
  L.list_rows = L.ws_win_rows;
  L.qr = alloc(T * cfg.q_lora * 2);
  L.kv = alloc(T * kDsv4Latent * 2);
  L.q = alloc(T * lh * kDsv4Latent * 2);
  L.o = alloc(T * lh * kDsv4Latent * 2);
  L.oa = alloc(T * lg * cfg.o_lora * 2);
  L.idx_q = alloc(T * cfg.index_heads * kDsv4IndexDim * 2);
  L.q_fp8 = alloc(T * cfg.index_heads * kDsv4IndexDim);
  L.q_scale = alloc(T * cfg.index_heads * 4);
  L.w = alloc(T * cfg.index_heads * 2);
  L.w_folded = alloc(T * cfg.index_heads * 4);
  // The compressors' projections: a prefill chunk's rows, or — the lazy
  // ratio-128 form at decode — a whole group per decode row.
  const size_t comp_elems = std::max(T * kCompWideMax, size_t(max_decode_rows) * 128 * kDsv4Latent);
  L.comp_kv = alloc(comp_elems * 4);
  L.comp_score = alloc(comp_elems * 4);
  L.pos = alloc(T * 8);
  L.req_ids = alloc(T * 4);
  L.req_zero = alloc(T * 4);
  L.slots = alloc(T * 8);
  L.pos_sel = alloc(T * 8);
  L.scratch_pos = alloc(T * 8);
  L.iota = alloc(T * 8);
  L.one_block = alloc(16);
  L.wlist = alloc(T * cfg.window * 4);
  L.wcounts = alloc(T * 4);
  L.dlist = alloc(size_t(max_decode_rows) * size_t(cfg.window + kMaxDraftBlock) * 4);
  L.dcounts = alloc(size_t(max_decode_rows) * 4);
  L.wscratch = alloc((size_t(cfg.window) - 1 + T) * Dsv4StatePool::kRowBytes);
  L.topk = alloc(T * cfg.index_topk * 4);
  L.counts = alloc(T * 4);
  L.mlist = alloc(size_t(L.list_rows) * size_t(L.dense_stride) * 4);
  L.m_main = alloc(size_t(L.ws_slots) * lh * 4);
  L.l_main = alloc(size_t(L.ws_slots) * lh * 4);
  L.c_main = alloc(size_t(L.ws_slots) * lh * kDsv4Latent * 4);
  L.m_win = alloc(size_t(L.ws_slots) * lh * 4);
  L.l_win = alloc(size_t(L.ws_slots) * lh * 4);
  L.c_win = alloc(size_t(L.ws_slots) * lh * kDsv4Latent * 4);
  L.gather_k = alloc(size_t(L.max_entries) * kDsv4IndexDim);
  L.gather_scale = alloc(size_t(L.max_entries) * 4);
  L.dot = alloc(size_t(tile_cap) * cfg.index_heads * size_t(L.max_entries) * 4);
  L.logits = alloc(size_t(tile_cap) * size_t(L.max_entries) * 4);
  L.select_ws = alloc(dsa_select_workspace_bytes(max_decode_rows, L.max_entries));
  L.counter = alloc(16);
  L.violations = alloc(16);
  L.total = align256(off);
  return L;
}

size_t Dsv4AttnLayer::scratch_bytes(const Dsv4AttnConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                    int max_decode_rows, int decode_n_split, size_t dot_budget) {
  return layout(cfg, max_tokens, max_cache_tokens, max_decode_rows, decode_n_split, dot_budget).total;
}

Dsv4AttnLayer::Dsv4AttnLayer(IGemm& gemm, const Dsv4AttnConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                             void* scratch, size_t scratch_capacity, void* gemm_workspace, size_t gemm_ws_bytes,
                             int max_decode_rows, int decode_n_split, size_t dot_budget)
    : gemm_(gemm), cfg_(cfg), max_tokens_(max_tokens), max_cache_tokens_(max_cache_tokens),
      max_decode_rows_(max_decode_rows), decode_n_split_(decode_n_split), gemm_ws_(gemm_workspace),
      gemm_ws_bytes_(gemm_ws_bytes) {
  const Layout L = layout(cfg, max_tokens, max_cache_tokens, max_decode_rows, decode_n_split, dot_budget);
  if (scratch == nullptr || scratch_capacity < L.total) throw std::invalid_argument("dsv4 attention: scratch too small");
  scratch_ = static_cast<uint8_t*>(scratch);
  tile_cap_ = L.tile_cap;
  max_entries_ = L.max_entries;
  dense_stride_ = L.dense_stride;
  ws_slots_ = L.ws_slots;
  ws_win_rows_ = L.ws_win_rows;
  list_rows_ = L.list_rows;
  attn_scale_ = static_cast<float>(std::pow(static_cast<double>(kDsv4Latent), -0.5));
  // The reference's `weights_proj(x) * (softmax_scale * n_heads ** -0.5)`: a
  // python double product, cast to the tensor's scalar type at the multiply.
  index_weight_scale_ = static_cast<float>(std::pow(static_cast<double>(kDsv4IndexDim), -0.5) *
                                           std::pow(static_cast<double>(cfg.index_heads), -0.5));
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
  pos_ = reinterpret_cast<int64_t*>(at(L.pos));
  req_ids_ = reinterpret_cast<int32_t*>(at(L.req_ids));
  req_zero_ = reinterpret_cast<int32_t*>(at(L.req_zero));
  slots_ = reinterpret_cast<int64_t*>(at(L.slots));
  pos_sel_ = reinterpret_cast<int64_t*>(at(L.pos_sel));
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
  mlist_ = reinterpret_cast<int32_t*>(at(L.mlist));
  m_main_ = reinterpret_cast<float*>(at(L.m_main));
  l_main_ = reinterpret_cast<float*>(at(L.l_main));
  c_main_ = reinterpret_cast<float*>(at(L.c_main));
  m_win_ = reinterpret_cast<float*>(at(L.m_win));
  l_win_ = reinterpret_cast<float*>(at(L.l_win));
  c_win_ = reinterpret_cast<float*>(at(L.c_win));
  gather_k_ = at(L.gather_k);
  gather_scale_ = reinterpret_cast<float*>(at(L.gather_scale));
  dot_ = reinterpret_cast<float*>(at(L.dot));
  logits_ = reinterpret_cast<float*>(at(L.logits));
  select_ws_ = at(L.select_ws);
  counter_ws_ = reinterpret_cast<int32_t*>(at(L.counter));
  violations_ = reinterpret_cast<unsigned*>(at(L.violations));
  // The constants: the iota, the zero request ids, the one-block table,
  // the select workspace and counters (zeroed once).
  std::vector<int64_t> iota(static_cast<size_t>(max_tokens));
  for (int i = 0; i < max_tokens; ++i) iota[static_cast<size_t>(i)] = i;
  DGPP_CUDA_OK(cudaMemcpy(iota_, iota.data(), iota.size() * 8, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(req_zero_, 0, static_cast<size_t>(max_tokens) * 4));
  DGPP_CUDA_OK(cudaMemset(one_block_, 0, 16));
  DGPP_CUDA_OK(cudaMemset(select_ws_, 0, dsa_select_workspace_bytes(max_decode_rows, max_entries_)));
  DGPP_CUDA_OK(cudaMemset(counter_ws_, 0, 16));
  DGPP_CUDA_OK(cudaMemset(violations_, 0, 16));
  DGPP_CUDA_OK(cudaMemset(wscratch_, 0, (size_t(cfg.window) - 1 + size_t(max_tokens)) * Dsv4StatePool::kRowBytes));
  DGPP_CUDA_OK(cudaMalloc(&d_dense_views_, size_t(kDenseViewsMax) * sizeof(MoeExpertView)));
  DGPP_CUDA_OK(cudaMalloc(&d_dense_segs_, (size_t(kDenseViewsMax + 2) / 3) * sizeof(MoeSegment)));
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&win_side_, cudaStreamNonBlocking));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&win_fork_, cudaEventDisableTiming));
  DGPP_CUDA_OK(cudaEventCreateWithFlags(&win_join_, cudaEventDisableTiming));
}

Dsv4AttnLayer::~Dsv4AttnLayer() {
  if (win_fork_) cudaEventDestroy(win_fork_);
  if (win_join_) cudaEventDestroy(win_join_);
  if (win_side_) cudaStreamDestroy(win_side_);
  cudaFree(d_dense_views_);
  cudaFree(d_dense_segs_);
}

void Dsv4AttnLayer::rebind(const Dsv4AttnWeights& w, int layer) {
  if (w.wq_a.payload == nullptr || w.wkv.payload == nullptr || w.wq_b.payload == nullptr || w.wo_a.payload == nullptr ||
      w.wo_b.payload == nullptr || w.q_norm == nullptr || w.kv_norm == nullptr || w.attn_sink == nullptr ||
      w.inv_freq == nullptr)
    throw std::invalid_argument("dsv4 attention: a null projection, norm, sink or rotary table");
  if (w.ratio != 0 && w.ratio != 4 && w.ratio != 128) throw std::invalid_argument("dsv4 attention: ratio must be 0, 4 or 128");
  if (w.ratio > 0 && (w.cache_ord < 0 || !w.comp.present() || w.comp.ape == nullptr || w.comp.norm == nullptr ||
                      w.comp.wgate == nullptr || w.comp.ratio != w.ratio || w.comp.dim != kDsv4Latent))
    throw std::invalid_argument("dsv4 attention: a compressing layer needs its cache and its compressor");
  if (w.ratio == 4 && (w.idx_wq_b.payload == nullptr || w.idx_wp == nullptr || !w.idx_comp.present() ||
                       w.idx_comp.ape == nullptr || w.idx_comp.norm == nullptr || w.idx_comp.wgate == nullptr ||
                       w.idx_comp.dim != kDsv4IndexDim))
    throw std::invalid_argument("dsv4 attention: a ratio-4 layer needs its indexer");
  if (w.wq_a.rows != cfg_.q_lora || w.wq_a.cols != cfg_.hidden || w.wkv.rows != kDsv4Latent || w.wkv.cols != cfg_.hidden ||
      w.wq_b.rows != int64_t(cfg_.local_heads()) * kDsv4Latent || w.wq_b.cols != cfg_.q_lora ||
      w.wo_a.rows != int64_t(cfg_.local_groups()) * cfg_.o_lora ||
      w.wo_a.cols != int64_t(cfg_.heads_per_group()) * kDsv4Latent || w.wo_b.rows != cfg_.hidden ||
      w.wo_b.cols != int64_t(cfg_.local_groups()) * cfg_.o_lora)
    throw std::invalid_argument("dsv4 attention: projection geometry disagrees with the config");
  if (w.ratio == 4 && (w.idx_wq_b.rows != int64_t(cfg_.index_heads) * kDsv4IndexDim || w.idx_wq_b.cols != cfg_.q_lora))
    throw std::invalid_argument("dsv4 attention: indexer geometry disagrees with the config");
  w_ = w;
  layer_ = layer;
}

bool Dsv4AttnLayer::prepare(int tokens) {
  if (tokens <= 0 || tokens > max_tokens_) throw std::invalid_argument("dsv4 attention: prepare rows out of range");
  dsa_prepare_kernel_smem();
  csa2_prepare_kernel_smem();
  bool ok = true;
  // The compressors' projections (fp32 out) and the indexer's weights (bf16 out).
  ok &= gemm_.ensure_plan(tokens, kDsv4Latent, cfg_.hidden, DType::BF16, GemmOut::F32, size_t(cfg_.hidden));
  ok &= gemm_.ensure_plan(tokens, 2 * kDsv4Latent, cfg_.hidden, DType::BF16, GemmOut::F32, size_t(cfg_.hidden));
  ok &= gemm_.ensure_plan(tokens, 2 * kDsv4IndexDim, cfg_.hidden, DType::BF16, GemmOut::F32, size_t(cfg_.hidden));
  ok &= gemm_.ensure_plan(tokens, cfg_.index_heads, cfg_.hidden, DType::BF16, GemmOut::BF16, size_t(cfg_.hidden));
  return ok;
}

unsigned Dsv4AttnLayer::index_violations() const {
  unsigned v = 0;
  DGPP_CUDA_OK(cudaMemcpy(&v, violations_, 4, cudaMemcpyDeviceToHost));
  return v;
}

void Dsv4AttnLayer::validate_pool(const Dsv4StatePool& pool) const {
  if (!pool.initialized()) throw std::invalid_argument("dsv4 attention: the pool is not initialized");
  if (pool.shape().ring_slots != cfg_.ring_slots || pool.shape().block_tokens != cfg_.block_tokens)
    throw std::invalid_argument("dsv4 attention: the pool's ring / block geometry disagrees with the config");
  if (layer_ < 0 || layer_ >= pool.shape().layers) throw std::invalid_argument("dsv4 attention: rebind a layer first");
  if (w_.ratio > 0 && (w_.cache_ord >= pool.caches() || pool.cache_ratio(w_.cache_ord) != w_.ratio))
    throw std::invalid_argument("dsv4 attention: the layer's cache ordinal / ratio disagrees with the pool");
}

// ---- the projections -------------------------------------------------------------------------

// One dense fp8 projection. A decode walk: the streaming tensor-core form
// (split across the part where the shape is small). A prefill chunk: the
// grouped tensor-core GEMM over one segment of all the chunk's rows
// (launch_moe_grouped_mma_bf16 — the shared expert's ldmatrix kernel, the
// scale GEMM's tile chain: a row's result is the same whatever rows share
// the launch, so a chunked prefill stays bitwise the one-shot) at 128-row
// m-tiles. launch_dense_mma_bf16 is the same chain on the reference tile
// kernel, which the ldmatrix one outruns. (Until 2026-10-01 the prefill
// took the streaming form too, 128 rows a launch: 12 TFLOPS, a fifth of an
// 8K prefill, where this kernel does 77.)
void Dsv4AttnLayer::dense(int view, const uint16_t* act, size_t act_stride, const uint8_t* payload,
                          const float* scales, uint16_t* out, int m, int n, int k, size_t out_stride,
                          cudaStream_t stream) {
  if (prefill_tiles_ && view < dense_views_live_ && (k % 16) == 0) {
    launch_moe_grouped_mma_bf16(act, act_stride, d_dense_segs_ + view / 3, 1, m, kDensePrefillSplit, d_dense_views_,
                                view % 3, out, out_stride != 0 ? out_stride : size_t(n), n, k, stream);
    return;
  }
  launch_scale_gemm_grid_bf16(act, act_stride, payload, scales, out, m, n, k, stream, out_stride, kGrid, kGrid,
                              cfg_.dense_mma, split_ws(), split_ws_bytes());
}

// The prefill chunk's view table (the layer's dense fp8 matrices, wo_a one
// view per local group) and its one-segment records, on the device before
// the chunk's launches (a prefill is never captured).
void Dsv4AttnLayer::upload_dense_views(int tokens, cudaStream_t stream) {
  const int lg = cfg_.local_groups();
  const int K = cfg_.heads_per_group() * kDsv4Latent;
  const size_t grid = size_t(1) << kGrid;
  const int views = kViewWoA + lg;
  if (views > kDenseViewsMax) throw std::logic_error("dsv4 attention: more dense views than the table holds");
  const auto sub = [&](const GlmQuantMatrix& m, size_t row_off, size_t cols) {
    MoeExpertView v;
    v.payload = m.payload + row_off * cols;
    v.scales = m.scales + (row_off / grid) * ((cols + grid - 1) / grid);
    v.scale_shift_rows = kGrid;
    v.scale_shift_cols = kGrid;
    return v;
  };
  h_dense_views_.assign(size_t(kDenseViewsMax), MoeExpertView{});
  h_dense_views_[kViewWqA] = sub(w_.wq_a, 0, size_t(cfg_.hidden));
  h_dense_views_[kViewWkv] = sub(w_.wkv, 0, size_t(cfg_.hidden));
  h_dense_views_[kViewWqB] = sub(w_.wq_b, 0, size_t(cfg_.q_lora));
  h_dense_views_[kViewWoB] = sub(w_.wo_b, 0, size_t(lg) * cfg_.o_lora);
  if (w_.ratio == 4) h_dense_views_[kViewIdxWqB] = sub(w_.idx_wq_b, 0, size_t(cfg_.q_lora));
  for (int g = 0; g < lg; ++g) h_dense_views_[size_t(kViewWoA + g)] = sub(w_.wo_a, size_t(g) * cfg_.o_lora, size_t(K));
  h_dense_segs_.resize(size_t(kDenseViewsMax + 2) / 3);
  for (size_t e = 0; e < h_dense_segs_.size(); ++e) h_dense_segs_[e] = MoeSegment{0, tokens, int32_t(e)};
  DGPP_CUDA_OK(cudaMemcpyAsync(d_dense_views_, h_dense_views_.data(), h_dense_views_.size() * sizeof(MoeExpertView),
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(d_dense_segs_, h_dense_segs_.data(), h_dense_segs_.size() * sizeof(MoeSegment),
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));  // the host tables are rewritten by the next chunk
  dense_views_live_ = views;
}

void Dsv4AttnLayer::project_kv(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream,
                               const RingAppend* ring) {
  const uint16_t* h = static_cast<const uint16_t*>(hidden_in);
  dense(kViewWkv, h, size_t(cfg_.hidden), w_.wkv.payload, w_.wkv.scales, kv_, tokens, kDsv4Latent, cfg_.hidden, 0,
        stream);
  if (ring != nullptr) {
    // The decode rows: the norm, the rotation, act_quant, the ring row and
    // the window list in one launch (bitwise the chain below and the
    // appends behind it).
    dsv4_kv_tail(kv_, w_.kv_norm, cfg_.eps, ring->req_ids, pos, w_.inv_freq, ring->table, cfg_.ring_slots, ring->ring,
                 tokens, cfg_.window, ring->window_lists ? wlist_ : nullptr, ring->window_lists ? wcounts_ : nullptr,
                 stream);
    return;
  }
  csa2_rmsnorm_bf16(kv_, kDsv4Latent, w_.kv_norm, kv_, kDsv4Latent, tokens, kDsv4Latent, cfg_.eps, stream);
  csa2_rope_apply(kv_ + kDsv4Nope, kDsv4Latent, kDsv4Latent, 1, kDsv4Rope, pos, w_.inv_freq, false, tokens, stream);
  dsv4_act_quant_nope(kv_, kDsv4Latent, tokens, pos, stream);
}

// ---- the decode walk's prefetch windows --------------------------------------------------------
// A window forks from the main stream where it is opened: its bytes start
// streaming once the chain reaches that point, behind whatever the side
// stream still carries. Each is opened where the bytes of the windows
// before it have been consumed, so the L2 never holds more than about one
// window ahead of the chain (24 MB on the GB10).

void Dsv4AttnLayer::window_fp8(const GlmQuantMatrix& m) {
  if (m.payload) prefetch_->add(m.payload, size_t(m.rows) * size_t(m.cols));
  if (m.scales) prefetch_->add(m.scales, m.scale_bytes());
}

// The compressor and indexer projections take no window of their own: they
// are DRAM-bound launches right behind the q projection, and a window that
// starts a few microseconds ahead of its consumer reads the same lines
// beside it (measured level on the fabric, 2026-10-01).
void Dsv4AttnLayer::open_compress_window(cudaStream_t stream) {
  if (windows() && w_.ratio == 0) open_output_window(stream);  // a window-only layer: nothing between q and the attention
}

void Dsv4AttnLayer::open_output_window(cudaStream_t stream) {
  if (!windows()) return;
  prefetch_->open_window(stream, kWindowBudget, prefetch_->layer_rate());
  window_fp8(w_.wo_a);
  window_fp8(w_.wo_b);
}

void Dsv4AttnLayer::project_q_kv(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream,
                                 const RingAppend* ring) {
  const uint16_t* h = static_cast<const uint16_t*>(hidden_in);
  const int lh = cfg_.local_heads();
  dense(kViewWqA, h, size_t(cfg_.hidden), w_.wq_a.payload, w_.wq_a.scales, qr_, tokens, cfg_.q_lora, cfg_.hidden, 0,
        stream);
  csa2_rmsnorm_bf16(qr_, cfg_.q_lora, w_.q_norm, qr_, cfg_.q_lora, tokens, cfg_.q_lora, cfg_.eps, stream);
  project_kv(hidden_in, tokens, pos, stream, ring);
  open_compress_window(stream);
  dense(kViewWqB, qr_, size_t(cfg_.q_lora), w_.wq_b.payload, w_.wq_b.scales, q_, tokens, lh * kDsv4Latent, cfg_.q_lora,
        0, stream);
  dsv4_q_head_rmsnorm_rope(q_, tokens, lh, cfg_.eps, pos, w_.inv_freq, stream);
}

void Dsv4AttnLayer::indexer_query(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream) {
  const int heads = cfg_.index_heads;
  dense(kViewIdxWqB, qr_, size_t(cfg_.q_lora), w_.idx_wq_b.payload, w_.idx_wq_b.scales, idx_q_, tokens,
        heads * kDsv4IndexDim, cfg_.q_lora, 0, stream);
  open_output_window(stream);  // decode: behind the indexer's projection, under the selection and the attention
  csa2_rope_apply(idx_q_ + (kDsv4IndexDim - kDsv4Rope), int64_t(heads) * kDsv4IndexDim, kDsv4IndexDim, heads, kDsv4Rope,
                  pos, w_.inv_freq, false, tokens, stream);
  dsv4_hadamard128(idx_q_, int64_t(tokens) * heads, stream);
  csa2_index_q_quant(idx_q_, tokens, heads, q_fp8_, q_scale_, violations_, stream);
  if (prefill_tiles_ && dense_mma_bf16w_shape_ok(w_.idx_wp, cfg_.hidden))
    launch_dense_mma_bf16w_bf16(static_cast<const uint16_t*>(hidden_in), size_t(cfg_.hidden), w_.idx_wp, iw_, tokens,
                                heads, cfg_.hidden, stream);
  else
    gemm_.matmul(hidden_in, w_.idx_wp, iw_, tokens, heads, cfg_.hidden, DType::BF16, GemmOut::BF16,
                 size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
  dsv4_fold_weights(iw_, q_scale_, w_folded_, int64_t(tokens) * heads, index_weight_scale_, stream);
}

// A compressor's per-token projections of `tokens` rows: comp_kv_ /
// comp_score_ fp32 [tokens, wide] (bf16 weights, fp32 accumulation — the
// reference promotes both to fp32).
void Dsv4AttnLayer::compress_project(const void* hidden_in, const Dsv4CompressorResident& c, int tokens,
                                     cudaStream_t stream) {
  // A prefill chunk: the dense tensor-core GEMM over the bf16 weights (a
  // row-count-invariant chain, like the fp8 projections' — dense()); the
  // decode walks keep the streaming form.
  if (prefill_tiles_ && dense_mma_bf16w_shape_ok(c.wkv, cfg_.hidden) && dense_mma_bf16w_shape_ok(c.wgate, cfg_.hidden)) {
    const uint16_t* h = static_cast<const uint16_t*>(hidden_in);
    launch_dense_mma_bf16w_f32(h, size_t(cfg_.hidden), c.wkv, comp_kv_, tokens, c.wide, cfg_.hidden, stream);
    launch_dense_mma_bf16w_f32(h, size_t(cfg_.hidden), c.wgate, comp_score_, tokens, c.wide, cfg_.hidden, stream);
    return;
  }
  gemm_.matmul(hidden_in, c.wkv, comp_kv_, tokens, c.wide, cfg_.hidden, DType::BF16, GemmOut::F32, size_t(cfg_.hidden),
               gemm_ws_, gemm_ws_bytes_, stream);
  gemm_.matmul(hidden_in, c.wgate, comp_score_, tokens, c.wide, cfg_.hidden, DType::BF16, GemmOut::F32,
               size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
}

void Dsv4AttnLayer::attend_rows(Dsv4StatePool& pool, const int32_t* req_ids_win, const uint8_t* win_cache,
                                int win_block_tokens, const int32_t* win_table, int win_blocks,
                                const int32_t* req_ids_main, const int64_t* pos, int row0, int rows, int n_split_main,
                                cudaStream_t stream,
                                const int32_t* list, int list_stride, const int32_t* counts, bool split_window) {
  attend_window(req_ids_win, win_cache, win_block_tokens, win_table, win_blocks, row0, rows, n_split_main, stream, list,
                list_stride, counts, split_window);
  attend_main(pool, req_ids_main, pos, row0, rows, n_split_main, stream, split_window);
}

void Dsv4AttnLayer::attend_window(const int32_t* req_ids_win, const uint8_t* win_cache, int win_block_tokens,
                                  const int32_t* win_table, int win_blocks, int row0, int rows, int n_split_main,
                                  cudaStream_t stream, const int32_t* list, int list_stride, const int32_t* counts,
                                  bool split_window) {
  const int lh = cfg_.local_heads();
  if (rows > ws_win_rows_ || rows * n_split_main > ws_slots_) throw std::logic_error("dsv4 attention: attention tile too wide");
  const uint16_t* q = q_ + size_t(row0) * lh * kDsv4Latent;
  // The window source: one split per row tile at prefill; at decode a fixed
  // split across the keys (a one-row window would otherwise be a single
  // latency-bound block per head group) — a different fp32 combine order
  // than the prefill's, the rounding class the decode audit certifies.
  const int n_split_win = split_window ? std::min(n_split_main, kWinDecodeSplit) : 1;
  // The tensor-core listed kernel where the head tiling takes it (16-head
  // groups: every deployment world), else the split kernel — the same
  // partial layout; a row's partials do not depend on the rows beside it.
  if (!(lh >= 16 && lh % 16 == 0 &&
        dsa_attn_listed(q, win_cache, req_ids_win, list + size_t(row0) * list_stride, list_stride, counts + row0, rows,
                        n_split_win, lh, kDsv4Latent, win_block_tokens, win_table, win_blocks, attn_scale_, m_win_,
                        l_win_, c_win_, stream, kRowFormat, nullptr, 0)))
    dsa_attn_partial(q, win_cache, req_ids_win, list + size_t(row0) * list_stride, list_stride, counts + row0, rows,
                     n_split_win, lh, kDsv4Latent, win_block_tokens, win_table, win_blocks, attn_scale_, m_win_, l_win_,
                     c_win_, stream, kRowFormat, nullptr, 0);
}

void Dsv4AttnLayer::attend_main(Dsv4StatePool& pool, const int32_t* req_ids_main, const int64_t* pos, int row0, int rows,
                                int n_split_main, cudaStream_t stream, bool split_window) {
  const int lh = cfg_.local_heads();
  const uint16_t* q = q_ + size_t(row0) * lh * kDsv4Latent;
  const int n_split_win = split_window ? std::min(n_split_main, kWinDecodeSplit) : 1;
  int n_main = 0;
  if (w_.ratio > 0) {
    const int ord = w_.cache_ord;
    const int epb = pool.entries_per_block(ord);
    const int32_t* mlist;
    const int32_t* mcounts;
    int stride;
    if (w_.ratio == 4) {
      mlist = topk_ + size_t(row0) * cfg_.index_topk;
      mcounts = counts_ + row0;
      stride = cfg_.index_topk;
    } else {
      // Every compressed entry the row sees: the tile's dense lists.
      if (rows > list_rows_) throw std::logic_error("dsv4 attention: dense list tile too wide");
      dsv4_dense_list(pos_sel_ + row0, rows, dense_stride_, mlist_, counts_ + row0, stream);
      mlist = mlist_;
      mcounts = counts_ + row0;
      stride = dense_stride_;
    }
    n_main = n_split_main;
    if (!(lh >= 16 && lh % 16 == 0 &&
          dsa_attn_listed(q, pool.main(ord), req_ids_main, mlist, stride, mcounts, rows, n_main, lh, kDsv4Latent, epb,
                          pool.block_tables(), int(pool.total_blocks()), attn_scale_, m_main_, l_main_, c_main_, stream,
                          kRowFormat, nullptr, 0)))
      dsa_attn_partial(q, pool.main(ord), req_ids_main, mlist, stride, mcounts, rows, n_main, lh, kDsv4Latent, epb,
                       pool.block_tables(), int(pool.total_blocks()), attn_scale_, m_main_, l_main_, c_main_, stream,
                       kRowFormat, nullptr, 0);
  }
  csa2_attn_finish(n_main ? m_main_ : nullptr, n_main ? l_main_ : nullptr, n_main ? c_main_ : nullptr, n_main, m_win_,
                   l_win_, c_win_, n_split_win, w_.attn_sink, rows, lh, pos, w_.inv_freq,
                   o_ + size_t(row0) * lh * kDsv4Latent, stream);
}

void Dsv4AttnLayer::project_out(int tokens, void* out, cudaStream_t stream) {
  const int lh = cfg_.local_heads(), lg = cfg_.local_groups(), hpg = cfg_.heads_per_group();
  const int K = hpg * kDsv4Latent;
  const size_t grid = size_t(1) << kGrid;
  // wo_a is block-diagonal over the groups: group g projects its own heads.
  // (TRIED 2026-10-01 and reverted: the groups in one block-diagonal
  // launch of the tensor-core GEMV. Level on the fabric, 39.0 ms per
  // depth-3 pass either way: these launches are bound by the L2 stream and
  // the two resident blocks per SM, not by a launch's fixed latency.)
  for (int g = 0; g < lg; ++g) {
    const size_t row_off = size_t(g) * cfg_.o_lora;
    dense(kViewWoA + g, o_ + size_t(g) * K, size_t(lh) * kDsv4Latent, w_.wo_a.payload + row_off * K,
          w_.wo_a.scales + (row_off / grid) * (size_t(K) / grid), oa_ + row_off, tokens, cfg_.o_lora, K,
          size_t(lg) * cfg_.o_lora, stream);
  }
  dense(kViewWoB, oa_, size_t(lg) * cfg_.o_lora, w_.wo_b.payload, w_.wo_b.scales, static_cast<uint16_t*>(out), tokens,
        cfg_.hidden, lg * cfg_.o_lora, 0, stream);
}

// ---- decode ---------------------------------------------------------------------------------------

void Dsv4AttnLayer::enqueue_decode(const void* hidden_in, Dsv4StatePool& pool, const int32_t* req_ids,
                                   const int64_t* pos, int tokens, void* out, cudaStream_t stream, bool prefetch) {
  validate_pool(pool);
  if (tokens <= 0 || tokens > max_decode_rows_) throw std::invalid_argument("dsv4 attention: decode rows out of range");
  if (!hidden_in || !req_ids || !pos || !out) throw std::invalid_argument("dsv4 attention: null buffer");

  decode_walk_ = true;
  prefetch_windows_ = prefetch;
  struct DecodeScope {
    bool& walk;
    bool& windows;
    ~DecodeScope() { walk = windows = false; }
  } decode_scope{decode_walk_, prefetch_windows_};
  // The projections; the window rows into the ring and their window lists
  // ride the K/V tail's launch (this batch's rows are visible to its own
  // queries: the ring is read after the append).
  const RingAppend ring{req_ids, pool.ring_table(), pool.ring(layer_), /*window_lists=*/true};
  project_q_kv(hidden_in, tokens, pos, stream, &ring);

  // A compressing layer's window partials on the side stream, beside the
  // compressor and the selection (they read the queries, the ring and the
  // window lists, all complete here; the partials are the finish's alone).
  const bool window_aside = w_.ratio > 0;
  if (window_aside) {
    DGPP_CUDA_OK(cudaEventRecord(win_fork_, stream));
    DGPP_CUDA_OK(cudaStreamWaitEvent(win_side_, win_fork_));
    attend_window(req_ids, pool.ring(layer_), cfg_.ring_slots, pool.ring_table_wide(), pool.ring_table_wide_blocks(), 0,
                  tokens, decode_n_split_, win_side_, wlist_, cfg_.window, wcounts_, /*split_window=*/true);
    DGPP_CUDA_OK(cudaEventRecord(win_join_, win_side_));
  }
  if (w_.ratio > 0) {
    const int ord = w_.cache_ord;
    const int epb = pool.entries_per_block(ord);
    const int blocks = int(pool.total_blocks());
    // The compressor: every row's projections into the ring, then the
    // entries whose groups complete in this batch.
    // (TRIED 2026-10-01 and reverted: the compressors' chain on a side
    // stream beside the query / window chain, joined before the selection.
    // Bitwise, and slower on the fabric — 41.8 against 41.1 ms per depth-3
    // pass: the walk is DRAM-bound nearly everywhere, and two chains
    // reading at once stretched each other's launches (wq_a 18 -> 40-70 us
    // beside an 8 MB compressor read).)
    const Dsv4CompGeom g = Dsv4StatePool::main_geom(w_.ratio);
    if (pool.lazy(ord)) {
      // The lazy form: the rows' inputs into the ring; a row that completes
      // a group projects the group whole — the prefill's tile chain, so the
      // entry is bitwise a prefill's of the same inputs — and publishes it.
      // A pass that completes none reads no compressor weights (two gated
      // launches return at once), 127 passes in 128 at one row a pass.
      dsv4_u_ring_write(hidden_in, size_t(cfg_.hidden), req_ids, pos, tokens, cfg_.hidden, g.slots, pool.u_ring(ord),
                        stream);
      open_output_window(stream);
      launch_dense_mma_bf16w_groups_f32(pool.u_ring(ord), g.slots, req_ids, pos, tokens, g.ratio, w_.comp.wkv,
                                        comp_kv_, g.wide, cfg_.hidden, stream);
      launch_dense_mma_bf16w_groups_f32(pool.u_ring(ord), g.slots, req_ids, pos, tokens, g.ratio, w_.comp.wgate,
                                        comp_score_, g.wide, cfg_.hidden, stream);
      dsv4_comp_publish_main_lazy(comp_kv_, comp_score_, w_.comp.ape, g, req_ids, pos, tokens, w_.comp.norm, cfg_.eps,
                                  w_.inv_freq, pool.block_tables(), blocks, epb, pool.main(ord), stream);
    } else {
      compress_project(hidden_in, w_.comp, tokens, stream);
      if (w_.ratio != 4) open_output_window(stream);
      dsv4_comp_ring_write(comp_kv_, comp_score_, w_.comp.ape, req_ids, pos, tokens, g, pool.comp_ring(ord), stream);
      dsv4_comp_publish_main_decode(pool.comp_ring(ord), g, req_ids, pos, tokens, w_.comp.norm, cfg_.eps, w_.inv_freq,
                                    pool.block_tables(), blocks, epb, pool.main(ord), stream);
    }
    csa2_entry_positions(pos, pos_sel_, tokens, w_.ratio, stream);
    if (w_.ratio == 4) {
      const Dsv4CompGeom ig = Dsv4StatePool::index_geom(w_.ratio);
      compress_project(hidden_in, w_.idx_comp, tokens, stream);
      dsv4_comp_ring_write(comp_kv_, comp_score_, w_.idx_comp.ape, req_ids, pos, tokens, ig, pool.index_comp_ring(ord),
                           stream);
      dsv4_comp_publish_index_decode(pool.index_comp_ring(ord), ig, req_ids, pos, tokens, w_.idx_comp.norm, cfg_.eps,
                                     w_.inv_freq, pool.block_tables(), blocks, epb, pool.index_k(ord),
                                     pool.index_scale(ord), violations_, stream);
      indexer_query(hidden_in, tokens, pos, stream);
      dsa_select_decode(q_fp8_, w_folded_, req_ids, pos_sel_, tokens, pool.block_tables(), blocks, pool.index_k(ord),
                        pool.index_scale(ord), epb, cfg_.index_heads, kDsv4IndexDim, cfg_.index_topk, 1,
                        cfg_.index_topk, topk_, counts_, select_ws_, max_entries_, counter_ws_, 0, stream, true);
    }
  }
  if (window_aside) {
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream, win_join_));
    attend_main(pool, req_ids, pos, 0, tokens, decode_n_split_, stream, /*split_window=*/true);
  } else {
    attend_rows(pool, req_ids, pool.ring(layer_), cfg_.ring_slots, pool.ring_table_wide(), pool.ring_table_wide_blocks(),
                req_ids, pos, 0, tokens, decode_n_split_, stream, wlist_, cfg_.window, wcounts_, /*split_window=*/true);
  }
  project_out(tokens, out, stream);
}

// ---- the DSpark draft --------------------------------------------------------------------------------

void Dsv4AttnLayer::append_window_rows(const void* x, Dsv4StatePool& pool, const int32_t* req_ids, const int64_t* pos,
                                       int n, cudaStream_t stream) {
  validate_pool(pool);
  if (n <= 0 || n > max_tokens_ || n > cfg_.ring_slots) throw std::invalid_argument("dsv4 attention: window rows out of range");
  if (!x || !req_ids || !pos) throw std::invalid_argument("dsv4 attention: null buffer");
  const RingAppend ring{req_ids, pool.ring_table(), pool.ring(layer_), /*window_lists=*/false};
  project_kv(x, n, pos, stream, &ring);
}

void Dsv4AttnLayer::enqueue_draft_block(const void* hidden_in, Dsv4StatePool& pool, const int32_t* req_ids,
                                        const int64_t* pos, int rows, int block, void* out, cudaStream_t stream) {
  validate_pool(pool);
  if (w_.ratio != 0) throw std::logic_error("dsv4 attention: the draft block runs on a window-only layer");
  if (block <= 0 || block > kMaxDraftBlock || rows <= 0 || rows % block != 0 || rows > max_decode_rows_)
    throw std::invalid_argument("dsv4 attention: draft block rows out of range");
  if (cfg_.ring_slots < cfg_.window + block) throw std::invalid_argument("dsv4 attention: the ring must hold window + block");
  if (!hidden_in || !req_ids || !pos || !out) throw std::invalid_argument("dsv4 attention: null buffer");
  decode_walk_ = true;
  prefetch_windows_ = true;
  struct DecodeScope {
    bool& walk;
    bool& windows;
    ~DecodeScope() { walk = windows = false; }
  } decode_scope{decode_walk_, prefetch_windows_};
  project_q_kv(hidden_in, rows, pos, stream);
  csa2_ring_slot_positions(pos, slots_, rows, cfg_.ring_slots, stream);
  dsa_latent_append(kv_, req_ids, slots_, rows, pool.ring_table(), 1, cfg_.ring_slots, pool.ring(layer_), kDsv4Latent,
                    stream, kRowFormat);
  csa2_dspark_window_slots(pos, rows, block, cfg_.window, cfg_.ring_slots, dlist_, dcounts_, stream);
  attend_rows(pool, req_ids, pool.ring(layer_), cfg_.ring_slots, pool.ring_table_wide(), pool.ring_table_wide_blocks(),
              req_ids, pos, 0, rows, decode_n_split_, stream, dlist_, cfg_.window + block, dcounts_,
              /*split_window=*/true);  // the decode rows' listed flash form, split like them (2026-10-20; the scalar
                                       // kernel took 131-170 us a draft layer on DeepSeek-V4.1 against ~20)
  project_out(rows, out, stream);
}

// ---- prefill --------------------------------------------------------------------------------------

void Dsv4AttnLayer::stage_prefill_rows(int req, int64_t pos0, int tokens, cudaStream_t stream) {
  // Host staging (prefill is never captured): positions and request ids.
  host_i64_.assign(static_cast<size_t>(tokens), 0);
  host_i32_.assign(static_cast<size_t>(tokens), req);
  for (int i = 0; i < tokens; ++i) host_i64_[static_cast<size_t>(i)] = pos0 + i;
  DGPP_CUDA_OK(cudaMemcpyAsync(pos_, host_i64_.data(), static_cast<size_t>(tokens) * 8, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(req_ids_, host_i32_.data(), static_cast<size_t>(tokens) * 4, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));  // the staging vectors are reused by the callers
}

void Dsv4AttnLayer::enqueue_prefill(const void* hidden_in, Dsv4StatePool& pool, int req, int64_t pos0, int tokens,
                                    void* out, cudaStream_t stream) {
  validate_pool(pool);
  if (tokens <= 0 || tokens > max_tokens_) throw std::invalid_argument("dsv4 attention: prefill rows out of range");
  if (req < 0 || req >= pool.shape().max_requests) throw std::invalid_argument("dsv4 attention: request out of range");
  if (pos0 < 0 || pos0 % cfg_.block_tokens != 0)
    throw std::invalid_argument("dsv4 attention: a prefill chunk starts on a block boundary");
  if (pool.request_blocks(req) * cfg_.block_tokens < pos0 + tokens)
    throw std::invalid_argument("dsv4 attention: the request's blocks do not cover the chunk");
  if (!hidden_in || !out) throw std::invalid_argument("dsv4 attention: null buffer");
  stage_prefill_rows(req, pos0, tokens, stream);
  upload_dense_views(tokens, stream);
  prefill_tiles_ = true;
  struct TilesOff {
    bool& flag;
    int& live;
    ~TilesOff() {
      flag = false;
      live = 0;
    }
  } tiles_off{prefill_tiles_, dense_views_live_};

  project_q_kv(hidden_in, tokens, pos_, stream);
  // The window scratch: the ring's last window - 1 rows, then the chunk's
  // rows as a one-block cache (scratch row window - 1 + i).
  uint8_t* ring = pool.ring(layer_) + size_t(req) * pool.ring_bytes_per_request();
  const size_t rb = Dsv4StatePool::kRowBytes;
  const int win_rows = cfg_.window - 1 + tokens;
  csa2_window_scratch_prologue(ring, cfg_.ring_slots, pos0, cfg_.window, rb, wscratch_, stream, 0);
  csa2_scaled_positions(iota_, scratch_pos_, tokens, 1, cfg_.window - 1, stream);
  dsa_latent_append(kv_, req_zero_, scratch_pos_, tokens, one_block_, 1, win_rows, wscratch_, kDsv4Latent, stream,
                    kRowFormat);
  csa2_window_slots_prefill(pos0, tokens, cfg_.window, wlist_, wcounts_, stream, 0);

  if (w_.ratio > 0) {
    const int ord = w_.cache_ord;
    const int epb = pool.entries_per_block(ord);
    const int32_t* table = pool.block_tables() + size_t(req) * pool.total_blocks();
    // The compressor: the chunk's complete groups published (an overlapping
    // first group reads the rows before the chunk off the ring), then the
    // chunk's last rows into the ring for the groups to come.
    const auto publish = [&](const Dsv4CompressorResident& c, const Dsv4CompGeom& g, float* rings, bool index) {
      compress_project(hidden_in, c, tokens, stream);
      // (A lazily pooled compressor has no projection ring and, without
      // overlap, reads none.)
      float* ring_req = rings != nullptr ? rings + size_t(req) * g.ring_elems() : nullptr;
      if (index)
        dsv4_comp_publish_index_prefill(comp_kv_, comp_score_, c.ape, ring_req, g, pos0, tokens, c.norm, cfg_.eps,
                                        w_.inv_freq, table, epb, pool.index_k(ord), pool.index_scale(ord), violations_,
                                        stream);
      else
        dsv4_comp_publish_main_prefill(comp_kv_, comp_score_, c.ape, ring_req, g, pos0, tokens, c.norm, cfg_.eps,
                                       w_.inv_freq, table, epb, pool.main(ord), stream);
      if (rings == nullptr) {
        // The lazy form: the open group's inputs (a chunk starts on a
        // block, a whole number of groups) for the decode walk to finish.
        const int n = tokens % g.ratio;
        const size_t r0 = size_t(tokens - n);
        if (n > 0)
          dsv4_u_ring_write(static_cast<const uint16_t*>(hidden_in) + r0 * size_t(cfg_.hidden), size_t(cfg_.hidden),
                            req_ids_ + r0, pos_ + r0, n, cfg_.hidden, g.slots, pool.u_ring(ord), stream);
        return;
      }
      const int n = std::min(tokens, g.slots);
      const size_t r0 = size_t(tokens - n);
      dsv4_comp_ring_write(comp_kv_ + r0 * g.wide, comp_score_ + r0 * g.wide, c.ape, req_ids_ + r0, pos_ + r0, n, g,
                           rings, stream);
    };
    const bool lazy = pool.lazy(ord);
    if (lazy && !(dense_mma_bf16w_shape_ok(w_.comp.wkv, cfg_.hidden) && dense_mma_bf16w_shape_ok(w_.comp.wgate, cfg_.hidden)))
      throw std::logic_error("dsv4 attention: a lazily pooled compressor's weights do not take the tile GEMM");
    publish(w_.comp, Dsv4StatePool::main_geom(w_.ratio), lazy ? nullptr : pool.comp_ring(ord), false);
    csa2_entry_positions(pos_, pos_sel_, tokens, w_.ratio, stream);
    if (w_.ratio == 4) {
      publish(w_.idx_comp, Dsv4StatePool::index_geom(w_.ratio), pool.index_comp_ring(ord), true);
      indexer_query(hidden_in, tokens, pos_, stream);
      // The visible entries of the chunk's last row, gathered contiguous
      // for the dot GEMM (the padded tail zeroed once for the sanitizer).
      const int64_t n_gather = (pos0 + tokens) / w_.ratio;
      const int64_t padded_n = n_gather > 0 ? round_up_to(n_gather, kEntryPad) : 0;
      if (padded_n > max_entries_) throw std::invalid_argument("dsv4 attention: the context exceeds max_cache_tokens");
      if (padded_n > gather_zeroed_) {
        DGPP_CUDA_OK(cudaMemsetAsync(gather_k_ + gather_zeroed_ * kDsv4IndexDim, 0,
                                     size_t(padded_n - gather_zeroed_) * kDsv4IndexDim, stream));
        DGPP_CUDA_OK(cudaMemsetAsync(gather_scale_ + gather_zeroed_, 0, size_t(padded_n - gather_zeroed_) * 4, stream));
        gather_zeroed_ = padded_n;
      }
      if (n_gather > 0)
        dsa_gather_index_pools(table, epb, pool.index_k(ord), pool.index_scale(ord), n_gather, gather_k_, gather_scale_,
                               kDsv4IndexDim, stream);
      // The select tile: as many rows as the dot scratch holds at THIS
      // chunk's entry count. (Until 2026-10-01 the tile was tile_cap_, the
      // rows the scratch holds at max_cache_tokens — four at the release's
      // 1M context — whatever the chunk saw: a 2K prompt's 525 entries ran
      // 512 four-row tiles per indexed layer, three launches each, a fifth
      // of the prefill in select launches alone.) A row's selection does
      // not depend on the rows beside it.
      const int tile_rows =
          padded_n > 0 ? int(std::min<int64_t>(tokens, std::max<int64_t>(tile_cap_, int64_t(tile_cap_) * max_entries_ / padded_n)))
                       : tile_cap_;
      for (int row0 = 0; row0 < tokens; row0 += tile_rows) {
        const int rows = std::min(tile_rows, tokens - row0);
        if (padded_n > 0) {
          gemm_.matmul(q_fp8_ + size_t(row0) * cfg_.index_heads * kDsv4IndexDim, gather_k_, dot_,
                       rows * cfg_.index_heads, int(padded_n), kDsv4IndexDim, DType::F8_E4M3, GemmOut::F32,
                       size_t(kDsv4IndexDim), gemm_ws_, gemm_ws_bytes_, stream);
          csa2_logits_prefill(dot_, padded_n, w_folded_ + size_t(row0) * cfg_.index_heads, gather_scale_,
                              pos_sel_ + row0, rows, n_gather, cfg_.index_heads, logits_, padded_n, stream);
        }
        const int64_t stride = std::max<int64_t>(padded_n, 1);
        dbg_logits_rows_ = padded_n > 0 ? rows : 0;
        dbg_logits_stride_ = stride;
        dbg_logits_entries_ = n_gather;
        csa2_select_rows_prefill(logits_, stride, pos_sel_ + row0, rows, cfg_.index_topk, nullptr, 0, nullptr, 1,
                                 topk_ + size_t(row0) * cfg_.index_topk, counts_ + row0, stream);
      }
    }
  }
  for (int row0 = 0; row0 < tokens; row0 += kPrefillAttnRows) {
    const int rows = std::min(kPrefillAttnRows, tokens - row0);
    attend_rows(pool, req_zero_, wscratch_, win_rows, one_block_, 1, req_ids_, pos_ + row0, row0, rows, kPrefillSplit,
                stream, wlist_, cfg_.window, wcounts_, false);
  }
  csa2_window_ring_writeback(wscratch_, cfg_.window, pos0, tokens, cfg_.ring_slots, rb, ring, stream);
  project_out(tokens, out, stream);
}

}  // namespace dgpp
