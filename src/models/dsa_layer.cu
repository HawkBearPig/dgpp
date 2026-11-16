#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/dsa.hpp"
#include "kernels/packq_gemm.hpp"
#include "kernels/packq_gemv.hpp"
#include "kernels/scale_gemm.hpp"
#include "models/dsa_layer.hpp"
#include "models/dsa_state.hpp"

namespace dgpp {

namespace {

// Dot-GEMM n granularity, in pools: plans are keyed per multiple of 256
// pools instead of per exact pool count, bounding the plan cache over
// arbitrary chunk boundaries (a 300k-token context yields ~293 shapes).
constexpr int64_t kPoolPadGranularity = 256;

size_t align256(size_t bytes) { return (bytes + 255) / 256 * 256; }

int64_t round_up_to(int64_t v, int64_t gran) {
  return (v + gran - 1) / gran * gran;
}

bool fused_prefill_scores(const DsaConfig& cfg, int64_t padded_pools) {
  // Full GLM's unpooled indexer. Preserve the Flash and short-context
  // paths while avoiding the full model's 32-head global dot buffer.
  return cfg.index_kpool == 1 && cfg.qk_rope_head_dim == 64 &&
         cfg.index_topk == 2048 && padded_pools > 2048;
}

}  // namespace

// ---------------------------------------------------------------------
// Scratch layout
// ---------------------------------------------------------------------

DsaLayer::ScratchLayout DsaLayer::scratch_layout(
    const DsaConfig& cfg, int max_tokens, int64_t max_cache_tokens,
    int max_decode_rows, int decode_n_split, size_t dot_budget) {
  DsaConfig::validate_config(cfg);
  if (cfg.index_n_heads != 32)
    throw std::invalid_argument(
        "dsa layer: the selection kernels pin index_n_heads to 32");
  // The bitonic select/expand networks are power-of-two sized: a
  // non-power-of-two select_k (index_topk/kpool) indexes past the smem
  // windows and silently drops every selected pool. The real checkpoints'
  // 2048/4 = 512 (Flash) and 2048/1 (the full model, the expansion's
  // bound) satisfy this; fail loudly rather than mis-select.
  {
    const int select_k = cfg.index_topk / cfg.index_kpool;
    if (select_k <= 0 || (select_k & (select_k - 1)) != 0)
      throw std::invalid_argument(
          "dsa layer: select_k (= index_topk / index_kpool) must be a "
          "power of two (bitonic select networks)");
    if (select_k > kDsaSelectMaxK)
      throw std::invalid_argument(
          "dsa layer: select_k exceeds the select kernels' bound (2048)");
  }
  // The attention kernel's head-group tiling (hpb = min(heads, 16) heads
  // per block, 128/hpb dim groups) is exercised for 4/8/64 heads; 1-2
  // heads route to a 64-group partition that mis-computes (found by the
  // M4 forward fixture — heads=2 produced 1e33-scale garbage with
  // selections matching). Reject rather than silently corrupt.
  if (cfg.num_heads < 4 || (cfg.num_heads & (cfg.num_heads - 1)) != 0)
    throw std::invalid_argument(
        "dsa layer: num_heads must be a power of two >= 4 (attention "
        "head-group tiling)");
  if (max_tokens <= 0 || max_cache_tokens <= 0)
    throw std::invalid_argument(
        "dsa layer: max_tokens and max_cache_tokens must be positive");
  // 32 since 2026-10-13 (16 from 2026-09-13; the decode batch's cap, eight
  // slots at depth 3): the fused select runs the rows in groups of eight
  // over a workspace laid out for 32, the attention tiles and the (row,
  // split) workspace below scale with the count.
  if (max_decode_rows <= 0 || max_decode_rows > 32)
    throw std::invalid_argument("dsa layer: max_decode_rows must be in [1, 32]");
  if (max_decode_rows > max_tokens)
    throw std::invalid_argument(
        "dsa layer: max_decode_rows must not exceed max_tokens");
  if (decode_n_split <= 0)
    throw std::invalid_argument("dsa layer: decode_n_split must be positive");
  if (dot_budget == 0)
    throw std::invalid_argument("dsa layer: dot_budget must be positive");

  const DsaGeometry g = DsaGeometry::from_config(cfg);
  const int heads = cfg.index_n_heads;
  const int dim = cfg.index_head_dim;
  // Attention tile rows: the decode/split tiles (max_decode_rows, at
  // least 8) or the dense prefill tiles (kDensePrefillRows), whichever is
  // larger, for q_tilde/c; the (row, split) workspace covers the wider of
  // the split path (8 rows x its split) and the dense path (its rows x
  // its split).
  const int A = std::max(std::max(max_decode_rows, 8), DsaLayer::kDensePrefillRows);
  const int split_cap = std::max(decode_n_split, 8);
  const size_t ws_slots = std::max(
      size_t(std::max(max_decode_rows, 8)) * size_t(split_cap),
      size_t(DsaLayer::kDensePrefillRows) * size_t(DsaLayer::kDensePrefillSplit));

  const int64_t max_pools = round_up_to(
      (max_cache_tokens + g.kpool - 1) / g.kpool, kPoolPadGranularity);
  int tile_cap =
      int(dot_budget / (size_t(heads) * size_t(max_pools) * 4));
  if (tile_cap <= 0) tile_cap = 1;  // degenerate budget: 1-row tiles
  tile_cap = std::min(tile_cap, max_tokens);

  ScratchLayout L;
  size_t off = 0;
  const auto alloc = [&](size_t bytes) {
    off = align256(off);
    const size_t at = off;
    off += bytes;
    return at;
  };
  const size_t T = size_t(max_tokens);
  const size_t Ta = std::max(T, size_t(A));
  L.off_qkv = alloc(T * size_t(cfg.q_lora_rank + cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 2);
  L.off_q_c = alloc(T * size_t(cfg.q_lora_rank) * 2);
  L.off_kv_c = alloc(T * size_t(cfg.kv_lora_rank) * 2);
  L.off_k_rot = alloc(T * size_t(cfg.qk_rope_head_dim) * 2);
  L.off_q = alloc(T * size_t(g.local_q_rows) * 2);
  L.off_q_idx = alloc(T * size_t(heads) * size_t(dim) * 2);
  L.off_k_raw = alloc(T * size_t(dim) * 2);
  L.off_k_rows = alloc(T * size_t(dim) * 2);
  L.off_gate = alloc(T * size_t(dim) * 2);
  L.off_weights = alloc(T * size_t(heads) * 4);
  L.off_q_fp8 = alloc(T * size_t(heads) * size_t(dim));
  L.off_q_scale = alloc(T * size_t(heads) * 4);
  L.off_w_folded = alloc(T * size_t(heads) * 4);
  L.off_topk = alloc(T * size_t(g.max_selected) * 4);
  L.off_counts = alloc(T * 4);
  L.off_pos = alloc(T * 8);
  L.off_req_ids = alloc(Ta * 4);
  L.off_attn_out = alloc(T * size_t(g.local_v_rows) * 2);
  L.off_q_tilde = alloc(size_t(A) * size_t(g.local_heads) *
                        size_t(g.score_width) * 2);
  L.off_c = alloc(size_t(A) * size_t(g.local_heads) *
                  size_t(cfg.kv_lora_rank) * 4);
  L.off_m = alloc(ws_slots * size_t(g.local_heads) * 4);
  L.off_l = alloc(ws_slots * size_t(g.local_heads) * 4);
  L.off_cws = alloc(ws_slots * size_t(g.local_heads) *
                    size_t(cfg.kv_lora_rank) * 4);
  L.off_dot = alloc(size_t(tile_cap) * size_t(heads) * size_t(max_pools) * 4);
  L.off_gather_k = alloc(size_t(max_pools) * size_t(dim));
  L.off_gather_scale = alloc(size_t(max_pools) * 4);
  // Decode and prefill do not overlap. The compact prefill keys occupy
  // 1/16 of the dot buffer and reuse the decode selector's workspace.
  L.off_select_ws = alloc(std::max(
      dsa_select_workspace_bytes(max_decode_rows, max_pools),
      dsa_select_prefill_workspace_bytes(tile_cap, max_pools, g.select_k)));
  L.off_counter = alloc(8);  // int32 [2]: the select ticket and rows-done
  // The staged copy of a listed prefill tile's distinct rows: a quantized
  // cache at the listed kernel's geometries (a bf16 cache is its own
  // decoded copy). Never more rows than a tile can list or a request hold.
  {
    const bool tail = cfg.qk_rope_head_dim != 0;
    const bool geometry = tail ? (cfg.kv_lora_rank == 512 && cfg.qk_rope_head_dim == 64)
                               : (cfg.kv_lora_rank == 512 || cfg.kv_lora_rank == 256);
    if (cfg.latent_format != LatentFormat::kBf16 && geometry && g.local_heads >= 16 && g.local_heads % 16 == 0) {
      L.stage_slots = dsa_attn_stage_slots(
          g.score_width, std::min<int64_t>(int64_t(DsaLayer::kDensePrefillRows) * g.max_selected, max_cache_tokens));
      L.off_stage = alloc(dsa_attn_stage_bytes(max_cache_tokens, L.stage_slots, g.score_width));
    }
  }
  L.total = align256(off);
  L.max_pools = max_pools;
  L.tile_cap = tile_cap;
  return L;
}

// ---------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------

DsaLayer::DsaLayer(IGemm& gemm, const DsaLayerWeights& w, const DsaConfig& cfg,
                   int max_tokens, int64_t max_cache_tokens, void* scratch,
                   size_t scratch_capacity, void* gemm_workspace,
                   size_t gemm_ws_bytes, int max_decode_rows,
                   int decode_n_split, size_t dot_budget)
    : gemm_(gemm),
      w_(w),
      cfg_(cfg),
      geo_(DsaGeometry::from_config(cfg)),
      max_tokens_(max_tokens),
      max_cache_tokens_(max_cache_tokens),
      max_decode_rows_(max_decode_rows),
      decode_n_split_(decode_n_split),
      decode_mma_([] {
        const char* e = std::getenv("DGPP_DSA_DECODE_MMA");
        return !(e != nullptr && std::string(e) == "off");
      }()),
      gemm_ws_(gemm_workspace),
      gemm_ws_bytes_(gemm_ws_bytes) {
  const ScratchLayout L =
      scratch_layout(cfg, max_tokens, max_cache_tokens, max_decode_rows,
                     decode_n_split, dot_budget);
  if (!scratch)
    throw std::invalid_argument("dsa layer: scratch buffer required");
  if (scratch_capacity < L.total)
    throw std::invalid_argument(
        "dsa layer: scratch buffer too small (" +
        std::to_string(scratch_capacity) + " < " + std::to_string(L.total) +
        "; use scratch_bytes()");
  if (!gemm_workspace || gemm_ws_bytes == 0)
    throw std::invalid_argument("dsa layer: GEMM workspace required");
  validate_weights(w_);

  attn_rows_ = std::max(max_decode_rows, 8);
  tile_cap_ = L.tile_cap;
  max_pools_ = L.max_pools;

  // Reference computes both scales in python float64 and rounds to fp32;
  // reproduce the double-precision products exactly (1/64 and 1/16 here).
  logit_scale_ = static_cast<float>(
      std::pow(double(cfg.index_head_dim), -0.5) *
      std::pow(double(cfg.index_n_heads), -0.5));
  // qk_head_dim^-0.5: nope^-0.5 for Flash (rope 0), (nope + rope)^-0.5 =
  // 1/16 for the full model.
  attn_scale_ = 1.0f / std::sqrt(float(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim));

  scratch_ = static_cast<uint8_t*>(scratch);
  const auto at = [&](size_t o) { return scratch_ + o; };
  qkv_ = reinterpret_cast<uint16_t*>(at(L.off_qkv));
  q_c_ = reinterpret_cast<uint16_t*>(at(L.off_q_c));
  kv_c_ = reinterpret_cast<uint16_t*>(at(L.off_kv_c));
  k_rot_ = reinterpret_cast<uint16_t*>(at(L.off_k_rot));
  q_ = reinterpret_cast<uint16_t*>(at(L.off_q));
  q_idx_ = reinterpret_cast<uint16_t*>(at(L.off_q_idx));
  k_raw_ = reinterpret_cast<uint16_t*>(at(L.off_k_raw));
  k_rows_ = reinterpret_cast<uint16_t*>(at(L.off_k_rows));
  gate_rows_ = reinterpret_cast<uint16_t*>(at(L.off_gate));
  weights_ = reinterpret_cast<float*>(at(L.off_weights));
  q_fp8_ = at(L.off_q_fp8);
  q_scale_ = reinterpret_cast<float*>(at(L.off_q_scale));
  w_folded_ = reinterpret_cast<float*>(at(L.off_w_folded));
  topk_ = reinterpret_cast<int32_t*>(at(L.off_topk));
  counts_ = reinterpret_cast<int32_t*>(at(L.off_counts));
  pos_dev_ = reinterpret_cast<int64_t*>(at(L.off_pos));
  req_ids_dev_ = reinterpret_cast<int32_t*>(at(L.off_req_ids));
  attn_out_ = reinterpret_cast<uint16_t*>(at(L.off_attn_out));
  q_tilde_ = reinterpret_cast<uint16_t*>(at(L.off_q_tilde));
  c_ = reinterpret_cast<float*>(at(L.off_c));
  m_ws_ = reinterpret_cast<float*>(at(L.off_m));
  l_ws_ = reinterpret_cast<float*>(at(L.off_l));
  c_ws_ = reinterpret_cast<float*>(at(L.off_cws));
  dot_ = reinterpret_cast<float*>(at(L.off_dot));
  gather_k_ = at(L.off_gather_k);
  gather_scale_ = reinterpret_cast<float*>(at(L.off_gather_scale));
  select_ws_ = at(L.off_select_ws);
  select_ws_pools_ = L.max_pools;
  counter_ws_ = reinterpret_cast<int32_t*>(at(L.off_counter));
  stage_slots_ = L.stage_slots;
  stage_ws_ = L.stage_slots > 0 ? at(L.off_stage) : nullptr;
}

size_t DsaLayer::scratch_bytes(const DsaConfig& cfg, int max_tokens,
                               int64_t max_cache_tokens, int max_decode_rows,
                               int decode_n_split, size_t dot_budget) {
  return scratch_layout(cfg, max_tokens, max_cache_tokens, max_decode_rows,
                        decode_n_split, dot_budget)
      .total;
}

// The weight view's contract: the indexer complete or absent as a whole
// (absent = the selection-reusing form), the projections in exactly one
// of the three forms with the local geometry, the rope table when the
// geometry has a tail.
void DsaLayer::validate_weights(const DsaLayerWeights& w) const {
  const auto require = [](const void* p, const char* what) {
    if (!p)
      throw std::invalid_argument(std::string("dsa layer: missing ") + what);
  };
  if (w.owns_indexer()) {
    require(w.wk, "wk");
    require(w.wp, "wp");
    require(w.k_norm_w, "k_norm_w");
    require(w.k_norm_b, "k_norm_b");
    if (cfg_.index_kpool > 1) {
      require(w.gate, "gate");
      require(w.ape, "ape");
    }
  } else if (w.wk || w.wp || w.gate || w.k_norm_w || w.k_norm_b || w.ape) {
    throw std::invalid_argument(
        "dsa layer: a selection-reusing view carries no indexer tensors");
  } else if (cfg_.index_layers() == cfg_.num_dsa_layers) {
    throw std::invalid_argument(
        "dsa layer: every layer of this configuration owns an indexer (wq_b missing)");
  }
  // The quantized projections: the bf16 bridge, the fp8 pairs or the
  // packed-int triples — one form.
  if (w.packed_int()) {
    if (w.quantized())
      throw std::invalid_argument("dsa layer: packed-int and fp8 projections together");
    const auto check = [&](const GlmPackedMatrix& m, int64_t rows, int64_t cols,
                           const char* what) {
      require(m.packed, what);
      require(m.scales, what);
      if (m.rows != rows || m.cols != cols || (m.bits != 4 && m.bits != 8))
        throw std::invalid_argument(std::string("dsa layer: packed geometry of ") + what);
      if (!packq_gemv_accepts(m))
        throw std::invalid_argument(std::string("dsa layer: the packed GEMV core rejects ") +
                                    what + " (K outside the compiled set or misaligned)");
    };
    check(w.qkv_a_p, int64_t(cfg_.q_lora_rank + cfg_.kv_lora_rank + cfg_.qk_rope_head_dim),
          cfg_.hidden, "qkv_a_p");
    check(w.q_b_p, geo_.local_q_rows, cfg_.q_lora_rank, "q_b_p");
    check(w.o_proj_p, cfg_.hidden, geo_.local_v_rows, "o_proj_p");
  } else if (w.quantized()) {
    if (cfg_.qk_rope_head_dim != 0)
      throw std::invalid_argument("dsa layer: the fp8 pair form is the rope-free layout");
    require(w.q_a_q.payload, "q_a_q.payload");
    require(w.q_a_q.scales, "q_a_q.scales");
    require(w.kv_a_q.payload, "kv_a_q.payload");
    require(w.kv_a_q.scales, "kv_a_q.scales");
    require(w.q_b_q.payload, "q_b_q.payload");
    require(w.q_b_q.scales, "q_b_q.scales");
    require(w.o_proj_q.payload, "o_proj_q.payload");
    require(w.o_proj_q.scales, "o_proj_q.scales");
    if (w.q_a_q.rows != cfg_.q_lora_rank || w.q_a_q.cols != cfg_.hidden ||
        w.kv_a_q.rows != cfg_.kv_lora_rank || w.kv_a_q.cols != cfg_.hidden ||
        w.q_b_q.rows != geo_.local_q_rows || w.q_b_q.cols != cfg_.q_lora_rank ||
        w.o_proj_q.rows != cfg_.hidden || w.o_proj_q.cols != geo_.local_v_rows)
      throw std::invalid_argument("DsaLayer: fp8 projection geometry");
  } else {
    require(w.qkv_a, "qkv_a");
    require(w.q_b, "q_b");
    require(w.o_proj, "o_proj");
  }
  require(w.q_aln, "q_aln");
  require(w.kv_aln, "kv_aln");
  require(w.kv_b, "kv_b");
  if (cfg_.qk_rope_head_dim > 0) {
    require(w.rope_table, "rope_table");
    if (w.rope_table_positions < max_cache_tokens_)
      throw std::invalid_argument(
          "dsa layer: the rope table must cover max_cache_tokens positions");
  }
}

void DsaLayer::note_selection(SelKind kind, int rows, int64_t start, int req,
                              const int64_t* pos) {
  SelNote& n = sel_notes_[sel_base_];
  n.kind = kind;
  n.rows = rows;
  n.start = start;
  n.req = req;
  n.pos = pos;
}

void DsaLayer::require_selection(SelKind kind, int rows, int64_t start, int req,
                                 const int64_t* pos) const {
  const auto it = sel_notes_.find(sel_base_);
  if (it == sel_notes_.end() || it->second.kind != kind || it->second.rows != rows ||
      it->second.start != start || it->second.req != req || it->second.pos != pos)
    throw std::logic_error(
        "dsa layer: a selection-reusing enqueue must follow the indexed "
        "layer's enqueue of the same rows, chunk and request on this scratch "
        "(nothing may run between a full layer and its dependants)");
}

// ---------------------------------------------------------------------
// Plan preparation
// ---------------------------------------------------------------------

bool DsaLayer::prepare(int tokens) {
  if (tokens <= 0 || tokens > max_tokens_)
    throw std::invalid_argument("dsa layer: token count out of range");
  const int hid = cfg_.hidden;
  const int heads = cfg_.index_n_heads;
  const int dim = cfg_.index_head_dim;
  const int qkv_cols = cfg_.q_lora_rank + cfg_.kv_lora_rank + cfg_.qk_rope_head_dim;
  // The bf16 projections' plans (the fp8 and packed forms run their own
  // kernels); the indexer's whether or not this view owns one — every
  // view of the layer shares the scratch and the plan cache.
  bool ok = gemm_.ensure_plan(tokens, heads * dim, cfg_.q_lora_rank, DType::BF16,
                              GemmOut::BF16, size_t(cfg_.q_lora_rank)) &&
            gemm_.ensure_plan(tokens, dim, hid, DType::BF16, GemmOut::BF16,
                              size_t(hid)) &&
            gemm_.ensure_plan(tokens, heads, hid, DType::BF16, GemmOut::F32,
                              size_t(hid));
  if (!w_.quantized() && !w_.packed_int())
    ok = ok &&
         gemm_.ensure_plan(tokens, qkv_cols, hid, DType::BF16, GemmOut::BF16,
                           size_t(hid)) &&
         gemm_.ensure_plan(tokens, geo_.local_q_rows, cfg_.q_lora_rank,
                           DType::BF16, GemmOut::BF16,
                           size_t(cfg_.q_lora_rank)) &&
         gemm_.ensure_plan(tokens, hid, geo_.local_v_rows, DType::BF16,
                           GemmOut::BF16, size_t(geo_.local_v_rows));
  // The kernel shared-memory opt-in is context state; do it now so the
  // first enqueue (which may be inside graph capture) never mutates it.
  dsa_prepare_kernel_smem();
  return ok;
}

int DsaLayer::prefill_query_tile_rows(int64_t padded_pools) const {
  // A reduced 64-bit key replaces 32 FP32 dots. Reuse exactly the same
  // allocation for 16 times as many queries; selection scratch is separate.
  const int factor = fused_prefill_scores(cfg_, padded_pools) ? 16 : 1;
  const int64_t row_pool_capacity = int64_t(tile_cap_) * max_pools_ * factor;
  return int(std::min<int64_t>(max_tokens_, row_pool_capacity /
                                           std::max<int64_t>(padded_pools, 1)));
}

bool DsaLayer::prepare_prefill(int tile_rows, int64_t visible_pools) {
  if (visible_pools < 0 || visible_pools > max_pools_)
    throw std::invalid_argument("dsa layer: visible pools out of range");
  const int64_t padded = round_up_to(visible_pools, kPoolPadGranularity);
  if (tile_rows <= 0 || tile_rows > prefill_query_tile_rows(padded))
    throw std::invalid_argument("dsa layer: prefill tile rows out of range");
  if (fused_prefill_scores(cfg_, padded)) return true;
  return gemm_.ensure_plan(tile_rows * cfg_.index_n_heads, int(padded),
                           cfg_.index_head_dim, DType::F8_E4M3, GemmOut::F32,
                           size_t(cfg_.index_head_dim));
}

// ---------------------------------------------------------------------
// Shared projection chain
// ---------------------------------------------------------------------

// The output projection [tokens, local_v] x o_proj^T -> [tokens, hidden]:
// the bf16 bridge through the GEMM interface, or the fp8 pair through the
// scale-aware GEMM (the same dequantized values; the two kernels' fp32
// summation orders differ).
void DsaLayer::project_out(void* out, int tokens, cudaStream_t stream) {
  if (w_.packed_int())
    (tokens >= kPackqMmaFromRows ? launch_packq_gemm_bf16 : launch_packq_gemv_bf16)(
        static_cast<const uint16_t*>(attn_out_), size_t(geo_.local_v_rows), w_.o_proj_p,
        static_cast<uint16_t*>(out), tokens, cfg_.hidden, geo_.local_v_rows, stream);
  else if (w_.quantized())
    launch_scale_gemm_bf16(static_cast<const uint16_t*>(attn_out_),
                           size_t(geo_.local_v_rows), w_.o_proj_q.payload,
                           w_.o_proj_q.scales, static_cast<uint16_t*>(out), tokens,
                           cfg_.hidden, geo_.local_v_rows, stream, 0, cfg_.gemm_mma_from_rows);
  else
    gemm_.matmul(attn_out_, w_.o_proj, out, tokens, cfg_.hidden,
                 geo_.local_v_rows, DType::BF16, GemmOut::BF16,
                 size_t(geo_.local_v_rows), gemm_ws_, gemm_ws_bytes_, stream);
}

void DsaLayer::project_common(const void* hidden_in, int tokens,
                              const int64_t* pos, cudaStream_t stream) {
  const int hid = cfg_.hidden;
  const int heads = cfg_.index_n_heads;
  const int dim = cfg_.index_head_dim;
  const int rope = cfg_.qk_rope_head_dim;
  const int nope = cfg_.qk_nope_head_dim;
  const int qkv_cols = cfg_.q_lora_rank + cfg_.kv_lora_rank + rope;
  // The rotations of this chain (the rope key, q's rope slice, the
  // indexer's q and k) are collected and issued as ONE launch after the
  // GEMVs and norms that feed them (dsa_rope_interleave_multi, bitwise the
  // separate launches): nothing between a rotation's producer and the
  // launch reads its output, so the order on the stream is free.
  DsaRopeSegment rot[kRopeMaxSegments];
  int nrot = 0;
  const auto rotate = [&](void* x, int64_t row_stride, int64_t head_stride,
                          int nheads, void* out, int64_t out_row_stride,
                          int64_t out_head_stride) {
    DsaRopeSegment& g = rot[nrot++];
    g.x = x;
    g.out = out;
    g.x_row_stride = row_stride;
    g.x_head_stride = head_stride;
    g.out_row_stride = out_row_stride;
    g.out_head_stride = out_head_stride;
    g.heads = nheads;
  };
  const auto rotate_all = [&] {
    if (nrot > 0)
      dsa_rope_interleave_multi(rot, nrot, rope, pos, w_.rope_table, w_.rope_table_positions, tokens, stream);
    nrot = 0;
  };

  // 1) fused [q_a | kv_a] projection, then RMSNorms on the split halves.
  // The fp8 form: two scale-aware GEMMs into the two column ranges of the
  // same [tokens, q_lora + kv_lora] buffer (the output row stride). The
  // packed form: one fused triple, with tensor-core lowering for bulk
  // prefill and unchanged GEMV arithmetic for short prompts and decode.
  if (w_.packed_int()) {
    (tokens >= kPackqMmaFromRows ? launch_packq_gemm_bf16 : launch_packq_gemv_bf16)(
        static_cast<const uint16_t*>(hidden_in), size_t(hid), w_.qkv_a_p, qkv_, tokens, qkv_cols,
        hid, stream);
  } else if (w_.quantized()) {
    const uint16_t* h = static_cast<const uint16_t*>(hidden_in);
    uint16_t* qkv = static_cast<uint16_t*>(qkv_);
    launch_scale_gemm_bf16(h, size_t(hid), w_.q_a_q.payload, w_.q_a_q.scales,
                           qkv, tokens, cfg_.q_lora_rank, hid, stream,
                           size_t(qkv_cols), cfg_.gemm_mma_from_rows);
    launch_scale_gemm_bf16(h, size_t(hid), w_.kv_a_q.payload, w_.kv_a_q.scales,
                           qkv + cfg_.q_lora_rank, tokens, cfg_.kv_lora_rank, hid,
                           stream, size_t(qkv_cols), cfg_.gemm_mma_from_rows);
  } else {
    gemm_.matmul(hidden_in, w_.qkv_a, qkv_, tokens, qkv_cols, hid, DType::BF16,
                 GemmOut::BF16, size_t(hid), gemm_ws_, gemm_ws_bytes_, stream);
  }
  dsa_fused_qkv_rmsnorm(qkv_, q_c_, kv_c_, cfg_.q_lora_rank,
                        cfg_.kv_lora_rank, tokens, w_.q_aln, w_.kv_aln,
                        cfg_.rms_norm_eps, stream, qkv_cols);
  // The rope key: the fused row's tail, rotated at the token's position
  // (not normed — the reference's kv_a_layernorm covers the latent only).
  if (rope > 0)
    rotate(qkv_ + cfg_.q_lora_rank + cfg_.kv_lora_rank, qkv_cols, 0, 1, k_rot_,
           rope, 0);
  // 2) MLA q from the normed q-lora rows; its rope slice rotated in place.
  if (w_.packed_int())
    (tokens >= kPackqMmaFromRows ? launch_packq_gemm_bf16 : launch_packq_gemv_bf16)(
        static_cast<const uint16_t*>(q_c_), size_t(cfg_.q_lora_rank), w_.q_b_p, q_, tokens,
        geo_.local_q_rows, cfg_.q_lora_rank, stream);
  else if (w_.quantized())
    launch_scale_gemm_bf16(static_cast<const uint16_t*>(q_c_), size_t(cfg_.q_lora_rank),
                           w_.q_b_q.payload, w_.q_b_q.scales,
                           static_cast<uint16_t*>(q_), tokens, geo_.local_q_rows,
                           cfg_.q_lora_rank, stream, 0, cfg_.gemm_mma_from_rows);
  else
    gemm_.matmul(q_c_, w_.q_b, q_, tokens, geo_.local_q_rows, cfg_.q_lora_rank,
                 DType::BF16, GemmOut::BF16, size_t(cfg_.q_lora_rank), gemm_ws_,
                 gemm_ws_bytes_, stream);
  if (rope > 0)
    rotate(q_ + nope, geo_.local_q_rows, nope + rope, geo_.local_heads, q_ + nope,
           geo_.local_q_rows, nope + rope);
  if (!w_.owns_indexer()) {  // a selection-reusing layer: done
    rotate_all();
    return;
  }
  // 3) the indexer q, also from the normed q-lora rows (its rope slice is
  // the head's first dims), then Hadamard-128 + fp8 quant.
  gemm_.matmul(q_c_, w_.wq_b, q_idx_, tokens, heads * dim, cfg_.q_lora_rank,
               DType::BF16, GemmOut::BF16, size_t(cfg_.q_lora_rank), gemm_ws_,
               gemm_ws_bytes_, stream);
  if (rope > 0)
    rotate(q_idx_, int64_t(heads) * dim, dim, heads, q_idx_, int64_t(heads) * dim, dim);
  // 4) indexer k (LayerNorm, eps 1e-6 — the indexer's own norm, not
  // rms_norm_eps), rotated after the norm, and the gate (kpool > 1), both
  // straight from hidden.
  gemm_.matmul(hidden_in, w_.wk, k_raw_, tokens, dim, hid, DType::BF16,
               GemmOut::BF16, size_t(hid), gemm_ws_, gemm_ws_bytes_, stream);
  dsa_k_layernorm(k_raw_, dim, w_.k_norm_w, w_.k_norm_b, k_rows_, tokens, dim,
                  1e-6f, stream);
  if (rope > 0) rotate(k_rows_, dim, 0, 1, k_rows_, dim, 0);
  if (w_.gate)
    gemm_.matmul(hidden_in, w_.gate, gate_rows_, tokens, dim, hid, DType::BF16,
                 GemmOut::BF16, size_t(hid), gemm_ws_, gemm_ws_bytes_, stream);
  // 5) indexer weights in fp32 with no bf16 rounding (reference pins this):
  // bf16 inputs, fp32 accumulate, fp32 out.
  gemm_.matmul(hidden_in, w_.wp, weights_, tokens, heads, hid, DType::BF16,
               GemmOut::F32, size_t(hid), gemm_ws_, gemm_ws_bytes_, stream);
  rotate_all();  // the chain's rotations, one launch (the quant below reads q_idx rotated)
  // 6) Hadamard-128 + fp8 quant of the indexer q, then fold the q scale and
  // the combined logit scale into the weights.
  dsa_fwht_quant_rows(q_idx_, int64_t(tokens) * heads, q_fp8_, q_scale_,
                      stream);
  dsa_fold_weights(weights_, q_scale_, w_folded_, int64_t(tokens) * heads,
                   logit_scale_, stream);
}

// ---------------------------------------------------------------------
// Attention (shared by both paths)
// ---------------------------------------------------------------------

void DsaLayer::attend_tile(DsaStatePool& state, int layer,
                           const int32_t* req_ids, int64_t row0, int rows,
                           int n_split, cudaStream_t stream) {
  for (int64_t a0 = row0; a0 < row0 + rows; a0 += attn_rows_) {
    const int arows = int(std::min<int64_t>(attn_rows_, row0 + rows - a0));
    dsa_absorb_q(q_ + a0 * geo_.local_q_rows, w_.kv_b, q_tilde_, arows,
                 geo_.local_heads, cfg_.qk_nope_head_dim, cfg_.v_head_dim,
                 cfg_.kv_lora_rank, stream, geo_.rope_dim, /*tensor_cores=*/false);
    dsa_attn_partial(q_tilde_, state.latent(layer), req_ids + a0,
                     topk_ + (sel_base_ + a0) * geo_.max_selected, geo_.max_selected,
                     counts_ + sel_base_ + a0, arows, n_split, geo_.local_heads,
                     cfg_.kv_lora_rank, cfg_.block_tokens, state.block_tables(),
                     int(state.total_blocks()), attn_scale_, m_ws_, l_ws_,
                     c_ws_, stream, cfg_.latent_format, state.latent_scale(layer),
                     geo_.rope_dim);
    dsa_attn_combine(m_ws_, l_ws_, c_ws_, arows, n_split, geo_.local_heads,
                     cfg_.kv_lora_rank, c_, stream);
    dsa_vout_gemm(c_, w_.kv_b, attn_out_ + a0 * geo_.local_v_rows, arows,
                  geo_.local_heads, cfg_.qk_nope_head_dim, cfg_.v_head_dim,
                  cfg_.kv_lora_rank, stream, /*tensor_cores=*/false);
  }
}

namespace {
// DGPP_SYNC_EAGER=1 (2026-09-06, the fault hunt): an eager decode syncs after
// each stage and names the stage whose kernels faulted; a capturing stream
// is never synced.
void dsa_debug_sync(cudaStream_t stream, const char* what) {
  static const bool on = std::getenv("DGPP_SYNC_EAGER") != nullptr;
  if (!on) return;
  cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess ||
      cap != cudaStreamCaptureStatusNone)
    return;
  const cudaError_t e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("dsa decode fault after ") + what +
                             ": " + cudaGetErrorString(e));
}
}  // namespace

void DsaLayer::attend_dense(DsaStatePool& state, int layer,
                            const int32_t* req_ids, int64_t row0, int rows,
                            bool listed, cudaStream_t stream, int n_split,
                            bool decode, int one_req) {
  const bool staged = listed && !decode && one_req >= 0 && staged_prefill();
  if (staged && !stage_zeroed_) {
    // The staged copy's bitmap is zero at rest (each launch leaves it so).
    DGPP_CUDA_OK(cudaMemsetAsync(stage_ws_, 0, size_t((max_cache_tokens_ + 31) / 32) * 4, stream));
    stage_zeroed_ = true;
  }
  for (int64_t a0 = row0; a0 < row0 + rows; a0 += kDensePrefillRows) {
    const int arows =
        int(std::min<int64_t>(kDensePrefillRows, row0 + rows - a0));
    dsa_absorb_q(q_ + a0 * geo_.local_q_rows, w_.kv_b, q_tilde_, arows,
                 geo_.local_heads, cfg_.qk_nope_head_dim, cfg_.v_head_dim,
                 cfg_.kv_lora_rank, stream, geo_.rope_dim, /*tensor_cores=*/!decode);
    dsa_debug_sync(stream, "absorb_q");
    // A prefill tile over a quantized cache: from the staged copy of the
    // tile's distinct rows (bit for bit the launch below).
    const bool launched_staged =
        staged && dsa_attn_listed_staged(q_tilde_, state.latent(layer), one_req, req_ids + a0,
                                         topk_ + (sel_base_ + a0) * geo_.max_selected, geo_.max_selected,
                                         counts_ + sel_base_ + a0, arows, n_split, geo_.local_heads,
                                         cfg_.kv_lora_rank, cfg_.block_tokens, state.block_tables(),
                                         int(state.total_blocks()), attn_scale_, m_ws_, l_ws_, c_ws_, stream,
                                         cfg_.latent_format, state.latent_scale(layer), geo_.rope_dim, -1,
                                         stage_ws_, max_cache_tokens_, stage_slots_);
    const bool launched =
        launched_staged ? true :
        listed ? dsa_attn_listed(q_tilde_, state.latent(layer), req_ids + a0,
                                 topk_ + (sel_base_ + a0) * geo_.max_selected, geo_.max_selected,
                                 counts_ + sel_base_ + a0, arows, n_split,
                                 geo_.local_heads, cfg_.kv_lora_rank,
                                 cfg_.block_tokens, state.block_tables(),
                                 int(state.total_blocks()), attn_scale_, m_ws_,
                                 l_ws_, c_ws_, stream, cfg_.latent_format,
                                 state.latent_scale(layer), geo_.rope_dim)
               : dsa_attn_dense(q_tilde_, state.latent(layer), req_ids + a0,
                                pos_dev_ + a0, arows, n_split,
                                geo_.local_heads, cfg_.kv_lora_rank,
                                cfg_.block_tokens, state.block_tables(),
                                int(state.total_blocks()), attn_scale_, m_ws_,
                                l_ws_, c_ws_, stream, cfg_.latent_format,
                                state.latent_scale(layer), geo_.rope_dim);
    if (!launched) {
      // Geometry outside the dense kernel's: the split kernel over the
      // selection (dense by construction here).
      attend_tile(state, layer, req_ids, a0, arows, 8, stream);
      continue;
    }
    dsa_debug_sync(stream, listed ? "listed flash" : "dense flash");
    if (listed && std::getenv("DGPP_SYNC_EAGER") != nullptr) {
      cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
      if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess &&
          cap == cudaStreamCaptureStatusNone) {
        long long a[6] = {0, 0, 0, 0, 0, 0};
        const unsigned long long n = dsa_attn_anomalies(a, /*clear=*/true, stream);
        if (n != 0)
          throw std::runtime_error(
              "listed attention anomaly: " + std::to_string(n) + " gather(s) out of range; first: token " +
              std::to_string(a[0]) + " block " + std::to_string(a[1]) + " query row " + std::to_string(a[2]) +
              " split " + std::to_string(a[3]) + " list index " + std::to_string(a[4]) + " of " +
              std::to_string(a[5]) + " (layer " + std::to_string(layer) + ", rows " + std::to_string(rows) + ")");
      }
    }
    dsa_attn_combine(m_ws_, l_ws_, c_ws_, arows, n_split,
                     geo_.local_heads, cfg_.kv_lora_rank, c_, stream);
    dsa_debug_sync(stream, "combine");
    dsa_vout_gemm(c_, w_.kv_b, attn_out_ + a0 * geo_.local_v_rows, arows,
                  geo_.local_heads, cfg_.qk_nope_head_dim, cfg_.v_head_dim,
                  cfg_.kv_lora_rank, stream, /*tensor_cores=*/!decode);
    dsa_debug_sync(stream, "vout");
  }
}

// ---------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------

void DsaLayer::enqueue_prefill(const void* hidden_in, DsaStatePool& state,
                               int layer, int req, int64_t token_start,
                               int tokens, void* out, cudaStream_t stream, int row_base,
                               bool state_only) {
  validate_pool(state, layer);
  if (tokens <= 0 || tokens > max_tokens_)
    throw std::invalid_argument("dsa layer: token count out of range");
  if (token_start < 0 || token_start % cfg_.index_kpool != 0)
    throw std::invalid_argument(
        "dsa layer: prefill chunks must start pool-aligned");
  if (token_start + tokens > max_cache_tokens_)
    throw std::invalid_argument("dsa layer: chunk exceeds max_cache_tokens");
  // A continuation chunk shorter than kpool is fine on the device ring
  // (2026-09-05, M7; before, rejected here because the reference tail
  // seed read only in-chunk k rows and parity was pinned to it): the tail
  // seed writes only the tokens it has into their slots pos % kpool, and
  // the other slots still hold the previous chunk's last tokens — exactly
  // the ring the reference keeps for the whole sequence. Chunk STARTS stay
  // pool-aligned: the complete-pool compression indexes pools from the
  // chunk's first row. glm_tp_test's prefix gate runs one- and two-token
  // continuations against the unchunked prefill and through decode steps.
  if (!hidden_in || !out)
    throw std::invalid_argument("dsa layer: null buffer");
  if (row_base < 0 || row_base + tokens > max_tokens_)
    throw std::invalid_argument("dsa layer: the selection scratch base puts the rows past max_tokens");
  sel_base_ = row_base;

  if (!state.ensure_request_blocks(req, token_start + tokens, stream))
    throw std::runtime_error("dsa layer: cache pool exhausted during prefill");

  const int heads = cfg_.index_n_heads;
  const int dim = cfg_.index_head_dim;
  const int kpool = cfg_.index_kpool;
  const int blocks_per_req = int(state.total_blocks());
  const bool reuse = reuses_selection();
  if (reuse) {
    require_selection(SelKind::kPrefill, tokens, token_start, req, nullptr);
    if (state.owns_index(layer))
      throw std::invalid_argument(
          "dsa layer: a selection-reusing view on a layer that owns an index cache");
  } else if (!state.owns_index(layer)) {
    throw std::invalid_argument(
        "dsa layer: an indexed view on a layer without an index cache");
  }

  // Per-token metadata (host staging — prefill is never graph-captured);
  // the projections rotate at these positions.
  pos_host_.resize(size_t(tokens));
  for (int i = 0; i < tokens; ++i) pos_host_[size_t(i)] = token_start + i;
  req_ids_host_.assign(std::max(size_t(tokens), size_t(attn_rows_)), req);
  DGPP_CUDA_OK(cudaMemcpyAsync(pos_dev_, pos_host_.data(),
                               size_t(tokens) * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(req_ids_dev_, req_ids_host_.data(),
                               req_ids_host_.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));

  project_common(hidden_in, tokens, pos_dev_, stream);

  // Latent cache append (quantized on the way in for an fp8/fp4 cache; the
  // rope key bf16 after it).
  dsa_latent_append(kv_c_, req_ids_dev_, pos_dev_, tokens, state.block_tables(),
                    blocks_per_req, cfg_.block_tokens, state.latent(layer),
                    cfg_.kv_lora_rank, stream, cfg_.latent_format,
                    state.latent_scale(layer), geo_.rope_dim > 0 ? k_rot_ : nullptr,
                    geo_.rope_dim);

  if (reuse) {
    dot_stride_last_ = 0;
    if (state_only) return;
  } else {
  // Complete pools fully inside this chunk -> compressed index cache.
  const int64_t pool_lo = token_start / kpool;
  const int64_t pool_hi = (token_start + tokens) / kpool;
  const uint16_t* gate_rows = w_.gate ? gate_rows_ : nullptr;
  dsa_kpool_compress_write(k_rows_, dim, gate_rows, dim, w_.ape,
                           state.block_tables() + size_t(req) * blocks_per_req,
                           geo_.pools_per_block, pool_lo, pool_hi - pool_lo,
                           state.index_k(layer), state.index_scale(layer),
                           kpool, dim, stream);

  // Tail ring: the last kpool tokens of the request so far.
  dsa_kpool_tail_seed(k_rows_, dim, gate_rows, dim, req_ids_dev_, pos_dev_,
                      tokens, state.tail(layer), kpool, dim, stream);
  // The rows' state is complete here; the rest computes their output.
  if (state_only) return;

  // Selection + attention over dot tiles. The gather is per-chunk (every
  // tile reads the same pools); only the dot buffer is per-tile.
  const int64_t n_gather = pool_hi;  // pools visible to the chunk's last row
  const int64_t padded_n =
      n_gather > 0 ? round_up_to(n_gather, kPoolPadGranularity) : 0;
  if (padded_n > 0) {
    // The dot GEMM reads the padded tail of the gather buffer; keep every
    // byte it will ever read initialized (compute-sanitizer initcheck
    // discipline — the select kernel itself never reads those pools).
    if (padded_n > gather_zeroed_) {
      DGPP_CUDA_OK(cudaMemsetAsync(gather_k_ + gather_zeroed_ * dim, 0,
                                   size_t(padded_n - gather_zeroed_) * dim,
                                   stream));
      DGPP_CUDA_OK(cudaMemsetAsync(gather_scale_ + gather_zeroed_, 0,
                                   size_t(padded_n - gather_zeroed_) * 4,
                                   stream));
      gather_zeroed_ = padded_n;
    }
    dsa_gather_index_pools(
        state.block_tables() + size_t(req) * blocks_per_req,
        geo_.pools_per_block, state.index_k(layer), state.index_scale(layer),
        n_gather, gather_k_, gather_scale_, dim, stream);
  }

  // The allocation is sized for the maximum context. At shorter contexts,
  // use the same bytes for more query rows instead of launching tiny tiles.
  const int tile_rows = prefill_query_tile_rows(padded_n);
  for (int row0 = 0; row0 < tokens; row0 += tile_rows) {
    const int rows = std::min(tile_rows, tokens - row0);
    if (fused_prefill_scores(cfg_, padded_n)) {
      auto* keys = reinterpret_cast<uint64_t*>(dot_);
      dsa_prefill_score_keys(q_fp8_ + size_t(row0) * heads * dim,
                             gather_k_, w_folded_ + size_t(row0) * heads,
                             gather_scale_, pos_dev_ + row0, rows, n_gather,
                             padded_n, geo_.select_k, kpool, keys, stream,
                             cfg_.index_relu != 0);
      dsa_select_prefill_keys(keys, padded_n, pos_dev_ + row0, rows, n_gather,
                              geo_.select_k, kpool, geo_.max_selected,
                              topk_ + size_t(sel_base_ + row0) * geo_.max_selected,
                              counts_ + sel_base_ + row0,
                              static_cast<uint64_t*>(select_ws_), stream);
      dot_stride_last_ = 0;  // this path stores reduced keys, not per-head dots
      continue;
    }
    if (padded_n > 0) {
      // Dots for this tile's (row, head) pairs against the gathered pools.
      gemm_.matmul(q_fp8_ + size_t(row0) * heads * dim, gather_k_, dot_,
                   rows * heads, int(padded_n), dim, DType::F8_E4M3,
                   GemmOut::F32, size_t(dim), gemm_ws_, gemm_ws_bytes_,
                   stream);
      dot_stride_last_ = padded_n;
    } else {
      dot_stride_last_ = 0;
    }
    dsa_select_prefill(dot_, std::max<int64_t>(padded_n, 1),
                       w_folded_ + size_t(row0) * heads, gather_scale_,
                       pos_dev_ + row0, rows, n_gather, heads, geo_.select_k,
                       kpool, geo_.max_selected,
                       topk_ + size_t(sel_base_ + row0) * geo_.max_selected, counts_ + sel_base_ + row0,
                       stream, cfg_.index_relu != 0, select_ws_);
  }
  note_selection(SelKind::kPrefill, tokens, token_start, req, nullptr);
  }

  // Attention. Rows whose context is below index_topk tokens select
  // densely — visible pools (pos + 1) / kpool <= select_k means every
  // visible pool plus the tail, i.e. tokens [0, pos] — and run on the
  // tensor-core kernel in wide tiles; the rest keep the
  // per-row split kernel over their selection, in 8-row tiles split by a
  // fixed 8 (<= the scratch capacity computed at construction).
  const int64_t dense_last_pos = int64_t(kpool) * (geo_.select_k + 1) - 2;
  int dense_rows = 0;
  if (dense_prefill_) {
    const int64_t n = dense_last_pos - token_start + 1;
    dense_rows = int(std::max<int64_t>(0, std::min<int64_t>(n, tokens)));
  }
  if (dense_rows > 0)
    attend_dense(state, layer, req_ids_dev_, 0, dense_rows, /*listed=*/false, stream);
  // The sparse regime: the same flash kernel over each row's selection,
  // in the same wide tiles (falls back to the split kernel per 8 rows when
  // the geometry is outside the kernel's).
  if (tokens > dense_rows)
    attend_dense(state, layer, req_ids_dev_, dense_rows, tokens - dense_rows,
                 /*listed=*/true, stream, kDensePrefillSplit, /*decode=*/false, /*one_req=*/req);

  // Output projection.
  project_out(out, tokens, stream);
}

// ---------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------

void DsaLayer::enqueue_decode(const void* hidden_in, DsaStatePool& state,
                              int layer, const int32_t* req_ids,
                              const int64_t* pos, const int32_t* req_spans,
                              int num_requests, int tokens, void* out,
                              cudaStream_t stream,
                              WeightPrefetcher* prefetch,
                              void* tail_snapshots) {
  sel_base_ = 0;  // the decode rows at the scratch's start
  validate_pool(state, layer);
  if (tokens <= 0 || tokens > max_decode_rows_)
    throw std::invalid_argument("dsa layer: decode rows out of range");
  if (num_requests <= 0 || num_requests > state.max_requests())
    throw std::invalid_argument("dsa layer: request count out of range");
  if (!hidden_in || !req_ids || !pos || !req_spans || !out)
    throw std::invalid_argument("dsa layer: null buffer");

  const int heads = cfg_.index_n_heads;
  const int dim = cfg_.index_head_dim;
  const int kpool = cfg_.index_kpool;
  dot_stride_last_ = 0;  // decode selects consume no dot buffer (debug probe)
  const bool reuse = reuses_selection();
  if (reuse) {
    require_selection(SelKind::kDecode, tokens, -1, -1, pos);
    if (state.owns_index(layer))
      throw std::invalid_argument(
          "dsa layer: a selection-reusing view on a layer that owns an index cache");
  } else if (!state.owns_index(layer)) {
    throw std::invalid_argument(
        "dsa layer: an indexed view on a layer without an index cache");
  }

  project_common(hidden_in, tokens, pos, stream);
  dsa_debug_sync(stream, "project");
  // Everything from here to the output projection is latency-bound at
  // decode (~130 us of small kernels): the longest window in the step, so
  // the budget is the whole projection, capped only by L2 headroom.
  if (prefetch) {
    prefetch->open_window(stream, size_t{16} << 20, prefetch->layer_rate());
    prefetch->add(w_.kv_b, kv_b_bytes());  // absorb_q and vout read it first
    if (w_.packed_int() || w_.quantized()) {
      prefetch->add(w_.packed_int() ? static_cast<const void*>(w_.o_proj_p.packed)
                                    : static_cast<const void*>(w_.o_proj_q.payload),
                    o_proj_bytes());
    } else {
      // A bf16 o_proj: the bytes the rows' launch streams (its packed
      // companion's when the GEMM holds one — kernels/bf12_gemv.hpp).
      const void* view = nullptr;
      size_t view_bytes = 0;
      gemm_.resident_view(w_.o_proj, o_proj_bytes(), tokens, &view, &view_bytes);
      prefetch->add_view(w_.o_proj, view, view_bytes);  // a companion is its own allocation
    }
  }

  // Latent rows first (this batch's own tokens are readable by this
  // batch's attention — causal self-include, reference semantics).
  dsa_latent_append(kv_c_, req_ids, pos, tokens, state.block_tables(),
                    int(state.total_blocks()), cfg_.block_tokens,
                    state.latent(layer), cfg_.kv_lora_rank, stream,
                    cfg_.latent_format, state.latent_scale(layer),
                    geo_.rope_dim > 0 ? k_rot_ : nullptr, geo_.rope_dim);
  dsa_debug_sync(stream, "latent append");
  if (!reuse) {
  // Ring update + pool compression for any pool completed by this batch.
  dsa_kpool_decode_update(
      k_rows_, dim, w_.gate ? gate_rows_ : nullptr, dim, w_.ape, req_ids, pos,
      req_spans, num_requests, state.block_tables(), int(state.total_blocks()),
      state.tail(layer), state.index_k(layer), state.index_scale(layer),
      geo_.pools_per_block, kpool, dim, stream, tail_snapshots);
  dsa_debug_sync(stream, "kpool update");
  // Fused select straight from the blocked index cache.
  dsa_select_decode(q_fp8_, w_folded_, req_ids, pos, tokens,
                    state.block_tables(), int(state.total_blocks()),
                    state.index_k(layer), state.index_scale(layer),
                    geo_.pools_per_block, heads, dim, geo_.select_k, kpool,
                    geo_.max_selected, topk_, counts_, select_ws_,
                    select_ws_pools_, counter_ws_, /*grid_blocks=*/0, stream,
                    cfg_.index_relu != 0);
  dsa_debug_sync(stream, "select");
  note_selection(SelKind::kDecode, tokens, -1, -1, pos);
  }
  if (std::getenv("DGPP_SYNC_EAGER") != nullptr) {
    // The fault hunt: every selected token inside the row's
    // context, every block-table entry it reaches reserved — checked on
    // the host before the attention gathers through them.
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) == cudaSuccess &&
        cap == cudaStreamCaptureStatusNone) {
      std::vector<int32_t> cnt(static_cast<size_t>(tokens)), list(static_cast<size_t>(tokens) * geo_.max_selected);
      std::vector<int64_t> posh(static_cast<size_t>(tokens));
      std::vector<int32_t> ids(static_cast<size_t>(tokens));
      const size_t bpr = size_t(state.total_blocks());
      std::vector<int32_t> table(size_t(state.max_requests()) * bpr);
      DGPP_CUDA_OK(cudaMemcpyAsync(cnt.data(), counts_, sizeof(int32_t) * cnt.size(), cudaMemcpyDeviceToHost, stream));
      DGPP_CUDA_OK(cudaMemcpyAsync(list.data(), topk_, sizeof(int32_t) * list.size(), cudaMemcpyDeviceToHost, stream));
      DGPP_CUDA_OK(cudaMemcpyAsync(posh.data(), pos, sizeof(int64_t) * posh.size(), cudaMemcpyDeviceToHost, stream));
      DGPP_CUDA_OK(cudaMemcpyAsync(ids.data(), req_ids, sizeof(int32_t) * ids.size(), cudaMemcpyDeviceToHost, stream));
      DGPP_CUDA_OK(cudaMemcpyAsync(table.data(), state.block_tables(), sizeof(int32_t) * table.size(), cudaMemcpyDeviceToHost, stream));
      DGPP_CUDA_OK(cudaStreamSynchronize(stream));
      for (int t = 0; t < tokens; ++t) {
        if (posh[size_t(t)] < 0) continue;
        const int64_t seq_len = posh[size_t(t)] + 1;
        const int32_t* row = list.data() + size_t(t) * geo_.max_selected;
        const int32_t* bt = table.data() + size_t(ids[size_t(t)]) * bpr;
        if (cnt[size_t(t)] < 0 || cnt[size_t(t)] > geo_.max_selected)
          throw std::runtime_error("select check: row " + std::to_string(t) + " count " + std::to_string(cnt[size_t(t)]));
        for (int j = 0; j < cnt[size_t(t)]; ++j) {
          const int32_t tok = row[j];
          if (tok < 0 || tok >= seq_len)
            throw std::runtime_error(
                "select check: layer " + std::to_string(layer) + " row " + std::to_string(t) +
                " (req " + std::to_string(ids[size_t(t)]) + ", pos " + std::to_string(posh[size_t(t)]) +
                ", count " + std::to_string(cnt[size_t(t)]) + ") entry " + std::to_string(j) +
                " = " + std::to_string(tok) + " outside [0, " + std::to_string(seq_len) + ")");
          const int64_t b = tok / cfg_.block_tokens;
          if (b >= int64_t(bpr) || bt[b] < 0 || bt[b] >= int32_t(state.total_blocks()))
            throw std::runtime_error(
                "select check: layer " + std::to_string(layer) + " row " + std::to_string(t) +
                " (req " + std::to_string(ids[size_t(t)]) + ", pos " + std::to_string(posh[size_t(t)]) +
                ") entry " + std::to_string(j) + " = " + std::to_string(tok) + " maps to block " +
                std::to_string(b) + " -> " + std::to_string(b < int64_t(bpr) ? bt[b] : -2));
        }
      }
    }
  }
  // Absorbed attention + v-absorb + output projection: the tensor-core
  // listed kernel when the geometry is its (16-head slabs), else the
  // register split kernel.
  if (decode_mma_ && geo_.local_heads >= 16 && geo_.local_heads % 16 == 0)
    attend_dense(state, layer, req_ids, 0, tokens, /*listed=*/true, stream,
                 std::min(decode_mma_split_, std::max(decode_n_split_, 8)), /*decode=*/true);
  else
    attend_tile(state, layer, req_ids, 0, tokens, decode_n_split_, stream);
  dsa_debug_sync(stream, "attention");
  project_out(out, tokens, stream);
  dsa_debug_sync(stream, "o_proj");
  // Padding rows (pos < 0: a fixed-shape batch's unoccupied rows, the
  // in-graph draft's rejected row) skipped every state write above but
  // still carry whatever the attention scratch held into the projection.
  // Zero them so a padding row's block output — and its share of the
  // boundary all-reduce — is deterministic by construction, the same
  // contract the KDA layer's padding rows already keep.
  dsa_zero_padding_rows(out, pos, tokens, cfg_.hidden, stream);
}

// ---------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------

void DsaLayer::validate_pool(const DsaStatePool& state, int layer) const {
  const DsaConfig& c = state.config();
  const bool same = c.hidden == cfg_.hidden && c.num_heads == cfg_.num_heads &&
                    c.q_lora_rank == cfg_.q_lora_rank &&
                    c.kv_lora_rank == cfg_.kv_lora_rank &&
                    c.qk_nope_head_dim == cfg_.qk_nope_head_dim &&
                    c.qk_rope_head_dim == cfg_.qk_rope_head_dim &&
                    c.v_head_dim == cfg_.v_head_dim &&
                    c.index_n_heads == cfg_.index_n_heads &&
                    c.index_head_dim == cfg_.index_head_dim &&
                    c.index_topk == cfg_.index_topk &&
                    c.index_kpool == cfg_.index_kpool &&
                    c.index_relu == cfg_.index_relu &&
                    c.block_tokens == cfg_.block_tokens &&
                    c.tp_size == cfg_.tp_size &&
                    c.num_dsa_layers == cfg_.num_dsa_layers &&
                    c.index_layers() == cfg_.index_layers() &&
                    c.latent_format == cfg_.latent_format;
  if (!same)
    throw std::invalid_argument(
        "dsa layer: state pool was built for a different configuration");
  if (layer < 0 || layer >= cfg_.num_dsa_layers)
    throw std::out_of_range("dsa layer: layer " + std::to_string(layer));
}

}  // namespace dgpp
