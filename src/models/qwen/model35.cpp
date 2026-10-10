// Qwen3.5-27B dense model (FP8 text-only, standard pre-norm residual).
//
// 48 GDN layers (swish gate) + 16 Full GQA layers, dense SwiGLU MLPs. Text
// only: no vision tower, no hyperconnections / PLE / MoE. GDN recurrent +
// conv state is model-owned per request slot; the Full layers' K/V lives in
// Qwen35KvPool (plus one draft plane when MTP is on). Eager + prefix-off
// serving shape; the graph path replays the same walks.

#include "models/qwen/model35.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "kernels/add_rmsnorm.hpp"
#include "kernels/dflash2.hpp"
#include "kernels/fp8_per_tensor.hpp"
#include "kernels/fp8_dequant.hpp"
#include "kernels/gemm.hpp"
#include "kernels/kernels.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/qwen_mtp.hpp"
#include "kernels/qwen_norm.hpp"
#include "kernels/qsa.hpp"
#include "kernels/scale_gemm.hpp"
#include "loaders/fp8_quant.hpp"
#include "models/qwen/config.hpp"

namespace dgpp {
namespace {

// Device-to-device copy on a stream (the snapshot helpers).
inline void d2d(void* dst, const void* src, size_t n, cudaStream_t stream) {
  DGPP_CUDA_OK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, stream));
}

// The qwen4_exp layer ctors take QwenTextConfig: adapt the fields they read.
QwenTextConfig qwen_text_adapter(const Qwen35TextConfig& c) {
  QwenTextConfig q;
  q.hidden_size = c.hidden_size;
  q.head_dim = c.head_dim;
  q.rotary_dim = c.rotary_dim;
  q.rope_theta = c.rope_theta;
  q.rms_norm_eps = c.rms_norm_eps;
  q.gdn_key_heads = c.gdn_key_heads;
  q.gdn_value_heads = c.gdn_value_heads;
  q.gdn_key_head_dim = c.gdn_key_head_dim;
  q.gdn_value_head_dim = c.gdn_value_head_dim;
  q.gdn_conv_width = c.gdn_conv_width;
  return q;
}

}  // namespace

// ---- Qwen35KvPool ------------------------------------------------------------

Qwen35KvPool::~Qwen35KvPool() {
  cudaFree(k_base_);
  cudaFree(v_base_);
  cudaFree(dk_base_);
  cudaFree(dv_base_);
}

size_t Qwen35KvPool::cache_bytes(const Qwen35KvPoolShape& shape) {
  const size_t kv_row = static_cast<size_t>(shape.kv_heads) * shape.dim;
  const size_t layer_kv = static_cast<size_t>(shape.token_slots) * kv_row * 2 * 2;
  const size_t dkv_row = static_cast<size_t>(shape.draft_kv_heads) * shape.draft_dim;
  const size_t draft_layer_kv = static_cast<size_t>(shape.token_slots) * dkv_row * 2 * 2;
  const int64_t num_blocks = shape.token_slots / shape.block_tokens;
  return static_cast<size_t>(shape.layers) * layer_kv +
         static_cast<size_t>(shape.draft_layers) * draft_layer_kv +
         PagedBlockTable::table_bytes(shape.max_requests, num_blocks);
}

void Qwen35KvPool::check_req(int req, const char* what) const {
  if (req < 0 || req >= shape_.max_requests)
    throw std::out_of_range(std::string("Qwen35KvPool: request out of range in ") + what);
}

void Qwen35KvPool::init(const Qwen35KvPoolShape& shape) {
  if (shape.layers <= 0 || shape.kv_heads <= 0 || shape.dim <= 0 || shape.block_tokens <= 0 ||
      shape.max_requests <= 0 || shape.token_slots <= 0 ||
      (shape.draft_layers > 0 && (shape.draft_kv_heads <= 0 || shape.draft_dim <= 0)))
    throw std::invalid_argument("Qwen35KvPool: shape fields must be positive");
  if (shape.token_slots % shape.block_tokens != 0)
    throw std::invalid_argument("Qwen35KvPool: token_slots must be a multiple of block_tokens");
  shape_ = shape;
  const int64_t num_blocks = shape.token_slots / shape.block_tokens;
  DGPP_CUDA_OK(cudaMalloc(&k_base_, layer_kv_elems() * shape.layers * 2));
  DGPP_CUDA_OK(cudaMalloc(&v_base_, layer_kv_elems() * shape.layers * 2));
  if (shape.draft_layers > 0) {
    DGPP_CUDA_OK(cudaMalloc(&dk_base_, draft_layer_kv_elems() * shape.draft_layers * 2));
    DGPP_CUDA_OK(cudaMalloc(&dv_base_, draft_layer_kv_elems() * shape.draft_layers * 2));
  }
  table_.init(shape.max_requests, shape.block_tokens, num_blocks);
  initialized_ = true;
}

QwenFullAttnCache Qwen35KvPool::view(int layer) const {
  if (layer < 0 || layer >= shape_.layers + shape_.draft_layers)
    throw std::out_of_range("Qwen35KvPool: layer out of range in view");
  QwenFullAttnCache c;
  if (layer < shape_.layers) {
    c.k_cache = k_base_ + static_cast<size_t>(layer) * layer_kv_elems();
    c.v_cache = v_base_ + static_cast<size_t>(layer) * layer_kv_elems();
  } else {
    const int plane = layer - shape_.layers;
    c.k_cache = dk_base_ + static_cast<size_t>(plane) * draft_layer_kv_elems();
    c.v_cache = dv_base_ + static_cast<size_t>(plane) * draft_layer_kv_elems();
  }
  c.block_tables = table_.device_tables();
  c.block_tokens = shape_.block_tokens;
  c.blocks_per_request = static_cast<int>(table_.total_blocks());
  c.max_requests = shape_.max_requests;
  return c;
}

void Qwen35KvPool::reset_request(int req, cudaStream_t stream) {
  check_req(req, "reset_request");
  table_.release_request_blocks(req, stream);
}

void Qwen35KvPool::reset_all(cudaStream_t stream) {
  const size_t bytes = layer_kv_elems() * shape_.layers * 2;
  DGPP_CUDA_OK(cudaMemsetAsync(k_base_, 0, bytes, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(v_base_, 0, bytes, stream));
  if (dk_base_) {
    const size_t dbytes = draft_layer_kv_elems() * shape_.draft_layers * 2;
    DGPP_CUDA_OK(cudaMemsetAsync(dk_base_, 0, dbytes, stream));
    DGPP_CUDA_OK(cudaMemsetAsync(dv_base_, 0, dbytes, stream));
  }
  table_.reset_all(stream);
}

void Qwen35KvPool::copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream) {
  if (src < 0 || src >= total_blocks() || dst < 0 || dst >= total_blocks())
    throw std::out_of_range("Qwen35KvPool: block out of range in copy_block_contents");
  const size_t row_elems = kv_row_elems();
  const size_t span = static_cast<size_t>(shape_.block_tokens) * row_elems * 2;
  for (int l = 0; l < shape_.layers; ++l) {
    const size_t base = static_cast<size_t>(l) * layer_kv_elems();
    d2d(k_base_ + base + static_cast<size_t>(dst) * span / 2, k_base_ + base + static_cast<size_t>(src) * span / 2,
        span, stream);
    d2d(v_base_ + base + static_cast<size_t>(dst) * span / 2, v_base_ + base + static_cast<size_t>(src) * span / 2,
        span, stream);
  }
  if (dk_base_) {
    const size_t drow = draft_row_elems();
    const size_t dspan = static_cast<size_t>(shape_.block_tokens) * drow * 2;
    for (int l = 0; l < shape_.draft_layers; ++l) {
      const size_t base = static_cast<size_t>(l) * draft_layer_kv_elems();
      d2d(dk_base_ + base + static_cast<size_t>(dst) * dspan / 2,
          dk_base_ + base + static_cast<size_t>(src) * dspan / 2, dspan, stream);
      d2d(dv_base_ + base + static_cast<size_t>(dst) * dspan / 2,
          dv_base_ + base + static_cast<size_t>(src) * dspan / 2, dspan, stream);
    }
  }
}

// ---- Qwen35Model ---------------------------------------------------------------

// The dequant bridge holds the largest dense FP8 matrix in BF16 so
// prefill-shaped (m>128) products dequantize once and run cuBLASLt BF16
// instead of the slow hand-rolled tile kernel (mirrors
// QwenModel::dense_bridge_bytes; world 1 takes full rows).
static size_t qwen35_dense_bridge_bytes(const Qwen35TextConfig& cfg) {
  const size_t H = static_cast<size_t>(cfg.hidden_size), I = static_cast<size_t>(cfg.intermediate_size);
  const size_t q = 2 * static_cast<size_t>(cfg.num_attention_heads) * cfg.head_dim;
  const size_t kv = static_cast<size_t>(cfg.num_key_value_heads) * cfg.head_dim;
  const size_t o = static_cast<size_t>(cfg.num_attention_heads) * cfg.head_dim;
  size_t elems = 0;
  const auto take = [&](size_t n, size_t k) { elems = std::max(elems, n * k); };
  take(q, H);    // q_proj
  take(kv, H);   // k / v
  take(H, o);    // o_proj
  take(I, H);    // gate / up
  take(H, I);    // down
  return elems * 2;
}

Qwen35Model::Qwen35Model(const Qwen35TextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
                         int64_t max_cache_tokens, LoaderResidency residency, BoundaryReducer* boundary,
                         int rank, int world, int max_requests, int decode_rows, bool mtp,
                         const std::string& dflash2_dir)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, rank, world, residency,
              world > 1 ? LoaderHeadSharding::VocabSharded : LoaderHeadSharding::Full,
              mtp && residency == LoaderResidency::Resident),
      mtp_(mtp) {
  if (max_tokens <= 0) throw std::invalid_argument("Qwen35Model: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("Qwen35Model: max_requests out of range");
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("Qwen35Model: decode_rows exceeds the limit of " + std::to_string(decode_rows_cap()));
  if (world < 1 || rank < 0 || rank >= world) throw std::invalid_argument("Qwen35Model: rank / world out of range");
  if (world > 1 && boundary == nullptr) throw std::invalid_argument("Qwen35Model: a TP world needs a boundary reducer");
  if (world == 1 && boundary != nullptr) throw std::invalid_argument("Qwen35Model: world 1 takes no boundary reducer");
  if (cfg_.eos_token_ids.empty()) throw std::invalid_argument("Qwen35Model: the config names no EOS token");
  if (mtp_ && cfg_.mtp_layer() < 0)
    throw std::invalid_argument("Qwen35Model: the config has no draft layer (mtp)");
  if (!dflash2_dir.empty()) {
    if (mtp_)
      throw std::invalid_argument("Qwen35Model: dflash2 replaces the MTP draft; enable one or the other");
    dfcfg_ = DFlash2Config::from_json_file(dflash2_dir + "/config.json");
    dfcfg_.validate_against(cfg_);
    df_tap_.assign(cfg_.num_hidden_layers, -1);
    for (size_t t = 0; t < dfcfg_.target_layer_ids.size(); ++t)
      df_tap_[dfcfg_.target_layer_ids[t]] = static_cast<int>(t);
    dflash2_ = true;
  }
  for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
    if (cfg_.layers[l] == Qwen35LayerKind::Gdn)
      ++num_gdn_;
    else
      ++num_full_;
  }
  // Per-layer kind ordinals (the attention PT slots); -1 for the other kind.
  pt_gdn_ord_.assign(cfg_.num_hidden_layers, -1);
  pt_full_ord_.assign(cfg_.num_hidden_layers, -1);
  {
    int g = 0, f = 0;
    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
      if (cfg_.layers[l] == Qwen35LayerKind::Gdn)
        pt_gdn_ord_[l] = g++;
      else
        pt_full_ord_[l] = f++;
    }
  }
  // The local TP geometry (loader35): this rank's GDN heads, attention
  // heads, MLP slice and vocab slice; the layer objects read their own
  // local widths from the resident slices.
  const Qwen35LocalGeometry& geo = loader_.geometry();
  const int64_t lv = geo.local_value_heads, V = cfg_.gdn_value_head_dim, K = cfg_.gdn_key_head_dim;
  const int64_t C = 2 * static_cast<int64_t>(geo.local_key_heads) * K + lv * V;
  rec_elems_ = lv * V * K;
  conv_elems_ = C * (cfg_.gdn_conv_width - 1);
  // The qwen4_exp layer ctors' config view.
  qcfg_ = qwen_text_adapter(cfg_);
  init_stream();
  loader_.set_reader_stream(stream_);
  globals_ = loader_.load_globals();
  const int H = cfg_.hidden_size;
  SessionParams sp;
  sp.max_tokens = max_tokens;
  sp.max_cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kv_block_tokens_static() - 1) /
       kv_block_tokens_static()) *
      kv_block_tokens_static();
  sp.rank = rank;
  sp.world = world;
  sp.boundary = boundary;
  sp.max_requests = max_requests;
  sp.decode_rows = decode_rows;
  sp.mtp = mtp_;
  sp.vocab_size = cfg_.vocab_size;
  sp.hidden = H;
  sp.lm_vocab_begin = geo.lm_vocab_begin;
  sp.lm_vocab_count = geo.lm_vocab_count;
  sp.max_position_embeddings = cfg_.max_position_embeddings;
  sp.block_tokens = kv_block_tokens_static();
  sp.snapshot_align = 1;
  sp.draft_width = mtp_ ? H : 0;
  sp.eos = static_cast<int32_t>(cfg_.eos_token_ids[0]);
  init_session(sp);
  // Paged K/V over the Full layers, plus the draft plane when MTP is on
  // or the DFlash2 drafter's planes (its own kv geometry, shared table).
  Qwen35KvPoolShape shape;
  shape.layers = num_full_ + (mtp_ ? 1 : 0);
  shape.kv_heads = geo.local_kv_heads;
  shape.dim = cfg_.head_dim;
  shape.block_tokens = kv_block_tokens_static();
  shape.max_requests = max_requests;
  shape.token_slots = sp.max_cache_tokens;
  if (dflash2_) {
    shape.draft_layers = dfcfg_.num_hidden_layers;
    shape.draft_kv_heads = dfcfg_.local_kv_heads(world);  // this rank's kv heads (sharded drafter)
    shape.draft_dim = dfcfg_.head_dim;
  }
  pool_.init(shape);
  // Lt workspace for the lm head (the dense MLP uses the fused scale GEMM).
  gemm_ws_bytes_ = std::max<size_t>(64u << 20, gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, H,
                                                                          DType::BF16));
  DGPP_CUDA_OK(cudaMalloc(&gemm_ws_, gemm_ws_bytes_));
  gw_ = QwenGemmWorkspace{&gemm_, gemm_ws_, gemm_ws_bytes_};
  // The dense sites' lowering (kernels/gemm.hpp dense_gemv_rows): the GEMV
  // chunks (and the fused multi-problem launches) to the bound, cuBLASLt's
  // algorithm (bf16) or the streaming tensor-core GEMM (fp8) above it.
  // Without this, multi-row decode batches shatter into per-row GEMV
  // launches that re-read the weights per row (mirrors QwenModel).
  gemm_.set_decode_rows(std::min(max_decode_rows_, dense_gemv_rows()));
  gw_.gemv_rows = dense_gemv_rows();
  // The prefill-shaped bf16 Lt products (the dequant bridge above the
  // scale GEMM's 128-row lowering) take one algorithm — the heuristic's for
  // the walk's widest row count — at every width, so a prompt's reduction
  // is the same alone and as a span of a group walk (2026-10-05).
  gemm_.set_pinned_rows(max_tokens_);
  // NOTE: dense_gemv_rows() defaults to 4, and the FP8 GEMV row loop
  // re-reads weights per ≤4-row chunk (and per single row when smem can't
  // stage more, e.g. down-proj k=17408). Every decode row count takes the
  // weights-once streaming MMA (2026-10-05; m >= 5 since 2026-10-03, the
  // <= 4-row decodes on the GEMV): its per-row chain is the same at any m,
  // 1..64, so a request's projections are bitwise the same alone, at
  // every scheduled verify depth (2 / 4 / 6 / 8 rows) and in an eight-slot
  // batch — the GEMV chain they took below five rows is a different
  // chain, and a scheduled solo request's transcript moved with its depth.
  gw_.mma_from_rows = 1;
  // Dense FP8 prefill bridge (mirrors QwenModel): m>128 products dequantize
  // the matrix into scratch and run Lt BF16. Weights here are always FP8,
  // so the bridge is unconditional.
  dense_bridge_bytes_ = qwen35_dense_bridge_bytes(cfg_);
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&gw_.dequant), dense_bridge_bytes_));
  gw_.dequant_bytes = dense_bridge_bytes_;
  // Activation scratch at max_tokens rows.
  const size_t M = static_cast<size_t>(max_tokens_);
  const size_t I = static_cast<size_t>(cfg_.intermediate_size);
  DGPP_CUDA_OK(cudaMalloc(&resid_, M * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&x_, M * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&attn_out_, M * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&mlp_out_, M * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&gate_tmp_, M * I * 2));
  DGPP_CUDA_OK(cudaMalloc(&up_tmp_, M * I * 2));
  // Per-tensor FP8 prefill recipe (engine.prefill_fp8_per_tensor): Resident stacks
  // requantize every layer's MLP once at boot (dequant to the bridge, then
  // absmax + x/448 quantize). Streaming stacks keep the bridge: their
  // layers are not all resident, so there is nothing eager to build from.
  pt_enabled_ = prefill_fp8_per_tensor_ && loader_.residency() == LoaderResidency::Resident;
  pt_attn_enabled_ = pt_enabled_;
  if (pt_enabled_) {
    const size_t IH = I * H;
    pt_slots_ = cfg_.num_hidden_layers + (mtp_ ? 1 : 0);
    DGPP_CUDA_OK(cudaMalloc(&pt_gate_, static_cast<size_t>(pt_slots_) * IH));
    DGPP_CUDA_OK(cudaMalloc(&pt_up_, static_cast<size_t>(pt_slots_) * IH));
    DGPP_CUDA_OK(cudaMalloc(&pt_down_, static_cast<size_t>(pt_slots_) * IH));
    DGPP_CUDA_OK(cudaMalloc(&pt_scales_, static_cast<size_t>(pt_slots_) * 3 * 4));
    DGPP_CUDA_OK(cudaMalloc(&pt_act_, M * I));
    DGPP_CUDA_OK(cudaMalloc(&pt_act_scales_, 2 * 4));
    for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
      const Qwen35LayerResident& r = loader_.load_layer(l);
      requant_mlp_pt(l, r.mlp, stream_);
    }
    if (mtp_) {
      const Qwen35LayerResident& r = loader_.load_layer(cfg_.mtp_layer());
      requant_mlp_pt(pt_slots_ - 1, r.mlp, stream_);
    }
    // The same recipe for the attention projections: GDN in_proj_qkv [C,
    // H] + in_proj_z [LV, H] + out_proj [H, LV] per GDN ordinal; Full q
    // [QW, H] + k/v [KW, H] + o [H, FH] per full ordinal (the MTP draft
    // layer takes the last full slot). The bridge is idle at boot, so it
    // stages each dequant like the MLP requant above.
    if (pt_attn_enabled_)
    {
      const int64_t lk = geo.local_key_heads, lv = geo.local_value_heads;
      const int64_t K = cfg_.gdn_key_head_dim, V = cfg_.gdn_value_head_dim;
      pt_gdn_C_ = 2 * lk * K + lv * V;
      pt_gdn_LV_ = lv * V;
      const int64_t lh = geo.local_heads, lkv = geo.local_kv_heads;
      const int64_t D = cfg_.head_dim;
      pt_full_QW_ = lh * 2 * D;
      pt_full_KW_ = lkv * D;
      pt_full_FH_ = lh * D;
      const size_t Hh = static_cast<size_t>(cfg_.hidden_size);
      const size_t C = static_cast<size_t>(pt_gdn_C_), LV = static_cast<size_t>(pt_gdn_LV_);
      const size_t QW = static_cast<size_t>(pt_full_QW_), KW = static_cast<size_t>(pt_full_KW_),
                     FH = static_cast<size_t>(pt_full_FH_);
      pt_gdn_slots_ = num_gdn_;
      pt_full_slots_ = num_full_ + (mtp_ ? 1 : 0);
      DGPP_CUDA_OK(cudaMalloc(&pt_gdn_qkv_, static_cast<size_t>(pt_gdn_slots_) * C * Hh));
      DGPP_CUDA_OK(cudaMalloc(&pt_gdn_z_, static_cast<size_t>(pt_gdn_slots_) * LV * Hh));
      DGPP_CUDA_OK(cudaMalloc(&pt_gdn_o_, static_cast<size_t>(pt_gdn_slots_) * Hh * LV));
      DGPP_CUDA_OK(cudaMalloc(&pt_gdn_scales_, static_cast<size_t>(pt_gdn_slots_) * 3 * 4));
      DGPP_CUDA_OK(cudaMalloc(&pt_full_q_, static_cast<size_t>(pt_full_slots_) * QW * Hh));
      DGPP_CUDA_OK(cudaMalloc(&pt_full_k_, static_cast<size_t>(pt_full_slots_) * KW * Hh));
      DGPP_CUDA_OK(cudaMalloc(&pt_full_v_, static_cast<size_t>(pt_full_slots_) * KW * Hh));
      DGPP_CUDA_OK(cudaMalloc(&pt_full_o_, static_cast<size_t>(pt_full_slots_) * Hh * FH));
      DGPP_CUDA_OK(cudaMalloc(&pt_full_scales_, static_cast<size_t>(pt_full_slots_) * 4 * 4));
      for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
        const Qwen35LayerResident& r = loader_.load_layer(l);
        if (r.kind == Qwen35LayerKind::Gdn)
          requant_gdn_pt(pt_gdn_ord_[l], r.gdn, stream_);
        else
          requant_full_pt(pt_full_ord_[l], r.full, stream_);
      }
      if (mtp_) {
        const Qwen35LayerResident& r = loader_.load_layer(cfg_.mtp_layer());
        requant_full_pt(num_full_, r.full, stream_);
      }
    }
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }
  // Blockwise-FP8 lm head (engine.dense_weights = fp8, Resident only): the 248k-row BF16
  // head is the only multi-GB BF16 weight left on the decode path. Host
  // requant (fp8_quant::encode_block128, the loader's own encoder) once;
  // decode rows read half the bytes through the F32 scale-GEMM path.
  head_fp8_enabled_ = dense_weights_fp8_ && loader_.residency() == LoaderResidency::Resident;
  if (head_fp8_enabled_) {
    const int64_t V = lm_vocab_count_, Hh = cfg_.hidden_size;
    const int64_t sr = (V + 127) / 128, sc = (Hh + 127) / 128;
    DGPP_CUDA_OK(cudaMalloc(&head_fp8_, static_cast<size_t>(V) * static_cast<size_t>(Hh)));
    DGPP_CUDA_OK(cudaMalloc(&head_scales_, static_cast<size_t>(sr) * static_cast<size_t>(sc) * 4));
    std::vector<uint16_t> host(static_cast<size_t>(V) * static_cast<size_t>(Hh));
    DGPP_CUDA_OK(cudaMemcpy(host.data(), globals_.lm_head, host.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> payload(host.size());
    std::vector<float> scales(static_cast<size_t>(sr) * static_cast<size_t>(sc));
    fp8_quant::encode_block128(host.data(), static_cast<size_t>(Hh), V, Hh, payload.data(),
                               scales.data());
    DGPP_CUDA_OK(cudaMemcpy(head_fp8_, payload.data(), payload.size(), cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(
        cudaMemcpy(head_scales_, scales.data(), scales.size() * 4, cudaMemcpyHostToDevice));
  }
  if (mtp_) {
    DGPP_CUDA_OK(cudaMalloc(&mtp_e_, M * H * 2));
    DGPP_CUDA_OK(cudaMalloc(&mtp_en_, M * H * 2));
    DGPP_CUDA_OK(cudaMalloc(&mtp_hn_, M * H * 2));
    DGPP_CUDA_OK(cudaMalloc(&mtp_hin_, M * H * 2));
    DGPP_CUDA_OK(cudaMalloc(&mtp_cat_, M * 2 * H * 2));
    DGPP_CUDA_OK(cudaMalloc(&mtp_r_, M * H * 2));
    DGPP_CUDA_OK(cudaMalloc(&mtp_h_, M * H * 2));
  }
  // The boundary prefetch windows' budget (run_rows): the measured default
  // (the 2026-10-04 sweep on the fabric), engine.l2_prefetch_boundary_window_mib
  // (0 takes the prefetcher's window budget).
  prefetch_window_bytes_ = l2_prefetch_settings().boundary_window_bytes;
  if (dflash2_) {
    // Weights (bf16, replicated; the shared embed/lm head ride globals_)
    // and the fp32 1/theta^(2i/128) rope table.
    if (!dfcfg_.tp_divisible(world))
      throw std::invalid_argument("Qwen35Model: the DFlash2 drafter's heads / kv heads / MLP rows must divide across the world");
    df_nh_ = dfcfg_.local_heads(world);
    df_kvh_ = dfcfg_.local_kv_heads(world);
    df_qw_ = dfcfg_.local_q_row(world);
    df_kvr_ = dfcfg_.local_kv_row(world);
    df_i_ = dfcfg_.local_intermediate(world);
    dfw_ = load_dflash2_weights(dfcfg_, dflash2_dir, stream_, rank, world, dflash2_fp8_);
    {
      std::vector<float> host(dfcfg_.head_dim / 2);
      const double theta = dfcfg_.rope_theta;
      for (size_t i = 0; i < host.size(); ++i)
        host[i] = static_cast<float>(
            std::pow(theta, -static_cast<double>(2 * i) / static_cast<double>(dfcfg_.head_dim)));
      DGPP_CUDA_OK(cudaMalloc(&df_inv_freq_, host.size() * 4));
      DGPP_CUDA_OK(cudaMemcpy(df_inv_freq_, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    }
    // Feature path at run rows; the block forward at query_rows rows (the
    // main scratch's gate/up buffers serve the draft MLPs at 8 rows).
    // The stacked draft batch (dflash2_draft_batch) widens the row-wise
    // scratch to df_batch_ slots: one verify batch's worth at the row
    // ceiling. The per-slot calls (attention, head, top-K, walk) run at
    // row offsets inside the same buffers.
    const int CR = df_rows_cap();
    const int QR = dfcfg_.query_rows();
    const int D = QR - 1;  // mask rows = drafts
    df_batch_ = std::max(1, max_decode_rows_ / QR);
    const int BR = df_batch_ * QR, BD = df_batch_ * D;
    DGPP_CUDA_OK(cudaMalloc(&df_acc_, M * H * 4));
    DGPP_CUDA_OK(cudaMalloc(&df_t32_, M * H * 4));
    DGPP_CUDA_OK(cudaMalloc(&df_norm_, static_cast<size_t>(CR) * H * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_kv_, static_cast<size_t>(CR) * 2 * df_kvr_ * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_resid_, static_cast<size_t>(BR) * H * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_x_, static_cast<size_t>(BR) * H * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_xc_, static_cast<size_t>(BR) * H * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_qkv_, static_cast<size_t>(BR) * (df_qw_ + 2 * df_kvr_) * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_q_, static_cast<size_t>(BR) * df_qw_ * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_attn_, static_cast<size_t>(BR) * df_qw_ * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_attn_part_, dflash2_block_attn_partials_bytes(BR, df_nh_)));
    DGPP_CUDA_OK(cudaMalloc(&df_o_, static_cast<size_t>(BR) * H * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_mlp_, static_cast<size_t>(BR) * df_i_ * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_gate_, static_cast<size_t>(BR) * df_i_ * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_up_, static_cast<size_t>(BR) * df_i_ * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_zero_, QR * 4));
    DGPP_CUDA_OK(cudaMemsetAsync(df_zero_, 0, QR * 4, stream_));
    DGPP_CUDA_OK(cudaMalloc(&df_delta_, static_cast<size_t>(BR) * 2 * dfcfg_.conv_taps * dfcfg_.conv_groups() * 2));
    DGPP_CUDA_OK(cudaMalloc(&df_h_, static_cast<size_t>(BR) * H * 2));
    // The head rows, top-K and hidden projection run once over every
    // stacked row (the anchor rows' entries unused): one head stream a
    // step instead of one per slot (2026-10-05; eight slots streamed the
    // head eight times — ~38 ms a step on one Spark).
    DGPP_CUDA_OK(cudaMalloc(&df_logits_, static_cast<size_t>(BR) * lm_vocab_count_ * 4));
    df_topk_ws_bytes_ = dflash2_topk_ws_bytes(lm_vocab_count_, BR, dfcfg_.selector_top_k);
    DGPP_CUDA_OK(cudaMalloc(&df_topk_ws_, df_topk_ws_bytes_));
    DGPP_CUDA_OK(cudaMalloc(&df_hidden32_, static_cast<size_t>(BR) * dfcfg_.selector_rank * 4));
    DGPP_CUDA_OK(cudaMalloc(&df_ids_, static_cast<size_t>(BR) * dfcfg_.selector_top_k * 4));
    DGPP_CUDA_OK(cudaMalloc(&df_sc_, static_cast<size_t>(BR) * dfcfg_.selector_top_k * 4));
    DGPP_CUDA_OK(cudaMalloc(&df_tok_, static_cast<size_t>(BD) * 4));
    DGPP_CUDA_OK(cudaMalloc(&df_pos_, static_cast<size_t>(BR) * 8));
    DGPP_CUDA_OK(cudaMalloc(&df_tokens_, static_cast<size_t>(BR) * 8));
    DGPP_CUDA_OK(cudaMallocHost(&df_tok_h_, static_cast<size_t>(BD) * 4));
    DGPP_CUDA_OK(cudaMallocHost(&df_io64_h_, static_cast<size_t>(2) * BR * 8));
    // World > 1: the ranks' slice top-K lists meet in one boundary fold
    // (the eager fold runs in place on df_table_; a capture folds the
    // recorder's staged buffer). The recorded drafts' pinned mirror, one
    // row per request slot.
    if (world > 1)
      DGPP_CUDA_OK(cudaMalloc(&df_table_, dflash2_topk_table_elems(BR, dfcfg_.selector_top_k, world) * 2));
    DGPP_CUDA_OK(cudaMallocHost(&df_mirror_h_, static_cast<size_t>(max_requests) * D * 4));
    DGPP_CUDA_OK(cudaMallocHost(&df_cands_h_, static_cast<size_t>(max_requests) * D * dfcfg_.selector_top_k * 4));
    DGPP_CUDA_OK(cudaMalloc(&df_conf_, static_cast<size_t>(max_requests) * D * 4));
    DGPP_CUDA_OK(cudaMemset(df_conf_, 0, static_cast<size_t>(max_requests) * D * 4));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }
  DGPP_CUDA_OK(cudaMalloc(&gdn_rec_base_, static_cast<size_t>(max_requests) * num_gdn_ * rec_elems_ * 4));
  DGPP_CUDA_OK(cudaMalloc(&gdn_conv_base_, static_cast<size_t>(max_requests) * num_gdn_ * conv_elems_ * 2));
  // The verify's per-row GDN conv snapshots (the speculative rollback's
  // source); the recurrent state replays instead (the checkpoint-and-replay
  // form: two copies of every slot's saved rows at the block's rows).
  const size_t spec_rows = static_cast<size_t>(max_decode_rows_);
  DGPP_CUDA_OK(
      cudaMalloc(&spec_conv_, spec_rows * static_cast<size_t>(num_gdn_) * conv_elems_ * 2));
  if (num_gdn_ > 0) {
    gdn_replay_ = true;
    gdn_replay_row_elems_ = ((C + 2 * lv) + 7) / 8 * 8;  // padded to 16 bytes: the commit's copy stays 16-byte-sized
    // The batched walk's launcher checks the batch's rows against the cap
    // (the kernel saves per request): the cap is the decode batch's rows,
    // as the Flash-Next model sizes it (1 GB at eight slots and 64 rows
    // against the 9.7 GB of per-row snapshots this replaces).
    gdn_replay_rows_cap_ = max_decode_rows_;
    const size_t per_req = gdn_replay_block_elems();
    DGPP_CUDA_OK(cudaMalloc(&gdn_replay_in_, static_cast<size_t>(max_requests) * per_req * 2));
    DGPP_CUDA_OK(cudaMalloc(&gdn_replay_save_, static_cast<size_t>(max_requests) * per_req * 2));
    DGPP_CUDA_OK(cudaMalloc(&gdn_pending_, static_cast<size_t>(max_requests) * 4));
    DGPP_CUDA_OK(cudaMemset(gdn_pending_, 0, static_cast<size_t>(max_requests) * 4));
  }
  if (Bf12Companions::enabled()) pack_companions();
}

// The bf16 matrices a decode pass streams through CublasLtGemm's GEMV
// lowering, packed to their 12-bit companions (kernels/bf12_companions):
// the drafter's five layers (qkv, o, gate, up, down) and its fc taps, the
// MTP fc, and the lm head when it serves in bf16. The bf16 bytes stay: the
// drafter's weights are one arena, the fc taps slices of one matrix, and the
// stacked redrafts (17+ rows) take the streaming mma form, which reads
// bf16 — so "bf12" and "bf12+bf16" are the same residency here.
void Qwen35Model::pack_companions() {
  const auto t0 = std::chrono::steady_clock::now();
  const int64_t H = cfg_.hidden_size;
  const auto pack = [&](const uint16_t* w, int64_t n, int64_t k) {
    if (w != nullptr) bf12_.pack(w, n, k, gemm_, stream_);
  };
  if (dflash2_) {
    const int64_t dH = dfcfg_.hidden_size, dI = df_i_;
    const int64_t QW = df_qw_, KV = df_kvr_;
    for (const DFlash2LayerWeights& w : dfw_.layers) {
      if (w.qkv_fp8 != nullptr) {  // the block matrices stream as fp8: only the k|v rows' companion
        pack(w.kv_rows, 2 * KV, dH);
        continue;
      }
      pack(w.qkv, QW + 2 * KV, dH);
      pack(w.o, dH, QW);
      pack(w.gate, dI, dH);
      pack(w.up, dI, dH);
      pack(w.down, dH, dI);
    }
    for (size_t t = 0; t < dfcfg_.target_layer_ids.size(); ++t)
      pack(dfw_.fc + t * static_cast<size_t>(dH) * static_cast<size_t>(dH), dH, dH);
    pack(dfw_.hidden_projection, dfcfg_.selector_rank, dH);
  }
  if (mtp_) pack(globals_.mtp_fc, H, 2 * H);
  if (!head_fp8_enabled_) pack(globals_.lm_head, lm_vocab_count_, H);
  bf12_.finish(gemm_, 1);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bf12_.log_summary(rank_, bf12_s_);
}

Qwen35Model::~Qwen35Model() {
  cudaFree(resid_);
  cudaFree(x_);
  cudaFree(attn_out_);
  cudaFree(mlp_out_);
  cudaFree(gate_tmp_);
  cudaFree(up_tmp_);
  cudaFree(mtp_e_);
  cudaFree(mtp_en_);
  cudaFree(mtp_hn_);
  cudaFree(mtp_hin_);
  cudaFree(mtp_cat_);
  cudaFree(mtp_r_);
  cudaFree(mtp_h_);
  cudaFree(df_inv_freq_);
  cudaFree(df_acc_);
  cudaFree(df_t32_);
  cudaFree(df_norm_);
  cudaFree(df_kv_);
  cudaFree(df_resid_);
  cudaFree(df_x_);
  cudaFree(df_xc_);
  cudaFree(df_qkv_);
  cudaFree(df_q_);
  cudaFree(df_attn_);
  cudaFree(df_attn_part_);
  cudaFree(df_o_);
  cudaFree(df_mlp_);
  cudaFree(df_gate_);
  cudaFree(df_up_);
  cudaFree(df_zero_);
  cudaFree(df_delta_);
  cudaFree(df_h_);
  cudaFree(df_logits_);
  cudaFree(df_topk_ws_);
  cudaFree(df_hidden32_);
  cudaFree(df_ids_);
  cudaFree(df_sc_);
  cudaFree(df_tok_);
  cudaFree(df_pos_);
  cudaFree(df_tokens_);
  cudaFree(df_tok_h_);
  cudaFree(df_io64_h_);
  cudaFree(df_table_);
  cudaFree(df_conf_);
  cudaFreeHost(df_mirror_h_);
  cudaFreeHost(df_cands_h_);
  cudaFree(gdn_rec_base_);
  cudaFree(gdn_conv_base_);
  cudaFree(spec_rec_);
  cudaFree(spec_conv_);
  cudaFree(gdn_replay_in_);
  cudaFree(gdn_replay_save_);
  cudaFree(gdn_pending_);
  cudaFree(gemm_ws_);
  if (gw_.dequant) cudaFree(gw_.dequant);
  cudaFree(pt_gate_);
  cudaFree(pt_up_);
  cudaFree(pt_down_);
  cudaFree(pt_scales_);
  cudaFree(pt_gdn_qkv_);
  cudaFree(pt_gdn_z_);
  cudaFree(pt_gdn_o_);
  cudaFree(pt_gdn_scales_);
  cudaFree(pt_full_q_);
  cudaFree(pt_full_k_);
  cudaFree(pt_full_v_);
  cudaFree(pt_full_o_);
  cudaFree(pt_full_scales_);
  cudaFree(head_fp8_);
  cudaFree(head_scales_);
  cudaFree(pt_act_);
  cudaFree(pt_act_scales_);
}

void Qwen35Model::configure_gemm_rows(int rows, bool decode) {
  qwen_configure_gemm_rows(gemm_, rows, decode);
  gemm_.set_bf12_wide(decode);  // the companions take the decode batch's 5..8-row calls too
  // 17..128 decode rows: the streaming tensor-core form reads a bf16 weight
  // once for every row of the launch where the kernel-only band's 4-row GEMV
  // chunks re-read it per chunk (eight times at the 32-row verify batch:
  // the 8K profile's draft phase, 260 -> 92 ms/pass). m <= 16 keeps its
  // dispatch, so the C1 verify and the lone-slot draft are unchanged.
  gemm_.set_decode_mma(decode && rows > 16, 17, kMmaGemvMaxRowsPerLaunch);
}

void Qwen35Model::build_layer_objects(const Qwen35LayerResident& r) {
  if (r.kind == Qwen35LayerKind::Gdn) {
    if (!gdn_)
      gdn_ = std::make_unique<QwenGdnLayer>(r.gdn, gw_, qcfg_, max_tokens_, true);
    else
      gdn_->rebind(r.gdn);
    gdn_->set_pt_attn(pt_gdn_view(r.layer));
  } else {
    if (!full_)
      full_ = std::make_unique<QwenFullAttnLayer>(r.full, gw_, qcfg_, max_tokens_);
    else
      full_->rebind(r.full);
    full_->set_pt_attn(pt_full_view(r.layer));
  }
}

// One GDN slot's boot requant: dequant each blockwise matrix to the bridge,
// then absmax + x/448 quantize into the slot.
void Qwen35Model::requant_gdn_pt(int slot, const QwenGdnResident& w, cudaStream_t stream) {
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  const size_t C = static_cast<size_t>(pt_gdn_C_), LV = static_cast<size_t>(pt_gdn_LV_);
  struct Task {
    const GlmQuantMatrix* src;
    uint8_t* dst;
    int n, k;
  };
  const Task tasks[3] = {
      {&w.in_proj_qkv_fp8, pt_gdn_qkv_ + static_cast<size_t>(slot) * C * H, static_cast<int>(C),
       static_cast<int>(H)},
      {&w.in_proj_z_fp8, pt_gdn_z_ + static_cast<size_t>(slot) * LV * H, static_cast<int>(LV),
       static_cast<int>(H)},
      {&w.out_proj_fp8, pt_gdn_o_ + static_cast<size_t>(slot) * H * LV, static_cast<int>(H),
       static_cast<int>(LV)},
  };
  for (int t = 0; t < 3; ++t) {
    const int n = tasks[t].n, k = tasks[t].k;
    launch_fp8_dequant_blocks(tasks[t].src->payload, tasks[t].src->scales, gw_.dequant, n, k,
                              stream);
    float* mx = pt_gdn_scales_ + static_cast<size_t>(slot) * 3 + t;
    launch_fp8_row_maxabs(gw_.dequant, static_cast<size_t>(n) * k, mx, stream);
    launch_fp8_quant_bf16(gw_.dequant, tasks[t].dst, static_cast<size_t>(n) * k, mx, stream);
  }
}

// One Full slot's boot requant: q/k/v off the hidden rows, o over them.
void Qwen35Model::requant_full_pt(int slot, const QwenFullAttnResident& w, cudaStream_t stream) {
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  const size_t QW = static_cast<size_t>(pt_full_QW_), KW = static_cast<size_t>(pt_full_KW_),
               FH = static_cast<size_t>(pt_full_FH_);
  struct Task {
    const GlmQuantMatrix* src;
    uint8_t* dst;
    int n, k;
  };
  const Task tasks[4] = {
      {&w.q_proj_fp8, pt_full_q_ + static_cast<size_t>(slot) * QW * H, static_cast<int>(QW),
       static_cast<int>(H)},
      {&w.k_proj_fp8, pt_full_k_ + static_cast<size_t>(slot) * KW * H, static_cast<int>(KW),
       static_cast<int>(H)},
      {&w.v_proj_fp8, pt_full_v_ + static_cast<size_t>(slot) * KW * H, static_cast<int>(KW),
       static_cast<int>(H)},
      {&w.o_proj_fp8, pt_full_o_ + static_cast<size_t>(slot) * H * FH, static_cast<int>(H),
       static_cast<int>(FH)},
  };
  for (int t = 0; t < 4; ++t) {
    const int n = tasks[t].n, k = tasks[t].k;
    launch_fp8_dequant_blocks(tasks[t].src->payload, tasks[t].src->scales, gw_.dequant, n, k,
                              stream);
    float* mx = pt_full_scales_ + static_cast<size_t>(slot) * 4 + t;
    launch_fp8_row_maxabs(gw_.dequant, static_cast<size_t>(n) * k, mx, stream);
    launch_fp8_quant_bf16(gw_.dequant, tasks[t].dst, static_cast<size_t>(n) * k, mx, stream);
  }
}

// The bound layer's attention view: GDN ordinals index the GDN slots, full
// ordinals the full slots (the MTP draft layer takes the last full slot).
// Anything unmapped (or PT off) yields a disabled view: the bridge path.
QwenPtAttnView Qwen35Model::pt_gdn_view(int layer) const {
  QwenPtAttnView v;
  if (!pt_attn_enabled_) return v;
  const int slot =
      (layer >= 0 && layer < static_cast<int>(pt_gdn_ord_.size())) ? pt_gdn_ord_[layer] : -1;
  if (slot < 0 || slot >= pt_gdn_slots_) return v;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  const size_t C = static_cast<size_t>(pt_gdn_C_), LV = static_cast<size_t>(pt_gdn_LV_);
  v.enabled = true;
  v.H = static_cast<int>(H);
  v.x_count = 2;
  v.xw[0] = pt_gdn_qkv_ + static_cast<size_t>(slot) * C * H;
  v.xrows[0] = static_cast<int>(C);
  v.xw[1] = pt_gdn_z_ + static_cast<size_t>(slot) * LV * H;
  v.xrows[1] = static_cast<int>(LV);
  v.xscales = pt_gdn_scales_ + static_cast<size_t>(slot) * 3;
  v.ow = pt_gdn_o_ + static_cast<size_t>(slot) * H * LV;
  v.oscale = pt_gdn_scales_ + static_cast<size_t>(slot) * 3 + 2;
  v.ocols = static_cast<int>(LV);
  v.act = pt_act_;
  v.act_scale = pt_act_scales_;
  return v;
}

QwenPtAttnView Qwen35Model::pt_full_view(int layer) const {
  QwenPtAttnView v;
  if (!pt_attn_enabled_) return v;
  int slot = -1;
  if (mtp_ && layer == cfg_.mtp_layer())
    slot = num_full_;
  else if (layer >= 0 && layer < static_cast<int>(pt_full_ord_.size()))
    slot = pt_full_ord_[layer];
  if (slot < 0 || slot >= pt_full_slots_) return v;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  const size_t QW = static_cast<size_t>(pt_full_QW_), KW = static_cast<size_t>(pt_full_KW_),
               FH = static_cast<size_t>(pt_full_FH_);
  v.enabled = true;
  v.H = static_cast<int>(H);
  v.x_count = 3;
  v.xw[0] = pt_full_q_ + static_cast<size_t>(slot) * QW * H;
  v.xrows[0] = static_cast<int>(QW);
  v.xw[1] = pt_full_k_ + static_cast<size_t>(slot) * KW * H;
  v.xrows[1] = static_cast<int>(KW);
  v.xw[2] = pt_full_v_ + static_cast<size_t>(slot) * KW * H;
  v.xrows[2] = static_cast<int>(KW);
  v.xscales = pt_full_scales_ + static_cast<size_t>(slot) * 4;
  v.ow = pt_full_o_ + static_cast<size_t>(slot) * H * FH;
  v.oscale = pt_full_scales_ + static_cast<size_t>(slot) * 4 + 3;
  v.ocols = static_cast<int>(FH);
  v.act = pt_act_;
  v.act_scale = pt_act_scales_;
  return v;
}

// One PT slot's boot requant: dequant the blockwise matrix to the bridge,
// then absmax + x/448 quantize into the slot (the bridge is idle at boot).
void Qwen35Model::requant_mlp_pt(int slot, const Qwen35DenseMlpResident& m, cudaStream_t stream) {
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  const size_t I = static_cast<size_t>(m.gate_fp8.rows);  // this rank's MLP slice
  const size_t IH = I * H;
  uint8_t* const dst[3] = {pt_gate_ + static_cast<size_t>(slot) * IH,
                           pt_up_ + static_cast<size_t>(slot) * IH,
                           pt_down_ + static_cast<size_t>(slot) * IH};
  const GlmQuantMatrix* const src[3] = {&m.gate_fp8, &m.up_fp8, &m.down_fp8};
  const int dims[3][2] = {{static_cast<int>(I), static_cast<int>(H)},
                          {static_cast<int>(I), static_cast<int>(H)},
                          {static_cast<int>(H), static_cast<int>(I)}};
  for (int t = 0; t < 3; ++t) {
    const int n = dims[t][0], k = dims[t][1];
    launch_fp8_dequant_blocks(src[t]->payload, src[t]->scales, gw_.dequant, n, k, stream);
    float* mx = pt_scales_ + static_cast<size_t>(slot) * 3 + t;
    launch_fp8_row_maxabs(gw_.dequant, static_cast<size_t>(n) * k, mx, stream);
    launch_fp8_quant_bf16(gw_.dequant, dst[t], static_cast<size_t>(n) * k, mx, stream);
  }
}

// The lm head over `rows` activation rows into F32 logits: the boot
// blockwise-FP8 head (half the bytes) when enabled, else the BF16 matmul.
// Boot-fixed addresses, so the branch replays under CUDA graphs.
void Qwen35Model::head_gemv(const uint16_t* act, float* out, int rows, cudaStream_t stream) {
  const int H = cfg_.hidden_size;
  const int64_t V = lm_vocab_count_;
  // Uniform FP8 dispatch: the weights-once streaming MMA at every decode
  // row count (~5 ms, half the BF16 bytes). Its row chains are bitwise
  // the same at any m (scale_gemm_test), so a request's logits are the
  // same alone or in a batch; the <= 4-row GEMV path it used to take
  // below five rows is a different chain, and a solo plain-decode request
  // (one row) diverged from its eight-slot batch (2026-10-05). Past 128
  // rows the blockwise dense kernel falls behind BF16 Lt (58ms vs 26ms at
  // m=500, measured), so wide heads (group walks, diagnostics) keep BF16.
  if (head_fp8_enabled_ && rows <= 128) {
    launch_scale_gemm_f32(act, static_cast<size_t>(H), head_fp8_, head_scales_, out, rows,
                          static_cast<int>(V), H, stream, static_cast<size_t>(V),
                          /*mma_from_rows=*/1);
    return;
  }
  gemm_.matmul(act, globals_.lm_head, out, rows, static_cast<int>(V), H, DType::BF16, GemmOut::F32,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
}

void Qwen35Model::dense_mlp(const uint16_t* x, uint16_t* out, int tokens,
                            const Qwen35DenseMlpResident& m, cudaStream_t stream, int layer,
                            bool resume) {
  // I is this rank's MLP slice (the resident gate rows): intermediate_size /
  // world under TP, the whole at world 1.
  const int64_t H = cfg_.hidden_size, I = m.gate_fp8.rows;
  // Per-tensor FP8 recipe: one shared activation quantize over the H rows
  // feeds both gate and up; the swiglu output is quantized once for down.
  // All addresses are boot-fixed (slots, scratch, scale cells), so the
  // branch replays under CUDA graphs like the bridge below it. resume marks
  // a prefill continuation chunk (pos0 > 0): it takes this path at any row
  // count, like the attention projections' resume.
  if (pt_enabled_ && (tokens > 128 || resume) && layer >= 0 && layer < pt_slots_) {
    const size_t IH = static_cast<size_t>(I) * static_cast<size_t>(H);
    float* const asc = pt_act_scales_;
    launch_fp8_row_maxabs(x, static_cast<size_t>(tokens) * H, asc, stream);
    launch_fp8_quant_bf16(x, pt_act_, static_cast<size_t>(tokens) * H, asc, stream);
    float* const ws = pt_scales_ + static_cast<size_t>(layer) * 3;
    gemm_.matmul_fp8_scaled(pt_act_, pt_gate_ + static_cast<size_t>(layer) * IH, asc, ws, gate_tmp_,
                            tokens, static_cast<int>(I), static_cast<int>(H), gw_.ws,
                            gw_.ws_bytes, stream);
    gemm_.matmul_fp8_scaled(pt_act_, pt_up_ + static_cast<size_t>(layer) * IH, asc, ws + 1, up_tmp_,
                            tokens, static_cast<int>(I), static_cast<int>(H), gw_.ws,
                            gw_.ws_bytes, stream);
    qwen35_swiglu_bf16(gate_tmp_, up_tmp_, gate_tmp_, static_cast<int64_t>(tokens) * I, stream);
    launch_fp8_row_maxabs(gate_tmp_, static_cast<size_t>(tokens) * I, asc + 1, stream);
    launch_fp8_quant_bf16(gate_tmp_, pt_act_, static_cast<size_t>(tokens) * I, asc + 1, stream);
    gemm_.matmul_fp8_scaled(pt_act_, pt_down_ + static_cast<size_t>(layer) * IH, asc + 1, ws + 2,
                            out, tokens, static_cast<int>(H), static_cast<int>(I), gw_.ws,
                            gw_.ws_bytes, stream);
    return;
  }
  // Prefill-shaped products run the dequant bridge + cuBLASLt BF16 (the same
  // lowering gemm_dense uses for the attention projections): nsys showed the
  // streaming tile kernel owning ~70% of a 2K prefill at ~24 TFLOP/s, while
  // the bridge scratch (178MB) sat unused by this direct caller.
  if (tokens > 128 && gw_.dequant) {
    const size_t up_bytes = static_cast<size_t>(I) * static_cast<size_t>(H) * 2;
    const size_t down_bytes = static_cast<size_t>(H) * static_cast<size_t>(I) * 2;
    if (up_bytes <= gw_.dequant_bytes && down_bytes <= gw_.dequant_bytes) {
      launch_fp8_dequant_blocks(m.gate_fp8.payload, m.gate_fp8.scales, gw_.dequant, I, H, stream);
      gw_.gemm->matmul(x, gw_.dequant, gate_tmp_, tokens, static_cast<int>(I), static_cast<int>(H),
                       DType::BF16, GemmOut::BF16, static_cast<size_t>(H), gw_.ws, gw_.ws_bytes,
                       stream);
      launch_fp8_dequant_blocks(m.up_fp8.payload, m.up_fp8.scales, gw_.dequant, I, H, stream);
      gw_.gemm->matmul(x, gw_.dequant, up_tmp_, tokens, static_cast<int>(I), static_cast<int>(H),
                       DType::BF16, GemmOut::BF16, static_cast<size_t>(H), gw_.ws, gw_.ws_bytes,
                       stream);
      qwen35_swiglu_bf16(gate_tmp_, up_tmp_, gate_tmp_, static_cast<int64_t>(tokens) * I, stream);
      launch_fp8_dequant_blocks(m.down_fp8.payload, m.down_fp8.scales, gw_.dequant, H, I, stream);
      gw_.gemm->matmul(gate_tmp_, gw_.dequant, out, tokens, static_cast<int>(H), static_cast<int>(I),
                       DType::BF16, GemmOut::BF16, static_cast<size_t>(I), gw_.ws, gw_.ws_bytes,
                       stream);
      return;
    }
  }
  // The matmul workspace lets the streaming form split K when a shard's
  // rows leave the grid under-filled (#93, 2026-10-04: the four-node gate /
  // up [4352 x 5120] ran 68 unsplit blocks at 184 GB/s; split 2 reads at
  // 208, the chunked cores at m <= 4 are unaffected). The split count is a
  // function of the shape only, so a row's chain is the same whatever m.
  launch_scale_gemm_bf16(x, static_cast<size_t>(H), m.gate_fp8.payload, m.gate_fp8.scales, gate_tmp_,
                         tokens, static_cast<int>(I), static_cast<int>(H), stream, 0, gw_.mma_from_rows,
                         gw_.ws, gw_.ws_bytes);
  launch_scale_gemm_bf16(x, static_cast<size_t>(H), m.up_fp8.payload, m.up_fp8.scales, up_tmp_, tokens,
                         static_cast<int>(I), static_cast<int>(H), stream, 0, gw_.mma_from_rows,
                         gw_.ws, gw_.ws_bytes);
  qwen35_swiglu_bf16(gate_tmp_, up_tmp_, gate_tmp_, static_cast<int64_t>(tokens) * I, stream);
  // The down projection's k = I = 17408 fits one activation row in the GEMV's
  // 48 KiB staging budget, so a 3-row MTP pass read this 89 MB matrix three
  // times (nsys 2026-10-03: 199 single-row launches x 373 us = 74 ms of a
  // 180 ms step). The streaming mma form reads it once at any row count, and
  // its per-row chain is the same whatever m, so the T=1 world and the MTP
  // verify stay bitwise (4/4 transcripts); it takes the C1 MTP pass from 197
  // to 151 ms and costs the 1-row T=1 step 8 ms (119 vs 111: the form streams
  // this matrix at ~215 GB/s against the GEMV's 240 — a kernel item).
  launch_scale_gemm_bf16(gate_tmp_, static_cast<size_t>(I), m.down_fp8.payload, m.down_fp8.scales, out,
                         tokens, static_cast<int>(H), static_cast<int>(I), stream, 0, gw_.mma_from_rows);
}

size_t Qwen35Model::session_snapshot_bytes(const Qwen35TextConfig& cfg, int world, bool mtp) {
  (void)world;
  (void)mtp;  // no draft state: MTP is off
  int num_gdn = 0;
  for (Qwen35LayerKind k : cfg.layers)
    if (k == Qwen35LayerKind::Gdn) ++num_gdn;
  const int64_t lv = cfg.gdn_value_heads, V = cfg.gdn_value_head_dim, K = cfg.gdn_key_head_dim;
  const int64_t C = 2 * static_cast<int64_t>(cfg.gdn_key_heads) * K + lv * V;
  const size_t rec = static_cast<size_t>(lv) * V * K;
  const size_t conv = static_cast<size_t>(C) * (cfg.gdn_conv_width - 1);
  size_t bytes = static_cast<size_t>(num_gdn) * (rec * 4 + conv * 2);
  if (mtp) bytes += static_cast<size_t>(cfg.hidden_size) * 2;  // the core window row write_snapshot appends
  return bytes;
}

size_t Qwen35Model::snapshot_state_bytes() const {
  return static_cast<size_t>(num_gdn_) * (rec_elems_ * 4 + conv_elems_ * 2);
}

MemoryPlan Qwen35Model::plan_memory(const Qwen35TextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                    int rank, int world, LoaderResidency residency, int max_requests,
                                    bool mtp, int decode_rows, const std::string& dflash2_dir) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests out of range");
  if (mtp && cfg.mtp_layer() < 0)
    throw std::invalid_argument("plan_memory: the config has no draft layer (mtp)");
  std::unique_ptr<DFlash2Config> dfcfg;
  if (!dflash2_dir.empty()) {
    if (mtp)
      throw std::invalid_argument("plan_memory: dflash2 replaces the MTP draft; enable one or the other");
    dfcfg = std::make_unique<DFlash2Config>(DFlash2Config::from_json_file(dflash2_dir + "/config.json"));
    dfcfg->validate_against(cfg);
  }
  const LoaderHeadSharding head = world > 1 ? LoaderHeadSharding::VocabSharded : LoaderHeadSharding::Full;
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kv_block_tokens_static() - 1) /
       kv_block_tokens_static()) *
      kv_block_tokens_static();
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.max_position_embeddings);
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t I = static_cast<size_t>(cfg.intermediate_size);
  if (residency == LoaderResidency::Resident) {
    plan.add("model weights (resident)",
             Qwen35LayerStream::resident_bytes(cfg, rank, world, head, mtp));
    plan.add("loader staging (pinned host, freed when the last layer is resident)", 0,
             Qwen35LayerStream::staging_plan_bytes(cfg, rank, world, head, mtp));
  } else {
    size_t largest = 0;
    for (int l = 0; l < cfg.num_hidden_layers; ++l)
      largest = std::max(largest, Qwen35LayerStream::layer_bytes(cfg, l, rank, world));
    plan.add("model weights (one streamed layer + globals)",
             largest + Qwen35LayerStream::globals_bytes(cfg, rank, world, head));
  }
  Qwen35KvPoolShape shape;
  shape.layers = 0;
  for (Qwen35LayerKind k : cfg.layers)
    if (k != Qwen35LayerKind::Gdn) ++shape.layers;
  if (mtp) ++shape.layers;  // the draft plane
  shape.kv_heads = Qwen35LocalGeometry::from_config(cfg, rank, world, head).local_kv_heads;
  shape.dim = cfg.head_dim;
  shape.block_tokens = kv_block_tokens_static();
  shape.max_requests = max_requests;
  shape.token_slots = cache_tokens;
  if (dfcfg) {
    shape.draft_layers = dfcfg->num_hidden_layers;
    shape.draft_kv_heads = dfcfg->local_kv_heads(world);
    shape.draft_dim = dfcfg->head_dim;
  }
  plan.add("kv pool", Qwen35KvPool::cache_bytes(shape));
  // Layer scratch at max_tokens rows (Full + GDN worst case).
  const QwenTextConfig qc = qwen_text_adapter(cfg);
  const int lh = cfg.num_attention_heads, lkv = cfg.num_key_value_heads;
  const int lk = cfg.gdn_key_heads, lv = cfg.gdn_value_heads;
  plan.add("layer scratch", std::max(QwenFullAttnLayer::scratch_bytes(qc, lh, lkv, max_tokens),
                                    QwenGdnLayer::scratch_bytes(qc, lk, lv, max_tokens)));
  // Activations: resid/x/attn/mlp [M,H] + gate/up tmps [M,I].
  plan.add("activations", 4 * M * H * 2 + 2 * M * I * 2);
  // Dense FP8 prefill bridge: the largest dense matrix dequantized to BF16.
  plan.add("dense fp8 prefill bridge (largest dense matrix in BF16)",
           qwen35_dense_bridge_bytes(cfg));
  // Per-tensor FP8 recipe (engine.prefill_fp8_per_tensor, Resident only):
  // boot-time E4M3 gate/up/down per layer plus one scale each, activation
  // scratch. Streaming stacks keep the bridge (nothing eager to build).
  if (prefill_fp8_per_tensor_ && residency == LoaderResidency::Resident) {
    const size_t slots = static_cast<size_t>(cfg.num_hidden_layers) + (mtp ? 1 : 0);
    const size_t IH = static_cast<size_t>(cfg.intermediate_size) * cfg.hidden_size;
    plan.add("per-tensor fp8 mlp (gate/up/down E4M3 + scales)", 3 * slots * IH + slots * 3 * 4);
    plan.add("per-tensor fp8 activation scratch", M * I + 8);
  }
  if (prefill_fp8_per_tensor_ && residency == LoaderResidency::Resident) {
    // Attention projections, the same recipe: GDN qkv/z/out per GDN layer,
    // Full q/k/v/o per full layer plus the MTP draft's. The activation
    // scratch above is shared (the sites run sequentially).
    int num_gdn = 0, num_full = 0;
    for (Qwen35LayerKind k : cfg.layers) (k == Qwen35LayerKind::Gdn ? num_gdn : num_full)++;
    const size_t Hp = static_cast<size_t>(cfg.hidden_size);
    const size_t C = 2 * static_cast<size_t>(cfg.gdn_key_heads) * cfg.gdn_key_head_dim +
                     static_cast<size_t>(cfg.gdn_value_heads) * cfg.gdn_value_head_dim;
    const size_t LV = static_cast<size_t>(cfg.gdn_value_heads) * cfg.gdn_value_head_dim;
    plan.add("per-tensor fp8 gdn attention (qkv/z/out E4M3 + scales)",
             static_cast<size_t>(num_gdn) * (C * Hp + LV * Hp + Hp * LV) +
                 static_cast<size_t>(num_gdn) * 3 * 4);
    const size_t QW = 2 * static_cast<size_t>(cfg.num_attention_heads) * cfg.head_dim;
    const size_t KW = static_cast<size_t>(cfg.num_key_value_heads) * cfg.head_dim;
    const size_t FH = static_cast<size_t>(cfg.num_attention_heads) * cfg.head_dim;
    const size_t full_slots = static_cast<size_t>(num_full) + (mtp ? 1 : 0);
    plan.add("per-tensor fp8 full attention (q/k/v/o E4M3 + scales)",
             full_slots * (QW * Hp + 2 * KW * Hp + Hp * FH) + full_slots * 4 * 4);
  }
  // Blockwise-FP8 lm head (engine.dense_weights = fp8, Resident only): half
  // the bytes per decode row.
  if (dense_weights_fp8_ && residency == LoaderResidency::Resident) {
    const size_t Vv = static_cast<size_t>(cfg.vocab_size), Hh = static_cast<size_t>(cfg.hidden_size);
    plan.add("blockwise fp8 lm head (E4M3 + scales)", Vv * Hh + ((Vv + 127) / 128) * ((Hh + 127) / 128) * 4);
  }
  if (mtp) {
    // Draft scratch: e/en/hn/hin/r/h [M,H] + cat [M,2H].
    plan.add("mtp scratch", 8 * M * H * 2);
  }
  if (Bf12Companions::enabled()) {
    // The 12-bit companions of the bf16 decode matrices (pack_companions):
    // the drafter's layers and fc taps, the MTP fc, the bf16 lm head.
    size_t packed = 0;
    if (dfcfg) {
      const int64_t dH = dfcfg->hidden_size, dI = dfcfg->local_intermediate(world);
      const int64_t QW = dfcfg->local_q_row(world), KV = dfcfg->local_kv_row(world);
      if (!dflash2_fp8_)  // the fp8 block matrices take no companions (pack_companions)
        packed += static_cast<size_t>(dfcfg->num_hidden_layers) *
                  (Bf12Companions::planned_bytes(QW + 2 * KV, dH) + Bf12Companions::planned_bytes(dH, QW) +
                   2 * Bf12Companions::planned_bytes(dI, dH) + Bf12Companions::planned_bytes(dH, dI));
      else  // only the k|v rows keep a bf16 copy (the feature path), packed
        packed += static_cast<size_t>(dfcfg->num_hidden_layers) * Bf12Companions::planned_bytes(2 * KV, dH);
      packed += dfcfg->target_layer_ids.size() * Bf12Companions::planned_bytes(dH, dH);
      packed += Bf12Companions::planned_bytes(dfcfg->selector_rank, dH);
    }
    if (mtp) packed += Bf12Companions::planned_bytes(H, 2 * H);
    if (!(dense_weights_fp8_ && residency == LoaderResidency::Resident))
      packed += Bf12Companions::planned_bytes(cfg.vocab_size, H);  // the head serves in bf16
    plan.add("bf16 decode packing (12-bit companions; the bf16 bytes stay)", packed);
  }
  if (dfcfg) {
    if (!dfcfg->tp_divisible(world))
      throw std::invalid_argument("plan_memory: the DFlash2 drafter's heads / kv heads / MLP rows must divide across the world");
    plan.add(dflash2_fp8_ ? "dflash2 drafter weights (this rank's slices; the block matrices as fp8, the rest bf16)"
                        : "dflash2 drafter weights (this rank's slices)",
             dflash2_weights_bytes(*dfcfg, world, dflash2_fp8_));
    const size_t QR = static_cast<size_t>(dfcfg->query_rows());
    // The stacked draft batch (dflash2_draft_batch): one verify batch's
    // slots per forward — max_decode_rows_ is max(decode_rows,
    // max_requests) floored at kDecodeRows, over QR rows per slot.
    const size_t BW =
        std::max<size_t>(1, static_cast<size_t>(std::max({kDecodeRows, decode_rows, max_requests})) / QR);
    const size_t CR = static_cast<size_t>(Qwen35Model::df_rows_cap());
    const size_t dI = static_cast<size_t>(dfcfg->local_intermediate(world));
    const size_t dQW = static_cast<size_t>(dfcfg->local_q_row(world)), dKV = static_cast<size_t>(dfcfg->local_kv_row(world));
    plan.add("dflash2 feature path",
             2 * M * H * 4 + CR * H * 2 + CR * 2 * dKV * 2 +
                 dfcfg->head_dim / 2 * 4);  // acc + tap F32, norm/kv scratch, rope table
    plan.add("dflash2 block scratch",
             BW * QR * (6 * H * 2 + (dQW + 2 * dKV) * 2 + dQW * 2 + 2 * dI * 2 +
                   2 * dfcfg->conv_taps * dfcfg->conv_groups() * 2));
    plan.add("dflash2 candidate buffers",
             BW * static_cast<size_t>(dfcfg->drafts()) *
                 (static_cast<size_t>(cfg.vocab_size) * 4 + dfcfg->selector_rank * 4 +
                  dfcfg->selector_top_k * (4 + 4)) +
                 BW * QR * 16 + BW * dfcfg->drafts() * 4);
  }
  // GDN recurrent + conv state per request slot, plus the verify's
  // per-row snapshots the speculative rollback reads.
  plan.add("gdn states", static_cast<size_t>(max_requests) * session_snapshot_bytes(cfg, world, mtp));
  {
    int num_gdn = 0;
    for (Qwen35LayerKind k : cfg.layers)
      if (k == Qwen35LayerKind::Gdn) ++num_gdn;
    const int64_t lv = cfg.gdn_value_heads, V = cfg.gdn_value_head_dim, K = cfg.gdn_key_head_dim;
    const int64_t C = 2 * static_cast<int64_t>(cfg.gdn_key_heads) * K + lv * V;
    const size_t conv = static_cast<size_t>(C) * (cfg.gdn_conv_width - 1);
    const size_t rows = static_cast<size_t>(std::max({kDecodeRows, decode_rows, max_requests}));
    plan.add("spec conv snapshot rows", rows * static_cast<size_t>(num_gdn) * conv * 2);
    // The recurrent state's replay rows: two copies of every slot's block
    // (the post-conv row + a_raw + beta_raw, padded to 16 bytes).
    const size_t row_elems = ((static_cast<size_t>(C) + 2 * static_cast<size_t>(lv)) + 7) / 8 * 8;
    const size_t rows_cap = static_cast<size_t>(std::max({kDecodeRows, decode_rows, max_requests}));
    plan.add("gdn replay rows (two copies)",
             2 * static_cast<size_t>(max_requests) * static_cast<size_t>(num_gdn) * rows_cap * row_elems * 2);
  }
  return plan;
}

void Qwen35Model::reset_slot_state(int req) {
  if (num_gdn_ > 0) {
    DGPP_CUDA_OK(cudaMemsetAsync(gdn_rec(req, 0), 0, static_cast<size_t>(num_gdn_) * rec_elems_ * 4, stream_));
    DGPP_CUDA_OK(
        cudaMemsetAsync(gdn_conv(req, 0), 0, static_cast<size_t>(num_gdn_) * conv_elems_ * 2, stream_));
    if (gdn_replay_) DGPP_CUDA_OK(cudaMemsetAsync(gdn_pending_ + req, 0, sizeof(int32_t), stream_));
  }
  pool_.reset_request(req, stream_);
}

// One request's saved rows, every GDN layer: [rows_cap][num_gdn][row] —
// row-major by row, then layer, so the rows a pass used (its first `rows`
// per request) are one contiguous span the commit copies (the Flash-Next
// model's layout).
size_t Qwen35Model::gdn_replay_block_elems() const {
  return static_cast<size_t>(num_gdn_) * static_cast<size_t>(gdn_replay_rows_cap_) *
         static_cast<size_t>(gdn_replay_row_elems_);
}

KdaReplay Qwen35Model::gdn_replay_view(int req, int ordinal, bool batched) const {
  KdaReplay rep;
  const size_t per_req = gdn_replay_block_elems();
  const size_t layer_off = static_cast<size_t>(ordinal) * static_cast<size_t>(gdn_replay_row_elems_);
  // A batched walk indexes the request inside the kernel (request_ids); a
  // single-request walk hands it the request's block.
  const size_t req_off = batched ? 0 : static_cast<size_t>(req) * per_req;
  rep.in = gdn_replay_in_ + req_off + layer_off;
  rep.save = gdn_replay_save_ + req_off + layer_off;
  rep.count = gdn_pending_ + (batched ? 0 : req);
  rep.in_stride = static_cast<int64_t>(num_gdn_) * gdn_replay_row_elems_;
  rep.request_stride = static_cast<int64_t>(per_req);
  rep.rows_cap = gdn_replay_rows_cap_;
  rep.checkpoint = true;
  return rep;
}

void Qwen35Model::materialize_gdn(int req, float* dst, int rows) {
  if (!gdn_replay_ || num_gdn_ <= 0) return;
  if (rows > gdn_replay_rows_cap_) throw std::invalid_argument("materialize_gdn: rows past the saved rows");
  int ordinal = 0;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const Qwen35LayerResident& r = loader_.load_layer(layer);
    if (r.kind != Qwen35LayerKind::Gdn) continue;
    build_layer_objects(r);
    KdaReplay rep = gdn_replay_view(req, ordinal, /*batched=*/false);
    rep.checkpoint = false;
    rep.save = nullptr;
    rep.materialize = true;
    rep.materialize_rows = rows;
    rep.dst = dst ? dst + static_cast<size_t>(ordinal) * rec_elems_ : nullptr;
    gdn_->materialize(gdn_rec(req, ordinal), rep, stream_);
    ++ordinal;
  }
  if (dst == nullptr) DGPP_CUDA_OK(cudaMemsetAsync(gdn_pending_ + req, 0, sizeof(int32_t), stream_));
}

GlmSpecSegments Qwen35Model::spec_segments(int req, int snapshot_row0) const {
  GlmSpecSegments segs;
  const auto add = [&](void* dst, const void* snapshots, size_t row_stride, size_t bytes) {
    if (segs.count >= kSpecMaxSegments) throw std::logic_error("spec_segments: too many state families");
    segs.seg[segs.count++] = GlmSpecSegment{dst, snapshots, row_stride, bytes};
  };
  const size_t row0 = static_cast<size_t>(snapshot_row0);
  if (num_gdn_ > 0) {
    const size_t rec_bytes = static_cast<size_t>(num_gdn_) * rec_elems_ * 4;
    const size_t conv_bytes = static_cast<size_t>(num_gdn_) * conv_elems_ * 2;
    if (gdn_replay_) {
      // The recurrent state replays: the commit copies this pass's saved
      // rows (one contiguous span: the used rows of every layer) over the
      // replay source and records the accepted count.
      const size_t per_req = gdn_replay_block_elems();
      const size_t used_rows = static_cast<size_t>(std::clamp(gdn_replay_rows_used_, 1, gdn_replay_rows_cap_));
      segs.replay_dst = gdn_replay_in_ + static_cast<size_t>(req) * per_req;
      segs.replay_src = gdn_replay_save_ + static_cast<size_t>(req) * per_req;
      segs.replay_bytes = used_rows * static_cast<size_t>(num_gdn_) * static_cast<size_t>(gdn_replay_row_elems_) * 2;
      segs.replay_pending = gdn_pending_ + req;
    } else {
      add(gdn_rec(req, 0), spec_rec_ + row0 * num_gdn_ * rec_elems_, rec_bytes, rec_bytes);
    }
    add(gdn_conv(req, 0), spec_conv_ + row0 * num_gdn_ * conv_elems_, conv_bytes, conv_bytes);
  }
  return segs;
}

void Qwen35Model::write_state_snapshot(int req, uint8_t* d, int spec_row) {
  const bool live = spec_row < 0;
  const size_t row = live ? 0 : static_cast<size_t>(spec_row);
  if (num_gdn_ == 0) return;
  const size_t rec_bytes = static_cast<size_t>(num_gdn_) * rec_elems_ * 4;
  const size_t conv_bytes = static_cast<size_t>(num_gdn_) * conv_elems_ * 2;
  if (gdn_replay_) {
    // The replay form: the live state materialized in place (the pending
    // rows applied, then copied), or the state after spec row `row`
    // replayed straight into the snapshot.
    if (live) {
      materialize_gdn(req);
      d2d(d, gdn_rec(req, 0), rec_bytes, stream_);
    } else {
      materialize_gdn(req, reinterpret_cast<float*>(d), spec_row + 1);
    }
  } else {
    d2d(d, live ? gdn_rec(req, 0) : spec_rec_ + row * num_gdn_ * rec_elems_, rec_bytes, stream_);
  }
  d2d(d + rec_bytes, live ? gdn_conv(req, 0) : spec_conv_ + row * num_gdn_ * conv_elems_, conv_bytes,
      stream_);
}

void Qwen35Model::read_state_snapshot(int req, const uint8_t* d) {
  if (num_gdn_ == 0) return;
  const size_t rec_bytes = static_cast<size_t>(num_gdn_) * rec_elems_ * 4;
  const size_t conv_bytes = static_cast<size_t>(num_gdn_) * conv_elems_ * 2;
  d2d(gdn_rec(req, 0), d, rec_bytes, stream_);
  d2d(gdn_conv(req, 0), d + rec_bytes, conv_bytes, stream_);
  if (gdn_replay_) DGPP_CUDA_OK(cudaMemsetAsync(gdn_pending_ + req, 0, sizeof(int32_t), stream_));
}

void Qwen35Model::graph_prepare() {
  if (loader_.residency() != LoaderResidency::Resident)
    throw std::logic_error("graph_prepare: the decode graph needs a resident stack");
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const Qwen35LayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
  }
  if (mtp_) {
    const Qwen35LayerResident& r = loader_.load_layer(cfg_.mtp_layer());
    build_layer_objects(r);
  }
}

// The MTP draft block: embed + hidden fusion through the fused fc [H, 2H],
// one Full draft layer + dense SwiGLU MLP over the fused rows, mtp.norm,
// then the shared lm head. Prefill rows read the main chunk's final hidden
// in place (h_); decode rows gather theirs from the slots' windows by
// position. head_rows == 0 fills the draft K/V only (prefill cache fill).
void Qwen35Model::mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T,
                               bool decode_row, bool capture, int head_rows, int batch_requests) {
  configure_gemm_rows(T, decode_row);
  if (!mtp_) throw std::logic_error("mtp_run_rows: MTP is not enabled");
  if (T <= 0 || T > max_tokens_) throw std::invalid_argument("mtp_run_rows: rows");
  if (head_rows < 0 || head_rows > T) throw std::invalid_argument("mtp_run_rows: head_rows");
  if (capture && loader_.residency() != LoaderResidency::Resident)
    throw std::logic_error("mtp_run_rows: a capture needs a resident stack");
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  const bool batched = batch_requests > 0;
  const int num_requests = batched ? batch_requests : 1;
  const int64_t* d_pos = decode_row ? d_step_pos_ : d_prefill_pos_;
  const int32_t* d_req = decode_row ? d_req_ids_ : d_prefill_req_;
  const int32_t* d_spans = decode_row ? d_req_spans_ : d_prefill_spans_;
  if (!decode_row) stage_prefill_meta(req, first_pos, T);

  // ---- the input fusion --------------------------------------------------
  // The reference (vLLM Qwen3_5MultiTokenPredictor): e = pre_fc_norm_embedding
  // (embed(tok)), h = pre_fc_norm_hidden(main hidden), cat([e, h]) -> fc.
  const uint16_t* hin = h_;
  if (decode_row) {
    gather_draft_hidden(d_req, d_pos, mtp_hin_, T);
    hin = mtp_hin_;
  }
  embed_gather_bf16(globals_.embed, tokens, mtp_e_, T, H, stream_);
  qwen_rmsnorm_bf16(mtp_e_, globals_.mtp_pre_fc_norm_embedding, mtp_en_, T, H, eps, stream_);
  qwen_rmsnorm_bf16(hin, globals_.mtp_pre_fc_norm_hidden, mtp_hn_, T, H, eps, stream_);
  qwen35_mtp_concat_bf16(mtp_en_, mtp_hn_, mtp_cat_, T, H, stream_);
  gemm_.matmul(mtp_cat_, globals_.mtp_fc, mtp_r_, T, H, 2 * H, DType::BF16, GemmOut::BF16,
               static_cast<size_t>(2 * H), gemm_ws_, gemm_ws_bytes_, stream_);

  // ---- the draft layer (the stack's objects rebound to its weights) ------
  const Qwen35LayerResident& r = loader_.load_layer(cfg_.mtp_layer());
  build_layer_objects(r);
  qwen_rmsnorm_bf16(mtp_r_, r.input_norm, x_, T, H, eps, stream_);
  // The draft layer's boundary folds (run_rows' pattern).
  const auto stage = [&](uint16_t* fallback, int width) -> uint16_t* {
    if (!boundary_) return fallback;
    uint16_t* s = boundary_->stage(T, width);
    if (s == nullptr && capture)
      throw std::runtime_error("mtp_run_rows: a capture fold does not fit the recorder's staged buffer");
    return s ? s : fallback;
  };
  const auto fold = [&](uint16_t* buf, int width) {
    if (!boundary_) return;
    if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    boundary_->reduce(buf, T, width);
  };
  uint16_t* ao = stage(attn_out_, H);
  {
    QwenFullAttnCache cache = pool_.view(num_full_);
    QwenQsaRows qrows;
    qrows.req_ids = d_req;
    qrows.pos = d_pos;
    qrows.decode = decode_row;
    qrows.request = req;
    qrows.pos0 = first_pos;
    qrows.spans = d_spans;
    qrows.num_requests = num_requests;
    full_->enqueue(x_, T, qrows, cache, ao, stream_);
  }
  fold(ao, H);
  // Fused residual-add + post norm (bitwise the pair): one launch.
  qwen_add_rmsnorm_bf16(mtp_r_, ao, r.post_norm, x_, T, H, eps, stream_);
  uint16_t* mo = stage(mlp_out_, H);
  dense_mlp(x_, mo, T, r.mlp, stream_, cfg_.num_hidden_layers, first_pos > 0 && !decode_row);
  fold(mo, H);
  add_inplace_bf16(mtp_r_, mo, static_cast<size_t>(T) * H, stream_);
  if (head_rows == 0) return;  // prefill rows fill the cache; no head

  // ---- head: the draft distribution over the last head_rows rows --------
  // mtp.norm runs over every row so draft_hidden_rows() stays row-indexed
  // for the chain; the head reads the last head_rows normed rows.
  qwen_rmsnorm_bf16(mtp_r_, globals_.mtp_norm, mtp_h_, T, H, eps, stream_);
  const uint16_t* head_in = mtp_h_ + static_cast<size_t>(T - head_rows) * H;
  const int64_t V = lm_vocab_count_;
  head_gemv(head_in, logits_, head_rows, stream_);
  if (decode_row && (!capture || decode_tail_mirrors_) && head_rows <= max_decode_rows_)
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_,
                                 static_cast<size_t>(head_rows) * V * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}

Qwen35Model::Outputs Qwen35Model::forward(const std::vector<int64_t>& token_ids, bool capture_layers) {
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("forward: empty token batch");
  if (T > max_tokens_) throw std::invalid_argument("forward: tokens exceed max_tokens");
  if (T > max_context_) throw std::invalid_argument("forward: tokens exceed the context bound");
  for (int64_t id : token_ids)
    if (id < 0 || id >= cfg_.vocab_size) throw std::invalid_argument("forward: token id out of range");
  if (session_pos_[0] != 0) throw std::logic_error("forward: slot 0 holds an open session");
  open_slot(0);
  if (!pool_.ensure_request_blocks(0, T, stream_))
    throw std::runtime_error("forward: the cache pool cannot cover the batch");
  RowRun run;
  run.req = 0;
  run.ids = token_ids.data();
  run.T = T;
  run.pos0 = 0;
  run.decode = false;
  run.all_rows = true;
  run.capture_layers = capture_layers;
  Outputs out = run_rows(run);
  session_close(0);
  return out;
}

// The cold diagnostic forward: slot 0, fresh state, every row through the
// main stack, then the draft block over the shifted tokens.
Qwen35Model::Outputs Qwen35Model::mtp_forward(const std::vector<int64_t>& token_ids) {
  if (!mtp_) refuse_mtp("mtp_forward");
  const int T = static_cast<int>(token_ids.size());
  if (T < 2) throw std::invalid_argument("mtp_forward: at least two tokens");
  if (T > max_tokens_) throw std::invalid_argument("mtp_forward: tokens exceed max_tokens");
  if (T > max_context_) throw std::invalid_argument("mtp_forward: tokens exceed the context bound");
  for (int64_t id : token_ids)
    if (id < 0 || id >= cfg_.vocab_size) throw std::invalid_argument("mtp_forward: token id out of range");
  if (session_pos_[0] != 0) throw std::logic_error("mtp_forward: slot 0 holds an open session");
  open_slot(0);
  if (!pool_.ensure_request_blocks(0, T, stream_))
    throw std::runtime_error("mtp_forward: the cache pool cannot cover the batch");
  RowRun run;
  run.req = 0;
  run.ids = token_ids.data();
  run.T = T;
  run.pos0 = 0;
  run.decode = false;
  run.all_rows = true;
  (void)run_rows(run);
  const int rows = T - 1;
  DGPP_CUDA_OK(cudaMemcpyAsync(d_tokens_, token_ids.data() + 1, static_cast<size_t>(rows) * 8,
                               cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  mtp_run_rows(0, d_tokens_, 0, rows, /*decode_row=*/false, /*capture=*/false, /*head_rows=*/rows, 0);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  Outputs out;
  out.lm_vocab_begin = lm_vocab_begin_;
  out.lm_vocab_count = lm_vocab_count_;
  const int H = cfg_.hidden_size;
  out.final_hidden_bits.resize(static_cast<size_t>(rows) * H);
  out.logits.resize(static_cast<size_t>(rows) * lm_vocab_count_);
  DGPP_CUDA_OK(
      cudaMemcpy(out.final_hidden_bits.data(), mtp_h_, out.final_hidden_bits.size() * 2, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(out.logits.data(), logits_, out.logits.size() * 4, cudaMemcpyDeviceToHost));
  session_close(0);
  return out;
}

Qwen35Model::Outputs Qwen35Model::run_rows(const RowRun& run) {
  const int T = run.T, req = run.req;
  if (run.capture && loader_.residency() != LoaderResidency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  // The DFlash2 target verify captures (the batched verify graph); only
  // the block draft itself stays eager (its host roundtrips) and is never
  // called under capture.
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  configure_gemm_rows(T, run.decode);
  walk_rows_ = T;
  const RowInputs in = begin_run(run);
  const bool batched = in.batched;
  const int num_requests = in.num_requests;
  const int64_t* tokens = in.tokens;
  const int64_t* d_pos = in.pos;
  const int32_t* d_req = in.req_ids;
  const int32_t* d_spans = in.spans;
  embed_gather_bf16(globals_.embed, tokens, resid_, T, H, stream_);
  Outputs out;
  // The boundary folds (world > 1): the attention / GDN output and the MLP
  // output are this rank's partial sums over its heads and its MLP slice;
  // each folds across the ranks in bf16 before the residual add — two
  // folds a layer. The producer writes into the reducer's staged buffer
  // when the shape fits (the collective sends straight from there; under
  // capture the recorder's one stable buffer, consumed before the next
  // handout), else into the model's own buffer. The eager producer
  // quiesces before the collective; under capture the fold is a recorded
  // node and the stream order is the drain.
  const auto stage = [&](uint16_t* fallback, int width) -> uint16_t* {
    if (!boundary_) return fallback;
    uint16_t* s = boundary_->stage(T, width);
    if (s == nullptr && run.capture)
      throw std::runtime_error("run_rows: a capture fold does not fit the recorder's staged buffer");
    return s ? s : fallback;
  };
  const auto fold = [&](uint16_t* buf, int width) {
    if (!boundary_) return;
    if (!run.capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    boundary_->reduce(buf, T, width);
  };
  // The speculative verify's per-row GDN snapshots (the rollback source);
  // the recurrent state replays (gdn_replay_) and a prefill walk reads it
  // as it stands: the pending rows are materialized first (a no-op kernel
  // when none are pending).
  const bool snapshots = run.decode && run.snapshots;
  if (gdn_replay_ && !run.decode) {
    if (run.num_spans > 0) {
      for (int sp = 0; sp < run.num_spans; ++sp) materialize_gdn(run.span_reqs[sp]);
    } else {
      materialize_gdn(req);
    }
  }
  int full_ord = 0, gdn_ord = 0;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const Qwen35LayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
    qwen_rmsnorm_bf16(resid_, r.input_norm, x_, T, H, eps, stream_);
    uint16_t* ao = stage(attn_out_, H);
    if (r.kind == Qwen35LayerKind::Gdn) {
      KdaStateSnapshots rec_snap;
      KdaConvSnapshots conv_snap;
      KdaReplay replay;
      if (snapshots) {
        if (!gdn_replay_) {
          rec_snap.states = spec_rec_ + static_cast<size_t>(gdn_ord) * rec_elems_;
          rec_snap.stride_elems = static_cast<int64_t>(num_gdn_) * rec_elems_;
        }
        conv_snap.states = spec_conv_ + static_cast<size_t>(gdn_ord) * conv_elems_;
        conv_snap.stride_elems = static_cast<int64_t>(num_gdn_) * conv_elems_;
      }
      // Every decode walk of the replay form takes the checkpoint form (a
      // one-row walk included: the live buffer is the checkpoint).
      if (run.decode && gdn_replay_) {
        replay = gdn_replay_view(req, gdn_ord, batched);
        gdn_replay_rows_used_ = batched ? T / std::max(num_requests, 1) : T;
      }
      if (batched) {
        KdaRequestRows requests;
        requests.request_ids = d_req;
        requests.positions = d_pos;
        requests.spans = d_spans;
        requests.num_requests = num_requests;
        gdn_->enqueue_rows(x_, gdn_rec(0, gdn_ord),
                           static_cast<int64_t>(num_gdn_) * rec_elems_, gdn_conv(0, gdn_ord),
                           static_cast<int64_t>(num_gdn_) * conv_elems_, ao, T, requests, stream_,
                           rec_snap, conv_snap, replay);
      } else if (run.num_spans > 0) {
        int64_t row0 = 0;
        for (int sp = 0; sp < run.num_spans; ++sp) {
          const int len = run.span_lens[sp], sreq = run.span_reqs[sp];
          gdn_->enqueue(x_ + static_cast<size_t>(row0) * H, gdn_rec(sreq, gdn_ord),
                        gdn_conv(sreq, gdn_ord), ao + static_cast<size_t>(row0) * H, len, stream_,
                        KdaStateSnapshots{}, KdaConvSnapshots{}, KdaReplay{},
                        run.span_pos0[sp] > 0 && !run.decode);
          row0 += len;
        }
      } else {
        const bool dbg = session_capture_layers_ && run.decode && !run.capture && gdn_ord == 0;
        if (dbg) {
          // The first GDN layer's input rows, before the layer reuses x_.
          DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
          std::vector<uint16_t> snap(static_cast<size_t>(T) * H);
          DGPP_CUDA_OK(cudaMemcpy(snap.data(), x_, snap.size() * 2, cudaMemcpyDeviceToHost));
          out.debug_stages.push_back(std::move(snap));
          out.debug_stage_names.emplace_back("layer input x");
          std::vector<uint16_t> res(static_cast<size_t>(T) * H);
          DGPP_CUDA_OK(cudaMemcpy(res.data(), resid_, res.size() * 2, cudaMemcpyDeviceToHost));
          out.debug_stages.push_back(std::move(res));
          out.debug_stage_names.emplace_back("residual in");
        }
        gdn_->enqueue(x_, gdn_rec(req, gdn_ord), gdn_conv(req, gdn_ord), ao, T, stream_, rec_snap,
                      conv_snap, replay, run.pos0 > 0 && !run.decode);
        if (dbg) {
          // The first GDN layer's intermediates (the rows-invariance test).
          DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
          for (const QwenGdnLayer::DebugStage& st : gdn_->debug_stages()) {
            std::vector<uint16_t> snap(static_cast<size_t>(T) * st.width);
            DGPP_CUDA_OK(cudaMemcpy(snap.data(), st.data, snap.size() * 2, cudaMemcpyDeviceToHost));
            out.debug_stages.push_back(std::move(snap));
            out.debug_stage_names.emplace_back(st.name);
          }
          std::vector<uint16_t> snap(static_cast<size_t>(T) * H);
          DGPP_CUDA_OK(cudaMemcpy(snap.data(), ao, snap.size() * 2, cudaMemcpyDeviceToHost));
          out.debug_stages.push_back(std::move(snap));
          out.debug_stage_names.emplace_back("out_proj");
        }
      }
      ++gdn_ord;
    } else {
      QwenFullAttnCache cache = pool_.view(full_ord);
      QwenQsaRows qrows;
      qrows.req_ids = d_req;
      qrows.pos = d_pos;
      qrows.decode = run.decode;
      qrows.request = req;
      qrows.pos0 = run.pos0;
      qrows.spans = d_spans;
      qrows.num_requests = num_requests;
      if (run.num_spans > 0) {
        int64_t row0 = 0;
        for (int sp = 0; sp < run.num_spans; ++sp) {
          const int len = run.span_lens[sp];
          QwenQsaRows srows = qrows;
          srows.req_ids = d_req + row0;
          srows.pos = d_pos + row0;
          srows.request = run.span_reqs[sp];
          srows.pos0 = run.span_pos0[sp];
          full_->enqueue(x_ + static_cast<size_t>(row0) * H, len, srows, cache, ao + static_cast<size_t>(row0) * H,
                         stream_);
          row0 += len;
        }
      } else {
        full_->enqueue(x_, T, qrows, cache, ao, stream_);
      }
      ++full_ord;
    }
    if (run.decode) prefetch_ffn_side(r);  // the MLP's weights into L2 while the fold waits
    fold(ao, H);
    // Fused residual-add + post norm (bitwise the pair): one launch.
    qwen_add_rmsnorm_bf16(resid_, ao, r.post_norm, x_, T, H, eps, stream_);
    // A resume chunk's short tail takes the per-tensor MLP like the
    // attention resume above; group spans start at pos0 (resume false).
    // Decode/verify walks (run.decode) keep their exact GEMV dispatch.
    bool mlp_resume = false;
    if (!run.decode) {
      if (run.num_spans > 0) {
        for (int sp = 0; sp < run.num_spans; ++sp)
          mlp_resume = mlp_resume || run.span_pos0[sp] > 0;
      } else {
        mlp_resume = run.pos0 > 0;
      }
    }
    uint16_t* mo = stage(mlp_out_, H);
    dense_mlp(x_, mo, T, r.mlp, stream_, layer, mlp_resume);
    if (run.decode) prefetch_attention_side(layer + 1);  // the next layer's input side (or the head)
    fold(mo, H);
    add_inplace_bf16(resid_, mo, static_cast<size_t>(T) * H, stream_);
    if (run.capture_layers || (session_capture_layers_ && run.decode && !run.capture)) {
      // The fixture gates' per-layer residual read (never under a graph
      // capture: the diagnostic forward, or a test's eager decode walk).
      DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      std::vector<uint16_t> snap(static_cast<size_t>(T) * H);
      DGPP_CUDA_OK(cudaMemcpy(snap.data(), resid_, snap.size() * 2, cudaMemcpyDeviceToHost));
      out.layer_states.push_back(std::move(snap));
    }
    // The drafter's tap: the layer's OUTPUT residual stream through that
    // tap's fc slice, fp32-accumulated (the reference's one wide cat-GEMM
    // with the weight split columnwise — the same bytes, five roundings
    // into fp32 instead of one, tolerance-identical).
    if (dflash2_ && df_tap_[layer] >= 0) {
      const uint16_t* fc_t = dfw_.fc + static_cast<size_t>(df_tap_[layer]) * H * H;
      gemm_.matmul(resid_, fc_t, df_t32_, T, H, H, DType::BF16, GemmOut::F32,
                   static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
      dflash2_acc_f32(df_acc_, df_t32_, static_cast<int64_t>(T) * H,
                      df_tap_[layer] == 0 ? 0 : 1, stream_);
    }
  }
  // Final norm + lm head.
  qwen_rmsnorm_bf16(resid_, globals_.final_norm, h_, T, H, eps, stream_);
  const int first = (run.decode || run.all_rows || run.num_spans > 0) ? 0 : T - 1;
  const int rows = T - first;
  head_gemv(h_ + static_cast<size_t>(first) * H, logits_ + static_cast<size_t>(first) * lm_vocab_count_,
            rows, stream_);
  // The draft block's input: the last rows' final hidden into the slots'
  // windows by position (the last window rows of a prefill chunk, every
  // decode row — distinct slots within one launch).
  if (mtp_) {
    const int n = run.num_spans > 0 ? T : std::min(T, max_decode_rows_);
    store_draft_hidden(h_ + static_cast<size_t>(T - n) * H, d_req + (T - n), d_pos + (T - n), n);
  }
  // Every run's rows become the drafter's context K/V at their positions
  // (rejected verify rows included: they land past the committed end and
  // the next draft's position mask hides them — no rollback). Inside a
  // captured verify this is a recorded node, NOT a skip: the eager
  // redrafts between replays read the planes, so a replay that omitted
  // the feed would draft over stale context (2026-10-02: the c4 graph's
  // acceptance decay was exactly this — the capture kept the verify
  // exact but starved the drafter).
  if (dflash2_) dflash2_store_features(T, d_req, d_pos);
  if (run.decode) prefetch_.join(stream_);  // every forked prefetch back on the main stream
  out = finish_run(run, std::move(out));
  return out;
}

// ---- the boundary prefetch windows ------------------------------------------------

void Qwen35Model::prefetch_add(const void* p, size_t bytes) {
  if (p != nullptr && bytes != 0) prefetch_.add(p, bytes);
}

// A bf16 decode weight through its resident view (the packed companion
// when engine.bf16_weights packs it — its own allocation, one launch).
void Qwen35Model::prefetch_bf16(const uint16_t* w, size_t bytes) {
  if (w == nullptr || bytes == 0) return;
  const void* view = nullptr;
  size_t view_bytes = 0;
  gemm_.resident_view(w, bytes, walk_rows_, &view, &view_bytes);
  prefetch_.add_view(w, view, view_bytes);
}

// The FP8 form's payload and scale grid (adjacent grants of one image).
void Qwen35Model::prefetch_fp8(const GlmQuantMatrix& q) {
  if (q.payload == nullptr) return;
  prefetch_add(q.payload, static_cast<size_t>(q.rows) * static_cast<size_t>(q.cols));
  prefetch_add(q.scales, static_cast<size_t>(q.scale_rows()) * static_cast<size_t>(q.scale_cols()) * 4);
}

// Before the attention fold: this layer's MLP in consumption order (gate,
// up, down); the window's budget takes the leading bytes.
void Qwen35Model::prefetch_ffn_side(const Qwen35LayerResident& r) {
  if (!prefetch_.enabled() || world_ < 2) return;
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  const Qwen35DenseMlpResident& m = r.mlp;
  if (m.gate_fp8.payload) {
    prefetch_fp8(m.gate_fp8);
    prefetch_fp8(m.up_fp8);
    prefetch_fp8(m.down_fp8);
  } else {
    const size_t I = static_cast<size_t>(loader_.geometry().local_inter);
    prefetch_bf16(m.gate, I * H * 2);
    prefetch_bf16(m.up, I * H * 2);
    prefetch_bf16(m.down, H * I * 2);
  }
}

// Before the MLP fold: the next layer's input projections (the GDN's
// qkv | z | a | b and conv, or the attention's q | k | v), then its output
// projection; past the last layer, the head.
void Qwen35Model::prefetch_attention_side(int layer) {
  if (!prefetch_.enabled() || world_ < 2) return;
  if (layer >= cfg_.num_hidden_layers) {
    prefetch_head();
    return;
  }
  // Resident stacks only (a decode walk's): load_layer is a lookup there.
  const Qwen35LayerResident& r = loader_.load_layer(layer);
  const Qwen35LocalGeometry& geo = loader_.geometry();
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  prefetch_add(r.input_norm, H * 2);
  if (r.kind == Qwen35LayerKind::Gdn) {
    const QwenGdnResident& g = r.gdn;
    const size_t lk = static_cast<size_t>(geo.local_key_heads), lv = static_cast<size_t>(geo.local_value_heads);
    const size_t dk = static_cast<size_t>(cfg_.gdn_key_head_dim), dv = static_cast<size_t>(cfg_.gdn_value_head_dim);
    const size_t qkv_rows = 2 * lk * dk + lv * dv;
    if (g.in_proj_qkv_fp8.payload) prefetch_fp8(g.in_proj_qkv_fp8);
    else prefetch_bf16(g.in_proj_qkv, qkv_rows * H * 2);
    if (g.in_proj_z_fp8.payload) prefetch_fp8(g.in_proj_z_fp8);
    else prefetch_bf16(g.in_proj_z, lv * dv * H * 2);
    prefetch_bf16(g.in_proj_a, lv * H * 2);
    prefetch_bf16(g.in_proj_b, lv * H * 2);
    prefetch_add(g.conv, qkv_rows * static_cast<size_t>(cfg_.gdn_conv_width) * 2);
    if (g.out_proj_fp8.payload) prefetch_fp8(g.out_proj_fp8);
    else prefetch_bf16(g.out_proj, H * lv * dv * 2);
  } else {
    const QwenFullAttnResident& f = r.full;
    const size_t lh = static_cast<size_t>(geo.local_heads), lkv = static_cast<size_t>(geo.local_kv_heads);
    const size_t d = static_cast<size_t>(cfg_.head_dim);
    if (f.q_proj_fp8.payload) prefetch_fp8(f.q_proj_fp8);
    else prefetch_bf16(f.q_proj, lh * 2 * d * H * 2);
    if (f.k_proj_fp8.payload) prefetch_fp8(f.k_proj_fp8);
    else prefetch_bf16(f.k_proj, lkv * d * H * 2);
    if (f.v_proj_fp8.payload) prefetch_fp8(f.v_proj_fp8);
    else prefetch_bf16(f.v_proj, lkv * d * H * 2);
    if (f.o_proj_fp8.payload) prefetch_fp8(f.o_proj_fp8);
    else prefetch_bf16(f.o_proj, H * lh * d * 2);
  }
}

void Qwen35Model::prefetch_head() {
  const size_t H = static_cast<size_t>(cfg_.hidden_size);
  const size_t V = static_cast<size_t>(lm_vocab_count_);
  prefetch_.open_window(stream_, prefetch_window_bytes_, prefetch_.boundary_rate());
  prefetch_add(globals_.final_norm, H * 2);
  if (head_fp8_enabled_ && head_fp8_ != nullptr) {
    prefetch_add(head_fp8_, V * H);
    prefetch_add(head_scales_, ((V + 127) / 128) * (H / 128) * 4);
  } else {
    prefetch_bf16(globals_.lm_head, V * H * 2);
  }
}

// ---- the DFlash2 drafter -------------------------------------------------------

void Qwen35Model::dflash2_store_features(int T, const int32_t* d_req, const int64_t* d_pos) {
  const int H = cfg_.hidden_size;
  const int L = dfcfg_.num_hidden_layers, KV = df_kvr_;
  const int KVH = df_kvh_, HD = dfcfg_.head_dim;
  const int32_t* tables = pool_.blocks().device_tables();
  const int bpr = static_cast<int>(pool_.total_blocks());
  const int bt = kv_block_tokens_static();
  for (int off = 0; off < T; off += df_rows_cap()) {
    const int rows = std::min(df_rows_cap(), T - off);
    dflash2_norm_f32_bf16(df_acc_ + static_cast<size_t>(off) * H, dfw_.hidden_norm, df_norm_, rows, H,
                          dfcfg_.rms_norm_eps, stream_);
    for (int l = 0; l < L; ++l) {
      // k|v rows of the fused layer weight (a view right after q, or their
      // own bf16 slot under the fp8 block matrices).
      const uint16_t* kv_w = dfw_.layers[l].kv_rows;
      gemm_.matmul(df_norm_, kv_w, df_kv_, rows, 2 * KV, H, DType::BF16, GemmOut::BF16,
                   static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
      dflash2_norm_rope_bf16(df_kv_, 2 * KV, dfw_.layers[l].k_norm, d_pos + off, df_inv_freq_, df_kv_,
                             2 * KV, rows, KVH, HD, dfcfg_.rms_norm_eps, stream_);
      const QwenFullAttnCache plane = pool_.view(num_full_ + l);
      qsa_kv_append(df_kv_, 2 * KV, df_kv_ + KV, 2 * KV, d_req + off, d_pos + off, rows, tables, bpr,
                    bt, KVH, HD, plane.k_cache, plane.v_cache, nullptr, nullptr, stream_);
    }
  }
}

bool Qwen35Model::dflash2_draft(int req, int64_t bonus, std::vector<int32_t>* drafts) {
  if (!dflash2_) throw std::logic_error("dflash2_draft: no drafter loaded");
  const int H = cfg_.hidden_size;
  const int QR = dfcfg_.query_rows(), D = dfcfg_.drafts();
  const int64_t pos = session_position(req);
  if (pos < 0 || pos + QR > max_context()) return false;
  // The block writes K/V at [pos, pos + QR): grow the table now (the next
  // verify's rows live in the same span). Pool exhausted -> plain step.
  if (!pool_.ensure_request_blocks(req, pos + QR, stream_)) return false;

  // Row inputs: [bonus, mask x D] at positions pos..pos+QR-1.
  for (int j = 0; j < QR; ++j) {
    df_io64_h_[j] = pos + j;
    df_io64_h_[QR + j] = j == 0 ? bonus : dfcfg_.mask_token_id;
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(df_pos_, df_io64_h_, QR * 8, cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tokens_, df_io64_h_ + QR, QR * 8, cudaMemcpyHostToDevice, stream_));
  embed_gather_bf16(globals_.embed, df_tokens_, df_resid_, QR, H, stream_);
  configure_gemm_rows(QR, true);
  const int reqs[1] = {req};
  df_block_layers(1, reqs, /*capture=*/false);
  df_block_select(1, reqs, /*capture=*/false);
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tok_h_, df_tok_, D * 4, cudaMemcpyDeviceToHost, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  drafts->assign(df_tok_h_, df_tok_h_ + D);
  return true;
}

// The five bidirectional layers over the staged blocks (the scalar draft,
// the stacked redraft and the recorded draft share this body).
void Qwen35Model::df_block_layers(int slots, const int* reqs, bool capture) {
  const int H = cfg_.hidden_size;
  const int QR = dfcfg_.query_rows();
  const int L = dfcfg_.num_hidden_layers, QW = df_qw_, KV = df_kvr_;
  const int KVH = df_kvh_, HD = dfcfg_.head_dim, NH = df_nh_;
  const int G = dfcfg_.conv_groups(), TAPS = dfcfg_.conv_taps;
  const int I = df_i_;
  const float eps = dfcfg_.rms_norm_eps;
  const int R = slots * QR;
  // The sharded drafter's boundary folds (run_rows' pattern): the o and
  // down projections' partial [R, H] outputs summed across the ranks
  // before the output-side convs read them.
  const auto stage = [&](uint16_t* fallback, int width) -> uint16_t* {
    if (!boundary_) return fallback;
    uint16_t* s = boundary_->stage(R, width);
    if (s == nullptr && capture)
      throw std::runtime_error("dflash2 draft: a capture fold does not fit the recorder's staged buffer");
    return s ? s : fallback;
  };
  const auto fold = [&](uint16_t* buf, int width) {
    if (!boundary_) return;
    if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    boundary_->reduce(buf, R, width);
  };
  const int32_t* tables = pool_.blocks().device_tables();
  const int bpr = static_cast<int>(pool_.total_blocks());
  const int bt = kv_block_tokens_static();
  const size_t qkv_stride = static_cast<size_t>(QW) + 2 * KV;
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
  for (int l = 0; l < L; ++l) {
    const DFlash2LayerWeights& w = dfw_.layers[l];
    dflash2_rmsnorm_bf16(df_resid_, w.input_norm, df_x_, R, H, eps, stream_);
    // The attention conv: deltas from the normed input, one GEMM; side 0
    // conditions the qkv input, side 1 the attention output.
    gemm_.matmul(df_x_, w.attn_conv_kp, df_delta_, R, 2 * TAPS * G, H, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
    dflash2_grouped_conv_bf16(df_x_, df_delta_, w.attn_conv_base, df_xc_, R, QR, H, TAPS,
                              dfcfg_.conv_group_size, 2 * TAPS * G, stream_);
    if (w.qkv_fp8 != nullptr)
      launch_mma_gemv_fp8_bf16(df_xc_, static_cast<size_t>(H), w.qkv_fp8, w.qkv_scales, df_qkv_, R,
                               static_cast<int>(qkv_stride), H, qkv_stride, 7, 7, stream_, gemm_ws_, gemm_ws_bytes_);
    else
      gemm_.matmul(df_xc_, w.qkv, df_qkv_, R, static_cast<int>(qkv_stride), H, DType::BF16,
                   GemmOut::BF16, static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
    dflash2_norm_rope_bf16(df_qkv_, qkv_stride, w.q_norm, df_pos_, df_inv_freq_, df_q_, QW, R, NH, HD,
                           eps, stream_);
    dflash2_norm_rope_bf16(df_qkv_ + QW, qkv_stride, w.k_norm, df_pos_, df_inv_freq_, df_qkv_ + QW,
                           qkv_stride, R, KVH, HD, eps, stream_);
    const QwenFullAttnCache plane = pool_.view(num_full_ + l);
    for (int k = 0; k < slots; ++k) {
      const size_t ro = static_cast<size_t>(k) * QR;
      const int32_t* table = tables + static_cast<size_t>(reqs[k]) * pool_.total_blocks();
      const uint16_t* krow = df_qkv_ + ro * qkv_stride + QW;
      qsa_kv_append(krow, qkv_stride, krow + KV, qkv_stride, df_zero_, df_pos_ + ro, QR, table, 1, bt, KVH,
                    HD, plane.k_cache, plane.v_cache, nullptr, nullptr, stream_);
      // The split-key form: the row's 2K window in 32 parallel ranges
      // (the serial walk cost ~2 ms a layer at a 2K context).
      dflash2_block_attn_split(df_q_ + ro * QW, QW, plane.k_cache, plane.v_cache, table, bt, bpr, QR,
                               dfcfg_.sliding_window, df_pos_ + ro, QR, NH, KVH, HD, scale,
                               df_attn_part_ + ro * static_cast<size_t>(NH) * dflash2_block_attn_splits() * 130,
                               df_attn_ + ro * QW, stream_);
    }
    uint16_t* ao = stage(df_o_, H);
    if (w.o_fp8 != nullptr)
      launch_mma_gemv_fp8_bf16(df_attn_, static_cast<size_t>(QW), w.o_fp8, w.o_scales, ao, R, H, QW,
                               static_cast<size_t>(H), 7, 7, stream_, gemm_ws_, gemm_ws_bytes_);
    else
      gemm_.matmul(df_attn_, w.o, ao, R, H, QW, DType::BF16, GemmOut::BF16, QW, gemm_ws_,
                   gemm_ws_bytes_, stream_);
    fold(ao, H);
    dflash2_grouped_conv_bf16(ao, df_delta_ + TAPS * G, w.attn_conv_base + static_cast<size_t>(TAPS) * H,
                              df_xc_, R, QR, H, TAPS, dfcfg_.conv_group_size, 2 * TAPS * G, stream_);
    dflash2_add_rmsnorm_bf16(df_resid_, df_xc_, w.post_norm, df_x_, R, H, eps, stream_);
    // The MLP conv: same shape around the SwiGLU.
    gemm_.matmul(df_x_, w.mlp_conv_kp, df_delta_, R, 2 * TAPS * G, H, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
    dflash2_grouped_conv_bf16(df_x_, df_delta_, w.mlp_conv_base, df_xc_, R, QR, H, TAPS,
                              dfcfg_.conv_group_size, 2 * TAPS * G, stream_);
    if (w.gate_fp8 != nullptr) {
      launch_mma_gemv_fp8_bf16(df_xc_, static_cast<size_t>(H), w.gate_fp8, w.gate_scales, df_gate_, R, I, H,
                               static_cast<size_t>(I), 7, 7, stream_, gemm_ws_, gemm_ws_bytes_);
      launch_mma_gemv_fp8_bf16(df_xc_, static_cast<size_t>(H), w.up_fp8, w.up_scales, df_up_, R, I, H,
                               static_cast<size_t>(I), 7, 7, stream_, gemm_ws_, gemm_ws_bytes_);
    } else {
      gemm_.matmul(df_xc_, w.gate, df_gate_, R, I, H, DType::BF16, GemmOut::BF16,
                   static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
      gemm_.matmul(df_xc_, w.up, df_up_, R, I, H, DType::BF16, GemmOut::BF16, static_cast<size_t>(H),
                   gemm_ws_, gemm_ws_bytes_, stream_);
    }
    swiglu_limit_bf16(df_gate_, df_up_, df_mlp_, static_cast<int64_t>(R) * I, INFINITY, stream_);
    uint16_t* mo = stage(df_o_, H);
    if (w.down_fp8 != nullptr)
      launch_mma_gemv_fp8_bf16(df_mlp_, static_cast<size_t>(I), w.down_fp8, w.down_scales, mo, R, H, I,
                               static_cast<size_t>(H), 7, 7, stream_, gemm_ws_, gemm_ws_bytes_);
    else
      gemm_.matmul(df_mlp_, w.down, mo, R, H, I, DType::BF16, GemmOut::BF16,
                   static_cast<size_t>(I), gemm_ws_, gemm_ws_bytes_, stream_);
    fold(mo, H);
    dflash2_grouped_conv_bf16(mo, df_delta_ + TAPS * G, w.mlp_conv_base + static_cast<size_t>(TAPS) * H,
                              df_xc_, R, QR, H, TAPS, dfcfg_.conv_group_size, 2 * TAPS * G, stream_);
    add_inplace_bf16(df_resid_, df_xc_, static_cast<int64_t>(R) * H, stream_);
  }
  dflash2_rmsnorm_bf16(df_resid_, dfw_.norm, df_h_, R, H, eps, stream_);
}

// The mask rows through the shared head and their top-K, per slot at row
// offsets; at world > 1 each rank's slice lists meet in one boundary fold
// and every rank merges the same global top-K (the walk then proposes the
// same tokens on every rank). The proposal: the reference chained selector
// walk (vLLM's _selector_walk_kernel at temperature 0) over each mask
// row's top-K — scores[l][p][c] = unary[l][c] + <pred[id(l-1,p)] *
// hidden[l], succ[id(l,c)]>.
void Qwen35Model::df_block_select(int slots, const int* reqs, bool capture) {
  const int H = cfg_.hidden_size;
  const int QR = dfcfg_.query_rows(), D = dfcfg_.drafts();
  // The recorded walk draws for a stochastic request (the armed
  // proposals); the eager walk is the argmax. Every walk with a slot list
  // writes the slots' confidence (the scheduled verify depth's input).
  const bool sampled = capture && df_specs_ != nullptr && reqs != nullptr;
  const int64_t V = lm_vocab_count_;
  const int topk = dfcfg_.selector_top_k, rank = dfcfg_.selector_rank;
  // One head stream for every slot's rows (the mask rows of slot k are rows
  // k*QR + 1 .. k*QR + D; the anchor rows' logits and top-K go unread).
  const int R = slots * QR;
  head_gemv(df_h_, df_logits_, R, stream_);
  dflash2_topk_f32(df_logits_, df_ids_, df_sc_, V, R, topk, stream_, df_topk_ws_, df_topk_ws_bytes_);
  if (world_ > 1) {
    const int rows = R;
    const size_t elems = dflash2_topk_table_elems(rows, topk, world_);
    uint16_t* table = boundary_->stage(1, static_cast<int>(elems));
    if (table == nullptr) {
      if (capture)
        throw std::runtime_error("dflash2 draft: the top-K gather does not fit the recorder's staged buffer");
      table = df_table_;
    }
    dflash2_topk_stage(df_ids_, df_sc_, rows, topk, lm_vocab_begin_, rank_, world_, table, stream_);
    if (!capture) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));  // the host-driven fold's contract
    boundary_->reduce(table, 1, static_cast<int>(elems));
    dflash2_topk_merge(table, rows, topk, world_, df_ids_, df_sc_, stream_);
  }
  gemm_.matmul(df_h_, dfw_.hidden_projection, df_hidden32_, R, rank, H, DType::BF16, GemmOut::F32,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  for (int k = 0; k < slots; ++k) {
    const size_t ro = static_cast<size_t>(k) * QR, co = static_cast<size_t>(k) * D, mo = ro + 1;
    const size_t po = sampled ? static_cast<size_t>(reqs[k]) * static_cast<size_t>(df_props_stride_) : 0;
    dflash2_selector_walk(df_ids_ + mo * topk, df_sc_ + mo * topk, df_hidden32_ + mo * rank,
                          dfw_.pred_codebook, dfw_.succ_codebook, df_tokens_ + ro, df_tok_ + co, D, topk,
                          rank, stream_, sampled ? df_pos_ + ro : nullptr,
                          sampled ? df_specs_ + reqs[k] : nullptr, sampled ? df_props_ + po : nullptr,
                          sampled && df_props_h_ != nullptr ? df_props_h_ + po : nullptr,
                          reqs != nullptr ? df_conf_ + static_cast<size_t>(reqs[k]) * D : nullptr);
  }
}

void Qwen35Model::session_graph_capture_block_draft(int req, const PickVerdict* verdict) {
  if (!dflash2_) throw std::logic_error("session_graph_capture_block_draft: no drafter loaded");
  if (req < 0 || req >= max_requests_) throw std::invalid_argument("session_graph_capture_block_draft: request");
  if (verdict == nullptr) throw std::invalid_argument("session_graph_capture_block_draft: null verdict");
  if (!graph_device_positions_ || !graph_device_tokens_)
    throw std::logic_error("session_graph_capture_block_draft: the step must be captured with device positions and tokens");
  const int H = cfg_.hidden_size;
  const int QR = dfcfg_.query_rows(), D = dfcfg_.drafts();
  if ((graph_feed_rows_ > 0 ? graph_feed_rows_ : decode_rows_) != QR)
    throw std::logic_error("session_graph_capture_block_draft: the token feed is [next, drafts] (T = query rows)");
  // The block's rows off the verdict: the anchor is the verify's next
  // token at the position the recorded commit advanced to; mask rows
  // follow (positions past the context -1: no state write).
  dflash2_stage_block(verdict, d_session_pos_ + req, dfcfg_.mask_token_id, QR, max_context(), df_pos_,
                      df_tokens_, stream_);
  embed_gather_bf16(globals_.embed, df_tokens_, df_resid_, QR, H, stream_);
  configure_gemm_rows(QR, true);
  const int reqs[1] = {req};
  df_block_layers(1, reqs, /*capture=*/true);
  df_block_select(1, reqs, /*capture=*/true);
  // The next replay's feed and the host's copy of the drafts.
  dflash2_block_feed(verdict, df_tok_, D, step_tokens_, stream_);
  dflash2_publish_drafts(df_tok_, df_mirror_h_ + static_cast<size_t>(req) * D, D, stream_);
  const int K = dfcfg_.selector_top_k;
  dflash2_publish_words(df_ids_ + K, df_cands_h_ + static_cast<size_t>(req) * D * K, D * K, stream_);  // the mask rows' lists
}

void Qwen35Model::session_graph_capture_block_draft_batch(const PickVerdict* verdicts, int requests) {
  if (!dflash2_) throw std::logic_error("session_graph_capture_block_draft_batch: no drafter loaded");
  if (verdicts == nullptr) throw std::invalid_argument("session_graph_capture_block_draft_batch: null verdicts");
  if (requests < 1 || requests > max_requests_ || requests > df_batch_)
    throw std::invalid_argument("session_graph_capture_block_draft_batch: requests exceed the slots or the stacked scratch");
  if (graph_batch_requests_ != requests || !graph_device_positions_ || !graph_device_tokens_)
    throw std::logic_error("session_graph_capture_block_draft_batch: requires the device-driven fixed batch of these slots");
  if (graph_batch_map_source_ != nullptr)
    throw std::logic_error("session_graph_capture_block_draft_batch: the block draft takes slot q as request q (no compaction)");
  const int H = cfg_.hidden_size;
  const int QR = dfcfg_.query_rows(), D = dfcfg_.drafts();
  // The persistent feed is the whole block per slot ([next, drafts] at
  // query rows); a reduced-row verify (engine.dflash_batch_rows) reads it
  // compacted and the block draft still writes the whole of it.
  const int feed_rows = graph_feed_rows_ > 0 ? graph_feed_rows_ : graph_rows_per_request_;
  if (feed_rows != QR || graph_rows_per_request_ > QR)
    throw std::logic_error("session_graph_capture_block_draft_batch: the feed is [next, drafts] per slot (T = query rows)");
  const int R = requests * QR;
  dflash2_stage_block_batched(verdicts, d_session_pos_, dfcfg_.mask_token_id, QR, requests, max_context(), df_pos_,
                              df_tokens_, stream_);
  embed_gather_bf16(globals_.embed, df_tokens_, df_resid_, R, H, stream_);
  configure_gemm_rows(R, true);
  std::vector<int> reqs(static_cast<size_t>(requests));
  for (int q = 0; q < requests; ++q) reqs[static_cast<size_t>(q)] = q;
  df_block_layers(requests, reqs.data(), /*capture=*/true);
  df_block_select(requests, reqs.data(), /*capture=*/true);
  // The fixed batch's feeds sit behind the decode rows at QR rows per slot
  // (session_graph_capture_batch's step_tokens_).
  int64_t* feeds = d_tokens_ + static_cast<size_t>(max_decode_rows_);
  dflash2_block_feed_batched(verdicts, df_tok_, D, requests, QR, feeds, stream_);
  dflash2_publish_drafts_batched(df_tok_, df_mirror_h_, D, requests, stream_);
  const int K = dfcfg_.selector_top_k;
  for (int q = 0; q < requests; ++q)  // slot q == request q: its mask rows' lists
    dflash2_publish_words(df_ids_ + (static_cast<size_t>(q) * QR + 1) * K, df_cands_h_ + static_cast<size_t>(q) * D * K,
                          D * K, stream_);
}

// The batched redraft: the S slots' [bonus, mask x D] blocks stacked into
// one R = S*QR forward. Every row-wise op (embed, norms, GEMMs, the 2-tap
// convs — whose kernel is block-boundary aware via r % block_rows — RoPE
// with the stacked positions) runs over all R rows at once, so the draft
// weights are read once, not S times; the KV appends, the sliding-window
// attention, the head, top-K and selector walk stay per-slot at row
// offsets (the same calls as the scalar path, bitwise identical). A slot
// that fails admission (context bound / pool exhausted) gets empty drafts,
// the scalar failure rule.
void Qwen35Model::dflash2_draft_batch(const std::vector<int>& reqs,
                                      const std::vector<int64_t>& bonuses,
                                      std::vector<std::vector<int32_t>>* drafts) {
  if (!dflash2_) throw std::logic_error("dflash2_draft_batch: no drafter loaded");
  if (reqs.empty()) throw std::invalid_argument("dflash2_draft_batch: no requests");
  if (reqs.size() != bonuses.size() || drafts == nullptr)
    throw std::invalid_argument("dflash2_draft_batch: reqs/bonuses/drafts shape");
  const int S = static_cast<int>(reqs.size());
  if (S > df_batch_)
    throw std::invalid_argument("dflash2_draft_batch: batch exceeds the stacked scratch");
  const int H = cfg_.hidden_size;
  const int QR = dfcfg_.query_rows(), D = dfcfg_.drafts();
  drafts->assign(static_cast<size_t>(S), {});
  // Admission first (no shared state touched): positions and block growth
  // per slot, compacting out the failures.
  std::vector<int> idx, creqs;
  idx.reserve(static_cast<size_t>(S));
  std::vector<int64_t> poss(static_cast<size_t>(S), -1);
  for (int s = 0; s < S; ++s) {
    const int req = reqs[static_cast<size_t>(s)];
    const int64_t pos = session_position(req);
    if (pos < 0 || pos + QR > max_context()) continue;
    if (!pool_.ensure_request_blocks(req, pos + QR, stream_)) continue;
    poss[static_cast<size_t>(s)] = pos;
    idx.push_back(s);
    creqs.push_back(req);
  }
  const int NS = static_cast<int>(idx.size());
  if (NS == 0) return;
  const int R = NS * QR;
  // Row inputs: each stacked block [bonus, mask x D] at its slot's positions.
  for (int k = 0; k < NS; ++k) {
    const int s = idx[static_cast<size_t>(k)];
    const int64_t pos = poss[static_cast<size_t>(s)];
    for (int j = 0; j < QR; ++j) {
      df_io64_h_[static_cast<size_t>(k) * QR + j] = pos + j;
      df_io64_h_[static_cast<size_t>(df_batch_) * QR + static_cast<size_t>(k) * QR + j] =
          j == 0 ? bonuses[static_cast<size_t>(s)] : dfcfg_.mask_token_id;
    }
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(df_pos_, df_io64_h_, static_cast<size_t>(R) * 8, cudaMemcpyHostToDevice,
                               stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tokens_, df_io64_h_ + static_cast<size_t>(df_batch_) * QR,
                               static_cast<size_t>(R) * 8, cudaMemcpyHostToDevice, stream_));
  embed_gather_bf16(globals_.embed, df_tokens_, df_resid_, R, H, stream_);
  configure_gemm_rows(R, true);
  df_block_layers(NS, creqs.data(), /*capture=*/false);
  df_block_select(NS, creqs.data(), /*capture=*/false);
  DGPP_CUDA_OK(cudaMemcpyAsync(df_tok_h_, df_tok_, static_cast<size_t>(NS) * D * 4,
                               cudaMemcpyDeviceToHost, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  for (int k = 0; k < NS; ++k)
    (*drafts)[static_cast<size_t>(idx[static_cast<size_t>(k)])].assign(
        df_tok_h_ + static_cast<size_t>(k) * D, df_tok_h_ + static_cast<size_t>(k + 1) * D);
}

// The static padded layout the captured verify stages: every slot's fed rows
// at [s*8, s*8+T), the block's tail rows position -1 (every state-writing
// kernel skips them; their compute-only results are never read back).
// Validates, grows blocks, uploads and returns the RowRun (capture=false;
// the caller flips it). Throws on any validation failure.
Qwen35Model::RowRun Qwen35Model::df_stage_padded(
    const std::vector<int>& reqs, const std::vector<std::vector<int64_t>>& feds,
    int* slots_out, std::vector<int>* offs_out) {
  const int kBlock = query_block_rows();
  if (reqs.empty() || reqs.size() > 4)
    throw std::invalid_argument("df_stage_padded: slot count");
  if (reqs.size() != feds.size())
    throw std::invalid_argument("df_stage_padded: reqs/feds shape");
  const int S = static_cast<int>(reqs.size());
  // One static shape per slot count: 8 rows for a lone slot (no padding at
  // all — the capture-mechanism control), 16 for two, 32 otherwise.
  const int slots = S == 1 ? 1 : (S == 2 ? 2 : 4);
  const int kRows = slots * kBlock;
  if (kRows > max_decode_rows_)
    throw std::invalid_argument("df_stage_padded: rows exceed the decode batch ceiling");
  const int req0 = reqs[0];
  // Validate + grow blocks BEFORE staging (host work, pre-capture).
  for (int s = 0; s < S; ++s) {
    const int req = reqs[static_cast<size_t>(s)];
    check_req(req, "df_stage_padded");
    const auto& ids = feds[static_cast<size_t>(s)];
    const int T = static_cast<int>(ids.size());
    if (T < 1 || T > kSpecRows)
      throw std::invalid_argument("df_stage_padded: row count");
    const int64_t pos = session_pos_[static_cast<size_t>(req)];
    if (pos <= 0) throw std::invalid_argument("df_stage_padded: no open session");
    for (int64_t id : ids)
      if (id < 0 || id >= vocab_size_) throw std::invalid_argument("df_stage_padded: token id");
    if (pos + T > max_context_) throw std::invalid_argument("df_stage_padded: context bound");
    if (!pool_.ensure_request_blocks(req, pos + T, stream_))
      throw std::runtime_error("df_stage_padded: cache pool exhausted");
  }
  std::vector<int> offs(static_cast<size_t>(S));
  for (int s = 0; s < slots; ++s) {
    const bool live = s < S;
    const int req = live ? reqs[static_cast<size_t>(s)] : req0;
    const int T = live ? static_cast<int>(feds[static_cast<size_t>(s)].size()) : 0;
    const int64_t pos = live ? session_pos_[static_cast<size_t>(req)] : 0;
    if (live) offs[static_cast<size_t>(s)] = s * kBlock;
    for (int r = 0; r < kBlock; ++r) {
      const int row = s * kBlock + r;
      if (r < T) {
        h_req_ids_[row] = req;
        h_step_pos_[row] = pos + r;
        h_token_[row] = feds[static_cast<size_t>(s)][static_cast<size_t>(r)];
      } else {
        h_req_ids_[row] = req0;
        h_step_pos_[row] = -1;
        h_token_[row] = 0;
      }
    }
    h_req_spans_[2 * s] = s * kBlock;
    h_req_spans_[2 * s + 1] = kBlock;
  }
  decode_rows_ = kRows;
  glm_upload_i64(h_token_, d_tokens_, kRows, stream_);
  glm_upload_i32(h_req_ids_, d_req_ids_, kRows, stream_);
  glm_upload_i64(h_step_pos_, d_step_pos_, kRows, stream_);
  glm_upload_i32(h_req_spans_, d_req_spans_, 2 * slots, stream_);
  if (slots_out) *slots_out = slots;
  if (offs_out) *offs_out = std::move(offs);
  RowRun run;
  run.req = req0;
  run.T = kRows;
  run.pos0 = session_pos_[static_cast<size_t>(req0)];
  run.decode = true;
  run.all_rows = true;
  run.snapshots = true;
  // A lone slot takes the exact scalar op stream (no batch flag, no spans
  // read): bitwise the eager C1 step. Batches need the span form.
  run.batch_requests = slots == 1 ? 0 : slots;
  return run;
}

std::vector<Qwen35Model::Outputs> Qwen35Model::session_verify_batch_graph(
    const std::vector<int>& reqs, const std::vector<std::vector<int64_t>>& feds,
    std::vector<int>* offsets) {
  // Not our shape (or a broken capture latched): the exact eager batch.
  if (!dflash_graph_verify_available() || df_verify_broken_)
    return session_verify_batch(reqs, feds, offsets);
  // Phase 1: stage the padded rows. This can fail transiently (the pool or
  // context bound at this batch size); the unpadded eager batch is more
  // flexible and may still serve the step. A staging failure is NOT a
  // broken graph — fall back without latching, since the condition can clear.
  int slots = 0;
  std::vector<int> offs;
  RowRun run;
  try {
    run = df_stage_padded(reqs, feds, &slots, &offs);
  } catch (const std::exception& e) {
    DGPP_LOG_WARN("dflash verify padded staging failed, eager this step: {}",
                  e.what());
    return session_verify_batch(reqs, feds, offsets);
  }
  // Phase 2: the graph capture/replay. A failure here is a genuine graph break.
  try {
    // A grown pool moves the tables pointer the capture baked in: drop
    // the stale graphs and recapture below (rare; admission windows).
    const int32_t* tables = pool_.blocks().device_tables();
    if (df_verify_tables_ != nullptr && df_verify_tables_ != tables) df_verify_graph_.clear();
    run.capture = true;
    const uint64_t key = 0xdf1a5e0u + static_cast<uint64_t>(run.T);
    const char* label = run.T == 8 ? "dflash-verify-8" : run.T == 16 ? "dflash-verify-16"
                                                                    : "dflash-verify-32";
    df_verify_graph_.replay_or_capture(key, label, stream_,
                                       [&](cudaStream_t) { (void)run_rows(run); });
    df_verify_tables_ = tables;
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));  // tail mirrors are in-graph D2H nodes
    const int S = static_cast<int>(reqs.size());
    if (offsets) *offsets = offs;
    std::vector<Outputs> outs(static_cast<size_t>(S));
    for (int s = 0; s < S; ++s) {
      const int T = static_cast<int>(feds[static_cast<size_t>(s)].size());
      Outputs& o = outs[static_cast<size_t>(s)];
      const size_t base = static_cast<size_t>(s) * query_block_rows();
      o.logits.assign(h_tail_logits_ + base * lm_vocab_count_,
                      h_tail_logits_ + (base + T) * lm_vocab_count_);
      o.lm_vocab_begin = lm_vocab_begin_;
      o.lm_vocab_count = lm_vocab_count_;
      o.final_hidden_bits.assign(h_tail_hidden_ + base * hidden_,
                                 h_tail_hidden_ + (base + T) * hidden_);
      session_pos_[static_cast<size_t>(reqs[static_cast<size_t>(s)])] += T;
      push_position(reqs[static_cast<size_t>(s)]);
    }
    return outs;
  } catch (const std::exception& e) {
    // A broken capture must never take the server down: drop the graph,
    // drain any latched stream error, and serve this step eager (latched:
    // the engine retries eager every step from here).
    df_verify_broken_ = true;
    df_verify_graph_.clear();
    (void)cudaStreamSynchronize(stream_);
    (void)cudaGetLastError();
    DGPP_LOG_ERROR("dflash verify graph broken, eager fallback: {}", e.what());
    return session_verify_batch(reqs, feds, offsets);
  }
}

}  // namespace dgpp
