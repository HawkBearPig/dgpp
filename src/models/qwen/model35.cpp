// Qwen3.5-27B dense model (FP8 text-only, standard pre-norm residual).
//
// 48 GDN layers (swish gate) + 16 Full GQA layers, dense SwiGLU MLPs. Text
// only: no vision tower, no hyperconnections / PLE / MoE. GDN recurrent +
// conv state is model-owned per request slot; the Full layers' K/V lives in
// Qwen35KvPool (plus one draft plane when MTP is on). Eager + prefix-off
// serving shape; the graph path replays the same walks.

#include "models/qwen/model35.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/add_rmsnorm.hpp"
#include "kernels/fp8_blockwise_dense.hpp"
#include "kernels/fp8_dequant.hpp"
#include "kernels/gemm.hpp"
#include "kernels/kernels.hpp"
#include "kernels/qwen_mtp.hpp"
#include "kernels/qwen_norm.hpp"
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
}

size_t Qwen35KvPool::cache_bytes(const Qwen35KvPoolShape& shape) {
  const size_t kv_row = static_cast<size_t>(shape.kv_heads) * shape.dim;
  const size_t layer_kv = static_cast<size_t>(shape.token_slots) * kv_row * 2 * 2;
  const int64_t num_blocks = shape.token_slots / shape.block_tokens;
  return static_cast<size_t>(shape.layers) * layer_kv +
         PagedBlockTable::table_bytes(shape.max_requests, num_blocks);
}

void Qwen35KvPool::check_req(int req, const char* what) const {
  if (req < 0 || req >= shape_.max_requests)
    throw std::out_of_range(std::string("Qwen35KvPool: request out of range in ") + what);
}

void Qwen35KvPool::init(const Qwen35KvPoolShape& shape) {
  if (shape.layers <= 0 || shape.kv_heads <= 0 || shape.dim <= 0 || shape.block_tokens <= 0 ||
      shape.max_requests <= 0 || shape.token_slots <= 0)
    throw std::invalid_argument("Qwen35KvPool: shape fields must be positive");
  if (shape.token_slots % shape.block_tokens != 0)
    throw std::invalid_argument("Qwen35KvPool: token_slots must be a multiple of block_tokens");
  shape_ = shape;
  const int64_t num_blocks = shape.token_slots / shape.block_tokens;
  DGPP_CUDA_OK(cudaMalloc(&k_base_, layer_kv_elems() * shape.layers * 2));
  DGPP_CUDA_OK(cudaMalloc(&v_base_, layer_kv_elems() * shape.layers * 2));
  table_.init(shape.max_requests, shape.block_tokens, num_blocks);
  initialized_ = true;
}

QwenFullAttnCache Qwen35KvPool::view(int layer) const {
  if (layer < 0 || layer >= shape_.layers)
    throw std::out_of_range("Qwen35KvPool: layer out of range in view");
  QwenFullAttnCache c;
  c.k_cache = k_base_ + static_cast<size_t>(layer) * layer_kv_elems();
  c.v_cache = v_base_ + static_cast<size_t>(layer) * layer_kv_elems();
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
                         int rank, int world, int max_requests, int decode_rows, bool mtp)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, rank, world, residency, LoaderHeadSharding::Full,
              mtp && residency == LoaderResidency::Resident),
      mtp_(mtp) {
  if (max_tokens <= 0) throw std::invalid_argument("Qwen35Model: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("Qwen35Model: max_requests out of range");
  if (decode_rows > decode_rows_cap())
    throw std::invalid_argument("Qwen35Model: decode_rows exceeds the limit of 32");
  if (world != 1) throw std::invalid_argument("Qwen35Model: TP>1 is not implemented (world must be 1)");
  if (boundary != nullptr) throw std::invalid_argument("Qwen35Model: world 1 takes no boundary reducer");
  if (cfg_.eos_token_ids.empty()) throw std::invalid_argument("Qwen35Model: the config names no EOS token");
  if (mtp_ && cfg_.mtp_layer() < 0)
    throw std::invalid_argument("Qwen35Model: the config has no draft layer (mtp)");
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
  const int64_t lv = cfg_.gdn_value_heads, V = cfg_.gdn_value_head_dim, K = cfg_.gdn_key_head_dim;
  const int64_t C = 2 * static_cast<int64_t>(cfg_.gdn_key_heads) * K + lv * V;
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
  sp.lm_vocab_begin = 0;
  sp.lm_vocab_count = cfg_.vocab_size;
  sp.max_position_embeddings = cfg_.max_position_embeddings;
  sp.block_tokens = kv_block_tokens_static();
  sp.snapshot_align = 1;
  sp.draft_width = mtp_ ? H : 0;
  sp.eos = static_cast<int32_t>(cfg_.eos_token_ids[0]);
  init_session(sp);
  // Paged K/V over the Full layers, plus the draft plane when MTP is on.
  Qwen35KvPoolShape shape;
  shape.layers = num_full_ + (mtp_ ? 1 : 0);
  shape.kv_heads = cfg_.num_key_value_heads;
  shape.dim = cfg_.head_dim;
  shape.block_tokens = kv_block_tokens_static();
  shape.max_requests = max_requests;
  shape.token_slots = sp.max_cache_tokens;
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
  // NOTE: dense_gemv_rows() defaults to 16, but the FP8 GEMV row loop
  // re-reads weights per ≤4-row chunk (and per single row when smem can't
  // stage more, e.g. down-proj k=17408). Route m>=5 through the
  // weights-once streaming MMA instead; ≤4-row decodes stay on GEMV.
  gw_.mma_from_rows = 5;
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
  // Per-tensor FP8 prefill recipe (DGPP_FP8_PT_DENSE): Resident stacks
  // requantize every layer's MLP once at boot (dequant to the bridge, then
  // absmax + x/448 quantize). Streaming stacks keep the bridge: their
  // layers are not all resident, so there is nothing eager to build from.
  pt_enabled_ = fp8_per_tensor_enabled() && loader_.residency() == LoaderResidency::Resident;
  pt_attn_enabled_ = pt_enabled_ && fp8_pt_attn_enabled();
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
    // stages each dequant like the MLP requant above. DGPP_FP8_PT_ATTN=0
    // skips this half (MLP-only ablation): the views stay disabled.
    if (pt_attn_enabled_)
    {
      const int64_t lk = cfg_.gdn_key_heads, lv = cfg_.gdn_value_heads;
      const int64_t K = cfg_.gdn_key_head_dim, V = cfg_.gdn_value_head_dim;
      pt_gdn_C_ = 2 * lk * K + lv * V;
      pt_gdn_LV_ = lv * V;
      const int64_t lh = cfg_.num_attention_heads, lkv = cfg_.num_key_value_heads;
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
  // Blockwise-FP8 lm head (DGPP_FP8_HEAD, Resident only): the 248k-row BF16
  // head is the only multi-GB BF16 weight left on the decode path. Host
  // requant (fp8_quant::encode_block128, the loader's own encoder) once;
  // decode rows read half the bytes through the F32 scale-GEMM path.
  head_fp8_enabled_ = fp8_head_enabled() && loader_.residency() == LoaderResidency::Resident;
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
  DGPP_CUDA_OK(cudaMalloc(&gdn_rec_base_, static_cast<size_t>(max_requests) * num_gdn_ * rec_elems_ * 4));
  DGPP_CUDA_OK(cudaMalloc(&gdn_conv_base_, static_cast<size_t>(max_requests) * num_gdn_ * conv_elems_ * 2));
  // The verify's per-row GDN snapshots (the speculative rollback's source).
  const size_t spec_rows = static_cast<size_t>(max_decode_rows_);
  DGPP_CUDA_OK(cudaMalloc(&spec_rec_, spec_rows * static_cast<size_t>(num_gdn_) * rec_elems_ * 4));
  DGPP_CUDA_OK(
      cudaMalloc(&spec_conv_, spec_rows * static_cast<size_t>(num_gdn_) * conv_elems_ * 2));
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
  cudaFree(gdn_rec_base_);
  cudaFree(gdn_conv_base_);
  cudaFree(spec_rec_);
  cudaFree(spec_conv_);
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
  const size_t I = static_cast<size_t>(cfg_.intermediate_size);
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
  // Uniform FP8 dispatch (m<=4 GEMV rows, wider the weights-once streaming
  // MMA): ~5ms at small row counts, half the BF16 bytes. Past 128 rows the
  // blockwise dense kernel falls behind BF16 Lt (58ms vs 26ms at m=500,
  // measured), so wide heads (group walks, diagnostics) keep BF16.
  if (head_fp8_enabled_ && rows <= 128) {
    launch_scale_gemm_f32(act, static_cast<size_t>(H), head_fp8_, head_scales_, out, rows,
                          static_cast<int>(V), H, stream, static_cast<size_t>(V),
                          /*mma_from_rows=*/5);
    return;
  }
  gemm_.matmul(act, globals_.lm_head, out, rows, static_cast<int>(V), H, DType::BF16, GemmOut::F32,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
}

void Qwen35Model::dense_mlp(const uint16_t* x, uint16_t* out, int tokens,
                            const Qwen35DenseMlpResident& m, cudaStream_t stream, int layer,
                            bool resume) {
  const int64_t H = cfg_.hidden_size, I = cfg_.intermediate_size;
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
  launch_scale_gemm_bf16(x, static_cast<size_t>(H), m.gate_fp8.payload, m.gate_fp8.scales, gate_tmp_,
                         tokens, static_cast<int>(I), static_cast<int>(H), stream, 0, gw_.mma_from_rows);
  launch_scale_gemm_bf16(x, static_cast<size_t>(H), m.up_fp8.payload, m.up_fp8.scales, up_tmp_, tokens,
                         static_cast<int>(I), static_cast<int>(H), stream, 0, gw_.mma_from_rows);
  qwen35_swiglu_bf16(gate_tmp_, up_tmp_, gate_tmp_, static_cast<int64_t>(tokens) * I, stream);
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
                                    bool mtp, int decode_rows) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests out of range");
  if (mtp && cfg.mtp_layer() < 0)
    throw std::invalid_argument("plan_memory: the config has no draft layer (mtp)");
  (void)decode_rows;
  const LoaderHeadSharding head = LoaderHeadSharding::Full;
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
  shape.kv_heads = cfg.num_key_value_heads;
  shape.dim = cfg.head_dim;
  shape.block_tokens = kv_block_tokens_static();
  shape.max_requests = max_requests;
  shape.token_slots = cache_tokens;
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
  // Per-tensor FP8 recipe (DGPP_FP8_PT_DENSE=0 disables, Resident only):
  // boot-time E4M3 gate/up/down per layer plus one scale each, activation
  // scratch. Streaming stacks keep the bridge (nothing eager to build).
  if (fp8_per_tensor_enabled() && residency == LoaderResidency::Resident) {
    const size_t slots = static_cast<size_t>(cfg.num_hidden_layers) + (mtp ? 1 : 0);
    const size_t IH = static_cast<size_t>(cfg.intermediate_size) * cfg.hidden_size;
    plan.add("per-tensor fp8 mlp (gate/up/down E4M3 + scales)", 3 * slots * IH + slots * 3 * 4);
    plan.add("per-tensor fp8 activation scratch", M * I + 8);
  }
  if (fp8_per_tensor_enabled() && fp8_pt_attn_enabled() &&
      residency == LoaderResidency::Resident) {    // Attention projections, the same recipe: GDN qkv/z/out per GDN layer,
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
  // Blockwise-FP8 lm head (DGPP_FP8_HEAD, Resident only): half the bytes
  // per decode row.
  if (fp8_head_enabled() && residency == LoaderResidency::Resident) {
    const size_t Vv = static_cast<size_t>(cfg.vocab_size), Hh = static_cast<size_t>(cfg.hidden_size);
    plan.add("blockwise fp8 lm head (E4M3 + scales)", Vv * Hh + ((Vv + 127) / 128) * ((Hh + 127) / 128) * 4);
  }
  if (mtp) {
    // Draft scratch: e/en/hn/hin/r/h [M,H] + cat [M,2H].
    plan.add("mtp scratch", 8 * M * H * 2);
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
    const size_t rec = static_cast<size_t>(lv) * V * K;
    const size_t conv = static_cast<size_t>(C) * (cfg.gdn_conv_width - 1);
    const size_t rows = static_cast<size_t>(std::max({kDecodeRows, decode_rows, max_requests}));
    plan.add("spec snapshot rows", rows * static_cast<size_t>(num_gdn) * (rec * 4 + conv * 2));
  }
  return plan;
}

void Qwen35Model::reset_slot_state(int req) {
  if (num_gdn_ > 0) {
    DGPP_CUDA_OK(cudaMemsetAsync(gdn_rec(req, 0), 0, static_cast<size_t>(num_gdn_) * rec_elems_ * 4, stream_));
    DGPP_CUDA_OK(
        cudaMemsetAsync(gdn_conv(req, 0), 0, static_cast<size_t>(num_gdn_) * conv_elems_ * 2, stream_));
  }
  pool_.reset_request(req, stream_);
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
    add(gdn_rec(req, 0), spec_rec_ + row0 * num_gdn_ * rec_elems_, rec_bytes, rec_bytes);
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
  d2d(d, live ? gdn_rec(req, 0) : spec_rec_ + row * num_gdn_ * rec_elems_, rec_bytes, stream_);
  d2d(d + rec_bytes, live ? gdn_conv(req, 0) : spec_conv_ + row * num_gdn_ * conv_elems_, conv_bytes,
      stream_);
}

void Qwen35Model::read_state_snapshot(int req, const uint8_t* d) {
  if (num_gdn_ == 0) return;
  const size_t rec_bytes = static_cast<size_t>(num_gdn_) * rec_elems_ * 4;
  const size_t conv_bytes = static_cast<size_t>(num_gdn_) * conv_elems_ * 2;
  d2d(gdn_rec(req, 0), d, rec_bytes, stream_);
  d2d(gdn_conv(req, 0), d + rec_bytes, conv_bytes, stream_);
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
  qwen_configure_gemm_rows(gemm_, T, decode_row);
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
    full_->enqueue(x_, T, qrows, cache, attn_out_, stream_);
  }
  // Fused residual-add + post norm (bitwise the pair): one launch.
  qwen_add_rmsnorm_bf16(mtp_r_, attn_out_, r.post_norm, x_, T, H, eps, stream_);
  dense_mlp(x_, mlp_out_, T, r.mlp, stream_, cfg_.num_hidden_layers, first_pos > 0 && !decode_row);
  add_inplace_bf16(mtp_r_, mlp_out_, static_cast<size_t>(T) * H, stream_);
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
  const int H = cfg_.hidden_size;
  const float eps = cfg_.rms_norm_eps;
  const RowInputs in = begin_run(run);
  const bool batched = in.batched;
  const int num_requests = in.num_requests;
  const int64_t* tokens = in.tokens;
  const int64_t* d_pos = in.pos;
  const int32_t* d_req = in.req_ids;
  const int32_t* d_spans = in.spans;
  embed_gather_bf16(globals_.embed, tokens, resid_, T, H, stream_);
  Outputs out;
  // The speculative verify's per-row GDN snapshots (the rollback source).
  const bool snapshots = run.decode && run.snapshots;
  int full_ord = 0, gdn_ord = 0;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const Qwen35LayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r);
    qwen_rmsnorm_bf16(resid_, r.input_norm, x_, T, H, eps, stream_);
    if (r.kind == Qwen35LayerKind::Gdn) {
      KdaStateSnapshots rec_snap;
      KdaConvSnapshots conv_snap;
      if (snapshots) {
        rec_snap.states = spec_rec_ + static_cast<size_t>(gdn_ord) * rec_elems_;
        rec_snap.stride_elems = static_cast<int64_t>(num_gdn_) * rec_elems_;
        conv_snap.states = spec_conv_ + static_cast<size_t>(gdn_ord) * conv_elems_;
        conv_snap.stride_elems = static_cast<int64_t>(num_gdn_) * conv_elems_;
      }
      if (batched) {
        KdaRequestRows requests;
        requests.request_ids = d_req;
        requests.positions = d_pos;
        requests.spans = d_spans;
        requests.num_requests = num_requests;
        gdn_->enqueue_rows(x_, gdn_rec(0, gdn_ord),
                           static_cast<int64_t>(num_gdn_) * rec_elems_, gdn_conv(0, gdn_ord),
                           static_cast<int64_t>(num_gdn_) * conv_elems_, attn_out_, T, requests, stream_,
                           rec_snap, conv_snap);
      } else if (run.num_spans > 0) {
        int64_t row0 = 0;
        for (int sp = 0; sp < run.num_spans; ++sp) {
          const int len = run.span_lens[sp], sreq = run.span_reqs[sp];
          gdn_->enqueue(x_ + static_cast<size_t>(row0) * H, gdn_rec(sreq, gdn_ord),
                        gdn_conv(sreq, gdn_ord), attn_out_ + static_cast<size_t>(row0) * H, len, stream_,
                        KdaStateSnapshots{}, KdaConvSnapshots{}, KdaReplay{},
                        run.span_pos0[sp] > 0 && !run.decode);
          row0 += len;
        }
      } else {
        gdn_->enqueue(x_, gdn_rec(req, gdn_ord), gdn_conv(req, gdn_ord), attn_out_, T, stream_, rec_snap,
                      conv_snap, KdaReplay{}, run.pos0 > 0 && !run.decode);
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
          full_->enqueue(x_ + static_cast<size_t>(row0) * H, len, srows, cache, attn_out_ + static_cast<size_t>(row0) * H,
                         stream_);
          row0 += len;
        }
      } else {
        full_->enqueue(x_, T, qrows, cache, attn_out_, stream_);
      }
      ++full_ord;
    }
    // Fused residual-add + post norm (bitwise the pair): one launch.
    qwen_add_rmsnorm_bf16(resid_, attn_out_, r.post_norm, x_, T, H, eps, stream_);
    // A resume chunk's short tail takes the per-tensor MLP like the
    // attention resume above; group spans start at pos0 (resume false).
    // Decode/verify walks (run.decode) keep their exact GEMV dispatch.
    bool mlp_resume = false;    if (!run.decode) {
      if (run.num_spans > 0) {
        for (int sp = 0; sp < run.num_spans; ++sp)
          mlp_resume = mlp_resume || run.span_pos0[sp] > 0;
      } else {
        mlp_resume = run.pos0 > 0;
      }
    }
    dense_mlp(x_, mlp_out_, T, r.mlp, stream_, layer, mlp_resume);
    add_inplace_bf16(resid_, mlp_out_, static_cast<size_t>(T) * H, stream_);
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
  out = finish_run(run, std::move(out));
  return out;
}

}  // namespace dgpp
