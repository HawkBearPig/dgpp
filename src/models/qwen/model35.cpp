// Qwen3.5-27B dense model (FP8 or mixed NVFP4, standard pre-norm residual).
//
// 48 GDN layers (swish gate) + 16 Full GQA layers, dense SwiGLU MLPs,
// and the checkpoint's optional BF16 vision tower. GDN recurrent +
// conv state is model-owned per request slot; the Full layers' K/V lives in
// Qwen35KvPool (plus draft planes for MTP or DFlash2). Eager and graph
// serving share the same walks and prefix snapshot protocol.

#include "models/qwen/model35.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "kernels/add_rmsnorm.hpp"
#include "kernels/dflash2.hpp"
#include "kernels/fp4w_gemm.hpp"
#include "kernels/fp8_per_tensor.hpp"
#include "kernels/fp8_dequant.hpp"
#include "kernels/moe_w4a4.hpp"
#include "kernels/fp8_gemm.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/kernels.hpp"
#include "kernels/mma_gemv.hpp"
#include "kernels/qwen_mtp.hpp"
#include "kernels/qwen_norm.hpp"
#include "kernels/qsa.hpp"
#include "kernels/scale_gemm.hpp"
#include "kernels/qwen_vision.hpp"
#include "models/qwen/image_stage_plan.hpp"
#include "loaders/fp8_quant.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/config.hpp"

namespace dgpp {
namespace {

// Device-to-device copy on a stream (the snapshot helpers).
inline void d2d(void* dst, const void* src, size_t n, cudaStream_t stream) {
  DGPP_CUDA_OK(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, stream));
}

}  // namespace

// ---- Qwen35KvPool ------------------------------------------------------------

Qwen35KvPool::~Qwen35KvPool() {
  cudaFree(k_base_);
  cudaFree(v_base_);
}

size_t Qwen35KvPool::cache_bytes(const Qwen35KvPoolShape& shape) {
  const size_t kv_row = static_cast<size_t>(shape.kv_heads) * shape.dim;
  const size_t layer_kv = static_cast<size_t>(shape.token_slots) * kv_row * (shape.fp8 ? 1 : 2) * 2;
  const int64_t num_blocks = shape.token_slots / shape.block_tokens;
  return static_cast<size_t>(shape.layers) * layer_kv +
         DraftKvPlanes::bytes(shape.draft_layers, shape.draft_kv_heads, shape.draft_dim, shape.token_slots) +
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
  const bool staged = shape.dim == 256 && shape.block_tokens >= 32 && (shape.block_tokens & (shape.block_tokens - 1)) == 0;
  if (shape.fp8 && !staged)
    throw std::invalid_argument("Qwen35KvPool: the fp8 form needs dim 256 and a power-of-two page of >= 32 tokens");
  shape_ = shape;
  const int64_t num_blocks = shape.token_slots / shape.block_tokens;
  DGPP_CUDA_OK(cudaMalloc(&k_base_, layer_kv_bytes() * shape.layers));
  DGPP_CUDA_OK(cudaMalloc(&v_base_, layer_kv_bytes() * shape.layers));
  draft_.init(shape.draft_layers, shape.draft_kv_heads, shape.draft_dim, shape.token_slots, shape.block_tokens);
  table_.init(shape.max_requests, shape.block_tokens, num_blocks);
  maps_ready_ = false;
  if (staged) {
    const int64_t rows = static_cast<int64_t>(shape.layers) * shape.token_slots;
    maps_ = shape.fp8 ? full_attn_stage_maps_fp8(k_base_, v_base_, rows, shape.kv_heads)
                      : full_attn_stage_maps(reinterpret_cast<const uint16_t*>(k_base_),
                                             reinterpret_cast<const uint16_t*>(v_base_), rows, shape.kv_heads);
    maps_ready_ = true;
  }
  initialized_ = true;
}

QwenFullAttnCache Qwen35KvPool::view(int layer) const {
  if (layer < 0 || layer >= shape_.layers + shape_.draft_layers)
    throw std::out_of_range("Qwen35KvPool: layer out of range in view");
  QwenFullAttnCache c;
  if (layer < shape_.layers) {
    uint8_t* kb = k_base_ + static_cast<size_t>(layer) * layer_kv_bytes();
    uint8_t* vb = v_base_ + static_cast<size_t>(layer) * layer_kv_bytes();
    if (shape_.fp8) {
      c.fp8 = true;
      c.fp8_attention = shape_.fp8_attention;
      c.k8 = kb;
      c.v8 = vb;
    } else {
      c.k_cache = reinterpret_cast<uint16_t*>(kb);
      c.v_cache = reinterpret_cast<uint16_t*>(vb);
    }
    if (maps_ready_) {
      c.maps = &maps_;
      c.row_base = static_cast<int64_t>(layer) * shape_.token_slots;
    }
  } else {
    const auto plane = draft_.view(layer - shape_.layers);
    c.k_cache = plane.k;
    c.v_cache = plane.v;
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
  const size_t bytes = layer_kv_bytes() * shape_.layers;
  DGPP_CUDA_OK(cudaMemsetAsync(k_base_, 0, bytes, stream));
  DGPP_CUDA_OK(cudaMemsetAsync(v_base_, 0, bytes, stream));
  draft_.reset(stream);
  table_.reset_all(stream);
}

std::vector<CachePlane> Qwen35KvPool::planes() const {
  std::vector<CachePlane> out;
  if (!initialized_) return out;
  const size_t span = static_cast<size_t>(shape_.block_tokens) * kv_row_elems() * elem_bytes();
  for (int l = 0; l < shape_.layers; ++l) {
    const size_t base = static_cast<size_t>(l) * layer_kv_bytes();
    out.push_back({k_base_ + base, span});
    out.push_back({v_base_ + base, span});
  }
  draft_.append_planes(out);
  return out;
}

void Qwen35KvPool::copy_block_contents(int32_t src, int32_t dst, cudaStream_t stream) {
  if (src < 0 || src >= total_blocks() || dst < 0 || dst >= total_blocks())
    throw std::out_of_range("Qwen35KvPool: block out of range in copy_block_contents");
  const size_t span = static_cast<size_t>(shape_.block_tokens) * kv_row_elems() * elem_bytes();  // bytes
  for (int l = 0; l < shape_.layers; ++l) {
    const size_t base = static_cast<size_t>(l) * layer_kv_bytes();
    d2d(k_base_ + base + static_cast<size_t>(dst) * span, k_base_ + base + static_cast<size_t>(src) * span, span,
        stream);
    d2d(v_base_ + base + static_cast<size_t>(dst) * span, v_base_ + base + static_cast<size_t>(src) * span, span,
        stream);
  }
  // Every layer's rows of physical block `src` into `dst`, stream-ordered.
  draft_.copy_block(src, dst, stream);
}

// ---- Qwen35Model ---------------------------------------------------------------

// The pool format a kv_dtype asks for: bf16, or the fp8 storage form —
// at the release's kv_cache_scheme scales where it ships them (the mixed
// release) and unit scales where it does not (the FP8 release, the MTP
// draft layer; 2026-10-18).
// engine.prefill_attention = fp8 (the attention's arithmetic on that pool)
// needs it: with a bf16 pool there are no e4m3 K/V codes to compute over.
static bool kv_format_is_fp8(LatentFormat f, const Qwen35TextConfig& cfg, bool mtp) {
  if (f == LatentFormat::kBf16) {
    if (Qwen35Model::fp8_attention())
      throw std::invalid_argument(
          "Qwen35Model: engine.prefill_attention fp8 needs engine.kv_dtype fp8 (the attention computes over the "
          "pool's e4m3 codes at the release's k_scale / v_scale); set kv_dtype fp8 or prefill_attention bf16");
    return false;
  }
  if (f != LatentFormat::kFp8)
    throw std::invalid_argument(std::string("Qwen35Model: engine.kv_dtype ") + latent_format_name(f) +
                                " — this family's K/V pool takes bf16 or fp8");
  (void)cfg;
  (void)mtp;
  return true;
}


// The dequant bridge holds the largest dense FP8 matrix in BF16 so
// prefill-shaped (m>128) products dequantize once and run cuBLASLt BF16
// instead of the slow hand-rolled tile kernel (mirrors
// QwenModel::dense_bridge_bytes; world 1 takes full rows).
static size_t qwen35_dense_bridge_bytes(const Qwen35TextConfig& cfg) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t q = 2 * static_cast<size_t>(cfg.num_attention_heads) * cfg.head_dim;
  const size_t kv = static_cast<size_t>(cfg.num_key_value_heads) * cfg.head_dim;
  const size_t o = static_cast<size_t>(cfg.num_attention_heads) * cfg.head_dim;
  size_t elems = 0;
  const auto take = [&](size_t n, size_t k) { elems = std::max(elems, n * k); };
  take(q, H);    // q_proj
  take(kv, H);   // k / v
  take(H, o);    // o_proj
  if (!cfg.is_moe) {
    const size_t I = static_cast<size_t>(cfg.intermediate_size);
    take(I, H);  // gate / up
    take(H, I);  // down
  }
  return elems * 2;
}

Qwen35Model::Qwen35Model(const Qwen35TextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
                         int64_t max_cache_tokens, LoaderResidency residency, BoundaryReducer* boundary,
                         int rank, int world, int max_requests, int decode_rows, bool mtp,
                         const std::string& dflash2_dir, LatentFormat kv_format)
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
  // The FP8-release levers (the boot block-FP8 lm head, the per-tensor
  // prefill recipe) requantize the checkpoint's block-FP8 bytes. The mixed
  // NVFP4 release serves its own formats (NVFP4 MLP, FP8-channel
  // attention): neither lever has a subject there, and an operator asking
  // for one has the wrong release or the wrong config. The MoE NVFP4
  // release (routed experts NVFP4, rest BF16) likewise serves its own
  // formats. The MoE MTP draft's BF16 experts use the explicit conversion
  // key below; these dense-family options do not convert the target.
  if ((cfg_.quant_kind == Qwen35QuantKind::Nvfp4Mixed ||
       cfg_.quant_kind == Qwen35QuantKind::Nvfp4Moe) &&
      (prefill_fp8_per_tensor_ || dense_weights_fp8_))
    throw std::invalid_argument(
        "Qwen35Model: the NVFP4 release serves the checkpoint's own formats; "
        "engine.prefill_fp8_per_tensor and engine.dense_weights = fp8 are the FP8 "
        "release's levers (unset them)");
  if (cfg_.is_moe && mtp_ && Qwen35LayerStream::mtp_expert_format() != "bf16")
    throw std::invalid_argument("Qwen35Model: the MoE draft needs engine.mtp_expert_format=bf16 "
                                "(per-expert BF16 encoded to block FP8 at load for the proposals only)");
  if (!cfg_.is_moe && mtp_ && Qwen35LayerStream::mtp_expert_format() == "bf16")
    throw std::invalid_argument("Qwen35Model: engine.mtp_expert_format=bf16 is the MoE draft's key; "
                                "the dense draft serves its BF16 MLP as shipped");
  DFlash2Config dfcfg;  // v2, parsed here for the pool's draft-plane geometry
  if (!dflash2_dir.empty()) {
    if (mtp_)
      throw std::invalid_argument("Qwen35Model: dflash2 replaces the MTP draft; enable one or the other");
    // v1 vs v2 by the drafter's own architecture string (never by guess).
    const std::string darch = drafter_architecture(dflash2_dir);
    if (darch == "DFlashDraftModel") {
      df1cfg_ = DFlashConfig::from_json_file(dflash2_dir + "/config.json");
      df1cfg_.validate_against(cfg_);
      df1_tap_.assign(cfg_.num_hidden_layers, -1);
      for (size_t t = 0; t < df1cfg_.target_layer_ids.size(); ++t)
        df1_tap_[df1cfg_.target_layer_ids[t]] = static_cast<int>(t);
      dflash1_ = true;
    } else {
      dfcfg = DFlash2Config::from_json_file(dflash2_dir + "/config.json");
      dfcfg.validate_against(DFlash2TargetShape{cfg_.hidden_size, cfg_.vocab_size,
                                                cfg_.num_hidden_layers, cfg_.max_position_embeddings});
      dflash2_ = true;
    }
  }
  if (dflash1_) {
    // The v1 draft keeps BF16 weights; batches reuse its block scratch
    // sequentially in both the eager and recorded paths.
    if (dflash2_fp8_)
      throw std::invalid_argument("Qwen35Model: engine.dflash_weights=fp8 is the DFlash2 block recipe; the v1 drafter serves BF16");
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
  {
    int dev = 0, sms = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    wave_rows_ = sms > 0 ? static_cast<int64_t>(16) * sms / std::max(1, geo.local_kv_heads) : 0;
  }
  conv_elems_ = C * (cfg_.gdn_conv_width - 1);
  // The qwen4_exp layer ctors' config view.
  qcfg_ = qwen35_text_adapter(cfg_);
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
  sp.max_position_embeddings = cfg_.context_limit();
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
  shape.fp8 = kv_format_is_fp8(kv_format, cfg_, mtp_);
  shape.fp8_attention = fp8_attention_;
  if (dflash2_) {
    shape.draft_layers = dfcfg.num_hidden_layers;
    shape.draft_kv_heads = dfcfg.local_kv_heads(world);  // this rank's kv heads (sharded drafter)
    shape.draft_dim = dfcfg.head_dim;
  }
  if (dflash1_) {
    shape.draft_layers = df1cfg_.num_hidden_layers;
    shape.draft_kv_heads = df1cfg_.local_kv_heads(world);
    shape.draft_dim = df1cfg_.head_dim;
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
  // Dense MLP width; the MoE variant runs the routed chain (its own
  // scratch inside QwenMoeLayer) and needs no dense gate/up tmps.
  const size_t I = cfg_.is_moe ? 0 : static_cast<size_t>(cfg_.intermediate_size);
  DGPP_CUDA_OK(cudaMalloc(&resid_, M * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&x_, M * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&attn_out_, M * H * 2));
  DGPP_CUDA_OK(cudaMalloc(&mlp_out_, M * H * 2));
  if (cfg_.is_moe) {
    gate_tmp_ = nullptr;
    up_tmp_ = nullptr;
    // The routed chain on this rank's expert slice (norm_topk from config).
    moe_cfg_ = QwenMoeLayer::routed_config(
        cfg_.hidden_size, static_cast<int>(loader_.geometry().local_moe_inter),
        cfg_.num_experts, cfg_.num_experts_per_tok, cfg_.norm_topk_prob);
  } else {
    DGPP_CUDA_OK(cudaMalloc(&gate_tmp_, M * I * 2));
    DGPP_CUDA_OK(cudaMalloc(&up_tmp_, M * I * 2));
  }
  // The NVFP4 dense path's tables (production ldmatrix kernel): one
  // {0, m, 0} segment per row count, staged once — the grouped launcher's
  // only per-launch table, indexed by (rows - 1), never written after.
  DGPP_CUDA_OK(cudaMalloc(&fp4_segs_, M * sizeof(MoeSegment)));
  {
    std::vector<MoeSegment> segs(M);
    for (size_t m = 1; m <= M; ++m) segs[m - 1] = MoeSegment{0, static_cast<int32_t>(m), 0};
    DGPP_CUDA_OK(cudaMemcpy(fp4_segs_, segs.data(), M * sizeof(MoeSegment),
                            cudaMemcpyHostToDevice));
  }
  // The native activation recipe's NVFP4 scratch (engine.prefill_activations
  // = native): one quantized row set at max_tokens rows of the widest k (I).
  if (native_activations_ && cfg_.quant_kind == Qwen35QuantKind::Nvfp4Mixed && loader_.residency() == LoaderResidency::Resident)
    build_native_maps();
  if (native_activations_ && cfg_.quant_kind == Qwen35QuantKind::Nvfp4Mixed) {
    DGPP_CUDA_OK(cudaMalloc(&q_codes_, M * I / 2));
    DGPP_CUDA_OK(cudaMalloc(&q_scales_, M * nvfp4_act_scale_stride(static_cast<int>(I))));
    DGPP_CUDA_OK(cudaMalloc(&q_gs_, M * sizeof(float)));
    // The dense W4A4 TMA form: maps over the code scratch for k = H (gate/up)
    // and k = I (down), and over every NVFP4 MLP matrix (Resident stacks).
    for (const int64_t kk : {static_cast<int64_t>(H), static_cast<int64_t>(I)}) {
      if (!dense_w4a4_tma_supported(static_cast<int>(kk)) || q_map_for(static_cast<int>(kk)) != nullptr) continue;
      q_tma_k_[q_tma_count_] = static_cast<int>(kk);
      q_tma_[q_tma_count_] = nvfp4_code_tma_map(q_codes_, static_cast<int64_t>(M), kk, 128);
      ++q_tma_count_;
    }
    if (loader_.residency() == LoaderResidency::Resident) build_native_fp4_maps();
    DGPP_LOG_INFO("Qwen35Model: engine.prefill_activations = native: the NVFP4 MLP recipe on every prefill row");
  }
  // The recipe's e4m3 half (both releases): per-token activation rows for the
  // channel grid's projections and late-layer MLPs (the mixed release) or the
  // blockwise 1 x 128 groups (the FP8 release), at max_tokens rows of the
  // widest k (I); the cell the layers' workspace copies read.
  native_on_ = native_activations_;
  if (native_on_) {
    DGPP_CUDA_OK(cudaMalloc(&gw_.a8, M * I));
    DGPP_CUDA_OK(cudaMalloc(&gw_.a8_scales, M * (I / 128 + 1) * sizeof(float)));
    gw_.a8_bytes = M * I;
    gw_.native_prefill = &native_prefill_cell_;
    // The a8 scratch's TMA maps for every dense k (the channel form's boxes).
    {
      const int64_t LV = static_cast<int64_t>(geo.local_value_heads) * cfg_.gdn_value_head_dim;
      const int64_t FH = static_cast<int64_t>(geo.local_heads) * cfg_.head_dim;
      for (const int64_t kk : {static_cast<int64_t>(H), LV, FH, static_cast<int64_t>(I)}) {
        if (!fp8_channel_tma_supported(static_cast<int>(kk)) || gw_.a8_map_for(static_cast<int>(kk)) != nullptr ||
            gw_.a8_tma_count >= 4)
          continue;
        gw_.a8_tma[gw_.a8_tma_count].k = static_cast<int>(kk);
        gw_.a8_tma[gw_.a8_tma_count].map = fp8_channel_tma_map(gw_.a8, static_cast<int64_t>(M), kk, 128);
        ++gw_.a8_tma_count;
      }
    }
    DGPP_LOG_INFO("Qwen35Model: engine.prefill_activations = native: per-token e4m3 activations for the FP8 projections on every prefill row");
  }
  if (cfg_.quant_kind == Qwen35QuantKind::Nvfp4Mixed &&
      loader_.residency() == LoaderResidency::Resident) {
    // Resident: harvest every layer's [gate,up,down] views once — the
    // resident bumps are stable, so the tables stay valid for the process
    // lifetime (and the decode graph stays kernels-only).
    fp4_view_layers_ = cfg_.num_hidden_layers + (mtp_ ? 1 : 0);
    DGPP_CUDA_OK(cudaMalloc(&fp4_views_,
                            static_cast<size_t>(fp4_view_layers_) * 3 * sizeof(MoeExpertView)));
    std::vector<MoeExpertView> views(static_cast<size_t>(fp4_view_layers_) * 3);
    for (int l = 0; l < fp4_view_layers_; ++l) {
      const Qwen35LayerResident& r = loader_.load_layer(l);
      if (r.mlp.gate_fp4.payload == nullptr) continue;  // BF16/channel layer: slot unused
      views[static_cast<size_t>(l) * 3 + 0] = MoeExpertView::of(r.mlp.gate_fp4);
      views[static_cast<size_t>(l) * 3 + 1] = MoeExpertView::of(r.mlp.up_fp4);
      views[static_cast<size_t>(l) * 3 + 2] = MoeExpertView::of(r.mlp.down_fp4);
    }
    DGPP_CUDA_OK(cudaMemcpy(fp4_views_, views.data(), views.size() * sizeof(MoeExpertView),
                            cudaMemcpyHostToDevice));
  } else {
    DGPP_CUDA_OK(cudaMalloc(&fp4_views_, 3 * sizeof(MoeExpertView)));
  }
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
    head_fp8_view_.payload = head_fp8_;
    head_fp8_view_.scales = head_scales_;
    head_fp8_view_.rows = static_cast<int64_t>(V);
    head_fp8_view_.cols = static_cast<int64_t>(Hh);
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
    // The DFlash2 drafter (models/dflash2/drafter.hpp, shared with the
    // Flash-Next model): the component owns the weights, the rope table,
    // the scratch and the execution; the family supplies the pool planes
    // (the draft planes ride the Full layers' ordinals), the draft-screen
    // head, the gemm row envelope and the session/graph state.
    DFlash2Host host;
    host.hidden = cfg_.hidden_size;
    host.num_target_layers = cfg_.num_hidden_layers;
    host.target_vocab = cfg_.vocab_size;
    host.vocab_count = lm_vocab_count_;
    host.vocab_begin = lm_vocab_begin_;
    host.rank = rank;
    host.world = world;
    host.stream = stream_;
    host.max_requests = max_requests;
    host.max_tokens = static_cast<int>(M);
    host.max_decode_rows = max_decode_rows_;
    host.max_context = max_context();
    host.position_limit = cfg_.context_limit();
    host.weights_fp8 = dflash2_fp8_;
    host.rope = cfg_.rope_scaling.has_value() ? &*cfg_.rope_scaling : nullptr;
    host.device_tables = pool_.blocks().device_tables();
    host.total_blocks = static_cast<int>(pool_.total_blocks());
    host.block_tokens = kv_block_tokens_static();
    host.ensure_request_blocks =
        [this](int req, int64_t pos, cudaStream_t s) { return pool_.ensure_request_blocks(req, pos, s); };
    host.plane = [this](int l) {
      const auto c = pool_.view(num_full_ + l);
      return DFlash2Host::Plane{c.k_cache, c.v_cache};
    };
    host.session_position = [this](int req) { return session_position(req); };
    host.session_pos = d_session_pos_;
    host.decode_tokens = d_tokens_;
    host.embed = globals_.embed;
    host.gemm = &gemm_;
    host.gemm_ws = gemm_ws_;
    host.gemm_ws_bytes = gemm_ws_bytes_;
    host.gemm_rows = [this](int rows, bool decode) { configure_gemm_rows(rows, decode); };
    host.boundary = [this] { return boundary_; };
    host.head = [this](const uint16_t* hidden, float* out, int rows) {
      head_gemv(hidden, out, rows, stream_, /*draft=*/true);
    };
    df_.init(host, dflash2_dir);
  }
  if (dflash1_) {
    // v1 weights (bf16, TP-sliced) and the rope table.
    if (!df1cfg_.tp_divisible(world))
      throw std::invalid_argument("Qwen35Model: the DFlash v1 drafter's heads / kv heads / MLP rows must divide across the world");
    df1w_ = load_dflash_weights(df1cfg_, dflash2_dir, stream_, rank, world);
    {
      std::vector<float> host(df1cfg_.head_dim / 2);
      const double theta = df1cfg_.rope_theta;
      for (size_t i = 0; i < host.size(); ++i)
        host[i] = static_cast<float>(
            std::pow(theta, -static_cast<double>(2 * i) / static_cast<double>(df1cfg_.head_dim)));
      DGPP_CUDA_OK(cudaMalloc(&df1_inv_freq_, host.size() * 4));
      DGPP_CUDA_OK(cudaMemcpy(df1_inv_freq_, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    }
    // One block of scratch, reused in stream order by each request.
    const int QR1 = df1cfg_.query_rows(), D1 = df1cfg_.drafts();
    const int64_t H1 = cfg_.hidden_size, I1 = df1cfg_.local_intermediate(world);
    const int64_t QW1 = df1cfg_.local_q_row(world), KW1 = df1cfg_.local_kv_row(world);
    DGPP_CUDA_OK(cudaMalloc(&df1_acc_, M * H * 4));
    DGPP_CUDA_OK(cudaMalloc(&df1_t32_, M * H * 4));
    DGPP_CUDA_OK(cudaMalloc(&df1_resid_, static_cast<size_t>(QR1) * H1 * 2));
    DGPP_CUDA_OK(cudaMalloc(&df1_x_, static_cast<size_t>(QR1) * H1 * 2));
    DGPP_CUDA_OK(cudaMalloc(&df1_q_, static_cast<size_t>(QR1) * QW1 * 2));
    DGPP_CUDA_OK(cudaMalloc(&df1_k_, static_cast<size_t>(QR1) * KW1 * 2));
    DGPP_CUDA_OK(cudaMalloc(&df1_v_, static_cast<size_t>(QR1) * KW1 * 2));
    DGPP_CUDA_OK(cudaMalloc(&df1_attn_, static_cast<size_t>(QR1) * QW1 * 2));
    DGPP_CUDA_OK(cudaMalloc(&df1_attn_part_, dflash2_block_attn_partials_bytes(QR1, df1cfg_.local_heads(world))));
    DGPP_CUDA_OK(cudaMalloc(&df1_o_, static_cast<size_t>(QR1) * H1 * 2));
    DGPP_CUDA_OK(cudaMalloc(&df1_mlp_, static_cast<size_t>(QR1) * I1 * 2));
    DGPP_CUDA_OK(cudaMalloc(&df1_gate_, static_cast<size_t>(QR1) * I1 * 2));
    DGPP_CUDA_OK(cudaMalloc(&df1_up_, static_cast<size_t>(QR1) * I1 * 2));
    DGPP_CUDA_OK(cudaMalloc(&df1_h_, static_cast<size_t>(QR1) * H1 * 2));
    DGPP_CUDA_OK(cudaMalloc(&df1_logits_, static_cast<size_t>(QR1) * lm_vocab_count_ * 4));
    df1_topk_ws_bytes_ = dflash2_topk_ws_bytes(lm_vocab_count_, QR1, 16);
    DGPP_CUDA_OK(cudaMalloc(&df1_topk_ws_, df1_topk_ws_bytes_));
    if (world > 1)
      DGPP_CUDA_OK(cudaMalloc(&df1_table_, dflash2_topk_table_elems(1, 16, world) * 2));
    DGPP_CUDA_OK(cudaMalloc(&df1_ids_, static_cast<size_t>(QR1) * 16 * 4));
    DGPP_CUDA_OK(cudaMalloc(&df1_sc_, static_cast<size_t>(QR1) * 16 * 4));
    DGPP_CUDA_OK(cudaMalloc(&df1_zero_, static_cast<size_t>(QR1) * 4));
    DGPP_CUDA_OK(cudaMemsetAsync(df1_zero_, 0, static_cast<size_t>(QR1) * 4, stream_));
    DGPP_CUDA_OK(cudaMalloc(&df1_pos_, static_cast<size_t>(QR1) * 8));
    DGPP_CUDA_OK(cudaMalloc(&df1_tokens_, static_cast<size_t>(QR1) * 8));
    DGPP_CUDA_OK(cudaMallocHost(&df1_io64_h_, static_cast<size_t>(2) * QR1 * 8));
    DGPP_CUDA_OK(cudaMalloc(&df1_tok_, static_cast<size_t>(D1) * 4));
    DGPP_CUDA_OK(cudaMallocHost(&df1_tok_h_, static_cast<size_t>(D1) * 4));
    DGPP_CUDA_OK(cudaMallocHost(&df1_mirror_h_, static_cast<size_t>(max_requests) * D1 * 4));
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
  // The tower is replicated (it will run only on image prefills, never
  // inside a captured graph), so its weights are this rank's own copy.
  // The weights stay resident; text-only requests never execute the encoder.
  if (cfg_.vision) vision_ = std::make_unique<QwenVisionEncoder>(*cfg_.vision, checkpoint_dir, stream_);
  // The draft head's screen (engine.draft_head): the MTP draft's head and
  // the DFlash2 block's mask rows read through it; off with a logged reason
  // when the setting or the head's form says so.
  if (mtp_ || dflash2_) draft_head_.init(draft_head_matrix(), max_decode_rows_, rank_, stream_, /*quiet=*/true);
}

HeadMatrix Qwen35Model::draft_head_matrix() const {
  HeadMatrix m;
  if (globals_.lm_head_fp8.payload) m.fp8 = &globals_.lm_head_fp8;         // the mixed release's channel-fp8 head
  else if (head_fp8_enabled_ && head_fp8_ != nullptr) m.fp8 = &head_fp8_view_;  // the boot block-fp8 copy
  else {
    m.bf16 = globals_.lm_head;
    m.bf12 = gemm_.bf12_lookup(globals_.lm_head);
  }
  m.n = lm_vocab_count_;
  m.k = cfg_.hidden_size;
  return m;
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
    const DFlash2Config& dfc = df_.cfg();
    const int64_t dH = dfc.hidden_size, dI = dfc.local_intermediate(world_);
    const int64_t QW = dfc.local_q_row(world_), KV = dfc.local_kv_row(world_);
    for (const DFlash2LayerWeights& w : df_.weights().layers) {
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
    for (size_t t = 0; t < dfc.target_layer_ids.size(); ++t)
      pack(df_.weights().fc + t * static_cast<size_t>(dH) * static_cast<size_t>(dH), dH, dH);
    pack(df_.weights().hidden_projection, dfc.selector_rank, dH);
  }
  if (mtp_) pack(globals_.mtp_fc, H, 2 * H);
  if (!head_fp8_enabled_) pack(globals_.lm_head, lm_vocab_count_, H);
  bf12_.finish(gemm_, 1);
  bf12_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  bf12_.log_summary(rank_, bf12_s_);
  // The draft head's screen from the companion when the bf16 bytes were
  // never device-resident (the 12-bit-only residency); a no-op once built.
  if (mtp_ || dflash2_) draft_head_.init(draft_head_matrix(), max_decode_rows_, rank_, stream_);
}

Qwen35Model::~Qwen35Model() {
  cudaFree(resid_);
  cudaFree(x_);
  cudaFree(attn_out_);
  cudaFree(mlp_out_);
  cudaFree(gate_tmp_);
  cudaFree(up_tmp_);
  cudaFree(fp4_segs_);
  cudaFree(q_codes_);
  cudaFree(q_scales_);
  cudaFree(q_gs_);
  if (native_on_) {
    cudaFree(gw_.a8);
    cudaFree(gw_.a8_scales);
  }
  cudaFree(fp4_views_);
  cudaFree(mtp_e_);
  cudaFree(mtp_en_);
  cudaFree(mtp_hn_);
  cudaFree(mtp_hin_);
  cudaFree(mtp_cat_);
  cudaFree(mtp_r_);
  cudaFree(mtp_h_);
  cudaFree(df1_acc_);
  cudaFree(df1_t32_);
  cudaFree(df1_table_);
  cudaFree(df1_resid_);
  cudaFree(df1_x_);
  cudaFree(df1_q_);
  cudaFree(df1_k_);
  cudaFree(df1_v_);
  cudaFree(df1_attn_);
  cudaFree(df1_attn_part_);
  cudaFree(df1_o_);
  cudaFree(df1_mlp_);
  cudaFree(df1_gate_);
  cudaFree(df1_up_);
  cudaFree(df1_h_);
  cudaFree(df1_logits_);
  cudaFree(df1_topk_ws_);
  cudaFree(df1_ids_);
  cudaFree(df1_sc_);
  cudaFree(df1_zero_);
  cudaFree(df1_pos_);
  cudaFree(df1_tokens_);
  cudaFree(df1_tok_);
  cudaFreeHost(df1_io64_h_);
  cudaFreeHost(df1_tok_h_);
  cudaFreeHost(df1_mirror_h_);
  cudaFree(df1_inv_freq_);
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
  native_prefill_cell_ = native_on_ && !decode;  // the native recipe's rows (engine.prefill_activations)
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
  if (cfg_.is_moe) {
    if (!moe_) {
      const int table_slots = loader_.residency() == LoaderResidency::Resident
                                  ? cfg_.num_hidden_layers + (mtp_ ? 1 : 0)
                                  : 0;
      moe_ = std::make_unique<QwenMoeLayer>(moe_view(r.moe), moe_cfg_, gemm_, max_tokens_,
                                            max_decode_rows_, table_slots);
    } else {
      moe_->rebind(moe_view(r.moe));
    }
  }
}

QwenMoeWeights Qwen35Model::moe_view(const QwenMoeResident& m) {
  QwenMoeWeights w;
  w.router = m.router;
  w.shared_gate = m.shared_gate;
  w.shared_gate_proj = m.shared[0];
  w.shared_up_proj = m.shared[1];
  w.shared_down_proj = m.shared[2];
  w.shared_fp8 = nullptr;  // MoE shared expert is BF16 (no dense_weights fp8 lever)
  w.shared_inter = m.local_shared_inter;
  w.experts = m.experts.empty() ? nullptr : m.experts.data();
  w.experts_fp4 = m.experts_fp4.empty() ? nullptr : m.experts_fp4.data();
  w.experts_packed = nullptr;
  w.act_scale_w13 = 0.0f;
  w.act_scale_w2 = 0.0f;
  w.act_scales_dev = nullptr;
  return w;
}

void Qwen35Model::moe_mlp(const uint16_t* x, uint16_t* out, int tokens,
                          const QwenMoeResident& m, cudaStream_t stream, int table_slot,
                          bool decode) {
  if (!moe_) throw std::logic_error("moe_mlp: no MoE layer (build_layer_objects first)");
  // Streaming stacks rebind one shared layer; resident stacks hold every
  // layer — either way the views must match this layer before enqueue.
  moe_->rebind(moe_view(m));
  if (decode)
    moe_->enqueue_decode(x, out, tokens, stream, table_slot);
  else
    moe_->enqueue_prefill(x, out, tokens, stream, nullptr);
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
void Qwen35Model::head_gemv(const uint16_t* act, float* out, int rows, cudaStream_t stream, bool draft) {
  const int H = cfg_.hidden_size;
  const int64_t V = lm_vocab_count_;
  // A decode draft's read through the screen (kernels/draft_head.hpp): the
  // candidate blocks exact, -inf elsewhere; the top-K / pick run unchanged.
  if (draft && draft_head_.logits(draft_head_matrix(), act, static_cast<size_t>(H), rows, out, static_cast<size_t>(V), stream))
    return;
  // The mixed release's native head: FP8-channel, read through the channel
  // grid's streaming form (the lm_head BF16 pointer is null there). Past
  // 128 rows it rides the bridge's channel dequant (a wide-head call is a
  // diagnostic, and the head is 1.3 GB in BF16 — the bridge holds it).
  if (globals_.lm_head_fp8.payload) {
    int rs = 0, cs = 0;
    if (!quant_scale_shifts(globals_.lm_head_fp8, rs, cs))
      throw std::runtime_error("qwen35 head: the native lm head's scale grid is malformed");
    if (rows <= 128) {
      launch_scale_gemm_grid_f32(act, static_cast<size_t>(H), globals_.lm_head_fp8.payload,
                                 globals_.lm_head_fp8.scales, out, rows, static_cast<int>(V), H,
                                 stream, static_cast<size_t>(V), rs, cs, /*decode_mma=*/true,
                                 gw_.ws, gw_.ws_bytes);
      return;
    }
    // Past 128 rows (a diagnostic wide head) the GEMV rows: the bridge is
    // sized for the largest dense MLP matrix, not the 1.3 GB head.
    launch_scale_gemm_grid_f32(act, static_cast<size_t>(H), globals_.lm_head_fp8.payload,
                               globals_.lm_head_fp8.scales, out, rows, static_cast<int>(V), H,
                               stream, static_cast<size_t>(V), rs, cs, /*decode_mma=*/false,
                               nullptr, 0);
    return;
  }
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
  if (!globals_.lm_head)
    throw std::runtime_error("qwen35 head: the release binds no BF16 lm head");
  gemm_.matmul(act, globals_.lm_head, out, rows, static_cast<int>(V), H, DType::BF16, GemmOut::F32,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
}

void Qwen35Model::dense_mlp(const uint16_t* x, uint16_t* out, int tokens,
                            const Qwen35DenseMlpResident& m, cudaStream_t stream, int layer,
                            bool resume, bool prefill) {
  // I is this rank's MLP slice (the resident gate rows): intermediate_size /
  // world under TP, the whole at world 1.
  const int64_t H = cfg_.hidden_size, I = m.inter ? m.inter : m.gate_fp8.rows;
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
  // The checkpoint's NVFP4 activation recipe (engine.prefill_activations =
  // native, 2026-10-15): every prefill row's MLP input quantized to NVFP4 per
  // row with the shipped input_global_scale (gate and up share one; down has
  // its own), the three products on the block-scaled fp4 tensor cores
  // (kernels/moe_w4a4 over one segment; 4x the bf16 mma rate measured), the
  // model's SwiGLU between them. Per-row quantization and a fixed K order keep
  // a row's chain the same at any row count and cut. Decode rows keep W4A16:
  // bandwidth-bound, their bf16 activations cost nothing.
  // (The instance's scratch is the switch: it exists only when the model was
  // built under the key, so a model keeps its recipe whatever the static
  // flag says later — the test builds both chains in one process.)
  if (prefill && q_codes_ && m.gate_fp4.payload && m.fp4_input_gs && tokens > 0) {
    const MoeExpertView* views = nullptr;
    if (fp4_view_layers_ > 0 && layer >= 0 && layer < fp4_view_layers_) {
      views = fp4_views_ + static_cast<size_t>(layer) * 3;
    } else {
      MoeExpertView slot[3] = {MoeExpertView::of(m.gate_fp4), MoeExpertView::of(m.up_fp4),
                               MoeExpertView::of(m.down_fp4)};
      DGPP_CUDA_OK(cudaMemcpyAsync(fp4_views_, slot, sizeof(slot), cudaMemcpyHostToDevice, stream));
      views = fp4_views_;
    }
    const MoeSegment* seg = fp4_segs_ + (tokens - 1);
    // The TMA-staged dense form when the maps exist (bitwise the grouped kernel).
    const CUtensorMap* qmH = q_map_for(static_cast<int>(H));
    const CUtensorMap* qmI = q_map_for(static_cast<int>(I));
    const bool boxes = qmH && qmI && m.gate_fp4.tma && m.up_fp4.tma && m.down_fp4.tma;
    const size_t ssH = nvfp4_act_scale_stride(static_cast<int>(H)), ssI = nvfp4_act_scale_stride(static_cast<int>(I));
    launch_quantize_rows_nvfp4(x, static_cast<size_t>(H), tokens, static_cast<int>(H), q_codes_, q_scales_, q_gs_,
                               stream, 0.f, m.fp4_input_gs);
    if (boxes) {
      launch_dense_w4a4_tma_bf16(*qmH, *m.gate_fp4.tma, q_scales_, ssH, q_gs_, m.gate_fp4.scales, m.gate_fp4.global_scale,
                                 gate_tmp_, static_cast<size_t>(I), tokens, static_cast<int>(I), static_cast<int>(H), stream);
      launch_dense_w4a4_tma_bf16(*qmH, *m.up_fp4.tma, q_scales_, ssH, q_gs_, m.up_fp4.scales, m.up_fp4.global_scale,
                                 up_tmp_, static_cast<size_t>(I), tokens, static_cast<int>(I), static_cast<int>(H), stream);
    } else {
      launch_moe_grouped_w4a4_bf16(q_codes_, q_scales_, q_gs_, nullptr, seg, 1, tokens, views, 0, gate_tmp_,
                                   static_cast<size_t>(I), static_cast<int>(I), static_cast<int>(H), stream);
      launch_moe_grouped_w4a4_bf16(q_codes_, q_scales_, q_gs_, nullptr, seg, 1, tokens, views, 1, up_tmp_,
                                   static_cast<size_t>(I), static_cast<int>(I), static_cast<int>(H), stream);
    }
    // The SwiGLU fused into the down projection's quantizer (the model's own
    // rounding; bitwise the swiglu kernel followed by the row quantizer).
    launch_swiglu_quantize_rows_nvfp4(gate_tmp_, up_tmp_, static_cast<size_t>(I), tokens, static_cast<int>(I), 0.f,
                                      q_codes_, q_scales_, q_gs_, stream, 0.f, m.fp4_input_gs + 1, /*qwen_rounding=*/true);
    if (boxes)
      launch_dense_w4a4_tma_bf16(*qmI, *m.down_fp4.tma, q_scales_, ssI, q_gs_, m.down_fp4.scales, m.down_fp4.global_scale,
                                 out, static_cast<size_t>(H), tokens, static_cast<int>(H), static_cast<int>(I), stream);
    else
      launch_moe_grouped_w4a4_bf16(q_codes_, q_scales_, q_gs_, nullptr, seg, 1, tokens, views, 2, out,
                                   static_cast<size_t>(H), static_cast<int>(H), static_cast<int>(I), stream);
    return;
  }
  // ---- mixed-release forms (the loader binds one form per layer) -------------
  if (m.gate_fp4.payload) {
    // The NVFP4 MLP: the fp4w prefill GEMM from 1024 rows, the production
    // ldmatrix kernel below (one chain everywhere at < 1024 rows;
    // tolerance-equal, never bitwise, across the boundary — a different
    // fp32 summation order). The 256-row tile amortizes weight decoding
    // over more rows and wins at the 1024-row boundary at TP 1/2/4.
    // Gate/up share one tile and retain both BF16 roundings before SwiGLU.
    // Smaller ragged tiles are not consistently faster. Decode keeps the
    // solo-vs-batch bitwise chain the gates pin.
    if (tokens <= 0) return;
    if (tokens >= 1024) {
      launch_fp4w_gate_up_swiglu_bf16(x, static_cast<size_t>(H), m.gate_fp4, m.up_fp4,
                                    gate_tmp_, tokens, static_cast<int>(I), H, stream);
      launch_fp4w_gemm_bf16(gate_tmp_, static_cast<size_t>(I), m.down_fp4, out, tokens, H,
                            static_cast<int>(I), stream);
      return;
    }
    // Resident: the boot-harvested per-layer table (graph-safe, no copies
    // in the walk). Streaming: re-stage the slot — the shared layer
    // allocation is rebuilt per load, so no table survives the walk (the
    // eager tests never capture, so the copy node is harmless there).
    const MoeExpertView* views = nullptr;
    if (fp4_view_layers_ > 0 && layer >= 0 && layer < fp4_view_layers_ &&
        m.gate_fp4.payload != nullptr) {
      views = fp4_views_ + static_cast<size_t>(layer) * 3;
    } else {
      MoeExpertView slot[3] = {MoeExpertView::of(m.gate_fp4), MoeExpertView::of(m.up_fp4),
                               MoeExpertView::of(m.down_fp4)};
      DGPP_CUDA_OK(cudaMemcpyAsync(fp4_views_, slot, sizeof(slot), cudaMemcpyHostToDevice,
                                   stream));
      views = fp4_views_;
    }
    const MoeSegment* seg = fp4_segs_ + (tokens - 1);
    launch_dense_mma_fp4_gate_up_bf16(x, static_cast<size_t>(H), seg, views, gate_tmp_, up_tmp_,
                                      tokens, static_cast<int>(I), H, stream,
                                      m.gate_fp4.scale_group);
    qwen35_swiglu_bf16(gate_tmp_, up_tmp_, gate_tmp_, static_cast<int64_t>(tokens) * I, stream);
    launch_dense_mma_fp4_prod_bf16(gate_tmp_, static_cast<size_t>(I), seg, views,
                                   /*which=*/2, out, tokens, H, static_cast<int>(I), stream,
                                   m.down_fp4.scale_group);
    return;
  }
  if (m.gate && m.gate_fp8.payload == nullptr) {
    // The BF16 draft layer (the mixed release's MTP MLP): the Lt matmuls.
    gw_.gemm->matmul(x, m.gate, gate_tmp_, tokens, static_cast<int>(I), H, DType::BF16,
                     GemmOut::BF16, static_cast<size_t>(H), gw_.ws, gw_.ws_bytes, stream);
    gw_.gemm->matmul(x, m.up, up_tmp_, tokens, static_cast<int>(I), H, DType::BF16, GemmOut::BF16,
                     static_cast<size_t>(H), gw_.ws, gw_.ws_bytes, stream);
    qwen35_swiglu_bf16(gate_tmp_, up_tmp_, gate_tmp_, static_cast<int64_t>(tokens) * I, stream);
    gw_.gemm->matmul(gate_tmp_, m.down, out, tokens, H, static_cast<int>(I), DType::BF16,
                     GemmOut::BF16, static_cast<size_t>(I), gw_.ws, gw_.ws_bytes, stream);
    return;
  }
  if (m.gate_fp8.payload && m.gate_fp8.scale_block_rows == 1) {
    // The channel-FP8 MLP (the mixed release's late layers): the streaming
    // MMA at decode rows (weights once, m-invariant chain), the dequant
    // bridge — reading the channel grid — above them, as the FP8 release's
    // block path below does for prefill.
    int rs = 0, cs = 0, rs2 = 0, cs2 = 0;
    if (!quant_scale_shifts(m.gate_fp8, rs, cs) || !quant_scale_shifts(m.down_fp8, rs2, cs2))
      throw std::runtime_error("qwen35 mlp: the channel grid is malformed");
    // The recipe's channel half (engine.prefill_activations = native): per-token
    // e4m3 rows, the three products on the e4m3 tensor cores with the scales
    // in the epilogue, the model's SwiGLU between; every prefill row.
    if (prefill && native_on_ && gw_.a8 && static_cast<size_t>(tokens) * static_cast<size_t>(I) <= gw_.a8_bytes &&
        H % 64 == 0 && I % 64 == 0) {
      const CUtensorMap* amH = gw_.a8_map_for(static_cast<int>(H));
      const CUtensorMap* amI = gw_.a8_map_for(static_cast<int>(I));
      const bool boxes = amH && amI && m.gate_fp8.tma && m.up_fp8.tma && m.down_fp8.tma;
      launch_fp8_quantize_tokens(x, static_cast<size_t>(H), tokens, static_cast<int>(H), gw_.a8, gw_.a8_scales, stream);
      if (boxes) {
        launch_fp8_gemm_channel_tma_bf16(*amH, *m.gate_fp8.tma, gw_.a8_scales, m.gate_fp8.scales, gate_tmp_, tokens,
                                         static_cast<int>(I), static_cast<int>(H), stream, static_cast<size_t>(I));
        launch_fp8_gemm_channel_tma_bf16(*amH, *m.up_fp8.tma, gw_.a8_scales, m.up_fp8.scales, up_tmp_, tokens,
                                         static_cast<int>(I), static_cast<int>(H), stream, static_cast<size_t>(I));
      } else {
        launch_fp8_gemm_channel_bf16(gw_.a8, gw_.a8_scales, m.gate_fp8.payload, m.gate_fp8.scales, gate_tmp_, tokens,
                                     static_cast<int>(I), static_cast<int>(H), stream, static_cast<size_t>(I));
        launch_fp8_gemm_channel_bf16(gw_.a8, gw_.a8_scales, m.up_fp8.payload, m.up_fp8.scales, up_tmp_, tokens,
                                     static_cast<int>(I), static_cast<int>(H), stream, static_cast<size_t>(I));
      }
      // The SwiGLU fused into the down projection's per-token quantizer.
      launch_fp8_swiglu_quantize_tokens(gate_tmp_, up_tmp_, static_cast<size_t>(I), tokens, static_cast<int>(I), gw_.a8,
                                        gw_.a8_scales, stream);
      if (boxes)
        launch_fp8_gemm_channel_tma_bf16(*amI, *m.down_fp8.tma, gw_.a8_scales, m.down_fp8.scales, out, tokens,
                                         static_cast<int>(H), static_cast<int>(I), stream, static_cast<size_t>(H));
      else
        launch_fp8_gemm_channel_bf16(gw_.a8, gw_.a8_scales, m.down_fp8.payload, m.down_fp8.scales, out, tokens,
                                     static_cast<int>(H), static_cast<int>(I), stream, static_cast<size_t>(H));
      return;
    }
    if (tokens <= 128) {
      launch_scale_gemm_grid_bf16(x, static_cast<size_t>(H), m.gate_fp8.payload, m.gate_fp8.scales,
                                  gate_tmp_, tokens, static_cast<int>(I), H, stream, 0, rs, cs,
                                  true, gw_.ws, gw_.ws_bytes);
      launch_scale_gemm_grid_bf16(x, static_cast<size_t>(H), m.up_fp8.payload, m.up_fp8.scales,
                                  up_tmp_, tokens, static_cast<int>(I), H, stream, 0, rs, cs,
                                  true, gw_.ws, gw_.ws_bytes);
      qwen35_swiglu_bf16(gate_tmp_, up_tmp_, gate_tmp_, static_cast<int64_t>(tokens) * I, stream);
      launch_scale_gemm_grid_bf16(gate_tmp_, static_cast<size_t>(I), m.down_fp8.payload,
                                  m.down_fp8.scales, out, tokens, H, static_cast<int>(I), stream,
                                  0, rs2, cs2, true, gw_.ws, gw_.ws_bytes);
      return;
    }
    const size_t up_bytes = static_cast<size_t>(I) * static_cast<size_t>(H) * 2;
    const size_t down_bytes = static_cast<size_t>(H) * static_cast<size_t>(I) * 2;
    if (up_bytes <= gw_.dequant_bytes && down_bytes <= gw_.dequant_bytes) {
      launch_fp8_dequant_blocks(m.gate_fp8.payload, m.gate_fp8.scales, gw_.dequant, I, H, stream,
                                rs, cs);
      gw_.gemm->matmul(x, gw_.dequant, gate_tmp_, tokens, static_cast<int>(I), H, DType::BF16,
                       GemmOut::BF16, static_cast<size_t>(H), gw_.ws, gw_.ws_bytes, stream);
      launch_fp8_dequant_blocks(m.up_fp8.payload, m.up_fp8.scales, gw_.dequant, I, H, stream, rs,
                                cs);
      gw_.gemm->matmul(x, gw_.dequant, up_tmp_, tokens, static_cast<int>(I), H, DType::BF16,
                       GemmOut::BF16, static_cast<size_t>(H), gw_.ws, gw_.ws_bytes, stream);
      qwen35_swiglu_bf16(gate_tmp_, up_tmp_, gate_tmp_, static_cast<int64_t>(tokens) * I, stream);
      launch_fp8_dequant_blocks(m.down_fp8.payload, m.down_fp8.scales, gw_.dequant, H, I, stream,
                                rs2, cs2);
      gw_.gemm->matmul(gate_tmp_, gw_.dequant, out, tokens, H, static_cast<int>(I), DType::BF16,
                       GemmOut::BF16, static_cast<size_t>(I), gw_.ws, gw_.ws_bytes, stream);
      return;
    }
    throw std::runtime_error("qwen35 mlp: the channel prefill needs the dequant bridge");
  }
  // Prefill-shaped products run the dequant bridge + cuBLASLt BF16 (the same
  // lowering gemm_dense uses for the attention projections): nsys showed the
  // streaming tile kernel owning ~70% of a 2K prefill at ~24 TFLOP/s, while
  // the bridge scratch (178MB) sat unused by this direct caller.
  // The FP8 release's recipe (engine.prefill_activations = native): the
  // blockwise product (per-token 1 x 128 e4m3 groups, the 128 x 128 weight
  // grid, fp32 promotion per group) on every prefill row, the bridge's
  // SwiGLU between.
  if (prefill && native_on_ && gw_.a8 && m.gate_fp8.payload && m.gate_fp8.scale_block_rows == 128 &&
      m.gate_fp8.scale_block_cols == 128 && static_cast<size_t>(tokens) * static_cast<size_t>(I) <= gw_.a8_bytes &&
      H % 128 == 0 && I % 128 == 0) {
    launch_fp8_quantize_rows(x, static_cast<size_t>(H), tokens, static_cast<int>(H), gw_.a8, gw_.a8_scales, stream);
    launch_fp8_gemm_bf16(gw_.a8, gw_.a8_scales, m.gate_fp8.payload, m.gate_fp8.scales, 128, gate_tmp_, tokens,
                         static_cast<int>(I), static_cast<int>(H), stream, static_cast<size_t>(I));
    launch_fp8_gemm_bf16(gw_.a8, gw_.a8_scales, m.up_fp8.payload, m.up_fp8.scales, 128, up_tmp_, tokens,
                         static_cast<int>(I), static_cast<int>(H), stream, static_cast<size_t>(I));
    qwen35_swiglu_bf16(gate_tmp_, up_tmp_, gate_tmp_, static_cast<int64_t>(tokens) * I, stream);
    launch_fp8_quantize_rows(gate_tmp_, static_cast<size_t>(I), tokens, static_cast<int>(I), gw_.a8, gw_.a8_scales,
                             stream);
    launch_fp8_gemm_bf16(gw_.a8, gw_.a8_scales, m.down_fp8.payload, m.down_fp8.scales, 128, out, tokens,
                         static_cast<int>(H), static_cast<int>(I), stream, static_cast<size_t>(H));
    return;
  }
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

// The channel matrices' TMA maps (engine.prefill_activations = native, the
// mixed release): one map over every channel-FP8 matrix a prefill-shaped
// product serves (the projections, the late-layer MLPs), the matrix's `tma`
// pointing at it. The vector is reserved once so the addresses stay put;
// the loader's resident slot is const to its readers, the map pointer the
// one field written after the build.
// The NVFP4 MLP matrices' TMA maps (the dense W4A4 TMA form): one over every
// gate / up / down of the mixed release's NVFP4 layers, the matrix's `tma`
// pointing at it; the vector is reserved once so the addresses stay put.
void Qwen35Model::build_native_fp4_maps() {
  std::vector<GlmFp4Matrix*> items;
  const auto take = [&](GlmFp4Matrix& m) {
    if (m.payload && m.scales && m.global_scale && !m.mxfp4() && dense_w4a4_tma_supported(static_cast<int>(m.cols)))
      items.push_back(&m);
  };
  for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
    Qwen35LayerResident& r = const_cast<Qwen35LayerResident&>(loader_.load_layer(l));
    take(r.mlp.gate_fp4);
    take(r.mlp.up_fp4);
    take(r.mlp.down_fp4);
  }
  native_fp4_tmaps_.clear();
  native_fp4_tmaps_.reserve(items.size());
  for (GlmFp4Matrix* m : items) {
    native_fp4_tmaps_.push_back(nvfp4_code_tma_map(m->payload, m->rows, m->cols, 256));
    m->tma = &native_fp4_tmaps_.back();
  }
  DGPP_LOG_INFO("Qwen35Model: dense W4A4 TMA form: maps over {} NVFP4 matrices", items.size());
}

void Qwen35Model::build_native_maps() {
  std::vector<GlmQuantMatrix*> items;
  const auto take = [&](GlmQuantMatrix& m) {
    if (m.payload && m.scales && m.scale_block_rows == 1 && fp8_channel_tma_supported(static_cast<int>(m.cols)))
      items.push_back(&m);
  };
  for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
    Qwen35LayerResident& r = const_cast<Qwen35LayerResident&>(loader_.load_layer(l));
    if (r.kind == Qwen35LayerKind::Gdn) {
      take(r.gdn.in_proj_qkv_fp8);
      take(r.gdn.in_proj_z_fp8);
      take(r.gdn.out_proj_fp8);
    } else {
      take(r.full.q_proj_fp8);
      take(r.full.k_proj_fp8);
      take(r.full.v_proj_fp8);
      take(r.full.o_proj_fp8);
    }
    take(r.mlp.gate_fp8);
    take(r.mlp.up_fp8);
    take(r.mlp.down_fp8);
  }
  native_tmaps_.clear();
  native_tmaps_.reserve(items.size());
  for (GlmQuantMatrix* m : items) {
    native_tmaps_.push_back(fp8_channel_tma_map(m->payload, m->rows, m->cols, 256));
    m->tma = &native_tmaps_.back();
  }
  DGPP_LOG_INFO("Qwen35Model: native channel form: TMA maps over {} matrices", items.size());
}

size_t Qwen35Model::kv_block_bytes_static(const Qwen35TextConfig& cfg, int rank, int world, bool mtp,
                                          const std::string& dflash2_dir, LatentFormat kv_format) {
  // The pool shape plan_memory builds, at one and at two blocks: the
  // difference is one block's bytes across every token-linear plane (the
  // Full layers' K/V, the drafter's planes); the block table is subtracted.
  std::unique_ptr<DFlash2Config> dfcfg;
  std::unique_ptr<DFlashConfig> df1cfg;
  if (!dflash2_dir.empty()) {
    if (drafter_architecture(dflash2_dir) == "DFlashDraftModel")
      df1cfg = std::make_unique<DFlashConfig>(DFlashConfig::from_json_file(dflash2_dir + "/config.json"));
    else
      dfcfg = std::make_unique<DFlash2Config>(DFlash2Config::from_json_file(dflash2_dir + "/config.json"));
  }
  const LoaderHeadSharding head = world > 1 ? LoaderHeadSharding::VocabSharded : LoaderHeadSharding::Full;
  const auto bytes = [&](int64_t tokens) {
    Qwen35KvPoolShape shape;
    shape.layers = 0;
    for (Qwen35LayerKind k : cfg.layers)
      if (k != Qwen35LayerKind::Gdn) ++shape.layers;
    if (mtp) ++shape.layers;
    shape.kv_heads = Qwen35LocalGeometry::from_config(cfg, rank, world, head).local_kv_heads;
    shape.dim = cfg.head_dim;
    shape.block_tokens = kv_block_tokens_static();
    shape.max_requests = 1;
    shape.token_slots = tokens;
    shape.fp8 = kv_format_is_fp8(kv_format, cfg, mtp);
    if (dfcfg) {
      shape.draft_layers = dfcfg->num_hidden_layers;
      shape.draft_kv_heads = dfcfg->local_kv_heads(world);
      shape.draft_dim = dfcfg->head_dim;
    }
    if (df1cfg) {
      shape.draft_layers = df1cfg->num_hidden_layers;
      shape.draft_kv_heads = df1cfg->local_kv_heads(world);
      shape.draft_dim = df1cfg->head_dim;
    }
    return Qwen35KvPool::cache_bytes(shape) - PagedBlockTable::table_bytes(1, tokens / shape.block_tokens);
  };
  const int64_t bt = kv_block_tokens_static();
  return bytes(2 * bt) - bytes(bt);
}

MemoryPlan Qwen35Model::plan_memory(const Qwen35TextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                    int rank, int world, LoaderResidency residency, int max_requests,
                                    bool mtp, int decode_rows, const std::string& dflash2_dir,
                                    LatentFormat kv_format) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests out of range");
  if (mtp && cfg.mtp_layer() < 0)
    throw std::invalid_argument("plan_memory: the config has no draft layer (mtp)");
  if ((cfg.quant_kind == Qwen35QuantKind::Nvfp4Mixed ||
       cfg.quant_kind == Qwen35QuantKind::Nvfp4Moe) &&
      (prefill_fp8_per_tensor_ || dense_weights_fp8_))
    throw std::invalid_argument(
        "plan_memory: the NVFP4 release serves the checkpoint's own formats; "
        "engine.prefill_fp8_per_tensor and engine.dense_weights = fp8 are the FP8 "
        "release's levers (unset them)");
  if (cfg.is_moe && mtp && Qwen35LayerStream::mtp_expert_format() != "bf16")
    throw std::invalid_argument("plan_memory: the MoE draft needs engine.mtp_expert_format=bf16");
  std::unique_ptr<DFlash2Config> dfcfg;
  std::unique_ptr<DFlashConfig> df1cfg;
  if (!dflash2_dir.empty()) {
    if (mtp)
      throw std::invalid_argument("plan_memory: dflash2 replaces the MTP draft; enable one or the other");
    const std::string darch = drafter_architecture(dflash2_dir);
    if (darch == "DFlashDraftModel") {
      df1cfg = std::make_unique<DFlashConfig>(DFlashConfig::from_json_file(dflash2_dir + "/config.json"));
      df1cfg->validate_against(cfg);
    } else {
      dfcfg = std::make_unique<DFlash2Config>(DFlash2Config::from_json_file(dflash2_dir + "/config.json"));
      dfcfg->validate_against(DFlash2TargetShape{cfg.hidden_size, cfg.vocab_size,
                                                 cfg.num_hidden_layers, cfg.max_position_embeddings});
    }
  }
  const LoaderHeadSharding head = world > 1 ? LoaderHeadSharding::VocabSharded : LoaderHeadSharding::Full;
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kv_block_tokens_static() - 1) /
       kv_block_tokens_static()) *
      kv_block_tokens_static();
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.context_limit());
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  // Dense gate/up tmps only; the MoE chain carries its own scratch below.
  const size_t I = cfg.is_moe ? 0 : static_cast<size_t>(cfg.intermediate_size);
  if (mtp || !dflash2_dir.empty())
    plan.add("draft head screen (2-bit rotated head copy, scratch)",
             DraftHead::plan_bytes(Qwen35LocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count,
                                   cfg.hidden_size, std::max({kDecodeRows, decode_rows, max_requests})));
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
  shape.fp8 = kv_format_is_fp8(kv_format, cfg, mtp);
  shape.fp8_attention = fp8_attention_;
  if (dfcfg) {
    shape.draft_layers = dfcfg->num_hidden_layers;
    shape.draft_kv_heads = dfcfg->local_kv_heads(world);
    shape.draft_dim = dfcfg->head_dim;
  }
  if (df1cfg) {
    shape.draft_layers = df1cfg->num_hidden_layers;
    shape.draft_kv_heads = df1cfg->local_kv_heads(world);
    shape.draft_dim = df1cfg->head_dim;
  }
  plan.add("kv pool", Qwen35KvPool::cache_bytes(shape));
  // Layer scratch at max_tokens rows (Full + GDN worst case).
  const QwenTextConfig qc = qwen35_text_adapter(cfg);
  const int lh = cfg.num_attention_heads, lkv = cfg.num_key_value_heads;
  const int lk = cfg.gdn_key_heads, lv = cfg.gdn_value_heads;
  plan.add("layer scratch", std::max(QwenFullAttnLayer::scratch_bytes(qc, lh, lkv, max_tokens),
                                    QwenGdnLayer::scratch_bytes(qc, lk, lv, max_tokens)));
  if (cfg.is_moe) {
    // Routed NVFP4 chain + BF16 shared tail (QwenMoeLayer): decode slots at
    // the row ceiling, with a graph route table for each resident layer.
    const Qwen35LocalGeometry geo = Qwen35LocalGeometry::from_config(cfg, rank, world, head);
    const GlmMoeConfig moe_cfg = QwenMoeLayer::routed_config(
        cfg.hidden_size, static_cast<int>(geo.local_moe_inter), cfg.num_experts,
        cfg.num_experts_per_tok, cfg.norm_topk_prob);
    size_t moe_pinned = 0;
    const int moe_tables = residency == LoaderResidency::Resident ? cfg.num_hidden_layers + (mtp ? 1 : 0) : 0;
    const size_t moe_dev = QwenMoeLayer::scratch_bytes(
        moe_cfg, geo.local_shared_inter, max_tokens, &moe_pinned, decode_rows, moe_tables);
    plan.add("moe scratch (routed slots, shared expert)", moe_dev, moe_pinned);
  }
  // Activations: resid/x/attn/mlp [M,H] + gate/up tmps [M,I].
  plan.add("activations", 4 * M * H * 2 + 2 * M * I * 2);
  // The NVFP4 dense path's tables (production ldmatrix kernel): one
  // {0, m, 0} segment per row count plus the view tables (per-layer on
  // resident mixed-release stacks, one 3-entry staging slot elsewhere —
  // the ctor's split).
  plan.add(
      "nvfp4 dense segment/view tables",
      M * sizeof(MoeSegment) +
          (residency == LoaderResidency::Resident && cfg.quant_kind == Qwen35QuantKind::Nvfp4Mixed
               ? (static_cast<size_t>(cfg.num_hidden_layers) + (mtp ? 1 : 0)) * 3 *
                     sizeof(MoeExpertView)
               : 3 * sizeof(MoeExpertView)));
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
  // The native activation recipe's NVFP4 scratch (engine.prefill_activations
  // = native, the mixed release): one quantized row set at max_tokens rows.
  if (native_activations_ && cfg.quant_kind == Qwen35QuantKind::Nvfp4Mixed)
    plan.add("native activation scratch (NVFP4 rows at max_tokens)",
             M * I / 2 + M * nvfp4_act_scale_stride(static_cast<int>(I)) + M * sizeof(float));
  if (native_activations_)
    plan.add("native activation scratch (e4m3 rows at max_tokens)", M * I + M * (I / 128 + 1) * sizeof(float));
  // Blockwise-FP8 lm head (engine.dense_weights = fp8, Resident only): half
  // the bytes per decode row.
  if (dense_weights_fp8_ && residency == LoaderResidency::Resident) {
    const size_t Vv = static_cast<size_t>(
        Qwen35LocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count);
    const size_t Hh = static_cast<size_t>(cfg.hidden_size);
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
    const bool head_bf16 = cfg.quant_kind == Qwen35QuantKind::Fp8Block ||
                           cfg.tensor_quant("lm_head") == Qwen35TensorQuant::Bf16;
    if (head_bf16 && !(dense_weights_fp8_ && residency == LoaderResidency::Resident))
      packed += Bf12Companions::planned_bytes(cfg.vocab_size, H);  // the head serves in bf16
    plan.add("bf16 decode packing (12-bit companions; the bf16 bytes stay)", packed);
  }
  if (dfcfg) {
    // The shared drafter's plan lines (models/dflash2/drafter.hpp): the
    // weights, the feature path and the block scratch exactly as the
    // component's init allocates them.
    std::vector<std::pair<std::string, size_t>> dflash2_entries;
    const auto vocab_count = Qwen35LocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count;
    DFlash2Drafter::plan_entries(*dfcfg, world, decode_rows, max_tokens, max_requests, vocab_count,
                                 dflash2_fp8_, &dflash2_entries);
    for (const auto& e : dflash2_entries) plan.add(e.first, e.second);
  }
  if (df1cfg) {
    if (!df1cfg->tp_divisible(world))
      throw std::invalid_argument("plan_memory: the DFlash v1 drafter's heads / kv heads / MLP rows must divide across the world");
    plan.add("dflash v1 drafter weights (this rank's slices)",
             dflash_weights_bytes(*df1cfg, world));
    const size_t QR1 = static_cast<size_t>(df1cfg->query_rows());
    const size_t dI1 = static_cast<size_t>(df1cfg->local_intermediate(world));
    const size_t dQW1 = static_cast<size_t>(df1cfg->local_q_row(world)),
                 dKV1 = static_cast<size_t>(df1cfg->local_kv_row(world));
    plan.add("dflash v1 feature path",
             2 * M * H * 4 + df1cfg->head_dim / 2 * 4);  // acc + tap F32, rope table
    plan.add("dflash v1 block scratch (reused per request)",
             QR1 * (7 * H * 2 + (dQW1 + 2 * dKV1) * 2 + dQW1 * 2 + 2 * dI1 * 2) +
                 dflash2_block_attn_partials_bytes(QR1, df1cfg->local_heads(world)));
    plan.add("dflash v1 candidate buffers",
             QR1 * static_cast<size_t>(cfg.vocab_size) * 4 +
                 dflash2_topk_ws_bytes(cfg.vocab_size, QR1, 16) + QR1 * 16 * (4 + 4) +
                 QR1 * 16 + (world > 1 ? dflash2_topk_table_elems(1, 16, world) * 2 : 0));
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
  if (cfg.vision) {
    plan.add("vision tower weights (bf16, replicated)", cfg.vision->weight_bytes());
    plan.add("vision workspace (patches, attention tiles, staged image rows)",
             cfg.vision->workspace_bytes());
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
    // The MoE decode fast path reads a prepared per-layer route table under
    // capture (no host round-trip inside the graph); eager passes -1.
    if (cfg_.is_moe) moe_->prepare_graph_table(layer, stream_);
  }
  if (mtp_) {
    const Qwen35LayerResident& r = loader_.load_layer(cfg_.mtp_layer());
    build_layer_objects(r);
    // The draft MoE's route table takes the slot after the main stack's.
    if (cfg_.is_moe) moe_->prepare_graph_table(cfg_.num_hidden_layers, stream_);
  }
}

// The MTP draft block: embed + hidden fusion through the fused fc [H, 2H],
// one Full draft layer + dense SwiGLU MLP over the fused rows, mtp.norm,
// then the shared lm head. Prefill rows read the main chunk's final hidden
// in place (h_); decode rows gather theirs from the slots' windows by
// position. head_rows == 0 fills the draft K/V only (prefill cache fill).
void Qwen35Model::mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T,
                               bool decode_row, bool capture, int head_rows, int batch_requests, int hidden_row) {
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
  const uint16_t* hin = prefill_hidden_span(h_, H, hidden_row, T);
  if (decode_row) {
    gather_draft_hidden(d_req, d_pos, mtp_hin_, T);
    hin = mtp_hin_;
  }
  embed_gather_bf16(globals_.embed, tokens, mtp_e_, T, H, stream_);
  // The draft's row at position p embeds token p + 1, so its image window
  // is the main walk's shifted by one.
  if (const auto* images = images_for_req(req))
    apply_image_embeddings(mtp_e_, first_pos, T, 1, images);
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
  if (cfg_.is_moe)
    moe_mlp(x_, mo, T, r.moe, stream_, capture ? cfg_.num_hidden_layers : -1,
            /*decode=*/decode_row);
  else
    dense_mlp(x_, mo, T, r.mlp, stream_, cfg_.num_hidden_layers, first_pos > 0 && !decode_row, !decode_row);
  fold(mo, H);
  add_inplace_bf16(mtp_r_, mo, static_cast<size_t>(T) * H, stream_);
  if (head_rows == 0) return;  // prefill rows fill the cache; no head

  // ---- head: the draft distribution over the last head_rows rows --------
  // mtp.norm runs over every row so draft_hidden_rows() stays row-indexed
  // for the chain; the head reads the last head_rows normed rows.
  qwen_rmsnorm_bf16(mtp_r_, globals_.mtp_norm, mtp_h_, T, H, eps, stream_);
  const uint16_t* head_in = mtp_h_ + static_cast<size_t>(T - head_rows) * H;
  const int64_t V = lm_vocab_count_;
  head_gemv(head_in, logits_, head_rows, stream_, /*draft=*/decode_row);
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
  const bool mixed = run.decode_spans > 0;
  const int decode_rows = run.decode_rows();
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
  // Image rows replace the embedding at their prompt positions; the tower
  // already projected them (see apply_image_embeddings). Only its own
  // slot's prefill walk carries a request's images; any other walk with a
  // live borrow is a lifecycle bug and fails loudly, never silently.
  if (const auto* images = images_for_req(run.req)) {
    if (run.decode || run.batch_requests || run.num_spans)
      throw std::logic_error("Qwen3.5: image prefill reached a non-prefill walk");
    apply_image_embeddings(resid_, run.pos0, T, 0, images);
  }
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
  const bool snapshots = (run.decode || mixed) && run.snapshots;
  if (gdn_replay_ && !run.decode) {
    if (run.num_spans > 0) {
      for (int sp = run.decode_spans; sp < run.num_spans; ++sp) materialize_gdn(run.span_reqs[sp]);
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
        if (mixed) {
          configure_gemm_rows(decode_rows, true);
          KdaRequestRows requests;
          requests.request_ids = d_req;
          requests.positions = d_pos;
          requests.spans = d_spans;
          requests.num_requests = run.decode_spans;
          if (gdn_replay_) {
            replay = gdn_replay_view(0, gdn_ord, true);
            for (int sp = 0; sp < run.decode_spans; ++sp)
              gdn_replay_rows_used_ = std::max(gdn_replay_rows_used_, run.span_lens[sp]);
          }
          gdn_->enqueue_rows(x_, gdn_rec(0, gdn_ord), static_cast<int64_t>(num_gdn_) * rec_elems_,
                              gdn_conv(0, gdn_ord), static_cast<int64_t>(num_gdn_) * conv_elems_,
                              ao, decode_rows, requests, stream_, rec_snap, conv_snap, replay);
          configure_gemm_rows(T, false);
          row0 = decode_rows;
        }
        for (int sp = run.decode_spans; sp < run.num_spans; ++sp) {
          const int len = run.span_lens[sp], sreq = run.span_reqs[sp];
          // Large readers share the projection and weight conversion;
          // each span still scans only its own recurrent and conv state.
          // Small tails retain their original projection recipe, and a
          // per-tensor scale cannot be shared across readers.
          if (!pt_enabled_ && len > 128 && sp + 1 < run.num_spans && run.span_lens[sp + 1] > 128) {
            std::vector<QwenGdnLayer::Span> spans;
            int count = 0;
            do {
              QwenGdnLayer::Span span;
              span.rows = run.span_lens[sp];
              span.recurrent = gdn_rec(run.span_reqs[sp], gdn_ord);
              span.conv = gdn_conv(run.span_reqs[sp], gdn_ord);
              count += span.rows;
              spans.push_back(span);
              ++sp;
            } while (sp < run.num_spans && run.span_lens[sp] > 128);
            --sp;
            gdn_->enqueue_spans(x_ + static_cast<size_t>(row0) * H, ao + static_cast<size_t>(row0) * H,
                                 count, spans, stream_);
            row0 += count;
            continue;
          }
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
        if (mixed) {
          configure_gemm_rows(decode_rows, true);
          auto prefix = qrows;
          prefix.decode = true;
          prefix.num_requests = run.decode_spans;
          full_->enqueue(x_, decode_rows, prefix, cache, ao, stream_);
          configure_gemm_rows(T, false);
          row0 = decode_rows;
        }
        for (int sp = run.decode_spans; sp < run.num_spans; ++sp) {
          const int len = run.span_lens[sp];
          if (!pt_enabled_ && len > 128 && sp + 1 < run.num_spans && run.span_lens[sp + 1] > 128) {
            std::vector<QwenFullAttnLayer::Span> spans;
            int count = 0;
            do {
              QwenFullAttnLayer::Span span;
              span.rows = run.span_lens[sp];
              span.map = qrows;
              span.map.req_ids = d_req + row0 + count;
              span.map.pos = d_pos + row0 + count;
              span.map.request = run.span_reqs[sp];
              span.map.pos0 = run.span_pos0[sp];
              count += span.rows;
              spans.push_back(span);
              ++sp;
            } while (sp < run.num_spans && run.span_lens[sp] > 128);
            --sp;
            full_->enqueue_spans(x_ + static_cast<size_t>(row0) * H, count, spans.front().map, spans,
                                   cache, ao + static_cast<size_t>(row0) * H, stream_);
            row0 += count;
            continue;
          }
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
    uint16_t* mo = stage(mlp_out_, H);
    if (cfg_.is_moe)
      moe_mlp(x_, mo, T, r.moe, stream_, run.capture ? layer : -1, run.decode);
    else if (run.num_spans > 0) {
      // Retain the decoder's projection chain. Small reader tails also
      // retain their streaming-MMA recipe when a large neighbor uses the
      // dequant bridge. Per-tensor scales remain request-owned.
      const int prefill_from = r.mlp.gate_fp4.payload ? 1024 : 129;
      run.for_each_kernel_group([&](int count, int next) {
        const bool prefill = count >= prefill_from;
        return !pt_enabled_ && (next >= prefill_from) == prefill &&
               (prefill || count + next < prefill_from);
      }, [&](int row0, int count, bool decode, int first) {
        configure_gemm_rows(count, decode);
        dense_mlp(x_ + static_cast<size_t>(row0) * H, mo + static_cast<size_t>(row0) * H,
                  count, r.mlp, stream_, layer, !decode && run.span_pos0[first] > 0, !decode);
      });
      configure_gemm_rows(T, run.decode);
    } else {
      dense_mlp(x_, mo, T, r.mlp, stream_, layer, !run.decode && run.pos0 > 0, !run.decode);
    }
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
    // into fp32 instead of one, tolerance-identical). v2's accumulator is
    // the component's; v1 keeps its own (one drafter at a time).
    const int tap = dflash2_ ? df_.tap(layer)
                             : (dflash1_ && layer < (int)df1_tap_.size() ? df1_tap_[layer] : -1);
    if ((dflash2_ || dflash1_) && tap >= 0) {
      const uint16_t* fc_t =
          (dflash2_ ? df_.weights().fc : df1w_.fc) + static_cast<size_t>(tap) * H * H;
      float* t32 = dflash2_ ? df_.tap_gemm_out() : df1_t32_;
      int row0 = 0;
      if (mixed) {
        configure_gemm_rows(decode_rows, true);
        gemm_.matmul(resid_, fc_t, t32, decode_rows, H, H, DType::BF16, GemmOut::F32,
                     static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
        row0 = decode_rows;
        configure_gemm_rows(T - row0, false);
      }
      gemm_.matmul(resid_ + static_cast<size_t>(row0) * H, fc_t,
                   t32 + static_cast<size_t>(row0) * H, T - row0, H, H, DType::BF16, GemmOut::F32,
                   static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
      configure_gemm_rows(T, run.decode);
      dflash2_acc_f32(dflash2_ ? df_.acc() : df1_acc_, t32, static_cast<int64_t>(T) * H,
                      tap == 0 ? 0 : 1, stream_);
    }
  }
  // Final norm + lm head.
  qwen_rmsnorm_bf16(resid_, globals_.final_norm, h_, T, H, eps, stream_);
  const bool packed_logits = packed_prefill_logits(run);
  if (packed_logits) {
    project_head_rows(run, [&](int input, int output, int count) {
      configure_gemm_rows(count, mixed && input == 0);
      head_gemv(h_ + static_cast<size_t>(input) * H,
                 logits_ + static_cast<size_t>(output) * lm_vocab_count_, count, stream_);
    });
    configure_gemm_rows(T, run.decode);
  } else {
    const int first = (run.decode || run.all_rows || run.num_spans > 0) ? 0 : T - 1;
    head_gemv(h_ + static_cast<size_t>(first) * H, logits_ + static_cast<size_t>(first) * lm_vocab_count_,
               T - first, stream_);
  }
  // The draft block's input: the last rows' final hidden into the slots'
  // windows by position (the last window rows of a prefill chunk, every
  // decode row — distinct slots within one launch).
  store_run_draft_hidden(run, in, h_);
  // Every run's rows become the drafter's context K/V at their positions
  // (rejected verify rows included: they land past the committed end and
  // the next draft's position mask hides them — no rollback). Inside a
  // captured verify this is a recorded node, NOT a skip: the eager
  // redrafts between replays read the planes, so a replay that omitted
  // the feed would draft over stale context (2026-10-02: the c4 graph's
  // acceptance decay was exactly this — the capture kept the verify
  // exact but starved the drafter).
  if (dflash2_) df_.store_features(T, d_req, d_pos);
  if (dflash1_) df1_store_features(T, d_req, d_pos);
  if (run.decode) prefetch_.join(stream_);  // every forked prefetch back on the main stream
  out = finish_run(run, std::move(out), packed_logits);
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
  const Qwen35DenseMlpResident& m = r.mlp;
  const bool small_fp4 =
      m.gate_fp4.payload &&
      dense_fp4_decode_shape(walk_rows_, m.gate_fp4.rows, m.gate_fp4.cols, m.gate_fp4.scale_group);
  // The paired small-row kernel benefits from a warm gate. Reading into
  // up as well competes with the live attention weights and the gate's
  // first reads. Keep the existing window for the other matrix forms.
  const size_t budget = small_fp4 ? std::min(prefetch_window_bytes_,
                                             m.gate_fp4.payload_bytes() + m.gate_fp4.scale_bytes())
                                  : prefetch_window_bytes_;
  prefetch_.open_window(stream_, budget, prefetch_.boundary_rate());
  if (m.gate_fp4.payload) {
    if (small_fp4) {
      prefetch_add(m.gate_fp4.payload, m.gate_fp4.payload_bytes());
      prefetch_add(m.gate_fp4.scales, m.gate_fp4.scale_bytes());
    }
  } else if (m.gate_fp8.payload) {
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


bool Qwen35Model::dflash2_draft(int req, int64_t bonus, std::vector<int32_t>* drafts) {
  if (!dflash2_ && !dflash1_) throw std::logic_error("dflash2_draft: no drafter loaded");
  if (dflash1_) return dflash1_draft(req, bonus, drafts);
  return df_.draft(req, bonus, drafts);
}

// ---- the DFlash v1 drafter --------------------------------------------------------

void Qwen35Model::df1_store_features(int T, const int32_t* d_req, const int64_t* d_pos) {
  // The draft's KV context from the fused tap features: hidden_norm, each
  // layer's k|v projections, roped, appended at the run's positions (the
  // v2 store_features minus the conv path).
  const int H = cfg_.hidden_size;
  const int L = df1cfg_.num_hidden_layers;
  const int KV = df1cfg_.local_kv_row(world_);
  const int KVH = df1cfg_.local_kv_heads(world_);
  const int HD = df1cfg_.head_dim;
  const int32_t* tables = pool_.blocks().device_tables();
  const int bpr = static_cast<int>(pool_.total_blocks());
  const int bt = kv_block_tokens_static();
  // The feature pass shares the draft block's query-row scratch.
  const int QR = df1cfg_.query_rows();
  for (int off = 0; off < T; off += QR) {
    const int rows = std::min(QR, T - off);
    dflash2_norm_f32_bf16(df1_acc_ + static_cast<size_t>(off) * H, df1w_.hidden_norm, df1_x_,
                          rows, H, df1cfg_.rms_norm_eps, stream_);
    for (int l = 0; l < L; ++l) {
      gemm_.matmul(df1_x_, df1w_.layers[l].k, df1_k_, rows, KV, H, DType::BF16, GemmOut::BF16,
                   static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
      gemm_.matmul(df1_x_, df1w_.layers[l].v, df1_v_, rows, KV, H, DType::BF16, GemmOut::BF16,
                   static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
      dflash2_norm_rope_bf16(df1_k_, KV, df1w_.layers[l].k_norm, d_pos + off, df1_inv_freq_,
                             df1_k_, KV, rows, KVH, HD, df1cfg_.rms_norm_eps, stream_);
      const QwenFullAttnCache plane = pool_.view(num_full_ + l);
      qsa_kv_append(df1_k_, KV, df1_v_, KV, d_req + off, d_pos + off, rows, tables, bpr, bt,
                    KVH, HD, plane.k_cache, plane.v_cache, nullptr, nullptr, stream_);
    }
  }
}

void Qwen35Model::df1_block_layers(int req, cudaStream_t stream, bool capture) {
  const int H = cfg_.hidden_size;
  const int QR = df1cfg_.query_rows();
  const int L = df1cfg_.num_hidden_layers;
  const int QW = df1cfg_.local_q_row(world_), KV = df1cfg_.local_kv_row(world_);
  const int KVH = df1cfg_.local_kv_heads(world_), NH = df1cfg_.local_heads(world_);
  const int HD = df1cfg_.head_dim;
  const int I = df1cfg_.local_intermediate(world_);
  const float eps = df1cfg_.rms_norm_eps;
  const int32_t* tables = pool_.blocks().device_tables();
  const int bpr = static_cast<int>(pool_.total_blocks());
  const int bt = kv_block_tokens_static();
  const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
  const int32_t* table = tables + static_cast<size_t>(req) * pool_.total_blocks();
  const auto project = [&](const uint16_t* x, const uint16_t* weight, int width) {
    // The graph recorder accepts only its stable base pointer, and one
    // latency slot need not fit the whole 16-row block. Project and fold
    // bounded chunks, copying each result out before reusing that pointer.
    const int chunk = boundary_ && world_ > 1 ? std::min(8, max_decode_rows_) : QR;
    for (int off = 0; off < QR; off += chunk) {
      const int rows = std::min(chunk, QR - off);
      uint16_t* dst = df1_o_ + static_cast<size_t>(off) * H;
      uint16_t* staged = boundary_ && world_ > 1 ? boundary_->stage(rows, H) : nullptr;
      if (capture && boundary_ && world_ > 1 && staged == nullptr)
        throw std::runtime_error("dflash v1 draft: projection exceeds the staged reduction buffer");
      uint16_t* out = staged ? staged : dst;
      gemm_.matmul(x + static_cast<size_t>(off) * width, weight, out, rows, H, width,
                   DType::BF16, GemmOut::BF16, width, gemm_ws_, gemm_ws_bytes_, stream);
      if (boundary_ && world_ > 1) {
        if (!capture && !boundary_->stream_ordered()) DGPP_CUDA_OK(cudaStreamSynchronize(stream));
        boundary_->reduce(out, rows, H);
      }
      if (out != dst) copy_rows_bf16(dst, out, rows, H, stream);
    }
  };
  for (int l = 0; l < L; ++l) {
    const DFlashLayerWeights& w = df1w_.layers[l];
    dflash2_rmsnorm_bf16(df1_resid_, w.input_norm, df1_x_, QR, H, eps, stream);
    // Split q/k/v projections into the stacked [q | k | v] rows, roped in
    // place (v1 ships separate matrices, not the v2 fused qkv).
    gemm_.matmul(df1_x_, w.q, df1_q_, QR, QW, H, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
    gemm_.matmul(df1_x_, w.k, df1_k_, QR, KV, H, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
    gemm_.matmul(df1_x_, w.v, df1_v_, QR, KV, H, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
    dflash2_norm_rope_bf16(df1_q_, QW, w.q_norm, df1_pos_, df1_inv_freq_, df1_q_, QW, QR,
                           NH, HD, eps, stream);
    dflash2_norm_rope_bf16(df1_k_, KV, w.k_norm, df1_pos_, df1_inv_freq_, df1_k_, KV, QR,
                           KVH, HD, eps, stream);
    const QwenFullAttnCache plane = pool_.view(num_full_ + l);
    // The block rows' own k/v join the planes (the prefix context arrived
    // via df1_store_features); single slot, all rows request zero's table.
    qsa_kv_append(df1_k_, KV, df1_v_, KV, df1_zero_, df1_pos_, QR, table, bpr, bt, KVH,
                  HD, plane.k_cache, plane.v_cache, nullptr, nullptr, stream);
    dflash2_block_attn_split(df1_q_, QW, plane.k_cache, plane.v_cache, table, bt, bpr, QR,
                             df1cfg_.attention_window(l), df1_pos_, QR, NH, KVH, HD, scale,
                             df1_attn_part_, df1_attn_, stream);
    project(df1_attn_, w.o, QW);
    dflash2_add_rmsnorm_bf16(df1_resid_, df1_o_, w.post_norm, df1_x_, QR, H, eps, stream);
    gemm_.matmul(df1_x_, w.gate, df1_gate_, QR, I, H, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
    gemm_.matmul(df1_x_, w.up, df1_up_, QR, I, H, DType::BF16, GemmOut::BF16,
                 static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream);
    swiglu_limit_bf16(df1_gate_, df1_up_, df1_mlp_, static_cast<int64_t>(QR) * I, INFINITY, stream);
    project(df1_mlp_, w.down, I);
    add_inplace_bf16(df1_resid_, df1_o_, static_cast<int64_t>(QR) * H, stream);
  }
  dflash2_rmsnorm_bf16(df1_resid_, df1w_.norm, df1_h_, QR, H, eps, stream_);
}

bool Qwen35Model::dflash1_draft(int req, int64_t bonus, std::vector<int32_t>* drafts) {
  if (!dflash1_) throw std::logic_error("dflash1_draft: no v1 drafter loaded");
  const int H = cfg_.hidden_size;
  const int QR = df1cfg_.query_rows();
  const int64_t pos = session_position(req);
  if (pos < 0 || pos + QR > max_context()) return false;
  // The block writes K/V at [pos, pos + QR): grow the table now (the next
  // verify's rows live in the same span). Pool exhausted -> plain step.
  if (!pool_.ensure_request_blocks(req, pos + QR, stream_)) return false;

  // Row inputs: [bonus, mask x D] at positions pos..pos+QR-1.
  for (int j = 0; j < QR; ++j) {
    df1_io64_h_[j] = pos + j;
    df1_io64_h_[QR + j] = j == 0 ? bonus : df1cfg_.mask_token_id;
  }
  DGPP_CUDA_OK(cudaMemcpyAsync(df1_pos_, df1_io64_h_, QR * 8, cudaMemcpyHostToDevice, stream_));
  DGPP_CUDA_OK(cudaMemcpyAsync(df1_tokens_, df1_io64_h_ + QR, QR * 8, cudaMemcpyHostToDevice, stream_));
  embed_gather_bf16(globals_.embed, df1_tokens_, df1_resid_, QR, H, stream_);
  configure_gemm_rows(QR, true);
  df1_block_layers(req, stream_, /*capture=*/false);
  df1_block_select(/*capture=*/false);
  const int N = dflash2_drafts();
  std::vector<int32_t> ids(N);
  DGPP_CUDA_OK(cudaMemcpyAsync(ids.data(), df1_tok_, static_cast<size_t>(N) * 4,
                               cudaMemcpyDeviceToHost, stream_));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  drafts->assign(ids.begin(), ids.end());
  return true;
}
void Qwen35Model::df1_block_select(bool capture) {
  // Only the verifiable prefix needs head rows (per-row independent: top-1
  // of row j reads no other row's logits — skipping rows N+1..15 is
  // bitwise-identical for proposals 1..N, and saves (15-N) full-vocab
  // GEMV rows a step). The whole block still runs, since later masks read
  // the earlier rows' keys.
  const int N = dflash2_drafts(), HR = 1 + N;
  head_gemv(df1_h_, df1_logits_, HR, stream_);
  dflash2_topk_f32(df1_logits_, df1_ids_, df1_sc_, lm_vocab_count_, HR, 16, stream_, df1_topk_ws_,
                   df1_topk_ws_bytes_);
  if (world_ > 1) {
    int rows = HR;
    uint16_t* table = boundary_->stage(1, static_cast<int>(dflash2_topk_table_elems(rows, 16, world_)));
    // Small models or short verify budgets may have a one-row gather
    // buffer. Every gather reuses the recorder's stable base address.
    if (table == nullptr) {
      rows = 1;
      table = boundary_->stage(1, static_cast<int>(dflash2_topk_table_elems(rows, 16, world_)));
    }
    if (table == nullptr && capture)
      throw std::runtime_error("dflash v1 draft: the top-16 gather needs a staged buffer");
    if (table == nullptr) table = df1_table_;
    for (int off = 0; off < HR; off += rows) {
      const int count = std::min(rows, HR - off);
      const int elems = static_cast<int>(dflash2_topk_table_elems(count, 16, world_));
      if (off > 0) {
        table = boundary_->stage(1, elems);
        if (table == nullptr) table = df1_table_;
      }
      dflash2_topk_stage(df1_ids_ + off * 16, df1_sc_ + off * 16, count, 16,
                          lm_vocab_begin_, rank_, world_, table, stream_);
      if (!capture && !boundary_->stream_ordered()) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
      boundary_->reduce(table, 1, elems);
      dflash2_topk_merge(table, count, 16, world_, df1_ids_ + off * 16, df1_sc_ + off * 16, stream_);
    }
  }
  // Mask rows 1..N propose their column 0; the anchor row's entry goes unread.
  // One gather kernel (kernels-only under capture); the eager path reads
  // the N device ints home.
  df1_collect_top1(df1_ids_, 16, df1_tok_, N, stream_);
}

void Qwen35Model::df1_capture_draft(int req, const PickVerdict* verdict, int64_t* feed) {
  const int QR = df1cfg_.query_rows(), N = dflash2_drafts();
  dflash2_stage_block_batched(verdict, d_session_pos_ + req, df1cfg_.mask_token_id, QR, 1,
                              max_context(), df1_pos_, df1_tokens_, stream_);
  embed_gather_bf16(globals_.embed, df1_tokens_, df1_resid_, QR, cfg_.hidden_size, stream_);
  configure_gemm_rows(QR, true);
  df1_block_layers(req, stream_, /*capture=*/true);
  df1_block_select(/*capture=*/true);
  dflash2_block_feed_batched(verdict, df1_tok_, N, 1, 1 + N, feed, stream_);
  dflash2_publish_drafts(df1_tok_, df1_mirror_h_ + static_cast<size_t>(req) * df1cfg_.drafts(), N, stream_);
}

void Qwen35Model::session_graph_capture_block_draft(int req, const PickVerdict* verdict) {
  if (!dflash2_ && !dflash1_) throw std::logic_error("session_graph_capture_block_draft: no drafter loaded");
  if (req < 0 || req >= max_requests_) throw std::invalid_argument("session_graph_capture_block_draft: request");
  if (verdict == nullptr) throw std::invalid_argument("session_graph_capture_block_draft: null verdict");
  if (!graph_device_positions_ || !graph_device_tokens_)
    throw std::logic_error("session_graph_capture_block_draft: the step must be captured with device positions and tokens");
  if (dflash1_) {
    if ((graph_feed_rows_ > 0 ? graph_feed_rows_ : decode_rows_) != 1 + dflash2_drafts())
      throw std::logic_error("session_graph_capture_block_draft: the v1 feed is [next, N drafts]");
    df1_capture_draft(req, verdict, step_tokens_);
    return;
  }
  df_.capture_block_draft(req, verdict, df_capture_ctx());
}

void Qwen35Model::session_graph_capture_block_draft_batch(const PickVerdict* verdicts, int requests) {
  if (!dflash2_ && !dflash1_) throw std::logic_error("session_graph_capture_block_draft_batch: no drafter loaded");
  if (verdicts == nullptr) throw std::invalid_argument("session_graph_capture_block_draft_batch: null verdicts");
  if (requests < 1 || requests > max_requests_ || (!dflash1_ && requests > df_.batch_slots()))
    throw std::invalid_argument("session_graph_capture_block_draft_batch: requests exceed the slots or the stacked scratch");
  if (graph_batch_requests_ != requests || !graph_device_positions_ || !graph_device_tokens_)
    throw std::logic_error("session_graph_capture_block_draft_batch: requires the device-driven fixed batch of these slots");
  if (graph_batch_map_source_ != nullptr)
    throw std::logic_error("session_graph_capture_block_draft_batch: the block draft takes slot q as request q (no compaction)");
  if (dflash1_) {
    const int feed_rows = graph_feed_rows_ > 0 ? graph_feed_rows_ : graph_rows_per_request_;
    if (feed_rows != 1 + dflash2_drafts() || graph_rows_per_request_ > feed_rows)
      throw std::logic_error("session_graph_capture_block_draft_batch: the v1 feed is [next, N drafts]");
    // Reuse the scalar block scratch in stream order. Each request reads
    // its own KV table and publishes into its own persistent feed/mirror.
    int64_t* feeds = d_tokens_ + static_cast<size_t>(max_decode_rows_);
    for (int q = 0; q < requests; ++q)
      df1_capture_draft(q, verdicts + q, feeds + static_cast<size_t>(q) * feed_rows);
    return;
  }
  df_.capture_block_draft_batch(verdicts, requests, df_capture_ctx());
}

DFlash2CaptureCtx Qwen35Model::df_capture_ctx() const {
  DFlash2CaptureCtx c;
  c.step_tokens = step_tokens_;
  c.decode_rows = decode_rows_;
  c.graph_feed_rows = graph_feed_rows_;
  c.graph_batch_requests = graph_batch_requests_;
  c.graph_rows_per_request = graph_rows_per_request_;
  c.graph_batch_map_source = graph_batch_map_source_;
  c.graph_device_positions = graph_device_positions_;
  c.graph_device_tokens = graph_device_tokens_;
  return c;
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
  if (!dflash2_ && !dflash1_) throw std::logic_error("dflash2_draft_batch: no drafter loaded");
  if (dflash1_) {
    // v1 C1 scope: singular drafts in turn (no stacked scratch yet).
    if (reqs.empty()) throw std::invalid_argument("dflash2_draft_batch: no requests");
    if (reqs.size() != bonuses.size() || drafts == nullptr)
      throw std::invalid_argument("dflash2_draft_batch: reqs/bonuses/drafts shape");
    drafts->assign(reqs.size(), {});
    for (size_t s = 0; s < reqs.size(); ++s)
      (void)dflash1_draft(reqs[s], bonuses[s], &(*drafts)[s]);
    return;
  }
  df_.draft_batch(reqs, bonuses, drafts);
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

// ---- image inputs ------------------------------------------------------------

const std::vector<ImageInput>* Qwen35Model::images_for_req(int req) const {
  if (req < 0 || req >= static_cast<int>(prefill_images_per_req_.size())) return nullptr;
  return prefill_images_per_req_[static_cast<size_t>(req)];
}

void Qwen35Model::store_request_images(int req, const std::vector<ImageInput>* images) {
  if (req < 0 || req >= max_requests_) throw std::out_of_range("Qwen3.5 image prefill: request slot");
  if (prefill_images_per_req_.size() < static_cast<size_t>(max_requests_))
    prefill_images_per_req_.assign(static_cast<size_t>(max_requests_), nullptr);
  prefill_images_per_req_[static_cast<size_t>(req)] = images;
}

class Qwen35Model::ImagePrefillScope {
 public:
  ImagePrefillScope(Qwen35Model& model, int req, const std::vector<ImageInput>* images)
      : model_(model), req_(req) {
    model_.store_request_images(req_, images && !images->empty() ? images : nullptr);
    invalidate_window();
  }
  ~ImagePrefillScope() {
    model_.prefill_images_per_req_[static_cast<size_t>(req_)] = nullptr;
    invalidate_window();
  }
  ImagePrefillScope(const ImagePrefillScope&) = delete;
  ImagePrefillScope& operator=(const ImagePrefillScope&) = delete;

 private:
  void invalidate_window() {
    model_.image_embeddings_ = nullptr;
    model_.image_window_first_ = model_.image_window_end_ = 0;
  }
  Qwen35Model& model_;
  int req_;
};

Qwen35Model::Outputs Qwen35Model::session_prefill_images(
    int req, const std::vector<int64_t>& prompt, const std::vector<ImageInput>& images) {
  return session_prefill_images(req, prompt, images, {}, nullptr);
}

Qwen35Model::Outputs Qwen35Model::session_prefill_images(
    int req, const std::vector<int64_t>& prompt, const std::vector<ImageInput>& images,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap) {
  return session_prefill_with_images(req, prompt, images, boundaries, snap, false);
}

Qwen35Model::Outputs Qwen35Model::session_prefill_resume_images(
    int req, const std::vector<int64_t>& suffix, const std::vector<ImageInput>& images,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap) {
  return session_prefill_with_images(req, suffix, images, boundaries, snap, true);
}

Qwen35Model::PrefillCursor Qwen35Model::session_prefill_begin(
    int req, const std::vector<int64_t>& prompt, int64_t reserve_tokens, int64_t chunk_tokens,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap, int64_t attach_position,
    const std::vector<ImageInput>* images) {
  const bool has_images = images && !images->empty();
  if (has_images) {
    if (!vision_) throw std::invalid_argument("Qwen3.5: checkpoint has no vision tower");
    validate_image_inputs(*images, prompt.size());
    qwen35_check_image_pads(prompt, 0, *images, image_pad_id());
  }
  ImagePrefillScope scope(*this, req, images);
  auto cursor = Base::session_prefill_begin(req, prompt, reserve_tokens, chunk_tokens, boundaries,
                                            snap, attach_position);
  return PrefillCursor{std::move(cursor), has_images ? images : nullptr};
}

bool Qwen35Model::session_prefill_advance(PrefillCursor& cursor, int64_t chunk_tokens) {
  ImagePrefillScope scope(*this, cursor.req, cursor.images);
  const bool done = Base::session_prefill_advance(cursor, chunk_tokens);
  if (done) cursor.images = nullptr;
  return done;
}

Qwen35Model::Outputs Qwen35Model::session_prefill_with_images(
    int req, const std::vector<int64_t>& ids, const std::vector<ImageInput>& images,
    const std::vector<int64_t>& boundaries, SnapshotRequest* snap, bool resume) {
  if (!vision_) throw std::invalid_argument("Qwen3.5: checkpoint has no vision tower");
  if (req < 0 || req >= max_requests_) throw std::out_of_range("Qwen3.5 image prefill: request slot");
  const int64_t start = resume ? session_pos_[static_cast<size_t>(req)] : 0;
  if (start < 0) throw std::invalid_argument("Qwen3.5 image prefill: closed session");
  validate_image_inputs(images, start + static_cast<size_t>(ids.size()));
  if (images.empty())
    return resume ? Base::session_prefill_resume(req, ids, boundaries, snap)
                  : Base::session_prefill(req, ids, boundaries, snap);
  qwen35_check_image_pads(ids, start, images, image_pad_id());
  ImagePrefillScope scope(*this, req, &images);
  return resume ? Base::session_prefill_resume(req, ids, boundaries, snap)
                : Base::session_prefill(req, ids, boundaries, snap);
}

void Qwen35Model::stage_image_embeddings(int64_t first, int64_t end,
                                         const std::vector<ImageInput>* images) {
  image_window_first_ = first;
  image_window_end_ = end;
  image_embeddings_ = images && !images->empty() && end > first
                          ? vision_->stage(*images, first, end)
                          : nullptr;
}

void Qwen35Model::apply_image_embeddings(uint16_t* dst, int64_t first, int rows, int shift,
                                         const std::vector<ImageInput>* images) {
  if (!images || images->empty() || rows <= 0) return;
  const int64_t begin_all = first + shift, end_all = begin_all + rows;
  // Windowed staging (any-size image prompts): the tower's staged window
  // holds at most kWindowTokens rows, so a chunk wider than the window is
  // consumed sub-span by sub-span — stage, copy, repeat (see
  // models/qwen/image_stage_plan.hpp for the plan; pinned by unit test).
  const int64_t window = QwenVisionConfig::kWindowTokens;
  const int h = cfg_.hidden_size;
  std::vector<image_stage::Span> spans;
  spans.reserve(images->size());
  for (const auto& im : *images) spans.push_back({im.offset, im.tokens});
  for (const auto& step :
       image_stage::plan(begin_all, end_all, window, spans.data(), spans.size())) {
    if (step.first < image_window_first_ || step.end > image_window_end_)
      stage_image_embeddings(step.first, step.end, images);
    for (const auto& copy : step.copies) {
      if (!image_embeddings_ || copy.begin < image_window_first_ || copy.end > image_window_end_)
        throw std::logic_error("Qwen3.5: image consumer escaped its staged window");
      const uint16_t* source = image_embeddings_ + (copy.begin - image_window_first_) * h;
      const int n = static_cast<int>(copy.end - copy.begin);
      qwen_image_copy(source, dst + (copy.begin - begin_all) * h, n, h, stream_);
    }
  }
}

}  // namespace dgpp
