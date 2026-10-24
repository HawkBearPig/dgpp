#include "models/glm/loader.hpp"

#include "common/bf16_residency.hpp"

#include "loaders/weight_build.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/fp8_dequant.hpp"

namespace dgpp {

namespace {

std::string_view layer_suffix(const std::string& name) {
  size_t dots = 0, i = 0;
  for (; i < name.size() && dots < 4; ++i)
    if (name[i] == '.') ++dots;
  return dots == 4 ? std::string_view(name).substr(i) : std::string_view(name);
}

// Replicated weights (DESIGN §5.2): every rank loads these byte-identically
// and the boot digest folds them. The scale partner of a replicated
// quantized tensor is replicated too — strip the "_scale_inv" suffix before
// matching. THIS LIST IS THE SINGLE SOURCE OF TRUTH for hash_replicated()
// and the build-time read-class assertions; the build functions must read
// exactly these tensors verbatim at world>1 (the assertions below enforce
// the agreement in the other direction).
bool is_replicated(const GlmExpectedTensor& e) {
  switch (e.cls) {
    case GlmWeightClass::Embed:
    case GlmWeightClass::LmHead:
    case GlmWeightClass::FinalNorm:
    case GlmWeightClass::LayerNorm:
    case GlmWeightClass::Mhc:
    case GlmWeightClass::Router:
    case GlmWeightClass::DsaIndexer:
      return true;
    case GlmWeightClass::Mtp:
      // v1: the draft head is replicated (the forward does not run the
      // MTP layer; sharding rules arrive with the M6 MTP spec).
      return true;
    case GlmWeightClass::Kda: {
      std::string_view s = layer_suffix(e.name);
      if (s.size() > 10 && s.substr(s.size() - 10) == "_scale_inv")
        s.remove_suffix(10);
      return s == "self_attn.f_a_proj.weight" ||
             s == "self_attn.g_a_proj.weight" ||
             s == "self_attn.o_norm.weight";
    }
    case GlmWeightClass::Dsa: {
      std::string_view s = layer_suffix(e.name);
      if (s.size() > 10 && s.substr(s.size() - 10) == "_scale_inv")
        s.remove_suffix(10);
      return s == "self_attn.q_a_proj.weight" ||
             s == "self_attn.kv_a_proj_with_mqa.weight" ||
             s == "self_attn.q_a_layernorm.weight" ||
             s == "self_attn.kv_a_layernorm.weight";
    }
    default:
      return false;  // DenseMlp, SharedExpert, RoutedExpert: sharded
  }
}

// Identify DSA tensors that need the BF16 bridge because a TP slice
// starts inside a quantization block. Each rank reads the full source,
// dequantizes it and slices the BF16 result, matching GlmTpViews.
// These reads count toward full-source byte accounting but are excluded
// from replicated-output digests because the resident results are sharded.
// When bridge_active is false, aligned FP8 pairs are sliced directly.

bool is_dsa_bridge(const GlmExpectedTensor& e, bool bridge_active) {
  if (!bridge_active || e.cls != GlmWeightClass::Dsa) return false;
  std::string_view s = layer_suffix(e.name);
  if (s.size() > 10 && s.substr(s.size() - 10) == "_scale_inv")
    s.remove_suffix(10);
  return s == "self_attn.q_b_proj.weight" || s == "self_attn.o_proj.weight";
}

// The rank-invariant re-read set for the byte reconcile (§5.2): replicated
// tensors, the DSA bridges, and the NVFP4 experts' per-tensor global scales
// (4 bytes each, read whole by every rank — the sliced payload and scale
// rows partition across ranks, the scalar cannot).
bool is_full_read(const GlmExpectedTensor& e, bool bridge_active) {
  return is_replicated(e) || is_dsa_bridge(e, bridge_active) ||
         e.role == GlmTensorRole::Fp4Global;
}

}  // namespace


namespace {

// Everything one layer build needs. `copy` false = counting pass: identical
// allocation sequence, no memcpys, no kernel launches (source lookups stay
// inside if(copy) — the counting pass has no tensors). Names and sizes come
// from the expected table (config-derived); mmap pointers only feed copies.
//
// SHARDING (M5 d4): rank/world produce the resident layer directly at this
// rank's local geometry. world=1 is the DEGENERATE RANK 0 — every slice
// covers the full extent, every grant sequence and byte is the M4 build's,
// so the world=1 path cannot drift. The local arithmetic mirrors
// GlmTpViews::bind exactly; glm_tp_test pins the two paths bitwise.
struct BuildCtx : WeightBuilder<GlmExpectedTensor> {
  const GlmTextConfig& cfg;
  GlmLayerResident& out;  // pointer stores are harmless in counting mode

  // Local geometry at tp_size=world (world=1: the full geometry).
  KdaGeometry kgeo{};
  DsaGeometry dgeo{};
  bool dsa_bridge = true;  // set by init_geometry
  int64_t dense_inter = 0;    // intermediate_size/world
  int64_t shared_inter = 0;   // moe_intermediate_size/world (routed too)
  int64_t dsa_kv_rows_head = 0;  // qk_nope + v per head

  BuildCtx(const GlmTextConfig& cfg_,
           const std::vector<GlmExpectedTensor>& table_,
           const std::unordered_map<std::string, const GlmExpectedTensor*>& by_name_,
           GlmLayerBump& bump_, GlmLayerResident& out_,
           const std::unordered_map<std::string, const TensorInfo*>& tensors_,
           std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_,
           bool copy_, int rank_, int world_)
      : WeightBuilder<GlmExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_,
                                         packs_, copy_, rank_, world_, "glm loader"),
        cfg(cfg_), out(out_) {}

  // The family's classification for the shared primitives: replicated
  // (rank-invariant reads), the DSA bridge (read whole, sliced after
  // dequant), the NVFP4 global scale.
  bool replicated(const GlmExpectedTensor& e) const override { return is_replicated(e); }
  bool verbatim_ok(const GlmExpectedTensor& e) const override {
    return is_replicated(e) || is_dsa_bridge(e, dsa_bridge);
  }
  bool full_read(const GlmExpectedTensor& e) const override {
    return is_full_read(e, dsa_bridge);
  }
  bool fp4_global(const GlmExpectedTensor& e) const override {
    return e.role == GlmTensorRole::Fp4Global;
  }

  void init_geometry() {
    KdaConfig kc = cfg.kda_config();
    kc.tp_size = world;
    kgeo = KdaGeometry::from_config(kc);
    DsaConfig dc = cfg.dsa_config();
    dc.tp_size = world;
    dgeo = DsaGeometry::from_config(dc);
    dense_inter = cfg.intermediate_size / world;
    shared_inter = cfg.moe_config().inter / world;
    dsa_kv_rows_head = cfg.qk_nope_head_dim + cfg.v_head_dim;
    // The DSA projections' form (see is_dsa_bridge): the FP8 pairs directly
    // when this rank's q_b rows and o_proj columns start on the 128-wide
    // scale grid, the bf16 bridge otherwise (the test fixtures at world 4).
    dsa_bridge = (dgeo.local_q_rows % 128) != 0 || (dgeo.local_v_rows % 128) != 0;
  }

  void build_mhc(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.mhc.attn_base = load_f32(p + "hc_attn_base");
    out.mhc.attn_fn = load_bf16(p + "hc_attn_fn");
    out.mhc.attn_scale = load_f32(p + "hc_attn_scale");
    out.mhc.ffn_base = load_f32(p + "hc_ffn_base");
    out.mhc.ffn_fn = load_bf16(p + "hc_ffn_fn");
    out.mhc.ffn_scale = load_f32(p + "hc_ffn_scale");
  }

  void build_norms(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.ln1 = load_bf16(p + "input_layernorm.weight");
    out.ln2 = load_bf16(p + "post_attention_layernorm.weight");
  }

  void build_kda(int layer) {
    const std::string p = "model.language_model.layers." +
                          std::to_string(layer) + ".self_attn.";
    // TP geometry: world=1 is the degenerate rank 0 — local == full, the
    // M4 merged layout, byte-for-byte. The local arithmetic mirrors
    // GlmTpViews::bind exactly.
    const int64_t head_dim = cfg.kda_head_dim;
    const int64_t hidden = cfg.hidden_size;
    const int64_t lp_s = kgeo.local_proj;        // this rank's q/k/v rows
    const int64_t lp_f = lp_s * world;           // full-heads rows
    const int64_t h_s = kgeo.local_heads;

    // Merged in_proj rows [f_a | g_a | q | k | v | b] at LOCAL geometry —
    // the M2 kernel contract; the checkpoint stores the six projections
    // separately. f_a/g_a are replicated (full rows); q/k/v and b carry
    // this rank's head rows.
    const int64_t in_rows = 2 * head_dim + 3 * lp_s + h_s;
    // (packable: the model's 12-bit companion replaces it — grant_bf16.)
    uint16_t* in_proj = static_cast<uint16_t*>(grant_bf16(in_rows, hidden, /*packable=*/true));
    struct Piece {
      const char* name;
      int64_t full_rows;  // source rows (geometry check)
      bool shard;         // head-sharded vs replicated
      int64_t src_off;    // this rank's first source row
      int64_t rows;       // this rank's rows
    };
    const Piece pieces[] = {
        {"f_a_proj.weight", head_dim, false, 0, head_dim},
        {"g_a_proj.weight", head_dim, false, 0, head_dim},
        {"q_proj.weight", lp_f, true, rank * lp_s, lp_s},
        {"k_proj.weight", lp_f, true, rank * lp_s, lp_s},
        {"v_proj.weight", lp_f, true, rank * lp_s, lp_s},
        {"b_proj.weight", h_s * world, true, rank * h_s, h_s},
    };
    int64_t dst_row = 0;
    for (const Piece& piece : pieces) {
      const GlmExpectedTensor& e = expected(p + piece.name);
      if (e.shape[0] != piece.full_rows || e.shape[1] != hidden)
        throw std::runtime_error("glm loader: KDA piece geometry mismatch");
      if (sharded() && piece.shard == is_replicated(e))
        throw std::runtime_error("glm loader: TP read-class drift on '" +
                                 p + piece.name + "'");
      if (copy) {
        std::memcpy(bump.host(in_proj) + static_cast<size_t>(dst_row) * hidden,
                    static_cast<const uint8_t*>(source(p + piece.name).data) +
                        static_cast<size_t>(piece.src_off) * hidden * 2,
                    static_cast<size_t>(piece.rows) * hidden * 2);
        note_read(e,
                  static_cast<size_t>(piece.rows) * hidden * 2);
      }
      dst_row += piece.rows;
    }
    if (dst_row != in_rows)
      throw std::runtime_error("glm loader: KDA in_proj row count mismatch");

    // Merged causal conv channels [q | k | v] at LOCAL geometry.
    const int64_t conv_w = cfg.kda_conv_width;
    uint16_t* conv = static_cast<uint16_t*>(
        bump.alloc(static_cast<size_t>(3 * lp_s) *
                   static_cast<size_t>(conv_w) * 2));
    static const char* convs[3] = {"q_conv1d.weight", "k_conv1d.weight",
                                   "v_conv1d.weight"};
    for (int i = 0; i < 3; ++i) {
      const GlmExpectedTensor& e = expected(p + convs[i]);
      if (e.shape[0] != lp_f || e.shape[2] != conv_w)
        throw std::runtime_error("glm loader: conv piece geometry mismatch");
      if (copy) {
        std::memcpy(bump.host(conv) + static_cast<size_t>(i) * lp_s * conv_w,
                    static_cast<const uint8_t*>(source(p + convs[i]).data) +
                        static_cast<size_t>(rank) * lp_s * conv_w * 2,
                    static_cast<size_t>(lp_s) * conv_w * 2);
        note_read(e, static_cast<size_t>(lp_s) * conv_w * 2);
      }
    }

    out.kda.in_proj = in_proj;
    out.kda.conv = conv;
    out.kda.f_b = load_bf16_rows(p + "f_b_proj.weight", rank * lp_s, lp_s);
    out.kda.g_b = load_bf16_rows(p + "g_b_proj.weight", rank * lp_s, lp_s);
    out.kda.a_log = load_f32_range(p + "A_log", rank * h_s, h_s);
    out.kda.dt_bias = load_f32_range(p + "dt_bias", rank * lp_s, lp_s);
    out.kda.o_norm = load_bf16(p + "o_norm.weight");

    // o_proj [hidden, lp_f] bf16: column slice -> packed [hidden, lp_s].
    // BF16 carries no scale grid, so any column start is representable;
    // the source is host mmap, so the pack runs in phase one (staging).
    {
      const std::string name = p + "o_proj.weight";
      const GlmExpectedTensor& e = expected(name);
      if (e.shape[0] != hidden || e.shape[1] != lp_f)
        throw std::runtime_error("glm loader: KDA o_proj geometry mismatch");
      uint16_t* packed = static_cast<uint16_t*>(grant_bf16(hidden, lp_s, /*packable=*/true));
      if (copy) {
        packs.push_back(PackJob{
            static_cast<const uint16_t*>(source(name).data) + rank * lp_s,
            packed, static_cast<size_t>(lp_f) * 2,
            static_cast<size_t>(lp_s) * 2, static_cast<size_t>(lp_s) * 2,
            static_cast<size_t>(hidden), /*src_on_device=*/false});
        note_read(e, static_cast<size_t>(hidden) * lp_s * 2);
      }
      out.kda.o_proj = packed;
    }
  }

  void build_dsa(int layer) {
    const std::string p = "model.language_model.layers." +
                          std::to_string(layer) + ".self_attn.";
    const int64_t hidden = cfg.hidden_size;
    const int64_t q_lora = cfg.q_lora_rank;
    const int64_t lh = dgeo.local_heads;
    const int64_t local_v_rows = dgeo.local_v_rows;

    if (!dsa_bridge) {
      // The FP8 pairs as resident: q_a / kv_a replicated in
      // full, q_b this rank's 128-aligned row slice, o_proj (below) this
      // rank's packed column slice — the scale-aware GEMM reads them as
      // they are, half the bytes of the bf16 bridge per token.
      out.dsa.q_a_q = load_quant(p + "q_a_proj.weight");
      out.dsa.kv_a_q = load_quant(p + "kv_a_proj_with_mqa.weight");
      out.dsa.qkv_a = nullptr;
    } else {
      // Fused qkv_a [q_lora + kv_lora, hidden]: replicated latents — q_a rows
      // then kv_a rows, both dequantized (their scale grids differ — two
      // jobs, one buffer). Full at every world.
      const GlmExpectedTensor& qa = expected(p + "q_a_proj.weight");
      const GlmExpectedTensor& kva = expected(p + "kv_a_proj_with_mqa.weight");
      uint16_t* qkv_a = static_cast<uint16_t*>(
          bump.alloc(static_cast<size_t>(qa.shape[0] + kva.shape[0]) *
                     static_cast<size_t>(hidden) * 2));
      const GlmQuantMatrix qa_q = load_quant(p + "q_a_proj.weight");
      jobs.push_back(DequantJob{qa_q.payload, qa_q.scales, qkv_a, qa_q.rows,
                                qa_q.cols});
      const GlmQuantMatrix kva_q =
          load_quant(p + "kv_a_proj_with_mqa.weight");
      jobs.push_back(DequantJob{
          kva_q.payload, kva_q.scales,
          qkv_a + static_cast<size_t>(q_lora) * static_cast<size_t>(hidden),
          kva_q.rows, kva_q.cols});
      out.dsa.qkv_a = qkv_a;
    }
    out.dsa.q_aln = load_bf16(p + "q_a_layernorm.weight");
    out.dsa.kv_aln = load_bf16(p + "kv_a_layernorm.weight");
    if (!dsa_bridge) {
      out.dsa.q_b_q =
          sharded() ? load_quant_rows(p + "q_b_proj.weight",
                                      static_cast<int64_t>(rank) * dgeo.local_q_rows,
                                      dgeo.local_q_rows)
                    : load_quant(p + "q_b_proj.weight");
      out.dsa.q_b = nullptr;
    } else {
      // q_b: the bridge dequantizes the full source (a head-row slice of the
      // quantized matrix starts mid-block at this world — unrepresentable
      // on the scale grid); the resident keeps this rank's rows of the bf16
      // bridge, exactly the view bind slices from the same buffer.
      uint16_t* q_b = load_dequant_bf16(p + "q_b_proj.weight");
      out.dsa.q_b = sharded()
                        ? q_b + static_cast<size_t>(rank) *
                                    static_cast<size_t>(dgeo.local_q_rows) *
                                    q_lora
                        : q_b;
    }
    // kv_b: BF16 in the checkpoint — this rank's head block is contiguous
    // source rows; a real slice read.
    out.dsa.kv_b = load_bf16_rows(
        p + "kv_b_proj.weight",
        static_cast<int64_t>(rank) * lh * dsa_kv_rows_head,
        lh * dsa_kv_rows_head);
    if (!dsa_bridge) {
      out.dsa.o_proj_q =
          sharded() ? load_quant_cols(p + "o_proj.weight",
                                      static_cast<int64_t>(rank) * local_v_rows,
                                      local_v_rows)
                    : load_quant(p + "o_proj.weight");
      out.dsa.o_proj = nullptr;
    } else {
      // o_proj: bridge dequant of the full source, then the packed column
      // slice [hidden, local_v_rows] in the post-dequant phase. world=1
      // keeps the bridge buffer itself (the M4 resident).
      uint16_t* o_bridge = load_dequant_bf16(p + "o_proj.weight");
      if (sharded()) {
        uint16_t* packed = static_cast<uint16_t*>(
            bump.alloc(static_cast<size_t>(hidden) * local_v_rows * 2));
        if (copy)
          packs.push_back(PackJob{o_bridge + rank * local_v_rows, packed,
                                  static_cast<size_t>(local_v_rows) * world * 2,
                                  static_cast<size_t>(local_v_rows) * 2,
                                  static_cast<size_t>(local_v_rows) * 2,
                                  static_cast<size_t>(hidden),
                                  /*src_on_device=*/true});
        out.dsa.o_proj = packed;
      } else {
        out.dsa.o_proj = o_bridge;
      }
    }
    const std::string ip = p + "indexer.";
    out.dsa.wq_b = load_bf16(ip + "wq_b.weight");
    out.dsa.wk = load_bf16(ip + "wk.weight");
    out.dsa.wp = load_bf16(ip + "weights_proj.weight");
    out.dsa.gate = load_bf16(ip + "index_kpool_compress_gate");
    out.dsa.k_norm_w = load_bf16(ip + "k_norm.weight");
    out.dsa.k_norm_b = load_bf16(ip + "k_norm.bias");
    // APE: checkpoint BF16 [kpool, index_head_dim]; the M3 kernel wants F32.
    // Read via memcpy: safetensors does not align individual tensors in
    // the file, so a typed uint16_t load from the mmap can be unaligned
    // (the fixture's byte layout lands one at an odd offset).
    {
      const std::string name = ip + "index_kpool_compress_ape";
      const GlmExpectedTensor& e = expected(name);
      const size_t n = e.numel();
      float* ape = static_cast<float*>(bump.alloc(n * 4));
      if (copy) {
        const uint8_t* src =
            static_cast<const uint8_t*>(source(name).data);
        float* h_ape = bump.host(ape);
        for (size_t i = 0; i < n; ++i) {
          uint16_t bits;
          std::memcpy(&bits, src + i * 2, 2);
          h_ape[i] = bf16_bits_to_float(bits);
        }
        note_read(e, n * 2);
      }
      out.dsa.ape = ape;
    }
  }

  void build_dense_mlp(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    const int64_t I = dense_inter;
    out.dense[0] = load_quant_rows(p + "mlp.gate_proj.weight", rank * I, I);
    out.dense[1] = load_quant_rows(p + "mlp.up_proj.weight", rank * I, I);
    out.dense[2] = load_quant_cols(p + "mlp.down_proj.weight", rank * I, I);
  }

  void build_moe(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.moe.router_gate = load_bf16(p + "mlp.gate.weight");
    out.moe.router_bias = load_f32(p + "mlp.gate.e_score_correction_bias");
    const std::string sp = p + "mlp.shared_experts.";
    const int64_t M = shared_inter;
    out.moe.shared[0] = load_quant_rows(sp + "gate_proj.weight", rank * M, M);
    out.moe.shared[1] = load_quant_rows(sp + "up_proj.weight", rank * M, M);
    out.moe.shared[2] = load_quant_cols(sp + "down_proj.weight", rank * M, M);
    // Every expert, sliced on the intermediate dim exactly like the shared
    // expert: this rank's M gate/up rows and M down columns. Each rank's
    // per-token expert bytes are then top_k * 3 slices whatever the routing
    // — no busiest rank at the FFN boundary. The column pack
    // runs on the CPU straight from the mmap (strided rows of M bytes).
    const int64_t E = cfg.moe_config().n_experts;
    if (cfg.expert_format(layer) == GlmExpertFormat::Nvfp4Group16) {
      // NVFP4 routed experts (docs/nvfp4_plan.md §4): the same inter
      // slice — this rank's M gate/up rows, its M down columns — on the
      // packed payload and the per-row scales; every matrix's global scale
      // gathered into one [E, 3] array so a view carries one pointer.
      float* globals = static_cast<float*>(
          bump.alloc(static_cast<size_t>(E) * 3 * sizeof(float)));
      out.moe.expert_global_scales = globals;
      out.moe.experts.clear();
      out.moe.experts_fp4.resize(static_cast<size_t>(E) * 3);
      static const char* kProj[3] = {"gate_proj.weight", "up_proj.weight",
                                     "down_proj.weight"};
      for (int64_t e = 0; e < E; ++e) {
        const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
        for (int i = 0; i < 3; ++i) {
          const std::string base = ep + kProj[i];
          float* g = globals + e * 3 + i;
          load_fp4_global(base, g);
          out.moe.experts_fp4[static_cast<size_t>(e) * 3 + i] =
              i < 2 ? load_fp4_rows(base, rank * M, M, g)
                    : load_fp4_cols(base, rank * M, M, g);
        }
      }
      return;
    }
    out.moe.experts_fp4.clear();
    out.moe.expert_global_scales = nullptr;
    out.moe.experts.resize(static_cast<size_t>(E) * 3);
    for (int64_t e = 0; e < E; ++e) {
      const std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
      out.moe.experts[static_cast<size_t>(e) * 3 + 0] =
          load_quant_rows(ep + "gate_proj.weight", rank * M, M);
      out.moe.experts[static_cast<size_t>(e) * 3 + 1] =
          load_quant_rows(ep + "up_proj.weight", rank * M, M);
      out.moe.experts[static_cast<size_t>(e) * 3 + 2] =
          load_quant_cols(ep + "down_proj.weight", rank * M, M);
    }
  }

  void build_mtp_head(int layer) {
    const std::string p =
        "model.language_model.layers." + std::to_string(layer) + ".";
    out.enorm = load_bf16(p + "enorm.weight");
    out.hnorm = load_bf16(p + "hnorm.weight");
    out.eh_proj = load_bf16(p + "eh_proj.weight", /*packable=*/true);
    out.shared_head_norm = load_bf16(p + "shared_head.norm.weight");
  }

  void build_layer(int layer) {
    init_geometry();
    const bool is_mtp = layer == cfg.mtp_layer();
    const int max_layer =
        cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    if (layer < 0 || layer >= max_layer)
      throw std::invalid_argument(
          "glm loader: layer index out of range: " + std::to_string(layer));

    out.layer = layer;
    out.kind = is_mtp || cfg.layers[layer] == GlmLayerKind::Dsa
                   ? GlmLayerKind::Dsa
                   : GlmLayerKind::Kda;
    if (!is_mtp) build_mhc(layer);
    build_norms(layer);
    if (out.kind == GlmLayerKind::Kda)
      build_kda(layer);
    else
      build_dsa(layer);
    const bool dense =
        !is_mtp && cfg.mlps[layer] == GlmMlpKind::Dense;
    if (dense)
      build_dense_mlp(layer);
    else
      build_moe(layer);
    if (is_mtp) build_mtp_head(layer);
  }
};

// Runs one layer's build in counting mode; returns the exact byte total at
// the given rank's geometry — and, of it, the bytes a side-grant build
// places aside (LayerBump: the packable bf16 matrices).
struct CountedLayer {
  size_t bytes = 0;
  size_t side_bytes = 0;
};
CountedLayer count_layer(const GlmTextConfig& cfg, int layer, int rank, int world) {
  GlmLayerBump bump;
  bump.counting = true;
  bump.side_mode = true;
  bump.capacity = SIZE_MAX;
  std::vector<GlmExpectedTensor> table =
      glm_expected_layer_tensors(cfg, layer);
  std::unordered_map<std::string, const GlmExpectedTensor*> by_name;
  for (const auto& e : table) by_name.emplace(e.name, &e);
  GlmLayerResident scratch;
  std::vector<DequantJob> jobs;
  std::vector<PackJob> packs;
  std::unordered_map<std::string, const TensorInfo*> no_tensors;
  BuildCtx ctx(cfg,    table, by_name, bump,  scratch,
                no_tensors, jobs,  packs,   false, rank, world);
  ctx.build_layer(layer);
  return CountedLayer{bump.cursor, bump.side_bytes};
}
size_t count_layer_bytes(const GlmTextConfig& cfg, int layer, int rank, int world) {
  return count_layer(cfg, layer, rank, world).bytes;
}

}  // namespace

size_t GlmLayerStream::layer_bytes(const GlmTextConfig& cfg, int layer,
                                   int rank, int world) {
  return count_layer_bytes(cfg, layer, rank, world);
}

size_t GlmLayerStream::layer_side_bytes(const GlmTextConfig& cfg, int layer, int rank, int world) {
  return count_layer(cfg, layer, rank, world).side_bytes;
}

namespace {
// The contiguous vocab-slice bounds for a rank's sharded lm head
// (integer split: gap-free and overlap-free at any vocab/world).
std::pair<int, int> lm_head_slice(const GlmTextConfig& cfg, int rank,
                                  int world) {
  const int64_t V = cfg.vocab_size;
  const int begin = static_cast<int>(V * rank / world);
  const int end = static_cast<int>(V * (rank + 1) / world);
  return {begin, end - begin};
}
}  // namespace

int GlmLayerStream::lm_vocab_count(const GlmTextConfig& cfg, int rank,
                                   int world, GlmHeadSharding head) {
  return head == GlmHeadSharding::VocabSharded
             ? lm_head_slice(cfg, rank, world).second
             : cfg.vocab_size;
}

size_t GlmLayerStream::globals_bytes(const GlmTextConfig& cfg, int rank,
                                       int world, GlmHeadSharding head) {
  const int head_vocab = head == GlmHeadSharding::VocabSharded
                             ? lm_head_slice(cfg, rank, world).second
                             : cfg.vocab_size;
  const size_t embed_bytes = align_up_256(static_cast<size_t>(cfg.vocab_size) *
                                          cfg.hidden_size * 2);
  const size_t head_bytes = align_up_256(static_cast<size_t>(head_vocab) *
                                         cfg.hidden_size * 2);
  const size_t norm_bytes = align_up_256(static_cast<size_t>(cfg.hidden_size) * 2);
  return embed_bytes + head_bytes + norm_bytes;
}

// The lm head is the globals' packable matrix (the model's 12-bit companion
// replaces it): its grant goes aside under side grants.
size_t GlmLayerStream::globals_side_bytes(const GlmTextConfig& cfg, int rank, int world,
                                          GlmHeadSharding head) {
  const int head_vocab = lm_vocab_count(cfg, rank, world, head);
  if (!bf12_shape_ok(head_vocab, cfg.hidden_size)) return 0;
  return align_up_256(static_cast<size_t>(head_vocab) * cfg.hidden_size * 2);
}

size_t GlmLayerStream::side_bytes(const GlmTextConfig& cfg, int rank, int world,
                                  GlmHeadSharding head, bool with_mtp) {
  size_t total = globals_side_bytes(cfg, rank, world, head);
  for (int l = 0; l < cfg.num_hidden_layers; ++l) total += layer_side_bytes(cfg, l, rank, world);
  if (with_mtp && cfg.mtp_layer() >= 0) total += layer_side_bytes(cfg, cfg.mtp_layer(), rank, world);
  return total;
}

size_t GlmLayerStream::resident_bytes(const GlmTextConfig& cfg, int rank,
                                      int world, GlmHeadSharding head) {
  const int max_layer =
      cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
  size_t total = globals_bytes(cfg, rank, world, head);
  for (int l = 0; l < max_layer; ++l)
    total += layer_bytes(cfg, l, rank, world);
  return total;
}

GlmLayerStream::GlmLayerStream(const GlmTextConfig& cfg,
                               const std::string& checkpoint_dir, int rank,
                               int world, GlmResidency residency,
                               GlmHeadSharding head, bool resident_mtp)
    : cfg_(cfg),
      rank_(rank),
      world_(world),
      residency_(residency),
      head_(head),
      resident_mtp_(resident_mtp),
      layer_bump_(std::make_unique<GlmLayerBump>()),
      globals_bump_(std::make_unique<GlmLayerBump>()) {
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument(
        "glm loader: rank/world out of range (rank " + std::to_string(rank) +
        ", world " + std::to_string(world) + ")");
  // TP geometry — BEFORE a single shard is opened, so a bad config fails
  // in milliseconds with its dim named, not after 328 GB of mmap.
  if (world > 1) glm_tp_validate_geometry(cfg, rank, world);

  // Open shards (sorted for determinism), index headers, validate the full
  // text binding before a single byte of payload moves.
  namespace fs = std::filesystem;
  std::vector<fs::path> shard_paths;
  for (const auto& entry : fs::directory_iterator(checkpoint_dir))
    if (entry.path().extension() == ".safetensors")
      shard_paths.push_back(entry.path());
  if (shard_paths.empty())
    throw std::runtime_error("glm loader: no .safetensors shards in " +
                             checkpoint_dir);
  std::sort(shard_paths.begin(), shard_paths.end());

  std::unordered_map<std::string, GlmTensorDesc> present;
  for (const auto& path : shard_paths) {
    auto f = SafetensorsFile::open(path.string());
    f->for_each([&](const TensorInfo& t) {
      auto [it, inserted] =
          tensors_.emplace(t.name, &t);
      if (!inserted)
        throw std::runtime_error("glm loader: duplicate tensor '" + t.name +
                                 "' in " + path.string());
      present.emplace(t.name, GlmTensorDesc{t.dtype, t.shape});
    });
    shards_.push_back(std::move(f));
  }

  const GlmBindReport rep = glm_validate_text_binding(cfg_, present);
  if (!rep.ok()) {
    std::string msg = "glm loader: checkpoint binding failed: ";
    for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
      if (i) msg += "; ";
      msg += rep.errors[i];
    }
    throw std::runtime_error(msg);
  }

  // Size the bump at the largest layer AT THIS RANK'S GEOMETRY (headers
  // say which; the shared build code says how much) and take a dedicated
  // stream for the dequant launches. RESIDENT mode never allocates the
  // shared bump — each layer materializes into its own exact-formula bump
  // on first load — so the stream opens without a single layer allocation
  // and the per-layer stores are sized instead.
  size_t capacity = 0;
  const int max_layer =
      cfg_.num_hidden_layers + (cfg_.mtp_layer() >= 0 ? 1 : 0);
  for (int i = 0; i < max_layer; ++i)
    capacity = std::max(capacity, count_layer_bytes(cfg_, i, rank_, world_));
  capacity_ = capacity;
  if (residency_ == GlmResidency::Resident) {
    resident_bumps_.resize(static_cast<size_t>(max_layer));  // null bumps
    resident_layers_.assign(static_cast<size_t>(max_layer),
                            GlmLayerResident{});
    check_resident_footprint_fits();
  } else {
    layer_bump_->init(capacity);
  }
  // Side grants (common/bf16_residency.hpp): resident stacks only — a
  // streaming bump is rewritten every forward and is never packed.
  side_grants_ = residency_ == GlmResidency::Resident && bf16_side_grants();
  const size_t globals_cap = globals_bytes(cfg_, rank_, world_, head_);
  globals_bump_->side_mode = side_grants_;
  globals_bump_->init(globals_cap -
                      (side_grants_ ? globals_side_bytes(cfg_, rank_, world_, head_) : 0));
  // one pinned staging mirror serves every bump (layers and globals build
  // one at a time and exit synced): sized for the largest of them.
  staging_bytes_ = std::max(capacity, globals_cap);
  DGPP_CUDA_OK(cudaHostAlloc(&staging_, staging_bytes_, cudaHostAllocDefault));
  DGPP_CUDA_OK(cudaStreamCreate(&stream_));
  checkpoint_dir_ = checkpoint_dir;
  if (residency_ == GlmResidency::Resident) open_resident_image();
}

// Fail in the constructor, not three minutes into the load: the resident
// footprint is known from the byte formula before a single byte moves.
// The measure is the larger of the device's free memory and the host's
// MemAvailable (the GB10's unified pool is the host's memory; the page
// cache is reclaimable); the headroom covers the staging mirror (2.2 GiB
// measured), the CUDA context and the bus's buffers — 4 GiB, the serving
// plan's figure (2026-10-09; the 8 GiB before it refused the full GLM-5.3
// diagnostic forward on a node with 104 GiB free).
void GlmLayerStream::check_resident_footprint_fits() const {
  size_t footprint = globals_bytes(cfg_, rank_, world_, head_);
  for (int l = 0; l < cfg_.num_hidden_layers; ++l)
    footprint += layer_bytes(cfg_, l, rank_, world_);
  // The MTP draft layer counts only when the model asked for it.
  if (resident_mtp_ && cfg_.mtp_layer() >= 0)
    footprint += layer_bytes(cfg_, cfg_.mtp_layer(), rank_, world_);
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return;
  const size_t available = host_mem_available_bytes();
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  const size_t headroom = static_cast<size_t>(4 * kGiB);
  DGPP_LOG_INFO("glm loader: rank {} resident footprint {:.1f} GiB; device "
                "free {:.1f} of {:.1f} GiB, host available {:.1f} GiB",
                rank_, footprint / kGiB, free_bytes / kGiB, total_bytes / kGiB,
                available / kGiB);
  free_bytes = std::max(free_bytes, available);
  if (footprint + headroom > free_bytes)
    throw std::runtime_error(
        "glm loader: the resident model (" + std::to_string(footprint >> 30) +
        " GiB + 4 GiB headroom) does not fit in the device's free memory (" +
        std::to_string(free_bytes >> 30) +
        " GiB) — free memory on this node, use a larger world, or run in "
        "streaming residency");
}

// ---- the resident image cache ----------------------------------------------

namespace {
std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}
}  // namespace

std::pair<const void*, size_t> GlmLayerStream::resident_layer_span(
    int layer) const {
  if (residency_ != GlmResidency::Resident || layer < 0 ||
      layer >= static_cast<int>(resident_layers_.size()) ||
      resident_layers_[static_cast<size_t>(layer)].layer != layer)
    return {nullptr, 0};
  const GlmLayerBump& b = *resident_bumps_[static_cast<size_t>(layer)];
  return {b.base, b.dev_cursor};
}

void GlmLayerStream::copy_resident_layer(int layer, void* dst) const {
  if (residency_ != GlmResidency::Resident || layer < 0 ||
      layer >= static_cast<int>(resident_bumps_.size()) ||
      !resident_bumps_[static_cast<size_t>(layer)])
    throw std::out_of_range("glm loader: copy_resident_layer of a layer that is not resident");
  resident_bumps_[static_cast<size_t>(layer)]->download(dst);
}

void GlmLayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage() = dir;
}
const std::string& GlmLayerStream::resident_image_dir() {
  return resident_image_dir_storage();
}

uint64_t GlmLayerStream::resident_image_key() const {
  uint64_t h = 1469598103934665603ull;
  h = fnv_mix(h, GlmResidentImage::kFormatVersion);
  h = fnv_mix(h, static_cast<uint64_t>(world_));
  h = fnv_mix(h, static_cast<uint64_t>(rank_));
  h = fnv_mix(h, static_cast<uint64_t>(head_));
  // config.json bytes: every geometry decision the build makes reads it.
  {
    std::ifstream f(std::filesystem::path(checkpoint_dir_) / "config.json",
                    std::ios::binary);
    char c;
    while (f.get(c)) h = (h ^ static_cast<uint8_t>(c)) * 1099511628211ull;
  }
  // The shards, in the loader's (sorted) order: header identity + size.
  for (const auto& shard : shards_) {
    h = fnv_mix(h, shard->header_fold());
    h = fnv_mix(h, shard->map_size());
  }
  return h;
}

void GlmLayerStream::open_resident_image() {
  const std::string& dir = resident_image_dir();
  if (dir.empty()) return;
  const int max_layer =
      cfg_.num_hidden_layers + (cfg_.mtp_layer() >= 0 ? 1 : 0);
  // The checkpoint's weight identity beside the key (ResidentImage::open_for):
  // the key reads config.json and the shard headers, which a refined
  // release can re-ship byte for byte.
  uint64_t payload = 1469598103934665603ull;
  for (const auto& shard : shards_) payload = fnv_mix(payload, shard->payload_sample_fold());
  bool replaced = false;
  try {
    image_ = GlmResidentImage::open_for(dir, resident_image_key(), max_layer, payload, &replaced);
  } catch (const std::exception& e) {
    // A cache that cannot open is a slower start, not a failed one.
    DGPP_LOG_WARN("glm loader: rank {} resident image unavailable ({}) — "
                  "building from the checkpoint",
                  rank_, e.what());
    image_.reset();
    return;
  }
  if (replaced)
    DGPP_LOG_WARN("glm loader: rank {} resident image {} was built from a checkpoint with these shard headers "
                  "but other weights — discarded; rebuilding it from this checkpoint",
                  rank_, image_->path());
  DGPP_LOG_INFO("glm loader: rank {} resident image {} — {}/{} layers present, "
                "{} I/O",
                rank_, image_->path(), image_->present(), image_->layers(),
                image_->direct_io() ? "direct" : "buffered");
}

bool GlmLayerStream::restore_layer_from_image(int layer, GlmLayerBump& bump,
                                              GlmLayerResident& out) {
  // Layout pass: the build with copy=false against a REAL bump hands out
  // the exact grant sequence (hence the exact view pointers) without
  // reading a source byte; jobs and packs are recorded and ignored — the
  // blob already holds their results.
  bump.stage = staging_;
  std::vector<GlmExpectedTensor> table =
      glm_expected_layer_tensors(cfg_, layer);
  std::unordered_map<std::string, const GlmExpectedTensor*> by_name;
  for (const auto& e : table) by_name.emplace(e.name, &e);
  std::vector<DequantJob> jobs;
  std::vector<PackJob> packs;
  std::unordered_map<std::string, const TensorInfo*> no_tensors;
  BuildCtx ctx(cfg_,       table, by_name, bump,  out,
               no_tensors, jobs,  packs,   false, rank_, world_);
  ctx.build_layer(layer);
  const size_t bytes = bump.cursor;
  if (bytes != layer_bytes(cfg_, layer, rank_, world_))
    throw std::runtime_error("glm loader: layout pass drifted from the byte "
                             "formula on layer " + std::to_string(layer));
  if (!image_->try_read_layer(layer, staging_, bytes)) return false;
  sync_load_boundary(reader_, stream_);
  bump.upload(stream_);  // stage == staging_: the blob is the mirror's layout
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  out.bytes = bytes;
  ++image_restored_;
  return true;
}

void GlmLayerStream::capture_layer_to_image(int layer, const GlmLayerBump& bump,
                                            size_t bytes) {
  // The bump is final and the stream is synced (build_layer_into's exit);
  // the staging mirror is free again, so it carries the D2H.
  bump.download(staging_);
  try {
    image_->write_layer(layer, staging_, bytes);
    ++image_captured_;
  } catch (const std::exception& e) {
    DGPP_LOG_WARN("glm loader: rank {} could not capture layer {} to the "
                  "resident image ({}) — cache disabled for this stream",
                  rank_, layer, e.what());
    image_.reset();
  }
}

GlmLayerStream::~GlmLayerStream() {
  if (stream_) cudaStreamDestroy(stream_);
  if (staging_) cudaFreeHost(staging_);
}

size_t GlmLayerStream::layer_capacity() const {
  return capacity_;
}

void GlmLayerStream::release_sources() {
  if (residency_ != GlmResidency::Resident || sources_released_) return;
  sources_released_ = true;
  // The header index points into the shards' TensorInfo records; clear it
  // before the mappings go so nothing can follow a dangling view.
  tensors_.clear();
  const size_t shard_count = shards_.size();
  uint64_t mapped_bytes = 0;
  for (auto& shard : shards_) {
    mapped_bytes += shard->map_size();
    shard->close_mapping(/*drop_page_cache=*/true);
  }
  shards_.clear();
  // The staging mirror stays until the destructor: this release runs at
  // the last layer inside the first forward, where a cudaFreeHost would
  // wait on the bus's persistent kernels for a watchdog period.
  DGPP_LOG_INFO(
      "glm loader: rank {} resident load complete — released {} shard "
      "mappings ({:.1f} GiB) and evicted their page cache; image: {} layers "
      "restored, {} captured",
      rank_, shard_count,
      static_cast<double>(mapped_bytes) / (1024.0 * 1024.0 * 1024.0),
      image_restored_, image_captured_);
}

// The one layer build both residency modes share: the grant sequence,
// the byte accounting, the formula check, and the dequant/pack phases.
// Resident bytes are streaming bytes by construction — the parity
// driver pins this bitwise at every world.
void GlmLayerStream::build_layer_into(int layer, GlmLayerBump& bump,
                                      GlmLayerResident& out) {
  bump.stage = staging_;  // the build writes here; the H2D below lands it
  std::vector<GlmExpectedTensor> table =
      glm_expected_layer_tensors(cfg_, layer);
  std::unordered_map<std::string, const GlmExpectedTensor*> by_name;
  for (const auto& e : table) by_name.emplace(e.name, &e);

  std::vector<DequantJob> jobs;
  std::vector<PackJob> packs;
  BuildCtx ctx(cfg_,    table, by_name, bump,          out,
               tensors_, jobs,  packs,   true,         rank_,   world_);
  ctx.one_pass_sources = residency_ == GlmResidency::Resident;
  ctx.build_layer(layer);

  // Phase one, continued: host-source packs land in the staging mirror
  // alongside the builder's memcpys; then one H2D copy of the whole layer.
  for (const PackJob& j : packs)
    if (!j.src_on_device) run_host_pack(j, bump);
  bump.upload(stream_);

  // Phase two: the dequants (device -> device inside the bump).
  for (const DequantJob& j : jobs)
    launch_fp8_dequant_blocks(j.payload, j.scales, j.out, j.rows, j.cols,
                              stream_);

  // Phase three: device-source packs — strided copies out of bridge
  // buffers the dequants just wrote (the DSA o_proj slice). Then the exit
  // sync: the staging mirror is reusable and the layer is readable.
  for (const PackJob& j : packs)
    if (j.src_on_device) run_device_pack(j, stream_);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));

  // The formula and the allocator share the build code; anything but
  // equality is a bug that must never pass silently. In resident mode the
  // bump was sized BY this formula, so allocation itself is a second,
  // independent enforcement of the same equality.
  const size_t used = bump.cursor;
  const size_t expected_bytes = layer_bytes(cfg_, layer, rank_, world_);
  if (used != expected_bytes)
    throw std::runtime_error(
        "glm loader: byte-formula drift on layer " + std::to_string(layer) +
        ": used " + std::to_string(used) + " != formula " +
        std::to_string(expected_bytes));
  out.bytes = used;
  source_bytes_ += ctx.source_bytes;
  verbatim_bytes_ += ctx.verbatim_bytes;
}

const GlmLayerResident& GlmLayerStream::load_layer(int layer) {
  // RESIDENT mode: a materialized layer is a pure cache hit — the stored
  // view, no storage reads (source_bytes_read cannot grow), and no sync
  // (the streaming contract's sync covers loader WRITES; a cache hit
  // writes nothing). First load materializes into the layer's own
  // exact-formula bump, which stays alive for the stream's lifetime.
  if (residency_ == GlmResidency::Resident) {
    if (layer < 0 || layer >= static_cast<int>(resident_layers_.size()))
      throw std::out_of_range(
          "glm loader: layer index out of range: " + std::to_string(layer));
    GlmLayerResident& slot =
        resident_layers_[static_cast<size_t>(layer)];
    if (slot.layer == layer) return slot;
    if (sources_released_)
      throw std::runtime_error(
          "glm loader: layer " + std::to_string(layer) +
          " was never materialized before release_sources() (load every "
          "layer you need — the MTP draft included — before releasing)");

    auto bump = std::make_unique<GlmLayerBump>();
    bump->side_mode = side_grants_;
    const CountedLayer counted = count_layer(cfg_, layer, rank_, world_);
    bump->init(counted.bytes - (side_grants_ ? counted.side_bytes : 0));
    const bool restored =
        image_ && image_->has_layer(layer) && restore_layer_from_image(layer, *bump, slot);
    if (!restored) {
      bump->reset();
      slot = GlmLayerResident{};
      sync_load_boundary(reader_, stream_);
      build_layer_into(layer, *bump, slot);
      if (image_) capture_layer_to_image(layer, *bump, slot.bytes);
    }
    resident_bumps_[static_cast<size_t>(layer)] = std::move(bump);
    return slot;
  }

  if (resident_.layer == layer) return resident_;

  // Phase one below writes weight bytes into the bump from the CPU — the
  // same region the previously loaded layer's kernels may still be
  // reading (this loader was first exercised mid-forward by the M4
  // diagnostic model; before that, callers always loaded with the device
  // idle). The boundary sync waits for exactly those readers — see
  // sync_load_boundary for why it must not be device-wide in a
  // one-process multi-rank world.
  sync_load_boundary(reader_, stream_);

  layer_bump_->reset();
  resident_ = GlmLayerResident{};
  build_layer_into(layer, *layer_bump_, resident_);
  return resident_;
}

// The digest note: u64 words [format, tensors, bytes, globals, n, layer...].
namespace {
constexpr uint64_t kDigestNoteFormat = 0x4447505044494731ull;  // "DGPPDIG1"
constexpr const char* kDigestNoteName = "digest";

std::vector<uint64_t> serialize_digest(const GlmReplicatedDigest& d) {
  std::vector<uint64_t> w = {kDigestNoteFormat, d.tensors, d.bytes, d.globals,
                             d.layer.size()};
  w.insert(w.end(), d.layer.begin(), d.layer.end());
  return w;
}

bool deserialize_digest(const std::vector<uint64_t>& w, size_t layers,
                        GlmReplicatedDigest& d) {
  if (w.size() != 5 + layers || w[0] != kDigestNoteFormat || w[4] != layers)
    return false;
  d.tensors = w[1];
  d.bytes = w[2];
  d.globals = w[3];
  d.layer.assign(w.begin() + 5, w.end());
  return true;
}
}  // namespace

GlmReplicatedDigest GlmLayerStream::hash_replicated() const {
  if (sources_released_)
    throw std::runtime_error(
        "glm loader: hash_replicated after the checkpoint sources were "
        "released — the digest is a BOOT check; take it before the first "
        "forward");
  // An image carries its digest: the layers it serves and the digest it
  // reports derive from the same checkpoint bytes under the same key, so
  // trusting one is trusting the other (a rebuilt layer gets the same
  // digest — the digest is over the SOURCE). Missing note: compute from
  // the shards and publish it for the next start.
  const size_t layers = static_cast<size_t>(
      cfg_.num_hidden_layers + (cfg_.mtp_layer() >= 0 ? 1 : 0));
  if (image_) {
    std::vector<uint64_t> words(5 + layers);
    GlmReplicatedDigest d;
    if (image_->read_note(kDigestNoteName, words.data(),
                          words.size() * sizeof(uint64_t)) &&
        deserialize_digest(words, layers, d)) {
      DGPP_LOG_INFO("glm loader: rank {} boot digest from the resident image "
                    "({} tensors, {:.2f} GiB)",
                    rank_, d.tensors,
                    static_cast<double>(d.bytes) / (1024.0 * 1024.0 * 1024.0));
      return d;
    }
  }
  GlmReplicatedDigest d = compute_replicated_digest();
  if (image_) {
    try {
      const std::vector<uint64_t> words = serialize_digest(d);
      image_->write_note(kDigestNoteName, words.data(),
                         words.size() * sizeof(uint64_t));
    } catch (const std::exception& e) {
      DGPP_LOG_WARN("glm loader: rank {} could not publish the digest note ({})",
                    rank_, e.what());
    }
  }
  return d;
}

GlmReplicatedDigest GlmLayerStream::compute_replicated_digest() const {
  const auto lookup = [this](const std::string& name) -> const TensorInfo& {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("glm loader: tensor not in checkpoint: " +
                               name);
    return *it->second;
  };
  GlmReplicatedDigest d;
  const int max_layer =
      cfg_.num_hidden_layers + (cfg_.mtp_layer() >= 0 ? 1 : 0);
  d.layer.assign(static_cast<size_t>(max_layer), 0);
  for (int l = 0; l < max_layer; ++l) {
    uint64_t sum = 0;
    for (const GlmExpectedTensor& e :
         glm_expected_layer_tensors(cfg_, l)) {
      if (!is_replicated(e)) continue;
      const TensorInfo& t = lookup(e.name);
      const size_t bytes = e.nbytes();
      if (t.numel() * dtype_size(t.dtype) != bytes)
        throw std::runtime_error(
            "glm loader: replicated tensor shape drift on '" + e.name +
            "' (expected table and checkpoint header disagree)");
      sum += fnv1a(t.data, bytes, fnv1a(e.name.data(), e.name.size(),
                                        1469598103934665603ull));
      d.bytes += bytes;
      ++d.tensors;
    }
    d.layer[static_cast<size_t>(l)] = sum;
  }
  static const char* kGlobals[] = {
      "model.language_model.embed_tokens.weight",
      "lm_head.weight",
      "model.language_model.norm.weight",
  };
  uint64_t g = 0;
  for (const char* name : kGlobals) {
    const TensorInfo& t = lookup(name);
    const size_t bytes = t.numel() * dtype_size(t.dtype);
    g += fnv1a(t.data, bytes,
               fnv1a(name, std::strlen(name), 1469598103934665603ull));
    d.bytes += bytes;
    ++d.tensors;
  }
  d.globals = g;
  return d;
}

const GlmGlobalsResident& GlmLayerStream::load_globals() {
  if (globals_.embed) return globals_;
  if (sources_released_)
    throw std::runtime_error(
        "glm loader: load_globals after the checkpoint sources were released "
        "(globals are loaded at construction; a release_globals/load_globals "
        "cycle is a streaming-mode pattern)");
  // Same entry-sync discipline as load_layer (phase-one CPU writes; the
  // boundary sync covers exactly the globals bump's readers).
  sync_load_boundary(reader_, stream_);
  globals_bump_->reset();
  globals_bump_->stage = staging_;
  globals_ = GlmGlobalsResident{};

  auto copy_global = [&](const std::string& name, bool packable = false) -> uint16_t* {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("glm loader: global tensor missing: " + name);
    const TensorInfo& t = *it->second;
    uint16_t* dst = static_cast<uint16_t*>(packable ? globals_bump_->alloc_side(t.nbytes())
                                                    : globals_bump_->alloc(t.nbytes()));
    std::memcpy(globals_bump_->host(dst), t.data, t.nbytes());
    source_bytes_ += t.nbytes();
    verbatim_bytes_ += t.nbytes();
    return dst;
  };
  globals_.embed = copy_global("model.language_model.embed_tokens.weight");
  if (head_ == GlmHeadSharding::VocabSharded) {
    // This rank's contiguous vocab slice only — the slice bytes are not
    // verbatim: with a sharded head the lm_head partition across ranks
    // (the §5.2 byte reconcile treats them like the sharded classes).
    auto it = tensors_.find("lm_head.weight");
    if (it == tensors_.end() || !it->second)
      throw std::runtime_error("glm loader: global tensor missing: lm_head.weight");
    const TensorInfo& t = *it->second;
    const auto [begin, count] = lm_head_slice(cfg_, rank_, world_);
    const size_t row_bytes = static_cast<size_t>(cfg_.hidden_size) * 2;
    uint16_t* dst = static_cast<uint16_t*>(
        bf12_shape_ok(count, cfg_.hidden_size)
            ? globals_bump_->alloc_side(static_cast<size_t>(count) * row_bytes)
            : globals_bump_->alloc(static_cast<size_t>(count) * row_bytes));
    std::memcpy(globals_bump_->host(dst),
                static_cast<const uint8_t*>(t.data) +
                    static_cast<size_t>(begin) * row_bytes,
                static_cast<size_t>(count) * row_bytes);
    source_bytes_ += static_cast<size_t>(count) * row_bytes;
    globals_.lm_head = dst;
    globals_.lm_vocab_begin = begin;
    globals_.lm_vocab_count = count;
  } else {
    globals_.lm_head = copy_global("lm_head.weight", bf12_shape_ok(cfg_.vocab_size, cfg_.hidden_size));
    globals_.lm_vocab_begin = 0;
    globals_.lm_vocab_count = cfg_.vocab_size;
  }
  globals_.final_norm = copy_global("model.language_model.norm.weight");
  globals_.bytes = globals_bump_->cursor;

  const size_t expected_bytes = globals_bytes(cfg_, rank_, world_, head_);
  if (globals_.bytes != expected_bytes)
    throw std::runtime_error(
        "glm loader: globals byte-formula drift: used " +
        std::to_string(globals_.bytes) + " != formula " +
        std::to_string(expected_bytes));
  globals_bump_->upload(stream_);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  return globals_;
}

// The bf16 bytes of a packed matrix go back to the device (bf12-only
// residency): `weight` is a side grant of the layer's bump — or of the
// globals' (layer < 0). Returns the bytes freed; 0 when the matrix was not
// granted aside (another mode, or outside the packing contract).
size_t GlmLayerStream::release_packed(int layer, const void* weight) {
  if (!side_grants_) return 0;
  if (layer < 0) return globals_bump_->release_side(weight);
  if (layer >= static_cast<int>(resident_bumps_.size()) ||
      !resident_bumps_[static_cast<size_t>(layer)])
    return 0;
  return resident_bumps_[static_cast<size_t>(layer)]->release_side(weight);
}

void GlmLayerStream::release_layer() {
  // RESIDENT mode: the contract is that materialized layers stay resident
  // for the stream's lifetime — a no-op, so existing call sites (the
  // model's preconstruct pass) stay correct unchanged. The views remain
  // valid; a streaming release after resident loads would be a silent
  // use-after-free factory, which is exactly what this refuses to be.
  if (residency_ == GlmResidency::Resident) return;
  resident_ = GlmLayerResident{};
  layer_bump_->reset();
}

void GlmLayerStream::release_globals() {
  globals_ = GlmGlobalsResident{};
  globals_bump_->reset();
}

}  // namespace dgpp
