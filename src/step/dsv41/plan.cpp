// DeepSeek-V4.1-Flash's stream plan generator (step/dsv41/plan.hpp).
#include "step/dsv41/plan.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "kernels/csa2.hpp"
#include "kernels/latent_format.hpp"
#include "models/dsv41/csa2_state.hpp"

namespace dgpp::step {

namespace {

constexpr uint64_t kGrantAlign = 256;  // loaders/weight_build.hpp kWeightAllocAlign
uint64_t align_grant(uint64_t b) { return (b + kGrantAlign - 1) / kGrantAlign * kGrantAlign; }

// The design chain's modeled latencies (MOONSHOT-1.md section 1.3, the
// phase-1 probe's design chain): the attention site's hyper-connection and
// norm, the selection, the fold after a sub-layer, the MoE site's update,
// hyper-connection and router, and the ordered accumulation as a worker
// phase. The fused chain of section 2.9 replaces them with measurements.
constexpr float kAttnInUs = 8.0f;
constexpr float kSelectUs = 22.0f;
constexpr float kFoldUs = 12.0f;
constexpr float kFfnInUs = 12.0f;
constexpr float kAccumulateUs = 8.0f;

// Emits a layer's ranges from the resident record's pointers.
struct Emitter {
  const Dsv41TextConfig& cfg;
  const Dsv41StepShape& shape;
  const void* base;
  LayerPlan& lp;

  uint64_t off(const void* p) const {
    return static_cast<uint64_t>(static_cast<const char*>(p) - static_cast<const char*>(base));
  }
  int phase(const char* name, PhaseKind kind, int resolves = -1, float latency_us = 0, float fold_after_us = 0) {
    lp.phases.push_back(Phase{name, kind, resolves, latency_us, fold_after_us});
    return static_cast<int>(lp.phases.size()) - 1;
  }
  int add(Range r) {
    lp.ranges.push_back(std::move(r));
    return static_cast<int>(lp.ranges.size()) - 1;
  }

  // A resident vector of n elements, staged in shared memory.
  int vec(const std::string& name, const void* p, int64_t n, Format f, Consumer c, int ph, Dependency dep = Dependency::Fixed,
          uint32_t reads = 1, uint64_t read_bytes = 0) {
    if (!p) return -1;
    Range r;
    r.name = name;
    r.offset = off(p);
    r.bytes = static_cast<uint64_t>(n) * (f == Format::F32 ? 4 : 2);
    r.read_bytes = read_bytes;
    r.reads = reads;
    r.format = f;
    r.staging = Staging::Shared;
    r.dep = dep;
    r.consumer = c;
    r.phase = ph;
    r.rows = 1;
    r.cols = n;
    return add(std::move(r));
  }
  // A resident bf16 matrix [rows, cols], streamed through registers.
  int bf16(const std::string& name, const void* p, int64_t rows, int64_t cols, Consumer c, int ph,
           Dependency dep = Dependency::Fixed, uint32_t reads = 1, uint64_t read_bytes = 0) {
    if (!p) return -1;
    Range r;
    r.name = name;
    r.offset = off(p);
    r.bytes = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols) * 2;
    r.read_bytes = read_bytes;
    r.reads = reads;
    r.format = Format::Bf16;
    r.staging = Staging::Register;
    r.dep = dep;
    r.consumer = c;
    r.phase = ph;
    r.rows = rows;
    r.cols = cols;
    return add(std::move(r));
  }
  // An fp8 matrix: the e4m3 payload and its F32 block-scale grid.
  void fp8(const std::string& name, const GlmQuantMatrix& m, Consumer c, int ph, int group = -1, int matrix = -1) {
    if (!m.payload) return;
    if (!m.scales) throw std::logic_error("dsv41 plan: fp8 matrix '" + name + "' without an F32 scale grid");
    const Dependency dep = group >= 0 ? Dependency::Router : Dependency::Fixed;
    Range p;
    p.name = name + ".payload";
    p.offset = off(m.payload);
    p.bytes = static_cast<uint64_t>(m.rows) * static_cast<uint64_t>(m.cols);
    p.format = Format::Fp8E4m3;
    p.staging = Staging::Register;
    p.dep = dep;
    p.consumer = c;
    p.phase = ph;
    p.rows = m.rows;
    p.cols = m.cols;
    p.group = group;
    p.matrix = matrix;
    const int pi = add(std::move(p));
    const int64_t sr = (m.rows + m.scale_block_rows - 1) / m.scale_block_rows;
    const int64_t sc = (m.cols + m.scale_block_cols - 1) / m.scale_block_cols;
    Range s;
    s.name = name + ".scales";
    s.offset = off(m.scales);
    s.bytes = static_cast<uint64_t>(sr) * static_cast<uint64_t>(sc) * 4;
    s.format = Format::ScaleF32;
    s.staging = Staging::Shared;
    s.dep = dep;
    s.consumer = c;
    s.phase = ph;
    s.rows = sr;
    s.cols = sc;
    s.scale_of = pi;
    s.group = group;
    s.matrix = matrix;
    add(std::move(s));
  }
  // An MXFP4 matrix: the e2m1 payload and its e8m0 scales, one a 32-code block.
  void fp4(const std::string& name, const GlmFp4Matrix& m, Consumer c, int ph, int group, int matrix) {
    if (!m.payload) return;
    if (!m.mxfp4()) throw std::logic_error("dsv41 plan: fp4 matrix '" + name + "' is not MXFP4");
    const Dependency dep = group >= 0 ? Dependency::Router : Dependency::Fixed;
    Range p;
    p.name = name + ".payload";
    p.offset = off(m.payload);
    p.bytes = m.payload_bytes();
    p.format = Format::Fp4E2m1;
    p.staging = Staging::Register;
    p.dep = dep;
    p.consumer = c;
    p.phase = ph;
    p.rows = m.rows;
    p.cols = m.cols;
    p.group = group;
    p.matrix = matrix;
    const int pi = add(std::move(p));
    Range s;
    s.name = name + ".scales";
    s.offset = off(m.scales);
    s.bytes = m.scale_bytes();
    s.format = Format::ScaleE8m0;
    s.staging = Staging::Register;
    s.dep = dep;
    s.consumer = c;
    s.phase = ph;
    s.rows = m.rows;
    s.cols = m.scale_cols();
    s.scale_of = pi;
    s.group = group;
    s.matrix = matrix;
    add(std::move(s));
  }
  // An EXL3 matrix: the trellis tiles, suh over the columns, svh over the rows.
  void exl3(const std::string& name, const Exl3Matrix& m, Consumer c, int ph, int group = -1, int matrix = -1) {
    if (!m.tiles) return;
    const Dependency dep = group >= 0 ? Dependency::Router : Dependency::Fixed;
    Range t;
    t.name = name + ".trellis";
    t.offset = off(m.tiles);
    t.bytes = m.tiles_bytes();
    t.format = Format::Exl3Trellis;
    t.staging = Staging::Register;
    t.dep = dep;
    t.consumer = c;
    t.phase = ph;
    t.rows = m.rows;
    t.cols = m.cols;
    t.group = group;
    t.matrix = matrix;
    add(std::move(t));
    for (int which = 0; which < 2; ++which) {
      Range v;
      v.name = name + (which ? ".svh" : ".suh");
      v.offset = off(which ? m.svh : m.suh);
      v.bytes = which ? m.svh_bytes() : m.suh_bytes();
      v.format = Format::Bf16;
      v.staging = Staging::Shared;
      v.dep = dep;
      v.consumer = c;
      v.phase = ph;
      v.rows = 1;
      v.cols = which ? m.rows : m.cols;
      v.group = group;
      v.matrix = matrix;
      add(std::move(v));
    }
  }
  // Rows outside the resident weights: `rows_per_read` rows of `row_bytes`
  // a read, `reads` reads a step.
  void rows(const std::string& name, Source source, Format f, uint64_t row_bytes, int64_t rows_per_read, uint32_t reads,
            Dependency dep, Consumer c, int ph) {
    if (rows_per_read <= 0 || reads == 0) return;
    Range r;
    r.name = name;
    r.source = source;
    r.bytes = row_bytes * static_cast<uint64_t>(rows_per_read);
    r.read_bytes = r.bytes;
    r.reads = reads;
    r.format = f;
    r.staging = source == Source::Host ? Staging::Direct : Staging::Register;
    r.dep = dep;
    r.consumer = c;
    r.phase = ph;
    r.rows = rows_per_read;
    r.cols = static_cast<int64_t>(row_bytes);
    add(std::move(r));
  }
};

}  // namespace

double dsv41_expected_distinct(int experts, int rows, int k) {
  if (experts <= 0 || rows <= 0 || k <= 0) return 0.0;
  const double e = experts;
  return e * (1.0 - std::pow(1.0 - 1.0 / e, static_cast<double>(rows) * k));
}

LayerPlan dsv41_layer_plan(const Dsv41TextConfig& cfg, const Dsv41LocalGeometry& geo, const Dsv41LayerResident& r,
                           const void* base, uint64_t span_bytes, const Dsv41StepShape& shape) {
  LayerPlan lp;
  lp.layer = r.layer;
  lp.span_bytes = span_bytes;
  const int layer = r.layer;
  const bool draft = cfg.is_draft(layer);
  const int rows = draft ? shape.draft_rows : shape.rows;
  const int top_k = draft ? cfg.dspark_num_experts_per_tok : cfg.num_experts_per_tok;
  lp.groups = r.moe.n_experts;
  const double distinct = draft ? shape.distinct_draft : shape.distinct_main;
  lp.expected_groups = distinct >= 0 ? std::min<double>(distinct, lp.groups)
                                     : dsv41_expected_distinct(lp.groups, rows, top_k);
  const auto reads = static_cast<uint32_t>(std::max(rows, 0));
  Emitter em{cfg, shape, base, lp};
  const int64_t H = cfg.hidden_size;
  const int64_t hc_rows = cfg.hc_coeff_rows();

  // ---- the draft stage's input projection (stage 0) --------------------------
  if (draft && cfg.draft_stage(layer) == 0) {
    const int ph = em.phase("draft_in", PhaseKind::Stream);
    em.fp8("main_proj", r.draft.main_proj, Consumer::Projection, ph);
    em.exl3("main_proj", r.draft.main_proj_x, Consumer::Projection, ph);
    em.vec("main_norm", r.draft.main_norm, H, Format::Bf16, Consumer::Norm, ph);
  }
  // ---- Engram: the hashed rows, their projection, the gate ---------------------
  if (r.engram.present) {
    const int ph = em.phase("engram", PhaseKind::Stream, -1, 0, kFoldUs);
    const int64_t hd = cfg.engram_head_dim;
    const int64_t rows_local = static_cast<int64_t>(cfg.engram_max_ngram_size - 1) * geo.engram_heads_local;
    em.rows("engram.table_rows", Source::Host, Format::Fp8E4m3, static_cast<uint64_t>(rows_local) * (hd + hd / 32), 1,
            reads, Dependency::Token, Consumer::TableGather, ph);
    em.fp8("engram.wkv", r.engram.wkv, Consumer::Projection, ph);
    em.exl3("engram.wkv", r.engram.wkv_x, Consumer::Projection, ph);
    em.bf16("engram.q_weight", r.engram.q_weight, cfg.hc_mult, H, Consumer::Projection, ph);
    em.bf16("engram.k_weight", r.engram.k_weight, cfg.hc_mult, H, Consumer::Projection, ph);
  }
  // ---- the attention site's hyper-connection and norm ------------------------
  {
    const int ph = em.phase("attn_in", PhaseKind::Latency, -1, kAttnInUs);
    em.bf16("hc_attn_fn", r.mhc.attn_fn, hc_rows, 4 * H, Consumer::HyperConn, ph);
    em.vec("hc_attn_base", r.mhc.attn_base, hc_rows, Format::F32, Consumer::HyperConn, ph);
    em.vec("hc_attn_scale", r.mhc.attn_scale, 3, Format::F32, Consumer::HyperConn, ph);
    em.vec("attn_norm", r.attn_norm, H, Format::Bf16, Consumer::Norm, ph);
  }
  // ---- q, kv, the indexer and the compressor -----------------------------------
  const Dsv41AttnResident& a = r.attn;
  {
    const int ph = em.phase("qkv", PhaseKind::Stream);
    em.fp8("attn.wq_a", a.wq_a, Consumer::Projection, ph);
    em.exl3("attn.wq_a", a.wq_a_x, Consumer::Projection, ph);
    em.vec("attn.q_norm", a.q_norm, cfg.q_lora_rank, Format::Bf16, Consumer::Norm, ph);
    em.fp8("attn.wq_b", a.wq_b, Consumer::Projection, ph);
    em.exl3("attn.wq_b", a.wq_b_x, Consumer::Projection, ph);
    em.fp8("attn.wkv", a.wkv, Consumer::Projection, ph);
    em.exl3("attn.wkv", a.wkv_x, Consumer::Projection, ph);
    em.vec("attn.kv_norm", a.kv_norm, cfg.head_dim, Format::Bf16, Consumer::Norm, ph);
    em.vec("attn.attn_sink", a.attn_sink, a.local_heads, Format::F32, Consumer::Attention, ph);
    if (a.index_source()) {
      em.fp8("attn.indexer.wq_b", a.idx_wq_b, Consumer::Indexer, ph);
      em.exl3("attn.indexer.wq_b", a.idx_wq_b_x, Consumer::Indexer, ph);
      em.bf16("attn.indexer.weights_proj", a.idx_wp, cfg.index_n_heads, H, Consumer::Indexer, ph);
      if (a.kv_source()) {
        em.bf16("attn.indexer.wk", a.idx_wk, cfg.index_head_dim, cfg.head_dim, Consumer::Indexer, ph);
        em.exl3("attn.indexer.wk", a.idx_wk_x, Consumer::Indexer, ph);
        em.vec("attn.indexer.k_norm", a.idx_k_norm, cfg.index_head_dim, Format::Bf16, Consumer::Indexer, ph);
      }
    }
    if (a.kv_source()) {
      em.bf16("attn.compressor.wkv", a.comp_wkv, cfg.head_dim, H, Consumer::Compressor, ph);
      em.exl3("attn.compressor.wkv", a.comp_wkv_x, Consumer::Compressor, ph);
      em.bf16("attn.compressor.wgate", a.comp_wgate, cfg.head_dim, H, Consumer::Compressor, ph);
      em.exl3("attn.compressor.wgate", a.comp_wgate_x, Consumer::Compressor, ph);
      em.vec("attn.compressor.norm", a.comp_norm, cfg.head_dim, Format::Bf16, Consumer::Norm, ph);
    }
  }
  // ---- the selection: an index source scores the request's keys -----------------
  if (!draft && cfg.is_index_source(layer)) {
    const int ph = em.phase("select", PhaseKind::Latency, static_cast<int>(Dependency::Selection), kSelectUs);
    int64_t scanned = shape.scanned_keys;
    if (cfg.uses_candidates(layer)) scanned = std::min<int64_t>(scanned, cfg.candidate_pool_entries());
    em.rows("attn.index_keys", Source::Pool, Format::IndexKey, static_cast<uint64_t>(kCsa2IndexDim) + sizeof(float), scanned,
            reads, Dependency::Token, Consumer::Indexer, ph);
  }
  // ---- the attention over the window and the selected entries -------------------
  {
    const int ph = em.phase("attend", PhaseKind::Stream);
    em.rows("attn.window_rows", Source::Pool, Format::KvFp8Block,
            latent_row_bytes(Csa2StatePool::kRingFormat, kCsa2Latent), cfg.sliding_window, reads, Dependency::Token,
            Consumer::Attention, ph);
    if (!draft && cfg.compress_ratio(layer) > 0)
      em.rows("attn.selected_entries", Source::Pool, Format::KvFp4Block,
              latent_row_bytes(Csa2StatePool::kMainFormat, kCsa2Latent), cfg.index_topk, reads, Dependency::Selection,
              Consumer::Attention, ph);
  }
  // ---- the output projections, then the fold -----------------------------------
  {
    const int ph = em.phase("attn_out", PhaseKind::Stream, -1, 0, kFoldUs);
    em.fp8("attn.wo_a", a.wo_a, Consumer::Projection, ph);
    for (size_t g = 0; g < a.wo_a_x.size(); ++g)
      em.exl3("attn.wo_a.slice." + std::to_string(a.group_begin + static_cast<int>(g)), a.wo_a_x[g], Consumer::Projection, ph);
    em.fp8("attn.wo_b", a.wo_b, Consumer::Projection, ph);
    em.exl3("attn.wo_b", a.wo_b_x, Consumer::Projection, ph);
  }
  // ---- the MoE site's hyper-connection and norm --------------------------------
  {
    const int ph = em.phase("ffn_in", PhaseKind::Latency, -1, kFfnInUs);
    em.bf16("hc_ffn_fn", r.mhc.ffn_fn, hc_rows, 4 * H, Consumer::HyperConn, ph);
    em.vec("hc_ffn_base", r.mhc.ffn_base, hc_rows, Format::F32, Consumer::HyperConn, ph);
    em.vec("hc_ffn_scale", r.mhc.ffn_scale, 3, Format::F32, Consumer::HyperConn, ph);
    em.vec("ffn_norm", r.ffn_norm, H, Format::Bf16, Consumer::Norm, ph);
  }
  // ---- the router, whose run list releases the experts --------------------------
  const Dsv41MoeResident& m = r.moe;
  {
    const int ph = em.phase("router", PhaseKind::Stream, static_cast<int>(Dependency::Router));
    em.bf16("ffn.gate.weight", m.router, m.n_experts, H, Consumer::Router, ph);
    em.vec("ffn.gate.bias", m.router_bias, m.n_experts, Format::F32, Consumer::Router, ph);
  }
  // ---- the experts: gate and up, then down, then the ordered accumulation ----------
  {
    const int gate_up = em.phase("gate_up", PhaseKind::Stream);
    const int down = em.phase("down", PhaseKind::Stream);
    for (int e = 0; e < m.n_experts; ++e) {
      const std::string ep = "ffn.experts." + std::to_string(e) + ".";
      if (m.exl3()) {
        em.exl3(ep + "w1", m.experts_x[static_cast<size_t>(e) * 3 + 0], Consumer::ExpertGateUp, gate_up, e, 0);
        em.exl3(ep + "w3", m.experts_x[static_cast<size_t>(e) * 3 + 1], Consumer::ExpertGateUp, gate_up, e, 1);
        em.exl3(ep + "w2", m.experts_x[static_cast<size_t>(e) * 3 + 2], Consumer::ExpertDown, down, e, 2);
      } else {
        em.fp4(ep + "w1", m.expert(e, 0), Consumer::ExpertGateUp, gate_up, e, 0);
        em.fp4(ep + "w3", m.expert(e, 1), Consumer::ExpertGateUp, gate_up, e, 1);
        em.fp4(ep + "w2", m.expert(e, 2), Consumer::ExpertDown, down, e, 2);
      }
    }
    em.fp8("ffn.shared_experts.w1", m.shared[0], Consumer::SharedGateUp, gate_up, -1, 0);
    em.fp8("ffn.shared_experts.w3", m.shared[1], Consumer::SharedGateUp, gate_up, -1, 1);
    em.fp8("ffn.shared_experts.w2", m.shared[2], Consumer::SharedDown, down, -1, 2);
    em.exl3("ffn.shared_experts.w1", m.shared_x[0], Consumer::SharedGateUp, gate_up, -1, 0);
    em.exl3("ffn.shared_experts.w3", m.shared_x[1], Consumer::SharedGateUp, gate_up, -1, 1);
    em.exl3("ffn.shared_experts.w2", m.shared_x[2], Consumer::SharedDown, down, -1, 2);
    em.phase("accumulate", PhaseKind::Latency, -1, kAccumulateUs, kFoldUs);
  }
  // ---- the draft's head (the last stage) ---------------------------------------
  if (draft && cfg.draft_stage(layer) == cfg.num_nextn_predict_layers - 1) {
    const int ph = em.phase("draft_head", PhaseKind::Stream);
    const int64_t rank_dim = cfg.dspark_markov_rank;
    em.vec("norm", r.draft.norm, H, Format::Bf16, Consumer::Norm, ph);
    // The Markov embedding row of each block row's token, the Markov head's
    // rows of this rank's vocabulary slice.
    em.bf16("markov_head.embed", r.draft.markov_embed, cfg.vocab_size, rank_dim, Consumer::DraftHead, ph, Dependency::Token,
            reads, static_cast<uint64_t>(rank_dim) * 2);
    em.bf16("markov_head.head", r.draft.markov_head, cfg.vocab_size, rank_dim, Consumer::DraftHead, ph, Dependency::Fixed, 1,
            static_cast<uint64_t>(geo.lm_vocab_count) * static_cast<uint64_t>(rank_dim) * 2);
    em.vec("confidence_head.proj", r.draft.confidence, H + rank_dim, Format::F32, Consumer::DraftHead, ph);
  }
  return lp;
}

LayerPlan dsv41_globals_plan(const Dsv41TextConfig& cfg, const Dsv41LocalGeometry& geo, const Dsv41StepShape& shape) {
  LayerPlan lp;
  lp.layer = -1;
  lp.span_bytes = Dsv41LoaderFamily::globals_bytes(cfg, geo.rank, geo.world, shape.head);
  const int64_t H = cfg.hidden_size;
  const uint32_t tokens = static_cast<uint32_t>(std::max(shape.rows, 0) + std::max(shape.draft_rows, 0));
  uint64_t cursor = 0;
  auto grant = [&](uint64_t bytes) {
    const uint64_t at = cursor;
    cursor = align_grant(cursor + bytes);
    return at;
  };
  // Grant order: build_globals (models/dsv41/loader.cpp).
  const int embed_ph = static_cast<int>(lp.phases.size());
  lp.phases.push_back(Phase{"embed", PhaseKind::Latency, -1});
  {
    Range r;
    r.name = "embed.weight";
    r.bytes = static_cast<uint64_t>(geo.embed_vocab_count) * H * 2;
    r.offset = grant(r.bytes);
    r.read_bytes = static_cast<uint64_t>(H) * 2;
    r.reads = tokens;
    r.format = Format::Bf16;
    r.staging = Staging::Shared;
    r.dep = Dependency::Token;
    r.consumer = Consumer::Embedding;
    r.phase = embed_ph;
    r.rows = geo.embed_vocab_count;
    r.cols = H;
    lp.ranges.push_back(std::move(r));
  }
  const int head_ph = static_cast<int>(lp.phases.size());
  lp.phases.push_back(Phase{"head", PhaseKind::Stream, -1});
  {
    Range r;
    r.name = "norm.weight";
    r.bytes = static_cast<uint64_t>(H) * 2;
    r.offset = grant(r.bytes);
    r.format = Format::Bf16;
    r.staging = Staging::Shared;
    r.consumer = Consumer::Norm;
    r.phase = head_ph;
    r.rows = 1;
    r.cols = H;
    lp.ranges.push_back(std::move(r));
  }
  const auto head_reads = static_cast<uint32_t>(std::max(shape.head_reads, 0));
  if (cfg.weights_exl3) {
    Exl3Matrix m;
    m.rows = geo.lm_vocab_count;
    m.cols = H;
    m.bits = cfg.exl3.head;
    const char* names[3] = {"head.trellis", "head.suh", "head.svh"};
    const uint64_t sizes[3] = {m.tiles_bytes(), m.suh_bytes(), m.svh_bytes()};
    const Format formats[3] = {Format::Exl3Trellis, Format::Bf16, Format::Bf16};
    for (int i = 0; i < 3; ++i) {
      Range r;
      r.name = names[i];
      r.bytes = sizes[i];
      r.offset = grant(r.bytes);
      r.reads = head_reads;
      r.format = formats[i];
      r.staging = i == 0 ? Staging::Register : Staging::Shared;
      r.consumer = Consumer::Head;
      r.phase = head_ph;
      r.rows = i == 2 ? 1 : m.rows;
      r.cols = i == 1 ? m.cols : (i == 2 ? m.rows : m.cols);
      if (i == 1) r.rows = 1;
      lp.ranges.push_back(std::move(r));
    }
  } else {
    // Under engine.bf16_weights bf12 the model packs the head into its
    // 12-bit companion at boot and releases these bf16 bytes (the span
    // keeps the grant): a read is the companion's three quarters.
    Range r;
    r.name = "head.weight";
    r.bytes = static_cast<uint64_t>(geo.lm_vocab_count) * H * 2;
    r.offset = grant(r.bytes);
    r.reads = head_reads;
    r.format = shape.bf12_head ? Format::Bf12 : Format::Bf16;
    if (shape.bf12_head) r.read_bytes = r.bytes / 4 * 3;
    r.staging = Staging::Register;
    r.consumer = Consumer::Head;
    r.phase = head_ph;
    r.rows = geo.lm_vocab_count;
    r.cols = H;
    lp.ranges.push_back(std::move(r));
  }
  return lp;
}

StepPlan dsv41_step_plan(const Dsv41TextConfig& cfg, int rank, int world, const Dsv41StepShape& shape) {
  StepPlan plan;
  plan.family = "dsv41";
  plan.rank = rank;
  plan.world = world;
  plan.layout_version = 0;
  const Dsv41LocalGeometry geo = Dsv41LocalGeometry::from_config(cfg, rank, world, shape.head);
  // The layout pass's base: any address, nothing is read or written there.
  void* base = reinterpret_cast<void*>(static_cast<uintptr_t>(1) << 40);
  const int layers = shape.draft_rows > 0 ? cfg.max_layer() : cfg.num_hidden_layers;
  plan.layers.reserve(static_cast<size_t>(layers));
  for (int l = 0; l < layers; ++l) {
    Dsv41LayerResident r;
    const size_t bytes = Dsv41LayerStream::layout_layer(cfg, l, rank, world, base, r);
    plan.layers.push_back(dsv41_layer_plan(cfg, geo, r, base, bytes, shape));
  }
  plan.globals = dsv41_globals_plan(cfg, geo, shape);
  return plan;
}

}  // namespace dgpp::step
