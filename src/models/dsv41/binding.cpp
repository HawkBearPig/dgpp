#include "models/dsv41/binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<Dsv41ExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         Dsv41WeightClass cls, int layer, int expert = -1,
         Dsv41TensorRole role = Dsv41TensorRole::Plain) {
  out.push_back(Dsv41ExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, expert, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              Dsv41WeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}
void add_f32(TensorList& out, const std::string& name, std::vector<int64_t> shape,
             Dsv41WeightClass cls, int layer) {
  add(out, name, DType::F32, std::move(shape), cls, layer);
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// One fp8 [rows, cols] matrix: the e4m3 payload and the e8m0 block scales.
void add_fp8(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
             Dsv41WeightClass cls, int layer, int block, int expert = -1) {
  add(out, base + ".weight", DType::F8_E4M3, {rows, cols}, cls, layer, expert, Dsv41TensorRole::Fp8Payload);
  add(out, base + ".scale", DType::F8_E8M0, {ceil_div(rows, block), ceil_div(cols, block)}, cls, layer, expert,
      Dsv41TensorRole::Fp8Scale);
}

// One MXFP4 [rows, cols] matrix: the packed nibbles and the per-32 e8m0 scales.
void add_fp4(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
             Dsv41WeightClass cls, int layer, int block, int expert) {
  if (cols % block != 0 || cols % 2 != 0)
    throw std::invalid_argument("dsv41 binding: MXFP4 K must be a multiple of the block on " + base);
  add(out, base + ".weight", DType::I8, {rows, cols / 2}, cls, layer, expert, Dsv41TensorRole::Fp4Payload);
  add(out, base + ".scale", DType::F8_E8M0, {rows, cols / block}, cls, layer, expert, Dsv41TensorRole::Fp4Scale);
}

// One EXL3 matrix with input width K and output width N at `bits`.
void add_exl3(TensorList& out, const std::string& base, int64_t K, int64_t N, int bits, Dsv41WeightClass cls,
              int layer, int expert = -1) {
  if (K % 128 != 0 || N % 128 != 0 || bits < 1 || bits > 8)
    throw std::invalid_argument("dsv41 binding: EXL3 K and N must be multiples of 128 at a width 1..8 on " + base);
  add(out, base + ".trellis", DType::I16, {K / 16, N / 16, 16LL * bits}, cls, layer, expert, Dsv41TensorRole::Exl3Trellis);
  add(out, base + ".suh", DType::F16, {K}, cls, layer, expert, Dsv41TensorRole::Exl3Suh);
  add(out, base + ".svh", DType::F16, {N}, cls, layer, expert, Dsv41TensorRole::Exl3Svh);
  add(out, base + ".mul1", DType::I32, {}, cls, layer, expert, Dsv41TensorRole::Exl3Marker);
}

void expect_attention(TensorList& out, const std::string& p, const Dsv41TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t heads = cfg.num_attention_heads, hd = cfg.head_dim;
  const int64_t ql = cfg.q_lora_rank, ol = cfg.o_lora_rank, og = cfg.o_groups;
  const int b = cfg.fp8_block_size;
  const Dsv41WeightClass c = Dsv41WeightClass::Attention;
  add_bf16(out, p + "q_norm.weight", {ql}, Dsv41WeightClass::LayerNorm, layer);
  add_bf16(out, p + "kv_norm.weight", {hd}, Dsv41WeightClass::LayerNorm, layer);
  add_f32(out, p + "attn_sink", {heads}, c, layer);
  if (cfg.weights_exl3) {
    // The pack (docs/deepseek_v41_flash_exl3_plan.md §1.1): every projection
    // an EXL3 matrix; wo_a as one matrix per output group (slice.g: the
    // group's heads to its o_lora rows); the compressors and the indexer's
    // wq_b and wk EXL3 too; weights_proj stored F16.
    const int ab = cfg.exl3_bits_attention(layer);
    add_exl3(out, p + "wq_a", H, ql, ab, c, layer);
    add_exl3(out, p + "wq_b", ql, heads * hd, ab, c, layer);
    add_exl3(out, p + "wkv", H, hd, ab, c, layer);
    for (int g = 0; g < og; ++g) add_exl3(out, p + "wo_a.slice." + std::to_string(g), heads * hd / og, ol, ab, c, layer);
    add_exl3(out, p + "wo_b", og * ol, H, ab, c, layer);
    if (cfg.is_index_source(layer)) {
      const std::string ip = p + "indexer.";
      const int64_t ih = cfg.index_n_heads, id = cfg.index_head_dim;
      add_exl3(out, ip + "wq_b", ql, ih * id, ab, Dsv41WeightClass::Indexer, layer);
      add(out, ip + "weights_proj.weight", DType::F16, {ih, H}, Dsv41WeightClass::Indexer, layer);
      if (cfg.is_kv_source(layer)) {
        add_exl3(out, ip + "wk", hd, id, cfg.exl3_bits_indexer_wk(layer), Dsv41WeightClass::Indexer, layer);
        add_bf16(out, ip + "k_norm.weight", {id}, Dsv41WeightClass::LayerNorm, layer);
      }
    }
    if (cfg.is_kv_source(layer)) {
      const std::string cp = p + "compressor.";
      add_exl3(out, cp + "wkv", H, hd, ab, Dsv41WeightClass::Compressor, layer);
      if (cfg.compress_ratio(layer) > 1) add_exl3(out, cp + "wgate", H, hd, ab, Dsv41WeightClass::Compressor, layer);
      add_bf16(out, cp + "norm.weight", {hd}, Dsv41WeightClass::LayerNorm, layer);
    }
    return;
  }
  add_fp8(out, p + "wq_a", ql, H, c, layer, b);
  add_fp8(out, p + "wq_b", heads * hd, ql, c, layer, b);
  add_fp8(out, p + "wkv", hd, H, c, layer, b);
  // wo_a is block-diagonal over the output groups: group g maps its
  // heads' (heads/og x hd) outputs to ol rows — stored as one
  // [og * ol, heads * hd / og] matrix.
  add_fp8(out, p + "wo_a", og * ol, heads * hd / og, c, layer, b);
  add_fp8(out, p + "wo_b", H, og * ol, c, layer, b);
  if (cfg.is_index_source(layer)) {
    const std::string ip = p + "indexer.";
    const int64_t ih = cfg.index_n_heads, id = cfg.index_head_dim;
    add_fp8(out, ip + "wq_b", ih * id, ql, Dsv41WeightClass::Indexer, layer, b);
    add_bf16(out, ip + "weights_proj.weight", {ih, H}, Dsv41WeightClass::Indexer, layer);
    if (cfg.is_kv_source(layer)) {
      add_bf16(out, ip + "wk.weight", {id, hd}, Dsv41WeightClass::Indexer, layer);
      add_bf16(out, ip + "k_norm.weight", {id}, Dsv41WeightClass::LayerNorm, layer);
    }
  }
  if (cfg.is_kv_source(layer)) {
    const std::string cp = p + "compressor.";
    add_bf16(out, cp + "wkv.weight", {hd, H}, Dsv41WeightClass::Compressor, layer);
    if (cfg.compress_ratio(layer) > 1)
      add_bf16(out, cp + "wgate.weight", {hd, H}, Dsv41WeightClass::Compressor, layer);
    add_bf16(out, cp + "norm.weight", {hd}, Dsv41WeightClass::LayerNorm, layer);
  }
}

void expect_moe(TensorList& out, const std::string& p, const Dsv41TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size;
  const int64_t S = cfg.shared_expert_inter();
  const bool draft = cfg.is_draft(layer);
  const int64_t E = draft ? cfg.dspark_n_routed_experts : cfg.n_routed_experts;
  if (cfg.weights_exl3) {
    // The pack stores the routers F16 (the bias too); the experts are EXL3
    // at the layer's routed / shared widths (a draft stage: mtp).
    add(out, p + "gate.weight", DType::F16, {E, H}, Dsv41WeightClass::Router, layer);
    add(out, p + "gate.bias", DType::F16, {E}, Dsv41WeightClass::Router, layer);
    if (cfg.vision_present) add_f32(out, p + "gate.bias_vl", {E}, Dsv41WeightClass::Router, layer);
    const int rb = cfg.exl3_bits_routed(layer), sb = cfg.exl3_bits_shared(layer);
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      add_exl3(out, ep + "w1", H, I, rb, Dsv41WeightClass::RoutedExpert, layer, e);
      add_exl3(out, ep + "w2", I, H, rb, Dsv41WeightClass::RoutedExpert, layer, e);
      add_exl3(out, ep + "w3", H, I, rb, Dsv41WeightClass::RoutedExpert, layer, e);
    }
    const std::string sp = p + "shared_experts.";
    add_exl3(out, sp + "w1", H, S, sb, Dsv41WeightClass::SharedExpert, layer);
    add_exl3(out, sp + "w2", S, H, sb, Dsv41WeightClass::SharedExpert, layer);
    add_exl3(out, sp + "w3", H, S, sb, Dsv41WeightClass::SharedExpert, layer);
    return;
  }
  add_bf16(out, p + "gate.weight", {E, H}, Dsv41WeightClass::Router, layer);
  add_f32(out, p + "gate.bias", {E}, Dsv41WeightClass::Router, layer);
  if (cfg.vision_present) add_f32(out, p + "gate.bias_vl", {E}, Dsv41WeightClass::Router, layer);
  for (int e = 0; e < E; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    add_fp4(out, ep + "w1", I, H, Dsv41WeightClass::RoutedExpert, layer, cfg.fp4_block_size, e);
    add_fp4(out, ep + "w2", H, I, Dsv41WeightClass::RoutedExpert, layer, cfg.fp4_block_size, e);
    add_fp4(out, ep + "w3", I, H, Dsv41WeightClass::RoutedExpert, layer, cfg.fp4_block_size, e);
  }
  const std::string sp = p + "shared_experts.";
  add_fp8(out, sp + "w1", S, H, Dsv41WeightClass::SharedExpert, layer, cfg.fp8_block_size);
  add_fp8(out, sp + "w2", H, S, Dsv41WeightClass::SharedExpert, layer, cfg.fp8_block_size);
  add_fp8(out, sp + "w3", S, H, Dsv41WeightClass::SharedExpert, layer, cfg.fp8_block_size);
}

void expect_mhc(TensorList& out, const std::string& p, const Dsv41TextConfig& cfg, int layer) {
  const int64_t rows = cfg.hc_coeff_rows(), width = static_cast<int64_t>(cfg.hc_mult) * cfg.hidden_size;
  for (const char* site : {"attn", "ffn"}) {
    const std::string s = p + "hc_" + site + "_";
    add_f32(out, s + "fn", {rows, width}, Dsv41WeightClass::Mhc, layer);
    add_f32(out, s + "base", {rows}, Dsv41WeightClass::Mhc, layer);
    add_f32(out, s + "scale", {3}, Dsv41WeightClass::Mhc, layer);
  }
}

void expect_engram(TensorList& out, const std::string& p, const Dsv41TextConfig& cfg, int layer) {
  const int idx = cfg.engram_index(layer);
  if (idx < 0) return;
  const std::string ep = p + "engram.";
  const int64_t rows = cfg.engram_num_embeddings[static_cast<size_t>(idx)];
  const int64_t hd = cfg.engram_head_dim;
  add(out, ep + "embed.weight", DType::F8_E4M3, {rows, hd}, Dsv41WeightClass::EngramTable, layer, -1,
      Dsv41TensorRole::TablePayload);
  add(out, ep + "embed.scale", DType::F8_E8M0, {rows, hd / cfg.fp4_block_size}, Dsv41WeightClass::EngramTable,
      layer, -1, Dsv41TensorRole::TableScale);
  if (cfg.weights_exl3) {
    add_exl3(out, ep + "wkv", cfg.engram_width(), static_cast<int64_t>(cfg.hidden_size) * (cfg.hc_mult + 1),
             cfg.exl3_bits_engram(layer), Dsv41WeightClass::Engram, layer);
    add_f32(out, ep + "q_weight", {cfg.hc_mult, cfg.hidden_size}, Dsv41WeightClass::Engram, layer);
    add_f32(out, ep + "k_weight", {cfg.hc_mult, cfg.hidden_size}, Dsv41WeightClass::Engram, layer);
    return;
  }
  add_fp8(out, ep + "wkv", static_cast<int64_t>(cfg.hidden_size) * (cfg.hc_mult + 1), cfg.engram_width(),
          Dsv41WeightClass::Engram, layer, cfg.fp8_block_size);
  add_bf16(out, ep + "q_weight", {cfg.hc_mult, cfg.hidden_size}, Dsv41WeightClass::Engram, layer);
  add_bf16(out, ep + "k_weight", {cfg.hc_mult, cfg.hidden_size}, Dsv41WeightClass::Engram, layer);
}

void expect_draft_extras(TensorList& out, const std::string& p, const Dsv41TextConfig& cfg, int layer) {
  const int stage = cfg.draft_stage(layer);
  const int64_t H = cfg.hidden_size, V = cfg.vocab_size, R = cfg.dspark_markov_rank;
  const Dsv41WeightClass c = Dsv41WeightClass::Draft;
  if (stage == 0) {
    if (cfg.weights_exl3)
      add_exl3(out, p + "main_proj", H * static_cast<int64_t>(cfg.dspark_target_layer_ids.size()), H, cfg.exl3.mtp, c,
               layer);
    else
      add_fp8(out, p + "main_proj", H, H * static_cast<int64_t>(cfg.dspark_target_layer_ids.size()), c, layer,
              cfg.fp8_block_size);
    add_bf16(out, p + "main_norm.weight", {H}, Dsv41WeightClass::LayerNorm, layer);
  }
  if (stage == cfg.num_nextn_predict_layers - 1) {
    add_bf16(out, p + "norm.weight", {H}, Dsv41WeightClass::LayerNorm, layer);
    add_bf16(out, p + "markov_head.embed.weight", {V, R}, c, layer);
    // The pack stores the Markov head and the confidence head F16.
    add(out, p + "markov_head.head.weight", cfg.weights_exl3 ? DType::F16 : DType::BF16, {V, R}, c, layer);
    add(out, p + "confidence_head.proj.weight", cfg.weights_exl3 ? DType::F16 : DType::BF16, {1, H + R}, c, layer);
  }
}

}  // namespace

std::string dsv41_layer_prefix(const Dsv41TextConfig& cfg, int layer) {
  if (cfg.is_draft(layer)) return "mtp." + std::to_string(cfg.draft_stage(layer)) + ".";
  return "layers." + std::to_string(layer) + ".";
}

std::vector<Dsv41ExpectedTensor> dsv41_expected_layer_tensors(const Dsv41TextConfig& cfg, int layer) {
  if (layer < 0 || layer >= cfg.max_layer())
    throw std::invalid_argument("dsv41_expected_layer_tensors: layer out of range");
  const std::string p = dsv41_layer_prefix(cfg, layer);
  const int64_t H = cfg.hidden_size;
  TensorList out;
  if (cfg.is_draft(layer)) expect_draft_extras(out, p, cfg, layer);
  expect_engram(out, p, cfg, layer);
  add_bf16(out, p + "attn_norm.weight", {H}, Dsv41WeightClass::LayerNorm, layer);
  add_bf16(out, p + "ffn_norm.weight", {H}, Dsv41WeightClass::LayerNorm, layer);
  expect_mhc(out, p, cfg, layer);
  expect_attention(out, p + "attn.", cfg, layer);
  expect_moe(out, p + "ffn.", cfg, layer);
  return out;
}

std::vector<Dsv41ExpectedTensor> dsv41_expected_global_tensors(const Dsv41TextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "embed.weight", {cfg.vocab_size, H}, Dsv41WeightClass::Embed, -1);
  add_bf16(out, "norm.weight", {H}, Dsv41WeightClass::FinalNorm, -1);
  if (cfg.weights_exl3)
    add_exl3(out, "head", H, cfg.vocab_size, cfg.exl3.head, Dsv41WeightClass::LmHead, -1);
  else
    add_bf16(out, "head.weight", {cfg.vocab_size, H}, Dsv41WeightClass::LmHead, -1);
  return out;
}

std::vector<Dsv41ExpectedTensor> dsv41_expected_vision_tensors(const Dsv41TextConfig& cfg) {
  TensorList out;
  if (!cfg.vision_present) return out;
  const Dsv41VisionConfig& v = cfg.vision;
  const int64_t D = v.hidden_size, I = v.intermediate_size, H = cfg.hidden_size;
  auto skip = [&](const std::string& name, std::vector<int64_t> shape) {
    add(out, name, DType::BF16, std::move(shape), Dsv41WeightClass::Vision, -1, -1, Dsv41TensorRole::Skipped);
  };
  skip("vision.patch_embed.proj.weight", {D, 3LL * v.patch_size * v.patch_size});
  skip("vision.patch_embed.proj.bias", {D});
  skip("vision.norm.weight", {D});
  for (int l = 0; l < v.num_hidden_layers; ++l) {
    const std::string p = "vision.blocks." + std::to_string(l) + ".";
    skip(p + "norm1.weight", {D});
    skip(p + "norm2.weight", {D});
    skip(p + "attn.wqkv.weight", {3 * D, D});
    skip(p + "attn.wqkv.bias", {3 * D});
    skip(p + "attn.wo.weight", {D, D});
    skip(p + "attn.wo.bias", {D});
    skip(p + "mlp.w1.weight", {2 * I, D});  // fused gate/up
    skip(p + "mlp.w2.weight", {D, I});
  }
  const int64_t unshuffled = D * v.downsample_ratio * v.downsample_ratio;
  skip("aligner.w1.weight", {H, unshuffled});
  skip("aligner.w1.bias", {H});
  skip("aligner.w2.weight", {H, H});
  skip("aligner.w2.bias", {H});
  skip("image_start", {H});
  skip("image_end", {H});
  skip("image_newline", {H});
  return out;
}

std::vector<Dsv41ExpectedTensor> dsv41_expected_text_tensors(const Dsv41TextConfig& cfg) {
  TensorList out = dsv41_expected_global_tensors(cfg);
  for (int l = 0; l < cfg.max_layer(); ++l) {
    TensorList layer = dsv41_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()), std::make_move_iterator(layer.end()));
  }
  TensorList vision = dsv41_expected_vision_tensors(cfg);
  out.insert(out.end(), std::make_move_iterator(vision.begin()), std::make_move_iterator(vision.end()));
  return out;
}

Dsv41BindReport dsv41_validate_text_binding(
    const Dsv41TextConfig& cfg, const std::unordered_map<std::string, Dsv41TensorDesc>& present,
    size_t max_errors) {
  Dsv41BindReport rep;
  const auto expected = dsv41_expected_text_tensors(cfg);
  rep.expected = expected.size();
  auto push_error = [&](std::string msg) {
    if (rep.errors.size() < max_errors) rep.errors.push_back(std::move(msg));
  };
  auto shape_str = [](const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(s[i]);
    }
    return out + "]";
  };
  std::unordered_map<std::string, int8_t> consumed;
  consumed.reserve(present.size());
  for (const auto& e : expected) {
    auto it = present.find(e.name);
    if (it == present.end()) {
      ++rep.missing;
      push_error(std::format("missing tensor '{}'", e.name));
      continue;
    }
    consumed.emplace(e.name, 1);
    if (it->second.dtype != e.dtype) {
      ++rep.dtype_mismatch;
      push_error(std::format("'{}' dtype {} != expected {}", e.name, dtype_name(it->second.dtype),
                             dtype_name(e.dtype)));
      continue;
    }
    if (it->second.shape != e.shape) {
      ++rep.shape_mismatch;
      push_error(std::format("'{}' shape {} != expected {}", e.name, shape_str(it->second.shape),
                             shape_str(e.shape)));
      continue;
    }
    ++rep.matched;
    switch (e.role) {
      case Dsv41TensorRole::Fp4Payload: ++rep.fp4_matrices; break;
      case Dsv41TensorRole::Fp8Payload: ++rep.fp8_matrices; break;
      case Dsv41TensorRole::Exl3Trellis: ++rep.exl3_matrices; break;
      case Dsv41TensorRole::TablePayload: ++rep.engram_tables; break;
      case Dsv41TensorRole::Skipped: ++rep.skipped; break;
      default: break;
    }
  }
  // A backbone layer past the config's stack belongs to a truncated
  // diagnostic stack, not to a binding error.
  const auto beyond_stack = [&](const std::string& name) {
    constexpr std::string_view prefix = "layers.";
    if (name.compare(0, prefix.size(), prefix) != 0) return false;
    size_t i = prefix.size();
    int layer = 0;
    bool digits = false;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
      layer = layer * 10 + (name[i] - '0');
      ++i;
      digits = true;
    }
    return digits && i < name.size() && name[i] == '.' && layer >= cfg.num_hidden_layers;
  };
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    if (beyond_stack(name)) {
      ++rep.beyond_stack;
      continue;
    }
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void dsv41_tp_validate_geometry(const Dsv41TextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) { throw std::invalid_argument("dsv41 tp geometry: " + what); };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  if (cfg.o_groups % world != 0)
    fail("o_groups must divide by world (a rank holds whole output groups of wo_a)");
  if (cfg.num_attention_heads % world != 0) fail("num_attention_heads must divide by world");
  if (cfg.index_n_heads % world != 0) fail("index_n_heads must divide by world");
  if (!cfg.engram_layer_ids.empty() && cfg.engram_n_heads % world != 0)
    fail("engram_n_heads must divide by world (the tables are sharded by hash head)");
  if (cfg.weights_exl3) {
    // Whole 128-blocks per rank (the plan's §2.3): the intermediate and the
    // vocabulary need at least one block per rank; the wo_b input slice (a
    // rank's groups x o_lora) and the head-block slice are 128-multiples
    // by the config's own checks.
    for (const int inter : {cfg.moe_intermediate_size, cfg.shared_expert_inter()})
      if (inter / 128 < world) fail("an intermediate size must hold at least one 128-block per rank (EXL3)");
    if (cfg.vocab_size / 128 < world) fail("vocab_size must hold at least one 128-block per rank (the EXL3 head)");
    if (((static_cast<int64_t>(cfg.o_groups) * cfg.o_lora_rank) / world) % 128 != 0)
      fail("the wo_b input slice must be a multiple of 128 (the EXL3 rotation block)");
    return;
  }
  if (cfg.vocab_size % world != 0) fail("vocab_size must divide by world (the head is vocab-sharded)");
  for (const int inter : {cfg.moe_intermediate_size, cfg.shared_expert_inter()}) {
    if (inter % world != 0) fail("an intermediate size must divide by world");
    if ((inter / world) % 32 != 0)
      fail("an intermediate slice must be a multiple of 32 (the MXFP4 block and the fp8 block of the down projection's column slice)");
  }
  if (((static_cast<int64_t>(cfg.o_groups) * cfg.o_lora_rank) / world) % cfg.fp8_block_size != 0)
    fail("the wo_b input slice must be a multiple of the fp8 block");
}

}  // namespace dgpp
