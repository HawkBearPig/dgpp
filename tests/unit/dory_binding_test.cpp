// The Dory1.0-4B expected-tensor table: its shape derived from the release's
// config, the one-rank world acceptance, and — when the checkpoint is in the
// hub cache — the full binding against every shard's header (every expected
// tensor present with its dtype and shape, nothing unexpected).
// (docs/dory_plan.md §1.7, §2, §3 D3-D5.)
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/dory/binding.hpp"
#include "models/dory/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The reference's generated `layer_types` (dory_config.py __post_init__), the
// same pattern the release's config.json carries.
std::string default_layer_types() {
  std::string s = "[";
  for (int i = 0; i < 72; ++i) {
    if (i % 2)
      s += "\"mlp\"";
    else if (i % 8 == 6)
      s += "\"full_attention\"";
    else
      s += "\"sliding_attention\"";
    if (i != 71) s += ", ";
  }
  return s + "]";
}

// nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4, the release's config.json.
std::string config_json() {
  std::string s = R"({
  "architectures": ["DoryForCausalLM"], "model_type": "dory",
  "vocab_size": 131072, "hidden_size": 2560, "num_hidden_layers": 72,
  "intermediate_size": 7168, "num_attention_heads": 16, "num_key_value_heads": 4,
  "head_dim": 256, "max_position_embeddings": 262144,
  "tie_word_embeddings": false, "attention_bias": false, "mlp_bias": false,
  "hidden_act": "silu", "initializer_range": 0.0177,
  "ngpt_alpha_init": 0.05, "attention_output_gate_logit_scale_init": null,
  "n_input_layers": 16, "n_recurrent_layers": 40, "n_output_layers": 16,
  "n_recurrent_loops": 2, "sliding_window": 1024, "rope_scaling": null,
  "rope_parameters": {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 1.0},
    "full_attention": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 1.0}
  },
  "layer_types": __LAYER_TYPES__,
  "quantization_config": {
    "config_groups": {"group_0": {
      "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
      "targets": ["Linear"]}},
    "ignore": ["backbone.embeddings", "lm_head"],
    "quant_algo": "W4A16_NVFP4",
    "producer": {"name": "modelopt", "version": "0.0.0"},
    "quant_method": "modelopt"
  }
})";
  const std::string marker = "__LAYER_TYPES__";
  s.replace(s.find(marker), marker.size(), default_layer_types());
  return s;
}

dgpp::DoryConfig release_config() {
  return dgpp::DoryConfig::parse(dgpp::minijson::parse(config_json()).root);
}

std::filesystem::path landed_snapshot() {
  namespace fs = std::filesystem;
  const char* home = std::getenv("HOME");
  if (!home) return {};
  const fs::path root =
      fs::path(home) / ".cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4/snapshots";
  if (!fs::is_directory(root)) return {};
  for (const auto& snap : fs::directory_iterator(root))
    if (fs::exists(snap.path() / "config.json") && fs::exists(snap.path() / "model.safetensors"))
      return snap.path();
  return {};
}

size_t count_role(const std::vector<dgpp::DoryExpectedTensor>& v, dgpp::DoryTensorRole r) {
  size_t n = 0;
  for (const auto& e : v) n += e.role == r;
  return n;
}

}  // namespace

DGPP_TEST(dory_binding_table_has_the_release_shape) {
  const dgpp::DoryConfig cfg = release_config();

  // One attention layer: five projections x 3 (codes, scale grid, fp32 scale)
  // + s_gate + sqk + attn_alpha.
  const auto attn = dgpp::dory_expected_attention_tensors(cfg, 0);
  require(attn.size() == 5 * 3 + 3, "attention layer " + std::to_string(attn.size()));
  require(count_role(attn, dgpp::DoryTensorRole::Fp4Payload) == 5, "5 attention fp4 matrices");
  // One MLP layer: three projections x 3 + suv_gate + suv_up + mlp_alpha.
  const auto mlp = dgpp::dory_expected_mlp_tensors(cfg, 1);
  require(mlp.size() == 3 * 3 + 3, "mlp layer " + std::to_string(mlp.size()));
  require(count_role(mlp, dgpp::DoryTensorRole::Fp4Payload) == 3, "3 mlp fp4 matrices");

  // The dispatch follows `layer_types`: even attention, odd mlp.
  require(dgpp::dory_expected_layer_tensors(cfg, 0).size() == attn.size(), "layer 0 attention");
  require(dgpp::dory_expected_layer_tensors(cfg, 1).size() == mlp.size(), "layer 1 mlp");

  // The exact shapes of one attention layer (the release's dims).
  auto find = [](const std::vector<dgpp::DoryExpectedTensor>& v, const std::string& n) {
    for (const auto& e : v)
      if (e.name == n) return e;
    throw std::runtime_error("no tensor " + n);
  };
  const std::string m = "backbone.layers.0.mixer.";
  using T = std::vector<int64_t>;
  require(find(attn, m + "q_proj.weight").shape == T{4096, 1280}, "q codes");
  require(find(attn, m + "q_proj.weight_scale").shape == T{4096, 160}, "q scale grid");
  require(find(attn, m + "q_proj.weight_scale_2").shape == T{}, "q per-tensor scale");
  require(find(attn, m + "k_proj.weight").shape == T{1024, 1280}, "k codes");
  require(find(attn, m + "o_proj.weight").shape == T{2560, 2048}, "o codes");
  require(find(attn, m + "g_proj.weight").shape == T{4096, 1280}, "g codes");
  require(find(attn, m + "s_gate").shape == T{4096}, "s_gate");
  require(find(attn, m + "sqk").shape == T{4096}, "sqk");
  require(find(attn, "backbone.layers.0.attn_alpha").shape == T{2560}, "attn_alpha at the root");
  require(find(mlp, "backbone.layers.1.mixer.gate_proj.weight").shape == T{7168, 1280}, "gate codes");
  require(find(mlp, "backbone.layers.1.mixer.down_proj.weight").shape == T{2560, 3584}, "down codes");
  require(find(mlp, "backbone.layers.1.mixer.suv_gate").shape == T{7168}, "suv_gate");
  require(find(mlp, "backbone.layers.1.mlp_alpha").shape == T{2560}, "mlp_alpha at the root");

  // The globals.
  const auto globals = dgpp::dory_expected_global_tensors(cfg);
  require(globals.size() == 3, "3 globals");
  require(find(globals, "backbone.embeddings.weight").shape == T{131072, 2560}, "embedding");
  require(find(globals, "lm_head.weight").shape == T{131072, 2560}, "head");
  require(find(globals, "lm_head.logit_scale").shape == T{131072}, "logit scale");

  // The whole table: 3 globals + 36 attention x 18 + 36 mlp x 12.
  const auto all = dgpp::dory_expected_text_tensors(cfg);
  require(all.size() == 3 + 36 * 18 + 36 * 12, "table size " + std::to_string(all.size()));
  require(count_role(all, dgpp::DoryTensorRole::Fp4Payload) == 36 * 5 + 36 * 3, "fp4 matrices");
  require(count_role(all, dgpp::DoryTensorRole::Fp4Scale) == 36 * 5 + 36 * 3, "scale grids");
  require(count_role(all, dgpp::DoryTensorRole::Fp4Global) == 36 * 5 + 36 * 3, "per-tensor scales");

  // There is no activation scale anywhere (W4A16).
  for (const auto& e : all) require(e.dtype != dgpp::DType::F32 || e.shape.empty(), "no input scale");
}

DGPP_TEST(dory_binding_validates_a_synthetic_release) {
  const dgpp::DoryConfig cfg = release_config();
  // A `present` map built from the table itself is exactly accepted.
  std::unordered_map<std::string, dgpp::DoryTensorDesc> present;
  for (const auto& e : dgpp::dory_expected_text_tensors(cfg))
    present.emplace(e.name, dgpp::DoryTensorDesc{e.dtype, e.shape});
  const dgpp::DoryBindReport ok = dgpp::dory_validate_text_binding(cfg, present);
  require(ok.ok() && ok.matched == ok.expected, "the table validates against itself");

  // A wrong shape, a wrong dtype, a missing tensor and an extra tensor are
  // each reported.
  auto bad_shape = present;
  bad_shape["backbone.layers.0.mixer.q_proj.weight"].shape = {4096, 1290};
  const dgpp::DoryBindReport s = dgpp::dory_validate_text_binding(cfg, bad_shape);
  require(s.shape_mismatch == 1 && !s.ok(), "shape mismatch");

  auto bad_dtype = present;
  bad_dtype["backbone.layers.1.mixer.gate_proj.weight_scale"].dtype = dgpp::DType::BF16;
  const dgpp::DoryBindReport d = dgpp::dory_validate_text_binding(cfg, bad_dtype);
  require(d.dtype_mismatch == 1 && !d.ok(), "dtype mismatch");

  auto missing = present;
  missing.erase("lm_head.logit_scale");
  const dgpp::DoryBindReport m = dgpp::dory_validate_text_binding(cfg, missing);
  require(m.missing == 1 && !m.ok(), "missing tensor");

  auto extra = present;
  extra.emplace("backbone.rotary_emb.inv_freq", dgpp::DoryTensorDesc{dgpp::DType::F32, {128}});
  const dgpp::DoryBindReport x = dgpp::dory_validate_text_binding(cfg, extra);
  require(x.unexpected == 1 && !x.ok(), "unexpected tensor");
}

DGPP_TEST(dory_world_must_be_one) {
  dgpp::dory_validate_world(1);
  bool refused = false;
  try {
    dgpp::dory_validate_world(2);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "world 2 is refused");
}

DGPP_TEST(dory_binding_matches_the_landed_checkpoint) {
  const auto snap = landed_snapshot();
  if (snap.empty()) return;
  namespace fs = std::filesystem;
  const dgpp::DoryConfig cfg = dgpp::DoryConfig::from_json_file((snap / "config.json").string());
  std::unordered_map<std::string, dgpp::DoryTensorDesc> present;
  for (const auto& entry : fs::directory_iterator(snap)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) {
      present.emplace(t.name, dgpp::DoryTensorDesc{t.dtype, t.shape});
    });
  }
  const dgpp::DoryBindReport rep = dgpp::dory_validate_text_binding(cfg, present);
  std::string errs;
  for (const auto& e : rep.errors) errs += "\n  " + e;
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + ", dtype " +
                        std::to_string(rep.dtype_mismatch) + ", shape " + std::to_string(rep.shape_mismatch) +
                        ", unexpected " + std::to_string(rep.unexpected) + errs);
  require(rep.fp4_matrices == 36 * 5 + 36 * 3, "fp4 matrices " + std::to_string(rep.fp4_matrices));
}
