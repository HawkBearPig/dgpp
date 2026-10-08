// The Qwen3.8-27B expected-tensor table: the transcribed release shapes
// (verified against the checkpoint's index) come out of the table, and the
// validator matches a complete map, reports a missing tensor, flags a dtype
// mismatch, and counts vision tensors without failing.
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "models/qwen/binding35.hpp"
#include "models/qwen/config35.hpp"
#include "qwen35_config_json.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Qwen35TextConfig cfg() {
  // The parsed values view the input text: the strings must outlive the
  // parse (a temporary would dangle).
  const std::string text = qwen35_fixture::text_json();
  const std::string quant = qwen35_fixture::kQuantFp8;
  const auto t = dgpp::minijson::parse(text);
  const auto q = dgpp::minijson::parse(quant);
  return dgpp::Qwen35TextConfig::parse(t.root, &q.root);
}

const dgpp::QwenExpectedTensor* find(const std::vector<dgpp::QwenExpectedTensor>& v,
                                     const std::string& name) {
  for (const auto& e : v)
    if (e.name == name) return &e;
  return nullptr;
}

void require_shape(const std::vector<dgpp::QwenExpectedTensor>& v, const std::string& name,
                   dgpp::DType dtype, std::vector<int64_t> shape) {
  const auto* e = find(v, name);
  require(e != nullptr, "missing table entry: " + name);
  require(e->dtype == dtype, "dtype: " + name);
  require(e->shape == shape, "shape: " + name);
}

std::unordered_map<std::string, dgpp::QwenTensorDesc> present_map(
    const std::vector<dgpp::QwenExpectedTensor>& v) {
  std::unordered_map<std::string, dgpp::QwenTensorDesc> out;
  for (const auto& e : v) out.emplace(e.name, dgpp::QwenTensorDesc{e.dtype, e.shape});
  return out;
}

}  // namespace

DGPP_TEST(qwen35_binding_linear_layer_shapes) {
  const auto c = cfg();
  const auto v = dgpp::qwen35_expected_layer_tensors(c, 0);
  using dgpp::DType;
  // GDN: [2*kdim + vdim, H] = [10240, 5120], vdim 6144, vh 48, vd 128.
  require_shape(v, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                DType::F8_E4M3, {10240, 5120});
  require_shape(v, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv",
                DType::BF16, {80, 40});
  require_shape(v, "model.language_model.layers.0.linear_attn.conv1d.weight", DType::BF16,
                {10240, 1, 4});
  require_shape(v, "model.language_model.layers.0.linear_attn.A_log", DType::BF16, {48});
  require_shape(v, "model.language_model.layers.0.linear_attn.norm.weight", DType::BF16,
                {128});
  require_shape(v, "model.language_model.layers.0.mlp.gate_proj.weight", DType::F8_E4M3,
                {17408, 5120});
  require_shape(v, "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv",
                DType::BF16, {136, 40});
  require_shape(v, "model.language_model.layers.0.input_layernorm.weight", DType::BF16,
                {5120});
  require(find(v, "model.language_model.layers.0.self_attn.q_proj.weight") == nullptr,
          "linear layer must not expect self_attn");
}

DGPP_TEST(qwen35_binding_full_layer_shapes) {
  const auto c = cfg();
  const auto v = dgpp::qwen35_expected_layer_tensors(c, 3);
  using dgpp::DType;
  // Full: q [2*qh*d, H] = [12288, 5120] ([q | gate]), o [H, qh*d].
  require_shape(v, "model.language_model.layers.3.self_attn.q_proj.weight", DType::F8_E4M3,
                {12288, 5120});
  require_shape(v, "model.language_model.layers.3.self_attn.q_proj.weight_scale_inv",
                DType::BF16, {96, 40});
  require_shape(v, "model.language_model.layers.3.self_attn.k_proj.weight", DType::F8_E4M3,
                {1024, 5120});
  require_shape(v, "model.language_model.layers.3.self_attn.o_proj.weight", DType::F8_E4M3,
                {5120, 6144});
  require_shape(v, "model.language_model.layers.3.self_attn.q_norm.weight", DType::BF16,
                {256});
  require(find(v, "model.language_model.layers.3.linear_attn.in_proj_qkv.weight") == nullptr,
          "full layer must not expect linear_attn");
}

DGPP_TEST(qwen35_binding_mtp_and_globals) {
  const auto c = cfg();
  const auto layer = dgpp::qwen35_expected_layer_tensors(c, 64);
  using dgpp::DType;
  require_shape(layer, "mtp.layers.0.self_attn.q_proj.weight", DType::F8_E4M3, {12288, 5120});
  require_shape(layer, "mtp.layers.0.mlp.down_proj.weight", DType::F8_E4M3, {5120, 17408});
  const auto g = dgpp::qwen35_expected_global_tensors(c);
  require_shape(g, "model.language_model.embed_tokens.weight", DType::BF16, {248320, 5120});
  require_shape(g, "lm_head.weight", DType::BF16, {248320, 5120});
  require_shape(g, "model.language_model.norm.weight", DType::BF16, {5120});
  require_shape(g, "mtp.fc.weight", DType::BF16, {5120, 10240});
}

DGPP_TEST(qwen35_binding_mixed_release) {
  // The NVFP4 mixed release's table: per-matrix forms resolved from the
  // parsed config_groups (the release's own group/ignore rules).
  const std::string text = qwen35_fixture::text_json();
  const std::string quant = qwen35_fixture::kQuantNvfp4Mixed;
  const auto t = dgpp::minijson::parse(text);
  const auto q = dgpp::minijson::parse(quant);
  const auto c = dgpp::Qwen35TextConfig::parse(t.root, &q.root);
  const auto v = dgpp::qwen35_expected_text_tensors(c);
  using dgpp::DType;
  using R = dgpp::QwenTensorRole;
  // Channel FP8 attention and lm_head, the late MLP layers; NVFP4 the
  // early MLPs; BF16 the draft layer (the `^mtp.*` ignore) and the norms.
  require_shape(v, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                DType::F8_E4M3, {10240, 5120});
  require_shape(v, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale",
                DType::BF16, {10240, 1});
  require_shape(v, "model.language_model.layers.0.mlp.gate_proj.weight_packed", DType::U8,
                {17408, 2560});
  require_shape(v, "model.language_model.layers.0.mlp.gate_proj.weight_scale", DType::F8_E4M3,
                {17408, 320});
  require_shape(v, "model.language_model.layers.0.mlp.gate_proj.weight_global_scale", DType::F32,
                {1});
  require_shape(v, "model.language_model.layers.0.mlp.gate_proj.input_global_scale", DType::F32,
                {1});
  require_shape(v, "model.language_model.layers.56.mlp.down_proj.weight", DType::F8_E4M3,
                {5120, 17408});
  require_shape(v, "model.language_model.layers.3.self_attn.k_scale", DType::BF16, {1});
  require_shape(v, "mtp.layers.0.self_attn.q_proj.weight", DType::BF16, {12288, 5120});
  require_shape(v, "mtp.layers.0.mlp.up_proj.weight", DType::BF16, {17408, 5120});
  require_shape(v, "lm_head.weight", DType::F8_E4M3, {248320, 5120});
  require(find(v, "model.language_model.layers.3.self_attn.q_proj.weight_scale_inv") == nullptr,
          "no block scales in the mixed release");
  size_t fp4 = 0, fp8 = 0, g16 = 0, insc = 0;
  for (const auto& e : v) {
    fp4 += e.role == R::Fp4Payload;
    fp8 += e.role == R::Fp8Payload;
    g16 += e.role == R::Fp4Global;
    insc += e.role == R::InputScale;
  }
  require(fp4 == 56 * 3, "the early layers' MLPs are the NVFP4 ones");  // 0..55 x 3
  require(fp8 == 48 * 3 + 16 * 4 + 8 * 3 + 1, "attn + late MLP + head channel-fp8");
  require(g16 == fp4, "one global per fp4 matrix");
  require(insc == fp4 + 16 * 2, "input globals + the full layers' kv scales");
  // A complete mixed map validates (vision counted, nothing unexpected):
  auto present = present_map(v);
  present.emplace("model.visual.blocks.0.attn.qkv.weight",
                  dgpp::QwenTensorDesc{DType::F8_E4M3, {1152 * 3, 3456}});
  const auto rep = dgpp::qwen35_validate_text_binding(c, present);
  require(rep.ok(), "complete mixed map validates");
  require(rep.vision == 1, "vision counted");
}

DGPP_TEST(qwen35_binding_validates_a_complete_map) {
  const auto c = cfg();
  const auto expected = dgpp::qwen35_expected_text_tensors(c);
  require(expected.size() == 48 * 20 + 16 * 18 + 18 + 7, "table size");
  auto present = present_map(expected);
  // Vision tensors are counted and skipped, not validated.
  present.emplace("model.visual.blocks.0.attn.qkv.weight",
                  dgpp::QwenTensorDesc{dgpp::DType::BF16, {1152 * 3, 1152}});
  const auto rep = dgpp::qwen35_validate_text_binding(c, present);
  require(rep.ok(), "complete map validates");
  require(rep.matched == expected.size(), "all matched");
  require(rep.vision == 1, "vision counted");

  auto missing_one = present;
  missing_one.erase("model.language_model.layers.0.mlp.gate_proj.weight");
  const auto rep_missing = dgpp::qwen35_validate_text_binding(c, missing_one);
  require(!rep_missing.ok() && rep_missing.missing == 1, "one missing fails");

  auto bad_dtype = present;
  bad_dtype["model.language_model.layers.3.self_attn.o_proj.weight"] =
      dgpp::QwenTensorDesc{dgpp::DType::BF16, {5120, 6144}};
  const auto rep_dtype = dgpp::qwen35_validate_text_binding(c, bad_dtype);
  require(!rep_dtype.ok() && rep_dtype.dtype_mismatch == 1, "dtype mismatch fails");
}

DGPP_TEST(qwen35_binding_tp_geometry) {
  const auto c = cfg();
  dgpp::qwen35_tp_validate_geometry(c, 0, 1);
  dgpp::qwen35_tp_validate_geometry(c, 3, 4);
  bool threw = false;
  try {
    dgpp::qwen35_tp_validate_geometry(c, 0, 3);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "world 3 must fail (intermediate 17408 % 3)");
}

DGPP_TEST(qwen35_binding_moe_modelopt_table) {
  // The 122B MoE table: router/shared BF16, routed experts ModelOpt NVFP4
  // (backbone), MTP experts BF16, attention/GDN BF16, no kv scales.
  const std::string text = qwen35_fixture::moe_text_json();
  const std::string quant = qwen35_fixture::kQuantModeloptNvfp4;
  const auto t = dgpp::minijson::parse(text);
  const auto q = dgpp::minijson::parse(quant);
  const auto c = dgpp::Qwen35TextConfig::parse(t.root, &q.root);
  const auto v = dgpp::qwen35_expected_layer_tensors(c, 0);
  using dgpp::DType;
  require_shape(v, "model.language_model.layers.0.mlp.gate.weight", DType::BF16, {256, 3072});
  require_shape(v, "model.language_model.layers.0.mlp.shared_expert_gate.weight", DType::BF16,
                {1, 3072});
  require_shape(v, "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
                DType::BF16, {1024, 3072});
  require_shape(v, "model.language_model.layers.0.mlp.experts.0.gate_proj.weight", DType::U8,
                {1024, 1536});
  require_shape(v, "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale",
                DType::F8_E4M3, {1024, 192});
  require_shape(v, "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_2",
                DType::F32, {});
  require_shape(v, "model.language_model.layers.0.mlp.experts.0.gate_proj.input_scale",
                DType::F32, {});
  require_shape(v, "model.language_model.layers.0.mlp.experts.0.down_proj.weight", DType::U8,
                {3072, 512});
  const auto full = dgpp::qwen35_expected_layer_tensors(c, 3);
  require_shape(full, "model.language_model.layers.3.self_attn.q_proj.weight", DType::BF16,
                {16384, 3072});
  require(find(full, "model.language_model.layers.3.self_attn.k_scale") == nullptr,
           "no kv scales in the MoE release");
  const auto mtp = dgpp::qwen35_expected_layer_tensors(c, 48);
  require_shape(mtp, "mtp.layers.0.mlp.experts.0.gate_proj.weight", DType::BF16,
                {1024, 3072});
  auto present = present_map(dgpp::qwen35_expected_text_tensors(c));
  const auto rep = dgpp::qwen35_validate_text_binding(c, present);
  require(rep.ok(), "complete MoE map validates");
}
