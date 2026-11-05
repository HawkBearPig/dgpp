#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <unordered_map>

#include "common/test.hpp"
#include "loaders/safetensors.hpp"
#include "models/minimax/binding.hpp"

namespace {
void require(bool ok, const std::string& msg) {
  if (!ok) throw std::runtime_error(msg);
}
dgpp::MinimaxConfig config() {
  return dgpp::MinimaxConfig::from_json_file(DGPP_SOURCE_DIR "/tests/data/minimax/config.json");
}
}  // namespace

DGPP_TEST(minimax_binding_published_census) {
  const auto c = config();
  const auto all = dgpp::minimax_expected_text_tensors(c);
  size_t bytes = 0, fp8 = 0, fp4 = 0, scalar_globals = 0;
  for (const auto& e : all) {
    bytes += e.nbytes();
    fp8 += e.role == dgpp::MinimaxTensorRole::Fp8Payload;
    fp4 += e.role == dgpp::MinimaxTensorRole::Fp4Payload;
    if (e.role == dgpp::MinimaxTensorRole::Fp4Global ||
        e.role == dgpp::MinimaxTensorRole::Fp4InputGlobal) {
      require(e.shape.empty() && e.dtype == dgpp::DType::F32,
              "published globals are rank-zero FP32 scalars");
      ++scalar_globals;
    }
    if (e.role == dgpp::MinimaxTensorRole::Fp8Scale)
      require(e.dtype == dgpp::DType::U8 && e.shape.size() == 2, "MXFP8 scales are E8M0 bytes");
  }
  // Independent census of every safetensors header at HF revision
  // 2f33a8b9ae49ce7b51c994df97166a7fd059bc2e, excluding vision and dspark.
  require(all.size() == 89109, "text tensor count: " + std::to_string(all.size()));
  require(bytes == 248369266176ULL, "text byte count: " + std::to_string(bytes));
  require(fp8 == 540 && fp4 == 21888 && scalar_globals == 43776, "quantization census");
  require(dgpp::minimax_expected_layer_tensors(c, 0).size() == 24, "dense binding");
  require(dgpp::minimax_expected_layer_tensors(c, 3).size() == 1562, "MoE binding");
}

DGPP_TEST(minimax_binding_reports_corruptions) {
  auto c = config();
  c.num_hidden_layers = 2;
  c.moe_layer = {0, 1};
  c.num_local_experts = 4;
  std::unordered_map<std::string, dgpp::MinimaxTensorDesc> present;
  for (const auto& e : dgpp::minimax_expected_text_tensors(c))
    present.emplace(e.name, dgpp::MinimaxTensorDesc{e.dtype, e.shape});
  auto r = dgpp::minimax_validate_text_binding(c, present);
  require(r.ok() && r.matched == r.expected, "clean binding");
  present.erase("language_model.model.norm.weight");
  present.at("language_model.model.layers.0.self_attn.q_proj.weight_scale_inv").dtype =
      dgpp::DType::F32;
  present.at("language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_global_scale")
      .shape = {1};
  present.emplace("language_model.model.layers.0.self_attn.index_v_proj.weight",
                  dgpp::MinimaxTensorDesc{});
  r = dgpp::minimax_validate_text_binding(c, present, 2);
  require(!r.ok() && r.missing == 1 && r.dtype_mismatch == 1 && r.shape_mismatch == 1 &&
              r.unexpected == 1,
          "report must retain every corruption count");
  require(r.errors.size() == 2, "bounded diagnostic text");
}

DGPP_TEST(minimax_binding_pinned_checkpoint) {
  namespace fs = std::filesystem;
  const auto* home = std::getenv("HOME");
  if (!home) dgpp::test::skip("HOME unset");
  const fs::path snapshot = fs::path(home) /
                            ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/"
                            "snapshots/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e";
  if (!fs::exists(snapshot / "model-00048-of-00048.safetensors"))
    dgpp::test::skip("pinned checkpoint download incomplete");
  std::unordered_map<std::string, dgpp::MinimaxTensorDesc> present;
  for (const auto& p : fs::directory_iterator(snapshot)) {
    if (p.path().extension() != ".safetensors") continue;
    const auto header = dgpp::SafetensorsFile::open(p.path().string());
    header->for_each([&](const dgpp::TensorInfo& t) {
      require(present.emplace(t.name, dgpp::MinimaxTensorDesc{t.dtype, t.shape}).second,
              "duplicate " + t.name);
    });
  }
  const auto r = dgpp::minimax_validate_text_binding(config(), present);
  require(r.ok(), r.errors.empty() ? "binding mismatch" : r.errors.front());
  require(r.matched == 89109 && present.size() == 89632, "full target header census");
}
