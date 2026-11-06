#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "glm_dsa_fixture.hpp"
#include "models/glm_dsa/model.hpp"
#include "models/glm_dsa/modification_targets.hpp"

namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
template <class T>
void exact(const std::vector<T>& a, const std::vector<T>& b, const char* message) {
  require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0,
          message);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 3, "usage: heretic_capture_test FIXTURE_DIR int8|m346");
    const bool m346 = std::string(argv[2]) == "m346";
    const auto cfg = glmdsafx::tiny_config(5, 1, false, false, 8, m346);
    glmdsafx::write_fixture(cfg, argv[1],
                            glmdsafx::tiny_config_json(5, 1, false, 16, 2, false, 0, 8, m346));
    std::ofstream oracle(std::string(argv[1]) + "/worker_oracle.json");
    oracle << '{';
    const auto targets = dgpp::glm_dsa_modification_targets(cfg, 0, 1);
    dgpp::GlmDsaModel model(cfg, argv[1], 64, 128, dgpp::GlmDsaResidency::Resident);
    bool refused = false;
    try {
      model.research_forward({1}, false);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    require(refused, "capture must require research opt-in");
    for (int length : {1, 3, 16, 33}) {
      std::vector<int64_t> ids(length);
      for (int i = 0; i < length; ++i) ids[i] = (i * 17 + 11) % cfg.vocab_size;
      const auto baseline = model.session_prefill(0, ids);
      model.session_close(0);
      const auto captured = model.research_forward(ids, true);
      exact(baseline.logits, captured.prediction.logits, "capture changed baseline logits");
      const auto diagnostic = model.forward(ids, true);
      for (int layer = 0; layer + 1 < cfg.num_hidden_layers; ++layer) {
        const auto* expected =
            diagnostic.layer_states[layer].data() + (length - 1) * cfg.hidden_size;
        const auto* actual = captured.residuals.data() + (layer + 1) * cfg.hidden_size;
        require(std::memcmp(expected, actual, cfg.hidden_size * sizeof(uint16_t)) == 0,
                "residual capture boundary/order differs from diagnostic forward");
      }
      require(
          std::memcmp(captured.residuals.data() + cfg.num_hidden_layers * cfg.hidden_size,
                      baseline.final_hidden_bits.data(), cfg.hidden_size * sizeof(uint16_t)) == 0,
          "last residual must be final normalized hidden state");
      const auto repeated = model.research_forward(ids, true);
      exact(captured.residuals, repeated.residuals, "residual capture not repeatable");
      require(captured.residuals.size() ==
                  static_cast<size_t>((cfg.num_hidden_layers + 1) * cfg.hidden_size),
              "residual shape");
      require(captured.projections.size() == targets.size(), "target capture count");
      for (size_t i = 0; i < targets.size(); ++i) {
        const auto& c = captured.projections[i];
        const auto& t = targets[i];
        require(c.id == t.id && c.input.size() == static_cast<size_t>(t.input_features) &&
                    c.output.size() == static_cast<size_t>(t.output_features),
                "target capture identity/shape");
        exact(c.input, repeated.projections[i].input, "input capture not repeatable");
        exact(c.output, repeated.projections[i].output, "output capture not repeatable");
        for (float v : c.output) require(std::isfinite(v), "non-finite captured output");
      }
      auto after = model.session_prefill(0, ids);
      exact(baseline.logits, after.logits, "capture leaked into next session");
      if (length != 1) oracle << ',';
      oracle << '\"' << length << "\":[";
      for (int i = 0; i < 4; ++i) {
        const auto token =
            std::max_element(after.logits.begin(), after.logits.end()) - after.logits.begin();
        if (i) oracle << ',';
        oracle << token;
        if (i + 1 < 4) after = model.session_step(0, token);
      }
      oracle << ']';
      model.session_close(0);
      std::printf("capture %s tokens=%d: exact logits and repeatable tensors\n", argv[2], length);
    }
    oracle << '}';
    require(oracle.good(), "write worker oracle");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "heretic_capture_test: %s\n", e.what());
    return 1;
  }
}
