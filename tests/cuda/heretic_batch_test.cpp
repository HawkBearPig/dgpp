#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "adapters/runtime.hpp"
#include "glm_dsa_fixture.hpp"
#include "models/glm/route_audit.hpp"
#include "models/glm_dsa/model.hpp"

namespace {
using namespace dgpp;
constexpr int kSlots = 16, kRows = 512;
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void exact(const GlmDsaModel::Outputs& a, const GlmDsaModel::Outputs& b, const char* message) {
  require(!a.logits.empty() && a.logits.size() == b.logits.size(), "logit shape");
  require(std::memcmp(a.logits.data(), b.logits.data(), a.logits.size() * sizeof(float)) == 0,
          message);
}
void close(const GlmDsaModel::Outputs& a, const GlmDsaModel::Outputs& b, const char* message) {
  require(!a.logits.empty() && a.logits.size() == b.logits.size(), "logit shape");
  double error = 0, norm = 0;
  for (size_t i = 0; i < a.logits.size(); ++i) {
    require(std::isfinite(a.logits[i]) && std::isfinite(b.logits[i]), "finite logits");
    error += std::pow(a.logits[i] - b.logits[i], 2);
    norm += std::pow(b.logits[i], 2);
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-30));
  if (relative >= 0.01) std::printf("%s: relative L2 %.8g\n", message, relative);
  require(relative < 0.01, message);
  const int vocab = b.lm_vocab_count;
  const auto rows = b.logits.size() / vocab;
  const auto audit = glm_route::audit_top1_near_ties(a.logits.data(), b.logits.data(), vocab, rows);
  glm_route::certify_top1_near_ties(audit, rows, message);
}
std::vector<int64_t> tokens(int count, int vocab, int salt) {
  std::vector<int64_t> ids(count);
  for (int i = 0; i < count; ++i) ids[i] = (17 * i + salt) % vocab;
  return ids;
}
adapters::Manifest factors(const GlmDsaTextConfig& cfg, bool zero) {
  adapters::Manifest manifest;
  manifest.world = 1;
  manifest.id = zero ? "zero" : "nonzero";
  for (const auto& target : glm_dsa_modification_targets(cfg, 0, 1)) {
    adapters::ProjectionFactors p;
    p.target = target;
    p.rank = 3;
    p.a.resize(p.rank * target.input_features);
    p.b.resize(p.rank * target.output_features);
    for (size_t i = 0; i < p.a.size(); ++i)
      p.a[i] = 0.01f * (static_cast<int>((i * 7 + target.layer) % 19) - 9);
    for (size_t i = 0; i < p.b.size(); ++i)
      p.b[i] = zero ? 0 : 0.03f * (static_cast<int>((i * 3 + target.layer) % 17) - 8);
    manifest.projections.push_back(std::move(p));
  }
  return manifest;
}
std::unique_ptr<GlmDsaModel> make(const GlmDsaTextConfig& cfg, const std::string& dir,
                                  const adapters::Manifest* manifest) {
  auto model = std::make_unique<GlmDsaModel>(cfg, dir, kRows, 8192, GlmDsaResidency::Resident,
                                             nullptr, 0, 1, kSlots, false, kSlots);
  if (manifest)
    model->set_runtime_adapter(
        std::make_unique<adapters::Runtime>(*manifest, 0, 1, kRows, cfg.hidden_size,
                                            cfg.num_hidden_layers, adapters::SumFp32{}, true),
        true);
  return model;
}
void close_all(GlmDsaModel& model) {
  for (int i = 0; i < kSlots; ++i) model.session_close(i);
  require(model.kv_blocks_in_use() == 0, "closed requests retain KV blocks");
}
void check(const GlmDsaTextConfig& cfg, const std::string& dir) {
  const auto nonzero = factors(cfg, false), zero = factors(cfg, true);
  auto model = make(cfg, dir, &nonzero), reference = make(cfg, dir, &nonzero);
  std::vector<std::vector<int64_t>> prompts;
  for (int i = 0; i < kSlots; ++i) prompts.push_back(tokens(1 + i * 2, cfg.vocab_size, i + 3));
  for (int count : {8, 16}) {
    // A nontrivial slot order catches scratch indexed by slot instead of row.
    std::vector<int> reqs;
    std::vector<const std::vector<int64_t>*> inputs;
    for (int i = 0; i < count; ++i) {
      reqs.push_back((i * 7 + 15) % kSlots);
      inputs.push_back(&prompts[i]);
    }
    auto base = make(cfg, dir, nullptr), identity = make(cfg, dir, &zero);
    auto base_reference = make(cfg, dir, nullptr);
    const auto plain = base->session_prefill_group(reqs, inputs);
    const auto unchanged = identity->session_prefill_group(reqs, inputs);
    const auto grouped = model->session_prefill_group(reqs, inputs);
    bool changed = false;
    for (int i = 0; i < count; ++i) {
      exact(plain[i], unchanged[i], "zero adapter changes grouped prefill");
      close(plain[i], base_reference->session_prefill(reqs[i], *inputs[i]),
            "native grouped prefill control");
      changed |= plain[i].logits != grouped[i].logits;
      close(grouped[i], reference->session_prefill(reqs[i], *inputs[i]),
            "adapted grouped prefill differs from isolated request");
    }
    require(changed, "nonzero adapter is not executing");
    for (int step = 0; step < 3; ++step) {
      std::vector<std::vector<int64_t>> feeds;
      for (int i = 0; i < count; ++i) feeds.push_back({(i + step + 7) % cfg.vocab_size});
      const auto decoded = model->session_verify_batch(reqs, feeds);
      const auto plain_decode = base->session_verify_batch(reqs, feeds);
      const auto zero_decode = identity->session_verify_batch(reqs, feeds);
      for (int i = 0; i < count; ++i) {
        exact(plain_decode[i], zero_decode[i], "zero adapter changes batched decode");
        close(plain_decode[i], base_reference->session_step(reqs[i], feeds[i][0]),
              "native batched decode control");
        close(decoded[i], reference->session_step(reqs[i], feeds[i][0]),
              "adapted batched decode differs from isolated request");
      }
    }
    close_all(*model);
    close_all(*reference);
    auto changed_prompt = tokens(prompts[0].size(), cfg.vocab_size, 101);
    inputs[0] = &changed_prompt;
    const auto neighbors = model->session_prefill_group(reqs, inputs);
    for (int i = 1; i < count; ++i)
      exact(grouped[i], neighbors[i], "adapted prefill reads another request");
    close_all(*model);
  }

  // Two unequal prefills share a walk with two decoders, then yield. The
  // isolated oracle follows the actual cuts, including the bulk GEMM path.
  const auto a = tokens(177, cfg.vocab_size, 3), b = tokens(53, cfg.vocab_size, 5);
  auto ca = model->session_prefill_begin(3, a, 256, 144);
  auto cb = model->session_prefill_begin(11, b, 256, 32);
  auto ra = reference->session_prefill_begin(3, a, 256, 144);
  auto rb = reference->session_prefill_begin(11, b, 256, 32);
  ra.relative_budget = rb.relative_budget = true;
  for (int req : {15, 4}) {
    (void)model->session_prefill(req, b);
    (void)reference->session_prefill(req, b);
  }
  while (ca.next < ca.end || cb.next < cb.end) {
    std::vector<GlmDsaModel::PrefillCursor*> cursors, oracles;
    std::vector<int64_t> budgets;
    if (ca.next < ca.end) {
      cursors.push_back(&ca);
      oracles.push_back(&ra);
      budgets.push_back(144);
    }
    if (cb.next < cb.end) {
      cursors.push_back(&cb);
      oracles.push_back(&rb);
      budgets.push_back(32);
    }
    const auto mixed =
        model->session_prefill_verify_group(cursors, budgets, {15, 4}, {{7, 8}, {9}});
    close(mixed[0], reference->session_verify(15, {7, 8}), "mixed decoder 15");
    close(mixed[1], reference->session_verify(4, {9}), "mixed decoder 4");
    for (size_t i = 0; i < cursors.size(); ++i) {
      reference->session_prefill_advance(*oracles[i], cursors[i]->next - oracles[i]->next);
      close(cursors[i]->output, oracles[i]->output, "mixed prefill correction");
    }
    model->session_rollback(15, 1, 2);
    reference->session_rollback(15, 1, 2);
  }
  bool rejected = false;
  try {
    model->set_runtime_adapter(nullptr, true);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected && model->has_runtime_adapter(), "replacement accepted with nonzero slots live");
  close_all(*model);
  close_all(*reference);

  auto cancelled = model->session_prefill_begin(15, a, 256, 16);
  rejected = false;
  try {
    model->set_runtime_adapter(nullptr, true);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "replacement accepted while a zero-position prefill owns KV");
  model->session_prefill_advance(cancelled);
  model->session_close(15);
  exact(model->session_prefill(15, b), reference->session_prefill(15, b),
        "cancelled prefill leaks into reused slot");
  close_all(*model);
  close_all(*reference);
  model->set_runtime_adapter(nullptr, true);
  require(!model->has_runtime_adapter(), "idle adapter reset failed");
  auto base = make(cfg, dir, nullptr);
  exact(model->session_prefill(15, b), base->session_prefill(15, b), "reset did not restore base");
  std::puts(
      "adapter C8/C16: identity, isolated oracle, neighbors, mixed walks, cancellation and reset "
      "passed");
}
void check_sparse_isolation(const GlmDsaTextConfig& cfg, const std::string& dir) {
  const auto manifest = factors(cfg, false);
  auto model = make(cfg, dir, &manifest), reference = make(cfg, dir, &manifest);
  std::vector<int> reqs;
  std::vector<std::vector<int64_t>> prompts;
  for (int i = 0; i < kSlots; ++i) {
    reqs.push_back((i * 7 + 15) % kSlots);
    prompts.push_back(tokens(1 + i * 2, cfg.vocab_size, i + 3));
  }
  std::vector<const std::vector<int64_t>*> inputs;
  for (const auto& p : prompts) inputs.push_back(&p);
  const auto first = model->session_prefill_group(reqs, inputs);
  const auto neighbor = tokens(prompts[0].size(), cfg.vocab_size, 103);
  inputs[0] = &neighbor;
  const auto second = reference->session_prefill_group(reqs, inputs);
  for (int i = 1; i < kSlots; ++i)
    exact(first[i], second[i], "sparse adapted prefill reads another request");
  for (int step = 0; step < 3; ++step) {
    std::vector<std::vector<int64_t>> feeds(kSlots, {7 + step});
    const auto a = model->session_verify_batch(reqs, feeds);
    feeds[0] = {101 + step};
    const auto b = reference->session_verify_batch(reqs, feeds);
    for (int i = 1; i < kSlots; ++i)
      exact(a[i], b[i], "sparse adapted decode reads another request");
  }
  close_all(*model);
  close_all(*reference);
  std::puts("sparse adapter C16: neighboring requests remain bitwise isolated");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 3, "usage: heretic_batch_test FIXTURE_DIR int8|m346");
    const bool m346 = std::string(argv[2]) == "m346";
    // Like the bulk-prefill numerical fixture, select every expert and keep
    // attention dense for cross-shape comparisons. Sparse near-tie decisions
    // can diverge between batch geometries even without an adapter.
    const std::string json = glmdsafx::tiny_config_json(5, 1, false, 256, 8, false, 0, 8, m346);
    const auto tree = dgpp::minijson::parse(json);
    const auto cfg = GlmDsaTextConfig::parse(tree.root);
    const std::string dir = std::string(argv[1]) + "_batch";
    glmdsafx::write_fixture(cfg, dir, json.c_str());
    check(cfg, dir);
    check_sparse_isolation(glmdsafx::tiny_config(5, 1, false, false, 8, m346), argv[1]);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "heretic_batch_test: %s\n", e.what());
    return 1;
  }
}
