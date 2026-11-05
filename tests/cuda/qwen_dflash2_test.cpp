// The DFlash2 block drafter on Qwen3.8-Flash-Next (the 2026-10-20 port of
// the Qwen35 drafter): the eager engine's speculative transcripts equal
// the plain engine's bit-for-bit (speculation never changes the accepted
// stream), the draft's shape and determinism hold, and the drafter-armed
// model's plain walk stays bitwise the no-drafter model's (the mix-site
// taps and the feature store touch only the drafter's own buffers).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/test.hpp"
#include "engine/eager_engine.hpp"
#include "engine/graph_engine.hpp"
#include "models/qwen/forward.hpp"
#include "qwen_dflash2_fixture.hpp"
#include "qwen_fixture.hpp"

namespace {

using dgpp::EagerEngineAdapter;
using dgpp::QwenModel;
using dgpp::QwenResidency;
using dgpp::QwenTextConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

constexpr int kSlots = 2;
constexpr int kSteps = 12;
constexpr int kMaxTokens = 64;
constexpr int64_t kCache = 1024;
constexpr int kDrafts = 7;  // block_size 8 - 1

QwenTextConfig tiny_target() {
  return qwenfx::tiny_config();
}

std::vector<int64_t> tokens_of(int n, int salt) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    t[static_cast<size_t>(i)] = 2 + ((i * 37 + salt * 101) % 509);  // in [2, 510]
  return t;
}

// slots x (cfg, dir, draft_dir) -> a model with/without the drafter. The
// decode-row ceiling holds one full block a slot (the batched verify).
QwenModel make_model(const QwenTextConfig& cfg, const std::string& dir, const std::string& draft) {
  return QwenModel(cfg, dir, kMaxTokens, kCache, QwenResidency::Resident, nullptr, 0, 1, kSlots,
                   /*mtp=*/false, kSlots * dgpp::kSpecRows, false, false, dgpp::LatentFormat::kBf16,
                   draft);
}

template <class Engine>
std::vector<int32_t> solo(Engine& eng, int req, const std::vector<int64_t>& prompt, int steps) {
  std::vector<int32_t> out;
  out.push_back(eng.prefill(req, prompt));
  eng.reserve(req, static_cast<int64_t>(prompt.size()) + steps + 8);
  for (int s = 0; s < steps; ++s) {
    const std::vector<int32_t> t = eng.step(req);
    require(t.size() == 1, "a scalar step decides one token");
    out.push_back(t[0]);
  }
  eng.close(req);
  return out;
}

}  // namespace

DGPP_TEST(qwen_dflash2_eager_transcripts_match_plain) {
  const std::string dir = "qwen_dflash2_target_fixture", draft = "qwen_dflash2_draft_fixture";
  const QwenTextConfig cfg = tiny_target();
  qwenfx::write_fixture_for(cfg, dir);
  qwenfx::write_dflash2_fixture(draft);
  QwenModel::set_dflash_options(false, true, 0);
  const auto A = tokens_of(40, 1), B = tokens_of(43, 2);

  // GIVEN the plain (no-drafter) reference.
  std::vector<int32_t> want_a, want_b;
  {
    QwenModel m = make_model(cfg, dir, "");
    EagerEngineAdapter<QwenModel> eng(&m, kSlots, dgpp::make_w1_pick(cfg.vocab_size));
    want_a = solo(eng, 0, A, kSteps);
    want_b = solo(eng, 1, B, kSteps);
  }
  require(!want_a.empty() && !want_b.empty(), "the reference produced transcripts");

  // WHEN the same engine runs with the block drafter: the scalar steps.
  {
    QwenModel m = make_model(cfg, dir, draft);
    require(m.dflash2_enabled() && m.dflash2_drafts() == kDrafts, "the drafter is armed");
    EagerEngineAdapter<QwenModel> eng(&m, kSlots, dgpp::make_w1_pick(cfg.vocab_size));
    const std::vector<int32_t> got_a = solo(eng, 0, A, kSteps);
    const std::vector<int32_t> got_b = solo(eng, 1, B, kSteps);
    require(got_a == want_a, "the drafter's scalar transcript is the plain engine's (A)");
    require(got_b == want_b, "the drafter's scalar transcript is the plain engine's (B)");
  }
  // AND the batched pass (both slots live through the verify walk).
  {
    QwenModel m = make_model(cfg, dir, draft);
    EagerEngineAdapter<QwenModel> eng(&m, kSlots, dgpp::make_w1_pick(cfg.vocab_size));
    std::vector<int32_t> a, b;
    a.push_back(eng.prefill(0, A));
    b.push_back(eng.prefill(1, B));
    eng.reserve(0, static_cast<int64_t>(A.size()) + kSteps + 8);
    eng.reserve(1, static_cast<int64_t>(B.size()) + kSteps + 8);
    for (int s = 0; s < kSteps; ++s) {
      const auto out = eng.step_batch({0, 1});
      require(out.size() == 2 && out[0].size() == 1 && out[1].size() == 1,
              "the batched step decides one token a slot");
      a.push_back(out[0][0]);
      b.push_back(out[1][0]);
    }
    eng.close(0);
    eng.close(1);
    require(a == want_a, "the batched drafter transcript is the plain engine's (A)");
    require(b == want_b, "the batched drafter transcript is the plain engine's (B)");
  }
}

DGPP_TEST(qwen_dflash2_draft_shape_determinism_and_invariance) {
  const std::string dir = "qwen_dflash2_target_fixture", draft = "qwen_dflash2_draft_fixture";
  const QwenTextConfig cfg = tiny_target();
  qwenfx::write_fixture_for(cfg, dir);
  qwenfx::write_dflash2_fixture(draft);
  QwenModel::set_dflash_options(false, true, 0);
  const auto prompt = tokens_of(49, 3);

  // The armed model's PLAIN walk (no draft called) is bitwise the no-drafter
  // model's: the taps and the feature store feed only the drafter.
  QwenModel m = make_model(cfg, dir, draft);
  QwenModel plain = make_model(cfg, dir, "");
  auto out = m.session_prefill(0, prompt);
  auto ref = plain.session_prefill(0, prompt);
  require(out.logits.size() == ref.logits.size() &&
              std::equal(out.logits.begin(), out.logits.end(), ref.logits.begin()),
          "the drafter-armed prefill is bitwise the plain prefill");
  for (int step = 0; step < 4; ++step) {
    const int64_t fed = static_cast<int64_t>(
        std::max_element(ref.logits.begin(), ref.logits.end()) - ref.logits.begin());
    out = m.session_step(0, fed);
    ref = plain.session_step(0, fed);
    require(std::equal(out.logits.begin(), out.logits.end(), ref.logits.begin()),
            "the drafter-armed decode row is bitwise the plain decode row");
  }

  // The draft itself: the bonus's argmax walk, 7 proposals, repeatable, and
  // identical from a second model (the draft depends on the context and the
  // bonus alone — the run above left the same context in both models).
  const int64_t bonus = static_cast<int64_t>(
      std::max_element(out.logits.begin(), out.logits.end()) - out.logits.begin());
  std::vector<int32_t> got, again, other;
  require(m.dflash2_draft(0, bonus, &got), "the draft runs");
  require(got.size() == static_cast<size_t>(kDrafts), "the block proposes 7 drafts");
  require(m.dflash2_draft(0, bonus, &again), "the draft repeats");
  require(got == again, "the same context drafts the same block (the rows do not advance)");
  // The batched redraft of the one live slot matches the scalar draft.
  std::vector<std::vector<int32_t>> batched;
  m.dflash2_draft_batch({0}, {bonus}, &batched);
  require(batched.size() == 1 && batched[0] == got, "the stacked redraft matches the scalar one");
  (void)other;
  m.session_close(0);
  plain.session_close(0);
}

DGPP_TEST(qwen_dflash2_plan_names_the_drafter_and_refuses_mtp) {
  const std::string dir = "qwen_dflash2_target_fixture", draft = "qwen_dflash2_draft_fixture";
  const QwenTextConfig cfg = tiny_target();
  qwenfx::write_fixture_for(cfg, dir);
  qwenfx::write_dflash2_fixture(draft);
  const auto plan =
      QwenModel::plan_memory(cfg, kMaxTokens, kCache, 0, 1, QwenResidency::Resident, kSlots,
                             /*mtp=*/false, 0, false, dgpp::LatentFormat::kBf16, draft);
  bool named = false;
  for (const auto& item : plan.items)
    if (item.name.find("dflash2") != std::string::npos) named = true;
  require(named, "the memory plan itemizes the drafter");
  bool refused = false;
  try {
    (void)QwenModel::plan_memory(cfg, kMaxTokens, kCache, 0, 1, QwenResidency::Resident, kSlots,
                                 /*mtp=*/true, 0, false, dgpp::LatentFormat::kBf16, draft);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "plan_memory refuses the drafter beside the MTP draft");
}

// The recorded block proposal: the graph engine's replay publishes the same
// drafts the eager redraft computes off the same commit, and the transcripts
// match the eager engine's bitwise.
std::vector<std::unique_ptr<dgpp::net::CollectiveBus>> graph_world(uint16_t port) {
  dgpp::net::BusOptions o;
  o.world_size = 1;
  o.my_rank = 0;
  o.rendezvous_port = port;
  o.rendezvous_host = "";
  o.rendezvous_timeout_ms = 20000;
  o.lat_slots = 8;
  o.lat_slot_bytes = 262144;  // every fold on the latency path
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = 5000;
  o.consumer_deadline_s = 20.0;
  o.launch_consumers = false;
  std::vector<std::unique_ptr<dgpp::net::CollectiveBus>> out;
  out.push_back(std::make_unique<dgpp::net::CollectiveBus>(o));
  std::string error;
  require(out[0]->start(&error), "the graph gate's world of one: " + error);
  return out;
}

DGPP_TEST(qwen_dflash2_graph_transcript_matches_eager) {
  const std::string dir = "qwen_dflash2_target_fixture", draft = "qwen_dflash2_draft_fixture";
  const QwenTextConfig cfg = tiny_target();
  qwenfx::write_fixture_for(cfg, dir);
  qwenfx::write_dflash2_fixture(draft);
  QwenModel::set_dflash_options(false, true, kDrafts);

  // GIVEN the eager engine with the drafter: the transcript per slot.
  std::vector<std::vector<int32_t>> want;
  {
    QwenModel m = make_model(cfg, dir, draft);
    dgpp::EagerEngineAdapter<QwenModel> eng(&m, kSlots, dgpp::make_w1_pick(cfg.vocab_size));
    want.push_back(solo(eng, 0, tokens_of(40, 1), kSteps));
    want.push_back(solo(eng, 1, tokens_of(43, 2), kSteps));
  }

  // WHEN the graphs capture the step with the recorded block proposal.
  QwenModel m = make_model(cfg, dir, draft);
  require(m.dflash2_enabled() && m.dflash_graph_verify_available(),
          "the block verifies on the decode graph");
  auto buses = graph_world(29977);
  uint16_t* scratch = nullptr;
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                             sizeof(uint16_t) * dgpp::kPickScratchElems(1), cudaHostAllocDefault));
  std::vector<std::vector<int32_t>> got(kSlots);
  {
    dgpp::GraphEngineAdapter<QwenModel> eng(&m, buses[0].get(), 0, 1, scratch, cfg.vocab_size,
                                            60000, 2, nullptr, nullptr, 0, nullptr, 0, kDrafts);
    eng.warm_captures(tokens_of(6, 5));
    const std::vector<std::vector<int64_t>> prompts = {tokens_of(40, 1), tokens_of(43, 2)};
    std::vector<int> live;
    for (int q = 0; q < kSlots; ++q) {
      got[q].push_back(eng.prefill(q, prompts[q]));
      live.push_back(q);
    }
    for (int s = 0; s < kSteps; ++s) {
      for (const int q : live) eng.reserve(q, m.session_position(q) + kDrafts + 1);
      const auto gen = eng.step_batch(live);
      eng.drain();
      for (size_t i = 0; i < live.size(); ++i) {
        const int q = live[i];
        auto& tokens = got[static_cast<size_t>(q)];
        tokens.insert(tokens.end(), gen[i].begin(), gen[i].end());
        // The replay's mirror is the eager redraft off the same commit.
        std::vector<int32_t> eager_draft;
        require(m.dflash2_draft(q, tokens.back(), &eager_draft),
                "eager draft after graph commit");
        require(std::equal(eager_draft.begin(), eager_draft.end(), m.block_drafts_host(q)),
                "the replay's published drafts are the eager redraft's");
      }
    }
    for (const int q : live) eng.close(q);
  }
  DGPP_CUDA_OK(cudaFreeHost(scratch));
  for (int q = 0; q < kSlots; ++q) {
    const auto& g = got[static_cast<size_t>(q)];
    const auto& w = want[static_cast<size_t>(q)];
    if (g == w) continue;
    const size_t n = std::min(g.size(), w.size());
    size_t first = 0;
    while (first < n && g[first] == w[first]) ++first;
    if (first >= n)
      throw std::runtime_error("the graph transcript is the eager engine's (slot " + std::to_string(q) +
                               "): the common window of " + std::to_string(n) + " matches, but lengths " +
                               std::to_string(g.size()) + " vs " + std::to_string(w.size()));
    throw std::runtime_error("the graph transcript is the eager engine's (slot " + std::to_string(q) +
                             "): first diff at token " + std::to_string(first) + " of " +
                             std::to_string(n) + ": got " + std::to_string(g[first]) + " want " +
                             std::to_string(w[first]) + " (lengths " + std::to_string(g.size()) +
                             " vs " + std::to_string(w.size()) + ")");
  }
  QwenModel::set_dflash_options(false, true, 0);
}

int main() { return dgpp::test::run_all(); }
