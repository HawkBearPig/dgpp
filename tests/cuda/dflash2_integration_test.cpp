// Shared drafter lifecycle: prefix snapshots and eager/recorded reducer changes.
#include <algorithm>
#include <barrier>
#include <chrono>
#include <filesystem>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/test.hpp"
#include "engine/eager_engine.hpp"
#include "engine/graph_engine.hpp"
#include "engine/nvme_tier.hpp"
#include "engine/prefix_arena.hpp"
#include "engine/tp_bus.hpp"
#include "models/minimax/forward.hpp"
#include "models/qwen/forward.hpp"
#include "models/qwen/model35.hpp"
#include "qwen35_fixture.hpp"
#include "qwen_dflash2_fixture.hpp"

namespace {
void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

std::vector<int64_t> prompt(int rows, int salt) {
  std::vector<int64_t> out(rows);
  for (int i = 0; i < rows; ++i) out[i] = 2 + (37 * i + salt * 101) % 509;
  return out;
}

struct ReplaySetting {
  const char* value = std::getenv("DGPP_QWEN_GDN_REPLAY");
  bool present = value != nullptr;
  std::string old = value ? value : "";
  ~ReplaySetting() {
    if (present) setenv("DGPP_QWEN_GDN_REPLAY", old.c_str(), 1);
    else unsetenv("DGPP_QWEN_GDN_REPLAY");
  }
};

template <class Model>
auto draft_plane(Model& model, int layer) {
  if constexpr (std::is_same_v<Model, dgpp::MinimaxModel>) {
    const auto v = model.pool().draft_view(layer);
    struct Plane {
      uint16_t* k_cache;
      uint16_t* v_cache;
    };
    return Plane{v.k, v.v};
  } else if constexpr (requires { model.pool().draft_view(layer); })
    return model.pool().draft_view(layer);
  else
    return model.pool().view(1 + layer);  // the tiny Qwen35 target has one full layer
}

template <class Model>
std::vector<uint16_t> context(Model& model, int req) {
  DGPP_CUDA_OK(cudaStreamSynchronize(model.stream()));
  const int rows = static_cast<int>(model.session_position(req));
  constexpr int width = std::is_same_v<Model, dgpp::MinimaxModel> ? 4 * 128 : 2 * 128;
  const auto* table = model.pool().request_table_row(req);
  const int block = model.kv_block_tokens();
  std::vector<uint16_t> out;
  for (int layer = 0; layer < 2; ++layer) {
    const auto plane = draft_plane(model, layer);
    for (const uint16_t* data : {plane.k_cache, plane.v_cache}) {
      const size_t start = out.size();
      out.resize(start + rows * width);
      for (int r = 0; r < rows; ++r) {
        const size_t physical = static_cast<size_t>(table[r / block]) * block + r % block;
        DGPP_CUDA_OK(cudaMemcpy(out.data() + start + r * width, data + physical * width,
                                width * sizeof(uint16_t), cudaMemcpyDeviceToHost));
      }
    }
  }
  return out;
}

template <class Model, class Factory>
void check_prefix(Factory make) {
  auto model = make(nullptr, 0, 1);
  dgpp::PrefixArena<Model> arena(model.get(), 1);
  for (int round = 0; round < 3; ++round) {
    // A neighboring live request owns physical block zero. The snapshot
    // includes both a shared full block and an independently copied tail.
    model->session_prefill(0, prompt(40, 1 + round));
    model->session_prefill(1, prompt(model->kv_block_tokens() + 16, 4 + round));
    const auto peer = context(*model, 0), original = context(*model, 1);
    arena.snapshot(1, 0, model->kv_block_tokens() + 16);
    require(context(*model, 0) == peer, "snapshot overwrote a neighboring request's draft context");
    require(context(*model, 1) == original, "snapshot overwrote the source draft context");
    model->session_close(1);
    arena.attach(1, 0);
    require(context(*model, 1) == original, "attached draft context differs from the cold prefix");
    require(context(*model, 0) == peer, "attach overwrote a neighboring request's draft context");
    // A subsequent eager redraft also exercises the restored table.
    std::vector<int32_t> drafts;
    require(model->dflash2_draft(1, 7, &drafts) && drafts.size() == 7, "attached prefix cannot draft");
    model->session_close(0);
    model->session_close(1);
    arena.release(0);
    require(model->kv_blocks_in_use() == 0, "snapshot or attach leaked draft blocks");
  }
}

template <class Model, class Factory>
auto check_graph(Factory make, uint16_t port, bool sampled = false, bool screened = false,
                 std::vector<std::vector<uint8_t>>* states = nullptr, int slots = 2) {
  constexpr int world = 2;
  std::vector<std::unique_ptr<dgpp::net::CollectiveBus>> buses;
  for (int rank = 0; rank < world; ++rank) {
    dgpp::net::BusOptions o;
    o.world_size = world;
    o.my_rank = rank;
    o.rendezvous_port = port;
    o.rendezvous_host = rank ? "127.0.0.1" : "";
    o.rendezvous_timeout_ms = 20000;
    o.lat_slots = o.bulk_slots = 8;
    o.lat_slot_bytes = o.bulk_slot_bytes = 262144;
    o.qp_depth = 1024;
    o.completion_timeout_ms = 10000;
    o.consumer_deadline_s = 60;
    o.launch_consumers = false;
    buses.push_back(std::make_unique<dgpp::net::CollectiveBus>(o));
  }
  std::vector<std::string> errors(world);
  std::vector<std::thread> workers;
  for (int rank = 0; rank < world; ++rank)
    workers.emplace_back([&, rank] { (void)buses[rank]->start(&errors[rank]); });
  for (auto& worker : workers) worker.join();
  for (const auto& error : errors) require(error.empty(), "start bus: " + error);
  workers.clear();
  std::barrier ready(world);
  std::vector<std::vector<std::vector<int32_t>>> transcripts(world);
  for (int rank = 0; rank < world; ++rank) workers.emplace_back([&, rank] {
    uint16_t *scratch = nullptr, *prefix = nullptr, *gather = nullptr;
    bool arrived = false;
    try {
      dgpp::BusBoundaryReducer reducer(*buses[rank]);
      auto model = make(&reducer, rank, world);
      dgpp::PrefixArena<Model> snapshots(model.get(), states ? 1 : 0);
      if constexpr (requires { model->draft_head_blocks(); })
        if (screened) require(model->draft_head_blocks() == 1, "drafter did not initialize its requested head screen");
      DGPP_CUDA_OK(cudaHostAlloc(&scratch, sizeof(uint16_t) * dgpp::kPickScratchElems(world),
                                  cudaHostAllocDefault));
      if (sampled) {
        DGPP_CUDA_OK(cudaHostAlloc(&prefix, 2 * dgpp::fabric_sampling_prefix_scratch_elems(world),
                                    cudaHostAllocDefault));
        DGPP_CUDA_OK(cudaHostAlloc(&gather, 2 * dgpp::sampling_gather_scratch_elems(512),
                                    cudaHostAllocDefault));
      }
      {
        dgpp::GraphEngineAdapter<Model> engine(model.get(), buses[rank].get(), rank, world,
                                               scratch, 512, 60000, 2, prefix, gather);
        require(model->dflash2_proposals_armed() == sampled, "engine did not arm sampled draft proposals");
        engine.configure_verify_schedule(true, 8.f, 0.045f, 1, 28.f, false, 3);
        int tick = 0;
        engine.set_verify_depth_hook([&](int, int, const float*, int) {
          const int depths[] = {1, 4, 7};
          return depths[tick++ % 3];
        });
        arrived = true;
        ready.arrive_and_wait();
        engine.warm_captures(prompt(4, 0));
        transcripts[rank].resize(slots);
        std::vector<int> live;
        for (int req = 0; req < slots; ++req) live.push_back(req);
        for (int req : live) {
          dgpp::sample::Params params;
          params.temperature = sampled ? .8f : 0.f;
          params.top_k = 16;
          engine.configure_sampling(req, params, 1234 + req);
          transcripts[rank][req].push_back(engine.prefill(req, prompt(48 + req * 4, req)));
        }
        for (int step = 0; step < 8; ++step) {
          for (int req : live) engine.reserve(req, model->session_position(req) + 8);
          const auto generated = engine.step_batch(live);
          engine.drain();
          for (size_t i = 0; i < live.size(); ++i) {
            const int req = live[i];
            auto& out = transcripts[rank][req];
            require(!generated[i].empty(), "graph made no progress");
            out.insert(out.end(), generated[i].begin(), generated[i].end());
            if (screened) {
              const int k = model->block_candidates();
              const int32_t* ids = model->block_candidates_host(req);
              for (int d = 0; d < 7; ++d) {
                std::set<int> blocks;
                for (int c = 0; c < k; ++c) {
                  require(ids[d * k + c] >= 0 && ids[d * k + c] < 512, "invalid screened candidate");
                  blocks.insert(ids[d * k + c] / 128);
                }
                require(blocks.size() <= world, "drafter bypassed its one-block-per-rank head screen");
              }
            }
            std::vector<int32_t> eager;
            require(model->dflash2_draft(req, out.back(), &eager), "eager draft after replay failed");
            require(sampled || std::equal(eager.begin(), eager.end(), model->block_drafts_host(req)),
                    "recorded draft differs from eager draft after restoring the reducer");
          }
          if (states && rank == 0 &&
              model->session_position(live.back()) % model->session_snapshot_align() == 0) {
            // Snapshot after speculative commits: replay must materialize
            // exactly the accepted recurrent state, including after a
            // batched-to-scalar transition and changing verify depths.
            const int req = live.back();
            snapshots.snapshot(req, 0, model->session_position(req));
            DGPP_CUDA_OK(cudaStreamSynchronize(model->stream()));
            states->emplace_back(snapshots.bytes());
            DGPP_CUDA_OK(cudaMemcpy(states->back().data(), snapshots.slot_data(0), snapshots.bytes(),
                                    cudaMemcpyDeviceToHost));
            snapshots.release(0);
          }
          if (step >= 3 && live.size() > 1) {
            engine.close(live.front());
            // Scalar replay after fixed-batch replays. A copy, not erase():
            // GCC 13's -Warray-bounds misreads the erase's move once the
            // TU's inlining changes (a one-element vector's end pointer).
            live = std::vector<int>(live.begin() + 1, live.end());
          }
        }
        for (int req : live) engine.close(req);
      }
    } catch (const std::exception& e) {
      errors[rank] = e.what();
      if (!arrived) ready.arrive_and_wait();
    }
    if (scratch) cudaFreeHost(scratch);
    if (prefix) cudaFreeHost(prefix);
    if (gather) cudaFreeHost(gather);
  });
  for (auto& worker : workers) worker.join();
  for (auto& bus : buses) bus->stop();
  for (const auto& error : errors) require(error.empty(), error);
  require(transcripts[0] == transcripts[1], "ranks produced different transcripts");
  return transcripts[0];
}

auto minimax_factory(bool draft = true, int slots = 2) {
  const std::string dir = "minimax_dspark_target_fixture";
  const auto cfg = dgpp::MinimaxConfig::from_json_file(dir + "/config.json");
  return [=](dgpp::BoundaryReducer* reducer, int rank, int world) {
    return std::make_unique<dgpp::MinimaxModel>(cfg, dir, 256, 1024,
                                                dgpp::LoaderResidency::Resident, reducer, rank,
                                                world, slots, draft, slots * 8);
  };
}
auto flash_factory() {
  const std::string dir = "qwen_dflash2_target_fixture", draft = "qwen_dflash2_draft_fixture";
  const auto cfg = qwenfx::tiny_config();
  qwenfx::write_fixture_for(cfg, dir);
  qwenfx::write_dflash2_fixture(draft);
  return [=](dgpp::BoundaryReducer* reducer, int rank, int world) {
    return std::make_unique<dgpp::QwenModel>(cfg, dir, 128, 1024, dgpp::QwenResidency::Resident,
        reducer, rank, world, 2, false, 16, false, false, dgpp::LatentFormat::kBf16, draft);
  };
}

auto qwen35_factory() {
  const std::string dir = "qwen35_dflash2_target_fixture", draft = "qwen_dflash2_draft_fixture";
  const auto cfg = qwen35fx::tiny_config();
  qwen35fx::write_fixture(dir);
  qwenfx::write_dflash2_fixture(draft);
  return [=](dgpp::BoundaryReducer* reducer, int rank, int world) {
    return std::make_unique<dgpp::Qwen35Model>(cfg, dir, 128, 1024, dgpp::LoaderResidency::Resident,
        reducer, rank, world, 2, 16, false, draft);
  };
}
}  // namespace

// The NVMe cold tier's round trip (issue #26) over a drafter world: the
// snapshot's blocks carry the drafter's K/V context on the shared table, so
// the planes the tier lists must include the draft's — a restored entry
// that attached with stale draft context would still verify exactly and
// collapse the acceptance with no diagnostic.
template <class Model, class Factory>
void check_prefix_tier(Factory make, const char* tag) {
  auto model = make(nullptr, 0, 1);
  const int prefix_tokens = static_cast<int>(model->kv_block_tokens()) + 16;
  dgpp::PrefixArena<Model> arena(model.get(), 2);
  model->session_prefill(0, prompt(40, 1));
  model->session_prefill(1, prompt(prefix_tokens, 4));
  const auto peer = context(*model, 0), original = context(*model, 1);
  arena.snapshot(1, 0, prefix_tokens);
  model->session_close(1);
  const int64_t blocks_before = model->kv_blocks_in_use();
  const std::string slab_dir =
      (std::filesystem::temp_directory_path() / ("dgpp_dflash2_tier_" + std::string(tag))).string();
  std::filesystem::create_directories(slab_dir);
  dgpp::sched::SchedulerEngine::DiskEnable en;
  en.slab_path = slab_dir + "/rank0.slab";
  en.capacity_bytes = size_t{64} << 20;
  en.chunk_bytes = size_t{1} << 20;
  size_t cache_block_bytes = 0;
  for (const auto& plane : model->pool().planes()) cache_block_bytes += plane.block_bytes;
  while (en.chunk_bytes < cache_block_bytes) en.chunk_bytes *= 2;
  en.rank = 0;
  en.allow_buffered = true;  // a tmpfs temp directory refuses direct I/O
  auto tier = std::make_unique<dgpp::NvmeTier<Model>>(model.get(), &arena, en);
  const std::vector<dgpp::CachePlane> planes = model->cache_planes();
  require(tier->block_bytes() == dgpp::cache_block_bytes(planes), "tier: the record is the planes' bytes");
  const auto draft0 = draft_plane(*model, 0);
  bool draft_listed = false;
  for (const dgpp::CachePlane& pl : planes)
    draft_listed = draft_listed || pl.base == reinterpret_cast<const uint8_t*>(draft0.k_cache);
  require(draft_listed, "tier: the planes omit the drafter's K/V context");
  const auto wait = [&](uint64_t op) {
    for (int i = 0; i < 2000; ++i) {
      tier->pump();
      for (const auto& c : tier->poll())
        if (c.op == op) return c.ok;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    throw std::runtime_error("tier: op " + std::to_string(op) + " never completed");
  };
  // One full block and a partial copy, at this family's physical block size.
  const dgpp::sched::SchedulerEngine::DiskBlocks ids = tier->entry_blocks(0);
  require(ids.identities.size() == 1 && ids.partial_identity != 0, "tier: the entry is one full block and a partial");
  dgpp::sched::SchedulerEngine::DiskSpill spill;
  spill.op = 1;
  spill.slot = 0;
  spill.blob_page = 0;
  spill.pages = {tier->blob_pages()};
  spill.partial_page = tier->blob_pages() + 1;
  spill.ids = prompt(prefix_tokens, 4);
  spill.ids_page = tier->blob_pages() + 2;
  spill.ids_pages = 1;
  tier->spill_begin(spill);
  require(wait(1), "tier: the spill completed");
  dgpp::sched::SchedulerEngine::DiskRestore rs;
  rs.op = 2;
  rs.slot = 1;
  rs.position = prefix_tokens;
  rs.mtp_position = arena.meta(0).mtp_position;
  rs.blob_page = 0;
  rs.pages = {tier->blob_pages()};
  rs.partial_page = tier->blob_pages() + 1;
  (void)tier->restore_begin(rs);
  require(wait(2), "tier: the restore completed");
  require(model->kv_blocks_in_use() == blocks_before + 2, "tier: the restore took two fresh blocks");
  // Attached from the restored slot: the draft context is the cold prefix's.
  arena.attach(1, 1);
  require(context(*model, 1) == original, "tier: the restored draft context differs from the cold prefix");
  require(context(*model, 0) == peer, "tier: the restore overwrote a neighboring request's draft context");
  std::vector<int32_t> drafts;
  require(model->dflash2_draft(1, 7, &drafts) && drafts.size() == 7, "tier: the restored prefix cannot draft");
  model->session_close(0);
  model->session_close(1);
  arena.release(0);
  arena.release(1);
  tier.reset();
  require(model->kv_blocks_in_use() == 0, "tier: the round trip leaked blocks");
  std::filesystem::remove_all(slab_dir);
}

DGPP_TEST(minimax_dspark_prefix_snapshot) {
  check_prefix<dgpp::MinimaxModel>(minimax_factory());
}
DGPP_TEST(minimax_dspark_prefix_tier) {
  check_prefix_tier<dgpp::MinimaxModel>(minimax_factory(), "minimax");
}
DGPP_TEST(minimax_dspark_graph_reducer_tp) {
  check_graph<dgpp::MinimaxModel>(minimax_factory(), 29978);
}
DGPP_TEST(minimax_dspark_sampled_graph_tp) {
  check_graph<dgpp::MinimaxModel>(minimax_factory(), 29978, true);
}
DGPP_TEST(minimax_dspark_four_slot_graph_tp) {
  // Capture and replay 32/24/16/8-row verification batches. The published
  // FC shape previously introduced cuBLAS memset nodes at 24 rows.
  check_graph<dgpp::MinimaxModel>(minimax_factory(true, 4), 29978, false, false, nullptr, 4);
}
DGPP_TEST(minimax_dspark_four_slot_sampled_graph_tp) {
  check_graph<dgpp::MinimaxModel>(minimax_factory(true, 4), 29978, true, false, nullptr, 4);
}
DGPP_TEST(minimax_dspark_features_independent_of_row_count) {
  auto model = minimax_factory(true, 4)(nullptr, 0, 1);
  const auto ids = prompt(129, 19);
  const int h = model->config().hidden_size;
  auto features = [&](int rows) {
    DGPP_CUDA_OK(cudaStreamSynchronize(model->stream()));
    std::vector<float> out(rows * h);
    DGPP_CUDA_OK(cudaMemcpy(out.data(), model->dspark_drafter().acc(), out.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
    return out;
  };
  model->session_prefill(0, ids);
  const auto reference = features(ids.size());
  model->session_close(0);
  for (int rows : {1, 8, 16, 24, 32}) {
    model->session_prefill(0, {ids.front()});
    const auto first = features(1);
    require(std::equal(first.begin(), first.end(), reference.begin()),
            "DSpark first-row features differ from full prefill");
    for (size_t at = 1; at < ids.size();) {
      const int n = std::min<size_t>(rows, ids.size() - at);
      const std::vector<int64_t> next{ids.begin() + at, ids.begin() + at + n};
      if (rows <= 8)
        model->session_verify(0, next);
      else
        model->session_prefill_resume(0, next, {});
      const auto got = features(n);
      require(std::equal(got.begin(), got.end(), reference.begin() + at * h),
              "DSpark features depend on row count " + std::to_string(rows));
      if (rows <= 8) model->session_rollback(0, n, n);
      at += n;
    }
    model->session_close(0);
  }
  for (int slots = 2; slots <= 4; ++slots) {
    std::vector<int> reqs;
    std::vector<std::vector<int64_t>> feeds;
    for (int req = 0; req < slots; ++req) {
      model->session_prefill(req, {ids.front()});
      reqs.push_back(req);
      feeds.emplace_back(ids.begin() + 1, ids.begin() + 9);
    }
    model->session_verify_batch(reqs, feeds);
    const auto got = features(slots * 8);
    for (int req : reqs) {
      require(std::equal(got.begin() + req * 8 * h, got.begin() + (req + 1) * 8 * h,
                         reference.begin() + h),
              "DSpark batched features differ from full prefill");
      model->session_close(req);
    }
  }
}
DGPP_TEST(minimax_dspark_preserves_target_and_all_accept_lengths) {
  auto plain = minimax_factory(false)(nullptr, 0, 1);
  auto draft = minimax_factory()(nullptr, 0, 1);
  for (int accepted = 1; accepted <= 8; ++accepted) {
    const auto a = plain->session_prefill(0, prompt(129, accepted));
    const auto b = draft->session_prefill(0, prompt(129, accepted));
    require(a.logits == b.logits, "feature taps change target prefill logits");
    std::vector<int32_t> proposals;
    require(draft->dflash2_draft(0, 7, &proposals) && proposals.size() == 7,
            "native block proposal");
    const auto feeds = prompt(8, accepted + 10);
    require(plain->session_verify(0, feeds).logits == draft->session_verify(0, feeds).logits,
            "draft changes target verify logits");
    plain->session_rollback(0, accepted, 8);
    draft->session_rollback(0, accepted, 8);
    require(plain->session_step(0, 117).logits == draft->session_step(0, 117).logits,
            "draft rollback changes continuation");
    require(draft->dflash2_draft(0, 117, &proposals), "draft after rollback");
    plain->session_close(0);
    draft->session_close(0);
  }
}
DGPP_TEST(minimax_dspark_eager_transcripts_match_plain) {
  auto transcript = [](bool draft) {
    auto m = minimax_factory(draft)(nullptr, 0, 1);
    dgpp::EagerEngineAdapter<dgpp::MinimaxModel> engine(m.get(), 2, dgpp::make_w1_pick(512));
    std::vector<int32_t> out{engine.prefill(0, prompt(63, 1))};
    while (out.size() < 24) {
      engine.reserve(0, m->session_position(0) + 8);
      const auto next = engine.step(0);
      require(!next.empty(), "eager draft makes progress");
      out.insert(out.end(), next.begin(), next.end());
    }
    engine.close(0);
    out.resize(24);
    return out;
  };
  require(transcript(true) == transcript(false), "DSpark changes the greedy transcript");
}

DGPP_TEST(dflash2_flash_next_prefix_snapshot) { check_prefix<dgpp::QwenModel>(flash_factory()); }
DGPP_TEST(dflash2_flash_next_prefix_tier) { check_prefix_tier<dgpp::QwenModel>(flash_factory(), "flash"); }
DGPP_TEST(dflash2_qwen35_prefix_snapshot) { check_prefix<dgpp::Qwen35Model>(qwen35_factory()); }
DGPP_TEST(dflash2_qwen35_prefix_tier) { check_prefix_tier<dgpp::Qwen35Model>(qwen35_factory(), "qwen35"); }
DGPP_TEST(dflash2_flash_next_graph_reducer_tp) { check_graph<dgpp::QwenModel>(flash_factory(), 29978); }
DGPP_TEST(dflash2_qwen35_graph_reducer_tp) { check_graph<dgpp::Qwen35Model>(qwen35_factory(), 29979); }
DGPP_TEST(dflash2_flash_next_sampled_graph_tp) { check_graph<dgpp::QwenModel>(flash_factory(), 29978, true); }
DGPP_TEST(dflash2_qwen35_sampled_graph_tp) { check_graph<dgpp::Qwen35Model>(qwen35_factory(), 29979, true); }

DGPP_TEST(dflash2_flash_next_head_screen) {
  struct Restore {
    dgpp::DraftHeadSettings old = dgpp::draft_head_settings();
    ~Restore() { dgpp::draft_head_configure(old); }
  } restore;
  dgpp::draft_head_configure({1});
  const auto make = flash_factory();
  const auto cfg = qwenfx::tiny_config();
  const auto plan = dgpp::QwenModel::plan_memory(cfg, 128, 1024, 0, 2,
      dgpp::QwenResidency::Resident, 2, false, 16, false, dgpp::LatentFormat::kBf16,
      "qwen_dflash2_draft_fixture");
  const auto item = std::find_if(plan.items.begin(), plan.items.end(),
      [](const auto& i) { return i.name.find("draft head screen") != std::string::npos; });
  require(item != plan.items.end() && item->device > 0, "drafter plan omits the requested head screen");
  check_graph<dgpp::QwenModel>(make, 29978, false, true);
  check_graph<dgpp::QwenModel>(make, 29978, true, true);
}

DGPP_TEST(dflash2_flash_next_replay_matches_snapshots) {
  ReplaySetting restore;
  const auto make = flash_factory();
  const auto cfg = qwenfx::tiny_config();
  const auto plan = [&] {
    return dgpp::QwenModel::plan_memory(cfg, 128, 1024, 0, 2,
        dgpp::QwenResidency::Resident, 2, false, 16, false, dgpp::LatentFormat::kBf16,
        "qwen_dflash2_draft_fixture");
  };
  const auto has_replay = [](const auto& p) {
    return std::any_of(p.items.begin(), p.items.end(),
        [](const auto& i) { return i.name.find("GDN replay rows") != std::string::npos && i.device > 0; });
  };
  std::vector<std::vector<uint8_t>> replay_states, snapshot_states;
  setenv("DGPP_QWEN_GDN_REPLAY", "off", 1);
  require(!has_replay(plan()), "disabled replay remains in the memory plan");
  const auto reference = check_graph<dgpp::QwenModel>(make, 29978, false, false, &snapshot_states);
  setenv("DGPP_QWEN_GDN_REPLAY", "on", 1);
  require(has_replay(plan()), "DFlash2 omits recurrent-state replay from its memory plan");
  const auto replay = check_graph<dgpp::QwenModel>(make, 29978, false, false, &replay_states);
  require(replay == reference, "replay changes speculative transcripts");
  require(snapshot_states.size() >= 2, "too few aligned decode snapshots exercised replay");
  require(replay_states == snapshot_states, "replay changes materialized session state");
}

DGPP_TEST(dflash2_flash_next_replay_all_accept_lengths) {
  ReplaySetting restore;
  const auto make = flash_factory();
  setenv("DGPP_QWEN_GDN_REPLAY", "off", 1);
  auto reference = make(nullptr, 0, 1);
  setenv("DGPP_QWEN_GDN_REPLAY", "on", 1);
  auto replay = make(nullptr, 0, 1);
  dgpp::PrefixArena<dgpp::QwenModel> old_cache(reference.get(), 1), new_cache(replay.get(), 1);
  for (int accepted = 1; accepted <= 8; ++accepted) {
    reference->session_prefill(0, prompt(48, accepted));
    replay->session_prefill(0, prompt(48, accepted));
    // Force every accepted prefix length, including all eight verified
    // rows; random draft fixtures otherwise mostly reject the first draft.
    const auto feeds = prompt(8, accepted + 10);
    const auto want = reference->session_verify(0, feeds);
    const auto got = replay->session_verify(0, feeds);
    require(got.logits == want.logits, "replay changes block verify logits");
    reference->session_rollback(0, accepted, 8);
    replay->session_rollback(0, accepted, 8);
    do {
      const auto a = reference->session_step(0, 117);
      const auto b = replay->session_step(0, 117);
      require(a.logits == b.logits, "replay restores the wrong accepted prefix");
    } while (reference->session_position(0) % reference->session_snapshot_align() != 0);
    const auto pos = reference->session_position(0);
    old_cache.snapshot(0, 0, pos);
    new_cache.snapshot(0, 0, pos);
    reference->session_close(0);
    replay->session_close(0);
    old_cache.attach(0, 0);
    new_cache.attach(0, 0);
    const auto a = reference->session_step(0, 211);
    const auto b = replay->session_step(0, 211);
    require(a.logits == b.logits, "replay changes continuation from a decoded prefix snapshot");
    reference->session_close(0);
    replay->session_close(0);
    old_cache.release(0);
    new_cache.release(0);
  }
}

DGPP_TEST(dflash2_qwen35_plan_uses_local_vocabulary) {
  const auto cfg = qwen35fx::tiny_config();
  const std::string draft = "qwen_dflash2_draft_fixture";
  qwenfx::write_dflash2_fixture(draft);
  for (int world : {1, 2}) {
    const auto plan = dgpp::Qwen35Model::plan_memory(cfg, 128, 1024, 0, world,
        dgpp::LoaderResidency::Resident, 2, false, 16, draft);
    const auto it = std::find_if(plan.items.begin(), plan.items.end(),
        [](const auto& item) { return item.name == "dflash2 head logits"; });
    require(it != plan.items.end(), "plan omits drafter head logits");
    require(it->device == 16 * (cfg.vocab_size / world) * sizeof(float),
            "plan does not shard drafter head logits with the target head");
  }
}

int main() { return dgpp::test::run_all(); }
