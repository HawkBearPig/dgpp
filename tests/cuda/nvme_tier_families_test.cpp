// The NVMe cold tier on the session-core families (issue #26, 2026-10-28):
// MiMo-V2.6-Flash, Flint and Qwen3.5 on their synthetic tiny checkpoints.
// For each, a snapshot at one full block plus a partial is spilled to a
// slab file and restored into another arena slot and fresh pool blocks;
// the blob and every plane of every block read back bitwise, and a
// request attached to the restored entry prefills its suffix and takes its
// first step bitwise the cold run. The geometry check at enable time
// guards a plane the family forgot to list; this guards a side table
// outside the planes, which only a restore's arithmetic would show.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "engine/nvme_tier.hpp"
#include "engine/prefix_arena.hpp"
#include "dory_fixture.hpp"
#include "flint_fixture.hpp"
#include "mimo_fixture.hpp"
#include "models/flint/forward.hpp"
#include "models/dory/forward.hpp"
#include "models/mimo/forward.hpp"
#include "models/qwen/model35.hpp"
#include "qwen35_fixture.hpp"

namespace fs = std::filesystem;

namespace {

void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

bool bitwise(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<int64_t> tokens(int vocab, int n, uint64_t seed) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = seed;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(5 + s % static_cast<uint64_t>(vocab - 5));
  }
  return t;
}

template <class Model>
void check_tier(Model& m, int vocab, const std::string& family, const std::string& slab_dir) {
  const int64_t bt = m.kv_block_tokens();
  require(bt > 0 && bt <= 100, family + ": a paged pool with blocks of at most 100 tokens");
  const int64_t cut = bt + 4;  // one full block and a four-token partial
  const std::vector<int64_t> A = tokens(vocab, static_cast<int>(bt + 12), 0x9E3779B97F4A7C15ull);
  const std::vector<int64_t> bounds{cut};
  dgpp::PrefixArena<Model> arena(&m, /*slots=*/2);
  typename Model::SnapshotRequest snap = arena.request(0, cut);
  const typename Model::Outputs cold = m.session_prefill(0, A, bounds, &snap);
  arena.commit(0, snap);
  require(arena.filled(0) && arena.meta(0).position == cut, family + ": the snapshot is in slot 0");
  require(arena.meta(0).full_blocks.size() == 1 && arena.meta(0).partial_block >= 0,
          family + ": one full block and a partial");
  const typename Model::Outputs cold_step = m.session_step(0, A.back());
  m.session_close(0);
  const int64_t blocks_before = m.kv_blocks_in_use();

  fs::remove_all(slab_dir);
  fs::create_directories(slab_dir);
  dgpp::sched::SchedulerEngine::DiskEnable en;
  en.slab_path = slab_dir + "/rank0.slab";
  en.capacity_bytes = size_t{256} << 20;
  en.chunk_bytes = size_t{4} << 20;
  en.rank = 0;
  en.allow_buffered = true;
  dgpp::NvmeTier<Model> tier(&m, &arena, en);
  require(tier.block_bytes() == dgpp::cache_block_bytes(m.cache_planes()), family + ": the record is the planes' bytes");
  const auto wait = [&](uint64_t op) {
    for (int i = 0; i < 4000; ++i) {
      tier.pump();
      for (const auto& c : tier.poll())
        if (c.op == op) return c.ok;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    throw std::runtime_error(family + ": op " + std::to_string(op) + " never completed");
  };
  const dgpp::sched::SchedulerEngine::DiskBlocks ids = tier.entry_blocks(0);
  require(ids.identities.size() == 1 && ids.partial_identity != 0, family + ": the entry's identities");
  dgpp::sched::SchedulerEngine::DiskSpill spill;
  spill.op = 1;
  spill.slot = 0;
  spill.blob_page = 0;
  spill.pages = {tier.blob_pages()};
  spill.partial_page = tier.blob_pages() + 1;
  spill.ids.assign(A.begin(), A.begin() + cut);
  spill.ids_page = tier.blob_pages() + 2;
  spill.ids_pages = 1;
  tier.spill_begin(spill);
  require(wait(1), family + ": the spill completed");
  dgpp::sched::SchedulerEngine::DiskRestore rs;
  rs.op = 2;
  rs.slot = 1;
  rs.position = cut;
  rs.mtp_position = arena.meta(0).mtp_position;
  rs.blob_page = 0;
  rs.pages = {tier.blob_pages()};
  rs.partial_page = tier.blob_pages() + 1;
  const dgpp::sched::SchedulerEngine::DiskBlocks fresh = tier.restore_begin(rs);
  require(fresh.identities.size() == 1 && fresh.identities[0] != ids.identities[0] &&
              fresh.partial_identity != ids.partial_identity,
          family + ": the restore took fresh blocks");
  require(m.kv_blocks_in_use() == blocks_before + 2, family + ": two blocks acquired for the restore");
  require(wait(2), family + ": the restore completed");
  require(arena.filled(1) && arena.meta(1).position == cut, family + ": slot 1 adopted the restored entry");
  {
    std::vector<uint8_t> a(arena.bytes()), b(arena.bytes());
    DGPP_CUDA_OK(cudaMemcpy(a.data(), arena.slot_data(0), a.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(b.data(), arena.slot_data(1), b.size(), cudaMemcpyDeviceToHost));
    require(a == b, family + ": the restored blob differs from the spilled one");
    const auto same_block = [&](int32_t x, int32_t y) {
      for (const dgpp::CachePlane& plane : m.cache_planes()) {
        std::vector<uint8_t> p(plane.block_bytes), q(plane.block_bytes);
        DGPP_CUDA_OK(cudaMemcpy(p.data(), plane.base + static_cast<size_t>(x) * plane.block_bytes, plane.block_bytes,
                                cudaMemcpyDeviceToHost));
        DGPP_CUDA_OK(cudaMemcpy(q.data(), plane.base + static_cast<size_t>(y) * plane.block_bytes, plane.block_bytes,
                                cudaMemcpyDeviceToHost));
        if (p != q) return false;
      }
      return true;
    };
    require(same_block(arena.meta(0).full_blocks[0], arena.meta(1).full_blocks[0]),
            family + ": a restored full block plane differs from the original");
    require(same_block(arena.meta(0).partial_block, arena.meta(1).partial_block),
            family + ": a restored partial block plane differs from the original");
  }
  // Attached from the restored slot, the suffix prefill and the first step
  // are bitwise the cold run's: no state outside the planes was lost.
  arena.attach(1, 1);
  const typename Model::Outputs hot =
      m.session_prefill_resume(1, std::vector<int64_t>(A.begin() + cut, A.end()), bounds);
  require(bitwise(hot.logits, cold.logits), family + ": the prefill after a restored attach differs");
  require(bitwise(m.session_step(1, A.back()).logits, cold_step.logits),
          family + ": the step after a restored attach differs");
  m.session_close(1);
  arena.release(1);
  require(m.kv_blocks_in_use() == blocks_before, family + ": the release returned the restored blocks");
  arena.release(0);
  require(m.kv_blocks_in_use() == 0, family + ": every block released");
  std::printf("[ OK ] %s: NVMe tier spill/restore bitwise, attach and step bitwise\n", family.c_str());
}

}  // namespace

int main() {
  try {
    const std::string base = (fs::temp_directory_path() / "dgpp_nvme_tier_families").string();
    fs::remove_all(base);
    fs::create_directories(base);
    {
      const std::string dir = base + "/dory";
      const dgpp::DoryConfig cfg = doryfx::tiny_config();
      doryfx::write_fixture(cfg, dir);
      // Exercise the serving override too: each recurrent attention layer
      // must expose all three loop planes, in addition to the fixed layers.
      dgpp::DoryModel::set_recurrent_loops(3);
      for (const auto policy : {dgpp::DoryKvPolicy::PerLoop, dgpp::DoryKvPolicy::LastLoop}) {
        dgpp::DoryModel m(cfg, dir, 128, 512, dgpp::DoryResidency::Resident, 2, 8, false, policy);
        require(dgpp::cache_block_bytes(m.cache_planes()) == dgpp::DoryModel::kv_block_bytes_static(cfg, policy),
                "Dory: planned block bytes match every allocated K/V plane");
        if (policy == dgpp::DoryKvPolicy::PerLoop)
          check_tier(m, cfg.vocab_size, "Dory", base + "/dory-slab");
      }
      dgpp::DoryModel::clear_recurrent_loops();
    }
    {
      const std::string dir = base + "/mimo";
      const dgpp::MimoTextConfig cfg = mimofx::tiny_config();
      mimofx::write_fixture(cfg, dir);
      dgpp::MimoModel m(cfg, dir, /*max_tokens=*/128, /*max_cache_tokens=*/512, dgpp::MimoResidency::Resident,
                        nullptr, 0, 1, /*max_requests=*/2, /*mtp=*/false);
      check_tier(m, cfg.vocab_size, "MiMo", base + "/mimo-slab");
    }
    {
      const std::string dir = base + "/flint";
      const dgpp::FlintTextConfig cfg = flintfx::tiny_config();
      flintfx::write_fixture(cfg, dir);
      dgpp::FlintModel m(cfg, dir, /*max_tokens=*/128, /*max_cache_tokens=*/512, dgpp::FlintResidency::Resident,
                         nullptr, 0, 1, /*max_requests=*/2, /*mtp=*/false);
      check_tier(m, cfg.vocab_size, "Flint", base + "/flint-slab");
    }
    {
      const std::string dir = base + "/qwen35";
      qwen35fx::write_fixture(dir);
      const dgpp::Qwen35TextConfig cfg = dgpp::Qwen35TextConfig::from_json_file(dir + "/config.json");
      dgpp::Qwen35Model m(cfg, dir, /*max_tokens=*/128, /*max_cache_tokens=*/512, dgpp::LoaderResidency::Resident,
                          nullptr, 0, 1, /*max_requests=*/2, /*decode_rows=*/8, /*mtp=*/false);
      check_tier(m, cfg.vocab_size, "Qwen3.5", base + "/qwen35-slab");
    }
    fs::remove_all(base);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[FAIL] %s\n", e.what());
    return 1;
  }
}
