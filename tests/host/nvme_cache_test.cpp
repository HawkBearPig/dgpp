#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "sched/disk_cache.hpp"
#include "sched/nvme_index.hpp"
#include "serve/cluster_config.hpp"
#include "serve/nvme_cache_check.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::sched::DiskCache;

void require(bool value, const std::string& why) {
  if (!value) throw std::runtime_error(why);
}

std::vector<uint8_t> read(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  require(static_cast<bool>(in), "read " + path.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write(const fs::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  require(static_cast<bool>(out), "write " + path.string());
}

struct Fixture {
  fs::path root;
  dgpp::serve::NvmeCachePlan plan;
  const double gib = 65536.0 / (1024 * 1024 * 1024);
  const std::string binding = dgpp::serve::nvme_cache_binding("old-build", "old-settings", "/model",
                                                            4096, 4096, gib, 4, 1);
  std::vector<int64_t> prompt{10, 11, 12, 13, 14, 15, 16, 17, 18};
  DiskCache::Config config;
  fs::path index_path, crc_path;

  Fixture() {
    char path[] = "/tmp/dgpp-nvme-reuse-XXXXXX";
    require(::mkdtemp(path) != nullptr, "temporary directory");
    root = path;
    plan = dgpp::serve::plan_nvme_cache(root.string(), gib, 0, 4096, 4096, 4, 8, true, 0, binding, 0, "", true);
    require(plan.error.empty(), plan.error);
    index_path = fs::path(plan.directory) / "rank0.index";
    crc_path = fs::path(plan.directory) / "rank0.crc";
    config.pages = 16;
    config.page_bytes = 4096;
    config.blob_pages = 1;
    config.block_tokens = 4;
    DiskCache cache(config);
    const auto spill = cache.begin_spill(prompt.data(), 8, 18, {}, 0, {100, 101}, 0, 1, 1, 0);
    require(spill.ok, "fixture spill");
    cache.commit_spill(spill.entry);
    cache.set_content(spill.entry, 77);
    std::vector<uint8_t> slab(65536, 0x5a);
    std::memcpy(slab.data() + spill.ids_page * 4096, prompt.data(), 8 * sizeof(int64_t));
    write(plan.slab, slab);
    write(crc_path, std::vector<uint8_t>(16 * sizeof(uint32_t), 0x2a));
    write(index_path, dgpp::sched::write_nvme_index(binding, cache.serialize(), {77}));
  }
  ~Fixture() { std::error_code ec; fs::remove_all(root, ec); }

  std::string check() const { return dgpp::serve::check_nvme_cache_reuse(plan, binding, 0, 4); }

  void refused_without_writes() const {
    const auto slab = read(plan.slab), crc = read(crc_path), index = read(index_path);
    require(!check().empty(), "reuse refused");
    require(read(plan.slab) == slab && read(crc_path) == crc && read(index_path) == index,
            "refusal preserves all existing files");
  }
};
}  // namespace

DGPP_TEST(nvme_cache_reuse_adopts_previous_build_without_rewriting_cache) {
  Fixture f;
  const std::string automatic = dgpp::serve::nvme_cache_binding("new-build", "new-settings", "/model",
                                                              4096, 4096, f.gib, 4, 1);
  require(automatic != f.binding, "automatic binding changes as before");
  require(dgpp::serve::nvme_cache_binding("new-build", "new-settings", "/model", 4096, 4096,
                                         f.gib, 4, 1, f.binding) == f.binding,
          "explicit selection retains the old binding across build and validation input changes");
  const auto before = read(f.index_path);
  require(f.check().empty(), f.check());
  std::vector<uint8_t> shared;
  std::vector<uint64_t> digests;
  require(dgpp::sched::parse_nvme_index(before, f.binding, &shared, &digests), "old envelope readable");
  const auto slab = read(f.plan.slab);
  DiskCache cache(f.config);
  const auto adopted = cache.restore(shared, [&](int64_t page, int64_t count, void* dst) {
    if (page < 0 || count < 0 || page + count > 16) return false;
    std::memcpy(dst, slab.data() + page * 4096, static_cast<size_t>(count) * 4096);
    return true;
  }, &digests);
  require(adopted.error.empty() && adopted.entries == 1 && adopted.dropped == 0, "old entry adopted");
  const uint64_t hash = dgpp::sched::PrefixCache::hash_prefix(f.prompt.data(), 8);
  require(cache.lookup(f.prompt, {8}, {hash}) >= 0, "same numeric prefix finds the adopted entry");
  require(read(f.index_path) == before && read(f.plan.slab) == slab, "reuse leaves the old files intact");
}

DGPP_TEST(nvme_cache_automatic_binding_keeps_legacy_formula) {
  // Literal input to the previous inline binding expression, including its
  // decimal geometry spelling; introducing the option must not alter it.
  require(dgpp::serve::nvme_cache_binding("v", "digest", "/model", 4096, 8192, 64.5, 256, 4) ==
              dgpp::serve::config_digest("v|digest|/model|4096|8192|64.5|256|4"),
          "automatic binding is unchanged");
}

DGPP_TEST(nvme_cache_reuse_refuses_incompatible_files_without_mutation) {
  Fixture f;
  const auto good = read(f.index_path);
  for (const size_t offset : {size_t{7}, size_t{12}, size_t{36}, size_t{44}, size_t{52}, size_t{68}}) {
    auto broken = good;
    broken.at(offset) ^= 0x40;  // envelope format/binding; payload format/pages/page bytes/block tokens
    write(f.index_path, broken);
    f.refused_without_writes();
  }
  write(f.index_path, {good.begin(), good.end() - 1});
  f.refused_without_writes();
  write(f.index_path, good);
  fs::resize_file(f.plan.slab, 65536 - 4096);
  f.refused_without_writes();
  fs::resize_file(f.plan.slab, 65536);
  fs::resize_file(f.crc_path, 4);
  f.refused_without_writes();
  fs::remove(f.index_path);
  require(!f.check().empty() && !fs::exists(f.index_path), "missing index is not recreated");
  auto missing = f.plan;
  missing.directory = (f.root / "missing").string();
  missing.slab = missing.directory + "/rank0.slab";
  require(!dgpp::serve::check_nvme_cache_reuse(missing, f.binding, 0, 4).empty() &&
              !fs::exists(missing.directory), "missing binding is not created");
}

DGPP_TEST(nvme_cache_index_matches_legacy_bytes_and_rejects_truncation) {
  const std::vector<uint8_t> legacy = {
      'D', 'G', 'P', 'P', 'N', 'V', 'M', '2', 16, 0, 0, 0,
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
      3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 0, 0, 0, 8, 7, 6, 5, 4, 3, 2, 1};
  const std::string binding = "0123456789abcdef";
  require(dgpp::sched::write_nvme_index(binding, {1, 2, 3}, {0x0102030405060708ull}) == legacy,
          "writer retains the existing file format");
  std::vector<uint8_t> shared;
  std::vector<uint64_t> digests;
  require(dgpp::sched::parse_nvme_index(legacy, binding, &shared, &digests) &&
              shared == std::vector<uint8_t>({1, 2, 3}) && digests == std::vector<uint64_t>({0x0102030405060708ull}),
          "legacy bytes readable");
  for (size_t length = 0; length < legacy.size(); ++length)
    require(!dgpp::sched::parse_nvme_index({legacy.begin(), legacy.begin() + static_cast<std::ptrdiff_t>(length)},
                                          binding, &shared, &digests), "truncated index refused");
}

DGPP_TEST(nvme_cache_reuse_config_requires_explicit_valid_selection) {
  const auto parse = [](const std::string& fields) {
    return dgpp::serve::parse_cluster_config(
        "{\"model\":\"m\",\"world_size\":1,\"nvme_cache\":{" + fields + "}}", "test");
  };
  require(parse("").nvme_cache.reuse_binding.empty(), "omission keeps automatic behavior");
  require(parse(R"("enabled":true,"capacity_gib":1,"reuse_binding":"0123456789abcdef")")
              .nvme_cache.reuse_binding == "0123456789abcdef", "binding parsed");
  for (const std::string fields : {
           R"("enabled":true,"capacity_gib":1,"reuse_binding":true)",
           R"("enabled":true,"capacity_gib":1,"reuse_binding":"../cache")",
           R"("enabled":true,"capacity_gib":1,"reuse_binding":"0123456789abcdeF")",
           R"("enabled":true,"capacity_gib":1,"persist":false,"reuse_binding":"0123456789abcdef")",
           R"("reuse_binding":"0123456789abcdef")"}) {
    bool refused = false;
    try { (void)parse(fields); }
    catch (const std::runtime_error& e) { refused = std::string(e.what()).find("reuse_binding") != std::string::npos; }
    require(refused, "invalid selection refused: " + fields);
  }
}

int main() { return dgpp::test::run_all(); }
