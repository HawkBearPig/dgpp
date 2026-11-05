#include "serve/nvme_cache_check.hpp"

#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "serve/cluster_config.hpp"
#include "sched/disk_cache.hpp"
#include "sched/nvme_index.hpp"

namespace dgpp::serve {

namespace {
constexpr size_t kAlign = 4096;

std::string gib(double bytes) { return std::format("{:.2f} GiB", bytes / (1024.0 * 1024.0 * 1024.0)); }
}  // namespace

bool valid_nvme_cache_binding(const std::string& binding) {
  return binding.size() == 16 && std::all_of(binding.begin(), binding.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

std::string nvme_cache_binding(const std::string& version, const std::string& digest,
                               const std::string& checkpoint, size_t block_bytes, size_t snapshot_bytes,
                               double capacity_gib, int64_t block_tokens, int world,
                               const std::string& reuse_binding) {
  if (!reuse_binding.empty()) {
    if (!valid_nvme_cache_binding(reuse_binding))
      throw std::invalid_argument("nvme_cache.reuse_binding must be 16 lowercase hexadecimal digits");
    return reuse_binding;
  }
  return config_digest(version + "|" + digest + "|" + checkpoint + "|" +
                       std::to_string(block_bytes) + "|" + std::to_string(snapshot_bytes) + "|" +
                       std::format("{:.17g}", capacity_gib) + "|" + std::to_string(block_tokens) + "|" +
                       std::to_string(world));
}

std::string check_nvme_cache_reuse(const NvmeCachePlan& plan, const std::string& binding,
                                  int rank, int64_t block_tokens) {
  namespace fs = std::filesystem;
  const auto refuse = [&](const std::string& why) {
    return "nvme_cache.reuse_binding " + binding + ": " + why + "; existing cache left untouched";
  };
  if (!valid_nvme_cache_binding(binding)) return refuse("invalid binding");
  const auto sized = [](const fs::path& path, uintmax_t bytes) {
    std::error_code ec;
    return fs::is_regular_file(path, ec) && fs::file_size(path, ec) == bytes && !ec;
  };
  const fs::path directory(plan.directory);
  if (!sized(plan.slab, static_cast<uintmax_t>(plan.pages) * plan.page_bytes))
    return refuse("slab missing or capacity/page geometry changed");
  if (!sized(directory / ("rank" + std::to_string(rank) + ".crc"),
             static_cast<uintmax_t>(plan.pages) * sizeof(uint32_t)))
    return refuse("checksum table missing or wrong size");
  std::ifstream input(directory / ("rank" + std::to_string(rank) + ".index"), std::ios::binary);
  if (!input) return refuse("index missing or unreadable");
  const std::vector<uint8_t> file{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  std::vector<uint8_t> shared;
  std::vector<uint64_t> digests;
  if (input.bad() || !sched::parse_nvme_index(file, binding, &shared, &digests))
    return refuse("index format or stored binding does not match");
  sched::DiskCache::Header header;
  std::vector<sched::DiskCache::Record> records;
  if (!sched::DiskCache::parse(shared, &header, &records) || records.size() != digests.size())
    return refuse("index payload is invalid");
  if (header.pages != static_cast<uint64_t>(plan.pages) || header.page_bytes != plan.page_bytes ||
      header.block_tokens != static_cast<uint64_t>(block_tokens))
    return refuse("cache geometry does not match");
  return {};
}

std::string NvmeCachePlan::describe() const {
  return std::format(
      "NVMe cache plan — {} ({}): {} pages of {} bytes, {} pages per snapshot blob; minimum {} for one entry at "
      "the context limit; {} free on the filesystem beside a {} reserve",
      slab, gib(static_cast<double>(capacity_bytes)), pages, page_bytes, blob_pages,
      gib(static_cast<double>(minimum_bytes)), gib(static_cast<double>(free_bytes)),
      gib(static_cast<double>(reserve_bytes)));
}

NvmeCachePlan plan_nvme_cache(const std::string& path, double capacity_gib, int rank, size_t block_bytes,
                              size_t blob_bytes, int64_t block_tokens, int64_t context_limit,
                              bool create_directory, size_t reserve_bytes, const std::string& binding,
                              size_t chunk_bytes, const std::string& required_mount, bool allow_buffered) {
  NvmeCachePlan plan;
  plan.directory = expand_home(path) + (binding.empty() ? "" : "/" + binding);
  plan.slab = plan.directory + "/rank" + std::to_string(rank) + ".slab";
  plan.reserve_bytes = reserve_bytes;
  if (!(capacity_gib > 0.0) || !std::isfinite(capacity_gib)) {
    plan.error = "nvme_cache.capacity_gib must be a positive number of GiB";
    return plan;
  }
  plan.capacity_bytes = static_cast<size_t>(capacity_gib * 1024.0 * 1024.0 * 1024.0);
  plan.page_bytes = std::max<size_t>(kAlign, (block_bytes + kAlign - 1) / kAlign * kAlign);
  plan.blob_pages = std::max<int64_t>(1, static_cast<int64_t>((blob_bytes + plan.page_bytes - 1) / plan.page_bytes));
  plan.pages = static_cast<int64_t>(plan.capacity_bytes / plan.page_bytes);
  if (chunk_bytes > 0 && (chunk_bytes % kAlign != 0 || chunk_bytes < plan.page_bytes)) {
    plan.error = std::format(
        "nvme_cache.chunk_mib ({} bytes) must be a 4 KiB multiple that holds one cache block's page ({} bytes "
        "for this deployment); raise it",
        chunk_bytes, plan.page_bytes);
    return plan;
  }
  const int64_t blocks_at_limit =
      block_tokens > 0 ? (context_limit + block_tokens - 1) / block_tokens : 0;
  // The entry's token ids ride the slab too (8 bytes a token, behind the blob).
  const int64_t ids_pages_at_limit =
      (static_cast<int64_t>(context_limit) * 8 + static_cast<int64_t>(plan.page_bytes) - 1) /
      static_cast<int64_t>(plan.page_bytes);
  plan.minimum_bytes =
      static_cast<size_t>(plan.blob_pages + ids_pages_at_limit + blocks_at_limit) * plan.page_bytes;
  // Inspect before mkdir/open/preallocation, including in normal serving.
  plan.storage = inspect_cache_directory(plan.directory);
  plan.error = check_cache_storage(plan.storage, required_mount, allow_buffered);
  if (!plan.error.empty()) return plan;
  plan.directory = plan.storage.path;
  plan.slab = plan.directory + "/rank" + std::to_string(rank) + ".slab";
  const std::string& probe = plan.storage.probe;
  std::error_code ec;
  if (::access(probe.c_str(), W_OK | X_OK) != 0) {
    plan.error = "nvme_cache.path " + probe + " is not writable: " + std::strerror(errno);
    return plan;
  }
  struct statvfs vfs {};
  if (::statvfs(probe.c_str(), &vfs) != 0) {
    plan.error = "nvme_cache.path " + probe + ": statvfs failed: " + std::strerror(errno);
    return plan;
  }
  plan.free_bytes = static_cast<size_t>(vfs.f_bavail) * static_cast<size_t>(vfs.f_frsize);
  // An earlier slab of this rank is recreated, so its bytes are free too.
  struct stat st {};
  if (::stat(plan.slab.c_str(), &st) == 0 && S_ISREG(st.st_mode))
    plan.free_bytes += static_cast<size_t>(st.st_blocks) * 512;
  if (plan.capacity_bytes < plan.minimum_bytes) {
    plan.error = std::format(
        "nvme_cache.capacity_gib {} ({}) is below the minimum of {} for this deployment — one entry at the "
        "{}-token context limit ({} blocks of {} bytes) plus its {}-page snapshot blob; raise the capacity or "
        "lower engine.kv_capacity",
        capacity_gib, gib(static_cast<double>(plan.capacity_bytes)), gib(static_cast<double>(plan.minimum_bytes)),
        context_limit, blocks_at_limit, plan.page_bytes, plan.blob_pages);
    return plan;
  }
  if (plan.capacity_bytes + plan.reserve_bytes > plan.free_bytes) {
    plan.error = std::format(
        "nvme_cache.capacity_gib {} ({}) plus the {} reserve exceeds the {} free on the filesystem holding {}; "
        "free space on that filesystem or lower the capacity",
        capacity_gib, gib(static_cast<double>(plan.capacity_bytes)), gib(static_cast<double>(plan.reserve_bytes)),
        gib(static_cast<double>(plan.free_bytes)), plan.directory);
    return plan;
  }
  if (create_directory) {
    std::filesystem::create_directories(plan.directory, ec);
    if (ec) {
      plan.error = "nvme_cache.path " + plan.directory + " cannot be created: " + ec.message();
      return plan;
    }
    // Recheck the directory actually created; the opener checks its descriptor
    // too, before truncation, in case mounts change while the model loads.
    plan.storage = inspect_cache_directory(plan.directory);
    plan.error = check_cache_storage(plan.storage, required_mount, allow_buffered);
  }
  return plan;
}

}  // namespace dgpp::serve
