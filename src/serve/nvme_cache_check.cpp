#include "serve/nvme_cache_check.hpp"

#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>

#include "serve/cluster_config.hpp"

namespace dgpp::serve {

namespace {
constexpr size_t kAlign = 4096;

std::string gib(double bytes) { return std::format("{:.2f} GiB", bytes / (1024.0 * 1024.0 * 1024.0)); }
}  // namespace

std::string NvmeCachePlan::describe() const {
  return std::format(
      "NVMe cache plan — {} ({}): {} pages of {} bytes, {} pages per snapshot blob; minimum {} for one entry at "
      "the context limit; {} free on the filesystem beside a {} reserve",
      slab, gib(static_cast<double>(capacity_bytes)), pages, page_bytes, blob_pages,
      gib(static_cast<double>(minimum_bytes)), gib(static_cast<double>(free_bytes)),
      gib(static_cast<double>(reserve_bytes)));
}

NvmeCachePlan plan_nvme_cache(const std::string& path, double capacity_gib, int rank, size_t block_bytes,
                              size_t blob_bytes, int64_t block_tokens, int64_t context_limit) {
  NvmeCachePlan plan;
  plan.directory = expand_home(path);
  plan.slab = plan.directory + "/rank" + std::to_string(rank) + ".slab";
  plan.reserve_bytes = kNvmeCacheReserveBytes;
  if (!(capacity_gib > 0.0) || !std::isfinite(capacity_gib)) {
    plan.error = "nvme_cache.capacity_gib must be a positive number of GiB";
    return plan;
  }
  plan.capacity_bytes = static_cast<size_t>(capacity_gib * 1024.0 * 1024.0 * 1024.0);
  plan.page_bytes = std::max<size_t>(kAlign, (block_bytes + kAlign - 1) / kAlign * kAlign);
  plan.blob_pages = std::max<int64_t>(1, static_cast<int64_t>((blob_bytes + plan.page_bytes - 1) / plan.page_bytes));
  plan.pages = static_cast<int64_t>(plan.capacity_bytes / plan.page_bytes);
  const int64_t blocks_at_limit =
      block_tokens > 0 ? (context_limit + block_tokens - 1) / block_tokens : 0;
  plan.minimum_bytes = static_cast<size_t>(plan.blob_pages + blocks_at_limit) * plan.page_bytes;
  // The directory: made now, so the probe below measures the filesystem
  // the slab lands on.
  std::error_code ec;
  std::filesystem::create_directories(plan.directory, ec);
  if (ec) {
    plan.error = "nvme_cache.path " + plan.directory + " cannot be created: " + ec.message();
    return plan;
  }
  if (::access(plan.directory.c_str(), W_OK | X_OK) != 0) {
    plan.error = "nvme_cache.path " + plan.directory + " is not writable: " + std::strerror(errno);
    return plan;
  }
  struct statvfs vfs {};
  if (::statvfs(plan.directory.c_str(), &vfs) != 0) {
    plan.error = "nvme_cache.path " + plan.directory + ": statvfs failed: " + std::strerror(errno);
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
  return plan;
}

}  // namespace dgpp::serve
