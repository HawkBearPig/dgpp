#pragma once

#include <cstdint>
#include <istream>
#include <string>

namespace dgpp {

// Placement in this process's mount namespace, not a claim about the physical
// device behind a filesystem. Reading this information never creates files.
struct CacheStorage {
  std::string path;   // absolute, existing symlinks resolved
  std::string probe;  // nearest existing directory for a path not yet created
  std::string mount_point;
  std::string filesystem;
  uint64_t mount_id = 0;
  bool ram_backed = false;
  bool container_layer = false;
  std::string mount_error;  // filesystem known, but mount metadata unavailable
  std::string error;

  std::string describe() const;
  std::string warning(bool allow_buffered) const;
};

// Matches the descriptor's mount ID, so bind mounts and stacked mounts are
// distinguished even when st_dev is identical. Exposed for parser tests.
CacheStorage read_cache_mount(std::istream& input, uint64_t mount_id);
CacheStorage inspect_cache_directory(const std::string& path);
CacheStorage inspect_cache_storage_fd(int fd);

// A nonempty required_mount names the exact containing mountpoint (aliases
// are resolved). Nested mounts are different storage and do not satisfy it.
// allow_buffered is the existing explicit test/experiment escape hatch.
std::string check_cache_storage(const CacheStorage& storage, const std::string& required_mount,
                                bool allow_buffered);

}  // namespace dgpp
