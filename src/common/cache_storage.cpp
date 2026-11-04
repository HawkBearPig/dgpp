#include "common/cache_storage.hpp"

#include <fcntl.h>
#include <linux/magic.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <sstream>

namespace dgpp {
namespace {
namespace fs = std::filesystem;

bool number(const std::string& text, uint64_t* value) {
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), *value);
  return ec == std::errc() && end == text.data() + text.size();
}

bool unescape(const std::string& input, std::string* output) {
  output->clear();
  for (size_t i = 0; i < input.size(); ++i) {
    if (input[i] != '\\') {
      output->push_back(input[i]);
      continue;
    }
    if (i + 3 >= input.size()) return false;
    const auto code = input.substr(i + 1, 3);
    if (code == "040") output->push_back(' ');
    else if (code == "011") output->push_back('\t');
    else if (code == "012") output->push_back('\n');
    else if (code == "134") output->push_back('\\');
    else return false;
    i += 3;
  }
  return true;
}

void classify(CacheStorage* storage) {
  const auto& type = storage->filesystem;
  storage->ram_backed = type == "tmpfs" || type == "ramfs" || type == "hugetlbfs";
  storage->container_layer = type == "overlay" || type == "aufs" || type == "fuse.overlayfs" ||
                             type == "fuse.fuse-overlayfs";
}
}  // namespace

CacheStorage read_cache_mount(std::istream& input, uint64_t mount_id) {
  CacheStorage out;
  out.mount_id = mount_id;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream row(line);
    std::string id;
    uint64_t parsed = 0;
    if (!(row >> id) || !number(id, &parsed) || parsed != mount_id) continue;
    std::string parent, device, root, point, options, field, type, source, super_options;
    if (!(row >> parent >> device >> root >> point >> options)) break;
    while (row >> field && field != "-") {}
    if (field != "-" || !(row >> type >> source >> super_options) ||
        !unescape(point, &out.mount_point) || out.mount_point.empty() || out.mount_point.front() != '/') break;
    out.filesystem = type;
    classify(&out);
    return out;
  }
  out.mount_point.clear();
  out.mount_error = "cannot identify mount " + std::to_string(mount_id) + " in /proc/self/mountinfo";
  return out;
}

CacheStorage inspect_cache_storage_fd(int fd) {
  CacheStorage out;
  struct statfs info {};
  if (::fstatfs(fd, &info) != 0) {
    out.error = "cannot inspect cache filesystem: " + std::string(std::strerror(errno));
    return out;
  }
  switch (info.f_type) {
    case TMPFS_MAGIC: out.filesystem = "tmpfs"; break;
    case RAMFS_MAGIC: out.filesystem = "ramfs"; break;
    case HUGETLBFS_MAGIC: out.filesystem = "hugetlbfs"; break;
    case OVERLAYFS_SUPER_MAGIC: out.filesystem = "overlay"; break;
    default: out.filesystem = std::format("type 0x{:x}", static_cast<uint64_t>(info.f_type)); break;
  }
  classify(&out);
  // fdinfo works for O_PATH descriptors too and does not require statx support.
  std::ifstream fdinfo("/proc/self/fdinfo/" + std::to_string(fd));
  std::string line;
  while (std::getline(fdinfo, line)) {
    std::istringstream row(line);
    std::string key, value;
    if (row >> key >> value && key == "mnt_id:" && number(value, &out.mount_id)) break;
  }
  if (out.mount_id) {
    std::ifstream mounts("/proc/self/mountinfo");
    const auto mount = read_cache_mount(mounts, out.mount_id);
    out.mount_point = mount.mount_point;
    out.mount_error = mount.mount_error;
    if (out.mount_error.empty()) {
      out.filesystem = mount.filesystem;
      // Keep the descriptor's RAM/overlay classification even if proc changes.
      out.ram_backed = out.ram_backed || mount.ram_backed;
      out.container_layer = out.container_layer || mount.container_layer;
    }
  } else {
    out.mount_error = "cannot read mount identity from /proc/self/fdinfo";
  }
  return out;
}

CacheStorage inspect_cache_directory(const std::string& path) {
  CacheStorage out;
  if (path.empty() || path.find('\0') != std::string::npos) {
    out.error = "cache directory must be a nonempty path without NUL bytes";
    return out;
  }
  std::error_code ec;
  fs::path ancestor = fs::absolute(path, ec), suffix;
  if (ec) {
    out.error = "cannot resolve cache directory: " + ec.message();
    return out;
  }
  for (;;) {
    const auto status = fs::symlink_status(ancestor, ec);
    if (ec && ec != std::errc::no_such_file_or_directory) {
      out.error = "cannot inspect cache directory " + ancestor.string() + ": " + ec.message();
      return out;
    }
    if (fs::exists(status)) break;
    if (ancestor == ancestor.parent_path() || ancestor.empty()) {
      out.error = "cache directory has no existing ancestor";
      return out;
    }
    suffix = ancestor.filename() / suffix;
    ancestor = ancestor.parent_path();
  }
  // canonical refuses dangling symlinks; a file in a directory component is
  // also an error, never a reason to walk past it to a writable ancestor.
  ancestor = fs::canonical(ancestor, ec);
  if (ec || !fs::is_directory(ancestor, ec)) {
    out.error = "cache ancestor is not an accessible directory: " + path;
    return out;
  }
  out.path = (ancestor / suffix).lexically_normal().string();
  if (out.path.size() > 1 && out.path.back() == '/') out.path.pop_back();
  // Collapsing a missing component followed by '..' can expose an existing
  // mount/symlink at the normalized destination. Inspect that destination.
  for (const auto& component : suffix)
    if (component == "..") return inspect_cache_directory(out.path);
  out.probe = ancestor.string();
  const int fd = ::open(out.probe.c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC);
  if (fd < 0) {
    out.error = "cannot open cache directory " + out.probe + ": " + std::strerror(errno);
    return out;
  }
  auto storage = inspect_cache_storage_fd(fd);
  ::close(fd);
  storage.path = out.path;
  storage.probe = out.probe;
  return storage;
}

std::string check_cache_storage(const CacheStorage& storage, const std::string& required_mount,
                                bool allow_buffered) {
  if (!storage.error.empty()) return storage.error;
  if (!required_mount.empty()) {
    if (!fs::path(required_mount).is_absolute() || required_mount.find('\0') != std::string::npos)
      return "nvme_cache.required_mount must be an absolute mountpoint path";
    if (!storage.mount_error.empty())
      return "nvme_cache.required_mount cannot be verified: " + storage.mount_error;
    std::error_code ec;
    const auto expected = fs::canonical(required_mount, ec);
    if (ec || expected.string() != storage.mount_point)
      return "nvme_cache.required_mount " + required_mount + " is not the cache's containing mount (" +
             storage.mount_point + "); mount the intended storage before starting";
  }
  if (storage.ram_backed && !allow_buffered)
    return "NVMe cache refuses RAM-backed storage (" + storage.filesystem +
           "); choose disk-backed storage, or use nvme_cache.allow_buffered for tests only";
  return {};
}

std::string CacheStorage::describe() const {
  return std::format("cache storage — resolved {}; probe {}; mount {}; filesystem {}", path, probe,
                     mount_point.empty() ? "unknown" : mount_point, filesystem.empty() ? "unknown" : filesystem);
}

std::string CacheStorage::warning(bool allow_buffered) const {
  std::string out;
  if (container_layer)
    out = "NVMe cache is on " + filesystem +
          ", which may be a container writable layer; cache data may be lost when the container is replaced. "
          "Use a writable storage mount and nvme_cache.required_mount to assert its placement.";
  if (ram_backed && allow_buffered)
    out += " NVMe cache is RAM-backed; nvme_cache.allow_buffered permits this for tests/experiments only.";
  if (!mount_error.empty()) out += " Cache mount metadata unavailable: " + mount_error + ".";
  return out;
}

}  // namespace dgpp
