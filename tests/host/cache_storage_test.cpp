#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#include "common/cache_storage.hpp"
#include "common/test.hpp"
#include "serve/nvme_cache_check.hpp"

namespace {
namespace fs = std::filesystem;
void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

struct Directory {
  fs::path path;
  explicit Directory(const std::string& parent = "/tmp") {
    std::string pattern = parent + "/dgpp-storage-XXXXXX";
    require(::mkdtemp(pattern.data()) != nullptr, "mkdtemp " + parent);
    path = pattern;
  }
  ~Directory() { std::error_code ec; fs::remove_all(path, ec); }
};

auto plan(const fs::path& path, const std::string& mount = "", bool create = false, bool allow = false) {
  return dgpp::serve::plan_nvme_cache(path.string(), 65536.0 / (1ull << 30), 0, 4096, 4096, 4, 8,
                                      create, 0, "", 4096, mount, allow);
}

std::string contents(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  require(static_cast<bool>(in), "read " + path.string());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void seed(const fs::path& directory) {
  fs::create_directories(directory);
  for (const char* name : {"rank0.slab", "rank0.index", "rank0.crc"})
    std::ofstream(directory / name) << "preserve " << name;
}

void preserved(const fs::path& directory) {
  for (const char* name : {"rank0.slab", "rank0.index", "rank0.crc"})
    require(contents(directory / name) == "preserve " + std::string(name), "refusal modified " + std::string(name));
}

std::string escaped(const std::string& path) {
  std::string result;
  for (const char c : path) {
    switch (c) {
      case ' ': result += "\\040"; break;
      case '\t': result += "\\011"; break;
      case '\n': result += "\\012"; break;
      case '\\': result += "\\134"; break;
      default: result += c;
    }
  }
  return result;
}
}  // namespace

DGPP_TEST(cache_storage_mount_parser_roundtrips_escaped_paths_and_mount_ids) {
  std::mt19937 random(154);
  const std::string alphabet = "abc /\t\n\\0123456789";
  for (int n = 0; n < 2000; ++n) {
    std::string path = "/cache";
    for (unsigned i = 0, count = random() % 100; i < count; ++i) path += alphabet[random() % alphabet.size()];
    const uint64_t id = 100 + random();
    // Same device and same pathname, but a different mount ID (stacked/bind).
    std::istringstream input("1 0 8:1 / / rw - ext4 /dev/nvme0n1 rw\n" +
                             std::to_string(id + 1) + " 1 8:1 / " + escaped(path) + " rw - tmpfs tmpfs rw\n" +
                             std::to_string(id) + " 1 8:1 / " + escaped(path) +
                             " rw,nosuid shared:12 master:2 - ext4 /dev/nvme0n1 rw\n");
    const auto result = dgpp::read_cache_mount(input, id);
    require(result.mount_error.empty() && result.mount_point == path && result.filesystem == "ext4" &&
                !result.ram_backed, "escaped mountpoint and exact mount identity");
  }
}

DGPP_TEST(cache_storage_mount_parser_refuses_unverifiable_assertions) {
  for (const std::string line : {"", "77", "77 1 8:1 / /cache rw", "77 1 8:1 / relative rw - ext4 x rw",
                                 "77 1 8:1 / /bad\\040\\ rw - ext4 x rw", "77 1 8:1 / /bad\\000 rw - ext4 x rw",
                                 "77 1 8:1 / /cache rw - ext4", "77x 1 8:1 / /cache rw - ext4 x rw"}) {
    std::istringstream input(line);
    const auto result = dgpp::read_cache_mount(input, 77);
    require(!result.mount_error.empty(), "malformed/missing mount refused: " + line);
    require(!dgpp::check_cache_storage(result, "/cache", true).empty(), "test override cannot bypass required mount");
  }
  // Exercise malformed input, including embedded NUL, independently of /proc.
  std::mt19937 random(154);
  for (int n = 0; n < 2000; ++n) {
    std::string line = "77 ";
    for (unsigned i = 0, count = random() % 256; i < count; ++i) line += static_cast<char>(random() % 256);
    std::istringstream input(line);
    (void)dgpp::read_cache_mount(input, 77);
  }
}

DGPP_TEST(cache_storage_policy_distinguishes_ram_from_container_layers) {
  for (const std::string type : {"tmpfs", "ramfs", "hugetlbfs", "overlay", "aufs", "fuse.overlayfs", "ext4"}) {
    std::istringstream input("77 1 8:1 / / rw - " + type + " source rw\n");
    const auto storage = dgpp::read_cache_mount(input, 77);
    const bool ram = type == "tmpfs" || type == "ramfs" || type == "hugetlbfs";
    const bool layer = type == "overlay" || type == "aufs" || type == "fuse.overlayfs";
    require(storage.ram_backed == ram && storage.container_layer == layer, "classification " + type);
    require(dgpp::check_cache_storage(storage, "", false).empty() == !ram, "RAM needs explicit allowance");
    require(dgpp::check_cache_storage(storage, "", true).empty(), "explicit test allowance");
    require(storage.warning(true).empty() == !(ram || layer), "visible warning " + type);
  }
}

DGPP_TEST(cache_storage_dry_run_resolves_missing_relative_paths_without_creation) {
  Directory temp;
  fs::create_directory_symlink(temp.path, temp.path / "alias");
  const auto target = temp.path / "alias" / "new" / "cache";
  const auto result = plan(target.lexically_relative(fs::current_path()), "", false, true);
  require(result.error.empty(), result.error);
  require(result.directory == (temp.path / "new/cache").string() && !fs::exists(temp.path / "new"),
          "canonical absolute destination without creation");
  require(!result.storage.mount_point.empty() && !result.storage.filesystem.empty(), "mount diagnostics");
  seed(temp.path);
  require(plan(temp.path, "", false, true).error.empty(), "existing cache dry run");
  preserved(temp.path);
  fs::create_directory_symlink(temp.path / "absent", temp.path / "dangling");
  require(!plan(temp.path / "dangling/cache", "", true, true).error.empty(), "dangling symlink refused");
  require(!plan(temp.path / "rank0.slab/child", "", true, true).error.empty(), "file is not a directory");
  require(!plan(std::string("/tmp/cache\0elsewhere", 20), "", true, true).error.empty(), "NUL path refused");
}

DGPP_TEST(cache_storage_required_mount_refuses_existing_unmounted_directory_without_writes) {
  Directory temp;
  seed(temp.path);
  for (bool create : {false, true}) {
    const auto rejected = plan(temp.path, temp.path.string(), create, true);
    require(rejected.error.find("required_mount") != std::string::npos, "existing directory is not a mount");
    preserved(temp.path);
    const auto absent = plan(temp.path / "absent/cache", (temp.path / "absent").string(), create, true);
    require(!absent.error.empty() && !fs::exists(temp.path / "absent"), "missing mount leaves no directory");
  }
}

DGPP_TEST(cache_storage_tmpfs_refusal_does_not_depend_on_direct_io_support) {
  if (!fs::is_directory("/dev/shm")) dgpp::test::skip("/dev/shm unavailable");
  Directory temp("/dev/shm");
  const auto storage = dgpp::inspect_cache_directory(temp.path.string());
  if (storage.filesystem != "tmpfs") dgpp::test::skip("/dev/shm is not tmpfs");
  seed(temp.path);
  for (bool create : {false, true}) {
    require(plan(temp.path, "", create).error.find("RAM-backed") != std::string::npos, "tmpfs refused unconditionally");
    preserved(temp.path);
    require(plan(temp.path / "absent", "", create).error.find("RAM-backed") != std::string::npos &&
                !fs::exists(temp.path / "absent"), "tmpfs refusal does not create a directory");
  }
  require(plan(temp.path, "", false, true).error.empty(), "explicit test allowance accepts tmpfs");
  const int fd = ::open((temp.path / "rank0.slab").c_str(), O_RDONLY | O_CLOEXEC);
  require(fd >= 0, "open tmpfs file");
  const auto actual = dgpp::inspect_cache_storage_fd(fd);
  ::close(fd);
  require(!dgpp::check_cache_storage(actual, "", false).empty(), "descriptor validation rejects RAM too");
}

DGPP_TEST(cache_storage_mount_namespace) {
  const char* fixture = std::getenv("DGPP_TEST_MOUNTS_ROOT");
  if (!fixture) dgpp::test::skip("run tests/python/nvme_storage_mount_check.py for real mount coverage");
  const fs::path root(fixture), mounted = root / "bind";
  const auto mount = dgpp::inspect_cache_directory(mounted.string());
  require(mount.mount_point == mounted.string(), "bind mount is visible despite identical st_dev");
  const auto dry = plan(mounted / "new/cache", mounted.string());
  require(dry.error.empty() && !fs::exists(mounted / "new"), "mounted dry run creates nothing");
  require(plan(mounted / "new/cache", mounted.string(), true).error.empty(), "writable bind mount accepted");
  require(fs::is_directory(root / "source/new/cache"), "creation lands on bind source");
  const auto special = root / "space tab\tback\\line\n";
  require(plan(special / "new", special.string()).error.empty(), "escaped actual mountpoint");
  fs::create_directory_symlink(mounted, root / "alias");
  require(plan(root / "alias/new/cache", mounted.string()).error.empty(), "symlink into required mount");
  fs::create_directory_symlink(root / "outside", mounted / "escape");
  require(!plan(mounted / "escape/new", mounted.string(), true).error.empty() &&
              !fs::exists(root / "outside/new"), "symlink escape refused before mkdir");
  require(!plan(mounted / "nested/new", mounted.string(), true).error.empty(), "nested mount is different storage");
  require(plan(mounted / "missing/../nested/new", (mounted / "nested").string()).error.empty(),
          "normalization still identifies actual nested mount");
  const auto ro = root / "readonly";
  require(!plan(ro / "refused-cache", ro.string(), true).error.empty() && !fs::exists(ro / "refused-cache"),
          "read-only mount refused");
  for (const auto& name : {"tmpfs", "ramfs"}) {
    const auto ram = root / name;
    seed(ram);
    require(plan(ram, ram.string(), true).error.find("RAM-backed") != std::string::npos, "RAM mount refused");
    preserved(ram);
  }
  const auto overlay = root / "overlay";
  const auto layer = plan(overlay / "new", "", false);
  require(layer.error.empty() && layer.storage.container_layer && !layer.storage.warning(false).empty(),
          "overlay warns without refusal");
  require(!plan(overlay / "new", mounted.string(), true).error.empty() && !fs::exists(overlay / "new"),
          "required mount rejects overlay fallback without creation");
  // Unmounting the required bind leaves an existing ordinary directory behind.
  // The Python driver invokes a separate test after removing the mount.
}

DGPP_TEST(cache_storage_unmounted_namespace) {
  const char* fixture = std::getenv("DGPP_TEST_UNMOUNTED_ROOT");
  if (!fixture) dgpp::test::skip("run tests/python/nvme_storage_mount_check.py for unmount coverage");
  const fs::path mount = fs::path(fixture) / "bind";
  seed(mount);
  require(!plan(mount, mount.string(), true).error.empty(), "previous mount now ordinary directory refused");
  preserved(mount);
  require(!plan(mount / "absent", mount.string(), true).error.empty() && !fs::exists(mount / "absent"),
          "no fallback directory creation after unmount");
}

int main() { return dgpp::test::run_all(); }
