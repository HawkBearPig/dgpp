#pragma once

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

namespace dgpp::test {

// Own a distinct fixture root, including when an assertion unwinds the test.
class TempDir {
 public:
  explicit TempDir(const char* prefix) {
    std::string name = (std::filesystem::temp_directory_path() / prefix).string() + "-XXXXXX";
    if (!::mkdtemp(name.data()))
      throw std::system_error(errno, std::generic_category(), "create test directory");
    path_ = name;
  }

  ~TempDir() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace dgpp::test
