#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace imagefx {
// Tests own this directory and force the default verification policy, restoring
// both the loader's cache destination and the caller's environment afterwards.
template <class Stream>
struct Cache {
  std::filesystem::path dir;
  std::string previous = Stream::resident_image_dir();
  std::optional<std::string> verify;
  explicit Cache(std::filesystem::path path) : dir(std::move(path)) {
    std::filesystem::remove_all(dir);
    if (const char* value = std::getenv("DGPP_RESIDENT_CACHE_VERIFY")) verify = value;
    ::unsetenv("DGPP_RESIDENT_CACHE_VERIFY");
    Stream::set_resident_image_dir(dir.string());
  }
  ~Cache() {
    Stream::set_resident_image_dir(previous);
    if (verify)
      ::setenv("DGPP_RESIDENT_CACHE_VERIFY", verify->c_str(), 1);
    else
      ::unsetenv("DGPP_RESIDENT_CACHE_VERIFY");
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }

  // The on-disk v2 table follows a 64-byte header. Corrupt layer zero's
  // offset (field 0), length (8) or fold (16) without changing the image key.
  void corrupt(size_t field) const {
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      if (entry.path().extension() != ".img") continue;
      std::fstream file(entry.path(), std::ios::in | std::ios::out | std::ios::binary);
      uint64_t value = 0;
      file.seekg(64 + field);
      file.read(reinterpret_cast<char*>(&value), sizeof(value));
      value = field == 0 ? 0 : value ^ 1;
      file.seekp(64 + field);
      file.write(reinterpret_cast<const char*>(&value), sizeof(value));
      if (!file) throw std::runtime_error("cannot corrupt test resident image");
      return;
    }
    throw std::runtime_error("test resident image missing");
  }
};
}  // namespace imagefx
