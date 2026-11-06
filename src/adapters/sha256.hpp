#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <openssl/evp.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace dgpp::adapters {
class Sha256 {
 public:
  Sha256() : context_(EVP_MD_CTX_new(), EVP_MD_CTX_free) {
    if (!context_ || EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1)
      throw std::runtime_error("SHA-256 initialization failed");
  }
  void update(const void* bytes, size_t count) {
    if (finished_ || EVP_DigestUpdate(context_.get(), bytes, count) != 1)
      throw std::runtime_error("SHA-256 update failed");
  }
  void update(std::string_view bytes) { update(bytes.data(), bytes.size()); }
  std::string finish() {
    std::array<unsigned char, 32> bytes;
    unsigned int size = 0;
    if (finished_ || EVP_DigestFinal_ex(context_.get(), bytes.data(), &size) != 1 ||
        size != bytes.size())
      throw std::runtime_error("SHA-256 finalization failed");
    finished_ = true;
    std::string hex;
    for (auto b : bytes) {
      hex += "0123456789abcdef"[b >> 4];
      hex += "0123456789abcdef"[b & 15];
    }
    return hex;
  }

 private:
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_;
  bool finished_ = false;
};
inline std::string sha256(const void* data, size_t bytes) {
  Sha256 hash;
  hash.update(data, bytes);
  return hash.finish();
}
inline std::string sha256(std::string_view text) {
  return sha256(text.data(), text.size());
}
inline std::string file_sha256(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot hash file: " + path);
  Sha256 hash;
  std::array<char, 1024 * 1024> chunk;
  while (input.read(chunk.data(), chunk.size()) || input.gcount())
    hash.update(chunk.data(), input.gcount());
  if (!input.eof()) throw std::runtime_error("cannot finish hashing file: " + path);
  return hash.finish();
}
}  // namespace dgpp::adapters
