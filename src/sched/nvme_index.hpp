#pragma once
// The existing DGPPNVM2 file envelope, shared by read-only reuse preflight
// and the tier. Its payload is DiskCache's world index; digests are rank-local.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace dgpp::sched {

inline constexpr char kNvmeIndexMagic[8] = {'D', 'G', 'P', 'P', 'N', 'V', 'M', '2'};

inline bool parse_nvme_index(const std::vector<uint8_t>& file, const std::string& binding,
                             std::vector<uint8_t>* shared, std::vector<uint64_t>* digests) {
  size_t at = 0;
  const auto integer = [&](unsigned bytes, uint64_t* value) {
    if (bytes > file.size() - at) return false;
    *value = 0;
    for (unsigned i = 0; i < bytes; ++i) *value |= static_cast<uint64_t>(file[at + i]) << (8 * i);
    at += bytes;
    return true;
  };
  if (file.size() < 8 || std::memcmp(file.data(), kNvmeIndexMagic, 8) != 0) return false;
  at = 8;
  uint64_t n = 0;
  if (!integer(4, &n) || n > file.size() - at) return false;
  if (std::string(reinterpret_cast<const char*>(file.data()) + at, static_cast<size_t>(n)) != binding)
    return false;
  at += static_cast<size_t>(n);
  uint64_t length = 0;
  if (!integer(8, &length) || length > file.size() - at) return false;
  const size_t payload = at;
  at += static_cast<size_t>(length);
  uint64_t count = 0;
  if (!integer(4, &count) || count != (file.size() - at) / 8 || (file.size() - at) % 8 != 0)
    return false;
  shared->assign(file.begin() + static_cast<std::ptrdiff_t>(payload),
                 file.begin() + static_cast<std::ptrdiff_t>(payload + length));
  digests->clear();
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t value = 0;
    if (!integer(8, &value)) return false;
    digests->push_back(value);
  }
  return true;
}

inline std::vector<uint8_t> write_nvme_index(const std::string& binding, const std::vector<uint8_t>& shared,
                                            const std::vector<uint64_t>& digests) {
  std::vector<uint8_t> out(kNvmeIndexMagic, kNvmeIndexMagic + 8);
  const auto integer = [&](uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
  };
  integer(binding.size(), 4);
  out.insert(out.end(), binding.begin(), binding.end());
  integer(shared.size(), 8);
  out.insert(out.end(), shared.begin(), shared.end());
  integer(digests.size(), 4);
  for (const uint64_t d : digests) integer(d, 8);
  return out;
}

}  // namespace dgpp::sched
