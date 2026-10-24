#include "loaders/resident_image.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"

namespace {
namespace fs = std::filesystem;
constexpr uint64_t kKey = 0x126;
constexpr size_t kHeaderBytes = 64;
constexpr size_t kEntryBytes = 32;

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

struct VerifyEnvironment {
  std::optional<std::string> saved;
  VerifyEnvironment() {
    if (const char* value = std::getenv("DGPP_RESIDENT_CACHE_VERIFY")) saved = value;
    ::unsetenv("DGPP_RESIDENT_CACHE_VERIFY");
  }
  ~VerifyEnvironment() {
    if (saved)
      ::setenv("DGPP_RESIDENT_CACHE_VERIFY", saved->c_str(), 1);
    else
      ::unsetenv("DGPP_RESIDENT_CACHE_VERIFY");
  }
};

struct Fixture {
  fs::path dir;
  fs::path path;
  std::vector<uint8_t> payload = std::vector<uint8_t>(8192, 0xa5);

  explicit Fixture(int layers = 2) {
    std::string pattern = (fs::temp_directory_path() / "dgpp_image_XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    require(::mkdtemp(name.data()) != nullptr, "create image fixture directory");
    dir = name.data();
    dgpp::ResidentImage image(dir.string(), kKey, layers);
    path = image.path();
    image.write_layer(0, payload.data(), payload.size());
    image.write_layer(layers - 1, payload.data(), payload.size());
  }
  ~Fixture() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }

  void patch(size_t offset, uint64_t value) const {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(static_cast<std::streamoff>(offset));
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
    require(bool(file), "patch image fixture");
  }

  void entry(int layer, uint64_t offset, uint64_t bytes) const {
    patch(kHeaderBytes + kEntryBytes * layer, offset);
    patch(kHeaderBytes + kEntryBytes * layer + 8, bytes);
  }
};
}  // namespace

DGPP_TEST(resident_image_rejects_invalid_blob_ranges) {
  const uint64_t max = std::numeric_limits<uint64_t>::max();
  const std::array<std::pair<uint64_t, uint64_t>, 8> ranges{{
      {0, 8192},           // header
      {64, 8192},          // table
      {4095, 8192},        // unaligned start
      {4097, 8192},        // unaligned, still inside the file
      {1ULL << 40, 8192},  // offset past EOF
      {4096, 1ULL << 40},  // length past EOF
      {max - 4095, 8192},  // aligned start whose end wraps
      {4096, max},         // length whose end wraps
  }};
  for (const auto& [offset, bytes] : ranges) {
    Fixture fixture;
    fixture.entry(0, offset, bytes);
    dgpp::ResidentImage image(fixture.dir.string(), kKey, 2);
    require(!image.has_layer(0) && image.layer_bytes(0) == 0, "invalid entry is a cache miss");
    require(image.has_layer(1) && image.present() == 1, "valid neighboring layer survives");
    std::vector<uint8_t> output(fixture.payload.size());
    image.read_layer(1, output.data(), output.size(), true);
    require(output == fixture.payload, "neighbor's payload survives");
    image.write_layer(0, fixture.payload.data(), fixture.payload.size());
    image.read_layer(0, output.data(), output.size(), true);
    require(output == fixture.payload, "invalid layer can be rebuilt");
  }
}

DGPP_TEST(resident_image_rejects_aligned_offsets_inside_large_table) {
  Fixture fixture(130);  // The table extends past the first 4 KiB boundary.
  fixture.entry(0, 4096, fixture.payload.size());
  dgpp::ResidentImage image(fixture.dir.string(), kKey, 130);
  require(!image.has_layer(0), "aligned offset inside table is rejected");
  require(image.has_layer(129), "valid layer after large table survives");
}

DGPP_TEST(resident_image_accepts_tail_at_eof_and_rejects_truncation) {
  Fixture fixture;
  {
    dgpp::ResidentImage image(fixture.dir.string(), kKey, 2);
    const std::vector<uint8_t> odd(4097, 0x3c);
    image.write_layer(1, odd.data(), odd.size());
  }
  {
    dgpp::ResidentImage image(fixture.dir.string(), kKey, 2);
    std::vector<uint8_t> output(4097);
    image.read_layer(1, output.data(), output.size(), true);
    require(output == std::vector<uint8_t>(4097, 0x3c), "odd tail ending at EOF is valid");
  }
  fs::resize_file(fixture.path, fs::file_size(fixture.path) - 1);
  dgpp::ResidentImage truncated(fixture.dir.string(), kKey, 2);
  require(!truncated.has_layer(1) && truncated.has_layer(0), "truncated blob is absent");
}

DGPP_TEST(resident_image_payload_verification_is_opt_out) {
  VerifyEnvironment environment;
  Fixture fixture;
  fixture.patch(4096, 0);  // Corrupt payload while preserving metadata and fold.
  std::vector<uint8_t> output(fixture.payload.size());
  for (const char* value : std::array<const char*, 5>{nullptr, "", "1", "false", "00"}) {
    if (value)
      ::setenv("DGPP_RESIDENT_CACHE_VERIFY", value, 1);
    else
      ::unsetenv("DGPP_RESIDENT_CACHE_VERIFY");
    dgpp::ResidentImage image(fixture.dir.string(), kKey, 2);
    require(!image.try_read_layer(0, output.data(), output.size()),
            "corrupt payload refused by default");
    require(!image.has_layer(0) && image.has_layer(1), "only corrupt layer invalidated");
  }
  ::setenv("DGPP_RESIDENT_CACHE_VERIFY", "0", 1);
  dgpp::ResidentImage image(fixture.dir.string(), kKey, 2);
  require(image.try_read_layer(0, output.data(), output.size()),
          "explicit opt-out skips payload fold");
  require(output != fixture.payload, "opt-out fixture has changed bytes");
}

DGPP_TEST(resident_image_opt_out_preserves_metadata_and_size_checks) {
  VerifyEnvironment environment;
  ::setenv("DGPP_RESIDENT_CACHE_VERIFY", "0", 1);
  Fixture fixture;
  fixture.entry(0, 0, fixture.payload.size());
  dgpp::ResidentImage image(fixture.dir.string(), kKey, 2);
  require(!image.has_layer(0), "opt-out still rejects invalid offsets");
  std::vector<uint8_t> output(fixture.payload.size(), 0x7c);
  require(!image.try_read_layer(1, output.data(), output.size() - 1),
          "size mismatch is a cache miss");
  require(output == std::vector<uint8_t>(output.size(), 0x7c),
          "size mismatch does not write destination");
  require(!image.has_layer(1), "stale layer invalidated");
  image.write_layer(1, fixture.payload.data(), fixture.payload.size());
  require(image.try_read_layer(1, output.data(), output.size()), "rebuilt layer readable");
  require(output == fixture.payload, "rebuilt layer bytes match");
}

DGPP_TEST(resident_image_read_failure_rebuilds_without_reopening) {
  VerifyEnvironment environment;
  Fixture fixture;
  dgpp::ResidentImage image(fixture.dir.string(), kKey, 2);
  fs::resize_file(fixture.path, 4096);  // Truncated after the table was read.
  std::vector<uint8_t> output(fixture.payload.size());
  require(!image.try_read_layer(0, output.data(), output.size()), "short read is a cache miss");
  image.write_layer(0, fixture.payload.data(), fixture.payload.size());
  require(image.try_read_layer(0, output.data(), output.size()),
          "replacement blob passes verification");
  require(output == fixture.payload, "replacement blob bytes match");
}
