#include "loaders/safetensors.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>

#include "../common/temp_dir.hpp"
#include "common/test.hpp"

namespace {
namespace fs = std::filesystem;

// Builds a tiny safetensors file with one F32, one BF16, one F8_E4M3 tensor.
fs::path make_fixture(const fs::path& dir) {
  fs::path file = dir / "fixture.safetensors";

  const char* header =
      R"({"a_f32":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},)"
      R"("b_bf16":{"dtype":"BF16","shape":[4],"data_offsets":[24,32]},)"
      R"("c_fp8":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[32,36]}})";
  uint64_t hlen = static_cast<uint64_t>(std::strlen(header));

  std::FILE* f = std::fopen(file.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write fixture");
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header, hlen, 1, f);

  float f32v[6] = {1.f, -2.5f, 3.25f, 4.f, -5.f, 0.125f};
  uint16_t bf16v[4];
  for (int i = 0; i < 4; ++i)
    bf16v[i] = dgpp::float_to_bf16_bits(0.5f * i);
  uint8_t fp8v[4] = {0x38, 0x40, 0x7E, 0x00};  // .28125?, 0.5, 448, 0

  std::fwrite(f32v, sizeof(f32v), 1, f);
  std::fwrite(bf16v, sizeof(bf16v), 1, f);
  std::fwrite(fp8v, sizeof(fp8v), 1, f);
  std::fclose(f);
  return file;
}
}  // namespace

DGPP_TEST(safetensors_mmap_reader_roundtrip) {
  const dgpp::test::TempDir fixture("dgpp_st_test");
  fs::path p = make_fixture(fixture.path());
  auto sf = dgpp::SafetensorsFile::open(p.string());

  const dgpp::TensorInfo* a = sf->find("a_f32");
  if (!a || a->dtype != dgpp::DType::F32) throw std::runtime_error("a meta");
  // This fixture deliberately has an unpadded JSON header. Read its byte
  // storage without imposing the alignment of float/uint16_t on the mapping.
  float af[6];
  std::memcpy(af, a->data, sizeof(af));
  if (af[0] != 1.f || af[1] != -2.5f || af[5] != 0.125f)
    throw std::runtime_error("a data");

  const dgpp::TensorInfo& b = sf->at("b_bf16");
  if (b.dtype != dgpp::DType::BF16 || b.numel() != 4)
    throw std::runtime_error("b meta");
  uint16_t bb[4];
  std::memcpy(bb, b.data, sizeof(bb));
  if (bb[2] != dgpp::float_to_bf16_bits(1.0f))
    throw std::runtime_error("b data");

  const dgpp::TensorInfo& c = sf->at("c_fp8");
  if (c.dtype != dgpp::DType::F8_E4M3) throw std::runtime_error("c meta");
  const uint8_t* cb = static_cast<const uint8_t*>(c.data);
  // fixture bytes: 0x38 -> +1.0, 0x40 -> +2.0, 0x7E -> max finite 448,
  // 0x00 -> +0
  if (cb[2] != 0x7E) throw std::runtime_error("c saturate byte");
  if (dgpp::fp8_e4m3_bits_to_float(cb[0]) != 1.0f)
    throw std::runtime_error("c[0] decode");
  if (dgpp::fp8_e4m3_bits_to_float(cb[1]) != 2.0f)
    throw std::runtime_error("c[1] decode");

  if (sf->find("missing") != nullptr) throw std::runtime_error("ghost tensor");

  std::error_code ec;
  fs::remove(p, ec);
}

namespace {
// Each case owns its directory, including when parsing throws. This also lets
// sanitizer runs check rejected opens without retaining files or mappings.
struct SafetensorsFixture {
  fs::path dir;
  fs::path path;

  SafetensorsFixture() {
    std::string pattern = (fs::temp_directory_path() / "dgpp_st_bounds_XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const char* created = ::mkdtemp(name.data());
    if (!created) throw std::runtime_error("cannot create safetensors test directory");
    dir = created;
    path = dir / "fixture.safetensors";
  }
  ~SafetensorsFixture() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }

  void write(std::string_view header, size_t payload,
             std::optional<uint64_t> declared_length = std::nullopt) const {
    std::ofstream out(path, std::ios::binary);
    const uint64_t length = declared_length.value_or(header.size());
    for (unsigned i = 0; i < 8; ++i) out.put(static_cast<char>((length >> (8 * i)) & 255));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (size_t i = 0; i < payload; ++i) out.put(static_cast<char>(i % 251));
    if (!out) throw std::runtime_error("cannot write safetensors test file");
  }
};

std::string tensor_header(std::string_view shape, std::string_view offsets,
                          std::string_view dtype = "U8") {
  return "{\"x\":{\"dtype\":\"" + std::string(dtype) + "\",\"shape\":" + std::string(shape) +
         ",\"data_offsets\":" + std::string(offsets) + "}}";
}

void expect_rejected(std::string_view header, size_t payload, std::string_view diagnostic,
                     std::optional<uint64_t> length = std::nullopt) {
  SafetensorsFixture fixture;
  fixture.write(header, payload, length);
  try {
    auto file = dgpp::SafetensorsFile::open(fixture.path.string());
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    if (message.find(diagnostic) == std::string::npos)
      throw std::runtime_error("missing safetensors diagnostic: " + message);
    return;
  }
  throw std::runtime_error("accepted invalid safetensors metadata: " + std::string(header));
}
}  // namespace

DGPP_TEST(safetensors_rejects_header_lengths_outside_file) {
  expect_rejected("", 0, "header length", std::numeric_limits<uint64_t>::max());
  expect_rejected("{}", 0, "header length", 3);
}

DGPP_TEST(safetensors_rejects_invalid_data_offsets) {
  for (const auto offsets : {"[-1,0]", "[-8,-4]", "[0,-1]", "[1,0]", "[1048576,0]", "[0,2]",
                             "[9223372036854775807,0]", "[18446744073709551615,0]", "[0.5,1]",
                             "[\"0\",1]", "[false,1]", "[0,1,2]", "[0]", "[]", "{}"}) {
    expect_rejected(tensor_header("[1]", offsets), 1, "data_offsets");
  }
}

DGPP_TEST(safetensors_rejects_invalid_shapes) {
  for (const auto shape :
       {"{}", "[-1]", "[0,-1]", "[1.5]", "[\"1\"]", "[true]", "[18446744073709551615]"}) {
    expect_rejected(tensor_header(shape, "[0,0]"), 0, "shape");
  }
}

DGPP_TEST(safetensors_rejects_size_mismatch_and_overflow) {
  expect_rejected(tensor_header("[2]", "[0,4]", "F32"), 4, "shape/dtype");
  expect_rejected(tensor_header("[1]", "[0,8]", "F32"), 8, "shape/dtype");
  expect_rejected(tensor_header("[4294967296,4294967296]", "[0,0]"), 0, "element count");
  const auto wide_elements = std::to_string(std::numeric_limits<size_t>::max() / 4 + 1);
  expect_rejected(tensor_header("[" + wide_elements + "]", "[0,0]", "F32"), 0, "shape/dtype");
  // Match the reference's ordered checked product: a later zero does not
  // retroactively make an earlier overflowing multiply valid.
  expect_rejected(tensor_header("[9223372036854775807,9223372036854775807,0]", "[0,0]"), 0,
                  "element count");
}

DGPP_TEST(safetensors_scalar_empty_and_payload_end_views) {
  const char* header = R"({"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]},)"
                       R"("empty":{"dtype":"BF16","shape":[2,0,3],"data_offsets":[4,4]},)"
                       R"("zero_first":{"dtype":"U8","shape":[0,5],"data_offsets":[4,4]}})";
  SafetensorsFixture fixture;
  fixture.write(header, 4);
  auto file = dgpp::SafetensorsFile::open(fixture.path.string());
  const auto& scalar = file->at("scalar");
  if (file->tensor_count() != 3 || scalar.numel() != 1 || scalar.nbytes() != 4)
    throw std::runtime_error("scalar metadata changed");
  std::array<uint8_t, 4> bytes{};
  std::memcpy(bytes.data(), scalar.data, bytes.size());
  if (bytes != std::array<uint8_t, 4>{0, 1, 2, 3}) throw std::runtime_error("scalar bytes changed");
  for (const auto name : {"empty", "zero_first"}) {
    const auto& tensor = file->at(name);
    if (tensor.numel() != 0 || tensor.nbytes() != 0 || tensor.owner != file.get() ||
        tensor.data != file->map_base() + file->map_size())
      throw std::runtime_error("empty payload-end view changed");
  }
  // A zero reached before any overflowing product remains a valid empty tensor.
  file.reset();
  const auto large = std::to_string(std::numeric_limits<size_t>::max() / 2);
  fixture.write(tensor_header("[0," + large + "," + large + "]", "[0,0]"), 0);
  auto empty = dgpp::SafetensorsFile::open(fixture.path.string());
  if (empty->at("x").numel() != 0) throw std::runtime_error("empty element count changed");
}

DGPP_TEST(safetensors_dtype_aliases_and_padded_header) {
  struct DTypeCase {
    const char* name;
    dgpp::DType dtype;
    size_t width;
  };
  for (const auto& c : std::array<DTypeCase, 12>{{{"F32", dgpp::DType::F32, 4},
                                                  {"F16", dgpp::DType::F16, 2},
                                                  {"BF16", dgpp::DType::BF16, 2},
                                                  {"F8_E4M3", dgpp::DType::F8_E4M3, 1},
                                                  {"F8", dgpp::DType::F8_E4M3, 1},
                                                  {"FP8", dgpp::DType::F8_E4M3, 1},
                                                  {"I64", dgpp::DType::I64, 8},
                                                  {"I32", dgpp::DType::I32, 4},
                                                  {"U8", dgpp::DType::U8, 1},
                                                  {"I8", dgpp::DType::I8, 1},
                                                  {"F8_E8M0", dgpp::DType::F8_E8M0, 1},
                                                  {"BOOL", dgpp::DType::U8, 1}}}) {
    std::string header = tensor_header("[3]", "[0," + std::to_string(3 * c.width) + "]", c.name);
    while ((8 + header.size()) % 8) header += ' ';
    SafetensorsFixture fixture;
    fixture.write(header, 3 * c.width);
    auto file = dgpp::SafetensorsFile::open(fixture.path.string());
    const auto& tensor = file->at("x");
    if (tensor.dtype != c.dtype || tensor.numel() != 3 || tensor.nbytes() != 3 * c.width)
      throw std::runtime_error("dtype/size changed: " + std::string(c.name));
    const auto* bytes = static_cast<const uint8_t*>(tensor.data);
    for (size_t i = 0; i < tensor.nbytes(); ++i)
      if (bytes[i] != i % 251) throw std::runtime_error("dtype fixture bytes changed");
  }
}
