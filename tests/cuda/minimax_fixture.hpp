#pragma once
// A second fixture, generated from the native binding table. The Python
// oracle's independent inventory remains a separate gate for table mistakes.
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "glm_rng.hpp"
#include "models/minimax/binding.hpp"

namespace minimaxfx {
inline void write(const std::string& dir, const std::string& config) {
  using namespace dgpp;
  const auto cfg = MinimaxConfig::from_json_file(config);
  const auto table = minimax_expected_text_tensors(cfg);
  std::string header = "{";
  std::vector<uint8_t> payload;
  for (const auto& e : table) {
    glmrng::Rng rng(glmrng::seed_for(e.name));
    std::vector<uint8_t> bytes(e.nbytes());
    for (size_t i = 0; i < e.numel(); ++i) {
      switch (e.dtype) {
        case DType::BF16: {
          const float scale = e.shape.size() == 1 ? .06f : .3f;
          const uint16_t value = float_to_bf16_bits(scale * rng.normal3());
          std::memcpy(bytes.data() + i * 2, &value, 2);
          break;
        }
        case DType::F32: {
          const bool global =
              e.role == MinimaxTensorRole::Fp4Global || e.role == MinimaxTensorRole::Fp4InputGlobal;
          const float value = global                ? 1.2f + .2f * rng.unit()
                              : e.shape.size() == 1 ? .15f * float(i)
                                                    : .04f * rng.normal3();
          std::memcpy(bytes.data() + i * 4, &value, 4);
          break;
        }
        case DType::U8:
          bytes[i] =
              e.role == MinimaxTensorRole::Fp8Scale ? 112 + rng.next() % 4 : rng.next() % 256;
          break;
        case DType::F8_E4M3:
          bytes[i] = e.role == MinimaxTensorRole::Fp4Scale ? 8 + rng.next() % 16 : rng.next() % 256;
          if ((bytes[i] & 127) == 127) bytes[i] &= 0xf0;
          break;
        default:
          throw std::runtime_error("unexpected MiniMax fixture dtype");
      }
    }
    if (header.size() > 1) header += ',';
    header +=
        '"' + e.name + "\":{\"dtype\":\"" + std::string(dtype_name(e.dtype)) + "\",\"shape\":[";
    for (size_t d = 0; d < e.shape.size(); ++d) {
      if (d) header += ',';
      header += std::to_string(e.shape[d]);
    }
    header += "],\"data_offsets\":[" + std::to_string(payload.size()) + ',' +
              std::to_string(payload.size() + bytes.size()) + "]}";
    payload.insert(payload.end(), bytes.begin(), bytes.end());
  }
  header += '}';
  header.append((8 - header.size() % 8) % 8, ' ');
  std::filesystem::create_directories(dir);
  std::filesystem::copy_file(config, std::filesystem::path(dir) / "config.json",
                             std::filesystem::copy_options::overwrite_existing);
  std::ofstream out(std::filesystem::path(dir) / "model.safetensors", std::ios::binary);
  const uint64_t length = header.size();
  out.write(reinterpret_cast<const char*>(&length), 8);
  out.write(header.data(), header.size());
  out.write(reinterpret_cast<const char*>(payload.data()), payload.size());
  if (!out) throw std::runtime_error("cannot write MiniMax binding fixture");
  std::printf("MiniMax binding fixture: %zu tensors, %zu bytes\n", table.size(), payload.size());
}
}  // namespace minimaxfx
