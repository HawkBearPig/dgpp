// CPU-only probe of the production encoder; compared with the independent oracle.
#include "loaders/fp8_quant.hpp"
#include <vector>

extern "C" void encode(const uint16_t* source, int64_t stride, int64_t rows,
                       int64_t columns, uint16_t* dequantized) {
  std::vector<uint8_t> payload(static_cast<size_t>(rows * columns));
  const int64_t scale_columns = dgpp::fp8_quant::scale_cols(columns);
  std::vector<float> scales(static_cast<size_t>(dgpp::fp8_quant::scale_rows(rows) * scale_columns));
  dgpp::fp8_quant::encode_block128(source, stride, rows, columns, payload.data(), scales.data(), 4);
  for (int64_t row = 0; row < rows; ++row)
    for (int64_t column = 0; column < columns; ++column) {
      const size_t index = static_cast<size_t>(row * columns + column);
      const float scale = scales[static_cast<size_t>((row / 128) * scale_columns + column / 128)];
      dequantized[index] = dgpp::float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(payload[index]) * scale);
    }
}
