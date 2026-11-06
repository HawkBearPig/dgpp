#pragma once
// The bundled DSpark model keeps its native MXFP8 tensors and uses the
// shared block-draft executor. Its taps address target layer INPUTS;
// block_size in the checkpoint counts proposals, excluding the anchor.
#include "models/dflash2/dflash2.hpp"

namespace dgpp {
DFlash2Config minimax_dspark_config(const minijson::Value&);
DFlash2Config minimax_dspark_config_file(const std::string&);
size_t minimax_dspark_weights_bytes(const DFlash2Config&, int world);
DFlash2Weights load_minimax_dspark(const DFlash2Config&, const std::string& directory, cudaStream_t,
                                   int rank, int world);
}  // namespace dgpp
