#pragma once
// Inkling 2's bundled draft model (mtp.safetensors, `mtp_config`) on the
// shared block-draft executor (models/dflash2/drafter.hpp). The released
// reference is vLLM's models/inkling2/nvidia/dspark.py.
//
// The draft reads the target's states at `target_layer_ids`: the partial
// state at the entry of each named visit, and the final-norm output under
// the last id (the target's visit count). Every draft layer mixes those
// states with its own softmax weights into context keys and values. A block
// is the last committed token's normed embedding followed by learned mask
// rows, each mask row predicts the token at its own position, and a
// recurrent head corrects the block's base logits one proposal after
// another through the target's head.
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "models/dflash2/dflash2.hpp"
#include "models/inkling2/config.hpp"

namespace dgpp {

// The executor's view of `cfg.draft`. Throws without an mtp_config.
DFlash2Config inkling2_draft_config(const Inkling2Config& cfg);

// What the loader takes from the target model.
struct Inkling2DraftTarget {
  const uint16_t* unembed = nullptr;     // device BF16 [vocab_count, hidden]: this rank's head rows
  int64_t vocab_count = 0;
  const uint16_t* embed_norm = nullptr;  // device BF16 [hidden]
  int64_t padded_vocab = 0;              // the rows the draft's token table stores
};

// The arena load_inkling2_draft allocates for one rank (the memory plan's line).
size_t inkling2_draft_weights_bytes(const DFlash2Config& cfg, int world, int64_t vocab_count, int64_t padded_vocab);

// Reads mtp.safetensors of `checkpoint_dir`, checks every name, dtype and
// shape, uploads this rank's slices (the heads, the key/value heads and the
// MLP rows split across the world, everything else replicated) and folds
// the target's head rows through the recurrent head's output projection.
DFlash2Weights load_inkling2_draft(const DFlash2Config& cfg, const std::string& checkpoint_dir, cudaStream_t stream,
                                   int rank, int world, const Inkling2DraftTarget& target);

}  // namespace dgpp
