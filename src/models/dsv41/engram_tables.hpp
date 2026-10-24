#pragma once
// The Engram hash constants of a DeepSeek-V4.1-Flash checkpoint (2026-09-13,
// docs/deepseek_v41_flash_plan.md §1.6.1): the compressed token map, the
// (layer, n-gram, head) primes and row offsets, the per-layer multipliers
// and the pad class — derived once by tools/dsv41_engram_tables.py from the
// tokenizer with the reference's own libraries and written beside the
// snapshot as `dgpp_engram_tables.json`. The loader reads and checks them
// against the config; nothing here is recomputed (numpy's PCG64 stream
// and the tokenizer's normalizer chain are not reproduced in C++).
#include <cstdint>
#include <string>
#include <vector>

#include "models/dsv41/config.hpp"

namespace dgpp {

constexpr const char* kDsv41EngramSidecarName = "dgpp_engram_tables.json";

struct Dsv41EngramSidecar {
  std::string tokenizer_sha256;
  int vocab_size = 0;
  int compressed_vocab_size = 0;
  int64_t pad_id = 0;
  int32_t pad_class = 0;
  std::vector<int> layer_ids;
  int max_ngram_size = 0;
  int n_heads = 0;
  std::vector<int64_t> num_embeddings;
  std::vector<int64_t> primes;       // [layers][max_ngram - 1][heads], row-major
  std::vector<int64_t> offsets;      // the same shape
  std::vector<int64_t> multipliers;  // [layers][max_ngram]
  std::vector<int32_t> token_map;    // [vocab_size]

  int layers() const { return static_cast<int>(layer_ids.size()); }
  int ngrams() const { return max_ngram_size - 1; }
  int64_t prime(int layer, int ngram, int head) const {
    return primes[(static_cast<size_t>(layer) * ngrams() + ngram) * n_heads + head];
  }
  int64_t offset(int layer, int ngram, int head) const {
    return offsets[(static_cast<size_t>(layer) * ngrams() + ngram) * n_heads + head];
  }
};

// Parses `path` and checks it against the config (layer ids, n-gram size,
// heads, table sizes, the pad class, the vocabulary): throws
// std::runtime_error naming the field on any disagreement.
Dsv41EngramSidecar dsv41_load_engram_sidecar(const std::string& path, const Dsv41TextConfig& cfg);
// The sidecar of a checkpoint directory (`<dir>/dgpp_engram_tables.json`),
// or, when that has none, of `fallback_dir` (the Engram source snapshot an
// EXL3 pack reads its tables from).
Dsv41EngramSidecar dsv41_load_engram_sidecar_for(const std::string& checkpoint_dir,
                                                 const Dsv41TextConfig& cfg, const std::string& fallback_dir = "");

}  // namespace dgpp
