#include "models/dsv41/engram_tables.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <stdexcept>

#include "loaders/minijson.hpp"

namespace dgpp {
namespace {

[[noreturn]] void reject(std::string_view field, std::string_view why) {
  throw std::runtime_error(std::format("engram sidecar.{}: {}", field, why));
}

const minijson::Value& require(const minijson::Value& v, std::string_view field) {
  const minijson::Value* f = v.find(field);
  if (!f || f->is_null()) reject(field, "missing");
  return *f;
}
int64_t require_int(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_number()) reject(field, "not a number");
  return f.as_int();
}
std::string require_string(const minijson::Value& v, std::string_view field) {
  const minijson::Value& f = require(v, field);
  if (!f.is_string()) reject(field, "not a string");
  return std::string(f.as_string());
}
// A flat list of numbers, or a nested list flattened row-major.
void flatten(const minijson::Value& v, std::string_view field, std::vector<int64_t>& out) {
  if (v.is_number()) {
    out.push_back(v.as_int());
    return;
  }
  if (!v.is_array()) reject(field, "not a number or array");
  for (const auto& item : v.items()) flatten(item, field, out);
}
std::vector<int64_t> require_flat(const minijson::Value& v, std::string_view field) {
  std::vector<int64_t> out;
  flatten(require(v, field), field, out);
  return out;
}

std::string read_file(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error(std::format("cannot open engram sidecar {}: {}", path, std::strerror(errno)));
  std::string text;
  char buf[1 << 16];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
  std::fclose(f);
  return text;
}

}  // namespace

Dsv41EngramSidecar dsv41_load_engram_sidecar(const std::string& path, const Dsv41TextConfig& cfg) {
  const std::string text = read_file(path);
  const auto parsed = minijson::parse(text);
  const minijson::Value& root = parsed.root;
  if (!root.is_object()) reject("", "root is not an object");
  if (require_string(root, "format") != "dgpp-dsv41-engram-tables-1")
    reject("format", "unknown sidecar format (regenerate with tools/dsv41_engram_tables.py)");
  Dsv41EngramSidecar s;
  s.tokenizer_sha256 = require_string(root, "tokenizer_sha256");
  s.vocab_size = static_cast<int>(require_int(root, "vocab_size"));
  s.compressed_vocab_size = static_cast<int>(require_int(root, "compressed_vocab_size"));
  s.pad_id = require_int(root, "pad_id");
  s.pad_class = static_cast<int32_t>(require_int(root, "pad_class"));
  for (int64_t v : require_flat(root, "layer_ids")) s.layer_ids.push_back(static_cast<int>(v));
  s.max_ngram_size = static_cast<int>(require_int(root, "max_ngram_size"));
  s.n_heads = static_cast<int>(require_int(root, "n_heads"));
  s.num_embeddings = require_flat(root, "num_embeddings");
  s.primes = require_flat(root, "primes");
  s.offsets = require_flat(root, "offsets");
  s.multipliers = require_flat(root, "multipliers");
  for (int64_t v : require_flat(root, "token_map")) s.token_map.push_back(static_cast<int32_t>(v));

  // Against the config.
  if (s.vocab_size != cfg.vocab_size) reject("vocab_size", "disagrees with the config");
  if (s.compressed_vocab_size != cfg.engram_compressed_vocab_size)
    reject("compressed_vocab_size", "disagrees with the config's engram_compressed_vocab_size");
  if (s.pad_id != cfg.engram_pad_token_id) reject("pad_id", "disagrees with the config's engram_pad_token_id");
  if (s.layer_ids != cfg.engram_layer_ids) reject("layer_ids", "disagree with the config's engram_layer_ids");
  if (s.max_ngram_size != cfg.engram_max_ngram_size) reject("max_ngram_size", "disagrees with the config");
  if (s.n_heads != cfg.engram_n_heads) reject("n_heads", "disagrees with the config");
  if (s.num_embeddings != cfg.engram_num_embeddings) reject("num_embeddings", "disagree with the config");
  const size_t L = s.layer_ids.size(), G = static_cast<size_t>(s.ngrams()), H = static_cast<size_t>(s.n_heads);
  if (s.primes.size() != L * G * H) reject("primes", "wrong count for layers x n-grams x heads");
  if (s.offsets.size() != L * G * H) reject("offsets", "wrong count for layers x n-grams x heads");
  if (s.multipliers.size() != L * static_cast<size_t>(s.max_ngram_size)) reject("multipliers", "wrong count");
  if (static_cast<int>(s.token_map.size()) != s.vocab_size) reject("token_map", "wrong length");
  for (size_t l = 0; l < L; ++l) {
    int64_t running = 0;
    for (size_t g = 0; g < G; ++g)
      for (size_t h = 0; h < H; ++h) {
        const size_t at = (l * G + g) * H + h;
        if (s.primes[at] <= 1) reject("primes", "a bucket modulus below 2");
        if (s.offsets[at] != running) reject("offsets", "not the running sum of the primes");
        running += s.primes[at];
      }
    if (running != s.num_embeddings[l]) reject("primes", "the bucket ranges do not sum to the table's rows");
  }
  for (int64_t m : s.multipliers)
    if (m <= 0 || (m & 1) == 0) reject("multipliers", "must be positive odd integers");
  for (int32_t c : s.token_map)
    if (c < 0 || c >= s.compressed_vocab_size) reject("token_map", "a class outside [0, compressed_vocab_size)");
  if (s.pad_class != s.token_map[static_cast<size_t>(s.pad_id)]) reject("pad_class", "is not token_map[pad_id]");
  return s;
}

Dsv41EngramSidecar dsv41_load_engram_sidecar_for(const std::string& checkpoint_dir,
                                                 const Dsv41TextConfig& cfg, const std::string& fallback_dir) {
  const std::filesystem::path p = std::filesystem::path(checkpoint_dir) / kDsv41EngramSidecarName;
  if (std::filesystem::exists(p)) return dsv41_load_engram_sidecar(p.string(), cfg);
  // An EXL3 pack ships the base release's tokenizer and no sidecar: the
  // Engram source snapshot's (engine.engram_source) serves it.
  if (!fallback_dir.empty()) {
    const std::filesystem::path q = std::filesystem::path(fallback_dir) / kDsv41EngramSidecarName;
    if (std::filesystem::exists(q)) return dsv41_load_engram_sidecar(q.string(), cfg);
  }
  throw std::runtime_error(
      "engram sidecar missing: " + p.string() +
      (fallback_dir.empty() ? std::string() : " (and in the Engram source " + fallback_dir + ")") +
      " (generate it with `python3 tools/dsv41_engram_tables.py --model-dir <snapshot>`)");
}

}  // namespace dgpp
