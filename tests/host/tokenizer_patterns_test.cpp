#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "text/tokenizer.hpp"

namespace {
std::filesystem::path data;

DGPP_TEST(tokenizer_dory_and_flint_keep_distinct_split_rules) {
  // Tiny vocabularies containing whole pretokens and their split pieces
  // isolate scanner dispatch from BPE. The checkpoint patterns differ on
  // CJK prefixes, digit clumping and contractions despite sharing case classes.
  for (size_t cache : {size_t{0}, size_t{128}}) {
    const auto dory = dgpp::text::Tokenizer::load((data / "dory_tokenizer_pattern.json").string(), cache);
    const auto flint = dgpp::text::Tokenizer::load((data / "flint_tokenizer_pattern.json").string(), cache);
    for (int repeat = 0; repeat < 2; ++repeat) {
      const auto check = [](const auto& tok, const std::string& text, const std::vector<int64_t>& ids) {
        if (tok.encode(text) != ids || tok.decode(ids, false) != text)
          throw std::runtime_error("tokenizer pattern mismatch for " + text);
      };
      check(dory, " 東京", {2});
      check(flint, " 東京", {0, 1});
      check(dory, "12", {3, 4});
      check(flint, "12", {5});
      check(dory, "a's", {6, 7});
      check(flint, "a's", {8});
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) throw std::invalid_argument("usage: tokenizer_patterns_test DATA_DIR");
  data = argv[1];
  return dgpp::test::run_all();
}
