#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <sstream>
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
DGPP_TEST(tokenizer_load_refuses_missing_file_and_non_pinned_split_pattern) {
  const auto require_refusal = [](const std::filesystem::path& file, const std::string& reason) {
    try {
      (void)dgpp::text::Tokenizer::load(file.string());
    } catch (const std::runtime_error& e) {
      if (std::string(e.what()).find(reason) != std::string::npos) return;
      throw;
    }
    throw std::runtime_error("expected tokenizer loader to refuse " + file.string());
  };
  require_refusal(data / "missing-tokenizer.json", "cannot open");
  require_refusal(data / "tokenizer_wrong_split_pattern.json", "pinned regex");
}

DGPP_TEST(tokenizer_one_stage_case_pattern_accepts_both_split_forms) {
  // The one-stage case pattern matches every codepoint, so a text has no
  // gaps between matches. "Isolated" (Inkling 2) and "Removed" with invert
  // (MiniMax) therefore cut the same pieces. The other two combinations
  // would keep the gaps instead of the matches and are refused.
  std::ifstream in(data / "inkling2_tokenizer_pattern.json");
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string isolated = buffer.str();
  const auto variant = [&](const std::string& behavior, bool invert) {
    std::string text = isolated;
    const auto swap = [&](const std::string& from, const std::string& to) {
      const size_t at = text.find(from);
      if (at == std::string::npos || text.find(from, at + 1) != std::string::npos)
        throw std::runtime_error("the fixture states '" + from + "' exactly once");
      text.replace(at, from.size(), to);
    };
    swap("\"behavior\": \"Isolated\"", "\"behavior\": \"" + behavior + "\"");
    swap("\"invert\": false", std::string("\"invert\": ") + (invert ? "true" : "false"));
    return text;
  };
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("dgpp_tokenizer_patterns_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  const auto write = [&](const std::string& name, const std::string& text) {
    std::ofstream(dir / name) << text;
    return (dir / name).string();
  };
  const auto refused = [](const std::string& file) {
    try {
      (void)dgpp::text::Tokenizer::load(file);
    } catch (const std::runtime_error& e) {
      return std::string(e.what()).find("Removed with invert=true or Isolated with invert=false") != std::string::npos;
    }
    return false;
  };
  bool ok = true;
  std::string why;
  try {
    for (const auto& file : {write("isolated.json", variant("Isolated", false)), write("removed.json", variant("Removed", true))}) {
      for (size_t cache : {size_t{0}, size_t{128}}) {
        const auto tok = dgpp::text::Tokenizer::load(file, cache);
        // A contraction stays with its word and a digit run is cut in threes.
        if (tok.encode("a's") != std::vector<int64_t>{0} || tok.encode("1234") != std::vector<int64_t>{1, 2} ||
            tok.decode({0, 1, 2}, false) != "a's1234") {
          ok = false;
          why = "the split forms disagree in " + file;
        }
      }
    }
    if (!refused(write("removed_plain.json", variant("Removed", false)))) ok = false, why = "Removed without invert was loaded";
    if (!refused(write("isolated_inverted.json", variant("Isolated", true)))) ok = false, why = "Isolated with invert was loaded";
  } catch (...) {
    std::filesystem::remove_all(dir);
    throw;
  }
  std::filesystem::remove_all(dir);
  if (!ok) throw std::runtime_error(why);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) throw std::invalid_argument("usage: tokenizer_patterns_test DATA_DIR");
  data = argv[1];
  return dgpp::test::run_all();
}
