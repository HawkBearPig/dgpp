#include "text/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "common/log.hpp"
#include "loaders/minijson.hpp"
#include "text/bpe.hpp"
#include "text/unicode_normalize.hpp"
#include "text/unicode_tables.hpp"

namespace dgpp::text {
namespace {

// The pinned Split patterns (tokenizer.json pre_tokenizer[0]). The scanner
// below hardcodes exactly these regexes; any other pattern is refused at
// load.
constexpr const char* kSplitPattern =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
// The Qwen3.8-Flash-Next pattern: letter runs admit marks, numbers split
// one per pretoken, marks stay out of the punctuation class.
constexpr const char* kSplitPatternQwen =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}"
    "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
// The Qwen2 pattern (MiMo-V2.6-Flash, 2026-09-22): the GLM classes (letter
// runs without marks, marks in the punctuation class) with numbers one per
// pretoken as the Qwen3.8 pattern splits them.
constexpr const char* kSplitPatternQwen2 =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
    "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
// The DeepSeek-V4.1 pre-tokenizer: THREE Split stages (each Isolated,
// applied to the previous stage's pieces) then ByteLevel. Stage 1 cuts
// number runs into pieces of at most three; stage 2 isolates CJK runs
// (the three literal ranges); stage 3's alternatives, leftmost-first:
//   B1 [ASCII punct][A-Za-z]+           one ASCII punctuation/symbol char + ASCII letters
//   B2 [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+   optional prefix (whitespace, format
//                                      or control chars — numbers never
//                                      reach this stage) + a letter/mark run
//   B3  ?[\p{P}\p{S}]+[\r\n]*           optional SPACE + a punctuation/symbol
//                                      run + trailing CR/LFs
//   B4-B6 the whitespace alternatives of the GLM pattern (A5-A7)
// Text no alternative matches (format and control characters outside a
// letter run's prefix) stays a piece of its own: Split's Isolated
// behaviour keeps the gaps between matches.
constexpr const char* kSplitPatternDsv41Numbers = "\\p{N}{1,3}";
// The three literal CJK ranges (raw codepoints in the file, decoded by the JSON reader).
constexpr const char* kSplitPatternDsv41Cjk = "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+";
// CR and LF stand in the file as the control characters themselves.
constexpr const char* kSplitPatternDsv41Main =
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

// The Flint pre-tokenizer (2026-10-22, docs/flint_plan.md §1.6): the
// DeepSeek-V4.1 three-stage shape (number runs of at most three, the CJK
// ranges) with stage 3's main pattern the o200k form over the case
// classes, leftmost-first:
//   C1 [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//                               an optional prefix, an "upper" run (Lu, Lt, Lm, Lo, M), a
//                               "lower" run (Ll, Lm, Lo, M) of at least one, an optional
//                               contraction suffix
//   C2 [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:...)?
//                               an upper run of at least one then a lower run
//   C3 \ ?[\p{P}\p{S}]+[\r\n]*     the DeepSeek B3
//   C4 \s*[\r\n]+                 B4;  C5 \s+(?!\S);  C6 \s+
// The classes overlap (Lm, Lo and the marks sit in both runs): C1's
// greedy upper run backtracks until a lower-class character opens the
// lower run, so a word of upper-only characters falls to C2 and a word of
// shared-class characters (CJK letters outside the stage-2 ranges, Lo
// scripts) matches C1 whole. The contraction suffix is matched
// case-insensitively and only after a letter run.
constexpr const char* kSplitPatternFlintMain =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    " ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

[[noreturn]] void reject(const std::string& what) {
  throw std::runtime_error("glm_tokenizer: " + what);
}

// Null-safe field access for the shape validator: a missing field is a
// different tokenizer, not a default to guess.
const minijson::Value& field(const minijson::Value& v, const char* name,
                            const char* where) {
  const minijson::Value* f = v.find(name);
  if (!f) reject(std::string(where) + ": missing field '" + name + "'");
  return *f;
}

std::string_view want_string(const minijson::Value& v, const char* name,
                             const char* where) {
  const minijson::Value& f = field(v, name, where);
  if (!f.is_string()) reject(std::string(where) + ": '" + name + "' not a string");
  return f.as_string();
}

bool want_bool(const minijson::Value& v, const char* name, const char* where) {
  const minijson::Value& f = field(v, name, where);
  if (f.kind() != minijson::Value::Kind::Bool)
    reject(std::string(where) + ": '" + name + "' not a bool");
  return f.as_bool();
}

uint64_t fnv1a_bytes(const char* data, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<unsigned char>(data[i]);
    h *= 1099511628211ull;
  }
  return h;
}

std::string read_file(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) reject("cannot open " + path);
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string buf(static_cast<size_t>(n), '\0');
  if (n > 0 && std::fread(buf.data(), 1, static_cast<size_t>(n), f) !=
                  static_cast<size_t>(n)) {
    std::fclose(f);
    reject("short read on " + path);
  }
  std::fclose(f);
  return buf;
}

// GPT-2's bytes_to_unicode: printable ASCII and the 0xA1-0xFF stretch map
// to themselves; every other byte maps to 256+n over the remaining
// codepoints. All mapped codepoints stay below 0x200.
void build_byte_level_maps(uint32_t* byte_to_cp, int32_t* cp_to_byte) {
  std::fill(cp_to_byte, cp_to_byte + 0x200, -1);
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    const bool direct = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                        (b >= 0xAE && b <= 0xFF);
    const uint32_t cp = direct ? static_cast<uint32_t>(b)
                               : static_cast<uint32_t>(256 + n++);
    byte_to_cp[b] = cp;
    cp_to_byte[cp] = b;
  }
}

// ---------------------------------------------------------------------------
// The Split scanner. Operates on CODEPOINTS (the regex is Unicode-class
// based) over one added-token-free segment. Successive matches emit the
// pretokens (the pattern covers every codepoint, so matches are
// gap-free; the defensive throw pins a scanner hole loudly instead of
// silently re-syncing).
//
// Alternation is leftmost-first (Rust regex semantics) — the scanner tries
// the alternatives in the pattern's order, with the greedy/backtracking
// outcomes each quantifier implies:
//   A1 (?i:'s|'t|'re|'ve|'m|'ll|'d)   apostrophe + case-insens form
//   A2 [^\r\n\p{L}\p{N}]?\p{L}+      optional non-CR/LF/letter/number
//                                    prefix + a letter run (also the
//                                    plain letter run when the prefix is
//                                    absent)
//   A3 \p{N}{1,3}                    up to 3 numbers
//   A4  ?[^\s\p{L}\p{N}]+[\r\n]*     optional literal SPACE + a punctuation
//                                    run + trailing CR/LFs
//   A5 \s*[\r\n]+                    whitespace through the run's last CR/LF
//   A6 \s+(?!\S)                     whitespace minus its last char when a
//                                    non-space follows (that char joins the
//                                    next match — the classic " a" space
//                                    prefix); the full run at end of text
//   A7 \s+                           the remaining whitespace run
// ---------------------------------------------------------------------------
// `qwen` selects the Qwen3.8 pattern's three differences (A2 admits marks
// in the run, A3 is one number, A4's class excludes marks).
class SplitScanner {
 public:
  // marks: the letter-run class admits \p{M} and the punctuation class
  // excludes it (the Qwen3.8 pattern); single_number: one \p{N} per
  // pretoken (Qwen3.8, Qwen2) instead of runs of up to three (GLM).
  SplitScanner(std::string_view text, std::vector<std::string_view>* out, bool marks, bool single_number)
      : text_(text), out_(out), marks_(marks), single_number_(single_number) {
    // Decode codepoint boundaries once. Strict UTF-8: a chat prompt
    // arrives as valid UTF-8; anything else is a caller bug, not a
    // tokenization ambiguity.
    size_t i = 0;
    while (i < text_.size()) {
      const int len = utf8_len(text_[i]);
      if (len < 1 || i + static_cast<size_t>(len) > text_.size())
        reject("input is not valid UTF-8");
      for (int k = 1; k < len; ++k) {
        const unsigned char cont =
            static_cast<unsigned char>(text_[i + k]);
        if ((cont & 0xC0) != 0x80)
          reject("input is not valid UTF-8 (bad continuation byte)");
      }
      uint32_t raw = 0;
      for (int k = 0; k < len; ++k)
        raw = (raw << 8) | static_cast<unsigned char>(text_[i + k]);
      cps_.push_back(decode_utf8(raw, len));
      starts_.push_back(i);
      i += static_cast<size_t>(len);
    }
    starts_.push_back(text_.size());
  }

  void run() {
    size_t i = 0;  // codepoint index
    while (i < cps_.size()) {
      const size_t end = match_at(i);
      if (end == i)
        reject("scanner hole at codepoint " + std::to_string(i));
      out_->push_back(text_.substr(starts_[i], starts_[end] - starts_[i]));
      i = end;
    }
  }

 private:
  static int utf8_len(char c) {
    const unsigned char b = static_cast<unsigned char>(c);
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return -1;
  }
  static uint32_t decode_utf8(uint32_t raw, int len) {
    switch (len) {
      case 1: return raw & 0x7F;
      case 2: return ((raw >> 8) & 0x1F) << 6 | (raw & 0x3F);
      case 3: return ((raw >> 16) & 0x0F) << 12 | ((raw >> 8) & 0x3F) << 6 |
                   (raw & 0x3F);
      default: return ((raw >> 24) & 0x07) << 18 | ((raw >> 16) & 0x3F) << 12 |
                    ((raw >> 8) & 0x3F) << 6 | (raw & 0x3F);
    }
  }
  static uint32_t lower_ascii(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp;
  }
  bool letter(size_t i) const { return unicode::is_letter(cps_[i]); }
  bool mark(size_t i) const { return marks_ && unicode::is_mark(cps_[i]); }
  // The letter-run class: \p{L} (GLM) or [\p{L}\p{M}] (Qwen).
  bool run_char(size_t i) const { return letter(i) || mark(i); }
  bool number(size_t i) const { return unicode::is_number(cps_[i]); }
  bool ws(size_t i) const { return unicode::is_white_space(cps_[i]); }
  bool punct(size_t i) const { return !ws(i) && !letter(i) && !number(i) && !mark(i); }
  bool eos(size_t i) const { return i >= cps_.size(); }
  uint32_t cp(size_t i) const { return eos(i) ? 0 : cps_[i]; }

  // A1: apostrophe + case-insensitive s|t|m|d (1) or re|ve|ll (2).
  bool match_contraction(size_t i, size_t* end) const {
    if (cp(i) != '\'') return false;
    const uint32_t a = lower_ascii(cp(i + 1));
    if (a == 's' || a == 't' || a == 'm' || a == 'd') {
      *end = i + 2;
      return true;
    }
    const uint32_t b = lower_ascii(cp(i + 2));
    if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
        (a == 'l' && b == 'l')) {
      *end = i + 3;
      return true;
    }
    return false;
  }

  // A5: \s*[\r\n]+ — through the whitespace run's LAST CR/LF.
  bool match_newline_run(size_t i, size_t* end) const {
    size_t j = i;
    while (!eos(j) && ws(j)) ++j;
    size_t last_nl = i;
    for (size_t k = i; k < j; ++k)
      if (cps_[k] == '\r' || cps_[k] == '\n') last_nl = k + 1;
    if (last_nl == i) return false;
    *end = last_nl;
    return true;
  }

  // A6: \s+(?!\S) — the whole run at end of text; one char short when a
  // non-space follows (needs a run of >= 2 then).
  bool match_trailing_ws(size_t i, size_t* end) const {
    size_t j = i;
    while (!eos(j) && ws(j)) ++j;
    if (j == i) return false;
    if (eos(j)) {
      *end = j;
      return true;
    }
    if (j - i >= 2) {
      *end = j - 1;
      return true;
    }
    return false;
  }

  // Returns the codepoint index one PAST the match at i (never i).
  size_t match_at(size_t i) const {
    size_t end = 0;
    if (match_contraction(i, &end)) return end;

    // A2: optional prefix + letter run (Qwen: a letter-or-mark run).
    if (!eos(i) && run_char(i)) {
      size_t j = i + 1;
      while (!eos(j) && run_char(j)) ++j;
      return j;
    }
    if (!eos(i) && !letter(i) && !number(i) && cp(i) != '\r' && cp(i) != '\n' &&
        !eos(i + 1) && run_char(i + 1)) {
      size_t j = i + 2;
      while (!eos(j) && run_char(j)) ++j;
      return j;
    }

    // A3: 1..3 numbers (Qwen3.8, Qwen2: exactly one).
    if (!eos(i) && number(i)) {
      size_t j = i + 1;
      if (!single_number_)
        while (j < i + 3 && !eos(j) && number(j)) ++j;
      return j;
    }

    // A4: optional literal space + punct run + CR/LF tail.
    {
      size_t j = i;
      if (cp(i) == ' ' && !eos(i + 1) && punct(i + 1)) j = i + 1;
      if (!eos(j) && punct(j)) {
        while (!eos(j) && punct(j)) ++j;
        while (!eos(j) && (cps_[j] == '\r' || cps_[j] == '\n')) ++j;
        return j;
      }
    }

    // A5-A7: whitespace (order matters — newline runs first).
    if (match_newline_run(i, &end)) return end;
    if (match_trailing_ws(i, &end)) return end;
    size_t j = i;
    while (!eos(j) && ws(j)) ++j;
    return j;  // A7 (i is whitespace, so j > i)
  }

  std::string_view text_;
  std::vector<std::string_view>* out_;
  bool marks_ = false;
  bool single_number_ = false;
  std::vector<uint32_t> cps_;
  std::vector<size_t> starts_;  // size = cps+1 (end sentinel)
};

// The DeepSeek-V4.1 scanner (kSplitPatternDsv41*): the three stages over
// codepoints. Every stage's pieces bound the next stage's matches (HF's
// Sequence runs each pre-tokenizer on the splits the previous one left,
// so the lookahead of \s+(?!\S) sees a piece's end as the text's end).
class Dsv41Scanner {
 public:
  // flint: stage 3 runs the Flint main pattern (kSplitPatternFlintMain)
  // instead of the DeepSeek-V4.1 one.
  Dsv41Scanner(std::string_view text, std::vector<std::string_view>* out, bool flint = false)
      : text_(text), out_(out), flint_(flint) {
    size_t i = 0;
    while (i < text_.size()) {
      const int len = utf8_len(text_[i]);
      if (len < 1 || i + static_cast<size_t>(len) > text_.size()) reject("input is not valid UTF-8");
      for (int k = 1; k < len; ++k) {
        const unsigned char cont = static_cast<unsigned char>(text_[i + k]);
        if ((cont & 0xC0) != 0x80) reject("input is not valid UTF-8 (bad continuation byte)");
      }
      uint32_t raw = 0;
      for (int k = 0; k < len; ++k) raw = (raw << 8) | static_cast<unsigned char>(text_[i + k]);
      cps_.push_back(decode_utf8(raw, len));
      starts_.push_back(i);
      i += static_cast<size_t>(len);
    }
    starts_.push_back(text_.size());
  }

  void run() {
    // Stage 1: number runs in pieces of at most three; the spans between
    // go on to stage 2.
    const size_t n = cps_.size();
    size_t i = 0;
    while (i < n) {
      if (number(i)) {
        size_t j = i;
        while (j < n && j - i < 3 && number(j)) ++j;
        emit(i, j);
        i = j;
        continue;
      }
      size_t e = i;
      while (e < n && !number(e)) ++e;
      stage2(i, e);
      i = e;
    }
  }

 private:
  static int utf8_len(char c) {
    const unsigned char b = static_cast<unsigned char>(c);
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return -1;
  }
  static uint32_t decode_utf8(uint32_t raw, int len) {
    switch (len) {
      case 1: return raw & 0x7F;
      case 2: return ((raw >> 8) & 0x1F) << 6 | (raw & 0x3F);
      case 3: return ((raw >> 16) & 0x0F) << 12 | ((raw >> 8) & 0x3F) << 6 | (raw & 0x3F);
      default: return ((raw >> 24) & 0x07) << 18 | ((raw >> 16) & 0x3F) << 12 | ((raw >> 8) & 0x3F) << 6 | (raw & 0x3F);
    }
  }
  void emit(size_t a, size_t b) {
    if (b > a) out_->push_back(text_.substr(starts_[a], starts_[b] - starts_[a]));
  }
  bool number(size_t i) const { return unicode::is_number(cps_[i]); }
  bool cjk(size_t i) const {
    const uint32_t c = cps_[i];
    return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
  }
  bool letter(size_t i) const { return unicode::is_letter(cps_[i]); }
  bool mark(size_t i) const { return unicode::is_mark(cps_[i]); }
  bool run_char(size_t i) const { return letter(i) || mark(i); }
  bool ps(size_t i) const { return unicode::is_punctuation(cps_[i]) || unicode::is_symbol(cps_[i]); }
  bool ws(size_t i) const { return unicode::is_white_space(cps_[i]); }
  bool crlf(size_t i) const { return cps_[i] == '\r' || cps_[i] == '\n'; }
  static bool ascii_punct(uint32_t c) {
    return (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40) || (c >= 0x5B && c <= 0x60) || (c >= 0x7B && c <= 0x7E);
  }
  static bool ascii_letter(uint32_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }

  // Stage 2 over [a, b): CJK runs are pieces; the spans between go on to
  // stage 3.
  void stage2(size_t a, size_t b) {
    size_t k = a;
    while (k < b) {
      size_t j = k;
      if (cjk(k)) {
        while (j < b && cjk(j)) ++j;
        emit(k, j);
      } else {
        while (j < b && !cjk(j)) ++j;
        stage3(k, j);
      }
      k = j;
    }
  }

  // Stage 3 over [a, b): the main pattern's matches, the gaps between
  // them pieces of their own.
  void stage3(size_t a, size_t b) {
    size_t k = a;
    size_t gap = b;  // b: no open gap
    while (k < b) {
      const size_t end = match3(k, b);
      if (end == k) {
        if (gap == b) gap = k;
        ++k;
        continue;
      }
      if (gap != b) {
        emit(gap, k);
        gap = b;
      }
      emit(k, end);
      k = end;
    }
    if (gap != b) emit(gap, b);
  }

  // The Flint classes: the "upper" run (Lu, Lt, Lm, Lo, M) and the "lower"
  // run (Ll, Lm, Lo, M); a character of a letter class outside Lu / Lt /
  // Ll (Lm, Lo) and a mark sit in both.
  bool upper_class(size_t i) const {
    const uint32_t c = cps_[i];
    return unicode::is_mark(c) || (unicode::is_letter(c) && !unicode::is_lower(c));
  }
  bool lower_class(size_t i) const {
    const uint32_t c = cps_[i];
    return unicode::is_mark(c) || (unicode::is_letter(c) && !unicode::is_upper_or_title(c));
  }
  // The optional contraction suffix at j within [j, b): the length matched
  // (0 when none) — (?i:'s|'t|'re|'ve|'m|'ll|'d), the longest alternative
  // first as the regex's leftmost-first order over the alternatives
  // ('re, 've, 'll before 's, 't, 'm, 'd matters only where both could
  // match, which they cannot: the two-letter forms begin with r, v, l).
  size_t contraction(size_t j, size_t b) const {
    if (j >= b || cps_[j] != '\'' || j + 1 >= b) return 0;
    auto lower = [&](size_t i) -> uint32_t {
      const uint32_t c = cps_[i];
      return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    };
    const uint32_t a = lower(j + 1);
    if (j + 2 < b) {
      const uint32_t c2 = lower(j + 2);
      if ((a == 'r' && c2 == 'e') || (a == 'v' && c2 == 'e') || (a == 'l' && c2 == 'l')) return 3;
    }
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    return 0;
  }
  // The Flint stage 3: C1 / C2 (the letter runs with an optional prefix
  // and suffix), then the DeepSeek B3 .. B6 tail. `letter_runs(s)` is the
  // regex's own backtracking over the case classes from a letter-class
  // character s: C1's greedy upper run [s, u) then a lower run of at
  // least one — at u when the character there is lower-only, else backed
  // up to the LAST lower-class (shared) character inside the upper run,
  // whose lower run ends at the upper-only character after it (the regex
  // gives the upper run back one character at a time until the lower run
  // can open); C2 when the upper run holds no lower-class character at
  // all: the run itself. The match is the regex's leftmost-first one, not
  // the longest (a Thai run followed by capitals cuts after the Thai).
  size_t letter_runs(size_t s, size_t b) const {
    size_t u = s;
    while (u < b && upper_class(u)) ++u;
    if (u < b && lower_class(u)) {  // C1: the lower run opens at u
      size_t l = u;
      while (l < b && lower_class(l)) ++l;
      return l + contraction(l, b);
    }
    size_t p = u;  // p - 1: the last lower-class character of the upper run
    while (p > s && !lower_class(p - 1)) --p;
    if (p > s) {  // C1 after the backtrack: the lower run is [p - 1, p)
      size_t l = p - 1;
      while (l < b && lower_class(l)) ++l;
      return l + contraction(l, b);
    }
    if (u > s) return u + contraction(u, b);  // C2: upper-only characters
    return s;
  }
  size_t match3_flint(size_t k, size_t b) const {
    // The optional prefix (any character but CR, LF, a letter or a
    // number) and the runs after it, tried first as the regex does; a
    // mark at k is both a prefix and a run character, and the two paths
    // meet on the same span.
    if (!crlf(k) && !letter(k) && !number(k) && !mark(k) && k + 1 < b) {
      const size_t e = letter_runs(k + 1, b);
      if (e > k + 1) return e;
    }
    if (upper_class(k) || lower_class(k)) {
      const size_t e = letter_runs(k, b);
      if (e > k) return e;
    }
    return match_tail(k, b);
  }

  // The end of the leftmost-first match at k within [k, b), or k when no
  // alternative matches.
  size_t match3(size_t k, size_t b) const {
    if (flint_) return match3_flint(k, b);
    // B1: one ASCII punctuation/symbol character + ASCII letters.
    if (ascii_punct(cps_[k]) && k + 1 < b && ascii_letter(cps_[k + 1])) {
      size_t j = k + 2;
      while (j < b && ascii_letter(cps_[j])) ++j;
      return j;
    }
    // B2: optional prefix + a letter/mark run.
    if (run_char(k)) {
      size_t j = k + 1;
      while (j < b && run_char(j)) ++j;
      return j;
    }
    if (!crlf(k) && !letter(k) && !ps(k) && k + 1 < b && run_char(k + 1)) {
      size_t j = k + 2;
      while (j < b && run_char(j)) ++j;
      return j;
    }
    return match_tail(k, b);
  }

  // B3 .. B6 (shared by both main patterns): the punctuation / symbol run
  // with its CR/LFs, the whitespace alternatives.
  size_t match_tail(size_t k, size_t b) const {
    // B3: optional literal space + a punctuation/symbol run + CR/LFs.
    {
      size_t j = k;
      if (cps_[k] == ' ' && k + 1 < b && ps(k + 1)) j = k + 1;
      if (j < b && ps(j)) {
        while (j < b && ps(j)) ++j;
        while (j < b && crlf(j)) ++j;
        return j;
      }
    }
    // B4: \s*[\r\n]+ — through the whitespace run's last CR/LF.
    {
      size_t j = k;
      while (j < b && ws(j)) ++j;
      size_t last_nl = k;
      for (size_t x = k; x < j; ++x)
        if (crlf(x)) last_nl = x + 1;
      if (last_nl > k) return last_nl;
      // B5: \s+(?!\S) — the whole run at the piece's end, one short before
      // a non-space (a run of >= 2 then); B6: \s+ — the run.
      if (j > k) {
        if (j == b) return j;
        if (j - k >= 2) return j - 1;
        return j;
      }
    }
    return k;
  }

  std::string_view text_;
  std::vector<std::string_view>* out_;
  bool flint_ = false;
  std::vector<uint32_t> cps_;
  std::vector<size_t> starts_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Load + validate.
// ---------------------------------------------------------------------------
Tokenizer Tokenizer::load(const std::string& path, size_t word_cache_entries) {
  Tokenizer t;
  if (word_cache_entries) t.word_cache_ = std::make_unique<detail::WordCache>(word_cache_entries);
  t.json_buf_ = read_file(path);
  t.revision_hash_ = fnv1a_bytes(t.json_buf_.data(), t.json_buf_.size());
  build_byte_level_maps(t.byte_to_cp_, t.cp_to_byte_);

  // The parse tree becomes a MEMBER: object keys (Member.key) are
  // tree-owned std::strings, so every view taken below stays valid for
  // the tokenizer's lifetime (json_buf_ is declared before doc_, and
  // both moves preserve their heap addresses).
  t.doc_ = minijson::parse(t.json_buf_);
  const minijson::Value& root = t.doc_.root;

  // --- normalizer: none, or NFC ----------------------------------------
  if (const minijson::Value* n = root.find("normalizer")) {
    if (!n->is_null()) {
      const minijson::Value* ty = n->find("type");
      if (ty && ty->is_string() && ty->as_string() == "NFC") {
        t.nfc_ = true;
      } else if (ty && ty->is_string() && ty->as_string() == "Sequence") {
        // DeepSeek-V4.1: an empty Sequence — no normalizer.
        const minijson::Value* list = n->find("normalizers");
        if (!list || !list->is_array() || !list->items().empty())
          reject("normalizer Sequence is not empty (only null, NFC and the empty Sequence are implemented)");
      } else {
        reject("normalizer is neither null, NFC nor the empty Sequence (the three implemented)");
      }
    }
  }

  // --- pre_tokenizer: the pinned Sequence ------------------------------
  const minijson::Value* pre = root.find("pre_tokenizer");
  if (!pre || !pre->is_object() ||
      want_string(*pre, "type", "pre_tokenizer") != "Sequence")
    reject("pre_tokenizer is not the pinned Sequence");
  const auto& pres = field(*pre, "pretokenizers", "pre_tokenizer").items();
  const size_t splits = pres.size() >= 1 ? pres.size() - 1 : 0;
  if ((splits != 1 && splits != 3) || want_string(pres[splits], "type", "pre_tokenizer[last]") != "ByteLevel")
    reject("pre_tokenizer is neither [Split, ByteLevel] nor [Split x 3, ByteLevel]");
  for (size_t k = 0; k < splits; ++k) {
    if (want_string(pres[k], "type", "pre_tokenizer[k]") != "Split") reject("pre_tokenizer stage is not a Split");
    if (want_string(pres[k], "behavior", "Split") != "Isolated") reject("Split behavior is not Isolated");
    if (const minijson::Value* inv = pres[k].find("invert"); inv && inv->kind() == minijson::Value::Kind::Bool && inv->as_bool())
      reject("Split invert must be false");
  }
  const auto split_regex = [&](size_t k) {
    const minijson::Value& pattern = field(pres[k], "pattern", "Split");
    return want_string(pattern, "Regex", "Split.pattern");
  };
  if (splits == 1) {
    const std::string_view regex = split_regex(0);
    if (regex == kSplitPattern) t.pattern_ = 0;
    else if (regex == kSplitPatternQwen) t.pattern_ = 1;
    else if (regex == kSplitPatternQwen2) t.pattern_ = 3;
    else
      reject("Split pattern differs from the pinned regexes (the scanner "
             "hardcodes them — update the scanner or the checkpoint)");
  } else {
    if (split_regex(0) != kSplitPatternDsv41Numbers || split_regex(1) != kSplitPatternDsv41Cjk)
      reject("the first two Split patterns differ from the pinned three-stage regexes (the scanner "
             "hardcodes them — update the scanner or the checkpoint)");
    if (split_regex(2) == kSplitPatternDsv41Main) t.pattern_ = 2;
    else if (split_regex(2) == kSplitPatternFlintMain) t.pattern_ = 4;
    else
      reject("the third Split pattern differs from the pinned DeepSeek-V4.1 and Flint regexes (the scanner "
             "hardcodes them — update the scanner or the checkpoint)");
  }
  if (want_bool(pres[splits], "use_regex", "ByteLevel"))
    reject("ByteLevel use_regex must be false (map-only)");
  if (want_bool(pres[splits], "add_prefix_space", "ByteLevel"))
    reject("ByteLevel add_prefix_space must be false");

  // --- decoder: ByteLevel (the reverse map) ----------------------------
  const minijson::Value* dec = root.find("decoder");
  if (!dec || !dec->is_object() ||
      want_string(*dec, "type", "decoder") != "ByteLevel")
    reject("decoder is not ByteLevel");

  // --- post_processor: offsets-only shapes -----------------------------
  // ByteLevel (GLM-5.3, Qwen) or a Sequence of ByteLevel processors
  // (GLM-4.7): both touch offsets only, never the ids.
  if (const minijson::Value* pp = root.find("post_processor")) {
    if (!pp->is_null()) {
      const minijson::Value* ty = pp->find("type");
      bool offsets_only = ty && ty->as_string() == "ByteLevel";
      if (ty && ty->as_string() == "Sequence") {
        const minijson::Value* procs = pp->find("processors");
        offsets_only = procs && procs->is_array();
        if (offsets_only)
          for (const auto& proc : procs->items()) {
            const minijson::Value* pt = proc.find("type");
            if (!pt || pt->as_string() != "ByteLevel") offsets_only = false;
          }
      }
      if (!offsets_only)
        reject("post_processor is neither null, ByteLevel nor a Sequence of "
               "ByteLevel (offsets-only) — special-token injection is not "
               "implemented at encode");
    }
  }

  // --- model: BPE with ignore_merges -----------------------------------
  const minijson::Value* model = root.find("model");
  if (!model || want_string(*model, "type", "model") != "BPE")
    reject("model.type is not BPE");
  // ignore_merges: absent (DeepSeek-V4.1) means false.
  if (const minijson::Value* im = model->find("ignore_merges"); im && !im->is_null())
    t.ignore_merges_ = want_bool(*model, "ignore_merges", "model");
  else
    t.ignore_merges_ = false;
  if (want_bool(*model, "byte_fallback", "model"))
    reject("model.byte_fallback must be false");
  if (const minijson::Value* v = model->find("unk_token"); v && !v->is_null())
    reject("model.unk_token must be null");
  // The affixes: null (GLM) or the empty string (Qwen) — both "none".
  for (const char* f : {"continuing_subword_prefix", "end_of_word_suffix"}) {
    const minijson::Value* v = model->find(f);
    if (v && !v->is_null() && !(v->is_string() && v->as_string().empty()))
      reject(std::string("model.") + f + " must be null or empty");
  }

  // --- vocab (dense id coverage validated below) -----------------------
  const auto& vocab = field(*model, "vocab", "model").members();
  t.vocab_.reserve(vocab.size());
  int64_t max_vocab_id = -1;
  for (const auto& [key, val] : vocab) {
    if (!val.is_number()) reject("vocab entry id is not a number");
    const int64_t id = val.as_int();
    if (!t.vocab_.emplace(key, id).second)
      reject("duplicate vocab token: " + key);
    max_vocab_id = std::max(max_vocab_id, id);
  }

  // --- merges (rank = list order) ---------------------------------------
  const auto& merges = field(*model, "merges", "model").items();
  t.merge_rank_.reserve(merges.size());
  for (size_t r = 0; r < merges.size(); ++r) {
    // Two formats: ["first", "second"] (GLM's file) or "first second"
    // (Qwen's — one space; ByteLevel symbols never contain a space).
    std::string_view first, second;
    if (merges[r].is_string()) {
      const std::string_view m = merges[r].as_string();
      const size_t sp = m.find(' ');
      if (sp == std::string_view::npos || sp == 0 || sp + 1 >= m.size() ||
          m.find(' ', sp + 1) != std::string_view::npos)
        reject("malformed merge string at rank " + std::to_string(r));
      first = m.substr(0, sp);
      second = m.substr(sp + 1);
    } else {
      const auto& pair = merges[r].items();
      if (pair.size() != 2 || !pair[0].is_string() || !pair[1].is_string())
        reject("malformed merge entry at rank " + std::to_string(r));
      first = pair[0].as_string();
      second = pair[1].as_string();
    }
    if (!t.merge_rank_.emplace(MergeKey{first, second}, static_cast<uint32_t>(r)).second)
      reject("duplicate merge pair at rank " + std::to_string(r));
  }

  // --- added tokens: plain matching only -------------------------------
  const auto& added = field(root, "added_tokens", "root").items();
  const auto fresh_node = [] {
    TrieNode n;
    for (auto& c : n.child) c = -1;
    return n;
  };
  t.trie_.clear();
  t.trie_.push_back(fresh_node());  // root = trie_[0]
  t.added_tokens_.reserve(added.size());
  for (const auto& at : added) {
    AddedToken a;
    a.id = field(at, "id", "added_tokens").as_int();
    a.content = std::string(want_string(at, "content", "added_tokens"));
    a.special = want_bool(at, "special", "added_tokens");
    for (const char* f : {"single_word", "lstrip", "rstrip"}) {
      if (want_bool(at, f, "added_tokens"))
        reject("added token with " + std::string(f) +
               " set (plain matching only): " + a.content);
    }
    t.added_tokens_.push_back(std::move(a));
    // Insert into the byte trie (leftmost-longest matching). Added
    // tokens are ASCII here, so a match can never start mid-codepoint
    // (UTF-8 continuation bytes are >= 0x80).
    int32_t node = 0;
    for (const char ch : t.added_tokens_.back().content) {
      const auto b = static_cast<unsigned char>(ch);
      int32_t next = t.trie_[static_cast<size_t>(node)].child[b];
      if (next < 0) {
        t.trie_.push_back(fresh_node());
        next = static_cast<int32_t>(t.trie_.size()) - 1;
        t.trie_[static_cast<size_t>(node)].child[b] = next;
      }
      node = next;
    }
    if (t.trie_[static_cast<size_t>(node)].id >= 0)
      reject("duplicate added token content: " + t.added_tokens_.back().content);
    t.trie_[static_cast<size_t>(node)].id = t.added_tokens_.back().id;
    t.max_id_ = std::max(t.max_id_, t.added_tokens_.back().id);
  }

  // --- reverse id map (decode path) ------------------------------------
  t.token_by_id_.assign(static_cast<size_t>(max_vocab_id) + 1, {});
  for (const auto& [key, val] : vocab) {
    const int64_t id = val.as_int();
    if (id < 0) reject("negative vocab id: " + key);
    auto& slot = t.token_by_id_[static_cast<size_t>(id)];
    if (!slot.empty()) reject("duplicate vocab id: " + std::to_string(id));
    slot = key;
  }
  for (size_t id = 0; id < t.token_by_id_.size(); ++id)
    if (t.token_by_id_[id].empty())
      reject("vocab ids are not dense (missing id " + std::to_string(id) +
             " — decode-by-id needs the dense base range)");

  DGPP_LOG_INFO(
      "tokenizer: loaded vocab {} merges {} added {} (revision 0x{:016x}; "
      "{} pattern, {}, ignore_merges {})",
      t.vocab_.size(), t.merge_rank_.size(), t.added_tokens_.size(),
      t.revision_hash_, t.pattern_ == 2 ? "deepseek-v4.1" : t.pattern_ == 4 ? "flint" : t.pattern_ == 1 ? "qwen" : t.pattern_ == 3 ? "qwen2" : "glm", t.nfc_ ? "NFC" : "no normalizer",
      t.ignore_merges_);
  return t;
}

// ---------------------------------------------------------------------------
// Encode.
// ---------------------------------------------------------------------------
std::vector<int64_t> Tokenizer::encode(std::string_view text) const {
  std::vector<int64_t> ids;
  size_t seg_start = 0;
  size_t i = 0;
  while (i < text.size()) {
    // Leftmost-longest added-token walk at this byte.
    int32_t node = 0;
    int64_t best_id = -1;
    size_t best_len = 0;
    for (size_t k = i; k < text.size(); ++k) {
      const auto b = static_cast<unsigned char>(text[k]);
      node = trie_[static_cast<size_t>(node)].child[b];
      if (node < 0) break;
      if (trie_[static_cast<size_t>(node)].id >= 0) {
        best_id = trie_[static_cast<size_t>(node)].id;
        best_len = k - i + 1;
      }
    }
    if (best_id >= 0) {
      if (i > seg_start)
        encode_segment(text.substr(seg_start, i - seg_start), &ids);
      ids.push_back(best_id);
      i += best_len;
      seg_start = i;
    } else {
      ++i;
    }
  }
  if (seg_start < text.size())
    encode_segment(text.substr(seg_start), &ids);
  return ids;
}

void Tokenizer::encode_segment(std::string_view segment,
                                  std::vector<int64_t>* out) const {
  // The normalizer runs on each added-token-free segment (HF applies it
  // to the splits the added vocabulary leaves, the tokens themselves
  // being non-normalized).
  std::string normalized;
  if (nfc_ && !unicode::nfc_quick_yes(segment)) {
    normalized = unicode::nfc(segment);
    segment = normalized;
  }
  std::vector<std::string_view> pretokens;
  if (pattern_ == 2 || pattern_ == 4) {
    Dsv41Scanner scanner(segment, &pretokens, /*flint=*/pattern_ == 4);
    scanner.run();
  } else {
    SplitScanner scanner(segment, &pretokens, /*marks=*/pattern_ == 1, /*single_number=*/pattern_ == 1 || pattern_ == 3);
    scanner.run();
  }
  for (const std::string_view p : pretokens) {
    // ByteLevel map: each BYTE -> its alphabet codepoint (UTF-8; every
    // mapped codepoint is < 0x200 so <= 3 bytes).
    std::string mapped;
    mapped.reserve(p.size() * 3);
    for (size_t b = 0; b < p.size(); ++b) {
      const uint32_t cp = byte_to_cp_[static_cast<unsigned char>(p[b])];
      if (cp < 0x80) {
        mapped.push_back(static_cast<char>(cp));
      } else if (cp < 0x800) {
        mapped.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        mapped.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      } else {
        mapped.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        mapped.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        mapped.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      }
    }
    encode_word(mapped, out);
  }
}

void Tokenizer::encode_word(std::string_view mapped,
                               std::vector<int64_t>* out) const {
  // ignore_merges: a whole-word vocab hit skips the merge loop entirely
  // (GLM); with it off the merges always run (Qwen).
  if (ignore_merges_) {
    if (const auto it = vocab_.find(mapped); it != vocab_.end()) {
      out->push_back(it->second);
      return;
    }
  }
  if (word_cache_) {
    word_cache_->append(mapped, out, [&] { merge_word(mapped, out); });
  } else {
    merge_word(mapped, out);
  }
}

void Tokenizer::merge_word(std::string_view mapped, std::vector<int64_t>* out) const {
  // Adjacent spans remain in `mapped` throughout both merge algorithms.
  std::vector<std::string_view> symbols;
  symbols.reserve(mapped.size());
  for (size_t i = 0; i < mapped.size();) {
    const unsigned char b = static_cast<unsigned char>(mapped[i]);
    const size_t len = b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : 3;
    symbols.push_back(mapped.substr(i, len));
    i += len;
  }
  detail::bpe_merge(symbols, [&](std::string_view first, std::string_view second) {
    const auto it = merge_rank_.find(MergeKey{first, second});
    return it == merge_rank_.end() ? UINT32_MAX : it->second;
  });
  for (const std::string_view s : symbols) {
    const auto it = vocab_.find(s);
    if (it == vocab_.end())
      reject("BPE symbol missing from vocab (byte-level coverage broken): " +
             std::string(s));
    out->push_back(it->second);
  }
}

// ---------------------------------------------------------------------------
// Decode.
// ---------------------------------------------------------------------------
namespace {
// Verbatim decode of one id (added content or the ByteLevel-reversed
// vocab symbol). Throws on ids outside the vocab.
std::string decode_verbatim(const Tokenizer& t, int64_t id) {
  // Added tokens first (disjoint from the base range).
  for (const auto& a : t.added_tokens())
    if (a.id == id) return a.content;
  if (id < 0 || id >= static_cast<int64_t>(t.token_by_id_size()) ||
      t.token_by_id(static_cast<size_t>(id)).empty())
    throw std::runtime_error("decode: id outside the vocab: " +
                             std::to_string(id));
  const std::string_view sv = t.token_by_id(static_cast<size_t>(id));
  std::string out;
  for (size_t i = 0; i < sv.size();) {
    const unsigned char b = static_cast<unsigned char>(sv[i]);
    uint32_t cp;
    size_t len;
    if (b < 0x80) {
      cp = b;
      len = 1;
    } else if ((b & 0xE0) == 0xC0) {
      cp = ((b & 0x1F) << 6) |
           (static_cast<unsigned char>(sv[i + 1]) & 0x3F);
      len = 2;
    } else {
      cp = ((b & 0x0F) << 12) |
           ((static_cast<unsigned char>(sv[i + 1]) & 0x3F) << 6) |
           (static_cast<unsigned char>(sv[i + 2]) & 0x3F);
      len = 3;
    }
    if (cp >= 0x200 || t.cp_to_byte_at(cp) < 0)
      throw std::runtime_error(
          "decode: symbol codepoint outside the byte-level alphabet");
    out.push_back(static_cast<char>(t.cp_to_byte_at(cp)));
    i += len;
  }
  return out;
}
}  // namespace

std::string Tokenizer::decode(int64_t id, bool skip_special_tokens) const {
  if (skip_special_tokens) {
    for (const AddedToken& a : added_tokens_)
      if (a.id == id && a.special) return {};
  }
  return decode_verbatim(*this, id);
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids,
                                 bool skip_special_tokens) const {
  std::string out;
  for (const int64_t id : ids) out += decode(id, skip_special_tokens);
  return out;
}

}  // namespace dgpp::text
