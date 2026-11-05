#include "loaders/minijson.hpp"

namespace dgpp::minijson {

namespace {

unsigned decode_hex_quad(std::string_view raw, size_t pos) {
  if (pos > raw.size() || raw.size() - pos < 4)
    throw std::runtime_error("minijson: bad \\u");
  unsigned cp = 0;
  const char* end = raw.data() + pos + 4;
  const auto res = std::from_chars(raw.data() + pos, end, cp, 16);
  if (res.ec != std::errc() || res.ptr != end)
    throw std::runtime_error("minijson: bad hex");
  return cp;
}

// Decodes JSON escapes, replacing unpaired UTF-16 surrogates with U+FFFD.
void decode_escapes(std::string_view raw, std::string& out) {
  out.clear();
  out.reserve(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    char c = raw[i];
    if (c != '\\') { out.push_back(c); continue; }
    if (++i >= raw.size()) throw std::runtime_error("minijson: bad escape");
    switch (raw[i]) {
      case '"': out.push_back('"'); break;
      case '\\': out.push_back('\\'); break;
      case '/': out.push_back('/'); break;
      case 'n': out.push_back('\n'); break;
      case 't': out.push_back('\t'); break;
      case 'r': out.push_back('\r'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'u': {
        unsigned cp = decode_hex_quad(raw, i + 1);
        i += 4;
        if (cp >= 0xD800 && cp <= 0xDBFF) {
          // JSON represents supplementary characters as two UTF-16 escapes.
          // Only consume the second escape when it completes this pair.
          if (raw.size() - i > 2 && raw[i + 1] == '\\' && raw[i + 2] == 'u') {
            const unsigned lo = decode_hex_quad(raw, i + 3);
            if (lo >= 0xDC00 && lo <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              i += 6;
            } else {
              cp = 0xFFFD;
            }
          } else {
            cp = 0xFFFD;
          }
        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
          cp = 0xFFFD;
        }
        if (cp < 0x80) {
          out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
          out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
          out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
          out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
          out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
          out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
          out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
          out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
          out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
          out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        break;
      }
      default: throw std::runtime_error("minijson: unsupported escape char");
    }
  }
}

struct Parser {
  std::string_view s;  // caller-owned; views into it outlive the parse call
  size_t pos = 0;
  std::string scratch_{};

  [[noreturn]] void fail(const char* msg) const {
    throw std::runtime_error(
        std::string("minijson: ") + msg + " at offset " + std::to_string(pos));
  }
  void skip_ws() {
    while (pos < s.size() &&
           (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
      ++pos;
  }
  char peek() {
    skip_ws();
    if (pos >= s.size()) fail("unexpected end of input");
    return s[pos];
  }
  void expect(char c) {
    if (peek() != c) fail("unexpected character");
    ++pos;
  }
  bool literal(std::string_view lit) {
    if (s.compare(pos, lit.size(), lit) == 0) {
      pos += lit.size();
      return true;
    }
    return false;
  }

  // Consumes a quoted string; returns its value either viewing the input or
  // decoding into owned storage when escapes are present.
  Value string_value() {
    expect('"');
    size_t start = pos;
    bool escaped = false;
    while (pos < s.size() && s[pos] != '"') {
      if (s[pos] == '\\') { escaped = true; ++pos; }
      ++pos;
    }
    if (pos >= s.size()) fail("unterminated string");
    std::string_view raw = s.substr(start, pos - start);
    ++pos;
    if (!escaped) return Value::make_string(raw);
    scratch_.clear();
    decode_escapes(raw, scratch_);
    return Value::make_owned_string(scratch_);
  }

  Value number_value() {
    skip_ws();
    size_t start = pos;
    if (pos < s.size() && s[pos] == '-') ++pos;
    while (pos < s.size() &&
           ((s[pos] >= '0' && s[pos] <= '9') || s[pos] == '+' || s[pos] == '-' ||
            s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E'))
      ++pos;
    std::string_view tok = s.substr(start, pos - start);
    size_t i = 0;
    auto digit = [&] { return i < tok.size() && tok[i] >= '0' && tok[i] <= '9'; };
    if (i < tok.size() && tok[i] == '-') ++i;
    if (!digit()) fail("malformed number");
    if (tok[i] == '0') {
      ++i;
    } else {
      while (digit()) ++i;
    }
    if (i < tok.size() && tok[i] == '.') {
      ++i;
      if (!digit()) fail("malformed number");
      while (digit()) ++i;
    }
    if (i < tok.size() && (tok[i] == 'e' || tok[i] == 'E')) {
      ++i;
      if (i < tok.size() && (tok[i] == '+' || tok[i] == '-')) ++i;
      if (!digit()) fail("malformed number");
      while (digit()) ++i;
    }
    if (i != tok.size()) fail("malformed number");
    int64_t iv{};
    auto r = std::from_chars(tok.data(), tok.data() + tok.size(), iv);
    if (r.ec == std::errc() && r.ptr == tok.data() + tok.size())
      return Value::make_int(iv);
    double dv{};
    auto r2 = std::from_chars(tok.data(), tok.data() + tok.size(), dv);
    if (r2.ec != std::errc() || r2.ptr != tok.data() + tok.size()) fail("malformed number");
    return Value::make_double(dv);
  }

  Value any() {
    char c = peek();
    switch (c) {
      case '{': return object_value();
      case '[': return array_value();
      case '"': return string_value();
      default: break;
    }
    if (literal("true")) return Value::make_bool(true);
    if (literal("false")) return Value::make_bool(false);
    if (literal("null")) return Value();
    if (c == '-' || (c >= '0' && c <= '9')) return number_value();
    fail("unexpected token");
  }

  // Nesting is bounded: the parser recurses per container, and a body of
  // ten thousand '[' would otherwise run the stack out — the malformed-HTTP
  // fuzzer's first find (2026-09-05, under AddressSanitizer). Real request
  // bodies nest a handful of levels; a document deeper than this is
  // refused as malformed, which the routes answer with 400.
  static constexpr int kMaxDepth = 256;
  int depth = 0;
  struct DepthScope {
    Parser& p;
    explicit DepthScope(Parser& parser) : p(parser) {
      if (++p.depth > kMaxDepth) p.fail("nesting deeper than 256 levels");
    }
    ~DepthScope() { --p.depth; }
  };

  Value array_value() {
    DepthScope scope(*this);
    expect('[');
    std::vector<Value> items;
    if (peek() == ']') { ++pos; return Value::make_array(std::move(items)); }
    while (true) {
      items.push_back(any());
      char c = peek();
      if (c == ',') { ++pos; continue; }
      if (c == ']') { ++pos; break; }
      fail("expected ',' or ']'");
    }
    return Value::make_array(std::move(items));
  }

  Value object_value() {
    DepthScope scope(*this);
    expect('{');
    std::vector<Member> members;
    if (peek() == '}') { ++pos; return Value::make_object(std::move(members)); }
    while (true) {
      Value key = string_value();
      expect(':');
      members.push_back({std::string(key.as_string()), any()});
      char c = peek();
      if (c == ',') { ++pos; continue; }
      if (c == '}') { ++pos; break; }
      fail("expected ',' or '}'");
    }
    return Value::make_object(std::move(members));
  }
};

}  // namespace

ParseResult parse(std::string_view in) {
  Parser p{in};
  Value root = p.any();
  const size_t consumed_after_value = p.pos;
  p.skip_ws();
  return ParseResult{std::move(root), consumed_after_value};
}

ParseResult parse_document(std::string_view in) {
  Parser p{in};
  Value root = p.any();
  const size_t consumed_after_value = p.pos;
  p.skip_ws();
  if (p.pos != in.size()) p.fail("unexpected trailing input");
  return ParseResult{std::move(root), consumed_after_value};
}

}  // namespace dgpp::minijson
