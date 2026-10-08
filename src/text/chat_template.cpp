// Chat-template interpreter. See the header for the
// design contract. This file: lexer (transformers' trim_blocks /
// lstrip_blocks semantics), recursive-descent parsers for the supported
// statement/expression subset, the Value model's operations, and the
// tree-walking renderer. Everything outside the subset refuses by name.
#include "text/chat_template.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace dgpp::text {

// ---------------------------------------------------------------------------
// AST — at glm scope: the header forward-declares MacroDef (Value's
// shared member), and MacroDef/Stmt reference Expr, so all three live
// together here where the parsers, the Value operations, and the
// renderer below see one set of types.
// ---------------------------------------------------------------------------

struct Expr;
struct MacroDef;
struct Stmt;
using ExprPtr = std::unique_ptr<Expr>;
using StmtPtr = std::unique_ptr<Stmt>;

// One node shape for every expression: a tagged union with the fields the
// tags use. Small, allocation-light, and enough for a subset this size.
struct Expr {
  enum class Tag {
    Const,     // value
    ListLit,   // kids[] as elements
    Name,      // name
    Getattr,   // kids[0].name
    Getitem,   // kids[0][kids[1]]
    Call,      // kids[0](args, kwargs)
    Filter,    // kids[0]|name(args, kwargs)
    Ternary,   // kids[0] ? kids[1] : kids[2]
    Binary,    // kids[0] op kids[1]       op: '+' '-' '~'
    Compare,   // kids[0] op kids[1]       op: ==,!=,<,<=,>,>=
    And,       // kids[0] and kids[1]  (returns operands, Python-style)
    Or,        // kids[0] or kids[1]
    Not,       // not kids[0]
    Test,      // kids[0] is [not] name
    In,        // kids[0] [not] in kids[1]
    Slice,     // kids[0][kids[1]:kids[2]:kids[3]] (absent parts are null)
  };
  Tag tag;
  Value value;       // Const
  std::string name;  // Name/Getattr/Filter/Compare/Test
  std::vector<ExprPtr> kids;  // operand slots (see the tags above)
  std::vector<std::pair<std::string, ExprPtr>> kwargs;
  bool inverted = false;  // Test ("is not") / In ("not in")
};

struct MacroDef {
  std::string name;
  std::vector<std::string> params;
  std::vector<ExprPtr> defaults;  // aligned with params; null = required
  std::vector<StmtPtr> body;  // owned here so the in-scope Value shares it
};

using Frame = std::unordered_map<std::string, Value>;

struct Stmt {
  enum class Tag { Text, Output, Set, If, For, Macro, Break };
  Tag tag;
  size_t line = 1;
  std::string text;               // Text
  ExprPtr expr;                   // Output/Set value; If+elif cond; For iter
  std::string target;             // Set target name
  std::string target_attr;        // Set target attribute ({% set ns.a = %})
  std::vector<std::string> names;  // For loop variables (1 or 2)
  std::shared_ptr<MacroDef> macro;      // Macro definition
  std::vector<StmtPtr> body;            // If-then / For / Macro body
  std::vector<std::pair<ExprPtr, std::vector<StmtPtr>>> elifs;
  std::vector<StmtPtr> else_body;
};

namespace {

// The house FNV-1a-64 (glm_loader.cpp / glm_tokenizer.cpp) — not the FNV
// standard basis; the golden corpus keys are defined by it.
uint64_t fnv1a64(const char* data, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<unsigned char>(data[i]);
    h *= 1099511628211ull;
  }
  return h;
}

[[noreturn]] void fail(size_t line, const std::string& what) {
  throw std::runtime_error("chat-template: line " + std::to_string(line) +
                           ": " + what);
}

size_t utf8_length(std::string_view s) {
  size_t n = 0;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    i += c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    ++n;
  }
  return n;
}

// Python repr(float) (str and json.dumps print the same): the shortest
// round-trip digits, in positional notation with a mandatory fraction while
// the decimal exponent lies in [-4, 16) ("1.0", "0.5", "100000.0", "0.0001",
// "1000000000000000.0") and in exponent notation outside it ("1e+16",
// "1e-05", "1.5e+30"). The choice is Python's — by the exponent, not by
// which spelling is shorter (std::to_chars alone prints 100000.0 as
// "1e+05" and 0.0001 as "1e-04"; fixed 2026-10-01).
std::string format_double(double d) {
  char buf[64];
  if (!std::isfinite(d)) {
    const auto res = std::to_chars(buf, buf + sizeof(buf), d);
    return std::string(buf, res.ptr - buf);
  }
  // The shortest digits as D[.DDD]e±XX.
  const auto res = std::to_chars(buf, buf + sizeof(buf), d, std::chars_format::scientific);
  const std::string sci(buf, res.ptr - buf);
  const size_t e = sci.find('e');
  const int exponent = std::atoi(sci.c_str() + e + 1);
  if (exponent < -4 || exponent >= 16) return sci;
  const bool negative = sci[0] == '-';
  std::string digits;
  for (size_t i = negative ? 1 : 0; i < e; ++i)
    if (sci[i] != '.') digits.push_back(sci[i]);
  std::string out = negative ? "-" : "";
  if (exponent < 0) {
    out += "0.";
    out.append(static_cast<size_t>(-exponent - 1), '0');
    out += digits;
    return out;
  }
  const size_t whole = static_cast<size_t>(exponent) + 1;
  if (digits.size() <= whole) {
    out += digits;
    out.append(whole - digits.size(), '0');
    out += ".0";
  } else {
    out.append(digits, 0, whole);
    out.push_back('.');
    out.append(digits, whole, std::string::npos);
  }
  return out;
}

bool is_all_digits(std::string_view s) {
  if (s.empty()) return false;
  for (const char c : s)
    if (c < '0' || c > '9') return false;
  return true;
}

// ---------------------------------------------------------------------------
// Lexer: text / {{ output }} / {% block %} / {# comment #} with the
// transformers whitespace semantics (trim_blocks, lstrip_blocks, and the
// explicit '-' markers).
// ---------------------------------------------------------------------------

struct Tok {
  enum class Kind { Text, Output, Block };
  Kind kind;
  std::string body;
  size_t line;
};

// Scans for the tag's closing delimiter, skipping quoted string bodies so
// "{{ '}' }}" and friends cannot terminate early.
size_t find_tag_end(std::string_view src, size_t from, char close1,
                    size_t line) {
  char quote = 0;
  for (size_t k = from; k + 1 < src.size(); ++k) {
    const char c = src[k];
    if (quote) {
      if (c == '\\' && k + 1 < src.size())
        ++k;
      else if (c == quote)
        quote = 0;
    } else if (c == '\'' || c == '"') {
      quote = c;
    } else if (c == close1 && src[k + 1] == '}') {
      return k;
    }
  }
  fail(line, "unterminated tag");
}

std::vector<Tok> lex(std::string_view src) {
  std::vector<Tok> out;
  size_t i = 0, line = 1;
  std::string text;
  auto flush_text = [&]() {
    if (!text.empty()) {
      out.push_back({Tok::Kind::Text, std::move(text), line});
      text.clear();
    }
  };
  while (i < src.size()) {
    const bool is_out = src.compare(i, 2, "{{") == 0;
    const bool is_block = !is_out && src.compare(i, 2, "{%") == 0;
    const bool is_comment = !is_out && !is_block && src.compare(i, 2, "{#") == 0;
    if (!is_out && !is_block && !is_comment) {
      if (src[i] == '\n') ++line;
      text.push_back(src[i++]);
      continue;
    }
    const bool ltrim = i + 2 < src.size() && src[i + 2] == '-';
    if (ltrim) {
      // The explicit marker strips ALL whitespace (spaces, tabs,
      // newlines) before the tag.
      const size_t last_ws = text.find_last_not_of(" \t\r\n");
      if (last_ws == std::string::npos)
        text.clear();
      else
        text.resize(last_ws + 1);
    } else if (is_block || is_comment) {
      // lstrip_blocks: strip the spaces/tabs between the last newline and
      // the tag (the newline itself stays — that one belongs to
      // trim_blocks).
      const size_t last_nl = text.find_last_of('\n');
      const size_t tail = last_nl == std::string::npos ? 0 : last_nl + 1;
      bool only_ws = true;
      for (size_t k = tail; k < text.size(); ++k)
        if (text[k] != ' ' && text[k] != '\t') {
          only_ws = false;
          break;
        }
      if (only_ws) text.resize(tail);
    }
    flush_text();

    const size_t body_start = i + 2 + (ltrim ? 1 : 0);
    const char close1 = is_out ? '}' : (is_comment ? '#' : '%');
    const size_t body_end = find_tag_end(src, body_start, close1, line);
    // A trailing '-' right before the closing delimiter is the trim
    // marker, not part of the expression.
    const bool rtrim = body_end > body_start && src[body_end - 1] == '-';
    const size_t body_len =
        rtrim ? body_end - 1 - body_start : body_end - body_start;
    const std::string body(src.substr(body_start, body_len));
    if (!is_comment)
      out.push_back(
          {is_out ? Tok::Kind::Output : Tok::Kind::Block, body, line});
    // Whitespace after the tag: the explicit marker strips everything;
    // trim_blocks (block tags) strips exactly one newline; output tags
    // trim only with the marker.
    i = body_end + 2;
    if (rtrim) {
      while (i < src.size() && (src[i] == ' ' || src[i] == '\t' ||
                               src[i] == '\r' || src[i] == '\n')) {
        if (src[i] == '\n') ++line;
        ++i;
      }
    } else if (is_block) {
      if (i + 1 < src.size() && src[i] == '\r' && src[i + 1] == '\n') {
        i += 2;
        ++line;
      } else if (i < src.size() && src[i] == '\n') {
        ++i;
        ++line;
      }
    }
  }
  flush_text();
  return out;
}

// ---------------------------------------------------------------------------
// Expression parsing
// ---------------------------------------------------------------------------

struct ETok {
  enum class T { Ident, Int, Str, Sym, End } t;
  std::string s;
  int64_t i = 0;
};

std::vector<ETok> lex_expr(std::string_view s, size_t line) {
  std::vector<ETok> out;
  size_t i = 0;
  while (i < s.size()) {
    const char c = s[i];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      ++i;
      continue;
    }
    if (c == '\'' || c == '"') {
      std::string str;
      ++i;
      while (i < s.size() && s[i] != c) {
        if (s[i] == '\\' && i + 1 < s.size()) {
          const char e = s[i + 1];
          switch (e) {
            case 'n': str.push_back('\n'); break;
            case 't': str.push_back('\t'); break;
            case 'r': str.push_back('\r'); break;
            case 'b': str.push_back('\b'); break;
            case 'f': str.push_back('\f'); break;
            default: str.push_back(e);
          }
          i += 2;
        } else {
          str.push_back(s[i++]);
        }
      }
      if (i >= s.size()) fail(line, "unterminated string literal");
      ++i;
      out.push_back({ETok::T::Str, std::move(str), 0});
      continue;
    }
    if ((c >= '0' && c <= '9') ||
        (c == '-' && i + 1 < s.size() && s[i + 1] >= '0' &&
         s[i + 1] <= '9' && out.empty())) {
      // Negative literals only in prefix position; a bare '-'/'+'
      // between operands is a binary operator handled at parse level.
      const size_t start = c == '-' ? i + 1 : i;
      int64_t v = 0;
      const auto res = std::from_chars(s.data() + start, s.data() + s.size(), v);
      if (res.ec != std::errc()) fail(line, "bad number literal");
      if (c == '-') v = -v;
      i = res.ptr - s.data();
      out.push_back({ETok::T::Int, "", v});
      continue;
    }
    const bool two_char =
        (c == '=' || c == '!' || c == '<' || c == '>') && i + 1 < s.size() &&
        s[i + 1] == '=';
    if (two_char) {
      out.push_back(ETok{ETok::T::Sym, std::string(s.substr(i, 2)), 0});
      i += 2;
      continue;
    }
    if (std::strchr("()[],.+-~|<>=:", c)) {
      out.push_back({ETok::T::Sym, std::string(1, c), 0});
      ++i;
      continue;
    }
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
      size_t j = i + 1;
      while (j < s.size() && ((s[j] >= 'a' && s[j] <= 'z') ||
                              (s[j] >= 'A' && s[j] <= 'Z') ||
                              (s[j] >= '0' && s[j] <= '9') || s[j] == '_'))
        ++j;
      out.push_back({ETok::T::Ident, std::string(s.substr(i, j - i)), 0});
      i = j;
      continue;
    }
    fail(line, std::string("unexpected character '") + c + "' in expression");
  }
  out.push_back({ETok::T::End, "", 0});
  return out;
}

class ExprParser {
 public:
  ExprParser(std::string_view src, size_t line)
      : toks_(lex_expr(src, line)), line_(line) {}

  ExprPtr parse() {
    ExprPtr e = ternary();
    expect_end();
    return e;
  }

 private:
  const ETok& peek() const { return toks_[pos_]; }
  bool accept_sym(const char* s) {
    if (peek().t == ETok::T::Sym && peek().s == s) {
      ++pos_;
      return true;
    }
    return false;
  }
  bool accept_kw(const char* s) {
    if (peek().t == ETok::T::Ident && peek().s == s) {
      ++pos_;
      return true;
    }
    return false;
  }
  void expect_sym(const char* s, const char* what) {
    if (!accept_sym(s))
      fail(line_, std::string(what) + ": expected '" + s + "'");
  }
  void expect_end() {
    if (peek().t != ETok::T::End) fail(line_, "trailing tokens in expression");
  }

  static ExprPtr make(Expr::Tag t) {
    auto e = std::make_unique<Expr>();
    e->tag = t;
    return e;
  }

  ExprPtr ternary() {
    ExprPtr a = or_();
    if (accept_kw("if")) {
      ExprPtr cond = or_();
      if (!accept_kw("else")) fail(line_, "ternary: expected 'else'");
      ExprPtr b = ternary();
      ExprPtr e = make(Expr::Tag::Ternary);
      e->kids.push_back(std::move(cond));
      e->kids.push_back(std::move(a));
      e->kids.push_back(std::move(b));
      return e;
    }
    return a;
  }

  ExprPtr or_() {
    ExprPtr a = and_();
    while (accept_kw("or")) {
      ExprPtr b = and_();
      ExprPtr e = make(Expr::Tag::Or);
      e->kids.push_back(std::move(a));
      e->kids.push_back(std::move(b));
      a = std::move(e);
    }
    return a;
  }

  ExprPtr and_() {
    ExprPtr a = not_();
    while (accept_kw("and")) {
      ExprPtr b = not_();
      ExprPtr e = make(Expr::Tag::And);
      e->kids.push_back(std::move(a));
      e->kids.push_back(std::move(b));
      a = std::move(e);
    }
    return a;
  }

  ExprPtr not_() {
    if (accept_kw("not")) {
      ExprPtr e = make(Expr::Tag::Not);
      e->kids.push_back(not_());
      return e;
    }
    return compare();
  }

  ExprPtr compare() {
    ExprPtr a = concat();
    const ETok& t = peek();
    if (t.t == ETok::T::Sym && (t.s == "==" || t.s == "!=" || t.s == "<" ||
                                 t.s == "<=" || t.s == ">" || t.s == ">=")) {
      const std::string op = t.s;
      ++pos_;
      ExprPtr b = concat();
      ExprPtr e = make(Expr::Tag::Compare);
      e->name = op;
      e->kids.push_back(std::move(a));
      e->kids.push_back(std::move(b));
      return e;
    }
    if (accept_kw("is")) {
      const bool inv = accept_kw("not");
      if (peek().t != ETok::T::Ident) fail(line_, "'is' needs a test name");
      static const char* kTests[] = {"defined", "undefined", "none", "string",
                                     "mapping", "sequence", "iterable", "true", "false"};
      ExprPtr e = make(Expr::Tag::Test);
      e->name = peek().s;
      e->inverted = inv;
      ++pos_;
      e->kids.push_back(std::move(a));
      bool known = false;
      for (const char* k : kTests)
        if (e->name == k) known = true;
      if (!known) fail(line_, "unsupported test 'is " + e->name + "'");
      return e;
    }
    const size_t save = pos_;
    if (accept_kw("in")) {
      ExprPtr e = make(Expr::Tag::In);
      e->kids.push_back(std::move(a));
      e->kids.push_back(concat());
      return e;
    }
    if (accept_kw("not") && accept_kw("in")) {
      ExprPtr e = make(Expr::Tag::In);
      e->inverted = true;
      e->kids.push_back(std::move(a));
      e->kids.push_back(concat());
      return e;
    }
    pos_ = save;
    return a;
  }

  ExprPtr concat() {
    ExprPtr a = add();
    while (accept_sym("~")) {
      ExprPtr b = add();
      ExprPtr e = make(Expr::Tag::Binary);
      e->name = "~";
      e->kids.push_back(std::move(a));
      e->kids.push_back(std::move(b));
      a = std::move(e);
    }
    return a;
  }

  ExprPtr add() {
    ExprPtr a = filter_chain();
    while (peek().t == ETok::T::Sym && (peek().s == "+" || peek().s == "-")) {
      const std::string op = peek().s;
      ++pos_;
      ExprPtr b = filter_chain();
      ExprPtr e = make(Expr::Tag::Binary);
      e->name = op;
      e->kids.push_back(std::move(a));
      e->kids.push_back(std::move(b));
      a = std::move(e);
    }
    return a;
  }

  ExprPtr filter_chain() {
    ExprPtr a = unary();
    while (accept_sym("|")) {
      if (peek().t != ETok::T::Ident) fail(line_, "expected filter name after '|'");
      ExprPtr e = make(Expr::Tag::Filter);
      e->name = peek().s;
      ++pos_;
      if (accept_sym("(")) parse_call_args(&e->kwargs, &e->kids);
      e->kids.insert(e->kids.begin(), std::move(a));  // kids[0] = filtered value
      if (e->name != "capitalize" && e->name != "tojson" &&
          e->name != "replace" && e->name != "length" && e->name != "trim" &&
          e->name != "default" && e->name != "string" && e->name != "safe" &&
          e->name != "items")
        fail(line_, "unsupported filter '" + e->name + "'");
      a = std::move(e);
    }
    return a;
  }

  ExprPtr unary() {
    if (accept_sym("-")) {
      if (peek().t != ETok::T::Int)
        fail(line_, "unary '-' is only supported on int literals");
      ExprPtr e = make(Expr::Tag::Const);
      e->value = Value::integer(-peek().i);
      ++pos_;
      return e;
    }
    return postfix();
  }

  ExprPtr postfix() {
    ExprPtr a = primary();
    for (;;) {
      if (accept_sym(".")) {
        // An attribute is an identifier or a digits-only token (foo.0 —
        // Jinja compiles digit attributes as indexes).
        if (peek().t != ETok::T::Ident && peek().t != ETok::T::Int)
          fail(line_, "expected attribute name after '.'");
        const std::string attr = peek().t == ETok::T::Ident
                                     ? peek().s
                                     : std::to_string(peek().i);
        ++pos_;
        ExprPtr e = make(Expr::Tag::Getattr);
        e->name = attr;
        e->kids.push_back(std::move(a));
        a = std::move(e);
      } else if (accept_sym("[")) {
        // A subscript, or a slice [start:stop:step] with any part absent
        // (the template's messages[::-1]).
        ExprPtr start, stop, step;
        bool slice = false;
        if (!(peek().t == ETok::T::Sym && peek().s == ":")) start = ternary();
        if (accept_sym(":")) {
          slice = true;
          if (!(peek().t == ETok::T::Sym && (peek().s == ":" || peek().s == "]")))
            stop = ternary();
          if (accept_sym(":")) {
            if (!(peek().t == ETok::T::Sym && peek().s == "]")) step = ternary();
          }
        }
        expect_sym("]", "subscript");
        if (slice) {
          ExprPtr e = make(Expr::Tag::Slice);
          e->kids.push_back(std::move(a));
          e->kids.push_back(std::move(start));
          e->kids.push_back(std::move(stop));
          e->kids.push_back(std::move(step));
          a = std::move(e);
        } else {
          ExprPtr e = make(Expr::Tag::Getitem);
          e->kids.push_back(std::move(a));
          e->kids.push_back(std::move(start));
          a = std::move(e);
        }
      } else if (accept_sym("(")) {
        ExprPtr e = make(Expr::Tag::Call);
        parse_call_args(&e->kwargs, &e->kids);
        e->kids.insert(e->kids.begin(), std::move(a));  // kids[0] = callee
        a = std::move(e);
      } else {
        break;
      }
    }
    return a;
  }

  ExprPtr primary() {
    const ETok& t = peek();
    if (t.t == ETok::T::Str) {
      ExprPtr e = make(Expr::Tag::Const);
      e->value = Value::string_value(t.s);
      ++pos_;
      return e;
    }
    if (t.t == ETok::T::Int) {
      ExprPtr e = make(Expr::Tag::Const);
      e->value = Value::integer(t.i);
      ++pos_;
      return e;
    }
    if (t.t == ETok::T::Ident) {
      if (t.s == "true" || t.s == "True") {
        ++pos_;
        return const_bool(true);
      }
      if (t.s == "false" || t.s == "False") {
        ++pos_;
        return const_bool(false);
      }
      if (t.s == "none" || t.s == "None") {
        ++pos_;
        ExprPtr e = make(Expr::Tag::Const);
        e->value = Value::null_value();
        return e;
      }
      ExprPtr e = make(Expr::Tag::Name);
      e->name = t.s;
      ++pos_;
      return e;
    }
    if (accept_sym("[")) {
      // List literal ('[a, b]' — the `in [...]` membership tests).
      ExprPtr e = make(Expr::Tag::ListLit);
      if (!accept_sym("]")) {
        for (;;) {
          e->kids.push_back(ternary());
          if (accept_sym(",")) continue;
          expect_sym("]", "list literal");
          break;
        }
      }
      return e;
    }
    if (accept_sym("(")) {
      // A parenthesized expression, or a tuple literal ('(a, b)' — the
      // template's `not in ('xhigh', 'medium', 'low')`).
      ExprPtr e = ternary();
      if (accept_sym(",")) {
        ExprPtr t = make(Expr::Tag::ListLit);
        t->kids.push_back(std::move(e));
        while (!(peek().t == ETok::T::Sym && peek().s == ")")) {
          t->kids.push_back(ternary());
          if (!accept_sym(",")) break;
        }
        expect_sym(")", "tuple literal");
        return t;
      }
      expect_sym(")", "parenthesized expression");
      return e;
    }
    fail(line_, "expected an expression");
  }

  static ExprPtr const_bool(bool b) {
    ExprPtr e = make(Expr::Tag::Const);
    e->value = Value::boolean(b);
    return e;
  }

  // Parses "arg, ..., name=arg, ...)" — the caller consumed '('. Positional
  // args land in `args`, named ones in `kwargs` (only tojson uses kwargs).
  void parse_call_args(std::vector<std::pair<std::string, ExprPtr>>* kwargs,
                       std::vector<ExprPtr>* args) {
    if (accept_sym(")")) return;
    for (;;) {
      const size_t save = pos_;
      if (peek().t == ETok::T::Ident) {
        const std::string nm = peek().s;
        ++pos_;
        if (accept_sym("=")) {
          kwargs->emplace_back(nm, ternary());
          if (accept_sym(",")) continue;
          expect_sym(")", "call arguments");
          return;
        }
        pos_ = save;  // a positional expression starting with a name
      }
      args->push_back(ternary());
      if (accept_sym(",")) continue;
      expect_sym(")", "call arguments");
      return;
    }
  }

  std::vector<ETok> toks_;
  size_t pos_ = 0;
  size_t line_;
};

// ---------------------------------------------------------------------------
// Statement parsing
// ---------------------------------------------------------------------------

class StmtParser {
 public:
  explicit StmtParser(const std::vector<Tok>& toks) : toks_(toks) {}

  // Parses statements until one of the `terminators` (empty list = to the
  // end). The terminator's keyword is reported through `ended`.
  std::vector<StmtPtr> parse_statements(const char* const* terminators,
                                        size_t n_term, std::string* ended) {
    std::vector<StmtPtr> out;
    for (;;) {
      if (pos_ >= toks_.size()) {
        if (n_term == 0) return out;
        fail(last_line_,
              "unexpected end of template (missing '" +
                  std::string(terminators[0]) + "')");
      }
      const Tok& t = toks_[pos_];
      if (t.kind == Tok::Kind::Text) {
        auto s = std::make_unique<Stmt>();
        s->tag = Stmt::Tag::Text;
        s->line = t.line;
        s->text = t.body;
        out.push_back(std::move(s));
        ++pos_;
        continue;
      }
      if (t.kind == Tok::Kind::Output) {
        auto s = std::make_unique<Stmt>();
        s->tag = Stmt::Tag::Output;
        s->line = t.line;
        s->expr = ExprParser(t.body, t.line).parse();
        out.push_back(std::move(s));
        ++pos_;
        continue;
      }
      last_line_ = t.line;
      // Normalize: strip the leading whitespace (the lexer keeps the
      // space that follows a '-%' or '{%' opener) so the keyword
      // dispatch sees '{%- set' and '{% set' identically.
      std::string body = t.body;
      const size_t body_b = body.find_first_not_of(" \t");
      if (body_b == std::string::npos) fail(t.line, "empty block tag");
      body = body.substr(body_b);
      const size_t sp = body.find_first_of(" \t");
      const std::string kw =
          sp == std::string::npos ? body : body.substr(0, sp);
      for (size_t k = 0; k < n_term; ++k) {
        if (kw == terminators[k]) {
          *ended = kw;
          ++pos_;
          return out;
        }
      }
      const std::string rest =
          sp == std::string::npos ? std::string() : body.substr(sp + 1);
      if (kw == "set") {
        out.push_back(parse_set(rest, t.line));
      } else if (kw == "if") {
        out.push_back(parse_if(t.line, rest));
      } else if (kw == "for") {
        out.push_back(parse_for(t.line, rest));
      } else if (kw == "macro") {
        out.push_back(parse_macro(t.line, rest));
      } else if (kw == "break") {
        auto s = std::make_unique<Stmt>();
        s->tag = Stmt::Tag::Break;
        s->line = t.line;
        out.push_back(std::move(s));
        ++pos_;
      } else {
        fail(t.line, "unsupported statement '" + kw + "'");
      }
    }
  }

 private:
  static ExprPtr parse_expr_str(std::string_view src, size_t line) {
    return ExprParser(src, line).parse();
  }

  StmtPtr parse_set(std::string rest, size_t line) {
    auto s = std::make_unique<Stmt>();
    s->tag = Stmt::Tag::Set;
    s->line = line;
    const size_t b = rest.find_first_not_of(" \t");
    if (b == std::string::npos) fail(line, "set: missing target");
    const size_t e = rest.find_last_not_of(" \t");
    rest = rest.substr(b, e - b + 1);
    const size_t eq = rest.find('=');
    if (eq == std::string::npos || (eq + 1 < rest.size() && rest[eq + 1] == '='))
      fail(line, "set: expected '=' after target");
    std::string target = rest.substr(0, eq);
    if (target.find(',') != std::string::npos)
      fail(line, "set: tuple targets are not supported");
    const size_t tb = target.find_first_not_of(" \t");
    const size_t te = target.find_last_not_of(" \t");
    target = target.substr(tb, te - tb + 1);
    const size_t dot = target.find('.');
    if (dot != std::string::npos) {
      s->target = target.substr(0, dot);
      s->target_attr = target.substr(dot + 1);
    } else {
      s->target = target;
    }
    s->expr = parse_expr_str(rest.substr(eq + 1), line);
    ++pos_;  // past the set tag (parsed from its text above)
    return s;
  }

  StmtPtr parse_if(size_t line, const std::string& rest) {
    auto s = std::make_unique<Stmt>();
    s->tag = Stmt::Tag::If;
    s->line = line;
    s->expr = parse_expr_str(rest, line);
    ++pos_;
    static const char* kBranchTerm[] = {"elif", "else", "endif"};
    std::string ended;
    s->body = parse_statements(kBranchTerm, 3, &ended);
    while (ended == "elif") {
      const Tok& et = toks_[pos_ - 1];  // the elif tag just consumed
      // Strip the leading whitespace and the 'elif' keyword itself.
      const size_t b = et.body.find_first_not_of(" \t");
      const size_t sp = et.body.find_first_of(" \t", b);
      const std::string cond_src = et.body.substr(
          sp == std::string::npos ? et.body.size() : sp + 1);
      auto cond = parse_expr_str(cond_src, et.line);
      std::vector<StmtPtr> branch = parse_statements(kBranchTerm, 3, &ended);
      s->elifs.emplace_back(std::move(cond), std::move(branch));
    }
    if (ended == "else") {
      static const char* kElseTerm[] = {"endif"};
      s->else_body = parse_statements(kElseTerm, 1, &ended);
    }
    if (ended != "endif") fail(line, "if: missing endif");
    return s;
  }

  StmtPtr parse_for(size_t line, const std::string& rest) {
    auto s = std::make_unique<Stmt>();
    s->tag = Stmt::Tag::For;
    s->line = line;
    const size_t in = rest.find(" in ");
    if (in == std::string::npos) fail(line, "for: expected 'in'");
    const std::string names = rest.substr(0, in);
    const size_t comma = names.find(',');
    if (comma != std::string::npos) {
      const std::string a = names.substr(0, comma);
      const std::string b = names.substr(comma + 1);
      const size_t ab = a.find_first_not_of(" \t");
      const size_t ae = a.find_last_not_of(" \t");
      const size_t bb = b.find_first_not_of(" \t");
      const size_t be = b.find_last_not_of(" \t");
      if (ab == std::string::npos || bb == std::string::npos)
        fail(line, "for: malformed tuple target");
      s->names.push_back(a.substr(ab, ae - ab + 1));
      s->names.push_back(b.substr(bb, be - bb + 1));
    } else {
      const size_t nb = names.find_first_not_of(" \t");
      const size_t ne = names.find_last_not_of(" \t");
      if (nb == std::string::npos) fail(line, "for: missing loop variable");
      s->names.push_back(names.substr(nb, ne - nb + 1));
    }
    const size_t ib = rest.find_first_not_of(" \t", in + 4);
    if (ib == std::string::npos) fail(line, "for: missing iterable");
    s->expr = parse_expr_str(rest.substr(ib), line);
    ++pos_;
    static const char* kForTerm[] = {"endfor"};
    std::string ended;
    s->body = parse_statements(kForTerm, 1, &ended);
    return s;
  }

  StmtPtr parse_macro(size_t line, const std::string& rest) {
    auto s = std::make_unique<Stmt>();
    s->tag = Stmt::Tag::Macro;
    s->line = line;
    const size_t lp = rest.find('(');
    const size_t rp = rest.find_last_of(')');
    if (lp == std::string::npos || rp == std::string::npos || rp < lp)
      fail(line, "macro: expected name and '(params)'");
    {
      const size_t nb = rest.find_first_not_of(" \t");
      if (nb == lp || nb == std::string::npos)
        fail(line, "macro: missing name");
    }
    s->macro = std::make_shared<MacroDef>();
    s->macro->name = rest.substr(0, lp);
    const std::string params = rest.substr(lp + 1, rp - lp - 1);
    size_t start = 0;
    while (start < params.size()) {
      const size_t comma = params.find(',', start);
      const std::string p =
          params.substr(start, comma == std::string::npos
                                   ? std::string::npos
                                   : comma - start);
      const size_t pb = p.find_first_not_of(" \t");
      const size_t pe = p.find_last_not_of(" \t");
      if (pb == std::string::npos) break;
      const std::string decl = p.substr(pb, pe - pb + 1);
      // name, or name=default (the default parsed as an expression and
      // evaluated at call time when the argument is absent).
      const size_t eq = decl.find('=');
      if (eq == std::string::npos) {
        s->macro->params.push_back(decl);
        s->macro->defaults.push_back(nullptr);
      } else {
        const size_t ne = decl.find_last_not_of(" \t", eq - 1);
        s->macro->params.push_back(decl.substr(0, ne + 1));
        s->macro->defaults.push_back(parse_expr_str(decl.substr(eq + 1), line));
      }
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
    ++pos_;
    static const char* kMacroTerm[] = {"endmacro"};
    std::string ended;
    s->macro->body = parse_statements(kMacroTerm, 1, &ended);
    return s;
  }

  const std::vector<Tok>& toks_;
  size_t pos_ = 0;
  size_t last_line_ = 1;
};

}  // namespace

// ---------------------------------------------------------------------------
// Value operations
// ---------------------------------------------------------------------------

bool Value::truthy() const {
  switch (kind_) {
    case Kind::Undefined:
    case Kind::Null:
      return false;
    case Kind::Bool:
      return b_;
    case Kind::Int:
      return i_ != 0;
    case Kind::Double:
      return d_ != 0.0;
    case Kind::String:
      return !s_.empty();
    case Kind::List:
      return !list_.empty();
    case Kind::Map:
    case Kind::Namespace:
      return !object_->empty();
    case Kind::Macro:
    case Kind::Method:
      return true;
  }
  return false;
}

bool Value::equals(const Value& o) const {
  if ((kind_ == Kind::Int || kind_ == Kind::Double) &&
      (o.kind_ == Kind::Int || o.kind_ == Kind::Double)) {
    const double a = kind_ == Kind::Int ? static_cast<double>(i_) : d_;
    const double b = o.kind_ == Kind::Int ? static_cast<double>(o.i_) : o.d_;
    return a == b;
  }
  if (kind_ != o.kind_)
    return false;  // (undefined == undefined lands here as kind == kind)
  switch (kind_) {
    case Kind::Undefined:
    case Kind::Null:
      return true;
    case Kind::Bool:
      return b_ == o.b_;
    case Kind::String:
      return s_ == o.s_;
    case Kind::List: {
      if (list_.size() != o.list_.size()) return false;
      for (size_t k = 0; k < list_.size(); ++k)
        if (!list_[k].equals(o.list_[k])) return false;
      return true;
    }
    case Kind::Map:
    case Kind::Namespace: {
      if (object_->size() != o.object_->size()) return false;
      for (const auto& [k, v] : *object_) {
        const Value* found = nullptr;
        for (const auto& m : *o.object_)
          if (m.first == k) {
            found = &m.second;
            break;
          }
        if (!found || !v.equals(*found)) return false;
      }
      return true;
    }
    default:
      return false;
  }
}

std::string Value::to_output_string() const {
  switch (kind_) {
    case Kind::Undefined:
      return std::string();
    case Kind::Null:
      return "None";
    case Kind::Bool:
      return b_ ? "True" : "False";
    case Kind::Int:
      return std::to_string(i_);
    case Kind::Double:
      return format_double(d_);
    case Kind::String:
      return s_;
    default:
      throw std::runtime_error(
          std::string("chat-template: cannot render a ") +
          (kind_ == Kind::List
               ? "list"
               : kind_ == Kind::Map
                     ? "map"
                     : kind_ == Kind::Namespace ? "namespace" : "callable") +
          " as text");
  }
}

static void json_escape(std::string_view s, std::string& out,
                        bool ensure_ascii) {
  out.push_back('"');
  for (size_t i = 0; i < s.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else if (ensure_ascii && c >= 0x80) {
          // Decode one UTF-8 sequence and emit it as \uXXXX (a surrogate
          // pair above the BMP), matching json.dumps(ensure_ascii=True).
          uint32_t cp = c;
          size_t len = 1;
          if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) {
            cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F);
            len = 2;
          } else if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) {
            cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) |
                 (s[i + 2] & 0x3F);
            len = 3;
          } else if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) {
            cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) |
                 ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
            len = 4;
          }
          i += len - 1;
          if (cp > 0xFFFF) {
            const uint32_t hi = 0xD800 + ((cp - 0x10000) >> 10);
            const uint32_t lo = 0xDC00 + ((cp - 0x10000) & 0x3FF);
            char buf[16];
            std::snprintf(buf, sizeof(buf), "\\u%04x\\u%04x", hi, lo);
            out += buf;
          } else {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", cp);
            out += buf;
          }
        } else {
          out.push_back(static_cast<char>(c));
        }
    }
  }
  out.push_back('"');
}

std::string Value::to_json(bool ensure_ascii) const {
  switch (kind_) {
    case Kind::Undefined:
      throw std::runtime_error("chat-template: tojson of an undefined value");
    case Kind::Null:
      return "null";
    case Kind::Bool:
      return b_ ? "true" : "false";
    case Kind::Int:
      return std::to_string(i_);
    case Kind::Double:
      if (!std::isfinite(d_))
        throw std::runtime_error("chat-template: tojson of non-finite number");
      return format_double(d_);
    case Kind::String: {
      std::string out;
      json_escape(s_, out, ensure_ascii);
      return out;
    }
    case Kind::List: {
      std::string out = "[";
      for (size_t k = 0; k < list_.size(); ++k) {
        if (k) out += ", ";
        out += list_[k].to_json(ensure_ascii);
      }
      out += "]";
      return out;
    }
    case Kind::Map: {
      std::string out = "{";
      for (size_t k = 0; k < object_->size(); ++k) {
        if (k) out += ", ";
        json_escape((*object_)[k].first, out, ensure_ascii);
        out += ": ";
        out += (*object_)[k].second.to_json(ensure_ascii);
      }
      out += "}";
      return out;
    }
    default:
      throw std::runtime_error(
          "chat-template: tojson of a non-serializable value");
  }
}

Value Value::get_attr(std::string_view name) const {
  // Jinja compiles foo.0 as foo[0]: digit attributes are indexes.
  if (is_all_digits(name)) {
    int64_t idx = 0;
    std::from_chars(name.data(), name.data() + name.size(), idx);
    return get_item(Value::integer(idx));
  }
  switch (kind_) {
    case Kind::Map:
      if (name == "items") {  // the dict method, never a key
        Value v;
        v.kind_ = Kind::Method;
        v.method_ = 1;
        return v;
      }
      for (const auto& m : *object_)
        if (m.first == name) return m.second;
      return Value();
    case Kind::Namespace:
      for (const auto& m : *object_)
        if (m.first == name) return m.second;
      return Value();
    case Kind::String:
      if (name == "split" || name == "strip" || name == "startswith" ||
          name == "endswith" || name == "rstrip" || name == "lstrip") {
        Value v;
        v.kind_ = Kind::Method;
        v.method_ = name == "split" ? 2 : name == "strip" ? 3 : name == "startswith" ? 4
                    : name == "endswith" ? 5 : name == "rstrip" ? 6 : 7;
        return v;
      }
      return Value();
    default:
      return Value();
  }
}

void Value::set_attr(std::string_view name, Value v) const {
  if (kind_ != Kind::Namespace && kind_ != Kind::Map)
    throw std::runtime_error("chat-template: set target is not a namespace");
  for (auto& m : *object_) {
    if (m.first == name) {
      m.second = std::move(v);
      return;
    }
  }
  object_->emplace_back(std::string(name), std::move(v));
}

Value Value::get_item(const Value& index) const {
  if (kind_ == Kind::List && index.kind_ == Kind::Int) {
    const int64_t i = index.i_;
    if (i >= 0 && i < static_cast<int64_t>(list_.size())) return list_[i];
    if (i < 0 && -i <= static_cast<int64_t>(list_.size()))
      return list_[list_.size() + i];
    return Value();
  }
  if (kind_ == Kind::Map && index.kind_ == Kind::String) {
    for (const auto& m : *object_)
      if (m.first == index.s_) return m.second;
    return Value();
  }
  if (kind_ == Kind::String && index.kind_ == Kind::Int) {
    // Python indexes strings by codepoint; out of range → undefined.
    const int64_t len = static_cast<int64_t>(utf8_length(s_));
    int64_t i = index.i_;
    if (i < 0) i += len;
    if (i < 0 || i >= len) return Value();
    size_t byte_pos = 0;
    for (int64_t k = 0; k < i; ++k) {
      const unsigned char c = static_cast<unsigned char>(s_[byte_pos]);
      byte_pos += c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    }
    const unsigned char c = static_cast<unsigned char>(s_[byte_pos]);
    const size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    return Value::string_value(s_.substr(byte_pos, n));
  }
  return Value();
}

Value Value::from_minijson(const minijson::Value& v) {
  switch (v.kind()) {
    case minijson::Value::Kind::Null:
      return null_value();
    case minijson::Value::Kind::Bool:
      return boolean(v.as_bool());
    case minijson::Value::Kind::Int:
      return integer(v.as_int());
    case minijson::Value::Kind::Double:
      return number(v.as_double());
    case minijson::Value::Kind::String:
      return string_value(std::string(v.as_string()));
    case minijson::Value::Kind::Array: {
      std::vector<Value> items;
      items.reserve(v.items().size());
      for (const auto& item : v.items()) items.push_back(from_minijson(item));
      return list_value(std::move(items));
    }
    case minijson::Value::Kind::Object: {
      Members members;
      members.reserve(v.members().size());
      for (const auto& m : v.members())
        members.emplace_back(m.key, from_minijson(m.value));
      return map_value(std::move(members));
    }
  }
  return Value();
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

namespace {

enum class Flow { Next, Break };

struct Ctx {
  Frame* root = nullptr;    // the render's global frame (macro closure)
  std::vector<Frame> stack; // loop frames and macro parameter frames
  size_t line = 1;          // current statement, for error messages

  const Value* find(std::string_view name) const {
    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
      auto f = it->find(std::string(name));
      if (f != it->end()) return &f->second;
    }
    const auto f = root->find(std::string(name));
    return f == root->end() ? nullptr : &f->second;
  }

  void set(const std::string& name, Value v) {
    if (!stack.empty())
      (*stack.rbegin())[name] = std::move(v);
    else
      (*root)[name] = std::move(v);
  }
};

struct Renderer {
  // A macro call renders its body into a fresh buffer with a stack of
  // just the parameter frame: macros close over the root frame only
  // (this template's macros reference nothing else across scopes).
  Value call_macro(const MacroDef& def, const std::vector<Value>& args) {
    if (args.size() > def.params.size())
      fail(ctx_.line, "macro '" + def.name + "': expected at most " +
                     std::to_string(def.params.size()) +
                     " argument(s), got " + std::to_string(args.size()));
    Frame params;
    for (size_t k = 0; k < args.size(); ++k) params[def.params[k]] = args[k];
    // Absent trailing arguments take their defaults (evaluated now, in
    // the caller's context); a parameter without one is required.
    for (size_t k = args.size(); k < def.params.size(); ++k) {
      if (!def.defaults[k])
        fail(ctx_.line, "macro '" + def.name + "': missing argument '" + def.params[k] + "'");
      params[def.params[k]] = eval(*def.defaults[k]);
    }
    std::vector<Frame> saved = std::move(ctx_.stack);
    ctx_.stack.clear();
    ctx_.stack.push_back(std::move(params));
    std::string buf;
    if (exec(def.body, &buf) == Flow::Break)
      fail(ctx_.line, "'break' outside of a loop");
    ctx_.stack = std::move(saved);
    return Value::string_value(std::move(buf));
  }

  Value call_method(int method, const Value& recv,
                    const std::vector<Value>& args) {
    if (method == 1) {  // items()
      if (recv.kind() != Value::Kind::Map)
        fail(ctx_.line, "'items' called on a non-map");
      if (!args.empty()) fail(ctx_.line, "items() takes no arguments");
      std::vector<Value> out;
      out.reserve(recv.as_members()->size());
      for (const auto& m : *recv.as_members()) {
        std::vector<Value> pair;
        pair.push_back(Value::string_value(m.first));
        pair.push_back(m.second);
        out.push_back(Value::list_value(std::move(pair)));
      }
      return Value::list_value(std::move(out));
    }
    if (method == 2) {  // split(sep)
      if (recv.kind() != Value::Kind::String)
        fail(ctx_.line, "'split' called on a non-string");
      if (args.size() != 1 || args[0].kind() != Value::Kind::String ||
          args[0].as_string().empty())
        fail(ctx_.line, "split() takes one non-empty separator");
      const std::string& sep = args[0].as_string();
      const std::string& s = recv.as_string();
      std::vector<Value> out;
      size_t start = 0;
      for (;;) {
        const size_t hit = s.find(sep, start);
        if (hit == std::string::npos) break;
        out.push_back(Value::string_value(s.substr(start, hit - start)));
        start = hit + sep.size();
      }
      out.push_back(Value::string_value(s.substr(start)));
      return Value::list_value(std::move(out));
    }
    if (method == 4 || method == 5) {  // startswith / endswith
      if (recv.kind() != Value::Kind::String)
        fail(ctx_.line, "'startswith'/'endswith' called on a non-string");
      if (args.size() != 1 || args[0].kind() != Value::Kind::String)
        fail(ctx_.line, "startswith()/endswith() take one string");
      const std::string& s = recv.as_string();
      const std::string& p = args[0].as_string();
      if (p.size() > s.size()) return Value::boolean(false);
      return Value::boolean(method == 4 ? s.compare(0, p.size(), p) == 0
                                        : s.compare(s.size() - p.size(), p.size(), p) == 0);
    }
    // strip() / rstrip() / lstrip() — Python's, over ' \t\n\r\x0b\x0c'
    // without an argument, over the argument's characters with one (the
    // GLM-4.7 template's rstrip('\n') / lstrip('\n'), 2026-09-10).
    const char* what = method == 6 ? "rstrip" : method == 7 ? "lstrip" : "strip";
    if (recv.kind() != Value::Kind::String) {
      std::string dbg = "kind=";
      dbg += std::to_string(static_cast<int>(recv.kind()));
      if (recv.kind() == Value::Kind::Undefined) dbg += " (undefined)";
      fail(ctx_.line, std::string("'") + what + "' called on a non-string (" + dbg + ")");
    }
    if (args.size() > 1 || (args.size() == 1 && args[0].kind() != Value::Kind::String))
      fail(ctx_.line, std::string(what) + "() takes at most one string argument");
    static const char* kWs = " \t\n\r\x0b\x0c";
    const std::string chars = args.empty() ? std::string(kWs) : args[0].as_string();
    const std::string& s = recv.as_string();
    size_t b = 0, e = s.size();
    if (method != 6) b = std::min(s.find_first_not_of(chars), s.size());
    if (method != 7) {
      const size_t last = s.find_last_not_of(chars);
      e = last == std::string::npos ? 0 : last + 1;
    }
    if (b >= e) return Value::string_value(std::string());
    return Value::string_value(s.substr(b, e - b));
  }

  Value apply_filter(const Expr& e, const Value& v,
                     const std::vector<Value>& args) {
    const std::string& name = e.name;
    if (name == "capitalize") {
      if (v.kind() != Value::Kind::String)
        fail(ctx_.line, "capitalize: not a string");
      // ASCII case mapping (the effort values are ASCII; other bytes
      // pass through untouched).
      std::string s = v.as_string();
      if (!s.empty()) {
        if (s[0] >= 'a' && s[0] <= 'z') s[0] = static_cast<char>(s[0] - 32);
        for (size_t k = 1; k < s.size(); ++k)
          if (s[k] >= 'A' && s[k] <= 'Z')
            s[k] = static_cast<char>(s[k] + 32);
      }
      return Value::string_value(std::move(s));
    }
    if (name == "tojson") {
      // transformers' override: tojson(x, ensure_ascii=False, ...) — the
      // template always passes ensure_ascii=False (raw UTF-8 output).
      bool ensure_ascii = false;
      for (const auto& [k, v] : e.kwargs) {
        if (k == "ensure_ascii") ensure_ascii = eval(*v).truthy();
      }
      return Value::string_value(v.to_json(ensure_ascii));
    }
    if (name == "replace") {
      if (v.kind() != Value::Kind::String || args.size() != 2 ||
          args[0].kind() != Value::Kind::String ||
          args[1].kind() != Value::Kind::String)
        fail(ctx_.line, "replace: needs a string and two string arguments");
      const std::string& from = args[0].as_string();
      const std::string& to = args[1].as_string();
      const std::string& s = v.as_string();
      std::string out;
      size_t start = 0;
      for (;;) {
        const size_t hit = s.find(from, start);
        if (hit == std::string::npos) break;
        out.append(s, start, hit - start);
        out += to;
        start = hit + from.size();
      }
      out.append(s, start, s.size() - start);
      return Value::string_value(std::move(out));
    }
    if (name == "length") {
      if (v.kind() == Value::Kind::List)
        return Value::integer(static_cast<int64_t>(v.as_list().size()));
      if (v.kind() == Value::Kind::Map)
        return Value::integer(static_cast<int64_t>(v.as_members()->size()));
      if (v.kind() == Value::Kind::String)
        return Value::integer(static_cast<int64_t>(utf8_length(v.as_string())));
      fail(ctx_.line, "length: not a list/map/string");
    }
    if (name == "trim") {  // Python str.strip(), the whitespace set
      if (v.kind() != Value::Kind::String) fail(ctx_.line, "trim: not a string");
      const std::string& x = v.as_string();
      const char* ws = " \t\n\r\x0b\x0c";
      const size_t b = x.find_first_not_of(ws);
      if (b == std::string::npos) return Value::string_value("");
      const size_t e = x.find_last_not_of(ws);
      return Value::string_value(x.substr(b, e - b + 1));
    }
    if (name == "default") {  // undefined -> the default (boolean=false form)
      if (args.size() != 1) fail(ctx_.line, "default: takes one argument");
      return v.is_defined() ? v : args[0];
    }
    if (name == "string") {  // str(x); strings stay themselves
      if (v.kind() == Value::Kind::String) return v;
      return Value::string_value(v.to_output_string());
    }
    if (name == "safe") return v;  // no autoescape: the identity
    if (name == "items") {
      if (v.kind() != Value::Kind::Map) fail(ctx_.line, "items: not a map");
      std::vector<Value> out;
      out.reserve(v.as_members()->size());
      for (const auto& m : *v.as_members()) {
        std::vector<Value> pair;
        pair.push_back(Value::string_value(m.first));
        pair.push_back(m.second);
        out.push_back(Value::list_value(std::move(pair)));
      }
      return Value::list_value(std::move(out));
    }
    fail(ctx_.line, "unsupported filter '" + name + "'");
  }

  Value eval(const Expr& e) {
    switch (e.tag) {
      case Expr::Tag::Const:
        return e.value;
      case Expr::Tag::ListLit: {
        std::vector<Value> items;
        items.reserve(e.kids.size());
        for (const auto& k : e.kids) items.push_back(eval(*k));
        return Value::list_value(std::move(items));
      }
      case Expr::Tag::Name: {
        const Value* v = ctx_.find(e.name);
        return v ? *v : Value();
      }
      case Expr::Tag::Getattr:
        return eval(*e.kids[0]).get_attr(e.name);
      case Expr::Tag::Getitem: {
        const Value base = eval(*e.kids[0]);
        return base.get_item(eval(*e.kids[1]));
      }
      case Expr::Tag::Slice: {
        const Value base = eval(*e.kids[0]);
        int64_t step = e.kids[3] ? eval(*e.kids[3]).as_int(1) : 1;
        if (step == 0) fail(ctx_.line, "slice step cannot be zero");
        const bool is_list = base.kind() == Value::Kind::List;
        if (!is_list && base.kind() != Value::Kind::String)
          fail(ctx_.line, "slice of a non-list/string");
        std::vector<Value> items;
        if (is_list) items = base.as_list();
        else {  // codepoints
          const std::string& x = base.as_string();
          for (size_t i = 0; i < x.size();) {
            const unsigned char c = static_cast<unsigned char>(x[i]);
            const size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
            items.push_back(Value::string_value(x.substr(i, n)));
            i += n;
          }
        }
        const int64_t len = static_cast<int64_t>(items.size());
        // Python's slice index normalization.
        auto norm = [&](const ExprPtr& k, int64_t dflt_pos, int64_t dflt_neg) {
          if (!k) return step > 0 ? dflt_pos : dflt_neg;
          int64_t i = eval(*k).as_int(0);
          if (i < 0) i += len;
          if (step > 0) return std::max<int64_t>(0, std::min<int64_t>(i, len));
          return std::max<int64_t>(-1, std::min<int64_t>(i, len - 1));
        };
        const int64_t start = norm(e.kids[1], 0, len - 1);
        const int64_t stop = norm(e.kids[2], len, -1);
        std::vector<Value> out;
        if (step > 0)
          for (int64_t i = start; i < stop; i += step) out.push_back(items[static_cast<size_t>(i)]);
        else
          for (int64_t i = start; i > stop; i += step) out.push_back(items[static_cast<size_t>(i)]);
        if (is_list) return Value::list_value(std::move(out));
        std::string joined;
        for (const Value& c : out) joined += c.as_string();
        return Value::string_value(std::move(joined));
      }
      case Expr::Tag::Call: {
        const Expr& callee = *e.kids[0];
        if (callee.tag == Expr::Tag::Name && callee.name == "raise_exception") {
          // transformers' raise_exception global: the template's own
          // validation errors (bad roles, unsupported content) surface as
          // render errors with the message.
          if (e.kids.size() != 2) fail(ctx_.line, "raise_exception(message) only");
          throw std::runtime_error("chat-template: " + eval(*e.kids[1]).to_output_string());
        }
        if (callee.tag == Expr::Tag::Name && callee.name == "range") {
          // range(n), range(a, b) and range(a, b, step) — python's, over
          // integers (GLM-5.3-Flash's template uses the two-argument form,
          // the full GLM-5.3's the one-argument form too).
          if (e.kids.size() < 2 || e.kids.size() > 4) fail(ctx_.line, "range takes one to three arguments");
          std::vector<Value> args;
          for (size_t i = 1; i < e.kids.size(); ++i) {
            args.push_back(eval(*e.kids[i]));
            if (args.back().kind() != Value::Kind::Int) fail(ctx_.line, "range: bounds must be integers");
          }
          const int64_t a = args.size() == 1 ? 0 : args[0].as_int(0);
          const int64_t b = args.size() == 1 ? args[0].as_int(0) : args[1].as_int(0);
          const int64_t step = args.size() == 3 ? args[2].as_int(0) : 1;
          if (step == 0) fail(ctx_.line, "range: step must not be zero");
          std::vector<Value> out;
          if (step > 0)
            for (int64_t k = a; k < b; k += step) out.push_back(Value::integer(k));
          else
            for (int64_t k = a; k > b; k += step) out.push_back(Value::integer(k));
          return Value::list_value(std::move(out));
        }
        if (callee.tag == Expr::Tag::Name && callee.name == "namespace") {
          if (e.kids.size() != 1)  // kids[0] is 'namespace' itself
            fail(ctx_.line, "namespace(...) takes only keyword arguments");
          Value::Members attrs;
          for (const auto& [k, v] : e.kwargs)
            attrs.emplace_back(k, eval(*v));
          return Value::namespace_value(std::move(attrs));
        }
        const Value target = eval(callee);
        std::vector<Value> args;
        args.reserve(e.kids.size() - 1);
        for (size_t k = 1; k < e.kids.size(); ++k)
          args.push_back(eval(*e.kids[k]));
        // A method call (obj.m()): the receiver is the OBJECT the
        // attribute was fetched from, not the bound method itself.
        if (callee.tag == Expr::Tag::Getattr) {
          const Value recv = eval(*callee.kids[0]);
          const Value m = recv.get_attr(callee.name);
          if (m.kind() == Value::Kind::Method)
            return call_method(m.as_method(), recv, args);
          fail(ctx_.line, "cannot call a non-callable value");
        }
        if (target.kind() == Value::Kind::Macro)
          return call_macro(*target.as_macro(), args);
        fail(ctx_.line, "cannot call a non-callable value");
      }
      case Expr::Tag::Filter: {
        const Value v = eval(*e.kids[0]);
        std::vector<Value> args;
        args.reserve(e.kids.size() - 1);
        for (size_t k = 1; k < e.kids.size(); ++k)
          args.push_back(eval(*e.kids[k]));
        return apply_filter(e, v, args);
      }
      case Expr::Tag::Ternary: {
        // Jinja evaluates only the taken branch (the undefined-guard
        // ternaries depend on the laziness).
        const Value cond = eval(*e.kids[0]);
        return cond.truthy() ? eval(*e.kids[1]) : eval(*e.kids[2]);
      }
      case Expr::Tag::Binary: {
        const Value a = eval(*e.kids[0]);
        const Value b = eval(*e.kids[1]);
        if (e.name == "~") {
          std::string out = a.to_output_string();
          out += b.to_output_string();
          return Value::string_value(std::move(out));
        }
        // '+' with two strings concatenates (Python semantics); otherwise
        // both operands must be numbers.
        if (a.kind() == Value::Kind::String && b.kind() == Value::Kind::String) {
          std::string out = a.as_string();
          out += b.as_string();
          return Value::string_value(std::move(out));
        }
        const bool a_num =
            a.kind() == Value::Kind::Int || a.kind() == Value::Kind::Double;
        const bool b_num =
            b.kind() == Value::Kind::Int || b.kind() == Value::Kind::Double;
        if (!a_num || !b_num)
          fail(ctx_.line, std::string("'") + e.name +
                         "' needs two numbers (or, for '+', two strings)");
        if (e.name == "+") {
          if (a.kind() == Value::Kind::Int && b.kind() == Value::Kind::Int)
            return Value::integer(a.as_int(0) + b.as_int(0));
          const double x = a.as_double(0);
          const double y = b.as_double(0);
          return Value::number(x + y);
        }
        if (e.name == "-") {
          if (a.kind() == Value::Kind::Int && b.kind() == Value::Kind::Int)
            return Value::integer(a.as_int(0) - b.as_int(0));
          const double x = a.as_double(0);
          const double y = b.as_double(0);
          return Value::number(x - y);
        }
        fail(ctx_.line, "unsupported operator '" + e.name + "'");
      }
      case Expr::Tag::Compare: {
        const Value a = eval(*e.kids[0]);
        const Value b = eval(*e.kids[1]);
        const std::string& op = e.name;
        if (op == "==") return Value::boolean(a.equals(b));
        if (op == "!=") return Value::boolean(!a.equals(b));
        const bool a_num =
            a.kind() == Value::Kind::Int || a.kind() == Value::Kind::Double;
        const bool b_num =
            b.kind() == Value::Kind::Int || b.kind() == Value::Kind::Double;
        if (a_num && b_num) {
          const double x = a.as_double(0);
          const double y = b.as_double(0);
          if (op == "<") return Value::boolean(x < y);
          if (op == "<=") return Value::boolean(x <= y);
          if (op == ">") return Value::boolean(x > y);
          return Value::boolean(x >= y);
        }
        if (a.kind() == Value::Kind::String && b.kind() == Value::Kind::String) {
          const std::string& x = a.as_string();
          const std::string& y = b.as_string();
          if (op == "<") return Value::boolean(x < y);
          if (op == "<=") return Value::boolean(x <= y);
          if (op == ">") return Value::boolean(x > y);
          return Value::boolean(x >= y);
        }
        fail(ctx_.line, "ordering comparison needs numbers or strings");
      }
      case Expr::Tag::And: {
        // Python and/or return the operand, not a boolean.
        const Value a = eval(*e.kids[0]);
        return a.truthy() ? eval(*e.kids[1]) : a;
      }
      case Expr::Tag::Or: {
        const Value a = eval(*e.kids[0]);
        return a.truthy() ? a : eval(*e.kids[1]);
      }
      case Expr::Tag::Not:
        return Value::boolean(!eval(*e.kids[0]).truthy());
      case Expr::Tag::Test: {
        const Value v = eval(*e.kids[0]);
        bool r;
        if (e.name == "defined")
          r = v.is_defined();
        else if (e.name == "undefined")
          r = !v.is_defined();
        else if (e.name == "true")
          r = v.kind() == Value::Kind::Bool && v.truthy();
        else if (e.name == "false")
          r = v.kind() == Value::Kind::Bool && !v.truthy();
        else if (e.name == "none")
          r = v.kind() == Value::Kind::Null;
        else if (e.name == "string")
          r = v.kind() == Value::Kind::String;
        else if (e.name == "mapping")
          r = v.kind() == Value::Kind::Map;  // namespaces are not mappings
        else if (e.name == "sequence")
          r = v.kind() == Value::Kind::String || v.kind() == Value::Kind::List;  // Jinja: str counts
        else  // iterable (strings, lists and maps — Python truth)
          r = v.kind() == Value::Kind::String || v.kind() == Value::Kind::List ||
              v.kind() == Value::Kind::Map;
        return Value::boolean(e.inverted ? !r : r);
      }
      case Expr::Tag::In: {
        const Value a = eval(*e.kids[0]);
        const Value b = eval(*e.kids[1]);
        bool r;
        if (a.kind() == Value::Kind::String && b.kind() == Value::Kind::String)
          r = b.as_string().find(a.as_string()) != std::string::npos;
        else if (b.kind() == Value::Kind::List) {
          r = false;
          for (const auto& item : b.as_list())
            if (item.equals(a)) {
              r = true;
              break;
            }
        } else if (b.kind() == Value::Kind::Map &&
                   a.kind() == Value::Kind::String) {
          r = b.get_attr(a.as_string()).is_defined();
        } else {
          fail(ctx_.line, "'in' needs string/string, or list/map containment");
        }
        return Value::boolean(e.inverted ? !r : r);
      }
    }
    fail(ctx_.line, "unhandled expression");
  }

  Flow exec(const std::vector<StmtPtr>& stmts, std::string* out) {
    for (const auto& s : stmts) {
      ctx_.line = s->line;
      switch (s->tag) {
        case Stmt::Tag::Text:
          if (!s->text.empty()) *out += s->text;
          break;
        case Stmt::Tag::Output:
          *out += eval(*s->expr).to_output_string();
          break;
        case Stmt::Tag::Set:
          if (s->target_attr.empty()) {
            ctx_.set(s->target, eval(*s->expr));
          } else {
            const Value* base = ctx_.find(s->target);
            if (!base || (base->kind() != Value::Kind::Namespace &&
                          base->kind() != Value::Kind::Map))
              fail(s->line,
                   "set target '" + s->target + "' is not a namespace");
            // Mutate through the shared object so every alias sees it.
            base->set_attr(s->target_attr, eval(*s->expr));
          }
          break;
        case Stmt::Tag::If: {
          if (eval(*s->expr).truthy()) {
            const Flow f = exec(s->body, out);
            if (f == Flow::Break) return f;
            break;
          }
          bool done = false;
          for (const auto& [cond, branch] : s->elifs) {
            if (eval(*cond).truthy()) {
              const Flow f = exec(branch, out);
              if (f == Flow::Break) return f;
              done = true;
              break;
            }
          }
          if (!done && !s->else_body.empty()) {
            const Flow f = exec(s->else_body, out);
            if (f == Flow::Break) return f;
          }
          break;
        }
        case Stmt::Tag::For: {
          const Value iter = eval(*s->expr);
          if (iter.kind() != Value::Kind::List)
            fail(s->line, "for: cannot iterate a non-list");
          const bool tuple = s->names.size() == 2;
          for (size_t idx = 0; idx < iter.as_list().size(); ++idx) {
            ctx_.stack.emplace_back();
            Frame& frame = *ctx_.stack.rbegin();
            if (tuple) {
              const Value& item = iter.as_list()[idx];
              if (item.kind() != Value::Kind::List || item.as_list().size() != 2)
                fail(s->line, "for: tuple target needs 2-element items");
              frame[s->names[0]] = item.as_list()[0];
              frame[s->names[1]] = item.as_list()[1];
            } else {
              frame[s->names[0]] = iter.as_list()[idx];
            }
            // Fresh loop metadata each iteration (index0/index/first/
            // last/length).
            Value::Members lm;
            lm.emplace_back("index0", Value::integer(static_cast<int64_t>(idx)));
            lm.emplace_back("index",
                            Value::integer(static_cast<int64_t>(idx) + 1));
            lm.emplace_back("first", Value::boolean(idx == 0));
            lm.emplace_back("last", Value::boolean(
                                        static_cast<int64_t>(idx) + 1 ==
                                        static_cast<int64_t>(
                                            iter.as_list().size())));
            lm.emplace_back("length",
                            Value::integer(
                                static_cast<int64_t>(iter.as_list().size())));
            // previtem / nextitem: undefined at the ends.
            lm.emplace_back("previtem", idx > 0 ? iter.as_list()[idx - 1] : Value());
            lm.emplace_back("nextitem", idx + 1 < iter.as_list().size()
                                            ? iter.as_list()[idx + 1]
                                            : Value());
            frame["loop"] = Value::namespace_value(std::move(lm));
            const Flow f = exec(s->body, out);
            ctx_.stack.pop_back();
            if (f == Flow::Break) break;
          }
          break;
        }
        case Stmt::Tag::Macro:
          ctx_.set(s->macro->name, Value::macro_value(s->macro));
          break;
        case Stmt::Tag::Break:
          return Flow::Break;
      }
    }
    return Flow::Next;
  }

  Ctx ctx_;
};

// ---------------------------------------------------------------------------
// Which globals the program reads — ChatTemplate::reads, resolved once at
// compile. The service's knob gate must ask "can this render depend on the
// global", which is a question about the syntax tree: a search of the source
// text counts a {# comment #}, a word the template prints, an attribute
// (m.preserve_thinking), a loop variable or a {% set %} that shadows the
// global as a knob the template has, and every one of those is how a
// request's flag ends up silently ignored.
// ---------------------------------------------------------------------------

// `bound` names what the current scope already rebinds (a {% set %} target,
// a {% for %} target, a macro parameter), which is what separates a read of
// the global from a read of a local with the same spelling.
void collect_name_reads(const Expr& e, const std::vector<std::string>& bound,
                        std::vector<std::string>& out) {
  if (e.tag == Expr::Tag::Name) {
    if (std::find(bound.begin(), bound.end(), e.name) == bound.end())
      out.push_back(e.name);
    return;
  }
  // Every other tag's `name` is a member, filter or test name, never a
  // global; only its operands can read one.
  for (const ExprPtr& k : e.kids)
    if (k) collect_name_reads(*k, bound, out);
  for (const auto& [key, v] : e.kwargs)
    if (v) collect_name_reads(*v, bound, out);
}

void collect_global_reads(const std::vector<StmtPtr>& body,
                          std::vector<std::string> bound,
                          std::vector<std::string>& out) {
  for (const StmtPtr& s : body) {
    if (!s) continue;
    switch (s->tag) {
      case Stmt::Tag::Text:
      case Stmt::Tag::Break:
        break;
      case Stmt::Tag::Output:
        collect_name_reads(*s->expr, bound, out);
        break;
      case Stmt::Tag::Set:
        collect_name_reads(*s->expr, bound, out);
        // From here down the scope, this name is the template's own.
        // (A {% set ns.attr = value %} target binds no name.)
        if (s->target_attr.empty()) bound.push_back(s->target);
        break;
      case Stmt::Tag::If:
        collect_name_reads(*s->expr, bound, out);
        collect_global_reads(s->body, bound, out);
        for (const auto& [cond, branch] : s->elifs) {
          collect_name_reads(*cond, bound, out);
          collect_global_reads(branch, bound, out);
        }
        collect_global_reads(s->else_body, bound, out);
        break;
      case Stmt::Tag::For: {
        collect_name_reads(*s->expr, bound, out);  // the iterable is the parent's
        std::vector<std::string> inner = bound;
        for (const std::string& n : s->names) inner.push_back(n);
        collect_global_reads(s->body, std::move(inner), out);
        break;
      }
      case Stmt::Tag::Macro:
        if (s->macro) {
          for (const ExprPtr& d : s->macro->defaults)
            if (d) collect_name_reads(*d, bound, out);
          // A macro body closes over the root frame only (call_macro), so
          // its scope is its parameters — not the definition site's.
          std::vector<std::string> inner;
          for (const std::string& p : s->macro->params) inner.push_back(p);
          collect_global_reads(s->macro->body, std::move(inner), out);
        }
        break;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// ChatTemplate
// ---------------------------------------------------------------------------

struct ChatTemplate::Impl {
  std::string source;
  uint64_t hash = 0;
  std::vector<StmtPtr> root;
  // Sorted, unique: every global the program can read (reads()'s table).
  std::vector<std::string> globals_read;
};

ChatTemplate::ChatTemplate(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ChatTemplate::~ChatTemplate() = default;
ChatTemplate::ChatTemplate(ChatTemplate&&) noexcept = default;
ChatTemplate& ChatTemplate::operator=(ChatTemplate&&) noexcept = default;

ChatTemplate ChatTemplate::compile(std::string source) {
  if (source.empty())
    throw std::runtime_error("chat-template: empty source");
  const std::vector<Tok> toks = lex(source);
  StmtParser parser(toks);
  static const char* kNone[] = {""};
  std::string ended;
  std::vector<StmtPtr> root = parser.parse_statements(kNone, 0, &ended);
  auto impl = std::make_unique<Impl>();
  impl->source = std::move(source);
  impl->hash = fnv1a64(impl->source.data(), impl->source.size());
  impl->root = std::move(root);
  collect_global_reads(impl->root, {}, impl->globals_read);
  std::sort(impl->globals_read.begin(), impl->globals_read.end());
  impl->globals_read.erase(std::unique(impl->globals_read.begin(),
                                       impl->globals_read.end()),
                           impl->globals_read.end());
  return ChatTemplate(std::move(impl));
}

ChatTemplate ChatTemplate::load(const std::string& chat_template_path) {
  FILE* f = std::fopen(chat_template_path.c_str(), "rb");
  if (!f)
    throw std::runtime_error("chat-template: cannot open " +
                             chat_template_path);
  std::string data;
  char buf[65536];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) data.append(buf, n);
  std::fclose(f);
  return compile(std::move(data));
}

std::string ChatTemplate::render(const Value& globals) const {
  const Value::Members* members = globals.as_members();
  if (!members)
    throw std::runtime_error("chat-template: render globals must be a map");
  Frame root;
  for (const auto& [k, v] : *members) root[k] = v;
  Renderer r;
  r.ctx_.root = &root;
  std::string out;
  out.reserve(4096);
  if (r.exec(impl_->root, &out) == Flow::Break)
    throw std::runtime_error("chat-template: 'break' outside of a loop");
  return out;
}

uint64_t ChatTemplate::source_hash() const { return impl_->hash; }

bool ChatTemplate::reads(std::string_view name) const {
  return std::binary_search(impl_->globals_read.begin(),
                            impl_->globals_read.end(), std::string(name));
}

}  // namespace dgpp::text
