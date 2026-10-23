#pragma once
// Minimal recursive-descent JSON reader for cold-path boot files
// (shardspec.json, config.json, safetensors headers). Unescaped strings are
// viewed into the caller-owned buffer; strings containing escape sequences
// are decoded into owned storage. Input must outlive the parsed tree.
#include <charconv>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dgpp::minijson {

class Value;
struct Member;  // defined after Value (C++17 vector-of-incomplete)

class Value {
 public:
  enum class Kind : int { Null, Bool, Int, Double, String, Array, Object };

  Value() : kind_(Kind::Null) {}
  explicit Value(Kind k) : kind_(k) {}
  // Default copy/move allowed deliberately: this is cold-path boot parsing
  // and callers occasionally want to stash subtrees (e.g. __metadata__).
  Value(const Value&) = default;
  Value& operator=(const Value&) = default;
  Value(Value&&) = default;
  Value& operator=(Value&&) = default;

  const std::vector<Member>& members() const;
  const Value* find(std::string_view key) const;
  const Value& at(std::string_view key) const;

  static Value make_bool(bool b) {
    Value v(Kind::Bool);
    v.bool_ = b;
    return v;
  }
  static Value make_int(int64_t i) {
    Value v(Kind::Int);
    v.int_ = i;
    return v;
  }
  static Value make_double(double d) {
    Value v(Kind::Double);
    v.dbl_ = d;
    return v;
  }
  static Value make_string(std::string_view s) {
    Value v(Kind::String);
    v.str_ = s;
    return v;
  }
  static Value make_owned_string(std::string s) {
    Value v(Kind::String);
    v.owned_ = std::make_shared<const std::string>(std::move(s));
    v.str_ = *v.owned_;
    return v;
  }
  static Value make_array(std::vector<Value> a) {
    Value v(Kind::Array);
    v.arr_ = std::move(a);
    return v;
  }
  static Value make_object(std::vector<Member> m) {
    Value v(Kind::Object);
    v.obj_ = std::move(m);
    return v;
  }

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::Null; }
  bool is_object() const { return kind_ == Kind::Object; }
  bool is_array() const { return kind_ == Kind::Array; }
  bool is_string() const { return kind_ == Kind::String; }
  bool is_number() const { return kind_ == Kind::Int || kind_ == Kind::Double; }
  bool is_bool() const { return kind_ == Kind::Bool; }

  bool as_bool(bool dflt = false) const { return kind_ == Kind::Bool ? bool_ : dflt; }
  double as_double(double dflt = 0) const {
    if (kind_ == Kind::Double) return dbl_;
    if (kind_ == Kind::Int) return static_cast<double>(int_);
    return dflt;
  }
  int64_t as_int(int64_t dflt = 0) const {
    if (kind_ == Kind::Int) return int_;
    if (kind_ == Kind::Double) return static_cast<int64_t>(dbl_);
    return dflt;
  }
  std::string_view as_string(std::string_view dflt = {}) const {
    return kind_ == Kind::String ? str_ : dflt;
  }

  const std::vector<Value>& items() const {
    static const std::vector<Value> empty;
    return is_array() ? arr_ : empty;
  }

 private:
  Kind kind_;
  union {
    bool bool_;
    int64_t int_;
    double dbl_;
  };
  std::string_view str_{};
  std::shared_ptr<const std::string> owned_{};
  std::vector<Value> arr_{};
  std::vector<Member> obj_{};
};

// Declared after Value: holds a complete value by member storage.
struct Member {
  std::string key;
  Value value;
};

// Out-of-line helpers; bodies need the complete Member type.
inline const std::vector<Member>& Value::members() const {
  static const std::vector<Member> empty;
  return is_object() ? obj_ : empty;
}
inline const Value* Value::find(std::string_view key) const {
  if (!is_object()) return nullptr;
  for (const auto& m : obj_)
    if (m.key == key) return &m.value;
  return nullptr;
}
inline const Value& Value::at(std::string_view key) const {
  const Value* v = find(key);
  if (!v)
    throw std::runtime_error(std::string("minijson: missing key '") +
                             std::string(key) + "'");
  return *v;
}

struct ParseResult {
  Value root;
  size_t consumed = 0;
};

ParseResult parse(std::string_view in);

// Requires only JSON whitespace after the value. Like parse(), returned string
// views borrow in, and consumed excludes trailing whitespace.
ParseResult parse_document(std::string_view in);

}  // namespace dgpp::minijson
