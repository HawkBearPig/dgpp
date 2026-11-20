#include "text/json_grammar.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <thread>
#include <utility>

#include "text/chat_template.hpp"
#include "text/tool_grammar.hpp"

namespace dgpp::text {

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

std::string json_text_of(const minijson::Value& v) {
  return Value::from_minijson(v).to_json(/*ensure_ascii=*/false);
}

namespace {

bool multiple_reachable(const IntegerPrefix& prefix, const IntegerBounds& bounds,
                        const std::string& step);
bool multiple_of(std::string_view text, const std::string& step);

uint32_t class_of_text(const std::string& text) {
  if (text.empty()) return 0;
  switch (text[0]) {
    case '"': return JsonSchemaNode::kString;
    case '{': return JsonSchemaNode::kObject;
    case '[': return JsonSchemaNode::kArray;
    case 't': case 'f': return JsonSchemaNode::kBoolean;
    case 'n': return JsonSchemaNode::kNull;
    default: {
      const bool integral = text.find_first_of(".eE") == std::string::npos;
      return integral ? (JsonSchemaNode::kNumber | JsonSchemaNode::kInteger)
                      : JsonSchemaNode::kNumber;
    }
  }
}

uint32_t type_bit(std::string_view t, const std::string& path) {
  if (t == "object") return JsonSchemaNode::kObject;
  if (t == "array") return JsonSchemaNode::kArray;
  if (t == "string") return JsonSchemaNode::kString;
  if (t == "number") return JsonSchemaNode::kNumber | JsonSchemaNode::kInteger;
  if (t == "integer") return JsonSchemaNode::kInteger;
  if (t == "boolean") return JsonSchemaNode::kBoolean;
  if (t == "null") return JsonSchemaNode::kNull;
  throw std::invalid_argument(path + ".type: unknown type '" + std::string(t) + "'");
}

struct Compiler {
  JsonSchema out;
  const minijson::Value* root = nullptr;
  std::map<const minijson::Value*, int> compiled;
  std::vector<std::string>* unenforced = nullptr;  // null: the strict compile

  const minijson::Value& resolve(std::string_view ref, const std::string& path) {
    if (ref.empty() || ref[0] != '#' || (ref.size() > 1 && ref[1] != '/'))
      throw std::invalid_argument(path + ".$ref: only local JSON Pointer references are supported");
    const auto* value = root;
    for (size_t pos = 2; pos <= ref.size() && ref.size() > 1;) {
      const auto end = ref.find('/', pos);
      const auto part = ref.substr(pos, end == std::string_view::npos ? ref.size()-pos : end-pos);
      std::string key;
      for (size_t i = 0; i < part.size(); ++i) {
        if (part[i] == '~') {
          if (++i == part.size() || (part[i] != '0' && part[i] != '1'))
            throw std::invalid_argument(path + ".$ref: invalid JSON Pointer escape");
          key += part[i] == '0' ? '~' : '/';
        } else key += part[i];
      }
      if (value && value->is_array()) {
        size_t index = 0;
        const auto [end_index, error] = std::from_chars(key.data(), key.data()+key.size(), index);
        value = error == std::errc{} && end_index == key.data()+key.size() &&
                (key.size() == 1 || key.front() != '0') && index < value->items().size()
                    ? &value->items()[index] : nullptr;
      } else value = value ? value->find(key) : nullptr;
      if (!value) throw std::invalid_argument(path + ".$ref: unresolved reference");
      if (end == std::string_view::npos) break;
      pos = end + 1;
    }
    return *value;
  }

  int compile(const minijson::Value& v, const std::string& path) {
    if (!v.is_object())
      throw std::invalid_argument(path + ": a schema must be an object");
    if (auto it = compiled.find(&v); it != compiled.end()) return it->second;
    const int index = static_cast<int>(out.nodes.size());
    compiled.emplace(&v, index);
    out.nodes.emplace_back();
    JsonSchemaNode n;
    static const char* kIgnored[] = {"title", "description", "default",
                                     "examples", "$schema", "$comment",
                                     "deprecated", "readOnly", "writeOnly",
                                     "$id", "$defs", "definitions"};
    static const char* kSupported[] = {"type", "properties", "required",
                                       "additionalProperties", "items",
                                       "minItems", "maxItems", "enum",
                                       "const", "anyOf", "minimum",
                                       "maximum", "exclusiveMinimum",
                                       "exclusiveMaximum", "$ref", "pattern", "format", "multipleOf",
                                       "x-dgpp-grammar"};
    // Keywords that narrow a typed value without an automaton behind
    // them: a tool argument's schema tolerates them (recorded, never
    // applied — the value keeps its type); a response_format schema
    // refuses them like any other unsupported one. The integer bounds
    // left this list on 2026-09-07 (compile_bounds decides them).
    static const char* kNarrowing[] = {
        "minLength",     "maxLength", "minProperties", "maxProperties", "uniqueItems",
        "minContains",   "maxContains",   "contentEncoding",  "contentMediaType"};
    const auto narrowing = [&](const std::string& key) {
      for (const char* k : kNarrowing)
        if (key == k) return true;
      return false;
    };
    const auto ignored = [&](const std::string& key) {
      for (const char* k : kIgnored)
        if (key == k) return true;
      return unenforced != nullptr && narrowing(key);
    };
    for (const minijson::Member& m : v.members()) {
      bool known = ignored(m.key);
      for (const char* k : kSupported) known = known || m.key == k;
      if (!known)
        throw std::invalid_argument(path + "." + m.key +
                                    ": unsupported keyword (the constrained "
                                    "subset is type, properties, required, "
                                    "additionalProperties, items, minItems, "
                                    "maxItems, enum, const, anyOf, and numeric "
                                    "minimum, maximum, "
                                    "exclusiveMinimum, exclusiveMaximum, multipleOf, "
                                    "pattern, format, and local $ref/$defs)");
      if (unenforced != nullptr && narrowing(m.key))
        unenforced->push_back(path + "." + m.key +
                              ": narrows the value; only its type is applied");
    }
    if (const auto* ref = v.find("$ref")) {
      if (!ref->is_string()) throw std::invalid_argument(path + ".$ref: must be a string");
      for (const auto& m : v.members())
        if (m.key != "$ref" && !ignored(m.key))
          throw std::invalid_argument(path + "." + m.key + ": constraints beside $ref are not supported");
      n.any_of.push_back(compile(resolve(ref->as_string(), path), path + ".$ref"));
      out.nodes[static_cast<size_t>(index)] = std::move(n);
      return index;
    }
    if (const minijson::Value* any = v.find("anyOf")) {
      if (!any->is_array() || any->items().empty())
        throw std::invalid_argument(path + ".anyOf: must be a non-empty array");
      for (const minijson::Member& m : v.members()) {
        if (m.key == "anyOf" || ignored(m.key)) continue;
        if (unenforced != nullptr && is_bound_keyword(m.key)) {
          unenforced->push_back(path + "." + m.key +
                                ": beside anyOf; only the alternatives apply");
          continue;
        }
        throw std::invalid_argument(path + "." + m.key +
                                    ": cannot be combined with anyOf here");
      }
      for (size_t i = 0; i < any->items().size(); ++i)
        n.any_of.push_back(
            compile(any->items()[i], path + ".anyOf[" + std::to_string(i) + "]"));
      out.nodes[static_cast<size_t>(index)] = std::move(n);
      return index;
    }
    bool typed = false;
    if (const minijson::Value* t = v.find("type")) {
      typed = true;
      n.types = 0;
      if (t->is_string()) {
        n.types = type_bit(t->as_string(), path);
      } else if (t->is_array()) {
        for (const minijson::Value& e : t->items()) {
          if (!e.is_string())
            throw std::invalid_argument(path + ".type: entries must be strings");
          n.types |= type_bit(e.as_string(), path);
        }
        if (n.types == 0)
          throw std::invalid_argument(path + ".type: empty type list");
      } else {
        throw std::invalid_argument(path + ".type: must be a string or a list");
      }
    }
    if (const minijson::Value* props = v.find("properties")) {
      if (!props->is_object())
        throw std::invalid_argument(path + ".properties: must be an object");
      for (const minijson::Member& pm : props->members()) {
        n.property_names.push_back(pm.key);
        n.property_nodes.push_back(compile(pm.value, path + ".properties." + pm.key));
        n.property_required.push_back(false);
      }
      if (!typed) n.types = JsonSchemaNode::kObject;
    }
    if (const minijson::Value* req = v.find("required")) {
      if (!req->is_array())
        throw std::invalid_argument(path + ".required: must be an array");
      for (const minijson::Value& r : req->items()) {
        if (!r.is_string())
          throw std::invalid_argument(path + ".required: entries must be strings");
        bool found = false;
        for (size_t i = 0; i < n.property_names.size(); ++i)
          if (n.property_names[i] == r.as_string()) {
            n.property_required[i] = true;
            found = true;
          }
        if (!found)
          throw std::invalid_argument(path + ".required: '" +
                                      std::string(r.as_string()) +
                                      "' is not a declared property");
      }
      if (!typed) n.types = JsonSchemaNode::kObject;
    }
    if (const minijson::Value* add = v.find("additionalProperties")) {
      if (add->is_bool())
        n.additional = add->as_bool() ? JsonSchemaNode::kAny : JsonSchemaNode::kClosed;
      else if (add->is_object())
        n.additional = compile(*add, path + ".additionalProperties");
      else
        throw std::invalid_argument(path + ".additionalProperties: must be a "
                                    "boolean or a schema");
      if (!typed) n.types = JsonSchemaNode::kObject;
    }
    if (const minijson::Value* items = v.find("items")) {
      if (!items->is_object())
        throw std::invalid_argument(path + ".items: must be a schema");
      n.items = compile(*items, path + ".items");
      if (!typed) n.types = JsonSchemaNode::kArray;
    }
    const auto bound = [&](const char* key, int* into) {
      if (const minijson::Value* b = v.find(key)) {
        if (!b->is_number() || !std::isfinite(b->as_double()) || b->as_double() < 0 ||
            b->as_double() > INT32_MAX || std::floor(b->as_double()) != b->as_double())
          throw std::invalid_argument(path + "." + key +
                                      ": must be an integer in [0, 2147483647]");
        *into = static_cast<int>(b->as_int());
        if (!typed) n.types = JsonSchemaNode::kArray;
      }
    };
    bound("minItems", &n.min_items);
    bound("maxItems", &n.max_items);
    if (n.max_items >= 0 && n.max_items < n.min_items)
      throw std::invalid_argument(path + ".maxItems: below minItems");
    const minijson::Value* en = v.find("enum");
    const minijson::Value* cs = v.find("const");
    if (en != nullptr && cs != nullptr)
      throw std::invalid_argument(path + ".const: cannot be combined with enum");
    if (en != nullptr || cs != nullptr) {
      n.has_enum = true;
      if (en != nullptr) {
        if (!en->is_array() || en->items().empty())
          throw std::invalid_argument(path + ".enum: must be a non-empty array");
        for (const minijson::Value& e : en->items())
          n.enum_texts.push_back(json_text_of(e));
      } else {
        n.enum_texts.push_back(json_text_of(*cs));
      }
      uint32_t classes = 0;
      for (const std::string& t : n.enum_texts) {
        const uint32_t c = class_of_text(t);
        if (c == JsonSchemaNode::kObject || c == JsonSchemaNode::kArray) out.container_enums = true;
        classes |= c;
      }
      if (typed) {
        for (const std::string& t : n.enum_texts)
          if ((class_of_text(t) & n.types) == 0)
            throw std::invalid_argument(path + "." + (en ? "enum" : "const") +
                                        ": the value " + t +
                                        " is outside the declared type");
        n.types &= classes;
      } else {
        n.types = classes;
      }
    }
    compile_bounds(v, path, &n);
    if (const auto* multiple = v.find("multipleOf")) {
      if (!multiple->is_number() || !(multiple->as_double() > 0) || !std::isfinite(multiple->as_double()))
        throw std::invalid_argument(path + ".multipleOf: must be a finite positive number");
      n.multiple_of = json_text_of(*multiple);
      if (n.bounds.active() && !multiple_reachable(IntegerPrefix{}, n.bounds, n.multiple_of)) {
        IntegerPrefix negative; negative.negative = true;
        if (!multiple_reachable(negative, n.bounds, n.multiple_of))
          throw std::invalid_argument(path + ".multipleOf: no integer multiple satisfies the bounds");
      }
    }
    for (const char* key : {"pattern", "format"}) {
      if (const auto* constraint = v.find(key)) {
        if (!constraint->is_string()) throw std::invalid_argument(path + "." + key + ": must be a string");
        try {
          n.strings.push_back(std::make_shared<StringConstraint>(std::string(constraint->as_string()),
            std::string_view(key) == "pattern" ? StringConstraint::Syntax::kPattern : StringConstraint::Syntax::kFormat));
        } catch (const std::invalid_argument& e) { throw std::invalid_argument(path + "." + key + ": " + e.what()); }
      }
    }
    if (const auto* grammar = v.find("x-dgpp-grammar")) {
      const auto* syntax = grammar->find("syntax");
      const auto* definition = grammar->find("definition");
      if (!syntax || !syntax->is_string() || !definition || !definition->is_string() ||
          (syntax->as_string() != "regex" && syntax->as_string() != "lark"))
        throw std::invalid_argument(path + ".x-dgpp-grammar: requires syntax (regex or lark) and definition");
      try {
        n.strings.push_back(std::make_shared<StringConstraint>(std::string(definition->as_string()),
          syntax->as_string() == "regex" ? StringConstraint::Syntax::kRegex : StringConstraint::Syntax::kLark));
      } catch (const std::invalid_argument& e) { throw std::invalid_argument(path + ".x-dgpp-grammar: " + e.what()); }
    }
    if (n.has_enum) {
      // Filter intersections at compile time. Leaving impossible enum
      // members live can steer decoding into a prefix with no completion.
      std::erase_if(n.enum_texts, [&](const std::string& text) {
        const auto parsed = minijson::parse(text);
        if (parsed.root.is_number() && !multiple_of(text, n.multiple_of)) return true;
        if (parsed.root.is_string())
          for (const auto& constraint : n.strings)
            if (!constraint->accepts(parsed.root.as_string(), true)) return true;
        return false;
      });
      if (n.enum_texts.empty())
        throw std::invalid_argument(path + (en ? ".enum" : ".const") + ": no value satisfies the constraints");
    }
    out.nodes[static_cast<size_t>(index)] = std::move(n);
    return index;
  }

  static bool is_bound_keyword(const std::string& key) {
    return key == "minimum" || key == "maximum" || key == "exclusiveMinimum" ||
           key == "exclusiveMaximum";
  }

  // Integer-only nodes retain the inclusive int64 range and its fast
  // prefix arithmetic. General numeric nodes use decimal bounds, which
  // also handle fractions and exponents. Enums are filtered at compile
  // time. Unsupported or empty ranges refuse by keyword under strict
  // compilation and are recorded as unenforced for non-strict tools.
  void compile_bounds(const minijson::Value& v, const std::string& path,
                      JsonSchemaNode* n) {
    static const char* kKeys[] = {"minimum", "exclusiveMinimum", "maximum",
                                  "exclusiveMaximum"};
    const minijson::Value* vals[4] = {nullptr, nullptr, nullptr, nullptr};
    int first = -1;
    for (int i = 0; i < 4; ++i) {
      vals[i] = v.find(kKeys[i]);
      if (vals[i] != nullptr && first < 0) first = i;
    }
    if (first < 0) return;
    // Strict: refuse at the offending keyword. Lax: every bound present is
    // recorded with the reason, none applied.
    const auto give_up = [&](int offender, const std::string& reason) {
      if (unenforced == nullptr)
        throw std::invalid_argument(path + "." + kKeys[offender] + ": " + reason);
      for (int i = 0; i < 4; ++i)
        if (vals[i] != nullptr)
          unenforced->push_back(path + "." + kKeys[i] + ": " + reason);
    };
    for (int i = 0; i < 4; ++i) {
      if (vals[i] == nullptr) continue;
      if (vals[i]->is_bool())
        return give_up(i, std::string("the draft-4 boolean form of ") + kKeys[i] +
                              " is not supported; no bound applied");
      if (!vals[i]->is_number())
        return give_up(i, std::string(kKeys[i]) + " must be a number; no bound applied");
    }
    // Integer alone by type; under an enum, every numeric member integral
    // (the texts are the constraint, whatever the declared type).
    bool integer_typed = (n->types & JsonSchemaNode::kInteger) != 0 &&
                         (n->types & JsonSchemaNode::kNumber) == 0;
    if (n->has_enum) {
      integer_typed = true;
      for (const std::string& t : n->enum_texts) {
        const uint32_t c = class_of_text(t);
        if ((c & JsonSchemaNode::kNumber) != 0 && (c & JsonSchemaNode::kInteger) == 0)
          integer_typed = false;
      }
    }
    if (!integer_typed) {
      DecimalBounds b;
      for (int i = 0; i < 4; ++i) {
        if (vals[i] == nullptr) continue;
        if (!std::isfinite(vals[i]->as_double()))
          return give_up(i, "must be a finite number");
        const DecimalNumber x = DecimalNumber::parse(json_text_of(*vals[i]));
        const bool lower = i < 2, exclusive = i == 1 || i == 3;
        bool& present = lower ? b.has_min : b.has_max;
        bool& open = lower ? b.exclusive_min : b.exclusive_max;
        DecimalNumber& limit = lower ? b.min : b.max;
        const int cmp = x.compare(limit);
        if (!present || (lower ? cmp > 0 : cmp < 0)) {
          limit = x;
          open = exclusive;
        } else if (cmp == 0) {
          open = open || exclusive;
        }
        present = true;
      }
      if (b.has_min && b.has_max &&
          (b.min.compare(b.max) > 0 ||
           (b.min.compare(b.max) == 0 && (b.exclusive_min || b.exclusive_max))))
        return give_up(vals[2] != nullptr ? 2 : 3,
                       "no number satisfies the range; no bound applied");
      if (n->has_enum) {
        std::vector<std::string> kept;
        uint32_t classes = 0;
        for (const std::string& t : n->enum_texts) {
          const uint32_t cls = class_of_text(t);
          if ((cls & JsonSchemaNode::kNumber) != 0 &&
              !b.contains(DecimalNumber::parse(t))) continue;
          kept.push_back(t);
          classes |= cls;
        }
        if (kept.empty())
          return give_up(first, "no enum member lies inside the range; no bound applied");
        n->enum_texts = std::move(kept);
        n->types &= classes;
      } else {
        n->decimal_bounds = std::move(b);
      }
      return;
    }
    // Each bound as an int64, rounded inward; a double beyond int64 (or
    // not a number at all) has no integer form worth applying.
    constexpr double kLimit = 9223372036854775808.0;  // 2^63
    const auto to_int = [&](const minijson::Value& b, bool lower, bool exclusive,
                            int64_t* out) {
      if (b.kind() == minijson::Value::Kind::Int) {
        const int64_t x = b.as_int();
        if (!exclusive) {
          *out = x;
          return true;
        }
        if (lower ? x == INT64_MAX : x == INT64_MIN) return false;
        *out = lower ? x + 1 : x - 1;
        return true;
      }
      const double d = b.as_double();
      if (!(d > -kLimit && d < kLimit)) return false;  // NaN included
      // minimum: ceil; exclusiveMinimum: floor + 1; maximum: floor;
      // exclusiveMaximum: ceil - 1.
      const double r = lower ? (exclusive ? std::floor(d) : std::ceil(d))
                             : (exclusive ? std::ceil(d) : std::floor(d));
      if (!(r > -kLimit && r < kLimit)) return false;
      *out = static_cast<int64_t>(r) + (exclusive ? (lower ? 1 : -1) : 0);
      return true;
    };
    IntegerBounds b;
    for (int i = 0; i < 4; ++i) {
      if (vals[i] == nullptr) continue;
      const bool lower = i < 2;
      const bool exclusive = i == 1 || i == 3;
      int64_t x = 0;
      if (!to_int(*vals[i], lower, exclusive, &x))
        return give_up(i, std::string(kKeys[i]) +
                              " is beyond the 64-bit integer range; no bound applied");
      if (lower) {
        b.min = b.has_min ? std::max(b.min, x) : x;
        b.has_min = true;
      } else {
        b.max = b.has_max ? std::min(b.max, x) : x;
        b.has_max = true;
      }
    }
    if (b.has_min && b.has_max && b.min > b.max)
      return give_up(vals[2] != nullptr ? 2 : 3,
                     "below the minimum: no integer satisfies the range; no "
                     "bound applied");
    if (n->has_enum) {
      // The enum's texts are the constraint: keep the members inside the
      // range (a non-numeric member is untouched) and carry no bound.
      std::vector<std::string> kept;
      uint32_t classes = 0;
      for (const std::string& t : n->enum_texts) {
        const uint32_t c = class_of_text(t);
        if ((c & JsonSchemaNode::kInteger) != 0) {
          IntegerPrefix p;
          for (const char ch : t) p.push(static_cast<uint8_t>(ch));
          if (!p.within(b)) continue;
        }
        kept.push_back(t);
        classes |= c;
      }
      if (kept.empty())
        return give_up(first, "no enum member lies inside the range; no bound applied");
      n->enum_texts = std::move(kept);
      n->types &= classes;
      return;
    }
    n->bounds = b;
  }
};

}  // namespace

JsonSchema compile_json_schema(const minijson::Value& schema,
                               std::vector<std::string>* unenforced) {
  Compiler c;
  c.root = &schema;
  c.unenforced = unenforced;
  c.out.root = c.compile(schema, "schema");
  // A recursive object/array is productive; a cycle of references/anyOf
  // alone never consumes a byte and must not recurse in leaves().
  std::vector<int> state(c.out.nodes.size());
  const auto visit = [&](const auto& self, int index) -> void {
    if (state[index] == 1) throw std::invalid_argument("schema.$ref: unproductive reference cycle");
    if (state[index] == 2) return;
    state[index] = 1;
    for (int alt : c.out.nodes[index].any_of) self(self, alt);
    state[index] = 2;
  };
  for (size_t i = 0; i < state.size(); ++i) visit(visit, static_cast<int>(i));
  return std::move(c.out);
}

JsonSchema json_object_schema() {
  JsonSchema s;
  JsonSchemaNode root;
  root.types = JsonSchemaNode::kObject;
  root.additional = JsonSchemaNode::kAny;
  s.nodes.push_back(std::move(root));
  s.root = 0;
  return s;
}

// ---------------------------------------------------------------------------
// The integer prefix
// ---------------------------------------------------------------------------

void IntegerPrefix::push(uint8_t b) {
  if (b == '-') {
    if (digits == 0) negative = true;
    return;
  }
  if (b < '0' || b > '9') return;
  ++digits;
  if (huge) return;
  if (digits >= 20) {  // no leading zeros in JSON: twenty digits is >= 10^19
    huge = true;
    return;
  }
  magnitude = magnitude * 10 + static_cast<uint64_t>(b - '0');
}

namespace {

// |x| as an unsigned magnitude, INT64_MIN included.
uint64_t abs_u64(int64_t x) {
  if (x >= 0) return static_cast<uint64_t>(x);
  return static_cast<uint64_t>(-(x + 1)) + 1u;
}

}  // namespace

bool IntegerPrefix::within(const IntegerBounds& b) const {
  if (digits == 0) return false;
  if (huge) return negative ? !b.has_min : !b.has_max;
  if (!negative) {
    // v = magnitude >= 0.
    if (b.has_min && b.min > 0 && magnitude < static_cast<uint64_t>(b.min)) return false;
    if (b.has_max && (b.max < 0 || magnitude > static_cast<uint64_t>(b.max))) return false;
    return true;
  }
  // v = -magnitude <= 0.
  if (b.has_min && (b.min > 0 || magnitude > abs_u64(b.min))) return false;
  if (b.has_max && b.max < 0 && magnitude < abs_u64(b.max)) return false;
  return true;
}

bool IntegerPrefix::can_reach(const IntegerBounds& b) const {
  if (digits == 0) {
    // A bare '-': any value at or below zero may follow ("-0" included).
    if (!negative) return true;
    return !b.has_min || b.min <= 0;
  }
  if (huge) return negative ? !b.has_min : !b.has_max;
  if (magnitude == 0) return within(b);  // "0" / "-0" admit no more digits
  // Unbounded in the direction the digits grow: some completion passes
  // the other bound.
  if (negative ? !b.has_min : !b.has_max) return true;
  // The completions of M are [M*10^k, (M+1)*10^k - 1] for k >= 0 (their
  // mirror for a negative); walk k up until the interval passes the bound
  // that caps it. Magnitudes: `low` is refused before it can overflow (it
  // is already past any int64 bound), `high` saturates.
  constexpr uint64_t kMax = ~0ull;
  uint64_t low = magnitude, high = magnitude;
  if (!negative) {
    if (b.max < 0) return false;  // every completion is positive
    const uint64_t cap = static_cast<uint64_t>(b.max);
    const bool floor_free = !b.has_min || b.min <= 0;
    const uint64_t floor = floor_free ? 0u : static_cast<uint64_t>(b.min);
    for (;;) {
      if (low > cap) return false;
      if (floor_free || high >= floor) return true;
      if (low > kMax / 10u) return false;
      low *= 10u;
      high = high > (kMax - 9u) / 10u ? kMax : high * 10u + 9u;
    }
  }
  if (b.min > 0) return false;  // every completion is negative
  const uint64_t cap = abs_u64(b.min);
  const bool floor_free = !b.has_max || b.max >= 0;
  const uint64_t floor = floor_free ? 0u : abs_u64(b.max);
  for (;;) {
    if (low > cap) return false;          // -low < min
    if (floor_free || high >= floor) return true;  // -high <= max
    if (low > kMax / 10u) return false;
    low *= 10u;
    high = high > (kMax - 9u) / 10u ? kMax : high * 10u + 9u;
  }
}

// ---------------------------------------------------------------------------
// Decimal bounds, including fractional and exponent prefixes.
// ---------------------------------------------------------------------------

namespace {

int compare_digits(const std::string& a, const std::string& b) {
  for (size_t i = 0; i < std::max(a.size(), b.size()); ++i) {
    const char x = i < a.size() ? a[i] : '0';
    const char y = i < b.size() ? b[i] : '0';
    if (x != y) return x < y ? -1 : 1;
  }
  return 0;
}

}  // namespace

DecimalNumber DecimalNumber::parse(std::string_view text) {
  DecimalNumber n;
  size_t i = 0;
  if (!text.empty() && text[0] == '-') { n.negative = true; ++i; }
  bool fraction = false;
  int64_t places = 0;
  for (; i < text.size() && text[i] != 'e' && text[i] != 'E'; ++i) {
    if (text[i] == '.') { fraction = true; continue; }
    if (!fraction) ++places;
    n.digits.push_back(text[i]);
  }
  const size_t first = n.digits.find_first_not_of('0');
  if (first == std::string::npos) return DecimalNumber{};
  n.exponent = places - static_cast<int64_t>(first) - 1;
  n.digits.erase(0, first);
  if (i < text.size()) {
    ++i;
    bool negative = false;
    if (i < text.size() && (text[i] == '+' || text[i] == '-'))
      negative = text[i++] == '-';
    // Exponents beyond this cannot approach any representable schema
    // bound, even after accounting for the mantissa's decimal places.
    constexpr int64_t cap = INT64_MAX / 4;
    int64_t exponent = 0;
    for (; i < text.size(); ++i)
      exponent = exponent > (cap - 9) / 10 ? cap : exponent * 10 + text[i] - '0';
    n.exponent += negative ? -exponent : exponent;
  }
  return n;
}

int DecimalNumber::compare(const DecimalNumber& other) const {
  if (negative != other.negative) return negative ? -1 : 1;
  int cmp = 0;
  if (digits.empty() || other.digits.empty())
    cmp = digits.empty() ? (other.digits.empty() ? 0 : -1) : 1;
  else if (exponent != other.exponent)
    cmp = exponent < other.exponent ? -1 : 1;
  else
    cmp = compare_digits(digits, other.digits);
  return negative ? -cmp : cmp;
}

bool DecimalBounds::contains(const DecimalNumber& value) const {
  if (has_min) {
    const int cmp = value.compare(min);
    if (cmp < 0 || (cmp == 0 && exclusive_min)) return false;
  }
  if (has_max) {
    const int cmp = value.compare(max);
    if (cmp > 0 || (cmp == 0 && exclusive_max)) return false;
  }
  return true;
}

bool DecimalPrefix::within(const DecimalBounds& bounds) const {
  return !text.empty() && bounds.contains(DecimalNumber::parse(text));
}

bool DecimalPrefix::can_reach(const DecimalBounds& bounds) const {
  const size_t e = text.find_first_of("eE");
  DecimalNumber mantissa = DecimalNumber::parse(std::string_view(text).substr(0, e));
  // Reflect negative candidates into a nonnegative magnitude interval.
  DecimalBounds b = bounds;
  if (!text.empty() && text[0] == '-') {
    std::swap(b.has_min, b.has_max);
    std::swap(b.exclusive_min, b.exclusive_max);
    std::swap(b.min, b.max);
    b.min.negative = !b.min.digits.empty() && !b.min.negative;
    b.max.negative = !b.max.digits.empty() && !b.max.negative;
  }
  mantissa.negative = false;
  const DecimalNumber zero;
  if (mantissa.digits.empty()) {
    if (b.contains(zero)) return true;
    // An unfinished zero mantissa can acquire significant digits and an
    // exponent. After 'e' its value is permanently zero.
    return e == std::string::npos &&
           (!b.has_max || b.max.compare(zero) > 0);
  }
  if (b.has_max && b.max.compare(zero) <= 0) return false;
  if (b.has_min && b.min.compare(zero) <= 0) b.has_min = false;
  if (e == std::string::npos) {
    // Further mantissa digits describe [prefix, successor) at any decimal
    // scale. Only scales adjacent to the bounds need checking: all other
    // scales lie wholly beyond a bound (or in the unbounded direction).
    DecimalNumber successor = mantissa;
    size_t i = successor.digits.size();
    while (i > 0 && successor.digits[i - 1] == '9') successor.digits[--i] = '0';
    const bool carry = i == 0;
    if (carry) successor.digits.insert(successor.digits.begin(), '1');
    else ++successor.digits[i - 1];
    const int64_t scales[] = {0, b.min.exponent - 1, b.min.exponent,
                              b.min.exponent + 1, b.max.exponent - 1,
                              b.max.exponent, b.max.exponent + 1};
    for (const int64_t scale : scales) {
      mantissa.exponent = scale;
      successor.exponent = scale + (carry ? 1 : 0);
      if (b.has_min && successor.compare(b.min) <= 0) continue;
      if (b.has_max) {
        const int cmp = mantissa.compare(b.max);
        if (cmp > 0 || (cmp == 0 && b.exclusive_max)) continue;
      }
      return true;
    }
    return false;
  }
  // Once the exponent opens, the mantissa is fixed. Convert its bounds
  // to an inclusive range of integer exponents, then test the exponent
  // prefix (which, unlike a JSON integer, permits '+' and leading zeros).
  IntegerBounds powers;
  if (b.has_min) {
    const int cmp = compare_digits(mantissa.digits, b.min.digits);
    powers.has_min = true;
    powers.min = b.min.exponent - mantissa.exponent +
                 ((cmp < 0 || (cmp == 0 && b.exclusive_min)) ? 1 : 0);
  }
  if (b.has_max) {
    const int cmp = compare_digits(mantissa.digits, b.max.digits);
    powers.has_max = true;
    powers.max = b.max.exponent - mantissa.exponent -
                 ((cmp > 0 || (cmp == 0 && b.exclusive_max)) ? 1 : 0);
  }
  if (powers.has_min && powers.has_max && powers.min > powers.max) return false;
  std::string_view tail(text.data() + e + 1, text.size() - e - 1);
  if (tail.empty()) return true;
  IntegerPrefix prefix;
  if (tail[0] == '-' || tail[0] == '+') {
    prefix.negative = tail[0] == '-';
    tail.remove_prefix(1);
  }
  for (const char ch : tail) {
    if (prefix.digits == 0 && ch == '0') continue;
    prefix.push(static_cast<uint8_t>(ch));
  }
  if (prefix.digits == 0)
    return prefix.negative ? (!powers.has_min || powers.min <= 0)
                           : (!powers.has_max || powers.max >= 0);
  return prefix.can_reach(powers);
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

namespace {

const char* literal_text(uint8_t lit) {
  switch (lit) {
    case 1: return "true";
    case 2: return "false";
    default: return "null";
  }
}

bool is_digit(uint8_t b) { return b >= '0' && b <= '9'; }
bool is_hex(uint8_t b) {
  return is_digit(b) || (b >= 'a' && b <= 'f') || (b >= 'A' && b <= 'F');
}

void push_event(JsonLexer::Step* s, JsonLexer::Event e) {
  for (JsonLexer::Event& slot : s->events)
    if (slot == JsonLexer::Event::kNone) {
      slot = e;
      return;
    }
}

}  // namespace

std::string JsonLexer::literal_rest() const {
  if (state_ != State::kLiteral) return "";
  return std::string(literal_text(lit_) + lit_pos_);
}

void JsonLexer::end_value(Step* s) {
  push_event(s, Event::kValueEnd);
  state_ = stack_.empty() ? State::kDone : State::kValueEnd;
  integer_only_ = false;
}

bool JsonLexer::string_byte(uint8_t b, Step* s, bool key) {
  const Event content = key ? Event::kKeyByte : Event::kValueByte;
  if (esc_ == 0) {
    if (b == '"') {
      if (key) {
        state_ = State::kKeyEnd;
        push_event(s, Event::kKeyEnd);
      } else {
        push_event(s, Event::kValueByte);
        end_value(s);
      }
      return true;
    }
    if (b == '\\') {
      esc_ = 1;
      push_event(s, content);
      return true;
    }
    if (b < 0x20) return false;  // an unescaped control character
    push_event(s, content);
    return true;
  }
  if (esc_ == 1) {
    switch (b) {
      case '"': case '\\': case '/': case 'b': case 'f': case 'n': case 'r':
      case 't':
        esc_ = 0;
        break;
      case 'u':
        esc_ = 2;  // four hex digits to come
        break;
      default:
        return false;
    }
    push_event(s, content);
    return true;
  }
  if (!is_hex(b)) return false;
  esc_ = esc_ == 5 ? 0 : static_cast<uint8_t>(esc_ + 1);
  push_event(s, content);
  return true;
}

JsonLexer::Step JsonLexer::feed(uint8_t b) {
  Step s;
  const auto reject = [&] {
    s.ok = false;
    return s;
  };
  // The whitespace-run cap: structural whitespace counts, anything else
  // (a string's content included) resets.
  if (in_string()) {
    ws_run_ = 0;
  } else if (is_ws(b)) {
    if (ws_run_ >= kMaxWsRun) return reject();
    ++ws_run_;
  } else {
    ws_run_ = 0;
  }
  for (int pass = 0; pass < 2; ++pass) {
    switch (state_) {
      case State::kValue:
      case State::kArrayOpen: {
        if (is_ws(b)) return s;
        if (b == ']' && state_ == State::kArrayOpen) {
          stack_.pop_back();
          push_event(&s, Event::kClose);
          end_value(&s);
          return s;
        }
        uint32_t cls = 0;
        switch (b) {
          case '{': cls = JsonSchemaNode::kObject; break;
          case '[': cls = JsonSchemaNode::kArray; break;
          case '"': cls = JsonSchemaNode::kString; break;
          case '-': cls = JsonSchemaNode::kNumber; break;
          case 't': case 'f': cls = JsonSchemaNode::kBoolean; break;
          case 'n': cls = JsonSchemaNode::kNull; break;
          default:
            if (is_digit(b)) cls = JsonSchemaNode::kNumber;
            else return reject();
        }
        s.cls = cls;
        push_event(&s, Event::kValueStart);
        switch (b) {
          case '{': stack_.push_back('{'); state_ = State::kObjectOpen; break;
          case '[': stack_.push_back('['); state_ = State::kArrayOpen; break;
          case '"': state_ = State::kInString; push_event(&s, Event::kValueByte); break;
          case '-': state_ = State::kNumMinus; push_event(&s, Event::kValueByte); break;
          case '0': state_ = State::kNumZero; push_event(&s, Event::kValueByte); break;
          case 't': case 'f': case 'n':
            lit_ = b == 't' ? 1 : b == 'f' ? 2 : 3;
            lit_pos_ = 1;
            state_ = State::kLiteral;
            push_event(&s, Event::kValueByte);
            break;
          default:
            state_ = State::kNumInt;
            push_event(&s, Event::kValueByte);
            break;
        }
        return s;
      }
      case State::kObjectOpen:
        if (is_ws(b)) return s;
        if (b == '"') {
          state_ = State::kInKey;
          push_event(&s, Event::kKeyStart);
          return s;
        }
        if (b == '}') {
          stack_.pop_back();
          push_event(&s, Event::kClose);
          end_value(&s);
          return s;
        }
        return reject();
      case State::kKeyStart:
        if (is_ws(b)) return s;
        if (b == '"') {
          state_ = State::kInKey;
          push_event(&s, Event::kKeyStart);
          return s;
        }
        return reject();
      case State::kInKey:
        if (!string_byte(b, &s, /*key=*/true)) return reject();
        return s;
      case State::kKeyEnd:
        if (is_ws(b)) return s;
        if (b == ':') {
          state_ = State::kValue;
          return s;
        }
        return reject();
      case State::kInString:
        if (!string_byte(b, &s, /*key=*/false)) return reject();
        return s;
      case State::kNumMinus:
        if (b == '0') state_ = State::kNumZero;
        else if (is_digit(b)) state_ = State::kNumInt;
        else return reject();
        push_event(&s, Event::kValueByte);
        return s;
      case State::kNumZero:
      case State::kNumInt:
      case State::kNumFrac:
      case State::kNumExp: {
        if (is_digit(b) && state_ != State::kNumZero) {
          push_event(&s, Event::kValueByte);
          return s;
        }
        if (b == '.' && (state_ == State::kNumZero || state_ == State::kNumInt)) {
          if (integer_only_) return reject();
          state_ = State::kNumDot;
          push_event(&s, Event::kValueByte);
          return s;
        }
        if ((b == 'e' || b == 'E') && state_ != State::kNumExp) {
          if (integer_only_) return reject();
          state_ = State::kNumE;
          push_event(&s, Event::kValueByte);
          return s;
        }
        if (is_digit(b)) return reject();  // a leading zero
        // The number ends here; the byte belongs to the enclosing state.
        end_value(&s);
        continue;
      }
      case State::kNumDot:
        if (!is_digit(b)) return reject();
        state_ = State::kNumFrac;
        push_event(&s, Event::kValueByte);
        return s;
      case State::kNumE:
        if (b == '+' || b == '-') {
          state_ = State::kNumESign;
          push_event(&s, Event::kValueByte);
          return s;
        }
        if (!is_digit(b)) return reject();
        state_ = State::kNumExp;
        push_event(&s, Event::kValueByte);
        return s;
      case State::kNumESign:
        if (!is_digit(b)) return reject();
        state_ = State::kNumExp;
        push_event(&s, Event::kValueByte);
        return s;
      case State::kLiteral: {
        const char* text = literal_text(lit_);
        if (b != static_cast<uint8_t>(text[lit_pos_])) return reject();
        ++lit_pos_;
        push_event(&s, Event::kValueByte);
        if (text[lit_pos_] == '\0') end_value(&s);
        return s;
      }
      case State::kValueEnd: {
        if (is_ws(b)) return s;
        const char t = top();
        if (b == ',') {
          push_event(&s, Event::kComma);
          state_ = t == '{' ? State::kKeyStart : State::kValue;
          return s;
        }
        if ((b == '}' && t == '{') || (b == ']' && t == '[')) {
          stack_.pop_back();
          push_event(&s, Event::kClose);
          end_value(&s);
          return s;
        }
        return reject();
      }
      case State::kDone:
        if (is_ws(b)) return s;
        return reject();
      case State::kCount:
        return reject();
    }
  }
  return reject();
}

bool JsonLexer::make_base(State state, char ctx, bool integer_only,
                          JsonLexer* out) {
  // A canonical prefix that lands the lexer in `state` with `ctx` as its
  // one open container (none at the top level). Keys exist only in
  // objects; a closer exists only inside a container.
  const std::string in_obj = "{\"k\":";
  const std::string in_arr = "[";
  std::string prefix;
  const std::string ctxp = ctx == '{' ? in_obj : ctx == '[' ? in_arr : "";
  switch (state) {
    case State::kValue: prefix = ctx == '[' ? "[1," : ctxp; break;
    case State::kArrayOpen: if (ctx != '[') return false; prefix = "["; break;
    case State::kObjectOpen: if (ctx != '{') return false; prefix = "{"; break;
    case State::kKeyStart: if (ctx != '{') return false; prefix = "{\"k\":1,"; break;
    case State::kInKey: if (ctx != '{') return false; prefix = "{\""; break;
    case State::kKeyEnd: if (ctx != '{') return false; prefix = "{\"k\""; break;
    case State::kInString: prefix = ctxp + "\""; break;
    case State::kNumMinus: prefix = ctxp + "-"; break;
    case State::kNumZero: prefix = ctxp + "0"; break;
    case State::kNumInt: prefix = ctxp + "1"; break;
    case State::kNumDot: prefix = ctxp + "1."; break;
    case State::kNumFrac: prefix = ctxp + "1.5"; break;
    case State::kNumE: prefix = ctxp + "1e"; break;
    case State::kNumESign: prefix = ctxp + "1e+"; break;
    case State::kNumExp: prefix = ctxp + "1e5"; break;
    case State::kValueEnd: if (ctx == 0) return false; prefix = ctxp + "1 "; break;
    case State::kDone: if (ctx != 0) return false; prefix = "1 "; break;
    default: return false;
  }
  JsonLexer lx;
  for (const char c : prefix)
    if (!lx.feed(static_cast<uint8_t>(c)).ok) return false;
  if (lx.state() != state || lx.top() != ctx) return false;
  lx.integer_only_ = integer_only;
  lx.ws_run_ = 0;  // a canonical base: the run is the machine's, applied by mask()
  *out = lx;
  return true;
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

namespace {

constexpr JsonLexer::State kTabled[] = {
    JsonLexer::State::kValue,      JsonLexer::State::kArrayOpen,
    JsonLexer::State::kObjectOpen, JsonLexer::State::kKeyStart,
    JsonLexer::State::kInKey,      JsonLexer::State::kKeyEnd,
    JsonLexer::State::kInString,   JsonLexer::State::kNumMinus,
    JsonLexer::State::kNumZero,    JsonLexer::State::kNumInt,
    JsonLexer::State::kNumDot,     JsonLexer::State::kNumFrac,
    JsonLexer::State::kNumE,       JsonLexer::State::kNumESign,
    JsonLexer::State::kNumExp,     JsonLexer::State::kValueEnd,
    JsonLexer::State::kDone,
};
constexpr int kTabledCount = static_cast<int>(sizeof(kTabled) / sizeof(kTabled[0]));
constexpr int kCtxCount = 3;  // 0, '{', '['
int ctx_index(char ctx) { return ctx == '{' ? 1 : ctx == '[' ? 2 : 0; }

void set_bit(std::vector<uint32_t>& w, int id) {
  w[static_cast<size_t>(id >> 5)] |= 1u << (id & 31);
}
void clear_bit(std::vector<uint32_t>& w, int id) {
  w[static_cast<size_t>(id >> 5)] &= ~(1u << (id & 31));
}
bool get_bit(const std::vector<uint32_t>& w, int id) {
  return (w[static_cast<size_t>(id >> 5)] >> (id & 31)) & 1u;
}
int popcount(const std::vector<uint32_t>& w) {
  int n = 0;
  for (const uint32_t x : w) n += __builtin_popcount(x);
  return n;
}

// The static simulation of a token from a base: 0 rejected, 1 allowed,
// 2 stack-dependent (the token popped the base's frame and went on).
int simulate_static(const JsonLexer& base, const std::string& text) {
  JsonLexer lx = base;
  const int depth0 = lx.depth();
  for (const char c : text) {
    const uint8_t b = static_cast<uint8_t>(c);
    if (lx.depth() < depth0 && !JsonLexer::is_ws(b)) return 2;
    if (!lx.feed(b).ok) return 0;
  }
  return 1;
}

}  // namespace

int JsonTables::index(JsonLexer::State s, char ctx, bool integer_only) {
  int si = -1;
  for (int i = 0; i < kTabledCount; ++i)
    if (kTabled[i] == s) si = i;
  if (si < 0) return -1;
  return (si * kCtxCount + ctx_index(ctx)) * 2 + (integer_only ? 1 : 0);
}

JsonTables::JsonTables(const GrammarVocab& vocab)
    : vocab_(vocab.vocab_size()), words_((vocab.vocab_size() + 31) / 32) {
  entries_.resize(static_cast<size_t>(kTabledCount * kCtxCount * 2));
  for (auto& c : class_) c.assign(static_cast<size_t>(words_), 0u);
  ws_.assign(static_cast<size_t>(words_), 0u);
  shapes_.resize(static_cast<size_t>(vocab_));
  lead_ws_ids_.assign(static_cast<size_t>(JsonLexer::kMaxWsRun) + 1, {});
  std::map<std::string, int32_t> prefix_ids, tail_ids, scalar_tail_ids;
  // Shapes and class masks.
  for (int id = 0; id < vocab_; ++id) {
    const std::string& text = vocab.text(id);
    Shape& sh = shapes_[static_cast<size_t>(id)];
    if (text.empty()) continue;
    size_t i = 0;
    while (i < text.size() && JsonLexer::is_ws(static_cast<uint8_t>(text[i]))) ++i;
    sh.lead_ws = static_cast<int32_t>(i);
    if (i > 0)
      lead_ws_ids_[static_cast<size_t>(std::min<size_t>(i, JsonLexer::kMaxWsRun))]
          .push_back(id);
    if (i < text.size()) {
      // Any numeric fragment, including fractions and exponent signs.
      // The lexical table checks its syntax at the current position.
      bool numeric = true;
      for (size_t j = i; j < text.size(); ++j)
        numeric = numeric && (is_digit(static_cast<uint8_t>(text[j])) ||
                               std::strchr("-+.eE", text[j]) != nullptr) && text[j] != '\0';
      sh.numeric = numeric;
      if (numeric) numeric_.push_back(id);
    }
    if (i == text.size()) {
      set_bit(ws_, id);
    } else {
      const uint8_t b = static_cast<uint8_t>(text[i]);
      int cls = -1;
      switch (b) {
        case '{': cls = 0; break;
        case '[': cls = 1; break;
        case '"': cls = 2; break;
        case '-': cls = 3; break;
        case 't': case 'f': cls = 5; break;
        case 'n': cls = 6; break;
        default: if (is_digit(b)) cls = 3; break;
      }
      if (cls >= 0) set_bit(class_[static_cast<size_t>(cls)], id);
      if (cls == 3) set_bit(class_[4], id);
    }
    // Unescaped quotes from esc = 0.
    int quotes = 0;
    size_t first_quote = std::string::npos;
    bool esc = false;
    for (size_t j = 0; j < text.size(); ++j) {
      const char c = text[j];
      if (esc) {
        esc = false;
        continue;
      }
      if (c == '\\') {
        esc = true;
        continue;
      }
      if (c == '"') {
        if (quotes == 0) first_quote = j;
        ++quotes;
      }
    }
    sh.quotes = quotes;
    if (quotes == 1) {
      const std::string prefix = text.substr(0, first_quote + 1);
      const std::string tail = text.substr(first_quote + 1);
      auto p = prefix_ids.emplace(prefix, static_cast<int32_t>(prefixes_.size()));
      if (p.second) prefixes_.push_back(prefix);
      sh.prefix_group = p.first->second;
      auto t = tail_ids.emplace(tail, static_cast<int32_t>(tails_.size()));
      if (t.second) tails_.push_back(tail);
      sh.tail_group = t.first->second;
      single_.push_back(id);
    } else if (quotes >= 2) {
      multi_.push_back(id);
    } else {
      bool structural = true;
      for (const char c : text)
        structural = structural && (std::strchr(" \t\n\r,:{}[]", c) != nullptr && c != '\0');
      sh.structural_only = structural;
      if (structural) {
        structural_.push_back(id);
      } else {
        size_t lead = 0;
        while (lead < text.size() && JsonLexer::is_ws(static_cast<uint8_t>(text[lead]))) ++lead;
        const size_t cut = text.find_first_of(",}] \t\n\r:{[", lead);
        if (cut != std::string::npos) {
          const std::string tail = text.substr(cut);
          auto t = scalar_tail_ids.emplace(tail, static_cast<int32_t>(scalar_tails_.size()));
          if (t.second) scalar_tails_.push_back(tail);
          sh.scalar_tail = t.first->second;
          scalar_tail_ids_.push_back(id);
        }
      }
    }
  }
  // Share token byte prefixes for content-dependent constraints. Once a
  // prefix is rejected, none of its descendant tokens need simulation.
  trie_.emplace_back();
  for (int id = 0; id < vocab_; ++id) {
    const auto& text = vocab.text(id);
    if (text.empty()) continue;
    int node = 0;
    for (unsigned char b : text) {
      int child = -1;
      for (const auto& edge : trie_[node].children) if (edge.first == b) { child = edge.second; break; }
      if (child < 0) {
        child = static_cast<int>(trie_.size());
        trie_[node].children.push_back({b, child});
        trie_.emplace_back();
      }
      node = child;
    }
    trie_[node].ids.push_back(id);
  }
  // The static entries: every existing base, every token, in parallel
  // over the bases (one-time startup work, ~1 s single-threaded).
  struct Job {
    int entry;
    JsonLexer base;
  };
  std::vector<Job> jobs;
  for (int si = 0; si < kTabledCount; ++si)
    for (int ci = 0; ci < kCtxCount; ++ci)
      for (int io = 0; io < 2; ++io) {
        const char ctx = ci == 1 ? '{' : ci == 2 ? '[' : 0;
        JsonLexer base;
        if (!JsonLexer::make_base(kTabled[si], ctx, io != 0, &base)) continue;
        jobs.push_back(Job{(si * kCtxCount + ci) * 2 + io, base});
      }
  const auto run = [&](size_t from, size_t to) {
    for (size_t j = from; j < to; ++j) {
      Entry& e = entries_[static_cast<size_t>(jobs[j].entry)];
      e.words.assign(static_cast<size_t>(words_), 0u);
      for (int id = 0; id < vocab_; ++id) {
        const std::string& text = vocab.text(id);
        if (text.empty()) continue;
        const int verdict = simulate_static(jobs[j].base, text);
        if (verdict == 1) set_bit(e.words, id);
        else if (verdict == 2) e.dynamic.push_back(id);
      }
    }
  };
  const unsigned hw = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  std::vector<std::thread> workers;
  const size_t per = (jobs.size() + hw - 1) / hw;
  for (unsigned t = 0; t < hw; ++t) {
    const size_t from = t * per;
    const size_t to = std::min(jobs.size(), from + per);
    if (from >= to) break;
    workers.emplace_back(run, from, to);
  }
  for (std::thread& w : workers) w.join();
}

const JsonTables::Entry& JsonTables::entry(JsonLexer::State s, char ctx,
                                            bool integer_only) const {
  const int i = index(s, ctx, integer_only);
  if (i < 0 || entries_[static_cast<size_t>(i)].words.empty())
    throw std::logic_error("JsonTables: state has no table");
  return entries_[static_cast<size_t>(i)];
}

const std::vector<uint32_t>& JsonTables::class_mask(uint32_t cls) const {
  switch (cls) {
    case JsonSchemaNode::kObject: return class_[0];
    case JsonSchemaNode::kArray: return class_[1];
    case JsonSchemaNode::kString: return class_[2];
    case JsonSchemaNode::kNumber: return class_[3];
    case JsonSchemaNode::kInteger: return class_[4];
    case JsonSchemaNode::kBoolean: return class_[5];
    default: return class_[6];
  }
}

// ---------------------------------------------------------------------------
// Machine
// ---------------------------------------------------------------------------

JsonMachine::JsonMachine(std::shared_ptr<const JsonSchema> schema,
                         const JsonTables* tables)
    : schema_(std::move(schema)), tables_(tables) {
  if (!schema_ || tables_ == nullptr) return;
  Cursor c;
  c.value_node = schema_->root;
  cursors_.push_back(std::move(c));
  alive_ = true;
}

const char* JsonMachine::state_name() const {
  if (!alive_) return "json-dead";
  switch (lexer_.state()) {
    case JsonLexer::State::kValue: return "json-value";
    case JsonLexer::State::kArrayOpen: return "json-array-open";
    case JsonLexer::State::kObjectOpen: return "json-object-open";
    case JsonLexer::State::kKeyStart: return "json-key-start";
    case JsonLexer::State::kInKey: return "json-key";
    case JsonLexer::State::kKeyEnd: return "json-key-end";
    case JsonLexer::State::kInString: return "json-string";
    case JsonLexer::State::kLiteral: return "json-literal";
    case JsonLexer::State::kValueEnd: return "json-value-end";
    case JsonLexer::State::kDone: return "json-done";
    default: return "json-number";
  }
}

int JsonMachine::expected_node(const Cursor& c) const {
  return c.stack.empty() ? c.value_node : c.stack.back().value_node;
}

void JsonMachine::leaves(int node, std::vector<int>* out) const {
  if (node == JsonSchemaNode::kAny) {
    out->push_back(JsonSchemaNode::kAny);
    return;
  }
  const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(node)];
  if (n.any_of.empty()) {
    out->push_back(node);
    return;
  }
  for (const int alt : n.any_of) leaves(alt, out);
}

bool JsonMachine::targets_integral(const Cursor& c) const {
  const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(c.target_node)];
  for (const int t : c.targets)
    if (n.enum_texts[static_cast<size_t>(t)].find_first_of(".eE") != std::string::npos)
      return false;
  return true;
}

namespace {

// A container frame imposes nothing further on its contents (its type and
// enum were checked at its value start).
bool frame_trivial(const JsonSchemaNode& n) {
  return n.property_names.empty() && n.additional == JsonSchemaNode::kAny &&
         n.items == JsonSchemaNode::kAny && n.min_items == 0 && n.max_items < 0;
}

// Exact decimal divisibility. Separating powers of 2 and 5 avoids either
// floating-point rounding or allocating 10^exponent for generated exponents.
bool multiple_of(std::string_view text, const std::string& step) {
  if (step.empty()) return true;
  auto a = DecimalNumber::parse(text), b = DecimalNumber::parse(step);
  if (a.digits.empty()) return true;
  int64_t scale = a.exponent - static_cast<int64_t>(a.digits.size()) -
                  b.exponent + static_cast<int64_t>(b.digits.size());
  const auto divide = [](std::string* digits, int divisor) {
    std::string result; int remainder = 0;
    for (char c : *digits) {
      int value = remainder * 10 + c - '0';
      if (!result.empty() || value / divisor) result += static_cast<char>('0' + value / divisor);
      remainder = value % divisor;
    }
    if (!remainder) *digits = result.empty() ? "0" : result;
    return remainder;
  };
  for (int factor : {2, 5}) {
    int64_t av = 0, bv = 0;
    while (!divide(&a.digits, factor)) ++av;
    while (!divide(&b.digits, factor)) ++bv;
    if (av + scale < bv) return false;
  }
  // The remaining divisor is coprime to ten. Decimal long division needs
  // at most nine subtractions for each input digit.
  std::string remainder = "0";
  const auto ge = [](const std::string& x, const std::string& y) {
    return x.size() != y.size() ? x.size() > y.size() : x >= y;
  };
  for (char digit : a.digits) {
    if (remainder == "0") remainder.clear();
    remainder += digit;
    while (ge(remainder, b.digits)) {
      int borrow = 0;
      for (size_t i = 0; i < remainder.size(); ++i) {
        size_t j = remainder.size() - i - 1;
        int diff = remainder[j] - '0' - borrow -
                   (i < b.digits.size() ? b.digits[b.digits.size()-i-1]-'0' : 0);
        borrow = diff < 0; remainder[j] = static_cast<char>('0' + diff + 10 * borrow);
      }
      auto first = remainder.find_first_not_of('0');
      remainder = first == std::string::npos ? "0" : remainder.substr(first);
    }
  }
  return remainder == "0";
}

bool multiple_reachable(const IntegerPrefix& prefix, const IntegerBounds& bounds,
                        const std::string& step) {
  if (step.empty() || !bounds.active()) return true;
  auto number = DecimalNumber::parse(step);
  int64_t power = number.exponent - static_cast<int64_t>(number.digits.size()) + 1;
  if (power < 0) {
    for (int factor : {2, 5}) {
      for (int64_t count = 0; count < -power; ++count) {
        std::string quotient; int rem = 0;
        for (char c : number.digits) {
          int v = rem * 10 + c - '0';
          if (!quotient.empty() || v / factor) quotient += static_cast<char>('0' + v / factor);
          rem = v % factor;
        }
        if (rem) break;
        number.digits = std::move(quotient);
      }
    }
  } else if (power <= 20) number.digits.append(static_cast<size_t>(power), '0');
  else number.digits.assign(21, '9');
  __extension__ using Wide = __int128;
  Wide divisor = 0;
  for (char c : number.digits) { divisor = divisor * 10 + c - '0'; if (divisor > (Wide{1} << 64)) break; }
  Wide lower = bounds.has_min ? bounds.min : INT64_MIN;
  Wide upper = bounds.has_max ? bounds.max : INT64_MAX;
  if (prefix.negative) { Wide old = lower; lower = -upper; upper = -old; }
  lower = std::max(Wide{0}, lower);
  if (upper < lower || prefix.huge) return false;
  const auto contains = [&](Wide lo, Wide hi) {
    lo = std::max(lo, lower); hi = std::min(hi, upper);
    return lo <= hi && (lo + divisor - 1) / divisor <= hi / divisor;
  };
  if (!prefix.digits) return contains(lower, upper);
  if (!prefix.magnitude) return contains(0, 0);
  for (Wide scale = 1; Wide{prefix.magnitude} * scale <= upper; scale *= 10)
    if (contains(Wide{prefix.magnitude} * scale, (Wide{prefix.magnitude} + 1) * scale - 1)) return true;
  return false;
}

}  // namespace

bool JsonMachine::sweep() {
  const size_t before = cursors_.size();
  cursors_.erase(std::remove_if(cursors_.begin(), cursors_.end(),
                                [](const Cursor& c) { return c.dead; }),
                 cursors_.end());
  if (cursors_.size() != before) ++epoch_;  // the frontier changed
  return !cursors_.empty();
}

bool JsonMachine::done() const {
  if (!alive_ || !lexer_.done()) return false;
  if (lexer_.state() == JsonLexer::State::kDone) return true;
  // A top-level number: complete where some cursor accepts the digits as
  // they stand — an enum target spelled out, a bounded value inside its
  // range (the cursors are an anyOf's alternatives: union semantics).
  for (const Cursor& c : cursors_) {
    if (c.bound_node >= 0) {
      const auto& n = schema_->nodes[static_cast<size_t>(c.bound_node)];
      if (n.decimal_bounds.active() ? !c.decimal.within(n.decimal_bounds)
                                    : !c.number.within(n.bounds)) continue;
      if (!multiple_of(c.decimal.text, n.multiple_of)) continue;
    }
    if (c.target_node >= 0) {
      const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(c.target_node)];
      bool complete = false;
      for (const int t : c.targets)
        complete = complete || n.enum_texts[static_cast<size_t>(t)].size() == c.target_pos;
      if (!complete) continue;
    }
    return true;
  }
  return false;
}

bool JsonMachine::on_value_start(uint32_t cls, uint8_t b) {
  ++epoch_;
  std::vector<Cursor> next;
  for (const Cursor& c : cursors_) {
    if (!c.stack.empty() && c.stack.back().kind == '[' &&
        c.stack.back().node != JsonSchemaNode::kAny) {
      const JsonSchemaNode& arr = schema_->nodes[static_cast<size_t>(c.stack.back().node)];
      if (arr.max_items >= 0 && c.stack.back().count >= arr.max_items) continue;
    }
    std::vector<int> options;
    leaves(expected_node(c), &options);
    for (const int leaf : options) {
      Cursor nc = c;
      nc.targets.clear();
      nc.target_node = -1;
      nc.target_pos = 0;
      nc.integer_only = false;
      nc.bound_node = -1;
      nc.number = IntegerPrefix{};
      nc.decimal = DecimalPrefix{};
      nc.string_node = -1;
      nc.string_json.clear();
      if (leaf == JsonSchemaNode::kAny) {
        if (cls == JsonSchemaNode::kObject || cls == JsonSchemaNode::kArray) {
          Frame f;
          f.kind = cls == JsonSchemaNode::kObject ? '{' : '[';
          nc.stack.push_back(std::move(f));
        }
        next.push_back(std::move(nc));
        continue;
      }
      const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(leaf)];
      const uint32_t want = cls == JsonSchemaNode::kNumber
                                ? (JsonSchemaNode::kNumber | JsonSchemaNode::kInteger)
                                : cls;
      if ((n.types & want) == 0) continue;
      if (n.has_enum) {
        for (size_t t = 0; t < n.enum_texts.size(); ++t)
          if (!n.enum_texts[t].empty() &&
              static_cast<uint8_t>(n.enum_texts[t][0]) == b)
            nc.targets.push_back(static_cast<int>(t));
        if (nc.targets.empty()) continue;
        nc.target_node = leaf;
        nc.target_pos = 0;  // the kValueByte of this same byte advances it
      }
      if (cls == JsonSchemaNode::kObject || cls == JsonSchemaNode::kArray) {
        Frame f;
        f.node = leaf;
        f.kind = cls == JsonSchemaNode::kObject ? '{' : '[';
        if (n.has_enum) {
          f.enum_targets = std::move(nc.targets);
          f.enum_pos = 1;  // opening delimiter is this byte, already matched
          nc.target_node = -1;
        }
        f.used.assign(n.property_names.size(), false);
        f.value_node = f.kind == '[' ? n.items : JsonSchemaNode::kAny;
        nc.stack.push_back(std::move(f));
      } else if (cls == JsonSchemaNode::kNumber) {
        nc.integer_only = (n.types & JsonSchemaNode::kNumber) == 0;
        // An enum's number targets decide the bytes themselves; the flag
        // is the tables' integer answer, so it follows the live targets
        // (2026-09-07: beside a bounded integer alternative the mask used
        // to admit a '.' every cursor then refused).
        if (nc.target_node >= 0) nc.integer_only = targets_integral(nc);
        // The kValueByte of this same byte pushes the first digit (or the
        // sign) and asks whether the range is still reachable.
        if (n.bounds.active() || n.decimal_bounds.active() || !n.multiple_of.empty()) nc.bound_node = leaf;
      } else if (cls == JsonSchemaNode::kString && !n.strings.empty()) {
        nc.string_node = leaf;
      }
      next.push_back(std::move(nc));
    }
  }
  cursors_.swap(next);
  return !cursors_.empty();
}

bool JsonMachine::on_value_byte(uint8_t b) {
  for (Cursor& c : cursors_) {
    if (c.integer_only && (b == '.' || b == 'e' || b == 'E')) {
      c.dead = true;
      continue;
    }
    if (c.bound_node >= 0) {
      const auto& n = schema_->nodes[static_cast<size_t>(c.bound_node)];
      bool reachable;
      c.decimal.push(b);
      if (n.decimal_bounds.active()) {
        reachable = c.decimal.can_reach(n.decimal_bounds);
      } else {
        c.number.push(b);
        reachable = c.number.can_reach(n.bounds) && multiple_reachable(c.number, n.bounds, n.multiple_of);
      }
      if (!reachable) {
        c.dead = true;
        continue;
      }
    }
    if (c.string_node >= 0) {
      c.string_json += static_cast<char>(b);
      std::string decoded; bool complete = false;
      if (!decode_json_string_prefix(c.string_json, &decoded, &complete)) { c.dead = true; continue; }
      for (const auto& constraint : schema_->nodes[c.string_node].strings)
        if (!constraint->accepts(decoded, complete)) { c.dead = true; break; }
      if (c.dead) continue;
    }
    if (c.target_node < 0) continue;
    const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(c.target_node)];
    size_t w = 0;
    for (const int t : c.targets) {
      const std::string& text = n.enum_texts[static_cast<size_t>(t)];
      if (c.target_pos < text.size() &&
          static_cast<uint8_t>(text[c.target_pos]) == b)
        c.targets[w++] = t;
    }
    c.targets.resize(w);
    ++c.target_pos;
    if (w == 0) c.dead = true;
    else if (!c.integer_only && (n.types & JsonSchemaNode::kNumber) != 0)
      c.integer_only = targets_integral(c);
  }
  return sweep();
}

bool JsonMachine::on_value_end() {
  ++epoch_;
  for (Cursor& c : cursors_) {
    if (c.string_node >= 0) {
      std::string decoded; bool complete = false;
      if (!decode_json_string_prefix(c.string_json, &decoded, &complete) || !complete) {
        c.dead = true;
        continue;
      }
    }
    if (c.bound_node >= 0) {
      const auto& n = schema_->nodes[static_cast<size_t>(c.bound_node)];
      if (n.decimal_bounds.active() ? !c.decimal.within(n.decimal_bounds)
                                    : !c.number.within(n.bounds)) {
        c.dead = true;
        continue;
      }
      if (!multiple_of(c.decimal.text, n.multiple_of)) { c.dead = true; continue; }
      c.bound_node = -1;
      c.number = IntegerPrefix{};
      c.decimal = DecimalPrefix{};
    }
    c.string_node = -1;
    c.string_json.clear();
    if (c.target_node >= 0) {
      const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(c.target_node)];
      bool complete = false;
      for (const int t : c.targets)
        complete = complete || n.enum_texts[static_cast<size_t>(t)].size() == c.target_pos;
      if (!complete) {
        c.dead = true;
        continue;
      }
      c.targets.clear();
      c.target_node = -1;
      c.target_pos = 0;
    }
    c.integer_only = false;
    if (!c.stack.empty() && c.stack.back().kind == '[') ++c.stack.back().count;
  }
  return sweep();
}

bool JsonMachine::on_key_start() {
  ++epoch_;
  for (Cursor& c : cursors_) {
    Frame& f = c.stack.back();
    f.key.clear();
    if (f.node == JsonSchemaNode::kAny) continue;
    const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
    if (n.additional == JsonSchemaNode::kClosed) {
      bool any_unused = false;
      for (size_t i = 0; i < f.used.size(); ++i) any_unused = any_unused || !f.used[i];
      if (!any_unused) c.dead = true;
    }
  }
  return sweep();
}

bool JsonMachine::on_key_byte(uint8_t b) {
  for (Cursor& c : cursors_) {
    Frame& f = c.stack.back();
    f.key.push_back(static_cast<char>(b));
    if (f.node == JsonSchemaNode::kAny) continue;
    const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
    if (n.additional != JsonSchemaNode::kClosed) continue;
    bool prefix = false;
    for (size_t i = 0; i < n.property_names.size() && !prefix; ++i)
      prefix = !f.used[i] && n.property_names[i].size() >= f.key.size() &&
               n.property_names[i].compare(0, f.key.size(), f.key) == 0;
    if (!prefix) c.dead = true;
  }
  return sweep();
}

bool JsonMachine::on_key_end() {
  ++epoch_;
  for (Cursor& c : cursors_) {
    Frame& f = c.stack.back();
    if (f.node == JsonSchemaNode::kAny) {
      f.value_node = JsonSchemaNode::kAny;
      continue;
    }
    const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
    int found = -1;
    for (size_t i = 0; i < n.property_names.size(); ++i)
      if (n.property_names[i] == f.key) found = static_cast<int>(i);
    if (found >= 0) {
      if (f.used[static_cast<size_t>(found)]) {
        c.dead = true;  // a duplicate key
        continue;
      }
      f.used[static_cast<size_t>(found)] = true;
      f.value_node = n.property_nodes[static_cast<size_t>(found)];
    } else if (n.additional == JsonSchemaNode::kClosed) {
      c.dead = true;
    } else {
      f.value_node = n.additional;
    }
  }
  return sweep();
}

bool JsonMachine::on_comma() {
  ++epoch_;
  for (Cursor& c : cursors_) {
    const Frame& f = c.stack.back();
    if (f.node == JsonSchemaNode::kAny) continue;
    const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
    if (f.kind == '{') {
      if (n.additional == JsonSchemaNode::kClosed) {
        bool any_unused = false;
        for (size_t i = 0; i < f.used.size(); ++i) any_unused = any_unused || !f.used[i];
        if (!any_unused) c.dead = true;
      }
    } else if (n.max_items >= 0 && f.count >= n.max_items) {
      c.dead = true;
    }
  }
  return sweep();
}

bool JsonMachine::on_close() {
  ++epoch_;
  for (Cursor& c : cursors_) {
    const Frame& f = c.stack.back();
    if (f.node != JsonSchemaNode::kAny) {
      const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
      if (n.has_enum) {
        bool complete = false;
        for (int target : f.enum_targets)
          complete = complete || n.enum_texts[target].size() == f.enum_pos;
        if (!complete) c.dead = true;
      }
      if (f.kind == '{') {
        for (size_t i = 0; i < f.used.size(); ++i)
          if (n.property_required[i] && !f.used[i]) c.dead = true;
      } else if (f.count < n.min_items) {
        c.dead = true;
      }
    }
    if (!c.dead) c.stack.pop_back();
  }
  return sweep();
}

bool JsonMachine::apply(const JsonLexer::Step& s, uint8_t b) {
  for (const JsonLexer::Event e : s.events) {
    bool ok = true;
    switch (e) {
      case JsonLexer::Event::kNone: return true;
      case JsonLexer::Event::kValueStart: ok = on_value_start(s.cls, b); break;
      case JsonLexer::Event::kValueByte: ok = on_value_byte(b); break;
      case JsonLexer::Event::kValueEnd: ok = on_value_end(); break;
      case JsonLexer::Event::kKeyStart: ok = on_key_start(); break;
      case JsonLexer::Event::kKeyByte: ok = on_key_byte(b); break;
      case JsonLexer::Event::kKeyEnd: ok = on_key_end(); break;
      case JsonLexer::Event::kComma: ok = on_comma(); break;
      case JsonLexer::Event::kClose: ok = on_close(); break;
    }
    if (!ok) return false;
  }
  return true;
}

bool JsonMachine::feed(uint8_t b) {
  if (!alive_) return false;
  // Unlike scalar enum cursors, these stay active through nested JSON
  // lexer events. The closing delimiter is matched before its frame pops.
  if (schema_->container_enums) for (auto& cursor : cursors_) {
    for (auto& frame : cursor.stack) {
      if (frame.enum_targets.empty()) continue;
      const auto& targets = schema_->nodes[frame.node].enum_texts;
      std::erase_if(frame.enum_targets, [&](int target) {
        return frame.enum_pos >= targets[target].size() ||
               static_cast<uint8_t>(targets[target][frame.enum_pos]) != b;
      });
      ++frame.enum_pos;
      if (frame.enum_targets.empty()) { cursor.dead = true; break; }
    }
  }
  if (schema_->container_enums && !sweep()) { alive_ = false; return false; }
  const JsonLexer::Step s = lexer_.feed(b);
  if (!s.ok || !apply(s, b)) {
    alive_ = false;
    return false;
  }
  return true;
}

bool JsonMachine::simulate(const std::string& text) const {
  if (text.empty()) return false;
  JsonMachine copy = *this;
  copy.cache_ = Cache{};
  for (const char c : text)
    if (!copy.feed(static_cast<uint8_t>(c))) return false;
  return true;
}

bool JsonMachine::allows(const GrammarVocab& vocab, int64_t id) const {
  if (!alive_) return false;
  return simulate(vocab.text(id));
}

bool JsonMachine::content_constrained() const {
  const JsonLexer::State st = lexer_.state();
  if (st == JsonLexer::State::kInKey) {
    for (const Cursor& c : cursors_) {
      const Frame& f = c.stack.back();
      if (f.node == JsonSchemaNode::kAny) return false;
      if (schema_->nodes[static_cast<size_t>(f.node)].additional !=
          JsonSchemaNode::kClosed)
        return false;
    }
    return !cursors_.empty();
  }
  if (st == JsonLexer::State::kInString || lexer_.in_number() || lexer_.in_literal()) {
    for (const Cursor& c : cursors_)
      if (c.target_node < 0) return false;
    return !cursors_.empty();
  }
  return false;
}

void JsonMachine::content_continuations(std::vector<std::string>* rem) const {
  const JsonLexer::State st = lexer_.state();
  for (const Cursor& c : cursors_) {
    if (st == JsonLexer::State::kInKey) {
      const Frame& f = c.stack.back();
      const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(f.node)];
      for (size_t i = 0; i < n.property_names.size(); ++i) {
        if (f.used[i]) continue;
        const std::string& p = n.property_names[i];
        if (p.size() < f.key.size() || p.compare(0, f.key.size(), f.key) != 0) continue;
        rem->push_back(p.substr(f.key.size()) + "\"");  // the rest, then the close
      }
    } else if (c.target_node >= 0) {
      const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(c.target_node)];
      for (const int t : c.targets) {
        const std::string& text = n.enum_texts[static_cast<size_t>(t)];
        if (c.target_pos <= text.size()) rem->push_back(text.substr(c.target_pos));
      }
    }
  }
  std::sort(rem->begin(), rem->end());
  rem->erase(std::unique(rem->begin(), rem->end()), rem->end());
}

bool JsonMachine::schema_constrains_here() const {
  for (const Cursor& c : cursors_) {
    if (c.stack.empty()) return true;  // the root value's class
    for (const Frame& f : c.stack) {
      if (f.node == JsonSchemaNode::kAny) continue;
      if (!frame_trivial(schema_->nodes[static_cast<size_t>(f.node)])) return true;
    }
    if (c.target_node >= 0 || c.integer_only || c.bound_node >= 0) return true;
  }
  return false;
}

bool JsonMachine::extended_constraints_live() const {
  for (const auto& n : schema_->nodes)
    if (!n.strings.empty() || !n.multiple_of.empty() ||
        (n.has_enum && (n.types & (JsonSchemaNode::kObject | JsonSchemaNode::kArray)))) return true;
  return false;
}

#if defined(DGPP_JSON_FREE_STRING_MASK)
bool JsonMachine::unconstrained_value_string() const {
  // Container enums retain the spelling of nested strings. Keys and escape
  // states stay on the trie path, as does any constrained cursor in an anyOf.
  if (!alive_ || !schema_ || schema_->container_enums || cursors_.empty() ||
      lexer_.state() != JsonLexer::State::kInString || lexer_.in_escape())
    return false;
  for (const Cursor& c : cursors_)
    if (c.dead || c.string_node >= 0 || c.target_node >= 0 || c.bound_node >= 0) return false;
  return true;
}
#endif

bool JsonMachine::bounds_live_here(bool value_start) const {
  for (const Cursor& c : cursors_) {
    if (!value_start) {
      if (c.bound_node >= 0) return true;
      continue;
    }
    if (!c.stack.empty() && c.stack.back().kind == '[' &&
        c.stack.back().node != JsonSchemaNode::kAny) {
      const JsonSchemaNode& arr = schema_->nodes[static_cast<size_t>(c.stack.back().node)];
      if (arr.max_items >= 0 && c.stack.back().count >= arr.max_items) continue;
    }
    std::vector<int> options;
    leaves(expected_node(c), &options);
    for (const int leaf : options) {
      if (leaf == JsonSchemaNode::kAny) continue;
      const auto& n = schema_->nodes[static_cast<size_t>(leaf)];
      if (n.bounds.active() || n.decimal_bounds.active()) return true;
    }
  }
  return false;
}

bool JsonMachine::numeric_token_ok(const std::string& text, bool value_start) const {
  size_t i = 0;
  while (i < text.size() && JsonLexer::is_ws(static_cast<uint8_t>(text[i]))) ++i;
  if (!value_start) {
    // Inside the number (a leading whitespace would have ended it, and the
    // static table refused such a token already). An enum target beside a
    // bound is the one shape the arithmetic does not cover: simulate.
    for (const Cursor& c : cursors_)
      if (c.target_node >= 0 ||
          (c.bound_node >= 0 && schema_->nodes[static_cast<size_t>(c.bound_node)].decimal_bounds.active()))
        return simulate(text);
    for (const Cursor& c : cursors_) {
      if (c.bound_node < 0) return true;
      IntegerPrefix p = c.number;
      for (size_t j = i; j < text.size(); ++j) p.push(static_cast<uint8_t>(text[j]));
      if (p.can_reach(schema_->nodes[static_cast<size_t>(c.bound_node)].bounds)) return true;
    }
    return false;
  }
  for (const Cursor& c : cursors_) {
    if (!c.stack.empty() && c.stack.back().kind == '[' &&
        c.stack.back().node != JsonSchemaNode::kAny) {
      const JsonSchemaNode& arr = schema_->nodes[static_cast<size_t>(c.stack.back().node)];
      if (arr.max_items >= 0 && c.stack.back().count >= arr.max_items) continue;
    }
    std::vector<int> options;
    leaves(expected_node(c), &options);
    for (const int leaf : options) {
      if (leaf == JsonSchemaNode::kAny) return true;
      const JsonSchemaNode& n = schema_->nodes[static_cast<size_t>(leaf)];
      if ((n.types & (JsonSchemaNode::kNumber | JsonSchemaNode::kInteger)) == 0) continue;
      if (n.has_enum) return simulate(text);
      if (n.decimal_bounds.active()) return simulate(text);
      if (!n.bounds.active()) return true;
      IntegerPrefix p;
      for (size_t j = i; j < text.size(); ++j) p.push(static_cast<uint8_t>(text[j]));
      if (p.can_reach(n.bounds)) return true;
    }
  }
  return false;
}

uint32_t JsonMachine::allowed_classes_here() const {
  uint32_t cls = 0;
  for (const Cursor& c : cursors_) {
    if (!c.stack.empty() && c.stack.back().kind == '[' &&
        c.stack.back().node != JsonSchemaNode::kAny) {
      const JsonSchemaNode& arr = schema_->nodes[static_cast<size_t>(c.stack.back().node)];
      if (arr.max_items >= 0 && c.stack.back().count >= arr.max_items) continue;
    }
    std::vector<int> options;
    leaves(expected_node(c), &options);
    for (const int leaf : options) {
      if (leaf == JsonSchemaNode::kAny) return JsonSchemaNode::kAnyClass;
      cls |= schema_->nodes[static_cast<size_t>(leaf)].types;
    }
  }
  return cls;
}

void JsonMachine::mask_brute_force(const GrammarVocab& vocab, TokenMask* out) const {
  const int words = tables_ != nullptr ? tables_->words() : (vocab.vocab_size() + 31) / 32;
  out->vocab = vocab.vocab_size();
  out->words.assign(static_cast<size_t>(words), 0u);
  out->allowed = 0;
  if (!alive_) return;
  for (int id = 0; id < vocab.vocab_size(); ++id)
    if (simulate(vocab.text(id))) set_bit(out->words, id);
  out->allowed = popcount(out->words);
}

void JsonMachine::mask(const GrammarVocab& vocab, TokenMask* out) const {
  // Content constraints cannot use the structural representatives. Traverse
  // the vocabulary trie instead, evaluating each shared byte prefix once.
  if (extended_constraints_live()
#if defined(DGPP_JSON_FREE_STRING_MASK)
      && !unconstrained_value_string()
#endif
  ) {
    out->vocab = vocab.vocab_size();
    out->words.assign(static_cast<size_t>(tables_->words()), 0u);
    out->allowed = 0;
    if (!alive_) return;
    const auto& trie = tables_->token_trie();
    const auto visit = [&](const auto& self, int node, const JsonMachine& machine) -> void {
      for (int id : trie[node].ids) set_bit(out->words, id);
      for (const auto& [byte, child] : trie[node].children) {
        JsonMachine next = machine;
        if (next.feed(byte)) self(self, child, next);
      }
    };
#if defined(DGPP_JSON_FREE_STRING_MASK)
    // Trie descendants only feed bytes; they never use the table-mask cache.
    // A preceding free string can populate it. Do not copy those answer vectors
    // at every trie edge, and retain the caller's cache for its next mask.
    JsonMachine root = *this;
    root.cache_ = Cache{};
    visit(visit, 0, root);
#else
    visit(visit, 0, *this);
#endif
    out->allowed = popcount(out->words);
    return;
  }
  const int words = tables_->words();
  out->vocab = vocab.vocab_size();
  out->words.assign(static_cast<size_t>(words), 0u);
  out->allowed = 0;
  if (!alive_) return;
  const JsonLexer::State st = lexer_.state();
  // The whitespace budget left at this position: a token whose leading
  // whitespace run exceeds it would be rejected by the lexer's cap.
  const auto apply_ws_budget = [&] {
    const int run = lexer_.ws_run();
    if (run <= 0 || lexer_.in_string()) return;
    const int budget = JsonLexer::kMaxWsRun - run;
    for (int n = budget + 1; n <= JsonLexer::kMaxWsRun; ++n)
      for (const int32_t id : tables_->lead_ws_ids(n)) clear_bit(out->words, id);
  };
  const auto finish = [&] {
    apply_ws_budget();
    out->allowed = popcount(out->words);
  };
  if (st == JsonLexer::State::kDone) {
    out->words = tables_->whitespace_mask();
    finish();
    return;
  }

  // Content the schema constrains byte by byte: a closed object's key, an
  // enum text (in any escape state). The remainders name the bytes that
  // may come; a token running past a remainder is simulated.
  if (content_constrained()) {
    std::vector<std::string> rems;
    content_continuations(&rems);
    bool complete = false;
    for (const std::string& r : rems) {
      if (r.empty()) {
        complete = true;
        continue;
      }
      for (const int32_t id : vocab.ids_starting_with(static_cast<unsigned char>(r[0]))) {
        const std::string& text = vocab.text(id);
        if (get_bit(out->words, id)) continue;
        if (text.size() <= r.size()) {
          if (r.compare(0, text.size(), text) == 0) set_bit(out->words, id);
        } else if (text.compare(0, r.size(), r) == 0 && simulate(text)) {
          set_bit(out->words, id);
        }
      }
    }
    // A number or literal target already complete: the delimiters decide.
    if (complete)
      for (const int32_t id : tables_->structural_ids())
        if (simulate(vocab.text(id))) set_bit(out->words, id);
    finish();
    return;
  }

  // Positions the tables do not cover: the middle of a literal, an escape.
  if (lexer_.in_literal()) {
    const std::string rest = lexer_.literal_rest();
    for (const int32_t id : vocab.ids_starting_with(static_cast<unsigned char>(rest[0]))) {
      const std::string& text = vocab.text(id);
      if (text.size() <= rest.size()) {
        if (rest.compare(0, text.size(), text) == 0) set_bit(out->words, id);
      } else if (text.compare(0, rest.size(), rest) == 0 && simulate(text)) {
        set_bit(out->words, id);
      }
    }
    finish();
    return;
  }
  if (lexer_.in_escape()) {
    // The escape's own bytes, then free string content; a token that
    // closes the string is simulated for what follows.
    const std::string firsts =
        lexer_.escape() == 1 ? std::string("\"\\/bfnrtu")
                             : std::string("0123456789abcdefABCDEF");
    for (const char f : firsts)
      for (const int32_t id : vocab.ids_starting_with(static_cast<unsigned char>(f))) {
        const std::string& text = vocab.text(id);
        uint8_t esc = lexer_.escape();
        bool ok = true, closes = false;
        for (const char ch : text) {
          const uint8_t b = static_cast<uint8_t>(ch);
          if (esc == 1) {
            if (std::strchr("\"\\/bfnrtu", ch) == nullptr || ch == '\0') { ok = false; break; }
            esc = ch == 'u' ? 2 : 0;
          } else if (esc >= 2) {
            if (!is_hex(b)) { ok = false; break; }
            esc = esc == 5 ? 0 : static_cast<uint8_t>(esc + 1);
          } else if (b == '"') {
            closes = true;
            break;
          } else if (b == '\\') {
            esc = 1;
          } else if (b < 0x20) {
            ok = false;
            break;
          }
        }
        if (!ok) continue;
        if (!closes || simulate(text)) set_bit(out->words, id);
      }
    finish();
    return;
  }

  const bool value_start = st == JsonLexer::State::kValue ||
                           st == JsonLexer::State::kArrayOpen;
  const bool in_string = lexer_.in_string();
  const bool constrained = schema_constrains_here();
  const uint32_t cls = value_start && constrained ? allowed_classes_here()
                                                  : JsonSchemaNode::kAnyClass;
  // The integer table: inside a number every cursor admits digits only;
  // at a value start the admitted classes hold kInteger without kNumber.
  bool all_integer = !cursors_.empty();
  for (const Cursor& c : cursors_) all_integer = all_integer && c.integer_only;
  if (value_start)
    all_integer = (cls & JsonSchemaNode::kInteger) != 0 && (cls & JsonSchemaNode::kNumber) == 0;
  const JsonTables::Entry& e = tables_->entry(st, lexer_.top(), all_integer);
  out->words = e.words;
  if (!constrained) {
    for (const int32_t id : e.dynamic)
      if (simulate(vocab.text(id))) set_bit(out->words, id);
    finish();
    return;
  }

  // Schema mode. Inside a string the answer only changes at the next
  // structural event (or when the frontier shrinks): reuse it.
  if (in_string && cache_.epoch == epoch_ && cache_.state == st &&
      cache_.top == lexer_.top()) {
    for (size_t i = 0; i < tables_->single_quote_ids().size(); ++i) {
      const int32_t id = tables_->single_quote_ids()[i];
      if (get_bit(out->words, id) && !cache_.tail_ok[i]) clear_bit(out->words, id);
    }
    for (size_t i = 0; i < tables_->multi_quote_ids().size(); ++i) {
      const int32_t id = tables_->multi_quote_ids()[i];
      if (get_bit(out->words, id) && !cache_.multi_ok[i]) clear_bit(out->words, id);
    }
    for (const int32_t id : e.dynamic)
      if (simulate(vocab.text(id))) set_bit(out->words, id);
    finish();
    return;
  }

  // 1. A value start under a type filter: only the admitted classes (and
  //    whitespace, and the structural tokens simulated below).
  if (value_start) {
    if (cls != JsonSchemaNode::kAnyClass) {
      std::vector<uint32_t> keep = tables_->whitespace_mask();
      for (const uint32_t bit : {JsonSchemaNode::kObject, JsonSchemaNode::kArray,
                                 JsonSchemaNode::kString, JsonSchemaNode::kNumber,
                                 JsonSchemaNode::kInteger, JsonSchemaNode::kBoolean,
                                 JsonSchemaNode::kNull}) {
        if ((cls & bit) == 0) continue;
        const std::vector<uint32_t>& m = tables_->class_mask(bit);
        for (size_t w = 0; w < keep.size(); ++w) keep[w] |= m[w];
      }
      for (const int32_t id : tables_->structural_ids()) set_bit(keep, id);
      for (size_t w = 0; w < keep.size(); ++w) out->words[w] &= keep[w];
    }
  }
  // 2. Every token whose structure can reach a schema check: the pure
  //    structure and scalar-tail tokens individually, the single-quote
  //    tokens by representative (their free content is the tables'), the
  //    multi-quote tokens individually.
  //    Inside a string these are content; only a quote matters there.
  if (!in_string) {
    for (const int32_t id : tables_->structural_ids())
      if (get_bit(out->words, id) && !simulate(vocab.text(id))) clear_bit(out->words, id);
    for (const int32_t id : tables_->scalar_tail_ids())
      if (get_bit(out->words, id) && !simulate(vocab.text(id))) clear_bit(out->words, id);
  }
  // An enum expected at a value start constrains the scalar content too:
  // a token's leading value bytes must be a prefix of a target.
  bool enum_ahead = false;
  if (value_start)
    for (const Cursor& c : cursors_) {
      std::vector<int> options;
      leaves(expected_node(c), &options);
      for (const int leaf : options)
        enum_ahead = enum_ahead || (leaf != JsonSchemaNode::kAny &&
                                    schema_->nodes[static_cast<size_t>(leaf)].has_enum);
    }
  // 2b. A bounded integer being spelled, or expected here:
  //     the numeric tokens' static answer is lexical, so each is judged by
  //     its digits against every cursor's range (the enum-ahead pass below
  //     simulates them all anyway).
  if (!in_string && !enum_ahead && bounds_live_here(value_start))
    for (const int32_t id : tables_->numeric_ids())
      if (get_bit(out->words, id) && !numeric_token_ok(vocab.text(id), value_start))
        clear_bit(out->words, id);
  if (enum_ahead) {
    // Simulate every admitted non-structural token (the class filter
    // already narrowed them to value starts).
    for (int w = 0; w < words; ++w) {
      uint32_t bits = out->words[static_cast<size_t>(w)];
      while (bits != 0u) {
        const int bit = __builtin_ctz(bits);
        bits &= bits - 1;
        const int id = w * 32 + bit;
        if (get_bit(tables_->whitespace_mask(), id)) continue;
        if (!simulate(vocab.text(id))) clear_bit(out->words, id);
      }
    }
  } else {
    std::vector<int8_t> prefix_ok(tables_->prefixes().size(), -1);  // -1 unknown, 0 no, 1 yes, 2 members
    std::vector<int8_t> tail_ok(tables_->tails().size(), -1);
    const std::vector<int32_t>& singles = tables_->single_quote_ids();
    std::vector<uint8_t> single_answers(singles.size(), 0);
    for (size_t i = 0; i < singles.size(); ++i) {
      const int32_t id = singles[i];
      if (!get_bit(out->words, id)) continue;
      const JsonTables::Shape& sh = tables_->shape(id);
      bool ok = false;
      if (in_string) {
        int8_t& t = tail_ok[static_cast<size_t>(sh.tail_group)];
        if (t < 0) t = simulate("\"" + tables_->tails()[static_cast<size_t>(sh.tail_group)]) ? 1 : 0;
        ok = t == 1;
      } else {
        int8_t& p = prefix_ok[static_cast<size_t>(sh.prefix_group)];
        if (p < 0) {
          const std::string& prefix = tables_->prefixes()[static_cast<size_t>(sh.prefix_group)];
          JsonMachine after = *this;
          after.cache_ = Cache{};
          bool alive = true;
          for (const char ch : prefix)
            if (!after.feed(static_cast<uint8_t>(ch))) {
              alive = false;
              break;
            }
          p = !alive ? 0 : after.content_constrained() ? 2 : 1;
        }
        ok = p == 1 || (p == 2 && simulate(vocab.text(id)));
      }
      if (!ok) clear_bit(out->words, id);
      single_answers[i] = ok ? 1 : 0;
    }
    std::vector<uint8_t> multi_answers(tables_->multi_quote_ids().size(), 0);
    for (size_t i = 0; i < tables_->multi_quote_ids().size(); ++i) {
      const int32_t id = tables_->multi_quote_ids()[i];
      if (!get_bit(out->words, id)) continue;
      const bool ok = simulate(vocab.text(id));
      if (!ok) clear_bit(out->words, id);
      multi_answers[i] = ok ? 1 : 0;
    }
    if (in_string) {
      cache_.epoch = epoch_;
      cache_.state = st;
      cache_.top = lexer_.top();
      cache_.tail_ok = std::move(single_answers);
      cache_.multi_ok = std::move(multi_answers);
    }
  }
  for (const int32_t id : e.dynamic)
    if (simulate(vocab.text(id))) set_bit(out->words, id);
  finish();
}

}  // namespace dgpp::text
