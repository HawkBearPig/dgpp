#pragma once
// Jinja chat-template interpreter for the supported model families.
// Parse chat_template.jinja once at model load and render its syntax
// tree per request. The rendering semantics follow the
// transformers chat-template environment's (trim_blocks, lstrip_blocks,
// jinja2.ext.loopcontrols, and the tojson override — plain
// json.dumps(ensure_ascii=False); see tools/gen_chat_template_goldens.py,
// which also generates the differential goldens that pin this code).
//
// Supported statement forms: text and {{ output }} (with the full
// whitespace-trim marker set), set (name or namespace-attribute target),
// if/elif/else, for (single or (k, v) tuple targets, with loop meta and
// {% break %}), and macro definitions. Macros resolve at RENDER time
// (tool_to_json lives inside {% if tools %} — a falsy `tools` leaves it
// undefined, exactly like Jinja), close over the root frame only, and
// return their rendered body as a string.
//
// Supported expressions: string/int/true/false/none literals, names,
// attribute access (digit names index lists, like Jinja's foo.0),
// subscripts (negative indexes included), method calls (items/split/
// strip), filters (capitalize/tojson/replace/length), the is-tests
// (defined/none/string/mapping/iterable, with `is not`), in/not in,
// ==/!=/</<=/>/>=, and/or/not, + and ~ concatenation, unary minus on
// int literals, and a if b else c. The globals namespace() and range(a,b)
// are the only functions.
//
// Unsupported constructs are rejected during parsing with the construct
// named in the error. Rendering errors (undefined names, bad call targets,
// non-serializable tojson input) throw std::runtime_error with the
// template line number.
//
// Validation uses differential golden tests
// (tests/data/glm_chat_template_goldens.jsonl) — byte-exact renders AND
// the glm_tokenizer ids of those renders — keyed by this file's
// FNV-1a-64 hash (header "template_hash"); a gate run against a
// different chat_template.jinja refuses rather than compares.
//
// Structural guards: compile/load allow at most 1 MiB source, 512 tokens per
// expression and 64 recursive parser entries. Each render allows 256 active
// eval/exec/macro entries and 10 million steps (expression evaluations, executed
// statements, macro calls and loop iterations). Macro buffers and block captures
// share the 64 MiB cumulative emission limit; each output buffer and newly built
// concatenation/replacement string is limited to 16 MiB. range() allows 262144
// elements. Identity replacement returns an existing value without this check.
// Exhaustion throws std::runtime_error. These are selective guards, not a bound
// on all work, collections, allocations, helper recursion, RSS or elapsed time.
// Input construction, implicit Value copies, tojson/equality traversal and
// shared-object cycles are outside this policy.
//
// Threading: compile/load once, then render() is const and thread-safe
// (all renderer state is per-call; namespace objects never escape).
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp::text {

struct MacroDef;  // internal; defined in chat_template.cpp

// The value model for chat-template data and evaluation. Undefined is a
// distinct kind from Null because `x is defined` (and the template's
// `not eid` guards) must tell them apart. Namespaces carry shared
// mutable storage: {% set ns.attr = v %} mutates through every alias —
// the template's only cross-scope state carrier.
class Value {
 public:
  enum class Kind {
    Undefined, Null, Bool, Int, Double, String, List, Map, Namespace, Macro, Method
  };
  using Members = std::vector<std::pair<std::string, Value>>;

  Value() = default;

  static Value undefined() { return Value(); }
  static Value null_value() { Value v; v.kind_ = Kind::Null; return v; }
  static Value boolean(bool b) { Value v; v.kind_ = Kind::Bool; v.b_ = b; return v; }
  static Value integer(int64_t i) { Value v; v.kind_ = Kind::Int; v.i_ = i; return v; }
  static Value number(double d) { Value v; v.kind_ = Kind::Double; v.d_ = d; return v; }
  static Value string_value(std::string s) {
    Value v; v.kind_ = Kind::String; v.s_ = std::move(s); return v;
  }
  static Value list_value(std::vector<Value> l) {
    Value v; v.kind_ = Kind::List; v.list_ = std::move(l); return v;
  }
  static Value map_value(Members m) {
    Value v; v.kind_ = Kind::Map; v.object_ = std::make_shared<Members>(std::move(m)); return v;
  }
  static Value namespace_value(Members attrs) {
    Value v; v.kind_ = Kind::Namespace;
    v.object_ = std::make_shared<Members>(std::move(attrs));
    return v;
  }

  Kind kind() const { return kind_; }
  bool is_defined() const { return kind_ != Kind::Undefined; }

  // Python truthiness: Undefined/Null/0/empty/'' are false.
  bool truthy() const;

  // Python ==/!=: numeric across Int/Double; kind mismatch is false
  // (never an exception, like Jinja).
  bool equals(const Value& other) const;

  // The {{ }} form: Python str() semantics ('True'/'False'/'None',
  // '' for Undefined). Throws for non-scalars — this template never
  // prints them, and a silent repr would hide a data bug.
  std::string to_output_string() const;

  // json.dumps(ensure_ascii=...) semantics — the transformers tojson
  // override. Throws for Namespace/Macro (Python raises on objects).
  std::string to_json(bool ensure_ascii) const;

  // Attribute read. Digit names are list/string indexes (Jinja's foo.0).
  // 'items' on a Map and 'split'/'strip' on a String produce Method
  // values (dict methods live on the object, not in its keys). A miss
  // yields Undefined so `x.y is defined` works; never throws.
  Value get_attr(std::string_view name) const;

  // {% set target.attr = value %}: Namespace (and Map) only; other
  // kinds refuse by name. Const because it mutates through the shared
  // member storage, not this value.
  void set_attr(std::string_view name, Value v) const;

  // Subscript: List/Map (string key)/String (int index); negatives
  // index from the end; a miss is Undefined.
  Value get_item(const Value& index) const;

  const Members* as_members() const {
    return (kind_ == Kind::Map || kind_ == Kind::Namespace) ? object_.get()
                                                           : nullptr;
  }
  const std::vector<Value>& as_list() const { return list_; }
  const std::string& as_string() const { return s_; }
  int as_method() const { return method_; }
  const std::shared_ptr<MacroDef>& as_macro() const { return macro_; }
  // Int/Double accessors (callers type-check kind() first; the default
  // is for diagnostics only).
  int64_t as_int(int64_t dflt = 0) const {
    return kind_ == Kind::Int ? i_ : dflt;
  }
  double as_double(double dflt = 0.0) const {
    if (kind_ == Kind::Double) return d_;
    return kind_ == Kind::Int ? static_cast<double>(i_) : dflt;
  }
  // The evaluator constructs macro-valued values when executing
  // {% macro %} definitions.
  static Value macro_value(std::shared_ptr<MacroDef> m) {
    Value v;
    v.kind_ = Kind::Macro;
    v.macro_ = std::move(m);
    return v;
  }

  // Converts a minijson DOM (the golden corpus's case inputs, the
  // service's request payloads) into render values. Object member
  // order is preserved (tojson emits it verbatim).
  static Value from_minijson(const minijson::Value& v);

 private:
  Kind kind_ = Kind::Undefined;
  bool b_ = false;
  int64_t i_ = 0;
  double d_ = 0.0;
  std::string s_;
  std::vector<Value> list_;
  std::shared_ptr<Members> object_;
  std::shared_ptr<MacroDef> macro_;
  int method_ = 0;  // 1 = items, 2 = split, 3 = strip
};

class ChatTemplate {
 public:
  // Compiles the template source. Throws std::runtime_error naming any
  // construct outside the supported subset (with its line number).
  static ChatTemplate compile(std::string source);

  // Loads <dir>/chat_template.jinja and compiles it.
  static ChatTemplate load(const std::string& chat_template_path);

  ChatTemplate() = default;
  ~ChatTemplate();
  ChatTemplate(ChatTemplate&&) noexcept;
  ChatTemplate& operator=(ChatTemplate&&) noexcept;

  // Renders. `globals` is a Map: "messages" (required, List of Maps) and
  // optional add_generation_prompt/tools/reasoning_effort/clear_thinking.
  // Unknown global keys are simply never read. Throws std::runtime_error
  // with a line number on evaluation errors (undefined names etc.).
  std::string render(const Value& globals) const;

  // FNV-1a-64 over the raw source bytes — the golden corpus's key.
  uint64_t source_hash() const;
  // Whether the compiled program can read the global `name` (the service's
  // knob gate: a template that never reads enable_thinking must not accept
  // it as if it did). Resolved from the syntax tree at compile, so a
  // comment, a printed word, an attribute (`m.preserve_thinking`) or a name
  // a {% set %} shadows does not count as a knob of the template.
  bool reads(std::string_view name) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  explicit ChatTemplate(std::unique_ptr<Impl> impl);
};

}  // namespace dgpp::text
