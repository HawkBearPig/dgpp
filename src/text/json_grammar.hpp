#pragma once
// JSON-constrained output (M6 6h, DESIGN §10): `response_format`
// json_object and json_schema as a grammar over TOKEN TEXTS, delivered
// through the same per-position mask the tool-call grammar uses
// (glm_tool_grammar.hpp). Three layers:
//
//   JsonLexer   — a byte-level automaton for JSON (RFC 8259): the
//                 structural states, strings with their escapes, the number
//                 grammar, the literals, and a container stack. Every byte
//                 either advances it or rejects; a token is a byte sequence.
//   JsonSchema  — OpenAI's structured-output subset compiled into nodes:
//                 type (and type lists), object properties / required /
//                 additionalProperties, array items / minItems / maxItems,
//                 enum and const, anyOf, and numeric minimum / maximum
//                 (exclusive forms included). Anything else refuses at
//                 compile time NAMING THE KEYWORD (the loud-refusal
//                 discipline).
//   JsonMachine — the lexer plus schema cursors: at a value position the
//                 expected node filters the value class (an anyOf splits the
//                 cursor per alternative, the frontier shrinks as bytes
//                 disambiguate), object keys match the declared properties
//                 byte by byte (closed objects) or run free (open ones),
//                 enum/const values match their JSON texts, commas and
//                 closers obey required keys and item bounds, a bounded
//                 number's digits are admitted only while some completion
//                 can still land inside the range (IntegerPrefix or
//                 DecimalPrefix). Free JSON mode is the
//                 schema {root: object, anything inside}.
//
// THE MASK. Per position the machine yields the set of token ids that may
// come next: a token is allowed iff feeding its bytes never rejects. Doing
// that for 155k tokens per step would cost milliseconds, so JsonTables
// precomputes, once per vocabulary, the static (lexical) answer for every
// tabled state and container context — the answer for a token that pops
// its frame and goes on depends on the parent and is re-simulated per
// position — and classifies every token by where free string content can
// begin or end inside it (its structural prefix before its one unescaped
// quote, its tail after it, its tail after a scalar). At a position the
// schema is applied by simulating REPRESENTATIVES: one per distinct
// prefix or tail, the pure-structure tokens and the few multi-quote
// tokens individually, group members only where the content itself is
// constrained (a closed object's key, an enum). Inside a string the answer
// is cached until the next structural event. Typical cost per position:
// microseconds; a structural position, some hundreds of simulations.
//
// EXACTNESS. The mask is a pure function of the schema, the bytes
// committed so far and the vocabulary's token texts — identical on every
// rank from the same journal record — and the unit gate proves it equal
// to the brute-force answer (every token simulated) over random prefixes.
// What a masked id means to the sampler is glm_sampler.hpp's rule (an
// absent candidate), unchanged.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "text/string_constraint.hpp"

namespace dgpp::text {

class GrammarVocab;
struct TokenMask;

// ---------------------------------------------------------------------------
// The schema
// ---------------------------------------------------------------------------

// An integer's range, inclusive after normalization (a fractional or
// exclusive bound is rounded inward at compile time). Applied only where the
// node's numeric type is integer alone. General numbers use DecimalBounds.
struct IntegerBounds {
  bool has_min = false;
  bool has_max = false;
  int64_t min = 0;
  int64_t max = 0;
  bool active() const { return has_min || has_max; }
};

// The digits of an integer being spelled — a JSON integer is '-'? then '0'
// or a run of digits without a leading zero — and the two questions a
// bound asks of them: can some further digits (possibly none) reach the
// range, and does the value as spelled lie in it. The magnitude saturates
// at twenty digits: past 10^19 the value is beyond any 64-bit bound in
// its direction, which is all the questions need. Pure arithmetic — the
// machine asks per byte, the mask per numeric token — so the two agree by
// construction.
struct IntegerPrefix {
  bool negative = false;
  bool huge = false;      // twenty or more digits
  uint64_t magnitude = 0; // the digits so far (valid while !huge)
  int digits = 0;
  void push(uint8_t b);   // '-' or a digit; anything else is ignored
  bool can_reach(const IntegerBounds& b) const;
  bool within(const IntegerBounds& b) const;
};

// Decimal comparisons keep all generated digits; converting a candidate to
// double would round an out-of-range value onto an inclusive boundary.
struct DecimalNumber {
  bool negative = false;
  std::string digits;  // significant digits, empty for zero
  int64_t exponent = 0;  // power of ten of the first significant digit
  static DecimalNumber parse(std::string_view text);
  int compare(const DecimalNumber& other) const;
};

struct DecimalBounds {
  bool has_min = false, has_max = false;
  bool exclusive_min = false, exclusive_max = false;
  DecimalNumber min, max;
  bool active() const { return has_min || has_max; }
  bool contains(const DecimalNumber& value) const;
};

struct DecimalPrefix {
  std::string text;
  void push(uint8_t b) { text.push_back(static_cast<char>(b)); }
  bool can_reach(const DecimalBounds& bounds) const;
  bool within(const DecimalBounds& bounds) const;
};

struct JsonSchemaNode {
  // Value classes, as the lexer sees them at a value's first byte.
  enum Class : uint32_t {
    kObject = 1u << 0,
    kArray = 1u << 1,
    kString = 1u << 2,
    kNumber = 1u << 3,   // a number with a fraction or exponent
    kInteger = 1u << 4,  // digits only
    kBoolean = 1u << 5,
    kNull = 1u << 6,
    kAnyClass = (1u << 7) - 1,
  };
  uint32_t types = kAnyClass;
  // object
  std::vector<std::string> property_names;
  std::vector<int> property_nodes;
  std::vector<bool> property_required;
  static constexpr int kClosed = -1;  // additionalProperties: false
  static constexpr int kAny = -2;     // any value (no schema)
  int additional = kAny;              // kClosed, kAny, or a node index
  // array
  int items = kAny;                   // node index or kAny
  int min_items = 0;
  int max_items = -1;                 // -1: unbounded
  // enum / const: the allowed values' JSON texts (json.dumps form — a
  // string target is quoted and escaped, a number is its shortest form).
  bool has_enum = false;
  std::vector<std::string> enum_texts;
  // integer: the range (never set beside an enum — the enum's texts are
  // filtered by the range at compile time instead).
  IntegerBounds bounds;
  DecimalBounds decimal_bounds;
  std::string multiple_of;
  std::vector<std::shared_ptr<const StringConstraint>> strings;
  // anyOf: the alternatives (a node with alternatives has no other content).
  std::vector<int> any_of;
};

struct JsonSchema {
  std::vector<JsonSchemaNode> nodes;
  int root = 0;
  bool container_enums = false;
};

// Compiles OpenAI's structured-output subset. Throws std::invalid_argument
// whose message starts with the offending keyword path (e.g.
// "schema.properties.city.pattern") followed by the reason. Numeric
// minimum / maximum / exclusiveMinimum / exclusiveMaximum compile into the
// node's bounds and are enforced. With `unenforced` given (a
// tool argument's schema, 2026-09-06), the keywords that only NARROW a
// typed value without an automaton behind them — minLength/maxLength,
// minProperties/maxProperties, uniqueItems, ...
// — compile instead of refusing: the value keeps its type, the narrowing
// is not applied, and "<keyword path>: <reason>" is appended to
// `unenforced`. Local references, pattern, format and multipleOf are enforced
// in both modes. Unsupported composition keywords (oneOf, allOf,
// patternProperties, ...) refuse either way.
JsonSchema compile_json_schema(const minijson::Value& schema,
                               std::vector<std::string>* unenforced = nullptr);
// JSON mode: a root object holding anything.
JsonSchema json_object_schema();
// The JSON text of a minijson value in json.dumps form (the enum targets).
std::string json_text_of(const minijson::Value& v);

// ---------------------------------------------------------------------------
// The lexer
// ---------------------------------------------------------------------------

class JsonLexer {
 public:
  enum class State : uint8_t {
    kValue,       // a value is expected (top level, after ':' or ',' in an array)
    kArrayOpen,   // after '[': a value or ']'
    kObjectOpen,  // after '{': a key or '}'
    kKeyStart,    // after ',' in an object: a key
    kInKey,       // inside a key string
    kKeyEnd,      // after a key: ':'
    kInString,    // inside a value string
    kNumMinus, kNumZero, kNumInt, kNumDot, kNumFrac, kNumE, kNumESign, kNumExp,
    kLiteral,     // inside true / false / null
    kValueEnd,    // a value closed inside a container: ',' or the closer
    kDone,        // the top-level value is complete: whitespace only
    kCount,
  };
  // What a byte did, for the schema layer (up to four per byte: a number's
  // end, the delimiter that ended it, the container it closed, its end).
  enum class Event : uint8_t {
    kNone,
    kValueStart,   // class in `cls`
    kValueByte,    // a byte of the current scalar (string quotes and content,
                   // number digits, literal letters)
    kValueEnd,     // the current value is complete (scalar or container)
    kKeyStart,     // the opening quote of a key
    kKeyByte,      // a byte of the key's content
    kKeyEnd,       // the key's closing quote
    kComma,
    kClose,        // '}' or ']' consumed
  };
  struct Step {
    bool ok = true;
    Event events[4] = {Event::kNone, Event::kNone, Event::kNone, Event::kNone};
    uint32_t cls = 0;  // JsonSchemaNode::Class at kValueStart
  };

  JsonLexer() = default;

  State state() const { return state_; }
  // The bytes so far are a complete JSON text: the top-level value closed,
  // or a top-level number that may still grow (a number has no closer).
  bool done() const {
    return state_ == State::kDone ||
           (stack_.empty() && (state_ == State::kNumZero || state_ == State::kNumInt ||
                               state_ == State::kNumFrac || state_ == State::kNumExp));
  }
  int depth() const { return static_cast<int>(stack_.size()); }
  char top() const { return stack_.empty() ? 0 : stack_.back(); }
  bool in_escape() const { return esc_ != 0; }
  bool in_literal() const { return state_ == State::kLiteral; }
  bool in_string() const { return state_ == State::kInKey || state_ == State::kInString; }
  bool in_number() const {
    return state_ >= State::kNumMinus && state_ <= State::kNumExp;
  }
  bool integer_only() const { return integer_only_; }
  void set_integer_only(bool v) { integer_only_ = v; }
  // The rest of the literal being spelled (kLiteral), else "".
  std::string literal_rest() const;
  // The escape sub-state: 0 none, 1 after '\', 2..5 hex digits pending.
  uint8_t escape() const { return esc_; }
  // Structural whitespace is capped at kMaxWsRun consecutive bytes (a
  // string's content resets and never counts): without the cap a model
  // whose mass sits on masked tokens can spend its whole budget on the
  // whitespace the grammar always admits — seen on the service, 2,667
  // bytes of tabs and newlines before a brace it never wrote. Sixteen bytes
  // covers any sane indentation; past it only a structural byte (or, when
  // the text is complete, the end) is allowed.
  static constexpr int kMaxWsRun = 16;
  uint8_t ws_run() const { return ws_run_; }

  Step feed(uint8_t b);

  // Whitespace per RFC 8259.
  static bool is_ws(uint8_t b) {
    return b == ' ' || b == '\n' || b == '\r' || b == '\t';
  }
  // A lexer standing in `state` with `ctx` (0, '{' or '[') as its one open
  // container — the tables' base; false when the pair does not exist.
  static bool make_base(State state, char ctx, bool integer_only, JsonLexer* out);

 private:
  void end_value(Step* s);
  bool string_byte(uint8_t b, Step* s, bool key);

  State state_ = State::kValue;
  uint8_t esc_ = 0;
  uint8_t ws_run_ = 0;   // consecutive structural whitespace bytes so far
  uint8_t lit_ = 0;      // 1 true, 2 false, 3 null
  uint8_t lit_pos_ = 0;
  bool integer_only_ = false;
  std::string stack_;    // '{' / '[' per open container
};

// ---------------------------------------------------------------------------
// The precomputed tables (per vocabulary)
// ---------------------------------------------------------------------------

class JsonTables {
 public:
  struct TrieNode {
    std::vector<std::pair<uint8_t, int>> children;
    std::vector<int> ids;
  };
  const std::vector<TrieNode>& token_trie() const { return trie_; }
  explicit JsonTables(const GrammarVocab& vocab);

  int vocab_size() const { return vocab_; }
  int words() const { return words_; }
  struct Entry {
    std::vector<uint32_t> words;    // the static answer
    std::vector<int32_t> dynamic;   // stack-dependent ids (re-simulated)
  };
  // The static mask of a lexical state with the given container context
  // (top: 0, '{', '[') and integer flag.
  const Entry& entry(JsonLexer::State s, char ctx, bool integer_only) const;
  // Tokens whose first non-whitespace byte starts a value of the class
  // (kInteger shares kNumber's mask).
  const std::vector<uint32_t>& class_mask(uint32_t cls) const;
  const std::vector<uint32_t>& whitespace_mask() const { return ws_; }

  // Token shape, from esc = 0: the unescaped quotes it holds.
  struct Shape {
    int32_t quotes = 0;        // unescaped quotes in the text
    int32_t prefix_group = -1; // quotes == 1: id of text[0..quote]
    int32_t tail_group = -1;   // quotes == 1: id of text[quote+1..]
    int32_t scalar_tail = -1;  // quotes == 0: id of the text from the first
                               // structural or whitespace byte after its
                               // leading whitespace (-1: none — pure content)
    bool structural_only = false;  // no quote, bytes all structure/ws
    int32_t lead_ws = 0;           // leading whitespace bytes (the run cap)
    bool numeric = false;          // numeric fragment: digits, signs, '.', 'e'/'E'
  };
  const Shape& shape(int id) const { return shapes_[static_cast<size_t>(id)]; }
  const std::vector<std::string>& prefixes() const { return prefixes_; }
  const std::vector<std::string>& tails() const { return tails_; }
  const std::vector<std::string>& scalar_tails() const { return scalar_tails_; }
  const std::vector<int32_t>& single_quote_ids() const { return single_; }
  const std::vector<int32_t>& multi_quote_ids() const { return multi_; }
  const std::vector<int32_t>& structural_ids() const { return structural_; }
  const std::vector<int32_t>& scalar_tail_ids() const { return scalar_tail_ids_; }
  // The tokens that stay inside a number: leading whitespace, an optional
  // '-', digits, nothing after. Their static answer is lexical only; a
  // bounded integer re-judges them by the digits (JsonMachine::mask).
  const std::vector<int32_t>& numeric_ids() const { return numeric_; }
  // Ids whose leading whitespace run is exactly `n` bytes (n in
  // 1..kMaxWsRun-1) or at least kMaxWsRun (n == kMaxWsRun).
  const std::vector<int32_t>& lead_ws_ids(int n) const {
    return lead_ws_ids_[static_cast<size_t>(n)];
  }

 private:
  static int index(JsonLexer::State s, char ctx, bool integer_only);
  int vocab_ = 0;
  int words_ = 0;
  std::vector<Entry> entries_;
  std::vector<uint32_t> class_[7];
  std::vector<uint32_t> ws_;
  std::vector<Shape> shapes_;
  std::vector<TrieNode> trie_;
  std::vector<std::string> prefixes_, tails_, scalar_tails_;
  std::vector<int32_t> single_, multi_, structural_, scalar_tail_ids_, numeric_;
  std::vector<std::vector<int32_t>> lead_ws_ids_;  // [kMaxWsRun + 1]
};

// ---------------------------------------------------------------------------
// The machine
// ---------------------------------------------------------------------------

class JsonMachine {
 public:
  JsonMachine() = default;
  JsonMachine(std::shared_ptr<const JsonSchema> schema, const JsonTables* tables);

  bool alive() const { return alive_; }
  // A complete, conforming text so far (a top-level number under an enum
  // counts only once a target is complete).
  bool done() const;
  // One byte of a token. false: rejected (the machine is dead afterwards).
  bool feed(uint8_t b);
  // The allowed ids for the next position: the bitmask over the vocabulary
  // and its count. Ids with empty text are never allowed; when done,
  // whitespace tokens only (EOS is the grammar layer's to add).
  void mask(const GrammarVocab& vocab, TokenMask* out) const;
  // Whether `id`'s text is allowed next (the same rule as mask()).
  bool allows(const GrammarVocab& vocab, int64_t id) const;
  // Brute force: every token simulated (the gates' oracle).
  void mask_brute_force(const GrammarVocab& vocab, TokenMask* out) const;
  const char* state_name() const;
  const JsonLexer& lexer() const { return lexer_; }

 private:
  struct Frame {
    int node = JsonSchemaNode::kAny;  // the container's node (kAny: free)
    char kind = 0;                    // '{' or '['
    std::vector<bool> used;           // object: declared properties used
    int count = 0;                    // array: items so far
    int value_node = JsonSchemaNode::kAny;  // the value being parsed
    std::string key;                  // the key so far (objects)
    // Container enums retain an exact JSON spelling across nested values.
    std::vector<int> enum_targets;
    size_t enum_pos = 0;
  };
  struct Cursor {
    std::vector<Frame> stack;
    int value_node = JsonSchemaNode::kAny;  // the top-level value's node
    // The scalar under an enum node: the targets still matching and the
    // position reached in them.
    std::vector<int> targets;
    int target_node = -1;
    size_t target_pos = 0;
    bool integer_only = false;  // the current number admits no fraction
    // The bounded number being spelled: its node and its digits so far.
    int bound_node = -1;
    IntegerPrefix number;
    DecimalPrefix decimal;
    int string_node = -1;
    std::string string_json;
    bool dead = false;
  };
  int expected_node(const Cursor& c) const;
  void leaves(int node, std::vector<int>* out) const;
  // None of the cursor's live enum targets carries a fraction or exponent.
  bool targets_integral(const Cursor& c) const;
  bool apply(const JsonLexer::Step& s, uint8_t b);
  bool on_value_start(uint32_t cls, uint8_t b);
  bool on_value_byte(uint8_t b);
  bool on_value_end();
  bool on_key_start();
  bool on_key_byte(uint8_t b);
  bool on_key_end();
  bool on_comma();
  bool on_close();
  bool sweep();  // drops dead cursors; false when none remain
  // The content of the string being read is constrained byte by byte by
  // every cursor (a closed object's key, an enum text).
  bool content_constrained() const;
  // The bytes the constrained content may continue with, and whether a
  // target is complete (the closing quote may come).
  void content_continuations(std::vector<std::string>* remainders) const;
  bool schema_constrains_here() const;
  uint32_t allowed_classes_here() const;
  bool simulate(const std::string& text) const;
  // A bounded integer is being spelled (or may start here): the numeric
  // tokens need the digit arithmetic beyond the tables' lexical answer.
  bool bounds_live_here(bool value_start) const;
  bool extended_constraints_live() const;
  bool unconstrained_value_string() const;
  // Whether a numeric token (JsonTables::numeric_ids) survives every
  // cursor's bounds — the prefix arithmetic where the cursors are plain,
  // a simulation where an enum target shares the position.
  bool numeric_token_ok(const std::string& text, bool value_start) const;

  std::shared_ptr<const JsonSchema> schema_;
  const JsonTables* tables_ = nullptr;
  JsonLexer lexer_;
  std::vector<Cursor> cursors_;
  bool alive_ = false;
  uint64_t epoch_ = 0;  // bumps on every structural event (the cache key)
  // The string-position cache: tail and multi-quote answers for this
  // (epoch, state, container).
  struct Cache {
    uint64_t epoch = ~0ull;
    JsonLexer::State state = JsonLexer::State::kCount;
    char top = 0;
    std::vector<uint8_t> tail_ok;
    std::vector<uint8_t> multi_ok;
  };
  mutable Cache cache_;
};

}  // namespace dgpp::text
