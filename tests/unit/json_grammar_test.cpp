// M6 6h: the JSON grammar over a fake vocabulary (host-only, always runs).
// Pins: the lexer on a corpus (every valid text accepted and done, every
// invalid one rejected at a byte), the schema compiler's subset and its
// refusals naming the keyword path, and — the exactness gate — the
// representative-based mask() equal to the brute-force answer (every
// token simulated) at every position of random walks driven by the mask
// itself, over the free machine and over schemas exercising every node
// kind; every finished walk parses as JSON and conforms.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include <array>
#include "loaders/minijson.hpp"
#include "text/json_grammar.hpp"
#include "text/tool_grammar.hpp"

namespace {

using dgpp::text::ChatMarker;
using dgpp::text::ChatMarkers;
using dgpp::text::GrammarVocab;
using dgpp::text::JsonLexer;
using dgpp::text::JsonMachine;
using dgpp::text::JsonSchema;
using dgpp::text::JsonTables;
using dgpp::text::TokenMask;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Ids 0..255 are single bytes; then the multi-byte tokens a BPE vocabulary
// holds around JSON: quote-bearing pieces with structural prefixes and
// tails, scalars with tails, whole literals and their pieces, whitespace
// runs, escapes, a complete object; markers and EOS ids; vocab_size pads.
const char* kWords[] = {
    "{\"", "\":", "\": \"", "\",", "\"}", ",\"", "\"]", "[\"", ": \"", "\"\n",
    "\"},", "\"}]", "true", "false", "null", "tr", "ue", "fal", "se", "nu",
    "ll", "12", "3.5", "e5", "-1", "0.", "1,", "2}", "5]", "10", "red",
    "blue", "gree", "n\"", "re", "d\"", "city", "days", "unit", "name",
    "abc", " \"", "\n  ", "\n", "  ", "},", "]}", "}}", ",\n", "\\\"",
    "\\n", "\\u00", "1f", "ab", "{\"a\":1}", "[1,2]", "{}", "[]", "\"\"",
    "\"a\"", " ", ":", "\": ", "\": [", "\": {", "\"c", "ity\"", "\"unit\":",
    "celsius", "fahrenheit", "cel", "sius\"", "\"celsius\"", "true,",
    "false}", "null]", "e", "E+", "-", "9", ".", "0", "\"\\", "x\"",
    // Numeric pieces for the bounded-integer gates: whole
    // numbers, a signed one, a whitespace-led one, digit runs.
    "100", "2000", "-5", " 12", "50", "99", "7", "000",
};
constexpr int kWordBase = 256;
constexpr int kWordCount = static_cast<int>(sizeof(kWords) / sizeof(kWords[0]));
constexpr int64_t kThinkOpen = 400, kThinkClose = 401, kToolOpen = 402,
                  kToolClose = 403, kKeyOpen = 404, kKeyClose = 405,
                  kValueOpen = 406, kValueClose = 407;
constexpr int64_t kEosText = 410, kEosUser = 411, kEosObs = 412;
constexpr int kVocab = 416;

GrammarVocab fake_vocab() {
  std::vector<std::string> texts(static_cast<size_t>(kVocab));
  for (int b = 0; b < 256; ++b)
    texts[static_cast<size_t>(b)] = std::string(1, static_cast<char>(b));
  for (int i = 0; i < kWordCount; ++i)
    texts[static_cast<size_t>(kWordBase + i)] = kWords[i];
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  m.arg_key_open = ChatMarker{kKeyOpen, "<arg_key>"};
  m.arg_key_close = ChatMarker{kKeyClose, "</arg_key>"};
  m.arg_value_open = ChatMarker{kValueOpen, "<arg_value>"};
  m.arg_value_close = ChatMarker{kValueClose, "</arg_value>"};
  return GrammarVocab(std::move(texts), m, {kEosText, kEosUser, kEosObs}, kVocab,
                      kEosObs);
}

const GrammarVocab& vocab() {
  static const GrammarVocab v = fake_vocab();
  return v;
}
const JsonTables& tables() {
  static const JsonTables t(vocab());
  return t;
}

std::shared_ptr<const JsonSchema> compile(const std::string& text) {
  const dgpp::minijson::ParseResult p = dgpp::minijson::parse(text);
  return std::make_shared<const JsonSchema>(dgpp::text::compile_json_schema(p.root));
}

bool lex_all(const std::string& text, JsonLexer* lx) {
  for (const char c : text)
    if (!lx->feed(static_cast<uint8_t>(c)).ok) return false;
  return true;
}

DGPP_TEST(json_grammar_lexerAcceptsTheCorpusAndRejectsTheRest) {
  const char* valid[] = {
      "{}", "[]", "1", "-0", "0.5", "1e5", "1E+5", "-1.25e-3", "\"\"",
      "\"a\\\"b\\\\c\\n\\u00e9\"", "true", "false", "null",
      "{\"a\":1,\"b\":[1,2,{\"c\":null}],\"d\":\"x\"}", " [ 1 , 2 ] ",
      "\n{\n  \"k\" : \"v\"\n}\n", "[[[]]]", "{\"\":{}}",
  };
  for (const char* t : valid) {
    JsonLexer lx;
    require(lex_all(t, &lx), std::string("rejected valid: ") + t);
    require(lx.done(), std::string("not done after: ") + t);
    require(lx.depth() == 0, "stack not empty");
  }
  const char* invalid[] = {
      "01", "1.", "1e", "1e+", ".5", "+1", "-", "[1,]", "{\"a\":1,}",
      "{a:1}", "{\"a\" 1}", "[1 2]", "tru", "truee", "nul", "\"\n\"",
      "\"\\x\"", "\"\\u12g4\"", "]", "}", "{]", "[}", "1 2", "{} {}",
      "\"a\" \"b\"", "{\"a\":}", "[,]", "{,}",
  };
  for (const char* t : invalid) {
    JsonLexer lx;
    const bool ok = lex_all(t, &lx) && lx.done();
    require(!ok, std::string("accepted invalid: ") + t);
  }
  // A number ends at its delimiter with the delimiter re-dispatched: the
  // events of "1}" inside an object end the value, close, end the object.
  JsonLexer lx;
  require(lex_all("{\"k\":1", &lx), "prefix");
  const JsonLexer::Step s = lx.feed('}');
  require(s.ok && s.events[0] == JsonLexer::Event::kValueEnd &&
              s.events[1] == JsonLexer::Event::kClose &&
              s.events[2] == JsonLexer::Event::kValueEnd &&
              s.events[3] == JsonLexer::Event::kNone,
          "number-end events");
  require(lx.done(), "done after 1}");
  // The bases exist where the states exist.
  JsonLexer base;
  require(JsonLexer::make_base(JsonLexer::State::kValue, 0, false, &base) &&
              base.state() == JsonLexer::State::kValue && base.top() == 0,
          "base value/0");
  require(JsonLexer::make_base(JsonLexer::State::kKeyStart, '{', false, &base) &&
              base.top() == '{',
          "base keystart/{");
  require(!JsonLexer::make_base(JsonLexer::State::kKeyStart, '[', false, &base),
          "no keystart in an array");
  require(!JsonLexer::make_base(JsonLexer::State::kValueEnd, 0, false, &base),
          "no value-end at the top");
  require(JsonLexer::make_base(JsonLexer::State::kNumInt, '[', true, &base) &&
              base.integer_only(),
          "integer base");
  require(lex_all("tr", &(lx = JsonLexer())) && lx.literal_rest() == "ue",
          "literal rest");
}

DGPP_TEST(json_grammar_schemaCompilesTheSubsetAndRefusesNamingTheKeyword) {
  // A tool argument's compile tolerates the keywords that only narrow a
  // value without an automaton behind them, naming each by path with the
  // reason; the strict compile refuses them. An integer's
  // bounds are neither: they compile and are enforced.
  {
    const dgpp::minijson::ParseResult p = dgpp::minijson::parse(
        R"({"type":"object","properties":{"n":{"type":"integer","minimum":0,"maximum":9},)"
        R"("r":{"type":"number","minimum":0},)"
        R"("s":{"type":"string","pattern":"^a","format":"date"}},"required":["n"]})");
    std::vector<std::string> unenforced;
    const dgpp::text::JsonSchema lax = dgpp::text::compile_json_schema(p.root, &unenforced);
    const dgpp::text::JsonSchemaNode& r = lax.nodes[static_cast<size_t>(lax.root)];
    require(r.property_names.size() == 3, "three properties");
    const dgpp::text::JsonSchemaNode& n = lax.nodes[static_cast<size_t>(r.property_nodes[0])];
    require(n.types == dgpp::text::JsonSchemaNode::kInteger && n.bounds.has_min &&
                n.bounds.has_max && n.bounds.min == 0 && n.bounds.max == 9,
            "an integer's bounds compile");
    require(lax.nodes[static_cast<size_t>(r.property_nodes[1])].decimal_bounds.active() &&
                lax.nodes[static_cast<size_t>(r.property_nodes[2])].types ==
                    dgpp::text::JsonSchemaNode::kString,
            "tolerated keywords leave the types");
    require(unenforced.empty(), "pattern and format are now enforced");
    const auto strict = dgpp::text::compile_json_schema(p.root);
    require(strict.nodes[strict.nodes[strict.root].property_nodes[2]].strings.size() == 2,
            "strict schemas compile both string constraints");
  }
  // The bounds' normalization: inclusive int64 after rounding inward and
  // stepping the exclusive forms; an enum is filtered instead of bounded.
  {
    const auto b = [](const std::string& text) {
      return compile(text)->nodes[0].bounds;
    };
    dgpp::text::IntegerBounds x = b(R"({"type":"integer","minimum":0.5,"exclusiveMaximum":10})");
    require(x.has_min && x.min == 1 && x.has_max && x.max == 9, "ceil(0.5) = 1, 10 exclusive = 9");
    x = b(R"({"type":"integer","exclusiveMinimum":0,"maximum":2.5})");
    require(x.has_min && x.min == 1 && x.has_max && x.max == 2, "0 exclusive = 1, floor(2.5) = 2");
    x = b(R"({"type":"integer","exclusiveMinimum":-1.5,"exclusiveMaximum":0.5})");
    require(x.min == -1 && x.max == 0, "the exclusive forms round inward: floor(-1.5)+1, ceil(0.5)-1");
    x = b(R"({"type":["integer","null"],"minimum":-5})");
    require(x.has_min && x.min == -5 && !x.has_max, "integer|null keeps the integer's bound");
    x = b(R"({"type":"integer","minimum":1,"exclusiveMinimum":3,"maximum":9,"exclusiveMaximum":9})");
    require(x.min == 4 && x.max == 8, "both forms: the tighter wins");
    x = b(R"({"type":"integer","maximum":9223372036854775807,"minimum":-9223372036854775808})");
    require(x.min == INT64_MIN && x.max == INT64_MAX, "the int64 ends are exact");
    const auto e = compile(R"({"type":"integer","enum":[1,5,50],"maximum":10})");
    require(e->nodes[0].has_enum && e->nodes[0].enum_texts == std::vector<std::string>{"1", "5"} &&
                !e->nodes[0].bounds.active(),
            "an enum is filtered by the range and carries no bound");
    const auto u = compile(R"({"enum":[1,"a",50],"maximum":10})");
    require(u->nodes[0].enum_texts == std::vector<std::string>{"1", "\"a\""} &&
                u->nodes[0].types == (dgpp::text::JsonSchemaNode::kNumber |
                                      dgpp::text::JsonSchemaNode::kInteger |
                                      dgpp::text::JsonSchemaNode::kString),
            "an untyped enum: the integral members filtered, the string kept");
    // Lax: the shapes the arithmetic does not cover are recorded, with the
    // reason, and nothing is applied.
    const auto lax_of = [](const std::string& text, std::vector<std::string>* out) {
      const dgpp::minijson::ParseResult p = dgpp::minijson::parse(text);
      return dgpp::text::compile_json_schema(p.root, out);
    };
    std::vector<std::string> notes;
    dgpp::text::JsonSchema l = lax_of(R"({"type":"integer","exclusiveMinimum":true,"minimum":0})", &notes);
    require(!l.nodes[0].bounds.active() && notes.size() == 2 &&
                notes[0].rfind("schema.minimum: ", 0) == 0 &&
                notes[1].rfind("schema.exclusiveMinimum: ", 0) == 0 &&
                notes[1].find("draft-4") != std::string::npos,
            "the draft-4 boolean form: both bounds recorded, none applied");
    notes.clear();
    l = lax_of(R"({"type":"integer","minimum":5,"maximum":3})", &notes);
    require(!l.nodes[0].bounds.active() && notes.size() == 2 &&
                notes[1].find("below the minimum") != std::string::npos,
            "an empty range: recorded, none applied");
    notes.clear();
    l = lax_of(R"({"anyOf":[{"type":"integer"}],"minimum":0})", &notes);
    require(l.nodes[0].any_of.size() == 1 && notes.size() == 1 &&
                notes[0].find("beside anyOf") != std::string::npos,
            "a bound beside anyOf: recorded");
    notes.clear();
    l = lax_of(R"({"type":"integer","enum":[1,2],"minimum":3})", &notes);
    require(l.nodes[0].enum_texts.size() == 2 && notes.size() == 1 &&
                notes[0].find("no enum member") != std::string::npos,
            "an enum no member of which fits: kept whole, recorded");
    notes.clear();
    l = lax_of(R"({"type":"integer","minimum":1e30})", &notes);
    require(!l.nodes[0].bounds.active() && notes.size() == 1 &&
                notes[0].find("64-bit") != std::string::npos,
            "a bound beyond int64: recorded");
  }
  const auto s = compile(R"({
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "x"},
      "days": {"type": "integer"},
      "unit": {"enum": ["celsius", "fahrenheit"]},
      "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
      "score": {"type": ["number", "null"]},
      "mode": {"const": "fast"},
      "nested": {"anyOf": [{"type": "string"}, {"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["a"], "additionalProperties": false}]}
    },
    "required": ["city", "unit"],
    "additionalProperties": false
  })");
  const dgpp::text::JsonSchemaNode& root = s->nodes[static_cast<size_t>(s->root)];
  require(root.types == dgpp::text::JsonSchemaNode::kObject, "root object");
  require(root.property_names.size() == 7 && root.additional == dgpp::text::JsonSchemaNode::kClosed,
          "root properties closed");
  require(root.property_required[0] && !root.property_required[1] && root.property_required[2],
          "required flags");
  const auto& days = s->nodes[static_cast<size_t>(root.property_nodes[1])];
  require(days.types == dgpp::text::JsonSchemaNode::kInteger, "integer type");
  const auto& unit = s->nodes[static_cast<size_t>(root.property_nodes[2])];
  require(unit.has_enum && unit.enum_texts.size() == 2 && unit.enum_texts[0] == "\"celsius\"" &&
              unit.types == dgpp::text::JsonSchemaNode::kString,
          "enum texts and inferred class");
  const auto& tags = s->nodes[static_cast<size_t>(root.property_nodes[3])];
  require(tags.min_items == 1 && tags.max_items == 3 && tags.items >= 0, "array bounds");
  const auto& score = s->nodes[static_cast<size_t>(root.property_nodes[4])];
  require(score.types == (dgpp::text::JsonSchemaNode::kNumber | dgpp::text::JsonSchemaNode::kInteger |
                          dgpp::text::JsonSchemaNode::kNull),
          "type list");
  const auto& mode = s->nodes[static_cast<size_t>(root.property_nodes[5])];
  require(mode.has_enum && mode.enum_texts == std::vector<std::string>{"\"fast\""}, "const");
  const auto& nested = s->nodes[static_cast<size_t>(root.property_nodes[6])];
  require(nested.any_of.size() == 2, "anyOf");

  struct Refusal {
    const char* schema;
    const char* path;
  };
  const Refusal refusals[] = {
      {R"({"type":"object","properties":{"city":{"type":"string","pattern":"["}}})",
       "schema.properties.city.pattern"},
      {R"({"type":"string","minLength":1})", "schema.minLength"},
      {R"({"type":"object","required":["zzz"]})", "schema.required"},
      {R"({"type":"whatever"})", "schema.type"},
      {R"({"anyOf":[{"type":"string"}],"type":"string"})", "schema.type"},
      {R"({"type":"array","items":{"type":"string","format":"unknown-format"}})",
       "schema.items.format"},
      {R"({"type":"string","enum":[1]})", "schema.enum"},
      {R"({"const":1,"enum":[1]})", "schema.const"},
      {R"({"type":"array","minItems":3,"maxItems":2})", "schema.maxItems"},
      // The bounds the arithmetic does not cover refuse under strict.
      {R"({"type":"number","minimum":2,"maximum":1})", "schema.maximum"},
      {R"({"type":"number","minimum":2,"exclusiveMaximum":2})", "schema.exclusiveMaximum"},
      {R"({"type":"integer","minimum":0,"exclusiveMinimum":true})", "schema.exclusiveMinimum"},
      {R"({"type":"integer","minimum":5,"maximum":3})", "schema.maximum"},
      {R"({"type":"integer","minimum":5,"exclusiveMaximum":5})", "schema.exclusiveMaximum"},
      {R"({"type":"integer","enum":[1,2],"minimum":3})", "schema.minimum"},
      {R"({"type":"integer","minimum":1e30})", "schema.minimum"},
      {R"({"type":"integer","maximum":"9"})", "schema.maximum"},
      {R"({"anyOf":[{"type":"integer"}],"minimum":0})", "schema.minimum"},
      {R"({"enum":[1.5, 2],"minimum":3})", "schema.minimum"},
  };
  for (const Refusal& r : refusals) {
    bool threw = false;
    std::string msg;
    try {
      compile(r.schema);
    } catch (const std::invalid_argument& e) {
      threw = true;
      msg = e.what();
    }
    require(threw, std::string("compiled: ") + r.schema);
    require(msg.rfind(r.path, 0) == 0,
            std::string("refusal '") + msg + "' does not start with " + r.path);
  }
  // Free JSON mode: a root object holding anything.
  const JsonSchema free = dgpp::text::json_object_schema();
  require(free.nodes.size() == 1 && free.nodes[0].types == dgpp::text::JsonSchemaNode::kObject &&
              free.nodes[0].additional == dgpp::text::JsonSchemaNode::kAny,
          "json_object schema");
}

// A light conformance check of a finished document against the schemas
// below (each walk's text must parse and satisfy the pinned facts).
void check_conforms(const std::string& which, const std::string& text) {
  dgpp::minijson::ParseResult p;
  try {
    p = dgpp::minijson::parse(text);
  } catch (const std::exception& e) {
    throw std::runtime_error(which + ": unparsable '" + text + "': " + e.what());
  }
  const dgpp::minijson::Value& v = p.root;
  if (which == "free" || which == "closed") require(v.is_object(), which + ": root not an object: " + text);
  if (which == "closed") {
    require(v.find("city") != nullptr && v.find("city")->is_string(), "closed: city missing: " + text);
    require(v.find("unit") != nullptr, "closed: unit missing: " + text);
    const std::string_view u = v.find("unit")->as_string();
    require(u == "celsius" || u == "fahrenheit", "closed: unit value: " + text);
    for (const dgpp::minijson::Member& m : v.members())
      require(m.key == "city" || m.key == "unit" || m.key == "days" || m.key == "tags" ||
                  m.key == "score" || m.key == "mode" || m.key == "nested",
              "closed: foreign key '" + m.key + "' in " + text);
    if (const dgpp::minijson::Value* d = v.find("days"))
      require(d->is_number() && d->as_double() == d->as_int(), "closed: days not integral: " + text);
    if (const dgpp::minijson::Value* t = v.find("tags")) {
      require(t->is_array() && !t->items().empty() && t->items().size() <= 3, "closed: tags bounds: " + text);
      for (const dgpp::minijson::Value& e : t->items()) require(e.is_string(), "closed: tag type: " + text);
    }
    if (const dgpp::minijson::Value* m = v.find("mode"))
      require(m->is_string() && m->as_string() == "fast", "closed: mode const: " + text);
  }
  if (which == "list") {
    require(v.is_array() && v.items().size() >= 2 && v.items().size() <= 4, "list: bounds: " + text);
    for (const dgpp::minijson::Value& e : v.items()) {
      require(e.is_object(), "list: item not an object: " + text);
      require(e.find("id") != nullptr && e.find("id")->is_number(), "list: id: " + text);
      for (const dgpp::minijson::Member& m : e.members())
        require(m.key == "id" || m.key == "kind", "list: foreign key '" + m.key + "' in " + text);
      if (const dgpp::minijson::Value* k = e.find("kind")) {
        require(k->is_string() || k->is_null(), "list: kind: " + text);
        if (k->is_string()) require(k->as_string() == "a" || k->as_string() == "b", "list: kind enum: " + text);
      }
    }
  }
  if (which == "scalar") {
    if (v.is_number()) require(v.as_double() == 3 || v.as_double() == 10, "scalar: number enum: " + text);
    else require(v.is_bool() && v.as_bool(), "scalar: not 3/10/true: " + text);
  }
  const auto integral = [&](const dgpp::minijson::Value& x, const char* key) {
    require(x.is_number() && x.as_double() == x.as_int(), std::string(which) + ": " + key + " not integral: " + text);
    return x.as_int();
  };
  if (which == "bounded") {
    require(v.is_object() && v.find("n") != nullptr, "bounded: root/n: " + text);
    for (const dgpp::minijson::Member& m : v.members()) {
      const std::string& k = m.key;
      const dgpp::minijson::Value& x = m.value;
      if (k == "n") { const int64_t i = integral(x, "n"); require(i >= 1 && i <= 2000, "bounded: n: " + text); }
      else if (k == "neg") { const int64_t i = integral(x, "neg"); require(i >= -12 && i <= -3, "bounded: neg: " + text); }
      else if (k == "lo") { require(integral(x, "lo") >= 100, "bounded: lo: " + text); }
      else if (k == "hi") { require(integral(x, "hi") <= 5, "bounded: hi: " + text); }
      else if (k == "one") { require(integral(x, "one") == 7, "bounded: one: " + text); }
      else if (k == "opt") { if (!x.is_null()) { const int64_t i = integral(x, "opt"); require(i >= 0 && i <= 10, "bounded: opt: " + text); } }
      else if (k == "either") { if (!x.is_string()) require(integral(x, "either") >= 50, "bounded: either: " + text); }
      else if (k == "mixed") { const int64_t i = integral(x, "mixed"); require(i >= 100 || i <= 10, "bounded: mixed: " + text); }
      else if (k == "pick") { const int64_t i = integral(x, "pick"); require(i == 5 || i == 500 || i <= 20, "bounded: pick: " + text); }
      else if (k == "list") {
        require(x.is_array() && x.items().size() <= 3, "bounded: list size: " + text);
        for (const dgpp::minijson::Value& e : x.items()) { const int64_t i = integral(e, "list"); require(i >= 0 && i <= 99, "bounded: list item: " + text); }
      } else {
        throw std::runtime_error("bounded: foreign key '" + k + "' in " + text);
      }
    }
  }
  if (which == "range") {
    const int64_t i = integral(v, "range");
    require(i >= -50 && i <= 150, "range: " + text);
  }
  if (which == "decimal") {
    require(v.is_number() && v.as_double() >= 1.25 && v.as_double() < 2.5,
            "decimal outside [1.25, 2.5): " + text);
  }
}

// Random walks driven by mask(): at every position the representative mask
// equals brute force; the walk feeds a random allowed token.
void oracle_walks(const std::string& which, std::shared_ptr<const JsonSchema> schema,
                  int walks, uint32_t seed, int* finished_out) {
  std::mt19937 rng(seed);
  int finished = 0;
  long positions = 0;
  std::string last_text, last_state;
  for (int w = 0; w < walks; ++w) {
    JsonMachine m(schema, &tables());
    std::string text;
    for (int step = 0; step < 120 && m.alive(); ++step) {
      TokenMask fast, slow;
      m.mask(vocab(), &fast);
      m.mask_brute_force(vocab(), &slow);
      ++positions;
      if (fast.words != slow.words || fast.allowed != slow.allowed) {
        std::string diff;
        for (int id = 0; id < kVocab; ++id)
          if (fast.allows(id) != slow.allows(id))
            diff += " " + std::to_string(id) + "('" + vocab().text(id) + "')" +
                    (fast.allows(id) ? "+" : "-");
        throw std::runtime_error(which + ": mask differs from brute force after '" + text +
                                 "' (" + m.state_name() + "):" + diff);
      }
      // allows() agrees with the mask on a few ids.
      for (int k = 0; k < 4; ++k) {
        const int id = static_cast<int>(rng() % kVocab);
        require(m.allows(vocab(), id) == fast.allows(id),
                which + ": allows() disagrees at id " + std::to_string(id) + " after '" + text + "'");
      }
      if (m.done()) {
        ++finished;
        check_conforms(which, text);
        break;
      }
      require(fast.allowed > 0, which + ": no token allowed after '" + text + "'");
      // Pick a random allowed id; most of the time steer toward tokens
      // carrying structure (a quote, a closer, a delimiter) so walks
      // finish. Empty-text ids are never allowed.
      std::vector<int> allowed, steering;
      for (int id = 0; id < kVocab; ++id)
        if (fast.allows(id)) {
          allowed.push_back(id);
          if (vocab().text(id).find_first_of("\"}],:") != std::string::npos)
            steering.push_back(id);
        }
      const bool steer = step > 3 && !steering.empty() && rng() % 100 < 70;
      const std::vector<int>& pool = steer ? steering : allowed;
      const int pick = pool[rng() % pool.size()];
      const std::string& t = vocab().text(pick);
      require(!t.empty(), which + ": an empty-text id allowed");
      for (const char c : t)
        require(m.feed(static_cast<uint8_t>(c)),
                which + ": an allowed token rejected: '" + t + "' after '" + text + "'");
      text += t;
      last_text = text;
      last_state = m.state_name();
    }
  }
  std::printf("      %s: %d walks, %ld positions, %d finished\n", which.c_str(), walks,
              positions, finished);
  if (std::getenv("DGPP_JSON_WALK_DEBUG") != nullptr)
    std::printf("      last walk: '%s' (%s)\n", last_text.c_str(), last_state.c_str());
  *finished_out = finished;
}

DGPP_TEST(json_grammar_maskEqualsBruteForceOnRandomWalks) {
  int finished = 0;
  oracle_walks("free", std::make_shared<const JsonSchema>(dgpp::text::json_object_schema()),
               120, 20260904u, &finished);
  require(finished > 20, "free walks rarely finish");
  const auto closed = compile(R"({
    "type": "object",
    "properties": {
      "city": {"type": "string"},
      "days": {"type": "integer"},
      "unit": {"enum": ["celsius", "fahrenheit"]},
      "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
      "score": {"type": ["number", "null"]},
      "mode": {"const": "fast"},
      "nested": {"anyOf": [{"type": "string"}, {"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["a"], "additionalProperties": false}]}
    },
    "required": ["city", "unit"],
    "additionalProperties": false
  })");
  oracle_walks("closed", closed, 150, 7u, &finished);
  require(finished > 20, "closed walks rarely finish");
  const auto list = compile(R"({
    "type": "array", "minItems": 2, "maxItems": 4,
    "items": {"type": "object", "properties": {"id": {"type": "number"}, "kind": {"enum": ["a", "b", null]}},
              "required": ["id"], "additionalProperties": false}
  })");
  oracle_walks("list", list, 120, 99u, &finished);
  require(finished > 10, "list walks rarely finish");
  const auto scalar = compile(R"({"anyOf": [{"enum": [3, 10]}, {"const": true}]})");
  oracle_walks("scalar", scalar, 40, 3u, &finished);
  require(finished > 20, "scalar walks rarely finish");
  // Bounded integers in every position the arithmetic reaches:
  // two-sided, negative, one-sided each way, a single value, beside null,
  // as an anyOf alternative beside a string, two bounded alternatives, an
  // enum beside a bound, and array items.
  const auto bounded = compile(R"({
    "type": "object",
    "properties": {
      "n": {"type": "integer", "minimum": 1, "maximum": 2000},
      "neg": {"type": "integer", "minimum": -12, "maximum": -3},
      "lo": {"type": "integer", "minimum": 100},
      "hi": {"type": "integer", "maximum": 5},
      "one": {"type": "integer", "minimum": 7, "maximum": 7},
      "opt": {"type": ["integer", "null"], "minimum": 0, "maximum": 10},
      "either": {"anyOf": [{"type": "integer", "minimum": 50}, {"type": "string"}]},
      "mixed": {"anyOf": [{"type": "integer", "minimum": 100}, {"type": "integer", "maximum": 10}]},
      "pick": {"anyOf": [{"enum": [5, 500]}, {"type": "integer", "maximum": 20}]},
      "list": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 99}, "maxItems": 3}
    },
    "required": ["n"],
    "additionalProperties": false
  })");
  oracle_walks("bounded", bounded, 200, 20260907u, &finished);
  require(finished > 40, "bounded walks rarely finish");
  const auto range = compile(R"({"type": "integer", "minimum": -50, "maximum": 150})");
  oracle_walks("range", range, 60, 11u, &finished);
  require(finished > 30, "range walks rarely finish");
}

DGPP_TEST(json_grammar_integerBoundsAreEnforcedDigitByDigit) {
  // The arithmetic alone, exhaustively over small ranges: can_reach(p)
  // holds iff some integer in the range has p as its JSON prefix, and
  // within(p) iff p is such an integer itself.
  {
    std::vector<dgpp::text::IntegerBounds> ranges;
    for (int lo = -12; lo <= 12; ++lo)
      for (int hi = lo; hi <= 12; ++hi) {
        dgpp::text::IntegerBounds b;
        b.has_min = b.has_max = true;
        b.min = lo;
        b.max = hi;
        ranges.push_back(b);
      }
    for (int one = -12; one <= 12; ++one) {
      dgpp::text::IntegerBounds b;
      b.has_min = true;
      b.min = one;
      ranges.push_back(b);
      b = dgpp::text::IntegerBounds{};
      b.has_max = true;
      b.max = one;
      ranges.push_back(b);
    }
    long checked = 0;
    for (const dgpp::text::IntegerBounds& b : ranges) {
      // Every JSON integer text within three digits, and every prefix of it.
      for (int v = -999; v <= 999; ++v) {
        const std::string text = std::to_string(v);
        for (size_t len = 1; len <= text.size(); ++len) {
          const std::string p = text.substr(0, len);
          dgpp::text::IntegerPrefix ip;
          for (const char ch : p) ip.push(static_cast<uint8_t>(ch));
          bool reach = false, exact = false;
          for (int w = -999; w <= 999; ++w) {
            const bool in = (!b.has_min || w >= b.min) && (!b.has_max || w <= b.max);
            if (!in) continue;
            const std::string wt = std::to_string(w);
            if (wt == p || (p == "-" && w <= 0)) exact = exact || wt == p;
            if (wt.compare(0, p.size(), p) == 0 || (p == "-" && w <= 0)) reach = true;
          }
          // "-0" is the value 0 (a prefix nothing extends).
          if (p == "-0") { reach = exact = (!b.has_min || b.min <= 0) && (!b.has_max || b.max >= 0); }
          require(ip.can_reach(b) == reach,
                  "can_reach('" + p + "') under [" + (b.has_min ? std::to_string(b.min) : "-inf") +
                      ", " + (b.has_max ? std::to_string(b.max) : "+inf") + "] should be " +
                      (reach ? "true" : "false"));
          require(ip.within(b) == exact,
                  "within('" + p + "') under [" + (b.has_min ? std::to_string(b.min) : "-inf") +
                      ", " + (b.has_max ? std::to_string(b.max) : "+inf") + "] should be " +
                      (exact ? "true" : "false"));
          ++checked;
        }
      }
    }
    require(checked > 100000, "the exhaustive check ran");
    // The int64 ends, and past them.
    dgpp::text::IntegerBounds top;
    top.has_max = true;
    top.max = INT64_MAX;
    dgpp::text::IntegerPrefix ip;
    for (const char ch : std::string("922337203685477580")) ip.push(static_cast<uint8_t>(ch));
    require(ip.can_reach(top), "…580 can still reach INT64_MAX");
    dgpp::text::IntegerPrefix ip7 = ip, ip8 = ip;
    ip7.push('7');
    ip8.push('8');
    require(ip7.within(top) && ip7.can_reach(top) && !ip8.can_reach(top), "INT64_MAX in, +1 out");
    dgpp::text::IntegerPrefix twenty;
    for (int i = 0; i < 20; ++i) twenty.push(i == 0 ? '1' : '0');
    require(twenty.huge && !twenty.can_reach(top) && twenty.within(dgpp::text::IntegerBounds{}),
            "twenty digits: past every int64 maximum, inside no bound");
    dgpp::text::IntegerBounds bottom;
    bottom.has_min = true;
    bottom.min = INT64_MIN;
    dgpp::text::IntegerPrefix neg;
    for (const char ch : std::string("-9223372036854775808")) neg.push(static_cast<uint8_t>(ch));
    require(neg.within(bottom) && neg.can_reach(bottom), "INT64_MIN in");
    dgpp::text::IntegerPrefix neg9 = neg;
    neg9 = dgpp::text::IntegerPrefix{};
    for (const char ch : std::string("-9223372036854775809")) neg9.push(static_cast<uint8_t>(ch));
    require(!neg9.can_reach(bottom), "INT64_MIN - 1 out");
  }
  // The machine, digit by digit: a two-sided range inside an object.
  const auto two = compile(R"({"type":"object","properties":{"n":{"type":"integer","minimum":1,"maximum":2000}},"required":["n"],"additionalProperties":false})");
  JsonMachine m(two, &tables());
  TokenMask mask;
  for (const char ch : std::string("{\"n\":")) require(m.feed(static_cast<uint8_t>(ch)), "open n");
  m.mask(vocab(), &mask);
  require(mask.allows('1') && mask.allows('9') && !mask.allows('0') && !mask.allows('-'),
          "1..9 start a value in [1, 2000]; 0 and a sign cannot");
  require(mask.allows(kWordBase + 29) /* 10 */ && mask.allows(kWordBase + 21) /* 12 */ &&
              mask.allows(kWordBase + 84) /* 100 */ && mask.allows(kWordBase + 85) /* 2000 */ &&
              mask.allows(kWordBase + 87) /* " 12" */,
          "whole numeric tokens inside the range");
  require(!mask.allows(kWordBase + 24) /* -1 */ && !mask.allows(kWordBase + 86) /* -5 */ &&
              !mask.allows(kWordBase + 78) /* - */ && !mask.allows(kWordBase + 81) /* 0 */ &&
              !mask.allows(kWordBase + 91) /* 000 */,
          "numeric tokens outside it");
  require(m.feed('2'), "2");
  m.mask(vocab(), &mask);
  require(mask.allows('0') && mask.allows('9') && mask.allows('}') && mask.allows(kWordBase + 81) &&
              mask.allows(kWordBase + 91) /* 000 -> 2000 */ && !mask.allows(kWordBase + 84) /* 2100 */,
          "after 2: any digit, the close, 000 but not 100");
  require(m.feed('0') && m.feed('0'), "200");
  m.mask(vocab(), &mask);
  require(mask.allows('0') && !mask.allows('1') && !mask.allows('9') && mask.allows('}') &&
              !mask.allows(kWordBase + 21) /* 12 */ && !mask.allows(kWordBase + 27) /* 2} = 2002 */,
          "after 200: only 0 can follow, or the close");
  require(m.feed('0'), "2000");
  m.mask(vocab(), &mask);
  require(!mask.allows('0') && !mask.allows('9') && mask.allows('}') && mask.allows(' '),
          "at the maximum: no digit; the close (a comma would owe a key the closed object lacks)");
  {
    JsonMachine over = m;
    require(!over.feed('1'), "20001 refused at its digit");
  }
  require(m.feed('}') && m.done(), "closed");
  // A lower bound refuses the close until the digits reach it; a value
  // that cannot grow (0) dies at once.
  const auto ten = compile(R"({"type":"integer","minimum":10})");
  JsonMachine t(ten, &tables());
  require(!JsonMachine(ten, &tables()).feed('0'), "0 under minimum 10 dies at its byte");
  require(!JsonMachine(ten, &tables()).feed('-'), "a sign under minimum 10 dies at its byte");
  require(t.feed('5') && !t.done(), "5: alive (50 is reachable), not done");
  t.mask(vocab(), &mask);
  require(mask.allows('0') && mask.allows('9') && mask.allowed > 0, "digits follow");
  require(t.feed('0') && t.done(), "50: done");
  // A negative range: the sign is the only start; the digits then aim
  // inside [-10, -5].
  const auto neg = compile(R"({"type":"integer","minimum":-10,"maximum":-5})");
  JsonMachine g(neg, &tables());
  g.mask(vocab(), &mask);
  require(mask.allows('-') && !mask.allows('1') && !mask.allows('0') &&
              mask.allows(kWordBase + 24) /* -1 */ && mask.allows(kWordBase + 86) /* -5 */ &&
              mask.allows(kWordBase + 78) /* - */ && !mask.allows(kWordBase + 79) /* 9 */,
          "only a signed start");
  require(g.feed('-'), "-");
  g.mask(vocab(), &mask);
  require(mask.allows('1') && mask.allows('5') && mask.allows('9') && !mask.allows('0') &&
              !mask.allows('2') && !mask.allows('4') && mask.allows(kWordBase + 29) /* 10 */ &&
              !mask.allows(kWordBase + 21) /* 12 */,
          "-1 (toward -10) and -5..-9; not -0, -2, -4; -10 as one token, not -12");
  require(g.feed('1') && !g.done(), "-1: alive, not done");
  g.mask(vocab(), &mask);
  require(mask.allows('0') && !mask.allows('1') && !mask.allows('9'), "-10 only");
  require(g.feed('0') && g.done(), "-10: done");
  // Union semantics at the top: done where some alternative accepts.
  const auto mixed = compile(R"({"anyOf":[{"type":"integer","minimum":100},{"type":"integer","maximum":10}]})");
  JsonMachine u(mixed, &tables());
  require(u.feed('5') && u.done(), "5 fits the second alternative");
  u.mask(vocab(), &mask);
  require(mask.allows('0') && mask.allows('9'), "the first alternative keeps the digits open");
  require(u.feed('0') && !u.done(), "50 fits neither");
  require(u.feed('0') && u.done(), "500 fits the first");
  // Array items under a range; integer|null under a range.
  const auto arr = compile(R"({"type":"array","items":{"type":"integer","minimum":0,"maximum":3}})");
  JsonMachine a(arr, &tables());
  require(a.feed('['), "[");
  a.mask(vocab(), &mask);
  require(mask.allows('0') && mask.allows('3') && !mask.allows('4') && mask.allows(']') &&
              mask.allows('-') /* "-0" is 0 */ && !mask.allows(kWordBase + 24) /* -1 */,
          "items in [0, 3]");
  require(a.feed('3') && a.feed(',') && !a.feed('4'), "4 refused as an item");
  const auto opt = compile(R"({"type":["integer","null"],"minimum":1})");
  JsonMachine o(opt, &tables());
  o.mask(vocab(), &mask);
  require(mask.allows('n') && mask.allows('1') && !mask.allows('0') && !mask.allows('-'),
          "null or a positive integer");
  // Huge digits: no upper bound, still alive and done.
  const auto pos = compile(R"({"type":"integer","minimum":0})");
  JsonMachine h(pos, &tables());
  for (int i = 0; i < 25; ++i) require(h.feed(i == 0 ? '1' : '0'), "a 25-digit integer");
  require(h.done(), "done at 25 digits");
  // A number's bound is enforced under the lax tool-argument compile too.
  std::vector<std::string> notes;
  const dgpp::minijson::ParseResult np = dgpp::minijson::parse(R"({"type":"number","minimum":5})");
  const auto num = std::make_shared<const JsonSchema>(dgpp::text::compile_json_schema(np.root, &notes));
  JsonMachine f(num, &tables());
  require(notes.empty() && f.feed('1') && !f.done() && f.feed('0') && f.done(),
          "a number cannot finish below its minimum");
}

DGPP_TEST(json_grammar_numberBoundsAcceptDecimalsAndExponents) {
  const auto check = [&](const std::string& schema, const std::string& text, bool expected) {
    JsonMachine m(compile(schema), &tables());
    bool accepted = true;
    for (const unsigned char ch : text) {
      TokenMask fast, slow;
      m.mask(vocab(), &fast);
      m.mask_brute_force(vocab(), &slow);
      std::string diff;
      if (fast.words != slow.words)
        for (int id = 0; id < kVocab; ++id)
          if (fast.allows(id) != slow.allows(id))
            diff += " " + std::to_string(id) + "(" + vocab().text(id) + ")";
      require(fast.words == slow.words && fast.allowed == slow.allowed,
              "number mask differs before '" + text + "' byte " + std::to_string(ch) + diff);
      if (!m.feed(ch)) { accepted = false; break; }
    }
    require((accepted && m.done()) == expected, schema + " on " + text);
    if (accepted) {
      TokenMask fast, slow;
      m.mask(vocab(), &fast);
      m.mask_brute_force(vocab(), &slow);
      require(fast.words == slow.words, "number mask differs after " + text);
    }
  };
  const std::string range = R"({"type":"number","minimum":0,"maximum":10})";
  for (const char* text : {"0", "-0", "0.25", "9.999", "10.0", "1e1", "100e-1",
                           "1E+0001", "1e-00000000000000000000000001", "0e999999999999999999999"})
    check(range, text, true);
  for (const char* text : {"-0.1", "10.000000000000000000001", "11", "1e2",
                           "100e+1", "1e99999999999999999999999", "1.", "1e-"})
    check(range, text, false);
  const std::string narrow = R"({"type":"number","exclusiveMinimum":0.1,"exclusiveMaximum":0.2})";
  for (const char* text : {"0.100000000000000000001", "0.15", "15e-2", "0.199999999999999999999"})
    check(narrow, text, true);
  for (const char* text : {"0.1", "0.2", "1e-1", "2e-1", "1e-999999999999999999"})
    check(narrow, text, false);
  const std::string negative = R"({"type":"number","minimum":-2.5,"exclusiveMaximum":-1.25})";
  for (const char* text : {"-2.5", "-2", "-12501e-4"}) check(negative, text, true);
  for (const char* text : {"-2.50001", "-1.25", "0", "1"}) check(negative, text, false);
  check(R"({"type":"number","minimum":5,"maximum":5})", "50e-1", true);
  check(R"({"type":"number","minimum":5,"maximum":5})", "6", false);
  check(R"({"minimum":0.5})", "null", true);
  check(R"({"type":["number","null"],"minimum":0.5})", "null", true);
  check(R"({"type":"array","items":{"type":"number","minimum":0,"maximum":10}})", "[0.5,1e1]", true);
  check(R"({"type":"array","items":{"type":"number","minimum":0,"maximum":10}})", "[0.5,11]", false);
  check(R"({"anyOf":[{"type":"integer","maximum":0},{"type":"number","minimum":1.5,"maximum":2}]})", "1.75", true);
  check(R"({"anyOf":[{"type":"integer","maximum":0},{"type":"number","minimum":1.5,"maximum":2}]})", "1.25", false);
  check(R"({"enum":[0.5,1.5,null],"minimum":1})", "1.5", true);
  check(R"({"enum":[0.5,1.5,null],"minimum":1})", "0.5", false);

  // Exhaustive finite values against an independent arithmetic oracle.
  const auto schema = compile(R"({"type":"number","minimum":-1.25,"exclusiveMaximum":2.5})");
  for (int numerator = -350; numerator <= 350; ++numerator) {
    const std::string text = std::to_string(numerator) + "e-2";
    JsonMachine m(schema, &tables());
    bool ok = true;
    for (const unsigned char ch : text) if (!m.feed(ch)) { ok = false; break; }
    require((ok && m.done()) == (numerator >= -125 && numerator < 250),
            "arithmetic oracle disagrees on " + text);
  }
  int finished = 0;
  oracle_walks("decimal", compile(R"({"type":"number","minimum":1.25,"exclusiveMaximum":2.5})"),
               60, 8391, &finished);
  require(finished > 0, "decimal random walks complete");
}

DGPP_TEST(json_grammar_machineFactsAtTheEdges) {
  // JSON mode: the root must be an object — the first token is '{' or
  // whitespace (or a whitespace-prefixed '{'), nothing else.
  JsonMachine m(std::make_shared<const JsonSchema>(dgpp::text::json_object_schema()), &tables());
  TokenMask mask;
  m.mask(vocab(), &mask);
  for (int id = 0; id < kVocab; ++id) {
    const std::string& t = vocab().text(id);
    if (t.empty()) {
      require(!mask.allows(id), "empty id allowed");
      continue;
    }
    size_t i = 0;
    while (i < t.size() && JsonLexer::is_ws(static_cast<uint8_t>(t[i]))) ++i;
    const bool expect = i == t.size() || t[i] == '{';
    if (expect) {
      // ... provided the rest is a valid object prefix.
      require(mask.allows(id) == m.allows(vocab(), id), "root allows");
    } else {
      require(!mask.allows(id), "root allowed '" + t + "'");
    }
  }
  require(mask.allows('{') && mask.allows(' ') && !mask.allows('[') && !mask.allows('"') &&
              !mask.allows('1'),
          "root object starts");
  // A disallowed byte kills the machine; the mask is then empty.
  require(m.feed('{'), "open");
  require(!m.feed(']'), "] after {");
  require(!m.alive(), "dead");
  m.mask(vocab(), &mask);
  require(mask.allowed == 0, "dead mask empty");
  // Done: whitespace only, and done() holds.
  JsonMachine d(std::make_shared<const JsonSchema>(dgpp::text::json_object_schema()), &tables());
  for (const char c : std::string("{\"a\": [1, 2]}")) require(d.feed(static_cast<uint8_t>(c)), "doc");
  require(d.done(), "done");
  d.mask(vocab(), &mask);
  require(mask.allows(' ') && mask.allows('\n') && !mask.allows('}') && !mask.allows(','),
          "done mask");
  require(d.feed(' ') && d.done(), "ws after done");
  // A closed object's keys are spelled from the declared names only, in
  // any tokenization; a foreign key dies at its first byte.
  const auto closed = compile(R"({"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"],"additionalProperties":false})");
  JsonMachine c(closed, &tables());
  for (const char ch : std::string("{\"")) require(c.feed(static_cast<uint8_t>(ch)), "open key");
  c.mask(vocab(), &mask);
  require(mask.allows('c') && mask.allows('d') && !mask.allows('x') && !mask.allows('"'),
          "closed key first bytes");
  require(mask.allows(kWordBase + 36) /* city */ && mask.allows(kWordBase + 37) /* days */,
          "whole key tokens");
  require(!mask.allows(kWordBase + 40) /* abc */, "foreign key token");
  for (const char ch : std::string("city\":")) require(c.feed(static_cast<uint8_t>(ch)), "city key");
  c.mask(vocab(), &mask);
  require(mask.allows('"') && mask.allows(' ') && !mask.allows('1') && !mask.allows('{'),
          "string value expected");
  for (const char ch : std::string("\"Oslo\",\"")) require(c.feed(static_cast<uint8_t>(ch)), "value, next key");
  c.mask(vocab(), &mask);
  require(mask.allows('d') && !mask.allows('c'), "city used, days left");
  for (const char ch : std::string("days\":")) require(c.feed(static_cast<uint8_t>(ch)), "days key");
  c.mask(vocab(), &mask);
  require(mask.allows('1') && mask.allows('-') && !mask.allows('"') && !mask.allows('t'),
          "integer expected");
  require(c.feed('3'), "3");
  c.mask(vocab(), &mask);
  require(!mask.allows('.') && !mask.allows('e') && mask.allows('}') && !mask.allows(','),
          "integer-only, no keys left");
  require(c.feed('}') && c.done(), "closed done");
  // The whitespace-run cap: sixteen structural whitespace bytes pass, the
  // seventeenth is refused, a structural byte is still allowed; the mask
  // drops the tokens whose leading run would overflow the budget; and a
  // complete text with the cap reached admits nothing (the grammar layer
  // adds EOS).
  JsonMachine w(std::make_shared<const JsonSchema>(dgpp::text::json_object_schema()), &tables());
  for (int i = 0; i < 15; ++i) require(w.feed('\n'), "whitespace within the cap");
  w.mask(vocab(), &mask);
  require(mask.allows(' ') && mask.allows('{') && !mask.allows(kWordBase + 44) /* "  " */ &&
              !mask.allows(kWordBase + 42) /* "\n  " */,
          "one byte of budget left: single whitespace and structure only");
  require(w.feed('\n'), "the sixteenth whitespace byte");
  w.mask(vocab(), &mask);
  require(!mask.allows(' ') && !mask.allows('\n') && mask.allows('{') && !mask.allows(kWordBase + 43),
          "budget spent: structure only");
  {
    JsonMachine over = w;
    require(!over.feed('\n'), "the seventeenth whitespace byte is refused");
  }
  require(w.feed('{') && w.feed('}') && w.done(), "the brace resets the run");
  for (int i = 0; i < 16; ++i) require(w.feed(' '), "trailing whitespace within the cap");
  w.mask(vocab(), &mask);
  require(mask.allowed == 0 && w.done(), "complete and capped: nothing but the end");
  // Content whitespace inside a string never counts.
  JsonMachine sw(std::make_shared<const JsonSchema>(dgpp::text::json_object_schema()), &tables());
  for (const char ch : std::string("{\"a\": \"")) require(sw.feed(static_cast<uint8_t>(ch)), "open string");
  for (int i = 0; i < 40; ++i) require(sw.feed(' '), "spaces are string content");
  require(sw.feed('"'), "the string closes");
  require(sw.lexer().ws_run() == 0, "the run is zero after content");
  // Required keys missing: the closer is refused.
  JsonMachine r(closed, &tables());
  for (const char ch : std::string("{\"days\":1")) require(r.feed(static_cast<uint8_t>(ch)), "days only");
  r.mask(vocab(), &mask);
  require(!mask.allows('}') && mask.allows(','), "city still owed");
}

DGPP_TEST(json_grammar_openaiReferencesPatternsFormatsAndMultiples) {
  const auto accepts = [](const std::string& schema, const std::string& text) {
    JsonMachine machine(compile(schema), &tables());
    for (unsigned char c : text) if (!machine.feed(c)) return false;
    return machine.done();
  };
  const std::string recursive = R"({"type":"object","properties":{"value":{"type":"integer"},"next":{"anyOf":[{"$ref":"#"},{"type":"null"}]}},"required":["value","next"],"additionalProperties":false})";
  require(accepts(recursive, R"({"value":1,"next":{"value":2,"next":null}})"), "recursive root object");
  const std::string containers = R"({"enum":[{"a":[1,{"b":true}],"s":"x"},[false,"z"],null]})";
  for (const auto& value : {R"({"a": [1, {"b": true}], "s": "x"})", R"([false, "z"])", "null"}) {
    require(accepts(containers, value), "container enum accepts a nested target");
    JsonMachine candidate(compile(containers), &tables());
    for (unsigned char c : std::string(value)) {
      TokenMask actual, oracle;
      candidate.mask(vocab(), &actual); candidate.mask_brute_force(vocab(), &oracle);
      require(actual.words == oracle.words && actual.allows(c), "container enum masks match exhaustive simulation");
      require(candidate.feed(c), "enum target byte accepted");
    }
  }
  require(!accepts(containers, R"({"a": [1, {"b": false}], "s": "x"})"), "nested enum value enforced");
  require(!accepts(containers, R"([false, "z", 1])"), "container enum length enforced");
  require(!accepts(recursive, R"({"value":1,"next":{"value":"bad","next":null}})"), "nested recursive type enforced");
  const std::string defs = R"({"type":"object","properties":{"id":{"$ref":"#/$defs/a~1b~0c"}},"required":["id"],"additionalProperties":false,"$defs":{"a/b~c":{"type":"string","pattern":"^[A-Z]{2}[0-9]{2}$"}}})";
  require(accepts(defs, R"({"id":"AB12"})"), "escaped pointer resolves pattern");
  require(accepts(defs, R"({"id":"\u0041B12"})"), "pattern sees decoded JSON escapes");
  require(!accepts(defs, R"({"id":"ab12"})"), "pattern rejects wrong case");
  require(!accepts(defs, R"({"id":"AB12x"})"), "anchored pattern rejects suffix");
  require(accepts(R"({"type":"string","pattern":"a+"})", R"("before aaa after")"), "unanchored schema pattern is search");
  for (const auto& value : {R"(".a@example.org")", R"("a..b@example.org")", R"("a@-example.org")"})
    require(!accepts(R"({"type":"string","format":"email"})", value), "email dot-atom and domain syntax");
  for (const auto& [format, good, bad] : std::vector<std::array<std::string,3>>{
    {"date","2024-02-29","2025-02-29"}, {"date-time","2024-01-01T12:34:56Z","2024-01-01T25:34:56Z"},
    {"time","12:34:56+01:00","12:99:56Z"}, {"duration","P1DT2H","PT"},
    {"email","a@example.org","missing-at"}, {"hostname","example.org","-example.org"},
    {"ipv4","192.168.50.51","256.1.2.3"}, {"ipv6","2001:db8::1","1:2:3"},
    {"uuid","550e8400-e29b-41d4-a716-446655440000","550e8400-e29b-41d4-a716-44665544000Z"}}) {
    auto schema = std::string("{\"type\":\"string\",\"format\":\"") + format + "\"}";
    require(accepts(schema, "\""+good+"\""), "valid " + format);
    require(!accepts(schema, "\""+bad+"\""), "invalid " + format);
  }
  for (const auto& value : {"0", "0.3", "-0.3", "3e2", "0.300000000000000000000"})
    require(accepts(R"({"type":"number","multipleOf":0.1})", value), std::string("exact decimal multiple: ") + value);
  require(!accepts(R"({"type":"number","multipleOf":0.1})", "0.300000000000000000001"), "no floating-point rounding of divisibility");
  require(accepts(R"({"type":"number","multipleOf":3})", "3e9999"), "large exponent divisibility without huge allocation");
  require(!accepts(R"({"type":"number","multipleOf":3})", "1e9999"), "large exponent nonmultiple");
  for (const auto& schema : {R"({"$ref":"#/missing"})", R"({"$ref":"#"})", R"({"type":"number","multipleOf":0})"}) {
    bool failed = false;
    try { compile(schema); } catch (const std::invalid_argument&) { failed = true; }
    require(failed, std::string("invalid schema rejected: ") + schema);
  }
  // A mask is tested against complete token texts, including tokens spanning
  // an escape or the closing quote and the parent object's delimiter.
  JsonMachine machine(compile(defs), &tables());
  for (unsigned char c : std::string(R"({"id":"A)")) require(machine.feed(c), "pattern prefix");
  TokenMask actual, oracle;
  machine.mask(vocab(), &actual); machine.mask_brute_force(vocab(), &oracle);
  require(actual.words == oracle.words && actual.allows('B') && !actual.allows('b'), "pattern token mask");
  JsonMachine alternatives(compile(R"({"type":"string","pattern":"^A(B|C)$"})"), &tables());
  require(alternatives.feed('"'), "open anchored pattern string");
  alternatives.mask(vocab(), &actual);
  require(actual.allows('A') && !actual.allows('b'), "nested alternatives retain anchored prefix pruning");
}

DGPP_TEST(json_grammar_freeStringMasksCheckFollowingFields) {
  const auto schema = compile(
      R"({"type":"object","properties":{"notes":{"type":"string"},"date":{"type":"string","format":"date"},"n":{"type":"number","multipleOf":0.1}},"required":["notes","date","n"],"additionalProperties":false})");
  std::vector<std::string> texts = {"tail",
                                    R"(","date":"2024-02-29","n":0.3})",
                                    R"(","date":"2025-02-29","n":0.3})",
                                    R"(","date":"2024-02-29","n":0.31})",
                                    R"(","date":"2024-02-29"})",
                                    R"(","notes":"duplicate","date":"2024-02-29","n":0.3})",
                                    R"(\u0041","date":"2024-02-29","n":0.3})",
                                    R"(","date":"2024-02-29","n":0.3,"other":1})"};
  const bool expected[] = {true, true, false, false, false, false, true, false};
  for (int b = 0; b < 256; ++b) texts.emplace_back(1, static_cast<char>(b));
  const int eos = static_cast<int>(texts.size());
  texts.emplace_back();
  const int marker = static_cast<int>(texts.size());
  texts.emplace_back("<tool_call>");
  ChatMarkers markers;
  markers.tool_call_open = ChatMarker{marker, "<tool_call>"};
  GrammarVocab v(std::move(texts), markers, {eos}, marker + 1);
  JsonMachine machine(schema, &v.json_tables());
  for (unsigned char b : std::string(R"({"notes":"free)")) require(machine.feed(b), "free prefix");
  const auto exact = [&](const JsonMachine& m) {
    TokenMask actual, oracle, repeated;
    m.mask(v, &actual);
    m.mask_brute_force(v, &oracle);
    m.mask(v, &repeated);
    require(actual.words == oracle.words && actual.allowed == oracle.allowed &&
                actual.vocab == oracle.vocab && actual.words == repeated.words &&
                actual.allowed == repeated.allowed && actual.vocab == repeated.vocab,
            "free string masks equal simulation, including a warm cache");
    return actual;
  };
  const auto mask = exact(machine);
  for (int i = 0; i < 8; ++i) require(mask.allows(i) == expected[i], "following-field constraint");
  const auto saved = machine;  // Warm-cache copies and rollback must stay independent.
  for (int b = 0; b < 256; ++b) {
    machine = saved;
    machine.feed(static_cast<uint8_t>(b));
    exact(machine);
  }
  for (const auto& suffix : {R"(\uD83D\uDE00)", R"(\u00)", R"(","date":"2024-02-29","n":0.3})"}) {
    machine = saved;
    for (unsigned char b : std::string(suffix)) {
      require(machine.feed(b), "continuation accepted");
      exact(machine);
    }
  }
  machine = saved;
  require(exact(machine).words == mask.words, "restored parent mask unchanged");

  dgpp::text::GrammarSpec spec;
  spec.mode = dgpp::text::GrammarSpec::Mode::kJson;
  spec.json_schema =
      R"({"type":"object","properties":{"notes":{"type":"string"},"date":{"type":"string","format":"date"},"n":{"type":"number","multipleOf":0.1}},"required":["notes","date","n"],"additionalProperties":false})";
  dgpp::text::GrammarState wrapper(&v, spec, false);
  for (unsigned char b : std::string(R"({"notes":"free)")) wrapper.advance(8 + b);
  TokenMask wrapped;
  wrapper.mask(&wrapped);
  require(!wrapped.allows(eos) && !wrapped.allows(marker), "wrapper withholds EOS and markers");
  for (int i = 0; i < 8; ++i)
    require(wrapped.allows(i) == expected[i] && wrapper.allows(i) == expected[i],
            "wrapper masks and pointwise decisions agree on following fields");
  auto after = wrapper;
  after.advance(1);
  after.mask(&wrapped);
  require(after.active() && wrapped.allows(eos), "copied wrapper completes and offers EOS");
  wrapper.mask(&wrapped);
  require(!wrapped.allows(eos), "draft copy did not advance committed wrapper");
}

DGPP_TEST(json_grammar_mixedConstraintMasksSurviveStringTransitions) {
  const std::vector<std::array<std::string, 2>> cases = {
      {R"({"type":"object","properties":{"free":{"type":"string"},"code":{"type":"string","pattern":"^[A-Z]+$"}},"required":["free","code"],"additionalProperties":false})",
       R"({"free":"a\"b\u00e9","code":"ABC"})"},
      {R"({"type":"object","properties":{"free":{"type":"string"},"code":{"enum":["red","blue"]},"n":{"type":"number","multipleOf":0.1}},"additionalProperties":false})",
       R"({"free":"first","code":"red","n":0.3})"},
      {R"({"anyOf":[{"type":"object","properties":{"free":{"type":"string"},"date":{"type":"string","format":"date"}}},{"type":"object","properties":{"free":{"type":"string","pattern":"^a.*$"}}}]})",
       R"({"free":"abc","date":"2024-02-29"})"},
      {R"({"anyOf":[{"type":"object","properties":{"free":{"type":"string"},"date":{"type":"string","format":"date"}}},{"type":"object","properties":{"free":{"type":"string"},"n":{"type":"number"}}}]})",
       R"({"free":"abc","n":2})"},
      {R"({"type":"object","properties":{"free":{"type":"array","items":{"type":"string"}},"date":{"type":"string","format":"date"}}})",
       R"({"free":["a","b"],"date":"2024-02-29"})"},
      {R"({"enum":[{"a":[1,{"b":true}],"s":"x"},[false,"z"],null]})",
       R"({"a": [1, {"b": true}], "s": "x"})"}};
  for (const auto& c : cases) {
    JsonMachine machine(compile(c[0]), &tables());
    for (unsigned char b : c[1]) {
      TokenMask actual, oracle, warm;
      machine.mask(vocab(), &actual);
      machine.mask_brute_force(vocab(), &oracle);
      machine.mask(vocab(), &warm);
      require(actual.words == oracle.words && actual.allowed == oracle.allowed &&
                  actual.vocab == oracle.vocab && actual.words == warm.words &&
                  actual.allowed == warm.allowed && actual.vocab == warm.vocab,
              "mixed schema exact mask across keys, escapes, alternatives and constraints");
      require(actual.allows(b) && machine.feed(b), "document byte allowed: " + c[0] + " document=" +
                                                       c[1] + " byte=" + std::to_string(b));
    }
    require(machine.done(), "mixed document completes");
  }
}

DGPP_TEST(json_grammar_customRegexAndLark) {
  using C = dgpp::text::StringConstraint;
  const C regex("[A-Z]+=[0-9]+", C::Syntax::kRegex);
  require(regex.accepts("ABC=", false) && regex.accepts("ABC=12", true), "custom regex prefix and complete");
  require(!regex.accepts("ABC=x", false) && !regex.accepts("ABC=", true), "custom regex rejects invalid/incomplete input");
  const C lark("start: \"SELECT\" CNAME \"FROM\" CNAME\n%import common.CNAME\n%import common.WS\n%ignore WS\n", C::Syntax::kLark);
  require(lark.accepts("SELECT name FROM users", true), "Lark imports and ignored whitespace");
  require(lark.accepts("SELECT name FR", false), "Lark partial literal");
  require(!lark.accepts("DELETE name FROM users", false), "Lark rejects another statement");
  const C recursive("start: \"(\" start \" )\" | \"x\"", C::Syntax::kLark);
  require(recursive.accepts("((x ) )", true), "recursive custom grammar");
  const C arithmetic("start: start \"+\" INT | INT\n%import common.INT", C::Syntax::kLark);
  require(arithmetic.accepts("1+2+3", true), "direct left recursion");
}

DGPP_TEST(json_grammar_edge_decimalMultiplesAgainstIntegerOracle) {
  // Independent rational arithmetic: generated numbers use cents, so no
  // binary floating-point arithmetic participates in the expected answer.
  for (const auto& [numerator, denominator] : std::vector<std::pair<int,int>>{
      {1,10}, {1,5}, {1,4}, {3,10}, {3,2}, {3,1}, {7,1}}) {
    const std::string schema = "{\"type\":\"number\",\"multipleOf\":" +
        std::to_string(double(numerator)/denominator) + ",\"minimum\":-2,\"maximum\":2}";
    const auto compiled = compile(schema);
    for (int cents = -250; cents <= 250; ++cents) {
      const int magnitude = std::abs(cents);
      const std::string value = std::string(cents < 0 ? "-" : "") + std::to_string(magnitude/100) + "." +
          (magnitude%100 < 10 ? "0" : "") + std::to_string(magnitude%100);
      JsonMachine machine(compiled, &tables());
      bool accepted = true;
      for (unsigned char c : value) if (!machine.feed(c)) { accepted = false; break; }
      accepted = accepted && machine.done();
      const bool expected = cents >= -200 && cents <= 200 && (cents * denominator) % (100 * numerator) == 0;
      require(accepted == expected, "rational oracle disagrees for " + schema + " value " + value);
    }
  }
}

DGPP_TEST(json_grammar_edge_largeArrayLimitsAndUnsatisfiableEnums) {
  for (const auto* schema : {
      R"({"type":"array","maxItems":4294967295})", R"({"type":"array","minItems":2147483648})",
      R"({"type":"array","maxItems":1e100})", R"({"type":"array","maxItems":1.5})",
      R"({"type":"string","enum":["bad"],"pattern":"^[A-Z]+$"})",
      R"({"type":"number","enum":[1,3,5],"multipleOf":2})"}) {
    bool rejected = false;
    try { compile(schema); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, std::string("invalid/unsatisfiable schema must be rejected before admission: ") + schema);
  }
  const auto schema = compile(R"({"type":"string","enum":["BAD","good"],"pattern":"^[a-z]+$"})");
  JsonMachine machine(schema, &tables());
  require(machine.feed('"'), "enum opening quote");
  TokenMask mask; machine.mask(vocab(), &mask);
  require(mask.allows('g') && !mask.allows('B'), "enum and pattern intersect");
}

DGPP_TEST(json_grammar_edge_jsonPointerArraysUnicodeAndCycles) {
  const auto schema = compile(R"({"$defs":{"values":[{"type":"string","pattern":"^ok$"}]},"$ref":"#/$defs/values/0"})");
  JsonMachine machine(schema, &tables());
  for (unsigned char c : std::string(R"("ok")")) require(machine.feed(c), "reference through an array index");
  require(machine.done(), "array reference accepts target schema");
  for (const auto* ref : {"#/$defs/values/01", "#/$defs/values/-1", "#/$defs/values/1", "#/$defs/values/9999999999999999999999999"}) {
    bool rejected = false;
    try { compile(std::string(R"({"$defs":{"values":[{}]},"$ref":")") + ref + "\"}"); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid array pointer rejected");
  }
  for (const auto* text : {R"({"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"})",
                          R"({"$ref":"#/~2"})", R"({"$ref":"https://example.org/schema"})"}) {
    bool rejected = false;
    try { compile(text); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid/cyclic reference rejected without recursion failure");
  }
}

DGPP_TEST(json_grammar_edge_stringUnicodeFormatsAndEscapes) {
  const auto accepts = [](const std::string& schema, const std::string& text) {
    JsonMachine machine(compile(schema), &tables());
    for (unsigned char c : text) if (!machine.feed(c)) return false;
    return machine.done();
  };
  const std::string emoji = R"({"type":"string","pattern":"^😀$"})";
  require(accepts(emoji, R"("😀")") && accepts(emoji, R"("\uD83D\uDE00")"), "UTF-8 and surrogate-pair spellings agree");
  for (const auto* bad : {R"("\uD83D")", R"("\uDE00")", R"("\uD83D\u0041")"})
    require(!accepts(emoji, bad), "unpaired surrogate rejected");
  for (const auto& [format, valid, invalid] : std::vector<std::array<std::string,3>>{
      {"date","2000-02-29","1900-02-29"}, {"time","00:59:60+01:00","12:34:60Z"},
      {"date-time","2016-12-31T23:59:60Z","2016-12-31T12:34:60Z"},
      {"duration","P2W","P1WT1H"}, {"email","first.last+tag@example.org","a..b@example.org"},
      {"hostname",std::string(63,'a')+".org",std::string(64,'a')+".org"},
      {"ipv4","0.0.0.0","01.2.3.4"}, {"ipv6","::ffff:192.0.2.128","1:2:3:4:5:6:7:8::"},
      {"uuid","550E8400-E29B-41D4-A716-446655440000","550e8400e29b41d4a716446655440000"}}) {
    const auto schema = "{\"type\":\"string\",\"format\":\"" + format + "\"}";
    require(accepts(schema, dgpp::text::json_text_of(dgpp::minijson::Value::make_string(valid))), "valid format boundary: " + format);
    require(!accepts(schema, dgpp::text::json_text_of(dgpp::minijson::Value::make_string(invalid))), "invalid format boundary: " + format);
  }
}

DGPP_TEST(json_grammar_edge_larkOfficialExampleAndMalformedGrammar) {
  using C = dgpp::text::StringConstraint;
  const C arithmetic("start: expr\nexpr: term (SP ADD SP term)* -> add\n | term\nterm: factor (SP MUL SP factor)* -> mul\n | factor\nfactor: INT\nSP: \" \"\nADD: \"+\"\nMUL: \"*\"\n%import common.INT", C::Syntax::kLark);
  require(arithmetic.accepts("4 + 4 * 2", true) && !arithmetic.accepts("4+4", true), "official example preserves explicit whitespace");
  const C literals(R"(start: "\u0061" "\x62" "\n")", C::Syntax::kLark);
  require(literals.accepts("ab\n", true), "Lark escaped literals preserve their characters");
  const C nested("start: \"(\" [start] \")\"", C::Syntax::kLark);
  require(nested.accepts("()", true) && nested.accepts("((()))", true) &&
          !nested.accepts("(()", true), "nullable recursive groups remain productive");
  const C right_recursive("start: \"x\" [tail]\ntail: \"y\" start", C::Syntax::kLark);
  require(right_recursive.accepts("x", true) && right_recursive.accepts("xyxyx", true) &&
          !right_recursive.accepts("xy", true), "indirect right recursion remains supported");
  const C zero_repeat("start: \"a\" ~0..2 \"b\"", C::Syntax::kLark);
  require(zero_repeat.accepts("b", true) && zero_repeat.accepts("aab", true) &&
          !zero_repeat.accepts("aaab", true), "zero repetitions have a productive empty branch");
  for (const auto* grammar : {"", "start: missing", "start: (\"x\"", "start: start", "start: a\na: start",
                            "start: \"x\" ~3..1", "start: \"x\" ~1..", "start: X\nX.1: \"x\"", "%declare X\nstart: X"}) {
    bool rejected = false;
    try { C invalid(grammar, C::Syntax::kLark); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, std::string("invalid or unsupported grammar rejected: ") + grammar);
  }
}

}  // namespace
