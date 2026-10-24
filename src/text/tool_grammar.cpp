#include "text/tool_grammar.hpp"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "loaders/minijson.hpp"
#include "text/tokenizer.hpp"

namespace dgpp::text {

struct GrammarVocab::JsonHolder {
  std::once_flag once;
  std::unique_ptr<JsonTables> tables;
};

// ---------------------------------------------------------------------------
// GrammarVocab
// ---------------------------------------------------------------------------

GrammarVocab::GrammarVocab(std::vector<std::string> texts, ChatMarkers markers,
                           std::vector<int64_t> eos_ids, int vocab_size,
                           int64_t call_turn_eos)
    : json_(std::make_shared<JsonHolder>()),
      texts_(std::move(texts)),
      markers_(std::move(markers)),
      eos_(std::move(eos_ids)),
      call_eos_(call_turn_eos),
      vocab_size_(vocab_size) {
  if (vocab_size_ < 1)
    throw std::invalid_argument("GrammarVocab: vocab_size must be positive");
  for (size_t id = 0; id < texts_.size(); ++id) {
    if (texts_[id].empty() || static_cast<int64_t>(id) >= vocab_size_) continue;
    if (markers_.xml_compact && texts_[id].find_first_of("<>\n\r") != std::string::npos)
      xml_end_ids_.push_back(static_cast<int32_t>(id));
    by_first_[static_cast<unsigned char>(texts_[id][0])].push_back(
        static_cast<int32_t>(id));
  }
  for (const ChatMarker* m :
       {&markers_.tool_call_open, &markers_.tool_call_close,
        &markers_.arg_key_open, &markers_.arg_key_close,
        &markers_.arg_value_open, &markers_.arg_value_close, &markers_.dsml})
    if (m->available()) marker_ids_.push_back(m->id);
  if (call_eos_ < 0 && !eos_.empty()) call_eos_ = eos_[0];
}

GrammarVocab GrammarVocab::from_tokenizer(const Tokenizer& tok,
                                          const std::vector<int64_t>& eos_ids,
                                          int vocab_size, DsmlDialect dsml_dialect,
                                          bool model_opens_thinking) {
  const int64_t max_id = tok.max_id();
  std::vector<std::string> texts(
      static_cast<size_t>(std::max<int64_t>(max_id + 1, 0)));
  for (int64_t id = 0; id <= max_id; ++id)
    texts[static_cast<size_t>(id)] = tok.decode(id, /*skip_special_tokens=*/true);
  // The turn-ending id after the last call: <|observation|> when it is an
  // EOS id of this checkpoint, else the first EOS id.
  int64_t call_eos = -1;
  for (const char* turn_end : {"<|observation|>", "<|im_end|>"}) {
    if (call_eos >= 0) break;
    for (const auto& added : tok.added_tokens())
      if (added.content == turn_end)
        for (const int64_t e : eos_ids)
          if (e == added.id) call_eos = e;
  }
  ChatMarkers markers = ChatMarkers::from_tokenizer(tok);
  markers.dsml_dialect = dsml_dialect;
  markers.model_opens_thinking = model_opens_thinking;
  return GrammarVocab(std::move(texts), std::move(markers), eos_ids, vocab_size, call_eos);
}

void GrammarVocab::prepare_json() const {
  std::call_once(json_->once,
                 [&] { json_->tables = std::make_unique<JsonTables>(*this); });
}

const JsonTables& GrammarVocab::json_tables() const {
  prepare_json();
  return *json_->tables;
}

bool GrammarVocab::is_eos(int64_t id) const {
  for (const int64_t e : eos_)
    if (e == id) return true;
  return false;
}

// ---------------------------------------------------------------------------
// The tool entry from a function definition (M6 6g keys, 6i typed values)
// ---------------------------------------------------------------------------

namespace {

std::string rendered_enum_text(const minijson::Value& v) {
  // What the template writes for the value: a string raw, anything else
  // through tojson.
  return v.is_string() ? std::string(v.as_string()) : json_text_of(v);
}

bool names_string(const minijson::Value& type) {
  if (type.is_string()) return type.as_string() == "string";
  if (type.is_array())
    for (const minijson::Value& e : type.items())
      if (e.is_string() && e.as_string() == "string") return true;
  return false;
}

// The property admits a raw-text value (string-typed, untyped, or a type
// list / anyOf with such an alternative): the grammar cannot type it.
bool string_possible(const minijson::Value& prop) {
  if (const minijson::Value* any = prop.find("anyOf")) {
    if (!any->is_array()) return true;
    for (const minijson::Value& alt : any->items())
      if (!alt.is_object() || string_possible(alt)) return true;
    return false;
  }
  const minijson::Value* type = prop.find("type");
  if (type == nullptr) return true;
  if (type->is_string()) return type->as_string() == "string";
  if (type->is_array()) {
    if (type->items().empty()) return true;
    return names_string(*type);
  }
  return true;  // an unrecognised type value: leave it free
}

// Keep local references relative to the complete parameter schema, including
// root recursion, when an argument is decoded by its own JSON machine.
minijson::Value relocate_refs(const minijson::Value& v) {
  if (v.is_array()) {
    std::vector<minijson::Value> items;
    for (const auto& item : v.items()) items.push_back(relocate_refs(item));
    return minijson::Value::make_array(std::move(items));
  }
  if (!v.is_object()) return v;
  std::vector<minijson::Member> members;
  for (const auto& m : v.members()) {
    if (m.key == "$ref" && m.value.is_string() && m.value.as_string().starts_with("#"))
      members.push_back({m.key, minijson::Value::make_owned_string(
          "#/$defs/__dgpp_parameters" + std::string(m.value.as_string().substr(1)))});
    else members.push_back({m.key, relocate_refs(m.value)});
  }
  return minijson::Value::make_object(std::move(members));
}

minijson::Value argument_schema(const minijson::Value& params, const std::string& key) {
  using V = minijson::Value;
  std::string pointer;
  for (char c : key) pointer += c == '~' ? "~0" : c == '/' ? "~1" : std::string(1, c);
  return V::make_object({
    {"$defs", V::make_object({{"__dgpp_parameters", relocate_refs(params)}})},
    {"$ref", V::make_owned_string("#/$defs/__dgpp_parameters/properties/" + pointer)}});
}

}  // namespace

GrammarTool grammar_tool_from_function(const minijson::Value& def,
                                       std::vector<std::string>* warnings,
                                       std::vector<std::string>* notes) {
  GrammarTool tool;
  if (const minijson::Value* name = def.find("name"))
    tool.name = std::string(name->as_string());
  const minijson::Value* strict_v = def.find("strict");
  const bool strict = strict_v != nullptr && strict_v->is_bool() && strict_v->as_bool();
  tool.strict = strict;
  const minijson::Value* params = def.find("parameters");
  if (params == nullptr || !params->is_object()) return tool;
  if (strict) {
    try { compile_json_schema(*params); }
    catch (const std::invalid_argument& e) {
      const std::string what = e.what();
      throw std::invalid_argument("parameters" + (what.starts_with("schema") ? what.substr(6) : ": " + what));
    }
  }
  const minijson::Value* props = params->find("properties");
  const minijson::Value* extra = params->find("additionalProperties");
  if (props == nullptr || !props->is_object()) return tool;
  // Constrained generation defaults to the declared top-level names. Only
  // explicit true opts out; a subschema cannot type an unknown name here.
  // Nested objects still use the JSON machine's schema semantics.
  const bool open_keys = extra != nullptr && extra->is_bool() && extra->as_bool();
  if (open_keys) {
    if (notes != nullptr)
      notes->push_back("additionalProperties: true on '" + tool.name +
                       "': the keys are free text, not the declared properties");
  } else {
    tool.constrain_keys = true;
    for (const minijson::Member& pm : props->members()) tool.keys.push_back(pm.key);
  }
  // `required`: the names among the declared properties (a name outside
  // them can never be satisfied — dropped, with a warning under strict).
  if (const minijson::Value* req = params->find("required");
      req != nullptr && req->is_array()) {
    for (const minijson::Value& e : req->items()) {
      if (!e.is_string()) continue;
      const std::string key(e.as_string());
      if (props->find(key) != nullptr) {
        if (std::find(tool.required_keys.begin(), tool.required_keys.end(),
                      key) == tool.required_keys.end())
          tool.required_keys.push_back(key);
      } else if (strict && warnings != nullptr) {
        warnings->push_back("required key '" + key + "' of '" + tool.name +
                            "' is not a declared property; not enforced");
      }
    }
  }
  for (const minijson::Member& pm : props->members()) {
    GrammarArg arg;
    arg.key = pm.key;
    const minijson::Value& prop = pm.value;
    const auto contextual = argument_schema(*params, pm.key);
    const std::string path = "parameters.properties." + pm.key;
    if (!prop.is_object()) {
      if (strict) throw std::invalid_argument(path + ": must be a schema object");
      tool.args.push_back(std::move(arg));
      continue;
    }
    if (strict) {
      // Every property must lie inside the enforceable subset.
      try {
        compile_json_schema(contextual);
      } catch (const std::invalid_argument& e) {
        const std::string what = e.what();  // "schema.<path>: reason"
        throw std::invalid_argument(
            path + (what.rfind("schema", 0) == 0 ? what.substr(6) : ": " + what));
      }
    }
    const minijson::Value* en = prop.find("enum");
    const minijson::Value* cs = prop.find("const");
    if (string_possible(prop) && !prop.find("pattern") && !prop.find("format") &&
        !prop.find("$ref") && !prop.find("anyOf") && !prop.find("x-dgpp-grammar")) {
      // Raw text — typable only through an enum's exact texts.
      if (en != nullptr && en->is_array() && !en->items().empty() &&
          prop.find("anyOf") == nullptr) {
        arg.kind = GrammarArg::Kind::kText;
        for (const minijson::Value& e : en->items())
          arg.texts.push_back(rendered_enum_text(e));
      } else if (cs != nullptr && prop.find("anyOf") == nullptr) {
        arg.kind = GrammarArg::Kind::kText;
        arg.texts.push_back(rendered_enum_text(*cs));
      }
      tool.args.push_back(std::move(arg));
      continue;
    }
    // A JSON-typed property: the machine under its own schema. Numeric
    // minimum / maximum are enforced; a keyword that only narrows the
    // value without an automaton behind it (pattern, ...) is tolerated —
    // the value stays typed,
    // the narrowing is not applied — and noted with the reason.
    try {
      std::vector<std::string> unenforced;
      compile_json_schema(contextual, &unenforced);
      arg.kind = GrammarArg::Kind::kJson;
      arg.schema = json_text_of(contextual);
      if (notes != nullptr)
        for (const std::string& u : unenforced) {
          // "schema.<path>.<keyword>: <reason>"
          const std::string entry = u.rfind("schema.", 0) == 0 ? u.substr(7) : u;
          const size_t colon = entry.find(": ");
          const std::string where = entry.substr(0, colon);
          const std::string reason =
              colon == std::string::npos
                  ? "the value keeps its type; the bound is not applied"
                  : entry.substr(colon + 2);
          notes->push_back("argument '" + pm.key + "' of '" + tool.name + "': " +
                           where + " is not enforced (" + reason + ")");
        }
    } catch (const std::invalid_argument& e) {
      if (warnings != nullptr)
        warnings->push_back("argument '" + pm.key + "' of '" + tool.name +
                            "' is outside the constrained subset (" + e.what() +
                            "); its value stays free text");
    }
    tool.args.push_back(std::move(arg));
  }
  return tool;
}

// ---------------------------------------------------------------------------
// GrammarState
// ---------------------------------------------------------------------------

GrammarState::GrammarState(const GrammarVocab* vocab, GrammarSpec spec,
                           bool prompt_opens_thinking, bool model_may_open_thinking)
    : vocab_(vocab), spec_(std::move(spec)) {
  if (!spec_.active()) return;
  think_openable_ = model_may_open_thinking && !prompt_opens_thinking && vocab_ != nullptr &&
                    vocab_->markers().think_open.available() && vocab_->markers().think_close.available();
  if (spec_.has_json()) {
    if (vocab_ == nullptr || vocab_->eos_ids().empty())
      throw std::invalid_argument(
          "GrammarState: a JSON grammar needs a vocabulary with an EOS id");
    std::shared_ptr<const JsonSchema> schema;
    if (spec_.json_schema.empty()) {
      schema = std::make_shared<const JsonSchema>(json_object_schema());
    } else {
      minijson::ParseResult parsed;
      try {
        parsed = minijson::parse(spec_.json_schema);
      } catch (const std::exception& e) {
        throw std::invalid_argument(
            std::string("GrammarState: the JSON schema text does not parse: ") +
            e.what());
      }
      schema = std::make_shared<const JsonSchema>(compile_json_schema(parsed.root));
    }
    json_ = JsonMachine(std::move(schema), &vocab_->json_tables());
    state_ = prompt_opens_thinking && vocab_->markers().think_close.available()
                 ? State::kThink
                 : State::kJsonBody;
    if (spec_.mode == GrammarSpec::Mode::kJson) return;
  }
  if (vocab_ == nullptr || !vocab_->usable())
    throw std::invalid_argument(
        "GrammarState: an active grammar needs a vocabulary with the "
        "tool-call markers and an EOS id");
  if (spec_.mode == GrammarSpec::Mode::kNamed) {
    bool found = false;
    for (const GrammarTool& t : spec_.tools) found = found || t.name == spec_.named;
    if (!found)
      throw std::invalid_argument(
          "GrammarState: the named function is not among the tools");
  }
  if ((spec_.mode == GrammarSpec::Mode::kRequired ||
       spec_.mode == GrammarSpec::Mode::kNamed) &&
      spec_.tools.empty())
    throw std::invalid_argument("GrammarState: a required call needs tools");
  // The typed arguments' schemas, compiled once per request.
  arg_schemas_.resize(spec_.tools.size());
  for (size_t t = 0; t < spec_.tools.size(); ++t) {
    const GrammarTool& tool = spec_.tools[t];
    arg_schemas_[t].resize(tool.args.size());
    for (size_t a = 0; a < tool.args.size(); ++a) {
      const GrammarArg& arg = tool.args[a];
      if (arg.kind != GrammarArg::Kind::kJson) continue;
      minijson::ParseResult parsed;
      try {
        parsed = minijson::parse(arg.schema);
      } catch (const std::exception& e) {
        throw std::invalid_argument("GrammarState: the schema text of argument '" +
                                    arg.key + "' of '" + tool.name +
                                    "' does not parse: " + e.what());
      }
      // A tool argument's text tolerates the narrowing keywords a
      // non-strict definition carries (a strict one passed the strict
      // compile at its definition; the value's type is what the mask uses).
      std::vector<std::string> unenforced;
      arg_schemas_[t][a] = std::make_shared<const JsonSchema>(
          compile_json_schema(parsed.root, &unenforced));
    }
  }
  state_ = prompt_opens_thinking && vocab_->markers().think_close.available()
               ? State::kThink
               : State::kTop;
}

const GrammarArg* GrammarState::current_arg() const {
  const GrammarTool* t = current_tool();
  if (t == nullptr || arg_ < 0 || arg_ >= static_cast<int>(t->args.size()))
    return nullptr;
  return &t->args[static_cast<size_t>(arg_)];
}

const char* GrammarState::state_name() const {
  if (!active()) return dead_ ? "dead" : "inactive";
  switch (state_) {
    case State::kThink: return "think";
    case State::kTop: return "top";
    case State::kName: return "name";
    case State::kKey: return "key";
    case State::kAfterKey: return "after-key";
    case State::kValue: return "value";
    case State::kAfterValue: return "after-value";
    case State::kEnd: return "end";
    case State::kDone: return "done";
    case State::kJsonBody: return json_.state_name();
    case State::kQName: return "q-name";
    case State::kDCalls: return "d-calls";
    case State::kDInvoke: return "d-invoke";
    case State::kDParamOrClose: return "d-param-or-close";
    case State::kDFreeKey: return "d-free-key";
    case State::kDFlag: return "d-flag";
    case State::kDValue: return "d-value";
    case State::kDInvokeOrClose: return "d-invoke-or-close";
    case State::kQKeyOrClose: return "q-key-or-close";
    case State::kQFreeKey: return "q-free-key";
    case State::kQValue: return "q-value";
    case State::kQClose: return "q-close";
  }
  return "?";
}

namespace {
// Qwen's newline form and MiMo's compact form share the key prefix.
constexpr const char* kQParam = "<parameter=";
bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}
bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
// The DSML literals with the tag as the sentinel byte (the parser's
// DsmlTags with "｜DSML｜" replaced), per dialect: V4.1 writes " calls" /
// " invoke" / " parameter" after the tag, V4 "tool_calls" / "invoke" /
// "parameter".
const std::string kDTag(1, GrammarState::kDsmlSentinel);
const std::string kDInvokeHeadEnd = "\">\n";
const std::string kDParamFlag = "\" string=\"";
const std::string kDFlagTrue = "true\">";
const std::string kDFlagFalse = "false\">";
struct DsmlLiterals {
  std::string calls_tail;    // after the opening "<" TAG
  std::string invoke_open;   // through name="
  std::string param_open;    // through name="
  std::string param_close;
  std::string invoke_close;
  std::string calls_close;
  // The close of an invoke that carries no parameter, as the reference
  // renders it — a blank line first ("<invoke name=...>\n\n</invoke>\n",
  // its empty parameter list between the template's two newlines) — offered
  // beside the plain close while no parameter has opened. Empty: not
  // offered (V4.1's grammar predates the finding and is left as it was).
  std::string empty_invoke_close;
  DsmlLiterals(const std::string& calls, const std::string& invoke, const std::string& parameter,
               bool blank_line_close)
      : calls_tail(calls + ">\n"),
        invoke_open("<" + kDTag + invoke + " name=\""),
        param_open("<" + kDTag + parameter + " name=\""),
        param_close("</" + kDTag + parameter + ">\n"),
        invoke_close("</" + kDTag + invoke + ">\n"),
        calls_close("</" + kDTag + calls + ">"),
        empty_invoke_close(blank_line_close ? "\n" + invoke_close : std::string()) {}
};
const DsmlLiterals& dsml_literals(const GrammarVocab* vocab) {
  static const DsmlLiterals v41(" calls", " invoke", " parameter", /*blank_line_close=*/false);
  static const DsmlLiterals v4("tool_calls", "invoke", "parameter", /*blank_line_close=*/true);
  return vocab != nullptr && vocab->markers().dsml_dialect == DsmlDialect::kV4 ? v4 : v41;
}
}  // namespace

const char* GrammarState::xml_function_open() const {
  return vocab_->markers().xml_compact ? "<function=" : "\n<function=";
}
const char* GrammarState::xml_header_end() const {
  return vocab_->markers().xml_compact ? ">" : ">\n";
}
const char* GrammarState::xml_parameter_end() const {
  return vocab_->markers().xml_compact ? "</parameter>" : "\n</parameter>\n";
}
const char* GrammarState::xml_function_end() const {
  return vocab_->markers().xml_compact ? "</function>" : "</function>\n";
}

bool GrammarState::compact_xml_text_state() const {
  if (!vocab_->markers().xml_compact) return false;
  return state_ == State::kQName || state_ == State::kQKeyOrClose || state_ == State::kQFreeKey ||
         state_ == State::kQValue;
}

// A tokenizer token may span several XML fields. Validate all its bytes,
// committing each completed field before interpreting the following byte.
bool GrammarState::compact_xml_text(const std::string& text) {
  const auto match_byte = [&](char ch) {
    match_.emitted += ch;
    for (const auto& target : match_.targets)
      if (target.compare(0, match_.emitted.size(), match_.emitted) == 0) return true;
    return false;
  };
  for (const char ch : text) {
    if (state_ == State::kQName || state_ == State::kQKeyOrClose) {
      if (!match_byte(ch)) return false;
      if (!match_.complete()) continue;
      if (state_ == State::kQName) {
        const std::string name =
            match_.emitted.substr(std::strlen(xml_function_open()),
                                  match_.emitted.size() - std::strlen(xml_function_open()) - 1);
        tool_ = -1;
        used_keys_.clear();
        for (size_t i = 0; i < spec_.tools.size(); ++i)
          if (spec_.tools[i].name == name) tool_ = static_cast<int>(i);
        enter(State::kQKeyOrClose);
      } else if (match_.emitted == xml_function_end()) {
        enter(State::kQClose);
      } else if (match_.emitted == kQParam) {
        enter(State::kQFreeKey);
      } else {
        key_ = match_.emitted.substr(std::strlen(kQParam),
                                     match_.emitted.size() - std::strlen(kQParam) - 1);
        used_keys_.push_back(key_);
        enter(State::kQValue);
      }
      continue;
    }
    if (state_ == State::kQFreeKey) {
      if (ch == '>') {
        if (match_.emitted.empty() || key_used(match_.emitted)) return false;
        key_ = match_.emitted;
        used_keys_.push_back(key_);
        enter(State::kQValue);
      } else {
        if (ch == '<' || ch == '\n' || ch == '\r') return false;
        match_.emitted += ch;
      }
      continue;
    }
    if (state_ != State::kQValue) return false;
    const GrammarArg* arg = current_arg();
    if (arg && arg->kind == GrammarArg::Kind::kText) {
      if (!match_byte(ch)) return false;
      if (match_.complete()) enter(State::kQKeyOrClose);
    } else if (arg && arg->kind == GrammarArg::Kind::kJson) {
      if (!term_.empty() || (value_json_.done() && ch == '<')) {
        term_ += ch;
        const std::string end = xml_parameter_end();
        if (end.compare(0, term_.size(), term_) != 0) return false;
        if (term_ == end) enter(State::kQKeyOrClose);
      } else if (!value_json_.feed(static_cast<uint8_t>(ch))) {
        return false;
      }
    } else {
      match_.emitted += ch;
      if (ends_with(match_.emitted, xml_parameter_end())) enter(State::kQKeyOrClose);
    }
  }
  // Free fields admit tokens whose decoder text is empty. They consume no
  // XML bytes and must not disable constraints for the rest of the call.
  return true;
}

void GrammarState::compact_xml_mask(TokenMask* out) const {
  const GrammarArg* arg = state_ == State::kQValue ? current_arg() : nullptr;
  const bool free_value =
      state_ == State::kQValue && (!arg || arg->kind == GrammarArg::Kind::kFree);
  if (free_value) {
    free_mask_in_call(out);
    const std::string end = xml_parameter_end();
    const std::string tail = match_.emitted.substr(
        match_.emitted.size() > end.size() ? match_.emitted.size() - end.size() : 0);
    for (const int32_t id : vocab_->xml_end_ids()) {
      if ((tail + vocab_->text(id)).find(end) == std::string::npos) continue;
      GrammarState trial = *this;
      if (trial.compact_xml_text(vocab_->text(id))) continue;
      uint32_t& word = out->words[static_cast<size_t>(id >> 5)];
      const uint32_t bit = 1u << (id & 31);
      if (word & bit) {
        word &= ~bit;
        --out->allowed;
      }
    }
    return;
  }
  std::vector<int64_t> candidates;
  const bool json_value = state_ == State::kQValue && arg && arg->kind == GrammarArg::Kind::kJson;
  if (state_ == State::kQFreeKey) {
    free_mask_in_call(out);
    for (const int32_t id : vocab_->xml_end_ids()) {
      GrammarState trial = *this;
      if (trial.compact_xml_text(vocab_->text(id))) continue;
      uint32_t& word = out->words[static_cast<size_t>(id >> 5)];
      const uint32_t bit = 1u << (id & 31);
      if (word & bit) {
        word &= ~bit;
        --out->allowed;
      }
    }
    return;
  } else if (json_value) {
    if (term_.empty())
      json_mask(value_json_, -2, out);
    else {
      out->vocab = vocab_->vocab_size();
      out->words.assign(static_cast<size_t>(TokenMask::words_for(out->vocab)), 0u);
      out->allowed = 0;
    }
    const std::string end = xml_parameter_end();
    const unsigned char first = static_cast<unsigned char>(end[term_.size()]);
    for (const int32_t id : vocab_->ids_starting_with(first)) candidates.push_back(id);
    for (const int32_t id : vocab_->xml_end_ids()) candidates.push_back(id);
  } else {
    bool seen[256] = {};
    for (const std::string& target : match_.targets) {
      if (target.size() <= match_.emitted.size() ||
          target.compare(0, match_.emitted.size(), match_.emitted) != 0)
        continue;
      const unsigned char first = static_cast<unsigned char>(target[match_.emitted.size()]);
      if (seen[first]) continue;
      seen[first] = true;
      for (const int32_t id : vocab_->ids_starting_with(first)) candidates.push_back(id);
    }
  }
  std::vector<int64_t> accepted;
  for (const int64_t id : candidates) {
    if (vocab_->is_eos(id) || id == vocab_->markers().think_open.id ||
        id == vocab_->markers().think_close.id ||
        std::find(vocab_->marker_ids().begin(), vocab_->marker_ids().end(), id) !=
            vocab_->marker_ids().end())
      continue;
    GrammarState trial = *this;
    if (trial.compact_xml_text(vocab_->text(id))) accepted.push_back(id);
  }
  if (!json_value) {
    list_mask(out, accepted);
  } else {
    for (const int64_t id : accepted) {
      uint32_t& word = out->words[static_cast<size_t>(id >> 5)];
      const uint32_t bit = 1u << (id & 31);
      if (!(word & bit)) {
        word |= bit;
        ++out->allowed;
      }
    }
  }
}

bool GrammarState::qwen() const {
  return vocab_ != nullptr && vocab_->markers().tool_format() == ToolFormat::kQwenXml;
}


bool GrammarState::dsml() const {
  return vocab_ != nullptr && vocab_->markers().tool_format() == ToolFormat::kDsml;
}

std::vector<int64_t> GrammarState::literal_ids(const std::string& target,
                                               const std::string& emitted) const {
  TextMatch m;
  m.targets.push_back(target);
  m.emitted = emitted;
  return match_ids(m, -1);
}

void GrammarState::free_mask_in_call(TokenMask* out) const {
  free_mask(out, -1, /*forbid_markers=*/true);
  const int vocab = vocab_->vocab_size();
  const auto clear = [&](int64_t id) {
    if (id < 0 || id >= vocab) return;
    uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
    const uint32_t bit = 1u << (id & 31);
    if (w & bit) {
      w &= ~bit;
      --out->allowed;
    }
  };
  for (const int64_t e : vocab_->eos_ids()) clear(e);
  clear(vocab_->markers().think_open.id);
  clear(vocab_->markers().think_close.id);
}

bool GrammarState::obligation_open() const {
  if (spec_.mode == GrammarSpec::Mode::kJson) return !json_.done();
  if (spec_.mode == GrammarSpec::Mode::kJsonOrTools) return calls_ == 0 && !json_.done();
  return (spec_.mode == GrammarSpec::Mode::kRequired ||
          spec_.mode == GrammarSpec::Mode::kNamed) &&
         calls_ == 0;
}

bool GrammarState::calls_remaining() const {
  switch (spec_.mode) {
    case GrammarSpec::Mode::kNone: return true;
    case GrammarSpec::Mode::kForbidCalls: return false;
    case GrammarSpec::Mode::kAuto: return spec_.parallel || calls_ == 0;
    case GrammarSpec::Mode::kJsonOrTools: return spec_.parallel || calls_ == 0;
    case GrammarSpec::Mode::kRequired: return spec_.parallel || calls_ == 0;
    case GrammarSpec::Mode::kNamed: return calls_ == 0;
    case GrammarSpec::Mode::kJson: return false;
  }
  return false;
}

const GrammarTool* GrammarState::current_tool() const {
  return tool_ >= 0 && tool_ < static_cast<int>(spec_.tools.size())
             ? &spec_.tools[static_cast<size_t>(tool_)]
             : nullptr;
}

bool GrammarState::keys_possible() const {
  const GrammarTool* t = current_tool();
  if (t == nullptr) return true;
  if (!t->constrain_keys) return true;
  for (const std::string& k : t->keys)
    if (!key_used(k)) return true;
  return false;
}

bool GrammarState::key_used(const std::string& key) const {
  return std::find(used_keys_.begin(), used_keys_.end(), key) != used_keys_.end();
}

bool GrammarState::call_closable() const {
  const GrammarTool* t = current_tool();
  if (t == nullptr || (!t->strict && !vocab_->markers().xml_compact)) return true;
  for (const std::string& k : t->required_keys)
    if (!key_used(k)) return false;
  return true;
}

bool GrammarState::call_closable_for(const std::string& name) const {
  // At the name, before any key: closable unless a strict tool requires one.
  for (const GrammarTool& t : spec_.tools)
    if (t.name == name)
      return (!t.strict && !vocab_->markers().xml_compact) || t.required_keys.empty();
  return true;
}

void GrammarState::enter(State s) {
  state_ = s;
  match_ = TextMatch{};
  term_.clear();
  if (s == State::kQName) {
    if (spec_.mode == GrammarSpec::Mode::kNamed) {
      match_.targets.push_back(std::string(xml_function_open()) + spec_.named + xml_header_end());
    } else {
      for (const GrammarTool& t : spec_.tools)
        match_.targets.push_back(std::string(xml_function_open()) + t.name + xml_header_end());
    }
    tool_ = -1;
    return;
  }
  if (s == State::kDCalls) {
    match_.targets.push_back(dsml_literals(vocab_).calls_tail);
    tool_ = -1;
    return;
  }
  if (s == State::kDInvoke || s == State::kDInvokeOrClose) {
    if (s == State::kDInvoke || calls_remaining()) {
      const DsmlLiterals& d = dsml_literals(vocab_);
      const auto add = [&](const std::string& name) {
        match_.targets.push_back(d.invoke_open + name + kDInvokeHeadEnd);
        // The reference's call without parameters, whole: its tokenizer
        // has ids that span the header's end and the blank line ("\">\n\n"),
        // which the header target alone would refuse.
        if (!d.empty_invoke_close.empty() && call_closable_for(name))
          match_.targets.push_back(d.invoke_open + name + kDInvokeHeadEnd + d.empty_invoke_close);
      };
      if (spec_.mode == GrammarSpec::Mode::kNamed) {
        add(spec_.named);
      } else {
        for (const GrammarTool& t : spec_.tools) add(t.name);
      }
    }
    if (s == State::kDInvokeOrClose) match_.targets.push_back(dsml_literals(vocab_).calls_close);
    tool_ = -1;
    return;
  }
  if (s == State::kDParamOrClose) {
    const GrammarTool* t = current_tool();
    if (t != nullptr && t->constrain_keys) {
      for (const std::string& k : t->keys)
        if (!key_used(k)) match_.targets.push_back(dsml_literals(vocab_).param_open + k + kDParamFlag);
    } else {
      match_.targets.push_back(dsml_literals(vocab_).param_open);
    }
    if (call_closable()) {
      match_.targets.push_back(dsml_literals(vocab_).invoke_close);
      const std::string& blank = dsml_literals(vocab_).empty_invoke_close;
      if (!blank.empty() && used_keys_.empty()) match_.targets.push_back(blank);
    }
    return;
  }
  if (s == State::kDFlag) {
    // The value's kind decides the flag: a raw text value (an enum) is a
    // string, a JSON-typed value is not, a free value may be either (the
    // parser types it).
    arg_ = -1;
    const GrammarTool* t = current_tool();
    if (t != nullptr)
      for (size_t i = 0; i < t->args.size(); ++i)
        if (t->args[i].key == key_) arg_ = static_cast<int>(i);
    const GrammarArg* a = current_arg();
    if (a == nullptr || a->kind == GrammarArg::Kind::kFree) {
      match_.targets.push_back(kDFlagTrue);
      match_.targets.push_back(kDFlagFalse);
    } else if (a->kind == GrammarArg::Kind::kText) {
      match_.targets.push_back(kDFlagTrue);
    } else {
      match_.targets.push_back(kDFlagFalse);
    }
    return;
  }
  if (s == State::kDValue) {
    const GrammarArg* a = current_arg();
    if (a != nullptr && a->kind == GrammarArg::Kind::kText) {
      for (const std::string& text : a->texts) match_.targets.push_back(text + dsml_literals(vocab_).param_close);
    } else if (a != nullptr && a->kind == GrammarArg::Kind::kJson) {
      value_json_ = JsonMachine(
          arg_schemas_[static_cast<size_t>(tool_)][static_cast<size_t>(arg_)],
          &vocab_->json_tables());
    }
    return;
  }
  if (s == State::kQKeyOrClose) {
    const GrammarTool* t = current_tool();
    if (t != nullptr && t->constrain_keys) {
      for (const std::string& k : t->keys)
        if (!key_used(k)) match_.targets.push_back(std::string(kQParam) + k + xml_header_end());
    } else {
      match_.targets.push_back(kQParam);
    }
    if (call_closable()) match_.targets.push_back(xml_function_end());
    return;
  }
  if (s == State::kQValue) {
    arg_ = -1;
    const GrammarTool* t = current_tool();
    if (t != nullptr)
      for (size_t i = 0; i < t->args.size(); ++i)
        if (t->args[i].key == key_) arg_ = static_cast<int>(i);
    const GrammarArg* a = current_arg();
    if (a != nullptr && a->kind == GrammarArg::Kind::kText) {
      for (const std::string& text : a->texts) match_.targets.push_back(text + xml_parameter_end());
    } else if (a != nullptr && a->kind == GrammarArg::Kind::kJson) {
      value_json_ = JsonMachine(
          arg_schemas_[static_cast<size_t>(tool_)][static_cast<size_t>(arg_)],
          &vocab_->json_tables());
    }
    return;
  }
  if (s == State::kName) {
    if (spec_.mode == GrammarSpec::Mode::kNamed) {
      match_.targets.push_back(spec_.named);
    } else {
      for (const GrammarTool& t : spec_.tools) match_.targets.push_back(t.name);
    }
    tool_ = -1;
  } else if (s == State::kKey) {
    // A closed key set, less the keys this call has already used.
    const GrammarTool* t = current_tool();
    if (t != nullptr && t->constrain_keys)
      for (const std::string& k : t->keys)
        if (!key_used(k)) match_.targets.push_back(k);
  } else if (s == State::kValue) {
    // The value's constraint is the key's declared argument, if any.
    arg_ = -1;
    const GrammarTool* t = current_tool();
    if (t != nullptr)
      for (size_t i = 0; i < t->args.size(); ++i)
        if (t->args[i].key == key_) arg_ = static_cast<int>(i);
    const GrammarArg* a = current_arg();
    if (a != nullptr && a->kind == GrammarArg::Kind::kText) {
      match_.targets = a->texts;
    } else if (a != nullptr && a->kind == GrammarArg::Kind::kJson) {
      value_json_ = JsonMachine(
          arg_schemas_[static_cast<size_t>(tool_)][static_cast<size_t>(arg_)],
          &vocab_->json_tables());
    }
  } else if (s == State::kTop) {
    tool_ = -1;
  }
}

std::vector<int64_t> GrammarState::match_ids(const TextMatch& m,
                                             int64_t closer) const {
  std::vector<int64_t> out;
  for (const std::string& target : m.targets) {
    if (target.size() <= m.emitted.size()) continue;
    if (target.compare(0, m.emitted.size(), m.emitted) != 0) continue;
    // A DSML target's tag: the marker id alone continues there, and no
    // token's text may run across it.
    size_t remaining = target.size() - m.emitted.size();
    const size_t sentinel = target.find(kDsmlSentinel, m.emitted.size());
    if (sentinel == m.emitted.size()) {
      out.push_back(vocab_->markers().dsml.id);
      continue;
    }
    if (sentinel != std::string::npos) remaining = sentinel - m.emitted.size();
    const unsigned char b = static_cast<unsigned char>(target[m.emitted.size()]);
    for (const int32_t id : vocab_->ids_starting_with(b)) {
      const std::string& text = vocab_->text(id);
      if (text.size() <= remaining &&
          target.compare(m.emitted.size(), text.size(), text) == 0)
        out.push_back(id);
    }
  }
  if (m.complete() && closer >= 0) out.push_back(closer);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

void GrammarState::free_mask(TokenMask* out, int64_t extra_allowed,
                             bool forbid_markers) const {
  const int vocab = vocab_->vocab_size();
  out->vocab = vocab;
  out->words.assign(static_cast<size_t>(TokenMask::words_for(vocab)), 0xffffffffu);
  int allowed = vocab;
  // The pad bits past the vocabulary are cleared for tidiness (never read).
  if (vocab % 32 != 0)
    out->words.back() &= (1u << (vocab % 32)) - 1u;
  const auto clear = [&](int64_t id) {
    if (id < 0 || id >= vocab || id == extra_allowed) return;
    uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
    const uint32_t bit = 1u << (id & 31);
    if (w & bit) {
      w &= ~bit;
      --allowed;
    }
  };
  if (forbid_markers)
    for (const int64_t m : vocab_->marker_ids()) clear(m);
  if (obligation_open())
    for (const int64_t e : vocab_->eos_ids()) clear(e);
  out->allowed = allowed;
}

void GrammarState::list_mask(TokenMask* out,
                             const std::vector<int64_t>& ids) const {
  const int vocab = vocab_->vocab_size();
  out->vocab = vocab;
  out->words.assign(static_cast<size_t>(TokenMask::words_for(vocab)), 0u);
  int allowed = 0;
  for (const int64_t id : ids) {
    if (id < 0 || id >= vocab) continue;
    uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
    const uint32_t bit = 1u << (id & 31);
    if (!(w & bit)) {
      w |= bit;
      ++allowed;
    }
  }
  out->allowed = allowed;
  if (allowed == 0)
    throw std::logic_error("GrammarState: a position with no allowed id");
}

void GrammarState::mask(TokenMask* out) const {
  out->allowed = 0;
  if (!active()) return;
  if (compact_xml_text_state()) {
    compact_xml_mask(out);
    return;
  }
  const ChatMarkers& m = vocab_->markers();
  if (think_openable_) {
    // The first position: the state's own mask, plus the model's opener.
    GrammarState settled = *this;
    settled.think_openable_ = false;
    settled.mask(out);
    if (out->allowed == 0) return;  // unconstrained already
    const int64_t id = m.think_open.id;
    if (id >= 0 && id < vocab_->vocab_size()) {
      uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
      const uint32_t bit = 1u << (id & 31);
      if (!(w & bit)) {
        w |= bit;
        ++out->allowed;
      }
    }
    return;
  }
  switch (state_) {
    case State::kThink:
      // Free (the markers included — reasoning is the model's), but the
      // turn may not end while a call is owed.
      if (!obligation_open()) return;  // unconstrained
      free_mask(out, /*extra_allowed=*/-1, /*forbid_markers=*/false);
      return;
    case State::kTop: {
      if (spec_.mode == GrammarSpec::Mode::kJsonOrTools) {
        if (dsml() && top_lt_) {
          list_mask(out, {m.dsml.id});
          return;
        }
        if (calls_ == 0) {
          json_mask(json_, -1, out);
        } else {
          list_mask(out, {vocab_->call_turn_eos()});
        }
        const auto add = [&](int64_t id) {
          if (id < 0 || id >= vocab_->vocab_size()) return;
          uint32_t& word = out->words[static_cast<size_t>(id >> 5)];
          const uint32_t bit = 1u << (id & 31);
          if (!(word & bit)) { word |= bit; ++out->allowed; }
        };
        if (!dsml()) {
          if (calls_remaining()) add(m.tool_call_open.id);
        }
        // DSML opens with ordinary text ending in '<', then its tag
        // token. Qwen permits a newline between consecutive calls.
        if (dsml() || calls_ > 0) {
          for (const unsigned char first : {'<', ' ', '\n', '\r', '\t'})
            for (const int32_t id : vocab_->ids_starting_with(first)) {
              const std::string& text = vocab_->text(id);
              size_t i = 0;
              while (i < text.size() && JsonLexer::is_ws(static_cast<uint8_t>(text[i]))) ++i;
              if ((calls_ > 0 && i == text.size()) ||
                  (dsml() && calls_remaining() && i + 1 == text.size() && text[i] == '<'))
                add(id);
            }
        }
        return;
      }
      // The Qwen format lets natural-language text precede a call ("You
      // may provide optional reasoning ... BEFORE the function call"), so
      // its top is free whenever no call is owed (auto, forbid, or an
      // obligation already met). While a call is owed (required / named
      // with no call closed yet) the top is the opener alone: a free top
      // only forbids EOS, so a greedy model that prefers prose emits text
      // until max_tokens without ever taking the low-probability
      // <tool_call> — the TC-45 probe ("reply OK, do not call tools" with
      // tool_choice required) then sees no tool call and reports the
      // endpoint as not enforcing required. DSML keeps the free top: its
      // opener is ordinary text ("<" + tag), not forceable by a token mask.
      // MiMo's compact dialect keeps it too: its template leaves the
      // opening to the model (it opens <think> itself), pinned by
      // tool_grammar_mimo_modelOpensThinkingAndStraddledTerminator.
      const bool free_top = dsml() || spec_.mode == GrammarSpec::Mode::kAuto ||
                            spec_.mode == GrammarSpec::Mode::kForbidCalls ||
                            (qwen() && (!obligation_open() || m.xml_compact));
      if (free_top) {
        // DSML: the text may carry on; the tag opens a block right after a
        // "<" (content runs before "\n\n<" in the format).
        const int64_t opener = dsml() ? (top_lt_ ? m.dsml.id : -1) : m.tool_call_open.id;
        free_mask(out, calls_remaining() ? opener : -1);
        return;
      }
      std::vector<int64_t> ids;
      if (calls_remaining()) ids.push_back(m.tool_call_open.id);
      if (!obligation_open()) ids.push_back(vocab_->call_turn_eos());
      list_mask(out, ids);
      return;
    }
    case State::kName: {
      std::vector<int64_t> ids = match_ids(match_, -1);
      if (match_.complete()) {
        if (keys_possible_for(match_.emitted)) ids.push_back(m.arg_key_open.id);
        if (call_closable_for(match_.emitted)) ids.push_back(m.tool_call_close.id);
      }
      list_mask(out, ids);
      return;
    }
    case State::kKey: {
      const GrammarTool* t = current_tool();
      if (t != nullptr && t->constrain_keys) {
        list_mask(out, match_ids(match_, m.arg_key_close.id));
        return;
      }
      free_mask(out, m.arg_key_close.id);
      return;
    }
    case State::kAfterKey:
      list_mask(out, {m.arg_value_open.id});
      return;
    case State::kValue: {
      const GrammarArg* a = current_arg();
      if (a == nullptr || a->kind == GrammarArg::Kind::kFree) {
        free_mask(out, m.arg_value_close.id);
      } else if (a->kind == GrammarArg::Kind::kText) {
        list_mask(out, match_ids(match_, m.arg_value_close.id));
      } else {
        json_mask(value_json_, m.arg_value_close.id, out);
      }
      return;
    }
    case State::kAfterValue: {
      // A closed set with every key used has every required key used, so
      // the two conditions are never both false: no dead end.
      std::vector<int64_t> ids;
      if (keys_possible()) ids.push_back(m.arg_key_open.id);
      if (call_closable()) ids.push_back(m.tool_call_close.id);
      list_mask(out, ids);
      return;
    }
    case State::kEnd:
    case State::kDone:
      list_mask(out, {vocab_->call_turn_eos()});
      return;
    case State::kJsonBody:
      json_mask(json_, /*closer=*/-1, out);
      return;
    case State::kQName:
    case State::kQKeyOrClose:
      list_mask(out, match_ids(match_, -1));
      return;
    case State::kQFreeKey:
      free_mask_in_call(out);
      return;
    case State::kQValue: {
      const GrammarArg* a = current_arg();
      if (a == nullptr || a->kind == GrammarArg::Kind::kFree) {
        free_mask_in_call(out);
      } else if (a->kind == GrammarArg::Kind::kText) {
        list_mask(out, match_ids(match_, -1));
      } else if (!term_.empty()) {
        list_mask(out, literal_ids(xml_parameter_end(), term_));
      } else {
        // The JSON text; once complete, also the terminator's first ids.
        json_mask(value_json_, /*closer=*/-2, out);
        if (value_json_.done()) {
          const int vocab = vocab_->vocab_size();
          for (const int64_t id : literal_ids(xml_parameter_end(), "")) {
            if (id < 0 || id >= vocab) continue;
            uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
            const uint32_t bit = 1u << (id & 31);
            if (!(w & bit)) {
              w |= bit;
              ++out->allowed;
            }
          }
        }
      }
      return;
    }
    case State::kQClose:
      list_mask(out, {m.tool_call_close.id});
      return;
    case State::kDCalls:
    case State::kDInvoke:
    case State::kDParamOrClose:
    case State::kDFlag:
    case State::kDInvokeOrClose:
      list_mask(out, match_ids(match_, -1));
      return;
    case State::kDFreeKey:
      free_mask_in_call(out);
      return;
    case State::kDValue: {
      const GrammarArg* a = current_arg();
      if (!term_.empty()) {
        list_mask(out, literal_ids(dsml_literals(vocab_).param_close, term_));
      } else if (a == nullptr || a->kind == GrammarArg::Kind::kFree) {
        // Free text; the tag is offered once the value ends in "</" (the
        // closer's start) — it then commits the closer.
        free_mask(out, ends_with(match_.emitted, "</") ? m.dsml.id : -1, /*forbid_markers=*/true);
        const int vocab = vocab_->vocab_size();
        const auto clear = [&](int64_t id) {
          if (id < 0 || id >= vocab || id == m.dsml.id) return;
          uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
          const uint32_t bit = 1u << (id & 31);
          if (w & bit) {
            w &= ~bit;
            --out->allowed;
          }
        };
        for (const int64_t e : vocab_->eos_ids()) clear(e);
        clear(m.think_open.id);
        clear(m.think_close.id);
        if (!ends_with(match_.emitted, "</")) clear(m.dsml.id);
      } else if (a->kind == GrammarArg::Kind::kText) {
        list_mask(out, match_ids(match_, -1));
      } else {
        // The JSON text; once complete, also the closer's first ids.
        json_mask(value_json_, /*closer=*/-2, out);
        if (value_json_.done()) {
          const int vocab = vocab_->vocab_size();
          for (const int64_t id : literal_ids(dsml_literals(vocab_).param_close, "")) {
            if (id < 0 || id >= vocab) continue;
            uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
            const uint32_t bit = 1u << (id & 31);
            if (!(w & bit)) {
              w |= bit;
              ++out->allowed;
            }
          }
        }
      }
      return;
    }
  }
}

// A JSON machine's position: its mask over token texts, never a marker (a
// "<think>" would be legal string content), and the closer — EOS for the
// body, </arg_value> for a typed argument — once the text is complete.
void GrammarState::json_mask(const JsonMachine& machine, int64_t closer,
                             TokenMask* out) const {
  machine.mask(*vocab_, out);
  const int vocab = vocab_->vocab_size();
  const auto set = [&](int64_t id, bool on) {
    if (id < 0 || id >= vocab) return;
    uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
    const uint32_t bit = 1u << (id & 31);
    if (on && !(w & bit)) {
      w |= bit;
      ++out->allowed;
    } else if (!on && (w & bit)) {
      w &= ~bit;
      --out->allowed;
    }
  };
  for (const int64_t m : vocab_->marker_ids()) set(m, false);
  set(vocab_->markers().think_open.id, false);
  set(vocab_->markers().think_close.id, false);
  if (machine.done()) {
    if (closer == -1)
      for (const int64_t e : vocab_->eos_ids()) set(e, true);
    else if (closer >= 0)
      set(closer, true);
    // closer -2: the caller adds its own text terminator.
  }
  if (out->allowed == 0 && closer != -2)
    throw std::logic_error("GrammarState: a JSON position with no allowed id");
}

bool GrammarState::json_allows(const JsonMachine& machine, int64_t closer,
                               int64_t id) const {
  if (closer < 0 ? vocab_->is_eos(id) : id == closer) return machine.done();
  if (vocab_->is_eos(id)) return false;
  for (const int64_t m : vocab_->marker_ids())
    if (m == id) return false;
  if (id == vocab_->markers().think_open.id || id == vocab_->markers().think_close.id)
    return false;
  return machine.allows(*vocab_, id);
}

bool GrammarState::keys_possible_for(const std::string& name) const {
  for (const GrammarTool& t : spec_.tools)
    if (t.name == name) return !t.constrain_keys || !t.keys.empty();
  return true;
}

bool GrammarState::allows(int64_t id) const {
  if (!active()) return true;
  if (compact_xml_text_state()) {
    TokenMask m;
    mask(&m);
    return m.allows(id);
  }
  if (think_openable_) {
    if (id == vocab_->markers().think_open.id) return true;
    GrammarState settled = *this;
    settled.think_openable_ = false;
    return settled.allows(id);
  }
  if (state_ == State::kJsonBody) return json_allows(json_, -1, id);
  if (state_ == State::kValue) {
    const GrammarArg* a = current_arg();
    if (a != nullptr && a->kind == GrammarArg::Kind::kJson)
      return json_allows(value_json_, vocab_->markers().arg_value_close.id, id);
  }
  if (state_ == State::kQValue || state_ == State::kDValue) {
    const GrammarArg* a = current_arg();
    if (a != nullptr && a->kind == GrammarArg::Kind::kJson) {
      const std::string& closer =
          state_ == State::kQValue ? std::string(xml_parameter_end()) : dsml_literals(vocab_).param_close;
      if (!term_.empty()) {
        for (const int64_t t : literal_ids(closer, term_))
          if (t == id) return true;
        return false;
      }
      if (value_json_.done())
        for (const int64_t t : literal_ids(closer, ""))
          if (t == id) return true;
      if (vocab_->is_eos(id)) return false;
      for (const int64_t mk : vocab_->marker_ids())
        if (mk == id) return false;
      if (id == vocab_->markers().think_open.id || id == vocab_->markers().think_close.id)
        return false;
      return value_json_.allows(*vocab_, id);
    }
  }
  TokenMask m;
  mask(&m);
  return m.allows(id);
}

void GrammarState::advance(int64_t id) {
  if (!active()) return;
  if (!allows(id)) {
    dead_ = true;
    return;
  }
  if (compact_xml_text_state()) {
    if (!compact_xml_text(vocab_->text(id))) dead_ = true;
    return;
  }
  const ChatMarkers& m = vocab_->markers();
  if (think_openable_) {
    // The first id settles the opening: the model's <think> enters the
    // free reasoning (its close returns to the mode's opening state).
    think_openable_ = false;
    if (id == m.think_open.id) {
      enter(State::kThink);
      return;
    }
  }
  const auto after_call = [&] {
    ++calls_;
    if (calls_remaining())
      enter(State::kTop);
    else
      enter(State::kEnd);
  };
  switch (state_) {
    case State::kThink:
      if (id == m.think_close.id)
        enter(spec_.mode == GrammarSpec::Mode::kJson ? State::kJsonBody
                                                     : State::kTop);
      return;
    case State::kJsonBody:
      if (vocab_->is_eos(id)) {
        state_ = State::kDone;
        return;
      }
      for (const char c : vocab_->text(id))
        if (!json_.feed(static_cast<uint8_t>(c))) {
          dead_ = true;
          return;
        }
      return;
    case State::kTop:
      if (spec_.mode == GrammarSpec::Mode::kJsonOrTools && calls_ == 0 &&
          !top_lt_ && id != m.tool_call_open.id) {
        const std::string& text = vocab_->text(id);
        const size_t first = text.find_first_not_of(" \n\r\t");
        if (dsml() && first != std::string::npos && first + 1 == text.size() && text[first] == '<') {
          top_lt_ = true;
          return;
        }
        bool value = false;
        for (const char ch : text) {
          value = value || !JsonLexer::is_ws(static_cast<uint8_t>(ch));
          if (!json_.feed(static_cast<uint8_t>(ch))) { dead_ = true; return; }
        }
        if (value) enter(State::kJsonBody);
        return;
      }
      if (dsml()) {
        if (id == m.dsml.id) {
          top_lt_ = false;
          enter(State::kDCalls);
        } else if (vocab_->is_eos(id)) {
          state_ = State::kDone;
        } else {
          const std::string& text = vocab_->text(id);
          top_lt_ = !text.empty() && text.back() == '<';
        }
        return;
      }
      if (id == m.tool_call_open.id) {
        enter(qwen() ? State::kQName : State::kName);
      } else if (vocab_->is_eos(id)) {
        state_ = State::kDone;
      }
      return;
    case State::kDCalls:
    case State::kDInvoke:
    case State::kDInvokeOrClose: {
      match_.emitted += id == m.dsml.id ? kDTag : vocab_->text(id);
      if (!match_.complete()) return;
      if (state_ == State::kDCalls) {
        enter(State::kDInvoke);
        return;
      }
      if (match_.emitted == dsml_literals(vocab_).calls_close) {
        enter(State::kEnd);
        return;
      }
      // The whole call without parameters (header, blank line, close).
      if (const std::string& blank = dsml_literals(vocab_).empty_invoke_close;
          !blank.empty() && ends_with(match_.emitted, blank)) {
        ++calls_;
        enter(State::kDInvokeOrClose);
        return;
      }
      // "<" TAG " invoke name=\"" NAME "\">\n": bind the tool; the key ledger opens.
      const size_t open = dsml_literals(vocab_).invoke_open.size();
      const std::string name = match_.emitted.substr(open, match_.emitted.size() - open - kDInvokeHeadEnd.size());
      tool_ = -1;
      used_keys_.clear();
      for (size_t i = 0; i < spec_.tools.size(); ++i)
        if (spec_.tools[i].name == name) tool_ = static_cast<int>(i);
      enter(State::kDParamOrClose);
      return;
    }
    case State::kDParamOrClose:
      match_.emitted += id == m.dsml.id ? kDTag : vocab_->text(id);
      if (!match_.complete()) return;
      if (match_.emitted == dsml_literals(vocab_).invoke_close ||
          (!dsml_literals(vocab_).empty_invoke_close.empty() &&
           match_.emitted == dsml_literals(vocab_).empty_invoke_close)) {
        ++calls_;
        enter(State::kDInvokeOrClose);
      } else if (match_.emitted == dsml_literals(vocab_).param_open) {
        enter(State::kDFreeKey);
      } else {
        const size_t open = dsml_literals(vocab_).param_open.size();
        key_ = match_.emitted.substr(open, match_.emitted.size() - open - kDParamFlag.size());
        used_keys_.push_back(key_);
        enter(State::kDFlag);
      }
      return;
    case State::kDFreeKey:
      match_.emitted += vocab_->text(id);
      if (ends_with(match_.emitted, kDParamFlag)) {
        key_ = match_.emitted.substr(0, match_.emitted.size() - kDParamFlag.size());
        if (key_.empty() || key_.find_first_of("\n<>\"") != std::string::npos) {
          dead_ = true;
          return;
        }
        used_keys_.push_back(key_);
        enter(State::kDFlag);
      }
      return;
    case State::kDFlag:
      match_.emitted += vocab_->text(id);
      if (!match_.complete()) return;
      flag_string_ = match_.emitted == kDFlagTrue;
      enter(State::kDValue);
      return;
    case State::kDValue: {
      const GrammarArg* a = current_arg();
      if (a != nullptr && a->kind == GrammarArg::Kind::kText) {
        match_.emitted += id == m.dsml.id ? kDTag : vocab_->text(id);
        if (match_.complete()) enter(State::kDParamOrClose);
        return;
      }
      if (!term_.empty()) {
        term_ += id == m.dsml.id ? kDTag : vocab_->text(id);
        if (term_ == dsml_literals(vocab_).param_close) enter(State::kDParamOrClose);
        return;
      }
      if (a != nullptr && a->kind == GrammarArg::Kind::kJson) {
        bool terminator = false;
        if (value_json_.done())
          for (const int64_t t : literal_ids(dsml_literals(vocab_).param_close, ""))
            if (t == id) terminator = true;
        if (terminator) {
          term_ += vocab_->text(id);
          if (term_ == dsml_literals(vocab_).param_close) enter(State::kDParamOrClose);
          return;
        }
        for (const char c : vocab_->text(id))
          if (!value_json_.feed(static_cast<uint8_t>(c))) {
            dead_ = true;
            return;
          }
        return;
      }
      // A free value: its text, until the tag after a "</" starts the closer.
      if (id == m.dsml.id) {
        match_.emitted.resize(match_.emitted.size() - 2);  // the "</" belongs to the closer
        term_ = "</" + kDTag;
        return;
      }
      match_.emitted += vocab_->text(id);
      return;
    }
    case State::kQName:
      match_.emitted += vocab_->text(id);
      if (match_.complete()) {
        // "\n<function=" NAME ">\n": bind the tool; the key ledger opens.
        const std::string name =
            match_.emitted.substr(std::strlen(xml_function_open()),
                                  match_.emitted.size() - std::strlen(xml_function_open()) -
                                      std::strlen(xml_header_end()));
        tool_ = -1;
        used_keys_.clear();
        for (size_t i = 0; i < spec_.tools.size(); ++i)
          if (spec_.tools[i].name == name) tool_ = static_cast<int>(i);
        enter(State::kQKeyOrClose);
      }
      return;
    case State::kQKeyOrClose:
      match_.emitted += vocab_->text(id);
      if (!match_.complete()) return;
      if (match_.emitted == xml_function_end()) {
        enter(State::kQClose);
      } else if (match_.emitted == kQParam) {
        enter(State::kQFreeKey);
      } else {
        key_ = match_.emitted.substr(
            std::strlen(kQParam),
            match_.emitted.size() - std::strlen(kQParam) - std::strlen(xml_header_end()));
        used_keys_.push_back(key_);
        enter(State::kQValue);
      }
      return;
    case State::kQFreeKey:
      match_.emitted += vocab_->text(id);
      if (ends_with(match_.emitted, xml_header_end())) {
        key_ = match_.emitted.substr(0, match_.emitted.size() - std::strlen(xml_header_end()));
        if (key_.empty() || key_.find_first_of("\n<>") != std::string::npos) {
          dead_ = true;
          return;
        }
        used_keys_.push_back(key_);
        enter(State::kQValue);
      }
      return;
    case State::kQValue: {
      const GrammarArg* a = current_arg();
      const std::string& text = vocab_->text(id);
      if (a != nullptr && a->kind == GrammarArg::Kind::kText) {
        match_.emitted += text;
        if (match_.complete()) enter(State::kQKeyOrClose);
        return;
      }
      if (a != nullptr && a->kind == GrammarArg::Kind::kJson) {
        bool terminator = !term_.empty();
        if (!terminator && value_json_.done())
          for (const int64_t t : literal_ids(xml_parameter_end(), ""))
            if (t == id) terminator = true;
        if (terminator) {
          term_ += text;
          if (term_ == xml_parameter_end()) enter(State::kQKeyOrClose);
          return;
        }
        for (const char c : text)
          if (!value_json_.feed(static_cast<uint8_t>(c))) {
            dead_ = true;
            return;
          }
        // A token that closes the value and runs on into the terminator's
        // newline ('"\n' on the Qwen2 tokenizers): the machine took the
        // newline as trailing whitespace; the terminator match starts there.
        if (value_json_.done() && !text.empty() && text.back() == '\n') term_ = "\n";
        return;
      }
      match_.emitted += text;
      if (ends_with(match_.emitted, xml_parameter_end())) enter(State::kQKeyOrClose);
      return;
    }
    case State::kQClose:
      after_call();
      return;
    case State::kName:
      if (id == m.arg_key_open.id || id == m.tool_call_close.id) {
        // The name is complete: bind the tool; the call's key ledger opens.
        tool_ = -1;
        used_keys_.clear();
        for (size_t i = 0; i < spec_.tools.size(); ++i)
          if (spec_.tools[i].name == match_.emitted) tool_ = static_cast<int>(i);
        if (id == m.arg_key_open.id)
          enter(State::kKey);
        else
          after_call();
        return;
      }
      match_.emitted += vocab_->text(id);
      return;
    case State::kKey:
      if (id == m.arg_key_close.id) {
        key_ = match_.emitted;
        used_keys_.push_back(key_);
        enter(State::kAfterKey);
        return;
      }
      match_.emitted += vocab_->text(id);
      return;
    case State::kAfterKey:
      enter(State::kValue);
      return;
    case State::kValue: {
      if (id == m.arg_value_close.id) {
        enter(State::kAfterValue);
        return;
      }
      const GrammarArg* a = current_arg();
      if (a != nullptr && a->kind == GrammarArg::Kind::kText) {
        match_.emitted += vocab_->text(id);
      } else if (a != nullptr && a->kind == GrammarArg::Kind::kJson) {
        for (const char c : vocab_->text(id))
          if (!value_json_.feed(static_cast<uint8_t>(c))) {
            dead_ = true;
            return;
          }
      }
      return;
    }
    case State::kAfterValue:
      if (id == m.arg_key_open.id)
        enter(State::kKey);
      else
        after_call();
      return;
    case State::kEnd:
      state_ = State::kDone;
      return;
    case State::kDone:
      return;
  }
}

}  // namespace dgpp::text
