#include "text/inkling_tool_grammar.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "loaders/minijson.hpp"
#include "text/chat_template.hpp"
#include "text/tool_grammar.hpp"

namespace dgpp::text {
namespace {

std::string quoted(const std::string& text) { return Value::string_value(text).to_json(false); }

// The request's tools reach the grammar as GrammarTool records whose
// arguments each carry a self-contained schema: the tool's whole parameters
// object under "$defs" and a "$ref" to the one property (tool_grammar.cpp
// argument_schema). The call schema hoists that "$defs" to its own root and
// points every property at it, so a property is typed exactly as it would
// be alone. The key set follows the tool record: the declared names unless
// the definition opened it.
struct ToolDefs {
  std::string defs;  // the "$defs" object's JSON text ("" when no argument carries one)
};

// A property's schema inside the call schema: its "$ref" into the hoisted
// definitions when the property compiles on its own, an enum for a text
// argument without a schema, any JSON value otherwise (a definition that
// asked for strict typing was refused at the request boundary).
std::string property_schema(const GrammarArg& arg, ToolDefs* defs) {
  if (!arg.schema.empty()) {
    try {
      const auto parsed = minijson::parse(arg.schema);
      std::vector<std::string> unenforced;
      (void)compile_json_schema(parsed.root, &unenforced);
      const minijson::Value* ref = parsed.root.find("$ref");
      const minijson::Value* d = parsed.root.find("$defs");
      if (ref != nullptr && ref->is_string() && d != nullptr && d->is_object()) {
        if (defs->defs.empty()) defs->defs = Value::from_minijson(*d).to_json(false);
        return R"({"$ref":)" + quoted(std::string(ref->as_string())) + "}";
      }
      // A plain schema without references stands where it is.
      if (ref == nullptr && d == nullptr) return arg.schema;
    } catch (const std::exception&) {
    }
    return "{}";
  }
  if (arg.kind == GrammarArg::Kind::kText) {
    std::string schema = R"({"type":"string","enum":[)";
    for (size_t i = 0; i < arg.texts.size(); ++i) {
      if (i) schema += ',';
      schema += quoted(arg.texts[i]);
    }
    return schema + "]}";
  }
  return "{}";
}

// The "args" object of one tool.
std::string args_schema(const GrammarTool& tool, ToolDefs* defs) {
  if (tool.args.empty() && !tool.constrain_keys) return R"({"type":"object"})";
  std::string out = R"({"type":"object","properties":{)";
  for (size_t i = 0; i < tool.args.size(); ++i) {
    if (i) out += ',';
    out += quoted(tool.args[i].key) + ":" + property_schema(tool.args[i], defs);
  }
  out += R"(},"required":[)";
  for (size_t i = 0; i < tool.required_keys.size(); ++i) {
    if (i) out += ',';
    out += quoted(tool.required_keys[i]);
  }
  out += R"(],"additionalProperties":)";
  out += tool.constrain_keys ? "false}" : "true}";
  return out;
}

// A call's body: {"name": NAME, "args": {...}}.
std::string call_schema(const GrammarTool& tool) {
  ToolDefs defs;
  const std::string args = args_schema(tool, &defs);
  std::string out = "{";
  if (!defs.defs.empty()) out += R"("$defs":)" + defs.defs + ",";
  out += R"("type":"object","properties":{"name":{"const":)" + quoted(tool.name) + R"(},"args":)" + args +
         R"(},"required":["name","args"],"additionalProperties":false})";
  return out;
}

std::shared_ptr<const JsonSchema> compile(const std::string& text, const char* what) {
  try {
    std::vector<std::string> unenforced;
    return std::make_shared<const JsonSchema>(compile_json_schema(minijson::parse(text).root, &unenforced));
  } catch (const std::exception& e) {
    throw std::invalid_argument(std::string(what) + ": " + e.what());
  }
}

void set_bit(TokenMask* out, int64_t id) {
  if (id < 0 || id >= out->vocab) return;
  uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
  const uint32_t bit = 1u << (id & 31);
  if (!(w & bit)) {
    w |= bit;
    ++out->allowed;
  }
}

void clear_bit(TokenMask* out, int64_t id) {
  if (id < 0 || id >= out->vocab) return;
  uint32_t& w = out->words[static_cast<size_t>(id >> 5)];
  const uint32_t bit = 1u << (id & 31);
  if (w & bit) {
    w &= ~bit;
    --out->allowed;
  }
}

void list_mask(TokenMask* out, int vocab, const std::vector<int64_t>& ids) {
  out->vocab = vocab;
  out->words.assign(static_cast<size_t>(TokenMask::words_for(vocab)), 0u);
  out->allowed = 0;
  for (const int64_t id : ids) set_bit(out, id);
}

bool contains(const std::vector<int64_t>& ids, int64_t id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

}  // namespace

struct InklingToolGrammar::Data {
  const GrammarVocab* vocab = nullptr;
  GrammarSpec spec;
  std::vector<int> callable;  // the tools this mode may call (indices into spec.tools)
  std::vector<std::shared_ptr<const JsonSchema>> call_schemas;  // per tool
  // A call block the model opened without naming its tool, at a position
  // the auto mode leaves unmasked: any JSON object.
  std::shared_ptr<const JsonSchema> unnamed_call_schema;
  std::shared_ptr<const JsonSchema> answer_schema;              // the response format's
  // Every id that decodes to text: a block's free text.
  std::vector<uint32_t> text_words;
  int text_count = 0;
};

InklingToolGrammar::InklingToolGrammar(const GrammarVocab* vocab, const GrammarSpec& spec) {
  auto data = std::make_shared<Data>();
  data->vocab = vocab;
  data->spec = spec;
  using Mode = GrammarSpec::Mode;
  const Mode mode = spec.mode;
  if ((mode == Mode::kRequired || mode == Mode::kNamed) && spec.tools.empty())
    throw std::invalid_argument("InklingToolGrammar: a required call needs tools");
  for (size_t t = 0; t < spec.tools.size(); ++t) {
    if (spec.tools[t].name.empty()) throw std::invalid_argument("tools.name: empty");
    data->call_schemas.push_back(compile(call_schema(spec.tools[t]), "tools"));
    const bool named = mode == Mode::kNamed && spec.tools[t].name == spec.named;
    if (named || mode == Mode::kAuto || mode == Mode::kRequired || mode == Mode::kJsonOrTools)
      data->callable.push_back(static_cast<int>(t));
  }
  if (mode == Mode::kNamed && data->callable.empty())
    throw std::invalid_argument("InklingToolGrammar: the named function is not among the tools");
  data->unnamed_call_schema = std::make_shared<const JsonSchema>(json_object_schema());
  if (spec.has_json()) {
    data->answer_schema = spec.json_schema.empty() ? std::make_shared<const JsonSchema>(json_object_schema())
                                                   : compile(spec.json_schema, "response_format.json_schema");
  }
  if (vocab != nullptr) {
    const int n = vocab->vocab_size();
    data->text_words.assign(static_cast<size_t>(TokenMask::words_for(n)), 0u);
    for (int id = 0; id < n; ++id)
      if (!vocab->text(id).empty()) {
        data->text_words[static_cast<size_t>(id >> 5)] |= 1u << (id & 31);
        ++data->text_count;
      }
  }
  data_ = std::move(data);
}

bool InklingToolGrammar::calls_allowed() const {
  using Mode = GrammarSpec::Mode;
  if (data_->callable.empty() || answered_) return false;
  switch (data_->spec.mode) {
    case Mode::kAuto:
    case Mode::kRequired:
    case Mode::kJsonOrTools:
      return data_->spec.parallel || calls_ == 0;
    case Mode::kNamed:
      return calls_ == 0;
    default:
      return false;
  }
}

bool InklingToolGrammar::text_allowed() const {
  using Mode = GrammarSpec::Mode;
  switch (data_->spec.mode) {
    case Mode::kJson:
      return !answered_;
    case Mode::kJsonOrTools:
      return !answered_ && calls_ == 0;  // the answer, unless the turn became a turn of calls
    case Mode::kRequired:
    case Mode::kNamed:
      // A turn that must call is its calls: an answer in text would let the
      // model put the call off block after block.
      return false;
    default:
      return true;
  }
}

// While the mode still owes a call or an answer, one thinking block may come
// before it. Without the bound a model that would rather end the turn could
// open reasoning blocks until the token limit.
bool InklingToolGrammar::thinking_allowed() const {
  using Mode = GrammarSpec::Mode;
  const Mode mode = data_->spec.mode;
  if (mode == Mode::kAuto || mode == Mode::kForbidCalls) return true;
  if (end_allowed()) return false;  // the obligation is met: more calls or the end
  return thinking_blocks_ == 0;
}

bool InklingToolGrammar::end_allowed() const {
  using Mode = GrammarSpec::Mode;
  switch (data_->spec.mode) {
    case Mode::kRequired:
    case Mode::kNamed:
      return calls_ > 0;
    case Mode::kJson:
      return answered_;
    case Mode::kJsonOrTools:
      return answered_ || calls_ > 0;
    default:
      return true;
  }
}

// A call block is not among them: it follows a header that names its tool,
// as the checkpoint's template writes it, so the body's schema is one
// tool's from its first byte.
void InklingToolGrammar::block_openers(std::vector<int64_t>* ids) const {
  const ChatMarkers& m = data_->vocab->markers();
  if (thinking_allowed()) ids->push_back(m.content_thinking.id);
  if (text_allowed()) ids->push_back(m.content_text.id);
}

int InklingToolGrammar::header_tool() const {
  for (const int t : data_->callable)
    if (data_->spec.tools[static_cast<size_t>(t)].name == header_) return t;
  return -1;
}

void InklingToolGrammar::header_text_ids(std::vector<int64_t>* ids) const {
  if (!calls_allowed()) return;
  const GrammarVocab& v = *data_->vocab;
  for (const int t : data_->callable) {
    const std::string& name = data_->spec.tools[static_cast<size_t>(t)].name;
    if (name.size() <= header_.size() || name.compare(0, header_.size(), header_) != 0) continue;
    const std::string_view rest(name.data() + header_.size(), name.size() - header_.size());
    for (const int32_t id : v.ids_starting_with(static_cast<unsigned char>(rest[0]))) {
      const std::string& text = v.text(id);
      if (text.size() <= rest.size() && rest.compare(0, text.size(), text) == 0 && !contains(*ids, id))
        ids->push_back(id);
    }
  }
}

void InklingToolGrammar::mask(TokenMask* out) const {
  out->allowed = 0;
  const GrammarVocab& v = *data_->vocab;
  const ChatMarkers& m = v.markers();
  const int vocab = v.vocab_size();
  std::vector<int64_t> ids;
  switch (stage_) {
    case Stage::kDone:
      return;
    case Stage::kEnd:
      list_mask(out, vocab, v.eos_ids());
      return;
    case Stage::kTop:
      block_openers(&ids);
      // A header leads to one of those blocks or to a call.
      if (!ids.empty() || calls_allowed()) ids.push_back(m.message_model.id);
      if (end_allowed())
        for (const int64_t e : v.eos_ids()) ids.push_back(e);
      list_mask(out, vocab, ids);
      if (out->allowed == 0) throw std::logic_error("InklingToolGrammar: a top position with no allowed id");
      return;
    case Stage::kHeader:
      header_text_ids(&ids);
      if (header_.empty()) block_openers(&ids);
      else if (header_tool() >= 0) ids.push_back(m.content_tool_json.id);
      list_mask(out, vocab, ids);
      if (out->allowed == 0) throw std::logic_error("InklingToolGrammar: a header position with no allowed id");
      return;
    case Stage::kThinking:
    case Stage::kText:
      // With calls open in the auto mode nothing is forbidden inside free
      // text: the position stays unmasked and advance() follows whatever
      // marker the model writes.
      if (data_->spec.mode == GrammarSpec::Mode::kAuto && calls_allowed()) return;
      out->vocab = vocab;
      out->words = data_->text_words;
      out->allowed = data_->text_count;
      set_bit(out, m.end_message.id);
      return;
    case Stage::kCall:
    case Stage::kAnswer:
      if (json_.done() && stage_ == Stage::kCall) {
        list_mask(out, vocab, {m.end_message.id});
        return;
      }
      json_.mask(v, out);
      if (json_.done()) set_bit(out, m.end_message.id);
      // The body starts at its value: JSON admits whitespace before it, and
      // a model steered away from its first choice would spend ids there.
      if (!json_started_)
        for (const unsigned char ws : {' ', '\t', '\n', '\r'})
          for (const int32_t id : v.ids_starting_with(ws)) clear_bit(out, id);
      if (out->allowed == 0) throw std::logic_error("InklingToolGrammar: a JSON position with no allowed id");
      return;
  }
}

bool InklingToolGrammar::allows(int64_t id) const {
  const GrammarVocab& v = *data_->vocab;
  const ChatMarkers& m = v.markers();
  std::vector<int64_t> ids;
  switch (stage_) {
    case Stage::kDone:
      return true;
    case Stage::kEnd:
      return v.is_eos(id);
    case Stage::kTop:
      if (v.is_eos(id)) return end_allowed();
      block_openers(&ids);
      if (id == m.message_model.id) return !ids.empty() || calls_allowed();
      return contains(ids, id);
    case Stage::kHeader:
      header_text_ids(&ids);
      if (header_.empty()) block_openers(&ids);
      else if (header_tool() >= 0) ids.push_back(m.content_tool_json.id);
      return contains(ids, id);
    case Stage::kThinking:
    case Stage::kText:
      if (data_->spec.mode == GrammarSpec::Mode::kAuto && calls_allowed()) return true;
      return id == m.end_message.id || !v.text(id).empty();
    case Stage::kCall:
      if (json_.done()) return id == m.end_message.id;
      return !leading_whitespace(id) && json_.allows(v, id);
    case Stage::kAnswer:
      if (json_.done() && id == m.end_message.id) return true;
      return !leading_whitespace(id) && json_.allows(v, id);
  }
  return false;
}

bool InklingToolGrammar::leading_whitespace(int64_t id) const {
  if (json_started_) return false;
  const std::string& text = data_->vocab->text(id);
  return !text.empty() && (text[0] == ' ' || text[0] == '\t' || text[0] == '\n' || text[0] == '\r');
}

bool InklingToolGrammar::feed_json(int64_t id) {
  json_started_ = true;
  for (const unsigned char byte : data_->vocab->text(id))
    if (!json_.feed(byte)) return false;
  return true;
}

void InklingToolGrammar::advance(int64_t id) {
  const GrammarVocab& v = *data_->vocab;
  const ChatMarkers& m = v.markers();
  // A block marker, wherever it arrives: the block it opens. Inside free
  // text this is the unmasked auto mode, where the model moved on without
  // closing the block before.
  const auto open_block = [&](int tool) -> bool {
    if (id == m.content_thinking.id) {
      ++thinking_blocks_;
      stage_ = Stage::kThinking;
      return true;
    }
    if (id == m.content_text.id) {
      if (data_->spec.has_json()) {
        json_ = JsonMachine(data_->answer_schema, &v.json_tables());
        json_started_ = false;
        stage_ = Stage::kAnswer;
      } else {
        stage_ = Stage::kText;
      }
      return true;
    }
    if (id == m.content_tool_json.id) {
      header_tool_ = tool;
      json_ = JsonMachine(tool >= 0 ? data_->call_schemas[static_cast<size_t>(tool)] : data_->unnamed_call_schema,
                          &v.json_tables());
      json_started_ = false;
      stage_ = Stage::kCall;
      return true;
    }
    return false;
  };
  switch (stage_) {
    case Stage::kDone:
      return;
    case Stage::kEnd:
      if (v.is_eos(id)) stage_ = Stage::kDone;
      return;
    case Stage::kTop:
      if (v.is_eos(id)) {
        stage_ = Stage::kDone;
      } else if (id == m.message_model.id) {
        header_.clear();
        stage_ = Stage::kHeader;
      } else {
        (void)open_block(-1);
      }
      return;
    case Stage::kHeader:
      if (open_block(header_.empty() ? -1 : header_tool())) return;
      header_ += v.text(id);
      return;
    case Stage::kThinking:
    case Stage::kText:
      if (id == m.end_message.id) {
        stage_ = Stage::kTop;
      } else if (v.is_eos(id)) {
        stage_ = Stage::kDone;
      } else if (id == m.message_model.id) {
        header_.clear();
        stage_ = Stage::kHeader;
      } else {
        (void)open_block(-1);
      }
      return;
    case Stage::kCall:
      if (id == m.end_message.id) {
        ++calls_;
        stage_ = data_->spec.mode == GrammarSpec::Mode::kNamed ? Stage::kEnd : Stage::kTop;
        return;
      }
      (void)feed_json(id);
      return;
    case Stage::kAnswer:
      if (id == m.end_message.id) {
        answered_ = true;
        stage_ = Stage::kEnd;
        return;
      }
      (void)feed_json(id);
      return;
  }
}

const char* InklingToolGrammar::state_name() const {
  switch (stage_) {
    case Stage::kTop: return "inkling top";
    case Stage::kHeader: return "inkling header";
    case Stage::kThinking: return "inkling thinking";
    case Stage::kText: return "inkling text";
    case Stage::kCall: return "inkling call";
    case Stage::kAnswer: return "inkling answer";
    case Stage::kEnd: return "inkling end";
    case Stage::kDone: return "done";
  }
  return "?";
}

}  // namespace dgpp::text
