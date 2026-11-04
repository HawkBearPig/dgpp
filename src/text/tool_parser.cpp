#include "text/tool_parser.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "text/chat_template.hpp"
#include "text/json_grammar.hpp"
#include "text/tokenizer.hpp"

namespace dgpp::text {

// ---------------------------------------------------------------------------
// Markers
// ---------------------------------------------------------------------------

ChatMarkers ChatMarkers::from_tokenizer(const Tokenizer& tok) {
  ChatMarkers m;
  const auto lookup = [&](const char* text) {
    ChatMarker out;
    out.text = text;
    for (const auto& added : tok.added_tokens())
      if (added.content == text) {
        out.id = added.id;
        break;
      }
    return out;
  };
  m.think_open = lookup("<think>");
  m.think_close = lookup("</think>");
  // MiniMax carries both canonical and model-specific reasoning tokens.
  // Its shipped template uses the latter, even though both pairs exist.
  m.minimax_namespace = lookup("]<]minimax[>[");
  if (m.minimax_namespace.available()) {
    m.think_open = lookup("<mm:think>");
    m.think_close = lookup("</mm:think>");
  }
  m.tool_call_open = lookup("<tool_call>");
  m.tool_call_close = lookup("</tool_call>");
  m.arg_key_open = lookup("<arg_key>");
  m.arg_key_close = lookup("</arg_key>");
  m.arg_value_open = lookup("<arg_value>");
  m.arg_value_close = lookup("</arg_value>");
  m.dsml = lookup("｜DSML｜");
  // The MiMo tokenizers (the Qwen2 vocabulary with the audio / video
  // markers) go with the compact XML call format.
  m.xml_compact = lookup("<|mimo_audio_start|>").available();
  for (const char* role :
       {"<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|im_start|>", "<|im_end|>",
        "<｜System｜>", "<｜User｜>", "<｜Assistant｜>", "]~!b[", "]~b]", "[e~["}) {
    const ChatMarker r = lookup(role);
    if (r.available()) m.role_markers.push_back(r);
  }
  {
    const std::vector<int64_t> nl = tok.encode("\n");
    if (nl.size() == 1) m.newline = ChatMarker{nl[0], "\n"};
    for (int run = 1; run <= 4; ++run) {
      const std::vector<int64_t> ids = tok.encode(std::string(static_cast<size_t>(run), '\n'));
      if (ids.size() == 1) m.newline_ids.push_back(ids[0]);
    }
  }
  return m;
}

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

ToolSchemas::ToolSchemas(const minijson::Value& tools) {
  for (const minijson::Value& entry : tools.items()) {
    if (!entry.is_object()) continue;
    const minijson::Value* fn = entry.find("function");
    const minijson::Value& tool = fn != nullptr && fn->is_object() ? *fn : entry;
    const minijson::Value* name = tool.find("name");
    if (name == nullptr || !name->is_string()) continue;
    std::map<std::string, Type>& params = types_[std::string(name->as_string())];
    const minijson::Value* parameters = tool.find("parameters");
    if (parameters == nullptr || !parameters->is_object()) continue;
    parameters_[std::string(name->as_string())] = json_text_of(*parameters);
    const minijson::Value* properties = parameters->find("properties");
    if (properties == nullptr || !properties->is_object()) continue;
    for (const minijson::Member& prop : properties->members()) {
      Type t = Type::kUnknown;
      if (prop.value.is_object()) {
        const minijson::Value* type = prop.value.find("type");
        if (type != nullptr && type->is_string()) {
          const std::string_view ty = type->as_string();
          if (ty == "string")
            t = Type::kString;
          else if (ty == "integer" || ty == "number" || ty == "boolean" ||
                   ty == "object" || ty == "array" || ty == "null")
            t = Type::kJson;
        }
        // Constrained strings are spelled as JSON string literals so their
        // escapes and arbitrary contents survive every model's delimiters.
        if (prop.value.find("pattern") || prop.value.find("format") ||
            prop.value.find("$ref") || prop.value.find("anyOf") || prop.value.find("x-dgpp-grammar"))
          t = Type::kJson;
      }
      params[prop.key] = t;
    }
  }
}

ToolSchemas::Type ToolSchemas::type_of(const std::string& function,
                                     const std::string& key) const {
  const auto fn = types_.find(function);
  if (fn == types_.end()) return Type::kUnknown;
  const auto param = fn->second.find(key);
  return param == fn->second.end() ? Type::kUnknown : param->second;
}

const std::string& ToolSchemas::parameters(const std::string& function) const {
  static const std::string empty = "{}";
  const auto it = parameters_.find(function);
  return it == parameters_.end() ? empty : it->second;
}

// ---------------------------------------------------------------------------
// The parser
// ---------------------------------------------------------------------------

ToolCallParser::ToolCallParser(ChatMarkers markers, Decode decode,
                               ToolSchemas schemas, Options options)
    : markers_(std::move(markers)),
      decode_(std::move(decode)),
      schemas_(std::move(schemas)),
      options_(std::move(options)) {
  if (options_.start_in_tool_call && markers_.tool_calls_available()) {
    state_ = State::kToolCall;
    sub_ = Sub::kName;
    seeded_name_ = options_.seeded_name;
    raw_has_prefix_ = false;
  } else if (options_.start_in_reasoning && markers_.reasoning_available()) {
    state_ = State::kReasoning;
  } else {
    state_ = State::kContent;
  }
}

void ToolCallParser::run_append(int64_t id, Event::Kind kind,
                                std::vector<Event>* out) {
  Event ev;
  ev.kind = kind;
  ev.text = decode_({id});
  if (ev.text.empty()) return;
  if (options_.track_tokens) ev.tokens.push_back({current_token_, 0, ev.text.size()});
  out->push_back(std::move(ev));
}

void ToolCallParser::enter_tool_call(int64_t opening_id) {
  state_ = State::kToolCall;
  sub_ = Sub::kName;
  raw_.clear();
  raw_.push_back(opening_id);
  raw_indices_.assign(1, current_token_);
  raw_has_prefix_ = true;
  seeded_name_.clear();
  name_ids_.clear();
  key_ids_.clear();
  value_ids_.clear();
  name_.clear();
  key_.clear();
  args_.clear();
  run_ = Run{};
}

void ToolCallParser::abort_block(std::vector<Event>* out) {
  // The block's literal text becomes content: markers decode to their
  // text (they are not special tokens), so a client sees exactly what the
  // model wrote.
  std::string text;
  if (!raw_has_prefix_) text = options_.forced_prefix_text;
  text += decode_(raw_);
  if (!text.empty()) {
    Event ev;
    ev.kind = Event::Kind::kContent;
    ev.text = std::move(text);
    annotate_block(&ev, false);
    out->push_back(std::move(ev));
  }
  state_ = State::kContent;
  run_ = Run{};
  raw_.clear();
}

void ToolCallParser::slice_tokens(Event* ev, const std::vector<Event::TokenSpan>& tokens,
                                  size_t begin, size_t end) {
  for (const auto& span : tokens)
    if (span.begin < end && span.end > begin)
      ev->tokens.push_back({span.token, std::max(span.begin, begin) - begin,
                            std::min(span.end, end) - begin});
}

void ToolCallParser::annotate_block(Event* ev, bool dsml) const {
  if (!options_.track_tokens) return;
  // Malformed blocks become literal content, including their delimiters.
  // Reconstruct the same suffix decodes to retain each generated token's
  // byte range; this uncommon path runs only when logprobs were requested.
  size_t offset = dsml ? dsml_held_.size()
                       : raw_has_prefix_ ? 0 : options_.forced_prefix_text.size();
  if (dsml) ev->tokens = dsml_held_tokens_;
  std::vector<int64_t> segment;
  size_t decoded = 0;
  for (size_t i = 0; i < raw_.size(); ++i) {
    size_t bytes = 0;
    if (dsml && is_marker(raw_[i], markers_.dsml)) {
      bytes = markers_.dsml.text.size();
      segment.clear();
      decoded = 0;
    } else {
      segment.push_back(raw_[i]);
      const size_t length = decode_(segment).size();
      bytes = length > decoded ? length - decoded : 0;
      decoded = length;
    }
    if (bytes) ev->tokens.push_back({raw_indices_[i], offset, offset + bytes});
    offset += bytes;
  }
}

namespace {

std::vector<DecimalNumber> numeric_values(const std::string& text) {
  std::vector<DecimalNumber> out;
  bool quoted = false, escaped = false;
  for (size_t i = 0; i < text.size(); ++i) {
    const char ch = text[i];
    if (quoted) {
      if (escaped) escaped = false;
      else if (ch == '\\') escaped = true;
      else if (ch == '"') quoted = false;
    } else if (ch == '"') {
      quoted = true;
    } else if (ch == '-' || (ch >= '0' && ch <= '9')) {
      const size_t start = i;
      while (i + 1 < text.size() && text[i + 1] != '\0' &&
             std::strchr("0123456789.eE+-", text[i + 1]) != nullptr) ++i;
      out.push_back(DecimalNumber::parse(std::string_view(text).substr(start, i - start + 1)));
    }
  }
  return out;
}

std::string normalized_json(const std::string& text) {
  try {
    const auto parsed = minijson::parse(text);
    size_t end = parsed.consumed;
    while (end < text.size() && JsonLexer::is_ws(static_cast<uint8_t>(text[end]))) ++end;
    if (end != text.size()) throw std::invalid_argument("trailing text");
    const std::string normalized = Value::from_minijson(parsed.root).to_json(false);
    const auto before = numeric_values(text), after = numeric_values(normalized);
    bool exact = before.size() == after.size();
    for (size_t i = 0; exact && i < before.size(); ++i)
      exact = before[i].compare(after[i]) == 0;
    if (exact) return normalized;
  } catch (const std::exception&) {
    // A valid JSON number may be outside the DOM's int64/double range.
  }
  // Preserve numeric spellings when normalization would move a value
  // across a schema bound, including numbers inside arrays and objects.
  JsonLexer lexer;
  for (const unsigned char ch : text)
    if (!lexer.feed(ch).ok) throw std::invalid_argument("not a JSON value");
  if (!lexer.done()) throw std::invalid_argument("incomplete JSON value");
  const size_t first = text.find_first_not_of(" \n\r\t");
  const size_t last = text.find_last_not_of(" \n\r\t");
  return text.substr(first, last - first + 1);
}

}  // namespace

std::string ToolCallParser::typed_value(const std::string& function,
                                        const std::string& key,
                                        const std::string& text) const {
  const ToolSchemas::Type type = schemas_.type_of(function, key);
  if (type != ToolSchemas::Type::kString) {
    // JSON when the whole text parses as one value (the template's tojson
    // of a non-string); the text itself otherwise.
    try {
      return normalized_json(text);
    } catch (const std::exception&) {
      // not JSON — a string it is
    }
  }
  return Value::string_value(text).to_json(/*ensure_ascii=*/false);
}

// "<function=NAME>\n(<parameter=K>\nV\n</parameter>\n)*</function>\n" —
// lenient about the newlines around the tags (a model may drop or double
// one), strict about the tags themselves; a value keeps its inner text
// verbatim (the template writes strings raw and other values as JSON,
// which typed_value() sorts out).
// Both dialects of the format parse here: Qwen3.8's writes a newline after
// every tag and its value is what sits between the newlines (one leading
// and one trailing newline are the format's, not the value's); the
// compact dialect (ChatMarkers::xml_compact — MiMo-V2.6) writes no
// newlines, so a value's newlines are all its own. A forced call under the
// compact dialect arrives in the newline form (the grammar emits it): the
// format's own newlines are the ones RIGHT AFTER a ">" and RIGHT BEFORE a
// "</parameter>", stripped once under the Qwen dialect; under the compact
// dialect a value between "\n" .. "\n" is stripped only when the block is
// in the newline form throughout (a newline follows the function name).
bool ToolCallParser::parse_qwen_block(const std::string& text) {
  size_t i = 0;
  const auto skip_ws = [&] {
    while (i < text.size() && (text[i] == '\n' || text[i] == ' ' || text[i] == '\r' || text[i] == '\t')) ++i;
  };
  const auto accept = [&](const char* lit) {
    const size_t n = std::strlen(lit);
    if (text.compare(i, n, lit) != 0) return false;
    i += n;
    return true;
  };
  skip_ws();
  if (!accept("<function=")) return false;
  const size_t name_end = text.find('>', i);
  if (name_end == std::string::npos || name_end == i) return false;
  name_ = text.substr(i, name_end - i);
  if (name_.find('\n') != std::string::npos || name_.find('<') != std::string::npos) return false;
  i = name_end + 1;
  const bool newline_form = !markers_.xml_compact || (i < text.size() && text[i] == '\n');
  if (newline_form && i < text.size() && text[i] == '\n') ++i;
  args_.clear();
  args_json_ = false;
  // One JSON object as the body (the MiMo template writes a client's
  // pre-serialized arguments so): its members are the call's arguments.
  {
    size_t j = i;
    while (j < text.size() && (text[j] == '\n' || text[j] == ' ')) ++j;
    if (j < text.size() && text[j] == '{') {
      const size_t end = text.rfind("</function>");
      if (end == std::string::npos || end < j) return false;
      minijson::ParseResult parsed;
      try {
        parsed = minijson::parse(std::string_view(text).substr(j, end - j));
      } catch (const std::exception&) {
        return false;
      }
      if (!parsed.root.is_object()) return false;
      for (const minijson::Member& m : parsed.root.members()) {
        for (const auto& seen : args_)
          if (seen.first == m.key) return false;
        args_.emplace_back(m.key, Value::from_minijson(m.value).to_json(false));
      }
      args_json_ = true;
      i = end + std::strlen("</function>");
      skip_ws();
      return i == text.size();
    }
  }
  for (;;) {
    if (accept("<parameter=")) {
      const size_t key_end = text.find('>', i);
      if (key_end == std::string::npos || key_end == i) return false;
      const std::string key = text.substr(i, key_end - i);
      if (key.find('\n') != std::string::npos || key.find('<') != std::string::npos) return false;
      for (const auto& seen : args_)
        if (seen.first == key) return false;  // a duplicate parameter
      i = key_end + 1;
      if (newline_form && i < text.size() && text[i] == '\n') ++i;
      const size_t close = text.find("</parameter>", i);
      if (close == std::string::npos) return false;
      std::string value = text.substr(i, close - i);
      if (newline_form && !value.empty() && value.back() == '\n') value.pop_back();
      args_.emplace_back(key, std::move(value));
      i = close + std::strlen("</parameter>");
      if (newline_form && i < text.size() && text[i] == '\n') ++i;
      continue;
    }
    break;
  }
  if (!accept("</function>")) return false;
  skip_ws();
  return i == text.size();
}

// MiniMax's namespace precedes every structural tag. Bare '<' and '&'
// inside scalar values are literal bytes, not XML entities or markup.
namespace {
class MiniMaxXmlReader {
 public:
  MiniMaxXmlReader(const std::string& text, const std::string& ns) : text_(text), ns_(ns) {}
  bool accept(const std::string& s) {
    if (text_.compare(pos_, s.size(), s) != 0) return false;
    pos_ += s.size();
    return true;
  }
  bool tag(const std::string& s) { return accept(ns_ + "<" + s + ">"); }
  void whitespace() {
    while (pos_ < text_.size() && JsonLexer::is_ws(static_cast<uint8_t>(text_[pos_]))) ++pos_;
  }
  bool end() const { return pos_ == text_.size(); }
  bool invoke(std::string* name) {
    if (!accept(ns_ + "<invoke name=\"")) return false;
    const auto end = text_.find('"', pos_);
    if (end == std::string::npos || end == pos_) return false;
    *name = text_.substr(pos_, end - pos_);
    if (name->find_first_of("<>\r\n") != std::string::npos) return false;
    pos_ = end;
    return accept("\">");
  }
  bool value(const std::string& close, const minijson::Value* schema, std::string* result,
             int depth = 0, bool root = false) {
    if (depth > 128) return false;
    if (root) root_schema_ = schema;
    schema = resolve_schema(schema);
    std::string type;
    if (schema) {
      const auto* t = schema->find("type");
      const auto* en = schema->find("enum");
      const auto* cs = schema->find("const");
      const auto value_class = [](const minijson::Value& v, std::string_view kind) {
        return (kind == "string" && v.is_string()) || (kind == "object" && v.is_object()) ||
               (kind == "array" && v.is_array()) || (kind == "boolean" && v.is_bool()) ||
               (kind == "null" && v.is_null()) ||
               (kind == "integer" && v.kind() == minijson::Value::Kind::Int) ||
               (kind == "number" && v.is_number());
      };
      if (t || en || cs)
        for (const char* candidate :
             {"string", "object", "array", "integer", "number", "boolean", "null"}) {
          bool declared = !t || (t->is_string() && t->as_string() == candidate);
          if (t && t->is_array())
            for (const auto& item : t->items()) declared |= item.as_string() == candidate;
          bool enumerated = !en && !cs;
          if (cs) enumerated = value_class(*cs, candidate);
          if (en && en->is_array())
            for (const auto& item : en->items()) enumerated |= value_class(item, candidate);
          if (declared && enumerated) {
            type = candidate;
            break;
          }
        }
    }
    const std::string closer = ns_ + "</" + close + ">";
    bool children = text_.compare(pos_, ns_.size() + 1, ns_ + "<") == 0 &&
                    text_.compare(pos_, closer.size(), closer) != 0;
    bool array = type == "array" || (type.empty() && children &&
                                     text_.compare(pos_, ns_.size() + 6, ns_ + "<item>") == 0);
    bool object = root || type == "object" || (type.empty() && children && !array);
    if (array || object) {
      *result = array ? "[" : "{";
      std::vector<std::string> seen;
      bool first = true;
      for (;;) {
        if (accept(closer)) break;
        // The template places no whitespace inside containers. Accept it
        // between tags, preserving all scalar whitespace in the other arm.
        whitespace();
        if (accept(closer)) break;
        if (!accept(ns_ + "<")) return false;
        const auto end = text_.find('>', pos_);
        if (end == std::string::npos || end == pos_) return false;
        const std::string key = text_.substr(pos_, end - pos_);
        if (key.find_first_of("<>/\"' \n\r\t") != std::string::npos) return false;
        if (array && key != "item") return false;
        if (!array && std::find(seen.begin(), seen.end(), key) != seen.end()) return false;
        seen.push_back(key);
        pos_ = end + 1;
        const minijson::Value* child = nullptr;
        if (schema) {
          if (array)
            child = schema->find("items");
          else if (const auto* properties = schema->find("properties"))
            child = properties->find(key);
          if (!array && !child) {
            const auto* extra = schema->find("additionalProperties");
            if (extra && extra->is_object()) child = extra;
          }
        }
        std::string item;
        if (!value(key, child, &item, depth + 1)) return false;
        if (!first) *result += ", ";
        first = false;
        if (!array) *result += Value::string_value(key).to_json(false) + ": ";
        *result += item;
      }
      *result += array ? "]" : "}";
      return true;
    }
    const auto end = text_.find(closer, pos_);
    if (end == std::string::npos) return false;
    const auto structural = text_.find(ns_ + "<", pos_);
    if (structural != std::string::npos && structural < end) return false;
    const std::string scalar = text_.substr(pos_, end - pos_);
    pos_ = end + closer.size();
    if (type != "string") {
      try {
        *result = normalized_json(scalar);
        // Containers must use the recursive XML spelling, never raw JSON.
        if (!result->empty() && ((*result)[0] == '{' || (*result)[0] == '[')) return false;
        if (type == "boolean" && *result != "true" && *result != "false") return false;
        if (type == "null" && *result != "null") return false;
        if (type == "integer" || type == "number") {
          if (result->empty() ||
              ((*result)[0] != '-' && ((*result)[0] < '0' || (*result)[0] > '9')))
            return false;
          if (type == "integer" && result->find_first_of(".eE") != std::string::npos) return false;
        }
        return true;
      } catch (const std::exception&) {
        if (!type.empty()) return false;
      }
    }
    *result = Value::string_value(scalar).to_json(false);
    return true;
  }

 private:
  const minijson::Value* resolve_schema(const minijson::Value* schema) const {
    for (int depth = 0; schema && depth < 128; ++depth) {
      if (const auto* ref = schema->find("$ref"); ref && ref->is_string()) {
        const std::string path(ref->as_string());
        if (path.empty() || path[0] != '#') return schema;
        schema = root_schema_;
        size_t begin = 2;
        while (schema && begin <= path.size() && path.size() > 1) {
          const auto end = path.find('/', begin);
          std::string key = path.substr(begin, end == std::string::npos ? end : end - begin);
          for (size_t p = 0; p + 1 < key.size(); ++p)
            if (key[p] == '~') {
              if (key[p + 1] == '0' || key[p + 1] == '1') {
                key.replace(p, 2, key[p + 1] == '0' ? "~" : "/");
              }
            }
          schema = schema->find(key);
          if (end == std::string::npos) break;
          begin = end + 1;
        }
        continue;
      }
      if (const auto* any = schema->find("anyOf"); any && any->is_array()) {
        const minijson::Value* selected = nullptr;
        for (const auto& alt : any->items()) {
          const auto* type = alt.find("type");
          if (!type || type->as_string() != "null") {
            selected = &alt;
            break;
          }
        }
        if (selected) {
          schema = selected;
          continue;
        }
      }
      return schema;
    }
    return schema;
  }
  const std::string& text_;
  const std::string& ns_;
  const minijson::Value* root_schema_ = nullptr;
  size_t pos_ = 0;
};
}  // namespace

bool ToolCallParser::parse_minimax_block(const std::string& text, std::vector<Call>* calls) const {
  MiniMaxXmlReader reader(text, markers_.minimax_namespace.text);
  if (!reader.tag("tool_call")) return false;
  reader.whitespace();
  while (!reader.tag("/tool_call")) {
    Call call;
    if (!reader.invoke(&call.name)) return false;
    const auto schema = minijson::parse(schemas_.parameters(call.name));
    if (!reader.value("invoke", &schema.root, &call.arguments, 0, true)) return false;
    calls->push_back(std::move(call));
    reader.whitespace();
  }
  return reader.end() && !calls->empty();
}

void ToolCallParser::minimax_flush(size_t bytes, std::vector<Event>* out) {
  if (!bytes) return;
  Event event;
  event.kind = Event::Kind::kContent;
  event.text = minimax_run_.text.substr(0, bytes);
  slice_tokens(&event, minimax_run_.tokens, 0, bytes);
  out->push_back(std::move(event));
  Event tail;
  slice_tokens(&tail, minimax_run_.tokens, bytes, minimax_run_.text.size());
  minimax_run_.text.erase(0, bytes);
  minimax_run_.tokens = std::move(tail.tokens);
}

void ToolCallParser::minimax_append(int64_t id, std::vector<Event>* out) {
  const size_t begin = minimax_run_.text.size();
  minimax_run_.text += decode_({id});
  if (options_.track_tokens && minimax_run_.text.size() > begin)
    minimax_run_.tokens.push_back({current_token_, begin, minimax_run_.text.size()});
  const std::string open = markers_.minimax_namespace.text + "<tool_call>";
  const std::string close = markers_.minimax_namespace.text + "</tool_call>";
  for (;;) {
    if (state_ == State::kContent) {
      const auto found = minimax_run_.text.find(open);
      if (found == std::string::npos) {
        size_t held = std::min(open.size() - 1, minimax_run_.text.size());
        while (held &&
               minimax_run_.text.compare(minimax_run_.text.size() - held, held, open, 0, held))
          --held;
        minimax_flush(minimax_run_.text.size() - held, out);
        return;
      }
      minimax_flush(found, out);
      state_ = State::kToolCall;
    }
    const auto end = minimax_run_.text.find(close, open.size());
    const auto nested = minimax_run_.text.find(open, open.size());
    if (nested != std::string::npos && (end == std::string::npos || nested < end)) {
      minimax_flush(nested, out);
      continue;
    }
    if (end == std::string::npos) return;
    const size_t bytes = end + close.size();
    std::vector<Call> calls;
    if (!parse_minimax_block(minimax_run_.text.substr(0, bytes), &calls)) {
      minimax_flush(bytes, out);
    } else {
      for (auto& call : calls) {
        Event event;
        event.kind = Event::Kind::kToolCall;
        event.call = std::move(call);
        out->push_back(std::move(event));
        ++calls_;
      }
      Event tail;
      slice_tokens(&tail, minimax_run_.tokens, bytes, minimax_run_.text.size());
      minimax_run_.text.erase(0, bytes);
      minimax_run_.tokens = std::move(tail.tokens);
    }
    state_ = State::kContent;
  }
}

// ---- the DSML format ------------------------------------------------------------

namespace {
constexpr const char* kDsmlText = "｜DSML｜";
// The block's tags per dialect (ChatMarkers::dsml_dialect): V4.1 writes a
// space after the tag token and names the block "calls"; V4 writes no
// space and names it "tool_calls".
struct DsmlTags {
  const char* calls_open;
  const char* calls_close;
  const char* invoke_open;
  const char* invoke_close;
  const char* param_open;
  const char* param_close;  // the reference's end token (the value ends with "<")
  bool namespaces;          // an invoke's "ns::name" reports the bare name
  // An invoke's header may end in a blank line (">\n\n"): both references
  // render an argument-less call as "<invoke name=...>\n\n</invoke>" (the
  // empty parameter list between the template's two newlines) and their
  // parsers accept it (the header regex's "$" matches before a final
  // newline). Read for V4; V4.1's parser predates the finding and is left
  // as it was (the block is then literal content there).
  bool blank_line_head;
};
constexpr DsmlTags kDsmlV41Tags = {"<｜DSML｜ calls", "</｜DSML｜ calls>", "<｜DSML｜ invoke", "</｜DSML｜ invoke",
                                   "<｜DSML｜ parameter", "/｜DSML｜ parameter", /*namespaces=*/true,
                                   /*blank_line_head=*/false};
constexpr DsmlTags kDsmlV4Tags = {"<｜DSML｜tool_calls", "</｜DSML｜tool_calls>", "<｜DSML｜invoke", "</｜DSML｜invoke",
                                  "<｜DSML｜parameter", "/｜DSML｜parameter", /*namespaces=*/false,
                                  /*blank_line_head=*/true};
const DsmlTags& dsml_tags(DsmlDialect dialect) {
  return dialect == DsmlDialect::kV4 ? kDsmlV4Tags : kDsmlV41Tags;
}

// The longest suffix of `text` that could begin a block: "\n\n<", "\n\n",
// "\n" (the reference writes the block after a blank line) or "<".
size_t dsml_held_suffix(const std::string& text) {
  for (const char* p : {"\n\n<", "\n\n", "\n", "<"}) {
    const size_t n = std::strlen(p);
    if (text.size() >= n && text.compare(text.size() - n, n, p) == 0) return n;
  }
  return 0;
}

// The reference's _read_until_stop: the text from `index` up to the
// nearest of the stop strings; the index past it; which stop matched
// (-1: none, the rest of the text).
struct Stop {
  size_t next = 0;
  std::string content;
  int which = -1;
};
Stop read_until(const std::string& text, size_t index, const std::vector<const char*>& stops) {
  Stop s;
  size_t best = std::string::npos;
  for (size_t k = 0; k < stops.size(); ++k) {
    const size_t pos = text.find(stops[k], index);
    if (pos != std::string::npos && pos < best) {
      best = pos;
      s.which = static_cast<int>(k);
    }
  }
  if (s.which < 0) {
    s.content = text.substr(index);
    s.next = text.size();
  } else {
    s.content = text.substr(index, best - index);
    s.next = best + std::strlen(stops[static_cast<size_t>(s.which)]);
  }
  return s;
}
}  // namespace

void ToolCallParser::dsml_flush_held(std::vector<Event>* out) {
  // Everything past the emitted length is content (the held prefix was
  // not a block after all — or the generation ended on it).
  if (run_.text.size() > dsml_emitted_) {
    Event ev;
    ev.kind = Event::Kind::kContent;
    ev.text.assign(run_.text, dsml_emitted_, std::string::npos);
    slice_tokens(&ev, run_.tokens, dsml_emitted_, run_.text.size());
    out->push_back(std::move(ev));
    dsml_emitted_ = run_.text.size();
  }
}

void ToolCallParser::dsml_content_append(int64_t id, std::vector<Event>* out) {
  if (is_marker(id, markers_.dsml)) {
    // The tag token after "<": the block opens; the content run keeps what
    // stood before the held prefix, the prefix rides into the block (the
    // reference cuts the content before its "\n\n<").
    if (!run_.text.empty() && run_.text.back() == '<') {
      const size_t held = dsml_held_suffix(run_.text);
      dsml_held_ = run_.text.substr(run_.text.size() - held);
      Event provenance;
      slice_tokens(&provenance, run_.tokens, run_.text.size() - held, run_.text.size());
      dsml_held_tokens_ = std::move(provenance.tokens);
      enter_dsml_block();
      raw_.push_back(id);
      raw_indices_.push_back(current_token_);
      return;
    }
    // A tag token without its "<": literal text of the run — but the
    // service's decode skips it (a special token), so the run's decode
    // would drop it: flush the held prefix and print the tag verbatim.
    dsml_flush_held(out);
    Event ev;
    ev.kind = Event::Kind::kContent;
    ev.text = kDsmlText;
    if (options_.track_tokens) ev.tokens.push_back({current_token_, 0, ev.text.size()});
    out->push_back(std::move(ev));
    run_.text += decode_({id});  // the production decoder skips this tag
    dsml_emitted_ = run_.text.size();
    return;
  }
  const size_t begin = run_.text.size();
  run_.text += decode_({id});
  if (options_.track_tokens && run_.text.size() > begin)
    run_.tokens.push_back({current_token_, begin, run_.text.size()});
  const size_t held = dsml_held_suffix(run_.text);
  const size_t emit_to = run_.text.size() - held;
  if (emit_to > dsml_emitted_) {
    Event ev;
    ev.kind = Event::Kind::kContent;
    ev.text.assign(run_.text, dsml_emitted_, emit_to - dsml_emitted_);
    slice_tokens(&ev, run_.tokens, dsml_emitted_, emit_to);
    out->push_back(std::move(ev));
    dsml_emitted_ = emit_to;
  }
}

void ToolCallParser::enter_dsml_block() {
  state_ = State::kToolCall;
  raw_.clear();
  raw_indices_.clear();
  raw_has_prefix_ = true;
  dsml_calls_.clear();
  run_ = Run{};
  dsml_emitted_ = 0;
}

// The block's text with the tag token restored: the service's decode
// skips it, so the runs between the tag ids decode separately.
std::string ToolCallParser::dsml_block_text() const {
  std::string text = dsml_held_;
  std::vector<int64_t> segment;
  for (const int64_t id : raw_) {
    if (is_marker(id, markers_.dsml)) {
      if (!segment.empty()) text += decode_(segment);
      segment.clear();
      text += kDsmlText;
    } else {
      segment.push_back(id);
    }
  }
  if (!segment.empty()) text += decode_(segment);
  return text;
}

bool ToolCallParser::dsml_block_closed(const std::string& text) const {
  const char* close = dsml_tags(markers_.dsml_dialect).calls_close;
  const size_t n = std::strlen(close);
  return text.size() >= n && text.compare(text.size() - n, n, close) == 0;
}

// The reference's parse_tool_calls over the closed block (the V4.1
// spelling shown; V4's tags are "tool_calls" / "invoke" / "parameter"
// without the space): after
// "<｜DSML｜ calls" exactly ">\n", then invokes — each ` name="NAME">\n`,
// parameters ` name="K" string="true|false">V<` closed by
// "/｜DSML｜ parameter" and followed by ">\n", the invoke closed by
// "</｜DSML｜ invoke" and ">\n" — up to "</｜DSML｜ calls>" with nothing
// after it. A string value is JSON-encoded; a JSON value that parses is
// kept as written (normalized), one that does not becomes a string.
bool ToolCallParser::parse_dsml_block(const std::string& text) {
  const DsmlTags& tags = dsml_tags(markers_.dsml_dialect);
  dsml_calls_.clear();
  size_t index = text.find(tags.calls_open);
  if (index == std::string::npos) return false;
  index += std::strlen(tags.calls_open);
  for (;;) {
    Stop s = read_until(text, index, {tags.invoke_open, tags.calls_close});
    if (s.content != ">\n" || s.which < 0) return false;
    index = s.next;
    if (s.which == 1) break;  // the calls end
    Stop head = read_until(text, index, {tags.param_open, tags.invoke_close});
    if (head.which < 0) return false;
    index = head.next;
    // ^\s*name="(.*?)">\n$
    std::string h = head.content;
    size_t ws = 0;
    while (ws < h.size() && (h[ws] == ' ' || h[ws] == '\n' || h[ws] == '\t' || h[ws] == '\r')) ++ws;
    h.erase(0, ws);
    const std::string pre = "name=\"";
    const std::string post = "\">\n";
    // ...">\n$: Python's "$" also matches before one final newline.
    if (tags.blank_line_head && h.size() > post.size() &&
        h.compare(h.size() - post.size() - 1, std::string::npos, post + "\n") == 0)
      h.pop_back();
    if (h.size() < pre.size() + post.size() || h.compare(0, pre.size(), pre) != 0 ||
        h.compare(h.size() - post.size(), post.size(), post) != 0)
      return false;
    const std::string qualified = h.substr(pre.size(), h.size() - pre.size() - post.size());
    if (qualified.find("\">\n") != std::string::npos) return false;
    Call call;
    const size_t ns = tags.namespaces ? qualified.find("::") : std::string::npos;
    call.name = ns == std::string::npos ? qualified : qualified.substr(ns + 2);
    std::vector<std::pair<std::string, std::string>> params;
    std::vector<std::string> seen;
    while (head.which == 0) {
      Stop p = read_until(text, index, {tags.param_close});
      if (p.which < 0) return false;
      index = p.next;
      // ^ name="(.*?)" string="(true|false)">(.*?)<$
      const std::string& c = p.content;
      const std::string p1 = " name=\"";
      if (c.compare(0, p1.size(), p1) != 0) return false;
      const size_t name_end = c.find("\" string=\"", p1.size());
      if (name_end == std::string::npos) return false;
      const std::string key = c.substr(p1.size(), name_end - p1.size());
      size_t k = name_end + std::strlen("\" string=\"");
      bool is_string = false;
      if (c.compare(k, 4, "true") == 0) { is_string = true; k += 4; }
      else if (c.compare(k, 5, "false") == 0) { k += 5; }
      else return false;
      if (c.compare(k, 2, "\">") != 0) return false;
      k += 2;
      if (c.empty() || c.back() != '<') return false;
      const std::string value = c.substr(k, c.size() - 1 - k);
      for (const std::string& s0 : seen)
        if (s0 == key) return false;  // a duplicate parameter
      seen.push_back(key);
      std::string json;
      if (is_string) {
        json = Value::string_value(value).to_json(false);
      } else {
        try {
          json = normalized_json(value);
        } catch (const std::exception&) {
          json = Value::string_value(value).to_json(false);
        }
      }
      params.emplace_back(key, json);
      Stop gap = read_until(text, index, {tags.param_open, tags.invoke_close});
      if (gap.content != ">\n" || gap.which < 0) return false;
      index = gap.next;
      head.which = gap.which;
    }
    std::string args = "{";
    for (size_t i = 0; i < params.size(); ++i) {
      if (i) args += ", ";
      args += Value::string_value(params[i].first).to_json(false) + ": " + params[i].second;
    }
    args += "}";
    call.arguments = std::move(args);
    dsml_calls_.push_back(std::move(call));
  }
  // Nothing may follow the calls end.
  return index == text.size();
}

void ToolCallParser::complete_dsml_block(std::vector<Event>* out) {
  for (Call& c : dsml_calls_) {
    Event ev;
    ev.kind = Event::Kind::kToolCall;
    ev.call = std::move(c);
    out->push_back(std::move(ev));
    ++calls_;
  }
  dsml_calls_.clear();
  state_ = State::kContent;
  run_ = Run{};
  raw_.clear();
  dsml_held_.clear();
  dsml_emitted_ = 0;
}

void ToolCallParser::abort_dsml_block(std::vector<Event>* out) {
  const std::string text = dsml_block_text();
  if (!text.empty()) {
    Event ev;
    ev.kind = Event::Kind::kContent;
    ev.text = text;
    annotate_block(&ev, true);
    out->push_back(std::move(ev));
  }
  state_ = State::kContent;
  run_ = Run{};
  raw_.clear();
  dsml_held_.clear();
  dsml_emitted_ = 0;
}

void ToolCallParser::complete_block(std::vector<Event>* out) {
  Event ev;
  ev.kind = Event::Kind::kToolCall;
  ev.call.name = name_;
  std::string args = "{";
  for (size_t i = 0; i < args_.size(); ++i) {
    if (i) args += ", ";
    args += Value::string_value(args_[i].first).to_json(false);
    args += ": ";
    args += args_json_ ? args_[i].second : typed_value(name_, args_[i].first, args_[i].second);
  }
  args += "}";
  ev.call.arguments = std::move(args);
  out->push_back(std::move(ev));
  ++calls_;
  state_ = State::kContent;
  run_ = Run{};
  raw_.clear();
}

void ToolCallParser::feed(int64_t id, std::vector<Event>* out) {
  current_token_ = next_token_++;
  switch (state_) {
    case State::kReasoning:
      if (is_marker(id, markers_.think_close)) {
        state_ = State::kContent;
        run_ = Run{};
        Event ev;
        ev.kind = Event::Kind::kReasoningClosed;
        if (options_.track_tokens)
          ev.tokens.push_back({current_token_, 0, markers_.think_close.text.size()});
        out->push_back(std::move(ev));
        return;
      }
      // The prompt already opened the block; a repeated opener is noise.
      if (is_marker(id, markers_.think_open)) return;
      run_append(id, Event::Kind::kReasoning, out);
      return;

    case State::kContent:
      // MiniMax's adaptive no-thinking reply starts with </mm:think>
      // without an opener. It is a mode delimiter, not answer text.
      if (options_.model_may_open_thinking && !content_started_ &&
          markers_.minimax_namespace.available() && is_marker(id, markers_.think_close)) {
        content_started_ = true;
        Event ev;
        ev.kind = Event::Kind::kReasoningClosed;
        if (options_.track_tokens)
          ev.tokens.push_back({current_token_, 0, markers_.think_close.text.size()});
        out->push_back(std::move(ev));
        return;
      }
      // The model's own opener (the prompt left it the choice): before any
      // content, <think> opens the reasoning.
      if (options_.model_may_open_thinking && !content_started_ && is_marker(id, markers_.think_open) &&
          markers_.reasoning_available()) {
        state_ = State::kReasoning;
        run_ = Run{};
        return;
      }
      if (markers_.tool_format() == ToolFormat::kDsml) {
        content_started_ = true;
        dsml_content_append(id, out);
        return;
      }
      if (markers_.minimax_namespace.available()) {
        content_started_ = true;
        minimax_append(id, out);
        return;
      }
      if (is_marker(id, markers_.tool_call_open) && markers_.tool_calls_available()) {
        content_started_ = true;
        enter_tool_call(id);
        return;
      }
      content_started_ = true;
      run_append(id, Event::Kind::kContent, out);
      return;

    case State::kToolCall:
      break;
  }

  if (markers_.minimax_namespace.available()) {
    minimax_append(id, out);
    return;
  }
  if (markers_.tool_format() == ToolFormat::kDsml) {
    // The DSML block buffers its ids until its closing text; parsed then.
    raw_.push_back(id);
    raw_indices_.push_back(current_token_);
    const std::string text = dsml_block_text();
    if (!dsml_block_closed(text)) return;
    if (parse_dsml_block(text)) complete_dsml_block(out);
    else abort_dsml_block(out);
    return;
  }

  // Inside a block. Every marker is structural; anything else is text of
  // the current segment. A wrong marker aborts the block (its id included
  // in the flushed text); a nested <tool_call> aborts and starts over.
  raw_.push_back(id);
  raw_indices_.push_back(current_token_);
  const bool open = is_marker(id, markers_.tool_call_open);
  const bool close = is_marker(id, markers_.tool_call_close);
  if (markers_.tool_format() == ToolFormat::kQwenXml) {
    // The Qwen format: the block's ids buffer until it closes; the text
    // between the markers is parsed then (a nested opener restarts).
    if (open) {
      raw_.pop_back();
      raw_indices_.pop_back();
      abort_block(out);
      enter_tool_call(id);
      return;
    }
    if (!close) return;
    std::vector<int64_t> inner(raw_.begin() + (raw_has_prefix_ ? 1 : 0), raw_.end() - 1);
    std::string text = raw_has_prefix_ ? "" : options_.forced_prefix_text;
    text += decode_(inner);
    if (parse_qwen_block(text)) complete_block(out);
    else abort_block(out);
    return;
  }
  const bool key_open = is_marker(id, markers_.arg_key_open);
  const bool key_close = is_marker(id, markers_.arg_key_close);
  const bool value_open = is_marker(id, markers_.arg_value_open);
  const bool value_close = is_marker(id, markers_.arg_value_close);
  const bool structural =
      open || close || key_open || key_close || value_open || value_close;

  if (open) {
    raw_.pop_back();  // the nested opener belongs to the next block
    raw_indices_.pop_back();
    abort_block(out);
    enter_tool_call(id);
    return;
  }

  switch (sub_) {
    case Sub::kName:
      if (!structural) {
        name_ids_.push_back(id);
        return;
      }
      name_ = seeded_name_ + decode_(name_ids_);
      if (key_open) {
        sub_ = Sub::kKey;
        key_ids_.clear();
        return;
      }
      if (close) {
        complete_block(out);
        return;
      }
      abort_block(out);
      return;

    case Sub::kKey:
      if (!structural) {
        key_ids_.push_back(id);
        return;
      }
      if (key_close) {
        key_ = decode_(key_ids_);
        sub_ = Sub::kAfterKey;
        return;
      }
      abort_block(out);
      return;

    case Sub::kAfterKey:
      if (value_open) {
        sub_ = Sub::kValue;
        value_ids_.clear();
        return;
      }
      abort_block(out);
      return;

    case Sub::kValue:
      if (!structural) {
        value_ids_.push_back(id);
        return;
      }
      if (value_close) {
        for (const auto& arg : args_) {
          if (arg.first == key_) {
            abort_block(out);
            return;
          }
        }
        args_.emplace_back(key_, decode_(value_ids_));
        sub_ = Sub::kAfterValue;
        return;
      }
      abort_block(out);
      return;

    case Sub::kAfterValue:
      if (key_open) {
        sub_ = Sub::kKey;
        key_ids_.clear();
        return;
      }
      if (close) {
        complete_block(out);
        return;
      }
      abort_block(out);
      return;
  }
}

void ToolCallParser::finish(std::vector<Event>* out) {
  if (markers_.minimax_namespace.available()) {
    minimax_flush(minimax_run_.text.size(), out);
    state_ = State::kContent;
    return;
  }
  if (markers_.tool_format() == ToolFormat::kDsml) {
    if (state_ == State::kToolCall) abort_dsml_block(out);
    else if (state_ == State::kContent) dsml_flush_held(out);
    return;
  }
  if (state_ == State::kToolCall) abort_block(out);
}

}  // namespace dgpp::text
