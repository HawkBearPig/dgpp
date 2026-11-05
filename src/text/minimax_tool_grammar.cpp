#include "text/minimax_tool_grammar.hpp"

#include <algorithm>
#include <stdexcept>

#include "text/chat_template.hpp"
#include "text/tool_grammar.hpp"

namespace dgpp::text {
namespace {
const std::string kNamespace = "]<]minimax[>[";
std::string tag(const std::string& name) {
  return kNamespace + "<" + name + ">";
}
bool xml_name(std::string_view name) {
  return !name.empty() && name.find_first_of("<>/\"' \t\r\n") == std::string_view::npos;
}
std::string quoted(std::string_view text) {
  return Value::string_value(std::string(text)).to_json(false);
}
uint32_t xml_enum_class(const minijson::Value& value) {
  if (value.is_null()) return 0;
  if (value.is_string()) return JsonSchemaNode::kString;
  if (value.is_bool()) return JsonSchemaNode::kBoolean;
  if (value.is_number()) return JsonSchemaNode::kNumber | JsonSchemaNode::kInteger;
  if (value.is_array()) {
    for (const auto& item : value.items())
      if (!xml_enum_class(item)) return 0;
    return JsonSchemaNode::kArray;
  }
  for (const auto& member : value.members())
    if (!xml_name(member.key) || !xml_enum_class(member.value)) return 0;
  return JsonSchemaNode::kObject;
}
std::shared_ptr<const std::vector<uint32_t>> xml_classes(const JsonSchema& schema) {
  // Null has no XML spelling. Compute which finite values remain possible,
  // including recursive schemas, before allowing a container to open.
  auto classes = std::make_shared<std::vector<uint32_t>>(schema.nodes.size(), 0);
  const auto child = [&](int i) -> uint32_t {
    return i == JsonSchemaNode::kAny ? JsonSchemaNode::kAnyClass & ~JsonSchemaNode::kNull
           : i < 0                   ? 0u
                                     : (*classes)[i];
  };
  for (size_t pass = 0; pass <= schema.nodes.size(); ++pass) {
    bool changed = false;
    for (size_t i = 0; i < schema.nodes.size(); ++i) {
      const auto& n = schema.nodes[i];
      uint32_t possible = 0;
      if (!n.any_of.empty()) {
        for (int alt : n.any_of) possible |= child(alt);
      } else {
        possible = n.types & (JsonSchemaNode::kString | JsonSchemaNode::kNumber |
                              JsonSchemaNode::kInteger | JsonSchemaNode::kBoolean);
        bool object = true;
        for (size_t k = 0; k < n.property_nodes.size(); ++k)
          if (n.property_required[k] && !child(n.property_nodes[k])) object = false;
        if (object) possible |= n.types & JsonSchemaNode::kObject;
        if (!n.min_items || child(n.items)) possible |= n.types & JsonSchemaNode::kArray;
      }
      if (n.has_enum) {
        uint32_t enums = 0;
        for (const auto& text : n.enum_texts) enums |= xml_enum_class(minijson::parse(text).root);
        possible &= enums;
      }
      if (possible != (*classes)[i]) {
        (*classes)[i] = possible;
        changed = true;
      }
    }
    if (!changed) break;
  }
  return classes;
}
}  // namespace

struct MiniMaxToolGrammar::Data {
  const GrammarVocab* vocab;
  GrammarSpec spec;
  std::vector<std::vector<std::shared_ptr<const JsonSchema>>> schemas;
  std::vector<std::vector<std::shared_ptr<const std::vector<uint32_t>>>> representable;
  std::vector<int32_t> xml_sensitive;
};

MiniMaxToolGrammar::MiniMaxToolGrammar(const GrammarVocab* vocab, const GrammarSpec& spec) {
  auto data = std::make_shared<Data>();
  data->vocab = vocab;
  data->spec = spec;
  if (vocab) {
    for (int id = 0; id < vocab->vocab_size(); ++id) {
      const auto& text = vocab->text(id);
      if (vocab->is_eos(id) || id == vocab->markers().think_open.id ||
          id == vocab->markers().think_close.id ||
          std::any_of(text.begin(), text.end(), [](unsigned char c) {
            return c < 32 || c == ']' || c == '"' || c == '\\' || c == '[' || c == '{' ||
                   c == '}' || c == ',' || c == ':';
          }))
        data->xml_sensitive.push_back(id);
    }
  }
  for (const auto& tool : spec.tools) {
    if (tool.name.empty() || tool.name.find_first_of("<>\"\r\n") != std::string::npos)
      throw std::invalid_argument("tools.name: cannot be represented in MiniMax XML");
    std::vector<std::shared_ptr<const JsonSchema>> args;
    std::vector<std::shared_ptr<const std::vector<uint32_t>>> representable;
    for (const auto& arg : tool.args) {
      if (!xml_name(arg.key))
        throw std::invalid_argument("parameters.properties." + arg.key +
                                    ": cannot be represented in MiniMax XML");
      std::string schema = arg.schema;
      if (schema.empty() && arg.kind == GrammarArg::Kind::kFree) schema = R"({"type":"string"})";
      if (schema.empty() && arg.kind == GrammarArg::Kind::kText) {
        schema = R"({"type":"string","enum":[)";
        for (size_t i = 0; i < arg.texts.size(); ++i) {
          if (i) schema += ',';
          schema += quoted(arg.texts[i]);
        }
        schema += "]}";
      }
      std::vector<std::string> unenforced;
      auto compiled = std::make_shared<JsonSchema>(
          compile_json_schema(minijson::parse(schema).root, &unenforced));
      for (const auto& n : compiled->nodes) {
        for (const auto& text : n.enum_texts) {
          const auto value = minijson::parse(text);
          if (value.root.is_object() || value.root.is_array())
            throw std::invalid_argument("parameters.properties." + arg.key +
                                        ": container enum/const is not supported by MiniMax XML; "
                                        "use typed properties/items");
        }
        size_t nonnull = 0;
        for (int alt : n.any_of) nonnull += compiled->nodes[alt].types != JsonSchemaNode::kNull;
        if (nonnull > 1)
          throw std::invalid_argument("parameters.properties." + arg.key +
                                      ": MiniMax XML supports nullable anyOf with one non-null "
                                      "alternative; multiple alternatives are ambiguous");
        for (const auto& key : n.property_names)
          if (!xml_name(key))
            throw std::invalid_argument("parameters.properties." + arg.key +
                                        ": nested key cannot be represented in MiniMax XML");
      }
      const auto classes = xml_classes(*compiled);
      if (!(*classes)[compiled->root] &&
          std::find(tool.required_keys.begin(), tool.required_keys.end(), arg.key) !=
              tool.required_keys.end())
        throw std::invalid_argument(
            "parameters.properties." + arg.key +
            ": required value has no finite MiniMax XML spelling without null");
      representable.push_back(classes);
      args.push_back(std::move(compiled));
    }
    data->schemas.push_back(std::move(args));
    data->representable.push_back(std::move(representable));
  }
  data_ = std::move(data);
}

const JsonSchemaNode* MiniMaxToolGrammar::node(int index) const {
  return index < 0 || !schema_ ? nullptr : &schema_->nodes.at(index);
}

int MiniMaxToolGrammar::resolve_node(int index) const {
  // XML has no discriminator for a raw string versus a number, or an empty
  // container. Pick a representable alternative deterministically. The JSON
  // machine still validates it against the original complete schema.
  for (int depth = 0; depth < 128; ++depth) {
    const auto* n = node(index);
    if (!n || n->any_of.empty()) return index;
    int next = -1;
    for (int alt : n->any_of) {
      if ((*representable_)[alt]) {
        next = alt;
        break;
      }
    }
    if (next < 0) return index;
    index = next;
  }
  return JsonSchemaNode::kClosed;
}

bool MiniMaxToolGrammar::feed_json(std::string_view text) {
  for (unsigned char byte : text)
    if (!value_.feed(byte)) return false;
  return true;
}

bool MiniMaxToolGrammar::begin_value(const std::string& name, int index) {
  if (stack_.size() >= 128) return false;
  index = resolve_node(index);
  if (index == JsonSchemaNode::kClosed) return false;
  const auto* n = node(index);
  const uint32_t types = n ? (*representable_)[index] : JsonSchemaNode::kAnyClass;
  char kind = 0;
  for (const auto& [cls, opener] :
       {std::pair{JsonSchemaNode::kString, '"'}, std::pair{JsonSchemaNode::kObject, '{'},
        std::pair{JsonSchemaNode::kArray, '['}}) {
    if (!(types & cls)) continue;
    auto candidate = value_;
    if (candidate.feed(static_cast<unsigned char>(opener))) {
      value_ = std::move(candidate);
      kind = opener;
      break;
    }
  }
  if (!kind &&
      (types & (JsonSchemaNode::kNumber | JsonSchemaNode::kInteger | JsonSchemaNode::kBoolean)))
    kind = 'v';
  if (!kind) return false;
  if (kind == 'v') scalar_ = JsonLexer{};
  stack_.push_back({name, index, kind, 0, {}});
  stage_ = kind == '{' || kind == '[' ? Stage::kContainer : Stage::kScalar;
  return true;
}

int MiniMaxToolGrammar::child_node(const std::string& key) const {
  const auto* n = node(stack_.back().node);
  if (!n) return JsonSchemaNode::kAny;
  if (stack_.back().kind == '[') return n->items;
  for (size_t i = 0; i < n->property_names.size(); ++i)
    if (n->property_names[i] == key) return n->property_nodes[i];
  return n->additional;
}

bool MiniMaxToolGrammar::property(const std::string& key) {
  if (!xml_name(key)) return false;
  auto& parent = stack_.back();
  if (std::find(parent.keys.begin(), parent.keys.end(), key) != parent.keys.end()) return false;
  if (stack_.size() == 1) {
    const auto& tool = data_->spec.tools[tool_];
    if (tool.constrain_keys &&
        std::find(tool.keys.begin(), tool.keys.end(), key) == tool.keys.end())
      return false;
    schema_.reset();
    for (size_t i = 0; i < tool.args.size(); ++i)
      if (tool.args[i].key == key) {
        schema_ = data_->schemas[tool_][i];
        representable_ = data_->representable[tool_][i];
      }
    if (!schema_) {
      static const auto any = std::make_shared<const JsonSchema>(
          compile_json_schema(minijson::parse(R"({"type":"string"})").root));
      schema_ = any;
      static const auto classes = xml_classes(*any);
      representable_ = classes;
    }
    value_ = JsonMachine(schema_, &data_->vocab->json_tables());
    parent.keys.push_back(key);
    ++parent.count;
    return begin_value(key, schema_->root);
  }
  const int child = child_node(key);
  if (child == JsonSchemaNode::kClosed) return false;
  if (parent.count && !feed_json(",")) return false;
  if (!feed_json(quoted(key) + ":")) return false;
  parent.keys.push_back(key);
  ++parent.count;
  return begin_value(key, child);
}

bool MiniMaxToolGrammar::can_close() const {
  MiniMaxToolGrammar copy = *this;
  return copy.close_value();
}

bool MiniMaxToolGrammar::close_value() {
  if (stack_.empty()) return false;
  const auto frame = stack_.back();
  if (stack_.size() == 1) {
    const auto& tool = data_->spec.tools[tool_];
    for (const auto& key : tool.required_keys)
      if (std::find(frame.keys.begin(), frame.keys.end(), key) == frame.keys.end()) return false;
    ++calls_;
    stack_.clear();
    stage_ = Stage::kInvoke;
    return true;
  }
  const char end = frame.kind == '{'   ? '}'
                   : frame.kind == '[' ? ']'
                   : frame.kind == '"' ? '"'
                                       : ' ';
  if (frame.kind == 'v' && (!scalar_.feed(' ').ok || !scalar_.done())) return false;
  if (!feed_json(std::string(1, end))) return false;
  if (stack_.size() == 2 && !value_.done()) return false;
  stack_.pop_back();
  stage_ = Stage::kContainer;
  return true;
}

std::vector<MiniMaxToolGrammar::Choice> MiniMaxToolGrammar::choices() const {
  std::vector<Choice> out;
  if (stage_ == Stage::kInvoke) {
    if (calls_) out.push_back({tag("/tool_call"), Action::kCloseBlock});
    if (!calls_ || (data_->spec.parallel && data_->spec.mode != GrammarSpec::Mode::kNamed)) {
      for (size_t i = 0; i < data_->spec.tools.size(); ++i)
        if (data_->spec.mode != GrammarSpec::Mode::kNamed ||
            data_->spec.tools[i].name == data_->spec.named)
          out.push_back({tag("invoke name=\"" + data_->spec.tools[i].name + "\""), Action::kInvoke,
                         static_cast<int>(i)});
    }
  } else if (stage_ == Stage::kContainer) {
    const auto& f = stack_.back();
    if (can_close()) out.push_back({tag("/" + f.tag), Action::kClose});
    if (f.kind == '[') {
      auto copy = *this;
      if (copy.act({tag("item"), Action::kItem})) out.push_back({tag("item"), Action::kItem});
    } else {
      const auto* n = stack_.size() == 1 ? nullptr : node(f.node);
      const auto& tool = data_->spec.tools[tool_];
      const auto& keys = stack_.size() == 1 ? tool.keys
                         : n                ? n->property_names
                                            : std::vector<std::string>{};
      for (const auto& key : keys) {
        if (std::find(f.keys.begin(), f.keys.end(), key) != f.keys.end()) continue;
        auto copy = *this;
        if (copy.property(key)) out.push_back({tag(key), Action::kProperty});
      }
      if ((stack_.size() == 1 && !tool.constrain_keys) ||
          (stack_.size() > 1 && (!n || n->additional != JsonSchemaNode::kClosed)))
        out.push_back({kNamespace + "<", Action::kFreeKey});
    }
  }
  return out;
}

bool MiniMaxToolGrammar::act(const Choice& choice) {
  switch (choice.action) {
    case Action::kOpen:
      stage_ = Stage::kInvoke;
      return true;
    case Action::kInvoke:
      tool_ = choice.index;
      stack_ = {{"invoke", JsonSchemaNode::kAny, '{', 0, {}}};
      stage_ = Stage::kContainer;
      return true;
    case Action::kCloseBlock:
      stage_ = Stage::kEnd;
      return true;
    case Action::kProperty:
      return property(
          choice.text.substr(kNamespace.size() + 1, choice.text.size() - kNamespace.size() - 2));
    case Action::kItem: {
      const int child = child_node("item");
      if (stack_.back().count && !feed_json(",")) return false;
      ++stack_.back().count;
      return begin_value("item", child);
    }
    case Action::kClose:
      return close_value();
    case Action::kFreeKey:
      stage_ = Stage::kFreeKey;
      return true;
  }
  return false;
}

bool MiniMaxToolGrammar::scalar_byte(unsigned char byte) {
  if (stack_.back().kind != '"') return scalar_.feed(byte).ok && value_.feed(byte);
  if (byte == '"' || byte == '\\') return feed_json(std::string("\\") + static_cast<char>(byte));
  if (byte < 32) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string escaped = "\\u00";
    escaped += hex[byte >> 4];
    escaped += hex[byte & 15];
    return feed_json(escaped);
  }
  return value_.feed(byte);
}

bool MiniMaxToolGrammar::feed(char byte) {
  if (stage_ == Stage::kEnd) return JsonLexer::is_ws(static_cast<unsigned char>(byte));
  if (stage_ == Stage::kTop) {
    const auto open = tag("tool_call");
    pending_ += byte;
    if (data_->spec.mode == GrammarSpec::Mode::kJsonOrTools) {
      if (pending_.size() == 1 && JsonLexer::is_ws(static_cast<unsigned char>(byte))) {
        pending_.clear();
        return true;
      }
      if (!open.starts_with(pending_)) return false;
    } else {
      while (!open.starts_with(pending_)) pending_.erase(0, 1);
    }
    if (pending_ == open) {
      if (data_->spec.mode == GrammarSpec::Mode::kForbidCalls) return false;
      pending_.clear();
      return act({open, Action::kOpen});
    }
    return true;
  }
  if (stage_ == Stage::kScalar) {
    const auto close = tag("/" + stack_.back().tag);
    pending_ += byte;
    if (close.starts_with(pending_)) {
      if (pending_ == close) {
        pending_.clear();
        return close_value();
      }
      if (can_close()) return true;
      auto copy = *this;
      for (unsigned char b : pending_)
        if (!copy.scalar_byte(b)) return false;
      return true;
    }
    // A namespace introduces structure. A different tag cannot occur in
    // a scalar: the parser would correctly reject that block.
    if (pending_.starts_with(kNamespace + "<")) return false;
    const auto text = std::move(pending_);
    pending_.clear();
    for (unsigned char b : text)
      if (!scalar_byte(b)) return false;
    return true;
  }
  if (stage_ == Stage::kFreeKey) {
    if (byte == '>') {
      const auto key = std::move(pending_);
      pending_.clear();
      return property(key);
    }
    pending_ += byte;
    return xml_name(pending_);
  }
  if (pending_.empty() && JsonLexer::is_ws(static_cast<unsigned char>(byte))) return true;
  pending_ += byte;
  bool possible = false;
  for (const auto& choice : choices()) {
    if (!choice.text.starts_with(pending_)) continue;
    possible = true;
    if (choice.text == pending_) {
      // A free key shares the namespace prefix with named keys and the
      // closer. Keep matching those until the next byte disambiguates.
      if (choice.action == Action::kFreeKey) continue;
      pending_.clear();
      return act(choice);
    }
  }
  if (possible) return true;
  const auto prefix = kNamespace + "<";
  for (const auto& choice : choices())
    if (choice.action == Action::kFreeKey && pending_.starts_with(prefix)) {
      pending_.erase(0, prefix.size());
      stage_ = Stage::kFreeKey;
      if (pending_.ends_with('>')) {
        pending_.pop_back();
        const auto key = std::move(pending_);
        pending_.clear();
        return property(key);
      }
      return xml_name(pending_);
    }
  return false;
}

bool MiniMaxToolGrammar::allows(int64_t id) const {
  const auto& vocab = *data_->vocab;
  if (vocab.is_eos(id)) {
    if (stage_ == Stage::kEnd) return true;
    return stage_ == Stage::kTop && data_->spec.mode != GrammarSpec::Mode::kRequired &&
           data_->spec.mode != GrammarSpec::Mode::kNamed &&
           data_->spec.mode != GrammarSpec::Mode::kJsonOrTools;
  }
  if (id == vocab.markers().think_open.id || id == vocab.markers().think_close.id) return false;
  const auto& text = vocab.text(id);
  if (text.empty()) return false;
  MiniMaxToolGrammar copy = *this;
  for (char b : text)
    if (!copy.feed(b)) return false;
  return true;
}

void MiniMaxToolGrammar::advance(int64_t id) {
  if (data_->vocab->is_eos(id)) {
    stage_ = Stage::kEnd;
    return;
  }
  for (char b : data_->vocab->text(id))
    if (!feed(b)) throw std::logic_error("MiniMax grammar advanced a disallowed token");
}

void MiniMaxToolGrammar::mask(TokenMask* out) const {
  if (stage_ == Stage::kScalar && pending_.empty()) {
    // Most BPE tokens are ordinary scalar bytes. Reuse the JSON machine's
    // vocabulary tables for them, then correct raw escapes and XML edges.
    value_.mask(*data_->vocab, out);
    for (int id : data_->xml_sensitive) {
      const bool permit = allows(id);
      const bool current = (out->words[id >> 5] >> (id & 31)) & 1u;
      if (permit == current) continue;
      out->words[id >> 5] ^= 1u << (id & 31);
      out->allowed += permit ? 1 : -1;
    }
    if (!out->allowed) throw std::runtime_error("MiniMax tool value has no valid continuation");
    return;
  }
  out->vocab = data_->vocab->vocab_size();
  out->words.assign(TokenMask::words_for(out->vocab), 0);
  out->allowed = 0;
  const auto add = [&](int id) {
    if (allows(id)) {
      out->words[id >> 5] |= 1u << (id & 31);
      ++out->allowed;
    }
  };
  if (stage_ == Stage::kInvoke || stage_ == Stage::kContainer || stage_ == Stage::kEnd) {
    bool first[256] = {};
    bool free_key = false;
    for (const auto& choice : choices()) {
      if (choice.text.starts_with(pending_) && choice.text.size() > pending_.size())
        first[static_cast<unsigned char>(choice.text[pending_.size()])] = true;
      free_key |= choice.action == Action::kFreeKey && pending_.starts_with(choice.text);
    }
    if (!free_key) {
      if (pending_.empty())
        for (unsigned char c : {' ', '\t', '\n', '\r'}) first[c] = true;
      for (int c = 0; c < 256; ++c)
        if (first[c])
          for (int id : data_->vocab->ids_starting_with(static_cast<unsigned char>(c))) add(id);
      if (stage_ == Stage::kEnd)
        for (auto id : data_->vocab->eos_ids()) {
          if (id >= 0 && id < out->vocab && !(out->words[id >> 5] & (1u << (id & 31)))) add(id);
        }
      if (!out->allowed) throw std::runtime_error("MiniMax XML tag has no valid continuation");
      return;
    }
  }
  const bool free_top = stage_ == Stage::kTop && pending_.empty() &&
                        data_->spec.mode != GrammarSpec::Mode::kJsonOrTools;
  for (int id = 0; id < out->vocab; ++id) {
    const auto& text = data_->vocab->text(id);
    const bool ordinary = free_top && !text.empty() && text.find(']') == std::string::npos &&
                          id != data_->vocab->markers().think_open.id &&
                          id != data_->vocab->markers().think_close.id && !data_->vocab->is_eos(id);
    if (ordinary || allows(id)) {
      out->words[id >> 5] |= 1u << (id & 31);
      ++out->allowed;
    }
  }
  if (!out->allowed) throw std::runtime_error("MiniMax tool grammar has no valid continuation");
}

const char* MiniMaxToolGrammar::state_name() const {
  switch (stage_) {
    case Stage::kTop:
      return "minimax-content";
    case Stage::kInvoke:
      return "minimax-invoke";
    case Stage::kContainer:
      return "minimax-container";
    case Stage::kScalar:
      return "minimax-value";
    case Stage::kFreeKey:
      return "minimax-key";
    case Stage::kEnd:
      return "minimax-end";
  }
  return "minimax";
}
}  // namespace dgpp::text
