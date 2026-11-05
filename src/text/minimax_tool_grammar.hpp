#pragma once

#include <memory>
#include <string>
#include <vector>

#include "text/json_grammar.hpp"

namespace dgpp::text {
class GrammarVocab;
struct GrammarSpec;
struct TokenMask;

// Recursive MiniMax XML is translated incrementally into the existing JSON
// schema machine. Copies own their cursors, while sharing compiled schemas.
class MiniMaxToolGrammar {
 public:
  MiniMaxToolGrammar(const GrammarVocab* vocab, const GrammarSpec& spec);
  bool allows(int64_t id) const;
  void advance(int64_t id);
  void mask(TokenMask* out) const;
  bool started() const { return stage_ != Stage::kTop || !pending_.empty(); }
  int calls() const { return calls_; }
  const char* state_name() const;

 private:
  struct Data;
  enum class Stage { kTop, kInvoke, kContainer, kScalar, kFreeKey, kEnd };
  enum class Action { kOpen, kInvoke, kCloseBlock, kProperty, kItem, kClose, kFreeKey };
  struct Choice {
    std::string text;
    Action action;
    int index = -1;
  };
  struct Frame {
    std::string tag;
    int node = JsonSchemaNode::kAny;
    char kind = 0;  // '{', '[', '"' or 'v' (unquoted scalar)
    int count = 0;
    std::vector<std::string> keys;
  };
  std::vector<Choice> choices() const;
  bool feed(char byte);
  bool feed_json(std::string_view bytes);
  bool scalar_byte(unsigned char byte);
  bool act(const Choice& choice);
  bool property(const std::string& key);
  bool begin_value(const std::string& tag, int node);
  bool close_value();
  bool can_close() const;
  int child_node(const std::string& key) const;
  int resolve_node(int node) const;
  const JsonSchemaNode* node(int index) const;

  std::shared_ptr<const Data> data_;
  Stage stage_ = Stage::kTop;
  std::string pending_;
  int tool_ = -1;
  int calls_ = 0;
  std::vector<Frame> stack_;
  std::shared_ptr<const JsonSchema> schema_;
  std::shared_ptr<const std::vector<uint32_t>> representable_;
  JsonMachine value_;
  JsonLexer scalar_;  // one raw scalar, excluding its enclosing JSON container
};
}  // namespace dgpp::text
