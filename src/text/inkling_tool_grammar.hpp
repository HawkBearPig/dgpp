#pragma once
// The constrained-decoding grammar of Inkling 2's reply format.
//
// A reply is a run of typed blocks and then the end-of-sampling id:
//
//   [<|message_model|> NAME?] <|content_thinking|> text <|end_message|>
//   [<|message_model|> NAME?] <|content_text|> text <|end_message|>
//   [<|message_model|> NAME?] <|content_invoke_tool_json|>
//       {"name": NAME, "args": {...}} <|end_message|>
//   <|content_model_end_sampling|>
//
// Every marker is one special id, so the block structure is a grammar over
// ids, and a call's body is one JSON value: the JSON schema machine runs it
// under a schema built from the request's tools, an object whose "name" is
// one of the callable names and whose "args" follow that tool's parameters.
// A call block follows a header that names its tool, as the checkpoint's
// template writes it: the name is one of the callable names and the body's
// "name" is held to it.
//
// What a mode constrains:
//   kForbidCalls   no call block
//   kAuto          calls at will (one when `parallel` is false), each well
//                  formed. The reasoning and answer text are not masked.
//   kRequired      the turn is its calls: no text block, and it cannot end
//                  before a call
//   kNamed         exactly one call, to the named tool, then the end
//   kJson          the answer is one text block holding a JSON value under
//                  the response schema, then the end
//   kJsonOrTools   that answer, or a turn of calls
// While a mode still owes its call or answer, one thinking block may come
// first. A JSON body starts at its value, with no whitespace before it.
//
// The mask is a pure function of the spec, the ids committed so far and the
// vocabulary's token texts, so every rank computes the same one.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "text/json_grammar.hpp"

namespace dgpp::text {

class GrammarVocab;
struct GrammarSpec;
struct TokenMask;

class InklingToolGrammar {
 public:
  // Throws std::invalid_argument for a spec this format cannot serve (a
  // required call without tools, a named function that is not a tool, a
  // schema that does not parse). `vocab` may be null to validate a spec.
  InklingToolGrammar(const GrammarVocab* vocab, const GrammarSpec& spec);

  // The mask for the next position. out->allowed == 0: unconstrained.
  void mask(TokenMask* out) const;
  bool allows(int64_t id) const;
  // The id was committed (allows(id) held).
  void advance(int64_t id);

  int calls() const { return calls_; }
  bool done() const { return stage_ == Stage::kDone; }
  const char* state_name() const;

 private:
  struct Data;
  enum class Stage {
    kTop,       // between blocks
    kHeader,    // after <|message_model|>: an optional tool name
    kThinking,  // a thinking block's text
    kText,      // a text block's free text
    kCall,      // a call block's JSON
    kAnswer,    // the JSON answer of a response format
    kEnd,       // the turn must end
    kDone,      // the end id was committed
  };
  bool calls_allowed() const;
  bool text_allowed() const;
  bool thinking_allowed() const;
  bool end_allowed() const;
  bool leading_whitespace(int64_t id) const;
  // The ids a block may open with at the top or after a header.
  void block_openers(std::vector<int64_t>* ids) const;
  // The text ids that continue a callable name from the header so far.
  void header_text_ids(std::vector<int64_t>* ids) const;
  int header_tool() const;  // the tool the header names in full, or -1
  bool feed_json(int64_t id);

  std::shared_ptr<const Data> data_;
  Stage stage_ = Stage::kTop;
  std::string header_;   // kHeader: the name text so far
  int header_tool_ = -1; // the tool a call block's header named (-1: none)
  int calls_ = 0;
  int thinking_blocks_ = 0;
  bool answered_ = false;
  bool json_started_ = false;  // the open JSON body has taken an id
  JsonMachine json_;     // kCall / kAnswer
};

}  // namespace dgpp::text
