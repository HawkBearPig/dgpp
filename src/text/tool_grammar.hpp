#pragma once
// Constrained decoding for the tool-call surface (M6 6g, DESIGN §10/§11):
// the grammar of the template's tool-call format as a state machine over
// TOKEN IDS that yields, at every position, the set of ids the model may
// emit next — the mask the sampler applies before the pick on every rank.
// This is how `tool_choice: required`, a named function, `none` and
// `parallel_tool_calls: false` become guarantees rather than hints: a
// sampled token is always inside the mask, so the turn is a valid call to
// an allowed function (or none, or exactly one), whatever the model would
// have liked to write.
//
// THE GRAMMAR (the format glm_tool_parser.hpp parses):
//   turn     := think? body
//   think    := <generated reasoning, any ids but EOS> </think>
//   body     := required: call+ EOS | named: call(named) EOS
//             | auto/single: free* (call EOS)? | forbid: free* (no <tool_call>)
//   call     := <tool_call> NAME (<arg_key> KEY </arg_key> <arg_value> VALUE
//               </arg_value>)* </tool_call>
//   NAME     := one of the request's tool names (a byte automaton over the
//               vocabulary's token texts, so any tokenization of the name
//               is accepted and nothing else)
//   KEY      := a declared property name unless additionalProperties is
//               explicitly true; free text when no properties are declared
//   VALUE    := typed by the property's schema (M6 6i): a JSON text under
//               the 6h machine for a JSON-typed property, one of the enum
//               texts for an enum string, free text otherwise; the closing
//               marker is the only marker allowed
//   EOS      := <|observation|> — the id that ends a tool-call turn
// Free text means every id except the structural markers and, while a
// call obligation is unmet, the EOS ids. Thinking is left free: the
// grammar only forbids ending the turn before its obligation is met.
//
// EXACTNESS. The mask is a pure function of (spec, the ids committed so
// far, the tokenizer's token texts), so every rank computes the same mask
// from the same journal record — the fabric's identical-rank-order rule
// holds with no new collective. What a masked id means to the sampler is
// defined in glm_sampler.hpp: an ABSENT candidate (logit -inf, never
// listed, zero mass, the normalizer over the allowed set) — the
// distribution restricted to the mask and renormalized, exactly.
//
// Keys and required properties: keys are constrained to the schema's
// property names whenever it declares properties and does not opt out with
// an explicit `additionalProperties: true` (JSON Schema's open default is a
// validation semantic; as a decoding grammar an open name slot invites an
// undeclared or a repeated name), and a declared key is offered at most
// once per call (a duplicate key is never a valid object); under
// `strict: true` the call cannot close while a required key is missing
// (OpenAI's strict guarantee — every required
// property present, each typed by its schema). A non-strict tool's
// free-text values stay the parser's schema typing and the client's
// validation. MiMo compact calls also retain required keys for non-strict
// schemas, avoiding empty calls from its unconstrained XML dialect.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "text/json_grammar.hpp"
#include "text/tool_parser.hpp"

namespace dgpp::text {
class Tokenizer;
}

namespace dgpp::text {

// One declared argument's value constraint (M6 6i): what the schema's
// `type` lets the grammar enforce inside <arg_value>...</arg_value>. The
// template writes a string argument RAW and every other value through
// tojson, so a JSON-typed property (integer, number, boolean, null,
// object, array, or a type list without string) is a JSON text the 6h
// machine can enforce under the property's own schema; a string property
// with an enum is one of its texts exactly; a plain string, a type list
// with string in it, an untyped property, and an unknown key are free
// text (the parser types them afterwards).
struct GrammarArg {
  enum class Kind : int { kFree = 0, kJson, kText };
  std::string key;
  Kind kind = Kind::kFree;
  std::string schema;              // kJson: the property's schema JSON text
  std::vector<std::string> texts;  // kText: the exact values allowed
  bool operator==(const GrammarArg& o) const {
    return key == o.key && kind == o.kind && schema == o.schema && texts == o.texts;
  }
};

// One tool the grammar may name, its closed key set (when it has one) and
// its typed arguments.
struct GrammarTool {
  std::string name;
  bool constrain_keys = false;     // the declared property names are the keys
  std::vector<std::string> keys;   // the property names, when closed
  std::vector<GrammarArg> args;    // the declared properties' value constraints
  // `parameters.required` (the names among the declared properties) and
  // the definition's `strict` flag: a strict call cannot close while a
  // required key is missing.
  std::vector<std::string> required_keys;
  bool strict = false;
};

// Derives a tool's grammar entry from its OpenAI function definition
// ({name, parameters, strict?} — the `function` object, or the flat form):
// the key set closes to the declared property names unless the schema opts
// out with an explicit `additionalProperties: true` (a note; the names are
// then free text), and every declared property gets its GrammarArg. Under
// `strict: true` every property's schema must lie inside the constrained
// subset, else std::invalid_argument whose message starts with the
// offending path
// ("parameters.properties.city.pattern: ..."). Otherwise a
// JSON-typed property keeps its type under the keywords that merely narrow
// a value (minimum, maxLength, pattern, format, ...) — each is a line in
// `notes`, "not enforced" — and only a schema outside the subset in shape
// ($ref, oneOf, ...) leaves the value free, with a line in `warnings`.
// Either vector may be null.
GrammarTool grammar_tool_from_function(const minijson::Value& def,
                                       std::vector<std::string>* warnings,
                                       std::vector<std::string>* notes = nullptr);

// The request's constraint — what rides the journal (fabric_serve.cpp) and
// reaches every rank's engine through SchedulerEngine::configure_constraint.
struct GrammarSpec {
  enum class Mode : int {
    kNone = 0,        // unconstrained
    kForbidCalls,     // tool_choice none: <tool_call> never
    kAuto,            // tool_choice auto: free text and calls at will (as
                      // many as `parallel` allows: one when false), every
                      // call well-formed with typed arguments
    kRequired,        // one or more calls (parallel) or exactly one
    kNamed,           // exactly one call to `named`
    kJson,            // response_format (M6 6h): the content is one JSON
                      // text under `json_schema` ("" = json_object: any
                      // object), then EOS; no tool calls
    kJsonOrTools,     // auto tools + response_format: choose a JSON answer
                      // or a tool-call turn, committing at the first value
  };
  Mode mode = Mode::kNone;
  bool parallel = true;            // several calls per turn allowed
  std::string named;               // kNamed's function
  std::vector<GrammarTool> tools;  // the callable functions (kRequired/kAuto:
                                   // all; kNamed: the one)
  std::string json_schema;         // JSON modes: schema text ("" = json_object)
  bool active() const { return mode != Mode::kNone; }
  bool has_json() const { return mode == Mode::kJson || mode == Mode::kJsonOrTools; }
  bool operator==(const GrammarSpec& o) const {
    if (mode != o.mode || parallel != o.parallel || named != o.named ||
        json_schema != o.json_schema || tools.size() != o.tools.size())
      return false;
    for (size_t i = 0; i < tools.size(); ++i)
      if (tools[i].name != o.tools[i].name ||
          tools[i].constrain_keys != o.tools[i].constrain_keys ||
          tools[i].keys != o.tools[i].keys || tools[i].args != o.tools[i].args ||
          tools[i].required_keys != o.tools[i].required_keys ||
          tools[i].strict != o.tools[i].strict)
        return false;
    return true;
  }
};

// The vocabulary as the grammar sees it: every id's text (bytes), indexed by
// first byte for the name/key automaton; the markers and EOS ids. Built once
// per process from the tokenizer (every rank has the same tokenizer.json,
// so every rank builds the same table); shared by all requests.
class GrammarVocab {
 public:
  // `texts[id]` is id's decoded bytes ("" for ids that decode to nothing —
  // the special tokens — and for ids beyond the table); `vocab_size` is
  // the lm head's (padded) vocabulary the masks cover; `call_turn_eos` the
  // id that ends a tool-call turn (<|observation|>; -1 = the first EOS id).
  GrammarVocab(std::vector<std::string> texts, ChatMarkers markers,
               std::vector<int64_t> eos_ids, int vocab_size,
               int64_t call_turn_eos = -1);
  // From the tokenizer: decode(id) of every id up to its max_id. The
  // markers are ChatMarkers::from_tokenizer's with `dsml_dialect` stated —
  // the DSML spelling is the family's, not the tokenizer's (a DeepSeek-V4
  // checkpoint passes DsmlDialect::kV4; it must agree with the frontend's
  // markers, which the parser reads).
  // dsml_dialect and model_opens_thinking are the family's declarations
  // (ChatMarkers): the tokenizer cannot tell the two DeepSeek families
  // apart, and whether the model opens its own <think> is the family's
  // statement, not a prompt's.
  static GrammarVocab from_tokenizer(const Tokenizer& tok,
                                     const std::vector<int64_t>& eos_ids,
                                     int vocab_size,
                                     DsmlDialect dsml_dialect = DsmlDialect::kV41,
                                     bool model_opens_thinking = false);

  int vocab_size() const { return vocab_size_; }
  const ChatMarkers& markers() const { return markers_; }
  const std::vector<int64_t>& eos_ids() const { return eos_; }
  // The turn-ending id the grammar forces after the last call:
  // <|observation|> when the tokenizer has it, else the first EOS id.
  int64_t call_turn_eos() const { return call_eos_; }
  const std::string& text(int64_t id) const {
    static const std::string empty;
    return id >= 0 && id < static_cast<int64_t>(texts_.size())
               ? texts_[static_cast<size_t>(id)]
               : empty;
  }
  const std::vector<int32_t>& ids_starting_with(unsigned char b) const {
    return by_first_[b];
  }
  // The structural ids free text may never contain: the eight markers.
  const std::vector<int64_t>& marker_ids() const { return marker_ids_; }
  const std::vector<int32_t>& xml_end_ids() const { return xml_end_ids_; }
  bool is_eos(int64_t id) const;
  bool usable() const { return markers_.tool_calls_available() && !eos_.empty(); }
  // The JSON grammar's per-vocabulary tables (M6 6h), built once on first
  // use — or eagerly here, so a serving rank pays the second or so at
  // boot rather than on the first json request. Copies share them.
  void prepare_json() const;
  const JsonTables& json_tables() const;

 private:
  struct JsonHolder;
  mutable std::shared_ptr<JsonHolder> json_;
  std::vector<std::string> texts_;
  std::vector<int32_t> by_first_[256];
  std::vector<int32_t> xml_end_ids_;
  ChatMarkers markers_;
  std::vector<int64_t> marker_ids_;
  std::vector<int64_t> eos_;
  int64_t call_eos_ = -1;
  int vocab_size_ = 0;
};

// One position's mask over [0, vocab_size): bit id set = allowed. `allowed`
// counts the set bits (0 means "unconstrained — the words are not
// meaningful"); the sampler treats a constrained row's allowed count as its
// vocabulary size.
struct TokenMask {
  std::vector<uint32_t> words;
  int vocab = 0;
  int allowed = 0;
  static int words_for(int vocab) { return (vocab + 31) / 32; }
  bool constrained() const { return allowed > 0; }
  bool allows(int64_t id) const {
    if (!constrained()) return true;
    if (id < 0 || id >= vocab) return false;
    return (words[static_cast<size_t>(id >> 5)] >> (id & 31)) & 1u;
  }
};

class GrammarState {
 public:
  // prompt_opens_thinking: the prompt ended in <think> (the reasoning is
  // free until </think>). model_may_open_thinking: the prompt left the
  // choice to the model (ChatMarkers::prompt_leaves_thinking_to_model) —
  // a <think> as the FIRST id opens the same free reasoning; any other
  // first id settles the grammar's opening state as before.
  GrammarState(const GrammarVocab* vocab, GrammarSpec spec,
               bool prompt_opens_thinking, bool model_may_open_thinking = false);

  bool active() const { return vocab_ != nullptr && spec_.active() && !dead_; }
  const GrammarSpec& spec() const { return spec_; }

  // The mask for the next position. An unconstrained position (a free
  // state with nothing forbidden, or an inactive grammar) leaves
  // out->allowed == 0.
  void mask(TokenMask* out) const;
  // Pointwise: would `id` be allowed next? (The same rule as mask().)
  bool allows(int64_t id) const;
  // The id was committed. A disallowed id (which the sampler never
  // produces; the MTP draft may propose one, to be rejected) kills the
  // grammar: every later position is free, so a row the step will discard
  // never carries an empty mask.
  void advance(int64_t id);

  // For the record: the state's name.
  const char* state_name() const;

  // The DSML tag's stand-in byte inside a target (never a token's text).
  static constexpr char kDsmlSentinel = '\x01';

 private:
  enum class State {
    kThink,       // before </think>
    kTop,         // between calls (or before the first)
    kName,        // inside a call: the function name (complete when a
                  // target matches; <arg_key> / </tool_call> then close it)
    kKey,         // an argument key
    kAfterKey,    // <arg_value>
    kValue,       // an argument value
    kAfterValue,  // <arg_key> or </tool_call>
    kEnd,         // the turn must end: EOS
    kDone,        // EOS emitted: nothing more (the scheduler retires)
    kJsonBody,    // kJson: the JSON text (JsonMachine), then EOS
    // The Qwen3.8 XML format: the block's structure is TEXT
    // between the <tool_call> ids, so every structural piece is a target
    // of the text automaton with the surrounding literals folded in
    // ("\n<function=NAME>\n", "<parameter=KEY>\n", "\n</parameter>\n",
    // "</function>\n") — any tokenization of the literals is accepted.
    kQName,        // "\n<function=" NAME ">\n"
    kQKeyOrClose,  // "<parameter=" (KEY ">\n" when the keys are closed) | "</function>\n"
    kQFreeKey,     // an open key set: free text through ">\n"
    kQValue,       // the typed value, then "\n</parameter>\n"
    kQClose,       // </tool_call>
    // The DeepSeek DSML format (the V4.1 spelling shown; under
    // DsmlDialect::kV4 the tag names are "tool_calls" / "invoke" /
    // "parameter" with no space after the tag): one marker id (the ｜DSML｜ tag token)
    // inside text tags — "<" TAG " calls>\n" opens the block, each call
    // is "<" TAG " invoke name=\"NAME\">\n" ... "</" TAG " invoke>\n" with
    // parameters "<" TAG " parameter name=\"K\" string=\"true|false\">" V
    // "</" TAG " parameter>\n", and "</" TAG " calls>" closes it; the turn
    // ends then. The tag rides the text automaton as a sentinel byte the
    // vocabulary never spells (match_ids offers the marker id there); at
    // the top the tag is offered right after a "<" only.
    kDCalls,          // " calls>\n" after the opening tag
    kDInvoke,         // "<" TAG " invoke name=\"" NAME "\">\n"
    kDParamOrClose,   // "<" TAG " parameter name=\"" (KEY "\" string=\"" when closed) | "</" TAG " invoke>\n"
    kDFreeKey,        // an open key set: free text through "\" string=\""
    kDFlag,           // "true\">" | "false\">" (the value's kind decides)
    kDValue,          // the typed value, then "</" TAG " parameter>\n"
    kDInvokeOrClose,  // another invoke | "</" TAG " calls>"
  };
  // The automaton over token texts: the targets still consistent with the
  // bytes emitted so far, and those bytes.
  struct TextMatch {
    std::vector<std::string> targets;
    std::string emitted;
    bool complete() const {
      for (const std::string& t : targets)
        if (t == emitted) return true;
      return false;
    }
  };
  void enter(State s);
  bool obligation_open() const;  // a call is still required
  bool calls_remaining() const;  // another call may open
  const GrammarTool* current_tool() const;
  // Whether a key may open: the open tool's key set is open, or closed
  // with a key this call has not used yet (a closed key is offered once).
  bool keys_possible() const;
  bool keys_possible_for(const std::string& name) const;
  // Whether the call may close: not while a strict tool's required key is
  // missing.
  bool call_closable() const;
  bool call_closable_for(const std::string& name) const;
  bool key_used(const std::string& key) const;
  // The free-text mask: everything but the markers (when forbidden) and
  // EOS while the obligation is open; `extra_allowed` reopens one marker.
  void free_mask(TokenMask* out, int64_t extra_allowed,
                 bool forbid_markers = true) const;
  // Inside a Qwen call: everything but the markers, the think markers and
  // every EOS id (a call never ends the turn mid-block).
  void free_mask_in_call(TokenMask* out) const;
  const char* xml_function_open() const;
  const char* xml_header_end() const;
  const char* xml_parameter_end() const;
  const char* xml_function_end() const;
  bool compact_xml_text_state() const;
  bool compact_xml_text(const std::string& text);
  void compact_xml_mask(TokenMask* out) const;
  bool qwen() const;
  bool dsml() const;
  // The ids that continue `target` from `emitted`.
  std::vector<int64_t> literal_ids(const std::string& target, const std::string& emitted) const;
  void list_mask(TokenMask* out, const std::vector<int64_t>& ids) const;
  // A JSON machine's position: its mask without the markers, plus the
  // closer once the text is complete (`closer` -1: the EOS ids).
  void json_mask(const JsonMachine& machine, int64_t closer, TokenMask* out) const;
  bool json_allows(const JsonMachine& machine, int64_t closer, int64_t id) const;
  const GrammarArg* current_arg() const;
  // The ids that continue the match (and the closer when a target is
  // complete).
  std::vector<int64_t> match_ids(const TextMatch& m, int64_t closer) const;

  const GrammarVocab* vocab_ = nullptr;
  GrammarSpec spec_;
  State state_ = State::kTop;
  bool dead_ = false;
  int calls_ = 0;          // calls closed so far
  int tool_ = -1;          // the open call's tool (index into spec_.tools)
  TextMatch match_;        // kName / kKey / a kText value
  std::string key_;        // the open argument's key (kAfterKey / kValue)
  std::string term_;       // kQValue / kDValue (JSON, or a free DSML value past its "</" tag): the terminator emitted so far
  bool top_lt_ = false;    // DSML top: the last committed text ended in "<" (the tag may follow)
  bool think_openable_ = false;  // the first id may be <think> (model_may_open_thinking, until the first advance)
  bool flag_string_ = true;  // DSML: the open parameter's string="true" (a raw value) or "false" (JSON)
  std::vector<std::string> used_keys_;  // the open call's keys so far
  int arg_ = -1;           // the open argument (index into the tool's args)
  JsonMachine json_;       // kJson: the body's machine (inactive otherwise)
  JsonMachine value_json_; // a kJson argument's machine
  // The compiled schemas of every kJson argument, [tool][arg] (null
  // where the argument is not kJson).
  std::vector<std::vector<std::shared_ptr<const JsonSchema>>> arg_schemas_;
};

}  // namespace dgpp::text
