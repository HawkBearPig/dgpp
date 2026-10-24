#pragma once
// The GLM tool-call and reasoning parser (M6 6f, DESIGN §11): a state
// machine over TOKEN IDS that splits one assistant turn's generated ids
// into reasoning text, content text and structured tool calls — the
// inverse of what the chat template renders for an assistant message.
//
// THE FORMAT IS THE TEMPLATE'S. The template renders an assistant turn as
//   <think>{reasoning}</think>{content}<tool_call>{name}<arg_key>{k}
//   </arg_key><arg_value>{v}</arg_value>...</tool_call>...
// and the generation prompt ends in "<think>", so the model's own output
// is the tail of that shape. The markers are ADDED TOKENS of the
// tokenizer (<think> 154841, </think> 154842, <tool_call> 154843,
// </tool_call> 154844, <arg_key>/</arg_key> 154847/8,
// <arg_value>/</arg_value> 154849/50 for the pinned revision): one id
// each, never produced by BPE for ordinary text, so keying on the ids is
// exact where keying on decoded text would be ambiguous. They are not
// "special" in the tokenizer's sense — decode() prints them literally —
// which is why a malformed block falls back to its literal text below.
//
// VALUES. The template writes a string argument raw and every other
// value through tojson; the inverse is lossy on its own ("123" and 123
// render alike), so the parser types a value from the request's tool
// schema when it names the key (parameters.properties[key].type: a
// string type keeps the text; a JSON type parses it, the text standing
// when the parse fails) and falls back to "JSON if it parses, else a
// string" for keys the schema does not know — vLLM's GLM parser makes the
// same call. Arguments are re-serialized in Python json.dumps form
// (`{"city": "Paris", "days": 3}`), the template's own tojson dialect.
//
// MALFORMED BLOCKS. A block is a call only when </tool_call> closes a
// consistent name / (key, value)* sequence with distinct argument keys.
// Anything else — a repeated key, a value without a key, a stray marker,
// a second <tool_call> before the first
// closed, the stream ending inside a block — flushes the block's literal
// text (the decode of its ids, markers included) as CONTENT, so nothing
// the model produced is silently lost and no half-parsed call reaches a
// client. A <tool_call> nested in an open block starts a fresh block.
//
// STREAMING. Reasoning and content ids stream as suffix-diff deltas of
// their run's full decode (the service's text discipline); tool-call ids
// buffer until the block completes or aborts. Events come out in arrival
// order. The parser is pure host code with no tokenizer dependency beyond
// the Decode callback, so the service gate drives it with a fake and the
// template gate with the real tokenizer.
//
// Threading: one parser per request, driven from one thread at a time.
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"

namespace dgpp::text {
class Tokenizer;
}

namespace dgpp::text {

// One marker: its id (-1 = the tokenizer has no such added token) and its
// literal text (what the template writes and decode() prints).
struct ChatMarker {
  int64_t id = -1;
  std::string text;
  bool available() const { return id >= 0; }
};

// The tool-call block's inner format: GLM keys every part on
// an added token (<arg_key>/<arg_value> pairs); Qwen3.8 writes
// "<function=NAME>\n<parameter=K>\nV\n</parameter>\n...</function>\n" as
// TEXT between the <tool_call> tokens — only the outer markers are single
// ids, so the block is parsed from its decoded text when it closes;
// DeepSeek-V4.1 writes DSML — "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke
// name=\"NAME\">\n<｜DSML｜ parameter name=\"K\" string=\"true|false\">V</｜DSML｜
// parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>" — where only the ｜DSML｜
// tag token is an id (a special one: the service's decode skips it) and
// every bracket and tag name is text; the block opens at the tag token
// after a "<" and closes at the text "</｜DSML｜ calls>", parsed then.
enum class ToolFormat { kNone, kGlmMarkers, kQwenXml, kDsml };

// The DSML dialect (kDsml): the two DeepSeek encoders share the ｜DSML｜ tag
// token and the block's grammar but not its spelling, so the tokenizer
// cannot tell them apart — the family's frontend states it.
//   kV41 (DeepSeek-V4.1, encoding.py): the tag names carry a LEADING SPACE
//     and the block is "calls" — "<｜DSML｜ calls>", "<｜DSML｜ invoke
//     name=...>", "<｜DSML｜ parameter ...>"; an invoke may name its tool
//     "namespace::name" (the parser reports the bare name).
//   kV4 (DeepSeek-V4, encoding_dsv4.py): no space and the block is
//     "tool_calls" — "<｜DSML｜tool_calls>", "<｜DSML｜invoke name=...>",
//     "<｜DSML｜parameter ...>"; no namespaces (a name is reported as written).
enum class DsmlDialect { kV41, kV4 };

struct ChatMarkers {
  ChatMarker think_open;        // "<think>"
  ChatMarker think_close;       // "</think>"
  ChatMarker tool_call_open;    // "<tool_call>"
  ChatMarker tool_call_close;   // "</tool_call>"
  ChatMarker arg_key_open;      // "<arg_key>"      (GLM)
  ChatMarker arg_key_close;     // "</arg_key>"     (GLM)
  ChatMarker arg_value_open;    // "<arg_value>"    (GLM)
  ChatMarker arg_value_close;   // "</arg_value>"   (GLM)
  ChatMarker dsml;              // "｜DSML｜"        (DeepSeek-V4.1 / V4: the tool-call tag token)
  // The DSML spelling (kDsml only). from_tokenizer leaves the default — the
  // two tokenizers carry the same tag token — so a DeepSeek-V4 frontend (and
  // the grammar vocabulary built beside it) sets kV4 explicitly.
  DsmlDialect dsml_dialect = DsmlDialect::kV41;
  // The XML format's dialect (kQwenXml): true when the template writes the
  // call without the newlines Qwen3.8's puts between its tags —
  // "<function=NAME><parameter=K>V</parameter></function>", the MiMo-V2.6
  // form, where a value's trailing newline is the value's. The parser
  // accepts both forms under either dialect; the constrained-decoding
  // grammar emits the newline form for both (tool_grammar.cpp). Set from
  // the tokenizer (the MiMo tokenizers carry the <|mimo_audio_start|>
  // marker).
  bool xml_compact = false;
  // The family's declaration (ServeFamily::model_opens_thinking, stated
  // by the family the checkpoint's architecture selects — never inferred
  // from a prompt's tokens): its template's generation prompt opens no
  // reasoning block and the model opens its own <think> (MiMo-V2.6's
  // ends in "assistant\n", Flint's at "<|assistant|>\n"). The parser then
  // splits the block the model opens, the usage counts it and the
  // constrained-decoding grammar admits it before the answer. False for
  // the families whose templates settle it (GLM's and Qwen's generation
  // prompts end inside <think> or with a closed block).
  bool model_opens_thinking = false;
  // The template's role markers (<|system|>, <|user|>, <|assistant|>,
  // <|observation|>; <|im_start|>, <|im_end|>), the ones the tokenizer
  // has: their positions in a prompt are the prefix cache's structural
  // boundaries (M7).
  std::vector<ChatMarker> role_markers;
  // The tokenizer's bare newline token, when "\n" encodes to one id: the
  // Qwen template's generation prompt ends in "<think>\n", the GLM one
  // in "<think>" — both open the reply inside the reasoning.
  ChatMarker newline;
  // The ids that are a run of newlines alone ("\n", "\n\n", ... up to four,
  // each when it encodes to one id): the whitespace a template writes
  // after a closed block (Qwen's enable_thinking=false form ends in
  // "</think>\n\n", one id for the pair) that prompt_leaves_thinking_to_model
  // steps over. Includes `newline` when it exists.
  std::vector<int64_t> newline_ids;
  bool is_newline_run(int64_t id) const {
    if (newline.available() && id == newline.id) return true;
    for (const int64_t n : newline_ids)
      if (n == id) return true;
    return false;
  }

  // Whether a rendered prompt leaves the reply inside an opened <think>
  // (its last token is think_open, or think_open followed by the bare
  // newline) — the parser then starts in the reasoning and the usage
  // counts reasoning tokens (2026-09-09: the Qwen turn's trailing newline
  // hid the opened block; the reasoning leaked into content).
  bool prompt_opens_thinking(const std::vector<int64_t>& prompt) const {
    if (!think_open.available() || prompt.empty()) return false;
    if (prompt.back() == think_open.id) return true;
    return newline.available() && prompt.size() >= 2 && prompt.back() == newline.id &&
           prompt[prompt.size() - 2] == think_open.id;
  }
  // Whether a rendered prompt leaves the choice to the model: the family
  // declares that it does (model_opens_thinking) and the template settled
  // nothing itself — the prompt's tail neither opens <think> (thinking
  // on) nor closes a block (MiMo's enable_thinking=false form ends in
  // "<think></think>", Qwen's in "</think>\n\n"), newline runs after the
  // markers aside. The declaration is the family's alone: a prompt's
  // tokens never select the rule, so every other family's parser and
  // constrained decoding keep their opening state bit for bit whatever a
  // prompt's tail (the loopback gates feed synthetic prompts).
  bool prompt_leaves_thinking_to_model(const std::vector<int64_t>& prompt) const {
    if (!model_opens_thinking || !think_open.available() || !think_close.available() || prompt.empty())
      return false;
    if (prompt_opens_thinking(prompt)) return false;
    size_t n = prompt.size();
    for (int k = 0; k < 4 && n > 0; ++k) {
      const int64_t id = prompt[n - 1];
      if (id == think_open.id || id == think_close.id) return false;
      if (!is_newline_run(id)) break;
      --n;
    }
    return true;
  }

  // The reasoning split needs </think>; tool calls need the six GLM
  // markers, the two Qwen ones, or DeepSeek's tag token.
  bool reasoning_available() const { return think_close.available(); }
  ToolFormat tool_format() const {
    if (tool_call_open.available() && tool_call_close.available() &&
        arg_key_open.available() && arg_key_close.available() &&
        arg_value_open.available() && arg_value_close.available())
      return ToolFormat::kGlmMarkers;
    if (tool_call_open.available() && tool_call_close.available())
      return ToolFormat::kQwenXml;
    if (dsml.available()) return ToolFormat::kDsml;
    return ToolFormat::kNone;
  }
  bool tool_calls_available() const { return tool_format() != ToolFormat::kNone; }
  // Whether a DSML invoke may name its tool "namespace::name" (the V4.1
  // dialect: the schema block lists the qualified name, the grammar spells
  // it and the parser reports the bare one). V4 has no namespaces.
  bool dsml_namespaces() const {
    return tool_format() == ToolFormat::kDsml && dsml_dialect == DsmlDialect::kV41;
  }

  // Looks every marker up by its text among the tokenizer's added tokens
  // (a revision without one leaves it unavailable — never guessed).
  static ChatMarkers from_tokenizer(const Tokenizer& tok);
};

// The request's tool schemas, reduced to what value typing needs: for
// every function name, each declared parameter's JSON-schema type.
class ToolSchemas {
 public:
  enum class Type { kUnknown, kString, kJson };

  ToolSchemas() = default;
  // `tools` is the OpenAI array: entries either {type: "function",
  // function: {name, parameters}} or the flat {name, parameters} form the
  // template also accepts. Entries without a usable name are skipped.
  explicit ToolSchemas(const minijson::Value& tools);

  bool has(const std::string& function) const {
    return types_.count(function) != 0;
  }
  Type type_of(const std::string& function, const std::string& key) const;

 private:
  std::map<std::string, std::map<std::string, Type>> types_;
};

class ToolCallParser {
 public:
  // ids -> bytes, special tokens skipped (the service's decode_ids).
  // Must equal the concatenation of decode({id}) for each id, including
  // incomplete UTF-8 bytes; streaming text decodes one new id at a time.
  using Decode = std::function<std::string(const std::vector<int64_t>&)>;

  struct Call {
    std::string name;
    std::string arguments;  // a JSON object, json.dumps form
    bool custom = false;   // serving adapter: arguments holds free-form input
  };

  struct Event {
    // kReasoningClosed marks the </think> id itself (no text): the fold
    // knob needs the position to reproduce the model's own transcript.
    enum class Kind { kReasoning, kReasoningClosed, kContent, kToolCall };
    Kind kind = Kind::kContent;
    std::string text;  // kReasoning / kContent: the delta
    Call call;         // kToolCall
    struct TokenSpan {
      size_t token;  // index in the generated sequence
      size_t begin, end;  // byte range in text
    };
    std::vector<TokenSpan> tokens;
  };

  struct Options {
    bool track_tokens = false;  // content logprobs need token provenance
    // The prompt ended in <think>: ids route to reasoning until </think>.
    bool start_in_reasoning = true;
    // The prompt left the choice to the model (ChatMarkers::
    // prompt_leaves_thinking_to_model): a <think> id before any content
    // opens the reasoning; otherwise the marker is content text.
    bool model_may_open_thinking = false;
    // The prompt itself ends inside a "<tool_call>" block (a prompt-side
    // forced block): the first ids are a function name. The service no
    // longer forces blocks — tool_choice is the grammar of constrained
    // decoding (M6 6g) — but the parser keeps the interface.
    bool start_in_tool_call = false;
    // The name text already in the prompt (tool_choice named a function);
    // the model may extend it.
    std::string seeded_name;
    // The forced prefix's literal text, so an aborted forced block
    // flushes as the text the model effectively produced.
    std::string forced_prefix_text;
  };

  ToolCallParser(ChatMarkers markers, Decode decode, ToolSchemas schemas,
                 Options options);

  // One generated id; appends zero or more events in order.
  void feed(int64_t id, std::vector<Event>* out);
  // End of the generation: flushes an open block as content.
  void finish(std::vector<Event>* out);

  int calls() const { return calls_; }
  bool in_tool_call() const { return state_ == State::kToolCall; }
  // Inside the reasoning (the prompt's or the model's own <think> block).
  bool in_reasoning() const { return state_ == State::kReasoning; }

 private:
  enum class State { kReasoning, kContent, kToolCall };
  enum class Sub { kName, kKey, kAfterKey, kValue, kAfterValue };

  // DSML content retains decoded bytes for its held block prefix.
  struct Run {
    std::string text;
    std::vector<Event::TokenSpan> tokens;
  };

  bool is_marker(int64_t id, const ChatMarker& m) const {
    return m.available() && id == m.id;
  }
  void run_append(int64_t id, Event::Kind kind,
                  std::vector<Event>* out);
  void enter_tool_call(int64_t opening_id);
  void abort_block(std::vector<Event>* out);
  void annotate_block(Event* ev, bool dsml) const;
  static void slice_tokens(Event* ev, const std::vector<Event::TokenSpan>& tokens,
                           size_t begin, size_t end);
  void complete_block(std::vector<Event>* out);
  // The Qwen format: the closed block's text into name_/args_ (false when
  // malformed — the caller aborts the block as content).
  bool parse_qwen_block(const std::string& text);
  // The DSML format (DeepSeek-V4.1 and V4, the tag names per
  // markers_.dsml_dialect): the content run with the block's
  // possible prefix ("\n\n<" or "<") held back until the next id decides;
  // the block's text (the tag token restored between the decoded runs);
  // the closed block into dsml_calls_ (false when malformed).
  void dsml_content_append(int64_t id, std::vector<Event>* out);
  void dsml_flush_held(std::vector<Event>* out);
  void enter_dsml_block();
  std::string dsml_block_text() const;
  bool dsml_block_closed(const std::string& text) const;
  bool parse_dsml_block(const std::string& text);
  void complete_dsml_block(std::vector<Event>* out);
  void abort_dsml_block(std::vector<Event>* out);
  std::string typed_value(const std::string& function, const std::string& key,
                          const std::string& text) const;

  ChatMarkers markers_;
  Decode decode_;
  ToolSchemas schemas_;
  Options options_;

  State state_ = State::kContent;
  Run run_;  // the current DSML content run

  // The open block.
  Sub sub_ = Sub::kName;
  std::vector<int64_t> raw_;   // every id since (and including) <tool_call>
  std::vector<size_t> raw_indices_;
  size_t next_token_ = 0;
  size_t current_token_ = 0;
  bool raw_has_prefix_ = false;  // raw_ lacks the forced prefix's ids
  bool content_started_ = false; // some content text or call was emitted
  std::string seeded_name_;
  std::vector<int64_t> name_ids_, key_ids_, value_ids_;
  std::string name_, key_;
  std::vector<std::pair<std::string, std::string>> args_;  // key, text
  // The XML block carried one JSON object instead of parameter tags (the
  // MiMo template's rendering of pre-serialized arguments): args_ holds
  // the members' JSON texts, emitted verbatim.
  bool args_json_ = false;

  // DSML: the content run's emitted length (the held prefix follows it),
  // the held prefix carried into an open block, the block's parsed calls.
  size_t dsml_emitted_ = 0;
  std::string dsml_held_;
  std::vector<Event::TokenSpan> dsml_held_tokens_;
  std::vector<Call> dsml_calls_;

  int calls_ = 0;
};

}  // namespace dgpp::text
