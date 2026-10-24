// M6 6f: the tool-call / reasoning parser over a fake decoder (host-only,
// always runs). The real tokenizer's round trip — render(assistant
// tool_calls) → encode → parse — lives in glm_chat_template_test; here
// the state machine's contract is pinned on synthetic ids: the reasoning
// split, exact streamed deltas, schema-typed and inferred values, nested
// JSON, several calls per turn, and every malformed shape falling back to
// literal content with no half-parsed call.
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/chat_template.hpp"
#include "text/tool_parser.hpp"

namespace {

using dgpp::text::ChatMarker;
using dgpp::text::ChatMarkers;
using dgpp::text::DsmlDialect;
using dgpp::text::ToolCallParser;
using dgpp::text::ToolSchemas;
using Event = ToolCallParser::Event;
using Kind = ToolCallParser::Event::Kind;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The fake tokenizer: ids below 256 are bytes; the markers are 1001..1008
// and decode to their literal text (they are not special tokens, exactly
// like the real ones); 999 is a special EOS that decodes to nothing.
constexpr int64_t kEos = 999;
constexpr int64_t kThinkOpen = 1001, kThinkClose = 1002, kToolOpen = 1003,
                  kToolClose = 1004, kKeyOpen = 1005, kKeyClose = 1006,
                  kValueOpen = 1007, kValueClose = 1008;
// DeepSeek-V4.1's tag token: SPECIAL in its tokenizer (the service's decode
// skips it), so the fake decode drops it like the EOS.
constexpr int64_t kDsml = 1010;

ChatMarkers fake_markers() {
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  m.arg_key_open = ChatMarker{kKeyOpen, "<arg_key>"};
  m.arg_key_close = ChatMarker{kKeyClose, "</arg_key>"};
  m.arg_value_open = ChatMarker{kValueOpen, "<arg_value>"};
  m.arg_value_close = ChatMarker{kValueClose, "</arg_value>"};
  return m;
}

std::string fake_decode(const std::vector<int64_t>& ids) {
  static const std::map<int64_t, std::string> markers = {
      {kThinkOpen, "<think>"},   {kThinkClose, "</think>"},
      {kToolOpen, "<tool_call>"}, {kToolClose, "</tool_call>"},
      {kKeyOpen, "<arg_key>"},   {kKeyClose, "</arg_key>"},
      {kValueOpen, "<arg_value>"}, {kValueClose, "</arg_value>"},
  };
  std::string out;
  for (const int64_t id : ids) {
    if (id == kEos || id == kDsml) continue;
    const auto m = markers.find(id);
    if (m != markers.end())
      out += m->second;
    else if (id >= 0 && id < 256)
      out.push_back(static_cast<char>(id));
  }
  return out;
}

// A token stream from text: bytes, with the marker strings mapped to
// their ids (leftmost-longest, like the real added-token scan).
std::vector<int64_t> ids_of(const std::string& text) {
  static const std::vector<std::pair<std::string, int64_t>> table = {
      {"</tool_call>", kToolClose}, {"<tool_call>", kToolOpen},
      {"</arg_value>", kValueClose}, {"<arg_value>", kValueOpen},
      {"</arg_key>", kKeyClose},   {"<arg_key>", kKeyOpen},
      {"</think>", kThinkClose},   {"<think>", kThinkOpen},
  };
  std::vector<int64_t> out;
  for (size_t i = 0; i < text.size();) {
    bool matched = false;
    for (const auto& [s, id] : table) {
      if (text.compare(i, s.size(), s) == 0) {
        out.push_back(id);
        i += s.size();
        matched = true;
        break;
      }
    }
    if (!matched) out.push_back(static_cast<unsigned char>(text[i++]));
  }
  return out;
}

ToolSchemas weather_schemas() {
  static const std::string tools =
      R"([{"type":"function","function":{"name":"get_weather",)"
      R"("parameters":{"type":"object","properties":{)"
      R"("city":{"type":"string"},"days":{"type":"integer"},)"
      R"("code":{"type":"string"},"opts":{"type":"object"}}}}},)"
      R"({"name":"flat_tool","parameters":{"type":"object",)"
      R"("properties":{"x":{"type":"number"}}}}])";
  static const dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(tools);
  return ToolSchemas(parsed.root);
}

struct Run {
  std::string reasoning, content;
  int reasoning_closed = 0;
  std::vector<ToolCallParser::Call> calls;
  std::vector<Kind> order;
};

Run drive(const std::vector<int64_t>& ids, ToolCallParser::Options opts = {},
          ToolSchemas schemas = weather_schemas()) {
  ToolCallParser parser(fake_markers(), fake_decode, std::move(schemas), opts);
  std::vector<Event> events;
  for (const int64_t id : ids) parser.feed(id, &events);
  parser.finish(&events);
  Run run;
  for (const Event& ev : events) {
    run.order.push_back(ev.kind);
    switch (ev.kind) {
      case Kind::kReasoning: run.reasoning += ev.text; break;
      case Kind::kReasoningClosed: ++run.reasoning_closed; break;
      case Kind::kContent: run.content += ev.text; break;
      case Kind::kToolCall: run.calls.push_back(ev.call); break;
    }
  }
  require(static_cast<int>(run.calls.size()) == parser.calls(),
          "calls() counts the emitted calls");
  return run;
}

// ---- the Qwen3.8 format: only the outer markers are ids; the
// block's "<function=...><parameter=...>" structure is text, parsed when
// the block closes.
ChatMarkers qwen_markers() {
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  return m;
}
std::vector<int64_t> qwen_ids_of(const std::string& text) {
  static const std::vector<std::pair<std::string, int64_t>> table = {
      {"</tool_call>", kToolClose}, {"<tool_call>", kToolOpen},
      {"</think>", kThinkClose},   {"<think>", kThinkOpen},
  };
  std::vector<int64_t> out;
  for (size_t i = 0; i < text.size();) {
    bool matched = false;
    for (const auto& [s, id] : table) {
      if (text.compare(i, s.size(), s) == 0) {
        out.push_back(id);
        i += s.size();
        matched = true;
        break;
      }
    }
    if (!matched) out.push_back(static_cast<unsigned char>(text[i++]));
  }
  return out;
}
Run drive_qwen(const std::string& text, ToolCallParser::Options opts = {}) {
  ToolCallParser parser(qwen_markers(), fake_decode, weather_schemas(), opts);
  std::vector<Event> events;
  for (const int64_t id : qwen_ids_of(text)) parser.feed(id, &events);
  parser.finish(&events);
  Run run;
  for (const Event& ev : events) {
    run.order.push_back(ev.kind);
    switch (ev.kind) {
      case Kind::kReasoning: run.reasoning += ev.text; break;
      case Kind::kReasoningClosed: ++run.reasoning_closed; break;
      case Kind::kContent: run.content += ev.text; break;
      case Kind::kToolCall: run.calls.push_back(ev.call); break;
    }
  }
  require(static_cast<int>(run.calls.size()) == parser.calls(), "calls() counts the emitted calls");
  return run;
}

// ---- the DeepSeek-V4.1 DSML format: the tag token is the only id; the
// brackets and tag names are text, the block opens at the tag token after
// a "<" and closes at "</｜DSML｜ calls>".
ChatMarkers dsml_markers(DsmlDialect dialect = DsmlDialect::kV41) {
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.dsml = ChatMarker{kDsml, "｜DSML｜"};
  m.dsml_dialect = dialect;
  return m;
}
std::vector<int64_t> dsml_ids_of(const std::string& text) {
  static const std::vector<std::pair<std::string, int64_t>> table = {
      {"｜DSML｜", kDsml}, {"</think>", kThinkClose}, {"<think>", kThinkOpen},
  };
  std::vector<int64_t> out;
  for (size_t i = 0; i < text.size();) {
    bool matched = false;
    for (const auto& [s, id] : table) {
      if (text.compare(i, s.size(), s) == 0) {
        out.push_back(id);
        i += s.size();
        matched = true;
        break;
      }
    }
    if (!matched) out.push_back(static_cast<unsigned char>(text[i++]));
  }
  return out;
}
Run drive_dsml(const std::string& text, ToolCallParser::Options opts = {}, DsmlDialect dialect = DsmlDialect::kV41) {
  ToolCallParser parser(dsml_markers(dialect), fake_decode, weather_schemas(), opts);
  std::vector<Event> events;
  for (const int64_t id : dsml_ids_of(text)) parser.feed(id, &events);
  parser.finish(&events);
  Run run;
  for (const Event& ev : events) {
    run.order.push_back(ev.kind);
    switch (ev.kind) {
      case Kind::kReasoning: run.reasoning += ev.text; break;
      case Kind::kReasoningClosed: ++run.reasoning_closed; break;
      case Kind::kContent: run.content += ev.text; break;
      case Kind::kToolCall: run.calls.push_back(ev.call); break;
    }
  }
  require(static_cast<int>(run.calls.size()) == parser.calls(), "calls() counts the emitted calls");
  return run;
}

// The prompt-tail rules (ChatMarkers::prompt_opens_thinking /
// prompt_leaves_thinking_to_model) over every generation-prompt shape the
// families render (2026-10-22, the Flint rule). Whether the model opens
// its own <think> is the family's declaration (model_opens_thinking,
// ServeFamily states it): without it no tail ever leaves the choice to
// the model; with it only a tail the template left unsettled does — the
// GLM shapes end inside <think> or with a closed block, Qwen's open the
// block or close it, MiMo's "assistant\n" and Flint's "<|assistant|>\n"
// carry neither. The tokens never select the family.
DGPP_TEST(tool_parser_prompt_tail_leaves_thinking_to_the_model_by_the_family_declaration) {
  constexpr int64_t kAssistant = 900, kImStart = 901, kNewline = 902, kWord = 903, kText = 904,
                    kNewline2 = 905;  // the "\n\n" id (Qwen's 271)
  auto glm = [&](bool declares) {
    ChatMarkers m = fake_markers();
    m.role_markers = {ChatMarker{kAssistant, "<|assistant|>"}};
    m.newline = ChatMarker{kNewline, "\n"};
    m.newline_ids = {kNewline, kNewline2};
    m.model_opens_thinking = declares;
    return m;
  };
  auto qwen = [&](bool declares) {
    ChatMarkers m = qwen_markers();
    m.role_markers = {ChatMarker{kImStart, "<|im_start|>"}};
    m.newline = ChatMarker{kNewline, "\n"};
    m.newline_ids = {kNewline, kNewline2};
    m.model_opens_thinking = declares;
    return m;
  };
  struct Case {
    const char* what;
    std::vector<int64_t> tail;
    bool opens;     // prompt_opens_thinking
    bool unsettled; // leaves_thinking_to_model under the declaration
  };
  const Case glm_cases[] = {
      {"GLM-5.3-Flash thinking: <|assistant|><think>", {kText, kAssistant, kThinkOpen}, true, false},
      {"GLM with a newline after <think>", {kText, kAssistant, kThinkOpen, kNewline}, true, false},
      {"GLM-4.7 enable_thinking false: <think></think>", {kText, kAssistant, kThinkOpen, kThinkClose}, false, false},
      {"GLM closed block then newlines", {kText, kAssistant, kThinkOpen, kThinkClose, kNewline, kNewline}, false, false},
      {"Flint: the prompt ends at <|assistant|>", {kText, kAssistant}, false, true},
      {"Flint: <|assistant|> then its newline", {kText, kAssistant, kNewline}, false, true},
  };
  const Case qwen_cases[] = {
      {"Qwen thinking: assistant then <think>\n", {kImStart, kWord, kNewline, kThinkOpen, kNewline}, true, false},
      {"Qwen enable_thinking false: closed block", {kImStart, kWord, kNewline, kThinkOpen, kThinkClose, kNewline, kNewline}, false, false},
      {"Qwen3.8 enable_thinking false: <think>\n\n</think>\n\n with the pair as one id",
       {kImStart, kWord, kNewline, kThinkOpen, kNewline2, kThinkClose, kNewline2}, false, false},
      {"MiMo: the bare assistant header", {kImStart, kWord, kNewline}, false, true},
      {"MiMo enable_thinking false: <think></think>", {kImStart, kWord, kNewline, kThinkOpen, kThinkClose}, false, false},
  };
  const auto check = [&](const ChatMarkers& declared, const ChatMarkers& silent, const Case& c) {
    require(declared.prompt_opens_thinking(c.tail) == c.opens, std::string(c.what) + ": opens_thinking");
    require(declared.prompt_leaves_thinking_to_model(c.tail) == c.unsettled,
            std::string(c.what) + ": leaves_thinking_to_model under the family's declaration");
    // No declaration (GLM, Qwen, DeepSeek): the same tail never leaves the choice.
    require(!silent.prompt_leaves_thinking_to_model(c.tail), std::string(c.what) + ": never without the declaration");
    require(silent.prompt_opens_thinking(c.tail) == c.opens, std::string(c.what) + ": opens_thinking is the template's");
  };
  for (const Case& c : glm_cases) check(glm(true), glm(false), c);
  for (const Case& c : qwen_cases) check(qwen(true), qwen(false), c);
  // The declaration without the reasoning markers: never.
  ChatMarkers plain = glm(true);
  plain.think_open = ChatMarker{};
  plain.think_close = ChatMarker{};
  require(!plain.prompt_leaves_thinking_to_model({kText, kAssistant}), "no think markers: never");
  require(!ChatMarkers{}.prompt_leaves_thinking_to_model({kText, kAssistant}), "default markers: never");
}

DGPP_TEST(tool_parser_incremental_text_has_linear_decode_work) {
  const std::string reasoning = std::string(2048, 'r') + "é📭";
  for (const auto& markers : {fake_markers(), qwen_markers(), dsml_markers(),
                              dsml_markers(DsmlDialect::kV4)}) {
    const bool dsml = markers.dsml.available();
    const std::string content = std::string(2048, 'c') + " 中\n\nx < y\n" +
                               (dsml ? "｜DSML｜ tail\n<" : "tail\n<");
    auto ids = dsml_ids_of(reasoning + "</think>" + content);
    ids.insert(ids.begin() + 1, kEos);  // an empty decode inside reasoning
    ids.push_back(kEos);               // an empty decode after a held suffix
    for (const bool track : {false, true}) {
      ToolCallParser::Options opts;
      opts.track_tokens = track;
      size_t decoded_ids = 0;
      ToolCallParser parser(markers, [&](const std::vector<int64_t>& input) {
        decoded_ids += input.size();
        require(decoded_ids <= ids.size(), "text decoding must process at most one id per input token");
        return fake_decode(input);
      }, weather_schemas(), opts);
      std::vector<Event> events;
      for (const auto id : ids) parser.feed(id, &events);
      parser.finish(&events);
      std::string got_reasoning, got_content;
      int closed = 0;
      for (const auto& ev : events) {
        require(ev.kind != Kind::kToolCall, "literal text does not open a call");
        if (ev.kind == Kind::kReasoningClosed) { ++closed; continue; }
        (ev.kind == Kind::kReasoning ? got_reasoning : got_content) += ev.text;
        if (!track) {
          require(ev.tokens.empty(), "token attribution is opt-in");
          continue;
        }
        size_t end = 0;
        for (const auto& span : ev.tokens) {
          require(span.begin == end && span.end <= ev.text.size() && span.token < ids.size(),
                  "ordered, complete byte attribution");
          const auto bytes = ids[span.token] == kDsml ? std::string("｜DSML｜")
                                                     : fake_decode({ids[span.token]});
          require(bytes == ev.text.substr(span.begin, span.end - span.begin),
                  "each delta retains the source token's bytes");
          end = span.end;
        }
        require(end == ev.text.size(), "no unattributed delta bytes");
      }
      require(got_reasoning == reasoning && got_content == content && closed == 1,
              "long text, split UTF-8, skipped tokens and DSML tails retain exact bytes");
    }
  }
}

DGPP_TEST(tool_parser_incremental_dsml_preserves_partial_token_spans) {
  // Tokens can contain both visible text and a held block prefix. Include
  // UTF-8 split between tokens and a skipped token between its two halves.
  const std::map<int64_t, std::string> pieces = {
      {2001, "é📭\n\n<"}, {2002, "tail\n"}, {2003, " calls>broken"},
      {2004, "\xE4\xB8"}, {2005, "\xAD\n\n"}};
  const auto decode = [&](const std::vector<int64_t>& ids) {
    std::string out;
    for (const auto id : ids) {
      const auto it = pieces.find(id);
      out += it == pieces.end() ? fake_decode({id}) : it->second;
    }
    return out;
  };
  for (const auto dialect : {DsmlDialect::kV41, DsmlDialect::kV4}) {
    for (const auto& ids : {std::vector<int64_t>{2001, kEos, 2002},
                           std::vector<int64_t>{2004, kEos, 2005, kDsml, 2002},
                           std::vector<int64_t>{2001, kDsml, 2003}}) {
      ToolCallParser::Options opts;
      opts.start_in_reasoning = false;
      opts.track_tokens = true;
      ToolCallParser parser(dsml_markers(dialect), decode, weather_schemas(), opts);
      std::vector<Event> events;
      for (const auto id : ids) parser.feed(id, &events);
      parser.finish(&events);
      std::string expected, content, attributed;
      std::vector<size_t> consumed(ids.size(), 0);
      for (const auto id : ids)
        expected += id == kDsml ? std::string("｜DSML｜") : decode({id});
      for (const auto& ev : events) {
        require(ev.kind == Kind::kContent, "incomplete DSML blocks remain literal content");
        content += ev.text;
        size_t end = 0;
        for (const auto& span : ev.tokens) {
          require(span.begin == end && span.end <= ev.text.size() && span.token < ids.size(),
                  "held prefix spans cover the emitted bytes in order");
          const auto id = ids[span.token];
          const auto bytes = id == kDsml ? std::string("｜DSML｜") : decode({id});
          const size_t length = span.end - span.begin;
          const auto part = bytes.substr(consumed[span.token], length);
          require(part == ev.text.substr(span.begin, length), "partial span retains its source bytes");
          consumed[span.token] += length;
          attributed += part;
          end = span.end;
        }
        require(end == ev.text.size(), "all emitted bytes have provenance");
      }
      require(content == expected && attributed == expected, "held and malformed DSML is byte-exact");
    }
  }
}

DGPP_TEST(tool_parser_dsml_format_one_call_string_and_json_values) {
  require(dsml_markers().tool_format() == dgpp::text::ToolFormat::kDsml, "the tag token alone is the DSML format");
  require(dsml_markers().tool_calls_available(), "DSML tool calls are available");
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const Run run = drive_dsml(
      "Sure, let me check.\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
      "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n"
      "<｜DSML｜ parameter name=\"days\" string=\"false\">3</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n</｜DSML｜ calls>",
      plain);
  require(run.content == "Sure, let me check.", "the content stops before the block's blank line: '" + run.content + "'");
  require(run.calls.size() == 1, "one call");
  require(run.calls[0].name == "get_weather", "the call's name");
  require(run.calls[0].arguments == "{\"city\": \"Paris\", \"days\": 3}", "typed arguments: " + run.calls[0].arguments);
}

DGPP_TEST(tool_parser_content_token_provenance) {
  ToolCallParser::Options opts;
  opts.start_in_reasoning = false;
  opts.track_tokens = true;
  for (const std::string text : {"if a < b\n\n", "x ｜DSML｜ y",
                                 "ok\n\n<｜DSML｜ calls>broken"}) {
    const auto ids = dsml_ids_of(text);
    ToolCallParser parser(dsml_markers(), fake_decode, weather_schemas(), opts);
    std::vector<Event> events;
    for (const auto id : ids) parser.feed(id, &events);
    parser.finish(&events);
    std::string content, attributed;
    for (const auto& ev : events) {
      if (ev.kind != Kind::kContent) continue;
      content += ev.text;
      size_t end = 0;
      for (const auto& span : ev.tokens) {
        require(span.begin == end && span.end <= ev.text.size(), "complete, ordered byte attribution");
        require(span.token < ids.size() && ids[span.token] != kEos, "source token index");
        const auto decoded = ids[span.token] == kDsml ? std::string("｜DSML｜") : fake_decode({ids[span.token]});
        require(decoded == ev.text.substr(span.begin, span.end - span.begin), "attributed bytes match source");
        attributed += decoded;
        end = span.end;
      }
      require(end == ev.text.size(), "no unattributed content bytes");
    }
    require(content == text && attributed == text, "held and malformed DSML retains token provenance");
  }
}

DGPP_TEST(tool_parser_dsml_format_two_calls_reasoning_and_namespace) {
  const Run run = drive_dsml(
      "think first</think>\n\n<｜DSML｜ calls>\n"
      "<｜DSML｜ invoke name=\"search::lookup\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">a \"quoted\" value</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
      "<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Rome</｜DSML｜ parameter>\n"
      "<｜DSML｜ parameter name=\"flags\" string=\"false\">{\"metric\": true, \"n\": [1, 2]}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
      "</｜DSML｜ calls>");
  require(run.reasoning == "think first" && run.reasoning_closed == 1, "the reasoning split");
  require(run.content.empty(), "no content before the block: '" + run.content + "'");
  require(run.calls.size() == 2, "two calls");
  require(run.calls[0].name == "lookup", "a namespaced call reports its bare name: " + run.calls[0].name);
  require(run.calls[0].arguments == "{\"query\": \"a \\\"quoted\\\" value\"}", "the quoted string value: " + run.calls[0].arguments);
  require(run.calls[1].arguments == "{\"city\": \"Rome\", \"flags\": {\"metric\": true, \"n\": [1, 2]}}",
          "the JSON value normalized: " + run.calls[1].arguments);
}

DGPP_TEST(tool_parser_dsml_format_holds_back_only_a_block_prefix) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  // A lone "<" and blank lines that never open a block are content, in full.
  const Run a = drive_dsml("if a < b then\n\nc < d\n\n", plain);
  require(a.content == "if a < b then\n\nc < d\n\n", "held prefixes flush as content: '" + a.content + "'");
  // The tag token without its "<" is literal text (the decode skips the special token; the parser restores it).
  const Run b = drive_dsml("x ｜DSML｜ y", plain);
  require(b.content == "x ｜DSML｜ y", "a stray tag token prints verbatim: '" + b.content + "'");
}

DGPP_TEST(tool_parser_dsml_format_malformed_block_falls_back_to_content) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  for (const char* bad : {
           "ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n<｜DSML｜ parameter name=\"city\">Paris</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
           "ok\n\n<｜DSML｜ calls><｜DSML｜ invoke name=\"get_weather\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
           "ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n",
       }) {
    const Run run = drive_dsml(bad, plain);
    require(run.calls.empty(), std::string("no call from a malformed block: ") + bad);
    require(run.content == bad, "the malformed block is literal content: '" + run.content + "'");
  }
  // Text after a closed block is content (the streaming parser completes
  // the block at its closing tag; the reference's whole-completion check
  // would reject the turn, which a client sees as the trailing content).
  const Run tail = drive_dsml(
      "ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n</｜DSML｜ invoke>\n</｜DSML｜ calls> trailing", plain);
  require(tail.calls.size() == 1 && tail.calls[0].arguments == "{}", "the closed block's call stands");
  require(tail.content == "ok trailing", "the text around the block: '" + tail.content + "'");
}

// ---- the DeepSeek-V4 dialect of DSML (encoding_dsv4.py): the same tag
// token, but no space after it and the block named "tool_calls"; no tool
// namespaces. The tokenizer cannot tell the dialects apart — the markers
// state it.
DGPP_TEST(tool_parser_dsml_v4_dialect_one_call_string_and_json_values) {
  require(dsml_markers().dsml_dialect == DsmlDialect::kV41 && dsml_markers().dsml_namespaces(),
          "the default dialect is V4.1's, with namespaces");
  const ChatMarkers v4 = dsml_markers(DsmlDialect::kV4);
  require(v4.tool_format() == dgpp::text::ToolFormat::kDsml && v4.tool_calls_available() && !v4.dsml_namespaces(),
          "the V4 dialect is the DSML format without namespaces");
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const Run run = drive_dsml(
      "Sure, let me check.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n"
      "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n"
      "<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n"
      "</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
      plain, DsmlDialect::kV4);
  require(run.content == "Sure, let me check.", "the content stops before the block's blank line: '" + run.content + "'");
  require(run.calls.size() == 1, "one call");
  require(run.calls[0].name == "get_weather", "the call's name");
  require(run.calls[0].arguments == "{\"city\": \"Paris\", \"days\": 3}", "typed arguments: " + run.calls[0].arguments);
}

DGPP_TEST(tool_parser_dsml_v4_dialect_calls_reasoning_and_no_namespace) {
  // The reference's own shape (encoding/tests/test_output_1.txt): the
  // reasoning, "</think>", a blank line, the block. A "::" in a name is
  // part of the name here — V4 has no namespaces to strip. An invoke
  // without parameters comes in both forms: the reference RENDERS it with
  // a blank line between its tags (the empty parameter list between the
  // template's two newlines) and its parser reads that and the plain form.
  const Run run = drive_dsml(
      "think first</think>\n\n<｜DSML｜tool_calls>\n"
      "<｜DSML｜invoke name=\"search::lookup\">\n<｜DSML｜parameter name=\"query\" string=\"true\">a \"quoted\" value</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
      "<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Rome</｜DSML｜parameter>\n"
      "<｜DSML｜parameter name=\"flags\" string=\"false\">{\"metric\": true, \"n\": [1, 2]}</｜DSML｜parameter>\n</｜DSML｜invoke>\n"
      "<｜DSML｜invoke name=\"now\">\n\n</｜DSML｜invoke>\n"
      "<｜DSML｜invoke name=\"ping\">\n</｜DSML｜invoke>\n"
      "</｜DSML｜tool_calls>",
      {}, DsmlDialect::kV4);
  require(run.reasoning == "think first" && run.reasoning_closed == 1, "the reasoning split");
  require(run.content.empty(), "no content before the block: '" + run.content + "'");
  require(run.calls.size() == 4, "four calls: " + std::to_string(run.calls.size()));
  require(run.calls[0].name == "search::lookup", "a V4 name is reported as written: " + run.calls[0].name);
  require(run.calls[0].arguments == "{\"query\": \"a \\\"quoted\\\" value\"}", "the quoted string value: " + run.calls[0].arguments);
  require(run.calls[1].arguments == "{\"city\": \"Rome\", \"flags\": {\"metric\": true, \"n\": [1, 2]}}",
          "the JSON value normalized: " + run.calls[1].arguments);
  require(run.calls[2].name == "now" && run.calls[2].arguments == "{}", "the reference's blank-line form of a call without parameters");
  require(run.calls[3].name == "ping" && run.calls[3].arguments == "{}", "the plain form of a call without parameters");
  // The blank line is the header's only slack: two of them, or one before
  // a later tag, is not the reference's shape.
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  for (const char* bad : {
           "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"now\">\n\n\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
           "ok\n\n<｜DSML｜tool_calls>\n\n<｜DSML｜invoke name=\"now\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
           "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"now\">\n</｜DSML｜invoke>\n\n</｜DSML｜tool_calls>",
       }) {
    const Run r = drive_dsml(bad, plain, DsmlDialect::kV4);
    require(r.calls.empty() && r.content == bad, std::string("not the reference's shape: ") + bad);
  }
}

DGPP_TEST(tool_parser_dsml_dialects_do_not_cross) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const std::string v41 =
      "ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
      "<｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
  const std::string v4 =
      "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n"
      "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
  // Each parser reads its own spelling...
  require(drive_dsml(v41, plain, DsmlDialect::kV41).calls.size() == 1, "V4.1 reads the V4.1 spelling");
  require(drive_dsml(v4, plain, DsmlDialect::kV4).calls.size() == 1, "V4 reads the V4 spelling");
  // ...and the other one's block never closes for it: literal content at
  // the end of the turn, nothing lost, no call.
  const Run a = drive_dsml(v4, plain, DsmlDialect::kV41);
  require(a.calls.empty() && a.content == v4, "the V4 spelling under the V4.1 dialect is content: '" + a.content + "'");
  const Run b = drive_dsml(v41, plain, DsmlDialect::kV4);
  require(b.calls.empty() && b.content == v41, "the V4.1 spelling under the V4 dialect is content: '" + b.content + "'");
}

DGPP_TEST(tool_parser_dsml_v4_dialect_malformed_block_and_held_prefix) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  for (const char* bad : {
           "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
           "ok\n\n<｜DSML｜tool_calls><｜DSML｜invoke name=\"get_weather\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
           "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n"
           "<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n"
           "<｜DSML｜parameter name=\"city\" string=\"true\">Rome</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
           "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n",
       }) {
    const Run run = drive_dsml(bad, plain, DsmlDialect::kV4);
    require(run.calls.empty(), std::string("no call from a malformed block: ") + bad);
    require(run.content == bad, "the malformed block is literal content: '" + run.content + "'");
  }
  const Run held = drive_dsml("if a < b then\n\nc < d\n\n", plain, DsmlDialect::kV4);
  require(held.content == "if a < b then\n\nc < d\n\n", "held prefixes flush as content: '" + held.content + "'");
  const Run stray = drive_dsml("x ｜DSML｜ y", plain, DsmlDialect::kV4);
  require(stray.content == "x ｜DSML｜ y", "a stray tag token prints verbatim: '" + stray.content + "'");
  const Run tail = drive_dsml(
      "ok\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls> trailing", plain,
      DsmlDialect::kV4);
  require(tail.calls.size() == 1 && tail.calls[0].arguments == "{}", "the closed block's call stands");
  require(tail.content == "ok trailing", "the text around the block: '" + tail.content + "'");
}

// The arguments' re-serialization is json.dumps' — floats included: the
// notation follows the decimal exponent as Python's repr does, not the
// shorter spelling (100000.0 is not "1e+05").
DGPP_TEST(tool_parser_json_floats_print_as_python_repr) {
  const auto dumps = [](const std::string& json) {
    const dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(json);
    return dgpp::text::Value::from_minijson(parsed.root).to_json(false);
  };
  for (const auto& [in, want] : std::vector<std::pair<std::string, std::string>>{
           {"0.5", "0.5"}, {"1.0", "1.0"}, {"100000.0", "100000.0"}, {"1e5", "100000.0"}, {"1E+5", "100000.0"},
           {"123456.789", "123456.789"}, {"1e15", "1000000000000000.0"}, {"1e16", "1e+16"}, {"1.5e16", "1.5e+16"},
           {"1e22", "1e+22"}, {"0.0001", "0.0001"}, {"0.00015", "0.00015"}, {"0.00001", "1e-05"}, {"1.5e-7", "1.5e-07"},
           {"-2.50", "-2.5"}, {"-0.0", "-0.0"}, {"0.0", "0.0"}, {"3.141592653589793", "3.141592653589793"},
           {"1.7976931348623157e308", "1.7976931348623157e+308"}, {"5e-324", "5e-324"}, {"12", "12"}, {"-7", "-7"}})
    require(dumps(in) == want, "json.dumps(" + in + ") is " + want + ", got " + dumps(in));
  require(dumps("[0.1, 1e5, {\"a\": 2.0}]") == "[0.1, 100000.0, {\"a\": 2.0}]", "nested floats");
}

DGPP_TEST(tool_parser_qwen_format_markers_and_one_call) {
  require(qwen_markers().tool_format() == dgpp::text::ToolFormat::kQwenXml, "the two-marker set is the Qwen format");
  require(fake_markers().tool_format() == dgpp::text::ToolFormat::kGlmMarkers, "the six-marker set is the GLM format");
  require(qwen_markers().tool_calls_available(), "Qwen tool calls are available");
  // The template's exact shape: typed by the schema (city string, days integer).
  const Run run = drive_qwen(
      "reasoning\n</think>\n\n<tool_call>\n<function=get_weather>\n"
      "<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
      "</function>\n</tool_call>");
  require(run.reasoning == "reasoning\n", "reasoning: " + run.reasoning);
  require(run.content == "\n\n", "content before the call: '" + run.content + "'");
  require(run.calls.size() == 1 && run.calls[0].name == "get_weather", "one call to get_weather");
  require(run.calls[0].arguments == "{\"city\": \"Paris\", \"days\": 3}", "arguments: " + run.calls[0].arguments);
}

DGPP_TEST(tool_parser_qwen_format_multiline_nested_and_two_calls) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const Run run = drive_qwen(
      "Let me check both.\n\n<tool_call>\n<function=get_weather>\n"
      "<parameter=code>\nprint(1)\nprint(2)\n\n</parameter>\n"
      "<parameter=opts>\n{\"a\": 1, \"b\": [1, 2]}\n</parameter>\n"
      "<parameter=note>\n\n</parameter>\n"
      "</function>\n</tool_call>\n<tool_call>\n<function=flat_tool>\n"
      "<parameter=x>\n0.5\n</parameter>\n</function>\n</tool_call>", plain);
  require(run.content == "Let me check both.\n\n\n", "content: '" + run.content + "'");
  require(run.calls.size() == 2, "two calls");
  require(run.calls[0].arguments ==
              "{\"code\": \"print(1)\\nprint(2)\\n\", \"opts\": {\"a\": 1, \"b\": [1, 2]}, \"note\": \"\"}",
          "first arguments: " + run.calls[0].arguments);
  require(run.calls[1].name == "flat_tool" && run.calls[1].arguments == "{\"x\": 0.5}",
          "second call: " + run.calls[1].arguments);
}

DGPP_TEST(tool_parser_qwen_format_malformed_blocks_fall_back_to_content) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  // No </function>, text after </function>, a missing name, an unterminated
  // parameter, the stream ending inside the block: literal content, no call.
  for (const char* bad : {
           "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</tool_call>",
           "<tool_call>\n<function=get_weather>\n</function>\nextra</tool_call>",
           "<tool_call>\n<function=>\n</function>\n</tool_call>",
           "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis</function>\n</tool_call>",
           "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n",
       }) {
    const Run run = drive_qwen(bad, plain);
    require(run.calls.empty(), std::string("malformed block parsed as a call: ") + bad);
    require(run.content == bad, std::string("literal fallback differs: ") + run.content);
  }
  // A nested opener restarts the block: the first block's text is content,
  // the second parses.
  const Run run = drive_qwen(
      "<tool_call>\n<function=get_weather>\n<tool_call>\n<function=get_weather>\n"
      "<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>", plain);
  require(run.content == "<tool_call>\n<function=get_weather>\n", "the aborted block is content: " + run.content);
  require(run.calls.size() == 1 && run.calls[0].arguments == "{\"city\": \"Oslo\"}", "the restarted block parses");
}

DGPP_TEST(tool_parser_rejects_aRepeatedParameterName) {
  // A repeated <parameter=NAME> is how the XML format writes the same
  // argument twice, and a duplicate key is never a valid object
  // (tool_grammar.hpp). The block is rejected — as the DSML path already
  // rejects one — and its text stands as content instead.
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const std::string dup =
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n"
      "<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";
  const Run qwen = drive_qwen(dup, plain);
  require(qwen.calls.empty(), "a repeated parameter is not a call");
  require(qwen.content == dup, "the block stands as content: " + qwen.content);
  // Distinct names over the same schema still parse.
  const Run ok = drive_qwen(
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n"
      "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
      plain);
  require(ok.calls.size() == 1, "distinct names produce one call");
  require(ok.calls[0].arguments == "{\"city\": \"Rome\", \"days\": 2}",
          "distinct names parse: " + ok.calls[0].arguments);
  // The DSML format's ledger, for parity: a repeat is content there too.
  const std::string dsml_dup =
      "ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n"
      "<｜DSML｜ parameter name=\"city\" string=\"true\">Rome</｜DSML｜ parameter>\n"
      "<｜DSML｜ parameter name=\"city\" string=\"true\">Oslo</｜DSML｜ parameter>\n"
      "</｜DSML｜ invoke>\n</｜DSML｜ calls>";
  const Run dsml = drive_dsml(dsml_dup, plain);
  require(dsml.calls.empty() && dsml.content == dsml_dup,
          "a repeated DSML parameter is content: " + dsml.content);
}

DGPP_TEST(tool_parser_qwen_duplicateKeysFallbackAndRecover) {
  const auto parameter = [](const std::string& name, const std::string& value) {
    return "<parameter=" + name + ">\n" + value + "\n</parameter>\n";
  };
  const auto block = [](const std::string& body) {
    return "<tool_call>\n<function=get_weather>\n" + body + "</function>\n</tool_call>";
  };
  const std::string good = block(parameter("city", "Paris"));
  for (const std::string key : {"city", "days", "unknown", "cité"}) {
    for (const std::string second : {"1", "2"}) {
      const std::string bad =
          block(parameter(key, "1") + parameter("between", "x") + parameter(key, second));
      const Run run = drive_qwen("Reason</think>Before" + good + bad + good + "After");
      require(run.reasoning == "Reason" && run.content == "Before" + bad + "After",
              "a duplicate preserves literal content and reasoning: " + key);
      require(run.calls.size() == 2, "valid calls before and after a duplicate survive");
      require(run.calls[0].arguments == "{\"city\": \"Paris\"}" &&
                  run.calls[1].arguments == run.calls[0].arguments,
              "key use is scoped to each call");
    }
  }
}

DGPP_TEST(tool_parser_splitsReasoningFromContentExactly) {
  // The prompt opened <think>; the ids before </think> are reasoning, the
  // rest content, both streamed as exact deltas; a repeated <think> in the
  // reasoning is dropped; </think> yields the structural event once.
  const Run run = drive(ids_of("<think>Let me think.</think>Hello, world."));
  require(run.reasoning == "Let me think.", "reasoning: " + run.reasoning);
  require(run.content == "Hello, world.", "content: " + run.content);
  require(run.reasoning_closed == 1, "one </think> event");
  require(run.calls.empty(), "no calls");
  // Deltas arrive per id, in order: every reasoning delta before the close,
  // every content delta after.
  bool closed = false;
  for (const Kind k : run.order) {
    if (k == Kind::kReasoningClosed) closed = true;
    require(k != Kind::kReasoning || !closed, "reasoning before the close");
    require(k != Kind::kContent || closed, "content after the close");
  }
  // Without the opening think (a prompt that does not end in <think>),
  // everything is content and a stray </think> is literal text.
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const Run flat = drive(ids_of("A</think>B"), plain);
  require(flat.reasoning.empty() && flat.content == "A</think>B",
          "no split without the opening think: " + flat.content);
}

DGPP_TEST(tool_parser_oneCallTypedByTheSchema) {
  // Schema-typed values: city (string) stays text even though "123" parses,
  // days (integer) parses, code (string) keeps JSON-looking text verbatim,
  // an unknown key infers JSON when it parses and text otherwise.
  const Run run = drive(ids_of(
      "</think><tool_call>get_weather"
      "<arg_key>city</arg_key><arg_value>123</arg_value>"
      "<arg_key>days</arg_key><arg_value>3</arg_value>"
      "<arg_key>code</arg_key><arg_value>{\"a\": 1}</arg_value>"
      "<arg_key>ratio</arg_key><arg_value>0.5</arg_value>"
      "<arg_key>note</arg_key><arg_value>asked twice</arg_value>"
      "<arg_key>flag</arg_key><arg_value>true</arg_value>"
      "</tool_call>"));
  require(run.calls.size() == 1, "one call");
  require(run.calls[0].name == "get_weather", "name: " + run.calls[0].name);
  require(run.calls[0].arguments ==
              "{\"city\": \"123\", \"days\": 3, \"code\": \"{\\\"a\\\": 1}\", "
              "\"ratio\": 0.5, \"note\": \"asked twice\", \"flag\": true}",
          "arguments: " + run.calls[0].arguments);
  require(run.content.empty() && run.reasoning.empty(), "nothing else");
}

DGPP_TEST(tool_parser_preservesNumbersAcrossSchemaBoundaries) {
  for (const std::string value : {
           "0.100000000000000000001", "1e-5000", "1e5000", "9223372036854775809",
           "{\"n\":[0.100000000000000000001],\"s\":\"123\"}"}) {
    const Run glm = drive(ids_of("</think><tool_call>flat_tool<arg_key>x</arg_key><arg_value>" +
                                 value + "</arg_value></tool_call>"));
    const Run qwen = drive_qwen("</think><tool_call>\n<function=flat_tool>\n<parameter=x>\n" +
                                value + "\n</parameter>\n</function>\n</tool_call>");
    const Run dsml = drive_dsml("</think><｜DSML｜ calls>\n<｜DSML｜ invoke name=\"flat_tool\">\n"
                                "<｜DSML｜ parameter name=\"x\" string=\"false\">" + value +
                                "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>");
    for (const Run& run : {glm, qwen, dsml})
      require(run.calls.size() == 1 && run.calls[0].arguments == "{\"x\": " + value + "}",
              "tool output preserves the numeric value accepted by the grammar: " + value);
  }
}

DGPP_TEST(tool_parser_nestedJsonAndNoArgsAndUnicode) {
  const Run run = drive(ids_of(
      "</think>Calling.<tool_call>get_weather"
      "<arg_key>opts</arg_key><arg_value>{\"a\": [1, 2, {\"b\": null}], \"c\": \"x\"}</arg_value>"
      "</tool_call><tool_call>flat_tool</tool_call>"
      "<tool_call>other<arg_key>k</arg_key><arg_value>[1, \"two\"]</arg_value>"
      "<arg_key>city</arg_key><arg_value>Zürich</arg_value></tool_call>Done."));
  require(run.calls.size() == 3, "three calls, got " +
                                     std::to_string(run.calls.size()));
  require(run.calls[0].arguments ==
              "{\"opts\": {\"a\": [1, 2, {\"b\": null}], \"c\": \"x\"}}",
          "nested JSON re-serialized in json.dumps form: " +
              run.calls[0].arguments);
  require(run.calls[1].name == "flat_tool" && run.calls[1].arguments == "{}",
          "a call without arguments is an empty object");
  // A function outside the schema: JSON when it parses, else text (the
  // UTF-8 passes through unescaped, ensure_ascii=False).
  require(run.calls[2].arguments ==
              "{\"k\": [1, \"two\"], \"city\": \"Zürich\"}",
          "unknown function: " + run.calls[2].arguments);
  require(run.content == "Calling.Done.", "content around the calls: " +
                                              run.content);
  // Order: content, call, call, call, content.
  require(run.order.size() >= 5 && run.order.back() == Kind::kContent,
          "trailing content after the calls");
}

DGPP_TEST(tool_parser_malformedBlocksFallBackToLiteralContent) {
  // A value without a key.
  {
    const Run run = drive(ids_of(
        "</think><tool_call>f<arg_value>x</arg_value></tool_call>tail"));
    require(run.calls.empty(), "no call from a value without a key");
    require(run.content == "<tool_call>f<arg_value>x</arg_value></tool_call>tail",
            "the literal block is content: " + run.content);
  }
  // A key without a value, then the close.
  {
    const Run run = drive(ids_of("</think><tool_call>f<arg_key>k</arg_key></tool_call>"));
    require(run.calls.empty() &&
                run.content == "<tool_call>f<arg_key>k</arg_key></tool_call>",
            "key without value: " + run.content);
  }
  // Text between </arg_key> and <arg_value>.
  {
    const Run run = drive(ids_of(
        "</think><tool_call>f<arg_key>k</arg_key> <arg_value>v</arg_value></tool_call>"));
    require(run.calls.empty(), "stray text inside the block");
    require(run.content ==
                "<tool_call>f<arg_key>k</arg_key> <arg_value>v</arg_value></tool_call>",
            "the block aborts at the stray id and everything is literal "
            "content: " + run.content);
  }
  // Unterminated at the end of the generation (the steps cap).
  {
    const Run run = drive(ids_of(
        "</think>Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par"));
    require(run.calls.empty(), "no call from an unterminated block");
    require(run.content ==
                "Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par",
            "unterminated: " + run.content);
  }
  // A nested <tool_call> aborts the open block and starts a fresh one.
  {
    const Run run = drive(ids_of(
        "</think><tool_call>f<arg_key>k</arg_key><tool_call>g<arg_key>x</arg_key>"
        "<arg_value>1</arg_value></tool_call>"));
    require(run.calls.size() == 1 && run.calls[0].name == "g" &&
                run.calls[0].arguments == "{\"x\": 1}",
            "the second block parses");
    require(run.content == "<tool_call>f<arg_key>k</arg_key>",
            "the first block is literal content: " + run.content);
  }
  // Markers in the wrong state outside a block are literal content.
  {
    const Run run = drive(ids_of("</think>a</arg_key>b<arg_value>c"));
    require(run.calls.empty() && run.content == "a</arg_key>b<arg_value>c",
            "stray markers outside a block: " + run.content);
  }
  // An EOS id inside a block decodes to nothing and leaves it open.
  {
    std::vector<int64_t> ids = ids_of("</think><tool_call>f");
    ids.push_back(kEos);
    const Run run = drive(ids);
    require(run.calls.empty() && run.content == "<tool_call>f",
            "EOS inside a block: " + run.content);
  }
}

DGPP_TEST(tool_parser_glm_duplicateKeysFallBackToContent) {
  for (const std::string key : {"city", "extra", "café"}) {
    for (const std::string middle : {"", "<arg_key>days</arg_key><arg_value>3</arg_value>"}) {
      for (const std::string value : {"Rome", "Oslo"}) {
        const std::string block = "<tool_call>get_weather<arg_key>" + key +
                                  "</arg_key><arg_value>Rome</arg_value>" + middle + "<arg_key>" +
                                  key + "</arg_key><arg_value>" + value +
                                  "</arg_value></tool_call>";
        const Run run = drive(ids_of("Think</think>Before" + block + "After"));
        require(run.calls.empty(), "a repeated GLM argument key must not emit a call: " + key);
        require(run.reasoning == "Think", "reasoning survives duplicate-key rejection");
        require(run.content == "Before" + block + "After",
                "the complete malformed block and surrounding content are preserved");
      }
    }
  }
}

DGPP_TEST(tool_parser_glm_duplicateKeysDoNotLeakAcrossCalls) {
  const std::string valid =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value>"
      "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
  const std::string duplicate =
      "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value>"
      "<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>";
  const Run run = drive(ids_of("</think>" + valid + duplicate + valid + valid));
  require(run.content == duplicate, "only the duplicate-key call becomes content");
  require(run.calls.size() == 3, "valid calls before and after rejection still parse");
  for (const auto& call : run.calls) {
    require(call.name == "get_weather" && call.arguments == "{\"city\": \"Rome\", \"days\": 3}",
            "distinct keys retain their types and may recur in a separate call");
  }
}

DGPP_TEST(tool_parser_forcedPrefixSeedsTheBlock) {
  // tool_choice required: the prompt ends in "</think><tool_call>", so the
  // first ids are the name.
  ToolCallParser::Options forced;
  forced.start_in_reasoning = false;
  forced.start_in_tool_call = true;
  forced.forced_prefix_text = "<tool_call>";
  {
    const Run run = drive(ids_of("get_weather<arg_key>city</arg_key>"
                                 "<arg_value>Paris</arg_value></tool_call>"),
                          forced);
    require(run.calls.size() == 1 && run.calls[0].name == "get_weather" &&
                run.calls[0].arguments == "{\"city\": \"Paris\"}",
            "required: the block parses from the seeded state");
  }
  // tool_choice named: the name is in the prompt; the model may extend it
  // and then argues.
  ToolCallParser::Options named = forced;
  named.seeded_name = "get_";
  named.forced_prefix_text = "<tool_call>get_";
  {
    const Run run = drive(ids_of("weather<arg_key>days</arg_key>"
                                 "<arg_value>2</arg_value></tool_call>"),
                          named);
    require(run.calls.size() == 1 && run.calls[0].name == "get_weather" &&
                run.calls[0].arguments == "{\"days\": 2}",
            "named: seeded name + generated tail: " + run.calls[0].name);
  }
  // A forced block that never closes flushes with the prefix text, so the
  // client sees what the model effectively produced.
  {
    const Run run = drive(ids_of("weather<arg_key>days"), named);
    require(run.calls.empty() &&
                run.content == "<tool_call>get_weather<arg_key>days",
            "forced + unterminated: " + run.content);
  }
}

// The opened-thinking test over a rendered prompt: GLM's turn ends in
// <think>, Qwen's in <think> + the bare newline.
DGPP_TEST(tool_parser_promptOpensThinkingSeesTheQwenNewline) {
  ChatMarkers m = qwen_markers();
  constexpr int64_t kNewline = 198;
  require(m.prompt_opens_thinking({7, kThinkOpen}), "a bare <think> tail opens");
  require(!m.prompt_opens_thinking({7, kThinkOpen, kNewline}),
          "without the newline marker the Qwen tail is not recognized");
  m.newline = ChatMarker{kNewline, "\n"};
  require(m.prompt_opens_thinking({7, kThinkOpen, kNewline}), "<think> + newline opens");
  require(m.prompt_opens_thinking({7, kThinkOpen}), "a bare <think> tail still opens");
  require(!m.prompt_opens_thinking({7, kNewline}), "a newline alone does not open");
  require(!m.prompt_opens_thinking({kThinkOpen, kNewline, 7}), "text after the tail closes nothing");
  require(!m.prompt_opens_thinking({}), "an empty prompt");
}

// ---- the MiMo-V2.6 dialect of the XML format (2026-09-22): the same two
// markers, the template writing the call without newlines, the model
// opening its own <think> block (the prompt ends in "assistant\n").
ChatMarkers mimo_markers() {
  ChatMarkers m = qwen_markers();
  m.xml_compact = true;
  m.model_opens_thinking = true;  // the family's declaration (ServeFamily)
  return m;
}
Run drive_mimo(const std::string& text, ToolCallParser::Options opts = {}) {
  ToolCallParser parser(mimo_markers(), fake_decode, weather_schemas(), opts);
  std::vector<Event> events;
  for (const int64_t id : qwen_ids_of(text)) parser.feed(id, &events);
  parser.finish(&events);
  Run run;
  for (const Event& ev : events) {
    run.order.push_back(ev.kind);
    switch (ev.kind) {
      case Kind::kReasoning: run.reasoning += ev.text; break;
      case Kind::kReasoningClosed: ++run.reasoning_closed; break;
      case Kind::kContent: run.content += ev.text; break;
      case Kind::kToolCall: run.calls.push_back(ev.call); break;
    }
  }
  require(static_cast<int>(run.calls.size()) == parser.calls(), "calls() counts the emitted calls");
  return run;
}

DGPP_TEST(tool_parser_mimo_compactCallsKeepTheirNewlines) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  // The template's exact shape: no newlines around the tags, so a value's
  // trailing newline is the value's (the Qwen dialect would strip it).
  const Run run = drive_mimo(
      "Checking.<tool_call><function=get_weather><parameter=city>Paris</parameter>"
      "<parameter=days>3</parameter></function></tool_call>"
      "<tool_call><function=get_weather><parameter=code>print(1)\nprint(2)\n</parameter>"
      "<parameter=opts>{\"a\": 1, \"b\": [1, 2]}</parameter></function></tool_call>", plain);
  require(run.content == "Checking.", "content: '" + run.content + "'");
  require(run.calls.size() == 2, "two calls");
  require(run.calls[0].arguments == "{\"city\": \"Paris\", \"days\": 3}", "first: " + run.calls[0].arguments);
  require(run.calls[1].arguments == "{\"code\": \"print(1)\\nprint(2)\\n\", \"opts\": {\"a\": 1, \"b\": [1, 2]}}",
          "second keeps the trailing newline: " + run.calls[1].arguments);
  // The newline form (a forced call: the grammar emits it) parses under the
  // compact dialect with the format's own newlines stripped.
  const Run forced = drive_mimo(
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
      "<parameter=code>\nprint(1)\nprint(2)\n\n</parameter>\n</function>\n</tool_call>", plain);
  require(forced.calls.size() == 1 && forced.calls[0].arguments == "{\"city\": \"Paris\", \"code\": \"print(1)\\nprint(2)\\n\"}",
          "the newline form under the compact dialect: " + (forced.calls.empty() ? std::string("no call") : forced.calls[0].arguments));
  // The Qwen dialect on the same newline form: identical.
  const Run qwen = drive_qwen(
      "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
      "<parameter=code>\nprint(1)\nprint(2)\n\n</parameter>\n</function>\n</tool_call>", plain);
  require(qwen.calls.size() == 1 && qwen.calls[0].arguments == forced.calls[0].arguments, "the dialects agree on the newline form");
}

DGPP_TEST(tool_parser_mimo_jsonBodyIsTheArguments) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  // The template renders a client's pre-serialized arguments as one JSON
  // object inside the function tags (no parameter tags).
  const Run run = drive_mimo("<tool_call><function=get_weather>{\"city\": \"Oslo\", \"days\": 2}</function></tool_call>", plain);
  require(run.calls.size() == 1 && run.calls[0].name == "get_weather", "one call");
  require(run.calls[0].arguments == "{\"city\": \"Oslo\", \"days\": 2}", "the object's members: " + run.calls[0].arguments);
  // Not an object, or malformed: the block falls back to content.
  const Run bad = drive_mimo("<tool_call><function=get_weather>{\"city\": </function></tool_call>", plain);
  require(bad.calls.empty() && bad.content.find("<function=get_weather>") != std::string::npos, "a malformed body is content");
}

DGPP_TEST(tool_parser_mimo_unclosed_and_mixed_dialect_calls_stay_content) {
  ToolCallParser::Options plain;
  plain.start_in_reasoning = false;
  const std::string function =
      "<tool_call>\n<function=read_file>\n"
      "<parameter=path>/tmp/dgpp-tool-probe.txt</parameter>\n</function>\n";
  const std::string mixed =
      function + "</invoke>\n<invoke name=\"read_file\">\n"
                 "<parameter name=\"path\">/tmp/dgpp-tool-probe.txt</parameter>";
  for (const std::string& text : {
           function, mixed, mixed + "</tool_call>",
           std::string("<tool_call><function=read_file><parameter=path>/tmp/dgpp"),
           std::string("<tool_call><function=read_file><parameter=path>a</parameter>"
                       "<parameter=path>b</parameter></function></tool_call>"),
       }) {
    const Run run = drive_mimo(text, plain);
    require(run.calls.empty() && run.content == text,
            "malformed MiMo output must remain literal content: " + text);
  }
}

DGPP_TEST(tool_parser_mimo_modelOpensItsOwnThinking) {
  ChatMarkers m = mimo_markers();
  constexpr int64_t kNewline = 198;
  m.newline = ChatMarker{kNewline, "\n"};
  // The generation prompt ends in "assistant\n": neither marker in the tail.
  require(m.prompt_leaves_thinking_to_model({7, 8, kNewline}), "a bare assistant header leaves the choice");
  require(!m.prompt_leaves_thinking_to_model({7, kThinkOpen}), "an opened block is the prompt's");
  require(!m.prompt_leaves_thinking_to_model({7, kThinkOpen, kThinkClose}), "enable_thinking=false settles it");
  require(!m.prompt_leaves_thinking_to_model({7, kThinkClose, kNewline}), "a closed block then a newline settles it");
  require(!m.prompt_opens_thinking({7, 8, kNewline}), "the bare header does not open");
  require(!ChatMarkers{}.prompt_leaves_thinking_to_model({7}), "no markers, no choice");
  // The parser: <think> as the first id opens the reasoning; </think> closes
  // it; the call follows as content-state structure.
  ToolCallParser::Options opts;
  opts.start_in_reasoning = false;
  opts.model_may_open_thinking = true;
  const Run run = drive_mimo(
      "<think>plan the call</think>Sure.<tool_call><function=get_weather><parameter=city>Rome</parameter></function></tool_call>", opts);
  require(run.reasoning == "plan the call", "reasoning: '" + run.reasoning + "'");
  require(run.reasoning_closed == 1, "one close event");
  require(run.content == "Sure.", "content: '" + run.content + "'");
  require(run.calls.size() == 1 && run.calls[0].arguments == "{\"city\": \"Rome\"}", "the call after the block");
  // Thinking disabled (the prompt ended in <think></think>): a stray opener
  // after content is text.
  ToolCallParser::Options settled;
  settled.start_in_reasoning = false;
  const Run plain = drive_mimo("Hi <think>x</think> there", settled);
  require(plain.reasoning.empty() && plain.content == "Hi <think>x</think> there", "markers are text once settled: '" + plain.content + "'");
  // With the choice left but content already out, an opener is text too.
  const Run late = drive_mimo("Hi <think>x</think>", opts);
  require(late.reasoning.empty() && late.content == "Hi <think>x</think>", "a late opener is content: '" + late.content + "'");
}

DGPP_TEST(tool_parser_schemasReadBothToolForms) {
  const ToolSchemas s = weather_schemas();
  require(s.has("get_weather") && s.has("flat_tool") && !s.has("none"),
          "names from the wrapped and the flat form");
  require(s.type_of("get_weather", "city") == ToolSchemas::Type::kString,
          "string");
  require(s.type_of("get_weather", "days") == ToolSchemas::Type::kJson,
          "integer is JSON-typed");
  require(s.type_of("get_weather", "nope") == ToolSchemas::Type::kUnknown,
          "unknown key");
  require(s.type_of("flat_tool", "x") == ToolSchemas::Type::kJson, "number");
  require(s.type_of("missing", "x") == ToolSchemas::Type::kUnknown,
          "unknown function");
  // Markers missing from a tokenizer disable the features, never guess.
  ChatMarkers none;
  require(!none.reasoning_available() && !none.tool_calls_available(),
          "no markers, no features");
  ToolCallParser::Options opts;
  ToolCallParser parser(none, fake_decode, ToolSchemas(), opts);
  std::vector<Event> events;
  for (const int64_t id : ids_of("</think><tool_call>f</tool_call>x"))
    parser.feed(id, &events);
  std::string content;
  for (const Event& ev : events) {
    require(ev.kind == Kind::kContent, "everything is content");
    content += ev.text;
  }
  require(content == "</think><tool_call>f</tool_call>x",
          "literal without markers: " + content);
}

}  // namespace
