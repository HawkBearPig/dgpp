// Inkling 2's typed-block dialect without a checkpoint: the reply parser
// over every token split of a reply, and the constrained-decoding grammar
// over a byte vocabulary that carries the block markers as single ids.
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/tool_grammar.hpp"
#include "text/tool_parser.hpp"

namespace {
using namespace dgpp::text;

const std::string M = "<|message_model|>";
const std::string TH = "<|content_thinking|>";
const std::string TX = "<|content_text|>";
const std::string TJ = "<|content_invoke_tool_json|>";
const std::string TT = "<|content_invoke_tool_text|>";
const std::string TE = "<|content_tool_error|>";
const std::string E = "<|end_message|>";
const std::vector<std::string> kMarkerTexts = {M, TH, TX, TJ, TT, TE, E};

void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

ChatMarkers markers_at(int64_t first) {
  ChatMarkers m;
  m.message_model = {first + 0, M};
  m.content_thinking = {first + 1, TH};
  m.content_text = {first + 2, TX};
  m.content_tool_json = {first + 3, TJ};
  m.content_tool_text = {first + 4, TT};
  m.content_tool_error = {first + 5, TE};
  m.end_message = {first + 6, E};
  return m;
}

// The marker that starts at `at`, or -1.
int marker_at(const std::string& text, size_t at) {
  for (size_t k = 0; k < kMarkerTexts.size(); ++k)
    if (text.compare(at, kMarkerTexts[k].size(), kMarkerTexts[k]) == 0) return static_cast<int>(k);
  return -1;
}

bool json_equal(const dgpp::minijson::Value& a, const dgpp::minijson::Value& b) {
  if (a.is_object() != b.is_object() || a.is_array() != b.is_array() || a.is_string() != b.is_string() ||
      a.is_number() != b.is_number() || a.is_bool() != b.is_bool() || a.is_null() != b.is_null())
    return false;
  if (a.is_object()) {
    if (a.members().size() != b.members().size()) return false;
    for (const auto& [key, value] : a.members()) {
      const dgpp::minijson::Value* other = b.find(key);
      if (!other || !json_equal(value, *other)) return false;
    }
    return true;
  }
  if (a.is_array()) {
    if (a.items().size() != b.items().size()) return false;
    for (size_t i = 0; i < a.items().size(); ++i)
      if (!json_equal(a.items()[i], b.items()[i])) return false;
    return true;
  }
  if (a.is_string()) return a.as_string() == b.as_string();
  if (a.is_number()) return a.as_double() == b.as_double();
  if (a.is_bool()) return a.as_bool() == b.as_bool();
  return true;
}

bool same_json(const std::string& a, const std::string& b) {
  const auto pa = dgpp::minijson::parse(a);
  const auto pb = dgpp::minijson::parse(b);
  return json_equal(pa.root, pb.root);
}

struct Parsed {
  std::string reasoning, content;
  std::vector<ToolCallParser::Call> calls;
  int reasoning_blocks = 0;
};

// The reply's markers become their ids and the text between them is cut
// into pieces of `chunk` bytes, one id each.
Parsed parse(const std::string& reply, size_t chunk) {
  const int64_t kFirstMarker = 1000000;
  const ChatMarkers markers = markers_at(kFirstMarker);
  std::vector<std::string> pieces;
  std::vector<int64_t> ids;
  std::string run;
  const auto flush = [&] {
    for (size_t i = 0; i < run.size(); i += chunk) {
      ids.push_back(static_cast<int64_t>(pieces.size()));
      pieces.push_back(run.substr(i, chunk));
    }
    run.clear();
  };
  for (size_t at = 0; at < reply.size();) {
    const int k = marker_at(reply, at);
    if (k < 0) {
      run += reply[at++];
      continue;
    }
    flush();
    ids.push_back(kFirstMarker + k);
    at += kMarkerTexts[k].size();
  }
  flush();
  ToolCallParser parser(
      markers,
      [&](const std::vector<int64_t>& v) {
        std::string out;
        for (const int64_t id : v) out += id >= kFirstMarker ? kMarkerTexts.at(id - kFirstMarker) : pieces.at(id);
        return out;
      },
      ToolSchemas(), ToolCallParser::Options{});
  std::vector<ToolCallParser::Event> events;
  for (const int64_t id : ids) parser.feed(id, &events);
  parser.finish(&events);
  Parsed out;
  for (const auto& ev : events) {
    using Kind = ToolCallParser::Event::Kind;
    if (ev.kind == Kind::kReasoning) out.reasoning += ev.text;
    if (ev.kind == Kind::kReasoningClosed) ++out.reasoning_blocks;
    if (ev.kind == Kind::kContent) out.content += ev.text;
    if (ev.kind == Kind::kToolCall) out.calls.push_back(ev.call);
  }
  require(parser.calls() == static_cast<int>(out.calls.size()), "the parser's call counter");
  return out;
}

DGPP_TEST(inkling_tools_markers_select_the_typed_block_format) {
  const ChatMarkers markers = markers_at(256);
  require(markers.inkling_blocks() && markers.reasoning_available(), "the block markers are complete");
  require(markers.tool_format() == ToolFormat::kInkling, "the block markers select the Inkling format");
  ChatMarkers partial = markers;
  partial.end_message = {};
  require(!partial.inkling_blocks() && partial.tool_format() != ToolFormat::kInkling,
          "a vocabulary without the end marker is not the Inkling format");
}

DGPP_TEST(inkling_tools_reply_parses_at_every_token_split) {
  const std::string args = R"({"city":"Paris","days":3,"note":"a \"quoted\" {brace}"})";
  const std::string reply = M + TH + "Check the weather first." + E + M + TX + "One moment. " + E + M + "get_weather" + TJ +
                            R"({"name":"get_weather","args":)" + args + "}" + E + M + "get_time" + TJ +
                            R"({ "args" : {} , "name" : "get_time" })" + E + M + TX + "Done." + E;
  for (size_t chunk = 1; chunk <= 96; ++chunk) {
    const Parsed p = parse(reply, chunk);
    const std::string at = ", chunk " + std::to_string(chunk);
    require(p.reasoning == "Check the weather first." && p.reasoning_blocks == 1, "the thinking block" + at);
    require(p.content == "One moment. Done.", "the text blocks: '" + p.content + "'" + at);
    require(p.calls.size() == 2, "two calls" + at);
    require(p.calls[0].name == "get_weather" && same_json(p.calls[0].arguments, args),
            "the first call: " + p.calls[0].arguments + at);
    require(p.calls[1].name == "get_time" && p.calls[1].arguments == "{}", "the second call" + at);
  }
}

DGPP_TEST(inkling_tools_irregular_replies_keep_their_text) {
  for (const size_t chunk : {size_t{1}, size_t{5}, size_t{10000}}) {
    const std::string at = ", chunk " + std::to_string(chunk);
    {
      const Parsed p = parse(TH + "plan" + E + TX + "answer" + E, chunk);
      require(p.reasoning == "plan" && p.content == "answer" && p.calls.empty(), "headerless blocks" + at);
    }
    {
      // A block left open when the next one starts, and a reply cut inside text.
      const Parsed p = parse(M + TH + "plan" + M + TX + "cut off he", chunk);
      require(p.reasoning == "plan" && p.reasoning_blocks == 1 && p.content == "cut off he", "unclosed blocks" + at);
    }
    {
      const Parsed p = parse(M + TH + "a" + E + M + TX + "b" + E + M + TH + "c" + E + M + TX + "d" + E, chunk);
      require(p.reasoning == "ac" && p.reasoning_blocks == 2 && p.content == "bd", "interleaved blocks" + at);
    }
    {
      // The body's own name wins over the header's, and a call without args takes none.
      const Parsed p = parse(M + "x" + TJ + R"({"name":"noargs"})" + E + M + TJ + R"({"name":"ping","args":{"n":1.50}})" + E, chunk);
      require(p.calls.size() == 2 && p.content.empty(), "two calls and no text" + at);
      require(p.calls[0].name == "noargs" && p.calls[0].arguments == "{}", "a call without args" + at);
      require(p.calls[1].name == "ping" && same_json(p.calls[1].arguments, R"({"n":1.5})"), "a headerless call" + at);
    }
    {
      // A body that is not a call is the text the model wrote.
      const Parsed bad = parse(M + "f" + TJ + R"({"name":"f","args":{"x":)" + E + M + TX + "sorry" + E, chunk);
      require(bad.calls.empty() && bad.content == std::string(R"({"name":"f","args":{"x":)") + "sorry",
              "a malformed call is text" + at);
      const Parsed arr = parse(TJ + R"(["f"])" + E, chunk);
      require(arr.calls.empty() && arr.content == R"(["f"])", "a body that is not an object is text" + at);
      const Parsed anonymous = parse(M + TJ + R"({"args":{"x":1}})" + E, chunk);
      require(anonymous.calls.empty() && anonymous.content == R"({"args":{"x":1}})", "a body without a name is text" + at);
      const Parsed cut = parse(M + "f" + TJ + R"({"name":"f","ar)", chunk);
      require(cut.calls.empty() && cut.content == R"({"name":"f","ar)", "a cut call is text" + at);
    }
  }
}

// Bytes 1..255 as single ids, the seven markers at 256..262, the end of
// sampling at 263, then any merged pieces.
constexpr int64_t kEos = 263;

GrammarVocab vocab(std::vector<std::string> extra = {}) {
  std::vector<std::string> texts(264);
  for (int i = 1; i < 256; ++i) texts[i] = std::string(1, static_cast<char>(i));
  for (size_t k = 0; k < kMarkerTexts.size(); ++k) texts[256 + k] = kMarkerTexts[k];
  texts.insert(texts.end(), extra.begin(), extra.end());
  const auto size = texts.size();
  return GrammarVocab(std::move(texts), markers_at(256), {kEos}, size);
}

std::vector<int64_t> byte_ids(const std::string& text) {
  std::vector<int64_t> ids;
  for (size_t at = 0; at < text.size();) {
    const int k = marker_at(text, at);
    if (k < 0) {
      ids.push_back(static_cast<unsigned char>(text[at++]));
      continue;
    }
    ids.push_back(256 + k);
    at += kMarkerTexts[k].size();
  }
  ids.push_back(kEos);
  return ids;
}

// The index of the first refused id, or ids.size(). The mask and the
// pointwise rule must agree on every id at every position.
size_t accepts(const GrammarVocab& v, const GrammarSpec& spec, const std::vector<int64_t>& ids) {
  GrammarState state(&v, spec, false);
  for (size_t i = 0; i < ids.size(); ++i) {
    TokenMask mask;
    state.mask(&mask);
    for (int id = 0; id < v.vocab_size(); ++id)
      require(mask.allows(id) == state.allows(id),
              "the mask and the pointwise rule differ in " + std::string(state.state_name()) + " at id " + std::to_string(id));
    if (!state.allows(ids[i])) return i;
    state.advance(ids[i]);
    require(state.active() || i + 1 == ids.size(), "an allowed id keeps the grammar alive");
  }
  return ids.size();
}

GrammarSpec tools_spec(GrammarSpec::Mode mode, bool parallel = true) {
  const auto weather = dgpp::minijson::parse(
      R"({"name":"get_weather","strict":true,"parameters":{"type":"object","properties":{"city":{"type":"string"},)"
      R"("days":{"type":"integer","minimum":0,"maximum":14},"units":{"type":"string","enum":["metric","imperial"]},)"
      R"("where":{"$ref":"#/$defs/place"}},"$defs":{"place":{"type":"object","properties":{"lat":{"type":"number"}},)"
      R"("required":["lat"],"additionalProperties":false}},"required":["city"],"additionalProperties":false}})");
  const auto clock = dgpp::minijson::parse(R"({"name":"get_time","parameters":{"type":"object","properties":{}}})");
  GrammarSpec spec;
  spec.mode = mode;
  spec.parallel = parallel;
  spec.tools.push_back(grammar_tool_from_function(weather.root, nullptr, nullptr, true));
  spec.tools.push_back(grammar_tool_from_function(clock.root, nullptr, nullptr, true));
  return spec;
}

std::string call(const std::string& name, const std::string& args, bool header = true) {
  return M + (header ? name : "") + TJ + R"({"name":")" + name + R"(","args":)" + args + "}" + E;
}

DGPP_TEST(inkling_tools_grammar_enforces_each_mode) {
  const GrammarVocab v = vocab();
  using Mode = GrammarSpec::Mode;
  const auto accepted = [&](const GrammarSpec& spec, const std::string& text) {
    const std::vector<int64_t> ids = byte_ids(text);
    return accepts(v, spec, ids) == ids.size();
  };
  const std::string think = M + TH + "Let me check the weather." + E;
  const std::string answer = M + TX + "It is sunny." + E;
  const std::string paris = call("get_weather", R"({"city":"Paris","days":3})");
  const std::string clock = call("get_time", "{}");

  const auto autos = tools_spec(Mode::kAuto);
  require(accepted(autos, think + answer), "auto: a plain answer");
  require(accepted(autos, think + paris), "auto: a call");
  require(accepted(autos, answer + paris + clock), "auto: text and two calls");
  require(accepted(autos, call("get_weather", R"({ "units" : "metric" , "city" : "Oslo" })")), "auto: JSON whitespace and key order");
  require(accepted(autos, call("get_weather", R"({"city":"Oslo","where":{"lat":59.9}})")), "auto: an argument behind a $ref");
  require(!accepted(autos, call("get_weather", R"({"city":"Oslo","where":{"lon":10.7}})")), "auto: a $ref argument off its schema");
  require(!accepted(autos, call("get_weather", R"({"city":"Paris"})", false)), "auto: a call block follows a named header");
  require(!accepted(autos, call("get_weather", R"({"days":3})")), "auto: a required argument is missing");
  require(!accepted(autos, call("get_weather", R"({"city":"Paris","days":"three"})")), "auto: an argument of the wrong type");
  require(!accepted(autos, call("get_weather", R"({"city":"Paris","days":15})")), "auto: an integer past its maximum");
  require(!accepted(autos, call("get_weather", R"({"city":"Paris","units":"kelvin"})")), "auto: a value outside the enum");
  require(!accepted(autos, call("get_weather", R"({"city":"Paris","extra":1})")), "auto: an undeclared key");
  require(!accepted(autos, call("launch", "{}")), "auto: a tool that was not declared");
  require(!accepted(autos, M + "get_time" + TJ + R"({"name":"get_weather","args":{"city":"x"}})" + E),
          "auto: the body names another tool than its header");
  require(!accepted(autos, M + "get_time" + TJ + " {\"name\":\"get_time\",\"args\":{}}" + E), "auto: whitespace before a call's body");
  require(!accepted(autos, M + TH + "x" + E + M + "get_weather" + TX + "y" + E), "auto: a named header opens a call only");

  const auto single = tools_spec(Mode::kAuto, false);
  require(accepted(single, paris + answer), "auto, one call: text after the call");
  require(!accepted(single, paris + clock), "auto, one call: a second call");

  const auto none = tools_spec(Mode::kForbidCalls);
  require(accepted(none, think + answer) && !accepted(none, think + paris), "none: text only");

  const auto required = tools_spec(Mode::kRequired);
  require(!accepted(required, think + answer), "required: a turn without a call");
  require(!accepted(required, answer + paris), "required: text before the call");
  require(accepted(required, think + paris), "required: reasoning then the call");
  require(!accepted(required, think + think + paris), "required: a second thinking block before the call");
  require(!accepted(required, paris + answer), "required: text after the call");
  require(accepted(required, paris + clock), "required: several calls");
  require(!accepted(tools_spec(Mode::kRequired, false), paris + clock), "required, one call: a second call");

  auto named = tools_spec(Mode::kNamed);
  named.named = "get_time";
  require(accepted(named, think + clock), "named: the named call");
  require(!accepted(named, paris), "named: another tool");
  require(!accepted(named, clock + answer), "named: text after the call");
  require(!accepted(named, clock + clock), "named: a second call");
  require(!accepted(named, answer), "named: no call");
}

DGPP_TEST(inkling_tools_grammar_json_answers_are_one_text_block) {
  const GrammarVocab v = vocab();
  using Mode = GrammarSpec::Mode;
  const auto accepted = [&](const GrammarSpec& spec, const std::string& text) {
    const std::vector<int64_t> ids = byte_ids(text);
    return accepts(v, spec, ids) == ids.size();
  };
  const std::string think = M + TH + "Scoring." + E;
  GrammarSpec json;
  json.mode = Mode::kJson;
  json.json_schema =
      R"({"type":"object","properties":{"score":{"type":"integer","minimum":0,"maximum":10}},"required":["score"],"additionalProperties":false})";
  require(accepted(json, think + M + TX + R"({"score": 7})" + E), "json: reasoning then the answer");
  require(accepted(json, TX + R"({"score":0})" + E), "json: the answer alone");
  require(!accepted(json, M + TX + "  " + R"({"score":0})" + E), "json: whitespace before the value");
  require(!accepted(json, think + think + M + TX + R"({"score":0})" + E), "json: a second thinking block before the answer");
  require(!accepted(json, M + TX + R"({"score": 11})" + E), "json: a value outside the schema");
  require(!accepted(json, M + TX + "seven" + E), "json: text that is not JSON");
  require(!accepted(json, think), "json: no answer");
  require(!accepted(json, M + TX + R"({"score": 7})" + E + M + TX + "More." + E), "json: text after the answer");
  GrammarSpec object;
  object.mode = Mode::kJson;
  require(accepted(object, M + TX + R"({"anything": [1, "two", null]})" + E), "json_object: any object");
  require(!accepted(object, M + TX + "[1]" + E), "json_object: not an object");

  auto either = tools_spec(Mode::kJsonOrTools);
  either.json_schema = json.json_schema;
  const std::string paris = call("get_weather", R"({"city":"Paris"})");
  require(accepted(either, think + M + TX + R"({"score": 3})" + E), "json or tools: the answer");
  require(accepted(either, think + paris), "json or tools: a call");
  require(!accepted(either, paris + M + TX + R"({"score": 3})" + E), "json or tools: an answer after a call");
  require(!accepted(either, think), "json or tools: neither");
}

DGPP_TEST(inkling_tools_grammar_reads_merged_tokens_and_copies) {
  // A call written in merged pieces of every size: a piece that spans a
  // JSON boundary is admitted exactly when its bytes are.
  const std::string body = R"({"name":"get_weather","args":{"city":"Paris","days":3}})";
  const auto spec = tools_spec(GrammarSpec::Mode::kRequired);
  for (const size_t chunk : {size_t{1}, size_t{2}, size_t{7}, size_t{17}, size_t{31}, size_t{10000}}) {
    std::vector<std::string> pieces;
    const std::string name = "get_weather";
    for (size_t i = 0; i < name.size(); i += chunk) pieces.push_back(name.substr(i, chunk));
    const size_t name_pieces = pieces.size();
    for (size_t i = 0; i < body.size(); i += chunk) pieces.push_back(body.substr(i, chunk));
    const GrammarVocab v = vocab(pieces);
    std::vector<int64_t> ids = {256};
    for (size_t i = 0; i < name_pieces; ++i) ids.push_back(264 + static_cast<int64_t>(i));
    ids.push_back(259);
    for (size_t i = name_pieces; i < pieces.size(); ++i) ids.push_back(264 + static_cast<int64_t>(i));
    ids.push_back(262);
    ids.push_back(kEos);
    GrammarState state(&v, spec, false);
    require(!state.allows(kEos), "a required call cannot be skipped");
    for (size_t i = 0; i < ids.size(); ++i) {
      require(state.allows(ids[i]), "a merged piece is refused in " + std::string(state.state_name()) + ", chunk " +
                                        std::to_string(chunk) + ", position " + std::to_string(i));
      const GrammarState before = state;
      state.advance(ids[i]);
      require(before.allows(ids[i]), "advancing a copy leaves the original where it was");
    }
    require(std::string(state.state_name()) == "done", "the turn ends after its call");
  }
  // A merged piece that would run past the body's end is refused.
  const GrammarVocab v = vocab({body + "x"});
  GrammarState state(&v, spec, false);
  for (const int64_t id : {int64_t{256}}) state.advance(id);
  for (const unsigned char c : std::string("get_weather")) state.advance(c);
  state.advance(259);
  require(!state.allows(264), "a piece that runs past the call's body");
}

}  // namespace
