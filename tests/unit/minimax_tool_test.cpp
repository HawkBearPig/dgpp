#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/minimax_tool_grammar.hpp"
#include "text/tool_grammar.hpp"
#include "text/tool_parser.hpp"

namespace {
using namespace dgpp::text;
const std::string ns = "]<]minimax[>[";
std::string tag(const std::string& text) {
  return ns + "<" + text + ">";
}
void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

ToolSchemas schemas() {
  // The DOM and its backing text die here: recursive typing must be owned.
  const std::string text = R"([{"name":"f","parameters":{"type":"object","properties":{
    "code":{"type":"string"},"n":{"type":"integer"},"empty":{"type":"array"},
    "obj":{"type":"object","properties":{"code":{"type":"string"},"empty":{"type":"object"}}},
    "items":{"type":"array","items":{"type":"object","properties":{"ok":{"type":"boolean"}}}}
  }}}])";
  return ToolSchemas(dgpp::minijson::parse(text).root);
}

struct Result {
  std::string content;
  std::vector<ToolCallParser::Call> calls;
  std::vector<ToolCallParser::Event> events;
  std::vector<std::string> pieces;
};

Result parse(const std::string& text, size_t chunk) {
  Result result;
  for (size_t i = 0; i < text.size(); i += chunk) result.pieces.push_back(text.substr(i, chunk));
  ChatMarkers markers;
  markers.minimax_namespace = {100001, ns};
  markers.tool_call_open = {100002, "<tool_call>"};
  markers.tool_call_close = {100003, "</tool_call>"};
  ToolCallParser::Options options;
  options.start_in_reasoning = false;
  options.track_tokens = true;
  ToolCallParser parser(
      markers,
      [&](const std::vector<int64_t>& ids) {
        std::string out;
        for (auto id : ids) out += result.pieces.at(id);
        return out;
      },
      schemas(), options);
  for (size_t i = 0; i < result.pieces.size(); ++i) parser.feed(i, &result.events);
  parser.finish(&result.events);
  for (const auto& event : result.events) {
    if (event.kind == ToolCallParser::Event::Kind::kToolCall)
      result.calls.push_back(event.call);
    else
      result.content += event.text;
  }
  require(parser.calls() == static_cast<int>(result.calls.size()), "call counter");
  return result;
}

DGPP_TEST(minimax_tools_recursive_schema_and_all_token_splits) {
  const std::string block =
      tag("tool_call") + "\n" + tag("invoke name=\"f\"") + tag("code") + "123<&\n\"" +
      tag("/code") + tag("n") + "-12" + tag("/n") + tag("empty") + tag("/empty") + tag("obj") +
      tag("code") + "false" + tag("/code") + tag("empty") + tag("/empty") + tag("/obj") +
      tag("items") + tag("item") + tag("ok") + "true" + tag("/ok") + tag("/item") + tag("/items") +
      tag("/invoke") + "\n" + tag("invoke name=\"f\"") + tag("/invoke") + "\n" + tag("/tool_call");
  for (size_t chunk = 1; chunk <= block.size() + 12; ++chunk) {
    const auto result = parse("before" + block + "after", chunk);
    require(result.content == "beforeafter",
            "content around block, chunk " + std::to_string(chunk));
    require(result.calls.size() == 2, "two atomic calls");
    require(result.calls[0].name == "f" && result.calls[1].arguments == "{}",
            "call names/empty args");
    require(
        result.calls[0].arguments ==
            R"({"code": "123<&\n\"", "n": -12, "empty": [], "obj": {"code": "false", "empty": {}}, "items": [{"ok": true}]})",
        "recursive argument typing: " + result.calls[0].arguments);
  }
}

DGPP_TEST(minimax_tools_malformed_atomic_and_token_provenance) {
  const auto prefix = tag("tool_call") + "\n" + tag("invoke name=\"f\"");
  const auto suffix = tag("/invoke") + "\n" + tag("/tool_call");
  const std::vector<std::string> bad = {
      prefix + tag("n") + "wrong" + tag("/n") + suffix,
      prefix + tag("n") + "1" + tag("/wrong") + suffix,
      prefix + tag("obj") + tag("a") + "1" + tag("/a") + tag("a") + "2" + tag("/a") + tag("/obj") +
          suffix,
      prefix + tag("items") + tag("wrong") + "1" + tag("/wrong") + tag("/items") + suffix,
      prefix + tag("/invoke") + tag("invoke name=\"f\"") + tag("n") + "x" + tag("/n") + suffix,
      prefix + tag("n") + "1</n>" + suffix,
      prefix + tag("code") + "unfinished",
      "orphan" + ns,
      tag("tool_call") + tag("/tool_call"),
  };
  for (const auto& block : bad) {
    const std::string input = "before" + block + "after";
    for (size_t chunk : {1u, 7u, 31u, 10000u}) {
      const auto result = parse(input, chunk);
      require(result.calls.empty() && result.content == input,
              "malformed block lost bytes or emitted a partial call");
      std::vector<size_t> covered(result.pieces.size());
      for (const auto& event : result.events) {
        size_t offset = 0;
        for (const auto& span : event.tokens) {
          require(span.begin == offset && span.end > span.begin && span.end <= event.text.size(),
                  "provenance gap");
          require(event.text.substr(span.begin, span.end - span.begin) ==
                      result.pieces[span.token].substr(covered[span.token], span.end - span.begin),
                  "provenance bytes changed");
          covered[span.token] += span.end - span.begin;
          offset = span.end;
        }
        require(offset == event.text.size(), "missing provenance");
      }
      for (size_t i = 0; i < covered.size(); ++i)
        require(covered[i] == result.pieces[i].size(), "lost token bytes");
    }
  }
}

GrammarSpec grammar_spec() {
  const auto def = dgpp::minijson::parse(R"({"name":"f","strict":true,"parameters":{
    "type":"object","additionalProperties":false,"required":["obj","items"],"properties":{
      "obj":{"type":"object","additionalProperties":false,"required":["code","n"],"properties":{
        "code":{"type":"string","enum":["123","abc]def"]},"n":{"type":"integer","minimum":10,"maximum":20}}},
      "items":{"type":"array","minItems":1,"maxItems":2,"items":{"type":"boolean"}}
    }}})");
  GrammarSpec spec;
  spec.mode = GrammarSpec::Mode::kRequired;
  spec.tools.push_back(grammar_tool_from_function(def.root, nullptr, nullptr, true));
  return spec;
}

GrammarVocab vocab(std::vector<std::string> extra = {}) {
  std::vector<std::string> texts(262);
  for (int i = 1; i < 256; ++i) texts[i] = std::string(1, static_cast<char>(i));
  texts[256] = ns;
  texts[257] = "<tool_call>";
  texts[258] = "</tool_call>";
  texts[259] = "<mm:think>";
  texts[260] = "</mm:think>";
  texts.insert(texts.end(), extra.begin(), extra.end());
  ChatMarkers markers;
  markers.minimax_namespace = {256, ns};
  markers.tool_call_open = {257, texts[257]};
  markers.tool_call_close = {258, texts[258]};
  markers.think_open = {259, texts[259]};
  markers.think_close = {260, texts[260]};
  const auto size = texts.size();
  return GrammarVocab(std::move(texts), markers, {261}, size);
}

std::string valid_block() {
  return tag("tool_call") + "\n" + tag("invoke name=\"f\"") + tag("obj") + tag("code") + "123" +
         tag("/code") + tag("n") + "12" + tag("/n") + tag("/obj") + tag("items") + tag("item") +
         "true" + tag("/item") + tag("/items") + tag("/invoke") + "\n" + tag("/tool_call");
}

void advance_text(GrammarState& state, const std::string& text) {
  for (unsigned char c : text) {
    require(state.allows(c), "rejected byte in " + std::string(state.state_name()) + ": " + text);
    state.advance(c);
    require(state.active(), "grammar became inactive");
  }
}

DGPP_TEST(minimax_tools_grammar_nested_schema_every_split_and_copy) {
  const auto block = valid_block();
  for (size_t chunk : {1u, 2u, 7u, 17u, 31u, 10000u}) {
    std::vector<std::string> pieces;
    for (size_t i = 0; i < block.size(); i += chunk) pieces.push_back(block.substr(i, chunk));
    auto v = vocab(pieces);
    GrammarState state(&v, grammar_spec(), false);
    require(!state.allows(261), "required grammar allowed early EOS");
    for (size_t i = 0; i < pieces.size(); ++i) {
      TokenMask mask;
      state.mask(&mask);
      for (int id = 0; id < v.vocab_size(); ++id)
        require(mask.allows(id) == state.allows(id), "mask differs from pointwise grammar");
      const int id = 262 + i;
      require(state.allows(id),
              "rejected merged token in " + std::string(state.state_name()) + ": " + pieces[i]);
      auto rollback = state;
      state.advance(id);
      require(rollback.allows(id), "advancing a speculative copy changed its parent");
      require(state.active(), "grammar died after valid token");
    }
    require(state.allows(261), "complete call cannot finish");
    require(!state.allows('x'), "content allowed after call block");
  }
}

DGPP_TEST(minimax_tools_grammar_rejects_missing_duplicate_and_wrong_typed_values) {
  auto v = vocab();
  GrammarState state(&v, grammar_spec(), false);
  advance_text(state, tag("tool_call") + tag("invoke name=\"f\""));
  auto missing = state;
  advance_text(missing, ns + "<");
  require(!missing.allows('/'), "closed call with missing required keys");
  advance_text(state, tag("obj") + tag("code"));
  require(!state.allows('z'), "invalid enum prefix");
  advance_text(state, "abc]def" + tag("/code") + tag("n"));
  require(!state.allows('x'), "non-number in integer");
  advance_text(state, "1");
  require(!state.allows(256), "number outside bound was closable");
  advance_text(state, "2" + tag("/n"));
  auto duplicate = state;
  advance_text(duplicate, ns + "<");
  require(!duplicate.allows('n'), "duplicate property offered");
  advance_text(state, tag("/obj") + tag("items"));
  auto empty = state;
  advance_text(empty, ns + "<");
  require(!empty.allows('/'), "array shorter than minItems was closable");
  advance_text(state, tag("item") + "true" + tag("/item") + tag("item") + "false" + tag("/item"));
  auto over = state;
  advance_text(over, ns + "<");
  require(!over.allows('i'), "array exceeded maxItems");
  advance_text(state, tag("/items") + tag("/invoke") + tag("/tool_call"));
  require(state.allows(261), "validated call EOS");
}

DGPP_TEST(minimax_tools_grammar_json_reasoning_named_and_forbidden) {
  auto v = vocab();
  auto spec = grammar_spec();
  spec.mode = GrammarSpec::Mode::kJsonOrTools;
  spec.json_schema =
      R"({"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false})";
  GrammarState json(&v, spec, false, true);
  require(json.allows(260), "adaptive no-thinking close should be allowed");
  json.advance(260);
  advance_text(json, R"({"answer":42})");
  require(json.allows(261), "JSON answer cannot end");
  GrammarState tools(&v, spec, false, true);
  tools.advance(259);
  advance_text(tools, "plan");
  tools.advance(260);
  advance_text(tools, valid_block());
  require(tools.allows(261), "tool alternative cannot end");
  spec.mode = GrammarSpec::Mode::kNamed;
  spec.named = "f";
  GrammarState named(&v, spec, false);
  const auto block = valid_block();
  advance_text(named, block.substr(0, block.size() - tag("/tool_call").size()));
  advance_text(named, ns + "<");
  require(!named.allows('i') && named.allows('/'), "named tool admitted a second invoke");
  spec.mode = GrammarSpec::Mode::kForbidCalls;
  GrammarState forbidden(&v, spec, false);
  advance_text(forbidden, "ordinary text " + ns);
  require(!forbidden.allows(257), "forbidden call opened");
  require(forbidden.allows(261), "forbidden mode cannot end content");
}

DGPP_TEST(minimax_tools_grammar_raw_scalar_cannot_inject_json_structure) {
  auto v = vocab({"true,false", "true]x", "12,", "12}", "true ", "123<&\n\""});
  auto spec = grammar_spec();
  GrammarState state(&v, spec, false);
  advance_text(state, tag("tool_call") + tag("invoke name=\"f\"") + tag("items") + tag("item"));
  for (int id = 262; id < 266; ++id) require(!state.allows(id), "raw scalar injected JSON syntax");
  require(state.allows(266), "trailing scalar whitespace should be valid");
  TokenMask mask;
  state.mask(&mask);
  for (int id = 0; id < v.vocab_size(); ++id)
    require(mask.allows(id) == state.allows(id), "scalar table mask differs from simulation");
}

DGPP_TEST(minimax_tools_nullable_references_and_open_nested_keys) {
  const auto def = dgpp::minijson::parse(R"({"name":"f","strict":true,"parameters":{
    "type":"object","required":["obj","items"],"additionalProperties":false,
    "$defs":{"code":{"type":"string","pattern":"^[0-9]+$"}},"properties":{
      "obj":{"type":"object","required":["code"],"properties":{"code":{"$ref":"#/$defs/code"}},
             "additionalProperties":{"type":"integer"}},
      "items":{"type":["array","null"],"items":{"type":"boolean"}}
    }}})");
  GrammarSpec spec;
  spec.mode = GrammarSpec::Mode::kRequired;
  spec.tools.push_back(grammar_tool_from_function(def.root, nullptr, nullptr, true));
  auto v = vocab();
  GrammarState state(&v, spec, false);
  const auto block = tag("tool_call") + tag("invoke name=\"f\"") + tag("obj") + tag("code") +
                     "123" + tag("/code") + tag("extra") + "42" + tag("/extra") + tag("/obj") +
                     tag("items") + tag("/items") + tag("/invoke") + tag("/tool_call");
  for (unsigned char c : block) {
    TokenMask mask;
    state.mask(&mask);
    for (int id = 0; id < v.vocab_size(); ++id)
      require(mask.allows(id) == state.allows(id), "open-key mask differs from simulation");
    require(state.allows(c), "ref/nullable/open-key XML rejected");
    state.advance(c);
  }
  require(state.allows(261), "ref/nullable call cannot end");
  const std::string tools = "[" + dgpp::text::json_text_of(def.root) + "]";
  ToolCallParser::Options opts;
  opts.start_in_reasoning = false;
  ToolCallParser parser(
      v.markers(),
      [](const std::vector<int64_t>& ids) {
        std::string text;
        for (auto id : ids) text += static_cast<char>(id);
        return text;
      },
      ToolSchemas(dgpp::minijson::parse(tools).root), opts);
  std::vector<ToolCallParser::Event> events;
  for (unsigned char c : block) parser.feed(c, &events);
  parser.finish(&events);
  require(events.size() == 1 &&
              events[0].call.arguments == R"({"obj": {"code": "123", "extra": 42}, "items": []})",
          "schema references or nullable empty containers lost their types");
}

DGPP_TEST(minimax_tools_nullable_enum_and_unrepresentable_schema_refusal) {
  auto v = vocab();
  const auto make = [&](const std::string& property) {
    const auto definition =
        R"({"name":"f","strict":true,"parameters":{"type":"object","properties":{"x":)" + property +
        R"(},"required":["x"],"additionalProperties":false}})";
    const auto def = dgpp::minijson::parse(definition);
    GrammarSpec spec;
    spec.mode = GrammarSpec::Mode::kRequired;
    spec.tools.push_back(grammar_tool_from_function(def.root, nullptr, nullptr, true));
    return spec;
  };
  GrammarState state(&v, make(R"({"type":["string","null"],"enum":["yes",null]})"), false);
  advance_text(state, tag("tool_call") + tag("invoke name=\"f\"") + tag("x"));
  require(!state.allows('n'), "nullable enum admitted the literal string null");
  advance_text(state, "yes" + tag("/x") + tag("/invoke") + tag("/tool_call"));
  require(state.allows(261), "nullable enum cannot finish");
  for (
      const auto& schema :
      {R"({"type":"null"})",
       R"({"type":"object","properties":{"required_null":{"type":"null"}},"required":["required_null"]})",
       R"({"anyOf":[{"type":"string"},{"type":"integer"}]})"}) {
    bool refused = false;
    try {
      MiniMaxToolGrammar invalid(nullptr, make(schema));
    } catch (const std::invalid_argument& e) {
      refused = std::string(e.what()).starts_with("parameters.properties.x:");
    }
    require(refused, "unrepresentable schema not refused by its field");
  }
}
}  // namespace
