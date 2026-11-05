#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "text/chat_template.hpp"
#include "text/tokenizer.hpp"
#include "text/tool_grammar.hpp"
#include "text/tool_parser.hpp"

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
std::string hex(uint64_t value) {
  char out[32];
  std::snprintf(out, sizeof(out), "%016llx", static_cast<unsigned long long>(value));
  return out;
}
dgpp::text::Value omit_null_fields(const dgpp::text::Value& value) {
  using V = dgpp::text::Value;
  if (const auto* members = value.as_members()) {
    V::Members out;
    for (const auto& [key, item] : *members)
      if (item.kind() != V::Kind::Null) out.emplace_back(key, omit_null_fields(item));
    return V::map_value(std::move(out));
  }
  if (value.kind() == V::Kind::List) {
    std::vector<V> out;
    for (const auto& item : value.as_list()) out.push_back(omit_null_fields(item));
    return V::list_value(std::move(out));
  }
  return value;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "usage: minimax_prompt_test GOLDENS");
    std::ifstream input(argv[1]);
    std::string line;
    require(static_cast<bool>(std::getline(input, line)), "missing golden header");
    const auto header = dgpp::minijson::parse_document(line);
    std::string error;
    const auto snapshot =
        dgpp::hf::model_dir(std::string(header.root.at("model").as_string()), &error);
    if (snapshot.empty()) {
      std::fprintf(stderr, "MiniMax checkpoint unavailable: %s\n", error.c_str());
      return 2;
    }
    const auto tpl = dgpp::text::ChatTemplate::load(snapshot + "/chat_template.jinja");
    const auto tok = dgpp::text::Tokenizer::load(snapshot + "/tokenizer.json");
    require(hex(tpl.source_hash()) == header.root.at("template_hash").as_string(),
            "template revision mismatch");
    require(hex(tok.revision_hash()) == header.root.at("tokenizer_revision").as_string(),
            "tokenizer revision mismatch");
    size_t count = 0;
    while (std::getline(input, line)) {
      const auto record = dgpp::minijson::parse_document(line);
      const auto& r = record.root;
      const std::string name(r.at("name").as_string());
      if (const auto* text = r.find("text")) {
        std::vector<int64_t> ids;
        for (const auto& value : r.at("ids").items()) ids.push_back(value.as_int());
        require(tok.encode(text->as_string()) == ids, name + ": tokenizer differs from HF");
        require(tok.decode(ids, false) == r.at("normalized").as_string(),
                name + ": NFC round trip");
        std::string incremental;
        for (auto id : ids) incremental += tok.decode({id}, true);
        require(tok.decode(ids, true) == incremental, name + ": incremental decode");
        ++count;
        continue;
      }
      const auto rendered = tpl.render(dgpp::text::Value::from_minijson(r.at("kwargs")));
      const std::string expected(r.at("render").as_string());
      if (rendered != expected) {
        size_t i = 0;
        while (i < rendered.size() && i < expected.size() && rendered[i] == expected[i]) ++i;
        throw std::runtime_error(name + ": prompt differs at byte " + std::to_string(i));
      }
      std::vector<int64_t> ids;
      for (const auto& value : r.at("ids").items()) ids.push_back(value.as_int());
      require(tok.encode(rendered) == ids, name + ": tokenizer differs from HF");
      if (name.starts_with("tool_") || name == "parallel_calls") {
        const std::string open = "]<]minimax[>[<tool_call>";
        const std::string close = "]<]minimax[>[</tool_call>";
        const auto start = rendered.rfind(open), end = rendered.find(close, start);
        require(start != std::string::npos && end != std::string::npos,
                name + ": missing call block");
        const auto& kwargs = r.at("kwargs");
        const dgpp::minijson::Value* calls = nullptr;
        for (const auto& message : kwargs.at("messages").items())
          if (const auto* c = message.find("tool_calls")) calls = c;
        require(calls != nullptr, name + ": missing expected calls");
        dgpp::text::ToolCallParser::Options opts;
        opts.start_in_reasoning = false;
        dgpp::text::ToolCallParser parsed(
            dgpp::text::ChatMarkers::from_tokenizer(tok),
            [&](const std::vector<int64_t>& token_ids) { return tok.decode(token_ids, true); },
            dgpp::text::ToolSchemas(kwargs.at("tools")), opts);
        std::vector<dgpp::text::ToolCallParser::Event> events;
        for (auto id : tok.encode(rendered.substr(start, end + close.size() - start)))
          parsed.feed(id, &events);
        parsed.finish(&events);
        require(events.size() == calls->items().size(), name + ": call count differs");
        for (size_t i = 0; i < events.size(); ++i) {
          const auto& expected_call = calls->items()[i].at("function");
          require(events[i].kind == dgpp::text::ToolCallParser::Event::Kind::kToolCall &&
                      events[i].call.name == expected_call.at("name").as_string(),
                  name + ": parsed wrong call");
          const auto args = dgpp::text::Value::from_minijson(
              dgpp::minijson::parse(events[i].call.arguments).root);
          require(args.equals(omit_null_fields(
                      dgpp::text::Value::from_minijson(expected_call.at("arguments")))),
                  name + ": recursive arguments differ: " + events[i].call.arguments);
        }
      }
      ++count;
    }
    require(count == static_cast<size_t>(header.root.at("cases").as_int()), "incomplete corpus");
    auto markers = dgpp::text::ChatMarkers::from_tokenizer(tok);
    require(markers.think_open.id == 200059 && markers.think_close.id == 200060,
            "MiniMax reasoning must use mm:think despite canonical think tokens");
    markers.model_opens_thinking = true;
    const auto prompt = tok.encode("]~b]ai\n");
    require(
        !markers.prompt_opens_thinking(prompt) && markers.prompt_leaves_thinking_to_model(prompt),
        "adaptive generation must leave reasoning to the model");
    dgpp::text::ToolCallParser::Options options;
    options.start_in_reasoning = false;
    options.model_may_open_thinking = true;
    dgpp::text::ToolCallParser parser(
        markers, [&](const std::vector<int64_t>& ids) { return tok.decode(ids); }, {}, options);
    std::vector<dgpp::text::ToolCallParser::Event> events;
    for (auto id : tok.encode("<mm:think>Consider.</mm:think>Answer.")) parser.feed(id, &events);
    parser.finish(&events);
    std::string reasoning, content;
    for (const auto& event : events) {
      using K = dgpp::text::ToolCallParser::Event::Kind;
      if (event.kind == K::kReasoning) reasoning += event.text;
      if (event.kind == K::kContent) content += event.text;
    }
    require(reasoning == "Consider." && content == "Answer.", "reasoning/content split differs");
    dgpp::text::ToolCallParser no_thought(
        markers, [&](const std::vector<int64_t>& ids) { return tok.decode(ids); }, {}, options);
    events.clear();
    for (auto id : tok.encode("</mm:think>你好！")) no_thought.feed(id, &events);
    no_thought.finish(&events);
    content.clear();
    for (const auto& event : events)
      if (event.kind == dgpp::text::ToolCallParser::Event::Kind::kContent) content += event.text;
    require(content == "你好！", "adaptive no-thinking delimiter must not leak into content");
    const auto def = dgpp::minijson::parse(
        R"({"name":"f","strict":true,"parameters":{"type":"object","properties":{
      "city":{"type":"string"},"days":{"type":"integer","minimum":1,"maximum":7}},
      "required":["city","days"],"additionalProperties":false}})");
    dgpp::text::GrammarSpec spec;
    spec.mode = dgpp::text::GrammarSpec::Mode::kRequired;
    spec.tools.push_back(dgpp::text::grammar_tool_from_function(def.root, nullptr, nullptr, true));
    auto vocab = dgpp::text::GrammarVocab::from_tokenizer(tok, {200020}, 200064);
    dgpp::text::GrammarState grammar(&vocab, spec, false, true);
    const std::string prefix = "]<]minimax[>[";
    const std::string block = "</mm:think>" + prefix + "<tool_call>\n" + prefix +
                              "<invoke name=\"f\">" + prefix + "<city>Paris & 東京" + prefix +
                              "</city>" + prefix + "<days>3" + prefix + "</days>" + prefix +
                              "</invoke>\n" + prefix + "</tool_call>";
    double mask_ms = 0;
    size_t masks = 0;
    for (auto id : tok.encode(block)) {
      if (vocab.text(id) == "Paris" || vocab.text(id) == "3" || masks == 0) {
        dgpp::text::TokenMask mask;
        const auto begin = std::chrono::steady_clock::now();
        grammar.mask(&mask);
        mask_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count();
        ++masks;
        for (int candidate = 0; candidate < vocab.vocab_size(); ++candidate)
          require(mask.allows(candidate) == grammar.allows(candidate),
                  "real-vocabulary mask differs from simulation");
      }
      require(grammar.allows(id), "real tokenizer tool token refused in " +
                                      std::string(grammar.state_name()) + ": " + vocab.text(id));
      grammar.advance(id);
      require(grammar.active(), "real-tokenizer grammar died");
    }
    require(grammar.allows(200020), "real tokenizer call cannot terminate");
    std::printf(
        "MiniMax tool grammar: %zu full-vocabulary masks checked, %.3f ms total mask time\n", masks,
        mask_ms);
    std::printf("MiniMax prompt: %zu Jinja/HF cases exact; adaptive reasoning PASS\n", count);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "MiniMax prompt: %s\n", error.what());
    return 1;
  }
}
