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
    std::printf("MiniMax prompt: %zu Jinja/HF cases exact; adaptive reasoning PASS\n", count);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "MiniMax prompt: %s\n", error.what());
    return 1;
  }
}
