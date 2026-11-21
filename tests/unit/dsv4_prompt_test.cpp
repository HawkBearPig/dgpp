#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/dsv4_prompt.hpp"
#include "text/dsv41_prompt.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::string render_json(const std::string& payload) {
  const auto globals = dgpp::minijson::parse(payload);
  return dgpp::text::Dsv4Prompt::render(globals.root);
}

bool refuses(const std::string& payload, const std::string& naming) {
  try {
    (void)render_json(payload);
  } catch (const std::runtime_error& e) {
    return std::string(e.what()).find(naming) != std::string::npos;
  }
  return false;
}

}  // namespace

// What the reference refuses (and the forms it has no rendering for) throws
// naming the field. No checkpoint needed.
DGPP_TEST(dsv4_prompt_refusals_name_the_field) {
  require(refuses(R"({"messages":[]})", "messages"), "an empty conversation");
  require(refuses(R"({"messages":[{"role":"critic","content":"x"}]})", "messages[0]: unknown role 'critic'"), "an unknown role");
  require(refuses(R"({"messages":[{"role":"user","content":"x"}],"reasoning_effort":"extreme"})", "reasoning_effort"),
          "an unknown effort");
  require(refuses(R"({"messages":[{"role":"user","content":"x"}],"reasoning_effort":"none"})", "reasoning_effort"),
          "none is the service's enable_thinking false, not an effort of the reference");
  require(refuses(R"({"messages":[{"role":"user","content":"x"}],"reasoning_effort":75})", "reasoning_effort"),
          "a numeric budget is V4.1's, not this encoder's");
  require(refuses(R"({"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[)"
                  R"({"function":{"name":"f","arguments":"[1, 2]"}}]}]})",
                  "messages[1].tool_calls[0].function.arguments"),
          "arguments that are JSON but not an object");
  require(refuses(R"({"messages":[{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[)"
                  R"({"function":{"arguments":"{}"}}]}]})",
                  "messages[1].tool_calls[0].function.name"),
          "a call without a name");
  require(refuses(R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]})",
                  "messages[0].content[0]"),
          "an image part (the encoder is text only)");
  require(refuses(R"({"messages":[{"role":"developer","content":""}]})", "messages[0].content"), "a developer message without content");
  require(refuses(R"({"messages":[{"role":"user","content":"x","task":"summarize"}]})", "messages[0].task"), "an unknown task");
  require(refuses(R"({"messages":[{"role":"user","content":"x"}],"tools":{"name":"f"}})", "tools"), "tools that are not an array");
  require(refuses(R"({"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function"}]})", "tools[0].function"),
          "a tool entry without a function");
  require(refuses(R"({"messages":[{"role":"user","content":"x"}],"enable_thinking":true,"thinking":false})", "thinking"),
          "the two spellings of the thinking switch disagreeing");
  require(refuses(R"({"messages":[{"role":"user","content":"x"}],"clear_thinking":true,"drop_thinking":false})", "drop_thinking"),
          "the two spellings of the history switch disagreeing");
  // The two renderers are different revisions of the prefix cache's key.
  require(dgpp::text::Dsv4Prompt::source_hash() != dgpp::text::Dsv41Prompt::source_hash(), "the renderer's own revision");
}
