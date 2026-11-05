// The 2026-10-26 serve failure (docs/dory_plan.md): a Dory1.0-4B-NVFP4
// serve with `tool_choice: auto` over a 12-tool request (the captured
// /tmp/opencode/req135.json shape) died on
// "GrammarState: a position with no allowed id" mid-decode — list_mask
// threw, the engine exited nonzero, llama-swap answered the rest of the
// campaign with 502s. The two probes below replay the exact machinery
// serving uses: the real checkpoint tokenizer, GrammarVocab::from_tokenizer
// the way apps/dgpp_serve.cpp builds it, the spec generation_service.cpp
// derives for tool_choice auto, and the decode path — kThink, the
// model itself closing the block with  closely (Dory's template
// generation prompt opens it) — into <tool_call> + a name. Each mask taken
// BEFORE the advance exactly as both engines do.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "common/log.hpp"

int g_argc = 0;
char** g_argv = nullptr;
#include "common/test.hpp"
#include "loaders/hf_cache.hpp"
#include "loaders/minijson.hpp"
#include "text/tokenizer.hpp"
#include "text/tool_grammar.hpp"

namespace fs = std::filesystem;

namespace {

const char* kDory = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4";

// The captured request's 12 tools (cap135_body.json), verbatim shapes.
const char* kToolsJson = R"TOOLJSON([
 {"type":"function","function":{"name":"web_search","description":"Search the web for current information",
  "parameters":{"type":"object","properties":{"query":{"type":"string"},"max_results":{"type":"integer","default":5}},
  "required":["query"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"get_weather","description":"Get current weather for a specific location",
  "parameters":{"type":"object","properties":{"location":{"type":"string"},
  "units":{"type":"string","enum":["celsius","fahrenheit"],"default":"celsius"}},
  "required":["location"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"calculator","description":"Perform mathematical calculations",
  "parameters":{"type":"object","properties":{"expression":{"type":"string"}},
  "required":["expression"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"send_email","description":"Send an email to a recipient",
  "parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},
  "body":{"type":"string"},"cc":{"type":"string","description":"CC recipient email address"},
  "bcc":{"type":"string","description":"BCC recipient email address"},
  "attachments":{"type":"array","items":{"type":"string"},"default":[]}},
  "required":["to","subject","body"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"search_files","description":"Search for files by name or content",
  "parameters":{"type":"object","properties":{"query":{"type":"string"},"directory":{"type":"string"}},
  "required":["query"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"read_file","description":"Read the contents of a file",
  "parameters":{"type":"object","properties":{"path":{"type":"string"}},
  "required":["path"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"create_calendar_event","description":"Create a calendar event",
  "parameters":{"type":"object","properties":{"title":{"type":"string"},"date":{"type":"string"},
  "time":{"type":"string"},"duration_minutes":{"type":"integer"},"attendees":{"type":"array",
  "items":{"type":"string"}},"description":{"type":"string"}},
  "required":["title","date","time"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"get_contacts","description":"Look up contacts",
  "parameters":{"type":"object","properties":{"search":{"type":"string"}},
  "required":["search"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"translate_text","description":"Translate text to another language",
  "parameters":{"type":"object","properties":{"text":{"type":"string"},"target_language":{"type":"string"}},
  "required":["text","target_language"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"get_stock_price","description":"Get the current stock price",
  "parameters":{"type":"object","properties":{"symbol":{"type":"string"}},
  "required":["symbol"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"set_reminder","description":"Set a reminder",
  "parameters":{"type":"object","properties":{"message":{"type":"string"},"when":{"type":"string"}},
  "required":["message","when"],"additionalProperties":false}}},
 {"type":"function","function":{"name":"run_code","description":"Run some code",
  "parameters":{"type":"object","properties":{"language":{"type":"string"},"code":{"type":"string"}},
  "required":["language","code"],"additionalProperties":false}}}
])TOOLJSON";

dgpp::text::GrammarSpec auto_spec() {
  const dgpp::minijson::ParseResult tools = dgpp::minijson::parse(kToolsJson);
  dgpp::text::GrammarSpec spec;
  spec.mode = dgpp::text::GrammarSpec::Mode::kAuto;
  spec.parallel = true;
  for (const dgpp::minijson::Value& entry : tools.root.items()) {
    const dgpp::minijson::Value* fn = entry.find("function");
    spec.tools.push_back(
        dgpp::text::grammar_tool_from_function(*fn, nullptr));
  }
  return spec;
}

std::filesystem::path dory_snapshot() {
  std::string err;
  const std::string snap = dgpp::hf::model_dir(kDory, &err);
  if (snap.empty()) {
    DGPP_LOG_WARN("dory checkpoint unavailable ({})", err);
    std::exit(2);
  }
  return std::filesystem::path(snap);
}

// eos ids (generation_config.json: [2, 11]) + vocab_size (config.json) the
// DoryFamily reports.
std::pair<std::vector<int64_t>, int> dory_vocab(const fs::path& snap) {
  const auto read = [](const fs::path& p) {
    std::FILE* f = std::fopen(p.c_str(), "rb");
    std::string out;
    char buf[1 << 16];
    size_t n;
    while (f && (n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    if (f) std::fclose(f);
    return out;
  };
  const dgpp::minijson::ParseResult cfg = dgpp::minijson::parse(read(snap / "config.json"));
  std::vector<int64_t> eos;
  const dgpp::minijson::ParseResult gen = dgpp::minijson::parse(read(snap / "generation_config.json"));
  if (const auto* eos_id = gen.root.find("eos_token_id")) {
    if (eos_id->is_number()) eos.push_back(eos_id->as_int());
    for (const auto& e : eos_id->items()) eos.push_back(e.as_int());
  }
  return {eos, static_cast<int>(cfg.root.find("vocab_size")->as_int())};
}

dgpp::text::GrammarVocab dory_grammar_vocab() {
  const fs::path snap = dory_snapshot();
  const auto tok = dgpp::text::Tokenizer::load((snap / "tokenizer.json").string());
  auto [eos, vocab_size] = dory_vocab(snap);
  // apps/dgpp_serve.cpp: the DSML dialect is the family's (dory is not a
  // DeepSeek family, so kV41 is irrelevant).
  return dgpp::text::GrammarVocab::from_tokenizer(
      tok, eos, vocab_size, dgpp::text::DsmlDialect::kV41);
}

}  // namespace

namespace {
void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}
}  // namespace

DGPP_TEST(dory_grammar_vocab_probe) {
  const fs::path snap = dory_snapshot();
  const auto tok = dgpp::text::Tokenizer::load((snap / "tokenizer.json").string());
  const auto vocab = dory_grammar_vocab();
  const auto mark = vocab.markers();
  const auto esc = [](const std::string& s) {
    std::string o;
    for (char c : s)
      if (c == '\n') o += "\\n";
      else if (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) {
        char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); o += b;
      } else o += c;
    return o;
  };
  std::printf("markers: think_close '%s' tool_call_open '%s' tool_call_close '%s'\n",
              esc(mark.think_close.text).c_str(), esc(mark.tool_call_open.text).c_str(),
              esc(mark.tool_call_close.text).c_str());
  std::printf("tool_format=%d xml_compact=%d think_close_id=%lld\n",
              static_cast<int>(mark.tool_format()), mark.xml_compact,
              static_cast<long long>(mark.think_close.id));
  for (int64_t id : {10, 11, 12, 13, 14, 15}) {
    const std::string t = vocab.text(id);
    std::printf(" tok(%lld) hex:", static_cast<long long>(id));
    for (unsigned char c : t) std::printf(" %02x('%c')", c, c >= 0x20 && c < 0x7f ? c : '.');
    std::printf("\n");
  }
  std::printf("added tokens (id<40):");
  for (const auto& a : tok.added_tokens())
    if (a.id < 40)
      std::printf(" %lld:'%s'%s", static_cast<long long>(a.id), a.content.c_str(),
                  a.special ? "!" : "");
  std::printf("\n");
  const std::vector<int64_t> ids = tok.encode("hello\nworld");
  std::printf("encode(hello\\nworld):");
  for (const int64_t id : ids) std::printf(" %lld", static_cast<long long>(id));
  std::printf("\n");
  for (const int64_t id : ids)
    std::printf(" %lld->'%s'", static_cast<long long>(id), tok.decode(id).c_str());
  std::printf("\n");
  std::printf("by_first sizes: '\\n' %zu '<' %zu 'a' %zu ' ' %zu '\\n'n texts: ",
              vocab.ids_starting_with('\n').size(), vocab.ids_starting_with('<').size(),
              vocab.ids_starting_with('a').size(), vocab.ids_starting_with(' ').size());
  for (const int64_t id : vocab.ids_starting_with('a'))
    std::printf(" %lld->'%s'", static_cast<long long>(id), vocab.text(id).c_str());
  std::printf("\n");
  const std::vector<int64_t> opener = tok.encode("\n<function=create_calendar_event>\n");
  std::printf("opener pieces:");
  for (const int64_t id : opener)
    std::printf(" %lld->'%s'", static_cast<long long>(id), tok.decode(id).c_str());
  std::printf("\n");
}

DGPP_TEST(dory_grammar_closes_thinking_into_an_allowed_call) {
  const auto vocab = dory_grammar_vocab();
  const auto mark = vocab.markers();
  require(mark.reasoning_available(), "dory markers see the reasoning markers");
  std::printf("[ .. ] dory markers: think_close id %lld, tool_call_open id %lld, eos %lld, texts: %s | %s\n",
              static_cast<long long>(mark.think_close.id),
              static_cast<long long>(mark.tool_call_open.id),
              static_cast<long long>(vocab.call_turn_eos()),
              mark.think_close.text.c_str(), mark.tool_call_open.text.c_str());
  const auto spec = auto_spec();
  const fs::path snap = dory_snapshot();
  const auto tok = dgpp::text::Tokenizer::load((snap / "tokenizer.json").string());

  const std::string turn =
      mark.think_close.text + "\n" + mark.tool_call_open.text +
      "\n<function=create_calendar_event>\n<parameter=title>\n"
      "Q3 budget review\n</parameter>\n<parameter=date>\n2026-03-21\n"
      "</parameter>\n<parameter=time>\n08:00\n</parameter>\n"
      "</function>\n" + mark.tool_call_close.text +
      vocab.text(vocab.call_turn_eos());

  dgpp::text::GrammarState state(&vocab, spec, /*prompt_opens_thinking=*/true);
  int depth = 0;
  for (const int64_t id : tok.encode(turn)) {
    dgpp::text::TokenMask mask;
    state.mask(&mask);
    require(mask.allows(id) == state.allows(id),
                        "mask/allows disagree on depth " + std::to_string(depth));
    require(mask.allows(id),
                        std::string("depth ") + std::to_string(depth) + ": id " +
                            std::to_string(id) + " refused in " + state.state_name());
    state.advance(id);
    require(state.active(), "the grammar died on depth " + std::to_string(depth));
    ++depth;
  }
}

// The captured decode: DGPP_OPS_LOG=<serve_rank0.ops path>,
// DGPP_OPS_REQ=<chatcmpl id> — the request the repro serve log pinned
// (2026-10-26, 16:02 boot, direct on :5999). Each mask/advance pair the
// engine takes, one id at a time.
DGPP_TEST(dory_grammar_replays_the_captured_decode) {
  const char* path = std::getenv("DGPP_OPS_LOG");
  const char* req = std::getenv("DGPP_OPS_REQ");
  if (!path || !req)
    dgpp::test::skip("captured replay requires DGPP_OPS_LOG and DGPP_OPS_REQ");
  const auto vocab = dory_grammar_vocab();
  const auto spec = auto_spec();

  std::vector<int64_t> ids;
  if (std::FILE* f = std::fopen(path, "rb")) {
    std::vector<char> buf(1 << 16);
    std::string ops;
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) ops.append(buf.data(), n);
    std::fclose(f);
    const std::string want = std::string("T ") + req + " ";
    size_t pos = 0;
    while ((pos = ops.find(want, pos)) != std::string::npos) {
      const size_t e = ops.find('\n', pos);
      if (e == std::string::npos) break;
      ids.push_back(std::atoll(ops.substr(pos + want.size(), e - pos - want.size()).c_str()));
      pos = e;
    }
  }
  require(!ids.empty(), "no captured decode ids matched the request");
  std::printf("[ .. ] replaying %zu committed ids\n", ids.size());

  dgpp::text::GrammarState state(&vocab, spec, /*prompt_opens_thinking=*/true);
  int depth = 0;
  bool prev_eos = false;
  for (const int64_t id : ids) {
    // The ops log keeps every decode of the chat under one id: each
    // engine stop (a committed eos) starts the next segment fresh.
    if (prev_eos) {
      state = dgpp::text::GrammarState(&vocab, spec, /*prompt_opens_thinking=*/true);
      depth = 0;
      prev_eos = false;
    }
    dgpp::text::TokenMask mask;
    state.mask(&mask);
    if (!mask.allows(id))
      std::printf("[ ?? ] depth %d state %s refuses id %lld\n", depth,
                  state.state_name(), static_cast<long long>(id));
    require(mask.allows(id) == state.allows(id), "mask/allows disagree");
    require(mask.allows(id),
                        std::string("depth ") + std::to_string(depth) +
                            " id " + std::to_string(id) + " refused in " + state.state_name());
    state.advance(id);
    require(state.active(),
                        "the grammar died on depth " + std::to_string(depth) + " (" +
                            std::to_string(id) + ")");
    prev_eos = vocab.is_eos(id);
    ++depth;
  }
}

int main(int argc, char** argv) {
  g_argc = argc;
  g_argv = argv;
  return ::dgpp::test::run_all();
}
