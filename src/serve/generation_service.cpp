#include "serve/generation_service.hpp"
#include "serve/image_inputs.hpp"
#include "text/dsv41_prompt.hpp"

#include <mutex>
#include <unordered_set>

#include <cstring>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <utility>

#include "common/log.hpp"
#include "serve/json_out.hpp"

namespace dgpp::serve {

namespace {

// A tool definition's schema notes are per DEFINITION, and an agent client
// sends the same definitions with every request: each distinct line is
// logged once — WARN for an argument left free, INFO for a bound merely not
// enforced — and at DEBUG after that (2026-09-06: the live log carried the
// same three WARN lines before every request).
void log_tool_schema_note(bool warn, size_t index, const std::string& text) {
  static std::mutex mu;
  static std::unordered_set<std::string> seen;
  constexpr size_t kMaxDistinct = 4096;
  bool first = false;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (seen.size() < kMaxDistinct) first = seen.insert(text).second;
  }
  if (!first) {
    DGPP_LOG_DEBUG("serve: tools[{}] {}", index, text);
  } else if (warn) {
    DGPP_LOG_WARN("serve: tools[{}] {} (logged once; at debug from now on)",
                  index, text);
  } else {
    DGPP_LOG_INFO("serve: tools[{}] {} (logged once; at debug from now on)",
                  index, text);
  }
}

using dgpp::sched::Scheduler;
using dgpp::sched::SchedulerRequest;
using dgpp::text::ToolCallParser;

const minijson::Value* optional_field(const minijson::Value& body, std::string_view name) {
  const auto* v = body.find(name);
  return v && !v->is_null() ? v : nullptr;
}

bool bounded_integer(const minijson::Value& v, int64_t lo, int64_t hi) {
  if (v.kind() == minijson::Value::Kind::Int) return v.as_int() >= lo && v.as_int() <= hi;
  // Bounds here fit exactly in double; int64 seeds are checked separately.
  const double x = v.as_double();
  return v.is_number() && std::isfinite(x) && x == std::floor(x) &&
         x >= static_cast<double>(lo) && x <= static_cast<double>(hi);
}

bool api_name_ok(const minijson::Value* value) {
  if (!value || !value->is_string()) return false;
  const auto name = value->as_string();
  return !name.empty() && name.size() <= 64 &&
         std::all_of(name.begin(), name.end(), [](unsigned char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '_' || c == '-';
         });
}

// OpenAI finish_reason from the scheduler's retire reason. A turn that
// ended naturally after at least one parsed tool call is "tool_calls";
// the steps cap is "length" whatever was parsed.
const char* finish_reason(Scheduler::Result::Reason r, bool tool_calls) {
  switch (r) {
    case Scheduler::Result::Reason::kEos:
      return tool_calls ? "tool_calls" : "stop";
    case Scheduler::Result::Reason::kSteps: return "length";
    // A cancelled stream has no reader; a cancelled non-stream request
    // cannot exist (cancellation only follows a client disconnect).
    case Scheduler::Result::Reason::kCancelled: return "stop";
    // Shed at pool exhaustion (grow-on-demand): the answer is cut short,
    // which OpenAI's vocabulary calls "length"; the metrics and the log
    // carry the cause.
    case Scheduler::Result::Reason::kPoolExhausted: return "length";
    // The client's stop string matched: a natural end.
    case Scheduler::Result::Reason::kStop: return "stop";
    default: return "stop";
  }
}

// The usage object, with the details OpenAI reports:
// prompt_tokens_details.cached_tokens — the prompt tokens the prefix cache
// served (the attach position) — and completion_tokens_details.
// reasoning_tokens — the generated ids that went to reasoning.
void append_usage(std::string* out, int prompt_tokens, int completion_tokens,
                  int cached_tokens, int reasoning_tokens) {
  out->append("\"usage\":{\"prompt_tokens\":");
  append_json_int(out, prompt_tokens);
  out->append(",\"completion_tokens\":");
  append_json_int(out, completion_tokens);
  out->append(",\"total_tokens\":");
  append_json_int(out, prompt_tokens + completion_tokens);
  out->append(",\"prompt_tokens_details\":{\"cached_tokens\":");
  append_json_int(out, cached_tokens);
  out->append("},\"completion_tokens_details\":{\"reasoning_tokens\":");
  append_json_int(out, reasoning_tokens);
  out->append("}}");
}

// The chat.completion.chunk preamble: id/object/created/model.
void append_chunk_preamble(std::string* out, const std::string& id,
                           int64_t created, const std::string& model) {
  out->append("{\"id\":");
  append_json_string(out, id);
  out->append(",\"object\":\"chat.completion.chunk\",\"created\":");
  append_json_int(out, created);
  out->append(",\"model\":");
  append_json_string(out, model);
  out->append(",\"system_fingerprint\":null,\"choices\":[");
}

std::string chat_chunk_first(const std::string& id, int64_t created,
                             const std::string& model, int index = 0) {
  std::string out;
  append_chunk_preamble(&out, id, created, model);
  out.append("{\"index\":");
  append_json_int(&out, index);
  out.append(
      ",\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
      "\"logprobs\":null,\"finish_reason\":null}]}");
  return out;
}

// One chunk carrying `delta_json` (a complete JSON object: the delta).
std::string chat_chunk_delta(const std::string& id, int64_t created,
                             const std::string& model,
                             const std::string& delta_json,
                             const std::string& logprobs_json = "",
                             int index = 0) {
  std::string out;
  append_chunk_preamble(&out, id, created, model);
  out.append("{\"index\":");
  append_json_int(&out, index);
  out.append(",\"delta\":");
  out.append(delta_json);
  out.append(",\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":null}]}");
  return out;
}

std::string delta_text(const char* field, const std::string& text) {
  std::string out = "{\"";
  out.append(field);
  out.append("\":");
  append_json_string(&out, text);
  out.push_back('}');
  return out;
}

// The tool-call id: call_<16 hex>, a pure function of the record and the
// call's index (reproducible under the gates' fixed seeds, distinct per
// request and per call).
std::string tool_call_id(uint64_t tag, int index) {
  uint64_t z = tag * 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(index) +
               0x632BE59BD9B4E019ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  char buf[24];
  std::snprintf(buf, sizeof(buf), "call_%016llx",
                static_cast<unsigned long long>(z));
  return buf;
}

// The streamed tool-call deltas: one announcing the call (index, id, type,
// name, empty arguments) and one carrying the complete arguments string
// — a client that concatenates argument fragments sees one fragment.
std::string delta_tool_call_start(int index, const std::string& call_id,
                                  const std::string& name, bool custom = false) {
  std::string out = "{\"tool_calls\":[{\"index\":";
  append_json_int(&out, index);
  out.append(",\"id\":");
  append_json_string(&out, call_id);
  out.append(custom ? ",\"type\":\"custom\",\"custom\":{\"name\":" :
                      ",\"type\":\"function\",\"function\":{\"name\":");
  append_json_string(&out, name);
  out.append(custom ? ",\"input\":\"\"}}]}" : ",\"arguments\":\"\"}}]}");
  return out;
}

std::string delta_tool_call_arguments(int index, const std::string& args, bool custom = false) {
  std::string out = "{\"tool_calls\":[{\"index\":";
  append_json_int(&out, index);
  out.append(custom ? ",\"custom\":{\"input\":" : ",\"function\":{\"arguments\":");
  append_json_string(&out, args);
  out.append("}}]}");
  return out;
}

std::string chat_chunk_final(const std::string& id, int64_t created,
                             const std::string& model, const char* finish,
                             const std::string& logprobs_json = "",
                             int index = 0) {
  std::string out;
  append_chunk_preamble(&out, id, created, model);
  out.append("{\"index\":");
  append_json_int(&out, index);
  out.append(",\"delta\":{},\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":");
  append_json_string(&out, finish);
  out.append("}]}");
  return out;
}

std::string chat_chunk_usage(const std::string& id, int64_t created,
                             const std::string& model, int prompt_tokens,
                             int completion_tokens, int cached_tokens,
                             int reasoning_tokens) {
  std::string out;
  out.append("{\"id\":");
  append_json_string(&out, id);
  out.append(",\"object\":\"chat.completion.chunk\",\"created\":");
  append_json_int(&out, created);
  out.append(",\"model\":");
  append_json_string(&out, model);
  out.append(",\"system_fingerprint\":null,\"choices\":[],");
  append_usage(&out, prompt_tokens, completion_tokens, cached_tokens,
               reasoning_tokens);
  out.push_back('}');
  return out;
}

// One choice of the one-shot chat.completion object (n choices join by
// index, 2026-09-06).
std::string chat_choice_json(int index, const std::string& message_json,
                             const char* finish,
                             const std::string& logprobs_json = "") {
  std::string out = "{\"index\":";
  append_json_int(&out, index);
  out.append(",\"message\":");
  out.append(message_json);
  out.append(",\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":");
  append_json_string(&out, finish);
  out.push_back('}');
  return out;
}

// The one-shot chat.completion object around its joined choices.
std::string chat_completion_body(const std::string& id, int64_t created,
                                 const std::string& model,
                                 const std::string& choices_json,
                                 int prompt_tokens, int completion_tokens,
                                 int cached_tokens, int reasoning_tokens) {
  std::string out;
  out.append("{\"id\":");
  append_json_string(&out, id);
  out.append(",\"object\":\"chat.completion\",\"created\":");
  append_json_int(&out, created);
  out.append(",\"model\":");
  append_json_string(&out, model);
  out.append(",\"system_fingerprint\":null,\"choices\":[");
  out.append(choices_json);
  out.append("],");
  append_usage(&out, prompt_tokens, completion_tokens, cached_tokens,
               reasoning_tokens);
  out.push_back('}');
  return out;
}

// The legacy text_completion chunk shapes (POST /v1/completions).
std::string text_chunk_preamble_fields(const std::string& id, int64_t created,
                                       const std::string& model) {
  std::string out;
  out.append("{\"id\":");
  append_json_string(&out, id);
  out.append(",\"object\":\"text_completion\",\"created\":");
  append_json_int(&out, created);
  out.append(",\"model\":");
  append_json_string(&out, model);
  out.append(",\"system_fingerprint\":null,\"choices\":[");
  return out;
}

std::string text_chunk_first(const std::string& id, int64_t created,
                             const std::string& model) {
  std::string out = text_chunk_preamble_fields(id, created, model);
  out.append(
      "{\"index\":0,\"text\":\"\",\"logprobs\":null,\"finish_reason\":null}]}");
  return out;
}

std::string text_chunk_delta(const std::string& id, int64_t created,
                             const std::string& model,
                             const std::string& delta,
                             const std::string& logprobs_json = "") {
  std::string out = text_chunk_preamble_fields(id, created, model);
  out.append("{\"index\":0,\"text\":");
  append_json_string(&out, delta);
  out.append(",\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":null}]}");
  return out;
}

std::string text_chunk_final(const std::string& id, int64_t created,
                             const std::string& model, const char* finish,
                             const std::string& logprobs_json = "") {
  std::string out = text_chunk_preamble_fields(id, created, model);
  out.append("{\"index\":0,\"text\":\"\",\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":");
  append_json_string(&out, finish);
  out.append("}]}");
  return out;
}

std::string text_choice_json(int index, const std::string& text,
                             const char* finish,
                             const std::string& logprobs_json = "") {
  std::string out = "{\"index\":";
  append_json_int(&out, index);
  out.append(",\"text\":");
  append_json_string(&out, text);
  out.append(",\"logprobs\":");
  out.append(logprobs_json.empty() ? "null" : logprobs_json);
  out.append(",\"finish_reason\":");
  append_json_string(&out, finish);
  out.push_back('}');
  return out;
}

std::string text_completion_body(const std::string& id, int64_t created,
                                 const std::string& model,
                                 const std::string& choices_json,
                                 int prompt_tokens, int completion_tokens,
                                 int cached_tokens) {
  std::string out;
  out.append("{\"id\":");
  append_json_string(&out, id);
  out.append(",\"object\":\"text_completion\",\"created\":");
  append_json_int(&out, created);
  out.append(",\"model\":");
  append_json_string(&out, model);
  out.append(",\"system_fingerprint\":null,\"choices\":[");
  out.append(choices_json);
  out.append("],");
  append_usage(&out, prompt_tokens, completion_tokens, cached_tokens, 0);
  out.push_back('}');
  return out;
}

// The model object, with the served surfaces as extensions: the sampling
// defaults (what an omitted field becomes), whether stochastic requests
// are executable on this engine at all, and the tool/reasoning surface.
std::string model_object(const ServiceConfig& scfg, const std::string& model_id,
                         int64_t created, bool sampling_available,
                         const sample::Params& d, bool tools_available,
                         bool constraints_available,
                         bool reasoning_in_content, const ModelFrontend& frontend, bool images_available) {
  std::string out;
  out.append("{\"id\":");
  append_json_string(&out, model_id);
  out.append(",\"object\":\"model\",\"created\":");
  append_json_int(&out, created);
  out.append(",\"owned_by\":\"dgpp\",\"sampling\":{\"available\":");
  out.append(sampling_available ? "true" : "false");
  out.append(",\"defaults\":{\"temperature\":");
  append_json_float(&out, d.temperature);
  out.append(",\"top_p\":");
  append_json_float(&out, d.top_p);
  out.append(",\"top_k\":");
  append_json_int(&out, d.top_k);
  out.append(",\"min_p\":");
  append_json_float(&out, d.min_p);
  out.append(",\"repetition_penalty\":");
  append_json_float(&out, d.repetition_penalty);
  out.append("}},\"tools\":{\"available\":");
  out.append(tools_available ? "true" : "false");
  out.append(",\"constrained\":");
  out.append(constraints_available ? "true" : "false");
  out.append("},\"response_format\":{\"json_object\":");
  out.append(constraints_available ? "true" : "false");
  out.append(",\"json_schema\":");
  out.append(constraints_available ? "true" : "false");
  out.append("},\"reasoning\":{\"in_content\":");
  out.append(reasoning_in_content ? "true" : "false");
  out.append(",\"effort_mapping\":{");
  bool first = true;
  for (const char* effort : {"none", "minimal", "low", "medium", "high", "xhigh", "max"}) {
    try {
      auto settings = frontend.reasoning_settings(effort);
      if (!first) out += ',';
      first = false;
      append_json_string(&out, effort); out += ":{";
      if (settings.enable_thinking) {
        out += "\"enable_thinking\":"; out += *settings.enable_thinking ? "true" : "false";
        if (settings.effort) out += ',';
      }
      if (settings.effort) { out += "\"reasoning_effort\":"; append_json_string(&out, *settings.effort); }
      out += '}';
    } catch (const std::invalid_argument&) { /* unsupported efforts are absent */ }
  }
  out += '}';
  out.append("},\"input_modalities\":[\"text\"");
  if (images_available) out.append(",\"image\"");
  out.append("]");
  // The rope ramp in force, under the names the deployment JSON uses, so
  // what an operator set is what a client reads, plus the three values the
  // ramp derives (the attention scale, the correction band, the ceiling it
  // reaches). Absent when the knob is off — the plain rope needs no
  // advertising, and a client that reads it must treat it as off.
  if (scfg.rope_scaling) {
    const dgpp::RopeScaling& rs = *scfg.rope_scaling;
    out.append(",\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":");
    append_json_float(&out, rs.factor);
    out.append(",\"original_max_position_embeddings\":");
    append_json_int(&out, rs.original_max_position_embeddings);
    out.append(",\"beta_fast\":");
    append_json_float(&out, rs.beta_fast);
    out.append(",\"beta_slow\":");
    append_json_float(&out, rs.beta_slow);
    out.append(",\"attn_factor\":");
    append_json_float(&out, rs.attn_factor);
    out.append(",\"mrope_cache_factor\":");
    append_json_float(&out, rs.mrope_cache_factor);
    out.append(",\"mscale\":");
    append_json_float(&out, rs.mscale());
    out.append(",\"correction_max_position_embeddings\":");
    append_json_int(&out, rs.correction_max_position());
    out.append(",\"scaled_max_position_embeddings\":");
    append_json_int(&out, rs.context_limit());
    out.push_back('}');
  }
  // The effective request context limit (review item 7): the MINIMUM of the
  // positional ceiling and the K/V pool, with both bounds beside it. A pool
  // past the ceiling seats concurrency, not a longer request, and a ceiling
  // past the pool asks for tokens the process never allocated — the two
  // numbers together are the honest answer, which is why all three ride the
  // response (docs/qwen_yarn_release_check.md reads them first).
  if (scfg.position_ceiling > 0 || scfg.kv_pool_tokens > 0) {
    const int64_t limit = scfg.position_ceiling > 0 && scfg.kv_pool_tokens > 0
                              ? std::min(scfg.position_ceiling, scfg.kv_pool_tokens)
                              : std::max(scfg.position_ceiling, scfg.kv_pool_tokens);
    out.append(",\"context\":{\"request_limit_tokens\":");
    append_json_int(&out, limit);
    out.append(",\"position_ceiling_tokens\":");
    if (scfg.position_ceiling > 0) append_json_int(&out, scfg.position_ceiling);
    else out.append("null");  // the family states no ceiling of its own
    out.append(",\"kv_pool_tokens\":");
    if (scfg.kv_pool_tokens > 0) append_json_int(&out, scfg.kv_pool_tokens);
    else out.append("null");
    out.append(",\"limited_by\":");
    append_json_string(
        &out, scfg.position_ceiling > 0 && scfg.kv_pool_tokens > 0
                  ? (scfg.position_ceiling <= scfg.kv_pool_tokens ? "position-ceiling" : "kv-pool")
                  : (scfg.position_ceiling > 0 ? "position-ceiling" : "kv-pool"));
    out.push_back('}');
  }
  out.push_back('}');
  return out;
}

// A minijson value is a string or an array whose items are strings or
// objects — the two content forms the template's visible_text renders
// (content parts: {type: "text", text} and the media kinds it answers
// with a reminder).
bool content_form_ok(const dgpp::minijson::Value& v) {
  if (v.is_string()) return true;
  if (!v.is_array()) return false;
  for (const auto& item : v.items()) {
    if (item.is_string()) continue;
    if (!item.is_object()) return false;
    const dgpp::minijson::Value* type = item.find("type");
    if (type == nullptr || !type->is_string()) return false;
  }
  return true;
}

}  // namespace

GenerationService::GenerationService(const ServiceConfig& cfg,
                                     dgpp::sched::SchedulerEngine* engine,
                                     const ModelFrontend* frontend,
                                     std::vector<int64_t> eos_token_ids)
    : cfg_(cfg),
      file_inputs_(cfg.file_inputs),
      engine_(engine),
      frontend_(frontend),
      markers_(frontend != nullptr ? frontend->markers()
                                   : dgpp::text::ChatMarkers{}),
      sched_(engine, std::move(eos_token_ids), cfg.queue_limit,
             cfg.admission) {
  if (engine_ == nullptr)
    throw std::invalid_argument("GenerationService: engine must not be null");
  if (frontend_ == nullptr)
    throw std::invalid_argument("GenerationService: frontend required");
  if (cfg_.model_id.empty())
    throw std::invalid_argument("GenerationService: model_id required");
  meters_ = sched_.meters();
  prefix_stats_ = engine_->prefix_engine_stats();
  boundary_ids_ = frontend_->boundary_token_ids();
  std::sort(boundary_ids_.begin(), boundary_ids_.end());
  boundary_ids_.erase(std::unique(boundary_ids_.begin(), boundary_ids_.end()),
                      boundary_ids_.end());
  if (sched_.prefix_slots() > 0)
    DGPP_LOG_INFO(
        "serve: prefix cache on — {} snapshot slots, {} boundary token(s)",
        sched_.prefix_slots(), boundary_ids_.size());
  // The sampling surface: the defaults must be a valid spec (an operator
  // error otherwise), and an engine that cannot draw serves GREEDY defaults
  // — loudly — rather than a stochastic mode it would silently ignore.
  sample::validate_params(cfg_.sampling_defaults);
  sampling_available_ = engine_->supports_sampling();
  if (!sampling_available_ && cfg_.sampling_defaults.temperature > 0.0f) {
    DGPP_LOG_WARN(
        "serve: the bound engine cannot sample yet — the model's default "
        "(temperature {} top_p {}) is NOT served; omitted fields default to "
        "greedy and temperature > 0 is refused (400 sampling_unsupported)",
        cfg_.sampling_defaults.temperature, cfg_.sampling_defaults.top_p);
    cfg_.sampling_defaults = sample::greedy_params();
  }
  seed_rng_.seed(cfg_.fixed_seed.has_value()
                     ? *cfg_.fixed_seed
                     : static_cast<uint64_t>(std::random_device{}()) << 32 ^
                           static_cast<uint64_t>(std::random_device{}()));
  DGPP_LOG_INFO(
      "serve: sampling {} — defaults temperature {} top_p {} top_k {} min_p "
      "{} repetition_penalty {}; seedless requests {}",
      sampling_available_ ? "available" : "unavailable (greedy engine)",
      cfg_.sampling_defaults.temperature, cfg_.sampling_defaults.top_p,
      cfg_.sampling_defaults.top_k, cfg_.sampling_defaults.min_p,
      cfg_.sampling_defaults.repetition_penalty,
      cfg_.fixed_seed.has_value()
          ? "use the fixed seed " + std::to_string(*cfg_.fixed_seed)
          : std::string("draw a fresh seed each"));
  DGPP_LOG_INFO(
      "serve: tool calls {}; tool_choice/parallel_tool_calls {}; reasoning {}",
      markers_.tool_calls_available()
          ? "available (the template's markers are in the tokenizer)"
          : "unavailable (no tool-call markers in this tokenizer)",
      constraints_available() ? "enforced by constrained decoding"
                              : "not enforceable (no masks on this engine)",
      !markers_.reasoning_available()
          ? "not split (no </think> marker)"
          : cfg_.reasoning_in_content ? "folded into content"
                                      : "on reasoning_content");
  // The SSE tap: tokens and retires ride the scheduler's observer
  // callbacks straight into the request records.
  sched_.set_observer(this);
  // The service takes every token through the observer above and answers
  // from its own records: the scheduler keeps no retired request, so the
  // process's memory tracks the live requests, not the cumulative traffic.
  sched_.set_keep_retired(false);
}

// ---------------------------------------------------------------------------
// The sampling spec (both completion routes)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The stop-string scanner
// ---------------------------------------------------------------------------

std::string StopScanner::feed(std::string text) {
  if (hit) return {};
  std::string buf = std::move(hold);
  buf += text;
  hold.clear();
  // The earliest match of any stop string decides; the match and what
  // follows are never shown.
  size_t best = std::string::npos;
  for (const std::string& s : stops) {
    const size_t at = buf.find(s);
    if (at < best) best = at;
  }
  if (best != std::string::npos) {
    hit = true;
    buf.resize(best);
    return buf;
  }
  // The longest suffix of the buffer that is a proper prefix of some stop
  // string is held: the next piece decides it.
  size_t keep = 0;
  for (const std::string& s : stops) {
    const size_t most = std::min(buf.size(), s.size() - 1);
    for (size_t k = most; k > keep; --k) {
      if (buf.compare(buf.size() - k, k, s, 0, k) == 0) {
        keep = k;
        break;
      }
    }
  }
  hold.assign(buf, buf.size() - keep, keep);
  buf.resize(buf.size() - keep);
  return buf;
}

std::string StopScanner::finish() {
  std::string rest;
  rest.swap(hold);
  return hit ? std::string() : rest;
}

namespace {
constexpr int kMaxChoices = 128;         // OpenAI's schema bound; queue capacity still applies
constexpr size_t kMaxLogitBias = 1024;   // logit_bias entries
}  // namespace

bool GenerationService::parse_n(const dgpp::minijson::Value& body,
                                HttpResponseWriter& w, int* n) {
  *n = 1;
  const dgpp::minijson::Value* v = body.find("n");
  if (v == nullptr || v->is_null()) return true;
  if (!bounded_integer(*v, 1, kMaxChoices)) {
    respond_error(w, 400,
                  "n must be an integer in [1, " + std::to_string(kMaxChoices) +
                      "]",
                  "invalid_request_error", "n");
    return false;
  }
  *n = static_cast<int>(v->as_int(0));
  return true;
}

bool GenerationService::parse_stop(const dgpp::minijson::Value& body,
                                   HttpResponseWriter& w,
                                   std::vector<std::string>* stops) {
  stops->clear();
  const dgpp::minijson::Value* v = body.find("stop");
  if (v == nullptr || v->is_null()) return true;
  const auto bad = [&] {
    respond_error(w, 400,
                  "stop must be a non-empty string or an array of one to "
                  "four non-empty strings",
                  "invalid_request_error", "stop");
    return false;
  };
  const auto take = [&](const dgpp::minijson::Value& s) {
    if (!s.is_string() || s.as_string().empty()) return bad();
    stops->emplace_back(s.as_string());
    return true;
  };
  if (v->is_string()) return take(*v);
  if (!v->is_array() || v->items().empty() || v->items().size() > 4)
    return bad();
  for (const dgpp::minijson::Value& s : v->items())
    if (!take(s)) return false;
  return true;
}

bool GenerationService::parse_logit_bias(
    const dgpp::minijson::Value& body, HttpResponseWriter& w,
    std::vector<dgpp::sched::LogitBias>* bias) {
  bias->clear();
  const dgpp::minijson::Value* v = body.find("logit_bias");
  if (v == nullptr || v->is_null()) return true;
  if (!v->is_object()) {
    respond_error(w, 400,
                  "logit_bias must be an object mapping token ids to biases "
                  "in [-100, 100]",
                  "invalid_request_error", "logit_bias");
    return false;
  }
  if (v->members().empty()) return true;
  if (!engine_->supports_logit_bias() || cfg_.vocab_size <= 0) {
    respond_error(w, 400,
                  "logit_bias is not available on this engine",
                  "invalid_request_error", "logit_bias",
                  "logit_bias_unsupported");
    return false;
  }
  if (v->members().size() > kMaxLogitBias) {
    respond_error(w, 400,
                  "logit_bias has more than " + std::to_string(kMaxLogitBias) +
                      " entries",
                  "invalid_request_error", "logit_bias");
    return false;
  }
  std::unordered_set<int64_t> seen;
  for (const dgpp::minijson::Member& m : v->members()) {
    int64_t id = 0;
    bool digits = !m.key.empty() && m.key.size() <= 12;
    for (const char c : m.key) {
      if (c < '0' || c > '9') {
        digits = false;
        break;
      }
      id = id * 10 + (c - '0');
    }
    if (!digits) {
      respond_error(w, 400,
                    "logit_bias keys must be token ids (decimal strings); "
                    "got '" + m.key + "'",
                    "invalid_request_error", "logit_bias");
      return false;
    }
    if (id >= cfg_.vocab_size) {
      respond_error(w, 400,
                    "logit_bias token " + m.key + " is outside the vocabulary "
                    "(" + std::to_string(cfg_.vocab_size) + " ids)",
                    "invalid_request_error", "logit_bias");
      return false;
    }
    if (!m.value.is_number() || !(m.value.as_double(0.0) >= -100.0) ||
        !(m.value.as_double(0.0) <= 100.0)) {
      respond_error(w, 400,
                    "logit_bias values must be numbers in [-100, 100]; token " +
                        m.key + " is not",
                    "invalid_request_error", "logit_bias");
      return false;
    }
    if (!seen.insert(id).second) {
      respond_error(w, 400, "logit_bias lists token " + m.key + " twice",
                    "invalid_request_error", "logit_bias");
      return false;
    }
    dgpp::sched::LogitBias b;
    b.token = static_cast<int32_t>(id);
    b.bias = static_cast<float>(m.value.as_double(0.0));
    bias->push_back(b);
  }
  return true;
}

bool GenerationService::parse_sampling(const dgpp::minijson::Value& body,
                                       HttpResponseWriter& w,
                                       sample::Params* sampling,
                                       uint64_t* seed) {
  sample::Params p = cfg_.sampling_defaults;
  // One numeric field: present → a finite number inside [lo, hi] (an
  // integer when asked), else the 400 names it.
  const auto field = [&](const char* name, double lo, double hi,
                         bool integer, const auto& apply) -> bool {
    const dgpp::minijson::Value* v = body.find(name);
    if (v == nullptr || v->is_null()) return true;
    const double x = v->is_number() ? v->as_double(0.0) : 0.0;
    if (!v->is_number() || !std::isfinite(x) ||
        (integer && x != std::floor(x))) {
      respond_error(w, 400,
                    std::string(name) + " must be " +
                        (integer ? "an integer" : "a finite number"),
                    "invalid_request_error", name);
      return false;
    }
    if (x < lo || x > hi) {
      std::string range = "[";
      append_json_float(&range, lo);
      range += ", ";
      append_json_float(&range, hi);
      range += "]";
      respond_error(w, 400, std::string(name) + " must be in " + range,
                    "invalid_request_error", name);
      return false;
    }
    apply(x);
    return true;
  };
  bool has_seed = false;
  if (!field("temperature", 0.0, 2.0, false,
             [&](double x) { p.temperature = static_cast<float>(x); }) ||
      !field("top_p", 0.0, 1.0, false,
             [&](double x) { p.top_p = static_cast<float>(x); }) ||
      !field("top_k", 0.0, 2147483647.0, true,
             [&](double x) { p.top_k = static_cast<int>(x); }) ||
      !field("min_p", 0.0, 1.0, false,
             [&](double x) { p.min_p = static_cast<float>(x); }) ||
      !field("repetition_penalty", 0.0, 100.0, false,
             [&](double x) { p.repetition_penalty = static_cast<float>(x); }) ||
      !field("presence_penalty", -2.0, 2.0, false,
             [&](double x) { p.presence_penalty = static_cast<float>(x); }) ||
      !field("frequency_penalty", -2.0, 2.0, false,
             [&](double x) { p.frequency_penalty = static_cast<float>(x); }))
    return false;
  if (const auto* s = optional_field(body, "seed")) {
    if (s->kind() != minijson::Value::Kind::Int) {
      respond_error(w, 400, "seed must be a signed 64-bit integer", "invalid_request_error", "seed");
      return false;
    }
    has_seed = true;
    *seed = static_cast<uint64_t>(s->as_int());
  }
  // A zero-mass nucleus retains the most likely token. Express it through
  // the sampler's existing one-candidate path without changing GPU numerics.
  if (p.top_p == 0.0f) {
    p.top_p = 1.0f;
    p.top_k = 1;
  }
  try {
    sample::validate_params(p);
  } catch (const std::invalid_argument& e) {
    // The message starts with the field's name ("top_p must be in (0, 1]").
    const std::string what = e.what();
    respond_error(w, 400, what, "invalid_request_error",
                  what.substr(0, what.find(' ')));
    return false;
  }
  if (p.temperature > 0.0f && !sampling_available_) {
    respond_error(w, 400,
                  "temperature > 0 asks for stochastic sampling, which this "
                  "engine cannot execute yet (the graph engine's sampler is "
                  "the next slice); omit temperature or set it to 0",
                  "invalid_request_error", "temperature",
                  "sampling_unsupported");
    return false;
  }
  if (!has_seed)
    *seed = cfg_.fixed_seed.has_value() ? *cfg_.fixed_seed : seed_rng_();
  *sampling = p;
  return true;
}

// ---------------------------------------------------------------------------
// The chat request's conversation and tools (M6 6f)
// ---------------------------------------------------------------------------

bool GenerationService::parse_chat(const dgpp::minijson::Value& body,
                                   HttpResponseWriter& w, ChatPlan* plan) {
  using dgpp::minijson::Member;
  using dgpp::minijson::Value;
  const auto refuse = [&](const std::string& message, const std::string& param,
                          const std::string& code = "") {
    respond_error(w, 400, message, "invalid_request_error", param, code);
    return false;
  };

  const Value* messages = body.find("messages");
  if (messages == nullptr || !messages->is_array() ||
      messages->items().empty())
    return refuse("messages is required and must be a non-empty array",
                  "messages");

  // ---- tools --------------------------------------------------------------
  const Value* tools = body.find("tools");
  Value normalized_tools;
  if (tools && tools->is_array()) {
    std::vector<Value> entries;
    for (size_t i = 0; i < tools->items().size(); ++i) {
      const auto& tool = tools->items()[i];
      const auto* type = tool.find("type");
      if (!type || !type->is_string() || type->as_string() != "custom") { entries.push_back(tool); continue; }
      const std::string where = "tools[" + std::to_string(i) + "].custom";
      const auto* custom = tool.find("custom");
      if (!custom || !custom->is_object()) return refuse("custom must be an object", where);
      const auto* name = custom->find("name");
      if (!api_name_ok(name)) return refuse("custom.name must contain 1-64 letters, digits, underscores or dashes", where + ".name");
      std::string description;
      if (const auto* d = custom->find("description")) {
        if (!d->is_string()) return refuse("description must be a string", where + ".description");
        description = d->as_string();
      }
      description += "\nPass the complete free-form tool input in the input parameter, encoded as a JSON string literal.";
      std::vector<Member> schema{{"type", Value::make_string("string")}};
      if (const auto* format = custom->find("format")) {
        const auto* kind = format->find("type");
        if (!format->is_object() || !kind || !kind->is_string() ||
            (kind->as_string() != "text" && kind->as_string() != "grammar"))
          return refuse("custom format.type must be text or grammar", where + ".format.type");
        if (kind->as_string() == "grammar") {
          if (!constraints_available()) return refuse("custom grammars require constrained decoding", where + ".format", "constrained_decoding_unsupported");
          const auto* grammar = format->find("grammar");
          if (!grammar || !grammar->is_object()) return refuse("grammar must be an object", where + ".format.grammar");
          schema.push_back({"x-dgpp-grammar", *grammar});
          try { dgpp::text::compile_json_schema(Value::make_object(schema)); }
          catch (const std::invalid_argument& e) { return refuse(e.what(), where + ".format.grammar", "invalid_grammar"); }
          description += "\nInput grammar: " + dgpp::text::json_text_of(*grammar);
        }
      }
      plan->custom_tools.emplace_back(name->as_string());
      entries.push_back(Value::make_object({{"type", Value::make_string("function")},
        {"function", Value::make_object({{"name", *name},
          {"description", Value::make_owned_string(std::move(description))},
          {"strict", Value::make_bool(constraints_available())},
          {"parameters", Value::make_object({{"type", Value::make_string("object")},
            {"properties", Value::make_object({{"input", Value::make_object({
              {"anyOf", Value::make_array({Value::make_object(std::move(schema))})}})}})},
            {"required", Value::make_array({Value::make_string("input")})},
            {"additionalProperties", Value::make_bool(false)}})}})}}));
    }
    normalized_tools = Value::make_array(std::move(entries));
    tools = &normalized_tools;
  }
  const auto is_custom = [&](std::string_view name) {
    return std::find(plan->custom_tools.begin(), plan->custom_tools.end(), name) != plan->custom_tools.end();
  };
  Value allowed_tools;
  std::vector<std::string> tool_names;
  if (tools != nullptr) {
    if (!tools->is_array()) return refuse("tools must be an array", "tools");
    if (tools->items().size() > 128) return refuse("at most 128 tools are supported", "tools");
    for (size_t i = 0; i < tools->items().size(); ++i) {
      const Value& t = tools->items()[i];
      const std::string where = "tools[" + std::to_string(i) + "]";
      if (!t.is_object()) return refuse(where + " must be an object", where);
      const Value* type = t.find("type");
      if (type != nullptr &&
          (!type->is_string() || type->as_string() != "function"))
        return refuse(where + ".type must be \"function\"", where + ".type");
      const Value* fn = t.find("function");
      if (fn != nullptr && !fn->is_object())
        return refuse(where + ".function must be an object",
                      where + ".function");
      const Value& def = fn != nullptr ? *fn : t;
      const Value* name = def.find("name");
      if (!api_name_ok(name))
        return refuse(where + ".function.name is required and must be a "
                      "1-64 character string of letters, digits, underscores or dashes",
                      where + ".function.name");
      const Value* params = def.find("parameters");
      if (std::find(tool_names.begin(), tool_names.end(), name->as_string()) != tool_names.end())
        return refuse("tool names must be unique", where + ".function.name");
      if (const auto* strict = optional_field(def, "strict")) {
        if (!strict->is_bool()) return refuse("strict must be a boolean", where + ".function.strict");
        if (strict->as_bool() && !constraints_available())
          return refuse("strict tools require constrained decoding", where + ".function.strict",
                        "constrained_decoding_unsupported");
      }
      if (const auto* description = def.find("description"))
        if (!description->is_string()) return refuse("description must be a string", where + ".function.description");
      if (params != nullptr && !params->is_object())
        return refuse(where + ".function.parameters must be an object (a "
                      "JSON schema)",
                      where + ".function.parameters");
      tool_names.emplace_back(name->as_string());
    }
  }
  const bool have_tools = tools != nullptr && !tools->items().empty();
  if (have_tools && !tool_calls_available())
    return refuse(
        "tools cannot be served: this model's tokenizer has no tool-call "
        "markers, so the service could not parse the model's calls",
        "tools", "tools_unsupported");

  // ---- tool_choice --------------------------------------------------------
  enum class Choice { kAuto, kNone, kRequired, kNamed };
  Choice choice = Choice::kAuto;
  std::string named;
  if (const Value* tc = body.find("tool_choice")) {
    if (!have_tools && !(tc->is_string() && tc->as_string() == "none"))
      return refuse("tool_choice requires a non-empty tools array",
                    "tool_choice");
    if (tc->is_string()) {
      const std::string_view s = tc->as_string();
      if (s == "auto")
        choice = Choice::kAuto;
      else if (s == "none")
        choice = Choice::kNone;
      else if (s == "required")
        choice = Choice::kRequired;
      else
        return refuse("tool_choice must be \"auto\", \"none\", \"required\" "
                      "or {\"type\": \"function\", \"function\": {\"name\": "
                      "…}}",
                      "tool_choice");
    } else if (tc->is_object()) {
      const Value* type = tc->find("type");
      if (type && type->is_string() && type->as_string() == "allowed_tools") {
        if (!constraints_available())
          return refuse("allowed_tools requires constrained decoding", "tool_choice",
                        "constrained_decoding_unsupported");
        const auto* allowed = tc->find("allowed_tools");
        if (!allowed || !allowed->is_object())
          return refuse("allowed_tools must be an object", "tool_choice.allowed_tools");
        const auto* mode = allowed->find("mode");
        const auto* subset = allowed->find("tools");
        if (!mode || !mode->is_string() || (mode->as_string() != "auto" && mode->as_string() != "required"))
          return refuse("allowed_tools.mode must be auto or required", "tool_choice.allowed_tools.mode");
        if (!subset || !subset->is_array() || subset->items().empty())
          return refuse("allowed_tools.tools must be a non-empty array", "tool_choice.allowed_tools.tools");
        std::vector<Value> selected;
        std::unordered_set<std::string> seen;
        for (const auto& ref : subset->items()) {
          const auto* kind = ref.find("type");
          const bool custom = kind && kind->is_string() && kind->as_string() == "custom";
          const auto* fn = ref.find(custom ? "custom" : "function");
          const auto* name = fn && fn->is_object() ? fn->find("name") : nullptr;
          if (!kind || !kind->is_string() || (!custom && kind->as_string() != "function") || !name || !name->is_string() || is_custom(name->as_string()) != custom)
            return refuse("allowed tool must name a declared function or custom tool of the same type", "tool_choice.allowed_tools.tools");
          const std::string key(name->as_string());
          const auto pos = std::find(tool_names.begin(), tool_names.end(), key);
          if (pos == tool_names.end() || !seen.insert(key).second)
            return refuse("allowed tools must name distinct declared functions", "tool_choice.allowed_tools.tools");
          selected.push_back(tools->items()[static_cast<size_t>(pos - tool_names.begin())]);
        }
        allowed_tools = Value::make_array(std::move(selected));
        tools = &allowed_tools;
        choice = mode->as_string() == "required" ? Choice::kRequired : Choice::kAuto;
      } else {
        const bool custom = type && type->is_string() && type->as_string() == "custom";
        if (type != nullptr &&
            (!type->is_string() || (type->as_string() != "function" && !custom)))
          return refuse("tool_choice.type must be function or custom",
                        "tool_choice.type");
        const Value* fn = tc->find(custom ? "custom" : "function");
        const std::string name_param = custom ? "tool_choice.custom.name" : "tool_choice.function.name";
        const Value* name = fn != nullptr && fn->is_object() ? fn->find("name")
                                                              : nullptr;
        if (name == nullptr || !name->is_string())
          return refuse(name_param + " must be a string", name_param);
        named = std::string(name->as_string());
        if (std::find(tool_names.begin(), tool_names.end(), named) ==
            tool_names.end())
          return refuse("tool_choice names '" + named +
                            "', which is not one of the request's tools",
                        name_param);
        if (is_custom(named) != custom) return refuse("tool_choice type does not match the declared tool", "tool_choice.type");
        choice = Choice::kNamed;
      }
    } else {
      return refuse("tool_choice must be a string or an object", "tool_choice");
    }
  }

  // ---- parallel_tool_calls ----------------------------------------------
  bool parallel = true;
  if (const Value* ptc = body.find("parallel_tool_calls")) {
    if (!ptc->is_bool())
      return refuse("parallel_tool_calls must be a boolean",
                    "parallel_tool_calls");
    parallel = ptc->as_bool(true);
  }

  // ---- the grammar (M6 6g) ---------------------------------------------
  // tool_choice none / required / named and parallel_tool_calls false are
  // enforced by constrained decoding: the pick's mask on every rank.
  {
    using dgpp::text::GrammarSpec;
    GrammarSpec& g = plan->grammar;
    switch (choice) {
      case Choice::kAuto:
        // auto arms the grammar too (M6 6i): calls at will, but every call
        // well-formed — a known name, the closed keys, typed values.
        g.mode = have_tools ? GrammarSpec::Mode::kAuto : GrammarSpec::Mode::kNone;
        break;
      case Choice::kNone:
        g.mode = have_tools ? GrammarSpec::Mode::kForbidCalls : GrammarSpec::Mode::kNone;
        break;
      case Choice::kRequired: g.mode = GrammarSpec::Mode::kRequired; break;
      case Choice::kNamed:
        g.mode = GrammarSpec::Mode::kNamed;
        g.named = named;
        break;
    }
    g.parallel = parallel;
    if (g.active() && !constraints_available()) {
      // tool_choice none is served by leaving the tools out of the render
      // on any engine (the grammar is belt and braces), and auto with
      // parallel calls asked for no guarantee; required, named and
      // parallel_tool_calls false are guarantees only a masked pick gives.
      if (choice == Choice::kNone || (choice == Choice::kAuto && parallel)) {
        g = GrammarSpec{};
      } else {
        const char* field = choice == Choice::kAuto ? "parallel_tool_calls"
                                                    : "tool_choice";
        return refuse(
            std::string(field) +
                " cannot be enforced on this engine (it has no constrained "
                "decoding); omit it — tool_choice auto with "
                "parallel_tool_calls true is served",
            field, "constrained_decoding_unsupported");
      }
    }
    if (g.active()) {
      for (size_t i = 0; i < tools->items().size(); ++i) {
        const Value& t = tools->items()[i];
        const Value* fn = t.find("function");
        const Value& def = fn != nullptr && fn->is_object() ? *fn : t;
        // The closed key set and the typed arguments (M6 6g / 6i); a
        // strict function outside the enforceable subset is a 400 naming
        // the keyword path, a non-strict one leaves that value free.
        std::vector<std::string> warnings, notes;
        try {
          g.tools.push_back(
              dgpp::text::grammar_tool_from_function(def, &warnings, &notes));
        } catch (const std::invalid_argument& e) {
          const std::string prefix =
              "tools[" + std::to_string(i) + "]" + (fn != nullptr ? ".function." : ".");
          const std::string what = e.what();
          const size_t colon = what.find(':');
          return refuse(prefix + what,
                        prefix + (colon == std::string::npos ? what : what.substr(0, colon)),
                        "unsupported_schema");
        }
        for (const std::string& w : warnings) log_tool_schema_note(true, i, w);
        for (const std::string& n : notes) log_tool_schema_note(false, i, n);
        // The DSML invoke names the tool as the schema lists it —
        // "namespace::name" under a namespace — and the grammar's targets
        // must spell the same (the parser strips the namespace again).
        if (markers_.tool_format() == dgpp::text::ToolFormat::kDsml) {
          try {
            g.tools.back().name = dgpp::text::Dsv41Prompt::qualified_tool_name(t);
          } catch (const std::exception& e) {
            const std::string prefix = "tools[" + std::to_string(i) + "]";
            return refuse(prefix + ": " + e.what(), prefix, "invalid_request_error");
          }
        }
      }
      // A named choice under DSML names the function; the target is its
      // qualified spelling.
      if (g.mode == GrammarSpec::Mode::kNamed && markers_.tool_format() == dgpp::text::ToolFormat::kDsml)
        for (const dgpp::text::GrammarTool& tool : g.tools)
          if (tool.name == g.named || (tool.name.size() > g.named.size() + 2 &&
                                       tool.name.compare(tool.name.size() - g.named.size() - 2, std::string::npos,
                                                         "::" + g.named) == 0))
            g.named = tool.name;
    }
  }

  // ---- response_format (M6 6h) -------------------------------------------
  // json_object and json_schema are guarantees of the same masked pick: the
  // content is one JSON text (an object in json_object mode; conforming to
  // the schema in json_schema mode), then the turn ends. The prompt is
  // untouched; the model reasons first when the template opens thinking.
  if (const Value* rf = body.find("response_format")) {
    if (!rf->is_object())
      return refuse("response_format must be an object", "response_format");
    const Value* type = rf->find("type");
    if (type == nullptr || !type->is_string())
      return refuse("response_format.type must be a string", "response_format.type");
    const std::string_view kind = type->as_string();
    if (kind != "text" && kind != "json_object" && kind != "json_schema")
      return refuse("response_format.type must be \"text\", \"json_object\" or "
                    "\"json_schema\"",
                    "response_format.type");
    if (kind != "text") {
      if (!constraints_available())
        return refuse(
            "response_format json_object / json_schema cannot be enforced "
            "on this engine (it has no constrained decoding)",
            "response_format", "constrained_decoding_unsupported");
      using dgpp::text::GrammarSpec;
      GrammarSpec& g = plan->grammar;
      // A required/named choice still owes a tool call. Auto can choose
      // either a call or the structured answer; none forces the answer.
      if (!have_tools || choice == Choice::kNone)
        g.mode = GrammarSpec::Mode::kJson;
      else if (choice == Choice::kAuto)
        g.mode = GrammarSpec::Mode::kJsonOrTools;
      if (kind == "json_schema") {
        const Value* js = rf->find("json_schema");
        if (js == nullptr || !js->is_object())
          return refuse("response_format.json_schema must be an object",
                        "response_format.json_schema");
        const Value* name = js->find("name");
        if (!api_name_ok(name))
          return refuse("response_format.json_schema.name must contain 1-64 "
                        "letters, digits, underscores or dashes",
                        "response_format.json_schema.name");
        const Value* strict_v = js->find("strict");
        if (strict_v != nullptr && !strict_v->is_bool() && !strict_v->is_null())
          return refuse("response_format.json_schema.strict must be a boolean",
                        "response_format.json_schema.strict");
        const bool strict = strict_v != nullptr && strict_v->as_bool(false);
        const Value* schema = js->find("schema");
        if (schema != nullptr && !schema->is_object())
          return refuse("response_format.json_schema.schema must be an object "
                        "(a JSON schema)",
                        "response_format.json_schema.schema");
        if (schema != nullptr) {
          // Compile here to refuse loudly: the subset is type, properties,
          // required, additionalProperties, items, minItems, maxItems,
          // enum, const, anyOf. strict: anything outside it is a 400 naming
          // the keyword; otherwise the schema falls back to free JSON with a
          // warning (OpenAI's non-strict mode promises no conformance).
          try {
            dgpp::text::compile_json_schema(*schema);
            g.json_schema = dgpp::text::json_text_of(*schema);
          } catch (const std::invalid_argument& e) {
            const std::string what = e.what();  // "schema.<path>: reason"
            const size_t colon = what.find(':');
            const std::string where =
                "response_format.json_schema." +
                (colon == std::string::npos ? std::string("schema") : what.substr(0, colon));
            if (strict)
              return refuse("response_format.json_schema.schema cannot be enforced: " +
                                what,
                            where, "unsupported_schema");
            DGPP_LOG_WARN(
                "serve: response_format json_schema '{}' is not strict and "
                "its schema is outside the enforceable subset ({}); serving "
                "it as json_object",
                std::string(name->as_string()), what);
            g.json_schema.clear();
          }
        }
      }
      if (!g.has_json()) g.json_schema.clear();  // required/named: tool-call turn
    }
  }

  // ---- reasoning_effort / chat_template_kwargs --------------------------
  std::optional<std::string> effort;
  const auto effort_ok = [](std::string_view s) {
    return s == "none" || s == "minimal" || s == "low" || s == "medium" ||
           s == "high" || s == "xhigh" || s == "max";
  };
  if (const Value* re = optional_field(body, "reasoning_effort")) {
    if (!re->is_string() || !effort_ok(re->as_string()))
      return refuse("reasoning_effort must be none, minimal, low, medium, high, xhigh or max",
                    "reasoning_effort");
    effort = std::string(re->as_string());
  }
  std::vector<Member> extra;
  if (const Value* kw = body.find("chat_template_kwargs")) {
    if (!kw->is_object())
      return refuse("chat_template_kwargs must be an object",
                    "chat_template_kwargs");
    std::unordered_set<std::string> seen_kwargs;
    for (const Member& m : kw->members()) {
      const std::string where = "chat_template_kwargs." + m.key;
      if (!seen_kwargs.insert(m.key).second) return refuse("duplicate template parameter", where);
      if (m.key == "clear_thinking") {
        if (!m.value.is_bool()) return refuse(where + " must be a boolean", where);
        extra.push_back(m);
      } else if (m.key == "reasoning_effort") {
        if (!m.value.is_string() || !effort_ok(m.value.as_string()))
          return refuse(where + " must be none, minimal, low, medium, high, xhigh or max",
                        where);
        if (effort.has_value() && *effort != m.value.as_string())
          return refuse("reasoning_effort and chat_template_kwargs."
                        "reasoning_effort disagree; send one",
                        where);
        effort = std::string(m.value.as_string());
      } else if (m.key == "enable_thinking") {
        // A knob of the templates that read it (Qwen3.8-Flash-Next,
        // GLM-4.7: false closes the think block in the generation prompt);
        // GLM-5.3-Flash's never does — thinking is always on there.
        if (!frontend_->template_reads("enable_thinking"))
          return refuse(
              "this template has no enable_thinking knob — thinking is always "
              "on (the generation prompt opens <think>); use reasoning_effort",
              where, "unsupported_parameter");
        if (!m.value.is_bool()) return refuse(where + " must be a boolean", where);
        extra.push_back(m);
      } else if (m.key == "thinking") {
        // The vLLM DeepSeek-V4.1 template's name for the same switch
        // (2026-09-14): an alias of enable_thinking, so a client written
        // for that stack turns thinking off here too.
        if (!frontend_->template_reads("enable_thinking"))
          return refuse(
              "this template has no thinking knob — thinking is always on "
              "(the generation prompt opens <think>); use reasoning_effort",
              where, "unsupported_parameter");
        if (!m.value.is_bool()) return refuse(where + " must be a boolean", where);
        Member alias = m;
        alias.key = "enable_thinking";
        extra.push_back(alias);
      } else {
        return refuse(where + " is not a knob of this template (it reads "
                      "enable_thinking / thinking, clear_thinking and "
                      "reasoning_effort)",
                      where, "unsupported_parameter");
      }
    }
  }

  // Resolve effort once, before rendering, including conflicts with either
  // spelling of the explicit thinking switch. Do not silently override it.
  std::optional<bool> thinking;
  for (const auto& m : extra) {
    if (m.key != "enable_thinking") continue;
    if (thinking && *thinking != m.value.as_bool())
      return refuse("thinking and enable_thinking disagree", "chat_template_kwargs.thinking");
    thinking = m.value.as_bool();
  }
  if (thinking) {
    std::erase_if(extra, [](const auto& m) { return m.key == "enable_thinking"; });
    extra.push_back({"enable_thinking", Value::make_bool(*thinking)});
  }
  if (effort) {
    ModelFrontend::ReasoningSettings settings;
    try { settings = frontend_->reasoning_settings(*effort); }
    catch (const std::invalid_argument& e) {
      return refuse(e.what(), "reasoning_effort", "unsupported_reasoning_effort");
    }
    if (thinking && ((*effort == "none") == *thinking))
      return refuse("reasoning_effort conflicts with the explicit thinking switch",
                    "chat_template_kwargs.enable_thinking");
    if (settings.enable_thinking && !thinking)
      extra.push_back({"enable_thinking", Value::make_bool(*settings.enable_thinking)});
    effort = settings.effort;
  }

  // ---- messages: validate and normalize ----------------------------------
  std::vector<Value> msgs;
  msgs.reserve(messages->items().size());
  for (size_t i = 0; i < messages->items().size(); ++i) {
    const Value& msg = messages->items()[i];
    const std::string where = "messages[" + std::to_string(i) + "]";
    if (!msg.is_object()) return refuse(where + " must be an object", where);
    const Value* role = msg.find("role");
    if (role == nullptr || !role->is_string())
      return refuse(where + ".role is required and must be a string",
                    where + ".role");
    // OpenAI's "developer" role is the renamed "system"; render it as such.
    const std::string_view raw_role = role->as_string();
    const std::string_view r = raw_role == "developer" ? "system" : raw_role;
    if (r != "system" && r != "user" && r != "assistant" && r != "tool")
      return refuse(where + ".role '" + std::string(r) +
                        "' is not one this template renders (developer, system, user, "
                        "assistant, tool)",
                    where + ".role");
    const Value* content = msg.find("content");
    Value normalized_content;
    for (const char* field : {"audio", "function_call"})
      if (optional_field(msg, field))
        return refuse("this message feature is not supported", where + "." + field, "unsupported_parameter");
    if (const auto* name = msg.find("name"))
      if (!name->is_string()) return refuse("name must be a string", where + ".name");
    const auto* refusal = optional_field(msg, "refusal");
    if (refusal && (r != "assistant" || !refusal->is_string()))
      return refuse("refusal must be a string on an assistant message", where + ".refusal");
    if ((!content || content->is_null()) && refusal) content = refusal;
    if (content && content->is_array() && r != "tool") {
      std::vector<Value> parts;
      for (size_t j = 0; j < content->items().size(); ++j) {
        const auto& part = content->items()[j];
        const std::string at = where + ".content[" + std::to_string(j) + "]";
        const auto* type = part.find("type");
        if (!part.is_object() || !type || !type->is_string())
          return refuse("content part must be an object with a type", at);
        if (part.find("prompt_cache_breakpoint"))
          return refuse("explicit cache breakpoints are not supported", at + ".prompt_cache_breakpoint", "unsupported_parameter");
        if (type->as_string() == "image_url") {
          if (r != "user") return refuse("images require a user message", at + ".type", "unsupported_content_type");
          if (!frontend_->supports_images() || !engine_->supports_images())
            return refuse("the served model does not support image inputs", at + ".type", "unsupported_content_type");
          const auto* image = part.find("image_url");
          if (!image || !image->is_object()) return refuse("image_url must be an object", at + ".image_url");
          parts.push_back(part);
          continue;
        }
        const bool is_refusal = r == "assistant" && type->as_string() == "refusal";
        if (is_refusal && content->items().size() != 1)
          return refuse("a refusal part must be the only content part", at);
        if (type->as_string() != "text" && !is_refusal)
          return refuse("this model accepts text content only", at + ".type", "unsupported_content_type");
        const auto* text = part.find(is_refusal ? "refusal" : "text");
        if (!text || !text->is_string()) return refuse("content part text must be a string", at);
        parts.push_back(is_refusal ? Value::make_object({{"type", Value::make_string("text")}, {"text", *text}}) : part);
      }
      normalized_content = Value::make_array(std::move(parts));
      content = &normalized_content;
    }
    std::vector<Member> members;
    members.reserve(msg.members().size() + 1);
    bool content_written = false;
    if (r == "assistant") {
      const Value* tcs = msg.find("tool_calls");
      std::vector<Value> calls;
      if (tcs != nullptr) {
        if (!tcs->is_array())
          return refuse(where + ".tool_calls must be an array",
                        where + ".tool_calls");
        for (size_t j = 0; j < tcs->items().size(); ++j) {
          Value tc = tcs->items()[j];
          const std::string at = where + ".tool_calls[" + std::to_string(j) + "]";
          if (!tc.is_object()) return refuse(at + " must be an object", at);
          if (const auto* type = tc.find("type"); type && type->is_string() && type->as_string() == "custom") {
            const auto* id = tc.find("id");
            if (!id || !id->is_string() || id->as_string().empty())
              return refuse("custom tool history requires a nonempty call id", at + ".id");
            const auto* custom = tc.find("custom");
            const auto* name = custom ? custom->find("name") : nullptr;
            const auto* input = custom ? custom->find("input") : nullptr;
            if (!custom || !custom->is_object() || !api_name_ok(name) || !input || !input->is_string())
              return refuse("custom tool history requires name and string input", at + ".custom");
            std::vector<Member> fields;
            if (const auto* id = tc.find("id")) fields.push_back({"id", *id});
            fields.push_back({"type", Value::make_string("function")});
            fields.push_back({"function", Value::make_object({{"name", *name},
              {"arguments", Value::make_object({{"input", *input}})}})});
            tc = Value::make_object(std::move(fields));
          }
          const Value* type = tc.find("type");
          if (type != nullptr &&
              (!type->is_string() || type->as_string() != "function"))
            return refuse(at + ".type must be \"function\"", at + ".type");
          const Value* id = tc.find("id");
          if (id != nullptr && !id->is_string())
            return refuse(at + ".id must be a string", at + ".id");
          const Value* fn = tc.find("function");
          if (fn == nullptr || !fn->is_object())
            return refuse(at + ".function must be an object", at + ".function");
          const Value* name = fn->find("name");
          if (name == nullptr || !name->is_string())
            return refuse(at + ".function.name must be a string",
                          at + ".function.name");
          // arguments: the OpenAI wire form is a JSON-object STRING; the
          // template iterates a mapping — parse it here (vLLM's
          // postprocess does the same); an object passes through.
          Value args_value;
          if (const Value* args = fn->find("arguments")) {
            if (args->is_object()) {
              args_value = *args;
            } else if (args->is_string()) {
              try {
                dgpp::minijson::ParseResult pr =
                    dgpp::minijson::parse(args->as_string());
                if (!pr.root.is_object()) throw std::runtime_error("not an object");
                args_value = std::move(pr.root);
              } catch (const std::exception& e) {
                return refuse(at + ".function.arguments must be a JSON object "
                              "or the string of one (" + e.what() + ")",
                              at + ".function.arguments");
              }
            } else {
              return refuse(at + ".function.arguments must be a JSON object "
                            "or the string of one",
                            at + ".function.arguments");
            }
          } else {
            args_value = Value::make_object({});
          }
          std::vector<Member> fn_members;
          for (const Member& fm : fn->members())
            if (fm.key != "arguments") fn_members.push_back(fm);
          fn_members.push_back(Member{"arguments", std::move(args_value)});
          std::vector<Member> tc_members;
          for (const Member& tm : tc.members())
            if (tm.key != "function") tc_members.push_back(tm);
          tc_members.push_back(
              Member{"function", Value::make_object(std::move(fn_members))});
          calls.push_back(Value::make_object(std::move(tc_members)));
        }
      }
      const bool has_calls = !calls.empty();
      if (content == nullptr || content->is_null()) {
        if (!has_calls)
          return refuse(where + ".content is required for an assistant "
                        "message without tool_calls",
                        where + ".content");
      } else if (!content_form_ok(*content)) {
        return refuse(where + ".content must be a string or an array of "
                      "content parts",
                      where + ".content");
      }
      if (const Value* rc = msg.find("reasoning_content"))
        if (!rc->is_string() && !rc->is_null())
          return refuse(where + ".reasoning_content must be a string",
                        where + ".reasoning_content");
      for (const Member& m : msg.members()) {
        if (m.key == "content") {
          members.push_back(Member{"content", (content->is_null()
                                                   ? Value::make_string("")
                                                   : *content)});
          content_written = true;
        } else if (m.key == "tool_calls") {
          if (has_calls)
            members.push_back(Member{"tool_calls", Value::make_array(calls)});
        } else if (m.key == "reasoning_content" && m.value.is_null()) {
          // a null reasoning_content is "none" — the template tests
          // `is string`, so drop it
        } else {
          members.push_back(m);
        }
      }
      if (!content_written)
        members.push_back(Member{"content", Value::make_string("")});
    } else if (r == "tool") {
      if (content == nullptr || !(content->is_string() || content->is_array()))
        return refuse(where + ".content is required for a tool message: a "
                      "string, or the template's list of outputs / tool "
                      "references",
                      where + ".content");
      bool standard_content = content->is_string();
      if (content->is_array()) {
        std::string text;
        bool all_text = true;
        for (const auto& part : content->items()) {
          if (!part.is_object()) return refuse("tool content parts must be objects", where + ".content");
          if (const auto* type = part.find("type")) {
            if (!type->is_string() || type->as_string() != "text")
              return refuse("this model accepts text content only", where + ".content", "unsupported_content_type");
            const auto* value = part.find("text");
            if (!value || !value->is_string()) return refuse("text must be a string", where + ".content");
            if (part.find("prompt_cache_breakpoint"))
              return refuse("explicit cache breakpoints are not supported", where + ".content", "unsupported_parameter");
            text += value->as_string();
          } else {
            // Preserve the existing template-specific output/reference list.
            all_text = false;
          }
        }
        if (all_text) {
          normalized_content = Value::make_owned_string(std::move(text));
          content = &normalized_content;
          standard_content = true;
        }
      }
      const Value* id = msg.find("tool_call_id");
      if ((standard_content && !id) || (id && !id->is_string()))
          return refuse(where + ".tool_call_id must be a string",
                        where + ".tool_call_id");
      for (const Member& m : msg.members())
        members.push_back(m.key == "content" ? Member{"content", *content} : m);
    } else {
      if (content == nullptr || !content_form_ok(*content))
        return refuse(where + ".content is required and must be a string or "
                      "an array of content parts",
                      where + ".content");
      for (const Member& m : msg.members())
        members.push_back(m.key == "content" ? Member{"content", *content} : m);
    }
    // The normalized role must reach the template (developer -> system);
    // the member copies above carried the original value.
    if (r != raw_role)
      for (Member& mm : members)
        if (mm.key == "role") mm.value = Value::make_owned_string("system");
    if (r == "assistant" && refusal)
      for (auto& m : members)
        if (m.key == "content" && (!msg.find("content") || msg.find("content")->is_null())) m.value = *content;
    msgs.push_back(Value::make_object(std::move(members)));
  }

  // ---- the template globals ----------------------------------------------
  std::vector<Member> globals;
  globals.push_back(Member{"messages", Value::make_array(std::move(msgs))});
  if (have_tools && choice != Choice::kNone)
    globals.push_back(Member{"tools", *tools});
  if (effort.has_value())
    globals.push_back(
        Member{"reasoning_effort", Value::make_owned_string(*effort)});
  for (Member& m : extra) globals.push_back(std::move(m));
  plan->globals = Value::make_object(std::move(globals));
  plan->tools_requested = have_tools;
  if (have_tools) plan->schemas = dgpp::text::ToolSchemas(*tools);
  return true;
}

// ---------------------------------------------------------------------------
// The error contract: the OpenAI error object, never a bare string.
// ---------------------------------------------------------------------------

void GenerationService::respond_error(HttpResponseWriter& w, int status,
                                      const std::string& message,
                                      const std::string& type,
                                      const std::string& param,
                                      const std::string& code) {
  if (status >= 400 && status < 500) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.rejects_bad;
  }
  std::string body = "{\"error\":{\"message\":";
  append_json_string(&body, message);
  body.append(",\"type\":");
  append_json_string(&body, type);
  body.append(",\"param\":");
  if (param.empty())
    body.append("null");
  else
    append_json_string(&body, param);
  body.append(",\"code\":");
  if (code.empty())
    body.append("null");
  else
    append_json_string(&body, code);
  body.append("}}");
  w.respond(status, "application/json", std::move(body));
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

void GenerationService::handle(const HttpRequest& req,
                               HttpResponseWriter& w) {
  const std::string& p = req.path;
  if (p == "/v1/files" || p.starts_with("/v1/files/")) {
    enqueue_file_work(req, w, false);
    return;
  }
  if (p == "/v1/chat/completions" || p == "/v1/completions") {
    if (req.method != "POST") {
      respond_error(w, 405, req.method + " is not allowed here; use POST",
                    "invalid_request_error");
      return;
    }
    if (p == "/v1/chat/completions") {
      if (FileInputs::needed(req)) enqueue_file_work(req, w, true);
      else route_chat_completions(req, w);
    } else
      route_completions(req, w);
    return;
  }
  if (p == "/v1/models" || p.rfind("/v1/models/", 0) == 0) {
    if (req.method != "GET") {
      respond_error(w, 405, req.method + " is not allowed here; use GET",
                    "invalid_request_error");
      return;
    }
    route_models(req, w);
    return;
  }
  if (p == "/health") {
    if (req.method != "GET") {
      respond_error(w, 405, "use GET for health", "invalid_request_error");
      return;
    }
    route_health(w);
    return;
  }
  if (p == "/metrics" || p == "/v1/metrics") {
    if (req.method != "GET") {
      respond_error(w, 405, "use GET for metrics", "invalid_request_error");
      return;
    }
    route_metrics(w);
    return;
  }
  respond_error(w, 404, "unknown route: " + req.method + " " + p,
                "invalid_request_error");
}

void GenerationService::route_health(HttpResponseWriter& w) const {
  // /health keeps the checkpoint id (as do the completion records); only
  // /v1/models reports the display alias (served_model_name, when set).
  std::string failure;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_) failure = failure_;
  }
  if (!failure.empty()) {
    // The engine failed (the v1 failure semantics): the process is on its
    // way out; a probe sees the reason, not a green light.
    w.respond(503, "application/json",
              "{\"status\":\"failed\",\"model\":" + json_string(cfg_.model_id) +
                  ",\"error\":" + json_string(failure) + "}");
    return;
  }
  w.respond(200, "application/json",
            "{\"status\":\"ok\",\"model\":" + json_string(cfg_.model_id) +
                "}");
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

bool GenerationService::validate_chat_parameters(const minijson::Value& body, HttpResponseWriter& w) {
  const auto refuse = [&](const std::string& param, const std::string& message,
                          const std::string& code = "unsupported_parameter") {
    respond_error(w, 400, message, "invalid_request_error", param, code);
    return false;
  };
  // Compatible clients forward provider-specific options (for example
  // OpenCode's preserveThinking) at the top level. Preserve our historical
  // tolerance for unknown extensions; they do not become template globals
  // or generation settings. Reject known unsupported API features explicitly,
  // and validate the fields we implement in their respective parsers.
  const std::unordered_set<std::string> unsupported = {
      "web_search_options", "functions", "function_call", "prompt_cache_options"};
  const std::unordered_set<std::string> nullable_unsupported = {
      "audio", "prediction", "moderation", "prompt_cache_key", "prompt_cache_retention", "verbosity"};
  std::unordered_set<std::string> seen;
  for (const auto& m : body.members()) {
    if (!seen.insert(m.key).second) return refuse(m.key, "duplicate request parameter", "invalid_parameter");
    if (unsupported.count(m.key) || (nullable_unsupported.count(m.key) && !m.value.is_null()))
      return refuse(m.key, "'" + m.key + "' is not supported by this server");
  }
  if (const auto* v = optional_field(body, "store")) {
    if (!v->is_bool()) return refuse("store", "store must be a boolean", "invalid_parameter");
    if (v->as_bool()) return refuse("store", "stored completions are not supported; use store: false");
  }
  if (const auto* v = optional_field(body, "service_tier")) {
    if (!v->is_string() || (v->as_string() != "auto" && v->as_string() != "default"))
      return refuse("service_tier", "this server supports only the auto and default service tiers");
  }
  if (const auto* v = optional_field(body, "modalities")) {
    if (!v->is_array() || v->items().size() != 1 || !v->items()[0].is_string() ||
        v->items()[0].as_string() != "text")
      return refuse("modalities", "this model supports only modalities: [\"text\"]");
  }
  const auto characters = [](std::string_view s) {
    return std::count_if(s.begin(), s.end(), [](unsigned char c) { return (c & 0xc0) != 0x80; });
  };
  for (const char* name : {"user", "safety_identifier"}) {
    if (const auto* v = optional_field(body, name)) {
      if (!v->is_string() || (std::string_view(name) == "safety_identifier" && characters(v->as_string()) > 64))
        return refuse(name, std::string(name) + " must be a string (safety_identifier: at most 64 characters)",
                      "invalid_parameter");
    }
  }
  if (const auto* v = optional_field(body, "metadata")) {
    if (!v->is_object() || v->members().size() > 16)
      return refuse("metadata", "metadata must be an object with at most 16 entries", "invalid_parameter");
    for (const auto& m : v->members())
      if (characters(m.key) > 64 || !m.value.is_string() || characters(m.value.as_string()) > 512)
        return refuse("metadata", "metadata keys are limited to 64 characters and string values to 512 characters",
                      "invalid_parameter");
  }
  return true;
}

bool GenerationService::parse_max_tokens(const minijson::Value& body, HttpResponseWriter& w,
                                         int* steps, bool chat) {
  const auto* modern = chat ? optional_field(body, "max_completion_tokens") : nullptr;
  const auto* legacy = optional_field(body, "max_tokens");
  if (modern && legacy) {
    respond_error(w, 400, "max_tokens and max_completion_tokens are aliases; send one",
                  "invalid_request_error", "max_tokens");
    return false;
  }
  const auto* limit = modern ? modern : legacy;
  *steps = cfg_.default_max_tokens;
  if (!limit) return true;
  if (!bounded_integer(*limit, 1, std::numeric_limits<int>::max())) {
    respond_error(w, 400, "token limit must be an integer in [1, 2147483647]",
                  "invalid_request_error", modern ? "max_completion_tokens" : "max_tokens");
    return false;
  }
  *steps = static_cast<int>(limit->as_int());
  return true;
}

bool GenerationService::parse_stream_options(const minijson::Value& body, HttpResponseWriter& w,
                                              bool stream, bool* usage, bool* obfuscation) {
  *usage = false;
  *obfuscation = true;
  const auto* options = optional_field(body, "stream_options");
  if (!options) return true;
  if (!options->is_object() || !stream) {
    respond_error(w, 400, "stream_options must be an object and requires stream: true",
                  "invalid_request_error", "stream_options");
    return false;
  }
  for (const auto& m : options->members()) {
    if (m.key != "include_usage" && m.key != "include_obfuscation") {
      respond_error(w, 400, "unsupported stream option", "invalid_request_error",
                    "stream_options." + m.key, "unsupported_parameter");
      return false;
    }
    if (!m.value.is_bool()) {
      respond_error(w, 400, "stream option must be a boolean", "invalid_request_error", "stream_options." + m.key);
      return false;
    }
    (m.key == "include_usage" ? *usage : *obfuscation) = m.value.as_bool();
  }
  return true;
}

void GenerationService::route_chat_completions(const HttpRequest& req,
                                              HttpResponseWriter& w) {
  dgpp::minijson::ParseResult parsed;
  try {
    parsed = dgpp::minijson::parse(req.body);
  } catch (const std::exception& e) {
    respond_error(w, 400, std::string("invalid JSON body: ") + e.what(),
                  "invalid_request_error", "body");
    return;
  }
  const dgpp::minijson::Value& body = parsed.root;
  if (!body.is_object()) {
    respond_error(w, 400, "the request body must be a JSON object",
                  "invalid_request_error", "body");
    return;
  }

  // model — must name the served model.
  const dgpp::minijson::Value* model = body.find("model");
  if (model == nullptr || !model->is_string()) {
    respond_error(w, 400, "model is required and must be a string",
                  "invalid_request_error", "model");
    return;
  }
  if (!cfg_.model_matches(model->as_string())) {
    respond_error(w, 404, "the model '" + std::string(model->as_string()) +
                              "' does not exist on this server (serving '" +
                              cfg_.display_model_id() + "')",
                  "invalid_request_error", "model", "model_not_found");
    return;
  }

  if (!validate_chat_parameters(body, w)) return;
  int steps = 0;
  if (!parse_max_tokens(body, w, &steps, true)) return;
  bool stream = false;
  if (const auto* sv = optional_field(body, "stream")) {
    if (!sv->is_bool()) {
      respond_error(w, 400, "stream must be a boolean", "invalid_request_error", "stream");
      return;
    }
    stream = sv->as_bool();
  }
  bool include_usage = false, include_obfuscation = true;
  if (!parse_stream_options(body, w, stream, &include_usage, &include_obfuscation)) return;
  // prefix_cache (ours, M7): false opts the request out of the prefix
  // cache — no attach, no snapshot of its state.
  bool prefix_cache = true;
  if (const dgpp::minijson::Value* pcv = body.find("prefix_cache")) {
    if (!pcv->is_bool()) {
      respond_error(w, 400, "prefix_cache must be a boolean",
                    "invalid_request_error", "prefix_cache");
      return;
    }
    prefix_cache = pcv->as_bool(true);
  }

  // Sampling: the spec over the model's defaults (M6 6b). temperature 0
  // is the exact greedy path; anything the engine cannot execute is a 400.
  sample::Params sampling;
  uint64_t seed = 0;
  if (!parse_sampling(body, w, &sampling, &seed)) return;

  // logprobs (bool) + top_logprobs (0..20): exact from the same sampler.
  int logprobs = -1;
  {
    const dgpp::minijson::Value* lp = optional_field(body, "logprobs");
    const dgpp::minijson::Value* top = optional_field(body, "top_logprobs");
    if (lp != nullptr && !lp->is_bool()) {
      respond_error(w, 400, "logprobs must be a boolean",
                    "invalid_request_error", "logprobs");
      return;
    }
    const bool want = lp != nullptr && lp->as_bool(false);
    if (top != nullptr) {
      const double x = top->is_number() ? top->as_double(-1.0) : -1.0;
      if (!top->is_number() || x != std::floor(x) || x < 0.0 || x > 20.0) {
        respond_error(w, 400, "top_logprobs must be an integer in [0, 20]",
                      "invalid_request_error", "top_logprobs");
        return;
      }
      if (!want) {
        respond_error(w, 400, "top_logprobs requires logprobs: true",
                      "invalid_request_error", "top_logprobs");
        return;
      }
    }
    if (want) {
      if (!engine_->supports_logprobs()) {
        respond_error(w, 400,
                      "logprobs are not available on this engine",
                      "invalid_request_error", "logprobs",
                      "logprobs_unsupported");
        return;
      }
      logprobs = top != nullptr ? static_cast<int>(top->as_double(0.0)) : 0;
      sampling.logprobs = logprobs;
    }
  }

  // n, stop and logit_bias.
  int n = 1;
  std::vector<std::string> stops;
  std::vector<dgpp::sched::LogitBias> logit_bias;
  if (!parse_n(body, w, &n) || !parse_stop(body, w, &stops) ||
      !parse_logit_bias(body, w, &logit_bias))
    return;

  // The conversation, the tools and the template knobs (M6 6f).
  ChatPlan plan;
  if (!parse_chat(body, w, &plan)) return;

  // The prompt: render the chat template over the globals, encode. A
  // constrained tool_choice changes nothing here — the grammar rides the
  // scheduler request and masks the pick on every rank (M6 6g).
  std::vector<int64_t> prompt;
  std::vector<ImageInput> images;
  try {
    auto input = frontend_->prepare_chat(plan.globals);
    prompt = std::move(input.tokens);
    images = std::move(input.images);
    validate_image_inputs(images, prompt.size());
  } catch (const ImageInputError& e) {
    respond_error(w, 400, e.what(), "invalid_request_error", e.param, "invalid_image");
    return;
  } catch (const std::exception& e) {
    respond_error(w, 400,
                  "the chat template rejected these messages: " +
                      std::string(e.what()),
                  "invalid_request_error", "messages");
    return;
  }
  if (prompt.empty()) {
    respond_error(w, 400, "the rendered prompt produced no tokens",
                  "invalid_request_error", "messages");
    return;
  }
  const bool opens_thinking = markers_.prompt_opens_thinking(prompt);
  ToolCallParser::Options popts;
  popts.track_tokens = logprobs >= 0;
  popts.start_in_reasoning = opens_thinking;

  // Full-reserve admission arithmetic — a request that can never fit is
  // a 400, never a scheduler deadlock.
  const int64_t reserve =
      engine_->blocks_for_tokens(static_cast<int64_t>(prompt.size()) + steps);
  if (reserve > engine_->pool_blocks_total()) {
    respond_error(
        w, 400,
        "the request's token budget (prompt " +
            std::to_string(prompt.size()) + " + max_tokens " +
            std::to_string(steps) + ") exceeds the model's KV capacity",
        "invalid_request_error", "max_tokens", "context_length_exceeded");
    return;
  }

  // The request's n choices (2026-09-06): one record and one scheduler
  // request per choice, the same prompt (the prefix cache makes every
  // choice past the first an attach), per-choice seeds, one group for the
  // answer; admitted or shed together.
  const uint64_t tag = next_tag_.fetch_add(1, std::memory_order_relaxed);
  std::string rid;
  {
    char suffix[17];
    std::snprintf(suffix, sizeof(suffix), "%016llx",
                  static_cast<unsigned long long>(tag));
    rid = "chatcmpl-" + std::string(suffix);
  }
  const int64_t created = std::time(nullptr);
  auto group = std::make_shared<ChoiceGroup>();
  group->n = n;
  group->choices.resize(static_cast<size_t>(n));
  const std::vector<int64_t> boundaries = prompt_boundaries(prompt);
  const auto arrived = std::chrono::steady_clock::now();
  std::vector<std::shared_ptr<StreamRecord>> records;
  std::vector<SchedulerRequest> requests;
  for (int choice = 0; choice < n; ++choice) {
    auto record = std::make_shared<StreamRecord>();
    record->tag = tag;
    record->id = rid;
    record->sched_id = choice == 0 ? rid : rid + "-" + std::to_string(choice);
    record->choice = choice;
    record->group = group;
    record->call_seed = tag ^ (static_cast<uint64_t>(choice) << 48);
    // The record keeps the checkpoint id (as does /health); /v1/models
    // reports the display alias (served_model_name, when set).
    record->model = cfg_.model_id;
    record->created_unix = created;
    record->chat = true;
    record->custom_tools = plan.custom_tools;
    record->stream = stream;
    record->include_usage = include_usage;
    record->include_obfuscation = include_obfuscation;
    record->report_service_tier = optional_field(body, "service_tier") != nullptr;
    if (const auto* metadata = optional_field(body, "metadata"))
      record->metadata = dgpp::text::json_text_of(*metadata);
    record->prompt_tokens = static_cast<int>(prompt.size());
    record->parser = std::make_unique<ToolCallParser>(
        markers_,
        [this](const std::vector<int64_t>& ids) {
          return frontend_->decode_ids(ids);
        },
        plan.schemas, popts);
    record->stop.stops = stops;
    record->reasoning_open = opens_thinking;
    record->writer = &w;
    record->logprobs = logprobs;
    record->arrived = arrived;

    SchedulerRequest sr;
    sr.id = record->sched_id;
    sr.boundaries = boundaries;
    sr.no_cache = !prefix_cache;
    sr.images = images;
    sr.prompt = prompt;
    sr.max_steps = steps;
    sr.sampling = sampling;
    sr.seed = seed + static_cast<uint64_t>(choice);
    sr.logprobs = logprobs;
    sr.grammar = plan.grammar;
    sr.logit_bias = logit_bias;
    records.push_back(std::move(record));
    requests.push_back(std::move(sr));
  }
  w.set_stream_tag(tag);
  enqueue_group(std::move(records), std::move(requests));
}

// ---------------------------------------------------------------------------
// POST /v1/completions (the legacy prompt API)
// ---------------------------------------------------------------------------

void GenerationService::route_completions(const HttpRequest& req,
                                          HttpResponseWriter& w) {
  dgpp::minijson::ParseResult parsed;
  try {
    parsed = dgpp::minijson::parse(req.body);
  } catch (const std::exception& e) {
    respond_error(w, 400, std::string("invalid JSON body: ") + e.what(),
                  "invalid_request_error", "body");
    return;
  }
  const dgpp::minijson::Value& body = parsed.root;
  if (!body.is_object()) {
    respond_error(w, 400, "the request body must be a JSON object",
                  "invalid_request_error", "body");
    return;
  }
  const dgpp::minijson::Value* model = body.find("model");
  if (model == nullptr || !model->is_string()) {
    respond_error(w, 400, "model is required and must be a string",
                  "invalid_request_error", "model");
    return;
  }
  if (!cfg_.model_matches(model->as_string())) {
    respond_error(w, 404, "the model '" + std::string(model->as_string()) +
                              "' does not exist on this server (serving '" +
                              cfg_.display_model_id() + "')",
                  "invalid_request_error", "model", "model_not_found");
    return;
  }
  const dgpp::minijson::Value* prompt = body.find("prompt");
  if (prompt == nullptr || !prompt->is_string()) {
    respond_error(w, 400,
                  "prompt is required and must be a string in v1 (prompt "
                  "arrays arrive with the batching stage)",
                  "invalid_request_error", "prompt");
    return;
  }

  int steps = 0;
  if (!parse_max_tokens(body, w, &steps, false)) return;
  bool stream = false;
  if (const auto* sv = optional_field(body, "stream")) {
    if (!sv->is_bool()) {
      respond_error(w, 400, "stream must be a boolean", "invalid_request_error", "stream");
      return;
    }
    stream = sv->as_bool();
  }
  bool include_usage = false, include_obfuscation = false;
  if (!parse_stream_options(body, w, stream, &include_usage, &include_obfuscation)) return;
  bool prefix_cache = true;
  if (const dgpp::minijson::Value* pcv = body.find("prefix_cache")) {
    if (!pcv->is_bool()) {
      respond_error(w, 400, "prefix_cache must be a boolean",
                    "invalid_request_error", "prefix_cache");
      return;
    }
    prefix_cache = pcv->as_bool(true);
  }

  sample::Params sampling;
  uint64_t seed = 0;
  if (!parse_sampling(body, w, &sampling, &seed)) return;
  // The legacy logprobs: an integer 0..5 (null = none).
  int logprobs = -1;
  if (const dgpp::minijson::Value* lp = optional_field(body, "logprobs")) {
    const double x = lp->is_number() ? lp->as_double(-1.0) : -1.0;
    if (!lp->is_number() || x != std::floor(x) || x < 0.0 || x > 5.0) {
      respond_error(w, 400, "logprobs must be an integer in [0, 5]",
                    "invalid_request_error", "logprobs");
      return;
    }
    if (!engine_->supports_logprobs()) {
      respond_error(w, 400, "logprobs are not available on this engine",
                    "invalid_request_error", "logprobs",
                    "logprobs_unsupported");
      return;
    }
    logprobs = static_cast<int>(x);
    sampling.logprobs = logprobs;
  }

  const char* unsupported[] = {"echo", "suffix", "top_logprobs",
                              "n",    "best_of", "user"};
  for (const char* param : unsupported) {
    if (body.find(param) != nullptr) {
      respond_error(w, 400,
                    std::string("'") + param +
                        "' is not supported in this server version",
                    "invalid_request_error", param, "unsupported_parameter");
      return;
    }
  }

  // stop and logit_bias.
  std::vector<std::string> stops;
  std::vector<dgpp::sched::LogitBias> logit_bias;
  if (!parse_stop(body, w, &stops) || !parse_logit_bias(body, w, &logit_bias))
    return;

  std::vector<int64_t> ids = frontend_->encode_text(prompt->as_string());
  if (ids.empty()) {
    respond_error(w, 400, "the prompt produced no tokens",
                  "invalid_request_error", "prompt");
    return;
  }
  const int64_t reserve =
      engine_->blocks_for_tokens(static_cast<int64_t>(ids.size()) + steps);
  if (reserve > engine_->pool_blocks_total()) {
    respond_error(w, 400,
                  "the request's token budget exceeds the model's KV capacity",
                  "invalid_request_error", "max_tokens",
                  "context_length_exceeded");
    return;
  }

  auto record = std::make_shared<StreamRecord>();
  record->tag = next_tag_.fetch_add(1, std::memory_order_relaxed);
  {
    char suffix[17];
    std::snprintf(suffix, sizeof(suffix), "%016llx",
                  static_cast<unsigned long long>(record->tag));
    record->id = "cmpl-" + std::string(suffix);
  }
  record->sched_id = record->id;
  record->group = std::make_shared<ChoiceGroup>();
  record->group->choices.resize(1);
  record->call_seed = record->tag;
  record->stop.stops = stops;
  // The record keeps the checkpoint id (as does /health); /v1/models
  // reports the display alias (served_model_name, when set).
  record->model = cfg_.model_id;
  record->created_unix = std::time(nullptr);
  record->chat = false;
  record->stream = stream;
  record->include_usage = include_usage;
  record->include_obfuscation = false;
  record->prompt_tokens = static_cast<int>(ids.size());
  record->writer = &w;
  w.set_stream_tag(record->tag);

  SchedulerRequest sr;
  sr.id = record->id;
  sr.boundaries = prompt_boundaries(ids);
  sr.no_cache = !prefix_cache;
  sr.prompt = std::move(ids);
  sr.max_steps = steps;
  sr.sampling = sampling;
  sr.seed = seed;
  sr.logprobs = logprobs;
  sr.logit_bias = logit_bias;
  record->logprobs = logprobs;
  record->arrived = std::chrono::steady_clock::now();
  enqueue_admission(std::move(record), std::move(sr));
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

void GenerationService::route_models(const HttpRequest& req,
                                     HttpResponseWriter& w) {
  const int64_t created = std::time(nullptr);
  if (req.path == "/v1/models") {
    w.respond(200, "application/json",
              "{\"object\":\"list\",\"data\":[" +
                  model_object(cfg_, cfg_.display_model_id(), created, sampling_available_,
                               cfg_.sampling_defaults, tool_calls_available(),
                               constraints_available(),
                               cfg_.reasoning_in_content, *frontend_, frontend_->supports_images() && engine_->supports_images()) +
                  "]}");
    return;
  }
  const std::string id = req.path.substr(std::string("/v1/models/").size());
  if (cfg_.model_matches(id)) {
    w.respond(200, "application/json",
              model_object(cfg_, cfg_.display_model_id(), created, sampling_available_,
                           cfg_.sampling_defaults, tool_calls_available(),
                           constraints_available(), cfg_.reasoning_in_content, *frontend_, frontend_->supports_images() && engine_->supports_images()));
    return;
  }
  respond_error(w, 404, "the model '" + id + "' does not exist",
                "invalid_request_error", "model", "model_not_found");
}

// ---------------------------------------------------------------------------
// GET /metrics, /v1/metrics (JSON scheduler meters, published by the engine)
// ---------------------------------------------------------------------------

Scheduler::Meters GenerationService::meters() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return meters_;
}

void GenerationService::route_metrics(HttpResponseWriter& w) {
  Scheduler::Meters m;
  Stats st;
  const auto prefills = engine_->prefill_monitor()->snapshot();
  dgpp::sched::SchedulerEngine::PrefixEngineStats pe;
  double snapshot_age_ms = 0;
  size_t pending_admissions = 0, pending_cancellations = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    m = meters_;
    st = stats_;
    pe = prefix_stats_;
    snapshot_age_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - meters_published_).count();
    pending_admissions = pending_admissions_.size();
    pending_cancellations = pending_cancels_.size();
    // Live gauges include admissions whose synchronous prefill has not yet
    // returned to the scheduler, as well as requests waiting in the HTTP queue.
    m.active = 0;
    m.queued = 0;
    m.prefilling = static_cast<int>(prefills.size());
    for (const auto& r : records_) {
      if (r->done || r->reject_overloaded) continue;
      const bool prefilling = std::any_of(prefills.begin(), prefills.end(),
                                         [&](const auto& p) { return p.id == r->sched_id; });
      if (prefilling || !r->ids.empty()) ++m.active;
      else ++m.queued;
    }
  }
  std::string out = "{\"scheduler\":{\"active\":";
  append_json_int(&out, m.active);
  out.append(",\"queued\":");
  append_json_int(&out, m.queued);
  out.append(",\"terminal\":");
  append_json_int(&out, m.terminal);
  out.append(",\"records\":");
  append_json_int(&out, m.records);
  out.append(",\"record_tokens\":");
  append_json_int(&out, m.record_tokens);
  out.append(",\"pool_blocks_total\":");
  append_json_int(&out, m.pool_blocks_total);
  out.append(",\"pool_blocks_in_use\":");
  append_json_int(&out, m.pool_blocks_in_use);
  out.append(",\"prefilling\":");
  append_json_int(&out, m.prefilling);
  out.append(",\"tokens_generated\":");
  append_json_int(&out, m.tokens_generated);
  out.append(",\"snapshot_age_ms\":");
  append_json_float(&out, snapshot_age_ms);
  // The throughput line's counters, cumulative: a scraper
  // differences them the way the line does.
  out.append(",\"prompts_prefilled\":");
  append_json_int(&out, m.prompts_prefilled);
  out.append(",\"prompt_tokens\":");
  append_json_int(&out, m.prompt_tokens);
  out.append(",\"prompt_tokens_computed\":");
  append_json_int(&out, m.prompt_tokens_computed);
  out.append(",\"decode_steps\":");
  append_json_int(&out, m.decode_steps);
  out.append(",\"decode_rows\":");
  append_json_int(&out, m.decode_rows);
  {
    char tbuf[192];
    std::snprintf(tbuf, sizeof(tbuf), ",\"prefill_ms\":%.1f,\"prefill_request_ms\":%.1f,\"step_ms\":%.1f",
                  m.prefill_ms, m.prefill_request_ms, m.step_ms);
    out.append(tbuf);
  }
  out.append("},\"service\":{\"requests_total\":");
  append_json_int(&out, static_cast<int64_t>(st.requests_total));
  out.append(",\"requests_shed\":");
  append_json_int(&out, static_cast<int64_t>(st.requests_shed));
  out.append(",\"pending_admissions\":");
  append_json_int(&out, static_cast<int64_t>(pending_admissions));
  out.append(",\"pending_cancellations\":");
  append_json_int(&out, static_cast<int64_t>(pending_cancellations));
  out.append(",\"requests_cancelled\":");
  append_json_int(&out, static_cast<int64_t>(st.requests_cancelled));
  out.append(",\"requests_shed_pool\":");
  append_json_int(&out, static_cast<int64_t>(st.requests_shed_pool));
  out.append(",\"requests_failed\":");
  append_json_int(&out, static_cast<int64_t>(st.requests_failed));
  out.append(failed() ? ",\"engine_failed\":true" : ",\"engine_failed\":false");
  out.append(",\"reservations_grown\":");
  append_json_int(&out, m.reservations_grown);
  out.append(",\"admission\":{\"mode\":\"");
  out.append(dgpp::sched::AdmissionPolicy::name(sched_.admission_policy().mode));
  out.append("\",\"window\":");
  append_json_int(&out, sched_.admission_policy().window_tokens);
  out.append(",\"prefill_budget_tokens\":");
  append_json_int(&out, sched_.admission_policy().prefill_budget_tokens);
  out.append(",\"prefill_idle_budget_tokens\":");
  append_json_int(&out, sched_.admission_policy().prefill_idle_budget_tokens);
  out.append("}");
  out.append(",\"tokens_out\":");
  append_json_int(&out, static_cast<int64_t>(st.tokens_out));
  out.append(",\"rejects_bad\":");
  append_json_int(&out, static_cast<int64_t>(st.rejects_bad));
  out.append(",\"tool_calls_out\":");
  append_json_int(&out, static_cast<int64_t>(st.tool_calls_out));
  out.append("}");
  // The prefix cache (M7): capacity, hits, tokens saved, entries taken and
  // evicted, blocks pinned, the arena's measured copy times and the TTFT
  // split — every number an operator needs to size the arena.
  const auto avg = [](double sum, int64_t n) { return n > 0 ? sum / static_cast<double>(n) : 0.0; };
  char buf[64];
  const auto append_ms = [&](const char* key, double v) {
    std::snprintf(buf, sizeof(buf), ",\"%s\":%.3f", key, v);
    out.append(buf);
  };
  out.append(",\"prefix_cache\":{\"enabled\":");
  out.append(m.prefix_slots > 0 ? "true" : "false");
  out.append(",\"slots\":");
  append_json_int(&out, m.prefix_slots);
  out.append(",\"entries\":");
  append_json_int(&out, m.prefix_entries);
  out.append(",\"hits\":");
  append_json_int(&out, m.prefix_hits);
  out.append(",\"misses\":");
  append_json_int(&out, m.prefix_misses);
  out.append(",\"tokens_saved\":");
  append_json_int(&out, m.prefix_tokens_saved);
  out.append(",\"snapshots\":");
  append_json_int(&out, m.prefix_snapshots);
  out.append(",\"close_entries\":");
  append_json_int(&out, m.prefix_close_entries);
  out.append(",\"rolling_snapshots\":");
  append_json_int(&out, m.prefix_rolling);
  out.append(",\"hop_snapshots\":");
  append_json_int(&out, m.prefix_hops);
  out.append(",\"evictions\":");
  append_json_int(&out, m.prefix_evictions);
  out.append(",\"duplicates\":");
  append_json_int(&out, m.prefix_duplicates);
  out.append(",\"skipped_no_block\":");
  append_json_int(&out, m.prefix_skipped_no_block);
  out.append(",\"skipped_no_slot\":");
  append_json_int(&out, m.prefix_skipped);
  out.append(",\"skipped_image_bytes\":");
  append_json_int(&out, m.prefix_skipped_image_bytes);
  out.append(",\"image_bytes\":");
  append_json_int(&out, m.prefix_image_bytes);
  out.append(",\"blocks_pinned\":");
  append_json_int(&out, m.prefix_blocks_pinned);
  out.append(",\"snapshot_bytes\":");
  append_json_int(&out, pe.snapshot_bytes);
  out.append(",\"arena_snapshots\":");
  append_json_int(&out, pe.snapshots);
  append_ms("snapshot_ms_avg", avg(pe.snapshot_ms, pe.snapshots));
  out.append(",\"arena_attaches\":");
  append_json_int(&out, pe.attaches);
  append_ms("attach_ms_avg", avg(pe.attach_ms, pe.attaches));
  out.append(",\"ttft_hit_count\":");
  append_json_int(&out, static_cast<int64_t>(st.ttft_hit_count));
  append_ms("ttft_hit_ms_avg", avg(st.ttft_hit_ms, static_cast<int64_t>(st.ttft_hit_count)));
  out.append(",\"ttft_miss_count\":");
  append_json_int(&out, static_cast<int64_t>(st.ttft_miss_count));
  append_ms("ttft_miss_ms_avg", avg(st.ttft_miss_ms, static_cast<int64_t>(st.ttft_miss_count)));
  out.append(",\"key\":");
  append_json_string(&out, cfg_.prefix_key);
  out.append("},\"prefill\":{\"requests\":[");
  int64_t total = 0, processed = 0, cached = 0;
  const auto now = std::chrono::steady_clock::now();
  for (size_t i = 0; i < prefills.size(); ++i) {
    const auto& p = prefills[i];
    if (i) out.push_back(',');
    out.append("{\"id\":");
    append_json_string(&out, p.id);
    out.append(",\"slot\":");
    append_json_int(&out, p.slot);
    out.append(",\"prompt_tokens\":");
    append_json_int(&out, p.total);
    out.append(",\"processed_tokens\":");
    append_json_int(&out, p.processed);
    out.append(",\"cached_tokens\":");
    append_json_int(&out, p.cached);
    out.append(",\"computed_tokens\":");
    append_json_int(&out, p.processed - p.cached);
    out.append(",\"remaining_tokens\":");
    append_json_int(&out, p.total - p.processed);
    out.append(",\"elapsed_ms\":");
    append_json_float(&out, std::chrono::duration<double, std::milli>(now - p.started).count());
    out.push_back('}');
    total += p.total;
    processed += p.processed;
    cached += p.cached;
  }
  out.append("],\"prompt_tokens\":");
  append_json_int(&out, total);
  out.append(",\"processed_tokens\":");
  append_json_int(&out, processed);
  out.append(",\"cached_tokens\":");
  append_json_int(&out, cached);
  out.append(",\"computed_tokens\":");
  append_json_int(&out, processed - cached);
  out.append(",\"remaining_tokens\":");
  append_json_int(&out, total - processed);
  out.append("}}");
  w.respond(200, "application/json", std::move(out));
}

// ---------------------------------------------------------------------------
// The admission interface
// ---------------------------------------------------------------------------

void GenerationService::enqueue_admission(std::shared_ptr<StreamRecord> record,
                                           SchedulerRequest request) {
  std::vector<std::shared_ptr<StreamRecord>> records;
  records.push_back(std::move(record));
  std::vector<SchedulerRequest> requests;
  requests.push_back(std::move(request));
  enqueue_group(std::move(records), std::move(requests));
}

void GenerationService::enqueue_group(
    std::vector<std::shared_ptr<StreamRecord>> records,
    std::vector<SchedulerRequest> requests) {
  std::lock_guard<std::mutex> lock(mutex_);
  ++stats_.requests_total;
  // EVERY record enters the lifecycle list at enqueue — the observer,
  // the disconnect hook, and the pump must see a request that is
  // still waiting for the engine thread (a disconnect can race the
  // engine pass, and the cancel has to land either way).
  for (auto& r : records) records_.push_back(r);
  // A request's choices are admitted or shed TOGETHER (n, 2026-09-06): a
  // half-shed group would answer with fewer choices than asked.
  if (shutdown_ || pending_admissions_.size() + records.size() >
                       static_cast<size_t>(cfg_.queue_limit)) {
    for (auto& r : records) {
      r->reject_overloaded = true;
      r->shutting_down = shutdown_;
      r->engine_failed = failed_;
    }
    records.front()->group->counted_shed = true;
    ++stats_.requests_shed;
    return;
  }
  if (records.front()->stream) records.front()->writer->begin_stream();
  for (size_t i = 0; i < records.size(); ++i)
    pending_admissions_.push_back(
        PendingAdmission{records[i], std::move(requests[i])});
}

void GenerationService::on_disconnect(uint64_t tag) {
  for (auto& pending : file_work_)
    if (pending.tag == tag) pending.writer = nullptr;
  std::lock_guard<std::mutex> lock(mutex_);
  // Every record on the connection (a request's n choices share the tag).
  for (auto& r : records_) {
    if (r->tag != tag) continue;
    r->writer_dead = true;
    r->writer = nullptr;
    if (!r->done && !r->cancel_armed) {
      // Stage 4's contract: a client disconnect maps onto the same
      // deterministic retire path as scripted cancellation.
      r->cancel_armed = true;
      pending_cancels_.push_back(PendingCancel{r->sched_id});
      if (!r->group->counted_cancelled) {
        r->group->counted_cancelled = true;
        ++stats_.requests_cancelled;
      }
      DGPP_LOG_INFO("serve: request {} cancelled by client disconnect",
                    r->sched_id);
    }
  }
  // An unknown tag: a long-finished record's connection finally closed.
}

// ---------------------------------------------------------------------------
// The observer (engine thread — the scheduler's inline callbacks)
// ---------------------------------------------------------------------------

void GenerationService::push_content(StreamRecord& r, std::string text) {
  if (text.empty()) return;
  r.content += text;
  mark_content_logprobs(r);
  if (r.stream) {
    ParserEvent ev;
    ev.kind = ParserEvent::Kind::kContent;
    ev.text = std::move(text);
    r.pending.push_back(std::move(ev));
  }
}

void GenerationService::mark_content_logprobs(StreamRecord& r) {
  // The stop scanner emits a prefix of its input and may hold a tail.
  // Attribute only bytes that actually reached message.content.
  while (r.content_span_cursor < r.content_spans.size()) {
    const auto& span = r.content_spans[r.content_span_cursor];
    if (span.begin >= (r.chat ? r.content.size() : r.out_text.size())) break;
    r.content_lps[span.token] = true;
    ++r.content_span_cursor;
  }
}

void GenerationService::request_stop(StreamRecord& r) {
  if (r.stopped) return;
  r.stopped = true;
  r.stop_tokens = static_cast<int>(r.ids.size());
  pending_stops_.push_back(r.sched_id);
}

void GenerationService::absorb(StreamRecord& r, ParserEvent ev) {
  // usage's reasoning_tokens: the ids up to and including </think>.
  if (ev.kind == ParserEvent::Kind::kReasoningClosed) r.reasoning_open = false;
  // Past a stop match nothing more is shown (the request is retiring).
  if (r.stop.hit) return;
  // The fold knob: reasoning rides as content text, the model's own
  // "</think>" included where it produced it — exactly the decode a
  // client of a server without a reasoning parser would see.
  if (cfg_.reasoning_in_content) {
    if (ev.kind == ParserEvent::Kind::kReasoning) {
      ev.kind = ParserEvent::Kind::kContent;
    } else if (ev.kind == ParserEvent::Kind::kReasoningClosed) {
      ev.kind = ParserEvent::Kind::kContent;
      ev.text = markers_.think_close.text;
    }
  }
  switch (ev.kind) {
    case ParserEvent::Kind::kReasoning:
      r.reasoning += ev.text;
      break;
    case ParserEvent::Kind::kContent:
      if (ev.text.empty()) return;
      for (const auto& span : ev.tokens)
        r.content_spans.push_back({span.token, r.content_input_bytes + span.begin,
                                   r.content_input_bytes + span.end});
      r.content_input_bytes += ev.text.size();
      if (r.stop.active()) {
        // The stop strings match the content: the scanner
        // shows what precedes a match and holds a tail that could still
        // begin one.
        std::string shown = r.stop.feed(ev.text);
        if (r.stop.hit) request_stop(r);
        if (shown.empty()) return;
        ev.text = std::move(shown);
      }
      r.content += ev.text;
      mark_content_logprobs(r);
      break;
    case ParserEvent::Kind::kToolCall:
      if (std::find(r.custom_tools.begin(), r.custom_tools.end(), ev.call.name) != r.custom_tools.end()) {
        try {
          const auto parsed = dgpp::minijson::parse(ev.call.arguments);
          const auto* input = parsed.root.find("input");
          if (!input || !input->is_string()) throw std::invalid_argument("input must be a string");
          ev.call.arguments = std::string(input->as_string());
          ev.call.custom = true;
        } catch (const std::exception&) {
          r.generation_error = "the model produced an invalid custom tool input";
          request_stop(r);
          return;
        }
      }
      r.calls.push_back(ev.call);
      stats_.tool_calls_out++;
      break;
    case ParserEvent::Kind::kReasoningClosed:
      return;  // structural only
  }
  if (r.stream) r.pending.push_back(std::move(ev));
}

void GenerationService::on_token(const std::string& id, int64_t token,
                                 int steps_done) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& r : records_) {
      if (r->sched_id != id || r->done) continue;
      if (steps_done == 1) {
        // The first token: the time to first token, split by whether the
        // prefix cache served the prompt's head (M7).
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - r->arrived)
                              .count();
        if (r->prefix_hit) {
          stats_.ttft_hit_count++;
          stats_.ttft_hit_ms += ms;
        } else {
          stats_.ttft_miss_count++;
          stats_.ttft_miss_ms += ms;
        }
      }
      r->ids.push_back(token);
      if (r->logprobs >= 0) r->content_lps.push_back(false);
      if (r->chat) {
        if (r->reasoning_open) ++r->reasoning_tokens;
        // The parser routes the id by state (reasoning / content / a tool
        // call block) and yields the exact text deltas of each run.
        std::vector<ParserEvent> events;
        r->parser->feed(token, &events);
        for (ParserEvent& ev : events) absorb(*r, std::move(ev));
      } else {
        // Exact incremental text: the suffix diff of successive full
        // decodes — UTF-8 splits and special tokens come out right by
        // construction (the tokenizer's own decode gates, pinned by its
        // differential goldens). The stop scanner decides
        // what of it is shown.
        const std::string full = frontend_->decode_ids(r->ids);
        std::string suffix;
        if (full.size() > r->text.size())
          suffix.assign(full, r->text.size(), std::string::npos);
        r->text = full;
        if (r->logprobs >= 0 && !suffix.empty()) {
          r->content_spans.push_back({r->ids.size() - 1, r->content_input_bytes,
                                      r->content_input_bytes + suffix.size()});
          r->content_input_bytes += suffix.size();
        }
        if (r->stop.hit) {
          suffix.clear();
        } else if (r->stop.active()) {
          suffix = r->stop.feed(suffix);
          if (r->stop.hit) request_stop(*r);
        }
        r->delta += suffix;
        r->out_text += suffix;
        mark_content_logprobs(*r);
      }
      stats_.tokens_out++;
      break;
    }
  }
  // The audit tap sees every engine event, even ones this record list
  // no longer knows (post-shutdown ticks) — cross-rank comparability
  // is the tap's entire job.
  if (audit_) audit_->on_token(id, token, steps_done);
}

void GenerationService::on_prefix(const std::string& id, const char* op,
                                  int64_t position, int slot) {
  if (std::strcmp(op, "attach") == 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& r : records_) {
      if (r->sched_id != id) continue;
      r->prefix_hit = true;
      r->prefix_position = position;
      break;
    }
  }
  if (audit_) audit_->on_prefix(id, op, position, slot);
}

std::vector<int64_t> GenerationService::prompt_boundaries(
    const std::vector<int64_t>& prompt) const {
  std::vector<int64_t> out;
  if (boundary_ids_.empty()) return out;
  for (size_t i = 1; i < prompt.size(); ++i)
    if (std::binary_search(boundary_ids_.begin(), boundary_ids_.end(), prompt[i]))
      out.push_back(static_cast<int64_t>(i));
  return out;
}

void GenerationService::on_token_logprobs(const std::string& id,
                                          int steps_done,
                                          const sample::Result& logprobs) {
  (void)steps_done;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& r : records_) {
    if (r->sched_id != id || r->done) continue;
    if (r->logprobs >= 0) {
      r->lps.push_back(logprobs);
      r->lps_reported.push_back(false);
    }
    break;
  }
}

namespace {

void append_bytes_array(std::string* out, const std::string& s) {
  out->push_back('[');
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) out->push_back(',');
    append_json_int(out, static_cast<unsigned char>(s[i]));
  }
  out->push_back(']');
}

}  // namespace

namespace {
void sanitize_utf8(std::string* text);  // the delta streams' UTF-8 discipline, below
}  // namespace

std::string GenerationService::logprobs_content(const StreamRecord& r,
                                                size_t from, size_t to,
                                                bool unreported_only) const {
  // {"content":[{"token","logprob","bytes","top_logprobs":[...]}, ...]}
  std::string out = "{\"content\":[";
  bool first = true;
  for (size_t i = from; i < to && i < r.lps.size(); ++i) {
    if (!r.content_lps[i] || (unreported_only && r.lps_reported[i])) continue;
    if (!first) out.push_back(',');
    first = false;
    const sample::Result& lp = r.lps[i];
    const std::string tok = frontend_->decode_ids({r.ids[i]});
    out.append("{\"token\":");
    {
      std::string shown(tok);  // the exact bytes ride in "bytes"; the text is UTF-8
      sanitize_utf8(&shown);
      append_json_string(&out, shown);
    }
    out.append(",\"logprob\":");
    append_json_float(&out, lp.logprob);
    out.append(",\"bytes\":");
    append_bytes_array(&out, tok);
    out.append(",\"top_logprobs\":[");
    for (size_t j = 0; j < lp.top_logprobs.size(); ++j) {
      if (j) out.push_back(',');
      const std::string alt = frontend_->decode_ids({lp.top_logprobs[j].first});
      out.append("{\"token\":");
      {
        std::string shown(alt);
        sanitize_utf8(&shown);
        append_json_string(&out, shown);
      }
      out.append(",\"logprob\":");
      append_json_float(&out, lp.top_logprobs[j].second);
      out.append(",\"bytes\":");
      append_bytes_array(&out, alt);
      out.push_back('}');
    }
    out.append("]}");
  }
  out.append("],\"refusal\":null}");
  return out;
}

std::string GenerationService::take_content_logprobs(StreamRecord& r) const {
  if (r.logprobs < 0) return {};
  std::string out = logprobs_content(r, 0, r.lps.size(), true);
  if (out == "{\"content\":[],\"refusal\":null}") return {};
  for (size_t i = 0; i < r.lps.size(); ++i)
    if (r.content_lps[i]) r.lps_reported[i] = true;
  return out;
}

std::string GenerationService::legacy_logprobs(const StreamRecord& r, size_t from,
                                              size_t to) const {
  // {"tokens":[...],"token_logprobs":[...],"top_logprobs":[{tok: lp}],
  //  "text_offset":[...]}
  std::string tokens = "[", lps = "[", tops = "[", offsets = "[";
  size_t offset = 0;
  bool first = true;
  for (size_t i = 0; i < r.lps.size() && i < to; ++i) {
    if (!r.content_lps[i]) continue;
    const std::string tok = frontend_->decode_ids({r.ids[i]});
    if (i < from) {
      offset += tok.size();
      continue;
    }
    if (!first) {
      tokens.push_back(',');
      lps.push_back(',');
      tops.push_back(',');
      offsets.push_back(',');
    }
    first = false;
    append_json_string(&tokens, tok);
    append_json_float(&lps, r.lps[i].logprob);
    tops.push_back('{');
    for (size_t j = 0; j < r.lps[i].top_logprobs.size(); ++j) {
      if (j) tops.push_back(',');
      append_json_string(&tops, frontend_->decode_ids({r.lps[i].top_logprobs[j].first}));
      tops.push_back(':');
      append_json_float(&tops, r.lps[i].top_logprobs[j].second);
    }
    tops.push_back('}');
    append_json_int(&offsets, static_cast<int64_t>(offset));
    offset += tok.size();
  }
  return "{\"tokens\":" + tokens + "],\"token_logprobs\":" + lps +
         "],\"top_logprobs\":" + tops + "],\"text_offset\":" + offsets + "]}";
}

void GenerationService::on_retire(const std::string& id,
                                 const Scheduler::Result& result) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& r : records_) {
      if (r->sched_id != id) continue;
      if (r->chat && r->parser && !r->done) {
        // An open block at the end flushes as content.
        std::vector<ParserEvent> events;
        r->parser->finish(&events);
        for (ParserEvent& ev : events) absorb(*r, std::move(ev));
      }
      if (!r->done) {
        // The stop scanner's held tail is real text when nothing matched.
        if (r->chat) {
          push_content(*r, r->stop.finish());
        } else {
          const std::string rest = r->stop.finish();
          r->delta += rest;
          r->out_text += rest;
          mark_content_logprobs(*r);
        }
      }
      r->done = true;
      r->reason = result.reason;
      // A stopped request's tokens past the match were generated but never
      // shown: the usage counts up to the token that completed the match.
      r->completion_tokens = r->stopped ? r->stop_tokens : result.steps_done;
      if (r->group) {
        r->group->completion_tokens += r->completion_tokens;
        r->group->reasoning_tokens += r->reasoning_tokens;
        if (r->choice == 0)
          r->group->cached_tokens =
              r->prefix_hit ? static_cast<int>(r->prefix_position) : 0;
      }
      if (result.reason == Scheduler::Result::Reason::kPoolExhausted && !r->group->counted_pool) {
        r->group->counted_pool = true;
        ++stats_.requests_shed_pool;
      }
      break;
    }
  }
  if (result.reason == Scheduler::Result::Reason::kPoolExhausted)
    DGPP_LOG_WARN("serve: request '{}' cut short at KV pool exhaustion after {} "
                  "tokens (finish_reason length; grow-on-demand shed)",
                  id, result.steps_done);
  if (audit_) audit_->on_retire(id, result);
}

// ---------------------------------------------------------------------------
// idle(): the record pump (HTTP thread)
// ---------------------------------------------------------------------------

void GenerationService::enqueue_file_work(const HttpRequest& req, HttpResponseWriter& writer, bool chat) {
  if (file_work_.size() >= static_cast<size_t>(cfg_.file_inputs.workers)) {
    respond_error(writer, 503, "file preprocessing is busy; retry later", "server_error", "file", "file_processing_busy");
    return;
  }
  const uint64_t tag = next_tag_.fetch_add(1, std::memory_order_relaxed);
  writer.set_stream_tag(tag);
  file_work_.push_back({tag, &writer, chat, std::async(std::launch::async, [this, req, chat] {
    if (chat) return FileResponse{200, "application/json", file_inputs_.prepare(req).body};
    return file_inputs_.route(req);
  })});
  pending_file_count_.fetch_add(1);
}

void GenerationService::pump_file_work() {
  for (size_t i = 0; i < file_work_.size();) {
    auto& pending = file_work_[i];
    if (pending.result.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) { ++i; continue; }
    auto job = std::move(pending);
    file_work_.erase(file_work_.begin() + static_cast<std::ptrdiff_t>(i));
    if (!job.writer) { pending_file_count_.fetch_sub(1); continue; }
    try {
      auto result = job.result.get();
      if (job.chat) {
        HttpRequest request;
        request.method = "POST"; request.path = "/v1/chat/completions"; request.body = std::move(result.body);
        route_chat_completions(request, *job.writer);
      } else job.writer->respond(result.status, result.content_type, std::move(result.body));
    } catch (const FileInputError& e) {
      respond_error(*job.writer, e.status, e.what(), e.status >= 500 ? "server_error" : "invalid_request_error", e.param, e.code);
    } catch (const std::exception&) {
      respond_error(*job.writer, 500, "file processing failed", "server_error", "file", "file_processing_failed");
    }
    pending_file_count_.fetch_sub(1);
  }
}

void GenerationService::idle() { pump_file_work(); pump_records(); }

void GenerationService::write_stream_event(StreamRecord& r, std::string event, bool usage) {
  event.pop_back();  // every caller supplies a complete completion JSON object
  if (r.include_usage && !usage) event.append(",\"usage\":null");
  if (r.report_service_tier) event.append(",\"service_tier\":\"default\"");
  if (r.chat && r.include_obfuscation && !usage) {
    // Pad to 128-byte buckets, with at least 16 random padding characters.
    // This is transport padding only; it never enters the model's transcript.
    event.append(",\"obfuscation\":\"");
    const size_t padding = 16 + (128 - (event.size() + 16 + 2) % 128) % 128;
    constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (size_t i = 0; i < padding; ++i) event.push_back(alphabet[obfuscation_rng_() % 62]);
    event.push_back('"');
  }
  event.push_back('}');
  if (!r.chat && usage) {
    const auto at = event.find("chat.completion.chunk");
    if (at != std::string::npos) event.replace(at, 21, "text_completion");
  }
  r.writer->write_event(event);
}

// ---------------------------------------------------------------------------
// UTF-8 discipline for the delta streams (the soak's find, 2026-09-05): a
// token's decoded bytes can end inside a multi-byte character, and a JSON
// text that is not UTF-8 breaks a strict client (Python's json.loads on
// the payload bytes raised, and the soak's chat workers died on the first
// accented name). carry_utf8 prepends the field's held bytes, holds back an
// incomplete trailing sequence for the next delta, and replaces invalid
// bytes with U+FFFD; finish_utf8 renders what is still held at the end.
// ---------------------------------------------------------------------------
namespace {

size_t utf8_sequence_length(unsigned char b) {
  if (b < 0x80) return 1;
  if ((b & 0xE0) == 0xC0) return 2;
  if ((b & 0xF0) == 0xE0) return 3;
  if ((b & 0xF8) == 0xF0) return 4;
  return 0;  // a stray continuation or an invalid lead byte
}

constexpr const char* kReplacement = "\xEF\xBF\xBD";  // U+FFFD

void carry_utf8(std::string* text, std::string* carry) {
  if (!carry->empty()) {
    text->insert(0, *carry);
    carry->clear();
  }
  std::string out;
  out.reserve(text->size());
  const size_t n = text->size();
  size_t i = 0;
  while (i < n) {
    const unsigned char b = static_cast<unsigned char>((*text)[i]);
    const size_t len = utf8_sequence_length(b);
    if (len == 0) {
      out.append(kReplacement);
      ++i;
      continue;
    }
    bool continuation_ok = true;
    const size_t have = std::min(len, n - i);
    for (size_t k = 1; k < have; ++k)
      if ((static_cast<unsigned char>((*text)[i + k]) & 0xC0) != 0x80) continuation_ok = false;
    if (!continuation_ok) {
      out.append(kReplacement);
      ++i;
      continue;
    }
    if (have < len) {  // incomplete at the end: the next delta completes it
      carry->assign(*text, i, n - i);
      break;
    }
    out.append(*text, i, len);
    i += len;
  }
  text->swap(out);
}

std::string finish_utf8(std::string* carry) {
  if (carry->empty()) return {};
  carry->clear();
  return kReplacement;  // an incomplete character at the very end
}

void sanitize_utf8(std::string* text) {
  std::string carry;
  carry_utf8(text, &carry);
  text->append(finish_utf8(&carry));
}

}  // namespace

void GenerationService::flush_chat_stream(StreamRecord& r) {
  // Take the unflushed events (and, when a chunk will carry them, the
  // logprobs entries since the last flush) under the lock; format and
  // write outside it.
  std::vector<ParserEvent> events;
  std::string lp_json;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    events.swap(r.pending);
    lp_json = take_content_logprobs(r);
  }
  if (!r.first_chunk_sent) {
    r.first_chunk_sent = true;
    write_stream_event(r,
        chat_chunk_first(r.id, r.created_unix, r.model, r.choice));
  }
  for (const ParserEvent& ev : events) {
    std::string delta;
    switch (ev.kind) {
      case ParserEvent::Kind::kReasoning: {
        std::string text = ev.text;
        carry_utf8(&text, &r.carry_reasoning);
        if (text.empty()) continue;  // wholly held back: an incomplete character
        delta = delta_text("reasoning_content", text);
        break;
      }
      case ParserEvent::Kind::kContent: {
        std::string text = ev.text;
        carry_utf8(&text, &r.carry_content);
        if (text.empty()) continue;
        delta = delta_text("content", text);
        break;
      }
      case ParserEvent::Kind::kToolCall: {
        const int index = r.calls_announced++;
        const std::string call_id = tool_call_id(r.call_seed, index);
        write_stream_event(r, chat_chunk_delta(
            r.id, r.created_unix, r.model,
            delta_tool_call_start(index, call_id, ev.call.name, ev.call.custom), "",
            r.choice));
        std::string args = ev.call.arguments;
        sanitize_utf8(&args);  // a whole call's arguments: complete by construction
        delta = delta_tool_call_arguments(index, args, ev.call.custom);
        break;
      }
      case ParserEvent::Kind::kReasoningClosed:
        continue;
    }
    write_stream_event(r, chat_chunk_delta(r.id, r.created_unix, r.model,
                                           delta, "", r.choice));
  }
  // Logprobs follow the batch's visible deltas. An empty delta is valid;
  // provenance may arrive after a buffered parser block or stop tail.
  if (!lp_json.empty())
    write_stream_event(r, chat_chunk_delta(r.id, r.created_unix, r.model,
                                           "{}", lp_json, r.choice));
}

// The stream's end: whatever a field still holds is an incomplete
// character — U+FFFD, in one last delta per field.
void GenerationService::flush_stream_carries(StreamRecord& r) {
  if (r.chat) {
    for (auto [carry, field] : {std::pair{&r.carry_reasoning, "reasoning_content"},
                                std::pair{&r.carry_content, "content"}}) {
      const std::string rest = finish_utf8(carry);
      if (rest.empty()) continue;
      write_stream_event(r, chat_chunk_delta(r.id, r.created_unix, r.model,
                                             delta_text(field, rest), "",
                                             r.choice));
    }
    return;
  }
  const std::string rest = finish_utf8(&r.carry_text);
  if (!rest.empty())
    write_stream_event(r, text_chunk_delta(r.id, r.created_unix, r.model, rest, ""));
}

void GenerationService::flush_legacy_stream(StreamRecord& r) {
  std::string delta;
  std::string lp_json;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    delta.swap(r.delta);
    if (!delta.empty() && r.logprobs >= 0 && r.lps.size() > r.lps_flushed) {
      lp_json = legacy_logprobs(r, r.lps_flushed, r.lps.size());
      r.lps_flushed = r.lps.size();
    }
  }
  if (!r.first_chunk_sent) {
    r.first_chunk_sent = true;
    write_stream_event(r, text_chunk_first(r.id, r.created_unix, r.model));
  }
  carry_utf8(&delta, &r.carry_text);
  if (!delta.empty())
    write_stream_event(r,
        text_chunk_delta(r.id, r.created_unix, r.model, delta, lp_json));
}

void GenerationService::pump_records() {
  // Snapshot the drainable records under the lock; writers/formatting
  // happen outside it.
  std::vector<std::shared_ptr<StreamRecord>> live;
  std::vector<std::shared_ptr<StreamRecord>> finished;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& r : records_) {
      if (r->done || r->reject_overloaded)
        finished.push_back(r);
      else
        live.push_back(r);
    }
  }

  for (auto& r : live) {
    if (r->writer == nullptr || r->writer_dead) continue;
    if (!r->stream) continue;  // one-shots answer once, at finish
    if (r->chat)
      flush_chat_stream(*r);
    else
      flush_legacy_stream(*r);
  }

  for (auto& r : finished) {
    if (r->writer != nullptr && !r->writer_dead) {
      ChoiceGroup& g = *r->group;
      if (r->reject_overloaded || r->shutting_down || r->engine_failed || !r->generation_error.empty()) {
        // A shed request, or one the stop interrupted (M6 6c): the
        // one-shot gets the 503 object; a stream that already began gets
        // the error event + [DONE] (the only shape an SSE client can
        // see) — after the tokens it did produce, which were real. An
        // engine failure (the v1 failure semantics) takes the same two
        // shapes under the engine_failure code: the tokens a stream got
        // were committed on every rank before the failure. A request's n
        // choices share one answer: the first choice through here writes
        // it (every choice's committed tokens first), the others find it
        // ended.
        const bool failed = r->engine_failed;
        const bool shutdown = r->shutting_down || failed;
        std::string failure;
        if (failed) {
          std::lock_guard<std::mutex> lock(mutex_);
          failure = failure_;
        }
        const std::string msg = !r->generation_error.empty() ? r->generation_error :
            failed ? (r->reject_overloaded
                          ? "the engine failed (" + failure +
                                "); the service is restarting — retry"
                          : "the engine failed (" + failure +
                                ") — this response is incomplete; the "
                                "service is restarting, retry")
            : !shutdown ? "the server is overloaded — the admission queue is "
                          "full; retry after backing off"
            : r->reject_overloaded
                ? "the server is shutting down; retry on another instance"
                : "the server is shutting down — this response is "
                  "incomplete; retry on another instance";
        const char* code = !r->generation_error.empty() ? "invalid_tool_output" : failed ? "engine_failure"
                           : shutdown ? "server_shutdown"
                                      : "overloaded";
        if (g.ended) {
          // A sibling choice already answered for the request.
        } else if (r->stream && r->writer->stream_open()) {
          if (shutdown && !r->reject_overloaded) {
            for (auto& sib : finished) {
              if (sib->group != r->group || sib->writer == nullptr ||
                  sib->writer_dead)
                continue;
              if (sib->chat)
                flush_chat_stream(*sib);
              else
                flush_legacy_stream(*sib);
              flush_stream_carries(*sib);
            }
          }
          // Headers were already sent in handle(); an SSE client can
          // only see an error event + [DONE] (OpenAI's stream-error
          // shape), whether or not a role chunk preceded it.
          std::string ev = "{\"error\":{\"message\":";
          append_json_string(&ev, msg);
          ev.append(",\"type\":\"server_error\",\"param\":null,\"code\":\"");
          ev.append(code);
          ev.append("\"}}");
          r->writer->write_event(ev);
          r->writer->write_event("[DONE]");
          r->writer->end_stream();
          r->first_chunk_sent = true;
          g.ended = true;
        } else {
          respond_error(*r->writer, 503, msg, "server_error", "", code);
          g.ended = true;
        }
      } else if (r->stream) {
        // Flush any straggler events, then this choice's terminal chunk.
        // The final chunk carries any remaining visible-token logprobs.
        // EOS and hidden reasoning/tool tokens are not content logprobs.
        // The stream ends — the usage chunk, [DONE] — once every choice
        // of the request is done.
        if (r->chat)
          flush_chat_stream(*r);
        else
          flush_legacy_stream(*r);
        flush_stream_carries(*r);
        std::string lp_json;
        if (r->logprobs >= 0) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (r->lps.size() > r->lps_flushed) {
            lp_json = r->chat ? take_content_logprobs(*r)
                              : legacy_logprobs(*r, r->lps_flushed, r->lps.size());
            r->lps_flushed = r->lps.size();
          }
        }
        const char* finish = finish_reason(r->reason, !r->calls.empty());
        write_stream_event(*r,
            r->chat ? chat_chunk_final(r->id, r->created_unix, r->model,
                                       finish, lp_json, r->choice)
                    : text_chunk_final(r->id, r->created_unix, r->model,
                                       finish, lp_json));
        if (++g.finished == g.n && !g.ended) {
          if (r->include_usage) {
            write_stream_event(*r, chat_chunk_usage(
                r->id, r->created_unix, r->model, r->prompt_tokens,
                g.completion_tokens, g.cached_tokens, g.reasoning_tokens), true);
          }
          r->writer->write_event("[DONE]");
          r->writer->end_stream();
          g.ended = true;
        }
      } else {
        // The one-shot completion object: this choice's JSON now, the
        // answer once every choice of the request is in. Its texts are
        // complete by construction except for a last character a cap cut
        // in half: that one becomes U+FFFD (JSON text must be UTF-8).
        sanitize_utf8(&r->content);
        sanitize_utf8(&r->reasoning);
        sanitize_utf8(&r->out_text);
        for (ToolCall& call : r->calls) sanitize_utf8(&call.arguments);
        std::string lp_json;
        if (r->logprobs >= 0) {
          std::lock_guard<std::mutex> lock(mutex_);
          lp_json = r->chat ? logprobs_content(*r, 0, r->lps.size())
                            : legacy_logprobs(*r);
        }
        const char* finish = finish_reason(r->reason, !r->calls.empty());
        if (r->chat) {
          // {"role","content"(null when only calls),"reasoning_content"?,
          //  "tool_calls"?}
          std::string msg = "{\"role\":\"assistant\",\"content\":";
          if (r->content.empty() && !r->calls.empty())
            msg.append("null");
          else
            append_json_string(&msg, r->content);
          if (!r->reasoning.empty()) {
            msg.append(",\"reasoning_content\":");
            append_json_string(&msg, r->reasoning);
          }
          if (!r->calls.empty()) {
            msg.append(",\"tool_calls\":[");
            for (size_t i = 0; i < r->calls.size(); ++i) {
              if (i) msg.push_back(',');
              msg.append("{\"id\":");
              append_json_string(
                  &msg, tool_call_id(r->call_seed, static_cast<int>(i)));
              msg.append(r->calls[i].custom ? ",\"type\":\"custom\",\"custom\":{\"name\":" :
                                             ",\"type\":\"function\",\"function\":{\"name\":");
              append_json_string(&msg, r->calls[i].name);
              msg.append(r->calls[i].custom ? ",\"input\":" : ",\"arguments\":");
              append_json_string(&msg, r->calls[i].arguments);
              msg.append("}}");
            }
            msg.push_back(']');
          }
          msg.append(",\"refusal\":null}");
          g.choices[static_cast<size_t>(r->choice)] =
              chat_choice_json(r->choice, msg, finish, lp_json);
        } else {
          g.choices[static_cast<size_t>(r->choice)] =
              text_choice_json(r->choice, r->out_text, finish, lp_json);
        }
        if (++g.finished == g.n && !g.ended) {
          std::string joined;
          for (size_t i = 0; i < g.choices.size(); ++i) {
            if (i) joined.push_back(',');
            joined += g.choices[i];
          }
          std::string response = r->chat ? chat_completion_body(r->id, r->created_unix, r->model,
                                             joined, r->prompt_tokens,
                                             g.completion_tokens,
                                             g.cached_tokens,
                                             g.reasoning_tokens)
                      : text_completion_body(r->id, r->created_unix, r->model,
                                             joined, r->prompt_tokens,
                                             g.completion_tokens,
                                             g.cached_tokens);
          response.pop_back();
          if (r->report_service_tier) response.append(",\"service_tier\":\"default\"");
          if (!r->metadata.empty()) response.append(",\"metadata\":" + r->metadata);
          response.push_back('}');
          r->writer->respond(200, "application/json", std::move(response));
          g.ended = true;
        }
      }
    }
    // Records leave the list only here — done and (writer drained or
    // dead). A cancelled-but-still-generating record keeps its entry
    // until on_retire lands.
  }

  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& r : finished) {
    auto it = std::find(records_.begin(), records_.end(), r);
    if (it != records_.end()) records_.erase(it);
  }
}

// ---------------------------------------------------------------------------
// engine_pass() — the app's engine loop body
// ---------------------------------------------------------------------------

bool GenerationService::engine_pass(const PreTickHook& pre_tick) {
  std::vector<PendingAdmission> admissions;
  std::vector<PendingCancel> cancels;
  std::vector<std::string> stops;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    admissions.swap(pending_admissions_);
    cancels.swap(pending_cancels_);
    stops.swap(pending_stops_);
  }
  PassEvents events;
  for (auto& a : admissions) {
    bool admitted = false;
    bool journaled = false;
    try {
      if (pre_tick) {
        // The journal serializes the EXACT request the scheduler takes
        // (a copy — std::move would leave a husk for the record pump).
        events.submits.push_back(a.request);
        journaled = true;
        admitted = sched_.try_submit(events.submits.back());
      } else {
        admitted = sched_.try_submit(std::move(a.request));
      }
    } catch (const std::exception& e) {
      // Unreachable by construction (the route validated everything
      // the scheduler re-checks) — but a bug here must not wedge the
      // waiting client.
      DGPP_LOG_ERROR("serve: admission rejected: {}", e.what());
    }
    if (!admitted) {
      // A shed (queue full) or a rejected request died HERE, on rank
      // 0 — peers must never learn it existed, or their identical
      // queues would diverge from rank 0's.
      if (journaled) events.submits.pop_back();
      std::lock_guard<std::mutex> lock(mutex_);
      a.record->reject_overloaded = true;
      if (!a.record->group->counted_shed) {
        a.record->group->counted_shed = true;
        ++stats_.requests_shed;
      }
    }
  }
  for (const auto& c : cancels) {
    // Only cancels that HIT ride the journal (rank 0's scheduler state
    // changed). A late cancel is a no-op everywhere — peers' state is
    // identical, so replaying it would no-op there too; silence is
    // cheaper than noise.
    if (sched_.cancel(c.scheduler_id) && pre_tick)
      events.cancels.push_back(c.scheduler_id);
  }
  // The stop-string retires: like cancels — only the ones
  // that hit ride the journal, every rank retires the request at this
  // quantum with Reason::kStop.
  for (const std::string& id : stops) {
    if (sched_.stop(id) && pre_tick) events.stops.push_back(id);
  }

  // The fixed journal position: this record and the tick below are one
  // atomic unit — rank 0 never ticks without broadcasting, a peer
  // never ticks without a record (see fabric_serve.hpp). The prefix
  // cache's digest after the previous tick rides along (M7).
  if (sched_.prefix_slots() > 0) {
    events.has_prefix_digest = true;
    events.prefix_digest = sched_.prefix_digest();
  }
  // The op stream's fold after the previous tick (M9): the peers compare
  // theirs before applying this record.
  if (audit_ != nullptr && audit_->has_digest()) {
    events.has_op_digest = true;
    events.op_digest = audit_->digest();
  }
  if (pre_tick) pre_tick(events);

  const bool more = sched_.tick();  // may throw on scheduler contract
                                       // violations — the app treats
                                       // that as fatal (operator class)

  {
    std::lock_guard<std::mutex> lock(mutex_);
    meters_ = sched_.meters();
    prefix_stats_ = engine_->prefix_engine_stats();
    meters_published_ = std::chrono::steady_clock::now();
    return more || !pending_admissions_.empty();
  }
}

int GenerationService::begin_shutdown() {
  file_inputs_.stop();
  std::lock_guard<std::mutex> lock(mutex_);
  shutdown_ = true;
  for (auto& a : pending_admissions_) {
    a.record->reject_overloaded = true;
    a.record->shutting_down = true;
    if (!a.record->group->counted_shed) {
      a.record->group->counted_shed = true;
      ++stats_.requests_shed;
    }
  }
  pending_admissions_.clear();
  int interrupted = 0;
  for (auto& r : records_) {
    if (r->done || r->reject_overloaded || r->shutting_down) continue;
    // Live (queued or generating): the next pass's cancel sweep retires
    // it on every rank before any engine op — exact, since the sweep
    // precedes the step — and on_retire marks it done; the pump then
    // answers it with the shutdown error, not a finish.
    r->shutting_down = true;
    if (!r->group->counted_cancelled) {
      r->group->counted_cancelled = true;
      ++stats_.requests_cancelled;
    }
    pending_cancels_.push_back(PendingCancel{r->sched_id});
    ++interrupted;
  }
  return interrupted;
}

bool GenerationService::drained() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (pending_file_count_.load() != 0) return false;
  if (!pending_admissions_.empty() || !pending_cancels_.empty()) return false;
  for (const auto& r : records_)
    if (r->writer != nullptr && !r->writer_dead) return false;  // an answer owed
  return true;
}

int GenerationService::fail_engine(const std::string& what) {
  file_inputs_.stop();
  engine_->prefill_monitor()->clear();
  std::lock_guard<std::mutex> lock(mutex_);
  if (!failed_) {
    failed_ = true;
    failure_ = what;
  }
  shutdown_ = true;  // the door closes with the engine
  for (auto& a : pending_admissions_) {
    a.record->reject_overloaded = true;
    a.record->engine_failed = true;
    if (!a.record->group->counted_shed) {
      a.record->group->counted_shed = true;
      ++stats_.requests_shed;
    }
  }
  pending_admissions_.clear();
  pending_cancels_.clear();  // no pass will ever apply them
  int interrupted = 0;
  for (auto& r : records_) {
    if (r->done || r->reject_overloaded || r->shutting_down || r->engine_failed)
      continue;
    // Live (queued or generating): no retire will come — the pump finishes
    // it now, after whatever it produced (an open parser block flushes as
    // content, as at a retire).
    if (r->chat && r->parser) {
      std::vector<ParserEvent> events;
      r->parser->finish(&events);
      for (ParserEvent& ev : events) absorb(*r, std::move(ev));
    }
    r->engine_failed = true;
    r->done = true;
    if (!r->group->counted_failed) {
      r->group->counted_failed = true;
      ++stats_.requests_failed;
    }
    ++interrupted;
  }
  return interrupted;
}

bool GenerationService::failed() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return failed_;
}

GenerationService::Stats GenerationService::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

}  // namespace dgpp::serve
