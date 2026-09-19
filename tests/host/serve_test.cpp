// Generation-service tests using the real HTTP server with a deterministic
// FakeEngine and FakeFrontend. These run without a GPU or model cache and
// cover response formats, request validation and lifecycle behavior:
//   * the exact chat.completion / chat.completion.chunk shapes;
//   * stream lifecycle: role chunk, ordered content deltas whose
//     concatenation equals the full text, finish_reason, usage,
//     [DONE];
//   * EOS → "stop" vs the steps cap → "length";
//   * the unsupported-field checks (sampling, tools, model) with the OpenAI
//     error object naming the param;
//   * overload: 503 at the admission door;
//   * client disconnect mid-stream → scheduler cancellation (metrics);
//   * /v1/models, /health, /metrics and its /v1/metrics alias.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <mutex>
#include <optional>
#include <cstring>
#include <thread>
#include <tuple>
#include <vector>

#include "common/log.hpp"

#include "common/test.hpp"
#include "sched/scheduler.hpp"
#include "serve/generation_service.hpp"
#include "serve/image_inputs.hpp"
#include "serve/http_server.hpp"
#include "serve/serve_stats.hpp"
#include "text/chat_template.hpp"

namespace {

using dgpp::sched::SchedulerEngine;
using dgpp::serve::GenerationService;
using dgpp::serve::HttpServer;
using dgpp::serve::ModelFrontend;
using dgpp::serve::ServiceConfig;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

constexpr int32_t kFakeEos = 999;  // the fake's end-of-sequence id

// The deterministic fake model: token i of a request whose rendered
// prompt is P bytes long is ((len*31 + i*7) % 250) + 1 — printable-ish,
// never 0, never kFakeEos. A prompt with len % 4 == 3 answers EOS as
// its second token (an early-stop scenario the gate can target).
// Printable ASCII without the two JSON-escaped characters, so a fake's text
// appears verbatim in a response (the service renders every text as UTF-8
// since the 2026-09-05 soak, replacing what is not — the earlier formula's
// bytes above 0x7F were exactly that).
int32_t fake_token(size_t prompt_len, int index) {
  static const std::string alphabet = []() {
    std::string a;
    for (int c = 33; c <= 126; ++c)
      if (c != '"' && c != '\\') a.push_back(static_cast<char>(c));
    return a;
  }();
  return static_cast<int32_t>(static_cast<unsigned char>(
      alphabet[(prompt_len * 31 + static_cast<size_t>(index) * 7) % alphabet.size()]));
}
bool fake_eos_second(size_t prompt_len) { return prompt_len % 4 == 3; }
bool fake_eos_prefill(size_t prompt_len) { return prompt_len % 4 == 2; }

class FakeEngine : public SchedulerEngine {
 public:
  std::atomic<bool> images_available{false};
  std::atomic<int> image_prefills{0};
  bool supports_images() const override { return images_available; }
  bool supports_image_prefix_cache() const override { return images_available; }
  int32_t prefill_images(int req, const std::vector<int64_t>& prompt, const std::vector<dgpp::ImageInput>& images) override {
    require(!images.empty() && images[0].rgb.size() == 112 * 112 * 3, "decoded image reaches the engine");
    ++image_prefills;
    return prefill(req, prompt);
  }

  struct Live {
    size_t prompt_len = 0;
    int64_t held_blocks = 0;
    int served = 0;
    int32_t last_token = -1;
  };
  FakeEngine(int slots, int64_t total_blocks, int64_t block_tokens,
             bool can_sample = false)
      : slots_(slots), total_blocks_(total_blocks),
        block_tokens_(block_tokens), can_sample_(can_sample) {}

  // The sampling interface: a sampling-capable fake records the spec each slot
  // was armed with (the request interface's evidence); a greedy fake inherits
  // the base refusal.
  struct Armed {
    dgpp::sample::Params params;
    uint64_t seed = 0;
  };
  bool supports_sampling() const override { return can_sample_; }
  void configure_sampling(int req, const dgpp::sample::Params& p,
                          uint64_t seed) override {
    if (!can_sample_) {
      SchedulerEngine::configure_sampling(req, p, seed);
      return;
    }
    std::lock_guard<std::mutex> lock(armed_mu_);
    armed_.push_back(Armed{p, seed});
  }
  std::vector<Armed> armed() const {
    std::lock_guard<std::mutex> lock(armed_mu_);
    return armed_;
  }
  // Logprobs: the sampling-capable fake reports every token with logprob
  // -(index+1)/4 and top-N alternatives (the token itself first).
  bool supports_logprobs() const override { return can_sample_; }
  void configure_logprobs(int req, int logprobs) override {
    if (!can_sample_) {
      SchedulerEngine::configure_logprobs(req, logprobs);
      return;
    }
    report_[req] = logprobs;
  }
  std::vector<dgpp::sample::Result> take_logprobs(int req) override {
    std::vector<dgpp::sample::Result> out;
    out.swap(pending_lps_[req]);
    return out;
  }
  // Constrained decoding (M6 6g): the sampling-capable fake can mask and
  // records every active grammar it is armed with (the request interface's
  // evidence); it does not enforce it — the scripts are the outputs.
  bool supports_constraints() const override { return can_sample_; }
  void configure_constraint(int req,
                            const dgpp::text::GrammarSpec& g) override {
    if (!can_sample_) {
      SchedulerEngine::configure_constraint(req, g);
      return;
    }
    if (!g.active()) return;
    std::lock_guard<std::mutex> lock(armed_mu_);
    grammars_.push_back(g);
  }
  std::vector<dgpp::text::GrammarSpec> grammars() const {
    std::lock_guard<std::mutex> lock(armed_mu_);
    return grammars_;
  }
  // The logit bias: the sampling-capable fake can bias and
  // records every non-empty table it is armed with (slot, entries).
  bool supports_logit_bias() const override { return can_sample_; }
  void configure_logit_bias(int req,
                            const std::vector<dgpp::sched::LogitBias>& bias) override {
    if (!can_sample_) {
      SchedulerEngine::configure_logit_bias(req, bias);
      return;
    }
    if (bias.empty()) return;
    std::lock_guard<std::mutex> lock(armed_mu_);
    biases_.emplace_back(req, bias);
  }
  std::vector<std::pair<int, std::vector<dgpp::sched::LogitBias>>> biases() const {
    std::lock_guard<std::mutex> lock(armed_mu_);
    return biases_;
  }

  // The prefix cache interface (M7): an arena of `slots` snapshot slots at pool
  // alignment `align`; the fake records the ops ("X:slot:pos" an attach,
  // "N:slot:pos" a prefill-cut snapshot, "RS:slot:pos" a rolling one,
  // "F:arena" a release) and pins the blocks an entry holds.
  void set_prefix_arena(int slots, int64_t align) {
    arena_slots_ = slots;
    arena_align_ = align;
  }
  dgpp::sched::SchedulerEngine::PrefixInfo prefix_info() const override {
    dgpp::sched::SchedulerEngine::PrefixInfo info;
    info.arena_slots = arena_slots_;
    info.align = arena_align_;
    info.block_tokens = block_tokens_;
    return info;
  }
  int32_t prefill_cached(int req, const std::vector<int64_t>& prompt,
                         dgpp::sched::SchedulerEngine::PrefixPrefill* plan) override {
    if (plan->images && !plan->images->empty()) {
      require(images_available && plan->images->front().rgb.size() == 112 * 112 * 3,
              "image payload survives cached admission");
      ++image_prefills;
    }
    if (plan->attach_slot >= 0) {
      std::lock_guard<std::mutex> lock(armed_mu_);
      prefix_ops_.push_back("X:" + std::to_string(req) + ":" +
                            std::to_string(plan->attach_position));
    }
    const int32_t token = prefill(req, prompt);
    if (plan->snap_slot >= 0) {
      pinned_[plan->snap_slot] = plan->snap_position / block_tokens_ +
                                 (plan->snap_position % block_tokens_ != 0 ? 1 : 0);
      plan->snap_taken = true;
      std::lock_guard<std::mutex> lock(armed_mu_);
      prefix_ops_.push_back("N:" + std::to_string(req) + ":" +
                            std::to_string(plan->snap_position));
    }
    return token;
  }
  void prefix_snapshot(int req, int slot, int64_t position) override {
    pinned_[slot] = position / block_tokens_ + (position % block_tokens_ != 0 ? 1 : 0);
    std::lock_guard<std::mutex> lock(armed_mu_);
    prefix_ops_.push_back("RS:" + std::to_string(req) + ":" + std::to_string(position));
  }
  void prefix_release(int slot) override {
    pinned_.erase(slot);
    std::lock_guard<std::mutex> lock(armed_mu_);
    prefix_ops_.push_back("F:" + std::to_string(slot));
  }
  std::vector<std::string> prefix_ops() const {
    std::lock_guard<std::mutex> lock(armed_mu_);
    return prefix_ops_;
  }

  int max_concurrent_requests() const override { return slots_; }
  int64_t pool_blocks_total() const override { return total_blocks_; }
  int64_t pool_blocks_in_use() const override {
    int64_t sum = 0;
    for (const auto& [slot, live] : live_) sum += live.held_blocks;
    for (const auto& [slot, blocks] : pinned_) sum += blocks;
    return sum;
  }
  int64_t blocks_for_tokens(int64_t tokens) const override {
    return (tokens + block_tokens_ - 1) / block_tokens_;
  }

  // A scripted answer for prompts of exactly `prompt_len` ids (the 6f
  // gates: tool-call and reasoning id streams); EOS once it runs out.
  void script(size_t prompt_len, std::vector<int32_t> ids) {
    scripts_[prompt_len] = std::move(ids);
  }
  int32_t next_token(const Live& live, int index) const {
    const auto s = scripts_.find(live.prompt_len);
    if (s != scripts_.end())
      return index < static_cast<int>(s->second.size()) ? s->second[index]
                                                        : kFakeEos;
    if (index == 0)
      return fake_eos_prefill(live.prompt_len) ? kFakeEos
                                               : fake_token(live.prompt_len, 0);
    return fake_eos_second(live.prompt_len) && index == 1
               ? kFakeEos
               : fake_token(live.prompt_len, index);
  }

  int32_t prefill(int req, const std::vector<int64_t>& prompt) override {
    if (hold_prefill.load()) {
      prefill_monitor()->update(req, static_cast<int64_t>(prompt.size() / 2));
      prefill_entered = true;
      const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (hold_prefill.load() && std::chrono::steady_clock::now() < until)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (live_.count(req) != 0)
      throw std::runtime_error("fake: prefill on live slot");
    Live live;
    live.prompt_len = prompt.size();
    live.served = 1;
    live.last_token = next_token(live, 0);
    live_[req] = live;
    note_logprobs(req, live.last_token, 0);
    return live.last_token;
  }
  std::atomic<bool> hold_prefill{false}, prefill_entered{false};
  void note_logprobs(int req, int32_t token, int index) {
    if (report_.count(req) == 0 || report_[req] < 0) return;
    dgpp::sample::Result r;
    r.token = token;
    r.logprob = -static_cast<float>(index + 1) / 4.0f;
    for (int j = 0; j < report_[req]; ++j)
      r.top_logprobs.emplace_back(token + j, r.logprob - static_cast<float>(j));
    pending_lps_[req].push_back(r);
  }

  void reserve(int req, int64_t tokens) override {
    Live& live = live_.at(req);
    live.held_blocks = blocks_for_tokens(tokens);
  }

  std::vector<int32_t> step(int req) override {
    // The failure injection (the v1 failure semantics gate): the n-th step
    // call across every slot throws, as a bus watchdog or a journal write
    // to a dead peer would inside an engine op.
    if (fail_at_step_ > 0 && ++steps_seen_ == fail_at_step_)
      throw std::runtime_error("fake: injected engine failure at step " +
                               std::to_string(fail_at_step_));
    Live& live = live_.at(req);
    live.last_token = next_token(live, live.served);
    note_logprobs(req, live.last_token, live.served);
    ++live.served;
    return {live.last_token};
  }
  void fail_at_step(int n) { fail_at_step_ = n; }

  void close(int req) override { live_.erase(req); }

 private:
  std::map<size_t, std::vector<int32_t>> scripts_;
  int slots_;
  int64_t total_blocks_;
  int64_t block_tokens_;
  bool can_sample_ = false;
  int fail_at_step_ = 0, steps_seen_ = 0;
  int arena_slots_ = 0;
  int64_t arena_align_ = 1;
  std::map<int, int64_t> pinned_;
  std::vector<std::string> prefix_ops_;  // under armed_mu_
  std::map<int, Live> live_;
  std::map<int, int> report_;
  std::map<int, std::vector<dgpp::sample::Result>> pending_lps_;
  mutable std::mutex armed_mu_;
  std::vector<Armed> armed_;
  std::vector<dgpp::text::GrammarSpec> grammars_;
  std::vector<std::pair<int, std::vector<dgpp::sched::LogitBias>>> biases_;
};

// The 6f markers of the fake tokenizer: 1001..1008, decoding to their
// literal text (not special, exactly like the real added tokens).
constexpr int64_t kThinkOpen = 1001, kThinkClose = 1002, kToolOpen = 1003,
                  kToolClose = 1004, kKeyOpen = 1005, kKeyClose = 1006,
                  kValueOpen = 1007, kValueClose = 1008;
const std::vector<std::pair<std::string, int64_t>>& marker_table() {
  static const std::vector<std::pair<std::string, int64_t>> t = {
      {"</tool_call>", kToolClose}, {"<tool_call>", kToolOpen},
      {"</arg_value>", kValueClose}, {"<arg_value>", kValueOpen},
      {"</arg_key>", kKeyClose},   {"<arg_key>", kKeyOpen},
      {"</think>", kThinkClose},   {"<think>", kThinkOpen},
  };
  return t;
}

// A minijson value back to compact JSON (the tests read what the service
// handed the template).
std::string json_of(const dgpp::minijson::Value& v) {
  using K = dgpp::minijson::Value::Kind;
  switch (v.kind()) {
    case K::Null: return "null";
    case K::Bool: return v.as_bool() ? "true" : "false";
    case K::Int: return std::to_string(v.as_int());
    case K::Double: {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%g", v.as_double());
      return buf;
    }
    case K::String: return "\"" + std::string(v.as_string()) + "\"";
    case K::Array: {
      std::string out = "[";
      for (size_t i = 0; i < v.items().size(); ++i)
        out += (i ? "," : "") + json_of(v.items()[i]);
      return out + "]";
    }
    case K::Object: {
      std::string out = "{";
      for (size_t i = 0; i < v.members().size(); ++i)
        out += (i ? "," : "") + ("\"" + v.members()[i].key + "\":") +
               json_of(v.members()[i].value);
      return out + "}";
    }
  }
  return "?";
}

// The fake frontend: bytes ↔ ids (the marker strings map to their ids,
// leftmost-longest like the real added-token scan), and a deterministic
// chat render — the concatenation of every message's content (string, or
// the text of its parts), plus "<think>" when the fake models this
// template's generation prompt — the test computes prompt lengths from
// the same rule. It keeps the last globals the service handed it.
class FakeFrontend : public ModelFrontend {
 public:
  std::atomic<bool> images_available{false};
  bool supports_images() const override { return images_available; }
  ChatInput prepare_chat(const dgpp::minijson::Value& globals) const override {
    auto input = ModelFrontend::prepare_chat(globals);
    const auto& messages = globals.at("messages").items();
    for (size_t i = 0; i < messages.size(); ++i) {
      const auto* content = messages[i].find("content");
      if (!content || !content->is_array()) continue;
      for (size_t j = 0; j < content->items().size(); ++j) {
        const auto* image = content->items()[j].find("image_url");
        if (!image) continue;
        auto im = dgpp::serve::prepare_glm_image(*image,
            "messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "].image_url");
        im.offset = input.tokens.size();
        input.tokens.insert(input.tokens.end(), im.tokens, 154854);
        input.images.push_back(std::move(im));
      }
    }
    return input;
  }

  explicit FakeFrontend(bool with_markers = false)
      : with_markers_(with_markers) {}
  // The template knob gate: a template that reads enable_thinking (Qwen3.8-
  // Flash-Next, GLM-4.7) accepts it in chat_template_kwargs; the default
  // fake, like GLM-5.3-Flash's template, does not.
  std::atomic<bool> reads_enable_thinking{false};
  std::atomic<bool> reads_reasoning_effort{true};
  bool template_reads(std::string_view name) const override {
    return (name == "reasoning_effort" && reads_reasoning_effort) || (name == "enable_thinking" && reads_enable_thinking);
  }

  std::vector<int64_t> encode_text(std::string_view text) const override {
    std::vector<int64_t> ids;
    for (size_t i = 0; i < text.size();) {
      bool matched = false;
      for (const auto& [s, id] : marker_table()) {
        if (text.compare(i, s.size(), s) == 0) {
          ids.push_back(id);
          i += s.size();
          matched = true;
          break;
        }
      }
      if (!matched) ids.push_back(static_cast<unsigned char>(text[i++]));
    }
    return ids;
  }
  std::string decode_ids(const std::vector<int64_t>& ids) const override {
    std::string out;
    for (int64_t id : ids) {
      if (id == kFakeEos) continue;
      bool matched = false;
      for (const auto& [s, mid] : marker_table())
        if (mid == id) {
          out += s;
          matched = true;
          break;
        }
      if (!matched && id >= 0 && id < 256) out.push_back(static_cast<char>(id));
    }
    return out;
  }
  std::string render_chat(const dgpp::minijson::Value& globals) const override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      last_globals_ = json_of(globals);
    }
    std::string out;
    for (const auto& msg : globals.at("messages").items()) {
      const auto* content = msg.find("content");
      if (content == nullptr) continue;
      if (content->is_string()) {
        out.append(content->as_string());
      } else {
        for (const auto& part : content->items())
          if (const auto* text = part.find("text")) out.append(text->as_string());
      }
    }
    if (with_markers_) out.append("<think>");
    return out;
  }
  dgpp::text::ChatMarkers markers() const override {
    dgpp::text::ChatMarkers m;
    if (!with_markers_) return m;
    m.think_open = {kThinkOpen, "<think>"};
    m.think_close = {kThinkClose, "</think>"};
    m.tool_call_open = {kToolOpen, "<tool_call>"};
    m.tool_call_close = {kToolClose, "</tool_call>"};
    m.arg_key_open = {kKeyOpen, "<arg_key>"};
    m.arg_key_close = {kKeyClose, "</arg_key>"};
    m.arg_value_open = {kValueOpen, "<arg_value>"};
    m.arg_value_close = {kValueClose, "</arg_value>"};
    return m;
  }
  std::string last_globals() const {
    std::lock_guard<std::mutex> lock(mu_);
    return last_globals_;
  }
  // The prefix cache's boundary token (M7): the byte '|' plays the role
  // marker — a prompt "ab|cd|ef" has boundaries at 2 and 5.
  std::vector<int64_t> boundary_token_ids() const override {
    return {static_cast<int64_t>('|')};
  }

 private:
  bool with_markers_;
  mutable std::mutex mu_;
  mutable std::string last_globals_;
};

// --- the raw-socket client (as the http gate's) ------------------------
class Client {
 public:
  explicit Client(uint16_t port) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd_ >= 0, "client socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    require(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1,
            "client inet_pton");
    require(::connect(fd_, reinterpret_cast<sockaddr*>(&addr),
                      sizeof(addr)) == 0,
            "client connect");
    int yes = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
  }
  ~Client() {
    if (fd_ >= 0) ::close(fd_);
  }
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  void send_all(std::string_view s) {
    size_t off = 0;
    while (off < s.size()) {
      const ssize_t put = ::send(fd_, s.data() + off, s.size() - off, 0);
      require(put > 0, "client send");
      off += static_cast<size_t>(put);
    }
  }
  // Everything that arrives within the budget (EOF sets closed_).
  std::string read_available(int timeout_ms) {
    std::string out;
    int waited = 0;
    while (true) {
      pollfd p{fd_, POLLIN, 0};
      const int r = ::poll(&p, 1, 10);
      if (r < 0) break;
      if (r == 0) {
        waited += 10;
        if (waited >= timeout_ms) break;
        continue;
      }
      char buf[4096];
      const ssize_t got = ::recv(fd_, buf, sizeof(buf), 0);
      if (got > 0) {
        out.append(buf, static_cast<size_t>(got));
        waited = 0;
        continue;
      }
      if (got == 0) closed_ = true;
      break;
    }
    return out;
  }
  // Reads until `needle` shows up (or the budget dies) — for responses
  // that only complete after the engine runs.
  std::string read_until(const std::string& needle, int budget_ms) {
    std::string out;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(budget_ms);
    while (out.find(needle) == std::string::npos &&
           std::chrono::steady_clock::now() < deadline && !closed_) {
      pollfd p{fd_, POLLIN, 0};
      if (::poll(&p, 1, 25) > 0) {
        char buf[4096];
        const ssize_t got = ::recv(fd_, buf, sizeof(buf), 0);
        if (got > 0) out.append(buf, static_cast<size_t>(got));
        if (got == 0) closed_ = true;
      }
    }
    return out;
  }
  void hard_close() {
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    fd_ = -1;
  }

 private:
  int fd_ = -1;
  bool closed_ = false;
};

// --- the service rig: HTTP thread + gated engine thread ---------------
struct ServiceRig {
  static constexpr int kSlots = 4;
  FakeEngine engine;
  FakeFrontend frontend;
  ServiceConfig cfg;
  GenerationService service;
  HttpServer http;

  std::thread http_loop;
  std::thread engine_loop;
  std::atomic<bool> gate{false};   // pause the engine thread (tests)
  std::atomic<int> pass_delay_ms{0};  // slow the engine to a human pace (tests)
  std::atomic<bool> stopping{false};
  std::atomic<bool> failed{false};  // an engine op threw: the app's failure path ran

  // `sampling_defaults`: the served defaults (greedy unless a test hands
  // the checkpoint's stochastic ones); `can_sample`: whether the fake
  // engine advertises the sampler.
  // `with_markers`: the fake tokenizer carries the template's markers and
  // the fake render opens <think> (the 6f rigs); `reasoning_in_content`:
  // the fold knob.
  explicit ServiceRig(int queue_limit = 8,
                      dgpp::sample::Params sampling_defaults =
                          dgpp::sample::greedy_params(),
                      bool can_sample = false,
                      std::optional<uint64_t> fixed_seed = std::nullopt,
                      bool with_markers = false,
                      bool reasoning_in_content = false,
                      dgpp::sched::AdmissionPolicy admission = {},
                      int prefix_slots = 0,
                      dgpp::serve::FileInputConfig file_inputs = {},
                      // The request-context surface (review item 7): the rig's
                      // rope ramp and its two bounds, advertised on /v1/models.
                      std::optional<dgpp::RopeScaling> rope_scaling = std::nullopt,
                      int64_t position_ceiling = 0, int64_t kv_pool_tokens = 0,
                      // The served-model alias (the cluster config's
                      // engine.served_model_name): when set, /v1/models
                      // reports it and requests may name it (or the
                      // checkpoint id). Empty: the checkpoint id alone.
                      std::string served_model_name = "")
      : engine(kSlots, /*total_blocks=*/100, /*block_tokens=*/4, can_sample),
        frontend(with_markers),
        cfg([&] {
          ServiceConfig c;
          c.model_id = "glm-5.3-flash-fp8";
          c.served_model_name = served_model_name;
          c.default_max_tokens = 8;
          c.queue_limit = queue_limit;
          c.sampling_defaults = sampling_defaults;
          c.fixed_seed = fixed_seed;
          c.reasoning_in_content = reasoning_in_content;
          c.admission = admission;
          c.vocab_size = 512;  // the fake's ids are bytes and markers
          c.rope_scaling = rope_scaling;
          c.position_ceiling = position_ceiling;
          c.kv_pool_tokens = kv_pool_tokens;
          c.file_inputs = std::move(file_inputs);
          return c;
        }()),
        service((engine.set_prefix_arena(prefix_slots, 4), cfg), &engine, &frontend,
                {kFakeEos}),
        http(0, &service, /*max_connections=*/64) {
    http_loop = std::thread([this] { http.serve(); });
    engine_loop = std::thread([this] {
      while (!stopping.load()) {
        if (gate.load()) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }
        bool more = false;
        try {
          more = service.engine_pass();
        } catch (const std::exception& e) {
          // The app's failure path (the v1 failure semantics): fail the
          // service, leave the loop; the HTTP pump answers everyone.
          service.fail_engine(e.what());
          failed = true;
          stopping = true;
          break;
        }
        if (!more)
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        else if (pass_delay_ms.load() > 0)
          std::this_thread::sleep_for(std::chrono::milliseconds(pass_delay_ms.load()));
      }
    });
  }
  // The app's drain-on-stop (M6 6c), minus the journal: the engine thread
  // stops at a pass boundary, begin_shutdown() flags the live requests and
  // sheds the queue, one more pass retires them, and the HTTP pump answers
  // everything before the server stops.
  int drain(bool stop_http) {
    stopping = true;
    gate = false;
    if (engine_loop.joinable()) engine_loop.join();
    const int interrupted = service.begin_shutdown();
    service.engine_pass();
    for (int i = 0; i < 200 && !service.drained(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (stop_http) {
      http.stop();
      if (http_loop.joinable()) http_loop.join();
    }
    return interrupted;
  }
  ~ServiceRig() {
    stopping = true;
    http.stop();
    if (http_loop.joinable()) http_loop.join();
    if (engine_loop.joinable()) engine_loop.join();
  }
  ServiceRig(const ServiceRig&) = delete;
  ServiceRig& operator=(const ServiceRig&) = delete;

  uint16_t port() const { return http.port(); }
};

const std::string kModel = "glm-5.3-flash-fp8";

std::string chat_body(const std::string& content, int max_tokens,
                      const std::string& extra = "") {
  return "{\"model\":\"" + kModel +
         "\",\"messages\":[{\"role\":\"user\",\"content\":\"" + content +
         "\"}],\"max_tokens\":" + std::to_string(max_tokens) + extra + "}";
}

// The text the fake model generates for a prompt of `prompt_len` bytes
// over `n` tokens (the same arithmetic as FakeEngine, decoded).
std::string fake_text(size_t prompt_len, int n) {
  std::string out;
  for (int i = 0; i < n; ++i)
    out.push_back(static_cast<char>(fake_token(prompt_len, i)));
  return out;
}

DGPP_TEST(serve_chatNonStream_exactCompletionShape) {
  // GIVEN the service with a 4-byte prompt ("abcd" → 4 ids, no early
  // EOS) and max_tokens 3,
  ServiceRig rig;
  Client c(rig.port());

  // WHEN the chat completion completes,
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(chat_body("abcd", 3).size()) +
             "\r\n\r\n" + chat_body("abcd", 3));
  const std::string resp = c.read_until("usage", 5000);

  // THEN the body is the chat.completion object: the exact generated
  // text, finish_reason "length" (the steps cap), and usage counting
  // prompt 4 / completion 3 / total 7.
  require(resp.find("200 OK") != std::string::npos, "status: " + resp);
  require(resp.find("\"object\":\"chat.completion\"") != std::string::npos,
          "object kind");
  require(resp.find("\"role\":\"assistant\",\"content\":\"" +
                    fake_text(4, 3) + "\"") != std::string::npos,
          "content text: " + resp);
  require(resp.find("\"finish_reason\":\"length\"") != std::string::npos,
          "steps cap → length");
  require(resp.find("\"prompt_tokens\":4,\"completion_tokens\":3,"
                    "\"total_tokens\":7") != std::string::npos,
          "usage arithmetic: " + resp);
}

DGPP_TEST(serve_chatOneTokenLimit_returnsExactlyOneToken) {
  // The prefill pick is completion token one. This boundary used to fall
  // through to the same tick's decode because only step() checked the cap.
  ServiceRig rig;
  Client c(rig.port());
  const std::string body = chat_body("abcd", 1);
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  const std::string resp = c.read_until("usage", 5000);

  require(resp.find("\"role\":\"assistant\",\"content\":\"" +
                    fake_text(4, 1) + "\"") != std::string::npos,
          "max_tokens=1 content: " + resp);
  require(resp.find("\"prompt_tokens\":4,\"completion_tokens\":1,"
                    "\"total_tokens\":5") != std::string::npos,
          "max_tokens=1 usage: " + resp);
  require(resp.find("\"finish_reason\":\"length\"") != std::string::npos,
          "one-token cap finishes by length");
}

DGPP_TEST(serve_chatStream_chunkLifecycleInOrder) {
  // GIVEN a streaming request with include_usage (prompt "abcde" — 5
  // bytes triggers neither EOS rule, so the run goes to the cap),
  ServiceRig rig;
  Client c(rig.port());
  const std::string body =
      chat_body("abcde", 3, ",\"stream\":true,"
                             "\"stream_options\":{\"include_usage\":true}");

  // WHEN the stream completes,
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  const std::string resp = c.read_until("[DONE]", 5000);

  // THEN the lifecycle is exactly OpenAI's: headers, role chunk, the
  // generated text arriving as in-order content deltas (tokens may
  // coalesce into a chunk — the client contract is their
  // concatenation), the final chunk (length), the usage chunk, [DONE].
  require(resp.find("text/event-stream") != std::string::npos,
          "SSE content type");
  const size_t role = resp.find("\"delta\":{\"role\":\"assistant\","
                                "\"content\":\"\"}");
  const size_t final =
      resp.find("\"delta\":{},\"logprobs\":null,\"finish_reason\":\"length\"");
  const size_t usage =
      resp.find("\"choices\":[],\"usage\":{\"prompt_tokens\":5,"
                "\"completion_tokens\":3,\"total_tokens\":8,"
                "\"prompt_tokens_details\":{\"cached_tokens\":0},"
                "\"completion_tokens_details\":{\"reasoning_tokens\":0}}");
  const size_t done = resp.find("data: [DONE]");
  require(role != std::string::npos, "role chunk present");
  // Concatenate every content payload in arrival order.
  std::string concatenated;
  size_t pos = 0;
  int content_chunks = 0;
  while ((pos = resp.find("\"content\":\"", pos)) != std::string::npos) {
    const size_t vstart = pos + std::string("\"content\":\"").size();
    const size_t vend = resp.find("\"", vstart);
    require(vend != std::string::npos, "content field terminated");
    // The role chunk's empty content carries no text.
    if (vend > vstart) {
      concatenated.append(resp, vstart, vend - vstart);
      ++content_chunks;
    }
    pos = vend;
  }
  require(content_chunks >= 1, "content deltas present");
  require(concatenated == fake_text(5, 3),
          "concatenated deltas == the generated text: got \"" +
              concatenated + "\" expected \"" + fake_text(5, 3) + "\"");
  require(final != std::string::npos && final > role, "final chunk present");
  require(usage != std::string::npos && usage > final,
          "usage chunk after the final chunk");
  require(done != std::string::npos && done > usage, "[DONE] last");
}

DGPP_TEST(serve_eosMidAnswer_finishReasonStop) {
  // GIVEN a prompt of 3 bytes (len % 4 == 3 → the fake answers EOS as
  // its second token),
  ServiceRig rig;
  Client c(rig.port());

  // WHEN the completion runs to its natural stop,
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(chat_body("abc", 8).size()) +
             "\r\n\r\n" + chat_body("abc", 8));
  const std::string resp = c.read_until("usage", 5000);

  // THEN finish_reason is "stop" (EOS beat the cap), completion tokens
  // count the EOS pick (2), and the EOS id itself contributes no text.
  require(resp.find("\"finish_reason\":\"stop\"") != std::string::npos,
          "EOS → stop: " + resp);
  require(resp.find("\"prompt_tokens\":3,\"completion_tokens\":2,"
                    "\"total_tokens\":5") != std::string::npos,
          "EOS counted as a completion token");
  require(resp.find("\"content\":\"" + fake_text(3, 1) + "\"") !=
              std::string::npos,
          "one text token, then EOS (decoded to nothing)");
}

DGPP_TEST(serve_refusalLadder_openAIErrorObjects) {
  // GIVEN the service,
  ServiceRig rig;

  // THEN every unimplemented knob refuses with the OpenAI error object
  // naming the param — never a silent ignore, never a bare string.
  const auto post_and_expect = [&](const std::string& body, int status,
                                   const std::string& needle) {
    Client c(rig.port());
    c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    const std::string resp = c.read_available(800);
    require(resp.find(std::to_string(status) + " ") != std::string::npos,
            "status " + std::to_string(status) + ": " + resp.substr(0, 120));
    require(resp.find("\"error\":{") != std::string::npos,
            "error object shape: " + resp.substr(0, 200));
    require(resp.find(needle) != std::string::npos,
            "needle " + needle + ": " + resp.substr(0, 400));
  };
  // A greedy-only engine refuses stochastic requests by capability; an
  // out-of-range top_p is refused by validation regardless of engine.
  post_and_expect(chat_body("abcd", 3, ",\"temperature\":0.7"), 400,
                  "\"code\":\"sampling_unsupported\"");
  post_and_expect(chat_body("abcd", 3, ",\"top_p\":5"), 400,
                  "\"param\":\"top_p\"");
  post_and_expect(chat_body("abcd", 3, ",\"logit_bias\":{\"1\":2}"), 400,
                  "\"param\":\"logit_bias\"");
  post_and_expect(chat_body("abcd", 3,
                            ",\"tools\":[{\"type\":\"function\"}]"),
                  400, "\"param\":\"tools[0].function.name\"");
  post_and_expect(chat_body("abcd", 3, ",\"user\":123"), 400,
                  "\"param\":\"user\"");
  post_and_expect(
      "{\"model\":\"wrong-model\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"hi\"}]}",
      404, "\"code\":\"model_not_found\"");
  post_and_expect("{not json", 400, "invalid JSON body");
  // Content parts render through the template (6f); a part without a
  // type, a role the template does not know, and tools on a frontend
  // without the markers all refuse by name.
  post_and_expect(
      "{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"user\","
      "\"content\":[{\"text\":\"hi\"}]}]}",
      400, "\"param\":\"messages[0].content[0]\"");
  post_and_expect(
      "{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"unknown\","
      "\"content\":\"hi\"}]}",
      400, "\"param\":\"messages[0].role\"");
  post_and_expect(chat_body("abcd", 3,
                            ",\"tools\":[{\"type\":\"function\","
                            "\"function\":{\"name\":\"f\"}}]"),
                  400, "\"code\":\"tools_unsupported\"");
}

DGPP_TEST(serve_overloadedQueue_503AtTheDoor) {
  // GIVEN a one-deep admission queue and a PAUSED engine (the pending
  // admission cannot drain),
  ServiceRig rig(/*queue_limit=*/1);
  rig.gate = true;

  // WHEN two requests arrive,
  Client first(rig.port());
  first.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                 "Content-Type: application/json\r\nContent-Length: " +
                 std::to_string(chat_body("abcd", 2).size()) +
                 "\r\n\r\n" + chat_body("abcd", 2));
  // (Non-streaming requests receive NOTHING until they complete — no
  // early bytes to await; the first one just sits pending.)
  Client second(rig.port());
  second.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                  "Content-Type: application/json\r\nContent-Length: " +
                  std::to_string(chat_body("efgh", 2).size()) +
                  "\r\n\r\n" + chat_body("efgh", 2));

  // THEN the second is shed at the door with 503 + the error object
  // (the overload contract), while the first still completes once the
  // engine resumes.
  const std::string shed = second.read_available(800);
  require(shed.find("503") != std::string::npos, "503 at the door: " +
                                                      shed.substr(0, 120));
  require(shed.find("\"code\":\"overloaded\"") != std::string::npos,
          "overload error object");
  Client third(rig.port());
  const auto streaming = chat_body("efgh", 2, ",\"stream\":true");
  third.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: " +
                  std::to_string(streaming.size()) + "\r\n\r\n" + streaming);
  const auto stream_shed = third.read_until("overloaded", 1000);
  require(stream_shed.starts_with("HTTP/1.1 503") && stream_shed.find("text/event-stream") == std::string::npos,
          "streaming overload is rejected before sending successful stream headers");
  rig.gate = false;
  const std::string done = first.read_until("usage", 5000);
  require(done.find("\"finish_reason\":\"length\"") != std::string::npos,
          "the first request completes after the pause: " + done.substr(0, 200));
}

DGPP_TEST(serve_clientDisconnectMidStream_mapsToSchedulerCancel) {
  // GIVEN a paused engine and a streaming request whose connection the
  // client will kill,
  ServiceRig rig;
  rig.gate = true;
  Client c(rig.port());
  const std::string body = chat_body("abcdefgh", 6, ",\"stream\":true");
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  require(c.read_available(500).find("text/event-stream") !=
              std::string::npos,
          "stream headers arrived before the disconnect");

  // WHEN the client vanishes and the engine resumes,
  c.hard_close();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  rig.gate = false;

  // THEN the disconnect became a scheduler cancellation: metrics
  // counts it, and the record settles without any writer touch (the
  // engine runs to a cancelled retire within a couple passes).
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  Client m(rig.port());
  m.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string metrics = m.read_until("requests_cancelled", 2000);
  require(metrics.find("\"requests_cancelled\":1") != std::string::npos,
          "the disconnect counted as one cancellation: " +
              metrics.substr(metrics.find("service")));
}

DGPP_TEST(serve_modelsHealthMetrics_theOpsSurface) {
  // GIVEN the service,
  ServiceRig rig;

  // WHEN the operational endpoints are read,
  Client models(rig.port());
  models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string ml = models.read_available(800);
  Client health(rig.port());
  health.send_all("GET /health HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string hl = health.read_available(800);
  Client metrics(rig.port());
  metrics.send_all("GET /metrics HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string met = metrics.read_available(800);
  Client legacy_metrics(rig.port());
  legacy_metrics.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string legacy_met = legacy_metrics.read_available(800);

  // THEN each answers its documented shape.
  require(ml.find("\"object\":\"list\"") != std::string::npos &&
              ml.find("\"id\":\"" + kModel + "\"") != ml.npos &&
              ml.find("\"object\":\"model\"") != std::string::npos,
          "models list: " + ml.substr(0, 200));
  require(hl.find("\"status\":\"ok\"") != std::string::npos, "health");
  require(met.find("HTTP/1.1 200 OK\r\n") == 0 &&
              met.find("Content-Type: application/json\r\n") != std::string::npos,
          "metrics returns JSON directly: " + met.substr(0, 200));
  require(met.find("\"scheduler\":{") != std::string::npos &&
              met.find("\"service\":{") != std::string::npos,
          "metrics sections: " + met.substr(0, 200));
  const auto first = dgpp::minijson::parse(std::string_view(met).substr(met.find("\r\n\r\n") + 4));
  const auto second = dgpp::minijson::parse(std::string_view(legacy_met).substr(legacy_met.find("\r\n\r\n") + 4));
  for (const char* key : {"service", "prefix_cache", "prefill"})
    require(json_of(first.root.at(key)) == json_of(second.root.at(key)), "metrics aliases agree");
  for (const auto& m : first.root.at("scheduler").members())
    if (m.key != "snapshot_age_ms")
      require(json_of(m.value) == json_of(second.root.at("scheduler").at(m.key)), "scheduler gauges agree");
}

DGPP_TEST(serve_legacyCompletions_theTextCompletionObject) {
  // GIVEN the legacy prompt API,
  ServiceRig rig;
  Client c(rig.port());
  const std::string body =
      "{\"model\":\"" + kModel + "\",\"prompt\":\"hello\","
      "\"max_tokens\":2}";

  // WHEN it completes,
  c.send_all("POST /v1/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  const std::string resp = c.read_until("usage", 5000);

  // THEN the body is the text_completion object with the same token
  // arithmetic.
  require(resp.find("\"object\":\"text_completion\"") != std::string::npos,
          "legacy object: " + resp.substr(0, 200));
  require(resp.find("\"text\":\"" + fake_text(5, 2) + "\"") !=
              std::string::npos,
          "legacy text");
  require(resp.find("\"prompt_tokens\":5,\"completion_tokens\":2,"
                    "\"total_tokens\":7") != std::string::npos,
          "legacy usage");
}

// The checkpoint's defaults for the sampling rigs (generation_config.json:
// temperature 1.0, top_p 0.95, nothing else).
dgpp::sample::Params model_defaults() {
  dgpp::sample::Params p;
  p.temperature = 1.0f;
  p.top_p = 0.95f;
  return p;
}

std::string post_chat(ServiceRig& rig, const std::string& body,
                      const std::string& until = "", int budget_ms = 3000) {
  Client c(rig.port());
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  return until.empty() ? c.read_available(800) : c.read_until(until, budget_ms);
}

void expect_invalid_request(const std::string& resp, const std::string& param) {
  require(resp.find("400 Bad Request\r\n") != std::string::npos,
          "expected 400 for " + param + ": " + resp);
  const size_t body = resp.find("\r\n\r\n");
  require(body != std::string::npos, "complete response headers");
  const auto parsed = dgpp::minijson::parse(std::string_view(resp).substr(body + 4));
  const auto& error = parsed.root.at("error");
  require(error.at("type").as_string() == "invalid_request_error",
          "invalid request error type: " + resp);
  require(error.at("param").as_string() == param, "error names " + param + ": " + resp);
}

DGPP_TEST(serve_api_nullableFieldsAndTextCapabilities) {
  ServiceRig rig;
  const auto response = post_chat(rig, chat_body("abcd", 2,
      R"(,"max_completion_tokens":null,"stream":null,"stream_options":null,"temperature":null,"top_p":null,"seed":null,"n":null,"logprobs":null,"top_logprobs":null,"reasoning_effort":null,"store":false,"modalities":["text"],"tool_choice":"none","parallel_tool_calls":false,"service_tier":"auto","metadata":{"job":"audit"},"user":"client","safety_identifier":"test")"), "\"metadata\"");
  require(response.find("200 OK") != std::string::npos, "nullable/default text fields accepted: " + response);
  const auto parsed = dgpp::minijson::parse(std::string_view(response).substr(response.find("\r\n\r\n") + 4));
  require(parsed.root.at("service_tier").as_string() == "default", "actual service tier reported");
  require(parsed.root.at("metadata").at("job").as_string() == "audit", "metadata echoed");
}

DGPP_TEST(serve_api_integerBoundsAndUnsupportedParameters) {
  ServiceRig rig;
  for (const auto& [extra, param] : {
      std::pair{R"(,"max_completion_tokens":1.5)", "max_completion_tokens"},
      std::pair{R"(,"max_completion_tokens":2147483648)", "max_completion_tokens"},
      std::pair{R"(,"max_completion_tokens":1e100)", "max_completion_tokens"},
      std::pair{R"(,"seed":9223372036854775808)", "seed"},
      std::pair{R"(,"seed":-9223372036854775809)", "seed"},
      std::pair{R"(,"n":1e100)", "n"},
      std::pair{R"(,"store":true)", "store"},
      std::pair{R"(,"modalities":["audio"])", "modalities"},
      std::pair{R"(,"verbosity":"high")", "verbosity"},
      std::pair{R"(,"prompt_cache_options":{"ttl":"30m"})", "prompt_cache_options"},
      std::pair{R"(,"web_search_options":{})", "web_search_options"},
      std::pair{R"(,"stream_options":{"include_usage":true})", "stream_options"}}) {
    const auto body = "{\"model\":\"" + kModel + R"(","messages":[{"role":"user","content":"abcd"}])" + extra + "}";
    expect_invalid_request(post_chat(rig, body, "\"param\""), param);
  }
  for (const char* seed : {"9223372036854775807", "-9223372036854775808"}) {
    const auto response = post_chat(rig, chat_body("abcd", 2, std::string(",\"seed\":") + seed), "usage");
    require(response.find("200 OK") != std::string::npos, "int64 boundary seed accepted");
  }
}

DGPP_TEST(serve_api_opencodeProviderExtensionsRemainCompatible) {
  ServiceRig rig(8, model_defaults(), true);
  const std::string settings = R"(,"seed":42,"temperature":0.7,"top_p":0.95,"min_p":0.05,"presence_penalty":0.2,"repetition_penalty":1.1,"chat_template_kwargs":{"clear_thinking":false})";
  require(post_chat(rig, chat_body("abcd", 2, settings), "usage").find("200 OK") != std::string::npos,
          "baseline request accepted");
  const auto globals = rig.frontend.last_globals();
  const auto baseline = rig.engine.armed().back();
  for (const std::string extension : {
       R"(,"preserveThinking":true)", R"(,"preserveThinking":false)",
       R"(,"preserveThinking":true,"client_extension":{"temperature":100,"store":true})"}) {
    for (bool stream : {false, true}) {
      const auto response = post_chat(rig, chat_body("abcd", 2, settings + extension +
          (stream ? R"(,"stream":true,"stream_options":{"include_usage":true})" : "")),
          stream ? "\r\n0\r\n\r\n" : "usage");
      require(response.starts_with("HTTP/1.1 200 OK") &&
                  response.find("\"completion_tokens\":2") != std::string::npos,
              "OpenCode provider extensions work on both response paths: " + response);
      require(rig.frontend.last_globals() == globals, "unknown options never alter template globals");
      const auto armed = rig.engine.armed().back();
      require(armed.seed == baseline.seed && armed.params.temperature == baseline.params.temperature &&
                  armed.params.top_p == baseline.params.top_p && armed.params.min_p == baseline.params.min_p &&
                  armed.params.presence_penalty == baseline.params.presence_penalty &&
                  armed.params.repetition_penalty == baseline.params.repetition_penalty,
              "provider extensions do not override validated sampling settings");
    }
  }
  for (const auto& [extra, param] : {
       std::pair{R"(,"temperature":100)", "temperature"},
       std::pair{R"(,"store":true)", "store"},
       std::pair{R"(,"web_search_options":{})", "web_search_options"},
       std::pair{R"(,"audio":{"voice":"alloy","format":"wav"})", "audio"}}) {
    expect_invalid_request(post_chat(rig, chat_body("abcd", 2,
        std::string(R"(,"preserveThinking":true)") + extra), "\"param\""), param);
  }
}

DGPP_TEST(serve_api_developerAndUnsupportedMedia) {
  ServiceRig rig;
  const auto response = post_chat(rig, "{\"model\":\"" + kModel +
      R"(","max_tokens":2,"messages":[{"role":"developer","content":[{"type":"text","text":"abcd"}]}]})", "usage");
  require(response.find("200 OK") != std::string::npos, "developer message accepted");
  require(rig.frontend.last_globals().find("\"role\":\"system\"") != std::string::npos,
          "developer instructions reach the checkpoint's system role");
  expect_invalid_request(post_chat(rig, "{\"model\":\"" + kModel +
      R"(","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.invalid/image"}}]}]})", "\"param\""),
      "messages[0].content[0].type");
}

DGPP_TEST(serve_api_streamUsageAndObfuscation) {
  ServiceRig rig;
  for (const bool obfuscation : {false, true}) {
    const auto response = post_chat(rig, chat_body("abcd", 3,
        std::string(R"(,"stream":true,"stream_options":{"include_usage":true,"include_obfuscation":)") +
        (obfuscation ? "true}" : "false}")), "\r\n0\r\n\r\n", 5000);
    size_t at = 0;
    int usage_chunks = 0, delta_chunks = 0;
    while ((at = response.find("data: ", at)) != std::string::npos) {
      at += 6;
      const auto end = response.find("\n\n", at);
      const auto payload = response.substr(at, end - at);
      if (payload == "[DONE]") break;
      const auto parsed = dgpp::minijson::parse(payload);
      const auto& chunk = parsed.root;
      require(chunk.at("object").as_string() == "chat.completion.chunk", "stream object");
      if (chunk.at("choices").items().empty()) {
        ++usage_chunks;
        require(chunk.at("usage").at("completion_tokens").as_int() == 3, "final usage");
      } else {
        ++delta_chunks;
        require(chunk.at("usage").is_null(), "non-final usage is null");
        require((chunk.find("obfuscation") != nullptr) == obfuscation, "obfuscation opt-out honored");
        if (obfuscation) require(payload.size() % 128 == 0, "stream payload size normalized");
      }
      at = end;
    }
    require(usage_chunks == 1 && delta_chunks >= 2, "one final usage chunk");
  }
}

DGPP_TEST(serve_metrics_liveDuringBlockedPrefill) {
  ServiceRig rig;
  rig.engine.hold_prefill = true;
  Client request(rig.port());
  const auto body = chat_body("abcdefgh", 2);
  request.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: " +
                   std::to_string(body.size()) + "\r\n\r\n" + body);
  for (int i = 0; i < 500 && !rig.engine.prefill_entered.load(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  require(rig.engine.prefill_entered.load(), "engine entered synchronous prefill");
  for (const char* path : {"/metrics", "/v1/metrics"}) {
    Client metrics(rig.port());
    metrics.send_all(std::string("GET ") + path + " HTTP/1.1\r\nHost: t\r\n\r\n");
    const auto response = metrics.read_until("\"remaining_tokens\":4}}", 1000);
    const auto parsed = dgpp::minijson::parse(std::string_view(response).substr(response.find("\r\n\r\n") + 4));
    require(parsed.root.at("scheduler").at("prefilling").as_int() == 1, "live prefill gauge");
    require(parsed.root.at("scheduler").at("active").as_int() == 1, "active during prefill");
    const auto& progress = parsed.root.at("prefill");
    require(progress.at("prompt_tokens").as_int() == 8 && progress.at("processed_tokens").as_int() == 4,
            "chunk progress visible before scheduler pass returns");
    require(progress.at("requests").items().size() == 1, "request detail");
  }
  rig.engine.hold_prefill = false;
  require(request.read_until("usage", 3000).find("200 OK") != std::string::npos, "request finishes after release");
  require(rig.engine.prefill_monitor()->snapshot().empty(), "completed prefill removed");
}

DGPP_TEST(serve_chatStream_usageRequiresOptIn) {
  ServiceRig rig;
  for (const std::string options :
       {"", ",\"stream_options\":{}", ",\"stream_options\":{\"include_usage\":false}"}) {
    // Read through the HTTP chunk terminator, not just an early content delta.
    const std::string resp =
        post_chat(rig, chat_body("abcd", 3, ",\"stream\":true" + options), "\r\n0\r\n\r\n", 5000);
    require(resp.find("200 OK\r\n") != std::string::npos && resp.ends_with("\r\n0\r\n\r\n") &&
                resp.find("data: [DONE]\n\n") != std::string::npos,
            "completed stream for options " + options + ": " + resp);
    require(resp.find("\"usage\":{") == std::string::npos,
            "no usage object without opt-in: " + resp);
    require(resp.find("\"choices\":[]") == std::string::npos,
            "no usage-only chunk without opt-in: " + resp);
  }
}

DGPP_TEST(serve_streamOptions_invalidTypesNameTheField) {
  ServiceRig rig;
  for (const auto& [options, param] :
       {std::pair{"false", "stream_options"},
        std::pair{"{\"include_usage\":null}", "stream_options.include_usage"}}) {
    expect_invalid_request(
        post_chat(rig, chat_body("abcd", 3,
                                 ",\"stream\":true,\"stream_options\":" + std::string(options))),
        param);
  }
}

bool same_float(float a, float b) {
  return std::memcmp(&a, &b, sizeof(float)) == 0;
}

DGPP_TEST(serve_sampling_defaultsFillOmittedFieldsAndReachTheEngine) {
  // GIVEN a sampling-capable engine and the checkpoint's defaults,
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true);

  // THEN /v1/models advertises the surface and the exact defaults,
  {
    Client models(rig.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string ml = models.read_available(800);
    require(ml.find("\"sampling\":{\"available\":true,\"defaults\":{"
                    "\"temperature\":1,\"top_p\":0.95,\"top_k\":0,"
                    "\"min_p\":0,\"repetition_penalty\":1}}") !=
                std::string::npos,
            "models sampling surface: " + ml.substr(0, 400));
  }

  // WHEN a request omits every sampling field, one names a seed and
  // overrides, and one asks for greedy explicitly,
  const std::string r1 = post_chat(rig, chat_body("abcd", 2), "usage");
  const std::string r2 = post_chat(
      rig,
      chat_body("abcd", 2,
                ",\"seed\":42,\"temperature\":0.7,\"top_p\":0.5,"
                "\"presence_penalty\":1.5,\"frequency_penalty\":-0.25,"
                "\"top_k\":40,\"min_p\":0.05,\"repetition_penalty\":1.1"),
      "usage");
  const std::string r3 =
      post_chat(rig, chat_body("abcd", 2, ",\"temperature\":0"), "usage");
  require(r1.find("\"object\":\"chat.completion\"") != std::string::npos &&
              r2.find("\"object\":\"chat.completion\"") != std::string::npos &&
              r3.find("\"object\":\"chat.completion\"") != std::string::npos,
          "all three requests complete");

  // THEN the engine was armed, in order, with the defaults + a drawn seed,
  // the explicit spec + seed 42, and the greedy spec.
  const std::vector<FakeEngine::Armed> armed = rig.engine.armed();
  require(armed.size() == 3, "three slots armed, got " +
                                 std::to_string(armed.size()));
  require(same_float(armed[0].params.temperature, 1.0f) &&
              same_float(armed[0].params.top_p, 0.95f) &&
              armed[0].params.top_k == 0,
          "omitted fields take the model defaults");
  require(same_float(armed[1].params.temperature, 0.7f) &&
              same_float(armed[1].params.top_p, 0.5f) &&
              same_float(armed[1].params.presence_penalty, 1.5f) &&
              same_float(armed[1].params.frequency_penalty, -0.25f) &&
              armed[1].params.top_k == 40 &&
              same_float(armed[1].params.min_p, 0.05f) &&
              same_float(armed[1].params.repetition_penalty, 1.1f) &&
              armed[1].seed == 42,
          "explicit fields override the defaults exactly");
  require(same_float(armed[2].params.temperature, 0.0f),
          "temperature 0 is the greedy spec");
  require(armed[0].seed != armed[2].seed,
          "seedless requests draw distinct seeds");
}

DGPP_TEST(serve_sampling_validationNamesTheField) {
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true);
  const auto refused = [&](const std::string& extra, const std::string& param) {
    const std::string resp = post_chat(rig, chat_body("abcd", 2, extra));
    require(resp.find("400 ") != std::string::npos &&
                resp.find("\"param\":\"" + param + "\"") != std::string::npos,
            "expected a 400 naming " + param + ": " + resp.substr(0, 300));
  };
  refused(",\"temperature\":2.5", "temperature");
  refused(",\"temperature\":\"hot\"", "temperature");
  require(post_chat(rig, chat_body("abcd", 2, ",\"top_p\":0"), "usage").find("200 OK") != std::string::npos,
          "top_p zero is valid and selects the highest-probability token");
  refused(",\"top_p\":-0.1", "top_p");
  refused(",\"top_p\":1.01", "top_p");
  refused(",\"presence_penalty\":2.5", "presence_penalty");
  refused(",\"frequency_penalty\":-3", "frequency_penalty");
  refused(",\"top_k\":1.5", "top_k");
  refused(",\"top_k\":-1", "top_k");
  refused(",\"min_p\":2", "min_p");
  refused(",\"repetition_penalty\":0", "repetition_penalty");
  refused(",\"seed\":1.5", "seed");
  const auto armed = rig.engine.armed();
  require(armed.size() == 1 && armed[0].params.top_k == 1 && armed[0].params.top_p == 1.0f,
          "only the valid top_p zero request reached the engine, as a singleton nucleus");
}

DGPP_TEST(serve_sampling_greedyEngineCollapsesDefaultsLoudly) {
  // GIVEN the checkpoint's stochastic defaults but an engine that cannot
  // sample (today's graph engine),
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/false);

  // THEN the served defaults are greedy and say so,
  Client models(rig.port());
  models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string ml = models.read_available(800);
  require(ml.find("\"sampling\":{\"available\":false,\"defaults\":{"
                  "\"temperature\":0,") != std::string::npos,
          "collapsed defaults: " + ml.substr(0, 400));
  // a field-less request is served (greedy),
  const std::string ok = post_chat(rig, chat_body("abcd", 2), "usage");
  require(ok.find("\"object\":\"chat.completion\"") != std::string::npos,
          "the greedy default is served");
  // and an explicit stochastic request is refused by capability.
  const std::string refused =
      post_chat(rig, chat_body("abcd", 2, ",\"temperature\":1"));
  require(refused.find("\"code\":\"sampling_unsupported\"") !=
              std::string::npos,
          "stochastic request refused: " + refused.substr(0, 300));
}

DGPP_TEST(serve_sampling_fixedSeedAppliesToSeedlessRequests) {
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true,
                 /*fixed_seed=*/uint64_t{777});
  (void)post_chat(rig, chat_body("abcd", 2), "usage");
  (void)post_chat(rig, chat_body("abcd", 2, ",\"seed\":5"), "usage");
  const std::vector<FakeEngine::Armed> armed = rig.engine.armed();
  require(armed.size() == 2 && armed[0].seed == 777 && armed[1].seed == 5,
          "the fixed seed fills seedless requests; explicit seeds win");
}

DGPP_TEST(serve_logprobs_openAIShapesOnEveryRoute) {
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true);
  // Non-stream chat: the content array with one entry per token, each with
  // the token text, its bytes and top_logprobs.
  const std::string one =
      post_chat(rig, chat_body("abcd", 3, ",\"logprobs\":true,\"top_logprobs\":2"),
                "usage");
  require(one.find("\"logprobs\":{\"content\":[{\"token\":") != std::string::npos,
          "chat logprobs content: " + one.substr(0, 400));
  require(one.find("\"logprob\":-0.25,\"bytes\":[") != std::string::npos,
          "the first token's logprob and bytes: " + one.substr(0, 600));
  require(one.find("\"top_logprobs\":[{\"token\":") != std::string::npos,
          "top_logprobs entries");
  {
    size_t count = 0, pos = 0;
    while ((pos = one.find("\"bytes\":[", pos)) != std::string::npos) {
      ++count;
      pos += 8;
    }
    // 3 tokens, each with 2 alternatives: 9 bytes arrays.
    require(count == 9, "three entries with two alternatives each, got " +
                            std::to_string(count));
  }
  // Streaming chat: content chunks carry the entries since the last one.
  {
    Client c(rig.port());
    const std::string body =
        chat_body("abcd", 3, ",\"stream\":true,\"logprobs\":true");
    c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    const std::string resp = c.read_until("[DONE]", 5000);
    require(resp.find("\"logprobs\":{\"content\":[{\"token\":") != std::string::npos,
            "streamed logprobs: " + resp.substr(0, 400));
    require(resp.find("\"top_logprobs\":[]") != std::string::npos,
            "logprobs without top_logprobs report empty alternatives");
  }
  // Legacy completions: the tokens/token_logprobs/top_logprobs/text_offset
  // object.
  {
    Client c(rig.port());
    const std::string body =
        "{\"model\":\"" + kModel + "\",\"prompt\":\"hello\",\"max_tokens\":2,"
        "\"logprobs\":1}";
    c.send_all("POST /v1/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    const std::string resp = c.read_until("usage", 5000);
    require(resp.find("\"logprobs\":{\"tokens\":[") != std::string::npos &&
                resp.find("\"token_logprobs\":[-0.25,-0.5]") != std::string::npos &&
                resp.find("\"text_offset\":[0,1]") != std::string::npos,
            "legacy logprobs object: " + resp.substr(0, 500));
  }
  // Refusals name the field.
  const std::string no_flag =
      post_chat(rig, chat_body("abcd", 2, ",\"top_logprobs\":2"));
  require(no_flag.find("\"param\":\"top_logprobs\"") != std::string::npos,
          "top_logprobs without logprobs");
  const std::string too_many =
      post_chat(rig, chat_body("abcd", 2, ",\"logprobs\":true,\"top_logprobs\":25"));
  require(too_many.find("\"param\":\"top_logprobs\"") != std::string::npos,
          "top_logprobs above 20");
  ServiceRig greedy_rig(/*queue_limit=*/8);
  const std::string unsupported =
      post_chat(greedy_rig, chat_body("abcd", 2, ",\"logprobs\":true"));
  require(unsupported.find("\"code\":\"logprobs_unsupported\"") != std::string::npos,
          "an engine without logprobs refuses");
}


// ---- M6 6f: tools and reasoning ------------------------------------------

std::string post_until_usage(ServiceRig& rig, const std::string& body) {
  return post_chat(rig, body, "usage", 5000);
}

std::vector<int32_t> script_of(const ServiceRig& rig, const std::string& text,
                               bool eos = true) {
  std::vector<int32_t> out;
  for (const int64_t id : rig.frontend.encode_text(text))
    out.push_back(static_cast<int32_t>(id));
  if (eos) out.push_back(kFakeEos);
  return out;
}

const std::string kWeatherTools =
    ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
    "\"description\":\"Weather\",\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"city\":{\"type\":\"string\"},\"days\":{\"type\":"
    "\"integer\"}},\"required\":[\"city\"]}}}]";

DGPP_TEST(serve_logprobs_visibleContentOnly) {
  ServiceRig rig(8, model_defaults(), true, std::nullopt, true);
  for (const auto& [turn, expected, extra] : {
       std::tuple{std::string("Think</think>OK<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value></tool_call>"),
                  std::string("OK"), std::string("")},
       std::tuple{std::string("Think</think>OKSTOPafter"), std::string("OK"), std::string(",\"stop\":\"STOP\"")},
       std::tuple{std::string("Think</think>OK<tool_call>broken"), std::string("OK<tool_call>broken"), std::string("")},
       std::tuple{std::string("Think</think>OKST"), std::string("OKST"), std::string(",\"stop\":\"STOP\"")}}) {
    for (bool stream : {false, true}) {
      rig.engine.script(5, script_of(rig, turn));
      const auto response = post_chat(rig, chat_body("abcd", 100,
          ",\"logprobs\":true" + extra + (stream ? ",\"stream\":true" : "")),
          stream ? "\r\n0\r\n\r\n" : "usage", 5000);
      std::string tokens, content;
      const auto inspect = [&](const dgpp::minijson::Value& choice) {
        const auto& msg = choice.at(stream ? "delta" : "message");
        if (const auto* text = msg.find("content"); text && text->is_string()) content += text->as_string();
        const auto& lp = choice.at("logprobs");
        if (!lp.is_null()) {
          require(lp.at("refusal").is_null(), "required refusal logprobs field");
          for (const auto& token : lp.at("content").items()) tokens += token.at("token").as_string();
        }
      };
      if (stream) {
        size_t at = 0;
        while ((at = response.find("data: ", at)) != std::string::npos) {
          at += 6;
          const auto end = response.find("\n\n", at);
          const auto payload = response.substr(at, end - at);
          if (payload == "[DONE]") break;
          const auto parsed = dgpp::minijson::parse(payload);
          for (const auto& choice : parsed.root.at("choices").items()) inspect(choice);
          at = end;
        }
      } else {
        const auto parsed = dgpp::minijson::parse(std::string_view(response).substr(response.find("\r\n\r\n") + 4));
        inspect(parsed.root.at("choices").items()[0]);
      }
      require(content == expected && tokens == expected,
              "visible tokens only, including buffered tails: content=" + content + " logprobs=" + tokens);
    }
  }
}

DGPP_TEST(serve_metrics_requestCountsDoNotCountChoices) {
  ServiceRig rig;
  (void)post_chat(rig, chat_body("abcd", 2, ",\"n\":3"), "usage");
  require(rig.service.stats().requests_total == 1, "n=3 is one HTTP request");
  (void)post_chat(rig, chat_body("abcd", 2, ",\"store\":true"), "unsupported_parameter");
  require(rig.service.stats().rejects_bad == 1, "bad request counter records the refusal");
}

DGPP_TEST(serve_api_allowedToolsAndToolTextParts) {
  ServiceRig rig(8, model_defaults(), true, std::nullopt, true);
  const std::string tools = R"(,"tools":[{"type":"function","function":{"name":"first"}},{"type":"function","function":{"name":"second"}}])";
  const std::string choice = R"(,"tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"required","tools":[{"type":"function","function":{"name":"second"}}]}})";
  (void)post_chat(rig, chat_body("abcd", 2, tools + choice), "usage");
  const auto grammars = rig.engine.grammars();
  require(grammars.size() == 1 && grammars[0].mode == dgpp::text::GrammarSpec::Mode::kRequired &&
              grammars[0].tools.size() == 1 && grammars[0].tools[0].name == "second",
          "only allowed function reaches the decoding constraint");
  const std::string body = "{\"model\":\"" + kModel +
      R"(","max_tokens":2,"messages":[{"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"ab"},{"type":"text","text":"cd"}]}]})";
  require(post_chat(rig, body, "usage").find("200 OK") != std::string::npos, "standard tool text parts accepted");
  require(rig.frontend.last_globals().find(R"("content":"abcd")") != std::string::npos,
          "tool text parts retain their order through template normalization");
  expect_invalid_request(post_chat(rig, chat_body("abcd", 2,
      R"(,"tools":[{"type":"function","function":{"name":"bad name"}}])")), "tools[0].function.name");
}

// Concatenates every `"<field>":"..."` payload in arrival order (the SSE
// delta contract: clients concatenate fragments).
std::string concat_field(const std::string& resp, const std::string& field) {
  std::string out;
  const std::string needle = "\"" + field + "\":\"";
  size_t pos = 0;
  while ((pos = resp.find(needle, pos)) != std::string::npos) {
    const size_t vstart = pos + needle.size();
    size_t vend = vstart;
    while (vend < resp.size() && resp[vend] != '"') vend += resp[vend] == '\\' ? 2 : 1;
    out.append(resp, vstart, vend - vstart);
    pos = vend;
  }
  return out;
}

DGPP_TEST(serve_tools_requestSideRendersThroughTheTemplateAndRefusesByName) {
  // GIVEN a frontend with the template's markers (tool calls available),
  ServiceRig rig(/*queue_limit=*/8, dgpp::sample::greedy_params(),
                 /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  {
    Client models(rig.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string ml = models.read_available(800);
    require(ml.find("\"tools\":{\"available\":true,\"constrained\":false},"
                    "\"response_format\":{\"json_object\":false,\"json_schema\":false},"
                    "\"reasoning\":{\"in_content\":false,\"effort_mapping\":") != std::string::npos,
            "models advertise the tool surface (no masks on a greedy engine): " +
                ml.substr(0, 400));
  }

  // WHEN requests carry tools and template knobs, THEN the template sees
  // exactly what OpenAI semantics prescribe.
  (void)post_until_usage(rig, chat_body("abcd", 2, kWeatherTools));
  std::string g = rig.frontend.last_globals();
  require(g.find("\"tools\":[{\"type\":\"function\",\"function\":{\"name\":"
                 "\"get_weather\"") != std::string::npos,
          "tools reach the template (tool_choice auto): " + g);
  require(g.find("\"messages\":[{\"role\":\"user\",\"content\":\"abcd\"}]") !=
              std::string::npos,
          "messages pass through: " + g);

  (void)post_until_usage(
      rig, chat_body("abcd", 2, kWeatherTools + ",\"tool_choice\":\"none\""));
  g = rig.frontend.last_globals();
  require(g.find("\"tools\"") == std::string::npos,
          "tool_choice none omits the tools from the render: " + g);

  (void)post_until_usage(
      rig, chat_body("abcd", 2,
                     ",\"reasoning_effort\":\"low\",\"chat_template_kwargs\":"
                     "{\"clear_thinking\":false}"));
  g = rig.frontend.last_globals();
  require(g.find("\"reasoning_effort\":\"low\"") != std::string::npos &&
              g.find("\"clear_thinking\":false") != std::string::npos,
          "reasoning_effort and chat_template_kwargs render: " + g);
  for (const std::string& extra : {
           std::string(",\"reasoning_effort\":\"xhigh\""),
           std::string(",\"chat_template_kwargs\":{\"reasoning_effort\":\"xhigh\"}")}) {
    (void)post_until_usage(rig, chat_body("abcd", 2, extra));
    require(rig.frontend.last_globals().find("\"reasoning_effort\":\"xhigh\"") != std::string::npos,
            "xhigh reaches the template through either API spelling");
  }

  // The assistant tool_calls wire form (arguments as a JSON string) is
  // parsed into the mapping the template iterates; a null assistant
  // content becomes ""; tool messages pass with their tool_call_id, in
  // both content forms.
  const std::string conversation =
      "{\"model\":\"" + kModel + "\",\"max_tokens\":2,\"messages\":["
      "{\"role\":\"user\",\"content\":\"hi\"},"
      "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"id\":"
      "\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
      "\"arguments\":\"{\\\"city\\\": \\\"Paris\\\", \\\"days\\\": 2}\"}}]},"
      "{\"role\":\"tool\",\"content\":\"18C\",\"tool_call_id\":\"call_1\"},"
      "{\"role\":\"assistant\",\"content\":\"It is 18C.\",\"reasoning_content\":"
      "\"looked it up\"},"
      "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"and Rome?\"}]},"
      "{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call_2\",\"type\":"
      "\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":"
      "{\"city\":\"Rome\"}}}]},"
      "{\"role\":\"tool\",\"content\":[{\"tool_call_id\":\"call_2\",\"output\":"
      "\"20C\"}]}]" + kWeatherTools + "}";
  const std::string ok = post_until_usage(rig, conversation);
  require(ok.find("\"object\":\"chat.completion\"") != std::string::npos,
          "the tool conversation is served: " + ok.substr(0, 300));
  g = rig.frontend.last_globals();
  require(g.find("{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":"
                 "\"call_1\",\"type\":\"function\",\"function\":{\"name\":"
                 "\"get_weather\",\"arguments\":{\"city\":\"Paris\",\"days\":2}}}]}") !=
              std::string::npos,
          "string arguments parsed to a mapping, null content to \"\": " + g);
  require(g.find("\"arguments\":{\"city\":\"Rome\"}") != std::string::npos,
          "object arguments pass through: " + g);
  require(g.find("\"reasoning_content\":\"looked it up\"") != std::string::npos,
          "reasoning_content passes through: " + g);
  require(g.find("{\"role\":\"tool\",\"content\":[{\"tool_call_id\":\"call_2\","
                 "\"output\":\"20C\"}]}") != std::string::npos,
          "the tool output list passes through: " + g);

  // Missing non-assistant content must not inherit assistant normalization.
  for (const std::string message :
       {"{\"role\":\"user\"}", "{\"role\":\"user\",\"content\":null}", "{\"role\":\"tool\"}"}) {
    expect_invalid_request(
        post_chat(rig, "{\"model\":\"" + kModel + "\",\"messages\":[" + message + "]}"),
        "messages[0].content");
  }

  // The refusals name the field.
  const auto refused = [&](const std::string& body, const std::string& needle) {
    const std::string resp = post_chat(rig, body);
    require(resp.find("400 ") != std::string::npos &&
                resp.find(needle) != std::string::npos,
            "expected a 400 with " + needle + ": " + resp.substr(0, 400));
  };
  refused(chat_body("abcd", 2, ",\"tool_choice\":\"required\""),
          "\"param\":\"tool_choice\"");
  refused(chat_body("abcd", 2, kWeatherTools + ",\"tool_choice\":\"sometimes\""),
          "\"param\":\"tool_choice\"");
  refused(chat_body("abcd", 2,
                    kWeatherTools + ",\"tool_choice\":{\"type\":\"function\","
                                    "\"function\":{\"name\":\"nope\"}}"),
          "\"param\":\"tool_choice.function.name\"");
  refused(chat_body("abcd", 2, kWeatherTools + ",\"parallel_tool_calls\":false"),
          "\"code\":\"constrained_decoding_unsupported\"");
  refused(chat_body("abcd", 2, kWeatherTools + ",\"tool_choice\":\"required\""),
          "\"code\":\"constrained_decoding_unsupported\"");
  refused(chat_body("abcd", 2, ",\"tools\":[{\"type\":\"function\","
                               "\"function\":{\"description\":\"x\"}}]"),
          "\"param\":\"tools[0].function.name\"");
  refused(chat_body("abcd", 2, ",\"reasoning_effort\":\"extreme\""),
          "\"param\":\"reasoning_effort\"");
  refused(chat_body("abcd", 2, ",\"chat_template_kwargs\":{\"enable_thinking\":false}"),
          "\"param\":\"chat_template_kwargs.enable_thinking\"");
  refused(chat_body("abcd", 2, ",\"chat_template_kwargs\":{\"foo\":1}"),
          "\"param\":\"chat_template_kwargs.foo\"");
  refused(chat_body("abcd", 2,
                    ",\"reasoning_effort\":\"low\",\"chat_template_kwargs\":"
                    "{\"reasoning_effort\":\"high\"}"),
          "\"param\":\"chat_template_kwargs.reasoning_effort\"");
  refused("{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"user\","
          "\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":null}]}",
          "\"param\":\"messages[1].content\"");
  refused("{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"user\","
          "\"content\":\"hi\"},{\"role\":\"assistant\",\"tool_calls\":[{\"function\":"
          "{\"name\":\"f\",\"arguments\":\"[1]\"}}]}]}",
          "\"param\":\"messages[1].tool_calls[0].function.arguments\"");
  refused("{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"tool\","
          "\"content\":7}]}",
          "\"param\":\"messages[0].content\"");
}

DGPP_TEST(serve_enableThinkingIsAcceptedOnlyWhenTheTemplateReadsIt) {
  // GLM-4.7's and Qwen3.8-Flash-Next's templates read enable_thinking (false
  // closes the think block in the generation prompt): the service passes
  // it through to the render as a boolean global. A template that never
  // reads it (GLM-5.3-Flash's) refuses it as before (the case above).
  ServiceRig rig;
  rig.frontend.reads_enable_thinking = true;
  const std::string ok = post_until_usage(
      rig, chat_body("abcd", 2, ",\"chat_template_kwargs\":{\"enable_thinking\":false}"));
  require(ok.find("\"object\":\"chat.completion\"") != std::string::npos,
          "enable_thinking accepted by a template that reads it: " + ok.substr(0, 300));
  const std::string g = rig.frontend.last_globals();
  require(g.find("\"enable_thinking\":false") != std::string::npos,
          "enable_thinking reaches the render globals: " + g);
  const std::string bad = post_chat(rig, chat_body("abcd", 2, ",\"chat_template_kwargs\":{\"enable_thinking\":\"no\"}"));
  require(bad.find("must be a boolean") != std::string::npos, "a non-boolean enable_thinking is refused: " + bad);
}

DGPP_TEST(serve_toolCalls_oneShotMessageShapeAndFinishReason) {
  // GIVEN a scripted turn: reasoning, </think>, content, one call with a
  // string and an integer argument, EOS (prompt "abcd" + <think> = 5 ids),
  ServiceRig rig(/*queue_limit=*/8, dgpp::sample::greedy_params(),
                 /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  const std::string turn =
      "Think</think>Sure<tool_call>get_weather<arg_key>city</arg_key>"
      "<arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3"
      "</arg_value></tool_call>";
  const std::vector<int32_t> script = script_of(rig, turn);
  rig.engine.script(5, script);

  // WHEN the chat completion completes,
  const std::string resp =
      post_until_usage(rig, chat_body("abcd", 64, kWeatherTools));

  // THEN the message carries content, reasoning_content and the parsed
  // call with json.dumps-form arguments; finish_reason is "tool_calls".
  require(resp.find("\"message\":{\"role\":\"assistant\",\"content\":\"Sure\","
                    "\"reasoning_content\":\"Think\",\"tool_calls\":[{\"id\":"
                    "\"call_") != std::string::npos,
          "message shape: " + resp);
  require(resp.find("\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
                    "\"arguments\":\"{\\\"city\\\": \\\"Paris\\\", \\\"days\\\": 3}\"}}],\"refusal\":null}") !=
              std::string::npos,
          "tool call shape: " + resp);
  require(resp.find("\"finish_reason\":\"tool_calls\"") != std::string::npos,
          "finish_reason tool_calls: " + resp);
  require(resp.find("\"prompt_tokens\":5,\"completion_tokens\":" +
                    std::to_string(script.size())) != std::string::npos,
          "usage counts every id (EOS included): " + resp);

  // A turn with calls only reports content null.
  ServiceRig rig2(/*queue_limit=*/8, dgpp::sample::greedy_params(),
                  /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  rig2.engine.script(
      6, script_of(rig2, "Think</think><tool_call>get_weather<arg_key>city"
                         "</arg_key><arg_value>Rome</arg_value></tool_call>"));
  const std::string only =
      post_until_usage(rig2, chat_body("abcde", 64, kWeatherTools));
  require(only.find("\"content\":null,\"reasoning_content\":\"Think\","
                    "\"tool_calls\":[") != std::string::npos,
          "content null with calls only: " + only);
  Client m(rig2.port());
  m.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string metrics = m.read_until("tool_calls_out", 2000);
  require(metrics.find("\"tool_calls_out\":1") != std::string::npos,
          "metrics count the call: " + metrics);
}

DGPP_TEST(serve_toolCalls_streamDeltasInOrder) {
  ServiceRig rig(/*queue_limit=*/8, dgpp::sample::greedy_params(),
                 /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  rig.engine.script(
      5, script_of(rig, "Think</think>Sure<tool_call>get_weather<arg_key>city"
                        "</arg_key><arg_value>Paris</arg_value><arg_key>days"
                        "</arg_key><arg_value>3</arg_value></tool_call>"
                        "<tool_call>get_weather<arg_key>city</arg_key>"
                        "<arg_value>Oslo</arg_value></tool_call>"));
  Client c(rig.port());
  const std::string body =
      chat_body("abcd", 128, kWeatherTools + ",\"stream\":true");
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  const std::string resp = c.read_until("[DONE]", 5000);

  // The deltas: reasoning_content fragments, content fragments, then per
  // call one announcing delta (index, id, name, empty arguments) and one
  // with the complete arguments; the final chunk says tool_calls.
  require(concat_field(resp, "reasoning_content") == "Think",
          "reasoning deltas concatenate: " + resp);
  require(concat_field(resp, "content") == "Sure",
          "content deltas concatenate: " + resp);
  const size_t role = resp.find("\"delta\":{\"role\":\"assistant\"");
  const size_t reasoning = resp.find("\"delta\":{\"reasoning_content\":\"");
  const size_t content = resp.find("\"delta\":{\"content\":\"");
  const size_t start0 = resp.find(
      "\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_");
  const size_t args0 = resp.find(
      "\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":"
      "\"{\\\"city\\\": \\\"Paris\\\", \\\"days\\\": 3}\"}}]}");
  const size_t start1 = resp.find(
      "\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_");
  const size_t args1 = resp.find(
      "\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":"
      "\"{\\\"city\\\": \\\"Oslo\\\"}\"}}]}");
  const size_t final = resp.find("\"delta\":{},\"logprobs\":null,"
                                 "\"finish_reason\":\"tool_calls\"");
  const size_t done = resp.find("data: [DONE]");
  require(role != std::string::npos && reasoning != std::string::npos &&
              content != std::string::npos && start0 != std::string::npos &&
              args0 != std::string::npos && start1 != std::string::npos &&
              args1 != std::string::npos && final != std::string::npos &&
              done != std::string::npos,
          "every chunk kind present: " + resp);
  require(role < reasoning && reasoning < content && content < start0 &&
              start0 < args0 && args0 < start1 && start1 < args1 &&
              args1 < final && final < done,
          "chunks in arrival order: " + resp);
  require(resp.find("\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
                    "\"arguments\":\"\"}}]}") != std::string::npos,
          "the announcing delta carries the name and empty arguments");
  // The two calls carry distinct ids.
  const size_t id0 = resp.find("\"id\":\"call_", start0);
  const size_t id1 = resp.find("\"id\":\"call_", start1);
  require(resp.substr(id0, 27) != resp.substr(id1, 27), "distinct call ids");
}

DGPP_TEST(serve_toolChoice_armsTheGrammarNotThePrompt) {
  // tool_choice required / named / none and parallel_tool_calls false ride
  // the request as a grammar (M6 6g): the prompt is untouched (the model
  // reasons first), the engine is armed with the spec, and the parsed
  // turn carries the reasoning and the call.
  using Mode = dgpp::text::GrammarSpec::Mode;
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true,
                 std::nullopt, /*with_markers=*/true);
  {
    Client models(rig.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    require(models.read_available(800).find(
                "\"tools\":{\"available\":true,\"constrained\":true}") !=
                std::string::npos,
            "models advertise constrained decoding");
  }
  const std::string closed_tools =
      ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
      "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":"
      "\"string\"},\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"],"
      "\"additionalProperties\":false}}},{\"type\":\"function\",\"function\":"
      "{\"name\":\"get_time\"}}]";
  rig.engine.script(5, script_of(rig, "Think</think><tool_call>get_weather"
                                      "<arg_key>city</arg_key><arg_value>Rome"
                                      "</arg_value></tool_call>"));
  // required: the prompt keeps its 5 ids, the turn reasons then calls.
  const std::string required = post_until_usage(
      rig, chat_body("abcd", 64, closed_tools + ",\"tool_choice\":\"required\""));
  require(required.find("\"prompt_tokens\":5,") != std::string::npos,
          "no forced prefix on the prompt: " + required);
  require(required.find("\"reasoning_content\":\"Think\"") != std::string::npos &&
              required.find("\"name\":\"get_weather\",\"arguments\":"
                            "\"{\\\"city\\\": \\\"Rome\\\"}\"") != std::string::npos &&
              required.find("\"finish_reason\":\"tool_calls\"") != std::string::npos,
          "the turn reasons first, then calls: " + required);
  // named, none, auto+single, required+single: the specs the engine got.
  (void)post_until_usage(
      rig, chat_body("abcd", 64,
                     closed_tools + ",\"tool_choice\":{\"type\":\"function\","
                                    "\"function\":{\"name\":\"get_time\"}}"));
  (void)post_until_usage(
      rig, chat_body("abcd", 64, closed_tools + ",\"tool_choice\":\"none\""));
  (void)post_until_usage(
      rig, chat_body("abcd", 64, closed_tools + ",\"parallel_tool_calls\":false"));
  (void)post_until_usage(
      rig, chat_body("abcd", 64, closed_tools + ",\"tool_choice\":\"required\","
                                                "\"parallel_tool_calls\":false"));
  (void)post_until_usage(rig, chat_body("abcd", 64, closed_tools));  // auto
  const std::vector<dgpp::text::GrammarSpec> g = rig.engine.grammars();
  require(g.size() == 6, "six tool requests armed a grammar (auto arms the "
                         "well-formed-call grammar too, M6 6i), got " +
                             std::to_string(g.size()));
  require(g[0].mode == Mode::kRequired && g[0].parallel &&
              g[0].tools.size() == 2 && g[0].tools[0].name == "get_weather" &&
              g[0].tools[0].constrain_keys &&
              g[0].tools[0].keys == std::vector<std::string>{"city", "days"} &&
              g[0].tools[1].name == "get_time" && !g[0].tools[1].constrain_keys,
          "required: the tools with their closed key set");
  // The typed arguments (M6 6i): city a free string, days an integer
  // under the JSON machine.
  using Kind = dgpp::text::GrammarArg::Kind;
  require(g[0].tools[0].args.size() == 2 && g[0].tools[0].args[0].key == "city" &&
              g[0].tools[0].args[0].kind == Kind::kFree &&
              g[0].tools[0].args[1].key == "days" && g[0].tools[0].args[1].kind == Kind::kJson &&
              g[0].tools[0].args[1].schema.find("integer") != std::string::npos &&
              g[0].tools[1].args.empty(),
          "the typed arguments ride with the tools");
  require(g[1].mode == Mode::kNamed && g[1].named == "get_time", "named");
  require(g[2].mode == Mode::kForbidCalls, "none forbids calls");
  require(rig.frontend.last_globals().find("\"tools\"") != std::string::npos,
          "the last (auto) request rendered the tools");
  require(g[3].mode == Mode::kAuto && !g[3].parallel, "auto + single call");
  require(g[4].mode == Mode::kRequired && !g[4].parallel, "required + single");
  require(g[5].mode == Mode::kAuto && g[5].parallel && g[5].tools.size() == 2,
          "auto: calls at will, every call well-formed");
  // An open schema (no additionalProperties: false) leaves the keys free.
  (void)post_until_usage(
      rig, chat_body("abcd", 64, kWeatherTools + ",\"tool_choice\":\"required\""));
  const std::vector<dgpp::text::GrammarSpec> g2 = rig.engine.grammars();
  require(g2.size() == 7 && !g2[6].tools[0].constrain_keys,
          "JSON Schema's default is open: keys unconstrained");
  // A strict function whose schema leaves the enforceable subset is a 400
  // naming the keyword path; the same schema without strict is served
  // with that value typed and the narrowing unenforced, and a
  // keyword that changes the value's shape leaves it free.
  const std::string strict_tools =
      ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"strict\":STRICT,"
      "\"parameters\":{\"type\":\"object\",\"properties\":{\"days\":{\"type\":"
      "\"number\",\"maxLength\":2}}}}}]";
  {
    std::string body = strict_tools;
    body.replace(body.find("STRICT"), 6, "true");
    const std::string resp = post_chat(rig, chat_body("abcd", 2, body));
    require(resp.find("400 ") != std::string::npos &&
                resp.find("\"param\":\"tools[0].function.parameters.properties.days.maxLength\"") !=
                    std::string::npos &&
                resp.find("\"code\":\"unsupported_schema\"") != std::string::npos,
            "strict refuses by keyword path: " + resp.substr(0, 400));
    body = strict_tools;
    body.replace(body.find("STRICT"), 6, "false");
    (void)post_until_usage(rig, chat_body("abcd", 64, body));
    const std::vector<dgpp::text::GrammarSpec> g3 = rig.engine.grammars();
    require(g3.size() == 8 && g3[7].tools[0].args.size() == 1 &&
                g3[7].tools[0].args[0].kind == Kind::kJson &&
                g3[7].tools[0].args[0].schema.find("maxLength") != std::string::npos,
            "non-strict: a narrowing keyword keeps the value typed");
    std::string shaped = strict_tools;
    shaped.replace(shaped.find("STRICT"), 6, "false");
    shaped.replace(shaped.find("\"maxLength\":2"), std::string("\"maxLength\":2").size(), "\"$ref\":\"#/x\"");
    (void)post_until_usage(rig, chat_body("abcd", 64, shaped));
    const std::vector<dgpp::text::GrammarSpec> g4 = rig.engine.grammars();
    require(g4.size() == 9 && g4[8].tools[0].args.size() == 1 &&
                g4[8].tools[0].args[0].kind == Kind::kFree,
            "non-strict: a shape keyword outside the subset leaves the value free");
    std::string bounded = strict_tools;
    bounded.replace(bounded.find("STRICT"), 6, "true");
    bounded.replace(bounded.find("\"maxLength\":2"), std::string("\"maxLength\":2").size(),
                    "\"minimum\":0,\"maximum\":10");
    (void)post_until_usage(rig, chat_body("abcd", 64, bounded));
    const auto numeric = rig.engine.grammars().back().tools[0];
    require(numeric.strict && numeric.args[0].kind == Kind::kJson &&
                numeric.args[0].schema.find("minimum") != std::string::npos,
            "strict tool functions accept and enforce number bounds");
  }
}

DGPP_TEST(serve_reasoning_foldKnobAndUnterminatedCallAtTheCap) {
  // The fold knob: reasoning rides as content with the model's own
  // </think>, no reasoning_content field.
  ServiceRig fold(/*queue_limit=*/8, dgpp::sample::greedy_params(),
                  /*can_sample=*/false, std::nullopt, /*with_markers=*/true,
                  /*reasoning_in_content=*/true);
  fold.engine.script(5, script_of(fold, "Think</think>Sure"));
  const std::string folded = post_until_usage(fold, chat_body("abcd", 64));
  require(folded.find("\"content\":\"Think</think>Sure\"") != std::string::npos &&
              folded.find("reasoning_content") == std::string::npos &&
              folded.find("\"finish_reason\":\"stop\"") != std::string::npos,
          "folded: " + folded);
  {
    Client models(fold.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    require(models.read_available(800).find("\"reasoning\":{\"in_content\":true,") !=
                std::string::npos,
            "models report the fold");
  }
  // Streamed, the fold's </think> is a content delta in place.
  {
    Client c(fold.port());
    const std::string body = chat_body("abcd", 64, ",\"stream\":true");
    c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    const std::string resp = c.read_until("[DONE]", 5000);
    require(concat_field(resp, "content") == "Think</think>Sure" &&
                resp.find("reasoning_content") == std::string::npos,
            "folded stream: " + resp);
  }

  // The steps cap inside a block: the literal text is content,
  // finish_reason length, no tool_calls.
  ServiceRig cap(/*queue_limit=*/8, dgpp::sample::greedy_params(),
                 /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  cap.engine.script(5, script_of(cap, "</think><tool_call>get_weather<arg_key>"
                                      "city</arg_key><arg_value>Paris"));
  const std::string capped =
      post_until_usage(cap, chat_body("abcd", 6, kWeatherTools));
  require(capped.find("\"content\":\"<tool_call>get_\"") != std::string::npos &&
              capped.find("tool_calls") == std::string::npos &&
              capped.find("\"finish_reason\":\"length\"") != std::string::npos &&
              capped.find("\"completion_tokens\":6,") != std::string::npos,
          "capped block: " + capped);
}

DGPP_TEST(serve_responseFormat_armsTheJsonGrammar) {
  // response_format (M6 6h): json_object and json_schema ride the request
  // as the JSON grammar — the prompt untouched, the engine armed with the
  // schema text — and the content is the JSON the model produced.
  using Mode = dgpp::text::GrammarSpec::Mode;
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true,
                 std::nullopt, /*with_markers=*/true);
  {
    Client models(rig.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    require(models.read_available(800).find(
                "\"response_format\":{\"json_object\":true,\"json_schema\":true}") !=
                std::string::npos,
            "models advertise response_format");
  }
  rig.engine.script(5, script_of(rig, "Think</think>{\"city\": \"Rome\"}"));
  const std::string json_object = post_until_usage(
      rig, chat_body("abcd", 64, ",\"response_format\":{\"type\":\"json_object\"}"));
  require(json_object.find("\"prompt_tokens\":5,") != std::string::npos &&
              json_object.find("\"reasoning_content\":\"Think\"") != std::string::npos &&
              json_object.find("\"content\":\"{\\\"city\\\": \\\"Rome\\\"}\"") !=
                  std::string::npos &&
              json_object.find("\"finish_reason\":\"stop\"") != std::string::npos,
          "json_object: reasoning then the JSON as content: " + json_object);
  const std::string schema =
      "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},"
      "\"required\":[\"city\"],\"additionalProperties\":false}";
  (void)post_until_usage(
      rig, chat_body("abcd", 64,
                     ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
                     "{\"name\":\"place\",\"strict\":true,\"schema\":" + schema + "}}"));
  // Not strict and outside the subset: served as json_object (a warning).
  (void)post_until_usage(
      rig, chat_body("abcd", 64,
                     ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
                     "{\"name\":\"p\",\"schema\":{\"type\":\"string\",\"maxLength\":3}}}"));
  // type text: no grammar.
  (void)post_until_usage(rig, chat_body("abcd", 64, ",\"response_format\":{\"type\":\"text\"}"));
  const std::vector<dgpp::text::GrammarSpec> g = rig.engine.grammars();
  require(g.size() == 3, "three JSON requests armed a grammar (text arms none), got " +
                             std::to_string(g.size()));
  require(g[0].mode == Mode::kJson && g[0].json_schema.empty() && g[0].tools.empty(),
          "json_object: the free JSON grammar");
  require(g[1].mode == Mode::kJson && g[1].json_schema.find("\"city\"") != std::string::npos &&
              g[1].json_schema.find("additionalProperties") != std::string::npos,
          "json_schema: the schema text rides: " + g[1].json_schema);
  require(g[2].mode == Mode::kJson && g[2].json_schema.empty(),
          "non-strict unsupported schema falls back to json_object");
  require(rig.frontend.last_globals().find("response_format") == std::string::npos,
          "the prompt does not carry the format");

  const std::string numeric_format =
      R"(,"response_format":{"type":"json_schema","json_schema":{"name":"score","strict":true,"schema":{"type":"number","minimum":0,"maximum":10}}})";
  (void)post_until_usage(rig, chat_body("abcd", 64, numeric_format));
  require(rig.engine.grammars().back().json_schema.find("\"minimum\": 0") != std::string::npos,
          "strict number bounds reach the engine");
  for (const std::string& format : {std::string(R"(,"response_format":{"type":"json_object"})"), numeric_format}) {
    for (const auto& choice : std::vector<std::pair<std::string, Mode>>{
             {"", Mode::kJsonOrTools},
             {R"(,"tool_choice":"none")", Mode::kJson},
             {R"(,"tool_choice":"required")", Mode::kRequired},
             {R"(,"tool_choice":{"type":"function","function":{"name":"get_weather"}})", Mode::kNamed}}) {
      (void)post_until_usage(rig, chat_body("abcd", 64, kWeatherTools + format + choice.first +
                                          R"(,"parallel_tool_calls":false)"));
      const auto grammar = rig.engine.grammars().back();
      require(grammar.mode == choice.second && !grammar.parallel && !grammar.tools.empty(),
              "response format preserves tool choice, definitions and parallel flag");
    }
    // The final turn keeps the tools and includes an assistant call and
    // its result, as a normal agent loop does.
    const std::string conversation =
        "{\"model\":\"" + kModel + "\",\"max_tokens\":64,\"messages\":["
        R"({"role":"user","content":"abcd"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Rome\"}"}}]},{"role":"tool","tool_call_id":"c1","content":"18C"}])" +
        kWeatherTools + format + "}";
    const std::string response = post_until_usage(rig, conversation);
    require(response.find("200 ") != std::string::npos &&
                rig.engine.grammars().back().mode == Mode::kJsonOrTools,
            "tool result followed by structured output is accepted: " + response.substr(0, 300));
  }

  // The refusals: strict + unsupported keyword, bad type, missing name.
  const auto refused = [&](const std::string& extra, const std::string& param,
                           const std::string& code) {
    const std::string resp = post_chat(rig, chat_body("abcd", 2, extra));
    require(resp.find("400 ") != std::string::npos &&
                resp.find("\"param\":\"" + param + "\"") != std::string::npos &&
                (code.empty() || resp.find("\"code\":\"" + code + "\"") != std::string::npos),
            "expected a 400 naming " + param + " / " + code + ": " + resp.substr(0, 400));
  };
  refused(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"p\","
          "\"strict\":true,\"schema\":{\"type\":\"object\",\"properties\":{\"city\":"
          "{\"type\":\"string\",\"maxLength\":3}}}}}",
          "response_format.json_schema.schema.properties.city.maxLength", "unsupported_schema");
  refused(",\"response_format\":{\"type\":\"yaml\"}", "response_format.type", "");
  refused(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"schema\":{}}}",
          "response_format.json_schema.name", "");
  refused(",\"response_format\":\"json_object\"", "response_format", "");
  // A greedy engine has no masks: JSON modes are refused, text is fine.
  ServiceRig greedy(/*queue_limit=*/8, dgpp::sample::greedy_params(),
                    /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  {
    const std::string resp = post_chat(
        greedy, chat_body("abcd", 2, ",\"response_format\":{\"type\":\"json_object\"}"));
    require(resp.find("400 ") != std::string::npos &&
                resp.find("\"code\":\"constrained_decoding_unsupported\"") != std::string::npos,
            "greedy engine refuses json_object: " + resp.substr(0, 300));
    Client models(greedy.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    require(models.read_available(800).find(
                "\"response_format\":{\"json_object\":false,\"json_schema\":false}") !=
                std::string::npos,
            "models report the absence");
  }
}

DGPP_TEST(serve_shutdown_drainsInFlightWorkWithTheShutdownError) {
  // Drain-on-stop (M6 6c): a stream mid-generation is retired by the drain
  // pass and answered with the server_shutdown error event and [DONE]
  // after the tokens it produced (no finish chunk); a request still in
  // the admission queue is shed with a 503 server_shutdown; a request
  // arriving after the door closed gets the same 503; the metrics count
  // the interruption as a cancellation; drained() turns true only once
  // every answer is out.
  ServiceRig rig(/*queue_limit=*/8);
  rig.engine.script(5, script_of(rig, std::string(300, 'x')));
  rig.pass_delay_ms = 2;  // ~2 ms per token: the pump sees it mid-generation
  // A stream: let it produce a few chunks, then hold the engine between
  // passes — the request is live in the scheduler, mid-generation.
  Client stream(rig.port());
  {
    const std::string body = chat_body("abcd", 300, ",\"stream\":true");
    stream.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                    "Content-Type: application/json\r\nContent-Length: " +
                    std::to_string(body.size()) + "\r\n\r\n" + body);
  }
  std::string head = stream.read_until("\"content\":\"x", 3000);
  require(head.find("\"content\":\"x") != std::string::npos,
          "the stream produced content before the stop: " + head.substr(0, 300));
  rig.gate = true;
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  // A one-shot that lands in the admission queue while the engine is held.
  Client queued(rig.port());
  {
    const std::string body = chat_body("efgh", 8);
    queued.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                    "Content-Type: application/json\r\nContent-Length: " +
                    std::to_string(body.size()) + "\r\n\r\n" + body);
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(60));
  require(!rig.service.drained(), "not drained while work is live");
  const int interrupted = rig.drain(/*stop_http=*/false);
  require(interrupted == 1, "one live request interrupted, got " + std::to_string(interrupted));
  const std::string tail = head + stream.read_until("[DONE]", 3000);
  require(tail.find("\"code\":\"server_shutdown\"") != std::string::npos &&
              tail.find("this response is incomplete") != std::string::npos &&
              tail.find("data: [DONE]") != std::string::npos,
          "the interrupted stream ends with the shutdown error event: " + tail.substr(tail.size() > 600 ? tail.size() - 600 : 0));
  require(tail.find("\"finish_reason\":\"stop\"") == std::string::npos &&
              tail.find("\"finish_reason\":\"length\"") == std::string::npos,
          "no finish chunk on an interrupted stream");
  const std::string shed = queued.read_until("}}", 3000);
  require(shed.find("503 ") != std::string::npos &&
              shed.find("\"code\":\"server_shutdown\"") != std::string::npos &&
              shed.find("retry on another instance") != std::string::npos,
          "the queued one-shot is shed with 503 server_shutdown: " + shed.substr(0, 400));
  // The door is closed: a new request gets the same 503.
  const std::string late = post_chat(rig, chat_body("ijkl", 4));
  require(late.find("503 ") != std::string::npos &&
              late.find("\"code\":\"server_shutdown\"") != std::string::npos,
          "a request after the stop is refused: " + late.substr(0, 300));
  require(rig.service.drained(), "drained once every answer is out");
  {
    Client m(rig.port());
    m.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string metrics = m.read_until("requests_cancelled", 2000);
    require(metrics.find("\"requests_cancelled\":1") != std::string::npos &&
                metrics.find("\"requests_shed\":2") != std::string::npos,
            "metrics: one interruption, two sheds: " + metrics.substr(0, 400));
  }
}

DGPP_TEST(serve_engineFailure_answersLiveStreamsAfterTheirCommittedTokensOnly) {
  // The v1 failure semantics (M8's exit criterion "injected rank failure
  // leaves committed state unchanged", DESIGN §9 item 6): an engine op
  // throws mid-generation. The stream gets exactly the tokens the scheduler
  // committed before the failure (the fake fails on its n-th step, so the
  // count is known: the prefill's token plus n-1 steps), then the
  // engine_failure error event and [DONE], no finish chunk; a later request
  // is refused 503 engine_failure at the door; /health turns 503 with the
  // reason; the metrics count the interruption; drained() turns true once
  // every answer is out — the app then exits nonzero.
  ServiceRig rig(/*queue_limit=*/8);
  std::string text;
  for (int i = 0; i < 300; ++i) text.push_back(static_cast<char>('a' + i % 26));
  rig.engine.script(4, script_of(rig, text));  // "abcd" renders to 4 ids
  rig.pass_delay_ms = 2;
  constexpr int kFailAt = 6;  // tokens committed: prefill + 5 steps = 6
  rig.engine.fail_at_step(kFailAt);
  Client stream(rig.port());
  {
    const std::string body = chat_body("abcd", 300, ",\"stream\":true");
    stream.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                    "Content-Type: application/json\r\nContent-Length: " +
                    std::to_string(body.size()) + "\r\n\r\n" + body);
  }
  const std::string raw = stream.read_until("data: [DONE]", 5000);
  require(rig.failed.load(), "the rig's engine loop ran the failure path");
  require(rig.service.failed(), "the service reports the failure");
  // The committed tokens, all of them and nothing more.
  std::string content;
  size_t at = 0;
  while ((at = raw.find("\"content\":\"", at)) != std::string::npos) {
    at += 11;
    const size_t end = raw.find('"', at);
    content.append(raw, at, end - at);
    at = end;
  }
  require(content == text.substr(0, kFailAt),
          "the stream carries exactly the " + std::to_string(kFailAt) +
              " committed tokens: got '" + content + "'");
  require(raw.find("\"code\":\"engine_failure\"") != std::string::npos &&
              raw.find("injected engine failure") != std::string::npos &&
              raw.find("this response is incomplete") != std::string::npos,
          "the stream ends with the engine_failure error event naming the cause: " +
              raw.substr(raw.size() > 500 ? raw.size() - 500 : 0));
  require(raw.find("\"finish_reason\":\"") == std::string::npos,
          "no finish chunk on a failed stream");
  const size_t err_at = raw.find("\"error\"");
  require(raw.find("\"content\":\"", err_at) == std::string::npos,
          "no token after the error event");
  // The door: a later request is refused with the failure.
  const std::string late = post_chat(rig, chat_body("ijkl", 4));
  require(late.find("503 ") != std::string::npos &&
              late.find("\"code\":\"engine_failure\"") != std::string::npos &&
              late.find("restarting") != std::string::npos,
          "a request after the failure is refused 503 engine_failure: " + late.substr(0, 300));
  {
    Client h(rig.port());
    h.send_all("GET /health HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string health = h.read_until("}", 2000);
    require(health.find("503 ") != std::string::npos &&
                health.find("\"status\":\"failed\"") != std::string::npos &&
                health.find("injected engine failure") != std::string::npos,
            "/health reports the failure: " + health.substr(0, 300));
  }
  {
    Client m(rig.port());
    m.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string metrics = m.read_until("engine_failed", 2000);
    require(metrics.find("\"requests_failed\":1") != std::string::npos &&
                metrics.find("\"engine_failed\":true") != std::string::npos,
            "metrics: one request failed, the engine flagged: " + metrics.substr(0, 400));
  }
  for (int i = 0; i < 200 && !rig.service.drained(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  require(rig.service.drained(), "drained once every answer is out");
  require(rig.service.fail_engine("again") == 0, "fail_engine is idempotent");
}

DGPP_TEST(serve_admission_growPolicyShedsTheYoungestWithFinishLength) {
  // Grow-on-demand (M6 6d) on the service: two 300-token requests whose
  // lifetimes (77 blocks each) exceed the 100-block pool together are both
  // admitted under grow (window 8), grow at tick top, and when the pool is
  // spent the younger is cut short with finish_reason "length" while the
  // older completes; the metrics carry the policy, the growth count and
  // the shed. The default rig reports the full-reserve policy.
  dgpp::sched::AdmissionPolicy grow;
  grow.mode = dgpp::sched::AdmissionPolicy::Mode::kGrowOnDemand;
  grow.window_tokens = 8;
  ServiceRig rig(/*queue_limit=*/8, dgpp::sample::greedy_params(),
                 /*can_sample=*/false, std::nullopt, /*with_markers=*/false,
                 /*reasoning_in_content=*/false, grow);
  rig.engine.script(5, script_of(rig, std::string(300, 'x')));
  Client a(rig.port()), b(rig.port());
  for (Client* c : {&a, &b}) {
    const std::string body = chat_body("abcd", 300);
    c->send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
                "Content-Type: application/json\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n" + body);
  }
  const std::string ra = a.read_until("\"usage\"", 10000);
  const std::string rb = b.read_until("\"usage\"", 10000);
  const auto completion_tokens = [](const std::string& r) {
    const size_t at = r.find("\"completion_tokens\":");
    return at == std::string::npos ? -1 : std::atoi(r.c_str() + at + 20);
  };
  const int ta = completion_tokens(ra), tb = completion_tokens(rb);
  require(ra.find("\"finish_reason\":\"length\"") != std::string::npos &&
              rb.find("\"finish_reason\":\"length\"") != std::string::npos,
          "both finish length: " + ra.substr(0, 200) + " / " + rb.substr(0, 200));
  require((ta == 300 && tb > 0 && tb < 300) || (tb == 300 && ta > 0 && ta < 300),
          "one completes 300 tokens, the other is cut short: " + std::to_string(ta) +
              " / " + std::to_string(tb));
  {
    Client m(rig.port());
    m.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string metrics = m.read_until("\"admission\"", 2000);
    require(metrics.find("\"requests_shed_pool\":1") != std::string::npos &&
                metrics.find("\"admission\":{\"mode\":\"grow\",\"window\":8,\"prefill_budget_tokens\":0,\"prefill_idle_budget_tokens\":0}") !=
                    std::string::npos &&
                metrics.find("\"reservations_grown\":0") == std::string::npos,
            "metrics: the shed, the policy, the growth: " + metrics.substr(0, 500));
  }
  ServiceRig plain(/*queue_limit=*/8);
  Client m(plain.port());
  m.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
  require(m.read_until("\"admission\"", 2000).find(
              "\"admission\":{\"mode\":\"full\",\"window\":256,\"prefill_budget_tokens\":0,\"prefill_idle_budget_tokens\":0}") != std::string::npos,
          "the default policy is full-reserve");
}

}  // namespace

// The prefix cache through the service (M7 stage B): the boundaries come
// from the frontend's boundary token ('|' here) at their positions in the
// prompt ids, the second identical request attaches to the first's entry
// at the deepest cut (position 4 of "ab|cd|ef": the image of the boundary
// at 5), "prefix_cache": false opts out, a non-boolean is refused by name,
// and /v1/metrics reports the cache.

// ---------------------------------------------------------------------------
// M9's malformed-HTTP fuzzing (2026-09-05): a byte-level mutator over the
// server's limit ladder. Valid requests (health, models, chat, legacy
// completions, a stream, a tools call) are mutated — bit flips, byte
// insertions and deletions, truncations, duplicated slices, hostile
// Content-Length values, a chunked Transfer-Encoding, oversized request
// lines, header floods past the 16 KiB cap, JSON garbage and invalid
// UTF-8 in the body, pipelined pairs, foreign HTTP versions, pure binary —
// and delivered whole, in fragments, half-closed or abandoned. The server
// must answer (any status), wait for more (an incomplete request) or
// close; it must never crash, wedge, or stop serving: after the storm a
// well-formed health probe and a chat completion answer exactly as before.
// Deterministic under DGPP_FUZZ_SEED; DGPP_FUZZ_ITERS scales the run
// (ctest's default is short; the ASan run goes long).
// ---------------------------------------------------------------------------
class FuzzConn {
 public:
  explicit FuzzConn(uint16_t port) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd_ >= 0, "fuzz socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    require(::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr) == 1, "fuzz inet_pton");
    require(::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
            "fuzz connect");
    int yes = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
  }
  ~FuzzConn() {
    if (fd_ >= 0) ::close(fd_);
  }
  FuzzConn(const FuzzConn&) = delete;
  FuzzConn& operator=(const FuzzConn&) = delete;
  // Sends what it can; a peer that already closed (a 431 + close) is fine.
  void send_lossy(std::string_view s) {
    size_t off = 0;
    while (off < s.size()) {
      const ssize_t put = ::send(fd_, s.data() + off, s.size() - off, MSG_NOSIGNAL);
      if (put <= 0) return;
      off += static_cast<size_t>(put);
    }
  }
  void half_close() { ::shutdown(fd_, SHUT_WR); }
  // Whatever arrives within the budget; true when the peer closed.
  std::string read_some(int timeout_ms, bool* closed) {
    std::string out;
    *closed = false;
    int waited = 0;
    while (waited < timeout_ms) {
      pollfd p{fd_, POLLIN, 0};
      const int r = ::poll(&p, 1, 10);
      if (r < 0) break;
      if (r == 0) {
        waited += 10;
        continue;
      }
      char buf[4096];
      const ssize_t n = ::recv(fd_, buf, sizeof buf, 0);
      if (n <= 0) {
        *closed = true;
        break;
      }
      out.append(buf, static_cast<size_t>(n));
    }
    return out;
  }

 private:
  int fd_ = -1;
};

DGPP_TEST(serve_fuzz_malformedHttpNeverBreaksTheServer) {
  ServiceRig rig(/*queue_limit=*/8);
  uint64_t seed = 0x9e3779b97f4a7c15ull;
  int iters = 800;
  if (const char* s = std::getenv("DGPP_FUZZ_SEED"); s && *s) seed = std::strtoull(s, nullptr, 0);
  if (const char* s = std::getenv("DGPP_FUZZ_ITERS"); s && *s) iters = std::atoi(s);
  auto next = [&]() {  // splitmix64
    seed += 0x9e3779b97f4a7c15ull;
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  };
  const auto pick = [&](size_t n) { return n == 0 ? size_t{0} : static_cast<size_t>(next() % n); };
  const auto framed = [](const std::string& method, const std::string& path,
                         const std::string& body) {
    std::string r = method + " " + path + " HTTP/1.1\r\nHost: t\r\n";
    if (!body.empty())
      r += "Content-Type: application/json\r\nContent-Length: " +
           std::to_string(body.size()) + "\r\n";
    return r + "\r\n" + body;
  };
  const std::vector<std::string> templates = {
      framed("GET", "/health", ""),
      framed("GET", "/v1/models", ""),
      framed("POST", "/v1/chat/completions", chat_body("abcd", 2)),
      framed("POST", "/v1/chat/completions", chat_body("abcd", 3, ",\"stream\":true")),
      framed("POST", "/v1/completions",
             "{\"model\":\"" + kModel + "\",\"prompt\":\"abcd\",\"max_tokens\":2}"),
      framed("POST", "/v1/chat/completions",
             chat_body("abcd", 2, kWeatherTools + ",\"tool_choice\":\"auto\"")),
      framed("POST", "/v1/chat/completions",
             chat_body("abcd", 2, ",\"temperature\":0.7,\"top_p\":0.9,\"logprobs\":true,"
                                  "\"top_logprobs\":2,\"seed\":7")),
  };
  const char* cl_values[] = {"-1", "0", "99999999999", "abc", "1e9", "", "4294967296",
                             "18446744073709551616", " 12", "12 13", "0x10"};
  size_t answered = 0, silent = 0, closed_on_us = 0;
  std::map<std::string, size_t> statuses;
  for (int it = 0; it < iters; ++it) {
    std::string req = templates[pick(templates.size())];
    const int k = 1 + static_cast<int>(pick(4));
    for (int m = 0; m < k; ++m) {
      switch (pick(13)) {
        case 0: if (!req.empty()) req[pick(req.size())] ^= static_cast<char>(1 << pick(8)); break;
        case 1: if (!req.empty()) req.erase(pick(req.size()), 1); break;
        case 2: req.insert(pick(req.size() + 1), 1, static_cast<char>(next() & 0xff)); break;
        case 3: req.resize(pick(req.size() + 1)); break;
        case 4: {
          if (req.empty()) break;
          const size_t a = pick(req.size()), n = pick(req.size() - a + 1);
          req.insert(pick(req.size() + 1), req.substr(a, n));
          break;
        }
        case 5: {
          const size_t at = req.find("Content-Length: ");
          if (at == std::string::npos) break;
          const size_t end = req.find("\r\n", at);
          req.replace(at + 16, end - (at + 16), cl_values[pick(sizeof cl_values / sizeof *cl_values)]);
          break;
        }
        case 6: {
          const size_t at = req.find("\r\n");
          if (at != std::string::npos) req.insert(at + 2, "Transfer-Encoding: chunked\r\n");
          break;
        }
        case 7: req = "GET /" + std::string(pick(24 * 1024), 'a') + " HTTP/1.1\r\nHost: t\r\n\r\n" + req; break;
        case 8: {
          const size_t at = req.find("\r\n");
          if (at == std::string::npos) break;
          std::string flood;
          const size_t n = 1 + pick(700);
          for (size_t i = 0; i < n; ++i) flood += "X-F" + std::to_string(i) + ": " + std::string(1 + pick(40), 'y') + "\r\n";
          req.insert(at + 2, flood);
          break;
        }
        case 9: {
          const size_t at = req.find("\r\n\r\n");
          if (at == std::string::npos) break;
          std::string body;
          switch (pick(6)) {
            case 0: body = std::string(1 + pick(200), '{'); break;
            case 1: body = "{\"messages\":[{\"role\":\"user\",\"content\":\"\xff\xfe\xc3\x28\"}],\"model\":\"" + kModel + "\"}"; break;
            case 2: body = "{\"model\":\"" + kModel + "\",\"messages\":[],\"max_tokens\":1e400}"; break;
            case 3: body = "[" + std::string(pick(5000), '[') + "]"; break;
            case 4: body = "{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"user\",\"content\":" + std::to_string(next()) + "}],\"max_tokens\":-5,\"temperature\":\"hot\",\"tools\":{},\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"schema\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\",\"pattern\":\"^\"}}}}}}"; break;
            default: for (size_t i = 0; i < 1 + pick(300); ++i) body.push_back(static_cast<char>(next() & 0xff)); break;
          }
          req = req.substr(0, at + 4) + body;
          const size_t cl = req.find("Content-Length: ");
          if (cl != std::string::npos && pick(2) == 0) {
            const size_t end = req.find("\r\n", cl);
            req.replace(cl + 16, end - (cl + 16), std::to_string(body.size()));
          }
          break;
        }
        case 10: req += templates[pick(templates.size())]; break;
        case 11: {
          const size_t at = req.find("HTTP/1.1");
          if (at != std::string::npos) req.replace(at, 8, pick(2) ? "HTTP/1.0" : "HTTP/2.0");
          break;
        }
        default: {
          req.clear();
          for (size_t i = 0; i < 1 + pick(600); ++i) req.push_back(static_cast<char>(next() & 0xff));
          break;
        }
      }
    }
    FuzzConn c(rig.port());
    const size_t style = pick(10);
    if (style < 7) {
      c.send_lossy(req);
    } else {
      size_t off = 0;
      const size_t pieces = 1 + pick(5);
      for (size_t p = 0; p < pieces && off < req.size(); ++p) {
        const size_t n = p + 1 == pieces ? req.size() - off : 1 + pick(req.size() - off);
        c.send_lossy(std::string_view(req).substr(off, n));
        off += n;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    if (style == 8) c.half_close();
    if (style == 9) continue;  // abandoned without reading
    bool closed = false;
    const std::string got = c.read_some(style == 8 ? 200 : 40, &closed);
    if (got.rfind("HTTP/1.1 ", 0) == 0) {
      ++answered;
      ++statuses[got.substr(9, 3)];
    } else if (closed) {
      ++closed_on_us;
    } else {
      require(got.empty(), "a reply that is not an HTTP/1.1 status line: " + got.substr(0, 80));
      ++silent;
    }
  }
  // The server is unharmed: a health probe and a completion as in the
  // first test, on fresh connections.
  {
    Client h(rig.port());
    h.send_all("GET /health HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string health = h.read_until("}", 3000);
    require(health.find("200 ") != std::string::npos && health.find("\"status\":\"ok\"") != std::string::npos,
            "after the fuzz storm /health still answers: " + health.substr(0, 200));
  }
  {
    const std::string resp = post_chat(rig, chat_body("abcd", 3), "usage", 5000);
    require(resp.find("200 ") != std::string::npos &&
                resp.find("\"content\":\"" + fake_text(4, 3) + "\"") != std::string::npos,
            "after the fuzz storm a completion answers exactly: " + resp.substr(0, 300));
  }
  require(!rig.failed.load() && !rig.service.failed(), "the engine never failed");
  std::string table;
  for (const auto& [code, n] : statuses) table += code + ":" + std::to_string(n) + " ";
  DGPP_LOG_INFO("fuzz: {} iterations, {} answered ({}), {} closed on us, {} silent (incomplete requests)",
                iters, answered, table, closed_on_us, silent);
  require(statuses.count("400") + statuses.count("431") + statuses.count("413") +
                  statuses.count("411") + statuses.count("501") > 0,
          "the limit ladder was exercised");
}

DGPP_TEST(serve_garbageAfterAPendingOneShot_waitsThenClosesCleanly) {
  // The fuzzer's second find (ASan, 2026-09-05): a valid one-shot followed
  // on the same connection by a malformed request. The server used to
  // parse the garbage at once, answer 400 with close, and free the
  // connection at the pass's sweep without telling the service, whose
  // pending record still held the writer — a use-after-free when the
  // engine's answer arrived. Two fixes: every close of a tagged connection
  // notifies the handler (defense in depth), and a connection carries one
  // request at a time — the garbage waits in the buffer behind the pending
  // one-shot, is parsed after its answer, and the 400 + close then dangles
  // nothing. Nothing is cancelled; the service serves on.
  ServiceRig rig(/*queue_limit=*/8);
  rig.gate = true;  // the one-shot stays pending in the scheduler
  Client c(rig.port());
  const std::string body = chat_body("abcd", 4);
  c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body + "GARBAGE\r\n\r\n");
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  require(c.read_available(50).empty(),
          "the garbage waits behind the pending one-shot (nothing answered yet)");
  rig.gate = false;
  const std::string raw = c.read_until("malformed request line", 5000);
  const size_t ok = raw.find("HTTP/1.1 200");
  const size_t bad = raw.find("HTTP/1.1 400");
  require(ok != std::string::npos && bad != std::string::npos && ok < bad &&
              raw.find("\"content\":\"" + fake_text(4, 4) + "\"") != std::string::npos,
          "the one-shot's exact answer, then the 400 for the garbage: " + raw.substr(0, 400));
  for (int i = 0; i < 200 && !rig.service.drained(); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  require(rig.service.drained(), "nothing owed to a closed connection");
  {
    Client m(rig.port());
    m.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string metrics = m.read_until("engine_failed", 2000);
    require(metrics.find("\"requests_cancelled\":0") != std::string::npos,
            "nothing was orphaned or cancelled: " + metrics.substr(0, 300));
  }
  const std::string resp = post_chat(rig, chat_body("abcd", 3), "usage", 5000);
  require(resp.find("\"content\":\"" + fake_text(4, 3) + "\"") != std::string::npos,
          "the service serves on: " + resp.substr(0, 200));
}

DGPP_TEST(serve_pipelinedRequests_areAnsweredOneAtATimeInOrder) {
  // The fuzzer's third find (ASan, 2026-09-05): two requests pipelined on
  // one connection were dispatched back to back, the second overwrote the
  // connection's disconnect tag, and the first record's writer was freed
  // at the close without notice. The server now reads one request at a
  // time per connection: the second waits in the buffer until the first is
  // answered, then is parsed at the loop's tail. Both answers arrive, in
  // order, each exact.
  ServiceRig rig(/*queue_limit=*/8);
  rig.gate = true;  // the first request stays pending while the second is buffered
  Client c(rig.port());
  const std::string b1 = chat_body("abcd", 2), b2 = chat_body("abcd", 3);
  const auto framed = [](const std::string& b) {
    return "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
           "Content-Type: application/json\r\nContent-Length: " +
           std::to_string(b.size()) + "\r\n\r\n" + b;
  };
  c.send_all(framed(b1) + framed(b2));
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  require(c.read_available(50).empty(), "nothing answered while the engine is held");
  rig.gate = false;
  std::string raw;
  for (int i = 0; i < 400; ++i) {
    raw += c.read_available(20);
    size_t n = 0, at = 0;
    while ((at = raw.find("HTTP/1.1 200", at)) != std::string::npos) { ++n; at += 12; }
    if (n >= 2 && raw.rfind("usage") != std::string::npos && raw.find("usage") != raw.rfind("usage")) break;
  }
  const size_t first = raw.find("\"content\":\"" + fake_text(4, 2) + "\"");
  const size_t second = raw.find("\"content\":\"" + fake_text(4, 3) + "\"");
  require(first != std::string::npos && second != std::string::npos && first < second,
          "both pipelined answers, in order: " + raw.substr(0, 400));
  require(rig.service.drained(), "nothing owed");
}

DGPP_TEST(serve_utf8_aCharacterSplitAcrossTokensIsHeldUntilComplete) {
  // The soak's find: a byte-level BPE token can end inside a
  // multi-byte character, and the delta carried its bytes as they came —
  // a JSON text that is not UTF-8, on which a strict client (Python's
  // json.loads over the payload bytes) raised; the soak's chat workers died
  // on the first accented name. A stream now holds an incomplete trailing
  // sequence back until the next delta completes it, renders one still
  // held at the end as U+FFFD, and a one-shot's cap-cut last character
  // becomes U+FFFD too. The fake tokenizer decodes ids below 256 to single
  // bytes, so "é" is the two tokens 0xC3 0xA9 and a lone 0xE2 an
  // incomplete character.
  ServiceRig rig(/*queue_limit=*/8);
  rig.engine.script(4, {0xC3, 0xA9, 'x', 0xE2});  // é, x, then a cut character
  const std::string want_e = std::string("\xC3\xA9");
  const std::string fffd = std::string("\xEF\xBF\xBD");
  {
    Client c(rig.port());
    const std::string body = chat_body("abcd", 4, ",\"stream\":true");
    c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    const std::string raw = c.read_until("data: [DONE]", 5000);
    // The split character arrives whole, never as a lone lead byte; the
    // cut one at the end arrives as U+FFFD; every content string is UTF-8.
    require(raw.find("\"content\":\"" + want_e + "\"") != std::string::npos,
            "the split character is delivered whole: " + raw.substr(0, 400));
    require(raw.find("\"content\":\"\xC3\"") == std::string::npos &&
                raw.find("\"content\":\"\xE2\"") == std::string::npos,
            "no delta ends inside a character");
    require(raw.find("\"content\":\"" + fffd + "\"") != std::string::npos,
            "the cap-cut character becomes U+FFFD: " + raw.substr(raw.size() > 500 ? raw.size() - 500 : 0));
    size_t at = 0;
    while ((at = raw.find("\"content\":\"", at)) != std::string::npos) {
      at += 11;
      const size_t end = raw.find('"', at);
      const std::string piece = raw.substr(at, end - at);
      // A strict check: every byte >= 0x80 sits inside a complete sequence.
      for (size_t i = 0; i < piece.size();) {
        const unsigned char b = static_cast<unsigned char>(piece[i]);
        const size_t len = b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 0;
        require(len > 0 && i + len <= piece.size(), "a delta that is not UTF-8: " + piece);
        for (size_t k = 1; k < len; ++k)
          require((static_cast<unsigned char>(piece[i + k]) & 0xC0) == 0x80, "a delta that is not UTF-8: " + piece);
        i += len;
      }
      at = end;
    }
  }
  // The one-shot: "éx" and the cut character as U+FFFD.
  const std::string resp = post_chat(rig, chat_body("abcd", 4), "usage", 5000);
  require(resp.find("\"content\":\"" + want_e + "x" + fffd + "\"") != std::string::npos,
          "the one-shot's content is UTF-8 with the cut character replaced: " + resp.substr(0, 400));
  // The legacy route streams the same discipline.
  {
    Client c(rig.port());
    const std::string body = "{\"model\":\"" + kModel + "\",\"prompt\":\"abcd\",\"max_tokens\":4,\"stream\":true}";
    c.send_all("POST /v1/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    const std::string raw = c.read_until("data: [DONE]", 5000);
    // (The legacy path coalesces the tokens a flush finds pending, so the
    // character may share its delta with the "x" that follows it.)
    require(raw.find("\"text\":\"" + want_e) != std::string::npos &&
                raw.find("\"text\":\"\xC3\"") == std::string::npos &&
                raw.find("\"text\":\"" + fffd + "\"") != std::string::npos,
            "the legacy stream holds the split character and replaces the cut one: " + raw.substr(0, 4000));
  }
}

DGPP_TEST(serve_prefixCache_boundariesAttachOptOutAndMetrics) {
  ServiceRig rig(/*queue_limit=*/8, dgpp::sample::greedy_params(),
                 /*can_sample=*/false, std::nullopt, /*with_markers=*/false,
                 /*reasoning_in_content=*/false, dgpp::sched::AdmissionPolicy{},
                 /*prefix_slots=*/4);
  const auto post = [&](const std::string& body) {
    Client c(rig.port());
    c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    return c.read_until("usage", 5000) + c.read_available(300);
  };
  // The message content field, JSON-escaped as the wire carries it.
  const auto content_of = [](const std::string& resp) {
    const size_t a = resp.find("\"content\":\"");
    if (a == std::string::npos) return std::string();
    const size_t b = resp.find("\"}", a);
    return b == std::string::npos ? std::string() : resp.substr(a, b - a);
  };
  const std::string first = post(chat_body("ab|cd|ef", 3));
  const std::string second = post(chat_body("ab|cd|ef", 3));
  require(first.find("200 OK") != std::string::npos && second.find("200 OK") != std::string::npos,
          "both requests answer: " + first.substr(0, 120));
  const std::string want = content_of(first);
  require(want.size() > 11 && content_of(second) == want,
          "identical answers, hot and cold: " + second.substr(0, 400));
  std::vector<std::string> ops = rig.engine.prefix_ops();
  // The first: an entry at cut 4, a rolling snapshot at 8 (the prompt end,
  // aligned), the close entry from it. The second: the attach at 4, its
  // own rolling at 8 (a duplicate of the close entry — released).
  std::string joined;
  for (const std::string& op : ops) joined += op + " ";
  require(joined.find("N:0:4 ") != std::string::npos && joined.find("X:0:4 ") != std::string::npos,
          "prefix ops: " + joined);

  const std::string out = post(chat_body("ab|cd|ef", 3, ",\"prefix_cache\":false"));
  require(out.find("200 OK") != std::string::npos && content_of(out) == want,
          "the opted-out request answers the same: " + out.substr(0, 400));
  require(rig.engine.prefix_ops().size() == ops.size(), "opted out: no prefix ops");
  const std::string bad = post(chat_body("ab|cd|ef", 3, ",\"prefix_cache\":3"));
  require(bad.find("400") != std::string::npos &&
              bad.find("\"param\":\"prefix_cache\"") != std::string::npos,
          "a non-boolean prefix_cache is refused by name: " + bad.substr(0, 300));

  Client metrics(rig.port());
  metrics.send_all("GET /v1/metrics HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string met = metrics.read_available(800);
  require(met.find("\"prefix_cache\":{\"enabled\":true,\"slots\":4") != std::string::npos &&
              met.find("\"hits\":1,\"misses\":1,\"tokens_saved\":4") != std::string::npos &&
              met.find("\"ttft_hit_count\":1") != std::string::npos &&
              met.find("\"ttft_miss_count\":2") != std::string::npos,
          "metrics: " + met);
}

DGPP_TEST(serve_throughputLog_reports_unfinished_prefill_work) {
  using dgpp::serve::ThroughputLog;
  const auto now = ThroughputLog::Clock::now();
  ThroughputLog log(1.0, 0);
  dgpp::sched::Scheduler::Meters m;
  m.prefix_slots = 4;
  require(log.observe(m, nullptr, now).empty(), "prime the interval");
  m.active = m.prefilling = 1;
  m.prompt_tokens = m.prompt_tokens_computed = 256;
  m.prefill_ms = 512;
  const auto line = log.observe(m, nullptr, now + std::chrono::seconds(1));
  require(line.find("prefill 0 prompts / 256 tok, 256 tok/s, 2.00 ms/tok (51 % of wall), 0 tok cached") !=
              std::string::npos && line.find("1 prefilling") != std::string::npos,
          "partial work has a rate without invented completed prompts or cache hits: " + line);
}

DGPP_TEST(serve_throughputLog_oneLinePerIntervalWithTheDeltas_thenQuiet) {
  // GIVEN a 10-second throughput line on rank 0 with an injected clock,
  using dgpp::serve::ServiceCounts;
  using dgpp::serve::ThroughputLog;
  using Clock = ThroughputLog::Clock;
  const std::string::size_type npos = std::string::npos;
  ThroughputLog log(/*interval_s=*/10.0, /*rank=*/0);
  const Clock::time_point t0 = Clock::now();
  const auto at = [&](double s) {
    return t0 + std::chrono::duration_cast<Clock::duration>(
                    std::chrono::duration<double>(s));
  };
  dgpp::sched::Scheduler::Meters m;
  m.pool_blocks_total = 100;
  m.prefix_slots = 4;
  ServiceCounts sc;

  // THEN the first call primes the baseline and says nothing, and nothing
  // inside the interval speaks either, whatever happened.
  require(log.observe(m, &sc, at(0)).empty(), "priming is silent");
  m.prompts_prefilled = 1;
  m.prompt_tokens = 120;
  m.prompt_tokens_computed = 100;
  m.prefill_ms = 500.0;
  m.decode_steps = 200;
  m.decode_rows = 400;
  m.tokens_generated = 700;
  m.step_ms = 8000.0;
  m.active = 2;
  m.queued = 1;
  m.pool_blocks_in_use = 25;
  m.prefix_entries = 3;
  m.prefix_hits = 1;
  sc.requests_total = 3;
  sc.requests_shed = 1;
  require(log.observe(m, &sc, at(5)).empty(), "no line inside the interval");

  // At the interval: one line with the deltas as rates over the elapsed time.
  const std::string line = log.observe(m, &sc, at(10));
  // Decode leads: the interval's tok/s, the pace while
  // decoding (8000 ms over 700 tokens), the step, the counts and share.
  require(line.find("stats: rank 0 | 10.0 s | decode 70.0 tok/s, 11.4 ms/tok, "
                    "40.0 ms/step (200 steps / 700 tok, 80 % of wall)") != npos,
          "the decode group: " + line);
  require(line.find("| mtp ") == npos, "no MTP group unless asked: " + line);
  require(line.find("| prefill 1 prompt / 100 tok, 10 tok/s, 5.00 ms/tok, "
                    "500 ms avg (5 % of wall), 20 tok cached (1/1 hit)") != npos,
          "the prefill group: " + line);
  require(line.find("| live 2, queued 1 | pool 25/100 blocks (25 %) | prefix "
                    "cache 3/4 entries | requests +3 (shed 1, cancelled 0)") !=
              npos,
          "the state tail: " + line);
  // With MTP the yield group follows decode: 700 tokens over 400 request
  // rows is 1.75 per row, and the engine's per-position counts give the
  // measured acceptance of each draft (here depth 2: 150/200 and 90/200).
  {
    ThroughputLog mlog(10.0, /*rank=*/0, /*mtp=*/true);
    dgpp::sched::Scheduler::Meters zero;
    require(mlog.observe(zero, &sc, at(0)).empty(), "priming is silent (mtp)");
    m.mtp.depth = 2;
    m.mtp.attempts[0] = 200;
    m.mtp.accepts[0] = 150;
    m.mtp.attempts[1] = 200;
    m.mtp.accepts[1] = 90;
    const std::string ml = mlog.observe(m, &sc, at(10));
    require(ml.find("| mtp 1.75 tok/step/req, accept p1 75 % p2 45 % | prefill") != npos,
            "the MTP group: " + ml);
    m.mtp = {};
  }

  // The interval after the work ends writes one closing line of zeros ...
  m.active = 0;
  m.queued = 0;
  const std::string closing = log.observe(m, &sc, at(20));
  require(closing.find("| prefill 0 prompts / 0 tok") != npos &&
              closing.find("(0 steps / 0 tok, 0 % of wall)") != npos &&
              closing.find("| live 0, queued 0 |") != npos,
          "the closing line: " + closing);
  // ... and an idle world then stays quiet.
  require(log.observe(m, &sc, at(30)).empty() &&
              log.observe(m, &sc, at(40)).empty(),
          "an idle world stays quiet");
  // Work returning brings the line back, over the interval's true length.
  m.decode_steps = 201;
  m.decode_rows = 401;
  m.tokens_generated = 702;
  const std::string back = log.observe(m, &sc, at(52));
  require(back.find("| 12.0 s | decode") != npos &&
              back.find("(1 step / 2 tok, ") != npos &&
              back.find("| prefill 0 prompts") != npos,
          "the line returns with work: " + back);

  // A peer's line (no service counts) names its rank and carries none.
  ThroughputLog peer(10.0, /*rank=*/2);
  require(peer.observe(m, nullptr, at(0)).empty(), "peer priming is silent");
  m.decode_steps = 202;
  const std::string pl = peer.observe(m, nullptr, at(10));
  require(pl.find("stats: rank 2 | 10.0 s |") != npos &&
              pl.find("requests +") == npos,
          "a peer's line: " + pl);

  // Interval 0 disables the line entirely.
  ThroughputLog off(0.0, /*rank=*/1);
  require(!off.enabled() && off.observe(m, nullptr, at(0)).empty() &&
              off.observe(m, nullptr, at(100)).empty(),
          "interval 0 disables the line");
}

// A raw POST to /v1/completions (the legacy route), read through `until`.
std::string post_legacy(ServiceRig& rig, const std::string& body,
                        const std::string& until, int budget_ms = 5000) {
  Client c(rig.port());
  c.send_all("POST /v1/completions HTTP/1.1\r\nHost: t\r\n"
             "Content-Type: application/json\r\nContent-Length: " +
             std::to_string(body.size()) + "\r\n\r\n" + body);
  return c.read_until(until, budget_ms);
}
// A one-shot chat read through the usage object's last key.
std::string post_full(ServiceRig& rig, const std::string& body) {
  return post_chat(rig, body, "reasoning_tokens\":", 5000);
}

DGPP_TEST(serve_stop_cutsTheVisibleTextAtTheMatchAndFinishesStop) {
  // GIVEN the fake's deterministic text for the 4-id prompt "abcd" and a
  // two-character stop string inside it (two one-character tokens: the
  // first is held until the second decides the match),
  ServiceRig rig;
  const std::string full = fake_text(4, 12);
  const std::string stop = full.substr(3, 2);
  const std::string prefix = full.substr(0, 3);
  // WHEN a one-shot carries it,
  const std::string one =
      post_full(rig, chat_body("abcd", 12, ",\"stop\":\"" + stop + "\""));
  // THEN the content ends before the match, finish_reason is stop, and the
  // usage counts the tokens through the one that completed the match.
  require(one.find("\"content\":\"" + prefix + "\"") != std::string::npos &&
              one.find("\"finish_reason\":\"stop\"") != std::string::npos &&
              one.find("\"completion_tokens\":5,") != std::string::npos,
          "one-shot stop: " + one);
  // Streamed: the deltas concatenate to the same prefix (no delta ever
  // shows the stop string), the final chunk says stop, the usage agrees.
  {
    Client c(rig.port());
    const std::string body =
        chat_body("abcd", 12,
                  ",\"stop\":[\"" + stop + "\"],\"stream\":true,"
                  "\"stream_options\":{\"include_usage\":true}");
    c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    const std::string resp = c.read_until("[DONE]", 5000);
    require(concat_field(resp, "content") == prefix &&
                resp.find("\"finish_reason\":\"stop\"") != std::string::npos &&
                resp.find("\"completion_tokens\":5,") != std::string::npos,
            "streamed stop: " + resp);
  }
  // The legacy route cuts its text the same way.
  {
    const std::string body = "{\"model\":\"" + kModel +
                             "\",\"prompt\":\"abcd\",\"max_tokens\":12,"
                             "\"stop\":\"" + stop + "\"}";
    const std::string resp = post_legacy(rig, body, "reasoning_tokens\":");
    require(resp.find("\"text\":\"" + prefix + "\"") != std::string::npos &&
                resp.find("\"finish_reason\":\"stop\"") != std::string::npos,
            "legacy stop: " + resp);
  }
  // A stop that never matches changes nothing: the whole text, length.
  {
    const std::string resp =
        post_full(rig, chat_body("abcd", 12, ",\"stop\":\"~~~~\""));
    require(resp.find("\"content\":\"" + full + "\"") != std::string::npos &&
                resp.find("\"finish_reason\":\"length\"") != std::string::npos,
            "no match: " + resp);
  }
  // A match at the very start: empty content, one token counted.
  {
    const std::string resp = post_full(
        rig, chat_body("abcd", 12, ",\"stop\":\"" + full.substr(0, 1) + "\""));
    require(resp.find("\"content\":\"\"") != std::string::npos &&
                resp.find("\"completion_tokens\":1,") != std::string::npos,
            "match at the start: " + resp);
  }
  // The shape is validated by name.
  for (const std::string bad :
       {",\"stop\":\"\"", ",\"stop\":[\"a\",\"b\",\"c\",\"d\",\"e\"]",
        ",\"stop\":5", ",\"stop\":[\"\"]"}) {
    const std::string resp = post_chat(rig, chat_body("abcd", 3, bad));
    require(resp.find("400") != std::string::npos &&
                resp.find("\"param\":\"stop\"") != std::string::npos,
            "refused by name: " + bad + " -> " + resp.substr(0, 200));
  }
}

DGPP_TEST(serve_n_answersEveryChoiceByIndexAndSumsTheUsage) {
  ServiceRig rig;
  const std::string text = fake_text(4, 3);
  // One-shot: two choices, both complete and indexed, the usage summed.
  const std::string one = post_full(rig, chat_body("abcd", 3, ",\"n\":2"));
  require(one.find("\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                   "\"content\":\"" + text + "\",\"refusal\":null}") != std::string::npos &&
              one.find("{\"index\":1,\"message\":{\"role\":\"assistant\","
                       "\"content\":\"" + text + "\",\"refusal\":null}") != std::string::npos &&
              one.find("\"prompt_tokens\":4,\"completion_tokens\":6,"
                       "\"total_tokens\":10") != std::string::npos,
          "n=2 one-shot: " + one);
  // Streamed: deltas for both indices, one finish per choice, one usage
  // chunk with the sum and one [DONE].
  {
    Client c(rig.port());
    const std::string body =
        chat_body("abcd", 3,
                  ",\"n\":2,\"stream\":true,"
                  "\"stream_options\":{\"include_usage\":true}");
    c.send_all("POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
               "Content-Type: application/json\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n" + body);
    const std::string resp = c.read_until("[DONE]", 5000);
    const auto count = [&](const std::string& s) {
      size_t n = 0, p = 0;
      while ((p = resp.find(s, p)) != std::string::npos) {
        ++n;
        p += s.size();
      }
      return n;
    };
    require(count("\"index\":1,\"delta\"") >= 3 && count("\"index\":0,\"delta\"") >= 3,
            "deltas for both choices: " + resp);
    require(count("\"finish_reason\":\"length\"") == 2 && count("[DONE]") == 1 &&
                count("\"usage\":{") == 1 &&
                resp.find("\"completion_tokens\":6,") != std::string::npos,
            "the end sequence once: " + resp);
  }
  // The bound: n outside [1, 128], or not an integer, is refused by name.
  for (const std::string bad : {",\"n\":0", ",\"n\":129", ",\"n\":1.5", ",\"n\":\"2\""}) {
    const std::string resp = post_chat(rig, chat_body("abcd", 3, bad));
    require(resp.find("400") != std::string::npos &&
                resp.find("\"param\":\"n\"") != std::string::npos,
            "n refused: " + bad + " -> " + resp.substr(0, 200));
  }
}

DGPP_TEST(serve_logitBias_reachesTheEngineAndIsValidatedByName) {
  ServiceRig rig(/*queue_limit=*/8, model_defaults(), /*can_sample=*/true);
  const std::string ok = post_full(
      rig, chat_body("abcd", 2,
                     ",\"logit_bias\":{\"5\":-100,\"7\":2.5},\"temperature\":0"));
  require(ok.find("200 OK") != std::string::npos, "accepted: " + ok.substr(0, 200));
  const auto armed = rig.engine.biases();
  require(armed.size() == 1 && armed[0].second.size() == 2 &&
              armed[0].second[0].token == 5 && armed[0].second[0].bias == -100.0f &&
              armed[0].second[1].token == 7 && armed[0].second[1].bias == 2.5f,
          "the entries reached the engine");
  // An empty object is no bias.
  (void)post_full(rig, chat_body("abcd", 2, ",\"logit_bias\":{}"));
  require(rig.engine.biases().size() == 1, "an empty logit_bias arms nothing");
  // Refused by name: a non-numeric key, a value outside [-100, 100], an id
  // outside the vocabulary, a non-object.
  for (const std::string bad :
       {",\"logit_bias\":{\"x\":1}", ",\"logit_bias\":{\"5\":101}",
        ",\"logit_bias\":{\"600\":1}", ",\"logit_bias\":[1,2]"}) {
    const std::string resp = post_chat(rig, chat_body("abcd", 2, bad));
    require(resp.find("400") != std::string::npos &&
                resp.find("\"param\":\"logit_bias\"") != std::string::npos,
            "refused: " + bad + " -> " + resp.substr(0, 300));
  }
  // A greedy-only engine refuses it with its own code.
  ServiceRig greedy;
  const std::string un =
      post_chat(greedy, chat_body("abcd", 2, ",\"logit_bias\":{\"5\":1}"));
  require(un.find("400") != std::string::npos &&
              un.find("\"code\":\"logit_bias_unsupported\"") != std::string::npos,
          "greedy-only: " + un.substr(0, 300));
}

DGPP_TEST(serve_usage_reportsCachedAndReasoningTokens) {
  // Reasoning: the ids up to and including </think>.
  ServiceRig think(/*queue_limit=*/8, dgpp::sample::greedy_params(),
                   /*can_sample=*/false, std::nullopt, /*with_markers=*/true);
  think.engine.script(5, script_of(think, "Th</think>Sure"));
  const std::string resp = post_full(think, chat_body("abcd", 64));
  require(resp.find("\"reasoning_content\":\"Th\"") != std::string::npos &&
              resp.find("\"completion_tokens_details\":{\"reasoning_tokens\":3") !=
                  std::string::npos &&
              resp.find("\"prompt_tokens_details\":{\"cached_tokens\":0}") !=
                  std::string::npos,
          "reasoning tokens: " + resp);
  // Cached: the prefix cache's attach position on the second identical
  // request (the sweep's cut at 4).
  ServiceRig cache(/*queue_limit=*/8, dgpp::sample::greedy_params(),
                   /*can_sample=*/false, std::nullopt, /*with_markers=*/false,
                   /*reasoning_in_content=*/false, dgpp::sched::AdmissionPolicy{},
                   /*prefix_slots=*/4);
  const std::string cold = post_full(cache, chat_body("ab|cd|ef", 3));
  const std::string hot = post_full(cache, chat_body("ab|cd|ef", 3));
  require(cold.find("\"cached_tokens\":0}") != std::string::npos &&
              hot.find("\"cached_tokens\":4}") != std::string::npos,
          "cached tokens: " + hot);
}

DGPP_TEST(serve_extensions_customToolsWireHistoryAndChoice) {
  ServiceRig rig(8, model_defaults(), true, std::nullopt, true);
  const std::string tools = R"(,"tools":[{"type":"custom","custom":{"name":"code_exec","description":"Run code","format":{"type":"text"}}},{"type":"function","function":{"name":"normal","parameters":{"type":"object","properties":{}}}}])";
  const std::string turn = "Think</think><tool_call>code_exec<arg_key>input</arg_key><arg_value>\"print(\\\"hi\\\")\\n\"</arg_value></tool_call>";
  rig.engine.script(5, script_of(rig, turn));
  const auto one = post_until_usage(rig, chat_body("abcd", 128, tools + R"(,"tool_choice":{"type":"custom","custom":{"name":"code_exec"}})"));
  require(one.find(R"("type":"custom","custom":{"name":"code_exec","input":"print(\"hi\")\n"})") != std::string::npos,
          "custom one-shot returns input, not JSON arguments: " + one);
  require(one.find(R"("finish_reason":"tool_calls")") != std::string::npos, "custom finish reason");
  const auto stream = post_chat(rig, chat_body("abcd", 128, tools + R"(,"stream":true,"tool_choice":{"type":"allowed_tools","allowed_tools":{"mode":"required","tools":[{"type":"custom","custom":{"name":"code_exec"}}]}})"), "[DONE]", 5000);
  require(stream.find(R"("type":"custom","custom":{"name":"code_exec","input":""})") != std::string::npos,
          "custom stream announces type/name/input: " + stream);
  require(stream.find(R"("custom":{"input":"print(\"hi\")\n"})") != std::string::npos,
          "custom input deltas reconstruct exact free-form text: " + stream);
  require(rig.engine.grammars().back().tools.size() == 1, "allowed_tools filters mixed tool types");
  const std::string history = "{\"model\":\"" + kModel + R"JSON(","messages":[{"role":"user","content":"abcd"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"custom","custom":{"name":"code_exec","input":"print(1)"}}]},{"role":"tool","tool_call_id":"call_1","content":"1"}],"max_tokens":2)JSON" + tools + "}";
  const auto response = post_until_usage(rig, history);
  require(response.find("200 OK") != std::string::npos, "custom call history accepted: " + response);
  require(rig.frontend.last_globals().find(R"JSON("arguments":{"input":"print(1)"})JSON") != std::string::npos,
          "custom history translated into model tool format");
  expect_invalid_request(post_chat(rig, chat_body("abcd", 2, tools + R"(,"tool_choice":{"type":"function","function":{"name":"code_exec"}})")), "tool_choice.type");
}

DGPP_TEST(serve_extensions_reasoningMapsAndConflicts) {
  ServiceRig rig(8, dgpp::sample::greedy_params(), false, std::nullopt, true);
  rig.frontend.reads_enable_thinking = true;
  auto response = post_until_usage(rig, chat_body("abcd", 2, R"(,"reasoning_effort":"none")"));
  require(response.find("200 OK") != std::string::npos, "none accepted for switchable model");
  require(rig.frontend.last_globals().find(R"("enable_thinking":false)") != std::string::npos &&
          rig.frontend.last_globals().find("reasoning_effort") == std::string::npos, "none disables native thinking switch");
  response = post_until_usage(rig, chat_body("abcd", 2, R"(,"reasoning_effort":"max")"));
  require(response.find("200 OK") != std::string::npos && rig.frontend.last_globals().find(R"("reasoning_effort":"max")") != std::string::npos,
          "native max reaches model");
  expect_invalid_request(post_chat(rig, chat_body("abcd", 2, R"(,"reasoning_effort":"none","chat_template_kwargs":{"thinking":true})")), "chat_template_kwargs.enable_thinking");
  expect_invalid_request(post_chat(rig, chat_body("abcd", 2, R"(,"reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":false})")), "chat_template_kwargs.enable_thinking");
  rig.frontend.reads_enable_thinking = false;
  expect_invalid_request(post_chat(rig, chat_body("abcd", 2, R"(,"reasoning_effort":"none")")), "reasoning_effort");
}

DGPP_TEST(serve_extensions_fileUploadsInlineReferencesAndDeletion) {
  ServiceRig rig;
  const auto http = [&](const std::string& method, const std::string& path,
                        const std::string& body = "", const std::string& type = "application/json") {
    Client c(rig.port());
    c.send_all(method + " " + path + " HTTP/1.1\r\nHost: t\r\nContent-Type: " + type +
      "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body);
    return c.read_available(800);
  };
  const std::string inline_body = "{\"model\":\"" + kModel + R"(","messages":[{"role":"user","content":[{"type":"text","text":"Read this"},{"type":"file","file":{"filename":"input.txt","file_data":"aGVsbG8="}}]}],"max_tokens":2})";
  auto answer = post_until_usage(rig, inline_body);
  require(answer.find("200 OK") != std::string::npos, "inline file accepted: " + answer);
  require(rig.frontend.last_globals().find("hello") != std::string::npos &&
          rig.frontend.last_globals().find("aGVsbG8=") == std::string::npos, "decoded text reaches prompt");
  const std::string upload = "--BOUND\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nuser_data\r\n--BOUND\r\nContent-Disposition: form-data; name=\"file\"; filename=\"data.txt\"\r\nContent-Type: text/plain\r\n\r\nhello\r\n--BOUND--\r\n";
  const auto created = http("POST", "/v1/files", upload, "multipart/form-data; boundary=BOUND");
  require(created.find("200 OK") != std::string::npos, "upload accepted: " + created);
  const auto meta = dgpp::minijson::parse(std::string_view(created).substr(created.find("\r\n\r\n")+4));
  const std::string id(meta.root.at("id").as_string());
  const auto reference = "{\"model\":\"" + kModel + "\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"file\",\"file\":{\"file_id\":\"" + id + "\"}}]}],\"max_tokens\":2}";
  const auto referenced = post_until_usage(rig, reference);
  require(referenced.find("200 OK") != std::string::npos, "file_id input accepted: " + referenced);
  require(http("GET", "/v1/files/"+id+"/content").ends_with("hello"), "download returns original file");
  require(http("GET", "/v1/files").find(id) != std::string::npos, "list includes upload");
  const auto deleted = http("DELETE", "/v1/files/"+id);
  const auto deletion = dgpp::minijson::parse(std::string_view(deleted).substr(deleted.find("\r\n\r\n")+4));
  require(deletion.root.at("deleted").as_bool(), "delete file: " + deleted);
  require(post_chat(rig, reference).find("404 Not Found") != std::string::npos, "deleted reference fails");
  std::string bad = inline_body;
  bad.replace(bad.find("aGVsbG8="), 8, "aGVsbG9=");
  require(post_chat(rig, bad).find("400 Bad Request") != std::string::npos, "invalid base64 padding rejected");
}

DGPP_TEST(serve_extensions_pdfWorkersKeepHealthResponsiveAndBounded) {
  std::string directory = "/tmp/dgpp-pdf-worker-test-XXXXXX";
  require(mkdtemp(directory.data()) != nullptr, "temporary extractor directory");
  struct Cleanup { std::string path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{directory};
  const std::string command = directory + "/extract";
  { std::ofstream out(command); out << "#!/bin/sh\nexec /bin/sleep 10\n"; }
  require(chmod(command.c_str(), 0700) == 0, "executable test extractor");
  dgpp::serve::FileInputConfig config;
  config.pdf_command = command; config.pdf_timeout_ms = 1000; config.workers = 1;
  ServiceRig rig(8, dgpp::sample::greedy_params(), false, std::nullopt, false, false, {}, 0, config);
  const std::string body = "{\"model\":\"" + kModel + R"(","messages":[{"role":"user","content":[{"type":"file","file":{"filename":"x.pdf","file_data":"JVBERi0="}}]}],"max_tokens":2})";
  const auto request = "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
  Client document(rig.port()); document.send_all(request);
  require(document.read_available(100).empty(), "extractor still processing");
  Client probe(rig.port()); probe.send_all("GET /health HTTP/1.1\r\nHost: t\r\n\r\n");
  require(probe.read_until("\"status\":\"ok\"", 250).find("200 OK") != std::string::npos,
          "health responds during PDF extraction");
  Client busy(rig.port()); busy.send_all(request);
  require(busy.read_until("file_processing_busy", 250).find("503 Service Unavailable") != std::string::npos,
          "configured worker cap bounds processing");
  require(document.read_until("PDF extraction timed out", 2000).find("408 Request Timeout") != std::string::npos,
          "extractor deadline returns an API error and reaps the process");
  Client interrupted(rig.port()); interrupted.send_all(request);
  require(interrupted.read_available(100).empty(), "next extractor processing");
  rig.drain(false);
  require(interrupted.read_until("server_shutdown", 500).find("503 Service Unavailable") != std::string::npos,
          "shutdown stops extraction and drains its error response");
  require(rig.service.drained(), "file preprocessing participates in shutdown drain");
}

DGPP_TEST(serve_edge_customToolValidationAndHistory) {
  ServiceRig rig(8, model_defaults(), true, std::nullopt, true);
  const std::string prefix = R"(,"tools":[{"type":"custom","custom":)";
  for (const auto& [custom, param] : std::vector<std::pair<std::string,std::string>>{
      {"null", "tools[0].custom"}, {"{}", "tools[0].custom.name"},
      {R"({"name":"bad name"})", "tools[0].custom.name"},
      {R"({"name":"exec","description":false})", "tools[0].custom.description"},
      {R"({"name":"exec","format":null})", "tools[0].custom.format.type"},
      {R"({"name":"exec","format":{"type":"json"}})", "tools[0].custom.format.type"},
      {R"({"name":"exec","format":{"type":"grammar"}})", "tools[0].custom.format.grammar"},
      {R"({"name":"exec","format":{"type":"grammar","grammar":{"syntax":"regex","definition":"["}}})", "tools[0].custom.format.grammar"},
      {R"({"name":"exec","format":{"type":"grammar","grammar":{"syntax":"lark","definition":"start: a\na: start"}}})", "tools[0].custom.format.grammar"},
      {R"({"name":"exec","format":{"type":"grammar","grammar":{"syntax":"unknown","definition":"x"}}})", "tools[0].custom.format.grammar"}}) {
    expect_invalid_request(post_chat(rig, chat_body("abcd", 2, prefix + custom + "}]"), "\"param\""), param);
  }
  const std::string tools = R"(,"tools":[{"type":"custom","custom":{"name":"exec"}}])";
  for (const auto& [choice, param] : std::vector<std::pair<std::string,std::string>>{
      {R"({"type":"custom","custom":{}})", "tool_choice.custom.name"},
      {R"({"type":"custom","custom":{"name":"missing"}})", "tool_choice.custom.name"},
      {R"({"type":"function","function":{"name":"exec"}})", "tool_choice.type"},
      {R"({"type":"allowed_tools","allowed_tools":{"mode":"required","tools":[{"type":"custom","custom":{"name":"exec"}},{"type":"custom","custom":{"name":"exec"}}]}})", "tool_choice.allowed_tools.tools"}}) {
    expect_invalid_request(post_chat(rig, chat_body("abcd", 2, tools + ",\"tool_choice\":" + choice), "\"param\""), param);
  }
  const std::string history_prefix = "{\"model\":\"" + kModel + R"(","messages":[{"role":"assistant","tool_calls":[)";
  for (const auto& [call, param] : std::vector<std::pair<std::string,std::string>>{
      {R"({"type":"custom","custom":{"name":"exec","input":""}})", "messages[0].tool_calls[0].id"},
      {R"({"id":"call_1","type":"custom","custom":{"name":"exec","input":{}}})", "messages[0].tool_calls[0].custom"}}) {
    expect_invalid_request(post_chat(rig, history_prefix + call + "]}]}", "\"param\""), param);
  }
  const auto accepted = post_until_usage(rig, history_prefix + R"({"id":"call_1","type":"custom","custom":{"name":"exec","input":""}})" + "]}],\"max_tokens\":2}");
  require(accepted.find("200 OK") != std::string::npos, "empty custom history input is a valid string");
}

DGPP_TEST(serve_edge_reasoningAliasesAndSwitchOnlyModels) {
  ServiceRig rig(8, dgpp::sample::greedy_params(), false, std::nullopt, true);
  rig.frontend.reads_enable_thinking = true;
  for (const auto* value : {"false", "1", "[]", "\"extreme\""})
    expect_invalid_request(post_chat(rig, chat_body("abcd", 2, std::string(",\"reasoning_effort\":") + value), "\"param\""), "reasoning_effort");
  expect_invalid_request(post_chat(rig, chat_body("abcd", 2, R"(,"reasoning_effort":"low","chat_template_kwargs":{"reasoning_effort":"high"})"), "\"param\""), "chat_template_kwargs.reasoning_effort");
  expect_invalid_request(post_chat(rig, chat_body("abcd", 2, R"(,"chat_template_kwargs":{"thinking":true,"enable_thinking":false})"), "\"param\""), "chat_template_kwargs.thinking");
  expect_invalid_request(post_chat(rig, chat_body("abcd", 2, R"(,"chat_template_kwargs":{"enable_thinking":true,"enable_thinking":false})"), "\"param\""), "chat_template_kwargs.enable_thinking");
  const auto response = post_until_usage(rig, chat_body("abcd", 2, R"(,"reasoning_effort":"none","chat_template_kwargs":{"thinking":false,"enable_thinking":false})"));
  require(response.find("200 OK") != std::string::npos, "agreeing reasoning aliases accepted");
  const auto globals_text = rig.frontend.last_globals();
  const auto globals = dgpp::minijson::parse(globals_text);
  int switches = 0;
  for (const auto& m : globals.root.members()) switches += m.key == "enable_thinking";
  require(switches == 1, "thinking aliases produce one unambiguous native global");
  rig.frontend.reads_reasoning_effort = false;
  for (const auto* effort : {"minimal","low","medium","high","xhigh","max"}) {
    const auto result = post_until_usage(rig, chat_body("abcd", 2, std::string(R"(,"reasoning_effort":")") + effort + "\""));
    require(result.find("200 OK") != std::string::npos, "switch-only model accepts positive effort");
    require(rig.frontend.last_globals().find(R"("enable_thinking":true)") != std::string::npos &&
            rig.frontend.last_globals().find("reasoning_effort") == std::string::npos, "no invented effort level on switch-only model");
  }
  rig.frontend.reads_enable_thinking = false;
  expect_invalid_request(post_chat(rig, chat_body("abcd", 2, R"(,"reasoning_effort":"low")"), "\"param\""), "reasoning_effort");
}

DGPP_TEST(serve_edge_filePreprocessingPipeliningAndDisconnect) {
  ServiceRig rig;
  const std::string body = "{\"model\":\"" + kModel + R"(","messages":[{"role":"user","content":[{"type":"\u0066ile","\u0066ile":{"\u0066ilename":"x.txt","\u0066ile_data":"aGVsbG8="}}]}],"max_tokens":2})";
  const std::string request = "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
  Client pipeline(rig.port());
  pipeline.send_all(request + "GET /health HTTP/1.1\r\nHost: t\r\n\r\n");
  const auto responses = pipeline.read_until(R"("status":"ok")", 3000);
  require(responses.find("chat.completion") < responses.find(R"("status":"ok")"), "file chat response precedes pipelined health response");
  require(responses.find("400 Bad Request") == std::string::npos && rig.frontend.last_globals().find("hello") != std::string::npos,
          "escaped file part reaches ordinary completion pipeline");
  for (int i = 0; i < 12; ++i) { Client gone(rig.port()); gone.send_all(request); gone.hard_close(); }
  Client health(rig.port()); health.send_all("GET /health HTTP/1.1\r\nHost: t\r\n\r\n");
  require(health.read_until(R"("status":"ok")", 3000).find("200 OK") != std::string::npos,
          "disconnect during preprocessing never leaves stale response writers");
}

DGPP_TEST(serve_edge_customToolsMultipleChoicesAndExactStreamingInput) {
  ServiceRig rig(8, model_defaults(), true, std::nullopt, true);
  const std::string input = "\nprint(\"日本語😀\")\\tail\t";
  const std::string encoded = dgpp::text::Value::string_value(input).to_json(false);
  const std::string turn = "ok</think><tool_call>exec<arg_key>input</arg_key><arg_value>" + encoded +
      "</arg_value></tool_call><tool_call>normal<arg_key>value</arg_key><arg_value>7</arg_value></tool_call>";
  rig.engine.script(5, script_of(rig, turn));
  const std::string tools = R"(,"n":2,"tools":[{"type":"custom","custom":{"name":"exec"}},{"type":"function","function":{"name":"normal","parameters":{"type":"object","properties":{"value":{"type":"integer"}}}}}])";
  const auto response = post_until_usage(rig, chat_body("abcd", 128, tools));
  const auto parsed = dgpp::minijson::parse(std::string_view(response).substr(response.find("\r\n\r\n")+4));
  const auto& choices = parsed.root.at("choices").items();
  require(choices.size() == 2, "two choices returned");
  for (const auto& choice : choices) {
    const auto& calls = choice.at("message").at("tool_calls").items();
    require(calls.size() == 2 && calls[0].at("type").as_string() == "custom" && calls[1].at("type").as_string() == "function",
            "mixed tool types retain independent wire shapes");
    require(calls[0].at("custom").at("input").as_string() == input, "one-shot custom input preserves exact Unicode and whitespace");
    require(choice.at("finish_reason").as_string() == "tool_calls", "both choices finish as tool calls");
  }
  require(choices[0].at("message").at("tool_calls").items()[0].at("id").as_string() !=
          choices[1].at("message").at("tool_calls").items()[0].at("id").as_string(), "call IDs differ across choices");
  const auto stream = post_chat(rig, chat_body("abcd", 128, tools + R"(,"stream":true,"stream_options":{"include_usage":true})"), "[DONE]", 5000);
  std::map<int, std::string> inputs;
  std::map<int, int> starts, finishes;
  int usage_chunks = 0;
  for (size_t at = 0; (at = stream.find("data: ", at)) != std::string::npos;) {
    at += 6;
    const auto end = stream.find('\n', at);
    const auto payload = std::string_view(stream).substr(at, end-at);
    if (payload == "[DONE]") break;
    const auto event = dgpp::minijson::parse(payload);
    if (const auto* usage = event.root.find("usage"); usage && usage->is_object()) ++usage_chunks;
    for (const auto& choice : event.root.at("choices").items()) {
      const int index = choice.at("index").as_int();
      if (const auto* finish = choice.find("finish_reason"); finish && finish->as_string() == "tool_calls") ++finishes[index];
      const auto* calls = choice.at("delta").find("tool_calls");
      if (!calls) continue;
      for (const auto& call : calls->items()) {
        if (const auto* custom = call.find("custom")) {
          require(call.at("index").as_int() == 0 && !call.find("function"), "custom delta preserves tool index and type");
          if (custom->find("name")) ++starts[index];
          if (const auto* delta = custom->find("input")) inputs[index] += delta->as_string();
        }
      }
    }
  }
  require(inputs.size() == 2 && inputs[0] == input && inputs[1] == input && starts[0] == 1 && starts[1] == 1 &&
          finishes[0] == 1 && finishes[1] == 1 && usage_chunks == 1, "SSE reconstructs both custom inputs, one finish per choice, one final usage");
}

DGPP_TEST(serve_edge_malformedModelCustomOutputTerminatesWithError) {
  ServiceRig rig(8, model_defaults(), true, std::nullopt, true);
  rig.engine.script(5, script_of(rig, "ok</think><tool_call>exec<arg_key>wrong</arg_key><arg_value>oops</arg_value></tool_call>"));
  const std::string tools = R"(,"tools":[{"type":"custom","custom":{"name":"exec"}}])";
  const auto once = post_chat(rig, chat_body("abcd", 128, tools), "invalid_tool_output", 3000);
  require(once.find("server_error") != std::string::npos && once.find("200 OK") == std::string::npos,
          "malformed custom result cannot become a successful completion");
  const auto stream = post_chat(rig, chat_body("abcd", 128, tools + R"(,"stream":true)"), "[DONE]", 3000);
  require(stream.find("invalid_tool_output") != std::string::npos && stream.find("[DONE]") != std::string::npos &&
          stream.find(R"("finish_reason":"tool_calls")") == std::string::npos, "stream emits terminal error and DONE");
}

int main() {
  dgpp::set_log_level_from_env("DGPP_LOG_LEVEL");
  return dgpp::test::run_all();
}

DGPP_TEST(serve_images_capability_validation_and_streaming) {
  ServiceRig rig;
  const std::string png =
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
      "x8AAwMCAO+a0ioAAAAASUVORK5CYII=";
  const auto body = [&](const std::string& url, bool stream) {
    return "{\"model\":\"glm-5.3-flash-fp8\",\"max_tokens\":1,\"stream\":" +
           std::string(stream ? "true" : "false") +
           ",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":"
           "\"describe\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"" +
           url + "\"}}]}]}";
  };
  auto response = post_chat(rig, body(png, false), "unsupported_content_type");
  require(response.find("400 Bad Request") != std::string::npos,
          "text-only frontend refuses images");
  rig.frontend.images_available = true;
  response = post_chat(rig, body(png, false), "unsupported_content_type");
  require(response.find("400 Bad Request") != std::string::npos, "text-only engine refuses images");
  rig.engine.images_available = true;
  response = post_chat(rig, body(png, false), "usage");
  require(response.find("200 OK") != std::string::npos &&
              response.find("\"prompt_tokens\":24") != std::string::npos,
          "image expansion counted in usage: " + response);
  response = post_chat(rig, body(png, true), "[DONE]");
  require(
      response.find("200 OK") != std::string::npos && response.find("[DONE]") != std::string::npos,
      "image SSE completes");
  require(rig.engine.image_prefills == 2, "image payload reached both prefills");
  response = post_chat(rig, body("data:image/png;base64,AAAA", false), "invalid_image");
  require(response.find("messages[0].content[1].image_url.url") != std::string::npos,
          "invalid image names the precise field");
}

DGPP_TEST(serve_models_reports_the_active_rope_and_the_effective_limit) {
  // The request-context surface (review item 7, 2026-09-18): what ramp is in
  // force, and what one request may actually reach. Both fields are additive —
  // a build with the knob off, or a rig that names neither bound, sends
  // exactly the model object it sent before.
  {
    ServiceRig rig;
    Client models(rig.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string ml = models.read_available(800);
    require(ml.find("\"rope_scaling\"") == std::string::npos &&
                ml.find("\"context\"") == std::string::npos,
            "nothing advertised when nothing is set: " + ml.substr(0, 600));
  }
  // The 512K recipe: YaRN x2 over 262 144 positions, in a 1 048 576-token
  // pool. The ramp rides the deployment's own field names with the values it
  // derives, and the limit is the MINIMUM of the two bounds — the ceiling
  // here, because the pool clears it.
  {
    ServiceRig rig(8, dgpp::sample::greedy_params(), false, std::nullopt, false, false, {}, 0, {},
                   dgpp::RopeScaling{2.0, 262144, 32.0, 1.0, 1.0, 4.0}, 524288, 1048576);
    Client models(rig.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string ml = models.read_available(800);
    require(ml.find("\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":2") != std::string::npos &&
                ml.find("\"original_max_position_embeddings\":262144") != std::string::npos &&
                ml.find("\"scaled_max_position_embeddings\":524288") != std::string::npos,
            "the active ramp: " + ml);
    require(ml.find("\"context\":{\"request_limit_tokens\":524288,"
                    "\"position_ceiling_tokens\":524288,\"kv_pool_tokens\":1048576,"
                    "\"limited_by\":\"position-ceiling\"}") != std::string::npos,
            "the ceiling is the smaller bound: " + ml);
  }
  // The other half of the sentence: a pool SHORTER than the ceiling (an
  // engine.kv_capacity below the YaRN limit) is what bounds the request, and
  // the response says so — and with the knob unset the ramp stays off the
  // wire, so a client can never read a YaRN it is not getting.
  {
    ServiceRig rig(8, dgpp::sample::greedy_params(), false, std::nullopt, false, false, {}, 0, {},
                   std::nullopt, 524288, 262144);
    Client models(rig.port());
    models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
    const std::string ml = models.read_available(800);
    require(ml.find("\"rope_scaling\"") == std::string::npos, "plain rope: " + ml);
    require(ml.find("\"request_limit_tokens\":262144") != std::string::npos &&
                ml.find("\"limited_by\":\"kv-pool\"") != std::string::npos,
            "the pool is the smaller bound: " + ml);
  }
}

DGPP_TEST(serve_images_prefix_cache_defaults_on_and_respects_opt_out) {
  ServiceRig rig(8, dgpp::sample::greedy_params(), false, std::nullopt, false, false, {}, 4);
  rig.frontend.images_available = true;
  rig.engine.images_available = true;
  const std::string body =
      R"({"model":"glm-5.3-flash-fp8","max_tokens":1,"messages":[{"role":"user","content":[)"
      R"({"type":"text","text":"ab|cd|ef"},{"type":"image_url","image_url":{"url":)"
      R"("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a0ioAAAAASUVORK5CYII="}}]}]})";
  const auto first = post_chat(rig, body, "usage");
  const auto second = post_chat(rig, body, "usage");
  require(first.find("200 OK") != std::string::npos && second.find("\"cached_tokens\":4") != std::string::npos,
          "image request uses its available prefix by default: " + second);
  const auto operations = rig.engine.prefix_ops();
  const auto disabled = body.substr(0, body.size() - 1) + ",\"prefix_cache\":false}";
  const auto uncached = post_chat(rig, disabled, "usage");
  require(uncached.find("\"cached_tokens\":0") != std::string::npos &&
              rig.engine.prefix_ops() == operations && rig.engine.image_prefills == 3,
          "explicit opt-out still performs image prefill without cache operations");

}

DGPP_TEST(serve_servedModelName_aliasServesBesideTheCheckpointId) {
  // GIVEN a service that serves the checkpoint under a stable alias (the
  // cluster config's served_model_name: the A/B lanes' gateway name),
  ServiceRig rig(8, dgpp::sample::greedy_params(), false, std::nullopt, false, false, {}, 0, {},
                 std::nullopt, 0, 0, "stable-alias");

  // WHEN a client names the alias, the checkpoint id, or an unknown model,
  // THEN the alias and the checkpoint id are both served, and the unknown
  // model is 404 with the error naming the alias (the name actually served).
  const std::string alias_body =
      "{\"model\":\"stable-alias\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"abcd\"}],\"max_tokens\":2}";
  const std::string alias_resp = post_until_usage(rig, alias_body);
  require(alias_resp.find("200 OK") != std::string::npos,
          "the alias is served: " + alias_resp.substr(0, 200));
  const std::string ckpt_resp = post_until_usage(rig, chat_body("abcd", 2));
  require(ckpt_resp.find("200 OK") != std::string::npos,
          "the checkpoint id is still served beside the alias: " + ckpt_resp.substr(0, 200));
  const std::string wrong_body =
      "{\"model\":\"nope\",\"messages\":[{\"role\":\"user\","
      "\"content\":\"abcd\"}],\"max_tokens\":2}";
  const std::string wrong_resp = post_chat(rig, wrong_body);
  require(wrong_resp.find("404 Not Found") != std::string::npos &&
              wrong_resp.find("\"code\":\"model_not_found\"") != std::string::npos &&
              wrong_resp.find("(serving 'stable-alias')") != std::string::npos,
          "the unknown model is 404 naming the alias: " + wrong_resp.substr(0, 300));

  // The legacy prompt API accepts the alias too.
  Client legacy(rig.port());
  const std::string legacy_body =
      "{\"model\":\"stable-alias\",\"prompt\":\"hello\",\"max_tokens\":2}";
  legacy.send_all("POST /v1/completions HTTP/1.1\r\nHost: t\r\n"
                  "Content-Type: application/json\r\nContent-Length: " +
                  std::to_string(legacy_body.size()) + "\r\n\r\n" + legacy_body);
  const std::string legacy_resp = legacy.read_until("usage", 5000);
  require(legacy_resp.find("200 OK") != std::string::npos &&
              legacy_resp.find("\"object\":\"text_completion\"") != std::string::npos,
          "the legacy API serves the alias: " + legacy_resp.substr(0, 200));

  // /v1/models reports the alias as the id (and never the checkpoint id),
  // /v1/models/<id> accepts the alias or the checkpoint id, and an unknown
  // id is 404.
  Client models(rig.port());
  models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string ml = models.read_available(800);
  require(ml.find("\"id\":\"stable-alias\"") != std::string::npos &&
              ml.find(kModel) == std::string::npos,
          "the list reports the alias as the id: " + ml.substr(0, 300));
  models.send_all("GET /v1/models/stable-alias HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string alias_get = models.read_available(800);
  require(alias_get.find("200 OK") != std::string::npos &&
              alias_get.find("\"id\":\"stable-alias\"") != std::string::npos,
          "/v1/models/<alias> is served: " + alias_get.substr(0, 300));
  models.send_all("GET /v1/models/" + kModel + " HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string ckpt_get = models.read_available(800);
  require(ckpt_get.find("200 OK") != std::string::npos &&
              ckpt_get.find("\"id\":\"stable-alias\"") != std::string::npos,
          "/v1/models/<checkpoint id> is served too: " + ckpt_get.substr(0, 300));
  models.send_all("GET /v1/models/nope HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string none_get = models.read_available(800);
  require(none_get.find("404 Not Found") != std::string::npos &&
              none_get.find("\"code\":\"model_not_found\"") != std::string::npos,
          "/v1/models/<unknown> is 404: " + none_get.substr(0, 300));

  // The non-regression half: without an alias, the checkpoint id alone is
  // served and reported — exactly today's behavior.
  ServiceRig plain;
  Client plain_models(plain.port());
  plain_models.send_all("GET /v1/models HTTP/1.1\r\nHost: t\r\n\r\n");
  const std::string plain_ml = plain_models.read_available(800);
  require(plain_ml.find("\"id\":\"" + kModel + "\"") != std::string::npos &&
              plain_ml.find("stable-alias") == std::string::npos,
          "absent alias: the checkpoint id is reported: " + plain_ml.substr(0, 300));
  const std::string plain_wrong = post_chat(plain, wrong_body);
  require(plain_wrong.find("404 Not Found") != std::string::npos &&
              plain_wrong.find("\"code\":\"model_not_found\"") != std::string::npos,
          "absent alias: the unknown model is 404: " + plain_wrong.substr(0, 300));
}
