#pragma once
#include <future>
#include <optional>
#include "serve/file_inputs.hpp"

#include "kernels/rope_scaling.hpp"  // ServiceConfig's rope surface (what /v1/models reports)
// OpenAI-compatible text-generation service over the scheduler.
// Supported fields are validated before admission. Known unsupported API
// features return 400 naming the parameter; unknown top-level client
// extensions are ignored for compatibility. Sampling defaults
// come from the model configuration, and constrained output requires
// an engine that supports token masks.
//

//   POST /v1/chat/completions   messages[] (developer/system/user/assistant/tool;
//                               string or content-part content, assistant
//                               tool_calls and reasoning_content, tool
//                               tool_call_id), max_tokens or
//                               max_completion_tokens, stream,
//                               stream_options.include_usage/include_obfuscation; sampling:
//                               temperature, top_p, presence_penalty,
//                               frequency_penalty, seed (the OpenAI
//                               fields) plus top_k, min_p and
//                               repetition_penalty (HF extensions) —
//                               every omitted field takes the model's
//                               generation_config.json default (M6 6b,
//                               DESIGN §10); temperature 0 is the exact
//                               greedy path. An engine that cannot
//                               sample yet serves greedy defaults and
//                               refuses temperature > 0 with
//                               sampling_unsupported. logprobs (bool) and
//                               top_logprobs (0..20) report visible content
//                               tokens' log-probabilities and
//                               alternatives (the OpenAI content shape),
//                               exact from the same sampler; temperature
//                               0 reports under the raw distribution.
//                               Tools (M6 6f, DESIGN §11): tools,
//                               tool_choice (auto / none / required /
//                               {function: name}), parallel_tool_calls,
//                               reasoning_effort and
//                               chat_template_kwargs render through the
//                               checkpoint's template; the response
//                               carries reasoning_content, content and
//                               tool_calls parsed from the token ids,
//                               finish_reason "tool_calls" when a call
//                               parsed. tool_choice required / named /
//                               none and parallel_tool_calls false are
//                               guarantees through constrained decoding
//                               (M6 6g, DESIGN §10): the request carries
//                               the tool-call grammar and every rank's
//                               pick obeys its mask — the model reasons
//                               first, then can only write a valid call
//                               to an allowed function (or none, or one).
//                               An engine without masks refuses those
//                               fields (constrained_decoding_unsupported).
//   POST /v1/completions        the legacy prompt API (string prompt).
//   GET  /v1/models, /v1/models/{id}   the model object with the sampling
//                               defaults and, additively, the rope ramp in
//                               force and the effective request context
//                               limit (the lesser of the positional ceiling
//                               and the K/V pool).
//   GET  /health               liveness (the fabric harnesses' probe).
//   GET  /metrics, /v1/metrics  JSON counters and live prefill progress (ours).
// See docs/openai-compatibility.md for the complete capability profile.
//
// Thread ownership (HTTP and engine threads, one shared mutex):
//   * HTTP thread — HttpServer::serve() calls handle()/idle()/
//     on_disconnect(). Parses, validates, tokenizes, renders the chat
//     template (rank 0 only — never on the fabric critical path),
//     creates request records, and formats SSE chunks from the rings.
//   * Engine thread — the app calls engine_pass() in a loop: it drains
//     the admission/cancel queue into the scheduler and runs one
//     scheduler quantum (sched.tick()). The scheduler observer (which
//     is this service) appends token/retire events to the request
//     records under the lock; the chat records' tool-call parser runs
//     there too (pure host work, rank 0 only — the fabric never sees it).
//   * The single mutex covers the event queue and the record list;
//     sockets are only ever written by the HTTP thread (idle()).
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "sched/scheduler.hpp"
#include "text/tool_parser.hpp"
#include "serve/http_server.hpp"

namespace dgpp::serve {

// The tokenizer + chat-template interface. The real binding (TextFrontend)
// wraps the Stage 3/3b exact tokenizer and template; tests bind a fake
// so the whole HTTP/SSE/lifecycle stack runs without a model cache.
class ModelFrontend {
 public:
  struct ReasoningSettings {
    std::optional<bool> enable_thinking;
    std::optional<std::string> effort;
  };
  virtual ~ModelFrontend() = default;
  // text -> token ids (the exact ByteLevel-BPE encode).
  virtual std::vector<int64_t> encode_text(std::string_view text) const = 0;
  // ids -> text with special tokens skipped (the stream's content
  // semantics; decode(ids) == decode(ids[0..n-1]) + suffix deltas, so
  // incremental text is the suffix diff of successive full decodes).
  virtual std::string decode_ids(const std::vector<int64_t>& ids) const = 0;
  // The template globals (minijson DOM object: "messages" — the OpenAI
  // messages array, normalized — plus optional "tools",
  // "reasoning_effort", "clear_thinking" and whatever else the request's
  // chat_template_kwargs carried) rendered through the checkpoint's chat
  // template with the generation prompt appended — the prompt the
  // scheduler will prefill.
  virtual std::string render_chat(const minijson::Value& globals) const = 0;
  struct ChatInput {
    std::vector<int64_t> tokens;
    std::vector<ImageInput> images;
  };
  virtual bool supports_images() const { return false; }
  virtual ChatInput prepare_chat(const minijson::Value& globals) const {
    return {encode_text(render_chat(globals)), {}};
  }

  // The template's marker tokens (DESIGN §11): the reasoning split and
  // the tool-call parser key on their ids. A frontend without them
  // (the default) serves plain chat: tool requests refuse, nothing is
  // split.
  virtual dgpp::text::ChatMarkers markers() const { return {}; }
  // Whether the checkpoint's template reads the global `name` (the knob
  // gate for chat_template_kwargs: enable_thinking is a knob of the
  // Qwen3.8-Flash-Next and GLM-4.7 templates, not of GLM-5.3-Flash's).
  virtual bool template_reads(std::string_view) const { return false; }
  // Resolve API effort into controls the model actually consumes. Models
  // with only a thinking switch map every positive effort to that switch.
  virtual ReasoningSettings reasoning_settings(std::string_view effort) const {
    ReasoningSettings out;
    if (effort == "none") {
      if (template_reads("enable_thinking")) out.enable_thinking = false;
      else if (markers().reasoning_available())
        throw std::invalid_argument("this model cannot disable reasoning");
    } else {
      if (template_reads("enable_thinking")) out.enable_thinking = true;
      if (template_reads("reasoning_effort")) out.effort = std::string(effort);
      if (!out.enable_thinking && !out.effort)
        throw std::invalid_argument("this model exposes no reasoning control");
    }
    return out;
  }
  // The prefix cache's boundary tokens (M7): the ids whose positions in a
  // prompt are its structural boundaries — the template's role markers
  // (<|system|>, <|user|>, <|assistant|>, <|observation|>), so consecutive
  // turns of one conversation cut at the same positions. Empty: no
  // boundaries (the cache still cuts at chunk multiples).
  virtual std::vector<int64_t> boundary_token_ids() const { return {}; }
};

struct ServiceConfig {
  FileInputConfig file_inputs;
  // What /v1/models reports and what requests must name in "model".
  std::string model_id;
  int default_max_tokens = 256;  // when the request omits max_tokens
  int queue_limit = 64;          // admission bound; beyond → 503
  int sse_ping_interval = kDefaultSsePingInterval;  // seconds; -1 disables
  dgpp::sched::AdmissionPolicy admission;  // full-reserve unless told otherwise
  // The sampling defaults every omitted request field takes: the
  // checkpoint's generation_config.json with the process's overrides
  // applied (DESIGN §10's HF contract). temperature 0 is greedy. When the
  // bound engine cannot sample, the constructor collapses them to greedy
  // and logs it — the service never advertises a mode it cannot execute.
  sample::Params sampling_defaults = sample::greedy_params();
  // The seed for requests that omit one. Absent: rank 0 draws a fresh seed
  // per request (and the journal carries it). Set: every seedless request
  // uses it — the gates' reproducible runs.
  std::optional<uint64_t> fixed_seed;
  // Reasoning on the wire: false (default) routes the ids before </think>
  // to the message's reasoning_content (the vLLM/DeepSeek convention);
  // true folds them into content as "<think>…</think>" text for clients
  // that expect the raw transcript.
  bool reasoning_in_content = false;
  // JSON object of defaults for supported template kwargs. Requests override
  // the same keys; native history names are never translated into each other.
  std::string default_chat_template_kwargs = "{}";
  // The prefix cache's key (M7), reported by /v1/metrics: the tokenizer
  // revision, the template hash and the checkpoint the entries were taken
  // under (the cache is per process; the key names what it is bound to).
  std::string prefix_key;
  // The vocabulary bound for logit_bias ids: the model's
  // vocab_size; 0 refuses logit_bias (the bound is unknown).
  int64_t vocab_size = 0;
  // The request-context surface (review item 7, 2026-09-18): the rope ramp
  // this process serves with (engine.rope_scaling, absent = the
  // checkpoint's plain rope) and the two bounds on one request — the
  // family's positional ceiling (what its rope table can address) and the
  // K/V pool a request seats in. /v1/models reports the ramp, both bounds
  // and the effective limit, which is their MINIMUM: a YaRN ramp lifts what
  // a request may reach and enlarges nothing (docs/qwen38_flash_next_plan.md
  // §1.9.1). 0 = the family does not state it; the field stays off the
  // response (every field here is additive to the model object).
  std::optional<dgpp::RopeScaling> rope_scaling;
  int64_t position_ceiling = 0;
  int64_t kv_pool_tokens = 0;
};

// The stop-string scanner (OpenAI's `stop`, 2026-09-06). Fed the visible
// text as it is produced, it returns what may be shown now: everything up
// to the first match of a stop string (the match and what follows are
// never shown; `hit` stays set), or everything but a tail that could still
// begin a stop string across the next piece (held, and prepended to the
// next piece). `finish()` releases the hold at the end of a stream that
// never matched.
struct StopScanner {
  std::vector<std::string> stops;
  std::string hold;
  bool hit = false;
  bool active() const { return !stops.empty(); }
  std::string feed(std::string text);
  std::string finish();
};

class GenerationService : public HttpHandler,
                          public dgpp::sched::SchedulerObserver {
 public:
  GenerationService(const ServiceConfig& cfg,
                    dgpp::sched::SchedulerEngine* engine,
                    const ModelFrontend* frontend,
                    std::vector<int64_t> eos_token_ids);

  // ---- HttpHandler (HTTP thread) --------------------------------------
  void handle(const HttpRequest& req, HttpResponseWriter& w) override;
  void idle() override;
  void on_disconnect(uint64_t tag) override;

  // ---- engine-loop side (the app's engine thread) ---------------------
  // One engine pass's scheduler-state changes — exactly what the fabric
  // journal broadcasts (service/fabric_serve.hpp). submits are the
  // requests try_submit ACCEPTED; cancels are the ids whose cancel()
  // hit. Sheds (503 at the door or at admission) die on rank 0 and
  // never ride the journal — peers only ever see state changes their
  // identical queues will replay.
  struct PassEvents {
    std::vector<dgpp::sched::SchedulerRequest> submits;
    std::vector<std::string> cancels;
    std::vector<std::string> stops;  // the stop-string retires
    // The prefix cache's decision digest after the previous tick (M7): the
    // record carries it so every peer compares before applying this one.
    bool has_prefix_digest = false;
    uint64_t prefix_digest = 0;
    // The audit observer's op-stream fold after the previous tick (M9's
    // continuous drift check): rides the record when an observer with a
    // fold is attached.
    bool has_op_digest = false;
    uint64_t op_digest = 0;
  };
  // Invoked (engine thread) with the pass's events AFTER the drain and
  // immediately BEFORE the tick — the one fixed position where the
  // fabric broadcasts the record. The tick and its record are atomic:
  // rank 0 never ticks without broadcasting, a peer never ticks
  // without a record, so tick counts are identical by construction.
  using PreTickHook = std::function<void(const PassEvents&)>;

  // Applies every queued admission/cancel, then runs one scheduler
  // quantum. Returns whether work remains pending (the app may idle-
  // sleep when false; the pending-admission queue is drained first, so
  // arrivals always make the next pass productive).
  bool engine_pass(const PreTickHook& pre_tick = nullptr);

  // Optional audit tap on the engine event stream (tokens + retires,
  // engine thread). The fabric verification hashes the per-rank streams
  // against each other — the smoke's md5 procedure, serving edition. w1
  // leaves it unset.
  void set_audit_observer(dgpp::sched::SchedulerObserver* audit) {
    audit_ = audit;
  }

  // Stops accepting: every pending record is answered 503 and the
  // Drain-on-stop (M6 6c). begin_shutdown() — engine thread, at a pass
  // boundary — marks the service closed (new requests answer 503
  // server_shutdown), sheds the not-yet-admitted queue the same way, and
  // flags every live request for cancellation; the caller then runs one
  // more engine_pass(): the cancels ride the journal and the tick's cancel
  // sweep retires them on every rank at the same quantum, with no engine
  // op (so the bus never comes down under a collective). The interrupted
  // streams get the shutdown error event and [DONE], one-shots a 503 —
  // pumped by the HTTP thread, which the app keeps alive until drained()
  // says every answer is out. Returns the number of live requests
  // interrupted.
  int begin_shutdown();
  // Every record is answered (or its client is gone) and nothing is
  // pending: the HTTP server may stop without cutting a client off.
  bool drained() const;

  // The v1 failure semantics (DESIGN §9 item 6; built 2026-09-05): any
  // engine or fabric failure fails the service. fail_engine() — from ANY
  // thread: the engine thread after an engine op threw (a bus watchdog, a
  // journal write to a dead peer, a contract violation), or rank 0's peer
  // death watch while the engine thread is inside a collective the dead
  // rank will never complete — marks the service failed: every live
  // request is answered with the engine_failure error AFTER the tokens it
  // produced (each was committed on every rank before the failure, and no
  // rank commits anything after it — nothing uncommitted ever reaches a
  // client), queued and later requests get 503 engine_failure, /health
  // turns 503, and drained() says when the HTTP pump has answered everyone
  // so the process may exit nonzero. Idempotent. Returns the number of
  // live requests interrupted.
  int fail_engine(const std::string& what);
  bool failed() const;

  struct Stats {
    // HTTP requests, regardless of n; scheduler/TTFT counters count choices.
    uint64_t requests_total = 0;     // validated requests reaching admission
    uint64_t requests_shed = 0;      // 503s at the door or at admission
    uint64_t requests_cancelled = 0;  // client disconnects (and the stop)
    uint64_t requests_shed_pool = 0;  // grow-on-demand: cut short at exhaustion
    uint64_t requests_failed = 0;    // live requests an engine failure cut off
    uint64_t tokens_out = 0;
    uint64_t rejects_bad = 0;        // 400-class refusals
    uint64_t tool_calls_out = 0;     // parsed tool calls
    // The prefix cache's time to first token (M7), door to first token, for
    // requests that attached to an entry and for those that did not.
    uint64_t ttft_hit_count = 0;
    double ttft_hit_ms = 0;
    uint64_t ttft_miss_count = 0;
    double ttft_miss_ms = 0;
  };
  Stats stats() const;
  // The scheduler's meters as published after the last engine pass (the
  // throughput line reads them on the engine thread; /v1/metrics too).
  dgpp::sched::Scheduler::Meters meters() const;

  // The defaults actually served (after the engine-capability collapse)
  // and whether stochastic requests are executable at all.
  const sample::Params& sampling_defaults() const {
    return cfg_.sampling_defaults;
  }
  bool sampling_available() const { return sampling_available_; }
  // Whether requests may carry tools (the frontend has the markers), and
  // whether tool_choice / parallel_tool_calls can be enforced (the engine
  // masks the pick).
  bool tool_calls_available() const {
    return markers_.tool_calls_available();
  }
  bool constraints_available() const {
    return tool_calls_available() && engine_->supports_constraints();
  }

 private:
  using ParserEvent = dgpp::text::ToolCallParser::Event;
  using ToolCall = dgpp::text::ToolCallParser::Call;

  // A request's n choices (2026-09-06): one record per choice, one writer,
  // one end sequence once every choice is done — the usage summed, the
  // one-shot's choices assembled by index.
  struct ChoiceGroup {
    int sse_ping_interval = kDefaultSsePingInterval;
    int n = 1;
    int finished = 0;            // choices whose end sequence is written
    bool ended = false;          // the stream ended / the one-shot answered
    bool counted_shed = false, counted_cancelled = false;
    bool counted_pool = false, counted_failed = false;
    int completion_tokens = 0;   // summed over the choices
    int reasoning_tokens = 0;
    int cached_tokens = 0;       // choice 0's prefix-cache attach position
    std::vector<std::string> choices;  // one-shot: each choice's JSON
  };

  struct StreamRecord {
    uint64_t tag = 0;        // on_disconnect correlation (one per connection)
    std::string id;          // the response id (shared by a request's choices)
    std::string sched_id;    // the scheduler request id: `id` for choice 0,
                             // `id-<choice>` for the others (n, 2026-09-06)
    int choice = 0;          // the choice index this record fills
    std::shared_ptr<ChoiceGroup> group;  // the request's choices together
    uint64_t call_seed = 0;  // tool_call ids: the tag and the choice
    std::string model;
    int64_t created_unix = 0;
    bool chat = false;       // chat route (the parser path) vs legacy
    bool stream = false;
    bool include_usage = false;
    bool include_obfuscation = true;
    bool report_service_tier = false;
    std::string metadata;
    bool first_chunk_sent = false;
    bool done = false;         // retired or rejected — ready to finish
    bool reject_overloaded = false;
    bool shutting_down = false;  // interrupted or shed by the stop (M6 6c):
                                 // answered with the server_shutdown error
    bool engine_failed = false;  // cut off or refused by an engine failure:
                                 // answered with the engine_failure error
    bool writer_dead = false;  // on_disconnect fired; never touch it
    bool cancel_armed = false; // disconnect seen; cancel enqueued
    dgpp::sched::Scheduler::Result::Reason reason =
        dgpp::sched::Scheduler::Result::Reason::kNone;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    // The stop strings: the scanner over the visible text
    // (chat: the content; legacy: the text), the stop enqueued once, the
    // token count at the match (the usage's completion_tokens).
    StopScanner stop;
    bool stopped = false;
    int stop_tokens = 0;
    // usage.completion_tokens_details.reasoning_tokens: the
    // ids the parser routed to reasoning (</think> included) while the
    // prompt's <think> was open.
    bool reasoning_open = false;
    int reasoning_tokens = 0;
    // The prefix cache (M7): when the request arrived at the door, whether
    // it attached to an entry and at what position (the TTFT split).
    std::chrono::steady_clock::time_point arrived;
    bool prefix_hit = false;
    int64_t prefix_position = 0;
    // UTF-8 carries (the soak's find, 2026-09-05): a byte-level BPE token
    // can end inside a multi-byte character, and JSON text must be UTF-8 —
    // an incomplete trailing sequence is held here and prepended to the
    // field's next delta; what is left at the end becomes U+FFFD.
    std::string carry_reasoning, carry_content, carry_args, carry_text;
    int logprobs = -1;         // -1 none; N = top-N alternatives requested
    std::vector<sample::Result> lps;  // one per id when logprobs >= 0
    std::vector<bool> content_lps;  // generated tokens contributing visible content
    std::vector<bool> lps_reported;
    std::vector<ParserEvent::TokenSpan> content_spans;
    size_t content_input_bytes = 0;
    size_t content_span_cursor = 0;
    size_t lps_flushed = 0;    // streaming: entries already sent
    std::vector<int64_t> ids;  // generated so far
    // Legacy completions: the suffix-diff text path.
    std::string text;          // decoded so far (the suffix-diff base)
    std::string delta;         // unflushed text delta (the ring)
    std::string out_text;      // the visible text: the stop cut applied
    // Chat: the parser splits the ids into reasoning / content / calls.
    std::unique_ptr<dgpp::text::ToolCallParser> parser;
    std::vector<ParserEvent> pending;  // streams: events not yet flushed
    std::string reasoning;     // accumulated (one-shots)
    std::string content;       // accumulated (one-shots)
    std::vector<ToolCall> calls;
    std::vector<std::string> custom_tools;
    std::string generation_error;
    int calls_announced = 0;   // streaming: calls already sent (HTTP thread)
    HttpResponseWriter* writer = nullptr;  // HTTP thread only
  };

  // Routes (HTTP thread). Each parses, validates, and either responds
  // directly or creates a record + enqueues an admission.
  void route_chat_completions(const HttpRequest& req,
                              HttpResponseWriter& w);
  void route_completions(const HttpRequest& req, HttpResponseWriter& w);
  void route_models(const HttpRequest& req, HttpResponseWriter& w);
  void route_health(HttpResponseWriter& w) const;
  void route_metrics(HttpResponseWriter& w);
  void route_metrics_prometheus(HttpResponseWriter& w);
  bool validate_chat_parameters(const minijson::Value& body, HttpResponseWriter& w);
  bool parse_max_tokens(const minijson::Value& body, HttpResponseWriter& w, int* steps, bool chat);
  // OpenAI's ignore_eos: generate to the token limit whatever is drawn.
  bool parse_ignore_eos(const minijson::Value& body, HttpResponseWriter& w, bool* ignore);
  bool parse_sse_ping_interval(const minijson::Value& body, HttpResponseWriter& w, bool stream,
                               int* interval);
  bool parse_stream_options(const minijson::Value& body, HttpResponseWriter& w,
                            bool stream, bool* usage, bool* obfuscation);
  void write_stream_event(StreamRecord& r, std::string event, bool usage = false);
  void mark_content_logprobs(StreamRecord& r);

  // The request's sampling spec: every present field validated and
  // applied over the defaults, the seed drawn when omitted. Responds 400
  // (naming the field) and returns false on any refusal.
  bool parse_sampling(const dgpp::minijson::Value& body,
                      HttpResponseWriter& w, sample::Params* sampling,
                      uint64_t* seed);
  // n, stop and logit_bias: validated per the schema, 400
  // naming the field otherwise; logit_bias also needs an engine that can
  // bias the pick and the vocabulary bound.
  bool parse_n(const dgpp::minijson::Value& body, HttpResponseWriter& w, int* n);
  bool parse_stop(const dgpp::minijson::Value& body, HttpResponseWriter& w,
                  std::vector<std::string>* stops);
  bool parse_logit_bias(const dgpp::minijson::Value& body, HttpResponseWriter& w,
                        std::vector<dgpp::sched::LogitBias>* bias);

  // The chat request's conversation and tool fields (M6 6f): validates
  // the messages (roles, content forms, assistant tool_calls, tool
  // messages), tools, tool_choice, parallel_tool_calls, reasoning_effort
  // and chat_template_kwargs; produces the template globals, the forced
  // prefix (tool_choice required / named), the parser's schemas and the
  // seeded function name. Responds 400 and returns false on any refusal.
  struct ChatPlan {
    dgpp::minijson::Value globals;
    dgpp::text::GrammarSpec grammar;  // the pick's constraint (inactive: none)
    bool tools_requested = false;
    dgpp::text::ToolSchemas schemas;
    std::vector<std::string> custom_tools;
  };
  bool parse_chat(const dgpp::minijson::Value& body, HttpResponseWriter& w,
                  ChatPlan* plan);
  void enqueue_file_work(const HttpRequest& req, HttpResponseWriter& writer, bool chat);
  void pump_file_work();

  // The OpenAI error body (never a bare string).
  void respond_error(HttpResponseWriter& w, int status,
                     const std::string& message, const std::string& type,
                     const std::string& param = "",
                     const std::string& code = "");

  // Admits a validated request: creates the record, hands the writer,
  // and enqueues the submit (the engine thread applies it).
  void enqueue_admission(std::shared_ptr<StreamRecord> record,
                         dgpp::sched::SchedulerRequest request);
  // A request's n choices at once: admitted or shed together.
  void enqueue_group(std::vector<std::shared_ptr<StreamRecord>> records,
                     std::vector<dgpp::sched::SchedulerRequest> requests);
  // The visible content (chat) past the stop scanner: accumulated for the
  // one-shot, queued for the stream. `request_stop` enqueues the record's
  // stop once (under the lock, on the engine thread).
  void push_content(StreamRecord& r, std::string text);
  void request_stop(StreamRecord& r);

  // Shared by both streaming and one-shot records: appends one token
  // (the observer, engine thread) and decodes the text suffix.
  void on_token(const std::string& id, int64_t token,
                int steps_done) override;
  void on_token_logprobs(const std::string& id, int steps_done,
                         const sample::Result& logprobs) override;
  // Folds one parser event into a chat record (under the lock): the
  // accumulated texts and calls, the fold knob, the stream's queue.
  void absorb(StreamRecord& r, ParserEvent ev);
  // The OpenAI logprobs content entries for record tokens [from, to).
  std::string logprobs_content(const StreamRecord& r, size_t from,
                               size_t to, bool unreported_only = false) const;
  std::string take_content_logprobs(StreamRecord& r) const;
  // The legacy text_completion logprobs object over every token.
  std::string legacy_logprobs(const StreamRecord& r, size_t from = 0,
                             size_t to = SIZE_MAX) const;
  void on_retire(const std::string& id,
                 const dgpp::sched::Scheduler::Result& result) override;
  // Grow-on-demand's growth events (M6 6d) ride to the audit observer:
  // they are rank-identical scheduler state, so the op streams carry them.
  void on_grow(const std::string& id, int64_t reserved_tokens) override {
    if (audit_) audit_->on_grow(id, reserved_tokens);
  }
  // The prefix cache's decisions (M7) ride to the audit observer too; an
  // attach marks the record for the TTFT split.
  void on_prefix(const std::string& id, const char* op, int64_t position,
                 int slot) override;
  // The prompt's structural boundaries: every position (>= 1) holding one
  // of the frontend's boundary tokens.
  std::vector<int64_t> prompt_boundaries(const std::vector<int64_t>& prompt) const;

  // idle()'s record pump: flushes deltas, finishes done records.
  void pump_records();
  // One stream's unflushed events (and logprobs) as SSE chunks.
  void flush_chat_stream(StreamRecord& r);
  void flush_legacy_stream(StreamRecord& r);
  void flush_stream_carries(StreamRecord& r);  // the held UTF-8 tails, at the end

  ServiceConfig cfg_;
  // Parsed once at startup; string values borrow storage from cfg_.
  minijson::Value default_chat_template_kwargs_;
  FileInputs file_inputs_;
  struct PendingFileWork {
    uint64_t tag;
    HttpResponseWriter* writer;
    bool chat;
    std::future<FileResponse> result;
  };
  std::vector<PendingFileWork> file_work_;  // HTTP thread only; destroyed before file_inputs_
  std::atomic<size_t> pending_file_count_{0};  // observed by shutdown drain
  dgpp::sched::SchedulerEngine* engine_;
  const ModelFrontend* frontend_;
  dgpp::text::ChatMarkers markers_;
  std::vector<int64_t> boundary_ids_;  // the frontend's, sorted (M7)
  dgpp::sched::Scheduler sched_;  // engine thread only (except try_submit
                                 // under the lock via engine_pass)
  // Engine-thread-only (set once before the loop, read in the observer
  // callbacks, which the scheduler invokes on the engine thread).
  dgpp::sched::SchedulerObserver* audit_ = nullptr;

  // Everything below lives under mutex_ (the one lock, held briefly).
  struct PendingAdmission {
    std::shared_ptr<StreamRecord> record;
    dgpp::sched::SchedulerRequest request;
  };
  struct PendingCancel {
    std::string scheduler_id;
  };
  std::vector<PendingAdmission> pending_admissions_;
  std::vector<PendingCancel> pending_cancels_;
  std::vector<std::string> pending_stops_;  // scheduler ids whose stop matched
  std::vector<std::shared_ptr<StreamRecord>> records_;
  dgpp::sched::Scheduler::Meters meters_;  // engine-published, mutex-guarded
  dgpp::sched::SchedulerEngine::PrefixEngineStats prefix_stats_;
  std::chrono::steady_clock::time_point meters_published_ = std::chrono::steady_clock::now();
  bool shutdown_ = false;
  bool failed_ = false;      // fail_engine() happened
  std::string failure_;      // its reason (the clients' message carries it)
  mutable std::mutex mutex_;

  // HTTP-thread-only counters guarded by std::atomic where cross-thread.
  std::atomic<uint64_t> next_tag_{(static_cast<uint64_t>(std::random_device{}()) << 32) ^
                                 static_cast<uint64_t>(std::random_device{}())};
  Stats stats_;  // written under mutex_ (cheap, exact)
  bool sampling_available_ = false;
  std::mt19937_64 seed_rng_;  // HTTP thread only (route handlers)
  std::mt19937_64 obfuscation_rng_{std::random_device{}()};  // independent of sampler seeds
};

}  // namespace dgpp::serve
