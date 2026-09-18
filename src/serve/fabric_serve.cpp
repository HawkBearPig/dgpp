#include "serve/fabric_serve.hpp"

#include <format>

#include <cstdlib>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>

#include "common/log.hpp"
#include "kernels/latent_format.hpp"
#include "loaders/minijson.hpp"
#include "serve/json_out.hpp"

namespace dgpp::serve {

using dgpp::sched::Scheduler;
using dgpp::sched::SchedulerRequest;

// ---- the audit tap --------------------------------------------------------

void OpStreamObserver::append(const std::string& line, bool flush) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (file_ != nullptr) {
    if (std::fwrite(line.data(), 1, line.size(), file_) != line.size() &&
        !write_failed_) {
      write_failed_ = true;
      DGPP_LOG_ERROR("serve: short write to the op stream file");
    }
    if (flush) std::fflush(file_);
  } else {
    text_ += line;
  }
  for (const char c : line) {  // FNV-1a over the bytes, line by line
    digest_ ^= static_cast<unsigned char>(c);
    digest_ *= 0x100000001b3ull;
  }
}

bool OpStreamObserver::open(const std::string& path) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (file_ != nullptr) std::fclose(file_);
  file_ = std::fopen(path.c_str(), "wb");
  if (file_ == nullptr) return false;
  // Lines recorded before the open go first, so the file is the whole
  // stream.
  if (!text_.empty()) {
    std::fwrite(text_.data(), 1, text_.size(), file_);
    std::string().swap(text_);
  }
  return true;
}

void OpStreamObserver::flush() {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (file_ != nullptr) std::fflush(file_);
}

OpStreamObserver::~OpStreamObserver() {
  if (file_ != nullptr) std::fclose(file_);
}

void OpStreamObserver::on_token(const std::string& id, int64_t token,
                                int steps_done) {
  append("T " + id + " " + std::to_string(token) + " " +
         std::to_string(steps_done) + "\n");
}

void OpStreamObserver::on_retire(const std::string& id,
                                const Scheduler::Result& result) {
  // The reason rides as its enum ordinal — same binary family, same
  // values, byte-comparable across ranks.
  append("R " + id + " " + std::to_string(static_cast<int>(result.reason)) +
             " " + std::to_string(result.steps_done) + "\n",
         /*flush=*/true);
}

void OpStreamObserver::on_grow(const std::string& id, int64_t reserved_tokens) {
  append("W " + id + " " + std::to_string(reserved_tokens) + "\n");
}

void OpStreamObserver::on_prefix(const std::string& id, const char* op,
                                 int64_t position, int slot) {
  append("X " + std::string(op) + " " + id + " " + std::to_string(position) +
         " " + std::to_string(slot) + "\n");
}

std::string OpStreamObserver::text() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return text_;
}

uint64_t OpStreamObserver::digest() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return digest_;
}

// ---- wire codec -----------------------------------------------------------

namespace {

// Floats ride as their IEEE bit patterns: the peers must apply the EXACT
// spec rank 0 applied, and a decimal round trip is one more place for a
// rank to differ by an ulp.
uint32_t float_bits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}
float float_from_bits(uint32_t u) {
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}
std::string hex64(uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

}  // namespace

std::string encode_journal_tick(const GenerationService::PassEvents& events) {
  std::string out = "{\"op\":\"tick\"";
  if (!events.submits.empty()) {
    out += ",\"s\":[";
    for (size_t i = 0; i < events.submits.size(); ++i) {
      if (i != 0) out.push_back(',');
      const SchedulerRequest& r = events.submits[i];
      out += "{\"id\":";
      append_json_string(&out, r.id);
      out += ",\"p\":[";
      for (size_t j = 0; j < r.prompt.size(); ++j) {
        if (j != 0) out.push_back(',');
        append_json_int(&out, r.prompt[j]);
      }
      out += "],\"m\":";
      append_json_int(&out, r.max_steps);
      if (r.cancel_after > 0) {
        out += ",\"ca\":";
        append_json_int(&out, r.cancel_after);
      }
      // The prefix cache's inputs (M7): a request without boundaries that
      // has not opted out writes neither — the pre-cache record, byte for
      // byte.
      if (!r.boundaries.empty()) {
        out += ",\"b\":[";
        for (size_t j = 0; j < r.boundaries.size(); ++j) {
          if (j != 0) out.push_back(',');
          append_json_int(&out, r.boundaries[j]);
        }
        out += "]";
      }
      if (r.no_cache) out += ",\"nc\":1";
      // The logit bias: [id, float bits] pairs.
      if (!r.logit_bias.empty()) {
        out += ",\"lb\":[";
        for (size_t i = 0; i < r.logit_bias.size(); ++i) {
          if (i != 0) out.push_back(',');
          out.push_back('[');
          append_json_int(&out, r.logit_bias[i].token);
          out.push_back(',');
          append_json_int(&out, float_bits(r.logit_bias[i].bias));
          out.push_back(']');
        }
        out.push_back(']');
      }
      // The sampling spec rides for stochastic requests and for greedy ones
      // that ask for logprobs; a plain greedy request's record is
      // byte-identical to the pre-sampling format.
      if (r.logprobs >= 0) {
        out += ",\"lp\":";
        append_json_int(&out, r.logprobs);
      }
      if (r.sampling.temperature > 0.0f || r.logprobs >= 0) {
        const sample::Params& g = r.sampling;
        out += ",\"g\":{\"t\":";
        append_json_int(&out, float_bits(g.temperature));
        out += ",\"p\":";
        append_json_int(&out, float_bits(g.top_p));
        out += ",\"k\":";
        append_json_int(&out, g.top_k);
        out += ",\"m\":";
        append_json_int(&out, float_bits(g.min_p));
        out += ",\"r\":";
        append_json_int(&out, float_bits(g.repetition_penalty));
        out += ",\"f\":";
        append_json_int(&out, float_bits(g.frequency_penalty));
        out += ",\"q\":";
        append_json_int(&out, float_bits(g.presence_penalty));
        out += ",\"l\":";
        append_json_int(&out, g.logprobs);
        out += ",\"s\":\"" + hex64(r.seed) + "\"}";
      }
      // The grammar (M6 6g) rides only when active: mode, the parallel
      // flag, the named function, the tools with their closed key sets.
      if (r.grammar.active()) {
        const dgpp::text::GrammarSpec& gr = r.grammar;
        out += ",\"gr\":{\"m\":";
        append_json_int(&out, static_cast<int>(gr.mode));
        out += ",\"pl\":";
        out += gr.parallel ? "true" : "false";
        out += ",\"n\":";
        append_json_string(&out, gr.named);
        out += ",\"t\":[";
        for (size_t t = 0; t < gr.tools.size(); ++t) {
          if (t != 0) out.push_back(',');
          out += "{\"n\":";
          append_json_string(&out, gr.tools[t].name);
          if (gr.tools[t].constrain_keys) {
            out += ",\"k\":[";
            for (size_t k = 0; k < gr.tools[t].keys.size(); ++k) {
              if (k != 0) out.push_back(',');
              append_json_string(&out, gr.tools[t].keys[k]);
            }
            out.push_back(']');
          }
          // A strict tool and its required keys (the call's closing gate).
          if (gr.tools[t].strict) out += ",\"s\":true";
          if (!gr.tools[t].required_keys.empty()) {
            out += ",\"r\":[";
            for (size_t k = 0; k < gr.tools[t].required_keys.size(); ++k) {
              if (k != 0) out.push_back(',');
              append_json_string(&out, gr.tools[t].required_keys[k]);
            }
            out.push_back(']');
          }
          // The typed arguments (M6 6i): only the constrained ones ride.
          bool any_typed = false;
          for (const auto& a : gr.tools[t].args)
            any_typed = any_typed || a.kind != dgpp::text::GrammarArg::Kind::kFree;
          if (any_typed) {
            out += ",\"a\":[";
            bool first = true;
            for (const auto& a : gr.tools[t].args) {
              if (a.kind == dgpp::text::GrammarArg::Kind::kFree) continue;
              if (!first) out.push_back(',');
              first = false;
              out += "{\"k\":";
              append_json_string(&out, a.key);
              if (a.kind == dgpp::text::GrammarArg::Kind::kJson) {
                out += ",\"j\":";
                append_json_string(&out, a.schema);
              } else {
                out += ",\"t\":[";
                for (size_t i = 0; i < a.texts.size(); ++i) {
                  if (i != 0) out.push_back(',');
                  append_json_string(&out, a.texts[i]);
                }
                out.push_back(']');
              }
              out.push_back('}');
            }
            out.push_back(']');
          }
          out.push_back('}');
        }
        out += "]";
        if (gr.has_json()) {
          out += ",\"js\":";
          append_json_string(&out, gr.json_schema);
        }
        out.push_back('}');
      }
      out.push_back('}');
    }
    out.push_back(']');
  }
  if (!events.cancels.empty()) {
    out += ",\"c\":[";
    for (size_t i = 0; i < events.cancels.size(); ++i) {
      if (i != 0) out.push_back(',');
      append_json_string(&out, events.cancels[i]);
    }
    out.push_back(']');
  }
  // The stop-string retires (2026-09-06): applied at the tick's sweep on
  // every rank, exactly like cancels, with their own reason.
  if (!events.stops.empty()) {
    out += ",\"sp\":[";
    for (size_t i = 0; i < events.stops.size(); ++i) {
      if (i != 0) out.push_back(',');
      append_json_string(&out, events.stops[i]);
    }
    out.push_back(']');
  }
  // The prefix cache's cross-rank check (M7): rank 0's decision digest
  // after the previous tick, as a decimal string (a uint64 is not a JSON
  // number the reader can trust to survive). A cache-less rank 0 writes
  // none — the pre-cache record.
  if (events.has_prefix_digest) {
    out += ",\"pd\":\"" + std::to_string(events.prefix_digest) + "\"";
  }
  // The continuous drift check (M9): rank 0's op-stream fold after the
  // previous tick, the same way. A rank 0 without an audit observer
  // writes none.
  if (events.has_op_digest) {
    out += ",\"od\":\"" + std::to_string(events.op_digest) + "\"";
  }
  out.push_back('}');
  return out;
}

std::string encode_journal_stop() { return "{\"op\":\"stop\"}"; }
std::string encode_journal_warm(const dgpp::sched::AdmissionPolicy& policy,
                                int prefix_slots,
                                const std::string& config_digest) {
  std::string out = "{\"op\":\"warm\",\"adm\":" +
                    std::to_string(static_cast<int>(policy.mode)) +
                    ",\"win\":" + std::to_string(policy.window_tokens) +
                    (prefix_slots > 0 ? ",\"pc\":" + std::to_string(prefix_slots) : "");
  // The effective configuration's digest: every peer compares
  // its own before serving; a config-less rank 0 writes none.
  if (policy.prefill_budget_tokens > 0) out += ",\"pfbudget\":" + std::to_string(policy.prefill_budget_tokens);
  if (policy.prefill_idle_budget_tokens > 0) out += ",\"pfidle\":" + std::to_string(policy.prefill_idle_budget_tokens);
  if (!config_digest.empty()) {
    out += ",\"cfg\":";
    append_json_string(&out, config_digest);
  }
  out.push_back('}');
  return out;
}

std::string encode_journal_settings(const WorldSettings& s) {
  std::string out = "{\"op\":\"settings\",\"ver\":";
  append_json_string(&out, s.version);
  out += ",\"model\":";
  append_json_string(&out, s.model);
  out += ",\"ckpt\":";
  append_json_string(&out, s.checkpoint);
  out += std::format(
      ",\"world\":{},\"fabric\":{},\"conc\":{},\"kv\":{},\"maxtok\":{},\"queue\":{},"
      "\"eos\":{},\"graph\":{},\"mtp\":{},\"mtpd\":{},\"batchmin\":{},\"cand\":{},\"pcgib\":{:.17g},"
      "\"adm\":",
      s.world, s.fabric_port, s.max_concurrency, s.kv_capacity, s.default_max_tokens,
      s.queue_limit, s.no_eos ? 1 : 0, s.decode_graph ? 1 : 0, s.mtp ? 1 : 0,
      s.mtp_depth, s.graph_batch_min_live, s.sampling_candidates, s.prefix_cache_gib);
  append_json_string(&out, s.admission);
  if (s.prefill_budget_tokens > 0) out += ",\"pfbudget\":" + std::to_string(s.prefill_budget_tokens);
  if (s.prefill_idle_budget_tokens > 0) out += ",\"pfidle\":" + std::to_string(s.prefill_idle_budget_tokens);
  out += std::format(",\"win\":{},\"pace\":{:.17g},\"inflight\":{},\"rdv\":{},\"stats\":{:.17g},\"ric\":{},\"kvdt\":",
      s.admission_window, s.bulk_pace_gbps, s.bulk_inflight, s.rendezvous_timeout_ms,
      s.stats_interval_s, s.reasoning_in_content ? 1 : 0);
  append_json_string(&out, s.kv_dtype);
  out += ",\"ngt\":";
  append_json_string(&out, s.ngram_table);
  out += ",\"dw\":";
  append_json_string(&out, s.dense_weights);
  out += ",\"pf\":";
  append_json_string(&out, s.prefill);
  // The opt-in YaRN ramp: absent when off, so a record from a plain run
  // carries no key at all (an older peer decodes it to "off", which is
  // what it would have run anyway — a world never mixes the two).
  if (s.rope_scaling) {
    const dgpp::RopeScaling& rs = *s.rope_scaling;
    out += std::format(",\"rs\":[{:.17g},{},{:.17g},{:.17g},{:.17g},{:.17g}]", rs.factor,
                       rs.original_max_position_embeddings, rs.beta_fast, rs.beta_slow,
                       rs.attn_factor, rs.mrope_cache_factor);
  }
  out += ",\"emsh\":";
  append_json_string(&out, s.embed_sharding);
  out += std::format(",\"mss\":{},\"msrow\":{:.17g},\"msbase\":{:.17g},\"mslam\":{:.17g},\"msmin\":{},\"msad\":{}",
                     s.mtp_schedule ? 1 : 0, s.mtp_schedule_row_ms, s.mtp_schedule_base_ms,
                     s.mtp_schedule_lambda, s.mtp_schedule_min_depth, s.mtp_schedule_adapt ? 1 : 0);
  out.push_back('}');
  return out;
}

void check_prefix_digest(const JournalRecord& rec,
                         const dgpp::sched::Scheduler& sched) {
  if (!rec.has_prefix_digest) return;
  if (rec.prefix_digest == sched.prefix_digest()) return;
  throw std::runtime_error(
      "journal: rank 0's prefix-cache digest " + std::to_string(rec.prefix_digest) +
      " differs from this rank's " + std::to_string(sched.prefix_digest()) +
      " — the cache decisions diverged (§11); fabric emergency");
}

void check_op_digest(const JournalRecord& rec,
                     const dgpp::sched::SchedulerObserver* oplog, int64_t tick) {
  if (!rec.has_op_digest || oplog == nullptr || !oplog->has_digest()) return;
  const uint64_t mine = oplog->digest();
  if (rec.op_digest == mine) return;
  throw std::runtime_error(
      "journal: op-stream divergence at tick " + std::to_string(tick) +
      " — rank 0's fold " + std::to_string(rec.op_digest) +
      ", this rank's " + std::to_string(mine) +
      " (the engine event streams differ within the last tick; §11); "
      "fabric emergency");
}

namespace {

// The peer's journal connect races rank 0's journal bind: a peer's bus
// rendezvous can complete milliseconds before rank 0 (still logging
// its own bus-up) reaches the bind — observed live 2026-09-01: peer 3
// was refused 30ms after its bus came up, and rank 0's listener bound
// 30ms after THAT. The 7-hour clock skew between boxes makes the
// timestamps look impossible; the mechanism is a plain single-shot
// connect losing a race. Retry inside the window (the same lesson the
// bus rendezvous taught the soak, learned the cheap way this time).
dgpp::net::TcpConn journal_connect_with_retry(const std::string& host,
                                              uint16_t port,
                                              int window_ms) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(window_ms);
  std::string last_error;
  for (;;) {
    try {
      dgpp::net::TcpConn conn = dgpp::net::TcpConn::connect(host, port, 5000);
      DGPP_LOG_INFO("journal: connected to rank 0 at {}:{} ({} attempt window)",
                    host, port, window_ms);
      return conn;
    } catch (const std::exception& e) {
      last_error = e.what();
      DGPP_LOG_INFO("journal: connect to {}:{} not ready yet ({}); retrying",
                    host, port, last_error);
    }
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error(
          "journal: cannot connect to " + host + ":" +
          std::to_string(port) + " within " + std::to_string(window_ms) +
          "ms (rank 0's journal never appeared?) — last: " + last_error);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

const dgpp::minijson::Value& field(const dgpp::minijson::Value& v,
                                  const char* name, const char* where) {
  const dgpp::minijson::Value* f = v.find(name);
  if (f == nullptr)
    throw std::runtime_error(std::string("journal: ") + where +
                             " is missing '" + name + "'");
  return *f;
}

}  // namespace

JournalRecord decode_journal_line(std::string_view line) {
  // minijson VIEWS ITS INPUT — ParseResult owns nothing, string
  // members are string_views into the parsed buffer. The buffer must
  // outlive every read: parse a NAMED local (a temporary std::string
  // would dangle every view the moment it dies), and copy the values
  // out into the record before returning.
  const std::string src(line);
  const dgpp::minijson::ParseResult pr = dgpp::minijson::parse(src);
  const dgpp::minijson::Value& v = pr.root;
  JournalRecord rec;
  if (!v.is_object())
    throw std::runtime_error("journal: record is not a JSON object");
  const std::string op(field(v, "op", "record").as_string());
  if (op == "stop") {
    rec.stop = true;
    return rec;
  }
  if (op == "settings") {
    rec.settings = true;
    WorldSettings& s = rec.world_settings;
    const auto num = [&](const char* key) -> const dgpp::minijson::Value& {
      const dgpp::minijson::Value& f = field(v, key, "settings");
      if (!f.is_number())
        throw std::runtime_error(std::string("journal: settings record with a bad ") + key);
      return f;
    };
    const auto flag = [&](const char* key) {
      const int64_t x = num(key).as_int();
      if (x != 0 && x != 1)
        throw std::runtime_error(std::string("journal: settings record with a bad flag ") + key);
      return x == 1;
    };
    s.version = std::string(field(v, "ver", "settings").as_string());
    s.model = std::string(field(v, "model", "settings").as_string());
    s.checkpoint = std::string(field(v, "ckpt", "settings").as_string());
    s.world = static_cast<int>(num("world").as_int());
    s.fabric_port = static_cast<int>(num("fabric").as_int());
    s.max_concurrency = static_cast<int>(num("conc").as_int());
    s.kv_capacity = num("kv").as_int();
    s.default_max_tokens = static_cast<int>(num("maxtok").as_int());
    s.queue_limit = static_cast<int>(num("queue").as_int());
    s.no_eos = flag("eos");
    s.decode_graph = flag("graph");
    s.mtp = flag("mtp");
    s.mtp_depth = static_cast<int>(num("mtpd").as_int());
    s.graph_batch_min_live = static_cast<int>(num("batchmin").as_int());
    s.sampling_candidates = static_cast<int>(num("cand").as_int());
    s.prefix_cache_gib = num("pcgib").as_double();
    s.admission = std::string(field(v, "adm", "settings").as_string());
    s.admission_window = static_cast<int>(num("win").as_int());
    if (const auto* budget = v.find("pfbudget")) {
      if (!budget->is_number() || budget->as_int() < 0 || budget->as_int() > (1 << 30))
        throw std::runtime_error("journal: settings record with a bad prefill budget");
      s.prefill_budget_tokens = static_cast<int>(budget->as_int());
    }
    if (const auto* budget = v.find("pfidle")) {
      if (!budget->is_number() || budget->as_int() < 0 || budget->as_int() > (1 << 30))
        throw std::runtime_error("journal: settings record with a bad idle prefill budget");
      s.prefill_idle_budget_tokens = static_cast<int>(budget->as_int());
    }
    s.bulk_pace_gbps = num("pace").as_double();
    s.bulk_inflight = static_cast<int>(num("inflight").as_int());
    s.rendezvous_timeout_ms = static_cast<int>(num("rdv").as_int());
    s.stats_interval_s = num("stats").as_double();
    s.reasoning_in_content = flag("ric");
    s.kv_dtype = std::string(field(v, "kvdt", "settings").as_string());
    // Records before 2026-09-10 carry no table residency: resident.
    if (const dgpp::minijson::Value* ngt = v.find("ngt")) s.ngram_table = std::string(ngt->as_string());
    if (const dgpp::minijson::Value* dw = v.find("dw")) s.dense_weights = std::string(dw->as_string());
    // Records before 2026-09-14 carry no prefill mode: bounded.
    if (const dgpp::minijson::Value* pf = v.find("pf")) s.prefill = std::string(pf->as_string());
    // Records without the rope ramp key carry none: the plain table
    // (records before the knob, 2026-09-18, and every plain run since).
    if (const dgpp::minijson::Value* rs = v.find("rs")) {
      if (!rs->is_array() || rs->items().size() != 6)
        throw std::runtime_error("journal: settings record with a bad rope scaling");
      for (const dgpp::minijson::Value& item : rs->items())
        if (!item.is_number())
          throw std::runtime_error("journal: settings record with a non-numeric rope scaling");
      dgpp::RopeScaling r;
      r.factor = rs->items()[0].as_double();
      r.original_max_position_embeddings = rs->items()[1].as_int();
      r.beta_fast = rs->items()[2].as_double();
      r.beta_slow = rs->items()[3].as_double();
      r.attn_factor = rs->items()[4].as_double();
      r.mrope_cache_factor = rs->items()[5].as_double();
      try {
        r.validate("journal settings rope scaling");
      } catch (const std::runtime_error& e) {
        throw std::runtime_error("journal: settings record with an impossible rope scaling (" +
                                 std::string(e.what()) + ")");
      }
      s.rope_scaling = r;
    }
    // Records before 2026-09-13 carry no embedding sharding: replicated.
    if (const dgpp::minijson::Value* es = v.find("emsh")) s.embed_sharding = std::string(es->as_string());
    // Records before 2026-09-14 carry no scheduled verify depth: off.
    if (const dgpp::minijson::Value* mss = v.find("mss")) {
      s.mtp_schedule = mss->as_int() != 0;
      s.mtp_schedule_row_ms = field(v, "msrow", "settings").as_double();
      s.mtp_schedule_base_ms = field(v, "msbase", "settings").as_double();
      s.mtp_schedule_lambda = field(v, "mslam", "settings").as_double();
      s.mtp_schedule_min_depth = static_cast<int>(field(v, "msmin", "settings").as_int());
      // Records before the adaptive lambda (2026-09-14, later) carry no msad: fixed.
      if (const dgpp::minijson::Value* msad = v.find("msad")) s.mtp_schedule_adapt = msad->as_int() != 0;
      else s.mtp_schedule_adapt = false;
    }
    if (s.world < 2 || s.max_concurrency < 1 || s.kv_capacity < 1 ||
        (s.admission != "full" && s.admission != "grow") ||
        !latent_format_from_string(s.kv_dtype) ||
        (s.ngram_table != "resident" && s.ngram_table != "mmap") ||
        (s.dense_weights != "checkpoint" && s.dense_weights != "fp8") ||
        (s.prefill != "bounded" && s.prefill != "exact") ||
        (s.embed_sharding != "replicated" && s.embed_sharding != "vocab"))
      throw std::runtime_error("journal: settings record with impossible values");
    return rec;
  }
  if (op == "warm") {
    rec.warm = true;
    if (const dgpp::minijson::Value* adm = v.find("adm")) {
      const dgpp::minijson::Value& win = field(v, "win", "warm");
      if (!adm->is_number() || !win.is_number() || adm->as_int() < 0 ||
          adm->as_int() > 1 || win.as_int() < 1)
        throw std::runtime_error("journal: warm record with a bad admission policy");
      rec.has_admission = true;
      rec.admission.mode =
          static_cast<dgpp::sched::AdmissionPolicy::Mode>(adm->as_int());
      rec.admission.window_tokens = static_cast<int>(win.as_int());
      if (const auto* budget = v.find("pfbudget")) {
        if (!budget->is_number() || budget->as_int() < 0 || budget->as_int() > (1 << 30))
          throw std::runtime_error("journal: warm record with a bad prefill budget");
        rec.admission.prefill_budget_tokens = static_cast<int>(budget->as_int());
      }
      if (const auto* budget = v.find("pfidle")) {
        if (!budget->is_number() || budget->as_int() < 0 || budget->as_int() > (1 << 30))
          throw std::runtime_error("journal: warm record with a bad idle prefill budget");
        rec.admission.prefill_idle_budget_tokens = static_cast<int>(budget->as_int());
      }
    }
    if (const dgpp::minijson::Value* pc = v.find("pc")) {
      if (!pc->is_number() || pc->as_int() < 0)
        throw std::runtime_error("journal: warm record with a bad prefix slot count");
      rec.prefix_slots = static_cast<int>(pc->as_int());
    }
    if (const dgpp::minijson::Value* cfg = v.find("cfg")) {
      if (!cfg->is_string() || cfg->as_string().empty())
        throw std::runtime_error("journal: warm record with a bad config digest");
      rec.config_digest = std::string(cfg->as_string());
    }
    return rec;
  }
  if (op != "tick")
    throw std::runtime_error("journal: unknown op '" + op + "'");

  if (const dgpp::minijson::Value* s = v.find("s")) {
    if (!s->is_array())
      throw std::runtime_error("journal: 's' is not an array");
    for (const dgpp::minijson::Value& item : s->items()) {
      SchedulerRequest r;
      r.id = std::string(field(item, "id", "submit").as_string());
      if (r.id.empty())
        throw std::runtime_error("journal: submit with empty id");
      const dgpp::minijson::Value& p = field(item, "p", "submit");
      if (!p.is_array() || p.items().empty())
        throw std::runtime_error("journal: submit '" + r.id +
                                 "' has no prompt ids");
      for (const dgpp::minijson::Value& t : p.items()) {
        if (!t.is_number())
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has a non-numeric prompt id");
        r.prompt.push_back(t.as_int());
      }
      const dgpp::minijson::Value& m = field(item, "m", "submit");
      if (!m.is_number() || m.as_int() < 1)
        throw std::runtime_error("journal: submit '" + r.id +
                                 "' has bad max_steps");
      r.max_steps = static_cast<int>(m.as_int());
      if (const dgpp::minijson::Value* ca = item.find("ca")) {
        if (!ca->is_number() || ca->as_int() < 0)
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has bad cancel_after");
        r.cancel_after = static_cast<int>(ca->as_int());
      }
      if (const dgpp::minijson::Value* b = item.find("b")) {
        if (!b->is_array())
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has non-array boundaries");
        for (const dgpp::minijson::Value& t : b->items()) {
          if (!t.is_number())
            throw std::runtime_error("journal: submit '" + r.id +
                                     "' has a non-numeric boundary");
          r.boundaries.push_back(t.as_int());
        }
      }
      if (const dgpp::minijson::Value* nc = item.find("nc")) {
        if (!nc->is_number() || (nc->as_int() != 0 && nc->as_int() != 1))
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has a bad no-cache flag");
        r.no_cache = nc->as_int() == 1;
      }
      if (const dgpp::minijson::Value* lb = item.find("lb")) {
        if (!lb->is_array())
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has a non-array logit_bias");
        for (const dgpp::minijson::Value& e : lb->items()) {
          if (!e.is_array() || e.items().size() != 2 ||
              !e.items()[0].is_number() || !e.items()[1].is_number())
            throw std::runtime_error("journal: submit '" + r.id +
                                     "' has a bad logit_bias entry");
          const int64_t bits = e.items()[1].as_int();
          if (bits < 0 || bits > 0xFFFFFFFFll)
            throw std::runtime_error("journal: submit '" + r.id +
                                     "' has bad logit_bias bits");
          dgpp::sched::LogitBias b;
          b.token = static_cast<int32_t>(e.items()[0].as_int());
          b.bias = float_from_bits(static_cast<uint32_t>(bits));
          r.logit_bias.push_back(b);
        }
      }
      if (const dgpp::minijson::Value* lp = item.find("lp")) {
        if (!lp->is_number() || lp->as_int() < 0 || lp->as_int() > 20)
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has bad logprobs");
        r.logprobs = static_cast<int>(lp->as_int());
      }
      if (const dgpp::minijson::Value* g = item.find("g")) {
        if (!g->is_object())
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has a non-object sampling spec");
        const auto bits = [&](const char* name) -> uint32_t {
          const dgpp::minijson::Value& f = field(*g, name, "sampling spec");
          if (!f.is_number() || f.as_int() < 0 || f.as_int() > 0xFFFFFFFFll)
            throw std::runtime_error("journal: submit '" + r.id +
                                     "' has bad sampling field " + name);
          return static_cast<uint32_t>(f.as_int());
        };
        sample::Params& p = r.sampling;
        p.temperature = float_from_bits(bits("t"));
        p.top_p = float_from_bits(bits("p"));
        p.min_p = float_from_bits(bits("m"));
        p.repetition_penalty = float_from_bits(bits("r"));
        p.frequency_penalty = float_from_bits(bits("f"));
        p.presence_penalty = float_from_bits(bits("q"));
        const dgpp::minijson::Value& k = field(*g, "k", "sampling spec");
        const dgpp::minijson::Value& l = field(*g, "l", "sampling spec");
        if (!k.is_number() || !l.is_number())
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has bad top_k/logprobs");
        p.top_k = static_cast<int>(k.as_int());
        p.logprobs = static_cast<int>(l.as_int());
        const std::string seed_hex(
            field(*g, "s", "sampling spec").as_string());
        if (seed_hex.size() != 16 ||
            seed_hex.find_first_not_of("0123456789abcdef") != std::string::npos)
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has a bad seed");
        r.seed = std::stoull(seed_hex, nullptr, 16);
        try {
          sample::validate_params(p);
        } catch (const std::invalid_argument& e) {
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' carries an invalid sampling spec: " +
                                   e.what());
        }
        if (!(p.temperature > 0.0f) && r.logprobs < 0)
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' carries a greedy sampling spec");
        if (r.logprobs >= 0 && p.logprobs != r.logprobs)
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' logprobs fields disagree");
      }
      if (const dgpp::minijson::Value* gr = item.find("gr")) {
        if (!gr->is_object())
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has a non-object grammar");
        dgpp::text::GrammarSpec& g = r.grammar;
        const dgpp::minijson::Value& m = field(*gr, "m", "grammar");
        if (!m.is_number() || m.as_int() < 1 ||
            m.as_int() > static_cast<int64_t>(dgpp::text::GrammarSpec::Mode::kJsonOrTools))
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has a bad grammar mode");
        g.mode = static_cast<dgpp::text::GrammarSpec::Mode>(m.as_int());
        const dgpp::minijson::Value& pl = field(*gr, "pl", "grammar");
        if (!pl.is_bool())
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has a bad grammar parallel flag");
        g.parallel = pl.as_bool();
        g.named = std::string(field(*gr, "n", "grammar").as_string());
        const dgpp::minijson::Value& tools = field(*gr, "t", "grammar");
        if (!tools.is_array())
          throw std::runtime_error("journal: submit '" + r.id +
                                   "' has non-array grammar tools");
        for (const dgpp::minijson::Value& t : tools.items()) {
          dgpp::text::GrammarTool tool;
          tool.name = std::string(field(t, "n", "grammar tool").as_string());
          if (tool.name.empty())
            throw std::runtime_error("journal: submit '" + r.id +
                                     "' has a grammar tool without a name");
          if (const dgpp::minijson::Value* k = t.find("k")) {
            if (!k->is_array())
              throw std::runtime_error("journal: submit '" + r.id +
                                       "' has non-array grammar keys");
            tool.constrain_keys = true;
            for (const dgpp::minijson::Value& key : k->items())
              tool.keys.emplace_back(key.as_string());
          }
          if (const dgpp::minijson::Value* s = t.find("s")) {
            if (!s->is_bool())
              throw std::runtime_error("journal: submit '" + r.id +
                                       "' has a non-boolean grammar strict flag");
            tool.strict = s->as_bool();
          }
          if (const dgpp::minijson::Value* req = t.find("r")) {
            if (!req->is_array())
              throw std::runtime_error("journal: submit '" + r.id +
                                       "' has non-array grammar required keys");
            for (const dgpp::minijson::Value& key : req->items())
              tool.required_keys.emplace_back(key.as_string());
          }
          if (const dgpp::minijson::Value* args = t.find("a")) {
            if (!args->is_array())
              throw std::runtime_error("journal: submit '" + r.id +
                                       "' has non-array grammar arguments");
            for (const dgpp::minijson::Value& a : args->items()) {
              dgpp::text::GrammarArg arg;
              arg.key = std::string(field(a, "k", "grammar argument").as_string());
              if (const dgpp::minijson::Value* j = a.find("j")) {
                if (!j->is_string())
                  throw std::runtime_error("journal: submit '" + r.id +
                                           "' has a non-string argument schema");
                arg.kind = dgpp::text::GrammarArg::Kind::kJson;
                arg.schema = std::string(j->as_string());
              } else {
                const dgpp::minijson::Value& texts = field(a, "t", "grammar argument");
                if (!texts.is_array())
                  throw std::runtime_error("journal: submit '" + r.id +
                                           "' has non-array argument texts");
                arg.kind = dgpp::text::GrammarArg::Kind::kText;
                for (const dgpp::minijson::Value& x : texts.items())
                  arg.texts.emplace_back(x.as_string());
              }
              tool.args.push_back(std::move(arg));
            }
          }
          g.tools.push_back(std::move(tool));
        }
        if (g.mode == dgpp::text::GrammarSpec::Mode::kNamed) {
          bool found = false;
          for (const auto& t : g.tools) found = found || t.name == g.named;
          if (!found)
            throw std::runtime_error("journal: submit '" + r.id +
                                     "' names a function outside its tools");
        }
        // The JSON grammar (M6 6h) carries its schema text ("" = json_object).
        if (g.has_json()) {
          const dgpp::minijson::Value& js = field(*gr, "js", "grammar");
          if (!js.is_string())
            throw std::runtime_error("journal: submit '" + r.id +
                                     "' has a non-string JSON schema");
          g.json_schema = std::string(js.as_string());
        }
      }
      rec.submits.push_back(std::move(r));
    }
  }
  if (const dgpp::minijson::Value* c = v.find("c")) {
    if (!c->is_array())
      throw std::runtime_error("journal: 'c' is not an array");
    for (const dgpp::minijson::Value& id : c->items()) {
      if (id.as_string().empty())
        throw std::runtime_error("journal: empty cancel id");
      rec.cancels.emplace_back(id.as_string());
    }
  }
  if (const dgpp::minijson::Value* s = v.find("sp")) {
    if (!s->is_array())
      throw std::runtime_error("journal: 'sp' is not an array");
    for (const dgpp::minijson::Value& id : s->items()) {
      if (id.as_string().empty())
        throw std::runtime_error("journal: empty stop id");
      rec.stops.emplace_back(id.as_string());
    }
  }
  if (const dgpp::minijson::Value* pd = v.find("pd")) {
    if (!pd->is_string() || pd->as_string().empty())
      throw std::runtime_error("journal: 'pd' is not a digest string");
    const std::string text(pd->as_string());
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0')
      throw std::runtime_error("journal: 'pd' is not a decimal digest");
    rec.has_prefix_digest = true;
    rec.prefix_digest = static_cast<uint64_t>(value);
  }
  if (const dgpp::minijson::Value* od = v.find("od")) {
    if (!od->is_string() || od->as_string().empty())
      throw std::runtime_error("journal: 'od' is not a digest string");
    const std::string text(od->as_string());
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0')
      throw std::runtime_error("journal: 'od' is not a decimal digest");
    rec.has_op_digest = true;
    rec.op_digest = static_cast<uint64_t>(value);
  }
  return rec;
}

// ---- rank 0 ----------------------------------------------------------------

JournalWriter::JournalWriter(uint16_t listen_port)
    : listener_(dgpp::net::TcpListener::bind(listen_port)) {}

void JournalWriter::accept_peers(int world, int accept_timeout_ms) {
  const int peers = world - 1;
  peers_.reserve(static_cast<size_t>(peers));
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(accept_timeout_ms);
  for (int i = 1; i <= peers; ++i) {
    const int left_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now())
            .count());
    dgpp::net::TcpConn conn = listener_.accept(std::max(left_ms, 0));
    if (!conn.valid())
      throw std::runtime_error(
          "journal: only " + std::to_string(i - 1) + " of " +
          std::to_string(peers) + " peers connected within " +
          std::to_string(accept_timeout_ms) +
          "ms — a short world cannot serve (check the peers' logs)");
    // The peer's hello names the connection by rank; after it a peer never
    // writes, so a readable connection is a close or a reset (the death
    // watch's premise).
    std::string hello;
    while (hello.find('\n') == std::string::npos) {
      const int left = static_cast<int>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              deadline - std::chrono::steady_clock::now())
              .count());
      if (left <= 0 || !conn.wait_readable(left))
        throw std::runtime_error("journal: peer connection " + std::to_string(i) +
                                 " sent no hello within the window");
      char buf[64];
      const int n = conn.read_some(buf, sizeof buf);
      if (n <= 0)
        throw std::runtime_error("journal: peer connection " + std::to_string(i) +
                                 " closed before its hello");
      hello.append(buf, static_cast<size_t>(n));
      if (hello.size() > sizeof buf)
        throw std::runtime_error("journal: peer connection " + std::to_string(i) +
                                 " sent garbage instead of a hello");
    }
    int rank = -1;
    if (std::sscanf(hello.c_str(), "hello %d", &rank) != 1 || rank < 1 ||
        rank >= world ||
        std::find(peer_ranks_.begin(), peer_ranks_.end(), rank) != peer_ranks_.end())
      throw std::runtime_error("journal: bad hello from peer connection " +
                               std::to_string(i) + ": '" + hello + "'");
    peers_.push_back(std::move(conn));
    peer_ranks_.push_back(rank);
    DGPP_LOG_INFO("journal: rank {} connected ({}/{}, {}ms left in window)",
                  rank, i, peers, left_ms);
  }
  DGPP_LOG_INFO("journal: world complete ({} peer connection(s))", peers);
}

void JournalWriter::broadcast(const std::string& line) {
  const std::string framed = line + "\n";
  for (size_t i = 0; i < peers_.size(); ++i) {
    if (!peers_[i].write_all(framed.data(), framed.size()))
      throw std::runtime_error(
          "journal: rank " + std::to_string(peer_ranks_[i]) +
          "'s connection stopped reading — the fabric is broken; refusing "
          "to serve on without it");
  }
}

int JournalWriter::dead_peer() const {
  for (size_t i = 0; i < peers_.size(); ++i)
    if (peers_[i].peer_closed()) return peer_ranks_[i];
  return 0;
}

void JournalWriter::watch_peers(
    std::function<void(int, const std::string&)> on_death, int poll_ms) {
  stop_watch();
  watch_stop_.store(false);
  watch_ = std::thread([this, on_death = std::move(on_death), poll_ms] {
    while (!watch_stop_.load()) {
      const int dead = dead_peer();
      if (dead > 0) {
        if (on_death)
          on_death(dead, "its journal connection closed — the process is gone");
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
    }
  });
}

void JournalWriter::stop_watch() {
  watch_stop_.store(true);
  if (watch_.joinable()) watch_.join();
}

void JournalWriter::close_peers() {
  for (auto& p : peers_) p.close();
}

JournalWriter::~JournalWriter() { stop_watch(); }

// ---- peers ------------------------------------------------------------------

JournalReader::JournalReader(const std::string& host, uint16_t port,
                             int connect_timeout_ms, int rank)
    : conn_(journal_connect_with_retry(host, port, connect_timeout_ms)) {
  const std::string hello = "hello " + std::to_string(rank) + "\n";
  if (!conn_.write_all(hello.data(), hello.size()))
    throw std::runtime_error("journal: could not send the hello to rank 0");
}

bool JournalReader::read_line(const std::function<bool()>& should_stop,
                              std::string* line) {
  for (;;) {
    const size_t nl = pending_.find('\n');
    if (nl != std::string::npos) {
      // A complete record is honored even when the stop flag already
      // fired — the stop RECORD itself arrives this way.
      line->assign(pending_, 0, nl);
      pending_.erase(0, nl + 1);
      return true;
    }
    if (should_stop()) return false;
    // 250ms slices keep SIGINT latency bounded while idle blocking
    // stays cheap; read_some does the one-syscall-per-record work.
    if (!conn_.wait_readable(250)) continue;
    char buf[4096];
    const int n = conn_.read_some(buf, sizeof(buf));
    if (n == 0) return false;  // EOF: rank 0's journal is closed
    if (n < 0)
      return false;  // readable but recv fails (reset): world's over
    pending_.append(buf, static_cast<size_t>(n));
  }
}

bool wait_journal_settings(JournalReader* reader,
                           const std::function<bool()>& should_stop,
                           WorldSettings* out, const std::string& my_version) {
  std::string line;
  if (!reader->read_line(should_stop, &line)) {
    DGPP_LOG_INFO("journal: rank 0's stream ended before the settings record — exiting");
    return false;
  }
  const JournalRecord rec = decode_journal_line(line);
  if (rec.stop) {
    DGPP_LOG_INFO("journal: stop record before the settings record — exiting");
    return false;
  }
  if (!rec.settings)
    throw std::runtime_error(
        "journal: rank 0's first record was not the settings record — protocol "
        "order violated (§11); fabric emergency");
  if (!my_version.empty() && rec.world_settings.version != my_version)
    throw std::runtime_error("journal: rank 0 runs dgpp " + rec.world_settings.version +
                             ", this rank runs " + my_version +
                             " — a mixed-version world refuses to form");
  *out = rec.world_settings;
  return true;
}

bool wait_journal_warm(JournalReader* reader,
                       const std::function<bool()>& should_stop,
                       dgpp::sched::AdmissionPolicy* policy, int* prefix_slots,
                       std::string* config_digest) {
  std::string line;
  if (!reader->read_line(should_stop, &line)) {
    DGPP_LOG_INFO("journal: rank 0's stream ended before the warm record — "
                  "exiting");
    return false;
  }
  const JournalRecord rec = decode_journal_line(line);
  if (rec.stop) {
    DGPP_LOG_INFO("journal: stop record before the warm record — exiting");
    return false;
  }
  if (!rec.warm)
    throw std::runtime_error(
        "journal: rank 0 ticked before the warm record — protocol order "
        "violated (§11); fabric emergency");
  if (policy != nullptr && rec.has_admission) *policy = rec.admission;
  if (prefix_slots != nullptr) *prefix_slots = rec.prefix_slots;
  if (config_digest != nullptr) *config_digest = rec.config_digest;
  return true;
}

void run_journal_peer(Scheduler* sched, JournalReader* reader,
                      const std::function<bool()>& should_stop,
                      const std::function<void()>& on_rank0_death,
                      int watch_poll_ms,
                      const dgpp::sched::SchedulerObserver* oplog,
                      ThroughputLog* stats) {
  int64_t ticks = 0;  // records applied (the drift check names the tick)
  // The in-tick watch (the death discipline, fabric_serve.hpp): only while
  // the loop is inside a tick can rank 0's death go unseen by the read
  // loop, and in the lockstep protocol no record can be pending then, so
  // a closed connection is the only thing the probe can find.
  std::atomic<bool> in_tick{false};
  std::atomic<bool> watch_stop{false};
  std::thread watch;
  if (on_rank0_death) {
    watch = std::thread([&] {
      while (!watch_stop.load()) {
        if (in_tick.load() && reader->rank0_closed()) {
          on_rank0_death();
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(watch_poll_ms));
      }
    });
  }
  struct WatchJoin {
    std::atomic<bool>& stop;
    std::thread& t;
    ~WatchJoin() {
      stop.store(true);
      if (t.joinable()) t.join();
    }
  } watch_join{watch_stop, watch};
  struct TickScope {
    std::atomic<bool>& flag;
    explicit TickScope(std::atomic<bool>& f) : flag(f) { flag.store(true); }
    ~TickScope() { flag.store(false); }
  };
  for (;;) {
    std::string line;
    if (!reader->read_line(should_stop, &line)) {
      DGPP_LOG_INFO("journal: rank 0's stream ended — exiting");
      return;
    }
    // Not const: the submits below are std::move'd into the scheduler —
    // a const record would silently turn every "move" into a copy
    // (const rvalueref binds the copy ctor; GCC's redundant-move
    // warning is how this line got caught).
    JournalRecord rec = decode_journal_line(line);
    if (rec.stop) {
      DGPP_LOG_INFO("journal: stop record — exiting");
      return;
    }
    if (rec.warm)
      throw std::runtime_error(
          "journal: warm record inside the serving loop — protocol order "
          "violated (§11); fabric emergency");
    // The prefix cache's decisions of the previous tick, compared before
    // this one is applied (M7): a divergence dies here, one tick late.
    check_prefix_digest(rec, *sched);
    // The op stream's fold after the previous tick (M9): the same, for
    // every engine event this rank recorded.
    check_op_digest(rec, oplog, ticks);
    for (auto& r : rec.submits) {
      const std::string id = r.id;  // try_submit takes by value
      // Rank 0 admitted this against a queue state identical to ours
      // (same records, same order — §11); a refusal here means the
      // schedulers DIVERGED, which is the one bug this design exists
      // to make impossible. Loud death beats continuing with inconsistent state.
      if (!sched->try_submit(std::move(r)))
        throw std::runtime_error(
            "journal: could not admit '" + id + "' that rank 0 admitted — "
            "scheduler divergence (§11); fabric emergency");
    }
    for (const std::string& id : rec.cancels) sched->cancel(id);
    for (const std::string& id : rec.stops) sched->stop(id);
    {
      TickScope scope(in_tick);
      sched->tick();
    }
    ++ticks;
    if (stats) stats->observe(sched->meters(), nullptr);
  }
}

}  // namespace dgpp::serve
