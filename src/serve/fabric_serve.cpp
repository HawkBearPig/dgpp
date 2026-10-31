#include "serve/fabric_serve.hpp"
#include "serve/nvme_cache_check.hpp"

#include <format>

#include <cstdlib>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>

#include "common/base64.hpp"
#include "common/bf16_residency.hpp"
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

static void append_latency_tuning(std::string& out, const sched::PrefillLatencyTuning& tuning) {
  const sched::PrefillLatencyTuning defaults;
  if (tuning.headroom_pct != defaults.headroom_pct) out += ",\"pfhr\":" + std::to_string(tuning.headroom_pct);
  if (tuning.growth_factor != defaults.growth_factor) out += ",\"pfgrow\":" + std::to_string(tuning.growth_factor);
  if (tuning.recovery_pct != defaults.recovery_pct) out += ",\"pfrec\":" + std::to_string(tuning.recovery_pct);
  if (tuning.initial_tokens != defaults.initial_tokens) out += ",\"pfstart\":" + std::to_string(tuning.initial_tokens);
  if (tuning.idle_floor_tokens != defaults.idle_floor_tokens) out += ",\"pfifloor\":" + std::to_string(tuning.idle_floor_tokens);
}

std::string encode_journal_tick(const GenerationService::PassEvents& events) {
  std::string out = "{\"op\":\"tick\"";
  if (events.prefill_budget) out += ",\"pfmax\":" + std::to_string(*events.prefill_budget);
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
      // Omitted when false, so every record that predates ignore_eos is
      // byte-identical to what it was.
      if (r.ignore_eos) out += ",\"ie\":true";
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
      if (!r.cache_key.empty()) {
        out += ",\"ck\":";
        append_json_string(&out, r.cache_key);
      }
      if (!r.images.empty()) {
        validate_image_inputs(r.images, r.prompt.size());
        out += ",\"images\":[";
        for (size_t j = 0; j < r.images.size(); ++j) {
          const auto& im = r.images[j];
          if (j) out += ',';
          out += '[';
          append_json_int(&out, im.offset);
          out += ',';
          append_json_int(&out, im.tokens);
          out += ',';
          append_json_int(&out, im.width);
          out += ',';
          append_json_int(&out, im.height);
          out += ',';
          append_json_int(&out, im.grid);
          out += ',';
          append_json_string(
              &out, encode_base64(std::string_view(reinterpret_cast<const char*>(im.rgb.data()),
                                                   im.rgb.size())));
          out += ']';
        }
        out += ']';
      }

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
      // that ask for logprobs or carry penalties; a plain greedy request's record is
      // byte-identical to the pre-sampling format.
      if (r.logprobs >= 0) {
        out += ",\"lp\":";
        append_json_int(&out, r.logprobs);
      }
      if (r.sampling.temperature > 0.0f || r.logprobs >= 0 ||
          sample::needs_context_penalties(r.sampling)) {
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
  // The NVMe cold tier's commits (issue #26): [op, ok] pairs, op as a
  // decimal string (a uint64), applied before the tick on every rank.
  if (!events.disk_commits.empty()) {
    out += ",\"dk\":[";
    for (size_t i = 0; i < events.disk_commits.size(); ++i) {
      if (i != 0) out.push_back(',');
      out += "[\"" + std::to_string(events.disk_commits[i].op) + "\"," +
             (events.disk_commits[i].ok ? "1" : "0") + "]";
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

std::string encode_disk_status(uint64_t op, bool ok) {
  return "disk " + std::to_string(op) + " " + (ok ? "1" : "0");
}

bool decode_disk_status(std::string_view line, uint64_t* op, bool* ok) {
  unsigned long long o = 0;
  int verdict = -1;
  char tail = 0;
  const std::string text(line);
  if (std::sscanf(text.c_str(), "disk %llu %d%c", &o, &verdict, &tail) != 2) return false;
  if (o == 0 || (verdict != 0 && verdict != 1)) return false;
  *op = static_cast<uint64_t>(o);
  *ok = verdict == 1;
  return true;
}

void DiskCommitCollector::add(int rank, uint64_t op, bool ok) {
  if (op == 0) throw std::invalid_argument("DiskCommitCollector: op 0");
  if (rank < 0 || rank >= world_) throw std::out_of_range("DiskCommitCollector: rank outside the world");
  Pending& p = pending_[op];
  if (std::find(p.ranks.begin(), p.ranks.end(), rank) != p.ranks.end())
    throw std::runtime_error("DiskCommitCollector: rank " + std::to_string(rank) + " reported op " +
                             std::to_string(op) + " twice — the return path diverged");
  p.ranks.push_back(rank);
  ++p.reports;
  p.ok = p.ok && ok;
}

std::vector<dgpp::sched::Scheduler::DiskCommit> DiskCommitCollector::take_ready() {
  std::vector<dgpp::sched::Scheduler::DiskCommit> out;
  for (auto it = pending_.begin(); it != pending_.end();) {
    if (it->second.reports < world_) {
      ++it;
      continue;
    }
    out.push_back({it->first, it->second.ok});
    it = pending_.erase(it);
  }
  return out;
}
std::string encode_journal_warm(const dgpp::sched::AdmissionPolicy& policy,
                                int prefix_slots,
                                const std::string& config_digest, bool disk_adopt) {
  std::string out = "{\"op\":\"warm\",\"adm\":" +
                    std::to_string(static_cast<int>(policy.mode)) +
                    ",\"win\":" + std::to_string(policy.window_tokens) +
                    (prefix_slots > 0 ? ",\"pc\":" + std::to_string(prefix_slots) : "");
  // The effective configuration's digest: every peer compares
  // its own before serving; a config-less rank 0 writes none.
  if (policy.prefill_budget_tokens > 0) out += ",\"pfbudget\":" + std::to_string(policy.prefill_budget_tokens);
  if (policy.prefill_idle_budget_tokens > 0) out += ",\"pfidle\":" + std::to_string(policy.prefill_idle_budget_tokens);
  if (policy.decode_gap_target_ms > 0) out += ",\"gapms\":" + std::to_string(policy.decode_gap_target_ms);
  if (policy.first_token_target_ms > 0) out += ",\"ttftms\":" + std::to_string(policy.first_token_target_ms);
  append_latency_tuning(out, policy.prefill_latency);
  if (policy.prefill_budget_automatic) out += ",\"pfauto\":1";
  if (policy.prefill_early_admission) out += ",\"pfearly\":1";
  if (policy.prefix_min_tokens > 0) out += ",\"pmin\":" + std::to_string(policy.prefix_min_tokens);
  if (policy.prefix_head_snapshots) out += ",\"phead\":1";
  if (policy.prefix_pin_max_pct != 50) out += ",\"ppin\":" + std::to_string(policy.prefix_pin_max_pct);
  if (!config_digest.empty()) {
    out += ",\"cfg\":";
    append_json_string(&out, config_digest);
  }
  // The NVMe cache's previous index: every rank adopts it, or none does.
  if (disk_adopt) out += ",\"da\":1";
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
  if (s.prefill_budget_tokens != 0) out += ",\"pfbudget\":" + std::to_string(s.prefill_budget_tokens);
  if (s.prefill_idle_budget_tokens > 0) out += ",\"pfidle\":" + std::to_string(s.prefill_idle_budget_tokens);
  if (s.decode_gap_target_ms > 0) out += ",\"gapms\":" + std::to_string(s.decode_gap_target_ms);
  if (s.first_token_target_ms > 0) out += ",\"ttftms\":" + std::to_string(s.first_token_target_ms);
  append_latency_tuning(out, s.prefill_latency);
  if (!s.prefill_early_admission.has_value()) out += ",\"pfearly\":-1";
  else if (*s.prefill_early_admission) out += ",\"pfearly\":1";
  if (s.prefix_min_tokens > 0) out += ",\"pmin\":" + std::to_string(s.prefix_min_tokens);
  if (s.prefix_head_snapshots) out += ",\"phead\":1";
  if (s.prefix_pin_max_pct != 50) out += ",\"ppin\":" + std::to_string(s.prefix_pin_max_pct);
  out += std::format(",\"win\":{},\"pace\":{:.17g},\"inflight\":{},\"rdv\":{},\"stats\":{:.17g},\"ric\":{},\"kvdt\":",
      s.admission_window, s.bulk_pace_gbps, s.bulk_inflight, s.rendezvous_timeout_ms,
      s.stats_interval_s, s.reasoning_in_content ? 1 : 0);
  append_json_string(&out, s.kv_dtype);
  // The NVMe cold tier (issue #26): written only when on, so a record
  // without it is byte for byte the earlier format.
  if (s.nvme_cache)
    out += std::format(",\"nvme\":1,\"nvmegib\":{:.17g},\"nvmemin\":{},\"nvmechunk\":{},\"nvmeops\":{},\"nvmereserve\":{:.17g},\"nvmepersist\":{},\"nvmeto\":{}",
                       s.nvme_cache_capacity_gib, s.nvme_cache_min_tokens, s.nvme_cache_chunk_mib,
                       s.nvme_cache_max_ops, s.nvme_cache_reserve_gib, s.nvme_cache_persist ? 1 : 0,
                       s.nvme_cache_op_timeout_s);
  if (s.nvme_cache && !s.nvme_cache_reuse_binding.empty()) {
    out += ",\"nvmereuse\":";
    append_json_string(&out, s.nvme_cache_reuse_binding);
  }
  out += ",\"ngt\":";
  append_json_string(&out, s.ngram_table);
  out += ",\"dw\":";
  append_json_string(&out, s.dense_weights);
  out += ",\"fp8_head\":";
  append_json_string(&out, s.fp8_head);
  out += ",\"dh\":";
  append_json_string(&out, s.draft_head);
  out += std::format(",\"dhb\":{}", s.draft_head_blocks);
  out += ",\"mtpef\":";
  append_json_string(&out, s.mtp_expert_format);
  out += ",\"bfw\":";
  append_json_string(&out, s.bf16_weights);
  out += ",\"dv\":";
  append_json_string(&out, s.draft_vocab);
  out += std::format(",\"pfb16\":{},\"pffold\":{},\"pffp8\":{},\"pfpt\":{}", s.prefill_bf16_partials ? 1 : 0,
                     s.prefill_fold_scales ? 1 : 0, s.prefill_fp8_gemm ? 1 : 0,
                     s.prefill_fp8_per_tensor ? 1 : 0);
  out += ",\"pfact\":";
  append_json_string(&out, s.prefill_activations);
  out += ",\"pfattn\":";
  append_json_string(&out, s.prefill_attention);
  out += ",\"dcattn\":";
  append_json_string(&out, s.decode_attention);
  out += ",\"dfm\":";
  append_json_string(&out, s.dflash_model);
  out += std::format(",\"dfvg\":{},\"dfdb\":{},\"dfdp\":{}", s.dflash_verify_graph ? 1 : 0,
                     s.dflash_draft_batch ? 1 : 0, s.dflash_depth);
  out += ",\"dfw\":";
  append_json_string(&out, s.dflash_weights);
  out += std::format(",\"pfg\":{}", s.prefill_group ? 1 : 0);
  out += std::format(",\"l2p\":{},\"l2m\":{},\"l2w\":{},\"l2bw\":{}", s.l2_prefetch ? 1 : 0, s.l2_prefetch_merge ? 1 : 0,
                     s.l2_prefetch_window_mib, s.l2_prefetch_boundary_window_mib);
  out += ",\"l2f\":";
  append_json_string(&out, s.l2_prefetch_form);
  out += ",\"l2ff\":";
  append_json_string(&out, s.l2_prefetch_full_form);
  out += ",\"l2b\":";
  append_json_string(&out, s.l2_prefetch_boundary_rate);
  out += ",\"l2l\":";
  append_json_string(&out, s.l2_prefetch_layer_rate);
  out += ",\"xgemm\":";
  append_json_string(&out, s.expert_gemm);
  out += std::format(",\"xpf\":{},\"xtl\":{},\"xpair\":{},\"npre\":{}", s.expert_gemm_prefetch,
                     s.expert_tile_list ? 1 : 0, s.expert_gemm_pair ? 1 : 0, s.ngram_prestage ? 1 : 0);
  out += std::format(",\"compact\":{}", s.compact_batches ? 1 : 0);
  out += ",\"pf\":";
  append_json_string(&out, s.prefill);
  // The Dory1.0-4B loop keys (docs/dory_plan.md D1, D2): the family is
  // world-1, so no peer ever reads them — carried for the uniform rule
  // (every lever rides the settings record) and the digest. Records before
  // them carry none: the checkpoint's loops, per_loop.
  if (s.dory_loops > 0) out += ",\"dlp\":" + std::to_string(s.dory_loops);
  out += ",\"dkv\":";
  append_json_string(&out, s.dory_kv_cache);
  out += ",\"pfd\":";
  append_json_string(&out, s.prefill_dense);
  out += std::format(",\"pfo\":{}", s.prefill_fold_overlap ? 1 : 0);
  out += std::format(",\"pdl\":{}", s.decode_pdl ? 1 : 0);
  out += ",\"pfw\":";
  append_json_string(&out, s.prefill_window);
  out += ",\"dcw\":";
  append_json_string(&out, s.decode_window);
  out += ",\"ctr\":";
  append_json_string(&out, s.collective_transport);
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
  out += ",\"mtph\":";
  append_json_string(&out, s.mtp_head);
  out += ",\"attw\":";
  append_json_string(&out, s.attention_weights);
  // The DeepSeek-V4.1 index budget: absent when the checkpoint's (a record
  // from a plain run carries no key; an older peer decodes it to 0).
  if (s.index_topk > 0) out += std::format(",\"itopk\":{}", s.index_topk);
  if (s.candidate_topk_blocks > 0) out += std::format(",\"ctb\":{}", s.candidate_topk_blocks);
  if (s.candidate_block_size > 0) out += std::format(",\"cbs\":{}", s.candidate_block_size);
  out += std::format(",\"mss\":{},\"msrow\":{:.17g},\"msbase\":{:.17g},\"mslam\":{:.17g},\"msmin\":{},\"msad\":{},"
                     "\"msss\":{:.17g}",
                     s.mtp_schedule ? 1 : 0, s.mtp_schedule_row_ms, s.mtp_schedule_base_ms,
                     s.mtp_schedule_lambda, s.mtp_schedule_min_depth, s.mtp_schedule_adapt ? 1 : 0,
                     s.mtp_schedule_sampled_scale);
  // The depth-option cap (2026-10-23): absent when 0 (an older peer decodes
  // a missing key to 0; a peer without the key would differ in the digest
  // and refuse the world, as every setting does).
  if (s.mtp_schedule_options > 0) out += std::format(",\"msopt\":{}", s.mtp_schedule_options);
  out += ",\"mdr\":";
  append_json_string(&out, s.mtp_draft);
  out += std::format(",\"mdt\":{:.17g}", s.mtp_draft_temperature);
  out += ",\"mvf\":";
  append_json_string(&out, s.mtp_verify);
  out += ",\"deng\":";
  append_json_string(&out, s.decode_engine);
  out += std::format(",\"ssm\":{},\"l2s\":{},\"sgov\":{}", s.step_sms, s.l2_staging_mib, s.stream_governor);
  out += std::format(",\"dbr\":{},\"lk\":{},\"lkn\":{},\"lks\":{},\"lka\":{},\"lkt\":{}", s.dflash_batch_rows,
                     s.lookup_draft ? 1 : 0, s.lookup_nmin, s.lookup_nstrong, s.lookup_agree, s.lookup_tail);
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

static int optional_policy_integer(const dgpp::minijson::Value& record, const char* key,
                                   int minimum, int maximum) {
  const auto* value = record.find(key);
  if (!value) return 0;
  if (!value->is_number() || !std::isfinite(value->as_double()) ||
      value->as_double() < minimum || value->as_double() > maximum ||
      std::floor(value->as_double()) != value->as_double())
    throw std::runtime_error(std::string("journal: invalid policy field ") + key);
  return static_cast<int>(value->as_int());
}

static sched::PrefillLatencyTuning read_latency_tuning(const dgpp::minijson::Value& record) {
  sched::PrefillLatencyTuning tuning;
  if (record.find("pfhr")) tuning.headroom_pct = optional_policy_integer(record, "pfhr", 0, 50);
  if (record.find("pfgrow")) tuning.growth_factor = optional_policy_integer(record, "pfgrow", 2, 4);
  if (record.find("pfrec")) tuning.recovery_pct = optional_policy_integer(record, "pfrec", 1, 100);
  if (record.find("pfstart")) tuning.initial_tokens = optional_policy_integer(record, "pfstart", 1, 1 << 30);
  if (record.find("pfifloor")) tuning.idle_floor_tokens = optional_policy_integer(record, "pfifloor", 0, 1 << 30);
  return tuning;
}

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
    if (v.find("compact")) s.compact_batches = flag("compact");
    s.mtp = flag("mtp");
    s.mtp_depth = static_cast<int>(num("mtpd").as_int());
    s.graph_batch_min_live = static_cast<int>(num("batchmin").as_int());
    s.sampling_candidates = static_cast<int>(num("cand").as_int());
    s.prefix_cache_gib = num("pcgib").as_double();
    s.admission = std::string(field(v, "adm", "settings").as_string());
    s.admission_window = static_cast<int>(num("win").as_int());
    s.decode_gap_target_ms = optional_policy_integer(v, "gapms", 0, sched::kMaxLatencyTargetMs);
    s.first_token_target_ms = optional_policy_integer(v, "ttftms", 0, sched::kMaxLatencyTargetMs);
    s.prefill_latency = read_latency_tuning(v);
    if (const auto* budget = v.find("pfbudget")) {
      if (!budget->is_number() || budget->as_int() < -1 || budget->as_int() > (1 << 30))
        throw std::runtime_error("journal: settings record with a bad prefill budget");
      s.prefill_budget_tokens = static_cast<int>(budget->as_int());
    }
    if (const auto* budget = v.find("pfidle")) {
      if (!budget->is_number() || budget->as_int() < 0 || budget->as_int() > (1 << 30))
        throw std::runtime_error("journal: settings record with a bad idle prefill budget");
      s.prefill_idle_budget_tokens = static_cast<int>(budget->as_int());
    }
    if (const auto* early = v.find("pfearly")) {
      if (!early->is_number() || (early->as_double() != -1 && early->as_double() != 0 && early->as_double() != 1))
        throw std::runtime_error("journal: settings record with a bad early admission mode");
      s.prefill_early_admission = early->as_int() == -1 ? std::nullopt
                                                     : std::optional<bool>(early->as_int() == 1);
    }
    // The prefix cache's entry policy (2026-09-28): records before it carry
    // neither key — no floor, no head cut, what such a rank 0 ran.
    if (const auto* pmin = v.find("pmin")) {
      if (!pmin->is_number() || pmin->as_int() < 0 || pmin->as_int() > (1 << 30))
        throw std::runtime_error("journal: settings record with a bad prefix entry floor");
      s.prefix_min_tokens = static_cast<int>(pmin->as_int());
    }
    if (v.find("phead")) s.prefix_head_snapshots = flag("phead");
    if (const auto* ppin = v.find("ppin")) {
      if (!ppin->is_number() || ppin->as_int() < 0 || ppin->as_int() > 100)
        throw std::runtime_error("journal: settings record with a bad prefix pin cap");
      s.prefix_pin_max_pct = static_cast<int>(ppin->as_int());
    }
    s.bulk_pace_gbps = num("pace").as_double();
    s.bulk_inflight = static_cast<int>(num("inflight").as_int());
    s.rendezvous_timeout_ms = static_cast<int>(num("rdv").as_int());
    s.stats_interval_s = num("stats").as_double();
    s.reasoning_in_content = flag("ric");
    s.kv_dtype = std::string(field(v, "kvdt", "settings").as_string());
    // The NVMe cold tier (issue #26): absent = off.
    if (v.find("nvme")) {
      s.nvme_cache = flag("nvme");
      s.nvme_cache_capacity_gib = num("nvmegib").as_double();
      if (!(s.nvme_cache_capacity_gib >= 0.0) || (s.nvme_cache && !(s.nvme_cache_capacity_gib > 0.0)))
        throw std::runtime_error("journal: settings record with a bad NVMe cache capacity");
      s.nvme_cache_min_tokens = num("nvmemin").as_int();
      if (s.nvme_cache_min_tokens < 0)
        throw std::runtime_error("journal: settings record with a bad NVMe cache minimum");
      // The pacing, op bound and reserve keys (absent in records before
      // 2026-10-27: the defaults they ran with).
      if (v.find("nvmechunk")) {
        s.nvme_cache_chunk_mib = num("nvmechunk").as_int();
        s.nvme_cache_max_ops = num("nvmeops").as_int();
        s.nvme_cache_reserve_gib = num("nvmereserve").as_double();
        if (s.nvme_cache_chunk_mib < 4 || s.nvme_cache_chunk_mib > 4096 || s.nvme_cache_max_ops < 1 ||
            s.nvme_cache_max_ops > 256 || !(s.nvme_cache_reserve_gib >= 0.0))
          throw std::runtime_error("journal: settings record with a bad NVMe cache chunk, op bound or reserve");
        s.nvme_cache_persist = v.find("nvmepersist") ? flag("nvmepersist") : true;
        // The op bound (absent in records before 2026-10-28: never).
        if (v.find("nvmeto")) {
          s.nvme_cache_op_timeout_s = num("nvmeto").as_int();
          if (s.nvme_cache_op_timeout_s < 0 || s.nvme_cache_op_timeout_s > 86400)
            throw std::runtime_error("journal: settings record with a bad NVMe cache op timeout");
        } else {
          s.nvme_cache_op_timeout_s = 0;
        }
      }
    }
    if (const auto* reuse = v.find("nvmereuse")) {
      if (!reuse->is_string() || !valid_nvme_cache_binding(std::string(reuse->as_string())) ||
          !s.nvme_cache || !s.nvme_cache_persist)
        throw std::runtime_error("journal: settings record with a bad NVMe cache reuse binding");
      s.nvme_cache_reuse_binding = std::string(reuse->as_string());
    }
    // Records before 2026-09-10 carry no table residency: resident.
    if (const dgpp::minijson::Value* ngt = v.find("ngt")) s.ngram_table = std::string(ngt->as_string());
    if (const auto* head = v.find("fp8_head")) {
      if (!head->is_string() || (head->as_string() != "gemv" && head->as_string() != "mma"))
        throw std::runtime_error("journal: engine.fp8_head must be gemv or mma");
      s.fp8_head = std::string(head->as_string());
    }
    // The draft head's read (2026-10-13): records before it carry none (exact).
    if (const auto* dh = v.find("dh")) {
      if (!dh->is_string() || (dh->as_string() != "exact" && dh->as_string() != "screen"))
        throw std::runtime_error("journal: engine.draft_head must be exact or screen");
      s.draft_head = std::string(dh->as_string());
    }
    if (const auto* dhb = v.find("dhb")) {
      if (!dhb->is_number()) throw std::runtime_error("journal: engine.draft_head_blocks must be a number");
      s.draft_head_blocks = static_cast<int>(dhb->as_int());
    }
    if (const dgpp::minijson::Value* dw = v.find("dw")) s.dense_weights = std::string(dw->as_string());
    if (const dgpp::minijson::Value* mtpef = v.find("mtpef")) s.mtp_expert_format = std::string(mtpef->as_string());
    // The bf16 weights' form (2026-09-19): records before it carry none.
    if (const dgpp::minijson::Value* bfw = v.find("bfw")) s.bf16_weights = std::string(bfw->as_string());
    // The draft vocabulary slice (2026-09-29): records before it carry none.
    if (const dgpp::minijson::Value* dv = v.find("dv")) s.draft_vocab = std::string(dv->as_string());
    // The opt-in prefill levers (2026-09-30): records before them carry none (off).
    if (v.find("pfb16")) s.prefill_bf16_partials = flag("pfb16");
    if (v.find("pffold")) s.prefill_fold_scales = flag("pffold");
    if (v.find("pffp8")) s.prefill_fp8_gemm = flag("pffp8");
    if (v.find("pfpt")) s.prefill_fp8_per_tensor = flag("pfpt");  // 2026-10-03: the 27B recipe
    if (const dgpp::minijson::Value* pa = v.find("pfact")) s.prefill_activations = std::string(pa->as_string());
    // The prefill attention's arithmetic (2026-10-16): records before it carry none (bf16).
    if (const dgpp::minijson::Value* pt = v.find("pfattn")) s.prefill_attention = std::string(pt->as_string());
    if (const dgpp::minijson::Value* dt = v.find("dcattn")) s.decode_attention = std::string(dt->as_string());
    // The DFlash2 drafter (2026-10-03): records before it carry none.
    if (const dgpp::minijson::Value* dfm = v.find("dfm")) s.dflash_model = std::string(dfm->as_string());
    if (v.find("dfvg")) s.dflash_verify_graph = flag("dfvg");
    if (v.find("dfdb")) s.dflash_draft_batch = flag("dfdb");
    if (const dgpp::minijson::Value* dfdp = v.find("dfdp")) s.dflash_depth = static_cast<int>(dfdp->as_int());
    // Records before 2026-10-05 carry no drafter weight format: the checkpoint's.
    if (const dgpp::minijson::Value* dfw = v.find("dfw")) s.dflash_weights = std::string(dfw->as_string());
    if (s.dflash_weights != "checkpoint" && s.dflash_weights != "fp8")
      throw std::runtime_error("worker settings: dfw must be checkpoint or fp8");
    if (v.find("pfg")) s.prefill_group = flag("pfg");  // records before 2026-10-05: grouped
    // The L2 prefetcher (2026-10-05); records before it carry the defaults.
    if (v.find("l2p")) s.l2_prefetch = flag("l2p");
    if (v.find("l2m")) s.l2_prefetch_merge = flag("l2m");
    if (const dgpp::minijson::Value* w = v.find("l2w")) s.l2_prefetch_window_mib = static_cast<int>(w->as_int());
    if (const dgpp::minijson::Value* w = v.find("l2bw")) s.l2_prefetch_boundary_window_mib = static_cast<int>(w->as_int());
    if (const dgpp::minijson::Value* f = v.find("l2f")) s.l2_prefetch_form = std::string(f->as_string());
    if (const dgpp::minijson::Value* f = v.find("l2ff")) s.l2_prefetch_full_form = std::string(f->as_string());
    if (const dgpp::minijson::Value* r = v.find("l2b")) s.l2_prefetch_boundary_rate = std::string(r->as_string());
    if (const dgpp::minijson::Value* r = v.find("l2l")) s.l2_prefetch_layer_rate = std::string(r->as_string());
    // The expert GEMM's form and companions (2026-09-30): records before them carry the defaults.
    if (const dgpp::minijson::Value* xg = v.find("xgemm")) s.expert_gemm = std::string(xg->as_string());
    if (const dgpp::minijson::Value* xpf = v.find("xpf")) s.expert_gemm_prefetch = static_cast<int>(xpf->as_int());
    if (v.find("xtl")) s.expert_tile_list = flag("xtl");
    if (v.find("xpair")) s.expert_gemm_pair = flag("xpair");
    if (v.find("npre")) s.ngram_prestage = flag("npre");
    // Records before 2026-09-14 carry no prefill mode: bounded.
    if (const dgpp::minijson::Value* pf = v.find("pf")) s.prefill = std::string(pf->as_string());
    // The Dory loop keys (2026-10-18): records before them carry none.
    if (const dgpp::minijson::Value* dlp = v.find("dlp")) s.dory_loops = static_cast<int>(dlp->as_int());
    if (const dgpp::minijson::Value* dkv = v.find("dkv")) {
      s.dory_kv_cache = std::string(dkv->as_string());
      if (s.dory_kv_cache != "per_loop" && s.dory_kv_cache != "last_loop")
        throw std::runtime_error("journal: engine.dory_kv_cache must be per_loop or last_loop");
    }
    // Older records carry no dense form: preserve each family's default.
    if (const dgpp::minijson::Value* pfd = v.find("pfd")) s.prefill_dense = std::string(pfd->as_string());
    else s.prefill_dense = "auto";
    // Records before the fold overlap key (2026-10-18) carry none: on.
    s.prefill_fold_overlap = v.find("pfo") ? flag("pfo") : true;
    s.decode_pdl = v.find("pdl") ? flag("pdl") : true;
    // Records before the window kernel key (2026-10-18) carry none: flash.
    if (const dgpp::minijson::Value* pfw = v.find("pfw")) s.prefill_window = std::string(pfw->as_string());
    else s.prefill_window = "flash";
    if (const dgpp::minijson::Value* dcw = v.find("dcw")) s.decode_window = std::string(dcw->as_string());
    else s.decode_window = "flash";
    // Records before the transport key (2026-10-19) carry none: send,
    // the form every such binary runs.
    if (const dgpp::minijson::Value* ctr = v.find("ctr")) s.collective_transport = std::string(ctr->as_string());
    else s.collective_transport = "send";
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
    // Records before 2026-10-09 carry no draft-head form: the checkpoint's.
    if (const dgpp::minijson::Value* mh = v.find("mtph")) s.mtp_head = std::string(mh->as_string());
    if (const dgpp::minijson::Value* aw = v.find("attw")) s.attention_weights = std::string(aw->as_string());
    // Records before 2026-10-18 carry no index budget: the checkpoint's.
    const auto budget = [&](const char* key, int hi) {
      const dgpp::minijson::Value* b = v.find(key);
      if (!b) return 0;
      if (!b->is_number() || b->as_int() < 1 || b->as_int() > hi)
        throw std::runtime_error(std::string("journal: settings record with a bad index budget ") + key);
      return static_cast<int>(b->as_int());
    };
    s.index_topk = budget("itopk", 2048);
    s.candidate_topk_blocks = budget("ctb", 2048);
    s.candidate_block_size = budget("cbs", 1024);
    // Records before 2026-09-14 carry no scheduled verify depth: off.
    if (const dgpp::minijson::Value* mss = v.find("mss")) {
      s.mtp_schedule = mss->as_int() != 0;
      s.mtp_schedule_row_ms = field(v, "msrow", "settings").as_double();
      s.mtp_schedule_base_ms = field(v, "msbase", "settings").as_double();
      s.mtp_schedule_lambda = field(v, "mslam", "settings").as_double();
      s.mtp_schedule_min_depth = static_cast<int>(field(v, "msmin", "settings").as_int());
      if (const dgpp::minijson::Value* msopt = v.find("msopt")) s.mtp_schedule_options = static_cast<int>(msopt->as_int());
      else s.mtp_schedule_options = 0;
      // Records before the adaptive lambda (2026-09-14, later) carry no msad: fixed.
      if (const dgpp::minijson::Value* msad = v.find("msad")) s.mtp_schedule_adapt = msad->as_int() != 0;
      else s.mtp_schedule_adapt = false;
      // Records before 2026-10-02 carry no sampled scale: sampled requests
      // verify the whole block.
      if (const dgpp::minijson::Value* msss = v.find("msss")) s.mtp_schedule_sampled_scale = msss->as_double();
      else s.mtp_schedule_sampled_scale = 0.0;
      if (!(s.mtp_schedule_sampled_scale >= 0.0 && s.mtp_schedule_sampled_scale <= 1.0))
        throw std::runtime_error("worker settings: msss must be in [0, 1]");
    }
    // Records before 2026-10-01 carry no draft rule: the family's default.
    if (const dgpp::minijson::Value* mdr = v.find("mdr")) s.mtp_draft = std::string(mdr->as_string());
    if (s.mtp_draft != "auto" && s.mtp_draft != "sampled" && s.mtp_draft != "greedy")
      throw std::runtime_error("worker settings: mdr must be auto, sampled or greedy");
    // Records before 2026-10-05 carry no draft temperature: the request's.
    if (const dgpp::minijson::Value* mdt = v.find("mdt")) s.mtp_draft_temperature = mdt->as_double();
    if (!(s.mtp_draft_temperature > 0.0 && s.mtp_draft_temperature <= 4.0))
      throw std::runtime_error("worker settings: mdt must be in (0, 4]");
    // Records before 2026-10-05 carry no verify rule: the token rule.
    if (const dgpp::minijson::Value* mvf = v.find("mvf")) s.mtp_verify = std::string(mvf->as_string());
    if (s.mtp_verify != "token" && s.mtp_verify != "block")
      throw std::runtime_error("worker settings: mvf must be token or block");
    // Records before the step engine's key carry none: the graph engines.
    if (const dgpp::minijson::Value* deng = v.find("deng")) s.decode_engine = std::string(deng->as_string());
    if (s.decode_engine != "graph" && s.decode_engine != "step")
      throw std::runtime_error("worker settings: deng must be graph or step");
    if (const dgpp::minijson::Value* ssm = v.find("ssm")) s.step_sms = static_cast<int>(ssm->as_int());
    if (const dgpp::minijson::Value* l2s = v.find("l2s")) s.l2_staging_mib = static_cast<int>(l2s->as_int());
    if (const dgpp::minijson::Value* sgov = v.find("sgov")) s.stream_governor = static_cast<int>(sgov->as_int());
    if (s.step_sms < 0 || s.step_sms > 256 || s.l2_staging_mib < 0 || s.l2_staging_mib > 24 || s.stream_governor < 0 ||
        s.stream_governor > 256)
      throw std::runtime_error("worker settings: ssm, l2s or sgov out of range");
    if (const dgpp::minijson::Value* dbr = v.find("dbr")) s.dflash_batch_rows = static_cast<int>(dbr->as_int());
    if (s.dflash_batch_rows < 0) throw std::runtime_error("worker settings: dbr must be >= 0");
    // Records before context-lookup drafting carry none: off with defaults.
    if (v.find("lk")) s.lookup_draft = flag("lk");
    if (const dgpp::minijson::Value* lkn = v.find("lkn")) s.lookup_nmin = static_cast<int>(lkn->as_int());
    if (const dgpp::minijson::Value* lks = v.find("lks")) s.lookup_nstrong = static_cast<int>(lks->as_int());
    if (const dgpp::minijson::Value* lka = v.find("lka")) s.lookup_agree = static_cast<int>(lka->as_int());
    // Records before the lookup tail carry none: the MTP block alone.
    if (const dgpp::minijson::Value* lkt = v.find("lkt")) s.lookup_tail = static_cast<int>(lkt->as_int());
    if (s.lookup_tail < 0) throw std::runtime_error("worker settings: lkt must be >= 0");
    if (!(s.lookup_nmin >= 1 && s.lookup_nstrong >= s.lookup_nmin && s.lookup_agree >= 0))
      throw std::runtime_error("worker settings: lookup thresholds must satisfy 1 <= lkn <= lks");
    if (s.world < 2 || s.max_concurrency < 1 || s.kv_capacity < 1 ||
        (s.admission != "full" && s.admission != "grow") ||
        !latent_format_from_string(s.kv_dtype) ||
        (s.ngram_table != "resident" && s.ngram_table != "mmap") ||
        (s.dense_weights != "checkpoint" && s.dense_weights != "fp8") ||
        !parse_bf16_residency(s.bf16_weights, nullptr) ||
        (s.prefill != "bounded" && s.prefill != "exact") ||
        (s.prefill_dense != "auto" && s.prefill_dense != "tile" && s.prefill_dense != "streaming") ||
        (s.prefill_window != "flash" && s.prefill_window != "scalar") ||
        (s.decode_window != "flash" && s.decode_window != "scalar") ||
        (s.collective_transport != "write" && s.collective_transport != "send") ||
        (s.embed_sharding != "replicated" && s.embed_sharding != "vocab") ||
        (s.mtp_head != "checkpoint" && s.mtp_head != "fp8") ||
        (s.prefill_attention != "bf16" && s.prefill_attention != "fp8") ||
        (s.decode_attention != "bf16" && s.decode_attention != "fp8") ||
        (s.attention_weights != "checkpoint" && s.attention_weights != "int4"))
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
      rec.admission.decode_gap_target_ms = optional_policy_integer(v, "gapms", 0, sched::kMaxLatencyTargetMs);
      rec.admission.first_token_target_ms = optional_policy_integer(v, "ttftms", 0, sched::kMaxLatencyTargetMs);
      rec.admission.prefill_latency = read_latency_tuning(v);
      rec.admission.prefill_budget_automatic = optional_policy_integer(v, "pfauto", 0, 1) != 0;
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
      if (const auto* early = v.find("pfearly")) {
        if (!early->is_number() || (early->as_int() != 0 && early->as_int() != 1))
          throw std::runtime_error("journal: warm record with a bad early-prefill-admission flag");
        rec.admission.prefill_early_admission = early->as_int() == 1;
      }
      if (const auto* pmin = v.find("pmin")) {
        if (!pmin->is_number() || pmin->as_int() < 0 || pmin->as_int() > (1 << 30))
          throw std::runtime_error("journal: warm record with a bad prefix entry floor");
        rec.admission.prefix_min_tokens = static_cast<int>(pmin->as_int());
      }
      if (const auto* phead = v.find("phead")) {
        if (!phead->is_number() || (phead->as_int() != 0 && phead->as_int() != 1))
          throw std::runtime_error("journal: warm record with a bad head-cut flag");
        rec.admission.prefix_head_snapshots = phead->as_int() == 1;
      }
      if (const auto* ppin = v.find("ppin")) {
        if (!ppin->is_number() || ppin->as_int() < 0 || ppin->as_int() > 100)
          throw std::runtime_error("journal: warm record with a bad prefix pin cap");
        rec.admission.prefix_pin_max_pct = static_cast<int>(ppin->as_int());
      }
    }
    if (const dgpp::minijson::Value* pc = v.find("pc")) {
      if (!pc->is_number() || pc->as_int() < 0)
        throw std::runtime_error("journal: warm record with a bad prefix slot count");
      rec.prefix_slots = static_cast<int>(pc->as_int());
    }
    if (const dgpp::minijson::Value* da = v.find("da")) {
      if (!da->is_number() || da->as_int() != 1) throw std::runtime_error("journal: warm record with a bad 'da'");
      rec.disk_adopt = true;
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

  if (v.find("pfmax")) rec.prefill_budget = optional_policy_integer(v, "pfmax", 1, 1 << 30);
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
      if (const dgpp::minijson::Value* ie = item.find("ie")) {
        if (!ie->is_bool())
          throw std::runtime_error("journal: submit '" + r.id + "' has bad ignore_eos");
        r.ignore_eos = ie->as_bool();
      }
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
      if (const dgpp::minijson::Value* ck = item.find("ck")) {
        if (!ck->is_string() || ck->as_string().empty())
          throw std::runtime_error("journal: submit '" + r.id + "' has a bad cache key");
        r.cache_key = std::string(ck->as_string());
      }

      if (const auto* images = item.find("images")) {
        if (!images->is_array() || images->items().size() > r.prompt.size())
          throw std::runtime_error("journal: invalid image list");
        size_t image_bytes = 0;
        for (const auto& entry : images->items()) {
          if (!entry.is_array() || (entry.items().size() != 5 && entry.items().size() != 6))
            throw std::runtime_error("journal: invalid image entry");
          const auto& a = entry.items();
          for (size_t j = 0; j + 1 < a.size(); ++j)
            if (!a[j].is_number() || !std::isfinite(a[j].as_double()) || a[j].as_double() < 0 ||
                a[j].as_double() > INT32_MAX || a[j].as_double() != std::floor(a[j].as_double()))
              throw std::runtime_error("journal: invalid image dimension or span");
          if (!a.back().is_string()) throw std::runtime_error("journal: invalid image pixels");
          ImageInput im;
          im.offset = a[0].as_int();
          im.tokens = static_cast<int>(a[1].as_int());
          im.width = static_cast<int>(a[2].as_int());
          im.height = static_cast<int>(a[3].as_int());
          // A five-field entry predates the per-family pixel grid.
          im.grid = a.size() == 6 ? static_cast<int>(a[4].as_int()) : 28;
          const auto bytes =
              decode_base64(a.back().as_string(), kMaxRequestImageBytes - image_bytes);
          image_bytes += bytes.size();
          im.rgb.assign(bytes.begin(), bytes.end());
          r.images.push_back(std::move(im));
        }
        validate_image_inputs(r.images, r.prompt.size());
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
        if (!(p.temperature > 0.0f) && r.logprobs < 0 && !sample::needs_context_penalties(p))
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
  if (const dgpp::minijson::Value* dk = v.find("dk")) {
    if (!dk->is_array())
      throw std::runtime_error("journal: 'dk' is not an array");
    for (const dgpp::minijson::Value& item : dk->items()) {
      if (!item.is_array() || item.items().size() != 2 || !item.items()[0].is_string() ||
          !item.items()[1].is_number())
        throw std::runtime_error("journal: a 'dk' commit is not [op, ok]");
      const std::string text(item.items()[0].as_string());
      char* end = nullptr;
      const unsigned long long op = std::strtoull(text.c_str(), &end, 10);
      if (text.empty() || end == nullptr || *end != '\0' || op == 0)
        throw std::runtime_error("journal: a 'dk' commit's op is not a decimal id");
      const int64_t ok = item.items()[1].as_int();
      if (ok != 0 && ok != 1) throw std::runtime_error("journal: a 'dk' commit's verdict is not 0 or 1");
      rec.disk_commits.push_back({static_cast<uint64_t>(op), ok == 1});
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
    // The peer's hello names the connection by rank. Its readiness message
    // follows settings and model construction, and is drained before warm.
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

bool JournalWriter::wait_peers_ready(int timeout_ms, const std::function<bool()>& should_stop,
                                     std::vector<std::pair<int, std::string>>* notes) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  std::vector<std::string> received(peers_.size());
  std::vector<bool> ready(peers_.size(), false);
  size_t remaining = peers_.size();
  DGPP_LOG_INFO("journal: waiting for {} peer model(s) to be ready", remaining);
  while (remaining != 0) {
    if (should_stop && should_stop()) return false;
    for (size_t i = 0; i < peers_.size(); ++i) {
      const std::string name = "journal: rank " + std::to_string(peer_ranks_[i]);
      if (peers_[i].peer_closed())
        throw std::runtime_error(name + " closed before model readiness completed");
      if (ready[i]) continue;
      if (peers_[i].wait_readable(0)) {
        char buf[64];
        const int n = peers_[i].read_some(buf, sizeof buf);
        if (n <= 0) throw std::runtime_error(name + " closed before model readiness completed");
        received[i].append(buf, static_cast<size_t>(n));
        // "ready\n", or "ready <note>\n" (the NVMe cache's candidate digest).
        const size_t nl = received[i].find('\n');
        const std::string head = received[i].substr(0, std::min<size_t>(received[i].size(), 6));
        const bool plausible = std::string_view("ready\n").starts_with(head) ||
                               std::string_view("ready ").starts_with(head);
        if (!plausible || received[i].size() > 96)
          throw std::runtime_error(name + " sent an invalid model readiness message");
        if (nl != std::string::npos) {
          const std::string line = received[i].substr(0, nl);
          if (line != "ready" && (line.rfind("ready ", 0) != 0 || line.size() == 6))
            throw std::runtime_error(name + " sent an invalid model readiness message");
          if (notes != nullptr) notes->emplace_back(peer_ranks_[i], line.size() > 6 ? line.substr(6) : "");
          ready[i] = true;
          --remaining;
          DGPP_LOG_INFO("journal: rank {} model ready ({} peer(s) remaining)", peer_ranks_[i], remaining);
        }
      }
    }
    if (remaining == 0) break;
    if (std::chrono::steady_clock::now() >= deadline) {
      std::string missing;
      for (size_t i = 0; i < peers_.size(); ++i)
        if (!ready[i]) missing += (missing.empty() ? "" : ", ") + std::to_string(peer_ranks_[i]);
      throw std::runtime_error("journal: model readiness timed out after " + std::to_string(timeout_ms) +
                               "ms waiting for rank(s) " + missing);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return !(should_stop && should_stop());
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

std::vector<std::pair<int, std::string>> JournalWriter::poll_peer_lines() {
  std::vector<std::pair<int, std::string>> out;
  if (peer_pending_.size() != peers_.size()) peer_pending_.resize(peers_.size());
  for (size_t i = 0; i < peers_.size(); ++i) {
    std::string& pending = peer_pending_[i];
    for (int rounds = 0; rounds < 64 && peers_[i].wait_readable(0); ++rounds) {
      char buf[4096];
      const int n = peers_[i].read_some(buf, sizeof buf);
      if (n <= 0) break;  // closed or reset: the death watch's finding
      pending.append(buf, static_cast<size_t>(n));
    }
    size_t nl;
    while ((nl = pending.find('\n')) != std::string::npos) {
      out.emplace_back(peer_ranks_[i], pending.substr(0, nl));
      pending.erase(0, nl + 1);
    }
  }
  return out;
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

void JournalReader::send_ready(const std::string& note) {
  if (note.find_first_of(" \n") != std::string::npos)
    throw std::invalid_argument("journal: a readiness note is one token");
  const std::string ready = note.empty() ? "ready\n" : "ready " + note + "\n";
  if (!conn_.write_all(ready.data(), ready.size()))
    throw std::runtime_error("journal: could not send model readiness to rank 0");
}

bool JournalReader::read_line(const std::function<bool()>& should_stop,
                              std::string* line) {
  size_t searched = 0;
  for (;;) {
    const size_t nl = pending_.find('\n', searched);
    if (nl != std::string::npos) {
      // A complete record is honored even when the stop flag already
      // fired — the stop RECORD itself arrives this way.
      line->assign(pending_, 0, nl);
      pending_.erase(0, nl + 1);
      return true;
    }
    // Pixel-bearing records can be tens of MiB. Re-scanning their full
    // prefix after every socket read makes admission quadratic and stalls
    // every rank before the scheduler can yield to active decodes.
    searched = pending_.size();
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

bool JournalReader::send_line(const std::string& line) {
  const std::string framed = line + "\n";
  return conn_.write_all(framed.data(), framed.size());
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
                       std::string* config_digest, bool* disk_adopt) {
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
  if (disk_adopt != nullptr) *disk_adopt = rec.disk_adopt;
  return true;
}

void run_journal_peer(Scheduler* sched, JournalReader* reader,
                      const std::function<bool()>& should_stop,
                      const std::function<void()>& on_rank0_death,
                      int watch_poll_ms,
                      const dgpp::sched::SchedulerObserver* oplog,
                      ThroughputLog* stats,
                      const std::function<void(const Scheduler::Meters&, int64_t)>& on_tick) {
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
    // The cold tier's commits (issue #26): the world's verdicts, before
    // the tick, as rank 0 applied them.
    if (!rec.disk_commits.empty()) sched->apply_disk_commits(rec.disk_commits);
    {
      TickScope scope(in_tick);
      if (sched->adaptive_prefill() != rec.prefill_budget.has_value())
        throw std::runtime_error("journal: adaptive prefill decision missing or unexpected");
      sched->tick(rec.prefill_budget);
    }
    ++ticks;
    // This rank's finished cold tier ops go back to rank 0 (the return
    // path); a write that fails means rank 0 is gone, which the next read
    // reports.
    if (sched->disk_enabled())
      for (const auto& c : sched->poll_disk_completions())
        if (!reader->send_line(encode_disk_status(c.op, c.ok)))
          DGPP_LOG_WARN("journal: could not report cold tier op {} to rank 0", c.op);
    if (stats || on_tick) {
      const Scheduler::Meters m = sched->meters();
      if (stats) stats->observe(m, nullptr);
      if (on_tick) on_tick(m, ticks);
    }
  }
}

}  // namespace dgpp::serve
